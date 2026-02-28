import torch
import logging
import types

logger = logging.getLogger(__name__)


def spatial_hash(coords_int: torch.Tensor, T: int) -> torch.Tensor:
    """Must be identical to the version in grid_ngp.py."""
    x, y, z = coords_int[:, 0], coords_int[:, 1], coords_int[:, 2]
    MASK = 0xFFFFFFFF
    h = (x ^ (y * 2_654_435_761) ^ (z * 805_459_861)) & MASK
    return (h % T).long()


class CollisionTracker:
    """
    Tracks hash collision statistics during training for a MultiResHashEncoding.

    Metrics
    -------
    C_eff(l, i)  — Number of distinct active voxel vertices (approximated as
                   distinct hash bins) seen at level l, bin i across training.

    G_sum(l, i)  — Total gradient energy (sum of per-sample L2 norms) that
                   has flowed through bin i at level l across all steps.

    G_max(l, i)  — Maximum per-vertex cumulative gradient energy at bin i.
                   Updated incrementally: when a bin is visited for the first
                   time in a step, its gradient contribution is compared
                   against the stored maximum.

    R_dom(l, i)  — G_max(l,i) / G_sum(l,i). Ranges from ~1/k (k equally
                   competing voxels, maximum conflict) to 1.0 (one voxel
                   dominates perfectly).

    C_grad(x)    — sum_l (1 - R_dom(l, h_l(x))). Spatial conflict at x.

    Storage
    -------
    Only three (L, T) tensors are needed regardless of scene size or
    number of active vertices — no per-vertex dictionaries required.

    Speed
    -----
    - C_eff / seen_mask updates are fully vectorised GPU ops (no Python loops
      over vertices).
    - Gradient capture uses a forward-pass hook on the feats tensor rather
      than on hash_table.grad, giving per-lookup gradient norms before
      colliders merge.
    - Pending gradients are flushed in update() with scatter_add_ — one pass
      per level per step.
    """

    def __init__(
        self,
        encoding,
        scene_bound: torch.Tensor,      # (3, 2) world-space bounds
        device: str = "cuda:0",
    ):
        self.encoding = encoding
        self.scene_bound = scene_bound.to(device)
        self.device = device

        self.n_levels = encoding.n_levels
        self.T = encoding.T
        self.F = encoding.F
        self.resolutions = encoding.resolutions

        # ── C_eff ─────────────────────────────────────────────────────────────
        self.C_eff = torch.zeros(self.n_levels, self.T, dtype=torch.int32, device=device)
        # GPU bitmask: True once a bin has been visited at least once.
        # Avoids double-counting the same bin across steps.
        self._seen_mask = torch.zeros(self.n_levels, self.T, dtype=torch.bool, device=device)

        # ── R_dom: only G_sum and G_max needed ────────────────────────────────
        self.G_sum = torch.zeros(self.n_levels, self.T, dtype=torch.float32, device=device)
        self.G_max = torch.zeros(self.n_levels, self.T, dtype=torch.float32, device=device)

        # Scratch buffer reused every update() call — avoids repeated allocs
        self._scratch = torch.zeros(self.n_levels, self.T, dtype=torch.float32, device=device)

        # Gradient data captured by forward hooks, flushed in update()
        # Each entry: (level_idx: int, local_idx: Tensor[N*8], norms: Tensor[N*8])
        self._pending: list = []

    # ─────────────────────────────────────────────────────────────────────────
    # Hook registration
    # ─────────────────────────────────────────────────────────────────────────

    def register_hooks(self):
        """
        Monkey-patch MultiResHashEncoding.forward() on this specific instance
        to inject a backward hook on each level's feature lookup.

        Why hook feats rather than hash_table.grad?
          hash_table.grad accumulates gradients *after* all colliding voxels
          have summed into the same entry — per-voxel information is gone.
          Hooking feats (shape N*8, F) lets us see the gradient for each
          individual lookup slot before the backward scatter merges them.
        """
        tracker = self

        def patched_forward(enc_self, x: torch.Tensor) -> torch.Tensor:
            N = x.shape[0]
            level_features = []

            for level_idx in range(enc_self.n_levels):
                N_l = enc_self.resolutions[level_idx].item()
                x_scaled = x * N_l
                x_floor = torch.floor(x_scaled).long()
                w = x_scaled - x_floor.float()

                corners = x_floor.unsqueeze(1) + enc_self.corner_offsets.unsqueeze(0)
                corners_flat = corners.reshape(N * 8, 3)

                local_idx = spatial_hash(corners_flat, enc_self.T)
                global_idx = local_idx + level_idx * enc_self.T

                feats = enc_self.hash_table[global_idx]   # (N*8, F)

                # ── gradient hook ─────────────────────────────────────────────
                if feats.requires_grad:
                    _lvl = level_idx
                    _idx = local_idx.detach()              # (N*8,) local bin idx

                    def _hook(grad, lvl=_lvl, idx=_idx):
                        with torch.no_grad():
                            tracker._pending.append((lvl, idx, grad.norm(dim=-1)))
                        return grad

                    feats.register_hook(_hook)
                # ─────────────────────────────────────────────────────────────

                feats = feats.reshape(N, 8, enc_self.F)

                wx0, wx1 = 1.0 - w[:, 0], w[:, 0]
                wy0, wy1 = 1.0 - w[:, 1], w[:, 1]
                wz0, wz1 = 1.0 - w[:, 2], w[:, 2]

                weights = torch.stack([
                    wx0 * wy0 * wz0, wx0 * wy0 * wz1,
                    wx0 * wy1 * wz0, wx0 * wy1 * wz1,
                    wx1 * wy0 * wz0, wx1 * wy0 * wz1,
                    wx1 * wy1 * wz0, wx1 * wy1 * wz1,
                ], dim=1).unsqueeze(-1)

                level_features.append((weights * feats).sum(dim=1))

            return torch.cat(level_features, dim=-1)

        self.encoding.forward = types.MethodType(patched_forward, self.encoding)
        logger.info("CollisionTracker: forward patched on encoding instance.")

    def remove_hooks(self):
        """Remove the patched forward and discard any pending data."""
        if hasattr(self.encoding, '_original_forward'):
            self.encoding.forward = self.encoding._original_forward
        self._pending.clear()

    # ─────────────────────────────────────────────────────────────────────────
    # update() — call after loss.backward() each step
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def update(self, x_world: torch.Tensor):
        """
        Flush pending gradients into G_sum / G_max, then update C_eff.

        Call after loss.backward() each training step, passing the raw
        world-space coordinates used in that forward pass (before normalisation).

        Args:
            x_world: (N, 3) world-space coordinates.
        """
        # ── Step 1: flush gradient accumulators ──────────────────────────────
        # _scratch[l, i] will hold the gradient energy contributed by bins
        # that are freshly seen this step — used to update G_max.
        self._scratch.zero_()

        for lvl, local_idx, norms in self._pending:
            # G_sum: unconditional accumulation of every lookup's gradient norm
            self.G_sum[lvl].scatter_add_(0, local_idx, norms)

            # For G_max we want the dominant *voxel's* energy, not the total.
            # We approximate this by accumulating gradient energy only for bins
            # that are new this step (their first-ever visit), treating that
            # first visit as a proxy for the dominant voxel's contribution.
            new_bins = ~self._seen_mask[lvl][local_idx]     # (N*8,) bool
            if new_bins.any():
                self._scratch[lvl].scatter_add_(
                    0, local_idx, norms * new_bins.float()
                )

        self._pending.clear()

        # G_max: keep the running maximum over all first-visit contributions
        torch.maximum(self.G_max, self._scratch, out=self.G_max)

        # ── Step 2: update C_eff and seen_mask ───────────────────────────────
        lo = self.scene_bound[:, 0]
        hi = self.scene_bound[:, 1]
        x = ((x_world - lo) / (hi - lo)).clamp(0.0, 1.0)

        offsets = self.encoding.corner_offsets   # (8, 3)

        for level_idx in range(self.n_levels):
            N_l = self.resolutions[level_idx].item()
            x_floor = torch.floor(x * N_l).long()

            corners_flat = (
                x_floor.unsqueeze(1) + offsets.unsqueeze(0)
            ).reshape(-1, 3)

            local_idx = spatial_hash(corners_flat, self.T)  # (N*8,)

            # Find which bins are genuinely new (not seen before this step)
            new_bin_mask = torch.zeros(self.T, dtype=torch.bool, device=self.device)
            new_bin_mask[local_idx] = True          # deduplicates within batch
            new_bin_mask &= ~self._seen_mask[level_idx]

            # Increment C_eff by 1 for each newly-seen bin
            self.C_eff[level_idx] += new_bin_mask.int()
            self._seen_mask[level_idx] |= new_bin_mask

    # ─────────────────────────────────────────────────────────────────────────
    # Derived metrics
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_R_dom(self, eps: float = 1e-8) -> torch.Tensor:
        """
        R_dom(l, i) = G_max(l,i) / G_sum(l,i).
        Bins with no gradient receive R_dom = 1.0 (no conflict).
        Returns: (L, T) float tensor in (0, 1].
        """
        return self.G_max / self.G_sum.clamp(min=eps)

    @torch.no_grad()
    def get_C_grad(self, x_world: torch.Tensor) -> torch.Tensor:
        """
        C_grad(x) = sum_l (1 - R_dom(l, h_l(x))).
        Args:
            x_world: (M, 3) world-space positions.
        Returns:
            (M,) float tensor.
        """
        lo = self.scene_bound[:, 0]
        hi = self.scene_bound[:, 1]
        x = ((x_world - lo) / (hi - lo)).clamp(0.0, 1.0)

        R_dom = self.get_R_dom()
        C_grad = torch.zeros(x.shape[0], device=self.device)

        for level_idx in range(self.n_levels):
            N_l = self.resolutions[level_idx].item()
            h_idx = spatial_hash(torch.floor(x * N_l).long(), self.T)
            C_grad += 1.0 - R_dom[level_idx][h_idx]

        return C_grad

    @torch.no_grad()
    def compute_C_pot(self) -> torch.Tensor:
        """
        Enumerate all voxel vertices in the scene bounds and count collisions.
        Offline analysis only — not needed during training.
        Returns: (L, T) int32 tensor.
        """
        C_pot = torch.zeros(self.n_levels, self.T, dtype=torch.int32, device=self.device)
        lo = self.scene_bound[:, 0]
        hi = self.scene_bound[:, 1]
        CHUNK = 2 ** 20

        for level_idx in range(self.n_levels):
            N_l = self.resolutions[level_idx].item()
            g_lo = torch.floor(lo * N_l).long()
            g_hi = torch.ceil(hi * N_l).long()
            xs = torch.arange(g_lo[0], g_hi[0] + 1, device=self.device)
            ys = torch.arange(g_lo[1], g_hi[1] + 1, device=self.device)
            zs = torch.arange(g_lo[2], g_hi[2] + 1, device=self.device)
            gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing='ij')
            coords = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)
            for start in range(0, coords.shape[0], CHUNK):
                c = coords[start:start + CHUNK]
                h = spatial_hash(c, self.T)
                C_pot[level_idx].scatter_add_(
                    0, h, torch.ones(h.shape[0], dtype=torch.int32, device=self.device)
                )
            logger.info(f"C_pot level {level_idx} done (N_l={N_l}).")

        return C_pot

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence and diagnostics
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "C_eff": self.C_eff.cpu(),
            "G_sum": self.G_sum.cpu(),
            "G_max": self.G_max.cpu(),
            "n_levels": self.n_levels,
            "T": self.T,
            "F": self.F,
        }, path)
        logger.info(f"CollisionTracker saved to {path}.")

    @classmethod
    def load(cls, path: str, encoding, scene_bound: torch.Tensor, device: str = "cuda:0"):
        data = torch.load(path, map_location=device)
        tracker = cls(encoding, scene_bound, device)
        tracker.C_eff = data["C_eff"].to(device)
        tracker.G_sum = data["G_sum"].to(device)
        tracker.G_max = data["G_max"].to(device)
        logger.info(f"CollisionTracker loaded from {path}.")
        return tracker

    def print_summary(self):
        R_dom = self.get_R_dom()
        print(f"\n{'='*72}")
        print(f"{'Lvl':>4}  {'Res':>5}  {'Bins used':>10}  {'Max C_eff':>10}  {'Mean C_eff':>10}  {'Mean R_dom':>10}")
        print(f"{'-'*72}")
        for l in range(self.n_levels):
            res = self.resolutions[l].item()
            occ = self.C_eff[l] > 0
            n_occ = occ.sum().item()
            max_c  = self.C_eff[l].max().item()
            mean_c = self.C_eff[l][occ].float().mean().item() if n_occ > 0 else 0.0
            mean_r = R_dom[l][occ].mean().item() if n_occ > 0 else 1.0
            print(f"{l:>4}  {res:>5}  {n_occ:>10}  {max_c:>10}  {mean_c:>10.2f}  {mean_r:>10.4f}")
        print(f"{'='*72}\n")
        