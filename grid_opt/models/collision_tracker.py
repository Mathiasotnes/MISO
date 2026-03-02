import torch
import logging
import types

logger = logging.getLogger(__name__)


def spatial_hash(coords_int: torch.Tensor, T: int) -> torch.Tensor:
    x, y, z = coords_int[:, 0], coords_int[:, 1], coords_int[:, 2]
    MASK = 0xFFFFFFFF # Cast to uint32 range explicitly to mimic TCNN behavior
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
        
        # If finest resolution is above 2**20, the _seen_voxels tensor overflows and breaks down
        assert self.resolutions[-1].item() <= 2**20, "Finest resolution must be <= 2^20 to avoid overflow in seen_voxels."
        
        # ── C_eff Voxel Tracking ──────────────────────────────────────────────
        self.C_eff = torch.zeros(self.n_levels, self.T, dtype=torch.int32, device=device)
        
        self._seen_voxels = [
            torch.empty(0, dtype=torch.long, device=device) for _ in range(self.n_levels)
        ]
        self._seen_G = [
            torch.empty(0, dtype=torch.float32, device=device) for _ in range(self.n_levels)
        ]

        # ── R_dom: Only G_sum is needed as a standalone tensor ────────────────
        self.G_sum = torch.zeros(self.n_levels, self.T, dtype=torch.float32, device=device)
        
        # ── C_pot & hook buffer ───────────────────────────────────────────────
        self._pending: list = []
        self.C_pot = None

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
        try:
            del self.encoding.forward
            logger.info("CollisionTracker: hooks removed, forward restored.")
        except AttributeError:
            pass            
        self._pending.clear()

    # ─────────────────────────────────────────────────────────────────────────
    # update() — call after loss.backward() each step
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def update(self, x_world: torch.Tensor):
        lo = self.scene_bound[:, 0]
        hi = self.scene_bound[:, 1]
        x_norm = ((x_world - lo) / (hi - lo)).clamp(0.0, 1.0)
        N = x_norm.shape[0]

        # Pre-group pending gradients by level to avoid looping multiple times
        pending_by_lvl = {l: ([], []) for l in range(self.n_levels)}
        for p_lvl, p_idx, p_norms in self._pending:
            pending_by_lvl[p_lvl][0].append(p_idx)
            pending_by_lvl[p_lvl][1].append(p_norms)

        for level_idx in range(self.n_levels):
            # 1. Standardize the batch gradients for this level
            if not pending_by_lvl[level_idx][0]:
                continue
                
            local_idx = torch.cat(pending_by_lvl[level_idx][0])
            norms = torch.cat(pending_by_lvl[level_idx][1])
            
            # G_sum unconditionally accumulates the bin's total energy
            self.G_sum[level_idx].scatter_add_(0, local_idx, norms)

            # 2. Calculate coordinates and pack exactly as before
            N_l = self.resolutions[level_idx].item()
            x_scaled = x_norm * N_l
            x_floor = torch.floor(x_scaled).long()
            corners_flat = (x_floor.unsqueeze(1) + self.encoding.corner_offsets.unsqueeze(0)).reshape(N * 8, 3)

            packed_voxels = (
                (corners_flat[:, 0] << 42) | 
                (corners_flat[:, 1] << 21) | 
                 corners_flat[:, 2]
            )

            # 3. Sum the gradient energy per unique voxel IN THIS BATCH
            unique_packed, inverse_indices = torch.unique(packed_voxels, return_inverse=True)
            batch_G = torch.zeros_like(unique_packed, dtype=torch.float32)
            batch_G.scatter_add_(0, inverse_indices, norms)

            # 4. Use searchsorted to find where these voxels belong in our global history
            history_voxels = self._seen_voxels[level_idx]
            history_G = self._seen_G[level_idx]
            
            idx = torch.searchsorted(history_voxels, unique_packed)
            
            # Create a mask of which voxels we have genuinely seen before
            is_present = (idx < len(history_voxels)) & \
                         (history_voxels[idx.clamp(max=len(history_voxels)-1)] == unique_packed)

            # --- Update EXISTING voxels ---
            if is_present.any():
                existing_idx = idx[is_present]
                history_G.scatter_add_(0, existing_idx, batch_G[is_present])

            # --- Add NEW voxels ---
            new_packed = unique_packed[~is_present]
            new_G = batch_G[~is_present]

            if new_packed.numel() > 0:
                history_voxels = torch.cat([history_voxels, new_packed])
                history_G = torch.cat([history_G, new_G])
                
                # Re-sort to maintain searchsorted integrity
                sorted_idx = torch.argsort(history_voxels)
                self._seen_voxels[level_idx] = history_voxels[sorted_idx]
                self._seen_G[level_idx] = history_G[sorted_idx]
                
                # Unpack and hash ONLY the new voxels to update C_eff
                new_x = new_packed >> 42
                new_y = (new_packed >> 21) & 0x1FFFFF
                new_z = new_packed & 0x1FFFFF
                new_corners_3d = torch.stack([new_x, new_y, new_z], dim=-1)
                new_local_idx = spatial_hash(new_corners_3d, self.T)
                
                self.C_eff[level_idx].scatter_add_(
                    0, new_local_idx, torch.ones_like(new_local_idx, dtype=torch.int32)
                )

        self._pending.clear()

    # ─────────────────────────────────────────────────────────────────────────
    # Derived metrics
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_R_dom(self, eps: float = 1e-8) -> torch.Tensor:
        """
        R_dom(l, i) = G_max(l,i) / G_sum(l,i).
        Calculates the true maximum voxel gradient.
        """
        G_max = torch.zeros_like(self.G_sum)
        
        for l in range(self.n_levels):
            if self._seen_voxels[l].numel() == 0:
                continue
                
            # 1. Unpack our global voxel history to 3D
            packed = self._seen_voxels[l]
            x = packed >> 42
            y = (packed >> 21) & 0x1FFFFF
            z = packed & 0x1FFFFF
            corners_3d = torch.stack([x, y, z], dim=-1)
            
            # 2. Hash them to find which bin each voxel maps to
            bins = spatial_hash(corners_3d, self.T)
            
            # 3. Find the maximum gradient score (G_max) inside each bin!
            # scatter_reduce with 'amax' perfectly extracts the single dominant voxel's score.
            G_max[l].scatter_reduce_(
                0, bins, self._seen_G[l], reduce="amax", include_self=False
            )

        R_dom = G_max / self.G_sum.clamp(min=eps)
        R_dom[self.G_sum == 0] = 1.0  # Untouched bins have no conflict
        return R_dom

    @torch.no_grad()
    def get_C_grad(self, x_world: torch.Tensor) -> torch.Tensor:
        """
        C_grad(x) = sum_l (1 - R_dom(l, h_l(x))).
        Uses exact 8-corner trilinear weighting to match the model's forward pass.
        
        Args:
            x_world: (M, 3) world-space positions.
        Returns:
            (M,) float tensor.
        """
        lo = self.scene_bound[:, 0]
        hi = self.scene_bound[:, 1]
        x_norm = ((x_world - lo) / (hi - lo)).clamp(0.0, 1.0)
        M = x_norm.shape[0]

        R_dom = self.get_R_dom()
        C_grad = torch.zeros(M, device=self.device)

        for level_idx in range(self.n_levels):
            N_l = self.resolutions[level_idx].item()
            
            x_scaled = x_norm * N_l
            x_floor = torch.floor(x_scaled).long()
            w = x_scaled - x_floor.float()

            corners = x_floor.unsqueeze(1) + self.encoding.corner_offsets.unsqueeze(0)
            corners_flat = corners.reshape(M * 8, 3)

            # 1. Hash all 8 corners
            h_idx = spatial_hash(corners_flat, self.T)  # (M*8,)

            # 2. Look up R_dom for all 8 corners and reshape back to (M, 8)
            r_dom_corners = R_dom[level_idx][h_idx].reshape(M, 8)
            
            # 3. Calculate conflict (1 - R_dom) for each corner
            conflict_corners = 1.0 - r_dom_corners

            # 4. Calculate trilinear weights EXACTLY as in the forward pass
            wx0, wx1 = 1.0 - w[:, 0], w[:, 0]
            wy0, wy1 = 1.0 - w[:, 1], w[:, 1]
            wz0, wz1 = 1.0 - w[:, 2], w[:, 2]

            weights = torch.stack([
                wx0 * wy0 * wz0, wx0 * wy0 * wz1,
                wx0 * wy1 * wz0, wx0 * wy1 * wz1,
                wx1 * wy0 * wz0, wx1 * wy0 * wz1,
                wx1 * wy1 * wz0, wx1 * wy1 * wz1,
            ], dim=1) # (M, 8)

            # 5. Blend the conflict using the exact interpolation weights
            C_grad += (weights * conflict_corners).sum(dim=1)

        return C_grad

    @torch.no_grad()
    def compute_C_pot(self) -> torch.Tensor:
        """
        Calculates the theoretical upper bound of collisions for the scene.
        """
        self.C_pot = torch.zeros(self.n_levels, self.T, dtype=torch.long, device=self.device)
        CHUNK = 2 ** 24

        for level_idx in range(self.n_levels):
            N_l = self.resolutions[level_idx].item()
            
            # Since inputs are normalized to [0, 1] before scaling by N_l, 
            # integer vertices always range exactly from 0 to N_l inclusive.
            total_vertices = (N_l + 1) ** 3

            # If vertices < 134 million, compute exact map
            if total_vertices <= 2 ** 27:
                xs = torch.arange(0, N_l + 1, device=self.device)
                ys = torch.arange(0, N_l + 1, device=self.device)
                zs = torch.arange(0, N_l + 1, device=self.device)
                gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing='ij')
                coords = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)
                
                for start in range(0, coords.shape[0], CHUNK):
                    c = coords[start:start + CHUNK]
                    h = spatial_hash(c, self.T)
                    self.C_pot[level_idx].scatter_add_(
                        0, h, torch.ones(h.shape[0], dtype=torch.long, device=self.device)
                    )
                logger.info(f"C_pot layer {level_idx + 1} done exactly (Vertices={total_vertices:,}).")
            
            # If vertices are massive, use uniform distribution expectation
            else:
                expected_collisions = total_vertices // self.T
                self.C_pot[level_idx] = expected_collisions
                logger.info(f"C_pot layer {level_idx + 1} approximated (Vertices={total_vertices:,}).")

        return self.C_pot

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
        """Prints a detailed, column-separated numerical comparison of the tracking metrics."""
        if self.C_pot is None:
            logger.info("Computing C_pot...")
            self.compute_C_pot()

        R_dom = self.get_R_dom()
        
        table_width = 121
        print(f"\n{'='*table_width}")
        print(
            f"{'Lvl':>3} | {'Res':>6} | {'Util %':>8} | {'Occ (Bins)':>10} | {'C_pot avg':>10} | "
            f"{'C_eff min':>9} | {'C_eff max':>9} | {'C_eff avg':>9} | "
            f"{'R_dom min':>9} | {'R_dom max':>9} | {'R_dom avg':>9}"
        )
        print(f"{'-'*table_width}")
        
        def fmt_large(x):
            if x >= 1e9: return f"{x/1e9:.1f}B"
            if x >= 1e6: return f"{x/1e6:.1f}M"
            if x >= 1e3: return f"{x/1e3:.1f}K"
            return f"{int(x)}"

        for l in range(self.n_levels):
            res = self.resolutions[l].item()
            
            # --- Grid Utilization ---
            total_potential_voxels = (res + 1) ** 3
            total_active_voxels = self.C_eff[l].sum().item()
            util_pct = (total_active_voxels / max(total_potential_voxels, 1)) * 100
            
            # --- C_pot stats ---
            cp_avg = self.C_pot[l].float().mean().item()
            str_cp_avg = fmt_large(cp_avg)
            
            # --- Occupancy ---
            occ = self.C_eff[l] > 0
            num_occ = occ.sum().item()
            
            # --- C_eff stats (over OCCUPIED bins only) ---
            ce = self.C_eff[l][occ].float()
            if num_occ > 0:
                ce_min = int(ce.min().item())
                ce_max = int(ce.max().item())
                ce_avg = ce.mean().item()
            else:
                ce_min, ce_max, ce_avg = 0, 0, 0.0
                
            # --- R_dom stats (over OCCUPIED bins only) ---
            rd = R_dom[l][occ]
            if num_occ > 0:
                rd_min = rd.min().item()
                rd_max = rd.max().item()
                rd_avg = rd.mean().item()
            else:
                rd_min, rd_max, rd_avg = 1.0, 1.0, 1.0
                
            # Print the row (l+1 for 1-16 numbering, 7.2f for percentage)
            print(
                f"{l+1:>3} | {res:>6} | {util_pct:>7.2f}% | {num_occ:>10} | {str_cp_avg:>10} | "
                f"{ce_min:>9} | {ce_max:>9} | {ce_avg:>9.2f} | "
                f"{rd_min:>9.4f} | {rd_max:>9.4f} | {rd_avg:>9.4f}"
            )
            
        print(f"{'='*table_width}\n")
        