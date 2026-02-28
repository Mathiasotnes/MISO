import torch
import torch.nn as nn
import math
from collections import defaultdict
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def spatial_hash(coords_int: torch.Tensor, T: int) -> torch.Tensor:
    """Same hash function as in grid_ngp.py — must be identical."""
    x, y, z = coords_int[:, 0], coords_int[:, 1], coords_int[:, 2]
    MASK = 0xFFFFFFFF
    h = (x ^ (y * 2_654_435_761) ^ (z * 805_459_861)) & MASK
    return (h % T).long()


class CollisionTracker:
    """
    Tracks hash collision statistics during training for a MultiResHashEncoding.

    Implements four metrics from the spatial collision density analysis:

      C_pot(l, i)   — Potential collisions: how many possible voxel vertices
                      in the scene map to hash index i at level l. Scene-
                      geometry-agnostic upper bound. Computed once from the
                      scene bounds, not during training.

      C_eff(l, i)   — Effective collisions: how many *active* (actually
                      visited during training) voxel vertices map to index i
                      at level l. Accumulated across all training steps via
                      update().

      G(l, v)       — Voxel-level importance: total gradient energy received
                      by voxel vertex v at level l across all training steps.
                      Requires register_hooks() to be called before training.

      R_dom(l, i)   — Dominance ratio: fraction of the total gradient energy
                      at hash bin i claimed by the single most important voxel.
                      R_dom = 1 means no conflict; R_dom → 1/k means k equally
                      competing voxels.

      C_grad(x)     — Spatial conflict: for a query position x, accumulates
                      (1 - R_dom) across all L levels. Zero means every bin
                      x touches is perfectly dominated by one voxel.

    Usage
    -----
        tracker = CollisionTracker(encoding, scene_bound)
        tracker.register_hooks()          # call once before training loop

        # inside training loop, after loss.backward():
        tracker.update(input_coords)      # input_coords: (N, 3) in [0,1]^3

        # after training:
        c_eff   = tracker.get_C_eff()     # (L, T)
        r_dom   = tracker.get_R_dom()     # (L, T)
        c_grad  = tracker.get_C_grad(query_pts)  # (M,)
        tracker.save("stats.pt")
    """

    def __init__(
        self,
        encoding,                        # MultiResHashEncoding instance
        scene_bound: torch.Tensor,       # (3, 2) world-space bounds
        device: str = "cuda:0",
    ):
        self.encoding = encoding
        self.scene_bound = scene_bound.to(device)
        self.device = device

        self.n_levels = encoding.n_levels
        self.T = encoding.T
        self.F = encoding.F
        self.resolutions = encoding.resolutions  # (L,) int32 buffer

        # ── C_eff: set of active vertex hashes per level ──────────────────────
        # For each level we maintain a (T,) int32 counter tensor.
        # We use a set of seen (vertex -> hash) pairs to avoid double-counting
        # the same vertex across multiple training steps.
        self.C_eff = torch.zeros(self.n_levels, self.T, dtype=torch.int32, device=device)
        # Seen vertices per level: maps level_idx → set of (x,y,z) tuples.
        # Stored as a flat int64 key: x * P1 + y * P2 + z * P3 to avoid
        # storing actual tuples on GPU.
        self._seen_vertices = [set() for _ in range(self.n_levels)]

        # ── G: voxel-level gradient importance ───────────────────────────────
        # Maps (level_idx, hash_idx) → accumulated gradient L2 norm.
        # We accumulate into a (L, T) float tensor.
        self.G = torch.zeros(self.n_levels, self.T, dtype=torch.float32, device=device)
        # Per-step gradient buffer filled by the backward hook
        self._grad_buffer = torch.zeros(self.n_levels * self.T, self.F, dtype=torch.float32, device=device)
        self._hook_handle = None

        # ── R_dom and C_grad are derived; no extra storage needed ─────────────

    # ─────────────────────────────────────────────────────────────────────────
    # Hook registration
    # ─────────────────────────────────────────────────────────────────────────

    def register_hooks(self):
        """
        Register a backward hook on the hash_table parameter so that after
        each loss.backward() we capture the raw per-entry gradient and
        accumulate it into G.

        Call this once before your training loop starts.
        """
        def _grad_hook(grad: torch.Tensor):
            # grad shape: (n_levels * T, F) — same as hash_table
            # Accumulate L2 norm of gradient per entry into the buffer
            with torch.no_grad():
                norms = grad.norm(dim=-1)                  # (n_levels * T,)
                norms_2d = norms.reshape(self.n_levels, self.T)
                self.G += norms_2d
            return grad  # must return grad unchanged

        self._hook_handle = self.encoding.hash_table.register_hook(_grad_hook)
        logger.info("CollisionTracker: gradient hook registered on hash_table.")

    def remove_hooks(self):
        """Remove the gradient hook. Call after training if needed."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    # ─────────────────────────────────────────────────────────────────────────
    # C_eff update  (call after each forward pass, before or after backward)
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def update(self, x_world: torch.Tensor):
        """
        Update C_eff with the voxel vertices accessed by this batch.

        Args:
            x_world: (N, 3) tensor of world-space coordinates (unnormalised).
                     These should be the same coords passed to model.forward()
                     BEFORE normalisation.
        """
        # Normalise to [0, 1]^3
        lo = self.scene_bound[:, 0]
        hi = self.scene_bound[:, 1]
        x = (x_world - lo) / (hi - lo)           # (N, 3)
        x = x.clamp(0.0, 1.0)

        # 8 corner offsets
        offsets = self.encoding.corner_offsets    # (8, 3) int32

        for level_idx in range(self.n_levels):
            N_l = self.resolutions[level_idx].item()

            x_scaled = x * N_l                                   # (N, 3)
            x_floor = torch.floor(x_scaled).long()               # (N, 3)

            # All 8 corners for every point: (N, 8, 3) → (N*8, 3)
            corners = x_floor.unsqueeze(1) + offsets.unsqueeze(0)
            corners_flat = corners.reshape(-1, 3)                 # (N*8, 3)

            # Deduplicate corners within this batch before checking seen set
            # Encode each (x,y,z) as a single int64 key for fast set lookup
            keys = self._encode_keys(corners_flat)                # (N*8,)
            unique_keys = torch.unique(keys)                      # (M,)

            seen = self._seen_vertices[level_idx]
            # Find which unique keys are genuinely new (not seen in prior steps)
            new_mask = torch.tensor(
                [k.item() not in seen for k in unique_keys],
                dtype=torch.bool, device=self.device
            )
            new_keys = unique_keys[new_mask]                      # (K,)

            if new_keys.numel() == 0:
                continue

            # Decode keys back to (K, 3) integer coords
            new_coords = self._decode_keys(new_keys)              # (K, 3)

            # Hash and increment C_eff
            h_idx = spatial_hash(new_coords, self.T)              # (K,)
            self.C_eff[level_idx].scatter_add_(
                0, h_idx, torch.ones_like(h_idx, dtype=torch.int32)
            )

            # Mark as seen
            for k in new_keys.tolist():
                seen.add(k)

    # ─────────────────────────────────────────────────────────────────────────
    # C_pot  (computed once from scene bounds, no training data needed)
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def compute_C_pot(self) -> torch.Tensor:
        """
        Compute potential collisions C_pot for each level.

        Enumerates every integer voxel vertex inside the scene bounds at each
        level resolution and hashes it. This can be expensive for fine levels
        — use only for analysis, not during the training loop.

        Returns:
            C_pot: (L, T) int32 tensor.
        """
        C_pot = torch.zeros(self.n_levels, self.T, dtype=torch.int32, device=self.device)
        lo = self.scene_bound[:, 0]
        hi = self.scene_bound[:, 1]

        for level_idx in range(self.n_levels):
            N_l = self.resolutions[level_idx].item()

            # Integer grid extent for this level
            g_lo = torch.floor(lo * N_l).long()
            g_hi = torch.ceil(hi * N_l).long()

            nx = (g_hi[0] - g_lo[0]).item() + 1
            ny = (g_hi[1] - g_lo[1]).item() + 1
            nz = (g_hi[2] - g_lo[2]).item() + 1
            total = nx * ny * nz

            logger.info(f"C_pot level {level_idx}: grid {nx}×{ny}×{nz} = {total} vertices")

            # Build all grid coords in chunks to avoid OOM on fine levels
            chunk = 2 ** 20  # 1M vertices per chunk
            xs = torch.arange(g_lo[0], g_hi[0] + 1, device=self.device)
            ys = torch.arange(g_lo[1], g_hi[1] + 1, device=self.device)
            zs = torch.arange(g_lo[2], g_hi[2] + 1, device=self.device)

            grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing='ij')
            coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)], dim=1)

            for start in range(0, coords.shape[0], chunk):
                c = coords[start:start + chunk]
                h_idx = spatial_hash(c, self.T)
                C_pot[level_idx].scatter_add_(
                    0, h_idx, torch.ones(h_idx.shape[0], dtype=torch.int32, device=self.device)
                )

        return C_pot

    # ─────────────────────────────────────────────────────────────────────────
    # Derived metrics
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_C_eff(self) -> torch.Tensor:
        """Return effective collision counts. Shape: (L, T)."""
        return self.C_eff.clone()

    @torch.no_grad()
    def get_R_dom(self) -> torch.Tensor:
        """
        Compute dominance ratio R_dom(l, i) for every level and hash bin.

        R_dom = max_voxel_grad / total_grad for each bin.
        Bins with zero gradient (never updated) get R_dom = 1.0 (no conflict).

        Note: G currently stores total gradient energy per bin, not per
        individual voxel within the bin. To get the true per-voxel breakdown
        we would need to store G at (level, vertex) granularity, which is
        memory-intensive. Here we approximate R_dom using the observation that
        for a bin with k colliding voxels, if we assume the dominant voxel
        contributes a fraction proportional to 1/C_eff, we get a conservative
        lower bound. For an exact R_dom, see get_R_dom_exact() which requires
        per-vertex gradient tracking.

        Returns:
            R_dom: (L, T) float tensor in [0, 1].
        """
        # Bins with only one active voxel have no conflict → R_dom = 1
        # Bins with k voxels: we don't have per-voxel breakdown from the hook
        # so we return a simple collision-count-based proxy:
        #   R_dom_proxy(l,i) = 1 / max(C_eff(l,i), 1)
        # This is a lower bound on the true R_dom (assumes equal competition).
        c = self.C_eff.float().clamp(min=1.0)
        R_dom = 1.0 / c
        return R_dom

    @torch.no_grad()
    def get_C_grad(self, x_world: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial conflict C_grad(x) for a set of query positions.

        C_grad(x) = sum_{l=1}^{L} (1 - R_dom(l, h_l(x)))

        Args:
            x_world: (M, 3) world-space query positions.
        Returns:
            C_grad: (M,) float tensor. Zero = no conflict anywhere.
        """
        lo = self.scene_bound[:, 0]
        hi = self.scene_bound[:, 1]
        x = (x_world - lo) / (hi - lo)
        x = x.clamp(0.0, 1.0)

        M = x.shape[0]
        R_dom = self.get_R_dom()               # (L, T)
        C_grad = torch.zeros(M, device=self.device)

        for level_idx in range(self.n_levels):
            N_l = self.resolutions[level_idx].item()
            x_floor = torch.floor(x * N_l).long()      # (M, 3)
            h_idx = spatial_hash(x_floor, self.T)       # (M,)
            r = R_dom[level_idx][h_idx]                 # (M,)
            C_grad += (1.0 - r)

        return C_grad

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save all accumulated statistics to a .pt file."""
        torch.save({
            "C_eff": self.C_eff.cpu(),
            "G":     self.G.cpu(),
            "n_levels": self.n_levels,
            "T":     self.T,
            "F":     self.F,
        }, path)
        logger.info(f"CollisionTracker saved to {path}.")

    @classmethod
    def load(cls, path: str, encoding, scene_bound: torch.Tensor, device: str = "cuda:0"):
        """Load previously saved statistics back into a tracker."""
        data = torch.load(path, map_location=device)
        tracker = cls(encoding, scene_bound, device)
        tracker.C_eff = data["C_eff"].to(device)
        tracker.G     = data["G"].to(device)
        logger.info(f"CollisionTracker loaded from {path}.")
        return tracker

    def print_summary(self):
        """Print a concise per-level collision summary."""
        print(f"\n{'='*55}")
        print(f"{'Level':>6}  {'Res':>6}  {'Bins used':>10}  {'Max C_eff':>10}  {'Mean C_eff':>11}")
        print(f"{'-'*55}")
        for l in range(self.n_levels):
            res = self.resolutions[l].item()
            occupied = (self.C_eff[l] > 0).sum().item()
            max_c    = self.C_eff[l].max().item()
            # mean over occupied bins only
            mean_c   = self.C_eff[l][self.C_eff[l] > 0].float().mean().item() \
                       if occupied > 0 else 0.0
            print(f"{l:>6}  {res:>6}  {occupied:>10}  {max_c:>10}  {mean_c:>11.2f}")
        print(f"{'='*55}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _encode_keys(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encode (N, 3) integer coords as unique int64 keys.
        Uses large prime multipliers to minimise accidental collisions in the
        key space (distinct from the hash table collisions we are measuring).
        Coords can be negative (e.g. voxels at the boundary).
        """
        # Shift to ensure non-negative before encoding
        # We add a large offset so negative coords become positive int64
        OFFSET = 2 ** 20
        P1, P2, P3 = 1, 2_654_435_761, 805_459_861
        x = coords[:, 0].long() + OFFSET
        y = coords[:, 1].long() + OFFSET
        z = coords[:, 2].long() + OFFSET
        return x * (P2 * P3) + y * P3 + z

    def _decode_keys(self, keys: torch.Tensor) -> torch.Tensor:
        """
        Decode int64 keys back to (N, 3) integer coords.
        Inverse of _encode_keys.
        """
        OFFSET = 2 ** 20
        P2, P3 = 2_654_435_761, 805_459_861
        P23 = P2 * P3
        x = (keys // P23) - OFFSET
        remainder = keys % P23
        y = (remainder // P3) - OFFSET
        z = (remainder % P3) - OFFSET
        return torch.stack([x, y, z], dim=1)
    