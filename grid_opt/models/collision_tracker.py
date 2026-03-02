import torch
import logging
import types

logger = logging.getLogger(__name__)


#########################################################
# Utilities
#########################################################

_COORD_BITS = 21  # each x/y/z must fit in 21 bits → max resolution 2^21
_COORD_MASK = (1 << _COORD_BITS) - 1  # 0x1FFFFF

def _pack(coords):   # (N, 3) int64 → (N,) int64
    return (coords[:, 0] << (2 * _COORD_BITS)) | \
           (coords[:, 1] << _COORD_BITS) | \
            coords[:, 2]

def _unpack(packed):  # (N,) int64 → (N, 3) int64
    x = packed >> (2 * _COORD_BITS)
    y = (packed >> _COORD_BITS) & _COORD_MASK
    z = packed & _COORD_MASK
    return torch.stack([x, y, z], dim=-1)

def spatial_hash(coords_int: torch.Tensor, T: int) -> torch.Tensor:
    x, y, z = coords_int[:, 0], coords_int[:, 1], coords_int[:, 2]
    MASK = 0xFFFFFFFF # Cast to uint32 range explicitly to mimic TCNN behavior
    h = (x ^ (y * 2_654_435_761) ^ (z * 805_459_861)) & MASK
    return (h % T).long()


#########################################################
# Collision Tracker
#########################################################

class CollisionTracker:
    def __init__(
        self,
        encoding,
        scene_bound: torch.Tensor, # (3, 2) world-space bounds
        device: str = "cuda:0",
    ):
        self.encoding = encoding
        self.scene_bound = scene_bound.to(device)
        self.device = device

        self.n_levels = encoding.n_levels
        self.T = encoding.T
        self.F = encoding.F
        self.resolutions = encoding.resolutions
        
        assert self.resolutions[-1].item() <= 2**20, "Finest resolution must be <= 2^20 to avoid overflow in _voxels buffer!"
        
        # Buffers
        self.C_pot      = None
        self.C_eff      = torch.zeros(self.n_levels, self.T, dtype=torch.int32, device=device)
        self._voxels    = [torch.empty(0, dtype=torch.long, device=device) for _ in range(self.n_levels)]
        self._G         = [torch.empty(0, dtype=torch.float32, device=device) for _ in range(self.n_levels)]
        self._pending: list = []
        

    # ─────────────────────────────────────────────────────────────────────────
    # Hook registration
    # ─────────────────────────────────────────────────────────────────────────

    def register_hooks(self):
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
                packed_voxels = _pack(corners_flat)

                feats = enc_self.hash_table[global_idx] # (N*8, F)

                # ── gradient hook ─────────────────────────────────────────────
                if feats.requires_grad:
                    _lvl = level_idx
                    _idx = local_idx.detach()
                    _packed = packed_voxels.detach()

                    def _hook(grad, lvl=_lvl, idx=_idx, pck=_packed):
                        with torch.no_grad():
                            tracker._pending.append((lvl, idx, pck, grad.norm(dim=-1)))
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
    def update(self):
        # Pre-group pending gradients by level
        pending_by_lvl = {l: ([], [], []) for l in range(self.n_levels)}
        for p_lvl, p_idx, p_packed, p_norms in self._pending:
            pending_by_lvl[p_lvl][0].append(p_idx)
            pending_by_lvl[p_lvl][1].append(p_packed)
            pending_by_lvl[p_lvl][2].append(p_norms)

        for level_idx in range(self.n_levels):
            if not pending_by_lvl[level_idx][0]:
                continue

            local_idx     = torch.cat(pending_by_lvl[level_idx][0])
            packed_voxels = torch.cat(pending_by_lvl[level_idx][1])
            norms         = torch.cat(pending_by_lvl[level_idx][2])

            # 1. Sum gradient energy per unique voxel in this batch
            unique_packed, inverse_indices = torch.unique(packed_voxels, return_inverse=True)
            batch_G = torch.zeros(unique_packed.shape[0], dtype=torch.float32, device=self.device)
            batch_G.scatter_add_(0, inverse_indices, norms)

            # 2. Merge into global voxel history via searchsorted
            history_voxels = self._voxels[level_idx]
            history_G      = self._G[level_idx]

            idx = torch.searchsorted(history_voxels, unique_packed)

            if len(history_voxels) == 0:
                is_present = torch.zeros(unique_packed.shape[0], dtype=torch.bool, device=self.device)
            else:
                clamped = idx.clamp(max=len(history_voxels) - 1)
                is_present = (idx < len(history_voxels)) & (history_voxels[clamped] == unique_packed)

            # Update existing voxels
            if is_present.any():
                history_G.scatter_add_(0, idx[is_present], batch_G[is_present])

            # Insert new voxels
            new_packed = unique_packed[~is_present]
            new_G      = batch_G[~is_present]

            if new_packed.numel() > 0:
                history_voxels = torch.cat([history_voxels, new_packed])
                history_G      = torch.cat([history_G, new_G])

                sorted_order = torch.argsort(history_voxels)
                self._voxels[level_idx] = history_voxels[sorted_order]
                self._G[level_idx]      = history_G[sorted_order]

                # Update C_eff only for genuinely new voxels
                new_coords   = _unpack(new_packed)
                new_bins     = spatial_hash(new_coords, self.T)
                self.C_eff[level_idx].scatter_add_(
                    0, new_bins, torch.ones_like(new_bins, dtype=torch.int32)
                )

        self._pending.clear()

    # ─────────────────────────────────────────────────────────────────────────
    # Derived metrics
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_R_dom(self, eps: float = 1e-8) -> torch.Tensor:
        G_sum = torch.zeros(self.n_levels, self.T, dtype=torch.float32, device=self.device)
        G_max = torch.zeros(self.n_levels, self.T, dtype=torch.float32, device=self.device)

        for l in range(self.n_levels):
            if self._voxels[l].numel() == 0:
                continue

            bins = spatial_hash(_unpack(self._voxels[l]), self.T)

            G_sum[l].scatter_add_(0, bins, self._G[l])
            G_max[l].scatter_reduce_(0, bins, self._G[l], reduce="amax", include_self=False)

        R_dom = G_max / G_sum.clamp(min=eps)
        R_dom[G_sum == 0] = 1.0
        return R_dom

    @torch.no_grad()
    def get_C_grad(self, x_world: torch.Tensor) -> torch.Tensor:
        lo = self.scene_bound[:, 0]
        hi = self.scene_bound[:, 1]
        x_norm = ((x_world - lo) / (hi - lo)).clamp(0.0, 1.0)
        M = x_norm.shape[0]

        R_dom  = self.get_R_dom()
        C_grad = torch.zeros(M, device=self.device)

        for level_idx in range(self.n_levels):
            N_l = self.resolutions[level_idx].item()

            x_scaled = x_norm * N_l
            x_floor  = torch.floor(x_scaled).long()
            w        = x_scaled - x_floor.float()

            corners      = x_floor.unsqueeze(1) + self.encoding.corner_offsets.unsqueeze(0)
            corners_flat = corners.reshape(M * 8, 3)

            h_idx            = spatial_hash(corners_flat, self.T)
            r_dom_corners    = R_dom[level_idx][h_idx].reshape(M, 8)
            conflict_corners = 1.0 - r_dom_corners

            wx0, wx1 = 1.0 - w[:, 0], w[:, 0]
            wy0, wy1 = 1.0 - w[:, 1], w[:, 1]
            wz0, wz1 = 1.0 - w[:, 2], w[:, 2]

            weights = torch.stack([
                wx0 * wy0 * wz0, wx0 * wy0 * wz1,
                wx0 * wy1 * wz0, wx0 * wy1 * wz1,
                wx1 * wy0 * wz0, wx1 * wy0 * wz1,
                wx1 * wy1 * wz0, wx1 * wy1 * wz1,
            ], dim=1)

            C_grad += (weights * conflict_corners).sum(dim=1)

        return C_grad

    @torch.no_grad()
    def compute_C_pot(self) -> torch.Tensor:
        self.C_pot = torch.zeros(self.n_levels, self.T, dtype=torch.long, device=self.device)
        CHUNK = 2 ** 24

        for level_idx in range(self.n_levels):
            N_l             = self.resolutions[level_idx].item()
            total_vertices  = (N_l + 1) ** 3

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
            else:
                self.C_pot[level_idx] = total_vertices // self.T
                logger.info(f"C_pot layer {level_idx + 1} approximated (Vertices={total_vertices:,}).")

        return self.C_pot

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence and diagnostics
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "C_eff":    self.C_eff.cpu(),
            "_voxels":  [v.cpu() for v in self._voxels],
            "_G":       [g.cpu() for g in self._G],
            "n_levels": self.n_levels,
            "T":        self.T,
            "F":        self.F,
        }, path)
        logger.info(f"CollisionTracker saved to {path}.")

    @classmethod
    def load(cls, path: str, encoding, scene_bound: torch.Tensor, device: str = "cuda:0"):
        data    = torch.load(path, map_location=device)
        tracker = cls(encoding, scene_bound, device)
        tracker.C_eff   = data["C_eff"].to(device)
        tracker._voxels = [v.to(device) for v in data["_voxels"]]
        tracker._G      = [g.to(device) for g in data["_G"]]
        logger.info(f"CollisionTracker loaded from {path}.")
        return tracker

    def print_summary(self):
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

            total_potential_voxels = (res + 1) ** 3
            total_active_voxels    = self.C_eff[l].sum().item()
            util_pct               = (total_active_voxels / max(total_potential_voxels, 1)) * 100

            cp_avg     = self.C_pot[l].float().mean().item()
            str_cp_avg = fmt_large(cp_avg)

            occ     = self.C_eff[l] > 0
            num_occ = occ.sum().item()

            ce = self.C_eff[l][occ].float()
            if num_occ > 0:
                ce_min = int(ce.min().item())
                ce_max = int(ce.max().item())
                ce_avg = ce.mean().item()
            else:
                ce_min, ce_max, ce_avg = 0, 0, 0.0

            rd = R_dom[l][occ]
            if num_occ > 0:
                rd_min = rd.min().item()
                rd_max = rd.max().item()
                rd_avg = rd.mean().item()
            else:
                rd_min, rd_max, rd_avg = 1.0, 1.0, 1.0

            print(
                f"{l+1:>3} | {res:>6} | {util_pct:>7.2f}% | {num_occ:>10} | {str_cp_avg:>10} | "
                f"{ce_min:>9} | {ce_max:>9} | {ce_avg:>9.2f} | "
                f"{rd_min:>9.4f} | {rd_max:>9.4f} | {rd_avg:>9.4f}"
            )

        print(f"{'='*table_width}\n")
        