import math
import torch
import torch.nn as nn
from .base_net import BaseNet
from .modules import MLPNet
from .grid_modules import *
import grid_opt.utils.utils_geometry as utils_geometry
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class NeuralPointsHash(BaseNet):
    """
    A lightweight implementation of neural point-based map inspired by PIN-SLAM:
    Paper: https://arxiv.org/abs/2401.09101
    Code:  https://github.com/PRBonn/PIN_SLAM/tree/main
    This is identical as the neural point implemented in neural_points.py except that it uses
    
    """
    def __init__(self,
        cfg: dict, 
        device = 'cuda:0',
        dtype = torch.float32,
    ):
        super(NeuralPointsHash, self).__init__(cfg, device, dtype)    
        self.device = device
        self.dtype = dtype
        self.init_hash_grid(cfg)
        self.init_neural_params(cfg)
        self.init_decoder(cfg)
        self.init_poses(cfg)
        self.print_trainable_params()
    
    def init_hash_grid(self, cfg):
        """ Initializes the hash-grid used for indexing parameters. """
        
        # TODO: Move this elsewhere
        # Config
        self.T = 2**19 # Hash table size (number of buckets)
        self.max_probe = 16
        self.max_points = 200_000
        self.cell_size = 0.1
        self.fdim = 4
        self.init_threshold = 0.3 # SDF threshold for initializing neural points (i.e., activating voxels)
        self.num_levels = 1 # To be compatible with trainer
        
        assert self.bound.shape == (3, 2), f"Invalid bound shape {self.bound.shape}!"
        
        self.grid_dims = torch.ceil((self.bound[:,1] - self.bound[:,0]) / self.cell_size).to(torch.long)  # (3,)
        
        # Hash table
        self.register_buffer("table_keys", torch.full((self.T, 3), -1, device=self.device, dtype=torch.int64))
        self.register_buffer("table_vals", torch.full((self.T,), -1, device=self.device, dtype=torch.int32))

        # Active points storage
        self.register_buffer("next_free", torch.zeros((), device=self.device, dtype=torch.int32))
    
    def init_neural_params(self, cfg):
        """ Initializes neural point parameters. """
        
        """
        Neural point structure:
            Variables:
                - Position
                - Feature vector
                - Orientation ?
                - Creation time ?
                - Last update time ?
                - Stability ?
        """
        # Non-trainable position buffer
        self.register_buffer("points", torch.zeros((self.max_points, 3), device=self.device, dtype=self.dtype))

        # Trainable feature buffer
        self.features = nn.Parameter(torch.zeros((self.max_points, self.fdim), device=self.device, dtype=self.dtype))
        
    def init_decoder(self, cfg):
        self.decoder_hidden_dim = cfg['decoder']['hidden_dim']
        self.decoder_hidden_layers = cfg['decoder']['hidden_layers']
        self.decoder_out_dim = cfg['decoder']['out_dim']
        self.pos_invariant = cfg['decoder']['pos_invariant']
        self.decoder_fixed = cfg['decoder']['fix']
        self.decoder_type = cfg['decoder']['type']
        input_dim = self.fdim + 3  # feature + position
        if not self.pos_invariant:
            input_dim += self.d
        
        if self.decoder_type == 'mlp':
            logger.debug(f"Using MLP decoder.")
            self.decoder = MLPNet(
                input_dim=input_dim,
                output_dim=self.decoder_out_dim,
                hidden_dim=self.decoder_hidden_dim,
                hidden_layers=self.decoder_hidden_layers,
                bias=True,
                pretrained_path=cfg['decoder']['pretrained_model'],
                no_optimize=self.decoder_fixed
            )
        elif self.decoder_type == 'none':
            logger.info("Not using decoder.")
            self.decoder = None
        else:
            raise ValueError(f"Unknown decoder type: {self.decoder_type}")
        logger.debug(f"Initialized docoder:\n {self.decoder}")

    def init_poses(self, cfg):
        """Initialize pose correction terms.
        The pose corrections can be optimized jointly with the feature grid, i.e., bundle adjustment.
        As an example, see PosedSdfLoss3D.
        """
        self.num_poses = cfg['pose']['num_poses']
        self.optimize_pose = cfg['pose']['optimize']
        self.rotation_corrections = torch.nn.Parameter(
            torch.zeros(self.num_poses, 3).float().to(self.device),
            requires_grad=self.optimize_pose
        )
        self.translation_corrections = torch.nn.Parameter(
            torch.zeros(self.num_poses, 3, 1).float().to(self.device),
            requires_grad=self.optimize_pose
        )
        self.pose_estimates_known = [False] * self.num_poses
        self.register_buffer('Rwk', utils_geometry.identity_rotations(self.num_poses).to(self.device))
        self.register_buffer('twk', torch.zeros(size=(self.num_poses, 3, 1), device=self.device))
        self.locked_pose_indices = set()
        self._pose_key_to_id = dict()
        logger.info(f"Initialized {self.num_poses} pose variables (optimize={self.optimize_pose}).")
        
    def hash_function(self, g: torch.Tensor) -> torch.Tensor:
        """ Hash function mapping voxel coordinates to hash table indices.
        This uses the hash function presented in InstantNGP: https://arxiv.org/abs/2201.05989.
        """
        g = g.to(torch.int64)
        p1, p2, p3 = 1, 2_654_435_761, 805_459_861
        h = (g[..., 0] * p1) ^ (g[..., 1] * p2) ^ (g[..., 2] * p3)
        return h % self.T
    
    def lookup_probing(self, g: torch.Tensor) -> torch.Tensor:
        """ Lookup voxel coordinates g in the hash table with linear probing. """
        g = g.to(torch.int64)
        N = g.shape[0]
        device = g.device

        out = torch.full((N,), -1, device=device, dtype=torch.int32)
        alive = torch.ones((N,), device=device, dtype=torch.bool)  # still searching

        h0 = self.hash_function(g).to(torch.int64)
        empty_key = torch.tensor([-1, -1, -1], device=device, dtype=torch.int64)

        for p in range(self.max_probe):
            if not alive.any():
                break
            h = (h0 + p) % self.T

            tk = self.table_keys[h]
            tv = self.table_vals[h]

            hit = (tk == g).all(dim=1)
            empty = (tk == empty_key).all(dim=1)

            write = alive & hit
            out = torch.where(write, tv, out)

            # stop searching after hit or empty
            alive = alive & (~hit) & (~empty)

        return out

    def insert_probing(self, g: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
        """
        Insert mapping g[i] -> vals[i] with linear probing.

        g: (U,3) int64 voxel coords
        vals: (U,) int32 point indices

        Returns: (U,) bool success per insert (False if probe budget exceeded).
        """
        assert g.ndim == 2 and g.shape[1] == 3
        assert vals.ndim == 1 and vals.shape[0] == g.shape[0]
        g = g.to(torch.int64)
        vals = vals.to(torch.int32)

        # Work on CPU copies to avoid concurrent write races
        g_cpu = g.detach().to("cpu")
        vals_cpu = vals.detach().to("cpu")

        table_keys = self.table_keys.detach().to("cpu").clone()
        table_vals = self.table_vals.detach().to("cpu").clone()

        T = int(self.T)
        P = int(self.max_probe)

        empty_key_cpu = torch.tensor([-1, -1, -1], device="cpu", dtype=torch.int64)

        # Use your hash_function on CPU tensors (same code path)
        h0 = self.hash_function(g_cpu).to(torch.int64)  # (U,)

        success = torch.zeros((g_cpu.shape[0],), dtype=torch.bool, device="cpu")

        for i in range(g_cpu.shape[0]):
            key = g_cpu[i]            # (3,)
            v = int(vals_cpu[i].item())
            start = int(h0[i].item())

            placed = False
            for p in range(P):
                h = (start + p) % T
                tk = table_keys[h]

                if (tk == empty_key_cpu).all() or (tk == key).all():
                    table_keys[h] = key
                    table_vals[h] = v
                    placed = True
                    break

            success[i] = placed

        # Copy updated tables back to GPU
        self.table_keys.copy_(table_keys.to(self.device))
        self.table_vals.copy_(table_vals.to(self.device))

        return success.to(g.device)
        
    def query_neighbors(self, x: torch.Tensor, K: int, Nn: int = 3) -> torch.Tensor:
        """ PIN-SLAM neighbor query:
        - collect candidates in an Nn^3 voxel cube around the query voxel
        - keep active points only
        - take K nearest by Euclidean distance

        Args:
            x: (N,3) world coords
            K: number of neighbors
            Nn: neighborhood cube side length (odd recommended)

        Returns:
            (N,K) long tensor with neighbor indices in [0, num_cells-1], or -1 if missing
        """
        assert x.ndim == 2 and x.shape[1] == 3, f"Invalid input shape {x.shape}"
        assert Nn >= 1, "Nn must be >= 1"
        if Nn % 2 == 0:
            # works, but "centered" is less clean; PIN-SLAM describes centered cube
            logger.warning(f"Nn={Nn} is even; consider using an odd Nn for a centered neighborhood.")

        N = x.shape[0]
        device = x.device

        g = self.world_to_vox(x)  # (N,3)

        r = Nn // 2
        o = torch.arange(-r, r + 1, device=device, dtype=torch.long)
        ox, oy, oz = torch.meshgrid(o, o, o, indexing="ij")
        offsets = torch.stack([ox, oy, oz], dim=-1).reshape(-1, 3)  # (M,3)
        M = offsets.shape[0]

        cand = g[:, None, :] + offsets[None, :, :]  # (N,M,3)

        inb = (
            (cand[..., 0] >= 0) & (cand[..., 0] < self.grid_dims[0]) &
            (cand[..., 1] >= 0) & (cand[..., 1] < self.grid_dims[1]) &
            (cand[..., 2] >= 0) & (cand[..., 2] < self.grid_dims[2])
        )

        cand_flat = cand.reshape(-1, 3)
        idx_flat = self.lookup_probing(cand_flat)  # (N*M,)

        idx = idx_flat.reshape(N, M).to(torch.long)
        idx = torch.where(inb, idx, torch.full_like(idx, -1))

        # mask out indices beyond allocated range
        nf = int(self.next_free.item())
        idx = torch.where(idx < nf, idx, torch.full_like(idx, -1))

        # gather positions & compute distances
        idx0 = idx.clamp(min=0)
        cand_pos = self.points[idx0]
        d2 = ((x[:, None, :] - cand_pos) ** 2).sum(dim=-1)

        inf = torch.tensor(float("inf"), device=device, dtype=d2.dtype)
        d2 = torch.where(idx >= 0, d2, inf)

        K_eff = min(K, M)
        vals, cols = torch.topk(d2, k=K_eff, dim=1, largest=False, sorted=True)
        nn_idx = idx.gather(1, cols)
        nn_idx = torch.where(torch.isfinite(vals), nn_idx, torch.full_like(nn_idx, -1))

        if K_eff < K:
            pad = torch.full((N, K - K_eff), -1, device=device, dtype=nn_idx.dtype)
            nn_idx = torch.cat([nn_idx, pad], dim=1)

        return nn_idx

    def lock_pose(self):
        self.rotation_corrections.requires_grad_(False)
        self.translation_corrections.requires_grad_(False)
        self.lock_all_pose_indices()
     
    def unlock_pose(self):
        self.rotation_corrections.requires_grad_(True)
        self.translation_corrections.requires_grad_(True)
        self.unlock_all_pose_indices()
        
    def lock_feature(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def unlock_feature(self):
        for param in self.parameters():
            param.requires_grad = True
    
    def lock_pose_index(self, pose_index:int):
        self.locked_pose_indices.add(pose_index)

    def lock_all_pose_indices(self):
        self.locked_pose_indices = set(range(self.num_poses))

    def unlock_pose_index(self, pose_index:int):
        self.locked_pose_indices.remove(pose_index)

    def unlock_all_pose_indices(self):
        self.locked_pose_indices.clear()

    def pose_correction(self, kf_id: int):
        r = self.rotation_corrections[[kf_id], :]  # (1,3)
        t = self.translation_corrections[kf_id, :, :]  # (3,1)
        if kf_id in self.locked_pose_indices:
            r = r.clone().detach()
            t = t.clone().detach()
        return r, t

    def set_initial_kf_pose(self, kf_id: int, Rwk: torch.Tensor, twk: torch.Tensor, kf_key=None):
        """Set the initial guess for the keyframe pose.
        # TODO: the kf_id is currently a local consecutive index, but 
        this should be replaced by the key completely in the future. 
        We keep it for now to be compatible with the previous version.

        Args:
            kf_id (int): local consecutive index / ID of the keyframe
            Rwk (torch.Tensor): rotation
            twk (torch.Tensor): translation
            kf_key: An optional key associated with this pose. Defaults to None.
        """
        assert Rwk.shape == (3,3)
        assert twk.shape == (3,1)
        assert kf_id < self.num_poses, f"KF ID {kf_id} exceeds the number of poses {self.num_poses}!"
        self.pose_estimates_known[kf_id] = True
        self.Rwk[kf_id,: ,: ] = Rwk.to(self.device)
        self.twk[kf_id, :, :] = twk.to(self.device)
        # Reset perturbations to zero
        with torch.no_grad():
            self.rotation_corrections[kf_id, :].copy_(torch.zeros(3, device=self.device))
            self.translation_corrections[kf_id, :, :].copy_(torch.zeros(3, 1, device=self.device))
        if kf_key is not None:
            self._pose_key_to_id[kf_key] = kf_id
    
    def pose_key_to_id(self, kf_key):
        assert kf_key in self._pose_key_to_id, f"Key {kf_key} not found in pose key to ID mapping!"
        return self._pose_key_to_id[kf_key]
    
    def initial_kf_pose(self, kf_id: int):
        assert self.pose_estimates_known[kf_id], f"Initial pose estimate for KF {kf_id} is not available!"
        return self.Rwk[kf_id, :, :], self.twk[kf_id, :, :]
    
    def initial_kf_pose_in_world(self, kf_id: int):
        return self.initial_kf_pose(kf_id)
    
    def initial_kf_pose_from_key(self, kf_key):
        kf_id = self.pose_key_to_id(kf_key)
        return self.initial_kf_pose(kf_id)
    
    def updated_kf_pose(self, kf_id: int):
        Rwk, twk = self.initial_kf_pose_in_world(kf_id)
        Dr, Dt = self.pose_correction(kf_id)  
        return utils_geometry.apply_pose_correction(
            Rwk, twk, Dr, Dt
        )
    
    def updated_kf_pose_in_world(self, kf_id: int):  
        return self.updated_kf_pose(kf_id)
    
    def updated_kf_pose_from_key(self, kf_key):
        kf_id = self.pose_key_to_id(kf_key)
        return self.updated_kf_pose(kf_id)
    
    def world_to_vox(self, x: torch.Tensor) -> torch.Tensor:
        """ Convert world coordinates to voxel coordinates. """
        g = torch.floor((x - self.bound[:,0]) / self.cell_size).to(torch.long)
        g[:,0] = torch.clamp(g[:,0], 0, self.grid_dims[0] - 1)
        g[:,1] = torch.clamp(g[:,1], 0, self.grid_dims[1] - 1)
        g[:,2] = torch.clamp(g[:,2], 0, self.grid_dims[2] - 1)
        return g

    def init_neural_points(self, x: torch.Tensor, sdf: torch.Tensor):
        """Initialize one neural point per voxel for samples near the surface.
        The stored position is the sample x that triggered it.
        """
        assert x.ndim == 2 and x.shape[1] == 3, f"Invalid x shape {x.shape}"
        assert sdf.ndim == 2 and sdf.shape[1] == 1, f"Invalid sdf shape {sdf.shape}"

        if not self.training:
            return

        # 1) pick candidate samples near surface
        mask = (sdf.squeeze(1).abs() < self.init_threshold)
        if not mask.any():
            return

        xm = x[mask]                       # (M,3)
        gm = self.world_to_vox(xm).to(torch.int64)  # (M,3)

        # 2) deduplicate voxels: keep FIRST triggering sample per voxel
        # torch.unique sorts; to keep "first", we’ll use return_inverse + scatter-min on an index.
        M = gm.shape[0]
        uniq_g, inv = torch.unique(gm, dim=0, return_inverse=True)  # uniq_g: (U,3), inv: (M,)
        U = uniq_g.shape[0]

        # pick a representative sample index per unique voxel (first occurrence)
        # create per-sample indices 0..M-1 and take min per group
        sample_ids = torch.arange(M, device=inv.device, dtype=torch.int64)
        rep = torch.full((U,), M, device=inv.device, dtype=torch.int64)
        rep.scatter_reduce_(0, inv, sample_ids, reduce="amin")  # requires PyTorch 1.12+ / 2.x

        x_rep = xm[rep]     # (U,3) triggering samples (first in each voxel)
        g_rep = uniq_g      # (U,3)

        # 3) check which voxels already exist
        idx_exist = self.lookup_probing(g_rep)  # (U,) int32, -1 if missing
        need = (idx_exist < 0)
        if not need.any():
            return

        g_new = g_rep[need]         # (Un,3)
        x_new = x_rep[need]         # (Un,3)
        Un = g_new.shape[0]

        # 4) allocate new point indices
        nf = int(self.next_free.item())
        if nf + Un > self.max_points:
            logger.warning(f"NeuralPoints full: need {Un} but only {self.max_points - nf} slots left.")
            return

        new_idx = torch.arange(nf, nf + Un, device=self.device, dtype=torch.int32)

        # 5) write point positions (features can stay at 0 or you can init small noise)
        with torch.no_grad():
            self.points[nf:nf + Un] = x_new.to(self.device, dtype=self.dtype)
            # optional: small feature init
            # self.features.data[nf:nf + Un].normal_(0.0, 1e-3)
            self.next_free += Un

        # 6) insert into hash table (returns success per element)
        success = self.insert_probing(g_new, new_idx)  # (Un,) bool
        if not success.all():
            # Roll back failed inserts: mark their points as "unused" by rewinding next_free
            # and (optionally) clearing their positions/features.
            # Note: this simplistic rollback assumes inserts are a contiguous allocation block
            # and failures are rare. If you expect many failures, handle per-element free-list.
            n_ok = int(success.sum().item())
            n_fail = Un - n_ok
            logger.warning(f"insert_probing: {n_fail}/{Un} failed (table too full or probe limit).")

            # Move successful ones to the front to keep storage compact
            ok_mask = success.to(device=self.device)
            if n_ok > 0 and n_ok < Un:
                ok_idx = new_idx[ok_mask]
                ok_pos = self.points[ok_idx]
                ok_feat = self.features.data[ok_idx]

                # compact them into [nf, nf+n_ok)
                with torch.no_grad():
                    self.points[nf:nf + n_ok] = ok_pos
                    self.features.data[nf:nf + n_ok] = ok_feat
                    # rewind allocation to nf+n_ok
                    self.next_free.fill_(nf + n_ok)

                # IMPORTANT: hash table currently points to old ok_idx values.
                # We must update those table entries to the new compacted indices.
                # Easiest: re-insert ok keys with new indices (overwrites same keys).
                new_compact_idx = torch.arange(nf, nf + n_ok, device=self.device, dtype=torch.int32)
                self.insert_probing(g_new[ok_mask], new_compact_idx)
            else:
                # all failed
                with torch.no_grad():
                    self.next_free.fill_(nf)

    def forward(self, x: torch.Tensor, K: int = 8) -> torch.Tensor:
        """ Predict SDF at world coords x using inverse-distance weighting over K neighbors:
            w_j = ||p - x_j||^{-2}
            s(p) = sum_j (w_j / sum_k w_k) * s_j
        """
        assert x.ndim == 2 and x.shape[1] == 3, f"Invalid input shape {x.shape}!"
        assert self.decoder is not None, "Decoder is not initialized."
        
        N = x.shape[0]
        
        # Overriding these for test purposes FIXME
        K = 15 
        
        # Neighbor lookup
        Np_idx = self.query_neighbors(x, K=K)
        valid = Np_idx >= 0 # (N,K) bool mask for valid neighbors

        # Gather neighbor positions/features
        idx0 = Np_idx.clamp(min=0) # (N,K) - Clamping to avoid indexing with -1 for invalid neighbors. We'll zero out these later.
        Np_pos = self.points[idx0] # (N,K,3)
        Np_feat = self.features[idx0] # (N,K,fdim)

        # Zero out invalid neighbor data so that they don't contribute to the final prediction
        if (~valid).any():
            Np_pos = Np_pos.clone()
            Np_feat = Np_feat.clone()
            Np_pos[~valid] = 0.0
            Np_feat[~valid] = 0.0

        # Compute weights: w = ||p - x_j||^{-2}
        # p is query x, x_j is neighbor position
        diff = x[:,None, :] - Np_pos # (N,K,3)
        d2 = (diff * diff).sum(dim=-1) # (N,K)
        eps = 1e-12
        w = 1.0 / (d2 + eps) # (N,K)
        w[~valid] = 0.0

        # normalize weights per point (avoid div-by-zero when no neighbors)
        w_sum = w.sum(dim=1, keepdim=True) # (N,1)
        w_norm = w / (w_sum + eps) # (N,K)

        # Build decoder input per neighbor: [feature, position]
        decoder_in = torch.cat([Np_feat, Np_pos], dim=-1) # (N,K,fdim+3)
        decoder_in = decoder_in.view(N * K, -1) # (N*K,D)

        # Decode per-neighbor SDF s_j        
        s_j = self.decoder(decoder_in) # (N*K,D)
        D = s_j.shape[-1]
        s_j = s_j.view(N, K, D) # (N,K,D)

        # Mask invalid
        s_j = s_j.masked_fill((~valid)[..., None], 0.0) # (N,K,D)

        # Weighted sum over neighbors
        s = (w_norm[..., None] * s_j).sum(dim=1) # (N,D)
        
        # Just set output to 1 if no valid neighbors to represent free space (for now)
        no_nb = (valid.sum(dim=1) == 0) # (N,)
        if no_nb.any():
            s = s.clone()
            s[no_nb] = 0.0
            s[no_nb, 0] = 1.0

        return s # (N,D)
    
    def print_active_info(self):
        n_active = int(self.next_free.item())
        n_max = int(self.max_points)
        fill_points = 100.0 * n_active / max(1, n_max)

        # table occupancy (how many hash slots are non-empty)
        empty_key = torch.tensor([-1, -1, -1], device=self.table_keys.device, dtype=self.table_keys.dtype)
        n_slots_used = int((self.table_keys != empty_key).all(dim=1).sum().item())
        fill_table = 100.0 * n_slots_used / max(1, int(self.T))

        logger.info(
            f"NeuralPoints active points: {n_active}/{n_max} ({fill_points:.2f}%). "
            f"Hash table used: {n_slots_used}/{int(self.T)} ({fill_table:.2f}%)."
        )


    def params_at_level(self, level):
        # FIXME: right now this always return the full set of params!
        return list(self.parameters())
    
    def print_kf_pose_info(self):
        max_rot = torch.max(torch.linalg.norm(self.rotation_corrections, dim=1))
        max_tran = torch.max(torch.linalg.norm(self.translation_corrections.squeeze(2), dim=1))
        logger.info(f"GridNet KF pose corrections: max_rot={math.degrees(max_rot):.3f}deg, max_tran={max_tran:.3f}m.")
        
    def print_feature_info(self):
        logger.warning("Feature info not implemented yet for neural points.")
