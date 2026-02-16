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

class NeuralPoints(BaseNet):
    """
    A lightweight implementation of neural point-based map inspired by PIN-SLAM:
    Paper: https://arxiv.org/abs/2401.09101
    Code:  https://github.com/PRBonn/PIN_SLAM/tree/main
    """
    def __init__(self,
        cfg: dict, 
        device = 'cuda:0',
        dtype = torch.float32,
    ):
        super(NeuralPoints, self).__init__(cfg, device, dtype)    
        self.device = device
        self.dtype = dtype
        self.init_grid(cfg)
        self.init_neural_points(cfg)
        self.init_decoder(cfg)
        self.init_poses(cfg)
        self.print_trainable_params()
    
    def init_grid(self, cfg):
        
        # TODO: Move this elsewhere
        # Config
        self.cell_size = 0.1
        self.fdim = 4
        self.num_levels = 1 # To be compatible with trainer
        
        assert self.bound.shape == (3, 2), f"Invalid bound shape {self.bound.shape}!"
        
        self.grid_dims = torch.ceil((self.bound[:,1] - self.bound[:,0]) / self.cell_size).to(torch.long)  # (3,)
        nx, ny, nz = self.grid_dims.tolist()
        self.num_cells = nx * ny * nz
        
        # Recording active points
        self.register_buffer("active", torch.zeros((self.num_cells,), device=self.device, dtype=torch.bool))
    
    def init_neural_points(self, cfg):
        """ Initializes neural point representation? Maybe it should be a class instead? """
        
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
        self.register_buffer("points", torch.zeros((self.num_cells, 3), device=self.device, dtype=self.dtype))

        # Trainable feature buffer
        self.features = nn.Parameter(torch.zeros((self.num_cells, self.fdim), device=self.device, dtype=self.dtype))

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

        # world -> voxel coords (N,3)
        g = torch.floor((x - self.bound[:,0]) / self.cell_size).to(torch.long)
        g[:,0] = torch.clamp(g[:,0], 0, self.grid_dims[0] - 1)
        g[:,1] = torch.clamp(g[:,1], 0, self.grid_dims[1] - 1)
        g[:,2] = torch.clamp(g[:,2], 0, self.grid_dims[2] - 1)

        # Build offsets for Nn x Nn x Nn cube (M,3)
        r = Nn // 2
        o = torch.arange(-r, r + 1, device=device, dtype=torch.long)
        ox, oy, oz = torch.meshgrid(o, o, o, indexing="ij")  # (Nn,Nn,Nn)
        offsets = torch.stack([ox, oy, oz], dim=-1).reshape(-1, 3)  # (M,3)
        M = offsets.shape[0]

        # Candidate voxel coords (N,M,3)
        cand = g[:,None, :] + offsets[None, :,:] # broadcast to (N,M,3)

        # in-bounds mask
        inb = (
            (cand[..., 0] >= 0) & (cand[..., 0] < self.grid_dims[0]) &
            (cand[..., 1] >= 0) & (cand[..., 1] < self.grid_dims[1]) &
            (cand[..., 2] >= 0) & (cand[..., 2] < self.grid_dims[2])
        )

        # Voxel coords -> linear cell idx (N,M)
        nx, ny = self.grid_dims[0], self.grid_dims[1]
        cand_idx = cand[..., 0] + nx * (cand[..., 1] + ny * cand[..., 2]) # (N,M)

        # Mark out-of-bounds as invalid
        cand_idx = torch.where(inb, cand_idx, torch.full_like(cand_idx, -1))
        cand_idx_clamped = cand_idx.clamp(min=0) # (N,M)
        is_active = self.active[cand_idx_clamped] & (cand_idx >= 0) # (N,M)

        # Gather positions (N,M,3), compute distances (N,M)
        cand_pos = self.points[cand_idx_clamped] # (N,M,3) (garbage for inactive, but masked out)
        d = x[:,None, :] - cand_pos
        d2 = (d * d).sum(dim=-1) # (N,M) using squared distance to make it scalar

        # Set inactive candidates to +inf distance so they won't be selected
        inf = torch.tensor(float("inf"), device=device, dtype=d2.dtype)
        d2 = torch.where(is_active, d2, inf)

        # Take K nearest
        # topk works even if many are inf; we’ll convert inf-selected entries to -1.
        K_eff = min(K, M)
        vals, cols = torch.topk(d2, k=K_eff, dim=1, largest=False, sorted=True) # (N,K_eff)
        nn_idx = cand_idx.gather(1, cols) # (N,K_eff)

        # any picked inf means "no neighbor"
        nn_idx = torch.where(torch.isfinite(vals), nn_idx, torch.full_like(nn_idx, -1))

        # pad if K > M
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
    
    def world_to_grid(self, x: torch.Tensor) -> torch.Tensor:
        """ Map world coords to a linear grid index (one cell -> one index).
        Args:
            x: (N,3) world coordinates
        Returns:
            idx: (N,) torch.long in [0, num_cells-1]
        """
        assert x.ndim == 2 and x.shape[1] == 3, f"Invalid input shape {x.shape}!"

        # voxel coords (N,3)
        g = torch.floor((x - self.bound[:,0]) / self.cell_size).to(torch.long)

        # clamp to grid bounds (should we discard out-of-bounds points instead?)
        g[:,0] = torch.clamp(g[:,0], 0, self.grid_dims[0] - 1)
        g[:,1] = torch.clamp(g[:,1], 0, self.grid_dims[1] - 1)
        g[:,2] = torch.clamp(g[:,2], 0, self.grid_dims[2] - 1)

        # 3D -> 1D index transformation: idx = ix + nx*(iy + ny*iz)
        nx, ny = self.grid_dims[0], self.grid_dims[1]
        idx = g[:,0] + nx * (g[:,1] + ny * g[:,2])

        return idx

    def query_feature(self, x: torch.Tensor, K: int):
        """
        Query neighbor neural-point inputs for samples x (world coords).

        Behavior:
            1) Ensures the voxel containing each x is activated (stores point position once).
            2) Queries K closest neighbor indices.
            3) Returns per-neighbor concatenated [feature, position].

        Args:
            x: (N,3) world coordinates
            K: number of neighbors

        Returns:
            Np_idx: (N,K) long neighbor indices, with -1 for missing
            valid: (N,K) bool mask for valid neighbors
        """
        assert x.ndim == 2 and x.shape[1] == 3

        center_idx = self.world_to_grid(x) # (N,)
        inactive = ~self.active[center_idx] # (N,) bool mask where inactive indices are True

        # Initialize neural points for inactive voxels: set position, mark active
        if inactive.any():
            idx_new = center_idx[inactive]
            with torch.no_grad():
                self.points[idx_new] = x[inactive]
                self.active[idx_new] = True

        Np_idx = self.query_neighbors(x, K) # (N,K), -1 for missing
        valid = Np_idx >= 0

        return Np_idx, valid

    def forward(self, x: torch.Tensor, K: int = 8) -> torch.Tensor:
        """ Predict SDF at world coords x using inverse-distance weighting over K neighbors:
            w_j = ||p - x_j||^{-2}
            s(p) = sum_j (w_j / sum_k w_k) * s_j
        """
        assert x.ndim == 2 and x.shape[1] == 3, f"Invalid input shape {x.shape}!"
        assert self.decoder is not None, "Decoder is not initialized."
        
        N = x.shape[0]
        
        # Neighbor lookup (also activates voxels if not already)
        Np_idx, valid = self.query_feature(x, K=K) # (N,K), (N,K)

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
        decoder_in = decoder_in.view(N * K, -1) # (N*K,input_dim)

        # Decode per-neighbor SDF s_j
        s_j = self.decoder(decoder_in) # (N*K, out_dim=1)
        s_j = s_j.view(N, K) # (N,K)
        s_j[~valid] = 0.0

        # Weighted sum: s = sum_j w_norm * s_j
        s = (w_norm * s_j).sum(dim=1) # (N,)

        return s
        
    def params_at_level(self, level):
        # FIXME: right now this always return the full set of params!
        return list(self.parameters())
    
    def print_kf_pose_info(self):
        max_rot = torch.max(torch.linalg.norm(self.rotation_corrections, dim=1))
        max_tran = torch.max(torch.linalg.norm(self.translation_corrections.squeeze(2), dim=1))
        logger.info(f"GridNet KF pose corrections: max_rot={math.degrees(max_rot):.3f}deg, max_tran={max_tran:.3f}m.")
        
    def print_feature_info(self):
        logger.warning("Feature info not implemented yet for neural points.")
