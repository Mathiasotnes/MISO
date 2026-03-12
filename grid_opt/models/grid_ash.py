import numpy as np
import math
import torch
import torch.nn as nn
from .base_net import BaseNet
from .modules import MLPNet
from .grid_modules import *
import grid_opt.utils.utils_geometry as utils_geometry
import logging

from ash.core import ASHEngine

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class GridASH(BaseNet):
    def __init__(self,
        cfg: dict, 
        device = 'cuda:0',
        dtype = torch.float32,
    ):
        super(GridASH, self).__init__(cfg, device, dtype)    
        assert self.d == 3, "Currently only 3D ASH grid is supported!"    
        self.init_grid(cfg)
        self.init_decoder(cfg)
        self.init_poses(cfg)
        self.print_summary()

    def init_grid(self, cfg):
        self.num_levels = cfg['grid']['n_levels']
        self.second_order_grid_sample = 'second_order_grid_sample' in cfg['grid'] and cfg['grid']['second_order_grid_sample']
        self.base_cell_size = cfg['grid']['base_cell_size']
        self.scale_factor = cfg['grid']['per_level_scale']
        self.fdim = cfg['grid']['feature_dim']
        self.cell_sizes = []
        self.features = nn.ParameterList()
        self.ash_engines = nn.ModuleList()
        
        self.max_voxels_per_level = int(1e6)  # TODO: make this configurable
        
        for level in range(self.num_levels):
            cell_size = self.base_cell_size / (self.scale_factor**level)
            self.cell_sizes.append(cell_size)
            
            
            ash_engine = ASHEngine(
                dim=3, # Key dimension. We want to use (x,y,z) as keys
                capacity=self.max_voxels_per_level, 
                device=self.device
            )
            
            feat = nn.Parameter(
                torch.zeros(
                    self.max_voxels_per_level,
                    self.fdim,
                    device=self.device
                )
            )
            
            self.ash_engines.append(ash_engine)
            self.features.append(feat)
            
        self.ignore_level_ = np.zeros(self.num_levels).astype(bool)
        
        # For trilinear interpolation calculations
        self.register_buffer(
            'corner_offsets',
            torch.tensor(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [1, 1, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [0, 1, 1],
                    [1, 1, 1],
                ],
                dtype=torch.int32,
                device=self.device,
            )
        )

    def init_decoder(self, cfg):
        self.decoder_hidden_dim = cfg['decoder']['hidden_dim']
        self.decoder_hidden_layers = cfg['decoder']['hidden_layers']
        self.decoder_out_dim = cfg['decoder']['out_dim']
        input_dim = self.num_levels * self.fdim
        
        logger.debug(f"Using MLP decoder.")
        self.decoder = MLPNet(
            input_dim=input_dim,
            output_dim=self.decoder_out_dim,
            hidden_dim=self.decoder_hidden_dim,
            hidden_layers=self.decoder_hidden_layers,
            bias=True
        )
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
        
    def print_summary(self):
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        encoding_params = sum(p.numel() for p in self.features)
        logger.info(
            f"\n{'='*60}\n"
            f" GridASH\n"
            f"   * Encoding levels          : {self.num_levels}\n"
            f"   * Encoding feature dim     : {self.fdim}\n"
            f"   * Base cell size           : {self.base_cell_size}\n"
            f"   * Per-level scale          : {self.scale_factor}\n"
            f"   * Decoder hidden layers    : {self.decoder_hidden_layers}\n"
            f"   * Decoder hidden dim       : {self.decoder_hidden_dim}\n"
            f"{'='*60}\n"
            f"   * Trainable params         : {total_trainable:,}\n"
            f"{'='*60}\n"
            f"   * Encoding params          : {encoding_params:,}\n"
            f"   * Decoder params           : {decoder_params:,}\n"
            f"{'='*60}"
        )
    
    def lock_pose(self):
        self.rotation_corrections.requires_grad_(False)
        self.translation_corrections.requires_grad_(False)
        self.lock_all_pose_indices()
     
    def unlock_pose(self):
        self.rotation_corrections.requires_grad_(True)
        self.translation_corrections.requires_grad_(True)
        self.unlock_all_pose_indices()
        
    def ignore_level(self, l):
        """ Ignoring a feature level. The corresponding contribution from this level to the decoder will be set to zero. """
        self.ignore_level_[l] = True
        logger.warning(f"Ignore level: {self.ignore_level_}")

    def include_level(self, l):
        self.ignore_level_[l] = False
        logger.warning(f"Ignore level: {self.ignore_level_}")

    def lock_level(self, l):
        """ Locking (fixing) the features at level l at the current value. """
        self.features[l].requires_grad = False

    def unlock_level(self, l):
        self.features[l].requires_grad = True

    def lock_feature(self):
        for level in range(self.num_levels):
            self.lock_level(level)
    
    def unlock_feature(self):
        for level in range(self.num_levels):
            self.unlock_level(level)
    
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
    
    @torch.no_grad()
    def activate_level(self, x: torch.Tensor, level: int, init_std: float = 1e-4) -> int:
        """ Activate the 8 trilinear corner vertices for points x at one level.

        Args:
            x: (N, 3) world coordinates
            level: grid level
        Returns:
            number of newly inserted vertices
        """
        assert x.ndim == 2 and x.shape[1] == 3

        cell_size = self.cell_sizes[level]
        x_grid = x / cell_size
        base = torch.floor(x_grid).to(torch.int32)  # (N,3)

        # 8 corner lattice coordinates
        corner_coords = base[:, None, :] + self.corner_offsets[None, :, :]   # (N,8,3)
        corner_coords = corner_coords.reshape(-1, 3)                          # (N*8,3)
        corner_coords = torch.unique(corner_coords, dim=0)

        indices, masks = self.ash_engines[level].find(corner_coords)
        new_coords = corner_coords[~masks]

        if new_coords.numel() == 0:
            return 0

        self.ash_engines[level].insert_keys(new_coords)

        new_indices, new_masks = self.ash_engines[level].find(new_coords)
        assert new_masks.all()

        if init_std > 0:
            self.features[level].data[new_indices].normal_(mean=0.0, std=init_std)
        else:
            self.features[level].data[new_indices].zero_()

        return int(new_coords.shape[0])

    @torch.no_grad()
    def activate_features(self, x: torch.Tensor, init_std: float = 1e-4):
        """
        Activate all levels for queried world coordinates.
        Returns list of inserted counts per level.
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.to(self.device, dtype=self.dtype)

        inserted = []
        for level in range(self.num_levels):
            inserted.append(self.activate_level(x, level, init_std=init_std))
        return inserted
    
    def _query_feature_level(self, x: torch.Tensor, level: int) -> torch.Tensor:
        """ 
        Trilinearly interpolate features from one level. Looks up voxels in the ASH engine.

        Args:
            x: (N, 3) world coordinates

        Returns:
            f_level: (N, fdim)
        """
        assert x.ndim == 2 and x.shape[1] == 3, f"Expected x to have shape (N,3), got {x.shape}"

        if self.ignore_level_[level]:
            return torch.zeros(x.shape[0], self.fdim, device=self.device, dtype=self.dtype)

        cell_size = self.cell_sizes[level]

        # Convert world coordinates to grid coordinates
        x_grid = x / cell_size  # (N, 3), float
        base = torch.floor(x_grid).to(torch.int32)  # (N, 3)
        frac = (x_grid - base.to(x_grid.dtype)).to(self.dtype)  # (N, 3) in [0,1)

        # 8 corner coordinates for each query point
        # shape: (N, 8, 3)
        corner_coords = base[:, None, :] + self.corner_offsets[None, :, :]
        corner_coords_flat = corner_coords.reshape(-1, 3)  # (N*8, 3)

        # Query ASH
        corner_indices, corner_masks = self.ash_engines[level].find(corner_coords_flat)
        # corner_indices: (N*8,)
        # corner_masks:   (N*8,)

        N = x.shape[0]

        # Features that's not found in ASH gets set to 0.
        # Not sure if this is the behavior we want.
        safe_indices = corner_indices.clone()
        safe_indices[~corner_masks] = 0
        corner_feats = self.features[level][safe_indices] # (N*8, fdim)
        corner_feats = corner_feats.view(N, 8, self.fdim) # (N, 8, fdim)
        corner_masks = corner_masks.view(N, 8)
        corner_feats = corner_feats * corner_masks.unsqueeze(-1).to(corner_feats.dtype)

        # Trilinear weights
        fx = frac[:, 0]
        fy = frac[:, 1]
        fz = frac[:, 2]

        weights = torch.stack(
            [
                (1 - fx) * (1 - fy) * (1 - fz),  # 000
                fx * (1 - fy) * (1 - fz),        # 100
                (1 - fx) * fy * (1 - fz),        # 010
                fx * fy * (1 - fz),              # 110
                (1 - fx) * (1 - fy) * fz,        # 001
                fx * (1 - fy) * fz,              # 101
                (1 - fx) * fy * fz,              # 011
                fx * fy * fz,                    # 111
            ],
            dim=1
        )  # (N, 8)

        # Missing corners should not contribute
        valid_weights = weights * corner_masks.to(weights.dtype)  # (N, 8)

        # Weighted sum
        f_level = (corner_feats * valid_weights.unsqueeze(-1)).sum(dim=1)  # (N, fdim)

        # Renormalize over existing corners only
        weight_sum = valid_weights.sum(dim=1, keepdim=True)  # (N, 1)
        has_any = weight_sum.squeeze(-1) > 0

        out = torch.zeros_like(f_level)
        out[has_any] = f_level[has_any] / weight_sum[has_any].clamp_min(1e-12)

        return out
    
    def query_feature(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)

        assert x.ndim == 2 and x.shape[1] == 3, f"Expected x to have shape (N,3), got {x.shape}"
        x = x.to(device=self.device, dtype=self.dtype)

        level_features = []
        for level in range(self.num_levels):
            f_level = self._query_feature_level(x, level)
            level_features.append(f_level)

        return torch.cat(level_features, dim=-1)
    
    def forward(self, x):
        f = self.query_feature(x)
        return self.decoder(f)
    
    def params_at_level(self, level):
        return [self.features[level]]
    
    def print_kf_pose_info(self):
        max_rot = torch.max(torch.linalg.norm(self.rotation_corrections, dim=1))
        max_tran = torch.max(torch.linalg.norm(self.translation_corrections.squeeze(2), dim=1))
        logger.info(f"GridNet KF pose corrections: max_rot={math.degrees(max_rot):.3f}deg, max_tran={max_tran:.3f}m.")
        
    def print_feature_info(self):
        logger.warning("Feature info not implemented yet for GridASH.")
        
    def print_ash_stats(self):
        dim_len = self.bound[:, 1] - self.bound[:, 0] # (d,)

        total_active = 0
        total_dense = 0

        lines = [
            f"\n{'='*60}",
            " ASH Engine Statistics",
            f"{'='*60}",
        ]

        for l in range(self.num_levels):
            active      = int(self.ash_engines[l].size())
            n_cells     = torch.ceil(dim_len / self.cell_sizes[l]).long()
            dense       = int(torch.prod(n_cells + 1).item())
            sparsity    = 100.0 * active / dense if dense > 0 else 0.0

            total_active    += active
            total_dense     += dense

            lines.extend([
                f" Level {l}:",
                f"   * Cell size        : {self.cell_sizes[l]:.6f}",
                f"   * Active features  : {active:,}",
                f"   * Dense features   : {dense:,}",
                f"   * Sparsity         : {sparsity:.2f}%",
                f"{'-'*60}",
            ])

        total_sparsity = 100.0 * total_active / total_dense if total_dense > 0 else 0.0

        lines.extend([
            " Total:",
            f"   * Active features  : {total_active:,}",
            f"   * Dense features   : {total_dense:,}",
            f"   * Sparsity         : {total_sparsity:.2f}%",
            f"{'='*60}",
        ])

        logger.info("\n".join(lines))
