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
    
    ######################################################
    # GridASH specific implementation
    ######################################################
    
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
        
        self.ash_engines = nn.ModuleList() # We create a hash map per layer.
        self._init_capacity = [int(1_024), int(8_192)]  # TODO: make this configurable (2^10, 2^13) for now.
        self.capacity = list(self._init_capacity) # Current buffer capacity. Buffers will double capacity whenever the capacity is exceeded.
        self.active_features = nn.ParameterList()
        self._active_ash_indices = []
        
        for level in range(self.num_levels):
            cell_size = self.base_cell_size / (self.scale_factor**level)
            self.cell_sizes.append(cell_size)
            ash_engine = ASHEngine(
                dim=3, # Key dimension. We want to use (x,y,z) as keys. We might want to pack these into a single int in the future?
                capacity=self._init_capacity[level], 
                device=self.device
            )
            self.ash_engines.append(ash_engine)
            self.active_features.append(nn.Parameter(torch.zeros(0, self.fdim, device=self.device, dtype=self.dtype)))
            self._active_ash_indices.append(torch.zeros(0, dtype=torch.long, device=self.device))
            self.register_buffer(f'_features_{level}', torch.zeros(self._init_capacity[level], self.fdim, device=self.device, dtype=self.dtype))
            self.register_buffer(f'_active_lut_{level}', torch.zeros(0, dtype=torch.long, device=self.device))

            
        self.ignore_level_ = np.zeros(self.num_levels).astype(bool)
        
        # For trilinear interpolation calculations
        self.register_buffer(
            'corner_offsets',
            torch.tensor(
                [[0,0,0],[1,0,0],[0,1,0],[1,1,0],
                [0,0,1],[1,0,1],[0,1,1],[1,1,1]],
                dtype=torch.int32, device=self.device,
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
    
    def print_summary(self):
        bytes_per_elem = torch.finfo(self.dtype).bits // 8

        def mb(num_elements):
            return (num_elements * bytes_per_elem) / (1024 ** 2)

        # Parameter counts
        total_trainable  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        decoder_params   = sum(p.numel() for p in self.decoder.parameters())
        encoding_params  = sum(p.numel() for p in self.active_features)

        # Other memory buffers
        feature_buf_elems  = sum(getattr(self, f'_features_{l}').numel() for l in range(self.num_levels))
        active_feat_elems  = sum(self.active_features[l].numel() for l in range(self.num_levels))
        lut_elems          = sum(getattr(self, f'_active_lut_{l}').numel() for l in range(self.num_levels))
        lut_mb             = (lut_elems * 8) / (1024 ** 2)  # long = 8 bytes

        lines = [
            f"\n{'='*60}",
            f" GridASH Summary ({'TRAINING' if self.training else 'EVAL'} mode)",
            f"{'='*60}",
            f" Architecture",
            f"   * Encoding levels          : {self.num_levels}",
            f"   * Encoding feature dim     : {self.fdim}",
            f"   * Base cell size           : {self.base_cell_size}",
            f"   * Per-level scale          : {self.scale_factor}",
            f"   * Decoder hidden layers    : {self.decoder_hidden_layers}",
            f"   * Decoder hidden dim       : {self.decoder_hidden_dim}",
            f"{'='*60}",
            f" Parameters",
            f"   * Total trainable          : {total_trainable:,}",
            f"   * Encoding (active)        : {encoding_params:,}  ({mb(active_feat_elems):.2f} MB)",
            f"   * Decoder                  : {decoder_params:,}  ({mb(decoder_params):.2f} MB)",
            f"{'='*60}",
            f" Feature Buffers (persistent)",
        ]

        total_allocated = 0
        total_active    = 0
        for l in range(self.num_levels):
            allocated   = getattr(self, f'_features_{l}').shape[0]  # capacity
            active      = int(self.ash_engines[l].size())           # actually occupied
            utilization = 100.0 * active / allocated if allocated > 0 else 0.0
            total_allocated += allocated
            total_active    += active
            lines += [
                f"   Level {l}:",
                f"     * Cell size            : {self.cell_sizes[l]:.6f}",
                f"     * Capacity (allocated) : {allocated:,}  ({mb(allocated * self.fdim):.2f} MB)",
                f"     * Occupied (ASH)       : {active:,}  ({utilization:.1f}% utilization)",
                f"     * Active (trainable)   : {self.active_features[l].shape[0]:,}",
                f"     * LUT size             : {getattr(self, f'_active_lut_{l}').shape[0]:,}",
            ]

        total_util = 100.0 * total_active / total_allocated if total_allocated > 0 else 0.0
        lines += [
            f"   Total:",
            f"     * Capacity (allocated) : {total_allocated:,}  ({mb(feature_buf_elems):.2f} MB)",
            f"     * Occupied (ASH)       : {total_active:,}  ({total_util:.1f}% utilization)",
            f"{'='*60}",
            f" Training-only State {'[CLEARED]' if self.active_features[0].numel() == 0 else ''}",
            f"   * Active features          : {active_feat_elems:,}  ({mb(active_feat_elems):.2f} MB)",
            f"   * LUT buffers              : {lut_elems:,}  ({lut_mb:.2f} MB)",
            f"{'='*60}",
            f" Estimated Total GPU Memory",
            f"   * Feature buffers          : {mb(feature_buf_elems):.2f} MB",
            f"   * Active features          : {mb(active_feat_elems):.2f} MB",
            f"   * LUT buffers              : {lut_mb:.2f} MB",
            f"   * Decoder weights          : {mb(decoder_params):.2f} MB",
            f"   * Grand total (est.)       : {mb(feature_buf_elems) + mb(active_feat_elems) + lut_mb + mb(decoder_params):.2f} MB",
            f"{'='*60}",
        ]

        logger.info("\n".join(lines))
        
    @property
    def features(self):
        return [getattr(self, f'_features_{l}') for l in range(self.num_levels)]

    @property
    def _active_lut(self):
        return [getattr(self, f'_active_lut_{l}') for l in range(self.num_levels)]
    
    @torch.no_grad()
    def prepare_features(self, x: torch.Tensor, init_std: float = 1e-4):
        """
        1. Insert all 8-corner keys for every point in x at each level if not present.
        2. Rebuild active_features as trainable nn.Parameters containing only
        the voxels that x touches at each level.

        Args:
            x: (N, 3) world coordinates
            init_std: std for initialising newly inserted features
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.to(self.device, dtype=self.dtype)
        new_features = 0

        for level in range(self.num_levels):

            # Insert new keys to ASH. This also extends buffers when capacity is exceeded
            corner_coords = self.world_to_grid_corners(x, level) # (M, 3), unique corner coordinates touched by x at this level            
            new_features += self.insert_features_at_level(corner_coords, level, init_std=init_std)

            # Find all features touched by x and make them trainable parameters
            active_indices, active_masks = self.ash_engines[level].find(corner_coords)
            assert active_masks.all(), "ERROR: Keys were just inserted but can't be found."

            active_feats = self.features[level][active_indices].clone()
            self.active_features[level] = nn.Parameter(active_feats.to(self.dtype))
            self._active_ash_indices[level] = active_indices
            
            # Create a LUT from features -> active_features. This will be used to query trainable params during training, and can be removed after training.
            lut = torch.full((self.ash_engines[level].capacity,), -1, dtype=torch.long, device=self.device)
            lut[active_indices] = torch.arange(active_indices.shape[0], device=self.device)
            self.register_buffer(f'_active_lut_{level}', lut)

        logger.info(f"prepare_features: Inserted: {new_features} | Active: {sum(len(p) for p in self.active_features)}")
    
    @torch.no_grad()
    def sync_active_to_store(self):
        """ Copy current active_features (post-optimizer-step values) back into features. """
        for level in range(self.num_levels):
            idx = self._active_ash_indices[level]
            if idx.numel() > 0:
                self.features[level][idx] = self.active_features[level].data
                
    def clear_training_state(self):
        """
        Call after training is complete to free memory used by training-only state:
        - active_features (trainable parameters, only needed for gradient flow)
        - _active_lut (only needed to map feat_idx -> active_features during training)
        - _active_ash_indices (only needed to rebuild the LUT and sync back to features)
        
        Make sure to call sync_active_to_store() before this to persist the final
        optimized values back into the feature buffers.
        """
        assert not self.training, "Call model.eval() before clearing training state."

        for level in range(self.num_levels):
            self.active_features[level] = nn.Parameter(
                torch.zeros(0, self.fdim, device=self.device, dtype=self.dtype),
                requires_grad=False
            )
            self.register_buffer(f'_active_lut_{level}', torch.zeros(0, dtype=torch.long, device=self.device))
            self._active_ash_indices[level] = torch.zeros(0, dtype=torch.long, device=self.device)

        logger.info("Cleared training state!")
                
    @torch.no_grad()
    def extend_capacity(self, level: int):
        """ Double capacity for one level in both ASH engine and feature buffer. Make sure 
        parameters are synced to feature buffer before calling this. """
        old_cap = int(self.capacity[level])
        new_cap = 2 * old_cap

        old_features = self.features[level]
        new_features = torch.zeros(new_cap, self.fdim, device=self.device, dtype=self.dtype)

        self.ash_engines[level].resize(
            new_cap,
            old_external_values={"features": old_features},
            new_external_values={"features": new_features},
        )

        self.register_buffer(f'_features_{level}', new_features)
        self.capacity[level] = new_cap

        assert self.features[level].shape[0] == self.ash_engines[level].capacity
        logger.info(f"Extended level {level} capacity: {old_cap} -> {new_cap}")
    
    def world_to_grid_corners(self, x: torch.Tensor, level: int):
        """ For each point in x, return the 8 corner coordinates of the grid cell it belongs to at the specified level. """
        assert x.ndim == 2 and x.shape[1] == 3

        cell_size = self.cell_sizes[level]
        x_grid = x / cell_size
        base = torch.floor(x_grid).to(torch.int32)

        corner_coords = base[:, None, :] + self.corner_offsets[None, :, :] # (N, 8, 3)
        corner_coords = corner_coords.reshape(-1, 3) # (N*8, 3)
        corner_coords = torch.unique(corner_coords, dim=0) # (M, 3)
        return corner_coords
    
    @torch.no_grad()
    def insert_features_at_level(self, corner_coords: torch.Tensor, level: int, init_std: float = 1e-4) -> int:
        """ Insert the 8 trilinear corner vertices for points x at one level into the ASH engine
        and initialize them in the feature buffer. Extends capacity whenever capacity is exceeded.
        """
        _, masks = self.ash_engines[level].find(corner_coords)
        new_coords = corner_coords[~masks]
        
        if new_coords.numel() == 0:
            return 0
        
        saved_coords = int(self.ash_engines[level].size())
        required_capacity = saved_coords + new_coords.shape[0]
        
        while required_capacity > self.capacity[level]:
            self.extend_capacity(level)


        self.ash_engines[level].insert_keys(new_coords)

        new_indices, new_masks = self.ash_engines[level].find(new_coords)
        assert new_masks.all()

        if init_std > 0:
            self.features[level][new_indices].normal_(mean=0.0, std=init_std)
        else:
            self.features[level][new_indices].zero_()

        return int(new_coords.shape[0])
        
    def _query_feature_level(self, x: torch.Tensor, level: int) -> torch.Tensor:
        """ Query the feature at level l for points x using trilinear interpolation. Uses active_features 
        as trainable features during training, and features buffer directly during eval. """
        
        assert x.ndim == 2 and x.shape[1] == 3

        if self.ignore_level_[level]:
            return torch.zeros(x.shape[0], self.fdim, device=self.device, dtype=self.dtype)

        N = x.shape[0]
        cell_size = self.cell_sizes[level]
        x_grid = x / cell_size
        base = torch.floor(x_grid).to(torch.int32)
        frac = (x_grid - base.to(x_grid.dtype)).to(self.dtype)

        # Include all 8 corners around all points in x as keys to query from ASH
        coord_keys = base[:, None, :] + self.corner_offsets[None, :, :]
        coord_keys = coord_keys.reshape(-1, 3) # (N*8, 3)

        # ASH maps our corner coordinates to indicies in the self.feature buffer
        feat_idx, feat_mask = self.ash_engines[level].find(coord_keys)
        feat_idx[~feat_mask] = 0 # Set invalid indices to 0 to avoid errors. These features must be filtered out later.
        # feat_idx:  (N*8,) Index in the feature buffer for each corner key.
        # feat_mask: (N*8,) Valid mask. True if key was found in ASH.

        if self.training:
            # Training: Use the features -> active_features LUT to find the corresponding indices in their trainable buffer
            active_rows = self._active_lut[level][feat_idx] # (N*8,), -1 if not active
            in_active = (active_rows >= 0) & feat_mask
            active_rows[~in_active] = 0
            feats = self.active_features[level][active_rows]  # (N*8, fdim), differentiable
            valid = in_active
        else:
            # Inference: Use the non-trainable feature buffer directly
            feats = self.features[level][feat_idx] # (N*8, fdim)
            valid = feat_mask
            
        feats = feats.view(N, 8, self.fdim)
        valid = valid.view(N, 8)
        feats = feats * valid.unsqueeze(-1).to(feats.dtype)

        # Trilinear interpolation weights
        fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]
        weights = torch.stack([
            (1-fx)*(1-fy)*(1-fz), fx*(1-fy)*(1-fz),
            (1-fx)*fy*(1-fz),     fx*fy*(1-fz),
            (1-fx)*(1-fy)*fz,     fx*(1-fy)*fz,
            (1-fx)*fy*fz,         fx*fy*fz,
        ], dim=1)  # (N, 8)

        valid_weights = weights * valid.to(weights.dtype)
        f_level = (feats * valid_weights.unsqueeze(-1)).sum(dim=1)

        weight_sum = valid_weights.sum(dim=1, keepdim=True)
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
        
    ######################################################
    # MISO Compatibility
    ######################################################
    # This is only to stay compatible with current trainer and loss implementations
    # to avoid writing these from scratch.
        
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
        self.active_features[l].requires_grad = False

    def unlock_level(self, l):
        self.active_features[l].requires_grad = True

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
    
    def params_at_level(self, level):
        return [self.active_features[level]]
    
    def print_kf_pose_info(self):
        max_rot = torch.max(torch.linalg.norm(self.rotation_corrections, dim=1))
        max_tran = torch.max(torch.linalg.norm(self.translation_corrections.squeeze(2), dim=1))
        logger.info(f"GridNet KF pose corrections: max_rot={math.degrees(max_rot):.3f}deg, max_tran={max_tran:.3f}m.")
        
    def print_feature_info(self):
        logger.warning("Feature info not implemented yet for GridASH.")
