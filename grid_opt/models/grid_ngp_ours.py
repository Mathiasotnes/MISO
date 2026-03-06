import os
import numpy as np
import math
import torch
import torch.nn as nn
from .collision_tracker import CollisionTracker
from .occupancy_grid import OccupancyGrid
from .saliency_grid import SaliencyGrid
from .base_net import BaseNet
from .grid_modules import *
import grid_opt.utils.utils_geometry as utils_geometry
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def spatial_hash(coords_int: torch.Tensor, T: int) -> torch.Tensor:
    x, y, z = coords_int[:, 0], coords_int[:, 1], coords_int[:, 2]
    MASK = 0xFFFFFFFF # Cast to uint32 range explicitly to mimic TCNN behavior
    h = (x ^ (y * 2_654_435_761) ^ (z * 805_459_861)) & MASK
    return (h % T).long()

def normalize_coordinates(x: torch.Tensor, bound: torch.Tensor) -> torch.Tensor:
    """
    Normalize world coordinates to [0, 1]^3.
    tcnn expects inputs in [0, 1], NOT [-1, 1] like utils.normalize_coordinates provides.

    Args:
        x:     (N, 3) tensor of world coordinates.
        bound: (3, 2) tensor of [[xmin,xmax],[ymin,ymax],[zmin,zmax]].
    Returns:
        (N, 3) tensor with values in [0, 1].
    """
    lo = bound[:, 0]  # (3,)
    hi = bound[:, 1]  # (3,)
    return (x - lo) / (hi - lo)


class MultiResHashEncoding(nn.Module):
    """
    Multiresolution Hash Encoding as described in:
      "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"

    For each of L levels we maintain a hash table of size T, where each
    entry is an F-dimensional trainable feature vector. Given a 3D input
    coordinate (already normalised to [0,1]^3) we:
      1. Scale the coordinate to that level's grid resolution.
      2. Find the 8 surrounding integer corners.
      3. Map every corner to a hash-table index via the spatial hash function.
      4. Look up the F-dim feature vector for every corner.
      5. Trilinearly interpolate the 8 vectors.
      6. Concatenate the L interpolated vectors → output of size L*F.
    """
    def __init__(
        self,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 15,
        base_resolution: int = 16,
        per_level_scale: float = 1.26,
    ):
        super().__init__()
        self.pi = [1, 2_654_435_761, 805_459_861]
        self.n_levels = n_levels
        self.F = n_features_per_level
        self.T = 2 ** log2_hashmap_size
        self.N_min = base_resolution
        self.b = per_level_scale

        self.n_output_dims = n_levels * n_features_per_level

        ### Hash table initialized with U(-1e-4, 1e-4) as recommended in the paper.
        self.hash_table = nn.Parameter(
            torch.empty(n_levels * self.T, self.F).uniform_(-1e-4, 1e-4)
        )

        ### Per-level grid resolutions
        resolutions = [
            math.floor(self.N_min * (self.b ** level))
            for level in range(n_levels)
        ]
        self.register_buffer("resolutions", torch.tensor(resolutions, dtype=torch.int32))

        ### Corner offsets for trilinear interpolation
        # 8 corners of the unit voxel: (0,0,0)…(1,1,1)
        offsets = torch.tensor([[i, j, k] for i in range(2) for j in range(2) for k in range(2)], dtype=torch.int32) # (8, 3)
        self.register_buffer("corner_offsets", offsets)

    def hash(self, coords_int: torch.Tensor) -> torch.Tensor:
        """ Spatial hash of integer grid coordinates. It's the same as spatial_hash(), but uses the class's T and pi values.
        Args:
            coords_int: (N, 3) long tensor of integer grid coordinates.
        Returns:
            (N,) long tensor of hash-table indices in [0, T).
        """
        x, y, z = coords_int[:, 0], coords_int[:, 1], coords_int[:, 2]
        MASK = 0xFFFFFFFF # Cast to uint32 range explicitly to mimic TCNN behavior
        h = (x ^ (y * self.pi[1]) ^ (z * self.pi[2])) & MASK
        return (h % self.T).long()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 3) float tensor of normalised coordinates in [0, 1]^3.
        Returns:
            (N, L*F) encoded feature tensor.
        """
        N = x.shape[0]
        level_features = []

        for level_idx in range(self.n_levels):
            N_l = self.resolutions[level_idx].item()   # scalar grid resolution

            ### Scale to level resolution
            x_scaled = x * N_l                         # (N, 3)

            ### Floor / ceil corners and interpolation weights
            x_floor = torch.floor(x_scaled).long()     # (N, 3)
            w = x_scaled - x_floor.float()             # (N, 3)  fractional offset

            ### Hash all 8 corners, look up features
            # corner_offsets: (8, 3) → broadcast to (N, 8, 3)
            corners = x_floor.unsqueeze(1) + self.corner_offsets.unsqueeze(0) # corners: (N, 8, 3)
            corners_flat = corners.reshape(N * 8, 3) # (N*8, 3)
            
            # Local index in [0, T), then offset into flat table for this level
            local_idx = spatial_hash(corners_flat, self.T) # (N*8,)
            global_idx = local_idx + level_idx * self.T    # (N*8,)

            feats = self.hash_table[global_idx] # (N*8, F)
            feats = feats.reshape(N, 8, self.F) # (N, 8, F)

            ### Trilinear interpolation
            # Decompose into per-axis weights for the 8 corners:
            # corner order: (i,j,k) for i,j,k ∈ {0,1}  (matches corner_offsets)
            wx0, wx1 = 1.0 - w[:, 0], w[:, 0] # (N,)
            wy0, wy1 = 1.0 - w[:, 1], w[:, 1]
            wz0, wz1 = 1.0 - w[:, 2], w[:, 2]

            # weight for each of the 8 corners → (N, 8, 1)
            weights = torch.stack([
                wx0 * wy0 * wz0,    # (0, 0, 0)
                wx0 * wy0 * wz1,    # (0, 0, 1)
                wx0 * wy1 * wz0,    # (0, 1, 0)
                wx0 * wy1 * wz1,    # (0, 1, 1)
                wx1 * wy0 * wz0,    # (1, 0, 0)
                wx1 * wy0 * wz1,    # (1, 0, 1)
                wx1 * wy1 * wz0,    # (1, 1, 0)
                wx1 * wy1 * wz1,    # (1, 1, 1)
            ], dim=1).unsqueeze(-1) # (N, 8, 1)

            interpolated = (weights * feats).sum(dim=1) # (N, F)
            level_features.append(interpolated)

        return torch.cat(level_features, dim=-1) # (N, L*F) Concatenate across levels

class GridNGPOurs(BaseNet):
    """
    An implementation similar to grid_ngp, but implements our own non-optimized version
    of the hash grid encoding instead of using tiny-cuda-nn.
    """
    def __init__(self,
        cfg: dict, 
        device = 'cuda:0',
        dtype = torch.float32,
        track_collisions = False,
        track_occupancy = False,
        track_saliency = False,
        n_levels = 16,
        n_features_per_level = 2,
        log2_hashmap_size = 15,
        base_resolution = 16,
        per_level_scale = 1.26,
        n_hidden_layers = 2,
        n_neurons = 64,
        n_output_dims = 1
    ):
        super(GridNGPOurs, self).__init__(cfg, device, dtype)    
        self.device = device
        self.dtype = dtype
        self.track_collisions = track_collisions
        self.track_occupancy = track_occupancy
        self.track_saliency = track_saliency
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.per_level_scale = per_level_scale
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.n_output_dims = n_output_dims
        self.num_levels = 1 # Hack to make it compatible with old MISO trainer.py
        
        if self.track_occupancy:
            self.occupancy_grid = OccupancyGrid(device=self.device, bound=self.bound)
            
        if self.track_saliency:
            self.saliency_grid = SaliencyGrid(res=64, device=self.device, bound=self.bound)
        
        self.init_ngp(cfg)
        self.init_poses(cfg)
        self.print_summary()
        
    def init_ngp(self, cfg):
        
        self.encoding = MultiResHashEncoding(
            n_levels             = self.n_levels,
            n_features_per_level = self.n_features_per_level,
            log2_hashmap_size    = self.log2_hashmap_size,
            base_resolution      = self.base_resolution,
            per_level_scale      = self.per_level_scale,
        ).to(self.device)
        
        ########################################
        # Decoder network (MLP)
        ########################################
        
        decoder = []
        input_dim = self.encoding.n_output_dims

        for i in range(self.n_hidden_layers):
            layer = nn.Linear(input_dim if i == 0 else self.n_neurons, self.n_neurons)
            nn.init.xavier_uniform_(layer.weight) # InstantNGP paper uses Glorot (Xavier) initialization
            decoder.append(layer)
            decoder.append(nn.ReLU())

        # Final output layer
        if self.n_hidden_layers == 0:
            final_layer = nn.Linear(input_dim, self.n_output_dims)
        else:
            final_layer = nn.Linear(self.n_neurons, self.n_output_dims)
        nn.init.xavier_uniform_(final_layer.weight)
        decoder.append(final_layer)

        self.decoder = nn.Sequential(*decoder)

        ########################################
        # Complete model
        ########################################
        
        if self.track_collisions:
            self.tracker = CollisionTracker(
                encoding=self.encoding, 
                scene_bound=self.bound, 
                device=self.device
            )
            self.tracker.register_hooks()
        
    def init_occupancy_grid(self, cfg):
        """ The loss function updates the occupancy grid when the track_occupancy flag is on. This is
        because we have access to the label here, and we want to add occupancy whenever a sample with low SDF is observed. """
        if self.track_occupancy:
            self.occupancy_grid = OccupancyGrid(device=self.device, bound=self.bound)

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
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"\n{'='*40}\n"
            f" GridNGPOurs\n"
            f"   * Levels              : {self.n_levels}\n"
            f"   * Features/level      : {self.n_features_per_level}\n"
            f"   * Hash table size     : 2^{self.log2_hashmap_size} = {2**self.log2_hashmap_size:,}\n"
            f"   * Base resolution     : {self.base_resolution}\n"
            f"   * Per-level scale     : {self.per_level_scale}\n"
            f"   * Hidden layers       : {self.n_hidden_layers}\n"
            f"   * Neurons/layer       : {self.n_neurons}\n"
            f"   * Trainable params    : {total:,}\n"
            f"{'='*40}"
        )
    
    def lock_pose(self):
        self.rotation_corrections.requires_grad_(False)
        self.translation_corrections.requires_grad_(False)
        self.lock_all_pose_indices()
     
    def unlock_pose(self):
        self.rotation_corrections.requires_grad_(True)
        self.translation_corrections.requires_grad_(True)
        self.unlock_all_pose_indices()
        
    def lock_feature(self):
        for param in self.encoding.parameters():
            param.requires_grad = False
    
    def unlock_feature(self):
        for param in self.encoding.parameters():
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
    
    def query_feature(self, x):
        x = normalize_coordinates(x, self.bound)
        return self.encoding(x)
    
    # This is a test where I multiplied the occupancy mask with the interpolated features inspired by the 
    # saliency map presented in "HollowNeRF". The idea is to avoid all contributions from "unimportant" voxels
    # to the features.
    # def forward(self, x):
    #     x_norm = normalize_coordinates(x, self.bound)
    #     if self.track_occupancy and self.occupancy_grid.grid.any():
    #         with torch.no_grad():
    #             occupied = self.occupancy_grid.get_occupancy(x)
    #         out = torch.ones(x.shape[0], self.n_output_dims, device=self.device, dtype=self.dtype)
    #         if occupied.any():
    #             out[occupied] = self.model(x_norm[occupied])
    #     else:
    #         out = self.model(x_norm)
    #     return out

    def forward(self, x):
        x_norm = normalize_coordinates(x, self.bound)
        f = self.encoding(x_norm)
        if self.track_saliency:
            p = self.saliency_grid(x)
            f = p * f
        return self.decoder(f)
    
    def params_at_level(self, level):
        # FIXME: right now this always return the full set of params!
        return list(self.encoding.parameters())
    
    def print_kf_pose_info(self):
        max_rot = torch.max(torch.linalg.norm(self.rotation_corrections, dim=1))
        max_tran = torch.max(torch.linalg.norm(self.translation_corrections.squeeze(2), dim=1))
        logger.info(f"GridNet KF pose corrections: max_rot={math.degrees(max_rot):.3f}deg, max_tran={max_tran:.3f}m.")
        
    def print_feature_info(self):
        logger.warning("Feature info not implemented yet for GridNGPOurs.")
