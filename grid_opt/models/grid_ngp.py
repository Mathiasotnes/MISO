import os
import numpy as np
import math
import torch
import torch.nn as nn
import grid_opt.utils.utils as utils
from .base_net import BaseNet
from .grid_modules import *
import grid_opt.utils.utils_geometry as utils_geometry
import logging

import tinycudann as tcnn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CollisionTracker:
    """ Utility class to track collision statistics during training of the GridNGP. """
    def __init__(self, n_levels=16, hashmap_size=2**15, base_res=16, per_level_scale=1.26):
        self.L = n_levels
        self.T = hashmap_size
        self.N_min = base_res
        self.b = per_level_scale
        self.PI = torch.tensor([1, 2654435761, 805459861], device='cuda', dtype=torch.long)
        
        # 1. Effective Collisions (Count of unique vertices per bin)
        self.eff_voxel_count = torch.zeros((self.L, self.T), dtype=torch.long, device='cuda')
        
        # 2. Total Gradient Magnitude (Informational Load)
        self.grad_sum = torch.zeros((self.L, self.T), dtype=torch.float32, device='cuda')
        
        # 3. For Variance (Conflict)
        self.grad_sq_sum = torch.zeros((self.L, self.T), dtype=torch.float32, device='cuda')
        self.sample_count = torch.zeros((self.L, self.T), dtype=torch.float32, device='cuda')

        # Bitfields to track "Seen" voxels per level (to ensure C_eff only counts unique ones)
        # For N_max=512, each level needs (512+1)^3 bits ~= 16MB per level.
        self.voxel_bitfields = []
        for l in range(self.L):
            res = math.floor(self.N_min * (self.b ** l))
            num_bits = (res + 1) ** 3
            # We use a byte-tensor as a simple bitfield for GPU speed
            self.voxel_bitfields.append(torch.zeros(num_bits, dtype=torch.uint8, device='cuda'))

        self.offsets = torch.stack(torch.meshgrid([torch.tensor([0, 1])] * 3, indexing='ij')).reshape(3, -1).t().to('cuda')

    def get_hash(self, coords: torch.Tensor) -> torch.Tensor:
        # TODO: Make sure this is consistent with tcnn implementation. We might need to change the computer precision to match.
        # coords shape: (8 * Batch, 3)
        h = (coords[:, 0] * self.PI[0]) ^ (coords[:, 1] * self.PI[1]) ^ (coords[:, 2] * self.PI[2])
        return h % self.T
    
    @torch.no_grad()
    def track_step(self, coords_world: torch.Tensor, bound: torch.Tensor, loss_vec: torch.Tensor):
        x = (coords_world - bound[:, 0]) / (bound[:, 1] - bound[:, 0])
        g = loss_vec.detach().reshape(-1) # (Batch,)
        
        for l in range(self.L):
            res = math.floor(self.N_min * (self.b ** l))
            base_v = torch.floor(x * res).long().clamp(min=0, max=res-1) # (Batch, 3)
            
            # 8 corners per sample
            all_v = (base_v.unsqueeze(0) + self.offsets.unsqueeze(1)).reshape(-1, 3) # (8*B, 3)
            indices = self.get_hash(all_v) # (8*B)
            
            # --- C_grad: Total Gradient Magnitude ---
            g_rep = g.repeat_interleave(8)
            self.grad_sum[l].index_add_(0, indices, g_rep)
            self.grad_sq_sum[l].index_add_(0, indices, g_rep**2)
            self.sample_count[l].index_add_(0, indices, torch.ones_like(g_rep))

            # --- C_eff: Unique Voxel Counting ---
            # Map 3D coords to 1D index: x + N(y + N*z)
            v_idx = all_v[:,0] + (res+1)*(all_v[:,1] + (res+1)*all_v[:,2])
            
            # Find which voxels in this batch are new to this level
            already_seen = self.voxel_bitfields[l][v_idx] 
            is_new = (already_seen == 0)
            
            if is_new.any():
                new_v_indices = indices[is_new]
                # Increment C_eff for every bin that just received a NEW unique voxel
                self.eff_voxel_count[l].index_add_(0, new_v_indices, torch.ones_like(new_v_indices, dtype=torch.long))
                # Mark as seen
                self.voxel_bitfields[l][v_idx[is_new]] = 1

    def save(self, path):
        data = {
            "eff_bin_mask": self.eff_bin_mask.cpu(),
            "grad_sum": self.grad_sum.cpu(),
            "grad_sq_sum": self.grad_sq_sum.cpu(),
            "sample_count": self.sample_count.cpu(),
            "config": {"L": self.L, "T": self.T, "b": self.b, "N_min": self.N_min}
        }
        torch.save(data, path)
        
    def print_collision_summary(self):
        """
        Summarizes the collision landscape based on the spatial collision density analysis.
        Calculates C_eff (Effective Collisions) and C_grad (Informational Load).
        """
        print("\n" + "="*105)
        print(f"{'MHE SPATIAL COLLISION DENSITY ANALYSIS SUMMARY':^105}")
        print("="*105)
        # Bins (Occ): Number of hash bins with C_eff > 0
        # Avg C_eff: Mean number of unique active voxels per occupied bin
        # Max C_eff: Worst-case aliasing at this level
        # Total C_grad: Cumulative gradient magnitude (Informational Load)
        # Avg Conflict: Mean variance in bins with C_eff > 1
        header = f"{'L':<3} | {'Res':<5} | {'Bins (Occ)':<10} | {'Avg C_eff':<10} | {'Max C_eff':<10} | {'Total C_grad':<14} | {'Avg Conflict'}"
        print(header)
        print("-" * len(header))

        for l in range(self.L):
            N_l = int(math.floor(self.N_min * (self.b ** l)))
            
            # 1. Effective Collision Stats (C_eff)
            # Find indices that are active at this level
            occ_mask = self.eff_voxel_count[l] > 0
            occupied_bins = occ_mask.sum().item()
            
            if occupied_bins > 0:
                c_eff_occ = self.eff_voxel_count[l][occ_mask].float()
                avg_c_eff = c_eff_occ.mean().item()
                max_c_eff = c_eff_occ.max().item()
            else:
                avg_c_eff, max_c_eff = 0.0, 0
            
            # 2. Informational Load (C_grad)
            total_c_grad = self.grad_sum[l].sum().item()
            
            # 3. Conflict Analysis (Variance)
            # Var = E[X^2] - (E[X])^2
            n = self.sample_count[l]
            # We only care about conflict in bins that have samples and aliasing
            conflict_mask = (n > 1) & (self.eff_voxel_count[l] > 1)
            
            if conflict_mask.any():
                n_m = n[conflict_mask]
                mean_g = self.grad_sum[l][conflict_mask] / n_m
                var_g = (self.grad_sq_sum[l][conflict_mask] / n_m) - (mean_g ** 2)
                avg_conflict = var_g.mean().item()
            else:
                avg_conflict = 0.0

            print(f"{l+1:<3} | {N_l:<5} | {occupied_bins:<10,} | {avg_c_eff:<10.2f} | {max_c_eff:<10} | {total_c_grad:<14.2e} | {avg_conflict:.6f}")

        # Global Metadata
        total_params = self.L * self.T
        active_bins = (self.eff_voxel_count > 0).sum().item()
        print("="*105)
        print(f"Overall Hash Table Utilization: {active_bins:,} / {total_params:,} ({active_bins/total_params:.2%})")
        print("="*105 + "\n")

class OccupancyGrid:
    """ A lightweight/simple occupancy grid. """
    def __init__(self, bound, res=0.1, device='cuda:0'):
        self.bound = bound
        self.res = res
        self.device = device
        self.Nx = math.ceil((self.bound[0,1] - self.bound[0,0]) / self.res)
        self.Ny = math.ceil((self.bound[1,1] - self.bound[1,0]) / self.res)
        self.Nz = math.ceil((self.bound[2,1] - self.bound[2,0]) / self.res)
        self.N = self.Nx * self.Ny * self.Nz
        self.grid = torch.zeros(self.N, dtype=torch.bool, device=device)
    
    def world_to_grid(self, x: torch.Tensor) -> torch.Tensor:
        """ Convert world coordinates to grid coordinates.
        Args:
            x: (N, 3) tensor of world coordinates
        Returns:
            (N, 3) tensor of grid coordinates
        """
        # in-bounds mask in world space
        mask = (
            (x[:, 0] >= self.bound[0, 0]) & (x[:, 0] < self.bound[0, 1]) &
            (x[:, 1] >= self.bound[1, 0]) & (x[:, 1] < self.bound[1, 1]) &
            (x[:, 2] >= self.bound[2, 0]) & (x[:, 2] < self.bound[2, 1])
        )

        if not mask.any():
            return None, mask

        g = (x[mask] - self.bound[:, 0]) / self.res
        g = torch.floor(g).long()
        return g, mask
    
    def grid_to_index(self, g: torch.Tensor) -> torch.Tensor:
        """ Convert grid coordinates to grid index. """
        return g[:, 0] + self.Nx * (g[:, 1] + self.Ny * g[:, 2])
    
    @torch.no_grad()
    def update(self, x: torch.Tensor, sdf: torch.Tensor, tau: float = 0.2):
        """ Update the occupancy grid based on the input world coordinates and their corresponding SDF values.

        Args:
            x (torch.Tensor):           (N,3) tensor of world coordinates corresponding to the SDF values.
            sdf (torch.Tensor):         (N,) tensor of SDF values corresponding to the input world coordinates.
            tau (float, optional):      Threshold SDF value for marking cell as occupied. Defaults to 0.1.
        """
        sdf = sdf.view(-1)
        occ = sdf < tau
        
        g = self.world_to_grid(x)
        g, inb = self.world_to_grid(x)
        
        if g is None:
            return
        
        valid = occ[inb]
        if not valid.any():
            return

        idx = self.grid_to_index(g)
        self.grid[idx[valid]] = True

    
    def get_occupancy(self, x: torch.Tensor) -> torch.Tensor:
        """ Get the occupancy status of the grid cells corresponding to the input world coordinates.
        Args:
            x: (N, 3) tensor of world coordinates
        Returns:
            (N,) tensor of occupancy status (True for occupied, False for free)
        """
        g, inb = self.world_to_grid(x)
        out = torch.zeros(x.shape[0], device=x.device, dtype=torch.bool)
        if g is None:
            return out
        idx = self.grid_to_index(g)
        out[inb] = self.grid[idx]
        return out

class GridNGP(BaseNet):
    """
    An implementation similar to grid_net, but with the regular grid
    replaced by a hash grid implemented using tiny-cuda-nn / instantNGP
    hash grids. It only contains a subset of the functionality of grid_net.
    """
    def __init__(self,
        cfg: dict, 
        device = 'cuda:0',
        dtype = torch.float32,
        track_collisions = False
    ):
        super(GridNGP, self).__init__(cfg, device, dtype)    
        self.device = device
        self.dtype = dtype
        self.track_collisions = track_collisions
        self.init_ngp(cfg)
        self.init_occupancy_grid(cfg)
        self.init_poses(cfg)
        
    def init_ngp(self, cfg):
        
        # TODO: Move config to another location.
        config_encoding = {
            "otype": "Grid",            # Component type.
            "type": "Hash",             # Type of backing storage of the
                                        # grids. Can be "Hash", "Tiled"
                                        # or "Dense".
            "n_levels": 16,             # Number of levels (resolutions)
            "n_features_per_level": 2,  # Dimensionality of feature vector
                                        # stored in each level's entries.
            "log2_hashmap_size": 15,    # If type is "Hash", is the base-2
                                        # logarithm of the number of elements
                                        # in each backing hash table.
            "base_resolution": 16,      # The resolution of the coarsest le-
                                        # vel is base_resolution^input_dims.
            "per_level_scale": 1.26,    # The geometric growth factor, i.e.
                                        # the factor by which the resolution
                                        # of each grid is larger (per axis)
                                        # than that of the preceding level.
            "interpolation": "Linear",  # How to interpolate nearby grid
                                        # lookups. Can be "Nearest", "Linear",
                                        # or "Smoothstep" (for smooth deri-
                                        # vatives).
            "n_input_dims": 3,          # Number of dimensions of input coordinates.
        }
        config_network = {
            "otype": "FullyFusedMLP",   # Component type.
            "activation": "ReLU",       # Activation function. Can be "ReLU",
            "output_activation": "None",# Activation function of the output layer.
            "n_output_dims": 1,         # Number of dimensions of output features. 1 for SDF prediction.
            "n_neurons": 64,            # Number of neurons in each hidden layer. (mus be 16, 32, 64 or 128)
            "n_hidden_layers": 2        # Number of hidden layers.
        }
        
        self.num_levels = 1 # Hack to make it compatible with trainer.py. I think we can make a much simpler trainer unless we still want
                            # to support coarse-to-fine curriculum learning (coordinate option).
        
        self.encoding = tcnn.Encoding(config_encoding["n_input_dims"], config_encoding)
        
        # Use this for an optimized fully fused MLP:
        # self.network = tcnn.Network(self.encoding.n_output_dims, config_network["n_output_dims"], config_network)
        
        # I'm using this for research purposes to access activation patterns:
        layers = []
        input_dim = self.encoding.n_output_dims
        hidden_dim = config_network["n_neurons"]

        for i in range(config_network["n_hidden_layers"]):
            layer = nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            nn.init.xavier_uniform_(layer.weight) # InstantNGP paper uses Glorot (Xavier) initialization
            layers.append(layer)
            layers.append(nn.ReLU())

        # Final output layer (Linear) 
        final_layer = nn.Linear(hidden_dim, config_network["n_output_dims"])
        nn.init.xavier_uniform_(final_layer.weight)
        layers.append(final_layer)

        self.network = nn.Sequential(*layers)
        
        self.model = torch.nn.Sequential(self.encoding, self.network)
        self.print_trainable_params()
        
        # Collision tracker
        if self.track_collisions:
            self.tracker = CollisionTracker(
                n_levels=config_encoding["n_levels"],
                hashmap_size=2**config_encoding["log2_hashmap_size"],
                base_res=config_encoding["base_resolution"],
                per_level_scale=config_encoding["per_level_scale"]
            )
        
    def init_occupancy_grid(self, cfg):
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
    
    def lock_pose(self):
        self.rotation_corrections.requires_grad_(False)
        self.translation_corrections.requires_grad_(False)
        self.lock_all_pose_indices()
     
    def unlock_pose(self):
        self.rotation_corrections.requires_grad_(True)
        self.translation_corrections.requires_grad_(True)
        self.unlock_all_pose_indices()
        
    def lock_feature(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unlock_feature(self):
        for param in self.model.parameters():
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
        x = utils.normalize_coordinates(x, self.bound)
        return self.encoding(x)
    
    def forward(self, x):
        x = utils.normalize_coordinates(x, self.bound)
        return self.model(x)
    
    def params_at_level(self, level):
        # FIXME: right now this always return the full set of params!
        return list(self.model.parameters())
    
    def print_kf_pose_info(self):
        max_rot = torch.max(torch.linalg.norm(self.rotation_corrections, dim=1))
        max_tran = torch.max(torch.linalg.norm(self.translation_corrections.squeeze(2), dim=1))
        logger.info(f"GridNet KF pose corrections: max_rot={math.degrees(max_rot):.3f}deg, max_tran={max_tran:.3f}m.")
        
    def print_feature_info(self):
        logger.warning("Feature info not implemented yet for GridNGP.")
