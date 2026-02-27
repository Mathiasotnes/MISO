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
    def __init__(self, encoding, n_feats=2, n_levels=16, hashmap_size=2**15, base_res=16, per_level_scale=1.26):
        self.n_feats    = n_feats
        self.L          = n_levels
        self.T          = hashmap_size
        self.N_min      = base_res
        self.b          = per_level_scale
        
        # Importance: G(l, v) and Voxel-to-Bin Mapping: V(l, v) -> i
        self.voxel_grads    = []
        self.voxel_to_bin   = [] 
        
        for l in range(self.L):
            res = math.floor(self.N_min * (self.b ** l))
            num_verts = (res + 1) ** 3
            self.voxel_grads.append(torch.zeros(num_verts, dtype=torch.float32, device='cuda'))
            self.voxel_to_bin.append(torch.full((num_verts,), -1, dtype=torch.long, device='cuda'))

        # Try to probe memory offset between layers to calculate stats.
        # I'm not partitioning layers during tracking in case this offset
        # detection is not perfectly aligned with the cuda implementation.
        self.level_offsets = self._detect_offsets(encoding)
        self._verify_offsets(encoding)
        self.offsets_3d = torch.stack(torch.meshgrid([torch.tensor([0, 1])] * 3, indexing='ij')).reshape(3, -1).t().to('cuda')

    def _detect_offsets(self, encoding):
        """ Probes the TCNN parameter vector to find the physical start of each layer. """
        offsets = []
        zero_coord = torch.zeros((1, 3), device='cuda', requires_grad=True)
        for l in range(self.L):
            if encoding.params.grad is not None: encoding.params.grad.zero_()
            # We isolate the backward pass to a specific feature slice
            encoding(zero_coord)[:, l*self.n_feats : (l+1)*self.n_feats].sum().backward()
            
            grad_indices = torch.where(encoding.params.grad != 0)[0]
            if grad_indices.numel() > 0:
                offsets.append(grad_indices.min().item())
            else:
                raise RuntimeError(f"Probe failed at level {l}. Is the model initialized?")
        encoding.params.grad = None
        return offsets

    def _verify_offsets(self, encoding):
        """ 
        Self-Correction Test: Ensures that detected offsets 
        consistently yield indices within [0, T-1].
        """
        print("\n--- TCNN Memory Layout Verification Report ---")
        total_params = encoding.params.shape[0]
        
        for l in range(self.L):
            start = self.level_offsets[l]
            end = self.level_offsets[l+1] if l < self.L-1 else total_params
            
            # Theoretical bucket size based on your config
            expected_size = self.T * self.n_feats
            actual_size = end - start
            
            # Check for alignment padding (TCNN often aligns to 16 or 32 bytes)
            padding = actual_size - expected_size
            
            status = "PASS" if actual_size >= expected_size else "FAIL"
            print(f"L{l+1:02} | Start: {start:10} | Gap: {actual_size:8} | Padding: {padding:4} | {status}")
            
            if status == "FAIL":
                raise RuntimeError(f"Level {l} offset detection is smaller than hashmap size!")
        print("---------------------------------------\n")

    def _get_tcnn_indices(self, encoding, x):
        """ Returns all indices touched by the batch, across all levels. """
        x_probe = x.detach().clone().requires_grad_(True)
        orig_grad = encoding.params.grad.clone() if encoding.params.grad is not None else None
        encoding.params.grad = None
        
        # Calculate gradients for the batch so that we can see which indices are touched
        encoding(x_probe).sum().backward()
        
        # This is a single 1D tensor of index touched by this batch across all layers:
        touched_indices = torch.where(encoding.params.grad != 0)[0]
        
        # Restore original gradients to make the probe non-intrusive to training:
        encoding.params.grad = orig_grad
        return touched_indices

    @torch.no_grad()
    def track_step(self, coords_world, bound, loss_vec, encoding):
        """ Record G(l, v) and the voxel-to-bin mapping. """
        x = (coords_world.detach() - bound[:, 0]) / (bound[:, 1] - bound[:, 0])
        valid = (x >= 0).all(dim=-1) & (x < 1).all(dim=-1)
        if not valid.any(): return
        x_valid = torch.clamp(x[valid], 0.0, 1.0 - 1e-6)
        g = loss_vec.detach().reshape(-1)[valid]
        
        with torch.enable_grad():
            touched_indices = self._get_tcnn_indices(encoding, x_valid)

        for l in range(self.L):
            res = math.floor(self.N_min * (self.b ** l))
            stride = res + 1
            
            # Identify Voxels v and map samples to current level's voxel grid
            base_v = torch.floor(x_valid * res).long()
            all_v = (base_v.unsqueeze(0) + self.offsets_3d.unsqueeze(1)).reshape(-1, 3)
            v_idx = all_v[:, 0] + stride * (all_v[:, 1] + stride * all_v[:, 2])
            
            # Accumulate G(l, v)
            self.voxel_grads[l].index_add_(0, v_idx, g.repeat_interleave(8))

            # Store the physical memory address for these voxels
            start = self.level_offsets[l]
            end = self.level_offsets[l+1] if l < self.L-1 else encoding.params.shape[0]
            
            level_indices = touched_indices[(touched_indices >= start) & (touched_indices < end)]
            if level_indices.numel() > 0:
                # Store the absolute global index
                self.voxel_to_bin[l][v_idx] = level_indices.min()

    def _compute_final_stats(self):
        """ Implementation of the formal metrics: C_eff, Total_Bin_Grad, and R_dom. """
        final_eff = torch.zeros((self.L, self.T), device='cuda')
        final_max = torch.zeros((self.L, self.T), device='cuda')
        final_sum = torch.zeros((self.L, self.T), device='cuda')

        for l in range(self.L):
            active_mask = self.voxel_to_bin[l] != -1
            if not active_mask.any(): continue
            
            v_indices = torch.where(active_mask)[0]
            v_grads = self.voxel_grads[l][v_indices]
            
            # Convert Global Memory Address -> Local Bin ID (0 to T-1)
            # Dividing by n_feats accounts for the fact that each bin has 2 features
            global_bins = self.voxel_to_bin[l][v_indices]
            local_bins = (global_bins - self.level_offsets[l]) // self.n_feats
            
            # Clamp to table size T just in case of detection epsilon
            local_bins = torch.clamp(local_bins, 0, self.T - 1)
            
            # C_eff: unique voxels per hash bin
            final_eff[l].index_add_(0, local_bins, torch.ones_like(local_bins, dtype=torch.float32))
            
            # G(l, v) totals
            final_sum[l].index_add_(0, local_bins, v_grads)
            final_max[l].index_reduce_(0, local_bins, v_grads, reduce='amax', include_self=False)

        return final_eff, final_max, final_sum

    def save(self, path):
        eff, mx, sm = self._compute_final_stats()
        torch.save({
            "eff_voxel_count": eff.cpu(),
            "max_voxel_grad": mx.cpu(),
            "total_bin_grad": sm.cpu(),
            "config": {"L": self.L, "T": self.T, "b": self.b, "N_min": self.N_min}
        }, path)

    def print_collision_summary(self):
        eff, mx, sm = self._compute_final_stats()
        print("\n" + "="*125)
        print(f"{'MHE SPATIAL COLLISION DENSITY & DOMINANCE ANALYSIS':^125}")
        print("="*125)
        header = f"{'L':<3} | {'Res':<5} | {'Bins (Occ)':<10} | {'C_eff (Min/Avg/Max)':<25} | {'R_dom (Min/Avg/Max)':<25}"
        print(header)
        print("-" * len(header))

        for l in range(self.L):
            occ = eff[l] > 0
            num_occ = occ.sum().item()
            if num_occ > 0:
                c_eff_occ = eff[l][occ]
                c_eff_str = f"{c_eff_occ.min():.0f} / {c_eff_occ.mean():.2f} / {c_eff_occ.max():.0f}"
                
                # R_dom calculation: Max/Sum
                r_dom = mx[l][occ] / sm[l][occ].clamp(min=1e-6)
                r_dom_str = f"{r_dom.min():.4f} / {r_dom.mean():.4f} / {r_dom.max():.4f}"
                
                res = math.floor(self.N_min * (self.b ** l))
                print(f"{l+1:<3} | {res:<5} | {num_occ:<10,} | {c_eff_str:<25} | {r_dom_str:<25}")

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
                encoding=self.encoding,
                n_feats=config_encoding["n_features_per_level"],
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
