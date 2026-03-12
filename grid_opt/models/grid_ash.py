import numpy as np
import math
import torch
from .base_net import BaseNet
from .modules import MLPNet
from .grid_modules import *
import grid_opt.utils.utils_geometry as utils_geometry
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class GridASH(BaseNet):
    def __init__(self,
        cfg: dict, 
        device = 'cuda:0',
        dtype = torch.float32,
    ):
        super(GridASH, self).__init__(cfg, device, dtype)        
        self.init_grid(cfg)
        self.init_decoder(cfg)
        self.init_poses(cfg)

    def init_grid(self, cfg):
        self.n_levels = cfg['grid']['n_levels']
        self.second_order_grid_sample = 'second_order_grid_sample' in cfg['grid'] and cfg['grid']['second_order_grid_sample']
        self.base_cell_size = cfg['grid']['base_cell_size']
        self.scale_factor = cfg['grid']['per_level_scale']
        self.fdim = cfg['grid']['feature_dim']
        
        # TODO: Initialize empty ASH hash grid for each level.

        for level in range(self.n_levels):
            cell_size = self.base_cell_size / (self.scale_factor**level)
            self.cell_sizes.append(cell_size)
            grid = ...
            self.features.append(grid)
            
        self.ignore_level_ = np.zeros(self.n_levels).astype(bool)

    def init_decoder(self, cfg):
        self.decoder_hidden_dim = cfg['decoder']['hidden_dim']
        self.decoder_hidden_layers = cfg['decoder']['hidden_layers']
        self.decoder_out_dim = cfg['decoder']['out_dim']
        input_dim = self.n_levels * self.fdim
        
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
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"\n{'='*40}\n"
            f" GridASH\n"
            f"   * Levels              : {self.n_levels}\n"
            f"   * Base cell size      : {self.base_cell_size}\n"
            f"   * Per-level scale     : {self.scale_factor}\n"
            f"   * Hidden layers       : {self.decoder_hidden_dim}\n"
            f"   * Neurons/layer       : {self.decoder_hidden_layers}\n"
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
        logger.warning("Query_feature() not implemented yet for GridASH.")
    
    def forward(self, x):
        # TODO: Fix this once encoding is implemented
        f = self.encoding(x)
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
