import argparse
import numpy as np
from grid_opt.datasets.submap_dataset import SubmapDataset
from grid_opt.slam.mapper import Mapper
from grid_opt.utils.utils_sdf import *
from grid_opt.configs import *
from os.path import join
import grid_opt.utils.utils_scannet as utils_scannet
import grid_opt.utils.utils_sdf as utils_sdf
import open3d as o3d
import logging
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to config file.', default='./configs/rgbd/scannet.yaml')
parser.add_argument('--default_config', type=str, help='Path to config file.', default='./configs/base.yaml')
parser.add_argument('--save_dir', type=str, default='./results/mapping')
parser.add_argument('--pose_init', type=str, default='gt')  # reg_icp OR kiss_icp OR gt
parser.add_argument('--scannet_root', type=str, default='../../data/ScanNet/scans')
parser.add_argument('--scene', type=str, default='0000_00')


##############################################
# Helpers
##############################################

def save_submap(grid:BaseNet, submap_id:int, save_dir=None, visualize=False):
    mesh_path = None
    if save_dir is not None:
        mesh_path = join(save_dir, f'pred_mesh.ply')
    mesh = utils_sdf.save_mesh(grid, grid.bound, save_path=mesh_path)
    if visualize:
        o3d.visualization.draw_geometries([mesh], window_name=f"Predicted Mesh")
        

##############################################
# Script Implementation
##############################################

def create_configs_scannet(args, dataset: SubmapDataset):
    cfg = load_config(args.config, args.default_config)
    
    # Model settings
    scene = utils_scannet.scannet_scenes()[args.scene]
    cfg['model']['grid']['bound'] = scene.bound
    cfg['model']['pose']['num_poses'] = dataset.num_kfs
    
    # System setting
    cfg['system']['log_dir'] = join(args.save_dir, "system")
    cfg['train']['log_dir'] = join(args.save_dir, "train")
    
    return cfg

def initialize_scannet(args):
    cfg = load_config(args.config, args.default_config)
    dataset = utils_scannet.create_scannet_dataset(args.scannet_root, args.scene, n_rays=cfg['sample']['n_rays'], frame_downsample=1)
    cfg = load_config(args.config, args.default_config)
    cfg = create_configs_scannet(args, dataset)

    grid = GridNet(cfg['model'], device=cfg['device'], dtype=torch.float32) 
    grid.to(cfg['device'])
    
    return cfg, grid, dataset

def mapping(cfg, grid:BaseNet, dataset:SubmapDataset):
    frame_start = 0  
    frame_end = dataset.num_kfs
    
    for kf_id in range(dataset.num_kfs):
        R, t = dataset.true_kf_pose_in_world(kf_id)
        grid.set_initial_kf_pose(kf_id, R, t, kf_key=f"KF{kf_id}")
        
    mapper = Mapper(
        model=grid,
        dataset=dataset,
        cfg=cfg
    )
    
    mapper.mapping(
        mapping_kfs=range(frame_start, frame_end),
        iterations=cfg['train']['epochs'],
        level_iterations=cfg['train']['max_epochs_in_level']
    )

def main_scannet():
    np.random.seed(55)
    torch.manual_seed(55)
    args = parser.parse_args()
    model_path = join(args.save_dir, 'grid.pth')
    cfg, grid, dataset = initialize_scannet(args)
    
    mapping(cfg, grid, dataset)
    
    # Visualize
    # save_submap(grid, 0, save_dir=join(args.save_dir, 'submaps'), visualize=True, postfix='Fine Level')
    torch.save(grid, model_path)

if __name__ == "__main__":
    main_scannet()
