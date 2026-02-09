import argparse
from os.path import join
import numpy as np
from grid_opt.configs import *
from grid_opt.utils.utils_sdf import *
from grid_opt.slam.mapper import Mapper
from grid_opt.models.grid_ngp import GridNGP
from grid_opt.datasets.submap_dataset import SubmapDataset
import grid_opt.utils.utils_sdf as utils_sdf
import grid_opt.utils.utils_scannet as utils_scannet
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


def create_configs_scannet(args, dataset: SubmapDataset):
    cfg = load_config(args.config, args.default_config)
    # Model settings
    scene = utils_scannet.scannet_scenes()[args.scene]
    cfg['model']['grid']['bound'] = scene.bound
    cfg['model']['pose']['num_poses'] = dataset.num_kfs
    # Mapping settings
    cfg['mapping']['verbose'] = True
    # System setting
    cfg['system']['log_dir'] = join(args.save_dir, "system")
    cfg['train']['log_dir'] = join(args.save_dir, "train")
    return cfg
    

def initialize_scannet(args):
    dataset = utils_scannet.create_scannet_dataset(args.scannet_root, args.scene, n_rays=200, frame_downsample=1)
    cfg = load_config(args.config, args.default_config)
    cfg = create_configs_scannet(args, dataset)
    # Disable incremental tracking, mapping, and vis
    cfg['tracking']['disable'] = True
    cfg['tracking']['verbose'] = False
    cfg['mapping']['disable'] = True
    cfg['mapping']['verbose'] = False
    cfg['visualizer']['enable'] = False
    hash_grid = GridNGP(cfg['model'], device=cfg['device'], dtype=torch.float32) 
    hash_grid.to(cfg['device'])
    return cfg, hash_grid, dataset

def submap_mapping(cfg, grid:BaseNet, dataset:SubmapDataset):
    cfg['mapping']['verbose'] = True
    cfg['mapping']['disable'] = False
    frame_start = 0  
    frame_end = dataset.num_kfs
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

def save_submap(grid_atlas:GridAtlas, submap_id:int, save_dir=None, visualize=True, postfix=''):
    submap = grid_atlas.get_submap(submap_id)
    R, t = grid_atlas.updated_submap_pose(submap_id)
    T = utils_geometry.pose_matrix(R, t)
    mesh_path = None
    if save_dir is not None:
        mesh_path = join(save_dir, f'submap_{submap_id}.ply')
    mesh = utils_sdf.save_mesh(submap, submap.bound, transform=T, save_path=mesh_path)
    if visualize:
        o3d.visualization.draw_geometries([mesh], window_name=f"Submap {submap_id} {postfix}")

def visualize_submap_split(args, grid_atlas:GridAtlas):
    submap_obbs = []
    for i in range(grid_atlas.num_submaps):
        obb = grid_atlas.submap_obb_in_world(i)
        submap_obbs.append(obb)
    
    gt_mesh_path = join(args.scannet_root, f"scene{args.scene}/scene{args.scene}_vh_clean_2.ply")
    gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
    o3d.visualization.draw_geometries(
        submap_obbs + [gt_mesh],
        window_name=f"Submaps and GT Mesh for {args.scene}"
    )

def main_scannet():
    np.random.seed(55)
    torch.manual_seed(55)
    args = parser.parse_args()
    model_path = join(args.save_dir, 'hash_grid.pth')
    cfg, hash_grid, dataset = initialize_scannet(args)
    
    submap_mapping(cfg, hash_grid, dataset, 0)
    
    # Visualize
    # save_submap(hash_grid, 0, save_dir=join(args.save_dir, 'submaps'), visualize=True, postfix='Fine Level')
    torch.save(hash_grid, model_path)
    

if __name__ == "__main__":
    main_scannet()
