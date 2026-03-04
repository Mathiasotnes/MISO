import argparse
import numpy as np
import json
from grid_opt.utils.utils_eval import compute_chamfer_metrics, sample_points_from_mesh
from grid_opt.datasets.submap_dataset import SubmapDataset
from grid_opt.models.neural_points_hash import NeuralPointsHash
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

    neural_points = NeuralPointsHash(cfg['model'], device=cfg['device'], dtype=torch.float32) 
    neural_points.to(cfg['device'])
    
    return cfg, neural_points, dataset

def mapping(cfg, neural_points:BaseNet, dataset:SubmapDataset):
    frame_start = 0  
    frame_end = dataset.num_kfs
    
    for kf_id in range(dataset.num_kfs):
        R, t = dataset.true_kf_pose_in_world(kf_id)
        neural_points.set_initial_kf_pose(kf_id, R, t, kf_key=f"KF{kf_id}")
        
    mapper = Mapper(
        model=neural_points,
        dataset=dataset,
        cfg=cfg,
        init_neural_points=True # Custom parameter for Neural Points
    )
    
    mapper.mapping(
        mapping_kfs=range(frame_start, frame_end),
        iterations=cfg['train']['epochs'],
        level_iterations=cfg['train']['max_epochs_in_level']
    )
    
def save_active_point_cloud(path, points_xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    o3d.io.write_point_cloud(path, pcd)
    

##############################################
# Main entry point
##############################################

def main_scannet():
    np.random.seed(55)
    torch.manual_seed(55)
    args = parser.parse_args()
    model_path = join(args.save_dir, 'neural_points.pth')
    mesh_path = join(args.save_dir, 'neural_points_mesh.ply')
    ptc_path = join(args.save_dir, 'neural_points_points.ply')
    cfg, neural_points, dataset = initialize_scannet(args)
    
    mapping(cfg, neural_points, dataset)

    # Evaluate
    neural_points.eval()
    neural_points.print_active_info()
    pts = neural_points.points[neural_points.active].detach().cpu().numpy()
    save_active_point_cloud(ptc_path, pts)
    torch.save(neural_points, model_path)
    mesh = utils_sdf.save_mesh(neural_points, neural_points.bound, save_path=mesh_path)
    gt_mesh_path = join(args.scannet_root, f"scene{args.scene}/scene{args.scene}_vh_clean.ply")
    
    verts_pred = sample_points_from_mesh(mesh_path, mesh_sample_point=1000000)
    verts_trgt = sample_points_from_mesh(gt_mesh_path, mesh_sample_point=1000000)
    
    metrics_results = compute_chamfer_metrics(verts_pred, verts_trgt, threshold=0.05)
    print(json.dumps(metrics_results, indent=4))

if __name__ == "__main__":
    main_scannet()
