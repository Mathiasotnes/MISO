import argparse
import numpy as np
import json
import time
from grid_opt.models.grid_ash import GridASH
from grid_opt.utils.utils_eval import compute_chamfer_metrics, sample_points_from_mesh
from grid_opt.datasets.submap_dataset import SubmapDataset
from grid_opt.slam.mapper import Mapper
from grid_opt.utils.utils_sdf import *
from grid_opt.configs import *
from os.path import join
import grid_opt.utils.utils_scannet as utils_scannet
import grid_opt.utils.utils_sdf as utils_sdf
import logging
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to config file.', default='./configs/rgbd/ash.yaml')
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
    cfg = create_configs_scannet(args, dataset)

    ash_grid = GridASH(cfg['model'], device=cfg['device'], dtype=torch.float32)
    ash_grid.to(cfg['device'])
    
    return cfg, ash_grid, dataset

def mapping(cfg, ash_grid:BaseNet, dataset:SubmapDataset):
    frame_start = 0  
    frame_end = dataset.num_kfs
    mapper = Mapper(
        model=ash_grid,
        dataset=dataset,
        cfg=cfg
    )
    
    for kf_id in range(dataset.num_kfs):
        R, t = dataset.true_kf_pose_in_world(kf_id)
        ash_grid.set_initial_kf_pose(kf_id, R, t, kf_key=f"KF{kf_id}")
    
    mapper.mapping(
        mapping_kfs=range(frame_start, frame_end),
        iterations=cfg['train']['epochs'],
        level_iterations=cfg['train']['max_epochs_in_level']
    )
    
class GPUMemoryTracker:
    def __init__(self, device="cuda"):
        self.device = device

    def start(self):
        torch.cuda.reset_peak_memory_stats(self.device)

    def report(self, label=""):
        peak_allocated = torch.cuda.max_memory_allocated(self.device)
        peak_reserved  = torch.cuda.max_memory_reserved(self.device)
        tag = f"[{label}] " if label else ""
        print(f"{tag}GPU memory — peak allocated: {peak_allocated / 1e9:.3f} GB | "
              f"peak reserved: {peak_reserved / 1e9:.3f} GB")
        return {
            "peak_allocated_gb": peak_allocated / 1e9,
            "peak_reserved_gb":  peak_reserved  / 1e9,
        }

##############################################
# Main entry point
##############################################

def main_scannet():
    np.random.seed(55)
    torch.manual_seed(55)
    args = parser.parse_args()
    
    model_path      = join(args.save_dir, f'ash_grid.pth')
    mesh_path       = join(args.save_dir, f'ash_pred_mesh.ply')
    metrics_path    = join(args.save_dir, f'ash_metrics.json')
    
    tracker = GPUMemoryTracker(device=cfg['device'])
    
    tracker.start()
    start_time = time.time()
    cfg, ash_grid, dataset = initialize_scannet(args)
    elapsed_time = time.time() - start_time
    mem_stats = tracker.report(label="mapping")
    
    mapping(cfg, ash_grid, dataset)
    
    # Evaluate
    ash_grid.print_ash_stats()
    # torch.save(ash_grid, model_path) # The ASHEngine doesn't want to get pickled, so we need to figure out how to save it.
    mesh = utils_sdf.save_mesh(ash_grid, ash_grid.bound, save_path=mesh_path)
    gt_mesh_path = join(args.scannet_root, f"scene{args.scene}/scene{args.scene}_vh_clean.ply")
    
    verts_pred = sample_points_from_mesh(mesh_path, mesh_sample_point=1000000)
    verts_trgt = sample_points_from_mesh(gt_mesh_path, mesh_sample_point=1000000)
    
    metrics_results = compute_chamfer_metrics(verts_pred, verts_trgt, threshold=0.05, truncation_acc=0.50, truncation_com=0.50)
    metrics_results["gpu_peak_allocated_gb"] = mem_stats["peak_allocated_gb"]
    metrics_results["gpu_peak_reserved_gb"]  = mem_stats["peak_reserved_gb"]
    metrics_results = {k: round(v, 2) for k, v in metrics_results.items()} # Round to 2 decimals
    metrics_results["training_time_s"] = round(elapsed_time, 3)
    print(json.dumps(metrics_results, indent=4))
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_results, f, indent=4)
    print(f"Saved metrics → {metrics_path}")

if __name__ == "__main__":
    main_scannet()
