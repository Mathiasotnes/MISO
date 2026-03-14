import argparse
import numpy as np
import json
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
    cfg = create_configs_scannet(args, dataset)

    ash_grid = GridASH(cfg['model'], device=cfg['device'], dtype=torch.float32)
    ash_grid.to(cfg['device'])
    
    return cfg, ash_grid, dataset

def train_epoch(model):
    """ Override this if there is a need to override the dataset iteration. """
    

def mapping(cfg, ash_grid:GridASH, dataset:SubmapDataset):
    # Mapping one frame at a time to work "online"
    cfg_map = cfg['mapping']
    ash_grid.train()
    timer = PerfTimer(activate=True)
    train_loader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=0)
    
    loss_fn = MisoLossMapping(
        weight_sdf=cfg_map['weight_sdf'],
        weight_eik=cfg_map['weight_eik'],
        weight_fs=cfg_map['weight_fs'],
        loss_type=cfg_map['loss_type'],
        trunc_dist=cfg_map['trunc_dist'],
        finite_diff_eps=cfg_map['finite_diff_eps'],
        grad_method=cfg_map['grad_method'],
        eik_trunc_dist=cfg_map['eik_trunc_dist']
    )
    
    cpu_time, gpu_time = 0, 0
    initial_kf_id = 0
    kf_ids = []
    R, t = dataset.true_kf_pose_in_world(initial_kf_id)
    ash_grid.set_initial_kf_pose(initial_kf_id, R, t, kf_key=f"KF{initial_kf_id}")
    
    # Iterate over frames
    # NOTE: We should probably have some replay buffer to avoid catastrophic forgetting.
    for kf_id in range(dataset.num_kfs):
        kf_ids.append(kf_id)
        dataset.select_keyframes(kf_ids) # Currently training equally on all frames up to the current frame (not ideal).
        timer.reset()
        
        # Taken from loss_fn.compute() to get coords in world frame. It's probably better to just call the "prepare_features()" directly in the loss function
        # to avoid calculating this twice.
        model_input, gt = next(iter(train_loader))
        model_input, gt = prepare_batch(model_input, gt)
        coords_frame = model_input['coords_frame'][0]
        sample_frame_ids = model_input['sample_frame_ids'][0, :, 0]
        sample_weights = model_input['weights'][0]
        gt_sdf = gt['sdf'][0]
        assert coords_frame.ndim == 2 and gt_sdf.ndim == 2
        assert sample_weights.shape == gt_sdf.shape
        # Transform coords from keyframe to world frame
        unique_frame_ids = np.unique(sample_frame_ids.detach().cpu().numpy()).tolist()
        coords_world = coords_frame.clone()
        for kf_id in unique_frame_ids:
            idxs_select = torch.nonzero(sample_frame_ids == kf_id, as_tuple=False).squeeze(1)
            if idxs_select.numel() == 0: continue
            R_world_frame, t_world_frame = loss_fn.query_kf_pose(ash_grid, kf_id)
            coords_world[idxs_select, :] = utils_geometry.transform_points_to(
                coords_frame[idxs_select, :],
                R_world_frame,
                t_world_frame
            )
        
        ash_grid.prepare_features(coords_world) # This will make the features at the current frame trainable, and freeze all other features.
        
        optimizer = torch.optim.Adam(ash_grid.parameters(), lr=cfg['train']['learning_rate'])
        optimizer.zero_grad()
                
        # Loss 
        total_loss = 0.
        loss_dict = loss_fn.compute(ash_grid, model_input, gt)
        for _, loss in loss_dict.items():
            single_loss = loss.mean()
            total_loss += single_loss

        # Backward step
        if not torch.isnan(total_loss):
            total_loss.backward(retain_graph=False)
            optimizer.step()
                    
        else:
            logger.warning(f"Loss at frame {kf_id} is nan! Skip backward step.")
            
        ash_grid.sync_active_to_store()

        # Logging
        step_cpu_time, step_gpu_time = timer.check()
        logger.info(f"Frame {kf_id} | train_loss={total_loss.item():.2e}")
        cpu_time += step_cpu_time
        gpu_time += step_gpu_time
    

##############################################
# Main entry point
##############################################

def main_scannet():
    np.random.seed(55)
    torch.manual_seed(55)
    args = parser.parse_args()
    
    model_path      = join(args.save_dir, f'ash_grid.pth')
    mesh_path       = join(args.save_dir, f'hash_pred_mesh.ply')
    metrics_path    = join(args.save_dir, f'metrics.json')
    
    cfg, ash_grid, dataset = initialize_scannet(args)
    
    mapping(cfg, ash_grid, dataset)
    
    # Evaluate
    ash_grid.print_ash_stats()
    # torch.save(ash_grid, model_path) # The ASHEngine doesn't want to get pickled, so we need to figure out how to save it.
    mesh = utils_sdf.save_mesh(ash_grid, ash_grid.bound, save_path=mesh_path)
    gt_mesh_path = join(args.scannet_root, f"scene{args.scene}/scene{args.scene}_vh_clean.ply")
    
    verts_pred = sample_points_from_mesh(mesh_path, mesh_sample_point=1000000)
    verts_trgt = sample_points_from_mesh(gt_mesh_path, mesh_sample_point=1000000)
    
    metrics_results = compute_chamfer_metrics(verts_pred, verts_trgt, threshold=0.05, truncation_acc=0.50, truncation_com=0.50)
    print(json.dumps(metrics_results, indent=4))
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_results, f, indent=4)
    print(f"Saved metrics → {metrics_path}")

if __name__ == "__main__":
    main_scannet()
