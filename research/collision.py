import torch
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from grid_opt.utils.utils_eval import nn_correspondance, sample_points_from_mesh


###############################################################
# Constants / Configuration
###############################################################

# Multiresolution Hash-Grid (MHE) parameters (taken from InstantNGP)
PI      = [1, 2654435761, 805459861]
T       = 2**19
L       = 16
N_min   = 16
N_max   = 2048
b       = torch.exp((torch.log(torch.tensor(N_max)) - torch.log(torch.tensor(N_min))) / (L - 1)).item()

# Data
BOUNDS                  = torch.tensor([[-0.02,  10.38], [-0.01, 8.74], [-0.01,  3.03]])
COLLISION_STATS_PATH    = "./collision_stats.pt"
MESH_PATH               = "./results/mapping/hash_pred_mesh.ply"
GT_MESH_PATH            = "../../data/ScanNet/scans/scene0000_00/scene0000_00_vh_clean.ply"

def print_config() -> None:
    print("\n" + "="*40)
    print("MHE CONFIGURATION")
    print("="*40)
    print(f" * Hash Table Size (T):     {T:,}")
    print(f" * Number of Levels (L):    {L}")
    print(f" * Min Resolution (N_min):  {N_min}")
    print(f" * Max Resolution (N_max):  {N_max}")
    print(f" * Growth Factor (b):       {b:.4f}")
    print(f" * Primes (π):              {PI}")
    print("="*40 + "\n")
    
    
###############################################################
# Spatial Collision Density Analysis
###############################################################

def spatial_hash(x: torch.Tensor) -> torch.Tensor:
    """ Hash function for 3D coordinates using bitwise XOR and primes. """
    assert x.shape[1] == 3
    # Use int64 for intermediate calculation to avoid overflow before modulo T
    h = (x[:, 0].long() * PI[0]) ^ (x[:, 1].long() * PI[1]) ^ (x[:, 2].long() * PI[2])
    return h % T

def get_vertices_for_index(target_index: int, N_l: int) -> torch.Tensor:
    """ Reverse-check: Which coordinates in a grid of resolution N_l hash to a specific index? """
    coords = []
    # NOTE: This is a brute-force check for analysis purposes
    for z in range(N_l + 1):
        x_c = torch.arange(N_l + 1)
        y_c = torch.arange(N_l + 1)
        grid_x, grid_y = torch.meshgrid(x_c, y_c, indexing='ij')
        verts = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.full_like(grid_x.flatten(), z)], dim=1)
        indices = spatial_hash(verts)
        matches = verts[indices == target_index]
        if matches.shape[0] > 0:
            coords.append(matches)
    return torch.cat(coords) if coords else torch.tensor([])

def analyze_hottest_bins(h_func, N_l: int, top_k: int = 1):
    """ 
    Finds the bins with the highest potential collisions (C_pot) 
    and analyzes their spatial distribution.
    """
    bin_counts = torch.zeros(T, dtype=torch.long)
    total_verts = (N_l + 1)**3
    
    # Calculate counts via chunking
    for z in range(N_l + 1):
        x_c = torch.arange(N_l + 1); y_c = torch.arange(N_l + 1)
        grid_x, grid_y = torch.meshgrid(x_c, y_c, indexing='ij')
        verts = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.full_like(grid_x.flatten(), z)], dim=1)
        indices = h_func(verts)
        bin_counts.put_(indices, torch.ones_like(indices), accumulate=True)

    # Get stats
    max_val, hottest_idx = torch.max(bin_counts, dim=0)
    # Find a 'least hot' bin that is still occupied (C_pot > 0)
    occupied_indices = torch.where(bin_counts > 0)[0]
    min_val, min_occ_idx = torch.min(bin_counts[occupied_indices], dim=0)
    least_hot_idx = occupied_indices[min_occ_idx]

    print(f"\nLevel Analysis: N_l = {N_l}")
    print("="*40)
    print(f"Hottest Index: {hottest_idx.item():<8} | Load: {max_val.item()}")
    print(f"Coldest Index: {least_hot_idx.item():<8} | Load: {min_val.item()}")
    print("="*40)

    # Find where the hottest collisions are located
    # This checks which coordinates mapped to that specific hot index
    hot_coords = []
    for z in range(N_l + 1):
        x_c = torch.arange(N_l + 1); y_c = torch.arange(N_l + 1)
        grid_x, grid_y = torch.meshgrid(x_c, y_c, indexing='ij')
        verts = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.full_like(grid_x.flatten(), z)], dim=1)
        indices = h_func(verts)
        matches = verts[indices == hottest_idx]
        if matches.shape[0] > 0:
            hot_coords.append(matches)
    
    all_hot_coords = torch.cat(hot_coords)
    print(f"Sample coordinates for Hottest Index {hottest_idx.item()}:")
    # Print first few to look for patterns
    for i in range(min(5, all_hot_coords.shape[0])):
        print(f"  - Vertex {i+1}: {all_hot_coords[i].tolist()}")

def potential_collisions(h_func, N_min: int, T: int, L: int, b: float) -> None:
    """ 
    Calculates the frequency and density of potential collisions for each level. 
    """
    header = f"{'Level':<6} | {'Res (N_l)':<10} | {'Total Verts':<14} | {'Coll. Ratio':<12} | {'Avg. Load'}"
    print(header)
    print("-" * len(header))

    for l_idx in range(L):
        N_l = int(torch.floor(torch.tensor(N_min * (b**(l_idx)))).item())
        total_vertices = (N_l + 1)**3
        
        # Tracking counts per bin
        bin_counts = torch.zeros(T, dtype=torch.long)
        
        # Memory-efficient XY-plane chunking
        for z in range(N_l + 1):
            x_coords = torch.arange(N_l + 1)
            y_coords = torch.arange(N_l + 1)
            grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij')
            
            slice_verts = torch.stack([
                grid_x.flatten(), 
                grid_y.flatten(), 
                torch.full_like(grid_x.flatten(), z)
            ], dim=1)
            
            indices = h_func(slice_verts)
            # Efficient histogram update
            bin_counts.put_(indices, torch.ones_like(indices, dtype=torch.long), accumulate=True)

        # Metrics calculation
        occupied_mask = bin_counts > 0
        num_occupied_bins = occupied_mask.sum().item()
        
        # Collision Ratio: Vertices that don't have their own unique bin
        collision_ratio = 0.0
        if total_vertices > num_occupied_bins:
            collision_ratio = (total_vertices - num_occupied_bins) / total_vertices

        # Average Load: For bins that have something in them, what is the average count?
        # A load of 1.0 means perfect 1:1 mapping (no collisions)[cite: 191].
        avg_load = 0.0
        if num_occupied_bins > 0:
            avg_load = total_vertices / num_occupied_bins

        print(f"{l_idx+1:<6} | {N_l:<10} | {total_vertices:<14} | {collision_ratio:>12.2%} | {avg_load:.4f}")

def analyze_disambiguation(stats_path, mesh_path, gt_mesh_path, bound):
    """ Correlates per-vertex Accuracy with C_grad and prints layer-wise conflict stats. """
    data = torch.load(stats_path)
    cfg, L, T = data["config"], data["config"]["L"], data["config"]["T"]
    PI = [1, 2654435761, 805459861]
    
    r_dom_table = torch.ones((L, T))
    
    print("\n" + "="*100)
    print(f"{'LAYER-WISE SPATIAL CONFLICT ANALYSIS (1 - R_dom)':^100}")
    print("="*100)
    print(f"{'L':<3} | {'Res':<6} | {'Bins (Occ)':<12} | {'C_grad (Min / Avg / Max)':<35}")
    print("-" * 100)

    for l in range(L):
        # Calculate R_dom for bins with effective collisions (C_eff > 1)
        # Bins with C_eff <= 1 have R_dom = 1.0 (no conflict)
        occ = data["eff_voxel_count"][l] > 1 
        num_occ = occ.sum().item()
        res = math.floor(cfg["N_min"] * (cfg["b"] ** l))
        
        if num_occ > 0:
            r_dom_vals = data["max_voxel_grad"][l][occ] / data["total_bin_grad"][l][occ].clamp(min=1e-6)
            r_dom_table[l, occ] = r_dom_vals
            
            # Level Conflict = 1 - R_dom
            conflicts = 1.0 - r_dom_vals
            c_str = f"{conflicts.min():.4f} / {conflicts.mean():.4f} / {conflicts.max():.4f}"
            print(f"{l+1:<3} | {res:<6} | {num_occ:<12,} | {c_str:<35}")
        else:
            print(f"{l+1:<3} | {res:<6} | {'0':<12} | {'0.0000 / 0.0000 / 0.0000':<35}")

    # Calculate distance to closest point in GT as error
    verts_pred = sample_points_from_mesh(mesh_path, mesh_sample_point=1000000)
    verts_trgt = sample_points_from_mesh(gt_mesh_path, mesh_sample_point=1000000)
    
    _, dist_p = nn_correspondance(verts_pred, verts_trgt, 0.50, True) 
    dist_p = np.array(dist_p)

    # Calculate conflict C_grad(x)
    verts_torch = torch.from_numpy(verts_pred).float()
    b_min, b_max = bound[:, 0], bound[:, 1]
    x = (verts_torch - b_min) / (b_max - b_min)
    x = torch.clamp(x, 0.0, 1.0 - 1e-6)
    
    c_grad_scores = torch.zeros(len(verts_pred))
    for l in range(L):
        res = math.floor(cfg["N_min"] * (cfg["b"] ** l))
        v_base = torch.floor(x * res).long()
        h_idx = ((v_base[:, 0] * PI[0]) ^ (v_base[:, 1] * PI[1]) ^ (v_base[:, 2] * PI[2])) % T
        c_grad_scores += (1.0 - r_dom_table[l, h_idx])

    # 4. Filter and Analyze Trends
    c_grad_np = c_grad_scores.numpy()[:len(dist_p)]
    mask = dist_p < 0.10 # Filter outliers for cleaner trendline
    c_f, e_f = c_grad_np[mask], dist_p[mask]

    # Initialize Plot
    plt.figure(figsize=(10, 6))
    plt.hexbin(c_f, e_f, gridsize=60, cmap='YlOrRd', mincnt=1)
    
    # Compute Trendline via Binned Averages
    bins = np.linspace(c_f.min(), c_f.max(), 40)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = np.array([e_f[np.digitize(c_f, bins) == i].mean() for i in range(1, len(bins))])
    
    valid = ~np.isnan(bin_means)
    plt.plot(bin_centers[valid], bin_means[valid], color='blue', lw=3, label='Disambiguation Slope')
    
    # Sensitivity Metric (Linear Fit)
    slope, intercept = np.polyfit(bin_centers[valid], bin_means[valid], 1)
    print("="*100)
    print(f"OVERALL DISAMBIGUATION POWER")
    print(f" * Sensitivity Slope: {slope:.8f} (m per conflict unit)")
    print(f" * Intercept (Base):  {intercept:.4f} m")
    print("="*100 + "\n")

    # Finalize and Save Plot
    plt.xlabel("Spatial Conflict Index $C_{grad}(\mathbf{x})$")
    plt.ylabel("Accuracy Error (m)")
    plt.title("Decoder Disambiguation Analysis: Conflict vs. Geometric Accuracy")
    plt.colorbar(label='Point Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("./disambiguation_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

###############################################################
# Main Program Entry
###############################################################

if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print_config()
    
    ####################################
    # Collision Analysis
    ####################################
    
    # Summarizes collisions statistics across all layers
    # potential_collisions(spatial_hash, N_min, T, L, b)
    
    # analyze_hottest_bins(spatial_hash, N_l=2048)
    # analyze_hottest_bins(spatial_hash, N_l=406)
    
    analyze_disambiguation(
        stats_path=COLLISION_STATS_PATH, 
        mesh_path=MESH_PATH, 
        gt_mesh_path=GT_MESH_PATH, 
        bound=BOUNDS
    )
    