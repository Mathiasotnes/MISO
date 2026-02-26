import torch
import matplotlib.pyplot as plt

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

###############################################################
# Main Program Entry
###############################################################

if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_config()
    
    ####################################
    # Collision Analysis
    ####################################
    
    # Summarizes collisions statistics across all layers
    # potential_collisions(spatial_hash, N_min, T, L, b)
    
    # analyze_hottest_bins(spatial_hash, N_l=2048)
    # analyze_hottest_bins(spatial_hash, N_l=406)
    