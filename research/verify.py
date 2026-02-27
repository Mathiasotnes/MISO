###############################################################
# This script aims to verify parts of the research so that 
# we can be more confident about the results.
###############################################################

import torch
import tinycudann as tcnn
import numpy as np
import math


###############################################################
# Verification Implementations
###############################################################

def manual_spatial_hash(coords, T):
    """ Matches TCNN bitwise XOR hashing logic using unsigned 32-bit primes. """
    PI = torch.tensor([1, 2654435761, 805459861], dtype=torch.uint32, device='cuda')
    x = coords.to(torch.uint32)
    h = (x[:, 0] * PI[0]) ^ (x[:, 1] * PI[1]) ^ (x[:, 2] * PI[2])
    return h % T

def verify_tcnn_hash(n_levels=16, log2_T=15, base_res=16, scale=1.26):
    T = 2**log2_T
    tcnn_config = {
        "otype": "HashGrid",
        "n_levels": n_levels,
        "n_features_per_level": 1,
        "log2_hashmap_size": log2_T,
        "base_resolution": base_res,
        "per_level_scale": scale,
    }
    encoding = tcnn.Encoding(3, tcnn_config).cuda()
    
    with torch.no_grad():
        encoding.params.fill_(0.0)

    # Targeting the finest layer (l = L - 1)
    target_level = n_levels - 1
    res_l = math.floor(base_res * (scale ** target_level))
    
    # Test coordinate
    test_voxel = torch.tensor([5, 5, 5], device='cuda') 
    normalized_input = (test_voxel.float() + 0.5) / res_l
    
    # Calculate the 8 corners of this voxel
    offsets = torch.stack(torch.meshgrid([torch.tensor([0, 1])] * 3, indexing='ij')).reshape(3, -1).t().cuda()
    corners = test_voxel + offsets # Shape: (8, 3)
    
    # Compute manual hash indices for these 8 corners
    local_indices = manual_spatial_hash(corners, T)

    # Calculate the global offset in the TCNN parameter vector
    # Level l params start after all params of levels 0 through l-1.
    # Each level has T parameters if (res+1)^3 > T, otherwise it has (res+1)^3.
    global_offset = 0
    for l in range(target_level):
        lvl_res = math.floor(base_res * (scale ** l))
        # TCNN chooses the minimum of the grid size and the hash table size
        global_offset += min(T, (lvl_res + 1)**3)

    # Set these 8 specific memory locations to 1.0
    with torch.no_grad():
        global_indices = global_offset + local_indices
        encoding.params[global_indices] = 1.0
    
    # TCNN output will be the trilinear interpolation of these 8 corners.
    # Since we set all 8 to 1.0, the result should be 1.0.
    output = encoding(normalized_input.view(1, 3))
    
    # Extract only the target level's feature (TCNN concatenates all level outputs)
    target_level_output = output[0, target_level] 

    if torch.allclose(target_level_output, torch.tensor(1.0).cuda(), atol=1e-3):
        print(f"✅ SUCCESS: Finest Level ({target_level}) hash verified.")
        print(f"   Resolution: {res_l}, Global Offset: {global_offset}")
    else:
        print(f"❌ FAILURE: Expected 1.0, got {target_level_output.item()}. Check indexing logic.")


###############################################################
# Verification Implementations
###############################################################

if __name__ == "__main__":
    verify_tcnn_hash()