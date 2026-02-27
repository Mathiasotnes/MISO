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

def manual_spatial_hash(coords, T, res):
    """ Matches TCNN bitwise XOR hashing logic using long for math to avoid CUDA errors. 
    Uses linear indexing when the grid size is small enough to fit in the hash table. """
    if (res + 1)**3 <= T:
        return coords[:, 0] + coords[:, 1] * (res + 1) + coords[:, 2] * (res + 1)**2
    
    # Perform multiplication in 64-bit to avoid PyTorch uint32 errors
    # then cast to 32-bit to simulate the 32-bit overflow TCNN expects
    PI = [1, 2654435761, 805459861]
    h_x = (coords[:, 0].long() * PI[0]).to(torch.int32)
    h_y = (coords[:, 1].long() * PI[1]).to(torch.int32)
    h_z = (coords[:, 2].long() * PI[2]).to(torch.int32)
    h = h_x ^ h_y ^ h_z
    return h.to(torch.uint32).long() % T

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
    
    # Targeting the finest layer for validation
    target_level = n_levels - 1
    res_l = math.floor(base_res * (scale ** target_level))
    
    # Test coordinate in the center of a voxel
    test_voxel = torch.tensor([5, 5, 5], device='cuda') 
    normalized_input = (test_voxel.float() + 0.5) / res_l
    
    # Predict indices manually
    offsets = torch.stack(torch.meshgrid([torch.tensor([0, 1])] * 3, indexing='ij')).reshape(3, -1).t().cuda()
    corners = test_voxel + offsets
    local_indices = manual_spatial_hash(corners, T, res_l)

    global_offset = 0
    for l in range(target_level):
        lvl_res = math.floor(base_res * (scale ** l))
        global_offset += min(T, (lvl_res + 1)**3)

    manual_indices = (global_offset + local_indices).sort().values

    # Extract indices from TCNN via Gradients
    encoding.params.grad = None # Ensure clean gradients
    input_batch = normalized_input.view(1, 3).requires_grad_(True)
    
    output = encoding(input_batch)
    # Sum only the feature of our target level to avoid noise from other layers
    loss = output[0, target_level].sum()
    loss.backward()

    # The indices with non-zero gradients are the ones TCNN actually used
    tcnn_indices = torch.where(encoding.params.grad != 0)[0].sort().values

    # Results
    print(f"\nMode: {'Linear' if (res_l+1)**3 <= T else 'Hash'} | Resolution: {res_l}")
    print(f"Manual Indices: {manual_indices.tolist()}")
    print(f"TCNN   Indices: {tcnn_indices.tolist()}")

    if torch.equal(manual_indices, tcnn_indices):
        print("✅ SUCCESS: Manual indices match TCNN exactly.")
    else:
        print("❌ FAILURE: Index mismatch detected.")

###############################################################
# Verification Implementations
###############################################################

if __name__ == "__main__":
    verify_tcnn_hash(n_levels=1, log2_T=15, base_res=16, scale=1.26)
    verify_tcnn_hash(n_levels=16, log2_T=15, base_res=16, scale=1.26)
    