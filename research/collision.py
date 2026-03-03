import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from grid_opt.utils.utils_eval import nn_correspondance, sample_points_from_mesh
from grid_opt.models.collision_tracker import CollisionTracker, spatial_hash


###############################################################
# Constants / Configuration
###############################################################

COLLISION_STATS_PATH = "./collision_stats.pt"
MESH_PATH            = "./results/mapping/hash_pred_mesh.ply"
GT_MESH_PATH         = "../../data/ScanNet/scans/scene0000_00/scene0000_00_vh_clean.ply"
MODEL_PATH           = "./results/mapping/hash_grid.pth"


###############################################################
# Conflict Metrics
###############################################################

@torch.no_grad()
def get_C_eff_grad(tracker: CollisionTracker, x_world: torch.Tensor) -> torch.Tensor:
    """
    C_eff_grad(x) = sum_l sum_c w_c * C_eff(l, h_l(c)) / C_pot_avg(l)
    
    Gradient-agnostic version of C_grad. Uses raw collision counts normalized
    by the expected collisions per bin at each level, making layers comparable.
    C_pot_avg(l) = (N_l + 1)^3 / T is the expected number of voxels per bin.
    """
    lo = tracker.scene_bound[:, 0]
    hi = tracker.scene_bound[:, 1]
    x_norm = ((x_world - lo) / (hi - lo)).clamp(0.0, 1.0)
    M = x_norm.shape[0]

    C_eff_grad = torch.zeros(M, device=tracker.device)

    for level_idx in range(tracker.n_levels):
        N_l = tracker.resolutions[level_idx].item()
        C_pot_avg = (N_l + 1) ** 3 / tracker.T  # expected voxels per bin

        x_scaled = x_norm * N_l
        x_floor  = torch.floor(x_scaled).long()
        w        = x_scaled - x_floor.float()

        corners      = x_floor.unsqueeze(1) + tracker.encoding.corner_offsets.unsqueeze(0)
        corners_flat = corners.reshape(M * 8, 3)

        h_idx             = spatial_hash(corners_flat, tracker.T)
        c_eff_corners     = tracker.C_eff[level_idx][h_idx].float().reshape(M, 8)
        normalized_corners = c_eff_corners / C_pot_avg

        wx0, wx1 = 1.0 - w[:, 0], w[:, 0]
        wy0, wy1 = 1.0 - w[:, 1], w[:, 1]
        wz0, wz1 = 1.0 - w[:, 2], w[:, 2]

        weights = torch.stack([
            wx0 * wy0 * wz0, wx0 * wy0 * wz1,
            wx0 * wy1 * wz0, wx0 * wy1 * wz1,
            wx1 * wy0 * wz0, wx1 * wy0 * wz1,
            wx1 * wy1 * wz0, wx1 * wy1 * wz1,
        ], dim=1)

        C_eff_grad += (weights * normalized_corners).sum(dim=1)

    return C_eff_grad


###############################################################
# Plotting
###############################################################

def plot_conflict_vs_error(
    conflict_scores: np.ndarray,
    errors: np.ndarray,
    label: str,
    save_path: str,
    error_threshold: float = 0.10,
):
    mask = errors < error_threshold
    c_f, e_f = conflict_scores[mask], errors[mask]

    slope, intercept, bin_centers, bin_means = _compute_trendline(c_f, e_f)

    print("\n" + "=" * 80)
    print(f"DISAMBIGUATION POWER — {label}")
    print(f"  Sensitivity Slope : {slope:.8f}  (m per conflict unit)")
    print(f"  Intercept (Base)  : {intercept:.4f} m")
    print("=" * 80 + "\n")

    fig, ax = plt.subplots(figsize=(10, 6))
    hb = ax.hexbin(c_f, e_f, gridsize=60, cmap='YlOrRd', mincnt=1)
    fig.colorbar(hb, ax=ax, label='Point Density')

    valid = ~np.isnan(bin_means)
    ax.plot(bin_centers[valid], bin_means[valid], color='blue', lw=3, label='Binned Mean')

    ax.set_xlabel(f"Conflict Score  [{label}]")
    ax.set_ylabel("Geometric Error (m)")
    ax.set_title(f"Decoder Disambiguation Analysis: {label} vs. Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot → {save_path}")
    return slope


def plot_conflict_comparison(
    c_grad: np.ndarray,
    c_eff_grad: np.ndarray,
    errors: np.ndarray,
    save_path: str,
    error_threshold: float = 0.10,
):
    """Side-by-side comparison of C_grad vs C_eff_grad."""
    mask = errors < error_threshold

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    titles   = ["$C_{grad}$ (gradient-weighted)", "$C_{eff-grad}$ (frequency, normalized)"]
    datasets = [c_grad[mask], c_eff_grad[mask]]
    e_f = errors[mask]

    for ax, scores, title in zip(axes, datasets, titles):
        hb = ax.hexbin(scores, e_f, gridsize=60, cmap='YlOrRd', mincnt=1)
        fig.colorbar(hb, ax=ax, label='Point Density')

        _, _, bin_centers, bin_means = _compute_trendline(scores, e_f)
        valid = ~np.isnan(bin_means)
        ax.plot(bin_centers[valid], bin_means[valid], color='blue', lw=3, label='Binned Mean')

        ax.set_xlabel("Conflict Score")
        ax.set_ylabel("Geometric Error (m)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("C_grad vs C_eff_grad: Disambiguation Power Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot → {save_path}")


def _compute_trendline(conflict: np.ndarray, errors: np.ndarray, n_bins: int = 40):
    bins        = np.linspace(conflict.min(), conflict.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    digitized   = np.digitize(conflict, bins)

    bin_means = np.array([
        errors[digitized == i].mean() if (digitized == i).any() else np.nan
        for i in range(1, len(bins))
    ])

    valid = ~np.isnan(bin_means)
    slope, intercept = np.polyfit(bin_centers[valid], bin_means[valid], 1)
    return slope, intercept, bin_centers, bin_means


###############################################################
# Main Analysis
###############################################################

def analyze_disambiguation(model_path, stats_path, mesh_path, gt_mesh_path, device="cuda"):
    # 1. Load model and tracker
    print("Loading model and collision statistics...")
    hash_grid = torch.load(model_path, map_location=device)
    tracker = CollisionTracker.load(
        path=stats_path,
        encoding=hash_grid.encoding,
        scene_bound=hash_grid.bound,
        device=device,
    )
    tracker.print_summary()

    # 2. Load meshes and compute geometric error
    print("Sampling points from meshes...")
    verts_pred = sample_points_from_mesh(mesh_path,    mesh_sample_point=1_000_000)
    verts_trgt = sample_points_from_mesh(gt_mesh_path, mesh_sample_point=1_000_000)

    print("Calculating nearest-neighbour correspondences...")
    _, dist_p = nn_correspondance(verts_pred, verts_trgt, 0.50, False)
    dist_p = np.array(dist_p).flatten()

    # 3. Compute both conflict metrics in batches
    print("Computing conflict scores...")
    verts_torch  = torch.from_numpy(verts_pred).float().to(device)
    BATCH        = 100_000
    n            = len(verts_torch)
    c_grad_t     = torch.zeros(n, device=device)
    c_eff_grad_t = torch.zeros(n, device=device)

    with torch.no_grad():
        for i in range(0, n, BATCH):
            batch = verts_torch[i : i + BATCH]
            c_grad_t    [i : i + BATCH] = tracker.get_C_grad(batch)
            c_eff_grad_t[i : i + BATCH] = get_C_eff_grad(tracker, batch)

    c_grad_np     = c_grad_t.cpu().numpy()
    c_eff_grad_np = c_eff_grad_t.cpu().numpy()

    # 4. Individual plots
    slope_c_grad = plot_conflict_vs_error(
        c_grad_np, dist_p,
        label="$C_{grad}$",
        save_path="./disambiguation_c_grad.png",
    )
    slope_c_eff_grad = plot_conflict_vs_error(
        c_eff_grad_np, dist_p,
        label="$C_{eff-grad}$",
        save_path="./disambiguation_c_eff_grad.png",
    )

    # 5. Side-by-side comparison
    plot_conflict_comparison(
        c_grad_np, c_eff_grad_np, dist_p,
        save_path="./disambiguation_comparison.png",
    )

    # 6. Summary
    print("\n" + "=" * 80)
    print("METRIC COMPARISON SUMMARY")
    print(f"  C_grad     slope: {slope_c_grad:.8f}")
    print(f"  C_eff_grad slope: {slope_c_eff_grad:.8f}")
    if abs(slope_c_grad) > abs(slope_c_eff_grad):
        print("  → C_grad correlates MORE strongly: gradient asymmetry is doing real work")
    else:
        print("  → C_eff_grad correlates MORE strongly: raw collision frequency drives error")
    print("=" * 80 + "\n")


###############################################################
# Entry Point
###############################################################

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    analyze_disambiguation(
        model_path=MODEL_PATH,
        stats_path=COLLISION_STATS_PATH,
        mesh_path=MESH_PATH,
        gt_mesh_path=GT_MESH_PATH,
        device=device,
    )
    