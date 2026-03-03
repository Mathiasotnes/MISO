import torch
import numpy as np
import matplotlib
import json
from os.path import join
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from grid_opt.utils.utils_eval import nn_correspondance, sample_points_from_mesh
from grid_opt.models.collision_tracker import CollisionTracker, spatial_hash


###############################################################
# Constants / Configuration
###############################################################

STATS_PATH          = "./collision_stats.pt"
MESH_PATH           = "./results/mapping/hash_pred_mesh.ply"
GT_MESH_PATH        = "../../data/ScanNet/scans/scene0000_00/scene0000_00_vh_clean.ply"
MODEL_PATH          = "./results/mapping/hash_grid.pth"

MODEL_PATH_HIGH_T   = "./results/mapping/hash_grid_high_T.pth"
STATS_PATH_HIGH_T   = "./collision_stats_high_T.pt"
MESH_PATH_HIGH_T    = "./results/mapping/hash_pred_mesh_high_T.ply"


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

@torch.no_grad()
def get_per_level_conflict(tracker: CollisionTracker, x_world: torch.Tensor) -> torch.Tensor:
    """
    Returns per-level conflict scores: (M, L) tensor.
    Each column l is the trilinearly-interpolated (1 - R_dom) at level l.
    """
    lo = tracker.scene_bound[:, 0]
    hi = tracker.scene_bound[:, 1]
    x_norm = ((x_world - lo) / (hi - lo)).clamp(0.0, 1.0)
    M = x_norm.shape[0]

    R_dom = tracker.get_R_dom()
    per_level = torch.zeros(M, tracker.n_levels, device=tracker.device)

    for level_idx in range(tracker.n_levels):
        N_l = tracker.resolutions[level_idx].item()

        x_scaled = x_norm * N_l
        x_floor  = torch.floor(x_scaled).long()
        w        = x_scaled - x_floor.float()

        corners      = x_floor.unsqueeze(1) + tracker.encoding.corner_offsets.unsqueeze(0)
        corners_flat = corners.reshape(M * 8, 3)

        h_idx            = spatial_hash(corners_flat, tracker.T)
        r_dom_corners    = R_dom[level_idx][h_idx].reshape(M, 8)
        conflict_corners = 1.0 - r_dom_corners

        wx0, wx1 = 1.0 - w[:, 0], w[:, 0]
        wy0, wy1 = 1.0 - w[:, 1], w[:, 1]
        wz0, wz1 = 1.0 - w[:, 2], w[:, 2]

        weights = torch.stack([
            wx0 * wy0 * wz0, wx0 * wy0 * wz1,
            wx0 * wy1 * wz0, wx0 * wy1 * wz1,
            wx1 * wy0 * wz0, wx1 * wy0 * wz1,
            wx1 * wy1 * wz0, wx1 * wy1 * wz1,
        ], dim=1)

        per_level[:, level_idx] = (weights * conflict_corners).sum(dim=1)

    return per_level  # (M, L)

def print_C_grad_stats(
    tracker: CollisionTracker,
    verts: np.ndarray,
    device: str = "cuda",
    label: str = "C_grad",
):
    verts_t  = torch.from_numpy(verts).float().to(device)
    BATCH    = 100_000
    n        = len(verts_t)
    c_grad_t = torch.zeros(n, device=device)

    with torch.no_grad():
        for i in range(0, n, BATCH):
            c_grad_t[i : i + BATCH] = tracker.get_C_grad(verts_t[i : i + BATCH])

    c = c_grad_t.cpu().numpy()
    print(f"\n{label} stats over {n:,} surface points:")
    print(f"  min  : {c.min():.4f}")
    print(f"  max  : {c.max():.4f}")
    print(f"  mean : {c.mean():.4f}")
    print(f"  std  : {c.std():.4f}")
    print(f"  p25  : {np.percentile(c, 25):.4f}")
    print(f"  p50  : {np.percentile(c, 50):.4f}")
    print(f"  p75  : {np.percentile(c, 75):.4f}")
    print(f"  p95  : {np.percentile(c, 95):.4f}")

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

    print_C_grad_stats(tracker, verts_pred, device=device)

def analyze_conflict_vs_T(
    T_values: list,
    results_dir: str,
    gt_mesh_path: str,
    device: str = "cuda",
):
    rows = []

    for T in T_values:
        model_path   = join(results_dir, f'hash_grid_T{T}.pth')
        stats_path   = join(results_dir, f'tracker_T{T}.pt')
        mesh_path    = join(results_dir, f'hash_pred_mesh_T{T}.ply')
        metrics_path = join(results_dir, f'metrics_T{T}.json')

        print(f"Loading T={T}...")
        hash_grid = torch.load(model_path, map_location=device)
        tracker   = CollisionTracker.load(
            path=stats_path,
            encoding=hash_grid.encoding,
            scene_bound=hash_grid.bound,
            device=device,
        )

        verts_pred = sample_points_from_mesh(mesh_path, mesh_sample_point=1_000_000)
        verts_t    = torch.from_numpy(verts_pred).float().to(device)

        BATCH    = 100_000
        n        = len(verts_t)
        c_grad_t = torch.zeros(n, device=device)

        with torch.no_grad():
            for i in range(0, n, BATCH):
                c_grad_t[i : i + BATCH] = tracker.get_C_grad(verts_t[i : i + BATCH])

        c = c_grad_t.cpu().numpy()

        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        rows.append({
            'T'             : T,
            'min'           : c.min(),
            'max'           : c.max(),
            'mean'          : c.mean(),
            'std'           : c.std(),
            'chamfer_l2'    : metrics.get('Chamfer_L2 (cm)', float('nan')),
            'f_score'       : metrics.get('F-Score (%)',    float('nan')),
            'n'             : n,
        })

        del hash_grid, tracker, verts_t, c_grad_t
        torch.cuda.empty_cache()

    # Print table
    w = 106
    print("\n" + "=" * w)
    print(
        f"{'T':>4} | {'2^T':>12} | {'C_grad min':>12} | {'C_grad max':>12} | {'C_grad mean':>12} | "
        f"{'std':>10} | {'Chamfer-L2':>10} | {'F-Score':>10}"
    )
    print("-" * w)
    for r in rows:
        print(
            f"{r['T']:>4} | {2**r['T']:>12,} | {r['min']:>12.2f} | {r['max']:>12.2f} | "
            f"{r['mean']:>12.2f} | {r['std']:>10.4f} | {r['chamfer_l2']:>10.2f} | {r['f_score']:>10.2f}"
        )
    print("=" * w + "\n")

    return rows

def analyze_collision_damage(
    model_path_high_T: str,
    stats_path_high_T: str,
    mesh_path_high_T: str,
    model_path_low_T: str,
    stats_path_low_T: str,
    mesh_path_low_T: str,
    gt_mesh_path: str,
    device: str = "cuda",
):
    """
    Computes collision_damage(x) = error_low_T(x) - error_high_T(x)
    and correlates it with conflict metrics from the low-T model.

    Error decomposition:
        error(x) = e_collision(x) + e_model(x) + e_sampling(x) + noise
    where:
        e_collision(x) = dist_low(x) - dist_high(x)   [what we measure here]
        e_model(x)     = dist_high(x)                  [irreducible model error]
    """
    # 1. Load both trackers
    print("Loading models...")
    hash_grid_low  = torch.load(model_path_low_T,  map_location=device)
    hash_grid_high = torch.load(model_path_high_T, map_location=device)

    tracker_low = CollisionTracker.load(
        path=stats_path_low_T,
        encoding=hash_grid_low.encoding,
        scene_bound=hash_grid_low.bound,
        device=device,
    )
    tracker_high = CollisionTracker.load(
        path=stats_path_high_T,
        encoding=hash_grid_high.encoding,
        scene_bound=hash_grid_high.bound,
        device=device,
    )

    print("High-T model (collision-free baseline):")
    tracker_high.print_summary()
    print("Low-T model (collision-heavy):")
    tracker_low.print_summary()

    # 2. Sample meshes — GT is the common query set
    print("Sampling meshes...")
    verts_trgt      = sample_points_from_mesh(gt_mesh_path,     mesh_sample_point=1_000_000)
    verts_pred_low  = sample_points_from_mesh(mesh_path_low_T,  mesh_sample_point=1_000_000)
    verts_pred_high = sample_points_from_mesh(mesh_path_high_T, mesh_sample_point=1_000_000)

    # 3. Compute errors at GT vertices (common query set)
    print("Computing per-point errors at GT vertices...")
    _, dist_low  = nn_correspondance(verts_trgt, verts_pred_low,  0.50, False)
    _, dist_high = nn_correspondance(verts_trgt, verts_pred_high, 0.50, False)
    dist_low  = np.array(dist_low).flatten()
    dist_high = np.array(dist_high).flatten()

    # Error decomposition
    e_collision = dist_low - dist_high   # collision component  (can be negative)
    e_model     = dist_high              # irreducible model error

    print(f"\nError decomposition summary:")
    print(f"  e_model     MAE  : {np.abs(e_model).mean():.4f} m  (irreducible)")
    print(f"  e_collision MAE  : {np.abs(e_collision).mean():.4f} m")
    print(f"  e_collision mean : {e_collision.mean():.4f} m  (+ = collisions hurt on avg)")
    print(f"  e_collision std  : {e_collision.std():.4f} m")
    print(f"  % where collisions hurt  : {(e_collision > 0).mean()*100:.1f}%")
    print(f"  % where collisions help  : {(e_collision < 0).mean()*100:.1f}%")
    print(f"  collision share of total : {np.abs(e_collision).mean() / dist_low.mean()*100:.1f}%")

    # 4. Compute conflict metrics at GT vertices (matching the error query points)
    print("\nComputing conflict scores at GT vertices...")
    verts_trgt_t = torch.from_numpy(verts_trgt).float().to(device)
    BATCH        = 100_000
    n            = len(verts_trgt_t)
    c_grad_t     = torch.zeros(n, device=device)
    c_eff_grad_t = torch.zeros(n, device=device)

    with torch.no_grad():
        for i in range(0, n, BATCH):
            batch = verts_trgt_t[i : i + BATCH]
            c_grad_t    [i : i + BATCH] = tracker_low.get_C_grad(batch)
            c_eff_grad_t[i : i + BATCH] = get_C_eff_grad(tracker_low, batch)

    c_grad_np     = c_grad_t.cpu().numpy()
    c_eff_grad_np = c_eff_grad_t.cpu().numpy()

    # 5. Correlations — use all points, not just positive damage
    # Masking to positive-only biases the sample and hides disambiguation success
    quality_mask = dist_low < 0.10   # exclude extreme outliers only

    r_collision_cgrad     = np.corrcoef(c_grad_np[quality_mask],     e_collision[quality_mask])[0, 1]
    r_collision_ceffgrad  = np.corrcoef(c_eff_grad_np[quality_mask], e_collision[quality_mask])[0, 1]
    r_model_cgrad         = np.corrcoef(c_grad_np[quality_mask],     e_model[quality_mask])[0, 1]

    print(f"\nCorrelation analysis:")
    print(f"  r(C_grad,     e_collision) : {r_collision_cgrad:.4f}  ← does conflict predict damage?")
    print(f"  r(C_eff_grad, e_collision) : {r_collision_ceffgrad:.4f}")
    print(f"  r(C_grad,     e_model)     : {r_model_cgrad:.4f}  ← does conflict predict irreducible error?")

    # 6. Binned variance analysis — key test for decoder disambiguation
    # If decoder succeeds: damage variance should be FLAT across conflict bins
    # If decoder fails:    damage variance should RISE with conflict
    _plot_binned_variance(
        c_grad_np[quality_mask], e_collision[quality_mask],
        xlabel="$C_{grad}$",
        title="Binned Damage Mean & Variance vs. Conflict\n(flat variance = decoder disambiguates successfully)",
        save_path="./damage_variance_analysis.png",
    )

    # 7. Main damage scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    plot_specs = [
        (c_grad_np,     e_collision, "$C_{grad}$ vs $e_{collision}$",     r_collision_cgrad),
        (c_eff_grad_np, e_collision, "$C_{eff-grad}$ vs $e_{collision}$", r_collision_ceffgrad),
        (c_grad_np,     e_model,     "$C_{grad}$ vs $e_{model}$",         r_model_cgrad),
    ]

    for ax, (scores, target, title, r) in zip(axes, plot_specs):
        m = quality_mask
        hb = ax.hexbin(scores[m], target[m], gridsize=50, cmap='YlOrRd', mincnt=1)
        fig.colorbar(hb, ax=ax, label='Point Density')

        _, _, bin_centers, bin_means = _compute_trendline(scores[m], target[m])
        valid = ~np.isnan(bin_means)
        ax.plot(bin_centers[valid], bin_means[valid], color='blue', lw=2, label=f'r={r:.3f}')
        ax.axhline(0, color='gray', lw=0.8, linestyle='--')

        ax.set_xlabel("Conflict Score")
        ax.set_ylabel("Error (m)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Conflict vs. Error Decomposition  (low_T - high_T)", fontsize=13)
    plt.tight_layout()
    plt.savefig("./collision_damage_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved → ./collision_damage_analysis.png")


def _plot_binned_variance(
    conflict: np.ndarray,
    damage: np.ndarray,
    xlabel: str,
    title: str,
    save_path: str,
    n_bins: int = 30,
):
    bins        = np.linspace(conflict.min(), conflict.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    digitized   = np.digitize(conflict, bins)

    bin_means = []
    bin_stds  = []
    valid_centers = []

    for i in range(1, len(bins)):
        vals = damage[digitized == i]
        if len(vals) < 10:
            continue
        bin_means.append(vals.mean())
        bin_stds.append(vals.std())
        valid_centers.append(bin_centers[i - 1])

    valid_centers = np.array(valid_centers)
    bin_means     = np.array(bin_means)
    bin_stds      = np.array(bin_stds)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(valid_centers, bin_means, color='blue', lw=2, label='Binned Mean')
    ax.fill_between(
        valid_centers,
        bin_means - bin_stds,
        bin_means + bin_stds,
        alpha=0.25, color='blue', label='±1 std'
    )
    ax.axhline(0, color='gray', lw=0.8, linestyle='--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Collision Damage (m)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved → {save_path}")


###############################################################
# Entry Point
###############################################################

if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # analyze_disambiguation(
    #     model_path=MODEL_PATH_HIGH_T,
    #     stats_path=STATS_PATH_HIGH_T,
    #     mesh_path=MESH_PATH_HIGH_T,
    #     gt_mesh_path=GT_MESH_PATH,
    #     device=device,
    # )
    
    T_values = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24]

    analyze_conflict_vs_T(
        T_values=T_values,
        results_dir="./results/mapping",
        gt_mesh_path=GT_MESH_PATH,
        device=device,
    )
    
    # analyze_collision_damage(
    #     model_path_high_T=MODEL_PATH_HIGH_T,
    #     stats_path_high_T=STATS_PATH_HIGH_T,
    #     mesh_path_high_T=MESH_PATH_HIGH_T,
    #     model_path_low_T=MODEL_PATH,
    #     stats_path_low_T=STATS_PATH,
    #     mesh_path_low_T=MESH_PATH,
    #     gt_mesh_path=GT_MESH_PATH,
    #     device=device,
    # )
    