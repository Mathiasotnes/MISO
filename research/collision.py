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


def plot_per_level_correlation(
    per_level_conflict: np.ndarray,   # (N, L)
    errors: np.ndarray,               # (N,)
    resolutions: list,
    save_path: str,
    error_threshold: float = 0.10,
):
    mask  = errors < error_threshold
    conf  = per_level_conflict[mask]   # (N', L)
    err   = errors[mask]
    L     = conf.shape[1]

    correlations = np.array([
        np.corrcoef(conf[:, l], err)[0, 1] for l in range(L)
    ])

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # ── Left: bar chart of per-level Pearson r ────────────────────────────────
    ax = axes[0]
    colors = ['tomato' if r > 0 else 'steelblue' for r in correlations]
    ax.bar(range(1, L + 1), correlations, color=colors)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlabel("Hash Level")
    ax.set_ylabel("Pearson r  (conflict vs. error)")
    ax.set_title("Per-Level Conflict–Error Correlation")
    ax.set_xticks(range(1, L + 1))
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate resolution on each bar
    for l, (r, res) in enumerate(zip(correlations, resolutions)):
        ax.text(l + 1, r + 0.005 * np.sign(r), f"N={int(res)}", ha='center',
                va='bottom' if r >= 0 else 'top', fontsize=7, rotation=45)

    # ── Right: scatter of |r| vs log(resolution) ─────────────────────────────
    ax = axes[1]
    log_res = np.log2(resolutions)
    ax.scatter(log_res, np.abs(correlations), c=correlations, cmap='RdBu_r',
               vmin=-max(abs(correlations)), vmax=max(abs(correlations)), s=60, zorder=3)
    ax.set_xlabel("log₂(Resolution)")
    ax.set_ylabel("|Pearson r|")
    ax.set_title("|Correlation| vs. Level Resolution")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Per-Level Hash Conflict vs. Geometric Error", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved per-level correlation plot → {save_path}")

    # Print table
    print("\n" + "=" * 55)
    print(f"{'Lvl':>4} | {'Res':>6} | {'Pearson r':>10} | {'|r|':>8}")
    print("-" * 55)
    for l, (r, res) in enumerate(zip(correlations, resolutions)):
        print(f"{l+1:>4} | {int(res):>6} | {r:>10.4f} | {abs(r):>8.4f}")
    print("=" * 55 + "\n")

    return correlations

@torch.no_grad()
def analyze_bin_level_correlation(
    tracker: CollisionTracker,
    verts_pred: np.ndarray,
    errors: np.ndarray,
    save_path: str,
    error_threshold: float = 0.10,
    device: str = "cuda",
):
    """
    Aggregates mesh vertices into hash bins and correlates bin-level
    mean conflict with bin-level mean error.
    """
    mask       = errors < error_threshold
    verts_filt = verts_pred[mask]
    errors_filt = errors[mask]

    lo = tracker.scene_bound[:, 0].cpu().numpy()
    hi = tracker.scene_bound[:, 1].cpu().numpy()
    x_norm = np.clip((verts_filt - lo) / (hi - lo), 0.0, 1.0)
    x_norm_t = torch.from_numpy(x_norm).float().to(device)

    R_dom = tracker.get_R_dom()  # (L, T)

    n_levels = tracker.n_levels
    all_r       = []
    all_res     = []
    all_n_bins  = []

    fig, axes = plt.subplots(2, n_levels // 2 + n_levels % 2, figsize=(24, 10))
    axes_flat = axes.flatten()

    for level_idx in range(n_levels):
        N_l = tracker.resolutions[level_idx].item()

        # Map each vertex to its nearest floor corner (single representative corner)
        x_scaled = x_norm_t * N_l
        x_floor  = torch.floor(x_scaled).long()  # (N, 3)

        # Hash the floor corner to get the bin each vertex "belongs to"
        from grid_opt.models.collision_tracker import spatial_hash
        bins = spatial_hash(x_floor, tracker.T).cpu().numpy()  # (N,)

        c_eff_bins  = tracker.C_eff[level_idx].cpu().numpy()   # (T,)
        r_dom_bins  = R_dom[level_idx].cpu().numpy()            # (T,)
        conflict_bins = 1.0 - r_dom_bins                        # (T,)

        # Aggregate: mean error per bin
        unique_bins = np.unique(bins)
        bin_mean_error    = []
        bin_mean_conflict = []
        bin_c_eff         = []
        bin_counts        = []

        for b in unique_bins:
            pts_in_bin = bins == b
            n_pts = pts_in_bin.sum()
            if n_pts < 5:  # skip bins with too few points for stable mean
                continue
            bin_mean_error.append(errors_filt[pts_in_bin].mean())
            bin_mean_conflict.append(conflict_bins[b])
            bin_c_eff.append(c_eff_bins[b])
            bin_counts.append(n_pts)

        if len(bin_mean_error) < 10:
            all_r.append(np.nan)
            all_res.append(N_l)
            all_n_bins.append(0)
            continue

        bin_mean_error    = np.array(bin_mean_error)
        bin_mean_conflict = np.array(bin_mean_conflict)
        bin_c_eff         = np.array(bin_c_eff)
        bin_counts        = np.array(bin_counts)

        r = np.corrcoef(bin_mean_conflict, bin_mean_error)[0, 1]
        all_r.append(r)
        all_res.append(N_l)
        all_n_bins.append(len(bin_mean_error))

        # Plot this level
        ax = axes_flat[level_idx]
        sc = ax.scatter(
            bin_mean_conflict, bin_mean_error,
            c=np.log1p(bin_c_eff), cmap='YlOrRd',
            s=np.sqrt(bin_counts) * 2, alpha=0.6, edgecolors='none'
        )
        fig.colorbar(sc, ax=ax, label='log(1 + C_eff)')

        # Trendline
        if len(bin_mean_conflict) > 2:
            z = np.polyfit(bin_mean_conflict, bin_mean_error, 1)
            xline = np.linspace(bin_mean_conflict.min(), bin_mean_conflict.max(), 100)
            ax.plot(xline, np.polyval(z, xline), color='blue', lw=2)

        ax.set_title(f"Level {level_idx+1}  (res={int(N_l)}, r={r:.3f}, bins={len(bin_mean_error)})")
        ax.set_xlabel("Bin Mean Conflict  (1 − R_dom)")
        ax.set_ylabel("Bin Mean Error (m)")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_levels, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.suptitle("Bin-Level Conflict vs. Mean Geometric Error", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved bin-level correlation plot → {save_path}")

    # Summary table
    print("\n" + "=" * 55)
    print(f"{'Lvl':>4} | {'Res':>6} | {'Pearson r':>10} | {'N bins':>8}")
    print("-" * 55)
    for l, (r, res, nb) in enumerate(zip(all_r, all_res, all_n_bins)):
        r_str = f"{r:>10.4f}" if not np.isnan(r) else f"{'N/A':>10}"
        print(f"{l+1:>4} | {int(res):>6} | {r_str} | {nb:>8}")
    print("=" * 55 + "\n")

    return np.array(all_r), np.array(all_res)

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
    
    bin_r, bin_res = analyze_bin_level_correlation(
        tracker       = tracker,
        verts_pred    = verts_pred,
        errors        = dist_p,
        save_path     = "./bin_level_correlation.png",
        device        = device,
    )
    
    # Compute per-level conflict: (N, L)
    per_level_t = torch.zeros(n, tracker.n_levels, device=device)
    with torch.no_grad():
        for i in range(0, n, BATCH):
            batch = verts_torch[i : i + BATCH]
            per_level_t[i : i + BATCH] = get_per_level_conflict(tracker, batch)

    per_level_np  = per_level_t.cpu().numpy()
    resolutions   = [tracker.resolutions[l].item() for l in range(tracker.n_levels)]

    correlations = plot_per_level_correlation(
        per_level_np, dist_p, resolutions,
        save_path="./per_level_correlation.png",
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
    most_predictive = int(np.argmax(np.abs(correlations)))
    print(f"  Most predictive level: {most_predictive + 1} "
          f"(res={int(resolutions[most_predictive])}, r={correlations[most_predictive]:.4f})")
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
    