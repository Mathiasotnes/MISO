import torch
import numpy as np
import matplotlib
import json
from os.path import join
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from grid_opt.utils.utils_eval import nn_correspondance, sample_points_from_mesh
from grid_opt.models.collision_tracker import CollisionTracker


###############################################################
# Constants / Configuration
###############################################################

STATS_PATH          = "./results/mapping/collision_stats.pt"
MESH_PATH           = "./results/mapping/hash_pred_mesh.ply"
GT_MESH_PATH        = "../../data/ScanNet/scans/scene0000_00/scene0000_00_vh_clean.ply"
MODEL_PATH          = "./results/mapping/hash_grid.pth"


###############################################################
# Helpers
###############################################################

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
    """ Analyzes the C_grad stats. It tells us that it shows that the conflict level is very uniform across space. """
    print("Loading model and collision statistics...")
    hash_grid = torch.load(model_path, map_location=device)
    tracker = CollisionTracker.load(
        path=stats_path,
        encoding=hash_grid.encoding,
        scene_bound=hash_grid.bound,
        device=device,
    )
    tracker.print_summary()

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

        # Average C_eff across all layers and hash table entries
        c_eff_avg = float(tracker.C_eff.mean().item())

        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        rows.append({
            'T'          : T,
            'c_grad_avg' : c.mean(),
            'c_grad_std' : c.std(),
            'c_eff_avg'  : c_eff_avg,
            'chamfer_l2' : metrics.get('Chamfer_L2 (cm)', float('nan')),
            'f_score'    : metrics.get('F-score (%)',     float('nan')),
            'precision'  : metrics.get('Precision (%)',   float('nan')),
            'recall'     : metrics.get('Recall (%)',      float('nan')),
        })

        del hash_grid, tracker, verts_t, c_grad_t
        torch.cuda.empty_cache()

    # Print table
    w = 130
    print("\n" + "=" * w)
    print(
        f"{'T':>4} | {'2^T':>12} | {'C_grad avg':>12} | {'C_grad std':>12} | {'C_eff avg':>12} | "
        f"{'Chamfer-L2':>12} | {'F-Score':>12} | {'Precision':>12} | {'Recall':>12}"
    )
    print("-" * w)
    for r in rows:
        print(
            f"{r['T']:>4} | {2**r['T']:>12,} | {r['c_grad_avg']:>12.2f} | {r['c_grad_std']:>12.4f} | "
            f"{r['c_eff_avg']:>12.2f} | {r['chamfer_l2']:>12.2f} | {r['f_score']:>12.2f} | "
            f"{r['precision']:>12.2f} | {r['recall']:>12.2f}"
        )
    print("=" * w + "\n")

    _plot_conflict_vs_metrics(rows, save_path="./conflict_vs_metrics_linear.png")
    return rows

def _plot_conflict_vs_metrics(rows: list, save_path: str):
    """ This plotting is only to see how the relationship is between C_grad and the reconstruction metrics. """
    T_vals      = np.array([r['T']          for r in rows])
    c_means     = np.array([r['mean']        for r in rows])
    chamfer     = np.array([r['chamfer_l2']  for r in rows])
    f_scores    = np.array([r['f_score']     for r in rows])

    r_chamfer = np.corrcoef(c_means, chamfer)[0, 1]
    r_fscore  = np.corrcoef(c_means, f_scores)[0, 1]

    print("\n" + "=" * 50)
    print("C_grad mean vs. Reconstruction Metrics")
    print(f"  r(C_grad, Chamfer-L2) : {r_chamfer:.4f}")
    print(f"  r(C_grad, F-Score)    : {r_fscore:.4f}")
    print("=" * 50 + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Top left: C_grad mean and Chamfer-L2 vs T ────────────────────────────
    ax = axes[0, 0]
    ax2 = ax.twinx()
    ax.plot(T_vals, c_means,  'o-', color='steelblue',  label='C_grad mean')
    ax2.plot(T_vals, chamfer, 's--', color='tomato',    label='Chamfer-L2')
    ax.set_xlabel("$\log_2(T)$")
    ax.set_ylabel("C_grad mean",   color='steelblue')
    ax2.set_ylabel("Chamfer-L2 (cm)", color='tomato')
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='tomato')
    ax.set_title("C_grad & Chamfer-L2 vs. T")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax.grid(True, alpha=0.3)

    # ── Top right: C_grad mean and F-Score vs T ───────────────────────────────
    ax = axes[0, 1]
    ax2 = ax.twinx()
    ax.plot(T_vals, c_means,   'o-', color='steelblue', label='C_grad mean')
    ax2.plot(T_vals, f_scores, 's--', color='seagreen',  label='F-Score')
    ax.set_xlabel("$\log_2(T)$")
    ax.set_ylabel("C_grad mean",  color='steelblue')
    ax2.set_ylabel("F-Score (%)", color='seagreen')
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='seagreen')
    ax.set_title("C_grad & F-Score vs. T")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax.grid(True, alpha=0.3)

    # ── Bottom left: C_grad mean vs Chamfer-L2 scatter ───────────────────────
    ax = axes[1, 0]
    ax.scatter(c_means, chamfer, c=T_vals, cmap='viridis', s=60, zorder=3)
    for r in rows:
        ax.annotate(f"T={r['T']}", (r['mean'], r['chamfer_l2']),
                    textcoords="offset points", xytext=(5, 3), fontsize=7)
    z = np.polyfit(c_means, chamfer, 1)
    xline = np.linspace(c_means.min(), c_means.max(), 100)
    ax.plot(xline, np.polyval(z, xline), color='tomato', lw=1.5, linestyle='--')
    ax.set_xlabel("C_grad mean")
    ax.set_ylabel("Chamfer-L2 (cm)")
    ax.set_title(f"C_grad vs Chamfer-L2  (r={r_chamfer:.3f})")
    ax.grid(True, alpha=0.3)

    # ── Bottom right: C_grad mean vs F-Score scatter ─────────────────────────
    ax = axes[1, 1]
    sc = ax.scatter(c_means, f_scores, c=T_vals, cmap='viridis', s=60, zorder=3)
    fig.colorbar(sc, ax=ax, label='$\log_2(T)$')
    for r in rows:
        ax.annotate(f"T={r['T']}", (r['mean'], r['f_score']),
                    textcoords="offset points", xytext=(5, 3), fontsize=7)
    z = np.polyfit(c_means, f_scores, 1)
    xline = np.linspace(c_means.min(), c_means.max(), 100)
    ax.plot(xline, np.polyval(z, xline), color='seagreen', lw=1.5, linestyle='--')
    ax.set_xlabel("C_grad mean")
    ax.set_ylabel("F-Score (%)")
    ax.set_title(f"C_grad vs F-Score  (r={r_fscore:.3f})")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Scene-Level Hash Conflict vs. Reconstruction Quality (linear decoder)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved → {save_path}")
    

###############################################################
# Entry Point
###############################################################

if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Assumes that these models exists on disk. We can only fit one set of T=10..24 models at a time on UCSD RC. This takes roughly 30GB.
    T_values = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

    analyze_conflict_vs_T(
        T_values=T_values,
        results_dir="./results/mapping",
        device=device,
    )
    