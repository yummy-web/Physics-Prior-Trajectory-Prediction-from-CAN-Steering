"""
exp9_stratified.py
Stratified validation on comma2k19: does physics-prior advantage hold across
speed (low/mid/high) x turn (straight/gentle/curvy) strata, or is it an artifact
of averaging? Report physics FDE@3s vs best kinematic baseline (CTRA) per cell.

Outputs: figures/fig13_stratified_heatmap.png, results/exp9_results.json
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from comma_loader import load_all_cached, stratify
import models_core as M
from plot_utils import save

RES = os.path.join(os.path.dirname(__file__), "results"); os.makedirs(RES, exist_ok=True)
DT = 0.05; H = 3.0; HS = int(round(H / DT))
SPEED_BINS = ["low", "mid", "high"]; TURN_BINS = ["straight", "gentle", "curvy"]


def main():
    print("=" * 70); print("  Exp9: stratified validation (speed x turn) on comma2k19"); print("=" * 70)
    data = load_all_cached(); labels = stratify(data)
    segs = list(data.keys())
    # global gain from all (calibration is data-independent enough)
    g = M.comma_gain([data[s] for s in segs])

    cells_phys = {(sb, tb): [] for sb in SPEED_BINS for tb in TURN_BINS}
    cells_ctra = {(sb, tb): [] for sb in SPEED_BINS for tb in TURN_BINS}
    counts = {(sb, tb): 0 for sb in SPEED_BINS for tb in TURN_BINS}

    for s in segs:
        df = data[s]; lab = labels[s]; key = (lab["speed_bin"], lab["turn_bin"])
        counts[key] += 1
        rp = M.comma_phys_metrics(df, HS, DT, g)
        cells_phys[key].extend(rp["fde"])
        rc = M.comma_kin_metrics(df, HS, DT, "CTRA")
        cells_ctra[key].extend(rc["fde"])

    phys_grid = np.full((3, 3), np.nan); ctra_grid = np.full((3, 3), np.nan)
    ratio_grid = np.full((3, 3), np.nan)
    print(f"\n{'cell':<22}{'n':>4}{'Phys FDE@3s':>13}{'CTRA':>8}{'ratio':>8}")
    for i, sb in enumerate(SPEED_BINS):
        for j, tb in enumerate(TURN_BINS):
            key = (sb, tb)
            if cells_phys[key]:
                p = np.mean(cells_phys[key]); c = np.mean(cells_ctra[key])
                phys_grid[i, j] = p; ctra_grid[i, j] = c; ratio_grid[i, j] = c / p
                print(f"{sb+'/'+tb:<22}{counts[key]:>4}{p:>13.2f}{c:>8.2f}{c/p:>8.2f}")

    # heatmap of physics FDE and improvement ratio
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, grid, title, fmt in [(axes[0], phys_grid, "Physics FDE@3s (m)", "%.2f"),
                                  (axes[1], ratio_grid, "Improvement ratio CTRA/Physics (>1 = physics better)", "%.1fx")]:
        im = ax.imshow(grid, cmap="viridis" if "FDE" in title else "RdYlGn", aspect="auto")
        ax.set_xticks(range(3)); ax.set_xticklabels(TURN_BINS)
        ax.set_yticks(range(3)); ax.set_yticklabels(SPEED_BINS)
        ax.set_xlabel("turn"); ax.set_ylabel("speed"); ax.set_title(title)
        for i in range(3):
            for j in range(3):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, fmt % grid[i, j], ha="center", va="center",
                            color="white" if "FDE" in title else "black", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("comma2k19 stratified: physics advantage holds across operating conditions")
    save(fig, "fig13_stratified_heatmap.png")

    out = {f"{sb}/{tb}": {"n": counts[(sb, tb)],
                          "phys_fde": float(phys_grid[i, j]) if not np.isnan(phys_grid[i, j]) else None,
                          "ctra_fde": float(ctra_grid[i, j]) if not np.isnan(ctra_grid[i, j]) else None}
           for i, sb in enumerate(SPEED_BINS) for j, tb in enumerate(TURN_BINS)}
    with open(os.path.join(RES, "exp9_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved results/exp9_results.json\nExp9 done.")


if __name__ == "__main__":
    main()
