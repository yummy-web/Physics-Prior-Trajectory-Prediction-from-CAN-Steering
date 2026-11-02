"""Generate the paper figures from frozen, public result files.

The plotting code deliberately reads only canonical result artefacts.  It does
not retrain a model, select a hyperparameter, or access raw/private data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results" / "canonical"
DEFAULT_OUT = ROOT / "figures" / "canonical"


def save(fig: plt.Figure, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def fig_shrinkage(out: Path) -> None:
    p = RESULTS / "exp_lambda_analytic_nusc_results.json"
    obj = json.loads(p.read_text(encoding="utf-8"))
    rows = obj.get("per_N", obj.get("results", obj))
    ns = [3, 5, 10, 20]
    gains, lows, highs = [], [], []
    for n in ns:
        r = rows[str(n)]
        stat = (r.get("scene_clustered_shrunk_vs_full")
                or r.get("scene_level_shrunk_vs_full")
                or r.get("paired_shrunk_vs_full_scene")
                or r.get("scene_clustered_inference"))
        if stat is None:
            # Stable canonical keys used by the currently frozen result.
            gain = r.get("gain_shrunk_vs_full_mean", r["shrunk_gain_vs_full"])
            ci = r.get("gain_shrunk_vs_full_scene_CI95", r["shrunk_gain_CI95_scene"])
        else:
            gain = (stat.get("mean_gain_m") or stat.get("mean_gain_ref_minus_candidate_m")
                    or stat.get("mean"))
            ci = (stat.get("CI95_m") or stat.get("paired_scene_bootstrap_CI95_m")
                  or stat.get("ci"))
        gains.append(float(gain)); lows.append(float(ci[0])); highs.append(float(ci[1]))
    yerr = np.vstack([np.asarray(gains) - lows, np.asarray(highs) - gains])
    fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
    ax.errorbar(ns, gains, yerr=yerr, marker="o", color="#B50909",
                linewidth=2, capsize=4, label="nuScenes scene-level gain")
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel("Training-group budget N")
    ax.set_ylabel("FDE gain over full residual (m)")
    ax.set_title("FDE gain from residual shrinkage across training budgets")
    style_axis(ax); ax.legend(frameon=False)
    save(fig, out, "Fig_3_nuscenes_shrinkage_gain")


def fig_transfer(out: Path) -> None:
    p = RESULTS / "exp_lambda_transfer_results.json"
    obj = json.loads(p.read_text(encoding="utf-8"))
    rows = obj.get("per_N", obj)
    ns = [3, 5, 10, 20]
    def vals(*keys):
        ans = []
        for n in ns:
            r = rows[str(n)]
            for k in keys:
                if k in r:
                    ans.append(float(r[k])); break
            else:
                raise KeyError(f"None of {keys} in N={n}: {sorted(r)}")
        return ans
    universal = vals("universal_fde_mean", "fde_universal", "universal_lambda_1_fde")
    source = vals("source_fde_mean", "fde_source", "source_schedule_fde")
    target = vals("target_val_fde_mean", "fde_target_val", "target_validation_fde")
    oracle = vals("oracle_fde_mean", "fde_oracle", "target_oracle_fde", "fde_target_oracle")
    fig, ax = plt.subplots(figsize=(5.5, 3.35), constrained_layout=True)
    for y, label, color in [
        (universal, "Universal lambda=1", "#A9A9A9"),
        (source, "comma2k19 source schedule", "#2E75B6"),
        (target, "Target validation", "#70AD47"),
        (oracle, "Target oracle", "#C00000"),
    ]:
        ax.plot(ns, y, marker="o", linewidth=2, label=label, color=color)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel("Training-group budget N")
    ax.set_ylabel("nuScenes FDE (m)")
    ax.set_title("Cross-platform transfer: direction transfers, level needs recalibration")
    style_axis(ax); ax.legend(frameon=False, fontsize=8)
    save(fig, out, "Fig_5_cross_platform_transfer")


def fig_cristal_grouped(out: Path) -> None:
    p = RESULTS / "crisal_external_validation" / "crisal_grouped_per_acquisition.csv"
    df = pd.read_csv(p)
    method_map = {
        "INS_velocity_ZOH": ("INS velocity", "#A6A6A6"),
        "CTRV_fit_CAN_steering_gain": ("CTRV fitted", "#70AD47"),
        "Residual_MLP_lambda_1": ("Full residual", "#2E75B6"),
        "Residual_MLP_validation_lambda_shrunk_kappa_acq_0.5": (r"$\kappa_{acq}=0.5$", "#C00000"),
    }
    ids = list(dict.fromkeys(df["acquisition_id"]))
    short = []
    for s in ids:
        if "anneau_cite_scientifique_1" in s: short.append("ACS-1")
        elif "anneau_cite_scientifique_2" in s: short.append("ACS-2")
        else: short.append(next((c for c in ["bleu", "jaune", "rouge", "vert", "violet"] if c in s), s))
    x = np.arange(len(ids)); width = 0.19
    fig, ax = plt.subplots(figsize=(7.0, 3.2), constrained_layout=True)
    for j, (method, (label, color)) in enumerate(method_map.items()):
        sub = df[df["method"] == method].set_index("acquisition_id")
        y = [float(sub.loc[i, "fde_3s_m"]) for i in ids]
        ax.bar(x + (j - 1.5) * width, y, width, label=label, color=color)
    ax.set_xticks(x, short); ax.set_ylabel("3 s FDE (m)")
    ax.set_title("CRIStAL/LOCSP grouped external validation across seven acquisitions")
    ax.set_ylim(0, max(3.2, ax.get_ylim()[1])); style_axis(ax)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    save(fig, out, "Fig_6_crisal_grouped_external_3s")


def fig_cristal_rollout(out: Path, endpoints_path: Path | None) -> None:
    """Draw a representative endpoint rollout exported from the experiment.

    Endpoint coordinates are never reconstructed from pixels.  The JSON must be
    exported from the grouped CRIStAL/LOCSP experiment (see
    export_cristal_representative_rollout.py).
    """
    path = endpoints_path or (RESULTS / "crisal_external_validation" /
                              "representative_endpoint_rollout.json")
    endpoints = json.loads(path.read_text(encoding="utf-8"))
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(8.2, 3.6),
                                  gridspec_kw={"width_ratios": [1.65, 1]}, constrained_layout=True)
    styles = [
        ("reference", "Reference endpoints", "black", "-"),
        ("ctrv", "CTRV prior", "#70AD47", "--"),
        ("full", "Full residual", "#2E75B6", "-"),
        ("shrunk", f"Shrunk residual (λ={endpoints.get('deployment_lambda', endpoints['validation_lambda']):.2f})", "#C00000", "-"),
    ]
    for key, label, color, ls in styles:
        a = np.asarray(endpoints[key], dtype=float)
        ax.plot(a[:, 0], a[:, 1], marker="o", color=color, linestyle=ls, label=label)
        axz.plot([1, 2, 3], a[:, 1], marker="o", color=color, linestyle=ls, label=label)
    ax.set_xlabel("Local longitudinal coordinate (m)"); ax.set_ylabel("Local lateral coordinate (m)")
    ax.set_title("Endpoint rollout"); style_axis(ax); ax.legend(frameon=False, fontsize=8)
    axz.set_xticks([1, 2, 3]); axz.set_xlabel("Horizon (s)"); axz.set_ylabel("Lateral endpoint coordinate (m)")
    axz.set_title("Lateral detail"); style_axis(axz)
    fig.suptitle("Representative CRIStAL/LOCSP three-horizon endpoint rollout")
    save(fig, out, "Fig_7_representative_crisal_endpoint_rollout")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--rollout-json", type=Path)
    args = ap.parse_args()
    fig_shrinkage(args.output_dir)
    fig_transfer(args.output_dir)
    fig_cristal_grouped(args.output_dir)
    fig_cristal_rollout(args.output_dir, args.rollout_json)
    print(f"Wrote manuscript figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
