"""
exp16_loader_identification_diagnostics.py

Diagnostic analysis for the articulated loader's lower steering-to-curvature R2.
The loader is the most industrially relevant and most challenging platform in
the paper. This script reports whether its lower median R2 still supports a
usable calibration backbone.

Outputs:
  figures/fig20_loader_identification_diagnostics.png
  results/exp16_results.json
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from loader_data_loader import load_all
from plot_utils import save


RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
os.makedirs(RES, exist_ok=True)


def loader_center(df):
    mask = (np.abs(df["yaw_rate"].values) < 3) & (df["obs_v"].values > 1.0)
    if mask.sum() > 5:
        return float(np.median(df["motor_angle"].values[mask]))
    return float(np.median(df["motor_angle"].values))


def fit_segment(df):
    c = loader_center(df)
    m = df["obs_v"].values > 0.5
    x = (df["motor_angle"].values - c)[m]
    y = df["curvature"].values[m]
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() > 5:
        limit = np.nanpercentile(np.abs(y[ok]), 97)
        ok = ok & (np.abs(y) < limit)
    x = x[ok]
    y = y[ok]
    slope, intercept, r, p, se = stats.linregress(x, y)
    pred = slope * x + intercept
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    iqr = float(np.percentile(y, 75) - np.percentile(y, 25))
    return {
        "n": int(len(x)),
        "gain": float(slope),
        "intercept": float(intercept),
        "R2": float(r**2),
        "p_value": float(p),
        "steer_std": float(np.std(x)),
        "curvature_std": float(np.std(y)),
        "curvature_iqr": iqr,
        "rmse": rmse,
        "nrmse_iqr": float(rmse / iqr) if iqr > 0 else None,
        "mean_speed": float(np.mean(df["obs_v"].values[m])),
        "mean_abs_yaw_rate": float(np.mean(np.abs(df["yaw_rate"].values[m]))),
    }


def pooled_fit(data):
    xs, ys = [], []
    for df in data.values():
        c = loader_center(df)
        m = df["obs_v"].values > 0.5
        x = (df["motor_angle"].values - c)[m]
        y = df["curvature"].values[m]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 5:
            limit = np.nanpercentile(np.abs(y[ok]), 97)
            ok = ok & (np.abs(y) < limit)
        xs.append(x[ok])
        ys.append(y[ok])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    slope, intercept, r, p, se = stats.linregress(x, y)
    pred = slope * x + intercept
    return {
        "n": int(len(x)),
        "gain": float(slope),
        "intercept": float(intercept),
        "R2": float(r**2),
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "curvature_std": float(np.std(y)),
        "curvature_iqr": float(np.percentile(y, 75) - np.percentile(y, 25)),
    }


def main():
    print("=" * 70)
    print("  Exp16: articulated-loader steering identification diagnostics")
    print("=" * 70)
    data = load_all()
    per_session = {name: fit_segment(df) for name, df in data.items()}
    pooled = pooled_fit(data)

    r2s = np.asarray([v["R2"] for v in per_session.values()])
    frac_gt07 = float(np.mean(r2s > 0.7))
    summary = {
        "n_sessions": int(len(per_session)),
        "median_R2": float(np.median(r2s)),
        "mean_R2": float(np.mean(r2s)),
        "frac_gt07": frac_gt07,
        "pooled": pooled,
    }

    print(f"median R2={summary['median_R2']:.3f}, pooled R2={pooled['R2']:.3f}, R2>0.7={100*frac_gt07:.0f}%")
    for name, vals in per_session.items():
        print(
            f"  {name}: R2={vals['R2']:.3f}, n={vals['n']}, "
            f"curv_std={vals['curvature_std']:.4g}, rmse={vals['rmse']:.4g}, speed={vals['mean_speed']:.2f}"
        )

    labels = [name[-8:] for name in per_session]
    r2_vals = [per_session[name]["R2"] for name in per_session]
    curv_std = [per_session[name]["curvature_std"] for name in per_session]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, r2_vals, color="tab:blue")
    axes[0].axhline(0.7, color="gray", ls=":")
    axes[0].set_ylabel("R2")
    axes[0].set_xlabel("Loader session")
    axes[1].bar(labels, curv_std, color="tab:orange")
    axes[1].set_ylabel("Curvature std. (1/m)")
    axes[1].set_xlabel("Loader session")
    save(fig, "fig20_loader_identification_diagnostics.png")

    path = os.path.join(RES, "exp16_results.json")
    with open(path, "w") as f:
        json.dump({"per_session": per_session, "summary": summary}, f, indent=2)
    print(f"\nsaved {path}\nExp16 done.")


if __name__ == "__main__":
    main()
