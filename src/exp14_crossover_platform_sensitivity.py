"""
exp14_crossover_platform_sensitivity.py

Sensitivity check for the data-scarcity crossover. The paper reports concrete
crossover values on the comma2k19 passenger-car protocol. This script tests how
the same training-budget procedure behaves on another public platform
(nuScenes CAN) and across random splits.

The goal is not to claim a vehicle-independent threshold. It is to make the
threshold interpretation explicit: the mechanism is transferable, while the
numerical crossover is platform/protocol specific.

Outputs:
  figures/fig18_crossover_platform_sensitivity.png
  results/exp14_results.json
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from comma_loader import load_all_cached as load_comma
from nuscenes_loader import load_all_cached as load_nusc
import models_core as M
from plot_utils import save


RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
os.makedirs(RES, exist_ok=True)


def median_dt(segs):
    dts = []
    for df in segs[: min(20, len(segs))]:
        t = df["t"].values
        if len(t) > 2:
            dts.append(np.median(np.diff(t)))
    return float(np.median(dts))


def build_direct(segs, hs, step):
    X, Y = [], []
    for df in segs:
        x = df["x"].values
        y = df["y"].values
        yaw = np.radians(df["yaw_unwrap"].values)
        v = df["obs_v"].values
        yr = np.radians(df["yaw_rate"].values)
        steer = df["steer"].values
        n = len(df)
        for i0 in range(0, n - hs, step):
            fut = slice(i0, i0 + hs)
            feat = [
                v[i0],
                yr[i0],
                float(np.mean(v[fut])),
                float(np.mean(steer[fut])),
                float(np.std(steer[fut])),
            ]
            dx = x[i0 + hs] - x[i0]
            dy = y[i0 + hs] - y[i0]
            c, s = np.cos(-yaw[i0]), np.sin(-yaw[i0])
            X.append(feat)
            Y.append([c * dx - s * dy, s * dx + c * dy])
    return np.asarray(X), np.asarray(Y)


def eval_direct(test, scaler, model, hs, step):
    fdes = []
    for df in test:
        x = df["x"].values
        y = df["y"].values
        yaw = np.radians(df["yaw_unwrap"].values)
        v = df["obs_v"].values
        yr = np.radians(df["yaw_rate"].values)
        steer = df["steer"].values
        n = len(df)
        for i0 in range(0, n - hs, step):
            fut = slice(i0, i0 + hs)
            feat = np.asarray(
                [[v[i0], yr[i0], np.mean(v[fut]), np.mean(steer[fut]), np.std(steer[fut])]]
            )
            lx, ly = model.predict(scaler.transform(feat))[0]
            c, s = np.cos(yaw[i0]), np.sin(yaw[i0])
            px = x[i0] + c * lx - s * ly
            py = y[i0] + s * lx + c * ly
            fdes.append(np.hypot(px - x[i0 + hs], py - y[i0 + hs]))
    return float(np.mean(fdes))


def platform_run(name, data, test_size, budgets, seeds):
    segs = list(data.values())
    dt = median_dt(segs)
    hs = int(round(3.0 / dt))
    step = max(5, int(round(0.5 / dt)))
    duration_per_segment = float(np.median([df["t"].iloc[-1] - df["t"].iloc[0] for df in segs]))
    out = {
        "dt": dt,
        "horizon_steps": hs,
        "step": step,
        "duration_per_segment_median_s": duration_per_segment,
        "seeds": {},
    }
    print(f"\n[{name}] n={len(segs)} dt={dt:.3f}s median_duration={duration_per_segment:.1f}s")

    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(segs))
        test_idx = idx[:test_size]
        pool_idx = idx[test_size:]
        test = [segs[i] for i in test_idx]
        pool = [segs[i] for i in pool_idx]
        g = M.comma_gain(pool)
        phys = float(np.mean([np.mean(M.comma_phys_metrics(df, hs, dt, g)["fde"]) for df in test]))

        curve = []
        for k in budgets:
            if k > len(pool):
                continue
            train = pool[:k]
            X, Y = build_direct(train, hs, step)
            scaler = StandardScaler().fit(X)
            model = MLPRegressor(
                hidden_layer_sizes=(48, 24),
                max_iter=500,
                alpha=1e-3,
                random_state=seed,
                early_stopping=True,
            ).fit(scaler.transform(X), Y)
            fde = eval_direct(test, scaler, model, hs, step)
            seconds = float(k * duration_per_segment)
            curve.append({"segments": int(k), "seconds": seconds, "fde": fde})
            print(f"  seed={seed} k={k:4d} ({seconds:6.0f}s): data={fde:.3f} phys={phys:.3f}")

        crossing = None
        for item in curve:
            if item["fde"] < phys:
                crossing = item["seconds"]
                break
        out["seeds"][str(seed)] = {"physics_fde": phys, "curve": curve, "first_measured_crossing_s": crossing}

    crossings = [
        v["first_measured_crossing_s"]
        for v in out["seeds"].values()
        if v["first_measured_crossing_s"] is not None
    ]
    out["summary"] = {
        "physics_fde_mean": float(np.mean([v["physics_fde"] for v in out["seeds"].values()])),
        "crossing_observed_count": int(len(crossings)),
        "crossing_s_median": float(np.median(crossings)) if crossings else None,
        "crossing_s_min": float(np.min(crossings)) if crossings else None,
        "crossing_s_max": float(np.max(crossings)) if crossings else None,
    }
    return out


def main():
    print("=" * 70)
    print("  Exp14: crossover platform/protocol sensitivity")
    print("=" * 70)
    comma = load_comma(verbose=False)
    nusc = load_nusc(verbose=False)

    results = {
        "comma2k19": platform_run("comma2k19", comma, test_size=30, budgets=[8, 24, 60, 120], seeds=[0, 1, 2]),
        "nuscenes": platform_run("nuScenes", nusc, test_size=100, budgets=[8, 24, 60, 120, 240], seeds=[0, 1, 2]),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for ax, name in zip(axes, ["comma2k19", "nuscenes"]):
        run = results[name]
        for seed, vals in run["seeds"].items():
            xs = [c["seconds"] for c in vals["curve"]]
            ys = [c["fde"] for c in vals["curve"]]
            ax.plot(xs, ys, "o-", alpha=0.75, label=f"seed {seed}")
        phys = run["summary"]["physics_fde_mean"]
        ax.axhline(phys, color="r", ls="--", label=f"physics mean {phys:.2f} m")
        ax.set_xscale("log")
        ax.set_xlabel("Training data (s, log)")
        ax.set_ylabel("FDE @ 3 s (m)")
        ax.legend(fontsize=9)
    save(fig, "fig18_crossover_platform_sensitivity.png")

    path = os.path.join(RES, "exp14_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {path}\nExp14 done.")


if __name__ == "__main__":
    main()
