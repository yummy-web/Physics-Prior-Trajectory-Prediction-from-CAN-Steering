"""
exp5_ablation.py
Physics-model component ablation (loader).

We turn components on/off and measure FDE to show each is necessary:
  Full        : steering integration + speed model + warmup center calibration
  -Steer      : drop steering term (=> straight-line, CV-like)
  -SpeedModel : replace CAN speed model with a global constant speed
  -Warmup     : use a fixed (population) articulation center instead of per-seg warmup
  -Unwrap     : use raw (wrapped) encoder value instead of unwrapped

Outputs: figures/fig8_ablation.png, results/exp5_results.json
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from loader_data_loader import load_all as load_loader
import models_core as M
from plot_utils import save

RES = os.path.join(os.path.dirname(__file__), "results"); os.makedirs(RES, exist_ok=True)
DT = 0.1; HORIZONS = [1.0, 2.0, 3.0, 5.0]


def phys_fde_variant(df, hs, g, c, coef, glob_speed, variant):
    x = df["x"].values; y = df["y"].values; yaw = np.radians(df["yaw_unwrap"].values)
    fb = df["footboard"].values; od = df["oil_door"].values
    ma = df["motor_angle"].values; ma_raw = df["motor_angle_raw"].values
    n = len(df); out = []
    for i0 in range(0, n - hs, 2):
        px, py, th = x[i0], y[i0], yaw[i0]
        for k in range(hs):
            j = min(i0 + k, n - 1)
            if variant == "-SpeedModel":
                v = glob_speed
            else:
                v = max(coef[0] + coef[1] * fb[j] + coef[2] * od[j], 0.0)
            if variant == "-Steer":
                dth = 0.0
            elif variant == "-Unwrap":
                dth = g * (ma_raw[j] - (c % 65530)) * DT
            else:
                dth = g * (ma[j] - c) * DT
            th += dth
            px += v * np.cos(th) * DT; py += v * np.sin(th) * DT
        out.append(np.hypot(px - x[i0 + hs], py - y[i0 + hs]))
    return np.array(out)


def main():
    print("=" * 70); print("  Experiment 5: physics-model ablation (loader LOSO)"); print("=" * 70)
    data = load_loader(); segs = list(data.keys())
    variants = ["Full", "-Steer", "-SpeedModel", "-Warmup", "-Unwrap"]
    per = {v: {H: [] for H in HORIZONS} for v in variants}
    glob_speed = float(np.mean([data[s]["obs_v"].mean() for s in segs]))
    for test in segs:
        train = [s for s in segs if s != test]
        g = M.loader_gain(data, train); coef = M.loader_speed_coef(data, train)
        c_warm = M.loader_center(data[test], warmup=True, dt=DT)
        c_pop = np.median([M.loader_center(data[s], warmup=False, dt=DT) for s in train])
        df = data[test]
        for v in variants:
            c = c_pop if v == "-Warmup" else c_warm
            for H in HORIZONS:
                hs = int(round(H / DT))
                fde = phys_fde_variant(df, hs, g, c, coef, glob_speed, v)
                per[v][H].append(float(np.mean(fde)))
    summary = {v: {H: M.bootstrap_ci(per[v][H])[0] for H in HORIZONS} for v in variants}
    print(f"\n{'variant':<14}" + "".join(f"{f'{H}s':>9}" for H in HORIZONS))
    for v in variants:
        print(f"{v:<14}" + "".join(f"{summary[v][H]:>9.2f}" for H in HORIZONS))

    fig, ax = plt.subplots(figsize=(8, 5))
    for v in variants:
        style = "s-" if v == "Full" else "o--"
        lw = 2.5 if v == "Full" else 1.2
        ax.plot(HORIZONS, [summary[v][H] for H in HORIZONS], style, lw=lw, label=v)
    ax.set_xlabel("Prediction horizon (s)"); ax.set_ylabel("FDE (m)")
    ax.set_title("Physics-model ablation (loader): every component contributes")
    ax.legend(); save(fig, "fig8_ablation.png")

    with open(os.path.join(RES, "exp5_results.json"), "w") as f:
        json.dump({v: {str(H): summary[v][H] for H in HORIZONS} for v in variants}, f, indent=2)
    print("\nsaved results/exp5_results.json\nExperiment 5 done.")


if __name__ == "__main__":
    main()
