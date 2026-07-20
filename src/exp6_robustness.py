"""
exp6_robustness.py
CAN signal robustness: physics prior vs data-driven MLP under degraded CAN.

We corrupt the CAN steering/speed signals at prediction time with:
  - Gaussian noise (std as fraction of signal range)
  - Dropout (held-last-value for a fraction of samples)
  - Latency (shift signal by tau seconds)
and measure FDE degradation. Hypothesis: physics prior degrades more gracefully
than the data-driven MLP because its structure constrains predictions.

Runs on comma2k19 (statistically solid). Outputs:
  figures/fig9_robustness.png, results/exp6_results.json
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from comma_loader import load_all_cached
import models_core as M
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from plot_utils import save

RES = os.path.join(os.path.dirname(__file__), "results"); os.makedirs(RES, exist_ok=True)
DT = 0.05; H = 3.0; HS = int(round(H / DT))
RNG = np.random.default_rng(0)


def corrupt(sig, mode, level, rng):
    s = sig.copy()
    if mode == "noise":
        rng_span = np.std(sig) + 1e-6
        s = s + rng.normal(0, level * rng_span, size=s.shape)
    elif mode == "dropout":
        mask = rng.random(len(s)) < level
        last = s[0]
        for i in range(len(s)):
            if mask[i]:
                s[i] = last
            else:
                last = s[i]
    elif mode == "latency":
        shift = int(round(level / DT))
        if shift > 0:
            s = np.concatenate([np.full(shift, s[0]), s[:-shift]])
    return s


def build_direct(segs):
    X, Y = [], []
    for df in segs:
        n = len(df)
        for i0 in range(0, n - HS, 5):
            fut = slice(i0, i0 + HS)
            feat = [df["obs_v"].values[i0], np.radians(df["yaw_rate"].values[i0]),
                    np.mean(df["obs_v"].values[fut]), np.mean(df["steer"].values[fut]),
                    np.std(df["steer"].values[fut])]
            yaw = np.radians(df["yaw_unwrap"].values[i0])
            dx = df["x"].values[i0 + HS] - df["x"].values[i0]; dy = df["y"].values[i0 + HS] - df["y"].values[i0]
            c, s = np.cos(-yaw), np.sin(-yaw)
            X.append(feat); Y.append([c * dx - s * dy, s * dx + c * dy])
    return np.array(X), np.array(Y)


def phys_fde_corrupt(df, g, mode, level, rng):
    x = df["x"].values; y = df["y"].values; yaw = np.radians(df["yaw_unwrap"].values)
    steer = corrupt(df["steer"].values, mode, level, rng)
    v_arr = df["obs_v"].values
    n = len(df); out = []
    for i0 in range(0, n - HS, 5):
        px, py, th = x[i0], y[i0], yaw[i0]
        for k in range(HS):
            j = min(i0 + k, n - 1)
            th += g * steer[j] * DT
            px += v_arr[j] * np.cos(th) * DT; py += v_arr[j] * np.sin(th) * DT
        out.append(np.hypot(px - x[i0 + HS], py - y[i0 + HS]))
    return float(np.mean(out))


def mlp_fde_corrupt(df, sc, mdl, mode, level, rng):
    steer = corrupt(df["steer"].values, mode, level, rng)
    x = df["x"].values; y = df["y"].values
    n = len(df); out = []
    for i0 in range(0, n - HS, 5):
        fut = slice(i0, i0 + HS)
        feat = np.array([[df["obs_v"].values[i0], np.radians(df["yaw_rate"].values[i0]),
                np.mean(df["obs_v"].values[fut]), np.mean(steer[fut]), np.std(steer[fut])]])
        lx, ly = mdl.predict(sc.transform(feat))[0]
        yaw = np.radians(df["yaw_unwrap"].values[i0]); c, s = np.cos(yaw), np.sin(yaw)
        ddx = c * lx - s * ly; ddy = s * lx + c * ly
        out.append(np.hypot((x[i0] + ddx) - x[i0 + HS], (y[i0] + ddy) - y[i0 + HS]))
    return float(np.mean(out))


def main():
    print("=" * 70); print("  Experiment 6: CAN robustness (physics vs data-driven)"); print("=" * 70)
    data = load_all_cached(); segs = list(data.values())
    test = segs[-30:]; train = segs[:80]
    g = M.comma_gain(train)
    X, Y = build_direct(train); sc = StandardScaler().fit(X)
    mdl = MLPRegressor(hidden_layer_sizes=(48, 24), max_iter=1000, alpha=1e-3,
                       random_state=0, early_stopping=True).fit(sc.transform(X), Y)

    modes = {"noise": [0.0, 0.1, 0.2, 0.4, 0.8], "dropout": [0.0, 0.1, 0.2, 0.4, 0.6],
             "latency": [0.0, 0.1, 0.2, 0.3, 0.5]}
    out = {}
    for mode, levels in modes.items():
        out[mode] = {"levels": levels, "physics": [], "mlp": []}
        for lv in levels:
            rng = np.random.default_rng(1)
            pf = np.mean([phys_fde_corrupt(df, g, mode, lv, rng) for df in test])
            rng = np.random.default_rng(1)
            mf = np.mean([mlp_fde_corrupt(df, sc, mdl, mode, lv, rng) for df in test])
            out[mode]["physics"].append(float(pf)); out[mode]["mlp"].append(float(mf))
        print(f"\n[{mode}] levels {levels}")
        print(f"  physics: {[round(v,2) for v in out[mode]['physics']]}")
        print(f"  mlp    : {[round(v,2) for v in out[mode]['mlp']]}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, mode in zip(axes, modes):
        ax.plot(out[mode]["levels"], out[mode]["physics"], "s-", label="Physics prior", lw=2)
        ax.plot(out[mode]["levels"], out[mode]["mlp"], "^--", label="Data-driven MLP", lw=2)
        ax.set_xlabel(f"{mode} level"); ax.set_ylabel("FDE@3s (m)"); ax.set_title(f"CAN {mode}")
        ax.legend()
    fig.suptitle("Robustness to CAN degradation (comma2k19)")
    save(fig, "fig9_robustness.png")

    with open(os.path.join(RES, "exp6_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved results/exp6_results.json\nExperiment 6 done.")


if __name__ == "__main__":
    main()
