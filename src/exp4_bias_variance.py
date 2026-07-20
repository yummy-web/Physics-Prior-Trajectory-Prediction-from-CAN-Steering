"""
exp4_bias_variance.py
Bias-variance theory of the data-scarcity crossover.

Theory:
  A physics-prior predictor has (roughly) data-independent error:
      E_phys(N) = B_phys           (high bias from model mismatch, ~zero variance)
  A data-driven predictor's expected error decomposes as:
      E_data(N) = B_data + V(N),   V(N) ~ C / N^alpha   (variance falls with data)
  The crossover N* solves E_data(N*) = B_phys:
      N* = ( C / (B_phys - B_data) )^(1/alpha)

We measure E_data(N) on comma2k19 by training the direct MLP with N training
segments (multiple seeds), fit E_data(N) = B_data + C/N^alpha, overlay B_phys,
and report the fitted crossover with CI.

Outputs: figures/fig7_bias_variance.png, results/exp4_results.json
"""
import os, json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from comma_loader import load_all_cached
import models_core as M
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from plot_utils import save

RES = os.path.join(os.path.dirname(__file__), "results"); os.makedirs(RES, exist_ok=True)
DT = 0.05; H = 3.0; HS = int(round(H / DT))


def build_direct(segs):
    X, Y = [], []
    for df in segs:
        n = len(df)
        for i0 in range(0, n - HS, 10):
            fut = slice(i0, i0 + HS)
            feat = [df["obs_v"].values[i0], np.radians(df["yaw_rate"].values[i0]),
                    np.mean(df["obs_v"].values[fut]), np.mean(df["steer"].values[fut]),
                    np.std(df["steer"].values[fut])]
            yaw = np.radians(df["yaw_unwrap"].values[i0])
            dx = df["x"].values[i0 + HS] - df["x"].values[i0]; dy = df["y"].values[i0 + HS] - df["y"].values[i0]
            c, s = np.cos(-yaw), np.sin(-yaw)
            X.append(feat); Y.append([c * dx - s * dy, s * dx + c * dy])
    return np.array(X), np.array(Y)


def eval_mlp(test, sc, mdl):
    fdes = []
    for df in test:
        n = len(df)
        for i0 in range(0, n - HS, 10):
            fut = slice(i0, i0 + HS)
            feat = np.array([[df["obs_v"].values[i0], np.radians(df["yaw_rate"].values[i0]),
                    np.mean(df["obs_v"].values[fut]), np.mean(df["steer"].values[fut]),
                    np.std(df["steer"].values[fut])]])
            lx, ly = mdl.predict(sc.transform(feat))[0]
            yaw = np.radians(df["yaw_unwrap"].values[i0]); c, s = np.cos(yaw), np.sin(yaw)
            ddx = c * lx - s * ly; ddy = s * lx + c * ly
            fdes.append(np.hypot((df["x"].values[i0] + ddx) - df["x"].values[i0 + HS],
                                 (df["y"].values[i0] + ddy) - df["y"].values[i0 + HS]))
    return float(np.mean(fdes))


def main():
    print("=" * 70); print("  Experiment 4: bias-variance theory of the crossover"); print("=" * 70)
    data = load_all_cached(); segs = list(data.values())
    test = segs[-30:]; pool = segs[:-30]
    g = M.comma_gain(pool)
    # physics constant
    B_phys = float(np.mean([np.mean(M.comma_phys_fde(df, HS, DT, g)) for df in test]))
    print(f"\nB_phys (physics FDE@3s, data-independent) = {B_phys:.3f} m")

    ks = [2, 4, 8, 16, 32, 64, 120]
    seeds = [0, 1]
    curve = {}
    for k in ks:
        if k > len(pool): break
        vals = []
        _seeds = seeds if k <= 32 else [0]
        for sd in _seeds:
            rng = np.random.default_rng(sd)
            sub = [pool[i] for i in rng.choice(len(pool), size=k, replace=False)]
            X, Y = build_direct(sub)
            sc = StandardScaler().fit(X)
            mdl = MLPRegressor(hidden_layer_sizes=(48, 24), max_iter=400, alpha=1e-3,
                               random_state=sd, early_stopping=True).fit(sc.transform(X), Y)
            vals.append(eval_mlp(test, sc, mdl))
        curve[k] = (float(np.mean(vals)), float(np.std(vals)))
        print(f"  N={k:4} segs: E_data={curve[k][0]:.3f} +/- {curve[k][1]:.3f}", flush=True)

    Ns = np.array(list(curve.keys()), float)
    Es = np.array([curve[k][0] for k in curve])

    # fit E = B + C / N^alpha
    def model(N, B, C, alpha):
        return B + C / np.power(N, alpha)
    try:
        popt, pcov = curve_fit(model, Ns, Es, p0=[0.5, 5.0, 1.0],
                               bounds=([0, 0, 0.2], [np.inf, np.inf, 3.0]), maxfev=10000)
        B_data, C, alpha = popt
    except Exception as e:
        print("fit failed:", e); B_data, C, alpha = Es.min(), 5.0, 1.0
    print(f"\nFit E_data(N) = {B_data:.3f} + {C:.2f}/N^{alpha:.2f}")
    # crossover
    if B_phys > B_data:
        Nstar = (C / (B_phys - B_data)) ** (1.0 / alpha)
        print(f"Crossover N* = {Nstar:.1f} segments (~{Nstar*60:.0f}s) where data-driven matches physics")
    else:
        Nstar = float("inf")
        print(f"B_data ({B_data:.3f}) <= B_phys ({B_phys:.3f}): data-driven never structurally worse asymptotically,"
              f" but needs large N; physics dominates in scarce regime")

    # figure
    fig, ax = plt.subplots(figsize=(8, 5))
    errs = [curve[k][1] for k in curve]
    ax.errorbar(Ns * 60, Es, yerr=errs, fmt="^", capsize=3, label="Data-driven MLP (measured)")
    xs = np.linspace(Ns.min(), Ns.max(), 200)
    ax.plot(xs * 60, model(xs, B_data, C, alpha), "b-", label=f"Fit: {B_data:.2f}+{C:.1f}/N^{alpha:.2f}")
    ax.axhline(B_phys, color="r", ls="--", label=f"Physics prior (B_phys={B_phys:.2f}m)")
    if np.isfinite(Nstar):
        ax.axvline(Nstar * 60, color="gray", ls=":", label=f"Crossover ~{Nstar*60:.0f}s")
    ax.set_xscale("log"); ax.set_xlabel("Training data (s, log)"); ax.set_ylabel("FDE@3s (m)")
    ax.set_title("Bias-variance view: data-driven error = bias + C/N^alpha vs constant physics prior")
    ax.legend(); save(fig, "fig7_bias_variance.png")

    with open(os.path.join(RES, "exp4_results.json"), "w") as f:
        json.dump({"B_phys": B_phys, "curve": {int(k): curve[k] for k in curve},
                   "fit": {"B_data": float(B_data), "C": float(C), "alpha": float(alpha)},
                   "N_star_segments": float(Nstar)}, f, indent=2)
    print("\nsaved results/exp4_results.json\nExperiment 4 done.")


if __name__ == "__main__":
    main()
