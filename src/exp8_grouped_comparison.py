"""
exp8_grouped_comparison.py
Reviewer-proof comparison with baselines split into two clearly-labeled groups,
plus ADE and lateral/longitudinal error decomposition (comma2k19).

Group A (motion extrapolation, NO future control): CV, CA, CTRV, CTRA, EKF
Group B (control-conditioned, USE future control):  Physics prior, Data-MLP
  -> Physics's fair competitor is Group B; Group A is context, not a fair rival.

Also reports the "control->trajectory mapping accuracy" of the physics model
(given the actual future steering), i.e. a prediction upper bound / sysID limit.

Outputs: figures/fig11_grouped_ade_fde.png, fig12_latlon_curves.png,
         results/exp8_results.json
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
DT = 0.05
HORIZONS = [1.0, 2.0, 3.0, 5.0]
GROUP_A = ["CV", "CA", "CTRV", "CTRA", "EKF"]


def build_direct(segs, hs):
    X, Y = [], []
    for df in segs:
        n = len(df)
        for i0 in range(0, n - hs, 5):
            fut = slice(i0, i0 + hs)
            feat = [df["obs_v"].values[i0], np.radians(df["yaw_rate"].values[i0]),
                    np.mean(df["obs_v"].values[fut]), np.mean(df["steer"].values[fut]),
                    np.std(df["steer"].values[fut])]
            yaw = np.radians(df["yaw_unwrap"].values[i0])
            dx = df["x"].values[i0 + hs] - df["x"].values[i0]; dy = df["y"].values[i0 + hs] - df["y"].values[i0]
            c, s = np.cos(-yaw), np.sin(-yaw)
            X.append(feat); Y.append([c * dx - s * dy, s * dx + c * dy])
    return np.array(X), np.array(Y)


def mlp_metrics(df, hs, sc, mdl):
    x = df["x"].values; y = df["y"].values
    n = len(df); ade, fde = [], []
    for i0 in range(0, n - hs, 5):
        fut = slice(i0, i0 + hs)
        feat = np.array([[df["obs_v"].values[i0], np.radians(df["yaw_rate"].values[i0]),
                np.mean(df["obs_v"].values[fut]), np.mean(df["steer"].values[fut]),
                np.std(df["steer"].values[fut])]])
        lx, ly = mdl.predict(sc.transform(feat))[0]
        yaw = np.radians(df["yaw_unwrap"].values[i0]); c, s = np.cos(yaw), np.sin(yaw)
        ddx = c * lx - s * ly; ddy = s * lx + c * ly
        # direct predictor only gives endpoint; ADE approximated by linear interp to endpoint
        fde.append(np.hypot((x[i0] + ddx) - x[i0 + hs], (y[i0] + ddy) - y[i0 + hs]))
        # linear path assumption for ADE
        ks = np.arange(1, hs + 1) / hs
        gxp = x[i0] + ddx * ks; gyp = y[i0] + ddy * ks
        gx = x[i0 + 1:i0 + hs + 1]; gy = y[i0 + 1:i0 + hs + 1]
        ade.append(np.mean(np.hypot(gxp - gx, gyp - gy)))
    return {"ade": np.array(ade), "fde": np.array(fde)}


def main():
    print("=" * 72); print("  Exp8: grouped baselines + ADE/FDE + lat/lon decomposition"); print("=" * 72)
    data = load_all_cached(); segs = list(data.values())
    test = segs[-30:]; train = segs[:80]
    g = M.comma_gain(train)

    # collect per-anchor metrics pooled over test segments
    agg = {m: {H: {"ade": [], "fde": []} for H in HORIZONS} for m in GROUP_A + ["Physics", "DataMLP"]}
    latlon = {H: {"lat": [], "lon": []} for H in HORIZONS}

    mlps = {}
    for H in HORIZONS:
        hs = int(round(H / DT))
        X, Y = build_direct(train, hs); sc = StandardScaler().fit(X)
        mdl = MLPRegressor(hidden_layer_sizes=(48, 24), max_iter=600, alpha=1e-3,
                           random_state=0, early_stopping=True).fit(sc.transform(X), Y)
        mlps[H] = (sc, mdl)

    for df in test:
        for H in HORIZONS:
            hs = int(round(H / DT))
            for mdl in GROUP_A:
                r = M.comma_kin_metrics(df, hs, DT, mdl)
                agg[mdl][H]["ade"].extend(r["ade"]); agg[mdl][H]["fde"].extend(r["fde"])
            rp = M.comma_phys_metrics(df, hs, DT, g)
            agg["Physics"][H]["ade"].extend(rp["ade"]); agg["Physics"][H]["fde"].extend(rp["fde"])
            latlon[H]["lat"].extend(rp["lat"]); latlon[H]["lon"].extend(rp["lon"])
            sc, mm = mlps[H]
            rm = mlp_metrics(df, hs, sc, mm)
            agg["DataMLP"][H]["ade"].extend(rm["ade"]); agg["DataMLP"][H]["fde"].extend(rm["fde"])

    # summarize
    summary = {}
    for m in agg:
        summary[m] = {}
        for H in HORIZONS:
            mean_f, lo_f, hi_f = M.bootstrap_ci(agg[m][H]["fde"], n_boot=1000)
            mean_a, _, _ = M.bootstrap_ci(agg[m][H]["ade"], n_boot=1000)
            summary[m][H] = {"fde": mean_f, "fde_lo": lo_f, "fde_hi": hi_f, "ade": mean_a}

    print("\n--- Group A (motion extrapolation, NO future control) ---")
    for m in GROUP_A:
        print(f"  {m:6} FDE@3s={summary[m][3.0]['fde']:.2f} ADE@3s={summary[m][3.0]['ade']:.2f}")
    print("--- Group B (control-conditioned, USE future control) ---")
    for m in ["Physics", "DataMLP"]:
        print(f"  {m:8} FDE@3s={summary[m][3.0]['fde']:.2f} [{summary[m][3.0]['fde_lo']:.2f},{summary[m][3.0]['fde_hi']:.2f}] ADE@3s={summary[m][3.0]['ade']:.2f}")

    print("\n--- Physics control->trajectory mapping accuracy (given true future steer) ---")
    for H in HORIZONS:
        print(f"  H={H}s: FDE={summary['Physics'][H]['fde']:.2f}m  (upper bound of control-conditioned prediction)")

    print("\n--- lat/lon decomposition (Physics) ---")
    for H in HORIZONS:
        lo = np.mean(latlon[H]["lon"]); la = np.mean(latlon[H]["lat"])
        print(f"  H={H}s: lon(speed)={lo:.2f}m lat(steer)={la:.2f}m -> {'lon' if lo>la else 'lat'} dominates")

    # Fig 11: grouped FDE with CI, Group A dashed/gray, Group B solid/bold
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in GROUP_A:
        ax.plot(HORIZONS, [summary[m][H]["fde"] for H in HORIZONS], "o:", color="gray", alpha=0.7, lw=1.1)
    ax.plot([], [], "o:", color="gray", label="Group A: motion extrapolation (no control)")
    for m, col in [("Physics", "tab:red"), ("DataMLP", "tab:blue")]:
        means = [summary[m][H]["fde"] for H in HORIZONS]
        lo = [means[i] - summary[m][HORIZONS[i]]["fde_lo"] for i in range(len(HORIZONS))]
        hi = [summary[m][HORIZONS[i]]["fde_hi"] - means[i] for i in range(len(HORIZONS))]
        ax.errorbar(HORIZONS, means, yerr=[lo, hi], fmt="s-", color=col, lw=2.5, capsize=3,
                    label=f"Group B: {m} (uses future control)")
    ax.set_xlabel("Prediction horizon (s)"); ax.set_ylabel("FDE (m)")
    ax.set_title("comma2k19: baselines grouped by information used (fair comparison)")
    ax.legend(fontsize=8); save(fig, "fig11_grouped_ade_fde.png")

    # Fig 12: lat/lon curves
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(HORIZONS, [np.mean(latlon[H]["lon"]) for H in HORIZONS], "s-", label="Longitudinal (speed-driven)", lw=2)
    ax.plot(HORIZONS, [np.mean(latlon[H]["lat"]) for H in HORIZONS], "^-", label="Lateral (steering-driven)", lw=2)
    ax.set_xlabel("Prediction horizon (s)"); ax.set_ylabel("Error (m)")
    ax.set_title("comma2k19: error decomposition (lateral vs longitudinal)")
    ax.legend(); save(fig, "fig12_latlon_curves.png")

    with open(os.path.join(RES, "exp8_results.json"), "w") as f:
        json.dump({"summary": {m: {str(H): summary[m][H] for H in HORIZONS} for m in summary},
                   "latlon": {str(H): {"lon": float(np.mean(latlon[H]["lon"])), "lat": float(np.mean(latlon[H]["lat"]))} for H in HORIZONS}},
                  f, indent=2)
    print("\nsaved results/exp8_results.json\nExp8 done.")


if __name__ == "__main__":
    main()
