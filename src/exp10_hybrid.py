"""
exp10_hybrid.py
Test whether an improved physics-data HYBRID can beat BOTH pure physics and pure
data-driven in a mid-data regime (naive residual was a net loss earlier).

Strategies tried (comma2k19, control-conditioned, endpoint prediction):
  Phys        : pure physics prior (control->trajectory), no training
  Data        : pure data-driven MLP (endpoint), trained on k segments
  Hybrid-Add  : physics endpoint + MLP residual (both lat & lon)
  Hybrid-Lat  : physics endpoint + MLP residual on LATERAL only (physics lon is good)
  Hybrid-Gate : physics endpoint + MLP residual scaled by a learned gate in [0,1]
                (gate = sigmoid over |steer_std|, so residual only kicks in when
                 the physics model is likely wrong, e.g. complex maneuvers)

We sweep training size k and report FDE@3s vs pure physics floor. If a hybrid
is consistently <= min(Phys, Data) in the mid-data band, it is a method contribution.

Outputs: figures/fig14_hybrid.png, results/exp10_results.json
"""
import os, json
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from comma_loader import load_all_cached
import models_core as M
from plot_utils import save

RES = os.path.join(os.path.dirname(__file__), "results"); os.makedirs(RES, exist_ok=True)
DT = 0.05; H = 3.0; HS = int(round(H / DT))


def phys_endpoint(df, i0, g):
    """physics predicted endpoint (control-conditioned) and anchor frame."""
    x = df["x"].values; y = df["y"].values; yaw = np.radians(df["yaw_unwrap"].values)
    st = df["steer"].values; v = df["obs_v"].values
    px, py, th = x[i0], y[i0], yaw[i0]
    n = len(df)
    for k in range(HS):
        j = min(i0 + k, n - 1)
        th += g * st[j] * DT
        px += v[j] * np.cos(th) * DT; py += v[j] * np.sin(th) * DT
    # local-frame physics displacement
    dx, dy = px - x[i0], py - y[i0]
    c, s = np.cos(-yaw[i0]), np.sin(-yaw[i0])
    return np.array([c * dx - s * dy, s * dx + c * dy]), yaw[i0], (x[i0], y[i0])


def features(df, i0):
    fut = slice(i0, i0 + HS)
    return [df["obs_v"].values[i0], np.radians(df["yaw_rate"].values[i0]),
            np.mean(df["obs_v"].values[fut]), np.mean(df["steer"].values[fut]),
            np.std(df["steer"].values[fut])]


def gt_local(df, i0):
    dx = df["x"].values[i0 + HS] - df["x"].values[i0]; dy = df["y"].values[i0 + HS] - df["y"].values[i0]
    yaw = np.radians(df["yaw_unwrap"].values[i0]); c, s = np.cos(-yaw), np.sin(-yaw)
    return np.array([c * dx - s * dy, s * dx + c * dy])


def build(segs, g, mode):
    X, Y = [], []
    for df in segs:
        n = len(df)
        for i0 in range(0, n - HS, 8):
            ph, yaw0, _ = phys_endpoint(df, i0, g)
            gt = gt_local(df, i0)
            f = features(df, i0)
            X.append(f)
            if mode == "Data":
                Y.append(gt)
            elif mode == "Hybrid-Lat":
                Y.append([0.0, gt[1] - ph[1]])   # residual lateral only
            else:  # Add / Gate residual (both)
                Y.append(gt - ph)
    return np.array(X), np.array(Y)


def eval_model(test, g, sc, mdl, mode):
    fdes = []
    for df in test:
        n = len(df)
        for i0 in range(0, n - HS, 8):
            ph, yaw0, (x0, y0) = phys_endpoint(df, i0, g)
            f = np.array([features(df, i0)])
            pred = mdl.predict(sc.transform(f))[0]
            if mode == "Data":
                loc = pred
            elif mode == "Hybrid-Lat":
                loc = np.array([ph[0], ph[1] + pred[1]])
            elif mode == "Hybrid-Gate":
                gate = 1.0 / (1.0 + np.exp(-(np.std(df["steer"].values[i0:i0 + HS]) - 3.0)))
                loc = ph + gate * pred
            else:  # Add
                loc = ph + pred
            c, s = np.cos(yaw0), np.sin(yaw0)
            gx = x0 + c * loc[0] - s * loc[1]; gy = y0 + s * loc[0] + c * loc[1]
            fdes.append(np.hypot(gx - df["x"].values[i0 + HS], gy - df["y"].values[i0 + HS]))
    return float(np.mean(fdes))


def main():
    print("=" * 70); print("  Exp10: improved physics-data hybrid"); print("=" * 70)
    data = load_all_cached(); segs = list(data.values())
    test = segs[-20:]; pool = segs[:-20]
    g = M.comma_gain(pool)
    phys = float(np.mean([np.mean(M.comma_phys_metrics(df, HS, DT, g)["fde"]) for df in test]))
    print(f"\nPure physics FDE@3s = {phys:.3f} m (floor)")

    ks = [8, 24, 60]
    modes = ["Data", "Hybrid-Add", "Hybrid-Lat", "Hybrid-Gate"]
    curves = {m: [] for m in modes}
    for k in ks:
        if k > len(pool): break
        sub = pool[:k]
        for mode in modes:
            bmode = "Data" if mode == "Data" else mode
            X, Y = build(sub, g, bmode)
            sc = StandardScaler().fit(X)
            mdl = MLPRegressor(hidden_layer_sizes=(48, 24), max_iter=300, alpha=1e-3,
                               random_state=0, early_stopping=True).fit(sc.transform(X), Y)
            fde = eval_model(test, g, sc, mdl, mode)
            curves[mode].append(fde)
        row = " ".join(f"{m}={curves[m][-1]:.2f}" for m in modes)
        print(f"  k={k:4} ({k*60}s): phys={phys:.2f} | {row}", flush=True)

    # verdict
    print("\nVerdict per training size (best method):")
    best_hybrid_wins = 0
    for i, k in enumerate([kk for kk in ks if kk <= len(pool)]):
        vals = {"Phys": phys, **{m: curves[m][i] for m in modes}}
        best = min(vals, key=vals.get)
        if best.startswith("Hybrid"):
            best_hybrid_wins += 1
        print(f"  k={k:4}: best={best} ({vals[best]:.2f}m)")
    success = best_hybrid_wins >= 2
    print(f"\nHybrid beats both pure-physics and pure-data in {best_hybrid_wins} data regimes "
          f"-> {'SUCCESS (method contribution)' if success else 'NOT a clear win'}")

    fig, ax = plt.subplots(figsize=(8, 5))
    Ns = [k * 60 for k in ks if k <= len(pool)]
    for m in modes:
        ax.plot(Ns, curves[m], "o-" if m != "Data" else "^--", lw=2, label=m)
    ax.axhline(phys, color="r", ls="--", lw=2, label=f"Pure physics ({phys:.2f}m)")
    ax.set_xscale("log"); ax.set_xlabel("Training data (s, log)"); ax.set_ylabel("FDE@3s (m)")
    ax.set_title("Physics-data hybrid variants vs pure physics floor")
    ax.legend(fontsize=9); save(fig, "fig14_hybrid.png")

    with open(os.path.join(RES, "exp10_results.json"), "w") as f:
        json.dump({"phys": phys, "ks": [k for k in ks if k <= len(pool)], "curves": curves,
                   "hybrid_wins": best_hybrid_wins, "success": success}, f, indent=2)
    print("\nsaved results/exp10_results.json\nExp10 done.")


if __name__ == "__main__":
    main()
