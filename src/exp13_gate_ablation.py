"""
exp13_gate_ablation.py
Ablation: does a LEARNABLE gate beat the simple HEURISTIC gate under data scarcity?

Reviewer concern addressed: "the gate (sigmoid over steering-std with a fixed
threshold) is too simple; why not learn it?" We test exactly that, keeping the
physics rollout, features, residual target, data split and training-size sweep
IDENTICAL to exp10_hybrid.py so the numbers are directly comparable.

Gating strategies for the residual r = f_theta(history) added to the physics
endpoint (both lat & lon), local anchor frame:
  NoGate            : loc = phys + r                        (always add)
  Heuristic         : loc = phys + sigmoid(std_steer-3) r   (fixed threshold, exp10)
  Learn-Scalar      : loc = phys + sigmoid(a*std_steer+b) r ; a,b fit on TRAIN by
                      1-D search to minimise train FDE (2 learned params)
  Learn-MLP         : loc = phys + g_phi(features) r ; g_phi a 2nd small MLP with
                      sigmoid output, trained on TRAIN to minimise residual-weighting
                      error (data-driven gate, the "sophisticated" version)

If the learnable gates do NOT beat the heuristic at small k (and/or need more data
to become competitive), the simple heuristic is empirically justified in the
scarce-data regime -- which is the paper's operating point.

Outputs: figures/fig_gate_ablation.png, results/exp13_results.json
Runs in the same env as exp1-exp12 (numpy/scipy/sklearn, comma cache).
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
SEED = 0


def phys_endpoint(df, i0, g):
    """physics predicted endpoint (control-conditioned) in local anchor frame."""
    x = df["x"].values; y = df["y"].values; yaw = np.radians(df["yaw_unwrap"].values)
    st = df["steer"].values; v = df["obs_v"].values
    px, py, th = x[i0], y[i0], yaw[i0]
    n = len(df)
    for k in range(HS):
        j = min(i0 + k, n - 1)
        th += g * st[j] * DT
        px += v[j] * np.cos(th) * DT; py += v[j] * np.sin(th) * DT
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


def collect(segs, g):
    """Return per-anchor arrays: features F, physics endpoint PH, ground-truth GT,
    steering-std STD, and anchor geometry for global FDE reconstruction."""
    F, PH, GT, STD, GEO, YAW = [], [], [], [], [], []
    for df in segs:
        n = len(df)
        for i0 in range(0, n - HS, 8):
            ph, yaw0, (x0, y0) = phys_endpoint(df, i0, g)
            F.append(features(df, i0)); PH.append(ph); GT.append(gt_local(df, i0))
            STD.append(np.std(df["steer"].values[i0:i0 + HS]))
            GEO.append((x0, y0, df["x"].values[i0 + HS], df["y"].values[i0 + HS])); YAW.append(yaw0)
    return (np.array(F), np.array(PH), np.array(GT), np.array(STD), GEO, np.array(YAW))


def fde_from_local(loc, GEO, YAW):
    out = []
    for p, (x0, y0, gx, gy), yaw0 in zip(loc, GEO, YAW):
        c, s = np.cos(yaw0), np.sin(yaw0)
        px = x0 + c * p[0] - s * p[1]; py = y0 + s * p[0] + c * p[1]
        out.append(np.hypot(px - gx, py - gy))
    return float(np.mean(out))


def main():
    print("=" * 70); print("  Exp13: learnable-gate vs heuristic-gate ablation"); print("=" * 70)
    data = load_all_cached(verbose=True); segs = list(data.values())
    test = segs[-20:]; pool = segs[:-20]
    g = M.comma_gain(pool)

    # physics floor on test
    phys = float(np.mean([np.mean(M.comma_phys_metrics(df, HS, DT, g)["fde"]) for df in test]))
    print(f"\nPure physics FDE@3s = {phys:.3f} m (floor)")

    Ft, PHt, GTt, STDt, GEOt, YAWt = collect(test, g)

    ks = [8, 24, 60]
    modes = ["NoGate", "Heuristic", "Learn-Scalar", "Learn-MLP"]
    curves = {m: [] for m in modes}
    gate_info = {}

    for k in ks:
        if k > len(pool):
            break
        sub = pool[:k]
        Ftr, PHtr, GTtr, STDtr, GEOtr, YAWtr = collect(sub, g)
        Rtr = GTtr - PHtr  # residual target (both components)

        # shared residual network (identical to exp10 Hybrid-Add/Gate)
        sc = StandardScaler().fit(Ftr)
        res = MLPRegressor(hidden_layer_sizes=(48, 24), max_iter=300, alpha=1e-3,
                           random_state=SEED, early_stopping=True).fit(sc.transform(Ftr), Rtr)
        r_test = res.predict(sc.transform(Ft))
        r_train = res.predict(sc.transform(Ftr))

        # ---- NoGate ----
        curves["NoGate"].append(fde_from_local(PHt + r_test, GEOt, YAWt))

        # ---- Heuristic (fixed threshold 3.0, exactly exp10) ----
        gh = 1.0 / (1.0 + np.exp(-(STDt - 3.0)))
        curves["Heuristic"].append(fde_from_local(PHt + gh[:, None] * r_test, GEOt, YAWt))

        # ---- Learn-Scalar: fit a,b of sigmoid(a*std+b) on TRAIN to minimise train FDE ----
        best = (None, np.inf)
        for a in np.linspace(0.2, 4.0, 20):
            for b in np.linspace(-12.0, 4.0, 33):
                gt_ = 1.0 / (1.0 + np.exp(-(a * STDtr + b)))
                f = fde_from_local(PHtr + gt_[:, None] * r_train, GEOtr, YAWtr)
                if f < best[1]:
                    best = ((a, b), f)
        (a_, b_), _ = best
        gs = 1.0 / (1.0 + np.exp(-(a_ * STDt + b_)))
        curves["Learn-Scalar"].append(fde_from_local(PHt + gs[:, None] * r_test, GEOt, YAWt))

        # ---- Learn-MLP: data-driven gate g_phi(features) in (0,1) ----
        # Target gate = clip( <r, gt-phys> / <r, r> , 0, 1 ): the per-anchor optimal
        # scalar that minimises || (phys + gate*r) - gt ||^2  (closed form).
        num = np.sum(r_train * Rtr, axis=1)
        den = np.sum(r_train * r_train, axis=1) + 1e-9
        gate_target = np.clip(num / den, 0.0, 1.0)
        gsc = StandardScaler().fit(Ftr)
        gate_net = MLPRegressor(hidden_layer_sizes=(24, 12), max_iter=400, alpha=1e-2,
                                random_state=SEED, early_stopping=True).fit(gsc.transform(Ftr), gate_target)
        gm = np.clip(gate_net.predict(gsc.transform(Ft)), 0.0, 1.0)
        curves["Learn-MLP"].append(fde_from_local(PHt + gm[:, None] * r_test, GEOt, YAWt))

        gate_info[k] = {"learn_scalar_ab": [float(a_), float(b_)],
                        "mean_gate_heur": float(gh.mean()),
                        "mean_gate_scalar": float(gs.mean()),
                        "mean_gate_mlp": float(gm.mean())}
        row = " ".join(f"{m}={curves[m][-1]:.3f}" for m in modes)
        print(f"  k={k:4} ({k*60}s): phys={phys:.3f} | {row}", flush=True)

    # ---- verdict ----
    Ns = [k for k in ks if k <= len(pool)]
    print("\nBest gating strategy per training size:")
    heur_not_beaten_lowk = True
    for i, k in enumerate(Ns):
        vals = {m: curves[m][i] for m in modes}
        best_m = min(vals, key=vals.get)
        learn_best = min(curves["Learn-Scalar"][i], curves["Learn-MLP"][i])
        # positive gap = heuristic FDE higher than best learnable = learnable wins
        gap = curves["Heuristic"][i] - learn_best
        print(f"  k={k:4}: best={best_m} ({vals[best_m]:.3f}m) | "
              f"heuristic={curves['Heuristic'][i]:.3f} learn_best={learn_best:.3f} "
              f"(heur-learn={gap:+.3f})")
        if i == 0 and gap > 0.005:  # a learnable gate clearly beats heuristic at smallest k
            heur_not_beaten_lowk = False

    print(f"\nAt the smallest data budget (k={Ns[0]}, {Ns[0]*60}s), the heuristic gate is "
          f"{'NOT beaten by any learnable gate -> simple heuristic justified under scarcity' if heur_not_beaten_lowk else 'beaten by a learnable gate'}.")

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(8, 5))
    Nsec = [k * 60 for k in Ns]
    styles = {"NoGate": "s--", "Heuristic": "o-", "Learn-Scalar": "^-", "Learn-MLP": "d-"}
    for m in modes:
        ax.plot(Nsec, curves[m], styles[m], lw=2, label=m)
    ax.axhline(phys, color="r", ls=":", lw=2, label=f"Pure physics ({phys:.2f} m)")
    ax.set_xscale("log"); ax.set_xlabel("Training data (s, log scale)")
    ax.set_ylabel("FDE @ 3 s (m)"); ax.legend(fontsize=10)
    save(fig, "fig_gate_ablation.png")

    with open(os.path.join(RES, "exp13_results.json"), "w") as f:
        json.dump({"phys": phys, "ks": Ns, "curves": curves, "gate_info": gate_info,
                   "heuristic_not_beaten_at_lowest_k": heur_not_beaten_lowk}, f, indent=2)
    print("\nsaved results/exp13_results.json\nExp13 done.")


if __name__ == "__main__":
    main()
