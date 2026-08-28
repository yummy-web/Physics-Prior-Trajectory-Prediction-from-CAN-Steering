"""
exp12_pi_deep_comparison.py
Comparison against physics-informed DEEP baselines, in OUR task setting
(ego-vehicle, control-conditioned lateral trajectory from CAN steering).

We reproduce the *paradigms* of recent physics-informed models fairly, adapted
to the ego-steering task (reimplementing their exact car-following/multi-agent
forms would be an unfair, ill-posed comparison since those solve different
problems). Concretely we compare:

  Pure-Physics     : steering->curvature kinematic rollout (no training)
  GRU              : pure data-driven sequence model (no physics)
  PERL-style       : physics rollout + GRU residual (fixed additive), i.e. the
                     Physics-Enhanced Residual Learning paradigm (Physics + NN residual)
  Ours-GatedHybrid : physics rollout + gated residual (residual gated by maneuver
                     complexity) -- our contribution vs PERL's fixed residual

All on comma2k19, control-conditioned, endpoint FDE@3s, across training sizes.
This is the paper's "comparison with SOTA physics-informed methods" section, done
honestly in the correct task setting.

Outputs: figures/fig17_pi_deep_comparison.png, results/exp12_results.json
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from comma_loader import load_all_cached
import models_core as M
import seq_dataset as S
import deep_baseline as D
from plot_utils import save

RES = os.path.join(os.path.dirname(__file__), "results"); os.makedirs(RES, exist_ok=True)
DT = 0.05; H = 3.0; HS = int(round(H / DT)); L = 10


def phys_endpoint_local(df, i0, g):
    x = df["x"].values; y = df["y"].values; yaw = np.radians(df["yaw_unwrap"].values)
    st = df["steer"].values; v = df["obs_v"].values; n = len(df)
    px, py, th = x[i0], y[i0], yaw[i0]
    for k in range(HS):
        j = min(i0 + k, n - 1)
        th += g * st[j] * DT
        px += v[j] * np.cos(th) * DT; py += v[j] * np.sin(th) * DT
    dx, dy = px - x[i0], py - y[i0]
    c, s = np.cos(-yaw[i0]), np.sin(-yaw[i0])
    return np.array([c * dx - s * dy, s * dx + c * dy])


def gt_local(df, i0):
    dx = df["x"].values[i0 + HS] - df["x"].values[i0]; dy = df["y"].values[i0 + HS] - df["y"].values[i0]
    yaw = np.radians(df["yaw_unwrap"].values[i0]); c, s = np.cos(-yaw), np.sin(-yaw)
    return np.array([c * dx - s * dy, s * dx + c * dy])


def build_seq_with_phys(segs, g, mode):
    """Return X_seq (N,L,F), Y (N,2), Phys (N,2). mode controls target."""
    Xs, Y, Ph = [], [], []
    for df in segs:
        x = df["x"].values; y = df["y"].values
        yaw = np.radians(df["yaw_unwrap"].values); v = df["obs_v"].values
        w = np.radians(df["yaw_rate"].values); steer = df["steer"].values
        n = len(df)
        for i0 in range(L, n - HS, 8):
            hist = [[v[k], w[k], steer[k], v[k] - v[k-1]] for k in range(i0 - L + 1, i0 + 1)]
            ph = phys_endpoint_local(df, i0, g); gt = gt_local(df, i0)
            Xs.append(hist)
            Ph.append(ph)
            if mode == "residual":
                Y.append(gt - ph)
            else:
                Y.append(gt)
    return np.array(Xs, np.float32), np.array(Y, np.float32), np.array(Ph, np.float32)


def eval_variant(test, g, model, mu, sd, mode):
    fdes = []
    for df in test:
        Xs, Y, Ph = build_seq_with_phys([df], g, mode)
        if len(Xs) == 0:
            continue
        Xn = S.standardize_apply(Xs, mu, sd)
        pred = D.predict_gru(model, Xn)
        if mode == "residual":
            # gate for ours: scale residual by maneuver complexity (steer std proxy from hist)
            if mode == "residual" and getattr(eval_variant, "_gate", False):
                sc_std = Xs[:, :, 2].std(axis=1)
                gate = 1.0 / (1.0 + np.exp(-(sc_std - sc_std.mean())))
                out = Ph + gate[:, None] * pred
            else:
                out = Ph + pred
        else:
            out = pred
        d = np.hypot(out[:, 0] - Y[:, 0] if mode != "residual" else out[:, 0] - (Ph[:, 0] + (Y[:, 0])),
                     out[:, 1] - Y[:, 1] if mode != "residual" else out[:, 1] - (Ph[:, 1] + (Y[:, 1])))
        # simpler: recompute gt
        gt = np.zeros_like(out)
        # reconstruct gt: for residual mode Y=gt-ph so gt=Y+ph
        gt = (Y + Ph) if mode == "residual" else Y
        d = np.hypot(out[:, 0] - gt[:, 0], out[:, 1] - gt[:, 1])
        fdes.extend(d)
    return float(np.mean(fdes))


def main():
    print("=" * 72); print("  Exp12: comparison vs physics-informed DEEP baselines (fair, our task)"); print("=" * 72)
    if not D.available():
        print("[skip] torch not installed."); return
    data = load_all_cached(verbose=False); segs = list(data.values())
    test = segs[-30:]; pool = segs[:-30]
    g = M.comma_gain(pool)
    phys = float(np.mean([np.mean(M.comma_phys_metrics(df, HS, DT, g)["fde"]) for df in test]))
    print(f"\nPure-Physics FDE@3s = {phys:.3f} m (no training)")

    ks = [8, 40, 100]
    curves = {"GRU": [], "PERL-style": [], "Ours-GatedHybrid": []}
    for k in ks:
        if k > len(pool): break
        sub = pool[:k]
        # GRU pure
        Xs, Y, Ph = build_seq_with_phys(sub, g, "absolute")
        mu, sd = S.standardize_fit(Xs); Xn = S.standardize_apply(Xs, mu, sd)
        m_gru = D.train_gru(Xn, Y, in_dim=Xs.shape[-1], epochs=25, seed=0)
        eval_variant._gate = False
        curves["GRU"].append(eval_variant(test, g, m_gru, mu, sd, "absolute"))
        # PERL-style residual (fixed additive)
        Xs, Y, Ph = build_seq_with_phys(sub, g, "residual")
        mu, sd = S.standardize_fit(Xs); Xn = S.standardize_apply(Xs, mu, sd)
        m_res = D.train_gru(Xn, Y, in_dim=Xs.shape[-1], epochs=25, seed=0)
        eval_variant._gate = False
        curves["PERL-style"].append(eval_variant(test, g, m_res, mu, sd, "residual"))
        # Ours gated (same trained residual model, gated application)
        eval_variant._gate = True
        curves["Ours-GatedHybrid"].append(eval_variant(test, g, m_res, mu, sd, "residual"))
        print(f"  k={k:4}({k*60}s): phys={phys:.2f} GRU={curves['GRU'][-1]:.2f} "
              f"PERL={curves['PERL-style'][-1]:.2f} Ours={curves['Ours-GatedHybrid'][-1]:.2f}", flush=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    Ns = [k * 60 for k in ks if k <= len(pool)]
    ax.axhline(phys, color="r", ls="--", lw=2, label="Pure-Physics (no training)")
    for m, st in [("GRU", "^:"), ("PERL-style", "o--"), ("Ours-GatedHybrid", "s-")]:
        ax.plot(Ns, curves[m], st, lw=2, label=m)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Training data (s, log)"); ax.set_ylabel("FDE@3s (m, log)")
    ax.legend(); save(fig, "fig17_pi_deep_comparison.png")

    with open(os.path.join(RES, "exp12_results.json"), "w") as f:
        json.dump({"phys": phys, "ks": [k for k in ks if k <= len(pool)], "curves": curves}, f, indent=2)
    print("\nsaved results/exp12_results.json\nExp12 done.")


if __name__ == "__main__":
    main()
