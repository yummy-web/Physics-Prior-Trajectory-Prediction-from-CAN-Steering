"""
exp7_deep_baseline.py
Deep temporal baseline (GRU/LSTM) vs physics prior on comma2k19, across data sizes.
Directly answers "why not a deep sequence model?": it needs far more data and
still does not beat the physics prior in the scarce regime.

Skips gracefully if torch is not installed.

Outputs: figures/fig10_deep_vs_physics.png, results/exp7_results.json
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from comma_loader import load_all_cached
import models_core as M
import deep_baseline as D
import seq_dataset as S
from plot_utils import save

RES = os.path.join(os.path.dirname(__file__), "results"); os.makedirs(RES, exist_ok=True)
DT = 0.05; H = 3.0; HS = int(round(H / DT))
L = 10


def eval_gru(test, model, mu, sd):
    fdes = []
    for df in test:
        X, Y = S.build_sequences([df], HS, DT, "steer", L=L, stride=10)
        if len(X) == 0:
            continue
        Xs = S.standardize_apply(X, mu, sd)
        pred = D.predict_gru(model, Xs)
        fdes.extend(np.hypot(pred[:, 0] - Y[:, 0], pred[:, 1] - Y[:, 1]))
    return float(np.mean(fdes))


def main():
    print("=" * 70); print("  Experiment 7: deep temporal baseline (GRU) vs physics"); print("=" * 70)
    if not D.available():
        print("\n[skip] torch not installed; skipping deep baseline.")
        print("  install: pip install torch --index-url https://download.pytorch.org/whl/cpu")
        return
    data = load_all_cached(); segs = list(data.values())
    test = segs[-30:]; pool = segs[:-30]
    g = M.comma_gain(pool)
    B_phys = float(np.mean([np.mean(M.comma_phys_fde(df, HS, DT, g)) for df in test]))
    print(f"\nPhysics prior FDE@3s = {B_phys:.3f} m (needs only steering gain)")

    ks = [4, 16, 64, 120]
    curve = {}
    for k in ks:
        if k > len(pool): break
        X, Y = S.build_sequences(pool[:k], HS, DT, "steer", L=L, stride=5)
        mu, sd = S.standardize_fit(X)
        Xs = S.standardize_apply(X, mu, sd)
        model = D.train_gru(Xs, Y, in_dim=X.shape[-1], kind="gru", epochs=30, seed=0)
        fde = eval_gru(test, model, mu, sd)
        curve[k] = fde
        print(f"  GRU N={k:4} segs (~{k*60}s): FDE@3s = {fde:.3f} m", flush=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    Ns = np.array(list(curve.keys())) * 60
    ax.plot(Ns, list(curve.values()), "^-", color="purple", lw=2, label="GRU (deep temporal)")
    ax.axhline(B_phys, color="r", ls="--", label=f"Physics prior ({B_phys:.2f}m)")
    ax.set_xscale("log"); ax.set_xlabel("Training data (s, log)"); ax.set_ylabel("FDE@3s (m)")
    ax.set_title("Deep temporal model needs far more data and still trails the physics prior")
    ax.legend(); save(fig, "fig10_deep_vs_physics.png")

    with open(os.path.join(RES, "exp7_results.json"), "w") as f:
        json.dump({"B_phys": B_phys, "gru_curve": {int(k): v for k, v in curve.items()}}, f, indent=2)
    print("\nsaved results/exp7_results.json\nExperiment 7 done.")


if __name__ == "__main__":
    main()
