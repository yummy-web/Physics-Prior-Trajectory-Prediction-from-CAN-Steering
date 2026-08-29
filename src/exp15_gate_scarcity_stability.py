"""
exp15_gate_scarcity_stability.py

Repeated-split stability check for the smallest-budget gate ablation. This
script addresses the reviewer risk that the heuristic gate's severe-scarcity
behavior may be an artifact of one split.

The expected interpretation is conservative: under the smallest budget, the
physics prior can remain the lowest-error baseline, while the heuristic gate is
tested only against other residual-gating strategies.

Outputs:
  figures/fig19_gate_scarcity_stability.png
  results/exp15_results.json
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from comma_loader import load_all_cached
import exp13_gate_ablation as G
import models_core as M
from plot_utils import save


RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
os.makedirs(RES, exist_ok=True)

DT = 0.05
HS = int(round(3.0 / DT))


def run_one(seed, train_k=8, test_k=30):
    data = load_all_cached(verbose=False)
    segs = list(data.values())
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(segs))
    test = [segs[i] for i in idx[:test_k]]
    pool = [segs[i] for i in idx[test_k:]]
    train = pool[:train_k]
    g = M.comma_gain(pool)

    physics = float(np.mean([np.mean(M.comma_phys_metrics(df, HS, DT, g)["fde"]) for df in test]))
    Ft, PHt, GTt, STDt, GEOt, YAWt = G.collect(test, g)
    Ftr, PHtr, GTtr, STDtr, GEOtr, YAWtr = G.collect(train, g)
    Rtr = GTtr - PHtr

    scaler = StandardScaler().fit(Ftr)
    residual = MLPRegressor(
        hidden_layer_sizes=(48, 24),
        max_iter=300,
        alpha=1e-3,
        random_state=seed,
        early_stopping=True,
    ).fit(scaler.transform(Ftr), Rtr)
    r_test = residual.predict(scaler.transform(Ft))
    r_train = residual.predict(scaler.transform(Ftr))

    out = {"Physics": physics}
    out["NoGate"] = G.fde_from_local(PHt + r_test, GEOt, YAWt)

    gh = 1.0 / (1.0 + np.exp(-(STDt - 3.0)))
    out["Heuristic"] = G.fde_from_local(PHt + gh[:, None] * r_test, GEOt, YAWt)

    best = (None, np.inf)
    for a in np.linspace(0.2, 4.0, 20):
        for b in np.linspace(-12.0, 4.0, 33):
            gt_ = 1.0 / (1.0 + np.exp(-(a * STDtr + b)))
            fde = G.fde_from_local(PHtr + gt_[:, None] * r_train, GEOtr, YAWtr)
            if fde < best[1]:
                best = ((a, b), fde)
    a_best, b_best = best[0]
    gs = 1.0 / (1.0 + np.exp(-(a_best * STDt + b_best)))
    out["LearnScalar"] = G.fde_from_local(PHt + gs[:, None] * r_test, GEOt, YAWt)

    num = np.sum(r_train * Rtr, axis=1)
    den = np.sum(r_train * r_train, axis=1) + 1e-9
    gate_target = np.clip(num / den, 0.0, 1.0)
    gate_scaler = StandardScaler().fit(Ftr)
    gate_net = MLPRegressor(
        hidden_layer_sizes=(24, 12),
        max_iter=400,
        alpha=1e-2,
        random_state=seed,
        early_stopping=True,
    ).fit(gate_scaler.transform(Ftr), gate_target)
    gm = np.clip(gate_net.predict(gate_scaler.transform(Ft)), 0.0, 1.0)
    out["LearnMLP"] = G.fde_from_local(PHt + gm[:, None] * r_test, GEOt, YAWt)
    out["mean_gate_heuristic"] = float(gh.mean())
    out["mean_gate_scalar"] = float(gs.mean())
    out["mean_gate_mlp"] = float(gm.mean())
    return out


def main():
    print("=" * 70)
    print("  Exp15: repeated-split stability of scarce-budget gating")
    print("=" * 70)
    seeds = list(range(10))
    rows = []
    for seed in seeds:
        row = run_one(seed)
        rows.append(row)
        print(
            f"seed={seed:2d} phys={row['Physics']:.3f} heur={row['Heuristic']:.3f} "
            f"nogate={row['NoGate']:.3f} scalar={row['LearnScalar']:.3f} mlp={row['LearnMLP']:.3f}"
        )

    methods = ["Physics", "NoGate", "Heuristic", "LearnScalar", "LearnMLP"]
    summary = {}
    for method in methods:
        vals = np.asarray([r[method] for r in rows], float)
        summary[method] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1))}
    hybrid_methods = ["NoGate", "Heuristic", "LearnScalar", "LearnMLP"]
    best_hybrid = []
    for r in rows:
        best_hybrid.append(min(hybrid_methods, key=lambda m: r[m]))
    summary["best_hybrid_counts"] = {m: int(best_hybrid.count(m)) for m in hybrid_methods}
    summary["heuristic_best_hybrid_fraction"] = float(best_hybrid.count("Heuristic") / len(best_hybrid))

    fig, ax = plt.subplots(figsize=(8, 5))
    vals = [[r[m] for r in rows] for m in methods]
    ax.boxplot(vals, labels=["Physics", "No gate", "Heuristic", "Learn scalar", "Learn MLP"], showmeans=True)
    ax.set_ylabel("FDE @ 3 s (m)")
    save(fig, "fig19_gate_scarcity_stability.png")

    path = os.path.join(RES, "exp15_results.json")
    with open(path, "w") as f:
        json.dump({"train_budget_segments": 8, "train_budget_nominal_s": 480, "runs": rows, "summary": summary}, f, indent=2)
    print(f"\nsaved {path}\nExp15 done.")


if __name__ == "__main__":
    main()
