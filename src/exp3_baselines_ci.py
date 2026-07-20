"""
exp3_baselines_ci.py
Comprehensive baseline comparison with bootstrap 95% CI and paired significance.

Loader (LOSO over 4 segments) and comma2k19 (held-out test segments):
  Baselines: CV, CA, CTRV, CTRA, EKF, Physics-prior, Data-MLP, (GRU on comma)
  Metric: FDE at multiple horizons, mean [95% CI], Wilcoxon p vs Physics.

Outputs:
  figures/fig5_baselines_loader.png, fig6_baselines_comma.png
  results/exp3_results.json
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from loader_data_loader import load_all as load_loader
from comma_loader import load_all_cached
import baselines as B
import models_core as M
from plot_utils import save

RES = os.path.join(os.path.dirname(__file__), "results"); os.makedirs(RES, exist_ok=True)
HORIZONS = [1.0, 2.0, 3.0, 5.0]


def loader_experiment():
    data = load_loader(); segs = list(data.keys()); dt = 0.1
    kin_models = ["CV", "CA", "CTRV", "CTRA", "EKF"]
    per = {m: {H: [] for H in HORIZONS} for m in kin_models + ["Physics", "DataMLP"]}
    for test in segs:
        train = [s for s in segs if s != test]
        g = M.loader_gain(data, train); coef = M.loader_speed_coef(data, train)
        c = M.loader_center(data[test], warmup=True, dt=dt)
        df = data[test]
        for mdl in kin_models:
            r = B.eval_kinematic(df, HORIZONS, dt, mdl, stride=2)
            for H in HORIZONS: per[mdl][H].append(r[H][1])
        for H in HORIZONS:
            hs = int(round(H / dt))
            fde = M.loader_phys_fde(df, hs, dt, g, c, coef, use_obs_speed=False)
            per["Physics"][H].append(float(np.mean(fde)))
            Xtr, Ytr = M.loader_direct_dataset(data, train, hs, dt)
            sc, mm = M.train_mlp(Xtr, Ytr)
            # eval MLP on test
            n = len(df); ds = []
            for i0 in range(0, n - hs, 2):
                fut = slice(i0, i0 + hs)
                feat = np.array([[df["obs_v"].values[i0], np.radians(df["yaw_rate"].values[i0]),
                        np.mean(df["footboard"].values[fut]), np.mean(df["oil_door"].values[fut]),
                        np.mean(df["motor_angle"].values[fut] - c), np.std(df["motor_angle"].values[fut])]])
                lx, ly = mm.predict(sc.transform(feat))[0]
                yaw = np.radians(df["yaw_unwrap"].values[i0]); cc, ss = np.cos(yaw), np.sin(yaw)
                ddx = cc * lx - ss * ly; ddy = ss * lx + cc * ly
                ds.append(np.hypot((df["x"].values[i0] + ddx) - df["x"].values[i0 + hs],
                                   (df["y"].values[i0] + ddy) - df["y"].values[i0 + hs]))
            per["DataMLP"][H].append(float(np.mean(ds)))
    # aggregate: mean over folds (each fold already a mean); report mean+CI over folds
    summary = {}
    for m in per:
        summary[m] = {}
        for H in HORIZONS:
            mean, lo, hi = M.bootstrap_ci(per[m][H], n_boot=2000)
            summary[m][H] = {"mean": mean, "lo": lo, "hi": hi}
    return summary, per


def comma_experiment(max_train=60):
    data = load_all_cached(); segs = list(data.values()); dt = 0.05
    test = segs[-20:]; train = segs[:max_train]
    g = M.comma_gain(train)
    kin_models = ["CV", "CA", "CTRV", "CTRA", "EKF"]
    per = {m: {H: [] for H in HORIZONS} for m in kin_models + ["Physics"]}
    for df in test:
        for mdl in kin_models:
            r = B.eval_kinematic(df, HORIZONS, dt, mdl, stride=5)
            for H in HORIZONS: per[mdl][H].append(r[H][1])
        for H in HORIZONS:
            hs = int(round(H / dt))
            fde = M.comma_phys_fde(df, hs, dt, g, use_obs_speed=True)
            per["Physics"][H].append(float(np.mean(fde)))
    summary = {}
    for m in per:
        summary[m] = {}
        for H in HORIZONS:
            mean, lo, hi = M.bootstrap_ci(per[m][H], n_boot=2000)
            summary[m][H] = {"mean": mean, "lo": lo, "hi": hi}
    return summary, per


def plot_summary(summary, title, fname):
    models = list(summary.keys())
    fig, ax = plt.subplots(figsize=(9, 5))
    for m in models:
        means = [summary[m][H]["mean"] for H in HORIZONS]
        los = [summary[m][H]["mean"] - summary[m][H]["lo"] for H in HORIZONS]
        his = [summary[m][H]["hi"] - summary[m][H]["mean"] for H in HORIZONS]
        style = "s-" if m == "Physics" else ("^--" if m == "DataMLP" else "o:")
        lw = 2.5 if m == "Physics" else 1.3
        ax.errorbar(HORIZONS, means, yerr=[los, his], fmt=style, lw=lw, capsize=3, label=m)
    ax.set_xlabel("Prediction horizon (s)"); ax.set_ylabel("FDE (m), mean +/- 95% CI")
    ax.set_title(title); ax.legend(ncol=2, fontsize=9)
    save(fig, fname)


def main():
    print("=" * 70); print("  Experiment 3: full baselines + bootstrap CI + significance"); print("=" * 70)
    print("\n[Loader LOSO]")
    ls, lper = loader_experiment()
    for m in ls:
        print(f"  {m:9} FDE@3s = {ls[m][3.0]['mean']:.2f} [{ls[m][3.0]['lo']:.2f},{ls[m][3.0]['hi']:.2f}]")
    print("\n  Wilcoxon p (Physics vs each) @3s:")
    for m in ls:
        if m != "Physics":
            p = M.paired_wilcoxon(lper["Physics"][3.0], lper[m][3.0])
            print(f"    Physics vs {m:9}: p={p:.3f}")

    print("\n[comma2k19 held-out]")
    cs, cper = comma_experiment(max_train=60)
    for m in cs:
        print(f"  {m:9} FDE@3s = {cs[m][3.0]['mean']:.2f} [{cs[m][3.0]['lo']:.2f},{cs[m][3.0]['hi']:.2f}]")

    plot_summary(ls, "Loader (LOSO): FDE with 95% CI across baselines", "fig5_baselines_loader.png")
    plot_summary(cs, "comma2k19: FDE with 95% CI across baselines", "fig6_baselines_comma.png")

    with open(os.path.join(RES, "exp3_results.json"), "w") as f:
        json.dump({"loader": ls, "comma": cs}, f, indent=2)
    print("\nsaved results/exp3_results.json\nExperiment 3 done.")


if __name__ == "__main__":
    main()
