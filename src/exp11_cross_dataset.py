"""
exp11_cross_dataset.py
Cross-DATASET validation: the "CAN steering -> path curvature" identification and
control-conditioned prediction hold on THREE independent vehicle classes:
  - Articulated loader (our real data)
  - comma2k19 (various sedans)
  - nuScenes CAN expansion (Renault Zoe)

Outputs: figures/fig15_cross_dataset_R2.png, fig16_cross_dataset_pred.png
         results/exp11_results.json
"""
import os, json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import models_core as M
from plot_utils import save

RES = os.path.join(os.path.dirname(__file__), "results"); os.makedirs(RES, exist_ok=True)


def steer_r2(data, vmin, steer_min, motor=False):
    r2s, gains = [], []
    for sc, df in data.items():
        m = df["obs_v"].values > vmin
        if motor:
            mask = (np.abs(df["yaw_rate"].values) < 3) & (df["obs_v"].values > 1.0)
            c = np.median(df["motor_angle"].values[mask]) if mask.sum() > 5 else np.median(df["motor_angle"].values)
            x = (df["motor_angle"].values - c)[m]
        else:
            x = df["steer"].values[m]
        y = df["curvature"].values[m]
        ok = np.isfinite(x) & np.isfinite(y) & (np.abs(y) < np.nanpercentile(np.abs(y[np.isfinite(y)]), 97))
        if ok.sum() > 30 and np.std(x[ok]) > steer_min:
            a, b, r, p, se = stats.linregress(x[ok], y[ok])
            r2s.append(r ** 2); gains.append(a)
    return np.array(r2s), np.array(gains)


def main():
    print("=" * 70); print("  Exp11: cross-DATASET validation (loader + comma2k19 + nuScenes)"); print("=" * 70)
    results = {}

    # loader
    from loader_data_loader import load_all as load_loader
    ld = load_loader()
    r2_l, g_l = steer_r2(ld, 0.5, 100, motor=True)
    results["loader"] = {"n": int(len(r2_l)), "R2_median": float(np.median(r2_l)),
                         "R2_mean": float(np.mean(r2_l)), "frac_gt07": float(np.mean(r2_l > 0.7))}

    # comma
    from comma_loader import load_all_cached as load_comma
    cd = load_comma(verbose=False)
    r2_c, g_c = steer_r2(cd, 8.0, 1.0)
    results["comma2k19"] = {"n": int(len(r2_c)), "R2_median": float(np.median(r2_c)),
                            "R2_mean": float(np.mean(r2_c)), "frac_gt07": float(np.mean(r2_c > 0.7))}

    # nuscenes
    from nuscenes_loader import load_all_cached as load_nusc
    nd = load_nusc(verbose=False)
    r2_n, g_n = steer_r2(nd, 3.0, 5.0)
    results["nuscenes"] = {"n": int(len(r2_n)), "R2_median": float(np.median(r2_n)),
                           "R2_mean": float(np.mean(r2_n)), "frac_gt07": float(np.mean(r2_n > 0.7))}

    print(f"\n{'dataset':<14}{'n':>6}{'R2 median':>12}{'R2>0.7':>10}")
    for k in ["loader", "comma2k19", "nuscenes"]:
        r = results[k]
        print(f"{k:<14}{r['n']:>6}{r['R2_median']:>12.3f}{100*r['frac_gt07']:>9.0f}%")

    # Fig15: boxplot of per-segment R2 across three datasets
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([r2_l, r2_c, r2_n], labels=["Loader\n(articulated)", "comma2k19\n(sedans)", "nuScenes\n(Renault Zoe)"],
               showmeans=True)
    ax.axhline(0.7, color="gray", ls=":", label="R2=0.7")
    ax.set_ylabel("Per-segment R2 (steering -> curvature)")
    ax.set_title("Innovation 1 generalizes across 3 independent vehicle classes")
    ax.legend(); save(fig, "fig15_cross_dataset_R2.png")

    # nuScenes control-conditioned prediction vs kinematic baselines
    import baselines as B
    DT_N = 0.02  # ~50Hz pose
    segs = list(nd.values())
    # estimate dt from data
    dts = [np.median(np.diff(df["t"].values)) for df in segs[:20]]
    DT_N = float(np.median(dts))
    HS = int(round(3.0 / DT_N))
    g = M.comma_gain(segs)  # same steering->yawrate calibration
    phys, ctrv, cv = [], [], []
    for df in segs[:200]:
        if len(df) <= HS + 2:
            continue
        phys.extend(M.comma_phys_metrics(df, HS, DT_N, g)["fde"])
        ctrv.extend(M.comma_kin_metrics(df, HS, DT_N, "CTRV")["fde"])
        cv.extend(M.comma_kin_metrics(df, HS, DT_N, "CV")["fde"])
    results["nuscenes_pred"] = {"physics_fde3": float(np.mean(phys)),
                                "ctrv_fde3": float(np.mean(ctrv)),
                                "cv_fde3": float(np.mean(cv)), "dt": DT_N}
    print(f"\nnuScenes control-conditioned FDE@3s: physics={np.mean(phys):.2f} CTRV={np.mean(ctrv):.2f} CV={np.mean(cv):.2f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["CV", "CTRV", "Physics\n(uses steering)"], [np.mean(cv), np.mean(ctrv), np.mean(phys)],
           color=["gray", "gray", "tab:red"])
    ax.set_ylabel("FDE@3s (m)"); ax.set_title("nuScenes: control-conditioned physics vs motion extrapolation")
    save(fig, "fig16_cross_dataset_pred.png")

    with open(os.path.join(RES, "exp11_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved results/exp11_results.json\nExp11 done.")


if __name__ == "__main__":
    main()
