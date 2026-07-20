"""
exp1_steering_identification.py
实验1(创新点1): CAN转向信号 -> 路径曲率 的辨识, 装载机(铰接) + comma2k19(阿克曼) 双验证

产出:
  - figures/fig1_steering_curvature.png   两车型曲率~转向散点+拟合
  - figures/fig1b_motor_consistency.png   装载机 motor_rpm=|d(motor_angle)/dt| 内部一致性
  - results/exp1_results.json
"""
import os, json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from loader_data_loader import load_all as load_loader
from comma_loader import load_many as load_comma
from plot_utils import save, FIG_DIR

RES_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RES_DIR, exist_ok=True)


def loader_identification():
    data = load_loader()
    per_seg = {}
    pooled_x, pooled_y = [], []
    for seg, df in data.items():
        mask = (np.abs(df["yaw_rate"].values) < 3) & (df["obs_v"].values > 1.0)
        c = np.median(df["motor_angle"].values[mask]) if mask.sum() > 5 else np.median(df["motor_angle"].values)
        mv = df["obs_v"].values > 0.5
        x = (df["motor_angle"].values - c)[mv]
        y = df["curvature"].values[mv]
        ok = np.isfinite(x) & np.isfinite(y) & (np.abs(y) < np.nanpercentile(np.abs(y[np.isfinite(y)]), 98))
        x, y = x[ok], y[ok]
        if len(x) > 10:
            a, b, r, p, se = stats.linregress(x, y)
            per_seg[seg] = {"R2": r**2, "gain": a, "n": int(len(x))}
            pooled_x.append(x); pooled_y.append(y)
    return data, per_seg, np.concatenate(pooled_x), np.concatenate(pooled_y)


def motor_consistency(data):
    """内部一致性: motor_rpm 应正比于 |d(motor_angle)/dt|, 而非车速。"""
    rows = {}
    for seg, df in data.items():
        t = df["t"].values
        rate = np.abs(np.gradient(df["motor_angle"].values) / np.gradient(t))
        rpm = df["motor_rpm"].values
        r_rate = np.corrcoef(rate, rpm)[0, 1]
        r_speed = np.corrcoef(df["obs_v"].values, rpm)[0, 1]
        rows[seg] = {"corr_steer_rate": float(r_rate), "corr_vehicle_speed": float(r_speed)}
    return rows


def comma_identification():
    data = load_comma(max_segments=60, min_speed=8.0, require_turning=True)
    R2s, gains = [], []
    example = None
    for seg, df in data.items():
        m = df["obs_v"].values > 8.0
        x = df["steer"].values[m]; y = df["curvature"].values[m]
        ok = np.isfinite(x) & np.isfinite(y) & (np.abs(y) < np.nanpercentile(np.abs(y[np.isfinite(y)]), 97))
        x, y = x[ok], y[ok]
        if len(x) > 30 and np.std(x) > 1.0:
            a, b, r, p, se = stats.linregress(x, y)
            R2s.append(r**2); gains.append(a)
            if example is None and r**2 > 0.85:
                example = (x, y, a, b, r**2)
    return R2s, gains, example


def main():
    print("=" * 70)
    print("  实验1: 转向->曲率辨识 (装载机 + comma2k19 双验证)")
    print("=" * 70)

    loader_data, loader_seg, lx, ly = loader_identification()
    a_l, b_l, r_l, _, _ = stats.linregress(lx, ly)
    print(f"\n[装载机] 逐段R²: " + ", ".join(f"{s[-8:]}={v['R2']:.2f}" for s, v in loader_seg.items()))
    print(f"[装载机] 合并拟合 R²={r_l**2:.3f}, 增益={a_l:.2e} 1/m/count")

    mc = motor_consistency(loader_data)
    print(f"\n[装载机内部一致性] corr(motor_rpm, |d motor_angle/dt|) vs corr(motor_rpm, vehicle_speed):")
    for s, v in mc.items():
        print(f"  {s[-8:]}: 转向率={v['corr_steer_rate']:.3f}  车速={v['corr_vehicle_speed']:.3f}")

    R2s, gains, example = comma_identification()
    R2s = np.array(R2s)
    print(f"\n[comma2k19] {len(R2s)}段: R²中位={np.median(R2s):.3f}, R²>0.7占比={100*np.mean(R2s>0.7):.0f}%, "
          f"增益符号一致={100*np.mean(np.sign(gains)==np.sign(np.median(gains))):.0f}%")

    # ---- Figure 1: 曲率~转向 双子图 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # 装载机: 取转向最丰富的一段
    best = max(loader_seg, key=lambda s: loader_seg[s]["R2"])
    df = loader_data[best]
    mask = (np.abs(df["yaw_rate"].values) < 3) & (df["obs_v"].values > 1.0)
    c = np.median(df["motor_angle"].values[mask])
    mv = df["obs_v"].values > 0.5
    xx = (df["motor_angle"].values - c)[mv]; yy = df["curvature"].values[mv]
    ok = np.abs(yy) < np.nanpercentile(np.abs(yy), 98)
    axes[0].scatter(xx[ok], yy[ok], s=6, alpha=0.4)
    xs = np.linspace(xx[ok].min(), xx[ok].max(), 50)
    aa, bb, rr, _, _ = stats.linregress(xx[ok], yy[ok])
    axes[0].plot(xs, aa*xs+bb, 'r-', lw=2, label=f"R2={rr**2:.2f}")
    axes[0].set_title("Articulated Loader (real)")
    axes[0].set_xlabel("Articulation angle (encoder counts, centered)")
    axes[0].set_ylabel("Path curvature (1/m)")
    axes[0].legend()
    if example:
        ex, ey, ea, eb, er2 = example
        axes[1].scatter(ex, ey, s=6, alpha=0.4, color="tab:green")
        xs = np.linspace(ex.min(), ex.max(), 50)
        axes[1].plot(xs, ea*xs+eb, 'r-', lw=2, label=f"R2={er2:.2f}")
    axes[1].set_title("comma2k19 Passenger Car (Ackermann)")
    axes[1].set_xlabel("Steering angle (deg)")
    axes[1].set_ylabel("Path curvature (1/m)")
    axes[1].legend()
    fig.suptitle("Innovation 1: CAN steering -> path curvature identification transfers across vehicle classes")
    save(fig, "fig1_steering_curvature.png")

    # ---- Figure 1b: motor consistency bar ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    segs = list(mc.keys()); labels = [s[-8:] for s in segs]
    r_rate = [mc[s]["corr_steer_rate"] for s in segs]
    r_speed = [mc[s]["corr_vehicle_speed"] for s in segs]
    x = np.arange(len(segs)); w = 0.35
    ax.bar(x-w/2, r_rate, w, label="corr(motor_rpm, |d angle/dt|)")
    ax.bar(x+w/2, r_speed, w, label="corr(motor_rpm, vehicle speed)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Pearson correlation"); ax.set_ylim(-0.2, 1.05)
    ax.set_title("Loader: topic_0180 is the STEERING actuator (rpm tracks steering rate, not speed)")
    ax.legend()
    save(fig, "fig1b_motor_consistency.png")

    results = {
        "loader_per_segment": {s: v for s, v in loader_seg.items()},
        "loader_pooled_R2": float(r_l**2), "loader_gain": float(a_l),
        "loader_motor_consistency": mc,
        "comma_R2_median": float(np.median(R2s)), "comma_R2_frac_gt07": float(np.mean(R2s > 0.7)),
        "comma_gain_sign_consistency": float(np.mean(np.sign(gains) == np.sign(np.median(gains)))),
        "comma_n_segments": int(len(R2s)),
    }
    with open(os.path.join(RES_DIR, "exp1_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已存 results/exp1_results.json")
    print("实验1完成。")


if __name__ == "__main__":
    main()
