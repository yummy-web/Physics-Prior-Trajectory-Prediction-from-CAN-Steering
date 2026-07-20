"""
test_E_cross_domain_steering.py
测试T-E: "转向角->路径曲率"辨识范式的跨车型可迁移性

论点: 本文创新点1(从CAN转向信号辨识车辆曲率)不是装载机个案,
      而是一个可迁移的范式。在完全不同的车型(comma2k19乘用车,阿克曼转向)上
      同一辨识流程应同样成立(高R²), 且增益符合各自的运动学。

对每段拟合: curvature = a * steer + b, 报告 Pearson R²、增益a。
物理一致性:
  - 装载机(铰接): curvature ∝ 铰接角 (近似线性)
  - 乘用车(阿克曼): curvature = tan(delta)/L ≈ delta/L (小角线性), 增益与轴距L相关
"""
import os
import numpy as np
from scipy import stats
from comma_loader import load_many


def fit_seg(df, vmin=8.0):
    """在稳定行驶(速度>vmin)时拟合曲率~转向。"""
    m = df["obs_v"].values > vmin
    x = df["steer"].values[m]
    y = df["curvature"].values[m]
    # 去极端(数值微分噪声)
    ok = np.isfinite(x) & np.isfinite(y) & (np.abs(y) < np.nanpercentile(np.abs(y[np.isfinite(y)]), 97))
    x, y = x[ok], y[ok]
    if len(x) < 30:
        return None
    a, b, r, p, se = stats.linregress(x, y)
    return {"R2": r**2, "gain": a, "n": len(x), "steer_std": np.std(x), "v": np.mean(df["obs_v"].values[m])}


def main():
    print("=" * 76)
    print("  测试T-E: 转向->曲率辨识范式的跨车型验证 (comma2k19乘用车)")
    print("=" * 76)
    data = load_many(max_segments=60, min_speed=8.0, require_turning=True)
    print(f"\n加载 {len(data)} 段有效行驶数据 (速度>8m/s且有转向)。\n")

    rows = []
    for seg, df in data.items():
        r = fit_seg(df)
        if r and r["steer_std"] > 1.0:  # 只统计确有转向变化的段
            rows.append(r)

    R2s = np.array([r["R2"] for r in rows])
    gains = np.array([r["gain"] for r in rows])
    print(f"有效拟合段数: {len(rows)}")
    print(f"曲率~转向 R²:  中位={np.median(R2s):.3f}  均值={np.mean(R2s):.3f}  "
          f"[{np.percentile(R2s,25):.3f}, {np.percentile(R2s,75):.3f}] (IQR)")
    print(f"R²>0.7 的段占比: {100*np.mean(R2s>0.7):.0f}%   R²>0.5: {100*np.mean(R2s>0.5):.0f}%")
    print(f"增益 a(1/m per deg): 中位={np.median(gains):.2e}  (符号一致性: "
          f"{100*np.mean(np.sign(gains)==np.sign(np.median(gains))):.0f}%)")

    # 物理一致性: 阿克曼 kappa = delta_road/L. 若转向盘->前轮有转向比S, delta_road=steer/S
    # gain = 1/(L*S). 取典型 L=2.8m, 估计等效转向比
    L = 2.8
    med_gain_per_rad = np.median(gains) * 180/np.pi  # 1/m per rad(steer)
    S_est = 1.0/(L*med_gain_per_rad) if med_gain_per_rad != 0 else np.nan
    print(f"\n物理一致性检验(阿克曼 kappa=delta/L):")
    print(f"  假设轴距L={L}m, 反推等效转向比 S≈{S_est:.1f} : 1  (乘用车典型15-18:1, 同量级即合理)")

    print("\n结论: 若多数段 R²高 且 增益符号一致, 说明'转向->曲率'辨识范式跨车型成立,")
    print("      创新点1从'单车个案'升级为'可迁移范式'。")
    print("\nT-E完成。")
    return rows


if __name__ == "__main__":
    main()
