"""
data_loader.py
统一数据加载与时间对齐模块（里程碑测试基础）

功能:
  - 加载4段CAN(40Hz)与定位(10Hz)
  - 解码 motor_angle 编码器环绕(0~65530) -> 连续铰接角(度)
  - 按定位时间戳最近邻对齐CAN信号
  - 从RTK位置计算观测速度、航向、航向角速度、曲率
  - 输出统一 DataFrame，每段一条记录序列

作者: 里程碑测试 (CPU-only)
"""

import os
import glob
import numpy as np
import pandas as pd

EXTRACTED_DIR = os.path.join(os.path.dirname(__file__), "..", "extracted_data")
ENCODER_FULL = 65530.0

SEGMENTS = [
    "2026-07-10-08-51-23",
    "2026-07-10-08-52-25",
    "2026-07-10-08-53-53",
    "2026-07-10-08-55-09",
]


def unwrap_encoder(raw, full=ENCODER_FULL):
    """把0~full的编码器读数解码为连续值(处理环绕)。返回同单位(counts)的连续序列。"""
    rad = raw / full * 2.0 * np.pi
    cont = np.unwrap(rad) * full / (2.0 * np.pi)
    return cont


def _nearest(t_src, vals_src, t_query):
    """最近邻重采样: 对每个t_query, 取t_src中最接近的vals_src。"""
    idx = np.searchsorted(t_src, t_query)
    idx = np.clip(idx, 0, len(t_src) - 1)
    # 检查左邻是否更近
    left = np.clip(idx - 1, 0, len(t_src) - 1)
    use_left = np.abs(t_src[left] - t_query) < np.abs(t_src[idx] - t_query)
    idx = np.where(use_left, left, idx)
    return vals_src[idx]


def load_can_topic(seg, topic):
    path = os.path.join(EXTRACTED_DIR, f"can_{topic}_{seg}.csv")
    return pd.read_csv(path)


def load_segment(seg):
    """加载并对齐单段数据，返回统一DataFrame(以10Hz定位为基准)。"""
    loc = pd.read_csv(os.path.join(EXTRACTED_DIR, f"location_{seg}.csv"))
    lt = loc["time_sec"].values.astype(float)
    lx = loc["info_0_x"].values.astype(float)
    ly = loc["info_0_y"].values.astype(float)
    lyaw = loc["info_0_yaw"].values.astype(float)     # 度
    lspd = loc["info_0_speed"].values.astype(float)   # 报告速度(存疑,大量为0)

    # CAN
    c011 = load_can_topic(seg, "topic_011E")  # footboard, arm_handle, bucket_handle, arm_tension
    c021 = load_can_topic(seg, "topic_021E")  # oil_door, bucket_tension
    c018 = load_can_topic(seg, "topic_0180")  # motor_angle, motor_rpm, motor_direction

    t011 = c011["time_sec"].values.astype(float)
    t021 = c021["time_sec"].values.astype(float)
    t018 = c018["time_sec"].values.astype(float)

    footboard = c011["footboard"].values.astype(float)
    oil_door = c021["oil_door"].values.astype(float)
    motor_angle_raw = c018["motor_angle"].values.astype(float)
    motor_rpm = c018["motor_rpm"].values.astype(float)
    motor_dir = c018["motor_direction"].values.astype(float)
    arm_tension = c011["arm_tension"].values.astype(float) if "arm_tension" in c011 else np.zeros_like(t011)
    bucket_tension = c021["bucket_tension"].values.astype(float) if "bucket_tension" in c021 else np.zeros_like(t021)

    # 解码铰接角(整段一起解码保证连续)
    motor_angle_cont = unwrap_encoder(motor_angle_raw)

    # 以定位时间为基准对齐CAN
    a_foot = _nearest(t011, footboard, lt)
    a_oil = _nearest(t021, oil_door, lt)
    a_mang = _nearest(t018, motor_angle_cont, lt)
    a_mraw = _nearest(t018, motor_angle_raw, lt)
    a_mrpm = _nearest(t018, motor_rpm, lt)
    a_mdir = _nearest(t018, motor_dir, lt)
    a_armT = _nearest(t011, arm_tension, lt)
    a_bktT = _nearest(t021, bucket_tension, lt)

    # 从位置计算运动学真值
    dt = np.gradient(lt)
    vx = np.gradient(lx) / dt
    vy = np.gradient(ly) / dt
    obs_v = np.hypot(vx, vy)                     # 观测速度(m/s)
    heading = np.degrees(np.arctan2(vy, vx))     # 运动方向(度)
    yaw_unwrap = np.degrees(np.unwrap(np.radians(lyaw)))
    yaw_rate = np.gradient(yaw_unwrap) / dt      # 航向角速度(度/s)
    curvature = np.radians(yaw_rate) / (obs_v + 1e-3)  # 曲率(1/m)

    df = pd.DataFrame({
        "seg": seg,
        "t": lt,
        "x": lx, "y": ly, "yaw": lyaw, "yaw_unwrap": yaw_unwrap,
        "reported_speed": lspd,
        "obs_v": obs_v, "heading": heading,
        "yaw_rate": yaw_rate, "curvature": curvature,
        "footboard": a_foot, "oil_door": a_oil,
        "motor_angle_raw": a_mraw, "motor_angle": a_mang,
        "motor_rpm": a_mrpm, "motor_direction": a_mdir,
        "arm_tension": a_armT, "bucket_tension": a_bktT,
    })
    return df


def load_all():
    return {seg: load_segment(seg) for seg in SEGMENTS}


if __name__ == "__main__":
    data = load_all()
    print("=" * 70)
    print("  统一数据加载与对齐 — 概览")
    print("=" * 70)
    for seg, df in data.items():
        dur = df["t"].iloc[-1] - df["t"].iloc[0]
        pathlen = np.sum(np.hypot(np.diff(df["x"]), np.diff(df["y"])))
        print(f"\n[{seg}] pts={len(df)} dur={dur:.1f}s pathlen={pathlen:.1f}m")
        print(f"  obs_v: min/mean/max = {df['obs_v'].min():.2f}/{df['obs_v'].mean():.2f}/{df['obs_v'].max():.2f} m/s")
        print(f"  yaw_rate: [{df['yaw_rate'].min():.1f}, {df['yaw_rate'].max():.1f}] deg/s")
        print(f"  motor_angle(解码后连续): [{df['motor_angle'].min():.0f}, {df['motor_angle'].max():.0f}]")
        print(f"  footboard: [{df['footboard'].min():.0f}, {df['footboard'].max():.0f}]  "
              f"oil_door: [{df['oil_door'].min():.0f}, {df['oil_door'].max():.0f}]")
    print("\n加载模块OK。")
