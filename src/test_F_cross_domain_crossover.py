"""
test_F_cross_domain_crossover.py
测试T-F: 在comma2k19上验证两大普适规律
  (F1) 数据稀缺交叉点: 物理先验(转向->曲率+车速) 在少数据下优于数据驱动;
       数据驱动需更多段才追平。
  (F2) 误差各向异性: 轨迹预测的横向误差(由转向决定)与纵向误差(由速度决定)贡献不同,
       量化谁主导 —— 与装载机"转向易/速度难"呼应的普适版本。

方法: 控制条件轨迹预测, 锚点+未来H秒控制 -> 预测(x,y), 报告FDE。
  物理P: yaw_rate = g*steer (g由训练段标定), v=CAN实测速度, 运动学积分
  数据D: 小MLP直接回归H秒后局部位移, 用k段训练
  留出固定测试段评估, k递增看交叉。
"""
import os
import numpy as np
from scipy import stats
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from comma_loader import load_many

DT = 0.05  # comma 20Hz
H = 3.0
HS = int(round(H / DT))


def calibrate_gain(train):
    xs, ys = [], []
    for df in train:
        m = df["obs_v"].values > 8.0
        x = df["steer"].values[m]
        y = np.radians(df["yaw_rate"].values)[m]
        ok = np.isfinite(x) & np.isfinite(y) & (np.abs(y) < np.nanpercentile(np.abs(y[np.isfinite(y)]), 97))
        xs.append(x[ok]); ys.append(y[ok])
    g, b, r, p, se = stats.linregress(np.concatenate(xs), np.concatenate(ys))
    return g


def physics_fde(df, g):
    n = len(df); fdes = []; lat=[]; lon=[]
    for i0 in range(0, n - HS, 5):
        x = df["x"].values[i0]; y = df["y"].values[i0]
        yaw = np.radians(df["yaw_unwrap"].values[i0])
        yaw0 = yaw
        for k in range(HS):
            j = min(i0 + k, n - 1)
            v = df["obs_v"].values[j]
            yaw += g * df["steer"].values[j] * DT
            x += v * np.cos(yaw) * DT; y += v * np.sin(yaw) * DT
        gx = df["x"].values[i0+HS]; gy = df["y"].values[i0+HS]
        ex, ey = x-gx, y-gy
        fdes.append(np.hypot(ex, ey))
        # 分解到锚点航向的 纵向/横向
        c,s=np.cos(yaw0),np.sin(yaw0)
        lon.append(abs(c*ex+s*ey)); lat.append(abs(-s*ex+c*ey))
    return np.mean(fdes), np.mean(lon), np.mean(lat)


def build_direct(segs):
    X, Y = [], []
    for df in segs:
        n = len(df)
        for i0 in range(0, n - HS, 5):
            fut = slice(i0, i0 + HS)
            feat = [df["obs_v"].values[i0], np.radians(df["yaw_rate"].values[i0]),
                    np.mean(df["obs_v"].values[fut]), np.mean(df["steer"].values[fut]),
                    np.std(df["steer"].values[fut])]
            yaw = np.radians(df["yaw_unwrap"].values[i0])
            dx = df["x"].values[i0+HS]-df["x"].values[i0]; dy = df["y"].values[i0+HS]-df["y"].values[i0]
            c,s=np.cos(-yaw),np.sin(-yaw)
            X.append(feat); Y.append([c*dx-s*dy, s*dx+c*dy])
    return np.array(X), np.array(Y)


def direct_fde(test_df, sc, mdl):
    n=len(test_df); fdes=[]
    for i0 in range(0, n-HS, 5):
        fut=slice(i0,i0+HS)
        feat=np.array([[test_df["obs_v"].values[i0], np.radians(test_df["yaw_rate"].values[i0]),
              np.mean(test_df["obs_v"].values[fut]), np.mean(test_df["steer"].values[fut]),
              np.std(test_df["steer"].values[fut])]])
        lx,ly=mdl.predict(sc.transform(feat))[0]
        yaw=np.radians(test_df["yaw_unwrap"].values[i0]); c,s=np.cos(yaw),np.sin(yaw)
        ddx=c*lx-s*ly; ddy=s*lx+c*ly
        fdes.append(np.hypot((test_df["x"].values[i0]+ddx)-test_df["x"].values[i0+HS],
                             (test_df["y"].values[i0]+ddy)-test_df["y"].values[i0+HS]))
    return np.mean(fdes)


def main():
    print("=" * 76)
    print("  测试T-F: comma2k19上验证 数据稀缺交叉点 + 误差各向异性")
    print("=" * 76)
    data = load_many(max_segments=70, min_speed=8.0, require_turning=True)
    segs = [df for df in data.values() if df["steer"].std() > 1.0]
    print(f"\n有效段: {len(segs)} (各~60s)")

    # 固定测试集: 最后10段; 训练池: 其余
    test_segs = segs[-10:]
    pool = segs[:-10]
    g = calibrate_gain(pool)

    # 物理基线(仅需g)
    pf = [physics_fde(df, g) for df in test_segs]
    phys_fde = np.mean([p[0] for p in pf])
    lon = np.mean([p[1] for p in pf]); lat = np.mean([p[2] for p in pf])
    print(f"\n铰接/阿克曼物理模型 FDE@{H}s = {phys_fde:.2f} m  (仅需转向增益g标定)")
    print(f"\n(F2) 误差各向异性: 纵向(速度方向)={lon:.2f}m  横向(转向方向)={lat:.2f}m  "
          f"-> {'纵向' if lon>lat else '横向'}主导 (比值{max(lon,lat)/(min(lon,lat)+1e-6):.1f}x)")

    print(f"\n(F1) 数据驱动 FDE@{H}s 随训练段数k:")
    print("-"*60)
    print(f"{'k段':<8}{'≈时长':<12}{'FDE(m)':<12}{'vs物理':<12}")
    for k in [1,2,4,8,16,32,min(len(pool),50)]:
        if k>len(pool): break
        Xtr,Ytr=build_direct(pool[:k])
        sc=StandardScaler().fit(Xtr)
        mdl=MLPRegressor(hidden_layer_sizes=(48,24),max_iter=1000,alpha=1e-3,
                         random_state=0,early_stopping=True).fit(sc.transform(Xtr),Ytr)
        dfde=np.mean([direct_fde(df,sc,mdl) for df in test_segs])
        rel=(phys_fde-dfde)/phys_fde*100
        flag="  <== 追平/反超" if dfde<=phys_fde else ""
        print(f"{k:<8}{f'{k*60}s':<12}{dfde:<12.2f}{rel:+.1f}%{flag}")

    print(f"\n结论: 若物理在k小时领先、数据驱动需较多段才追平 => 交叉点规律跨车型普适(创新点3)。")
    print("\nT-F完成。")


if __name__ == "__main__":
    main()
