"""
exp2_prediction_and_crossover.py
实验2(创新点2+3): 轨迹预测基线对比 + 数据稀缺交叉点 + 误差各向异性

内容:
  A. 装载机 LOSO: 纯物理 vs 纯数据驱动 vs 匀速基线, 多时长 ADE/FDE
  B. 数据稀缺交叉点(装载机, 用标定仿真器) + comma2k19 交叉点
  C. 误差各向异性: 装载机(纵向/速度主导) vs comma(横向/转向主导)

产出:
  - figures/fig2_prediction_fde.png     LOSO FDE曲线
  - figures/fig3_crossover.png          两车型交叉点
  - figures/fig4_error_anisotropy.png   误差各向异性对比
  - results/exp2_results.json
"""
import os, json
import numpy as np
from scipy import stats
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from loader_data_loader import load_all as load_loader
from comma_loader import load_many as load_comma
from plot_utils import save

RES_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RES_DIR, exist_ok=True)
RNG = np.random.default_rng(42)


# ============ 装载机部分 ============
DT_L = 0.1
HORIZONS = [1.0, 2.0, 3.0, 5.0]


def loader_gain(data, segs):
    xs, ys = [], []
    for s in segs:
        df = data[s]
        mask = (np.abs(df["yaw_rate"].values) < 3) & (df["obs_v"].values > 1.0)
        c = np.median(df["motor_angle"].values[mask]) if mask.sum() > 5 else np.median(df["motor_angle"].values)
        mv = df["obs_v"].values > 0.5
        x = (df["motor_angle"].values - c)[mv]; y = np.radians(df["yaw_rate"].values)[mv]
        ok = np.abs(y) < np.nanpercentile(np.abs(y), 98)
        xs.append(x[ok]); ys.append(y[ok])
    g, *_ = stats.linregress(np.concatenate(xs), np.concatenate(ys))
    return g


def loader_center(df, warmup=True):
    if warmup:
        sub = df.iloc[:int(3.0/DT_L)]
        mask = (np.abs(sub["yaw_rate"].values) < 4) & (sub["obs_v"].values > 0.8)
        return np.median(sub["motor_angle"].values[mask]) if mask.sum() > 3 else np.median(sub["motor_angle"].values)
    mask = (np.abs(df["yaw_rate"].values) < 3) & (df["obs_v"].values > 1.0)
    return np.median(df["motor_angle"].values[mask]) if mask.sum() > 5 else np.median(df["motor_angle"].values)


def loader_speed_coef(data, segs):
    X, y = [], []
    for s in segs:
        df = data[s]; m = df["obs_v"].values > 0.2
        X.append(np.column_stack([df["footboard"].values[m], df["oil_door"].values[m]])); y.append(df["obs_v"].values[m])
    X = np.vstack(X); y = np.concatenate(y); A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None); return coef


def loader_speed(df, j, coef):
    return max(coef[0] + coef[1]*df["footboard"].values[j] + coef[2]*df["oil_door"].values[j], 0.0)


def loader_phys_rollout(df, i0, hs, g, c, coef, cv=False):
    x = df["x"].values[i0]; y = df["y"].values[i0]; yaw = np.radians(df["yaw_unwrap"].values[i0])
    for k in range(hs):
        j = min(i0+k, len(df)-1)
        v = loader_speed(df, j, coef)
        if not cv:
            yaw += g*(df["motor_angle"].values[j]-c)*DT_L
        x += v*np.cos(yaw)*DT_L; y += v*np.sin(yaw)*DT_L
    return x, y


def loader_direct_dataset(data, segs, hs):
    X, Y = [], []
    for s in segs:
        df = data[s]; n = len(df); c = loader_center(df, warmup=False)
        for i0 in range(0, n-hs, 2):
            fut = slice(i0, i0+hs)
            feat = [df["obs_v"].values[i0], np.radians(df["yaw_rate"].values[i0]),
                    np.mean(df["footboard"].values[fut]), np.mean(df["oil_door"].values[fut]),
                    np.mean(df["motor_angle"].values[fut]-c), np.std(df["motor_angle"].values[fut])]
            yaw = np.radians(df["yaw_unwrap"].values[i0])
            dx = df["x"].values[i0+hs]-df["x"].values[i0]; dy = df["y"].values[i0+hs]-df["y"].values[i0]
            cc, ss = np.cos(-yaw), np.sin(-yaw)
            X.append(feat); Y.append([cc*dx-ss*dy, ss*dx+cc*dy])
    return np.array(X), np.array(Y)


def loader_loso():
    data = load_loader(); segs = list(data.keys())
    res = {m: {h: [] for h in HORIZONS} for m in ["CV", "Physics", "DataDriven"]}
    aniso = {"lon": [], "lat": []}
    for test in segs:
        train = [s for s in segs if s != test]
        g = loader_gain(data, train); coef = loader_speed_coef(data, train)
        c_test = loader_center(data[test], warmup=True)
        df = data[test]; n = len(df)
        for H in HORIZONS:
            hs = int(round(H/DT_L))
            Xtr, Ytr = loader_direct_dataset(data, train, hs)
            sc = StandardScaler().fit(Xtr)
            mdl = MLPRegressor(hidden_layer_sizes=(48,24), max_iter=1000, alpha=1e-3,
                               random_state=0, early_stopping=True).fit(sc.transform(Xtr), Ytr)
            for i0 in range(0, n-hs, 2):
                gx = df["x"].values[i0+hs]; gy = df["y"].values[i0+hs]
                # physics
                px, py = loader_phys_rollout(df, i0, hs, g, c_test, coef)
                res["Physics"][H].append(np.hypot(px-gx, py-gy))
                if H == 3.0:
                    yaw0 = np.radians(df["yaw_unwrap"].values[i0]); ex, ey = px-gx, py-gy
                    cc, ss = np.cos(yaw0), np.sin(yaw0)
                    aniso["lon"].append(abs(cc*ex+ss*ey)); aniso["lat"].append(abs(-ss*ex+cc*ey))
                # CV
                cx, cy = loader_phys_rollout(df, i0, hs, g, c_test, coef, cv=True)
                res["CV"][H].append(np.hypot(cx-gx, cy-gy))
                # data-driven
                fut = slice(i0, i0+hs)
                feat = np.array([[df["obs_v"].values[i0], np.radians(df["yaw_rate"].values[i0]),
                        np.mean(df["footboard"].values[fut]), np.mean(df["oil_door"].values[fut]),
                        np.mean(df["motor_angle"].values[fut]-c_test), np.std(df["motor_angle"].values[fut])]])
                lx, ly = mdl.predict(sc.transform(feat))[0]
                yaw = np.radians(df["yaw_unwrap"].values[i0]); cc, ss = np.cos(yaw), np.sin(yaw)
                ddx = cc*lx-ss*ly; ddy = ss*lx+cc*ly
                res["DataDriven"][H].append(np.hypot((df["x"].values[i0]+ddx)-gx, (df["y"].values[i0]+ddy)-gy))
    fde = {m: {h: float(np.mean(res[m][h])) for h in HORIZONS} for m in res}
    return fde, {"lon": float(np.mean(aniso["lon"])), "lat": float(np.mean(aniso["lat"]))}


# ============ 交叉点(装载机仿真器) ============
def loader_calibrate_sim(data):
    g = loader_gain(data, list(data.keys()))
    speeds = np.concatenate([data[s]["obs_v"].values for s in data]); speeds = speeds[speeds > 0.2]
    xs, ys = [], []
    for s in data:
        df = data[s]; mask=(np.abs(df["yaw_rate"].values)<3)&(df["obs_v"].values>1.0)
        c=np.median(df["motor_angle"].values[mask]); mv=df["obs_v"].values>0.5
        x=(df["motor_angle"].values-c)[mv]; y=np.radians(df["yaw_rate"].values)[mv]
        ok=np.abs(y)<np.nanpercentile(np.abs(y),98); xs.append(x[ok]);ys.append(y[ok])
    resid=np.concatenate(ys)-g*np.concatenate(xs)
    return {"g":g,"sm":np.mean(speeds),"ss":np.std(speeds),"noise":np.std(resid)}


def loader_sim_segment(cal, length=480):
    g=cal["g"]; x=0;y=0;yaw=RNG.uniform(-np.pi,np.pi); c=40000.0; v=cal["sm"]
    xs=[];ys=[];yaws=[];vs=[];mang=[];yr=[]
    ta=c; tv=v
    for k in range(length):
        if k%40==0:
            tv=np.clip(RNG.normal(cal["sm"],cal["ss"]),0.2,4.5)
            ta=c+RNG.uniform(-1,1)/(g if g else 1)*RNG.uniform(0,0.4)
        v+=0.1*(tv-v)+RNG.normal(0,0.05); v=max(v,0)
        m=ta+RNG.normal(0,200); dy=g*(m-c)+RNG.normal(0,cal["noise"])
        yaw+=dy*DT_L; x+=v*np.cos(yaw)*DT_L+RNG.normal(0,0.01); y+=v*np.sin(yaw)*DT_L+RNG.normal(0,0.01)
        xs.append(x);ys.append(y);yaws.append(yaw);vs.append(v);mang.append(m);yr.append(np.degrees(dy))
    return {"x":np.array(xs),"y":np.array(ys),"yaw_unwrap":np.degrees(np.array(yaws)),
            "obs_v":np.array(vs),"yaw_rate":np.array(yr),"motor_angle":np.array(mang),"c":c}


def loader_crossover(data):
    cal = loader_calibrate_sim(data); hs = int(3.0/DT_L)
    # physics on real
    g = cal["g"]; fdes=[]
    for s,df in data.items():
        c=loader_center(df,warmup=True); n=len(df)
        for i0 in range(0,n-hs,2):
            x=df["x"].values[i0];y=df["y"].values[i0];yaw=np.radians(df["yaw_unwrap"].values[i0])
            for k in range(hs):
                j=min(i0+k,n-1); v=df["obs_v"].values[j]
                yaw+=g*(df["motor_angle"].values[j]-c)*DT_L; x+=v*np.cos(yaw)*DT_L;y+=v*np.sin(yaw)*DT_L
            fdes.append(np.hypot(x-df["x"].values[i0+hs],y-df["y"].values[i0+hs]))
    phys=float(np.mean(fdes))
    ks=[1,2,4,8,16,32,64]; pool=[loader_sim_segment(cal) for _ in range(max(ks))]
    curve=[]
    for k in ks:
        X,Y=[],[]
        for sim in pool[:k]:
            n=len(sim["x"]);c=sim["c"]
            for i0 in range(0,n-hs,2):
                fut=slice(i0,i0+hs)
                feat=[sim["obs_v"][i0],np.radians(sim["yaw_rate"][i0]),np.mean(sim["obs_v"][fut]),
                      np.mean(sim["motor_angle"][fut]-c),np.std(sim["motor_angle"][fut])]
                yaw=np.radians(sim["yaw_unwrap"][i0]); dx=sim["x"][i0+hs]-sim["x"][i0];dy=sim["y"][i0+hs]-sim["y"][i0]
                cc,ss=np.cos(-yaw),np.sin(-yaw); X.append(feat);Y.append([cc*dx-ss*dy,ss*dx+cc*dy])
        X=np.array(X);Y=np.array(Y); sc=StandardScaler().fit(X)
        mdl=MLPRegressor(hidden_layer_sizes=(48,24),max_iter=1000,alpha=1e-3,random_state=0,early_stopping=True).fit(sc.transform(X),Y)
        fd=[]
        for s,df in data.items():
            c=np.median(df["motor_angle"].values); n=len(df)
            for i0 in range(0,n-hs,2):
                fut=slice(i0,i0+hs)
                feat=np.array([[df["obs_v"].values[i0],np.radians(df["yaw_rate"].values[i0]),np.mean(df["obs_v"].values[fut]),
                      np.mean(df["motor_angle"].values[fut]-c),np.std(df["motor_angle"].values[fut])]])
                lx,ly=mdl.predict(sc.transform(feat))[0]; yaw=np.radians(df["yaw_unwrap"].values[i0]);cc,ss=np.cos(yaw),np.sin(yaw)
                ddx=cc*lx-ss*ly;ddy=ss*lx+cc*ly
                fd.append(np.hypot((df["x"].values[i0]+ddx)-df["x"].values[i0+hs],(df["y"].values[i0]+ddy)-df["y"].values[i0+hs]))
        curve.append((k*48, float(np.mean(fd))))
    return phys, curve


# ============ comma2k19 交叉点 + 各向异性 ============
DT_C = 0.05
def comma_crossover_aniso():
    data = load_comma(max_segments=70, min_speed=8.0, require_turning=True)
    segs = [df for df in data.values() if df["steer"].std() > 1.0]
    test_segs = segs[-10:]; pool = segs[:-10]; hs = int(3.0/DT_C)
    xs,ys=[],[]
    for df in pool:
        m=df["obs_v"].values>8.0; x=df["steer"].values[m];y=np.radians(df["yaw_rate"].values)[m]
        ok=np.isfinite(x)&np.isfinite(y)&(np.abs(y)<np.nanpercentile(np.abs(y[np.isfinite(y)]),97)); xs.append(x[ok]);ys.append(y[ok])
    g,*_=stats.linregress(np.concatenate(xs),np.concatenate(ys))
    # physics + aniso
    fdes=[];lon=[];lat=[]
    for df in test_segs:
        n=len(df)
        for i0 in range(0,n-hs,5):
            x=df["x"].values[i0];y=df["y"].values[i0];yaw=np.radians(df["yaw_unwrap"].values[i0]);yaw0=yaw
            for k in range(hs):
                j=min(i0+k,n-1);v=df["obs_v"].values[j]; yaw+=g*df["steer"].values[j]*DT_C
                x+=v*np.cos(yaw)*DT_C;y+=v*np.sin(yaw)*DT_C
            gx=df["x"].values[i0+hs];gy=df["y"].values[i0+hs];ex,ey=x-gx,y-gy
            fdes.append(np.hypot(ex,ey)); cc,ss=np.cos(yaw0),np.sin(yaw0)
            lon.append(abs(cc*ex+ss*ey));lat.append(abs(-ss*ex+cc*ey))
    phys=float(np.mean(fdes))
    def build(segs_):
        X,Y=[],[]
        for df in segs_:
            n=len(df)
            for i0 in range(0,n-hs,5):
                fut=slice(i0,i0+hs)
                feat=[df["obs_v"].values[i0],np.radians(df["yaw_rate"].values[i0]),np.mean(df["obs_v"].values[fut]),
                      np.mean(df["steer"].values[fut]),np.std(df["steer"].values[fut])]
                yaw=np.radians(df["yaw_unwrap"].values[i0]);dx=df["x"].values[i0+hs]-df["x"].values[i0];dy=df["y"].values[i0+hs]-df["y"].values[i0]
                cc,ss=np.cos(-yaw),np.sin(-yaw);X.append(feat);Y.append([cc*dx-ss*dy,ss*dx+cc*dy])
        return np.array(X),np.array(Y)
    curve=[]
    for k in [1,2,4,8,16,32,min(len(pool),50)]:
        if k>len(pool): break
        Xtr,Ytr=build(pool[:k]);sc=StandardScaler().fit(Xtr)
        mdl=MLPRegressor(hidden_layer_sizes=(48,24),max_iter=1000,alpha=1e-3,random_state=0,early_stopping=True).fit(sc.transform(Xtr),Ytr)
        fd=[]
        for df in test_segs:
            n=len(df)
            for i0 in range(0,n-hs,5):
                fut=slice(i0,i0+hs)
                feat=np.array([[df["obs_v"].values[i0],np.radians(df["yaw_rate"].values[i0]),np.mean(df["obs_v"].values[fut]),
                      np.mean(df["steer"].values[fut]),np.std(df["steer"].values[fut])]])
                lx,ly=mdl.predict(sc.transform(feat))[0];yaw=np.radians(df["yaw_unwrap"].values[i0]);cc,ss=np.cos(yaw),np.sin(yaw)
                ddx=cc*lx-ss*ly;ddy=ss*lx+cc*ly
                fd.append(np.hypot((df["x"].values[i0]+ddx)-df["x"].values[i0+hs],(df["y"].values[i0]+ddy)-df["y"].values[i0+hs]))
        curve.append((k*60,float(np.mean(fd))))
    return phys,curve,{"lon":float(np.mean(lon)),"lat":float(np.mean(lat))}


def main():
    print("="*70); print("  实验2: 轨迹预测 + 数据稀缺交叉点 + 误差各向异性"); print("="*70)
    loader_data = load_loader()

    print("\n[A] 装载机 LOSO 轨迹预测...")
    fde, aniso_l = loader_loso()
    for m in ["CV","Physics","DataDriven"]:
        print(f"  {m:<12} FDE: " + " ".join(f"{h}s={fde[m][h]:.2f}" for h in HORIZONS))

    print("\n[B] 数据稀缺交叉点...")
    phys_l, curve_l = loader_crossover(loader_data)
    print(f"  装载机物理FDE@3s={phys_l:.2f}m; 数据驱动曲线: "+", ".join(f"{d:.0f}s:{f:.2f}m" for d,f in curve_l))
    phys_c, curve_c, aniso_c = comma_crossover_aniso()
    print(f"  comma物理FDE@3s={phys_c:.2f}m; 数据驱动曲线: "+", ".join(f"{d:.0f}s:{f:.2f}m" for d,f in curve_c))

    print("\n[C] 误差各向异性:")
    print(f"  装载机: 纵向(速度)={aniso_l['lon']:.2f}m 横向(转向)={aniso_l['lat']:.2f}m -> {'纵向' if aniso_l['lon']>aniso_l['lat'] else '横向'}主导")
    print(f"  comma : 纵向(速度)={aniso_c['lon']:.2f}m 横向(转向)={aniso_c['lat']:.2f}m -> {'纵向' if aniso_c['lon']>aniso_c['lat'] else '横向'}主导")

    # ---- Fig2: LOSO FDE ----
    fig, ax = plt.subplots(figsize=(7,5))
    for m,mk in [("CV","o--"),("Physics","s-"),("DataDriven","^:")]:
        ax.plot(HORIZONS,[fde[m][h] for h in HORIZONS],mk,label=m,lw=2)
    ax.set_xlabel("Prediction horizon (s)"); ax.set_ylabel("FDE (m)")
    ax.set_title("Loader LOSO: Physics prior beats data-driven under scarce data"); ax.legend()
    save(fig,"fig2_prediction_fde.png")

    # ---- Fig3: crossover (two panels) ----
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    d,f=zip(*curve_l); axes[0].plot(d,f,"^-",label="Data-driven MLP",lw=2)
    axes[0].axhline(phys_l,color="r",ls="--",label=f"Physics prior ({phys_l:.2f}m)")
    axes[0].set_xscale("log"); axes[0].set_xlabel("Training data (s, log)"); axes[0].set_ylabel("FDE@3s (m)")
    axes[0].set_title("Articulated Loader"); axes[0].legend()
    d,f=zip(*curve_c); axes[1].plot(d,f,"^-",color="tab:green",label="Data-driven MLP",lw=2)
    axes[1].axhline(phys_c,color="r",ls="--",label=f"Physics prior ({phys_c:.2f}m)")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("Training data (s, log)"); axes[1].set_ylabel("FDE@3s (m, log)")
    axes[1].set_title("comma2k19 Passenger Car"); axes[1].legend()
    fig.suptitle("Innovation 3: Data-scarcity crossover is universal across vehicle classes")
    save(fig,"fig3_crossover.png")

    # ---- Fig4: anisotropy ----
    fig, ax = plt.subplots(figsize=(7,5))
    labels=["Loader\n(no reliable speed)","comma2k19\n(true CAN speed)"]
    lon=[aniso_l["lon"],aniso_c["lon"]]; lat=[aniso_l["lat"],aniso_c["lat"]]
    x=np.arange(2);w=0.35
    ax.bar(x-w/2,lon,w,label="Longitudinal error (speed-driven)")
    ax.bar(x+w/2,lat,w,label="Lateral error (steering-driven)")
    ax.set_xticks(x);ax.set_xticklabels(labels); ax.set_ylabel("Error @3s (m)")
    ax.set_title("Innovation 2: Dominant error axis is set by which signal is observable")
    ax.legend()
    save(fig,"fig4_error_anisotropy.png")

    results={"loader_loso_fde":fde,"loader_anisotropy":aniso_l,
             "loader_crossover":{"physics":phys_l,"curve":curve_l},
             "comma_crossover":{"physics":phys_c,"curve":curve_c},"comma_anisotropy":aniso_c}
    with open(os.path.join(RES_DIR,"exp2_results.json"),"w") as fp: json.dump(results,fp,indent=2)
    print("\n结果已存 results/exp2_results.json\n实验2完成。")


if __name__ == "__main__":
    main()
