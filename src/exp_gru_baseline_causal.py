"""Compact causal GRU baseline under the frozen grouped protocol.

This is a supplementary baseline, not a replacement for the canonical MLP
results.  It uses the same causal signal family and the exact split manifests
from the grouped calibration result.  Each outer draw trains a direct GRU and
a physics-residual GRU on fit groups only; test groups are untouched.  Results
are collapsed by route/scene before group-level bootstrap inference.
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, sys, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as e:
    raise SystemExit("PyTorch is required for this supplementary experiment") from e

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
CANONICAL = RESULTS / "canonical" / "exp_lambda_group_calibration_results.json"
OUT = RESULTS / "supplementary" / "exp_gru_baseline_causal_results.json"
REPORT = RESULTS / "reports_zh" / "EXP_GRU_BASELINE_CAUSAL_zh.md"

sys.path.insert(0, str(HERE))
import comma_loader as C
import causal_eval as E
import causal_protocol as P
import exp_causal_nuscenes as NU
import exp_lambda_nusc_causal as NC

HORIZON = 3.0
HIST = 10
EPOCHS = 80
PATIENCE = 12
BATCH = 256
LR = 1e-3

class GRURegressor(nn.Module):
    def __init__(self, in_dim=4, hidden=32):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        z, _ = self.gru(x)
        return self.head(z[:, -1, :])

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def load_nusc_cache():
    import pickle
    candidates = [HERE / ".nusc_cache.pkl", HERE.parent.parent.parent / "数据分析" / "paper_experiments" / ".nusc_cache.pkl"]
    for p in candidates:
        if p.exists(): return pickle.loads(p.read_bytes())
    raise FileNotFoundError("nuScenes cache not found")

def keys_for(group_map, groups):
    return [k for g in groups for k in group_map[str(g)]]

def state_features(df, dataset):
    # Both datasets use the same four-channel, history-only representation.
    # The second speed channel is recomputed from positions with a backward
    # difference; never use the loader's central-difference obs_v here.
    s = E.causal_state(df)
    st = df["steer"].to_numpy(float)
    return s["v_can"], s["v_bwd"], s["wr_bwd"], st

def arrays(dataset, data, keys, gain):
    dt_s = 0.05 if dataset == "comma" else 0.02
    hs = round(HORIZON / dt_s)
    stride = 15 if dataset == "comma" else 5
    X, Y, PH, GRP = [], [], [], []
    for key in keys:
        df = data[key]
        vcan, vbwd, wr, st = state_features(df, dataset)
        if dataset == "comma":
            anchors = E.anchors_for(df, hs, stride)
        else:
            anchors = range(HIST, len(df) - hs, stride)
        for i0 in anchors:
            hist = np.column_stack([vcan[i0-HIST+1:i0+1], vbwd[i0-HIST+1:i0+1], wr[i0-HIST+1:i0+1], st[i0-HIST+1:i0+1]])
            if len(hist) != HIST or not np.isfinite(hist).all(): continue
            if dataset == "comma":
                ph = E.physics_endpoint_local(df, i0, hs, dt_s, gain)
                gt = E.target_local(df, i0, hs)
                group = E.route_of(df["seg"].iloc[0])
            else:
                # collect() is the canonical causal physics/target implementation;
                # obtain one anchor through the shared implementation below.
                one = NC.collect([(key, df)], gain)
                # Rebuilding one complete segment per anchor is costly; use the
                # corresponding row order from collect(), which has the same stride.
                # This branch is replaced by direct local construction below.
                raise RuntimeError("internal nusc path should use nusc_arrays")
            X.append(hist); Y.append(gt); PH.append(ph); GRP.append(str(group))
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), np.asarray(PH, np.float32), np.asarray(GRP, str)

def nusc_arrays(keys, data, gain):
    X, Y, PH, GRP = [], [], [], []
    for key in keys:
        df = data[key]; x,y,yaw,v,wr,st = NC.causal_state(df); causal = E.causal_state(df); vbwd = causal["v_bwd"]; n=len(df); hs=round(HORIZON/0.02)
        for i0 in range(HIST, n-hs, 5):
            h=slice(i0-HIST+1,i0+1)
            hist=np.column_stack([v[h], vbwd[h], wr[h], st[h]])
            if len(hist)!=HIST or not np.isfinite(hist).all(): continue
            px,py,th=x[i0],y[i0],yaw[i0]
            for _ in range(hs):
                th += gain*st[i0]*v[i0]*0.02; px += v[i0]*np.cos(th)*0.02; py += v[i0]*np.sin(th)*0.02
            cc,ss=np.cos(-yaw[i0]),np.sin(-yaw[i0])
            PH.append([cc*(px-x[i0])-ss*(py-y[i0]), ss*(px-x[i0])+cc*(py-y[i0])])
            dx=x[i0+hs]-x[i0]; dy=y[i0+hs]-y[i0]
            Y.append([cc*dx-ss*dy, ss*dx+cc*dy]); X.append(hist); GRP.append(str(key))
    return np.asarray(X,np.float32),np.asarray(Y,np.float32),np.asarray(PH,np.float32),np.asarray(GRP,str)

def fit_predict(Xtr, Ttr, Xte, seed, epochs=EPOCHS):
    torch.manual_seed(seed); np.random.seed(seed)
    mu=Xtr.reshape(-1,Xtr.shape[-1]).mean(0); sd=Xtr.reshape(-1,Xtr.shape[-1]).std(0)+1e-6
    Xt=(Xtr-mu)/sd; Xe=(Xte-mu)/sd
    model=GRURegressor(Xtr.shape[-1]); opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=1e-4); lossf=nn.MSELoss()
    n=len(Xt); rng=np.random.default_rng(seed); order=rng.permutation(n); nv=max(1,int(0.2*n)); vi=order[:nv]; ti=order[nv:]
    best=float("inf"); best_state=None; bad=0
    Xt_t=torch.tensor(Xt); Tt=torch.tensor(Ttr); Vi=torch.tensor(Xt[vi]); Tv=torch.tensor(Ttr[vi])
    for _ in range(epochs):
        model.train(); perm=rng.permutation(ti)
        for j in range(0,len(perm),BATCH):
            ix=perm[j:j+BATCH]; opt.zero_grad(); loss=lossf(model(Xt_t[ix]),Tt[ix]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): val=float(lossf(model(Vi),Tv).item())
        if val < best-1e-6:
            best=val; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>PATIENCE: break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(): pred=model(torch.tensor(Xe)).numpy()
    return pred, {"epochs_run": _+1, "best_val_mse": best, "n_train": len(ti), "n_val": len(vi)}

def group_mean(pred, target, groups):
    e=np.linalg.norm(pred-target,axis=1); return {str(g):float(e[groups==g].mean()) for g in np.unique(groups)}

def ci(ref, cand, seed):
    ids=sorted(set(ref)&set(cand)); d=np.asarray([ref[g]-cand[g] for g in ids]);
    rng=np.random.default_rng(seed); boot=d[rng.integers(0,len(d),(5000,len(d)))].mean(1)
    return {"n_groups":len(d),"mean_gain_m":float(d.mean()),"CI95_m":[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))],"fraction_cand_better":float(np.mean(d>0))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--datasets",default="comma,nuscenes"); args=ap.parse_args()
    can=json.loads(CANONICAL.read_text(encoding="utf-8")); out={"title":"Causal GRU baseline","status":"FULL_RUN","configuration":{"epochs":EPOCHS,"patience":PATIENCE,"hidden":32,"input_features":["causal_speed","causal_pose_speed","backward_yaw_rate","steer"],"split_source":str(CANONICAL.relative_to(ROOT)).replace('\\\\','/')},"per_dataset":{},"provenance":{"script":str(Path(__file__).relative_to(ROOT)).replace('\\\\','/'),"script_sha256":sha256(Path(__file__).resolve()),"stat_unit":"route for comma; scene for nuScenes","test_not_used_for_training":True}}
    if OUT.exists():
        try:
            old=json.loads(OUT.read_text(encoding="utf-8"))
            out["per_dataset"].update(old.get("per_dataset",{}))
        except Exception:
            pass
    for ds in [x.strip() for x in args.datasets.split(',') if x.strip()]:
        if ds=="comma": data=C.load_all_cached(verbose=False); gm=P.group_keys(data,"comma")
        else:
            raw=load_nusc_cache(); data={NU.key_of(df,k):df for k,df in raw.items()}; gm={k:[k] for k in data}
        rows={}
        for ns, nd in sorted(can["per_dataset"][ds]["per_N"].items(),key=lambda kv:int(kv[0])):
            n=int(ns); stores=defaultdict(list); diagnostics=[]
            # The canonical manifests repeat the same outer test group across
            # subset seeds; run all manifests, then collapse by independent
            # route/scene below.
            for di,man in enumerate(nd["split_manifests"]):
                fit=keys_for(gm,man["fit_groups"]); test=keys_for(gm,man["test_groups"])
                fit_items=[(k,data[k]) for k in fit]; gain=(E.gain_curv(fit_items,True) if ds=="comma" else NC.gain_causal(fit_items))
                if ds=="comma": Xtr,Ytr,PHtr,Gtr=arrays(ds,data,fit,gain); Xte,Yte,PHte,Gte=arrays(ds,data,test,gain)
                else: Xtr,Ytr,PHtr,Gtr=nusc_arrays(fit,data,gain); Xte,Yte,PHte,Gte=nusc_arrays(test,data,gain)
                if len(Xtr)<20 or len(Xte)<20: continue
                pdirect,diag=fit_predict(Xtr,Ytr,Xte,seed=1000+di)
                pres,diag2=fit_predict(Xtr,Ytr-PHtr,Xte,seed=2000+di)
                stores["direct_gru"].append(group_mean(pdirect,Yte,Gte)); stores["residual_gru"].append(group_mean(PHte+pres,Yte,Gte)); stores["physics"].append(group_mean(PHte,Yte,Gte)); diagnostics.append({"draw":di,"gain":float(gain),"fit_anchors":len(Xtr),"test_anchors":len(Xte),"direct":diag,"residual":diag2})
            collapsed={m:{g:float(np.mean([r[g] for r in rs if g in r])) for g in set().union(*rs)} for m,rs in stores.items()}
            row={"N":n,"n_draws":len(diagnostics),"n_groups":len(collapsed.get("physics",{})),"mean_fde_m":{m:float(np.mean(list(v.values()))) for m,v in collapsed.items()},"comparisons":{"residual_gru_vs_direct_gru":ci(collapsed["direct_gru"],collapsed["residual_gru"],300+n),"residual_gru_vs_physics":ci(collapsed["physics"],collapsed["residual_gru"],400+n)},"diagnostics":diagnostics}
            rows[ns]=row; print(ds,ns,row["mean_fde_m"],flush=True)
        out["per_dataset"][ds]={"grouping_unit":"route" if ds=="comma" else "scene","per_N":rows}
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# 因果 GRU 同信息集补充基线", "", "GRU 使用与主实验相同的 CAN/运动信号族、历史窗口和外层分组清单；训练仅使用 fit groups，测试组在训练和选择中不使用。该结果是补充基线，不替换 canonical MLP 结果。", "", "| 数据集 | N | Physics | Direct GRU | Residual GRU | Residual GRU 相对 Direct GRU |", "|---|---:|---:|---:|---:|---:|"]
    for ds,block in out["per_dataset"].items():
        for n,row in block["per_N"].items():
            c=row["comparisons"]["residual_gru_vs_direct_gru"]; m=row["mean_fde_m"]; lines.append(f"| {ds} | {n} | {m['physics']:.4f} | {m['direct_gru']:.4f} | {m['residual_gru']:.4f} | {c['mean_gain_m']:.4f} [{c['CI95_m'][0]:.4f},{c['CI95_m'][1]:.4f}] |")
    lines += ["", "## 解释边界", "", "组级区间在折叠测试组内计算；GRU 的序列特征仍是回放数据中的因果化输入，不等同于真实 IMU-only 传感器验证。由于 GRU 与 MLP 的优化器和特征参数化不同，结果用于同信息信号族的补充审计，不宣称模型族全面优越。"]
    REPORT.write_text("\n".join(lines)+"\n",encoding="utf-8"); print(OUT); print(REPORT)
if __name__=="__main__": main()
