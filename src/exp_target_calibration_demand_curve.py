"""Target-platform calibration demand curve for the frozen causal protocol.

For each canonical nested split, the residual model and physics gain are fitted
only on fit groups.  A target-platform calibration coefficient is then computed
from the first K validation groups, without reading test targets.  K=0 is the
comma-derived source schedule; K>0 uses the causal analytic estimator with the
declared anchor-count shrinkage.  Results are descriptive supplementary
evidence: repeated draws reuse outer test scenes and are collapsed by scene
before bootstrap inference.
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
CANONICAL = RESULTS / "canonical" / "exp_lambda_group_calibration_results.json"
OUT = RESULTS / "supplementary" / "exp_target_calibration_demand_curve_results.json"
REPORT = RESULTS / "reports_zh" / "EXP_TARGET_CALIBRATION_DEMAND_CURVE_zh.md"
SOURCE_SCHEDULE = {3: 0.35, 5: 0.52, 10: 0.72, 20: 0.89}
KAPPA = 300
BOOT = 5000

sys.path.insert(0, str(HERE))
import exp_causal_nuscenes as NU
from exp_lambda_nusc_causal import collect, gain_causal, route_mean

def load_nusc_cache():
    import pickle
    candidates = [HERE / ".nusc_cache.pkl", HERE.parent.parent.parent / "数据分析" / "paper_experiments" / ".nusc_cache.pkl"]
    for p in candidates:
        if p.exists():
            return pickle.loads(p.read_bytes())
    raise FileNotFoundError("nuScenes cache not found; prepare the canonical local cache before running")

def sha256(p: Path) -> str:
    h = hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def ci(a, b, seed):
    ids = sorted(set(a) & set(b)); d = np.asarray([a[k] - b[k] for k in ids], float)
    if not len(d): return {"n_scenes": 0}
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(d), (BOOT, len(d)))
    boot = d[idx].mean(axis=1)
    return {"n_scenes": len(d), "mean_diff_m": float(d.mean()),
            "CI95_m": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "fraction_target_better": float(np.mean(d > 0))}

def analytic(r, e):
    raw = float(np.sum(r * e) / (np.sum(r * r) + 1e-9))
    clipped = float(np.clip(raw, 0.0, 1.5))
    return raw, clipped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
    data = load_nusc_cache(); by = {NU.key_of(df, k): df for k, df in data.items()}
    out = {"title": "Target-platform calibration demand curve",
           "status": "FULL_RUN", "dataset": "nuScenes",
           "provenance": {"script": str(Path(__file__).relative_to(ROOT)).replace('\\\\','/'),
                          "script_sha256": sha256(Path(__file__).resolve()),
                          "canonical_split_source": str(CANONICAL.relative_to(ROOT)).replace('\\\\','/'),
                          "causal": True, "calibration_unit": "target validation scenes",
                          "kappa": KAPPA, "kappa_unit": "raw overlapping validation anchor count",
                          "test_not_used_for_selection": True,
                          "interpretation": "supplementary demand curve; not an online update policy"},
           "K_list": [0, 1, 2, 4, 7],
           "scope_note": "K is the number of validation scenes available within each frozen canonical N split; unavailable K values are not imputed. For K>1, the first K validation scenes in the frozen manifest order are used; this is a controlled sensitivity path, not a random-design sample-size law.",
           "per_N": {}}
    for n_s, nd in sorted(canonical["per_dataset"]["nuscenes"]["per_N"].items(), key=lambda kv:int(kv[0])):
        n = int(n_s); recs = []; scene_store = defaultdict(lambda: defaultdict(list))
        for di, man in enumerate(nd["split_manifests"]):
            fit = [str(x) for x in man["fit_groups"]]; val = [str(x) for x in man["validation_groups"]]; test = [str(x) for x in man["test_groups"]]
            fit_items = [(k, by[k]) for k in fit]; val_items = [(k, by[k]) for k in val]; test_items = [(k, by[k]) for k in test]
            gain = gain_causal(fit_items)
            Ff, PHf, GTf, _ = collect(fit_items, gain); Fv, PHv, GTv, GRPv = collect(val_items, gain); Ft, PHt, GTt, GRPt = collect(test_items, gain)
            sc = StandardScaler().fit(Ff)
            mdl = MLPRegressor(hidden_layer_sizes=(48,24), max_iter=300, alpha=1e-3, random_state=0, early_stopping=True)
            mdl.fit(sc.transform(Ff), GTf-PHf); rv = mdl.predict(sc.transform(Fv)); rt = mdl.predict(sc.transform(Ft))
            policies = {"source_schedule": SOURCE_SCHEDULE[n]}
            for k in [1,2,4,7]:
                if k > len(val):
                    continue
                use = val[:k]
                ix = np.isin(GRPv, np.asarray(use, dtype=str)); raw, clipped = analytic(rv[ix], (GTv-PHv)[ix])
                nav = int(np.sum(ix)); policies[f"target_calibration_K{k}"] = float(nav/(nav+KAPPA)*clipped)
            for name, lam in policies.items():
                fde = np.hypot((PHt+lam*rt)[:,0]-GTt[:,0], (PHt+lam*rt)[:,1]-GTt[:,1]); means, groups = route_mean(fde, GRPt)
                for g, v in zip(groups, means): scene_store[name][str(g)].append(float(v))
            recs.append({"draw": di, "fit_groups": fit, "validation_groups": val, "test_groups": test,
                         "validation_anchor_count": int(len(Fv)), "policy_lambdas": policies})
        collapsed = {name:{g:float(np.mean(v)) for g,v in gm.items()} for name,gm in scene_store.items()}
        ref = collapsed["source_schedule"]
        row = {"N":n, "n_draws":len(recs), "n_scenes":len(ref), "policies":{}, "draw_records":recs,
               "available_validation_group_count_by_draw": sorted(set(len(r["validation_groups"]) for r in recs))}
        for name, vals in collapsed.items():
            row["policies"][name] = {"mean_fde_m":float(np.mean(list(vals.values()))),
                                      "vs_source_CI95": None if name=="source_schedule" else ci(ref, vals, 5000+n*10+len(name))}
        out["per_N"][str(n)] = row
        print(f"N={n} source={row['policies']['source_schedule']['mean_fde_m']:.3f} " + " ".join(f"{k}={v['mean_fde_m']:.3f}" for k,v in row['policies'].items() if k!='source_schedule'), flush=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    lines=["# 目标平台校准数据需求曲线", "", "该实验重放冻结的 nuScenes 嵌套分组清单。K=0 使用 comma 源 schedule；K>0 只用目标平台验证场景估计收缩系数，测试场景不参与选择。K 表示当前冻结 N 划分中实际可用的验证场景数；不可用的 K 不进行插值或截断。K>1 沿冻结清单顺序取前 K 个验证场景，是控制变量敏感性路径，不是普适样本量定律。", "", "注意：该实验重放的是 exp_lambda_group_calibration_results.json 的冻结清单；它与 exp_lambda_transfer.py 的历史迁移清单不是同一结果快照，因此 K=0 的绝对 FDE 不应直接与跨平台迁移表中的 source FDE 逐项混用。", "", "| N | K=0 源 schedule | K=1 | K=2 | K=4 | K=7 |", "|---:|---:|---:|---:|---:|---:|"]
    for n,row in out["per_N"].items():
        vals=[]
        for k in ["source_schedule","target_calibration_K1","target_calibration_K2","target_calibration_K4","target_calibration_K7"]:
            vals.append(row["policies"].get(k,{}).get("mean_fde_m"))
        lines.append("| %s | %s |"%(n," | ".join("—" if v is None else f"{v:.4f}" for v in vals)))
    lines += ["", "## 统计与解释边界", "", "所有 FDE 为场景内锚点均值再按场景聚合；同一外层测试场景被多个 subset seed 重复评估后先平均。区间是场景级配对 bootstrap。该实验回答目标平台需要多少留出场景进行离线标定，不表示在线切换或普适样本阈值。"]
    REPORT.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"wrote {OUT}\nwrote {REPORT}")

if __name__ == "__main__": main()
