"""Audit the available-data selection funnel without inventing unavailable raw counts.

The public release contains post-filter caches, not the original raw segment
manifest.  This script reports every check that can be reconstructed from the
cache and explicitly records when a pre-cache candidate count is unavailable.
"""
from pathlib import Path
import datetime as dt, json, sys
import numpy as np
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
OUT=ROOT/"results"/"supplementary"/"exp_sample_funnel_audit_results.json"
REPORT=ROOT/"results"/"reports_zh"/"EXP_SAMPLE_FUNNEL_AUDIT_zh.md"
sys.path.insert(0,str(HERE)); import comma_loader as C; import exp_causal_nuscenes as N

def load_nusc_cache():
    import pickle
    candidates = [HERE / ".nusc_cache.pkl", HERE.parent.parent.parent / "数据分析" / "paper_experiments" / ".nusc_cache.pkl"]
    for p in candidates:
        if p.exists():
            return pickle.loads(p.read_bytes())
    raise FileNotFoundError("nuScenes cache not found; prepare the canonical local cache before running")

def summarize(name,data):
    rows=[]
    for key,df in data.items():
        rows.append({"key":str(key),"n_rows":int(len(df)),"duration_s":float(df["t"].iloc[-1]-df["t"].iloc[0]),
                     "finite_required":bool(np.isfinite(df[["x","y","steer","can_speed"]].to_numpy(float)).all()),
                     "has_future_3s":bool(len(df)>round(3.0/(0.05 if name=='comma2k19' else 0.02)+10)),
                     "mean_speed":float(np.nanmean(df["obs_v"])),"steer_std":float(np.nanstd(df["steer"]))})
    return rows

def main():
    comma=C.load_all_cached(verbose=False); nusc=load_nusc_cache()
    nusc2={N.key_of(df,k):df for k,df in nusc.items()}
    out={"timestamp":dt.datetime.now().isoformat(timespec="seconds"),"status":"CACHE_LEVEL_AUDIT",
         "scope_note":"Original raw candidate counts are not reconstructible from post-filter caches; no candidate total is inferred.","datasets":{}}
    for name,data,group in [("comma2k19",comma,"route"),("nuScenes",nusc2,"scene")]:
        rows=summarize(name,data)
        checks={"cached_retained":len(rows),"finite_required":sum(r["finite_required"] for r in rows),"has_future_3s":sum(r["has_future_3s"] for r in rows),"grouping_unit":group}
        out["datasets"][name]={"candidate_count":"unavailable_from_cache","retained_count":len(rows),"checks":checks,"rows":rows}
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# 可用数据筛选漏斗审计", "", "该审计只使用当前公开实验缓存。缓存已经是统一预处理后的输入，因此原始下载目录中的候选总数和逐步排除数不能从缓存反推；报告不填入未经观测的候选数量。", "", "| 数据集 | 原始候选数 | 缓存保留数 | 必需字段有限值 | 具备 3 s 未来标签 | 统计单位 |", "|---|---:|---:|---:|---:|---|"]
    for name,d in out["datasets"].items():
        c=d["checks"]; lines.append(f"| {name} | 不可由缓存恢复 | {d['retained_count']} | {c['finite_required']} | {c['has_future_3s']} | {c['grouping_unit']} |")
    lines += ["", "## 解释边界", "", "该结果能证明当前 canonical 输入的基本字段、长度和分组状态，但不能替代从原始下载目录重新执行预处理时的完整排除漏斗。若需要报告候选总数和每步排除原因，必须在拥有原始数据目录的环境中运行同一脚本的 raw-enumeration 模式。"]
    REPORT.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(OUT); print(REPORT)
if __name__=='__main__': main()
