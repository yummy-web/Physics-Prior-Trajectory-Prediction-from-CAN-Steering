# MLP 收敛敏感性检查：复现诊断说明

**状态：已完成；本文件是诊断记录，不替换 canonical 结果，也不修改论文或 PROVENANCE.md。**

## 1. 诊断对象

收敛敏感性实验位于：

- `src/exp_lambda_group_calibration_convergence.py`
- `results/canonical/exp_lambda_group_calibration_convergence_results.json`
- `results/reports_zh/EXP_LAMBDA_GROUP_CALIBRATION_CONVERGENCE_zh.md`

实验重放 comma 和 nuScenes 的 canonical 分组清单，分别比较 canonical `max_iter`（comma=250、nuScenes=300）与 higher `max_iter=1000`。每个数据集、每个训练预算 `N∈{3,5,10,20}` 使用 3 个 outer folds × 4 个 subset seeds，共 12 个 draw；最终按 route（comma）或 scene（nuScenes）汇总，而不把滑动窗口作为独立统计单位。

## 2. 结果性结论

higher `max_iter=1000` 后，收缩增益 `FDE(full residual)−FDE(shrunk residual)` 的方向在两个平台和所有预算上均保持：

| 数据集 | N=3 | N=5 | N=10 | N=20 |
|---|---:|---:|---:|---:|
| comma，canonical | 0.2090 | 0.0615 | 0.0162 | −0.0069 |
| comma，higher | 0.2097 | 0.0615 | 0.0162 | −0.0047 |
| nuScenes，canonical | 1.9591 | 0.9347 | 0.4291 | 0.1258 |
| nuScenes，higher | 2.0877 | 0.9347 | 0.4340 | 0.1258 |

据此，检查支持以下有限结论：

1. 收缩收益的方向和主要随预算下降的趋势，对 MLP `max_iter` 的这一改变具有一定稳健性。
2. nuScenes 的收缩收益在 N=3、5、10、20 均保持为正；comma 的优势主要集中在低预算，N=20 不支持稳定优于 full residual。
3. higher 条件消除了本次重放中所有记录的 `ConvergenceWarning`/`hit_max_iter`，但这只能说明训练停止诊断改善，不能证明模型已经达到全局最优。

## 3. canonical 数值复现诊断

`canonical_reproduction_check.all_pass=false`，因此不能声称当前环境逐数值复现了历史 canonical MLP 结果。

已对代表性 draw 使用默认线程设置和显式单线程设置进行对照：

```text
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

默认线程与单线程重放的 MLP 权重、迭代次数、分组 FDE 和 λ 结果一致（仅有浮点舍入级差异）。physics-only/prior 结果也基本精确一致；差异主要集中在 MLP 相关结果。因此目前不能把 canonical 差异归因于线程数，也不能在没有历史环境记录的情况下确定唯一原因。

合理的剩余解释包括：历史 sklearn/BLAS/OpenMP 版本差异、未记录的代码或依赖参数差异，以及带 early stopping 的 MLP 优化路径对数值环境的敏感性。

## 4. 可写入实验记录的边界

可以写：

> 在保持数据划分、因果特征、物理先验、网络结构和早停设置不变的条件下，将 MLP 的最大迭代次数提高到 1000 后，两个数据集上的收缩收益方向保持，说明该现象对本次停止条件变化具有一定稳健性。

不可以写：

- “MLP 已完全收敛”；
- “canonical 结果已完全复现”；
- “不同停止条件下所有绝对 FDE 完全一致”；
- “收缩收益在所有预算和所有平台都显著”；
- 用 higher 结果替换 canonical 主结果。

## 5. 后续处理

- 将 higher 条件作为收敛敏感性/稳健性分析，不作为主结果替代。
- 保留 canonical JSON 和 higher 结果 JSON 的独立记录。
- 在论文重构阶段只引用上述有限结论，并披露 MLP canonical 逐数值复现未通过。
- 本诊断不改变 `PROVENANCE.md`，待所有实验冻结后统一决定是否补充 provenance。
