# 第六轮实验补充与中文稿更新记录

## 已完成

1. **comma2k19 路线级解析式 lambda 审计**：42 条路线内先聚合重复抽样，再计算路线级 bootstrap 区间。N=3、5、10、20 的收益为 0.1616、0.0569、0.0040、−0.0012 m，四个区间均包含 0。
2. **校准差距审计**：复用已有预测结果，比较验证集收缩与测试集 grid oracle 的差距。oracle 只作描述性上界；nuScenes 的 gap-closed 点估计为 0.981、0.983、0.994、0.885。
3. **RLS/EKF 配对审计**：使用同一外层折内的 route/scene 配对结果。comma 三折区间均包含 0；nuScenes 三折 RLS−EKF 差异均为负。该结果只用于两个辅助基线之间的审计。
4. **中文稿同步**：在摘要、实验设置、结果、局限性和结论中区分 comma 路线级点估计与区间、nuScenes 场景级证据以及 oracle 描述性上界，并补充 RLS/EKF 的范围边界。

## 论文可支持的结论

- nuScenes 提供主要的场景级确认性证据：在 N=3、5、10、20，收缩相对满残差的收益为 1.663、0.960、0.420、0.137 m，区间均不包含 0。
- comma2k19 的路线级结果只支持“低预算点估计收益较大”的描述性观察，不支持路线级确认性显著性结论。
- 闭式系数和经验收缩构成可复核的受限 CAN 部署校准流程，但不构成新的统计理论；系数仍需按目标平台验证数据校准。
- RLS 和 steering-input EKF 已作为信息集匹配的补充基线完成，但不足以支持“优于所有递推或滤波方法”。

## 不支持的结论

不支持严格 lambda 单调、严格 1/N、普适 schedule、跨车辆参数直接迁移、heuristic gate 最优、oracle 可部署、嵌入式实时已验证或 CRIStAL/LOCSP 跨车型泛化。

## 文件对应关系

- exp_lambda_analytic_route_ci_results.json / exp_lambda_analytic_route_ci_results_zh.md
- exp_calibration_regret_audit_results.json / exp_calibration_regret_audit_results_zh.md
- exp_recursive_baselines_causal_results.json / exp_recursive_baselines_causal_results_zh.md
- exp_recursive_baselines_paired_audit_results.json / exp_recursive_baselines_paired_audit_results_zh.md
- 中文稿：论文投稿目录中的最终中文稿（不随代码仓库发布）。
