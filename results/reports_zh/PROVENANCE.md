# 公开复现实验 Provenance

## 仓库范围

本文件描述当前公开仓库中可引用的实验代码和结果快照。原始 comma2k19、nuScenes CAN bus expansion、CRIStAL/LOCSP 2025 和私有装载机数据均不随仓库发布。

## 方法

预测器写为 `y_hat = p + lambda*r`。`p` 是控制条件物理先验，`r` 是残差 MLP。残差网络在拟合分区训练，lambda 在验证分区由闭式最小二乘对齐量估计，再使用 `n/(n+kappa)` 进行经验收缩并在外层测试前固定。实验协议使用后向差分航向率、锚点 CAN 速度和转向零阶保持，不读取预测时刻之后的控制或位姿样本。该约束是时间信息约束，不等同于真实 IMU-only 传感器验证。

## 统计单位与主结果

- nuScenes lambda 实验：566 个 scene 聚类；3 s 收缩相对满残差的改善量为 N=3/5/10/20 时 1.663/0.960/0.420/0.137 m，scene-level 95% CI 分别为 [1.461,1.877]、[0.860,1.069]、[0.380,0.464]、[0.114,0.163]。
- comma2k19：42 条 route；统一分组校准实验的路线级收益为 N=3/5/10/20 时 0.2786/0.0835/0.0024/−0.0008 m，95% CI 分别为 [0.1003,0.4712]、[0.0120,0.1652]、[−0.0245,0.0328]、[−0.0119,0.0128]。点估计与区间来自同一批预测。
- 跨平台迁移：支持收缩方向可复用，但具体 lambda 数值仍需目标平台验证校准。
- CRIStAL/LOCSP：7 次同一辆 Renault Zoe、同日采集；统计单位为 acquisition，3 s FDE 的 INS velocity ZOH、拟合 CTRV、满残差和 kappa=0.5 分别为 2.655、2.438、2.425 和 2.421 m。该结果是同车多采集外部检查，不是跨车型泛化。

## 后续补充审计（不替换 canonical 主结果）

- supplementary/exp_target_calibration_demand_curve_results.json：目标平台验证场景需求曲线。K=0 使用 comma 源 schedule；K>0 使用当前冻结划分中实际可用的目标验证场景。N=3/5/10/20 的验证组数分别为 1/2/4/7，因此不可把缺失 K 值解释为零样本或插值结果。
- supplementary/exp_lambda_transfer_demand_unified_results.json：统一跨平台迁移与目标校准实验已完成，N=3/5/10/20 均为 12 个 draw、566 个测试场景，并通过独立结构审计与 deterministic canonical replay；审计文件为 supplementary/exp_lambda_transfer_demand_unified_audit.json。该结果可作为补充证据，但不支持普适样本复杂度定律。
- supplementary/exp_gru_baseline_causal_results.json：同一因果信号族下的紧凑 GRU 补充基线。它不是 canonical MLP 的替代，也不构成对所有时序网络的比较。
- supplementary/exp_sample_funnel_audit_results.json：缓存级筛选审计。原始下载候选总数不在缓存中，未从缓存反推候选数。
- supplementary/exp_calibration_regret_audit_results.json：校准后悔闭合率描述性审计。nuScenes 为 98.1%/98.3%/99.4%/88.5%，comma 为 94.8%/78.9%/10.4%/-14.7%；均为点估计，oracle 只作描述性上界。
- supplementary/exp_lambda_group_calibration_convergence_results.json：仅改变 MLP max_iter 的收敛敏感性检查。脚本按主实验的 RNG 规则恢复有序 fit/validation/test 分组；基线逐 test group 复现主结果（最大绝对差为 0），再与高迭代条件配对比较。

## 辅助审计

RLS/EKF 仅作信息集公平性审计；MLP 收敛敏感性只支持对停止条件变化的有限稳健性；gate 是否定性消融；oracle 仅作描述性上界；各向异性和辨识结果不作为独立机制定理。

## 不可写结论

不声称 lambda 点估计严格单调、严格 1/N、普适 lambda 定律、跨车辆参数直接迁移、heuristic gate 最优、嵌入式实时已验证或 CRIStAL/LOCSP 跨车型泛化。历史泄漏结果和未纳入当前证据链的数据集审计不属于 `results/canonical/`。

## 数据与代码

公开数据请从提供方官方渠道获取并遵守许可：comma2k19、nuScenes CAN bus expansion、CRIStAL/LOCSP 2025（DOI `10.57745/YCXRWF`）。公开代码仓库为：

https://github.com/yummy-web/Physics-Prior-Trajectory-Prediction-from-CAN-Steering

- 统一跨平台需求曲线：deterministic canonical replay（N=3, outer_fold=0, subset_seed=0）与 canonical 的 189 个 scene 值逐项一致，当前 JSON 为 FULL_RUN；独立审计结果为 PASS。

- supplementary/exp_lambda_group_calibration_reproduction_v2_results.json：完整复现 canonical 分组校准实验通过；各数据集 N=3/5/10/20 的方法均值及前部 draw 记录与 canonical JSON 对齐。

- 统一需求曲线的当前有效数字见 supplementary/exp_lambda_transfer_demand_unified_results.json；历史 supplementary/exp_target_calibration_demand_curve_results.json 与 exp_lambda_transfer_results.json 不与统一快照逐项混用。
