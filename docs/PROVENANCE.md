# 公开复现实验 Provenance

## 仓库范围

本文件描述当前公开仓库中可引用的实验代码和结果快照。原始 comma2k19、nuScenes CAN bus expansion、CRIStAL/LOCSP 2025 和私有装载机数据均不随仓库发布。

## 方法

预测器写为 `y_hat = p + lambda*r`。`p` 是控制条件物理先验，`r` 是残差 MLP。残差网络在拟合分区训练，lambda 在验证分区由闭式最小二乘对齐量估计，再使用 `n/(n+kappa)` 进行经验收缩并在外层测试前固定。原始 CAN 到位姿/评价时间轴的对齐采用严格过去值保持：每个查询时刻只取时间戳不晚于该时刻的最新 CAN 消息，首条 CAN 之前的查询样本被剔除。锚点之后的滚动固定锚点 CAN 速度和转向，航向率由截至锚点的位姿后向差分获得。该约束是时间信息约束，不等同于真实 IMU-only 传感器验证。

## 统计单位与主结果

- nuScenes lambda 实验：566 个 scene 聚类；3 s 收缩相对满残差的改善量为 N=3/5/10/20 时 1.785/0.947/0.526/0.164 m，scene-level 95% CI 分别为 [1.664,1.919]、[0.864,1.042]、[0.476,0.582]、[0.137,0.193]。
- comma2k19：42 条 route；统一分组校准实验的路线级收益为 N=3/5/10/20 时 0.3275/0.0550/−0.0136/−0.0039 m，95% CI 分别为 [0.1362,0.5329]、[−0.0079,0.1253]、[−0.0517,0.0268]、[−0.0140,0.0075]。点估计与区间来自同一批预测。
- 跨平台迁移：支持收缩方向可复用，但具体 lambda 数值仍需目标平台验证校准。
- CRIStAL/LOCSP：7 次同一辆 Renault Zoe、同日采集；统计单位为 acquisition，3 s FDE 的 INS velocity ZOH、拟合 CTRV、满残差和 kappa=0.5 分别为 2.655、2.438、2.425 和 2.421 m。该结果是同车多采集外部检查，不是跨车型泛化。

## 辅助审计

RLS/EKF 仅作信息集公平性审计。MLP 收敛敏感性先逐 test group 复现主校准结果，再仅改变 `max_iter`；两个平台各预算的收缩收益方向均保持。gate 是否定性消融；oracle 仅作描述性上界。方向性端点误差表按 comma route、nuScenes scene 聚类汇总，其中 nuScenes 为 200-scene 因果子集；该表仅作误差分量描述，不作为传感器因果性、可观测性或机制定理。

## 不可写结论

不声称 lambda 点估计严格单调、严格 1/N、普适 lambda 定律、跨车辆参数直接迁移、heuristic gate 最优、嵌入式实时已验证或 CRIStAL/LOCSP 跨车型泛化。历史泄漏结果和未纳入当前证据链的数据集审计不属于 `results/canonical/`。

## 数据与代码

公开数据请从提供方官方渠道获取并遵守许可：comma2k19、nuScenes CAN bus expansion、CRIStAL/LOCSP 2025（DOI `10.57745/YCXRWF`）。公开代码仓库为：

https://github.com/yummy-web/Physics-Prior-Trajectory-Prediction-from-CAN-Steering
