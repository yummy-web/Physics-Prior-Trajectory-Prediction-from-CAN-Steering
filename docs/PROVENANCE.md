# 公开复现实验 Provenance

## 仓库范围

本文件描述当前公开仓库中可引用的实验代码和结果快照。原始 comma2k19、nuScenes CAN bus expansion、CRIStAL/LOCSP 2025 和私有装载机数据均不随仓库发布。

## 方法

预测器写为 `y_hat = p + lambda*r`。`p` 是控制条件物理先验，`r` 是残差 MLP。残差网络在拟合分区训练，lambda 在验证分区由闭式最小二乘对齐量估计，再使用 `n/(n+kappa)` 进行经验收缩并在外层测试前固定。实验协议使用后向差分航向率、锚点 CAN 速度和转向零阶保持，不读取预测时刻之后的控制或位姿样本。该约束是时间信息约束，不等同于真实 IMU-only 传感器验证。

## 统计单位与主结果

- nuScenes lambda 实验：566 个 scene 聚类；3 s 收缩相对满残差的改善量为 N=3/5/10/20 时 1.663/0.960/0.420/0.137 m，scene-level 95% CI 分别为 [1.461,1.877]、[0.860,1.069]、[0.380,0.464]、[0.114,0.163]。
- comma2k19：42 条 route；解析式 lambda 结果仅报告点估计，路线级审计收益为 0.1616/0.0569/0.0040/−0.0012 m，区间均包含 0。
- 跨平台迁移：支持收缩方向可复用，但具体 lambda 数值仍需目标平台验证校准。
- CRIStAL/LOCSP：7 次同一辆 Renault Zoe、同日采集；统计单位为 acquisition，3 s FDE 的 INS velocity ZOH、拟合 CTRV、满残差和 kappa=0.5 分别为 2.655、2.438、2.425 和 2.421 m。该结果是同车多采集外部检查，不是跨车型泛化。

## 辅助审计

RLS/EKF 仅作信息集公平性审计。MLP 收敛敏感性采用同一次当前代码运行内的 baseline 与 higher-`max_iter` 受控配对；历史冻结 JSON 的脚本 SHA 与当前开源脚本不同，因此逐数值复现核对单列为失败的快照审计，不参与 `max_iter` 归因。gate 是否定性消融；oracle 仅作描述性上界。方向性端点误差表按 comma route、nuScenes scene 聚类汇总，其中 nuScenes 为 200-scene 因果子集；该表仅作误差分量描述，不作为传感器因果性、可观测性或机制定理。

## 不可写结论

不声称 lambda 点估计严格单调、严格 1/N、普适 lambda 定律、跨车辆参数直接迁移、heuristic gate 最优、嵌入式实时已验证或 CRIStAL/LOCSP 跨车型泛化。历史泄漏结果和未纳入当前证据链的数据集审计不属于 `results/canonical/`。

## 数据与代码

公开数据请从提供方官方渠道获取并遵守许可：comma2k19、nuScenes CAN bus expansion、CRIStAL/LOCSP 2025（DOI `10.57745/YCXRWF`）。公开代码仓库为：

https://github.com/yummy-web/Physics-Prior-Trajectory-Prediction-from-CAN-Steering
