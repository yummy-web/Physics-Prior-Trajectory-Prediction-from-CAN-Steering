# CRISAL 同车多采集外部验证

本目录记录 CRISAL 2025 的数据准备要求、运行入口和解释边界。当前发布快照使用七次同一辆 Renault Zoe 的采集，按采集分组进行留一采集评估。原始 ZIP、ROS bag、缓存和内部下载脚本均不随仓库发布。

## 数据源与边界

- 官方数据记录：DOI 10.57745/YCXRWF（Recherche Data Gouv / Dataverse）。
- 原始下载与解压数据放在仓库外的 CRISAL_ROOT 目录，不得提交原始数据。
- 七次采集来自同一车辆、同一日期的多次运行，支持同车多采集的时间外部检查，不支持跨车型或跨平台泛化。

## 复现入口

在仓库根目录运行：

    python -m pip install -r requirements-crisal.txt
    python src/exp_crisal_external_pilot.py --bag <CRISAL_ROOT>/extracted/<acquisition>/<file>.bag
    python src/exp_crisal_grouped_external.py --data-root <CRISAL_ROOT>

正式 grouped 运行应发现七个合格采集，并将结果写入 results/crisal_external_validation/。若只使用单个 bag，只能进行 loader、QC 或探索性 pilot，不能复现七采集外部结果。

## 协议与统计单位

- 采用 4 个采集拟合、2 个采集验证、1 个采集外层测试的 leave-one-acquisition-out 方案。
- 统计单位为 acquisition；重叠锚点、验证组合和模型 seed 不作为额外独立样本。
- CAN 速度和转向只使用锚点时刻之前可用的消息；物理滚动采用锚点零阶保持。
- INSPVA 是融合导航输出，不是独立定位真值。

## 当前结果

- 3 s FDE 的采集级宏平均：INS velocity ZOH 2.655 m，拟合转向增益 CTRV 2.438 m，满残差 2.425 m，收缩系数 κ=0.5 为 2.421 m。
- 2 s FDE 的对应结果为 1.251、1.150、1.128 和 1.127 m。
- 结果应解释为同车多采集外部检查，不能外推为跨车型泛化，也不构成独立真值精度验证。

结果快照位于 results/canonical/crisal_external_validation/，包括 JSON、CSV、QC 记录和中文报告。
