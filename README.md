# 受限 CAN 信号下数据稀缺车辆轨迹预测的验证集残差信任校准

本仓库对应论文“受限 CAN 信号下数据稀缺车辆轨迹预测的验证集残差信任校准”。仓库公开的是可复现实验代码、论文实际引用的结果快照和中文实验说明，不重新分发任何原始数据。

## 研究范围

研究对象是无地图、无周围车辆轨迹输入的单车短时预测。预测器由控制条件物理先验 p 和残差网络 r 组成：

ŷ = p + λ r

λ 是离线标量：残差网络只在拟合分区训练，λ 在独立验证分区由闭式最小二乘对齐量估计并按验证锚点数量进行经验收缩，随后在外层测试前固定。论文把它定位为受限 CAN 和数据稀缺条件下的可审计校准流程，而不是 physics-plus-residual、标量融合或收缩思想本身的首次提出。

当前证据支持的范围是：nuScenes 中收缩收益随平台特定预算增加而下降；comma2k19 的路线级收益仅在 N=3 时具有高于零的 95% bootstrap 区间，N=5 的区间跨 0，N≥10 接近零；跨平台迁移支持“收缩方向可复用、具体数值仍需目标平台校准”；CRIStAL/LOCSP 只提供同一车辆七次采集的时间外部验证。旧的 heuristic gate、严格 1/N 定律、普适 λ 定律和嵌入式实时结论均不属于当前主张。

## 仓库结构

src/                         核心模块与当前因果实验脚本
deprecated/                  历史、泄漏或未纳入当前证据链的文件，仅作审计留档
results/canonical/           论文实际使用或明确引用的 JSON/CSV 快照
results/reports_zh/          中文实验报告、收敛诊断和 provenance
figures/canonical/           论文配套高清 PNG/PDF 图表
experiments/                 外部数据实验的运行说明
results/supplementary/      新增补充审计结果（不替换 canonical 主结果）
docs/                        数据、复现和发布政策

## 数据来源与许可

| 数据 | 用途 | 获取方式 |
|---|---|---|
| comma2k19 | 公开路线级主实验 | https://github.com/commaai/comma2k19 |
| nuScenes CAN bus expansion | 公开场景级主实验 | https://www.nuscenes.org/nuscenes |
| CRIStAL/LOCSP 2025 | 同车多采集外部验证 | https://doi.org/10.57745/YCXRWF |
| 装载机数据 | 私有辅助案例 | 不随仓库发布 |

数据由各自提供方的许可和使用条款约束。用户应从官方来源获取数据，并自行确认使用许可。仓库不上传 comma2k19、nuScenes、CRIStAL/LOCSP 原始文件、缓存、ROS bag 或私有装载机数据。

## 环境

python -m venv .venv
pip install -r requirements.txt
pip install -r requirements-crisal.txt
# Optional causal GRU supplement:
# pip install -r requirements-gru.txt

## 复现

详细路径和命令见 docs/REPRODUCE_zh.md、docs/DATA_POLICY_zh.md、docs/PROVENANCE.md 和 experiments/crisal_external_validation/README.md。

先在本地准备数据，再按数据集运行对应脚本。结果快照已经放在 results/canonical/；没有私有装载机数据时，不能复现该辅助案例的数值。

## 结果与边界

所有可引用数字、统计单位、split、seed 和限制见 results/reports_zh/PROVENANCE.md。canonical 目录中的 oracle 量仅作描述性上界，不能作为部署结果；CRIStAL/LOCSP 的 INSPVA 是融合导航输出，不是独立定位真值；时间信息约束实验不等同于真实 IMU-only 或嵌入式硬件验证。

## 代码引用

https://github.com/yummy-web/Physics-Prior-Trajectory-Prediction-from-CAN-Steering
