# 复现实验说明

## 1. 数据准备

数据不随仓库发布。请从 README 中的官方链接获取并解压到任意本地目录，然后通过脚本参数或仓库现有的路径配置指向数据。comma2k19 需要能够被 comma_loader.py 发现，nuScenes 需要 CAN bus expansion 的 JSON 文件。CRISAL 需要先解压 ROS1 bag，并通过 --data-root 指定数据根目录。

## 2. 基础环境

pip install -r requirements.txt
pip install -r requirements-crisal.txt

## 3. 当前主实验脚本

在仓库根目录执行，例如：

python src/exp_lambda_group_calibration.py
python src/exp_lambda_group_calibration_convergence.py
python src/exp_lambda_analytic.py
python src/exp_lambda_analytic_nusc.py
python src/exp_lambda_transfer.py
python src/exp_horizonwise_v2.py
python src/exp_identifiability.py
python src/exp_recursive_baselines_causal.py

具体输入位置和输出文件以脚本的 --help 及 results/reports_zh/PROVENANCE.md 为准。重新运行前请先确认数据许可、split、seed 和统计单位；发布快照位于 results/canonical/。

## 4. CRISAL

python src/exp_crisal_grouped_external.py --data-root <CRISAL根目录>

该协议要求正好七个通过 QC 的采集，并按 4 个拟合、2 个验证、1 个外层测试进行留一采集评估。公开仓库不包含 bag 文件。

## 5. 结果解释

不要用窗口数量代替路线、scene 或 acquisition 数量；不要把测试集 oracle 当作可部署系数；不要把 CRISAL 同车多采集写成跨车辆验证。冻结结果及其可写边界见 results/reports_zh/PROVENANCE.md。
