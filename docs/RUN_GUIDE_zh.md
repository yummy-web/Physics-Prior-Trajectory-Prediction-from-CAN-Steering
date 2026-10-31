# 运行指南

建议先运行单个小实验验证数据路径和依赖，再运行完整实验。所有脚本从仓库根目录调用，避免把 src 目录外的本地稿件、投稿包或原始数据复制到仓库。

python -m compileall src
python src/exp_lambda_group_calibration_convergence.py --help

若数据不在默认位置，请使用脚本提供的参数或环境变量。运行日志和缓存不应提交；结果快照应复制到 results/canonical/ 前经过数字和 provenance 核对。
