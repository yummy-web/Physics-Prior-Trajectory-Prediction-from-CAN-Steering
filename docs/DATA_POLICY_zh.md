# 数据与开源政策

## 发布内容

仓库发布当前论文实际需要的核心代码、结果 JSON/CSV 快照、中文实验报告和高清图表。发布内容不包含原始公开数据，也不包含私有装载机数据、ROS bag、缓存、运行日志、投稿 Word 文件、回复信或内部构建脚本。

## 数据获取

comma2k19、nuScenes CAN bus expansion 和 CRISAL 2025 必须从各自官方来源获取，使用者应遵守提供方许可。CRISAL 数据页面为 DOI 10.57745/YCXRWF。仓库只记录解析协议和结果 provenance，不重新托管数据。

## 可复现边界

公开数据和依赖齐备时，核心实验可按 docs/REPRODUCE_zh.md 运行。私有装载机案例不能由公开仓库独立复现。CRISAL 的结果依赖本地下载并解压的七个 acquisition；其参考轨迹为 INSPVA 融合导航输出，不应解释为独立真值。

## 不发布的历史内容

泄漏版本、旧的 heuristic gate 主张、严格 1/N 定律和未进入最终证据链的 Oxford/A2D2/AEGIS 探索不属于发布结果。它们不应被读者当作当前论文的 canonical 实验。
