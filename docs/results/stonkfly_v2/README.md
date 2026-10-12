# Stonkfly v2 轻量结果包

本目录公开足以审阅主要结论的机器可读数据：

- `summary.json`：阶段 verdict、gate、效应量、资源用量及原始产物哈希；
- `m2_stage_scores.csv`：M2-E、M2-B、M2-C paired 臂逐气味对、逐阶段连续 score。

阶段定义：P0=训练前，P1=acquisition 后，P2=reversal 后，P3=无奖励 retention 后。M2-E 的 score 是固定归一化关联差；M2-B 为 10 个单次 probe 的均值；M2-C 为预注册的 10-probe 聚合均值。

完整 `runs/` 和 MaleCNS `data/` 受仓库忽略规则保护，没有提交到 GitHub。`summary.json` 中的 SHA-256 将本目录结论绑定到本地保留的 formal/statistics/raw/manifest 产物。所有数值均来自已完成运行，没有重跑、重训或重判历史结果。
