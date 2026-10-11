# R0c v1 implementation amendment: histogram schema equivalence

状态：冻结的实现勘误；不改变 R0c 预注册、刺激、seed、dt、kernel、判读规则或正式数据集合。

R0c v1 的正式运行在 0.1 ms R0b 复现门禁处 fail-closed。复核确认，spike bins、active bins、`T_ign`、ignition、state boundary、weight hashes 以及直方图数值均逐项一致；失败原因仅是 R0c 的直方图记录缺少 R0b schema 中的描述字段：`bin_width_mV`、`range_min_mV`、`range_max_mV`、`num_bins`。

本勘误仅将 R0c `hist()` 的输出 schema 补齐为 R0b 的 exact schema。所有数值计算、bin 边界和比较口径保持不变。v1 failure artifact 保留，不纳入正式结论；v2 必须从全新目录按原 126-trial 顺序重跑。

勘误写入时间：2026-10-07 UTC
