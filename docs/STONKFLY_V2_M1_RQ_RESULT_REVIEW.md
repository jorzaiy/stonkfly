# Stonkfly v2 M1-RQ freeze-v5 正式结果审查

日期：2026-10-12 UTC
run ID：`m1-rq-freeze-v5-20261011T133740Z`
正式 verdict：**`M1_REPRESENTATION_NO_GO`**
后续状态：**M1-RT 与 M2 继续 NO-GO；不追加实验，不调参。**

## 完整性结论

运行于 2026-10-11 19:57 UTC 正常写出 `COMPLETE.json`，无存活进程、无 failure attempt。独立只读审查验证了 129 组 NPZ/sidecar、6 个最终产物、40 个拓扑哈希和 31 个结果相关源码/协议哈希；source hashes 首尾一致，COMPLETE 对 manifest 的哈希一致。

正式目录已逐字节归档到 `runs/mb_lab/m1-rq-freeze-v5-20261011T133740Z/`。源与归档副本共有 266 个文件、36,337,514 bytes，确定性树哈希均为 `0f2e1ecaa0927782743895a0b552faa72a3efeb3e5e270e8c60981b3fbfa1f20`。

## 冻结 gate 结果（科学 dt = 0.025 ms）

| Gate | 结果 | 关键统计与解释 |
|---|---:|---|
| S1 稀疏度 | FAIL | 1,907 个可驱动 KC 的 odor-level median 总中位数为 6.922%，但 odor 16 为 10.121%，略高于逐 odor 上限 10%。其余范围与单 trial 20% 上限满足。|
| S2 稳定性/特异性 | PASS | median reliability 0.9158；16/16 odors ≥ 0.5；`T_sep=0.8257`；固定 PCG64 检验 `p=0.0001`。|
| S3 撤刺激恢复 | PASS | 最大 tail active fraction 为 0；100% trial 的 state error ≤ 0.05，最大误差约 `5.36e-11`；后半 tail 无上升。|
| S4 APL-off | FAIL | 16/16 odors 方向均为正，精确符号检验 `p=1.526e-5`，但 median 增量仅 0.003671（0.367 个百分点），未达到冻结的 0.02（2 个百分点）效应门槛。|
| S5 B vs 39 C | FAIL | `p=0.625`，`T_B - median(T_C)=-0.003751`；没有显示真实 B 拓扑优于固定度随机 C。|

三个 dt 的 gate 向量完全一致：`[S1=false, S2=true, S3=true, S4=false, S5=false]`。

## 双分母与数值检查

以 1,907 个可驱动 KC 为主分母：单 trial active fraction 范围为 3.723%–10.540%，各 odor median 范围为 4.300%–10.121%，各 odor median 的总中位数为 6.922%。唯一越过 S1 odor 上限的是 odor 16。

以全部 2,045 KC 为描述性分母：各 odor median 范围为 4.010%–9.438%，总中位数为 6.455%；odor 16 为 9.438%。该分母不用于改变预注册 verdict。

I8 跨 dt 检查通过：

- dt 0.1 ms：最大 active-fraction error 0.001049，最大 NRMSE 0.000224，最小 Jaccard 0.9804；
- dt 0.05 ms：最大 active-fraction error 0.001049，最大 NRMSE 0.0000748，最小 Jaccard 0.9855；
- 无 halving failure，最大有限 halving ratio 为 0.4999；
- ZERO 全部通过，三个 dt 的科学分类一致。

## 资源

- CPU：6.294 hours；
- wall time：6.296 hours；
- peak RSS：0.949 GiB；
- 正式产物：34.60 MiB。

全部低于 24 CPU-hours、24 wall-hours、4 GiB 和 5 GiB 的冻结硬上限。

## 科学解释

本次结果支持一个较窄的工程结论：在该代理动力学与固定合成气味下，B 网络产生了高度可重复、可区分、撤刺激后恢复且跨 dt 稳定的 KC 表征。

它不支持以下更强结论：

- 不能称 M1-RQ qualified，因为 S1 和 S4 未通过；
- 不能认为冻结的 APL 代理动力学产生了足够大的稀疏化因果效应；虽然方向一致且统计显著，但效应量不足；
- 不能声称真实果蝇 PN→KC 拓扑优于度保持随机网络；S5 明确未显示优势；
- 不能据此作生物学验证、学习验证或交易能力声明。

因此按预注册停止：不打开 RT，不进入 M2，不根据结果修改参数或补跑实验。若未来重启研究，应作为新的、独立评审的协议版本处理，而不是对本次 RQ 的补救。
