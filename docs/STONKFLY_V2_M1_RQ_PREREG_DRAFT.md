# Stonkfly v2 M1-RQ 预注册草案

状态：**CANDIDATE FROZEN / AWAITING INDEPENDENT GO-NO-GO**。本文不是运行授权；M1-RQ、M1-RT 与 M2 均保持 sealed。

## 1. 问题、范围与不可接受推断

M1-RQ 只检验：真实 MaleCNS 右侧 PN→KC adjacency 与一个固定的代理 KC/APL rate model 是否能形成稀疏、可靠、可分离和可恢复的合成表征；以及真实 adjacency 的固定分离统计是否高于 39 个 degree-preserving rewires。

即使通过，也只支持“该连接拓扑和代理动力学的工程模型在此合成任务内工作”。只有 B 相对 C 的预注册增量在 RQ 和未来独立 RT 都重现，才允许表述为“MaleCNS adjacency 在该代理模型内有增量贡献”。不得称为真实果蝇动力学、已验证 APL 数值、学习、意识或交易能力。

## 2. 唯一候选方程和参数

唯一机器可读来源是 `protocols/mb_lab/m1_v1/frozen_config.json`。方程为：

\[
10\,\mathrm{ms}\,\dot u_i=-u_i+\operatorname{mean}_{p\to i}x_p-a,
\quad r_i=[u_i-0.35]_+,
\]

\[
50\,\mathrm{ms}\,\dot a=-a+N_K^{-1}\sum_i r_i.
\]

`W` 是二值 adjacency，并按每个 covered KC 的 PN 入度归一化。显式 Euler、float64、`u(0)=0`、`a(0)=0`。APL-off 只把 `g_A=1` 改成 0。无状态噪声、无 clipping、无学习。

证据等级：V 表示独立实验/解剖约束；P 表示本实验冻结的代理选择；M 表示缺少可直接映射的独立数值。`P(M)` 表示因 M 而必须选定的代理值。

| 参数 | 唯一值与单位 | 等级 | 独立来源与选择理由 |
|---|---:|---|---|
| `tau_K` | 10 ms | P(M) | 成人 KC 可做在体 patch clamp，但没有一个已验证数值可直接映射到本 rate state；10 ms 是预先选择的快速低通工程常数，不来自 v1 或正式刺激。[KC electrophysiology methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC12487611/) |
| `tau_A` | 50 ms | P(M) | APL 负反馈方向有实验支持，但没有适用于单标量全局 proxy 的唯一常数；固定为比 KC 慢 5 倍的工程值。[Lin et al. 2014](https://doi.org/10.1038/nn.3660)、[APL normalization](https://elifesciences.org/articles/74172) |
| `s` | 1 dimensionless state / normalized PN rate | P | 归一化约定，消除与阈值共同缩放的冗余自由度；不是接触数到电导的换算。 |
| `g_A` | 1 dimensionless | P(M) | 只编码 APL 抑制/正规化的已验证方向；大小不宣称已测得，也不根据表征输出拟合。来源同上。 |
| `theta` | 0.35 dimensionless state | P | 在输入被归一到 `[0,1]` 后预先要求显著的组合驱动；与 KC 通常依赖多个 active inputs 的定性约束一致，但 0.35 本身不是实验阈值。[Input density study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10529417/) |
| active threshold | stimulus-window mean `r_i ≥ 0.05` | P | 代理输出分类阈值；生物实验的响应阈值依成像/电生理 assay 而异，不能数值搬运。[population imaging](https://pmc.ncbi.nlm.nih.gov/articles/PMC3180869/) |
| PN 输入 | `[0,1]` dimensionless | P | 合成特征，不映射具体化学气味或发放 Hz。每 odor 恰有 29/145 PN 非零，base amplitude `U[0.8,1.0]`，repeat jitter `N(0,0.05)` 后 clip。 |

结构与现象级 V 约束包括：约 2,000 KC、单 KC 约 5–6 个 PN inputs、KC coincidence detection、APL 负反馈和平均约 5–6% KC odor responsiveness。[input density](https://pmc.ncbi.nlm.nih.gov/articles/PMC10529417/)、[population imaging](https://pmc.ncbi.nlm.nih.gov/articles/PMC3180869/)。这些 V 约束不把任何上述代理数值升级为 V。

明确禁止使用：v1 O1/O2 剂量、点燃/失败区间、`APL×2`、市场结果、盈亏、旧权重或任何 M1 正式输出。候选值是在正式集合 sealed 且从未运行时提交。

## 3. 冻结对象与双分母

- B：145 PN、1,907 个至少接收一条合格 PN 边的右侧 KC、10,732 条二值边；B hash 为 `975fa9d333563bd5ee41afe48074e7a869507caa71b23a5fcb423a2ad3b0b96a`。
- C：39 个以公开域分离 SHA256 seeds 构造的 simple bipartite double-edge-swap 图；每图执行 `5×10,732` 次成功 swap，逐 PN 出度和逐 KC 入度与 B 完全相同；40 个图 hash 已写入 `capacity_benchmark_v2.json`。
- 主科学门分母固定为 1,907 covered KC。每项 sparsity、active set、rate vector、recovery 和 APL 主统计使用同一集合。
- 同时强制报告全部 2,045 KC 的描述性指标：把 138 个结构未覆盖 KC 附加为零响应，分母用 2,045。该视图不参与通过，不能替代 covered 结果。
- A-static 只报告 2,045 KC 的原始诱导图、边界和类型边；不运行其 KC→KC 动力学。

## 4. 时间线与数值方案

每 trial 2,000 ms：`[0,500)` baseline，`[500,1000)` stimulus，`[1000,2000)` withdrawal。response window 为 `[500,1000)`；recovery window 为 `[1500,2000)`。500 ms odor pulse 与成人果蝇 PN/KC 电生理常用刺激尺度一致，但本合成输入不代表真实 odor 剂量。[Turner et al.](https://journals.physiology.org/doi/full/10.1152/jn.01283.2007)

运行 `dt=0.1,0.05,0.025 ms`；科学统计只用 0.025 ms。跨 dt 的实施门：相对 0.025 ms，active fraction 绝对差分别不超过 0.5/0.25 percentage point，rate-vector NRMSE 不超过 5%/2.5%，active Jaccard 不低于 0.90/0.95；误差减半比不超过 0.6，除非两误差均低于 `1e-4`；所有科学门布尔分类一致。

active 定义为 response window 的 mean proxy rate `≥0.05`。tail active 使用 recovery window 同一 mean-rate threshold。恢复 state error 固定为 trial 结束时 `max(max_i |u_i|, |a|)`，相对于模型 unit scale 1；95% trial `≤0.05`、任何 trial `>0.10` 失败。最后两个各 250 ms withdrawal bins 的 total proxy rate 不得上升。ZERO 在所有 dt、APL on/off 下不得产生 active KC。

## 5. 刺激、重复与数据隔离

M1-RQ 和 M1-RT 各为 16 odors ×10 repeats，独立 seed。每 odor 选择 29 个 PN，不放回；inactive PN 恒为零。四个域的最终公开 commitments 位于 `protocols/mb_lab/m1_v1/domains-freeze-v4/`：

- M1_RQ：内存生成 collection 后只记录 content SHA256；
- M1_RT：同上，seed、content 和 AES-GCM key 均独立；
- M2_LQ/M2_LT：只保留独立 seed reservation，不生成学习任务内容。

公开 manifest 不含 seed；seed envelope 使用 AES-256-GCM、domain label 作 AAD，key 与密文为 0600 且在 git-ignored `data/mb_lab/sealed/m1_v1-freeze-v4/`。runner 不接受任意输入路径；domain 必须与授权 artifact 精确一致。RQ 失败不解封 RT；RQ authorization 不能读取 RT manifest/key。

`domains/`、`domains-freeze-v2/` 和 `domains-freeze-v3/` 是准备期失败/被取代证据，不是有效候选；不得混用。唯一候选为 `freeze-v4`。

## 6. 冻结科学判定

先在每 odor 内聚合；16 个 odor 是独立单位，trial 不是。

1. **S1 sparsity**：`F_o=median_t f_ot`；16/16 在 `[0.02,0.10]`，其 odor median 在 `[0.04,0.08]`；任一 trial 为 0 或 `>0.20` 失败。
2. **S2 reliability/separation**：重复按 `(0,1)…(8,9)` 五个不重叠 pair，`R_o` 为五个 Jaccard 的 median。每 odor 的 between controls 固定为 noiseless base PN cosine 最大的三个其他 odor（cosine 降序、odor index 升序破同值），比较相同 repeat，30 个 Jaccard 的 median 为 `D_o`。`T_sep=mean_o(R_o-D_o)`；median margin 只描述。通过需 `median R_o≥0.60`、至少 14/16 的 `R_o≥0.50`、`T_sep≥0.20`，并对 16 个 margin 做 9,999 次固定-seed Monte Carlo sign-flip，单侧加一 p `≤0.01`。
3. **S3 recovery**：所有 trial tail active fraction `≤0.01`；至少 95% state error `≤0.05`，任一 `>0.10` 失败；后半 recovery rate 不得高于前半。
4. **S4 APL-off**：B 的 paired input/common random numbers；`Delta_o=median_t(f_off-f_on)`。需 median delta `≥0.02`、至少 13/16 正值，并以 mean delta 做完整 `2^16` exact sign-flip，单侧 `p≤0.01`。
5. **S5 B/C**：B 和每个 C 使用同一 `T_sep`。`p_topo=(1+#(T_C≥T_B))/40≤0.05` 且 `T_B-median(T_C)≥0.05`。只有 RQ 与未来 RT 同向通过才支持 proxy 内拓扑增量；B 能通过 S1–S4 本身不支持真实拓扑优越。

## 7. 实施门与失败优先级

I1 provenance、I2 import/runtime isolation、I3 graph invariants、I4 RNG/domain isolation、I5 scalar oracle、I6 finite/bounds、I7 ZERO/recovery mechanics、I8 cross-dt 必须依次通过。

终止优先级固定：

1. `M1_IMPLEMENTATION_FAILURE`；
2. `M1_BUDGET_EXHAUSTED`；
3. `M1_NUMERICAL_NO_GO`；
4. `M1_REPRESENTATION_NO_GO`（RQ S1–S4 任一失败，不开 RT）；
5. `M1_GENERALIZATION_NO_GO`；
6. `M1_MECHANISM_QUALIFIED_TOPOLOGY_NOT_DEMONSTRATED`；
7. `M1_MECHANISM_AND_PROXY_TOPOLOGY_INCREMENT_QUALIFIED`。

预算是运行时硬停止而非结果门。每正式阶段 24 CPU-hours、24 wall-hours；M1 总 48 CPU-hours；peak RAM 4 GiB；artifacts 5 GiB；无 GPU。失败后不换 seed、删 odor、放宽阈值、增减 C 或补跑有利部分。

## 8. 当前授权状态

当前 runner 仅允许 `BENCHMARK_ONLY` toy 路径。M1-RQ、M1-RT、M2 访问均 fail-closed。正式运行需要独立评审给出 M1-RQ GO 和单独 authorization artifact；本草案不构成该授权。
