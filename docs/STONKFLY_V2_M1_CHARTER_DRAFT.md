# Stonkfly v2 M1 实施章程草案

> 2026-10-11 冻结准备修订：唯一候选参数、时间线、统计量、双分母、39 个 C seeds、四域 `freeze-v4` manifests 和预算证据见 `protocols/mb_lab/m1_v1/frozen_config.json`、`docs/STONKFLY_V2_M1_RQ_PREREG_DRAFT.md` 与 `docs/STONKFLY_V2_M1_FREEZE_REVIEW.md`。本文件中与这些文件冲突的候选描述由上述文件取代；正式运行仍未授权。

状态：**DRAFT / IMPLEMENTATION-PREPARATION ONLY。不是预注册，不授权正式数据集生成、解封或实验运行。**
里程碑：M1 合成气味表征
明确排除：M2 可塑性、奖励、MBON/DAN 学习与反转；全部市场和交易模块。

## 1. M1 问题与可接受结论

M1 只问两个分开的、预先可证伪的问题：

1. **机制问题 H-MECH**：真实 PN→KC adjacency 与一个固定、公开的代理 KC/APL 动力学组合，能否产生稀疏、可靠、可分离、撤刺激可恢复的合成气味表征？
2. **拓扑问题 H-TOPO**：在动力学、权重规则、刺激和噪声完全相同时，真实 PN→KC adjacency 的预注册 separation statistic 是否超过 degree-preserving rewires？

允许的结论：

- H-MECH 通过、H-TOPO 不通过：一般稀疏扩展代理机制工作，没有真实 adjacency 增量优势证据；
- H-MECH 和 H-TOPO 均通过并在 M1-RT 重现：此 MaleCNS adjacency 在该代理模型和任务内有增量贡献；
- 任一失败：保留否定结果并停止，不调参追点。

禁止的结论：真实果蝇已被模拟、APL 数值被验证、真实连接结构普遍优越、模型能学习/反转/交易。

## 2. 授权范围

当前允许：

- 只读 inventory；
- A 的静态编译和边界损失账本；
- B/C topology 编译与 degree 校验；
- 方程 reference、toy oracle、统计函数、stage lock、artifact writer；
- 仅使用 `BENCHMARK_ONLY` fixture 的单测与最小 CLI smoke。

当前禁止：

- 生成、查看或运行 M1-RQ/M1-RT 的刺激内容；
- 生成、查看或运行 M2-LQ/M2-LT 内容；
- 实现 plasticity、reward、MBON/DAN 学习或 reversal；
- import 或调用 `stonkfly.market`、`stonkfly.reinforcement`、`stonkfly.actions`、`stonkfly.broker`、`stonkfly.risk`、`stonkfly.ledger`、交易 controller/decoder；
- 修改 v1 文件、锁、结论或产物。

## 3. M1 静态对象

### 3.1 候选选择

初始审计以右侧为 reference hemisphere：145 个 O0 规则候选 cholinergic direct PN、2,045 个总 KC、1 APL。结构复核显示只有 1,907 个右侧 KC 至少接收一条来自合格右侧 PN 的边；其余 138 个不得作为 B/C 的零响应样本。B/C 在不读取模型输出的条件下使用这 1,907 个 olfactory-covered KC。左侧对应为 145 PN、2,019 总 KC、1,887 covered KC。正式冻结前必须由 `inventory.py` 重建并输出：source IDs、cell types、`somaSide`、neurotransmitter、covered/uncovered 集合、每类节点 hash、每类边 hash、接触数、内部边和双向边界边。

`somaSide` 而不是 `rootSide` 用于侧别；本地 annotations 中这些 MB cell 的 `rootSide` 为空。左侧作为结构 replication inventory，不参与参数选择。

### 3.2 三个对象

- **A-static**：PN/KC/APL 原始诱导子图，只作结构与成本审计。右侧目前约 344,658 条内部边，其中 KC→KC 约 323,853 条。无正式动力学 verdict。
- **B-real**：只保留真实 PN→olfactory-covered-KC adjacency；APL 是独立代理方程。PN→KC 权重规则对 B/C 相同。
- **C-rewire[0..38]**：对 B 的二分图做 double-edge swaps，精确保留所有 PN 出度和 KC 入度。禁止 self/multiedge（若 B 本身为简单二分图）；每个 C 必须与 B 不同，且 C 间 hash 唯一。

B/C 共同使用相同 PN/KC 身份数量、输入、common random numbers、积分法、参数和统计。

## 4. 候选模型方程：冻结前阻塞项

以下是实现 reference 的候选形式，不是已冻结参数：

\[
\tau_K \frac{du_i}{dt}=-u_i+s\sum_p \tilde W_{ip}x_p(t)-g_A a(t),
\]

\[
r_i(t)=[u_i(t)-\theta]_+,
\]

\[
\tau_A \frac{da}{dt}=-a+\frac{1}{N_K}\sum_i r_i(t).
\]

其中 `x_p(t)` 是外部合成 PN rate；`W` 只表示 adjacency；`W_tilde` 按每个 KC 的 PN 入度做预先声明的归一化；`u` 是无主观含义的 KC proxy state；`r` 是非负 proxy output；`a` 是单标量 APL feedback proxy。模型无 KC→KC recurrence、无 DAN/MBON、无权重学习。

APL-off 的唯一实现是设 `g_A=0`；不得更改初态、刺激、噪声或其他参数。

正式申请前必须冻结且逐项标注 V/P/M：`tau_K, tau_A, s, g_A, theta`、输入单位/范围、噪声过程、积分法、初态、active threshold、trial 时间线和恢复容差。当前 toy benchmark 中使用的 20/50 ms 仅为吞吐测试常数，**不是候选参数来源，不得自动进入正式配置**。

如果在不查看正式集合的条件下仍找不到独立依据选择唯一参数集，M1 formal 为 NO-GO；不得提交 grid。

## 5. 四套集合和访问控制

### 5.1 seed domains

必须使用密码学域分离，例如：

- `STONKFLY-V2/M1/RQ/STIMULUS/v1`
- `STONKFLY-V2/M1/RT/STIMULUS/v1`
- `STONKFLY-V2/M2/LQ/STIMULUS/v1`
- `STONKFLY-V2/M2/LT/STIMULUS/v1`
- `STONKFLY-V2/BENCHMARK_ONLY/v1`

正式 seeds 不得出现在源码、普通日志或 smoke artifact 中；sealed manifest 只公开 schema、数量、生成器 hash、密文/文件 hash 和 custodian 状态。

### 5.2 集合用途

- M1-RQ：16 odor × 10 repeats；只作一次 M1 qualification。
- M1-RT：独立 16 odor × 10 repeats；只有 RQ 全部门通过才可解封一次。
- M2-LQ/M2-LT：在 M1 期间只保留 sealed manifest；M1 runner 必须拒绝其路径。

RQ 失败后禁止访问 RT。RT 失败后禁止用 RT 诊断性调参；只能报告已冻结的 per-odor metrics 和失败原因。任何实现/参数/统计改变都产生新协议版本和四套新 seeds，旧结果不合并。

## 6. 观测定义

正式冻结必须给出具体刺激和观测窗。以下数学定义固定其形式：

- `z_{o,t,i}=1` 当 trial `(odor=o, repeat=t)` 中 KC `i` 在 stimulus response window 的积分输出达到冻结的 active threshold，否则为 0；
- active fraction `f_{o,t}=N_K^{-1} sum_i z_{o,t,i}`；
- rate vector `q_{o,t}` 为同一窗口每 KC 的积分输出除以窗口长度；
- Jaccard `J(z,z')=|z∩z'|/|z∪z'|`；两者均空时定义为 missing，而不是 1；任一正式 odor 出现全空 trial 触发 sparsity failure；
- cosine 只作描述性指标，零向量记 missing；不得替代主指标。

所有聚合先在 odor 内完成，再以 odor 为独立单位；不能把同一 odor 的 45 个 trial pairs 当作 45 个独立样本。

## 7. 实施正确性与数值资格（I-gates）

I-gates 失败只说明实现/数值不合格，不检验生物假设。

1. **I1 provenance**：输入、节点、边、参数、runner、tests、environment hashes 完整，start/end 相同。
2. **I2 isolation**：AST/import audit 和运行时 module audit 均未加载任何市场、交易、reward、plasticity 或 M2 模块。
3. **I3 graph invariants**：B 与 39 个 C 的 PN/KC 数量、逐节点度数完全相同；hash 唯一；没有未声明的边或权重差异。
4. **I4 RNG isolation**：五个 seed domains 无重用；B/C 和 APL on/off 对应 trial 使用 common random numbers；same-seed 重跑 bitwise 相同。
5. **I5 equation oracle**：4-PN/32-KC toy fixture 上 vectorized reference 与独立 scalar/dense oracle 在 float64 下 `rtol≤1e-10, atol≤1e-12`。
6. **I6 finite/bounds**：所有状态有限，数组形状/单位正确，无 clipping、overflow 或 silent NaN replacement。
7. **I7 ZERO/recovery mechanics**：ZERO 无受刺激响应；无输入时方程按解析被动解趋向初态。科学恢复指标另见 S3。
8. **I8 cross-dt convergence**：相同连续输入路径和 seeds 下，以 0.025 ms 为 reference：
   - 每 trial active fraction 的绝对差：0.1 vs 0.025 ≤0.5 percentage point，0.05 vs 0.025 ≤0.25 point；
   - rate-vector normalized RMSE：分别 ≤5% 和 ≤2.5%；
   - active-set Jaccard：分别 ≥0.90 和 ≥0.95；
   - 步长减半误差 `E(0.05,0.025) ≤ 0.6 E(0.1,0.05)`，除非两者都低于冻结 absolute floor；
   - 所有科学 gate 的布尔分类相同。

I8 是工程容差 P，不是生物范围。科学统计只使用 0.025 ms 结果；较粗 `dt` 不构成额外样本。

## 8. 科学假设与统计（S-gates）

所有检验同时要求统计显著性与最小实际效应。RQ 是资格门，RT 是一次性确认；不跨集合合并 p 值。

### S1 稀疏性

对每个 odor 先取 `F_o=median_t(f_{o,t})`。

- 16/16 odor 均须 `0.02≤F_o≤0.10`；
- `median_o(F_o)` 须在 `[0.04,0.08]`；
- 任一 trial `f=0` 或 `f>0.20` 直接失败。

2–10% 是围绕约 6% 生物现象的预注册工程容忍带，不是实验置信区间。S1 是等价区间判定，不计算“显著不为零”的 p 值。

### S2 试次可靠性与刺激分离

对每个 odor 的 10 repeats，按冻结 seed 顺序形成 5 个不重叠 pairs，定义：

\[
R_o=\operatorname{median}_{k=1..5}J(z_{o,2k-1},z_{o,2k}).
\]

以冻结的输入 cosine strata 为条件，为每个 odor/repeat 匹配不同 odor 的对照，定义 `D_o` 为 matched between-odor Jaccard 中位数。主统计量：

\[
T_{sep}=\operatorname{mean}_o(R_o-D_o).
\]

通过要求：

- `median_o(R_o)≥0.60` 且至少 14/16 odor 的 `R_o≥0.50`；
- `T_sep≥0.20`；
- 对 16 个 odor-level `R_o-D_o` 做 9,999 次固定-seed Monte Carlo sign-flip，使用均值统计量，单侧 `p=(1+#T_perm≥T_obs)/10000≤0.01`。odor 是唯一交换单位；median margin 只作描述性报告。

绝对阈值是 P；permutation 防止把重复 trial 伪装成独立样本。若匹配算法不能为每个 odor 找到预注册数量的对照，判 protocol failure，不临时放宽。

### S3 撤刺激恢复

每 trial 定义 withdrawal 最后 500 ms 的 active fraction `w_{o,t}` 和全部连续状态相对该 trial pre-stimulus baseline 的归一化误差 `e_{o,t}`。

- 所有 trial `w_{o,t}≤0.01`；
- 至少 95% trial `e_{o,t}≤0.05`，且任何 trial `e>0.10` 失败；
- 不得存在最后两个等长 withdrawal bins 活动上升的 trial。

这是模型稳定/可退出假设；阈值为 P。

### S4 APL-off 因果检验

对 B-real 的每个 odor/repeat，用相同输入和噪声分别运行 APL-on/off。先按 odor 聚合：

\[
\Delta_o=\operatorname{median}_t(f^{off}_{o,t}-f^{on}_{o,t}).
\]

定义检验统计量 `T_APL=mean_o(Delta_o)`。唯一主指标仍是 active-fraction 配对变化：

- `median_o(Delta_o)≥0.02`；
- 至少 13/16 odor `Delta_o>0`；
- 对 16 个 odor 的 `Delta_o` 做完整 `2^16` exact sign-flip permutation，以 `T_APL` 计算单侧 `p≤0.01`。

跨气味相关变化只作 secondary descriptive report，不参与通过，不允许与 active fraction 使用 OR。S4 支持的是“所实现的 APL proxy 对稀疏性有因果作用”，不验证真实 APL gain。

### S5 B/C 拓扑贡献

对 B 和每个 C 使用同一 `T_sep`。主检验：

\[
p_{topo}=\frac{1+\#\{T_C\ge T_B\}}{40}.
\]

39 个 rewires 使 Monte Carlo 检验的最小 p 为 `1/40=0.025`，并允许在不增加模型自由度的情况下检验预注册的 0.05 门；不能在看到 B 的位置后增加 ensemble。

只有同时满足以下条件才判 `TOPOLOGY_INCREMENT_SUPPORTED_IN_PROXY`：

- `p_topo≤0.05`；
- `T_B-median(T_C)≥0.05`；
- B 自身通过 S1–S4；
- 同一方向和门槛在 M1-RT 独立重现。

若 B 通过 S1–S4 但 S5 失败，M1 机制仍可通过，拓扑优势 verdict 必须为 `NOT_DEMONSTRATED`。不得用 sparsity、cosine、最好 rewire 或学习成绩替换 `T_sep`。

## 9. 多重性与缺失值

- confirmatory primary tests 只有 S2、S4、S5；它们检验不同声明，不互相替换；每项使用上述固定 alpha。
- S1/S3 是硬区间/安全门；不对其做事后显著性包装。
- secondary metrics 全量报告但不产生通过。
- missing/zero vector 不剔除；按定义触发对应 failure 或 protocol failure。
- 不报告只包含成功 odors/seeds 的汇总。

## 10. Verdict precedence 与停止

优先级从高到低：

1. `M1_IMPLEMENTATION_FAILURE`：任一 I1–I7 失败；
2. `M1_NUMERICAL_NO_GO`：I8 失败；
3. `M1_REPRESENTATION_NO_GO`：RQ 的 S1–S4 任一失败；不解封 RT；
4. `M1_GENERALIZATION_NO_GO`：RQ 通过但 RT 的 S1–S4 任一失败；
5. `M1_MECHANISM_QUALIFIED_TOPOLOGY_NOT_DEMONSTRATED`：RQ/RT S1–S4 通过，S5 未双阶段通过；
6. `M1_MECHANISM_AND_PROXY_TOPOLOGY_INCREMENT_QUALIFIED`：RQ/RT S1–S5 全通过；
7. `M1_BUDGET_EXHAUSTED`：在任何未得出更高优先级 verdict 前达到硬预算。

失败后不增加参数点、不更换 seeds、不删 odor、不改变 active 定义、不放宽 alpha/效应量、不缩减 C ensemble。代码 bug 修复必须有独立 toy reproduction，升版本并重新密封四套集合。

## 11. 资源预算

### 已测基线

- 静态 inventory：约 2.7 s wall、401 MiB peak RSS；
- 145 PN/2,045 KC/10,732 edges toy：4 s neural time 在 `dt=0.1/0.05/0.025` 分别约 0.89/1.76/3.53 s；合计约 6.18 s，33 MiB peak RSS；
- 以上是容量数据，不是模型结果；toy 方程/seed 不得进入 formal 配置。

### M1 硬预算草案

- 实现准备与 toy smoke：累计 4 CPU-hours；
- 只读 inventory/编译：2 CPU-hours、1 GiB RAM；
- 每个正式阶段（RQ 或 RT）：24 CPU-hours；
- M1 正式总计：48 CPU-hours、4 GiB peak RAM、5 GiB artifacts、无 GPU；
- 单阶段 wall 上限：24 h；并行度不得使总 CPU-hours 越界；
- 不保存所有 time-step × KC dense state，只保存冻结的摘要、active bitsets、rate vectors、选择性诊断 traces 和 hashes。

按当前 toy 吞吐，预计 B+39C 的每个 16×10 集合约 11 单核小时，RQ+RT 约 22 单核小时；48 CPU-hours 是约 2× 的硬上限而非目标。摘要式产物预计 0.2–1 GiB，5 GiB 是硬上限；若实现要求保存逐步 dense state，必须在 formal 前改为流式摘要或停止审阅，不能超预算运行。已配置本地环境的边际服务费用记为 0，但必须报告 CPU-hours、wall、峰值 RSS 和实际字节数；不以不稳定的云单价替代资源计量。

正式申请前用最终 runner 在 BENCHMARK_ONLY fixture 上重新估算；预测超过任一硬预算则回到人工审阅，不运行 formal。

## 12. Artifact 与人工复核

每个 smoke/formal 目录必须不可覆盖，并包含：

- `m1_raw.npz`：逐 odor/repeat/network/dt 的冻结原始摘要；
- `m1_formal.json`：I/S gates、统计量、p 值、效应量、verdict；
- `m1_report.md`：不隐藏失败项的可读报告；
- `m1_manifest.json`：输入/代码/test/environment/start/end/artifact hashes；
- `launch.txt`、`stdout.txt`、资源计量；
- failure 时同样生成最小完整链并保留目录。

主代理必须独立检查进程状态、资源、trial 数、集合访问日志、artifact 完整性、hashes、所有 gate 和 verdict。M1 formal 若获准且预计超过 30 分钟，必须在 tmux 中运行。

## 13. 正式授权前检查表

- [ ] 方程、积分法和全部数值参数已冻结，来源标为 V/P/M；
- [ ] 唯一参数集，不含 grid 或输出驱动选择；
- [ ] A/B/C inventory、边界损失、39 rewires 和 hashes 已人工审阅；
- [ ] 四个 seed domains 与 sealed manifests 已建立，内容未泄露；
- [ ] I/S gates 与统计实现通过 toy oracle；
- [ ] 交易/M2 import deny-tests 通过；
- [ ] 最小 CLI smoke 生成 raw/formal/report/manifest/launch/stdout；
- [ ] 最终 runner benchmark 在预算内；
- [ ] v1 worktree/归档 hashes 未变；
- [ ] 已提交单独的 M1-RQ 正式实验授权申请。

在全部勾选并获得人工授权前，M1 formal 保持 **NO-GO**。
