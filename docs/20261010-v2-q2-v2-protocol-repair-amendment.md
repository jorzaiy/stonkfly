# V2-Q2 v2 protocol-repair replication amendment（冻结）

状态：**2026-10-10 pre-formal smoke 后澄清并重新冻结；授权只修复 Q2-A 证据链与 B1 fixture/reporting，然后在新目录进行一次 protocol-repair replication。候选 C、Q2 kernel、图、Wgg、B1 数值设置和停止条件全部不变。**

## 1. 修复原因与 v1 地位

Q2 v1 目录原样保留：

`runs/whole_brain_v2/Q2/20261010T-bidirectional-graded-kernel-v1/`

v1 runner 自报 `Q2_LINEAR_INSTABILITY`，但 A2–A7 缺少冻结预注册要求的双重复、完整独立 dense float64 数值 reference 和两 hop 首影响 clock。按原预注册第 7 节优先级，v1 的正式可接纳状态为 `Q2_IMPLEMENTATION_FAILURE`。v1 B1 只保留为诊断证据。

本 amendment 不允许借修复改变 B1 或候选机制。Q2 v1 结果已经可见；因此 v2 必须证明 B1 输入与 Hessenberg 输出逐位未变，排除结果后适配。

固定审阅输入：

| 输入 | SHA256 |
|---|---|
| 原 Q2 预注册 | `dc4661734f88fcbbbdc1e98f8391ca149b897795fd5d46b4d34a2e044bc19c94` |
| Q2 v1 raw | `28c0eb9e3d2f6edbc88dc0521ed503e87cb7a424075974c77130c1f0f5c63b68` |
| Q2 v1 formal | `fab03dbccf54b0835ab2791ff04a02ea8f1752a53cf8514114be8281e0935092` |
| Q2 v1 manifest | `b3ab50d46b908bf27f252aafd3c4ea237d647ef26edb0faf3adcc4adbf035429` |
| Q2 结果与 Opus 审阅 | `11ea7d56c3b6fb29c3868edd3ff0381b96371dd750e410a460f1b0ac15794116` |

原预注册的 graph、neurons、edges、Q1、D0 pins 继续全部生效。

## 2. 绝对禁止改变的项目

- 候选 C 仍为 `clip((v-rest)/7mV,-1,1)`；不得改变 span、clamp、gain、tau、delay、threshold、rest 或 mask。
- `stonkfly/neural/v2_q2_kernel.cpp` 必须保持 SHA256 `5a578df80129358ddc1aaa15d2f9495b76210c404e639e52a3db124dd920457e`。
- Q2 binary 必须保持 SHA256 `74faea7acb5772d741bff996481c3c46a483da4fd3d4c16c50bcf8277b617a42`。
- `stonkfly/neural/v2_q2_brain.py` 必须保持 SHA256 `8cf97bbf92571c4af73b28f104cc963b0830ad0ec8b55e62ddf7ff8faf938f08`。
- Arnoldi helper source/binary 必须分别保持 `3fc5c493094b8f1954ded5949821bfeb12002d38fcf67a640eb29e23f8eb0d88` 与 `09bc25e3826704afe23a64d51db946897626aac1660794a32d3c8eb8c6ae76c9`。
- 不修改、剪枝、重排、筛选或重标 MaleCNS v1.0 图、节点、边、contacts 或 weights。
- 不运行 B2/C、calibration、KC/MBON、market、reward、plasticity、decoder 或 execution，除非 B1 没有不稳定模态；而本 replication 不允许通过改变 B1 得到这种结果。

## 3. 冻结 Wgg 与 B1 replication lock

Wgg 仍为 11,822 个 graded nodes、38,018 条 graded→graded 原图边，source-major、target stable order。规范化 dtype 为 ptr `int64`、post `int32`、weight `float64`。

| 项目 | SHA256 |
|---|---|
| graded global indices | `04d1eed8b76a69836605459866cb5770350fb4dae56df600086a08a310b5ef39` |
| Wgg ptr bytes | `430bf75068abaaf2a54743bb060921b9678c9b0ebe8687c46723a423d25796f9` |
| Wgg post bytes | `e43ac82774b5d88c07098b9a865b78c92991b403e4693697cc66dad3fcfc829a` |
| Wgg float64 weight bytes | `9656a0a9a79a27ec57d1852484bbcb0016f264fcbb2b4a44ed1718e0cacb7a6e` |
| canonical CSR schema+shape+bytes | `c40fddad5ec32314eaf39e244e55f8eb993a94cc43c3f485029d7827846875aa` |

B1 继续使用：

- 起点 `sin(i+1)` 与 PCG64 seed `20261010` Rademacher；
- Krylov dimension 160；
- C++ double、两遍 modified Gram–Schmidt；
- residual `<=1e-6*max(1,|lambda|)`；
- `(20s+1)(5s+1)=5lambda/(7*2.2)`；
- 两起点取并集，只问是否存在 converged unstable mode。

v2 生成的 Hessenberg 矩阵必须与 v1 raw 逐位相等：

| array | shape/dtype | v1 byte SHA256 |
|---|---|---|
| `sin_h` | `160×160 float64` | `51327817610d1c17abd98c3777ade08b3e2e7d5e5ea50e0041371f05c6887147` |
| `rademacher_h` | `160×160 float64` | `5dc832551f0da1e853e71a988421c909bbbbdb83bc22ed6edc3f490617cffc9f` |

任一 Wgg 或 H hash 不一致立即判 `Q2_IMPLEMENTATION_FAILURE`，不得继续解释新谱结果。

## 4. 唯一允许的 A 门修复

原 Q2-A 的机制、fixture 符号、dt 和容差不变。v2 必须补齐：

1. `dt=0.1/0.05/0.025ms` 下 A1–A7 各两次独立 clean-start 重复；保存每次 state/trace hash，两次必须相等。
2. 新增不调用 C++ kernel、不导入 Q2 wrapper 的独立 graded-only dense float64 reference。A1–A5 的预期 trajectory 由它计算。
3. A2 四象限逐 edge 保存 release deviation、delivery、arrival clock、`graded_drive`、`g_graded` 与 voltage；kernel/reference 最大误差 `<=1e-4`。
4. A3 四种两 hop sign product 保存第一 hop 与第二 hop 的首次非零 input/voltage clock，并与 reference 精确相同；状态误差 `<=1e-4`。
5. A4 正负 external drive、A5 sign permutation 同样与 reference 比较完整 trace，而不只比较终点方向。
6. A6 继续证明 graded mask 外阈下单元不传 deviation，并与冻结 Q1 spiking 行为比较。
7. A7 empty mask 与冻结 v1/Q1 完整状态 bitwise；正半轴 fixture 与 Q1 完整状态、ledger、queue 和 counts bitwise。
8. 测试必须显式断言每个 A gate 有三个 dt、每个 dt 两次、reference 字段和 trace hash；缺任一字段即失败。

独立 dense reference 只允许用于微回路期望值，不得用于完整图或 B1，不得读取 v1 B1 结果。

## 5. B1 fixture 修复

除原 stable/unstable diagonal fixture 外，新增：

- 一个具有已知复共轭特征值的三节点负反馈环 fixture；
- 一个具有已知特征值但明显非正规的上三角或相似矩阵 fixture。

fixture 必须验证 eigenvalue、residual 与 stable/unstable classification。它们只验证 audit helper，不能改变 Wgg 或 B1 门。

报告按起点分别列 converged/unstable 数量；不得相加并声称为唯一 mode 数。正式门只看任一起点是否存在至少一个 converged `Re(s)>=0` mode。

## 6. 输出、判定与停止

新目录：

`runs/whole_brain_v2/Q2/20261010T-bidirectional-graded-kernel-v2-protocol-repair/`

报告必须明确标记：`protocol_repair_replication: true`、`b1_result_known_before_run: true`、`production_eligible: false`。

判定优先级沿用原预注册：

1. 任一 pin、A coverage、reference、fixture、Wgg/H lock、hash stability 或产物缺失：`Q2_IMPLEMENTATION_FAILURE`；
2. A 数值/语义失败：`Q2_ANALYTIC_FAILURE`；
3. B1 无 converged mode：`Q2_LINEAR_STABILITY_INCONCLUSIVE`；
4. B1 存在 converged unstable mode：`Q2_LINEAR_INSTABILITY`，立即停止；
5. 只有 B1 未发现不稳定 mode 才需要回到原预注册 B2/C；这种与 v1 H 不一致的情况会先触发 implementation failure，因此本 replication 不得自行进入 B2/C。

若得到合格的 `Q2_LINEAR_INSTABILITY`：正式终止候选 C 和当前 whole-brain visual graded 路线；不追加 damping、gain、APL、反转电位、threshold、tau_emit 或删边补救。随后另写 stop/scope 文档，交用户选择更换 neuron/synapse model 或改用非视觉感觉入口。

## 7. audit-only 描述边界

Opus 的 SCC 稠密复核提示 L1↔L5 高增益负反馈导致 Hopf 型振荡失稳；该结果在本 amendment 中只作描述，不另设门。可以在最终 stop/scope 文档引用，但不得用来删 SCC、边或选择缩放系数。

正式运行前必须由主代理审核：代码差异仅限新 dense reference、v2 runner/tests/report；自动测试覆盖 prereg/runner 一致性；真实 CLI smoke 生成 raw/formal/report/manifest/launch/stdout 全链；固定 Q2 kernel/wrapper/helper hashes 不变。长跑使用 tmux。

## 8. Pre-formal smoke 澄清（正式运行前冻结）

首版 amendment SHA256 为 `ecce4ffc032248f94ef6b2739d4693fa89728f908a23fd497a3bdf44bb6b4bb4`。它只用于临时 smoke，未启动正式目录。smoke 暴露两个 fixture/观测定义问题；本节在任何正式运行前公开澄清，不改变 `1e-4` 数值门、机制或 B1。

1. dense float64 reference 会在 target voltage 仍小于一个 float32 ULP 时记录非零，而 float32 kernel 只能在累积到可表示量后记录非零。因此“两 hop 首个影响 clock”冻结为动力学的结构 clock：首次非零 `graded_drive` clock，以及按 kernel 步序紧随其后的 voltage-update clock。两者必须由 kernel queue/input trace 与 reference 精确一致。另行保存 kernel/reference 的“首次超过 `1e-7mV` 的可表示 voltage clock”，允许它们因 float32 量化不同；完整 trajectory 仍必须通过原定 `<=1e-4` 门。
2. A6 的 graded→graded Q1-positive regression fixture 必须真正满足“所有 graded nodes 始终 `v>=rest`”。首版 fixture 的 target 在正输入开始时因 float32 解析式抵消短暂落到 `rest-1 ULP`，违反 fixture 前提并触发候选 C 的负 deviation。修复仅把该 fixture 的 target 初态设为 `rest+1mV` 并显式激活，使比较窗口始终位于正半轴；不得放宽 bitwise Q1 equality。

本澄清后的 amendment SHA256 才是正式 replication 的固定输入，runner/manifest 必须记录其 start/end 一致性。
