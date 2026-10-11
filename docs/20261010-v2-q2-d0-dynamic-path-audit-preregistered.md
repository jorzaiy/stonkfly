# Whole-brain dynamics v2：Q2-D0 动态路径与反事实账本审计（冻结预注册）

状态：**2026-10-10 冻结；授权实现并执行 Q2-D0。D0 只是只读机制审计，不授权实现 Q2 kernel、重新校准 C0、连接市场或训练。**

Q2-D0 的目的，是在完全冻结 Q1/C0 动力学的前提下定位 `photoreceptor → lamina → downstream spiking → KC` 的首个动态传播断点，并离线回答一个窄问题：如果把已观测到的 graded voltage 按固定、无新增参数的 signed deviation 公式解释，L1/L2/L3/L5/APL 会向哪些原图 target 产生多少一阶反事实 delivery。

D0 不把反事实 delivery 反馈进模拟，不改变任何节点、边、contact、weight、release、threshold、gain、drive 或 duration。D0 不能证明候选 Q2 动力学稳定或能驱动 KC；它只为是否值得冻结 Q2 提供机制证据。

## 1. 固定输入与 SHA256

runner 启动时必须重新计算下列 SHA256；任何不一致均写 `Q2_D0_IMPLEMENTATION_FAILURE` 并停止。

| 输入 | 冻结 SHA256 |
|---|---|
| `data/graph.npz` | `346b8af85a11af13b8324e18669812c1924569e7d1adcb4e6f45cc461a2c344b` |
| `data/normalized/neurons.feather` | `2103b5203d2ba5502c06fb70a4e0b23edcbcf79ed0c344722101941fc289ba2d` |
| `data/normalized/edges.arrow` | `842b762c9427c794eaedd453178aa034727a5084850a91fdcc4d3fffe09f5b7e` |
| Q1 v7 formal JSON | `394c2cd05ecc54c979b9d871176a86bb82670c850c108fe91c63ba8354ead574` |
| Q1 kernel source | `99f63402e2a6393c229aab139915233d24dff8ad411f068d91ec965edfbef79e` |
| Q1 brain wrapper | `2673de4e7c4aa4ee0fd20636b17bf2c2828103bacfbbb05b2ff0f559979434e6` |
| Q1 binary | `9df8110f1e2134f8a6d88cde46d3a8a6278590f291ad87b49161926d2fd53f41` |
| C0 wrapper | `cb66dcec500f57d815e918b104aa92cac9aee4c21e733fbe2d608017d168759c` |
| C0 v3 calibration JSON | `62f9bc19dbe5da93e4e05bc42d02ecf597abdcaf3f5c3e59e8843bfb7b5e2307` |
| C0 v3 raw NPZ | `b00e9fb902fbcb9d16a36705565095b84ae61b48a5d6b1d0d9285498ef173173` |
| C0 v3 formal JSON | `6dd290e4758453bdc3bf52c61bd9d1fd2f28b9161b8aeffdf56db4838a6c00c3` |
| C0 v3 manifest | `0799cbd07ef8f2a768d6d2a27d742332f3659098fc5f156c0f125828362e1705` |
| C0 v3 report | `121fdc81ef7e92abfb4c67b22f96f3f6d212805c4794c3079abbdc5e24338293` |
| Q2/Opus 审阅文档 | `11c5598f6c80f04fbf7207b8f2d12bd9006783e332908adb4da750f77e2227d8` |

图规模必须仍为 166,700 nodes、25,582,938 directed edges、124,177,617 contacts。Q1 分类必须仍为 modulatory 541、photoreceptor 4,706、L1/L2/L3/L5 合计 7,114、APL 2、KC 4,064、generic spiking 150,273。完整原始 weight array 在运行前后必须 bitwise 不变。

## 2. 审计边界

- 使用冻结的 `C0HybridBrain`，`gamma=1.0`、`dt=0.05 ms`、learning off。
- graded mask 仍严格为 photoreceptor、L1/L2/L3/L5、APL 共 11,822 个节点。
- non-KC rest `-52 mV`，KC rest `-60 mV`，其余 Q1 常数不变。
- 唯一 stimulus 为 C0 的 `UNIFORM_4`：全部 4,706 photoreceptor 施加 `+4.0` current。
- 不构造 CAL/HOLD pattern，不读取市场、reward、action、P&L、旧 S2/R0 输出或 C0 holdout。
- 不修改 Q1/C0 文件；D0 使用独立 runner、tests 和输出目录。
- observer 只能读取状态；observer-on/off 的动力学边界状态必须 bitwise 等价。

## 3. D0-A：C0 诊断复现门

从 clean state 执行 C0 v3 transfer diagnostic 的精确序列：

1. 10 ms zero；
2. 10 ms `UNIFORM_4`。

必须逐项复现 C0 v3 formal 中的：source count/hash、三个 delivery integrals、六类节点的 count、voltage min/max、voltage-delta min/max、spike nodes/spikes 和五个 boolean checks。相同 binary、dt、顺序与主机下要求 float/整数逐值相等；任一不一致即 `Q2_D0_REPRODUCTION_FAILURE`，停止，不生成机制结论。

## 4. D0-B：完整 C0 trial 的 observer replay

从 clean state 执行两次相同 replay：

1. 50 ms zero baseline；
2. 100 ms `UNIFORM_4` warm-up；
3. 50 ms `UNIFORM_4` measurement；
4. 200 ms zero washout；
5. 100 ms zero post-measurement。

每次 replay 在 stimulus 的 150 ms 内按每个 `dt` 观察一次；washout/post 可按 1 ms 观察。必须保存：

- 每个节点的 baseline、warm-up end、measurement end、washout end、post end voltage；
- 每个节点 stimulus 期间首次 `|v-baseline|>1e-7`、首次非零 `g_event/g_graded`、首次 spike 的 clock；
- 六个声明层及 L1/L2/L3/L5/APL 的 voltage-delta 分布摘要、input-reached 数、spike 数；
- 每个 graded source 在 stimulus 期间的 Q1 positive release integral；
- §5 候选 C deviation 的 per-source integral；
- queue、overflow、finite/bounds、完整 weights hash 与规范化状态 hash。

两次 replay 的全部保存数组和规范化 JSON 必须 SHA256 相同。另运行一个不做逐步观察、仅按五个原始 segment 调用的 reference replay；除 segment-local `counts` 表示外，其最终动力学状态、累计 spikes、release/delivery integrals 必须与 observer replay bitwise 等价。

## 5. 固定的一阶反事实公式

D0 只离线计算 Opus 建议的候选 C，不把结果送回 kernel：

`delta_r_i(t) = clip((v_i(t) - V_rest_i) / 7 mV, -1, 1)`

`Q_i = sum_t delta_r_i(t) * dt / 2.2 ms`

对每条原图 graded-source edge：

`counterfactual_delivery_ij = w_ij * Q_i`

`7 mV = V_th - V_rest`，不是从 KC/MBON 输出拟合。`delta_r<0` 只表示低于静息基线的工程化 release deviation，不称为负递质释放。候选 C 在 `v>=rest` 时与 Q1 clamp 相同。

该账本是一阶、open-loop 反事实：它使用 Q1 轨迹，不包含 deviation 对网络状态的反馈、循环、稳定性或后续 spike。因此任何结果都不得描述为“Q2 会产生的 KC/MBON 活动”。

## 6. D0-C：分层与逐跳报告

### 6.1 Source-class × target-class 账本

必须分别报告 photoreceptor、L1、L2、L3、L5、APL；不得把四类 lamina 合并后代替细分结果。每个 source class 保存：

- source node 数、递质声明、正/负/零原图出边数及 signed/absolute weight sum；
- stimulus 期间低于/高于 rest 的 source 数、最大幅度、持续步数；
- Q1 positive release 与候选 C signed deviation integral 的分布；
- 对六个冻结 target layer 分别计算正、负、signed、absolute counterfactual delivery；
- 收到非零 absolute delivery、正 net delivery、负 net delivery 的 target node 数与 source-id；
- 对 generic spiking、KC、MBON07、MBON11 单独报告，不以动作方向解释。

raw NPZ 必须保存 per-source `Q_i`、per-target signed/absolute counterfactual totals，以及用于复算的 source-id、layer、cell-type code、first-input/first-spike arrays。JSON 不能只存均值。

### 6.2 静态 hop 与动态传播

在完整 retained graph 上从 4,706 个 photoreceptor 做无权有向 BFS，冻结 shortest hop 为 `0..6`，其余为 `>6/unreached`。每个 hop 报告：

- 静态 node 数及声明层构成；
- Q1 replay 中 input-reached node 数、voltage-changed node 数、spiking node 数；
- graded node 的 Q1 nonzero release 数与候选 C nonzero deviation 数；
- 首次 input/voltage/spike clock 分布。

同时沿 lamina 原图出边单独报告：当前 Q1 delivery 是否因 lamina `r=0` 为零；候选 C 是否向至少一个 spiking downstream 产生非零 absolute delivery；其中 positive/negative net target 各多少。存在 photoreceptor→generic 的旁路或 generic 收到输入但不 spike 时，必须标为 competing bottleneck，不得把全部沉寂归因于 lamina。

### 6.3 APL 专项

报告 APL 的原图入边按 source class/sign/contact 分解、stimulus 期实际 voltage delta、Q1 release、候选 C deviation，以及其反事实 KC delivery。APL 未低于 rest 时必须明确写“本次刺激下无影响”，不得因理论去抑制而宣称已观察到。

## 7. 预注册解释标志

D0 formal 只允许以下机器判定：

1. `Q2_D0_IMPLEMENTATION_FAILURE`：pin/schema/count/hash/artifact/observer/weight/finite/queue 任一失败；
2. `Q2_D0_REPRODUCTION_FAILURE`：§3 未精确复现 C0 v3；
3. `Q2_D0_AUDIT_COMPLETE`：全部复现、确定性和产物门通过。

`Q2_D0_AUDIT_COMPLETE` 不是模型通过。它必须附带以下预注册 boolean，不另造成功 verdict：

- `lamina_hyperpolarization_observed`；
- `lamina_q1_output_clamped_zero`；
- `counterfactual_lamina_reaches_spiking_downstream`；
- `counterfactual_lamina_positive_net_reaches_spiking_downstream`；
- `counterfactual_lamina_reaches_kc_directly`；
- `generic_spiking_received_input`；
- `generic_spiking_emitted_spikes`；
- `competing_spiking_bottleneck`，定义为 generic spiking 收到动态输入但没有 spike；
- `tonic_hypothesis_upgrade_condition`，定义为前三项均真；它只允许把候选 C 提升为 Q2 预注册候选，不表示 Q2 会驱动 KC。

若 upgrade condition 为假，停止 tonic redesign 并根据实际断点重新评审。若为真但 competing bottleneck 也为真，Q2 只能以修正 graded 语义立项，必须预先写明它可能仍无法驱动 KC；不得顺手扩大 graded mask 或降低 spiking threshold。

## 8. 产物与完整性

正式目录固定为：

`runs/whole_brain_v2/Q2/D0/20261010T-dynamic-path-audit-v1/`

至少生成：

- `q2_d0_raw.npz`；
- `q2_d0_formal.json`；
- `q2_d0_report.md`；
- `q2_d0_manifest.json`；
- `launch.txt` 与 `runner.stdout.log`。

manifest 必须包含所有非自引用产物 SHA256、固定输入 pins、Python/platform、Q1 binary/compiler 信息，以及 prereg/runner/tests 的 start/end hashes。正式运行开始后 runner、tests、prereg、Q1/C0 source/binary 不得变化。输出目录已存在时 runner 必须拒绝覆盖。

正式运行前必须以小型合成图完成端到端 CLI smoke，实际生成 raw/formal/report/manifest，并测试：公式四象限、L1/L2 符号方向、target 聚合、BFS、判定优先级、失败产物、输出保护和 observer/reference 等价。正式长跑使用 tmux；主代理独立检查进程、日志、产物、hash、gates 和 verdict。

## 9. 后续边界

D0 完成后先把 formal/report 交给 Opus 审阅。只有 upgrade condition 与审阅均支持时，才另写 Q2 kernel qualification 预注册。未来 Q2 至少还必须验证线性化动态稳定性、±epsilon 初态微扰、撤刺激回落、负向 photoreceptor drive、符号置换阴性对照、独立 dense reference 和 graded 语义不泄漏。

在新的 Q2 qualification 与全新 synthetic calibration/holdout 通过前，sensory adapter、plasticity、S3 和市场训练继续阻塞。
