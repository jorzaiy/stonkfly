# Whole-brain dynamics v2：P0 模型清单与可辨识性审计（冻结预注册）

状态：**2026-10-08 冻结；允许执行 V2-P0，尚未授权实现或运行 v2 动力学内核。**

V2-P0 是从已停止的 S2/R0 分支切换到 whole-brain dynamics model v2 的第一项资格实验。它只做全图结构、注释覆盖和历史证据审计，不使用市场数据、收益、动作或标签，不改变神经动力学，也不产生交易。

## 1. 为什么先做 P0

S2–R0c 已经说明，当前统一 LIF 模型中的 KC 活动具有硬跃迁、非单调、初态敏感和撤刺激后持续高活动；该现象在 0.05/0.025 ms 下仍存在。继续调感觉增益、KC adaptation、APL 倍率或局部时序已被正式停止。

v2 的核心问题因此不是“哪个旋钮能让旧 60 帧变好”，而是：

1. 哪些细胞有足够明确的类型声明，可进入 graded/non-spiking 候选集合；
2. 这些候选单元经完整 retained graph 如何连接到 KC、APL 和 MBON07/11；
3. 粗粒度递质符号和缺失注释在关键通路中占多大比例；
4. 是否有足够的结构可辨识性，允许下一阶段只实现一个清楚、可归因的 hybrid-dynamics 机制族。

P0 不会因为某种划分“看起来能修好 R0”而选择参数。S2/R0/R0b/R0c 只用于冻结失败形态、资格门槛和禁止事项。

## 2. 固定输入与权威证据

执行程序必须在启动时重新计算以下 SHA256；任一不一致立即停止并写失败产物：

| 输入 | 冻结 SHA256 |
|---|---|
| `data/graph.npz` | `346b8af85a11af13b8324e18669812c1924569e7d1adcb4e6f45cc461a2c344b` |
| `data/normalized/neurons.feather` | `2103b5203d2ba5502c06fb70a4e0b23edcbcf79ed0c344722101941fc289ba2d` |
| `data/normalized/edges.arrow` | `842b762c9427c794eaedd453178aa034727a5084850a91fdcc4d3fffe09f5b7e` |
| `stonkfly/neural/kernel.cpp`（v1 参考） | `aea2b97d2f9d78c57787f40f42a9d7dd777c3059d8c7f453e61c4d94398438d4` |
| R0/R0b/Opus 结果汇总 | `fae16a8946299356976397f2925c429d4a81b2ac2526e4386e6658a6bd69b2f0` |
| R0c 预注册 | `a4d6baeedf7944aa1968d678d80a47c17dd4ecb50cb48d60aa927f8d65efb9f0` |
| R0c 正式 JSON | `b531533ae1a401afebeae5317e4a1ef6b963e1afedd1ad2018dd071ecc0dec51` |
| 模型范围与停止审查 | `47396caeca7b4b57c0c1db4e12c1e053d774901e95a3dec3e2c20090ce42284e` |

图规模也必须严格复核为：166,700 个 retained nodes、25,582,938 条 directed edges、124,177,617 个 synaptic contacts。不得丢弃弱边、自连接或不确定符号边。

## 3. 冻结的节点声明层

P0 依据 `cell_type` 和 `neurotransmitter` 的原始注释建立互斥的声明层。它不是最终生理参数表，也不得改变边权。

优先级从高到低如下：

1. `modulatory_declared`：递质严格为 dopamine、octopamine 或 serotonin；仍保留全部边，但后续效应必须单独声明，不能默认当作普通快突触。
2. `graded_candidate_photoreceptor`：`cell_type == "R1-R6"` 或以 `R8` 开头。
3. `graded_candidate_lamina`：`cell_type` 严格为 `L1`、`L2`、`L3` 或 `L5`。
4. `graded_candidate_apl`：`cell_type == "APL"`。
5. `adaptive_spiking_kc`：`cell_type` 以 `KC` 开头。
6. `generic_spiking_declared`：其余 retained neurons。

同一节点必须且只能进入一个主声明层。另行记录 `uncertain_fast_sign` 布尔标志：递质为缺失、`unclear`、仅调制递质、冲突共递质，或无法得到唯一快符号时为真。该标志不删除边，也不把边权归零。

冻结前的只读清点值仅用于核对规则实现：R1–R6 3,377；全部 R8 前缀 1,329；L1/L2/L3/L5 分别 1,776/1,779/1,772/1,787；APL 2；KC 前缀 4,064；MBON07 4；MBON11 2。正式结果仍必须从固定输入重新计算。

## 4. 唯一允许的 P0 计算

### 4.1 完整性与分层清单

- 每个声明层的节点数、细胞类型频数、递质频数和 source-id SHA256；
- 每个节点只分配一次，所有 retained nodes 全覆盖；
- 逐层统计出/入 edge 数、contact 数、基线 signed weight 和 absolute weight；
- 统计 self-edge，不得排除。

### 4.2 声明层之间的完整流量矩阵

对每个有向 `source_layer → target_layer` 组合统计：

- edge count；
- synaptic contact count；
- positive/negative/uncertain-sign edge count；
- signed 和 absolute baseline weight；
- uncertain-sign absolute-weight fraction。

所有矩阵的边数、contact 数和权重总量必须与图数组逐项守恒；浮点汇总使用 float64。

### 4.3 关键靶点的输入可辨识性

分别对以下目标集合计算输入来源分解：全部 KC、两只 APL、MBON07、MBON11，以及中央脑 `superclass` 集合（只报告，不把它等同于全脑）：

- 每个来源声明层的 edge/contact/absolute-weight 占比；
- 正、负、不确定符号的 absolute-weight 占比；
- 直接上游 unique neuron 数；
- 前 20 个 source `cell_type` 的 absolute-weight 占比，仅作描述，不用于删边或挑参数。

### 4.4 保留路径覆盖

在不删边、不看权重正负和不设最小 contact 阈值的条件下，计算从三组 graded candidate 分别到 KC、APL、MBON07/11 的最短有向 hop 分布，最大记录到 6 hops；超过 6 记为 `>6/unreached`。必须使用完整 retained edge set。

另计算：

- graded candidates 可到达的 KC 比例；
- 可到达 KC 中进一步可到达 MBON07/11 的比例；
- APL 是否处在这些路径上只作结构描述，不解释为因果控制。

### 4.5 历史证据锁定

程序只读解析正式 R0、R0b、R0c JSON，并把下列值原样写入 P0 结果，作为 v2 后续不能悄悄改写的基线：

- R0 正式判定 `inconclusive`；
- R0b 正式判定 `inconclusive`；
- R0c 判定 `DT_CONVERGED_FOR_CURRENT_AUDIT`；
- R0c 三个 dt 的经验首达风险率、2 秒点燃比例、点燃后 CB/KC 同时持续比例；
- 明确标注这些值不证明随机过程、生物吸收态、视觉编码或学习。

不得重新跑、合并或挑选历史 trial。

## 5. 资格门禁

### 5.1 实现门禁（任一失败即 `P0_IMPLEMENTATION_FAILURE`）

- 所有冻结 SHA256 和图规模一致；
- 166,700 个节点恰好分配一次；
- 25,582,938 条边及 124,177,617 contacts 在全部汇总中守恒；
- CSR 与 normalized edge table 的 source/post/contact 对齐抽查及全量聚合一致；
- 所有比例有限且位于 `[0,1]`；
- 相同输入连续运行两次，规范化 JSON（排除 wall time 与输出路径）SHA256 一致；
- 源图、权重、内核和历史正式产物运行前后 SHA256 不变。

### 5.2 模型资格门禁

P0 只可产生下列三个判定之一：

1. `READY_FOR_V2_Q1_WITH_SIGN_SENSITIVITY`
   - 实现门禁全部通过；
   - 三组 graded candidate 均非空；
   - graded candidates 在 6 hops 内可到达至少 95% 的 KC；
   - KC 经 retained paths 在 6 hops 内可到达 MBON07 与 MBON11；
   - 任一关键靶点集合（KC、APL、MBON07、MBON11）的 uncertain-sign absolute-weight fraction **达到或超过 0.20**。下一阶段必须预注册 ambiguous-sign `+/-` 灵敏度臂，不能选较好结果作为生产值。
2. `READY_FOR_V2_Q1`
   - 满足上述覆盖条件；
   - 四个关键靶点的 uncertain-sign absolute-weight fraction 均小于 0.20。
3. `BLOCKED_BY_MODEL_IDENTIFIABILITY`
   - 实现门禁通过，但 graded candidate 为空、6-hop KC 覆盖低于 95%，或 KC 不能在 6 hops 内到达任一目标 MBON 类型。

0.20 是预先冻结的“符号不确定性足以支配局部净驱动”的保守触发线，不表示低于 0.20 就具有受体级生理真实性。

## 6. P0 明确不做什么

- 不改 `kernel.cpp`、`brain.py`、`graph.npz` 或任何权重；
- 不实现 graded dynamics、cell-type 参数、噪声、适应或 APL 连续抑制；
- 不运行 RGB、市场帧、旧 60 帧、reward、plasticity 或交易解码；
- 不以 active-KC、MBON、动作或收益选择声明层；
- 不把结构可达性解释为功能驱动、感觉编码或行为因果；
- 不把 `READY` 解释为 v2 模型已验证。

## 7. 产物与执行约束

正式目录固定为：

`runs/whole_brain_v2/P0/20261008T-model-inventory-v1/`

至少包含：

- `v2_p0_formal.json`：判定、门禁、固定输入哈希、层计数、关键比例、历史证据锁定；
- `v2_p0_matrices.npz`：完整层间矩阵、路径分布及必要索引；
- `v2_p0_report.md`：人可读、只陈述结果和限制；
- `formal.log` 与 `launch.txt`；
- 两次确定性运行的规范化摘要哈希。

大图遍历由 Luna 子代理执行。若超过交互时限，必须在 tmux 中运行；主代理只审核代码、测试、产物哈希和判定。失败产物不得覆盖，修复纯实现错误时必须新建 `v2` 目录并追加 amendment。

## 8. P0 之后的唯一合法下一步

若 P0 为 `READY_FOR_V2_Q1*`，下一步先单独冻结 **V2-Q1 hybrid-kernel qualification**，再允许修改动力学代码。Q1 只允许一个机制族：将本预注册列出的明确 graded candidates 改为显式 graded-output 单元，同时保留所有节点、边和 contacts；它必须先过微回路解析测试、dt/事件队列/不应期语义、零输入稳定性和 full-graph smoke，不能接触市场数据。

若 P0 为 `BLOCKED_BY_MODEL_IDENTIFIABILITY`，停止 v2 动力学实现并回到模型声明评审；不得通过放宽类型规则、延长 hop 直到通过或读取旧 KC 结果来补救。

只有未来 Q1、合成感觉 calibration、合成 holdout、KC→MBON 通路和持续状态门禁逐级通过后，才可讨论孤立塑性；纸面市场 replay 和固定交易解码仍排在最后。
