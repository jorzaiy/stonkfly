# Whole-brain dynamics v2：Q1 hybrid-kernel qualification（冻结预注册）

状态：**2026-10-09 冻结；允许实现并执行 V2-Q1。Q1 不是动力学校准，不接触市场、交易、reward 或 plasticity。**

Q1 的唯一目的，是验证一个同时容纳明确 graded candidates 与原有 spiking neurons 的混合内核能否被明确实现、数值复现，并在完整 retained MaleCNS v1.0 图上安全短跑。Q1 不以 KC 稀疏度、MBON 响应或旧 R0 结果选择参数；即使通过，也只能进入下一阶段的 synthetic calibration 预注册。

## 1. 前置资格与固定输入

V2-P0 正式判定为 `READY_FOR_V2_Q1`。Q1 启动时必须重新计算以下 SHA256；任一不一致即写失败产物并停止：

| 输入 | 冻结 SHA256 |
|---|---|
| `data/graph.npz` | `346b8af85a11af13b8324e18669812c1924569e7d1adcb4e6f45cc461a2c344b` |
| `data/normalized/neurons.feather` | `2103b5203d2ba5502c06fb70a4e0b23edcbcf79ed0c344722101941fc289ba2d` |
| `data/normalized/edges.arrow` | `842b762c9427c794eaedd453178aa034727a5084850a91fdcc4d3fffe09f5b7e` |
| v1 `stonkfly/neural/kernel.cpp` | `aea2b97d2f9d78c57787f40f42a9d7dd777c3059d8c7f453e61c4d94398438d4` |
| v1 `stonkfly/neural/brain.py` | `753b04324c274c9b522a0aa2459a81edc1056e4c8f6af34c332b3df371ace4b4` |
| v1 `stonkfly/neural/state.py` | `4356965ea5fe70ef3ab323b3b05753dde4cf4ef7788506ef38eb29609955bbee` |
| P0 formal JSON | `7fa88023cc09a12955174ab76f8cebb5a05ddf9c258867511187440bad69b45f` |
| P0 matrices NPZ | `bf1ff3298886ac274337f9c7f3c89cace57158a504a643ab2c8c19269bcbe0eb` |
| P0 预注册 | `3d845207b394b67646470d16894a37c1afb2a61365e602135d7a4c2f88d2cc18` |
| P0 v1/v2 amendment | `8f05ba4f2dbfd43a6c052068473b1278f111e59e45408ab4998472f02082afb7` |

图规模必须仍为 166,700 nodes、25,582,938 directed edges、124,177,617 contacts。实现不得修改、删除、屏蔽或重新缩放任何原图节点、边、contact count 或基线权重。

## 2. 冻结的节点分类

逐节点应用 P0 已冻结的互斥优先级，禁止根据 Q1 活动重新分类：

1. `modulatory_declared`：递质严格为 dopamine、octopamine、serotonin；保留 v1 调制语义，不进入 graded 集合。
2. `graded_candidate_photoreceptor`：`cell_type == "R1-R6"` 或以 `R8` 开头。
3. `graded_candidate_lamina`：`cell_type` 为 `L1/L2/L3/L5`。
4. `graded_candidate_apl`：`cell_type == "APL"`。
5. `adaptive_spiking_kc`：`cell_type` 以 `KC` 开头。
6. `generic_spiking_declared`：其余 retained neurons。

Q1 的 graded mask 是第 2–4 类的并集，必须为 11,822 个节点；其余 154,878 个节点保持原有 spiking/modulatory 声明。三类 graded 子集必须分别为 4,706、7,114、2 个节点，source-id 集合 SHA256 必须与 P0 formal JSON 一致。

## 3. 唯一允许的新机制族

Q1 只允许增加一个机制族：**明确 graded candidates 不再阈值放电，而以连续 release 经其全部原有出边传递；其他细胞的 v1 动力学不变。**

必须新增隔离实现，例如 `stonkfly/neural/v2_kernel.cpp` 与 `stonkfly/neural/v2_brain.py`。不得修改上述已固定的 v1 kernel、brain、state 文件。Q1 通过前，现有 controller、visual adapter、plasticity、decoder 和 execution 路径不得默认使用 v2。

### 3.1 共同常数

以下数值沿用 v1，Q1 不扫描：

| 常数 | 数值 |
|---|---:|
| membrane time constant `tau_m` | 20 ms |
| synaptic time constant `tau_s` | 5 ms |
| reference threshold `V_th` | -45 mV |
| non-KC rest `V_rest` | -52 mV |
| transmission delay | 1.8 ms |
| spiking refractory | 2.2 ms |
| graded release unit `tau_emit` | 2.2 ms |
| 正式 dt | 0.1、0.05、0.025 ms |

`tau_emit=2.2 ms` 是工程化单位定义：release=1 对应 v1 单元在 2.2 ms refractory 限制下的最大事件频率。它不是已验证的 graded 生理参数，不得在 Q1 后直接称为生产值；未来若要改变，只能在独立 calibration 预注册中基于外部证据或独立 synthetic calibration 冻结。

### 3.2 Graded 单元状态与 release

每个 graded node 保留连续膜电位 `v_i`、事件突触状态 `g_event_i` 和 graded 突触状态 `g_graded_i`，但没有 spike threshold、spike count、reset、refractory 或 KC adaptation。

冻结 release 函数：

`r_i(t) = clip((v_i(t) - V_rest_i) / (V_th - V_rest_i), 0, 1)`

其中 Q1 的所有 graded candidates 均使用其既有 non-KC rest `-52 mV`。`r=0` 表示不释放，`r=1` 只表示上述工程化最大 release 单位，不表示真实递质饱和。

### 3.3 连续传递

对于任意现有 graded source `i` 到 target `j` 的边，使用原始 signed weight `w_ij`，在 1.8 ms 延迟后施加分段常数输入率：

`u_j(t + 1.8 ms) += w_ij * r_i(t) / tau_emit`

每个 dt 步对应的传递量为 `w_ij * r_i(t) * dt / tau_emit`。不得按边强度筛选，不得聚合后删掉单边身份，不得改变不确定符号边的 P0 正号 fallback。实现可用 target-indexed float64/float32 ring buffer 加速，但审计账本必须能复算每一声明层的 signed/absolute delivery 总量。

### 3.4 连续状态的精确分段更新

在一个 dt 内，把外部 drive `I` 和延迟后到达的 graded rate `u` 视为常数。令：

- `a = exp(-dt/tau_m)`；
- `b = exp(-dt/tau_s)`；
- `g_ss = u * tau_s`。

graded conductance 的冻结更新为：

`g_graded' = g_ss + (g_graded - g_ss) * b`

其对膜电位的贡献必须使用同一线性系统的解析分段解：

`v' = V_rest + I + g_ss + (v - V_rest - I - g_ss) * a + (g_graded - g_ss) * tau_s/(tau_m-tau_s) * (a-b)`

原有离散 spike event 对 `g_event` 的指数衰减及其膜电位贡献沿用 v1 解析式。实现可以合并代数，但微回路输出必须与上述分离定义相符。

### 3.5 四类连接与 refractory 语义

| source → target | 冻结语义 |
|---|---|
| spiking → spiking | 与 v1 bitwise 等价；target refractory 时到达的离散事件仍按 v1 丢弃 |
| spiking → graded | 保留全部原边；事件进入 `g_event`，graded target 没有 refractory，故不丢弃 |
| graded → graded | 连续 release，经 1.8 ms ring buffer 进入 `g_graded` |
| graded → spiking | 连续 release 在 target refractory 期间仍进入并衰减于独立 `g_graded`；膜电位在 refractory 期间保持 v1 clamp/reset 语义，解除后受剩余 `g_graded` 影响 |

spiking target 发放时只执行 v1 对膜电位和 `g_event` 的 reset；不得清零 `g_graded`。这是 hybrid 接口的显式语义，不得依据 KC 输出选择是否保留。

### 3.6 延迟和步序

`delay_steps = round(1.8/dt)`，三档 dt 必须分别为 18/36/72；`refractory_steps = round(2.2/dt)`，分别为 22/44/88。

每步固定顺序：

1. 以本步开始时状态解析演化到当前 clock；
2. 判定 spiking nodes 是否发放，并排入离散 spike delay queue；
3. 从演化后的 graded voltage 计算 release，按全部出边排入 graded delay queue；
4. 处理当前 slot 的离散事件和 graded rate，使其从下一膜状态区间起生效；
5. 推进 clock。

测试必须明确记录 release sample clock、arrival clock 和首次影响 target voltage 的 clock，禁止用容许 ±1 step 的模糊门槛掩盖顺序错误。

## 4. Q1-A：微回路解析资格

微回路使用独立小数组，不读取市场或 MaleCNS 输出，不选择参数。每项在三个 dt 下运行两次。

### A1. Release transfer

固定 graded source voltage 对应 `r={0,0.25,0.5,0.75,1}`，包含低于 rest 和高于 threshold 的 clamp case。计算值与冻结公式最大绝对误差必须 `<=1e-7`。

### A2. Delay 与全边传递

单 graded source 分别连接两个正权和两个负权 target，包含 self-edge。要求：

- 1.8 ms 前 delivery 严格为零；
- arrival clock 与 §3.6 完全一致；
- 每条边的 signed delivery 与 `w*r*dt/tau_emit` 相符；
- self-edge 和弱边均存在且参与。

### A3. 解析 excitatory/inhibitory response

对固定 `u` 的正、负单边回路，逐步比较 `g_graded`、`v` 与 §3.4 解析递推。float64 harness 的最大绝对误差 `<=1e-10`；正式内核 float32/混合精度路径的绝对或相对误差 `<=1e-4`。

### A4. 四种 source/target 组合

构造 spiking→spiking、spiking→graded、graded→graded、graded→spiking 四条互不混淆的边，验证 §3.5 的事件数量、arrival、符号和 target 状态。不得以聚合总量代替逐边断言。

### A5. Refractory audit

让 graded input 覆盖 spiking target 的完整 refractory 窗口。要求 `g_graded` 在该窗口仍按解析式积累/衰减，target voltage 保持冻结语义，解除后的首步响应与解析值误差 `<=1e-4`。另设离散 spike input 对照，确认 v1 的 refractory-drop 语义未改变。

### A6. Zero-input equilibrium

从 rest、零 conductance、零 queue 开始运行 100 ms。所有 graded voltage 必须逐 bit 保持 rest，release/delivery/spike 均为零；三档 dt 均如此。

### A7. Fine-dt convergence

使用同一 40 ms 分段常数 graded stimulus，比较 0.05 与 0.025 ms：最终 `v`、最终 `g_graded`、release integral、signed delivery integral 的对称相对差均 `<=0.02`。0.1 ms 只报告，不参与该门槛。解析误差门槛仍优先，不能用 convergence 掩盖错误方程。

## 5. Q1-B：v1 compatibility

将 graded mask 设为空，使用 v2 kernel 的 compatibility path 与固定 v1 kernel 对照。

### 微回路

至少覆盖：兴奋/抑制、自连接、弱边、重复发放、refractory 内到达、modulatory source、KC adaptation。每个 observation boundary 的以下数组必须 bitwise 相同：`v`、原有 `g`/`g_event`、refractory、drive、previous_drive、queue、queue_count、counts、active、active_flag、nactive、last、adaptation、modulation 及 clock/total spikes。

### 完整图

从 clean checkpoint 开始，在全部三个 dt 下，对 source-id 最小的 64 个 `generic_spiking_declared` 节点施加：10 ms zero、10 ms `+8.0` current、20 ms washout。v1 与 v2 compatibility path 在每个 segment boundary 的上述状态数组、spike counts、queue 和权重 SHA256 必须全部 bitwise 相同。

任何差异均判为 `Q1_COMPATIBILITY_FAILURE`，不得以“差异很小”通过。

## 6. Q1-C：完整 retained graph hybrid smoke

learning 固定为 off；不构造 RGB，不读取行情、旧 60 帧、label、reward、action、P&L 或 decoder。每个 trial 从同一 clean state 独立开始，在三个 dt 下各重复两次。

### 冻结 trial

每个 trial 为 10 ms zero + 10 ms stimulus + 20 ms washout，共 40 ms：

1. `ZERO`：全程 external current 为零；
2. `PHOTO128`：只对 `graded_candidate_photoreceptor` 中 source-id 最小的 128 个节点，在 stimulus 段施加 `+3.5`；
3. `LAMINA128`：同理，只刺激 `graded_candidate_lamina` 中 source-id 最小的 128 个；
4. `APL2`：刺激全部两个 `graded_candidate_apl`，电流 `+3.5`。

选择集合只由 P0 分类和 source-id 排序决定，必须记录 IDs 与 SHA256。禁止读取 active-KC/MBON 后更换 source 或 current。

### 完整图 smoke 门禁

- 所有固定输入、图规模、节点/边/contact 数与运行前一致；
- learning off，运行前后完整 weight array bitwise 相同；
- 全部状态、queue、release 和 delivery 数组 finite；
- `abs(v) < 1e6`、`abs(g_event) < 1e9`、`abs(g_graded) < 1e9`；
- 无 ring-buffer overflow、索引越界、负 queue count 或 clock 回退；
- `ZERO` 在 40 ms 内保持精确 equilibrium：零 spike、零 release、零 delivery；`v/g_event/g_graded/refractory/queue/counts/adaptation/modulation` 与 clean equilibrium 的物理值逐 bit 相同。clock 与 `last` 只允许按预期推进，不与起始时间戳直接比较；
- 三个 stimulus trial 的直接受激 source release integral 和 absolute graded delivery integral 均严格大于零；
- APL2 的原图出边 delivery signed sum 必须小于零；
- 每个 trial 的两次运行规范化状态/账本 SHA256 完全一致；
- 对三个 stimulus trial，0.05 与 0.025 ms 的受激 source release integral、signed delivery integral、absolute delivery integral的对称相对差分别 `<=0.02`；spike count、KC/MBON 活动与最终行为只报告，不作为 Q1 门禁。

若因为短刺激没有任何 KC 或 MBON spike，不算 Q1 失败；Q1 不验证功能驱动。若出现大量 KC 活动，也不算成功，不得据此调参。

## 7. 实现与审计门禁

正式运行前必须有自动测试覆盖：

- P0 分类计数和 source-id hash；
- release clamp 与解析递推；
- 四类连接；
- delay/refractory 精确 clock；
- v1 empty-mask bitwise compatibility；
- full graph fixture 的边/contact/weight 守恒；
- manifest pin、两次确定性和失败产物路径。

编译命令、编译器版本、flags、v2 source SHA256 和 binary SHA256 必须写入 manifest。正式运行开始后，v2 source、runner、tests 和 binary 的 SHA256 前后必须不变。

观察器或审计账本开启/关闭时，动力学主状态与 counts 必须 bitwise 相同。observer 改变结果判为实现失败。

## 8. 正式判定

Q1 只允许以下结果：

1. `Q1_IMPLEMENTATION_QUALIFIED`
   - 固定输入、分类、解析微回路、三档 dt、v1 compatibility、observer equivalence、完整图 smoke、确定性和权重冻结全部通过。
2. `Q1_COMPATIBILITY_FAILURE`
   - empty-mask v2 与 v1 任一 bitwise gate 不一致。
3. `Q1_NUMERICAL_FAILURE`
   - 方程、delay、refractory、解析误差或 fine-dt convergence 任一失败。
4. `Q1_FULL_GRAPH_SMOKE_FAILURE`
   - 微回路与 compatibility 通过，但完整图出现非有限值、溢出、非确定性、权重变化、守恒失败或预注册 smoke gate 失败。
5. `Q1_IMPLEMENTATION_FAILURE`
   - manifest/hash/schema/编译/测试基础设施错误，尚不能解释为模型数值失败。

判定优先级为 implementation infrastructure → compatibility → numerical → full-graph smoke → qualified。不得通过改变误差容限、dt、刺激、电流、时长、source 数、release 函数或 `tau_emit` 把失败改成通过。

纯实现错误必须保留失败目录、追加 amendment 并使用新版本目录补跑。模型/数值门禁失败则停止 Q1，不增加第二个动力学机制补救。

## 9. 产物与执行

正式目录固定为：

`runs/whole_brain_v2/Q1/20261009T-hybrid-kernel-qualification-v1/`

至少包含：

- `q1_formal.json`：判定、全部 gates、固定输入、分类与代码/binary hashes；
- `q1_raw.npz`：微回路 trace、full-graph segment summaries、release/delivery ledger；
- `q1_report.md`：人类可读结果与限制；
- `q1_manifest.json`：编译器、flags、源文件、binary、runner、tests 和环境；
- `formal.log`、`runner.stdout.log`、`launch.txt`；
- clean checkpoint 或其精确生成记录及 SHA256。

大规模完整图执行必须由 Luna 子代理在 tmux 中运行。主代理负责预注册、代码/测试审查、产物哈希和最终判定审核，不持续轮询。

## 10. 明确禁止与通过后的边界

- 不得用 R0/R0b/R0c、旧 60 帧、active-KC、MBON、动作或收益调 Q1 参数；
- 不得删除/屏蔽节点、边、弱边、自连接或不确定符号边；
- 不得把 graded output 只接到“有用”通路；
- 不得改变 v1 默认路径或让 controller 自动切到 v2；
- 不得运行 plasticity、reward、paper replay 或真实交易；
- 不得把工程化连续 release 称为完整受体生理、真实 APL 局部性或生物学验证。

`Q1_IMPLEMENTATION_QUALIFIED` 只表示 hybrid 内核按声明实现且具备进入 **V2-C0 synthetic calibration** 设计评审的资格。C0 必须另行冻结参数来源、calibration/holdout 分离、持续状态、刺激身份、KC 稀疏性与 MBON 可驱动门禁；Q1 本身不解除 S3 和市场训练阻塞。
