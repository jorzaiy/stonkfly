# Whole-brain dynamics v2：Q2 tonic graded transmission 诊断与候选机制（供 Opus 审阅）

状态：**2026-10-10 审阅草案；不是冻结预注册，不授权改代码或运行 Q2。**

本文只讨论 Q1 hybrid kernel 与 C0 synthetic calibration 暴露出的 graded transmission 语义问题。它不接触市场数据、reward、plasticity、decoder、交易或收益，也不授权 S3。完整 retained MaleCNS v1.0 图仍须保留；不得剪枝、改写原始边符号、按 KC/MBON 输出选参，或把脚本结果描述成神经交易能力。

## 1. 已冻结的前置结论

### 1.1 Q1 只证明实现资格

Q1 v7 正式判定为 `Q1_IMPLEMENTATION_QUALIFIED`。它验证了 release clamp、解析状态更新、四类连接、延迟、refractory、fine-dt convergence、empty-mask v1 compatibility 和完整图安全短跑；它明确不以 KC/MBON 功能响应作为门禁。

Q1 的 graded release 为：

`r_i(t) = clip((v_i(t) - V_rest_i) / (V_th - V_rest_i), 0, 1)`

连续传递为：

`u_j(t + 1.8 ms) += w_ij * r_i(t) / tau_emit`

其中 `V_rest=-52 mV`、`V_th=-45 mV`、`tau_emit=2.2 ms`。因此 `v<=V_rest` 时 `r=0`；静息态没有 tonic release，超极化也不能产生低于静息基线的 release change。

### 1.2 C0 v3 是有效的否定结果

C0 v3 完成 `90/90` 个独立 trial 与 `10/10` 条 history ramp，协议测试 `11 passed`，manifest 登记的 raw/JSON hash 匹配，起止 source/test/kernel hash 一致，`missing_gates=[]`。正式 verdict 为：

`C0_NO_QUALIFYING_GAIN`

五个冻结 gain `0.125, 0.25, 0.5, 1.0, 2.0` 的基础门均通过，但所有 uniform level、CAL_A 与 CAL_B 的 active KC 都为 0；MBON07/11 不可驱动，两个 pattern 的 KC set 都为空。按预注册没有启动 holdout。

关键产物：

| 产物 | SHA256 |
|---|---|
| C0 冻结预注册 | `db68a5f1d2bb5bd7d0db0d4546bd9d017e96e8c9d161eae5dd73c066b5be4b25` |
| v3 calibration JSON | `62f9bc19dbe5da93e4e05bc42d02ecf597abdcaf3f5c3e59e8843bfb7b5e2307` |
| v3 raw NPZ | `b00e9fb902fbcb9d16a36705565095b84ae61b48a5d6b1d0d9285498ef173173` |
| v3 formal JSON | `6dd290e4758453bdc3bf52c61bd9d1fd2f28b9161b8aeffdf56db4838a6c00c3` |
| v3 report | `121fdc81ef7e92abfb4c67b22f96f3f6d212805c4794c3079abbdc5e24338293` |
| v3 manifest | `0799cbd07ef8f2a768d6d2a27d742332f3659098fc5f156c0f125828362e1705` |

旧 C0 v1/v2 是实现失败证据，不得用来替代 v3 或与 v3 合并挑选结果。

## 2. C0 传递诊断的事实

C0 v3 在 `gamma=1`、`UNIFORM_4` 下另存了 audit-only transfer diagnostic。已观测：

| 指标 | 结果 |
|---|---:|
| photoreceptor 数 | 4,706 |
| release integral 墀量 | 5,752.2382857409875 |
| absolute delivery 增量 | 71,842.51089745859 |
| signed delivery 增量 | -71,842.51030404138 |
| photoreceptor voltage delta | +0.8257 至 +1.5739 mV |
| lamina voltage delta | -2.9924 至 0 mV |
| generic spiking voltage delta | -3.9509 至约 +0.000011 mV |
| KC voltage delta | 精确 0 mV |
| KC measurement spikes | 0 |

下列检查为真：external current 改变 photoreceptor 电压；photoreceptor 产生 graded release；graded source 有非零 outgoing delivery；部分 graded/generic downstream voltage 响应；KC spike count 为零。

`signed_delivery / absolute_delivery` 约为 `-0.99999999`。因此该诊断窗口中的 graded signal 几乎完全表现为净抑制。扩大同一个正的全局 `gamma` 只能同比放大已有 signed delivery，不能改变其方向，也不能赋予 `v<=rest` 的 graded 单元负的 release deviation。

## 3. 当前最强机制假设

以下是**待验证推断**，不是 C0 已直接证明的事实：

1. photoreceptor 的正 drive 产生连续 release，并沿负权边使大量 lamina 单元超极化；
2. 当前 Q1 clamp 把所有 `v<=rest` 映射为同一个 `r=0`；
3. 因此 lamina 的超极化不能表达“相对于 tonic baseline，递质释放减少”；
4. 若下游功能依赖符号反转或去抑制，信号会在此处被截断；
5. 这与 C0 中 lamina 负电压变化、几乎全负 signed delivery、KC 电压精确不变相符。

这并不证明真实果蝇视觉通路的完整生理机制，也不证明所有 photoreceptor→KC 路径都以去抑制工作。它只说明当前工程模型的单侧零基线 release 无法表示超极化方向的连续输出变化，而 C0 的观测与这个限制一致。

## 4. 冻结 Q2 前必须完成的只读机制审计

建议先做小规模、audit-only 的 `Q2-D0`，不改变动力学，不扫描参数，不读取市场或旧收益：

1. 对 `UNIFORM_4` 记录 photoreceptor、L1/L2/L3/L5、generic spiking、KC 的分层 voltage delta 分布，而不只保存 min/max。
2. 按 graded source class × target class × edge sign 分解 release、signed/absolute delivery、首次到达 clock 和非零 target 数。
3. 对 P0 中 photoreceptor 可达 KC 的最短路径，报告每一 hop 是否实际收到非零动态输入；区分“图可达”与“动力学可达”。
4. 统计 stimulus 期间低于 rest 的 lamina 数量、幅度、持续时间，以及这些节点的原图出边符号/contacts；验证其当前 outgoing release 是否恒为零。
5. 检查 KC 的直接 graded-source 输入是否为零、符号抵消，还是幅度非零但低于可检测电压变化。
6. 分别报告现有 150 ms stimulus 末端和更早 observation boundaries；只用于定位，不得把延长 trial 当成补救性通过。
7. observer/edge ledger 开关必须与主状态 bitwise 等价；图、权重和节点分类不得变化。

若上述审计不支持第 3 节假设，应停止 tonic redesign，改为根据实际断点另写预注册；不得因为本文已提出某机制而强行实现。

## 5. 候选机制族

本节仅列出供审阅的候选，不在本文选择生产公式或数值。

### 5.1 候选 A：以 tonic baseline 为中心的 release deviation

定义有界的绝对 release proxy：

`r_abs(v) = clip(r0 + (v - V_rest) / S, 0, 1)`

网络传递使用相对静息基线的变化：

`delta_r(v) = r_abs(v) - r0`

`u_j(t + delay) += w_ij * delta_r_i(t) / tau_emit`

性质：

- rest 时 `delta_r=0`，可以保持 Q1 的精确零输入平衡；
- 去极化产生正 deviation，超极化产生负 deviation；
- 负权边乘负 deviation 可表示相对基线的去抑制；
- `delta_r<0` 表示“低于 tonic baseline 的释放变化”，不能称为负递质释放。

未解决的可识别性问题：`r0` 与跨度 `S` 如何冻结；是否所有 photoreceptor、L1/L2/L3/L5、APL 共用同一数值；饱和上下限是否对称；`tau_emit` 是否仍能沿用。不得用 KC 稀疏度、MBON 动作或收益同时选择多个自由参数。

### 5.2 候选 B：显式 absolute tonic release 与平衡项

每个 graded node 在 rest 时发出 `r0>0` 的 absolute release，膜方程显式减去静息网络输入，使 clean rest 仍为平衡；刺激传播 absolute release 的变化。

该表述在局部线性范围可能与候选 A 等价，但实现和审计更复杂：需要定义每个 target 的 baseline conductance、确认有向循环下的平衡、处理 saturation，并证明没有用预计算抵消项隐式改写原图权重。优点是物理语言更接近 tonic release；缺点是更容易引入巨大的基线流和新的数值/可识别性问题。

### 5.3 候选 C：直接 signed normalized deviation

定义：

`delta_r(v) = clip((v - V_rest) / S, -1, 1)`

只把它称为 normalized release deviation，不称为 absolute release。它参数更少、零输入平衡直接，但缺少显式 `r0`，与真实释放上下界的关系更弱。若采用，文档必须持续区分工程化 deviation 与生理递质释放。

### 5.4 暂不建议的补救

- 不扩大 C0 `gamma` 网格；它不改变单侧 clamp 或传递方向。
- 不提高 direct current、延长 stimulus 直到 KC 点燃；这会使用失败结果事后选刺激。
- 不直接刺激 lamina、KC 或 MBON 来绕过断点。
- 不按 cell type、edge sign、target 或 KC 输出设置不同 gain，除非未来有独立外部依据并单独预注册。
- 不把不确定符号改成有利符号，不剪边，不把脚本点燃包装成神经输出。
- 不在 Q2 中同时加入 plasticity、noise、sensory adapter、market observation 或 decoder。

## 6. 若进入 Q2，实现前建议冻结的资格门

Q2 应是新的隔离 kernel qualification，而不是修改或重解释已通过的 Q1。Q1 与 C0 v3 的源码和产物保持不可变引用。

### 6.1 解析微回路

至少在 `dt=0.1/0.05/0.025 ms`、两次重复下分别验证：

1. rest 时 deviation、delivery 和状态变化精确为零；
2. excitatory edge × depolarization：target 正向响应；
3. excitatory edge × hyperpolarization：target 负向响应；
4. inhibitory edge × depolarization：target 负向响应；
5. inhibitory edge × hyperpolarization：target 正向响应；
6. 上下 saturation、delay、逐边 ledger 与解析公式一致；
7. graded→graded→spiking 两 hop sign inversion 在预定 clock 首次影响 target；
8. refractory 期间 deviation input 的积累/衰减语义明确；
9. empty graded mask 与冻结 v1 bitwise 等价。

### 6.2 完整图安全门

- 仍为 166,700 nodes、25,582,938 directed edges、124,177,617 contacts；完整 weight array bitwise freeze。
- zero input 从 clean rest 保持精确 equilibrium；不得用未记录的 baseline correction 漂移。
- 所有状态 finite、无 queue overflow、observer 不改变主状态。
- start/end source、binary、tests、runner hash 一致。
- 0.05 与 0.025 ms 对预注册连续刺激满足解析误差与 convergence 门。
- 只验证传递方向和数值安全，不用 KC 稀疏/MBON 响应选择 Q2 参数。

### 6.3 功能门与资格边界

Q2 kernel qualification 通过只说明双向 graded deviation 被一致实现，不说明视觉、生理或交易有效。之后仍须另写新的 synthetic calibration 预注册；不得把旧 C0 的 calibration/holdout 门直接事后套到新机制上，也不得复用旧 C0 holdout 数据选择公式。

如果 `Q2-D0` 不支持 tonic-baseline 假设，或候选公式的自由参数无法用独立规则冻结，应判 `Q2_MODEL_NOT_IDENTIFIED` 并停止实现。

## 7. 建议的实验顺序与停止条件

1. Opus 审阅本文，只评估机制诊断、候选公式和可识别性，不改代码。
2. 根据审阅意见冻结 `Q2-D0` 只读审计；先定位实际动态断点。
3. 只有 D0 支持 tonic-baseline 假设时，冻结唯一 Q2 机制及其参数来源。
4. 实现隔离 Q2 kernel，并只跑解析微回路与完整图安全资格。
5. Q2 资格通过后，另行冻结新的 synthetic calibration；校准与 holdout 继续隔离。
6. 在新 synthetic holdout 通过前，sensory adapter、plasticity、S3 和市场训练继续阻塞。

任一步失败都不得通过追加参数、扩大网格、读取收益或选择更有利输出补救。

## 8. 请 Opus 重点回答

1. C0 证据是否足以把“单侧零基线 release 无法传递 lamina 超极化”列为首要假设？还缺哪项最小审计证据？
2. 候选 A/B/C 中哪一种最适合作为工程化 whole-brain kernel 语义？是否存在更少自由参数、同时保持精确 rest equilibrium 的表达？
3. 若选择候选 A，`r0` 与 `S` 应如何从独立规则冻结，避免用 KC/MBON 输出调两个参数？
4. photoreceptor、L1/L2/L3/L5 与 APL 是否应共享同一 deviation 公式；若不共享，需要什么外部证据才能避免任意分类调参？
5. Q2 是否必须显式模拟 tonic absolute release，还是只传播相对基线 deviation 已足以满足当前工程目标？
6. 四象限微回路、两 hop sign inversion、zero equilibrium、dt convergence 和完整图安全门是否足够；还缺哪些会导致假通过的测试？
7. P0 的 uncertain-sign fractions 均低于 20% 触发阈值。Q2 改为双向 deviation 后，是否仍应加入固定的 ambiguous-sign sensitivity arm，还是保持 P0 规则即可？
8. 若 D0 证实 KC 的动态图路径断在其他 generic spiking 层而非 lamina release clamp，应该停止 Q2 redesign，还是仍有独立理由修正 release 语义？
9. 是否同意旧 C0 永久保留为对 Q1 单侧 clamp 的失败结果，而新机制必须使用新阶段名和全新 calibration，不能称为 C0 补跑？

## 9. 当前建议

在 Opus 回答和 Q2-D0 完成前，不实现 tonic release，不运行新的 calibration。当前最保守的主张仅为：

> Q1 的单侧、零静息基线 release 在实现上合格，但 C0 证明它无法在冻结 synthetic photoreceptor drive 下驱动任何 KC；现有传递账本与电压方向提示，不能表达超极化相对 tonic baseline 的释放下降可能是首要瓶颈。该解释仍需分层动态路径审计验证。

## 10. Opus 审阅意见（2026-10-10）

本节只依据本文第 1–2 节的数字作出，Opus 没有对照 C0 v3 产物重新计算。本节不构成预注册，不授权修改代码或运行实验。

### 10.1 总体判断

方向基本同意，但第 3 节的假设目前**只能算候选之一，不能算首要**，理由如下：

1. **净抑制比值说明的是 photoreceptor 的出边，不是 lamina 的截断。** 在 Q1 clamp 下，v ≤ rest 的 lamina 不释放，所以诊断窗口里的 delivery 基本都来自 photoreceptor。signed/absolute ≈ −0.99999999 只说明 photoreceptor 出边几乎全为负权。photoreceptor 是组胺能，光照使 L1–L3 超极化，这在生物学上本来就是预期结果，lamina 变负说明这一段工作正常。
2. **第一层 spiking 已经没有发放。** generic spiking 的最大电压变化约 +0.000011 mV，KC 的电压变化精确为 0。只要中间任何一层 spiking 不点燃，下游就收不到信号。修正 release 语义最多只能证明 lamina 能把信号送出去，不能证明信号能穿过 medulla 的 spiking 层。
3. **缺少最便宜的证据：反事实账本。** 用 C0 v3 已记录的 lamina 电压，离线按候选公式计算"如果允许负 deviation，lamina 会给下游送出多少带符号的输入、落在哪些节点"。这只需读取数据并做一步线性计算，不改变动力学，应作为 Q2-D0 的第一项。
4. **L1 与 L2 的方向相反。** L2 为胆碱能（兴奋性边），超极化后向下游传出的是更强的抑制；L1 为谷氨酸能（图中多为抑制边），超极化后才是去抑制。所以在增光刺激下，双向 deviation 只让 ON 通路受益，OFF 通路会更负。D0 必须按 L1、L2、L3、L5 分别报告出边符号和反事实 delivery，不能笼统地称为"去抑制"。

升为首要假设的最低条件：反事实账本显示非零的去抑制项能到达某个 spiking 下游；**同时**逐跳审计（第 4 节第 3 项）显示，第一个没有动态输入的 hop 恰好位于 lamina 之后。

### 10.2 对第 8 节的逐条回答

1. **首要假设与缺失证据**：目前只列为候选之一。还缺 10.1 中的反事实账本和逐跳的首个断点。
2. **候选公式**：推荐**候选 C，并取 S = V_th − V_rest = 7 mV**，直接继承 Q1 的跨度。理由：
   - 在 v ≥ rest 时，它与 Q1 逐位等价，Q1 是它的特例；
   - 不引入新的自由参数；
   - rest 时 deviation 恒为 0，零输入平衡自动成立。

   候选 A 多出 r0 和 S 两个参数，Q1 也不是它的特例。候选 B 在线性区与 A 等价，但会带来巨大的基线流和抵消项，审计成本高，不推荐。
3. **若采用候选 A，如何冻结 r0 和 S**：S 继承 7 mV；r0 只能按约定取值（例如 0.5，即 rest 位于动态范围的中点），并明确写为约定而非拟合。采用 C 则这个问题不存在。C 的下饱和 −1 是同类约定，可以另设一个**固定的、只做描述**的敏感性臂（例如下限 −0.5），不参与任何门禁或选择。
4. **各类 graded 细胞是否共享公式**：photoreceptor、L1/L2/L3/L5、APL 共用同一个公式和同一个 S。只有拿到按细胞类型的外部电生理定量证据，并单独预注册后，才允许分类设定。APL 需单独审计：它若没有抑制性入边，就永远不会低于 rest，改动对它没有影响；它若有抑制性入边，APL 的超极化会去抑制 KC，D0 必须统计这些入边及其在刺激期的实际电压。
5. **是否需要显式 absolute tonic release**：不需要。对当前工程目标来说，只传播相对基线的 deviation 已经足够。
6. **资格门是否足够**：不够。以下缺口可能导致假通过，建议补齐：
   - **rest 平衡的稳定性（最重要）**。双向 deviation 后，graded 子图中的"抑制 × 抑制"环路会变成正反馈，rest 可能是**不稳定**的平衡点。"零输入保持精确平衡"这一门禁发现不了这个问题。需要两项检查：graded 子图在 rest 处线性化后的谱半径或最大实部特征值；完整图上 ±ε 初态微扰后是否发散或漂移。R0b 的吸收上态就是这类失败的先例。
   - **撤刺激后回到 rest**。必须验证系统不会锁进新的吸收态，这是 R0b 中 H_stim 失败的教训。
   - **photoreceptor 的负向 drive**（暗刺激或负电流）。只用正电流只测到了 ON 的一半。
   - **符号置换的阴性对照**。打乱 graded 出边的符号后，四象限测试和两 hop sign inversion 都应失败；否则说明测试本身不敏感。
   - **独立参考实现**。微回路的期望值由一个独立的小型 dense 实现计算，不得复用 kernel 的代码路径。
   - **graded 语义不泄漏**。generic spiking 单元在阈下不得向外传递任何 deviation，graded mask 外的单元行为必须与 Q1 逐位一致。
7. **ambiguous-sign 敏感性臂**：门禁仍按 P0 规则执行。但双向化后，符号错误会在两个方向上传播，影响变大，所以建议加入一个固定的"uncertain sign 全翻转"描述臂，不参与任何选择。
8. **若断点在 generic spiking 层**：修正 release 语义仍有独立理由，因为 photoreceptor→LMC 的信号本来就是超极化方向，单侧 clamp 是已知的语义缺陷。但必须满足以下条件：
   - Q2 只能以"语义正确"立项，不能以"KC 会响应"立项，并且要预先写明它很可能仍然驱动不了 KC。
   - 届时真正的问题变成：从 clean rest 出发、spiking 层没有任何 tonic 活动的整图，能否逐跳传递阈上信号。这和 R0b 正好是两个极端：R0b 是失控的吸收上态，C0 是完全沉寂。任何提高传递的改动都必须同时防止滑回失控上态。
   - 把 medulla 的 Mi/Tm 等单元改为 graded，需要外部证据并单独预注册，不得因为 D0 的结果顺手扩大 graded mask。
9. **C0 的地位**：同意。C0 v3 永久保留为 Q1 单侧 clamp 的否定结果。新机制使用新的阶段名，做全新的 calibration 和 holdout，不得称为 C0 补跑。

### 10.3 额外提醒

全视野均匀刺激下 KC 为 0，未必是病理现象。果蝇 MB 的视觉输入有限，主要经视觉投射神经元进入部分 KC 亚型。新的 synthetic calibration 预注册必须**事先**写明哪些刺激条件下"KC 为 0"是可接受的结果。否则门禁天然偏向"点燃"，等于把失败结果当成选参依据。

### 10.4 建议顺序

1. Q2-D0 只读审计，以反事实账本和逐跳首个断点为第一优先，并按 L1、L2、L3、L5 与 APL 分别报告。
2. 只有当 D0 满足 10.1 中的升级条件，或者按 10.2 第 8 条、以"语义正确"单独立项时，才冻结 Q2，并采用候选 C（S = 7 mV，无新增自由参数）。
3. Q2 资格门在第 6 节的基础上，必须补充 10.2 第 6 条的各项，其中 rest 稳定性和撤刺激回落两项不可省略。

