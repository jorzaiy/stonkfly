# Stonkfly v2 M2-E 最小探索性实施设计

状态：**FORMAL EXPLORATION COMPLETE / `ENGINEERED_ASSOCIATIVE_UPDATE_NOT_OBSERVED`**
日期：2026-10-12 UTC
范围：仅合成气味—人工奖励关联；不解封旧 M1-RT/M2，不修改 M1-RQ 或 M1-E1，不接入市场、交易、P&L 或执行。

## 1. 目标与结论边界

M2-E 只回答一个工程问题：在 M1-E1 已稳定的连续 PN→KC→全局 APL 感觉表征上，一个受 MaleCNS 连接约束、但含人工 DAN 脉冲和代理学习率的 KC→MBON 可塑性头，能否：

1. acquisition：使两个气味形成相反的输出关联；
2. reversal：在奖励身份交换后更新关联方向；
3. retention：在没有新奖励时保留更新后的方向。

若全部成立，唯一允许的成功表述是 **“当前简化模型展示了可重复的工程化气味—奖励关联学习”**。它不证明真实果蝇采用同一方程、同一学习率或同一 DAN 幅度，不证明行为、交易或收益学习，也不改变旧 M1 的正式 `NO-GO`。

## 2. 现有实现与静态结构审查

### 2.1 可复用部分

- `stonkfly/mb_lab/e1.py`：复用 M1-E1 的右侧 145 PN、1,907 个可驱动 KC、全局 APL（G 臂）、时间线和数值积分；感觉参数保持不变。
- `stonkfly/mb_lab/topology.py`、`inventory.py`：复用只读 MaleCNS 图索引和拓扑编译。
- `stonkfly/neural/circuit.py`：复用细胞身份规则和“只允许现有 KC→MBON 边”的原则。
- `stonkfly/neural/rule.py`：复用 baseline-centered、时序敏感的数学形式及其 closed-form 衰减代码；不直接继承 spike-Hz 语义。
- `stonkfly/mb_lab/budget.py`、`checkpoint.py`：复用预算、不可覆盖 checkpoint、失败证据和哈希工具。

现有测试已经覆盖 reward/aversive 分路、frozen weights、checkpoint 确定性和无 reward 不产生新 plastic drive；但多数 full-graph 测试只验证“有突触变化”，没有验证两个气味的 acquisition、reversal、retention 或对照分离。

### 2.2 不能直接复用的部分

现有 `MemoryBrain`/`FlyController` 面向完整 LIF/视觉/市场历史路径，塑性输入为 spike Hz，选择双侧全部 KC→MBON07/11 共 7,835 条边。E1 输出是右侧 1,907 KC 的连续无量纲 rate。直接把 E1 rate 当 Hz 会制造未声明的单位转换，因此 M2-E 不直接调用 controller 或交易 decoder。

另有文档/代码差异必须在 M2-E 配置中显式消解：当前 `rule.py` 是 `memory_decay_seconds=900`、`minimum_fraction=0.3`，而 `docs/model.md` 仍写 1,800 s 和 0.1。M2-E 以运行时源码值 900 s、0.3 为候选实现事实，并在 provenance 中记录；这不是改写历史模型结论。

### 2.3 右侧结构 inventory

源图保持完整且只读；下列是 M2-E assay 对完整图的索引视图，不删除源图节点或边。

| 路径 | 目标细胞 | 边数 | contacts | 有输入的 1,907 KC |
|---|---:|---:|---:|---:|
| KC_R_covered→MBON07_R | 2 | 1,806 | 11,507 | 931 |
| KC_R_covered→MBON11_R | 1 | 1,705 | 22,107 | 1,705 |
| KC_R_covered→PAM11_R | 7 | 1,789 | 2,541 | 881 |
| KC_R_covered→PPL101_R | 1 | 1,469 | 6,168 | 1,469 |
| PAM11_R→MBON07_R | 7→2 | 14 | 441 | — |
| PPL101_R→MBON11_R | 1→1 | 1 | 781 | — |

MBON07_R body IDs 为 `12859`、`515338`，递质注释为 glutamate；MBON11_R body ID 为 `11402`，注释为 GABA。右侧 KC→MBON11 的 contacts 约为 KC→MBON07 的 1.92 倍，因此不得直接以未归一化总输出作为“学习”。MaleCNS 结构与实验文献支持 PPL101/γ1pedc 对 KC→MBON11 的厌恶学习相关抑制；PAM11/α1 与 MBON07 的映射用于本工程 reward compartment，但其迁移到短时双价 reversal 是代理假设。

## 3. 固定的最小模型

### 3.1 感觉层

- 只使用 E1 的 G 臂；`tau_k=10 ms`、`tau_apl=50 ms`、`input_scale=1`、`apl_gain=1`、`threshold=0.35`。
- 0–500 ms baseline、500–1,000 ms odor、1,000–2,000 ms recovery，`dt=0.025 ms`，10 ms 分析 bin。
- 每次感觉呈现从零快速状态开始；学习权重跨训练 trial 保留。
- M2-E 实现必须对同一 toy/真实输入逐位或严格数值复现 E1-G 的 KC integral、population trace 和 final state，才可运行探索矩阵。
- 因 MBON 权重在本简化模型中不反馈到 KC，感觉轨迹可在五个实验臂间只读复用；缓存键必须包含 stimulus、E1 参数、topology 和源码哈希。

### 3.2 两个连续 valence 输出

只读取 1,907 KC 到三个右侧 MBON 的现有正权边。保留每个 MBON 的单细胞输出，同时定义两个固定聚合通道：

- `avoidance channel`：两个 MBON07_R；
- `approach channel`：一个 MBON11_R。

对通道 `c`：

`y_c(o,t) = sum_e[w0_e * m_e(t) * r_pre(e)(o)] / sum_e[w0_e]`

其中 `w0` 是 MaleCNS aggregate weight，`m=1` 为初始 efficacy，`r` 是 E1 KC pulse mean rate。按初始总输入归一化只消除两条通路的规模差异；不拟合气味标签。所有单 MBON 值必须同时报告，避免聚合掩盖一个 MBON 的异常。

固定的关联分数只用于 assay，不映射行为或动作：

`V_t(o) = y_approach,t(o)/y_approach,0(o) - y_avoidance,t(o)/y_avoidance,0(o)`

因此每个气味在训练前 `V=0`。若任一基线通道近零，实验停止，不加入偏置电流救活输出。主对比为 `D_t = V_t(A)-V_t(B)`。

### 3.3 可塑性适配层

复用 `rule.advance` 的形式：

`drive_e = eta * [x_pre(e) * Dtrace_c - D_c * xtrace_pre(e)]`

并复用两个一阶记忆状态、50 ms efficacy filter、900 s 回归基线和 efficacy `[0.3, 2.0] × w0` 的 bounds。差异是 `x` 明确为 E1 无量纲 KC rate，`D` 为人工无量纲脉冲；故 `eta` 是工程有效学习率，不是生物测量值，也不能与旧 LIF 的 `eta=0.001` 直接比较。

唯一候选适配参数在实施前冻结：

| 参数 | 候选值 | 等级 | 说明 |
|---|---:|---|---|
| KC/DAN trace | 1 s | P | 沿用现有规则；真实时序依赖有实验依据，精确常数未在本图验证 |
| memory return | 900 s | P | 当前源码值；不是 MaleCNS 测量 |
| efficacy filter | 50 ms | P | 当前源码值 |
| bounds | 0.3–2.0 × baseline | P | 数值保护/模型选择 |
| DAN amplitude | 1.0 dimensionless | M | 人工 teaching pulse，不称为 dopamine concentration 或 firing rate |
| DAN duration | 100 ms | P | 固定短脉冲；非个体 DAN 生理拟合 |
| paired window | odor 900–1,000 ms 同时给 DAN | P | odor 已建立后共现，预期产生目标通路 depression |
| unpaired delay | odor offset 后 6 s | P | 独立实验显示约 6 s 间隔时塑性很小；不是硬生物常数 |
| effective `eta` | 0.1 s⁻¹ | M | 在 `x,D≤1` 时每次 100 ms pairing 的理论最大直接变化约 1%；按数值安全选定，不根据结果调参 |
| inter-trial washout | 6 s | M | 使 1 s trace 残余约 0.25%，减少相邻气味串扰 |

等级：V=连接组直接验证；P=文献支持方向但数值为代理；M=纯工程模型假设。

reward pulse 只路由到 PAM11/MBON07 compartment，使同时活跃 KC→MBON07 efficacy 倾向下降；aversive pulse 只路由到 PPL101/MBON11，使 KC→MBON11 倾向下降。DAN contact fractions 用于同一 compartment 内的 target 分配，不解释为实测 dopamine 浓度。

## 4. 独立合成气味与实验矩阵

实施冻结时公开新的 seed label/hash，不读取 RQ、RT 或 E1 的刺激内容。生成后不按神经结果筛选：

- 8 个独立 replicate，每个 replicate 独立生成 A/B 两个气味；
- 每个气味激活 29/145 PN，幅度和 jitter 生成规则沿用 E1，但使用全新 seed domain；
- training 和 probe 使用独立派生 seed；probe 只用于阶段读出，不进入权重更新；
- 每阶段 5 个独立 probe repeat/odor；probe 计算使用权重 snapshot，不推进塑性时钟。

每个 replicate：

1. `P0`：naive probe；
2. acquisition：10 个 cycle；每 cycle 各呈现一次 A+reward、B+aversive，cycle 内顺序固定 seed 随机化；
3. `P1`：无奖励 probe；
4. reversal：10 个 cycle；交换为 A+aversive、B+reward；
5. `P2`：无奖励 probe；
6. 60 s 零输入、零 DAN，仅运行被动 memory decay；
7. `P3`：无奖励 retention probe。

共 `8 × [40 training + 40 probe] = 640` 个唯一感觉路径。五个学习/对照臂共享相同只读感觉缓存，不能因 arm 改变刺激或 KC 表征。

## 5. 五个固定实验臂

| 臂 | 奖励时序 | 权重更新 | 科学用途 |
|---|---|---|---|
| paired | 正确 odor-valence，900–1,000 ms | 开 | 候选关联学习 |
| frozen weights | 与 paired 相同 | `frozen=True`，精确不变 | 排除固定感觉/读出偏置 |
| shuffled reward | 每阶段/每气味各 5 reward、5 aversive，标签与 odor 独立 | 开 | 排除奖励次数、顺序和全局漂移 |
| unpaired timing | 正确 valence，但 offset 后 6 s | 开 | 检查时间配对依赖 |
| DAN omitted | 正确日程但 `D=0` | 开 | 检查无 teaching signal 时的数值漂移 |

shuffled arm 使用预生成的平衡排列，不允许“随机后恰好仍与气味相关”。所有臂有完全相同的 trial 数、washout、memory decay 和 probe。

## 6. 预先固定的分析

### 6.1 实施正确性

- E1-G 等价、ZERO、finite、时间线和 cache/no-cache 等价；
- 只有现有右侧 KC→MBON 边可改变，符号保持兴奋性；
- frozen 与 DAN-omitted 的 edge weights 必须 bitwise 不变；
- reward 对 MBON11、aversive 对 MBON07 的**新增 plastic drive** 必须精确为零；已有记忆状态仍可按固定 900 s 规则被动回归基线，并须与解析预期一致；
- 每阶段保存 weights/hash、两个通道及每个 MBON 输出、trace/drive 摘要和 saturation；
- 任一实现检查失败均为 `M2_E_IMPLEMENTATION_FAILURE`，不能解释科学结果。

### 6.2 工程学习证据

每个 replicate 计算：

- acquisition：`D_P1-D_P0`，预期 `>0`；
- reversal update：`D_P2-D_P1`，预期 `<0`；
- reversal endpoint：`D_P2<0`，要求真正跨过基线，而不只是减弱旧关联；
- retention：`sign(D_P3)=sign(D_P2)`，并报告 `|D_P3/D_P2|`；
- 分通道、分气味、分 MBON 的 fractional response 和 weight change；
- paired 相对四个对照的差中差。

8 个 replicate 使用全部 `2^8=256` 个 sign flips 的精确随机化检验，并报告 replicate bootstrap 95% CI 和原始点。层级顺序固定为 acquisition→reversal update→reversal crossing→retention；前一项不成立时，后项仍报告但不构成成功。

探索性标签 `ENGINEERED_ASSOCIATIVE_UPDATE_OBSERVED` 仅在以下均满足时使用：

1. paired acquisition 和 reversal update 的平均方向正确，95% CI 不跨 0；
2. 至少 7/8 replicate 的 reversal endpoint 跨过 0；
3. 60 s retention 至少 7/8 保持 reversal 符号，且中位幅度保留率 ≥75%；
4. paired 的 acquisition/reversal 差中差分别超过每个对照，且 frozen、DAN-omitted 无权重变化；
5. 无非有限、跨通路污染或大规模 bound saturation。

否则报告 `ENGINEERED_ASSOCIATIVE_UPDATE_NOT_OBSERVED`，保留效应量，不改 eta、脉冲、trial 数或气味后重跑。该标签不是 M1/M2 正式 PASS。

## 7. 预算、工作量和产物

### 预算

- 640 个 E1-G `dt=0.025 ms` 感觉路径；塑性头计算相对便宜；
- 预计 CPU 0.6–1.0 h，wall 0.7–1.2 h；
- 硬上限：2 CPU-hours、2 wall-hours、2 GiB RSS、250 MiB 产物；不使用 GPU；
- 只保存每 trial 的 KC integral/eligibility snapshot、10 ms population/APL trace 和阶段权重摘要，不保存每一步全 KC 状态。

### 实施工作量

- topology/readout 与静态 inventory：0.5–1 人日；
- E1 rate→plasticity 适配、五臂时序和 toy oracle：1–1.5 人日；
- runner、统计、预算、不可覆盖产物与 manifest：1–1.5 人日；
- 独立代码/方案复核及一次小矩阵：0.5–1 人日。

合计约 **3–5 人日**。正式矩阵前只需 toy 和极小 smoke，不建立四域密封或签名授权体系。

产物至少包括 `config`、seed/stimulus manifest、static inventory、每阶段 raw summaries、weights、statistics、report、stdout、resource usage、attempt ledger 和完整哈希 manifest。失败目录永久保留。

## 8. 停止条件与可能失败原因

立即停止且不自动调参：

1. E1-G 等价失败，或感觉缓存跨不等价状态复用；
2. MaleCNS 哈希、1,907 KC、右侧 MBON/DAN 身份或上述边数不一致；
3. 任一基线 MBON 通道为零/非有限；不得加 tonic current 或改感觉阈值救活；
4. frozen/DAN-omitted 改变权重，或 reward/aversive 污染错误 compartment；
5. >1% plastic edges 触及任一 efficacy bound，表明候选学习尺度不可解释；本轮记为失败，不就地降低 eta；
6. 预算越界、产物不完整或已有目录将被覆盖；
7. 固定矩阵完成后停止，无论结果是否理想。

预期失败模式包括：

- 两气味激活的 KC→两个 MBON 覆盖太少或高度重叠，导致关联弱/泛化；
- MBON07/11 先天覆盖差异在固定归一化后仍形成高方差；
- 单向 depression 在等量 reversal 后只能抵消 acquisition，不能跨过零；
- 1 s eligibility 与 E1 活动时间尺度不匹配；
- `eta=0.1 s⁻¹` 太弱而不可测，或太强导致 saturation；二者都应如实成为候选模型结果，而不是触发调参；
- 60 s retention 只反映代理 900 s memory return，而非真实果蝇记忆；
- 简化 readout 没有 MBON 动力学、MBON↔DAN 反馈、其他 compartments 和行为选择，因此即使成功也仅是工程关联头。

## 9. 解释责任

真实连接拓扑贡献仅限：哪些 KC 可驱动哪些右侧 MBON、每条 aggregate contact weight、DAN/MBON compartment 身份与同 compartment contact fractions。E1 rate dynamics、总输入归一化、人工 DAN、有效 eta、记忆常数、固定 valence score 和无反馈 readout 都是代理。B/C 或 control 差异只能支持这些组件组合下的工程机制，不能把成功归因于“真实果蝇连接组有学习优势”，除非未来加入预先规定的度保持随机拓扑对照并单独检验；本最小 M2-E 不作该主张。

## 参考

- Hige et al., *Heterosynaptic Plasticity Underlies Aversive Olfactory Learning in Drosophila*, Neuron 2015, doi:10.1016/j.neuron.2015.11.003。
- Handler et al., *Distinct dopamine receptor pathways underlie the temporal sensitivity of associative learning*, Cell 2019, doi:10.1016/j.cell.2019.05.040。
- Li et al., *The connectome of the adult Drosophila mushroom body provides insights into function*, eLife 2020, doi:10.7554/eLife.62576。
- Ichinose et al., *Reward signal in a recurrent circuit drives appetitive long-term memory formation*, eLife 2015, doi:10.7554/eLife.10719。
- Aso et al., *Mushroom body output neurons encode valence and guide memory-based action selection in Drosophila*, eLife 2014, doi:10.7554/eLife.04580。

## 10. 完成结果

正式 run `m2-e-20261012T053756Z` 完成 8 对气味和五臂矩阵。acquisition 变化
8/8 为正，均值 `+0.011776`；reversal 变化 8/8 为负，均值 `-0.013861`。
但 P2 仅 6/8 气味对跨零，低于冻结的 7/8，因此保持 NO-GO。结果只能支持
当前人工 DAN/代理可塑性头中的工程化双向关联更新，不能改写 M1-RQ 或解释
为真实果蝇行为。逐 pair 数据与 provenance 见 `results/stonkfly_v2/`。
