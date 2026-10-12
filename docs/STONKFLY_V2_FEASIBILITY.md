# Stonkfly v2 Mushroom Body Lab：可行性评审（M1/M2 修订稿）

状态：**原则 GO；当前只授权 M1 实现准备，不授权 M1 正式实验，不授权 M2 实现或实验。**
日期：2026-10-11
控制性实施章程：`docs/STONKFLY_V2_M1_CHARTER_DRAFT.md`
范围：M1 合成气味 PN→KC→APL 表征；M2 奖励关联与反转延期。不读取市场数据，不训练交易策略，不调用交易解码或执行层，不接入账户。

## 0. 2026-10-11 审阅修订（控制性条款）

本节和 `docs/STONKFLY_V2_M1_CHARTER_DRAFT.md` 取代本文后续旧草案中关于阶段授权、数据集、表征门、B/C 比较、预算和 M2 实施的安排。后续旧草案保留为评审轨迹；若有冲突，以本节和 M1 章程为准。

### 0.1 两个串行里程碑

- **M1：合成气味表征。** 只研究 PN→KC→APL 的稀疏性、可靠性、刺激分离、撤刺激恢复和数值收敛，并比较真实 PN→KC adjacency 与 degree-preserving 随机对照。
- **M2：奖励关联与反转。** 未来另行立项和授权；M1 期间不实现 plasticity、reward、MBON/DAN learning 或 reversal。

当前只允许 M1 的静态 inventory、非科学 toy benchmark、代码/测试/runner 准备、BENCHMARK_ONLY CLI smoke 和冻结包。M1-RQ 正式运行仍为 NO-GO。

### 0.2 四套严格隔离的数据

| 集合 | 唯一用途 | 解封条件 |
|---|---|---|
| M1-RQ | 表征资格 | M1 方程、参数、统计、runner、预算冻结并获正式授权 |
| M1-RT | 表征独立测试 | M1-RQ 通过后另行一次性授权 |
| M2-LQ | 未来学习资格 | M2 另行批准后；M1 不得访问 |
| M2-LT | 未来学习独立测试 | M2-LQ 通过后；M1 不得访问 |

四套集合使用独立 seed domains、文件和 SHA256。测试集不能用于 debug、调参、选择刺激/网络 seed/统计口径。BENCHMARK_ONLY 是第五个永不进入正式集合的域。

### 0.3 门槛重新分类

- **实施/数值资格**：provenance、graph/degree invariants、RNG 隔离、toy oracle、finite/bounds、ZERO/recovery mechanics 和跨 `dt` 收敛。跨 `dt` 只检查离散化，不是生物学证据。
- **科学检验**：S1 稀疏性；S2 以 odor 为独立单位的 within-minus-between separation；S3 撤刺激恢复；S4 APL-off 配对因果效应；S5 B/C topology randomization。
- 试次 pair 不作为独立样本；S2 使用 odor-label permutation。
- APL-off 的唯一主指标是 same-odor/same-noise active-fraction 差，使用 exact sign-flip permutation；跨气味相关仅描述，不再允许 OR 门。
- B/C 的唯一主指标是 separation statistic；B 相对 39 个 degree-preserving rewires 做 Monte Carlo rank test。

所有科学检验同时要求统计门和最小实际效应，精确定义见 M1 章程。

### 0.4 拓扑与代理动力学的可识别性

- A 原始 PN/KC/APL 诱导子图当前只做静态 inventory；因 KC→KC 等边缺少独立生理参数，不进入 M1 动力学 verdict。
- B 是真实 PN→KC adjacency 加固定代理动力学。
- C 只替换为 degree-preserving PN→KC rewires；动力学、权重规则、刺激、噪声和统计与 B 完全相同。
- `B passes` 与 `B > C` 是独立结论。B 通过不证明真实 topology 有优势；只有 B 在 M1-RQ 和 M1-RT 均超过 C 的预注册零分布，才支持“该 adjacency 在该代理模型内有增量贡献”。仍不得外推为真实果蝇的普遍结构优势。

### 0.5 静态规模与 toy 成本

只读 inventory 得到：

| 单侧 | PN | KC | APL | PN→KC | KC→APL | APL→KC | 原始内部边 |
|---|---:|---:|---:|---:|---:|---:|---:|
| L | 145 | 2,019 | 1 | 10,734 | 2,019 | 2,019 | 338,836 |
| R | 145 | 2,045 | 1 | 10,732 | 2,044 | 2,044 | 344,658 |

其中 B/C 只纳入至少接收一条合格同侧 PN 输入的 olfactory-covered KC：左侧 1,887、右侧 1,907；未覆盖 KC 不得作为零响应样本改善稀疏度。A-static 仍保留表中的全部 KC。inventory 实测约 2.7 s wall、401 MiB peak RSS。右侧边界另有约 115,258 条外部→候选和 170,412 条候选→外部边，因此 A 不能被描述为完整 MB。

145 PN/2,045 KC/10,732 edge 的 BENCHMARK_ONLY 数组更新，4 s 模拟时间实测：`dt=0.1/0.05/0.025 ms` 分别约 0.89/1.76/3.53 s，三档合计 6.18 s/trial，peak RSS 约 33 MiB。该 toy 的 20/50 ms 常数和 seed 只测吞吐，不是 M1 候选参数。

按 16 odor × 10 repeats、B+39C、RQ/RT 两阶段估算朴素单核约 22 CPU-hours（约 11 小时/阶段）；加入 APL-off、ZERO、产物和约 2× 安全系数，M1 formal 预算草案为 **48 CPU-hours、4 GiB RAM、5 GiB artifacts、无 GPU**。摘要式产物预计 0.2–1 GiB；本地已配置环境的边际服务费用为 0，但必须计量 CPU-hours、wall、峰值 RSS 和字节数。正式申请前必须用冻结 runner 的 BENCHMARK_ONLY smoke 重测。

### 0.6 当前 GO / NO-GO

- **GO**：M1 实现准备，预计 7–11 工程人日。
- **NO-GO**：M1 正式实验、M2 实现/实验、A 原始子图动力学、行情、交易策略、paper/live execution。
- 正式申请前必须冻结唯一模型方程/参数、来源标签、39 个 rewire seeds、四个 sealed manifests、统计规则、verdict precedence、失败条件、runner/tests hashes 和预算。

---

## 以下为初版评审轨迹

以下内容不授权执行；其中与 §0 或 M1 章程冲突的条款已被取代。

## 1. 结论

**总体判断：GO，但仅限另立 v2、先实现并冻结一个离线最小可验证原型（MVP）。**

这个 GO 不表示现有 MaleCNS v1 动力学可继续训练，也不表示已经有一个生物学验证模型。它只表示：仓库已有足够的连接组来源锁、图审计、确定性、跨 `dt`、产物哈希和 fail-closed 基础设施；外部研究也足以约束一个小规模 PN→KC→APL→MBON/DAN 实验的结构与定性机制。相反，单细胞动力学、突触强度换算、APL 增益和学习速率没有形成可直接拼装的完整实测参数集，必须显式标为代理假设。

以下事项是 **NO-GO**：

- 把 v1 的 `contact × 0.275`、统一 LIF 参数或失败剂量带迁入 v2；
- 用 v1 的沉寂/失稳边界反推阈值、APL 增益、时间常数或学习率；
- 在表征门失败后继续扫参数，或按 KC 输出、反转成绩、收益选择参数；
- 在本项目接入行情、交易策略、paper/live execution；
- 将模型反转成绩表述为真实果蝇行为、意识、痛苦、愉悦或盈利学习。

推荐的主问题不是“果蝇能否交易”，而是：**在真实 MaleCNS 蘑菇体拓扑约束与独立文献约束下，一个明确简化的工程模型能否产生稀疏、可区分、可重复、撤刺激可恢复、跨步长一致的 KC 表征，并在固定的 KC→MBON 三因子规则下完成合成关联的反转？**

## 2. 已审阅的仓库证据

### 2.1 权威结论

- `docs/20261010-male-cns-v1-final-archive.md`：v1 在完整图、通用 LIF/电流型突触和统一权重映射下只有沉寂或持续高活动；视觉 graded 路线另有线性失稳。归档明确禁止把结果外推为真实果蝇或真实蘑菇体失败。
- `docs/20261010-o2-review-and-project-direction-opus.md`：若开展科研/展示型新项目，应另立版本，先用外部生物依据选参数，并预先冻结表征、基线、预算与停止门。
- `docs/model.md`：连接组提供解剖而非完整生理；现有动力学、感觉代理、奖励输入、记忆规则和交易解码包含大量工程假设。

### 2.2 相关实现与测试

| 现有内容 | 路径 | 评审结论 |
|---|---|---|
| MaleCNS 导入、来源 SHA、整数 ID、保留/排除审计 | `stonkfly/neural/connectome.py` | 可复用来源与审计框架；不得改写 v1 数据目录或锁文件 |
| CSR 编译、接触数和递质符号映射 | `stonkfly/neural/prepare.py` | CSR 构建可抽取；`contact × 0.275` 和递质到统一正负号的生理解释不可复用 |
| 状态形状、有限值、允许 `dt` 检查 | `stonkfly/neural/state.py` | 可复用验证模式；v2 应使用独立状态类和独立模型签名 |
| 原生 LIF、事件队列、延迟、适应、trace observers | `stonkfly/neural/kernel.cpp` | 队列、观测账本和边界检查思路可复用；动力学方程及常数不能视为已验证生理 |
| 隔离的 v2 wrapper、独立编译产物、跨 `dt` 与 dense-reference 资格思路 | `stonkfly/neural/v2_brain.py`、`stonkfly/neural/v2_q2_brain.py` | 模块隔离、编译 provenance、状态哈希可复用；视觉 graded kernel 不属于本实验 |
| MaleCNS 中 KC/MBON/DAN 的固定识别 | `stonkfly/neural/circuit.py` | 细胞身份审计和“只选原有边”思路可复用；α1/γ1pedc 两区室和 DAN-contact gain 不是通用学习模型 |
| baseline-centered anti-Hebbian rate rule | `stonkfly/neural/rule.py` | 数值闭式更新、冻结权重和时间顺序单测可参考；常数、双区室迁移和精确规则必须重写/重新声明 |
| O0 PN inventory | `experiments/a_share/run_o0_olfactory_pn_inventory.py`、`tests/test_o0_olfactory_pn_inventory.py` | ORN→PN→KC 结构筛选、集合/边哈希可复用；“290 个 eligible PN”和 8-PN hash panel 只属于 O0，不是 v2 气味面板 |
| O1/O2 单事件、退出、重复、跨 `dt`、零输入和非覆盖产物链 | `experiments/a_share/run_o1_pn_single_event.py`、`experiments/a_share/run_o2_pn_normalized_dose.py` 及对应测试 | 测试模式和 raw/formal/report/manifest 链可复用；v1 剂量、刺激注入语义和结果不得用于选 v2 参数 |
| 可塑性与 checkpoint 测试 | `tests/test_neural.py`、`tests/test_neural_linked_reinforcement.py` | frozen/shuffled、保存恢复、时间特异性测试模式可复用；交易反馈、旧 decoder 和 full-graph 成绩不可复用 |
| 跨 `dt` 与资格测试 | `tests/test_r0c_dt_convergence.py`、`tests/test_v2_q2_qualification.py`、`tests/test_v2_q2_protocol_repair.py` | 步数、重复、dense reference、verdict precedence、失败产物和 output protection 可复用 |

### 2.3 必须新建或重写

建议全部放入新的命名空间，例如 `stonkfly/mb_lab/` 和 `experiments/mb_lab/`，不修改 v1 归档、v1 runner 或 v1 模型常数。

| 新组件 | 建议路径 | 原因 |
|---|---|---|
| 只读蘑菇体 inventory 与左右侧/区室归属 | `stonkfly/mb_lab/inventory.py` | O0 的 PN proxy 规则过宽；需审计真正的嗅觉 PN、KC、APL、目标 MBON/DAN 及边界边 |
| 版本化子图编译器 | `stonkfly/mb_lab/subgraph.py` | 同时产生原始诱导子图、约束简化图和度数匹配随机图；输出节点/边/接触哈希和损失账本 |
| 合成气味生成器 | `stonkfly/mb_lab/stimuli.py` | 以 glomerulus/PN channel 为单位，固定训练集、独立测试集、噪声和种子；完全不依赖市场 |
| 独立动力学 | `stonkfly/mb_lab/dynamics.py`（先用 NumPy reference） | 不继承 v1 的全图 LIF 参数；先用可解释、小图、确定性 reference，必要时再写原生 kernel |
| APL 正规化/反馈抑制 | `stonkfly/mb_lab/apl.py` | APL 是局部且气味相关的正规化机制，不能只复制全局常数抑制或 v1 `APL×k` 扫描 |
| 区室化 KC→MBON/DAN 三因子规则 | `stonkfly/mb_lab/plasticity.py` | 固定规则、可审计 eligibility、权重界限和权重冻结；与交易 reinforcement 完全隔离 |
| 固定研究读出 | `stonkfly/mb_lab/readout.py` | 只报告 MBON 差分/分类，不产生 BUY/SELL/HOLD，更不产生订单 |
| 表征和反转协议 runner | `experiments/mb_lab/run_representation.py`、`run_reversal.py` | 预注册输入、预算、verdict、失败证据和不可覆盖输出 |
| 独立测试 | `tests/mb_lab/` | 小图单测、dense reference、跨 `dt`、对照、泄漏、哈希和停止规则 |

第一版不应从 `MemoryBrain` 继承，也不应导入 `stonkfly.market`、`stonkfly.reinforcement`、`stonkfly.actions`、`stonkfly.broker`、`stonkfly.risk` 或任何 controller/decoder 交易路径。测试应显式阻止这些 import。

## 3. 独立生物依据与证据等级

证据标签：

- **V（validated constraint）**：来自成人果蝇连接组、在体成像、电生理或干预研究，足以约束结构或定性机制；不表示给出了本模型全部数值。
- **P（proxy assumption）**：MVP 为可计算、可证伪而预先声明的工程值；不可称为测得的果蝇参数。
- **M（missing）**：当前没有找到可直接映射到 MaleCNS v1 这个子图与所选状态方程的独立数值。

| 项目 | 依据与拟采用方式 | 等级 |
|---|---|---|
| 数据来源 | MaleCNS v1.0 是成人雄蝇 CNS 的版本化 EM 连接组；项目页和下载页提供 annotations/connectivity。继续使用现有 source hashes，不重新下载或覆盖。([MaleCNS project](https://male-cns.janelia.org/), [download](https://male-cns.janelia.org/download/)) | V（解剖） |
| MB 规模与基本流向 | 成人单侧 MB 约 2,000 KC；约 51 个嗅觉 glomeruli 的少量 PN 投射至 calyx；KC 平均约 6 个 claw；KC→MBON，DAN 与 MBON 在区室中相交。([adult MB connectome, 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7909955/)) | V（解剖） |
| PN→KC 汇聚 | 成人嗅觉 KC 常见 3–10 个 claw，每个 claw 接受一个 PN bouton，KC 需要多个输入共同活动才发放。([visual inputs to KCs, 2024](https://www.nature.com/articles/s41467-024-49616-z)) | V（范围/机制） |
| KC 稀疏度 | 在体成像报告单个气味平均激活约 6% KC，并且对混合物、自然气味和浓度变化仍保持稀疏。([cellular-resolution population imaging](https://pmc.ncbi.nlm.nih.gov/articles/PMC3180869/)) MVP 表征通过带设为 2–10%，核心目标带 4–8%。 | V（现象）；具体门限 P |
| APL 作用 | KC↔APL 负反馈维持稀疏、降低气味间相关；破坏回路会降低稀疏性并损害相似气味辨别。([Lin et al., 2014](https://www.nature.com/articles/nn.3660)) Calyx 数据进一步支持 APL 对 PN:KC 层的正规化和气味特异局部活动，而非简单全局均匀抑制。([APL normalization study, 2022](https://elifesciences.org/articles/74172)) | V（定性机制） |
| 区室结构 | 成人 MB 的 KC、约 20 类 DAN 和 21 类 MBON 形成区室化学习单元。([Aso et al., 2014](https://pubmed.ncbi.nlm.nih.gov/25535793/)) | V（解剖/功能组织） |
| KC→MBON 可塑性 | 气味与特定 DAN 激活配对可产生气味特异的 KC→MBON 抑制，依赖刺激时序并局限于解剖区室；不同气味表征重叠影响泛化。([Hige et al., 2015](https://pubmed.ncbi.nlm.nih.gov/26637800/)) | V（机制方向和时序依赖） |
| 反转/更新 | 惩罚省略可激活 reward-encoding DAN，并参与果蝇反转学习；说明真实反转包含预测误差和回路反馈，而不只是把外部奖励标签改号。([Felsenberg et al., 2021](https://www.nature.com/articles/s41467-021-21388-w)) | V（反转机制）；MVP 外部 DAN pulse 是 P |
| 可塑性多时间尺度 | PPL1 DAN 可双向编码先天/学习 valence；互联的短、长时记忆单元参与动态记忆。([Huang, Luo et al., 2024](https://www.nature.com/articles/s41586-024-07819-w)) | V（现象）；现有 `rule.py` 的具体常数和双区室迁移仍是 P |
| 单细胞 `tau_m`、阈值、reset、refractory | 文献并未提供一组能直接覆盖 PN、KC、APL、MBON、DAN 且可与 MaleCNS 接触数逐边映射的统一参数。不得沿用 v1 的 20/5 ms、−45 mV、2.2 ms 等作为“已验证”。 | M |
| 接触数→电导/电流 | EM contact count 不是突触效能；缺少按 pre/post cell type、受体、位置和短时可塑性校准的转换。 | M |
| APL 数值增益与空间核 | 已验证其正规化、反馈和局部性，但没有一条可直接用于本离散子图的唯一增益/空间衰减常数。 | M；预注册单一正规化形式为 P |
| DAN 浓度、eligibility、学习率、权重上下界 | 区室化、时序依赖和 depression 有实验支持；精确 trace、学习率和边界无法由这些研究唯一确定。 | M；MVP 固定值为 P |
| 合成气味到 PN 发放率 | 合成向量是实验输入，不对应具体化学气味或真实 ORN/PN 剂量反应。 | P |

严禁把以下 v1 数值当作独立依据：O1/O2 的剂量档、点燃边界、`APL×2`、KC 高态比例、R0c 的失败区间、任何市场训练的权重变化或盈亏。

## 4. 三类模型臂

### A. 原始 MaleCNS 蘑菇体诱导子图（高保真困难臂）

定义：在一个半球内，以结构规则选出嗅觉 PN、全部相应 KC、APL、预先选定的 MBON/DAN 区室；保留这些节点之间 MaleCNS 的全部原始有向边、自边和接触数，并为所有被切断的跨边建立边界损失账本。拓扑不按结果剪枝；动力学参数仍来自独立声明，而不是 v1。

- 科学意义：检验已重建的局部拓扑在明确简化的生理假设下是否足够；最接近“真实连接组约束”。
- 限制：诱导子图切掉 DPM、MBON↔DAN、其他 MB 输入输出和全脑状态，不能称为完整 MB；contact count 仍非电导。
- 工程成本：高。需要 cell identity、侧别、区室、边界边和 sign 的逐项审计；更容易因缺失生理参数而失败。
- 判定角色：不得成为唯一成功路径。它可以失败而 B 臂仍继续，但失败必须保留并解释为“当前原始子图参数化不合格”。

### B. 独立参数化的生物约束简化模型（MVP 主臂）

定义：保留真实 PN→KC adjacency（或 claw-level 二值化）、真实 KC→目标 MBON adjacency 和实际节点身份；APL 实现为明示的局部 divisive normalization/反馈抑制；DAN 是区室化外部教学输入；排除未建模的旁路和全脑递归。所有状态方程和代理参数在看正式输出前冻结。

- 科学意义：回答“这些受生物结构约束的计算原则是否足以产生目标现象”；可解释、可证伪。
- 限制：成功只说明工程模型具备这些性质；不能证明 MaleCNS 原始回路或真实果蝇以该方程工作。
- 工程成本：中。NumPy reference 足以完成首轮，不需要立刻写 C++。
- 判定角色：MVP 的主要 GO/NO-GO 依据。

### C. 随机稀疏网络对照

定义：保持 PN/KC/MBON 数量、每个 KC 的 PN 入度、每个 PN 的 KC 出度分布、KC→MBON 入度、总权重分布和 APL/学习规则不变；用固定 seed 做二分图 degree-preserving rewiring。至少 20 个预注册 seeds，不挑最好 seed。

- 科学意义：区分“真实拓扑贡献”与“任意稀疏随机扩展 + 同一规则”即可产生的效果。
- 限制：若 B 与 C 相当，只能说通用稀疏扩展有效，不能声称真实连接组结构带来优势。
- 工程成本：低到中；主要成本是严格的匹配检验、seed 哈希和多重复预算。

三臂结论矩阵：

| 结果 | 可接受解释 |
|---|---|
| A、B 通过，且优于 C | 值得继续研究真实拓扑是否贡献了额外结构；仍不是生物学验证 |
| B、C 通过且相当，A 失败 | 通用稀疏扩展工程模型可学习；没有真实拓扑优势证据 |
| 仅 B 通过 | 简化假设可能是成功关键；需要新的独立实验约束，不能生物外推 |
| 仅 C 通过 | 真实/约束结构没有显示优势；MVP 科学目标 NO-GO |
| 三臂均失败 | 停止，不调参追点 |

## 5. 冻结的合成刺激与表征实验

### 5.1 刺激集

在实施前生成并哈希两个不重叠集合：

- `qualification`：16 个合成气味，只用于一次正式表征资格判定；
- `heldout`：16 个独立合成气味，只在实现、参数、runner 和资格判定规则全部锁定后打开；
- 每个集合含 4 个相似度家族，每族 4 个气味；以 PN/glomerulus channel 的 cosine overlap 预先形成低、中、高相似度，不按 KC 输出挑选；
- 每个气味激活 8–12 个 glomerular channels，强度从预先固定的有界分布抽样；这是 P，不是化学气味模型；
- 每个气味 10 次重复，PN 输入使用固定的 Poisson/jitter seeds；另含 ZERO、单 channel、全 channel 上限和 APL-off 正控制；
- 时序固定为 1 s baseline、1 s stimulus、2 s withdrawal。任何更改都形成新预注册版本，不能覆盖旧结果。

开发 smoke 只能使用手工 4-PN/32-KC toy fixture，检查文件、数学和产物链；不得查看正式气味集输出后修改参数。

### 5.2 指标和通过标准

以下阈值是预注册的工程门，不冒充生物测量误差界限：

1. **有限与稳定**：所有状态有限、无 queue/计数溢出；ZERO 不产生受刺激活动；任何 trial 在 withdrawal 最后 500 ms 的 KC 活动比例 ≤1%，且不得出现持续增长或非衰减振荡。
2. **稀疏性**：每个正式气味在 stimulus 窗内激活 2–10% KC；所有气味的中位数在 4–8%。任一气味 >20% 或为 0 直接失败，不以平均值掩盖。
3. **试次稳定性**：同气味重复的 KC 二值 active-set Jaccard 中位数 ≥0.60，population-rate cosine 中位数 ≥0.80；每个气味至少 8/10 重复满足。
4. **特异性/分离**：同气味 Jaccard 中位数必须比不同家族气味高至少 0.25；不同家族的中位 Jaccard ≤0.30；输入相似度与 KC 相似度关系须单调但不能塌缩为同一 active set。
5. **APL 因果正控制**：固定 APL-off 必须使 KC 中位激活比例上升至少 50% 或跨气味相关上升至少 0.15；同时正常 APL 臂仍满足 1–4。该测试只支持实现的抑制机制生效，不验证真实 APL 增益。
6. **撤刺激恢复**：刺激结束 2 s 时，KC 活动回到 ZERO baseline 的绝对差 ≤1 percentage point；内部连续状态相对 baseline 的归一化 L∞ ≤5%，事件队列清空。
7. **跨 `dt`**：正式使用 `0.1, 0.05, 0.025 ms`；每气味 active-set 在相邻 `dt` 间 Jaccard ≥0.90，KC 总 spike/rate 相对差 ≤10%，所有 pass/fail 分类完全相同。`0.05/0.025` 是数值收敛检查，不是额外参数选择机会。
8. **独立测试集**：B 臂须在 qualification 和 heldout 都通过 1–7；C 臂报告全部 20 seeds 的分布；A 臂独立报告，不允许由 B 的输出改参数。

若 rate/threshold 模型没有 spike queue，则第 1、6 条中的 queue 条件记为 `not_applicable`，但不能伪造通过；必须用该模型的所有隐藏状态做恢复检查。

## 6. 冻结的奖励反转学习实验

只有 B 臂通过完整表征门后才允许进入；A、C 使用同一刺激、读出和学习规则做比较，不为各臂单独调参。

### 6.1 协议

- 从 heldout 中预先按输入相似度分层选择 8 对气味，不按 KC 或 MBON 输出选择；4 对 acquisition/reversal，4 对完全独立 replication。
- 每对先做 10 次无强化 baseline，确认无预存 valence 偏置；固定读出是两个相反 valence MBON population 的归一化差，不映射为交易动作。
- acquisition：A 与正 DAN pulse 配对，B 与负 DAN pulse 配对，最多 20 个交错 trial；cue 与 DAN 的相对时序在协议中固定。
- reversal：不重置权重，交换 A/B 的 DAN pulse，再做最多 20 个交错 trial。
- retention：10 个无 DAN probe；probe 不更新权重。
- 每次 trial 保存 cue、DAN、eligibility、每条可塑边前后权重、MBON 输出和全部 seeds；学习只能修改声明的 KC→MBON 边。

真实果蝇反转涉及 prediction-error、MBON↔DAN 回路和省略信号。本 MVP 直接提供 DAN pulse，因此实验名称必须是 **externally instructed reward reversal proxy**，不能声称复制真实反转回路。

### 6.2 必需对照

| 对照 | 必须保持相同 | 预期用途 |
|---|---|---|
| frozen weights | 所有刺激、噪声、初始权重、读出 | 排除动态状态或固定偏置造成的假学习 |
| shuffled DAN | DAN pulse 数量和幅度，打乱 cue 配对 | 检验关联特异性 |
| unpaired timing | cue/DAN 总量相同但超出 eligibility 窗 | 检验时间关联 |
| DAN omitted | 只有 cue | 排除纯暴露/适应 |
| label swap | 正负 MBON/DAN 标签交换 | 检查代码符号与 verdict 对称性 |
| APL-off | 只作机制对照，不参与主 verdict | 测试稀疏性丧失对特异学习的影响 |
| degree-matched random | C 臂 20 seeds | 判断真实拓扑是否优于通用稀疏扩展 |

### 6.3 学习通过标准

1. baseline 的绝对 MBON valence score 中位数 ≤正式满量程的 10%，A/B 无系统偏置。
2. acquisition 的最后 5 个 probe 中，至少 4/5 对 A/B 给出正确相反排序；8 个气味对中至少 7 对通过。
3. reversal 后同一标准必须翻转；不得重置动力学或权重。达到标准的 trial 数中位数 ≤16，任何 pair 超过 20 记失败。
4. retention 的 10 个无强化 probe 中至少 8 个保持反转后的排序，且权重在 probe 中逐位不变。
5. frozen、shuffled、unpaired、DAN-omitted 均不得达到主臂标准；若任一达到，判 `CONTROL_FAILURE`，不是“弱阳性”。
6. cue-specific 可塑边变化必须与 cue-active KC 有关；inactive KC→MBON 边的绝对变化中位数 ≤ active 边的 10%。
7. 三个 `dt` 的 acquisition/reversal verdict 完全一致，达到标准的 trial 数差 ≤2。
8. replication 四对必须满足与 acquisition/reversal 四对相同的预先标准；不能合并后用总体均值掩盖失败 pair。

这些阈值只定义工程学习成功。除非未来加入真实气味、真实 PN/KC/MBON 活动数据拟合、跨个体连接组和预注册的生物预测，否则不支持“模型复制了果蝇反转学习”。

## 7. 实施前必须冻结的章程

人工批准后、任何 formal run 前，单独提交一个不可变预注册，至少锁定：

1. MaleCNS 文件、annotations、节点集合、边集合和边界损失账本 SHA256；
2. 三臂生成规则、全部随机 seeds 和 degree-matching 检验；
3. 所有方程、单位、初始状态、参数、证据标签和模型签名；
4. qualification/heldout 列表哈希，在 formal 前不得由 runner 输出反解；
5. 指标实现、聚合顺序、缺失值规则、verdict precedence；
6. 代码、测试、compiler/environment、runner 和预注册 hashes；
7. raw JSON/NPZ、formal JSON、report、manifest、launch、stdout 的完整产物链；
8. 明确 `market_used=false`、`trading_policy_used=false`、`execution_used=false`、`production_eligible=false`。

只允许先运行一个极小 CLI smoke，且它必须产出上述每种文件。之后冻结 runner；正式目录存在或非空时拒绝覆盖。实现 bug 可以修复并升版本，但旧失败目录保留，新版本从头运行全部臂和对照；不能只补跑有利部分。

## 8. 最大计算预算与停止条件

### 8.1 硬预算

- 开发单测和 toy smoke：累计 ≤4 CPU-hours；
- 静态 inventory/subgraph 审计：≤4 CPU-hours、峰值 RAM ≤16 GiB；
- 三臂表征（含 3 `dt`、重复和 20 random seeds）：≤48 CPU-hours；
- 反转学习（含全部对照、3 `dt` 和 replication）：≤48 CPU-hours；
- 总 formal 预算：**≤100 CPU-hours、峰值 RAM ≤32 GiB、产物 ≤20 GiB、单阶段 wall time ≤24 h、无 GPU**；
- 每个冻结版本只允许一次完整 formal run。预算耗尽即 `BUDGET_EXHAUSTED`，不得缩减失败 trial 或增加预算后沿用同一预注册 verdict。

这不是长时间正式计算的授权。未来若获准且预计超过 30 分钟，应按仓库规则在 tmux 中运行并保留进程、日志和不完整目录。

### 8.2 停止条件

按优先级：

1. provenance/hash、对照泄漏、非有限值、输出覆盖或确定性失败 → `IMPLEMENTATION_FAILURE`，停止；
2. B 臂 qualification 表征失败 → `REPRESENTATION_NO_GO`，不开 learning；
3. B 臂 heldout 表征失败 → `GENERALIZATION_NO_GO`，不开 learning；
4. 学习任一必需对照失败 → `CONTROL_FAILURE`，停止；
5. acquisition 失败 → `ACQUISITION_NO_GO`，不解释 reversal；
6. reversal 或 replication 失败 → `REVERSAL_NO_GO`；
7. 仅 C 通过、或 B 不优于 C 且研究问题要求真实拓扑优势 → `BIOLOGICAL_TOPOLOGY_NO_GO`；
8. 预算耗尽 → `BUDGET_EXHAUSTED`。

失败后允许：修复可独立证明的代码 bug；或在新的文献/实验数据出现后另立 v2.1。失败后不允许：追加剂量点、扫描 APL gain/threshold/学习率、删除困难气味、改 seed、改变聚合口径或放宽门槛。

## 9. 结果能支持什么

### 9.1 只支持工程模型学习

满足下列任一情况，结论最多是工程层面：

- B 或 C 在合成输入和外部 DAN label 下通过反转；
- 参数含代理阈值、代理 APL normalization 或人工 learning rate；
- 只在单个 MaleCNS 个体、单一半球或单一 MBON/DAN 区室上成立；
- 原始子图 A 失败而简化 B 成功；
- B 与 degree-matched C 无显著差异。

合适表述：“冻结的、受蘑菇体结构约束的工程模型在合成任务上展示了关联更新/反转。”

### 9.2 才可能支持有限生物学解释

需要在本 MVP 之外同时满足：

- 参数由独立电生理/成像校准，并明确到对应细胞类型、受体和实验条件；
- 模型在未用于定参的真实 PN、KC、APL、MBON/DAN 活动数据上预测时序和幅度；
- APL、DAN、特定 KC→MBON 边的 in-silico 干预方向与独立生物干预一致；
- 跨多个个体/连接组版本复现，并优于 degree-matched 随机网络和更简单模型；
- 在运行前提出并随后被实验验证的新预测。

即便如此，也只能支持具体回路机制的有限解释，不能支持意识、主观体验或交易能力结论。

## 10. 最小工作量与主要风险

### 10.1 人工工作量

在不写优化 C++、不运行正式实验的前提下，MVP 预计 **12–18 个工程人日**：

- 2–3 日：inventory、侧别/区室核验、三臂子图和边界损失账本；
- 3–4 日：NumPy reference 动力学、APL、单位和 dense-reference 测试；
- 2–3 日：合成刺激、资格指标、heldout sealing；
- 3–4 日：区室化可塑性、固定读出、反转及对照；
- 2–4 日：CLI、产物链、预算/停止逻辑、端到端 smoke 和人工审阅材料。

若 reference 太慢而需要原生 kernel，再增加约 4–7 日；必须先让 reference 成为 oracle，不能直接以优化内核作为唯一实现。

### 10.2 风险

| 风险 | 等级 | 缓解 |
|---|---|---|
| contact count 无法识别功能权重 | 高 | B 臂二值/归一化拓扑；A 臂单独报告；不按输出拟合 |
| APL 局部性与增益不可唯一确定 | 高 | 固定单一代理方程，APL-off 只作因果实现控制；失败即停 |
| 学习规则和时间常数缺少一一对应实测 | 高 | 将反转定位为 external-instruction proxy；不作生物复制声明 |
| MaleCNS cell type/侧别/区室映射错误 | 中高 | 静态 inventory、人工抽查、集合/边哈希、与发布注释交叉核对 |
| 真实拓扑并不优于随机稀疏网络 | 中高 | 这是有意义的否定结果，不通过调参掩盖 |
| `dt` 一致但方程仍非生物 | 中 | 明确数值资格不等于生理有效性 |
| 多重比较与 seed 选择 | 中 | 预先固定所有 seeds、逐 seed 报告、禁止挑选最好结果 |
| v1/v2 路径污染或误接交易模块 | 低到中 | 独立 namespace、import deny-test、所有 formal 标记 false |

## 11. 建议的人工决策

本节是初版意见，已被 §0 和 `docs/STONKFLY_V2_M1_CHARTER_DRAFT.md` 取代。当前不授权 M2 实现；只授权 M1 实现准备和 BENCHMARK_ONLY smoke。

最终项目判断：

> **当前控制性结论：GO — M1 实现准备；NO-GO — M1 formal、M2、继续 v1、行情接入、交易训练、实盘及未经独立数据校准的生物学强结论。**
