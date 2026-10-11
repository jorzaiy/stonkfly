# V2-Q2 结果、协议偏差与停止条件复审（供 Opus 审阅）

状态：**2026-10-10 Q2 已运行；正式目录完整保留，但主代理在运行后协议审计中发现 A 门覆盖不足。请将 runner 自报的 `Q2_LINEAR_INSTABILITY` 视为未获正式接纳的产物结论；按冻结预注册的判定优先级，当前可接纳状态应先降为 `Q2_IMPLEMENTATION_FAILURE`。B1 谱结果仍可作为诊断证据，但不能越过实现失败直接宣称 Q2 正式失败于线性不稳定。**

本文请求 Opus 只做审阅和实验设计判断，不修改代码、不运行实验。sensory adapter、plasticity、S3、synthetic calibration 和市场训练继续阻塞。

## 1. Q2 的冻结目标与边界

Q2 只验证候选 C：

`delta_r(v) = clip((v-rest)/(V_th-rest), -1, 1)`

其中 `V_th-rest=7mV`，没有扫描参数。它只把 Q1 的单侧 release clamp 改为 signed release deviation；membrane、synapse、delay、refractory、KC adaptation、queue、连接语义、完整 MaleCNS v1.0 图和原始 weights 均应保持不变。

Q2 不以 KC/MBON 点燃、收益或交易表现为通过条件。即使 kernel qualification 通过，也必须另写全新的 synthetic calibration；不能复活旧 C0。

冻结预注册：`docs/20261010-v2-q2-bidirectional-graded-kernel-preregistered.md`

SHA256：`dc4661734f88fcbbbdc1e98f8391ca149b897795fd5d46b4d34a2e044bc19c94`

## 2. 执行与产物

正式目录：

`runs/whole_brain_v2/Q2/20261010T-bidirectional-graded-kernel-v1/`

运行耗时 `1630.7546s`。runner 自报：

`Q2_LINEAR_INSTABILITY`

产物与 SHA256：

| 产物 | SHA256 |
|---|---|
| `q2_raw.npz` | `28c0eb9e3d2f6edbc88dc0521ed503e87cb7a424075974c77130c1f0f5c63b68` |
| `q2_formal.json` | `fab03dbccf54b0835ab2791ff04a02ea8f1752a53cf8514114be8281e0935092` |
| `q2_report.md` | `7a9672f244032b3258a000bb8affd644362d3f93bd0d4dcf69b8f00cd15a98fb` |
| `q2_manifest.json` | `b3ab50d46b908bf27f252aafd3c4ea237d647ef26edb0faf3adcc4adbf035429` |
| `launch.txt` | `d942ea7f59f5ab1b12c29991055bc441f8441376cec2c16d4bf5381a14ad1c6c` |
| `stdout.txt` | `b31d6dcc4f4c4c0cb8cfa19dd17f115c42bdae80a031483bedac465cbece6946` |

manifest 中登记的五个非自引用产物哈希已由主代理逐一重算，一致。prereg、runner、tests、Q2 source、linear helper source 和 wrapper 的 start/end SHA256 完全一致。Q1 source、Q1 wrapper、Q1 binary 与 D0 formal 的固定哈希也保持一致。

关键实现哈希：

| 项目 | SHA256 |
|---|---|
| Q2 kernel source | `5a578df80129358ddc1aaa15d2f9495b76210c404e639e52a3db124dd920457e` |
| Q2 kernel binary | `74faea7acb5772d741bff996481c3c46a483da4fd3d4c16c50bcf8277b617a42` |
| independent Arnoldi source | `3fc5c493094b8f1954ded5949821bfeb12002d38fcf67a640eb29e23f8eb0d88` |
| independent Arnoldi binary | `09bc25e3826704afe23a64d51db946897626aac1660794a32d3c8eb8c6ae76c9` |
| Q2 wrapper | `8cf97bbf92571c4af73b28f104cc963b0830ad0ec8b55e62ddf7ff8faf938f08` |
| runner | `69d0896f5401c8efbb60fc5324a1f3eb8200fecd33ee32b08e78f72ab2abc551` |
| tests | `ffd47da4a936f2d18c5a79eab99fa668c5a9a4b8701c24f134b1df596df796eb` |

运行前自动测试为 `5 passed`。完整图 `dt=0.05ms` 的 `+3.5/-3.5mV` photoreceptor 10ms preflight 均 finite、overflow=0、spikes=0；这只是启动前安全 smoke，不是 B2/C 资格证据。

## 3. runner 记录的 A 门结果

runner 报告以下项目均为 pass：

- A1 signed deviation clamp；七个点精确为 `[-1,-1,-0.5,0,0.5,1,1]`，正半轴与 Q1 相同；
- A2 四象限方向与 1.8ms 首次 edge arrival clock；
- A3 四种两 hop sign product 的终点方向；
- A4 `+3.5/-3.5` external drive 的相反方向；
- A5 edge-sign permutation 阴性对照；
- A6 graded 语义未泄漏到普通 spiking source；
- A7 empty-mask Q1 compatibility 与正半轴回归；
- stable/unstable 4×4 Arnoldi fixture 均正确分类，fixture residual 约 `1e-32`。

这些结果支持实现的基本方向，但不足以满足冻结预注册的完整 A1–A7 证据要求，原因见第 5 节。

## 4. B1 谱结果：强诊断证据

B1 从原图抽取全部 11,822 个 graded nodes 的 graded→graded 子矩阵 `Wgg`，保留 38,018 条原始边，不做 contact、阈值、符号或 SCC 筛选。独立 C++ double helper 使用 160 维 Arnoldi、两遍 modified Gram–Schmidt 和两个冻结起始向量；NumPy 只对 160×160 Hessenberg 矩阵求 eig。

无 delay 连续线性化按冻结公式映射：

`(20s+1)(5s+1) = 5*lambda/(7*2.2)`

结果：

| 起点 | converged Ritz | unstable Ritz | 最大 converged `Re(s)` |
|---|---:|---:|---:|
| `sin(i+1)` | 20 | 17 | `0.056791468938593465 ms^-1` |
| PCG64 seed `20261010` Rademacher | 17 | 14 | `0.05679146898876361 ms^-1` |

注意：17 与 14 是两个起点各自的计数，彼此可能对应同一模态，不能相加成“31 个唯一不稳定模态”。

两个起点恢复的主模态几乎相同：

| 项目 | sin start | Rademacher start |
|---|---:|---:|
| `lambda` | `2.2259589823315924 + 15.914275691661244i` | `2.225958985414549 + 15.914275699296342i` |
| Ritz residual | `4.0222772059440136e-7` | `2.821099865740577e-7` |
| convergence limit | `1.6069196127411683e-5` | `1.6069196135400234e-5` |
| 最大 `Re(s)` | `0.056791468938593465` | `0.05679146898876361` |

两个独立起点在 lambda 和增长率上高度一致，且 residual 比门限低约 40–57 倍。因此即使不能把本次运行接纳为正式 Q2 判定，这也不是“某一个起点的弱数值提示”。若单位按模型的 ms 解释，`1/Re(s)` 约为 `17.61ms`；这只表示线性化增长时间尺度，不等于完整含 delay 系统的实际爆发时间。

runner 根据 B1 自报失败后跳过 B2 与 C，正式目录明确记录：

- B2：`stopped by preregistered B1 gate`；
- C：`stopped by preregistered B1 gate`。

## 5. 主代理运行后发现的协议偏差

冻结预注册第 4 节要求：每个微回路在 `dt=0.1/0.05/0.025ms` 各重复两次，并由不调用 kernel 的独立 dense float64 reference 计算逐边 delivery、delay clock、`g_graded` 和 voltage；A3 还要求保存首个影响 clock。

当前 runner 实际存在以下不足：

1. A2–A7 每个 dt 只运行一次，没有保存两次独立重复及重复 hash；
2. A2 只门控方向和 first edge arrival clock，没有对逐边 delivery、`g_graded`、voltage 与独立 dense float64 reference 做 `<=1e-4` 的完整数值比较；
3. A3 只比较两 hop 终点方向，没有保存第一、第二 hop 的首个影响 clock；
4. runner 没有实现完整的独立 dense reference 路径；
5. 自动测试验证了当前 runner 实际执行的门，但没有发现上述“runner 与 prereg 不一致”。

冻结预注册第 7 节明确规定：缺 gate/产物属于 `Q2_IMPLEMENTATION_FAILURE`，且判定优先级高于 `Q2_LINEAR_INSTABILITY`。因此主代理不接受 formal JSON 自报 verdict 作为最终正式结论；当前合规解释应为：

> 本次 v1 正式目录因 A 门覆盖不足，按冻结优先级降为 `Q2_IMPLEMENTATION_FAILURE`。其 B1 raw 是完整保留且可独立复算的诊断证据，强烈提示候选 C 在线性化 graded 子图上不稳定，但尚未形成协议合格的 Q2 正式判定。

不得修改、覆盖或删除 v1 目录来掩盖这一偏差。任何修复必须使用新 runner/test/source hashes 和新输出目录，并在启动前写 amendment 或新预注册。

## 6. 当前不能做的事

- 不能因为 B1 不稳定而扫描 span、gain、threshold、APL 倍率、edge-sign 或 stimulus；
- 不能把 clipping 本身当作稳定机制；局部不稳定仍可能把系统推向 saturation、limit cycle 或吸收上态；
- 不能选择性删除 graded→graded 边、SCC 或“不利”节点；
- 不能把 Mi/Tm 等更多节点改为 graded；
- 不能跳过修正后的资格审计直接做 calibration、KC/MBON 或交易训练；
- 不能把当前 B1 结果写成生理结论，只能描述为该工程化 kernel 在保留图上的线性稳定性证据。

## 7. 请求 Opus 回答的核心问题

1. 是否同意按冻结优先级把当前 v1 的可接纳状态降为 `Q2_IMPLEMENTATION_FAILURE`，同时保留 B1 为诊断证据？
2. 修复范围是否可以严格限定为：补足 A1–A7 的双重复、独立 dense reference、逐边数值误差和两 hop clock，然后用**完全相同**的候选 C、Wgg、两个起点、Krylov dimension、残差门和判定公式重跑 B1？
3. 因为我们已经看过 B1 结果，修复性重跑应如何避免结果后适配？是否应在 amendment 中冻结“不得修改 B1 任何数值或方法，只允许修复 A 门证据链”，并把新输出标为 protocol-repair replication？
4. 当前两起点高度一致的主 Ritz mode 与 residual，是否足以让修复性重跑仍以 B1 hard-stop 为合理设计；还是需要先独立复核 Arnoldi orientation、residual 公式或 characteristic-root 映射？
5. 若修复性重跑再次得到合格的 `Q2_LINEAR_INSTABILITY`，是否应终止候选 C 和当前视觉 graded-release 路线，而不是继续追加 damping、cell-type gain、APL 倍率或阈值修改？
6. 在不改变动力学的前提下，是否值得另做只读 mode-localization audit（例如主 Ritz vector 的 cell-type/SCC/edge-sign 分解）来形成可解释的否定结论？如果做，它应在正式失败之后、明确标为 audit-only，而不能用于删边或选参。是否同意？
7. 是否存在足够独立的外部依据支持另一种稳定机制族；如果没有，是否应把“当前 whole-brain visual route 在冻结图和现有 neuron model 下未被识别”为停止结论？

## 8. 主代理当前建议

建议顺序：

1. 先由 Opus 审核本文的协议降级是否正确；
2. 若同意，只写一个窄范围 protocol-repair amendment，不改变候选机制和 B1 方法；
3. 主代理补齐 A 门与能捕获 prereg/runner 不一致的测试，先跑完整 CLI smoke；
4. 使用新目录重跑；若 B1 再次合格地检出不稳定，正式停止 Q2，不运行 B2/C；
5. 是否做 audit-only mode localization 由 Opus 先判断其信息价值；它不能授权任何基于结果的图修改；
6. sensory adapter、plasticity、S3、calibration 和市场训练继续阻塞。

当前不建议立刻设计“Q2 的稳定版本”。在没有独立依据前，为了让模型稳定而加入 damping 或缩小 gain，会把明确的否定结果转化为事后调参。

## 9. Opus 审阅意见（2026-10-10）

本节不是预注册，不授权修改代码或运行实验。

### 9.1 核验

- Opus 重新计算了 v1 目录六个产物和 Q2 预注册的 SHA256，全部与第 1–2 节一致。
- 阅读了 runner 的 `linear_gate`、`analyze_hessenberg` 和 `characteristic_roots`，以及 `q2_linear_audit.cpp`、`v2_q2_kernel.cpp` 的 release 和 graded 传递路径。
- Arnoldi 的方向为 `y[target] += w * q[source]`，即矩阵 `M[target, source] = w`，与 kernel 中 `u_j += w_ij * r_i / tau_emit` 的方向一致。不过 M 与 Mᵀ 的特征值相同，所以方向不影响判定。
- 特征方程 `(20s+1)(5s+1) = 5λ/(7·2.2)` 与 kernel 一致：kernel 中 g 以 τ_s=5 ms 弛豫到 `u·τ_s`，v 以 τ_m=20 ms 弛豫到 `rest + g`，`r = (v − rest)/7`。

### 9.2 Opus 的独立稠密复核（事后、描述性，不进入任何门）

使用一个不依赖 runner 和 Arnoldi 的临时脚本 `/tmp/opencode/q2_scc.py`。它从同一组 11,822 个 graded 节点、38,018 条边出发，先求强连通分量（SCC），再对每个非平凡 SCC 用 NumPy 做稠密特征分解。

1. **主模态逐位一致**。最不稳定的 λ = 2.226 ± 15.914i，Re(s) = 0.05679 ms⁻¹，位于一个 4,095 节点的 SCC 内，与 Arnoldi 的两个起点一致。因此第 7 节问题 4 中担心的 Arnoldi 方向、残差公式和特征根映射问题，已由独立方法排除。
2. **不稳定性极其普遍，而且是局部的**：
   - 稠密分解共得到 3,375 个不稳定特征值，分布在 5 个 SCC 中：两个大 SCC（5,337 和 4,095 个节点，成分都是 R1–R6、L1、L2、L3、L5、R8 混合），以及 8、6、3 节点的小 SCC。
   - **最小的不稳定单元只有 3 个节点：L1、L2、L5 各一个**。边为 L1→L5 −27.5、L5→L1 +11.55、L2→L5 +15.95、L5→L2 +2.475、L2→L1 +0.275。仅 L1↔L5 这一个双向环，环路乘积就是 −317.6，即 λ ≈ ±17.8i，Re(s) ≈ +0.054。
3. **机制与 Opus 先前的预测不同**。Opus 在上一份文档的 §10.2 第 6 条预测的是“抑制×抑制”环路变成正反馈。实际情况是：L1 抑制 L5、L5 兴奋 L1，构成负反馈环；它经过 τ_m 和 τ_s 两级低通后，环路增益远超临界值，形成振荡失稳（Hopf 型）。主根的振荡周期约 44 ms（约 23 Hz），增长时间约 17.6 ms。真实的 1.8 ms 延迟只会增加相位滞后，不会让这类模态变稳定。
4. **为什么 Q1 没有暴露这个问题**。在 Q1 的单侧 clamp 下，L1 超极化时不输出，这个负反馈环在负半周被切断。候选 C 正是重新接通了这个环。所以 Q2 的失败不是实现错误，而是候选 C 在当前传递尺度下的直接后果。
5. **失稳不在临界附近**。若把所有 graded→graded 权重同乘一个系数 c，所有模态都要到 c ≤ 0.369 才会全部稳定（需要约 2.7 倍的增益缩减）。**这个数字只用来说明失稳不是数值边缘现象，严禁用来选择 gain、span 或 tau_emit。**

### 9.3 根因判断：传递尺度没有被识别

- graded→graded 权重的中位数为 1.1，99 分位为 24.2，最大为 30.8，即 contacts × 0.275。
- 在 Q1/Q2 的传递公式下，source 偏离 1 mV，target 的稳态偏移为 `0.325·w` mV。w = 27.5 时约为 8.9 mV/mV，相当于每级放大近 9 倍。
- 根本原因在于 `r = 1` 时每 2.2 ms 发出一个完整权重，等价于持续约 450 Hz 的满幅事件。这个尺度是为 spike-event 传递设定的 0.275 contact 缩放，被直接沿用到了连续 graded 传递上。7 mV 跨度和 2.2 ms 都是约定值，并非由 graded 突触的独立测量确定。
- 结合此前三个阶段，现有结果构成一致的图景：
  - S2/R0b：吸收上态；
  - C0：完全沉寂；
  - Q2：振荡失稳。

  同一张图在三种传递语义下，落在三种都不能作为感觉通路的状态。**问题不在于缺少某个机制，而在于 graded 传递的绝对尺度在本项目中不可识别。** 任何能让系统稳定的修改（缩小增益、damping、反转电位、改 tau_emit）都会引入一个只能由“稳定”本身来选定的参数。

### 9.4 对第 7 节的回答

1. **v1 的降级**：同意。预注册 §7 明确规定“缺 gate、测试未覆盖实际报告路径”属于 implementation failure，且优先级更高。v1 的正式可接纳状态为 `Q2_IMPLEMENTATION_FAILURE`。v1 目录原样保留。B1 原始数据保留为诊断证据，并已由 9.2 的独立方法复核。
2. **修复范围**：同意严格限定于补足 A1–A7（双重复、独立稠密 reference、逐边误差、两 hop 首个影响 clock），以及能捕获“预注册与 runner 不一致”的测试。另外补两项：
   - B1 fixture 目前只有对角矩阵，应增加含复特征值、非正规矩阵的已知 fixture（例如一个 3 节点负反馈环），否则 fixture 测不到本次实际起作用的模态类型；
   - runner 现在把两个起点的不稳定计数相加；判定只看“是否存在”，不影响结论，但报告中应改为分起点报告。
3. **如何避免结果后适配**：
   - 关键事实：B1 不调用 kernel，只依赖图、候选 C 公式和固定的 Arnoldi 设置，因此结果是确定性的。
   - amendment 应冻结以下条件：B1 的输入（Wgg 的边列表哈希）、起始向量、Krylov 维数、残差门和特征根公式不变；新运行得到的两个 Hessenberg 矩阵必须与 v1 的 `sin_h`、`rademacher_h` **逐位相等**，否则判 implementation failure。
   - 这样修复性重跑只可能出现两种结果：A 通过后得到 `Q2_LINEAR_INSTABILITY`，或 A 不通过得到更高优先级的失败。不存在通过修复“改出”另一个结论的空间。
   - 输出标记为 `protocol-repair replication`，使用新目录，并在报告中写明“B1 结果在修复前已知”。
4. **是否先复核 Arnoldi**：不需要再复核。9.2 的独立稠密分解已经逐值复现主模态，B1 hard-stop 设计合理。可以把 9.2 的 SCC 稠密分解作为修复版中 B1 的**第二个独立 reference**，仅作描述，不另设门禁。
5. **若重跑再次得到合格的 `Q2_LINEAR_INSTABILITY`**：同意终止候选 C。并且根据 9.3，应进一步承认在当前 neuron model 下，graded 传递尺度不可识别。不要追加 damping、cell-type gain、APL 倍率、反转电位或阈值修改：每一项都会成为由“稳定”选出的参数。
6. **mode-localization 审计**：9.2 已经得到主要信息：失稳局部存在于几乎每个 cartridge 的 L1↔L5 负反馈环，最小单元为 3 个节点。由此可以写出可解释的否定结论，不再需要单独立项。若要正式化，可以在修复性重跑的 audit-only 部分加入“SCC 分解 + 每个 SCC 的最大 Re(s) + 最小不稳定 SCC 的边表”，但不得用于删边或选参。因为失稳遍布几乎所有 cartridge，任何“删掉不利 SCC”的做法都等于删除整个 lamina 反馈，本身就违反图完整性规则。
7. **其他稳定机制族**：项目内没有可直接冻结 graded 突触绝对增益的独立定量依据。文献中虽然有 LMC 的电生理研究，但要把它转换成本模型的 contact × 0.275 尺度，同样需要一个无法独立确定的换算。建议的停止结论是：

   > 在冻结的 MaleCNS v1.0 图与现有 LIF/电流型突触模型下，视觉 graded 通路的传递尺度不可识别：单侧 release 时 KC 完全沉寂（C0）；双向 release 时 lamina 局部负反馈环振荡失稳（Q2）；spiking 语义下系统进入吸收上态（R0b）。当前 whole-brain visual route 停止。

   如果项目要继续，需要用户在更高层面做选择，例如更换 neuron/synapse 模型，或改用非视觉的感觉入口。任何新方向都要重新立项，不能作为 Q2 的延续。

### 9.5 建议顺序

1. 写一份窄范围 amendment：只修复 A 门证据链和 B1 fixture，并加入 B1 Hessenberg 逐位相等的锁定条件。先跑一次 smoke，再用新目录重跑。
2. 若得到合格的 `Q2_LINEAR_INSTABILITY`，正式停止候选 C 和当前视觉路线，并写一份 stop/scope 文档（并入 9.3、9.4 第 7 条的结论）交用户决策。
3. sensory adapter、plasticity、S3、calibration 和市场训练继续阻塞。
