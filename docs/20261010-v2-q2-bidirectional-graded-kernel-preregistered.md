# Whole-brain dynamics v2：Q2 bidirectional graded-deviation kernel qualification（冻结预注册）

状态：**2026-10-10 冻结；授权实现并执行 V2-Q2。Q2 只验证 kernel 语义、数值与稳定性，不以 KC/MBON 点燃为成功，不接触市场、reward、plasticity、decoder 或交易。**

Q2 根据 C0 与 Q2-D0 的否定结果修正一个明确语义缺口：Q1 的单侧 release clamp 不能表达低于静息基线的 graded output change。Q2 只实现无新增自由参数的候选 C，并把失控上态作为首要风险。Q2 通过也不能复活旧 C0；之后必须另写全新的 synthetic calibration/holdout。

## 1. 固定输入

启动时必须重新计算以下 SHA256；任一不一致即 `Q2_IMPLEMENTATION_FAILURE`。

| 输入 | SHA256 |
|---|---|
| `data/graph.npz` | `346b8af85a11af13b8324e18669812c1924569e7d1adcb4e6f45cc461a2c344b` |
| `data/normalized/neurons.feather` | `2103b5203d2ba5502c06fb70a4e0b23edcbcf79ed0c344722101941fc289ba2d` |
| `data/normalized/edges.arrow` | `842b762c9427c794eaedd453178aa034727a5084850a91fdcc4d3fffe09f5b7e` |
| Q1 v7 formal | `394c2cd05ecc54c979b9d871176a86bb82670c850c108fe91c63ba8354ead574` |
| Q1 kernel source | `99f63402e2a6393c229aab139915233d24dff8ad411f068d91ec965edfbef79e` |
| Q1 wrapper | `2673de4e7c4aa4ee0fd20636b17bf2c2828103bacfbbb05b2ff0f559979434e6` |
| Q1 binary | `9df8110f1e2134f8a6d88cde46d3a8a6278590f291ad87b49161926d2fd53f41` |
| D0 预注册 | `e5a22f4ec89b2f3f5dcd80d3849913bdb2e823abc284e6c7554ddbff1f099f8f` |
| D0 formal | `a3b5c65f20f78f4a2eeed3d10e32e24dc0ad2f5df370298f4f783f42fd124929` |
| D0 raw | `06a97159d7bb5db097cd7eb5cb1881b2b7d72962788d93d4e44793c56329d9be` |
| D0 manifest | `34c33f2ee8174d54f8ea33ce982b6328eebd32850f66d95bc2a4b5c5dd3d4166` |
| D0/Opus 结果审阅 | `b736a1422c156b14de6d6fcb87ad00ec7426ec303e766c66af0271f9fe44a288` |

图与分类仍必须为 166,700 nodes、25,582,938 edges、124,177,617 contacts；graded mask 仍为 photoreceptor 4,706、L1/L2/L3/L5 7,114、APL 2，共 11,822。KC 4,064。不得改变、删除、屏蔽、重排或重标原图节点、边、contacts、weight 或类型。

## 2. 唯一新机制

Q2 只把 Q1 的 release clamp：

`clip((v-rest)/7mV, 0, 1)`

替换为工程化 signed release deviation：

`delta_r(v) = clip((v-rest)/7mV, -1, 1)`

传递仍为：

`u_j(t+1.8ms) += w_ij * delta_r_i(t) / 2.2ms`

约束：

- `7mV = V_th - V_rest`，不是拟合参数；不得扫描 span、gain、threshold 或 baseline。
- `delta_r<0` 表示低于 tonic baseline 的 release change，不称为负递质释放。
- `v>=rest` 时必须与 Q1 release 逐位相同。
- rest 时 deviation 必须精确为零；zero-input equilibrium 仍要求 bitwise，不因 D0 的受刺激 1-ULP 响应而放宽。
- 原有 signed weight 决定 target 方向；不得按 cell type、edge sign、target 或活动另设 gain。
- membrane、synapse、delay、refractory、KC adaptation、queue 和四类连接语义全部沿用 Q1。
- learning 永久 off；Q2 不修改 Q1/v1 文件或默认生产路径。

## 3. 隔离实现

新增独立 `v2_q2_kernel.cpp`、`v2_q2_brain.py`、qualification runner 和 tests。Q1 source/binary 必须保持固定。Q2 binary 必须包含独立的 v1 compatibility entry point；empty graded mask 仍与固定 v1 bitwise 等价。

正式开始后 prereg、runner、tests、Q2 source/binary 的 start/end SHA256 必须一致。observer、ledger 和 stability instrumentation 开关不得改变主动力学。

## 4. Q2-A：解析与阴性对照

全部微回路使用独立小图，在 `dt=0.1/0.05/0.025ms` 各重复两次。期望值由不调用 kernel 的 dense float64 reference 计算。

### A1. Deviation clamp

覆盖 `v-rest = {-14,-7,-3.5,0,+3.5,+7,+14}mV`，要求输出 `{-1,-1,-0.5,0,+0.5,+1,+1}`。Python helper 与 C++ 最大误差 `<=1e-7`；正半轴与 Q1 逐位相同。

### A2. 四象限

分别验证：

| edge | source deviation | target response |
|---|---|---|
| positive | positive | positive |
| positive | negative | negative |
| negative | positive | negative |
| negative | negative | positive |

逐边 delivery、delay clock、`g_graded` 和 voltage 与 dense reference 误差 `<=1e-4`。

### A3. 两 hop sign inversion

独立验证 negative×negative、negative×positive、positive×negative、positive×positive 两 hop 链。首个影响 clock 必须与 1.8ms delay 和固定步序一致，不能只比较终值。

### A4. 正/负 external drive

同一 graded source 接受等幅 `+3.5/-3.5` drive；release deviation、delivery 与 target voltage 必须方向相反且符合 reference。负向 drive 不得被 clamp 为零。

### A5. Sign-permutation 阴性对照

在固定两 hop fixture 中交换 edge signs，预注册的目标方向必须随之改变；如果原测试在 signs 置换后仍得到同一方向，判 `Q2_ANALYTIC_FAILURE`。阴性对照不在完整 MaleCNS 图上改符号。

### A6. Graded 语义不泄漏

graded mask 外的 spiking source 即使阈下偏离 rest，也不得连续传递 deviation；它只按 Q1/v1 threshold-spike 语义输出。四类 connection、refractory 和 reset 行为不得变化。

### A7. Compatibility 与正半轴回归

- empty graded mask：Q2 与固定 v1 的全部原状态 bitwise 等价；
- 所有 graded node 始终 `v>=rest` 的 fixtures：Q2 与 Q1 的完整状态、ledger、queue 和 counts bitwise 等价。

任一 A gate 失败立即停止，不运行稳定性或完整图。

## 5. Q2-B：rest 线性稳定性审计

候选 C 在 rest 附近斜率固定为 `1/7mV`。先抽取 11,822-node graded→graded 原图子矩阵 `Wgg`，不做阈值或 contact 筛选。

### B1. 独立稀疏 Arnoldi 搜索

使用独立 C++ float64 audit helper，不复用 kernel 更新代码：

- 两个固定起始向量：`sin(i+1)` 与 PCG64 seed `20261010` 生成的 Rademacher；规范化后使用；
- Krylov dimension `160`；两遍 modified Gram–Schmidt；
- 对 Hessenberg 矩阵用独立 dense eigensolver；每个 Ritz value 保存 residual；
- residual `<=1e-6 * max(1, |lambda|)` 才算 converged；至少应有一个 converged Ritz value，否则 `Q2_LINEAR_STABILITY_INCONCLUSIVE`；
- 两个起点只取并集，禁止选择较有利的一次。

无 delay 的连续线性化对每个 `Wgg` eigenvalue `lambda` 使用：

`(20s+1)(5s+1) = 5*lambda/(7*2.2)`

任一 converged mode 的任一 root 满足 `Re(s)>=0`，立即判 `Q2_LINEAR_INSTABILITY`。未发现不稳定 Ritz mode 不等于证明稳定；真实 delay 必须继续通过 B2。

Arnoldi helper 必须用已知 2×2/4×4 stable/unstable matrix 验证 eigenvalue 与 residual。若本地无现成 eigensolver，可在 C++ 只生成 H，再由 NumPy 对固定 `160×160` H 求 eig；不得安装新包或联网。

### B2. 含真实 delay 的 matrix-free 扰动门

只有 B1 未发现不稳定 mode 才执行。完整图从 rest、零 conductance/queue 开始，外部 drive 永远为零；只扰动 graded voltage：

- uniform `+epsilon`、uniform `-epsilon`；
- PCG64 seed `202610101/202610102` 的两个 Rademacher vector；
- `epsilon=1e-4mV`；不得扫描。

每个 initial condition 在 `dt=0.05` 与 `0.025ms` 下各运行两次 2,000ms。记录每 1ms 的 graded/all-node `L2/Linf` deviation、spikes、release/delivery、saturation、queue 和 state hash。

硬门：

- 重复确定；finite/bounds/queue/weights 全通过；
- 不得出现任何 spiking-node spike；
- 2,000ms 的 `L2` 不得高于初始，最后 500ms 的最大 `L2` 不得高于前 500ms；
- final `Linf <= 4 ULP at -52mV`；
- 不得出现持续非零 delivery、limit cycle 或吸收上态；
- 两个 dt 的 stability class 必须一致。

任一失败判 `Q2_PERTURBATION_INSTABILITY`，停止。

## 6. Q2-C：完整图刺激与回落

只有 A/B 全通过才运行。learning off，每个条件独立 clean start，在 `dt=0.05/0.025ms` 各重复两次：

1. `ZERO`：500ms zero；
2. `PHOTO_POS`：50ms zero + 100ms 全 photoreceptor `+3.5` + 1,000ms zero；
3. `PHOTO_NEG`：同样但 `-3.5`；
4. `L1_POS`：只对全部 L1 `+3.5`，同样时序；
5. `L1_NEG`：只对全部 L1 `-3.5`，同样时序。

`3.5` 沿用 Q1 full-graph smoke，不是从 D0 结果选出的 calibration 值。Q2-C 只验证方向、安全和退出，不以 KC/MBON 活动作为通过条件。

硬门：

- ZERO bitwise equilibrium；
- 正/负 source deviation 与 delivery 符号符合 A 门；
- 所有状态 finite，`abs(v)<1e6`、conductance/delivery/queue bounds 通过，无 overflow；
- 最后 500ms 不得有任何 spike；最后 100ms delivery 必须为零或在 `4 ULP` 对应数值容差内；
- final graded/generic/KC `Linf <=0.05mV`，且最后 500ms envelope 单调不增；
- 不得出现持续高活动、周期活动或 R0b 型吸收上态；
- 重复 hash 相同，两个 dt 的稳定/退出 class 一致；
- 报告但不门控：各层 spikes、KC/MBON activity、lamina saturation fraction、全图最低 voltage、恢复到 `0.05mV` 的时间。

任一失败判 `Q2_FULL_GRAPH_FAILURE`，不得调低 drive、gain、span，或提高 threshold/APL。

## 7. 判定优先级

只允许：

1. `Q2_IMPLEMENTATION_FAILURE`；
2. `Q2_ANALYTIC_FAILURE`；
3. `Q2_LINEAR_STABILITY_INCONCLUSIVE`；
4. `Q2_LINEAR_INSTABILITY`；
5. `Q2_PERTURBATION_INSTABILITY`；
6. `Q2_FULL_GRAPH_FAILURE`；
7. `Q2_KERNEL_QUALIFIED`。

按顺序优先。缺 gate/产物、observer 改状态或测试未覆盖实际报告路径均为 implementation failure。任何失败均停止，不补扫参数、不扩大 graded mask、不添加反转电位/noise/tonic spike/APL 倍率。

## 8. 产物、smoke 与执行

正式目录：

`runs/whole_brain_v2/Q2/20261010T-bidirectional-graded-kernel-v1/`

至少包含 `q2_raw.npz`、`q2_formal.json`、`q2_report.md`、`q2_manifest.json`、`launch.txt`、stdout，以及独立解析 reference/Arnoldi 原始输出。manifest 保存所有非自引用产物、固定 pins、compiler/flags、source/binary/tests/runner/prereg start/end SHA256 和未运行阶段的预注册原因。

正式运行前必须：

- 自动测试覆盖 A1–A7、stable/unstable Arnoldi fixtures、判定优先级、失败产物和输出保护；
- 真实 CLI smoke 完整生成 raw/formal/report/manifest；
- full-graph 10ms 正/负 drive smoke 无异常；
- 主代理审核代码与 smoke 后，长跑才可进入 tmux。

若 B1 提前失败，正式产物仍必须完整记录 A 与 B1，明确 B2/C 未运行的预注册原因。

## 9. 资格边界与后续分支

`Q2_KERNEL_QUALIFIED` 只说明 signed graded deviation 的实现与稳定性通过上述工程门，不证明生理、视觉、KC 功能、学习或盈利。

若 Q2 通过，下一步另行冻结全新的 synthetic calibration；刺激网格必须在查看 Q2 功能输出前冻结并覆盖候选 C 非饱和区。如果新 calibration 仍无法产生预先定义的可分辨通路响应，则停止当前视觉路线并报告否定结果，不再追加机制，除非出现新的外部依据。

sensory adapter、plasticity、S3 和市场训练继续阻塞。
