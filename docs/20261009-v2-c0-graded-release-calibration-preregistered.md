# Whole-brain dynamics v2：C0 graded-release synthetic calibration（冻结预注册）

状态：**2026-10-09 冻结；允许实现并执行 V2-C0。C0 仅使用合成、无标签、直接 photoreceptor drive，不接触市场、RGB 图表、reward、plasticity 或交易解码。**

C0 的目标是为 Q1 已通过资格审计的 hybrid kernel 冻结唯一一个全局 graded-release gain，并在完全隔离的 synthetic holdout 上一次性验证：剂量响应不退化、KC 工作点不过度双峰、刺激身份可区分、历史依赖受控、撤刺激后能退出高活动、且 KC 稀疏时 MBON07/11 仍可驱动。

C0 不证明视觉生理、市场感知、学习或盈利。任何通过都只允许进入下一阶段的 synthetic sensory-adapter 设计评审。

## 1. 前置资格与固定输入

Q1 v7 正式判定必须为 `Q1_IMPLEMENTATION_QUALIFIED`。两个独立 runner 启动时均重新计算：

| 输入 | 冻结 SHA256 |
|---|---|
| `data/graph.npz` | `346b8af85a11af13b8324e18669812c1924569e7d1adcb4e6f45cc461a2c344b` |
| `data/normalized/neurons.feather` | `2103b5203d2ba5502c06fb70a4e0b23edcbcf79ed0c344722101941fc289ba2d` |
| `data/normalized/edges.arrow` | `842b762c9427c794eaedd453178aa034727a5084850a91fdcc4d3fffe09f5b7e` |
| Q1 v7 formal JSON | `394c2cd05ecc54c979b9d871176a86bb82670c850c108fe91c63ba8354ead574` |
| Q1 v7 raw NPZ | `15623018571bd78437779236c89effd8542ad56df19407d2dbb9221b013e0c20` |
| Q1 v7 manifest | `036e5a8bba71b40ed2595f635c6883afad0c25d760718aa94164f58ee872ee02` |
| v2 qualification kernel source | `99f63402e2a6393c229aab139915233d24dff8ad411f068d91ec965edfbef79e` |
| v2 brain wrapper | `2673de4e7c4aa4ee0fd20636b17bf2c2828103bacfbbb05b2ff0f559979434e6` |
| Q1 formal runner | `635c8202882182c8fda56123379338d36de515db4d6ef19c12e56113c61d221a` |
| Q1 automatic tests | `6c021d8979ea73203dc07101826a085f0e2a7206cf30488bc4b9a351f0ca35bb` |

图规模仍必须为 166,700 nodes、25,582,938 edges、124,177,617 contacts；graded mask 固定为 Q1 的 11,822 nodes，其中 photoreceptor 4,706、lamina 7,114、APL 2。KC 固定为 4,064；MBON07/11 分别 4/2。

固定输入、Q1 代码和 Q1 产物不得为 C0 修改。C0 如需 gain 参数，只能在新增的隔离 C0 wrapper/kernel argument 中实现；v1 与 Q1 默认行为保持不变。

## 2. 唯一可校准参数

Q1 的连续传递为：

`u_j(t + 1.8 ms) += w_ij * r_i(t) / 2.2 ms`

C0 只允许加入一个全局、无量纲 graded-release gain `gamma`：

`u_j(t + 1.8 ms) += gamma * w_ij * r_i(t) / 2.2 ms`

`gamma` 同时作用于全部 graded source 的全部原有出边；不得按细胞类、边、符号、target、活动或时间使用不同 gain。它不修改持久 weight array，正式结果必须标注 `counterfactual_calibration: true`、`production_eligible: false`。

冻结候选网格：

`G = [0.125, 0.25, 0.5, 1.0, 2.0]`

不插值、不扩网格、不二次扫描。`gamma=1` 是 Q1 工程单位；选择规则优先最接近 1，而不是收益或动作。

其余常数固定为 Q1：`tau_m=20 ms`、`tau_s=5 ms`、`V_th=-45 mV`、non-KC rest `-52 mV`、KC rest `-60 mV`、delay `1.8 ms`、spiking refractory `2.2 ms`、KC adaptation `8 mV/200 ms`。learning 固定 off。

## 3. 刺激定义

### 3.1 直接 drive 边界

C0 直接向 4,706 个 `graded_candidate_photoreceptor` 节点写入工程化 current；不调用 `visual.py`、retinal projection、RGB、lamina bias 或 Naka–Rushton adapter。lamina 与 APL 只接受网络内源输入，绝不直接刺激。

这只是 hybrid-network calibration，不是视觉生理。

### 3.2 Uniform stimulus

`UNIFORM_d` 对全部 4,706 photoreceptor 施加 `+d` current，其余 external drive 为零。

- calibration levels：`D_cal = [0, 1, 2, 3, 4, 5, 6]`
- holdout levels：`D_hold = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]`

holdout 全部位于 calibration level 的中点，不得在 calibration 阶段计算或查看。

### 3.3 Hash pattern stimulus

对每个 photoreceptor source ID，计算 `SHA256(UTF8(salt + ":" + decimal_source_id))`，按完整 256-bit digest 升序；选择前 `floor(fraction * 4706)` 个节点。这样每个 pattern 大小固定、与图输出无关。

全部 pattern current 固定为 `+4.0`：

- calibration：`CAL_A`、`CAL_B`，fraction 均为 0.50；
- holdout：`HOLD_A25` fraction 0.25；`HOLD_B50` fraction 0.50；`HOLD_C75` fraction 0.75；`HOLD_D50` fraction 0.50。

salt 即上述大写名称。runner 必须写出每个集合的 source-id SHA256。calibration runner 禁止构造四个 `HOLD_*` mask。

## 4. Trial 与读出定义

### 4.1 独立 trial

每个独立 trial 从 clean rest、零 conductance、零 queue、零 adaptation/modulation、冻结权重开始：

1. 50 ms zero baseline；
2. 100 ms stimulus warm-up；
3. 50 ms 同一 stimulus measurement；
4. 200 ms zero washout；
5. 100 ms zero post-measurement。

总计 500 ms。`HybridBrain.run()` 的 counts 在每段开始清零，因此 stimulus readout 只使用第 3 段，post readout 只使用第 5 段。

### 4.2 Active sets

- active KC：4,064 个 KC 中 measurement 段 `counts > 0` 的 unique nodes；
- active MBON07/11：各自精确 `cell_type` 集合中 `counts > 0` 的 nodes；
- sparse KC band：`41..812`，对应 `[1%,20%]`；
- fixed-low-like：`<=10`；
- dense：`>812`。

每个 trial 保存完整 KC active-set bitset、KC/MBON counts、分群 spike counts、first/last spike、release/delivery integrals、post active sets、状态摘要和 hashes。不得只保存均值。

### 4.3 History ramp

从 clean state 连续运行，不在 level 间 reset：

`0,1,2,3,4,5,6,6,5,4,3,2,1,0`

每 level 先 50 ms warm-up、再 50 ms measurement。序列结束后 300 ms zero，并用最后 100 ms作 post readout。同一 level 的升序/降序 KC set 和 MBON counts成对比较。

## 5. C0-A calibration phase

### 5.1 执行规模

- dt 固定 `0.05 ms`；
- 每个 `gamma`：7 个独立 uniform trials + `CAL_A/CAL_B` 两个 pattern trials + 1 个 history ramp；
- 所有独立 trial 和 ramp 各运行两次，要求规范化原始状态/读出 SHA256 相同；
- 共 5 个 gamma，不得早停或根据中间结果删除候选。

### 5.2 每个 gamma 的 calibration gates

#### 基础门禁

- 全部 finite、bounds、queue/overflow、observer-off 等价和 weight bitwise freeze 通过；
- `UNIFORM_0` 为精确 equilibrium；
- 两次重复全部 deterministic。

#### Uniform dose-response

对六个非零 levels：

- 至少 3 个 level 的 KC 位于 `41..812`；
- fixed-low-like level 不超过 1 个；
- dense level 不超过 1 个；
- active-KC count 至少有 4 个 distinct values；
- Spearman rank correlation `rho(d, active_KC) >= 0.8`，ties 使用 average ranks；
- 任一相邻 dose 的向下反转不得超过 406 KC；
- 任一相邻 dose 的绝对跃迁不得超过 406 KC。

若曲线从未超过 812，允许保持稀疏；不得为了形成 dense endpoint 强制点燃。

#### Washout / persistence

全部九个独立 stimulus trials 的 post-measurement active KC 必须 `<=40`；history ramp 的最终 post active KC 也必须 `<=40`。不得把持续高活动解释为记忆通过。

#### MBON 可驱动

- 至少两个 sparse uniform levels 同时具有 MBON07 和 MBON11 非零 spike；
- `CAL_A` 与 `CAL_B` 均须 active KC 位于 sparse band，且两种 MBON type 均非零。

#### Calibration identity

`CAL_A` 与 `CAL_B` 的 KC set Jaccard 必须 `<=0.80`，且两者 KC set 不相等；报告 MBON07/11 node-count 与 spike-count 向量，不以收益或动作解释。

#### History dependence

对 level 1..6：

- active-KC count 对称相对差 `2|up-down|/(|up|+|down|)`，分母为零时按 0；至少 5/6 levels `<=0.25`；
- KC-set Jaccard 至少 4/6 levels `>=0.50`；
- 任一方向出现 post persistent dense 状态即失败。

### 5.3 冻结选择规则

只有全部 calibration gates 通过的 gamma 才是 qualifying candidate。

- 无 qualifying candidate：判 `C0_NO_QUALIFYING_GAIN`，停止，不运行 holdout；
- 有候选：按 `abs(log2(gamma))` 升序选择，完全相同时选择较小 gamma；
- 若选中 `0.125` 或 `2.0`：判 `C0_GRID_BOUNDARY`，停止，不扩网格、不运行 holdout；
- 否则写 `c0_selection.json`，包含全部候选结果、选中 gamma、规则、calibration code/data hashes；写入后计算 SHA256，文件不可再改。

不得以 holdout、KC 个体身份、MBON 动作方向或任何收益信息改变选择。

## 6. C0-B synthetic holdout phase

holdout 必须由独立进程执行，只读取冻结 `c0_selection.json`。它必须验证 selection SHA256、calibration verdict、选中 gamma 非边界，并拒绝任何 CLI gain override。

### 6.1 Initial conditions

四个固定 initial conditions：

1. `CLEAN`：精确 clean state；
2. `P1/P2/P3`：分别使用 NumPy `PCG64` seed `202610091/202610092/202610093`，只对非-graded nodes 加 `Uniform[-0.05,+0.05] mV` 初始 voltage perturbation；graded nodes 保持 rest。

所有被 perturb 的 nodes 必须进入 initial active list，以便状态被正常解析演化。运行期间没有随机噪声。

### 6.2 执行规模

- dt：`0.05` 与 `0.025 ms`；
- stimuli：7 个 `D_hold` uniform + 4 个 `HOLD_*` pattern；
- 4 个 initial conditions；
- 每个组合重复两次；
- 另在 `CLEAN` 下、两个 dt 各运行一次 midpoint continuous ramp：
  `0.5,1.5,2.5,3.5,4.5,5.5,6.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5`。

独立 holdout trial 共 `2 × 11 × 4 × 2 = 176`，另加两条 ramp。不得删除失败或无 spike trial。

### 6.3 Holdout gates

#### 完整性

- 176/176 trials 和两条 ramp 完成；
- 全部 fixed hashes、selection hash、source masks、weights、finite/bounds/queue/overflow 通过；
- 每个重复对的状态与读出 SHA256 完全相同。

#### Fine-dt

对 `CLEAN`、repeat 1 的 11 个 stimuli，比较 0.05/0.025：

- active-KC count 对称相对差 `<=0.10`；
- KC-set Jaccard `>=0.70`；
- MBON07 和 MBON11 active-node count 各自绝对差 `<=1`；
- 至少 10/11 stimuli 同时满足上述三项；
- 两条 midpoint ramp 的 history gate 判定必须一致。

#### 稀疏分布与退出

在全部非零 holdout stimulus trials中：

- 至少 50% active KC 位于 `41..812`；
- fixed-low-like (`<=10`) 不超过 10%；
- dense (`>812`) 不超过 10%；
- 至少 90% trial 的 post-measurement active KC `<=40`；
- 不允许任一 stimulus × initial-condition 组合在两个 dt 下都保持 post dense。

#### Initial-condition robustness

将 KC state class 定义为 `low/sparse/dense`，其中 11..40 记为 `sub_sparse`：

- P1/P2/P3 相对 CLEAN 的 class 一致率至少 90%；
- active-KC set Jaccard 的总体 median `>=0.50`；
- 任一 perturbation 不得把超过 20% 的 CLEAN sparse cases 变为 dense 或 fixed-low-like。

#### Holdout identity 与 MBON

- `HOLD_B50` 与 `HOLD_D50`：所有 dt/initial-condition/repeat 的 KC sets 均非空；Jaccard distribution median `<=0.80`，至少 75% pairs `<=0.90`；
- 两个 50% patterns 均至少 75% trials 位于 sparse band；
- 在所有 sparse holdout trials 中，至少 50% 同时具有 MBON07/MBON11 非零 spike；
- `HOLD_B50` 与 `HOLD_D50` 的 `[MBON07 active nodes, MBON11 active nodes, total spikes by type]` 向量至少 75% pairs 不完全相同。

#### Holdout history

两条 midpoint ramp 各自要求：

- 6 个配对非端点 levels 中至少 5 个 count 对称相对差 `<=0.25`；
- 至少 4/6 KC-set Jaccard `>=0.50`；
- 最终 300 ms washout 的最后 100 ms active KC `<=40`。

## 7. 判定优先级

全文件统一使用以下数值约定：对称相对差在两值均为零时定义为 0；Jaccard 在两个集合均为空时定义为 1、仅一方为空时定义为 0。凡 identity gate 已要求集合非空者，不能借空集约定通过。

C0 只允许以下正式结果：

1. `C0_IMPLEMENTATION_FAILURE`：hash/schema/count/runner/observer/artifact 或缺失 gate；
2. `C0_NUMERICAL_FAILURE`：非有限、bounds、queue/overflow、重复确定性或 fine-dt 基础失败；
3. `C0_NO_QUALIFYING_GAIN`：五个 calibration gains 均未通过；
4. `C0_GRID_BOUNDARY`：选中 gain 位于 0.125 或 2.0；
5. `C0_HOLDOUT_FAILURE`：calibration 选中非边界 gain，但任一 holdout gate 失败；
6. `C0_SYNTHETIC_QUALIFIED`：calibration 选择与全部 holdout gates 通过。

优先级按上述顺序。失败不得通过放宽门槛、增加 gain、改变 stimulus/duration、删除 initial condition 或读取旧 R0/市场结果补救。

## 8. 两进程隔离与产物

正式根目录：

`runs/whole_brain_v2/C0/20261009T-graded-release-calibration-v1/`

至少包含：

- `calibration/c0_calibration.json` 与 `c0_calibration_raw.npz`；
- `c0_selection.json`（仅在存在非边界选中 gain 时生成）；
- `holdout/c0_holdout.json` 与 `c0_holdout_raw.npz`（仅合法 selection 后生成）；
- `c0_formal.json`、`c0_report.md`、`c0_manifest.json`；
- calibration/holdout 各自的 `launch.txt`、stdout、source/binary/tests start/end hashes；
- 每个 stimulus mask、initial perturbation 与 active-set bitset hashes。

calibration 与 holdout 使用两个独立 runner。calibration runner 的 import graph 不得包含 holdout mask 构造函数或打开 holdout 目录；holdout runner 不得含 gain 搜索/选择逻辑。合并器只能按冻结规则汇总，不得重新选择。

若 calibration 停止，formal 必须明确写出未运行 holdout 的预注册原因，不能伪造空 holdout pass。

## 9. 执行与停止规则

正式大跑由 Luna 子代理在 tmux 中执行。主代理审核代码、测试、进程、selection 文件时间顺序、产物 hashes 和正式判定，不持续轮询。

纯实现错误必须保留失败目录、先写 amendment、再使用新版本目录。模型/数值/无候选/边界/holdout 失败是正式科学结果，停止 C0，不增加机制或调参。

## 10. 明确禁止与通过后的边界

- 不读取或复用 R0/R0b/R0c stimuli、旧 60 帧、市场图像、行情、label、reward、action 或 P&L；
- 不直接刺激 lamina、APL、KC、MBON；
- 不按 active-KC/MBON 在线调 gain、drive、threshold 或 inhibition；
- 不按细胞类/边/符号使用不同 gain；
- 不运行 plasticity、固定交易解码、paper replay 或真实交易；
- 不将 engineered dopamine 称为痛苦、快乐或意识；C0 根本不注入 dopamine；
- 不把 direct-current hash pattern 称为视觉图像或生物刺激。

`C0_SYNTHETIC_QUALIFIED` 只说明一个冻结的 global graded-release gain 在本预注册 synthetic calibration/holdout 下满足工程化动力学门禁。下一阶段仍需另行预注册 sensory adapter；在 adapter、通路与持续状态门禁通过前，S3 和市场训练继续阻塞。
