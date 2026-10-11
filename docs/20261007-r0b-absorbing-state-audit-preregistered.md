# R0b 预注册：首达时间 / 吸收态审计

状态：在运行任何 R0b 代码前冻结。来源：`docs/20261007-r0-results-for-opus.md` §9.4。用户已授权执行。

性质：观察器增强、标签无关、只用 C 臂、paper-only、`learning=false`、`production_eligible=false`。不改图、权重、阈值、动力学或 kernel。只有 (c) 的初态微扰是对初始状态的反事实，标注 `counterfactual=true`。不读取市场数据、reward、P/L、动作、S2 low/high 标签或决策器输出。

## 1. 固定输入

- 图：`data/graph.npz`，SHA256 `346b8af85a11af13b8324e18669812c1924569e7d1adcb4e6f45cc461a2c344b`。
- 历史 clean checkpoint：`runs/wc006_purged_fly_training/20261004T-balanced-reward-v2/brain-clean.npz`，SHA256 `a92f547e720e7d9830ae519f26c0bacdf4f76c59761fb2be4adb6f34cac2a225`。和 R0 一样生成 fresh observer-build checkpoint，并对所有持久数组做 bitwise 桥接；任一字段不等即 fail-closed。
- R0 参照：`runs/plasticity_diagnosis/R0/20261007T-synthetic-dose-response-v2/r0_formal.json`（SHA256 `d5f294f5c84aebfafa22a3539315791a56827a76e907ebadcfd8743e809c4e21`）与 `r0_formal_raw.npz`（SHA256 `7f3a228f5e880368a4937b63c1bde6ff9ff7650b438cfdf7a276de929fb8253b`）。
- 刺激：完全复用 R0 的帧构造（`uniform_frame`、`contrast_frame`）。uniform level `40,42,45,48,50,52,55,58,60,62,65,68,70,72,75,78,80`（17 个），contrast amplitude `0,5,…,60`（13 个），共 30 个。不新增刺激点。
- 视觉参数：`retina_half_saturation=0.02`、`lamina_bias=12.0`（生产默认）。
- 撤刺激帧：`uniform_frame(0)`。

## 2. 协议

每次运行都从 fresh clean checkpoint `restore` 开始，按 10 ms bin 连续调用 `rgb_step(frame, 10.0, learning=False, retina_half_saturation=0.02, lamina_bias=12.0)`，并记录每个 bin 的 counts。

- **(a)+(b) 主序列，30 次运行**：每个刺激先呈现 2000 ms（200 个 bin），然后立刻接 1000 ms 的 uniform0（100 个 bin）。所有 30 个刺激都跑完整 3000 ms；(b) 的判读只用 (a) 中点燃过的刺激。
- **(c) 初态微扰，40 次运行**：刺激固定为 `uniform_50`、`uniform_60`、`uniform_70`、`contrast_15`、`contrast_30`，seed 固定为 0–7。restore 之后、第一个 bin 之前，执行 `v += np.random.default_rng(seed).uniform(-0.5, 0.5, size=n).astype(float32)`，作用于全部细胞。drive、权重和其他状态不变。每次运行呈现 2000 ms，不接撤刺激段。
- **确定性复跑**：`uniform_50` 与 `contrast_15` 不加微扰，各重复跑一次完整的 3000 ms 主序列，要求全部 bin 数组逐位相等。

## 3. 观察量（只读）

- 每 10 ms bin 的 spike 数和活跃细胞数，按以下组别统计：retina（`brain.retina`）、lamina（`brain.lamina`）、ol_sensory、ol_intrinsic、visual_projection、cb_intrinsic（以上四组按 `graph.npz` 的 superclass 划分，均排除 KC）、KC（`circuit["kc"]`）、APL（type == "APL"）、MBON07（`circuit["reward_mb"]`）、MBON11（`circuit["aversive_mb"]`）。
- **T_ign**：KC spike 数 ≥50 的第一个 10 ms bin 的起点，单位 ms；未出现记为 null。
- 每次运行在 2000 ms 时刻读取 CB（cb_intrinsic，非 KC）与 KC 的膜电位 v，做直方图：1 mV 宽，范围 [−80, −44)，两端外的值计入两侧溢出桶。
- 前 500 ms 的累计 counts（全细胞）与累计 KC bitset。
- 撤刺激段每 100 ms 的 CB 和 KC spike 数。

## 4. 完整性门禁（任一失败即 fail-closed，不产出结论）

1. 主序列 30 个刺激前 500 ms 的累计 counts，与 R0 C 臂 `protocol=independent, repeat_index=1` 同名刺激的 raw counts 行逐位相等。
2. 确定性复跑逐位相等。
3. 每次运行前后权重 hash 不变；`learning=false`。
4. 所有数组 finite；各组索引非空；KC 数量为 4064。
5. JSON 和 NPZ 互相 pin SHA256，原子写入。

## 5. 主指标与判读（冻结）

定义：
- `ignited(run)`：T_ign 非 null，且 T_ign < 2000。
- `pre_CB`：1900–2000 ms 内 CB spike 总数。
- `post_CB_end`：2900–3000 ms 内 CB spike 总数。
- `post_CB_300`：2200–2300 ms 内 CB spike 总数。

判读：
- **H_abs（吸收上态 + 首达）**，需三条同时成立：
  1. 主序列中 ≥27/30 个刺激点燃；
  2. 已点燃的刺激中，≥90% 满足 `post_CB_end ≥ 0.5 × pre_CB`；
  3. (c) 中 5 个刺激各自 8 个 seed 的 T_ign 标准差（未点燃的记为 2000，作为删失值）取平均，结果 ≥ 0.5 × 主序列 30 个刺激间 T_ign 的标准差（同样把未点燃记为 2000）。
- **H_thr（真实剂量阈值）**，需两条同时成立：
  1. 主序列中，uniform 和 contrast 两个家族按 level 排序后，点燃与否各自最多换态一次；
  2. (c) 中 5 个刺激各自 8 个 seed 的点燃结果完全一致。
- **H_stim（刺激维持型上态）**：已点燃的刺激中，≥90% 满足 `post_CB_300 < 0.2 × pre_CB`。

总判定：
- 若 H_stim 为真，判 H_stim（与 H_abs 第 2 条互斥）。
- 否则，若 H_abs 为真且 H_thr 为假，判 H_abs。
- 否则，若 H_thr 为真且 H_abs 为假，判 H_thr。
- 其余情况判 inconclusive，停止并回到设计评审。

各条的原始布尔值与数值必须全部报告。

## 6. 禁止

不改刺激、seed、微扰幅度、窗口、bin 宽、T_ign 定义或阈值；不在 R0 或 R0b 结果上补点；不用结果挑选 APL 倍率、lamina_bias、half_saturation、rest、阈值或 0.275 缩放；不改 `stonkfly/` 下的任何文件或 kernel；不交易；不提交 git。smoke 产物不能作为正式结果。
