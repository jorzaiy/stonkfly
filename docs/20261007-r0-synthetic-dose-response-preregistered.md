# R0 预注册：标签无关的合成剂量–反应与滞后测定

状态：**冻结设计；尚未实现、尚未运行。** R0 只用于决定唯一的 R1 稳定化机制族。S3 继续阻塞。

## 1. 目的与边界

R0 回答一个限定问题：当前 KC 双峰主要是生产感觉输入落在网络陡峭工作区，还是当前神经动力学本身缺少可用的渐变区？R0 不优化收益、动作、市场标签或现有 60 帧 holdout，也不证明生物学真实性或盈利学习。

- 仅使用本文件预先定义的合成 RGB 刺激；禁止读取市场数据、日期、reward、P/L、动作、S2 low/high 标签或解码结果。
- `learning=false`、`reinforcement=none`、paper-only、diagnostic-only。
- 保留完整 MaleCNS v1.0 CSR 图。不得删除节点、边或接触。
- C 不改变权重。B2 仅在内存中把已有 APL→KC 边的有效权重乘 `2.0`；拓扑、磁盘图和 checkpoint 不变。B2 必须标记 `counterfactual=true`、`production_eligible=false`。
- 不读取 active-KC 数量来动态调整刺激、增益、权重、阈值或运行长度。
- R0 运行后不得增加灰阶、对比度或参数点；任何失败回到设计评审，不在 R0 结果上补点。

## 2. 固定来源与 SHA256

- 图：`data/graph.npz`  
  SHA256 `346b8af85a11af13b8324e18669812c1924569e7d1adcb4e6f45cc461a2c344b`
- clean checkpoint：`runs/wc006_purged_fly_training/20261004T-balanced-reward-v2/brain-clean.npz`  
  SHA256 `a92f547e720e7d9830ae519f26c0bacdf4f76c59761fb2be4adb6f34cac2a225`
- S2i 证据：`runs/plasticity_diagnosis/S2i/20261005T-synthetic-calibration-grid-v1/calibration_grid.json`  
  SHA256 `5f843d9ee6e3aae6d47102669b6b5189a304807b0763fae994f3f0472aea81ef`
- S2k B2 定义：`runs/plasticity_diagnosis/S2k/20261006T-kc-recurrence-counterfactual-v1/s2k_counterfactual.json`  
  SHA256 `40363db184551381972adc5159888d79634ee8cbaa96219d80c85b7fd378f339`
- S2n 正式结果：`runs/plasticity_diagnosis/S2n/20261007T-windowed-causal-attenuation-v1/s2n_formal.json`  
  SHA256 `faca8be81ac8954329f28cafcfb7a33e03c51bf76c1e5a0f0281ca2d54b7bfc9`
- Opus 评审稿：`docs/20261007-stabilization-redesign-for-opus.md`  
  SHA256 `b3b9dc356874935642773cba384b973bcd932a8dbb43d9699fd038fb3801c392`

实现完成后，runner 必须记录本预注册文件、实现脚本、kernel source/binary、配置签名和所有输出的 SHA256。实现不得反向修改本文件；若需勘误，另写带时间戳的 amendment，并在任何 R0 数据生成前完成。

## 3. 固定模型配置与实验臂

共同配置：

- `retina_half_saturation=0.02`
- `lamina_bias=12.0`
- `neural_ms=500`
- 每个刺激保持 `500 ms`
- `learning=false`
- 相同 fresh observer-build checkpoint 必须与上述历史 clean checkpoint 的全部持久数组 bitwise equal。

固定执行顺序：`C`，然后 `B2`。

1. **C**：生产图、生产权重和上述生产视觉参数。
2. **B2**：从同一 clean checkpoint 恢复；仅 APL→KC 有效权重在内存中乘 `2.0`。APL body IDs、命中边数、基线符号计数和 overlay 后倍率必须与 S2k B2 一致。

每次独立运行或连续序列开始前，必须重新从 clean checkpoint 恢复并从基线权重重新构造 overlay；禁止从上一臂或上一序列继承状态。

## 4. 固定刺激

所有帧均为 contiguous `uint8 RGB`，尺寸 `64×64×3`；三个通道相同。像素构造、取整和 SHA256 必须写入产物。

### 4.1 Uniform 主剂量网格

精确强度列表：

`[40, 42, 45, 48, 50, 52, 55, 58, 60, 62, 65, 68, 70, 72, 75, 78, 80]`

每帧全部像素等于该强度。该 17 级网格是唯一用于主阶跃宽度判读的 uniform 网格。

### 4.2 Uniform 锚点

独立 clean-run 锚点：`[20, 160, 255]`。锚点只用于检查更宽范围的平台、回落或额外翻转，不参与 17 级主网格的渐变/硬阶跃计数，也不得据此补点。

### 4.3 固定均值的对比度网格

- 基准均值：像素强度 `60`。
- 空间图案：固定相位的 `8×8` checker；`q=(y//8+x//8)%2`。
- 对比度幅度：`[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]`。
- 对每个幅度 `a`：`q=0` 的像素为 `60-a`，`q=1` 的像素为 `60+a`。因帧尺寸与 cell 对齐，算术均值必须精确为 `60`。

该 13 级网格是唯一用于对比度阶跃宽度判读的网格。不得增加方向、cell size、均值或颜色臂。

## 5. 固定运行协议

每臂严格按以下顺序执行，共 `126` 个 500 ms 刺激；两臂合计 `252` 个刺激。

### 5.1 独立 clean-run 与确定性重复（每臂 66 个）

每个刺激前恢复 clean checkpoint 并重新应用本臂 overlay：

1. uniform 主网格 17 级，每级运行两次：34。
2. uniform 锚点 3 级，每级运行两次：6。
3. 对比度网格 13 级，每级运行两次：26。

同臂、同刺激的两次结果必须在全 counts array、KC bitset/set hash、spike hash 和记录的持久状态边界上完全一致；否则 R0 无效并停止。

### 5.2 连续升降序（每臂 60 个）

每条序列仅在开始前恢复一次 clean checkpoint；序列内部不恢复，以测量历史依赖：

1. uniform 主网格升序：17。
2. 重新恢复 clean 后，uniform 主网格降序：17。
3. 重新恢复 clean 后，对比度网格升序：13。
4. 重新恢复 clean 后，对比度网格降序：13。

升序和降序使用完全相同的刺激集合，只反转顺序。锚点不进入连续序列。

## 6. 必须记录的原始数据

每个 arm/protocol/order/level 至少记录：

- 刺激名、uniform 强度或 contrast 幅度、frame SHA256、实际通道均值；
- active-KC count/fraction、KC spike count、KC bitset、KC set hash；
- 全 counts array hash、全 spike hash、total spikes；
- retinal luminance 的 min/mean/max/hash；
- retina-source drive 的 min/mean/max/hash；
- whole-graph drive 的 min/mean/max/hash；
- 各 superclass 的 active-cell 与 spike count；
- 已识别 MBON07、MBON11 的逐 cell active/spike count；
- 权重 hash、图 hash、checkpoint hash、配置签名、wall time；
- B2 overlay 的 APL body IDs、APL→KC 边数、基线与 overlay 后符号/倍率审计。

原始数组写原子 NPZ；可审阅摘要写原子 JSON。JSON 不重复嵌入大 bitset/count arrays。两者使用 manifest pin 并在 success rename 前自检。

## 7. 冻结指标定义

KC 总数必须由 authoritative circuit mask 读取；预期为 `4064`，不允许硬编码后忽略不一致。

响应带按整数 active-KC count 定义：

- `sub_sparse`：`0..40`（低于 1%）；
- `target_sparse`：`41..812`（约 1% 至小于 20%）；
- `dense`：`>=813`（至少约 20%）；
- 另报告 `fixed-low-like`：`<=10`，但它不替代上述三带。

对每个 arm×主网格，固定计算：

- 13/17 个 level 的 active-KC count、fraction 和 response band；
- 相邻 level 的 active-count 差、KC Jaccard 和 set-hash 是否相同；
- `target_sparse_level_count`：落在 `41..812` 的 level 数；
- `hard_step = target_sparse_level_count <= 1`；
- `gradual = target_sparse_level_count >= 3`；
- 若恰为 2，标记 `borderline`；
- response-band 转换次数；任何 `dense→sub_sparse`、重复进入 dense 或超过两次 band 转换标记 `nonmonotonic`；
- 第一次进入 1% 与 20% 带时对应的 stimulus、retina-source drive 和 whole-graph drive；未跨越则显式为 null。

### 滞后

对同臂、同刺激 level 比较连续升序与降序：

- active-KC count 差、response-band 是否改变、KC Jaccard、set-hash 是否相同；
- `category_hysteresis_count` 为 response-band 不同的 level 数；
- `set_hysteresis_count` 为 set-hash 不同的 level 数。

`category_hysteresis_count>0` 即判存在类别级滞后；不以均值掩盖单点滞后。

## 8. 冻结判读与唯一后续路径

判读只使用 clean-run 的主 uniform/contrast 网格和连续序列的滞后指标；锚点只作审计与非单调警告。

1. **选择 R1-A（感光细胞光适应/对比度编码）**，仅当：
   - C 在 uniform 与 contrast 两个主网格均为 `gradual`；
   - C 两个网格均非 `nonmonotonic`；
   - C 两个连续序列均无类别级滞后。
2. **选择 R1-C（分级、非尖峰 APL 机制族）**，仅当：
   - C 未满足 R1-A；
   - B2 在 uniform 与 contrast 两个主网格均为 `gradual`；
   - B2 两个网格均非 `nonmonotonic`；
   - B2 两个连续序列均无类别级滞后。
   
   B2 通过只说明抑制机制是有效杠杆；不授权把 `×2` 作为生产参数。R1-C 仍需单独预注册模型声明、calibration 与 holdout。
3. **动力学重设计评审，但不自动实现任何 R1**，当 C 与 B2 在两个主网格均为 `hard_step`。这排除 A 单独作为充分修复，但不自动验证分级 APL。
4. **R0 inconclusive，停止并回到设计评审**，包括但不限于：
   - 任一关键网格为 `borderline`；
   - uniform 与 contrast 指向不同机制；
   - 候选臂出现 `nonmonotonic` 或类别级滞后；
   - 上述规则无法唯一选择机制族。

不得因某个臂更接近预期、运行更快、下游更活跃或未来可能更赚钱而改变判读。

## 9. R0 完整性门槛

任一失败均不得生成 success artifact：

- exact 252 个刺激记录，C/B2 各 126；协议、顺序和 level 集合精确匹配；
- fresh checkpoint 与历史 clean checkpoint 全部持久数组 bitwise equal；
- C clean-run 的 uniform20/40/80/160/255 必须与固定 S2i 生产配置记录的 frame/count/set/spike hashes 一致；
- B2 overlay mask、APL body IDs、边数、倍率和符号审计与 S2k 定义一致；
- 每个独立刺激的两次 clean-run 确定性完全一致；
- 每个独立 run 和连续序列开始时，权重从相同 baseline 构造；C 权重等于 baseline，B2 仅允许 APL→KC 命中边不同；
- learning 始终关闭，weight hash 在每个刺激结束后保持该臂预期值；
- 所有数值 finite；KC bitset popcount 等于 active-KC count；
- MBON07/11 身份集合非空且固定；
- JSON/NPZ 原子写、schema 校验、互 pin hash、自检通过；
- 不存在 market/reward/action/return/decoder 字段或数据依赖。

失败目录只能留下明确 failure artifact、日志、launch 记录和必要 checkpoint；不得留下可被误认作正式 success 的 JSON/NPZ。

## 10. R1 与 S3 的后续限制

- R0 只选择机制族，不选择 R1 参数。
- R1 calibration 必须使用与冻结 60 帧 holdout 隔离的数据或合成统计；最多 1–2 个自由参数，参数网格边界最优即停止。
- R1 holdout 只允许一次性运行；失败后不得在同一 holdout 上调参。
- R1 即使稳定性通过，若 MBON07/11 全部静默，也不得进入 S3。
- R1 稳定性通过后才能另行评审 S3；R0/R1 均不使用收益作为通过条件。

