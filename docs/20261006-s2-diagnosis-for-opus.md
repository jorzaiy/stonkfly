# Stonkfly S1–S2j 实验状态（供 Opus 审阅）

**更新：2026-10-06**
**状态：S2 感觉表征门槛未通过；S2j 已关闭；尚未进入 S3。Opus 审阅已附于文末，提出 KC→KC 递归兴奋假设与 S2k 诊断方案（待授权）。**

> 本文是当前权威交接。旧 `plasticity_diagnosis_for_opus.md` 和 `20261004-balanced-reward-training-summary.md` 仅作历史记录；其“继续扫塑性参数/构造对手学习”建议已被后续证据取代。

## 请 Opus 优先回答

1. 现有证据是否足以判定 S2 失败，并继续阻止 S3？
2. 当前 LIF/感觉代理下的“KC 沉默或大规模爆发”更像感觉动态范围错误，还是全图 LIF 吸引子伪影？
3. 如果允许新的固定、可审计感觉代理，应在感觉神经元、PN→KC 边界或其他通路输入？必须保留完整 MaleCNS 图，不得将脚本交易冒充神经输出。
4. S3 是否可以仅作为“塑性机制独立诊断”运行，但不得宣称交易感觉链路已验证？
5. 请复核 trace ABI、sample-perfect 判据、不等长窗口限制与 S2j 编码设计。

## 不可改变的约束

- 保留完整 MaleCNS v1.0 图；不剪枝、不隐藏利润选择、不使用 LLM 交易政策。
- 分离市场观测、感觉代理、神经传播、塑性、固定解码和执行限制。
- 风控只能拒单，不能选择替代交易。
- 损益强化只是 DAN 的工程输入；不声称意识或已验证的盈利学习。
- 仅 paper/离线；S2c–S2j 无删边、改权或反事实干预，不声称因果。

## 总结论

60 个固定观测在 4064 个 KC 上形成两个离散状态：

- 43/60：约 4 个 KC 活跃；
- 17/60：1114–1433 个 KC 活跃。

两种状态不能稳定区分 reward/aversive。S2 汇总 Jaccard 虽标记为 `MODERATE_OVERLAP`，详细分布却显示表征坍缩/爆发，因此不满足原 S2 的稳定可区分条件。

时序追踪将分叉收缩到 central-brain intrinsic 同步/递归波，但未找到稳定的单一感觉投射边或因果开关。无标签视觉标定和三轮 FixedVector 代理也未找到稀疏、平滑、通道可区分的 KC 工作区间。

按 [p1_plasticity_fix_review.md](p1_plasticity_fix_review.md) 的预注册顺序，当前不进入 S3。

## 实验链

| 阶段 | 目的 | 结果 | 状态 |
|---|---|---|---|
| S1 | 流程/decoder/provenance/回归 | 工程修正已做；`runs/` 无独立权威 S1 产物 | 前置完成，非机制结论 |
| S2/S2b | KC overlap 与重复集 | 60 帧可确定复现；43 low / 17 high | 门槛未通过 |
| S2c | 定位最早层级分叉 | sample-perfect 分界首先出现在 `visual_projection`，后被 `cb_intrinsic` 放大 | 描述性 |
| S2d | 投射阈值/候选结构 | event508=low 边界，event240=high 边界 | 非因果 |
| S2e | kernel 时序传播 | high 帧候选 visual projection 大多晚于 KC10，无法解释初始点燃 | 完成 |
| S2f | 全 visual projection/CB 早期源 | 14 个 stable CB source 为 high 17/17、low 0/43 phase0；边界 projection 源不稳定 | 完成 |
| S2g | 14 stable CB 的实际输入 | 4 条 perfect-high-only pair 全部汇聚到 13334 | 样本内描述 |
| S2h v3 | `4 sources→13334→downstream` | 四源 high 17/17 严格先投递；13334 在 14 stable CB 中中位排名13 | 后段广播枢纽兼容，非点火起点 |
| S2i A | 最早 stable CB leader 输入 | leader=10530（rank≤3: 14/17）；无 perfect-high-only 输入；主要来自 `cb_intrinsic` | 完成 |
| S2i B | RGB→retina 合成传递 | uniform 0–20: 0 KC；40: 2 KC；80: 1400 KC；空间模式仍可分 | 陡峭阈值 |
| S2i B2 | half-saturation×lamina 无标签标定 | 12×24=288 runs；`no_acceptable_calibration` | 生产默认值不变 |
| S2j v1 | FixedVector→R1–R6 | 392 runs；one-hot 全沉默 | superseded |
| S2j v2 | 加入 811 mapped R8 | one-hot 几乎沉默，稠密组合大量激活 | superseded |
| S2j v3 | 16-bin Hadamard 分布码 | scale1–12 one-hot 全0；scale18 仅一通道跃迁1414 KC（34.8%） | `S2j_closed` |

## 关键网络与时序细节

S2f 的 14 个 stable CB body ID：

`10209, 10262, 10305, 10530, 11050, 11390, 13334, 14391, 14437, 16648, 18049, 19845, 19873, 516097`

S2g 中四条 high 17/17、low 0/43 的 pair：

- `10530 / VL1_ilPN → 13334`
- `11930 / M_imPNl92 → 13334`
- `14437 / VP3+VP1l_ivPN → 13334`
- `19845 / VP3+VP1l_ivPN → 13334`

S2h v3：

- 四源 high 的 spike/delivery/strict-pre 均为 17/17；low 无 delivery。
- 最后 pre-target delivery 领先 13334 首 spike 的中位值为 10/8/8/8 steps，总范围 3–23 steps。
- 13334 high 17/17 放电，low 0/43；在 stable CB 首 spike 中 rank 12–14，中位13。
- KC1 在 17/17 high 早于13334（中位 -3089 steps），不能当作其下游证据。
- KC10 在 17/17 high 晚于13334，中位 +52 steps，范围 +44–+70。
- 13334 每个 high 帧有 473 个真实一跳目标；延迟队列使单次 spike 与投递不可一一因果归属。

## 感觉层结论

### RGB 路径

生产默认为 `retina_half_saturation=0.02`、`lamina_bias=12`。高灰阶 KC 集合并非完全饱和，等均值空间图案也能产生不同响应；故证据更支持“亮度/中枢阈值 + 空间信息仍存在”，不支持简单的像素饱和。12 个标定配置无一在三个冻结指标上 Pareto-可接受，因此未改默认值，也未用60帧反向选参。

### FixedVector 视网膜代理

三轮映射均没有稀疏、平滑、可区分的 KC 工作区间。最终 v3 在 scale1–12 时全部 one-hot 为0；scale18 只有一通道跃迁到1414/4064 KC，超出预注册20%上限，其他七通道仍为0。不应继续扩大 scale 或事后放宽门槛。

## 审计事故与权威版本

| 事件 | 处理 |
|---|---|
| S2f 原排名有 raw frame-count 偏差 | 以 rate-normalized `early_sources_review.json` 为准 |
| S2h v1 仅旧 JSON 后处理，v2 缺 strict-pre/正确 KC10 | 以 direct-kernel v3 为准 |
| S2i v1 漏 `trace5` delivery | 停止无效全量，v2 修复重跑 |
| S2i 四个 mean=0.5 图案 uint8 256→0 | 仅补跑四帧，以 merged v2 为准 |
| S2j v3 raw `preregistered.scales` 误留 v2 列表 | 保留 raw；review 绑定 raw SHA256，按实际 records 重算 selection |

正式 native trace 要求 trace-on/off 的 spike hash、全脑 counts、KC counts、`v/g/refractory` 等价，overflow fail-closed。

## 权威产物与 SHA256

| 阶段 | 路径 | SHA256 |
|---|---|---|
| S2 | `runs/plasticity_diagnosis/S2/20261004T-kc-overlap-balanced-v2/kc_overlap_analysis.json` | `ef90155210e2ec7937ccae15c9e1f918f6b19d11fd90291c092ceb2c3238d617` |
| S2b | `runs/plasticity_diagnosis/S2/20261004T-kc-overlap-detailed-v2/kc_overlap_analysis.json` | `177c0017a80f84e4d3bfa119fb19aed18d12274bf1b3c03b9385ff993f3515ce` |
| S2c | `runs/plasticity_diagnosis/S2c/20261004T-sensory-layer-diagnostic-v1/layer_group_comparison.json` | `1ec21a21acb9629d366aa460e54f3c1486ad3726d1ea272bc1e03c5e9d70e22f` |
| S2d | `runs/plasticity_diagnosis/S2d/20261004T-projection-threshold-v1/projection_threshold_analysis.json` | `5d3d97d1c7913bfa8d64a042bd9e8b7ebb86c642361eb13ff574ee0cb77c17f5` |
| S2e | `runs/plasticity_diagnosis/S2e/20261004T-time-resolved-v1/time_resolved_propagation.json` | `31f87f1103fcbc58ce76ba224d26c86348ab822601ec5b077e6d886544a0adf6` |
| S2f | `runs/plasticity_diagnosis/S2f/20261004T-early-sources-v1/early_sources_review.json` | `d9cf6b0ec59a69ce07eccadc3f46c274823c5484a4203caf170511003572a051` |
| S2g | `runs/plasticity_diagnosis/S2g/20261005T-stable-cb-inputs-v1/stable_cb_inputs_analysis.json` | `1ac43e6f70d12b2347efdc195c929df9ac9da33ebd0017af703435d10c4beb32` |
| S2h | `runs/plasticity_diagnosis/S2h/20261005T-13334-temporal-chain-v3/temporal_chain_summary.json` | `d257e71b76d4e32b4c8988687320d9024e3c960aaf6dd04c8a26a855db6c4fff` |
| S2i A | `runs/plasticity_diagnosis/S2i/20261005T-earliest-cb-boundary-v2/earliest_cb_boundary_summary.json` | `0fcb67ccc1220c1854b4166f9a853f907f60d8b71113cde202ea8df1f73d8ca2` |
| S2i B | `runs/plasticity_diagnosis/S2i/20261005T-synthetic-visual-transfer-v1/visual_transfer_assay_v2.json` | `f12b8c0c61304b7e84efef761000ebe23f84ac944afb5954febf4b6227b6f926` |
| S2i B2 | `runs/plasticity_diagnosis/S2i/20261005T-synthetic-calibration-grid-v1/calibration_grid.json` | `5f843d9ee6e3aae6d47102669b6b5189a304807b0763fae994f3f0472aea81ef` |
| S2j | `runs/plasticity_diagnosis/S2j/20261006T-fixed-vector-synthetic-v3/fixed_vector_assay_review.json` | `3ef24dcdf6f0fa6b344b11127cc73b7fe772b2d6513ec4fac149e3d4bcfc3aa3` |

## Opus 审阅意见（2026-10-06）

已核对：上表 12 个产物 SHA256 全部一致；S2 原始数字与 JSON 一致。

### 对五个问题的回答

1. **S2 判失败，继续阻止 S3。** `MODERATE_OVERLAP` 是均值假象：reward_within 与 reward_vs_punishment 的 Jaccard 中位数均为 1.0、std≈0.48，分布是两团离散点，不是连续表征。门槛应改看“状态数 + 状态内重叠”，该自动标签作废。
2. **更像全图 LIF 吸引子伪影，而非感觉动态范围问题。** 主要嫌疑是 KC→KC 递归兴奋（见下文）。S2i B2 与 S2j 三轮输入侧标定全部失败，与“下游双稳态，输入怎么调都只能二选一”一致。
3. **暂不设计新感觉代理。** 在 PN→KC 边界或感觉神经元输入都会撞上同一双稳态；先验证下文假设。
4. **S3 可作为纯塑性机制诊断**，但必须标注“不代表交易感觉链路已验证”，且前提是合成刺激下 KC 不处于爆发态；否则 A/B/C 激活同一团 KC，S3 无信息量。
5. trace ABI 与 fail-closed 判据合理。sample-perfect 判据在 60 帧、两态数据上必然偏选后段广播节点（13334 中位排名 13/14 即为例证），只能作描述，不能用于定位起点。

### 漏掉的线索：KC→KC 递归兴奋

由 `data/edges.feather`、`data/annotations.feather`、`data/neurotransmitters.feather` 直接统计（接触数）：

| 输入到 KC | 接触数 | 模型符号 |
|---|---|---|
| KC→KC | 1,153,845 | +（4064 KC consensus_nt 全为 acetylcholine） |
| 全部 PN→KC | 390,947 | + |
| APL→KC（2 个 APL，覆盖 4063 KC） | 196,200 | −（GABA） |
| DPM→KC | 32,795 | — |

单个 KC 中位：KC→KC 输入 288 接触、约 149 个 KC 上游伙伴；PN 输入仅 101 接触。`prepare.py:28` 以 `count × sign × 0.275` 赋权，KC→KC 因此成为 KC 最大的兴奋源。

解释（Opus 判断，未经干预验证）：生物上 KC–KC 接触主要是叶区轴突–轴突突触，作为单室 LIF 的胞体兴奋输入并无生理依据。少数 KC 越阈后经 KC→KC 正反馈拉起上千 KC，2 个 APL 的抑制压不住，形成“~4 个 / ~1400 个”双稳态。与现有证据一致：

- S2h：KC1 早于 13334 中位 3089 steps，KC 点燃先于所谓 CB 枢纽；
- S2i B：灰阶 40→80 时 KC 2→1400；S2j v3：scale18 单通道 0→1414；
- S2f 的 14 个 stable CB 中 11 个是 ALPN（superclass 归为 `cb_intrinsic`），并非独立“中枢点火源”。13334 从 KC 收 13 接触、向 KC 送 1257；仅 LHPV12a1（10209、10305）从 KC 各收约 1550。这与“KC 已点燃、波在扩散”兼容。

### 建议下一步：S2k 预注册诊断性反事实（需用户授权）

性质：**仅诊断，不进生产、不改默认值、不删边**（图保持完整，边保留在数组中），产物目录与 JSON 显式标注 `counterfactual: true`。与“保留完整图”约束的关系须在预注册中写明。

- 输入：S2 的同 60 帧（同 frame hash、同 seed、同 brain 初始态），不得换帧或按 reward/aversive 选帧。
- 臂：
  - **C** 对照：生产配置，须复现 S2 的 spike hash 与 KC counts；
  - **A** KC→KC 边有效权重置 0；
  - **B** 图不变，APL→KC 权重 ×k，k 预先固定少量值（如 2、4），不得事后加值；
  - （可选）**A+B**。
- 主指标（预注册，非收益）：每帧活跃 KC 数分布是否单峰/连续、非双峰；目标稀疏区间约 5–10% KC（约 200–400），上限 20%；状态内与跨标签 Jaccard 分布（看中位数与分布形状，不只看均值）。
- 次指标：全脑 spike 数、MBON07/11 是否仍有活动、是否出现全脑静默。
- 判读：
  - A 消除双稳态 → 根因指向 KC→KC 符号/作用位置假设，下一步是在 `docs/model.md` 中声明并讨论如何建模（如轴突间突触不进入胞体 LIF），而非直接改生产；
  - 仅 B 有效 → 指向全局抑制不足；
  - 都无效 → 回到 PN 层或全图 LIF 参数（阈值、0.275 缩放）另行预注册。
- 禁止：以 Alpha/收益或 reward/aversive 区分度挑选 k；结果不论好坏如实记录。
- 执行：按 AGENTS.md 交 Luna 子代理跑，主代理审产物与 SHA256。

## 暂不建议

- 不用 Alpha/收益选 retina、lamina、塑性或 decoder 参数。
- 不扩大 S2j 电流直到某通道爆发。
- 不把 13334 或 sample-perfect 节点宣称为因果开关。
- 不在 S2 未通过时把 S3 结果宣称为交易链路已验证。
- 不剪枝完整图、不换成 LLM 政策、不使用隐藏利润选择。
