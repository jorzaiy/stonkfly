# R0c 预注册：积分步长数值收敛审计

状态：**冻结设计；尚未实现、尚未运行。** R0c 是数值审计，不是参数优化或稳定化机制实验。R1 与 S3 继续阻塞。

## 1. 目的与结论边界

R0c 只回答一个问题：R0b 中观察到的首达时间分散、`cb_intrinsic`/KC 高活动态和撤刺激后持续性，是否对积分步长 `dt` 具有足够的数值稳定性。

- 只改变积分步长：`0.1 ms`、`0.05 ms`、`0.025 ms`。
- 保持当前 kernel 的不应期输入丢弃、同步更新、阈值、rest、时间常数、连接、权重和视觉代理语义不变。
- **不**在本实验中加入“不应期暂存输入”或其他更新顺序反事实；那属于独立模型语义变更。
- 不读取市场数据、日期、reward、P/L、动作、S2 low/high 标签或解码输出。
- `learning=false`、paper-only、diagnostic-only、`production_eligible=false`。
- R0c 不能证明生物学真实性、意识或盈利学习。即使通过，也只能说当前现象没有被本次三个 dt 否定。
- 固定 stimulus/seed 集合上的结果称为“首达时间分布”或“固定初态集合经验风险率”，不得称为内生随机过程或无记忆过程；当前模型没有时间噪声。

## 2. 固定来源与 SHA256

- 图：`data/graph.npz`  
  SHA256 `346b8af85a11af13b8324e18669812c1924569e7d1adcb4e6f45cc461a2c344b`
- 历史 clean checkpoint：`runs/wc006_purged_fly_training/20261004T-balanced-reward-v2/brain-clean.npz`  
  SHA256 `a92f547e720e7d9830ae519f26c0bacdf4f76c59761fb2be4adb6f34cac2a225`
- R0b 预注册：`docs/20261007-r0b-absorbing-state-audit-preregistered.md`  
  SHA256 `de986a67b829e64c06656145d77b0256a7a3ea6df3980d4d15944affa7a1d373`
- R0b 正式 JSON：`runs/plasticity_diagnosis/R0b/20261007T-absorbing-state-v1/r0b_formal.json`  
  SHA256 `5f40904f57098802d28733f1b7929d5de5a0a8254969cf9900a32dc43f71c38b`
- R0b 原始 NPZ：`runs/plasticity_diagnosis/R0b/20261007T-absorbing-state-v1/r0b_formal_raw.npz`  
  SHA256 `3f510d9fc840e502d5b9d612a74aea69ddaa25bf6256561ef45bdad9953fea85`
- R0b runner：`experiments/a_share/run_r0b_absorbing_state_audit.py`  
  正式启动时 SHA256 `303501eeb1d0f4b11788a4947b50c209ba942e6b4795fc0b920f4c28e4418b67`
- 当前 kernel source 基线 SHA256 `aea2b97d2f9d78c57787f40f42a9d7dd777c3059d8c7f453e61c4d94398438d4`
- 当前 kernel binary 基线 SHA256 `9fa6e5cac39a5d865cf0ca50909cc3527c4ff697fd7c302f4d3c657dbfdde344`

实现后必须记录本预注册、runner、所有被改源码、kernel source/binary、配置签名和产物 SHA256。不得反向修改本文；任何实现勘误必须在生成正式数据前另写 amendment。

## 3. 允许的最小实现改动

当前 `stonkfly/neural/state.py` 明确拒绝 `dt != 0.1`，而 native kernel 已接收 `dt` 并按毫秒换算时间常数、`1.8 ms` 传递延迟和 `2.2 ms` 不应期。R0c 只允许做使现有积分器接受冻结 dt 集合所必需的最小参数化：

1. 生产默认值必须仍为 `dt=0.1 ms`；没有显式 diagnostic 参数时行为和配置签名不得改变。
2. 仅允许 `dt ∈ {0.1, 0.05, 0.025}`，其他值 fail-closed。
3. `MemoryBrain`/controller 必须把显式 dt 传到已有状态分配和 kernel 参数，不得复制第二套动力学。
4. kernel 方程、更新顺序和不应期语义不得修改。若实现需要改 `kernel.cpp` 的可执行语义，R0c 在运行前停止并回到设计评审。
5. 所有以毫秒定义的量保持不变：膜时间常数 `20 ms`、突触电导时间常数 `5 ms`、延迟 `1.8 ms`、不应期 `2.2 ms`、KC adaptation `200 ms`、视觉适应 `10 ms`。对应 tick 数随 dt 精确缩放。
6. `10 ms` observer bin、`2000 ms` 刺激期与 `1000 ms` 黑屏期对三个 dt 都必须整除；禁止用截断或不一致取整。
7. 图、权重、rest、阈值、tonic、lamina bias、retina half-saturation、刺激、RNG 算法和 seed 均不变。

实现阶段必须有单元测试证明：

- 默认构造仍是 `0.1 ms`；
- 三个冻结 dt 对应 delay tick 为 `18/36/72`，refractory tick 为 `22/44/88`；
- 非冻结 dt 被拒绝；
- 各 duration/bin 的 tick 数精确；
- diagnostic dt 不进入生产配置或 checkpoint 路径。

## 4. dt 特定 clean state

固定执行顺序：`0.1`，然后 `0.05`，最后 `0.025 ms`。

- `dt=0.1` 必须从 fresh observer build 生成 checkpoint，并与历史 clean checkpoint 的全部持久数组 bitwise equal。
- 更细 dt 的 queue 长度随 `1.8/dt+1` 改变，不能伪称与 0.1 checkpoint 整体 bitwise equal。它们必须从同一图和同一构造参数生成 dt-specific clean state。
- 对 `0.05/0.025`，所有与 dt 无关的持久数组（含 weight、v、g、drive、previous_drive、counts、luminance、active flags、eligibility、modulation、adaptation 及可比状态）必须与 0.1 clean state bitwise equal。
- dt 相关 queue/queue_count 必须尺寸正确且全零；clock/sim time 为零，refractory 全零，所有 delay slots 为空。差异白名单只能包含由 dt 决定的 queue 形状及明确记录的 dt 元数据。
- 每个 trial 前从对应 dt 的 clean state 恢复；不得在 trial 之间继承状态。

## 5. 固定 trial 集合

完全复用 R0b 在查看其结果前已冻结的五个刺激和八个 seed：

- 刺激顺序：`uniform_50`、`uniform_60`、`uniform_70`、`contrast_15`、`contrast_30`。
- seed 顺序：每个刺激依次 `0,1,2,3,4,5,6,7`。
- 初态微扰：restore 后、第一个 bin 前，对全部细胞执行 `v += numpy.random.default_rng(seed).uniform(-0.5, +0.5, size=n).astype(float32)`。
- 同一 seed 生成的 delta-v 数组在三个 dt 下必须 bitwise equal，并记录 SHA256。
- 每个 trial：固定刺激 `2000 ms`，随后立即呈现 `uniform0` 黑屏 `1000 ms`，总计 `300` 个 `10 ms` bin。
- 三个 dt 各 40 个主 trial；另在每个 dt 下对 `uniform_50/seed0` 与 `contrast_15/seed0` 各确定性复跑一次。
- 正式总数：`3 × (40 + 2) = 126` 个 trial。smoke 不计入正式数据。

不得添加无微扰 trial、刺激 level、seed、微扰幅度、窗口长度或其他 dt。

## 6. 固定观察量

每个 10 ms bin 记录与 R0b 相同的 spike count 和 active-cell count：retina、lamina、ol_sensory、ol_intrinsic、visual_projection、`cb_intrinsic`（排除 KC）、KC、APL、MBON07、MBON11。

每个 trial 还必须记录：

- stimulus/frame SHA256、seed、delta-v SHA256、dt、tick 数；
- 全 counts/bin hash、各组 bin arrays、weight before/after hash、state-boundary hash；
- `T_ign`：KC spike count ≥50 的第一个 10 ms bin 起点；未出现为 null；
- `ignited_by_2s`：`T_ign != null && T_ign < 2000`；
- `pre_CB`/`pre_KC`：1900–2000 ms 的 `cb_intrinsic`/KC spike 总数；
- `post_CB_end`/`post_KC_end`：2900–3000 ms 的对应总数；
- 撤刺激后每 100 ms 的 `cb_intrinsic`/KC spike 总数；
- 2000 ms 与 3000 ms 边界的 `cb_intrinsic`/KC v 直方图；
- dt-specific clean state、图、配置和 kernel provenance。

本文中的 CB 指标仅指 `cb_intrinsic`（排除 KC），不得写成“全脑 CB”或“全局 CB”。

## 7. 0.1 ms 复现门禁

在查看细 dt 结果前，`dt=0.1` 的 40 个主 trial 必须与 R0b 对应 perturbation trial 的前 2000 ms 逐项一致：

- stimulus、seed、delta-v；
- 200 个 group spike bins 与 active-cell bins；
- `T_ign`、`ignited_by_2s`；
- 2000 ms 的 v 直方图和 state boundary（若 R0b 已记录相应字段）；
- weight hashes。

R0b 没有记录的 1000 ms 黑屏段不作为复现门禁。任一已有字段不一致则 R0c fail-closed，不得用细 dt 结果下结论。

## 8. 冻结主指标

所有指标按 dt 分别计算，只使用 40 个主 trial；确定性复跑只作门禁。

### 8.1 固定初态集合经验风险率

把 0–2000 ms 分为 20 个 `[100k,100(k+1)) ms` 区间。对每个区间：

- `at_risk_k`：区间开始时尚未点燃的 trial 数；
- `event_k`：该区间内首次点燃的 trial 数；
- 报告 `event_k/at_risk_k`，但不称为随机 hazard。

冻结汇总量：

`H_dt = sum(event_k) / sum(at_risk_k)`。

同时报告 500/1000/1500/2000 ms 的经验未点燃比例，以及 40 个 trial 的完整 T_ign/censor 向量。未点燃只作右删失，不把 2000 当作真实点燃时间来计算主指标。

### 8.2 高活动态与撤刺激持续性

对 `ignited_by_2s=true` 的 trial：

- `persistent_CB`：`post_CB_end >= 0.5 × pre_CB`；
- `persistent_KC`：`post_KC_end >= 0.5 × pre_KC`；
- `persistent_both`：两者同时为真。

分别报告通过数量和比例。另报告这些 trial 在 1900–2000 ms 与 2900–3000 ms 的每 10 ms `cb_intrinsic`/KC spike rate 中位数及完整分布。

若 trial 在 2000 ms 前点燃但 `pre_CB` 或 `pre_KC` 为零，标记 integrity failure，不以除零绕过。

### 8.3 dt 间差异

以 `0.1 ms` 为 frozen baseline，同时重点检查两个更细 dt 是否彼此接近：

定义对称相对差：`reldiff(a,b)=2×|a-b|/(|a|+|b|)`；两者均为零时定义为 0，只有一者为零时定义为 2。`H_0.1=0` 时无法定义 risk ratio，直接判 `R0C_INCONCLUSIVE`，不得改用其他分母。

- `risk_ratio(dt) = H_dt / H_0.1`；
- 2000 ms ignition fraction 的绝对差；
- 四个时间点经验未点燃比例的最大绝对差；
- `persistent_both` 比例的绝对差；
- pre/post `cb_intrinsic` 与 KC 中位 spike rate 的对称相对差。该 rate 比较只使用在 `0.05` 与 `0.025 ms` 下均于 2000 ms 前点燃的相同 stimulus/seed 配对集合 `I_common`，防止因两臂点燃成员不同造成选择偏差；`I_common` 为空时判 `DT_NOT_CONVERGED`。

所有分刺激、分 seed 的配对差异必须保留；不得只报告聚合均值。

## 9. 冻结判读

### 9.1 `DT_CONVERGED_FOR_CURRENT_AUDIT`

仅当以下全部满足：

1. 所有完整性门禁及 0.1 ms R0b 复现门禁通过；
2. `0.05` 与 `0.025` 相对 `0.1` 的 `risk_ratio` 均在 `[0.5, 2.0]`；
3. `0.05` 与 `0.025` 的 `H_dt` 相对差不超过 25%；
4. 每个 dt 的 `persistent_both` 比例至少 90%，且任意两 dt 的绝对差不超过 10 个百分点；
5. `I_common` 非空，且 `0.05` 与 `0.025` 的 pre/post `cb_intrinsic`、KC 中位 spike rate 对称相对差均不超过 20%；
6. 每个 dt 至少 1 个 trial 在 2000 ms 前点燃，且至少 1 个对应 trial 满足 `persistent_both`。

该结论只表示本次现象对三个 dt 足够稳定。它不把 R0b 的正式 `inconclusive` 改判为 H_abs，也不证明随机性、刺激无关性或生物学真实性。

若成立：停止继续调 A/B/C 感觉稳定化旋钮；进入模型范围评审，更新文档说明当前模拟器不能作为已验证的刺激敏感感觉通路。S3 仍不得接入交易链路；是否进行孤立塑性机制测试需另行授权和预注册。

### 9.2 `DT_NOT_CONVERGED`

出现任一项即成立：

- 任一细 dt 的 `risk_ratio` 小于 0.5 或大于 2.0；
- `0.05` 与 `0.025` 的 `H_dt` 对称相对差超过 25%；
- 任一 dt 完全不出现点燃，或出现点燃但 `persistent_both` 为 0；
- `I_common` 为空，或 `0.05` 与 `0.025` 的任一 pre/post `cb_intrinsic` 或 KC 中位 spike rate 对称相对差超过 20%。

若成立：S2–R0b 对具体动力学机制的解释全部降级为数值伪影候选；停止添加生物学机制，下一步只能评审积分器与事件/不应期语义。不得从三个 dt 中挑选“看起来最好”的一个作为生产配置。

### 9.3 `R0C_INCONCLUSIVE`

若复现门禁通过，但结果既不满足 9.1 的全部条件，也不命中 9.2，判 inconclusive 并停止。不得补 dt、刺激、seed 或延长窗口。

完整性门禁失败不属于上述科学判读，只生成 failure artifact。

## 10. 完整性与产物门禁

任一失败均不得生成 success artifact：

- 精确 126 个正式 trial；每 dt 40 个主 trial加 2 个确定性复跑，顺序精确；
- 0.1 ms fresh checkpoint 与历史 clean checkpoint 全持久数组 bitwise equal；细 dt clean state 只允许第 4 节白名单差异；
- 同 seed 的 delta-v 在三个 dt 下 bitwise equal；
- 0.1 ms 前 2000 ms 与 R0b 全部可比字段一致；
- 每个 dt 的两次确定性复跑在全部 bin arrays、T_ign、v histograms、state boundary 和 weight hash 上一致；
- learning 始终关闭；每个 trial 权重不变；完整图、边数、权重和 circuit identities 不变；
- 延迟和不应期的毫秒值不变，tick 换算精确；
- 所有原始数组和派生数值 finite，组索引非空，KC 数为 4064；
- JSON/NPZ 原子写、schema 自检、互 pin SHA256；
- 产物明确记录 `counterfactual=true`、`diagnostic_only=true`、`production_eligible=false`；
- 不存在 market、reward、action、return、decoder 或 reinforcement 数据字段/依赖；
- 失败时删除可能被误认作 success 的 JSON/NPZ，仅保留明确 failure artifact、日志和必要 checkpoint。

## 11. 明确禁止

- 不得修改或比较不应期输入暂存语义、更新顺序、阈值、rest、连接符号、0.275 缩放、APL、KC adaptation、lamina bias 或 half-saturation。
- 不得根据结果增加 dt、seed、刺激、时长或观察窗口。
- 不得把固定 seed 集合的经验比例称为内生随机概率、无记忆过程或统计显著性结论。
- 不得以收益、动作、市场表现、MBON 解码方向或未来可交易性选择 dt。
- 不得把 `cb_intrinsic` 指标外推为整个 central brain 或真实果蝇现象。
- 不得在 R0c 后继续用同一 R0/R0b/R0c 数据调稳定化参数。

## 12. 执行要求

- 实现与正式大数据运行必须交给 Luna 子代理；主代理只负责指挥和审核。
- 正式运行使用 tmux 后台会话，不在前台持续轮询。
- 先运行单元测试和最小 smoke；smoke 只验证代码路径与门禁，不参与正式结论。
- 预计本实验明显慢于 R0b；启动时必须依据 smoke 给出 ETA、session、日志和输出目录。
- 未获得用户明确授权前，不得修改代码或启动实验。
