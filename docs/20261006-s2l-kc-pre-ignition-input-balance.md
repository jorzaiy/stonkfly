# S2l 预注册：KC pre-ignition input balance audit

## 状态与性质

本文件在查看 S2l 新结果、正式运行或按结果选择实现之前冻结。S2l 是 observational、full retained graph 的诊断；`counterfactual=false`、`learning=false`、`production_eligible=false`。不修改 graph、权重、阈值、动力学、解码器或输入帧，不使用收益、标签、交易或性能指标。

## 固定输入与可复现性

使用 S2d 已冻结的同一 60 个事件、同一帧 hash 和数组级等价的初始状态；不重新采样。事件来源为 `runs/plasticity_diagnosis/S2d/20261004T-projection-threshold-v1/projection_threshold_analysis.json`，其事件列表及 `events_sha256` 必须逐字复用。历史 clean checkpoint `runs/wc006_purged_fly_training/20261004T-balanced-reward-v2/brain-clean.npz` 只读用于审计比较；不能跨 kernel build 直接 restore。S2l observer build 必须生成自身 provenance-valid clean checkpoint，并逐字段比较历史文件中所有持久数组（weight 及 brain fields）的 shape/dtype/bitwise equality，以及 cursor/total_spikes/weights_frozen/config/graph/circuit hashes；build metadata 允许不同且须明确记录。任一状态字段不等即 fail closed。图及 kernel/build provenance 必须记录真实 path、SHA256；无法验证时 fail closed，不以静态图统计替代动态 delivery。

原 S2/J 的 active-set bitsets 仅作只读派生输入：集合元素为 4064 个 KC 的稳定 ordering/global index，不将 hash 当作集合。低组为原 43 帧，高组为原 17 帧；低组交集、出现频次及对应 KC body ID/global index 单独报告，不能把交集误称为因果来源。

## 互斥 source class（固定优先级）

每一个真实成功执行的 KC delivery（仅指 kernel 中实际执行 `g[j] += weight[e]` 的事件）必须且只能归入以下第一条匹配类别，类别顺序即优先级：

1. `KC`：source 节点属于 KC；
2. `APL`：source body ID 为 10540 或 10977，且 type 为 APL；
3. `DPM`：source type 为 DPM；
4. `PN/visual_projection`：优先使用权威 circuit 的 `visual_projection` 节点集合；若另报 PN annotation，必须标注二者差异，不得默认为同一集合；
5. `other_cb`：source 属于 `cb_intrinsic` 且不匹配以上类别；
6. `other`：其余 source。

实现必须记录每类节点数、稳定 body-ID 集合 hash、分类版本和优先级。所有类别 event_count 之和必须等于全量 KC delivery audit counter；不相等、slot 越界、ABI magic/version/size 不匹配或 buffer overflow 均 fail closed。

## 时间相位与指标

按真实 delivery step 与按真实 spike emission 记录的 `KC1`、`KC10`（按 spike emission 时间排序的 unique KC）划分：

- `before_kc1`：严格 `step < KC1`；
- `kc1_to_before_kc10`：`KC1 <= step < KC10`，因此 KC1 同一步归入此段；
- `kc10_and_after`：`step >= KC10`，因此 KC10 同一步归入此段；若该帧未达到 KC10，则该段为空并明确记录 `KC10=null`，不把未达到阈值的帧补成后段。

KC1 与 KC10 由 observer 在线冻结为首次及第十个 unique KC spike emission step；重复 spike 不推进 rank。上述三相互斥且穷尽所有成功 KC deliveries。主指标的 cumulative strict-before-KC10 只取 step `< KC10`，自然排除 KC10 同步步。

每个 source class × phase 输出 delivery count、signed/positive/negative weight sum、unique source、unique KC target、first/last step。主指标只使用 `before_kc1` 与 cumulative strict-before-KC10；后段广播量不得解释为点火原因。

对每个 KC 输出其自身 first-spike 之前的累计输入（按上述 class 的 count、signed、positive、negative）及 first-spike step；可用紧凑 NPZ 保存 4064×class 数组，JSON 只保留 schema、shape、hash 和汇总。

## APL 专项

记录首个成功 APL→KC delivery 相对 KC1、KC10 的 step；严格 before-KC1 与 strict-before-KC10 覆盖的 unique KC 数及 signed inhibition。记录全 KC、全来源 coverage，用于描述“APL timing vs effective strength”。膜电位距阈值直方图只有在 observer 不改变计算顺序且开销可控时才记录；否则在 limitations 明确未记录。

## 等价性硬门禁

必须使用已审计的 observer 设计：production trace 默认关闭；magic/version/size 校验失败立即报错；固定长度聚合器溢出 fail closed。历史 checkpoint→trace 的桥接必须记录隔离 source/binary SHA，不得把不同 kernel 当作等价。

至少运行 2 个 low 与 2 个 high 固定帧 smoke。trace-off 必须逐帧复现历史 S2/S2k C 的 frame hash、active KC count、KC spike count、全脑 spike hash（若权威 hash 不存在必须写 null 及原因，禁止伪称通过）；trace-on/off 必须逐帧 counts、spike hash、最终 v/g/refractory/queue/clock 等价。任一 gate 失败不得启动 60 帧正式运行。

## 预注册判读与限制

仅允许以下描述性判读：若 before-KC1 已有广泛 PN/visual 输入，提示感觉增益/全局 scale 方向；若少数 other_cb 来源先覆盖多数 KC，提示该通路定位方向；若 APL 显著晚于 KC1/KC10，提示时序方向；若 delivery 不大但广泛同时越阈，提示 LIF/初始/更新规则方向。所有结果仅为观察性时序/分布，不是干预或因果证据，不涉及盈利。

low 帧可能未达到 KC10，其 strict-before-KC10 窗口与 high 帧不等长；禁止按不等长 raw totals 排名 high/low。完整图必须保留；不得删边、改权重、屏蔽神经元、训练或交易。
