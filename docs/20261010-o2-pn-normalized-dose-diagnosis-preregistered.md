# O2 PN-normalized dose diagnosis（冻结预注册）

状态：**2026-10-10 冻结；授权实现、smoke 和正式执行。** O2 是 O1 停线后的 counterfactual 诊断，不是 O1 补跑或放宽。O1 的 `O1_SINGLE_EVENT_INSTABILITY` 继续有效，原离散 PN-event 路线保持关闭。

## 1. 问题与边界

O1 显示同一个完整图上的八个 PN single events 呈两极化：七个只产生亚阈值传播，`DM4_adPN` 在三个 `dt` 都进入持续活动。O2 只回答：把每个 PN 的全部原始出边按一个不依赖输出的结构规则归一化后，是否存在同一固定剂量，使多个 PN 在完整图上产生稳定、非零、可区分且跨 `dt` 一致的早期 KC 响应。

- `counterfactual: true`，`production_eligible: false`；
- 不使用 market、return、profit、learning、reward、decoder、execution 或任何交易标签；
- 不删除节点或边，不改变传播内核、阈值、rest、delay、refractory、adaptation 或图权重；
- 不按结果替换 O0 panel、剂量、门槛或停止条件；
- 无论结果如何，O2 本身不授权训练。

O1 预注册第 3 节的 `3 × 10 × 2 = 60` 是算术笔误：固定输入实际为 ZERO 加八个 PN，所以正式 runner 正确执行了 `3 × 9 × 2 = 54`。O1 产物完整，判定不受影响。

## 2. 固定输入与 pins

继承 O0 的八节点 panel 和 panel SHA256：

`cc011844e8ed4e415353e8a835bca04a0e091d0179aaccef94e86d3e426d5301`

除 O1 已冻结的 graph、neurons、Q1 wrapper/binary pins 外，O2 固定：

| 输入 | SHA256 |
|---|---|
| O1 raw | `046e9865a5310da6c56379b1b23647186af47f7cd0148634eef405f8c9b2a665` |
| O1 formal | `8b19cd27314cdfeb1870c6307c7aac4ee1c41e5ccc86871b1ee0ae910e3cdb08` |
| O1 manifest | `1e7ce86d22badb94fb988075dba51824bf5e947d344f72b2af6773bcdb7ff93f` |

O1 verdict 必须仍为 `O1_SINGLE_EVENT_INSTABILITY`，且 O1 gates 必须保持 implementation、propagation、deterministic repeats、dt-class consistency 为 true，stability 为 false。

## 3. 静态首跳审计

对每个固定 PN 保存全部原图出边的：edge count、unique target count、direct-KC edge/target count、positive/negative/zero count、signed sum、L1、L2、min/max 和分位数；保存目标与权重字节摘要。另关联 O1 的 direct target/KC counts、2ms peak conductance、total/KC spikes 和 stable-exit class。静态审计只解释差异，不参与 panel 或剂量选择。

## 4. 唯一 counterfactual 输入

对 PN `s` 的全部原始出边 `w_e` 使用：

`g_e = dose × w_e / Σ_e |w_e|`

同一目标的平行边用加法累积。所有原始出边、符号和相对比例保留；图内后续传播完全使用冻结 Q1 empty-graded-mask compatibility path。该向量在 arrival clock 0 作为 engineered PN efferent-release proxy 一次性加入其全部直接目标的 `g_event`，并把这些目标加入 active set。不向其他节点加输入，不排除 KC，也不注入 source membrane current。

这不是原图中的一个离散 spike，不能用于推翻 O1；它只隔离“源节点总出边规模”与“下游拓扑/动力学”。每个 trial 从 clean rest、zero conductance、zero drive、zero queue 开始。图的 `weight` 数组在整个 trial 中必须字节不变。

固定总 L1 dose grid：

`[128, 256, 512, 1024, 2048, 4096]`

该对数网格在查看 O2 输出前冻结，覆盖 O1 panel 原始出边 L1（约 334 至 4484）的两侧；不得插值、追加或按结果选择新剂量。

## 5. Trials 与记录

在 `dt=[0.1, 0.05, 0.025]ms`：

- ZERO clean control 两次；
- 八个 PN × 六个 dose，各自 clean-start 两次；
- 共 `3 × (1 + 8 × 6) × 2 = 294` trials。

每个正式 trial 固定 1,000ms，segments 为 `[2, 48, 450, 500]ms`。前 50ms 保存全脑/KC spike count、unique KC spike node IDs 与摘要；每段保存 spike counts、active nodes、queue、voltage 和 conductance范围。最后 500ms执行退出门。

## 6. Trial 分类与门禁

### 6.1 安全与退出

- 所有状态 finite，`abs(v)<1e6`、`abs(g)<1e9`、overflow=0；
- 最后 500ms spike count 为 0；final queue empty；`g_event<=1e-7`、KC adaptation `<=1e-4`；final voltage 回到 rest 的 float32 4 ULP 内；
- graph weights start/end 相同；ZERO 无 spike、无 conductance且退出；
- 两次重复的 segment records、early KC IDs 和 final-state hash 完全相同。

### 6.2 固定分类

- `unstable`：未通过安全或退出门；
- `silent`：稳定退出，但前 50ms unique KC spike nodes 为 0；
- `dense_nonzero`：稳定退出，但前 50ms unique KC spike nodes 大于 KC 总数的 20%；
- `stable_nonzero`：稳定退出，前 50ms unique KC spike nodes 在 `[1, floor(0.2 × n_KC)]`。

同一个 PN/dose 的三个 `dt` 必须获得相同分类，否则标记 numerical inconsistency。

### 6.3 候选动态区间

一个 dose 只有同时满足以下条件才是 candidate：

1. 八个 PN 在三个 `dt`、两次重复中全部安全并稳定退出；
2. 至少四个 PN 在三个 `dt` 中均为 `stable_nonzero`；
3. 对每个响应 PN，三个 `dt` 的 early-KC set 两两 Jaccard 均不低于 0.8；
4. 响应 PN 在 `dt=0.1ms` 的 early-KC sets 至少形成三个不同摘要，且任意两组 Jaccard 不高于 0.9；
5. 两次重复完全确定。

只判候选是否存在；不得用收益或偏好的神经形状挑选剂量。

## 7. Verdict 与停止条件

只允许：

1. `O2_IMPLEMENTATION_FAILURE`：pin、panel、输入 ledger、trial 数、重复、hash 或产物门失败；
2. `O2_NO_STABLE_NONZERO_BAND_CLOSE_V1`：无 candidate dose，包括 silent/unstable 二分、稠密响应或跨 `dt` 不一致；此结果正式关闭 MaleCNS v1 的金融学习/交易实验，不再做剂量微调；
3. `O2_COUNTERFACTUAL_BAND_FOUND_REVIEW_REQUIRED`：至少一个 candidate dose；只允许独立审阅是否另立新模型版本，仍禁止直接训练或交易。

## 8. 实施与产物

正式目录：

`runs/whole_brain_olfactory/O2/20261010T-pn-normalized-dose-diagnosis-v1/`

至少包含 `o2_raw.npz`、`o2_formal.json`、`o2_report.md`、`o2_manifest.json`、`launch.txt`、`stdout.txt`。正式前必须通过 micro injection/normalization/duplicate-target tests、classification/evaluation tests、output protection、CLI smoke 和完整图短 preflight。正式计算在 tmux 中执行；失败目录保留，不覆盖。
