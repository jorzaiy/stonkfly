# O1 olfactory-PN single-event qualification（冻结预注册）

状态：**2026-10-10 冻结；授权实现并执行 O1。O1 只验证显式 engineered PN spike-event intent 在完整图上的传播、确定性和退出，不使用 market、learning、reward、decoder 或 execution，不以 KC spike 为通过条件。**

## 1. 固定输入与 panel

| 输入 | SHA256 |
|---|---|
| `data/graph.npz` | `346b8af85a11af13b8324e18669812c1924569e7d1adcb4e6f45cc461a2c344b` |
| `data/normalized/neurons.feather` | `2103b5203d2ba5502c06fb70a4e0b23edcbcf79ed0c344722101941fc289ba2d` |
| O0 raw | `d94f9815a061c7bd533193f24335933aeb699d948e8af02ca199cdd982420492` |
| O0 formal | `8c09ccfccc9946bf95892cb1c77aa80ce60331e79394deb0ebb9f5372602f69b` |
| O0 manifest | `71a81418d40d7cb152cbf1a45bd87f7afd50c7973c181da6995788160de15c95` |
| Q1 wrapper | `2673de4e7c4aa4ee0fd20636b17bf2c2828103bacfbbb05b2ff0f559979434e6` |
| Q1 binary | `9df8110f1e2134f8a6d88cde46d3a8a6278590f291ad87b49161926d2fd53f41` |

O0 verdict 必须为 `O0_PN_PROXY_INVENTORY_QUALIFIED`。panel rule 与 panel SHA256 必须保持：

`cc011844e8ed4e415353e8a835bca04a0e091d0179aaccef94e86d3e426d5301`

固定 panel（node index / source ID / cell type）：

1. `18051 / 30021 / M_lPNm11C`
2. `130497 / 520003 / DL1_adPN`
3. `18497 / 30539 / M_l2PNm17`
4. `6892 / 17540 / DL2v_adPN`
5. `11829 / 23045 / DL3_lPN`
6. `31238 / 44995 / M_lPNm11D`
7. `2769 / 12958 / DL2d_adPN`
8. `571 / 10613 / DM4_adPN`

每个节点必须仍为 O0 eligible 集合成员、acetylcholine、收到 ORN 输入并直接连 KC。不得按 O1 输出替换 panel。

## 2. 动力学与唯一输入

使用固定 Q1 binary 的 empty-graded-mask v1 compatibility path；学习永久关闭，weights 不变。rest/KC adaptation、delay、refractory 和 event semantics 沿用冻结 Q1/v1。

每个实验从 clean rest、zero conductance、zero drive、zero queue 开始。唯一输入是一个序列化的：

`engineered_pn_spike_event_intent(source_id, node_index, emission_clock=0, count=1)`

执行方式冻结为：在 advance 前把该 PN index 放入 emission clock 0 对应的 future arrival ring slot `delay_steps mod (delay_steps+1)`，该 event 在 `1.8ms` arrival clock 按原图全部出边传递。

这表示工程化 sensory proxy event，不声称模拟 ORN transduction、气味或 PN 自发电位。不得用 external current 把 PN 推过阈值，不得直接修改 KC/MBON，不得提交第二个 event，也不得按结果重试。

## 3. Trials

在 `dt=0.1/0.05/0.025ms`：

- `ZERO` clean control 两次；
- panel 八个 PN 各自独立 single-event 两次；
- 共 `3 × 10 × 2 = 60` 个 clean-start trials。

每个 trial 总长 2,000ms，固定 segment 为：

`[2, 48, 450, 500, 500, 500]ms`

保存每段 spike counts、KC spike counts、active nodes、queue counts、voltage/g bounds、state hash。2ms boundary 用于确认 direct PN targets/KC 收到 event；最后 500ms 用于退出门。

## 4. Gates

### 4.1 Intent 与原图传播

- single-event trial 的 intent 恰为 1，source 与 panel 匹配；ZERO 为 0；
- queue 初始除冻结 arrival slot 的一个 source index 外全零；
- source 的所有原图出边在 arrival 时按固定 weight 传递，不筛 target；
- 2ms boundary 至少一个 direct target `g_event != 0`；
- 若该 PN 有 direct KC edges，则至少一个 direct KC `g_event != 0`；KC 是否 spike 只报告不门控。

### 4.2 安全、退出与确定性

- 所有状态 finite，`abs(v)<1e6`、`abs(g)<1e9`，queue count 在 `[0,n]`，overflow=0；
- weights SHA256 start/end 一致；
- 最后 500ms spike count 必须为 0；
- final queue empty，final `g_event` 与 adaptation 在数值静息容差内，final voltage 回到 rest 的 float32 `4 ULP` 内；
- 两次重复完整 final-state hash、segment counts 与 intent ledger 相同；
- 三个 dt 的 stable/exit class 一致；
- ZERO 无 spike、无 conductance、除 clock 外 physical state 保持 rest。

任一 single event 触发持续发放、吸收上态、overflow 或无法退出，即停止 O 路线；不得降低原图权重、提高阈值或缩小 event。

## 5. 判定

只允许：

1. `O1_IMPLEMENTATION_FAILURE`：pin、panel、intent、实际报告路径、测试、hash stability 或产物缺失；
2. `O1_PROPAGATION_FAILURE`：event 未按完整原图到达 direct target/KC；
3. `O1_SINGLE_EVENT_INSTABILITY`：finite/queue/退出/determinism/dt class 任一失败；
4. `O1_SINGLE_EVENT_QUALIFIED`。

`O1_SINGLE_EVENT_QUALIFIED` 只说明单个 PN proxy event 可安全传播并退出，不说明有可分辨 KC code、学习或交易能力。

## 6. 产物与后续

正式目录：

`runs/whole_brain_olfactory/O1/20261010T-pn-single-event-v1/`

至少包含 `o1_raw.npz`、`o1_formal.json`、`o1_report.md`、`o1_manifest.json`、`launch.txt`、`stdout.txt`。正式前必须通过 micro event-injection tests、duplicate/invalid intent rejection、实际 CLI smoke 和一个完整图 10ms preflight。长跑使用 tmux。

若 O1 通过，下一步 O2 才能冻结多 PN coincidence/synthetic coding；feature mapping、pulse count/rate 必须在查看 O2 响应前冻结。sensory adapter 集成、plasticity、S3 和市场训练继续阻塞。
