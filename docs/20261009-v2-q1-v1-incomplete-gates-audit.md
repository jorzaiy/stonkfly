# V2-Q1 v1 不完整门禁审计与 v2 补跑约束

状态：**2026-10-09 正式审计。v1 的 `Q1_IMPLEMENTATION_QUALIFIED` 无效，冻结改判为 `Q1_IMPLEMENTATION_FAILURE_INCOMPLETE_GATES`。**

本审计不改 `docs/20261009-v2-q1-hybrid-kernel-preregistered.md` 的方程、参数、刺激、容限或判定门槛。v1 代码和产物全部保留，只能在新目录补跑。

## 1. v1 产物状态

目录：`runs/whole_brain_v2/Q1/20261009T-hybrid-kernel-qualification-v1/`

现有 `q1_formal.json` 自报 `Q1_IMPLEMENTATION_QUALIFIED`，但没有执行冻结预注册要求的全部必要门禁，因此该 verdict 不具正式效力。它可以保留为代码 smoke 证据，不能授权 V2-C0。

此外，第一次运行因 `numpy.bool_` 无法 JSON 序列化而失败；随后在同一 v1 目录内修复并重跑，没有先写 amendment、也没有使用新版本目录。这违反了原预注册的失败保留规则，进一步阻止 v1 成为正式通过产物。

## 2. 已确认完成的有限内容

- 固定输入 SHA256 与 P0 READY 状态被检查；
- P0 分类计数与 source-id hashes 被写出；
- isolated v2 translation unit 成功编译；
- release clamp 的简单数组检查通过；
- 12 个单次 full-graph 40 ms smoke trial 完成且 finite；
- ZERO 三档 dt 没有 spike/release/delivery；
- 三个刺激臂产生非零 release/delivery，APL signed delivery 为负；
- 现有 5 个自动测试通过。

这些内容不足以满足 Q1。

## 3. 缺失或无效的冻结门禁

### 3.1 v1 compatibility 完全缺失

正式 JSON 没有 compatibility 字段或 trial。没有把 empty graded mask 的 v2 path 与固定 v1 kernel 在微回路和完整图上进行 bitwise 比较，也没有比较预注册列出的 `v/g/refractory/drive/previous_drive/queue/counts/active/last/adaptation/modulation/clock/total_spikes`。

当前 v2 state 甚至没有完整的 v1 `last`、modulation 和 eligibility 状态；KC adaptation 在内核中衰减/跳增，但没有按 v1 公式作用到 voltage。因此不能用现有代码结构推定 compatibility。

### 3.2 微回路解析门禁没有实际执行

runner 的 `analytic_segment` 只检查结果 finite，没有与解析参考比较误差；现有测试同样只断言 finite。以下冻结项目没有正式结果：

- 正/负边与 self/weak edge 的逐边 delivery；
- 精确 arrival clock 和首次影响 voltage 的 clock；
- 四类连接的逐边断言；
- graded input 跨完整 refractory 的 `g_graded`/voltage 解析值；
- 离散 spike refractory-drop 对照；
- 100 ms zero-input equilibrium；
- 0.05 vs 0.025 ms 的四项 fine-dt convergence。

现有 `test_four_connection_types_and_exact_arrival` 只检查非零/finite，没有验证四类连接或精确时钟。

### 3.3 完整图重复、dt 与 bounds 门禁缺失

正式结果只有 12 个 trial：3 dt × 4 arms × 1 次；预注册要求 24 个独立 trial，即每个组合两次。没有：

- 两次规范化状态/账本 SHA256 相等门禁；
- 0.05 vs 0.025 的 release/signed/absolute delivery 对称相对差；
- `abs(v/g_event/g_graded)` bounds；
- queue count、clock、ring-buffer 的正式完整性字段；
- 完整 weight array 运行前后 bitwise hash；
- source/runner/tests/binary 正式运行前后 hash 不变。

### 3.4 Observer equivalence 不成立为正式门禁

现有 `observer` 参数未改变或启用任何 observer/ledger 路径，测试只是比较两个等价 no-op 调用。正式 JSON 也没有 observer gate。v2 必须显式提供 ledger on/off，证明关闭审计写入后主动力学状态与 counts bitwise 相同。

### 3.5 Verdict 逻辑不完整

v1 runner 只以 fixed hashes、两个浅层 micro 布尔值、finite 和 nonzero smoke 决定 `QUALIFIED`，没有把 compatibility、解析误差、delay/refractory、重复确定性、fine-dt、bounds、observer 和权重冻结纳入 verdict。因此 verdict 逻辑本身违反预注册。

## 4. v2 补跑的冻结要求

1. 正式目录必须为 `runs/whole_brain_v2/Q1/20261009T-hybrid-kernel-qualification-v2/`；v1 不得覆盖或删除。
2. 先补齐自动测试；每一项预注册 gate 必须在 `q1_formal.json` 有独立布尔值、实测值、门槛和证据 hash。
3. empty-mask compatibility 必须真实调用固定 v1 实现与 v2 compatibility path；不能把两个对象都指向同一 no-op helper 来制造相等。
4. 如果当前 v2 内核结构不能表达 v1 的完整状态/语义，必须重构隔离 v2 实现；不得降低 bitwise 门槛。
5. 正式 full-graph 必须执行 24 个独立 hybrid smoke trial，并另执行预注册的三档 dt 完整图 compatibility 对照。
6. observer/ledger on 与 off 必须运行独立对象，比较全部主状态；off 路径不得累计审计统计，但动力学必须相同。
7. v2 source、runner、tests、binary 的 run-start/run-end SHA256 必须一致并进入 manifest/formal。
8. verdict 优先级严格沿用原预注册。任何 gate 缺失应判 `Q1_IMPLEMENTATION_FAILURE`，不得把缺失当作通过。
9. 若出现新的纯实现错误，必须先写 amendment，再使用 v3 目录；禁止在 v2 正式目录就地重跑成功。

## 5. 结论边界

v1 没有证明 hybrid kernel 错误，也没有证明它正确；它只说明一个不完整 smoke runner 能运行。V2-C0、S3、市场 replay 和交易训练继续阻塞，直到新的正式 Q1 结果完整通过。
