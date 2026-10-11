# V2-Q1 v6 微回路与自动测试审计

状态：**2026-10-09 正式审计。v6 的 `Q1_IMPLEMENTATION_QUALIFIED` 无效，冻结改判为 `Q1_IMPLEMENTATION_FAILURE_INCOMPLETE_MICRO_GATES`。**

本审计保留 v2–v6 全部失败/候选产物，不修改原 Q1 方程、参数、刺激或门槛。v6 已补齐多项 v1 缺失内容，但仍未满足冻结的微回路和自动测试资格，因此不能授权 V2-C0。

## 1. v6 已有效补齐的部分

- v2/v3 基础设施失败均在下一版本启动前写入 amendment；
- v4 的未初始化 `active` 尾部导致 observer/repeat hash 失败，被保留并在 v5 前记录；
- v5 artifact 缺失在 v6 前记录；
- v6 有 24 个 full-graph hybrid trial，重复 hash、finite、bounds、queue count、APL negative delivery 和 fine-dt 汇总均报告通过；
- empty-mask v2 binary 通过私有 symbol 调用固定 `kernel.cpp`，并与独立 v1 `MemoryBrain` 在三档 dt、三个 segment、完整图状态字段上 bitwise 相同；
- observer ledger on/off 的主状态 bitwise 相同；
- source/binary/runner/tests 的 run-start/run-end hashes 相同；
- clean-state schema 与 raw JSON traces 被写入 v6 产物。

这些结果可保留为局部门禁证据，但不覆盖以下缺口。

## 2. 自动测试仍是旧版

`tests/test_v2_q1_hybrid_qualification.py` SHA256 仍为 v1 的
`2b70069a9d7d10d976d0aa94a08dada65020189f5bc1fbaf2a26bde0d37ebec6`，并继续导入旧 runner `run_v2_q1_hybrid_qualification.py`，没有导入或测试正式 v2 runner。

现有 5 tests 没有覆盖：

- v2 runner verdict precedence；
- embedded-v1 compatibility；
- full refractory audit；
-有效的 fine-dt delivery convergence；
- 24-trial/repeat gate；
- ledger on/off；
- run-start/run-end hash gate；
-失败产物与版本目录。

因此“tests hashes 稳定”只证明旧测试文件没变，不证明正式 v2 实现被自动测试覆盖，违反原预注册 §7 和 v1 audit §4.2。

## 3. A2 没有逐边验证

A2 fixture 包含 self/weak/positive/negative edges，但 gate 只检查聚合 `signed_delivery != 0`、`absolute_delivery > 0` 和一个 target 状态首次变化时钟。它没有记录或断言每条边的 `w*r*dt/tau_emit`，无法证明四条边都被传递；正负边的抵消也可能隐藏单边错误。

`event_probe_clock` 在三档 dt 均为 null；`pre_arrival_event_zero` 只比较早期 `g_event`，不是 graded queue/delivery 的逐步零值证明。

## 4. A4 没有验证四类连接

A4 的正式布尔值实际是：

- 一个合并的 `spiking_to_spiking_or_graded_event`；
- `graded_to_graded` 非零；
- `graded_to_spiking_state` 仅检查 voltage finite；
- 总 spike count 非零。

它没有分别核对 spiking→spiking、spiking→graded、graded→graded、graded→spiking 的逐边 arrival、符号、事件数量和 target 状态。`graded_to_spiking_state = finite` 不能证明该边产生了任何影响。

## 5. A5 只检查 refractory 第一步

A5 只在 dt=0.1 下设置 refractory 后运行一个 0.1 ms step，确认 voltage 暂时不变且 `g_graded>0`。它没有：

- 覆盖三档 dt；
- 覆盖完整 2.2 ms refractory；
- 比较每步 `g_graded` 解析值；
- 检查解除 refractory 后首步 voltage 的解析响应；
- 设置离散 spike input 对照并验证 v1 refractory-drop。

因此不能满足冻结 A5。

## 6. A7 使用无效 CSR，delivery convergence 是伪通过

A7 fixture 写入：

`ptr = [1, 1, 1]`，`post = [1]`。

合法 CSR 必须从 `ptr[0] == 0` 开始。当前 `HybridBrain` 只检查 `ptr[-1] == len(post)`，没有检查 `ptr[0]` 和单调性，因此接受了无效 fixture。对 source 0 而言，遍历范围 `ptr[0]:ptr[1]` 为 `1:1`，没有任何出边；signed/absolute delivery 在 0.05 和 0.025 两档均为零，`relerr(0,0)` 返回零，于是两个 delivery convergence 指标伪通过。

v7 必须使用合法 CSR（例如两节点单边 `ptr=[0,1,1]`），并明确要求两档 signed/absolute delivery 均非零后才计算 `<=0.02` convergence。`HybridBrain` 本身也必须拒绝 `ptr[0] != 0`、非单调 ptr、post 越界与非有限 weight。

## 7. compatibility 仍缺微回路覆盖

v6 的完整图 empty-mask compatibility 是有效的新证据，但原预注册还要求 compatibility 微回路覆盖兴奋/抑制、自连接、弱边、重复发放、refractory 内到达、modulatory source 和 KC adaptation。v6 没有独立的 empty-mask v1/v2 compatibility micro suite；hybrid A2–A5 不能替代该门禁。

## 8. full-graph 审计的小缺口

- trial 中 `weights_before_sha256` 与 `weights_after_sha256` 都在运行结束后从同一 array 读取；应在第一段运行前保存 before hash，最后再读 after hash。
- queue integrity 没有正式记录 `overflow == 0`、graded queue finite/count 语义；应写出。
- graph contact count 来自冻结常量而非本次从 normalized edge table 聚合复核；固定 hash 虽通过，正式 graph fixture 仍应实际读取并求和。

## 9. v7 补跑约束

1. 新正式目录使用 `runs/whole_brain_v2/Q1/20261009T-hybrid-kernel-qualification-v7/`；v2–v6 不覆盖。
2. 先改写自动测试，使其直接导入正式 v2 runner，并覆盖 §2–§8；测试通过后才可启动正式 run。
3. A2 写出逐边 expected/observed delivery 与 exact clocks；任何边缺失即失败。
4. A4 分别写出四类连接的 arrival、sign、count/state delta，不能用 finite 代替影响。
5. A5 在三档 dt 覆盖完整 refractory、解析 `g_graded`、解除后 voltage 及离散事件 drop 对照。
6. A7 使用合法非零边 fixture，并增加 nonzero-delivery gate；同时增加 CSR validation tests。
7. 增加 empty-mask v1/v2 compatibility micro suite，覆盖原预注册列出的全部语义。
8. 修复 before/after weight hash 和完整 queue/overflow/contact 守恒记录。
9. 任一必须 gate 缺失时 verdict 必须为 `Q1_IMPLEMENTATION_FAILURE`，不得产生 `QUALIFIED`。

## 10. 结论

v6 表明 full-graph hybrid smoke 与完整图 empty-mask compatibility 已取得实质进展，但微回路资格仍不完整，且 A7 存在明确伪通过。Q1 尚未完成，V2-C0、S3 与市场训练继续阻塞。
