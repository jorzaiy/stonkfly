# Stonkfly v2 M2-E 实现准备与 toy/smoke 审查

状态：**PREPARATION COMPLETE / FULL 8-PAIR MATRIX NOT AUTHORIZED**
日期：2026-10-12 UTC

## 结论

建议对固定设计的 M2-E 8 气味对探索矩阵作 **GO（等待另行授权）**。这只表示实现、最小机制测试和容量证据足以开始一次探索运行，不是学习成功结论，也不改变 M1-RQ、M1-RT、历史 M2 或 E1。

## 实现

- `stonkfly/mb_lab/e1.py`：只增加可选 10 ms KC trace observer；observer off/on 的 E1-G integral、population trace、final state 等价，未改变感觉参数或方程。
- `stonkfly/mb_lab/m2e.py`：新增右侧真实 KC→MBON07/11 映射、两个固定归一化输出、独立 plasticity state、paired/unpaired/DAN-omitted 时序和 exact-key E1-G 感觉缓存。
- `protocols/mb_lab/m2_e/config.json`：状态固定为 `AUTHORIZED_TOY_SMOKE_ONLY`，`full_exploration.authorized=false`。
- `experiments/mb_lab/run_m2e_smoke.py`：只允许 2 odors × 2 repeats、2 cycles；已有目录拒绝覆盖，保存结果、资源和哈希 manifest。
- `experiments/mb_lab/m2e_attempt.py`：append-only toy/smoke 尝试账本。首轮 fixture 维度错误及其失败日志永久保留。

没有导入或调用市场、交易、执行、旧 RT 或旧 M2 路径。

## 测试结果

最终：`14 passed in 6.97s`。

已验证：

- odor→DAN 的 paired 顺序产生目标通路 depression；DAN→odor 产生相反方向；
- offset 后 6 s 的 unpaired 效应显著衰减；
- frozen weights 与 DAN omitted 的权重 bitwise 不变；
- 平衡 shuffled reward 不形成 toy 的气味差异；
- 已学习权重在 reward-free probe 中不变；显式 blank 只按 900 s memory return 被动衰减；
- 真实右侧通道 inventory 为 MBON07 1,806 edges/931 input KCs、MBON11 1,705 edges/1,705 input KCs；
- 五臂共享同一不可写感觉对象，但不共享 eligibility、memory 或 weights；
- M1-E1 回归测试全部通过。

## 真实拓扑 smoke

运行：`runs/mb_lab/m2-e-preparation/smoke-20261012T052449Z/`

- 状态 `M2_E_SMOKE_COMPLETE`，退出码 0；
- 4 个唯一感觉计算，后续五臂访问产生 4 个 exact-key cache hits；
- E1-G observer-on 与独立 observer-off 重算等价；
- 两个 MBON 通道在全部 4 个 smoke 刺激中均为正；
- 未强化 raw approach/avoidance ratio 为 0.682–1.270，说明存在显著的气味相关结构偏置；固定的 odor-specific baseline normalization 正确地把 naive `V` 置零；
- paired：`D(A-B)=+0.00206145`，A 为正、B 为负；
- shuffled：`D=-0.00035078`，表明有限次序/trace 会留下小残差，完整矩阵必须依靠预注册 replicate 对照，不能要求单次恰为零；
- unpaired：`D=+0.00001007`，约为 paired 幅度的 0.49%；
- frozen、DAN-omitted：`D=0`、0 changed edges；
- paired/shuffled/unpaired 各有 786 个浮点非零 edge deviation；无臂触及 efficacy bound。

smoke 只能说明机制方向和对照路径正确；2 cycles 的效应量不构成 acquisition、reversal 或 retention 科学结果。

## 预算

- smoke：18.21 CPU s、18.22 wall s、峰值 RSS 676.4 MiB；实际目录 9,463 bytes；
- 4 个正式尺度 E1-G 感觉路径：11.74 CPU s；
- 按 640 paths 线性外推并乘 1.5 安全系数：0.783 CPU-hours；
- 低于固定 2 CPU-hours、2 wall-hours、2 GiB RSS、250 MiB 上限。

主要预算风险是 KC trace observer 的内存和 I/O，而不是 plasticity head。正式实现应逐 trial 消费 trace，不累计全部 640 个 `(200,1907)` 数组。

## GO 的限制

若获得探索运行授权，只能运行设计文档中的一次固定 8-pair/五臂矩阵。不得根据 smoke 的效应大小修改 `eta=0.1 s^-1`、DAN pulse、气味、cycle 数、归一化或成功条件。完整运行结束后必须停止并报告 `ENGINEERED_ASSOCIATIVE_UPDATE_OBSERVED` 或 `...NOT_OBSERVED`；两者都不升级旧 M1，也不支持真实果蝇或交易学习主张。

## 证据哈希

- smoke manifest：`928d8f8235eb3f58718c4ba5e38033544866af5b2a978237207607fbd4bce3d8`
- smoke result：`f2f8fb77fc42798703e1ab2716b5b9ee0c037ed82ccf137dafdb4c8a3b094476`
- config：`2b67bbb4f7e20abe9319b5cd79608eab272ac7e07da39e41483ae058985d804b`
- `m2e.py`：`65588dde8f484dd1d60cb570a968003f18b8fefc6efd5b3b95aeae0cf03cfc28`
- final tests：`234a9759499013fd843e2f31c90d9187453d398a410b577f0c7e63333b9d07eb`
