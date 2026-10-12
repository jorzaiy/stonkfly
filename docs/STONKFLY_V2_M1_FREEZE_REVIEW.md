# Stonkfly v2 M1 冻结准备审查报告

日期：2026-10-11
结论：**冻结候选包已形成；建议提交独立 GO/NO-GO。当前不得运行 M1-RQ、M1-RT 或 M2。**

## 1. 交付物

- 唯一候选配置：`protocols/mb_lab/m1_v1/frozen_config.json`；
- M1-RQ 预注册草案：`docs/STONKFLY_V2_M1_RQ_PREREG_DRAFT.md`；
- 四域最终 commitments：`protocols/mb_lab/m1_v1/domains-freeze-v4/`；
- 真实规模 benchmark 与 40 个 topology hashes：`protocols/mb_lab/m1_v1/capacity_benchmark_v2.json`；
- 候选 runner：`experiments/mb_lab/run_m1_preregistered.py`；
- domain sealer：`experiments/mb_lab/seal_m1_domains.py`；
- 基础与冻结测试：`tests/mb_lab/test_m1_foundation.py`、`tests/mb_lab/test_m1_freeze.py`。

## 2. 审查中发现并修复的问题

1. 2,045 个右侧 KC 中 138 个没有合格 PN 输入。主分母冻结为 1,907 covered KC；全部 2,045 的零填充视图必须并列报告但不参与 gate。
2. 初稿以 `median_o(R_o-D_o)` 做 sign-flip；toy 证明在 even-n 与相同正 margin 下零分布严重离散、功效不符合检验目的。正式集合未运行前改为 `mean_o(R_o-D_o)`，median 仅描述。
3. freeze-v2 使用相对路径触发 sealer 失败；失败目录保留。freeze-v3 缺少充分 generator provenance；被 freeze-v4 取代。只有 freeze-v4 有效。
4. 首轮真实规模 benchmark 每 timestep 重算恒定 PN projection，估算 RQ+RT 138.37 CPU-hours，超过预算，判该实现 NO-GO。加入分段常数输入的输出等价缓存后，scalar oracle 和 bitwise 回归通过；相同 benchmark 降为 17.18 CPU-hours 的双阶段保守估算。首轮证据未覆盖。

## 3. 测试范围

最终命令 `.venv/bin/pytest -q tests/mb_lab/test_m1_foundation.py tests/mb_lab/test_m1_freeze.py`：**20 passed in 7.20 s**。候选 runner 的最终 toy-only CLI：`M1_BENCHMARK_ONLY_RUNNER_PATH_OK`；其 manifest 的 source start/end hashes 完全相同，且 `rq_accessed=false`、`rt_accessed=false`、`m2_accessed=false`。

冻结测试覆盖：

- vectorized equation 对独立 scalar oracle；
- piecewise-input cache 的 bitwise 等价；
- 0.1/0.05/0.025 ms toy convergence、active Jaccard 和撤刺激恢复；
- 39 个固定 control seeds、hash 唯一和逐节点度保持；
- 双 sparsity 分母；
- odor-level separation sign-flip、APL exact sign-flip 与 B/C rank p；
- 四个 public manifests 无 seed 泄露、commitments 唯一；
- RQ→RT cross-domain read 拒绝；
- runner AST/runtime forbidden-module isolation；
- sealed stage 拒绝发生在创建输出目录之前；
- 不可覆盖、完整 artifact chain、start/end provenance；
- CPU/RAM/artifact/wall 预算超限 fail-closed。

## 4. 资源证据

代表性 benchmark 使用真实 145 PN、1,907 KC、10,732 edges，B+39 C，三档 dt，每 network/dt 模拟 250 ms，输入为公开 `linspace(0,1,145)` 解析负载，不是任何 sealed stimulus，不计算科学指标。

| 版本 | 测得 CPU | 测得 wall | peak RSS | 双阶段线性估算 | 判定 |
|---|---:|---:|---:|---:|---|
| v1 无缓存 | 144.80 s | 144.83 s | 591,860 KiB | 138.37 CPU-h | 预算 NO-GO，保留 |
| v2 等价缓存 | 17.98 s | 17.98 s | 595,620 KiB | 17.18 CPU-h | 低于 48 CPU-h |

估算按 2,000/250 ×160 trials，再乘 43/40 计入 B APL-off/ZERO 路径，并乘 1.25 工程余量；不假设并行加速。优化版 graph build 为 26.18 s，是一次性成本。存储只保留摘要、rate vectors 和 bitsets；按数组尺寸预计低于 1 GiB，硬上限仍为 5 GiB。

限制：benchmark 是线性容量估算，不是 formal wall-time 保证；正式 runner 必须运行时计量并在 24 CPU-hours/stage、4 GiB RAM、5 GiB artifact 或 24 wall-hours任一触顶时停止。

## 5. 隔离与 provenance

最终四个 manifest 具有不同 seed commitment、content hash、ciphertext hash 与 key。M1-RQ/RT collection 在内存生成后取 hash，但没有进入动力学；M2 仅作 reservation。公开 manifest 不含 seed。密文/key 位于 git-ignored data 目录，mode 0600。

`run_m1_preregistered.py` 没有市场、reinforcement、actions、broker、risk、ledger、plasticity 或 M2 import；当前 CLI 只有 toy fixture 可执行，所有 sealed domain 均由 stage lock 拒绝。runner 记录 config、实现、tests 的起止 hashes，非空目录拒绝覆盖。

## 6. 静态审查结论与剩余风险

- 参数的生物定性约束有独立依据，但全部关键数值仍是 P 或 P(M)；因此即使通过也不能验证真实时间常数、APL gain 或 firing threshold。
- B/C 只隔离 adjacency；不能把 B 学会/通过表征门解释为真实拓扑优势。只有 S5 在 RQ 与 RT 重现，才支持限定的 proxy 内增量结论。
- 39 个 C 给 rank test 最小 p=0.025；`p≤0.05` 实际只允许至多一个 C 不低于 B，必须同时满足 0.05 effect floor。
- 全局单标量 APL 忽略局部性，是高风险代理假设。
- 同一 MaleCNS 个体/版本不能支持跨动物泛化。
- 正式 runner 的 sealed 数据解封和长程 formal 执行尚未获授权；本报告没有执行该路径。

## 7. 提交独立决策的问题

建议评审者分别回答：

1. 是否接受所有关键动力学数值均为 P/P(M)，并把 M1 定位为连接组约束工程机制实验？
2. 是否接受 covered-1,907 为主门、all-2,045 为强制描述性视图？
3. 是否接受 mean odor-margin + 9,999 sign-flip 和 39-control rank test？
4. 是否接受 17.18 CPU-hours 的线性保守估算与 48 CPU-hours 硬停止？
5. 若均接受，是否单独授权 **M1-RQ only**？M1-RT 与 M2 必须继续 NO-GO。
