# Stonkfly v2 M1 freeze-v5 最小协议修订草案

状态：**DRAFT FOR INDEPENDENT REVIEW / M1-RQ、M1-RT、M2 全部 NO-GO**
范围：只补齐 freeze-v4 无法实现正式 runner 的最小协议缺口。不是工程实现，不生成或解密正式集合，不修改模型参数、刺激规则、拓扑或科学效应门槛。

机器可读草案：`protocols/mb_lab/m1_freeze_v5/AMENDMENT_DRAFT.json`。

## 1. 为什么需要 v5

freeze-v4 固定了方程、参数、刺激、S1–S5 和大部分 I8 门，但没有固定 S2 permutation RNG seed、I8 跨大量 trial 的聚合语义、RQ-only 成功 verdict 和 ZERO 的确切重复数。这些缺口会影响 p 值、数值失败判定或预算，不能由 runner 作者自行补全。

本文件中的每一项新增内容都明确是 **freeze-v5 协议变更**，不声称 freeze-v4 已经规定。v4 的文件、密封载荷、哈希和两次 NO-GO 记录永久保留。

## 2. 协议差异总表

| 项目 | freeze-v4 | freeze-v5 草案 | 是否改变模型/任务 |
|---|---|---|---|
| S2 permutation seed | 只写“fixed seed”，未定义 | 公开域标签经 SHA-256 派生固定 uint64，并固定 PCG64 和 draw order | 改变统计复现规范；不改刺激或模型 |
| I8 聚合 | 有阈值，无精确聚合单位 | 每 topology/path × odor × trial 计算，取最坏 active-fraction error、NRMSE 和 Jaccard | 改变数值门解释；不改阈值 |
| I8 halving `E` | 未定义 | 用共同 0.025-ms reference scale 定义逐单位 `E_coarse/E_fine` | 改变数值门解释 |
| 科学 gate 跨 dt | “所有分类一致” | 固定比较 `(S1_B,S2_B,S3_B,S4_B_APL,S5_B_vs_39C)` 五元布尔向量 | 澄清适用范围 |
| RQ 成功 verdict | 只有要求 RQ+RT 的最终 M1 verdict | 新增 `M1_RQ_QUALIFIED`；只表示 RQ 资格通过 | 防止提前宣称最终成功 |
| ZERO | 数量未定 | 每 dt、APL on/off 各 10 repeats，共每 dt 20 个；不计入 160 odor trials | 增加明确控制工作量 |
| C 输出缺失 | 未定 | 非有限/图不变量失败属于 implementation failure；有限但 `T_C` 不可定义则保留并以 `+∞` 进入 S5 | 防止删除不利 control |
| 密封 | v4 已存在 | 四域使用全新 seeds 形成 v5；topology seeds 不变 | 正式集合重新密封，v4 永久保留 |

明确不变：`tau_K=10 ms`、`tau_A=50 ms`、`s=1`、`g_A=1`、`theta=0.35`、active threshold、2,000-ms 时间线、29/145 PN 刺激规则、16×10 正式 odor trials、B topology、39 个 C seeds/hashes、S1–S5 效应门槛、1,907/2,045 双分母和硬预算上限。

## 3. S2 确定性 RNG

规范化步骤：

1. 输入是下列精确 ASCII 字符串；按 UTF-8 编码，无 BOM、无终止 NUL、无前后空白、无换行。
2. 对编码字节计算 SHA-256，得到 32 bytes。
3. 取 digest 的前 8 bytes，按 unsigned little-endian 解释为 `[0,2^64-1]` 的整数。
4. 使用 `numpy.random.Generator(numpy.random.PCG64(seed_uint64))`。
5. 一次生成 shape `(9999,16)`、`dtype=uint8`、范围 `[0,2)` 的 C-order draws；0 映射 −1，1 映射 +1。第 `j` 列只作用于 odor `j` 的 `R_o-D_o`。
6. `p=(1 + count(T_perm >= T_obs-1e-15))/10000`。不根据结果换 seed 或增加 permutations。

RQ：

- label：`STONKFLY-V2/M1/FREEZE-V5/RQ/S2-SIGNFLIP/v1`
- digest：`013d654c748b4da8c88bf6e0df8c37880e7441ff144ddcd4b76f903ddd6cce76`
- seed：`12127502703192784129`

RT 预先公开但仍禁止使用：

- label：`STONKFLY-V2/M1/FREEZE-V5/RT/S2-SIGNFLIP/v1`
- digest：`085d908c1ae25d611e2685288ebc5cf01fcf9ff46821cba1f77ca34422f08edf`
- seed：`7016012398168399112`

这些是统计 RNG seeds，不是刺激 seeds，也不能推导刺激内容。

## 4. I8 的逐单位定义

### 4.1 范围

stimulated paths 共 41 条：B APL-on、B APL-off、39 个 C APL-on。每条 path 对 16 odors ×10 trials，在三档 dt 上使用完全相同的已密封 PN 输入。I8 单位是一个 `path × odor × trial`，不得先按 odor、topology 或 trial 求平均。

ZERO 另按第 6 节执行，属于 I7/I8 实施控制，不混入 160 个正式 odor trials。

### 4.2 误差

令 `q_d` 为指定 dt 下、500-ms response window 的每 KC mean proxy rate，长度为 1,907；`z_d` 使用冻结 active threshold。

- active-fraction error：`A_d=|mean(z_d)-mean(z_0.025)|`；
- `NRMSE_d=RMS(q_d-q_0.025)/max(RMS(q_0.025),0.05)`；
- active Jaccard：普通交并比。仅对 I8 数值同一性，两边都空定义为 1，一边为空定义为 0；科学 S1/S2 中双空仍是 missing/failure，不能借此变成成功。

聚合采用保守最坏值：所有单位上的 `max A_d`、`max NRMSE_d` 和 `min Jaccard_d`。现有阈值逐单位保持不变：0.1 ms 为 `0.005/0.05/0.90`，0.05 ms 为 `0.0025/0.025/0.95`。

步长减半对每个单位使用同一 reference scale `max(RMS(q_0.025),0.05)`：

- `E_coarse=RMS(q_0.1-q_0.05)/scale`；
- `E_fine=RMS(q_0.05-q_0.025)/scale`；
- 两者均 `≤1e-4` 时通过 floor；否则必须 `E_fine≤0.6 E_coarse`。最终报告最大有限 ratio、floor 命中数和全部失败单位，不只给汇总。

### 4.3 阈值适用性审查

采用最大误差后，原阈值不再是“平均表现”，而是 6,560 个 stimulated path-trials 中每一个都必须满足的确定性数值容差。这没有推断统计上的多重比较问题，但会显著提高 near-threshold active set 导致 I8 失败的概率，尤其是 2% 稀疏时少数 KC 翻转即可压低 Jaccard。

草案建议暂不放宽阈值：先用 toy 和无正式刺激的代表性解析输入检查最终 runner。若最大误差失败，应优先修复积分/缓存实现；若确认是阈值与 max 聚合结构性不相容，则返回独立协议评审，不能查看正式输出后调整。因此这些阈值在 v5 获批前仍标为“待独立接受”，而非自动沿用结论。

### 4.4 跨 dt 科学分类

每档 dt 独立计算布尔向量：

`(S1_B, S2_B, S3_B, S4_B_APL, S5_B_vs_all_39C)`。

三个向量必须完全相同。S1–S3 只适用于 B APL-on；S4 只适用于 paired B on/off；S5 使用 B 和全部 39 个 C。科学主报告和 RQ verdict 仍只使用 0.025 ms 数值。

## 5. 随机对照与失败语义

- 所有 C 必须使用原有 39 个 topology seeds；freeze-v5 不重新挑图。
- graph hash、逐 PN 出度、逐 KC 入度或唯一性失败：`M1_IMPLEMENTATION_FAILURE`。
- C 数值非有限：`M1_IMPLEMENTATION_FAILURE`，不得替换该 C。
- C 数值有限，但因全空 active sets 令科学 `T_C` 不可定义：保留该 control，S5 中令 `T_C=+∞`，同时报告具体 missing 项。这是对 B 最保守的处理，避免把失败 C 删除后人为提高 B 的 rank。
- B 的空响应按 S1/S2 直接失败；不得使用上述 `+∞` 规则挽救 B。
- `p_topo=(1+#(T_C≥T_B))/40` 和 effect floor `T_B-median(T_C)≥0.05` 不变。

RQ 错误优先级：

1. `M1_IMPLEMENTATION_FAILURE`：provenance、授权/隔离、图不变量、RNG、oracle、finite 或 artifact 完整性失败；
2. `M1_BUDGET_EXHAUSTED`：尚无更高优先级错误时触及任一硬预算；
3. `M1_NUMERICAL_NO_GO`：I8 失败；
4. `M1_REPRESENTATION_NO_GO`：I gates 通过但 B 的 S1–S4 任一失败；
5. `M1_RQ_QUALIFIED`：I1–I8 与 B S1–S4 全部通过。

`M1_RQ_QUALIFIED` 只附带 `SUPPORTED_AT_RQ_ONLY` 或 `NOT_DEMONSTRATED_AT_RQ` 的 S5 子状态。两者都不构成最终 M1 成功，不解封 RT，不自动进入下一阶段。

## 6. ZERO 修订

每档 dt 分别运行：APL-on 10 repeats、APL-off 10 repeats。每个 repeat 使用全程精确 float64 零 PN 向量、相同零初态，无随机噪声。共 20 ZERO paths/dt，不计入 160 个正式 odor trials，也不参与 S1–S5。

10 repeats 的科学信息并不独立；它们用于验证调度、checkpoint、缓存和 trial identity 不会引入状态泄漏。要求所有重复 bitwise 相同、finite、无 active KC、终态为解析容差内零，且三档 dt 的 active sets 一致。

## 7. 预算影响

旧 8.59 CPU-hours 不能继续作为 v5 预算结论，因为它没有绑定最终 runner hash，且旧 `43/40` 因子没有清楚表达 ZERO 数量。

v5 每个 dt 的任务单位为：

- B+39C APL-on：`40×160=6,400`；
- B APL-off：`1×160=160`；
- ZERO on/off：`2×10=20`；
- 合计：6,580 path-trials/dt。

仅把旧代表性 benchmark 按 `2,000/250 × (6,580/40) ×1.25` 归一化，得到约 **8.215 CPU-hours/RQ**。这是工作量参考，不是通过结论；它不充分测量 checkpoint、hash、统计、压缩、异常保留和最终 runner 的开销。

硬上限维持 24 CPU-hours、24 wall-hours、4 GiB peak RSS、5 GiB artifacts。v5 密封前必须用最终 runner、相同代码哈希和 41 条 stimulated paths 加 ZERO 调度重新做无正式刺激 benchmark；只有含工程开销的保守估算低于硬上限，预算门才可通过。

## 8. freeze-v5 重新密封方案

获得本修订独立批准后，另行执行密封准备：

1. 为 M1-RQ、M1-RT、M2-LQ、M2-LT 生成四个全新随机 seeds；不得复用或解密 v4 seeds。
2. M1 刺激仍使用完全相同的生成规则，但形成新的 content hashes；M2 仍仅为 reservation，不实现学习内容。
3. topology seeds 和 40 个 topology hashes保持不变。
4. v5 manifest 承诺一个自包含 protocol bundle：配置、本文、预注册、runner、全部影响结果的 imported modules、stimulus/sealer、连接组 source locks、40 个 topology hashes、tests/oracle、environment lock、授权公钥、容量证据。
5. v4 目录、密文、keys、manifest 和哈希不修改、不删除、不改名。v5 新建 sidecar，引用四个 v4 manifest hashes，并标记 `SUPERSEDED_PROTOCOL_NOT_RUN`。
6. v5 使用新的目录与 schema version；所有写入 refuse-overwrite。

未来工程方案应让 RQ 进程运行在独立 OS identity/mount namespace：只读挂载 RQ key/payload，RT 与 M2 custody paths 不挂载且由不同 owner 持有；代码/协议只读、唯一 run 目录可写。验收必须证明 `test -r` 对 RT/M2 返回非零，并记录 mount table 与 UID/GID。这里只规定方案，本轮不创建用户、不改权限。

RQ 授权应是单独签名 artifact，只允许 `M1_RQ`，并绑定 protocol-bundle hash、RQ manifest hash、唯一 run ID 和输出目录。不存在有效授权时 fail-closed；RT/M2 不得共享授权入口。

## 9. 剩余风险与独立审阅问题

- max-I8 很保守，可能因单个 near-threshold KC 产生 numerical NO-GO；这应被视为预注册选择，而不是运行后调参理由。
- `T_C=+∞` 会保守地削弱 topology 优势证据，但避免删除退化 control；评审者应明确接受或提出另一种运行前规则。
- 公开 PCG64 统计 seed 提高可复现性，但 Monte Carlo p 仍有有限分辨率；9,999 次数不变。
- ZERO 10 repeats主要检测状态泄漏，不提供十个独立科学样本。
- 8.215 CPU-hours 只是重新归一化参考；最终 hash-bound benchmark 仍是工程 GO 前置条件。
- 重新密封会使 v5 刺激与 v4 不同，但不是为了选择更有利样本；这是统计规则变化后的干净协议版本。

请独立评审明确批准或否决：S2 派生规范、max-I8 定义及原阈值、C missing 的 `+∞` 规则、RQ verdict、ZERO 10×2 设计、v5 全新四域 seeds，以及重新 benchmark 要求。在批准前，所有正式阶段继续 NO-GO。
