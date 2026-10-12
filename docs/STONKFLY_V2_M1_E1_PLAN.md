# Stonkfly v2 M1-E1 最小探索性实验方案

状态：**PLAN ONLY / NOT RUN**
日期：2026-10-12 UTC
范围：仅 M1 表征动力学；不读取原 M1-RT，不修改 M1-RQ，不启动 M2，不接入市场或交易。

## 1. 研究问题与解释边界

问题：M1-RQ 使用的单一全局平均 APL 状态，是否掩盖了成人果蝇 APL 的局部反馈抑制可能产生的功能差异？

E1 是新探索，不沿用 RQ 的 S1–S5 通过门槛，也不产生 `M1_RQ_QUALIFIED`。它比较固定的三个机制臂并报告连续效应量和不确定性，不以结果选择参数。

生物学依据：成人 APL 是每侧一个广泛支配蘑菇体的 GABAergic、非尖峰神经元；实验显示 APL 内活动和对 KC 的抑制会随空间衰减，且不同蘑菇体区室可接受不同抑制。结合突触位置和 APL skeleton 的研究预测单个 KC 经 APL 对自身的抑制平均强于对其他任一 KC。APL 也被报告具有降低/归一化 KC 气味响应变异的作用。

关键限制：当前仓库的 `graph.npz` 只有 neuron-pair 级接触总数，没有突触坐标或 APL neurite geodesic。故 E1 的“局部”臂是 **KC 亚型区室约束的局部极限代理**，不是重建的真实空间电缆 APL。官方 MaleCNS 提供 6.8 GB synapse-partner、12.7 GB synapse-point 表和 skeleton；这些不纳入最小 E1，也不在本阶段下载。

## 2. MaleCNS APL↔KC 静态审查

M1-E1 沿用右半球 145 PN、1,907 个 PN 可驱动 KC 的主集合。右侧 APL body ID 为 `10540`。

- 右侧共有 2,045 KC；APL→KC 与 KC→APL 各覆盖 2,044 个 KC。唯一双向缺失为 `KCg-d` body `104513`，它不在 1,907 主集合内。因此主集合的 1,907 KC 全部与 APL 双向连接。
- APL→KC：2,044 条负号边、95,175 contacts；contact 中位数 48，IQR 33–60，范围 6–206，CV 0.350。
- KC→APL：2,044 条正号边、107,862 contacts；中位数 53，IQR 42–62，范围 9–218，CV 0.258。
- 同一 KC 的两个方向 contact 数 Pearson 相关约 0.670，说明双向强度相关但并不相同。
- 左侧作为静态交叉检查：2,019 KC 全部双向连接；APL→KC 99,853 contacts，KC→APL 101,119 contacts。最小 E1 不运行左侧复制。

这些事实支持“广泛、双向且强度异质”的反馈结构；pair-level 总数本身不证明空间局部性。

## 3. 三个固定模型臂

所有臂保留 PN→KC 图、KC 时间常数、输入尺度、KC 阈值、active classifier、刺激时间线和初始状态。非 APL 参数与 M1-RQ 相同：`tau_k=10 ms`、`input_scale=1`、`threshold=0.35`、描述性 active threshold `0.05`。APL-on 臂均保留 `tau_apl=50 ms` 和 `apl_gain=1`；不做增益搜索。

### G：原全局 APL

一个标量状态：

`tau_apl da/dt = -a + mean_k(r_k)`

每个 KC 接受同一抑制项 `-apl_gain*a`。该臂完全复现 RQ 的 APL 定义，不使用 APL↔KC contact 异质性。

### L：连接组/亚型约束的局部 APL 极限代理

把 1,907 KC 按 annotation 固定映射为 `gamma`、`alpha-beta`、`alpha-prime-beta-prime` 和 `untyped` 四组；最后一组只有一个未细分 KC，保留在动力学中但不作组间生物解释。

每组 `c` 有独立 APL 状态：

`tau_apl da_c/dt = -a_c + sum(k in c, p_k*r_k)/sum(k in c, p_k)`

其中 `p_k` 是 MaleCNS 的 KC_k→APL contact 数。KC_k 接收：

`-apl_gain*h_k*a_c(k)`

其中 `h_k = contacts(APL→KC_k) / mean_contacts_within_c`，所以每组平均输出倍率为 1，不引入新的总增益。组间传播固定为零，表示“完全区室化”的探索性极限；这是工程假设，不是已验证的 APL cable constant。所有 1,907 KC 均保留，不删除边或细胞。

### O：APL-off

保持相同 KC/PN 动力学和代码路径，但 APL 抑制项为零。它是去抑制参考，不代表生物消融的全部效应。

## 4. 新合成刺激与时间线

- 新公开 seed label：`STONKFLY-V2/M1-E1/STIMULUS/v1`；
- SHA-256：`5592f3331c82b75ca4551c3839be27174c6c650983b9ca697cd013780eaa79a8`；
- little-endian uint64 seed：`6680951629869716053`；
- 16 个独立合成气味，每个激活 29/145 PN；每个气味 10 次独立 jitter repeat；不读取、复制或比较 RQ/RT 刺激内容；
- 0–500 ms baseline，500–1,000 ms pulse，1,000–2,000 ms recovery；所有 trial 从零状态开始；
- dt 固定为 0.1、0.05、0.025 ms；
- 每个 dt/模型另有 10 个 ZERO repeats，不计入 160 个气味 trial。

本方案不为气味挑选“能显示局部效应”的 PN 子集，不根据 KC 亚型反向设计刺激。

## 5. 分析指标（描述性，不设旧 gate）

### 连续活动（主分析）

- 每个 KC 在 pulse window 的 mean rate、分位数、AUC；
- `O-G`、`O-L`、`L-G` 的 paired KC/odor/trial 连续差值；
- 组内和组间的抑制量、APL 状态峰值、峰时和 AUC；
- odor-level 均值/中位效应及固定 seed 的 odor-resampling 95% bootstrap interval。

### 气味特异性

- 连续响应向量的 within-odor cosine/correlation；
- between-odor 相似度及 within-minus-between margin；
- 三臂响应几何的 representational-similarity matrix 差异。

### 稀疏度

- threshold-free population sparseness/participation ratio为主；
- 沿用 `0.05` classifier 的 1,907 与 2,045 分母 active fraction 仅作描述，不使用 RQ 2%/10%/20% 门槛。

### 恢复

- offset 后 0–1,000 ms 的 population/compartment rate trace；
- recovery AUC、半衰减时间、1500/2000 ms residual 和 final-state norm；
- 保存 10 ms bin 的群体与区室 trace，不保存每一步完整 1,907-KC trace。

### 跨 dt

- 对每个 model×odor×trial 报告连续 rate NRMSE、cosine、active-fraction difference；
- 报告最大值和分位数，不套用 RQ 的 I8 pass threshold；
- 三个 dt 出现方向不一致时如实标记 numerical sensitivity，不选择有利 dt。

## 6. 尝试记录与产物

只需轻量机制：

- append-only `E1_ATTEMPTS.jsonl`：每次 toy、benchmark 或正式探索尝试记录时间、目的、命令、代码/config/seed hashes、退出状态和输出目录；
- 每个目录不可覆盖；失败目录原样保留；
- 产物至少包括 config、inventory、raw summaries、10 ms traces、statistics、report、manifest、stdout 和资源用量；
- 原 M1-RQ 与所有旧哈希保持不变；代码不得打开 M1-RT 路径。

正式 E1 最小矩阵只运行一次。toy 测试不得使用正式 seed 生成的刺激。

## 7. 资源估算

固定矩阵为 `3 models × 3 dt × 160 trials = 1,440` stimulated paths，另有 90 ZERO paths，共 1,530。RQ 的 19,740 paths 实测约 6.29 CPU-hours；按路径线性换算约 0.49 CPU-hours。局部状态、10 ms trace 和报告开销后预计：

- CPU：0.55–0.8 hours；
- wall：0.6–1.0 hours，单进程；
- peak RAM：约 1.0 GiB，预留上限 2 GiB；
- storage：预计 100–250 MiB，上限 1 GiB。

本阶段预算上限：3 CPU-hours、4 wall-hours、2 GiB RAM、1 GiB 产物。不使用 GPU。若未来获取 synapse-position/skeleton 数据，应单列 E1b 预算，不并入本次。

## 8. 明确停止条件

以下任一情况立即停止，不自动修参数或补跑：

1. 源图/hash、APL/KC 数量、1,907 主集合或 APL 双向覆盖与本方案不一致；
2. L 臂接触数映射不完整、组归属不唯一、归一化非有限，或三臂除 APL 结构外存在差异；
3. toy 独立 oracle、ZERO、缓存/无缓存或 dt 时间线测试失败；
4. 任何正式状态非有限、产物不完整、已有目录将被覆盖，或资源超过上限；
5. 固定 1,530 paths 完成后停止，无论效应大、小或为零；
6. 若 L 与 G 几乎相同，报告“在此局部极限与刺激下未见差异”；若差异明显，报告效应和限制；两者都不触发新的参数尝试；
7. 若解释必须依赖真实突触间 geodesic、APL cable constant 或 receptor kinetics，则停止于“信息缺失”，提交 E1b 数据方案，不能用结果反推这些参数；
8. 不打开原 RT，不进入 M2，不接入交易。

## 9. 实施前最小检查清单

实施需另行明确批准。批准后只需：实现三臂共同 runner、contact/亚型 mapping 单元测试、toy 局部反馈 oracle、无正式刺激 benchmark、一次正式 E1。无需重建四域密封、独立 OS 用户或签名授权体系。

## 参考

- Amin et al., *Localized inhibition in the Drosophila mushroom body*, eLife 2020, doi:10.7554/eLife.56954。
- Liu & Davis, *The GABAergic anterior paired lateral neuron suppresses and is suppressed by olfactory learning*, Nature Neuroscience 2009。
- Prisco et al., *The anterior paired lateral neuron normalizes odour-evoked activity in the Drosophila mushroom body calyx*, eLife 2021, doi:10.7554/eLife.74172。
- HHMI Janelia MaleCNS v1.0 official download/documentation: `https://male-cns.janelia.org/download/`。
