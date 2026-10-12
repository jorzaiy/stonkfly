# Stonkfly v2 M2-B：虚拟二选一气味选择探索方案

状态：**FORMAL RUN COMPLETE / `ENGINEERED_VIRTUAL_PREFERENCE_DIRECTION_UPDATE_NOT_OBSERVED`**
日期：2026-10-12 UTC

## 1. 边界与研究问题

M2-B 是新的探索性实验，不是 M2-E 的补跑或重判。M2-E 的
`ENGINEERED_ASSOCIATIVE_UPDATE_NOT_OBSERVED`、原门槛、原始产物和旧
M1-RT/M2 密封状态保持不变。

研究问题仅为：固定的 E1-G 感觉动力学、真实右侧 KC→MBON07/11 边和
M2-E 工程可塑性，能否在全新合成气味上产生可重复的虚拟二选一偏好，
并在奖励标签反转后改变选择方向？这里的“选择”只是两个固定 MBON
score 的数值比较，不模拟运动回路，也不解释为真实果蝇行为。

禁止市场、交易、执行模块及旧 M1-RT/M2 数据；不得读取或复用 M2-E 的
8 对气味来生成 M2-B 刺激。

## 2. 固定模型与唯一新增读出

- 感觉层、KC→MBON 拓扑、学习率、DAN 脉冲、trace、bounds、训练周期、
  washout 和 retention 均原样复用 M2-E；不因 M2-E 结果修改参数。
- 五臂固定为 `paired`、`frozen`、`shuffled`、`unpaired`、`dan_omitted`。
  感觉轨迹可只读共享；各臂的 trace、memory 和 efficacy 必须独立。
- 使用新的 seed 域 `STONKFLY-V2/M2-B/VIRTUAL-2AFC/v1`。SHA-256 域分离
  派生 pair、odor、calibration、probe、training、order 和 shuffled seeds；
  生成后不按神经结果筛选气味。

M2-E 在每个 probe 中用同一刺激的未学习反事实读出归一化，因此 P0 的
`V` 按构造精确为零。M2-B 为形成可执行的固定比较器，新增训练前
calibration：每个气味使用 5 个独立、无奖励 probe，固定两个通道基线
`y0_c(o)`。calibration 不更新权重，并与全部选择 probe 使用不同 seed。
此后所有阶段和所有臂都只能使用该冻结基线：

`V(o,r,t) = y_approach(o,r,t)/y0_approach(o) - y_avoidance(o,r,t)/y0_avoidance(o)`

对匹配的 A/B probe repeat：

`d(r,t) = V(A,r,t) - V(B,r,t)`

离散规则在运行前固定为：`d>0` 选择 A，`d<0` 选择 B，浮点值精确等于
零则记为 tie；不设可调 deadband，也不事后移动零阈值。每阶段同时保存
全部 `d`、两个通道、三个单 MBON 输出及选择。离散符号不得单独用于
“强偏好”表述；任何解释必须同时给出连续幅度、区间和原始点。

calibration-based normalization 是 M2-B 的工程代理，不是已知果蝇解码
机制，也不能被描述为动物记住了气味基线。

## 3. 最小矩阵

- 4 个独立气味对；每个气味仍固定激活 29/145 PN，幅度和 jitter 规则
  沿用 E1/M2-E，但 seed 与内容全新。
- 每对先做 5 repeats/odor calibration；随后在 P0、P1、P2、P3 各做
  10 个匹配 A/B choice repeats。
- acquisition：10 cycles，A+reward、B+aversive；P1 后 reversal 10 cycles，
  A+aversive、B+reward；P2 后 60 s 无气味、无 DAN retention，再做 P3。
- 五个对照臂沿用同一感觉输入和日程；shuffled 每阶段、每气味维持
  5 reward/5 aversive 的平衡排列。

每对唯一感觉路径为 `10 calibration + 40 training + 80 choice probes = 130`，
全实验共 520 paths。五臂共享完全等价的感觉计算，不共享学习状态。

## 4. 冻结报告规则

每个 pair、arm、stage 报告：

1. `d` 的 10 个原始值、均值、中位数、bootstrap 95% CI；
2. A/B/tie 数，以及 `PI=(n_A-n_B)/(n_A+n_B)`；tie 单列，不进入分母；
3. acquisition 变化 `D_P1-D_P0`、reversal 变化 `D_P2-D_P1`、P2 endpoint
   和 P3 retention；
4. paired 相对 frozen、shuffled、unpaired、DAN-omitted 的 pair-level 差中差；
5. 每个 MBON 通道、权重变化、bound fraction 和非有限检查。

4 个气味对是独立探索 replicate；同一 pair 内的 10 个 probe 只刻画刺激
jitter 下的测量稳定性，不冒充 40 个独立生物样本。由于 4 replicate 的
双侧 exact sign-flip p 最小只能到 0.125，本实验不以显著性通过/失败为
目标，不创造新的正式 GO gate。

预先冻结的描述性结论规则：只有 paired 在至少 3/4 pairs 上同时满足
P1 `median(d)>0`、P2 `median(d)<0`，且 P3 保持 P2 方向，才能称为
“工程化虚拟偏好方向更新”；还必须报告其连续幅度及相对每个对照的
效应。否则报告“未观察到稳定的虚拟偏好更新”。无论哪种结果，都不改变
M2-E NO-GO，也不支持真实果蝇行为或生物学学习结论。

实施正确性优先于上述描述：frozen 与 DAN-omitted 权重必须 bitwise 不变，
所有输出有限，无跨通路更新，五臂学习状态不共享，reward-free probe 不得
更新权重。任一失败则只报告实现失败，不解释选择结果。

## 5. 产物与停止条件

新 run namespace 必须拒绝覆盖，保存 config、seed/content hashes、每 pair
原始连续输出、选择记录、权重快照、统计、资源记录、报告和 manifest。
不需要四域密封、解密或新增复杂授权系统。

以下任一发生即停止并保留失败目录：实现检查失败、预算超限、非有限值、
缓存键不等价、旧 M1-RT/M2 或市场路径被访问。实验完整结束后停止；不得
根据结果增添气味、改变零阈值、增加 cycle 或修改学习率后续跑。

## 6. 资源与工作量

M2-E 的 640 paths 实测为 2,045.5 CPU s、2,046.0 wall s、峰值 RSS
679 MiB、产物 0.84 MiB。按相同感觉计算线性估算，M2-B 的 520 paths
约需 1,662 CPU s（0.46 CPU h）和约 28 min wall time。

探索运行硬预算冻结为：1 CPU-hour、1 wall-hour、1 GiB 实测 RSS、50 MiB
产物、不使用 GPU。另设 2 GiB `RLIMIT_AS` 内核后备限制；虚拟地址空间与
RSS 不是同一指标，不能用 1 GiB `RLIMIT_AS` 冒充 1 GiB RSS 限制。RSS 在
每条感觉路径后检查。该预算约为线性估算的 2.2 倍，并覆盖 choice 统计和
manifest 开销。实现、测试和一次 toy smoke 预计 4–6 工程小时；正式小矩阵
运行预计 30–45 分钟。

实施前只需完成：独立 calibration/choice seeds、固定 choice reader、五臂
状态隔离测试、toy 符号/反转/tie 测试、缓存等价和预算中止测试。无需新增
与本科学问题无关的审批或密封设施。

## 7. 完成结果

正式 run `m2-b-20261012T0740Z` 完成全部固定矩阵和实现检查。paired 臂 0/4
气味对满足 P1 正、P2 负且 P3 保持负向，故按冻结规则得到上述 NO-GO。
paired 相对 frozen 的 acquisition 差中差均值为 `+0.010574`（bootstrap 95%
CI `[+0.009185,+0.011596]`），reversal 差中差为 `-0.013755`（95% CI
`[-0.016831,-0.010678]`）。结果支持工程化配对差分更新，但不支持稳定的
零阈值虚拟偏好方向更新。完整汇总见 `STONKFLY_V2_EXPERIMENT_SUMMARY.md`。
