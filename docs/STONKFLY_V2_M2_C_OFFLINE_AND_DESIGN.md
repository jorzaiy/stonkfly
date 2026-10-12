# Stonkfly v2 M2-C：固定模型读出可靠性

状态：**FORMAL VALIDATION COMPLETE / `ENGINEERED_AGGREGATE_PREFERENCE_DIRECTION_UPDATE_NOT_OBSERVED`**
日期：2026-10-12 UTC

## 1. 边界

M2-C 不修改 E1-G 感觉动力学、KC→MBON 拓扑或权重、学习率、DAN 脉冲、
奖励时序和 M2-B 的零阈值。M2-B 原始
`ENGINEERED_VIRTUAL_PREFERENCE_DIRECTION_UPDATE_NOT_OBSERVED` 保持不变。
本文件第一部分只读取 M2-B 保存的 P0/P1/P2 probe、channel 和 efficacy
记录；没有重放刺激或推进塑性。

离线聚合次数预先限定为 `{1,2,5,10}`，只是固定的尺度比较，不据此选择
“最佳”次数或阈值。第二阶段的 10 次平均来自研究问题中预先提出的规则，
不是对 M2-B 结果寻优。

## 2. 离线结果

### 2.1 单次与聚合读出

对每个 pair/stage 的 10 个已保存 `d=V(A)-V(B)`，以经验分布有放回抽样
50,000 次。选择一致性定义为聚合结果与该 cell 全部 10 次均值方向相同；
这是内部稳定性而非独立泛化准确率。

| 聚合次数 | paired 中位方差 | paired 平均一致性 | paired 最差 cell | frozen 中位方差 | frozen 平均一致性 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.001627 | 70.8% | 49.8% | 0.001634 | 70.0% |
| 2 | 0.000821 | 72.6% | 54.0% | 0.000820 | 71.8% |
| 5 | 0.000328 | 79.6% | 53.7% | 0.000325 | 80.1% |
| 10 | 0.000163 | 85.2% | 56.3% | 0.000163 | 85.6% |

方差近似按 `1/k` 下降，但阈值方向一致性没有同步接近 100%。最差 cell
即使做 10 次平均也仍接近随机，因为其分布中心靠近零。

paired−frozen 的实际学习位移不会因平均而变大：P1 acquisition endpoint
四对均值为 `0.00856、0.01105、0.01185、0.01084`，总体均值 `0.01057`；
P2 reversal endpoint 为 `−0.00052、−0.00498、−0.00042、−0.00680`，总体
均值 `−0.00318`。相对于单次 frozen probe SD 的中位绝对效应为 P1
`0.239 SD`、P2 `0.0465 SD`；10 次平均的名义尺度分别提高到 `0.756 SD`
和 `0.147 SD`。聚合降低方差，但不改变实际效应量或固定偏置。

### 2.2 误差来源

| pair | 固定权重 pooled 偏置 | pooled within-stage SD | stage-mean SD | calibration raw A−B 结构差 |
|---:|---:|---:|---:|---:|
| 1 | −0.01909 | 0.02631 | 0.00951 | −0.00283 |
| 2 | −0.01823 | 0.05376 | 0.00485 | +0.00383 |
| 3 | +0.00868 | 0.04215 | 0.00997 | −0.00625 |
| 4 | +0.01597 | 0.04898 | 0.00290 | −0.00362 |

1. **独立刺激 jitter**：固定 efficacy 下，同一 stage 的 10 个独立 probe
   仍产生 `0.026–0.054` 的 pooled SD；这是主要的单次选择波动。模型本身
   是确定性的，不能把它称为内源神经噪声。
2. **calibration 分母失配**：5 次 calibration 的固定通道均值与随后 30 次
   frozen probe 的通道均值相差约 `−2.4%` 到 `+3.0%`。两个通道和两个气味
   的相对误差相减后，恰好产生上表 `−0.0191` 到 `+0.0160` 的 normalized
   `D` 偏置。旧 chunk 保存了 calibration 均值但没有保存5个单次 calibration
   输出，所以不能再把“calibration 样本方差”与“30次 probe 均值误差”
   唯一分开；只能量化二者的实现后失配。
3. **原始结构偏置**：calibration raw `approach−avoidance` 的 A−B 差异确实
   存在，但其符号在 pair 2–4 与 normalized pooled 偏置并不一致。逐气味、
   逐通道归一化消除了大部分 raw 结构尺度差；当前 normalized 固定偏置主要
   由有限 calibration/probe 样本失配决定，结构仍通过分母尺度影响灵敏度。

因此，多次观察能降低独立 jitter，却不能消除 calibration 固定偏置；当
P2 净学习效应只有约 `4e-4–5e-4` 时，10 次平均也不能保证跨零。

## 3. M2-C 独立验证设计

### 3.1 固定模型和刺激

- 新 seed 域 `STONKFLY-V2/M2-C/READOUT-RELIABILITY/v1`；4 个全新气味对，
  不读取或复用 M2-B 气味、旧 RT 或旧 M2。
- 完整复用 M2-B 的 E1-G、KC→MBON、plasticity、10-cycle acquisition、
  10-cycle reversal、60 s retention 和五臂对照。
- calibration 仍为 5 repeats/odor，不因离线结果加大；但必须保存5个单次
  channel/MBON 输出，以便独立估计 denominator variance。
- P0、P1、P2、P3 各固定 10 个独立 probe repeats/odor。

### 3.2 预先冻结的两个读出

对匹配 repeat 定义 `d_r=V(A,r)-V(B,r)`，阈值始终为零。

1. **主要聚合读出**：`D10 = mean(d_0...d_9)`；`D10>0` 选 A，`D10<0`
   选 B，精确零为 tie。
2. **单次对照**：预先指定 `r=0` 的符号为单次选择，同时报告全部10个
   单次选择的 A/B/tie、PI、方差和原始点。不得事后挑一个有利 repeat。

`k=2` 和 `k=5` 不作为新实验的主要或次要判定；离线比较不赋予其选择权。

### 3.3 报告和解释

每个 pair、arm、stage 同时报告：10个连续 `d`、`D10`、单次 `r=0`
选择、10次 PI、calibration 单次分布、通道/单 MBON 输出和权重摘要。
报告 acquisition change、reversal change、P2 endpoint、P3 retention，及
paired 相对 frozen、shuffled、unpaired、DAN-omitted 的配对差分。

描述性方向标签预先保持与 M2-B 对称：只有 paired 的 `D10` 在至少 3/4
pairs 上同时满足 P1 `>0`、P2 `<0` 且 P3 保持负向，才称为“观察到聚合
虚拟偏好方向更新”。实施检查失败优先。无论结果如何，必须报告连续幅度
和单次对照；该标签不是果蝇行为或生物学验证。

预期限制明确冻结：10次平均只降低独立 jitter，无法消除错误的 calibration
分母、固定初始偏置或过小的 reversal endpoint，也不保证跨零。

### 3.4 规模、预算和停止

路径数与 M2-B 相同：4 pairs × 130 = 520。沿用实测容量，预计约 27 分钟、
峰值 RSS 约 680 MiB。硬预算：1 CPU-hour、1 wall-hour、1 GiB RSS、2 GiB
虚拟地址空间后备、50 MiB 产物、无 GPU。

使用新目录、拒绝覆盖并保留失败产物。非有限值、状态共享、frozen 或
DAN-omitted 权重变化、缓存不等价、预算超限或访问旧 RT/M2/市场路径时
立即停止。只运行一次固定矩阵，不根据中途结果调整阈值、聚合次数、气味、
cycle、学习率或奖励协议；不接入市场、交易或执行。

## 4. 独立验证结果

正式 run `m2-c-20261012T100756Z` 完成全部固定矩阵和八项实现检查。10-probe
均值的方向更新计数为 0/4，未达到 3/4 描述性标准。paired 相对 frozen 的
acquisition 差中差为 `+0.013412`（bootstrap 95% CI
`[+0.011712,+0.016224]`），reversal 差中差为 `-0.014369`（95% CI
`[-0.015956,-0.012532]`）。聚合降低了独立刺激 jitter，但没有消除固定偏置
或保证跨零反转；M2-B 与本实验的 NO-GO 均保持不变。
