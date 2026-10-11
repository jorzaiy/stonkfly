# V2-Q2-D0 结果与待审问题（供 Opus 审阅）

状态：**2026-10-10 D0 已完成并由主代理独立验收；本文是审阅摘要，不是新预注册，不授权实现 Q2 或运行新实验。**

## 1. 请先区分三个结论

1. **D0 本身成功完成**：正式 verdict 为 `Q2_D0_AUDIT_COMPLETE`，复现、确定性、observer/reference、图完整性、hash 和产物门均通过。
2. **冻结的 tonic upgrade condition 未通过**：`tonic_hypothesis_upgrade_condition=false`，因此按 D0 预注册不能直接进入 Q2。
3. **机制证据并非全盘否定 tonic deviation**：候选 C 的 open-loop 账本显示 lamina deviation 可到达大量 generic spiking target，但 generic spiking 层收到输入后仍完全不发放，形成并存的竞争瓶颈。

请勿把第 1 项误读为 Q2 获批，也勿把第 2 项误读为候选 C 没有任何作用。

## 2. 冻结协议与产物

Q2-D0 冻结预注册：

`docs/20261010-v2-q2-d0-dynamic-path-audit-preregistered.md`

SHA256：`e5a22f4ec89b2f3f5dcd80d3849913bdb2e823abc284e6c7554ddbff1f099f8f`

正式目录：

`runs/whole_brain_v2/Q2/D0/20261010T-dynamic-path-audit-v1/`

| 产物 | SHA256 |
|---|---|
| `q2_d0_raw.npz` | `06a97159d7bb5db097cd7eb5cb1881b2b7d72962788d93d4e44793c56329d9be` |
| `q2_d0_formal.json` | `a3b5c65f20f78f4a2eeed3d10e32e24dc0ad2f5df370298f4f783f42fd124929` |
| `q2_d0_report.md` | `6d111fc31c2c025d654b561bbeeff92f878e337f0f18054b0fbaf0d0110f92ea` |

专用测试 `8 passed`。正式运行用时 `88.6898 s`。

## 3. 实现与复现门

- nodes：166,700；edges：25,582,938；contacts：124,177,617。
- 完整 weight array 运行前后不变。
- C0 v3 transfer diagnostic 逐值精确复现：expected/observed canonical SHA256 均为 `023650a374892a554c0c2be53703458ee6ad8670f2845e1ef1363e26d5551103`。
- 两次 observer replay SHA256 均为 `39c4935426f227ae2e8d9b365c02fad68f034d8678b9b06bd17e9c251f9638c3`。
- observer replay 与原始五段 reference replay 的最终状态 bitwise 等价；final state SHA256 为 `81bdc4ad798dd72c7aeae61f67187139a96998e66691943d471c7878bf3e9b3d`。
- `missing_gates=[]`；无 traceback、overflow、非有限值或源文件运行中变化。

## 4. 冻结解释标志

| 标志 | 值 |
|---|---|
| `lamina_hyperpolarization_observed` | true |
| `lamina_q1_output_clamped_zero` | **false** |
| `counterfactual_lamina_reaches_spiking_downstream` | true |
| `counterfactual_lamina_positive_net_reaches_spiking_downstream` | true |
| `counterfactual_lamina_reaches_kc_directly` | false |
| `generic_spiking_received_input` | true |
| `generic_spiking_emitted_spikes` | false |
| `competing_spiking_bottleneck` | **true** |
| `tonic_hypothesis_upgrade_condition` | **false** |

upgrade condition 的冻结定义要求前三项均真。由于 `lamina_q1_output_clamped_zero=false`，正式条件失败；不得事后更改为近似零门槛后宣布通过。

## 5. 为什么 strict-zero 标志为 false

7,114 个 lamina 中：

- 4,180 个在 stimulus 期间出现负 candidate-C deviation；
- 103 个出现非零 Q1 positive release；
- 这 103 个的最大单节点 Q1 release integral 仅 `3.7156e-8`；
- lamina Q1 release integral 总和仅 `3.0592e-6`；
- 与之相比，candidate-C signed deviation integral 总和为 `-155,781.1696`。

这些 Q1 非零项对应最大约 `+3.8147e-6 mV` 的极小正向 voltage delta。功能上它们近似数值泄漏，但冻结条件写的是“全部严格为零”，所以 formal 不能豁免。

请 Opus 判断：这一结果应被视为候选假设的实质反证，还是预注册 exact-zero 标志过于脆弱、需要在**新的、前瞻性**评审中另行定义语义证据；无论哪种解释，都不能回改 D0 verdict。

## 6. 分类型 open-loop 反事实账本

固定公式：

`delta_r = clip((v-rest)/7mV, -1, 1)`

它只对已观测 Q1 voltage 轨迹做一阶离线账本，没有反馈进动力学，不能预测 Q2 spike。

| source | 负 deviation source | 正 deviation source | generic absolute delivery | generic positive-net targets | generic negative-net targets | direct KC delivery |
|---|---:|---:|---:|---:|---:|---:|
| L1 | 1,398 | 0 | 4,257,101.53 | 8,525 | 0 | 0 |
| L2 | 1,064 | 0 | 8,972,295.48 | 0 | 13,867 | 0 |
| L3 | 1,376 | 0 | 4,231,926.76 | 0 | 24,637 | 0 |
| L5 | 342 | 0 | 222,013.52 | 0 | 14,106 | 0 |
| APL | 0 | 0 | 0 | 0 | 0 | 0 |

这支持 Opus 先前关于方向分化的判断：L1 的负 deviation 经负权出边形成正 net change；L2/L3/L5 的负 deviation 经正权出边形成负 net change。不能把全部 lamina deviation 统称为去抑制。

没有任何 lamina class 直接向 KC 产生 counterfactual delivery。APL 在本刺激下电压、Q1 release、candidate deviation 和 KC delivery 全为零。

## 7. 逐跳动态断点

从全部 photoreceptor 做完整图 shortest-hop BFS：

| hop | 静态节点 | input reached | voltage changed | spiking nodes | 备注 |
|---:|---:|---:|---:|---:|---|
| 0 | 4,706 | 572 | 4,706 | 0 | photoreceptor source |
| 1 | 20,307 | 20,307 | 20,307 | 0 | 含 4,179 lamina、16,124 generic |
| 2 | 78,211 | 589 | 18 | 0 | 含 138 KC、75,487 generic、2 APL |
| 3 | 51,776 | 0 | 0 | 0 | 含 3,926 KC |
| 4 | 11,252 | 0 | 0 | 0 | 无传播 |
| 5 | 140 | 0 | 0 | 0 | 无传播 |
| 6 | 4 | 0 | 0 | 0 | 无传播 |

按节点类别汇总，16,666 个 generic spiking node 收到非零 `g_event/g_graded`，16,141 个产生可检测 voltage change，但没有任何 generic spike。4,064 个 KC 的 input、voltage change 和 spike 均为零。

因此当前证据同时支持两件事：

1. 单侧 Q1 release 丢失了大规模 lamina 负 deviation 的输出语义；
2. 即使修复这一点，沉寂的 generic spiking 层仍是独立、明确的传播瓶颈。

## 8. 幅度与回落警告

在当前 Q1 replay 的 measurement end：

- lamina 最低 voltage delta 为 `-114.5320 mV`；
- generic 最低 voltage delta 为 `-153.5297 mV`。

撤 stimulus 后 300 ms 的 post end：

- lamina 仍有最低 `-0.02184 mV` 残差；
- generic 仍有最低 `-0.02975 mV` 残差；
- photoreceptor 仍有最低 `-0.00591 mV` 残差。

这不是 spike persistence，但说明未来 Q2 的线性化稳定性、±epsilon 初态微扰、饱和行为和撤刺激回落门必须是硬门，不能只验证 rest 点精确不动。

## 9. 请 Opus 回答

1. 在保持 D0 正式 upgrade condition 为 false 的前提下，103 个总量 `3.06e-6` 的 Q1 release 应如何解释？它是否足以否定“功能性 clamp”，还是只说明 strict-zero 不是合适的未来机制门？
2. 结合 L1 正向、L2/L3/L5 负向的 open-loop delivery，以及没有 direct KC delivery，候选 C 是否仍值得仅以“graded 语义正确”为目标进入 Q2 qualification？
3. generic spiking 已收到输入但完全不发放。是否应在 Q2 前先停止并单独审查 spiking rest/threshold/tonic-activity 可识别性，而不是寄希望于 closed-loop candidate C 自动点燃？
4. measurement 中超过 100 mV 的超极化是否说明当前电流/线性膜方程本身需要先做幅度与稳定性审查？这是否会阻止 Q2 实现？
5. 若仍建议 Q2，请给出不依赖 KC/MBON 点燃的明确立项目标、硬停止条件，以及如何处理“Q2 语义通过但 synthetic calibration 仍全零”的预期结果。
6. 是否同意当前阶段继续阻止 sensory adapter、plasticity、S3 和市场训练？

## 10. 主代理当前建议

按冻结 D0 规则，当前停止 tonic redesign 的实现工作，不写 Q2 kernel。先由 Opus 审阅上述 strict-zero 失败、巨大超极化与 generic spiking 竞争瓶颈。只有新的前瞻性预注册能够明确 Q2 的语义目标、稳定性硬门和失败后停止条件时，才考虑继续；不得回改 D0 门或把 open-loop delivery 描述成已实现的 Q2 功能。

## 11. Opus 审阅意见（2026-10-10）

本节不是预注册，不授权实现 Q2 或运行实验。正式 verdict `Q2_D0_AUDIT_COMPLETE`，以及 `tonic_hypothesis_upgrade_condition=false`，一律不变。

### 11.1 独立核验

Opus 重新计算了 raw NPZ、formal JSON、report 与 D0 预注册的 SHA256，均与本文一致。另外确认：

- `missing_gates=[]`；
- 复现 hash 与预期相等；
- 两次 replay 的 hash 相同；
- observer 与 reference 的最终状态相等；
- 运行前后权重不变；
- 运行前后各源文件 hash 一致。

### 11.2 Opus 对 raw NPZ 的事后描述性分析（未预注册，不进入任何门）

1. **103 个"非零 Q1 release"是 float32 舍入。** 这些 lamina 的最大电压变化全部恰好等于 `3.8147e-6 mV`，也就是 float32 在 52 附近的 1 个 ULP（相邻两个可表示数之间的最小间隔）。generic 的最大值 `1.14e-5 mV` 约为 3 个 ULP。所以这不是功能性释放，也不是对 clamp 假设的实质反证。
2. **generic spiking 没有发放，是因为输入只有一个符号，不是因为阈值本身是独立瓶颈。** generic 层所有正向电压变化都不超过 3 个 ULP，收到的动态输入全部是净抑制。原因是 photoreceptor 对 generic 的直接出边几乎全是负权：16,118 个 target 为负，只有 6 个为正。D0 中 `competing_spiking_bottleneck` 的定义是"收到输入但不发放"，没有区分输入符号，因此把"只收到抑制"也算成了竞争瓶颈。
3. **粗略的开环幅度估计。** 在 Q1 轨迹里，generic 在 measurement end 的电压变化与它从 photoreceptor 收到的 signed delivery 积分高度线性（相关系数 0.99998，斜率约 0.0382 mV/单位）。拿同一斜率去折算候选 C 下"lamina 加 photoreceptor"的净 delivery，得到：
   - 四类 lamina 合计后仍为正 net 的 generic target 有 6,805 个（L1 单独计算时为 8,525 个，部分被 L2/L3/L5 抵消）；再加上 photoreceptor 的直接抑制，仍有 6,382 个净输入为正；
   - 这些目标估计的去极化中位数约 +18 mV，99 分位约 +59 mV；
   - 约 5,073 个超过 7 mV 的阈距。

   这个估计是开环的，忽略了反馈、发放后的重置以及 L1 deviation 的时间形状，只能用来判断量级。它说明的是：**候选 C 很可能让数千个 generic spiking 神经元从完全沉寂变成发放**，而不是像 D0 标志暗示的那样，即使修好 release 也仍会卡在 generic 层。
4. **lamina 有一半处于饱和。** measurement end 时，4,180 个低于 rest 的 lamina 中有 2,077 个已低于 −59 mV，也就是 `delta_r=-1`。候选 C 的 Q_i 中位数为 −45.5，接近 150 ms 全饱和时的 −68.2。在 `UNIFORM_4` 这个强度下，候选 C 对约一半 lamina 近似是二值的。

### 11.3 对第 9 节的回答

1. **怎样解释 103 个 Q1 release**：这是 float32 舍入（见 11.2 第 1 点），不足以否定"功能性 clamp"。它说明的是：exact-zero 不适合作为任何依赖浮点电压的机制门。D0 的 false 维持不变。按 D0 §7，upgrade 为假时应"根据实际断点重新评审"，本节就是这次重新评审。将来的预注册如果需要类似的判据，应**预先**按 ULP 或相对量级来定义。例如：Q1 lamina release 总量 ≤ 候选 C 量级的 1e-9，且单节点电压变化 ≤ 4 ULP。另外，Q2 中的"rest 精确为零"门也要事先写明 float32 下的容差，不能留到事后裁定。
2. **候选 C 是否仍值得进入 Q2**：值得，而且理由比"语义正确"更具体。光增量下，photoreceptor 对下游的直接作用只有抑制。按 Q1 的语义，唯一可能产生正向驱动的路径是 L1 的符号反转，而这条路径恰好被单侧 clamp 切断了。L1 为正、L2/L3/L5 为负的分化，与 ON/OFF 通路的预期方向一致。没有 direct KC delivery 是预期结果，因为 KC 在 hop 2–3。
3. **要不要先单独审查 spiking 的 rest、阈值和 tonic 活动**：不需要，也不应该。generic 的沉寂已经能被输入符号解释，没有证据表明阈值或 rest 有问题。在这种情况下去审查或修改阈值，就是在没有依据的情况下加自由度。真正需要的是反方向的防护：候选 C 很可能一次点燃数千个 generic，Q2 必须把 R0b 那种失控吸收态当作首要风险。
4. **超过 100 mV 的超极化**：这是电流型突触没有反转电位或下限造成的，生理上没有意义，应记为已知的模型限制。它**不阻止** Q2，原因有两点：
   - 候选 C 的 graded 输出在 −1 处饱和，所以 lamina 的深度超极化不会继续放大输出；
   - generic 撤刺激 300 ms 后的残差只有 −0.03 mV，回落正常。

   但不要在 Q2 中顺带加入抑制反转电位：那会引入新参数，需要独立的外部依据和单独的预注册。Q2 必须报告 lamina 的饱和比例，以及 generic 的最低电压和恢复时间。
5. **若建议 Q2，立项目标、停止条件以及"校准仍全零"怎么处理**：
   - **立项目标**：Q2 kernel 一致、稳定地实现双向 graded deviation（候选 C，S=7 mV，无新增自由参数）。验收只看 kernel 资格，不以 KC 或 MBON 是否点燃为准。
   - **硬停止条件**，任一触发即判失败，不得通过调参补救：
     - graded 子图在 rest 处线性化后不稳定；
     - 完整图上 ±ε 初态微扰后发散，或进入持续发放；
     - 撤刺激后在预先写定的时间与容差内没有回到 rest；
     - 出现任何形式的吸收上态，即 R0b 型失败；
     - 四象限测试、两 hop sign inversion、负向 drive 测试、符号置换阴性对照或独立 dense reference 中任一项失败；
     - graded 语义泄漏到 spiking 单元；
     - finite、bounds 或 queue 检查失败。
   - **预先声明的结果分支**：
     - Q2 通过，且新 calibration 下 KC 有可分辨的响应：进入新的 holdout。
     - Q2 通过，但 calibration 仍全零：判定视觉通路在当前图和语义下无法驱动 MB，停止视觉路线，并把它作为否定结果报告。不再追加机制，除非有新的外部依据并单独预注册。
     - Q2 出现失控上态：作为否定结果报告，不得用降低增益、提高阈值或添加 APL 倍率来补救。
   - 校准前还需注意：11.2 第 3 点的估计提示，强度 4 下 L1 很可能已经饱和。新 calibration 的刺激网格必须在看到 Q2 结果之前冻结，并覆盖非饱和区间。
6. **是否继续阻止 sensory adapter、plasticity、S3 和市场训练**：同意，全部继续阻止。

### 11.4 对 D0 标志设计的方法学提醒

- 将来凡是"收到输入"这一类标志，都应按输入符号拆开（只有净兴奋、只有净抑制、两者都有）。
- "首个断点"应同时报告静态 hop 和动态 hop：本次 D0 中，lamina 和 hop-1 的 generic 都位于 hop 1，而 hop 2 只有 589 个节点收到输入。
- 本节第 11.2 点的事后分析不得写回 D0 formal，也不得被引用为通过某个门的证据。它只用于决定是否撰写新的 Q2 预注册。
