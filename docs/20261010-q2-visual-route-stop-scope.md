# Q2 与 whole-brain visual route：正式停止与后续范围

状态：**2026-10-10 正式范围决策。Q2 candidate C 与当前视觉感觉入口停止；不再进入 B2/C、synthetic calibration、plasticity、S3 或市场训练。后续嗅觉 PN proxy 必须作为独立 O 系列重新立项。**

## 1. 被关闭的路线

关闭范围是：

`视觉/RGB proxy → photoreceptor/lamina → whole-brain propagation → KC/MBON → plasticity/decoder`

这不删除 MaleCNS v1.0 图，不否定实验平台的软件价值，也不声称真实果蝇视觉系统失效。停止结论只适用于本项目冻结图、当前 LIF/电流型突触模型及已尝试的三种传递语义。

## 2. 正式证据链

1. R0b/spiking 语义出现持续吸收上态；
2. Q1 单侧 graded release 的 C0 v3 得到 `C0_NO_QUALIFYING_GAIN`，KC 始终沉寂；
3. Q2 candidate C 的 v1 因 A 门证据不足降级为 `Q2_IMPLEMENTATION_FAILURE`；
4. 窄范围 v2 protocol-repair 补齐 A1–A7、independent dense reference、复模态/non-normal Arnoldi fixtures、Wgg/Hessenberg locks 后，正式得到 `Q2_LINEAR_INSTABILITY`。

Q2 v2 固定产物：

| 产物 | SHA256 |
|---|---|
| repair amendment | `fa9d567eddc8a9ecf47b0d5f378ed5a584b0f73f3a2fe7259a75b16968b059a6` |
| raw | `d3ddc82076d367d56115802dfd884319279708e4148f85d498fb18b9189b5f45` |
| formal | `c97ebade9075545ad3ee97d7992c2478687a4516d19904a695206925c0bab971` |
| report | `441bb33e42203bf5edb96e77078cd741c490a8c474332bb849edf37fadb9a9f9` |
| manifest | `10713f62015ca3809a64834c279e54595ff6cbfa14832b883c71c6b6bb5fb654` |

正式 Q2 v2 事实：

- A1–A7 与 coverage 全通过；三个 dt、每项两次；
- Wgg 为 11,822 nodes、38,018 edges，所有 CSR locks 通过；
- 两个 160×160 Hessenberg 与 v1 raw 逐位一致；
- `sin` 起点 20 个 converged Ritz、17 个 unstable；
- Rademacher 起点 17 个 converged Ritz、14 个 unstable；
- 两起点最大 `Re(s)` 均约 `0.05679147 ms^-1`；
- B2/C 按 hard stop 未运行。

Opus 的事后描述性 SCC 复核把主模态定位为 lamina 的高增益负反馈振荡；它支持解释，但不用于选择删边或缩放参数。

## 3. 停止解释

在冻结 MaleCNS v1.0 图与当前 neuron/synapse model 下，视觉 graded 传递的绝对尺度不可独立识别：

- 单侧 release 截断负向 deviation，路线沉寂；
- 双向 release 接通 lamina 高增益反馈，线性失稳；
- 全 spiking 语义曾进入吸收上态。

项目内没有独立依据冻结一个新的 graded gain、damping、reversal potential、cell-type gain、APL multiplier、threshold 或 `tau_emit`。用“稳定”或 KC/收益结果选择这些参数属于结果后适配。

## 4. 永久禁止的 Q2 补救

- 不扫描或缩小 graded gain/span；
- 不删除、不屏蔽、不重标 lamina SCC/edge；
- 不按 cell type、edge sign 或 target 设置有利倍率；
- 不改变 threshold/rest/APL 以制造稳定；
- 不把旧 C0/Q2 结果混成新的 calibration；
- 不把 B1 线性失稳描述成真实果蝇生理结论；
- 不恢复视觉 sensory adapter、plasticity、S3 或市场训练。

## 5. 新路线边界

用户选择先尝试独立的 olfactory PN proxy，而不是立即更换 neuron/synapse model。新路线命名为 O 系列，不是 Q2/Q3 延续。

O 系列允许：

- 只读识别 ORN→antennal-lobe PN→KC 的原图结构；
- 把市场特征明确描述为工程化 sensory proxy；
- 使用预注册、整数个、可审计的 PN spike-event intent；
- 先验证单脉冲传播、退出、确定性、dt 与吸收上态，再考虑任何功能编码。

O 系列禁止：

- 直接刺激 KC/MBON；
- 用 KC、MBON、动作或收益选择 PN、输入强度或脉冲率；
- 把 proxy spike 称为真实气味转导；
- 在资格通过前启用 learning、reward、decoder 或交易；
- 若单个稀疏 PN event 已触发持续活动，不得靠降低原图权重或提高阈值补救；届时应停止 O 路线并重新评估 neuron/synapse model。

## 6. 当前阻塞与下一门

当前仍阻塞：sensory adapter 集成、plasticity、S3、paper-market training 和所有 real execution。

下一阶段为 `O0 olfactory-PN inventory`。只有 O0 能以固定、可复现规则识别 PN 输入集合并证明原图存在 PN→KC 通路，才允许另写 `O1 single-event qualification` 预注册。
