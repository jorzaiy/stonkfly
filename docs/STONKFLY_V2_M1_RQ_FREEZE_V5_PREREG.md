# Stonkfly v2 M1-RQ freeze-v5 预注册

状态：**仅 M1-RQ 获准一次执行**。M1-RT、M2 和全部市场/交易路径继续禁止。

本文件接受并冻结 `STONKFLY_V2_M1_FREEZE_V5_AMENDMENT_DRAFT.md` 的 S2、I8、ZERO、C 对照失败语义和 `M1_RQ_QUALIFIED` 定义；文件名中的 `DRAFT` 是保留历史轨迹，不表示本次 RQ 仍待批准。模型参数、刺激规则、40 个拓扑及科学门槛仍由既有 frozen config 与 v5 amendment 共同定义。

执行单位是 B 与固定 39 个 C 拓扑、16 odor × 10 trial、三个 dt；B 另有 APL-off，同一 B 的 ZERO 在每个 dt、APL on/off 各 10 次。正式刺激总计每 dt 6560 path-trials，ZERO 每 dt 20 次。

判定优先级固定为：实现/非有限/完整性失败，预算耗尽，I8 数值失败，B 的 S1–S4 表征失败，最后才是 `M1_RQ_QUALIFIED`。S5 只产生 RQ 内拓扑子状态，不得单凭 B 可学习或 S5 通过宣称真实果蝇拓扑具有学习优势。C 有限但科学统计量未定义时保留该 C 并赋 `+∞`；任何 C 非有限或度不变量失败属于实现失败。

RQ 运行不能读取 RT/M2 custody 路径。成功只表示资格集通过，不能称为 M1 最终成功，也不能自动进入 RT。失败后不改参数、seed、刺激、统计或门槛，不追加实验。
