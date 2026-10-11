# Fly 2024–2025 heldout paper result

本记录对应冻结的 2024-01-02 至 2025-12-29 信号窗口，结算日为
2025-12-30。实验仅限 paper/research；没有实盘、shadow authorization 或
收益驱动的配置选择。

## 输入锁与数据

- Input lock：[input_lock.json](../../runs/fly_purged_heldout_2024_2025/input_lock.json)，SHA256 `bec9afc806b96ad2c1e70253f0da51acb019ed593f7e104dde79e142c45733d6`
- Candidate：[candidate manifest](../../runs/wc006_v3_2024_2025_inputs/20260927T-purged-candidate-stream/manifest.json)，SHA256 `9ebd315f77d129609aa5467510c8e5cf63187a4ee5418c577e518173cc0eaa81`
- Market v3：[market manifest](../../runs/fly_purged_xysz_2024_2025/20260927T-supplemented-market-v3/manifest.json)，SHA256 `d90a3b7573fb06ee84c9f91fef25ded36025bbf8df5beb2e5a00fc109bcb028f`
- Market QA：[quality report](../../runs/fly_purged_xysz_2024_2025/20260927T-supplemented-market-v3/quality_report.json)，SHA256 `d0759227c61fc1e754f827fff0bc97769fa8002e45b50482d33cdf0dc864bbfa`
- 预注册协议：[fly_2024_2025_heldout_protocol.md](fly_2024_2025_heldout_protocol.md)

## 结果

483 个交易日完成，Fly 与独立 Base 账本的事件 hash chain 均为 483/483
通过；两本账本 `UNKNOWN=0`、`PENDING=0`。

| 指标 | Fly | Base |
| --- | ---: | ---: |
| 最终权益 | 113965.770098 | 98169.6203585 |
| 净收益 | +13.965770098% | -1.8303796415% |
| 最大回撤 | 47.7090102201% | 52.2123135220% |
| 费用 | 4281.635402 | 3347.5996415 |

Fly 减 Base 为 `15.7961497395` 个百分点，Fly 的预设非负门槛通过。

年度含结算收益：

- 2024：Fly `-0.0701387405%`，Base `-0.3348698275%`
- 2025：Fly `+14.0457603579%`，Base `-1.5005346518%`

2021–2023 的既有结果（Fly `-18.932%`、Base `-7.802%`）仅作背景，来自不同阶段的历史 paper 记录。

完整结果：[heldout_performance.json](../../runs/fly_purged_heldout_2024_2025/20260927T-paper/heldout_performance.json)；运行报告：[heldout_report.json](../../runs/fly_purged_heldout_2024_2025/20260927T-paper/heldout_report.json)。

本结果是一次冻结输入上的 paper 对账，不能证明稳定盈利、可交易性或神经学习有效性；模型中的损益强化仍是工程化输入。

## 结果解释与后续核查

2024–2025 的正收益不能解释为 Fly 在 2021–2023 亏损后学会了交易：
两个窗口均采用干净初始化和冻结可塑性，2024–2025 也没有接续前一窗口的
账户或脑状态。两段的市场时期、候选篮子和长度不同，不能把收益变化当作
学习效果。新窗口的年度结果显示，转正主要发生在 2025 年；2024 年的
Fly 收益仍略低于零。

同窗口 Base 为负而 Fly 为正，说明固定 Fly 动作形成的持仓与交易路径
在这一次回放中优于固定倍率 Base；目前尚未完成逐笔归因或跨窗口稳健性
检验，不能据此断定优势来自稳定择时能力。47.71% 的最大回撤也说明
非负终值没有解决中途大幅亏损的问题。

下一轮工作先在已锁定的账本上做只读归因：按月份、候选篮子、持仓、
暴露变化、费用和少数极端交易拆分 Fly–Base 差异，并检查收益是否集中
在少数日期或标的。记录完整亏损路径及回撤恢复过程。之后先写协议，
再用未用于调参的新窗口做前瞻 paper 检验；不得根据本窗口收益挑选
参数、种子或替代交易。此次文档更新不启动新回放，也不授权实盘。
