# R0c v2 implementation amendment: 2 s gate versus 3 s observation

状态：冻结的实现勘误；不改变 R0c 预注册、刺激、seed、dt、kernel、判读规则或正式数据集合。

R0c v2 在 0.1 ms R0b 复现门禁的 40 条记录中，只有三条失败。复核确认失败记录的前 2000 ms spike/active bins、`T_ign` 前 2 秒信息、state boundary、直方图和 weight hashes 均与 R0b 一致；差异来自 R0c 观察完整 3000 ms 后记录了黑屏期越过 2000 ms 的首次 KC 点燃时间，而 R0b 的 `T_ign` 只定义在 2000 ms 观察窗口内。

本勘误仅在记录中显式区分：

- `T_ign`：完整 0–3000 ms 观察窗的首次点燃时间，供 R0c 自身指标使用；
- `T_ign_2s`：当且仅当 `T_ign < 2000 ms` 时保留，否则为 null，专供 R0b 2 秒复现门禁；
- `ignited_by_2s`：继续按 `T_ign_2s != null` 定义。

v2 failure artifact 保留；v3 必须从全新目录按原 126-trial 顺序重跑。不得修改冻结实验设计或把越过 2 秒的黑屏期点燃改写为 2 秒内点燃。

勘误写入时间：2026-10-08 UTC
