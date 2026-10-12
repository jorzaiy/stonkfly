# Stonkfly v2 M1-RQ freeze-v5 最终预检

日期：2026-10-11 UTC
结论：**GO，仅限一次 M1-RQ**。M1-RT、M2、市场与交易路径仍为 NO-GO。

## 冻结身份

- provenance bundle SHA-256：`df891a3f906c2aa2bd16263cc5471b2740c7ec73a56988e7972f0c7a28c6ab8a`；31 个结果相关文件在只读运行快照内逐一复核通过。
- run ID：`m1-rq-freeze-v5-20261011T133740Z`。
- 授权签名精确绑定 bundle、RQ manifest、run ID 和唯一输出路径；授权 payload 仅含 `M1_RQ`。
- RQ 进程运行在专用 OS 用户 `stonkfly-rq` 下。它的只读运行快照只包含 RQ 密封载荷；RT/M2 custody 路径不存在，签名私钥也未复制。
- 进程以 24 CPU-hours、24 wall-hours、4 GiB address-space/RSS 与 5 GiB 总产物为硬上限；预算在不可变失败记录中跨恢复累计。

## 完整性和容量

- 全部 40 个网络（B + 固定 39 C）由隔离进程重建，网络哈希与 v5 topology manifest 全部一致；所有 C 保持 PN/KC 度序列。
- RQ 内容仅在授权后的最终预检中解封；`16 × 10 × 145` 形状及内容哈希匹配。预检未运行神经动力学，也未查看神经输出。
- v5 容量基准使用全部 40 个代表性全尺寸拓扑、全部三个 dt 和解析性非正式输入。外推分母明确为每个 dt 6,560 个正式刺激 path-trials 加 20 个 ZERO path-trials。
- 实测基准为 18.10 CPU-seconds / 18.10 wall-seconds；带 1.25 裕量的 RQ 估计为 8.27 CPU-hours / 8.27 wall-hours；基准峰值约 581 MiB。该估算覆盖 39 C，不沿用旧的 0.21 秒 smoke 外推。

## 测试与失败语义

- 26 项 toy、静态和代表性真实拓扑测试通过。
- 缓存实现与独立无缓存逐步投影在真实 145 PN × 1,907 KC 拓扑上逐元素一致。
- S2 使用 v5 固定 PCG64 uint8 位流；I8 逐 path × odor × trial 取保守极值；C 的有限但未定义统计量按 `+∞` 保留。
- 结果目录、chunk、sidecar、raw、formal、report、stdout、manifest 与 COMPLETE 均不可覆盖。异常写入新的 append-only failure attempt；恢复必须匹配同一 bundle/run 身份和 chunk 哈希。
- verdict 优先级保持：实现失败、预算耗尽、数值 NO-GO、表征 NO-GO、`M1_RQ_QUALIFIED`。后者不是最终 M1 成功，不能自动进入 RT。

## 启动记录

正式进程已在 tmux 会话 `stonkfly-m1-rq-v5` 中启动。首次 tmux 命令因专用用户无登录 shell 而在 runner 启动前退出，未创建 run 目录或实验产物；随后用显式 `/bin/sh` 启动同一签名 run。该环境启动事件不改变实验身份、预算、刺激或统计规则。
