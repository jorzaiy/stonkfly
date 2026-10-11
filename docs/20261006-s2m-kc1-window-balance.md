# S2m 预注册：KC1-relative input-window balance

本诊断为 observational、full retained graph；`counterfactual=false`、`learning=false`、`production_eligible=false`。固定 S2d 同一 60 事件、fresh observer-build 与历史 checkpoint 的数组级等价初态，不改图、权重、阈值、动力学、解码、标签、收益或交易。

source class 沿用 S2l 的固定优先级：KC、APL、DPM、`PN/visual_projection`、other_cb、other。`PN/visual_projection` 只是历史 class label；权威集合是 visual_projection，已知与 annotation PN 交集为 0，不称其为 PN 输入。

KC1 必须在线由第一个 unique KC spike emission 冻结；KC1 未出现即 fail closed。只统计真实 kernel 成功执行 `g[j]+=weight[e]` 的 KC delivery，负 delta 不纳入 S2m。相对窗口固定为互斥且穷尽的：W0 `[0,64)`、W1 `[64,128)`、W2 `[128,256)`、W3 `[256,512)`、W4 `[512,1024)`，delta=`delivery_step-kc1_step`。

每帧×class×window 记录 events、signed/positive/negative、unique source、unique KC target、first/last delta，并以独立 all-KC accepted counter 守恒校验六类总和。target_seen 为 KC×class×window，source_seen 为 source×window；sentinel reset 为 -1。other_cb 另记录逐 source×window 的 sparse COO（非零项 בלבד），并单独汇总固定四个 low-intersection KC。

只做等长窗口的 low43/high17 描述：每 class×window 输出 all/low/high 的 n、min、median、mean、max 及 high-low mean/median difference。other_cb 未出现帧补 0。固定 top20 规则在查看正式 60 帧结果前冻结为：对每个 other_cb source 计算每个窗口的 `diff_w = mean_high(signed_w) - mean_low(signed_w)`，source score 为 `max_w(abs(diff_w))`；按 score 降序、body_id 升序取 top20；同时报告 winning_window、每窗 diff、all/low/high 支持帧数（该 source 该窗口 events>0 的帧数）和 winning_window 的 direction。APL 同样只按窗口统计，不使用不等长 KC10 窗口。所有排序、阈值和 perfect 支持均为 sample-only descriptive，不作因果或盈利结论。

observer 默认 null，magic/version/size/overflow fail closed；trace-off 对 S2k-C 逐帧匹配，trace-on/off 比较全部持久字段、weight、counts、queue、cursor、sim_ms、total_spikes、weights_frozen。JSON/NPZ 原子写并互相 pin hash。
