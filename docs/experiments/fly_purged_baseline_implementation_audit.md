# Purged Fly 基线实施审计（研究草案）

日期：2026-09-25

状态：仅为实施审计和协议草案；未授权回放、未启动真实交易、未修改 live/shadow 配置。

> **状态注释（2026-09-25）：superseded-by-implementation。** 历史审计结论和阻断记录保留不变；其实施状态已由 research-only purged phase-1 产物取代。实际产物：[`preregister`](fly_purged_baseline_phase1_preregister.md)、[`baseline_report`](../../runs/wc006_purged_fly_base_baseline_phase1/20260925T-frozen-rgb-seed17/baseline_report.json)、[`performance_summary`](../../runs/wc006_purged_fly_base_baseline_phase1/20260925T-frozen-rgb-seed17/performance_summary.md)、[`performance_audit_manifest`](../../runs/wc006_purged_fly_base_baseline_phase1/20260925T-frozen-rgb-seed17/performance_audit_manifest.json)。该实现仍仅限 paper/research-only；旧 V5 授权失效，不得复用或转正式授权。

## 结论摘要

现有 purged 产物还不能直接作为新的 V5.1/Fly 基线输入。2021 purged
selector 已经重导，但只保存了 selector mirror/checkpoint；没有新的
`wc006-candidate-stream/v1` manifest，也没有和它绑定的 XYSZ raw market
manifest。更重要的是，现有年度 runner 的 `_validate_annual_manifest` 把年度
slice 逐字绑定到旧的 46-record unpurged source manifest，因此直接替换年度
`records` 会被 fail-closed 拒绝。

当前应保持旧 V5 授权失效测试原样，不重签旧授权。purged 基线应先以
`formal=False`、纸面、研究状态运行；只有新候选流、市场授权、预注册和门禁
测试齐全后，才讨论是否另行签发新的授权。

## 已核对的输入事实

### Purged 2021 候选与 XYSZ 覆盖

从 `audit/p0-2021-selector-leakage/purged-reexport-2021/checkpoint.json` 的
`state_audit` 提取到 55 个代码。按现有 loader 正确绑定 manifest 自身 SHA256
后：

| XYSZ bundle | manifest SHA256 | raw code 数 | purged 2021 覆盖 | 缺失 |
| --- | --- | ---: | ---: | ---: |
| 2021 clean-start bundle | `5f3f147fd1b7338166b068e8bb78e09c9914b0d4ed8426d01317150808690434` | 151 | 23/55 | 32 |
| 2021–2023 annual bundle | `92f0e11b2524a17e6160d74611a8506139110f320ba89510e346aff582bbab85` | 125 | 14/55 | 41 |

因此“现有 125 + 26 补采数据已覆盖新 purged 股票”不成立。现有 2021
bundle 覆盖的是旧候选流中的代码；purged 选出的代码中有大量缺 raw Kline。
必须先补采并生成新的 hash-pinned XYSZ market bundle，或者把缺失代码列为
明确的市场覆盖阻断项；不能退回 QFQ 执行层，也不能用缺失 bar 的占位值撮合。

现有 unpurged 46-record candidate stream 的 125 个代码全部在 125-code
annual bundle 中，但这只能证明旧流的 coverage，不证明 purged 流 coverage。

### Runner / manifest 约束

- `experiments/a_share/run_isolated_selector_slice.py` 的
  `--purge-train-labels` overlay 已经可以按严格日期顺序重导 selector；它
  不会写交易仓库，并在 checkpoint 中记录 overlay/source hash。
- `experiments/a_share/build_wc006_candidate_stream.py` 可以把
  `rebalance_target_*.json` 转成 immutable candidate stream，但输入 artifact
  必须有 `source=rdlab_leading_rebalance`、`rebalance_applied=true`、精确
  `prefer_tag`、`effective_target_topk` 和 signal/next-session cache。
- `experiments/a_share/wc006_male_cns_v51_adapter.py` 的年度校验要求
  `schema=wc006-v5.1-annual-input-slice/v1`、242 个年度 observation、且
  annual `records` 与固定 source manifest 逐项 canonical-equal。当前默认
  source hash 为旧 unpurged stream 的
  `5874b6cefab5f1c4b1ee04db819fd76430cdf67495db67db5242f5d83f42cb52`。
- `_run_v51` 的 annual path 已要求 supplied XYSZ market manifest 并验证 raw
  execution；但 `formal=True` 会明确拒绝，因为 V5.1 新授权尚未存在。
- V5.1 当前只允许 `arm="combined"`，它在同一次 paper replay 中写入
  `ledger.sqlite`（Fly）和 `paired_base.sqlite`（WC006-only）。可选
  `plasticity_mode` 为 `frozen` 或 `enabled`。

## 推荐的实施顺序

1. **先完成 purged 46-record stream，不跑 Fly。**

   以正式旧源的同一日期范围 `2021-07-01..2023-12-29` 生成 46 个 purged
   selector signal，保持 seed、topk、hold、turnover overlay、signal lag 和
   exact cache tag 不变；输出到新目录，不覆盖现有 stream。2021 Jan–Jun 的
   19-record clean-start purged 结果可作为独立审计附件；若要把它并入正式
   46-record协议，需另写 observation-window 预注册，不能默默改变 scope。

2. **由新 stream 的完整代码并集反推 XYSZ 覆盖。**

   对每个 target code 验证 signal、execution、全 replay window、settlement
   boundary 的 raw Kline、status、limit、adj-factor、dividend 和 right-issue
   数据。补采完成后生成新 market manifest、QA、父清单和 paper-only 授权绑定。
   任何缺失都在 gate 中 fail closed。

3. **新增 research-only purged annual validator/runner。**

   不要放宽旧 `_validate_annual_manifest`，也不要改旧 source hash。推荐新增
   一个显式的 `purged_source_manifest` 路径（或独立函数），要求：新 source
   manifest hash、purge overlay hash、selector artifact hashes、market hashes、
   46 条记录和 2021–2023 逐年 slice 全部一致，然后调用 `_run_v51(formal=False)`。
   这保持旧正式授权的 fail-closed 语义，同时允许研究产物有完整 provenance。

4. **先跑 frozen baseline，再考虑 enabled。**

   第一阶段只跑 `plasticity_mode="frozen"`，因为它回答的是“相同 purged
   candidate stream 下，执行/paired-base 账本能否复现”；enabled 会引入
   reward/plasticity 机制变量，不能用来掩盖输入流、XYSZ 覆盖或执行层问题。
   frozen run 通过 preflight、完整 artifact/hash、恢复一致性和纸面交易门禁
   后，才另行预注册 enabled arm。enabled 不是 frozen baseline 的替代品。

5. **按预注册加入 null/对照。**

   - primary：purged stream + frozen `WC006+Fly`，自动 paired
     `WC006-only`，seed 17，NullSensoryAdapter，XYSZ raw execution。
   - random/permuted-action null：固定同一输入、日历、市场和账本；只对动作
     序列做预先定义的随机/置换，seed 列表在回放前冻结。不能用盈利结果挑选
     seed，也不能让 risk guard 替换动作。
   - Base-with-retry：作为单独的 deterministic execution control，明确定义
     “在与 Fly 相同的非-selector exposure-rebalance 机会重试被 veto 的目标，
     不调用神经输出、不改变候选选择”。它不能通过事后从 Fly ledger 删除或
     复制 fills 得到；应独立写 ledger，并记录每次 retry 的原始 veto、重试日期、
     目标和结果。

   这三个对照应共用 market/candidate/seed/account 初始状态，但不能把
   Base-with-retry 的收益解释为神经收益。若 runner 目前不支持该控制，应新增
   独立 deterministic runner 和测试，不把 `V51_ARMS` 随意扩成伪造 neural arm。

6. **最后才跑 enabled arm。**

   enabled/frozen 必须使用相同 purged manifest、同一 XYSZ bundle、相同 scope
   和相同 seed；分别独立 clean-start，不能从 frozen checkpoint 恢复。报告应
   只给机制对照、动作/奖励/资格覆盖和账本差异；不把单年或单 seed 差异称为
   盈利学习。

## WC009 量化回测审计

量化仓库中存在多个 WC009 变体，名称不能直接假定为用户所说的“wc009”：

- `rdlab/configs/wc009_mining.json`：37 个特征、`weight_cap=0.06`，主要是
  mining 配置；
- `rdlab/configs/wc009a_pruned.json`：25 个 pruned 特征、`weight_cap=0.06`；
- `rdlab/configs/wf_ok1867_shadow_wc009a_pruned_weight_cap_null.json`：25 个
  pruned 特征、`weight_cap=null`，但使用另一版本 cache/universe tag；
- `rdlab/configs/wf_ok1867_shadow_wc009a_pruned.json`：shadow 配置、
  `weight_cap=null`，也没有显式 `signal_lag_one_day`。

这些 config 目前都可在 walk-forward 层加
`cfg.walkforward.purge_train_labels=True`；代码会在
`run_walkforward_continuous_daily` 记录该字段。正式补跑前必须由用户/父任务
明确选择要审计的 WC009 config，建议优先选择与当前 operational shadow
一致的 `wf_ok1867_shadow_wc009a_pruned.json`，并另做 `weight_cap=null` 与
旧 `0.06` 的敏感性，不要把不同 cache/universe tag 混为一组。

WC006 的 purge A/B 已经完成（56 windows，A/B 同成本和 lag；A 25.30%/80.4%，
B 3.49%/48.2%），但这不替代 WC009 回测，也不构成 selector 优势证明。
WC009 需要同样的 A/B：原 config（purge false）和只增加
`purge_train_labels=true` 的副本，固定 cost/slippage、signal lag、universe、
cache tag、日期窗和 seed；输出必须采用新文件名，禁止覆盖 live/shadow 结果。

## 必须新增的测试/门禁

- purged source manifest 与 candidate stream：46 records、严格 signal order、
  artifact/config/cache hash、purge overlay hash、next-session execution；
- purged annual validator：拒绝旧 unpurged source hash、缺 market manifest、
  market hash/QA hash drift、代码 raw Kline 缺失、年度 record mismatch；
- XYSZ coverage：对每个 purged target code 验证 raw execution coverage，不能
  只检查旧 125/151 code count；
- frozen/enabled resume parity：相同模式可恢复，不同模式或不同 stream/hash
  必须拒绝；
- Base-with-retry：重试只复制明确的 veto opportunity，不产生新的 selector
  或 neural action；
- 保留现有 V5 formal authorization rejection 测试，特别是旧 V4 adapter hash
  mismatch / legacy authorization rejection；不更新旧 authorization hash。

## 当前阻断与交接

在新 XYSZ market bundle、purged 46-record stream 和新预注册完成前，不能启动
新的 Fly replay。可以先完成 selector benchmark（等权、random/置换 null、
逐月 paired/significance），因为它不触碰 Fly/neural ledger；但它的结果不能
倒过来改变已经冻结的 purged stream 或挑选 Fly 配置。

