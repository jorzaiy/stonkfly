# Purged XYSZ 缺口补采审计

日期：2026-09-25

本文件只描述离线审计和补采计划；没有联网、没有调用 AmazingData、没有修改旧
market bundle、authorization、live 或 shadow 配置。

## 旧 26-code supplemental 的边界

`runs/wc006_v5_1_2021_baseline/inputs/market_collection/` 下的五个 fragment
（`kline_26codes`、`history_stock_status_26codes`、`dividend_26codes`、
`right_issue_26codes`、`equity_structure_26codes`）都绑定相同的 26 个旧代码。
它们的 manifest 均记录：

- `begin_date=20210101`；
- `end_date=20220110`；
- kline/status 的数据并不覆盖 2022-01-11 以后，更不覆盖 2023；
- `exit_mode=hard_exit_after_commit`、`sdk_version=1.1.9`、
  `credential_values_persisted=false`。

因此这批文件只能作为 2021 clean-start bundle 的历史 supplemental，不能当作
最终 purged 2021–2023 缺口的补采结果。它也没有 `adj_factor_26codes` fragment；
当前 151-code parent 复用了旧 125-code adj-factor children。新 purged 代码
并集变更后，adj_factor 仍必须对完整代码并集重新覆盖并 QA。

## 精确补采计划

现有 SDK 端点和建议参数如下。`2020-01-01` 的 action begin 有意保留 2021
年内生效但 2021-01-01 前公告的公司行动。

```text
AmazingData.MarketData.query_kline(
    codes, begin_date=20210101, end_date=20240103,
    period=ad.constant.Period.day.value)

AmazingData.InfoData().get_history_stock_status(
    codes, local_path=cache_dir, is_local=False,
    begin_date=20210101, end_date=20240103)

AmazingData.BaseData().get_adj_factor(
    codes, local_path=cache_dir, is_local=False)
# pinned SDK 不暴露日期参数；保留原始返回，QA 检查所需窗口

AmazingData.InfoData().get_dividend(
    codes, local_path=cache_dir, is_local=False,
    begin_date=20200101, end_date=20240103)

AmazingData.InfoData().get_right_issue(
    codes, local_path=cache_dir, is_local=False,
    begin_date=20200101, end_date=20240103)

AmazingData.InfoData().get_equity_structure(
    codes, local_path=cache_dir, is_local=False,
    begin_date=20200101, end_date=20240103)
```

六个 dataset 都需要：kline 和 status 直接影响成交/估值/风控；adj_factor、
equity_structure、dividend、right_issue 是 loader 要求 hash-verified 的公司
行动证据。不能因某个代码“看起来没有分红/配股”而省略它的原始查询。

## 新 bundle 的构建与 QA

1. 先用 `tools/plan_purged_xysz_gap.py` 读取最终 purged candidate-stream
   manifest 和旧 parent manifest。工具只读 manifest、校验 child SHA256、按
   dataset 输出缺口和 query plan，不登录、不联网。
2. collector 将每个 dataset 写入新、唯一的 research run 目录；不覆盖旧
   parent。每个 fragment 的 `manifest.json` 必须包含 dataset、完整 qualified
   code list、query window、SDK/退出模式、artifact path、artifact SHA256、行数
   和字段。
3. 新 parent `manifest.json` 的 `source_manifest.codes` 必须精确等于 purged
   stream 代码并集，`code_count` 一致；`child_manifests[].fragment` 使用相对
   新 parent 的路径。不要把旧 125-code source manifest 原样留下，否则 loader
   会按旧 code count 拒绝新 raw Kline 集合。
4. 按 `tools/rebuild_v5_market_qa.py` 的检查原则重建新 QA：raw Kline
   code-date 唯一、OHLCV 有限/正值/范围合法；status 覆盖每个 raw code-date、
   字段/flag/limit 合法；adj_factor 每列覆盖、日期唯一、正值；equity
   structure code 完整且主键无重复；dividend/right_issue 保留 raw rows，按
   `ANN_DATE <= effective_date` 和 progress=3 重算 PIT eligible rows；记录
   raw-vs-QFQ date-key 差异但不填补。
5. 新 QA 和 parent 完成后，另生成新的 development-only paper
   authorization，绑定新 parent/QA SHA256、purged stream SHA256、adapter/code
   lock 和 scope。旧 V5 authorization 不得复用；正式 `formal=True` 仍保持
   fail-closed，直到治理层另行批准。

新增离线规划器：`tools/plan_purged_xysz_gap.py`。
测试：`tests/test_plan_purged_xysz_gap.py`，结果 `2 passed`。
