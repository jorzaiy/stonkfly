# V5.1 Independent Clean-Start 2021 Baseline

Status: **NON-AUTHORITATIVE (2026-09-24).** The 243-session replay completed, but it deviates from the preregistered protocol and its selector inputs leak future labels. Do not use it as a performance baseline or as Fly-vs-Base evidence. See "Known defects" below.

## Known defects (2026-09-24)

Audit reports: `audit/p0-2021-002432-concentration/` (`report.md`, `results.json`; review in `HANDOFF.md` §7) and `audit/p0-2021-selector-leakage/`.

1. **Execution ran on QFQ, not XYSZ raw.**
   - Cause: `validate_v4_stream` loaded the pinned market manifest only when `expected_records == 46`. The 2021 runner passes 19 records, so it silently used `market=None`.
   - Effect: fills, marks and rewards used the QFQ observation cache. The official limit/ST/suspension guards and corporate actions never ran.
   - Evidence: all 165 Fly fills and all 93 Base fills equal QFQ open × (1 ± 5 bp).
   - Re-checking the fills with the official guards would reject 31 Fly fills and 16 Base fills. Most of these are ST-policy rejections (28 and 13). About 3 per arm could not physically have filled: buys at an open equal to the limit-up price, and a sell of 002356 on a suspended day.
   - Fixed: the adapter now loads any supplied manifest and fails closed, and the V5.1 annual gate refuses to run without a loaded market.
   - Scope: only this run was affected. Every 2022 and 2023 run provenance records the full raw bundle (125 kline hashes).
2. **The WC006 selector trains on unpurged forward labels.**
   - Cause: `export_leading_daily_signal.py` sets `train_end` to the start of the signal month and drops no samples. The labels are `close.shift(-20)/close - 1` computed on a cache that runs through 2026-04-03.
   - Effect: roughly the last 20 training samples before each signal date have labels that end after the signal date.
   - Scope: this affects every candidate stream built with this exporter. That includes the 2021–2023 46-record stream (the same post-overlay exporter hash `c76a6f8d…`), and therefore the 2022 corrected-B and 2023 inputs.
   - Absolute Base returns (+198% in 2021) are not credible.
3. **The Fly excess depends on a protocol asymmetry.**
   - Mechanism: the Base arm orders only on selector dates, and its entries are vetoed by `total_exposure` when close-based lot sizing meets an open gap-up. Fly re-enters the same names through non-selector exposure-rebalance batches.
   - Size: those "rebalance after a Base veto" lots contribute +495,633, which is more than the whole Fly−Base gap. 002432 alone accounts for 100.8% of the gap.
   - This is not evidence of neural skill.

Purged checks (2026-09-24; audit/p0-2021-selector-leakage/):

- **2021 re-export with purged labels.** Same mirror, same seed; only the leaked tail is dropped (`run_isolated_selector_slice.py --purge-train-labels`).
  - On 2021-12-01, 002432 falls from rank 1 (97.4%) to rank 6 (0%). The purged top-3 is 300264 / 300199 / 300081 (50.4 / 27.7 / 21.9%).
  - On 2021-07-01, 002356 falls from rank 1 (97.7%) to rank 3 (0.9%). 300350 takes 97.8%.
  - The top-3 baskets share on average 0.89 of 3 names.
  - A crude next-open-to-next-signal basket return (no costs, lots or limits; direction only) goes from +1293% to −50%.
- **Walk-forward A/B in the trading repo** (56 monthly windows, 2021-07..2026-02).
  - Control (operational config): mean monthly net 25.3%, win rate 80.4%.
  - Purged: mean monthly net 3.5%, win rate 48.2%. It is worse in 44 of 56 months and in every year.
- Conclusion: the selector's historical edge, including the 002432 pick, is attributable to the label leak.
- Any re-run of the Fly replay needs separate approval, a purged candidate stream, and a new preregistration.

## Fixed protocol

- Scope: observations `2021-01-04` through `2021-12-31` (243 sessions).
- Settlement: `2022-01-04`, settlement-only; no boundary observation or reward pulse.
- Candidate stream: frozen 19-record WC006 selector stream, 3 candidates per signal, next-session execution.
- Selector: existing `wf_ok1867_candidate_return_leading_hold20_contiguous_continuous_wc006` config, seed 42 as recorded in selector artifacts; extended QFQ cache supplies causal 2020 lookback only.
- Fly: full retained MaleCNS v1.0 graph, clean brain on first 2021 session, RGB-only, `NullSensoryAdapter`, seed 17, frozen plasticity.
- Comparator: paired deterministic `WC006-only` base arm in the same replay.
- Execution: paper-only, Decimal, 100-share lots, T+1, existing fees/slippage/guard, intent-before-fill.

## Input pins

- Candidate manifest: `runs/wc006_v5_1_2021_baseline/inputs/candidate_stream/manifest.json` (hash is recorded in the completed runner provenance).
- Extended QFQ cache: `runs/wc006_v5_1_2021_baseline/inputs/cache/manifest.json` (1,867 files; 2020 prepend).
- Market manifest: `runs/wc006_v5_1_2021_baseline/inputs/market_inputs/manifest.json` (SHA256 `5f3f147fd1b7338166b068e8bb78e09c9914b0d4ed8426d01317150808690434`).
- Derived 2021 market QA: `quality_report.json` (SHA256 `8a6733705ef44804a970faa52b698132657112dde7c0ec40aa822d9754920900`); historical parent QA is unchanged.
- Graph manifest: `data/manifest.json` (SHA256 `7f7c3bda9418c04d2e86cc7245b3569c7b63c772459f28b77ee328b192a86340`).

## Selector overlay provenance

`inputs/candidate_stream/manifest.json` embeds the mirror provenance hash and patch records. The presentation (`none-safe-target-topk-render-v1`) is report-only; the stateful watch carry is an isolated causal artifact fix; the eligible-prior turnover-cap overlay is the pre-registered selector addendum, not a neural/execution policy. No selector output is regenerated during Fly replay.

## Reproducibility and outputs

Run `PYTHONPATH=. .venv/bin/python experiments/a_share/run_2021_chunk_replay.py --audit-only` before replay. Use increasing `--target-days` values against the same output directory; the ledger checkpoint and brain checkpoint resume the exact prior state. Each chunk writes `runs/wc006_v5_1_2021_baseline/chunks/chunk_2021_<target>.json`; stdout/stderr should be captured by the coordinator.

Required final artifacts: `events.jsonl`, `observation_outcomes.jsonl`, `forward_outcomes.jsonl`, `ledger.sqlite`, `paired_base.sqlite`, `provenance.json`, `runtime_metrics.jsonl`, alternating brain checkpoints, chunk manifests, `metrics.json`, `comparison.json`, and a final input/output manifest with hashes.

Finalization policy: SQLite connections are closed only after `PRAGMA wal_checkpoint(TRUNCATE)`; `ledger.sqlite` and `paired_base.sqlite` are stable pinned artifacts. The transient `*-wal` and `*-shm` siblings are explicitly excluded from `final_manifest.json` and must not be treated as missing artifacts. The final stable-artifact consistency test is `tests/test_v51_2021_final_artifacts.py` (2 passed).

No Jev A/B or performance conclusion is authorized until the baseline audit is complete.

## Completed result

- Fly: final equity `766408.0303783079`, return `666.4080%`, annualized Sharpe `3.83164`, max drawdown `-33.5952%`, turnover notional `5477490.56`, fees `4640.85`, fills `165`, vetoes `10`, daily-positive fraction `57.38%`.
- Direct Quant Base: final equity `297633.3732180176`, return `197.6334%`, annualized Sharpe `2.38422`, max drawdown `-32.5231%`, turnover notional `3632515.36`, fees `3043.19`, fills `93`, vetoes `7`, daily-positive fraction `54.92%`.
- Average holding period: unavailable; current lots table stores no acquisition date.
- These are paper replay values only and are not a validated profitability claim.

## Concentration / anomaly-risk note

The exceptionally high annual result is not diffuse evidence of a generally profitable Fly policy. The event ledger shows a concentrated path: on `2021-12-08` the replay bought `7,300` shares of `002432`, and on `2021-12-13` it bought another `7,900` shares while the active basket remained centered on that code. The subsequent portfolio returns include approximately `+9.45%` on 2021-12-13, `+8.31%` on 12-14, `+8.61%` on 12-15, `+8.50%` on 12-17, `+8.76%` on 12-20, `+8.90%` on 12-21, `+8.85%` on 12-22, `+8.85%` on 12-24, `+9.03%` on 12-27, and `+9.17%` on 12-29. It also suffered approximately `-7.04%` on 12-23 and `-6.71%` on 12-31. The paired base did not build the same `002432` position and had returns near zero on those dates. On 12-30 the Fly sold `5,900` shares of `002432`.

This concentration and timing make the headline return highly path-dependent. It must be audited against raw execution bars, status/limit handling, candidate weights, and ledger fills before any interpretation; it must not be generalized to broad profitability or learning ability.

### Priority 0 audit TODO (report-only; canonical baseline immutable)

- [x] Reconcile `2021-12-07`–`2022-01-04` XYSZ raw OHLCV, status, limit, adj-factor and corporate-action inputs against Fly/base fills, marks and settlement. (Done. XYSZ raw was never used for execution; see Known defects 1. The three 002432 fills would pass the official guards.)
- [x] Explain why Fly bought `7,300` shares on `2021-12-08` and `7,900` on `2021-12-13` while the paired base did not co-hold: candidate timing, vetoes, cash, lot sizing, repeated `EXPOSE`, and effective/requested multipliers. (Done. Base was vetoed by `total_exposure` on 12-02 and 12-30, and Fly re-entered via rebalance batches; see Known defects 3.)
- [x] Produce report-only attribution artifacts for the baseline prefix, `002432` contribution, and no-`002432` counterfactual; never overwrite canonical replay artifacts. (Done: `audit/p0-2021-002432-concentration/results.json`. The no-002432 figure is an open-loop approximation.)
- [ ] Run 20% and 30% single-name caps only as sensitivity analyses; do not treat them as baseline changes.
- [x] Audit whether selector/config inputs or feature generation improperly saw 2021 data. (Done. The selector trains on unpurged forward labels; see Known defects 2. The universe is survivor-conditioned. Whether the leak drove specific picks is pending the purged re-export.)
- [ ] Keep the Jev performance gate closed until this audit is complete.

The authoritative machine-readable outputs are `runs/wc006_v5_1_2021_baseline/metrics.json`, `comparison.json`, and `final_manifest.json`; chunk stdout/stderr is under `replay_logs/`.
