# V5.1 corrected Branch B 2022 baseline audit

Status: evidence and usage document for the completed, paper-only 2022 replay. It
is not a performance claim for the historical Branch A run, and it does not
authorize a live order, a Jev run, or a change to the trading strategy.

> **Caveat (2026-09-24):** the Branch B candidate stream comes from the WC006 exporter that trains on unpurged 20-day forward labels (`train_end` is the start of the signal month; post-overlay exporter hash `c76a6f8d…`). This baseline is valid for execution and lineage fidelity only. Its absolute returns and its Fly-vs-Base comparison are not performance evidence until they are rebuilt on a purged stream. See `audit/p0-2021-selector-leakage/` and `docs/experiments/v51_independent_clean_start_2021.md`.

## Scope and decision boundary

The 2022 QFQ investigation began with a BaoStock observation-cache concern:
002703.SZ contained five flat-price rows with empty volume during
2022-06-29--2022-07-05, while the XingYao/XYSZ raw execution feed omits the
suspension interval. The audit separates the observation cache, selector,
neural observation, fixed decoder, execution vetoes, and accounting. The
cache is not used for execution fills, stale marks, or portfolio reward.

The independent checks found:

- QFQ prices agree with XYSZ on common trading sessions up to a constant
  adjustment factor; 2021--2023 has no split/dividend factor jump in the
  tested symbols and common-day return differences are at machine precision.
- The production selector drops rows with empty OHLCV fields. The clean
  1,266-row 002703 panel and the 2022-06-30 target selection are identical in
  the original and corrected selector streams. Therefore selector and direct
  quant/base conclusions are exact, not tolerance-based.
- The Fly observation frame did change on 2022-06-29. The direct candidate
  frame reconverged on 2022-06-30, but later frame/state differences are
  portfolio/neural feedback. The full Fly path did not permanently
  reconverge. Consequently historical Branch A is **sensitivity-qualified and
  not invariant**; its performance must not be presented as an invariant
  baseline.
- On 2022-07-01 and 2022-07-04 the 002703 intents were rejected by
  `missing_open_or_unverified_suspension`. There were zero false suspension
  fills in either branch.

The intended split is therefore: historical Branch A performance claim
`BLOCKED_NONINVARIANT`; corrected Branch B is the complete trustworthy 2022
baseline; the current Jev gate explicitly allows only the corrected-B lineage
after an exact provenance check. This permission only permits a future
paper-only plan; it never starts an experiment and does not waive
preregistration or NullAdapter regression requirements.

## Full-year replay evidence

The replay covers 242 observed sessions, 2022-01-04 through 2022-12-30, with
115 warm-up days. Settlement was already applied once at the 2023-01-03
boundary; it creates no new 2022 observation or reward.

| Metric | Preserved A | Corrected B |
|---|---:|---:|
| 2022-12-30 Fly equity | 558676.741040 | 555930.09604600 |
| 2022-12-30 Fly return | 4.5867674104 | 4.55930096046 |
| 2023-01-03 settlement equity | 544307.741040 | 535410.09604600 |
| 2023-01-03 settlement return | 4.4430774104 | 4.35410096046 |
| Fly MDD | -23.2709808915561368% (2022-07-14) | -22.5018598816303439% (2022-09-19) |
| turnover | 7132524.185 | 7584436.171 |
| fees | 5816.99396 | 6285.640954 |
| fills / vetoes | 157 / 18 | 172 / 16 |
| reward / aversive / none | 110 / 125 / 7 | 105 / 128 / 9 |

The paired direct/base arms are exact A/B matches: 12/30 equity
547898.94029, return 4.4789894029; settlement equity 504274.94029, return
4.0427494029; zero eventwise reward-base mismatches and zero paired snapshot
mismatches. Direct/base MDD is computed from the 242 observed day-end equity
marks, starting from the recorded initial equity, as the largest
high-water-to-later-equity drawdown. Settlement is explicitly excluded from
that path. The resulting A/B exact MDD is -33.1946818258900391% on
2022-09-19.

Decoder/action and exposure differences are reported separately:

- action disagreement: 82/242, 33.8842975206611570%; first 2022-06-29,
  last 2022-12-29;
- requested multiplier disagreement: 27/242, 11.1570247933884298%; first
  2022-07-07, last 2022-12-29;
- effective multiplier disagreement: 26/242, 10.74380165289256198%; first
  2022-07-08, last 2022-12-30;
- 2022-12-30 action strings match (`EXPOSE`), while full decoder/neural frames
  do not. `decoder_action_match` is the strict action-string comparison; it
  must not be confused with full frame equality.

## Corrected-B provenance and hash chain

The authoritative replay artifacts are under
`audit/independent-qfq-2022/counterfactual-replay/`. Branch B has 242 events,
last index 241/date 2022-12-30, no `.partial` files, and no SQLite WAL left
open. The root manifest records `COMPLETED`, canonical Branch A fidelity
`PASS`, direct/base identity, and the one-time 2023-01-03 settlement.

The four atomic chunks and their boundary/hash continuity are:

| Manifest | Coverage | Manifest SHA256 | Chain fact |
|---|---|---|---|
| `chunks/chunk_branch-b_138_to_160.json` | 137/2022-07-29 → 159/2022-08-30 | `9128f8f5fdd5298335adbc70113e6d5ebc153d3af5640caab089cf6f74bfb437` | input is clean day-137 state |
| `chunks/chunk_branch-b_160_to_185.json` | 159/2022-08-30 → 184/2022-10-12 | `8f13b4f1789d84605d09329f9d76fce29c7a4fb1270d8985f80f2cacebe50968` | input equals prior output |
| `chunks/chunk_branch-b_185_to_215.json` | 184/2022-10-12 → 214/2022-11-23 | `88b863310a0dc35397dd4ce29f9da8dd1fee6a89a445b333eb8b1d38bae68544` | input equals prior output |
| `chunks/chunk_branch-b_215_to_242.json` | 214/2022-11-23 → 241/2022-12-30 | `5ea5caf61c5d33527e7520a0945ad59edf03e25660dffb2a7fb96ae3a582b1c3` | final output is active B |

The root artifact hashes currently recorded by the replay evidence are:

- replay manifest: `e0916657cd8e0f77a3ac90845ff64256697bae86b469cc93cfced0575c0ab8d2`;
- comparison: `b20217bc65a76f4b3dcb5a166544162d4cd4daba5c122138940dd38d5ffc9310`;
- Branch B events: `5404d8927f2e6cacd5fa9829aaadb80d1403aca6de9c8828c1101c1c6e6bbe38`;
- Branch B ledger: `3f181ab516a5795fbac8b51747a74674ccd86deb867aac6ccfb4f015d5415be8`;
- Branch B paired base: `55e5de4b5ce17952312213333d8361022aab0d20f26209639525c5fd983bc55e`;
- Branch B `brain-day-0.npz`: `230f574a729dd8e8b39a13354f17e11c6c143086ff93dd355532938f20868e89`;
- Branch B `brain-day-1.npz`: `7a19260fb03aa7e519ad3eeee2a9b51008946781e29dedd94ae9c27ff7b05cc4`;
- Branch B provenance: `4f5e6c0ae010a6cf73bba6d291ba4fbf2a17a44d789d4c8b5f76e55898d092b0`;
- corrected candidate-stream manifest: `8d449810bc277d9103f2d21f7bae758f65874d9840a84caebb7600177381c9cb`;
- corrected 002703 cache: `61bef8f57dad6b421f8116bfc721c6e11c0aa94993486b2c9eb60363f1193564`;
- retained graph manifest: file SHA256 `7f7c3bda9418c04d2e86cc7245b3569c7b63c772459f28b77ee328b192a86340` (`70b253b10eaecd2492ced3b143b3b391e10b49b8a62ced7a0ec105930b68cd7f`, previously listed here, is the `brain.configuration_signature()` hash).

The required corrected-B lineage label is
`corrected-branch-b-2022-v5.1`. Its immutable inputs are seed `17`, frozen
plasticity, paper-only execution (`broker_orders_enabled=false`,
`live_execution_authorized=false`), the retained graph above, the corrected
candidate/cache stream, the replay manifest/comparison, the Branch B
checkpoint/state hashes, and the XYSZ market-input manifest
`92f0e11b2524a17e6160d74611a8506139110f320ba89510e346aff582bbab85`.
Changing any path, file hash, seed, graph, cache, candidate stream,
plasticity mode, or paper-only flag makes the lineage invalid.

## What this does and does not establish

This document establishes reproducible replay provenance and the causal
boundary between QFQ/selector/direct-base evidence and Fly observation
sensitivity. It does not establish profitable neural learning, consciousness,
pain, pleasure, biological fidelity, live trading performance, or that a
corrected Branch B will repeat in another year. It does not authorize a broker
or real order. Risk guards reject intents; they do not choose replacement
trades. Reward and aversive labels remain engineered inputs, not modeled
subjective experience.

Historical Branch A may be retained as sensitivity evidence and as preserved
canonical-fidelity evidence, but must not be used as the 2022 invariant
performance baseline. Future Jev experiments, if separately approved, must
start from corrected-B data/brain/config/seed/plasticity lineage, remain
paper-only, and pass their preregistration and NullAdapter regression gates.
An explicit lineage check must reject missing/unknown lineage, historical A,
path mismatch, any hash mismatch, seed mismatch, graph/cache/config mismatch,
non-frozen plasticity, and non-paper execution. A permitted lineage means
“eligible to plan a future experiment”; it is not automatic start.

## Reproduction and verification commands

These commands verify existing artifacts and do not run Jev or a new neural
replay:

```sh
sha256sum audit/independent-qfq-2022/counterfactual-replay/manifest.json \
  audit/independent-qfq-2022/counterfactual-replay/comparison.json \
  audit/independent-qfq-2022/counterfactual-replay/branch-b/events.jsonl \
  audit/independent-qfq-2022/counterfactual-replay/branch-b/ledger.sqlite \
  audit/independent-qfq-2022/counterfactual-replay/branch-b/paired_base.sqlite

.venv/bin/python audit/independent-qfq-2022/audit_xysz_qfq.py
.venv/bin/python -m pytest -q audit/independent-qfq-2022/test_audit.py \
  audit/independent-qfq-2022/test_observation_replay.py \
  tests/test_independent_qfq_audit.py tests/test_jev_ab.py
```

The current gate API is `stonkfly.jev_ab.ABConfig` plus
`make_plan(config)`. A corrected-B-only plan must explicitly provide
`baseline_lineage_id="corrected-branch-b-2022-v5.1"` and all four audited
paths: `candidate_stream` (the candidate manifest), `market_data` (the market
manifest), `brain_checkpoint` (the clean Branch B checkpoint), and
`direct_quant_baseline` (the paired-base SQLite file). `config` must include
`lineage_id`, `graph_manifest_sha256`, `candidate_cache_002703_sha256`,
`baseline_plasticity_mode="frozen"`, `paper_only=true`,
`broker_orders_enabled=false`, and `live_execution_authorized=false`; the
seed must be `17`.

The validator binds those paths and hashes to `audit.json`, the replay
manifest/comparison, Branch B provenance, and the graph/config evidence. It
rejects historical A, simple `{"status":"PASS"}`, missing/unknown lineage,
path or hash mismatch, seed/cache/graph/plasticity mismatch, and non-paper
flags. This baseline-provenance field does not override arm behavior:
the existing plan still keeps A/B frozen and C/D enabled. On an exact match,
`plan["performance_gate"]` reports
`status="UNBLOCKED_CORRECTED_B_LINEAGE_ONLY"`, `allowed=true`, and
`auto_start=false`; `performance_gate(plan)` only validates that plan and
does not start Jev. Existing preregistration and NullAdapter regression gates
remain mandatory.

For complete project architecture, governance gates, and handoff instructions, see [Project Handoff](../PROJECT_HANDOFF.md).
