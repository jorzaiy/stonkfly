# WC006 + complete MaleCNS runtime: 2021--2025 preregistration v2

Status: preregistered specification only; no training, replay, return analysis,
broker order, or service action is authorized by this file.
Version: `2026-09-18.v2`

## Primary arm (the requested WC006 + fly experiment)

The primary arm is repaired WC006 selector output driving the complete existing
MaleCNS/v3 runtime. It is not the single-ETF v3 ledger. For every rebalance,
the repaired WC006 walk-forward selector must causally produce the candidate
set and base target weights; the selector artifact, active config, universe,
calendar, code version, and exact canonical price files are copied or
hash-pinned into an independent `stonkfly-lab/runs/` directory.

The fly receives WC006's multi-symbol composite observation over the candidate
symbols and uses the fixed WC006 decoder:

- `BUY` -> `EXPOSE` with multiplier `1.0`;
- `SELL` -> `DEEXPOSE` with multiplier `0.5`;
- `HOLD` -> `DEFER`, retaining the prior multiplier.

The paired ledger is identical to repaired WC006: close-T observation,
intent at T, next valid session open fill, canonical unadjusted OHLC semantics,
100-share lots, T+1 sellability, WC006 commission/minimum commission,
slippage, stamp duty, exposure guard, checkpoint/signature, and resumable
SQLite/event-chain persistence. Candidate symbols may be a changing subset
of the pinned universe; no fixed six-symbol 2026 smoke set is allowed.
Official ST/suspension/limit state is required for an authoritative run; absent
external verification, artifacts remain paper-only and non-authoritative.

The brain is newly initialized at the start of 2021. No existing v2 or v3
checkpoint may be loaded into the primary arm. The causal schedule is fixed:

| phase | dates | learning | rule |
|---|---|---:|---|
| train | 2021-01-01 through 2023-12-31 | enabled | one prespecified run |
| validation | 2024-01-01 through 2024-12-31 | frozen | one read-only pass |
| exam | 2025-01-01 through 2025-12-31 | frozen | one pass after validation seal |

No validation result may change selector, graph, reward, decoder, costs, seed,
initialization, or hyperparameters before the 2025 exam.

## What v3 means here

“v3” identifies the existing complete MaleCNS graph/runtime and its neural
state machinery; it does not introduce a new graph for this experiment. The
important differences from the original single-ETF v3 replay are environment
and state/reward handling: WC006 supplies a multi-symbol composite market
frame, WC006 candidate/base weights, WC006 action decoder, and WC006 ledger
state. The single-ETF `510300` portfolio ledger and its ETF-specific stamp-duty
assumption are not the primary arm.

The audited checkpoint
`runs/a-share-510300-v3-train-2018-2022/brain-1.npz` has SHA-256
`5e7a3c52d894810ba4658b10684eed26b9f088faf50471408e967236364ce641` and was
learned online through 2022-12-31. Loading it from 2021 would include future
2021--2022 experience and is direct lookahead. It is therefore forbidden in
the primary arm.

## Secondary transfer ablation

The audited 2018--2022 v3 checkpoint may be used only in a secondary transfer
ablation beginning no earlier than 2023. The ablation must preserve its exact
checkpoint hash and report the overlap with 2021--2022 training. It may not be
used to populate the 2021 primary arm, replace the clean brain, or support a
claim that the single-ETF ledger is equivalent to WC006.

## Candidate-stream gate before any training

The selector stream is a hard prerequisite. A small causal slice must pass
before full-period construction is considered:

1. The active WC006 config must explicitly have `signal_lag_one_day=true`.
2. Each signal date must use only prior training labels/features, and its
   execution date must equal the first later date in the pinned trading
   calendar.
3. Each artifact must have `rebalance_applied=true`, nonempty effective
   candidates and positive base weights. Empty schedule placeholders are a
   hard failure, not zero-weight candidates.
4. Candidate codes must be in the pinned universe and every selected code must
   have signal/execution bars in one exact canonical cache tag. Each copied
   source/config/universe/calendar/selector/adapter file is hash-pinned.
5. The resulting manifest must be paper-only, broker-disabled, and written
   outside the trading repository. No returns, PnL, training, or parameter
   search may be inspected during this gate.

The existing 2021 artifacts audited on 2026-09-18
(`rebalance_target_20210406.json`, `20210506.json`, and `20210601.json` under
`data/watch/validate_hold20_wc006`) are schedule placeholders: each has
`rebalance_applied=false` and empty target arrays. They cannot establish the
required stream. The independent adapter therefore fails closed on this slice
and creates no output. A future selector run must emit nonempty artifacts into
Stonkfly-owned output before full training is unblocked.

## Isolation and freeze

All derived manifests, copied inputs, checkpoints, SQLite ledgers, event logs,
and reports belong below unique Stonkfly run directories. The trading repo's
live/shadow/watch/config files remain read-only, and no service is started or
stopped. The v1 preregistration file is retained unchanged; this v2 file is a
new specification and has its own frozen SHA-256.

No financial conclusion, executable authorization, or claim of historical
performance is permitted until the candidate-stream gate, exact hashes,
causal-date proof, authoritative-state decision, parity/resume tests, and the
v2 hash are independently reviewed.
