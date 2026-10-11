# WC006 + complete MaleCNS v4 protocol

Status: immutable staged preregistration. This phase authorizes implementation and tests only; after independent input/parity/resume/live-shadow gates pass, one clean-primary v4 training run is authorized. Smoke, broker and service actions remain forbidden.
Version: `2026-09-18.v4`

This document extends, but does not edit, `wc006_v3_2021_preregister.v2.md`
and the experiment-only turnover overlay
`wc006_v3_2021_preregister.v3-turnover-overlay.md`. The original v1 document
and both earlier files remain frozen. v4 is an independent adapter and output
namespace; it must not overwrite v3 runs or modify the trading repository.

## Failure record motivating v4

The completed v3 primary paper run at
`runs/wc006_v3_2021_train/20260918T134417Z-clean-primary` used manifest
SHA-256 `5874b6cefab5f1c4b1ee04db819fd76430cdf67495db67db5242f5d83f42cb52`,
graph SHA-256
`70b253b10eaecd2492ced3b143b3b391e10b49b8a62ced7a0ec105930b68cd7f`, and a
clean initial brain. Its final checkpoint was
`brain-1.npz`, SHA-256
`658ff98797303344b0322ceb861fa9bebd8c8bab16c117cc866ac53b8bee60e1`.
The run is retained as an immutable diagnostic failure record, not as a
checkpoint source: 46 observations yielded 39 DEFER, 7 EXPOSE and 0 DEEXPOSE;
reinforcement was 28 aversive, 17 reward and 1 none. No 2024/2025 result may
be used to change this protocol.

The v3 run's low-frequency observation schedule is the failure mode under
test, not evidence that the MaleCNS graph should be pruned or that the fixed
decoder should be changed.

## Primary arm and causal dates

The primary arm uses the exact repaired WC006 candidate stream and complete
MaleCNS/v3 `FlyController` graph. The brain is clean at the first causal date;
no v2/v3 historical checkpoint is accepted. The development window is
`2021-07-01` through `2023-12-29`. A next-session settlement boundary may be
read on `2024-01-02` only to close the already-created `2023-12-29` intent;
learning is disabled on that boundary and it creates no observation or reward.
It is not a validation result. The frozen validation and exam schedule remains:

| phase | dates | use |
|---|---|---|
| development/training | 2021-07-01 through 2023-12-29 | learning enabled |
| validation | 2024-01-01 through 2024-12-31 | frozen, not used in v4 implementation |
| exam | 2025-01-01 through 2025-12-31 | frozen, not used in v4 implementation |

The current implementation phase does not run the 46-date production schedule. A later gated phase may run exactly one clean-primary development training, with no smoke, broker, service, or 2024/2025 validation execution.

## Daily observation and signal-only execution

After the first WC006 signal date, every valid calendar session receives one
composite observation and may update neural/plastic state. The observation
uses only data at or before that date. The first signal date itself is the
first observation; no pre-signal observation is silently synthesized.

Only the 46 WC006 signal dates are execution-eligible. On all other sessions,
the decoder output is persisted as `schedule_blocked`; it creates no intent,
fill, or replacement order. On signal dates the fixed decoder is unchanged:

- `BUY` -> `EXPOSE` multiplier `1.0`;
- `SELL` -> `DEEXPOSE` multiplier `0.5`;
- `HOLD` -> `DEFER` retaining the current multiplier.

The most recent approved candidate set and weights are carried forward between
signals. A changing candidate set is never replaced by the 2026 smoke set.

## Composite feature contract

Each active candidate requires the last 61 available close/volume bars at or before the observation date: the current causal bar plus 60 prior available bars. This gives exactly 60 return intervals; fewer than 61 available bars fails closed. If the market calendar has no bar for the observation date, close is carried from the last available bar, volume is zero, and staleness/missing-bar provenance is encoded. The plotted normalized trajectory may use a fixed 60-row view of those 61 rows. The RGB composite encodes, for
each current candidate, causal normalized price and volume trajectories plus:

- 5-, 20-, and 60-session close returns;
- 5-, 20-, and 60-session close-return volatility;
- target exposure and current exposure;
- cash fraction and current drawdown.

No future row, future candidate, future weight, or future execution price may
enter the frame. The full retained graph and sensory/plasticity separation are
preserved.

## Risk, ledger and paired base

The WC006 paired base and Fly use identical candidates, canonical prices,
calendar, costs, 100-share lots, T+1 sellability, candidate-exit liquidation,
and guards. The explicit risk budget is:

`target_exposure = target_weight * 0.85 * multiplier`.

At each open, pending SELL intents are settled first, then BUY intents; within
each side ordering is deterministic by symbol and intent id. The guard can veto
an intent for cash, aggregate exposure, T+1, lot or market state; it can never
choose a replacement symbol or action. Base target weights
remain hash-pinned and are not silently renormalized.

Every daily step records primary and paired-base SQLite state, an event-chain
entry, and a Fly checkpoint. A checkpoint is written after each observation.
Interruption/resume must match uninterrupted daily neural decisions, signal
intents/fills, both ledgers, event chains, and final checkpoint bytes/hash.

Daily reinforcement uses `incremental_feedback` against the paired base with
actual same-day turnover, drawdown and no future 5-day labels. Execution costs
are charged on the actual fill date only. A pending execution with no exact
open bar is explicitly vetoed as `missing_open_or_unverified_suspension`; no
future bar is substituted. Primary and paired base persist identical mark
provenance for last-known closes.

## Input, isolation and stop rules

The adapter must fail closed on manifest/source/config/universe/calendar/cache
hashes, exact tag, 46 records, signal-to-next-session dates, 61 available close/volume bars for 60 return intervals,
positive normalized weights, or provenance mismatch. All
outputs are below unique `stonkfly-lab/runs/` directories. The trading
repository, live/shadow state, service and broker are read-only/out of scope.

At completion, only the 46 signal-eligible decisions are counted. If BUY,
SELL, or HOLD is absent, or DEFER is at least 90%, the result is persisted as
`quarantined_action_collapse`; decoder thresholds are never changed to force
BUY/SELL/HOLD balance. Any checkpoint, ledger, event,
unknown-outcome, duplicate-intent, negative-cash, non-lot or ghost-holding
failure stops the run. No performance result may select a seed or alter the
protocol. Official ST/suspension/limit evidence remains absent in the current
bundle, so all outputs are non-authoritative paper accounting.
