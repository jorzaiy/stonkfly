# WC006 + Fly V5.1/V6 mechanism preregistration

Status: immutable staged specification. This document authorizes mechanism
assays and one paper-only annual replay only after the input, isolation and
resume gates below pass. It does not authorize a broker, service, live, shadow,
or funded run. It does not authorize changing the trading strategy
implementation.

Version: `2026-09-20.v5.1-v6-mechanism`

## Interpretation and scope

Fly is not a supervised trading policy and there is no LLM policy or hidden
profit-based action selector. The complete retained MaleCNS graph receives the
declared sensory and engineered reinforcement inputs; plasticity is a declared,
unvalidated mechanism hypothesis; a fixed decoder maps the measured DNp20 and
DNpe017 activity to `BUY`, `SELL`, or `HOLD`; and execution limits may only
reject an order. A random seed makes a run reproducible; it is not a replacement
trading policy. A mechanism result is not evidence of profitable learning,
consciousness, pain, pleasure, biological replication, or live performance.

This protocol is a continuation of the frozen V5 contract in
`docs/experiments/wc006_v5_preregister.md`. The V5 graph, sensory/plasticity
separation, fixed decoder, paper execution, Decimal accounting, and no-future-
data rules remain in force. V5.1/V6 adds audits and mechanism assays; it does
not prune the graph, add scripted trades, or fit a market policy.

## Fixed annual window and why it is fixed

The only market replay window is calendar year **2023**:

* observation and learning boundary: `2023-01-01` through `2023-12-31`;
* available A-share sessions in that boundary: exactly **242** hash-pinned
  observation sessions, `2023-01-03` through `2023-12-29`;
* candidate stream in that slice: exactly **17** signal records, each retaining
  its pinned next-session execution date;
* final T+1 settlement boundary: `2024-01-02`, only to close an intent created
  on `2023-12-29`; it creates no new observation, signal, reward, or learning;
* no 2024 observation, validation metric, or exam metric is opened.

2023 is the latest complete natural year inside the current V5 input coverage
(`2021-01-01` through `2023-12-29`) and the existing formal V5 runs. The fixed
slice was selected from the pre-run coverage audit: **125/125** required source
coverage, including the `002833` corporate-action correction (`corpfix`), is
present. These are coverage gates, not performance criteria. The three V5 arms
have a recorded resume-parity PASS in
`runs/wc006_v5_resume_parity_v2/parity-v2-20260919T200059Z-r3/summary.json`.
This choice is made from coverage and reproducibility, not from a return,
drawdown, action mix, or any other outcome. The 2021--2022 data may be used
only as causal feature lookback or to identify the already-pinned carried target
at the first 2023 signal; it creates no pre-year neural observation or reward.
No pre-year Fly checkpoint is imported. The annual replay starts from a clean
brain at its first 2023 observation.

The fixed replay seed is `17`, inherited from V5. The seed, year, arm order,
decoder threshold, reward deadband, cost profile, graph, candidate stream and
market artifacts are recorded before the first run. They cannot be selected or
changed after looking at annual results.

## Frozen graph, prices, arms, and execution

All arms retain the full MaleCNS v1.0 graph: 166,700 retained neurons and all
25,582,938 retained directed connections under the committed importer policy.
The plastic set is all existing KC-to-MBON07/11 connections, exactly 7,835
edges; no graph cropping or added edges is permitted. The existing V5 graph
manifest hash is `70b253b10eaecd2492ced3b143b3b391e10b49b8a62ced7a0ec105930b68cd7f`,
and the candidate-stream manifest hash is
`5874b6cefab5f1c4b1ee04db819fd76430cdf67495db67db5242f5d83f42cb52`. A fresh
run must re-verify these values rather than trusting this text.

The annual replay has exactly two paired accounting arms:

1. `WC006-only`: deterministic candidate and target-weight stream, multiplier
   `1.0`, no neural controller;
2. `WC006+Fly`: the complete retained graph, clean brain, declared plasticity,
   and the fixed decoder below.

The `Fly-only` single-ETF contract is not an arm of this protocol. It must not
consume the WC006 candidate union or be used to select a year, seed, threshold,
or result.

Both paired arms use identical hash-pinned candidates, raw execution/mark/reward
prices, qfq-only observations, official state gates, corporate-action rules,
100-share lots, A-share T+1, costs, risk budget `0.85`, and initial cash
`100000`. Paper execution is mandatory; `broker_orders_enabled=false` and
`live_execution_authorized=false` are hard gates. Order intent is persisted
before any simulated send. Unknown outcomes stop the run and are reconciled;
they are never blindly retried.

## Mechanism assay A: same-checkpoint reward/aversive persistence

This is a no-market, no-order assay. A single clean checkpoint, stimulus hash,
neural clock, parameter set, and seed are copied byte-for-byte into independent
branches before either branch is opened. The branch labels are fixed in advance:

* `reward`: the documented 200 ms pulse to the 15 PAM11 (alpha1) cells;
* `aversive`: the same-duration/current pulse to the two PPL101 (gamma1pedc)
  cells.

The pre-pulse cue, timing, and input hash are identical. The assay logs all
7,835 edge weights, KC eligibility traces, DAN rates, selected gain rows,
baseline weights, and checkpoint hashes before and after the pulse. It then
advances without another reinforcement pulse and records at 1 s, 60 s, and
1,800 s. Bounds are observed, not reset after an unfavorable result.

The 1 s point is retained as a full-network propagation smoke check. Because
the V5.1 sensory phase uses `learning=False`, the 60 s and 1,800 s retention
points are additionally produced by a hash-locked analytical passive
projection of the post-pulse `memory_u`/`memory_w` and baseline efficacy using
the closed-form zero-drive equations in `rule.py`. The projection records its
formula parameters, input/output hashes, and efficacy bounds; it does not
advance or claim to represent the complete neural state (membrane variables,
spikes, eligibility, or other traces).

The phrase “opposite-sign” in this protocol refers first to the sign of the
reinforcement labels (`reward` versus `aversive`), not to an assumed biological
sign of the two compartments. Under the current centered rule, a KC trace
followed by a positive DAN gain is expected to depress eligible edges in the
corresponding compartment for either label; it is not valid to preregister that
PAM11 and PPL101 must produce opposite weight signs when their gain rows are
both non-negative. The primary reward-versus-aversive checks are therefore:

* each branch has a nonzero, compartment-localized signed delta on the fixed
  cue, or is explicitly reported `INCONCLUSIVE_NO_ELIGIBILITY` if the fixed cue
  produced no eligible KC trace;
* the reward and aversive branch deltas are separable by their declared
  compartment and are not identical cross-wired updates;
* at 60 s, the absolute median delta over affected edges is at least 10% of its
  1 s value, with no edge outside the declared `0.1--2.0x` efficacy bounds.

The pre-registered true opposite-sign control is temporal, not semantic:
using the same checkpoint, same KC cue, same DAN group, and same pulse magnitude,
one branch presents KC activity before DAN and the other DAN before KC. The
anti-Hebbian rule predicts depression for KC→DAN and potentiation for DAN→KC on
the same eligible edge set. Pass requires opposite median signed deltas, both
nonzero, at 1 s and retained at 60 s. A failure is a mechanism failure, not a
reason to change the graph, gain, pulse, or decoder. The control cannot be
relabelled as a trading result.

This temporal control is marked `mechanism-only`: it delivers the fixed DAN
pulse through the assay's low-level neural path and presents the KC cue with
the assay learning path so the timing hypothesis is testable. It does not call
production `reinforce_linked`/`observe`, and its matched-`none` branch uses the
same low-level cue path without DAN stimulation. A production linked call with
no eligibility remains `INCONCLUSIVE_NO_ELIGIBILITY`; the assay control does
not rewrite that production fact.

## Mechanism assay B: signal-level reward and eligibility trace

For every eligible 2023 signal, the following causal chain is persisted:

`signal_id -> signal_date -> checkpoint/input -> execution_date -> intent_ids ->
intent outcomes -> settled reward -> reinforcement delivered -> eligibility /
weight snapshot`.

The reward is computed only after the exact execution/mark boundary settles. It
must never use the signal-day close or a later signal's outcome. A settled
reward is attached exactly once to its own `signal_id`, before the documented
observation at which that reward is delivered, and the event records the
reinforcement label passed to the controller. No reward may be attached to an
unrelated signal, and no missing reward may be silently filled from a later
date.

The audit exports a deterministic digest of the corresponding KC eligibility
trace vector immediately before delivery, the DAN spike counts during delivery,
and the plastic-weight digest immediately after delivery. The primary
trace-entry test compares the linked signal's trace/gain entries with a matched
same-checkpoint `none` control: the reward/aversive label must be consumed by
the DAN-to-plasticity update for the linked trace entries, while unrelated
signal entries remain unchanged. A matched no-reward control uses the same
checkpoint/input and explicitly `none`; it must not mutate plastic weights. A
reward/aversive signal may be recorded for an empty or vetoed execution outcome
as an execution fact, but the report must not claim that an unfilled action
caused the portfolio change.

Pass thresholds are exact linkage, not a return threshold: 100% of resolved
links have one and only one reward record when the arm requires attribution,
100% of linked rewards have a corresponding non-null trace/gain audit entry,
zero reward-to-signal mismatches, zero duplicate signal IDs, and zero weight or
trace changes in the matched `none` control. Any orphan, duplicate, cross-date
reward, missing corresponding trace entry, or unexplained trace mutation fails
the arm and stops interpretation.

## Mechanism assay C: daily exposure transitions and explicit state resolution

V5.1 separates daily decoder exposure transitions from candidate selection. The
candidate stream still has exactly 17 selection records. Each selected target
basket and its weights remain frozen until the next hash-pinned candidate
selection date; a daily neural decision is never allowed to choose a new symbol,
renormalize the basket, or substitute an asset. The decoder nevertheless runs on
all 242 annual observation sessions. On any day, including a non-candidate day,
when `requested_multiplier != effective_multiplier`, V5.1 creates an independent
paper `exposure-rebalance` batch against the currently frozen target basket. This
batch is an exposure-state transition, not a selector output and not a new
candidate signal.

The transition batch persists every intent before the simulated send and uses the
same Decimal quantities, frozen lot/precision rules, next-session/T+1 rules,
official price/state checks, and risk guard as a candidate batch. A guard may
veto an intent only; it cannot choose a replacement symbol, side, quantity,
or multiplier. If the frozen basket is already at the requested exposure under
the exact lot/precision contract, the batch is recorded as
`ALREADY_AT_TARGET` without an order. If the decoder says `HOLD`, it retains the
current requested/effective exposure and creates no exposure-rebalance batch,
no replacement trade, and no selector action. Thus a non-candidate-day BUY or
SELL may create only a rebalance over the already-known frozen basket; it may
never create a substitute trade.

The annual run uses an explicit outcome enum for every candidate signal batch or
daily exposure-rebalance batch. Child intent statuses remain in the ledger; the
aggregate outcome is resolved as:

| outcome | exact definition | effective multiplier |
|---|---|---|
| `FILLED` | at least one intent in the candidate or daily rebalance batch is filled; mixed fill/veto retains every child status | commit the requested decoder state |
| `ALREADY_AT_TARGET` | the frozen Decimal target already equals requested exposure under the lot/precision contract, so the transition is recorded without an intent | idempotently record requested state; no order |
| `EMPTY` | a candidate or daily transition requested an exposure change but produced no intents and was not already at target | retain prior effective state |
| `VETO` | intents were prepared but every intent was rejected by a guard | retain prior effective state |

Any unknown, unresolved, or contradictory state is a hard stop, not a fifth
action. `requested_multiplier` is the decoder output at the signal. The
`effective_multiplier` is the state allowed to affect the next observation;
`EMPTY` and `VETO` cannot permanently change it. The guard may veto for cash,
lot, T+1, market-state, price, exposure, or drawdown reasons, but it may never
choose a replacement symbol, side, multiplier, or trade. No veto is converted to
`HOLD`, and no empty batch is converted to a scripted order.

The state audit must show one aggregate outcome per candidate signal or daily
transition batch, explicit linkage to the frozen candidate basket, child intent
sets that agree with the ledger, no effective-state change on `EMPTY`/`VETO`,
and an exactly reproducible state sequence on resume. It must also show that
non-candidate-day BUY/SELL batches never contain a symbol outside the currently
frozen basket and that HOLD days contain no replacement intent. Any violation
fails before annual performance numbers are opened.

## Mechanism assay D: fixed decoder prediction audit

The decoder is frozen at the existing V5 contract:

* `difference_hz = mean(right DNp20 rate) - mean(left DNp20 rate)`;
* `BUY` iff `gate_spikes >= 1` and `difference_hz >= +2 Hz`;
* `SELL` iff `gate_spikes >= 1` and `difference_hz <= -2 Hz`;
* `HOLD` otherwise.

The audit recomputes the side from persisted counts/rates and cell identities,
including exact boundary cases (`+2`, `-2`, zero gate, and values just inside
the threshold). It compares the recomputed side, rates, gate count and
threshold with every logged observation. Success requires zero mismatches,
zero non-finite values, and exact equality of the persisted spike/input hashes.
Any one mismatch, missing cell identity, threshold drift, or hand-edited side
is a decoder-audit failure.

The decoder audit covers all 242 daily observations. For the 17 eligible
candidate-selection dates, all three sides must occur and `DEFER` must be below
90%; daily non-candidate transition counts are reported separately. Otherwise
the arm is marked `quarantined_action_collapse`. This quarantine never
authorizes changing the threshold or injecting a replacement action. The annual
replay remains a mechanism/accounting record, not a successful trading claim.

## Logs, hashes, and interruption/resume parity

Before execution, the run records the preregistration hash, code/config hashes,
graph/source locks, candidate and market-manifest hashes, seed, year boundary,
decoder parameters, reward parameters, and paper-only flags. Required outputs
are the primary and paired SQLite ledgers, append-only `events.jsonl`, both
alternating Fly checkpoints, clean initial brain, provenance, and runtime
telemetry. Wall-clock timing is diagnostic only and is excluded from deterministic
state comparisons.

Every event includes the prior event hash and canonical payload hash. The
canonical payload binds date, arm, signal and execution dates, signal ID,
requested/effective state, outcome, child intents, fills/vetoes, reward linkage,
checkpoint hash, input/spike hashes, Decimal cash/equity/fees/turnover, market
price provenance, and corporate-action IDs. The SQLite event chain is
authoritative; `events.jsonl` is a materialized audit view. Any hash mismatch,
missing chain link, duplicate intent, negative cash, non-lot holding, or ghost
holding stops the run.

For each fixed arm, an uninterrupted annual run is compared with interrupted
resumes at predeclared session indices corresponding to approximately 25%, 50%,
and 75% of the 2023 calendar. The comparison must be byte-identical for clean
and day checkpoints, event JSONL, normalized primary and paired ledgers,
signal links, child outcomes, effective-state history, reward/trace snapshots,
provenance gate metadata, and final checkpoint hashes. A failed parity check
quarantines the run; it cannot be repaired by rerunning only the failed date or
by selecting a more favorable checkpoint.

## Corporate-action correctness audit

The annual replay uses the V4 point-in-time action contract unchanged. Only
implemented `DIV_PROGRESS=3` or `PROGRESS=3` records pass; announcement/effective
ordering and de-duplication by code, dates, and economic terms are hash-checked.
At the record date, end-of-day holdings create one idempotent entitlement. The
cash/stock receivable is excluded before ex-date, enters raw-mark equity from
ex-date, pays into cash only at payout, and listed whole shares become
T+1-sellable lots only at listing. An unsupported fractional stock entitlement
fails closed. Rights policy is fixed at `DECLINE_NO_SUBSCRIPTION` for both Fly
and paired base; neither side invents cash or shares.

The audit uses fixed record/ex/payout/listing boundary fixtures plus all 2023
observed action rows. It requires zero duplicate entitlements, zero pre-ex-date
value inclusion, zero payout value jumps, exact Decimal receivable/cash/stock
transitions, correct listing/T+1 sellability, identical Fly/base treatment, and
matching action IDs across resume. Missing raw open/status, suspension/ST,
limit violations, or action hash mismatches veto the affected fill or stop the
run; no future bar or qfq substitute is allowed.

## Analysis and stopping rules

Primary outputs are mechanism pass/fail and audit counts: signed plastic deltas,
trace-link coverage, state-resolution counts, decoder mismatches, event/hash
integrity, resume parity, and corporate-action discrepancies. Annual return,
drawdown, turnover, and action counts are descriptive and are opened only after
all integrity gates pass. They cannot select an arm, seed, year, checkpoint,
decoder threshold, or follow-up experiment.

The run stops on any unknown outcome, unresolved ledger state, missing or
unverified official market state, hash drift, checkpoint mismatch, failed
corporate-action invariant, live/broker flag, or non-paper output. All outputs
remain below new `runs/wc006_v5_1_v6_*` directories. No real trade is placed.
