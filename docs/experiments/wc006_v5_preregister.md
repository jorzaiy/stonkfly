# WC006 + Fly V5 preregistration

Status: immutable staged specification. This file authorizes implementation and
tests only. A later independent parity/resume/input/isolation gate may authorize
one clean-brain development run for each prespecified arm. No broker, smoke,
paper service, live/shadow change, or 2024 sealed validation is authorized.

Version: `2026-09-19.v5`

## Frozen V4 record

The completed V4 xysz raw-action run remains immutable and is not a checkpoint
source. Its independently rebuilt values were Fly final `125912.084801`,
25.912084801% cumulative return and -38.3637086272% maximum drawdown; the
paired WC006 base final was `998447.552391`, 898.447552391% cumulative return
and -32.8247679266% maximum drawdown. These values are diagnostic inputs to
this protocol, not tuning targets.

## Arms

The primary comparison is same-universe, same-date, same-price, same-cost and
same-corporate-action accounting:

* `WC006-only` is the deterministic paired base: repaired WC006 candidate and
  target-weight stream, no neural controller, multiplier 1.0.
* `WC006+Fly` is the complete retained MaleCNS graph with the same WC006
  candidates and weights, clean brain, and fixed `BUY -> EXPOSE 1.0`,
  `SELL -> DEEXPOSE 0.5`, `HOLD -> DEFER` decoder.

The existing single-ETF `510300.SH` Fly contract may be reported as a
secondary descriptive `Fly-only` arm. It uses its own pinned symbol contract,
raw execution/mark/reward prices, costs and corporate-action rules. It is not
causally comparable to the multi-symbol WC006 arms and must never use the
2021--2023 WC006 candidate union or any future selector output.

Implementation provenance has two explicit layers: `comparison_arm` is
`WC006+Fly`, with the paired ledger identified as `WC006-only`; `ablation` is
one of `attribution-only`, `state-only`, or `combined`. The descriptive
`Fly-only` contract above is not executed by this V5 protocol and must not be
represented as an observed arm. This is a provenance clarification only; it
does not add an arm, change a threshold, or alter the preregistered order.

All arms use clean brains, initial cash `100000`, 100-share lots, T+1,
identical xysz raw execution/mark/reward semantics, qfq-only observations,
actual fees/slippage/stamp duty, and the same paper-only isolation.

## V5 minimal change

The fixed decoder and risk budget remain unchanged:
`target_exposure = base_weight * 0.85 * effective_multiplier`.
V5 records a deterministic `signal_id` for every signal date and links it to
all T+1 intents, fills/vetoes, pending age and the first settled reward. A
signal is never rewarded using its signal-day close; its outcome is recorded
only after its exact execution date is settled. The reward uses only settled
Fly/base equity increments, actual fees and turnover, and drawdown. No future
label or look-ahead return is introduced. The `attribution-only` and `combined`
arms attach that first settled reward to the V5 signal link; `state-only` keeps
only the minimum execution-outcome linkage needed to resolve effective state
and retains V4 daily reinforcement, without signal-level settled-reward
attribution.

`requested_multiplier` is the decoder result at T. `effective_multiplier` is
the state used for the next observation. A requested state is committed only
after its linked execution batch has at least one fill; a wholly vetoed or
empty batch cannot permanently change effective state. This is bookkeeping of
execution outcome, not an alternative action: guards still only veto.

Each daily event persists target exposure, actual gross exposure, guard reason,
pending age, signal outcome and reward linkage. SELL intents are settled before
BUY intents in deterministic symbol/intent order. Both ledgers use the same
price, corporate-action, cost, T+1 and guard rules.

## Prespecified ablations and gates

After the independent gate, run exactly the following clean-brain arms in the
listed order, with fixed seeds recorded before execution:

1. `attribution-only`: V5 signal/fill/reward linkage; V4 effective-state
   behavior retained.
2. `state-only`: V5 requested/effective multiplier outcome handling; V4 reward
   linkage retained.
3. `combined`: both changes.

No seed, arm, or hyperparameter may be selected from development returns.
Every arm must pass interruption/resume parity for daily neural decisions,
signal links, fills/vetoes, rewards, both ledgers, event chain and checkpoint
bytes before any metric is opened. Action collapse is quarantined rather than
fixed by changing the decoder. Validation 2024 and exam 2025 remain sealed.

All code, manifests, checkpoints, logs and ledgers are written only below new
`stonkfly-lab/runs/wc006_v5_*` directories. The trading repository, services,
live and shadow state are read-only and out of scope.
