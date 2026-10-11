# Fly/Base Purged Baseline — Phase 1 preregistration

Status: `PASS_RESEARCH_ONLY` input protocol; paper/research only. This document
does not issue or renew a V5/V5.1 authorization and does not permit live or
shadow execution.

## Frozen inputs and scope

- Candidate stream: `wc006-candidate-stream/v1`, SHA256
  `9a07849c4b910390a949934f3329bad7288142cce2acbea7ed3835bd5f77d7bc`.
- Market parent: `wc006-purged-xysz-market/v2`, SHA256
  `bebf4922faef670d665da46d8aece75665a0ec090426997172a71ea26f33c39c`.
- Market QA: `wc006-purged-xysz-market/quality-report/v2`, SHA256
  `4fb438b60da507b443639746fdc11224da14239c5dedc0a8d4977ee358d267fe`;
  both market and QA must remain `PASS_RESEARCH_ONLY`.
- Signal window: `2021-07-01..2023-12-29`; execution/settlement uses the
  first next valid session, with boundary `2024-01-02`.
- Candidate generation provenance is mandatory and hash-checked. The stream
  contains 46 strictly ordered records and a 131-code target union.
- Seed: `17`; clean-start brain; no historical checkpoint restore; RGB-only
  observations via `NullSensoryAdapter`; plasticity is `frozen`.

Execution, valuation and reward-return marks use only the pinned raw XYSZ
bundle (Kline, official status and PIT corporate-action evidence). QFQ is not
an execution source. Decimal order quantities and the existing paper ledger
intent-before-settlement path remain in force; an execution guard can only veto.

## Paired protocol

Fly and Base use the same candidate stream, calendar, raw market bundle, cost
configuration, risk guards, corporate-action handling and retry semantics.
Fly uses the complete retained MaleCNS v1.0 graph, fixed decoder and frozen
plasticity. Base is a separate deterministic zero-hypothesis ledger: on each
candidate date it independently holds that date's frozen target basket at
multiplier `1.0`; it emits no neural or selector output and never reads Fly
actions, rewards, state or fills. The Base result is not neural reinforcement
or evidence of profitable learning.

The output must live under the new
`runs/wc006_purged_fly_base_baseline_phase1/` namespace and contain independent
`ledger.sqlite` and `paired_base.sqlite` files plus hash-pinned provenance and
report. Resume is allowed only with identical input hashes, seed and frozen
plasticity mode. Any failed gate, unknown outcome or hash drift stops the run.

Reports may describe engineered reinforcement delivered to identified dopamine
cells. They must not claim consciousness, pain, pleasure, or validated
profitable learning.
