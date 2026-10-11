# Purged Fly execution controls — frozen protocol

Status: paper research only. Frozen before running these controls on 2026-09-27.
The primary Fly and paired Base results are already known, so these controls
are diagnostic and cannot create a prospective performance claim.

## Inputs and boundaries

- Use the exact candidate and XYSZ market pins in
  `fly_purged_baseline_phase1_preregister.md`: candidate
  `9a07849c4b910390a949934f3329bad7288142cce2acbea7ed3835bd5f77d7bc`,
  market `bebf4922faef670d665da46d8aece75665a0ec090426997172a71ea26f33c39c`,
  QA `4fb438b60da507b443639746fdc11224da14239c5dedc0a8d4977ee358d267fe`.
- Dates: 2021-07-01 through 2023-12-29, next-session settlement 2024-01-02.
- Initial cash 100000 CNY, no positions. Same 85% budget, costs, 100-share
  lots, T+1, raw open fills and raw close marks. Every intent is persisted
  before settlement. Unknown outcomes halt the run.
- Write independent SQLite ledgers under `runs/wc006_purged_controls/`. Do not
  modify primary Fly/Base ledgers. No broker access or model learning.

## Base-with-retry

First run a no-retry Base parity arm using the same independent ledger code and
require exact final equity agreement with the frozen paired Base. A mismatch
blocks interpretation of all control returns.

On each candidate date, target the frozen basket at multiplier 1.0 exactly as
the original Base. If an intent is vetoed at the next open, keep that symbol's
target until the next candidate date and retry it at the next close for the
following open. Recompute only its quantity from the control ledger's current
equity, holdings and raw close. Stop retrying when that symbol has a fill,
becomes less than one lot away from target, or a new candidate basket arrives.
A new veto can trigger another retry. Never choose a different symbol or use
Fly actions, fills, rewards, or outcomes. Record original veto ID/reason and
each retry result. This is a separate deterministic execution control.

## Permuted-action nulls

Read only the 609 fixed decoder action labels from the primary frozen Fly
event file (SHA256
`5bc2430aae1384d0cfbe1dcf9a969faecfafb88ff485b1ac2ff1089c1f43454b`).
For each seed in `[17, 23, 43]`, sort source action indices by
`SHA256("wc006-null-v1:{seed}:{index}")`, then assign that ordered list of
labels to chronological dates. This preserves the counts of EXPOSE, DEEXPOSE
and DEFER and changes their dates without inspecting returns. At each date,
use the v5.1 fixed multiplier mapping: EXPOSE=1.0, DEEXPOSE=0.5,
DEFER=retain effective multiplier. On candidate dates, target the new basket;
on other dates, rebalance only when a non-DEFER action requests a different
effective multiplier. Effective multiplier advances only after a fill or an
already-at-target result. Vetoes never choose replacement trades.

These are external null policies, not neural output. The original MaleCNS
graph remains untouched and is not run for these controls. Compare each
control's settled return with the original Base and Fly descriptively; report
all three seeds. Do not select a winning seed or treat this small null set as
a statistical significance test.
