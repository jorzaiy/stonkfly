# WC006 + v4 xysz price/action addendum

Status: immutable implementation addendum. This document does not authorize a
training run by itself. A separate input/parity/resume gate remains required.

The observation frame continues to use the pinned candidate-stream qfq cache.
Execution, fills, marks, equity and reinforcement use only the hash-pinned
xysz raw daily Kline in the C-stage parent manifest
(`ca925aa8d01c947da9eb71195b80bceae7d8a26b8d67cd1327dfdf1442e19a69`) and QA
(`4a1d4fa06c0578a73099a4954a7479e734502cac675ddaf70144c3fad8b44c4b`). A raw
execution bar must also have an official status row. Suspended rows veto fills;
ST rows veto fills; official limit values veto a slippage-adjusted fill outside
the limit. Missing raw open or status is `missing_open_or_unverified_suspension`.
Missing raw close is marked with the last valid raw close and provenance; it is
never replaced with a qfq value.

## Point-in-time corporate actions

Only implemented records with `DIV_PROGRESS=3` or `PROGRESS=3` are eligible (the
collected schemas do not expose a separate `IS_VALID` field), and
the announcement date must be no later than the applicable effective date
(the latest available execution/ex-date/listing date for rights, and the
record/ex-date/payout/listing chain for dividends). Records are de-duplicated by a hash of code,
dates and economic terms; raw files remain preserved and no silent overwrite is
allowed.

At exactly the record date, end-of-day holdings create a cash/stock
entitlement once (resume-safe and idempotent). The entitlement is recorded but
is not included in equity until the ex-date, because the record-date raw close
still carries the right. From ex-date onward, cash is a pre-tax receivable until
payout (then transfers to ledger cash without a value jump), and stock remains a
non-tradable receivable valued at the raw mark until listing. Listed whole-share
quantities become lots with next-session sellability under A-share T+1. If the
stock entitlement is fractional and no cash-in-lieu rule is present, the run
fails closed; no permanent fractional ghost receivable is invented.

Rights issues use a fixed conservative policy shared by Fly and paired base:
`DECLINE_NO_SUBSCRIPTION`. No cash or shares are invented. The C-stage data
contains two implemented rights rows (000065.SZ and 000977.SZ); whether either
falls on an actual held position is an audit fact, not a reason to change this
policy. A later policy would need explicit cash sufficiency, subscription,
listing-date and T+1 tests for both arms.

From ex-date onward, corporate-action receivables are included in raw-mark equity
and therefore in incremental paired-base feedback. Cash uses the pre-tax source
field; personal after-tax treatment is intentionally not modeled. The decoder and risk guard never select a
replacement action. Fly and paired base receive identical action treatment,
raw prices, status gates, costs, lots and calendar.

The earlier v4 run is retained but quarantined with reason
`invalid_execution_price_semantics`; its brain is not resumable.
