# Fly next evaluation gate (2026-09-27)

This is a work order, not a new trading policy or authorization. The user goal
is nonnegative net paper return. It cannot be guaranteed or established by
selecting a favorable seed or action rule on the already inspected 2021–2023
period. Keep the full MaleCNS graph, fixed decoder, veto-only guard and
paper-only execution unchanged while preparing the next evaluation.

## Current evidence

- Purged frozen Fly: -18.93%; paired Base: -7.80% over 2021-07-01 to
  2023-12-29 plus settlement.
- Three prespecified permuted-action nulls: -7.50%, -14.99%, +1.62%.
  Selecting the positive seed after seeing its return is invalid.
- Independent Base-with-retry: -35.73%; original-action and no-retry Base
  execution parity both match exactly.
- No completed purged WC006 candidate stream beyond 2023 is present. The
  2024 single ETF regression is a different experiment.

## Next work, in order

1. Finish the selector-only benchmark and paired path analysis on the current
   frozen stream. Report costs, exposure, concentration, and uncertainty
   without changing the candidate, decoder, or risk guard. Keep all trial
   outcomes in the record.
2. Build a new causally purged 2024–2025 WC006 candidate stream with its own
   hash-pinned QFQ observation provenance. Acquire and audit complete XYSZ raw
   execution, status, limits and point-in-time corporate actions for its
   selected symbols. Abort if coverage or date-boundary checks fail.
3. Before viewing 2024–2025 returns, freeze one Fly configuration and the
   paired Base/null comparison. Primary acceptance gate: Fly net paper return
   after the declared costs is greater than or equal to zero on the full
   evaluation window. Report Fly-minus-Base, drawdown, turnover and vetoes as
   separate diagnostics. A positive return on a cherry-picked subwindow is
   not a pass.
4. If the gate fails, report the failure. Any mechanism change or enabled
   plasticity run needs a separately specified experiment and a later unused
   evaluation window. Do not turn a paper control, seed, or risk veto into a
   hidden profit-based action selector.

This gate makes a nonnegative result meaningful if it occurs; it cannot make
the market deliver one.

## 2026-09-27 status update

Steps 2–3 above are complete for the 2024–2025 purged heldout paper run.
The prespecified nonnegative gate passed: Fly +13.965770098% versus paired
Base -1.8303796415%, with Fly maximum drawdown 47.7090102201%.
See [the heldout result](fly_2024_2025_heldout_result.md) for input locks,
audited outputs, annual figures and interpretation. The earlier statement
that no post-2023 candidate stream exists is historical context only.

Next: read-only attribution of the frozen result, including monthly and
position-level Fly–Base differences, exposure, costs and drawdown path.
Any subsequent paper evaluation needs a new protocol and unused window;
this observed result cannot be used to choose a seed, decoder or trade rule.
