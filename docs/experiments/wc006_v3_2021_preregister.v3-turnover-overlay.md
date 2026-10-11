# WC006 + complete MaleCNS runtime: 2021--2025 preregistration v3 addendum

Status: preregistered experiment-only implementation addendum. This file does
not modify the frozen v2 specification and does not authorize training,
replay, return analysis, broker orders, or service actions.
Version: `2026-09-18.v3-turnover-overlay`

## Scope and frozen v2 relationship

The v2 preregistration remains the governing primary-arm specification. This
addendum defines one isolated selector-runtime overlay required to prevent
ineligible prior holdings from entering a WC006 target artifact. It is applied
only to a hash-verified Stonkfly mirror of the copied `walkforward.py`; the
trading repository source is never edited. The mirror records the original
source SHA, patched source SHA, overlay patch ID, and overlay content hash.

Frozen v2 SHA-256 at addendum creation: `71dde295a8c3a20f8bfeb713cce50e8633eba66786f1b373d90d3a3e95f895a9`.

## Experiment-only turnover-cap semantics

For `apply_turnover_cap(desired, prev, topk, cap)` with a prior set, the
number of retained positions is `keep_n = round(topk * (1 - cap))`. Retained
positions are drawn only from `desired ∩ prev`, preserving `desired` order.
Prior codes outside the current eligible `desired` list are never inserted as
fallbacks. If fewer than `keep_n` eligible prior codes exist, the remainder is
filled from `desired` in its existing order until `topk` is reached.

Therefore, whenever at least `topk` eligible candidates exist, `picks` has
exactly `topk` unique codes and `set(picks) ⊆ set(desired) ⊆ codes(g)`. The
weighted selection must have exactly the same instrument codes as `picks`,
finite strictly positive weights, and weights summing to one. The exporter
must emit non-null `score` and `target_weight` for every selected target.

A prior position excluded by the current candidate/eligibility set is a
`forced turnover due to ineligible prior holding`. It is not retained merely
to claim compliance with a turnover cap, and no performance or economic
conclusion may be drawn from this overlay itself.

## Overlay identity and validation

The mirror applies exactly one textual replacement in `walkforward.py`:

- patch ID: `eligible-prev-turnover-cap-v1`;
- the replacement must match exactly once;
- the original and patched SHA-256 values and replacement content hash are
  recorded in mirror provenance;
- the existing `none-safe-target-topk-render-v1` presentation overlay remains
  separate and is also recorded;
- selector, target JSON, feature, model, calendar, candidate, and price input
  semantics are otherwise unchanged.

Focused regression tests cover ghost-code exclusion, complete top-k selection,
weighted-selection/index parity, finite positive normalized weights, and three
exporter targets with non-null score/weight. The three-date causal slice uses
only the fixed dates `2021-07-01`, `2021-07-29`, and `2021-08-02`, one exact
canonical cache tag, and remains paper-only and outside the trading repo.
