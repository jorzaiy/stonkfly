# Fly 2024–2025 purged paper continuation

Frozen on 2026-09-27, before the 2024–2025 raw market bundle or replay returns
were available.

- Selector: continue the frozen purged WC006 V3 mirror and state from the
  2021–2023 checkpoint; use its 36 scheduled signal dates in 2024–2025. Do
  not use future prices or replay profit to change target stocks or dates.
- Observation: the existing pinned QFQ cache, solely for causal sensory
  observations. Execution prices, marks, and rewards use XYSZ raw Kline;
  official stock state and point in time corporate actions come from XYSZ.
  Query Kline/status from 2024-01-01 through 2026-01-02 and corporate action
  announcements from 2023-01-01 through 2026-01-02, retaining pre-window
  announcements that can affect in-window holdings. The wider retrieval end
  is for settlement coverage; observations remain causal by replay date.
  A suspended holding may use its latest earlier raw close as a settlement
  mark, with official settlement-day status present; it is never given a
  fabricated execution price.
- Fly: full retained MaleCNS v1.0 graph, fixed v5.1 decoder, seed 17, clean
  initial brain, frozen plasticity, RGB only NullSensoryAdapter, same paper
  execution limits and fees as the 2021–2023 purged baseline.
- Base: independently hold the selector's target basket at fixed multiplier
  1.0, with its own ledger. Share market data, costs and risk vetoes with Fly.
- Window: first signal 2024-01-02, last signal 2025-12-29, first next session
  execution, and settlement mark on 2025-12-30. Review after cost returns of
  both Fly and Base plus their difference, drawdown, turnover, order vetoes,
  retries, and accounting reconciliation. No seed or factor tuning on this
  window.
- Desired practical gate: Fly net return at least zero. A negative result is
  reported as a failed gate; no alternative trade or seed is substituted.
- Research only: no live or shadow authorization, no broker order route.
  Profit/loss reinforcement is an engineered input to identified dopamine
  cells. This experiment does not validate profitable learning or experience.

## Dividend target policy frozen before replay

The XYSZ raw dividend evidence contains one 2025-12-23 announcement for
000615.SZ with two implemented rows. The ordinary shareholder row has
`DIV_TARGET=普通股股东` and `DVD_PER_SHARE_STK=0.1`; the restructuring-manager
row has `DIV_TARGET=重整管理人` and `DVD_PER_SHARE_STK=3.91097858`, with other
source fields also differing. The parser retains both source rows through
their hash-bound raw artifact, skips the restructuring-manager row for
ordinary-holder entitlement, and keeps the ordinary shareholder row. A
nonempty unknown `DIV_TARGET` fails closed; an empty target remains compatible
with older fixtures. This policy and evidence were frozen before any replay or
return inspection.
