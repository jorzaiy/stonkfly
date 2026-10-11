# Stonkfly 510300 replay

This is an offline paper experiment. It imports the unmodified Stonkfly neural
controller and reads the existing quant archive without writing to it.

Timing is deliberately causal: the fly sees the close on T and its pending
BUY/SELL/HOLD decision can only execute at the open on T+1. The simulation uses
raw prices, cash dividends, 100-share buy lots, T+1 sell availability, 3 bps
commission with a CNY 5 minimum, and 5 bps modeled slippage on each fill.

These fee and slippage values are experiment assumptions, not a broker quote.

Run state and alternating brain checkpoints are written under `runs/halfyear`.
The run is resumable with the identical command.

The v2 training policy adds a fixed RGB portfolio-status strip containing
exposure, cash, unrealized return, drawdown, relative return, and order-veto
feedback. v3 (`replay_510300_v3.py`) extends that strip and every event with
close-time buy-capacity/equity, sellable-value/equity, last-fill side, clipped
days-since-fill, and current FILLED action-advantage sign/ratio. v3 also applies
an inclusive three-trading-day opposite-side cooldown before any notional
mutation and records its configuration in the signature and provenance.
Reinforcement is based on daily return relative to a causal 510300 buy-and-hold
path, adjusted for drawdown change, turnover, invalid orders, and the default
weight-1.0 action-advantage ratio.
An exposure guard rejects buys that would take the position above 85% by
default; it never substitutes another action. These are engineered inputs and
paper-risk controls, not evidence of pain, awareness, or profitable learning.

`--initial-brain` starts a new chronological interval from a saved checkpoint.
Use `--freeze-learning` for a true out-of-sample exam that cannot update the
learned weights during the test period.
