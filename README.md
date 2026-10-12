![Stonkfly: a pixel fly beside a candlestick chart](assets/stonkfly.png)

# Stonkfly

A fly-connectome simulation that can operate a crypto trading account. Actual neural output, actual Coinbase integration. Profitable learning has not been demonstrated.

> **Research archive (2026-10-10):** the MaleCNS v1 financial-learning route is closed after preregistered visual and olfactory qualification found no stable, sparse neural working regime. Historical execution code remains for reproducibility; it is not evidence of learned or profitable trading. See the [final archive](docs/20261010-male-cns-v1-final-archive.md).

> **Mushroom Body Lab v2 (2026-10-12):** synthetic-odor representation and engineered association experiments, with no market or trading input. Formal NO-GO results and the lightweight data package are summarized in the [v2 experiment archive](docs/STONKFLY_V2_EXPERIMENT_SUMMARY.md).

**How it works:** Public Coinbase prices become an RGB chart. It stimulates 3,335 brightness inputs and 811 R8 color inputs in the retained **MaleCNS v1.0 graph: 166,700 neurons, 25.6 million connections**. A fixed neural readout proposes buy, sell or hold. A custom **Coinbase AgentKit ActionProvider** checks limits and places spot orders through Coinbase Advanced.

Positive portfolio P&L stimulates 15 identified PAM11 dopamine cells; negative P&L stimulates two PPL101 aversive dopamine cells. A candidate memory rule changes existing KC-to-MBON connections. These are engineered reinforcement signals, **not modeled pain receptors**. Synaptic changes do not establish that it learns to trade profitably. [Model and evidence](docs/model.md).

## Run it

Python 3.11, a C++17 compiler, macOS/Linux. Allow several GB for the dataset and dependencies; 16 GB RAM recommended.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
python -m stonkfly prepare
python -m stonkfly run
```

Default: **paper trades, real public BTC-USDC data, $100 simulated balance**. No key needed. Local logs, sensory images and resumable brain state go in `runs/paper/`. Ctrl-C stops it; the same command resumes.

For real orders, first create a dedicated Coinbase Advanced portfolio with **at most 100 USDC** and a portfolio-scoped **ECDSA API key with View + Trade, no Transfer**. Copy `.env.example` to `.env`, fill it in locally, then run these commands yourself:

```sh
python -m stonkfly run --live --preflight-only
python -m stonkfly run --live
```

Defaults: $10 maximum order including reserved fees, 24 attempts/day, no shorts or leverage. A $20 drawdown stops new orders; **it does not liquidate holdings or cap further losses**. [Operation and recovery](docs/operations.md).

```sh
python -m stonkfly status
python -m pytest -q
```

The repo does not come funded or connected to anyone’s account. Live execution needs your local credentials and explicit opt-in.

### v3 replay (isolated)

`experiments/a_share/replay_510300_v3.py` is an experimental, paper-only successor to v2. It preserves the 85% exposure and T+1 execution guards, adds an inclusive three-trading-day reversal cooldown, and logs Decimal action-advantage attribution only for FILLED executions. Each RGB portfolio frame and event records close-time buy capacity/equity, sellable value/equity, last fill side, clipped days-since-fill, and the current execution's advantage sign/ratio. Configuration is hashed into the run signature and provenance. It is not a profitability claim and must be evaluated separately from the retained v2 runs.
