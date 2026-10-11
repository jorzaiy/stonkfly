# Stonkfly

- 默认使用中文回复，除非用户明确要求使用其他语言。
- 子进程（subagent）除非特别声明，否则一律默认使用 sonnet 4.6 模型。
- Preserve the full retained MaleCNS v1.0 graph. No pruning, scripted trades presented as neural output, LLM trading policy, or hidden profit-based action selection.
- Separate market observations, sensory proxies, neural propagation, plasticity, fixed decoding, and execution limits. The risk guard may reject an order; it must never choose a replacement trade.
- Profit/loss reinforcement is an engineered input to identified dopamine cells. Do not claim modeled pain, pleasure, consciousness, or validated profitable learning.
- Paper execution is the default. Never place real trades while testing. Credentials, account identifiers, balances, logs, checkpoints and data stay ignored.
- Use Decimal quantities and persist order intent before sending it. Unknown exchange outcomes stop execution until reconciliation; never blindly retry a trade.
- Keep README short. Detailed model and execution caveats belong in docs.
- The primary agent owns protocol interpretation, code changes, test design, end-to-end artifact validation, and final result interpretation. Do not delegate these responsibilities solely to a subagent.
- Luna delegation is optional, not mandatory. Use Luna only for mechanical large-data execution after the primary agent has reviewed and frozen the runner and a small end-to-end CLI smoke test has successfully produced every required artifact (including JSON, report, and manifest).
- The primary agent must independently inspect the process state, logs, artifact completeness, hashes, gates, and verdict before accepting any delegated run. Smaller or protocol-sensitive computation may be run by the primary agent or another suitable worker when that is more reliable.
- Run long formal computations in tmux. Preserve failed or incomplete run directories as evidence; never overwrite or present them as valid results.
