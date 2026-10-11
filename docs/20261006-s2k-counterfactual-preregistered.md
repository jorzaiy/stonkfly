# S2k counterfactual KC recurrence diagnosis (pre-registered)

Status: authorized paper-only counterfactual diagnostic; `counterfactual=true`, `production_eligible=false`.

Hypothesis: the single-compartment LIF placement of retained KC→KC signed weights may contribute to the observed low/high KC state split. This is a diagnostic counterfactual, not a production graph change, pruning claim, causal proof, or trading experiment.

Fixed input: the exact 60 event indices from S2d, source `runs/wc006_purged_fly_training/20261004T-balanced-reward-v2/events.jsonl`, events SHA256 `85ba83be96eb3b4c2cb1d3297f68c206e59b447c05354af60f77ada96dfb1cdb`; clean checkpoint `brain-clean.npz`, SHA256 `a92f547e720e7d9830ae519f26c0bacdf4f76c59761fb2be4adb6f34cac2a225`; graph `data/graph.npz`, SHA256 `346b8af85a11af13b8324e18669812c1924569e7d1adcb4e6f45cc461a2c344b`.

Arms, fixed before results: C production weights; A KC→KC effective runtime weights ×0; B2 APL→KC weights ×2; B4 APL→KC weights ×4; AB2 A+B2; AB4 A+B4. CSR topology remains intact. Overlay is in-memory per run only; graph.npz and persistent checkpoints are never modified.

Primary metrics: per-frame active KC count distribution shape, sparsity, active-set Jaccard. No reward, label, decoder, trade, P&L, or performance gate is used for arm selection. Secondary audit fields may include spike hash/count and layer counts only for equivalence and contamination checks.

Isolation gates: every arm/frame restores the same clean checkpoint; C must reproduce S2 frame hashes and KC counts; repeated trace-off/on-equivalent executions must match counts and spike hashes; overlay edge hit counts/signs/multipliers must match the precomputed masks; no cross-frame or cross-arm state may persist. Failure of any gate stops formal launch.

Interpretation is descriptive only: A-only change compatible with KC→KC placement/sign contribution; B-only change compatible with global inhibition strength; neither does not identify a cause. Results cannot validate production trading sensation or justify changing the retained graph.

Forbidden: changing graph topology, persistent weights, thresholds, dynamics, decoder/trade logic, labels, rewards, market data, or selecting an arm by outcome.
