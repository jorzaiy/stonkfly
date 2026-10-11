# WC006 + Fly V5.1 frozen-plasticity counterfactual v1

Status: final post-hoc development protocol. This protocol was written after inspection of the enabled 2023 result and is therefore **not prospective**. It is not an authorization artifact, not a formal run plan, and does not authorize broker, live, shadow, or funded execution.

## Post-hoc boundary and interpretation

The already observed enabled arm reported settlement uplift versus its paired base of `41.212998` in account-currency units, i.e. `4.1212998 bp` under the recorded 100,000 initial capital convention. It also showed a drawdown difference of `+1.06148 pp`, gross fees of `138.913502000`, turnover of `129746.877500`, and `3489/7835` final event plastic edges changed. Those observations motivated this counterfactual; they must not be used to select parameters, dates, inputs, output paths, or stop rules.

There are no pass, near-zero, or performance thresholds in this post-hoc protocol. The only comparison quantity, if and when a full-year run is separately authorized, is the exact frozen-versus-enabled plasticity contrast relative to each arm’s paired WC006-only base. This smoke only establishes execution, resume, provenance, and frozen-weight invariants; it computes no return or uplift.

## Fixed replay

* Year: 2023; observations 2023-01-03 through 2023-12-29 (242 sessions); 2024-01-02 is settlement-only.
* Seed: 17.
* Inputs: the existing annual candidate manifest, full source manifest, corrected raw market manifest, raw market QA, and corporate-action evidence named in the frozen plan JSON.
* Neural mode: `plasticity_mode=frozen`, constructed as `Settings(learning=False)` and enforced by `brain.weights_frozen=True`.
* Neural contract: complete retained MaleCNS v1.0 graph, all sensory and engineered reinforcement propagation, eligibility/DAN records, and the existing fixed decoder.
* Execution: paper-only; `formal=false`, `broker_orders_enabled=false`, `live_execution_authorized=false`, Decimal ledger and existing risk/intent rules.
* Output paths: resumed A=`runs/wc006_v5_1_v6_development/annual-2023-frozen-smoke-resumed-v1`; uninterrupted B=`runs/wc006_v5_1_v6_development/annual-2023-frozen-smoke-uninterrupted-v1`; parity=`runs/wc006_v5_1_v6_development/annual-2023-frozen-smoke-5d-parity-v1`.
* Stop points: day 61, 121, 182, and 242 for any future staged annual run. The current smoke is limited to five calendar sessions and uses a day-1 pause/resume boundary.

The resumed smoke is run for one day, stopped, then resumed to five days. The uninterrupted smoke runs the same five days fresh. Both are independent driver invocations with their own training logs.

## Frozen invariants

For every checkpoint, including clean and each daily checkpoint, the full weight array and the 7,835-edge plastic subset must have identical hashes to clean; `changed_edges` must remain zero. Linked reward/aversive delivery must remain in the V5.1 path and retain auditable eligibility, DAN, pulse, and reward-linkage fields. `INCONCLUSIVE_NO_ELIGIBILITY` is recorded as such and is not converted to a pass or a zero effect.

Parity requires zero unknown/pending outcomes; byte-identical events; normalized primary and paired ledgers; equal checkpoint content/hashes; identical runtime identity fields (`arm`, `seed`, `date`, `index`), excluding wall-clock compute telemetry; and provenance equal except output path. Any failed check stops interpretation.

## Prohibited interpretation

This is a closed-loop counterfactual: actions can change exposure, drawdown, subsequent frames, and engineered reinforcement. It is not a pure off-policy weight-only causal estimate under identical future sensory frames. A future full-year contrast cannot establish general profitability, validated biological learning, consciousness, pain, pleasure, or live performance.

For the completed 2022 QFQ replay evidence, corrected Branch B hash chain, settlement, and usage boundary, see [V5.1 corrected Branch B 2022 baseline audit](v51_corrected_branch_b_baseline.md). Historical Branch A remains sensitivity-qualified and non-invariant for performance claims.

