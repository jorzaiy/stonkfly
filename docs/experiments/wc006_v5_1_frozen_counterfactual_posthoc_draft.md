# WC006 + Fly V5.1 frozen-plasticity counterfactual (post-hoc draft)

Status: development-only, post-hoc draft, not pinned, not formal, and not an authorization artifact. This document must not be used to authorize a broker, live, shadow, or funded run.

## Question

Using the already trusted 2023 V5.1 paper replay inputs, does allowing the declared plasticity mechanism improve or harm the fixed decoder relative to a counterfactual in which plastic efficacies remain frozen?

## Fixed protocol

The counterfactual uses exactly the existing 242 observation sessions from 2023-01-03 through 2023-12-29, the existing hash-pinned candidate stream, raw execution/mark prices, corporate-action rules, seed 17, paper execution, clean brain, complete retained MaleCNS v1.0 graph, and the existing fixed decoder. 2024-01-02 remains settlement-only. No input, threshold, candidate, cost, risk, decoder, graph, or output-path selection may change after observing results.

The enabled arm uses `Settings(learning=True)`. The frozen arm uses `Settings(learning=False)`, which sets `brain.weights_frozen=True`. Both arms still propagate sensory input, KC eligibility, and linked PAM11/PPL101 reinforcement pulses. The frozen arm must have zero changes to the full weight array and to the 7,835-edge plastic subset.

The frozen run is labelled `development-only`, `paper-only`, and `fixed-decoder-frozen-plasticity-counterfactual`. Its input, graph, market, adapter, and mode hashes are recorded by the run provenance when available; no undetermined hash is asserted here.

## Primary calculation

Let `U_enabled` and `U_frozen` be each Fly arm's net uplift versus its paired deterministic WC006-only base. The preregistered contrast is `U_enabled - U_frozen`, reported in basis points with raw equity, costs, action counts, reward deliveries, eligibility counts, and weight-hash diagnostics.

A positive contrast is consistent with plasticity improving the fixed decoder on this fixed closed-loop replay; a negative contrast is consistent with harm; a near-zero contrast is inconclusive. A missing eligibility path remains `INCONCLUSIVE_NO_ELIGIBILITY`, not evidence of zero effect. This single-year, single-seed paper result is not evidence of general profitability, validated learning, biological replication, consciousness, pain, pleasure, or live performance.

Because actions alter exposure, drawdown, subsequent frames, and engineered reinforcement, this is a closed-loop plasticity ablation. It is not a claim of a pure off-policy weight-only causal effect under identical later sensory frames.

## Required gates

* The frozen run's mode is present in the input gate, signature, provenance, CLI result, and resume checkpoint validation.
* The full graph remains retained; no graph crop or scripted action path is introduced.
* Every frozen checkpoint and linked reinforcement record has unchanged full-weight and 7,835-edge hashes; linked pulse/DAN and eligibility records remain auditable.
* Uninterrupted and split/resumed frozen runs have identical canonical events, ledger state, intent linkage, and checkpoint state, apart from explicitly non-hash-bound wall-clock telemetry.
* Formal mode remains blocked until a separate V5.1 authorization artifact and mechanism pass exist.
