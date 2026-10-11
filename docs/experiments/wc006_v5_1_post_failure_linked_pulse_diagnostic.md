# WC006 V5.1 post-failure linked-pulse diagnostic

Status: bounded, post-failure, non-authorizing diagnostic. This document does
not authorize annual training, market replay, paper execution, broker access,
or a formal mechanism pass.

## Fixed input and scope

The sole input is the sequential sensory-only coverage endpoint
`runs/wc006_v5_1_eligibility_coverage/annual-2023-sensory-only-v1/brain-end.npz`,
SHA-256
`b3bcc48d6b90190c5f6e2fd4fc00b1b613038a87e652916c1eba40428c64d269`.
It is the fixed 2023-12-29 endpoint and retains 7,835 plastic edges, with
reward eligibility 1,578 and aversive eligibility 2,044 before branching.
The coverage endpoint is known to have unchanged clean plastic weights; this
diagnostic does not add another RGB observation.

Three branches (`reward`, `aversive`, `none`) are copied byte-for-byte from
that same checkpoint. Each branch makes exactly one call to the production
`V51Controller.reinforce_linked(label, linkage_id)` method. The diagnostic
code itself makes no `observe`, `rgb_step`, or direct `MemoryBrain.step` call;
the only permitted neural integration is the `brain.step` performed internally
by production `reinforce_linked` for an applied pulse. No market input, order,
or execution path is allowed. The `none` call is an explicit no-op control.

## Recorded evidence

Before and after the call, the diagnostic records all 7,835 edge weights,
eligibility traces, DAN rates/counts, selected gain rows, and checkpoint
hashes. It records total changed edges, eligible changed edges, changes
outside the selected label's gain compartment, pulse status, linkage ID, and
neural clock. Branch events use a canonical append-only SHA-256 chain.

The 60-second and 1,800-second retention values are analytic projections from
the post-pulse snapshot using the existing
`rule.advance.zero_drive_closed_form` formula. A 1-second projection is also
recorded solely for the preregistered retention comparison. These projections
advance only the declared passive `memory_u`/`memory_w` and weight-filter
state; they do not claim full-network spikes or advance the neural clock.

## Fixed interpretation thresholds

The inherited mechanism thresholds are descriptive checks only: selected
reward and aversive branches must have nonzero eligible changes unless the
fixed endpoint has no eligibility; changes must be localized to the declared
gain compartment; reward and aversive deltas must be separable; and all
weight/baseline ratios remain within the declared `0.1--2.0x` efficacy bounds.
The `none` branch must have zero changed edges and zero clock advance.

Any result is a production-linked-pulse mechanism observation at one warmed
endpoint. It is not evidence of profitable learning, a market policy, pain,
pleasure, consciousness, biological replication, or formal authorization.
