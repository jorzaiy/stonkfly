# MBON Plasticity Rule Diagnosis for Opus 5.5

> **SUPERSEDED (2026-10-06).** This is a historical diagnosis and its proposed next steps are no longer authoritative. See [20261006-s2-diagnosis-for-opus.md](20261006-s2-diagnosis-for-opus.md) for the S1–S2j evidence chain, corrected artifacts, current sensory/KC blocker, and questions for Opus.

## Executive Summary

The current plasticity rule causes **both MBON pathways to change in the same direction**, making it impossible for the decoder to distinguish reward vs aversive learning. After 1699 days of training, MBON07/11 are nearly silent (active only 1-6 days out of 483), and all alpha comes from the stock selector, not MBON learning.

## Architecture Overview

### Two-Compartment Design (circuit.py)
```
Reward pathway:    PAM11 (15 neurons) → KC → MBON07 (4 neurons)
Aversive pathway:  PPL101 (2 neurons) → KC → MBON11 (2 neurons)
```

- **7835 plastic edges**: All KC→MBON07 and KC→MBON11 connections
- **gain matrix** (17×7835): Correctly routes each DAN to its target MBON edges only
  - PAM11 neurons only affect edges targeting MBON07
  - PPL101 neurons only affect edges targeting MBON11

### Decoder Logic (controller.py:126-179)
```python
ratio = mbon07_hz / mbon11_hz
if ratio > 0.15: side = "BUY"    # approach dominates
elif ratio < 0.05: side = "SELL"  # avoidance dominates
else: side = "HOLD"
```

**Critical insight**: The decoder needs MBON07 and MBON11 to move in **opposite directions** to shift the ratio. If both increase or both decrease together, the ratio stays roughly constant → always HOLD.

## Current Rule Implementation (rule.py)

### Parameters
```python
PARAMETERS = {
    "trace_kc_seconds": 1.0,
    "trace_dan_seconds": 1.0,
    "memory_decay_seconds": 900.0,
    "weight_filter_seconds": 0.05,
    "minimum_fraction": 0.3,   # w ∈ [-0.7, +1.0]
    "maximum_fraction": 2.0,
}
```

### The Problematic Update Equation (L47-59)
```python
drive = eta * (kc_hz * (gain.T @ dmid) - (gain.T @ dan_hz) * kmid)
# ...
u[:] = old_u * eu + drive * tu * (...)
w[:] = w * ew + old_u * c + drive * (...)
```

**Problem**: `drive` is computed as a single vector for all 7835 edges. Even though `gain.T` correctly routes DAN signals to their respective edges, **all edges receive qualitatively similar updates** because:

1. KC activity (`kc_hz`, `kmid`) is shared across both pathways
2. Both `u` and `w` are single arrays indexed by edge, not separated by pathway
3. When reward DAN fires: edges to MBON07 get suppressed (anti-Hebbian)
4. When aversive DAN fires: edges to MBON11 get suppressed (anti-Hebbian)
5. **Both pathways use the same anti-Hebbian rule** → both get suppressed over time

### Empirical Evidence: Training Produces Uniform Suppression

After 1699 days of training:
```
memory_w statistics:
  mean: -0.214 (78.6% of baseline)
  std:  0.166
  min:  -0.700 (touched floor)
  max:  +0.269

Edge distribution:
  44.4% suppressed (< -0.05)
  54.0% near baseline
  1.6% enhanced (> +0.05)
  904 edges at minimum floor (-0.7)
```

**Correlation between memory_u and memory_w: 1.0** — they track identically, offering no independent signal.

### Result: MBON Silence

Out of 483 test days (2024-2025):
- MBON07 active: 6 days (1.2%)
- MBON11 active: 1 day (0.2%)
- Both silent: 476 days → ratio undefined → HOLD

The +53.69% alpha comes entirely from WC006 selector picking good stocks, not from MBON learning.

## Root Cause Analysis

### Why Anti-Hebbian Suppresses Everything

The centered anti-Hebbian rule from Huang et al. 2024:
```
Δw ∝ kc_hz × dan_signal - dan_hz × kc_trace
```

In their 9-unit reduced model, this creates **opponent** dynamics because reward and aversive share a single KC population that can be differentially associated.

In our 7835-edge full connectome:
- Each KC connects to BOTH MBON07 and MBON11
- When a KC is active during reward, its edges to MBON07 get suppressed
- When the same KC is active during punishment, its edges to MBON11 get suppressed
- **Net effect**: All edges get suppressed, just at different rates

### The Missing Ingredient: Opponent Learning

For the ratio decoder to work, we need:
- Reward → strengthen MBON07 pathway OR weaken MBON11 pathway
- Punishment → strengthen MBON11 pathway OR weaken MBON07 pathway

Current rule only does:
- Reward DAN active → weaken MBON07 edges (anti-Hebbian)
- Aversive DAN active → weaken MBON11 edges (anti-Hebbian)

## Constraints for Redesign

### Must Preserve
1. **Full MaleCNS v1.0 graph**: No pruning of the 7835 edges
2. **Anatomical gain routing**: PAM11→MBON07, PPL101→MBON11 contact fractions
3. **No LLM policy injection**: Plasticity rule must be mechanistic, not learned
4. **Separation of concerns**: Risk guard can reject orders, never choose replacements
5. **Bounds**: Weight fraction must stay in [minimum_fraction, maximum_fraction]

### Biological Plausibility Considerations
- Dopamine in mushroom body: reward and aversive compartments are anatomically separate
- Anti-Hebbian depression is well-documented for KC→MBON synapses
- But opponent dynamics require **cross-compartment inhibition** or **sign-reversed learning**

## Proposed Directions (for Opus 5.5 to evaluate)

### Option A: Sign-Reversed Cross-Compartment Rule
```python
# When reward DAN fires:
drive_mbon07 = -eta * (...)  # suppress (existing)
drive_mbon11 = +eta * (...)  # potentiate (new: disinhibit avoidance)

# When aversive DAN fires:
drive_mbon11 = -eta * (...)  # suppress (existing)
drive_mbon07 = +eta * (...)  # potentiate (new: disinhibit approach)
```

### Option B: Separate u/w States Per Pathway
```python
u_reward = np.zeros(n_reward_edges)  # KC→MBON07
u_aversive = np.zeros(n_aversive_edges)  # KC→MBON11
w_reward = np.zeros(n_reward_edges)
w_aversive = np.zeros(n_aversive_edges)
```
Then apply different learning rates or signs to each.

### Option C: Asymmetric Trace Time Constants
Make reward pathway learn faster than aversive (or vice versa) to create temporal asymmetry in the ratio.

### Option D: Threshold-Gated Learning
Only update weights when the MBON output crosses a threshold, preventing runaway suppression.

## Files to Modify

| File | Current State | Needed Change |
|------|--------------|---------------|
| `stonkfly/neural/rule.py` | Single drive vector, shared u/w | Separate pathway logic |
| `stonkfly/neural/brain.py` | `memory_u`, `memory_w` as flat arrays | May need pathway indexing |
| `stonkfly/neural/circuit.py` | Has `reward_mb`, `aversive_mb` indices | Use these to split updates |

## Test Criteria for New Rule

1. **Divergent pathway response**: After training, MBON07 and MBON11 edge weights should have **negative correlation** (or at least low positive correlation)
2. **Active MBON output**: Both MBONs should fire on >10% of test days
3. **Ratio variability**: The ratio should span the decision thresholds (0.05, 0.15) regularly
4. **Learning signal**: Clean brain vs trained brain should produce different action distributions

## Appendix: Key Code References

- `circuit.py:31-45`: gain matrix construction
- `rule.py:47-59`: drive computation and weight update
- `brain.py:345-357`: how memory_u/w are passed to rule and applied
- `controller.py:162-179`: ratio decoder decision logic
