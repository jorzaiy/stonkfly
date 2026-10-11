"""Only RGB and engineered reinforcement enter the network. No market policy."""

import hashlib

import numpy as np

from .common import annotations
from .visual import VisualMemoryBrain

# Wall-clock timing is a runtime diagnostic, not deterministic neural state.
NONDETERMINISTIC_NEURAL_FIELDS = frozenset({"compute_seconds"})


def deterministic_neural_payload(neural):
    """Remove host-dependent diagnostics before replay persistence."""
    return {key: value for key, value in neural.items() if key not in NONDETERMINISTIC_NEURAL_FIELDS}


class Decoder:
    """Original DNp20-based decoder. Note: MBON cannot influence DNp20 directly."""

    def __init__(self, ids, annotation, threshold):
        types = annotation.type.fillna("")
        sides = annotation.somaSide.fillna("")
        self.left = np.flatnonzero(types.eq("DNp20") & sides.eq("L"))
        self.right = np.flatnonzero(types.eq("DNp20") & sides.eq("R"))
        self.gate = np.flatnonzero(types.eq("DNpe017"))
        if not len(self.left) or not len(self.right) or not len(self.gate):
            raise RuntimeError("Missing annotated BCI outputs")
        self.threshold = threshold
        self.identities = {
            k: [str(ids[i]) for i in getattr(self, k)]
            for k in ["left", "right", "gate"]
        }

    def decode(self, counts, seconds):
        # Mean rates prevent side population size from creating a built-in bias.
        left = float(np.mean(counts[self.left]) / seconds)
        right = float(np.mean(counts[self.right]) / seconds)
        difference = right - left
        gate = int(counts[self.gate].sum())
        side = (
            "HOLD"
            if not gate or abs(difference) < self.threshold
            else "BUY"
            if difference > 0
            else "SELL"
        )
        return {
            "side": side,
            "left_hz": left,
            "right_hz": right,
            "difference_hz": difference,
            "gate_spikes": gate,
            "cell_ids": self.identities,
        }


class MBONDecoder:
    """MBON-based decoder that reads directly from mushroom body output neurons.
    
    Based on Aso 2014: KC-DAN pairing causes KC→MBON suppression.
    - Reward (PAM11) pairs with KC → suppresses KC→MBON07 → MBON07 decreases
    - Aversive input (PPL101) pairs with KC → suppresses KC→MBON11 → MBON11 decreases
    
    Correct decoding (P0 fix, 2026-10-02):
    - MBON11 higher than MBON07 → approach bias → BUY
    - MBON07 higher than MBON11 → avoidance bias → SELL
    
    This is counter-intuitive but follows from suppression-based learning:
    a rewarded pattern has LOWER MBON07, so we buy when MBON11 dominates.
    """

    def __init__(self, ids, annotation, threshold=2.0, normalize=False):
        types = annotation.type.fillna("")
        # MBON07 activity is avoidance-associated; MBON11 activity is
        # approach-associated. Reward/aversive name the DAN compartments.
        self.mbon07 = np.flatnonzero(types.eq("MBON07"))
        self.mbon11 = np.flatnonzero(types.eq("MBON11"))
        # Use DNpe017 as gate (same as original)
        self.gate = np.flatnonzero(types.eq("DNpe017"))
        if not len(self.mbon07) or not len(self.mbon11):
            raise RuntimeError("Missing MBON07 or MBON11 neurons")
        self.threshold = threshold
        self.normalize = normalize
        # Anatomical scaling factor to balance MBON07 vs MBON11
        # MBON11 has ~1.82x more total KC input weight than MBON07
        # We scale MBON07 up to compensate for the anatomical imbalance
        self.mbon07_scale = 1.82 if normalize else 1.0
        self.identities = {
            "mbon07": [str(ids[i]) for i in self.mbon07],
            "mbon11": [str(ids[i]) for i in self.mbon11],
            "gate": [str(ids[i]) for i in self.gate] if len(self.gate) else [],
        }

    def decode(self, counts, seconds):
        mbon07_hz_raw = float(np.mean(counts[self.mbon07]) / seconds)
        mbon11_hz_raw = float(np.mean(counts[self.mbon11]) / seconds)
        
        # Apply anatomical normalization if enabled
        mbon07_hz = mbon07_hz_raw * self.mbon07_scale
        mbon11_hz = mbon11_hz_raw
        
        # Difference: positive = MBON11 dominates (approach), negative = MBON07 dominates (avoid)
        # P0 fix: reversed from original (was mbon07 - mbon11)
        difference = mbon11_hz - mbon07_hz
        gate = int(counts[self.gate].sum()) if len(self.gate) else 0
        
        # Decision logic (P0 fix):
        # - MBON11 > MBON07 (difference > 0) → BUY (approach bias from reward learning)
        # - MBON07 > MBON11 (difference < 0) → SELL (avoidance bias from punishment learning)
        side = (
            "HOLD"
            if abs(difference) < self.threshold
            else "BUY"
            if difference > 0
            else "SELL"
        )
        return {
            "side": side,
            "left_hz": mbon07_hz,   # avoidance output (P0: swapped)
            "right_hz": mbon11_hz,  # approach output (P0: swapped)
            "difference_hz": difference,
            "gate_spikes": gate,
            "cell_ids": self.identities,
            "mbon07_hz": mbon07_hz,
            "mbon11_hz": mbon11_hz,
            "mbon07_hz_raw": mbon07_hz_raw,
            "normalized": self.normalize,
        }


class RatioMBONDecoder:
    """MBON decoder using firing rate ratio instead of difference.
    
    This handles the anatomical imbalance where MBON11 fires ~12x more than MBON07.
    Instead of comparing raw Hz difference, we use the ratio:
    
    Based on Aso et al. 2014 eLife (PMC4273436):
    - MBON07 (α1) is glutamatergic → activation causes AVOIDANCE
    - MBON11 (γ1pedc>α/β) is GABAergic → activation causes APPROACH
    
    Therefore:
    - ratio HIGH (MBON07 dominates) → avoidance → SELL
    - ratio LOW (MBON11 dominates) → approach → BUY
    - otherwise → HOLD
    
    Default thresholds are calibrated for typical MBON07/MBON11 ratio of ~0.08.
    """

    def __init__(self, ids, annotation, sell_threshold=0.15, buy_threshold=0.05):
        types = annotation.type.fillna("")
        self.mbon07 = np.flatnonzero(types.eq("MBON07"))
        self.mbon11 = np.flatnonzero(types.eq("MBON11"))
        self.gate = np.flatnonzero(types.eq("DNpe017"))
        if not len(self.mbon07) or not len(self.mbon11):
            raise RuntimeError("Missing MBON07 or MBON11 neurons")
        self.sell_threshold = sell_threshold  # ratio above this → SELL (avoidance)
        self.buy_threshold = buy_threshold    # ratio below this → BUY (approach)
        self.identities = {
            "mbon07": [str(ids[i]) for i in self.mbon07],
            "mbon11": [str(ids[i]) for i in self.mbon11],
            "gate": [str(ids[i]) for i in self.gate] if len(self.gate) else [],
        }

    def decode(self, counts, seconds):
        mbon07_hz = float(np.mean(counts[self.mbon07]) / seconds)
        mbon11_hz = float(np.mean(counts[self.mbon11]) / seconds)
        gate = int(counts[self.gate].sum()) if len(self.gate) else 0
        
        # Calculate ratio (with small epsilon to avoid division by zero)
        # Based on Aso 2014: MBON07 (glutamate) = avoidance, MBON11 (GABA) = approach
        # Both silent -> HOLD (no information)
        # Only MBON07 active -> strong avoidance -> SELL
        # Only MBON11 active -> strong approach -> BUY
        if mbon07_hz < 0.1 and mbon11_hz < 0.1:
            ratio = None  # Both silent, no decision basis
            side = "HOLD"
        elif mbon11_hz < 0.1:
            ratio = 10.0  # MBON07 active, MBON11 silent -> strong avoidance
            side = "SELL"
        elif mbon07_hz < 0.1:
            ratio = 0.0  # MBON11 active, MBON07 silent -> strong approach
            side = "BUY"
        else:
            ratio = mbon07_hz / mbon11_hz
            # Decision based on ratio thresholds
            # High ratio = MBON07 dominates = avoidance = SELL
            # Low ratio = MBON11 dominates = approach = BUY
            if ratio > self.sell_threshold:
                side = "SELL"
            elif ratio < self.buy_threshold:
                side = "BUY"
            else:
                side = "HOLD"
        
        return {
            "side": side,
            "left_hz": mbon11_hz,
            "right_hz": mbon07_hz,
            "ratio": ratio if ratio is not None else 0.0,
            "difference_hz": mbon07_hz - mbon11_hz,
            "gate_spikes": gate,
            "cell_ids": self.identities,
            "mbon07_hz": mbon07_hz,
            "mbon11_hz": mbon11_hz,
            "sell_threshold": self.sell_threshold,
            "buy_threshold": self.buy_threshold,
        }


class FlyController:
    def __init__(self, settings, *, diagnostic_dt_ms=None):
        self.s = settings
        brain_kwargs = {} if diagnostic_dt_ms is None else {"diagnostic_dt_ms": diagnostic_dt_ms}
        self.brain = VisualMemoryBrain(**brain_kwargs)
        self.brain.weights_frozen = not settings.learning
        ann = annotations(self.brain.ids)
        decoder_type = getattr(settings, 'decoder_type', 'dnp20')
        normalize_mbon = getattr(settings, 'normalize_mbon', False)
        if decoder_type == "mbon":
            self.decoder = MBONDecoder(
                self.brain.ids, ann, settings.decoder_threshold_hz,
                normalize=normalize_mbon
            )
        elif decoder_type == "mbon_ratio":
            # sell_threshold: ratio above this → SELL (MBON07/avoidance dominates)
            # buy_threshold: ratio below this → BUY (MBON11/approach dominates)
            sell_thresh = getattr(settings, 'mbon_sell_threshold', 0.15)
            buy_thresh = getattr(settings, 'mbon_buy_threshold', 0.05)
            self.decoder = RatioMBONDecoder(
                self.brain.ids, ann, sell_thresh, buy_thresh
            )
        else:
            self.decoder = Decoder(
                self.brain.ids, ann, settings.decoder_threshold_hz
            )

    def observe(self, rgb, reinforcement):
        if reinforcement not in ("none", "reward", "aversive"):
            raise ValueError("Unknown reinforcement")
        b = self.brain
        counts = np.zeros(b.n, dtype=np.int32)
        wall = 0.0
        remaining = round(self.s.neural_ms / b.dt)
        
        # Delayed pulse timing: wait pulse_delay_ms before delivering pulse
        # This allows KC trace from previous observation to decay, enabling
        # bidirectional plasticity (both potentiation and depression).
        # See docs/plasticity_diagnosis_results.md for rationale.
        delay = round(self.s.pulse_delay_ms / b.dt) if reinforcement != "none" else 0
        pulse = round(self.s.pulse_ms / b.dt) if reinforcement != "none" else 0
        delivered = 0
        
        while remaining:
            n = min(remaining, round(self.s.neural_bin_ms / b.dt))
            
            # During delay phase: no pulse, just let KC activity build
            if delay:
                n = min(n, delay)
                stimulus = None
                delay -= n
            # During pulse phase: deliver reinforcement
            elif pulse:
                n = min(n, pulse)
                stimulus = (b.circuit[reinforcement], self.s.pulse_current)
                delivered += n
                pulse -= n
            # After pulse: continue simulation without stimulus
            else:
                stimulus = None
            
            c, elapsed = b.rgb_step(
                rgb, n * b.dt, learning=self.s.learning, stimulation=stimulus
            )
            counts += c
            wall += elapsed
            remaining -= n
            
        b.counts[:] = counts
        self.last_compute_seconds = wall
        return {
            **self.decoder.decode(counts, self.s.neural_ms / 1000),
            "brain_ms": b.sim_ms,
            "stimulus": reinforcement,
            "stimulus_ms": delivered * b.dt,
            "reward_spikes": int(counts[b.circuit["reward"]].sum()),
            "aversive_spikes": int(counts[b.circuit["aversive"]].sum()),
            "KC_spikes": int(counts[b.circuit["kc"]].sum()),
            "total_spikes": int(counts.sum()),
            "spike_sha256": hashlib.sha256(counts.tobytes()).hexdigest(),
            "input_sha256": hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest(),
            "memory": b.memory(),
        }

    def save(self, path):
        self.brain.checkpoint(path)

    def restore(self, path):
        self.brain.restore(path)

    def restore_weights_only(self, path):
        """Restore only synaptic weights, keeping dynamic state fresh."""
        self.brain.restore_weights_only(path)
