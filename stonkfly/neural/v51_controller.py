"""V5.1-only controller extensions.

The frozen :class:`FlyController` remains the V5 controller contract.  This
subclass adds linked dopamine delivery used by the separate V5.1 mechanism
assay/adapter without changing the V5 controller source hash.
"""

import hashlib

import numpy as np

from .controller import FlyController
from .sensory_adapter import SensoryStimulation


def _array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


class V51Controller(FlyController):
    """FlyController plus explicitly linked, eligibility-gated reinforcement."""

    def __init__(self, settings, *, diagnostic_dt_ms=None):
        super().__init__(settings, diagnostic_dt_ms=diagnostic_dt_ms)
        self.linked_reinforcement_records = []
        self.last_linked_reinforcement = None
        self.last_sensory_stimulation = None

    def observe(self, rgb, reinforcement="none", sensory_stimulation=None):
        """Propagate one RGB observation without opening a plasticity drive.

        V5.1 separates the sensory/eligibility phase from reinforcement. RGB
        input still drives the complete retained graph and KC spikes still
        write eligibility traces, but the centered rate-rule drive is disabled
        for this phase. Consequently a ``none`` observation cannot create a
        new plastic delta; any existing ``memory_u``/``memory_w`` continues its
        declared passive decay in :meth:`MemoryBrain.step`.

        Reward and aversive delivery must use :meth:`reinforce_linked`, which
        is deliberately a separate operation with no new RGB cue.
        """
        if reinforcement not in ("none",):
            raise ValueError(
                "V5.1 reward/aversive delivery requires reinforce_linked"
            )
        b = self.brain
        if sensory_stimulation is not None and not isinstance(
            sensory_stimulation, SensoryStimulation
        ):
            raise TypeError("sensory_stimulation must be SensoryStimulation or None")
        if sensory_stimulation is not None:
            # Direct controller callers cannot bypass the runner's
            # adapter-level graph check. Validate before neural state advances.
            sensory_stimulation.validate_against_graph(b.ids)
        self.last_sensory_stimulation = sensory_stimulation
        extra_pulses = (
            [] if sensory_stimulation is None else sensory_stimulation.brain_pulses()
        )
        counts = np.zeros(b.n, dtype=np.int32)
        wall = 0.0
        remaining = round(self.s.neural_ms / b.dt)
        while remaining:
            n = min(remaining, round(self.s.neural_bin_ms / b.dt))
            # Keep the no-adapter call byte-for-byte compatible with V5.1:
            # omitted stimulation is the original RGB/R8-only path.
            if sensory_stimulation is None or not extra_pulses:
                c, elapsed = b.rgb_step(
                    rgb,
                    n * b.dt,
                    # Sensory propagation and eligibility remain active; only
                    # the new rate-rule plastic drive is disabled for RGB
                    # observations.
                    learning=False,
                )
            else:
                c, elapsed = b.rgb_step(
                    rgb,
                    n * b.dt,
                    learning=False,
                    stimulation=extra_pulses,
                )
            counts += c
            wall += elapsed
            remaining -= n
        b.counts[:] = counts
        self.last_compute_seconds = wall
        activation = {}
        seconds = self.s.neural_ms / 1000
        for label, key in (("MBON07", "reward_mb"), ("MBON11", "aversive_mb"), ("PAM11", "reward"), ("PPL101", "aversive")):
            indices = np.asarray(b.circuit.get(key, []), dtype=np.int64)
            values = counts[indices] if indices.size else np.empty(0, dtype=np.int32)
            activation[label] = {
                "spikes": int(values.sum()),
                "mean_hz": float(np.mean(values) / seconds) if values.size else 0.0,
                "max_spikes": int(values.max()) if values.size else 0,
                "cell_count": int(values.size),
            }
        result = {
            **self.decoder.decode(counts, seconds),
            "brain_ms": b.sim_ms,
            "stimulus": "none",
            "stimulus_ms": 0.0,
            "reward_spikes": int(counts[b.circuit["reward"]].sum()),
            "aversive_spikes": int(counts[b.circuit["aversive"]].sum()),
            "KC_spikes": int(counts[b.circuit["kc"]].sum()),
            "total_spikes": int(counts.sum()),
            "spike_sha256": hashlib.sha256(counts.tobytes()).hexdigest(),
            "input_sha256": hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest(),
            "memory": b.memory(),
            "activation_summary": activation,
        }
        if sensory_stimulation is not None:
            result["sensory_stimulation"] = sensory_stimulation.to_dict()
        return result

    def _linked_eligibility_mask(self, stimulus):
        """Return edge eligibility for the selected DAN label before delivery."""
        if stimulus not in ("reward", "aversive"):
            raise ValueError("Unknown linked reinforcement")
        circuit = self.brain.circuit
        edges = np.asarray(circuit["edges"], dtype=np.int64)
        pre = np.asarray(circuit["pre"], dtype=np.int64)
        gain = np.asarray(circuit["gain"])
        if gain.ndim != 2 or gain.shape[1] != len(edges):
            raise RuntimeError("Invalid DAN gain shape")
        dan_rows = np.asarray(circuit["dan"], dtype=np.int64)
        if dan_rows.shape != (gain.shape[0],):
            raise RuntimeError("Invalid DAN gain rows")
        if pre.shape != edges.shape or np.any(pre < 0) or np.any(pre >= self.brain.n):
            raise RuntimeError("Invalid plastic edge source indices")
        label_rows = np.flatnonzero(np.isin(dan_rows, np.asarray(circuit[stimulus], dtype=np.int64)))
        if label_rows.size == 0:
            raise RuntimeError("Invalid DAN label rows")
        return np.any(gain[label_rows] != 0, axis=0) & (np.abs(self.brain.eligibility[pre]) > 0)

    def reinforce_linked(self, stimulus, linkage_id=None):
        """Deliver a linked DAN pulse without presenting a new RGB cue.

        ``none`` is an explicit no-op. Reward/aversive delivery is gated by
        the eligibility trace that exists before the call; if no edge linked to
        the selected DAN group is eligible, the call does not advance state and
        returns ``INCONCLUSIVE_NO_ELIGIBILITY``. A linked pulse uses the current
        retinal state as a neural background state, not a new RGB observation.
        """
        if stimulus not in ("none", "reward", "aversive"):
            raise ValueError("Unknown linked reinforcement")
        if linkage_id is not None and not isinstance(linkage_id, str):
            raise TypeError("linkage_id must be a string or None")
        b = self.brain
        edges = np.asarray(b.circuit["edges"], dtype=np.int64)
        weights_before = np.asarray(b.weight[edges]).copy()
        trace_before = np.asarray(b.eligibility).copy()
        before_ms = float(b.sim_ms)

        def record(status, *, dan_indices, dan_counts, eligible_mask, advanced, wall=0.0):
            weights_after = np.asarray(b.weight[edges]).copy()
            trace_after = np.asarray(b.eligibility).copy()
            changed = np.abs(weights_after - weights_before) > 0
            dan_counts = np.asarray(dan_counts, dtype=np.int32)
            result = {
                "status": status,
                "stimulus": stimulus,
                "linkage_id": linkage_id,
                "advanced": bool(advanced),
                "stimulus_ms": float(self.s.pulse_ms if advanced else 0.0),
                "pulse_current": float(self.s.pulse_current if advanced else 0.0),
                "brain_ms_before": before_ms,
                "brain_ms_after": float(b.sim_ms),
                "trace_before_digest": _array_digest(trace_before),
                "trace_after_digest": _array_digest(trace_after),
                "plastic_weight_before_digest": _array_digest(weights_before),
                "plastic_weight_after_digest": _array_digest(weights_after),
                "dan_spikes": int(dan_counts.sum()),
                "dan_spike_counts": dan_counts.tolist(),
                "dan_spike_digest": _array_digest(dan_counts),
                "changed_edges": int(np.count_nonzero(changed)),
                "eligible_edge_count": int(np.count_nonzero(eligible_mask)),
                "eligible_changed_edges": int(np.count_nonzero(changed & eligible_mask)),
            }
            self.last_linked_reinforcement = result
            self.linked_reinforcement_records.append(dict(result))
            return result

        if stimulus == "none":
            return record(
                "NO_OP_NONE",
                dan_indices=np.empty(0, dtype=np.int32),
                dan_counts=np.empty(0, dtype=np.int32),
                eligible_mask=np.zeros(len(edges), dtype=bool),
                advanced=False,
            )

        eligible_mask = self._linked_eligibility_mask(stimulus)
        dan_indices = np.asarray(b.circuit[stimulus], dtype=np.int64)
        if not np.any(eligible_mask):
            return record(
                "INCONCLUSIVE_NO_ELIGIBILITY",
                dan_indices=dan_indices,
                dan_counts=np.zeros(len(dan_indices), dtype=np.int32),
                eligible_mask=eligible_mask,
                advanced=False,
            )

        luminance = np.asarray(b.luminance, dtype=np.float32).copy()
        gain = np.asarray(b.circuit["gain"])
        dan_rows = np.asarray(b.circuit["dan"], dtype=np.int64)
        label_rows = np.flatnonzero(np.isin(dan_rows, np.asarray(b.circuit[stimulus], dtype=np.int64)))
        masked_gain = np.zeros_like(gain)
        masked_gain[label_rows, :] = gain[label_rows, :]
        counts, wall = b.step(
            luminance,
            self.s.pulse_ms,
            # Linked reinforcement is the only V5.1 plasticity trigger. It
            # reuses retinal state as background input and never presents a
            # new RGB cue.
            learning=True,
            stimulation=(dan_indices, self.s.pulse_current),
            plasticity_gain=masked_gain,
        )
        return record(
            "APPLIED",
            dan_indices=dan_indices,
            dan_counts=np.asarray(counts)[dan_indices],
            eligible_mask=eligible_mask,
            advanced=True,
            wall=wall,
        )

    def reinforce_only(self, stimulus, linkage_id=None):
        """Compatibility entry point for V5.1 mechanism-B callers."""
        return self.reinforce_linked(stimulus, linkage_id=linkage_id)
