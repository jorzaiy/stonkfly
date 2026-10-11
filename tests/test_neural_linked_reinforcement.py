import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from stonkfly.neural.v51_controller import V51Controller


class _FakeBrain:
    def __init__(self):
        self.n = 5
        self.dt = 0.1
        self.counts = np.zeros(self.n, dtype=np.int32)
        self.learning_calls = []
        self.circuit = {
            "edges": np.arange(4, dtype=np.int64),
            "pre": np.arange(4, dtype=np.int32),
            "gain": np.zeros((2, 4), dtype=np.float32),
            "dan": np.asarray([0, 1], dtype=np.int32),
            "reward": np.asarray([0], dtype=np.int32),
            "aversive": np.asarray([1], dtype=np.int32),
            "kc": np.asarray([2], dtype=np.int32),
        }
        self.circuit["gain"][0, 0] = 1.0
        self.circuit["gain"][1, 1] = 1.0
        self.weight = np.ones(5, dtype=np.float32)
        self.eligibility = np.zeros(5, dtype=np.float64)
        self.eligibility[0] = 1.0
        self.luminance = np.zeros(3, dtype=np.float32)
        self.sim_ms = 0.0
        self.last_plasticity_gain = None

    def rgb_step(self, frame, duration_ms, *, learning, stimulation=None):
        self.learning_calls.append(bool(learning))
        self.sim_ms += float(duration_ms)
        self.eligibility[0] += 1.0
        counts = np.zeros(self.n, dtype=np.int32)
        self.counts[:] = counts
        return counts, 0.1

    def memory(self):
        return {"changed_edges": int(np.count_nonzero(self.weight != 1.0))}

    def step(self, luminance, duration_ms, *, learning, stimulation, plasticity_gain=None):
        np.testing.assert_array_equal(luminance, self.luminance)
        assert learning is True
        indices, current = stimulation
        assert float(current) == 20.0
        self.last_plasticity_gain = np.asarray(plasticity_gain).copy()
        np.testing.assert_array_equal(self.last_plasticity_gain.shape, self.circuit["gain"].shape)
        self.sim_ms += float(duration_ms)
        self.eligibility[np.asarray(self.circuit["pre"])[np.any(self.last_plasticity_gain != 0, axis=0)]] += 1.0
        changed_edges = np.any(self.last_plasticity_gain != 0, axis=0)
        self.weight[np.asarray(self.circuit["edges"])[changed_edges]] *= 0.9
        counts = np.zeros(self.n, dtype=np.int32)
        counts[np.asarray(indices)] = 2
        return counts, 0.25

    def checkpoint(self, path):
        np.savez_compressed(path, weight=self.weight, eligibility=self.eligibility, sim_ms=np.asarray(self.sim_ms))

    def restore(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.weight[:] = data["weight"]
            self.eligibility[:] = data["eligibility"]
            self.sim_ms = float(data["sim_ms"])


def _controller():
    controller = V51Controller.__new__(V51Controller)
    controller.brain = _FakeBrain()
    controller.s = SimpleNamespace(pulse_ms=2.0, pulse_current=20.0, learning=True)
    controller.linked_reinforcement_records = []
    controller.last_linked_reinforcement = None
    controller.s.neural_ms = 2.0
    controller.s.neural_bin_ms = 1.0
    controller.decoder = SimpleNamespace(decode=lambda counts, seconds: {"side": "HOLD"})
    return controller


# Provenance record for V51Controller implementation:
# - Legacy 2022 replay hash (pre-sensory hook): "aa8dd895b4853b2b8be9c58893c63ef5e0f1960ca6fad28afd48a77df6eb05bf"
# - Current active implementation hash (adds SensoryStimulation optional hook): "bd43e83da45b58517561fd72657feaf653a6841d56114e704028b04e551311bb"
EXPECTED_CURRENT_V51_CONTROLLER_SHA256 = "bd43e83da45b58517561fd72657feaf653a6841d56114e704028b04e551311bb"
LEGACY_2022_V51_CONTROLLER_SHA256 = "aa8dd895b4853b2b8be9c58893c63ef5e0f1960ca6fad28afd48a77df6eb05bf"


def test_controller_source_hash_is_pinned_for_provenance():
    source = Path(__import__("inspect").getsourcefile(V51Controller))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == EXPECTED_CURRENT_V51_CONTROLLER_SHA256


def test_none_is_explicit_no_op_without_advancing_or_mutating():
    controller = _controller()
    before_weight = controller.brain.weight.copy()
    before_trace = controller.brain.eligibility.copy()
    result = controller.reinforce_linked("none", "batch-none")
    assert result["status"] == "NO_OP_NONE"
    assert result["advanced"] is False
    assert result["brain_ms_before"] == result["brain_ms_after"] == 0.0
    assert result["changed_edges"] == 0
    assert np.array_equal(before_weight, controller.brain.weight)
    assert np.array_equal(before_trace, controller.brain.eligibility)


def test_unlinked_reward_returns_inconclusive_without_fake_pulse():
    controller = _controller()
    controller.brain.eligibility[0] = 0.0
    before_weight = controller.brain.weight.copy()
    result = controller.reinforce_linked("reward", "batch-reward")
    assert result["status"] == "INCONCLUSIVE_NO_ELIGIBILITY"
    assert result["advanced"] is False
    assert result["eligible_edge_count"] == 0
    assert result["dan_spikes"] == 0
    assert np.array_equal(before_weight, controller.brain.weight)


def test_linked_reward_records_trace_weight_and_dan_spikes():
    controller = _controller()
    result = controller.reinforce_linked("reward", "batch-reward")
    assert result["status"] == "APPLIED"
    assert result["advanced"] is True
    assert result["stimulus"] == "reward"
    assert result["linkage_id"] == "batch-reward"
    assert result["eligible_edge_count"] == 1
    assert result["eligible_changed_edges"] == 1
    assert result["changed_edges"] == 1
    assert result["dan_spikes"] == 2
    assert result["trace_before_digest"] != result["trace_after_digest"]
    assert result["plastic_weight_before_digest"] != result["plastic_weight_after_digest"]
    assert controller.last_linked_reinforcement == result


def test_linked_pulse_masks_cross_compartment_gain_and_preserves_source_gain():
    controller = _controller()
    source_gain = controller.brain.circuit["gain"].copy()
    controller.brain.eligibility[0] = 0.0
    controller.brain.eligibility[1] = 1.0
    result = controller.reinforce_linked("aversive", "batch-aversive")
    assert result["status"] == "APPLIED"
    np.testing.assert_array_equal(controller.brain.last_plasticity_gain, [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    assert controller.brain.weight[0] == 1.0
    assert controller.brain.weight[1] == np.float32(0.9)
    np.testing.assert_array_equal(controller.brain.circuit["gain"], source_gain)


def test_memory_brain_step_default_gain_and_strict_override_validation(monkeypatch):
    from stonkfly.neural.brain import MemoryBrain
    import stonkfly.neural.rule as rule

    class DummyBrain:
        n = 2
        dt = 0.1
        eta = 0.0
        weights_frozen = True
        circuit = {
            "gain": np.arange(6, dtype=np.float32).reshape(2, 3),
            "pre": np.asarray([0, 1, 0], dtype=np.int32),
            "dan": np.asarray([0, 1], dtype=np.int32),
        }
        rate_kc = np.zeros(3, dtype=np.float64)
        rate_dan = np.zeros(2, dtype=np.float64)
        dan_baseline_hz = np.zeros(2, dtype=np.float64)
        memory_u = np.zeros(3, dtype=np.float64)
        memory_w = np.zeros(3, dtype=np.float64)
        counts = np.zeros(2, dtype=np.int32)

        def _neural_step(self, *args, **kwargs):
            return np.asarray([1, 2], dtype=np.int32), 0.0

    dummy = DummyBrain()
    seen = []
    monkeypatch.setattr(rule, "advance", lambda *args, **kwargs: seen.append(args[6]))
    MemoryBrain.step(dummy, np.zeros(1, dtype=np.float32), 0.1)
    np.testing.assert_array_equal(seen[0], dummy.circuit["gain"])
    with pytest.raises(ValueError, match="plasticity_gain"):
        MemoryBrain.step(dummy, np.zeros(1, dtype=np.float32), 0.1, plasticity_gain=np.zeros((1, 3)))


def test_save_restore_repeats_linked_reinforcement_deterministically(tmp_path):
    controller = _controller()
    clean = tmp_path / "clean.npz"
    controller.save(clean)
    first = controller.reinforce_linked("reward", "batch-reward")
    first_weight = controller.brain.weight.copy()
    controller.restore(clean)
    second = controller.reinforce_linked("reward", "batch-reward")
    assert first == second
    assert np.array_equal(first_weight, controller.brain.weight)
    assert callable(getattr(controller, "reinforce_only"))


def test_none_observe_propagates_and_updates_eligibility_without_plastic_drive():
    controller = _controller()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    before_weight = controller.brain.weight.copy()
    before_trace = controller.brain.eligibility.copy()
    result = controller.observe(frame, "none")
    assert result["stimulus"] == "none"
    assert controller.brain.learning_calls and all(
        call is False for call in controller.brain.learning_calls
    )
    assert np.array_equal(before_weight, controller.brain.weight)
    assert controller.brain.eligibility[0] > before_trace[0]
    with pytest.raises(ValueError, match="reinforce_linked"):
        controller.observe(frame, "reward")


@pytest.mark.skipif(
    os.environ.get("STONKFLY_FULL_TEST") != "1",
    reason="small full MaleCNS V5.1 smoke",
)
def test_full_graph_none_observe_has_eligibility_without_new_plastic_delta():
    from stonkfly.config import Settings

    controller = V51Controller(
        Settings(neural_ms=500.0, neural_bin_ms=10.0, pulse_ms=200.0, learning=True)
    )
    frame = np.full((180, 320, 3), 255, dtype=np.uint8)
    before_trace = controller.brain.eligibility.copy()
    before_weight = controller.brain.weight[controller.brain.circuit["edges"]].copy()
    result = controller.observe(frame, "none")
    after_trace = controller.brain.eligibility.copy()
    after_weight = controller.brain.weight[controller.brain.circuit["edges"]].copy()
    assert len(controller.brain.circuit["edges"]) == 7_835
    assert result["KC_spikes"] > 0
    assert np.count_nonzero(after_trace != before_trace) > 0
    assert np.array_equal(before_weight, after_weight)
