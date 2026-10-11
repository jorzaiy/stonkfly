from types import SimpleNamespace

import numpy as np

from stonkfly.neural.sensory_adapter import FixedVectorSensoryAdapter
from stonkfly.neural.v51_controller import V51Controller


class _Brain:
    n = 8
    dt = 0.1
    ids = np.arange(n, dtype=np.int64)

    def __init__(self):
        self.counts = np.zeros(self.n, dtype=np.int32)
        self.sim_ms = 0.0
        self.circuit = {
            "reward": np.asarray([0]),
            "aversive": np.asarray([1]),
            "kc": np.asarray([2]),
            "reward_mb": np.asarray([3]),
            "aversive_mb": np.asarray([4]),
        }
        self.calls = []

    def rgb_step(self, rgb, duration_ms, *, learning, **kwargs):
        self.calls.append(kwargs)
        self.sim_ms += duration_ms
        counts = np.arange(1, self.n + 1, dtype=np.int32)
        self.counts[:] = counts
        return counts, 0.0

    def memory(self):
        return {"changed_edges": 0}


def _controller():
    controller = V51Controller.__new__(V51Controller)
    controller.brain = _Brain()
    controller.s = SimpleNamespace(neural_ms=1.0, neural_bin_ms=1.0)
    controller.decoder = SimpleNamespace(decode=lambda counts, seconds: {"side": "HOLD"})
    return controller


def _stimulation():
    adapter = FixedVectorSensoryAdapter(
        [f"f{i}" for i in range(8)],
        [[i] for i in range(8)],
        neuron_ids=[[i] for i in range(8)],
    )
    return adapter.encode([0.5] * 8)


def test_null_path_omits_stimulation_keyword_and_reports_summaries():
    controller = _controller()
    result = controller.observe(np.zeros((2, 2, 3), dtype=np.uint8))
    assert controller.brain.calls == [{}]
    assert set(result["activation_summary"]) == {"MBON07", "MBON11", "PAM11", "PPL101"}
    assert "sensory_stimulation" not in result


def test_fixed_stimulation_is_serialised_and_passed_as_extra_pulses():
    controller = _controller()
    stimulation = _stimulation()
    result = controller.observe(
        np.zeros((2, 2, 3), dtype=np.uint8), sensory_stimulation=stimulation
    )
    assert "stimulation" in controller.brain.calls[0]
    assert len(controller.brain.calls[0]["stimulation"]) == 8
    assert result["sensory_stimulation"] == stimulation.to_dict()
