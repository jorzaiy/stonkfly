import hashlib
from pathlib import Path

import numpy as np

import experiments.a_share.wc006_v5_1_post_failure_linked_pulse as diagnostic


class _Brain:
    def __init__(self):
        self.n = 20
        self.ids = np.arange(self.n, dtype=np.int64)
        self.ptr = np.arange(self.n + 1, dtype=np.int64)
        self.post = np.arange(self.n, dtype=np.int32)
        self.weight = np.ones(self.n, dtype=np.float32)
        self.baseline_plastic = np.ones(3, dtype=np.float32)
        self.eligibility = np.zeros(self.n, dtype=np.float64)
        self.eligibility[:3] = 1.0
        self.rate_dan = np.zeros(17, dtype=np.float64)
        self.counts = np.zeros(self.n, dtype=np.int32)
        self.memory_u = np.zeros(3, dtype=np.float64)
        self.memory_w = np.zeros(3, dtype=np.float64)
        self.cursor = 0
        self.sim_ms = 0.0
        self.luminance = np.zeros((2, 2), dtype=np.float32)
        gain = np.zeros((17, 3), dtype=np.float32)
        gain[:15, :2] = 1.0
        gain[15:, 2:] = 1.0
        self.circuit = {
            "edges": np.arange(3, dtype=np.int64),
            "pre": np.arange(3, dtype=np.int32),
            "dan": np.arange(17, dtype=np.int32),
            "reward": np.arange(15, dtype=np.int32),
            "aversive": np.arange(15, 17, dtype=np.int32),
            "gain": gain,
        }


class _Controller:
    instances = []

    def __init__(self, settings):
        self.brain = _Brain()
        self.calls = []
        self.__class__.instances.append(self)

    def save(self, path):
        np.savez_compressed(
            path,
            weight=self.brain.weight,
            eligibility=self.brain.eligibility,
            rate_dan=self.brain.rate_dan,
            counts=self.brain.counts,
            memory_u=self.brain.memory_u,
            memory_w=self.brain.memory_w,
            cursor=np.asarray(self.brain.cursor),
            sim_ms=np.asarray(self.brain.sim_ms),
        )

    def restore(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.brain.weight[:] = data["weight"]
            self.brain.eligibility[:] = data["eligibility"]
            self.brain.rate_dan[:] = data["rate_dan"]
            self.brain.counts[:] = data["counts"]
            self.brain.memory_u[:] = data["memory_u"]
            self.brain.memory_w[:] = data["memory_w"]
            self.brain.cursor = int(data["cursor"])
            self.brain.sim_ms = float(data["sim_ms"])

    def reinforce_linked(self, stimulus, linkage_id=None):
        self.calls.append((stimulus, linkage_id))
        before = self.brain.weight[self.brain.circuit["edges"]].copy()
        if stimulus == "none":
            status = "NO_OP_NONE"
            advanced = False
            indices = np.empty(0, dtype=np.int32)
        else:
            status = "APPLIED"
            advanced = True
            indices = self.brain.circuit[stimulus]
            edge = 0 if stimulus == "reward" else 2
            self.brain.weight[edge] *= np.float32(0.9)
            self.brain.cursor += 200
            self.brain.sim_ms += 200.0
            self.brain.counts[indices] += 1
            self.brain.rate_dan[indices] += 1.0
        after = self.brain.weight[self.brain.circuit["edges"]]
        changed = np.abs(after - before) > 0
        return {
            "status": status,
            "stimulus": stimulus,
            "linkage_id": linkage_id,
            "advanced": advanced,
            "stimulus_ms": 200.0 if advanced else 0.0,
            "pulse_current": 20.0 if advanced else 0.0,
            "brain_ms_before": float(self.brain.sim_ms - (200.0 if advanced else 0.0)),
            "brain_ms_after": float(self.brain.sim_ms),
            "dan_spikes": int(self.brain.counts[indices].sum()) if len(indices) else 0,
            "dan_spike_counts": self.brain.counts[indices].tolist(),
            "changed_edges": int(np.count_nonzero(changed)),
            "eligible_edge_count": int(np.count_nonzero(changed)),
            "eligible_changed_edges": int(np.count_nonzero(changed)),
        }


def test_fixture_runs_three_production_linked_calls_and_none_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "EXPECTED_PLASTIC_EDGES", 3)
    monkeypatch.setattr(diagnostic, "EXPECTED_REWARD_ELIGIBILITY", 2)
    monkeypatch.setattr(diagnostic, "EXPECTED_AVERSIVE_ELIGIBILITY", 1)
    source = tmp_path / "endpoint.npz"
    _Controller.instances.clear()
    _Controller(None).save(source)
    endpoint_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    result = diagnostic.run_diagnostic(
        tmp_path / "diagnostic",
        endpoint=source,
        endpoint_sha256=endpoint_hash,
        controller_factory=_Controller,
    )
    assert result["status"] == "COMPLETE"
    assert result["mechanism_result"] == "PASS"
    assert result["checks"]["none_no_op"] is True
    assert result["checks"]["reward_localized"] is True
    assert result["checks"]["aversive_localized"] is True
    assert result["branches"]["reward"]["delivery"]["status"] == "APPLIED"
    assert result["branches"]["aversive"]["delivery"]["status"] == "APPLIED"
    assert result["branches"]["none"]["delivery"]["status"] == "NO_OP_NONE"
    assert [call[0] for instance in _Controller.instances for call in instance.calls] == ["reward", "aversive", "none"]
    assert result["log"]["records"] == 8


def test_eligible_delivery_failure_is_fail_not_inconclusive(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "EXPECTED_PLASTIC_EDGES", 3)
    monkeypatch.setattr(diagnostic, "EXPECTED_REWARD_ELIGIBILITY", 2)
    monkeypatch.setattr(diagnostic, "EXPECTED_AVERSIVE_ELIGIBILITY", 1)

    class _FailDeliveryController(_Controller):
        def reinforce_linked(self, stimulus, linkage_id=None):
            result = super().reinforce_linked(stimulus, linkage_id=linkage_id)
            if stimulus == "reward":
                result["status"] = "FAILED"
            return result

    source = tmp_path / "endpoint.npz"
    _FailDeliveryController(None).save(source)
    endpoint_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    result = diagnostic.run_diagnostic(
        tmp_path / "diagnostic",
        endpoint=source,
        endpoint_sha256=endpoint_hash,
        controller_factory=_FailDeliveryController,
    )

    assert result["branches"]["reward"]["eligible_edge_count"] > 0
    assert result["branches"]["aversive"]["eligible_edge_count"] > 0
    assert result["branches"]["reward"]["delivery"]["status"] == "FAILED"
    assert result["mechanism_result"] == "FAIL"


def test_diagnostic_source_has_no_direct_rgb_or_low_level_step_call():
    source = Path(diagnostic.__file__).read_text(encoding="utf-8")
    assert ".observe(" not in source
    assert ".rgb_step(" not in source
    assert ".step(" not in source
    assert "reinforce_linked(label" in source


def test_applied_delivery_still_uses_all_checks_for_pass_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "EXPECTED_PLASTIC_EDGES", 3)
    # Deliberately make the preregistered reward count check fail while both
    # production linked pulses remain APPLIED and eligible.
    monkeypatch.setattr(diagnostic, "EXPECTED_REWARD_ELIGIBILITY", 99)
    monkeypatch.setattr(diagnostic, "EXPECTED_AVERSIVE_ELIGIBILITY", 1)
    source = tmp_path / "endpoint.npz"
    _Controller(None).save(source)
    endpoint_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    result = diagnostic.run_diagnostic(
        tmp_path / "diagnostic",
        endpoint=source,
        endpoint_sha256=endpoint_hash,
        controller_factory=_Controller,
    )
    assert result["branches"]["reward"]["eligible_edge_count"] > 0
    assert result["branches"]["aversive"]["eligible_edge_count"] > 0
    assert result["branches"]["reward"]["delivery"]["status"] == "APPLIED"
    assert result["branches"]["aversive"]["delivery"]["status"] == "APPLIED"
    assert result["checks"]["reward_eligible_count_expected"] is False
    assert result["mechanism_result"] == "FAIL"
