from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.a_share.wc006_v5_1_reward_eligibility_coverage import (
    EXPECTED_SOURCE_MANIFEST_SHA256,
    run_coverage_audit,
    verify_append_only_log,
    sha256,
)


class _Brain:
    def __init__(self):
        self.n = 8
        self.eligibility = np.zeros(8, dtype=np.float64)
        self.counts = np.zeros(8, dtype=np.int32)
        self.weight = np.ones(8, dtype=np.float32)
        self.memory_u = np.zeros(3, dtype=np.float64)
        self.memory_w = np.zeros(3, dtype=np.float64)
        self.circuit = {
            "edges": np.arange(3, dtype=np.int64),
            "pre": np.arange(3, dtype=np.int32),
            "gain": np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
            "reward": np.asarray([3], dtype=np.int32),
            "aversive": np.asarray([4], dtype=np.int32),
            "kc": np.asarray([5, 6], dtype=np.int32),
        }


class _Controller:
    def __init__(self, settings):
        self.brain = _Brain()

    def observe(self, frame, reinforcement):
        assert reinforcement == "none"
        self.brain.eligibility[0] += 1
        self.brain.counts[:] = 0
        self.brain.counts[[3, 5]] = 1
        raw = np.asarray(frame)
        return {
            "input_sha256": __import__("hashlib").sha256(raw.tobytes()).hexdigest(),
            "spike_sha256": __import__("hashlib").sha256(self.brain.counts.tobytes()).hexdigest(),
            "KC_spikes": 1,
            "reward_spikes": 1,
            "aversive_spikes": 0,
        }

    def save(self, path):
        np.savez_compressed(path, eligibility=self.brain.eligibility, counts=self.brain.counts)


def test_fixture_coverage_audit_is_sensory_only_and_hash_chained(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"source_manifest_sha256": "' + EXPECTED_SOURCE_MANIFEST_SHA256 + '"}\n')
    records = [
        SimpleNamespace(signal_date="2023-01-03", targets=[{"code": "000001", "weight": 1.0, "score": 0.1}]),
        SimpleNamespace(signal_date="2023-12-29", targets=[{"code": "000001", "weight": 1.0, "score": 0.1}]),
    ]
    stream = SimpleNamespace(base=SimpleNamespace(calendar=("2023-01-03", "2023-12-29"), records=records), prices={})

    def loader(path):
        return stream, {"candidate_manifest_sha256": sha256(path), "status": "PASS", "paper_only": True}

    result = run_coverage_audit(
        tmp_path / "audit",
        manifest,
        controller_factory=_Controller,
        stream_loader=loader,
        frame_factory=lambda *args: np.zeros((180, 320, 3), dtype=np.uint8),
        expected_dates=2,
        expected_signals=2,
    )
    assert result["status"] == "COMPLETE"
    assert result["observation_count"] == 2
    assert result["candidate_day_count"] == 2
    assert result["reward_eligibility"]["days_nonzero"] == 2
    assert result["aversive_eligibility"]["days_nonzero"] == 0
    assert result["checks"]["all_reinforcement_none"] is True
    assert result["checks"]["observation_count_expected"] is True
    assert result["checks"]["candidate_day_count_expected"] is True
    assert result["expected_counts"] == {"observations": 2, "candidate_days": 2}
    assert result["checks"]["no_market_execution"] is True
    assert result["checks"]["plastic_edge_weights_unchanged"] is True
    assert result["checks"]["memory_u_unchanged"] is True
    assert result["checks"]["memory_w_unchanged"] is True
    assert result["provenance"]["v51_controller_sha256"]
    assert verify_append_only_log(tmp_path / "audit" / "coverage-events.jsonl")["records"] == 4


def test_real_annual_manifest_preflight_renders_frozen_first_frame():
    from experiments.a_share.wc006_v5_1_reward_eligibility_coverage import preflight_annual_inputs

    result = preflight_annual_inputs()
    assert result["records"] == 17
    assert result["frame_sha256"] == "3ab3e37867cbb9b2a860ad8d2760cf5f5b707001c5357da58f4c1c45e0867971"
    assert result["shape"] == [180, 320, 3]
    assert result["dtype"] == "|u1"
