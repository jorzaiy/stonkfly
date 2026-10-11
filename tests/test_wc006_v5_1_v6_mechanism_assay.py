import json

import numpy as np
import pytest

from stonkfly.neural.rule import PARAMETERS as RULE_PARAMETERS

from experiments.a_share.wc006_v5_1_v6_mechanism_assay import (
    DEFAULT_HORIZONS_MS,
    AppendOnlyLog,
    analytic_passive_projection,
    run_assay,
    verify_append_only_log,
    _label_eligibility_mask,
)


class _FakeBrain:
    def __init__(self):
        n = 7_835
        self.n = n
        self.ids = np.arange(n, dtype=np.int64)
        self.ptr = np.arange(n + 1, dtype=np.int64)
        self.post = np.arange(n, dtype=np.int32)
        self.weight = np.ones(n, dtype=np.float32)
        self.baseline_plastic = self.weight.copy()
        self.eligibility = np.zeros(n, dtype=np.float64)
        self.rate_dan = np.zeros(17, dtype=np.float64)
        self.counts = np.zeros(n, dtype=np.int32)
        self.memory_u = np.zeros(n, dtype=np.float64)
        self.memory_w = np.zeros(n, dtype=np.float64)
        self.cursor = 0
        self.sim_ms = 0.0
        gain = np.zeros((17, n), dtype=np.float32)
        gain[:15, :3_900] = 1.0
        gain[15:, 3_900:] = 1.0
        self.circuit = {
            "edges": np.arange(n, dtype=np.int64),
            "pre": np.arange(n, dtype=np.int32),
            "dan": np.arange(17, dtype=np.int32),
            "reward": np.arange(15, dtype=np.int32),
            "aversive": np.arange(15, 17, dtype=np.int32),
            "gain": gain,
        }

    def rgb_step(self, frame, duration_ms, *, learning=True, stimulation=None):
        self.cursor += int(duration_ms)
        self.sim_ms = float(self.cursor)
        if stimulation is None:
            self.eligibility += 1.0
            return
        indices, _ = stimulation
        indices = np.asarray(indices)
        if int(indices[0]) < 15:
            self.weight[:3_900] *= np.float32(0.99)
        else:
            self.weight[3_900:] *= np.float32(0.99)
        self.counts[indices] += 1
        self.rate_dan[indices] += 1.0

    def save(self, path):
        with open(path, "wb") as handle:
            np.savez_compressed(handle, weight=self.weight, eligibility=self.eligibility, cursor=np.asarray(self.cursor))

    def restore(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.weight[:] = data["weight"]
            self.eligibility[:] = data["eligibility"]
            self.cursor = int(data["cursor"])
            self.sim_ms = float(self.cursor)


class _FakeController:
    def __init__(self, settings):
        self.brain = _FakeBrain()

    def save(self, path):
        self.brain.save(path)

    def restore(self, path):
        self.brain.restore(path)


def test_label_eligibility_requires_gain_mask_intersection():
    gain = np.zeros((17, 4), dtype=np.float32)
    gain[0, :2] = 1.0
    gain[15, 2:] = 1.0
    snapshot = {
        "selected_gain_rows": gain,
        "edge_eligibility": np.asarray([1.0, 0.0, 1.0, 0.0]),
    }
    np.testing.assert_array_equal(_label_eligibility_mask(snapshot, "reward"), [True, False, False, False])
    np.testing.assert_array_equal(_label_eligibility_mask(snapshot, "aversive"), [False, False, True, False])


def test_analytic_passive_projection_uses_rule_closed_form_and_bounds():
    projected = analytic_passive_projection(
        np.asarray([0.5, -0.25]),
        np.asarray([0.2, -0.1]),
        np.asarray([2.0, 4.0]),
        1_800_000,
    )
    assert np.all(np.isfinite(projected["memory_u"]))
    assert np.all(np.isfinite(projected["memory_w"]))
    # The expectation follows the active declared decay constant rather than
    # silently assuming the historical 1,800 s value.
    decay = np.exp(-1_800 / float(RULE_PARAMETERS["memory_decay_seconds"]))
    np.testing.assert_allclose(projected["memory_u"], [0.5 * decay, -0.25 * decay])
    np.testing.assert_allclose(projected["memory_w"], projected["memory_u"], rtol=1e-4)
    np.testing.assert_allclose(projected["edge_weights"], [2.0 * (1 + projected["memory_w"][0]), 4.0 * (1 + projected["memory_w"][1])])


def test_append_only_log_rejects_tampering(tmp_path):
    path = tmp_path / "events.jsonl"
    log = AppendOnlyLog(path)
    log.append("start", {"seed": 17})
    log.append("end", {"status": "FAIL"})
    assert verify_append_only_log(path)["records"] == 2
    lines = path.read_text().splitlines()
    tampered = json.loads(lines[1])
    tampered["payload"]["status"] = "PASS"
    lines[1] = json.dumps(tampered, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_append_only_log(path)


def test_no_market_assay_records_full_edge_trace_dan_artifacts_and_fail_closed(tmp_path):
    output = tmp_path / "assay"
    result = run_assay(output, controller_factory=_FakeController, horizons_ms=DEFAULT_HORIZONS_MS)
    assert result["status"] == "FAIL"  # The fake timing control does not manufacture a pass.
    assert result["checks"]["plastic_edge_count"]
    assert result["checks"]["required_horizons"]
    assert result["checks"]["temporal_opposite_sign"] is False
    assert "raw_temporal_opposite_sign" in result["checks"]
    assert "raw_temporal_opposite_sign" in result["diagnostics"]
    assert "aversive_stimulation_specific_inconclusive_no_eligibility" in result["diagnostics"]
    assert "aversive_stimulation_specific_inconclusive_no_eligibility" not in result["gate_checks"]
    assert result["checks"]["matched_none_controls"] is True
    assert result["checks"]["passive_projection_integrity"] is True
    assert result["checks"]["full_network_1s_smoke"] is True
    assert set(result["branches"]["reward"]["passive_projection"]) == {"t1000ms", "t60000ms", "t1800000ms"}
    for projection in result["branches"]["reward"]["passive_projection"].values():
        assert projection["full_neural_state_advanced"] is False
        assert projection["input_sha256"] and projection["output_sha256"]
        assert projection["formula"]["name"] == "rule.advance.zero_drive_closed_form"
    dan_before = result["temporal"]["dan-before-kc"]
    assert dan_before["retention_origin_elapsed_ms"] == 500.0
    assert dan_before["snapshots"]["retention_origin"]["stage"] == "retention-origin"
    assert dan_before["retention_origin_snapshot_sha256"] == dan_before["snapshots"]["retention_origin"]["sha256"]
    assert dan_before["passive_projection"]["t60000ms"]["source_stage"] == "retention-origin"
    assert dan_before["passive_projection"]["t60000ms"]["formula"]["duration_ms"] == 59_500
    assert dan_before["passive_projection"]["t60000ms"]["source_snapshot_sha256"] == dan_before["retention_origin_snapshot_sha256"]
    assert dan_before["production_eligibility_status"] == "INCONCLUSIVE_NO_ELIGIBILITY"
    assert dan_before["production_eligibility_edge_count"] == 0
    assert dan_before["assay_direct_low_level_changed_edges"] == dan_before["pulse_delta"]["changed_edges"]
    did = result["temporal_did"]["t60000ms"]["dan-before-kc"]
    assert did["raw_delta"]["median_signed_delta"] == 0.0
    assert did["matched_none_did"]["median_signed_delta"] == 0.0
    assert result["branches"]["reward"]["eligible_edge_count"] == 3_900
    assert result["branches"]["reward"]["inconclusive_no_eligibility"] is False
    assert result["temporal"]["dan-before-kc"]["inconclusive_no_eligibility"] is True
    assert result["checks"]["temporal_reward_inconclusive_no_eligibility"] is False
    assert "stimulation_specific_delta" in result["branches"]["reward"]
    assert "stimulation_specific_delta" in result["branches"]["aversive"]
    assert set(result["temporal_none"]) == {"kc-before-dan", "dan-before-kc"}
    assert len({item["clean_checkpoint_sha256"] for item in result["branches"].values()}) == 1
    assert len({item["input_sha256"] for item in result["branches"].values()}) == 1
    post = result["branches"]["reward"]["snapshots"]["post_pulse"]
    with np.load(output / post["file"], allow_pickle=False) as arrays:
        assert arrays["edge_weights"].shape == (7_835,)
        assert arrays["baseline_weights"].shape == (7_835,)
        assert arrays["edge_eligibility"].shape == (7_835,)
        assert arrays["dan_rates"].shape == (17,)
        assert arrays["selected_gain_rows"].shape == (17, 7_835)
    assert verify_append_only_log(output / "assay-events.jsonl")["records"] > 0
    assert (output / "summary.json").is_file()


def test_short_assay_does_not_tune_missing_preregistered_horizons(tmp_path):
    result = run_assay(tmp_path / "short", controller_factory=_FakeController, horizons_ms=(1_000,))
    assert result["status"] == "FAIL"
    assert result["checks"]["required_horizons"] is False
