"""Fast protocol/isolated-wrapper tests for V2-C0.

These tests intentionally avoid the retained 166k-node run.  The formal
processes perform the large calibration/holdout runs only after this suite
passes.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stonkfly.neural.c0_common import (
    D_CAL,
    D_HOLD,
    GAINS,
    calibration_ramp_count,
    calibration_trial_count,
    calibration_trials_per_gain,
    formal_verdict,
    holdout_trial_count,
    jaccard,
    selection_verdict,
    spearman,
    symmetric_relative_difference,
)
from stonkfly.neural.v2_c0 import C0HybridBrain


def test_frozen_input_pins_and_q1_qualification():
    from experiments.a_share.run_v2_c0_calibration import fixed_inputs
    result = fixed_inputs()
    assert result["q1_verdict"] == "Q1_IMPLEMENTATION_QUALIFIED"
    assert result["pins"]["q1_formal.json"] == "394c2cd05ecc54c979b9d871176a86bb82670c850c108fe91c63ba8354ead574"
    assert result["pins"]["q1_raw.npz"] == "15623018571bd78437779236c89effd8542ad56df19407d2dbb9221b013e0c20"


def test_metric_conventions_and_tie_rank():
    assert symmetric_relative_difference(0, 0) == 0
    assert symmetric_relative_difference(0, 2) == 2
    assert jaccard([], []) == 1
    assert jaccard([], [1]) == 0
    assert spearman([0, 1, 1, 2], [0, 2, 2, 4]) == pytest.approx(1.0)


def test_trial_counts_selection_and_precedence():
    assert D_CAL == (0, 1, 2, 3, 4, 5, 6)
    assert D_HOLD == (0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5)
    assert GAINS == (0.125, 0.25, 0.5, 1.0, 2.0)
    assert calibration_trials_per_gain() == 18
    assert calibration_trial_count() == 90
    assert calibration_ramp_count() == 10
    assert holdout_trial_count() == 11
    assert selection_verdict([])["calibration_verdict"] == "C0_NO_QUALIFYING_GAIN"
    assert selection_verdict([.125])["calibration_verdict"] == "C0_GRID_BOUNDARY"
    assert selection_verdict([.5, 1.0])["selected_gamma"] == 1.0
    assert formal_verdict(implementation_ok=False, numerical_ok=False, calibration_verdict="C0_NO_QUALIFYING_GAIN", holdout_ok=None) == "C0_IMPLEMENTATION_FAILURE"
    assert formal_verdict(implementation_ok=True, numerical_ok=False, calibration_verdict="C0_NO_QUALIFYING_GAIN", holdout_ok=None) == "C0_NUMERICAL_FAILURE"
    assert formal_verdict(implementation_ok=True, numerical_ok=True, calibration_verdict="C0_NO_QUALIFYING_GAIN", holdout_ok=None) == "C0_NO_QUALIFYING_GAIN"
    assert formal_verdict(implementation_ok=True, numerical_ok=True, calibration_verdict="C0_GRID_BOUNDARY", holdout_ok=None) == "C0_GRID_BOUNDARY"
    assert formal_verdict(implementation_ok=True, numerical_ok=True, calibration_verdict="C0_SELECTION_READY", holdout_ok=False) == "C0_HOLDOUT_FAILURE"


def test_mask_families_are_disjoint_in_process_modules():
    from experiments.a_share.run_v2_c0_calibration import calibration_masks, load_graph_and_classification
    from experiments.a_share.run_v2_c0_holdout import holdout_masks
    _, ids, layer, _ = load_graph_and_classification()
    cal = calibration_masks(ids, layer); hold = holdout_masks(ids, layer)
    assert set(cal) == {"CAL_A", "CAL_B"}
    assert set(hold) == {"HOLD_A25", "HOLD_B50", "HOLD_C75", "HOLD_D50"}
    assert len(cal["CAL_A"]) == len(cal["CAL_B"]) == 2353
    assert len(hold["HOLD_A25"]) == 1176
    assert len(hold["HOLD_B50"]) == len(hold["HOLD_D50"]) == 2353
    cal_source = Path("experiments/a_share/run_v2_c0_calibration.py").read_text()
    assert "HOLD_" not in cal_source and "run_v2_c0_holdout" not in cal_source
    assert "20261010T-graded-release-calibration-v2" in cal_source
    assert '"expected_trial_count": 100' not in cal_source


def test_calibration_output_protection_and_precedence_helpers(tmp_path):
    from experiments.a_share.run_v2_c0_calibration import _ensure_fresh_output
    out = tmp_path / "already-existing"
    out.mkdir(); (out / "sentinel").write_text("do not overwrite")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        _ensure_fresh_output(out)
    assert formal_verdict(implementation_ok=False, numerical_ok=True,
                          calibration_verdict="C0_NO_QUALIFYING_GAIN",
                          holdout_ok=None) == "C0_IMPLEMENTATION_FAILURE"


def test_calibration_run_writes_report_manifest_and_formal(tmp_path, monkeypatch):
    """Exercise the real artifact-writing path without a 166k-node run."""
    import experiments.a_share.run_v2_c0_calibration as cal

    class _Column:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=object)

        def to_numpy(self, zero_copy_only=False):
            return self.values

    class _Table:
        def column(self, name):
            assert name == "cell_type"
            return _Column(["MBON07", "MBON11", "x", "x", "x", "x"])

    graph = {"ptr": np.array([0, 1, 2, 2, 2, 2, 2], np.int64),
             "post": np.array([1, 2], np.int32),
             "weight": np.array([1., 1.], np.float32),
             "ids": np.arange(6, dtype=np.int64)}
    layer = np.array([1, 2, 3, 4, 5, 5], np.int8)
    names = ("modulatory_declared", "graded_candidate_photoreceptor", "graded_candidate_lamina",
             "graded_candidate_apl", "adaptive_spiking_kc", "generic_spiking_declared")
    monkeypatch.setattr(cal, "fixed_inputs", lambda: {"pins": {}, "q1_verdict": "Q1_IMPLEMENTATION_QUALIFIED", "q1_formal_sha256": "stub"})
    monkeypatch.setattr(cal, "load_graph_and_classification", lambda: (graph, graph["ids"], layer, names))
    monkeypatch.setattr(cal.feather, "read_table", lambda path: _Table())
    monkeypatch.setattr(cal, "build", lambda: {"source": "stub.cpp", "binary": "stub.so"})
    monkeypatch.setattr(cal, "_hash_snapshot", lambda build_info: {"stub": "same"})
    monkeypatch.setattr(cal, "_sha", lambda path: "stub-sha")
    monkeypatch.setattr(cal, "transfer_chain_diagnostic", lambda *args: {"schema": "v2-c0-transfer-diagnostic-v1", "conclusion": "stub", "layers": {"adaptive_spiking_kc": {"measurement_spikes": 0}}})
    monkeypatch.setattr(cal, "calibration_masks", lambda ids, layer: {"CAL_A": ids[:1], "CAL_B": ids[:1]})

    def fake_trial(gamma, stimulus, selected, graph, ids, layer, mb07, mb11, repeat):
        row = {"gamma": gamma, "stimulus": stimulus, "repeat": repeat, "dt_ms": .05,
               "measurement_active_kc": 0, "post_active_kc": 0, "measurement_active_kc_ids": [],
               "mb07_active": 0, "mb11_active": 0, "mb07_spikes": 0, "mb11_spikes": 0,
               "finite": True, "bounds": True, "queue_integrity": True, "overflow": 0,
               "state_sha256": f"{gamma}-{stimulus}-{repeat}", "weights_sha256": "w"}
        return row, np.zeros(1, np.uint8)

    monkeypatch.setattr(cal, "_run_trial_with_mbon", fake_trial)
    monkeypatch.setattr(cal, "_run_history", lambda gamma, ids, layer, mb07, mb11, repeat: {"gamma": gamma, "repeat": repeat, "state_sha256": f"{gamma}-r", "post_active_kc": 0})
    gate = {"pass": True, "basic": {"pass": True}, "uniform_dose_response": {"pass": True}, "washout": {"pass": True}, "mbon_drivable": {"pass": True}, "calibration_identity": {"pass": True}, "history_dependence": {"pass": True}}
    monkeypatch.setattr(cal, "_evaluate_gain", lambda *args: gate)

    out = tmp_path / "c0-artifacts"
    result = cal.run(out)
    assert result["verdict"] == "C0_SELECTION_READY" or result["verdict"] == "C0_HOLDOUT_FAILURE"
    assert (out / "calibration" / "c0_calibration.json").is_file()
    assert (out / "calibration" / "c0_calibration_raw.npz").is_file()
    assert (out / "c0_formal.json").is_file()
    assert (out / "c0_manifest.json").is_file()
    assert (out / "c0_report.md").is_file()
    assert "90/90" in (out / "c0_report.md").read_text()


def test_runner_failure_writes_implementation_formal(tmp_path):
    from experiments.a_share.run_v2_c0_calibration import _write_failure_formal
    out = tmp_path / "failed-c0"
    (out / "calibration").mkdir(parents=True)
    (out / "c0_formal.json").write_text(json.dumps({"verdict": "C0_NO_QUALIFYING_GAIN"}))
    failure = _write_failure_formal(out, TypeError("artifact write failed"))
    assert failure["verdict"] == "C0_IMPLEMENTATION_FAILURE"
    saved = json.loads((out / "c0_formal.json").read_text())
    assert saved["verdict"] == "C0_IMPLEMENTATION_FAILURE"
    assert saved["holdout_not_run_reason"]


def test_c0_gain_is_global_counterfactual_and_weights_restore(tmp_path):
    graph = tmp_path / "tiny.npz"
    np.savez(graph, ptr=np.array([0, 1, 1], np.int64), post=np.array([1], np.int32), weight=np.array([.8], np.float32), ids=np.array([1, 2], np.int64))
    kwargs = {"dt": .05, "graded_mask": np.array([1, 0], np.uint8), "rest": np.array([-52., -52.], np.float32), "compile": True}
    one = C0HybridBrain(graph, gamma=1.0, **kwargs); half = C0HybridBrain(graph, gamma=.5, **kwargs)
    before = one.weight.copy(); drive = np.array([8., 0.], np.float32)
    one.run(10., drive); half.run(10., drive)
    assert np.array_equal(one.weight, before)
    assert half.signed_delivery[0] == pytest.approx(one.signed_delivery[0] * .5, rel=1e-5)
    assert one.counterfactual_calibration and not one.production_eligible


def test_transfer_chain_current_release_delivery_and_downstream_response(tmp_path):
    """A tiny frozen-rule chain catches a disconnected C0 runner wiring."""
    graph = tmp_path / "chain.npz"
    np.savez(graph,
             ptr=np.array([0, 1, 2, 2], np.int64),
             post=np.array([1, 2], np.int32),
             weight=np.array([1.0, 1.0], np.float32),
             ids=np.array([101, 102, 103], np.int64),
             retina=np.empty(0, np.int32), lamina=np.empty(0, np.int32),
             sugar=np.empty(0, np.int32), superclass=np.array(["photo", "lamina", "kc"]))
    graded = np.array([1, 1, 0], np.uint8)
    b = C0HybridBrain(graph, gamma=1.0, dt=.05,
                      graded_mask=graded, rest=np.array([-52., -52., -60.], np.float32))
    b.kc_mask[:] = np.array([0, 0, 1], np.uint8)
    zero = np.zeros(3, np.float32)
    b.run(5., zero)
    baseline = b.v.copy()
    drive = np.array([8., 0., 0.], np.float32)
    counts = b.run(10., drive)
    assert float(b.release_integral[0]) > 0.0
    assert float(b.absolute_delivery[0]) > 0.0
    assert float(b.v[1] - baseline[1]) != 0.0
    assert float(b.v[2] - baseline[2]) != 0.0
    assert int(counts[2]) == 0  # downstream voltage response need not imply a KC spike


def test_holdout_rejects_cli_gain_override():
    from experiments.a_share.run_v2_c0_holdout import parse_args
    with pytest.raises(SystemExit, match="rejects CLI gain overrides"):
        parse_args(["--gamma", "1.0"])


def test_initial_perturbations_are_deterministic_and_active():
    from experiments.a_share.run_v2_c0_holdout import initial_condition
    rest = np.full(8, -52., np.float32); graded = np.array([True, False, False, False, True, False, False, False]); base = np.array([0, 4], np.int64)
    a = initial_condition("P1", rest, graded, base); b = initial_condition("P1", rest, graded, base)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert set(np.flatnonzero(a[0] != rest)).issubset(set(a[1]))
    assert a[2]["seed"] == 202610091
