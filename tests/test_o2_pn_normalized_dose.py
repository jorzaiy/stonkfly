from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("o2", ROOT / "experiments/a_share/run_o2_pn_normalized_dose.py")
O2 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(O2)

from stonkfly.neural.v2_brain import HybridBrain


def graph(tmp_path):
    path = tmp_path / "g.npz"
    np.savez(
        path,
        ptr=np.array([0, 3, 3, 3], np.int64),
        post=np.array([1, 1, 2], np.int32),
        weight=np.array([1.0, 2.0, -1.0], np.float32),
        ids=np.array([11, 12, 13], np.int64),
    )
    return path


def test_normalized_release_preserves_edges_signs_and_duplicate_sum(tmp_path):
    brain = HybridBrain(graph(tmp_path), dt=0.1, graded_mask=np.zeros(3, np.uint8))
    before = O2.arrsha(brain.weight)
    intent = O2.inject_normalized_release(brain, 0, 40.0)
    assert intent["edge_count"] == 3 and intent["unique_target_count"] == 2
    assert intent["edge_delivery_l1"] == pytest.approx(40.0)
    assert intent["target_delivery_l1"] == pytest.approx(40.0)
    assert brain.g_event.tolist() == pytest.approx([0.0, 30.0, -10.0])
    assert brain.active_flag[1] and brain.active_flag[2]
    assert O2.arrsha(brain.weight) == before


def test_normalized_release_rejects_invalid_or_dirty_input(tmp_path):
    brain = HybridBrain(graph(tmp_path), dt=0.1, graded_mask=np.zeros(3, np.uint8))
    with pytest.raises(ValueError):
        O2.inject_normalized_release(brain, 3, 1.0)
    with pytest.raises(ValueError):
        O2.inject_normalized_release(brain, 0, 0.0)
    O2.inject_normalized_release(brain, 0, 1.0)
    with pytest.raises(RuntimeError):
        O2.inject_normalized_release(brain, 0, 1.0)


def test_fixed_classification_boundaries():
    assert O2.classify(False, [], 100) == "unstable"
    assert O2.classify(True, [], 100) == "silent"
    assert O2.classify(True, list(range(20)), 100) == "stable_nonzero"
    assert O2.classify(True, list(range(21)), 100) == "dense_nonzero"
    assert O2.jaccard([], []) == 1.0
    assert O2.jaccard([1, 2], [2, 3]) == pytest.approx(1 / 3)


def synthetic_row(dt, source, dose, repeat, response, ids):
    zero = source is None
    intent = {"kind": "ZERO", "dose_l1": 0.0} if zero else {
        "kind": "engineered_pn_efferent_release_proxy", "edge_delivery_l1": dose,
        "target_delivery_l1": dose, "dose_l1": dose,
    }
    return {
        "dt_ms": dt, "source_node_index": source, "dose_l1": dose, "repeat": repeat,
        "response_class": "zero" if zero else response, "early_kc_ids": ids,
        "segments": [{"spikes": 0}], "intent": intent, "final_state_sha256": f"{dt}:{source}:{dose}",
        "weights_unchanged": True, "zero_ok": True, "finite": True, "bounds": True, "stable_exit": True,
    }


def test_evaluate_finds_only_preregistered_candidate():
    panel = [{"node_index": i} for i in range(8)]
    rows = []
    for dt in O2.DT:
        for repeat in (1, 2):
            rows.append(synthetic_row(dt, None, None, repeat, "zero", []))
            for source in range(8):
                for dose in O2.DOSES:
                    responsive = dose == 128.0 and source < 4
                    rows.append(synthetic_row(dt, source, dose, repeat, "stable_nonzero" if responsive else "silent", [100 + source] if responsive else []))
    gates = O2.evaluate(rows, panel, O2.DOSES, formal=True)
    assert gates["trial_count"] == 294
    assert gates["implementation"] and gates["deterministic_repeats"]
    assert gates["candidate_doses"] == [128.0]


def test_fixed_o1_stop_and_panel_pins():
    _, o0, o1 = O2.fixed()
    assert o0["panel"]["panel_sha256"] == O2.PANEL_SHA
    assert o1["verdict"] == "O1_SINGLE_EVENT_INSTABILITY"
    assert o1["gates"]["trial_count"] == 54


def test_cli_smoke_and_output_protection(tmp_path):
    out = tmp_path / "smoke"
    result = subprocess.run(
        [sys.executable, str(ROOT / "experiments/a_share/run_o2_pn_normalized_dose.py"), "--smoke", "--out", str(out)],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    assert "O2_SMOKE_OK" in result.stdout
    assert {x.name for x in out.iterdir()} == {"o2_raw.npz", "o2_formal.json", "o2_report.md", "o2_manifest.json", "launch.txt", "stdout.txt"}
    formal = json.loads((out / "o2_formal.json").read_text())
    manifest = json.loads((out / "o2_manifest.json").read_text())
    assert formal["counterfactual"] and formal["o1_route_closed"] and not formal["production_eligible"]
    assert formal["verdict"] == "O2_SMOKE_OK"
    assert manifest["hashes_start"] == manifest["hashes_end"]
    with pytest.raises(RuntimeError):
        O2.prepare(out)
