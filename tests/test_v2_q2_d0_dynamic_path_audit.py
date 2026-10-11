from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stonkfly.neural.q2_d0 import (
    bfs_shortest_hops,
    counterfactual_delivery,
    interpretation_flags,
    q1_positive_release,
    signed_deviation,
)
from stonkfly.neural.v2_c0 import C0HybridBrain


def test_candidate_c_matches_q1_above_rest_and_is_signed_below():
    voltage = np.array([-60.0, -52.0, -48.5, -45.0, -40.0])
    rest = np.full(len(voltage), -52.0)
    candidate = signed_deviation(voltage, rest)
    q1 = q1_positive_release(voltage, rest)
    assert candidate.tolist() == pytest.approx([-1.0, 0.0, 0.5, 1.0, 1.0])
    assert np.array_equal(candidate[voltage >= rest], q1[voltage >= rest])
    assert np.all(q1[voltage <= rest] == 0.0)


def test_four_quadrants_and_l1_l2_direction():
    # Sources 0/1 both have negative deviation.  L1-like negative weight
    # produces positive target change; L2-like positive weight produces
    # negative target change.
    ptr = np.array([0, 1, 2, 2, 2], np.int64)
    post = np.array([2, 3], np.int32)
    weight = np.array([-2.0, 3.0], np.float32)
    q = np.array([-0.5, -0.5, 0.0, 0.0])
    groups = np.array([2, 2, 5, 5])
    l1 = counterfactual_delivery(ptr, post, weight, np.array([0]), q, groups, 6)
    l2 = counterfactual_delivery(ptr, post, weight, np.array([1]), q, groups, 6)
    assert l1["target_signed"][2] == 1.0
    assert l2["target_signed"][3] == -1.5
    assert l1["positive_by_group"][5] == 1.0
    assert l2["negative_by_group"][5] == -1.5


def test_target_aggregation_keeps_cancellation_and_absolute_delivery_distinct():
    ptr = np.array([0, 1, 2, 2], np.int64)
    post = np.array([2, 2], np.int32)
    weight = np.array([2.0, -2.0], np.float32)
    q = np.array([0.5, 0.5, 0.0])
    groups = np.array([1, 2, 5])
    result = counterfactual_delivery(ptr, post, weight, np.array([0, 1]), q, groups, 6)
    assert result["target_signed"][2] == 0.0
    assert result["target_absolute"][2] == 2.0
    assert result["absolute_by_group"][5] == 2.0


def test_bfs_shortest_hops_and_unreached():
    # 0 -> 1 -> 2, 0 -> 3; node 4 is unreachable.
    ptr = np.array([0, 2, 3, 3, 3, 3], np.int64)
    post = np.array([1, 3, 2], np.int32)
    hops = bfs_shortest_hops(ptr, post, np.array([0]), max_hops=6)
    assert hops.tolist() == [0, 1, 2, 1, -1]


def test_interpretation_flags_do_not_hide_competing_bottleneck():
    flags = interpretation_flags(
        lamina_hyperpolarization_observed=True,
        lamina_q1_output_clamped_zero=True,
        counterfactual_lamina_reaches_spiking_downstream=True,
        counterfactual_lamina_positive_net_reaches_spiking_downstream=True,
        counterfactual_lamina_reaches_kc_directly=False,
        generic_spiking_received_input=True,
        generic_spiking_emitted_spikes=False,
    )
    assert flags["tonic_hypothesis_upgrade_condition"]
    assert flags["competing_spiking_bottleneck"]
    assert not flags["counterfactual_lamina_reaches_kc_directly"]


def test_observer_segmentation_does_not_change_tiny_dynamics(tmp_path):
    graph = tmp_path / "tiny.npz"
    np.savez(
        graph,
        ptr=np.array([0, 1, 2, 2], np.int64),
        post=np.array([1, 2], np.int32),
        weight=np.array([-1.0, 1.0], np.float32),
        ids=np.array([11, 12, 13], np.int64),
        retina=np.empty(0, np.int32), lamina=np.empty(0, np.int32),
        sugar=np.empty(0, np.int32), superclass=np.array(["p", "l", "s"]),
    )
    kwargs = dict(path=graph, gamma=1.0, dt=.05,
                  graded_mask=np.array([1, 1, 0], np.uint8),
                  rest=np.array([-52., -52., -52.], np.float32))
    whole = C0HybridBrain(**kwargs); split = C0HybridBrain(**kwargs)
    drive = np.array([4., 0., 0.], np.float32)
    whole.run(5.0, drive)
    for _ in range(100):
        split.run(.05, drive)
    for name in ("v", "g_event", "g_graded", "refractory", "drive",
                 "previous_drive", "spike_queue", "spike_queue_count",
                 "graded_queue", "graded_drive", "active", "active_flag"):
        assert np.array_equal(getattr(whole, name), getattr(split, name)), name
    assert int(whole.clock[0]) == int(split.clock[0])
    assert whole.release_integral[0] == split.release_integral[0]
    assert whole.signed_delivery[0] == split.signed_delivery[0]


def test_cli_smoke_writes_complete_artifact_chain(tmp_path):
    out = tmp_path / "smoke"
    runner = ROOT / "experiments/a_share/run_v2_q2_d0_dynamic_path_audit.py"
    result = subprocess.run(
        [sys.executable, str(runner), "--smoke", "--out", str(out)],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    assert "Q2_D0_SMOKE_OK" in result.stdout
    required = ("q2_d0_raw.npz", "q2_d0_formal.json", "q2_d0_report.md", "q2_d0_manifest.json")
    assert all((out / name).is_file() for name in required)
    manifest = json.loads((out / "q2_d0_manifest.json").read_text())
    assert set(manifest["artifacts"]) == set(required) - {"q2_d0_manifest.json"}
    assert json.loads((out / "q2_d0_formal.json").read_text())["verdict"] == "Q2_D0_SMOKE_OK"


def test_output_protection_and_failure_precedence(tmp_path):
    from experiments.a_share.run_v2_q2_d0_dynamic_path_audit import _prepare_output, write_failure
    out = tmp_path / "occupied"; out.mkdir(); (out / "sentinel").write_text("keep")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        _prepare_output(out)
    failed = tmp_path / "failed"
    value = write_failure(failed, "Q2_D0_IMPLEMENTATION_FAILURE", RuntimeError("boom"))
    assert value["verdict"] == "Q2_D0_IMPLEMENTATION_FAILURE"
    assert not value["q2_authorized"]
