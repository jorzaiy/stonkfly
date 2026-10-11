"""Automatic v7 Q1 gates, importing the formal v2 runner directly."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "q1_formal_v2_runner",
    ROOT / "experiments/a_share/run_v2_q1_hybrid_qualification_v2.py",
)
Q1 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(Q1)


def test_formal_runner_import_and_frozen_inputs():
    actual, p0 = Q1.fixed_inputs()
    assert p0["qualification"]["verdict"] == "READY_FOR_V2_Q1"
    assert Q1.OUT.name.endswith("qualification-v7")
    assert actual["data/graph.npz"] == Q1.FIXED["data/graph.npz"]


def test_classification_and_graph_contact_conservation():
    import pyarrow.feather as feather
    nodes = {k: np.asarray(v) for k, v in feather.read_table(
        ROOT / "data/normalized/neurons.feather"
    ).to_pydict().items()}
    layer, _ = Q1.classify(nodes)
    assert [int(np.sum(layer == i)) for i in range(6)] == [541, 4706, 7114, 2, 4064, 150273]
    assert int(np.isin(layer, [1, 2, 3]).sum()) == 11822
    _, graph = Q1.graph_fixture_v7()
    assert graph["pass"] and graph["normalized_alignment"]
    assert graph["contact_sum"] == 124177617


def test_all_analytic_gates_are_executed():
    result = Q1.analytic_gates_v7()
    required = {
        "A1_release_clamp", "A2_delay_full_edge", "A3_analytic_excitatory_inhibitory",
        "A4_four_connection_types", "A5_refractory_audit", "A6_zero_input_100ms",
        "A7_fine_dt_convergence",
    }
    assert required.issubset(result)
    assert result["pass"], result
    assert all(result[name]["pass"] for name in required)


def test_a2_is_per_edge_and_exact_clock():
    result = Q1.analytic_gates_v7()["A2_delay_full_edge"]
    assert result["edge_count"] == 5
    for row in result["rows"]:
        assert row["pass"]
        assert all(edge["event_count"] > 0 for edge in row["edges"])
        assert all(edge["expected_arrival_clock"] == edge["observed_arrival_clock"] for edge in row["edges"])
        assert row["first_target_voltage_influence_clock"] == row["expected_first_target_voltage_influence_clock"]


def test_a4_a5_and_a7_have_required_subgates():
    micro = Q1.analytic_gates_v7()
    a4 = micro["A4_four_connection_types"]
    assert {r["connection"] for r in a4["rows"]} == {
        "spiking_to_spiking", "spiking_to_graded", "graded_to_graded", "graded_to_spiking"
    }
    assert all(r["pass"] and r["count_ok"] and r["state_delta_nonzero"] for r in a4["rows"])
    a5 = micro["A5_refractory_audit"]
    assert len(a5["rows"]) == 3
    assert all(r["pass"] and r["g_graded_analytic"] and r["voltage_frozen"] and r["discrete_spike_refractory_drop"] for r in a5["rows"])
    a7 = micro["A7_fine_dt_convergence"]
    assert a7["nonzero_delivery"] and all(a7["invalid_csr_rejected"])


def test_empty_mask_micro_compatibility_all_semantics():
    result = Q1.compatibility_micro_suite()
    required = {"excitatory", "inhibitory", "self_edge", "weak_edge", "repeated_firing", "refractory_drop", "modulatory_source", "kc_adaptation"}
    assert required == set(result["required_cases"])
    assert result["pass"], result
    assert all(row["pass"] for row in result["rows"])


def test_csr_validation_rejects_malformed_graphs(tmp_path):
    from stonkfly.neural.v2_brain import HybridBrain

    def write(name, ptr, post, weight):
        path = tmp_path / f"{name}.npz"
        Q1._write_micro_graph(path, ptr, post, weight)
        return path

    invalid = [
        ([1, 1, 1], [1], [.8]), ([0, 2, 1], [1, 0], [.8, .7]),
        ([0, 1, 1], [2], [.8]), ([0, 1, 1], [1], [np.nan]),
    ]
    for i, (ptr, post, weight) in enumerate(invalid):
        with pytest.raises(ValueError):
            HybridBrain(write(str(i), ptr, post, weight), dt=.1, graded_mask=np.array([1, 0], np.uint8))
