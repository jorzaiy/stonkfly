import importlib.util
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location("v2_p0_inventory", Path(__file__).parents[1] / "experiments/a_share/run_v2_p0_model_inventory.py")
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)
LAYER_INDEX = _MOD.LAYER_INDEX
classify_layers = _MOD.classify_layers
qualification = _MOD.qualification
history_lock = _MOD.history_lock
_all_fractions = _MOD._all_fractions


def test_declaration_priority_and_uncertain_fast_sign():
    cell_type = np.array(["R1-R6", "R8y", "L1", "APL", "KCg-m", "X", "X", "X"], dtype=object)
    transmitter = np.array(["dopamine", "acetylcholine", "gaba", "serotonin", "acetylcholine", "unclear", None, "gaba,acetylcholine"], dtype=object)
    layer, uncertain, sign = classify_layers(cell_type, transmitter)
    assert layer.tolist() == [
        LAYER_INDEX["modulatory_declared"],
        LAYER_INDEX["graded_candidate_photoreceptor"],
        LAYER_INDEX["graded_candidate_lamina"],
        LAYER_INDEX["modulatory_declared"],
        LAYER_INDEX["adaptive_spiking_kc"],
        LAYER_INDEX["generic_spiking_declared"],
        LAYER_INDEX["generic_spiking_declared"],
        LAYER_INDEX["generic_spiking_declared"],
    ]
    assert uncertain.tolist() == [True, False, False, True, False, True, True, True]
    assert sign.tolist() == [1, 1, -1, 1, 1, 1, 1, 1]


def test_qualification_requires_conservation_and_coverage():
    reach = {
        name: {
            "candidate_node_count": 1,
            "kc_reached_fraction": 1.0,
            "kc_reached_further_to_mbon_within_6_hops": {"MBON07": 1, "MBON11": 1},
        }
        for name in ("graded_candidate_photoreceptor", "graded_candidate_lamina", "graded_candidate_apl")
    }
    decomp = {name: {"uncertain_sign_absolute_weight_fraction": 0.0} for name in ("KC", "APL", "MBON07", "MBON11")}
    impl = {
        "all_nodes_assigned_once": True,
        "conservation": {"flow_edge_count": 3, "graph_edge_count": 3, "flow_contact_count": 7, "graph_contact_count": 7, "flow_signed_weight": 1.0, "graph_signed_weight": 1.0, "flow_absolute_weight": 3.0, "graph_absolute_weight": 3.0},
        "path_analysis": {"candidate_kc_reachability": reach, "graded_union_kc_reachability": {"kc_reached_fraction": 1.0}, "kc_union_to_mbon_within_6_hops": {"MBON07": {"reachable": True}, "MBON11": {"reachable": True}}},
        "target_input_decomposition": decomp,
    }
    assert qualification(impl)["verdict"] == "READY_FOR_V2_Q1"
    impl["conservation"]["flow_contact_count"] = 6
    assert qualification(impl)["verdict"] == "P0_IMPLEMENTATION_FAILURE"


def test_uncertain_sign_trigger_is_not_a_parameter_selection():
    reach = {
        name: {"candidate_node_count": 1, "kc_reached_fraction": 1.0, "kc_reached_further_to_mbon_within_6_hops": {"MBON07": 1, "MBON11": 1}}
        for name in ("graded_candidate_photoreceptor", "graded_candidate_lamina", "graded_candidate_apl")
    }
    impl = {
        "all_nodes_assigned_once": True,
        "conservation": {"flow_edge_count": 3, "graph_edge_count": 3, "flow_contact_count": 7, "graph_contact_count": 7, "flow_signed_weight": 1.0, "graph_signed_weight": 1.0, "flow_absolute_weight": 3.0, "graph_absolute_weight": 3.0},
        "path_analysis": {"candidate_kc_reachability": reach, "graded_union_kc_reachability": {"kc_reached_fraction": 1.0}, "kc_union_to_mbon_within_6_hops": {"MBON07": {"reachable": True}, "MBON11": {"reachable": True}}},
        "target_input_decomposition": {"KC": {"uncertain_sign_absolute_weight_fraction": 0.20}, "APL": {"uncertain_sign_absolute_weight_fraction": 0.0}, "MBON07": {"uncertain_sign_absolute_weight_fraction": 0.0}, "MBON11": {"uncertain_sign_absolute_weight_fraction": 0.0}},
    }
    result = qualification(impl)
    assert result["verdict"] == "READY_FOR_V2_Q1_WITH_SIGN_SENSITIVITY"
    assert result["model_gate"]["uncertain_sign_trigger"] is True


def test_fraction_gate_checks_bounds_and_finiteness():
    assert _all_fractions({"x_fraction": 0.0, "nested": [{"p_proportion": 1.0}]})
    assert not _all_fractions({"x_fraction": -1e-9})
    assert not _all_fractions({"x_fraction": 1.000001})
    assert not _all_fractions({"x_fraction": float("nan")})


def test_history_lock_uses_frozen_formal_json_paths():
    locked = history_lock()
    assert locked["R0"]["formal_verdict"] == "inconclusive"
    assert locked["R0b"]["formal_verdict"] == "inconclusive"
    assert locked["R0c"]["formal_verdict"] == "DT_CONVERGED_FOR_CURRENT_AUDIT"
    assert len(locked["R0c"]["dt_values_ms"]) == 3
