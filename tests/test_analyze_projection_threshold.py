import importlib.util
import json
from pathlib import Path


_PATH = Path(__file__).parents[1] / "experiments/a_share/analyze_projection_threshold.py"
_SPEC = importlib.util.spec_from_file_location("analyze_projection_threshold", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
classify = _MODULE.classify
jaccard = _MODULE.jaccard
sample_event_indices = _MODULE.sample_event_indices
zero_error_thresholds = _MODULE.zero_error_thresholds
trace_projection_edges = _MODULE.trace_projection_edges
select_projection_activity_boundary = _MODULE.select_projection_activity_boundary

_SUM_PATH = Path(__file__).parents[1] / "experiments/a_share/summarize_projection_candidates.py"
_SUM_SPEC = importlib.util.spec_from_file_location("summarize_projection_candidates", _SUM_PATH)
_SUM_MODULE = importlib.util.module_from_spec(_SUM_SPEC)
_SUM_SPEC.loader.exec_module(_SUM_MODULE)
trace_candidate = _SUM_MODULE.trace_candidate


def test_group_definitions_are_explicit():
    assert classify(4) == "low"
    assert classify(10) == "low"
    assert classify(11) == "mid"
    assert classify(999) == "mid"
    assert classify(1000) == "high"


def test_stable_threshold_is_marked_sample_only_and_directional():
    result = zero_error_thresholds([1, 2, 5, 6], ["low", "low", "high", "high"])
    assert result == [{"direction": "value>=threshold_implies_high", "threshold": 3.5, "low_max": 2.0, "high_min": 5.0, "errors": 0, "sample_only": True}]


def test_jaccard_and_sampling_are_reproducible():
    assert jaccard({1, 2}, {2, 3}) == 1 / 3
    events = [{"reward": {"stimulus": x}} for x in ("reward", "aversive", "none")]
    assert sample_event_indices(events, sample_size=3, seed=42) == [0, 1, 2]


def test_edge_direction_and_hop_semantics_are_explicit():
    import numpy as np
    class Brain:
        ptr = np.array([0, 1, 2, 2, 2], dtype=np.int64)
        post = np.array([1, 2], dtype=np.int32)
        weight = np.array([2.0, -3.0], dtype=np.float32)
        ids = np.array([10, 20, 30, 40], dtype=np.int64)
    result = trace_projection_edges(Brain(), [0], {1}, {2})
    assert result["direct_1hop_projection_to_cb"]["node_count"] == 1
    assert result["direct_1hop_projection_to_kc"]["node_count"] == 0
    assert result["two_hop_projection_to_cb_to_kc"]["node_count"] == 1
    assert result["two_hop_projection_to_cb_to_kc"]["path_count"] == 1
    assert "projection -> cb_intrinsic -> KC" in result["two_hop_projection_to_cb_to_kc"]["hop_semantics"]


def test_boundary_selection_uses_projection_activity_not_kc_count():
    records = [
        {"event_index": 1, "group": "low", "active_kc_count": 4, "projection_active_count": 55},
        {"event_index": 2, "group": "low", "active_kc_count": 4, "projection_active_count": 74},
        {"event_index": 3, "group": "high", "active_kc_count": 1300, "projection_active_count": 86},
        {"event_index": 4, "group": "high", "active_kc_count": 1100, "projection_active_count": 90},
    ]
    low, high = select_projection_activity_boundary(records)
    assert low["event_index"] == 2
    assert high["event_index"] == 3


def test_candidate_connectivity_preserves_direct_and_two_hop_semantics():
    import numpy as np
    brain = {"ptr": np.array([0, 1, 2, 2]), "post": np.array([1, 2]), "weight": np.array([2.0, -3.0]), "ids": np.array([10, 20, 30, 40])}
    result = trace_candidate(0, brain, {1}, {2})
    assert result["direct_projection_to_cb"]["node_count"] == 1
    assert result["direct_projection_to_kc"]["node_count"] == 0
    assert result["two_hop_projection_to_cb_to_kc"]["path_count"] == 1
    assert result["two_hop_projection_to_cb_to_kc"]["hop_semantics"] == "projection -> cb_intrinsic -> KC"


def test_random_sample_indices_are_python_int_json_safe():
    events = [{"reward": {"stimulus": stim}} for stim in ("reward", "aversive", "none") for _ in range(10)]
    indices = sample_event_indices(events, sample_size=6, seed=42)
    assert all(type(index) is int for index in indices)
    json.dumps({"parameters": {"event_indices": indices}}, allow_nan=False)
