import json

from experiments.a_share.run_r0_synthetic_dose_response import (
    CONTRAST_AMPLITUDES,
    UNIFORM_LEVELS,
    contrast_frame,
    independent_plan,
    stimuli,
    set_hash,
    uniform_frame,
)
from experiments.a_share.run_s2i_calibration_grid import digest_ids


def test_frozen_stimulus_shapes_and_contrast_mean():
    assert uniform_frame(40).shape == (64, 64, 3)
    assert uniform_frame(40).dtype.name == "uint8"
    assert uniform_frame(40).flags.c_contiguous
    for amplitude in CONTRAST_AMPLITUDES:
        frame = contrast_frame(amplitude)
        assert frame.shape == (64, 64, 3)
        assert frame.dtype.name == "uint8"
        assert frame.flags.c_contiguous
        assert frame.mean() == 60


def test_frozen_grid_and_independent_count():
    items = stimuli()
    assert [x["level"] for x in items if x["family"] == "uniform"] == list(UNIFORM_LEVELS)
    assert [x["level"] for x in items if x["family"] == "uniform_anchor"] == [20, 160, 255]
    assert [x["level"] for x in items if x["family"] == "contrast"] == list(CONTRAST_AMPLITUDES)
    plan = independent_plan(items)
    assert len(plan) == 66
    assert all(item["frame"].flags.c_contiguous for item, *_ in plan)


def test_runner_has_no_market_or_action_schema_fields():
    path = "runs/plasticity_diagnosis/R0/20261007T-synthetic-dose-response-smoke-v2/r0_smoke.json"
    try:
        result = json.loads(open(path).read())
    except FileNotFoundError:
        return
    forbidden = {"market", "reward", "action", "return", "decoder", "reinforcement"}
    assert not forbidden.intersection(result)
    assert not forbidden.intersection(result["records"][0])


def test_nonempty_kc_set_hash_matches_s2i_graph_index_digest():
    # S2i's digest_ids receives authoritative circuit indices, not body IDs.
    active_graph_indices = [3, 19, 4063]
    assert set_hash(active_graph_indices) == digest_ids(active_graph_indices)
    body_ids = [10540, 10977, 515338]
    assert set_hash(active_graph_indices) != digest_ids(body_ids)
