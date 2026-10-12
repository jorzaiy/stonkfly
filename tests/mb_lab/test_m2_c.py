from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.mb_lab.run_m2b_exploration import ARMS
from experiments.mb_lab.run_m2c_validation import (
    FORMAL_SEED_LABEL,
    SMOKE_SEED_LABEL,
    analyze,
    mode_values,
)

ROOT = Path(__file__).resolve().parents[2]


def test_m2c_keeps_neural_model_and_freezes_aggregate_and_single_readouts():
    current = json.loads((ROOT / "protocols/mb_lab/m2_c/config.json").read_text())
    prior = json.loads((ROOT / "protocols/mb_lab/m2_b/config.json").read_text())
    assert current["sensory"] == prior["sensory"]
    assert current["plasticity"] == prior["plasticity"]
    assert current["readout"]["aggregate_count"] == 10
    assert current["readout"]["single_control_repeat_index"] == 0
    assert current["readout"]["threshold"] == 0.0
    assert current["exploration"]["estimated_unique_sensory_paths"] == 520


def test_m2c_seed_domain_is_new_and_mode_dimensions_are_fixed():
    assert FORMAL_SEED_LABEL == "STONKFLY-V2/M2-C/READOUT-RELIABILITY/v1"
    assert FORMAL_SEED_LABEL != SMOKE_SEED_LABEL
    assert "M2-B" not in FORMAL_SEED_LABEL
    config = json.loads((ROOT / "protocols/mb_lab/m2_c/config.json").read_text())
    formal = mode_values(config, "formal")
    assert formal["pair_count"] == 4
    assert formal["calibration_repeats"] == 5
    assert formal["probe_repeats"] == 10
    assert formal["cycles"] == 10


def test_analysis_uses_all_repeats_and_repeat_zero_without_selection():
    chunks = []
    for pair in range(4):
        d = np.zeros((len(ARMS), 4, 10), dtype=np.float64)
        d[0, 0] = np.linspace(-.01, .01, 10)
        d[0, 1] = .10 + pair * .01
        d[0, 2] = -.08 - pair * .01
        d[0, 3] = -.07 - pair * .01
        choices = np.sign(d).astype(np.int8)
        efficacy = np.ones((len(ARMS), 4, 3), dtype=np.float64)
        efficacy[0, 1:] = .99
        efficacy[2, 1:] = .999
        efficacy[3, 1:] = .9999
        chunks.append({
            "choice_d": d,
            "choices": choices,
            "efficacy": efficacy,
            "finite": np.ones((len(ARMS), 4), dtype=bool),
            "bound_fraction": np.zeros((len(ARMS), 4), dtype=np.float64),
            "calibration_channels": np.ones((2, 5, 2), dtype=np.float64),
        })
    statistics, raw = analyze(chunks, FORMAL_SEED_LABEL, 3, 0)
    assert statistics["verdict"] == "ENGINEERED_AGGREGATE_PREFERENCE_DIRECTION_UPDATE_OBSERVED"
    assert statistics["paired_primary"]["direction_update_count"] == 4
    assert all(statistics["implementation_checks"].values())
    np.testing.assert_array_equal(raw["aggregate_D10"], raw["choice_d"].mean(axis=-1))
    np.testing.assert_array_equal(raw["single_choice"], raw["choices"][..., 0])
    assert statistics["arms"]["paired"]["pairs"]["0"]["P1"]["D10"] == .1


def test_analysis_rejects_two_of_four_aggregate_direction_updates():
    chunks = []
    for pair in range(4):
        d = np.zeros((len(ARMS), 4, 10), dtype=np.float64)
        d[0, 1] = .1
        d[0, 2] = -.1 if pair < 2 else .1
        d[0, 3] = -.1 if pair < 2 else .1
        efficacy = np.ones((len(ARMS), 4, 2), dtype=np.float64)
        efficacy[0, 1:] = .99
        chunks.append({
            "choice_d": d,
            "choices": np.sign(d).astype(np.int8),
            "efficacy": efficacy,
            "finite": np.ones((len(ARMS), 4), dtype=bool),
            "bound_fraction": np.zeros((len(ARMS), 4), dtype=np.float64),
            "calibration_channels": np.ones((2, 5, 2), dtype=np.float64),
        })
    statistics, _ = analyze(chunks, FORMAL_SEED_LABEL, 3, 0)
    assert statistics["paired_primary"]["direction_update_count"] == 2
    assert statistics["verdict"] == "ENGINEERED_AGGREGATE_PREFERENCE_DIRECTION_UPDATE_NOT_OBSERVED"
