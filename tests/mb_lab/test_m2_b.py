from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.mb_lab.run_m2b_exploration import (
    ARMS,
    FORMAL_SEED_LABEL,
    SMOKE_SEED_LABEL,
    analyze,
    build_bases,
    derived_seed,
    expected_paths,
    preference_index,
    schedule,
    stimulus_content_hash,
)
from stonkfly.mb_lab.m2e import normalized_valence

ROOT = Path(__file__).resolve().parents[2]


def test_m2b_config_preserves_m2e_model_and_freezes_zero_choice_rule():
    current = json.loads((ROOT / "protocols/mb_lab/m2_b/config.json").read_text())
    prior = json.loads((ROOT / "protocols/mb_lab/m2_e/config.json").read_text())
    assert current["status"] == "AUTHORIZED_EXPLORATION_ONCE"
    assert current["sensory"] == prior["sensory"]
    assert current["plasticity"] == prior["plasticity"]
    assert current["choice"]["threshold"] == 0.0
    assert current["choice"]["tie_rule"] == "exact_float_zero"
    assert current["exploration"]["odor_pair_count"] == 4
    assert expected_paths(4, 5, 10, 10) == 520


def test_new_seed_domain_is_deterministic_and_distinct_from_smoke_and_m2e():
    assert FORMAL_SEED_LABEL == "STONKFLY-V2/M2-B/VIRTUAL-2AFC/v1"
    assert FORMAL_SEED_LABEL != SMOKE_SEED_LABEL
    assert FORMAL_SEED_LABEL != "STONKFLY-V2/M2-E/EXPLORATION/v1"
    assert derived_seed(FORMAL_SEED_LABEL, "base", 0, 0) == derived_seed(
        FORMAL_SEED_LABEL, "base", 0, 0
    )
    assert derived_seed(FORMAL_SEED_LABEL, "base", 0, 0) != derived_seed(
        SMOKE_SEED_LABEL, "base", 0, 0
    )
    first = build_bases(FORMAL_SEED_LABEL, 4, 145, 29)
    second = build_bases(FORMAL_SEED_LABEL, 4, 145, 29)
    np.testing.assert_array_equal(first, second)
    assert np.all(np.count_nonzero(first, axis=-1) == 29)
    assert stimulus_content_hash(FORMAL_SEED_LABEL, first, 5, 10, 10) == stimulus_content_hash(
        FORMAL_SEED_LABEL, second, 5, 10, 10
    )


def test_balanced_schedule_and_choice_are_frozen():
    for pair in range(4):
        for phase in ("acquisition", "reversal"):
            item = schedule(FORMAL_SEED_LABEL, pair, phase, 10)
            assert len(item["orders"]) == 10
            for labels in item["shuffled"]:
                assert labels.count("reward") == labels.count("aversive") == 5
    d = np.asarray([1e-300, -1e-300, 0.0])
    choices = np.sign(d).astype(np.int8)
    assert choices.tolist() == [1, -1, 0]
    assert preference_index(choices) == {"A": 1, "B": 1, "tie": 1, "PI": 0.0}


def test_calibration_readout_is_fixed_and_not_refit_to_current_output():
    baseline = np.asarray([2.0, 1.0])
    assert normalized_valence(baseline, baseline) == 0.0
    learned = np.asarray([1.8, 1.0])
    assert normalized_valence(learned, baseline) > 0
    # Refitting the denominator would erase the learned score; M2-B must retain
    # the original calibration baseline for every later stage.
    assert normalized_valence(learned, learned) == 0.0


def test_analysis_uses_pair_level_direction_and_preserves_controls():
    chunks = []
    for pair in range(4):
        d = np.zeros((len(ARMS), 4, 4), dtype=np.float64)
        d[0, 0] = np.asarray([-.01, 0, .01, 0])
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
        })
    statistics, raw = analyze(chunks, FORMAL_SEED_LABEL, 3)
    assert statistics["verdict"] == "ENGINEERED_VIRTUAL_PREFERENCE_DIRECTION_UPDATE_OBSERVED"
    assert statistics["paired_primary"]["direction_update_count"] == 4
    assert all(statistics["implementation_checks"].values())
    assert raw["choice_d"].shape == (5, 4, 4, 4)
    assert statistics["paired_vs_controls"]["shuffled"]["acquisition_difference"]["mean"] > 0
    assert statistics["paired_vs_controls"]["unpaired"]["reversal_difference"]["mean"] < 0
