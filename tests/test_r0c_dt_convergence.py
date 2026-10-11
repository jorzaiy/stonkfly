import pytest
import json
from pathlib import Path

from experiments.a_share.run_r0c_dt_convergence import DT_VALUES, hist, ignition_2s, ticks
from stonkfly.config import Settings
from stonkfly.neural.v51_controller import V51Controller


def test_frozen_dt_values_and_ticks():
    assert DT_VALUES == (0.1, 0.05, 0.025)
    assert [ticks(1.8, dt) for dt in DT_VALUES] == [18, 36, 72]
    assert [ticks(2.2, dt) for dt in DT_VALUES] == [22, 44, 88]
    for ms in (10.0, 2000.0, 1000.0):
        assert [ticks(ms, dt) for dt in DT_VALUES] == [int(ms / dt) for dt in DT_VALUES]


def test_non_frozen_duration_rejected():
    with pytest.raises(ValueError):
        ticks(1.0, 0.03)


def test_three_second_ignition_is_not_two_second_ignition():
    assert ignition_2s(1900) == 1900
    assert ignition_2s(1990) == 1990
    assert ignition_2s(2000) is None
    assert ignition_2s(2830) is None
    assert ignition_2s(None) is None


def test_histogram_schema_matches_r0b():
    got = hist(__import__("numpy").asarray([-81.0, -79.5, -44.0], dtype="float32"))
    assert tuple(got) == ("bin_width_mV", "range_min_mV", "range_max_mV", "num_bins", "counts", "underflow", "overflow", "total")
    assert got["bin_width_mV"] == 1.0
    assert got["range_min_mV"] == -80.0
    assert got["range_max_mV"] == -44.0
    assert got["num_bins"] == 36
    assert got["counts"] == [1] + [0] * 35
    assert got["underflow"] == 1 and got["overflow"] == 1 and got["total"] == 3
    root = Path(__file__).resolve().parents[1]
    old = json.loads((root / "runs/plasticity_diagnosis/R0b/20261007T-absorbing-state-v1/r0b_formal.json").read_text())
    ref = next(r["v_histogram_cb"] for r in old["records"] if r["protocol"] == "perturbation_c")
    assert set(got) == set(ref)


def test_real_controller_dt_paths_and_production_default():
    settings = Settings(neural_ms=500, pulse_ms=200, learning=False)
    prod = V51Controller(settings)
    assert prod.brain.dt == 0.1
    production_signature = prod.brain.configuration_signature()
    for dt in DT_VALUES:
        diagnostic = V51Controller(settings, diagnostic_dt_ms=dt)
        assert diagnostic.brain.dt == dt
        assert diagnostic.brain.queue.shape[0] == ticks(1.8, dt) + 1
        assert diagnostic.brain.refractory.dtype.name == "int16"
        assert diagnostic.brain.configuration_signature() == production_signature
    with pytest.raises(ValueError):
        V51Controller(settings, diagnostic_dt_ms=0.03)
