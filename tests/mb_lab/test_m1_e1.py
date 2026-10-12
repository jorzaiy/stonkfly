from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stonkfly.mb_lab.dynamics import M1Parameters, simulate
from stonkfly.mb_lab.e1 import APLMapping, build_apl_mapping, population_sparseness, simulate_e1
from stonkfly.mb_lab.inventory import load_source, selected_nodes
from stonkfly.mb_lab.topology import BipartiteTopology, compile_real_topology
from experiments.mb_lab.run_m1_e1 import arm_statistics, cross_dt

ROOT = Path(__file__).resolve().parents[2]


def topology() -> BipartiteTopology:
    return BipartiteTopology(
        4,
        8,
        np.repeat(np.arange(4, dtype=np.int32), 2),
        np.arange(8, dtype=np.int32),
        np.arange(4, dtype=np.int32),
        np.arange(8, dtype=np.int32),
    )


def mapping() -> APLMapping:
    groups = np.repeat(np.arange(4, dtype=np.int8), 2)
    result = APLMapping(
        99,
        10540,
        groups,
        ("gamma", "alpha_beta", "alpha_prime_beta_prime", "untyped"),
        np.ones(8),
        np.ones(8),
        np.ones(8),
    )
    result.validate(8)
    return result


def parameters() -> M1Parameters:
    return M1Parameters(10.0, 50.0, 1.0, 1.0, 0.35, 25.0)


def test_global_and_off_match_original_equations_exactly():
    graph = topology()
    pulse = np.asarray([1.0, 0.8, 0.5, 0.2])
    for model, enabled in (("G", True), ("O", False)):
        candidate = simulate_e1(
            graph, parameters(), mapping(), model=model, dt_ms=0.1,
            duration_ms=2000.0, pulse=pulse,
        )
        zero = np.zeros(4)
        reference = simulate(
            graph, parameters(), dt_ms=0.1, duration_ms=2000.0,
            pn_rate=lambda time: pulse if 500 <= time < 1000 else zero,
            apl_enabled=enabled, integration_window_ms=(500, 1000),
        )
        np.testing.assert_array_equal(candidate.integrated_rate, reference.integrated_rate)
        np.testing.assert_array_equal(candidate.final_u, reference.final_u)
        if enabled:
            assert candidate.final_apl[0] == reference.final_apl
        else:
            assert candidate.final_apl[0] == 0


def test_local_compartments_are_independent_and_finite():
    pulse = np.asarray([1.0, 0.8, 0.5, 0.2])
    local = simulate_e1(
        topology(), parameters(), mapping(), model="L", dt_ms=0.1,
        duration_ms=2000.0, pulse=pulse,
    )
    assert local.finite and local.final_apl.shape == (4,)
    assert local.apl_trace.shape == (200, 4)
    assert np.all(np.max(local.apl_trace, axis=0)[:-1] > 0)
    assert np.max(local.apl_trace[:, 3]) == 0.0


def test_local_cached_drive_matches_independent_uncached_oracle():
    graph, p, apl_map = topology(), parameters(), mapping()
    pulse = np.asarray([1.0, 0.8, 0.5, 0.2])
    candidate = simulate_e1(
        graph, p, apl_map, model="L", dt_ms=0.1, duration_ms=100.0,
        pulse=pulse, pulse_window_ms=(20.0, 60.0), trace_bin_ms=10.0,
    )
    u = np.zeros(graph.n_kc)
    apl = np.zeros(4)
    integrated = np.zeros(graph.n_kc)
    contact_sums = np.bincount(
        apl_map.group_index, weights=apl_map.kc_to_apl_contacts, minlength=4
    )
    for step in range(1000):
        time_ms = step * 0.1
        pn = pulse.copy() if 20.0 <= time_ms < 60.0 else np.zeros(4)
        drive = graph.project(pn)  # deliberately uncached independent projection
        inhibition = p.apl_gain * apl_map.output_multiplier * apl[apl_map.group_index]
        u += 0.1 / p.tau_k_ms * (-u + p.input_scale * drive - inhibition)
        rate = np.maximum(u - p.threshold, 0.0)
        target = np.bincount(
            apl_map.group_index,
            weights=apl_map.kc_to_apl_contacts * rate,
            minlength=4,
        ) / contact_sums
        apl += 0.1 / p.tau_apl_ms * (-apl + target)
        if 20.0 <= time_ms < 60.0:
            integrated += rate * 0.1
    np.testing.assert_array_equal(candidate.integrated_rate, integrated)
    np.testing.assert_array_equal(candidate.final_u, u)
    np.testing.assert_array_equal(candidate.final_apl, apl)


def test_toy_cross_dt_converges_without_applying_a_science_gate():
    pulse = np.asarray([1.0, 0.8, 0.5, 0.2])
    coarse = simulate_e1(
        topology(), parameters(), mapping(), model="L", dt_ms=0.1,
        duration_ms=100.0, pulse=pulse, pulse_window_ms=(20.0, 60.0),
    )
    fine = simulate_e1(
        topology(), parameters(), mapping(), model="L", dt_ms=0.05,
        duration_ms=100.0, pulse=pulse, pulse_window_ms=(20.0, 60.0),
    )
    relative_integral_error = np.max(np.abs(coarse.integrated_rate - fine.integrated_rate)) / np.max(
        np.abs(fine.integrated_rate)
    )
    assert relative_integral_error < 0.002
    assert np.max(np.abs(coarse.final_u - fine.final_u)) < 0.001


def test_zero_is_exact_for_all_three_models():
    for model in ("G", "L", "O"):
        result = simulate_e1(
            topology(), parameters(), mapping(), model=model, dt_ms=0.1,
            duration_ms=2000.0, pulse=np.zeros(4),
        )
        assert result.finite
        assert np.array_equal(result.integrated_rate, np.zeros(8))
        assert np.array_equal(result.population_trace, np.zeros(200))
        assert np.array_equal(result.final_u, np.zeros(8))


def test_real_apl_mapping_is_complete_and_group_normalized():
    source = load_source()
    pn, kc = selected_nodes("R", olfactory_covered_only=True)
    graph = compile_real_topology(source["ptr"], source["post"], pn, kc)
    result = build_apl_mapping(ROOT / "data/graph.npz", source, graph.kc_node_index)
    assert result.apl_source_id == 10540
    assert len(result.group_index) == 1907
    assert np.bincount(result.group_index, minlength=4).tolist() == [670, 884, 352, 1]
    assert int(np.rint(result.apl_to_kc_contacts).sum()) > 0
    assert int(np.rint(result.kc_to_apl_contacts).sum()) > 0
    for group in range(4):
        np.testing.assert_allclose(result.output_multiplier[result.group_index == group].mean(), 1.0)


def test_population_sparseness_bounds_and_config_has_no_old_gates():
    values = np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    sparse = population_sparseness(values)
    np.testing.assert_allclose(sparse, [1.0, 0.0])
    config = json.loads((ROOT / "protocols/mb_lab/m1_e1/config.json").read_text())
    serialized = json.dumps(config)
    assert config["status"] == "AUTHORIZED_EXPLORATORY_M1_E1_ONLY"
    assert "M1_RQ_QUALIFIED" not in serialized
    assert "M1_RT" not in serialized
    runner = (ROOT / "experiments/mb_lab/run_m1_e1.py").read_text().lower()
    assert "m1_rt" not in runner
    expected_paths = (
        len(config["models"]) * len(config["dt_ms"])
        * (config["stimulus"]["odor_count"] * config["stimulus"]["repeat_count"]
           + config["zero_repeats_per_model_dt"])
    )
    assert expected_paths == 1530


def test_formal_statistics_path_on_toy_arrays():
    response = np.arange(1, 1 + 2 * 2 * 8, dtype=float).reshape(2, 2, 8)
    trace = np.zeros((2, 2, 200), dtype=float)
    trace[..., 50:100] = 0.2
    trace[..., 100:110] = np.linspace(0.2, 0.0, 10)
    chunk = {
        "response_integral": response,
        "population_trace": trace,
        "group_trace": np.zeros((2, 2, 200, 4)),
        "apl_trace": np.zeros((2, 2, 200, 1)),
        "final_state_norm": np.zeros((2, 2)),
        "finite": np.ones((2, 2), dtype=bool),
    }
    timeline = {"pulse_start": 500.0, "pulse_end": 1000.0, "trace_bin": 10.0}
    summary, rsm, rate = arm_statistics(chunk, parameters(), 10, timeline)
    assert rsm.shape == (2, 2) and rate.shape == (2, 2, 8)
    assert summary["active_fraction_1907"]["max"] <= 1.0
    comparison = cross_dt(rate * 1.001, rate, 0.05)
    assert comparison["nrmse"]["max"] < 0.01
