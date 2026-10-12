from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from stonkfly.mb_lab.dynamics import M1Parameters, simulate
from stonkfly.mb_lab.metrics import (
    frozen_neighbor_matches,
    frozen_separation_statistic,
    jaccard,
    separation_permutation_pvalue,
    sparsity_views,
)
from stonkfly.mb_lab.protocol import (
    CONFIG_PATH,
    BudgetUsage,
    enforce_stage_budget,
    load_frozen_config,
    sha256_file,
    topology_control_seeds,
)
from stonkfly.mb_lab.stages import DataDomain, read_public_manifest_for_stage
from stonkfly.mb_lab.stimuli import collection_sha256, generate_m1_collection
from stonkfly.mb_lab.topology import BipartiteTopology

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments/mb_lab/run_m1_preregistered.py"
DOMAIN_ROOT = ROOT / "protocols/mb_lab/m1_v1/domains-freeze-v4"


def graph() -> BipartiteTopology:
    target = np.arange(64, dtype=np.int32)
    source = target % 8
    return BipartiteTopology(8, 64, source, target, np.arange(8), np.arange(64))


def frozen_parameters() -> M1Parameters:
    config = load_frozen_config()
    values = config["parameters"]
    return M1Parameters(
        float(values["tau_k_ms"]),
        float(values["tau_apl_ms"]),
        float(values["input_scale"]),
        float(values["apl_gain"]),
        float(values["threshold"]),
        float(values["active_mean_rate_threshold"]) * 500.0,
    )


def pulse(time_ms: float) -> np.ndarray:
    active = np.asarray([1.0, 0.9, 0.8, 0.7, 0.0, 0.0, 0.0, 0.0])
    return active if 500.0 <= time_ms < 1000.0 else np.zeros(8)


def test_unique_candidate_config_has_units_evidence_and_no_grid():
    config = load_frozen_config()
    assert config["status"] == "CANDIDATE_FROZEN_AWAITING_INDEPENDENT_GO_NO_GO"
    assert set(config["parameters"]) == set(config["parameter_evidence"])
    for evidence in config["parameter_evidence"].values():
        assert evidence["grade"] in {"V", "P", "M", "P(M)"}
        assert evidence["unit"] and evidence["reason"]
    assert all(not isinstance(value, list) for value in config["parameters"].values())


def test_stimulus_generation_is_deterministic_sparse_and_domain_sensitive():
    spec = load_frozen_config()["stimulus"]
    first = generate_m1_collection(11, spec)
    repeated = generate_m1_collection(11, spec)
    other = generate_m1_collection(12, spec)
    assert collection_sha256(first) == collection_sha256(repeated)
    assert collection_sha256(first) != collection_sha256(other)
    np.testing.assert_array_equal(np.count_nonzero(first["base"], axis=1), 29)
    assert first["repeated"].shape == (16, 10, 145)


def test_four_public_manifests_are_distinct_and_contain_no_seed_material():
    manifests = []
    for domain in (DataDomain.M1_RQ, DataDomain.M1_RT, DataDomain.M2_LQ, DataDomain.M2_LT):
        manifest = json.loads((DOMAIN_ROOT / f"{domain.value.lower()}-manifest.json").read_text())
        serialized = json.dumps(manifest).lower()
        assert manifest["domain"] == domain.value
        assert manifest["seed_exposed"] is False
        assert "seed_hex" not in serialized and "plaintext" not in serialized
        assert len(manifest["seed_commitment_sha256"]) == 64
        assert len(manifest["content_sha256"]) == 64
        assert manifest["config_sha256"] == sha256_file(CONFIG_PATH)
        assert manifest["sealer_sha256"] == sha256_file(
            ROOT / "experiments/mb_lab/seal_m1_domains.py"
        )
        assert manifest["stimulus_generator_sha256"] == sha256_file(
            ROOT / "stonkfly/mb_lab/stimuli.py"
        )
        for field in ("sealed_payload_relative_path", "key_relative_path"):
            private_path = ROOT / manifest[field]
            assert private_path.is_file()
            assert os.stat(private_path).st_mode & 0o777 == 0o600
        manifests.append(manifest)
    assert len({item["seed_commitment_sha256"] for item in manifests}) == 4
    assert len({item["content_sha256"] for item in manifests}) == 4


def test_cross_domain_manifest_read_is_denied(tmp_path):
    for domain in (DataDomain.M1_RQ, DataDomain.M1_RT):
        (tmp_path / f"{domain.value.lower()}-manifest.json").write_text(
            json.dumps({"domain": domain.value})
        )
    assert read_public_manifest_for_stage(
        tmp_path, authorized_domain=DataDomain.M1_RQ, requested_domain=DataDomain.M1_RQ
    )["domain"] == "M1_RQ"
    with pytest.raises(PermissionError, match="cross-domain"):
        read_public_manifest_for_stage(
            tmp_path, authorized_domain=DataDomain.M1_RQ, requested_domain=DataDomain.M1_RT
        )


def test_39_topology_seeds_and_degree_preserving_controls_are_fixed():
    config = load_frozen_config()
    seeds = topology_control_seeds(config)
    assert len(seeds) == len(set(seeds)) == 39
    base = graph()
    controls = [base.rewire(seed, swaps=base.edge_count) for seed in seeds]
    assert len({item.sha256 for item in controls}) == 39
    for control in controls:
        np.testing.assert_array_equal(control.pn_degree, base.pn_degree)
        np.testing.assert_array_equal(control.kc_degree, base.kc_degree)


def test_toy_vectorized_windowed_dynamics_matches_scalar_oracle():
    topology = graph()
    parameters = frozen_parameters()
    result = simulate(
        topology,
        parameters,
        dt_ms=1.0,
        duration_ms=2000.0,
        pn_rate=pulse,
        integration_window_ms=(500.0, 1000.0),
        tail_window_ms=(1500.0, 2000.0),
    )
    u = np.zeros(topology.n_kc)
    integrated = np.zeros(topology.n_kc)
    tail = np.zeros(topology.n_kc)
    apl = 0.0
    for step in range(2000):
        time_ms = float(step)
        drive = np.zeros(topology.n_kc)
        values = pulse(time_ms)
        for kc in range(topology.n_kc):
            incoming = topology.source[topology.target == kc]
            drive[kc] = values[incoming].mean()
        for kc in range(topology.n_kc):
            u[kc] += 0.1 * (-u[kc] + drive[kc] - apl)
        rates = np.maximum(u - 0.35, 0.0)
        apl += 0.02 * (-apl + rates.mean())
        if 500.0 <= time_ms < 1000.0:
            integrated += rates
        if 1500.0 <= time_ms < 2000.0:
            tail += rates
    np.testing.assert_allclose(result.final_u, u, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result.integrated_rate, integrated, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result.tail_integrated_rate, tail, rtol=1e-10, atol=1e-12)
    assert result.final_apl == pytest.approx(apl, rel=1e-10, abs=1e-12)


def test_piecewise_input_cache_is_bitwise_output_neutral():
    topology = graph()
    parameters = frozen_parameters()
    shared_on = np.asarray([1.0, 0.9, 0.8, 0.7, 0.0, 0.0, 0.0, 0.0])
    shared_zero = np.zeros(8)
    shared = simulate(
        topology,
        parameters,
        dt_ms=0.5,
        duration_ms=2000.0,
        pn_rate=lambda time: shared_on if 500 <= time < 1000 else shared_zero,
        integration_window_ms=(500.0, 1000.0),
        tail_window_ms=(1500.0, 2000.0),
    )
    copied = simulate(
        topology,
        parameters,
        dt_ms=0.5,
        duration_ms=2000.0,
        pn_rate=lambda time: (shared_on if 500 <= time < 1000 else shared_zero).copy(),
        integration_window_ms=(500.0, 1000.0),
        tail_window_ms=(1500.0, 2000.0),
    )
    assert np.array_equal(shared.integrated_rate, copied.integrated_rate)
    assert np.array_equal(shared.tail_integrated_rate, copied.tail_integrated_rate)
    assert np.array_equal(shared.final_u, copied.final_u)
    assert shared.final_apl == copied.final_apl


def test_toy_cross_dt_consistency_and_recovery():
    topology = graph()
    parameters = frozen_parameters()
    results = {
        dt: simulate(
            topology,
            parameters,
            dt_ms=dt,
            duration_ms=2000.0,
            pn_rate=pulse,
            integration_window_ms=(500.0, 1000.0),
            tail_window_ms=(1500.0, 2000.0),
        )
        for dt in (0.1, 0.05, 0.025)
    }
    reference = results[0.025]
    for dt, max_nrmse, min_jaccard in ((0.1, 0.05, 0.90), (0.05, 0.025, 0.95)):
        candidate = results[dt]
        rmse = np.sqrt(np.mean((candidate.integrated_rate - reference.integrated_rate) ** 2))
        scale = max(np.sqrt(np.mean(reference.integrated_rate**2)), 1e-12)
        assert rmse / scale <= max_nrmse
        assert jaccard(candidate.active, reference.active) >= min_jaccard
        assert abs(candidate.active.mean() - reference.active.mean()) <= (0.005 if dt == 0.1 else 0.0025)
    assert max(np.max(item.tail_integrated_rate) for item in results.values()) < 1e-6
    assert max(np.max(np.abs(item.final_u)) for item in results.values()) < 1e-6


def test_frozen_statistics_and_both_sparsity_denominators():
    base = np.zeros((16, 145))
    for odor in range(16):
        base[odor, odor : odor + 29] = 1.0
    matches = frozen_neighbor_matches(base)
    active = np.zeros((16, 10, 1907), dtype=bool)
    for odor in range(16):
        active[odor, :, odor * 110 : odor * 110 + 100] = True
    statistic = frozen_separation_statistic(active, matches)
    assert statistic["t_sep"] > 0.9
    assert separation_permutation_pvalue(active, matches, seed=7, permutations=999) <= 0.01
    views = sparsity_views(active)
    np.testing.assert_allclose(views["covered_1907_primary"], 100 / 1907)
    np.testing.assert_allclose(views["all_2045_secondary"], 100 / 2045)


def test_budget_is_fail_closed():
    config = load_frozen_config()
    enforce_stage_budget(BudgetUsage(23.0, 3.0, 4.0, 23.0), config)
    with pytest.raises(RuntimeError, match="M1_BUDGET_EXHAUSTED"):
        enforce_stage_budget(BudgetUsage(24.0001, 3.0, 4.0, 23.0), config)


def load_runner():
    spec = importlib.util.spec_from_file_location("m1_preregistered", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_runner_static_isolation_stage_lock_provenance_and_no_overwrite(tmp_path):
    tree = ast.parse(RUNNER.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = {
        "stonkfly.market", "stonkfly.reinforcement", "stonkfly.actions",
        "stonkfly.broker", "stonkfly.risk", "stonkfly.ledger",
    }
    assert imported.isdisjoint(forbidden)
    runner = load_runner()
    with pytest.raises(PermissionError, match="sealed"):
        runner.run_toy(tmp_path / "rq", DataDomain.M1_RQ)
    assert not (tmp_path / "rq").exists()
    output = tmp_path / "toy"
    result = runner.run_toy(output)
    assert result["verdict"] == "M1_BENCHMARK_ONLY_RUNNER_PATH_OK"
    assert result["formal_experiment"] is False
    manifest = json.loads((output / "m1_manifest.json").read_text())
    assert manifest["source_hashes_start"] == manifest["source_hashes_end"]
    assert len(manifest["artifacts"]) == 5
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        runner.run_toy(output)
