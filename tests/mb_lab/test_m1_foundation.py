from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from stonkfly.mb_lab.dynamics import M1Parameters, simulate
from stonkfly.mb_lab.metrics import (
    exact_sign_flip_pvalue,
    jaccard,
    odor_reliability,
    topology_rank_pvalue,
)
from stonkfly.mb_lab.stages import DataDomain, require_implementation_prep_access
from stonkfly.mb_lab.topology import BipartiteTopology

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments/mb_lab/run_m1_smoke.py"


def topology() -> BipartiteTopology:
    return BipartiteTopology(
        2,
        3,
        np.asarray([0, 1, 0, 1], dtype=np.int32),
        np.asarray([0, 1, 2, 2], dtype=np.int32),
        np.arange(2, dtype=np.int32),
        np.arange(3, dtype=np.int32),
    )


def parameters() -> M1Parameters:
    return M1Parameters(10.0, 20.0, 1.0, 0.5, 0.1, 0.01)


def test_topology_projection_is_indegree_normalized():
    graph = topology()
    np.testing.assert_allclose(graph.project(np.asarray([2.0, 4.0])), [2.0, 4.0, 3.0])


def test_degree_preserving_rewire_changes_only_adjacency():
    graph = BipartiteTopology(
        4,
        8,
        np.tile(np.arange(4, dtype=np.int32), 8),
        np.repeat(np.arange(8, dtype=np.int32), 4),
        np.arange(4, dtype=np.int32),
        np.arange(8, dtype=np.int32),
    )
    # A complete bipartite graph cannot be rewired without recreating edges.
    with pytest.raises(RuntimeError, match="only completed"):
        graph.rewire(7, swaps=1)
    sparse = BipartiteTopology(
        4,
        8,
        np.repeat(np.arange(4, dtype=np.int32), 4),
        np.asarray([0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 0, 1], dtype=np.int32),
        np.arange(4, dtype=np.int32),
        np.arange(8, dtype=np.int32),
    )
    rewired = sparse.rewire(7, swaps=8)
    np.testing.assert_array_equal(rewired.pn_degree, sparse.pn_degree)
    np.testing.assert_array_equal(rewired.kc_degree, sparse.kc_degree)
    assert rewired.sha256 != sparse.sha256


def test_vectorized_dynamics_matches_independent_dense_oracle():
    graph = topology()
    p = parameters()
    rate = np.asarray([2.0, 4.0])
    result = simulate(
        graph,
        p,
        dt_ms=1.0,
        duration_ms=5.0,
        pn_rate=lambda _: rate,
    )
    dense = np.zeros((3, 2), dtype=np.float64)
    for source, target in zip(graph.source, graph.target):
        dense[target, source] = 1.0 / graph.kc_degree[target]
    u = np.zeros(3, dtype=np.float64)
    integrated = np.zeros(3, dtype=np.float64)
    apl = 0.0
    for _ in range(5):
        u += 0.1 * (-u + dense @ rate - 0.5 * apl)
        response = np.maximum(u - 0.1, 0.0)
        apl += 0.05 * (-apl + response.mean())
        integrated += response
    np.testing.assert_allclose(result.final_u, u, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.integrated_rate, integrated, rtol=1e-12, atol=1e-12)
    assert result.final_apl == pytest.approx(apl, abs=1e-12)


def test_apl_off_is_only_a_gain_intervention_and_increases_activity():
    graph = topology()
    kwargs = dict(
        topology=graph,
        parameters=parameters(),
        dt_ms=0.5,
        duration_ms=100.0,
        pn_rate=lambda _: np.asarray([2.0, 4.0]),
    )
    on = simulate(**kwargs, apl_enabled=True)
    off = simulate(**kwargs, apl_enabled=False)
    assert off.integrated_rate.sum() > on.integrated_rate.sum()


def test_metrics_use_fixed_pairs_and_exact_nulls():
    left = np.asarray([1, 1, 0, 0], dtype=bool)
    right = np.asarray([1, 0, 1, 0], dtype=bool)
    assert jaccard(left, right) == pytest.approx(1 / 3)
    assert np.isnan(jaccard(np.zeros(4, bool), np.zeros(4, bool)))
    active = np.zeros((2, 10, 4), dtype=bool)
    active[0, :, :2] = True
    active[1, :, 2:] = True
    np.testing.assert_array_equal(odor_reliability(active), [1.0, 1.0])
    assert exact_sign_flip_pvalue(np.ones(16)) == pytest.approx(1 / 65536)
    assert topology_rank_pvalue(2.0, np.arange(39, dtype=float) / 39) == 1 / 40


def test_implementation_prep_policy_is_fail_closed():
    assert require_implementation_prep_access("BENCHMARK_ONLY") is DataDomain.BENCHMARK_ONLY
    for sealed in (DataDomain.M1_RQ, DataDomain.M1_RT, DataDomain.M2_LQ, DataDomain.M2_LT):
        with pytest.raises(PermissionError, match="sealed"):
            require_implementation_prep_access(sealed)


def load_runner():
    specification = importlib.util.spec_from_file_location("m1_smoke", RUNNER)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


def test_runner_has_no_forbidden_imports():
    tree = ast.parse(RUNNER.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = {
        "stonkfly.market",
        "stonkfly.reinforcement",
        "stonkfly.actions",
        "stonkfly.broker",
        "stonkfly.risk",
        "stonkfly.ledger",
    }
    assert imported.isdisjoint(forbidden)


def test_benchmark_only_smoke_writes_complete_artifact_chain(tmp_path):
    runner = load_runner()
    output = tmp_path / "smoke"
    result = runner.run(output)
    assert result["verdict"] == "M1_BENCHMARK_ONLY_SMOKE_OK"
    expected = {
        "m1_raw.npz",
        "m1_formal.json",
        "m1_report.md",
        "m1_manifest.json",
        "launch.txt",
        "stdout.txt",
    }
    assert expected == {path.name for path in output.iterdir()}
    formal = json.loads((output / "m1_formal.json").read_text())
    assert formal["formal_experiment"] is False
    assert formal["m1_rq_used"] is False and formal["m1_rt_used"] is False
    assert formal["m2_used"] is False and formal["market_used"] is False
    assert formal["learning_used"] is False and formal["execution_used"] is False
    assert formal["production_eligible"] is False
    assert formal["cpu_time_seconds"] >= 0
    assert formal["max_rss_kib"] > 0
    manifest = json.loads((output / "m1_manifest.json").read_text())
    assert set(manifest["artifact_bytes"]) == expected - {"m1_manifest.json"}
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        runner.run(output)


def test_cli_failure_path_does_not_overwrite_existing_evidence(tmp_path):
    runner = load_runner()
    output = tmp_path / "existing"
    output.mkdir()
    evidence = output / "evidence.txt"
    evidence.write_text("preserve me\n")
    runner.failure(output, RuntimeError("output protection"))
    assert evidence.read_text() == "preserve me\n"
    assert {path.name for path in output.iterdir()} == {"evidence.txt"}
