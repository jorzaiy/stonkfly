from __future__ import annotations

import ast
import base64
import json
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
import numpy as np
import pytest

from stonkfly.mb_lab.authorization import AUTH_SCHEMA, canonical_payload, verify_rq_authorization
from stonkfly.mb_lab.budget import BudgetExceeded, RuntimeBudget
from stonkfly.mb_lab.checkpoint import validated_chunk, write_chunk_once
from stonkfly.mb_lab.dynamics import M1Parameters, simulate
from stonkfly.mb_lab.inventory import load_source, selected_nodes
from stonkfly.mb_lab.metrics import frozen_separation_statistic, separation_permutation_pvalue
from stonkfly.mb_lab.topology import compile_real_topology

ROOT = Path(__file__).resolve().parents[2]


def test_v5_s2_rng_matches_literal_protocol_draw():
    rng = np.random.default_rng(7)
    active = rng.random((16, 10, 31)) > 0.8
    matches = np.asarray([[(i + 1) % 16, (i + 2) % 16, (i + 3) % 16] for i in range(16)])
    margins = frozen_separation_statistic(active, matches)["margins"]
    seed = 12127502703192784129
    bits = np.random.Generator(np.random.PCG64(seed)).integers(
        0, 2, size=(9999, 16), dtype=np.uint8
    )
    values = np.mean((bits.astype(float) * 2 - 1) * margins, axis=1)
    expected = (1 + np.count_nonzero(values >= np.mean(margins) - 1e-15)) / 10000
    assert separation_permutation_pvalue(active, matches, seed=seed) == expected


def test_signed_authorization_is_exactly_rq_and_run_bound(tmp_path):
    private = Ed25519PrivateKey.generate()
    public = tmp_path / "public.pem"
    public.write_bytes(private.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo))
    payload = {"authorized": True, "domain": "M1_RQ", "protocol_bundle_sha256": "a" * 64,
               "rq_manifest_sha256": "b" * 64, "run_id": "run-1", "output": "/tmp/rq"}
    document = {"schema": AUTH_SCHEMA, "payload": payload,
                "signature_base64": base64.b64encode(private.sign(canonical_payload(payload))).decode()}
    authorization = tmp_path / "authorization.json"
    authorization.write_text(json.dumps(document))
    verify_rq_authorization(authorization, public, expected_bundle_sha256="a" * 64,
                            expected_manifest_sha256="b" * 64, expected_run_id="run-1",
                            expected_output="/tmp/rq")
    with pytest.raises(PermissionError):
        verify_rq_authorization(authorization, public, expected_bundle_sha256="a" * 64,
                                expected_manifest_sha256="b" * 64, expected_run_id="run-2",
                                expected_output="/tmp/rq")


def test_checkpoint_is_immutable_hash_checked_and_provenance_bound(tmp_path):
    metadata = {"bundle": "frozen", "run": "one"}
    write_chunk_once(tmp_path, "toy", {"x": np.arange(5)}, metadata)
    assert validated_chunk(tmp_path, "toy", metadata).is_file()
    with pytest.raises(FileExistsError):
        write_chunk_once(tmp_path, "toy", {"x": np.arange(5)}, metadata)
    with pytest.raises(RuntimeError, match="provenance"):
        validated_chunk(tmp_path, "toy", {"bundle": "changed"})


def test_budget_counts_prior_attempts_and_artifacts(tmp_path):
    (tmp_path / "large").write_bytes(b"x" * 32)
    budget = RuntimeBudget(tmp_path, 10, 10, 10**9, 16, prior_cpu_seconds=2, prior_wall_seconds=3)
    budget.start()
    usage = budget.usage()
    assert usage["cpu_seconds"] >= 2 and usage["wall_seconds"] >= 3
    with pytest.raises(BudgetExceeded, match="artifact_bytes"):
        budget.check()


def test_cached_real_topology_matches_independent_uncached_loop():
    source = load_source()
    pn, kc = selected_nodes("R", olfactory_covered_only=True)
    topology = compile_real_topology(source["ptr"], source["post"], pn, kc)
    parameters = M1Parameters(10.0, 50.0, 1.0, 1.0, 0.35, 25.0)
    vector = np.linspace(0.0, 1.0, topology.n_pn)
    cached = simulate(topology, parameters, dt_ms=0.1, duration_ms=20.0,
                      pn_rate=lambda _time: vector)
    u = np.zeros(topology.n_kc); integrated = np.zeros(topology.n_kc); apl = 0.0
    for _ in range(200):
        drive = np.bincount(topology.target,
                            weights=vector[topology.source] / topology.kc_degree[topology.target],
                            minlength=topology.n_kc)
        u += 0.01 * (-u + drive - apl)
        rate = np.maximum(u - 0.35, 0.0)
        apl += 0.002 * (-apl + rate.mean())
        integrated += rate * 0.1
    np.testing.assert_array_equal(cached.integrated_rate, integrated)
    np.testing.assert_array_equal(cached.final_u, u)
    assert cached.final_apl == apl


def test_formal_runner_has_no_market_or_other_stage_execution_surface():
    path = ROOT / "experiments/mb_lab/run_m1_rq_formal.py"
    tree = ast.parse(path.read_text())
    imports = {node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import)}
    text = path.read_text().lower()
    assert not imports.intersection({"ccxt", "backtrader", "alpaca_trade_api"})
    assert "m1_rt" not in text.replace('"m1_rt"', "") or "forbidden_paths" in text
    assert "m2_lq" in text and "forbidden_paths" in text
