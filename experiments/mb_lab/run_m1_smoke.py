#!/usr/bin/env python3
"""BENCHMARK_ONLY M1 implementation smoke; never a formal representation run."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stonkfly.mb_lab.dynamics import M1Parameters, simulate
from stonkfly.mb_lab.stages import DataDomain, require_implementation_prep_access
from stonkfly.mb_lab.topology import BipartiteTopology

SCHEMA = "stonkfly-m1-benchmark-only-smoke-v1"
DOMAIN_LABEL = b"STONKFLY-V2/BENCHMARK_ONLY/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def benchmark_seed(label: bytes) -> int:
    digest = hashlib.sha256(DOMAIN_LABEL + b"/" + label).digest()
    return int.from_bytes(digest[:8], "little")


def toy_topology() -> BipartiteTopology:
    target = np.repeat(np.arange(32, dtype=np.int32), 2)
    source = np.empty_like(target)
    source[0::2] = target[0::2] % 4
    source[1::2] = (target[1::2] + 1) % 4
    return BipartiteTopology(
        4,
        32,
        source,
        target,
        np.arange(4, dtype=np.int32),
        np.arange(32, dtype=np.int32),
    )


def prepare_output(path: Path) -> Path:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def snapshot() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "package_init": ROOT / "stonkfly/mb_lab/__init__.py",
        "dynamics": ROOT / "stonkfly/mb_lab/dynamics.py",
        "inventory": ROOT / "stonkfly/mb_lab/inventory.py",
        "metrics": ROOT / "stonkfly/mb_lab/metrics.py",
        "topology": ROOT / "stonkfly/mb_lab/topology.py",
        "stages": ROOT / "stonkfly/mb_lab/stages.py",
        "foundation_tests": ROOT / "tests/mb_lab/test_m1_foundation.py",
        "feasibility": ROOT / "docs/STONKFLY_V2_FEASIBILITY.md",
        "charter": ROOT / "docs/STONKFLY_V2_M1_CHARTER_DRAFT.md",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def write_artifacts(
    output: Path,
    formal: dict[str, object],
    raw: dict[str, np.ndarray],
    start_hashes: dict[str, str],
    started: float,
) -> None:
    raw_path = output / "m1_raw.npz"
    np.savez_compressed(raw_path, **raw)
    formal["wall_time_seconds"] = time.perf_counter() - started
    formal_path = output / "m1_formal.json"
    formal_path.write_text(json.dumps(formal, indent=2, sort_keys=True) + "\n")
    report_path = output / "m1_report.md"
    report_path.write_text(
        "# M1 BENCHMARK_ONLY smoke\n\n"
        f"Verdict: `{formal['verdict']}`\n\n"
        "This is an implementation smoke, not M1-RQ, M1-RT, or a scientific result.\n"
    )
    launch_path = output / "launch.txt"
    launch_path.write_text(" ".join(sys.argv) + "\n")
    stdout_path = output / "stdout.txt"
    stdout_path.write_text(json.dumps({"verdict": formal["verdict"]}) + "\n")
    end_hashes = snapshot()
    manifest = {
        "schema": "stonkfly-m1-smoke-manifest-v1",
        "data_domain": DataDomain.BENCHMARK_ONLY.value,
        "formal_experiment": False,
        "hashes_start": start_hashes,
        "hashes_end": end_hashes,
        "python": sys.version,
        "platform": platform.platform(),
        "artifacts": {
            "raw": sha256_file(raw_path),
            "formal": sha256_file(formal_path),
            "report": sha256_file(report_path),
            "launch": sha256_file(launch_path),
            "stdout": sha256_file(stdout_path),
        },
        "artifact_bytes": {
            path.name: path.stat().st_size
            for path in (
                raw_path,
                formal_path,
                report_path,
                launch_path,
                stdout_path,
            )
        },
    }
    (output / "m1_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def run(output: Path) -> dict[str, object]:
    started = time.perf_counter()
    cpu_started = time.process_time()
    require_implementation_prep_access(DataDomain.BENCHMARK_ONLY)
    output = prepare_output(output)
    start_hashes = snapshot()
    topology = toy_topology()
    control = topology.rewire(benchmark_seed(b"rewire"), swaps=topology.edge_count)
    parameters = M1Parameters(
        tau_k_ms=20.0,
        tau_apl_ms=50.0,
        input_scale=1.0,
        apl_gain=0.5,
        threshold=0.1,
        active_integral_threshold=2.0,
    )
    active_pn = np.asarray([1.0, 0.8, 0.0, 0.0], dtype=np.float64)

    def stimulus(time_ms: float) -> np.ndarray:
        return active_pn if 50.0 <= time_ms < 150.0 else np.zeros(4)

    rows = []
    active_rows = []
    integrated_rows = []
    for name, graph in (("B_TOY", topology), ("C_TOY", control)):
        for dt in (1.0, 0.5, 0.25):
            result = simulate(
                graph,
                parameters,
                dt_ms=dt,
                duration_ms=300.0,
                pn_rate=stimulus,
            )
            rows.append(
                {
                    "network": name,
                    "dt_ms": dt,
                    "finite": result.finite,
                    "active_count": int(result.active.sum()),
                    "integrated_sum": float(result.integrated_rate.sum()),
                }
            )
            active_rows.append(result.active)
            integrated_rows.append(result.integrated_rate)
    repeated = simulate(
        topology,
        parameters,
        dt_ms=0.5,
        duration_ms=300.0,
        pn_rate=stimulus,
    )
    reference = integrated_rows[1]
    deterministic = bool(np.array_equal(repeated.integrated_rate, reference))
    degree_preserved = bool(
        np.array_equal(topology.pn_degree, control.pn_degree)
        and np.array_equal(topology.kc_degree, control.kc_degree)
    )
    hashes_unchanged = snapshot() == start_hashes
    passed = (
        all(row["finite"] for row in rows)
        and deterministic
        and degree_preserved
        and hashes_unchanged
    )
    formal: dict[str, object] = {
        "schema": SCHEMA,
        "verdict": "M1_BENCHMARK_ONLY_SMOKE_OK"
        if passed
        else "M1_IMPLEMENTATION_FAILURE",
        "data_domain": DataDomain.BENCHMARK_ONLY.value,
        "formal_experiment": False,
        "m1_rq_used": False,
        "m1_rt_used": False,
        "m2_used": False,
        "market_used": False,
        "learning_used": False,
        "trading_policy_used": False,
        "execution_used": False,
        "production_eligible": False,
        "benchmark_parameters_not_formal": True,
        "gates": {
            "finite": all(row["finite"] for row in rows),
            "same_seed_bitwise": deterministic,
            "degree_preserved": degree_preserved,
            "source_hashes_unchanged": hashes_unchanged,
        },
        "rows": rows,
        "topology": {
            "real_sha256": topology.sha256,
            "rewire_sha256": control.sha256,
            "edge_count": topology.edge_count,
        },
        "cpu_time_seconds": time.process_time() - cpu_started,
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    raw = {
        "active": np.asarray(active_rows, dtype=np.uint8),
        "integrated_rate": np.asarray(integrated_rows, dtype=np.float64),
        "row_json": np.asarray(json.dumps(rows, sort_keys=True)),
    }
    write_artifacts(output, formal, raw, start_hashes, started)
    return formal


def failure(output: Path, error: Exception) -> dict[str, object]:
    formal: dict[str, object] = {
        "schema": SCHEMA,
        "verdict": "M1_IMPLEMENTATION_FAILURE",
        "data_domain": DataDomain.BENCHMARK_ONLY.value,
        "formal_experiment": False,
        "production_eligible": False,
        "error": repr(error),
        "traceback": traceback.format_exc(),
    }
    # Output protection has precedence over the failure artifact chain.  If a
    # target already contains evidence, never overwrite any part of it.
    if output.exists() and any(output.iterdir()):
        return formal
    output.mkdir(parents=True, exist_ok=True)
    write_artifacts(
        output,
        formal,
        {"failure": np.asarray(repr(error))},
        snapshot(),
        time.perf_counter(),
    )
    return formal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        result = run(arguments.out)
    except Exception as error:
        failure(arguments.out, error)
        raise
    print(json.dumps({"verdict": result["verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
