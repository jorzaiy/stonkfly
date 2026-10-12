#!/usr/bin/env python3
"""M1 preregistered runner candidate; preparation build permits toy fixture only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stonkfly.mb_lab.dynamics import M1Parameters, simulate
from stonkfly.mb_lab.protocol import CONFIG_PATH, load_frozen_config, sha256_file
from stonkfly.mb_lab.stages import DataDomain, require_implementation_prep_access
from stonkfly.mb_lab.topology import BipartiteTopology

FORBIDDEN_PREFIXES = (
    "stonkfly.market",
    "stonkfly.reinforcement",
    "stonkfly.actions",
    "stonkfly.broker",
    "stonkfly.risk",
    "stonkfly.ledger",
)


def source_snapshot() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "config": CONFIG_PATH,
        "dynamics": ROOT / "stonkfly/mb_lab/dynamics.py",
        "metrics": ROOT / "stonkfly/mb_lab/metrics.py",
        "protocol": ROOT / "stonkfly/mb_lab/protocol.py",
        "stages": ROOT / "stonkfly/mb_lab/stages.py",
        "stimuli": ROOT / "stonkfly/mb_lab/stimuli.py",
        "topology": ROOT / "stonkfly/mb_lab/topology.py",
        "tests": ROOT / "tests/mb_lab/test_m1_freeze.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def toy_topology() -> BipartiteTopology:
    target = np.arange(64, dtype=np.int32)
    source = target % 8
    return BipartiteTopology(
        8, 64, source, target, np.arange(8, dtype=np.int32), np.arange(64, dtype=np.int32)
    )


def prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def runtime_isolation() -> tuple[bool, list[str]]:
    loaded = sorted(
        name for name in sys.modules if any(name == item or name.startswith(item + ".") for item in FORBIDDEN_PREFIXES)
    )
    return not loaded, loaded


def run_toy(output: Path, domain: DataDomain | str = DataDomain.BENCHMARK_ONLY) -> dict[str, object]:
    require_implementation_prep_access(domain)
    prepare_output(output)
    start = time.perf_counter()
    cpu_start = time.process_time()
    hashes_start = source_snapshot()
    config = load_frozen_config()
    values = config["parameters"]
    parameters = M1Parameters(
        tau_k_ms=float(values["tau_k_ms"]),
        tau_apl_ms=float(values["tau_apl_ms"]),
        input_scale=float(values["input_scale"]),
        apl_gain=float(values["apl_gain"]),
        threshold=float(values["threshold"]),
        active_integral_threshold=float(values["active_mean_rate_threshold"]) * 500.0,
    )
    graph = toy_topology()
    pulse = np.asarray([1.0, 0.9, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0])

    def stimulus(time_ms: float) -> np.ndarray:
        return pulse if 500.0 <= time_ms < 1000.0 else np.zeros(8)

    rows = []
    raw_rates = []
    raw_active = []
    for dt in config["numerics"]["dt_ms"]:
        result = simulate(
            graph,
            parameters,
            dt_ms=float(dt),
            duration_ms=2000.0,
            pn_rate=stimulus,
            integration_window_ms=(500.0, 1000.0),
            tail_window_ms=(1500.0, 2000.0),
        )
        rows.append(
            {
                "dt_ms": float(dt),
                "finite": result.finite,
                "active_count": int(result.active.sum()),
                "tail_integral_max": float(result.tail_integrated_rate.max()),
            }
        )
        raw_rates.append(result.integrated_rate)
        raw_active.append(result.active)
    isolated, forbidden_loaded = runtime_isolation()
    hashes_end = source_snapshot()
    passed = all(row["finite"] for row in rows) and isolated and hashes_start == hashes_end
    formal = {
        "schema": "stonkfly-m1-preregistered-toy-path-v1",
        "verdict": "M1_BENCHMARK_ONLY_RUNNER_PATH_OK" if passed else "M1_IMPLEMENTATION_FAILURE",
        "domain": DataDomain.BENCHMARK_ONLY.value,
        "fixture": "TOY_ONLY",
        "formal_experiment": False,
        "production_eligible": False,
        "rq_accessed": False,
        "rt_accessed": False,
        "m2_accessed": False,
        "market_used": False,
        "learning_used": False,
        "execution_used": False,
        "rows": rows,
        "runtime_forbidden_modules": forbidden_loaded,
        "cpu_seconds": time.process_time() - cpu_start,
        "wall_seconds": time.perf_counter() - start,
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    raw_path = output / "m1_raw.npz"
    np.savez_compressed(raw_path, rates=np.asarray(raw_rates), active=np.asarray(raw_active))
    formal_path = output / "m1_formal.json"
    formal_path.write_text(json.dumps(formal, indent=2, sort_keys=True) + "\n")
    report_path = output / "m1_report.md"
    report_path.write_text(
        "# M1 preregistered runner toy-path review\n\n"
        f"Verdict: `{formal['verdict']}`. This is not an M1-RQ or M1-RT run.\n"
    )
    launch_path = output / "launch.txt"
    launch_path.write_text(" ".join(sys.argv) + "\n")
    stdout_path = output / "stdout.txt"
    stdout_path.write_text(json.dumps({"verdict": formal["verdict"]}) + "\n")
    artifact_paths = (raw_path, formal_path, report_path, launch_path, stdout_path)
    manifest = {
        "schema": "stonkfly-m1-preregistered-toy-manifest-v1",
        "domain": DataDomain.BENCHMARK_ONLY.value,
        "formal_experiment": False,
        "source_hashes_start": hashes_start,
        "source_hashes_end": hashes_end,
        "artifacts": {path.name: sha256_file(path) for path in artifact_paths},
        "artifact_bytes": {path.name: path.stat().st_size for path in artifact_paths},
    }
    (output / "m1_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return formal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--domain", default=DataDomain.BENCHMARK_ONLY.value, choices=[item.value for item in DataDomain])
    parser.add_argument("--fixture", default="toy", choices=["toy"])
    args = parser.parse_args(argv)
    result = run_toy(args.out, args.domain)
    print(json.dumps({"verdict": result["verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
