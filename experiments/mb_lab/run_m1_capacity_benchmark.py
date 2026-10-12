#!/usr/bin/env python3
"""Representative-scale BENCHMARK_ONLY capacity run; no formal stimuli or verdict."""

from __future__ import annotations

import argparse
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
from stonkfly.mb_lab.inventory import hemisphere_inventory, load_source, selected_nodes
from stonkfly.mb_lab.protocol import CONFIG_PATH, load_frozen_config, sha256_file, topology_control_seeds
from stonkfly.mb_lab.stages import DataDomain, require_implementation_prep_access
from stonkfly.mb_lab.topology import compile_real_topology
from stonkfly.mb_lab.provenance import sha256_file as provenance_sha256


def run(output: Path, *, duration_ms: float = 250.0) -> dict[str, object]:
    require_implementation_prep_access(DataDomain.BENCHMARK_ONLY)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite benchmark output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    config = load_frozen_config()
    source = load_source()
    pn, kc = selected_nodes("R", olfactory_covered_only=True)
    real = compile_real_topology(source["ptr"], source["post"], pn, kc)
    seeds = topology_control_seeds(config)
    swaps = int(config["topology_controls"]["accepted_double_edge_swaps_per_edge"]) * real.edge_count
    graphs = [real]
    build_start = time.perf_counter()
    for seed in seeds:
        graphs.append(real.rewire(seed, swaps=swaps))
    build_seconds = time.perf_counter() - build_start
    values = config["parameters"]
    parameters = M1Parameters(
        tau_k_ms=float(values["tau_k_ms"]),
        tau_apl_ms=float(values["tau_apl_ms"]),
        input_scale=float(values["input_scale"]),
        apl_gain=float(values["apl_gain"]),
        threshold=float(values["threshold"]),
        active_integral_threshold=float(values["active_mean_rate_threshold"]) * duration_ms,
    )
    # Analytic deterministic load vector, unrelated to any sealed odor collection.
    benchmark_vector = np.linspace(0.0, 1.0, real.n_pn, dtype=np.float64)
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    rows = []
    for graph_index, graph in enumerate(graphs):
        for dt in config["numerics"]["dt_ms"]:
            result = simulate(
                graph,
                parameters,
                dt_ms=float(dt),
                duration_ms=duration_ms,
                pn_rate=lambda _time, vector=benchmark_vector: vector,
            )
            rows.append(
                {
                    "graph_index": graph_index,
                    "dt_ms": float(dt),
                    "finite": result.finite,
                    "graph_sha256": graph.sha256,
                }
            )
    measured_wall = time.perf_counter() - wall_start
    measured_cpu = time.process_time() - cpu_start
    # v5 has 6560 stimulated path-trials plus 20 ZERO path-trials at each dt.
    # This benchmark executes 40 path simulations at each dt.
    base_scale_to_stage = (2000.0 / duration_ms) * (6580.0 / 40.0)
    engineering_margin = 1.25
    scale_to_stage = base_scale_to_stage * engineering_margin
    estimate = {
        "schema": "stonkfly-m1-capacity-benchmark-v3",
        "domain": DataDomain.BENCHMARK_ONLY.value,
        "formal_experiment": False,
        "scientific_metrics_computed": False,
        "sealed_domains_accessed": False,
        "n_pn": real.n_pn,
        "n_kc": real.n_kc,
        "edge_count": real.edge_count,
        "real_topology_sha256": real.sha256,
        "network_hashes": [graph.sha256 for graph in graphs],
        "right_inventory": hemisphere_inventory("R"),
        "left_replication_inventory": hemisphere_inventory("L"),
        "source_hashes": {
            "graph_npz": sha256_file(ROOT / "data/graph.npz"),
            "normalized_neurons": sha256_file(ROOT / "data/normalized/neurons.feather"),
            "annotations": sha256_file(ROOT / "data/annotations.feather"),
            "source_lock": sha256_file(ROOT / "data/source.lock.json"),
            "frozen_config": sha256_file(CONFIG_PATH),
            "formal_runner": provenance_sha256(ROOT / "experiments/mb_lab/run_m1_rq_formal.py"),
            "dynamics": provenance_sha256(ROOT / "stonkfly/mb_lab/dynamics.py"),
            "formal_kernel": provenance_sha256(ROOT / "stonkfly/mb_lab/formal.py"),
            "checkpoint": provenance_sha256(ROOT / "stonkfly/mb_lab/checkpoint.py"),
            "budget": provenance_sha256(ROOT / "stonkfly/mb_lab/budget.py"),
            "metrics": provenance_sha256(ROOT / "stonkfly/mb_lab/metrics.py"),
        },
        "network_count": len(graphs),
        "dt_ms": config["numerics"]["dt_ms"],
        "measured_neural_duration_ms_per_network_dt": duration_ms,
        "measured_rows": len(rows),
        "rewire_build_wall_seconds": build_seconds,
        "measured_cpu_seconds": measured_cpu,
        "measured_wall_seconds": measured_wall,
        "estimated_one_stage_cpu_hours_linear": measured_cpu * scale_to_stage / 3600.0,
        "estimated_two_stage_cpu_hours_linear": 2 * measured_cpu * scale_to_stage / 3600.0,
        "estimated_rq_cpu_hours_linear": measured_cpu * scale_to_stage / 3600.0,
        "estimated_rq_wall_hours_linear": measured_wall * scale_to_stage / 3600.0,
        "estimated_rq_path_trials_per_dt": 6580,
        "estimation_assumption": "linear from 40 representative full-size topology paths at every frozen dt to exactly 6580 path-trials per dt, with a 1.25 engineering margin; excludes one-time graph build and assumes no parallel speedup",
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "all_finite": all(row["finite"] for row in rows),
        "degree_preserved_all": all(
            np.array_equal(graph.pn_degree, real.pn_degree)
            and np.array_equal(graph.kc_degree, real.kc_degree)
            for graph in graphs[1:]
        ),
        "unique_graph_hashes": len({graph.sha256 for graph in graphs}) == len(graphs),
    }
    output.write_text(json.dumps(estimate, indent=2, sort_keys=True) + "\n")
    return estimate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--duration-ms", type=float, default=250.0)
    args = parser.parse_args(argv)
    result = run(args.out, duration_ms=args.duration_ms)
    print(json.dumps({"estimated_two_stage_cpu_hours_linear": result["estimated_two_stage_cpu_hours_linear"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
