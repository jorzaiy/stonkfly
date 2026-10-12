#!/usr/bin/env python3
"""Run the fixed M1-E1 exploratory benchmark or full G/L/O matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stonkfly.mb_lab.budget import BudgetExceeded, RuntimeBudget, exclusive_json
from stonkfly.mb_lab.checkpoint import sha256_file, validated_chunk, write_chunk_once
from stonkfly.mb_lab.dynamics import M1Parameters
from stonkfly.mb_lab.e1 import build_apl_mapping, cosine, population_sparseness, simulate_e1
from stonkfly.mb_lab.inventory import load_source, selected_nodes
from stonkfly.mb_lab.stimuli import collection_sha256, generate_m1_collection
from stonkfly.mb_lab.topology import compile_real_topology

CONFIG = ROOT / "protocols/mb_lab/m1_e1/config.json"
GRAPH = ROOT / "data/graph.npz"
PROVENANCE = (
    CONFIG,
    ROOT / "docs/STONKFLY_V2_M1_E1_PLAN.md",
    ROOT / "experiments/mb_lab/run_m1_e1.py",
    ROOT / "experiments/mb_lab/e1_attempt.py",
    ROOT / "stonkfly/mb_lab/e1.py",
    ROOT / "stonkfly/mb_lab/dynamics.py",
    ROOT / "stonkfly/mb_lab/stimuli.py",
    ROOT / "stonkfly/mb_lab/topology.py",
    ROOT / "stonkfly/mb_lab/inventory.py",
    ROOT / "tests/mb_lab/test_m1_e1.py",
    GRAPH,
    ROOT / "data/normalized/neurons.feather",
    ROOT / "data/annotations.feather",
)
CURRENT_BUDGET: RuntimeBudget | None = None


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def source_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in PROVENANCE}


def parameters(config: dict[str, object]) -> M1Parameters:
    p = config["parameters"]
    return M1Parameters(
        float(p["tau_k_ms"]), float(p["tau_apl_ms"]), float(p["input_scale"]),
        float(p["apl_gain"]), float(p["threshold"]),
        float(p["active_mean_rate_threshold"]) * 500.0,
    )


def validate_seed(config: dict[str, object]) -> int:
    stimulus = config["stimulus"]
    digest = hashlib.sha256(str(stimulus["seed_label"]).encode()).digest()
    if digest.hex() != stimulus["seed_sha256"]:
        raise RuntimeError("E1 seed label digest mismatch")
    seed = int.from_bytes(digest[:8], "little")
    if seed != int(stimulus["seed_uint64_little_endian"]):
        raise RuntimeError("E1 uint64 seed mismatch")
    return seed


def simulate_matrix_chunk(topology, p, mapping, repeated, model: str, dt: float, timeline):
    n_odor, n_repeat, _ = repeated.shape
    n_kc = topology.n_kc
    bins = round(float(timeline["duration"]) / float(timeline["trace_bin"]))
    apl_states = 4 if model == "L" else 1
    response = np.empty((n_odor, n_repeat, n_kc), dtype=np.float64)
    population = np.empty((n_odor, n_repeat, bins), dtype=np.float64)
    groups = np.empty((n_odor, n_repeat, bins, 4), dtype=np.float64)
    apl = np.empty((n_odor, n_repeat, bins, apl_states), dtype=np.float64)
    final_norm = np.empty((n_odor, n_repeat), dtype=np.float64)
    finite = np.empty((n_odor, n_repeat), dtype=bool)
    for odor in range(n_odor):
        for trial in range(n_repeat):
            item = simulate_e1(
                topology, p, mapping, model=model, dt_ms=dt,
                duration_ms=float(timeline["duration"]), pulse=repeated[odor, trial],
                pulse_window_ms=(float(timeline["pulse_start"]), float(timeline["pulse_end"])),
                trace_bin_ms=float(timeline["trace_bin"]),
            )
            response[odor, trial] = item.integrated_rate
            population[odor, trial] = item.population_trace
            groups[odor, trial] = item.group_trace
            apl[odor, trial] = item.apl_trace
            final_norm[odor, trial] = np.linalg.norm(item.final_u)
            finite[odor, trial] = item.finite
    return {
        "response_integral": response, "population_trace": population,
        "group_trace": groups, "apl_trace": apl,
        "final_state_norm": final_norm, "finite": finite,
    }


def zero_chunk(topology, p, mapping, model: str, dt: float, timeline, repeats: int):
    maxima = np.empty((repeats, 4), dtype=np.float64)
    finite = np.empty(repeats, dtype=bool)
    for trial in range(repeats):
        item = simulate_e1(
            topology, p, mapping, model=model, dt_ms=dt,
            duration_ms=float(timeline["duration"]), pulse=np.zeros(topology.n_pn),
            pulse_window_ms=(float(timeline["pulse_start"]), float(timeline["pulse_end"])),
            trace_bin_ms=float(timeline["trace_bin"]),
        )
        maxima[trial] = (
            np.max(np.abs(item.integrated_rate)), np.max(np.abs(item.population_trace)),
            np.max(np.abs(item.final_u)), np.max(np.abs(item.final_apl)),
        )
        finite[trial] = item.finite
    return {"max_abs": maxima, "finite": finite}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def quantiles(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {"min": float(np.min(x)), "q05": float(np.quantile(x, .05)),
            "median": float(np.median(x)), "q95": float(np.quantile(x, .95)),
            "max": float(np.max(x)), "mean": float(np.mean(x))}


def bootstrap_odor_ci(values: np.ndarray, label: str) -> list[float]:
    per_odor = np.asarray(values, dtype=np.float64)
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(per_odor), size=(5000, len(per_odor)))
    means = per_odor[draws].mean(axis=1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def geometry(rate: np.ndarray) -> tuple[dict[str, object], np.ndarray]:
    centroids = rate.mean(axis=1)
    rsm = np.asarray([[cosine(left, right) for right in centroids] for left in centroids])
    within = []
    for repeats in rate:
        for left in range(len(repeats)):
            for right in range(left + 1, len(repeats)):
                within.append(cosine(repeats[left], repeats[right]))
    between = rsm[np.triu_indices(len(rsm), 1)]
    margin = float(np.mean(within) - np.mean(between))
    return {"within_cosine": quantiles(np.asarray(within)),
            "between_centroid_cosine": quantiles(between),
            "mean_margin": margin}, rsm


def arm_statistics(chunk: dict[str, np.ndarray], p: M1Parameters, all_kc: int, timeline):
    pulse_ms = float(timeline["pulse_end"]) - float(timeline["pulse_start"])
    rate = chunk["response_integral"] / pulse_ms
    active_count = np.sum(rate > p.active_integral_threshold / pulse_ms, axis=-1)
    participation = np.divide(
        np.sum(rate, axis=-1) ** 2, np.sum(rate**2, axis=-1),
        out=np.zeros(rate.shape[:2]), where=np.sum(rate**2, axis=-1) > 0,
    )
    geom, rsm = geometry(rate)
    trace = chunk["population_trace"]
    bin_ms = float(timeline["trace_bin"])
    centers = (np.arange(trace.shape[-1]) + .5) * bin_ms
    pulse = (centers >= float(timeline["pulse_start"])) & (centers < float(timeline["pulse_end"]))
    post = centers >= float(timeline["pulse_end"])
    post_trace = trace[..., post]
    peak = np.max(trace[..., pulse], axis=-1)
    half_times = np.full(peak.shape, np.nan)
    for index in np.ndindex(peak.shape):
        hits = np.flatnonzero(post_trace[index] <= .5 * peak[index])
        if len(hits):
            half_times[index] = (hits[0] + .5) * bin_ms
    residual_1500 = trace[..., int(round(1500.0 / bin_ms - .5))]
    return {
        "mean_rate": quantiles(rate),
        "population_sparseness": quantiles(population_sparseness(rate)),
        "participation_ratio": quantiles(participation),
        "active_fraction_1907": quantiles(active_count / rate.shape[-1]),
        "active_fraction_2045": quantiles(active_count / all_kc),
        "geometry": geom,
        "recovery": {
            "auc_1000_2000": quantiles(np.sum(post_trace, axis=-1) * bin_ms),
            "half_decay_ms_after_offset": quantiles(half_times[np.isfinite(half_times)]),
            "population_rate_at_1500_ms": quantiles(residual_1500),
            "final_state_norm": quantiles(chunk["final_state_norm"]),
        },
        "apl_trace": quantiles(chunk["apl_trace"]),
    }, rsm, rate


def cross_dt(candidate: np.ndarray, reference: np.ndarray, threshold: float):
    difference = candidate - reference
    rms = np.sqrt(np.mean(difference**2, axis=-1))
    denominator = np.maximum(np.sqrt(np.mean(reference**2, axis=-1)), .05)
    cosines = np.asarray([cosine(a, b) for a, b in zip(candidate.reshape(-1, candidate.shape[-1]), reference.reshape(-1, reference.shape[-1]))]).reshape(candidate.shape[:2])
    active_diff = np.mean(candidate > threshold, axis=-1) - np.mean(reference > threshold, axis=-1)
    return {"nrmse": quantiles(rms / denominator), "cosine": quantiles(cosines),
            "active_fraction_difference": quantiles(active_diff)}


def append_e1_failure(root: Path, value: dict[str, object]) -> None:
    directory = root / "failures"
    directory.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        target = directory / f"attempt-{index:04d}.json"
        try:
            exclusive_json(target, value)
            return
        except FileExistsError:
            index += 1


def run_benchmark(args, config):
    if args.out.exists():
        raise FileExistsError(args.out)
    source = load_source()
    pn, kc = selected_nodes("R", olfactory_covered_only=True)
    topology = compile_real_topology(source["ptr"], source["post"], pn, kc)
    mapping = build_apl_mapping(GRAPH, source, topology.kc_node_index)
    p = parameters(config)
    pulse = np.linspace(.2, 1.0, topology.n_pn)
    start_cpu, start_wall = time.process_time(), time.monotonic()
    for model in config["models"]:
        for dt in config["dt_ms"]:
            result = simulate_e1(topology, p, mapping, model=model, dt_ms=float(dt),
                                 duration_ms=250.0, pulse=pulse,
                                 pulse_window_ms=(0.0, 250.0), trace_bin_ms=10.0)
            if not result.finite:
                raise FloatingPointError("nonfinite benchmark")
    cpu = time.process_time() - start_cpu
    wall = time.monotonic() - start_wall
    scale = (1530 / 9) * (2000 / 250)
    result = {
        "schema": "stonkfly-m1-e1-representative-benchmark-v1",
        "formal_stimulus_used": False, "paths": 9, "duration_ms_per_path": 250.0,
        "cpu_seconds": cpu, "wall_seconds": wall,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "linear_formal_cpu_hours": cpu * scale / 3600,
        "conservative_1_5x_cpu_hours": cpu * scale * 1.5 / 3600,
        "topology_sha256": topology.sha256, "source_hashes": source_hashes(),
    }
    exclusive_json(args.out, result)
    return result


def run_formal(args, config):
    global CURRENT_BUDGET
    output = args.out.resolve()
    if (output / "COMPLETE.json").exists():
        raise FileExistsError("completed E1 run is immutable")
    output.mkdir(parents=True, exist_ok=True)
    start_hashes = source_hashes()
    identity = {"schema": "stonkfly-m1-e1-run-identity-v1", "run_id": args.run_id,
                "config_sha256": sha256_file(CONFIG), "source_hashes": start_hashes}
    run_path = output / "run.json"
    if run_path.exists():
        if json.loads(run_path.read_text()) != identity:
            raise RuntimeError("resume provenance mismatch")
    else:
        exclusive_json(run_path, identity)
    limits = config["budgets"]
    budget = RuntimeBudget(output, float(limits["cpu_hours"]) * 3600,
                           float(limits["wall_hours"]) * 3600,
                           int(float(limits["ram_gib"]) * 1024**3),
                           int(float(limits["artifact_gib"]) * 1024**3))
    CURRENT_BUDGET = budget
    budget.start()
    source = load_source()
    pn, kc = selected_nodes("R", olfactory_covered_only=True)
    _, all_kc_nodes = selected_nodes("R", olfactory_covered_only=False)
    topology = compile_real_topology(source["ptr"], source["post"], pn, kc)
    mapping = build_apl_mapping(GRAPH, source, topology.kc_node_index)
    if (topology.n_pn, topology.n_kc, len(all_kc_nodes)) != (145, 1907, 2045):
        raise RuntimeError("E1 inventory differs from reviewed plan")
    seed = validate_seed(config)
    collection = generate_m1_collection(seed, config["stimulus"])
    p = parameters(config)
    common = {"run_id": args.run_id, "config_sha256": sha256_file(CONFIG),
              "topology_sha256": topology.sha256,
              "stimulus_sha256": collection_sha256(collection)}
    budget.install_process_limits()
    for model in config["models"]:
        for dt_value in config["dt_ms"]:
            dt = float(dt_value)
            name = f"{model}-dt{dt:.3f}"
            if validated_chunk(output, name, common) is None:
                budget.check(pending_bytes=16 * 1024**2)
                arrays = simulate_matrix_chunk(topology, p, mapping, collection["repeated"], model, dt, config["timeline_ms"])
                write_chunk_once(output, name, arrays, common)
            zero_name = f"ZERO-{model}-dt{dt:.3f}"
            if validated_chunk(output, zero_name, common) is None:
                arrays = zero_chunk(topology, p, mapping, model, dt, config["timeline_ms"], int(config["zero_repeats_per_model_dt"]))
                write_chunk_once(output, zero_name, arrays, common)
            budget.check()
    statistics = {"schema": "stonkfly-m1-e1-statistics-v1", "exploratory": True,
                  "old_rq_gates_used": False, "arms": {}, "contrasts": {}, "cross_dt": {}}
    raw = {}
    rates = {}
    for model in config["models"]:
        statistics["arms"][model] = {}
        rates[model] = {}
        for dt_value in config["dt_ms"]:
            dt = float(dt_value)
            chunk = load_npz(output / "chunks" / f"{model}-dt{dt:.3f}.npz")
            if not bool(chunk["finite"].all()):
                raise FloatingPointError(f"nonfinite formal state: {model} {dt}")
            zero = load_npz(output / "chunks" / f"ZERO-{model}-dt{dt:.3f}.npz")
            if not bool(zero["finite"].all()) or np.max(zero["max_abs"]) != 0:
                raise RuntimeError(f"ZERO failure: {model} {dt}")
            summary, rsm, rate = arm_statistics(chunk, p, len(all_kc_nodes), config["timeline_ms"])
            statistics["arms"][model][str(dt)] = summary
            rates[model][dt] = rate
            raw[f"rsm_{model}_dt{dt:.3f}"] = rsm
    reference_dt = min(float(value) for value in config["dt_ms"])
    for left, right in (("O", "G"), ("O", "L"), ("L", "G")):
        delta = rates[left][reference_dt] - rates[right][reference_dt]
        per_odor = delta.mean(axis=(1, 2))
        label = f"{left}-{right}"
        statistics["contrasts"][label] = {
            "reference_dt_ms": reference_dt, "paired_rate_difference": quantiles(delta),
            "odor_mean_difference": quantiles(per_odor),
            "odor_resampling_mean_95ci": bootstrap_odor_ci(per_odor, f"M1-E1/{label}"),
        }
        raw[f"contrast_{left}_{right}_odor_means"] = per_odor
    threshold = float(config["parameters"]["active_mean_rate_threshold"])
    for model in config["models"]:
        statistics["cross_dt"][model] = {}
        for dt in sorted(rates[model]):
            if dt != reference_dt:
                statistics["cross_dt"][model][str(dt)] = cross_dt(rates[model][dt], rates[model][reference_dt], threshold)
    exclusive_json(output / "e1_statistics.json", jsonable(statistics))
    npz_path = output / "e1_raw.npz"
    if npz_path.exists():
        raise FileExistsError(npz_path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".e1_raw.", suffix=".npz", dir=output)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **raw)
        os.link(temporary, npz_path)
    finally:
        temporary.unlink(missing_ok=True)
    inventory = {
        "pn": topology.n_pn, "covered_kc": topology.n_kc, "all_kc": len(all_kc_nodes),
        "topology_sha256": topology.sha256, "apl_source_id": mapping.apl_source_id,
        "local_group_counts": np.bincount(mapping.group_index, minlength=4).tolist(),
        "apl_to_kc_contacts": quantiles(mapping.apl_to_kc_contacts),
        "kc_to_apl_contacts": quantiles(mapping.kc_to_apl_contacts),
    }
    exclusive_json(output / "e1_inventory.json", inventory)
    exclusive_json(output / "e1_config.json", config)
    usage = budget.check()
    report_path = output / "e1_report.md"
    descriptor = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(descriptor, "w") as handle:
        handle.write("# M1-E1 exploratory report\n\n")
        handle.write("Status: `E1_COMPLETED_EXPLORATORY`\n\n")
        handle.write("This run estimates G/L/O effects. It does not apply M1-RQ gates, qualify M1, access RT, or run M2.\n")
    exclusive_json(output / "stdout.json", {"status": "E1_COMPLETED_EXPLORATORY", "resource_usage": usage})
    end_hashes = source_hashes()
    if end_hashes != start_hashes:
        raise RuntimeError("source changed during E1")
    artifacts = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"e1_manifest.json", "COMPLETE.json"}:
            artifacts[str(path.relative_to(output))] = sha256_file(path)
    manifest = {"schema": "stonkfly-m1-e1-result-manifest-v1", "identity": identity,
                "source_hashes_start": start_hashes, "source_hashes_end": end_hashes,
                "artifacts": artifacts, "resource_usage": budget.check(),
                "rt_accessed": False, "m2_accessed": False, "market_used": False}
    exclusive_json(output / "e1_manifest.json", manifest)
    exclusive_json(output / "COMPLETE.json", {"status": "E1_COMPLETED_EXPLORATORY",
                                                "manifest_sha256": sha256_file(output / "e1_manifest.json")})
    return {"status": "E1_COMPLETED_EXPLORATORY", "resource_usage": budget.check()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("benchmark", "formal"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    try:
        result = run_benchmark(args, config) if args.mode == "benchmark" else run_formal(args, config)
    except BaseException as error:
        if args.mode == "formal" and args.out.exists() and not (args.out / "COMPLETE.json").exists():
            append_e1_failure(args.out, {
                "status": "E1_STOPPED", "error": repr(error),
                "traceback": traceback.format_exc(),
                "resource_usage": CURRENT_BUDGET.usage() if CURRENT_BUDGET else {},
                "recorded_utc": datetime.now(timezone.utc).isoformat(),
            })
        raise
    print(json.dumps(jsonable(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
