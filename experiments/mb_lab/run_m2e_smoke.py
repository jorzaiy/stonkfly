#!/usr/bin/env python3
"""Run the authorized two-odor M2-E mechanism smoke; never the full matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stonkfly.mb_lab.budget import BudgetExceeded, RuntimeBudget, exclusive_json
from stonkfly.mb_lab.checkpoint import sha256_file
from stonkfly.mb_lab.dynamics import M1Parameters
from stonkfly.mb_lab.e1 import build_apl_mapping, simulate_e1
from stonkfly.mb_lab.inventory import load_source, selected_nodes
from stonkfly.mb_lab.m2e import (
    PlasticityState, SensoryCache, build_connectivity, normalized_valence,
    raw_readout, train_event,
)
from stonkfly.mb_lab.stimuli import collection_sha256, generate_m1_collection
from stonkfly.mb_lab.topology import compile_real_topology
from stonkfly.neural.rule import PARAMETERS as RULE_PARAMETERS

CONFIG = ROOT / "protocols/mb_lab/m2_e/config.json"
GRAPH = ROOT / "data/graph.npz"
PROVENANCE = (
    CONFIG,
    ROOT / "docs/STONKFLY_V2_M2_E_DESIGN.md",
    ROOT / "stonkfly/mb_lab/e1.py",
    ROOT / "stonkfly/mb_lab/m2e.py",
    ROOT / "stonkfly/neural/rule.py",
    ROOT / "experiments/mb_lab/run_m2e_smoke.py",
    ROOT / "tests/mb_lab/test_m2_e.py",
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


def hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in PROVENANCE}


def digest_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def readout_for(connectivity, sensory, state):
    return raw_readout(
        connectivity, sensory.integrated_rate / 500.0, state.efficacy
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    global CURRENT_BUDGET
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(CONFIG.read_text())
    if config["status"] != "AUTHORIZED_TOY_SMOKE_ONLY" or config["full_exploration"]["authorized"]:
        raise RuntimeError("M2-E smoke authorization is not fail-closed")
    if int(config["smoke"]["odor_count"]) != 2 or int(config["smoke"]["repeat_count"]) != 2:
        raise RuntimeError("smoke must remain the reviewed 2x2 fixture")
    for name in ("trace_kc_seconds", "trace_dan_seconds", "memory_decay_seconds",
                 "weight_filter_seconds", "minimum_fraction", "maximum_fraction"):
        if config["plasticity"][name] != RULE_PARAMETERS[name]:
            raise RuntimeError(f"plasticity source/config mismatch: {name}")
    output.mkdir(parents=True)
    start_hashes = hashes()
    exclusive_json(output / "run.json", {
        "schema": "stonkfly-m2-e-smoke-run-v1", "run_id": args.run_id,
        "config_sha256": sha256_file(CONFIG), "source_hashes": start_hashes,
        "full_matrix_authorized": False,
    })
    limits = config["budgets"]
    budget = RuntimeBudget(
        output, float(limits["smoke_cpu_seconds"]), float(limits["smoke_wall_seconds"]),
        int(float(limits["smoke_rss_gib"]) * 1024**3),
        int(float(limits["smoke_artifact_mib"]) * 1024**2),
    )
    CURRENT_BUDGET = budget
    budget.start()
    source = load_source()
    pn, kc = selected_nodes("R", olfactory_covered_only=True)
    topology = compile_real_topology(source["ptr"], source["post"], pn, kc)
    apl_mapping = build_apl_mapping(GRAPH, source, topology.kc_node_index)
    connectivity = build_connectivity(GRAPH, source, topology.kc_node_index)
    s = config["sensory"]
    parameters = M1Parameters(
        float(s["tau_k_ms"]), float(s["tau_apl_ms"]), float(s["input_scale"]),
        float(s["apl_gain"]), float(s["threshold"]), 25.0,
    )
    cache = SensoryCache(topology, parameters, apl_mapping, dt_ms=float(s["dt_ms"]))
    label = str(config["smoke"]["seed_label"])
    seed_digest = hashlib.sha256(label.encode()).digest()
    seed = int.from_bytes(seed_digest[:8], "little")
    stimulus_spec = {
        "pn_count": topology.n_pn,
        "odor_count": 2,
        "repeat_count": 2,
        "active_pn_count_per_odor": int(config["smoke"]["active_pn_count_per_odor"]),
        "base_amplitude_uniform": [0.8, 1.0],
        "repeat_jitter_normal_sd": 0.05,
        "clip": [0.0, 1.0],
    }
    collection = generate_m1_collection(seed, stimulus_spec)
    sensory_cpu_start = time.process_time()
    sensory: list[list[object]] = []
    keys: list[list[str]] = []
    for odor in range(2):
        sensory.append([])
        keys.append([])
        for repeat in range(2):
            key, item = cache.get(collection["repeated"][odor, repeat])
            keys[odor].append(key)
            sensory[odor].append(item)
    sensory_cpu = time.process_time() - sensory_cpu_start
    if cache.builds != 4 or cache.hits != 0 or len(set(sum(keys, []))) != 4:
        raise RuntimeError("smoke sensory cache identity failure")
    # Independently rerun one item without KC trace: adding the observer must not
    # alter the already-completed E1-G equations.
    direct = simulate_e1(
        topology, parameters, apl_mapping, model="G", dt_ms=float(s["dt_ms"]),
        duration_ms=2000.0, pulse=collection["repeated"][0, 0],
        pulse_window_ms=(500.0, 1000.0), trace_bin_ms=10.0,
        record_kc_trace=False,
    )
    if not (
        np.array_equal(direct.integrated_rate, sensory[0][0].integrated_rate)
        and np.array_equal(direct.population_trace, sensory[0][0].population_trace)
        and np.array_equal(direct.final_u, sensory[0][0].final_u)
    ):
        raise RuntimeError("KC-trace observer changed E1-G dynamics")
    baseline_state = PlasticityState.fresh(connectivity)
    baseline = [[readout_for(connectivity, sensory[o][r], baseline_state) for r in range(2)] for o in range(2)]
    channel_matrix = np.asarray([[baseline[o][r]["channels"] for r in range(2)] for o in range(2)])
    if np.any(channel_matrix <= 1e-12):
        raise RuntimeError("one MBON channel is not drivable in smoke")
    baseline_ratio = channel_matrix[..., 1] / channel_matrix[..., 0]
    arms = {name: PlasticityState.fresh(connectivity, eta=float(config["plasticity"]["effective_eta_per_second"])) for name in ("paired", "frozen", "shuffled", "unpaired", "dan_omitted")}
    for left_name, left in arms.items():
        for right_name, right in arms.items():
            if left_name != right_name and np.shares_memory(left.memory_w, right.memory_w):
                raise RuntimeError("learning states share plastic memory")
    initial_weights = {name: state.weight_digest() for name, state in arms.items()}
    schedules = {
        "paired": [["reward", "aversive"], ["reward", "aversive"]],
        "frozen": [["reward", "aversive"], ["reward", "aversive"]],
        "unpaired": [["reward", "aversive"], ["reward", "aversive"]],
        "dan_omitted": [["reward", "aversive"], ["reward", "aversive"]],
        # Each odor receives one pulse of each valence, while global counts match.
        "shuffled": [["reward", "aversive"], ["aversive", "reward"]],
    }
    for cycle in range(2):
        for odor in range(2):
            item = sensory[odor][cycle]
            # A second lookup proves all arms address the same immutable cache item.
            key, shared = cache.get(collection["repeated"][odor, cycle])
            if shared is not item or key != keys[odor][cycle]:
                raise RuntimeError("arm cache reuse is not equivalent")
            for name, state in arms.items():
                train_event(
                    state, shared, valence=schedules[name][cycle][odor],
                    timing="unpaired" if name == "unpaired" else "paired",
                    frozen=name == "frozen", omitted=name == "dan_omitted",
                    dan_amplitude=float(config["plasticity"]["dan_amplitude_dimensionless"]),
                    washout_seconds=float(config["plasticity"]["inter_trial_washout_ms"]) / 1000.0,
                )
            budget.check()
    results = {}
    for name, state in arms.items():
        before_probe = state.weight_digest()
        odor_values = []
        raw_channels = []
        for odor in range(2):
            values = []
            channels = []
            for repeat in range(2):
                current = readout_for(connectivity, sensory[odor][repeat], state)
                values.append(normalized_valence(current["channels"], baseline[odor][repeat]["channels"]))
                channels.append(current["channels"])
            odor_values.append(float(np.mean(values)))
            raw_channels.append(np.mean(channels, axis=0))
        if state.weight_digest() != before_probe:
            raise RuntimeError("no-reward probe mutated plastic weights")
        results[name] = {
            "weight_changed": state.weight_digest() != initial_weights[name],
            "changed_edges": int(np.count_nonzero(state.memory_w)),
            "weight_digest": state.weight_digest(),
            "memory_w_min": float(state.memory_w.min()),
            "memory_w_max": float(state.memory_w.max()),
            "bound_fraction": float(np.mean((state.efficacy <= .3) | (state.efficacy >= 2.0))),
            "odor_valence": odor_values,
            "association_D_A_minus_B": odor_values[0] - odor_values[1],
            "raw_channels_avoidance_approach": np.asarray(raw_channels),
        }
    if results["frozen"]["weight_changed"] or results["dan_omitted"]["weight_changed"]:
        raise RuntimeError("negative control changed weights")
    if any(results[name]["bound_fraction"] > .01 for name in arms):
        raise RuntimeError("smoke exceeded plasticity saturation stop")
    estimated_cpu_hours = sensory_cpu / 4 * 640 * 1.5 / 3600
    smoke = {
        "schema": "stonkfly-m2-e-smoke-result-v1",
        "status": "M2_E_SMOKE_COMPLETE",
        "full_matrix_run": False,
        "legacy_domains_accessed": False,
        "market_or_trading_used": False,
        "stimulus": {"seed_label": label, "seed_sha256": seed_digest.hex(),
                     "content_sha256": collection_sha256(collection), "unique_paths": 4},
        "inventory": {
            "pn": topology.n_pn, "kc": topology.n_kc,
            "plastic_edges": len(connectivity.pre_slot),
            "edges_by_channel": np.bincount(connectivity.edge_channel, minlength=2),
            "input_kc_by_channel": [len(np.unique(connectivity.pre_slot[connectivity.edge_channel == c])) for c in range(2)],
            "target_ids": connectivity.target_source_id,
        },
        "cache": {"builds": cache.builds, "hits": cache.hits,
                  "keys_sha256": digest_arrays(np.asarray(sum(keys, []), dtype="S64"))},
        "initial_readout": {
            "channels_avoidance_approach": channel_matrix,
            "approach_to_avoidance_ratio": baseline_ratio,
            "ratio_min": float(baseline_ratio.min()), "ratio_max": float(baseline_ratio.max()),
            "ratio_mean": float(baseline_ratio.mean()),
            "normalized_valence_is_zero_by_definition": True,
        },
        "arms": results,
        "sensory_cpu_seconds_for_4_paths": sensory_cpu,
        "future_640_path_conservative_cpu_hours": estimated_cpu_hours,
        "resource_usage": budget.check(),
    }
    exclusive_json(output / "smoke.json", jsonable(smoke))
    report = output / "report.md"
    descriptor = os.open(report, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(descriptor, "w") as handle:
        handle.write("# M2-E toy/smoke\n\nStatus: `M2_E_SMOKE_COMPLETE`\n\n")
        handle.write("Only the authorized two-odor smoke ran; the 8-pair matrix remains blocked.\n")
    exclusive_json(output / "stdout.json", {"status": smoke["status"], "resource_usage": budget.check()})
    end_hashes = hashes()
    if end_hashes != start_hashes:
        raise RuntimeError("M2-E source changed during smoke")
    artifacts = {str(path.relative_to(output)): sha256_file(path) for path in sorted(output.rglob("*")) if path.is_file()}
    manifest = {
        "schema": "stonkfly-m2-e-smoke-manifest-v1", "run_id": args.run_id,
        "source_hashes_start": start_hashes, "source_hashes_end": end_hashes,
        "artifacts": artifacts, "resource_usage": budget.check(),
    }
    exclusive_json(output / "manifest.json", manifest)
    exclusive_json(output / "COMPLETE.json", {
        "status": smoke["status"], "manifest_sha256": sha256_file(output / "manifest.json")
    })
    return smoke


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        result = run(args)
    except BaseException as error:
        if args.out.exists() and not (args.out / "COMPLETE.json").exists():
            exclusive_json(args.out / "FAILURE.json", {
                "status": "M2_E_SMOKE_FAILED", "error": repr(error),
                "traceback": traceback.format_exc(),
                "resource_usage": CURRENT_BUDGET.usage() if CURRENT_BUDGET else {},
                "recorded_utc": datetime.now(timezone.utc).isoformat(),
            })
        raise
    print(json.dumps({"status": result["status"], "future_cpu_hours": result["future_640_path_conservative_cpu_hours"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
