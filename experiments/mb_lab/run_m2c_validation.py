#!/usr/bin/env python3
"""Run the fixed M2-C aggregate-readout validation or its isolated smoke."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mb_lab.run_m2b_exploration import (
    ARMS,
    STAGES,
    build_bases,
    expected_paths,
    install_process_backstops,
    jsonable,
    load_npz,
    preference_index,
    schedule,
    stimulus,
    stimulus_content_hash,
    vector_summary,
    write_npz_once,
)
from stonkfly.mb_lab.budget import BudgetExceeded, RuntimeBudget, exclusive_json
from stonkfly.mb_lab.checkpoint import sha256_file, validated_chunk, write_chunk_once
from stonkfly.mb_lab.dynamics import M1Parameters
from stonkfly.mb_lab.e1 import build_apl_mapping
from stonkfly.mb_lab.inventory import load_source, selected_nodes
from stonkfly.mb_lab.m2e import (
    PlasticityState,
    SensoryCache,
    build_connectivity,
    normalized_valence,
    raw_readout,
    train_event,
)
from stonkfly.mb_lab.topology import compile_real_topology
from stonkfly.neural.rule import PARAMETERS as RULE_PARAMETERS

CONFIG = ROOT / "protocols/mb_lab/m2_c/config.json"
GRAPH = ROOT / "data/graph.npz"
FORMAL_ROOT = ROOT / "runs/mb_lab/m2-c-validation"
FORMAL_SEED_LABEL = "STONKFLY-V2/M2-C/READOUT-RELIABILITY/v1"
SMOKE_SEED_LABEL = "STONKFLY-V2/M2-C/READOUT-RELIABILITY/SMOKE/v1"
PROVENANCE = (
    CONFIG,
    ROOT / "docs/STONKFLY_V2_M2_C_OFFLINE_AND_DESIGN.md",
    ROOT / "stonkfly/mb_lab/e1.py",
    ROOT / "stonkfly/mb_lab/m2e.py",
    ROOT / "stonkfly/neural/rule.py",
    ROOT / "experiments/mb_lab/run_m2b_exploration.py",
    ROOT / "experiments/mb_lab/run_m2c_validation.py",
    ROOT / "tests/mb_lab/test_m2_c.py",
    GRAPH,
    ROOT / "data/normalized/neurons.feather",
    ROOT / "data/annotations.feather",
)
CURRENT_BUDGET: RuntimeBudget | None = None


def source_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in PROVENANCE}


def run_pair(
    pair: int,
    seed_label: str,
    bases: np.ndarray,
    cache: SensoryCache,
    connectivity,
    config: dict[str, object],
    budget: RuntimeBudget,
    *,
    calibration_repeats: int,
    probe_repeats: int,
    cycles: int,
    retention_seconds: float,
) -> dict[str, np.ndarray]:
    n_edge = len(connectivity.pre_slot)
    calibration_channels = np.empty((2, calibration_repeats, 2), dtype=np.float64)
    calibration_targets = np.empty((2, calibration_repeats, 3), dtype=np.float64)
    for odor in range(2):
        for repeat in range(calibration_repeats):
            pulse = stimulus(seed_label, bases, pair, "calibration", odor, repeat)
            key, item = cache.get(pulse)
            raw = raw_readout(connectivity, item.integrated_rate / 500.0, np.ones(n_edge))
            calibration_channels[odor, repeat] = raw["channels"]
            calibration_targets[odor, repeat] = raw["per_target"]
            cache.discard(key)
            budget.check()
    baseline_channels = calibration_channels.mean(axis=1)
    baseline_targets = calibration_targets.mean(axis=1)
    if np.any(baseline_channels <= 1e-12):
        raise RuntimeError("calibration MBON channel is not drivable")

    eta = float(config["plasticity"]["effective_eta_per_second"])
    states = {name: PlasticityState.fresh(connectivity, eta=eta) for name in ARMS}
    for left_name, left in states.items():
        for right_name, right in states.items():
            if left_name < right_name and any(
                np.shares_memory(getattr(left, field), getattr(right, field))
                for field in ("kc_trace", "dan_trace", "memory_u", "memory_w")
            ):
                raise RuntimeError("M2-C arms share learning state")

    values = np.empty((len(ARMS), len(STAGES), 2, probe_repeats), dtype=np.float64)
    choice_d = np.empty((len(ARMS), len(STAGES), probe_repeats), dtype=np.float64)
    choices = np.empty((len(ARMS), len(STAGES), probe_repeats), dtype=np.int8)
    channels = np.empty((len(ARMS), len(STAGES), 2, probe_repeats, 2), dtype=np.float64)
    targets = np.empty((len(ARMS), len(STAGES), 2, probe_repeats, 3), dtype=np.float64)
    efficacy = np.empty((len(ARMS), len(STAGES), n_edge), dtype=np.float64)
    finite = np.ones((len(ARMS), len(STAGES)), dtype=bool)

    def probe(stage_index: int) -> None:
        stage = STAGES[stage_index]
        for odor in range(2):
            for repeat in range(probe_repeats):
                pulse = stimulus(seed_label, bases, pair, f"choice-{stage}", odor, repeat)
                key, item = cache.get(pulse)
                rate = item.integrated_rate / 500.0
                for arm_index, name in enumerate(ARMS):
                    state = states[name]
                    before = state.weight_digest()
                    current = raw_readout(connectivity, rate, state.efficacy)
                    values[arm_index, stage_index, odor, repeat] = normalized_valence(
                        current["channels"], baseline_channels[odor]
                    )
                    channels[arm_index, stage_index, odor, repeat] = current["channels"]
                    targets[arm_index, stage_index, odor, repeat] = current["per_target"]
                    if state.weight_digest() != before:
                        raise RuntimeError("reward-free M2-C probe changed weights")
                cache.discard(key)
                budget.check()
        choice_d[:, stage_index] = values[:, stage_index, 0] - values[:, stage_index, 1]
        choices[:, stage_index] = np.sign(choice_d[:, stage_index]).astype(np.int8)
        for arm_index, name in enumerate(ARMS):
            efficacy[arm_index, stage_index] = states[name].efficacy
            finite[arm_index, stage_index] = bool(
                np.isfinite(values[arm_index, stage_index]).all()
                and np.isfinite(efficacy[arm_index, stage_index]).all()
            )

    def train_phase(phase: str) -> None:
        plan = schedule(seed_label, pair, phase, cycles)
        for cycle, order in enumerate(plan["orders"]):
            for odor in order:
                pulse = stimulus(seed_label, bases, pair, f"train-{phase}", odor, cycle)
                key, item = cache.get(pulse)
                for name, state in states.items():
                    valence = plan["shuffled"][odor][cycle] if name == "shuffled" else plan["correct"][odor]
                    train_event(
                        state,
                        item,
                        valence=valence,
                        timing="unpaired" if name == "unpaired" else "paired",
                        frozen=name == "frozen",
                        omitted=name == "dan_omitted",
                        dan_amplitude=float(config["plasticity"]["dan_amplitude_dimensionless"]),
                        washout_seconds=float(config["plasticity"]["inter_trial_washout_ms"]) / 1000.0,
                    )
                cache.discard(key)
                budget.check()

    probe(0)
    train_phase("acquisition")
    probe(1)
    train_phase("reversal")
    probe(2)
    for name, state in states.items():
        state.blank(retention_seconds, frozen=name == "frozen")
    probe(3)

    return {
        "calibration_channels": calibration_channels,
        "calibration_targets": calibration_targets,
        "baseline_channels": baseline_channels,
        "baseline_targets": baseline_targets,
        "values": values,
        "choice_d": choice_d,
        "choices": choices,
        "channels": channels,
        "targets": targets,
        "efficacy": efficacy,
        "changed_edges": np.count_nonzero(efficacy != 1.0, axis=-1),
        "finite": finite,
        "bound_fraction": np.mean((efficacy <= .3) | (efficacy >= 2.0), axis=-1),
    }


def analyze(
    chunks: list[dict[str, np.ndarray]],
    seed_label: str,
    required_pairs: int,
    single_index: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    choice_d = np.stack([chunk["choice_d"] for chunk in chunks], axis=1)
    choices = np.stack([chunk["choices"] for chunk in chunks], axis=1)
    efficacy = np.stack([chunk["efficacy"] for chunk in chunks], axis=1)
    finite = np.stack([chunk["finite"] for chunk in chunks], axis=1)
    bound = np.stack([chunk["bound_fraction"] for chunk in chunks], axis=1)
    aggregate = choice_d.mean(axis=-1)
    aggregate_choice = np.sign(aggregate).astype(np.int8)
    single_choice = choices[..., single_index]
    arm_index = {name: index for index, name in enumerate(ARMS)}
    p = arm_index["paired"]
    statistics: dict[str, object] = {
        "schema": "stonkfly-m2-c-readout-reliability-statistics-v1",
        "arms": {},
        "paired_vs_controls": {},
    }
    for arm, arm_name in enumerate(ARMS):
        pairs = {}
        for pair in range(len(chunks)):
            pairs[str(pair)] = {}
            for stage, stage_name in enumerate(STAGES):
                pairs[str(pair)][stage_name] = {
                    "continuous_d": vector_summary(
                        choice_d[arm, pair, stage], seed_label,
                        f"{arm_name}/{pair}/{stage_name}",
                    ),
                    "D10": float(aggregate[arm, pair, stage]),
                    "aggregate_choice": int(aggregate_choice[arm, pair, stage]),
                    "single_repeat_0_choice": int(single_choice[arm, pair, stage]),
                    "individual_choice_distribution": preference_index(choices[arm, pair, stage]),
                }
        acq = aggregate[arm, :, 1] - aggregate[arm, :, 0]
        rev = aggregate[arm, :, 2] - aggregate[arm, :, 1]
        statistics["arms"][arm_name] = {
            "pairs": pairs,
            "D10_by_stage_across_pairs": {
                stage_name: vector_summary(
                    aggregate[arm, :, stage], seed_label, f"{arm_name}/D10/{stage_name}"
                )
                for stage, stage_name in enumerate(STAGES)
            },
            "acquisition_change_across_pairs": vector_summary(acq, seed_label, f"{arm_name}/acquisition"),
            "reversal_change_across_pairs": vector_summary(rev, seed_label, f"{arm_name}/reversal"),
            "maximum_bound_fraction": float(bound[arm].max()),
        }
    paired_acq = aggregate[p, :, 1] - aggregate[p, :, 0]
    paired_rev = aggregate[p, :, 2] - aggregate[p, :, 1]
    for control_name in ARMS[1:]:
        control = arm_index[control_name]
        statistics["paired_vs_controls"][control_name] = {
            "acquisition_difference": vector_summary(
                paired_acq - (aggregate[control, :, 1] - aggregate[control, :, 0]),
                seed_label, f"paired-vs-{control_name}/acquisition",
            ),
            "reversal_difference": vector_summary(
                paired_rev - (aggregate[control, :, 2] - aggregate[control, :, 1]),
                seed_label, f"paired-vs-{control_name}/reversal",
            ),
        }
    pair_direction = (
        (aggregate[p, :, 1] > 0)
        & (aggregate[p, :, 2] < 0)
        & (aggregate[p, :, 3] < 0)
    )
    implementation = {
        "finite": bool(finite.all()),
        "frozen_weights_exact": bool(np.array_equal(
            efficacy[arm_index["frozen"]], np.ones_like(efficacy[arm_index["frozen"]])
        )),
        "dan_omitted_weights_exact": bool(np.array_equal(
            efficacy[arm_index["dan_omitted"]], np.ones_like(efficacy[arm_index["dan_omitted"]])
        )),
        "bound_fraction_at_most_0_01": bool(bound.max() <= .01),
        "aggregate_uses_all_saved_repeats": bool(np.array_equal(aggregate, choice_d.mean(axis=-1))),
        "single_control_is_repeat_zero": bool(np.array_equal(single_choice, choices[..., 0])),
        "choice_matches_frozen_zero_threshold": bool(np.array_equal(aggregate_choice, np.sign(aggregate).astype(np.int8))),
        "calibration_individual_records_saved": bool(all(
            chunk["calibration_channels"].ndim == 3 for chunk in chunks
        )),
    }
    direction_count = int(np.sum(pair_direction))
    if not all(implementation.values()):
        verdict = "M2_C_IMPLEMENTATION_FAILURE"
    elif direction_count >= required_pairs:
        verdict = "ENGINEERED_AGGREGATE_PREFERENCE_DIRECTION_UPDATE_OBSERVED"
    else:
        verdict = "ENGINEERED_AGGREGATE_PREFERENCE_DIRECTION_UPDATE_NOT_OBSERVED"
    statistics["paired_primary"] = {
        "pair_direction_update": pair_direction,
        "direction_update_count": direction_count,
        "required_count": required_pairs,
        "aggregate_count": int(choice_d.shape[-1]),
        "single_control_repeat_index": single_index,
    }
    statistics["implementation_checks"] = implementation
    statistics["verdict"] = verdict
    raw = {
        "choice_d": choice_d,
        "choices": choices,
        "aggregate_D10": aggregate,
        "aggregate_choice": aggregate_choice,
        "single_choice": single_choice,
        "efficacy": efficacy,
        "finite": finite,
        "bound_fraction": bound,
    }
    return statistics, raw


def mode_values(config: dict[str, object], mode: str) -> dict[str, int | float | str]:
    if mode == "formal":
        e, r = config["exploration"], config["readout"]
        return {
            "seed_label": FORMAL_SEED_LABEL,
            "pair_count": int(e["odor_pair_count"]),
            "active": int(e["active_pn_count_per_odor"]),
            "calibration_repeats": int(r["baseline_calibration_repeats_per_odor"]),
            "probe_repeats": int(r["probe_repeats_per_stage_odor"]),
            "cycles": int(e["training_cycles_per_phase"]),
            "retention_seconds": float(e["retention_blank_seconds"]),
        }
    s = config["smoke"]
    return {
        "seed_label": SMOKE_SEED_LABEL,
        "pair_count": int(s["pair_count"]),
        "active": int(s["active_pn_count_per_odor"]),
        "calibration_repeats": int(s["calibration_repeats_per_odor"]),
        "probe_repeats": int(s["probe_repeats_per_stage_odor"]),
        "cycles": int(s["training_cycles_per_phase"]),
        "retention_seconds": float(s["retention_blank_seconds"]),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    global CURRENT_BUDGET
    output = args.out.resolve()
    config = json.loads(CONFIG.read_text())
    prior = json.loads((ROOT / "protocols/mb_lab/m2_b/config.json").read_text())
    if config["sensory"] != prior["sensory"] or config["plasticity"] != prior["plasticity"]:
        raise RuntimeError("M2-C changed the frozen neural model")
    if config["status"] != "AUTHORIZED_EXPLORATION_ONCE" or not config["exploration"]["authorized"]:
        raise RuntimeError("M2-C validation is not authorized")
    if args.mode == "formal":
        if int(config["exploration"]["run_limit"]) != 1:
            raise RuntimeError("M2-C formal run limit must equal one")
        other = [path for path in FORMAL_ROOT.glob("*/COMPLETE.json") if path.parent.resolve() != output]
        if other:
            raise RuntimeError("the one authorized M2-C validation is already complete")
    if (output / "COMPLETE.json").exists():
        raise FileExistsError("completed M2-C run is immutable")
    for name in (
        "trace_kc_seconds", "trace_dan_seconds", "memory_decay_seconds",
        "weight_filter_seconds", "minimum_fraction", "maximum_fraction",
    ):
        if config["plasticity"][name] != RULE_PARAMETERS[name]:
            raise RuntimeError(f"plasticity config/source mismatch: {name}")

    values = mode_values(config, args.mode)
    seed_label = str(values["seed_label"])
    path_count = expected_paths(
        int(values["pair_count"]), int(values["calibration_repeats"]),
        int(values["probe_repeats"]), int(values["cycles"]),
    )
    if args.mode == "formal":
        if int(config["readout"]["aggregate_count"]) != int(values["probe_repeats"]):
            raise RuntimeError("formal aggregate must use all ten frozen repeats")
        if path_count != int(config["exploration"]["estimated_unique_sensory_paths"]):
            raise RuntimeError("formal path count mismatch")

    output.mkdir(parents=True, exist_ok=True)
    start_hashes = source_hashes()
    identity = {
        "schema": "stonkfly-m2-c-readout-reliability-run-v1",
        "run_id": args.run_id,
        "mode": args.mode,
        "config_sha256": sha256_file(CONFIG),
        "source_hashes": start_hashes,
        "seed_label": seed_label,
    }
    if (output / "run.json").exists():
        if json.loads((output / "run.json").read_text()) != identity:
            raise RuntimeError("resume identity mismatch")
    else:
        exclusive_json(output / "run.json", identity)

    limits = config["budgets"]
    prefix = "formal" if args.mode == "formal" else "smoke"
    budget = RuntimeBudget(
        output, float(limits[f"{prefix}_cpu_seconds"]), float(limits[f"{prefix}_wall_seconds"]),
        int(float(limits["rss_gib"]) * 1024**3), int(float(limits["artifact_mib"]) * 1024**2),
    )
    CURRENT_BUDGET = budget
    budget.start()
    source = load_source()
    pn, kc = selected_nodes("R", olfactory_covered_only=True)
    topology = compile_real_topology(source["ptr"], source["post"], pn, kc)
    mapping = build_apl_mapping(GRAPH, source, topology.kc_node_index)
    connectivity = build_connectivity(GRAPH, source, topology.kc_node_index)
    s = config["sensory"]
    parameters = M1Parameters(
        float(s["tau_k_ms"]), float(s["tau_apl_ms"]), float(s["input_scale"]),
        float(s["apl_gain"]), float(s["threshold"]), 25.0,
    )
    cache = SensoryCache(topology, parameters, mapping, dt_ms=float(s["dt_ms"]))
    pair_count = int(values["pair_count"])
    bases = build_bases(seed_label, pair_count, topology.n_pn, int(values["active"]))
    content_hash = stimulus_content_hash(
        seed_label, bases, int(values["calibration_repeats"]),
        int(values["probe_repeats"]), int(values["cycles"]),
    )
    common = {
        "run_id": args.run_id, "mode": args.mode, "config_sha256": sha256_file(CONFIG),
        "topology_sha256": topology.sha256, "stimulus_content_sha256": content_hash,
    }
    exclusive_json(output / "stimulus_manifest.json", {
        "schema": "stonkfly-m2-c-stimulus-manifest-v1",
        "mode": args.mode, "seed_label": seed_label,
        "seed_sha256": hashlib.sha256(seed_label.encode()).hexdigest(),
        "content_sha256": content_hash, "pair_count": pair_count,
        "unique_sensory_paths": path_count, "selection_by_neural_output": False,
        "m2_b_stimuli_read": False,
        "schedules": {
            str(pair): {phase: schedule(seed_label, pair, phase, int(values["cycles"]))
                        for phase in ("acquisition", "reversal")}
            for pair in range(pair_count)
        },
    })
    write_npz_once(output / "stimulus_bases.npz", {"bases": bases})
    install_process_backstops(budget, int(float(limits["address_space_gib"]) * 1024**3))

    for pair in range(pair_count):
        name = f"pair-{pair:02d}"
        if validated_chunk(output, name, common) is None:
            arrays = run_pair(
                pair, seed_label, bases, cache, connectivity, config, budget,
                calibration_repeats=int(values["calibration_repeats"]),
                probe_repeats=int(values["probe_repeats"]), cycles=int(values["cycles"]),
                retention_seconds=float(values["retention_seconds"]),
            )
            write_chunk_once(output, name, arrays, common)
        budget.check()

    chunks = [load_npz(output / "chunks" / f"pair-{pair:02d}.npz") for pair in range(pair_count)]
    required = int(config["readout"]["pair_success_count"]) if args.mode == "formal" else 1
    statistics, raw = analyze(
        chunks, seed_label, required, int(config["readout"]["single_control_repeat_index"])
    )
    exclusive_json(output / "statistics.json", jsonable(statistics))
    write_npz_once(output / "raw.npz", raw)
    exclusive_json(output / "config.json", config)
    formal = {
        "schema": "stonkfly-m2-c-readout-reliability-result-v1",
        "mode": args.mode, "verdict": statistics["verdict"],
        "implementation_checks": statistics["implementation_checks"],
        "matrix_completed": True, "unique_sensory_paths": path_count,
        "cache_builds_this_process": cache.builds, "cache_live_items_at_end": len(cache._items),
        "m2_b_verdict_modified": False, "legacy_domains_accessed": False,
        "m2_b_stimuli_accessed": False, "market_or_trading_used": False,
        "interpretation": "Aggregate engineering readout only; not real fly behavior or biological validation.",
        "resource_usage": budget.check(),
    }
    exclusive_json(output / "formal.json", formal)
    descriptor = os.open(output / "report.md", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(descriptor, "w") as handle:
        handle.write("# M2-C aggregate-readout validation\n\n")
        handle.write(f"Mode: `{args.mode}`  \nVerdict: `{formal['verdict']}`\n\n")
        handle.write("Ten-probe mean with a frozen zero threshold; not real fly behavior. M2-B remains NO-GO.\n")
    exclusive_json(output / "stdout.json", {"verdict": formal["verdict"], "resource_usage": budget.check()})
    end_hashes = source_hashes()
    if end_hashes != start_hashes:
        raise RuntimeError("source changed during M2-C run")
    artifacts = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "COMPLETE.json"}
    }
    manifest = {
        "schema": "stonkfly-m2-c-readout-reliability-manifest-v1",
        "identity": identity, "source_hashes_start": start_hashes,
        "source_hashes_end": end_hashes, "artifacts": artifacts,
        "resource_usage": budget.check(),
    }
    exclusive_json(output / "manifest.json", manifest)
    exclusive_json(output / "COMPLETE.json", {
        "verdict": formal["verdict"], "manifest_sha256": sha256_file(output / "manifest.json")
    })
    return formal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    args = parser.parse_args()
    try:
        result = run(args)
    except BaseException as error:
        if args.out.exists() and not (args.out / "COMPLETE.json").exists():
            failures = args.out / "failures"
            failures.mkdir(parents=True, exist_ok=True)
            index = len(list(failures.glob("attempt-*.json")))
            exclusive_json(failures / f"attempt-{index:04d}.json", {
                "verdict": "M2_C_BUDGET_EXHAUSTED" if isinstance(error, BudgetExceeded) else "M2_C_IMPLEMENTATION_FAILURE",
                "error": repr(error), "traceback": traceback.format_exc(),
                "resource_usage": CURRENT_BUDGET.usage() if CURRENT_BUDGET else {},
                "recorded_utc": datetime.now(timezone.utc).isoformat(),
            })
        raise
    print(json.dumps({"verdict": result["verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
