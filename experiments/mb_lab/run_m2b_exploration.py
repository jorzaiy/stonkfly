#!/usr/bin/env python3
"""Run the fixed small M2-B virtual two-choice exploration or its isolated smoke."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import tempfile
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

CONFIG = ROOT / "protocols/mb_lab/m2_b/config.json"
GRAPH = ROOT / "data/graph.npz"
FORMAL_ROOT = ROOT / "runs/mb_lab/m2-b-exploration"
FORMAL_SEED_LABEL = "STONKFLY-V2/M2-B/VIRTUAL-2AFC/v1"
SMOKE_SEED_LABEL = "STONKFLY-V2/M2-B/VIRTUAL-2AFC/SMOKE/v1"
ARMS = ("paired", "frozen", "shuffled", "unpaired", "dan_omitted")
STAGES = ("P0", "P1", "P2", "P3")
PROVENANCE = (
    CONFIG,
    ROOT / "docs/STONKFLY_V2_M2_B_DESIGN.md",
    ROOT / "stonkfly/mb_lab/e1.py",
    ROOT / "stonkfly/mb_lab/m2e.py",
    ROOT / "stonkfly/neural/rule.py",
    ROOT / "experiments/mb_lab/run_m2b_exploration.py",
    ROOT / "tests/mb_lab/test_m2_b.py",
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


def derived_seed(seed_label: str, *parts: object) -> int:
    canonical = "\x1f".join([seed_label, *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(canonical).digest()[:8], "little")


def build_bases(seed_label: str, pair_count: int, n_pn: int, active: int) -> np.ndarray:
    bases = np.zeros((pair_count, 2, n_pn), dtype=np.float64)
    for pair in range(pair_count):
        for odor in range(2):
            rng = np.random.default_rng(derived_seed(seed_label, "base", pair, odor))
            selected = np.sort(rng.choice(n_pn, size=active, replace=False))
            bases[pair, odor, selected] = rng.uniform(.8, 1.0, size=active)
    return bases


def stimulus(
    seed_label: str,
    bases: np.ndarray,
    pair: int,
    category: str,
    odor: int,
    index: int,
) -> np.ndarray:
    base = bases[pair, odor]
    selected = np.flatnonzero(base)
    rng = np.random.default_rng(
        derived_seed(seed_label, "pulse", pair, category, odor, index)
    )
    pulse = np.zeros(base.shape, dtype=np.float64)
    pulse[selected] = np.clip(
        base[selected] + rng.normal(0.0, .05, len(selected)), 0.0, 1.0
    )
    return pulse


def schedule(seed_label: str, pair: int, phase: str, cycles: int) -> dict[str, object]:
    if cycles <= 0 or cycles % 2:
        raise ValueError("balanced shuffled schedule requires positive even cycles")
    correct = ["reward", "aversive"] if phase == "acquisition" else ["aversive", "reward"]
    rng_order = np.random.default_rng(derived_seed(seed_label, "order", pair, phase))
    orders = [
        [0, 1] if rng_order.integers(0, 2) == 0 else [1, 0]
        for _ in range(cycles)
    ]
    shuffled = []
    for odor in range(2):
        labels = np.asarray(["reward"] * (cycles // 2) + ["aversive"] * (cycles // 2))
        rng = np.random.default_rng(
            derived_seed(seed_label, "shuffled", pair, phase, odor)
        )
        shuffled.append(labels[rng.permutation(cycles)].tolist())
    return {"correct": correct, "orders": orders, "shuffled": shuffled}


def expected_paths(pair_count: int, calibration_repeats: int, probe_repeats: int, cycles: int) -> int:
    return pair_count * (
        2 * calibration_repeats + 4 * 2 * probe_repeats + 2 * 2 * cycles
    )


def stimulus_content_hash(
    seed_label: str,
    bases: np.ndarray,
    calibration_repeats: int,
    probe_repeats: int,
    cycles: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"STONKFLY-M2-B/VIRTUAL-2AFC-STIMULI/v1")
    for pair in range(len(bases)):
        for odor in range(2):
            for repeat in range(calibration_repeats):
                digest.update(stimulus(seed_label, bases, pair, "calibration", odor, repeat).tobytes())
        for stage in STAGES:
            for odor in range(2):
                for repeat in range(probe_repeats):
                    digest.update(stimulus(seed_label, bases, pair, f"choice-{stage}", odor, repeat).tobytes())
        for phase in ("acquisition", "reversal"):
            for odor in range(2):
                for cycle in range(cycles):
                    digest.update(stimulus(seed_label, bases, pair, f"train-{phase}", odor, cycle).tobytes())
    return digest.hexdigest()


def write_npz_once(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _bootstrap_ci(values: np.ndarray, seed_label: str, label: str) -> list[float]:
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(derived_seed(seed_label, "bootstrap", label))
    draws = rng.integers(0, len(x), size=(10000, len(x)))
    means = x[draws].mean(axis=1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def vector_summary(values: np.ndarray, seed_label: str, label: str) -> dict[str, object]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "values": x,
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "bootstrap_mean_95ci": _bootstrap_ci(x, seed_label, label),
    }


def preference_index(choices: np.ndarray) -> dict[str, int | float | None]:
    values = np.asarray(choices, dtype=np.int8)
    n_a = int(np.sum(values > 0))
    n_b = int(np.sum(values < 0))
    ties = int(np.sum(values == 0))
    denominator = n_a + n_b
    return {
        "A": n_a,
        "B": n_b,
        "tie": ties,
        "PI": None if denominator == 0 else float((n_a - n_b) / denominator),
    }


def install_process_backstops(budget: RuntimeBudget, address_space_bytes: int) -> None:
    """Keep the frozen RSS metric distinct from the kernel virtual-AS ceiling."""
    remaining_cpu = max(1, math.ceil(budget.cpu_limit_seconds - budget.prior_cpu_seconds))
    resource.setrlimit(resource.RLIMIT_CPU, (remaining_cpu, remaining_cpu + 1))
    resource.setrlimit(resource.RLIMIT_AS, (address_space_bytes, address_space_bytes))

    def cpu_exhausted(_signum, _frame):
        raise BudgetExceeded("M2_B_BUDGET_EXHAUSTED kernel CPU limit")

    signal.signal(signal.SIGXCPU, cpu_exhausted)


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
    baseline_channels = np.empty((2, 2), dtype=np.float64)
    baseline_targets = np.empty((2, 3), dtype=np.float64)
    for odor in range(2):
        channel_items = []
        target_items = []
        for repeat in range(calibration_repeats):
            pulse = stimulus(seed_label, bases, pair, "calibration", odor, repeat)
            key, item = cache.get(pulse)
            raw = raw_readout(connectivity, item.integrated_rate / 500.0, np.ones(n_edge))
            channel_items.append(raw["channels"])
            target_items.append(raw["per_target"])
            cache.discard(key)
            budget.check()
        baseline_channels[odor] = np.mean(channel_items, axis=0)
        baseline_targets[odor] = np.mean(target_items, axis=0)
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
                raise RuntimeError("M2-B arms share learning state")

    values = np.empty((len(ARMS), len(STAGES), 2, probe_repeats), dtype=np.float64)
    choice_d = np.empty((len(ARMS), len(STAGES), probe_repeats), dtype=np.float64)
    choices = np.empty((len(ARMS), len(STAGES), probe_repeats), dtype=np.int8)
    channels = np.empty((len(ARMS), len(STAGES), 2, 2), dtype=np.float64)
    targets = np.empty((len(ARMS), len(STAGES), 2, 3), dtype=np.float64)
    efficacy = np.empty((len(ARMS), len(STAGES), n_edge), dtype=np.float64)
    finite = np.ones((len(ARMS), len(STAGES)), dtype=bool)

    def probe(stage_index: int) -> None:
        stage = STAGES[stage_index]
        raw_channels = np.empty((len(ARMS), 2, probe_repeats, 2), dtype=np.float64)
        raw_targets = np.empty((len(ARMS), 2, probe_repeats, 3), dtype=np.float64)
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
                    raw_channels[arm_index, odor, repeat] = current["channels"]
                    raw_targets[arm_index, odor, repeat] = current["per_target"]
                    if state.weight_digest() != before:
                        raise RuntimeError("reward-free choice probe changed weights")
                cache.discard(key)
                budget.check()
        choice_d[:, stage_index] = (
            values[:, stage_index, 0] - values[:, stage_index, 1]
        )
        choices[:, stage_index] = np.sign(choice_d[:, stage_index]).astype(np.int8)
        channels[:, stage_index] = raw_channels.mean(axis=2)
        targets[:, stage_index] = raw_targets.mean(axis=2)
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

    bound_fraction = np.mean((efficacy <= .3) | (efficacy >= 2.0), axis=-1)
    changed_edges = np.count_nonzero(efficacy != 1.0, axis=-1)
    return {
        "baseline_channels": baseline_channels,
        "baseline_targets": baseline_targets,
        "values": values,
        "choice_d": choice_d,
        "choices": choices,
        "channels": channels,
        "targets": targets,
        "efficacy": efficacy,
        "changed_edges": changed_edges,
        "finite": finite,
        "bound_fraction": bound_fraction,
    }


def analyze(chunks: list[dict[str, np.ndarray]], seed_label: str, required_pairs: int) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    choice_d = np.stack([chunk["choice_d"] for chunk in chunks], axis=1)
    choices = np.stack([chunk["choices"] for chunk in chunks], axis=1)
    efficacy = np.stack([chunk["efficacy"] for chunk in chunks], axis=1)
    finite = np.stack([chunk["finite"] for chunk in chunks], axis=1)
    bound = np.stack([chunk["bound_fraction"] for chunk in chunks], axis=1)
    D = choice_d.mean(axis=-1)
    medians = np.median(choice_d, axis=-1)
    arm_index = {name: index for index, name in enumerate(ARMS)}
    p = arm_index["paired"]
    statistics: dict[str, object] = {
        "schema": "stonkfly-m2-b-virtual-2afc-statistics-v1",
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
                    "choice": preference_index(choices[arm, pair, stage]),
                }
        acq = D[arm, :, 1] - D[arm, :, 0]
        rev = D[arm, :, 2] - D[arm, :, 1]
        statistics["arms"][arm_name] = {
            "pairs": pairs,
            "D_by_stage_across_pairs": {
                stage_name: vector_summary(D[arm, :, stage], seed_label, f"{arm_name}/D/{stage_name}")
                for stage, stage_name in enumerate(STAGES)
            },
            "acquisition_change_across_pairs": vector_summary(acq, seed_label, f"{arm_name}/acquisition"),
            "reversal_change_across_pairs": vector_summary(rev, seed_label, f"{arm_name}/reversal"),
            "maximum_bound_fraction": float(bound[arm].max()),
        }
    paired_acq = D[p, :, 1] - D[p, :, 0]
    paired_rev = D[p, :, 2] - D[p, :, 1]
    for control_name in ARMS[1:]:
        control = arm_index[control_name]
        control_acq = D[control, :, 1] - D[control, :, 0]
        control_rev = D[control, :, 2] - D[control, :, 1]
        statistics["paired_vs_controls"][control_name] = {
            "acquisition_difference": vector_summary(
                paired_acq - control_acq, seed_label, f"paired-vs-{control_name}/acquisition"
            ),
            "reversal_difference": vector_summary(
                paired_rev - control_rev, seed_label, f"paired-vs-{control_name}/reversal"
            ),
        }
    pair_direction = (
        (medians[p, :, 1] > 0)
        & (medians[p, :, 2] < 0)
        & (np.sign(medians[p, :, 3]) == np.sign(medians[p, :, 2]))
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
        "choice_matches_zero_threshold": bool(np.array_equal(choices, np.sign(choice_d).astype(np.int8))),
    }
    direction_count = int(np.sum(pair_direction))
    if not all(implementation.values()):
        verdict = "M2_B_IMPLEMENTATION_FAILURE"
    elif direction_count >= required_pairs:
        verdict = "ENGINEERED_VIRTUAL_PREFERENCE_DIRECTION_UPDATE_OBSERVED"
    else:
        verdict = "ENGINEERED_VIRTUAL_PREFERENCE_DIRECTION_UPDATE_NOT_OBSERVED"
    statistics["paired_primary"] = {
        "pair_direction_update": pair_direction,
        "direction_update_count": direction_count,
        "required_count": required_pairs,
        "note": "Descriptive exploration; continuous magnitudes and controls remain mandatory.",
    }
    statistics["implementation_checks"] = implementation
    statistics["verdict"] = verdict
    raw = {
        "choice_d": choice_d,
        "choices": choices,
        "D": D,
        "medians": medians,
        "efficacy": efficacy,
        "finite": finite,
        "bound_fraction": bound,
    }
    return statistics, raw


def _mode_values(config: dict[str, object], mode: str) -> dict[str, int | float | str]:
    if mode == "formal":
        e = config["exploration"]
        c = config["choice"]
        return {
            "seed_label": FORMAL_SEED_LABEL,
            "pair_count": int(e["odor_pair_count"]),
            "active": int(e["active_pn_count_per_odor"]),
            "calibration_repeats": int(c["baseline_calibration_repeats_per_odor"]),
            "probe_repeats": int(c["probe_repeats_per_stage_odor"]),
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
    if config["status"] != "AUTHORIZED_EXPLORATION_ONCE" or not config["exploration"]["authorized"]:
        raise RuntimeError("M2-B exploration is not authorized")
    if args.mode == "formal":
        if int(config["exploration"]["run_limit"]) != 1:
            raise RuntimeError("M2-B formal run limit must equal one")
        other_completed = [
            path for path in FORMAL_ROOT.glob("*/COMPLETE.json")
            if path.parent.resolve() != output
        ]
        if other_completed:
            raise RuntimeError("the one authorized M2-B exploration run is already complete")
    if (output / "COMPLETE.json").exists():
        raise FileExistsError("completed M2-B run is immutable")
    for name in (
        "trace_kc_seconds", "trace_dan_seconds", "memory_decay_seconds",
        "weight_filter_seconds", "minimum_fraction", "maximum_fraction",
    ):
        if config["plasticity"][name] != RULE_PARAMETERS[name]:
            raise RuntimeError(f"plasticity config/source mismatch: {name}")

    mode_values = _mode_values(config, args.mode)
    seed_label = str(mode_values["seed_label"])
    path_count = expected_paths(
        int(mode_values["pair_count"]), int(mode_values["calibration_repeats"]),
        int(mode_values["probe_repeats"]), int(mode_values["cycles"]),
    )
    if args.mode == "formal" and path_count != int(config["exploration"]["estimated_unique_sensory_paths"]):
        raise RuntimeError("formal path count does not match frozen estimate")
    output.mkdir(parents=True, exist_ok=True)
    start_hashes = source_hashes()
    identity = {
        "schema": "stonkfly-m2-b-virtual-2afc-run-v1",
        "run_id": args.run_id,
        "mode": args.mode,
        "config_sha256": sha256_file(CONFIG),
        "source_hashes": start_hashes,
        "seed_label": seed_label,
    }
    run_path = output / "run.json"
    if run_path.exists():
        if json.loads(run_path.read_text()) != identity:
            raise RuntimeError("resume identity mismatch")
    else:
        exclusive_json(run_path, identity)

    limits = config["budgets"]
    prefix = "formal" if args.mode == "formal" else "smoke"
    budget = RuntimeBudget(
        output,
        float(limits[f"{prefix}_cpu_seconds"]),
        float(limits[f"{prefix}_wall_seconds"]),
        int(float(limits["rss_gib"]) * 1024**3),
        int(float(limits["artifact_mib"]) * 1024**2),
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
    pair_count = int(mode_values["pair_count"])
    bases = build_bases(seed_label, pair_count, topology.n_pn, int(mode_values["active"]))
    content_hash = stimulus_content_hash(
        seed_label, bases, int(mode_values["calibration_repeats"]),
        int(mode_values["probe_repeats"]), int(mode_values["cycles"]),
    )
    common = {
        "run_id": args.run_id,
        "mode": args.mode,
        "config_sha256": sha256_file(CONFIG),
        "topology_sha256": topology.sha256,
        "stimulus_content_sha256": content_hash,
    }
    exclusive_json(output / "stimulus_manifest.json", {
        "schema": "stonkfly-m2-b-stimulus-manifest-v1",
        "mode": args.mode,
        "seed_label": seed_label,
        "seed_sha256": hashlib.sha256(seed_label.encode()).hexdigest(),
        "content_sha256": content_hash,
        "pair_count": pair_count,
        "unique_sensory_paths": path_count,
        "selection_by_neural_output": False,
        "m2_e_stimuli_read": False,
        "schedules": {
            str(pair): {
                phase: schedule(seed_label, pair, phase, int(mode_values["cycles"]))
                for phase in ("acquisition", "reversal")
            }
            for pair in range(pair_count)
        },
    })
    write_npz_once(output / "stimulus_bases.npz", {"bases": bases})
    install_process_backstops(
        budget, int(float(limits["address_space_gib"]) * 1024**3)
    )

    for pair in range(pair_count):
        name = f"pair-{pair:02d}"
        if validated_chunk(output, name, common) is None:
            arrays = run_pair(
                pair, seed_label, bases, cache, connectivity, config, budget,
                calibration_repeats=int(mode_values["calibration_repeats"]),
                probe_repeats=int(mode_values["probe_repeats"]),
                cycles=int(mode_values["cycles"]),
                retention_seconds=float(mode_values["retention_seconds"]),
            )
            write_chunk_once(output, name, arrays, common)
        budget.check()

    chunks = [load_npz(output / "chunks" / f"pair-{pair:02d}.npz") for pair in range(pair_count)]
    required_pairs = int(config["choice"]["pair_success_count"])
    if args.mode == "smoke":
        required_pairs = 1
    statistics, raw = analyze(chunks, seed_label, required_pairs)
    exclusive_json(output / "statistics.json", jsonable(statistics))
    write_npz_once(output / "raw.npz", raw)
    exclusive_json(output / "config.json", config)
    usage = budget.check()
    formal = {
        "schema": "stonkfly-m2-b-virtual-2afc-result-v1",
        "mode": args.mode,
        "verdict": statistics["verdict"],
        "implementation_checks": statistics["implementation_checks"],
        "matrix_completed": True,
        "unique_sensory_paths": path_count,
        "cache_builds_this_process": cache.builds,
        "cache_live_items_at_end": len(cache._items),
        "m2_e_verdict_modified": False,
        "legacy_domains_accessed": False,
        "m2_e_stimuli_accessed": False,
        "market_or_trading_used": False,
        "interpretation": "Virtual engineering readout only; not real fly behavior or biological validation.",
        "resource_usage": usage,
    }
    exclusive_json(output / "formal.json", formal)
    descriptor = os.open(output / "report.md", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(descriptor, "w") as handle:
        handle.write("# M2-B virtual two-choice exploratory report\n\n")
        handle.write(f"Mode: `{args.mode}`  \nVerdict: `{formal['verdict']}`\n\n")
        handle.write("This is a fixed MBON-score comparison, not simulated or observed fly behavior. ")
        handle.write("M2-E remains NO-GO and its artifacts were not modified.\n")
    exclusive_json(output / "stdout.json", {
        "verdict": formal["verdict"], "resource_usage": budget.check(),
    })
    end_hashes = source_hashes()
    if end_hashes != start_hashes:
        raise RuntimeError("source changed during M2-B run")
    artifacts = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "COMPLETE.json"}
    }
    manifest = {
        "schema": "stonkfly-m2-b-virtual-2afc-manifest-v1",
        "identity": identity,
        "source_hashes_start": start_hashes,
        "source_hashes_end": end_hashes,
        "artifacts": artifacts,
        "resource_usage": budget.check(),
    }
    exclusive_json(output / "manifest.json", manifest)
    exclusive_json(output / "COMPLETE.json", {
        "verdict": formal["verdict"],
        "manifest_sha256": sha256_file(output / "manifest.json"),
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
                "verdict": "M2_B_BUDGET_EXHAUSTED" if isinstance(error, BudgetExceeded) else "M2_B_IMPLEMENTATION_FAILURE",
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "resource_usage": CURRENT_BUDGET.usage() if CURRENT_BUDGET else {},
                "recorded_utc": datetime.now(timezone.utc).isoformat(),
            })
        raise
    print(json.dumps({"verdict": result["verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
