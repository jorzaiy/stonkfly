#!/usr/bin/env python3
"""Run exactly one fixed 8-pair M2-E exploratory association matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
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

from stonkfly.mb_lab.budget import BudgetExceeded, RuntimeBudget, exclusive_json
from stonkfly.mb_lab.checkpoint import sha256_file, validated_chunk, write_chunk_once
from stonkfly.mb_lab.dynamics import M1Parameters
from stonkfly.mb_lab.e1 import build_apl_mapping
from stonkfly.mb_lab.inventory import load_source, selected_nodes
from stonkfly.mb_lab.m2e import (
    PlasticityState, SensoryCache, build_connectivity, normalized_valence,
    raw_readout, train_event,
)
from stonkfly.mb_lab.topology import compile_real_topology
from stonkfly.neural.rule import PARAMETERS as RULE_PARAMETERS

CONFIG = ROOT / "protocols/mb_lab/m2_e/config.json"
GRAPH = ROOT / "data/graph.npz"
RUN_ROOT = ROOT / "runs/mb_lab/m2-e-exploration"
SEED_LABEL = "STONKFLY-V2/M2-E/EXPLORATION/v1"
ARMS = ("paired", "frozen", "shuffled", "unpaired", "dan_omitted")
STAGES = ("P0", "P1", "P2", "P3")
PROVENANCE = (
    CONFIG,
    ROOT / "docs/STONKFLY_V2_M2_E_DESIGN.md",
    ROOT / "docs/STONKFLY_V2_M2_E_PREPARATION_REVIEW.md",
    ROOT / "stonkfly/mb_lab/e1.py",
    ROOT / "stonkfly/mb_lab/m2e.py",
    ROOT / "stonkfly/neural/rule.py",
    ROOT / "experiments/mb_lab/run_m2e_exploration.py",
    ROOT / "experiments/mb_lab/m2e_attempt.py",
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


def source_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in PROVENANCE}


def derived_seed(*parts: object) -> int:
    canonical = "\x1f".join([SEED_LABEL, *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(canonical).digest()[:8], "little")


def build_bases(pair_count: int, n_pn: int, active: int) -> np.ndarray:
    bases = np.zeros((pair_count, 2, n_pn), dtype=np.float64)
    for pair in range(pair_count):
        for odor in range(2):
            rng = np.random.default_rng(derived_seed("base", pair, odor))
            selected = np.sort(rng.choice(n_pn, size=active, replace=False))
            bases[pair, odor, selected] = rng.uniform(.8, 1.0, size=active)
    return bases


def stimulus(bases: np.ndarray, pair: int, category: str, odor: int, index: int) -> np.ndarray:
    base = bases[pair, odor]
    selected = np.flatnonzero(base)
    rng = np.random.default_rng(derived_seed("pulse", pair, category, odor, index))
    pulse = np.zeros(base.shape, dtype=np.float64)
    pulse[selected] = np.clip(base[selected] + rng.normal(0.0, .05, len(selected)), 0.0, 1.0)
    return pulse


def stimulus_content_hash(bases: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(b"STONKFLY-M2-E/640-STIMULI/v1")
    for pair in range(len(bases)):
        for stage in STAGES:
            for odor in range(2):
                for repeat in range(5):
                    digest.update(stimulus(bases, pair, f"probe-{stage}", odor, repeat).tobytes())
        for phase in ("acquisition", "reversal"):
            for odor in range(2):
                for cycle in range(10):
                    digest.update(stimulus(bases, pair, f"train-{phase}", odor, cycle).tobytes())
    return digest.hexdigest()


def schedule(pair: int, phase: str) -> dict[str, object]:
    correct = (["reward", "aversive"] if phase == "acquisition" else ["aversive", "reward"])
    orders = []
    rng_order = np.random.default_rng(derived_seed("order", pair, phase))
    for _ in range(10):
        orders.append([0, 1] if rng_order.integers(0, 2) == 0 else [1, 0])
    shuffled = []
    for odor in range(2):
        labels = np.asarray(["reward"] * 5 + ["aversive"] * 5)
        rng = np.random.default_rng(derived_seed("shuffled", pair, phase, odor))
        shuffled.append(labels[rng.permutation(10)].tolist())
    return {"correct": correct, "orders": orders, "shuffled": shuffled}


def signflip_p(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    observed = abs(float(x.mean()))
    distribution = [abs(float(np.mean(x * np.asarray(signs)))) for signs in itertools.product((-1, 1), repeat=len(x))]
    return float(np.mean(np.asarray(distribution) >= observed - 1e-15))


def bootstrap_ci(values: np.ndarray, label: str) -> list[float]:
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(derived_seed("bootstrap", label))
    draws = rng.integers(0, len(x), size=(20000, len(x)))
    means = x[draws].mean(axis=1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def vector_summary(values: np.ndarray, label: str) -> dict[str, object]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "values": x, "mean": float(x.mean()), "median": float(np.median(x)),
        "bootstrap_mean_95ci": bootstrap_ci(x, label),
        "exact_two_sided_signflip_p": signflip_p(x),
    }


def write_npz_once(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_pair(pair: int, bases: np.ndarray, cache: SensoryCache, connectivity, config, budget) -> dict[str, np.ndarray]:
    eta = float(config["plasticity"]["effective_eta_per_second"])
    states = {name: PlasticityState.fresh(connectivity, eta=eta) for name in ARMS}
    for left, right in itertools.combinations(states.values(), 2):
        if any(np.shares_memory(getattr(left, field), getattr(right, field)) for field in ("kc_trace", "dan_trace", "memory_u", "memory_w")):
            raise RuntimeError("M2-E arms share learning state")
    n_edge = len(connectivity.pre_slot)
    D = np.empty((len(ARMS), len(STAGES)), dtype=np.float64)
    channels = np.empty((len(ARMS), len(STAGES), 2, 2), dtype=np.float64)
    targets = np.empty((len(ARMS), len(STAGES), 2, 3), dtype=np.float64)
    efficacy = np.empty((len(ARMS), len(STAGES), n_edge), dtype=np.float64)
    weight_hash = np.empty((len(ARMS), len(STAGES)), dtype="S64")
    finite = np.ones((len(ARMS), len(STAGES)), dtype=bool)

    def probe(stage_index: int) -> None:
        stage = STAGES[stage_index]
        values = np.empty((len(ARMS), 2, 5), dtype=np.float64)
        raw_channels = np.empty((len(ARMS), 2, 5, 2), dtype=np.float64)
        raw_targets = np.empty((len(ARMS), 2, 5, 3), dtype=np.float64)
        for odor in range(2):
            for repeat in range(5):
                pulse = stimulus(bases, pair, f"probe-{stage}", odor, repeat)
                key, item = cache.get(pulse)
                baseline = raw_readout(connectivity, item.integrated_rate / 500.0, np.ones(n_edge))
                if np.any(baseline["channels"] <= 1e-12):
                    raise RuntimeError("baseline MBON channel is not drivable")
                for arm_index, name in enumerate(ARMS):
                    state = states[name]
                    before = state.weight_digest()
                    current = raw_readout(connectivity, item.integrated_rate / 500.0, state.efficacy)
                    values[arm_index, odor, repeat] = normalized_valence(current["channels"], baseline["channels"])
                    raw_channels[arm_index, odor, repeat] = current["channels"]
                    raw_targets[arm_index, odor, repeat] = current["per_target"]
                    if state.weight_digest() != before:
                        raise RuntimeError("reward-free probe changed weights")
                cache.discard(key)
                budget.check()
        for arm_index, name in enumerate(ARMS):
            odor_means = values[arm_index].mean(axis=1)
            D[arm_index, stage_index] = odor_means[0] - odor_means[1]
            channels[arm_index, stage_index] = raw_channels[arm_index].mean(axis=1)
            targets[arm_index, stage_index] = raw_targets[arm_index].mean(axis=1)
            efficacy[arm_index, stage_index] = states[name].efficacy
            weight_hash[arm_index, stage_index] = states[name].weight_digest().encode()
            finite[arm_index, stage_index] = bool(
                np.isfinite(values[arm_index]).all()
                and np.isfinite(efficacy[arm_index, stage_index]).all()
            )

    def train_phase(phase: str) -> None:
        plan = schedule(pair, phase)
        for cycle, order in enumerate(plan["orders"]):
            for odor in order:
                pulse = stimulus(bases, pair, f"train-{phase}", odor, cycle)
                key, item = cache.get(pulse)
                for name, state in states.items():
                    valence = plan["shuffled"][odor][cycle] if name == "shuffled" else plan["correct"][odor]
                    train_event(
                        state, item, valence=valence,
                        timing="unpaired" if name == "unpaired" else "paired",
                        frozen=name == "frozen", omitted=name == "dan_omitted",
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
        state.blank(float(config["full_exploration"]["retention_blank_seconds"]), frozen=name == "frozen")
    probe(3)
    bound_fraction = np.mean((efficacy <= .3) | (efficacy >= 2.0), axis=-1)
    return {
        "D": D, "channels": channels, "targets": targets, "efficacy": efficacy,
        "weight_hash": weight_hash, "finite": finite, "bound_fraction": bound_fraction,
    }


def analyze(chunks: list[dict[str, np.ndarray]]) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    D = np.stack([chunk["D"] for chunk in chunks], axis=1)  # arm, replicate, stage
    efficacy = np.stack([chunk["efficacy"] for chunk in chunks], axis=1)
    finite = np.stack([chunk["finite"] for chunk in chunks], axis=1)
    bound = np.stack([chunk["bound_fraction"] for chunk in chunks], axis=1)
    arm_index = {name: index for index, name in enumerate(ARMS)}
    p = arm_index["paired"]
    acq = D[:, :, 1] - D[:, :, 0]
    reversal = D[:, :, 2] - D[:, :, 1]
    retention_ratio = np.divide(
        np.abs(D[p, :, 3]), np.abs(D[p, :, 2]),
        out=np.full(8, np.nan), where=np.abs(D[p, :, 2]) > 0,
    )
    statistics: dict[str, object] = {
        "schema": "stonkfly-m2-e-exploration-statistics-v1",
        "arms": {}, "paired_primary": {}, "paired_vs_controls": {},
    }
    for index, name in enumerate(ARMS):
        statistics["arms"][name] = {
            "D_by_stage": {stage: vector_summary(D[index, :, stage_index], f"{name}/{stage}") for stage_index, stage in enumerate(STAGES)},
            "acquisition_change": vector_summary(acq[index], f"{name}/acquisition"),
            "reversal_change": vector_summary(reversal[index], f"{name}/reversal"),
            "maximum_bound_fraction": float(bound[index].max()),
        }
    statistics["paired_primary"] = {
        "reversal_crossing_count_D_P2_below_zero": int(np.sum(D[p, :, 2] < 0)),
        "retention_same_sign_count": int(np.sum((np.sign(D[p, :, 3]) == np.sign(D[p, :, 2])) & (D[p, :, 2] != 0))),
        "retention_ratio": vector_summary(retention_ratio, "paired/retention-ratio"),
    }
    control_checks = {}
    for control in ARMS[1:]:
        index = arm_index[control]
        acq_delta = acq[p] - acq[index]
        rev_delta = reversal[p] - reversal[index]
        acq_summary = vector_summary(acq_delta, f"paired-vs-{control}/acquisition")
        rev_summary = vector_summary(rev_delta, f"paired-vs-{control}/reversal")
        statistics["paired_vs_controls"][control] = {
            "acquisition_difference": acq_summary,
            "reversal_difference": rev_summary,
        }
        control_checks[control] = bool(
            acq_summary["bootstrap_mean_95ci"][0] > 0
            and rev_summary["bootstrap_mean_95ci"][1] < 0
        )
    acq_summary = statistics["arms"]["paired"]["acquisition_change"]
    rev_summary = statistics["arms"]["paired"]["reversal_change"]
    frozen_exact = bool(np.array_equal(efficacy[arm_index["frozen"]], np.ones_like(efficacy[arm_index["frozen"]])))
    omitted_exact = bool(np.array_equal(efficacy[arm_index["dan_omitted"]], np.ones_like(efficacy[arm_index["dan_omitted"]])))
    criteria = {
        "acquisition_positive_ci": bool(acq_summary["bootstrap_mean_95ci"][0] > 0),
        "reversal_negative_ci": bool(rev_summary["bootstrap_mean_95ci"][1] < 0),
        "reversal_crossing_at_least_7_of_8": statistics["paired_primary"]["reversal_crossing_count_D_P2_below_zero"] >= 7,
        "retention_sign_at_least_7_of_8": statistics["paired_primary"]["retention_same_sign_count"] >= 7,
        "retention_median_at_least_0_75": bool(np.nanmedian(retention_ratio) >= .75),
        "paired_exceeds_each_control": bool(all(control_checks.values())),
        "frozen_weights_exact": frozen_exact,
        "dan_omitted_weights_exact": omitted_exact,
        "finite": bool(finite.all()),
        "bound_fraction_at_most_0_01": bool(bound.max() <= .01),
    }
    verdict = "ENGINEERED_ASSOCIATIVE_UPDATE_OBSERVED" if all(criteria.values()) else "ENGINEERED_ASSOCIATIVE_UPDATE_NOT_OBSERVED"
    statistics["criteria"] = criteria
    statistics["verdict"] = verdict
    raw = {"D": D, "acquisition_change": acq, "reversal_change": reversal,
           "retention_ratio": retention_ratio, "efficacy": efficacy,
           "finite": finite, "bound_fraction": bound}
    return statistics, raw


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def run(args: argparse.Namespace) -> dict[str, object]:
    global CURRENT_BUDGET
    output = args.out.resolve()
    config = json.loads(CONFIG.read_text())
    if config["status"] != "AUTHORIZED_EXPLORATION_ONCE" or not config["full_exploration"]["authorized"]:
        raise RuntimeError("M2-E exploration is not authorized")
    if int(config["full_exploration"]["run_limit"]) != 1:
        raise RuntimeError("M2-E run limit must equal one")
    other_completed = [path for path in RUN_ROOT.glob("*/COMPLETE.json") if path.parent.resolve() != output]
    if other_completed:
        raise RuntimeError("the one authorized M2-E exploration run is already complete")
    if (output / "COMPLETE.json").exists():
        raise FileExistsError("completed M2-E exploration is immutable")
    for name in ("trace_kc_seconds", "trace_dan_seconds", "memory_decay_seconds", "weight_filter_seconds", "minimum_fraction", "maximum_fraction"):
        if config["plasticity"][name] != RULE_PARAMETERS[name]:
            raise RuntimeError(f"plasticity config/source mismatch: {name}")
    output.mkdir(parents=True, exist_ok=True)
    start_hashes = source_hashes()
    identity = {"schema": "stonkfly-m2-e-exploration-run-v1", "run_id": args.run_id,
                "config_sha256": sha256_file(CONFIG), "source_hashes": start_hashes,
                "seed_label": SEED_LABEL}
    run_path = output / "run.json"
    if run_path.exists():
        if json.loads(run_path.read_text()) != identity:
            raise RuntimeError("resume identity mismatch")
    else:
        exclusive_json(run_path, identity)
    limits = config["budgets"]
    budget = RuntimeBudget(
        output, float(limits["future_exploration_cpu_hours"]) * 3600,
        float(limits["future_exploration_cpu_hours"]) * 3600,
        int(float(limits["smoke_rss_gib"]) * 1024**3),
        250 * 1024**2,
    )
    CURRENT_BUDGET = budget
    budget.start()
    source = load_source()
    pn, kc = selected_nodes("R", olfactory_covered_only=True)
    topology = compile_real_topology(source["ptr"], source["post"], pn, kc)
    apl_mapping = build_apl_mapping(GRAPH, source, topology.kc_node_index)
    connectivity = build_connectivity(GRAPH, source, topology.kc_node_index)
    s = config["sensory"]
    parameters = M1Parameters(float(s["tau_k_ms"]), float(s["tau_apl_ms"]), float(s["input_scale"]), float(s["apl_gain"]), float(s["threshold"]), 25.0)
    cache = SensoryCache(topology, parameters, apl_mapping, dt_ms=float(s["dt_ms"]))
    pair_count = int(config["full_exploration"]["odor_pair_count"])
    bases = build_bases(pair_count, topology.n_pn, int(config["smoke"]["active_pn_count_per_odor"]))
    content_hash = stimulus_content_hash(bases)
    common = {"run_id": args.run_id, "config_sha256": sha256_file(CONFIG),
              "topology_sha256": topology.sha256, "stimulus_content_sha256": content_hash}
    exclusive_json(output / "stimulus_manifest.json", {
        "schema": "stonkfly-m2-e-stimulus-manifest-v1", "seed_label": SEED_LABEL,
        "seed_sha256": hashlib.sha256(SEED_LABEL.encode()).hexdigest(),
        "content_sha256": content_hash, "pair_count": pair_count,
        "unique_sensory_paths": 640, "selection_by_neural_output": False,
        "schedules": {str(pair): {phase: schedule(pair, phase) for phase in ("acquisition", "reversal")} for pair in range(pair_count)},
    })
    write_npz_once(output / "stimulus_bases.npz", {"bases": bases})
    budget.install_process_limits()
    for pair in range(pair_count):
        name = f"pair-{pair:02d}"
        if validated_chunk(output, name, common) is None:
            arrays = run_pair(pair, bases, cache, connectivity, config, budget)
            write_chunk_once(output, name, arrays, common)
        budget.check()
    chunks = [load_npz(output / "chunks" / f"pair-{pair:02d}.npz") for pair in range(pair_count)]
    statistics, raw = analyze(chunks)
    exclusive_json(output / "statistics.json", jsonable(statistics))
    write_npz_once(output / "raw.npz", raw)
    exclusive_json(output / "config.json", config)
    usage = budget.check()
    formal = {
        "schema": "stonkfly-m2-e-exploration-result-v1",
        "verdict": statistics["verdict"], "criteria": statistics["criteria"],
        "full_matrix_completed": True, "unique_sensory_paths": 640,
        "cache_builds_this_process": cache.builds, "cache_live_items_at_end": len(cache._items),
        "legacy_domains_accessed": False, "market_or_trading_used": False,
        "interpretation": "Engineering association learning only; not biological validation or M1 promotion.",
        "resource_usage": usage,
    }
    exclusive_json(output / "formal.json", formal)
    report = output / "report.md"
    descriptor = os.open(report, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(descriptor, "w") as handle:
        handle.write("# M2-E exploratory association report\n\n")
        handle.write(f"Verdict: `{formal['verdict']}`\n\n")
        handle.write("This is an engineering association-learning result only. M1-RQ remains NO-GO; old RT/M2 were not accessed.\n")
    exclusive_json(output / "stdout.json", {"verdict": formal["verdict"], "resource_usage": budget.check()})
    end_hashes = source_hashes()
    if end_hashes != start_hashes:
        raise RuntimeError("source changed during M2-E exploration")
    artifacts = {str(path.relative_to(output)): sha256_file(path) for path in sorted(output.rglob("*")) if path.is_file() and path.name not in {"manifest.json", "COMPLETE.json"}}
    manifest = {"schema": "stonkfly-m2-e-exploration-manifest-v1", "identity": identity,
                "source_hashes_start": start_hashes, "source_hashes_end": end_hashes,
                "artifacts": artifacts, "resource_usage": budget.check()}
    exclusive_json(output / "manifest.json", manifest)
    exclusive_json(output / "COMPLETE.json", {"verdict": formal["verdict"],
                                                "manifest_sha256": sha256_file(output / "manifest.json")})
    return formal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        result = run(args)
    except BaseException as error:
        if args.out.exists() and not (args.out / "COMPLETE.json").exists():
            failures = args.out / "failures"
            failures.mkdir(parents=True, exist_ok=True)
            index = len(list(failures.glob("attempt-*.json")))
            exclusive_json(failures / f"attempt-{index:04d}.json", {
                "verdict": "M2_E_IMPLEMENTATION_FAILURE" if not isinstance(error, BudgetExceeded) else "M2_E_BUDGET_EXHAUSTED",
                "error": repr(error), "traceback": traceback.format_exc(),
                "resource_usage": CURRENT_BUDGET.usage() if CURRENT_BUDGET else {},
                "recorded_utc": datetime.now(timezone.utc).isoformat(),
            })
        raise
    print(json.dumps({"verdict": result["verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
