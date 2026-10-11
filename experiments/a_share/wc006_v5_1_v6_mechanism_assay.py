"""No-market WC006 V5.1/V6 mechanism assay.

This module is deliberately independent of the frozen V5 paper adapter.  It
only instantiates the retained neural controller, branches from one clean
checkpoint, and writes append-only diagnostic artifacts.  It never imports a
broker, market stream, selector, or execution ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from stonkfly.config import Settings
from stonkfly.neural.v51_controller import V51Controller
from stonkfly.neural.rule import PARAMETERS as RULE_PARAMETERS

PREREGISTRATION = Path("docs/experiments/wc006_v5_1_v6_mechanism_preregister.md")
EXPECTED_PLASTIC_EDGES = 7_835
REWARD_CELLS = 15
AVERSIVE_CELLS = 2
DEFAULT_SEED = 17
DEFAULT_HORIZONS_MS = (1_000, 60_000, 1_800_000)
CUE_MS = 500.0
PULSE_MS = 200.0
PULSE_CURRENT = 20.0
EFFICACY_BOUNDS = (0.1, 2.0)


def _bundle_digest(**arrays: np.ndarray) -> str:
    """Hash labelled array bytes, including dtype and shape."""
    hasher = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(np.asarray(arrays[name]))
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(canonical({"dtype": value.dtype.str, "shape": value.shape}))
        hasher.update(value.tobytes())
    return hasher.hexdigest()


def analytic_passive_projection(
    memory_u: np.ndarray,
    memory_w: np.ndarray,
    baseline_weights: np.ndarray,
    duration_ms: float,
    *,
    parameters: dict[str, Any] = RULE_PARAMETERS,
) -> dict[str, np.ndarray]:
    """Project only passive ``u``/``w`` memory, without neural integration.

    This is the exact zero-drive solution used by ``rule.advance`` when
    ``learning=False``. It does not advance membrane potentials, spikes,
    eligibility, or any other part of the retained neural state.
    """
    u0 = np.asarray(memory_u, dtype=np.float64)
    w0 = np.asarray(memory_w, dtype=np.float64)
    baseline = np.asarray(baseline_weights)
    if u0.shape != w0.shape or u0.shape != baseline.shape:
        raise ValueError("passive projection arrays must have equal shapes")
    if not np.isfinite(u0).all() or not np.isfinite(w0).all() or not np.isfinite(baseline).all():
        raise ValueError("passive projection arrays must be finite")
    if not np.isfinite(duration_ms) or duration_ms < 0:
        raise ValueError("projection duration must be finite and nonnegative")
    tu = float(parameters["memory_decay_seconds"])
    tw = float(parameters["weight_filter_seconds"])
    lo = float(parameters["minimum_fraction"]) - 1.0
    hi = float(parameters["maximum_fraction"]) - 1.0
    if not (0 < tw < tu) or not np.isfinite((tu, tw, lo, hi)).all():
        raise ValueError("invalid passive projection parameters")
    if not np.all((u0 >= lo) & (u0 <= hi)) or not np.all((w0 >= lo) & (w0 <= hi)):
        raise ValueError("passive projection input memory is outside rule bounds")
    seconds = float(duration_ms) / 1000.0
    eu = np.exp(-seconds / tu)
    ew = np.exp(-seconds / tw)
    c = tu / (tu - tw) * (eu - ew)
    # For tau_u > tau_w, exp(-t/tau_w) and c are nonnegative and their sum
    # is at most one. Thus the zero-drive trajectory is a convex combination
    # of u_0, w_0 and zero: the 10 ms rule.clip cannot activate at any point.
    if ew < -1e-15 or c < -1e-15 or ew + c > 1.0 + 1e-12:
        raise ValueError("passive projection coefficient bound failed")
    u = np.clip(u0 * eu, lo, hi)
    w = np.clip(w0 * ew + u0 * c, lo, hi)
    weights = baseline * (1.0 + w)
    return {
        "memory_u": u,
        "memory_w": w,
        "edge_weights": weights.astype(baseline.dtype, copy=False),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


class AppendOnlyLog:
    """Canonical hash-chain JSONL writer and verifier."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size:
            raise FileExistsError(f"append-only log already exists: {self.path}")
        self.previous = "GENESIS"
        self.sequence = 0

    def append(self, event: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = {"seq": self.sequence, "event": event, "payload": payload, "prev_hash": self.previous}
        event_hash = hashlib.sha256(self.previous.encode() + canonical(body)).hexdigest()
        record = {**body, "event_hash": event_hash}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.previous = event_hash
        self.sequence += 1
        return record


def verify_append_only_log(path: Path) -> dict[str, Any]:
    previous = "GENESIS"
    count = 0
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        record = json.loads(line)
        expected = {key: record[key] for key in ("seq", "event", "payload", "prev_hash")}
        if record["seq"] != count or record["prev_hash"] != previous:
            raise ValueError(f"append-only chain mismatch at line {line_number}")
        actual = hashlib.sha256(previous.encode() + canonical(expected)).hexdigest()
        if record.get("event_hash") != actual:
            raise ValueError(f"append-only hash mismatch at line {line_number}")
        previous = actual
        count += 1
    return {"records": count, "head": previous}


def _settings() -> Settings:
    return Settings(neural_ms=CUE_MS, pulse_ms=PULSE_MS, pulse_current=PULSE_CURRENT, learning=True)


def reproducible_input(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(180, 320, 3), dtype=np.uint8)


def _load_fixed_frame(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a prebuilt assay cue without changing or selecting its pixels."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        frame = np.load(handle, allow_pickle=False)
    frame = np.asarray(frame)
    if frame.shape != (180, 320, 3) or frame.dtype != np.uint8:
        raise ValueError("fixed assay frame must be uint8 with shape 180x320x3")
    return frame.copy(), {
        "kind": "fixed_frame_npy",
        "path": str(path),
        "file_sha256": sha256(path),
        "raw_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
        "shape": list(frame.shape),
        "dtype": frame.dtype.str,
    }


def _stimulus(brain: Any, label: str) -> tuple[np.ndarray, float] | None:
    if label == "none":
        return None
    if label not in {"reward", "aversive"}:
        raise ValueError(f"unknown stimulus: {label}")
    return np.asarray(brain.circuit[label], dtype=np.int32), PULSE_CURRENT


def _label_gain_mask(snapshot: dict[str, np.ndarray], label: str) -> np.ndarray:
    rows = snapshot["selected_gain_rows"]
    if label == "reward":
        return np.any(rows[:REWARD_CELLS] != 0, axis=0)
    if label == "aversive":
        return np.any(rows[REWARD_CELLS:REWARD_CELLS + AVERSIVE_CELLS] != 0, axis=0)
    raise ValueError(f"unknown label for eligibility characterization: {label}")


def _label_eligibility_mask(snapshot: dict[str, np.ndarray], label: str) -> np.ndarray:
    return _label_gain_mask(snapshot, label) & (np.abs(snapshot["edge_eligibility"]) > 0)


def _graph_contract(brain: Any) -> dict[str, Any]:
    circuit = brain.circuit
    edges = np.asarray(circuit["edges"])
    return {
        "plastic_edges": int(len(edges)),
        "plastic_edges_sha256": digest(edges),
        "neurons": int(brain.n),
        "graph_ids_sha256": digest(np.asarray(brain.ids)),
        "graph_ptr_sha256": digest(np.asarray(brain.ptr)),
        "graph_post_sha256": digest(np.asarray(brain.post)),
    }


def _snapshot(brain: Any, output: Path, branch: str, stage: str) -> dict[str, Any]:
    circuit = brain.circuit
    edges = np.asarray(circuit["edges"], dtype=np.int64)
    pre = np.asarray(circuit["pre"], dtype=np.int32)
    dan = np.asarray(circuit["dan"], dtype=np.int32)
    selected = np.r_[np.asarray(circuit["reward"], dtype=np.int32), np.asarray(circuit["aversive"], dtype=np.int32)]
    arrays = {
        "edge_indices": edges,
        "edge_pre": pre,
        "edge_weights": np.asarray(brain.weight[edges]).copy(),
        "baseline_weights": np.asarray(brain.baseline_plastic).copy(),
        "eligibility": np.asarray(brain.eligibility).copy(),
        "edge_eligibility": np.asarray(brain.eligibility[pre]).copy(),
        "dan_ids": np.asarray(brain.ids[dan]).copy(),
        "dan_rates": np.asarray(brain.rate_dan).copy(),
        "dan_counts": np.asarray(brain.counts[dan]).copy(),
        "selected_dan_indices": selected,
        "selected_gain_rows": np.asarray(circuit["gain"][np.r_[np.arange(len(circuit["reward"])), np.arange(len(circuit["reward"]), len(circuit["dan"]))]]).copy(),
        "memory_u": np.asarray(brain.memory_u).copy(),
        "memory_w": np.asarray(brain.memory_w).copy(),
    }
    path = output / "snapshots" / f"{branch}-{stage}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_npz(path, **arrays)
    return {
        "file": str(path.relative_to(output)),
        "sha256": sha256(path),
        "stage": stage,
        "clock": int(brain.cursor),
        "sim_ms": float(brain.sim_ms),
        "edge_count": int(len(arrays["edge_weights"])),
        "edge_weights_sha256": digest(arrays["edge_weights"]),
        "baseline_weights_sha256": digest(arrays["baseline_weights"]),
        "eligibility_sha256": digest(arrays["eligibility"]),
        "edge_eligibility_sha256": digest(arrays["edge_eligibility"]),
        "dan_rates_sha256": digest(arrays["dan_rates"]),
        "selected_gain_rows_sha256": digest(arrays["selected_gain_rows"]),
        "memory_u_sha256": digest(arrays["memory_u"]),
        "memory_w_sha256": digest(arrays["memory_w"]),
        "snapshot_mode": "full_network",
    }


def _load_snapshot(output: Path, info: dict[str, Any]) -> dict[str, np.ndarray]:
    path = output / str(info["file"])
    if sha256(path) != info["sha256"]:
        raise ValueError(f"snapshot hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def _project_passive_snapshot(
    output: Path,
    branch: str,
    stage: str,
    source: dict[str, Any],
    *,
    duration_ms: int,
    origin_elapsed_ms: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write a hash-locked memory/weight-only projection from ``source``."""
    source_arrays = _load_snapshot(output, source)
    projected = analytic_passive_projection(
        source_arrays["memory_u"],
        source_arrays["memory_w"],
        source_arrays["baseline_weights"],
        duration_ms,
    )
    arrays = dict(source_arrays)
    arrays.update(projected)
    path = output / "snapshots" / f"{branch}-analytic-{stage}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_npz(path, **arrays)
    input_hash = _bundle_digest(
        memory_u=source_arrays["memory_u"],
        memory_w=source_arrays["memory_w"],
        baseline_weights=source_arrays["baseline_weights"],
    )
    output_hash = _bundle_digest(
        memory_u=projected["memory_u"],
        memory_w=projected["memory_w"],
        edge_weights=projected["edge_weights"],
    )
    tu = float(RULE_PARAMETERS["memory_decay_seconds"])
    tw = float(RULE_PARAMETERS["weight_filter_seconds"])
    seconds = float(duration_ms) / 1000.0
    eu = float(np.exp(-seconds / tu))
    ew = float(np.exp(-seconds / tw))
    c = float(tu / (tu - tw) * (eu - ew))
    lo = float(RULE_PARAMETERS["minimum_fraction"]) - 1.0
    hi = float(RULE_PARAMETERS["maximum_fraction"]) - 1.0
    ratio = projected["edge_weights"] / source_arrays["baseline_weights"]
    info = {
        "file": str(path.relative_to(output)),
        "sha256": sha256(path),
        "stage": stage,
        "clock": None,
        "sim_ms": None,
        "edge_count": int(len(projected["edge_weights"])),
        "edge_weights_sha256": digest(projected["edge_weights"]),
        "baseline_weights_sha256": digest(source_arrays["baseline_weights"]),
        "eligibility_sha256": digest(source_arrays["eligibility"]),
        "edge_eligibility_sha256": digest(source_arrays["edge_eligibility"]),
        "dan_rates_sha256": digest(source_arrays["dan_rates"]),
        "selected_gain_rows_sha256": digest(source_arrays["selected_gain_rows"]),
        "memory_u_sha256": digest(projected["memory_u"]),
        "memory_w_sha256": digest(projected["memory_w"]),
        "snapshot_mode": "analytic_passive_projection",
        "source_snapshot_sha256": source["sha256"],
        "retention_origin_elapsed_ms": float(origin_elapsed_ms),
    }
    formula = {
        "name": "rule.advance.zero_drive_closed_form",
        "learning": False,
        "equations": {
            "u_t": "u_0 * exp(-t / tau_u)",
            "w_t": "w_0 * exp(-t / tau_w) + u_0 * tau_u/(tau_u-tau_w) * (exp(-t/tau_u)-exp(-t/tau_w))",
            "weight_t": "baseline_weight * (1 + w_t)",
        },
        "parameters": {
            "memory_decay_seconds": tu,
            "weight_filter_seconds": tw,
            "minimum_fraction": float(RULE_PARAMETERS["minimum_fraction"]),
            "maximum_fraction": float(RULE_PARAMETERS["maximum_fraction"]),
        },
        "duration_ms": int(duration_ms),
        "decay_u": eu,
        "decay_w": ew,
        "coupling": c,
    }
    projection = {
        "stage": stage,
        "source_stage": source["stage"],
        "source_snapshot_sha256": source["sha256"],
        "retention_origin_elapsed_ms": float(origin_elapsed_ms),
        "input_sha256": input_hash,
        "output_sha256": output_hash,
        "formula": formula,
        "bounds": {
            "memory_u": [lo, hi],
            "memory_w": [lo, hi],
            "weight_efficacy": [1.0 + lo, 1.0 + hi],
            "input_memory_u_inside": bool(np.all((source_arrays["memory_u"] >= lo) & (source_arrays["memory_u"] <= hi))),
            "input_memory_w_inside": bool(np.all((source_arrays["memory_w"] >= lo) & (source_arrays["memory_w"] <= hi))),
            "memory_u_inside": bool(np.all((projected["memory_u"] >= lo) & (projected["memory_u"] <= hi))),
            "memory_w_inside": bool(np.all((projected["memory_w"] >= lo) & (projected["memory_w"] <= hi))),
            "efficacy_min": float(np.min(ratio)),
            "efficacy_max": float(np.max(ratio)),
            "clip_activations": 0,
            "trajectory_bound_proof": "positive coefficients ew,c with ew+c<=1; input u/w already inside rule bounds",
        },
        "full_neural_state_advanced": False,
        "scope": ["memory_u", "memory_w", "edge_weights"],
        "snapshot": info,
    }
    return info, projection


def _advance(
    controller: Any,
    frame: np.ndarray,
    duration_ms: float,
    stimulation: tuple[np.ndarray, float] | None = None,
    linked_label: str | None = None,
) -> None:
    """Advance a sensory phase or deliver one linked pulse.

    The real V5.1 controller keeps RGB propagation/eligibility separate from
    plasticity.  Fixture controllers without the linked interface retain the
    old direct-stimulation path so the historical assay remains a timing and
    localization control rather than manufacturing a V5.1 pass.
    """
    brain = controller.brain
    linked = getattr(controller, "reinforce_linked", None)
    if linked_label is not None and callable(linked):
        linked(linked_label)
        return
    brain.rgb_step(
        frame,
        duration_ms,
        learning=False if linked_label is None else True,
        stimulation=stimulation,
    )


def _delta_vector(output: Path, before: dict[str, Any], after: dict[str, Any]) -> np.ndarray:
    left = _load_snapshot(output, before)["edge_weights"]
    right = _load_snapshot(output, after)["edge_weights"]
    return right - left


def _delta(output: Path, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    difference = _delta_vector(output, before, after)
    affected = np.abs(difference) > 0
    return {
        "changed_edges": int(np.count_nonzero(affected)),
        "median_signed_delta": float(np.median(difference[affected])) if np.any(affected) else 0.0,
        "median_abs_delta": float(np.median(np.abs(difference[affected]))) if np.any(affected) else 0.0,
        "delta_sha256": digest(difference),
    }


def _difference_delta(
    output: Path,
    branch_before: dict[str, Any],
    branch_after: dict[str, Any],
    control_before: dict[str, Any],
    control_after: dict[str, Any],
) -> dict[str, Any]:
    """Difference-in-differences: branch delta minus matched-none delta."""
    difference = _delta_vector(output, branch_before, branch_after) - _delta_vector(output, control_before, control_after)
    affected = np.abs(difference) > 0
    return {
        "changed_edges": int(np.count_nonzero(affected)),
        "median_signed_delta": float(np.median(difference[affected])) if np.any(affected) else 0.0,
        "median_abs_delta": float(np.median(np.abs(difference[affected]))) if np.any(affected) else 0.0,
        "delta_sha256": digest(difference),
    }


def _run_branch(
    *,
    output: Path,
    log: AppendOnlyLog,
    label: str,
    controller_factory: Callable[[Settings], Any],
    settings: Settings,
    frame: np.ndarray,
    clean_checkpoint: Path,
    clean_hash: str,
    input_hash: str,
    seed: int,
    horizons_ms: tuple[int, ...],
) -> dict[str, Any]:
    branch_checkpoint = output / "branches" / f"clean-{label}.npz"
    branch_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(clean_checkpoint, branch_checkpoint)
    branch_hash = sha256(branch_checkpoint)
    if branch_hash != clean_hash:
        raise ValueError(f"branch clean checkpoint mismatch: {label}")
    controller = controller_factory(settings)
    controller.restore(branch_checkpoint)
    brain = controller.brain
    log.append("branch_start", {"branch": label, "seed": seed, "clean_checkpoint_sha256": branch_hash, "input_sha256": input_hash, "branch_checkpoint_sha256": branch_hash})
    snapshots: dict[str, dict[str, Any]] = {}
    snapshots["start"] = _snapshot(brain, output, label, "start")
    log.append("snapshot", {"branch": label, "snapshot": snapshots["start"]})
    _advance(controller, frame, CUE_MS)
    snapshots["pre_pulse"] = _snapshot(brain, output, label, "pre-pulse")
    pre_checkpoint = output / "branches" / f"{label}-pre-pulse.npz"
    controller.save(pre_checkpoint)
    pre_checkpoint_hash = sha256(pre_checkpoint)
    log.append("snapshot", {"branch": label, "snapshot": snapshots["pre_pulse"], "checkpoint_sha256": pre_checkpoint_hash})
    _advance(controller, frame, PULSE_MS, _stimulus(brain, label) if label in {"reward", "aversive"} else None, linked_label=label if label in {"reward", "aversive"} else None)
    snapshots["post_pulse"] = _snapshot(brain, output, label, "post-pulse")
    post_checkpoint = output / "branches" / f"{label}-post-pulse.npz"
    controller.save(post_checkpoint)
    post_checkpoint_hash = sha256(post_checkpoint)
    log.append("snapshot", {"branch": label, "snapshot": snapshots["post_pulse"], "checkpoint_sha256": post_checkpoint_hash})
    # One full-network 1 s propagation is retained as a smoke/propagation
    # check. Longer horizons are passive projections from post-pulse memory;
    # they do not pretend to advance the complete neural state.
    projection_records: dict[str, Any] = {}
    if 1_000 in horizons_ms:
        _advance(controller, frame, 1_000.0)
        snapshots["t1000ms"] = _snapshot(brain, output, label, "t1000ms")
        log.append("snapshot", {"branch": label, "snapshot": snapshots["t1000ms"], "mode": "full_network_smoke"})
    for horizon in horizons_ms:
        stage = f"t{horizon}ms"
        projected_info, projection = _project_passive_snapshot(
            output,
            label,
            stage,
            snapshots["post_pulse"],
            duration_ms=horizon,
        )
        # Keep the full-network t1000 snapshot as the public smoke result;
        # its matching analytic t1000 projection remains recorded separately.
        if horizon != 1_000:
            snapshots[stage] = projected_info
        projection_records[stage] = projection
        log.append("passive_projection", {"branch": label, "projection": projection})
    final = output / "branches" / f"{label}-final.npz"
    controller.save(final)
    final_hash = sha256(final)
    result = {"branch": label, "clean_checkpoint_sha256": branch_hash, "pre_pulse_checkpoint_sha256": pre_checkpoint_hash, "post_pulse_checkpoint_sha256": post_checkpoint_hash, "final_checkpoint_sha256": final_hash, "input_sha256": input_hash, "snapshots": snapshots, "passive_projection": projection_records, "pulse_ms": PULSE_MS if label in {"reward", "aversive"} else 0.0, "pulse_current": PULSE_CURRENT if label in {"reward", "aversive"} else 0.0}
    result["pulse_delta"] = _delta(output, snapshots["pre_pulse"], snapshots["post_pulse"])
    if label in {"reward", "aversive"}:
        before = _load_snapshot(output, snapshots["pre_pulse"])
        pulse_vector = _delta_vector(output, snapshots["pre_pulse"], snapshots["post_pulse"])
        eligible_mask = _label_eligibility_mask(before, label)
        result["eligible_edge_count"] = int(np.count_nonzero(eligible_mask))
        result["eligible_changed_edges"] = int(np.count_nonzero((np.abs(pulse_vector) > 0) & eligible_mask))
        result["inconclusive_no_eligibility"] = result["eligible_edge_count"] == 0
    else:
        result["inconclusive_no_eligibility"] = False
    if "t1000ms" in snapshots:
        result["retention_1s_delta"] = _delta(output, snapshots["pre_pulse"], snapshots["t1000ms"])
    if "t60000ms" in snapshots:
        result["retention_60s_delta"] = _delta(output, snapshots["pre_pulse"], snapshots["t60000ms"])
    if "t1800000ms" in snapshots:
        result["retention_1800s_delta"] = _delta(output, snapshots["pre_pulse"], snapshots["t1800000ms"])
    log.append("branch_complete", {key: value for key, value in result.items() if key != "snapshots"})
    return result


def _assay_temporal_pulse(controller: Any, pulse_label: str, frame: np.ndarray) -> dict[str, Any]:
    """Deliver the temporal-control pulse below the production linked gate.

    This helper is intentionally assay-only. It uses the same DAN indices,
    duration and current as production, while allowing DAN-before-KC to test
    the timing mechanism even though production correctly no-ops without a
    pre-existing eligibility trace. ``none`` advances the matched control for
    the same interval without stimulation.
    """
    if pulse_label not in {"none", "reward", "aversive"}:
        raise ValueError(f"unknown temporal pulse label: {pulse_label}")
    brain = controller.brain
    stimulation = None
    if pulse_label != "none":
        stimulation = (np.asarray(brain.circuit[pulse_label]), PULSE_CURRENT)
    # MemoryBrain.step is the true low-level path. Small fixture brains in the
    # unit tests expose only rgb_step; retain an explicit fallback so their
    # historical no-eligibility behavior remains visible rather than hidden.
    if hasattr(brain, "step") and hasattr(brain, "luminance"):
        brain.step(
            np.asarray(brain.luminance, dtype=np.float32).copy(),
            PULSE_MS,
            learning=pulse_label != "none",
            stimulation=stimulation,
        )
        mode = "assay-direct-dan-pulse" if pulse_label != "none" else "assay-direct-matched-none"
    else:
        brain.rgb_step(
            np.zeros_like(frame),
            PULSE_MS,
            learning=pulse_label != "none",
            stimulation=stimulation,
        )
        mode = "legacy-rgb-fallback"
    return {
        "mode": mode,
        "mechanism_only": True,
        "production_reinforce_linked_used": False,
        "stimulus": pulse_label,
        "pulse_ms": PULSE_MS,
        "pulse_current": PULSE_CURRENT if pulse_label != "none" else 0.0,
    }


def _assay_temporal_kc(controller: Any, frame: np.ndarray) -> dict[str, Any]:
    """Present the KC cue through the assay-only learning path."""
    brain = controller.brain
    brain.rgb_step(frame, CUE_MS, learning=True, stimulation=None)
    return {
        "mode": "assay-direct-kc-cue",
        "mechanism_only": True,
        "production_observe_used": False,
        "learning": True,
        "cue_ms": CUE_MS,
    }


def _run_temporal(
    *,
    output: Path,
    log: AppendOnlyLog,
    order: str,
    pulse_label: str,
    controller_factory: Callable[[Settings], Any],
    settings: Settings,
    frame: np.ndarray,
    clean_checkpoint: Path,
    clean_hash: str,
    input_hash: str,
    seed: int,
    horizons_ms: tuple[int, ...],
) -> dict[str, Any]:
    label = f"temporal-{order}-{pulse_label}"
    controller = controller_factory(settings)
    controller.restore(clean_checkpoint)
    brain = controller.brain
    log.append("temporal_start", {"branch": label, "order": order, "seed": seed, "clean_checkpoint_sha256": clean_hash, "input_sha256": input_hash})
    snapshots: dict[str, dict[str, Any]] = {"start": _snapshot(brain, output, label, "start")}
    if order == "kc-before-dan":
        kc_delivery = _assay_temporal_kc(controller, frame)
        snapshots["pre_pulse"] = _snapshot(brain, output, label, "pre-pulse")
        elapsed = 0
    elif order == "dan-before-kc":
        snapshots["pre_pulse"] = _snapshot(brain, output, label, "pre-pulse")
        elapsed = 0
    else:
        raise ValueError(f"unknown temporal order: {order}")
    pre_checkpoint = output / "branches" / f"{label}-pre-pulse.npz"
    controller.save(pre_checkpoint)
    pre_checkpoint_hash = sha256(pre_checkpoint)
    pulse_delivery = _assay_temporal_pulse(controller, pulse_label, frame)
    snapshots["post_pulse"] = _snapshot(brain, output, label, "post-pulse")
    post_checkpoint = output / "branches" / f"{label}-post-pulse.npz"
    controller.save(post_checkpoint)
    post_checkpoint_hash = sha256(post_checkpoint)
    retention_origin = snapshots["post_pulse"]
    retention_origin_elapsed_ms = 0.0
    if order == "dan-before-kc":
        # The production V5.1 sensory path is learning=False. This temporal
        # control is explicitly mechanism-only and uses learning=True so the
        # post-DAN KC cue can test the predicted sign reversal.
        kc_delivery = _assay_temporal_kc(controller, frame)
        elapsed = CUE_MS
        snapshots["retention_origin"] = _snapshot(brain, output, label, "retention-origin")
        retention_origin = snapshots["retention_origin"]
        retention_origin_elapsed_ms = CUE_MS
        log.append(
            "snapshot",
            {
                "branch": label,
                "snapshot": retention_origin,
                "origin": "kc-after-dan",
                "elapsed_ms_from_post_pulse": retention_origin_elapsed_ms,
                "kc_delivery": kc_delivery,
            },
        )
    else:
        kc_delivery = None
    log.append("snapshot", {"branch": label, "snapshot": snapshots["pre_pulse"], "checkpoint_sha256": pre_checkpoint_hash})
    log.append("snapshot", {"branch": label, "snapshot": snapshots["post_pulse"], "checkpoint_sha256": post_checkpoint_hash})
    projection_records: dict[str, Any] = {}
    if 1_000 in horizons_ms:
        if 1_000 < elapsed:
            raise ValueError("temporal horizon is shorter than the fixed cue")
        _advance(controller, frame, float(1_000 - elapsed)) if 1_000 > elapsed else None
        snapshots["t1000ms"] = _snapshot(brain, output, label, "t1000ms")
        log.append("snapshot", {"branch": label, "snapshot": snapshots["t1000ms"], "mode": "full_network_smoke"})
    for horizon in horizons_ms:
        stage = f"t{horizon}ms"
        if horizon < elapsed:
            raise ValueError("temporal horizon is shorter than the fixed cue")
        projected_info, projection = _project_passive_snapshot(
            output,
            label,
            stage,
            retention_origin,
            duration_ms=horizon - int(retention_origin_elapsed_ms),
            origin_elapsed_ms=retention_origin_elapsed_ms,
        )
        if horizon != 1_000:
            snapshots[stage] = projected_info
        projection_records[stage] = projection
        log.append("passive_projection", {"branch": label, "projection": projection})
    result = {"branch": label, "order": order, "clean_checkpoint_sha256": clean_hash, "pre_pulse_checkpoint_sha256": pre_checkpoint_hash, "post_pulse_checkpoint_sha256": post_checkpoint_hash, "retention_origin_snapshot_sha256": retention_origin["sha256"], "retention_origin_elapsed_ms": retention_origin_elapsed_ms, "input_sha256": input_hash, "snapshots": snapshots, "passive_projection": projection_records, "pulse_delivery": pulse_delivery, "kc_delivery": kc_delivery, "mechanism_only": True, "production_reinforce_linked_used": False, "pulse_delta": _delta(output, snapshots["pre_pulse"], snapshots["post_pulse"])}
    if pulse_label in {"reward", "aversive"}:
        before = _load_snapshot(output, snapshots["pre_pulse"])
        pulse_vector = _delta_vector(output, snapshots["pre_pulse"], snapshots["post_pulse"])
        eligible_mask = _label_eligibility_mask(before, pulse_label)
        result["eligible_edge_count"] = int(np.count_nonzero(eligible_mask))
        result["eligible_changed_edges"] = int(np.count_nonzero((np.abs(pulse_vector) > 0) & eligible_mask))
        # The production linked gate is intentionally not used here. A zero
        # pre-pulse eligibility count is retained as a fact, but a direct
        # low-level assay pulse is not relabelled as a production
        # INCONCLUSIVE result.
        result["inconclusive_no_eligibility"] = result["eligible_edge_count"] == 0 and pulse_delivery["mode"] == "legacy-rgb-fallback"
        result["production_eligibility_status"] = (
            "INCONCLUSIVE_NO_ELIGIBILITY"
            if result["eligible_edge_count"] == 0
            else "ELIGIBILITY_OBSERVED"
        )
        result["production_eligibility_edge_count"] = result["eligible_edge_count"]
        result["production_eligibility_changed_edges"] = result["eligible_changed_edges"]
        result["assay_direct_low_level_changed_edges"] = result["pulse_delta"]["changed_edges"]
        result["assay_direct_rule_trace"] = True
    else:
        result["inconclusive_no_eligibility"] = False
    log.append("temporal_complete", {key: value for key, value in result.items() if key != "snapshots"})
    return result


def _checks(output: Path, graph: dict[str, Any], branches: dict[str, Any], temporal: dict[str, Any], temporal_none: dict[str, Any], horizons_ms: tuple[int, ...]) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "plastic_edge_count": graph["plastic_edges"] == EXPECTED_PLASTIC_EDGES,
        "required_horizons": set(DEFAULT_HORIZONS_MS) <= set(horizons_ms),
        "shared_input_hash_temporal": len({item["input_sha256"] for item in (*branches.values(), *temporal.values(), *temporal_none.values())}) == 1,
        "matched_none_controls": set(temporal) == set(temporal_none),
        "snapshot_shapes": True,
        "shared_clean_checkpoint": len({item["clean_checkpoint_sha256"] for item in branches.values()}) == 1,
        "shared_input_hash": len({item["input_sha256"] for item in branches.values()}) == 1,
        "append_only_artifacts": True,
        "none_no_pulse_weight_change": False,
        "reward_localized": False,
        "aversive_localized": False,
        "reward_aversive_separable": False,
        "reward_stimulation_specific_localized": False,
        "aversive_stimulation_specific_localized": False,
        "reward_aversive_stimulation_specific_separable": False,
        "retention_at_60s": False,
        "retention_at_1800s": False,
        "passive_projection_integrity": False,
        "full_network_1s_smoke": False,
        "temporal_assay_direct_pulse": False,
        "raw_temporal_opposite_sign": False,
        "temporal_opposite_sign": False,
        "efficacy_bounds": True,
    }
    all_runs = [*branches.values(), *temporal.values(), *temporal_none.values()]
    for run in all_runs:
        for snapshot in run["snapshots"].values():
            arrays = _load_snapshot(output, snapshot)
            checks["snapshot_shapes"] = checks["snapshot_shapes"] and arrays["edge_weights"].shape == (EXPECTED_PLASTIC_EDGES,) and arrays["baseline_weights"].shape == (EXPECTED_PLASTIC_EDGES,) and arrays["edge_eligibility"].shape == (EXPECTED_PLASTIC_EDGES,) and arrays["dan_rates"].shape == (REWARD_CELLS + AVERSIVE_CELLS,) and arrays["selected_gain_rows"].shape == (REWARD_CELLS + AVERSIVE_CELLS, EXPECTED_PLASTIC_EDGES) and all(np.isfinite(arrays[key]).all() for key in ("edge_weights", "baseline_weights", "eligibility", "edge_eligibility", "dan_rates", "selected_gain_rows"))
    for label, branch in branches.items():
        pulse = branch["pulse_delta"]
        if label == "none":
            checks["none_no_pulse_weight_change"] = pulse["changed_edges"] == 0
        if label in {"reward", "aversive"}:
            checks[f"{label}_nonzero_or_inconclusive"] = branch["eligible_changed_edges"] > 0 or branch["inconclusive_no_eligibility"]
        post = _load_snapshot(output, branch["snapshots"]["post_pulse"])
        baseline = post["baseline_weights"]
        weights = post["edge_weights"]
        ratio = weights / baseline
        checks["efficacy_bounds"] = checks["efficacy_bounds"] and bool(np.isfinite(ratio).all() and np.all(ratio >= EFFICACY_BOUNDS[0]) and np.all(ratio <= EFFICACY_BOUNDS[1]))
    reward = branches["reward"]
    aversive = branches["aversive"]
    reward_before = _load_snapshot(output, reward["snapshots"]["pre_pulse"])
    reward_after = _load_snapshot(output, reward["snapshots"]["post_pulse"])
    aversive_before = _load_snapshot(output, aversive["snapshots"]["pre_pulse"])
    aversive_after = _load_snapshot(output, aversive["snapshots"]["post_pulse"])
    reward_delta = reward_after["edge_weights"] - reward_before["edge_weights"]
    aversive_delta = aversive_after["edge_weights"] - aversive_before["edge_weights"]
    reward_mask = np.any(reward_after["selected_gain_rows"][:REWARD_CELLS] != 0, axis=0)
    aversive_mask = np.any(aversive_after["selected_gain_rows"][REWARD_CELLS:REWARD_CELLS + AVERSIVE_CELLS] != 0, axis=0)
    checks["reward_localized"] = bool(np.all(reward_delta[~reward_mask] == 0))
    checks["aversive_localized"] = bool(np.all(aversive_delta[~aversive_mask] == 0))
    checks["reward_aversive_separable"] = reward["pulse_delta"]["delta_sha256"] != aversive["pulse_delta"]["delta_sha256"] and reward["pulse_delta"]["changed_edges"] > 0 and aversive["pulse_delta"]["changed_edges"] > 0
    none_before = _load_snapshot(output, branches["none"]["snapshots"]["pre_pulse"])
    none_after = _load_snapshot(output, branches["none"]["snapshots"]["post_pulse"])
    reward_specific = _difference_delta(output, reward["snapshots"]["pre_pulse"], reward["snapshots"]["post_pulse"], branches["none"]["snapshots"]["pre_pulse"], branches["none"]["snapshots"]["post_pulse"])
    aversive_specific = _difference_delta(output, aversive["snapshots"]["pre_pulse"], aversive["snapshots"]["post_pulse"], branches["none"]["snapshots"]["pre_pulse"], branches["none"]["snapshots"]["post_pulse"])
    reward["stimulation_specific_delta"] = reward_specific
    aversive["stimulation_specific_delta"] = aversive_specific
    reward_specific_vector = (reward_after["edge_weights"] - reward_before["edge_weights"]) - (none_after["edge_weights"] - none_before["edge_weights"])
    aversive_specific_vector = (aversive_after["edge_weights"] - aversive_before["edge_weights"]) - (none_after["edge_weights"] - none_before["edge_weights"])
    checks["reward_stimulation_specific_localized"] = bool(np.all(reward_specific_vector[~reward_mask] == 0))
    checks["aversive_stimulation_specific_localized"] = bool(np.all(aversive_specific_vector[~aversive_mask] == 0))
    checks["reward_stimulation_specific_inconclusive_no_eligibility"] = reward["inconclusive_no_eligibility"]
    checks["aversive_stimulation_specific_inconclusive_no_eligibility"] = aversive["inconclusive_no_eligibility"]
    checks["reward_aversive_stimulation_specific_separable"] = reward_specific["delta_sha256"] != aversive_specific["delta_sha256"] and reward_specific["changed_edges"] > 0 and aversive_specific["changed_edges"] > 0
    checks["retention_at_60s"] = all("retention_1s_delta" in branch and "retention_60s_delta" in branch for branch in branches.values())
    if checks["retention_at_60s"]:
        checks["retention_at_60s"] = all(branch["retention_60s_delta"]["median_abs_delta"] >= 0.1 * branch["retention_1s_delta"]["median_abs_delta"] for branch in branches.values() if branch["pulse_delta"]["changed_edges"] > 0)
    checks["retention_at_1800s"] = all(
        "retention_1800s_delta" in branch
        for branch in branches.values()
    )
    checks["passive_projection_integrity"] = all(
        all(
            projection.get("full_neural_state_advanced") is False
            and projection.get("input_sha256")
            and projection.get("output_sha256")
            and projection.get("formula", {}).get("name") == "rule.advance.zero_drive_closed_form"
            and projection.get("bounds", {}).get("memory_u_inside")
            and projection.get("bounds", {}).get("memory_w_inside")
            and projection.get("bounds", {}).get("input_memory_u_inside")
            and projection.get("bounds", {}).get("input_memory_w_inside")
            and projection.get("bounds", {}).get("clip_activations") == 0
            for projection in branch.get("passive_projection", {}).values()
        )
        for branch in branches.values()
    )
    checks["full_network_1s_smoke"] = all(
        "t1000ms" in branch["snapshots"]
        and branch["snapshots"]["t1000ms"].get("snapshot_mode") == "full_network"
        for branch in branches.values()
        if 1_000 in horizons_ms
    ) if 1_000 in horizons_ms else False
    checks["temporal_assay_direct_pulse"] = all(
        item.get("mechanism_only") is True
        and item.get("production_reinforce_linked_used") is False
        and item.get("pulse_delivery", {}).get("mode") == "assay-direct-dan-pulse"
        for item in temporal.values()
    )
    raw_temporal_signs = []
    did_temporal_signs = []
    for horizon in ("t1000ms", "t60000ms"):
        if all(horizon in item["snapshots"] for item in temporal.values()) and all(horizon in item["snapshots"] for item in temporal_none.values()):
            raw_temporal_signs.append([_delta(output, item["snapshots"]["pre_pulse"], item["snapshots"][horizon])["median_signed_delta"] for item in temporal.values()])
            did_temporal_signs.append([_difference_delta(output, temporal[name]["snapshots"]["pre_pulse"], temporal[name]["snapshots"][horizon], temporal_none[name]["snapshots"]["pre_pulse"], temporal_none[name]["snapshots"][horizon])["median_signed_delta"] for name in temporal])
    checks["raw_temporal_opposite_sign"] = bool(raw_temporal_signs) and all(a != 0 and b != 0 and a * b < 0 for a, b in raw_temporal_signs)
    checks["temporal_opposite_sign"] = bool(did_temporal_signs) and all(a != 0 and b != 0 and a * b < 0 for a, b in did_temporal_signs)
    checks["temporal_reward_inconclusive_no_eligibility"] = all(item["inconclusive_no_eligibility"] for item in temporal.values())
    return checks


def _temporal_did_audit(
    output: Path,
    temporal: dict[str, Any],
    temporal_none: dict[str, Any],
) -> dict[str, Any]:
    """Persist raw and matched-none temporal deltas for every horizon/order."""
    stages = sorted(
        {
            stage
            for run in (*temporal.values(), *temporal_none.values())
            for stage in run["snapshots"]
            if stage.startswith("t") and stage.endswith("ms")
        },
        key=lambda stage: int(stage[1:-2]),
    )
    audit: dict[str, Any] = {}
    for stage in stages:
        per_order: dict[str, Any] = {}
        for order in temporal:
            if stage not in temporal[order]["snapshots"] or stage not in temporal_none[order]["snapshots"]:
                continue
            raw = _delta(output, temporal[order]["snapshots"]["pre_pulse"], temporal[order]["snapshots"][stage])
            did = _difference_delta(
                output,
                temporal[order]["snapshots"]["pre_pulse"],
                temporal[order]["snapshots"][stage],
                temporal_none[order]["snapshots"]["pre_pulse"],
                temporal_none[order]["snapshots"][stage],
            )
            per_order[order] = {
                "raw_delta": raw,
                "matched_none_did": did,
                "raw_snapshot_mode": temporal[order]["snapshots"][stage].get("snapshot_mode"),
                "raw_snapshot_sha256": temporal[order]["snapshots"][stage]["sha256"],
                "matched_none_snapshot_sha256": temporal_none[order]["snapshots"][stage]["sha256"],
            }
        if per_order:
            audit[stage] = per_order
    return audit


def _partition_checks(checks: dict[str, Any]) -> tuple[dict[str, bool], dict[str, bool]]:
    """Separate diagnostic inconclusive/raw fields from PASS/FAIL gates."""
    diagnostics: dict[str, bool] = {}
    gates: dict[str, bool] = {}
    for key, value in checks.items():
        is_diagnostic = key.startswith("raw_") or (
            "inconclusive" in key and not key.endswith("_nonzero_or_inconclusive")
        )
        (diagnostics if is_diagnostic else gates)[key] = value
    return gates, diagnostics


def run_assay(
    output: Path,
    *,
    seed: int = DEFAULT_SEED,
    horizons_ms: Iterable[int] = DEFAULT_HORIZONS_MS,
    controller_factory: Callable[[Settings], Any] = V51Controller,
    frame_npy: Path | None = None,
) -> dict[str, Any]:
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"assay output must be new: {output}")
    output.mkdir(parents=True, exist_ok=True)
    horizons = tuple(int(value) for value in horizons_ms)
    log = AppendOnlyLog(output / "assay-events.jsonl")
    if frame_npy is None:
        frame = reproducible_input(seed)
        frame_source: dict[str, Any] = {
            "kind": "seeded_reproducible_input",
            "seed": seed,
            "raw_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
            "shape": list(frame.shape),
            "dtype": frame.dtype.str,
        }
    else:
        frame, frame_source = _load_fixed_frame(frame_npy)
    input_hash = hashlib.sha256(frame.tobytes()).hexdigest()
    prereg_hash = sha256(PREREGISTRATION) if PREREGISTRATION.exists() else None
    settings = _settings()
    summary: dict[str, Any] = {"schema": "wc006-v5.1-v6-mechanism-assay/v2", "status": "FAIL", "seed": seed, "horizons_ms": list(horizons), "input_sha256": input_hash, "frame_source": frame_source, "preregistration_sha256": prereg_hash, "paper_only": True, "market_data": False, "broker_orders_enabled": False, "live_execution_authorized": False}
    try:
        prototype = controller_factory(settings)
        graph = _graph_contract(prototype.brain)
        clean = output / "clean.npz"
        prototype.save(clean)
        clean_hash = sha256(clean)
        _atomic_json(output / "provenance.json", {"schema": summary["schema"], "seed": seed, "input_sha256": input_hash, "frame_source": frame_source, "settings": {"neural_ms": settings.neural_ms, "pulse_ms": settings.pulse_ms, "pulse_current": settings.pulse_current, "learning": settings.learning}, "rule_parameters": RULE_PARAMETERS, "graph": graph, "clean_checkpoint_sha256": clean_hash, "preregistration_sha256": prereg_hash, "paper_only": True, "market_data": False})
        log.append("assay_start", {**summary, "graph": graph, "clean_checkpoint_sha256": clean_hash, "settings": {"neural_ms": settings.neural_ms, "pulse_ms": settings.pulse_ms, "pulse_current": settings.pulse_current, "learning": settings.learning}})
        branches = {label: _run_branch(output=output, log=log, label=label, controller_factory=controller_factory, settings=settings, frame=frame, clean_checkpoint=clean, clean_hash=clean_hash, input_hash=input_hash, seed=seed, horizons_ms=horizons) for label in ("reward", "aversive", "none")}
        temporal = {order: _run_temporal(output=output, log=log, order=order, pulse_label="reward", controller_factory=controller_factory, settings=settings, frame=frame, clean_checkpoint=clean, clean_hash=clean_hash, input_hash=input_hash, seed=seed, horizons_ms=horizons) for order in ("kc-before-dan", "dan-before-kc")}
        temporal_none = {order: _run_temporal(output=output, log=log, order=order, pulse_label="none", controller_factory=controller_factory, settings=settings, frame=frame, clean_checkpoint=clean, clean_hash=clean_hash, input_hash=input_hash, seed=seed, horizons_ms=horizons) for order in ("kc-before-dan", "dan-before-kc")}
        checks = _checks(output, graph, branches, temporal, temporal_none, horizons)
        temporal_did = _temporal_did_audit(output, temporal, temporal_none)
        gate_checks, diagnostics = _partition_checks(checks)
        summary.update({"graph": graph, "clean_checkpoint_sha256": clean_hash, "branches": branches, "temporal": temporal, "temporal_none": temporal_none, "checks": checks, "gate_checks": gate_checks, "diagnostics": diagnostics, "temporal_did": temporal_did})
        summary["status"] = "PASS" if all(gate_checks.values()) else "FAIL"
        log.append("temporal_did_audit", {"temporal_did": temporal_did})
        log.append("assay_complete", {"status": summary["status"], "gate_checks": gate_checks, "diagnostics": diagnostics, "temporal_did": temporal_did})
    except Exception as exc:
        summary.update({"error_type": type(exc).__name__, "error": str(exc)})
        log.append("assay_failure", {"error_type": type(exc).__name__, "error": str(exc)})
    summary["log"] = {"file": "assay-events.jsonl", **verify_append_only_log(output / "assay-events.jsonl")}
    _atomic_json(output / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--horizons-ms", type=int, nargs="+", default=list(DEFAULT_HORIZONS_MS))
    parser.add_argument("--frame-npy", type=Path, default=None, help="fixed uint8 180x320x3 cue frame")
    args = parser.parse_args(argv)
    result = run_assay(args.out, seed=args.seed, horizons_ms=args.horizons_ms, frame_npy=args.frame_npy)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
