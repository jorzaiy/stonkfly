"""Bounded post-failure diagnostic for the production linked pulse.

This is deliberately not a market replay, training run, authorization, or
formal mechanism assay.  It starts from the fixed sequential coverage
endpoint and calls only ``V51Controller.reinforce_linked`` once per branch.
Retention after the pulse is an analytic zero-drive projection; no new RGB or
low-level neural integration is performed by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from experiments.a_share.wc006_v5_1_v6_mechanism_assay import (
    AVERSIVE_CELLS,
    EFFICACY_BOUNDS,
    EXPECTED_PLASTIC_EDGES,
    REWARD_CELLS,
    _graph_contract,
    _label_eligibility_mask,
    _load_snapshot,
    _project_passive_snapshot,
    _snapshot,
    sha256,
    verify_append_only_log,
)
from experiments.a_share.wc006_v5_1_v6_mechanism_assay import AppendOnlyLog
from stonkfly.config import Settings
from stonkfly.neural.v51_controller import V51Controller


ROOT = Path(__file__).resolve().parents[2]
COVERAGE_DIR = ROOT / "runs/wc006_v5_1_eligibility_coverage/annual-2023-sensory-only-v1"
ENDPOINT = COVERAGE_DIR / "brain-end.npz"
EXPECTED_ENDPOINT_SHA256 = "b3bcc48d6b90190c5f6e2fd4fc00b1b613038a87e652916c1eba40428c64d269"
PROTOCOL = ROOT / "docs/experiments/wc006_v5_1_post_failure_linked_pulse_diagnostic.md"
V51_CONTROLLER_SOURCE = ROOT / "stonkfly/neural/v51_controller.py"
DEFAULT_HORIZONS_MS = (1_000, 60_000, 1_800_000)
PULSE_MS = 200.0
PULSE_CURRENT = 20.0
EXPECTED_REWARD_ELIGIBILITY = 1_578
EXPECTED_AVERSIVE_ELIGIBILITY = 2_044


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _delta_stats(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(right) - np.asarray(left)
    changed = np.abs(delta) > 0
    return {
        "changed_edges": int(np.count_nonzero(changed)),
        "median_signed_delta": float(np.median(delta[changed])) if np.any(changed) else 0.0,
        "median_abs_delta": float(np.median(np.abs(delta[changed]))) if np.any(changed) else 0.0,
        "delta_sha256": _digest(delta),
    }


def _same_snapshot(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> bool:
    keys = ("edge_weights", "eligibility", "edge_eligibility", "dan_rates", "dan_counts", "memory_u", "memory_w")
    return all(np.array_equal(left[key], right[key]) for key in keys)


def _run_branch(
    *,
    output: Path,
    label: str,
    endpoint: Path,
    endpoint_sha256: str,
    controller_factory: Callable[[Settings], Any],
    settings: Settings,
    log: AppendOnlyLog,
    horizons_ms: tuple[int, ...],
) -> dict[str, Any]:
    clean = output / "branches" / f"{label}-clean.npz"
    clean.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(endpoint, clean)
    clean_hash = sha256(clean)
    if clean_hash != endpoint_sha256:
        raise ValueError(f"branch clean checkpoint mismatch: {label}")
    controller = controller_factory(settings)
    controller.restore(clean)
    pre = _snapshot(controller.brain, output, label, "pre-pulse")
    pre_arrays = _load_snapshot(output, pre)
    if label == "none":
        eligible_mask = np.zeros(len(pre_arrays["edge_weights"]), dtype=bool)
    else:
        eligible_mask = _label_eligibility_mask(pre_arrays, label)
    linkage_id = f"wc006-v5.1-post-failure-diagnostic-v1:{label}"
    log.append(
        "branch_pre_pulse",
        {
            "branch": label,
            "clean_checkpoint_sha256": clean_hash,
            "pre_snapshot": pre,
            "linkage_id": linkage_id,
            "eligible_edge_count": int(np.count_nonzero(eligible_mask)),
        },
    )
    # This is the only pulse/state-advance call in the real diagnostic.
    delivery = controller.reinforce_linked(label, linkage_id=linkage_id)
    post = _snapshot(controller.brain, output, label, "post-pulse")
    post_arrays = _load_snapshot(output, post)
    post_checkpoint = output / "branches" / f"{label}-post-pulse.npz"
    controller.save(post_checkpoint)
    post_checkpoint_hash = sha256(post_checkpoint)
    weight_delta = post_arrays["edge_weights"] - pre_arrays["edge_weights"]
    changed = np.abs(weight_delta) > 0
    outside = changed & ~eligible_mask
    gain_mask = np.any(
        post_arrays["selected_gain_rows"][:REWARD_CELLS]
        if label == "reward"
        else post_arrays["selected_gain_rows"][REWARD_CELLS:REWARD_CELLS + AVERSIVE_CELLS]
        if label == "aversive"
        else np.zeros((1, EXPECTED_PLASTIC_EDGES), dtype=bool),
        axis=0,
    )
    projections: dict[str, Any] = {}
    for horizon in horizons_ms:
        stage = f"t{int(horizon)}ms"
        info, projection = _project_passive_snapshot(
            output,
            label,
            stage,
            post,
            duration_ms=int(horizon),
            origin_elapsed_ms=0.0,
        )
        projected_arrays = _load_snapshot(output, info)
        projections[stage] = {
            "snapshot": info,
            "projection": projection,
            "post_to_projection": _delta_stats(post_arrays["edge_weights"], projected_arrays["edge_weights"]),
            "pre_to_projection": _delta_stats(pre_arrays["edge_weights"], projected_arrays["edge_weights"]),
        }
    log.append(
        "branch_post_pulse",
        {
            "branch": label,
            "delivery": delivery,
            "post_snapshot": post,
            "post_checkpoint_sha256": post_checkpoint_hash,
            "changed_edges": int(np.count_nonzero(changed)),
            "eligible_changed_edges": int(np.count_nonzero(changed & eligible_mask)),
            "changed_outside_selected_gain": int(np.count_nonzero(changed & ~gain_mask)),
            "none_snapshot_equal": _same_snapshot(pre_arrays, post_arrays) if label == "none" else None,
        },
    )
    ratio = post_arrays["edge_weights"] / post_arrays["baseline_weights"]
    projection_integrity = all(
        item["projection"].get("full_neural_state_advanced") is False
        and item["projection"].get("formula", {}).get("name") == "rule.advance.zero_drive_closed_form"
        and item["projection"].get("source_snapshot_sha256") == post["sha256"]
        and item["projection"].get("snapshot", {}).get("sha256") == item["snapshot"]["sha256"]
        and item["projection"].get("bounds", {}).get("memory_u_inside") is True
        and item["projection"].get("bounds", {}).get("memory_w_inside") is True
        for item in projections.values()
    )
    return {
        "branch": label,
        "clean_checkpoint_sha256": clean_hash,
        "pre_snapshot": pre,
        "post_snapshot": post,
        "post_checkpoint_sha256": post_checkpoint_hash,
        "linkage_id": linkage_id,
        "delivery": delivery,
        "eligible_edge_count": int(np.count_nonzero(eligible_mask)),
        "pulse_delta": _delta_stats(pre_arrays["edge_weights"], post_arrays["edge_weights"]),
        "eligible_changed_edges": int(np.count_nonzero(changed & eligible_mask)),
        "changed_outside_selected_gain": int(np.count_nonzero(changed & ~gain_mask)),
        "changed_outside_eligibility": int(np.count_nonzero(outside)),
        "none_snapshot_equal": _same_snapshot(pre_arrays, post_arrays) if label == "none" else None,
        "clock_advanced_ms": float(post["sim_ms"] - pre["sim_ms"]),
        "efficacy_ratio": {"min": float(np.min(ratio)), "max": float(np.max(ratio)), "within_bounds": bool(np.all(np.isfinite(ratio)) and np.all(ratio >= EFFICACY_BOUNDS[0]) and np.all(ratio <= EFFICACY_BOUNDS[1]))},
        "passive_projection_integrity": projection_integrity,
        "passive_projection": projections,
    }


def run_diagnostic(
    output: Path,
    *,
    endpoint: Path = ENDPOINT,
    endpoint_sha256: str = EXPECTED_ENDPOINT_SHA256,
    controller_factory: Callable[[Settings], Any] = V51Controller,
    horizons_ms: Iterable[int] = DEFAULT_HORIZONS_MS,
) -> dict[str, Any]:
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"diagnostic output must be new: {output}")
    output.mkdir(parents=True, exist_ok=True)
    endpoint = Path(endpoint).resolve()
    if sha256(endpoint) != endpoint_sha256:
        raise ValueError("coverage endpoint hash mismatch")
    horizons = tuple(int(item) for item in horizons_ms)
    if horizons != DEFAULT_HORIZONS_MS:
        raise ValueError("post-failure diagnostic horizons are fixed at 1s/60s/1800s")
    log = AppendOnlyLog(output / "diagnostic-events.jsonl")
    settings = Settings(neural_ms=500.0, pulse_ms=PULSE_MS, pulse_current=PULSE_CURRENT, learning=True)
    prototype = controller_factory(settings)
    graph = _graph_contract(prototype.brain)
    if graph["plastic_edges"] != EXPECTED_PLASTIC_EDGES:
        raise ValueError("full retained plastic edge count is not 7,835")
    source_copy = output / "coverage-brain-end.npz"
    shutil.copyfile(endpoint, source_copy)
    source_copy_hash = sha256(source_copy)
    provenance = {
        "protocol_sha256": sha256(PROTOCOL),
        "diagnostic_code_sha256": sha256(Path(__file__).resolve()),
        "v51_controller_sha256": sha256(V51_CONTROLLER_SOURCE),
        "coverage_endpoint_sha256": endpoint_sha256,
        "coverage_endpoint_copy_sha256": source_copy_hash,
    }
    summary: dict[str, Any] = {
        "schema": "wc006-v5.1-post-failure-linked-pulse-diagnostic/v1",
        "status": "COMPLETE",
        "interpretation": "DESCRIPTIVE_ONLY_NON_AUTHORIZING_NO_FORMAL_PASS",
        "paper_only": True,
        "market_execution": False,
        "annual_training": False,
        "formal_authorization_created": False,
        "endpoint": {"file": str(endpoint), "sha256": endpoint_sha256, "date": "2023-12-29", "reward_eligible_edge_count": 1578, "aversive_eligible_edge_count": 2044},
        "settings": {"neural_ms": settings.neural_ms, "pulse_ms": settings.pulse_ms, "pulse_current": settings.pulse_current, "learning": settings.learning},
        "horizons_ms": list(horizons),
        "graph": graph,
        "provenance": provenance,
    }
    log.append("diagnostic_start", {**summary, "branches": ["reward", "aversive", "none"], "reinforcement_calls": "exactly one production reinforce_linked call per branch; no RGB or low-level direct call"})
    branches = {label: _run_branch(output=output, label=label, endpoint=source_copy, endpoint_sha256=endpoint_sha256, controller_factory=controller_factory, settings=settings, log=log, horizons_ms=horizons) for label in ("reward", "aversive", "none")}
    reward = branches["reward"]
    aversive = branches["aversive"]
    none = branches["none"]
    retention_branches = [branch for branch in (reward, aversive) if branch["pulse_delta"]["changed_edges"] > 0]
    checks = {
        "shared_endpoint_hash": len({item["clean_checkpoint_sha256"] for item in branches.values()}) == 1 and next(iter({item["clean_checkpoint_sha256"] for item in branches.values()})) == endpoint_sha256,
        "reward_eligible_count_expected": reward["eligible_edge_count"] == EXPECTED_REWARD_ELIGIBILITY,
        "aversive_eligible_count_expected": aversive["eligible_edge_count"] == EXPECTED_AVERSIVE_ELIGIBILITY,
        "reward_delivery_applied": reward["delivery"]["status"] == "APPLIED" and reward["pulse_delta"]["changed_edges"] > 0,
        "aversive_delivery_applied": aversive["delivery"]["status"] == "APPLIED" and aversive["pulse_delta"]["changed_edges"] > 0,
        "none_no_op": none["delivery"]["status"] == "NO_OP_NONE" and none["pulse_delta"]["changed_edges"] == 0 and none["none_snapshot_equal"] is True and none["clock_advanced_ms"] == 0.0,
        "reward_localized": reward["changed_outside_selected_gain"] == 0,
        "reward_localized_to_eligibility": reward["changed_outside_eligibility"] == 0,
        "aversive_localized_to_eligibility": aversive["changed_outside_eligibility"] == 0,
        "aversive_localized": aversive["changed_outside_selected_gain"] == 0,
        "reward_aversive_separable": reward["pulse_delta"]["changed_edges"] > 0 and aversive["pulse_delta"]["changed_edges"] > 0 and reward["pulse_delta"]["delta_sha256"] != aversive["pulse_delta"]["delta_sha256"],
        "efficacy_bounds": all(branch["efficacy_ratio"]["within_bounds"] for branch in branches.values()),
        "retention_60s_threshold": len(retention_branches) == 2 and all(branch["passive_projection"]["t60000ms"]["pre_to_projection"]["median_abs_delta"] >= 0.1 * branch["passive_projection"]["t1000ms"]["pre_to_projection"]["median_abs_delta"] for branch in retention_branches),
        "passive_projection_only": all(branch["passive_projection_integrity"] for branch in branches.values()),
    }
    eligible_branches = (reward, aversive)
    if any(branch["eligible_edge_count"] == 0 for branch in eligible_branches):
        mechanism_result = "INCONCLUSIVE_NO_ELIGIBILITY"
    elif any(branch["delivery"]["status"] != "APPLIED" for branch in eligible_branches):
        mechanism_result = "FAIL"
    else:
        mechanism_result = "PASS" if all(checks.values()) else "FAIL"
    summary.update({"branches": branches, "checks": checks, "mechanism_result": mechanism_result, "fixed_thresholds": {"efficacy_bounds": list(EFFICACY_BOUNDS), "retention_60s_fraction_of_1s": 0.1, "required_horizons_ms": list(DEFAULT_HORIZONS_MS)}})
    log.append("diagnostic_complete", {"status": summary["status"], "mechanism_result": mechanism_result, "checks": checks, "branches": {name: {"delivery": branch["delivery"], "pulse_delta": branch["pulse_delta"]} for name, branch in branches.items()}})
    summary["log"] = {"file": "diagnostic-events.jsonl", **verify_append_only_log(output / "diagnostic-events.jsonl")}
    temporary = output / "summary.json.partial"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "summary.json")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_diagnostic(args.out)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
