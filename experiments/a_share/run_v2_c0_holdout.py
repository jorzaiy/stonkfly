#!/usr/bin/env python3
"""V2-C0-B independent synthetic holdout process.

The process reads exactly one immutable ``c0_selection.json``.  It has no
gain search, candidate ranking, or calibration-mask constructor.  A command
line gain override is rejected before any model is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

from stonkfly.neural.c0_common import (
    D_HOLD,
    DT_HOLD,
    GAINS,
    SPARSE_HIGH,
    SPARSE_LOW,
    classify_kc,
    formal_verdict,
    jaccard,
    json_sha256,
    sha256_file,
    source_id_sha256,
    symmetric_relative_difference,
)
from stonkfly.neural.v2_brain import classify_nodes
from stonkfly.neural.v2_c0 import C0HybridBrain


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/whole_brain_v2/C0/20261010T-graded-release-calibration-v2"
GRAPH = ROOT / "data/graph.npz"
NEURONS = ROOT / "data/normalized/neurons.feather"
EDGES = ROOT / "data/normalized/edges.arrow"
Q1_FORMAL = ROOT / "runs/whole_brain_v2/Q1/20261009T-hybrid-kernel-qualification-v7/q1_formal.json"
Q1_RAW = ROOT / "runs/whole_brain_v2/Q1/20261009T-hybrid-kernel-qualification-v7/q1_raw.npz"
Q1_MANIFEST = ROOT / "runs/whole_brain_v2/Q1/20261009T-hybrid-kernel-qualification-v7/q1_manifest.json"
C0_WRAPPER = ROOT / "stonkfly/neural/v2_c0.py"
TESTS = ROOT / "tests/test_v2_c0_protocol.py"


def _sha(path: Path) -> str:
    return sha256_file(path)


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _hash_snapshot(build_info: dict) -> dict:
    paths = {
        "v2_source": ROOT / build_info["source"],
        "v2_binary": ROOT / build_info["binary"],
        "c0_wrapper": C0_WRAPPER,
        "c0_runner": Path(__file__).resolve(),
        "c0_tests": TESTS,
    }
    return {name: _sha(path) for name, path in paths.items()}


def _state_sha(brain: C0HybridBrain) -> str:
    h = hashlib.sha256()
    for value in (brain.v, brain.g_event, brain.g_graded, brain.refractory, brain.drive,
                  brain.previous_drive, brain.spike_queue, brain.spike_queue_count,
                  brain.graded_queue, brain.graded_drive, brain.counts, brain.active,
                  brain.active_flag, brain.nactive, brain.last, brain.adaptation, brain.modulation):
        h.update(np.ascontiguousarray(value).tobytes())
    h.update(str(int(brain.clock[0])).encode())
    return h.hexdigest()


def load_selection(path: Path) -> tuple[dict, str]:
    path = Path(path)
    if not path.exists():
        raise RuntimeError("missing immutable c0_selection.json")
    selection_sha = _sha(path)
    selection = json.loads(path.read_text())
    if selection.get("schema") != "v2-c0-selection-v1":
        raise RuntimeError("selection schema mismatch")
    if selection.get("calibration_verdict") != "C0_SELECTION_READY":
        raise RuntimeError("selection is not a legal calibration selection")
    gamma = float(selection.get("selected_gamma"))
    if gamma not in GAINS[1:-1]:
        raise RuntimeError("selection gain is grid boundary or not frozen")
    return selection, selection_sha


def _pattern(ids: np.ndarray, fraction: float, salt: str) -> np.ndarray:
    ranked = sorted(((hashlib.sha256(f"{salt}:{int(x)}".encode()).digest(), int(x)) for x in ids), key=lambda z: z[0])
    return np.asarray([x for _, x in ranked[: int(np.floor(fraction * len(ids)))]], dtype=np.int64)


def holdout_masks(ids: np.ndarray, layer: np.ndarray) -> dict:
    photo = ids[layer == 1]
    return {"HOLD_A25": _pattern(photo, .25, "HOLD_A25"), "HOLD_B50": _pattern(photo, .50, "HOLD_B50"),
            "HOLD_C75": _pattern(photo, .75, "HOLD_C75"), "HOLD_D50": _pattern(photo, .50, "HOLD_D50")}


def _load_data():
    graph = np.load(GRAPH, allow_pickle=False)
    table = feather.read_table(NEURONS).to_pydict()
    layer, names = classify_nodes(table["cell_type"], table["neurotransmitter"])
    ids = np.asarray(graph["ids"], dtype=np.int64)
    if len(ids) != len(layer) or int(np.sum(layer == 1)) != 4706:
        raise RuntimeError("holdout graph/classification mismatch")
    cell_type = np.asarray(table["cell_type"], dtype=object)
    return graph, ids, layer, cell_type, names


def initial_condition(name: str, rest: np.ndarray, graded: np.ndarray, base_active: np.ndarray):
    voltage = np.asarray(rest, dtype=np.float32).copy()
    non_graded = np.flatnonzero(~graded)
    if name == "CLEAN":
        return voltage, np.asarray(base_active, dtype=np.int64), {"name": name, "seed": None, "perturbed_count": 0, "perturbation_sha256": _array_sha(np.empty(0, np.float32))}
    seeds = {"P1": 202610091, "P2": 202610092, "P3": 202610093}
    if name not in seeds:
        raise ValueError("unknown initial condition")
    rng = np.random.Generator(np.random.PCG64(seeds[name]))
    perturb = rng.uniform(-.05, .05, size=len(non_graded)).astype(np.float32)
    voltage[non_graded] += perturb
    active = np.unique(np.concatenate([base_active, non_graded]))
    return voltage, active, {"name": name, "seed": seeds[name], "perturbed_count": int(len(non_graded)), "perturbed_node_sha256": _array_sha(non_graded), "perturbation_sha256": _array_sha(perturb)}


def _trial(gamma, dt, stimulus, selected, initial_name, repeat, graph, ids, layer, cell_type, base_active):
    graded = np.isin(layer, [1, 2, 3]); kc = layer == 4
    rest = np.full(len(ids), -52., np.float32); rest[kc] = -60.
    brain = C0HybridBrain(GRAPH, gamma=gamma, dt=dt, graded_mask=graded.astype(np.uint8), rest=rest)
    mb07_mask = cell_type == "MBON07"; mb11_mask = cell_type == "MBON11"
    voltage, active, init_meta = initial_condition(initial_name, rest, graded, base_active)
    brain.reset_with_initial_voltage(voltage, active); brain.kc_mask[:] = kc.astype(np.uint8); brain.modulation_mask.fill(0)
    zero = np.zeros(len(ids), np.float32); drive = zero.copy()
    current = float(stimulus.split("_")[1]) if stimulus.startswith("UNIFORM_") else 4.0
    drive[np.isin(ids, selected)] = current
    brain.run(50., zero); brain.run(100., drive); counts = brain.run(50., drive)
    active_kc = np.flatnonzero((counts > 0) & kc).astype(np.int64)
    mb07 = np.flatnonzero((counts > 0) & mb07_mask).astype(np.int64); mb11 = np.flatnonzero((counts > 0) & mb11_mask).astype(np.int64)
    brain.run(200., zero); post = brain.run(100., zero); post_kc = np.flatnonzero((post > 0) & kc).astype(np.int64)
    finite = bool(np.isfinite(brain.v).all() and np.isfinite(brain.g_event).all() and np.isfinite(brain.g_graded).all() and np.isfinite(brain.graded_queue).all())
    queue = bool(np.all(brain.spike_queue_count >= 0) and np.all(brain.spike_queue_count <= len(ids)) and np.isfinite(brain.graded_drive).all() and int(brain.overflow[0]) == 0)
    return {"dt_ms": dt, "stimulus": stimulus, "initial_condition": initial_name, "repeat": repeat,
            "measurement_active_kc": int(len(active_kc)), "post_active_kc": int(len(post_kc)), "measurement_active_kc_ids": active_kc.tolist(), "post_active_kc_ids": post_kc.tolist(),
            "mb07_active": int(len(mb07)), "mb11_active": int(len(mb11)), "mb07_spikes": int(counts[mb07_mask].sum()), "mb11_spikes": int(counts[mb11_mask].sum()),
            "total_spikes": int(brain.total_spikes), "release_integral": float(brain.release_integral[0]), "signed_delivery_integral": float(brain.signed_delivery[0]), "absolute_delivery_integral": float(brain.absolute_delivery[0]),
            "finite": finite, "bounds": bool(np.max(np.abs(brain.v)) < 1e6 and np.max(np.abs(brain.g_event)) < 1e9 and np.max(np.abs(brain.g_graded)) < 1e9), "queue_integrity": queue, "overflow": int(brain.overflow[0]), "clock": int(brain.clock[0]),
            "state_sha256": _state_sha(brain), "weights_sha256": _array_sha(brain.weight), "initial": init_meta, "kc_bitset": np.packbits(np.isin(np.arange(len(ids)), active_kc).astype(np.uint8)).tolist()}


def _ramp(gamma, dt, ids, layer, cell_type, base_active):
    graded = np.isin(layer, [1, 2, 3]); kc = layer == 4; rest = np.full(len(ids), -52., np.float32); rest[kc] = -60.
    brain = C0HybridBrain(GRAPH, gamma=gamma, dt=dt, graded_mask=graded.astype(np.uint8), rest=rest); brain.reset_with_initial_voltage(rest, base_active); brain.kc_mask[:] = kc.astype(np.uint8); brain.modulation_mask.fill(0)
    photo = ids[layer == 1]; zero = np.zeros(len(ids), np.float32); levels = list(D_HOLD) + list(reversed(D_HOLD))
    rows = []
    for index, level in enumerate(levels):
        drive = zero.copy(); drive[np.isin(ids, photo)] = float(level); brain.run(50., drive); counts = brain.run(50., drive); active = np.flatnonzero((counts > 0) & kc).astype(np.int64)
        rows.append({"index": index, "level": level, "direction": "up" if index < len(D_HOLD) else "down", "active_kc": int(len(active)), "active_kc_ids": active.tolist()})
    brain.run(200., zero); post = brain.run(100., zero)
    return {"dt_ms": dt, "levels": rows, "post_active_kc": int(np.sum((post > 0) & kc)), "state_sha256": _state_sha(brain), "finite": bool(np.isfinite(brain.v).all()), "queue_integrity": bool(np.all(brain.spike_queue_count >= 0) and int(brain.overflow[0]) == 0), "clock": int(brain.clock[0])}


def evaluate_holdout(rows, ramps):
    expected = 2 * 11 * 4 * 2
    keys = {(r["dt_ms"], r["stimulus"], r["initial_condition"], r["repeat"]) for r in rows}
    completeness = {"trial_count": len(rows), "expected_trial_count": expected, "ramp_count": len(ramps), "pass": len(rows) == expected and len(ramps) == 2 and len(keys) == expected}
    numerical = {"finite": all(r["finite"] for r in rows) and all(r["finite"] for r in ramps), "bounds": all(r["bounds"] for r in rows) if rows else False, "queue_integrity": all(r["queue_integrity"] for r in rows) and all(r["queue_integrity"] for r in ramps), "pass": False}
    numerical["pass"] = bool(numerical["finite"] and numerical["bounds"] and numerical["queue_integrity"])
    by = {(r["dt_ms"], r["stimulus"], r["initial_condition"], r["repeat"]): r for r in rows}
    deterministic = len({r["state_sha256"] for r in rows}) == expected // 2 * 1 if False else all(by[k]["state_sha256"] == by[(k[0], k[1], k[2], 2 if k[3] == 1 else 1)]["state_sha256"] for k in by)
    clean = {s: by[(.05, s, "CLEAN", 1)] for s in [f"UNIFORM_{x}" for x in D_HOLD] + ["HOLD_A25", "HOLD_B50", "HOLD_C75", "HOLD_D50"]}
    fine_rows = []
    for s, row in clean.items():
        other = by[(.025, s, "CLEAN", 1)]
        fine_rows.append({"stimulus": s, "active_kc_relerr": symmetric_relative_difference(row["measurement_active_kc"], other["measurement_active_kc"]), "kc_jaccard": jaccard(row["measurement_active_kc_ids"], other["measurement_active_kc_ids"]), "mb07_abs_diff": abs(row["mb07_active"] - other["mb07_active"]), "mb11_abs_diff": abs(row["mb11_active"] - other["mb11_active"])})
    fine = {"rows": fine_rows, "passing": sum(x["active_kc_relerr"] <= .10 and x["kc_jaccard"] >= .70 and x["mb07_abs_diff"] <= 1 and x["mb11_abs_diff"] <= 1 for x in fine_rows), "pass": False}
    fine["pass"] = bool(fine["passing"] >= 10)
    nonzero = [r for r in rows if not r["stimulus"].startswith("UNIFORM_0")]
    sparse = sum(SPARSE_LOW <= r["measurement_active_kc"] <= SPARSE_HIGH for r in nonzero) / max(1, len(nonzero))
    low = sum(r["measurement_active_kc"] <= 10 for r in nonzero) / max(1, len(nonzero)); dense = sum(r["measurement_active_kc"] > SPARSE_HIGH for r in nonzero) / max(1, len(nonzero))
    post_ok = sum(r["post_active_kc"] <= 40 for r in nonzero) / max(1, len(nonzero))
    combo = defaultdict(list)
    for r in nonzero: combo[(r["stimulus"], r["initial_condition"], r["repeat"])].append(r)
    no_dense_both_dt = all(not all(x["post_active_kc"] > SPARSE_HIGH for x in vals) for vals in combo.values())
    distribution = {"sparse_fraction": sparse, "fixed_low_fraction": low, "dense_fraction": dense, "post_le_40_fraction": post_ok, "no_stimulus_initial_repeat_dense_both_dt": no_dense_both_dt, "pass": bool(sparse >= .50 and low <= .10 and dense <= .10 and post_ok >= .90 and no_dense_both_dt)}
    clean_sparse = [r for r in rows if r["initial_condition"] == "CLEAN" and r["stimulus"] != "UNIFORM_0" and SPARSE_LOW <= r["measurement_active_kc"] <= SPARSE_HIGH]
    robust_rows = []
    for r in rows:
        if r["initial_condition"] == "CLEAN" or r["stimulus"] == "UNIFORM_0": continue
        c = by[(r["dt_ms"], r["stimulus"], "CLEAN", r["repeat"])]
        robust_rows.append({"class_equal": classify_kc(r["measurement_active_kc"]) == classify_kc(c["measurement_active_kc"]), "jaccard": jaccard(r["measurement_active_kc_ids"], c["measurement_active_kc_ids"]), "sparse_to_bad": SPARSE_LOW <= c["measurement_active_kc"] <= SPARSE_HIGH and classify_kc(r["measurement_active_kc"]) in ("dense", "fixed-low-like")})
    robustness = {"class_agreement": sum(x["class_equal"] for x in robust_rows) / max(1, len(robust_rows)), "jaccard_median": float(np.median([x["jaccard"] for x in robust_rows])) if robust_rows else 0., "sparse_to_bad_fraction": sum(x["sparse_to_bad"] for x in robust_rows) / max(1, len(robust_rows)), "pass": False}
    robustness["pass"] = bool(robustness["class_agreement"] >= .90 and robustness["jaccard_median"] >= .50 and robustness["sparse_to_bad_fraction"] <= .20)
    b_rows = [r for r in rows if r["stimulus"] == "HOLD_B50"]; d_rows = [r for r in rows if r["stimulus"] == "HOLD_D50"]
    pairs = []
    for b in b_rows:
        d = by[(b["dt_ms"], "HOLD_D50", b["initial_condition"], b["repeat"])]
        pairs.append({"jaccard": jaccard(b["measurement_active_kc_ids"], d["measurement_active_kc_ids"]), "nonempty": bool(b["measurement_active_kc"] and d["measurement_active_kc"]), "vector_equal": [b["mb07_active"], b["mb11_active"], b["mb07_spikes"] + b["mb11_spikes"]] == [d["mb07_active"], d["mb11_active"], d["mb07_spikes"] + d["mb11_spikes"]], "b_sparse": SPARSE_LOW <= b["measurement_active_kc"] <= SPARSE_HIGH, "d_sparse": SPARSE_LOW <= d["measurement_active_kc"] <= SPARSE_HIGH})
    identity = {"pairs": pairs, "median_jaccard": float(np.median([x["jaccard"] for x in pairs])) if pairs else 0., "jaccard_le_0_90_fraction": sum(x["jaccard"] <= .90 for x in pairs) / max(1, len(pairs)), "nonempty": all(x["nonempty"] for x in pairs), "both_sparse_fraction": sum(x["b_sparse"] and x["d_sparse"] for x in pairs) / max(1, len(pairs)), "mbon_both_sparse_fraction": sum(x["b_sparse"] and x["d_sparse"] and not x["vector_equal"] for x in pairs) / max(1, len(pairs)), "pass": False}
    identity["pass"] = bool(identity["nonempty"] and identity["median_jaccard"] <= .80 and identity["jaccard_le_0_90_fraction"] >= .75 and identity["both_sparse_fraction"] >= .75 and identity["mbon_both_sparse_fraction"] >= .75)
    hist_rows = []
    for r in ramps:
        levels = r["levels"]; symmetry = []; overlaps = []
        for i in range(1, 7):
            up, down = levels[i], levels[14 - i]; symmetry.append(symmetric_relative_difference(up["active_kc"], down["active_kc"])); overlaps.append(jaccard(up["active_kc_ids"], down["active_kc_ids"]))
        hist_rows.append({"dt_ms": r["dt_ms"], "symmetry_pass": sum(x <= .25 for x in symmetry), "jaccard_pass": sum(x >= .50 for x in overlaps), "post_active_kc": r["post_active_kc"], "pass": sum(x <= .25 for x in symmetry) >= 5 and sum(x >= .50 for x in overlaps) >= 4 and r["post_active_kc"] <= 40})
    history = {"rows": hist_rows, "pass": bool(all(x["pass"] for x in hist_rows) and len(hist_rows) == 2 and hist_rows[0]["pass"] == hist_rows[1]["pass"])}
    gates = {"completeness": completeness, "numerical": numerical, "repeat_determinism": {"pass": bool(deterministic)}, "fine_dt": fine, "sparse_distribution_exit": distribution, "initial_condition_robustness": robustness, "identity_mbon": identity, "history": history}
    gates["pass"] = bool(all(v.get("pass") for v in gates.values()))
    return gates


def run(out=OUT, selection_path=None):
    out = Path(out); holdir = out / "holdout"
    if holdir.exists() and any(holdir.iterdir()):
        raise RuntimeError(f"refusing to overwrite existing C0 holdout output: {holdir}")
    holdir.mkdir(parents=True, exist_ok=True)
    selection_path = Path(selection_path) if selection_path else out / "c0_selection.json"
    selection, selection_sha = load_selection(selection_path); gamma = float(selection["selected_gamma"])
    from stonkfly.neural.v2_brain import build
    build_info = build(); hashes_start = _hash_snapshot(build_info)
    graph, ids, layer, cell_type, names = _load_data(); masks = holdout_masks(ids, layer); graded = np.isin(layer, [1, 2, 3]); base_parts = [graph[k] for k in ("retina", "lamina", "sugar") if k in graph.files]; base_active = np.unique(np.concatenate(base_parts)) if base_parts else np.empty(0, np.int64)
    stimuli = {f"UNIFORM_{d}": ids[layer == 1] for d in D_HOLD}; stimuli.update(masks)
    rows = []; ramps = []; start = time.time(); bitsets = []
    for dt in DT_HOLD:
        for repeat in (1, 2):
            for initial in ("CLEAN", "P1", "P2", "P3"):
                for stimulus, selected in stimuli.items():
                    row = _trial(gamma, dt, stimulus, selected, initial, repeat, graph, ids, layer, cell_type, base_active); rows.append(row); bitsets.append(np.asarray(row.pop("kc_bitset"), dtype=np.uint8))
        ramps.append(_ramp(gamma, dt, ids, layer, cell_type, base_active))
    gates = evaluate_holdout(rows, ramps)
    mask_records = {name: {"count": int(len(sel)), "source_id_sha256": source_id_sha256(sel), "source_ids": [int(x) for x in sel]} for name, sel in stimuli.items()}
    perturbations = {name: initial_condition(name, np.full(len(ids), -52., np.float32), graded, base_active)[2] for name in ("CLEAN", "P1", "P2", "P3")}
    hashes_end = _hash_snapshot(build_info)
    calibration_path = out / "calibration" / "c0_calibration.json"
    calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else {}
    completeness_ok = bool(gates.get("completeness", {}).get("pass", False))
    calibration_impl = calibration.get("implementation", {})
    implementation_ok = bool(completeness_ok and calibration_impl.get("counts_ok", False) and calibration_impl.get("schema_ok", False) and not calibration_impl.get("missing_gates") and hashes_start == hashes_end)
    numerical_ok = bool(calibration_impl.get("numerical_ok", False) and gates.get("numerical", {}).get("pass", False) and gates.get("repeat_determinism", {}).get("pass", False) and gates.get("fine_dt", {}).get("pass", False))
    final_verdict = formal_verdict(implementation_ok=implementation_ok, numerical_ok=numerical_ok, calibration_verdict="C0_SELECTION_READY", holdout_ok=bool(gates["pass"]))
    formal = {"version": "c0-holdout-v2", "phase": "C0-B", "verdict": "C0_HOLDOUT_PASS" if gates["pass"] else "C0_HOLDOUT_FAILURE", "formal_verdict": final_verdict, "selection_sha256": selection_sha, "selected_gamma": gamma, "counterfactual_calibration": True, "production_eligible": False, "pins": selection.get("pins", {}), "mask_records": mask_records, "initial_conditions": perturbations, "trial_count": len(rows), "expected_trial_count": 176, "ramp_count": len(ramps), "expected_ramp_count": 2, "gates": gates, "implementation": {"completeness_ok": completeness_ok, "calibration_ok": implementation_ok, "hash_stability": hashes_start == hashes_end}, "hashes_start": hashes_start, "hashes_end": hashes_end, "build": build_info, "wall_time_seconds": time.time() - start}
    np.savez_compressed(holdir / "c0_holdout_raw.npz", trial_json=np.asarray(json.dumps(rows, sort_keys=True)), ramp_json=np.asarray(json.dumps(ramps, sort_keys=True)), kc_active_bitset=np.asarray(bitsets, dtype=np.uint8))
    (holdir / "c0_holdout.json").write_text(json.dumps(formal, indent=2, sort_keys=True) + "\n")
    artifact_paths = {"calibration_json": calibration_path, "selection": selection_path, "holdout_json": holdir / "c0_holdout.json", "holdout_raw": holdir / "c0_holdout_raw.npz"}
    formal["artifact_sha256"] = {name: _sha(path) for name, path in artifact_paths.items() if path.exists()}
    formal_path = out / "c0_formal.json"
    formal_path.write_text(json.dumps(formal, indent=2, sort_keys=True) + "\n")
    (out / "c0_report.md").write_text("# V2-C0 graded-release calibration v2\n\n"
                                     f"Verdict: `{final_verdict}`\n\n"
                                     f"Calibration selected gamma `{gamma}`; holdout trials: `{len(rows)}/176`, ramps: `{len(ramps)}/2`.\n\n"
                                     "This synthetic holdout is independent of calibration masks and reads only the immutable selection.\n")
    manifest_path = out / "c0_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update({"holdout_hashes_start": hashes_start, "holdout_hashes_end": hashes_end, "holdout_artifacts": {name: _sha(path) for name, path in artifact_paths.items() if path.exists()}, "final_verdict": final_verdict})
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return formal


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if "--gamma" in argv or "--gain" in argv:
        raise SystemExit("C0 holdout rejects CLI gain overrides; use the immutable c0_selection.json")
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, default=OUT); parser.add_argument("--selection", type=Path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(); print(json.dumps(run(args.out, args.selection), indent=2, sort_keys=True))
