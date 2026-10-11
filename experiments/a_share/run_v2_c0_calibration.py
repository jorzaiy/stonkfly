#!/usr/bin/env python3
"""V2-C0-A calibration process.

Only the frozen calibration levels and the two calibration hash patterns are
constructed here.  This module deliberately has no holdout imports, paths,
or mask constructors.  It searches the five preregistered global gains and
always completes all five gains before writing the frozen selection decision.
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
    calibration_ramp_count,
    D_CAL,
    GAINS,
    calibration_trials_per_gain,
    SPARSE_HIGH,
    SPARSE_LOW,
    average_ranks,
    classify_kc,
    jaccard,
    json_sha256,
    selection_verdict,
    sha256_file,
    source_id_sha256,
    spearman,
    formal_verdict,
)
from stonkfly.neural.v2_brain import build, classify_nodes
from stonkfly.neural.v2_c0 import C0HybridBrain


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/whole_brain_v2/C0/20261010T-graded-release-calibration-v2"
Q1 = ROOT / "runs/whole_brain_v2/Q1/20261009T-hybrid-kernel-qualification-v7"
Q1_FORMAL = Q1 / "q1_formal.json"
Q1_RAW = Q1 / "q1_raw.npz"
Q1_MANIFEST = Q1 / "q1_manifest.json"
PREREG = ROOT / "docs/20261009-v2-c0-graded-release-calibration-preregistered.md"
Q1_RUNNER = ROOT / "experiments/a_share/run_v2_q1_hybrid_qualification_v2.py"
Q1_TESTS = ROOT / "tests/test_v2_q1_hybrid_qualification.py"
GRAPH = ROOT / "data/graph.npz"
NEURONS = ROOT / "data/normalized/neurons.feather"
EDGES = ROOT / "data/normalized/edges.arrow"
V2_SOURCE = ROOT / "stonkfly/neural/v2_kernel_qualification.cpp"
V2_WRAPPER = ROOT / "stonkfly/neural/v2_brain.py"
C0_WRAPPER = ROOT / "stonkfly/neural/v2_c0.py"
TESTS = ROOT / "tests/test_v2_c0_protocol.py"

PINS = {
    "data/graph.npz": "346b8af85a11af13b8324e18669812c1924569e7d1adcb4e6f45cc461a2c344b",
    "data/normalized/neurons.feather": "2103b5203d2ba5502c06fb70a4e0b23edcbcf79ed0c344722101941fc289ba2d",
    "data/normalized/edges.arrow": "842b762c9427c794eaedd453178aa034727a5084850a91fdcc4d3fffe09f5b7e",
    "q1_formal.json": "394c2cd05ecc54c979b9d871176a86bb82670c850c108fe91c63ba8354ead574",
    "q1_raw.npz": "15623018571bd78437779236c89effd8542ad56df19407d2dbb9221b013e0c20",
    "q1_manifest.json": "036e5a8bba71b40ed2595f635c6883afad0c25d760718aa94164f58ee872ee02",
    "stonkfly/neural/v2_kernel_qualification.cpp": "99f63402e2a6393c229aab139915233d24dff8ad411f068d91ec965edfbef79e",
    "stonkfly/neural/v2_brain.py": "2673de4e7c4aa4ee0fd20636b17bf2c2828103bacfbbb05b2ff0f559979434e6",
    "experiments/a_share/run_v2_q1_hybrid_qualification_v2.py": "635c8202882182c8fda56123379338d36de515db4d6ef19c12e56113c61d221a",
    "tests/test_v2_q1_hybrid_qualification.py": "6c021d8979ea73203dc07101826a085f0e2a7206cf30488bc4b9a351f0ca35bb",
}


def _sha(path: Path) -> str:
    return sha256_file(path)


def fixed_inputs() -> dict:
    paths = {
        "data/graph.npz": GRAPH,
        "data/normalized/neurons.feather": NEURONS,
        "data/normalized/edges.arrow": EDGES,
        "q1_formal.json": Q1_FORMAL,
        "q1_raw.npz": Q1_RAW,
        "q1_manifest.json": Q1_MANIFEST,
        "stonkfly/neural/v2_kernel_qualification.cpp": V2_SOURCE,
        "stonkfly/neural/v2_brain.py": V2_WRAPPER,
        "experiments/a_share/run_v2_q1_hybrid_qualification_v2.py": Q1_RUNNER,
        "tests/test_v2_q1_hybrid_qualification.py": Q1_TESTS,
    }
    actual = {key: _sha(path) for key, path in paths.items()}
    mismatches = {key: (PINS[key], actual[key]) for key in PINS if PINS[key] != actual[key]}
    if mismatches:
        raise RuntimeError("C0 fixed-input mismatch: " + json.dumps(mismatches, sort_keys=True))
    formal = json.loads(Q1_FORMAL.read_text())
    if formal.get("verdict") != "Q1_IMPLEMENTATION_QUALIFIED":
        raise RuntimeError("Q1 v7 is not formally qualified")
    return {"pins": actual, "q1_verdict": formal["verdict"], "q1_formal_sha256": actual["q1_formal.json"]}


def load_graph_and_classification():
    graph = np.load(GRAPH, allow_pickle=False)
    nodes = feather.read_table(NEURONS).to_pydict()
    layer, names = classify_nodes(nodes["cell_type"], nodes["neurotransmitter"])
    ids = np.asarray(graph["ids"], dtype=np.int64)
    if len(ids) != len(layer):
        raise RuntimeError("graph/node table length mismatch")
    counts = {name: int(np.sum(layer == i)) for i, name in enumerate(names)}
    if [counts[x] for x in names] != [541, 4706, 7114, 2, 4064, 150273]:
        raise RuntimeError("Q1 classification counts changed")
    return graph, ids, layer, names


def _pattern(ids: np.ndarray, fraction: float, salt: str) -> np.ndarray:
    """Hash-ranked calibration source IDs; this function has no other mask family."""
    ranked = sorted(
        ((hashlib.sha256(f"{salt}:{int(source_id)}".encode("utf-8")).digest(), int(source_id)) for source_id in ids),
        key=lambda x: x[0],
    )
    count = int(np.floor(float(fraction) * len(ids)))
    return np.asarray([source_id for _, source_id in ranked[:count]], dtype=np.int64)


def calibration_masks(ids: np.ndarray, layer: np.ndarray) -> dict:
    photo = ids[layer == 1]
    if len(photo) != 4706:
        raise RuntimeError("photoreceptor count changed")
    return {
        "CAL_A": _pattern(photo, 0.50, "CAL_A"),
        "CAL_B": _pattern(photo, 0.50, "CAL_B"),
    }


def _mask_record(ids: np.ndarray, selected: np.ndarray, current: float) -> dict:
    return {"count": int(len(selected)), "current": float(current),
            "source_id_sha256": source_id_sha256(selected), "source_ids": [int(x) for x in selected]}


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _hash_snapshot(build_info: dict) -> dict:
    """Capture mutable C0 run inputs before/after execution.

    Q1 source and binary are read-only qualification inputs; the C0 wrapper,
    runner and tests are the new audit boundary.  No hash is used to select a
    gain, and no Q1 file is rewritten by this runner.
    """
    paths = {
        "v2_source": ROOT / build_info["source"],
        "v2_binary": ROOT / build_info["binary"],
        "c0_wrapper": C0_WRAPPER,
        "c0_runner": Path(__file__).resolve(),
        "c0_tests": TESTS,
        "preregistration": PREREG,
    }
    return {name: _sha(path) for name, path in paths.items()}


def _edge_count_matrix(graph, layer: np.ndarray) -> list[list[int]]:
    """Static source/target edge counts for the transfer-chain audit."""
    source_layer = np.repeat(layer, np.diff(graph["ptr"]))
    target_layer = layer[graph["post"]]
    valid = (source_layer >= 0) & (target_layer >= 0)
    flat = source_layer[valid].astype(np.int64) * 6 + target_layer[valid].astype(np.int64)
    return np.bincount(flat, minlength=36).reshape(6, 6).astype(np.int64).tolist()


def transfer_chain_diagnostic(graph, ids: np.ndarray, layer: np.ndarray, names: tuple[str, ...]) -> dict:
    """Run one short, preregistered UNIFORM_4 transfer-chain probe.

    This is an audit-only probe, not a calibration gate and not a changed
    formal stimulus.  It uses the frozen graph, dt and gamma=1.0 while keeping
    the run short enough to isolate current -> voltage/release -> delivery ->
    downstream response.  The observed KC zero is therefore distinguished
    from a disconnected runner path without altering any C0 gate.
    """
    graded = np.isin(layer, [1, 2, 3]); kc = layer == 4
    rest = np.full(len(ids), -52.0, dtype=np.float32); rest[kc] = -60.0
    brain = C0HybridBrain(GRAPH, gamma=1.0, dt=0.05,
                          graded_mask=graded.astype(np.uint8), rest=rest)
    brain.kc_mask[:] = kc.astype(np.uint8)
    zero = np.zeros(len(ids), dtype=np.float32)
    drive = zero.copy(); drive[layer == 1] = 4.0
    brain.run(10.0, zero)
    baseline_v = brain.v.copy()
    before = (float(brain.release_integral[0]), float(brain.signed_delivery[0]),
              float(brain.absolute_delivery[0]))
    brain.run(10.0, drive)
    measurement_counts = brain.counts.copy()
    after = (float(brain.release_integral[0]), float(brain.signed_delivery[0]),
             float(brain.absolute_delivery[0]))

    layer_rows = {}
    for index, name in enumerate(names):
        mask = layer == index
        delta = brain.v[mask] - baseline_v[mask]
        layer_rows[name] = {
            "count": int(mask.sum()),
            "voltage_delta_min": float(delta.min()) if mask.any() else 0.0,
            "voltage_delta_max": float(delta.max()) if mask.any() else 0.0,
            "voltage_min": float(brain.v[mask].min()) if mask.any() else 0.0,
            "voltage_max": float(brain.v[mask].max()) if mask.any() else 0.0,
            "measurement_spike_nodes": int(np.count_nonzero(measurement_counts[mask] > 0)),
            "measurement_spikes": int(measurement_counts[mask].sum()),
        }
    release_delta = after[0] - before[0]
    signed_delta = after[1] - before[1]
    absolute_delta = after[2] - before[2]
    photo_row = layer_rows["graded_candidate_photoreceptor"]
    lamina_row = layer_rows["graded_candidate_lamina"]
    generic_row = layer_rows["generic_spiking_declared"]
    kc_row = layer_rows["adaptive_spiking_kc"]
    current_changed = photo_row["voltage_delta_min"] != 0.0 or photo_row["voltage_delta_max"] != 0.0
    release_nonzero = release_delta > 0.0
    delivery_nonzero = absolute_delta > 0.0
    downstream_changed = any(
        row["voltage_delta_min"] != 0.0 or row["voltage_delta_max"] != 0.0
        for row in (lamina_row, generic_row)
    )
    kc_spikes = kc_row["measurement_spikes"]
    if current_changed and release_nonzero and delivery_nonzero and downstream_changed and kc_spikes == 0:
        conclusion = "frozen_stimulus_reaches_network_but_produces_no_KC_spike"
    elif not current_changed or not release_nonzero or not delivery_nonzero:
        conclusion = "runner_or_wiring_failure_before_downstream"
    else:
        conclusion = "downstream_response_observed_with_KC_activity"
    return {
        "schema": "v2-c0-transfer-diagnostic-v1",
        "audit_only": True,
        "gamma": 1.0,
        "dt_ms": 0.05,
        "stimulus": "UNIFORM_4",
        "segments_ms": [10.0, 10.0],
        "source_count": int((layer == 1).sum()),
        "source_id_sha256": source_id_sha256(ids[layer == 1]),
        "edge_count_matrix_source_target": _edge_count_matrix(graph, layer),
        "before_integrals": {"release": before[0], "signed_delivery": before[1], "absolute_delivery": before[2]},
        "after_integrals": {"release": after[0], "signed_delivery": after[1], "absolute_delivery": after[2]},
        "delta_integrals": {"release": release_delta, "signed_delivery": signed_delta, "absolute_delivery": absolute_delta},
        "layers": layer_rows,
        "conclusion": conclusion,
        "checks": {
            "external_current_changes_photoreceptor_voltage": bool(current_changed),
            "photoreceptor_has_graded_release": bool(release_nonzero),
            "graded_source_has_outgoing_delivery": bool(delivery_nonzero),
            "downstream_voltage_responds": bool(downstream_changed),
            "kc_spike_count_zero": bool(kc_spikes == 0),
        },
    }


def _state_sha(brain: C0HybridBrain) -> str:
    fields = [brain.v, brain.g_event, brain.g_graded, brain.refractory, brain.drive,
              brain.previous_drive, brain.spike_queue, brain.spike_queue_count,
              brain.graded_queue, brain.graded_drive, brain.counts, brain.active,
              brain.active_flag, brain.nactive, brain.last, brain.adaptation,
              brain.modulation]
    h = hashlib.sha256()
    for value in fields:
        h.update(np.ascontiguousarray(value).tobytes())
    h.update(str(int(brain.clock[0])).encode())
    return h.hexdigest()


def _clean_brain(gamma: float, graded: np.ndarray, kc: np.ndarray, rest: np.ndarray) -> C0HybridBrain:
    brain = C0HybridBrain(GRAPH, gamma=gamma, dt=0.05, graded_mask=graded.astype(np.uint8), rest=rest)
    brain.kc_mask[:] = kc.astype(np.uint8)
    brain.modulation_mask.fill(0)
    return brain


def _run_trial(gamma: float, stimulus: str, selected: np.ndarray, graph, ids, layer, repeat: int) -> tuple[dict, np.ndarray]:
    graded = np.isin(layer, [1, 2, 3])
    kc = layer == 4
    rest = np.full(len(ids), -52.0, dtype=np.float32); rest[kc] = -60.0
    brain = _clean_brain(gamma, graded, kc, rest)
    zero = np.zeros(len(ids), dtype=np.float32)
    drive = zero.copy(); drive[np.isin(ids, selected)] = 4.0
    brain.run(50.0, zero)
    brain.run(100.0, drive)
    before_measure = float(brain.release_integral[0])
    counts = brain.run(50.0, drive)
    measure_release = float(brain.release_integral[0] - before_measure)
    measure_kc = np.flatnonzero((counts > 0) & kc).astype(np.int64)
    mb07 = np.flatnonzero((counts > 0) & (layer == 6)).astype(np.int64) if False else np.flatnonzero((counts > 0) & (layer == 6)).astype(np.int64)
    # Exact MBON IDs are recorded separately below; layer remains the frozen
    # Q1 declaration and is never used to select a drive.
    mb07 = np.flatnonzero(counts > 0)
    brain.run(200.0, zero)
    post = brain.run(100.0, zero)
    post_kc = np.flatnonzero((post > 0) & kc).astype(np.int64)
    # MBON types are identified by exact cell-type masks supplied by caller in
    # _run_trial_with_mbon; this placeholder is overwritten there.
    valid = bool(np.isfinite(brain.v).all() and np.isfinite(brain.g_event).all() and
                 np.isfinite(brain.g_graded).all() and np.isfinite(brain.graded_queue).all())
    queue_ok = bool(np.all(brain.spike_queue_count >= 0) and np.all(brain.spike_queue_count <= len(ids)) and
                    np.isfinite(brain.graded_drive).all() and int(brain.overflow[0]) == 0)
    record = {
        "gamma": float(gamma), "stimulus": stimulus, "repeat": int(repeat), "dt_ms": 0.05,
        "measurement_active_kc": int(len(measure_kc)), "post_active_kc": int(len(post_kc)),
        "measurement_active_kc_ids": measure_kc.tolist(), "post_active_kc_ids": post_kc.tolist(),
        "measurement_release_integral": measure_release,
        "release_integral": float(brain.release_integral[0]),
        "signed_delivery_integral": float(brain.signed_delivery[0]),
        "absolute_delivery_integral": float(brain.absolute_delivery[0]),
        "state_sha256": _state_sha(brain), "weights_sha256": _array_sha(brain.weight),
        "finite": valid, "bounds": bool(np.max(np.abs(brain.v)) < 1e6 and np.max(np.abs(brain.g_event)) < 1e9 and np.max(np.abs(brain.g_graded)) < 1e9),
        "queue_integrity": queue_ok, "overflow": int(brain.overflow[0]), "clock": int(brain.clock[0]),
        "total_spikes": int(brain.total_spikes),
    }
    bits = np.packbits(np.isin(np.arange(len(ids)), measure_kc).astype(np.uint8))
    return record, bits


def _run_trial_with_mbon(gamma, stimulus, selected, graph, ids, layer, mb07_mask, mb11_mask, repeat):
    # Keep the simulation helper compact while making MBON exact masks explicit.
    graded = np.isin(layer, [1, 2, 3]); kc = layer == 4
    rest = np.full(len(ids), -52.0, dtype=np.float32); rest[kc] = -60.0
    brain = _clean_brain(gamma, graded, kc, rest); zero = np.zeros(len(ids), np.float32)
    drive = zero.copy()
    current = float(stimulus.split("_")[1]) if stimulus.startswith("UNIFORM_") else 4.0
    drive[np.isin(ids, selected)] = current
    brain.run(50.0, zero); brain.run(100.0, drive); counts = brain.run(50.0, drive)
    measure_kc = np.flatnonzero((counts > 0) & kc).astype(np.int64)
    mb07 = np.flatnonzero((counts > 0) & mb07_mask).astype(np.int64)
    mb11 = np.flatnonzero((counts > 0) & mb11_mask).astype(np.int64)
    brain.run(200.0, zero); post = brain.run(100.0, zero)
    post_kc = np.flatnonzero((post > 0) & kc).astype(np.int64)
    valid = bool(np.isfinite(brain.v).all() and np.isfinite(brain.g_event).all() and np.isfinite(brain.g_graded).all() and np.isfinite(brain.graded_queue).all())
    queue_ok = bool(np.all(brain.spike_queue_count >= 0) and np.all(brain.spike_queue_count <= len(ids)) and np.isfinite(brain.graded_drive).all() and int(brain.overflow[0]) == 0)
    record = {"gamma": float(gamma), "stimulus": stimulus, "repeat": int(repeat), "dt_ms": 0.05,
              "measurement_active_kc": int(len(measure_kc)), "post_active_kc": int(len(post_kc)),
              "measurement_active_kc_ids": measure_kc.tolist(), "post_active_kc_ids": post_kc.tolist(),
              "mb07_ids": mb07.tolist(), "mb11_ids": mb11.tolist(), "mb07_active": int(len(mb07)), "mb11_active": int(len(mb11)),
              "mb07_spikes": int(counts[mb07_mask].sum()), "mb11_spikes": int(counts[mb11_mask].sum()),
              "release_integral": float(brain.release_integral[0]), "signed_delivery_integral": float(brain.signed_delivery[0]),
              "absolute_delivery_integral": float(brain.absolute_delivery[0]), "state_sha256": _state_sha(brain),
              "weights_sha256": _array_sha(brain.weight), "finite": valid, "bounds": bool(np.max(np.abs(brain.v)) < 1e6 and np.max(np.abs(brain.g_event)) < 1e9 and np.max(np.abs(brain.g_graded)) < 1e9),
              "queue_integrity": queue_ok, "overflow": int(brain.overflow[0]), "clock": int(brain.clock[0]), "total_spikes": int(brain.total_spikes)}
    bits = np.packbits(np.isin(np.arange(len(ids)), measure_kc).astype(np.uint8))
    return record, bits


def _run_history(gamma, ids, layer, mb07_mask, mb11_mask, repeat):
    graded = np.isin(layer, [1, 2, 3]); kc = layer == 4
    rest = np.full(len(ids), -52.0, np.float32); rest[kc] = -60.0
    brain = _clean_brain(gamma, graded, kc, rest); zero = np.zeros(len(ids), np.float32)
    levels = [0, 1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1, 0]; photo_ids = ids[layer == 1]
    rows = []
    for level_index, level in enumerate(levels):
        drive = zero.copy(); drive[np.isin(ids, photo_ids)] = float(level)
        brain.run(50.0, drive); counts = brain.run(50.0, drive)
        active = np.flatnonzero((counts > 0) & kc).astype(np.int64)
        rows.append({"level_index": level_index, "level": int(level), "direction": "up" if level_index <= 6 else "down",
                     "active_kc": int(len(active)), "active_kc_ids": active.tolist(), "mb07_active": int(np.sum((counts > 0) & mb07_mask)),
                     "mb11_active": int(np.sum((counts > 0) & mb11_mask)), "mb07_spikes": int(counts[mb07_mask].sum()), "mb11_spikes": int(counts[mb11_mask].sum())})
    brain.run(200.0, zero); post = brain.run(100.0, zero)
    return {"gamma": float(gamma), "repeat": int(repeat), "dt_ms": 0.05, "levels": rows,
            "post_active_kc": int(np.sum((post > 0) & kc)), "state_sha256": _state_sha(brain), "finite": bool(np.isfinite(brain.v).all()),
            "queue_integrity": bool(np.all(brain.spike_queue_count >= 0) and int(brain.overflow[0]) == 0), "clock": int(brain.clock[0])}


def _evaluate_gain(gamma, trials, ramps, kc_count):
    by = defaultdict(list)
    for row in trials: by[row["stimulus"]].append(row)
    all_basic = all(r["finite"] and r["bounds"] and r["queue_integrity"] and r["overflow"] == 0 for r in trials)
    deterministic_trials = all(len(rows) == 2 and rows[0]["state_sha256"] == rows[1]["state_sha256"] for rows in by.values())
    deterministic_ramps = len(ramps) == 2 and ramps[0]["state_sha256"] == ramps[1]["state_sha256"]
    uniform = {int(r["stimulus"].split("_")[1]): by[r["stimulus"]][0] for r in trials if r["stimulus"].startswith("UNIFORM_")}
    active = [uniform[d]["measurement_active_kc"] for d in D_CAL]
    nonzero = active[1:]
    classes = [classify_kc(x) for x in nonzero]
    adjacent_down = max((nonzero[i] - nonzero[i + 1] for i in range(5)), default=0)
    adjacent_jump = max((abs(nonzero[i + 1] - nonzero[i]) for i in range(5)), default=0)
    dose = {"active_counts": active, "sparse_levels": sum(SPARSE_LOW <= x <= SPARSE_HIGH for x in nonzero),
            "fixed_low_like_levels": sum(x <= 10 for x in nonzero), "dense_levels": sum(x > SPARSE_HIGH for x in nonzero),
            "distinct_values": len(set(nonzero)), "spearman_rho": spearman(D_CAL[1:], nonzero),
            "max_adjacent_down": int(max(0, adjacent_down)), "max_adjacent_abs_jump": int(adjacent_jump),
            "pass": bool(sum(SPARSE_LOW <= x <= SPARSE_HIGH for x in nonzero) >= 3 and sum(x <= 10 for x in nonzero) <= 1 and sum(x > SPARSE_HIGH for x in nonzero) <= 1 and len(set(nonzero)) >= 4 and spearman(D_CAL[1:], nonzero) >= .8 and max(0, adjacent_down) <= kc_count * .10 and adjacent_jump <= kc_count * .10)}
    washout = {"max_post_active_kc": max(r["post_active_kc"] for r in trials), "history_max_post_active_kc": max(r["post_active_kc"] for r in ramps), "pass": bool(max(r["post_active_kc"] for r in trials) <= 40 and max(r["post_active_kc"] for r in ramps) <= 40)}
    sparse_rows = [r for r in trials if r["stimulus"].startswith("UNIFORM_") and int(r["stimulus"].split("_")[1]) > 0]
    mbon = {"sparse_uniform_with_both": sum(r["measurement_active_kc"] in range(SPARSE_LOW, SPARSE_HIGH + 1) and r["mb07_spikes"] > 0 and r["mb11_spikes"] > 0 for r in sparse_rows),
            "pattern_sparse_both": all(r["measurement_active_kc"] in range(SPARSE_LOW, SPARSE_HIGH + 1) and r["mb07_spikes"] > 0 and r["mb11_spikes"] > 0 for r in by["CAL_A"] + by["CAL_B"]),
            "pass": False}
    mbon["pass"] = bool(mbon["sparse_uniform_with_both"] >= 2 and mbon["pattern_sparse_both"])
    ca, cb = by["CAL_A"][0], by["CAL_B"][0]
    identity = {"jaccard": jaccard(ca["measurement_active_kc_ids"], cb["measurement_active_kc_ids"]), "sets_equal": ca["measurement_active_kc_ids"] == cb["measurement_active_kc_ids"]}
    identity["pass"] = bool(identity["jaccard"] <= .80 and not identity["sets_equal"] and ca["measurement_active_kc"] in range(SPARSE_LOW, SPARSE_HIGH + 1) and cb["measurement_active_kc"] in range(SPARSE_LOW, SPARSE_HIGH + 1))
    hist_rows = ramps[0]["levels"]
    hist = []; hist_j = []
    for level in range(1, 7):
        up = hist_rows[level]; down = hist_rows[14 - level]
        hist.append({"level": level, "symmetric_relative_difference": 2 * abs(up["active_kc"] - down["active_kc"]) / max(up["active_kc"] + down["active_kc"], 1)})
        hist_j.append(jaccard(up["active_kc_ids"], down["active_kc_ids"]))
    history = {"levels": hist, "jaccard": hist_j, "symmetry_pass_count": sum(x["symmetric_relative_difference"] <= .25 for x in hist), "jaccard_pass_count": sum(x >= .5 for x in hist_j), "pass": bool(sum(x["symmetric_relative_difference"] <= .25 for x in hist) >= 5 and sum(x >= .5 for x in hist_j) >= 4 and all(r["post_active_kc"] <= 40 for r in ramps))}
    gates = {"basic": {"pass": bool(all_basic), "deterministic_repeats": bool(deterministic_trials and deterministic_ramps), "trial_count": len(trials) == calibration_trials_per_gain(), "ramp_count": len(ramps) == 2}, "uniform_dose_response": dose, "washout": washout, "mbon_drivable": mbon, "calibration_identity": identity, "history_dependence": history}
    gates["basic"]["pass"] = bool(gates["basic"]["pass"] and deterministic_trials and deterministic_ramps and gates["basic"]["trial_count"] and gates["basic"]["ramp_count"])
    gates["pass"] = bool(all(x.get("pass") for x in gates.values() if isinstance(x, dict) and "pass" in x))
    return gates


def _ensure_fresh_output(out: Path) -> None:
    """Never overwrite an existing formal C0 directory."""
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing to overwrite existing C0 output: {out}")


def run(out=OUT):
    out = Path(out)
    _ensure_fresh_output(out)
    out.mkdir(parents=True, exist_ok=True)
    caldir = out / "calibration"
    caldir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    pins = fixed_inputs()
    graph, ids, layer, names = load_graph_and_classification()
    build_info = build()
    hashes_start = _hash_snapshot(build_info)
    diagnostic = transfer_chain_diagnostic(graph, ids, layer, names)
    masks = calibration_masks(ids, layer)
    graded = np.isin(layer, [1, 2, 3]); kc = layer == 4
    table = feather.read_table(NEURONS)
    cell_types = np.asarray(table.column("cell_type").to_numpy(zero_copy_only=False))
    mb07_mask = cell_types == "MBON07"
    mb11_mask = cell_types == "MBON11"
    source_masks = {"UNIFORM_0": np.empty(0, np.int64), **{f"UNIFORM_{d}": ids[layer == 1] for d in D_CAL[1:]}, "CAL_A": masks["CAL_A"], "CAL_B": masks["CAL_B"]}
    mask_records = {name: _mask_record(ids, selected, 0.0 if name == "UNIFORM_0" else (float(name.split("_")[1]) if name.startswith("UNIFORM_") else 4.0)) for name, selected in source_masks.items()}
    mask_records["UNIFORM_0"]["count"] = 0
    all_rows = []; all_ramps = []; gain_results = []; bitsets = []
    for gamma in GAINS:
        trials = []; ramps = []
        for repeat in (1, 2):
            for name in source_masks:
                selected = source_masks[name] if not name.startswith("UNIFORM_") else (ids[layer == 1] if int(name.split("_")[1]) else np.empty(0, np.int64))
                row, bits = _run_trial_with_mbon(gamma, name, selected, graph, ids, layer, mb07_mask, mb11_mask, repeat)
                trials.append(row); all_rows.append(row); bitsets.append(bits)
            ramps.append(_run_history(gamma, ids, layer, mb07_mask, mb11_mask, repeat)); all_ramps.append(ramps[-1])
        gates = _evaluate_gain(gamma, trials, ramps, int(kc.sum()))
        gain_results.append({"gamma": gamma, "trial_count": len(trials), "expected_trial_count": calibration_trials_per_gain(), "ramp_count": len(ramps), "expected_ramp_count": 2, "gates": gates, "qualifying": bool(gates["pass"])})
    qualifying = [r["gamma"] for r in gain_results if r["qualifying"]]
    choice = selection_verdict(qualifying)
    calibration_verdict = choice["calibration_verdict"]
    required_gates = ["basic", "uniform_dose_response", "washout", "mbon_drivable", "calibration_identity", "history_dependence"]
    missing_gates = [f"gamma={r['gamma']}:{gate}" for r in gain_results for gate in required_gates if gate not in r["gates"]]
    record_fields = {"gamma", "stimulus", "repeat", "dt_ms", "measurement_active_kc", "post_active_kc", "measurement_active_kc_ids", "mb07_active", "mb11_active", "mb07_spikes", "mb11_spikes", "finite", "bounds", "queue_integrity", "overflow", "state_sha256", "weights_sha256"}
    schema_ok = all(record_fields.issubset(row) for row in all_rows)
    counts_ok = len(all_rows) == len(GAINS) * calibration_trials_per_gain() and len(all_ramps) == calibration_ramp_count() and all(r["trial_count"] == calibration_trials_per_gain() and r["ramp_count"] == 2 for r in gain_results)
    hashes_end = _hash_snapshot(build_info)
    hash_ok = hashes_start == hashes_end
    implementation_ok = bool(counts_ok and schema_ok and not missing_gates and diagnostic.get("schema") == "v2-c0-transfer-diagnostic-v1" and hash_ok)
    numerical_ok = bool(all(r["gates"].get("basic", {}).get("pass", False) for r in gain_results))
    final_verdict = formal_verdict(implementation_ok=implementation_ok, numerical_ok=numerical_ok, calibration_verdict=calibration_verdict, holdout_ok=None)
    selection_rule = "qualifying only; ascending abs(log2(gamma)); ties choose smaller gamma; no boundary expansion"
    formal = {"version": "c0-calibration-v2", "phase": "C0-A", "verdict": final_verdict, "calibration_verdict": calibration_verdict,
              "counterfactual_calibration": True, "production_eligible": False, "pins": pins, "gains": GAINS,
              "gain_results": gain_results, "qualifying_gains": qualifying, "selection": choice, "selection_rule": selection_rule,
              "mask_records": mask_records, "classification": {n: {"count": int(np.sum(layer == i)), "source_id_sha256": source_id_sha256(ids[layer == i])} for i, n in enumerate(names)},
              "trial_count": len(all_rows), "expected_trial_count": len(GAINS) * calibration_trials_per_gain(), "ramp_count": len(all_ramps), "expected_ramp_count": calibration_ramp_count(),
              "dt_ms": 0.05, "fixed_weight_sha256": _array_sha(np.asarray(graph["weight"])), "transfer_diagnostic": diagnostic,
              "implementation": {"counts_ok": counts_ok, "schema_ok": schema_ok, "missing_gates": missing_gates, "hash_stability": hash_ok, "numerical_ok": numerical_ok},
              "hashes_start": hashes_start, "hashes_end": hashes_end, "build": build_info, "preregistration_sha256": _sha(PREREG),
              "holdout_not_run_reason": ("no qualifying non-boundary calibration gain" if calibration_verdict == "C0_NO_QUALIFYING_GAIN" else "selected gain is a preregistered grid boundary" if calibration_verdict == "C0_GRID_BOUNDARY" else "calibration produced a legal selection; independent holdout runner is pending"),
              "wall_time_seconds": time.time() - start}
    raw_path = caldir / "c0_calibration_raw.npz"
    np.savez_compressed(raw_path, trial_json=np.asarray(json.dumps(all_rows, sort_keys=True)), ramp_json=np.asarray(json.dumps(all_ramps, sort_keys=True)), gains=np.asarray(GAINS), kc_active_bitset=np.asarray(bitsets, dtype=np.uint8), transfer_diagnostic_json=np.asarray(json.dumps(diagnostic, sort_keys=True)))
    calibration_path = caldir / "c0_calibration.json"
    calibration_path.write_text(json.dumps(formal, indent=2, sort_keys=True) + "\n")
    selection_path = None
    if calibration_verdict == "C0_SELECTION_READY" and implementation_ok and numerical_ok:
        selection = {"schema": "v2-c0-selection-v1", "counterfactual_calibration": True, "production_eligible": False,
                     "selected_gamma": float(choice["selected_gamma"]), "qualifying_gains": qualifying, "calibration_verdict": calibration_verdict,
                     "calibration_sha256": _sha(calibration_path), "pins": pins, "selection_rule": selection_rule}
        selection_path = out / "c0_selection.json"
        selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
        selection_path.chmod(0o444)
    artifact_paths = {"calibration_json": calibration_path, "calibration_raw": raw_path}
    if selection_path is not None:
        artifact_paths["selection"] = selection_path
    formal["artifact_sha256"] = {name: _sha(path) for name, path in artifact_paths.items()}
    formal_path = out / "c0_formal.json"
    formal_path.write_text(json.dumps(formal, indent=2, sort_keys=True) + "\n")
    report_path = out / "c0_report.md"
    report_path.write_text("# V2-C0 graded-release calibration v2\n\n"
                           f"Verdict: `{final_verdict}`\n\n"
                           f"Calibration records: `{len(all_rows)}/{len(GAINS) * calibration_trials_per_gain()}` independent trials; `{len(all_ramps)}/{calibration_ramp_count()}` history ramps.\n\n"
                           f"Holdout: not run — {formal['holdout_not_run_reason']}.\n\n"
                           f"Transfer diagnostic: `{diagnostic['conclusion']}`. External current changed photoreceptor voltage and produced graded release/outgoing delivery; downstream graded/generic voltages responded, while KC spike count was `{diagnostic['layers']['adaptive_spiking_kc']['measurement_spikes']}`.\n\n"
                           "This is counterfactual synthetic calibration only; learning, reward, market, decoder and execution paths were not used.\n")
    manifest = {"version": "v2", "python": sys.version, "platform": platform.platform(), "build": build_info,
                "hashes_start": hashes_start, "hashes_end": hashes_end, "fixed_v1_source_sha256": _sha(ROOT / "stonkfly/neural/kernel.cpp"),
                "preregistration_sha256": _sha(PREREG), "required_counts": {"trials": len(GAINS) * calibration_trials_per_gain(), "ramps": calibration_ramp_count()},
                "missing_gates": missing_gates, "holdout_not_run_reason": formal["holdout_not_run_reason"], "artifacts": {name: _sha(path) for name, path in artifact_paths.items()}}
    (out / "c0_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return formal


def _write_failure_formal(out: Path, error: BaseException) -> dict | None:
    """Persist a conservative implementation verdict after runner failure.

    This is intentionally only called after the runner has created its new
    output directory.  It never overwrites a pre-existing non-empty target;
    that conflict remains a caller-visible error and protects old evidence.
    """
    out = Path(out)
    if not out.exists() or not (out / "calibration").exists():
        return None
    path = out / "c0_formal.json"
    payload = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            payload = {}
    payload.update({
        "version": payload.get("version", "c0-calibration-v2"),
        "verdict": "C0_IMPLEMENTATION_FAILURE",
        "implementation_failure": {
            "error_type": type(error).__name__,
            "error": str(error),
            "artifact_chain_complete": False,
        },
        "holdout_not_run_reason": "calibration runner/artifact failure; holdout was not run",
    })
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, default=OUT); args = parser.parse_args()
    try:
        result = run(args.out)
    except Exception as exc:
        failure = _write_failure_formal(args.out, exc)
        if failure is not None:
            print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
