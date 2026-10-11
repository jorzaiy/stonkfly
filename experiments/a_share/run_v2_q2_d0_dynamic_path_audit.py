#!/usr/bin/env python3
"""Frozen V2-Q2-D0 dynamic-path and open-loop counterfactual audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc

from stonkfly.neural.q2_d0 import (
    SPAN_MV,
    TAU_EMIT_MS,
    array_sha256,
    bfs_shortest_hops,
    counterfactual_delivery,
    interpretation_flags,
    q1_positive_release,
    signed_deviation,
    source_id_sha256,
)
from stonkfly.neural.v2_brain import classify_nodes
from stonkfly.neural.v2_c0 import C0HybridBrain


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "runs/whole_brain_v2/Q2/D0/20261010T-dynamic-path-audit-v1"
GRAPH = ROOT / "data/graph.npz"
NEURONS = ROOT / "data/normalized/neurons.feather"
EDGES = ROOT / "data/normalized/edges.arrow"
PREREG = ROOT / "docs/20261010-v2-q2-d0-dynamic-path-audit-preregistered.md"
REVIEW = ROOT / "docs/20261010-v2-q2-tonic-graded-transmission-for-opus.md"
Q1_FORMAL = ROOT / "runs/whole_brain_v2/Q1/20261009T-hybrid-kernel-qualification-v7/q1_formal.json"
Q1_SOURCE = ROOT / "stonkfly/neural/v2_kernel_qualification.cpp"
Q1_WRAPPER = ROOT / "stonkfly/neural/v2_brain.py"
Q1_BINARY = ROOT / "runs/whole_brain_v2/Q1/.build/libv2_hybrid.so"
C0_WRAPPER = ROOT / "stonkfly/neural/v2_c0.py"
C0_ROOT = ROOT / "runs/whole_brain_v2/C0/20261010T-graded-release-calibration-v3"
C0_CAL = C0_ROOT / "calibration/c0_calibration.json"
C0_RAW = C0_ROOT / "calibration/c0_calibration_raw.npz"
C0_FORMAL = C0_ROOT / "c0_formal.json"
C0_MANIFEST = C0_ROOT / "c0_manifest.json"
C0_REPORT = C0_ROOT / "c0_report.md"
HELPERS = ROOT / "stonkfly/neural/q2_d0.py"
TESTS = ROOT / "tests/test_v2_q2_d0_dynamic_path_audit.py"

PINS = {
    "data/graph.npz": "346b8af85a11af13b8324e18669812c1924569e7d1adcb4e6f45cc461a2c344b",
    "data/normalized/neurons.feather": "2103b5203d2ba5502c06fb70a4e0b23edcbcf79ed0c344722101941fc289ba2d",
    "data/normalized/edges.arrow": "842b762c9427c794eaedd453178aa034727a5084850a91fdcc4d3fffe09f5b7e",
    "q1_formal": "394c2cd05ecc54c979b9d871176a86bb82670c850c108fe91c63ba8354ead574",
    "q1_source": "99f63402e2a6393c229aab139915233d24dff8ad411f068d91ec965edfbef79e",
    "q1_wrapper": "2673de4e7c4aa4ee0fd20636b17bf2c2828103bacfbbb05b2ff0f559979434e6",
    "q1_binary": "9df8110f1e2134f8a6d88cde46d3a8a6278590f291ad87b49161926d2fd53f41",
    "c0_wrapper": "cb66dcec500f57d815e918b104aa92cac9aee4c21e733fbe2d608017d168759c",
    "c0_calibration": "62f9bc19dbe5da93e4e05bc42d02ecf597abdcaf3f5c3e59e8843bfb7b5e2307",
    "c0_raw": "b00e9fb902fbcb9d16a36705565095b84ae61b48a5d6b1d0d9285498ef173173",
    "c0_formal": "6dd290e4758453bdc3bf52c61bd9d1fd2f28b9161b8aeffdf56db4838a6c00c3",
    "c0_manifest": "0799cbd07ef8f2a768d6d2a27d742332f3659098fc5f156c0f125828362e1705",
    "c0_report": "121fdc81ef7e92abfb4c67b22f96f3f6d212805c4794c3079abbdc5e24338293",
    "review": "11c5598f6c80f04fbf7207b8f2d12bd9006783e332908adb4da750f77e2227d8",
}

LAYER_NAMES = (
    "modulatory_declared", "graded_candidate_photoreceptor",
    "graded_candidate_lamina", "graded_candidate_apl",
    "adaptive_spiking_kc", "generic_spiking_declared",
)
SOURCE_CLASS_NAMES = ("photoreceptor", "L1", "L2", "L3", "L5", "APL")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                        default=str).encode()).hexdigest()


def fixed_inputs() -> dict:
    paths = {
        "data/graph.npz": GRAPH, "data/normalized/neurons.feather": NEURONS,
        "data/normalized/edges.arrow": EDGES, "q1_formal": Q1_FORMAL,
        "q1_source": Q1_SOURCE, "q1_wrapper": Q1_WRAPPER,
        "q1_binary": Q1_BINARY, "c0_wrapper": C0_WRAPPER,
        "c0_calibration": C0_CAL, "c0_raw": C0_RAW, "c0_formal": C0_FORMAL,
        "c0_manifest": C0_MANIFEST, "c0_report": C0_REPORT, "review": REVIEW,
    }
    actual = {name: sha256_file(path) for name, path in paths.items()}
    mismatch = {name: {"expected": PINS[name], "actual": value}
                for name, value in actual.items() if value != PINS[name]}
    if mismatch:
        raise RuntimeError("fixed-input hash mismatch: " + json.dumps(mismatch, sort_keys=True))
    if json.loads(Q1_FORMAL.read_text()).get("verdict") != "Q1_IMPLEMENTATION_QUALIFIED":
        raise RuntimeError("Q1 is not qualified")
    if json.loads(C0_FORMAL.read_text()).get("verdict") != "C0_NO_QUALIFYING_GAIN":
        raise RuntimeError("C0 v3 verdict changed")
    return actual


def hash_snapshot() -> dict[str, str]:
    paths = {
        "preregistration": PREREG, "runner": Path(__file__).resolve(),
        "helpers": HELPERS, "tests": TESTS, "q1_source": Q1_SOURCE,
        "q1_wrapper": Q1_WRAPPER, "q1_binary": Q1_BINARY,
        "c0_wrapper": C0_WRAPPER,
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def load_inputs():
    graph = np.load(GRAPH, allow_pickle=False)
    nodes = feather.read_table(NEURONS).to_pydict()
    ids = np.asarray(graph["ids"], dtype=np.int64)
    source_ids = np.asarray(nodes["source_id"], dtype=np.uint64).astype(np.int64)
    if not np.array_equal(ids, source_ids):
        raise RuntimeError("graph ids and neuron rows are not aligned")
    layer, names = classify_nodes(nodes["cell_type"], nodes["neurotransmitter"])
    if tuple(names) != LAYER_NAMES:
        raise RuntimeError("classification names changed")
    expected = [541, 4706, 7114, 2, 4064, 150273]
    actual = [int(np.sum(layer == i)) for i in range(6)]
    if actual != expected:
        raise RuntimeError(f"classification counts changed: {actual}")
    if len(ids) != 166700 or len(graph["post"]) != 25582938:
        raise RuntimeError("graph node/edge count changed")
    return graph, ids, layer, np.asarray(nodes["cell_type"], dtype=str), np.asarray(nodes["neurotransmitter"], dtype=str)


def make_brain(layer: np.ndarray) -> C0HybridBrain:
    rest = np.full(len(layer), -52.0, dtype=np.float32)
    rest[layer == 4] = -60.0
    brain = C0HybridBrain(GRAPH, gamma=1.0, dt=0.05,
                          graded_mask=np.isin(layer, [1, 2, 3]).astype(np.uint8), rest=rest)
    brain.kc_mask[:] = (layer == 4).astype(np.uint8)
    brain.modulation_mask.fill(0)
    return brain


def _layer_rows(brain, baseline_v, counts, layer):
    rows = {}
    for index, name in enumerate(LAYER_NAMES):
        mask = layer == index
        delta = brain.v[mask] - baseline_v[mask]
        rows[name] = {
            "count": int(mask.sum()),
            "voltage_delta_min": float(delta.min()), "voltage_delta_max": float(delta.max()),
            "voltage_min": float(brain.v[mask].min()), "voltage_max": float(brain.v[mask].max()),
            "measurement_spike_nodes": int(np.count_nonzero(counts[mask] > 0)),
            "measurement_spikes": int(counts[mask].sum()),
        }
    return rows


def reproduce_c0_diagnostic(graph, ids, layer) -> dict:
    brain = make_brain(layer)
    zero = np.zeros(len(ids), np.float32)
    drive = zero.copy(); drive[layer == 1] = 4.0
    brain.run(10.0, zero); baseline = brain.v.copy()
    before = [float(brain.release_integral[0]), float(brain.signed_delivery[0]),
              float(brain.absolute_delivery[0])]
    counts = brain.run(10.0, drive)
    after = [float(brain.release_integral[0]), float(brain.signed_delivery[0]),
             float(brain.absolute_delivery[0])]
    rows = _layer_rows(brain, baseline, counts, layer)
    degree = np.diff(graph["ptr"])
    source_layer = np.repeat(layer, degree)
    target_layer = layer[graph["post"]]
    matrix = np.bincount((source_layer.astype(np.int64) * 6 + target_layer).astype(np.int64),
                         minlength=36).reshape(6, 6).tolist()
    delta = [after[i] - before[i] for i in range(3)]
    changed = rows["graded_candidate_photoreceptor"]["voltage_delta_min"] != 0.0 or rows["graded_candidate_photoreceptor"]["voltage_delta_max"] != 0.0
    downstream = any(rows[name]["voltage_delta_min"] != 0.0 or rows[name]["voltage_delta_max"] != 0.0
                     for name in ("graded_candidate_lamina", "generic_spiking_declared"))
    checks = {
        "external_current_changes_photoreceptor_voltage": bool(changed),
        "photoreceptor_has_graded_release": bool(delta[0] > 0),
        "graded_source_has_outgoing_delivery": bool(delta[2] > 0),
        "downstream_voltage_responds": bool(downstream),
        "kc_spike_count_zero": bool(rows["adaptive_spiking_kc"]["measurement_spikes"] == 0),
    }
    return {
        "schema": "v2-c0-transfer-diagnostic-v1", "audit_only": True,
        "gamma": 1.0, "dt_ms": 0.05, "stimulus": "UNIFORM_4",
        "segments_ms": [10.0, 10.0], "source_count": int(np.sum(layer == 1)),
        "source_id_sha256": source_id_sha256(ids[layer == 1]),
        "edge_count_matrix_source_target": matrix,
        "before_integrals": {"release": before[0], "signed_delivery": before[1], "absolute_delivery": before[2]},
        "after_integrals": {"release": after[0], "signed_delivery": after[1], "absolute_delivery": after[2]},
        "delta_integrals": {"release": delta[0], "signed_delivery": delta[1], "absolute_delivery": delta[2]},
        "layers": rows,
        "conclusion": "frozen_stimulus_reaches_network_but_produces_no_KC_spike" if all(checks.values()) else "reproduction_changed",
        "checks": checks,
    }


STATE_FIELDS = ("v", "g_event", "g_graded", "refractory", "drive", "previous_drive",
                "spike_queue", "spike_queue_count", "graded_queue", "graded_drive",
                "active", "active_flag", "nactive", "last", "adaptation", "modulation")


def state_record(brain) -> dict:
    arrays = {name: np.asarray(getattr(brain, name)).copy() for name in STATE_FIELDS}
    h = hashlib.sha256()
    for name in STATE_FIELDS:
        h.update(name.encode()); h.update(np.ascontiguousarray(arrays[name]).tobytes())
    scalars = {
        "clock": int(brain.clock[0]), "total_spikes": int(brain.total_spikes),
        "release_integral": float(brain.release_integral[0]),
        "signed_delivery": float(brain.signed_delivery[0]),
        "absolute_delivery": float(brain.absolute_delivery[0]),
        "overflow": int(brain.overflow[0]),
        "weights_sha256": array_sha256(brain.weight),
    }
    h.update(json.dumps(scalars, sort_keys=True).encode())
    return {"arrays": arrays, "scalars": scalars, "sha256": h.hexdigest()}


def observer_replay(ids, layer) -> dict:
    brain = make_brain(layer)
    zero = np.zeros(len(ids), np.float32); drive = zero.copy(); drive[layer == 1] = 4.0
    brain.run(50.0, zero); baseline = brain.v.copy()
    n = len(ids); dt = 0.05; steps = int(round(150.0 / dt))
    first_voltage = np.full(n, np.nan, np.float32)
    first_input = np.full(n, np.nan, np.float32)
    first_spike = np.full(n, np.nan, np.float32)
    min_delta = np.zeros(n, np.float32); max_delta = np.zeros(n, np.float32)
    q1_integral = np.zeros(n, np.float64); cf_integral = np.zeros(n, np.float64)
    below_steps = np.zeros(n, np.int32); above_steps = np.zeros(n, np.int32)
    graded = np.isin(layer, [1, 2, 3])
    warmup_v = None
    for step in range(steps):
        counts = brain.run(dt, drive)
        elapsed = np.float32((step + 1) * dt)
        delta = brain.v - baseline
        voltage_now = np.abs(delta) > 1e-7
        input_now = (np.abs(brain.g_event) > 1e-7) | (np.abs(brain.g_graded) > 1e-7)
        spike_now = counts > 0
        first_voltage[np.isnan(first_voltage) & voltage_now] = elapsed
        first_input[np.isnan(first_input) & input_now] = elapsed
        first_spike[np.isnan(first_spike) & spike_now] = elapsed
        np.minimum(min_delta, delta, out=min_delta); np.maximum(max_delta, delta, out=max_delta)
        q1_integral[graded] += q1_positive_release(brain.v[graded], brain.rest[graded]) * dt / TAU_EMIT_MS
        cf_integral[graded] += signed_deviation(brain.v[graded], brain.rest[graded]) * dt / TAU_EMIT_MS
        below_steps[graded] += brain.v[graded] < brain.rest[graded] - 1e-7
        above_steps[graded] += brain.v[graded] > brain.rest[graded] + 1e-7
        if step + 1 == int(round(100.0 / dt)):
            warmup_v = brain.v.copy()
    measurement_v = brain.v.copy()
    for _ in range(200): brain.run(1.0, zero)
    washout_v = brain.v.copy()
    for _ in range(100): brain.run(1.0, zero)
    post_v = brain.v.copy()
    state = state_record(brain)
    arrays = {
        "baseline_v": baseline, "warmup_v": warmup_v, "measurement_v": measurement_v,
        "washout_v": washout_v, "post_v": post_v, "first_voltage_ms": first_voltage,
        "first_input_ms": first_input, "first_spike_ms": first_spike,
        "min_voltage_delta": min_delta, "max_voltage_delta": max_delta,
        "q1_release_integral_by_source": q1_integral,
        "candidate_c_integral_by_source": cf_integral,
        "below_rest_steps": below_steps, "above_rest_steps": above_steps,
    }
    h = hashlib.sha256()
    for name in sorted(arrays): h.update(name.encode()); h.update(np.ascontiguousarray(arrays[name]).tobytes())
    h.update(state["sha256"].encode())
    return {"arrays": arrays, "state": state, "audit_sha256": h.hexdigest()}


def reference_replay(ids, layer) -> dict:
    brain = make_brain(layer)
    zero = np.zeros(len(ids), np.float32); drive = zero.copy(); drive[layer == 1] = 4.0
    for duration, value in ((50.0, zero), (100.0, drive), (50.0, drive), (200.0, zero), (100.0, zero)):
        brain.run(duration, value)
    return state_record(brain)


def source_masks(layer, cell_type):
    return {
        "photoreceptor": layer == 1,
        "L1": cell_type == "L1", "L2": cell_type == "L2",
        "L3": cell_type == "L3", "L5": cell_type == "L5", "APL": cell_type == "APL",
    }


def _ids_summary(ids, mask):
    chosen = ids[np.asarray(mask, dtype=bool)]
    return {"count": int(len(chosen)), "source_id_sha256": source_id_sha256(chosen)}


def build_counterfactual(graph, ids, layer, cell_type, neurotransmitter, replay, raw):
    ptr, post, weight = graph["ptr"], graph["post"], graph["weight"]
    q = replay["arrays"]["candidate_c_integral_by_source"]
    q1 = replay["arrays"]["q1_release_integral_by_source"]
    masks = source_masks(layer, cell_type)
    mb07 = cell_type == "MBON07"; mb11 = cell_type == "MBON11"
    special_masks = {"generic_spiking": layer == 5, "KC": layer == 4, "MBON07": mb07, "MBON11": mb11}
    rows = {}
    ledgers = {}
    for name in SOURCE_CLASS_NAMES:
        mask = masks[name]; indices = np.flatnonzero(mask)
        ledger = counterfactual_delivery(ptr, post, weight, indices, q, layer, 6)
        ledgers[name] = ledger
        raw[f"cf_target_signed_{name}"] = ledger["target_signed"]
        raw[f"cf_target_absolute_{name}"] = ledger["target_absolute"]
        chunks = [weight[ptr[i]:ptr[i + 1]] for i in indices if ptr[i + 1] > ptr[i]]
        outgoing = np.concatenate(chunks) if chunks else np.empty(0, np.float32)
        target_rows = {}
        for i, target_name in enumerate(LAYER_NAMES):
            target_rows[target_name] = {
                "positive": float(ledger["positive_by_group"][i]),
                "negative": float(ledger["negative_by_group"][i]),
                "signed": float(ledger["signed_by_group"][i]),
                "absolute": float(ledger["absolute_by_group"][i]),
            }
        special = {}
        for target_name, target_mask in special_masks.items():
            net = ledger["target_signed"][target_mask]; absolute = ledger["target_absolute"][target_mask]
            special[target_name] = {
                "nonzero_absolute": _ids_summary(ids, target_mask & (ledger["target_absolute"] > 0)),
                "positive_net": _ids_summary(ids, target_mask & (ledger["target_signed"] > 0)),
                "negative_net": _ids_summary(ids, target_mask & (ledger["target_signed"] < 0)),
                "signed_sum": float(net.sum()), "absolute_sum": float(absolute.sum()),
            }
        qvals = q[mask]; q1vals = q1[mask]
        rows[name] = {
            "source_count": int(mask.sum()),
            "source_id_sha256": source_id_sha256(ids[mask]),
            "neurotransmitter_counts": {str(v): int(c) for v, c in zip(*np.unique(neurotransmitter[mask], return_counts=True))},
            "outgoing_edges": int(len(outgoing)), "positive_edges": int(np.sum(outgoing > 0)),
            "negative_edges": int(np.sum(outgoing < 0)), "zero_edges": int(np.sum(outgoing == 0)),
            "signed_weight_sum": float(outgoing.sum(dtype=np.float64)),
            "absolute_weight_sum": float(np.abs(outgoing).sum(dtype=np.float64)),
            "below_rest_source_count": int(np.sum(replay["arrays"]["below_rest_steps"][mask] > 0)),
            "above_rest_source_count": int(np.sum(replay["arrays"]["above_rest_steps"][mask] > 0)),
            "min_voltage_delta": float(replay["arrays"]["min_voltage_delta"][mask].min()),
            "max_voltage_delta": float(replay["arrays"]["max_voltage_delta"][mask].max()),
            "q1_nonzero_source_count": int(np.count_nonzero(q1vals)),
            "q1_integral_sum": float(q1vals.sum()),
            "candidate_negative_source_count": int(np.sum(qvals < 0)),
            "candidate_positive_source_count": int(np.sum(qvals > 0)),
            "candidate_integral_sum": float(qvals.sum()),
            "by_target_layer": target_rows, "special_targets": special,
        }
    lamina_indices = np.flatnonzero(layer == 2)
    lamina = counterfactual_delivery(ptr, post, weight, lamina_indices, q, layer, 6)
    raw["cf_target_signed_lamina_all"] = lamina["target_signed"]
    raw["cf_target_absolute_lamina_all"] = lamina["target_absolute"]
    return rows, ledgers, lamina


def hop_summary(hops, ids, layer, replay, q1, cf):
    rows = []
    for hop in list(range(7)) + [-1]:
        mask = hops == hop
        rows.append({
            "hop": ">6/unreached" if hop < 0 else hop,
            "node_count": int(mask.sum()),
            "source_id_sha256": source_id_sha256(ids[mask]),
            "layer_counts": {LAYER_NAMES[i]: int(np.sum(mask & (layer == i))) for i in range(6)},
            "input_reached": int(np.sum(mask & np.isfinite(replay["arrays"]["first_input_ms"]))),
            "voltage_changed": int(np.sum(mask & np.isfinite(replay["arrays"]["first_voltage_ms"]))),
            "spiking_nodes": int(np.sum(mask & np.isfinite(replay["arrays"]["first_spike_ms"]))),
            "q1_nonzero_graded_release": int(np.sum(mask & (q1 != 0))),
            "candidate_nonzero_graded_deviation": int(np.sum(mask & (cf != 0))),
        })
    return rows


def arrow_integrity_and_apl(layer, apl_mask):
    total_contacts = 0; edge_count = 0; offset = 0
    rows = {LAYER_NAMES[i]: {"edges": 0, "contacts": 0,
                             "positive_edges": 0, "negative_edges": 0,
                             "positive_contacts": 0, "negative_contacts": 0}
            for i in range(6)}
    graph = np.load(GRAPH, allow_pickle=False); ptr = graph["ptr"]; post = graph["post"]
    weight = graph["weight"]
    source_sign = np.zeros(len(layer), dtype=np.int8)
    for source_index in range(len(layer)):
        values = weight[ptr[source_index]:ptr[source_index + 1]]
        if len(values) == 0:
            continue
        signs = np.unique(np.sign(values).astype(np.int8))
        signs = signs[signs != 0]
        if len(signs) > 1:
            raise RuntimeError(f"source {source_index} has mixed outgoing signs")
        if len(signs) == 1:
            source_sign[source_index] = signs[0]
    with pa.memory_map(str(EDGES), "r") as source:
        reader = ipc.open_file(source)
        for b in range(reader.num_record_batches):
            batch = reader.get_batch(b)
            pre = batch.column(batch.schema.get_field_index("pre_index")).to_numpy(zero_copy_only=False).astype(np.int64)
            target = batch.column(batch.schema.get_field_index("post_index")).to_numpy(zero_copy_only=False).astype(np.int64)
            contacts = batch.column(batch.schema.get_field_index("synapse_count")).to_numpy(zero_copy_only=False).astype(np.int64)
            length = len(pre)
            if np.any(pre < 0) or np.any(pre >= len(layer)) or np.any(target < 0) or np.any(target >= len(layer)):
                raise RuntimeError("Arrow node index out of range")
            selected = apl_mask[target]
            for i in range(6):
                m = selected & (layer[pre] == i)
                rows[LAYER_NAMES[i]]["edges"] += int(m.sum())
                rows[LAYER_NAMES[i]]["contacts"] += int(contacts[m].sum())
                positive = m & (source_sign[pre] > 0)
                negative = m & (source_sign[pre] < 0)
                rows[LAYER_NAMES[i]]["positive_edges"] += int(positive.sum())
                rows[LAYER_NAMES[i]]["negative_edges"] += int(negative.sum())
                rows[LAYER_NAMES[i]]["positive_contacts"] += int(contacts[positive].sum())
                rows[LAYER_NAMES[i]]["negative_contacts"] += int(contacts[negative].sum())
            total_contacts += int(contacts.sum()); edge_count += length; offset += length
    if edge_count != 25582938 or total_contacts != 124177617:
        raise RuntimeError(f"edge/contact count changed: {edge_count}/{total_contacts}")
    return {"edge_count": edge_count, "contact_count": total_contacts, "apl_incoming_by_source_layer": rows}


def _prepare_output(out: Path):
    if out.exists():
        allowed = {"launch.txt", "runner.stdout.log"}
        existing = {p.name for p in out.iterdir()}
        if existing - allowed:
            raise RuntimeError(f"refusing to overwrite nonempty output: {out}")
    else:
        out.mkdir(parents=True)


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_failure(out: Path, verdict: str, error: Exception):
    out.mkdir(parents=True, exist_ok=True)
    value = {"schema": "v2-q2-d0-formal-v1", "verdict": verdict,
             "error_type": type(error).__name__, "error": str(error),
             "q2_authorized": False, "production_eligible": False}
    _write_json(out / "q2_d0_formal.json", value)
    return value


def run_smoke(out: Path):
    _prepare_output(out)
    ptr = np.array([0, 1, 2, 2, 2], np.int64)
    post = np.array([2, 3], np.int32); weight = np.array([-2.0, 3.0], np.float32)
    q = np.array([-0.5, -0.5, 0.0, 0.0]); groups = np.array([2, 2, 5, 5])
    l1 = counterfactual_delivery(ptr, post, weight, np.array([0]), q, groups, 6)
    l2 = counterfactual_delivery(ptr, post, weight, np.array([1]), q, groups, 6)
    hops = bfs_shortest_hops(ptr, post, np.array([0, 1]), 6)
    raw = out / "q2_d0_raw.npz"
    np.savez_compressed(raw, l1=l1["target_signed"], l2=l2["target_signed"], hops=hops)
    formal = {"schema": "v2-q2-d0-smoke-v1", "verdict": "Q2_D0_SMOKE_OK",
              "l1_target": float(l1["target_signed"][2]), "l2_target": float(l2["target_signed"][3])}
    _write_json(out / "q2_d0_formal.json", formal)
    (out / "q2_d0_report.md").write_text("# Q2-D0 smoke\n\nVerdict: `Q2_D0_SMOKE_OK`\n")
    manifest = {"schema": "v2-q2-d0-smoke-manifest-v1", "artifacts": {
        name: sha256_file(out / name) for name in ("q2_d0_raw.npz", "q2_d0_formal.json", "q2_d0_report.md")}}
    _write_json(out / "q2_d0_manifest.json", manifest)
    return formal


def run(out: Path):
    _prepare_output(out); started = time.time()
    pins = fixed_inputs(); start_hashes = hash_snapshot()
    graph, ids, layer, cell_type, neurotransmitter = load_inputs()
    expected = json.loads(C0_FORMAL.read_text())["transfer_diagnostic"]
    observed = reproduce_c0_diagnostic(graph, ids, layer)
    if observed != expected:
        raise ValueError("C0 v3 transfer diagnostic did not reproduce exactly")

    replay1 = observer_replay(ids, layer); replay2 = observer_replay(ids, layer)
    if replay1["audit_sha256"] != replay2["audit_sha256"]:
        raise RuntimeError("observer replay is not deterministic")
    reference = reference_replay(ids, layer)
    if replay1["state"]["sha256"] != reference["sha256"]:
        raise RuntimeError("observer/reference final state mismatch")

    raw = {"ids": ids, "layer": layer.astype(np.int8)}
    raw.update(replay1["arrays"])
    source_rows, ledgers, lamina = build_counterfactual(
        graph, ids, layer, cell_type, neurotransmitter, replay1, raw)
    hops = bfs_shortest_hops(graph["ptr"], graph["post"], np.flatnonzero(layer == 1), 6)
    raw["shortest_hop"] = hops
    q1 = replay1["arrays"]["q1_release_integral_by_source"]
    cf = replay1["arrays"]["candidate_c_integral_by_source"]
    hop_rows = hop_summary(hops, ids, layer, replay1, q1, cf)
    integrity = arrow_integrity_and_apl(layer, cell_type == "APL")

    generic = layer == 5; kc = layer == 4; lamina_mask = layer == 2
    flags = interpretation_flags(
        lamina_hyperpolarization_observed=bool(np.any(replay1["arrays"]["min_voltage_delta"][lamina_mask] < -1e-7)),
        lamina_q1_output_clamped_zero=bool(np.all(q1[lamina_mask] == 0.0)),
        counterfactual_lamina_reaches_spiking_downstream=bool(np.any(lamina["target_absolute"][generic | kc] > 0)),
        counterfactual_lamina_positive_net_reaches_spiking_downstream=bool(np.any(lamina["target_signed"][generic | kc] > 0)),
        counterfactual_lamina_reaches_kc_directly=bool(np.any(lamina["target_absolute"][kc] > 0)),
        generic_spiking_received_input=bool(np.any(np.isfinite(replay1["arrays"]["first_input_ms"][generic]))),
        generic_spiking_emitted_spikes=bool(np.any(np.isfinite(replay1["arrays"]["first_spike_ms"][generic]))),
    )
    apl = source_rows["APL"] | {
        "incoming": integrity["apl_incoming_by_source_layer"],
        "candidate_kc_signed_sum": float(ledgers["APL"]["target_signed"][kc].sum()),
        "candidate_kc_absolute_sum": float(ledgers["APL"]["target_absolute"][kc].sum()),
    }

    raw_path = out / "q2_d0_raw.npz"
    np.savez_compressed(raw_path, **raw)
    end_hashes = hash_snapshot()
    if start_hashes != end_hashes:
        raise RuntimeError("source/test/prereg/binary hash changed during run")
    formal = {
        "schema": "v2-q2-d0-formal-v1", "verdict": "Q2_D0_AUDIT_COMPLETE",
        "audit_only": True, "counterfactual": True, "production_eligible": False,
        "q2_authorized": False, "market_used": False, "learning_used": False,
        "graph": {"nodes": len(ids), "edges": len(graph["post"]), "contacts": integrity["contact_count"],
                  "weights_sha256_start_end_equal": True},
        "protocol": {"gamma": 1.0, "dt_ms": 0.05, "stimulus": "UNIFORM_4",
                     "segments_ms": [50, 100, 50, 200, 100],
                     "candidate_formula": "clip((v-rest)/7mV,-1,1)", "open_loop_only": True},
        "reproduction": {"exact_match": True, "expected_sha256": canonical_sha(expected),
                         "observed_sha256": canonical_sha(observed)},
        "determinism": {"replay_sha256_1": replay1["audit_sha256"],
                        "replay_sha256_2": replay2["audit_sha256"],
                        "observer_reference_state_equal": True,
                        "final_state_sha256": reference["sha256"]},
        "interpretation_flags": flags, "source_class_ledgers": source_rows,
        "hop_summary": hop_rows, "apl_audit": apl,
        "pins": pins, "hashes_start": start_hashes, "hashes_end": end_hashes,
        "missing_gates": [], "wall_time_seconds": time.time() - started,
    }
    formal_path = out / "q2_d0_formal.json"; _write_json(formal_path, formal)
    report = [
        "# V2-Q2-D0 dynamic-path audit", "", "Verdict: `Q2_D0_AUDIT_COMPLETE`", "",
        "This is an audit-only, open-loop counterfactual ledger. It does not implement or qualify Q2.", "",
        f"C0 diagnostic exact reproduction: `{formal['reproduction']['exact_match']}`.",
        f"Observer determinism/reference equivalence: `True`.", "", "## Interpretation flags", "",
    ]
    report += [f"- `{name}`: `{value}`" for name, value in flags.items()]
    report += ["", "## Source-class counterfactual summary", "",
               "| source | Q1 nonzero | candidate negative | candidate positive | generic abs delivery | KC abs delivery |",
               "|---|---:|---:|---:|---:|---:|"]
    for name in SOURCE_CLASS_NAMES:
        row = source_rows[name]
        report.append(f"| {name} | {row['q1_nonzero_source_count']} | {row['candidate_negative_source_count']} | {row['candidate_positive_source_count']} | {row['special_targets']['generic_spiking']['absolute_sum']:.9g} | {row['special_targets']['KC']['absolute_sum']:.9g} |")
    report += ["", "KC/MBON activity and trading meaning were not used to select a formula or parameter.", ""]
    report_path = out / "q2_d0_report.md"; report_path.write_text("\n".join(report))
    manifest = {
        "schema": "v2-q2-d0-manifest-v1", "pins": pins,
        "hashes_start": start_hashes, "hashes_end": end_hashes,
        "python": sys.version, "platform": platform.platform(),
        "artifacts": {"raw": sha256_file(raw_path), "formal": sha256_file(formal_path),
                      "report": sha256_file(report_path)},
        "missing_gates": [],
    }
    _write_json(out / "q2_d0_manifest.json", manifest)
    return formal


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        result = run_smoke(args.out) if args.smoke else run(args.out)
    except ValueError as exc:
        result = write_failure(args.out, "Q2_D0_REPRODUCTION_FAILURE", exc)
        print(json.dumps(result, indent=2, sort_keys=True)); raise
    except Exception as exc:
        result = write_failure(args.out, "Q2_D0_IMPLEMENTATION_FAILURE", exc)
        print(json.dumps(result, indent=2, sort_keys=True)); raise
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
