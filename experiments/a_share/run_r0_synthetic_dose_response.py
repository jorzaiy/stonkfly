"""Frozen R0 synthetic dose-response assay.

This runner is deliberately label-free.  It presents only the two synthetic
RGB stimulus families in the R0 preregistration to the complete retained
MaleCNS graph.  C uses the fresh observer-build checkpoint; B2 is an in-memory
APL->KC weight overlay and is never written back to the graph or checkpoint.
"""

import argparse
import base64
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from stonkfly.config import Settings
from stonkfly.neural.common import annotations, digest
from stonkfly.neural.v51_controller import V51Controller


ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "docs/20261007-r0-synthetic-dose-response-preregistered.md"
GRAPH = ROOT / "data/graph.npz"
HIST = ROOT / "runs/wc006_purged_fly_training/20261004T-balanced-reward-v2/brain-clean.npz"
S2I = ROOT / "runs/plasticity_diagnosis/S2i/20261005T-synthetic-calibration-grid-v1/calibration_grid.json"
S2K = ROOT / "runs/plasticity_diagnosis/S2k/20261006T-kc-recurrence-counterfactual-v1/s2k_counterfactual.json"

GRAPH_SHA256 = "346b8af85a11af13b8324e18669812c1924569e7d1adcb4e6f45cc461a2c344b"
HIST_SHA256 = "a92f547e720e7d9830ae519f26c0bacdf4f76c59761fb2be4adb6f34cac2a225"
S2I_SHA256 = "5f843d9ee6e3aae6d47102669b6b5189a304807b0763fae994f3f0472aea81ef"
S2K_SHA256 = "40363db184551381972adc5159888d79634ee8cbaa96219d80c85b7fd378f339"

HALF_SATURATION = 0.02
LAMINA_BIAS = 12.0
NEURAL_MS = 500.0
UNIFORM_LEVELS = (40, 42, 45, 48, 50, 52, 55, 58, 60, 62, 65, 68, 70, 72, 75, 78, 80)
ANCHORS = (20, 160, 255)
CONTRAST_AMPLITUDES = (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60)
ARMS = ("C", "B2")
RESPONSE_BANDS = ((0, 40, "sub_sparse"), (41, 812, "target_sparse"), (813, 4064, "dense"))


def sha_bytes(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def state_boundary_hash(brain):
    """Digest every persisted dynamic array at the post-stimulus boundary."""
    parts = []
    for name in ["weight", *brain.fields]:
        parts.append(name.encode() + b"=" + sha_bytes(getattr(brain, name)).encode())
    return hashlib.sha256(b";".join(parts)).hexdigest()


def all_finite(value):
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial.npz")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


def manifest_digest(manifest):
    probe = dict(manifest)
    probe["raw_npz_sha256"] = None
    probe["json_manifest_sha256"] = None
    return hashlib.sha256(json.dumps(probe, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def verify_manifest_pin(json_path, npz_path):
    manifest = json.loads(Path(json_path).read_text())
    with np.load(npz_path, allow_pickle=False) as arrays:
        pinned = str(np.asarray(arrays["json_manifest_sha256"]).item())
    return {
        "json_manifest_sha256": manifest.get("json_manifest_sha256") == manifest_digest(manifest),
        "npz_manifest_sha256": pinned == manifest.get("json_manifest_sha256"),
        "raw_npz_sha256": file_sha(npz_path) == manifest.get("raw_npz_sha256"),
    }


def uniform_frame(level):
    return np.ascontiguousarray(np.full((64, 64, 3), int(level), dtype=np.uint8))


def contrast_frame(amplitude):
    yy, xx = np.indices((64, 64))
    q = (yy // 8 + xx // 8) % 2
    values = np.where(q == 0, 60 - int(amplitude), 60 + int(amplitude)).astype(np.uint8)
    return np.ascontiguousarray(np.repeat(values[..., None], 3, axis=2))


def stimuli():
    out = []
    for level in UNIFORM_LEVELS:
        out.append({"family": "uniform", "level": int(level), "name": f"uniform_{level}", "frame": uniform_frame(level)})
    for level in ANCHORS:
        out.append({"family": "uniform_anchor", "level": int(level), "name": f"uniform_anchor_{level}", "frame": uniform_frame(level)})
    for amplitude in CONTRAST_AMPLITUDES:
        out.append({"family": "contrast", "level": int(amplitude), "name": f"contrast_{amplitude}", "frame": contrast_frame(amplitude)})
    return out


def frame_descriptor(item):
    frame = item["frame"]
    channel_means = frame.mean(axis=(0, 1)).astype(np.float64)
    return {
        "family": item["family"],
        "name": item["name"],
        "level": item["level"],
        "shape": list(frame.shape),
        "dtype": str(frame.dtype),
        "contiguous": bool(frame.flags.c_contiguous),
        "frame_sha256": sha_bytes(frame),
        "channel_means_uint8": channel_means.tolist(),
        "channel_mean": float(channel_means.mean()),
        "pixel_min": int(frame.min()),
        "pixel_max": int(frame.max()),
    }


def checkpoint_bridge(fresh, historical):
    fields = []
    all_equal = True
    with np.load(fresh, allow_pickle=False) as current, np.load(historical, allow_pickle=False) as old:
        current_keys = [key for key in current.files if key != "metadata"]
        old_keys = [key for key in old.files if key != "metadata"]
        all_equal = current_keys == old_keys
        for key in old_keys:
            if key not in current:
                fields.append({"field": key, "bitwise_equal": False, "reason": "missing"})
                all_equal = False
                continue
            left, right = np.asarray(current[key]), np.asarray(old[key])
            equal = left.shape == right.shape and left.dtype == right.dtype and np.array_equal(left, right, equal_nan=False)
            fields.append({"field": key, "bitwise_equal": bool(equal)})
            all_equal = all_equal and bool(equal)
    return {"all_persistent_arrays_bitwise_equal": bool(all_equal), "fields": fields}


def overlay_masks(brain):
    pre = np.searchsorted(brain.ptr, np.arange(len(brain.post)), side="right") - 1
    post = np.asarray(brain.post)
    kc = np.zeros(brain.n, dtype=bool)
    kc[np.asarray(brain.circuit["kc"], dtype=np.int64)] = True
    ann = annotations(brain.ids)
    apl = np.flatnonzero(ann.type.fillna("").astype(str).to_numpy() == "APL")
    apl_kc = np.isin(pre, apl) & kc[post]
    return kc, apl_kc, {
        "apl_body_ids": [int(brain.ids[index]) for index in apl],
        "apl_kc_edges": int(apl_kc.sum()),
        "baseline_positive": int((brain.weight[apl_kc] > 0).sum()),
        "baseline_negative": int((brain.weight[apl_kc] < 0).sum()),
        "baseline_zero": int((brain.weight[apl_kc] == 0).sum()),
    }


def verify_overlay_definition(brain, apl_kc, overlay_meta):
    expected = json.loads(S2K.read_text())["edge_overlay"]
    checks = {
        "apl_body_ids": overlay_meta["apl_body_ids"] == expected["apl_body_ids"],
        "apl_kc_edges": overlay_meta["apl_kc_edges"] == expected["apl_kc_edges"],
        "baseline_positive": overlay_meta["baseline_positive"] == expected["apl_kc_baseline_signs"]["positive"],
        "baseline_negative": overlay_meta["baseline_negative"] == expected["apl_kc_baseline_signs"]["negative"],
        "baseline_zero": overlay_meta["baseline_zero"] == expected["apl_kc_baseline_signs"]["zero"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"S2k B2 overlay definition mismatch: {checks}")
    return checks


def response_band(count):
    count = int(count)
    if count <= 10:
        fixed = "fixed-low-like"
    else:
        fixed = None
    for low, high, name in RESPONSE_BANDS:
        if low <= count <= high:
            return name, fixed
    raise ValueError(f"KC count outside authoritative range: {count}")


def set_hash(ids):
    return hashlib.sha256(",".join(map(str, sorted(map(int, ids)))).encode()).hexdigest()


def describe_counts(brain, counts, kc):
    counts = np.asarray(counts, dtype=np.int32)
    active = counts > 0
    kc_active = active[kc]
    kc_indices = np.asarray(brain.circuit["kc"], dtype=np.int64)
    active_kc_graph_indices = kc_indices[kc_active]
    active_kc_body_ids = np.asarray(brain.ids)[active_kc_graph_indices]
    band, fixed = response_band(int(kc_active.sum()))
    layers = {}
    for superclass in np.unique(brain.superclass):
        ix = np.flatnonzero(np.asarray(brain.superclass) == superclass)
        layers[str(superclass)] = {"active_cells": int(active[ix].sum()), "spike_count": int(counts[ix].sum())}
    mbons = {}
    for label, key in (("MBON07", "reward_mb"), ("MBON11", "aversive_mb")):
        ix = np.asarray(brain.circuit[key], dtype=np.int64)
        mbons[label] = {
            "cell_ids": [int(brain.ids[i]) for i in ix],
            "active_cells": int(active[ix].sum()),
            "spike_counts": [int(x) for x in counts[ix]],
        }
    packed = np.packbits(kc_active.astype(np.uint8), bitorder="little")
    bitset_popcount = int(np.unpackbits(packed, bitorder="little")[: len(kc)].sum())
    if bitset_popcount != int(kc_active.sum()):
        raise RuntimeError("KC bitset popcount mismatch")
    if not np.isfinite(brain.luminance).all() or not np.isfinite(brain.drive).all():
        raise RuntimeError("nonfinite sensory or graph drive")
    return {
        "active_kc_count": int(kc_active.sum()),
        "active_kc_fraction": float(kc_active.mean()),
        "response_band": band,
        "fixed_low_like": fixed,
        "kc_spike_count": int(counts[np.asarray(brain.circuit["kc"], dtype=np.int64)].sum()),
        # Frozen S2i digest_ids hashes authoritative KC graph indices.  Body
        # IDs are retained under a distinct name and never substitute for it.
        "kc_set_hash": set_hash(active_kc_graph_indices),
        "kc_body_id_set_hash": set_hash(active_kc_body_ids),
        "kc_bitset_sha256": sha_bytes(packed),
        "kc_bitset_popcount": bitset_popcount,
        "all_counts_sha256": sha_bytes(counts),
        "spike_sha256": sha_bytes(counts),
        "total_spikes": int(counts.sum()),
        "retinal_luminance": {
            "min": float(brain.luminance.min()), "mean": float(brain.luminance.mean()), "max": float(brain.luminance.max()), "sha256": sha_bytes(brain.luminance)
        },
        "retina_source_drive": {
            "min": float(brain.drive[brain.retina].min()), "mean": float(brain.drive[brain.retina].mean()), "max": float(brain.drive[brain.retina].max()), "sha256": sha_bytes(brain.drive[brain.retina])
        },
        "whole_graph_drive": {
            "min": float(brain.drive.min()), "mean": float(brain.drive.mean()), "max": float(brain.drive.max()), "sha256": sha_bytes(brain.drive)
        },
        "layers": layers,
        "mbons": mbons,
        "kc_graph_indices": [int(x) for x in active_kc_graph_indices],
        "kc_body_ids": [int(x) for x in active_kc_body_ids],
        "kc_bitset": packed,
        "counts": counts.copy(),
    }


def run_one(controller, clean, base_weight, apl_kc, arm, item, protocol, order, sequence_index, repeat_index):
    controller.restore(clean)
    expected_weight = base_weight.copy()
    if arm == "B2":
        expected_weight[apl_kc] *= 2.0
    controller.brain.weight[:] = expected_weight
    before_hash = digest(controller.brain.weight)
    counts, wall = controller.brain.rgb_step(
        item["frame"], NEURAL_MS, learning=False,
        retina_half_saturation=HALF_SATURATION, lamina_bias=LAMINA_BIAS,
    )
    b = controller.brain
    result = describe_counts(b, counts, np.asarray(b.circuit["kc"], dtype=np.int64))
    after_hash = digest(b.weight)
    if after_hash != before_hash:
        raise RuntimeError(f"weight changed while learning=false: {arm}/{protocol}/{item['name']}")
    result.update({
        "arm": arm, "protocol": protocol, "order": order, "sequence_index": int(sequence_index), "repeat_index": repeat_index,
        "stimulus": frame_descriptor(item), "wall_seconds": float(wall), "learning": False,
        "counterfactual": arm == "B2", "production_eligible": arm == "C", "weight_sha256_before": before_hash, "weight_sha256_after": after_hash,
        "state_boundary_sha256": state_boundary_hash(b),
    })
    if not all_finite(result):
        raise RuntimeError(f"nonfinite R0 record: {arm}/{protocol}/{item['name']}")
    return result


def independent_plan(items):
    main = [x for x in items if x["family"] == "uniform"]
    anchor = [x for x in items if x["family"] == "uniform_anchor"]
    contrast = [x for x in items if x["family"] == "contrast"]
    plan = []
    for family_items in (main, anchor, contrast):
        for i, x in enumerate(family_items):
            for repeat in (1, 2):
                plan.append((x, "independent", f"repeat_{repeat}", i, repeat))
    return plan


def continuous_plan(items):
    main = [x for x in items if x["family"] == "uniform"]
    contrast = [x for x in items if x["family"] == "contrast"]
    plan = []
    for order, sequence in (("asc", main), ("desc", list(reversed(main))), ("asc", contrast), ("desc", list(reversed(contrast)))):
        family = sequence[0]["family"]
        for i, x in enumerate(sequence):
            plan.append((x, "continuous", f"{family}_{order}", i, None))
    return plan


def run_sequence(controller, clean, base_weight, apl_kc, arm, sequence, protocol, order):
    controller.restore(clean)
    results = []
    for i, item in enumerate(sequence):
        expected_weight = base_weight.copy()
        if arm == "B2":
            expected_weight[apl_kc] *= 2.0
        controller.brain.weight[:] = expected_weight
        counts, wall = controller.brain.rgb_step(item["frame"], NEURAL_MS, learning=False, retina_half_saturation=HALF_SATURATION, lamina_bias=LAMINA_BIAS)
        result = describe_counts(controller.brain, counts, np.asarray(controller.brain.circuit["kc"], dtype=np.int64))
        before_hash = digest(expected_weight)
        after_hash = digest(controller.brain.weight)
        if after_hash != before_hash:
            raise RuntimeError(f"weight changed in continuous sequence: {arm}/{order}/{item['name']}")
        result.update({"arm": arm, "protocol": protocol, "order": order, "sequence_index": i, "repeat_index": None,
                       "stimulus": frame_descriptor(item), "wall_seconds": float(wall), "learning": False,
                       "counterfactual": arm == "B2", "production_eligible": arm == "C", "weight_sha256_before": before_hash, "weight_sha256_after": after_hash})
        result["state_boundary_sha256"] = state_boundary_hash(controller.brain)
        if not all_finite(result):
            raise RuntimeError(f"nonfinite R0 record: {arm}/{order}/{item['name']}")
        results.append(result)
    return results


def compact(result):
    out = dict(result)
    out.pop("counts", None)
    out.pop("kc_bitset", None)
    out.pop("kc_graph_indices", None)
    out.pop("kc_body_ids", None)
    return out


def compare_independent(records):
    groups = {}
    for row in records:
        if row["protocol"] == "independent":
            key = (row["arm"], row["stimulus"]["name"])
            groups.setdefault(key, []).append(row)
    checks = []
    for key, rows in groups.items():
        if len(rows) != 2:
            raise RuntimeError(f"independent repeat count mismatch: {key}={len(rows)}")
        checks.append({"key": list(key), "counts_equal": rows[0]["all_counts_sha256"] == rows[1]["all_counts_sha256"],
                       "kc_set_equal": rows[0]["kc_set_hash"] == rows[1]["kc_set_hash"],
                       "spike_equal": rows[0]["spike_sha256"] == rows[1]["spike_sha256"],
                       "kc_bitset_equal": rows[0]["kc_bitset_sha256"] == rows[1]["kc_bitset_sha256"],
                       "state_boundary_equal": rows[0]["state_boundary_sha256"] == rows[1]["state_boundary_sha256"],
                       "weight_equal": rows[0]["weight_sha256_before"] == rows[1]["weight_sha256_before"]})
    return {"checks": checks, "all_equal": bool(checks) and all(all(x.values()) for x in checks)}


def grid_summary(records, family):
    clean = [r for r in records if r["protocol"] == "independent" and r["stimulus"]["family"] == family and r["repeat_index"] == 1]
    clean.sort(key=lambda r: r["stimulus"]["level"])
    counts = [r["active_kc_count"] for r in clean]
    bands = [r["response_band"] for r in clean]
    target = sum(41 <= x <= 812 for x in counts)
    changes = sum(a != b for a, b in zip(bands, bands[1:]))
    dense_to_sub = any(a == "dense" and b == "sub_sparse" for a, b in zip(bands, bands[1:]))
    dense_entries = sum(b == "dense" and (i == 0 or bands[i - 1] != "dense") for i, b in enumerate(bands))
    repeated_dense_entry = dense_entries > 1
    nonmonotonic = dense_to_sub or repeated_dense_entry or changes > 2
    return {"family": family, "levels": [r["stimulus"]["level"] for r in clean], "active_kc_counts": counts, "response_bands": bands,
            "target_sparse_level_count": int(target), "hard_step": target <= 1, "gradual": target >= 3, "borderline": target == 2,
            "response_band_transition_count": int(changes), "dense_entry_count": int(dense_entries), "nonmonotonic": bool(nonmonotonic),
            "first_target_sparse": next((r["stimulus"]["name"] for r in clean if 41 <= r["active_kc_count"] <= 812), None),
            "first_dense": next((r["stimulus"]["name"] for r in clean if r["active_kc_count"] >= 813), None)}


def hysteresis_summary(records, family):
    rows = [r for r in records if r["protocol"] == "continuous" and r["stimulus"]["family"] == family]
    asc = {r["stimulus"]["level"]: r for r in rows if r["order"].endswith("_asc")}
    desc = {r["stimulus"]["level"]: r for r in rows if r["order"].endswith("_desc")}
    levels = sorted(set(asc) & set(desc))
    comparisons = []
    for level in levels:
        a, d = asc[level], desc[level]
        comparisons.append({"level": level, "active_kc_count_difference": abs(a["active_kc_count"] - d["active_kc_count"]),
                            "response_band_changed": a["response_band"] != d["response_band"], "kc_jaccard": jaccard(a["kc_graph_indices"], d["kc_graph_indices"]),
                            "set_hash_equal": a["kc_set_hash"] == d["kc_set_hash"]})
    return {"family": family, "comparisons": comparisons,
            "category_hysteresis_count": sum(x["response_band_changed"] for x in comparisons),
            "set_hysteresis_count": sum(not x["set_hash_equal"] for x in comparisons)}


def jaccard(a, b):
    left, right = set(map(int, a)), set(map(int, b))
    return float(len(left & right) / max(1, len(left | right)))


def run(output, smoke=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for path, expected in ((PREREG, None), (GRAPH, GRAPH_SHA256), (HIST, HIST_SHA256), (S2I, S2I_SHA256), (S2K, S2K_SHA256)):
        if not path.exists():
            raise RuntimeError(f"missing provenance input: {path}")
        if expected is not None and file_sha(path) != expected:
            raise RuntimeError(f"provenance SHA256 mismatch: {path}")
    controller = V51Controller(Settings(neural_ms=NEURAL_MS, pulse_ms=200, learning=False))
    b = controller.brain
    kc, apl_kc, overlay_meta = overlay_masks(b)
    if int(kc.sum()) != 4064:
        raise RuntimeError(f"authoritative KC count mismatch: {int(kc.sum())}")
    overlay_checks = verify_overlay_definition(b, apl_kc, overlay_meta)
    mbon_identity = {
        label: [int(b.ids[i]) for i in np.asarray(b.circuit[key], dtype=np.int64)]
        for label, key in (("MBON07", "reward_mb"), ("MBON11", "aversive_mb"))
    }
    if not mbon_identity["MBON07"] or not mbon_identity["MBON11"]:
        raise RuntimeError("MBON07/MBON11 identity set is empty")
    clean = output / "brain-clean-observer-build.npz"
    controller.save(clean)
    bridge = checkpoint_bridge(clean, HIST)
    if not bridge["all_persistent_arrays_bitwise_equal"]:
        raise RuntimeError("fresh observer-build checkpoint persistent-array bridge failed")
    base_weight = b.weight.copy()
    items = stimuli()
    if smoke:
        # Smoke preserves the same code paths but runs one arm/one repeat over
        # representative levels; it cannot be named or consumed as formal R0.
        items = [x for x in items if x["name"] in {"uniform_40", "uniform_80", "uniform_anchor_20", "contrast_0", "contrast_60"}]
        arms = ARMS
        independent = [(x, "independent", "repeat_1", i, 1) for i, x in enumerate(items)]
        plans = {"independent": independent}
        sequences = []
    else:
        arms = ARMS
        plans = {"independent": independent_plan(items)}
        sequences = [([x for x in items if x["family"] == family], f"{family}_asc") for family in ("uniform", "contrast")]
        sequences += [(list(reversed([x for x in items if x["family"] == family])), f"{family}_desc") for family in ("uniform", "contrast")]
    records = []
    arrays_counts, arrays_bitsets = [], []
    for arm in arms:
        for item, protocol, order, index, repeat in plans["independent"]:
            row = run_one(controller, clean, base_weight, apl_kc, arm, item, protocol, order, index, repeat)
            arrays_counts.append(row.pop("counts")); arrays_bitsets.append(row.pop("kc_bitset")); records.append(row)
        for sequence, order in sequences:
            for row in run_sequence(controller, clean, base_weight, apl_kc, arm, sequence, "continuous", order):
                arrays_counts.append(row.pop("counts")); arrays_bitsets.append(row.pop("kc_bitset")); records.append(row)
    expected_count = 252 if not smoke else 10
    if len(records) != expected_count * (1 if smoke else 1):
        raise RuntimeError(f"unexpected record count: {len(records)}")
    if not smoke:
        per_arm = {arm: sum(r["arm"] == arm for r in records) for arm in ARMS}
        if per_arm != {"C": 126, "B2": 126}:
            raise RuntimeError(f"per-arm record count mismatch: {per_arm}")
        repeats = compare_independent(records)
        if not repeats["all_equal"]:
            raise RuntimeError("independent clean-run determinism gate failed")
    else:
        repeats = {"checks": [], "all_equal": True}
    summaries = {} if smoke else {arm: {family: grid_summary([r for r in records if r["arm"] == arm], family) for family in ("uniform", "contrast")} for arm in ARMS}
    hysteresis = {} if smoke else {arm: {family: hysteresis_summary([r for r in records if r["arm"] == arm], family) for family in ("uniform", "contrast")} for arm in ARMS}
    raw = output / ("r0_smoke_raw.npz" if smoke else "r0_formal_raw.npz")
    atomic_npz(raw, counts=np.asarray(arrays_counts, dtype=np.int32), kc_bitsets=np.asarray(arrays_bitsets, dtype=np.uint8), json_manifest_sha256=np.asarray("pending"))
    # Exact C production-anchor bridge to the frozen S2i evidence.
    anchor_checks = []
    if not smoke:
        anchor_refs = {}
        for ref in json.loads(S2I.read_text())["records"]:
            if ref.get("retina_half_saturation") == HALF_SATURATION and ref.get("lamina_bias") == LAMINA_BIAS and ref.get("name") in {f"uniform_{x}" for x in ANCHORS + (40, 80)}:
                anchor_refs[ref["name"]] = ref
        for level in (20, 40, 80, 160, 255):
            name = f"uniform_{level}"
            rows = [r for r in records if r["arm"] == "C" and r["protocol"] == "independent" and r["repeat_index"] == 1 and r["stimulus"]["name"] in {name, f"uniform_anchor_{level}"}]
            row = rows[0] if rows else None
            ref = anchor_refs.get(name)
            checks = {"level": level, "present": bool(row and ref)}
            if row and ref:
                checks.update({"frame_hash": row["stimulus"]["frame_sha256"] == ref["frame_sha256"],
                               "active_kc_count": row["active_kc_count"] == ref["kc_count"],
                               "kc_set_hash": row["kc_set_hash"] == ref["kc_set_hash"],
                               "spike_hash": row["spike_sha256"] == ref["spike_sha256"]})
            checks["all"] = all(value for key, value in checks.items() if key != "level")
            anchor_checks.append(checks)
        if not all(check["all"] for check in anchor_checks):
            raise RuntimeError(f"C S2i anchor bridge failed: {anchor_checks}")
    result = {
        "observer": "R0", "formal": not smoke, "success": True, "diagnostic_only": True, "counterfactual": True, "production_eligible": False,
        "preregistered": True, "learning": False,
        "arms": list(ARMS), "stimulus_count": len(records), "record_count_expected": expected_count, "records": [compact(r) for r in records],
        "response_bands": {"sub_sparse": "0..40", "target_sparse": "41..812", "dense": "813..4064", "fixed_low_like": "0..10"},
        "grid_summaries": summaries, "hysteresis": hysteresis, "repeat_determinism": repeats,
        "overlay": overlay_meta | {"checks": overlay_checks, "b2_multiplier": 2.0, "counterfactual": True, "production_eligible": False},
        "identity_gate": {"MBON07": mbon_identity["MBON07"], "MBON11": mbon_identity["MBON11"], "nonempty": True},
        "s2i_anchor_gate": {"checks": anchor_checks, "all": True if smoke else all(check["all"] for check in anchor_checks)},
        "checkpoint_bridge": bridge, "raw_npz": str(raw), "raw_npz_sha256": None, "json_manifest_sha256": None,
        "provenance": {"pre_registration": str(PREREG), "pre_registration_sha256": file_sha(PREREG), "graph": str(GRAPH), "graph_sha256": file_sha(GRAPH),
                       "historical_checkpoint": str(HIST), "historical_checkpoint_sha256": file_sha(HIST), "fresh_checkpoint": str(clean), "fresh_checkpoint_sha256": file_sha(clean),
                       "s2i": str(S2I), "s2i_sha256": file_sha(S2I), "s2k": str(S2K), "s2k_sha256": file_sha(S2K),
                       "kernel_source_sha256": b.build["source_sha256"], "kernel_binary_sha256": b.build["binary_sha256"],
                       "script_sha256": file_sha(Path(__file__)), "configuration_sha256": hashlib.sha256(json.dumps(b.configuration_signature(), sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()},
    }
    result["json_manifest_sha256"] = manifest_digest(result)
    atomic_npz(raw, counts=np.asarray(arrays_counts, dtype=np.int32), kc_bitsets=np.asarray(arrays_bitsets, dtype=np.uint8), json_manifest_sha256=np.asarray(result["json_manifest_sha256"]))
    result["raw_npz_sha256"] = file_sha(raw)
    json_name = "r0_smoke.json" if smoke else "r0_formal.json"
    candidate = output / (json_name + ".partial")
    atomic_json(candidate, result)
    pin = verify_manifest_pin(candidate, raw)
    if not all(pin.values()):
        raise RuntimeError(f"manifest pin self-check failed: {pin}")
    candidate.replace(output / json_name)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.output, smoke=args.smoke)
    except Exception as exc:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        for name in ("r0_formal.json", "r0_smoke.json", "r0_formal_raw.npz", "r0_smoke_raw.npz"):
            candidate = output / name
            if candidate.exists():
                candidate.unlink()
        failure = {"observer": "R0", "formal": not args.smoke, "success": False, "diagnostic_only": True,
                   "counterfactual": True, "production_eligible": False, "error": f"{type(exc).__name__}: {exc}"}
        atomic_json(output / ("r0_smoke_failure.json" if args.smoke else "r0_formal_failure.json"), failure)
        print(json.dumps({"formal": not args.smoke, "path": str(output), "success": False, "error": str(exc)}))
        raise
    print(json.dumps({"formal": result["formal"], "records": result["stimulus_count"], "path": str(args.output), "success": result["success"]}))


if __name__ == "__main__":
    main()
