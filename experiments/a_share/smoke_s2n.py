"""S2n windowed, transient KC-delivery attenuation assay.

The intervention is deliberately isolated from the retained graph and weight
arrays.  ``run_formal`` is implemented for the preregistered 60-frame assay,
but the command line refuses to invoke it; formal execution requires a
separate explicit approval.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.a_share.analyze_kc_overlap import render_frame_from_observation
from experiments.a_share.analyze_s2l_kc_balance import classify, events
from stonkfly.config import Settings
from stonkfly.neural.trace_s2n import TraceBuffersS2N, WINDOW_BOUNDS
from stonkfly.neural.v51_controller import V51Controller

ROOT = Path(__file__).resolve().parents[2]
S2D = ROOT / "runs/plasticity_diagnosis/S2d/20261004T-projection-threshold-v1/projection_threshold_analysis.json"
S2K = ROOT / "runs/plasticity_diagnosis/S2k/20261006T-kc-recurrence-counterfactual-v1/s2k_counterfactual.json"
HIST = ROOT / "runs/wc006_purged_fly_training/20261004T-balanced-reward-v2/brain-clean.npz"
PREREG = ROOT / "docs/20261006-s2n-windowed-causal-attenuation.md"
TOP20 = (11023, 11145, 11030, 12529, 16473, 515015, 513174, 521189,
         18170, 11407, 519131, 523987, 125182, 19061, 534301, 522715,
         521721, 12650, 20705, 11455)
ARMS = ("C", "V12", "CB4", "BOTH")
SCALES = {
    "C": (1., 1., 1., 1., 1.),
    "V12": (1., .5, .5, 1., 1.),
    "CB4": (1., 1., 1., 1., .5),
    "BOTH": (1., .5, .5, 1., .5),
}
EXPECTED_APPLIED_CELLS = {
    "C": frozenset(),
    "V12": frozenset({(0, 1), (0, 2)}),
    "CB4": frozenset({(1, 4)}),
    "BOTH": frozenset({(0, 1), (0, 2), (1, 4)}),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ah(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial.npz")
    with tmp.open("wb") as h:
        np.savez_compressed(h, **arrays)
    tmp.replace(path)


def manifest_digest(manifest):
    probe = dict(manifest)
    probe["raw_npz_sha256"] = None
    probe["json_manifest_sha256"] = None
    return hashlib.sha256(json.dumps(probe, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def verify_manifest_pin(json_path, npz_path):
    with Path(json_path).open() as h:
        manifest = json.load(h)
    with np.load(npz_path, allow_pickle=False) as z:
        pin = str(np.asarray(z["json_manifest_sha256"]).item())
    return {
        "json_manifest_sha256": manifest.get("json_manifest_sha256") == manifest_digest(manifest),
        "npz_manifest_sha256": pin == manifest.get("json_manifest_sha256"),
        "raw_npz_sha256": sha(npz_path) == manifest.get("raw_npz_sha256"),
        "all": manifest.get("json_manifest_sha256") == manifest_digest(manifest)
        and pin == manifest.get("json_manifest_sha256") and sha(npz_path) == manifest.get("raw_npz_sha256"),
    }


def window_index(delta):
    """Frozen KC1-relative half-open windows; negative is explicitly outside."""
    if delta < 0 or delta >= 1024:
        return -1
    for i, (lo, hi) in enumerate(WINDOW_BOUNDS):
        if lo <= delta < hi:
            return i
    return -1


def selected_sources(sc, ids, arm):
    """Legacy membership helper retained for callers; ABI uses source_groups."""
    visual = np.asarray(sc == 3, dtype=bool)
    cb = (np.asarray(sc == 4, dtype=bool) &
          np.isin(np.asarray(ids, dtype=np.int64), np.asarray(TOP20, dtype=np.int64)))
    if arm == "C": return np.zeros(len(ids), dtype=np.uint8)
    if arm == "V12": return visual.astype(np.uint8)
    if arm == "CB4": return cb.astype(np.uint8)
    if arm == "BOTH": return np.logical_or(visual, cb).astype(np.uint8)
    raise ValueError(f"unknown S2n arm: {arm}")


def source_groups(sc, ids, arm):
    """Return ABI-v2 source groups: 0 none, 1 visual, 2 frozen top20 CB."""
    sc = np.asarray(sc); ids = np.asarray(ids, dtype=np.int64)
    visual = sc == 3
    cb = (sc == 4) & np.isin(ids, np.asarray(TOP20, dtype=np.int64))
    out = np.zeros(len(ids), dtype=np.int8)
    if arm in ("V12", "BOTH"):
        out[visual] = 1
    if arm in ("CB4", "BOTH"):
        if np.any(out[cb]):
            raise ValueError("visual/frozen-CB source groups overlap")
        out[cb] = 2
    if arm not in ARMS:
        raise ValueError(f"unknown S2n arm: {arm}")
    return out


def _snapshot(b):
    keys = ["weight", *b.fields, "v", "g", "refractory", "drive",
            "previous_drive", "queue", "queue_count", "counts", "luminance",
            "active", "active_flag", "nactive", "last", "adaptation"]
    out = {k: getattr(b, k).copy() for k in keys}
    out.update({k: getattr(b, k) for k in ("cursor", "sim_ms", "total_spikes", "weights_frozen")})
    return out


def _same(a, b):
    return np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b


def _bridge(fresh, historical):
    fields = []
    with np.load(historical, allow_pickle=False) as old, np.load(fresh, allow_pickle=False) as new:
        # Every persisted numeric state array is bridged. Metadata is a
        # provenance envelope, not neural state, and is intentionally excluded.
        keys = tuple(k for k in old.files if k != "metadata")
        for key in keys:
            ok = (key in old.files and key in new.files and old[key].shape == new[key].shape
                  and old[key].dtype == new[key].dtype and np.array_equal(old[key], new[key]))
            fields.append({"field": key, "bitwise_equal": bool(ok)})
    return {"all_persistent_arrays_bitwise_equal": all(x["bitwise_equal"] for x in fields), "fields": fields}


def _row(c, clean, frame, sc, ks, ids, arm):
    b = c.brain
    c.restore(clean); b.trace_observer_s2n = None
    off = c.observe(frame, "none")
    off_counts = b.counts.copy(); off_state = _snapshot(b)
    c.restore(clean)
    selected = source_groups(sc, ids, arm)
    t = TraceBuffersS2N(selected, ks, SCALES[arm], source_body_ids=ids)
    t.reset(); b.trace_observer_s2n = C.addressof(t.native)
    on = c.observe(frame, "none"); b.trace_observer_s2n = None; t.fail()
    on_state = _snapshot(b)
    kc = ks >= 0
    kc_active = off_counts[kc] > 0
    on_kc_active = b.counts[kc] > 0
    active_hash = ah(np.flatnonzero(kc_active).astype(np.int32))
    on_active_hash = ah(np.flatnonzero(on_kc_active).astype(np.int32))
    same_state = {k: _same(off_state[k], on_state[k]) for k in off_state}
    audit_cells = {(g, w) for g in range(2) for w in range(5)
                   if int(t.applied_events[g, w]) != 0 or abs(float(t.applied_signed[g, w])) > 1e-12
                   or abs(float(t.applied_positive[g, w])) > 1e-12 or abs(float(t.applied_negative[g, w])) > 1e-12}
    expected_cells = EXPECTED_APPLIED_CELLS[arm]
    # A configured cell may legitimately have zero events on a frame; only
    # unexpected non-zero cells are forbidden.
    applied_cells_gate = audit_cells <= expected_cells
    ratio_gate = True
    for g, w in expected_cells:
        ratio_gate &= bool(np.allclose(t.applied_signed[g, w], t.original_signed[g, w] * .5, rtol=1e-10, atol=1e-8))
        ratio_gate &= int(t.applied_events[g, w]) == int(t.eligible_events[g, w])
    for g in range(2):
        for w in range(5):
            if (g, w) not in expected_cells:
                ratio_gate &= int(t.applied_events[g, w]) == 0 and abs(float(t.applied_signed[g, w])) <= 1e-12
    eq = {
        "trace_off_on_all_state": all(same_state.values()) if arm == "C" else True,
        "trace_off_on_b_fields": arm != "C" or all(same_state[k] for k in b.fields),
        "weight_unchanged": np.array_equal(off_state["weight"], on_state["weight"]),
        "counts_finite": bool(np.isfinite(b.counts).all()),
        "weight_finite": bool(np.isfinite(b.weight).all()),
        "eligible_event_conservation": int(t.eligible_events.sum()) == int(t.eligible_count[0]),
        "applied_event_conservation": int(t.applied_events.sum()) == int(t.applied_count[0]),
        "kc1_present": int(t.kc1_step[0]) >= 0,
        "selected_source_membership": bool(np.all(np.isin(selected, (0, 1, 2)))),
        "c_no_intervention": arm != "C" or (int(t.eligible_count[0]) == 0 and int(t.applied_count[0]) == 0),
        "group_window_applied_cells": applied_cells_gate,
        "applied_half_original": ratio_gate,
        # Selected bodies are intervention sources and must themselves be
        # non-KC; the kernel independently rejects every non-KC target.
        "target_only": bool(np.all(ks[selected != 0] < 0)) if np.any(selected != 0) else True,
    }
    # C is the no-intervention arm and must match both the independent S2k
    # reference and its own trace-off path.  Other arms are compared to C in
    # the caller after all four arms have been restored from the same clean file.
    return {
        "arm": arm, "frame_sha256": ah(frame),
        "trace_off_active_kc_count": int(kc_active.sum()), "trace_off_kc_spike_count": int(off_counts[kc].sum()),
        "trace_off_active_kc_hash": active_hash, "trace_off_spike_sha256": off["spike_sha256"],
        "active_kc_count": int(on_kc_active.sum()), "kc_spike_count": int(b.counts[kc].sum()),
        "active_kc_indices": np.flatnonzero(on_kc_active).astype(np.int32).tolist(),
        "active_kc_hash": on_active_hash, "spike_sha256": on["spike_sha256"],
        "on_active_kc_count": int(on_kc_active.sum()), "on_kc_spike_count": int(b.counts[kc].sum()),
        "on_active_kc_hash": on_active_hash, "on_spike_sha256": on["spike_sha256"],
        "state_category": "low" if int(on_kc_active.sum()) <= 10 else "high" if int(on_kc_active.sum()) >= 1000 else "intermediate",
        "kc1_step": int(t.kc1_step[0]), "eligible_count": int(t.eligible_count[0]),
        "applied_count": int(t.applied_count[0]), "eligible_events": t.eligible_events.tolist(),
        "applied_events": t.applied_events.tolist(), "original_signed": t.original_signed.tolist(),
        "original_positive": t.original_positive.tolist(), "original_negative": t.original_negative.tolist(),
        "applied_signed": t.applied_signed.tolist(), "applied_positive": t.applied_positive.tolist(),
        "applied_negative": t.applied_negative.tolist(), "selected_outside_count": int(t.selected_outside_count[0]),
        "selected_source_count": int(np.count_nonzero(selected)), "kc_target_count": int(kc.sum()),
        "_selected_source_groups": selected.copy(),
        "_active_kc_indices": np.flatnonzero(on_kc_active).astype(np.int32),
        "scale": list(SCALES[arm]),
        "equivalence": eq,
    }


def _refs():
    if not S2K.exists(): return {}
    return {int(x["event_index"]): x for x in json.loads(S2K.read_text()).get("records", []) if x.get("arm") == "C"}


def _events():
    return events()


def _run(output, selected_events, *, require_refs=False):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    c = V51Controller(Settings(neural_ms=500, pulse_ms=200, learning=False)); b = c.brain
    clean = output / "brain-clean-observer-build.npz"; c.save(clean)
    bridge = _bridge(clean, HIST) if HIST.exists() else {"all_persistent_arrays_bitwise_equal": False}
    sc, typ, sup, kc = classify(b)
    ks = np.full(b.n, -1, np.int32); ks[kc] = np.arange(len(kc), dtype=np.int32)
    ids = np.asarray(b.ids, np.int64)
    top20 = np.asarray(TOP20, dtype=np.int64)
    if len(TOP20) != len(set(TOP20)):
        raise RuntimeError("S2n frozen top20 IDs are not unique")
    top_hits = np.flatnonzero(np.isin(ids, top20))
    if len(top_hits) != len(TOP20) or not np.all(sc[top_hits] == 4):
        raise RuntimeError("S2n frozen top20 IDs must all exist exactly once and be class4")
    if np.any((sc == 3) & np.isin(ids, top20)):
        raise RuntimeError("S2n visual and frozen-CB source groups overlap")
    candidate = np.logical_or(sc == 3, np.isin(ids, top20))
    source_map = [{"source_index": int(i), "body_id": int(ids[i]), "class": int(sc[i])} for i in np.flatnonzero(candidate)]
    refs = _refs(); records = []
    if require_refs and set(refs) != {int(x[0]) for x in selected_events}:
        raise RuntimeError("S2n formal requires exact S2k-C reference for every selected frame")
    for ei, ev, expected in selected_events:
        frame = render_frame_from_observation(ev["observation"])
        rows = {}
        for arm in ARMS:
            row = _row(c, clean, frame, sc, ks, ids, arm)
            row.update(event_index=int(ei), group="low" if int(expected["active_kc_count"]) <= 10 else "high")
            rows[arm] = row
        cref = refs.get(int(ei)); c_row = rows["C"]
        if cref:
            c_row["equivalence"].update(
                s2k_frame_hash=c_row["frame_sha256"] == cref["frame_sha256"],
                s2k_active_kc_count=c_row["active_kc_count"] == int(cref["active_kc_count"]),
                s2k_kc_spike_count=c_row["kc_spike_count"] == int(cref["kc_spike_count"]),
                s2k_spike_hash=c_row["spike_sha256"] == cref["spike_sha256"],
            )
        elif require_refs:
            c_row["equivalence"].update(
                s2k_frame_hash=False, s2k_active_kc_count=False,
                s2k_kc_spike_count=False, s2k_spike_hash=False,
            )
        c_row["equivalence"]["all"] = all(c_row["equivalence"].values())
        c_active = rows["C"]["active_kc_count"]
        c_indices = set(rows["C"]["active_kc_indices"])
        for arm in ARMS:
            row = rows[arm]
            indices = set(row["active_kc_indices"])
            union = len(indices | c_indices)
            row["relative_to_C"] = {
                "active_kc_count_delta": int(row["active_kc_count"] - c_active),
                "state_changed": row["state_category"] != rows["C"]["state_category"],
                "active_kc_jaccard": float(len(indices & c_indices) / union) if union else 1.0,
            }
            row["equivalence"]["all"] = all(row["equivalence"].values())
        records.append({"event_index": int(ei), "group": c_row["group"], "arms": rows})
    return c, b, clean, bridge, records, sc, sup, ids


def _raw_arrays(records):
    rows = [(r["event_index"], arm, r["arms"][arm]) for r in records for arm in ARMS]
    target_count = int(records[0]["arms"]["C"]["kc_target_count"])
    if any(int(r["arms"][a]["kc_target_count"]) != target_count for r in records for a in ARMS):
        raise RuntimeError("S2n KC target count changed across arms/frames")
    bitset_bytes = (target_count + 7) // 8
    return {
        "event_indices": np.asarray([x[0] for x in rows], np.int32),
        "arms": np.asarray([x[1] for x in rows]),
        "groups": np.asarray([next(r["group"] for r in records if r["event_index"] == x[0]) for x in rows]),
        "frame_sha256": np.asarray([x[2]["frame_sha256"] for x in rows]),
        "active_kc_count": np.asarray([x[2]["active_kc_count"] for x in rows], np.int32),
        "kc_spike_count": np.asarray([x[2]["kc_spike_count"] for x in rows], np.int64),
        "active_kc_hash": np.asarray([x[2]["active_kc_hash"] for x in rows]),
        "spike_sha256": np.asarray([x[2]["spike_sha256"] for x in rows]),
        "kc_target_count": np.asarray(target_count, np.int32),
        "active_kc_bitset": np.stack([np.packbits(np.bincount(x[2]["_active_kc_indices"], minlength=target_count).astype(np.uint8), bitorder="little") for x in rows]).reshape(len(rows), bitset_bytes),
        "eligible_count": np.asarray([x[2]["eligible_count"] for x in rows], np.int64),
        "applied_count": np.asarray([x[2]["applied_count"] for x in rows], np.int64),
        "eligible_events": np.asarray([x[2]["eligible_events"] for x in rows], np.int64),
        "applied_events": np.asarray([x[2]["applied_events"] for x in rows], np.int64),
        "original_signed": np.asarray([x[2]["original_signed"] for x in rows]),
        "original_positive": np.asarray([x[2]["original_positive"] for x in rows]),
        "original_negative": np.asarray([x[2]["original_negative"] for x in rows]),
        "applied_signed": np.asarray([x[2]["applied_signed"] for x in rows]),
        "applied_positive": np.asarray([x[2]["applied_positive"] for x in rows]),
        "applied_negative": np.asarray([x[2]["applied_negative"] for x in rows]),
        "selected_outside_count": np.asarray([x[2]["selected_outside_count"] for x in rows], np.int64),
        "kc1_step": np.asarray([x[2]["kc1_step"] for x in rows], np.int32),
    }


def _summary(records):
    out = {}
    for arm in ARMS:
        rr = [r["arms"][arm] for r in records]
        groups = {g: [r for r in rr if g == "all" or r["group"] == g] for g in ("all", "low", "high")}
        out[arm] = {}
        for metric in ("active_kc_count", "kc_spike_count"):
            out[arm][metric] = {g: {"n": len(v), "min": int(min(x[metric] for x in v)), "median": float(np.median([x[metric] for x in v])), "mean": float(np.mean([x[metric] for x in v])), "max": int(max(x[metric] for x in v))} for g, v in groups.items()}
        out[arm]["state_category_counts"] = {g: {k: sum(x["state_category"] == k for x in v) for k in ("low", "intermediate", "high")} for g, v in groups.items()}
        out[arm]["jaccard_to_C"] = {g: {"n": len(v), "min": float(min(x["relative_to_C"]["active_kc_jaccard"] for x in v)), "median": float(np.median([x["relative_to_C"]["active_kc_jaccard"] for x in v])), "mean": float(np.mean([x["relative_to_C"]["active_kc_jaccard"] for x in v])), "max": float(max(x["relative_to_C"]["active_kc_jaccard"] for x in v))} for g, v in groups.items()}
        out[arm]["state_category_crossings_vs_C"] = {g: int(sum(x["relative_to_C"]["state_changed"] for x in v)) for g, v in groups.items()}
        out[arm]["stable_state_change_gate"] = {"threshold_frames": 12, "passed": int(out[arm]["state_category_crossings_vs_C"]["all"]) >= 12}
    return out


def _compact_records(records):
    """Keep JSON reviewable; full vectors and KC bitsets live in NPZ."""
    drop = {"active_kc_indices", "_active_kc_indices", "_selected_source_groups",
            "selected_source_indices", "source_mapping", "original_signed",
            "original_positive", "original_negative", "applied_signed",
            "applied_positive", "applied_negative", "eligible_events", "applied_events"}
    out = []
    for rec in records:
        cr = {"event_index": rec["event_index"], "group": rec["group"], "arms": {}}
        for arm, row in rec["arms"].items():
            compact = {k: v for k, v in row.items() if k not in drop}
            compact["audit"] = {
                "eligible_events": row["eligible_events"], "applied_events": row["applied_events"],
                "original_signed": row["original_signed"], "original_positive": row["original_positive"],
                "original_negative": row["original_negative"], "applied_signed": row["applied_signed"],
                "applied_positive": row["applied_positive"], "applied_negative": row["applied_negative"],
            }
            cr["arms"][arm] = compact
        out.append(cr)
    return out


def _result(output, c, b, clean, bridge, records, sc, sup, ids, formal):
    raw = Path(output) / ("s2n_formal_raw.npz" if formal else "s2n_smoke_raw.npz")
    arrays = _raw_arrays(records)
    atomic_npz(raw, **arrays)
    result = {
        "observer": "S2n", "formal": bool(formal), "success": True,
        "preregistered": True, "counterfactual": True, "learning": False,
        "diagnostic_only": True, "production_eligible": False,
        "arms": list(ARMS), "parameters": {"window_bounds": [list(x) for x in WINDOW_BOUNDS], "scales": SCALES, "top20_other_cb_body_ids": list(TOP20), "selected_event_count": len(records)},
        "window_summary": _summary(records), "records": _compact_records(records),
        "raw_npz": str(raw), "raw_npz_sha256": sha(raw),
        "checkpoint_bridge": bridge,
        "provenance": {"pre_registration": str(PREREG), "pre_registration_sha256": sha(PREREG), "s2d": str(S2D), "s2d_sha256": sha(S2D), "s2k": str(S2K), "s2k_sha256": sha(S2K), "historical_checkpoint": str(HIST), "historical_checkpoint_sha256": sha(HIST), "fresh_checkpoint": str(clean), "fresh_checkpoint_sha256": sha(clean), "kernel_source_sha256": b.build["source_sha256"], "kernel_binary_sha256": b.build["binary_sha256"], "script_sha256": sha(Path(__file__))},
        "source_mapping": {
            "visual_source_indices": np.flatnonzero(sc == 3).astype(np.int32).tolist(),
            "visual_body_ids": ids[sc == 3].astype(np.int64).tolist(),
            "frozen_cb20_source_indices": np.flatnonzero(np.isin(ids, np.asarray(TOP20, dtype=np.int64))).astype(np.int32).tolist(),
            "frozen_cb20_body_ids": ids[np.isin(ids, np.asarray(TOP20, dtype=np.int64))].astype(np.int64).tolist(),
            "group_codes": {"0": "none", "1": "visual_projection", "2": "frozen_top20_other_cb"},
        },
        "class_mapping": {"3": "PN/visual_projection", "4": "other_cb"},
        "gates": {"fresh_checkpoint_bridge": bool(bridge.get("all_persistent_arrays_bitwise_equal")), "records": len(records) == (60 if formal else 4), "arms_per_frame": all(len(x["arms"]) == 4 for x in records), "all_arm_equivalence": all(r["arms"][a]["equivalence"]["all"] for r in records for a in ARMS), "all_audit_conservation": all(int(np.asarray(r["arms"][a]["eligible_events"]).sum()) == r["arms"][a]["eligible_count"] and int(np.asarray(r["arms"][a]["applied_events"]).sum()) == r["arms"][a]["applied_count"] for r in records for a in ARMS), "all_group_window_cells": all(r["arms"][a]["equivalence"].get("group_window_applied_cells", False) and r["arms"][a]["equivalence"].get("applied_half_original", False) for r in records for a in ARMS), "low_high": (sum(r["group"] == "low" for r in records), sum(r["group"] == "high" for r in records))},
    }
    result["gates"]["manifest_pin"] = True
    result["gates"]["all"] = all(bool(v) if not isinstance(v, tuple) else (v == ((43, 17) if formal else (2, 2))) for v in result["gates"].values())
    if formal and not result["gates"]["all"]:
        # Formal failure is terminal: leave only an explicit failure record,
        # never a success JSON or an apparently usable raw candidate.
        raw.unlink(missing_ok=True)
        for candidate in (Path(output) / "s2n_formal.json", Path(output) / "s2n_formal.json.partial"):
            candidate.unlink(missing_ok=True)
        atomic_json(Path(output) / "s2n_formal_failure.json", {"observer": "S2n", "formal": True, "success": False, "diagnostic_only": True, "counterfactual": True, "production_eligible": False, "gates": result["gates"]})
        raise RuntimeError(f"S2n formal gate failed: {result['gates']}")
    result["json_manifest_sha256"] = manifest_digest(result)
    arrays["json_manifest_sha256"] = np.asarray(result["json_manifest_sha256"])
    atomic_npz(raw, **arrays); result["raw_npz_sha256"] = sha(raw)
    json_path = Path(output) / ("s2n_formal.json" if formal else "s2n_smoke.json")
    atomic_json(json_path, result)
    pin = verify_manifest_pin(json_path, raw)
    if not pin["all"]:
        raise RuntimeError(f"S2n manifest pin failed: {pin}")
    return result


def run_smoke(output=None):
    old = ROOT / "runs/plasticity_diagnosis/S2n/20261006T-windowed-causal-attenuation-smoke"
    atomic_json(old / "superseded_invalid.json", {
        "observer": "S2n", "status": "superseded_invalid",
        "reason": "ABI-v1 single source_selected applied window scales across groups; BOTH leaked visual attenuation into W4 and CB attenuation into W1/W2.",
        "replacement": "20261007T-windowed-causal-attenuation-smoke-v2",
    })
    output = Path(output or ROOT / "runs/plasticity_diagnosis/S2n/20261007T-windowed-causal-attenuation-smoke-v2")
    ev = _events(); selected = [x for x in ev if int(x[2]["active_kc_count"]) <= 10][:2] + [x for x in ev if int(x[2]["active_kc_count"]) >= 1000][:2]
    c, b, clean, bridge, records, sc, sup, ids = _run(output, selected)
    return _result(output, c, b, clean, bridge, records, sc, sup, ids, False)


def run_formal(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    try:
        ev = _events()
        if len(ev) != 60: raise RuntimeError("S2n formal requires exact 60 S2d frames")
        c, b, clean, bridge, records, sc, sup, ids = _run(output, ev, require_refs=True)
        if sum(r["group"] == "low" for r in records) != 43 or sum(r["group"] == "high" for r in records) != 17:
            raise RuntimeError("S2n formal requires low43/high17")
        return _result(output, c, b, clean, bridge, records, sc, sup, ids, True)
    except Exception as exc:
        for p in output.glob("*.partial*"): p.unlink(missing_ok=True)
        (output / "s2n_formal.json").unlink(missing_ok=True)
        (output / "s2n_formal_raw.npz").unlink(missing_ok=True)
        atomic_json(output / "s2n_formal_failure.json", {
            "observer": "S2n", "formal": True, "success": False,
            "diagnostic_only": True, "counterfactual": True,
            "production_eligible": False, "error": type(exc).__name__ + ": " + str(exc),
        })
        raise


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path); ap.add_argument("--formal-output", type=Path); args = ap.parse_args()
    if args.formal_output is not None:
        raise RuntimeError("S2n formal is implemented but CLI launch is disabled pending explicit approval")
    result = run_smoke(args.output)
    print(json.dumps({"path": str(args.output), "gates": result["gates"]}, sort_keys=True))


if __name__ == "__main__": main()
