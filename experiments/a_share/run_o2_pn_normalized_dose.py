#!/usr/bin/env python3
"""O2 post-stop counterfactual PN-normalized dose diagnosis."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stonkfly.neural.v2_brain import HybridBrain

OUT = ROOT / "runs/whole_brain_olfactory/O2/20261010T-pn-normalized-dose-diagnosis-v1"
PREREG = ROOT / "docs/20261010-o2-pn-normalized-dose-diagnosis-preregistered.md"
TESTS = ROOT / "tests/test_o2_pn_normalized_dose.py"
O0 = ROOT / "runs/whole_brain_olfactory/O0/20261010T-pn-proxy-inventory-v1"
O1 = ROOT / "runs/whole_brain_olfactory/O1/20261010T-pn-single-event-v1"
PANEL_SHA = "cc011844e8ed4e415353e8a835bca04a0e091d0179aaccef94e86d3e426d5301"
PREREG_SHA = "cd864ec9e1fc1250b638bef5df8159e796970fe7319f2e972af68bd292440488"
DT = (0.1, 0.05, 0.025)
DOSES = (128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0)
SEGMENTS = (2.0, 48.0, 450.0, 500.0)

PINS = {
    "data/graph.npz": "346b8af85a11af13b8324e18669812c1924569e7d1adcb4e6f45cc461a2c344b",
    "data/normalized/neurons.feather": "2103b5203d2ba5502c06fb70a4e0b23edcbcf79ed0c344722101941fc289ba2d",
    "o0_formal": "8c09ccfccc9946bf95892cb1c77aa80ce60331e79394deb0ebb9f5372602f69b",
    "o1_raw": "046e9865a5310da6c56379b1b23647186af47f7cd0148634eef405f8c9b2a665",
    "o1_formal": "8b19cd27314cdfeb1870c6307c7aac4ee1c41e5ccc86871b1ee0ae910e3cdb08",
    "o1_manifest": "1e7ce86d22badb94fb988075dba51824bf5e947d344f72b2af6773bcdb7ff93f",
    "q1_wrapper": "2673de4e7c4aa4ee0fd20636b17bf2c2828103bacfbbb05b2ff0f559979434e6",
    "q1_binary": "9df8110f1e2134f8a6d88cde46d3a8a6278590f291ad87b49161926d2fd53f41",
}
PIN_PATH = {
    "data/graph.npz": ROOT / "data/graph.npz",
    "data/normalized/neurons.feather": ROOT / "data/normalized/neurons.feather",
    "o0_formal": O0 / "o0_formal.json",
    "o1_raw": O1 / "o1_raw.npz",
    "o1_formal": O1 / "o1_formal.json",
    "o1_manifest": O1 / "o1_manifest.json",
    "q1_wrapper": ROOT / "stonkfly/neural/v2_brain.py",
    "q1_binary": ROOT / "runs/whole_brain_v2/Q1/.build/libv2_hybrid.so",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def arrsha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def jsonsha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fixed():
    actual = {key: sha(path) for key, path in PIN_PATH.items()}
    bad = {key: (PINS[key], actual[key]) for key in PINS if PINS[key] != actual[key]}
    if sha(PREREG) != PREREG_SHA:
        bad["prereg"] = (PREREG_SHA, sha(PREREG))
    o0 = json.loads((O0 / "o0_formal.json").read_text())
    o1 = json.loads((O1 / "o1_formal.json").read_text())
    expected_o1 = {
        "implementation": True,
        "propagation": True,
        "stability": False,
        "deterministic_repeats": True,
        "dt_class_consistent": True,
        "trial_count": 54,
    }
    if o0.get("verdict") != "O0_PN_PROXY_INVENTORY_QUALIFIED" or o0.get("panel", {}).get("panel_sha256") != PANEL_SHA:
        bad["o0_verdict_panel"] = "changed"
    if o1.get("verdict") != "O1_SINGLE_EVENT_INSTABILITY" or o1.get("gates") != expected_o1:
        bad["o1_verdict_gates"] = "changed"
    if bad:
        raise RuntimeError("O2 pin mismatch " + json.dumps(bad, sort_keys=True))
    return actual, o0, o1


def activate(brain, nodes):
    for node in np.unique(np.asarray(nodes, dtype=np.int32)):
        if not brain.active_flag[node]:
            brain.active[brain.nactive[0]] = node
            brain.nactive[0] += 1
            brain.active_flag[node] = 1


def inject_normalized_release(brain, source, dose):
    source = int(source)
    dose = float(dose)
    if not 0 <= source < brain.n:
        raise ValueError("source outside graph")
    if not math.isfinite(dose) or dose <= 0:
        raise ValueError("dose must be positive and finite")
    if int(brain.clock[0]) != 0 or np.any(brain.spike_queue_count) or np.any(brain.g_event):
        raise RuntimeError("normalized release requires clean clock/queue/conductance")
    lo, hi = int(brain.ptr[source]), int(brain.ptr[source + 1])
    if hi <= lo:
        raise ValueError("source has no outgoing edges")
    targets = brain.post[lo:hi]
    weights = brain.weight[lo:hi]
    l1 = float(np.abs(weights.astype(np.float64)).sum())
    if not math.isfinite(l1) or l1 <= 0:
        raise ValueError("source has invalid outgoing L1")
    edge_delivery = weights.astype(np.float64) * (dose / l1)
    np.add.at(brain.g_event, targets, edge_delivery.astype(np.float32))
    activate(brain, targets)
    return {
        "kind": "engineered_pn_efferent_release_proxy",
        "source_node_index": source,
        "dose_l1": dose,
        "source_edge_l1": l1,
        "scale": dose / l1,
        "edge_count": int(hi - lo),
        "unique_target_count": int(len(np.unique(targets))),
        "edge_delivery_l1": float(np.abs(edge_delivery).sum()),
        "target_delivery_l1": float(np.abs(brain.g_event[np.unique(targets)].astype(np.float64)).sum()),
        "target_delivery_sha256": arrsha(brain.g_event),
    }


def state_hash(brain):
    h = hashlib.sha256()
    for name in ("v", "g_event", "refractory", "spike_queue", "spike_queue_count", "active", "active_flag", "nactive", "adaptation", "counts"):
        h.update(name.encode())
        h.update(np.ascontiguousarray(getattr(brain, name)).tobytes())
    h.update(str(int(brain.clock[0])).encode())
    h.update(str(int(brain.total_spikes)).encode())
    return h.hexdigest()


def classify(stable_exit, early_kc_ids, kc_count):
    if not stable_exit:
        return "unstable"
    n = len(early_kc_ids)
    if n == 0:
        return "silent"
    if n > math.floor(0.2 * kc_count):
        return "dense_nonzero"
    return "stable_nonzero"


def jaccard(left, right):
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def structural_rows(brain, panel, kc, o1):
    o1_lookup = {(r["source_node_index"], r["dt_ms"]): r for r in o1["trials"] if r["repeat"] == 1 and r["source_node_index"] is not None}
    rows = []
    for meta in panel:
        source = int(meta["node_index"])
        lo, hi = int(brain.ptr[source]), int(brain.ptr[source + 1])
        post, weight = brain.post[lo:hi], brain.weight[lo:hi]
        q = np.quantile(weight.astype(np.float64), [0, 0.25, 0.5, 0.75, 1]).tolist()
        direct_kc = post[kc[post]]
        prior = [o1_lookup[(source, dt)] for dt in DT]
        rows.append({
            "cell_type": meta["cell_type"], "source_id": int(meta["source_id"]), "source_node_index": source,
            "edge_count": int(len(post)), "unique_target_count": int(len(np.unique(post))),
            "direct_kc_edge_count": int(len(direct_kc)), "direct_kc_target_count": int(len(np.unique(direct_kc))),
            "positive_count": int(np.count_nonzero(weight > 0)), "negative_count": int(np.count_nonzero(weight < 0)), "zero_count": int(np.count_nonzero(weight == 0)),
            "signed_sum": float(weight.astype(np.float64).sum()), "l1": float(np.abs(weight.astype(np.float64)).sum()),
            "l2": float(np.linalg.norm(weight.astype(np.float64))), "quantiles": q,
            "edge_vector_sha256": arrsha(np.rec.fromarrays([post, weight], names="post,weight")),
            "o1": [{"dt_ms": r["dt_ms"], "direct_target_count": r["direct_target_count"], "direct_kc_count": r["direct_kc_count"], "g_abs_max_2ms": r["segments"][0]["g_abs_max"], "total_spikes": r["total_spikes"], "kc_spikes": int(sum(s["kc_spikes"] for s in r["segments"])), "stable_exit": r["stable_exit"]} for r in prior],
        })
    return rows


def trial(dt, source, meta, dose, repeat, nodes, segments=SEGMENTS):
    brain = HybridBrain(ROOT / "data/graph.npz", dt=dt, graded_mask=np.zeros(166700, np.uint8))
    cell_type = np.asarray(nodes["cell_type"], str)
    kc = np.char.startswith(cell_type.astype(str), "KC")
    brain.kc_mask[:] = kc
    weights_before = arrsha(brain.weight)
    intent = {"kind": "ZERO", "dose_l1": 0.0}
    if source is not None:
        intent = inject_normalized_release(brain, source, dose)
    segment_rows, early_kc = [], set()
    finite, bounds = True, True
    for index, duration in enumerate(segments):
        counts = brain.run(duration)
        if index < 2:
            early_kc.update(np.flatnonzero(kc & (counts > 0)).tolist())
        finite = finite and bool(np.isfinite(brain.v).all() and np.isfinite(brain.g_event).all() and np.isfinite(brain.adaptation).all())
        bounds = bounds and bool(np.max(np.abs(brain.v)) < 1e6 and np.max(np.abs(brain.g_event)) < 1e9)
        segment_rows.append({
            "duration_ms": duration, "spikes": int(counts.sum()), "kc_spikes": int(counts[kc].sum()),
            "kc_unique_spike_nodes": int(np.count_nonzero(kc & (counts > 0))), "active_nodes": int(brain.nactive[0]),
            "queue_count": int(brain.spike_queue_count.sum()), "v_min": float(brain.v.min()), "v_max": float(brain.v.max()),
            "g_abs_max": float(np.max(np.abs(brain.g_event))),
        })
    early_kc_ids = sorted(early_kc)
    ulp = float(abs(np.spacing(np.float32(-52.0))))
    final_linf = float(np.max(np.abs(brain.v - brain.rest)))
    queue_empty = bool(np.all(brain.spike_queue_count == 0))
    g_quiet = bool(np.max(np.abs(brain.g_event)) <= 1e-7)
    adaptation_quiet = bool(np.max(np.abs(brain.adaptation)) <= 1e-4)
    stable_exit = bool(finite and bounds and not brain.overflow[0] and segment_rows[-1]["spikes"] == 0 and queue_empty and g_quiet and adaptation_quiet and final_linf <= 4 * ulp)
    zero_ok = bool(source is not None or (brain.total_spikes == 0 and max(r["g_abs_max"] for r in segment_rows) == 0 and final_linf == 0))
    response_class = "zero" if source is None else classify(stable_exit, early_kc_ids, int(kc.sum()))
    return {
        "name": "ZERO" if source is None else meta["cell_type"], "source_node_index": None if source is None else int(source),
        "source_id": None if source is None else int(meta["source_id"]), "dose_l1": None if source is None else float(dose),
        "dt_ms": dt, "repeat": repeat, "intent": intent, "segments": segment_rows, "total_spikes": int(brain.total_spikes),
        "early_kc_ids": early_kc_ids, "early_kc_count": len(early_kc_ids), "early_kc_sha256": jsonsha(early_kc_ids),
        "response_class": response_class, "finite": finite, "bounds": bounds, "overflow": int(brain.overflow[0]),
        "final_linf_mv": final_linf, "four_ulp_mv": 4 * ulp, "queue_empty": queue_empty, "g_quiet": g_quiet,
        "adaptation_quiet": adaptation_quiet, "last_500ms_spikes": segment_rows[-1]["spikes"], "stable_exit": stable_exit,
        "zero_ok": zero_ok, "weights_unchanged": weights_before == arrsha(brain.weight), "final_state_sha256": state_hash(brain),
    }


def evaluate(rows, panel, doses, formal=True):
    by_repeat = {}
    for row in rows:
        by_repeat.setdefault((row["dt_ms"], row["source_node_index"], row["dose_l1"]), []).append(row)
    deterministic = all(len(v) == 2 and v[0]["segments"] == v[1]["segments"] and v[0]["early_kc_ids"] == v[1]["early_kc_ids"] and v[0]["intent"] == v[1]["intent"] and v[0]["final_state_sha256"] == v[1]["final_state_sha256"] for v in by_repeat.values())
    ledger = all(r["source_node_index"] is None or (abs(r["intent"]["edge_delivery_l1"] - r["dose_l1"]) <= 1e-9 * r["dose_l1"] and abs(r["intent"]["target_delivery_l1"] - r["dose_l1"]) <= 2e-6 * r["dose_l1"]) for r in rows)
    expected = 294 if formal else len(rows)
    implementation = bool(len(rows) == expected and all(r["weights_unchanged"] and r["zero_ok"] for r in rows) and ledger and deterministic)
    dt_consistency = {}
    for source in [int(x["node_index"]) for x in panel]:
        for dose in doses:
            classes = sorted({r["response_class"] for r in rows if r["source_node_index"] == source and r["dose_l1"] == dose})
            dt_consistency[f"{source}:{dose:g}"] = {"classes": classes, "consistent": len(classes) == 1}
    candidates = []
    if formal:
        for dose in doses:
            dose_rows = [r for r in rows if r["source_node_index"] is not None and r["dose_l1"] == dose]
            all_stable = len(dose_rows) == 8 * 3 * 2 and all(r["stable_exit"] for r in dose_rows)
            dose_dt_consistent = all(dt_consistency[f"{int(meta['node_index'])}:{dose:g}"]["consistent"] for meta in panel)
            responsive = []
            robustness = {}
            for meta in panel:
                source = int(meta["node_index"])
                source_rows = [r for r in dose_rows if r["source_node_index"] == source]
                if len(source_rows) == 6 and all(r["response_class"] == "stable_nonzero" for r in source_rows):
                    responsive.append(source)
                    sets = {r["dt_ms"]: r["early_kc_ids"] for r in source_rows if r["repeat"] == 1}
                    pairs = [jaccard(sets[DT[i]], sets[DT[j]]) for i in range(3) for j in range(i + 1, 3)]
                    robustness[str(source)] = {"pairwise_dt_jaccard": pairs, "minimum": min(pairs)}
            base = {r["source_node_index"]: r["early_kc_ids"] for r in dose_rows if r["repeat"] == 1 and r["dt_ms"] == 0.1 and r["source_node_index"] in responsive}
            pairwise = [jaccard(base[responsive[i]], base[responsive[j]]) for i in range(len(responsive)) for j in range(i + 1, len(responsive))]
            distinct = len({jsonsha(base[s]) for s in responsive}) if responsive else 0
            robust = bool(responsive and all(x["minimum"] >= 0.8 for x in robustness.values()))
            separated = bool(len(responsive) >= 4 and distinct >= 3 and pairwise and max(pairwise) <= 0.9)
            qualifies = bool(all_stable and dose_dt_consistent and len(responsive) >= 4 and robust and separated and deterministic)
            candidates.append({"dose_l1": dose, "all_stable": all_stable, "dt_class_consistent": dose_dt_consistent, "responsive_sources": responsive, "responsive_count": len(responsive), "dt_robustness": robustness, "distinct_code_count": distinct, "pairwise_source_jaccard": pairwise, "qualifies": qualifies})
    return {
        "implementation": implementation, "deterministic_repeats": deterministic, "input_ledger": ledger,
        "trial_count": len(rows), "all_finite": all(r["finite"] for r in rows), "all_bounds": all(r["bounds"] for r in rows),
        "all_zero_controls": all(r["zero_ok"] for r in rows if r["source_node_index"] is None),
        "global_dt_class_consistent": all(x["consistent"] for x in dt_consistency.values()) if dt_consistency else True,
        "dt_consistency": dt_consistency, "candidates": candidates, "candidate_doses": [x["dose_l1"] for x in candidates if x["qualifies"]],
    }


def snapshot():
    return {
        "prereg": sha(PREREG), "runner": sha(Path(__file__)), "tests": sha(TESTS) if TESTS.exists() else None,
        "q1_wrapper": sha(PIN_PATH["q1_wrapper"]), "q1_binary": sha(PIN_PATH["q1_binary"]),
    }


def prepare(out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def write(out, formal, raw, start, started):
    raw_path = out / "o2_raw.npz"
    np.savez_compressed(raw_path, **raw)
    formal["wall_time_seconds"] = time.time() - started
    formal_path = out / "o2_formal.json"
    formal_path.write_text(json.dumps(formal, indent=2, sort_keys=True) + "\n")
    report = out / "o2_report.md"
    report.write_text(f"# O2 PN-normalized dose diagnosis\n\nVerdict: `{formal['verdict']}`\n\nCounterfactual diagnostic only; O1 remains closed. No market, learning, decoder, or execution.\n")
    launch = out / "launch.txt"
    launch.write_text(" ".join(sys.argv) + "\n")
    stdout = out / "stdout.txt"
    stdout.write_text(json.dumps({"verdict": formal["verdict"], "candidate_doses": formal.get("gates", {}).get("candidate_doses", [])}, sort_keys=True) + "\n")
    manifest = {
        "schema": "o2-manifest-v1", "hashes_start": start, "hashes_end": snapshot(), "python": sys.version,
        "platform": platform.platform(), "artifacts": {"raw": sha(raw_path), "formal": sha(formal_path), "report": sha(report), "launch": sha(launch), "stdout": sha(stdout)},
    }
    (out / "o2_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def run(out=OUT, smoke=False):
    started = time.time()
    out = prepare(out)
    pins, o0, o1 = fixed()
    start = snapshot()
    panel = o0["panel"]["nodes"]
    nodes = feather.read_table(ROOT / "data/normalized/neurons.feather").to_pydict()
    cell_type = np.asarray(nodes["cell_type"], str)
    kc = np.char.startswith(cell_type.astype(str), "KC")
    audit_brain = HybridBrain(ROOT / "data/graph.npz", dt=0.1, graded_mask=np.zeros(166700, np.uint8))
    structure = structural_rows(audit_brain, panel, kc, o1)
    dts = (0.1,) if smoke else DT
    selected = panel[:2] if smoke else panel
    doses = (128.0,) if smoke else DOSES
    segments = (2.0, 8.0) if smoke else SEGMENTS
    rows = []
    for dt in dts:
        for repeat in (1, 2):
            rows.append(trial(dt, None, {}, None, repeat, nodes, segments))
            for meta in selected:
                for dose in doses:
                    rows.append(trial(dt, int(meta["node_index"]), meta, dose, repeat, nodes, segments))
    gates = evaluate(rows, selected, doses, formal=not smoke)
    if smoke:
        verdict = "O2_SMOKE_OK" if gates["implementation"] and gates["deterministic_repeats"] and gates["input_ledger"] else "O2_IMPLEMENTATION_FAILURE"
    elif not gates["implementation"]:
        verdict = "O2_IMPLEMENTATION_FAILURE"
    elif gates["candidate_doses"]:
        verdict = "O2_COUNTERFACTUAL_BAND_FOUND_REVIEW_REQUIRED"
    else:
        verdict = "O2_NO_STABLE_NONZERO_BAND_CLOSE_V1"
    formal = {
        "schema": "o2-pn-normalized-dose-v1", "verdict": verdict, "smoke": smoke, "counterfactual": True,
        "o1_route_closed": True, "production_eligible": False, "market_used": False, "learning_used": False,
        "reward_used": False, "decoder_used": False, "pins": pins, "preregistration_sha256": sha(PREREG),
        "panel_sha256": PANEL_SHA, "dose_l1_grid": list(doses), "segments_ms": list(segments), "kc_count": int(kc.sum()),
        "normalization": "dose * edge_weight / source_outgoing_L1", "structure": structure, "gates": gates, "trials": rows,
    }
    if snapshot() != start:
        raise RuntimeError("O2 sources changed during run")
    raw = {
        "trial_json": np.asarray(json.dumps(rows, sort_keys=True)), "structure_json": np.asarray(json.dumps(structure, sort_keys=True)),
        "source_node_index": np.asarray([-1 if r["source_node_index"] is None else r["source_node_index"] for r in rows]),
        "dose_l1": np.asarray([np.nan if r["dose_l1"] is None else r["dose_l1"] for r in rows]),
        "dt_ms": np.asarray([r["dt_ms"] for r in rows]), "response_class": np.asarray([r["response_class"] for r in rows]),
        "early_kc_count": np.asarray([r["early_kc_count"] for r in rows]), "total_spikes": np.asarray([r["total_spikes"] for r in rows]),
        "final_state_sha256": np.asarray([r["final_state_sha256"] for r in rows]),
    }
    write(out, formal, raw, start, started)
    return formal


def failure(out, error):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    formal = {"schema": "o2-pn-normalized-dose-v1", "verdict": "O2_IMPLEMENTATION_FAILURE", "error": repr(error), "traceback": traceback.format_exc(), "counterfactual": True, "production_eligible": False}
    write(out, formal, {"failure": np.asarray(repr(error))}, snapshot(), time.time())
    return formal


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run(args.out, args.smoke)
    except Exception as error:
        result = failure(args.out, error)
        print(json.dumps(result, indent=2))
        raise
    if args.smoke and result["verdict"] == "O2_SMOKE_OK":
        print("O2_SMOKE_OK")
    else:
        print(json.dumps({"verdict": result["verdict"], "candidate_doses": result["gates"]["candidate_doses"], "wall_time_seconds": result["wall_time_seconds"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
