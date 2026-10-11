#!/usr/bin/env python3
"""V2-P0 whole-retained-graph model inventory.

Structural only: this runner does not import the neural kernel, market data,
controller, plasticity, action decoder, or any trading result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "runs/whole_brain_v2/P0/20261008T-model-inventory-v3"
LAYERS = ("modulatory_declared", "graded_candidate_photoreceptor", "graded_candidate_lamina", "graded_candidate_apl", "adaptive_spiking_kc", "generic_spiking_declared")
LAYER_INDEX = {x: i for i, x in enumerate(LAYERS)}
PATH_TARGETS = ("KC", "APL", "MBON07", "MBON11")
BUCKET_LABELS = ("0", "1", "2", "3", "4", "5", "6", ">6/unreached")
FROZEN_HASHES = {
    "data/graph.npz": "346b8af85a11af13b8324e18669812c1924569e7d1adcb4e6f45cc461a2c344b",
    "data/normalized/neurons.feather": "2103b5203d2ba5502c06fb70a4e0b23edcbcf79ed0c344722101941fc289ba2d",
    "data/normalized/edges.arrow": "842b762c9427c794eaedd453178aa034727a5084850a91fdcc4d3fffe09f5b7e",
    "stonkfly/neural/kernel.cpp": "aea2b97d2f9d78c57787f40f42a9d7dd777c3059d8c7f453e61c4d94398438d4",
    "docs/20261007-r0-results-for-opus.md": "fae16a8946299356976397f2925c429d4a81b2ac2526e4386e6658a6bd69b2f0",
    "docs/20261007-r0c-dt-convergence-preregistered.md": "a4d6baeedf7944aa1968d678d80a47c17dd4ecb50cb48d60aa927f8d65efb9f0",
    "runs/plasticity_diagnosis/R0c/20261008T-dt-convergence-v3/r0c_formal.json": "b531533ae1a401afebeae5317e4a1ef6b963e1afedd1ad2018dd071ecc0dec51",
    "docs/20261008-model-scope-and-stop-review.md": "47396caeca7b4b57c0c1db4e12c1e053d774901e95a3dec3e2c20090ce42284e",
}
HISTORY_JSON = {
    "R0": ROOT / "runs/plasticity_diagnosis/R0/20261007T-synthetic-dose-response-v2/r0_formal.json",
    "R0b": ROOT / "runs/plasticity_diagnosis/R0b/20261007T-absorbing-state-v1/r0b_formal.json",
    "R0c": ROOT / "runs/plasticity_diagnosis/R0c/20261008T-dt-convergence-v3/r0c_formal.json",
}
EXPECTED_GRAPH_SCALE = {"nodes": 166_700, "edges": 25_582_938, "synaptic_contacts": 124_177_617}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _strings(values: Any) -> np.ndarray:
    return np.asarray(["" if x is None else str(x) for x in values])


def classify_layers(cell_type: np.ndarray, transmitter: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the frozen mutually-exclusive declaration priority."""
    from stonkfly.neural.transmitters import transmitter_signs

    ct, nt = _strings(cell_type), _strings(transmitter)
    layer = np.full(len(ct), LAYER_INDEX["generic_spiking_declared"], dtype=np.int8)
    mod = np.isin(nt, ("dopamine", "octopamine", "serotonin"))
    layer[mod] = LAYER_INDEX["modulatory_declared"]
    mask = ~mod
    layer[mask & ((ct == "R1-R6") | np.char.startswith(ct, "R8"))] = LAYER_INDEX["graded_candidate_photoreceptor"]
    layer[mask & np.isin(ct, ("L1", "L2", "L3", "L5"))] = LAYER_INDEX["graded_candidate_lamina"]
    layer[mask & (ct == "APL")] = LAYER_INDEX["graded_candidate_apl"]
    layer[mask & np.char.startswith(ct, "KC")] = LAYER_INDEX["adaptive_spiking_kc"]
    signs, uncertain = transmitter_signs(nt)
    return layer, uncertain.astype(bool), signs.astype(np.int8)


def _read_nodes(path: Path) -> dict[str, np.ndarray]:
    t = feather.read_table(path, columns=["node_index", "source_id", "retained", "superclass", "cell_type", "neurotransmitter"])
    return {name: np.asarray(t.column(name).to_pylist()) for name in t.column_names}


def _read_edges(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with pa.memory_map(str(path), "r") as f:
        t = ipc.open_file(f).read_all()
    return tuple(t.column(k).to_numpy(zero_copy_only=False) for k in ("pre_index", "post_index", "synapse_count"))


def _layer_masks(layer: np.ndarray, cell_type: np.ndarray, superclass: np.ndarray) -> dict[str, np.ndarray]:
    ct, sc = _strings(cell_type), _strings(superclass)
    return {"KC": np.char.startswith(ct, "KC"), "APL": ct == "APL", "MBON07": np.char.startswith(ct, "MBON07"), "MBON11": np.char.startswith(ct, "MBON11"), "central_brain": np.char.startswith(sc, "cb_")}


def _histogram(dist: np.ndarray, target_mask: np.ndarray) -> list[int]:
    vals = dist[target_mask]
    return [int(np.sum(vals == i)) for i in range(7)] + [int(np.sum(vals < 0) + np.sum(vals > 6))]


def _bfs_dist(ptr: np.ndarray, post: np.ndarray, starts: np.ndarray, max_hops: int = 6) -> np.ndarray:
    """Shortest directed hop distance through every retained CSR edge."""
    n = len(ptr) - 1
    starts = np.flatnonzero(starts) if starts.dtype == bool else np.asarray(starts, dtype=np.int64)
    dist = np.full(n, -1, dtype=np.int8)
    dist[starts] = 0
    frontier = np.zeros(n, dtype=bool)
    frontier[starts] = True
    counts = np.diff(ptr)
    for hop in range(1, max_hops + 1):
        targets = post[np.repeat(frontier, counts)]
        if targets.size == 0:
            break
        unseen = dist[targets] < 0
        nxt = np.zeros(n, dtype=bool)
        nxt[targets[unseen]] = True
        if not np.any(nxt):
            break
        dist[nxt] = hop
        frontier = nxt
    return dist


def _edge_sums(mask: np.ndarray, weight: np.ndarray, contacts: np.ndarray, uncertain_edge: np.ndarray) -> dict[str, Any]:
    w, c, u = weight[mask], contacts[mask], uncertain_edge[mask]
    return {"edge_count": int(mask.sum()), "synaptic_contact_count": int(c.sum(dtype=np.uint64)), "positive_edge_count": int((~u & (w > 0)).sum()), "negative_edge_count": int((~u & (w < 0)).sum()), "uncertain_sign_edge_count": int(u.sum()), "zero_edge_count": int((~u & (w == 0)).sum()), "signed_baseline_weight": float(w.sum(dtype=np.float64)), "absolute_baseline_weight": float(np.abs(w).sum(dtype=np.float64)), "uncertain_sign_absolute_weight": float(np.abs(w[u]).sum(dtype=np.float64))}


def _target_input(name: str, target_mask: np.ndarray, source_idx: np.ndarray, post: np.ndarray, source_layer: np.ndarray, source_uncertain: np.ndarray, source_sign: np.ndarray, source_ct: np.ndarray, weight: np.ndarray, contacts: np.ndarray) -> dict[str, Any]:
    em = target_mask[post]
    src, w, c = source_idx[em], weight[em], contacts[em]
    u, layer, absw = source_uncertain[src], source_layer[src], np.abs(w)
    total_abs = float(absw.sum(dtype=np.float64))
    by_layer = {}
    for lname, li in LAYER_INDEX.items():
        m = layer == li
        by_layer[lname] = {"edge_count": int(m.sum()), "synaptic_contact_count": int(c[m].sum(dtype=np.uint64)), "absolute_weight": float(absw[m].sum(dtype=np.float64)), "absolute_weight_fraction": float(absw[m].sum(dtype=np.float64) / total_abs) if total_abs else 0.0, "unique_source_neuron_count": int(np.unique(src[m]).size)}
    sign_m = {"positive": (~u) & (source_sign[src] > 0), "negative": (~u) & (source_sign[src] < 0), "uncertain": u}
    signs = {k: {"edge_count": int(m.sum()), "absolute_weight": float(absw[m].sum(dtype=np.float64)), "absolute_weight_fraction": float(absw[m].sum(dtype=np.float64) / total_abs) if total_abs else 0.0} for k, m in sign_m.items()}
    top = []
    for ct, _ in sorted(Counter(source_ct[src].tolist()).items(), key=lambda kv: (-float(absw[source_ct[src] == kv[0]].sum(dtype=np.float64)), kv[0]))[:20]:
        m = source_ct[src] == ct; val = float(absw[m].sum(dtype=np.float64))
        top.append({"cell_type": ct, "edge_count": int(m.sum()), "absolute_weight": val, "absolute_weight_fraction": val / total_abs if total_abs else 0.0})
    return {"target": name, "target_neuron_count": int(target_mask.sum()), "incoming_edge_count": int(em.sum()), "incoming_synaptic_contact_count": int(c.sum(dtype=np.uint64)), "incoming_absolute_weight": total_abs, "by_source_layer": by_layer, "sign_absolute_weight": signs, "top20_source_cell_type": top, "uncertain_sign_absolute_weight_fraction": signs["uncertain"]["absolute_weight_fraction"]}


def load_fixed_inputs() -> dict[str, Any]:
    actual = {rel: sha256_file(ROOT / rel) for rel in FROZEN_HASHES}
    bad = {k: (v, actual[k]) for k, v in FROZEN_HASHES.items() if actual[k] != v}
    if bad:
        raise RuntimeError("frozen input hash mismatch: " + json.dumps(bad, sort_keys=True))
    nodes = _read_nodes(ROOT / "data/normalized/neurons.feather")
    pre_norm, post_norm, contacts_norm = _read_edges(ROOT / "data/normalized/edges.arrow")
    graph = np.load(ROOT / "data/graph.npz", allow_pickle=False)
    ptr = graph["ptr"].astype(np.int64, copy=False)
    post = graph["post"].astype(np.int64, copy=False)
    weight = graph["weight"].astype(np.float64, copy=False)
    ids = graph["ids"].astype(np.uint64, copy=False)
    n = len(ids)
    # The normalized Arrow table is authoritative for contact counts but is
    # not guaranteed to have CSR row order.  Reproduce prepare.py's stable
    # pre-index ordering before checking source/post alignment.
    norm_order = np.argsort(pre_norm, kind="stable")
    pre_sorted, post_sorted = pre_norm[norm_order], post_norm[norm_order]
    contacts_sorted = contacts_norm[norm_order]
    graph_scale = {"nodes": n, "edges": len(post), "synaptic_contacts": int(contacts_sorted.sum(dtype=np.uint64))}
    if graph_scale != EXPECTED_GRAPH_SCALE:
        raise RuntimeError(f"graph scale mismatch: expected {EXPECTED_GRAPH_SCALE}, got {graph_scale}")
    expected_pre = np.repeat(np.arange(n, dtype=pre_norm.dtype), np.diff(ptr))
    if ptr.shape != (n + 1,) or len(post) != len(weight) or len(pre_norm) != len(post):
        raise RuntimeError("graph/normalized edge shape mismatch")
    if not (np.array_equal(ids, nodes["source_id"].astype(np.uint64)) and np.array_equal(np.arange(n, dtype=np.uint32), nodes["node_index"].astype(np.uint32))):
        raise RuntimeError("normalized node order does not match graph ids")
    if not (np.array_equal(pre_sorted, expected_pre) and np.array_equal(post_sorted, post.astype(post_sorted.dtype))):
        raise RuntimeError("normalized edge table does not align with CSR source/post order")
    layer, uncertain, sign = classify_layers(nodes["cell_type"], nodes["neurotransmitter"])
    return {"nodes": nodes, "ptr": ptr, "post": post, "weight": weight, "contacts": contacts_sorted.astype(np.uint64, copy=False), "ids": ids, "layer": layer, "uncertain": uncertain, "sign": sign, "superclass": _strings(nodes["superclass"]), "cell_type": _strings(nodes["cell_type"]), "source_idx": expected_pre.astype(np.int32, copy=False), "frozen_hashes": actual}


def _layer_inventory(d: dict[str, Any]) -> dict[str, Any]:
    layer, src, ct, nt = d["layer"], d["source_idx"], d["cell_type"], _strings(d["nodes"]["neurotransmitter"])
    out = {"node_count_total": int(len(d["ids"])), "layers": {}}
    source_uncertain_edge = d["uncertain"][src]
    for lname, li in LAYER_INDEX.items():
        nm = layer == li
        em, im = layer[src] == li, layer[d["post"]] == li
        out["layers"][lname] = {
            "node_count": int(nm.sum()),
            "source_id_sha256": hashlib.sha256(d["ids"][nm].astype("<u8", copy=False).tobytes()).hexdigest(),
            "cell_type_frequency": dict(sorted(Counter(ct[nm].tolist()).items())),
            "neurotransmitter_frequency": dict(sorted(Counter(nt[nm].tolist()).items())),
            "outgoing": _edge_sums(em, d["weight"], d["contacts"], source_uncertain_edge),
            "incoming": _edge_sums(im, d["weight"], d["contacts"], source_uncertain_edge),
        }
    return out


def compute_inventory(d: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    n, e = len(d["ids"]), len(d["post"])
    layer, src = d["layer"], d["source_idx"]
    masks = _layer_masks(layer, d["cell_type"], d["superclass"])
    uedge, tlayer = d["uncertain"][src], layer[d["post"]]
    key = layer[src] * len(LAYERS) + tlayer
    shape = (len(LAYERS), len(LAYERS))
    flow_edges = np.bincount(key, minlength=len(LAYERS) ** 2).reshape(shape)
    flow_contacts = np.bincount(key, weights=d["contacts"].astype(np.float64), minlength=len(LAYERS) ** 2).reshape(shape)
    flow_signed = np.bincount(key, weights=d["weight"], minlength=len(LAYERS) ** 2).reshape(shape)
    flow_abs = np.bincount(key, weights=np.abs(d["weight"]), minlength=len(LAYERS) ** 2).reshape(shape)
    flow_unc_abs = np.bincount(key, weights=np.where(uedge, np.abs(d["weight"]), 0.0), minlength=len(LAYERS) ** 2).reshape(shape)
    flow_pos = np.bincount(key, weights=(~uedge & (d["weight"] > 0)).astype(np.int64), minlength=len(LAYERS) ** 2).reshape(shape)
    flow_neg = np.bincount(key, weights=(~uedge & (d["weight"] < 0)).astype(np.int64), minlength=len(LAYERS) ** 2).reshape(shape)
    flow_unc = np.bincount(key, weights=uedge.astype(np.int64), minlength=len(LAYERS) ** 2).reshape(shape)
    matrices = {"flow_edge_count": flow_edges.astype(np.int64), "flow_contact_count": flow_contacts, "flow_signed_weight": flow_signed, "flow_absolute_weight": flow_abs, "flow_uncertain_absolute_weight": flow_unc_abs, "flow_positive_edge_count": flow_pos.astype(np.int64), "flow_negative_edge_count": flow_neg.astype(np.int64), "flow_uncertain_edge_count": flow_unc.astype(np.int64)}
    cells = []
    for i, a in enumerate(LAYERS):
        for j, b in enumerate(LAYERS):
            total = float(flow_abs[i, j])
            cells.append({"source_layer": a, "target_layer": b, "edge_count": int(flow_edges[i, j]), "synaptic_contact_count": int(round(flow_contacts[i, j])), "positive_edge_count": int(flow_pos[i, j]), "negative_edge_count": int(flow_neg[i, j]), "uncertain_sign_edge_count": int(flow_unc[i, j]), "signed_baseline_weight": float(flow_signed[i, j]), "absolute_baseline_weight": total, "uncertain_sign_absolute_weight": float(flow_unc_abs[i, j]), "uncertain_sign_absolute_weight_fraction": float(flow_unc_abs[i, j] / total) if total else 0.0})
    flow = {"layer_order": list(LAYERS), "cells": cells}
    decompositions = {name: _target_input(name, mask, src, d["post"], layer, d["uncertain"], d["sign"], d["cell_type"], d["weight"], d["contacts"]) for name, mask in masks.items()}

    # One reverse CSR serves both MBON targets; this keeps all edges and does
    # not consult weights or contacts for reachability.
    order = np.argsort(d["post"], kind="stable")
    rev_ptr = np.r_[0, np.cumsum(np.bincount(d["post"][order], minlength=n))].astype(np.int64)
    rev_post = src[order]
    reverse = {name: _bfs_dist(rev_ptr, rev_post, masks[name]) for name in ("MBON07", "MBON11")}
    candidates = ("graded_candidate_photoreceptor", "graded_candidate_lamina", "graded_candidate_apl")
    hist = np.zeros((3, len(PATH_TARGETS), 8), dtype=np.int64)
    reach = {}
    kc_indices = np.flatnonzero(masks["KC"])
    for ci, cname in enumerate(candidates):
        dist = _bfs_dist(d["ptr"], d["post"], layer == LAYER_INDEX[cname])
        for ti, tname in enumerate(PATH_TARGETS):
            hist[ci, ti] = _histogram(dist, masks[tname])
        kc_dist = dist[kc_indices]; kc_reached = (kc_dist >= 0) & (kc_dist <= 6); count = int(kc_reached.sum())
        further = {t: int(np.sum(kc_reached & (reverse[t][kc_indices] >= 0) & (reverse[t][kc_indices] <= 6))) for t in ("MBON07", "MBON11")}
        reach[cname] = {"candidate_node_count": int((layer == LAYER_INDEX[cname]).sum()), "kc_count": int(len(kc_indices)), "kc_reached_within_6_hops": count, "kc_reached_fraction": float(count / len(kc_indices)) if len(kc_indices) else 0.0, "kc_reached_further_to_mbon_within_6_hops": further, "kc_reached_further_fraction": {t: float(further[t] / count) if count else 0.0 for t in further}}
    reverse_hist = np.asarray([_histogram(reverse[t], masks["KC"]) for t in ("MBON07", "MBON11")], dtype=np.int64)
    cons = {"flow_edge_count": int(flow_edges.sum()), "flow_contact_count": int(round(flow_contacts.sum())), "flow_signed_weight": float(flow_signed.sum()), "flow_absolute_weight": float(flow_abs.sum()), "graph_edge_count": int(e), "graph_contact_count": int(d["contacts"].sum(dtype=np.uint64)), "graph_signed_weight": float(d["weight"].sum(dtype=np.float64)), "graph_absolute_weight": float(np.abs(d["weight"]).sum(dtype=np.float64))}
    # The union is the formal multi-source gate; per-group results above remain
    # descriptive and are deliberately retained in the output.
    graded_union = np.isin(layer, [LAYER_INDEX[x] for x in candidates])
    union_dist = _bfs_dist(d["ptr"], d["post"], graded_union)
    union_kc_dist = union_dist[kc_indices]
    union_kc_reached = (union_kc_dist >= 0) & (union_kc_dist <= 6)
    union_further = {t: int(np.sum((reverse[t][kc_indices] >= 0) & (reverse[t][kc_indices] <= 6))) for t in ("MBON07", "MBON11")}
    path_analysis = {"bucket_labels": list(BUCKET_LABELS), "candidate_order": list(candidates), "target_order": list(PATH_TARGETS), "candidate_to_target_histogram": hist.tolist(), "candidate_kc_reachability": reach, "graded_union_to_target_histogram": np.asarray([_histogram(union_dist, masks[t]) for t in PATH_TARGETS], dtype=np.int64).tolist(), "graded_union_kc_reachability": {"candidate_node_count": int(graded_union.sum()), "kc_count": int(len(kc_indices)), "kc_reached_within_6_hops": int(union_kc_reached.sum()), "kc_reached_fraction": float(union_kc_reached.mean()) if len(kc_indices) else 0.0}, "kc_to_mbon_reverse_histogram": reverse_hist.tolist(), "kc_to_mbon_order": ["MBON07", "MBON11"], "kc_union_to_mbon_within_6_hops": {t: {"kc_reached_within_6_hops": union_further[t], "reachable": bool(union_further[t] > 0)} for t in ("MBON07", "MBON11")}}
    impl = {"node_count": int(n), "edge_count": int(e), "contact_count": cons["graph_contact_count"], "all_nodes_assigned_once": bool(np.bincount(layer, minlength=len(LAYERS)).sum() == n), "layer_inventory": _layer_inventory(d), "flow_matrix": flow, "target_input_decomposition": decompositions, "path_analysis": path_analysis, "conservation": cons}
    matrices = matrices | {"graded_union_to_target_histogram": np.asarray(path_analysis["graded_union_to_target_histogram"], dtype=np.int64), "kc_union_to_mbon_within_6_hops": np.asarray([union_further[t] for t in ("MBON07", "MBON11")], dtype=np.int64)}
    return impl, matrices


def _all_fractions(value: Any, key: str = "") -> bool:
    """Validate every named fraction/proportion, including its [0,1] bound."""
    named = any(token in key.lower() for token in ("fraction", "proportion"))
    if isinstance(value, dict):
        return all(_all_fractions(v, str(k)) for k, v in value.items())
    if isinstance(value, list):
        return all(_all_fractions(v, key) for v in value)
    if named:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            return False
        return bool(np.isfinite(value) and 0.0 <= value <= 1.0)
    return True


def qualification(impl: dict[str, Any]) -> dict[str, Any]:
    reach = impl["path_analysis"]["candidate_kc_reachability"]
    coverage = {k: v["kc_reached_fraction"] for k, v in reach.items()}
    union_coverage = float(impl["path_analysis"]["graded_union_kc_reachability"]["kc_reached_fraction"])
    nonempty = all(v["candidate_node_count"] > 0 for v in reach.values())
    coverage_ok = union_coverage >= 0.95
    target_ok = all(impl["path_analysis"]["kc_union_to_mbon_within_6_hops"][t]["reachable"] for t in ("MBON07", "MBON11"))
    decomp = impl["target_input_decomposition"]
    uncertainty = {t: float(decomp[t]["uncertain_sign_absolute_weight_fraction"]) for t in ("KC", "APL", "MBON07", "MBON11")}
    cons = impl["conservation"]
    impl_ok = bool(impl["all_nodes_assigned_once"] and cons["flow_edge_count"] == cons["graph_edge_count"] and cons["flow_contact_count"] == cons["graph_contact_count"] and np.isclose(cons["flow_signed_weight"], cons["graph_signed_weight"], rtol=0.0, atol=1e-12) and np.isclose(cons["flow_absolute_weight"], cons["graph_absolute_weight"], rtol=0.0, atol=1e-12) and _all_fractions(impl))
    if not impl_ok:
        verdict = "P0_IMPLEMENTATION_FAILURE"
    elif not (nonempty and coverage_ok and target_ok):
        verdict = "BLOCKED_BY_MODEL_IDENTIFIABILITY"
    elif any(v >= 0.20 for v in uncertainty.values()):
        verdict = "READY_FOR_V2_Q1_WITH_SIGN_SENSITIVITY"
    else:
        verdict = "READY_FOR_V2_Q1"
    return {"verdict": verdict, "implementation_gate": {"passed": impl_ok}, "model_gate": {"all_graded_candidate_nonempty": nonempty, "kc_coverage_by_candidate": coverage, "graded_union_kc_coverage": union_coverage, "kc_coverage_at_least_95_percent": coverage_ok, "kc_to_each_mbon_reachable": target_ok, "uncertain_sign_absolute_weight_fraction": uncertainty, "uncertain_sign_trigger": any(v >= 0.20 for v in uncertainty.values())}}


def history_lock() -> dict[str, Any]:
    locked = {}
    for name, path in HISTORY_JSON.items():
        d = json.loads(path.read_text())
        verdict = "inconclusive" if name == "R0" else d["metrics_and_hypotheses"]["verdict"] if name == "R0b" else d["metrics"]["comparison"]["verdict"]
        item = {"formal_verdict": verdict, "json_sha256": sha256_file(path), "formal": d.get("formal"), "success": d.get("success"), "production_eligible": d.get("production_eligible")}
        if name == "R0c":
            item["dt_values_ms"] = d["dt_values_ms"]
            item["by_dt_empirical_first_arrival_hazard_bins"] = {str(dt): d["metrics"]["by_dt"][str(dt)]["hazard_bins"] for dt in d["dt_values_ms"]}
            item["by_dt_ignition_fraction_2s"] = {str(dt): d["metrics"]["by_dt"][str(dt)]["ignition_fraction"] for dt in d["dt_values_ms"]}
            item["by_dt_post_ignition_cb_kc_persistent_fraction"] = {str(dt): d["metrics"]["by_dt"][str(dt)]["persistent_both_fraction"] for dt in d["dt_values_ms"]}
        locked[name] = item
    return locked


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--single-pass", action="store_true", help="test-only shortcut; formal invocation uses two passes")
    args = ap.parse_args(argv)
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    (out / "launch.txt").write_text(f"pid={os.getpid()}\nstarted_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\ncommand={' '.join(sys.argv)}\n")
    try:
        d = load_fixed_inputs()
        first, matrices = compute_inventory(d)
        second = first if args.single_pass else compute_inventory(d)[0]
        first_hash, second_hash = canonical_hash(first), canonical_hash(second)
        deterministic = first_hash == second_hash
        q = qualification(first)
        after = {rel: sha256_file(ROOT / rel) for rel in FROZEN_HASHES}
        unchanged = after == d["frozen_hashes"]
        scale = {"nodes": len(d["ids"]), "edges": len(d["post"]), "synaptic_contacts": int(d["contacts"].sum(dtype=np.uint64))}
        gates = {"all_frozen_hashes_match": True, "graph_scale_matches_preregistration": scale == EXPECTED_GRAPH_SCALE, "all_retained_nodes_assigned_once": first["all_nodes_assigned_once"], "edge_contact_weight_conservation": first["conservation"], "normalized_edge_alignment": True, "all_proportions_finite_in_0_1": _all_fractions(first), "deterministic_two_runs": deterministic, "source_weights_kernel_history_unchanged": unchanged}
        implementation_all = bool(q["implementation_gate"]["passed"] and gates["all_frozen_hashes_match"] and gates["graph_scale_matches_preregistration"] and gates["all_retained_nodes_assigned_once"] and gates["normalized_edge_alignment"] and gates["all_proportions_finite_in_0_1"] and gates["deterministic_two_runs"] and gates["source_weights_kernel_history_unchanged"])
        formal = {"schema": "v2-p0-model-inventory-v3", "preregistration_sha256": sha256_file(ROOT / "docs/20261008-v2-p0-model-inventory-preregistered.md"), "fixed_input_sha256": d["frozen_hashes"], "graph_scale": scale, "determinism": {"first_summary_sha256": first_hash, "second_summary_sha256": second_hash, "equal": deterministic}, "source_hashes_unchanged_after_run": unchanged, "history_lock": history_lock(), "inventory": first, "qualification": q, "implementation_gates": gates | {"all_pass": implementation_all}, "limitations": ["P0 is structural and descriptive only; it does not implement v2 dynamics.", "Reachability is not functional drive, sensory encoding, or behavioral causality.", "R0/R0b/R0c values are locked historical evidence and do not establish randomness, biological absorption, visual coding, learning, or profitable trading.", "Central brain target is the complete set of retained superclasses beginning with cb_; it is reported separately and is not called whole brain."], "wall_time_seconds": round(time.time() - start, 3), "output_path": str(out)}
        (out / "v2_p0_formal.json").write_text(json.dumps(formal, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        np.savez_compressed(out / "v2_p0_matrices.npz", **matrices)
        report = ["# V2-P0 whole-brain model inventory", "", f"判定：`{q['verdict']}`", "", "本产物只做完整 retained graph 的结构清单、流量守恒、输入来源分解、6-hop 可达性和历史证据锁定；未运行市场、收益、动作、标签、神经动力学或塑性。", "", "## 固定规模", "", f"节点 {len(d['ids']):,}；directed edges {len(d['post']):,}；synaptic contacts {int(d['contacts'].sum()):,}。", "", "## 门禁摘要", "", json.dumps(formal["implementation_gates"], ensure_ascii=False, indent=2), "", "## 模型资格", "", json.dumps(q, ensure_ascii=False, indent=2), "", "## 历史证据", "", json.dumps(formal["history_lock"], ensure_ascii=False, indent=2), "", "## 限制", "", "- 结构可达性不等于功能驱动或因果控制。", "- 不得把 READY 解读为 v2 已验证或已证明盈利学习。", "- 中央脑报告只对应 `cb_` superclass 集合，不外推为全脑。", ""]
        (out / "v2_p0_report.md").write_text("\n".join(report))
        (out / "formal.log").write_text(f"v2-p0 verdict={q['verdict']}\nsummary_sha256={first_hash}\ndeterministic={deterministic}\nwall_time_seconds={formal['wall_time_seconds']}\n")
        return 0 if implementation_all else 2
    except Exception as exc:
        failure = {"schema": "v2-p0-model-inventory-v3", "status": "P0_IMPLEMENTATION_FAILURE", "error": repr(exc), "fixed_input_sha256": {rel: sha256_file(ROOT / rel) if (ROOT / rel).exists() else None for rel in FROZEN_HASHES}, "wall_time_seconds": round(time.time() - start, 3)}
        (out / "v2_p0_formal_failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        (out / "formal.log").write_text("P0_IMPLEMENTATION_FAILURE\n" + repr(exc) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
