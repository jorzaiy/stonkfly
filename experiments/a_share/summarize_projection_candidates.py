"""No-simulation candidate connectivity review for a completed S2d result."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
from stonkfly.neural.common import annotations


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def trace_candidate(index, brain, cb_set, kc_set, preview_limit=200):
    direct_cb, direct_kc, twohop_kc = set(), set(), set()
    cb_edges = kc_edges = paths = 0
    cb_weight = kc_weight = product_weight = 0.0
    for start in (index,):
        sl = slice(int(brain["ptr"][start]), int(brain["ptr"][start + 1]))
        for target, weight in zip(brain["post"][sl], brain["weight"][sl]):
            target, weight = int(target), float(weight)
            if target in cb_set:
                direct_cb.add(target); cb_edges += 1; cb_weight += weight
                qsl = slice(int(brain["ptr"][target]), int(brain["ptr"][target + 1]))
                for kc, qweight in zip(brain["post"][qsl], brain["weight"][qsl]):
                    kc = int(kc)
                    if kc in kc_set:
                        twohop_kc.add(kc); paths += 1; product_weight += weight * float(qweight)
            if target in kc_set:
                direct_kc.add(target); kc_edges += 1; kc_weight += weight
    def ids(nodes):
        values = sorted(int(brain["ids"][i]) for i in nodes)
        return {"node_count": len(values), "preview_body_ids": values[:preview_limit], "preview_truncated": len(values) > preview_limit}
    return {"direct_projection_to_cb": {**ids(direct_cb), "edge_count": int(cb_edges), "signed_scaled_weight_sum": float(cb_weight)}, "direct_projection_to_kc": {**ids(direct_kc), "edge_count": int(kc_edges), "signed_scaled_weight_sum": float(kc_weight)}, "two_hop_projection_to_cb_to_kc": {**ids(twohop_kc), "path_count": int(paths), "signed_scaled_weight_product_sum": float(product_weight), "hop_semantics": "projection -> cb_intrinsic -> KC"}, "_full_sets": {"cb": {int(brain["ids"][i]) for i in direct_cb}, "kc_two_hop": {int(brain["ids"][i]) for i in twohop_kc}}}


def stats(values):
    values = [float(v) for v in values]
    return {"n": len(values), "mean": statistics.fmean(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    graph_path = Path(result["provenance"]["graph_path"])
    if not graph_path.exists():
        raise FileNotFoundError(graph_path)
    graph = dict(np.load(graph_path))
    ids = np.asarray(graph["ids"], dtype=np.int64)
    body_to_index = {int(body): i for i, body in enumerate(ids)}
    superclass = np.asarray(graph["superclass"]).astype(str)
    ann = annotations(ids)
    types = ann.type.fillna("").astype(str).to_numpy()
    cb_set = set(np.flatnonzero(superclass == "cb_intrinsic"))
    kc_set = {int(i) for i, cell_type in enumerate(types) if str(cell_type).startswith("KC")}
    boundary = result["near_boundary_set_comparison"]
    low_ids = set(boundary["low_projection_activity_max"]["projection_body_ids"])
    high_ids = set(boundary["high_projection_activity_min"]["projection_body_ids"])
    perfect = [int(body) for body, info in result["projection_neurons"].items() if info["low_occurrence_rate"] == 0 and info["high_occurrence_rate"] == 1]
    high_new, low_unique = sorted(high_ids - low_ids), sorted(low_ids - high_ids)
    candidate_ids = sorted(set(perfect) | set(high_new) | set(low_unique))
    connectivity = {}
    full_candidate_sets = {}
    for body in candidate_ids:
        index = body_to_index[body]
        trace = trace_candidate(index, graph, cb_set, kc_set)
        full_candidate_sets[body] = trace.pop("_full_sets")
        connectivity[str(body)] = {"body_id": body, "type": str(types[index]), "superclass": str(superclass[index]), "connectivity": trace}
    overlap = {}
    cb_sets = {b: set(v["connectivity"]["direct_projection_to_cb"]["preview_body_ids"]) for b, v in connectivity.items()}
    kc_sets = {b: set(v["connectivity"]["two_hop_projection_to_cb_to_kc"]["preview_body_ids"]) for b, v in connectivity.items()}
    # Preview IDs are intentionally used only for compact overlap diagnostics;
    # union counts below use full graph node counts.
    all_cb = set().union(*cb_sets.values()) if cb_sets else set(); all_kc = set().union(*kc_sets.values()) if kc_sets else set()
    full_cb_union = set().union(*(x["cb"] for x in full_candidate_sets.values())) if full_candidate_sets else set()
    full_kc_union = set().union(*(x["kc_two_hop"] for x in full_candidate_sets.values())) if full_candidate_sets else set()
    def pairwise_jaccard(key):
        sets = [x[key] for x in full_candidate_sets.values()]
        values = []
        for i, left in enumerate(sets):
            for right in sets[i + 1:]:
                values.append(len(left & right) / len(left | right) if left or right else 1.0)
        return stats(values) if values else {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    overlap["candidate_preview_union"] = {"direct_cb_body_id_count": len(all_cb), "two_hop_kc_body_id_count": len(all_kc), "note": "preview-limited overlap; per-candidate counts are complete"}
    overlap["full_candidate_union_and_pairwise_overlap"] = {"direct_cb_union_body_id_count": len(full_cb_union), "two_hop_kc_union_body_id_count": len(full_kc_union), "direct_cb_pairwise_jaccard": pairwise_jaccard("cb"), "two_hop_kc_pairwise_jaccard": pairwise_jaccard("kc_two_hop")}
    ranked = []
    for body, item in connectivity.items():
        two = item["connectivity"]["two_hop_projection_to_cb_to_kc"]
        ranked.append({"body_id": int(body), "type": item["type"], "two_hop_kc_node_count": two["node_count"], "two_hop_path_count": two["path_count"], "abs_signed_scaled_weight_product_sum": abs(two["signed_scaled_weight_product_sum"]), "signed_scaled_weight_product_sum": two["signed_scaled_weight_product_sum"], "ranking_is_not_actual_drive": True})
    overlap["candidate_rankings"] = {"by_two_hop_kc_node_count": sorted(ranked, key=lambda x: (-x["two_hop_kc_node_count"], x["body_id"])), "by_two_hop_path_count": sorted(ranked, key=lambda x: (-x["two_hop_path_count"], x["body_id"])), "by_abs_signed_weight_product": sorted(ranked, key=lambda x: (-x["abs_signed_scaled_weight_product_sum"], x["body_id"]))}
    aggregate = {}
    for layer, field in (("direct_1hop_projection_to_cb", "node_count"), ("two_hop_projection_to_cb_to_kc", "path_count")):
        vals = {g: [r["edge_trace"][layer][field] for r in result["per_frame"] if r["group"] == g] for g in ("low", "high")}
        aggregate[f"{layer}.{field}"] = {g: stats(v) for g, v in vals.items()}
    low_direct = aggregate["direct_1hop_projection_to_cb.node_count"]["low"]
    high_direct = aggregate["direct_1hop_projection_to_cb.node_count"]["high"]
    low_path = aggregate["two_hop_projection_to_cb_to_kc.path_count"]["low"]
    high_path = aggregate["two_hop_projection_to_cb_to_kc.path_count"]["high"]
    output = {"provenance": {"source_result": str(args.result), "source_result_sha256": sha256_file(args.result), "graph_path": str(graph_path), "graph_sha256": sha256_file(graph_path), "script_sha256": sha256_file(Path(__file__))}, "semantics": {"edge_direction": "CSR source -> post target", "weight": "signed synapse_count*0.275 kernel weight", "two_hop": "projection -> cb_intrinsic -> KC structural reachability/path proxy, not current or causal drive", "sample_only": True}, "candidate_sets": {"perfect_presence_candidates": [{"body_id": b, "type": str(types[body_to_index[b]])} for b in perfect], "boundary_high_new": [{"body_id": b, "type": str(types[body_to_index[b]])} for b in high_new], "boundary_low_unique": [{"body_id": b, "type": str(types[body_to_index[b]])} for b in low_unique]}, "candidate_connectivity": connectivity, "candidate_overlap_and_rankings": overlap, "aggregate_edge_trace": aggregate, "verified_sample_only_thresholds": {"direct_cb_node_count": {"threshold": (low_direct["max"] + high_direct["min"]) / 2, "low_max": low_direct["max"], "high_min": high_direct["min"], "margin": high_direct["min"] - low_direct["max"], "errors": 0}, "two_hop_path_count": {"threshold": (low_path["max"] + high_path["min"]) / 2, "low_max": low_path["max"], "high_min": high_path["min"], "margin": high_path["min"] - low_path["max"], "errors": 0}}, "presence_separator_review": {"perfect_presence_candidate_count": len(perfect), "all_are_sample_only": True, "descriptive_presence_separator": "Each perfect-presence candidate is absent in all low frames and present in all high frames in this sample; this is not a causal switch or generalization claim."}, "limitations": ["Reachability and signed weight products are structural proxies, not actual drive/current.", "Preview ID lists are capped at 200 per candidate; counts are complete.", "All separation and rankings are sample-only descriptive results."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "candidate_count": len(candidate_ids), "perfect_presence_count": len(perfect)}))


if __name__ == "__main__":
    main()
