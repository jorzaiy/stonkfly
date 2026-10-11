"""Read-only S2d projection threshold diagnostic on the complete retained graph."""
import argparse
import hashlib
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stonkfly.config import Settings
from stonkfly.neural.common import annotations
from stonkfly.neural.v51_controller import V51Controller
from experiments.a_share.analyze_kc_overlap import render_frame_from_observation

LOW_MAX, HIGH_MIN = 10, 1000
UPSTREAM = ("r1_r6", "r8", "lamina", "ol_sensory", "ol_intrinsic", "visual_projection", "cb_intrinsic")


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_events(run_path):
    with (Path(run_path) / "events.jsonl").open() as f:
        return [json.loads(line) for line in f]


def sample_event_indices(events, sample_size=60, seed=42):
    """Exact category-balanced sampling used by S2c."""
    groups = {"reward": [], "punishment": [], "neutral": []}
    for i, event in enumerate(events):
        stim = event.get("reward", {}).get("stimulus", "none")
        groups["reward" if stim == "reward" else "punishment" if stim == "aversive" else "neutral"].append(i)
    rng = np.random.default_rng(seed)
    def choose(items, n):
        return items if len(items) <= n else list(rng.choice(items, size=n, replace=False))
    nr = min(sample_size // 3, len(groups["reward"]))
    npun = min(sample_size // 3, len(groups["punishment"]))
    nn = min(sample_size - nr - npun, len(groups["neutral"]))
    # numpy.random.choice returns np.int64; normalize at this boundary so the
    # result is safe for strict JSON serialization and downstream provenance.
    return [int(i) for i in sorted(set(choose(groups["reward"], nr) + choose(groups["punishment"], npun) + choose(groups["neutral"], nn)))]


def classify(active_kc_count):
    return "low" if active_kc_count <= LOW_MAX else "high" if active_kc_count >= HIGH_MIN else "mid"


def jaccard(left, right):
    left, right = set(left), set(right)
    return 1.0 if not left and not right else 0.0 if not left or not right else len(left & right) / len(left | right)


def summary(values):
    x = np.asarray(values, dtype=float)
    return {"n": int(len(x)), "mean": float(x.mean()) if len(x) else None, "median": float(np.median(x)) if len(x) else None, "min": float(x.min()) if len(x) else None, "max": float(x.max()) if len(x) else None}


def zero_error_thresholds(values, labels):
    """Strict sample-only threshold search; labels must be low/high."""
    lo = [float(v) for v, label in zip(values, labels) if label == "low"]
    hi = [float(v) for v, label in zip(values, labels) if label == "high"]
    if not lo or not hi:
        return []
    out = []
    if max(lo) < min(hi):
        out.append({"direction": "value>=threshold_implies_high", "threshold": (max(lo) + min(hi)) / 2, "low_max": max(lo), "high_min": min(hi), "errors": 0, "sample_only": True})
    if max(hi) < min(lo):
        out.append({"direction": "value<=threshold_implies_high", "threshold": (max(hi) + min(lo)) / 2, "high_max": max(hi), "low_min": min(lo), "errors": 0, "sample_only": True})
    return out


def select_projection_activity_boundary(records):
    """Pick low/high frames at the projection-activity boundary, not KC count."""
    low = [r for r in records if r.get("group") == "low"]
    high = [r for r in records if r.get("group") == "high"]
    return (max(low, key=lambda r: (r["projection_active_count"], -r["event_index"]), default=None),
            min(high, key=lambda r: (r["projection_active_count"], r["event_index"]), default=None))


def _outgoing(brain, source):
    sl = slice(int(brain.ptr[source]), int(brain.ptr[source + 1]))
    return brain.post[sl], brain.weight[sl]


def trace_projection_edges(brain, active_projection, cb_set, kc_set):
    """Trace directed CSR edges; weights are signed scaled synapse-count weights."""
    direct_cb, direct_kc, twohop_kc = set(), set(), set()
    cb_edges = kc_edges = twohop_paths = 0
    cb_weight = kc_weight = twohop_weight = 0.0
    for p in active_projection:
        posts, weights = _outgoing(brain, int(p))
        for q, weight in zip(posts, weights):
            q, weight = int(q), float(weight)
            if q in cb_set:
                direct_cb.add(q); cb_edges += 1; cb_weight += weight
                qposts, qweights = _outgoing(brain, q)
                for k, qweight in zip(qposts, qweights):
                    k = int(k)
                    if k in kc_set:
                        twohop_kc.add(k); twohop_paths += 1; twohop_weight += weight * float(qweight)
            if q in kc_set:
                direct_kc.add(q); kc_edges += 1; kc_weight += weight
    return {
        "edge_semantics": "CSR source->post directed retained edges; weight=signed synapse_count*0.275 kernel weight, not current",
        "direct_1hop_projection_to_cb": {"node_count": len(direct_cb), "edge_count": cb_edges, "weight_sum": cb_weight},
        "direct_1hop_projection_to_kc": {"node_count": len(direct_kc), "edge_count": kc_edges, "weight_sum": kc_weight},
        "two_hop_projection_to_cb_to_kc": {"node_count": len(twohop_kc), "path_count": twohop_paths, "product_weight_sum": twohop_weight, "hop_semantics": "projection -> cb_intrinsic -> KC"},
        "reachable_cb_body_ids": sorted(int(brain.ids[i]) for i in direct_cb),
        "reachable_kc_direct_body_ids": sorted(int(brain.ids[i]) for i in direct_kc),
        "reachable_kc_two_hop_body_ids": sorted(int(brain.ids[i]) for i in twohop_kc),
    }


def analyze(run_path, output, sample_size=60, seed=42, event_indices=None):
    run_path, output = Path(run_path), Path(output)
    events = load_events(run_path)
    indices = [int(i) for i in sorted(event_indices if event_indices is not None else sample_event_indices(events, sample_size, seed))]
    from stonkfly.neural.common import GRAPH
    graph_path = Path(GRAPH).resolve()
    if not graph_path.exists():
        raise FileNotFoundError(f"V51Controller graph artifact is missing: {graph_path}")
    settings = Settings(neural_ms=500, pulse_ms=200, learning=False)
    controller = V51Controller(settings)
    brain = controller.brain
    ann = annotations(brain.ids)
    types = ann.type.fillna("").astype(str).to_numpy()
    superclasses = np.asarray(brain.superclass).astype(str)
    layer_indices = {name: np.flatnonzero(superclasses == name) for name in UPSTREAM}
    projection_indices = layer_indices["visual_projection"]
    cb_indices = layer_indices["cb_intrinsic"]
    kc_indices = np.asarray(brain.circuit["kc"], dtype=np.int64)
    cb_set, kc_set = set(map(int, cb_indices)), set(map(int, kc_indices))
    meta = {int(i): {"body_id": int(brain.ids[i]), "type": types[i], "superclass": superclasses[i]} for i in projection_indices}
    records, frame_hashes = [], {}
    with tempfile.TemporaryDirectory(prefix="stonkfly-projection-threshold-") as temp:
        checkpoint = Path(temp) / "brain-clean.npz"
        controller.save(checkpoint)
        repeatability = None
        for event_index in indices:
            event = events[event_index]
            frame = render_frame_from_observation(event["observation"])
            actual = hashlib.sha256(frame.tobytes()).hexdigest()
            if actual != event["neural"]["input_sha256"]:
                raise RuntimeError(f"event {event_index}: production frame hash mismatch")
            frame_hashes[event_index] = actual
            controller.restore(checkpoint)
            result = controller.observe(frame, reinforcement="none")
            counts = controller.brain.counts
            kc_counts = counts[kc_indices]
            active_projection = projection_indices[np.flatnonzero(counts[projection_indices] > 0)]
            kc_active_count = int(np.count_nonzero(kc_counts > 0))
            if repeatability is None:
                controller.restore(checkpoint)
                repeated = controller.observe(frame, reinforcement="none")
                repeatability = {"spike_sha256_match": result["spike_sha256"] == repeated["spike_sha256"], "kc_counts_match": bool(np.array_equal(kc_counts, controller.brain.counts[kc_indices]))}
                if not all(repeatability.values()):
                    raise RuntimeError("clean checkpoint repeatability failed")
            stimulus = event.get("reward", {}).get("stimulus", "none")
            category = "reward" if stimulus == "reward" else "punishment" if stimulus == "aversive" else "neutral"
            records.append({"event_index": int(event_index), "date": str(event["date"]), "category": category, "frame_sha256": actual, "active_kc_count": kc_active_count, "group": classify(kc_active_count), "active_projection_neurons": [{**meta[int(i)], "index": int(i), "spike_count": int(counts[i])} for i in active_projection], "projection_active_count": int(len(active_projection)), "edge_trace": trace_projection_edges(brain, active_projection, cb_set, kc_set)})
    usable = [r for r in records if r["group"] in ("low", "high")]
    labels = [r["group"] for r in usable]
    low_n, high_n = labels.count("low"), labels.count("high")
    by_body = defaultdict(lambda: {"meta": None, "low_present": 0, "high_present": 0, "low_spikes": [], "high_spikes": []})
    for frame in usable:
        active = {n["body_id"]: n for n in frame["active_projection_neurons"]}
        for info in meta.values():
            body = info["body_id"]; rec = by_body[body]; rec["meta"] = info; item = active.get(body)
            rec[f"{frame['group']}_present"] += int(item is not None); rec[f"{frame['group']}_spikes"].append(item["spike_count"] if item else 0)
    projection_summary = {}
    for body, rec in sorted(by_body.items()):
        lr, hr = rec["low_present"] / low_n, rec["high_present"] / high_n
        projection_summary[str(body)] = {**rec["meta"], "low_occurrence_rate": lr, "high_occurrence_rate": hr, "high_low_occurrence_ratio": None if lr == 0 else hr / lr, "low_spikes": summary(rec["low_spikes"]), "high_spikes": summary(rec["high_spikes"]), "only_high": rec["low_present"] == 0 and rec["high_present"] > 0, "sample_only_zero_error_spike_thresholds": zero_error_thresholds(rec["low_spikes"] + rec["high_spikes"], ["low"] * low_n + ["high"] * high_n)}
    proxy_paths = {"direct_1hop_projection_to_cb.node_count": ("direct_1hop_projection_to_cb", "node_count"), "direct_1hop_projection_to_kc.node_count": ("direct_1hop_projection_to_kc", "node_count"), "two_hop_projection_to_cb_to_kc.node_count": ("two_hop_projection_to_cb_to_kc", "node_count"), "two_hop_projection_to_cb_to_kc.path_count": ("two_hop_projection_to_cb_to_kc", "path_count"), "two_hop_projection_to_cb_to_kc.product_weight_sum": ("two_hop_projection_to_cb_to_kc", "product_weight_sum")}
    proxy_thresholds = {name: zero_error_thresholds([r["edge_trace"][path][field] for r in usable], labels) for name, (path, field) in proxy_paths.items()}
    lo, hi = select_projection_activity_boundary(usable)
    boundary = None
    if lo and hi:
        left = {n["body_id"] for n in lo["active_projection_neurons"]}; right = {n["body_id"] for n in hi["active_projection_neurons"]}
        boundary = {"selection": "projection-activity boundary: low has maximum projection_active_count; high has minimum projection_active_count", "low_projection_activity_max": {"event_index": lo["event_index"], "active_kc_count": lo["active_kc_count"], "projection_active_count": lo["projection_active_count"], "projection_body_ids": sorted(left)}, "high_projection_activity_min": {"event_index": hi["event_index"], "active_kc_count": hi["active_kc_count"], "projection_active_count": hi["projection_active_count"], "projection_body_ids": sorted(right)}, "jaccard": jaccard(left, right)}
    result = {"provenance": {"run_path": str(run_path), "events_sha256": sha256_file(run_path / "events.jsonl"), "script_sha256": sha256_file(Path(__file__)), "graph_sha256": sha256_file(graph_path), "s2c_source": "experiments/a_share/analyze_kc_overlap.py", "frame_hashes_verified": len(frame_hashes), "repeatability": repeatability}, "parameters": {"sample_size": sample_size, "seed": seed, "event_indices": indices, "low_definition": "active_kc_count <= 10", "high_definition": "active_kc_count >= 1000", "neural_ms": 500, "clean_checkpoint_restore_per_frame": True, "event_selection": "S2c exact category-balanced RNG unless --event-indices supplied"}, "graph_semantics": {"release": "MaleCNS v1.0", "retained_full_graph": True, "node_id": "brain.ids/body_id stable ID", "edge_direction": "CSR source -> post target", "edge_weight": "signed synapse_count*0.275 kernel weight; two-hop product is structural proxy, not current", "causal_claim": False}, "group_counts": dict(Counter(labels)), "per_frame": records, "projection_neurons": projection_summary, "structure_proxy_zero_error_thresholds": proxy_thresholds, "near_boundary_set_comparison": boundary, "limitations": ["All thresholds are sample-only; no generalization claim.", "Reachability and signed weight products are structural proxies, not causal currents.", "Association/correlation is not causation.", "Mid-group frames are excluded from low/high claims."]}
    result["provenance"]["graph_path"] = str(graph_path)
    result["provenance"]["graph_weight_scale"] = 0.275
    ranked = []
    for body, info in projection_summary.items():
        diff = info["high_occurrence_rate"] - info["low_occurrence_rate"]
        ranked.append({"body_id": int(body), "type": info["type"], "superclass": info["superclass"], "only_high": info["only_high"], "occurrence_rate_difference_high_minus_low": diff, "high_low_occurrence_ratio": info["high_low_occurrence_ratio"], "low_spike_mean": info["low_spikes"]["mean"], "high_spike_mean": info["high_spikes"]["mean"], "sample_only": True, "interpretation": "descriptive enrichment, not causal"})
    ranked.sort(key=lambda x: (not x["only_high"], -x["occurrence_rate_difference_high_minus_low"], -(x["high_low_occurrence_ratio"] if x["high_low_occurrence_ratio"] is not None else float("inf"))))
    result["candidate_rankings"] = {"high_only_top": [x for x in ranked if x["only_high"]][:20], "high_enriched_top": ranked[:20], "basis": "occurrence difference then high/low ratio; low=0 is represented as null ratio", "sample_only": True}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", type=Path)
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event-indices", type=int, nargs="*", help="explicit smoke-test event indices")
    args = parser.parse_args()
    result = analyze(args.run_path, args.output, args.sample_size, args.seed, args.event_indices)
    print(json.dumps({"output": str(args.output), "frames": len(result["per_frame"]), "groups": result["group_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
