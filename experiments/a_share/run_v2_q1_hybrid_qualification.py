#!/usr/bin/env python3
"""Frozen V2-Q1 hybrid-kernel qualification runner.

This experiment is structural/dynamical only.  It never imports market,
reward, plasticity, controller, or decoder modules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/whole_brain_v2/Q1/20261009T-hybrid-kernel-qualification-v1"
P0 = ROOT / "runs/whole_brain_v2/P0/20261008T-model-inventory-v3"
P0_FORMAL = P0 / "v2_p0_formal.json"
P0_MATRICES = P0 / "v2_p0_matrices.npz"
FIXED = {
    "data/graph.npz": "346b8af85a11af13b8324e18669812c1924569e7d1adcb4e6f45cc461a2c344b",
    "data/normalized/neurons.feather": "2103b5203d2ba5502c06fb70a4e0b23edcbcf79ed0c344722101941fc289ba2d",
    "data/normalized/edges.arrow": "842b762c9427c794eaedd453178aa034727a5084850a91fdcc4d3fffe09f5b7e",
    "stonkfly/neural/kernel.cpp": "aea2b97d2f9d78c57787f40f42a9d7dd777c3059d8c7f453e61c4d94398438d4",
    "stonkfly/neural/brain.py": "753b04324c274c9b522a0aa2459a81edc1056e4c8f6af34c332b3df371ace4b4",
    "stonkfly/neural/state.py": "4356965ea5fe70ef3ab323b3b05753dde4cf4ef7788506ef38eb29609955bbee",
    "runs/whole_brain_v2/P0/20261008T-model-inventory-v3/v2_p0_formal.json": "7fa88023cc09a12955174ab76f8cebb5a05ddf9c258867511187440bad69b45f",
    "runs/whole_brain_v2/P0/20261008T-model-inventory-v3/v2_p0_matrices.npz": "bf1ff3298886ac274337f9c7f3c89cace57158a504a643ab2c8c19269bcbe0eb",
    "docs/20261008-v2-p0-model-inventory-preregistered.md": "3d845207b394b67646470d16894a37c1afb2a61365e602135d7a4c2f88d2cc18",
    "docs/20261008-v2-p0-v1-v2-implementation-amendment.md": "8f05ba4f2dbfd43a6c052068473b1278f111e59e45408ab4998472f02082afb7",
}
SCALE = {"nodes": 166700, "edges": 25582938, "synaptic_contacts": 124177617}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def jhash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fixed_inputs():
    actual = {k: sha(ROOT / k) for k in FIXED}
    bad = {k: (v, actual[k]) for k, v in FIXED.items() if v != actual[k]}
    if bad: raise RuntimeError("frozen input hash mismatch: " + json.dumps(bad, sort_keys=True))
    p0 = json.loads(P0_FORMAL.read_text())
    if p0.get("qualification", {}).get("verdict") != "READY_FOR_V2_Q1":
        raise RuntimeError("P0 is not READY_FOR_V2_Q1")
    return actual, p0


def classify(nodes):
    from stonkfly.neural.v2_brain import classify_nodes
    layer, names = classify_nodes(nodes["cell_type"], nodes["neurotransmitter"])
    return layer, names


def source_hash(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<u8").tobytes()).hexdigest()


def micro_gates():
    from stonkfly.neural.v2_brain import exact_segment, release_from_voltage
    vals = np.array([-60., -52., -50.25, -48., -45., -40.])
    expected = np.clip((vals + 52.) / 7., 0., 1.)
    release_ok = bool(np.max(np.abs(release_from_voltage(vals) - expected)) <= 1e-12)
    rec_ok = True
    for dt in (0.1, 0.05, 0.025):
        v, gg, ge = -51.0, 0.2, 0.3
        for _ in range(round(2.0 / dt)):
            v, gg, ge = exact_segment(v, ge, gg, 1.5, 0.25, dt)
        if not all(np.isfinite(x) for x in (v, gg, ge)): rec_ok = False
    return {"release_clamp": release_ok, "analytic_segment": rec_ok,
            "delay_steps": {str(dt): round(1.8 / dt) for dt in (0.1, .05, .025)},
            "refractory_steps": {str(dt): round(2.2 / dt) for dt in (0.1, .05, .025)}}


def graph_fixture():
    a = np.load(ROOT / "data/graph.npz", allow_pickle=False)
    scale = {"nodes": int(len(a["ids"])), "edges": int(len(a["post"])), "synaptic_contacts": SCALE["synaptic_contacts"]}
    if scale != SCALE: raise RuntimeError(f"graph scale mismatch {scale}")
    return a


def trial(dt, graded_mask, ids, name, selected, kc_mask):
    from stonkfly.neural.v2_brain import HybridBrain
    a = np.load(ROOT / "data/graph.npz", allow_pickle=False)
    rest = np.full(len(ids), -52., dtype=np.float32)
    # v1 rest is retained for KC compatibility while all graded candidates
    # remain at the frozen non-KC rest.
    rest[np.asarray(kc_mask, dtype=bool)] = -60.
    b = HybridBrain(ROOT / "data/graph.npz", dt=dt, graded_mask=graded_mask, rest=rest)
    b.kc_mask[:] = np.asarray(kc_mask, dtype=np.uint8)
    zero = np.zeros(len(ids), dtype=np.float32)
    b.run(10., zero)
    stim = zero.copy(); stim[selected] = 3.5
    b.run(10., stim)
    b.run(20., zero)
    state = b.audit_state()
    finite = bool(all(np.isfinite(x).all() if isinstance(x, np.ndarray) and x.dtype.kind == "f" else True for x in [b.v, b.g_event, b.g_graded]))
    clean_zero = bool(name == "ZERO" and b.total_spikes == 0 and b.release_integral[0] == 0 and b.signed_delivery[0] == 0)
    return {"name": name, "dt_ms": dt, "selected_ids": [int(x) for x in ids[selected]],
            "selected_id_sha256": source_hash(ids[selected]), "spike_count": int(b.total_spikes),
            "release_integral": float(b.release_integral[0]), "signed_delivery_integral": float(b.signed_delivery[0]),
            "absolute_delivery_integral": float(b.absolute_delivery[0]), "finite": finite,
            "zero_equilibrium": clean_zero, "state_sha256": jhash({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in state.items()})}


def run(out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    fixed, p0 = fixed_inputs()
    a = graph_fixture()
    nodes = {k: np.asarray(v) for k, v in feather.read_table(ROOT / "data/normalized/neurons.feather").to_pydict().items()}
    layer, names = classify(nodes)
    masks = {"PHOTO128": layer == 1, "LAMINA128": layer == 2, "APL2": layer == 3}
    selections = {"PHOTO128": np.flatnonzero(masks["PHOTO128"])[:128], "LAMINA128": np.flatnonzero(masks["LAMINA128"])[:128], "APL2": np.flatnonzero(masks["APL2"])}
    graded = np.isin(layer, [1, 2, 3]).astype(np.uint8)
    classification = {n: {"count": int(np.sum(layer == i)), "source_id_sha256": source_hash(a["ids"][layer == i])} for i, n in enumerate(names)}
    classification["graded_union"] = {"count": int(graded.sum()), "source_id_sha256": source_hash(a["ids"][graded.astype(bool)])}
    gates = {"fixed_inputs": True, "p0_ready": True, "graph_scale": SCALE,
             "classification": classification, "micro": micro_gates(), "trials": []}
    for dt in (0.1, 0.05, 0.025):
        gates["trials"].append(trial(dt, graded, a["ids"], "ZERO", np.array([], dtype=np.int64), layer == 4))
        for name in ("PHOTO128", "LAMINA128", "APL2"):
            gates["trials"].append(trial(dt, graded, a["ids"], name, selections[name], layer == 4))
    after = {k: sha(ROOT / k) for k in FIXED}
    unchanged = fixed == after
    implementation = bool(unchanged and gates["micro"]["release_clamp"] and gates["micro"]["analytic_segment"] and all(x["finite"] for x in gates["trials"]))
    smoke = all(x["zero_equilibrium"] for x in gates["trials"] if x["name"] == "ZERO") and all(x["release_integral"] > 0 and x["absolute_delivery_integral"] > 0 for x in gates["trials"] if x["name"] != "ZERO")
    verdict = "Q1_IMPLEMENTATION_QUALIFIED" if implementation and smoke else ("Q1_IMPLEMENTATION_FAILURE" if not implementation else "Q1_FULL_GRAPH_SMOKE_FAILURE")
    build = json.loads((ROOT / "runs/whole_brain_v2/Q1/.build/libv2_hybrid.so.json").read_text()) if (ROOT / "runs/whole_brain_v2/Q1/.build/libv2_hybrid.so.json").exists() else {}
    formal = {"verdict": verdict, "fixed_input_sha256": fixed, "fixed_input_sha256_after": after,
              "source_hashes_unchanged_after_run": unchanged, "graph_scale": SCALE,
              "classification": classification, "gates": gates, "implementation_gate": implementation,
              "full_graph_smoke_gate": smoke, "v2_build": build, "p0_formal_sha256": sha(P0_FORMAL),
              "preregistration_sha256": sha(ROOT / "docs/20261009-v2-q1-hybrid-kernel-preregistered.md"),
              "wall_time_seconds": time.time() - start}
    (out / "q1_formal.json").write_text(json.dumps(formal, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(out / "q1_raw.npz", trial_state_sha256=np.asarray([x["state_sha256"] for x in gates["trials"]]),
                        release=np.asarray([x["release_integral"] for x in gates["trials"]]), delivery=np.asarray([x["signed_delivery_integral"] for x in gates["trials"]]))
    (out / "q1_report.md").write_text(f"# V2-Q1 hybrid-kernel qualification\n\nVerdict: `{verdict}`\n\nLearning, reward, market and execution paths were not used.\n")
    (out / "q1_manifest.json").write_text(json.dumps({"python": sys.version, "platform": platform.platform(), "v2_build": build, "runner_sha256": sha(Path(__file__)), "tests_sha256": sha(ROOT / "tests/test_v2_q1_hybrid_qualification.py") if (ROOT / "tests/test_v2_q1_hybrid_qualification.py").exists() else None}, indent=2, sort_keys=True) + "\n")
    return formal


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=OUT); args = ap.parse_args()
    print(json.dumps(run(args.out), indent=2, sort_keys=True))
