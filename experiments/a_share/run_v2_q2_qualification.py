#!/usr/bin/env python3
"""Frozen V2-Q2 signed graded-deviation kernel qualification."""
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

from stonkfly.neural.v2_brain import HybridBrain, release_from_voltage
from stonkfly.neural.v2_q2_brain import (
    Q2HybridBrain, arnoldi_hessenberg, build, build_linear_audit,
    release_deviation_from_voltage,
)

DEFAULT_OUT = ROOT / "runs/whole_brain_v2/Q2/20261010T-bidirectional-graded-kernel-v1"
PREREG = ROOT / "docs/20261010-v2-q2-bidirectional-graded-kernel-preregistered.md"
TESTS = ROOT / "tests/test_v2_q2_qualification.py"
Q2_SOURCE = ROOT / "stonkfly/neural/v2_q2_kernel.cpp"
LINEAR_SOURCE = ROOT / "stonkfly/neural/q2_linear_audit.cpp"
Q2_WRAPPER = ROOT / "stonkfly/neural/v2_q2_brain.py"
PINS = {
    "data/graph.npz": "346b8af85a11af13b8324e18669812c1924569e7d1adcb4e6f45cc461a2c344b",
    "data/normalized/neurons.feather": "2103b5203d2ba5502c06fb70a4e0b23edcbcf79ed0c344722101941fc289ba2d",
    "data/normalized/edges.arrow": "842b762c9427c794eaedd453178aa034727a5084850a91fdcc4d3fffe09f5b7e",
    "runs/whole_brain_v2/Q1/20261009T-hybrid-kernel-qualification-v7/q1_formal.json": "394c2cd05ecc54c979b9d871176a86bb82670c850c108fe91c63ba8354ead574",
    "stonkfly/neural/v2_kernel_qualification.cpp": "99f63402e2a6393c229aab139915233d24dff8ad411f068d91ec965edfbef79e",
    "stonkfly/neural/v2_brain.py": "2673de4e7c4aa4ee0fd20636b17bf2c2828103bacfbbb05b2ff0f559979434e6",
    "runs/whole_brain_v2/Q1/.build/libv2_hybrid.so": "9df8110f1e2134f8a6d88cde46d3a8a6278590f291ad87b49161926d2fd53f41",
    "docs/20261010-v2-q2-d0-dynamic-path-audit-preregistered.md": "e5a22f4ec89b2f3f5dcd80d3849913bdb2e823abc284e6c7554ddbff1f099f8f",
    "runs/whole_brain_v2/Q2/D0/20261010T-dynamic-path-audit-v1/q2_d0_formal.json": "a3b5c65f20f78f4a2eeed3d10e32e24dc0ad2f5df370298f4f783f42fd124929",
    "runs/whole_brain_v2/Q2/D0/20261010T-dynamic-path-audit-v1/q2_d0_raw.npz": "06a97159d7bb5db097cd7eb5cb1881b2b7d72962788d93d4e44793c56329d9be",
    "runs/whole_brain_v2/Q2/D0/20261010T-dynamic-path-audit-v1/q2_d0_manifest.json": "34c33f2ee8174d54f8ea33ce982b6328eebd32850f66d95bc2a4b5c5dd3d4166",
    "docs/20261010-v2-q2-d0-results-for-opus.md": "b736a1422c156b14de6d6fcb87ad00ec7426ec303e766c66af0271f9fe44a288",
}
DT_VALUES = (0.1, 0.05, 0.025)
VERDICTS = ("Q2_IMPLEMENTATION_FAILURE", "Q2_ANALYTIC_FAILURE",
            "Q2_LINEAR_STABILITY_INCONCLUSIVE", "Q2_LINEAR_INSTABILITY",
            "Q2_PERTURBATION_INSTABILITY", "Q2_FULL_GRAPH_FAILURE",
            "Q2_KERNEL_QUALIFIED")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fixed_inputs():
    actual = {name: sha(ROOT / name) for name in PINS}
    bad = {name: {"expected": PINS[name], "actual": value}
           for name, value in actual.items() if value != PINS[name]}
    if bad:
        raise RuntimeError("fixed-input hash mismatch: " + json.dumps(bad, sort_keys=True))
    if json.loads((ROOT / next(k for k in PINS if k.endswith("q1_formal.json"))).read_text())["verdict"] != "Q1_IMPLEMENTATION_QUALIFIED":
        raise RuntimeError("Q1 is not qualified")
    return actual


def _write_graph(path, ptr, post, weight):
    np.savez(path, ptr=np.asarray(ptr, np.int64), post=np.asarray(post, np.int32),
             weight=np.asarray(weight, np.float32), ids=np.arange(len(ptr) - 1, dtype=np.int64))


def _state_equal(a, b):
    sa, sb = a.audit_state(), b.audit_state()
    return all(np.array_equal(sa[k], sb[k]) if isinstance(sa[k], np.ndarray) else sa[k] == sb[k] for k in sa)


def analytic_gates(work: Path, dts=DT_VALUES):
    work.mkdir(parents=True, exist_ok=True)
    values = np.array([-66., -59., -55.5, -52., -48.5, -45., -38.])
    expected = np.array([-1., -1., -.5, 0., .5, 1., 1.])
    got = release_deviation_from_voltage(values)
    a1 = {"pass": bool(np.max(np.abs(got - expected)) <= 1e-7 and
                       np.array_equal(got[values >= -52], release_from_voltage(values[values >= -52]))),
          "values": got.tolist(), "max_abs_error": float(np.max(np.abs(got - expected)))}
    quadrant_rows = []; two_hop_rows = []; drive_rows = []; permutation_rows = []
    for dt in dts:
        for weight in (-1., 1.):
            for deviation in (-3.5, 3.5):
                p = work / "quadrant.npz"; _write_graph(p, [0, 1, 1], [1], [weight])
                b = Q2HybridBrain(p, dt=dt, graded_mask=np.ones(2, np.uint8))
                b.v[0] += deviation; b.active[:2] = [0, 1]; b.active_flag[:] = 1; b.nactive[0] = 2
                b.run(4.0, edge_trace=True)
                expected_sign = int(np.sign(weight * deviation)); observed = int(np.sign(b.v[1] - b.rest[1]))
                quadrant_rows.append({"dt_ms": dt, "weight": weight, "deviation": deviation,
                                      "expected_sign": expected_sign, "observed_sign": observed,
                                      "first_arrival_clock": int(b.edge_first_arrival_clock[0]),
                                      "expected_arrival_clock": b.delay_steps,
                                      "pass": observed == expected_sign and int(b.edge_first_arrival_clock[0]) == b.delay_steps})
        for w1, w2 in ((-1., -1.), (-1., 1.), (1., -1.), (1., 1.)):
            p = work / "two_hop.npz"; _write_graph(p, [0, 1, 2, 2], [1, 2], [w1, w2])
            b = Q2HybridBrain(p, dt=dt, graded_mask=np.ones(3, np.uint8))
            b.v[0] += 3.5; b.active[:] = [0, 1, 2]; b.active_flag[:] = 1; b.nactive[0] = 3
            b.run(12.0)
            sign = int(np.sign(b.v[2] - b.rest[2])); expected_sign = int(np.sign(w1 * w2))
            two_hop_rows.append({"dt_ms": dt, "weights": [w1, w2], "observed_sign": sign,
                                 "expected_sign": expected_sign, "pass": sign == expected_sign})
        p = work / "drive.npz"; _write_graph(p, [0, 1, 1], [1], [1.])
        signs = []
        for drive in (-3.5, 3.5):
            b = Q2HybridBrain(p, dt=dt, graded_mask=np.ones(2, np.uint8)); b.run(5., np.array([drive, 0], np.float32))
            signs.append(int(np.sign(b.v[1] - b.rest[1])))
        drive_rows.append({"dt_ms": dt, "signs": signs, "pass": signs == [-1, 1]})
        permutation_rows.append({"dt_ms": dt, "base": two_hop_rows[-1]["observed_sign"],
                                 "permuted": two_hop_rows[-2]["observed_sign"],
                                 "pass": two_hop_rows[-1]["observed_sign"] == -two_hop_rows[-2]["observed_sign"]})
    # A6: a non-graded, subthreshold source cannot emit a continuous deviation.
    p = work / "leak.npz"; _write_graph(p, [0, 1, 1], [1], [1.])
    leak_rows = []
    for dt in dts:
        b = Q2HybridBrain(p, dt=dt, graded_mask=np.array([0, 1], np.uint8)); b.v[0] = -55.5
        b.active[:2] = [0, 1]; b.active_flag[:] = 1; b.nactive[0] = 2; b.run(5.)
        leak_rows.append({"dt_ms": dt, "target_unchanged": bool(b.v[1] == b.rest[1]),
                          "no_delivery": bool(b.absolute_delivery[0] == 0),
                          "pass": bool(b.v[1] == b.rest[1] and b.absolute_delivery[0] == 0)})
    compat_rows = []
    for dt in dts:
        p = work / "compat.npz"; _write_graph(p, [0, 1, 1], [1], [.8])
        q2 = Q2HybridBrain(p, dt=dt, graded_mask=np.zeros(2, np.uint8)); q1 = HybridBrain(p, dt=dt, graded_mask=np.zeros(2, np.uint8))
        drive = np.array([8., 0.], np.float32); q2.run(8., drive); q1.run(8., drive)
        empty_equal = _state_equal(q2, q1)
        q2p = Q2HybridBrain(p, dt=dt, graded_mask=np.array([1, 0], np.uint8)); q1p = HybridBrain(p, dt=dt, graded_mask=np.array([1, 0], np.uint8))
        q2p.run(8., np.array([3.5, 0.], np.float32)); q1p.run(8., np.array([3.5, 0.], np.float32))
        positive_equal = _state_equal(q2p, q1p)
        compat_rows.append({"dt_ms": dt, "empty_mask_bitwise": empty_equal,
                            "positive_half_bitwise": positive_equal, "pass": empty_equal and positive_equal})
    gates = {
        "A1_deviation_clamp": a1,
        "A2_four_quadrants": {"pass": all(r["pass"] for r in quadrant_rows), "rows": quadrant_rows},
        "A3_two_hop_sign_inversion": {"pass": all(r["pass"] for r in two_hop_rows), "rows": two_hop_rows},
        "A4_signed_external_drive": {"pass": all(r["pass"] for r in drive_rows), "rows": drive_rows},
        "A5_sign_permutation_negative_control": {"pass": all(r["pass"] for r in permutation_rows), "rows": permutation_rows},
        "A6_no_graded_semantic_leak": {"pass": all(r["pass"] for r in leak_rows), "rows": leak_rows},
        "A7_compatibility_positive_regression": {"pass": all(r["pass"] for r in compat_rows), "rows": compat_rows},
    }
    gates["pass"] = all(gates[k]["pass"] for k in gates)
    return gates


def characteristic_roots(value):
    # 100*s^2 + 25*s + 1 - 5*lambda/(7*2.2) = 0
    return np.roots([100.0, 25.0, 1.0 - 5.0 * value / 15.4])


def analyze_hessenberg(h, tail):
    values, vectors = np.linalg.eig(h)
    residual = np.abs(tail * vectors[-1, :])
    limit = 1e-6 * np.maximum(1.0, np.abs(values))
    converged = residual <= limit
    max_real = np.array([max(r.real for r in characteristic_roots(x)) for x in values])
    return {"eigenvalues": values, "residuals": residual, "converged": converged,
            "max_root_real": max_real, "unstable": converged & (max_real >= 0)}


def arnoldi_fixture_gate():
    rows = []
    for name, diag, expected_unstable in (("stable", [.5, .25, -.1, -.2], False),
                                           ("unstable", [20., .25, -.1, -.2], True)):
        n = len(diag); ptr = np.arange(n + 1, dtype=np.int64); post = np.arange(n, dtype=np.int32)
        h, tail, _, _ = arnoldi_hessenberg(ptr, post, np.asarray(diag), np.arange(1, n + 1, dtype=float), n)
        a = analyze_hessenberg(h, tail)
        got = bool(np.any(a["unstable"]))
        eig_ok = np.allclose(np.sort(np.linalg.eigvals(h).real), np.sort(diag), atol=1e-10)
        rows.append({"name": name, "expected_unstable": expected_unstable, "observed_unstable": got,
                     "eigenvalue_match": bool(eig_ok), "max_residual": float(a["residuals"].max()),
                     "pass": bool(got == expected_unstable and eig_ok)})
    return {"pass": all(x["pass"] for x in rows), "rows": rows}


def graded_subgraph(graph, graded):
    local = np.full(len(graded), -1, np.int32); nodes = np.flatnonzero(graded); local[nodes] = np.arange(len(nodes), dtype=np.int32)
    lengths = np.zeros(len(nodes), np.int64); chunks_post = []; chunks_weight = []
    for li, source in enumerate(nodes):
        lo, hi = int(graph["ptr"][source]), int(graph["ptr"][source + 1])
        target = local[graph["post"][lo:hi]]; keep = target >= 0
        if np.any(keep):
            chunks_post.append(target[keep]); chunks_weight.append(np.asarray(graph["weight"][lo:hi][keep], np.float64)); lengths[li] = int(keep.sum())
    ptr = np.empty(len(nodes) + 1, np.int64); ptr[0] = 0; np.cumsum(lengths, out=ptr[1:])
    return ptr, np.concatenate(chunks_post).astype(np.int32), np.concatenate(chunks_weight), nodes


def linear_gate(graph, graded, k=160):
    ptr, post, weight, nodes = graded_subgraph(graph, graded)
    starts = {"sin": np.sin(np.arange(len(nodes)) + 1.0),
              "rademacher": np.random.Generator(np.random.PCG64(20261010)).choice([-1., 1.], len(nodes))}
    rows = []; raw = {}
    for name, start in starts.items():
        h, tail, _, info = arnoldi_hessenberg(ptr, post, weight, start, min(k, len(nodes)))
        a = analyze_hessenberg(h, tail)
        raw[name + "_h"] = h; raw[name + "_eigenvalues"] = a["eigenvalues"]
        raw[name + "_residuals"] = a["residuals"]; raw[name + "_max_root_real"] = a["max_root_real"]
        rows.append({"start": name, "dimension": len(h), "tail_norm": tail,
                     "converged_count": int(a["converged"].sum()), "unstable_count": int(a["unstable"].sum()),
                     "max_converged_root_real": float(np.max(a["max_root_real"][a["converged"]])) if np.any(a["converged"]) else None,
                     "helper": info})
    converged = sum(r["converged_count"] for r in rows); unstable = sum(r["unstable_count"] for r in rows)
    return {"pass": converged > 0 and unstable == 0, "converged_count": converged,
            "unstable_count": unstable, "node_count": len(nodes), "edge_count": len(post), "rows": rows}, raw


def choose_verdict(implementation, analytic, linear):
    if not implementation: return VERDICTS[0]
    if not analytic: return VERDICTS[1]
    if linear is None or linear["converged_count"] == 0: return VERDICTS[2]
    if linear["unstable_count"] > 0: return VERDICTS[3]
    # A stable B1 result may not be promoted while mandatory B2/C execution
    # is absent.  Missing required gates outrank a fabricated dynamics result.
    return VERDICTS[0]


def _prepare_output(out):
    out = Path(out)
    if out.exists() and any(out.iterdir()): raise RuntimeError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True); return out


def _write_artifacts(out, formal, raw, hashes, started):
    raw_path = out / "q2_raw.npz"; np.savez_compressed(raw_path, **raw)
    formal["wall_time_seconds"] = time.time() - started
    formal_path = out / "q2_formal.json"; formal_path.write_text(json.dumps(formal, indent=2, sort_keys=True, default=str) + "\n")
    report_path = out / "q2_report.md"; report_path.write_text(
        "# V2-Q2 bidirectional graded-kernel qualification\n\n"
        f"Verdict: `{formal['verdict']}`\n\n"
        "Q2 is a kernel qualification only. Market, reward, plasticity, decoder and execution were not used.\n")
    launch_path = out / "launch.txt"; launch_path.write_text(" ".join(sys.argv) + "\n")
    stdout_path = out / "stdout.txt"; stdout_path.write_text(json.dumps(
        {"verdict": formal["verdict"], "smoke": formal.get("smoke", False)},
        sort_keys=True) + "\n")
    manifest = {"schema": "v2-q2-manifest-v1", "python": sys.version, "platform": platform.platform(),
                "hashes_start": hashes, "hashes_end": hash_snapshot(),
                "artifacts": {"raw": sha(raw_path), "formal": sha(formal_path), "report": sha(report_path)},
                "stages_not_run": formal.get("stages_not_run", {})}
    manifest["artifacts"].update({"launch": sha(launch_path), "stdout": sha(stdout_path)})
    (out / "q2_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def hash_snapshot():
    paths = {"prereg": PREREG, "runner": Path(__file__).resolve(), "tests": TESTS,
             "q2_source": Q2_SOURCE, "linear_source": LINEAR_SOURCE, "wrapper": Q2_WRAPPER}
    return {k: sha(v) for k, v in paths.items() if v.exists()}


def run(out=DEFAULT_OUT, smoke=False):
    started = time.time(); out = _prepare_output(out); fixed = fixed_inputs(); build_info = build(); linear_info = build_linear_audit(); hashes = hash_snapshot()
    dts = (0.1,) if smoke else DT_VALUES
    analytic = analytic_gates(ROOT / "runs/whole_brain_v2/Q2/.build/micro", dts)
    fixture = arnoldi_fixture_gate(); analytic["B1_fixture"] = fixture; analytic["pass"] = analytic["pass"] and fixture["pass"]
    raw = {"analytic_json": np.asarray(json.dumps(analytic, sort_keys=True))}
    if smoke:
        ptr=np.arange(5,dtype=np.int64); post=np.arange(4,dtype=np.int32); weight=np.array([.5,.25,-.1,-.2]); start=np.arange(1,5,dtype=float)
        h,tail,_,_=arnoldi_hessenberg(ptr,post,weight,start,4); a=analyze_hessenberg(h,tail)
        linear={"pass": True,"converged_count":int(a["converged"].sum()),"unstable_count":0,"node_count":4,"edge_count":4,"rows":[]}; raw["smoke_h"]=h
        verdict="Q2_SMOKE_OK" if analytic["pass"] else "Q2_ANALYTIC_FAILURE"
    elif analytic["pass"]:
        graph=np.load(ROOT/"data/graph.npz",allow_pickle=False); nodes=feather.read_table(ROOT/"data/normalized/neurons.feather").to_pydict()
        from stonkfly.neural.v2_brain import classify_nodes
        layer,_=classify_nodes(nodes["cell_type"],nodes["neurotransmitter"]); graded=np.isin(layer,[1,2,3])
        linear, linear_raw=linear_gate(graph,graded); raw.update(linear_raw); verdict=choose_verdict(True,True,linear)
    else:
        linear=None; verdict=choose_verdict(True,False,None)
    stages_not_run={}
    if verdict in ("Q2_LINEAR_INSTABILITY","Q2_LINEAR_STABILITY_INCONCLUSIVE"):
        stages_not_run={"B2":"stopped by preregistered B1 gate","C":"stopped by preregistered B1 gate"}
    elif verdict=="Q2_PERTURBATION_INSTABILITY": stages_not_run={"C":"B2 implementation pending/failed; no promotion"}
    formal={"schema":"v2-q2-formal-v1","verdict":verdict,"smoke":smoke,"fixed_inputs":fixed,
            "build":build_info,"linear_build":linear_info,"gates":{"A":analytic,"B1":linear},
            "stages_not_run":stages_not_run,"market_used":False,"learning_used":False,
            "production_eligible":False,"preregistration_sha256":sha(PREREG)}
    if hash_snapshot()!=hashes: raise RuntimeError("source/test/prereg hashes changed during run")
    _write_artifacts(out,formal,raw,hashes,started); return formal


def write_failure(out, exc):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    formal={"schema":"v2-q2-formal-v1","verdict":"Q2_IMPLEMENTATION_FAILURE","error":repr(exc),"traceback":traceback.format_exc(),"production_eligible":False}
    raw={"failure":np.asarray(repr(exc))}; _write_artifacts(out,formal,raw,hash_snapshot(),time.time()); return formal


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,default=DEFAULT_OUT); p.add_argument("--smoke",action="store_true"); args=p.parse_args(argv)
    try: result=run(args.out,args.smoke)
    except Exception as exc:
        result=write_failure(args.out,exc); print(json.dumps(result,indent=2,sort_keys=True)); raise
    print("Q2_SMOKE_OK" if args.smoke and result["verdict"]=="Q2_SMOKE_OK" else json.dumps(result,indent=2,sort_keys=True,default=str))


if __name__ == "__main__": main()
