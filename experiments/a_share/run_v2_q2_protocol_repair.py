#!/usr/bin/env python3
"""Q2 v2 protocol-repair replication; B1 is byte-locked to v1."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from stonkfly.neural.q2_dense_reference import simulate_graded, first_nonzero_clock
from stonkfly.neural.v2_brain import HybridBrain, classify_nodes, release_from_voltage
from stonkfly.neural.v2_q2_brain import Q2HybridBrain, arnoldi_hessenberg, release_deviation_from_voltage, build, build_linear_audit

SPEC = importlib.util.spec_from_file_location("q2_v1_runner", ROOT / "experiments/a_share/run_v2_q2_qualification.py")
V1 = importlib.util.module_from_spec(SPEC); assert SPEC.loader; SPEC.loader.exec_module(V1)

OUT = ROOT / "runs/whole_brain_v2/Q2/20261010T-bidirectional-graded-kernel-v2-protocol-repair"
AMENDMENT = ROOT / "docs/20261010-v2-q2-v2-protocol-repair-amendment.md"
PREREG = ROOT / "docs/20261010-v2-q2-bidirectional-graded-kernel-preregistered.md"
REVIEW = ROOT / "docs/20261010-v2-q2-results-for-opus.md"
V1_DIR = ROOT / "runs/whole_brain_v2/Q2/20261010T-bidirectional-graded-kernel-v1"
V1_RAW = V1_DIR / "q2_raw.npz"
TESTS = ROOT / "tests/test_v2_q2_protocol_repair.py"
REFERENCE = ROOT / "stonkfly/neural/q2_dense_reference.py"
DT_VALUES = (0.1, 0.05, 0.025)
LOCKS = {
    "prereg": (PREREG, "dc4661734f88fcbbbdc1e98f8391ca149b897795fd5d46b4d34a2e044bc19c94"),
    "review": (REVIEW, "11ea7d56c3b6fb29c3868edd3ff0381b96371dd750e410a460f1b0ac15794116"),
    "v1_raw": (V1_RAW, "28c0eb9e3d2f6edbc88dc0521ed503e87cb7a424075974c77130c1f0f5c63b68"),
    "v1_formal": (V1_DIR / "q2_formal.json", "fab03dbccf54b0835ab2791ff04a02ea8f1752a53cf8514114be8281e0935092"),
    "v1_manifest": (V1_DIR / "q2_manifest.json", "b3ab50d46b908bf27f252aafd3c4ea237d647ef26edb0faf3adcc4adbf035429"),
    "q2_source": (ROOT / "stonkfly/neural/v2_q2_kernel.cpp", "5a578df80129358ddc1aaa15d2f9495b76210c404e639e52a3db124dd920457e"),
    "q2_wrapper": (ROOT / "stonkfly/neural/v2_q2_brain.py", "8cf97bbf92571c4af73b28f104cc963b0830ad0ec8b55e62ddf7ff8faf938f08"),
    "linear_source": (ROOT / "stonkfly/neural/q2_linear_audit.cpp", "3fc5c493094b8f1954ded5949821bfeb12002d38fcf67a640eb29e23f8eb0d88"),
}
WGG_LOCK = {
    "graded_global_indices_sha256": "04d1eed8b76a69836605459866cb5770350fb4dae56df600086a08a310b5ef39",
    "ptr_sha256": "430bf75068abaaf2a54743bb060921b9678c9b0ebe8687c46723a423d25796f9",
    "post_sha256": "e43ac82774b5d88c07098b9a865b78c92991b403e4693697cc66dad3fcfc829a",
    "weight_float64_sha256": "9656a0a9a79a27ec57d1852484bbcb0016f264fcbb2b4a44ed1718e0cacb7a6e",
    "csr_canonical_sha256": "c40fddad5ec32314eaf39e244e55f8eb993a94cc43c3f485029d7827846875aa",
}
H_LOCK = {"sin_h": "51327817610d1c17abd98c3777ade08b3e2e7d5e5ea50e0041371f05c6887147",
          "rademacher_h": "5dc832551f0da1e853e71a988421c909bbbbdb83bc22ed6edc3f490617cffc9f"}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


def ah(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def digest_arrays(values):
    h=hashlib.sha256()
    for name in sorted(values):
        a=np.ascontiguousarray(values[name]); h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(np.asarray(a.shape,dtype="<i8").tobytes()); h.update(a.tobytes())
    return h.hexdigest()


def fixed_inputs():
    base=V1.fixed_inputs(); actual={name:sha(path) for name,(path,_) in LOCKS.items()}
    bad={name:{"expected":expected,"actual":actual[name]} for name,(path,expected) in LOCKS.items() if actual[name]!=expected}
    q2=build(); linear=build_linear_audit()
    if q2["binary_sha256"]!="74faea7acb5772d741bff996481c3c46a483da4fd3d4c16c50bcf8277b617a42": bad["q2_binary"]={"actual":q2["binary_sha256"]}
    if linear["binary_sha256"]!="09bc25e3826704afe23a64d51db946897626aac1660794a32d3c8eb8c6ae76c9": bad["linear_binary"]={"actual":linear["binary_sha256"]}
    if bad: raise RuntimeError("repair lock mismatch: "+json.dumps(bad,sort_keys=True))
    return {"v1_fixed":base,"repair_locks":actual},q2,linear


def _write_graph(path, ptr, post, weight):
    np.savez(path,ptr=np.asarray(ptr,np.int64),post=np.asarray(post,np.int32),weight=np.asarray(weight,np.float32),ids=np.arange(len(ptr)-1,dtype=np.int64))


def _brain_trace(path, dt, steps, initial_v, drive):
    data=np.load(path,allow_pickle=False); n=len(data["ids"])
    b=Q2HybridBrain(path,dt=dt,graded_mask=np.ones(n,np.uint8)); b.v[:]=np.asarray(initial_v,np.float32)
    b.active[:n]=np.arange(n,dtype=np.int32); b.active_flag[:]=1; b.nactive[0]=n
    v=[];g=[];u=[];r=[];edge=[]
    for step in range(steps):
        b.run(dt,np.asarray(drive,np.float32) if step==0 else None,edge_trace=True)
        v.append(b.v.copy());g.append(b.g_graded.copy());u.append(b.graded_drive.copy());r.append(release_deviation_from_voltage(b.v,b.rest));edge.append(b.edge_signed_delivery.copy())
    arrays={"v":np.asarray(v),"g_graded":np.asarray(g),"graded_drive":np.asarray(u),"release":np.asarray(r),"edge_signed_delivery":np.asarray(edge)}
    return arrays,b.edge_first_arrival_clock.copy(),digest_arrays(arrays)


def _compare_fixture(path, *, dt, duration, initial_v, drive):
    data=np.load(path,allow_pickle=False); steps=round(duration/dt)
    got,first,h=_brain_trace(path,dt,steps,initial_v,drive)
    ref=simulate_graded(data["ptr"],data["post"],data["weight"],dt=dt,steps=steps,drive=drive,initial_v=initial_v)
    errors={k:float(np.max(np.abs(got[k]-ref[k]))) if got[k].size else 0.0 for k in got}
    got_input=[first_nonzero_clock(got["graded_drive"][:,i],threshold=1e-7) for i in range(len(initial_v))]
    ref_input=[first_nonzero_clock(ref["graded_drive"][:,i],threshold=1e-7) for i in range(len(initial_v))]
    got_voltage=[first_nonzero_clock(got["v"][:,i]+52.0,threshold=1e-7) for i in range(len(initial_v))]
    ref_voltage=[first_nonzero_clock(ref["v"][:,i]+52.0,threshold=1e-7) for i in range(len(initial_v))]
    structural_voltage=[None if x is None else x+1 for x in got_input]
    reference_structural_voltage=[None if x is None else x+1 for x in ref_input]
    return {"trace_sha256":h,"max_abs_error":errors,"max_error":max(errors.values()),
            "first_arrival_clock":first.tolist(),"reference_first_arrival_clock":ref["edge_first_arrival_clock"].tolist(),
            "first_input_clock":got_input,"reference_first_input_clock":ref_input,
            "structural_voltage_clock":structural_voltage,"reference_structural_voltage_clock":reference_structural_voltage,
            "representable_voltage_clock":got_voltage,"reference_representable_voltage_clock":ref_voltage,
            "final_voltage_delta":(got["v"][-1]+52.0).tolist(),
            "pass":bool(max(errors.values())<=1e-4 and np.array_equal(first,ref["edge_first_arrival_clock"]) and got_input==ref_input and structural_voltage==reference_structural_voltage)}


def _state_digest(brain):
    s=brain.audit_state(); arrays={k:v for k,v in s.items() if isinstance(v,np.ndarray)}
    h=digest_arrays(arrays); scalars={k:v for k,v in s.items() if not isinstance(v,np.ndarray)}
    return hashlib.sha256((h+json.dumps(scalars,sort_keys=True)).encode()).hexdigest()


def _state_equal(a,b):
    x,y=a.audit_state(),b.audit_state()
    return all(np.array_equal(x[k],y[k]) if isinstance(x[k],np.ndarray) else x[k]==y[k] for k in x)


def analytic_gates(work, dts=DT_VALUES):
    work.mkdir(parents=True,exist_ok=True); gates={}
    vals=np.array([-66.,-59.,-55.5,-52.,-48.5,-45.,-38.]); expected=np.array([-1.,-1.,-.5,0.,.5,1.,1.])
    a1=[]
    for dt in dts:
        reps=[]
        for repeat in (1,2):
            got=release_deviation_from_voltage(vals); payload={"got":got,"expected":expected}
            reps.append({"repeat":repeat,"trace_sha256":digest_arrays(payload),"max_error":float(np.max(np.abs(got-expected))),"pass":bool(np.array_equal(got,expected) and np.array_equal(got[vals>=-52],release_from_voltage(vals[vals>=-52])))})
        a1.append({"dt_ms":dt,"repeats":reps,"repeat_equal":reps[0]["trace_sha256"]==reps[1]["trace_sha256"],"pass":all(r["pass"] for r in reps) and reps[0]["trace_sha256"]==reps[1]["trace_sha256"]})
    gates["A1_deviation_clamp"]={"rows":a1,"pass":all(x["pass"] for x in a1)}
    configs={
        "A2_four_quadrants":[("quadrant",[0,1,1],[1],[w],[-52+d,-52],[0,0]) for w in (-1.,1.) for d in (-3.5,3.5)],
        "A3_two_hop_sign_inversion":[("two_hop",[0,1,2,2],[1,2],[a,b],[-48.5,-52,-52],[0,0,0]) for a,b in ((-1.,-1.),(-1.,1.),(1.,-1.),(1.,1.))],
        "A4_signed_external_drive":[("drive",[0,1,1],[1],[1.],[-52,-52],[d,0]) for d in (-3.5,3.5)],
        "A5_sign_permutation_negative_control":[("permute",[0,1,2,2],[1,2],[1.,w],[-48.5,-52,-52],[0,0,0]) for w in (-1.,1.)],
    }
    durations={"A2_four_quadrants":4.0,"A3_two_hop_sign_inversion":12.0,"A4_signed_external_drive":5.0,"A5_sign_permutation_negative_control":12.0}
    for gate,fixtures in configs.items():
        rows=[]
        for dt in dts:
            for fi,(name,ptr,post,weight,iv,drive) in enumerate(fixtures):
                path=work/f"{gate}_{fi}.npz";_write_graph(path,ptr,post,weight)
                reps=[_compare_fixture(path,dt=dt,duration=durations[gate],initial_v=iv,drive=drive) for _ in (1,2)]
                row={"dt_ms":dt,"fixture":fi,"weights":weight,"repeats":reps,"repeat_equal":reps[0]["trace_sha256"]==reps[1]["trace_sha256"]}
                row["pass"]=bool(row["repeat_equal"] and all(r["pass"] for r in reps))
                if gate=="A3_two_hop_sign_inversion":
                    row["first_hop_input_clock"]=reps[0]["first_input_clock"][1];row["second_hop_input_clock"]=reps[0]["first_input_clock"][2]
                    row["first_hop_voltage_clock"]=reps[0]["structural_voltage_clock"][1];row["second_hop_voltage_clock"]=reps[0]["structural_voltage_clock"][2]
                    row["pass"] &= all(x is not None for x in (row["first_hop_input_clock"],row["second_hop_input_clock"],row["first_hop_voltage_clock"],row["second_hop_voltage_clock"]))
                rows.append(row)
        if gate=="A5_sign_permutation_negative_control":
            for dt in dts:
                pair=[r for r in rows if r["dt_ms"]==dt]
                opposite=np.sign(pair[0]["repeats"][0]["final_voltage_delta"][-1])==-np.sign(pair[1]["repeats"][0]["final_voltage_delta"][-1])
                for r in pair:r["negative_control_opposite"]=bool(opposite);r["pass"] &= bool(opposite)
        gates[gate]={"rows":rows,"pass":all(r["pass"] for r in rows)}
    # A6/A7 use the fixed Q1 implementation as the behavioral reference.
    p=work/"semantic.npz";_write_graph(p,[0,1,1],[1],[1.])
    a6=[];a7=[]
    for dt in dts:
        reps=[];creps=[]
        for repeat in (1,2):
            q2=Q2HybridBrain(p,dt=dt,graded_mask=np.array([0,1],np.uint8));q1=HybridBrain(p,dt=dt,graded_mask=np.array([0,1],np.uint8))
            for b in (q2,q1):b.v[0]=-55.5;b.active[:2]=[0,1];b.active_flag[:]=1;b.nactive[0]=2;b.run(5.)
            ok=_state_equal(q2,q1) and q2.v[1]==q2.rest[1] and q2.absolute_delivery[0]==0
            # Force the Q2 hybrid path with a disconnected graded node while
            # exercising all four source/target semantic combinations.
            connection_rows=[]; connection_digests=[]
            cp=work/"connections.npz";_write_graph(cp,[0,1,1,1],[1],[1.])
            for label,pair_mask,source_drive in (("spiking_to_spiking",[0,0,1],8.),("spiking_to_graded",[0,1,1],8.),("graded_to_spiking",[1,0,1],3.5),("graded_to_graded",[1,1,1],3.5)):
                mask=np.asarray(pair_mask,np.uint8);x2=Q2HybridBrain(cp,dt=dt,graded_mask=mask);x1=HybridBrain(cp,dt=dt,graded_mask=mask);d=np.array([source_drive,0.,0.],np.float32)
                if label=="graded_to_graded":
                    for b in (x2,x1):b.v[1]=-51.;b.active[:3]=[0,1,2];b.active_flag[:]=1;b.nactive[0]=3
                x2.run(8.,d);x1.run(8.,d);same=_state_equal(x2,x1);connection_digests.append(_state_digest(x2));connection_rows.append({"connection":label,"q1_bitwise_equal":same,"pass":same})
            # Refractory target behavior remains exactly Q1 while signed
            # graded input continues to update g_graded.
            rp=work/"refractory.npz";_write_graph(rp,[0,1,1],[1],[1.]);r2=Q2HybridBrain(rp,dt=dt,graded_mask=np.array([1,0],np.uint8));r1=HybridBrain(rp,dt=dt,graded_mask=np.array([1,0],np.uint8))
            for b in (r2,r1):b.active[:2]=[0,1];b.active_flag[:]=1;b.nactive[0]=2;b.refractory[1]=b.refractory_steps;b.run((b.refractory_steps+b.delay_steps+8)*dt,np.array([3.5,0.],np.float32))
            refractory_equal=_state_equal(r2,r1);ok &= all(x["pass"] for x in connection_rows) and refractory_equal
            combined=hashlib.sha256((_state_digest(q2)+"".join(connection_digests)+_state_digest(r2)).encode()).hexdigest()
            reps.append({"repeat":repeat,"trace_sha256":combined,"q1_equal":_state_equal(q2,q1),"connection_semantics":connection_rows,"refractory_reset_q1_equal":refractory_equal,"pass":bool(ok)})
            empty2=Q2HybridBrain(p,dt=dt,graded_mask=np.zeros(2,np.uint8));empty1=HybridBrain(p,dt=dt,graded_mask=np.zeros(2,np.uint8));drive=np.array([8.,0.],np.float32);empty2.run(8.,drive);empty1.run(8.,drive)
            pos2=Q2HybridBrain(p,dt=dt,graded_mask=np.array([1,0],np.uint8));pos1=HybridBrain(p,dt=dt,graded_mask=np.array([1,0],np.uint8));drive=np.array([3.5,0.],np.float32);pos2.run(8.,drive);pos1.run(8.,drive)
            ok2=_state_equal(empty2,empty1) and _state_equal(pos2,pos1)
            creps.append({"repeat":repeat,"trace_sha256":hashlib.sha256((_state_digest(empty2)+_state_digest(pos2)).encode()).hexdigest(),"empty_mask_bitwise":_state_equal(empty2,empty1),"positive_half_bitwise":_state_equal(pos2,pos1),"pass":bool(ok2)})
        a6.append({"dt_ms":dt,"repeats":reps,"repeat_equal":reps[0]["trace_sha256"]==reps[1]["trace_sha256"],"pass":all(x["pass"] for x in reps) and reps[0]["trace_sha256"]==reps[1]["trace_sha256"]})
        a7.append({"dt_ms":dt,"repeats":creps,"repeat_equal":creps[0]["trace_sha256"]==creps[1]["trace_sha256"],"pass":all(x["pass"] for x in creps) and creps[0]["trace_sha256"]==creps[1]["trace_sha256"]})
    gates["A6_no_graded_semantic_leak"]={"rows":a6,"pass":all(x["pass"] for x in a6)}
    gates["A7_compatibility_positive_regression"]={"rows":a7,"pass":all(x["pass"] for x in a7)}
    gates["coverage"]={"pass":all(len(gates[k]["rows"])>=len(dts) and all(len(r["repeats"])==2 for r in gates[k]["rows"]) for k in gates if k.startswith("A")),"dt_values":list(dts),"repeat_count":2,"independent_reference":"stonkfly/neural/q2_dense_reference.py"}
    gates["pass"]=all(gates[k]["pass"] for k in gates)
    return gates


def _matrix_csr(m):
    post=[];weight=[];ptr=[0]
    for source in range(len(m)):
        targets=np.flatnonzero(m[:,source]!=0);post.extend(targets.tolist());weight.extend(m[targets,source].tolist());ptr.append(len(post))
    return np.asarray(ptr,np.int64),np.asarray(post,np.int32),np.asarray(weight,float)


def arnoldi_fixture_gate():
    matrices={"stable_diagonal":np.diag([.5,.25,-.1,-.2]),"unstable_diagonal":np.diag([20.,.25,-.1,-.2]),
              "complex_negative_feedback":np.array([[0.,0.,-20.],[20.,0.,0.],[0.,20.,0.]]),
              "nonnormal_triangular":np.array([[.5,10.,0.],[0.,.25,10.],[0.,0.,-.1]])}
    rows=[]
    for name,m in matrices.items():
        ptr,post,w=_matrix_csr(m);start=np.arange(1,len(m)+1,dtype=float);h,tail,_,_=arnoldi_hessenberg(ptr,post,w,start,len(m));a=V1.analyze_hessenberg(h,tail)
        got=np.sort_complex(np.linalg.eigvals(h));expected=np.sort_complex(np.linalg.eigvals(m));eig_ok=np.allclose(got,expected,atol=1e-10,rtol=1e-10)
        expected_unstable=bool(np.any([max(x.real for x in V1.characteristic_roots(z))>=0 for z in expected]));observed=bool(np.any(a["unstable"]))
        rows.append({"name":name,"eigenvalue_match":bool(eig_ok),"expected_unstable":expected_unstable,"observed_unstable":observed,"max_residual":float(a["residuals"].max()),"pass":bool(eig_ok and expected_unstable==observed)})
    return {"rows":rows,"pass":all(x["pass"] for x in rows)}


def graded_subgraph_locked(ptr,post,weight,graded):
    nodes=np.flatnonzero(graded);local=np.full(len(graded),-1,np.int32);local[nodes]=np.arange(len(nodes),dtype=np.int32);lens=np.zeros(len(nodes),np.int64);ps=[];ws=[]
    for li,source in enumerate(nodes):
        lo,hi=int(ptr[source]),int(ptr[source+1]);target=local[post[lo:hi]];keep=target>=0
        if np.any(keep):ps.append(np.ascontiguousarray(target[keep],np.int32));ws.append(np.ascontiguousarray(weight[lo:hi][keep],np.float64));lens[li]=int(keep.sum())
    gp=np.empty(len(nodes)+1,np.int64);gp[0]=0;np.cumsum(lens,out=gp[1:]);gpost=np.concatenate(ps);gw=np.concatenate(ws)
    combined=hashlib.sha256()
    for name,a in (("ptr",gp),("post",gpost),("weight",gw)):combined.update(name.encode());combined.update(str(a.dtype).encode());combined.update(np.asarray(a.shape,dtype="<i8").tobytes());combined.update(np.ascontiguousarray(a).tobytes())
    hashes={"graded_global_indices_sha256":ah(nodes.astype(np.int64)),"ptr_sha256":ah(gp),"post_sha256":ah(gpost),"weight_float64_sha256":ah(gw),"csr_canonical_sha256":combined.hexdigest()}
    return gp,gpost,gw,nodes,hashes


def linear_gate_locked(ptr,post,weight,graded):
    gp,gpost,gw,nodes,hashes=graded_subgraph_locked(ptr,post,weight,graded);wgg_match=hashes==WGG_LOCK
    old=np.load(V1_RAW,allow_pickle=False);starts={"sin":np.sin(np.arange(len(nodes))+1.0),"rademacher":np.random.Generator(np.random.PCG64(20261010)).choice([-1.,1.],len(nodes))}
    rows=[];raw={};h_lock=True
    for name,start in starts.items():
        h,tail,_,_=arnoldi_hessenberg(gp,gpost,gw,start,160);a=V1.analyze_hessenberg(h,tail);key=name+"_h";same=bool(np.array_equal(h,old[key]) and ah(h)==H_LOCK[key]);h_lock &= same
        raw[key]=h;raw[name+"_eigenvalues"]=a["eigenvalues"];raw[name+"_residuals"]=a["residuals"];raw[name+"_max_root_real"]=a["max_root_real"]
        rows.append({"start":name,"converged_count":int(a["converged"].sum()),"unstable_count":int(a["unstable"].sum()),"h_sha256":ah(h),"v1_h_bitwise_equal":same,"max_converged_root_real":float(np.max(a["max_root_real"][a["converged"]])) if np.any(a["converged"]) else None})
    converged=sum(x["converged_count"] for x in rows);exists=any(x["unstable_count"]>0 for x in rows)
    result={"wgg_hashes":hashes,"wgg_lock_pass":wgg_match,"h_lock_pass":bool(h_lock),"node_count":len(nodes),"edge_count":len(gpost),"rows":rows,"converged_exists":converged>0,"unstable_exists":exists,"pass":bool(wgg_match and h_lock and converged>0 and not exists)}
    return result,raw


def hash_snapshot():
    paths={"amendment":AMENDMENT,"prereg":PREREG,"review":REVIEW,"runner":Path(__file__).resolve(),"tests":TESTS,"reference":REFERENCE}
    paths.update({k:p for k,(p,_) in LOCKS.items() if k in ("q2_source","q2_wrapper","linear_source")})
    return {k:sha(v) for k,v in paths.items() if v.exists()}


def _prepare(out):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise RuntimeError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True,exist_ok=True);return out


def _write(out,formal,raw,start_hashes,started):
    rawp=out/"q2_raw.npz";np.savez_compressed(rawp,**raw);formal["wall_time_seconds"]=time.time()-started
    fp=out/"q2_formal.json";fp.write_text(json.dumps(formal,indent=2,sort_keys=True,default=str)+"\n")
    rp=out/"q2_report.md";rp.write_text(f"# V2-Q2 v2 protocol-repair replication\n\nVerdict: `{formal['verdict']}`\n\nB1 was known before this repair. Q2 is not production eligible.\n")
    lp=out/"launch.txt";lp.write_text(" ".join(sys.argv)+"\n");sp=out/"stdout.txt";sp.write_text(json.dumps({"verdict":formal["verdict"],"protocol_repair_replication":True},sort_keys=True)+"\n")
    manifest={"schema":"v2-q2-repair-manifest-v1","hashes_start":start_hashes,"hashes_end":hash_snapshot(),"python":sys.version,"platform":platform.platform(),"artifacts":{"raw":sha(rawp),"formal":sha(fp),"report":sha(rp),"launch":sha(lp),"stdout":sha(sp)},"stages_not_run":formal.get("stages_not_run",{})}
    (out/"q2_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")


def run(out=OUT,smoke=False):
    started=time.time();out=_prepare(out);fixed,q2build,lbuild=fixed_inputs();start_hashes=hash_snapshot();dts=(.1,) if smoke else DT_VALUES
    A=analytic_gates(ROOT/"runs/whole_brain_v2/Q2/.build/repair_micro",dts);fixture=arnoldi_fixture_gate();A["B1_fixture"]=fixture;A["pass"] &= fixture["pass"]
    raw={"analytic_json":np.asarray(json.dumps(A,sort_keys=True))}
    linear=None;verdict="Q2_ANALYTIC_FAILURE"
    if A["pass"]:
        if smoke:
            verdict="Q2_REPAIR_SMOKE_OK"
        else:
            g=np.load(ROOT/"data/graph.npz",allow_pickle=False);ptr=np.asarray(g["ptr"],np.int64);post=np.asarray(g["post"],np.int32);weight=np.asarray(g["weight"],np.float32)
            nodes=feather.read_table(ROOT/"data/normalized/neurons.feather").to_pydict();layer,_=classify_nodes(nodes["cell_type"],nodes["neurotransmitter"]);linear,lr=linear_gate_locked(ptr,post,weight,np.isin(layer,[1,2,3]));raw.update(lr)
            if not linear["wgg_lock_pass"] or not linear["h_lock_pass"]:verdict="Q2_IMPLEMENTATION_FAILURE"
            elif not linear["converged_exists"]:verdict="Q2_LINEAR_STABILITY_INCONCLUSIVE"
            elif linear["unstable_exists"]:verdict="Q2_LINEAR_INSTABILITY"
            else:verdict="Q2_IMPLEMENTATION_FAILURE"
    stages={}
    if verdict=="Q2_LINEAR_INSTABILITY":stages={"B2":"stopped by repaired B1 hard gate","C":"stopped by repaired B1 hard gate"}
    formal={"schema":"v2-q2-protocol-repair-v2","verdict":verdict,"smoke":smoke,"protocol_repair_replication":True,"b1_result_known_before_run":True,"production_eligible":False,"market_used":False,"learning_used":False,"fixed_inputs":fixed,"build":q2build,"linear_build":lbuild,"gates":{"A":A,"B1":linear},"stages_not_run":stages,"amendment_sha256":sha(AMENDMENT)}
    if hash_snapshot()!=start_hashes:raise RuntimeError("repair sources changed during run")
    _write(out,formal,raw,start_hashes,started);return formal


def write_failure(out,exc):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);formal={"schema":"v2-q2-protocol-repair-v2","verdict":"Q2_IMPLEMENTATION_FAILURE","protocol_repair_replication":True,"production_eligible":False,"error":repr(exc),"traceback":traceback.format_exc()};_write(out,formal,{"failure":np.asarray(repr(exc))},hash_snapshot(),time.time());return formal


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=OUT);p.add_argument("--smoke",action="store_true");a=p.parse_args(argv)
    try:r=run(a.out,a.smoke)
    except Exception as exc:r=write_failure(a.out,exc);print(json.dumps(r,indent=2));raise
    print("Q2_REPAIR_SMOKE_OK" if a.smoke and r["verdict"]=="Q2_REPAIR_SMOKE_OK" else json.dumps(r,indent=2,sort_keys=True,default=str))


if __name__=="__main__":main()
