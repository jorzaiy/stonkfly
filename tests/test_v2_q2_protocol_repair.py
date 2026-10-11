from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("q2_repair",ROOT/"experiments/a_share/run_v2_q2_protocol_repair.py")
Q2=importlib.util.module_from_spec(SPEC);assert SPEC.loader;SPEC.loader.exec_module(Q2)


def test_repair_locks_and_fixed_kernel_unchanged():
    _,q2,linear=Q2.fixed_inputs()
    assert q2["source_sha256"]=="5a578df80129358ddc1aaa15d2f9495b76210c404e639e52a3db124dd920457e"
    assert q2["binary_sha256"]=="74faea7acb5772d741bff996481c3c46a483da4fd3d4c16c50bcf8277b617a42"
    assert linear["source_sha256"]=="3fc5c493094b8f1954ded5949821bfeb12002d38fcf67a640eb29e23f8eb0d88"


def test_dense_reference_is_independent_and_signed():
    text=(ROOT/"stonkfly/neural/q2_dense_reference.py").read_text()
    assert "v2_q2_brain" not in text and "ctypes" not in text
    from stonkfly.neural.q2_dense_reference import signed_deviation
    assert np.array_equal(signed_deviation([-59,-55.5,-52,-48.5,-45],-52),[-1,-.5,0,.5,1])


def test_a1_a7_full_coverage_and_reference(tmp_path):
    gates=Q2.analytic_gates(tmp_path/"micro")
    assert gates["pass"],gates
    assert gates["coverage"]["pass"] and gates["coverage"]["dt_values"]==[.1,.05,.025]
    required={f"A{i}" for i in range(1,8)}
    names={k.split("_")[0] for k in gates if k.startswith("A")}
    assert names==required
    for name,gate in gates.items():
        if not name.startswith("A"):continue
        assert {r["dt_ms"] for r in gate["rows"]}=={.1,.05,.025}
        assert all(len(r["repeats"])==2 and r["repeat_equal"] for r in gate["rows"])
        assert all("trace_sha256" in repeat for r in gate["rows"] for repeat in r["repeats"])
    for name in ("A2_four_quadrants","A3_two_hop_sign_inversion","A4_signed_external_drive","A5_sign_permutation_negative_control"):
        assert all(repeat["max_error"]<=1e-4 for row in gates[name]["rows"] for repeat in row["repeats"])
    assert all(row["first_hop_input_clock"] is not None and row["second_hop_input_clock"] is not None for row in gates["A3_two_hop_sign_inversion"]["rows"])


def test_complex_and_nonnormal_arnoldi_fixtures():
    result=Q2.arnoldi_fixture_gate()
    assert result["pass"],result
    assert {r["name"] for r in result["rows"]}=={"stable_diagonal","unstable_diagonal","complex_negative_feedback","nonnormal_triangular"}
    assert all(r["eigenvalue_match"] for r in result["rows"])
    complex_row=next(r for r in result["rows"] if r["name"]=="complex_negative_feedback")
    assert complex_row["expected_unstable"] and complex_row["observed_unstable"]


def test_wgg_and_v1_h_locks_are_explicit():
    old=np.load(Q2.V1_RAW,allow_pickle=False)
    assert Q2.ah(old["sin_h"])==Q2.H_LOCK["sin_h"]
    assert Q2.ah(old["rademacher_h"])==Q2.H_LOCK["rademacher_h"]
    assert Q2.WGG_LOCK["csr_canonical_sha256"]=="c40fddad5ec32314eaf39e244e55f8eb993a94cc43c3f485029d7827846875aa"


def test_cli_smoke_writes_complete_repair_chain(tmp_path):
    out=tmp_path/"smoke"
    r=subprocess.run([sys.executable,str(ROOT/"experiments/a_share/run_v2_q2_protocol_repair.py"),"--smoke","--out",str(out)],cwd=ROOT,text=True,capture_output=True,check=True)
    assert "Q2_REPAIR_SMOKE_OK" in r.stdout
    assert {x.name for x in out.iterdir()}=={"q2_raw.npz","q2_formal.json","q2_report.md","q2_manifest.json","launch.txt","stdout.txt"}
    f=json.loads((out/"q2_formal.json").read_text());m=json.loads((out/"q2_manifest.json").read_text())
    assert f["protocol_repair_replication"] and f["b1_result_known_before_run"] and not f["production_eligible"]
    assert f["verdict"]=="Q2_REPAIR_SMOKE_OK" and m["hashes_start"]==m["hashes_end"]


def test_output_protection_and_failure_artifacts(tmp_path):
    p=tmp_path/"occupied";p.mkdir();(p/"keep").write_text("x")
    with pytest.raises(RuntimeError,match="refusing to overwrite"):Q2._prepare(p)
    failed=tmp_path/"failed";f=Q2.write_failure(failed,RuntimeError("boom"))
    assert f["verdict"]=="Q2_IMPLEMENTATION_FAILURE"
    assert all((failed/x).is_file() for x in ("q2_raw.npz","q2_formal.json","q2_report.md","q2_manifest.json","launch.txt","stdout.txt"))
