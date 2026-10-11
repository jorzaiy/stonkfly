from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "q2_runner", ROOT / "experiments/a_share/run_v2_q2_qualification.py")
Q2 = importlib.util.module_from_spec(SPEC); assert SPEC.loader; SPEC.loader.exec_module(Q2)


def test_signed_clamp_and_positive_half_identity():
    from stonkfly.neural.v2_brain import release_from_voltage
    from stonkfly.neural.v2_q2_brain import release_deviation_from_voltage
    v = np.array([-66., -59., -55.5, -52., -48.5, -45., -38.])
    assert np.array_equal(release_deviation_from_voltage(v), [-1., -1., -.5, 0., .5, 1., 1.])
    assert np.array_equal(release_deviation_from_voltage(v[v >= -52]), release_from_voltage(v[v >= -52]))


def test_all_a_gates_and_arnoldi_fixtures(tmp_path):
    gates = Q2.analytic_gates(tmp_path / "micro")
    assert gates["pass"], gates
    assert {k for k in gates if k.startswith("A")} == {
        "A1_deviation_clamp", "A2_four_quadrants", "A3_two_hop_sign_inversion",
        "A4_signed_external_drive", "A5_sign_permutation_negative_control",
        "A6_no_graded_semantic_leak", "A7_compatibility_positive_regression"}
    fixture = Q2.arnoldi_fixture_gate()
    assert fixture["pass"], fixture
    assert [x["observed_unstable"] for x in fixture["rows"]] == [False, True]


def test_verdict_precedence():
    stable = {"converged_count": 2, "unstable_count": 0}
    unstable = {"converged_count": 2, "unstable_count": 1}
    inconclusive = {"converged_count": 0, "unstable_count": 0}
    assert Q2.choose_verdict(False, False, None) == "Q2_IMPLEMENTATION_FAILURE"
    assert Q2.choose_verdict(True, False, None) == "Q2_ANALYTIC_FAILURE"
    assert Q2.choose_verdict(True, True, inconclusive) == "Q2_LINEAR_STABILITY_INCONCLUSIVE"
    assert Q2.choose_verdict(True, True, unstable) == "Q2_LINEAR_INSTABILITY"
    assert Q2.choose_verdict(True, True, stable) == "Q2_IMPLEMENTATION_FAILURE"


def test_cli_smoke_complete_artifact_chain(tmp_path):
    out = tmp_path / "smoke"
    result = subprocess.run([sys.executable, str(ROOT / "experiments/a_share/run_v2_q2_qualification.py"),
                             "--smoke", "--out", str(out)], cwd=ROOT, text=True,
                            capture_output=True, check=True)
    assert "Q2_SMOKE_OK" in result.stdout
    required = {"q2_raw.npz", "q2_formal.json", "q2_report.md", "q2_manifest.json", "launch.txt", "stdout.txt"}
    assert required == {x.name for x in out.iterdir()}
    formal = json.loads((out / "q2_formal.json").read_text())
    assert formal["verdict"] == "Q2_SMOKE_OK"
    manifest = json.loads((out / "q2_manifest.json").read_text())
    assert set(manifest["artifacts"]) == {"raw", "formal", "report", "launch", "stdout"}


def test_output_protection_and_failure_artifacts(tmp_path):
    occupied=tmp_path/"occupied"; occupied.mkdir(); (occupied/"sentinel").write_text("keep")
    with pytest.raises(RuntimeError,match="refusing to overwrite"):
        Q2._prepare_output(occupied)
    failed=tmp_path/"failed"; formal=Q2.write_failure(failed,RuntimeError("boom"))
    assert formal["verdict"]=="Q2_IMPLEMENTATION_FAILURE"
    assert all((failed/name).is_file() for name in ("q2_raw.npz","q2_formal.json","q2_report.md","q2_manifest.json"))
