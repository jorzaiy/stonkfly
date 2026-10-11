#!/usr/bin/env python3
"""Independent C0 merger.

No simulation, mask construction, or candidate selection occurs here.  The
merger reads the immutable phase artifacts and applies only the preregistered
verdict precedence.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from stonkfly.neural.c0_common import artifact_sha256, formal_verdict, sha256_file


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/whole_brain_v2/C0/20261009T-graded-release-calibration-v1"


def merge(out=OUT):
    out = Path(out); caldir = out / "calibration"; holdir = out / "holdout"
    cal_path, raw_path = caldir / "c0_calibration.json", caldir / "c0_calibration_raw.npz"
    if not cal_path.exists() or not raw_path.exists():
        raise RuntimeError("calibration artifacts are incomplete")
    calibration = json.loads(cal_path.read_text())
    implementation_ok = bool(calibration.get("version") == "c0-calibration-v1" and calibration.get("trial_count") == calibration.get("expected_trial_count") == 100 and calibration.get("ramp_count") == calibration.get("expected_ramp_count") == 10 and tuple(calibration.get("gains", ())) == (.125, .25, .5, 1.0, 2.0))
    gains = calibration.get("gain_results", [])
    numerical_ok = bool(len(gains) == 5 and all(g.get("gates", {}).get("basic", {}).get("pass") is True for g in gains))
    calibration_verdict = calibration.get("verdict")
    holdout = None; holdout_path = holdir / "c0_holdout.json"
    selection_path = out / "c0_selection.json"
    if holdout_path.exists():
        holdout = json.loads(holdout_path.read_text())
    selection_ready = calibration_verdict == "C0_SELECTION_READY"
    if selection_ready and (not selection_path.exists() or holdout is None):
        implementation_ok = False
    holdout_ok = None if holdout is None else bool(holdout.get("verdict") == "C0_HOLDOUT_PASS" and holdout.get("gates", {}).get("pass") is True)
    verdict = formal_verdict(implementation_ok=implementation_ok, numerical_ok=numerical_ok, calibration_verdict=calibration_verdict, holdout_ok=holdout_ok)
    artifacts = {"calibration/c0_calibration.json": cal_path, "calibration/c0_calibration_raw.npz": raw_path}
    if selection_path.exists(): artifacts["c0_selection.json"] = selection_path
    if holdout_path.exists(): artifacts["holdout/c0_holdout.json"] = holdout_path
    holdout_raw = holdir / "c0_holdout_raw.npz"
    if holdout_raw.exists(): artifacts["holdout/c0_holdout_raw.npz"] = holdout_raw
    formal = {"version": "c0-formal-v1", "verdict": verdict, "precedence": ["C0_IMPLEMENTATION_FAILURE", "C0_NUMERICAL_FAILURE", "C0_NO_QUALIFYING_GAIN", "C0_GRID_BOUNDARY", "C0_HOLDOUT_FAILURE", "C0_SYNTHETIC_QUALIFIED"], "calibration_verdict": calibration_verdict, "calibration_sha256": sha256_file(cal_path), "selection_sha256": sha256_file(selection_path) if selection_path.exists() else None, "holdout_sha256": sha256_file(holdout_path) if holdout_path.exists() else None, "holdout_run": holdout is not None, "holdout_not_run_reason": None if holdout is not None else ("calibration had no qualifying gain" if calibration_verdict == "C0_NO_QUALIFYING_GAIN" else "selected gain was a grid boundary" if calibration_verdict == "C0_GRID_BOUNDARY" else "selection was not legally frozen"), "gates": {"implementation": implementation_ok, "numerical": numerical_ok, "calibration": {"verdict": calibration_verdict, "qualifying_gains": calibration.get("qualifying_gains", [])}, "holdout": None if holdout is None else holdout.get("gates")}, "artifact_sha256": artifact_sha256(artifacts), "counterfactual_calibration": True, "production_eligible": False}
    (out / "c0_formal.json").write_text(json.dumps(formal, indent=2, sort_keys=True) + "\n")
    report = ["# V2-C0 graded-release calibration", "", f"Verdict: `{verdict}`", "", f"Calibration verdict: `{calibration_verdict}`", "", "All gain candidates were evaluated without using holdout, market, RGB, reward, plasticity, controller, or decoder inputs."]
    if holdout is None: report.append(f"Holdout was not run: {formal['holdout_not_run_reason']}.")
    else: report.append(f"Holdout verdict: `{holdout.get('verdict')}`; all 176 trials and 2 ramps are retained in the holdout artifact.")
    (out / "c0_report.md").write_text("\n".join(report) + "\n")
    manifest = {"version": "c0-manifest-v1", "platform": platform.platform(), "python": sys.version, "formal_sha256": sha256_file(out / "c0_formal.json"), "artifact_sha256": artifact_sha256(artifacts), "calibration_runner": "experiments/a_share/run_v2_c0_calibration.py", "holdout_runner": "experiments/a_share/run_v2_c0_holdout.py", "merger": "experiments/a_share/merge_v2_c0.py", "holdout_run": holdout is not None}
    (out / "c0_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return formal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, default=OUT); args = parser.parse_args(); print(json.dumps(merge(args.out), indent=2, sort_keys=True))
