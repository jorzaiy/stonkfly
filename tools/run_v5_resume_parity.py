#!/usr/bin/env python3
"""Prepare/run the independent V5 interrupted/resumed parity gate.

This is a gate-only harness. It is never a development result and requires
--execute explicitly; without it, only the exact commands are printed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from experiments.a_share.wc006_male_cns_v5_adapter import V5_ARMS, audit_v5_resume_parity, validate_v5_inputs

TRAIN_END = "2023-12-29"


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def command(python: Path, repo: Path, candidate: Path, market: Path, output: Path, arm: str, seed: int, max_days: int) -> list[str]:
    return [str(python), str(repo / "experiments/a_share/wc006_male_cns_v5_adapter.py"), "--manifest", str(candidate), "--out", str(output), "--market-manifest", str(market), "--arm", arm, "--seed", str(seed), "--max-days", str(max_days)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--market-manifest", type=Path, required=True)
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--execute", action="store_true", help="Actually run parity; omitted means dry-run only.")
    args = parser.parse_args()
    if args.seed != 17:
        raise SystemExit("fixed V5 parity seed must be 17")
    if not args.run_id or "/" in args.run_id or "\\" in args.run_id:
        raise SystemExit("run-id must be one path component")
    repo = Path(__file__).resolve().parents[1]
    python = repo / ".venv/bin/python"
    candidate = args.manifest.resolve()
    market = args.market_manifest.resolve()
    gate_root = args.gate_root.resolve() / args.run_id
    if gate_root.exists():
        raise SystemExit(f"refusing existing parity gate directory: {gate_root}")
    _, gate = validate_v5_inputs(candidate, expected_records=46, training_end=TRAIN_END, market_manifest=market, formal=True)
    if gate["training_authorized"] is not True:
        raise SystemExit("parity input gate is not authorized")

    plan = []
    for arm in V5_ARMS:
        full = repo / "runs" / f"wc006_v5_{arm}" / "parity_gate" / f"{args.run_id}-full"
        resumed = repo / "runs" / f"wc006_v5_{arm}" / "parity_gate" / f"{args.run_id}-resumed"
        plan.append({
            "arm": arm,
            "full": full,
            "resumed": resumed,
            "commands": [
                command(python, repo, candidate, market, full, arm, args.seed, 0),
                command(python, repo, candidate, market, resumed, arm, args.seed, 3),
                command(python, repo, candidate, market, resumed, arm, args.seed, 0),
            ],
        })
    print(json.dumps({"status": "READY" if not args.execute else "RUNNING", "purpose": "resume_parity_gate_only", "seed": args.seed, "arms": list(V5_ARMS), "gate_root": str(gate_root), "plan": [[" ".join(item) for item in row["commands"]] for row in plan]}, ensure_ascii=False, indent=2))
    if not args.execute:
        return 0

    gate_root.mkdir(parents=True)
    results = []
    for row in plan:
        arm = row["arm"]
        arm_gate = gate_root / arm
        arm_gate.mkdir(parents=True)
        log = arm_gate / "training.log"
        with log.open("w", encoding="utf-8") as handle:
            handle.write(f"[parity-gate] started_utc={utc()} purpose=resume_parity_gate_only arm={arm} seed={args.seed}\n")
            handle.write("[parity-gate] formal development result: FALSE\n")
        for index, cmd in enumerate(row["commands"], start=1):
            with log.open("a", encoding="utf-8") as handle:
                handle.write(f"\n[parity-gate] command_{index}=" + " ".join(cmd) + "\n")
            completed = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with log.open("a", encoding="utf-8") as handle:
                handle.write("----- stdout -----\n" + completed.stdout)
                handle.write("----- stderr -----\n" + completed.stderr)
                handle.write(f"[parity-gate] command_{index}_exit={completed.returncode}\n")
            if completed.returncode != 0:
                (arm_gate / "PARITY_GATE_ONLY.json").write_text(json.dumps({"status": "FAIL", "arm": arm, "failed_command": index}, indent=2) + "\n")
                raise SystemExit(f"parity command failed for {arm}: {index}")
        result = audit_v5_resume_parity(row["full"], row["resumed"])
        results.append({"arm": arm, "result": result})
        if result.get("status") != "PASS":
            (arm_gate / "PARITY_GATE_ONLY.json").write_text(json.dumps({"status": "FAIL", "arm": arm, "result": result}, indent=2) + "\n")
            raise SystemExit(f"parity failed for {arm}: {result}")
        (arm_gate / "PARITY_GATE_ONLY.json").write_text(json.dumps({"status": "PASS", "purpose": "resume_parity_gate_only", "arm": arm, "seed": args.seed, "result": result}, indent=2, sort_keys=True) + "\n")
        with log.open("a", encoding="utf-8") as handle:
            handle.write("[parity-gate] final_status=PASS\n")
    (gate_root / "summary.json").write_text(json.dumps({"status": "PASS", "purpose": "resume_parity_gate_only", "seed": args.seed, "results": results}, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
