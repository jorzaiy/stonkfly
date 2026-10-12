#!/usr/bin/env python3
"""Run one M2-E preparation command and append its evidence record."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "runs/mb_lab/m2-e-preparation/ATTEMPTS.jsonl"
TRACKED = (
    ROOT / "protocols/mb_lab/m2_e/config.json",
    ROOT / "stonkfly/mb_lab/e1.py",
    ROOT / "stonkfly/mb_lab/m2e.py",
    ROOT / "experiments/mb_lab/run_m2e_smoke.py",
    ROOT / "tests/mb_lab/test_m2_e.py",
)


def sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("toy", "smoke", "exploration"), required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("command required after --")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    if args.log.exists():
        raise FileExistsError(args.log)
    started = datetime.now(timezone.utc).isoformat()
    before = time.monotonic()
    with args.log.open("x") as handle:
        process = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        handle.flush()
        os.fsync(handle.fileno())
    record = {
        "schema": "stonkfly-m2-e-preparation-attempt-v1",
        "kind": args.kind,
        "purpose": args.purpose,
        "started_utc": started,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.monotonic() - before,
        "argv": command,
        "exit_code": process.returncode,
        "log": str(args.log.resolve().relative_to(ROOT)),
        "log_sha256": sha(args.log),
        "source_hashes": {str(path.relative_to(ROOT)): sha(path) for path in TRACKED},
    }
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(LEDGER, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o640)
    with os.fdopen(descriptor, "a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    sys.stdout.write(args.log.read_text())
    print(json.dumps({"attempt_recorded": True, "exit_code": process.returncode}, sort_keys=True))
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
