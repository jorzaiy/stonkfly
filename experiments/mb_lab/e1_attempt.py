#!/usr/bin/env python3
"""Execute one E1 command and append an immutable attempt-ledger record."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "runs/mb_lab/m1-e1/E1_ATTEMPTS.jsonl"
TRACKED = (
    ROOT / "protocols/mb_lab/m1_e1/config.json",
    ROOT / "stonkfly/mb_lab/e1.py",
    ROOT / "experiments/mb_lab/run_m1_e1.py",
    ROOT / "tests/mb_lab/test_m1_e1.py",
)


def digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_record(record: dict[str, object]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(LEDGER, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o640)
    with os.fdopen(descriptor, "a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("toy", "benchmark", "formal"))
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    if args.log.exists():
        raise FileExistsError(f"attempt log already exists: {args.log}")
    started = datetime.now(timezone.utc).isoformat()
    wall_start = time.monotonic()
    with args.log.open("x") as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        log.flush()
        os.fsync(log.fileno())
    record = {
        "schema": "stonkfly-m1-e1-attempt-v1",
        "kind": args.kind,
        "purpose": args.purpose,
        "started_utc": started,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.monotonic() - wall_start,
        "argv": command,
        "cwd": str(ROOT),
        "exit_code": process.returncode,
        "log": str(args.log.resolve().relative_to(ROOT)),
        "log_sha256": digest(args.log),
        "source_hashes": {
            str(path.relative_to(ROOT)): digest(path) for path in TRACKED
        },
    }
    append_record(record)
    sys.stdout.write(args.log.read_text())
    print(json.dumps({"attempt_recorded": True, "exit_code": process.returncode}, sort_keys=True))
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
