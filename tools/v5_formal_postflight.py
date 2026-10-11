#!/usr/bin/env python3
"""Fail-closed postflight checks for one completed formal V5 arm."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from experiments.a_share.wc006_male_cns_v5_adapter import audit_v5_isolation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", required=True, choices=("attribution-only", "state-only", "combined"))
    args = parser.parse_args()
    output = args.output.resolve()
    log_path = output / "training.log"
    if not log_path.is_file():
        raise SystemExit("postflight blocked: training.log is missing")
    log = log_path.read_text(encoding="utf-8")
    required_log_markers = ("exact_command=", "stdout", "stderr", "exit_code=0")
    if any(marker not in log for marker in required_log_markers) or f"--arm {args.arm}" not in log or "--seed 17" not in log:
        raise SystemExit("postflight blocked: training.log is incomplete or not bound to arm/seed")
    if (output / "QUARANTINED.json").exists():
        raise SystemExit("postflight blocked: QUARANTINED.json exists")
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("training_authorized") is not True or provenance.get("paper_only") is not True:
        raise SystemExit("postflight blocked: provenance authorization/paper gate failed")
    if provenance.get("live_execution_authorized") is not False or provenance.get("broker_orders_enabled") is not False:
        raise SystemExit("postflight blocked: live/broker flag is enabled")
    if provenance.get("final_coverage") is None or provenance.get("final_settlement") is None:
        raise SystemExit("postflight blocked: arm did not reach final settlement")
    db = sqlite3.connect(output / "ledger.sqlite")
    try:
        unknown = db.execute("SELECT COUNT(*) FROM intents WHERE status='UNKNOWN'").fetchone()[0]
        pending = db.execute("SELECT COUNT(*) FROM intents WHERE status='PENDING'").fetchone()[0]
        if unknown or pending:
            raise SystemExit(f"postflight blocked: UNKNOWN={unknown} PENDING={pending}")
    finally:
        db.close()
    isolation = audit_v5_isolation(output, arm=args.arm, formal=True)
    if isolation.get("status") != "PASS":
        raise SystemExit(json.dumps({"postflight": "isolation_failed", "gate": isolation}, ensure_ascii=False))
    print(json.dumps({"status": "PASS", "arm": args.arm, "output": str(output), "isolation": isolation}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
