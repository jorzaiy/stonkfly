"""Read-only integrity audit of independent purged Fly paper controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from decimal import Decimal
from pathlib import Path

from experiments.a_share.run_purged_controls import SEEDS, permuted_actions, source_actions
from experiments.a_share.run_purged_fly_base_baseline import ROOT
from experiments.a_share.replay_wc006 import sha256


class AuditBlocked(RuntimeError):
    pass


def audit_arm(directory: Path, expected_actions: list[str] | None) -> dict:
    report_path = directory / "report.json"
    events_path = directory / "events.jsonl"
    ledger_path = directory / "ledger.sqlite"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    if len(events) != 609 or len({event["date"] for event in events}) != 609:
        raise AuditBlocked(f"{directory.name}: incomplete event calendar")
    if expected_actions is not None and [event["policy_action"] for event in events] != expected_actions:
        raise AuditBlocked(f"{directory.name}: permuted action sequence mismatch")
    if any(event["neural_output"] is not False for event in events):
        raise AuditBlocked(f"{directory.name}: control is mislabeled as neural")
    connection = sqlite3.connect(f"file:{ledger_path.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute("SELECT trade_date,payload_json,prev_hash,event_hash FROM events ORDER BY rowid").fetchall()
        if len(rows) != len(events):
            raise AuditBlocked(f"{directory.name}: SQLite/JSONL event count mismatch")
        prior = "GENESIS"
        for event, (date, payload_json, prev_hash, event_hash) in zip(events, rows):
            stored = dict(event)
            stored.pop("event_hash")
            if date != event["date"] or json.loads(payload_json) != stored or prev_hash != prior:
                raise AuditBlocked(f"{directory.name}: event chain content mismatch on {date}")
            if hashlib.sha256((prior + payload_json).encode()).hexdigest() != event_hash or event["event_hash"] != event_hash:
                raise AuditBlocked(f"{directory.name}: event hash mismatch on {date}")
            prior = event_hash
        if prior != report["event_chain_last_hash"]:
            raise AuditBlocked(f"{directory.name}: report chain tip mismatch")
        statuses = dict(connection.execute("SELECT status,COUNT(*) FROM intents GROUP BY status"))
        if statuses != report["intent_statuses"] or set(statuses) - {"FILLED", "VETO"}:
            raise AuditBlocked(f"{directory.name}: unresolved intent status")
        invalid_fills = connection.execute(
            "SELECT COUNT(*) FROM fills f LEFT JOIN intents i ON i.id=f.intent_id "
            "WHERE i.id IS NULL OR i.status!='FILLED' OR i.execution_date!=f.execution_date"
        ).fetchone()[0]
        if invalid_fills:
            raise AuditBlocked(f"{directory.name}: fill without settled prior intent")
        if directory.name == "base_with_retry":
            retries = []
            for event in events:
                for plan in event["planned"]:
                    if "source_veto_id" not in plan:
                        continue
                    source = connection.execute(
                        "SELECT signal_date,symbol,status,veto_reason FROM intents WHERE id=?",
                        (plan["source_veto_id"],),
                    ).fetchone()
                    if source is None or source[1] != plan["symbol"] or source[2] != "VETO" or source[0] >= event["date"]:
                        raise AuditBlocked("retry lacks an earlier same-symbol veto")
                    if plan["result"] == "PREPARED":
                        successor = connection.execute(
                            "SELECT signal_date,symbol FROM intents WHERE id=?",
                            (plan["intent_id"],),
                        ).fetchone()
                        if successor != (event["date"], plan["symbol"]):
                            raise AuditBlocked("retry intent changed date or symbol")
                    retries.append(plan)
            retry_count = sum(item["result"] == "PREPARED" for item in retries)
        else:
            retry_count = 0
    finally:
        connection.close()
    return {
        "arm": directory.name, "events": len(events), "chain_tip": prior,
        "ledger_sha256": sha256(ledger_path), "events_sha256": sha256(events_path),
        "report_sha256": sha256(report_path), "intent_statuses": statuses,
        "retry_intents": retry_count, "return": report["return"],
    }


def audit_controls(run: Path) -> dict:
    run = Path(run).resolve()
    dates, source = source_actions()
    expected = {"base_no_retry": None, "base_with_retry": None}
    expected.update({f"permuted_seed_{seed}": permuted_actions(source, seed) for seed in SEEDS})
    expected["fly_action_replay_parity"] = source
    arms = [audit_arm(run / name, action) for name, action in expected.items()]
    no_retry = next(item for item in arms if item["arm"] == "base_no_retry")
    summary = json.loads((ROOT / "runs/wc006_purged_fly_base_baseline_phase1/20260925T-frozen-rgb-seed17/performance_summary.json").read_text(encoding="utf-8"))
    if Decimal(no_retry["return"]) != Decimal(summary["base"]["performance"]["cumulative_return"]):
        raise AuditBlocked("no-retry Base parity failed")
    fly_replay = next(item for item in arms if item["arm"] == "fly_action_replay_parity")
    if Decimal(fly_replay["return"]) != Decimal(summary["fly"]["performance"]["cumulative_return"]):
        raise AuditBlocked("original action replay failed exact Fly parity")
    return {
        "schema": "wc006-purged-controls-audit/v1", "status": "PASS_RESEARCH_ONLY",
        "source_sessions": len(dates), "arms": arms,
        "implementation_sha256": sha256(ROOT / "experiments/a_share/run_purged_controls.py"),
        "protocol_sha256": sha256(ROOT / "docs/experiments/fly_purged_controls_preregister.md"),
        "interpretation": "Execution integrity only; no neural attribution or statistical significance claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(args.run.resolve()):
        raise AuditBlocked("audit output must not modify the control run")
    result = audit_controls(args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "arms": len(result["arms"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
