#!/usr/bin/env python3
"""Read-only, Decimal-reconciled Fly/Base 2024–2025 heldout accounting."""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from decimal import Decimal
from pathlib import Path

from tools.collect_purged_xysz_2024_2025 import sha, write_json
from tools.summarize_purged_fly_base_performance import (
    STARTING_EQUITY, SummaryBlocked, compound, ledger_counts,
)

START, END, SETTLE = "2024-01-02", "2025-12-29", "2025-12-30"


def summarize(run: Path) -> dict:
    run = run.resolve()
    report = json.loads((run / "heldout_report.json").read_text())
    if report.get("status") != "PASS_RESEARCH_ONLY" or report.get("research_protocol", {}).get("date_window") != [START, END]:
        raise SummaryBlocked("heldout_run_incomplete_or_scope_mismatch")
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    dates = [str(e.get("date")) for e in events]
    if not dates or dates != sorted(set(dates)) or dates[0] != START or dates[-1] != END:
        raise SummaryBlocked("heldout_event_window_mismatch")
    uri = f"file:{(run / 'ledger.sqlite').resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        row = db.execute("SELECT value FROM meta WHERE key='final_settlement'").fetchone()
    if row is None:
        raise SummaryBlocked("final_settlement_missing")
    settlement = json.loads(row[0])
    if settlement.get("date") != SETTLE or settlement.get("semantic") != "settlement-only; no boundary observation, signal, or learning pulse":
        raise SummaryBlocked("settlement_boundary_mismatch")
    fly = compound(events, settlement, "portfolio_return")
    base = compound(events, settlement, "base_return")
    fly_return = Decimal(fly["cumulative_return"])
    base_return = Decimal(base["cumulative_return"])
    counts = {"fly": ledger_counts(run / "ledger.sqlite"),
              "base": ledger_counts(run / "paired_base.sqlite")}
    if any(x["unknown"] for x in counts.values()):
        raise SummaryBlocked("unknown_exchange_outcome_in_paper_ledger")
    result = {"schema": "fly-purged-heldout-2024-2025/performance/v1", "status": "PASS_RESEARCH_ONLY",
              "start_equity": str(STARTING_EQUITY), "event_window": [START, END], "settlement": SETTLE,
              "fly": fly, "base": base,
              "fly_minus_base_percentage_points": str((fly_return - base_return) * 100),
              "fly_nonnegative_gate_passed": fly_return >= 0,
              "action_distribution": dict(Counter(str(e.get("decoder", {}).get("action", "MISSING")) for e in events)),
              "ledger_counts": counts,
              "inputs_sha256": {name: sha(run / name) for name in ("heldout_report.json", "provenance.json", "ledger.sqlite", "paired_base.sqlite", "events.jsonl")},
              "interpretation": "Engineered reinforcement and paper accounting only; no validated profitable learning or subjective experience claim."}
    out = run / "heldout_performance.json"
    if out.exists():
        if json.loads(out.read_text()) != result:
            raise SummaryBlocked("existing_heldout_performance_mismatch")
    else:
        write_json(out, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
