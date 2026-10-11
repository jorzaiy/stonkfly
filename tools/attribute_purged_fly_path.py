"""Read-only accounting attribution for the frozen purged Fly/Base replay."""
from __future__ import annotations

import argparse
import json
import sqlite3
from decimal import Decimal, getcontext
from pathlib import Path

from tools.summarize_purged_fly_base_performance import RUN_DIR, dec, read_events, sha256

getcontext().prec = 50
D = Decimal


def _fills_by_year(path: Path) -> dict[str, dict[str, object]]:
    result = {year: {"fills": 0, "fees": D("0"), "notional": D("0")} for year in ("2021", "2022", "2023", "2024")}
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
        rows = db.execute("SELECT execution_date,fee,notional FROM fills ORDER BY execution_date,intent_id")
        for date, fee, notional in rows:
            year = str(date)[:4]
            if year not in result:
                raise ValueError(f"unexpected fill year: {date}")
            result[year]["fills"] += 1
            result[year]["fees"] += dec(fee, "fill fee")
            result[year]["notional"] += dec(notional, "fill notional")
    return result


def attribute(run_dir: Path) -> dict:
    run_dir = Path(run_dir).resolve()
    events_path = run_dir / "events.jsonl"
    events = read_events(events_path)
    fly_fills = _fills_by_year(run_dir / "ledger.sqlite")
    base_fills = _fills_by_year(run_dir / "paired_base.sqlite")
    annual = {}
    for year in ("2021", "2022", "2023"):
        rows = [row for row in events if row["date"].startswith(year)]
        exposure = [dec(row["actual_exposure"], "Fly actual exposure") for row in rows]
        if any(value < 0 for value in exposure):
            raise ValueError("negative gross exposure in event")
        fly_turnover = sum((dec(row["turnover"], "Fly turnover") for row in rows), D("0"))
        annual[year] = {
            "sessions": len(rows),
            "fly_mean_close_gross_exposure": str(sum(exposure, D("0")) / len(exposure)),
            "fly_zero_exposure_sessions": sum(value == 0 for value in exposure),
            "fly_close_gross_exposure_at_least_80pct_sessions": sum(value >= D("0.8") for value in exposure),
            "fly_event_turnover_notional": str(fly_turnover),
            "fly_fills": fly_fills[year]["fills"],
            "base_fills": base_fills[year]["fills"],
            "fly_fees": str(fly_fills[year]["fees"]),
            "base_fees": str(base_fills[year]["fees"]),
        }
    fly_fee = sum((item["fees"] for item in fly_fills.values()), D("0"))
    base_fee = sum((item["fees"] for item in base_fills.values()), D("0"))
    summary_path = run_dir / "performance_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["fly"]["ledger_counts"]["fills"] != sum(item["fills"] for item in fly_fills.values()):
        raise ValueError("Fly fill count does not match frozen summary")
    if summary["base"]["ledger_counts"]["fills"] != sum(item["fills"] for item in base_fills.values()):
        raise ValueError("Base fill count does not match frozen summary")
    return {
        "schema": "stonkfly-purged-fly-path-accounting/v1",
        "status": "DESCRIPTIVE_POSTHOC_RESEARCH_ONLY",
        "source_sha256": {
            "events": sha256(events_path), "summary": sha256(summary_path),
        },
        "annual": annual,
        "whole_window": {
            "fly_fees": str(fly_fee), "base_fees": str(base_fee),
            "fly_minus_base_fees": str(fly_fee - base_fee),
            "fly_final_equity": summary["fly"]["performance"]["final_equity"],
            "base_final_equity": summary["base"]["performance"]["final_equity"],
        },
        "interpretation": "Fees and exposures are observed accounting quantities. Subtracting fees from the final equity gap would not isolate a causal trading-cost effect because positions and compounding differ.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(args.run_dir.resolve()):
        raise ValueError("output must not alter the frozen run")
    result = attribute(args.run_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
