"""Read-only path diagnosis for the frozen, purged Fly/Base paper replay."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from decimal import Decimal, getcontext
from pathlib import Path

from tools.summarize_purged_fly_base_performance import (
    RUN_DIR,
    SummaryBlocked,
    dec,
    read_events,
    read_settlement,
    sha256,
)

getcontext().prec = 50
ONE = Decimal("1")


def diagnose(run_dir: Path) -> dict:
    """Compute paired path facts without opening either ledger for writing."""
    run_dir = Path(run_dir).resolve()
    event_path = run_dir / "events.jsonl"
    summary_path = run_dir / "performance_summary.json"
    events = read_events(event_path)
    settlement = read_settlement(run_dir / "ledger.sqlite")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS_RESEARCH_ONLY":
        raise SummaryBlocked("frozen performance summary is not research-only PASS")

    fly_ratio = base_ratio = ONE
    annual = {}
    daily_gaps = []
    for year in ("2021", "2022", "2023"):
        rows = [row for row in events if row["date"].startswith(year)]
        if not rows:
            raise SummaryBlocked(f"no paired events for {year}")
        year_fly = year_base = ONE
        actions = Counter()
        for row in rows:
            reward = row.get("reward")
            decoder = row.get("decoder")
            if not isinstance(reward, dict) or not isinstance(decoder, dict):
                raise SummaryBlocked(f"missing reward or decoder on {row['date']}")
            fly_return = dec(reward.get("portfolio_return"), "Fly daily return")
            base_return = dec(reward.get("base_return"), "Base daily return")
            if fly_return <= -ONE or base_return <= -ONE:
                raise SummaryBlocked(f"invalid paired return on {row['date']}")
            side = decoder.get("action")
            if side not in {"EXPOSE", "DEEXPOSE", "DEFER"}:
                raise SummaryBlocked(f"invalid Fly action on {row['date']}")
            actions[side] += 1
            year_fly *= ONE + fly_return
            year_base *= ONE + base_return
            daily_gaps.append({
                "date": row["date"], "fly_action": side,
                "fly_return": str(fly_return), "base_return": str(base_return),
                "fly_minus_base_daily_return": str(fly_return - base_return),
            })
        fly_ratio *= year_fly
        base_ratio *= year_base
        annual[year] = {
            "sessions": len(rows),
            "fly_return": str(year_fly - ONE),
            "base_return": str(year_base - ONE),
            "fly_minus_base_return": str(year_fly - year_base),
            "actions": {key: actions[key] for key in ("EXPOSE", "DEEXPOSE", "DEFER")},
            "cumulative_fly_minus_base_return_at_year_end": str(fly_ratio - base_ratio),
        }

    fly_ratio *= ONE + dec(settlement["feedback"]["portfolio_return"], "Fly settlement return")
    base_ratio *= ONE + dec(settlement["feedback"]["base_return"], "Base settlement return")
    expected_fly = dec(summary["fly"]["performance"]["cumulative_return"], "summary Fly return")
    expected_base = dec(summary["base"]["performance"]["cumulative_return"], "summary Base return")
    if abs(fly_ratio - ONE - expected_fly) > Decimal("1e-24"):
        raise SummaryBlocked("Fly path does not reconcile to frozen summary")
    if abs(base_ratio - ONE - expected_base) > Decimal("1e-24"):
        raise SummaryBlocked("Base path does not reconcile to frozen summary")
    daily_gaps.sort(key=lambda item: (-abs(Decimal(item["fly_minus_base_daily_return"])), item["date"]))
    return {
        "schema": "stonkfly-purged-fly-base-path-diagnostic/v1",
        "status": "DESCRIPTIVE_POSTHOC_RESEARCH_ONLY",
        "source": {"events_sha256": sha256(event_path), "summary_sha256": sha256(summary_path)},
        "annual": annual,
        "largest_absolute_daily_return_gaps": daily_gaps[:10],
        "settled_fly_minus_base_return": str(fly_ratio - base_ratio),
        "interpretation": "Paired accounting path only; daily gaps are not causal action attribution or a significance test.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    if output.is_relative_to(args.run_dir.resolve()):
        raise SummaryBlocked("diagnostic output must not modify the frozen replay")
    result = diagnose(args.run_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
