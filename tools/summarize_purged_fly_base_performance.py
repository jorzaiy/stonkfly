"""Read-only performance summarizer for the frozen purged Fly/Base run.

This tool never opens a ledger in write mode and never touches events or
outcomes.  It compounds the daily rewards in ``events.jsonl``, adds the
explicit settlement-only boundary from the Fly ledger, and checks both
products against the settlement equity before writing independent summaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from datetime import date
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any

getcontext().prec = 50

STARTING_EQUITY = Decimal("100000")
RUN_DIR = Path("runs/wc006_purged_fly_base_baseline_phase1/20260925T-frozen-rgb-seed17")
SETTLEMENT_META_KEY = "final_settlement"
SUMMARY_JSON = "performance_summary.json"
SUMMARY_MD = "performance_summary.md"
AUDIT_MANIFEST = "performance_audit_manifest.json"


class SummaryBlocked(RuntimeError):
    """Fail closed when the frozen run cannot be summarized safely."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dec(value: Any, label: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise SummaryBlocked(f"invalid Decimal {label}: {value!r}") from exc
    if not result.is_finite():
        raise SummaryBlocked(f"non-finite Decimal {label}")
    return result


def read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SummaryBlocked(f"invalid event JSON at line {line_number}") from exc
            if not isinstance(event, dict):
                raise SummaryBlocked(f"event line {line_number} is not an object")
            events.append(event)
    if not events:
        raise SummaryBlocked("events.jsonl is empty")
    dates = [str(item.get("date")) for item in events]
    if dates != sorted(dates) or len(set(dates)) != len(dates):
        raise SummaryBlocked("events are not strictly date ordered")
    if dates[0] != "2021-07-01" or dates[-1] != "2023-12-29":
        raise SummaryBlocked("event date window is not the frozen 2021-07-01..2023-12-29 window")
    for value in dates:
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise SummaryBlocked(f"invalid event date: {value}") from exc
    return events


def read_settlement(ledger_path: Path) -> dict[str, Any]:
    uri = f"file:{ledger_path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        row = connection.execute(
            "SELECT value FROM meta WHERE key = ?", (SETTLEMENT_META_KEY,)
        ).fetchone()
    if row is None:
        raise SummaryBlocked("Fly ledger has no final_settlement meta")
    try:
        settlement = json.loads(row[0])
    except json.JSONDecodeError as exc:
        raise SummaryBlocked("final_settlement is not valid JSON") from exc
    if not isinstance(settlement, dict):
        raise SummaryBlocked("final_settlement is not an object")
    if settlement.get("date") != "2024-01-02":
        raise SummaryBlocked("unexpected settlement date")
    if settlement.get("semantic") != "settlement-only; no boundary observation, signal, or learning pulse":
        raise SummaryBlocked("unexpected settlement semantic")
    feedback = settlement.get("feedback")
    if not isinstance(feedback, dict):
        raise SummaryBlocked("settlement feedback is missing")
    for key in ("portfolio_return", "base_return"):
        dec(feedback.get(key), f"settlement feedback {key}")
    dec(settlement.get("equity"), "settlement equity")
    dec(settlement.get("base_equity"), "settlement base equity")
    return settlement


def ledger_counts(path: Path) -> dict[str, Any]:
    uri = f"file:{path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        statuses = dict(connection.execute("SELECT status, COUNT(*) FROM intents GROUP BY status"))
        intents = int(connection.execute("SELECT COUNT(*) FROM intents").fetchone()[0])
        fills = int(connection.execute("SELECT COUNT(*) FROM fills").fetchone()[0])
    return {
        "intents": intents,
        "fills": fills,
        "vetoes": int(statuses.get("VETO", 0)),
        "unknown": int(statuses.get("UNKNOWN", 0)),
        "intent_statuses": {str(key): int(value) for key, value in sorted(statuses.items())},
    }


def compound(events: list[dict[str, Any]], settlement: dict[str, Any], key: str) -> dict[str, Any]:
    equity = STARTING_EQUITY
    peak = equity
    max_drawdown = Decimal("0")
    positive_days = 0
    product = Decimal("1")
    path: list[dict[str, Any]] = []
    for event in events:
        reward = event.get("reward")
        if not isinstance(reward, dict):
            raise SummaryBlocked(f"missing reward on {event.get('date')}")
        daily_return = dec(reward.get(key), f"{event.get('date')} reward {key}")
        if daily_return > 0:
            positive_days += 1
        product *= Decimal("1") + daily_return
        equity *= Decimal("1") + daily_return
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, Decimal("1") - equity / peak)
        path.append({"date": event["date"], "equity": equity})

    settlement_return = dec(settlement["feedback"].get(key), f"settlement feedback {key}")
    if settlement_return > 0:
        positive_days += 1
    product *= Decimal("1") + settlement_return
    equity *= Decimal("1") + settlement_return
    peak = max(peak, equity)
    max_drawdown = max(max_drawdown, Decimal("1") - equity / peak)
    path.append({"date": settlement["date"], "equity": equity})

    expected_equity = dec(settlement["equity" if key == "portfolio_return" else "base_equity"], "final equity")
    difference = equity / STARTING_EQUITY - expected_equity / STARTING_EQUITY
    # Event rewards are persisted at finite Decimal precision; the expected
    # difference is bounded multiplication round-off (about 1e-27 here).
    if abs(difference) > Decimal("1e-24"):
        raise SummaryBlocked(f"{key} compound does not match final settlement ratio: {difference}")
    return {
        "final_equity": str(expected_equity),
        "cumulative_return": str(expected_equity / STARTING_EQUITY - Decimal("1")),
        "cumulative_return_percent": str((expected_equity / STARTING_EQUITY - Decimal("1")) * Decimal("100")),
        "max_drawdown": str(max_drawdown),
        "max_drawdown_percent": str(max_drawdown * Decimal("100")),
        "positive_return_days": positive_days,
        "event_positive_return_days": positive_days - int(settlement_return > 0),
        "settlement_positive_return": settlement_return > 0,
        "return_intervals": len(path),
        "compound_product": str(product),
        "compound_final_ratio": str(equity / STARTING_EQUITY),
        "settlement_final_ratio": str(expected_equity / STARTING_EQUITY),
        "compound_difference": str(difference),
    }


def independence_evidence(root: Path) -> dict[str, Any]:
    source = root / "experiments/a_share/wc006_male_cns_v51_adapter.py"
    lines = source.read_text(encoding="utf-8").splitlines()
    matches = [
        index
        for index, line in enumerate(lines, 1)
        if "_prepare_target_intents(base, base_record" in line
        and 'multiplier=D("1.0")' in line
        and 'neural_side="BUY"' in line
    ]
    if len(matches) != 1:
        raise SummaryBlocked(
            f"expected one fixed paired-base action call, found {len(matches)}"
        )
    index = matches[0]
    selected = {str(index): lines[index - 1]}
    return {
        "source": "experiments/a_share/wc006_male_cns_v51_adapter.py",
        "source_sha256": sha256(source),
        "lines": str(index),
        "excerpt": selected,
        "finding": (
            "Base uses the candidate record with fixed multiplier 1.0 and fixed BUY neural_side; "
            "the Fly neural action/side is not read. The shared observation_id is provenance only."
        ),
    }


def build_summary(run_dir: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    events_path = run_dir / "events.jsonl"
    settlement = read_settlement(run_dir / "ledger.sqlite")
    events = read_events(events_path)
    fly = ledger_counts(run_dir / "ledger.sqlite")
    base = ledger_counts(run_dir / "paired_base.sqlite")
    fly_performance = compound(events, settlement, "portfolio_return")
    base_performance = compound(events, settlement, "base_return")
    actions = Counter(str(event.get("decoder", {}).get("action", "MISSING")) for event in events)
    candidate_days = sum(bool(event.get("signal_eligible")) for event in events)
    summary = {
        "schema": "stonkfly-purged-fly-base-performance-summary/v1",
        "status": "PASS_RESEARCH_ONLY",
        "run_dir": str(run_dir),
        "event_count": len(events),
        "event_window": {"start": events[0]["date"], "end": events[-1]["date"]},
        "settlement_date": settlement["date"],
        "starting_equity": str(STARTING_EQUITY),
        "fly": {
            "performance": fly_performance,
            "action_distribution": dict(sorted(actions.items())),
            "ledger_counts": fly,
        },
        "base": {
            "performance": base_performance,
            "action_distribution": {
                "FIXED_TARGET_BASKET": candidate_days,
                "NO_ACTION": len(events) - candidate_days,
            },
            "ledger_counts": base,
            "hypothesis": "fixed_zero_selector_independent_base",
        },
        "relative": {
            "fly_minus_base_return": str(
                fly_performance["cumulative_return"] and dec(fly_performance["cumulative_return"], "Fly return")
                - dec(base_performance["cumulative_return"], "Base return")
            ),
            "fly_lag_base_percentage_points": str(
                (dec(base_performance["cumulative_return"], "Base return")
                 - dec(fly_performance["cumulative_return"], "Fly return")) * Decimal("100")
            ),
        },
        "base_action_independence": independence_evidence(root),
        "interpretation": (
            "This is an engineered reinforcement/replay accounting summary only. It makes no "
            "claim about consciousness, pain, pleasure, learning validity, statistical significance, "
            "or profitability."
        ),
    }
    inputs = [
        "baseline_report.json", "provenance.json", "ledger.sqlite", "paired_base.sqlite",
        "events.jsonl", "observation_outcomes.jsonl", "forward_outcomes.jsonl",
    ]
    prereg = ["docs/experiments/fly_purged_baseline_phase1_preregister.md", "docs/experiments/fly_purged_baseline_phase1_preregister.json"]
    runner = root / "experiments/a_share/run_purged_fly_base_baseline.py"
    audit = {
        "schema": "stonkfly-purged-fly-base-performance-audit/v1",
        "status": "PASS_RESEARCH_ONLY",
        "run_dir": str(run_dir),
        "summary": {"json": SUMMARY_JSON, "markdown": SUMMARY_MD},
        "artifacts": {name: {"path": name, "sha256": sha256(run_dir / name)} for name in inputs},
        "preregister": {name: {"path": name, "sha256": sha256(root / name)} for name in prereg},
        "runner": {"path": str(runner.relative_to(root)), "sha256": sha256(runner)},
        "summarizer": {"path": str(Path(__file__).resolve().relative_to(root)), "sha256": sha256(Path(__file__).resolve())},
        "base_action_independence": summary["base_action_independence"],
        "notes": "Independent read-only summarizer output; original baseline_report, events, outcomes, and ledgers are unchanged.",
    }
    return summary, audit


def markdown(summary: dict[str, Any]) -> str:
    fly = summary["fly"]["performance"]
    base = summary["base"]["performance"]
    rel = summary["relative"]
    return f"""# Purged Fly/Base phase-1 performance summary

Status: `{summary['status']}`; event window `{summary['event_window']['start']}..{summary['event_window']['end']}`; settlement `{summary['settlement_date']}`.

| Arm | Final equity | Cumulative return | Max drawdown | Positive return days |
| --- | ---: | ---: | ---: | ---: |
| Fly | {fly['final_equity']} | {Decimal(fly['cumulative_return_percent']):.2f}% | {Decimal(fly['max_drawdown_percent']):.2f}% | {fly['positive_return_days']} |
| Base | {base['final_equity']} | {Decimal(base['cumulative_return_percent']):.2f}% | {Decimal(base['max_drawdown_percent']):.2f}% | {base['positive_return_days']} |

Fly is behind Base by `{Decimal(rel['fly_lag_base_percentage_points']):.2f}` percentage points (Fly minus Base is `{Decimal(rel['fly_minus_base_return']) * 100:.2f}` pp). This is an accounting result, not a significance or profitability claim.

Fly actions: `{json.dumps(summary['fly']['action_distribution'], sort_keys=True)}`. Base fixed-policy actions: `{json.dumps(summary['base']['action_distribution'], sort_keys=True)}`.

Ledger counts (intents / fills / vetoes / UNKNOWN): Fly `{summary['fly']['ledger_counts']['intents']} / {summary['fly']['ledger_counts']['fills']} / {summary['fly']['ledger_counts']['vetoes']} / {summary['fly']['ledger_counts']['unknown']}`; Base `{summary['base']['ledger_counts']['intents']} / {summary['base']['ledger_counts']['fills']} / {summary['base']['ledger_counts']['vetoes']} / {summary['base']['ledger_counts']['unknown']}`.

Both compounded event reward products include the explicit settlement-only boundary and match `final_settlement` equity/100000 within exact Decimal arithmetic. Base independence evidence is bound in `performance_audit_manifest.json`, lines {summary['base_action_independence']['lines']} of the pinned adapter source: fixed multiplier `1.0`, fixed `BUY`, and no Fly decoder output read.

This report describes engineered reinforcement/replay accounting only. It does not claim consciousness, pain, pleasure, validated profitable learning, statistical significance, or profitability.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path, default=RUN_DIR)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    root = Path(__file__).resolve().parents[1]
    summary, audit = build_summary(run_dir, root)
    (run_dir / SUMMARY_JSON).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / SUMMARY_MD).write_text(markdown(summary), encoding="utf-8")
    # Bind the just-written summaries only after their complete bytes exist.
    audit["summary"]["json_sha256"] = sha256(run_dir / SUMMARY_JSON)
    audit["summary"]["markdown_sha256"] = sha256(run_dir / SUMMARY_MD)
    (run_dir / AUDIT_MANIFEST).write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(run_dir / SUMMARY_JSON), "audit": str(run_dir / AUDIT_MANIFEST), "fly_return": summary["fly"]["performance"]["cumulative_return"], "base_return": summary["base"]["performance"]["cumulative_return"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
