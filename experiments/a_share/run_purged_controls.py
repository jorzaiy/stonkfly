"""Independent paper ledgers for purged Fly execution controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any

from experiments.a_share.replay_wc006 import CostConfig, WC006Ledger, map_decoder_to_multiplier, sha256
from experiments.a_share.run_purged_fly_base_baseline import (
    CANDIDATE_SHA256, END_DATE, MARKET_PARENT_SHA256, MARKET_QA_SHA256,
    ROOT, SETTLEMENT_DATE, START_DATE, _validate_stream,
    validate_purged_candidate, validate_purged_market,
)
from experiments.a_share.wc006_male_cns_v4_adapter import _settle_v4
from experiments.a_share.wc006_male_cns_v51_adapter import (
    _create_batch, _prepare_target_intents, _resolve_batches, _scope_dates,
)
from experiments.a_share.wc006_v4_market import apply_corporate_actions, raw_equity, raw_mark

D = Decimal
RISK_BUDGET = D("0.85")
LOT = D("100")
SEEDS = (17, 23, 43)
SOURCE_EVENTS_SHA256 = "5bc2430aae1384d0cfbe1dcf9a969faecfafb88ff485b1ac2ff1089c1f43454b"
SOURCE_EVENTS = ROOT / "runs/wc006_purged_fly_base_baseline_phase1/20260925T-frozen-rgb-seed17/events.jsonl"
SPEC = ROOT / "docs/experiments/fly_purged_controls_preregister.md"


class ControlBlocked(RuntimeError):
    """A paper control input, execution, or output gate failed."""


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_actions(path: Path = SOURCE_EVENTS) -> tuple[list[str], list[str]]:
    if sha256(path) != SOURCE_EVENTS_SHA256:
        raise ControlBlocked("source Fly event hash mismatch")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    dates = [str(row.get("date")) for row in rows]
    actions = [row.get("decoder", {}).get("action") for row in rows]
    if len(rows) != 609 or dates != sorted(set(dates)) or dates[0] != START_DATE or dates[-1] != END_DATE:
        raise ControlBlocked("source Fly dates mismatch")
    if any(action not in {"EXPOSE", "DEEXPOSE", "DEFER"} for action in actions):
        raise ControlBlocked("source Fly action is invalid")
    return dates, actions


def permuted_actions(actions: list[str], seed: int) -> list[str]:
    if seed not in SEEDS:
        raise ControlBlocked("null seed is outside frozen protocol")
    indices = sorted(range(len(actions)), key=lambda index: hashlib.sha256(
        f"wc006-null-v1:{seed}:{index}".encode()).digest())
    return [actions[index] for index in indices]


def _intent_outcomes(ledger: WC006Ledger, execution_date: str) -> list[dict[str, str]]:
    rows = ledger.db.execute(
        "SELECT id,signal_date,symbol,status,COALESCE(veto_reason,'') FROM intents WHERE execution_date=? ORDER BY signal_date,symbol",
        (execution_date,),
    ).fetchall()
    return [dict(zip(("intent_id", "signal_date", "symbol", "status", "veto_reason"), map(str, row))) for row in rows]


def _retry_symbol(
    ledger: WC006Ledger, stream: Any, record: Any, symbol: str, date: str,
    observation_id: int, source_veto: dict[str, str],
) -> dict[str, str]:
    target = next((item for item in record.targets if str(item["code"]) == symbol), None)
    if target is None:
        raise ControlBlocked("retry symbol escaped current candidate basket")
    equity, _ = raw_equity(ledger, stream.market, date, {symbol})
    _, close = raw_mark(stream.market, symbol, date)
    exposure = D(str(target["weight"])) * RISK_BUDGET
    delta = equity * exposure - ledger.holdings(symbol) * close
    quantity = (abs(delta) / close / LOT).to_integral_value(rounding=ROUND_DOWN) * LOT
    if quantity == 0:
        return {"symbol": symbol, "source_veto_id": source_veto["intent_id"], "result": "ALREADY_AT_TARGET"}
    side = "BUY" if delta > 0 else "SELL"
    next_date = next((item for item in stream.base.calendar if item > date), None)
    if next_date is None:
        raise ControlBlocked("retry has no next trading session")
    intent_id = ledger.prepare_intent(
        signal_date=date, execution_date=next_date, symbol=symbol,
        neural_side=side, order_side=side,
        intent_kind="EXPOSE", target_exposure=exposure, requested_qty=quantity,
        observation_id=observation_id, checkpoint_sha256="base-with-retry-control",
    )
    return {
        "symbol": symbol, "source_veto_id": source_veto["intent_id"],
        "source_veto_reason": source_veto["veto_reason"],
        "intent_id": intent_id, "side": side, "quantity": str(quantity),
        "target_exposure": str(exposure), "result": "PREPARED",
    }


def _run_arm(stream: Any, dates: list[str], output: Path, arm: str, actions: list[str] | None) -> dict[str, Any]:
    if output.exists():
        raise ControlBlocked(f"control output already exists: {output}")
    output.mkdir(parents=True)
    signature = _json_hash({
        "schema": "wc006-purged-controls/v1", "arm": arm,
        "candidate": CANDIDATE_SHA256, "market": MARKET_PARENT_SHA256,
        "market_qa": MARKET_QA_SHA256, "spec": sha256(SPEC),
        "actions": _json_hash(actions) if actions is not None else None,
    })
    ledger = WC006Ledger(output / "ledger.sqlite", initial_state={"cash": "100000", "lots": []}, signature=signature, authoritative_trading_state=True)
    events_path = output / "events.jsonl"
    record_by_date = {item.signal_date: item for item in stream.base.records}
    active = record_by_date[dates[0]]
    candidate_epoch = 0
    outstanding: dict[str, dict[str, str]] = {}
    costs = CostConfig()
    try:
        ledger.put_meta("effective_multiplier", "1.0")
        ledger.db.commit()
        for index, date in enumerate(dates):
            apply_corporate_actions(ledger, stream.market, date, stream.base.calendar, stage="start_of_day")
            turnover = _settle_v4(ledger, date, stream.base.calendar, stream.market.raw_prices, costs, stream.market)
            apply_corporate_actions(ledger, stream.market, date, stream.base.calendar, stage="after_open_settlement")
            if ledger.unresolved():
                raise ControlBlocked("unknown outcome requires reconciliation")
            outcomes = _intent_outcomes(ledger, date)
            if any(item["status"] == "PREPARED" for item in outcomes):
                raise ControlBlocked("unsettled prepared intent")
            if actions is not None:
                resolved = _resolve_batches(ledger, date, D(ledger.get_meta("effective_multiplier")))
                # The null has no neural reward delivery. Close legacy batch
                # linkage explicitly so it cannot be mistaken for a pulse.
                for batch in resolved:
                    ledger.db.execute("UPDATE v51_batches SET reward_delivery_status='NONE',reward_label='none' WHERE batch_id=?", (batch["batch_id"],))
                ledger.db.commit()
            else:
                resolved = []
                if arm == "base_with_retry":
                    for item in outcomes:
                        if item["status"] == "FILLED":
                            outstanding.pop(item["symbol"], None)
                        elif item["status"] == "VETO":
                            outstanding[item["symbol"]] = item
            candidate = date in record_by_date
            if candidate:
                active = record_by_date[date]
                candidate_epoch += 1
                if actions is None:
                    outstanding.clear()  # A new basket supersedes old retries.
            active_codes = {str(item["code"]) for item in active.targets}
            equity, marks = raw_equity(ledger, stream.market, date, active_codes)
            previous = D(ledger.get_meta("previous_equity") or "100000")
            peak = max(D(ledger.get_meta("peak_equity") or "100000"), equity)
            drawdown = equity / peak - D("1")
            action = actions[index] if actions is not None else ("FIXED_TARGET" if candidate else "RETRY_OR_HOLD")
            observation_id = ledger.record_observation(
                date=date, input_sha256=_json_hash({"date": date, "arm": arm, "action": action}),
                reinforcement="none",
                neural={"control_policy": arm, "action": action, "neural_output": False},
                checkpoint={"control": arm, "date": date},
            )
            planned: list[dict[str, Any]] = []
            batch = None
            if actions is None:
                if candidate:
                    planned, _, _ = _prepare_target_intents(
                        ledger, active, date=date, multiplier=D("1.0"), neural_side="BUY",
                        observation_id=observation_id, checkpoint_sha256="base-with-retry-control",
                        stream=stream, target_codes=active_codes, intent_kind="EXPOSE",
                    )
                elif arm == "base_with_retry":
                    for symbol, veto in sorted(outstanding.items()):
                        planned.append(_retry_symbol(ledger, stream, active, symbol, date, observation_id, veto))
                    outstanding = {symbol: veto for symbol, veto in outstanding.items()
                                   if any(item.get("symbol") == symbol and item.get("result") == "PREPARED" for item in planned)}
            else:
                effective = D(ledger.get_meta("effective_multiplier"))
                side = {"EXPOSE": "BUY", "DEEXPOSE": "SELL", "DEFER": "HOLD"}[action]
                _, requested = map_decoder_to_multiplier(side, effective)
                if candidate or (action != "DEFER" and requested != effective):
                    plan_multiplier = requested if action != "DEFER" else effective
                    planned, ids, empty = _prepare_target_intents(
                        ledger, active, date=date, multiplier=plan_multiplier,
                        neural_side=side, observation_id=observation_id,
                        checkpoint_sha256=f"permuted-action-control:{arm}", stream=stream,
                        target_codes=active_codes,
                        intent_kind="EXPOSE" if plan_multiplier >= effective else "DEEXPOSE",
                    )
                    next_date = next((item for item in stream.base.calendar if item > date), None)
                    if next_date is None:
                        raise ControlBlocked("null action has no next trading session")
                    target = sum((D(str(item["weight"])) * RISK_BUDGET for item in active.targets), D("0"))
                    batch = _create_batch(
                        ledger, kind="candidate_selection" if candidate else "exposure_rebalance",
                        request_date=date, execution_date=next_date,
                        candidate_epoch=candidate_epoch, requested=plan_multiplier,
                        effective=effective, requested_target_exposure=target * plan_multiplier,
                        effective_target_exposure=target * effective,
                        intent_ids=ids, empty_outcome=empty,
                    )
            event = {
                "date": date, "arm": arm, "candidate_day": candidate,
                "policy_action": action, "neural_output": False,
                "equity": str(equity), "return": str(equity / previous - D("1")),
                "turnover": str(turnover), "outcomes": outcomes,
                "planned": json.loads(json.dumps(planned, default=str)),
                "batch": batch, "resolved_batches": resolved,
                "mark_sources": {code: item["source"] for code, item in marks.items()},
            }
            event_hash, _ = ledger.record_event(date=date, payload=event)
            event["event_hash"] = event_hash
            with events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            ledger.complete_date(date=date, index=index, multiplier=D(ledger.get_meta("effective_multiplier")), equity=equity, drawdown=drawdown, checkpoint={"control": arm, "date": date})
        if SETTLEMENT_DATE not in stream.base.calendar:
            raise ControlBlocked("settlement boundary missing")
        settled = _settle_v4(ledger, SETTLEMENT_DATE, stream.base.calendar, stream.market.raw_prices, costs, stream.market)
        if actions is not None:
            resolved = _resolve_batches(ledger, SETTLEMENT_DATE, D(ledger.get_meta("effective_multiplier")))
            for batch in resolved:
                ledger.db.execute("UPDATE v51_batches SET reward_delivery_status='NONE',reward_label='none' WHERE batch_id=?", (batch["batch_id"],))
        else:
            resolved = []
        if ledger.unresolved() or ledger.pending():
            raise ControlBlocked("unresolved settlement outcome")
        final_equity, marks = raw_equity(ledger, stream.market, SETTLEMENT_DATE, set())
        final = {
            "date": SETTLEMENT_DATE, "semantic": "settlement-only; no observation or new control action",
            "equity": str(final_equity), "return": str(final_equity / D(ledger.get_meta("previous_equity")) - D("1")),
            "turnover": str(settled), "outcomes": _intent_outcomes(ledger, SETTLEMENT_DATE),
            "resolved_batches": resolved, "mark_sources": {code: item["source"] for code, item in marks.items()},
        }
        ledger.put_meta("final_settlement", final)
        ledger.db.commit()
        ledger.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        counts = dict(ledger.db.execute("SELECT status,COUNT(*) FROM intents GROUP BY status"))
        report = {
            "schema": "wc006-purged-controls/v1", "arm": arm,
            "paper_only": True, "neural_output": False,
            "sessions": len(dates), "settlement_date": SETTLEMENT_DATE,
            "final_equity": str(final_equity),
            "return": str(final_equity / D("100000") - D("1")),
            "intent_statuses": counts, "source_events_sha256": SOURCE_EVENTS_SHA256,
            "spec_sha256": sha256(SPEC), "candidate_sha256": CANDIDATE_SHA256,
            "market_sha256": MARKET_PARENT_SHA256, "market_qa_sha256": MARKET_QA_SHA256,
            "event_chain_last_hash": event_hash,
            "action_sequence_sha256": _json_hash(actions) if actions is not None else None,
        }
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report
    finally:
        ledger.close()


def run_controls(candidate: Path, market: Path, output: Path) -> list[dict[str, Any]]:
    """Run all frozen controls from clean, independent paper ledgers."""
    candidate, market, output = Path(candidate).resolve(), Path(market).resolve(), Path(output).resolve()
    if not output.is_relative_to((ROOT / "runs/wc006_purged_controls").resolve()):
        raise ControlBlocked("output must use the purged controls namespace")
    if output.exists():
        raise ControlBlocked("control output must be new")
    validate_purged_candidate(candidate)
    validate_purged_market(market, json.loads(candidate.read_text(encoding="utf-8")))
    source_dates, source = source_actions()
    stream = _validate_stream(candidate, market)
    dates, settlement = _scope_dates(stream, START_DATE, END_DATE, SETTLEMENT_DATE)
    if dates != source_dates or settlement != SETTLEMENT_DATE:
        raise ControlBlocked("control dates differ from source Fly dates")
    output.mkdir(parents=True)
    reports = [_run_arm(stream, dates, output / "base_no_retry", "base_no_retry", None)]
    baseline_summary = json.loads((ROOT / "runs/wc006_purged_fly_base_baseline_phase1/20260925T-frozen-rgb-seed17/performance_summary.json").read_text(encoding="utf-8"))
    expected_base = D(baseline_summary["base"]["performance"]["final_equity"])
    if D(reports[0]["final_equity"]) != expected_base:
        raise ControlBlocked("independent no-retry Base failed exact baseline parity")
    reports.append(_run_arm(stream, dates, output / "base_with_retry", "base_with_retry", None))
    for seed in SEEDS:
        arm = f"permuted_seed_{seed}"
        reports.append(_run_arm(stream, dates, output / arm, arm, permuted_actions(source, seed)))
    (output / "controls_report.json").write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--market", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_controls(args.candidate, args.market, args.out), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
