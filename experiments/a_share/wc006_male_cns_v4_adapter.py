"""WC006 v4 daily-observation, signal-only paper adapter.

This is an independent v4 adapter.  It preserves the complete MaleCNS graph,
uses a clean brain, and never talks to a broker.  The production entry point is
intentionally exercised only by hermetic tests in this phase.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw

from experiments.a_share.replay_wc006 import (
    CostConfig,
    D,
    WC006Ledger,
    _lot_quantity,
    commission,
    incremental_feedback,
    map_decoder_to_multiplier,
    sha256,
)
from experiments.a_share.wc006_male_cns_adapter import (
    AdapterBlocked,
    StreamRecord,
    ValidatedStream,
    _code,
    _date,
    _next_calendar,
    validate_candidate_stream,
)
from experiments.a_share.wc006_v4_market import (
    EXPECTED_PARENT_SHA256,
    EXPECTED_QA_SHA256,
    MarketInputsBlocked,
    XyszMarketInputs,
    apply_corporate_actions,
    execution_guard,
    load_xysz_market_inputs,
    raw_equity,
    raw_open_equity,
    raw_open_mark,
    raw_mark,
    raw_execution_price,
)
from stonkfly.config import Settings
from stonkfly.neural.controller import FlyController, deterministic_neural_payload

V4_SCHEMA = "wc006-male-cns-paper/v4"
RISK_BUDGET = D("0.85")
LOOKBACK = 60
LOOKBACK_CLOSES = 61
TRAIN_START = "2021-07-01"
TRAIN_END = "2023-12-29"


def _adapter_sha256() -> str:
    """Hash the exact adapter source so resume cannot cross code revisions."""
    return sha256(Path(__file__).resolve())


@dataclass(frozen=True)
class V4Stream:
    base: ValidatedStream
    prices: dict[str, dict[str, dict[str, Decimal]]]
    spec_sha256: str
    market: XyszMarketInputs | None = None


def _load_price_rows(stream: ValidatedStream) -> dict[str, dict[str, dict[str, Decimal]]]:
    result: dict[str, dict[str, dict[str, Decimal]]] = {}
    for code, path in stream.cache_paths.items():
        rows: dict[str, dict[str, Decimal]] = {}
        previous = None
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"date", "open", "close", "volume"}
            if not required <= set(reader.fieldnames or ()):
                raise AdapterBlocked(f"{code}: v4 requires OHLCV fields")
            for raw in reader:
                date = _date(raw.get("date"))
                if previous is not None and date <= previous:
                    raise AdapterBlocked(f"{code}: cache dates are not increasing")
                previous = date
                try:
                    open_price, close = D(raw["open"]), D(raw["close"])
                    volume = D(raw["volume"]) if str(raw.get("volume", "")).strip() else None
                except (KeyError, ValueError) as exc:
                    raise AdapterBlocked(f"{code}: invalid OHLCV at {date}") from exc
                if open_price <= 0 or close <= 0 or (volume is not None and volume < 0):
                    raise AdapterBlocked(f"{code}: invalid OHLCV values at {date}")
                rows[date] = {"open": open_price, "close": close, "volume": volume}
        result[code] = rows
    return result


def validate_v4_stream(path: Path, *, expected_records: int = 46, lookback: int = LOOKBACK, training_end: str = TRAIN_END, market_manifest: Path | None = None, market_expected_parent_sha256: str | None = None, market_expected_qa_sha256: str | None = None) -> V4Stream:
    if lookback != LOOKBACK:
        raise AdapterBlocked("v4 lookback is frozen at 60 sessions")
    if expected_records == 46 and training_end != TRAIN_END:
        raise AdapterBlocked("formal v4 training_end is frozen at TRAIN_END")
    base = validate_candidate_stream(path, expected_records=expected_records)
    prices = _load_price_rows(base)
    first = base.records[0].signal_date
    final = base.records[-1].execution_date
    required_dates = tuple(item for item in base.calendar if first <= item <= final)
    if any(len([item for item in prices[target["code"]] if item <= first]) < LOOKBACK_CLOSES for record in base.records[:1] for target in record.targets):
        raise AdapterBlocked("pinned cache cannot satisfy the pre-signal 61-bar lookback")
    current: StreamRecord | None = None
    for date in required_dates:
        if date in {record.signal_date for record in base.records}:
            current = next(record for record in base.records if record.signal_date == date)
        if current is None:
            raise AdapterBlocked("observation before first signal")
        history = tuple(item for item in base.calendar if item <= date)
        if len(history) < LOOKBACK_CLOSES:
            raise AdapterBlocked(f"{date}: missing 61 close rows for 60 return intervals")
        for target in current.targets:
            code = target["code"]
            available = [item for item in prices.get(code, {}) if item <= date]
            if len(available) < LOOKBACK_CLOSES:
                raise AdapterBlocked(f"{code}: fewer than 61 available bars at {date}")
    spec = Path(__file__).resolve().parents[2] / "docs/experiments/wc006_v4_2021_2025_preregister.md"
    if not spec.is_file():
        raise AdapterBlocked("v4 preregistration is missing")
    market = None
    if expected_records == 46 and market_manifest is None:
        raise AdapterBlocked("formal v4 requires hash-pinned xysz market manifest")
    # A supplied manifest is always loaded (fail closed): silently falling back
    # to QFQ fills when a caller pinned raw execution inputs is a protocol breach
    # (audit/p0-2021-002432-concentration).
    if market_manifest is not None:
        try:
            market = load_xysz_market_inputs(
                Path(market_manifest),
                expected_parent_sha256=market_expected_parent_sha256 or EXPECTED_PARENT_SHA256,
                expected_qa_sha256=market_expected_qa_sha256 or EXPECTED_QA_SHA256,
            )
        except MarketInputsBlocked as exc:
            raise AdapterBlocked(str(exc)) from exc
    return V4Stream(base, prices, sha256(spec), market)


def _held_codes(ledger: WC006Ledger) -> set[str]:
    return {str(row[0]) for row in ledger.db.execute("SELECT DISTINCT symbol FROM lots WHERE CAST(quantity AS REAL)>0")}


def _asof_bar(prices: dict[str, dict[str, dict[str, Decimal]]], code: str, date: str) -> tuple[str, dict[str, Decimal | None]]:
    available = [item for item in prices.get(code, {}) if item <= date]
    if not available:
        raise AdapterBlocked(f"no causal bar for {code} at {date}")
    mark_date = max(available)
    return mark_date, prices[code][mark_date]


def _mark_provenance(ledger: WC006Ledger, prices: dict[str, dict[str, dict[str, Decimal]]], date: str, active: set[str]) -> dict[str, dict[str, object]]:
    result = {}
    for code in sorted(set(active) | _held_codes(ledger)):
        mark_date, row = _asof_bar(prices, code, date)
        result[code] = {"mark_date": mark_date, "stale": mark_date != date, "missing_bar": date not in prices.get(code, {}), "volume_missing": row.get("volume") is None}
    return result


def _equity(ledger: WC006Ledger, prices: dict[str, dict[str, dict[str, Decimal]]], date: str, field: str, active: set[str]) -> Decimal:
    total = D(ledger.get_meta("cash"))
    for code in set(active) | _held_codes(ledger):
        _, row = _asof_bar(prices, code, date)
        value = row.get(field)
        if value is None:
            raise AdapterBlocked(f"missing causal {field} mark for {code} at {date}")
        total += ledger.holdings(code) * value
    return total



def _open_mark(prices: dict[str, dict[str, dict[str, Decimal]]], code: str, date: str) -> tuple[Decimal, bool]:
    if code in prices and date in prices[code] and prices[code][date].get("open") is not None:
        return prices[code][date]["open"], False
    mark_date, row = _asof_bar(prices, code, date)
    close = row.get("close")
    if close is None:
        raise AdapterBlocked(f"missing causal close mark for {code} at {date}")
    return close, True


def _open_equity(ledger: WC006Ledger, prices: dict[str, dict[str, dict[str, Decimal]]], date: str, active: set[str]) -> Decimal:
    total = D(ledger.get_meta("cash"))
    for code in set(active) | _held_codes(ledger):
        mark, _ = _open_mark(prices, code, date)
        total += ledger.holdings(code) * mark
    return total

def _settle_v4(
    ledger: WC006Ledger,
    date: str,
    calendar: tuple[str, ...],
    prices: dict[str, dict[str, dict[str, Decimal]]],
    costs: CostConfig,
    market: XyszMarketInputs | None = None,
) -> Decimal:
    pending = sorted(ledger.pending(date), key=lambda item: (0 if item["order_side"] == "SELL" else 1, str(item["symbol"]), str(item["id"])))
    active = {str(item["symbol"]) for item in pending}
    open_equity = raw_open_equity(ledger, market, date, active)[0] if market is not None else _open_equity(ledger, prices, date, active)
    turnover = D("0")
    for intent in pending:
        code = str(intent["symbol"])
        if market is None and (code not in prices or date not in prices[code] or prices[code][date].get("open") is None):
            ledger.mark_veto(str(intent["id"]), "missing_open_or_unverified_suspension")
            continue
        quantity = D(intent["requested_qty"])
        side = str(intent["order_side"])
        if market is not None:
            try:
                price = raw_execution_price(market, code, date, side, costs.slippage)
            except MarketInputsBlocked as exc:
                ledger.mark_veto(str(intent["id"]), str(exc))
                continue
        else:
            price = prices[code][date]["open"] * (D("1") + costs.slippage if side == "BUY" else D("1") - costs.slippage)
        fee = commission(quantity * price, costs.commission_rate, costs.minimum_commission)
        if side == "SELL":
            fee += quantity * price * costs.stamp_duty_rate
        reason = None
        if side == "SELL" and ledger.holdings(code, sellable_on=date) < quantity:
            reason = "t_plus_1_or_inventory"
        elif side == "BUY":
            cash = D(ledger.get_meta("cash"))
            if cash < quantity * price + fee:
                reason = "cash_and_fee"
            else:
                gross = sum(ledger.holdings(item) * (raw_open_mark(market, item, date)[1] if market is not None else _open_mark(prices, item, date)[0]) for item in (_held_codes(ledger) | {code}))
                if open_equity <= 0 or (gross + quantity * price) / open_equity > RISK_BUDGET:
                    reason = "total_exposure"
        if reason:
            ledger.mark_veto(str(intent["id"]), reason)
        else:
            ledger.settle(str(intent["id"]), execution_date=date, quantity=quantity, price=price, fee=fee, sellable_from=_next_calendar(calendar, date))
            turnover += quantity * price
    return turnover

def _stats(values: list[Decimal], window: int) -> tuple[float, float]:
    if len(values) < window + 1:
        raise AdapterBlocked(f"missing {window}-session lookback")
    tail = values[-(window + 1):]
    ret = float(tail[-1] / tail[0] - D("1"))
    changes = [float(tail[i] / tail[i - 1] - D("1")) for i in range(1, len(tail))]
    vol = float(np.std(np.asarray(changes, dtype=np.float64))) if changes else 0.0
    return ret, vol


def _frame(
    record: StreamRecord,
    prices: dict[str, dict[str, dict[str, Decimal]]],
    date: str,
    calendar: tuple[str, ...],
    ledger: WC006Ledger,
    multiplier: Decimal,
    drawdown: Decimal,
) -> np.ndarray:
    dates = [item for item in calendar if item <= date]
    history_dates = dates[-LOOKBACK_CLOSES:]
    equity = _equity(ledger, prices, date, "close", {item["code"] for item in record.targets})
    cash_fraction = D(ledger.get_meta("cash")) / equity if equity else D("0")
    image = Image.new("RGB", (320, 180), (235, 240, 249))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 319, 20), fill=(19, 36, 71))
    draw.text((4, 4), f"wc006v4 {date} M{multiplier:.1f} C{float(cash_fraction):.2f} D{float(drawdown):.2f}", fill=(219, 229, 249))
    row_height = max(1, 150 // len(record.targets))
    for index, target in enumerate(record.targets):
        code = target["code"]
        bars = [_asof_bar(prices, code, item) for item in history_dates]
        available_bars = sorted((bar_date, row) for bar_date, row in prices.get(code, {}).items() if bar_date <= date)[-LOOKBACK_CLOSES:]
        if len(available_bars) < LOOKBACK_CLOSES:
            raise AdapterBlocked(f"{code}: frame lookback incomplete at {date}")
        feature_closes = [row["close"] for _, row in available_bars]
        feature_volumes = [row.get("volume") or D("0") for _, row in available_bars]
        normalized = [float(value / feature_closes[0] - D("1")) for value in feature_closes[-LOOKBACK:]]
        vmax = max(feature_volumes) or D("1")
        volume_norm = [float(value / vmax) for value in feature_volumes[-LOOKBACK:]]
        r5, v5 = _stats(feature_closes, 5)
        r20, v20 = _stats(feature_closes, 20)
        r60, v60 = _stats(feature_closes, 60)
        current_exposure = ledger.holdings(code) * feature_closes[-1] / equity if equity else D("0")
        target_exposure = D(str(target["weight"])) * RISK_BUDGET * multiplier
        top = 24 + index * row_height
        bottom = min(176, top + row_height - 4)
        stale = int(date not in prices.get(code, {}))
        draw.text((3, top), f"{code} T{float(target_exposure):.3f} E{float(current_exposure):.3f} M{stale} r{r5:.2%}/{r20:.2%}/{r60:.2%}"[:38], fill=(28, 46, 82))
        for series, color in ((normalized, (0, 101, 183)), (volume_norm, (160, 90, 30))):
            lo, hi = min(series), max(series)
            span = max(hi - lo, 1e-12)
            points = []
            for pos, value in enumerate(series):
                x = 60 + pos * 255 / max(1, len(series) - 1)
                y = bottom - (value - lo) / span * max(2, row_height - 18)
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill=color, width=1)
        draw.text((4, bottom - 10), f"v{v5:.3f}/{v20:.3f}/{v60:.3f}", fill=(92, 65, 42))
    return np.asarray(image, dtype=np.uint8)


def _prepare(
    ledger: WC006Ledger,
    record: StreamRecord,
    prices: dict[str, dict[str, dict[str, Decimal]]],
    date: str,
    multiplier: Decimal,
    neural_side: str,
    observation_id: int,
    checkpoint_sha256: str,
    calendar: tuple[str, ...],
    market: XyszMarketInputs | None = None,
) -> list[dict[str, Any]]:
    active = {item["code"] for item in record.targets}
    equity = raw_equity(ledger, market, date, active)[0] if market is not None else _equity(ledger, prices, date, "close", active)
    all_targets = list(record.targets)
    for code in sorted(_held_codes(ledger) - active):
        all_targets.append({"code": code, "weight": 0.0, "score": None})
    plans = []
    for target in all_targets:
        code = target["code"]
        close = raw_mark(market, code, date)[1] if market is not None else _asof_bar(prices, code, date)[1]["close"]
        target_exposure = D(str(target["weight"])) * RISK_BUDGET * multiplier
        delta = equity * target_exposure - ledger.holdings(code) * close
        side = "BUY" if delta > 0 else "SELL"
        quantity = _lot_quantity(abs(delta), close)
        plan = {"code": code, "weight": target["weight"], "target_exposure": str(target_exposure), "side": side, "quantity": str(quantity)}
        if quantity:
            intent = ledger.prepare_intent(signal_date=date, execution_date=_next_calendar(calendar, date) or "END", symbol=code, neural_side=neural_side if neural_side in {"BUY", "SELL", "HOLD"} else ("BUY" if side == "BUY" else "SELL"), order_side=side, intent_kind="EXPOSE" if side == "BUY" else "DEEXPOSE", target_exposure=target_exposure, requested_qty=quantity, observation_id=observation_id, checkpoint_sha256=checkpoint_sha256)
            plan["intent_id"] = intent
        plans.append(plan)
    return plans



def assess_action_coverage(events: list[dict[str, Any]], *, expected_records: int) -> dict[str, Any]:
    eligible = [event for event in events if event.get("signal_eligible")]
    side_counts = {side: sum(event.get("neural", {}).get("side") == side for event in eligible) for side in ("BUY", "SELL", "HOLD")}
    action_counts = {action: sum(event.get("action") == action for event in eligible) for action in ("EXPOSE", "DEEXPOSE", "DEFER")}
    defer_ratio = action_counts["DEFER"] / max(1, len(eligible))
    quarantined = expected_records == 46 and (any(count == 0 for count in side_counts.values()) or defer_ratio >= 0.90)
    return {"eligible_count": len(eligible), "side_counts": side_counts, "action_counts": action_counts, "defer_ratio": defer_ratio, "quarantined_action_collapse": quarantined}

def run_v4_fixture(
    stream_path: Path,
    output: Path,
    *,
    max_days: int = 0,
    expected_records: int = 46,
    training_end: str = TRAIN_END,
    controller_factory: Callable[[Settings], Any] = FlyController,
    market_manifest: Path | None = None,
) -> dict[str, Any]:
    stream = validate_v4_stream(stream_path, expected_records=expected_records, training_end=training_end, market_manifest=market_manifest)
    output = Path(output).resolve()
    if output.exists() and not (output / "ledger.sqlite").exists():
        # A launcher may have created a log before the adapter initialized. It
        # is safe to reuse only that explicitly named empty-run marker; every
        # other pre-existing file remains fail-closed.
        allowed_launcher_files = {"training.log"}
        existing = {item.name for item in output.iterdir()}
        if not existing <= allowed_launcher_files:
            raise AdapterBlocked("v4 output must be new or resumable")
    output.mkdir(parents=True, exist_ok=True)
    settings = Settings(learning=True)
    fly = controller_factory(settings)
    graph_hash = hashlib.sha256(json.dumps(fly.brain.configuration_signature(), sort_keys=True).encode()).hexdigest()
    costs = CostConfig()
    adapter_sha256 = _adapter_sha256()
    market_signature = None if stream.market is None else {"parent_sha256": stream.market.parent_sha256, "qa_sha256": stream.market.qa_sha256, "artifact_hashes": stream.market.artifact_hashes}
    signature = hashlib.sha256(json.dumps({"schema": V4_SCHEMA, "stream": stream.base.stream_sha256, "spec": stream.spec_sha256, "graph": graph_hash, "adapter_sha256": adapter_sha256, "lookback": LOOKBACK, "lookback_closes": LOOKBACK_CLOSES, "training_end": training_end, "risk_budget": str(RISK_BUDGET), "costs": costs.as_json(), "market": market_signature}, sort_keys=True).encode()).hexdigest()
    ledger = WC006Ledger(output / "ledger.sqlite", initial_state={"cash": "100000", "lots": []}, signature=signature, authoritative_trading_state=stream.market is not None)
    base = WC006Ledger(output / "paired_base.sqlite", initial_state={"cash": "100000", "lots": []}, signature=signature + ":paired-base", authoritative_trading_state=stream.market is not None)
    last_raw = ledger.get_meta("last_completed_index"); base_last_raw = base.get_meta("last_completed_index")
    last_day = -1 if last_raw is None else int(last_raw); base_last = -1 if base_last_raw is None else int(base_last_raw)
    if last_day != base_last:
        ledger.close(); base.close(); raise AdapterBlocked("v4 primary/paired daily indices diverge")
    checkpoint = ledger.get_meta("checkpoint")
    if checkpoint:
        cp = output / checkpoint["file"]
        if not cp.is_file() or sha256(cp) != checkpoint["sha256"]:
            ledger.close(); base.close(); raise AdapterBlocked("v4 checkpoint missing or hash mismatch")
        fly.restore(cp)
    else:
        clean = output / "brain-clean.npz"; fly.save(clean)
        ledger.put_meta("initial_brain_sha256", sha256(clean)); ledger.put_meta("initial_brain_source", "new FlyController; no historical checkpoint"); ledger.db.commit()
        base.put_meta("initial_brain_sha256", None); base.db.commit()
    provenance = output / "provenance.json"
    if not provenance.exists():
        provenance.write_text(json.dumps({"schema": V4_SCHEMA, "stream_sha256": stream.base.stream_sha256, "spec_sha256": stream.spec_sha256, "graph_manifest_sha256": graph_hash, "adapter_sha256": adapter_sha256, "initial_brain": "clean/new; no historical checkpoint", "lookback_sessions": LOOKBACK, "risk_budget": str(RISK_BUDGET), "paper_only": True, "broker_orders_enabled": False, "live_execution_authorized": False, "authoritative_trading_state": stream.market is not None, "market_inputs": market_signature, "price_layers": {"observation": "pinned qfq cache", "execution_mark_reward": "xysz raw Kline"}, "training_window": [TRAIN_START, TRAIN_END], "sealed_metrics": True}, indent=2) + "\n")
    records_by_date = {item.signal_date: (index, item) for index, item in enumerate(stream.base.records)}
    active = None
    started = 0
    events_path = output / "events.jsonl"
    first = stream.base.records[0].signal_date; final = stream.base.records[-1].execution_date
    dates = [item for item in stream.base.calendar if first <= item <= training_end]
    settlement_date = final if final > training_end else None
    try:
        for day_index, date in enumerate(dates):
            if day_index <= last_day:
                if date in records_by_date: active = records_by_date[date][1]
                continue
            execution_prices = stream.market.raw_prices if stream.market is not None else stream.prices
            if stream.market is not None:
                # Lifecycle events happen before the open; dividend ownership
                # is locked only after today's open fills settle below.
                apply_corporate_actions(ledger, stream.market, date, stream.base.calendar, stage="start_of_day")
                apply_corporate_actions(base, stream.market, date, stream.base.calendar, stage="start_of_day")
            settled = _settle_v4(ledger, date, stream.base.calendar, execution_prices, costs, stream.market)
            base_settled = _settle_v4(base, date, stream.base.calendar, execution_prices, costs, stream.market)
            if stream.market is not None:
                # Record-date eligibility is based on holdings after open settlement.
                apply_corporate_actions(ledger, stream.market, date, stream.base.calendar, stage="after_open_settlement")
                apply_corporate_actions(base, stream.market, date, stream.base.calendar, stage="after_open_settlement")
            if date in records_by_date: active = records_by_date[date][1]
            if active is None: raise AdapterBlocked("v4 observed before first signal")
            active_codes = {item["code"] for item in active.targets}
            previous_equity = D(ledger.get_meta("previous_equity") or "100000")
            previous_base = D(base.get_meta("previous_equity") or "100000")
            if stream.market is not None:
                equity, raw_marks = raw_equity(ledger, stream.market, date, active_codes)
                base_equity, base_raw_marks = raw_equity(base, stream.market, date, active_codes)
            else:
                equity = _equity(ledger, stream.prices, date, "close", active_codes)
                base_equity = _equity(base, stream.prices, date, "close", active_codes)
                raw_marks = base_raw_marks = {}
            peak = max(previous_equity, D(ledger.get_meta("peak_equity") or previous_equity), equity)
            previous_dd = D(ledger.get_meta("drawdown") or "0")
            drawdown = equity / peak - D("1") if peak else D("0")
            feedback = incremental_feedback(equity=equity, previous_equity=previous_equity, base_equity=base_equity, previous_base_equity=previous_base, drawdown=drawdown, previous_drawdown=previous_dd, turnover_ratio=settled / previous_equity if previous_equity else D("0"))
            multiplier = D(ledger.get_meta("multiplier") or "1.0")
            fly_marks = raw_marks if stream.market is not None else _mark_provenance(ledger, stream.prices, date, active_codes)
            base_marks = base_raw_marks if stream.market is not None else _mark_provenance(base, stream.prices, date, active_codes)
            frame = _frame(active, stream.prices, date, stream.base.calendar, ledger, multiplier, drawdown)
            neural = fly.observe(frame, str(feedback["stimulus"]))
            action, next_multiplier = map_decoder_to_multiplier(str(neural["side"]), multiplier)
            neural["wc006_action"] = action; neural["wc006_multiplier_after"] = str(next_multiplier)
            neural_payload = deterministic_neural_payload(neural)
            cp = output / f"brain-day-{day_index % 2}.npz"; fly.save(cp); cp_info = {"file": cp.name, "sha256": sha256(cp)}
            observation_id = ledger.record_observation(date=date, input_sha256=str(neural["input_sha256"]), reinforcement=str(feedback["stimulus"]), neural=neural_payload, checkpoint=cp_info)
            eligible = date in records_by_date
            if eligible:
                signal_index, signal_record = records_by_date[date]
                multiplier = next_multiplier if action != "DEFER" else multiplier
                plan_prices = stream.market.raw_prices if stream.market is not None else stream.prices
                fly_plan = _prepare(ledger, signal_record, plan_prices, date, multiplier, str(neural["side"]), observation_id, cp_info["sha256"], stream.base.calendar, stream.market) if action != "DEFER" else []
                base_plan = _prepare(base, signal_record, plan_prices, date, D("1.0"), "BUY", 0, "paired-base", stream.base.calendar, stream.market)
            else:
                signal_index, fly_plan, base_plan = None, [], []
            paired_plan = {"candidates": [{"code": x["code"], "weight": x["weight"]} for x in active.targets], "risk_budget": str(RISK_BUDGET), "fly_multiplier": str(multiplier), "fly": fly_plan, "base": base_plan, "execution_date": active.execution_date if eligible else None}
            event = {"date": date, "signal_eligible": eligible, "signal_date": date if eligible else None, "execution_date": active.execution_date if eligible else None, "schedule_status": "eligible" if eligible else "schedule_blocked", "neural": neural_payload, "action": action, "reinforcement": str(feedback["stimulus"]), "feedback": {key: str(value) if isinstance(value, Decimal) else value for key, value in feedback.items()}, "turnover": str(settled), "mark_provenance": {"fly": fly_marks, "base": base_marks}, "paired_plan": paired_plan, "checkpoint": cp_info, "drawdown": str(drawdown)}
            event_hash, created = ledger.record_event(date=date, payload=event); event["event_hash"] = event_hash
            if created:
                with events_path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            base.record_event(date=date, payload={"date": date, "signal_eligible": eligible, "mark_provenance": {"fly": fly_marks, "base": base_marks}, "paired_plan": paired_plan})
            ledger.complete_date(date=date, index=day_index, multiplier=multiplier, equity=equity, drawdown=drawdown, checkpoint=cp_info)
            base.complete_date(date=date, index=day_index, multiplier=D("1.0"), equity=base_equity, drawdown=D("0"), checkpoint={"paired": hashlib.sha256(json.dumps(base_plan, sort_keys=True).encode()).hexdigest()})
            ledger.db.commit(); base.db.commit(); started += 1
            if max_days and started >= max_days: break
        current_raw = ledger.get_meta("last_completed_index")
        training_complete = current_raw is not None and int(current_raw) == len(dates) - 1
        if training_complete and settlement_date:
            # Boundary settlement only: no observation, reward, or learning.
            settlement_prices = stream.market.raw_prices if stream.market is not None else stream.prices
            _settle_v4(ledger, settlement_date, stream.base.calendar, settlement_prices, costs, stream.market)
            _settle_v4(base, settlement_date, stream.base.calendar, settlement_prices, costs, stream.market)
            ledger.put_meta("final_settlement_date", settlement_date); base.put_meta("final_settlement_date", settlement_date)
            ledger.db.commit(); base.db.commit()
        complete = training_complete and not ledger.pending()
        coverage = None
        if complete:
            signal_events = [json.loads(line) for line in events_path.read_text().splitlines() if json.loads(line).get("signal_eligible")]
            coverage = assess_action_coverage(signal_events, expected_records=expected_records)
            ledger.put_meta("action_coverage", coverage); ledger.db.commit()
        quarantined = bool(coverage and coverage["quarantined_action_collapse"])
        return {"status": "quarantined_action_collapse" if quarantined else ("complete" if complete else "paused"), "days": started, "calendar_days": len(dates), "signal_records": len(stream.base.records), "action_coverage": coverage, "signature": signature, "graph_manifest_sha256": graph_hash}
    finally:
        ledger.close(); base.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument("--market-manifest", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run_v4_fixture(args.manifest, args.out, max_days=args.max_days, market_manifest=args.market_manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
