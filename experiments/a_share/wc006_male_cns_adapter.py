"""Isolated WC006 candidate-stream to complete MaleCNS paper adapter.

This module is deliberately a dry-run/training harness.  It consumes a
hash-pinned candidate stream, renders one composite observation per signal
date, and uses the retained :class:`FlyController` graph.  It never talks to
a broker and it refuses an old brain checkpoint for the primary arm.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
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
from stonkfly.config import Settings
from stonkfly.neural.controller import FlyController


TRAIN_START = "2021-07-01"
TRAIN_END = "2023-12-29"
ADAPTER_SCHEMA = "wc006-male-cns-paper/v1"


class AdapterBlocked(ValueError):
    """Raised when a candidate stream cannot prove its causal contract."""


@dataclass(frozen=True)
class StreamRecord:
    signal_date: str
    execution_date: str
    targets: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ValidatedStream:
    path: Path
    payload: dict[str, Any]
    records: tuple[StreamRecord, ...]
    calendar: tuple[str, ...]
    cache_paths: dict[str, Path]
    stream_sha256: str


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterBlocked(f"invalid JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise AdapterBlocked(f"JSON object required: {path}")
    return value


def _date(value: object) -> str:
    text = str(value).strip()
    if len(text) != 10 or text[4] != "-" or text[7] != "-":
        raise AdapterBlocked(f"non-ISO date: {value!r}")
    return text


def _code(value: object) -> str:
    text = str(value).strip().upper()
    if text.endswith((".SH", ".SZ", ".BJ")):
        text = text[:-3]
    if len(text) != 6 or not text.isdigit():
        raise AdapterBlocked(f"invalid candidate code: {value!r}")
    return text


def _finite_positive(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AdapterBlocked(f"invalid {label}") from exc
    if not math.isfinite(number) or number <= 0:
        raise AdapterBlocked(f"{label} must be finite and positive")
    return number


def _next(calendar: tuple[str, ...], date: str) -> str:
    later = [item for item in calendar if item > date]
    if not later:
        raise AdapterBlocked(f"no next session after {date}")
    return min(later)


def _pinned_file(root: Path, relative: str, expected: str, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise AdapterBlocked(f"{label} escapes manifest root") from exc
    if not path.is_file() or sha256(path) != expected:
        raise AdapterBlocked(f"{label} hash mismatch or missing")
    return path


def validate_candidate_stream(
    path: Path,
    *,
    expected_records: int = 46,
    require_selector_generation_provenance: bool = False,
) -> ValidatedStream:
    """Validate all provenance before any neural state is created.

    Legacy streams may omit selector-generation provenance by default.  When
    present, however, the field is always validated so a partially copied or
    tampered selector run cannot pass downstream.  New purged research callers
    can set ``require_selector_generation_provenance`` to require both pinned
    selector-generation files.
    """
    path = Path(path).resolve()
    payload = _json(path)
    root = path.parent
    if payload.get("schema") != "wc006-candidate-stream/v1":
        raise AdapterBlocked("unsupported candidate manifest schema")
    if payload.get("paper_only") is not True or payload.get("broker_orders_enabled") is not False or payload.get("live_execution_authorized") is not False:
        raise AdapterBlocked("candidate manifest is not paper-only")
    if payload.get("authoritative_trading_state") is not False:
        raise AdapterBlocked("primary fixture requires non-authoritative input")
    if payload.get("signal_lag_one_day") is not True or not payload.get("exact_tag"):
        raise AdapterBlocked("signal lag/exact tag provenance missing")
    if not isinstance(payload.get("limitations"), list) or not payload["limitations"]:
        raise AdapterBlocked("limitations/provenance missing")
    source_hashes = payload.get("source_hashes")
    if not isinstance(source_hashes, dict) or not {"active_config", "universe", "calendar", "adapter"} <= set(source_hashes):
        raise AdapterBlocked("source hash set incomplete")
    selector_provenance = payload.get("selector_generation_provenance")
    if selector_provenance is None:
        if require_selector_generation_provenance:
            raise AdapterBlocked("selector generation provenance is required")
    else:
        required_provenance = {"checkpoint", "mirror_provenance"}
        allowed_provenance = required_provenance | {"extension_provenance"}
        if not isinstance(selector_provenance, dict) or not required_provenance <= set(selector_provenance) or not set(selector_provenance) <= allowed_provenance:
            raise AdapterBlocked("selector generation provenance field is incomplete")
        for name in ("checkpoint", "mirror_provenance", "extension_provenance"):
            if name not in selector_provenance:
                continue
            item = selector_provenance[name]
            if not isinstance(item, dict) or set(item) != {"source_name", "sha256", "copied_path"}:
                raise AdapterBlocked(f"selector generation {name} provenance is incomplete")
            source_name = item["source_name"]
            copied_path = item["copied_path"]
            expected_hash = item["sha256"]
            if not isinstance(source_name, str) or not source_name:
                raise AdapterBlocked(f"selector generation {name} source name is invalid")
            if not isinstance(copied_path, str) or not copied_path:
                raise AdapterBlocked(f"selector generation {name} copied path is invalid")
            if not isinstance(expected_hash, str) or not expected_hash:
                raise AdapterBlocked(f"selector generation {name} hash is invalid")
            _pinned_file(root, copied_path, expected_hash, f"selector generation {name}")
    _pinned_file(root, "inputs/active_config.json", str(source_hashes["active_config"]), "active_config")
    universe_path = _pinned_file(root, "inputs/universe.txt", str(source_hashes["universe"]), "universe")
    universe_codes = {_code(line) for line in universe_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    # Historical packages pin two byte-exact builder releases. Retain those
    # source copies so their hashes remain verifiable after an additive update.
    adapter_sources = (
        Path(__file__).with_name("build_wc006_candidate_stream.py"),
        Path(__file__).with_name("build_wc006_candidate_stream_v1.py"),
        Path(__file__).with_name("build_wc006_candidate_stream_v0.py"),
    )
    if not any(source.is_file() and sha256(source) == str(source_hashes["adapter"]) for source in adapter_sources):
        raise AdapterBlocked("candidate builder source hash is not pinned")
    calendar_path = _pinned_file(root, "inputs/calendar.json", str(source_hashes["calendar"]), "calendar")
    calendar_payload = _json(calendar_path)
    calendar = tuple(sorted({_date(item) for item in calendar_payload.get("dates", [])}))
    if not calendar:
        raise AdapterBlocked("calendar is empty")
    records_raw = payload.get("records")
    if not isinstance(records_raw, list) or len(records_raw) != expected_records:
        raise AdapterBlocked(f"expected {expected_records} candidate records")
    artifacts = payload.get("selector_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(records_raw):
        raise AdapterBlocked("selector artifact provenance is incomplete")
    cache_files = payload.get("cache_files")
    if not isinstance(cache_files, dict) or not cache_files:
        raise AdapterBlocked("cache provenance is incomplete")
    cache_paths: dict[str, Path] = {}
    for code, item in cache_files.items():
        if not isinstance(item, dict):
            raise AdapterBlocked("invalid cache provenance")
        canonical = _code(code)
        cache_paths[canonical] = _pinned_file(root, str(item.get("copied_path", "")), str(item.get("sha256", "")), f"cache {canonical}")
    cache_dates: dict[str, set[str]] = {}
    for code, cache_path in cache_paths.items():
        with cache_path.open(newline="", encoding="utf-8") as handle:
            cache_dates[code] = {_date(row.get("date")) for row in csv.DictReader(handle)}
    records: list[StreamRecord] = []
    previous = None
    exact_tag = str(payload["exact_tag"])
    for index, raw in enumerate(records_raw):
        if not isinstance(raw, dict):
            raise AdapterBlocked("candidate record is not an object")
        signal = _date(raw.get("signal_date"))
        execution = _date(raw.get("execution_date"))
        if previous is not None and signal <= previous:
            raise AdapterBlocked("candidate dates are not strictly increasing")
        previous = signal
        if execution != _next(calendar, signal):
            raise AdapterBlocked(f"execution is not first next session at {signal}")
        selector = raw.get("selector")
        if not isinstance(selector, dict) or selector.get("rebalance_applied") is not True or selector.get("prefer_tag") != exact_tag:
            raise AdapterBlocked(f"selector provenance mismatch at {signal}")
        target_rows = selector.get("target_topk")
        if not isinstance(target_rows, list) or len(target_rows) != 3:
            raise AdapterBlocked(f"dynamic three-candidate contract failed at {signal}")
        seen: set[str] = set()
        total = 0.0
        targets = []
        for row in target_rows:
            if not isinstance(row, dict):
                raise AdapterBlocked("target row is not an object")
            code = _code(row.get("code"))
            if code in seen or code not in cache_paths or code not in universe_codes:
                raise AdapterBlocked(f"candidate universe/cache mismatch at {signal}: {code}")
            weight = _finite_positive(row.get("target_weight"), f"weight {code}")
            seen.add(code)
            total += weight
            # Audit metadata only: this does not affect rendering, neural
            # propagation, decoding, or execution.  Do not synthesize rank.
            target = {"code": code, "weight": weight, "score": row.get("score")}
            if row.get("rank") is not None:
                target["rank"] = row.get("rank")
            targets.append(target)
        if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-9):
            raise AdapterBlocked(f"target weights do not sum to one at {signal}")
        cache_hashes = selector.get("cache_sha256")
        if not isinstance(cache_hashes, dict):
            raise AdapterBlocked("record cache hashes missing")
        for item in targets:
            if cache_hashes.get(item["code"]) != sha256(cache_paths[item["code"]]):
                raise AdapterBlocked(f"record cache hash mismatch at {signal}")
        artifact = artifacts[index]
        if not isinstance(artifact, dict):
            raise AdapterBlocked("selector artifact is not an object")
        artifact_path = _pinned_file(root, str(artifact.get("copied_path", "")), str(artifact.get("sha256", "")), "selector artifact")
        if selector.get("artifact_sha256") != sha256(artifact_path):
            raise AdapterBlocked(f"record artifact hash mismatch at {signal}")
        artifact_payload = _json(artifact_path)
        if _date(artifact_payload.get("rebalance_date", artifact_payload.get("date"))) != signal or artifact_payload.get("prefer_tag") != exact_tag or artifact_payload.get("rebalance_applied") is not True:
            raise AdapterBlocked(f"selector artifact semantics mismatch at {signal}")
        artifact_targets = artifact_payload.get("effective_target_topk")
        if not isinstance(artifact_targets, list) or [( _code(row.get("code")), float(row.get("target_weight")) ) for row in artifact_targets] != [(item["code"], item["weight"]) for item in targets]:
            raise AdapterBlocked(f"selector artifact targets mismatch at {signal}")
        for item in targets:
            if signal not in cache_dates[item["code"]] or execution not in cache_dates[item["code"]]:
                raise AdapterBlocked(f"cache lacks signal/execution bar at {signal}: {item['code']}")
        records.append(StreamRecord(signal, execution, tuple(targets)))
    return ValidatedStream(path, payload, tuple(records), calendar, cache_paths, sha256(path))


def _load_prices(stream: ValidatedStream) -> dict[str, dict[str, dict[str, Decimal]]]:
    result: dict[str, dict[str, dict[str, Decimal]]] = {}
    for code, path in stream.cache_paths.items():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = csv.DictReader(handle)
            by_date = {}
            for raw in rows:
                date = _date(raw.get("date"))
                values = {}
                for field in ("open", "close"):
                    try:
                        value = D(raw[field])
                    except (KeyError, ValueError) as exc:
                        raise AdapterBlocked(f"missing {field} for {code} {date}") from exc
                    if value <= 0:
                        raise AdapterBlocked(f"nonpositive {field} for {code} {date}")
                    values[field] = value
                by_date[date] = values
            result[code] = by_date
    return result


def _frame(record: StreamRecord, prices: dict[str, dict[str, dict[str, Decimal]]], multiplier: Decimal) -> np.ndarray:
    """Render a deterministic multi-symbol composite, not a policy."""
    image = Image.new("RGB", (320, 180), (235, 240, 249))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 319, 20), fill=(19, 36, 71))
    draw.text((5, 5), f"wc006 {multiplier:.1f}x {record.signal_date}", fill=(219, 229, 249))
    for index, target in enumerate(record.targets):
        code = target["code"]
        close = prices[code][record.signal_date]["close"]
        draw.text((5, 28 + index * 44), f"{code} W {target['weight']:.6f} C {close:.4f}", fill=(28, 46, 82))
        level = max(0, min(255, int(float(close % D("100")) * 2.55)))
        draw.rectangle((5, 45 + index * 44, 315, 55 + index * 44), fill=(level, 100, 180))
    return np.asarray(image, dtype=np.uint8)


def _stream_signature(stream: ValidatedStream, costs: CostConfig, graph_hash: str) -> str:
    payload = {"adapter": ADAPTER_SCHEMA, "stream": stream.stream_sha256, "graph": graph_hash, "costs": costs.as_json(), "initial_brain": None, "training_window": [TRAIN_START, TRAIN_END], "timing": "close T intent; next valid session open fill"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _held_codes(ledger: WC006Ledger) -> set[str]:
    return {str(row[0]) for row in ledger.db.execute("SELECT DISTINCT symbol FROM lots WHERE CAST(quantity AS REAL) > 0")}


def _equity(ledger: WC006Ledger, prices: dict[str, dict[str, dict[str, Decimal]]], date: str, field: str, active_codes: set[str] | None = None) -> Decimal:
    cash = D(ledger.get_meta("cash"))
    codes = set(active_codes or ()) | _held_codes(ledger)
    total = cash
    for code in codes:
        if code not in prices or date not in prices[code]:
            raise AdapterBlocked(f"missing price for held/candidate {code} on {date}")
        total += ledger.holdings(code) * prices[code][date][field]
    return total


def _next_calendar(calendar: tuple[str, ...], date: str) -> str | None:
    later = [item for item in calendar if item > date]
    return min(later) if later else None


def _settle_pending(
    ledger: WC006Ledger,
    date: str,
    calendar: tuple[str, ...],
    prices: dict[str, dict[str, dict[str, Decimal]]],
    costs: CostConfig,
    *,
    max_total_exposure: Decimal,
) -> None:
    """Resolve every T-1 intent at the current session's open, with guards."""
    pending = ledger.pending(date)
    active_codes = {str(intent["symbol"]) for intent in pending}
    open_equity = _equity(ledger, prices, date, "open", active_codes)
    turnover = D("0")
    for intent in pending:
        code = str(intent["symbol"])
        quantity = D(intent["requested_qty"])
        side = str(intent["order_side"])
        price = prices[code][date]["open"] * (D("1") + costs.slippage if side == "BUY" else D("1") - costs.slippage)
        fee = commission(quantity * price, costs.commission_rate, costs.minimum_commission)
        if side == "SELL":
            fee += quantity * price * costs.stamp_duty_rate
        reason = None
        if side == "SELL" and ledger.holdings(code, sellable_on=date) < quantity:
            reason = "t_plus_1_or_inventory"
        elif side == "BUY":
            if D(ledger.get_meta("cash")) < quantity * price + fee:
                reason = "cash_and_fee"
            else:
                gross = sum((ledger.holdings(item) * prices[item][date]["open"] for item in (_held_codes(ledger) | {code})), D("0"))
                if open_equity <= 0 or (gross + quantity * price) / open_equity > max_total_exposure:
                    reason = "total_exposure"
        if reason is not None:
            ledger.mark_veto(str(intent["id"]), reason)
            continue
        ledger.settle(
            str(intent["id"]),
            execution_date=date,
            quantity=quantity,
            price=price,
            fee=fee,
            sellable_from=_next_calendar(calendar, date),
        )
        turnover += quantity * price
    return turnover


def _prepare_target_intents(
    ledger: WC006Ledger,
    record: StreamRecord,
    prices: dict[str, dict[str, dict[str, Decimal]]],
    multiplier: Decimal,
    *,
    neural_side: str,
    observation_id: int,
    checkpoint_sha256: str,
    costs: CostConfig,
    max_total_exposure: Decimal,
) -> list[dict[str, Any]]:
    """Build the same candidate/weight/cost plan for Fly or paired base."""
    equity = _equity(ledger, prices, record.signal_date, "close", {target["code"] for target in record.targets})
    plans: list[dict[str, Any]] = []
    target_by_code = {target["code"]: target for target in record.targets}
    all_targets = list(record.targets)
    # A prior holding absent from the changing candidate set is an explicit
    # target=0 liquidation, never a silently retained ghost position.
    for code in sorted(prices):
        if code not in target_by_code and ledger.holdings(code) > 0:
            all_targets.append({"code": code, "weight": 0.0, "score": None})
    for target in all_targets:
        code = target["code"]
        close = prices[code][record.signal_date]["close"]
        current = ledger.holdings(code)
        target_exposure = D(str(target["weight"])) * multiplier
        delta = equity * target_exposure - current * close
        side = "BUY" if delta > 0 else "SELL"
        quantity = _lot_quantity(abs(delta), close)
        plan = {"code": code, "target_weight": target["weight"], "target_exposure": str(target_exposure), "side": side, "quantity": str(quantity), "price_reference": str(close), "execution_date": record.execution_date}
        if quantity:
            intent = ledger.prepare_intent(
                signal_date=record.signal_date,
                execution_date=record.execution_date,
                symbol=code,
                neural_side=neural_side if neural_side in {"BUY", "SELL", "HOLD"} else ("BUY" if side == "BUY" else "SELL"),
                order_side=side,
                intent_kind="EXPOSE" if side == "BUY" else "DEEXPOSE",
                target_exposure=target_exposure,
                requested_qty=quantity,
                observation_id=observation_id,
                checkpoint_sha256=checkpoint_sha256,
            )
            # The guard rejects an order; it never substitutes another symbol.
            if target_exposure > max_total_exposure:
                ledger.mark_veto(intent, "exposure_cap")
                plan["pre_veto"] = "exposure_cap"
        plans.append(plan)
    return plans


def run_dry_fixture(
    stream_path: Path,
    output: Path,
    *,
    max_steps: int = 0,
    expected_records: int = 46,
    controller_factory: Callable[[Settings], Any] = FlyController,
) -> dict[str, Any]:
    """Run a small paper-only slice with the complete graph and clean brain.

    ``max_steps`` is intentionally used by tests to interrupt/resume.  The
    function never accepts an initial checkpoint: primary-arm state starts
    from a newly-created controller in this output directory.
    """
    stream = validate_candidate_stream(stream_path, expected_records=expected_records)
    output = Path(output).resolve()
    if output.exists() and not (output / "ledger.sqlite").exists():
        raise AdapterBlocked("output must be a new isolated run")
    output.mkdir(parents=True, exist_ok=True)
    prices = _load_prices(stream)
    settings = Settings(learning=True)
    fly = controller_factory(settings)
    graph_hash = hashlib.sha256(json.dumps(fly.brain.configuration_signature(), sort_keys=True).encode()).hexdigest()
    costs = CostConfig()
    max_total_exposure = D("0.85")
    signature = _stream_signature(stream, costs, graph_hash)
    ledger = WC006Ledger(output / "ledger.sqlite", initial_state={"cash": "100000", "lots": []}, signature=signature, authoritative_trading_state=False)
    base_signature = signature + ":paired-base"
    base = WC006Ledger(output / "paired_base.sqlite", initial_state={"cash": "100000", "lots": []}, signature=base_signature, authoritative_trading_state=False)
    checkpoint_state = ledger.get_meta("checkpoint")
    base_last_raw = base.get_meta("last_completed_index")
    last_raw = ledger.get_meta("last_completed_index")
    base_last = int(-1 if base_last_raw is None else base_last_raw)
    last = int(-1 if last_raw is None else last_raw)
    if base_last != last:
        ledger.close()
        base.close()
        raise AdapterBlocked("primary/paired ledgers are not at the same resumable index")
    if checkpoint_state:
        checkpoint = output / checkpoint_state["file"]
        if not checkpoint.exists() or sha256(checkpoint) != checkpoint_state["sha256"]:
            ledger.close()
            base.close()
            raise AdapterBlocked("resume checkpoint missing or hash mismatch")
        fly.restore(checkpoint)
    else:
        # This is a clean brain, never a v2/v3 historical checkpoint.
        clean = output / "brain-clean.npz"
        fly.save(clean)
        clean_hash = sha256(clean)
        ledger.put_meta("initial_brain_sha256", clean_hash)
        ledger.put_meta("initial_brain_source", "new FlyController; no historical checkpoint")
        ledger.db.commit()
        base.put_meta("initial_brain_sha256", None)
        base.db.commit()
    provenance = output / "provenance.json"
    if not provenance.exists():
        provenance.write_text(json.dumps({"schema": ADAPTER_SCHEMA, "stream_sha256": stream.stream_sha256, "graph_manifest_sha256": graph_hash, "initial_brain": "clean/new; no historical checkpoint", "paper_only": True, "broker_orders_enabled": False, "live_execution_authorized": False, "authoritative_trading_state": False, "paired_base_ledger": "paired_base.sqlite", "training_window": [TRAIN_START, TRAIN_END], "sealed_metrics": True, "limitations": ["paper-only training; no performance conclusions", "official ST/suspension/limits unavailable; strict executable replay is not authorized", "no returns/PnL are computed or reported"]}, indent=2) + "\n")
    records_by_date = {item.signal_date: (index, item) for index, item in enumerate(stream.records)}
    stop_after = max_steps if max_steps > 0 else None
    started = 0
    events_path = output / "events.jsonl"
    try:
        final_date = max(item.execution_date for item in stream.records)
        for date in stream.calendar:
            if date < stream.records[0].signal_date or date > final_date:
                continue
            # This is intentionally before the close decision on every date.
            settled_turnover = _settle_pending(ledger, date, stream.calendar, prices, costs, max_total_exposure=max_total_exposure)
            settled_base_turnover = _settle_pending(base, date, stream.calendar, prices, costs, max_total_exposure=max_total_exposure)
            turnover_since = D(ledger.get_meta("turnover_since_observation") or "0") + settled_turnover
            base_turnover_since = D(base.get_meta("turnover_since_observation") or "0") + settled_base_turnover
            ledger.put_meta("turnover_since_observation", str(turnover_since))
            base.put_meta("turnover_since_observation", str(base_turnover_since))
            ledger.db.commit()
            base.db.commit()
            item = records_by_date.get(date)
            if item is None:
                continue
            index, record = item
            if index <= last:
                continue
            previous_equity = D(ledger.get_meta("previous_equity") or "100000")
            previous_base_equity = D(base.get_meta("previous_equity") or "100000")
            active_codes = {target["code"] for target in record.targets}
            equity = _equity(ledger, prices, date, "close", active_codes)
            base_equity = _equity(base, prices, date, "close", active_codes)
            peak_equity = max(previous_equity, D(ledger.get_meta("peak_equity") or previous_equity), equity)
            previous_drawdown = D(ledger.get_meta("drawdown") or "0")
            drawdown = equity / peak_equity - D("1") if peak_equity else D("0")
            turnover = D(ledger.get_meta("turnover_since_observation") or "0")
            base_turnover = D(base.get_meta("turnover_since_observation") or "0")
            feedback = incremental_feedback(equity=equity, previous_equity=previous_equity, base_equity=base_equity, previous_base_equity=previous_base_equity, drawdown=drawdown, previous_drawdown=previous_drawdown, turnover_ratio=turnover / previous_equity if previous_equity else D("0"))
            reinforcement = str(feedback["stimulus"])
            multiplier = D(ledger.get_meta("multiplier") or "1.0")
            frame = _frame(record, prices, multiplier)
            neural = fly.observe(frame, reinforcement)
            action, new_multiplier = map_decoder_to_multiplier(str(neural["side"]), multiplier)
            neural["wc006_action"] = action
            neural["wc006_multiplier_after"] = str(new_multiplier)
            checkpoint = output / f"brain-{index % 2}.npz"
            fly.save(checkpoint)
            checkpoint_info = {"file": checkpoint.name, "sha256": sha256(checkpoint)}
            observation_id = ledger.record_observation(date=record.signal_date, input_sha256=str(neural["input_sha256"]), reinforcement=reinforcement, neural=neural, checkpoint=checkpoint_info)
            fly_multiplier = new_multiplier if action != "DEFER" else multiplier
            if action != "DEFER":
                multiplier = fly_multiplier
                fly_plan = _prepare_target_intents(ledger, record, prices, fly_multiplier, neural_side=str(neural["side"]), observation_id=observation_id, checkpoint_sha256=checkpoint_info["sha256"], costs=costs, max_total_exposure=max_total_exposure)
            else:
                fly_plan = []
            base_plan = _prepare_target_intents(base, record, prices, D("1.0"), neural_side="BUY", observation_id=0, checkpoint_sha256="paired-base", costs=costs, max_total_exposure=max_total_exposure)
            paired_plan = {"candidates": [{"code": x["code"], "target_weight": x["weight"]} for x in record.targets], "execution_date": record.execution_date, "costs": costs.as_json(), "max_total_exposure": str(max_total_exposure), "fly": fly_plan, "base": base_plan}
            event = {"signal_date": record.signal_date, "execution_date": record.execution_date, "targets": paired_plan["candidates"], "paired_plan": paired_plan, "fly_action": action, "neural": neural, "reinforcement": reinforcement, "feedback": {key: str(value) if isinstance(value, Decimal) else value for key, value in feedback.items()}, "equity": str(equity), "base_equity": str(base_equity), "turnover_since_observation": str(turnover), "base_turnover_since_observation": str(base_turnover), "drawdown": str(drawdown), "checkpoint": checkpoint_info, "market_guard": {"authoritative": False, "strict": False, "official_state": "unverified"}}
            event_hash, created = ledger.record_event(date=record.signal_date, payload=event)
            event["event_hash"] = event_hash
            if created:
                with events_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, allow_nan=False, sort_keys=True) + "\n")
            base.record_event(date=record.signal_date, payload={"signal_date": record.signal_date, "execution_date": record.execution_date, "paired_plan": paired_plan})
            ledger.complete_date(date=record.signal_date, index=index, multiplier=multiplier, equity=equity, drawdown=drawdown, checkpoint=checkpoint_info)
            ledger.put_meta("turnover_since_observation", "0")
            base.put_meta("turnover_since_observation", "0")
            base_peak = max(previous_base_equity, D(base.get_meta("peak_equity") or previous_base_equity), base_equity)
            base_drawdown = base_equity / base_peak - D("1") if base_peak else D("0")
            base.complete_date(date=record.signal_date, index=index, multiplier=D("1.0"), equity=base_equity, drawdown=base_drawdown, checkpoint={"paired_plan": hashlib.sha256(json.dumps(base_plan, sort_keys=True).encode()).hexdigest()})
            ledger.db.commit()
            base.db.commit()
            started += 1
            if stop_after is not None and started >= stop_after:
                break
        complete_raw = ledger.get_meta("last_completed_index")
        complete = int(-1 if complete_raw is None else complete_raw) == len(stream.records) - 1 and not ledger.pending()
        return {"status": "complete" if complete else "paused", "steps": started, "signature": signature, "records": len(stream.records), "graph_manifest_sha256": graph_hash}
    finally:
        ledger.close()
        base.close()

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run_dry_fixture(args.manifest, args.out, max_steps=args.max_steps), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
