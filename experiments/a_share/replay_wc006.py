#!/usr/bin/env python3
"""wc006 v1: one shared MaleCNS controller over a fixed multi-symbol target.

This module is deliberately a paper/replay implementation.  It does not alter
the retained graph or the original decoder.  A close-time neural decision is
persisted as an intent and can only be filled at the next valid session open.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from stonkfly.config import D, Settings
from stonkfly.neural.controller import FlyController


POLICY_VERSION = "wc006-v1"
LOT = Decimal("100")
VALID_STATUS = {"OPEN", "TRADE", "TRADING"}
SCHEMA_VERSION = "wc006-ledger-v2"
# Reserved date sentinel: later than every supported ISO trading date. A lot
# bought on the final supplied session is never sellable in-data.
AFTER_DATA_END = "9999-12-31"


class MarketStateError(ValueError):
    """The replay cannot prove that a row is a valid trading session."""


@dataclass(frozen=True)
class SymbolData:
    symbol: str
    rows: tuple[dict[str, object], ...]
    source: Path
    source_sha256: str


@dataclass(frozen=True)
class Manifest:
    path: Path
    symbols: tuple[str, ...]
    base_targets: dict[str, Decimal]
    data: dict[str, SymbolData]
    version: str
    decision_stream_sha256: str | None
    initial_cash: Decimal
    initial_positions: dict[str, Decimal]
    initial_lots: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class CostConfig:
    profile: str = "wc006-research-5bp"
    commission_rate: Decimal = Decimal("0.0003")
    minimum_commission: Decimal = Decimal("5")
    slippage: Decimal = Decimal("0.0005")
    stamp_duty_rate: Decimal = Decimal("0.001")

    def as_json(self) -> dict[str, str]:
        return {
            "profile": self.profile,
            "commission_rate": str(self.commission_rate),
            "minimum_commission": str(self.minimum_commission),
            "slippage": str(self.slippage),
            "stamp_duty_rate": str(self.stamp_duty_rate),
        }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decimal(value: object, label: str) -> Decimal:
    try:
        result = D(value)
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise ValueError(f"invalid Decimal {label}") from exc
    if not result.is_finite():
        raise ValueError(f"nonfinite Decimal {label}")
    return result


def _read_ohlc(path: Path, *, symbol: str, strict_market_state: bool) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        required = {"date", "open", "high", "low", "close"}
        missing = required - fields
        if missing:
            raise ValueError(f"{symbol}: missing OHLC fields: {sorted(missing)}")
        official_fields = {"official_status", "official_limit_up", "official_limit_down"}
        if strict_market_state and not official_fields <= fields:
            raise MarketStateError(
                f"{symbol}: strict mode requires official status/limit evidence"
            )
        previous = None
        for raw in reader:
            date = str(raw["date"])
            if not date or (previous is not None and date <= previous):
                raise ValueError(f"{symbol}: dates must be unique and increasing")
            previous = date
            row: dict[str, object] = {"date": date, "symbol": symbol}
            for name in ("open", "high", "low", "close"):
                value = _decimal(raw[name], f"{symbol}.{date}.{name}")
                if value <= 0:
                    raise ValueError(f"{symbol}: nonpositive {name} on {date}")
                row[name] = value
            row["volume"] = raw.get("volume", "")
            evidence = all(str(raw.get(name, "")).strip() for name in official_fields)
            if strict_market_state and not evidence:
                raise MarketStateError(f"{symbol}: incomplete official evidence on {date}")
            row["authoritative_trading_state"] = evidence
            row["market_status"] = str(raw.get("official_status", ""))
            row["official_limit_up"] = raw.get("official_limit_up", "")
            row["official_limit_down"] = raw.get("official_limit_down", "")
            rows.append(row)
    if not rows:
        raise ValueError(f"{symbol}: empty OHLC file")
    return tuple(rows)


def load_manifest(path: Path, *, strict_market_state: bool = False) -> Manifest:
    """Load and hash-lock the immutable wc006 input manifest."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != POLICY_VERSION:
        raise ValueError("manifest version must be wc006-v1")
    symbols = tuple(payload.get("symbols", ()))
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("manifest symbols must be nonempty and unique")
    targets_raw = payload.get("base_targets")
    if not isinstance(targets_raw, dict) or not set(targets_raw) <= set(symbols):
        raise ValueError("base_targets must be a subset of manifest symbols")
    targets = {symbol: _decimal(targets_raw[symbol], f"base_targets.{symbol}") for symbol in symbols if symbol in targets_raw}
    if any(value < 0 or value > 1 for value in targets.values()):
        raise ValueError("base targets must be in [0, 1]")
    if sum(targets.values(), D("0")) > D("1"):
        raise ValueError("base target weights cannot exceed 100%")
    entries = payload.get("ohlc")
    if not isinstance(entries, dict) or set(entries) != set(symbols):
        raise ValueError("ohlc must cover exactly the manifest symbols")
    data: dict[str, SymbolData] = {}
    for symbol in symbols:
        item = entries[symbol]
        if not isinstance(item, dict) or "path" not in item or "sha256" not in item:
            raise ValueError(f"{symbol}: ohlc entry needs path and sha256")
        source = (path.parent / str(item["path"])).resolve()
        actual = sha256(source)
        if actual != str(item["sha256"]):
            raise ValueError(f"{symbol}: OHLC hash mismatch")
        data[symbol] = SymbolData(
            symbol, _read_ohlc(source, symbol=symbol, strict_market_state=strict_market_state), source, actual
        )
    date_sets = {symbol: tuple(str(row["date"]) for row in item.rows) for symbol, item in data.items()}
    first_dates = next(iter(date_sets.values()))
    if any(dates != first_dates for dates in date_sets.values()):
        raise ValueError("all manifest symbols must share the same session dates")
    initial = payload.get("initial_state")
    if initial is None and payload.get("initial_state_source") is not None:
        source_spec = payload["initial_state_source"]
        if not isinstance(source_spec, dict) or "plan" not in source_spec or "execution" not in source_spec:
            raise ValueError("initial_state_source needs pinned plan and execution files")

        def read_pinned_state(spec: object) -> dict[str, object]:
            if not isinstance(spec, dict) or "path" not in spec or "sha256" not in spec:
                raise ValueError("pinned state source needs path and sha256")
            source = (path.parent / str(spec["path"])).resolve()
            if sha256(source) != str(spec["sha256"]):
                raise ValueError("pinned state source hash mismatch")
            raw = json.loads(source.read_text(encoding="utf-8"))
            state = raw.get("state_before") if isinstance(raw, dict) else None
            if not isinstance(state, dict):
                raise ValueError("pinned state source lacks state_before")
            return state

        plan_state = read_pinned_state(source_spec["plan"])
        execution_state = read_pinned_state(source_spec["execution"])
        if json.dumps(plan_state, sort_keys=True) != json.dumps(execution_state, sort_keys=True):
            raise ValueError("plan and execution state_before differ")
        initial = plan_state
    if not isinstance(initial, dict) or "cash" not in initial or "positions" not in initial or "lots" not in initial:
        raise ValueError("initial_state with cash, positions and lots is required")
    initial_cash = _decimal(initial["cash"], "initial_state.cash")
    if initial_cash <= 0:
        raise ValueError("initial_state.cash must be positive")
    raw_positions = initial["positions"]
    if not isinstance(raw_positions, dict) or not set(raw_positions) <= set(symbols):
        raise ValueError("initial_state.positions must use manifest symbols")
    initial_positions = {symbol: _decimal(value, f"initial_state.positions.{symbol}") for symbol, value in raw_positions.items()}
    if any(value < 0 or value % LOT for value in initial_positions.values()):
        raise ValueError("initial positions must be nonnegative 100-share lots")
    raw_lots = initial["lots"]
    if not isinstance(raw_lots, list):
        raise ValueError("initial_state.lots must be a list")
    initial_lots: list[dict[str, str]] = []
    lot_totals: dict[str, Decimal] = {}
    for lot in raw_lots:
        if not isinstance(lot, dict) or not {"symbol", "quantity", "sellable_from", "cost_basis"} <= set(lot):
            raise ValueError("each initial lot needs symbol, quantity, sellable_from and cost_basis")
        symbol = str(lot["symbol"])
        quantity = _decimal(lot["quantity"], f"initial_state.lots.{symbol}.quantity")
        cost_basis = _decimal(lot["cost_basis"], f"initial_state.lots.{symbol}.cost_basis")
        if symbol not in symbols or quantity <= 0 or quantity % LOT or cost_basis < 0:
            raise ValueError("invalid initial lot")
        item = {"symbol": symbol, "quantity": str(quantity), "sellable_from": str(lot["sellable_from"]), "cost_basis": str(cost_basis)}
        initial_lots.append(item)
        lot_totals[symbol] = lot_totals.get(symbol, D("0")) + quantity
    for symbol, quantity in initial_positions.items():
        if lot_totals.get(symbol, D("0")) != quantity:
            raise ValueError(f"initial lots do not equal position for {symbol}")
    if not set(targets) <= set(initial_positions) | set(lot_totals):
        raise ValueError("base target symbols are not covered by the pinned initial state")
    stream_hash = None
    if payload.get("decision_stream") is not None:
        stream = payload["decision_stream"]
        if not isinstance(stream, dict) or "path" not in stream or "sha256" not in stream:
            raise ValueError("decision_stream needs path and sha256")
        stream_path = (path.parent / str(stream["path"])).resolve()
        stream_hash = sha256(stream_path)
        if stream_hash != str(stream["sha256"]):
            raise ValueError("decision stream hash mismatch")
    return Manifest(path, symbols, targets, data, POLICY_VERSION, stream_hash, initial_cash, initial_positions, tuple(initial_lots))


def load_smoke_bundle(path: Path, *, strict_market_state: bool = False) -> Manifest:
    """Adapt only a pinned wc006 manifest; legacy v3 state is rejected."""
    manifest_path = Path(path) / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("smoke bundle lacks manifest.json with pinned initial state")
    return load_manifest(manifest_path, strict_market_state=strict_market_state)


def brain_hash(path: Path) -> str:
    """Hash the complete checkpoint file used to seed a run."""
    return sha256(Path(path))


def plastic_weight_hash(controller: FlyController) -> str:
    """Hash only the candidate plastic weights, not transient neural state."""
    return str(controller.brain.memory()["sha256"])


def verify_plastic_weight_hash(controller: FlyController, expected: str) -> None:
    if plastic_weight_hash(controller) != expected:
        raise RuntimeError("plastic weight hash changed during frozen run")


def map_decoder_to_multiplier(side: str, prior: Decimal) -> tuple[str, Decimal]:
    """Fixed wc006 mapping; decoder magnitude is intentionally ignored."""
    prior = _decimal(prior, "prior multiplier")
    if side == "BUY":
        return "EXPOSE", D("1.0")
    if side == "SELL":
        return "DEEXPOSE", D("0.5")
    if side == "HOLD":
        return "DEFER", prior
    raise ValueError(f"unknown decoder side: {side}")


def _line(frame: Image.Image, values: list[float], box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    if len(values) < 2:
        return
    left, top, right, bottom = box
    lo, hi = min(values), max(values)
    span = max(hi - lo, abs(sum(values) / len(values)) * 0.002, 1e-12)
    points = []
    for index, value in enumerate(values):
        x = left + index * (right - left) / max(1, len(values) - 1)
        y = bottom - (value - lo) / span * (bottom - top)
        points.append((x, y))
    ImageDraw.Draw(frame).line(points, fill=color, width=2)


def composite_frame(
    manifest: Manifest,
    date_index: int,
    exposures: dict[str, Decimal],
    multiplier: Decimal,
) -> np.ndarray:
    """Render all symbols and fixed portfolio state into one RGB observation."""
    image = Image.new("RGB", (320, 180), (235, 240, 249))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 319, 20), fill=(19, 36, 71))
    draw.text((5, 5), f"wc006 {multiplier:.1f}x", fill=(219, 229, 249))
    rows_per_symbol = max(1, 145 // len(manifest.symbols))
    for position, symbol in enumerate(manifest.symbols):
        row = manifest.data[symbol].rows[date_index]
        history = [float(item["close"]) for item in manifest.data[symbol].rows[: date_index + 1]]
        top = 23 + position * rows_per_symbol
        bottom = min(171, top + rows_per_symbol - 5)
        draw.text((4, top), symbol[:13], fill=(28, 46, 82))
        target = manifest.base_targets.get(symbol, D("0")) * multiplier
        exposure = exposures.get(symbol, D("0"))
        draw.text((180, top), f"T {target:.1%} E {exposure:.1%}"[:25], fill=(28, 46, 82))
        _line(image, history[-100:], (72, top + 2, 315, bottom), (0, 101, 183))
    return np.asarray(image, dtype=np.uint8)


def source_manifest_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2] / "stonkfly" / "neural"
    result = {}
    for name in ("sources.lock.json", "neurons.lock.json", "arrays.lock.json", "datasets.json"):
        path = root / name
        if not path.exists():
            raise RuntimeError(f"missing neural source manifest: {name}")
        result[name] = sha256(path)
    return result


def graph_manifest_hash(controller: FlyController) -> str:
    payload = controller.brain.configuration_signature()
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def canonical_signature(manifest: Manifest, *, learning: bool, strict_market_state: bool, capital: Decimal, costs: CostConfig, initial_brain_sha256: str | None = None, source_hashes: dict[str, str] | None = None) -> str:
    payload = {
        "policy": POLICY_VERSION,
        "symbols": manifest.symbols,
        "base_targets": {k: str(v) for k, v in manifest.base_targets.items()},
        "ohlc_sha256": {k: v.source_sha256 for k, v in manifest.data.items()},
        "decision_stream_sha256": manifest.decision_stream_sha256,
        "learning": learning,
        "strict_market_state": strict_market_state,
        "capital": str(capital),
        "costs": costs.as_json(),
        "lot": str(LOT),
        "initial_brain_sha256": initial_brain_sha256,
        "source_manifest_hashes": source_hashes or source_manifest_hashes(),
        "initial_state": {
            "cash": str(manifest.initial_cash),
            "positions": {k: str(v) for k, v in manifest.initial_positions.items()},
            "lots": list(manifest.initial_lots),
        },
        "timing": "close T intent; next valid session open fill",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class WC006Ledger:
    """SQLite Decimal ledger with intent-before-fill and idempotent settlement."""

    def __init__(self, path: Path, *, initial_state: dict[str, object], signature: str, authoritative_trading_state: bool):
        self.path = Path(path)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS observations(
              id INTEGER PRIMARY KEY, trade_date TEXT NOT NULL, phase TEXT NOT NULL,
              input_sha256 TEXT NOT NULL, reinforcement TEXT NOT NULL,
              neural_json TEXT NOT NULL, checkpoint_json TEXT NOT NULL,
              UNIQUE(trade_date, phase));
            CREATE TABLE IF NOT EXISTS intents(
              id TEXT PRIMARY KEY, signal_date TEXT NOT NULL, execution_date TEXT NOT NULL,
              symbol TEXT NOT NULL, neural_side TEXT NOT NULL, order_side TEXT,
              intent_kind TEXT NOT NULL,
              target_exposure TEXT NOT NULL, requested_qty TEXT NOT NULL,
              status TEXT NOT NULL, veto_reason TEXT, observation_id INTEGER NOT NULL,
              checkpoint_sha256 TEXT NOT NULL, UNIQUE(signal_date, symbol));
            CREATE TABLE IF NOT EXISTS fills(
              intent_id TEXT PRIMARY KEY, execution_date TEXT NOT NULL, symbol TEXT NOT NULL,
              quantity TEXT NOT NULL, price TEXT NOT NULL, notional TEXT NOT NULL,
              fee TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS lots(
              lot_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, quantity TEXT NOT NULL,
              sellable_from TEXT NOT NULL, cost_basis TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS account_snapshots(
              trade_date TEXT PRIMARY KEY, cash TEXT NOT NULL, equity TEXT NOT NULL,
              total_exposure TEXT NOT NULL, drawdown TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events(
              event_id TEXT PRIMARY KEY, trade_date TEXT NOT NULL, phase TEXT NOT NULL,
              payload_json TEXT NOT NULL, prev_hash TEXT NOT NULL, event_hash TEXT NOT NULL UNIQUE);
            """
        )
        existing = self.get_meta("signature")
        if existing not in (None, signature):
            raise RuntimeError("wc006 signature mismatch; use a new run directory")
        if existing is None:
            capital = _decimal(initial_state["cash"], "initial_state.cash")
            self.put_meta("signature", signature)
            self.put_meta("cash", str(capital))
            self.put_meta("initial_capital", str(capital))
            self.put_meta("authoritative_trading_state", bool(authoritative_trading_state))
            self.put_meta("schema_version", SCHEMA_VERSION)
            self.put_meta("last_completed_index", -1)
            self.put_meta("last_completed_date", None)
            self.put_meta("multiplier", "1.0")
            for index, lot in enumerate(initial_state["lots"]):
                item = {**lot, "lot_id": hashlib.sha256(json.dumps({"initial": index, **lot}, sort_keys=True).encode()).hexdigest()}
                self.db.execute("INSERT INTO lots VALUES (?,?,?,?,?)", (item["lot_id"], item["symbol"], item["quantity"], item["sellable_from"], item["cost_basis"]))
        elif self.get_meta("schema_version") != SCHEMA_VERSION:
            raise RuntimeError("unsupported wc006 ledger schema; use a new run directory")

    def get_meta(self, key: str):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put_meta(self, key: str, value: object) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value, allow_nan=False)))

    def holdings(self, symbol: str, *, sellable_on: str | None = None) -> Decimal:
        if sellable_on is None:
            rows = self.db.execute("SELECT quantity FROM lots WHERE symbol=?", (symbol,)).fetchall()
        else:
            rows = self.db.execute("SELECT quantity FROM lots WHERE symbol=? AND sellable_from<=?", (symbol, sellable_on)).fetchall()
        return sum((_decimal(row[0], "holding") for row in rows), D("0"))

    def close(self) -> None:
        self.db.close()

    def unresolved(self) -> list[dict[str, object]]:
        rows = self.db.execute("SELECT id,status FROM intents WHERE status='UNKNOWN'").fetchall()
        return [{"id": row[0], "status": row[1]} for row in rows]

    def mark_unknown(self, intent_id: str) -> None:
        with self.db:
            self.db.execute("UPDATE intents SET status='UNKNOWN' WHERE id=? AND status='PREPARED'", (intent_id,))

    def get_observation(self, date: str) -> dict[str, object] | None:
        row = self.db.execute("SELECT id,neural_json,checkpoint_json,reinforcement,input_sha256 FROM observations WHERE trade_date=? AND phase='CLOSE'", (date,)).fetchone()
        if row is None:
            return None
        return {"id": row[0], "neural": json.loads(row[1]), "checkpoint": json.loads(row[2]), "reinforcement": row[3], "input_sha256": row[4]}

    def record_event(self, *, date: str, payload: dict[str, object]) -> tuple[str, bool]:
        event_id = hashlib.sha256(json.dumps({"signature": self.get_meta("signature"), "date": date, "phase": "CLOSE"}, sort_keys=True).encode()).hexdigest()
        payload_json = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)
        existing = self.db.execute("SELECT payload_json,event_hash FROM events WHERE event_id=?", (event_id,)).fetchone()
        if existing is not None:
            if existing[0] != payload_json:
                raise RuntimeError("event payload changed during resume")
            return existing[1], False
        prior = self.db.execute("SELECT event_hash FROM events ORDER BY rowid DESC LIMIT 1").fetchone()
        prev_hash = prior[0] if prior else "GENESIS"
        event_hash = hashlib.sha256((prev_hash + payload_json).encode()).hexdigest()
        with self.db:
            self.db.execute("INSERT INTO events VALUES (?,?,?,?,?,?)", (event_id, date, "CLOSE", payload_json, prev_hash, event_hash))
        return event_hash, True

    def complete_date(self, *, date: str, index: int, multiplier: Decimal, equity: Decimal, drawdown: Decimal, checkpoint: dict[str, object]) -> None:
        with self.db:
            self.put_meta("last_completed_date", date)
            self.put_meta("last_completed_index", index)
            self.put_meta("multiplier", str(multiplier))
            self.put_meta("previous_equity", str(equity))
            self.put_meta("drawdown", str(drawdown))
            self.put_meta("peak_equity", str(equity / (D("1") + drawdown) if drawdown > D("-1") else equity))
            self.put_meta("checkpoint", checkpoint)

    def pending(self, execution_date: str | None = None) -> list[dict[str, object]]:
        query = "SELECT id,signal_date,execution_date,symbol,neural_side,order_side,intent_kind,target_exposure,requested_qty,status FROM intents WHERE status='PREPARED'"
        args: tuple[object, ...] = ()
        if execution_date is not None:
            query += " AND execution_date=?"
            args = (execution_date,)
        query += " ORDER BY execution_date, symbol"
        columns = ("id", "signal_date", "execution_date", "symbol", "neural_side", "order_side", "intent_kind", "target_exposure", "requested_qty", "status")
        return [dict(zip(columns, row)) for row in self.db.execute(query, args)]

    def prepare_intent(self, *, signal_date: str, execution_date: str, symbol: str, neural_side: str, order_side: str | None, intent_kind: str, target_exposure: Decimal, requested_qty: Decimal, observation_id: int, checkpoint_sha256: str, status: str = "PREPARED") -> str:
        if neural_side not in {"BUY", "SELL", "HOLD"}:
            raise ValueError("neural_side must be BUY, SELL or HOLD")
        if order_side not in {None, "BUY", "SELL"}:
            raise ValueError("order_side must be BUY, SELL or null")
        if intent_kind not in {"EXPOSE", "DEEXPOSE", "DEFER"} or requested_qty < 0:
            raise ValueError("invalid intent")
        if status not in {"PREPARED", "UNEXECUTED_END"}:
            raise ValueError("invalid intent status")
        if status == "PREPARED" and (order_side is None or requested_qty <= 0):
            raise ValueError("prepared intent needs a positive order")
        if status == "UNEXECUTED_END" and execution_date != "END":
            raise ValueError("terminal intent must use execution_date END")
        identity = {"signal_date": signal_date, "execution_date": execution_date, "symbol": symbol, "neural_side": neural_side, "order_side": order_side, "intent_kind": intent_kind, "target_exposure": str(target_exposure), "requested_qty": str(requested_qty), "observation_id": observation_id, "checkpoint_sha256": checkpoint_sha256}
        intent_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO intents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (intent_id, signal_date, execution_date, symbol, neural_side, order_side, intent_kind, str(target_exposure), str(requested_qty), status, None, observation_id, checkpoint_sha256),
            )
        return intent_id

    def mark_veto(self, intent_id: str, reason: str) -> None:
        with self.db:
            self.db.execute("UPDATE intents SET status='VETO',veto_reason=? WHERE id=? AND status='PREPARED'", (reason, intent_id))

    def record_observation(self, *, date: str, input_sha256: str, reinforcement: str, neural: dict[str, object], checkpoint: dict[str, object]) -> int:
        existing = self.get_observation(date)
        if existing is not None:
            if existing["input_sha256"] != input_sha256 or existing["checkpoint"] != checkpoint:
                raise RuntimeError("observation changed during resume")
            return int(existing["id"])
        with self.db:
            cursor = self.db.execute(
                "INSERT INTO observations(trade_date,phase,input_sha256,reinforcement,neural_json,checkpoint_json) VALUES (?,?,?,?,?,?)",
                (date, "CLOSE", input_sha256, reinforcement, json.dumps(neural, allow_nan=False), json.dumps(checkpoint, allow_nan=False)),
            )
        return int(cursor.lastrowid)

    def settle(self, intent_id: str, *, execution_date: str, quantity: Decimal, price: Decimal, fee: Decimal, sellable_from: str | None = None) -> None:
        quantity, price, fee = map(lambda x: _decimal(x, "settlement"), (quantity, price, fee))
        if price <= 0 or fee < 0:
            raise ValueError("invalid settlement price or fee")
        with self.db:
            row = self.db.execute("SELECT symbol,neural_side,order_side,requested_qty,status FROM intents WHERE id=?", (intent_id,)).fetchone()
            if row is None:
                raise RuntimeError("unknown intent")
            symbol, neural_side, side, requested, status = row
            if status == "FILLED":
                old = self.db.execute("SELECT quantity,price,fee FROM fills WHERE intent_id=?", (intent_id,)).fetchone()
                if old != (str(quantity), str(price), str(fee)):
                    raise RuntimeError("settlement changed after finalization")
                return
            if status != "PREPARED":
                raise RuntimeError(f"cannot settle intent in {status}")
            if quantity <= 0 or quantity > _decimal(requested, "requested_qty") or quantity % LOT:
                raise ValueError("invalid or over-sized lot settlement")
            notional = quantity * price
            cash = _decimal(self.get_meta("cash"), "cash")
            if side == "BUY":
                if cash < notional + fee:
                    raise RuntimeError("cash overrun")
                cash -= notional + fee
                lot_id = hashlib.sha256(json.dumps({"intent_id": intent_id, "symbol": symbol}, sort_keys=True).encode()).hexdigest()
                self.db.execute("INSERT INTO lots VALUES (?,?,?,?,?)", (lot_id, symbol, str(quantity), sellable_from or AFTER_DATA_END, str(notional + fee)))
            elif side == "SELL":
                remaining = quantity
                lots = self.db.execute("SELECT lot_id,quantity FROM lots WHERE symbol=? AND sellable_from<=? ORDER BY sellable_from,lot_id", (symbol, execution_date)).fetchall()
                for lot_id, lot_qty_raw in lots:
                    if remaining <= 0:
                        break
                    lot_qty = _decimal(lot_qty_raw, "lot quantity")
                    take = min(remaining, lot_qty)
                    remaining -= take
                    if take == lot_qty:
                        self.db.execute("DELETE FROM lots WHERE lot_id=?", (lot_id,))
                    else:
                        self.db.execute("UPDATE lots SET quantity=? WHERE lot_id=?", (str(lot_qty - take), lot_id))
                if remaining:
                    raise RuntimeError("T+1 or inventory limit")
                cash += notional - fee
            else:
                raise RuntimeError("intent decoder side is not executable")
            self.put_meta("cash", str(cash))
            self.db.execute("INSERT INTO fills VALUES (?,?,?,?,?,?,?)", (intent_id, execution_date, symbol, str(quantity), str(price), str(notional), str(fee)))
            self.db.execute("UPDATE intents SET status='FILLED' WHERE id=?", (intent_id,))


def _next_date(rows: Iterable[dict[str, object]], date: str) -> str | None:
    return next((str(row["date"]) for row in rows if str(row["date"]) > date), None)


def _lot_quantity(notional: Decimal, price: Decimal) -> Decimal:
    if notional <= 0 or price <= 0:
        return D("0")
    return (notional / price / LOT).to_integral_value(rounding=ROUND_DOWN) * LOT


def commission(notional: Decimal, rate: Decimal, minimum: Decimal) -> Decimal:
    return max(minimum, notional * rate).quantize(D("0.01"))


def daily_stimulus(equity: Decimal, previous_equity: Decimal, deadband: Decimal = D("0.0001")) -> str:
    """One engineered global reinforcement pulse per trade date."""
    if previous_equity <= 0:
        raise ValueError("previous equity must be positive")
    change = equity / previous_equity - D("1")
    if change >= deadband:
        return "reward"
    if change <= -deadband:
        return "aversive"
    return "none"


def mark_equity(ledger: WC006Ledger, manifest: Manifest, date_index: int) -> Decimal:
    cash = _decimal(ledger.get_meta("cash"), "cash")
    return cash + sum(
        (ledger.holdings(symbol) * manifest.data[symbol].rows[date_index]["close"] for symbol in manifest.symbols),
        D("0"),
    )


def incremental_feedback(
    *,
    equity: Decimal,
    previous_equity: Decimal,
    base_equity: Decimal,
    previous_base_equity: Decimal,
    drawdown: Decimal,
    previous_drawdown: Decimal,
    turnover_ratio: Decimal,
    drawdown_weight: Decimal = D("0.25"),
    turnover_weight: Decimal = D("0.001"),
    deadband: Decimal = D("0.0001"),
    absolute_loss_weight: Decimal = D("1.0"),  # reduced from 1.5 to balance reward/aversive
    absolute_loss_threshold: Decimal = D("-0.005"),
    absolute_gain_weight: Decimal = D("2.0"),  # increased from 1.0 to boost reward signal
    absolute_gain_threshold: Decimal = D("0.002"),  # lowered from 0.005 to trigger reward more often
) -> dict[str, object]:
    """Compute reinforcement feedback for Fly.
    
    The score combines:
    1. Relative return vs Base (original behavior)
    2. Drawdown change penalty
    3. Turnover penalty
    4. **Absolute loss penalty**: When portfolio_return < threshold,
       apply additional penalty proportional to the loss magnitude.
       This prevents Fly from being rewarded when losing money just
       because Base lost more.
    5. **Absolute gain bonus** (NEW): When portfolio_return > threshold,
       apply additional bonus proportional to the gain magnitude.
       This encourages Fly to seek absolute gains, not just beat Base.
    """
    portfolio_return = equity / previous_equity - D("1")
    base_return = base_equity / previous_base_equity - D("1")
    score = portfolio_return - base_return
    score += drawdown_weight * (drawdown - previous_drawdown)
    score -= turnover_weight * turnover_ratio
    
    # Penalize absolute losses even when beating Base
    absolute_loss_penalty = D("0")
    if portfolio_return < absolute_loss_threshold:
        absolute_loss_penalty = absolute_loss_weight * (portfolio_return - absolute_loss_threshold)
        score += absolute_loss_penalty
    
    # NEW: Bonus for absolute gains
    absolute_gain_bonus = D("0")
    if absolute_gain_weight > 0 and portfolio_return > absolute_gain_threshold:
        absolute_gain_bonus = absolute_gain_weight * (portfolio_return - absolute_gain_threshold)
        score += absolute_gain_bonus
    
    stimulus = "reward" if score >= deadband else "aversive" if score <= -deadband else "none"
    return {
        "stimulus": stimulus,
        "score": score,
        "incremental_return": portfolio_return - base_return,
        "portfolio_return": portfolio_return,
        "base_return": base_return,
        "drawdown_change": drawdown - previous_drawdown,
        "turnover_ratio": turnover_ratio,
        "absolute_loss_penalty": absolute_loss_penalty,
        "absolute_gain_bonus": absolute_gain_bonus,
    }


def _base_curve_qty(lots: list[dict[str, str]], symbol: str, sellable_on: str | None = None) -> Decimal:
    """Holdings helper with the end-of-data lot sentinel respected."""
    return sum(
        (_decimal(lot["quantity"], "base lot") for lot in lots if lot["symbol"] == symbol and (sellable_on is None or (lot["sellable_from"] != AFTER_DATA_END and lot["sellable_from"] <= sellable_on))),
        D("0"),
    )


def paired_base_curve(manifest: Manifest, *, costs: CostConfig, max_total_exposure: Decimal) -> list[Decimal]:
    """Deterministic fixed-target arm using the same initial state and costs."""
    cash = manifest.initial_cash
    lots = [dict(lot) for lot in manifest.initial_lots]
    dates = [str(row["date"]) for row in manifest.data[manifest.symbols[0]].rows]

    def qty(symbol: str, sellable_on: str | None = None) -> Decimal:
        return _base_curve_qty(lots, symbol, sellable_on)

    curve: list[Decimal] = []
    for date_index, date in enumerate(dates):
        open_value = cash + sum(qty(symbol) * manifest.data[symbol].rows[date_index]["open"] for symbol in manifest.symbols)
        for symbol, target_weight in manifest.base_targets.items():
            row = manifest.data[symbol].rows[date_index]
            current = qty(symbol)
            target_value = open_value * target_weight
            delta = target_value - current * row["open"]
            side = "BUY" if delta > 0 else "SELL"
            price = row["open"] * (D("1") + costs.slippage if side == "BUY" else D("1") - costs.slippage)
            quantity = _lot_quantity(abs(delta), price)
            if not quantity:
                continue
            fee = commission(quantity * price, costs.commission_rate, costs.minimum_commission)
            if side == "SELL":
                fee += quantity * price * costs.stamp_duty_rate
            if side == "BUY" and cash >= quantity * price + fee:
                total_after = sum(qty(other) * manifest.data[other].rows[date_index]["open"] for other in manifest.symbols) + quantity * price
                if total_after / open_value <= max_total_exposure:
                    cash -= quantity * price + fee
                    lots.append({"symbol": symbol, "quantity": str(quantity), "sellable_from": _next_date(manifest.data[symbol].rows, date) or AFTER_DATA_END, "cost_basis": str(quantity * price + fee)})
            elif side == "SELL" and quantity <= qty(symbol, date):
                remaining = quantity
                for lot in list(lots):
                    if remaining <= 0 or lot["symbol"] != symbol or lot["sellable_from"] > date:
                        continue
                    take = min(remaining, _decimal(lot["quantity"], "base lot"))
                    remaining -= take
                    left = _decimal(lot["quantity"], "base lot") - take
                    if left:
                        lot["quantity"] = str(left)
                    else:
                        lots.remove(lot)
                if not remaining:
                    cash += quantity * price - fee
        curve.append(cash + sum(qty(symbol) * manifest.data[symbol].rows[date_index]["close"] for symbol in manifest.symbols))
    return curve


def validate_intent(
    *,
    manifest: Manifest,
    symbol: str,
    kind: str,
    requested_qty: Decimal,
    current_qty: Decimal,
    sellable_qty: Decimal,
    cash: Decimal,
    price: Decimal,
    equity: Decimal,
    target_exposure: Decimal,
    max_total_exposure: Decimal,
) -> str | None:
    """Return a veto reason, or None. This function never changes the request."""
    if symbol not in manifest.base_targets:
        return "symbol_outside_base_target"
    if kind not in {"EXPOSE", "DEEXPOSE"}:
        return "invalid_intent"
    if requested_qty <= 0 or requested_qty % LOT:
        return "lot_size"
    if price <= 0 or equity <= 0:
        return "invalid_mark"
    if target_exposure < 0 or target_exposure > max_total_exposure:
        return "exposure_cap"
    if kind == "EXPOSE":
        if price * (current_qty + requested_qty) / equity > target_exposure:
            return "target_exposure"
        if cash < price * requested_qty:
            return "cash"
    else:
        if requested_qty > current_qty or requested_qty > sellable_qty:
            return "t_plus_1_or_inventory"
    return None


def _controller_settings(learning: bool):
    # FlyController only consumes these fields; Settings keeps the audited
    # neural defaults while its crypto product allowlist is irrelevant here.
    return Settings(learning=learning)


def run_replay(
    manifest: Manifest,
    out: Path,
    *,
    capital: Decimal = D("100000"),
    costs: CostConfig = CostConfig(),
    max_total_exposure: Decimal = D("0.85"),
    learning: bool = True,
    initial_brain: Path | None = None,
    strict_market_state: bool = False,
    max_steps: int = 0,
) -> dict[str, object]:
    """Run the causal paper loop.  Callers must review tests before invoking."""
    if max_steps < 0:
        raise ValueError("max_steps cannot be negative")
    if capital != manifest.initial_cash:
        raise ValueError("capital must equal the manifest pinned initial cash")
    initial_hash = brain_hash(initial_brain) if initial_brain else None
    source_hashes = source_manifest_hashes()
    signature = canonical_signature(
        manifest,
        learning=learning,
        strict_market_state=strict_market_state,
        capital=capital,
        costs=costs,
        initial_brain_sha256=initial_hash,
        source_hashes=source_hashes,
    )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    authoritative = strict_market_state and all(
        bool(row["authoritative_trading_state"])
        for item in manifest.data.values()
        for row in item.rows
    )
    ledger = WC006Ledger(
        out / "ledger.sqlite",
        initial_state={"cash": str(manifest.initial_cash), "lots": list(manifest.initial_lots)},
        signature=signature,
        authoritative_trading_state=authoritative,
    )
    settings = _controller_settings(learning)
    fly = FlyController(settings)
    if initial_brain:
        fly.restore(initial_brain)
        fly.brain.weights_frozen = not learning
    elif not learning:
        fly.brain.weights_frozen = True
    if ledger.unresolved():
        raise RuntimeError("unknown intent outcome requires reconciliation")
    frozen_hash = plastic_weight_hash(fly) if not learning else None
    base_curve = paired_base_curve(manifest, costs=costs, max_total_exposure=max_total_exposure)
    provenance = {
        "policy": POLICY_VERSION,
        "signature": signature,
        "symbols": manifest.symbols,
        "base_targets": {k: str(v) for k, v in manifest.base_targets.items()},
        "authoritative_trading_state": authoritative,
        "initial_brain_sha256": initial_hash,
        "source_manifest_hashes": source_hashes,
        "graph_manifest_hash": graph_manifest_hash(fly),
        "costs": costs.as_json(),
        "initial_state": {
            "cash": str(manifest.initial_cash),
            "positions": {key: str(value) for key, value in manifest.initial_positions.items()},
            "lots": list(manifest.initial_lots),
        },
        "learning": learning,
        "plastic_weight_sha256_initial": frozen_hash,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    dates = [str(row["date"]) for row in manifest.data[manifest.symbols[0]].rows]
    checkpoint_state = ledger.get_meta("checkpoint")
    if checkpoint_state:
        checkpoint_path = out / checkpoint_state["file"]
        if not checkpoint_path.exists() or sha256(checkpoint_path) != checkpoint_state["sha256"]:
            raise RuntimeError("latest checkpoint missing or hash mismatch")
        fly.restore(checkpoint_path)
        fly.brain.weights_frozen = not learning
    last_index_raw = ledger.get_meta("last_completed_index")
    last_index = int(-1 if last_index_raw is None else last_index_raw)
    start_index = last_index + 1
    multiplier = _decimal(ledger.get_meta("multiplier") or "1.0", "resume multiplier")
    previous_equity = _decimal(ledger.get_meta("previous_equity") or manifest.initial_cash, "resume equity")
    previous_base_equity = base_curve[last_index] if last_index >= 0 else manifest.initial_cash
    peak_equity = max(previous_equity, _decimal(ledger.get_meta("peak_equity") or previous_equity, "resume peak"))
    previous_drawdown = _decimal(ledger.get_meta("drawdown") or "0", "resume drawdown")
    events_path = out / "events.jsonl"
    completed = 0
    try:
        for date_index in range(start_index, len(dates)):
            date = dates[date_index]
            if max_steps and completed >= max_steps:
                break
            # Open execution is always resolved before the close observation.
            daily_turnover = D("0")
            open_equity = _decimal(ledger.get_meta("cash"), "cash")
            for symbol in manifest.symbols:
                open_equity += ledger.holdings(symbol) * manifest.data[symbol].rows[date_index]["open"]
            for intent in ledger.pending(date):
                symbol = str(intent["symbol"])
                row = manifest.data[symbol].rows[date_index]
                reason = None
                if strict_market_state and not row["authoritative_trading_state"]:
                    reason = "missing_official_market_state"
                if strict_market_state and str(row["market_status"]).upper() not in VALID_STATUS:
                    reason = "session_not_open"
                if strict_market_state and reason is None:
                    upper = _decimal(row["official_limit_up"], "official_limit_up")
                    lower = _decimal(row["official_limit_down"], "official_limit_down")
                    if not lower <= row["open"] <= upper:
                        reason = "official_limit"
                price = row["open"] * (
                    D("1") + costs.slippage if intent["order_side"] == "BUY" else D("1") - costs.slippage
                )
                quantity = _decimal(intent["requested_qty"], "requested_qty")
                fee = commission(quantity * price, costs.commission_rate, costs.minimum_commission)
                if intent["order_side"] == "SELL":
                    fee += quantity * price * costs.stamp_duty_rate
                current = ledger.holdings(symbol)
                sellable = ledger.holdings(symbol, sellable_on=date)
                if reason is None:
                    reason = validate_intent(
                        manifest=manifest,
                        symbol=symbol,
                        kind=str(intent["intent_kind"]),
                        requested_qty=quantity,
                        current_qty=current,
                        sellable_qty=sellable,
                        cash=_decimal(ledger.get_meta("cash"), "cash"),
                        price=price,
                        equity=open_equity,
                        target_exposure=_decimal(intent["target_exposure"], "target_exposure"),
                        max_total_exposure=max_total_exposure,
                    )
                    if reason is None and intent["order_side"] == "BUY" and _decimal(ledger.get_meta("cash"), "cash") < quantity * price + fee:
                        reason = "cash_and_fee"
                    if reason is None and intent["order_side"] == "BUY":
                        total_after = sum(
                            ledger.holdings(other) * manifest.data[other].rows[date_index]["open"]
                            for other in manifest.symbols
                        ) + quantity * price
                        if total_after / open_equity > max_total_exposure:
                            reason = "total_exposure"
                if reason is not None:
                    ledger.mark_veto(str(intent["id"]), reason)
                    continue
                next_session = _next_date(manifest.data[symbol].rows, date)
                ledger.settle(
                    str(intent["id"]),
                    execution_date=date,
                    quantity=quantity,
                    price=price,
                    fee=fee,
                    sellable_from=next_session,
                )
                daily_turnover += quantity * price
            close_equity = mark_equity(ledger, manifest, date_index)
            peak_equity = max(peak_equity, close_equity)
            drawdown = close_equity / peak_equity - D("1")
            feedback = incremental_feedback(
                equity=close_equity,
                previous_equity=previous_equity,
                base_equity=base_curve[date_index],
                previous_base_equity=previous_base_equity,
                drawdown=drawdown,
                previous_drawdown=previous_drawdown,
                turnover_ratio=daily_turnover / previous_equity if previous_equity else D("0"),
            )
            stimulus = str(feedback["stimulus"])
            exposures = {
                symbol: (
                    ledger.holdings(symbol) * manifest.data[symbol].rows[date_index]["close"] / close_equity
                    if close_equity > 0 else D("0")
                )
                for symbol in manifest.symbols
            }
            stored = ledger.get_observation(date)
            if stored is not None:
                neural = stored["neural"]
                stimulus = str(stored["reinforcement"])
                checkpoint_info = stored["checkpoint"]
                action = str(neural["wc006_action"])
                new_multiplier = _decimal(neural["wc006_multiplier_after"], "stored multiplier")
                observation_id = int(stored["id"])
            else:
                frame = composite_frame(manifest, date_index, exposures, multiplier)
                neural = fly.observe(frame, stimulus)
                if frozen_hash is not None:
                    verify_plastic_weight_hash(fly, frozen_hash)
                action, new_multiplier = map_decoder_to_multiplier(str(neural["side"]), multiplier)
                neural["wc006_action"] = action
                neural["wc006_multiplier_after"] = str(new_multiplier)
                checkpoint = out / f"brain-{date_index % 2}.npz"
                fly.save(checkpoint)
                checkpoint_info = {"file": checkpoint.name, "sha256": sha256(checkpoint)}
                observation_id = ledger.record_observation(
                    date=date,
                    input_sha256=str(neural["input_sha256"]),
                    reinforcement=stimulus,
                    neural=neural,
                    checkpoint=checkpoint_info,
                )
            if action != "DEFER":
                multiplier = new_multiplier
                for symbol in manifest.base_targets:
                    row = manifest.data[symbol].rows[date_index]
                    target = manifest.base_targets[symbol] * multiplier
                    current = ledger.holdings(symbol)
                    current_exposure = current * row["close"] / close_equity if close_equity > 0 else D("0")
                    delta = (target - current_exposure) * close_equity
                    side = "BUY" if delta > 0 else "SELL"
                    price = row["close"] * (D("1") + costs.slippage if side == "BUY" else D("1") - costs.slippage)
                    quantity = _lot_quantity(abs(delta), price)
                    execution_date = _next_date(manifest.data[symbol].rows, date)
                    terminal = execution_date is None
                    if not quantity and not terminal:
                        continue
                    ledger.prepare_intent(
                        signal_date=date,
                        execution_date=execution_date or "END",
                        symbol=symbol,
                        neural_side=str(neural["side"]),
                        order_side=side,
                        intent_kind="EXPOSE" if side == "BUY" else "DEEXPOSE",
                        target_exposure=target,
                        requested_qty=quantity,
                        observation_id=observation_id,
                        checkpoint_sha256=checkpoint_info["sha256"],
                        status="UNEXECUTED_END" if terminal else "PREPARED",
                    )
            elif date_index == len(dates) - 1:
                for symbol in manifest.base_targets:
                    ledger.prepare_intent(
                        signal_date=date,
                        execution_date="END",
                        symbol=symbol,
                        neural_side="HOLD",
                        order_side=None,
                        intent_kind="DEFER",
                        target_exposure=manifest.base_targets[symbol] * multiplier,
                        requested_qty=D("0"),
                        observation_id=observation_id,
                        checkpoint_sha256=checkpoint_info["sha256"],
                        status="UNEXECUTED_END",
                    )
            event = {
                "date": date,
                "symbols": manifest.symbols,
                "decision": str(neural["side"]),
                "intent_kind": action,
                "multiplier": str(multiplier),
                "reinforcement": stimulus,
                "feedback": {key: str(value) if isinstance(value, Decimal) else value for key, value in feedback.items()},
                "equity": str(close_equity),
                "authoritative_trading_state": authoritative,
                "neural": neural,
                "checkpoint": checkpoint_info,
            }
            event_hash, created = ledger.record_event(date=date, payload=event)
            event["event_hash"] = event_hash
            if created:
                with events_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, allow_nan=False) + "\n")
            ledger.complete_date(
                date=date,
                index=date_index,
                multiplier=multiplier,
                equity=close_equity,
                drawdown=drawdown,
                checkpoint=checkpoint_info,
            )
            previous_equity = close_equity
            previous_base_equity = base_curve[date_index]
            previous_drawdown = drawdown
            completed += 1
        return {"status": "complete", "steps": completed, "signature": signature}
    finally:
        ledger.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--capital", default="100000")
    parser.add_argument("--commission-rate", default="0.0003")
    parser.add_argument("--minimum-commission", default="5")
    parser.add_argument("--slippage", default="0.0005")
    parser.add_argument("--stamp-duty-rate", default="0.001")
    parser.add_argument("--cost-profile", default="wc006-research-5bp")
    parser.add_argument("--max-total-exposure", default="0.85")
    parser.add_argument("--initial-brain", type=Path)
    parser.add_argument("--freeze-learning", action="store_true")
    parser.add_argument("--strict-market-state", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()
    if args.max_steps < 0:
        raise SystemExit("--max-steps cannot be negative")
    manifest = load_manifest(args.manifest, strict_market_state=args.strict_market_state)
    capital = _decimal(args.capital, "capital")
    rate = _decimal(args.commission_rate, "commission_rate")
    minimum = _decimal(args.minimum_commission, "minimum_commission")
    slippage = _decimal(args.slippage, "slippage")
    stamp_duty = _decimal(args.stamp_duty_rate, "stamp_duty_rate")
    max_total = _decimal(args.max_total_exposure, "max_total_exposure")
    if not D("0") < max_total <= D("1") or any(x < 0 for x in (rate, minimum, slippage, stamp_duty)):
        raise SystemExit("invalid financial configuration")
    costs = CostConfig(args.cost_profile, rate, minimum, slippage, stamp_duty)
    result = run_replay(
        manifest,
        args.out,
        capital=capital,
        costs=costs,
        max_total_exposure=max_total,
        learning=not args.freeze_learning,
        initial_brain=args.initial_brain,
        strict_market_state=args.strict_market_state,
        max_steps=args.max_steps,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    main()
