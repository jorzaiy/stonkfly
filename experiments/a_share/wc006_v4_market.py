"""Hash-pinned raw-price, official-state and PIT corporate-action layer.

The WC006 observation frame intentionally consumes the pinned qfq cache.  This
module is the only source allowed for execution prices, marks and reward
equity.  It is paper-only and has no broker or service integration.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from stonkfly.config import D


EXPECTED_PARENT_SHA256 = "ca925aa8d01c947da9eb71195b80bceae7d8a26b8d67cd1327dfdf1442e19a69"
EXPECTED_QA_SHA256 = "4a1d4fa06c0578a73099a4954a7479e734502cac675ddaf70144c3fad8b44c4b"
STATUS_FIELDS = {
    "MARKET_CODE", "TRADE_DATE", "PRECLOSE", "HIGH_LIMITED", "LOW_LIMITED",
    "PRICE_HIGH_LMT_RATE", "PRICE_LOW_LMT_RATE", "IS_ST_SEC", "IS_SUSP_SEC",
    "IS_WD_SEC", "IS_XR_SEC",
}
DIVIDEND_TARGET_ORDINARY = "普通股股东"
DIVIDEND_TARGET_REORGANIZATION_MANAGER = "重整管理人"


class MarketInputsBlocked(ValueError):
    """The raw/status/action evidence cannot support causal execution."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _date(value: object) -> str | None:
    text = str(value or "").strip().replace("-", "")
    if text.endswith(".0"):
        text = text[:-2]
    if not text or text.lower() in {"nan", "none"}:
        return None
    if len(text) != 8 or not text.isdigit():
        raise MarketInputsBlocked(f"invalid ISO date value: {text[:16]}")
    return f"{text[:4]}-{text[4:6]}-{text[6:]}"


def _decimal(value: object, label: str, *, allow_empty: bool = False) -> Decimal | None:
    if value is None or str(value).strip().lower() in {"", "nan", "none"}:
        if allow_empty:
            return None
        raise MarketInputsBlocked(f"missing numeric field: {label}")
    try:
        result = D(value)
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise MarketInputsBlocked(f"invalid numeric field: {label}") from exc
    if not result.is_finite():
        raise MarketInputsBlocked(f"nonfinite numeric field: {label}")
    return result


def _truth(value: object) -> bool:
    return str(value).strip() in {"1", "1.0", "True", "true"}


def _code_alias(code: str) -> str:
    text = str(code).strip().upper()
    return text.split(".", 1)[0] if "." in text else text


def _price_rows(bundle: "XyszMarketInputs", code: str) -> dict[str, dict[str, Decimal]]:
    text = str(code).strip().upper()
    candidates = (text, _code_alias(text), f"{_code_alias(text)}.SZ")
    for candidate in candidates:
        if candidate in bundle.raw_prices:
            return bundle.raw_prices[candidate]
    return {}


def _status_row(bundle: "XyszMarketInputs", code: str, date: str) -> "StatusRow | None":
    text = str(code).strip().upper()
    for candidate in (text, _code_alias(text), f"{_code_alias(text)}.SZ"):
        if (candidate, date) in bundle.status:
            return bundle.status[(candidate, date)]
    return None


@dataclass(frozen=True)
class StatusRow:
    code: str
    date: str
    preclose: Decimal
    high_limit: Decimal
    low_limit: Decimal
    high_rate: Decimal
    low_rate: Decimal
    is_st: bool
    is_suspended: bool
    is_wd: bool
    is_xr: bool


@dataclass(frozen=True)
class CorporateAction:
    action_id: str
    kind: str
    code: str
    announcement_date: str
    record_date: str | None = None
    ex_date: str | None = None
    payout_date: str | None = None
    listing_date: str | None = None
    cash_per_share: Decimal = D("0")
    stock_ratio: Decimal = D("0")
    rights_ratio: Decimal = D("0")
    rights_price: Decimal = D("0")
    status: str = ""


@dataclass(frozen=True)
class XyszMarketInputs:
    root: Path
    parent_sha256: str
    qa_sha256: str
    raw_prices: dict[str, dict[str, dict[str, Decimal]]]
    status: dict[tuple[str, str], StatusRow]
    actions: tuple[CorporateAction, ...]
    artifact_hashes: dict[str, str]


def _artifact_path(fragment: Path, relative: str) -> Path:
    direct = fragment / relative
    if direct.is_file():
        return direct
    raw = fragment / "raw" / relative
    if raw.is_file():
        return raw
    raise MarketInputsBlocked(f"missing pinned artifact: {relative}")


def _child_manifests(root: Path, payload: dict[str, Any], dataset: str) -> list[tuple[Path, dict[str, Any]]]:
    result = []
    for child in payload.get("child_manifests", []):
        if child.get("dataset") != dataset:
            continue
        fragment = root / str(child["fragment"])
        if not fragment.is_file() or sha256(fragment) != child.get("manifest_sha256"):
            raise MarketInputsBlocked(f"child manifest hash mismatch: {child.get('fragment')}")
        result.append((fragment.parent, json.loads(fragment.read_text(encoding="utf-8"))))
    if not result:
        raise MarketInputsBlocked(f"parent manifest has no {dataset} child manifests")
    return result


def _parse_actions(root: Path, payload: dict[str, Any], artifacts: dict[str, str]) -> tuple[CorporateAction, ...]:
    actions: dict[str, CorporateAction] = {}
    for dataset, kind in (("dividend", "DIVIDEND"), ("right_issue", "RIGHTS")):
        for fragment, manifest in _child_manifests(root, payload, dataset):
            for item in manifest.get("artifacts", []):
                path = _artifact_path(fragment, str(item["path"]))
                actual = sha256(path)
                if actual != item.get("sha256"):
                    raise MarketInputsBlocked(f"action artifact hash mismatch: {path.name}")
                artifacts[str(path.relative_to(root))] = actual
                with path.open(newline="", encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        code = str(row.get("MARKET_CODE") or "").strip().upper()
                        ann = _date(row.get("ANN_DATE"))
                        if not code or not ann:
                            continue
                        if kind == "DIVIDEND":
                            # The provider can return a second implemented row
                            # for a reorganization manager. It is not an
                            # ordinary shareholder entitlement. Keep the raw
                            # row hash bound in the bundle, but exclude it from
                            # causal ordinary-holder actions. Unknown nonempty
                            # targets are ambiguous and fail closed.
                            target = str(row.get("DIV_TARGET") or "").strip()
                            if target == DIVIDEND_TARGET_REORGANIZATION_MANAGER:
                                continue
                            if target and target != DIVIDEND_TARGET_ORDINARY:
                                raise MarketInputsBlocked(f"unknown dividend target: {target}")
                            if str(row.get("DIV_PROGRESS") or "") not in {"3", "3.0"}:
                                continue
                            record = _date(row.get("DATE_EQY_RECORD")); ex = _date(row.get("DATE_EX"))
                            payout = _date(row.get("DATE_DVD_PAYOUT")); listing = _date(row.get("LISTINGDATE_OF_DVD_SHR"))
                            effective_dates = [x for x in (record, ex, payout, listing) if x]
                            effective = max(effective_dates) if effective_dates else None
                            if effective is None or ann > effective:
                                continue
                            # Use the preregistered pre-tax cash field only;
                            # personal after-tax treatment is not PIT-stable.
                            cash = _decimal(row.get("DVD_PER_SHARE_PRE_TAX_CASH"), "dividend.pre_tax_cash", allow_empty=True) or D("0")
                            stock = _decimal(row.get("DVD_PER_SHARE_STK"), "dividend.stock", allow_empty=True) or D("0")
                            bonus = _decimal(row.get("DIV_BONUSRATE"), "dividend.bonus", allow_empty=True) or D("0")
                            ratio = stock if stock > 0 else bonus
                            identity = json.dumps([kind, code, record, ex, payout, listing], separators=(",", ":"))
                            action_id = hashlib.sha256(identity.encode()).hexdigest()
                            candidate = CorporateAction(action_id, kind, code, ann, record, ex, payout, listing, cash, ratio, status="implemented")
                            previous = actions.get(action_id)
                            if previous is None or candidate.announcement_date > previous.announcement_date:
                                actions[action_id] = candidate
                            elif previous.announcement_date == candidate.announcement_date and previous != candidate:
                                raise MarketInputsBlocked("conflicting same-date dividend revision")
                        else:
                            if str(row.get("PROGRESS") or "") not in {"3", "3.0"}:
                                continue
                            execute = _date(row.get("EXECUTE_DATE")); ex = _date(row.get("EX_DIVIDEND_DATE")); listed = _date(row.get("LISTED_DATE"))
                            effective_dates = [x for x in (execute, ex, listed) if x]
                            effective = max(effective_dates) if effective_dates else None
                            if effective is None or ann > effective:
                                continue
                            ratio = _decimal(row.get("RATIO"), "rights.ratio", allow_empty=True) or D("0")
                            price = _decimal(row.get("PRICE"), "rights.price", allow_empty=True) or D("0")
                            identity = json.dumps([kind, code, execute, ex, listed], separators=(",", ":"))
                            ident = hashlib.sha256(identity.encode()).hexdigest()
                            candidate = CorporateAction(ident, kind, code, ann, ex_date=ex, listing_date=listed, rights_ratio=ratio, rights_price=price, status="implemented")
                            previous = actions.get(ident)
                            if previous is None or candidate.announcement_date > previous.announcement_date:
                                actions[ident] = candidate
                            elif previous.announcement_date == candidate.announcement_date and previous != candidate:
                                raise MarketInputsBlocked("conflicting same-date rights revision")
    return tuple(sorted(actions.values(), key=lambda item: (item.code, item.announcement_date, item.action_id)))


def load_xysz_market_inputs(parent_manifest: Path, *, expected_parent_sha256: str = EXPECTED_PARENT_SHA256, expected_qa_sha256: str = EXPECTED_QA_SHA256) -> XyszMarketInputs:
    parent_manifest = Path(parent_manifest).resolve()
    root = parent_manifest.parent
    actual_parent = sha256(parent_manifest)
    if actual_parent != expected_parent_sha256:
        raise MarketInputsBlocked("xysz parent manifest hash mismatch")
    qa = root / "quality_report.json"
    if not qa.is_file() or sha256(qa) != expected_qa_sha256:
        raise MarketInputsBlocked("xysz QA hash mismatch")
    payload = json.loads(parent_manifest.read_text(encoding="utf-8"))
    if payload.get("authoritative_trading_state") is not True:
        raise MarketInputsBlocked("xysz market state is not authoritative")
    raw: dict[str, dict[str, dict[str, Decimal]]] = {}
    status: dict[tuple[str, str], StatusRow] = {}
    artifacts: dict[str, str] = {"manifest.json": actual_parent, "quality_report.json": sha256(qa)}
    for fragment, manifest in _child_manifests(root, payload, "kline"):
        for item in manifest.get("artifacts", []):
            path = _artifact_path(fragment, str(item["path"])); actual = sha256(path)
            if actual != item.get("sha256"):
                raise MarketInputsBlocked(f"raw Kline hash mismatch: {path.name}")
            artifacts[str(path.relative_to(root))] = actual
            code = str(item.get("code") or "").strip().upper()
            rows: dict[str, dict[str, Decimal]] = {}
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    date = _date(row.get("kline_time"));
                    if not date or date in rows:
                        raise MarketInputsBlocked(f"raw Kline duplicate/invalid date: {path.name}")
                    values = {name: _decimal(row.get(name), f"{code}.{date}.{name}") for name in ("open", "high", "low", "close")}
                    volume = _decimal(row.get("volume"), f"{code}.{date}.volume", allow_empty=True)
                    if any(value is None or value <= 0 for value in values.values()) or (volume is not None and volume < 0):
                        raise MarketInputsBlocked(f"raw Kline nonpositive/nonfinite row: {path.name}")
                    rows[date] = {**values, "volume": volume}
            raw[code] = rows
    for fragment, manifest in _child_manifests(root, payload, "history_stock_status"):
        for item in manifest.get("artifacts", []):
            path = _artifact_path(fragment, str(item["path"])); actual = sha256(path)
            if actual != item.get("sha256"):
                raise MarketInputsBlocked(f"status artifact hash mismatch: {path.name}")
            artifacts[str(path.relative_to(root))] = actual
            with path.open(newline="", encoding="utf-8") as handle:
                fields = set(handle.readline().strip().split(","))
                if not STATUS_FIELDS <= fields:
                    raise MarketInputsBlocked("status artifact missing required fields")
                handle.seek(0)
                for row in csv.DictReader(handle):
                    code = str(row.get("MARKET_CODE") or "").strip().upper(); date = _date(row.get("TRADE_DATE"))
                    if not code or not date or (code, date) in status:
                        raise MarketInputsBlocked("status duplicate or invalid code-date")
                    flags = [row.get(name) for name in ("IS_ST_SEC", "IS_SUSP_SEC", "IS_WD_SEC", "IS_XR_SEC")]
                    if any(str(value) not in {"0", "1", "0.0", "1.0"} for value in flags):
                        raise MarketInputsBlocked("status flag is not binary")
                    pre = _decimal(row.get("PRECLOSE"), "status.preclose")
                    hi = _decimal(row.get("HIGH_LIMITED"), "status.high_limit", allow_empty=True)
                    lo = _decimal(row.get("LOW_LIMITED"), "status.low_limit", allow_empty=True)
                    hr = _decimal(row.get("PRICE_HIGH_LMT_RATE"), "status.high_rate")
                    lr = _decimal(row.get("PRICE_LOW_LMT_RATE"), "status.low_rate")
                    if hi is None or lo is None:
                        # The pinned collector permits absent limits only on
                        # suspended days. The risk guard vetoes those orders
                        # before looking at price limits; represent their
                        # unavailable quotes with the existing sentinel.
                        if not (_truth(row.get("IS_SUSP_SEC")) and hi is None and lo is None):
                            raise MarketInputsBlocked("status limit values missing outside suspension")
                        hi = lo = D("0")
                        hr = lr = D("999")
                    assert pre is not None and hi is not None and lo is not None and hr is not None and lr is not None
                    sentinel = hi == 0 and lo == 0 and hr == 999 and lr == 999
                    if pre <= 0 or (not sentinel and (hi <= 0 or lo <= 0 or lo > hi or hr <= 0 or lr <= 0)):
                        raise MarketInputsBlocked("status limit values invalid")
                    value = StatusRow(code, date, pre, hi, lo, hr, lr, _truth(row.get("IS_ST_SEC")), _truth(row.get("IS_SUSP_SEC")), _truth(row.get("IS_WD_SEC")), _truth(row.get("IS_XR_SEC")))
                    status[(code, date)] = value
    # These files are not substituted into the execution price layer, but they
    # remain part of the immutable action/factor evidence set and must be
    # hash-verified before the bundle can be accepted.
    for dataset in ("adj_factor", "equity_structure"):
        for fragment, manifest in _child_manifests(root, payload, dataset):
            for item in manifest.get("artifacts", []):
                path = _artifact_path(fragment, str(item["path"])); actual = sha256(path)
                if actual != item.get("sha256"):
                    raise MarketInputsBlocked(f"{dataset} artifact hash mismatch: {path.name}")
                artifacts[str(path.relative_to(root))] = actual
    if not raw or set(raw) != {str(x).upper() for x in json.loads((root / "manifest.json").read_text()).get("source_manifest", {}).get("codes", [])}:
        expected = int(payload.get("source_manifest", {}).get("code_count", 0))
        if len(raw) != expected:
            raise MarketInputsBlocked("raw Kline code coverage mismatch")
    if not status or len({code for code, _ in status if "." in code}) != len(raw):
        raise MarketInputsBlocked("official status code coverage mismatch")
    actions = _parse_actions(root, payload, artifacts)
    return XyszMarketInputs(root, actual_parent, expected_qa_sha256, raw, status, actions, artifacts)


def raw_mark(bundle: XyszMarketInputs, code: str, date: str) -> tuple[str, Decimal]:
    prices = _price_rows(bundle, code)
    available = [item for item in prices if item <= date]
    if not available:
        raise MarketInputsBlocked(f"no raw causal mark for {code} at {date}")
    mark_date = max(available); return mark_date, prices[mark_date]["close"]


def raw_open_mark(bundle: XyszMarketInputs, code: str, date: str) -> tuple[str, Decimal]:
    prices = _price_rows(bundle, code)
    if date in prices and prices[date].get("open") is not None:
        return date, prices[date]["open"]
    available = [item for item in prices if item <= date]
    if not available:
        raise MarketInputsBlocked(f"no raw causal open mark for {code} at {date}")
    mark_date = max(available)
    return mark_date, prices[mark_date]["close"]


def raw_open_equity(ledger: Any, bundle: XyszMarketInputs, date: str, active: set[str]) -> tuple[Decimal, dict[str, dict[str, object]]]:
    total = D(ledger.get_meta("cash")); marks: dict[str, dict[str, object]] = {}
    held = {str(row[0]) for row in ledger.db.execute("SELECT DISTINCT symbol FROM lots WHERE CAST(quantity AS REAL)>0")}
    for code in sorted(set(active) | held):
        mark_date, mark = raw_open_mark(bundle, code, date)
        total += ledger.holdings(code) * mark
        marks[code] = {"mark_date": mark_date, "stale": mark_date != date, "source": "xysz_raw_kline.open_or_last_close", "missing_bar": mark_date != date}
    return total, marks


def execution_guard(bundle: XyszMarketInputs, code: str, date: str, side: str, price: Decimal) -> str | None:
    row = _price_rows(bundle, code).get(date)
    state = _status_row(bundle, code, date)
    if row is None or state is None:
        return "missing_open_or_unverified_suspension"
    if state.is_suspended:
        return "missing_open_or_unverified_suspension"
    if state.is_st:
        return "official_st_security"
    sentinel = state.high_limit == 0 and state.low_limit == 0 and state.high_rate == 999 and state.low_rate == 999
    if not sentinel and side == "BUY" and price > state.high_limit:
        return "official_limit_up"
    if not sentinel and side == "SELL" and price < state.low_limit:
        return "official_limit_down"
    return None


def raw_execution_price(bundle: XyszMarketInputs, code: str, date: str, side: str, slippage: Decimal) -> Decimal:
    row = _price_rows(bundle, code).get(date)
    if row is None or row.get("open") is None:
        raise MarketInputsBlocked("missing_open_or_unverified_suspension")
    price = row["open"] * (D("1") + slippage if side == "BUY" else D("1") - slippage)
    reason = execution_guard(bundle, code, date, side, price)
    if reason:
        raise MarketInputsBlocked(reason)
    return price


def ensure_action_tables(ledger: Any) -> None:
    ledger.db.executescript("""
      CREATE TABLE IF NOT EXISTS corporate_action_entitlements(
        action_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, record_date TEXT,
        ex_date TEXT, payout_date TEXT, listing_date TEXT, cash_receivable TEXT NOT NULL,
        stock_receivable TEXT NOT NULL, cash_paid TEXT NOT NULL,
        stock_listed TEXT NOT NULL, rights_decision TEXT NOT NULL);
    """)
    columns = {row[1] for row in ledger.db.execute("PRAGMA table_info(corporate_action_entitlements)")}
    if "ex_date" not in columns:
        ledger.db.execute("ALTER TABLE corporate_action_entitlements ADD COLUMN ex_date TEXT")


def _decimal_text(value: Decimal) -> str:
    """Serialize Decimal state without exponent notation or redundant zeroes."""
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text if "." in text else f"{text}.0"


def apply_corporate_actions(ledger: Any, bundle: XyszMarketInputs, date: str, calendar: tuple[str, ...], *, stage: str = "start_of_day") -> list[dict[str, str]]:
    """Apply one explicit corporate-action phase for *date*.

    ``start_of_day`` handles rights/payout/listing before the open. The
    ``after_open_settlement`` phase records DIVIDEND ownership after all open
    fills, making record-date ownership depend on resulting holdings.
    ``start_of_day`` is the only pre-open phase: it advances already locked
    entitlements and records rights lifecycle events. It must not inspect
    holdings for a dividend. ``after_open_settlement`` is the only phase
    that locks record-date dividend ownership, after all open fills have
    settled.
    """
    if stage not in {"start_of_day", "after_open_settlement"}:
        raise ValueError("invalid corporate-action stage")
    ensure_action_tables(ledger); events: list[dict[str, str]] = []
    lifecycle = stage == "start_of_day"
    capture = stage == "after_open_settlement"
    for action in bundle.actions:
        if action.announcement_date > date:
            continue
        row = ledger.db.execute("SELECT action_id,symbol,record_date,ex_date,payout_date,listing_date,cash_receivable,stock_receivable,cash_paid,stock_listed,rights_decision FROM corporate_action_entitlements WHERE action_id=?", (action.action_id,)).fetchone()
        if action.kind == "RIGHTS":
            if lifecycle and row is None and action.ex_date and date == action.ex_date:
                ledger.db.execute("INSERT INTO corporate_action_entitlements (action_id,symbol,record_date,ex_date,payout_date,listing_date,cash_receivable,stock_receivable,cash_paid,stock_listed,rights_decision) VALUES (?,?,?,?,?,?,?,?,?,?,?)", (action.action_id, action.code, None, action.ex_date, None, action.listing_date, "0", "0", "0", "0", "DECLINE_NO_SUBSCRIPTION"))
                events.append({"action_id": action.action_id, "kind": action.kind, "decision": "DECLINE_NO_SUBSCRIPTION"})
            continue
        if capture and row is None and action.record_date and date == action.record_date:
            symbol = action.code if ledger.holdings(action.code) > 0 else _code_alias(action.code)
            qty = ledger.holdings(symbol)
            cash = qty * action.cash_per_share; stock = qty * action.stock_ratio
            if stock != stock.to_integral_value():
                raise MarketInputsBlocked("stock entitlement is fractional without a defined cash-in-lieu rule")
            ledger.db.execute("INSERT INTO corporate_action_entitlements (action_id,symbol,record_date,ex_date,payout_date,listing_date,cash_receivable,stock_receivable,cash_paid,stock_listed,rights_decision) VALUES (?,?,?,?,?,?,?,?,?,?,?)", (action.action_id, symbol, action.record_date, action.ex_date, action.payout_date, action.listing_date, _decimal_text(cash), _decimal_text(stock), "0.0", "0.0", "PENDING"))
            events.append({"action_id": action.action_id, "kind": action.kind, "event": "ENTITLEMENT", "quantity": str(qty)})
            row = ledger.db.execute("SELECT action_id,symbol,record_date,ex_date,payout_date,listing_date,cash_receivable,stock_receivable,cash_paid,stock_listed,rights_decision FROM corporate_action_entitlements WHERE action_id=?", (action.action_id,)).fetchone()
        if row is None:
            continue
        # columns: id,symbol,record,ex,payout,listing,cash,stock,cash_paid,stock_listed,decision
        if lifecycle and action.payout_date and date >= action.payout_date and D(row[8]) == D("0"):
            cash = D(row[6]); ledger.put_meta("cash", _decimal_text(D(ledger.get_meta("cash")) + cash)); ledger.db.execute("UPDATE corporate_action_entitlements SET cash_paid=? WHERE action_id=?", (_decimal_text(cash), action.action_id)); events.append({"action_id": action.action_id, "event": "CASH_PAYOUT", "cash": _decimal_text(cash)})
        if lifecycle and action.listing_date and date >= action.listing_date and D(row[9]) == D("0"):
            stock = D(row[7]); lot_qty = stock
            if lot_qty != lot_qty.to_integral_value():
                raise MarketInputsBlocked("stock listing is fractional without a defined cash-in-lieu rule")
            if lot_qty > 0:
                lot_id = hashlib.sha256(f"action:{action.action_id}".encode()).hexdigest(); next_date = next((x for x in calendar if x > date), "9999-12-31")
                symbol = action.code if ledger.holdings(action.code) > 0 else _code_alias(action.code)
                ledger.db.execute("INSERT OR IGNORE INTO lots VALUES (?,?,?,?,?)", (lot_id, symbol, str(lot_qty), next_date, "0"))
            ledger.db.execute("UPDATE corporate_action_entitlements SET stock_listed=? WHERE action_id=?", (_decimal_text(lot_qty), action.action_id)); events.append({"action_id": action.action_id, "event": "STOCK_LISTING", "listed_quantity": _decimal_text(lot_qty), "receivable_quantity": _decimal_text(stock - lot_qty)})
    ledger.db.commit(); return events


def action_receivable_value(ledger: Any, bundle: XyszMarketInputs, date: str) -> Decimal:
    ensure_action_tables(ledger); value = D("0")
    for row in ledger.db.execute("SELECT symbol,ex_date,cash_receivable,cash_paid,stock_receivable,stock_listed FROM corporate_action_entitlements"):
        if row[1] is None or date < row[1]:
            continue
        value += D(row[2]) - D(row[3]); remaining = D(row[4]) - D(row[5])
        if remaining > 0:
            _, mark = raw_mark(bundle, row[0], date); value += remaining * mark
    return value


def raw_equity(ledger: Any, bundle: XyszMarketInputs, date: str, active: set[str]) -> tuple[Decimal, dict[str, dict[str, object]]]:
    total = D(ledger.get_meta("cash")); marks: dict[str, dict[str, object]] = {}
    held = {str(row[0]) for row in ledger.db.execute("SELECT DISTINCT symbol FROM lots WHERE CAST(quantity AS REAL)>0")}
    for code in sorted(set(active) | held):
        mark_date, close = raw_mark(bundle, code, date); total += ledger.holdings(code) * close
        marks[code] = {"mark_date": mark_date, "stale": mark_date != date, "source": "xysz_raw_kline", "missing_bar": mark_date != date}
    total += action_receivable_value(ledger, bundle, date)
    return total, marks


def quarantine_run(run: Path, *, reason: str, manifest_sha256: str | None = None) -> Path:
    run = Path(run); sidecar = run / "QUARANTINED.json"
    if sidecar.exists():
        return sidecar
    payload = {"status": "QUARANTINED", "reason": reason, "run": str(run), "manifest_sha256": manifest_sha256, "brain_resume_forbidden": True}
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sidecar
