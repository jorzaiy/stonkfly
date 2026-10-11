#!/usr/bin/env python3
"""Build an isolated, strict-ready wc006 paper-smoke manifest.

The report-only bundle is an immutable input description, not execution
authority.  This builder derives a fresh ``wc006-v1`` manifest only after
verifying all pinned hashes and obtaining complete official market-state rows
from AmazingData.  It never writes to the trading-agent repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[2]
TRADING_ROOT = ROOT.parent / "trading_agent_20260408"
DEFAULT_REPORT_MANIFEST = (
    TRADING_ROOT
    / "project/data/report_only/stonkfly_wc006/smoke_20260831_to_20260901/manifest.json"
)
SIGNAL_DATE = "2026-08-31"
EXECUTION_DATE = "2026-09-01"
REQUIRED_STATUS_FIELDS = (
    "MARKET_CODE",
    "TRADE_DATE",
    "HIGH_LIMITED",
    "LOW_LIMITED",
    "IS_ST_SEC",
    "IS_SUSP_SEC",
    "IS_WD_SEC",
    "IS_XR_SEC",
)


class BuildBlocked(ValueError):
    """Raised when strict paper-smoke evidence is incomplete."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BuildBlocked(f"JSON object required: {path}")
    return value


def _resolve(root: Path, raw: object) -> Path:
    value = Path(str(raw))
    return value if value.is_absolute() else (root / value).resolve()


def _canonical_code(value: object) -> str:
    raw = str(value).strip().upper()
    for suffix in (".SH", ".SZ", ".BJ"):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
    if not raw.isdigit() or len(raw) != 6:
        raise BuildBlocked(f"invalid official code mapping: {value!r}")
    return raw


def _qualified_code(code: str) -> str:
    # All six pinned candidates are Shenzhen A shares; keep the mapping
    # explicit and validate the returned MARKET_CODE rather than guessing.
    return f"{code}.SZ"


def _date8(value: object) -> str:
    raw = str(value).strip()
    if isinstance(value, datetime):
        raw = value.strftime("%Y-%m-%d")
    elif isinstance(value, date):
        raw = value.isoformat()
    digits = raw.replace("-", "").replace("/", "")
    if len(digits) != 8 or not digits.isdigit():
        raise BuildBlocked(f"invalid official trade date: {value!r}")
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"


def _decimal_text(value: object, label: str, *, positive: bool = False) -> str:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise BuildBlocked(f"{label} is not numeric") from exc
    if not result.is_finite() or (positive and result <= 0):
        raise BuildBlocked(f"{label} is not positive finite numeric")
    return format(result, "f")


def _status_records(response: object) -> list[dict[str, Any]]:
    if hasattr(response, "to_dict"):
        response = response.to_dict("records")
    if isinstance(response, dict):
        if all(isinstance(value, list) for value in response.values()):
            keys = list(response)
            response = [dict(zip(keys, values)) for values in zip(*(response[key] for key in keys))]
        else:
            response = list(response.values())
    if not isinstance(response, (list, tuple)):
        raise BuildBlocked("history_stock_status response is not tabular")
    records = []
    for item in response:
        if hasattr(item, "to_dict"):
            item = item.to_dict()
        if not isinstance(item, dict):
            raise BuildBlocked("history_stock_status row is not an object")
        records.append(dict(item))
    return records


def _verify_report_bundle(path: Path) -> tuple[dict[str, Any], Path, dict[str, str]]:
    path = Path(path).resolve()
    manifest = _load_json(path)
    if manifest.get("schema") != "stonkfly-wc006-report-only-smoke/v1":
        raise BuildBlocked("unsupported report-only smoke manifest schema")
    if manifest.get("immutable") is not True or manifest.get("report_only") is not True:
        raise BuildBlocked("report-only immutable boundary missing")
    if manifest.get("signal_date_iso") != SIGNAL_DATE or manifest.get("execution_date_iso") != EXECUTION_DATE:
        raise BuildBlocked("report bundle dates do not match the fixed smoke window")
    root = _resolve(path.parent, manifest.get("repository_root", "../../../../../"))
    checked: dict[str, str] = {}
    source_files = manifest.get("source_files")
    if not isinstance(source_files, dict):
        raise BuildBlocked("report source_files missing")
    for name in ("signal", "plan", "execution", "calendar", "active_config", "universe"):
        item = source_files.get(name)
        if not isinstance(item, dict):
            raise BuildBlocked(f"report source {name} missing")
        source = _resolve(root, item.get("path"))
        if _sha256(source) != str(item.get("sha256")):
            raise BuildBlocked(f"report source hash mismatch: {name}")
        checked[name] = str(source)
    cache_files = manifest.get("canonical_cache", {}).get("files")
    candidates = manifest.get("candidate_set", {}).get("codes")
    if not isinstance(cache_files, list) or not isinstance(candidates, list):
        raise BuildBlocked("report canonical cache or candidate set missing")
    candidate_codes = [str(code) for code in candidates]
    if candidate_codes != sorted(set(candidate_codes)) or len(candidate_codes) != 6:
        raise BuildBlocked("report candidate set must contain six sorted unique symbols")
    cache_hashes: dict[str, str] = {}
    for item in cache_files:
        code = str(item.get("code"))
        source = _resolve(root, item.get("path"))
        if code not in candidate_codes or _sha256(source) != str(item.get("sha256")):
            raise BuildBlocked(f"canonical cache hash/code mismatch: {code}")
        cache_hashes[code] = str(item["sha256"])
    if set(cache_hashes) != set(candidate_codes):
        raise BuildBlocked("canonical cache does not cover exactly six candidates")
    signal = _load_json(Path(checked["signal"]))
    plan = _load_json(Path(checked["plan"]))
    execution = _load_json(Path(checked["execution"]))
    calendar = _load_json(Path(checked["calendar"]))
    if signal.get("asof_date") != "2026-08-31" or plan.get("signal_date") != "20260831":
        raise BuildBlocked("pinned signal/plan as-of mismatch")
    if execution.get("signal_date") != "20260831" or execution.get("execution_date") != "20260901":
        raise BuildBlocked("pinned execution dates mismatch")
    target_rows = manifest.get("base_target_weights")
    signal_targets = signal.get("target_topk")
    plan_targets = plan.get("target_topk")
    if not isinstance(target_rows, list) or not isinstance(signal_targets, list) or not isinstance(plan_targets, list):
        raise BuildBlocked("pinned target weights missing")
    def target_signature(rows: list[dict[str, Any]]) -> list[tuple[object, ...]]:
        return [(str(row.get("code")), int(row.get("rank")), row.get("score"), row.get("target_weight")) for row in rows]
    if target_signature(target_rows) != target_signature(signal_targets) or target_signature(target_rows) != target_signature(plan_targets):
        raise BuildBlocked("pinned base targets disagree with signal/plan")
    dates = calendar.get("dates")
    if not isinstance(dates, list) or SIGNAL_DATE not in dates or EXECUTION_DATE not in dates:
        raise BuildBlocked("pinned calendar lacks signal/execution dates")
    if sorted(str(item) for item in dates if str(item) > SIGNAL_DATE)[:1] != [EXECUTION_DATE]:
        raise BuildBlocked("execution date is not first next valid session")
    return manifest, root, {**checked, **{f"cache:{code}": str(_resolve(root, next(item["path"] for item in cache_files if str(item["code"]) == code))) for code in candidate_codes}}


def _canonical_rows(path: Path, dates: tuple[str, ...] = (SIGNAL_DATE, EXECUTION_DATE)) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    wanted = set(dates)
    selected = [row for row in rows if str(row.get("date")) in wanted]
    if len(selected) != len(wanted) or {str(row.get("date")) for row in selected} != wanted:
        raise BuildBlocked(f"canonical OHLC lacks exact smoke dates: {path}")
    return selected


def _normalise_status(records: Iterable[dict[str, Any]], codes: list[str]) -> dict[tuple[str, str], dict[str, str]]:
    expected = {(code, day) for code in codes for day in (SIGNAL_DATE, EXECUTION_DATE)}
    result: dict[tuple[str, str], dict[str, str]] = {}
    for record in records:
        missing = [field for field in REQUIRED_STATUS_FIELDS if field not in record]
        if missing:
            raise BuildBlocked(f"official status fields missing: {missing}")
        code = _canonical_code(record["MARKET_CODE"])
        day = _date8(record["TRADE_DATE"])
        if code not in codes or (code, day) not in expected or (code, day) in result:
            continue
        st = str(record["IS_ST_SEC"]).strip()
        susp = str(record["IS_SUSP_SEC"]).strip()
        wd = str(record["IS_WD_SEC"]).strip()
        xr = str(record["IS_XR_SEC"]).strip()
        if any(flag not in {"0", "1"} for flag in (st, susp, wd, xr)):
            raise BuildBlocked(f"official status flags incomplete for {code} {day}")
        result[(code, day)] = {
            "market_code": str(record["MARKET_CODE"]),
            "trade_date": day,
            "official_status": "SUSPENDED" if susp == "1" else "ST" if st == "1" else "OPEN",
            "official_limit_up": _decimal_text(record["HIGH_LIMITED"], f"{code} {day} HIGH_LIMITED", positive=True),
            "official_limit_down": _decimal_text(record["LOW_LIMITED"], f"{code} {day} LOW_LIMITED", positive=True),
            "is_st_sec": st,
            "is_susp_sec": susp,
            "is_wd_sec": wd,
            "is_xr_sec": xr,
        }
    if set(result) != expected:
        missing = sorted(expected - set(result))
        raise BuildBlocked(f"official status coverage incomplete: {missing}")
    return result


def _assert_output_isolated(output_dir: Path, independent_root: Path) -> Path:
    output_dir = Path(output_dir).resolve()
    independent_root = Path(independent_root).resolve()
    try:
        output_dir.relative_to(independent_root)
    except ValueError as exc:
        raise BuildBlocked("output directory must be inside the isolated Stonkfly root") from exc
    if output_dir == independent_root or output_dir.exists():
        raise BuildBlocked("output directory must be a new unique path")
    return output_dir


def build_manifest(
    report_manifest: Path,
    output_dir: Path,
    *,
    status_rows: Iterable[dict[str, Any]] | None = None,
    status_client: Callable[[list[str], Path], object] | None = None,
    independent_root: Path = ROOT,
    provisional: bool = False,
    extra_session: str | None = None,
) -> Path:
    """Build a fresh manifest, or fail closed without output.

    The normal path is strict and requires complete official market-state
    evidence. ``provisional=True`` is an explicit paper-only degradation for
    execution-timing/ledger tests: it never queries the status API and never
    writes official status or limit fields.
    """
    manifest, root, checked = _verify_report_bundle(Path(report_manifest))
    if extra_session is not None:
        if not provisional:
            raise BuildBlocked("extra_session is provisional-only")
        extra_session = _date8(extra_session)
        if extra_session == EXECUTION_DATE:
            raise BuildBlocked("extra_session must extend beyond the execution date")
        calendar = _load_json(Path(checked["calendar"]))
        calendar_dates = [str(item) for item in calendar.get("dates", [])]
        if extra_session not in calendar_dates or sorted(day for day in calendar_dates if day > EXECUTION_DATE)[:1] != [extra_session]:
            raise BuildBlocked("extra_session is not the next valid session in the pinned calendar")
    output = _assert_output_isolated(Path(output_dir), independent_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent)))
    candidates = [str(code) for code in manifest["candidate_set"]["codes"]]
    source_paths = {code: Path(checked[f"cache:{code}"]) for code in candidates}
    extra_sources: dict[str, Path] = {}
    extra_tag: str | None = None
    if extra_session is not None:
        tags = set()
        parents = set()
        for code, source in source_paths.items():
            stem = source.stem
            prefix, old_date = stem.rsplit("_", 1)
            if old_date != "2026-09-01" or "_" not in prefix:
                raise BuildBlocked("canonical source exact_tag is not the pinned 2026-09-01 version")
            tags.add(prefix.split("_", 1)[1])
            parents.add(source.parent.resolve())
            extra_sources[code] = source.parent / f"{prefix}_{extra_session}.csv"
        if len(tags) != 1 or len(parents) != 1:
            raise BuildBlocked("canonical sources do not share one exact_tag directory")
        extra_tag = f"{next(iter(tags))}_{extra_session}"
        for code, source in extra_sources.items():
            expected_name = f"{code}_{extra_tag}.csv"
            if source.name != expected_name or not source.is_file():
                raise BuildBlocked(f"missing same-tag extra canonical source: {expected_name}")
    session_dates = (SIGNAL_DATE, EXECUTION_DATE, extra_session) if extra_session is not None else (SIGNAL_DATE, EXECUTION_DATE)
    try:
        if provisional and (status_rows is not None or status_client is not None):
            raise BuildBlocked("provisional mode cannot accept or query official status evidence")
        normalized: dict[tuple[str, str], dict[str, str]] | None = None
        if status_rows is None:
            if provisional:
                pass
            elif status_client is None:
                missing = [name for name in ("AD_USERNAME", "AD_PASSWORD", "AD_HOST", "AD_PORT") if not os.getenv(name)]
                if missing:
                    raise BuildBlocked(f"AmazingData credentials missing: {','.join(missing)}")
                try:
                    import AmazingData as ad
                except Exception as exc:
                    raise BuildBlocked("AmazingData import unavailable") from exc
                try:
                    cache_path = staging / "amazingdata_cache"
                    cache_path.mkdir(parents=True, exist_ok=False)
                    ad.login(
                        username=os.environ["AD_USERNAME"],
                        password=os.environ["AD_PASSWORD"],
                        host=os.environ["AD_HOST"],
                        port=int(os.environ["AD_PORT"]),
                    )
                    status_rows = ad.InfoData().get_history_stock_status(
                        [_qualified_code(code) for code in candidates],
                        local_path=str(cache_path),
                        is_local=False,
                        begin_date=20260831,
                        end_date=20260901,
                    )
                except Exception as exc:
                    raise BuildBlocked(f"AmazingData status query failed: {type(exc).__name__}") from exc
            else:
                try:
                    status_rows = status_client([_qualified_code(code) for code in candidates], staging / "amazingdata_cache")
                except Exception as exc:
                    raise BuildBlocked(f"official status query failed: {type(exc).__name__}") from exc
        if not provisional:
            raw_records = _status_records(status_rows)
            normalized = _normalise_status(raw_records, candidates)

        evidence_dir = staging / "evidence"
        evidence_dir.mkdir()
        ohlc_dir = staging / "ohlc"
        ohlc_dir.mkdir()
        if not provisional:
            raw_path = evidence_dir / "history_stock_status.raw.json"
            raw_path.write_text(json.dumps(raw_records, ensure_ascii=False, allow_nan=False, default=str, indent=2) + "\n", encoding="utf-8")
            normalized_path = evidence_dir / "history_stock_status.normalized.json"
            normalized_payload = [normalized[key] for key in sorted(normalized)]
            normalized_path.write_text(json.dumps(normalized_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        ohlc_entries: dict[str, dict[str, str]] = {}
        for code in candidates:
            source = extra_sources.get(code, source_paths[code])
            destination = ohlc_dir / f"{code}.csv"
            rows = _canonical_rows(source, session_dates)
            with destination.open("w", newline="", encoding="utf-8") as handle:
                fields = ["date", "open", "high", "low", "close", "volume"]
                if not provisional:
                    fields += ["official_status", "official_limit_up", "official_limit_down"]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    day = _date8(row["date"])
                    output_row = {
                        "date": day,
                        "open": row["open"], "high": row["high"], "low": row["low"], "close": row["close"], "volume": row["volume"],
                    }
                    if not provisional:
                        state = normalized[(code, day)]
                        output_row.update({
                            "official_status": state["official_status"],
                            "official_limit_up": state["official_limit_up"],
                            "official_limit_down": state["official_limit_down"],
                        })
                    writer.writerow(output_row)
            ohlc_entries[code] = {"path": f"ohlc/{code}.csv", "sha256": _sha256(destination)}

        targets = manifest.get("base_target_weights")
        pinned_targets = {str(item["code"]): str(item["target_weight"]) for item in targets}
        # Preserve the pinned target subset. The replay loader and executor
        # intentionally iterate only these symbols; zero-weight exit-only
        # candidates must not become synthetic SELL intents.
        base_targets = pinned_targets
        provenance = {
            "builder": "build_wc006_strict_smoke.py",
            "derived_from": {
                "report_manifest": _sha256(Path(report_manifest)),
                "pinned_source_files": {name: _sha256(Path(path)) for name, path in checked.items()},
                **({"extra_canonical_source_files": {
                    code: {"path": str(source.relative_to(root)), "sha256": _sha256(source), "exact_tag": extra_tag}
                    for code, source in extra_sources.items()
                }} if extra_session is not None else {}),
            },
            "paper_only": True,
            "clean_paper_initial_state": {"cash": "100000", "positions": {code: "0" for code in candidates}, "lots": []},
            "authoritative_trading_state": not provisional,
            "official_status_source": (
                "not queried; external official-state access unavailable"
                if provisional else "AmazingData.InfoData.get_history_stock_status"
            ),
            "official_status_evidence": (
                {"verified": False, "status": "unverified", "fields": ["ST", "suspension", "limit_up", "limit_down", "adjustment_factor"]}
                if provisional else {
                    "verified": True,
                    "raw_sha256": _sha256(raw_path),
                    "normalized_sha256": _sha256(normalized_path),
                    "requested_codes": [_qualified_code(code) for code in candidates],
                    "dates": [SIGNAL_DATE, EXECUTION_DATE],
                }
            ),
            "price_semantics": "pinned canonical unadjusted OHLC rows; official limits are state evidence only; no price replacement or adjustment",
            "session_dates": list(session_dates),
            "limitations": [
                "clean-paper state is synthetic and is not shadow state",
                "no broker/live orders",
                f"{len(session_dates)} sessions are represented",
                *(["official ST/suspension/limit and adjustment-factor evidence is unverified; timing/ledger only"] if provisional else []),
            ],
        }
        executable = {
            "version": "wc006-v1",
            "policy": "paper-only",
            "paper_only": True,
            "paper_replay_authorized": True,
            "broker_orders_enabled": False,
            "input_report_only": True,
            "live_execution_authorized": False,
            "authoritative_trading_state": not provisional,
            "provisional": provisional,
            "signal_date": SIGNAL_DATE,
            "execution_date": EXECUTION_DATE,
            "symbols": candidates,
            "base_targets": base_targets,
            "ohlc": ohlc_entries,
            "initial_state": {"cash": "100000", "positions": {code: "0" for code in candidates}, "lots": []},
            "execution_semantics": {"signal": "T close", "fill": "next valid session open", "price_semantics": "canonical unadjusted OHLC", "lot": "100", "t_plus_1": True},
            "costs": {"commission_rate": "0.0003", "minimum_commission": "5", "slippage": "0.0005", "stamp_duty_rate": "0.001"},
            "provenance": provenance,
        }
        output_manifest = staging / "manifest.json"
        output_manifest.write_text(json.dumps(executable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        provenance_path = staging / "provenance.json"
        provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if output.exists():
            raise BuildBlocked("output directory appeared before atomic commit")
        staging.rename(output)
        staging = None
        return output / "manifest.json"
    except BaseException:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-manifest", type=Path, default=DEFAULT_REPORT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provisional", action="store_true", help="explicit paper-only timing/ledger path without official state evidence")
    parser.add_argument("--extra-session", help="provisional-only next session to extend canonical timing input")
    args = parser.parse_args()
    try:
        result = build_manifest(args.report_manifest, args.output_dir, provisional=args.provisional, extra_session=args.extra_session)
    except BuildBlocked as exc:
        print(f"FAIL_CLOSED: {exc}")
        return 2
    print(json.dumps({"status": "PASS", "manifest": str(result), "paper_only": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
