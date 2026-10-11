#!/usr/bin/env python3
"""Bounded, report-only AmazingData probe for WC006 v4 price/action semantics.

This script is intended to run inside the pinned ``amazingdata-x86:1.1.9``
container.  It never writes to the trading repository: the output directory is
created under a caller-provided Stonkfly run root and committed atomically only
after all requested datasets validate.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _credentials(path: Path) -> tuple[str, str]:
    """Read credentials in the existing safe formats without persisting them."""
    text = path.read_text(encoding="utf-8").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        user = str(payload.get("AD_USERNAME") or payload.get("username") or payload.get("user") or "").strip()
        password = str(payload.get("AD_PASSWORD") or payload.get("password") or "").strip()
    else:
        values: dict[str, str] = {}
        plain: list[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("'\"")
            else:
                plain.append(line)
        user = values.get("AD_USERNAME") or values.get("username") or values.get("user") or ""
        password = values.get("AD_PASSWORD") or values.get("password") or ""
        if not user and not password and len(plain) == 2:
            user, password = plain
    if not user or not password:
        raise RuntimeError("credential_format_incomplete")
    return user, password


@contextlib.contextmanager
def _silence_sdk():
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield


def _code6(value: object) -> str:
    raw = str(value).upper().replace("SH.", "").replace("SZ.", "").replace(".SH", "").replace(".SZ", "")
    return raw.zfill(6)


def _serializable(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_frame(frame: Any, path: Path) -> dict[str, object]:
    if frame is None:
        raise RuntimeError(f"empty_response:{path.name}")
    if not hasattr(frame, "to_csv"):
        raise RuntimeError(f"non_tabular_response:{path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=True)
    fields = [str(x) for x in getattr(frame, "columns", [])]
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "rows": int(len(frame)),
        "fields": fields,
    }


def _write_kline(value: Any, directory: Path, codes: list[str]) -> dict[str, dict[str, object]]:
    if not isinstance(value, dict):
        raise RuntimeError("kline_response_not_dict")
    result: dict[str, dict[str, object]] = {}
    for requested in codes:
        match = next((key for key in value if _code6(key) == _code6(requested)), None)
        if match is None:
            raise RuntimeError(f"kline_missing_code:{_code6(requested)}")
        frame = value[match]
        result[_code6(requested)] = _write_frame(frame, directory / f"kline_{_code6(requested)}.csv")
    return result


def _date_text(value: object) -> str:
    return str(value).strip()[:10].replace("/", "-")


def _event_summary(frame: Any, date_fields: tuple[str, ...], begin: str, end: str) -> dict[str, object]:
    fields = {str(x) for x in getattr(frame, "columns", [])}
    present = [name for name in date_fields if name in fields]
    dates: list[str] = []
    for name in present:
        for value in frame[name].tolist():
            text = _date_text(value)
            if len(text) >= 10 and begin <= text <= end and text not in {"NaT", "nan", "None", ""}:
                dates.append(text)
    return {"date_fields": present, "events_in_window": len(dates), "event_dates": sorted(set(dates))}


def run_probe(*, output: Path, codes: list[str], begin_date: int, end_date: int, action_begin: int, credential_file: Path, host: str, port: int) -> dict[str, object]:
    if output.exists():
        raise RuntimeError("output_must_be_new")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        import AmazingData as ad
        import pandas as pd  # noqa: F401  # validates the container data path

        clean_codes = [str(x).strip().upper() for x in codes]
        if len(clean_codes) != len(set(clean_codes)) or not clean_codes:
            raise RuntimeError("invalid_codes")
        user, password = _credentials(credential_file)
        cache_dir = staging / "cache"
        cache_dir.mkdir()
        with _silence_sdk():
            login_result = ad.login(username=user, password=password, host=host, port=port)
        if getattr(login_result, "error_code", "0") not in (None, "", "0", 0):
            raise RuntimeError("login_failed")
        try:
            base = ad.BaseData()
            info = ad.InfoData()
            calendar = base.get_calendar()
            market = ad.MarketData(calendar)
            with _silence_sdk():
                kline = market.query_kline(clean_codes, begin_date=begin_date, end_date=end_date, period=ad.constant.Period.day.value)
                adj_factor = base.get_adj_factor(clean_codes, local_path=str(cache_dir), is_local=False)
                dividend = info.get_dividend(clean_codes, local_path=str(cache_dir), is_local=False, begin_date=action_begin, end_date=end_date)
                right_issue = info.get_right_issue(clean_codes, local_path=str(cache_dir), is_local=False, begin_date=action_begin, end_date=end_date)
                equity_structure = info.get_equity_structure(clean_codes, local_path=str(cache_dir), is_local=False, begin_date=action_begin, end_date=end_date)
        finally:
            try:
                with _silence_sdk():
                    ad.logout()
            except TypeError:
                with _silence_sdk():
                    ad.logout(user)

        artifacts: dict[str, object] = {}
        artifacts["kline"] = _write_kline(kline, staging / "raw", clean_codes)
        artifacts["adj_factor"] = _write_frame(adj_factor, staging / "raw" / "adj_factor.csv")
        artifacts["dividend"] = _write_frame(dividend, staging / "raw" / "dividend.csv")
        artifacts["right_issue"] = _write_frame(right_issue, staging / "raw" / "right_issue.csv")
        artifacts["equity_structure"] = _write_frame(equity_structure, staging / "raw" / "equity_structure.csv")
        dividend_summary = _event_summary(dividend, ("DATE_EX", "DATE_EQY_RECORD", "DATE_DVD_PAYOUT"), str(begin_date), str(end_date))
        right_summary = _event_summary(right_issue, ("EX_DIVIDEND_DATE", "LISTED_DATE", "EXECUTE_DATE", "RESULT_DATE"), str(begin_date), str(end_date))
        equity_summary = _event_summary(equity_structure, ("EX_CHANGE_DATE", "CHANGE_DATE"), str(begin_date), str(end_date))
        files = {}
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                files[str(path.relative_to(staging))] = _sha256(path)
        manifest = {
            "schema": "wc006-xysz-v4-action-probe/v1",
            "status": "complete",
            "provider": "China Galaxy Securities AmazingData",
            "sdk_version": str(getattr(ad, "__version__", "unknown")),
            "credential_values_persisted": False,
            "codes": clean_codes,
            "kline_range": [begin_date, end_date],
            "action_query_range": [action_begin, end_date],
            "artifacts": artifacts,
            "event_summary": {"dividend": dividend_summary, "right_issue": right_summary, "equity_structure": equity_summary},
            "cache_files_sha256": files,
            "limitations": ["retrieval-time provider snapshot; historical revision/deletion semantics are not proven", "probe is report-only and does not authorize execution"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        (staging / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        staging.rename(output)
        return {"status": "complete", "output": str(output), "manifest_sha256": manifest["manifest_sha256"], "event_summary": manifest["event_summary"]}
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(f"probe_failed:{type(exc).__name__}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codes", nargs=3, required=True)
    parser.add_argument("--begin-date", type=int, default=20210101)
    parser.add_argument("--end-date", type=int, default=20240102)
    parser.add_argument("--action-begin", type=int, default=20200101)
    parser.add_argument("--credential-file", type=Path, default=Path("/run/secrets/xysz"))
    parser.add_argument("--host", default="101.230.159.234")
    parser.add_argument("--port", type=int, default=8600)
    args = parser.parse_args()
    try:
        result = run_probe(output=args.output, codes=args.codes, begin_date=args.begin_date, end_date=args.end_date, action_begin=args.action_begin, credential_file=args.credential_file, host=args.host, port=args.port)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
