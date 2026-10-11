#!/usr/bin/env python3
"""Single bounded AmazingData connectivity probe; emits no credentials or data."""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path


def credentials(path: Path) -> tuple[str, str]:
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


def main() -> int:
    sink = open(os.devnull, "w", encoding="utf-8")
    started = time.monotonic()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        import AmazingData as ad

    user, password = credentials(Path("/run/secrets/xysz"))
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        result = ad.login(username=user, password=password, host="101.230.159.234", port=8600)
    code = getattr(result, "error_code", None)
    bad = code not in (None, "", 0, "0")
    print(f"login_exit=0 elapsed_s={time.monotonic() - started:.3f} error_code_present={bad}", flush=True)
    if bad:
        return 2
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            base = ad.BaseData()
            market = ad.MarketData(base.get_calendar())
            frame_map = market.query_kline(
                ["sz.002802"],
                begin_date=20210104,
                end_date=20210108,
                period=ad.constant.Period.day.value,
            )
        if not isinstance(frame_map, dict) or not frame_map:
            print("kline_exit=2 rows=0 schema=[] error=empty_response", flush=True)
            return 3
        frame = next(iter(frame_map.values()))
        columns = [str(value) for value in getattr(frame, "columns", [])]
        print(f"kline_exit=0 elapsed_s={time.monotonic() - started:.3f} rows={len(frame)} schema={columns}", flush=True)
        return 0
    except Exception as exc:
        print(f"kline_exit=2 elapsed_s={time.monotonic() - started:.3f} error_type={type(exc).__name__}", flush=True)
        return 4
    finally:
        try:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                ad.logout()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"probe_exit=2 error_type={type(exc).__name__}", flush=True)
        raise SystemExit(5)
