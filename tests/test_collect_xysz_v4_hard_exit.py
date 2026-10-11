from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path("/home/ubuntu/trading_agent_20260408/project/scripts/collect_xysz_v4_dataset.py")
SPEC = importlib.util.spec_from_file_location("collect_xysz_v4_dataset", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _fake_sdk(counter: dict[str, int]) -> types.SimpleNamespace:
    class LoginResult:
        error_code = 0

    class Base:
        def get_calendar(self):
            return []

    class Market:
        def __init__(self, _calendar):
            pass

        def query_kline(self, codes, **_kwargs):
            frame = pd.DataFrame(
                [{"code": codes[0], "kline_time": "2021-01-04", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "amount": 1}]
            )
            return {codes[0]: frame}

    class Info:
        pass

    class Period:
        day = types.SimpleNamespace(value=1)

    def login(**_kwargs):
        return LoginResult()

    def logout():
        counter["logout"] = counter.get("logout", 0) + 1

    return types.SimpleNamespace(
        BaseData=Base,
        InfoData=Info,
        MarketData=Market,
        constant=types.SimpleNamespace(Period=Period),
        login=login,
        logout=logout,
        __version__="test",
    )


def _args(output: Path, hard: bool) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        dataset="kline",
        output=output,
        codes=["002802.SZ"],
        begin_date=20210101,
        end_date=20240102,
        action_begin_date=20200101,
        credential_file=Path("unused"),
        host="unused",
        port=8600,
        hard_exit_after_complete=hard,
    )


def test_default_path_logs_out_and_records_manifest(tmp_path, monkeypatch):
    counter: dict[str, int] = {}
    monkeypatch.setitem(sys.modules, "AmazingData", _fake_sdk(counter))
    monkeypatch.setattr(MODULE, "credentials", lambda _path: ("u", "p"))
    output = tmp_path / "normal"
    assert MODULE.fetch(_args(output, False)) == 0
    assert counter["logout"] == 1
    assert json.loads((output / "manifest.json").read_text())["exit_mode"] == "logout"


def test_hard_exit_mode_skips_logout_but_commits_manifest(tmp_path, monkeypatch):
    counter: dict[str, int] = {}
    monkeypatch.setitem(sys.modules, "AmazingData", _fake_sdk(counter))
    monkeypatch.setattr(MODULE, "credentials", lambda _path: ("u", "p"))
    output = tmp_path / "hard"
    assert MODULE.fetch(_args(output, True)) == 0
    assert counter.get("logout", 0) == 0
    assert json.loads((output / "manifest.json").read_text())["exit_mode"] == "hard_exit_after_commit"


def test_hard_exit_requires_result_before_os_exit(tmp_path, monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(MODULE.os, "_exit", lambda rc: calls.append(rc))
    with pytest.raises(RuntimeError, match="durable_result"):
        MODULE.finalize(0, hard_exit=True, output=tmp_path / "missing")
    assert calls == []
    MODULE.atomic_json(tmp_path / "manifest.json", {"status": "complete"})
    assert MODULE.finalize(0, hard_exit=True, output=tmp_path) == 0
    assert calls == [0]


def test_default_finalize_does_not_hard_exit(tmp_path, monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(MODULE.os, "_exit", lambda rc: calls.append(rc))
    assert MODULE.finalize(2, hard_exit=False, output=tmp_path) == 2
    assert calls == []


def _status_frame(**overrides):
    row = {
        "MARKET_CODE": "002802.SZ",
        "TRADE_DATE": "2021-01-04",
        "PRECLOSE": 10.0,
        "HIGH_LIMITED": 11.0,
        "LOW_LIMITED": 9.0,
        "PRICE_HIGH_LMT_RATE": 0.1,
        "PRICE_LOW_LMT_RATE": 0.1,
        "IS_ST_SEC": 0,
        "IS_SUSP_SEC": 0,
        "IS_WD_SEC": 0,
        "IS_XR_SEC": 0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_history_stock_status_schema_and_market_state_validation():
    result = MODULE.validate_history_stock_status(_status_frame(), ["002802.SZ"], 20210101, 20240102)
    assert result["codes"] == 1
    assert result["unique_code_dates"] == 1


def test_history_stock_status_rejects_duplicate_or_missing_schema():
    duplicate = pd.concat([_status_frame(), _status_frame()], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        MODULE.validate_history_stock_status(duplicate, ["002802.SZ"], 20210101, 20240102)
    with pytest.raises(ValueError, match="schema_missing"):
        MODULE.validate_history_stock_status(_status_frame().drop(columns=["IS_SUSP_SEC"]), ["002802.SZ"], 20210101, 20240102)


def test_history_stock_status_coerces_sdk_dict_of_frames():
    frame = _status_frame().drop(columns=["MARKET_CODE"]).set_index("TRADE_DATE")
    result = MODULE.coerce_history_stock_status({"002802.SZ": frame}, ["002802.SZ"])
    assert result.loc[0, "MARKET_CODE"] == "002802.SZ"
    assert result.loc[0, "TRADE_DATE"] == "2021-01-04"


def test_history_stock_status_accepts_explicit_no_limit_sentinel():
    result = MODULE.validate_history_stock_status(
        _status_frame(HIGH_LIMITED=0, LOW_LIMITED=0, PRICE_HIGH_LMT_RATE=999, PRICE_LOW_LMT_RATE=999),
        ["002802.SZ"], 20210101, 20240102,
    )
    assert result["no_price_limit_rows"] == 1


def test_history_stock_status_rejects_dict_key_code_mismatch():
    frame = _status_frame().set_index("TRADE_DATE")
    with pytest.raises(ValueError, match="key_code_mismatch"):
        MODULE.coerce_history_stock_status({"002802.SZ": frame.assign(MARKET_CODE="002900.SZ")}, ["002802.SZ"])
