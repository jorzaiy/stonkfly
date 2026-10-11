"""Fail-closed fragment checks before using collected market data."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.collect_purged_xysz_2024_2025 import (
    ACTION_BEGIN, BEGIN, END, CollectionBlocked, sha, verify_fragment,
)
from experiments.a_share.run_purged_fly_base_2024_2025 import HeldoutBlocked, validate_market_policy, PARSER_PATH


def test_fragment_hash_and_scope_are_checked_before_resume(tmp_path: Path) -> None:
    child = tmp_path / "kline" / "batch_001"
    raw = child / "raw"
    raw.mkdir(parents=True)
    artifact = raw / "kline_000001_SZ.csv"
    artifact.write_text("kline_time,open,high,low,close,volume,amount\n20240102,1,1,1,1,1,1\n")
    manifest = {
        "status": "complete", "dataset": "kline", "codes": ["000001.SZ"],
        "begin_date": BEGIN, "end_date": END, "action_begin_date": ACTION_BEGIN,
        "artifacts": [{"path": artifact.name, "sha256": sha(artifact), "code": "000001.SZ"}],
    }
    (child / "manifest.json").write_text(json.dumps(manifest))
    verify_fragment(child, "kline", ["000001.SZ"])
    artifact.write_text(artifact.read_text().replace("20240102,1", "20240102,2"))
    with pytest.raises(CollectionBlocked, match="artifact_hash_mismatch"):
        verify_fragment(child, "kline", ["000001.SZ"])


def test_fragment_scope_mismatch_fails_closed(tmp_path: Path) -> None:
    child = tmp_path / "history_stock_status" / "batch_001"
    raw = child / "raw"
    raw.mkdir(parents=True)
    artifact = raw / "history_stock_status.csv"
    artifact.write_text("MARKET_CODE,TRADE_DATE\n000001.SZ,20240102\n")
    (child / "manifest.json").write_text(json.dumps({
        "status": "complete", "dataset": "history_stock_status",
        "codes": ["000001.SZ"], "begin_date": BEGIN,
        "end_date": 20240103, "action_begin_date": ACTION_BEGIN,
        "artifacts": [{"path": artifact.name, "sha256": sha(artifact)}],
    }))
    with pytest.raises(CollectionBlocked, match="scope_mismatch"):
        verify_fragment(child, "history_stock_status", ["000001.SZ"])


def test_supplemented_market_policy_binds_current_parser_and_exclusion_evidence() -> None:
    policy = {
        "ordinary_target": "普通股股东", "excluded_target": "重整管理人",
        "unknown_nonempty_target": "FAIL_CLOSED", "empty_target": "legacy_compatible",
        "parser_source_sha256": sha(PARSER_PATH), "excluded_row_count": 1,
        "excluded_source_rows": [{"row_sha256": "a" * 64}],
    }
    market = {"schema": "fly-purged-xysz-2024-2025-supplemented-market/v2", "dividend_target_policy": policy}
    qa = {"schema": "fly-purged-xysz-2024-2025-supplemented-market/v2/quality-report", "dividend_target_policy": policy}
    validate_market_policy(market, qa)
    policy["parser_source_sha256"] = "0" * 64
    with pytest.raises(HeldoutBlocked, match="parser_source_hash"):
        validate_market_policy(market, qa)
