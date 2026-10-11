import hashlib
import json
import shutil
from pathlib import Path

import pytest

from experiments.a_share.build_wc006_strict_smoke import (
    BuildBlocked,
    DEFAULT_REPORT_MANIFEST,
    _normalise_status,
    build_manifest,
)
from experiments.a_share.replay_wc006 import MarketStateError, load_manifest


def _status_rows():
    report = json.loads(DEFAULT_REPORT_MANIFEST.read_text(encoding="utf-8"))
    rows = []
    for code in report["candidate_set"]["codes"]:
        for day in ("2026-08-31", "2026-09-01"):
            rows.append(
                {
                    "MARKET_CODE": f"{code}.SZ",
                    "TRADE_DATE": day,
                    "HIGH_LIMITED": "1000",
                    "LOW_LIMITED": "0.1",
                    "IS_ST_SEC": "0",
                    "IS_SUSP_SEC": "0",
                    "IS_WD_SEC": "0",
                    "IS_XR_SEC": "0",
                }
            )
    return rows


def _status_client(rows):
    def client(codes, cache_path):
        assert codes == [f"{code}.SZ" for code in json.loads(DEFAULT_REPORT_MANIFEST.read_text())["candidate_set"]["codes"]]
        assert not cache_path.exists()
        return rows

    return client


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def hermetic_report_manifest(tmp_path):
    """Copy immutable smoke inputs so tests do not trust mutable repo files."""
    source = json.loads(DEFAULT_REPORT_MANIFEST.read_text(encoding="utf-8"))
    root = tmp_path / "report-repo"
    root.mkdir()
    manifest_path = root / "manifest.json"
    source["repository_root"] = "."
    trading_root = DEFAULT_REPORT_MANIFEST.parents[5]
    for item in source["source_files"].values():
        original = trading_root / item["path"]
        destination = root / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
        item["sha256"] = _sha256(destination)
    for item in source["canonical_cache"]["files"]:
        original = trading_root / item["path"]
        destination = root / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
        item["sha256"] = _sha256(destination)
        if destination.name.endswith("_2026-09-01.csv"):
            extra_original = original.with_name(original.name.replace("_2026-09-01.csv", "_2026-09-02.csv"))
            extra_destination = destination.with_name(destination.name.replace("_2026-09-01.csv", "_2026-09-02.csv"))
            shutil.copyfile(extra_original, extra_destination)
    manifest_path.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def test_builder_emits_hash_bound_clean_paper_wc006_manifest(tmp_path, hermetic_report_manifest):
    output = tmp_path / "inputs" / "run-unique"
    manifest_path = build_manifest(
        hermetic_report_manifest,
        output,
        status_client=_status_client(_status_rows()),
        independent_root=tmp_path,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["version"] == "wc006-v1"
    assert payload["paper_only"] is True
    assert payload["paper_replay_authorized"] is True
    assert payload["broker_orders_enabled"] is False
    assert payload["live_execution_authorized"] is False
    assert payload["symbols"] == json.loads(DEFAULT_REPORT_MANIFEST.read_text())["candidate_set"]["codes"]
    assert set(payload["base_targets"]) == {"000603", "001331", "002536"}
    assert payload["initial_state"] == {
        "cash": "100000",
        "positions": {code: "0" for code in payload["symbols"]},
        "lots": [],
    }
    assert len(payload["ohlc"]) == 6
    assert payload["provenance"]["paper_only"] is True
    assert payload["provenance"]["clean_paper_initial_state"]["cash"] == "100000"
    loaded = load_manifest(manifest_path, strict_market_state=True)
    assert loaded.symbols == tuple(payload["symbols"])
    assert set(loaded.base_targets) == {"000603", "001331", "002536"}
    assert loaded.initial_cash == 100000
    assert len(payload["provenance"]["derived_from"]["pinned_source_files"]) == 12
    for code, item in payload["ohlc"].items():
        path = output / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        assert path.read_text().splitlines()[0].endswith("official_limit_down")
    assert json.loads((output / "evidence/history_stock_status.normalized.json").read_text())


def test_provisional_manifest_is_non_authoritative_and_strict_rejected(tmp_path, hermetic_report_manifest):
    output = tmp_path / "inputs" / "provisional"
    manifest_path = build_manifest(
        hermetic_report_manifest,
        output,
        independent_root=tmp_path,
        provisional=True,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["paper_replay_authorized"] is True
    assert payload["broker_orders_enabled"] is False
    assert payload["live_execution_authorized"] is False
    assert payload["authoritative_trading_state"] is False
    assert payload["provisional"] is True
    assert payload["provenance"]["official_status_evidence"]["verified"] is False
    assert "official ST/suspension/limit and adjustment-factor evidence is unverified" in payload["provenance"]["limitations"][-1]
    for item in payload["ohlc"].values():
        header = (output / item["path"]).read_text(encoding="utf-8").splitlines()[0]
        assert "official_status" not in header
        assert "official_limit_up" not in header
        assert "official_limit_down" not in header
    assert not (output / "evidence/history_stock_status.raw.json").exists()
    assert not (output / "evidence/history_stock_status.normalized.json").exists()
    loaded = load_manifest(manifest_path, strict_market_state=False)
    assert loaded.symbols == tuple(payload["symbols"])
    with pytest.raises(MarketStateError):
        load_manifest(manifest_path, strict_market_state=True)


def test_default_without_official_status_remains_fail_closed(tmp_path, hermetic_report_manifest):
    output = tmp_path / "inputs" / "default-strict"
    with pytest.raises(BuildBlocked, match="credentials missing"):
        build_manifest(hermetic_report_manifest, output, independent_root=tmp_path)
    assert not output.exists()

def test_extra_session_is_same_tag_hash_bound_and_still_non_strict(tmp_path, hermetic_report_manifest):
    output = tmp_path / "inputs" / "provisional-three-session"
    manifest_path = build_manifest(
        hermetic_report_manifest,
        output,
        independent_root=tmp_path,
        provisional=True,
        extra_session="2026-09-02",
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["session_dates"] == ["2026-08-31", "2026-09-01", "2026-09-02"]
    extra = payload["provenance"]["derived_from"]["extra_canonical_source_files"]
    assert set(extra) == set(payload["symbols"])
    assert {item["exact_tag"] for item in extra.values()} == {"5y_2021-01-01_2026-09-02"}
    for code, item in extra.items():
        source = Path("/home/ubuntu/trading_agent_20260408") / item["path"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == item["sha256"]
        assert len((output / payload["ohlc"][code]["path"]).read_text().splitlines()) == 4
    loaded = load_manifest(manifest_path, strict_market_state=False)
    assert len(loaded.data[loaded.symbols[0]].rows) == 3
    with pytest.raises(Exception, match="strict mode requires official|incomplete official"):
        load_manifest(manifest_path, strict_market_state=True)


def test_extra_session_is_provisional_only(tmp_path, hermetic_report_manifest):
    output = tmp_path / "inputs" / "strict-extra-rejected"
    with pytest.raises(BuildBlocked, match="provisional-only"):
        build_manifest(hermetic_report_manifest, output, independent_root=tmp_path, extra_session="2026-09-02")
    assert not output.exists()


def test_incomplete_official_evidence_fails_closed_without_manifest(tmp_path, hermetic_report_manifest):
    rows = _status_rows()[:-1]
    output = tmp_path / "inputs" / "incomplete"
    with pytest.raises(BuildBlocked, match="coverage incomplete"):
        build_manifest(
            hermetic_report_manifest,
            output,
            status_client=_status_client(rows),
            independent_root=tmp_path,
        )
    assert not output.exists()


def test_status_and_st_flags_keep_distinct_official_semantics():
    rows = _status_rows()
    rows[0]["IS_ST_SEC"] = "1"
    rows[1]["IS_SUSP_SEC"] = "1"
    codes = json.loads(DEFAULT_REPORT_MANIFEST.read_text())["candidate_set"]["codes"]
    normalized = _normalise_status(rows, codes)
    assert normalized[(codes[0], "2026-08-31")]["official_status"] == "ST"
    assert normalized[(codes[0], "2026-09-01")]["official_status"] == "SUSPENDED"


def test_api_query_exception_cleans_staging_and_target(tmp_path, hermetic_report_manifest):
    output = tmp_path / "inputs" / "query-fails"

    def query_fails(_codes, _cache_path):
        raise RuntimeError("network unavailable")

    with pytest.raises(BuildBlocked, match="official status query failed"):
        build_manifest(
            hermetic_report_manifest,
            output,
            status_client=query_fails,
            independent_root=tmp_path,
        )
    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}.staging-*"))


def test_output_cannot_escape_isolated_stonkfly_root(tmp_path, hermetic_report_manifest):
    trading_repo_output = Path("/home/ubuntu/trading_agent_20260408/project/data/watch/wc006-strict-smoke-test")
    with pytest.raises(BuildBlocked, match="isolated Stonkfly root"):
        build_manifest(
            hermetic_report_manifest,
            trading_repo_output,
            status_client=_status_client(_status_rows()),
        )
    assert not trading_repo_output.exists()


def test_missing_ad_credentials_fails_closed(monkeypatch, tmp_path, hermetic_report_manifest):
    for name in ("AD_USERNAME", "AD_PASSWORD", "AD_HOST", "AD_PORT"):
        monkeypatch.delenv(name, raising=False)
    output = tmp_path / "inputs" / "no-credentials"
    with pytest.raises(BuildBlocked, match="credentials missing"):
        build_manifest(hermetic_report_manifest, output, independent_root=tmp_path)
    assert not output.exists()
