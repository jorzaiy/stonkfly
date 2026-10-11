import json
from pathlib import Path
import pytest

from experiments.a_share.run_2021_chunk_replay import (
    audit_2021_feasibility, run_chunk_2021, FeasibilityBlockerError,
    FULL_START_DATE, FULL_END_DATE, FULL_SETTLEMENT_DATE, FULL_TOTAL_DAYS,
    FULL_SIGNAL_RECORDS, STREAM_PATH, MARKET_MANIFEST_PATH, MARKET_SHA256,
    MARKET_QA_SHA256, sha256_file, GRAPH_MANIFEST_PATH, GRAPH_SHA256,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

def test_2021_full_year_input_package_passes():
    res = audit_2021_feasibility(FULL_START_DATE, FULL_END_DATE)
    assert res["feasible"] is True
    assert res["reason"] == "PASS_INDEPENDENT_CLEAN_START_2021"
    assert res["calendar_days"] == FULL_TOTAL_DAYS
    assert res["candidate_records"] == FULL_SIGNAL_RECORDS
    assert res["neural_warmup_days"] == 0
    assert res["selector_lookback_source"] == "2020 QFQ cache; causal only"
    assert res["market_manifest_verified"] and res["market_qa_verified"]
    assert res["graph_manifest_verified"]

def test_shortened_2021_window_fails_closed():
    res = audit_2021_feasibility("2021-07-01", FULL_END_DATE)
    assert res["feasible"] is False
    assert res["reason"] == "ONLY_FULL_CALENDAR_2021_IS_PREREGISTERED"

def test_2021_candidate_stream_has_causal_lookback_and_provenance():
    payload = json.loads(STREAM_PATH.read_text())
    calendar = json.loads((STREAM_PATH.parent / "inputs/calendar.json").read_text())["dates"]
    sessions = [d for d in calendar if d.startswith("2021-")]
    assert len(sessions) == FULL_TOTAL_DAYS
    assert sessions[0] == FULL_START_DATE and sessions[-1] == FULL_END_DATE
    assert FULL_SETTLEMENT_DATE in calendar
    assert len(payload["records"]) == FULL_SIGNAL_RECORDS
    assert payload["records"][0]["signal_date"] == FULL_START_DATE
    assert payload["records"][-1]["signal_date"] == "2021-12-29"
    assert len([d for d in calendar if d < FULL_START_DATE]) >= 61
    assert payload["selector_provenance"]["overlays"]["turnover_cap"]["patch_id"] == "eligible-prev-turnover-cap-v1"
    assert payload["selector_cache_override"]["manifest_sha256"]

def test_2021_market_and_graph_pins():
    assert sha256_file(MARKET_MANIFEST_PATH) == MARKET_SHA256
    assert sha256_file(MARKET_MANIFEST_PATH.parent / "quality_report.json") == MARKET_QA_SHA256
    assert sha256_file(GRAPH_MANIFEST_PATH) == GRAPH_SHA256

def test_2021_chunk_runner_dry_run_preflight(tmp_path):
    res = run_chunk_2021(target_days=1, output_dir=tmp_path / "preflight_test", dry_run=True)
    assert res["status"] == "PREFLIGHT_DRY_RUN_PASSED"
    assert res["dry_run"] is True
    assert not (tmp_path / "preflight_test/ledger.sqlite").exists()

def test_unsupported_window_blocks(tmp_path):
    with pytest.raises(FeasibilityBlockerError):
        run_chunk_2021(target_days=1, output_dir=tmp_path / "block_test", start_date="2021-01-05", end_date=FULL_END_DATE)

def test_2022_corrected_branch_b_gate_remains_active():
    audit_file = REPO_ROOT / "audit/independent-qfq-2022/audit.json"
    audit = json.loads(audit_file.read_text())
    assert audit["status"] == "PASS_CORRECTED_BASELINE_ESTABLISHED_WITH_HISTORICAL_A_INVALIDATED"
    assert audit["historical_branch_a_performance_claim"]["status"] == "BLOCKED_NONINVARIANT"
    assert audit["corrected_branch_b_baseline"]["status"] == "PASS_TRUSTWORTHY"
    assert audit["baseline_performance_gate"]["status"] == "UNBLOCKED_CORRECTED_B_LINEAGE_ONLY"
    assert audit["jev_performance_gate"]["auto_start"] is False
