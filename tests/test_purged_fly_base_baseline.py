import json
from pathlib import Path

import pytest

import experiments.a_share.run_purged_fly_base_baseline as baseline


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "runs/wc006_v3_2021_inputs/20260925T-purged-candidate-stream-2021-2023-v2/manifest.json"
MARKET = ROOT / "runs/wc006_purged_xysz_market/20260925T-131codes-v6/manifest.json"


def test_fixed_inputs_and_graph_contract_are_research_only():
    candidate = baseline.validate_purged_candidate(CANDIDATE)
    market, qa = baseline.validate_purged_market(MARKET, candidate)
    assert len(candidate["records"]) == 46
    assert len(market["codes"]) == 131
    assert qa["status"] == "PASS_RESEARCH_ONLY"
    assert baseline.BASE_HYPOTHESIS["neural_output_read"] is False
    assert baseline.BASE_HYPOTHESIS["fixed_multiplier"] == "1.0"


def test_candidate_hash_and_selector_provenance_fail_closed(tmp_path):
    payload = json.loads(CANDIDATE.read_text())
    payload["records"][0]["selector"]["signal_seed"] = 17
    tampered = tmp_path / "candidate.json"
    tampered.write_text(json.dumps(payload))
    with pytest.raises(baseline.PurgedBaselineBlocked, match="hash mismatch"):
        baseline.validate_purged_candidate(tampered)


def test_research_output_cannot_reuse_old_v5_namespace(tmp_path):
    with pytest.raises(baseline.PurgedBaselineBlocked, match="new phase1 namespace"):
        baseline.run_purged_baseline(CANDIDATE, MARKET, ROOT / "runs/wc006_v51_combined/reused")


def test_raw_research_loader_has_full_target_coverage():
    stream = baseline._validate_stream(CANDIDATE, MARKET)
    assert len(stream.market.raw_prices) == 131
    assert stream.market.parent_sha256 == baseline.MARKET_PARENT_SHA256
    assert stream.market.qa_sha256 == baseline.MARKET_QA_SHA256
