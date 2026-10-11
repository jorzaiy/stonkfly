import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from stonkfly.jev_ab import ABConfig, compare_arms, make_plan, performance_gate, regression_compare, validate_feature_payload

REPO = Path(__file__).resolve().parents[1]
REPLAY = REPO / "audit/independent-qfq-2022/counterfactual-replay"
AUDIT_JSON = REPO / "audit/independent-qfq-2022/audit.json"

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _corrected_fixture(tmp_path):
    """Use real replay files; only the small audit JSON is materialized in tmp_path."""
    audit = copy.deepcopy(json.loads(AUDIT_JSON.read_text(encoding="utf-8")))
    b = audit.setdefault("corrected_branch_b_baseline", {})
    audit["status"] = "PASS_CORRECTED_BASELINE_ESTABLISHED_WITH_HISTORICAL_A_INVALIDATED"
    b.update({
        "status": "PASS_TRUSTWORTHY",
        "lineage_id": "corrected-branch-b-2022-v5.1",
        "manifest_path": "audit/independent-qfq-2022/counterfactual-replay/manifest.json",
        "manifest_sha256": _sha(REPLAY / "manifest.json"),
        "comparison_path": "audit/independent-qfq-2022/counterfactual-replay/comparison.json",
        "comparison_sha256": _sha(REPLAY / "comparison.json"),
        "candidate_stream_manifest_sha256": _sha(REPLAY / "candidate-stream-b/manifest.json"),
        "initial_brain_checkpoint": "audit/independent-qfq-2022/counterfactual-replay/branch-b/brain-clean.npz",
        "initial_brain_sha256": _sha(REPLAY / "branch-b/brain-clean.npz"),
        "direct_quant_baseline": "audit/independent-qfq-2022/counterfactual-replay/branch-b/paired_base.sqlite",
        "direct_quant_baseline_sha256": _sha(REPLAY / "branch-b/paired_base.sqlite"),
        "branch_provenance_path": "audit/independent-qfq-2022/counterfactual-replay/branch-b/provenance.json",
        "branch_provenance_sha256": _sha(REPLAY / "branch-b/provenance.json"),
        "market_manifest": "runs/wc006_v5_market_inputs/manual-bridge/authorized-v5-development-20260922T-right-issue-overlay-2022/manifest.json",
        "market_manifest_sha256": _sha(REPO / "runs/wc006_v5_market_inputs/manual-bridge/authorized-v5-development-20260922T-right-issue-overlay-2022/manifest.json"),
        "graph_manifest_sha256": "70b253b10eaecd2492ced3b143b3b391e10b49b8a62ced7a0ec105930b68cd7f",
        "candidate_cache_002703_sha256": "61bef8f57dad6b421f8116bfc721c6e11c0aa94993486b2c9eb60363f1193564",
        "config_seed": 17, "plasticity_mode": "frozen", "paper_only": True,
    })
    audit["jev_performance_gate"] = {"status": "UNBLOCKED_CORRECTED_B_LINEAGE_ONLY", "auto_start": False}
    audit_path = tmp_path / "corrected-audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    config = {
        "lineage_id": "corrected-branch-b-2022-v5.1",
        "paper_only": True, "broker_orders_enabled": False, "live_execution_authorized": False,
        "baseline_plasticity_mode": "frozen", "graph_manifest_sha256": b["graph_manifest_sha256"],
        "candidate_cache_002703_sha256": b["candidate_cache_002703_sha256"],
    }
    return audit_path, ABConfig(
        output_root=str(tmp_path / "valid-ab"), baseline_audit=str(audit_path),
        candidate_stream=str(REPLAY / "candidate-stream-b/manifest.json"),
        market_data=str(REPO / b["market_manifest"]), feature_dataset=None,
        feature_names=tuple(f"f{i}" for i in range(8)), mapping_hash="m", normalization_hash="n",
        brain_checkpoint=str(REPLAY / "branch-b/brain-clean.npz"), seed=17,
        direct_quant_baseline=str(REPLAY / "branch-b/paired_base.sqlite"),
        baseline_lineage_id="corrected-branch-b-2022-v5.1", config=config,
    )


def cfg(tmp_path, **overrides):
    args = dict(output_root=str(tmp_path / "ab"), baseline_audit=None, candidate_stream=None, market_data=None, feature_dataset=None, feature_names=tuple(f"f{i}" for i in range(8)), mapping_hash="m", normalization_hash="n")
    args.update(overrides)
    return ABConfig(**args)


def test_plan_has_four_independent_arms_and_hashes(tmp_path):
    plan = make_plan(cfg(tmp_path))
    assert set(plan["arms"]) == {"A", "B", "C", "D"}
    assert len({v["output_root"] for v in plan["arms"].values()}) == 4
    assert plan["performance_gate"]["allowed"] is False
    assert Path(tmp_path / "ab" / "plan.json").exists()


def test_performance_is_blocked_without_exact_pass(tmp_path):
    plan = make_plan(cfg(tmp_path))
    with pytest.raises(RuntimeError, match="blocked"):
        performance_gate(plan)


def test_feature_payload_rejects_leakage():
    with pytest.raises(ValueError, match="forbidden"):
        validate_feature_payload({"broad_market_risk": 0.2, "future_return_1d": 0.4})


def event(event_id, date, action, *, difference="0.5", gate=1, requested="1", effective="1", vetoes=None, stimulus="none", changed=0):
    return {
        "event_id": event_id, "date": date,
        "visual": {"frame_sha256": "frame"},
        "decoder": {"action": action, "requested_multiplier": requested, "effective_multiplier": effective},
        "neural": {"left_hz": "1.0", "right_hz": "1.5", "difference_hz": difference, "gate_spikes": gate, "memory": {"changed_edges": changed, "mean_efficacy": "1.0", "minimum_efficacy": "0.9"}},
        "reward": {"portfolio_return": "0.01", "stimulus": stimulus},
        "execution": {"turnover": "1", "fees": "0.1", "vetoes": vetoes, "fills": []},
    }


def test_compare_aligns_by_date_not_provenance_event_id_and_reports_metrics():
    a1 = event("A-001", "2022-01-01", "HOLD", gate=0, stimulus="reward")
    a2 = event("A-002", "2022-01-02", "BUY", changed=2)
    # Reversed row order and different provenance-bearing IDs must still align.
    b2 = event("B-002", "2022-01-02", "BUY", effective="0.5", vetoes=["limit"], changed=3)
    b1 = event("B-001", "2022-01-01", "BUY", difference="0.7", gate=2, stimulus="aversive")
    report = compare_arms({"A": [a1, a2], "B": [b2, b1]})
    pair = report["pairs"]["B-A"]
    assert pair["alignment"]["status"] == "ALIGNED"
    assert pair["action_disagreement_count"] == 1
    assert pair["disagreement_dates"] == ["2022-01-01"]
    assert pair["disagreement_events"][0]["left_event_id"] == "B-001"
    assert pair["disagreement_events"][0]["right_event_id"] == "A-001"
    metrics = report["arms"]["B"]
    assert metrics["left_right_difference_hz"]["p50"] == "0.600"
    assert metrics["gate_activation"]["active_rate"] == 1.0
    assert metrics["multiplier_path_summary"]["effective"]["min"] == "0.5"
    assert metrics["veto_rate"] == 0.5
    assert metrics["reward_count"] == 0 and metrics["aversive_count"] == 1
    assert metrics["changed_plastic_edges"]["max"] == "3"
    assert metrics["memory_efficacy"]["max"]["available"] is False


def test_compare_fails_closed_for_misaligned_dates():
    report = compare_arms({"A": [event("A-1", "2022-01-01", "HOLD")], "B": [event("B-2", "2022-01-02", "BUY")]})
    pair = report["pairs"]["B-A"]
    assert pair["alignment"]["status"] == "MISALIGNED"
    assert pair["action_disagreement_count"] is None


def test_null_regression_aligns_by_date_despite_different_event_ids():
    original = [event("original-1", "2022-01-01", "HOLD")]
    null = [event("null-1", "2022-01-01", "HOLD")]
    assert regression_compare(original, null)["status"] == "PASS"


def test_null_regression_detects_mismatch():
    one = [event("original-1", "2022-01-01", "HOLD")]
    two = [event("null-1", "2022-01-01", "BUY")]
    assert regression_compare(one, two)["status"] == "FAIL"


def test_real_corrected_branch_b_provenance_is_allowed_and_nonautomatic(tmp_path):
    _, config = _corrected_fixture(tmp_path)
    plan = make_plan(config)
    assert plan["performance_gate"]["allowed"] is True
    assert plan["performance_gate"]["status"] == "UNBLOCKED_CORRECTED_B_LINEAGE_ONLY"
    assert plan["performance_gate"]["auto_start"] is False
    assert plan["performance_gate"]["paper_only"] is True
    performance_gate(plan)


@pytest.mark.parametrize("mutator", [
    lambda a, c: a.update(status="PASS"),
    lambda a, c: a["historical_branch_a_performance_claim"].update(status="PASS"),
    lambda a, c: a["corrected_branch_b_baseline"].pop("lineage_id", None),
    lambda a, c: a["corrected_branch_b_baseline"].update(config_seed=18),
    lambda a, c: c.config.update(paper_only=False),
    lambda a, c: c.config.update(graph_manifest_sha256="bad"),
    lambda a, c: c.config.update(candidate_cache_002703_sha256="bad"),
    lambda a, c: c.config.update(baseline_plasticity_mode="enabled"),
    lambda a, c: a["corrected_branch_b_baseline"].update(candidate_stream_manifest_sha256="bad"),
    lambda a, c: a["corrected_branch_b_baseline"].update(market_manifest_sha256="bad"),
    lambda a, c: a["corrected_branch_b_baseline"].update(initial_brain_sha256="bad"),
    lambda a, c: a["corrected_branch_b_baseline"].update(direct_quant_baseline_sha256="bad"),
])
def test_corrected_branch_b_rejects_unknown_or_tampered_provenance(tmp_path, mutator):
    audit_path, config = _corrected_fixture(tmp_path)
    audit = json.loads(audit_path.read_text())
    mutator(audit, config)
    audit_path.write_text(json.dumps(audit))
    plan = make_plan(config)
    assert plan["performance_gate"]["allowed"] is False
    assert plan["performance_gate"]["auto_start"] is False
    with pytest.raises(RuntimeError, match="Jev performance run blocked"):
        performance_gate(plan)


def test_corrected_branch_b_rejects_declared_path_mismatch(tmp_path):
    _, config = _corrected_fixture(tmp_path)
    mismatched = replace(config, candidate_stream=config.market_data)
    plan = make_plan(mismatched)
    assert plan["performance_gate"]["allowed"] is False
    assert "candidate_stream" in plan["performance_gate"]["reason"]
