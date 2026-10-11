"""Tests for independent QFQ audit scripts, safety constraints, and fail-closed paths."""

import importlib.util
import json
import tempfile
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pandas as pd
import pytest
from stonkfly.jev_ab import ABConfig, make_plan, performance_gate

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = REPO_ROOT / "audit" / "independent-qfq-2022"
if str(AUDIT_DIR) not in sys.path:
    sys.path.insert(0, str(AUDIT_DIR))



def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_qmt_export_safety_and_static_constraints():
    script_path = AUDIT_DIR / "export_qmt_qfq.py"
    assert script_path.is_file()
    text = script_path.read_text(encoding="utf-8")

    # 1. No trading APIs imported
    assert "import xttrader" not in text
    assert "from xtquant import xttrader" not in text
    # 2. Strict no-suspension-fill
    assert "fill_data=False" in text
    # 3. Fixed codes and window
    assert "002703.SZ" in text
    assert "000721.SZ" in text
    assert "20210101" in text
    assert "20260403" in text


def test_qmt_export_fail_closed_on_empty_raw_or_qfq():
    script_path = AUDIT_DIR / "export_qmt_qfq.py"
    mod = load_module("export_qmt_qfq", script_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Mock xtdata returning empty result
        mock_xtdata = MagicMock()
        mock_xtdata.get_market_data_ex.return_value = {}

        exit_code = mod.export_market_data(
            output_dir=tmp_path,
            target_codes=["002703.SZ"],
            xtdata_module=mock_xtdata,
        )

        assert exit_code == 1  # Must fail closed with non-zero exit

        manifest_file = tmp_path / "manifest.json"
        assert manifest_file.is_file()
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        assert manifest["status"] == "INCOMPLETE"
        assert len(manifest["missing_required_data"]) > 0


def test_qmt_export_success_path_and_factor_error_recording():
    script_path = AUDIT_DIR / "export_qmt_qfq.py"
    mod = load_module("export_qmt_qfq", script_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Construct dummy market data DataFrame
        dates = ["2022-01-04", "2022-01-05"]
        df_dummy = pd.DataFrame({
            "time": dates,
            "open": [10.0, 10.5],
            "high": [11.0, 11.2],
            "low": [9.8, 10.1],
            "close": [10.5, 10.8],
            "volume": [1000.0, 1200.0],
            "amount": [10500.0, 12960.0],
        })

        mock_xtdata = MagicMock()
        mock_xtdata.get_market_data_ex.return_value = {"002703.SZ": df_dummy}
        # Mock factor call raising exception
        mock_xtdata.get_divid_factors.side_effect = RuntimeError("Broker factor subscription unavailable")

        exit_code = mod.export_market_data(
            output_dir=tmp_path,
            target_codes=["002703.SZ"],
            xtdata_module=mock_xtdata,
        )

        # Raw & QFQ succeeded, so exit is 0 and status is COMPLETE
        assert exit_code == 0

        manifest_file = tmp_path / "manifest.json"
        assert manifest_file.is_file()
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        assert manifest["status"] == "COMPLETE"
        assert manifest["missing_required_data"] == []

        # Check raw CSV metadata recorded
        raw_meta = manifest["artifacts"]["raw/002703_SZ_none_raw.csv"]
        assert raw_meta["rows"] == 2
        assert raw_meta["date_range"] == ["2022-01-04", "2022-01-05"]
        assert "close" in raw_meta["fields"]
        assert len(raw_meta["sha256"]) == 64

        # Check factor failure was explicitly recorded without silent omission
        factor_meta = manifest["artifacts"]["factors/002703_SZ_divid_factors.csv"]
        assert factor_meta["status"] == "error"
        assert "Broker factor subscription unavailable" in factor_meta["error"]


def test_audit_json_semantics_and_boundaries():
    audit_json_path = AUDIT_DIR / "audit.json"
    assert audit_json_path.is_file()
    data = json.loads(audit_json_path.read_text(encoding="utf-8"))

    # Baseline gate is PASS_CORRECTED_BASELINE_ESTABLISHED_WITH_HISTORICAL_A_INVALIDATED pending Branch B completion
    assert data["status"] == "PASS_CORRECTED_BASELINE_ESTABLISHED_WITH_HISTORICAL_A_INVALIDATED"
    assert data["baseline_performance_gate"]["status"] == "UNBLOCKED_CORRECTED_B_LINEAGE_ONLY"
    assert data["baseline_performance_gate"]["action"] == "ALLOW_FUTURE_PAPER_AB_WITH_CORRECTED_B_LINEAGE_ONLY"

    # Subchecks explicitly separate data integrity, selector sensitivity, and Fly replay
    assert data["subchecks"]["qfq_price_and_factor_data_subcheck"] == "PASS"
    assert data["subchecks"]["selector_and_direct_quant_base_subcheck"] == "PASS"
    assert data["subchecks"]["fly_observation_full_year_sensitivity_subcheck"] == "PASS"

    # Semantic discrepancy check: documented potential vs actual impact
    disc = data["provenance_investigation"]["existing_cache"]["documented_semantic_discrepancy"]
    assert "audit_concern" in disc
    assert "potential_impact" in disc
    assert "actual_impact_determination" in disc

    # Usage boundary check: raw execution & accounting did not use qfq
    exec_info = data["provenance_investigation"]["execution_and_accounting_inputs"]
    assert "xysz raw" in exec_info["behavior"]
    assert "rejected by risk guards" in exec_info["behavior"]


def test_xysz_audit_returns_and_suspension_verification():
    from audit_xysz_qfq import run_audit

    payload, md_text = run_audit()
    assert payload["status"] == "PASS_CORRECTED_BASELINE_ESTABLISHED_WITH_HISTORICAL_A_INVALIDATED"
    assert payload["historical_branch_a_performance_claim"]["status"] == "BLOCKED_NONINVARIANT"
    assert payload["subchecks"]["qfq_price_and_factor_data_subcheck"] == "PASS"
    assert payload["subchecks"]["selector_and_direct_quant_base_subcheck"] == "PASS"
    assert payload["subchecks"]["fly_observation_full_year_sensitivity_subcheck"] == "PASS"
    assert payload["baseline_performance_gate"]["status"] == "UNBLOCKED_CORRECTED_B_LINEAGE_ONLY"

    # Verify both codes for Tier 1: common day price and factor parity
    for code in ("002703.SZ", "000721.SZ"):
        target = payload["targets"][code]
        assert target["adjustment_factor_analysis"]["split_dividend_events_in_2021_2023"] == 0
        ret_diffs = target["adjustment_factor_analysis"]["common_day_return_diffs"]
        for k in (1, 5, 20, 60):
            assert ret_diffs[f"ret_{k}d_max_abs_diff"] < 1e-12

    # Verify Tier 2: 002703 suspension feature divergence on resumption (2022-07-06)
    res_2703 = payload["targets"]["002703.SZ"]
    key_comps = {r["date"]: r for r in res_2703["feature_sensitivity_analysis"]["key_date_comparisons"]}
    row_0706 = key_comps["2022-07-06"]
    assert row_0706["diff_mom5"] is not None
    assert abs(row_0706["diff_mom5"]) > 0.30
    assert row_0706["seq_a_mom5"] < 0.0
    assert row_0706["seq_b_mom5"] > 0.20

    # Verify 5 suspension days confirmed by status table
    susp_days = res_2703["calendar_and_suspension"]["suspension_details"]
    assert len(susp_days) == 5
    for s in susp_days:
        assert s["status_is_susp"] == 1
        assert s["cache_volume_isnan"] is True


def test_jev_ab_performance_gate_rejects_legacy_simple_pass(tmp_path):
    mock_audit = tmp_path / "mock_pass_audit.json"
    mock_audit.write_text(json.dumps({"status": "PASS"}))

    plan = make_plan(ABConfig(
        output_root=str(tmp_path / "ab"), baseline_audit=str(mock_audit),
        candidate_stream=None, market_data=None, feature_dataset=None,
        feature_names=tuple(f"f{i}" for i in range(8)), mapping_hash="m", normalization_hash="n",
    ))
    assert plan["baseline_audit"]["status"] == "PASS"
    assert plan["performance_gate"]["allowed"] is False
    with pytest.raises(RuntimeError, match="Jev performance run blocked"):
        performance_gate(plan)


def test_jev_ab_performance_gate_strictly_fails_closed_when_blocked(tmp_path):
    import pytest
    from stonkfly.jev_ab import ABConfig, make_plan, performance_gate

    audit_json_path = AUDIT_DIR / "audit.json"
    assert audit_json_path.is_file()

    plan = make_plan(
        ABConfig(
            output_root=str(tmp_path / "ab_blocked"),
            baseline_audit=str(audit_json_path),
            candidate_stream=None,
            market_data=None,
            feature_dataset=None,
            feature_names=tuple(f"f{i}" for i in range(8)),
            mapping_hash="m",
            normalization_hash="n",
        )
    )
    assert plan["baseline_audit"]["status"] == "PASS_CORRECTED_BASELINE_ESTABLISHED_WITH_HISTORICAL_A_INVALIDATED"
    assert plan["performance_gate"]["allowed"] is False
    assert plan["performance_gate"]["reason"]

    # Must fail closed with RuntimeError
    with pytest.raises(RuntimeError, match="Jev performance run blocked"):
        performance_gate(plan)


def test_selector_frozen_artifacts_and_no_speculative_claims():
    from audit_xysz_qfq import parse_frozen_selector_artifacts

    evidence = parse_frozen_selector_artifacts(code_num="002703")

    # 1. 20220601 target: Rank 1, target weight 0.615958
    ev_0601 = evidence["20220601"]
    assert ev_0601["target_topk_status"] == "present_in_target_topk"
    assert ev_0601["rank"] == 1
    assert abs(ev_0601["target_weight"] - 0.615958) < 1e-4

    # 2. 20220630 target: Absent from target_topk, target weight 0.0, listed in previous_target_codes
    ev_0630 = evidence["20220630"]
    assert ev_0630["target_topk_status"] == "absent_from_target_topk"
    assert ev_0630["target_weight"] == 0.0
    assert ev_0630["in_previous_target_codes"] is True
    assert "002703" in ev_0630["previous_target_codes"]
    assert ev_0630["full_cross_sectional_rank"] == "unavailable"

    # 3. 20220701 target: Absent from target_topk, target weight 0.0
    ev_0701 = evidence["20220701"]
    assert ev_0701["target_topk_status"] == "absent_from_target_topk"
    assert ev_0701["target_weight"] == 0.0
    assert ev_0701["full_cross_sectional_rank"] == "unavailable"

    # 4. Strict check against speculative phrases in audit files
    json_path = AUDIT_DIR / "audit.json"
    md_path = AUDIT_DIR / "audit.md"
    json_text = json_path.read_text(encoding="utf-8")
    md_text = md_path.read_text(encoding="utf-8")

    for bad_phrase in [
        "NOT_IN_TOPK_OR_CANDIDATES_100",
        "candidates_100",
        "跌出 Top 100",
        "跌出Top100",
        "跌出 Top 100",
        "因平价填充而跌出",
        "因填充导致退出",
    ]:
        assert bad_phrase not in json_text, f"Forbidden phrase '{bad_phrase}' in audit.json"
        assert bad_phrase not in md_text, f"Forbidden phrase '{bad_phrase}' in audit.md"


