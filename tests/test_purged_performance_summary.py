from pathlib import Path

from tools.summarize_purged_fly_base_performance import build_summary


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/wc006_purged_fly_base_baseline_phase1/20260925T-frozen-rgb-seed17"


def test_frozen_performance_compounds_to_settlement_and_keeps_base_independent():
    summary, _audit = build_summary(RUN, ROOT)
    assert summary["status"] == "PASS_RESEARCH_ONLY"
    assert summary["event_count"] == 609
    assert summary["fly"]["performance"]["final_equity"] == "81067.820017500"
    assert summary["base"]["performance"]["final_equity"] == "92197.628508500"
    assert summary["relative"]["fly_lag_base_percentage_points"] == "11.129808491000"
    assert summary["fly"]["performance"]["compound_difference"] != "0"
    assert abs(float(summary["fly"]["performance"]["compound_difference"])) < 1e-24
    assert summary["fly"]["ledger_counts"] == {
        "intents": 351,
        "fills": 332,
        "vetoes": 19,
        "unknown": 0,
        "intent_statuses": {"FILLED": 332, "VETO": 19},
    }
    assert summary["base"]["ledger_counts"] == {
        "intents": 239,
        "fills": 225,
        "vetoes": 14,
        "unknown": 0,
        "intent_statuses": {"FILLED": 225, "VETO": 14},
    }
    evidence = summary["base_action_independence"]
    assert list(evidence["excerpt"]) == [evidence["lines"]]
    excerpt = evidence["excerpt"][evidence["lines"]]
    assert 'multiplier=D("1.0")' in excerpt
    assert 'neural_side="BUY"' in excerpt


def test_summary_inputs_are_read_only_and_stable():
    before = {
        name: __import__("hashlib").sha256((RUN / name).read_bytes()).hexdigest()
        for name in ("ledger.sqlite", "paired_base.sqlite", "events.jsonl")
    }
    build_summary(RUN, ROOT)
    after = {
        name: __import__("hashlib").sha256((RUN / name).read_bytes()).hexdigest()
        for name in before
    }
    assert after == before
