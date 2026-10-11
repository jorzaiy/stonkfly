from decimal import Decimal as D

import numpy as np

from experiments.a_share.replay_510300_v3 import (
    action_advantage,
    action_advantage_sign,
    default_state,
    days_since_fill,
    portfolio_frame,
    record_veto,
    restore_state,
    reversal_cooldown,
    shaped_feedback,
)


def feedback(**changes):
    values = {
        "equity": D("100"),
        "previous_equity": D("100"),
        "benchmark_equity": D("100"),
        "previous_benchmark_equity": D("100"),
        "drawdown": 0.0,
        "previous_drawdown": 0.0,
        "turnover_ratio": 0.0,
        "vetoed": False,
        "deadband_bps": 1.0,
        "drawdown_weight": 0.25,
        "turnover_weight": 0.001,
        "veto_penalty_bps": 2.0,
    }
    values.update(changes)
    return shaped_feedback(**values)


def test_action_advantage_buy_sell_hold_uses_decimal_formula():
    assert action_advantage("BUY", 100, D("10"), D("9.9"), D("1")) == D("9")
    assert action_advantage("SELL", 100, D("10"), D("10.2"), D("1")) == D("19")
    assert action_advantage("HOLD", 100, D("10"), D("1"), D("1")) == D("0")
    assert action_advantage_sign(D("9")) == 1.0
    assert action_advantage_sign(D("-1")) == -1.0
    assert action_advantage_sign(D("0")) == 0.0


def test_advantage_changes_score_and_stimulus_with_default_weight_one():
    positive = feedback(action_advantage_ratio=0.002)
    negative = feedback(action_advantage_ratio=-0.002)
    assert positive["score"] == 0.002
    assert negative["score"] == -0.002
    assert positive["stimulus"] == "reward"
    assert negative["stimulus"] == "aversive"


def test_advantage_weight_is_explicit_and_scales_score():
    result = feedback(action_advantage_ratio=0.002, action_advantage_weight=2.0)
    assert result["score"] == 0.004
    assert result["action_advantage_ratio"] == 0.002


def test_reversal_cooldown_inclusive_boundary_and_hold_are_correct():
    assert reversal_cooldown("BUY", 10, "SELL", 10)
    assert reversal_cooldown("BUY", 10, "SELL", 13)
    assert not reversal_cooldown("BUY", 10, "SELL", 14)
    assert not reversal_cooldown("BUY", 10, "BUY", 13)
    assert not reversal_cooldown("BUY", 10, "HOLD", 11)
    assert not reversal_cooldown("BUY", 10, "SELL", 9)


def test_days_since_fill_no_fill_and_clip():
    assert days_since_fill(None, 0) == 10.0
    assert days_since_fill(4, 4) == 0.0
    assert days_since_fill(4, 7) == 3.0
    assert days_since_fill(4, 99) == 10.0


def _frame(**changes):
    values = {
        "exposure": 0.2,
        "cash_ratio": 0.8,
        "unrealized_return": 0.0,
        "drawdown": 0.0,
        "relative_return": 0.0,
        "buy_capacity_ratio": 0.1,
        "sellable_ratio": 0.1,
        "last_fill_side": "NONE",
        "days_since_fill": 10.0,
        "action_advantage_sign": 0.0,
        "action_advantage_ratio": 0.0,
    }
    values.update(changes)
    return portfolio_frame(np.zeros((180, 320, 3), dtype=np.uint8), **values)


def test_each_new_visual_state_changes_rgb_input():
    baseline = _frame()
    for change in (
        {"buy_capacity_ratio": 0.8},
        {"sellable_ratio": 0.8},
        {"last_fill_side": "BUY"},
        {"days_since_fill": 1.0},
        {"action_advantage_sign": 1.0},
        {"action_advantage_ratio": 0.1},
    ):
        assert not np.array_equal(baseline, _frame(**change)), change


def test_state_restore_and_veto_accounting_keep_resume_fields():
    state = default_state("sig", D("100"))
    record_veto(state, "reversal_cooldown")
    assert state["veto_count"] == 1
    assert state["veto_reasons"] == {"reversal_cooldown": 1}
    assert state["reversal_count"] == 1
    old = {"signature": "sig", "reversal_veto_count": 2}
    restored = restore_state(old, "sig", D("100"))
    for key in (
        "cash", "shares", "sellable_shares", "unsettled", "last_fill_side",
        "last_fill_index", "action_advantage_cny", "action_advantage_ratio",
        "action_advantage_sign", "last_portfolio_observation", "veto_reasons",
        "reversal_count", "checkpoint",
    ):
        assert key in restored
    assert restored["reversal_count"] == 2
