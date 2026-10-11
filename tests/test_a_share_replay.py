from decimal import Decimal as D
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.a_share.replay_510300 import (
    buy_capacity,
    portfolio_frame,
    shaped_feedback,
)


def feedback(**changes):
    values = {
        "equity": D("100.2"),
        "previous_equity": D("100"),
        "benchmark_equity": D("100.1"),
        "previous_benchmark_equity": D("100"),
        "drawdown": -0.01,
        "previous_drawdown": -0.02,
        "turnover_ratio": 0.0,
        "vetoed": False,
        "deadband_bps": 1.0,
        "drawdown_weight": 0.25,
        "turnover_weight": 0.001,
        "veto_penalty_bps": 2.0,
    }
    values.update(changes)
    return shaped_feedback(**values)


def test_relative_risk_feedback_rewards_outperformance_and_recovery():
    result = feedback()
    assert result["stimulus"] == "reward"
    assert result["excess_return"] > 0
    assert result["drawdown_change"] > 0


def test_relative_risk_feedback_penalizes_drawdown_turnover_and_veto():
    result = feedback(
        equity=D("99.9"),
        benchmark_equity=D("100"),
        drawdown=-0.03,
        previous_drawdown=-0.01,
        turnover_ratio=0.1,
        vetoed=True,
    )
    assert result["stimulus"] == "aversive"
    assert result["veto_penalty_applied"] is True


def test_exposure_guard_preserves_cash_buffer():
    assert buy_capacity(D("100"), 0, D("1"), D("20"), D("0.85")) == D("20")
    assert buy_capacity(D("15"), 85, D("1"), D("20"), D("0.85")) == 0


def test_portfolio_feedback_is_visible_rgb_without_changing_shape():
    source = np.zeros((180, 320, 3), dtype=np.uint8)
    calm = portfolio_frame(
        source,
        exposure=0.2,
        cash_ratio=0.8,
        unrealized_return=0.0,
        drawdown=0.0,
        relative_return=0.0,
        vetoed=False,
    )
    stressed = portfolio_frame(
        source,
        exposure=0.85,
        cash_ratio=0.15,
        unrealized_return=-0.1,
        drawdown=-0.15,
        relative_return=-0.08,
        vetoed=True,
    )
    assert calm.shape == source.shape and calm.dtype == np.uint8
    assert not np.array_equal(calm, stressed)
    assert stressed[28, 0, 0] > calm[28, 0, 0]
