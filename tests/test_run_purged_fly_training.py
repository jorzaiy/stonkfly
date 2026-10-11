from types import SimpleNamespace

from stonkfly.config import Settings

import experiments.a_share.run_purged_fly_training as training


def test_controller_factory_preserves_caller_settings(monkeypatch):
    captured = {}

    class FakeController:
        def __init__(self, settings):
            captured["settings"] = settings

    monkeypatch.setattr(training, "V51Controller", FakeController)
    original = Settings(
        learning=True,
        neural_ms=700.0,
        pulse_delay_ms=300.0,
        pulse_ms=200.0,
        pulse_current=17.0,
        reward_deadband="0.02",
    )

    controller = training._make_controller_factory("mbon", 3.5)(original)

    assert isinstance(controller, FakeController)
    updated = captured["settings"]
    assert updated.decoder_type == "mbon"
    assert updated.decoder_threshold_hz == 3.5
    assert updated.neural_ms == original.neural_ms
    assert updated.pulse_delay_ms == original.pulse_delay_ms
    assert updated.pulse_ms == original.pulse_ms
    assert updated.pulse_current == original.pulse_current
    assert updated.reward_deadband == original.reward_deadband


def test_ratio_threshold_defaults_leave_a_hold_band():
    settings = Settings()
    assert settings.mbon_buy_threshold < settings.mbon_sell_threshold
