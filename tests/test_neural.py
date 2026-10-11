import os

import numpy as np
import pandas as pd
import pytest

from stonkfly.config import Settings
from stonkfly.neural.controller import Decoder
from stonkfly.neural.rule import advance
from stonkfly.reinforcement import reinforcement


def test_deterministic_neural_payload_drops_only_wall_clock_diagnostic():
    from stonkfly.neural.controller import deterministic_neural_payload

    neural = {"side": "BUY", "spike_sha256": "spike", "memory": {"changed_edges": 1}, "compute_seconds": 0.5}
    payload = deterministic_neural_payload(neural)
    assert "compute_seconds" not in payload
    assert payload["side"] == neural["side"]
    assert payload["spike_sha256"] == neural["spike_sha256"]
    assert payload["memory"] == neural["memory"]

def test_fixed_neuron_decoder():
    a = pd.DataFrame(
        {"type": ["DNp20", "DNp20", "DNpe017"], "somaSide": ["L", "R", "L"]}
    )
    d = Decoder(np.array([1, 2, 3]), a, 2)
    assert d.decode(np.array([0, 10, 0]), 0.5)["side"] == "HOLD"
    assert d.decode(np.array([0, 10, 1]), 0.5)["side"] == "BUY"
    assert d.decode(np.array([10, 0, 1]), 0.5)["side"] == "SELL"
    assert d.decode(np.array([4, 4, 1]), 0.5)["side"] == "HOLD"


@pytest.mark.parametrize(
    "equity,expected",
    [("100.03", "reward"), ("99.97", "aversive"), ("100.001", "none"), ("100", "none")],
)
def test_explicit_feedback(equity, expected):
    assert reinforcement(equity, "100", ".01")[0] == expected


def trace_protocol(order, frozen=False):
    k = np.zeros(2)
    d = np.zeros(1)
    u = np.zeros(2)
    w = np.zeros(2)
    gain = np.ones((1, 2))
    for phase in order:
        for _ in range(20):
            kh = np.array([20.0, 0.0]) if phase == "cue" else np.zeros(2)
            dh = np.array([30.0]) if phase == "reinforce" else np.zeros(1)
            advance(k, d, u, w, kh, dh, gain, 0.01, 0.001, frozen=frozen)
    return w


def test_memory_rule_temporal_specificity():
    paired = trace_protocol(["cue", "reinforce"])
    reverse = trace_protocol(["reinforce", "cue"])
    assert paired[0] < 0 and reverse[0] > 0
    assert paired[1] == 0 and reverse[1] == 0  # Unactivated input is unchanged.
    assert np.array_equal(trace_protocol(["cue", "reinforce"], True), np.zeros(2))


@pytest.mark.skipif(
    os.environ.get("STONKFLY_FULL_TEST") != "1",
    reason="Downloads/uses full MaleCNS; explicit integration test",
)
def test_full_graph_sensory_reinforcement_checkpoint(tmp_path):
    from stonkfly.data import verify
    from stonkfly.neural.controller import FlyController

    assert verify()["neurons"] == 166700
    c = FlyController(Settings())
    assert len(c.brain.post) == 25582938 and len(c.brain.circuit["edges"]) == 7835
    assert len(c.brain.retina) == 3335 and len(c.brain.r8) == 811
    white = np.full((180, 320, 3), 255, np.uint8)
    for _ in range(3):
        c.observe(white, "none")
    c.save(tmp_path / "before.npz")
    before = c.brain.weight[c.brain.circuit["edges"]].copy()
    reward = c.observe(white, "reward")
    assert reward["reward_spikes"] > 0 and reward["stimulus_ms"] == 200
    assert reward["KC_spikes"] > 0 and reward["memory"]["changed_edges"] > 0
    reward_weights = c.brain.weight[c.brain.circuit["edges"]].copy()
    c.restore(tmp_path / "before.npz")
    control = c.observe(white, "none")
    assert not np.array_equal(reward_weights, c.brain.weight[c.brain.circuit["edges"]])
    c.restore(tmp_path / "before.npz")
    c.brain.weights_frozen = True
    c.observe(white, "reward")
    assert np.array_equal(before, c.brain.weight[c.brain.circuit["edges"]])
    c.restore(tmp_path / "before.npz")
    loss = c.observe(white, "aversive")
    assert loss["aversive_spikes"] > 0 and loss["stimulus_ms"] == 200
    assert np.isfinite(c.brain.weight).all()
    print({"reward": reward, "unpaired_control": control, "loss": loss})


@pytest.mark.skipif(os.environ.get("STONKFLY_FULL_TEST") != "1", reason="full MaleCNS integration")
def test_reward_and_aversive_diverge_from_one_clean_checkpoint_in_native_and_saved_weights(tmp_path):
    from stonkfly.neural.controller import FlyController
    controller = FlyController(Settings(neural_ms=200, pulse_ms=100, learning=True))
    frame = np.full((180, 320, 3), 255, np.uint8)
    clean = tmp_path / "clean.npz"
    controller.save(clean)
    controller.restore(clean)
    reward = controller.observe(frame, "reward")
    reward_native = controller.brain.weight[controller.brain.circuit["edges"]].copy()
    reward_checkpoint = tmp_path / "reward.npz"
    controller.save(reward_checkpoint)
    controller.restore(clean)
    aversive = controller.observe(frame, "aversive")
    aversive_native = controller.brain.weight[controller.brain.circuit["edges"]].copy()
    aversive_checkpoint = tmp_path / "aversive.npz"
    controller.save(aversive_checkpoint)
    with np.load(reward_checkpoint, allow_pickle=False) as saved_reward, np.load(aversive_checkpoint, allow_pickle=False) as saved_aversive:
        reward_persisted = saved_reward["weight"][controller.brain.circuit["edges"]]
        aversive_persisted = saved_aversive["weight"][controller.brain.circuit["edges"]]
    assert reward["stimulus"] == "reward" and aversive["stimulus"] == "aversive"
    assert not np.array_equal(reward_native, aversive_native)
    assert not np.array_equal(reward_persisted, aversive_persisted)
    assert np.array_equal(reward_native, reward_persisted)
    assert np.array_equal(aversive_native, aversive_persisted)


def test_mbon_decoder_direction_p0_fix():
    """Verify MBONDecoder direction: MBON11 > MBON07 → BUY (P0 fix).
    
    Based on Aso 2014: KC-DAN pairing suppresses KC→MBON.
    - Reward suppresses MBON07 → approach bias when MBON11 dominates
    - Punishment suppresses MBON11 → avoidance bias when MBON07 dominates
    """
    from stonkfly.neural.controller import MBONDecoder
    
    # Create mock annotation with MBON07 and MBON11
    a = pd.DataFrame({
        "type": ["MBON07", "MBON07", "MBON11", "MBON11", "DNpe017"],
    })
    ids = np.array([1, 2, 3, 4, 5])
    decoder = MBONDecoder(ids, a, threshold=2.0, normalize=False)
    
    # Test case 1: MBON11 dominates (indexes 2,3) → should be BUY
    # This represents a pattern that was previously rewarded (MBON07 suppressed)
    counts_mbon11_high = np.array([5, 5, 20, 20, 1])  # MBON11 high, MBON07 low
    result = decoder.decode(counts_mbon11_high, 0.5)
    assert result["side"] == "BUY", f"MBON11 > MBON07 should → BUY, got {result['side']}"
    assert result["mbon11_hz"] > result["mbon07_hz"]
    assert result["difference_hz"] > 0  # positive = MBON11 dominates
    
    # Test case 2: MBON07 dominates (indexes 0,1) → should be SELL  
    # This represents a pattern that was previously punished (MBON11 suppressed)
    counts_mbon07_high = np.array([20, 20, 5, 5, 1])  # MBON07 high, MBON11 low
    result = decoder.decode(counts_mbon07_high, 0.5)
    assert result["side"] == "SELL", f"MBON07 > MBON11 should → SELL, got {result['side']}"
    assert result["mbon07_hz"] > result["mbon11_hz"]
    assert result["difference_hz"] < 0  # negative = MBON07 dominates
    
    # Test case 3: Balanced → HOLD (within threshold)
    counts_balanced = np.array([10, 10, 10, 10, 1])
    result = decoder.decode(counts_balanced, 0.5)
    assert result["side"] == "HOLD", f"Balanced should → HOLD, got {result['side']}"
    assert abs(result["difference_hz"]) < 2.0


def test_mbon_decoder_difference_sign_convention():
    """Verify MBONDecoder difference calculation: difference = mbon11_hz - mbon07_hz."""
    from stonkfly.neural.controller import MBONDecoder
    
    a = pd.DataFrame({"type": ["MBON07", "MBON11", "DNpe017"]})
    ids = np.array([1, 2, 3])
    decoder = MBONDecoder(ids, a, threshold=1.0, normalize=False)
    
    # MBON07=10Hz, MBON11=20Hz → difference should be +10
    counts = np.array([5, 10, 1])  # 5 spikes / 0.5s = 10Hz, 10 spikes / 0.5s = 20Hz
    result = decoder.decode(counts, 0.5)
    
    assert result["mbon07_hz"] == 10.0
    assert result["mbon11_hz"] == 20.0
    assert result["difference_hz"] == 10.0  # mbon11 - mbon07 = 20 - 10 = 10
    assert result["side"] == "BUY"  # positive difference → BUY
