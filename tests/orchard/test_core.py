import json

import pytest

from stonkfly.orchard.core import OrchardConfig, OrchardSimulation, ToyOrchardBrain
from stonkfly.orchard.server import OrchardRuntime, STATIC


def make(seed=17, *, neural_gain=1.35, exploration_gain=0.72):
    config = OrchardConfig(
        neural_gain=neural_gain,
        exploration_gain=exploration_gain,
        log_every_steps=1,
    )
    return OrchardSimulation(ToyOrchardBrain, seed=seed, config=config)


def compact(state):
    return {
        "step": state["step"],
        "fly": {key: state["fly"][key] for key in ("x", "y", "heading", "energy")},
        "neural": state["neural"],
        "control": state["control"],
    }


def test_fixed_seed_replay_is_exact():
    simulation = make(42)
    first = [compact(simulation.advance()) for _ in range(40)]
    simulation.reset()
    second = [compact(simulation.advance()) for _ in range(40)]
    assert first == second


def test_control_terms_are_explicit_and_sum_before_clip():
    simulation = make(8)
    state = simulation.advance()
    control = state["control"]
    raw = (
        control["neural_gain"] * control["neural_turn"]
        + control["exploration_gain"] * control["exploration_turn"]
    )
    limit = simulation.config.turn_rate_radians_per_second
    assert control["total_turn"] == pytest.approx(max(-limit, min(limit, raw)))
    assert "neural_turn" in control and "exploration_turn" in control


def test_neural_output_changes_motion_without_hidden_targeting():
    driven = make(5, neural_gain=1.35, exploration_gain=0.0)
    neutral = OrchardSimulation(
        lambda: ToyOrchardBrain(preference_scale=0.0),
        seed=5,
        config=OrchardConfig(neural_gain=0.0, exploration_gain=0.0),
    )
    for _ in range(30):
        driven.advance()
        neutral.advance()
    assert (driven.x, driven.y) != pytest.approx((neutral.x, neutral.y))
    assert neutral.last_control["total_turn"] == 0.0


def test_banana_contact_drives_declared_reward_and_memory():
    simulation = make(3, exploration_gain=0.0)
    banana = next(patch for patch in simulation.patches if patch.kind == "banana")
    simulation.x, simulation.y = banana.x, banana.y
    before = simulation.brain_snapshot.memory_norm
    state = simulation.advance()
    assert state["contacts"]["banana"]
    assert state["neural"]["reward_dan"] == 1.0
    assert state["neural"]["memory_norm"] > before


def test_commands_pause_step_start_and_reset():
    simulation = make(22)
    assert simulation.command("start")["running"] is True
    assert simulation.command("pause")["running"] is False
    stepped = simulation.command("step")
    assert stepped["step"] == 1 and not stepped["running"]
    reset = simulation.command("reset")
    assert reset["step"] == 0 and not reset["running"]
    with pytest.raises(ValueError):
        simulation.command("seek-banana")


def test_runtime_appends_jsonl_without_overwrite(tmp_path):
    path = tmp_path / "events.jsonl"
    runtime = OrchardRuntime(make(9), log_path=path)
    runtime.command("step")
    runtime.command("step")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["sequence"] for row in rows] == list(range(len(rows)))
    assert rows[0]["kind"] == "reset"
    runtime.close()


def test_static_demo_exposes_required_controls_and_signals():
    html = (STATIC / "index.html").read_text()
    script = (STATIC / "app.js").read_text()
    for item in ("start", "pause", "step", "reset", "orchard", "preferences"):
        assert f'id="{item}"' in html
    for field in ("neural_turn", "exploration_turn", "memory_norm", "/api/state"):
        assert field in script
    assert "前量化交易员，现居果园" in html
    assert "对人类安排的奖励机制持保留态度" in html
