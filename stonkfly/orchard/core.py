"""Deterministic 2-D orchard physics and explicitly separated fly control."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable, Protocol

import numpy as np

ODORS = ("banana", "flower", "rest")


@dataclass(frozen=True)
class OrchardConfig:
    width: float = 960.0
    height: float = 620.0
    world_dt_seconds: float = 0.1
    speed_px_per_second: float = 54.0
    turn_rate_radians_per_second: float = 2.4
    neural_gain: float = 1.35
    exploration_gain: float = 0.72
    exploration_persistence: float = 0.88
    odor_sigma_px: float = 155.0
    reward_radius_px: float = 42.0
    fly_radius_px: float = 7.0
    energy_drain_per_second: float = 0.007
    rest_recovery_per_second: float = 0.10
    maximum_trail_points: int = 700
    log_every_steps: int = 10

    def validate(self) -> None:
        values = asdict(self)
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise ValueError("orchard configuration must be finite")
        if self.width <= 100 or self.height <= 100 or self.world_dt_seconds <= 0:
            raise ValueError("orchard dimensions and dt must be positive")
        if self.speed_px_per_second < 0 or self.turn_rate_radians_per_second < 0:
            raise ValueError("speed and turn rate must be nonnegative")
        if not 0 <= self.exploration_persistence < 1:
            raise ValueError("exploration persistence must be in [0, 1)")
        if self.maximum_trail_points < 2 or self.log_every_steps < 1:
            raise ValueError("invalid trail or log cadence")


@dataclass(frozen=True)
class OrchardPatch:
    name: str
    kind: str
    x: float
    y: float
    radius: float


@dataclass(frozen=True)
class BrainSnapshot:
    preferences: dict[str, float]
    innate_preferences: dict[str, float]
    learned_preferences: dict[str, float]
    kc_activity: float
    apl_activity: float
    memory_norm: float
    memory_digest: str
    reward_dan: float
    model: str


class OrchardBrain(Protocol):
    def reset(self) -> None: ...

    def step(
        self, concentrations: dict[str, float], *, reward: float, dt_seconds: float
    ) -> BrainSnapshot: ...


class ToyOrchardBrain:
    """Small deterministic fixture for tests and the optional dependency-free demo."""

    def __init__(self, preference_scale: float = 1.0) -> None:
        self.preference_scale = float(preference_scale)
        self.memory = 0.0

    def reset(self) -> None:
        self.memory = 0.0

    def step(
        self, concentrations: dict[str, float], *, reward: float, dt_seconds: float
    ) -> BrainSnapshot:
        banana = float(concentrations.get("banana", 0.0))
        self.memory += dt_seconds * (reward * banana * 0.45 - self.memory / 90.0)
        self.memory = float(np.clip(self.memory, -1.0, 1.0))
        innate = {
            "banana": 0.12 * self.preference_scale,
            "flower": 0.04 * self.preference_scale,
            "rest": -0.02 * self.preference_scale,
        }
        learned = {"banana": self.memory, "flower": 0.0, "rest": 0.0}
        preferences = {
            odor: float(np.clip(innate[odor] + learned[odor], -1.0, 1.0))
            for odor in ODORS
        }
        digest = f"toy-{self.memory:+.9f}"
        return BrainSnapshot(
            preferences=preferences,
            innate_preferences=innate,
            learned_preferences=learned,
            kc_activity=sum(concentrations.values()) * 0.08,
            apl_activity=sum(concentrations.values()) * 0.025,
            memory_norm=abs(self.memory),
            memory_digest=digest,
            reward_dan=float(reward),
            model="toy-fixture",
        )


def default_patches() -> tuple[OrchardPatch, ...]:
    return (
        OrchardPatch("ripe-banana", "banana", 770.0, 145.0, 34.0),
        OrchardPatch("wildflowers", "flower", 188.0, 168.0, 48.0),
        OrchardPatch("mossy-rest", "rest", 485.0, 500.0, 64.0),
    )


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


class OrchardSimulation:
    """Owns the world and records every control contribution separately."""

    def __init__(
        self,
        brain_factory: Callable[[], OrchardBrain],
        *,
        seed: int = 20261012,
        config: OrchardConfig | None = None,
        patches: tuple[OrchardPatch, ...] | None = None,
    ) -> None:
        self.config = config or OrchardConfig()
        self.config.validate()
        self.seed = int(seed)
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        self.patches = patches or default_patches()
        if {patch.kind for patch in self.patches} != set(ODORS):
            raise ValueError("orchard needs exactly banana, flower, and rest odor kinds")
        self._brain_factory = brain_factory
        self.brain = brain_factory()
        self.running = False
        self.events: list[dict[str, object]] = []
        self.reset()

    def reset(self) -> dict[str, object]:
        self.rng = np.random.default_rng(self.seed)
        self.brain.reset()
        self.step_index = 0
        self.sim_time = 0.0
        self.x = self.config.width * 0.50
        self.y = self.config.height * 0.47
        self.heading = -math.pi / 2.0
        self.exploration_state = 0.0
        self.energy = 0.82
        self._banana_contact_latched = False
        self.running = False
        self.trail: list[list[float]] = [[self.x, self.y]]
        concentrations, _ = self._odor_field()
        self.brain_snapshot = self.brain.step(
            concentrations, reward=0.0, dt_seconds=0.0
        )
        self.last_control = {
            "neural_turn": 0.0,
            "exploration_turn": 0.0,
            "total_turn": 0.0,
            "boundary_collision": False,
        }
        self._event("reset", {"seed": self.seed})
        return self.snapshot()

    def command(self, action: str) -> dict[str, object]:
        if action == "start":
            self.running = True
            self._event("control", {"action": "start"})
        elif action == "pause":
            self.running = False
            self._event("control", {"action": "pause"})
        elif action == "reset":
            return self.reset()
        elif action == "step":
            self.running = False
            self.advance()
        else:
            raise ValueError("action must be start, pause, reset, or step")
        return self.snapshot()

    def _odor_field(self) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
        concentrations: dict[str, float] = {}
        gradients: dict[str, tuple[float, float]] = {}
        sigma2 = self.config.odor_sigma_px**2
        for patch in self.patches:
            dx, dy = patch.x - self.x, patch.y - self.y
            distance2 = dx * dx + dy * dy
            intensity = math.exp(-distance2 / (2.0 * sigma2))
            concentrations[patch.kind] = float(intensity)
            if distance2 <= 1e-12:
                gradients[patch.kind] = (0.0, 0.0)
            else:
                distance = math.sqrt(distance2)
                gradients[patch.kind] = (
                    intensity * dx / distance,
                    intensity * dy / distance,
                )
        return concentrations, gradients

    def _contact(self, kind: str) -> bool:
        patch = next(item for item in self.patches if item.kind == kind)
        return math.hypot(self.x - patch.x, self.y - patch.y) <= (
            patch.radius + self.config.fly_radius_px
        )

    def advance(self) -> dict[str, object]:
        dt = self.config.world_dt_seconds
        concentrations, gradients = self._odor_field()
        banana_contact = self._contact("banana")
        reward = 1.0 if banana_contact and not self._banana_contact_latched else 0.0
        self._banana_contact_latched = banana_contact
        neural = self.brain.step(concentrations, reward=reward, dt_seconds=dt)

        desired_x = 0.0
        desired_y = 0.0
        for odor in ODORS:
            preference = neural.preferences[odor]
            desired_x += preference * gradients[odor][0]
            desired_y += preference * gradients[odor][1]
        neural_turn = 0.0
        if abs(desired_x) + abs(desired_y) > 1e-12:
            target_heading = math.atan2(desired_y, desired_x)
            neural_turn = float(
                np.clip(
                    _wrap_angle(target_heading - self.heading),
                    -self.config.turn_rate_radians_per_second,
                    self.config.turn_rate_radians_per_second,
                )
            )

        noise = float(self.rng.normal())
        persistence = self.config.exploration_persistence
        self.exploration_state = (
            persistence * self.exploration_state
            + math.sqrt(1.0 - persistence * persistence) * noise
        )
        exploration_turn = self.exploration_state
        total_turn = (
            self.config.neural_gain * neural_turn
            + self.config.exploration_gain * exploration_turn
        )
        total_turn = float(
            np.clip(
                total_turn,
                -self.config.turn_rate_radians_per_second,
                self.config.turn_rate_radians_per_second,
            )
        )
        self.heading = _wrap_angle(self.heading + total_turn * dt)

        rest_contact = self._contact("rest")
        if rest_contact:
            self.energy += self.config.rest_recovery_per_second * dt
        else:
            self.energy -= self.config.energy_drain_per_second * dt
        self.energy = float(np.clip(self.energy, 0.18, 1.0))
        speed = self.config.speed_px_per_second * (0.45 + 0.55 * self.energy)
        next_x = self.x + math.cos(self.heading) * speed * dt
        next_y = self.y + math.sin(self.heading) * speed * dt
        margin = self.config.fly_radius_px
        collision = False
        if next_x < margin or next_x > self.config.width - margin:
            self.heading = _wrap_angle(math.pi - self.heading)
            next_x = float(np.clip(next_x, margin, self.config.width - margin))
            collision = True
        if next_y < margin or next_y > self.config.height - margin:
            self.heading = _wrap_angle(-self.heading)
            next_y = float(np.clip(next_y, margin, self.config.height - margin))
            collision = True
        self.x, self.y = next_x, next_y
        self.step_index += 1
        self.sim_time = self.step_index * dt
        self.brain_snapshot = neural
        self.last_control = {
            "neural_turn": neural_turn,
            "exploration_turn": exploration_turn,
            "total_turn": total_turn,
            "boundary_collision": collision,
        }
        self.trail.append([round(self.x, 4), round(self.y, 4)])
        if len(self.trail) > self.config.maximum_trail_points:
            self.trail = self.trail[-self.config.maximum_trail_points :]
        if reward or self.step_index % self.config.log_every_steps == 0:
            self._event(
                "reward" if reward else "tick",
                {
                    "position": [self.x, self.y],
                    "reward_dan": reward,
                    "preferences": neural.preferences,
                    "control": self.last_control,
                },
            )
        return self.snapshot()

    def _event(self, kind: str, payload: dict[str, object]) -> None:
        self.events.append(
            {
                "sequence": len(self.events),
                "step": self.step_index,
                "sim_time": round(self.sim_time, 3),
                "kind": kind,
                "payload": payload,
            }
        )

    def snapshot(self) -> dict[str, object]:
        concentrations, _ = self._odor_field()
        return {
            "schema": "stonkfly-orchard-state-v1",
            "seed": self.seed,
            "running": self.running,
            "step": self.step_index,
            "sim_time": round(self.sim_time, 3),
            "world": {
                "width": self.config.width,
                "height": self.config.height,
                "patches": [asdict(patch) for patch in self.patches],
            },
            "fly": {
                "x": self.x,
                "y": self.y,
                "heading": self.heading,
                "energy": self.energy,
                "trail": self.trail,
            },
            "odor": concentrations,
            "neural": asdict(self.brain_snapshot),
            "control": {
                **self.last_control,
                "equation": "turn = neural_gain*neural_turn + exploration_gain*exploration_turn",
                "neural_gain": self.config.neural_gain,
                "exploration_gain": self.config.exploration_gain,
            },
            "contacts": {odor: self._contact(odor) for odor in ODORS},
            "event_count": len(self.events),
        }
