"""MaleCNS-backed Orchard brain using the retained E1-G and M2 mechanisms."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from stonkfly.mb_lab.dynamics import M1Parameters
from stonkfly.mb_lab.e1 import build_apl_mapping
from stonkfly.mb_lab.inventory import GRAPH, load_source, selected_nodes
from stonkfly.mb_lab.m2e import (
    PlasticityState,
    SensoryCache,
    build_connectivity,
    normalized_valence,
    raw_readout,
    train_event,
)
from stonkfly.mb_lab.topology import compile_real_topology

from .core import BrainSnapshot, ODORS


class MaleCNSOrchardBrain:
    """Interactive adapter, not a new formal M1/M2 experimental result.

    Three deterministic odor templates are run once through E1-G at the tested
    0.1 ms display timestep. Runtime concentrations mix those cached KC rates.
    Plasticity then advances at the existing 10 ms M2 bin size. This preserves
    the implemented mechanisms while keeping a browser demo responsive.
    """

    def __init__(self, *, seed: int = 20261012, graph_path: Path = GRAPH) -> None:
        self.seed = int(seed)
        source = load_source()
        pn, kc = selected_nodes("R", olfactory_covered_only=True)
        topology = compile_real_topology(source["ptr"], source["post"], pn, kc)
        mapping = build_apl_mapping(graph_path, source, topology.kc_node_index)
        self.connectivity = build_connectivity(graph_path, source, topology.kc_node_index)
        parameters = M1Parameters(
            tau_k_ms=10.0,
            tau_apl_ms=50.0,
            input_scale=1.0,
            apl_gain=1.0,
            threshold=0.35,
            active_integral_threshold=25.0,
        )
        cache = SensoryCache(topology, parameters, mapping, dt_ms=0.1)
        pulses = self._odor_pulses(topology.n_pn)
        self.rates: dict[str, np.ndarray] = {}
        self.sensory = {}
        self.apl_levels: dict[str, float] = {}
        self.baseline_channels: dict[str, np.ndarray] = {}
        unit_efficacy = np.ones_like(self.connectivity.baseline_weight)
        for odor in ODORS:
            _, sensory = cache.get(pulses[odor])
            self.sensory[odor] = sensory
            rate = np.asarray(sensory.integrated_rate / 500.0, dtype=np.float64)
            rate.setflags(write=False)
            self.rates[odor] = rate
            self.apl_levels[odor] = float(np.mean(sensory.apl_trace[50:100]))
            channels = raw_readout(self.connectivity, rate, unit_efficacy)["channels"]
            if np.any(channels <= 1e-12):
                raise RuntimeError(f"{odor} does not drive both MBON channels")
            channels.setflags(write=False)
            self.baseline_channels[odor] = channels
        raw_bias = {
            odor: float(
                (channels[1] - channels[0]) / (channels[1] + channels[0])
            )
            for odor, channels in self.baseline_channels.items()
        }
        center = float(np.mean(list(raw_bias.values())))
        self.innate = {
            odor: float(np.clip(1.5 * (raw_bias[odor] - center), -0.35, 0.35))
            for odor in ODORS
        }
        self.topology_sha256 = topology.sha256
        self.model_digest = self._model_digest(pulses)
        self.reset()

    def _odor_pulses(self, n_pn: int) -> dict[str, np.ndarray]:
        digest = hashlib.sha256(
            f"STONKFLY-ORCHARD/ODORS/v1/{self.seed}".encode()
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        pulses: dict[str, np.ndarray] = {}
        for odor in ODORS:
            pulse = np.zeros(n_pn, dtype=np.float64)
            active = rng.choice(n_pn, size=29, replace=False)
            pulse[active] = rng.uniform(0.8, 1.0, size=len(active))
            pulse.setflags(write=False)
            pulses[odor] = pulse
        return pulses

    def _model_digest(self, pulses: dict[str, np.ndarray]) -> str:
        digest = hashlib.sha256(b"STONKFLY-ORCHARD-BRAIN/v1")
        digest.update(self.topology_sha256.encode())
        digest.update(np.ascontiguousarray(self.connectivity.baseline_weight).tobytes())
        for odor in ODORS:
            digest.update(odor.encode())
            digest.update(np.ascontiguousarray(pulses[odor]).tobytes())
        return digest.hexdigest()

    def reset(self) -> None:
        self.state = PlasticityState.fresh(self.connectivity, eta=0.1)

    def _preferences(self) -> tuple[dict[str, float], dict[str, float]]:
        learned: dict[str, float] = {}
        combined: dict[str, float] = {}
        efficacy = self.state.efficacy
        for odor in ODORS:
            current = raw_readout(self.connectivity, self.rates[odor], efficacy)[
                "channels"
            ]
            learned[odor] = normalized_valence(
                current, self.baseline_channels[odor]
            )
            combined[odor] = float(
                np.clip(self.innate[odor] + 5.0 * learned[odor], -1.0, 1.0)
            )
        return combined, learned

    def step(
        self, concentrations: dict[str, float], *, reward: float, dt_seconds: float
    ) -> BrainSnapshot:
        if dt_seconds < 0 or not math.isclose(
            dt_seconds / 0.01, round(dt_seconds / 0.01), abs_tol=1e-9
        ):
            raise ValueError("Orchard brain dt must be a nonnegative multiple of 10 ms")
        weights = np.asarray(
            [max(0.0, float(concentrations.get(odor, 0.0))) for odor in ODORS],
            dtype=np.float64,
        )
        scale = max(1.0, float(weights.sum()))
        mixed_rate = sum(
            (weights[index] / scale) * self.rates[odor]
            for index, odor in enumerate(ODORS)
        )
        zero_dan = np.zeros(2, dtype=np.float64)
        for _ in range(round(dt_seconds / 0.01)):
            self.state.advance_bin(mixed_rate, zero_dan, dt_seconds=0.01)
        if reward > 0:
            train_event(
                self.state,
                self.sensory["banana"],
                valence="reward",
                timing="paired",
                dan_amplitude=float(reward),
                washout_seconds=0.0,
            )
        preferences, learned = self._preferences()
        apl_activity = sum(
            (weights[index] / scale) * self.apl_levels[odor]
            for index, odor in enumerate(ODORS)
        )
        return BrainSnapshot(
            preferences=preferences,
            innate_preferences=dict(self.innate),
            learned_preferences=learned,
            kc_activity=float(np.mean(mixed_rate)),
            apl_activity=float(apl_activity),
            memory_norm=float(np.linalg.norm(self.state.memory_w) / math.sqrt(len(self.state.memory_w))),
            memory_digest=self.state.weight_digest()[:16],
            reward_dan=float(reward),
            model="MaleCNS E1-G + M2 plasticity (interactive proxy)",
        )
