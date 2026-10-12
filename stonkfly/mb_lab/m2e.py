"""Exploratory E1-G to KC-MBON/DAN association-learning adapter."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .dynamics import M1Parameters
from .e1 import E1Result, APLMapping, simulate_e1
from .topology import BipartiteTopology
from stonkfly.neural.rule import advance as advance_rule

CHANNEL_NAMES = ("avoidance_mbon07", "approach_mbon11")
VALENCE_DAN = {"reward": 0, "aversive": 1}


@dataclass(frozen=True)
class M2EConnectivity:
    n_kc: int
    pre_slot: np.ndarray
    target_slot: np.ndarray
    target_node_index: np.ndarray
    target_source_id: np.ndarray
    target_channel: np.ndarray
    baseline_weight: np.ndarray
    edge_channel: np.ndarray
    gain: np.ndarray

    def validate(self, n_kc: int) -> None:
        n_edge = len(self.pre_slot)
        if n_edge != 3511 or len(self.target_node_index) != 3:
            raise ValueError("unexpected right KC-to-MBON assay inventory")
        if any(len(array) != n_edge for array in (
            self.target_slot, self.baseline_weight, self.edge_channel
        )):
            raise ValueError("unaligned KC-to-MBON edge arrays")
        if self.gain.shape != (2, n_edge):
            raise ValueError("gain must have two DAN channels")
        if self.n_kc != n_kc or np.any(self.pre_slot < 0) or np.any(self.pre_slot >= n_kc):
            raise ValueError("KC edge endpoint outside sensory population")
        if np.any(self.baseline_weight <= 0) or not np.isfinite(self.baseline_weight).all():
            raise ValueError("KC-to-MBON baseline weights must be positive and finite")
        if np.bincount(self.edge_channel, minlength=2).tolist() != [1806, 1705]:
            raise ValueError("unexpected MBON07/11 edge counts")
        if np.any(self.gain.sum(axis=0) != 1.0):
            raise ValueError("each plastic edge must belong to exactly one DAN compartment")


def build_connectivity(
    graph_path: Path,
    source: dict[str, np.ndarray],
    kc_node_index: np.ndarray,
    *,
    side: str = "R",
) -> M2EConnectivity:
    kc_nodes = np.asarray(kc_node_index, dtype=np.int32)
    targets = np.flatnonzero(
        np.isin(source["cell_type"], ("MBON07", "MBON11"))
        & (source["soma_side"] == side)
    ).astype(np.int32)
    target_channels = np.asarray(
        [0 if source["cell_type"][node] == "MBON07" else 1 for node in targets],
        dtype=np.int8,
    )
    order = np.lexsort((source["ids"][targets], target_channels))
    targets, target_channels = targets[order], target_channels[order]
    if target_channels.tolist() != [0, 0, 1]:
        raise ValueError("expected two right MBON07 and one right MBON11")
    target_lookup = {int(node): slot for slot, node in enumerate(targets)}
    pre_slot: list[int] = []
    target_slot: list[int] = []
    weights: list[float] = []
    with np.load(graph_path, allow_pickle=False) as graph:
        ptr = np.asarray(graph["ptr"], dtype=np.int64)
        post = np.asarray(graph["post"], dtype=np.int32)
        weight = np.asarray(graph["weight"], dtype=np.float64)
        for kc_slot, node_value in enumerate(kc_nodes):
            node = int(node_value)
            for edge in range(int(ptr[node]), int(ptr[node + 1])):
                slot = target_lookup.get(int(post[edge]))
                if slot is None:
                    continue
                if weight[edge] <= 0:
                    raise ValueError("KC-to-MBON assay edge must be excitatory")
                pre_slot.append(kc_slot)
                target_slot.append(slot)
                weights.append(float(weight[edge]))
    target_slot_array = np.asarray(target_slot, dtype=np.int8)
    edge_channel = target_channels[target_slot_array]
    gain = np.zeros((2, len(pre_slot)), dtype=np.float64)
    gain[edge_channel, np.arange(len(pre_slot))] = 1.0
    result = M2EConnectivity(
        n_kc=len(kc_nodes),
        pre_slot=np.asarray(pre_slot, dtype=np.int32),
        target_slot=target_slot_array,
        target_node_index=targets,
        target_source_id=source["ids"][targets].astype(np.int64),
        target_channel=target_channels,
        baseline_weight=np.asarray(weights, dtype=np.float64),
        edge_channel=edge_channel,
        gain=gain,
    )
    result.validate(len(kc_nodes))
    return result


@dataclass
class PlasticityState:
    connectivity: M2EConnectivity
    eta: float
    kc_trace: np.ndarray
    dan_trace: np.ndarray
    memory_u: np.ndarray
    memory_w: np.ndarray

    @classmethod
    def fresh(cls, connectivity: M2EConnectivity, *, eta: float = 0.1) -> "PlasticityState":
        n = len(connectivity.pre_slot)
        return cls(
            connectivity=connectivity,
            eta=float(eta),
            kc_trace=np.zeros(n, dtype=np.float64),
            dan_trace=np.zeros(2, dtype=np.float64),
            memory_u=np.zeros(n, dtype=np.float64),
            memory_w=np.zeros(n, dtype=np.float64),
        )

    @property
    def efficacy(self) -> np.ndarray:
        return 1.0 + self.memory_w

    def digest(self) -> str:
        digest = hashlib.sha256()
        for value in (self.kc_trace, self.dan_trace, self.memory_u, self.memory_w):
            digest.update(np.ascontiguousarray(value).tobytes())
        return digest.hexdigest()

    def weight_digest(self) -> str:
        return hashlib.sha256(
            np.ascontiguousarray(self.connectivity.baseline_weight * self.efficacy).tobytes()
        ).hexdigest()

    def advance_bin(
        self,
        kc_rate: np.ndarray,
        dan: np.ndarray,
        *,
        dt_seconds: float = 0.01,
        learning: bool = True,
        frozen: bool = False,
    ) -> None:
        values = np.asarray(kc_rate, dtype=np.float64)
        dan_values = np.asarray(dan, dtype=np.float64)
        if values.shape != (self.connectivity.n_kc,):
            raise ValueError("KC rate does not match connectivity population")
        if dan_values.shape != (2,) or np.any(dan_values < 0):
            raise ValueError("DAN input must be a nonnegative two-channel vector")
        advance_rule(
            self.kc_trace,
            self.dan_trace,
            self.memory_u,
            self.memory_w,
            values[self.connectivity.pre_slot],
            dan_values,
            self.connectivity.gain,
            dt_seconds,
            self.eta,
            learning=learning,
            frozen=frozen,
        )

    def blank(self, seconds: float, *, frozen: bool = False) -> None:
        steps = round(seconds / 0.01)
        if steps < 0 or not np.isclose(steps * 0.01, seconds):
            raise ValueError("blank interval must align to 10 ms")
        zero_kc = np.zeros(self.connectivity.n_kc)
        zero_dan = np.zeros(2)
        for _ in range(steps):
            self.advance_bin(zero_kc, zero_dan, learning=False, frozen=frozen)


def train_event(
    state: PlasticityState,
    sensory: E1Result,
    *,
    valence: str,
    timing: str = "paired",
    dan_amplitude: float = 1.0,
    frozen: bool = False,
    omitted: bool = False,
    washout_seconds: float = 6.0,
) -> None:
    if sensory.kc_trace is None or sensory.kc_trace.shape[0] != 200:
        raise ValueError("M2-E requires a 10 ms KC trace from the fixed E1 timeline")
    if valence not in VALENCE_DAN or timing not in {"paired", "unpaired"}:
        raise ValueError("invalid valence or timing")
    dan_index = VALENCE_DAN[valence]
    zero_dan = np.zeros(2)
    for slot, kc_rate in enumerate(sensory.kc_trace):
        center_ms = (slot + 0.5) * 10.0
        dan = zero_dan.copy()
        if timing == "paired" and 900.0 <= center_ms < 1000.0 and not omitted:
            dan[dan_index] = dan_amplitude
        state.advance_bin(kc_rate, dan, frozen=frozen)
    if timing == "unpaired":
        # Odor offset is 1000 ms; the sensory trial ends at 2000 ms.  Five more
        # blank seconds place the DAN pulse six seconds after odor offset.
        state.blank(5.0, frozen=frozen)
        for _ in range(10):
            dan = zero_dan.copy()
            if not omitted:
                dan[dan_index] = dan_amplitude
            state.advance_bin(np.zeros(sensory.kc_trace.shape[1]), dan, frozen=frozen)
    state.blank(washout_seconds, frozen=frozen)


def raw_readout(
    connectivity: M2EConnectivity,
    kc_mean_rate: np.ndarray,
    efficacy: np.ndarray,
) -> dict[str, np.ndarray]:
    rate = np.asarray(kc_mean_rate, dtype=np.float64)
    factor = np.asarray(efficacy, dtype=np.float64)
    if factor.shape != connectivity.baseline_weight.shape:
        raise ValueError("efficacy shape mismatch")
    contribution = (
        connectivity.baseline_weight * factor * rate[connectivity.pre_slot]
    )
    target_numerator = np.bincount(
        connectivity.target_slot, weights=contribution, minlength=3
    )
    target_denominator = np.bincount(
        connectivity.target_slot, weights=connectivity.baseline_weight, minlength=3
    )
    per_target = target_numerator / target_denominator
    channel_numerator = np.bincount(
        connectivity.edge_channel, weights=contribution, minlength=2
    )
    channel_denominator = np.bincount(
        connectivity.edge_channel, weights=connectivity.baseline_weight, minlength=2
    )
    return {"per_target": per_target, "channels": channel_numerator / channel_denominator}


def normalized_valence(current: np.ndarray, baseline: np.ndarray) -> float:
    current_values = np.asarray(current, dtype=np.float64)
    baseline_values = np.asarray(baseline, dtype=np.float64)
    if current_values.shape != (2,) or baseline_values.shape != (2,):
        raise ValueError("readout needs avoidance and approach channels")
    if np.any(baseline_values <= 1e-12):
        raise ValueError("baseline MBON channel is not drivable")
    normalized = current_values / baseline_values
    return float(normalized[1] - normalized[0])


class SensoryCache:
    """Cache only E1-G outputs under an exact topology/parameter/input key."""

    def __init__(
        self,
        topology: BipartiteTopology,
        parameters: M1Parameters,
        mapping: APLMapping,
        *,
        dt_ms: float = 0.025,
    ) -> None:
        self.topology = topology
        self.parameters = parameters
        self.mapping = mapping
        self.dt_ms = float(dt_ms)
        self._items: dict[str, E1Result] = {}
        self.builds = 0
        self.hits = 0

    def key(self, pulse: np.ndarray) -> str:
        digest = hashlib.sha256()
        digest.update(b"STONKFLY-M2-E/E1-G-CACHE/v1")
        digest.update(self.topology.sha256.encode())
        digest.update(json.dumps(self.parameters.__dict__, sort_keys=True).encode())
        digest.update(np.float64(self.dt_ms).tobytes())
        digest.update(np.ascontiguousarray(pulse, dtype=np.float64).tobytes())
        return digest.hexdigest()

    def get(self, pulse: np.ndarray) -> tuple[str, E1Result]:
        key = self.key(pulse)
        if key in self._items:
            self.hits += 1
            return key, self._items[key]
        result = simulate_e1(
            self.topology,
            self.parameters,
            self.mapping,
            model="G",
            dt_ms=self.dt_ms,
            duration_ms=2000.0,
            pulse=np.asarray(pulse, dtype=np.float64),
            pulse_window_ms=(500.0, 1000.0),
            trace_bin_ms=10.0,
            record_kc_trace=True,
        )
        if not result.finite or result.kc_trace is None:
            raise FloatingPointError("nonfinite E1-G sensory cache item")
        for array in (
            result.integrated_rate, result.population_trace, result.group_trace,
            result.apl_trace, result.final_u, result.final_apl, result.kc_trace,
        ):
            array.setflags(write=False)
        self._items[key] = result
        self.builds += 1
        return key, result

    def discard(self, key: str) -> None:
        """Release a consumed item; formal M2-E streams one stimulus at a time."""
        if key not in self._items:
            raise KeyError("cannot discard an absent sensory cache key")
        del self._items[key]
