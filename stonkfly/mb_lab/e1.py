"""M1-E1 exploratory global, compartment-local, and APL-off dynamics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .dynamics import M1Parameters
from .topology import BipartiteTopology

GROUP_NAMES = ("gamma", "alpha_beta", "alpha_prime_beta_prime", "untyped")


@dataclass(frozen=True)
class APLMapping:
    apl_node_index: int
    apl_source_id: int
    group_index: np.ndarray
    group_names: tuple[str, ...]
    kc_to_apl_contacts: np.ndarray
    apl_to_kc_contacts: np.ndarray
    output_multiplier: np.ndarray

    def validate(self, n_kc: int) -> None:
        arrays = (
            self.group_index,
            self.kc_to_apl_contacts,
            self.apl_to_kc_contacts,
            self.output_multiplier,
        )
        if any(np.asarray(item).shape != (n_kc,) for item in arrays):
            raise ValueError("APL mapping arrays must match the KC population")
        if not np.isfinite(self.output_multiplier).all():
            raise ValueError("APL output multipliers must be finite")
        if np.any(self.kc_to_apl_contacts <= 0) or np.any(self.apl_to_kc_contacts <= 0):
            raise ValueError("every E1 KC must be reciprocally connected to APL")
        if np.any(self.group_index < 0) or np.any(self.group_index >= len(self.group_names)):
            raise ValueError("invalid local group assignment")
        for group in range(len(self.group_names)):
            selected = self.group_index == group
            if not selected.any():
                raise ValueError(f"empty local group: {self.group_names[group]}")
            if not math.isclose(float(self.output_multiplier[selected].mean()), 1.0, abs_tol=1e-12):
                raise ValueError("local output multipliers must have group mean one")


def kc_group(cell_type: str) -> int:
    if cell_type.startswith("KCg"):
        return 0
    if cell_type.startswith("KCab"):
        return 1
    if cell_type.startswith("KCa'b'"):
        return 2
    return 3


def build_apl_mapping(
    graph_path: Path,
    source: dict[str, np.ndarray],
    kc_node_index: np.ndarray,
    *,
    side: str = "R",
    contact_scale: float = 0.275,
) -> APLMapping:
    kc_nodes = np.asarray(kc_node_index, dtype=np.int32)
    apl_nodes = np.flatnonzero(
        (source["cell_type"] == "APL") & (source["soma_side"] == side)
    )
    if len(apl_nodes) != 1:
        raise ValueError(f"expected exactly one {side} APL")
    apl = int(apl_nodes[0])
    with np.load(graph_path, allow_pickle=False) as graph:
        ptr = np.asarray(graph["ptr"], dtype=np.int64)
        post = np.asarray(graph["post"], dtype=np.int32)
        weight = np.asarray(graph["weight"], dtype=np.float64)
    if contact_scale <= 0:
        raise ValueError("contact scale must be positive")
    apl_out = {
        int(post[edge]): abs(float(weight[edge])) / contact_scale
        for edge in range(int(ptr[apl]), int(ptr[apl + 1]))
    }
    kc_to_apl = np.empty(len(kc_nodes), dtype=np.float64)
    apl_to_kc = np.empty(len(kc_nodes), dtype=np.float64)
    groups = np.empty(len(kc_nodes), dtype=np.int8)
    for slot, node_value in enumerate(kc_nodes):
        node = int(node_value)
        hits = [
            edge
            for edge in range(int(ptr[node]), int(ptr[node + 1]))
            if int(post[edge]) == apl
        ]
        if len(hits) != 1 or node not in apl_out:
            raise ValueError(f"KC node {node} lacks one reciprocal APL pair")
        if weight[hits[0]] <= 0:
            raise ValueError("KC-to-APL edge must have positive sign")
        apl_edge_weights = [
            weight[edge]
            for edge in range(int(ptr[apl]), int(ptr[apl + 1]))
            if int(post[edge]) == node
        ]
        if len(apl_edge_weights) != 1 or apl_edge_weights[0] >= 0:
            raise ValueError("APL-to-KC edge must have negative sign")
        kc_to_apl[slot] = abs(float(weight[hits[0]])) / contact_scale
        apl_to_kc[slot] = apl_out[node]
        groups[slot] = kc_group(str(source["cell_type"][node]))
    multipliers = np.empty(len(kc_nodes), dtype=np.float64)
    for group in range(len(GROUP_NAMES)):
        selected = groups == group
        multipliers[selected] = apl_to_kc[selected] / apl_to_kc[selected].mean()
    mapping = APLMapping(
        apl_node_index=apl,
        apl_source_id=int(source["ids"][apl]),
        group_index=groups,
        group_names=GROUP_NAMES,
        kc_to_apl_contacts=kc_to_apl,
        apl_to_kc_contacts=apl_to_kc,
        output_multiplier=multipliers,
    )
    mapping.validate(len(kc_nodes))
    return mapping


@dataclass(frozen=True)
class E1Result:
    integrated_rate: np.ndarray
    population_trace: np.ndarray
    group_trace: np.ndarray
    apl_trace: np.ndarray
    final_u: np.ndarray
    final_apl: np.ndarray
    finite: bool
    kc_trace: np.ndarray | None = None


def simulate_e1(
    topology: BipartiteTopology,
    parameters: M1Parameters,
    mapping: APLMapping,
    *,
    model: str,
    dt_ms: float,
    duration_ms: float,
    pulse: np.ndarray,
    pulse_window_ms: tuple[float, float] = (500.0, 1000.0),
    trace_bin_ms: float = 10.0,
    record_kc_trace: bool = False,
) -> E1Result:
    parameters.validate()
    mapping.validate(topology.n_kc)
    if model not in {"G", "L", "O"}:
        raise ValueError("model must be G, L, or O")
    if pulse.shape != (topology.n_pn,) or not np.isfinite(pulse).all():
        raise ValueError("invalid PN pulse")
    steps = round(duration_ms / dt_ms)
    bin_steps = round(trace_bin_ms / dt_ms)
    if (
        steps < 1
        or bin_steps < 1
        or not math.isclose(steps * dt_ms, duration_ms, abs_tol=1e-9)
        or not math.isclose(bin_steps * dt_ms, trace_bin_ms, abs_tol=1e-9)
        or steps % bin_steps
    ):
        raise ValueError("duration and trace bins must align to dt")
    start, end = pulse_window_ms
    if not 0 <= start < end <= duration_ms:
        raise ValueError("invalid pulse window")
    zero = np.zeros(topology.n_pn, dtype=np.float64)
    drive_zero = topology.project(zero)
    drive_pulse = topology.project(pulse)
    u = np.zeros(topology.n_kc, dtype=np.float64)
    integrated = np.zeros(topology.n_kc, dtype=np.float64)
    group_count = len(mapping.group_names)
    group_sizes = np.bincount(mapping.group_index, minlength=group_count).astype(np.float64)
    group_contact_sums = np.bincount(
        mapping.group_index,
        weights=mapping.kc_to_apl_contacts,
        minlength=group_count,
    )
    apl = np.zeros(1 if model in {"G", "O"} else group_count, dtype=np.float64)
    bins = steps // bin_steps
    population_trace = np.zeros(bins, dtype=np.float64)
    group_trace = np.zeros((bins, group_count), dtype=np.float64)
    apl_trace = np.zeros((bins, len(apl)), dtype=np.float64)
    kc_trace = np.zeros((bins, topology.n_kc), dtype=np.float64) if record_kc_trace else None
    bin_population = 0.0
    bin_groups = np.zeros(group_count, dtype=np.float64)
    bin_apl = np.zeros(len(apl), dtype=np.float64)
    bin_kc = np.zeros(topology.n_kc, dtype=np.float64) if record_kc_trace else None
    k_step = dt_ms / parameters.tau_k_ms
    a_step = dt_ms / parameters.tau_apl_ms
    for step in range(steps):
        time_ms = step * dt_ms
        drive = drive_pulse if start <= time_ms < end else drive_zero
        if model == "G":
            inhibition = parameters.apl_gain * apl[0]
        elif model == "L":
            inhibition = (
                parameters.apl_gain
                * mapping.output_multiplier
                * apl[mapping.group_index]
            )
        else:
            inhibition = 0.0
        u += k_step * (-u + parameters.input_scale * drive - inhibition)
        rate = np.maximum(u - parameters.threshold, 0.0)
        if model == "G":
            apl[0] += a_step * (-apl[0] + float(rate.mean()))
        elif model == "L":
            targets = np.bincount(
                mapping.group_index,
                weights=mapping.kc_to_apl_contacts * rate,
                minlength=group_count,
            ) / group_contact_sums
            apl += a_step * (-apl + targets)
        if start <= time_ms < end:
            integrated += rate * dt_ms
        bin_population += float(rate.mean())
        bin_groups += np.bincount(
            mapping.group_index, weights=rate, minlength=group_count
        ) / group_sizes
        bin_apl += apl
        if bin_kc is not None:
            bin_kc += rate
        if (step + 1) % bin_steps == 0:
            slot = step // bin_steps
            population_trace[slot] = bin_population / bin_steps
            group_trace[slot] = bin_groups / bin_steps
            apl_trace[slot] = bin_apl / bin_steps
            if kc_trace is not None and bin_kc is not None:
                kc_trace[slot] = bin_kc / bin_steps
            bin_population = 0.0
            bin_groups.fill(0.0)
            bin_apl.fill(0.0)
            if bin_kc is not None:
                bin_kc.fill(0.0)
    finite = bool(
        np.isfinite(u).all()
        and np.isfinite(integrated).all()
        and np.isfinite(population_trace).all()
        and np.isfinite(group_trace).all()
        and np.isfinite(apl).all()
    )
    return E1Result(
        integrated_rate=integrated,
        population_trace=population_trace,
        group_trace=group_trace,
        apl_trace=apl_trace,
        final_u=u,
        final_apl=apl.copy(),
        finite=finite,
        kc_trace=kc_trace,
    )


def population_sparseness(rate: np.ndarray) -> np.ndarray:
    values = np.asarray(rate, dtype=np.float64)
    n = values.shape[-1]
    numerator = np.mean(values, axis=-1) ** 2
    denominator = np.mean(values**2, axis=-1)
    ratio = np.divide(numerator, denominator, out=np.ones_like(numerator), where=denominator > 0)
    return (1.0 - ratio) / (1.0 - 1.0 / n)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return 1.0 if not np.any(left) and not np.any(right) else 0.0
    return float(np.dot(left, right) / denominator)
