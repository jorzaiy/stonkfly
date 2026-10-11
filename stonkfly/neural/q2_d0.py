"""Pure audit helpers for the frozen V2-Q2-D0 mechanism audit.

Nothing in this module advances neural state or changes graph weights.  The
functions turn already-observed Q1 voltages into an explicitly open-loop,
first-order counterfactual ledger and provide static graph summaries.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np


SPAN_MV = 7.0
TAU_EMIT_MS = 2.2


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def source_id_sha256(values: Iterable[int]) -> str:
    return array_sha256(np.asarray(list(values), dtype="<u8"))


def signed_deviation(voltage, rest, span_mv: float = SPAN_MV):
    """Candidate-C release *deviation*, never absolute transmitter release."""
    if float(span_mv) <= 0 or not np.isfinite(span_mv):
        raise ValueError("span_mv must be finite and positive")
    return np.clip(
        (np.asarray(voltage, dtype=np.float64) - np.asarray(rest, dtype=np.float64))
        / float(span_mv),
        -1.0,
        1.0,
    )


def q1_positive_release(voltage, rest, span_mv: float = SPAN_MV):
    """Frozen Q1 unilateral clamp, expressed independently for D0 auditing."""
    return np.clip(
        (np.asarray(voltage, dtype=np.float64) - np.asarray(rest, dtype=np.float64))
        / float(span_mv),
        0.0,
        1.0,
    )


def bfs_shortest_hops(ptr: np.ndarray, post: np.ndarray, sources: np.ndarray,
                      max_hops: int = 6) -> np.ndarray:
    """Unweighted directed shortest hops, with -1 meaning >max_hops/unreached."""
    ptr = np.asarray(ptr, dtype=np.int64)
    post = np.asarray(post, dtype=np.int32)
    n = len(ptr) - 1
    if ptr.shape != (n + 1,) or ptr[0] != 0 or ptr[-1] != len(post):
        raise ValueError("invalid CSR")
    if max_hops < 0:
        raise ValueError("max_hops must be nonnegative")
    source = np.asarray(sources, dtype=np.int64)
    if np.any(source < 0) or np.any(source >= n):
        raise ValueError("source index out of range")
    hops = np.full(n, -1, dtype=np.int8)
    frontier = np.zeros(n, dtype=bool)
    frontier[source] = True
    hops[frontier] = 0
    degree = np.diff(ptr)
    for hop in range(1, max_hops + 1):
        edge_mask = np.repeat(frontier, degree)
        if not np.any(edge_mask):
            break
        targets = np.unique(post[edge_mask])
        unseen = hops[targets] < 0
        next_frontier = np.zeros(n, dtype=bool)
        next_frontier[targets[unseen]] = True
        if not np.any(next_frontier):
            break
        hops[next_frontier] = hop
        frontier = next_frontier
    return hops


def counterfactual_delivery(
    ptr: np.ndarray,
    post: np.ndarray,
    weight: np.ndarray,
    source_indices: np.ndarray,
    source_integral: np.ndarray,
    target_group: np.ndarray,
    group_count: int,
) -> dict[str, np.ndarray]:
    """Aggregate w_ij * Q_i without feeding it back into neural dynamics."""
    ptr = np.asarray(ptr, dtype=np.int64)
    post = np.asarray(post, dtype=np.int32)
    weight = np.asarray(weight, dtype=np.float64)
    q = np.asarray(source_integral, dtype=np.float64)
    groups = np.asarray(target_group, dtype=np.int64)
    n = len(ptr) - 1
    if q.shape != (n,) or groups.shape != (n,):
        raise ValueError("node array shape mismatch")
    if len(post) != len(weight) or ptr[-1] != len(post):
        raise ValueError("edge array shape mismatch")
    target_signed = np.zeros(n, dtype=np.float64)
    target_absolute = np.zeros(n, dtype=np.float64)
    positive = np.zeros(group_count, dtype=np.float64)
    negative = np.zeros(group_count, dtype=np.float64)
    absolute = np.zeros(group_count, dtype=np.float64)
    for source in np.asarray(source_indices, dtype=np.int64):
        if q[source] == 0.0:
            continue
        sl = slice(int(ptr[source]), int(ptr[source + 1]))
        targets = post[sl]
        values = weight[sl] * q[source]
        np.add.at(target_signed, targets, values)
        np.add.at(target_absolute, targets, np.abs(values))
        target_groups = groups[targets]
        positive += np.bincount(
            target_groups, weights=np.maximum(values, 0.0), minlength=group_count
        )[:group_count]
        negative += np.bincount(
            target_groups, weights=np.minimum(values, 0.0), minlength=group_count
        )[:group_count]
        absolute += np.bincount(
            target_groups, weights=np.abs(values), minlength=group_count
        )[:group_count]
    return {
        "target_signed": target_signed,
        "target_absolute": target_absolute,
        "positive_by_group": positive,
        "negative_by_group": negative,
        "signed_by_group": positive + negative,
        "absolute_by_group": absolute,
    }


def interpretation_flags(
    *,
    lamina_hyperpolarization_observed: bool,
    lamina_q1_output_clamped_zero: bool,
    counterfactual_lamina_reaches_spiking_downstream: bool,
    counterfactual_lamina_positive_net_reaches_spiking_downstream: bool,
    counterfactual_lamina_reaches_kc_directly: bool,
    generic_spiking_received_input: bool,
    generic_spiking_emitted_spikes: bool,
) -> dict[str, bool]:
    values = {
        "lamina_hyperpolarization_observed": bool(lamina_hyperpolarization_observed),
        "lamina_q1_output_clamped_zero": bool(lamina_q1_output_clamped_zero),
        "counterfactual_lamina_reaches_spiking_downstream": bool(counterfactual_lamina_reaches_spiking_downstream),
        "counterfactual_lamina_positive_net_reaches_spiking_downstream": bool(counterfactual_lamina_positive_net_reaches_spiking_downstream),
        "counterfactual_lamina_reaches_kc_directly": bool(counterfactual_lamina_reaches_kc_directly),
        "generic_spiking_received_input": bool(generic_spiking_received_input),
        "generic_spiking_emitted_spikes": bool(generic_spiking_emitted_spikes),
    }
    values["competing_spiking_bottleneck"] = bool(
        values["generic_spiking_received_input"]
        and not values["generic_spiking_emitted_spikes"]
    )
    values["tonic_hypothesis_upgrade_condition"] = bool(
        values["lamina_hyperpolarization_observed"]
        and values["lamina_q1_output_clamped_zero"]
        and values["counterfactual_lamina_reaches_spiking_downstream"]
    )
    return values
