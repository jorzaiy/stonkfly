"""Frozen-form candidate metrics for M1 toy and protocol qualification."""

from __future__ import annotations

import itertools

import numpy as np


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Jaccard inputs must be aligned vectors")
    union = np.count_nonzero(left | right)
    if union == 0:
        return float("nan")
    return float(np.count_nonzero(left & right) / union)


def odor_reliability(active: np.ndarray) -> np.ndarray:
    """Return R_o from five fixed, disjoint pairs for each odor."""
    active = np.asarray(active, dtype=bool)
    if active.ndim != 3 or active.shape[1] != 10:
        raise ValueError("active must have shape (odors, 10, kcs)")
    result = []
    for odor in active:
        pairs = [jaccard(odor[index], odor[index + 1]) for index in range(0, 10, 2)]
        result.append(float(np.median(pairs)))
    return np.asarray(result, dtype=np.float64)


def separation_statistic(active: np.ndarray, matched_odor: np.ndarray) -> dict[str, object]:
    active = np.asarray(active, dtype=bool)
    matched = np.asarray(matched_odor, dtype=np.int64)
    if active.ndim != 3 or active.shape[1] != 10:
        raise ValueError("active must have shape (odors, 10, kcs)")
    if matched.shape != active.shape[:2]:
        raise ValueError("matched_odor must have shape (odors, 10)")
    odors = active.shape[0]
    if np.any(matched < 0) or np.any(matched >= odors):
        raise ValueError("matched odor outside collection")
    if np.any(matched == np.arange(odors)[:, None]):
        raise ValueError("between-odor matches cannot use the same odor")
    reliability = odor_reliability(active)
    between = np.empty(odors, dtype=np.float64)
    for odor in range(odors):
        values = [
            jaccard(active[odor, repeat], active[matched[odor, repeat], repeat])
            for repeat in range(10)
        ]
        between[odor] = np.median(values)
    margins = reliability - between
    return {
        "reliability": reliability,
        "between": between,
        "margins": margins,
        "t_sep": float(np.mean(margins)),
        "median_margin_descriptive": float(np.median(margins)),
    }


def frozen_neighbor_matches(base_pn: np.ndarray, neighbors: int = 3) -> np.ndarray:
    """Return hard between-odor matches with deterministic tie breaking."""
    base = np.asarray(base_pn, dtype=np.float64)
    if base.ndim != 2 or not np.isfinite(base).all() or neighbors < 1:
        raise ValueError("base_pn must be a finite odor-by-PN matrix")
    norms = np.linalg.norm(base, axis=1)
    if np.any(norms == 0) or neighbors >= len(base):
        raise ValueError("each odor must be nonzero and have enough alternatives")
    cosine = base @ base.T / np.outer(norms, norms)
    matches = np.empty((len(base), neighbors), dtype=np.int64)
    indices = np.arange(len(base))
    for odor in range(len(base)):
        candidates = indices[indices != odor]
        order = np.lexsort((candidates, -cosine[odor, candidates]))
        matches[odor] = candidates[order[:neighbors]]
    return matches


def frozen_separation_statistic(
    active: np.ndarray, neighbor_matches: np.ndarray
) -> dict[str, object]:
    """M1 statistic using 3 fixed hard neighbors and aligned repeat indices."""
    active = np.asarray(active, dtype=bool)
    matches = np.asarray(neighbor_matches, dtype=np.int64)
    if active.ndim != 3 or active.shape[1] != 10:
        raise ValueError("active must have shape (odors, 10, kcs)")
    if matches.shape != (active.shape[0], 3):
        raise ValueError("neighbor_matches must have shape (odors, 3)")
    if np.any(matches < 0) or np.any(matches >= active.shape[0]):
        raise ValueError("neighbor match outside collection")
    if np.any(matches == np.arange(active.shape[0])[:, None]):
        raise ValueError("an odor cannot match itself")
    reliability = odor_reliability(active)
    between = np.asarray(
        [
            np.median(
                [
                    jaccard(active[odor, repeat], active[other, repeat])
                    for other in matches[odor]
                    for repeat in range(10)
                ]
            )
            for odor in range(active.shape[0])
        ],
        dtype=np.float64,
    )
    margins = reliability - between
    return {
        "reliability": reliability,
        "between": between,
        "margins": margins,
        "t_sep": float(np.median(margins)),
    }


def separation_permutation_pvalue(
    active: np.ndarray,
    neighbor_matches: np.ndarray,
    *,
    seed: int,
    permutations: int = 9999,
) -> float:
    """One-sided Monte Carlo sign-flip of R_o-D_o at the odor unit."""
    margins = np.asarray(
        frozen_separation_statistic(active, neighbor_matches)["margins"],
        dtype=np.float64,
    )
    if not np.isfinite(margins).all():
        raise ValueError("separation margins must all be finite")
    observed = float(np.mean(margins))
    rng = np.random.Generator(np.random.PCG64(seed))
    bits = rng.integers(
        0, 2, size=(permutations, len(margins)), dtype=np.uint8
    )
    signs = bits.astype(np.float64) * 2.0 - 1.0
    values = np.mean(signs * margins[None, :], axis=1)
    greater = int(np.count_nonzero(values >= observed - 1e-15))
    return float((1 + greater) / (1 + permutations))


def sparsity_views(active_covered: np.ndarray, *, total_kc: int = 2045) -> dict[str, np.ndarray]:
    """Report gated covered-KC and descriptive all-KC denominators together."""
    active = np.asarray(active_covered, dtype=bool)
    if active.ndim < 1 or active.shape[-1] <= 0 or total_kc < active.shape[-1]:
        raise ValueError("invalid covered/full population dimensions")
    counts = active.sum(axis=-1)
    return {
        "covered_1907_primary": counts / active.shape[-1],
        "all_2045_secondary": counts / total_kc,
    }


def exact_sign_flip_pvalue(deltas: np.ndarray) -> float:
    """One-sided exact p-value for the mean paired APL-off effect."""
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.shape != (16,) or not np.isfinite(deltas).all():
        raise ValueError("APL deltas must be 16 finite odor-level values")
    observed = float(deltas.mean())
    greater = 0
    total = 1 << len(deltas)
    for bits in range(total):
        signs = np.fromiter(
            (1.0 if bits & (1 << index) else -1.0 for index in range(len(deltas))),
            dtype=np.float64,
            count=len(deltas),
        )
        if float(np.mean(signs * deltas)) >= observed - 1e-15:
            greater += 1
    return greater / total


def topology_rank_pvalue(real: float, controls: np.ndarray) -> float:
    controls = np.asarray(controls, dtype=np.float64)
    if controls.shape != (39,) or not np.isfinite(controls).all() or not np.isfinite(real):
        raise ValueError("topology test requires one real and 39 finite controls")
    return float((1 + np.count_nonzero(controls >= real)) / 40)


def all_binary_vectors(width: int):
    """Small toy helper; never use for formal KC-sized enumeration."""
    for values in itertools.product((False, True), repeat=width):
        yield np.asarray(values, dtype=bool)
