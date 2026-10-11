"""Small, deterministic protocol helpers for the V2-C0 experiment.

This module contains no stimulus constructors and no execution/model imports.
The calibration and holdout processes intentionally own their disjoint mask
construction code.  Keeping these helpers data-only also makes the process
boundary auditable.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


GAINS = (0.125, 0.25, 0.5, 1.0, 2.0)
D_CAL = (0, 1, 2, 3, 4, 5, 6)
D_HOLD = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5)
DT_CAL = 0.05
DT_HOLD = (0.05, 0.025)
SPARSE_LOW = 41
SPARSE_HIGH = 812


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_id_sha256(ids: Iterable[int]) -> str:
    """Canonical source-id hash used by Q1 and the C0 artifact schema."""
    return sha256_bytes(np.asarray(list(ids), dtype="<u8").tobytes())


def json_sha256(value) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode())


def symmetric_relative_difference(a: float, b: float) -> float:
    den = abs(float(a)) + abs(float(b))
    if den == 0.0:
        return 0.0
    return 2.0 * abs(float(a) - float(b)) / den


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    aa, bb = set(a), set(b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def average_ranks(values: Iterable[float]) -> np.ndarray:
    x = np.asarray(list(values), dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def spearman(values_x: Iterable[float], values_y: Iterable[float]) -> float:
    x, y = average_ranks(values_x), average_ranks(values_y)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0 if not np.array_equal(x, y) else 1.0
    return float(np.corrcoef(x, y)[0, 1])


def classify_kc(active_count: int) -> str:
    n = int(active_count)
    if n <= 10:
        return "fixed-low-like"
    if n <= 40:
        return "sub_sparse"
    if n <= SPARSE_HIGH:
        return "sparse"
    return "dense"


def calibration_trial_count() -> int:
    # Independent trials only: 7 uniform + 2 calibration patterns, each
    # repeated twice, per gain.  History ramps are reported separately.
    return len(GAINS) * (len(D_CAL) + 2) * 2


def calibration_trials_per_gain() -> int:
    """Number of independent C0-A trial records for one gamma."""
    return (len(D_CAL) + 2) * 2


def calibration_ramp_count() -> int:
    """Number of history ramps across the five-gain calibration grid."""
    return len(GAINS) * 2


def holdout_trial_count() -> int:
    return len(D_HOLD) + 4


def choose_gain(qualifying: Iterable[float]) -> float | None:
    values = sorted(float(x) for x in qualifying)
    if not values:
        return None
    return min(values, key=lambda g: (abs(math.log2(g)), g))


def selection_verdict(qualifying: Iterable[float]) -> dict:
    values = [float(x) for x in qualifying]
    selected = choose_gain(values)
    if selected is None:
        return {"calibration_verdict": "C0_NO_QUALIFYING_GAIN", "selected_gamma": None, "qualifying_gains": values}
    if selected in (GAINS[0], GAINS[-1]):
        return {"calibration_verdict": "C0_GRID_BOUNDARY", "selected_gamma": selected, "qualifying_gains": values}
    return {"calibration_verdict": "C0_SELECTION_READY", "selected_gamma": selected, "qualifying_gains": values}


def formal_verdict(*, implementation_ok: bool, numerical_ok: bool,
                   calibration_verdict: str, holdout_ok: bool | None) -> str:
    """Apply §7 precedence; absence of a required artifact is not a pass."""
    if not implementation_ok:
        return "C0_IMPLEMENTATION_FAILURE"
    if not numerical_ok:
        return "C0_NUMERICAL_FAILURE"
    if calibration_verdict == "C0_NO_QUALIFYING_GAIN":
        return "C0_NO_QUALIFYING_GAIN"
    if calibration_verdict == "C0_GRID_BOUNDARY":
        return "C0_GRID_BOUNDARY"
    if calibration_verdict != "C0_SELECTION_READY":
        return "C0_IMPLEMENTATION_FAILURE"
    if holdout_ok is None or not holdout_ok:
        return "C0_HOLDOUT_FAILURE"
    return "C0_SYNTHETIC_QUALIFIED"


def artifact_sha256(paths: Mapping[str, Path]) -> dict[str, str]:
    return {str(name): sha256_file(path) for name, path in sorted(paths.items()) if Path(path).exists()}
