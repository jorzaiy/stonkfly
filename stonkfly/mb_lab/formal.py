"""Freeze-v5 formal task kernel and statistics. Never selects parameters."""

from __future__ import annotations

import hashlib

import numpy as np

from .dynamics import M1Parameters, simulate
from .metrics import (
    exact_sign_flip_pvalue,
    frozen_neighbor_matches,
    frozen_separation_statistic,
    jaccard,
    separation_permutation_pvalue,
    sparsity_views,
    topology_rank_pvalue,
)
from .topology import BipartiteTopology


def domain_seed(label: str) -> tuple[str, int]:
    encoded = label.encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    return digest.hex(), int.from_bytes(digest[:8], "little", signed=False)


def simulate_collection_chunk(
    topology: BipartiteTopology,
    parameters: M1Parameters,
    collection: np.ndarray,
    *,
    dt_ms: float,
    apl_enabled: bool,
) -> dict[str, np.ndarray]:
    values = np.asarray(collection, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != 10 or values.shape[2] != topology.n_pn:
        raise ValueError("collection must be odor x 10 x PN")
    shape = values.shape[:2] + (topology.n_kc,)
    response = np.empty(shape, dtype=np.float64)
    tail = np.empty(shape, dtype=np.float64)
    tail_first_total = np.empty(values.shape[:2], dtype=np.float64)
    tail_second_total = np.empty(values.shape[:2], dtype=np.float64)
    state_error = np.empty(values.shape[:2], dtype=np.float64)
    finite = np.empty(values.shape[:2], dtype=bool)
    zero = np.zeros(topology.n_pn, dtype=np.float64)
    for odor in range(values.shape[0]):
        for trial in range(10):
            pulse = values[odor, trial]
            result = simulate(
                topology,
                parameters,
                dt_ms=dt_ms,
                duration_ms=2000.0,
                pn_rate=lambda time, on=pulse: on if 500.0 <= time < 1000.0 else zero,
                apl_enabled=apl_enabled,
                integration_window_ms=(500.0, 1000.0),
                tail_window_ms=(1500.0, 2000.0),
            )
            response[odor, trial] = result.integrated_rate
            tail[odor, trial] = result.tail_integrated_rate
            tail_first_total[odor, trial] = result.tail_first_integrated_rate.sum()
            tail_second_total[odor, trial] = result.tail_second_integrated_rate.sum()
            state_error[odor, trial] = max(
                float(np.max(np.abs(result.final_u))), abs(result.final_apl)
            )
            finite[odor, trial] = result.finite
    return {
        "response_integral": response,
        "tail_integral": tail,
        "tail_first_total": tail_first_total,
        "tail_second_total": tail_second_total,
        "state_error": state_error,
        "finite": finite,
    }


def simulate_zero_chunk(
    topology: BipartiteTopology,
    parameters: M1Parameters,
    *,
    dt_ms: float,
    apl_enabled: bool,
    repeats: int = 10,
) -> dict[str, np.ndarray]:
    zero = np.zeros(topology.n_pn, dtype=np.float64)
    responses = []
    states = []
    finite = []
    for _ in range(repeats):
        result = simulate(
            topology,
            parameters,
            dt_ms=dt_ms,
            duration_ms=2000.0,
            pn_rate=lambda _time: zero,
            apl_enabled=apl_enabled,
            integration_window_ms=(500.0, 1000.0),
            tail_window_ms=(1500.0, 2000.0),
        )
        responses.append(result.integrated_rate)
        states.append(max(float(np.max(np.abs(result.final_u))), abs(result.final_apl)))
        finite.append(result.finite)
    return {
        "response_integral": np.asarray(responses),
        "state_error": np.asarray(states),
        "finite": np.asarray(finite),
    }


def active_from_integral(response: np.ndarray, threshold_mean: float = 0.05) -> np.ndarray:
    return np.asarray(response) >= threshold_mean * 500.0


def science_statistics(
    b_on: dict[str, np.ndarray],
    b_off: dict[str, np.ndarray],
    controls: list[dict[str, np.ndarray]],
    base_pn: np.ndarray,
    *,
    permutation_seed: int,
) -> dict[str, object]:
    active = active_from_integral(b_on["response_integral"])
    fractions = sparsity_views(active)
    primary = fractions["covered_1907_primary"]
    odor_fraction = np.median(primary, axis=1)
    s1 = bool(
        np.all((odor_fraction >= 0.02) & (odor_fraction <= 0.10))
        and 0.04 <= np.median(odor_fraction) <= 0.08
        and np.all(primary > 0)
        and np.all(primary <= 0.20)
    )
    matches = frozen_neighbor_matches(base_pn)
    sep = frozen_separation_statistic(active, matches)
    p_sep = separation_permutation_pvalue(active, matches, seed=permutation_seed)
    reliability = np.asarray(sep["reliability"])
    s2 = bool(
        np.median(reliability) >= 0.60
        and np.count_nonzero(reliability >= 0.50) >= 14
        and float(sep["t_sep"]) >= 0.20
        and p_sep <= 0.01
    )
    tail_active = active_from_integral(b_on["tail_integral"])
    tail_fraction = tail_active.mean(axis=2)
    state_error = b_on["state_error"]
    nonrising = b_on["tail_second_total"] <= b_on["tail_first_total"]
    s3 = bool(
        np.all(tail_fraction <= 0.01)
        and np.mean(state_error <= 0.05) >= 0.95
        and np.all(state_error <= 0.10)
        and np.all(nonrising)
    )
    off_fraction = active_from_integral(b_off["response_integral"]).mean(axis=2)
    deltas = np.median(off_fraction - primary, axis=1)
    p_apl = exact_sign_flip_pvalue(deltas)
    s4 = bool(
        np.median(deltas) >= 0.02
        and np.count_nonzero(deltas > 0) >= 13
        and p_apl <= 0.01
    )
    control_t = []
    control_missing = []
    for index, item in enumerate(controls):
        try:
            statistic = frozen_separation_statistic(
                active_from_integral(item["response_integral"]), matches
            )
            value = float(statistic["t_sep"])
            if not np.isfinite(value):
                raise ValueError("nonfinite T_C")
        except ValueError as error:
            value = float("inf")
            control_missing.append({"index": index, "reason": str(error)})
        control_t.append(value)
    control_array = np.asarray(control_t)
    # Freeze-v5 deliberately assigns undefined-but-finite C statistics +inf.
    # That is a conservative loss for B, not a reason to discard/redraw C.
    if control_array.shape != (39,) or np.isnan(control_array).any():
        raise ValueError("topology controls must contain 39 non-NaN statistics")
    p_topology = float((1 + np.count_nonzero(control_array >= float(sep["t_sep"]))) / 40)
    topology_effect = float(sep["t_sep"] - np.median(control_array))
    s5 = bool(p_topology <= 0.05 and topology_effect >= 0.05)
    return {
        "S1_B": s1,
        "S2_B": s2,
        "S3_B": s3,
        "S4_B_APL": s4,
        "S5_B_VS_39C": s5,
        "covered_fraction": primary,
        "all_2045_fraction": fractions["all_2045_secondary"],
        "separation": sep,
        "separation_p": p_sep,
        "apl_deltas": deltas,
        "apl_p": p_apl,
        "control_t_sep": control_array,
        "control_missing": control_missing,
        "topology_p": p_topology,
        "topology_effect": topology_effect,
    }


def i8_unit_errors(candidate: np.ndarray, reference: np.ndarray) -> dict[str, np.ndarray]:
    q_candidate = np.asarray(candidate, dtype=np.float64) / 500.0
    q_reference = np.asarray(reference, dtype=np.float64) / 500.0
    difference = q_candidate - q_reference
    rms_error = np.sqrt(np.mean(difference**2, axis=-1))
    scale = np.maximum(np.sqrt(np.mean(q_reference**2, axis=-1)), 0.05)
    active_candidate = active_from_integral(candidate)
    active_reference = active_from_integral(reference)
    fraction_error = np.abs(active_candidate.mean(axis=-1) - active_reference.mean(axis=-1))
    jac = np.empty(fraction_error.shape, dtype=np.float64)
    for index in np.ndindex(jac.shape):
        left, right = active_candidate[index], active_reference[index]
        jac[index] = 1.0 if not (left | right).any() else jaccard(left, right)
    return {"active_fraction_error": fraction_error, "nrmse": rms_error / scale, "jaccard": jac}


def i8_summary(
    paths: dict[str, dict[float, np.ndarray]],
    science_vectors: dict[float, tuple[bool, ...]],
) -> dict[str, object]:
    """Conservative freeze-v5 maximum-error aggregation over every unit."""
    summary: dict[str, object] = {"path_count": len(paths), "per_dt": {}}
    halving_ratios = []
    floor_count = 0
    halving_failures = []
    for candidate_dt, af_limit, nrmse_limit, jaccard_limit in (
        (0.1, 0.005, 0.05, 0.90),
        (0.05, 0.0025, 0.025, 0.95),
    ):
        errors = [i8_unit_errors(values[candidate_dt], values[0.025]) for values in paths.values()]
        max_af = max(float(item["active_fraction_error"].max()) for item in errors)
        max_nrmse = max(float(item["nrmse"].max()) for item in errors)
        min_jaccard = min(float(item["jaccard"].min()) for item in errors)
        summary["per_dt"][str(candidate_dt)] = {
            "max_active_fraction_error": max_af,
            "max_nrmse": max_nrmse,
            "min_jaccard": min_jaccard,
            "pass": bool(max_af <= af_limit and max_nrmse <= nrmse_limit and min_jaccard >= jaccard_limit),
        }
    for name, values in paths.items():
        q100, q050, q025 = (np.asarray(values[dt]) / 500.0 for dt in (0.1, 0.05, 0.025))
        scale = np.maximum(np.sqrt(np.mean(q025**2, axis=-1)), 0.05)
        coarse = np.sqrt(np.mean((q100 - q050) ** 2, axis=-1)) / scale
        fine = np.sqrt(np.mean((q050 - q025) ** 2, axis=-1)) / scale
        floor = (coarse <= 1e-4) & (fine <= 1e-4)
        floor_count += int(floor.sum())
        ratio = np.divide(fine, coarse, out=np.full_like(fine, np.inf), where=coarse > 0)
        valid = floor | (fine <= 0.6 * coarse)
        if not np.all(valid):
            halving_failures.append({"path": name, "count": int((~valid).sum())})
        halving_ratios.extend(ratio[~floor & np.isfinite(ratio)].ravel().tolist())
    classification_equal = len(set(science_vectors.values())) == 1
    pass_value = bool(
        all(item["pass"] for item in summary["per_dt"].values())
        and not halving_failures
        and classification_equal
    )
    summary.update(
        {
            "halving_floor_count": floor_count,
            "halving_max_finite_ratio": max(halving_ratios, default=0.0),
            "halving_failures": halving_failures,
            "science_vectors": {str(key): list(value) for key, value in science_vectors.items()},
            "science_classification_equal": classification_equal,
            "pass": pass_value,
        }
    )
    return summary
