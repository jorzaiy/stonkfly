"""Candidate continuous-rate PN/KC/APL equations for M1 toy qualification.

The equation form is reviewable, but no parameter set in this module is
authorized for formal M1 data.  Callers must provide every parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from .topology import BipartiteTopology


@dataclass(frozen=True)
class M1Parameters:
    tau_k_ms: float
    tau_apl_ms: float
    input_scale: float
    apl_gain: float
    threshold: float
    active_integral_threshold: float

    def validate(self) -> None:
        values = (
            self.tau_k_ms,
            self.tau_apl_ms,
            self.input_scale,
            self.apl_gain,
            self.threshold,
            self.active_integral_threshold,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("all M1 parameters must be finite")
        if self.tau_k_ms <= 0 or self.tau_apl_ms <= 0:
            raise ValueError("time constants must be positive")
        if self.input_scale < 0 or self.apl_gain < 0:
            raise ValueError("input scale and APL gain must be nonnegative")
        if self.active_integral_threshold <= 0:
            raise ValueError("active threshold must be positive")


@dataclass(frozen=True)
class SimulationResult:
    integrated_rate: np.ndarray
    tail_integrated_rate: np.ndarray
    tail_first_integrated_rate: np.ndarray
    tail_second_integrated_rate: np.ndarray
    active: np.ndarray
    final_u: np.ndarray
    final_apl: float
    finite: bool
    dt_ms: float
    duration_ms: float


def simulate(
    topology: BipartiteTopology,
    parameters: M1Parameters,
    *,
    dt_ms: float,
    duration_ms: float,
    pn_rate: Callable[[float], np.ndarray],
    apl_enabled: bool = True,
    integration_window_ms: tuple[float, float] | None = None,
    tail_window_ms: tuple[float, float] | None = None,
) -> SimulationResult:
    parameters.validate()
    if not math.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError("dt_ms must be positive and finite")
    if not math.isfinite(duration_ms) or duration_ms <= 0:
        raise ValueError("duration_ms must be positive and finite")
    steps = round(duration_ms / dt_ms)
    if steps < 1 or not math.isclose(steps * dt_ms, duration_ms, abs_tol=1e-9):
        raise ValueError("duration_ms must contain an integer number of steps")
    integration_window_ms = integration_window_ms or (0.0, duration_ms)
    tail_window_ms = tail_window_ms or (duration_ms, duration_ms)
    for name, window in (
        ("integration", integration_window_ms),
        ("tail", tail_window_ms),
    ):
        if (
            len(window) != 2
            or not 0 <= window[0] <= window[1] <= duration_ms
            or not math.isclose(window[0] / dt_ms, round(window[0] / dt_ms), abs_tol=1e-9)
            or not math.isclose(window[1] / dt_ms, round(window[1] / dt_ms), abs_tol=1e-9)
        ):
            raise ValueError(f"{name} window must align to dt within the trial")
    u = np.zeros(topology.n_kc, dtype=np.float64)
    integrated = np.zeros(topology.n_kc, dtype=np.float64)
    tail_integrated = np.zeros(topology.n_kc, dtype=np.float64)
    tail_first = np.zeros(topology.n_kc, dtype=np.float64)
    tail_second = np.zeros(topology.n_kc, dtype=np.float64)
    tail_midpoint = 0.5 * (tail_window_ms[0] + tail_window_ms[1])
    apl = 0.0
    k_step = dt_ms / parameters.tau_k_ms
    a_step = dt_ms / parameters.tau_apl_ms
    gain = parameters.apl_gain if apl_enabled else 0.0
    cached_pn: np.ndarray | None = None
    cached_drive: np.ndarray | None = None
    for step in range(steps):
        time_ms = step * dt_ms
        pn_values = np.asarray(pn_rate(time_ms), dtype=np.float64)
        if cached_pn is None or not np.array_equal(pn_values, cached_pn):
            cached_pn = pn_values.copy()
            cached_drive = topology.project(cached_pn)
        assert cached_drive is not None
        input_drive = cached_drive
        u += k_step * (
            -u + parameters.input_scale * input_drive - gain * apl
        )
        rate = np.maximum(u - parameters.threshold, 0.0)
        apl += a_step * (-apl + float(rate.mean()))
        if integration_window_ms[0] <= time_ms < integration_window_ms[1]:
            integrated += rate * dt_ms
        if tail_window_ms[0] <= time_ms < tail_window_ms[1]:
            tail_integrated += rate * dt_ms
            if time_ms < tail_midpoint:
                tail_first += rate * dt_ms
            else:
                tail_second += rate * dt_ms
    finite = bool(
        np.isfinite(u).all() and np.isfinite(integrated).all() and math.isfinite(apl)
    )
    active = integrated >= parameters.active_integral_threshold
    return SimulationResult(
        integrated_rate=integrated,
        tail_integrated_rate=tail_integrated,
        tail_first_integrated_rate=tail_first,
        tail_second_integrated_rate=tail_second,
        active=active,
        final_u=u,
        final_apl=float(apl),
        finite=finite,
        dt_ms=float(dt_ms),
        duration_ms=float(duration_ms),
    )
