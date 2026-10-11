"""Isolated V2-C0 counterfactual gain wrapper.

The Q1 ``HybridBrain`` and its binary remain immutable.  C0 creates a
counterfactual edge-weight view for one global ``gamma``: every original edge
whose source is graded receives the same multiplier, while the persistent
weight array exposed by the brain is restored after every segment.  No
cell-type, edge, sign, target, activity, or time-dependent selection exists
here.
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import numpy as np

from stonkfly.neural.v2_brain import HybridBrain


CALIBRATION_GAINS = (0.125, 0.25, 0.5, 1.0, 2.0)


class C0HybridBrain(HybridBrain):
    """Q1 dynamics with one fixed, counterfactual graded-release gain."""

    def __init__(self, path=Path("data/graph.npz"), *, gamma=1.0, **kwargs):
        gamma_decimal = Decimal(str(gamma))
        if not gamma_decimal.is_finite() or gamma_decimal <= 0:
            raise ValueError("gamma must be a finite positive scalar")
        super().__init__(path, **kwargs)
        self.gamma = float(gamma_decimal)
        self.gamma_decimal = format(gamma_decimal, "f")
        edge_source = np.repeat(self.graded_mask.astype(bool), np.diff(self.ptr))
        if edge_source.shape != self.weight.shape:
            raise ValueError("CSR edge/source shape mismatch")
        self._counterfactual_weight = self.weight.copy()
        self._counterfactual_weight[edge_source] *= np.float32(self.gamma)
        self.production_eligible = False
        self.counterfactual_calibration = True

    def run(self, duration_ms: float, drive=None, **kwargs):
        # HybridBrain reads self.weight during its C call.  The original array
        # is swapped back even if the kernel raises, so no persistent weight
        # array is ever rewritten by calibration.
        original = self.weight
        self.weight = self._counterfactual_weight
        try:
            return super().run(duration_ms, drive, **kwargs)
        finally:
            self.weight = original

    def reset_with_initial_voltage(self, voltage=None, initial_active=None):
        """Reset to clean state, then apply an explicitly audited perturbation."""
        super().reset()
        if voltage is not None:
            x = np.asarray(voltage, dtype=np.float32)
            if x.shape != self.v.shape or not np.isfinite(x).all():
                raise ValueError("initial voltage shape/nonfinite")
            self.v[:] = x
        if initial_active is not None:
            ids = np.asarray(initial_active, dtype=np.int64)
            if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= self.n):
                raise ValueError("initial active ids out of bounds")
            ids = np.unique(ids)
            self.active_flag[ids] = 1
            self.active[: len(ids)] = ids
            self.nactive[0] = len(ids)

