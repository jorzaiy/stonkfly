"""Isolated Stonkfly v2 M1 representation laboratory.

This package is intentionally independent of market, reward, plasticity,
controller, and execution modules.  It does not authorize formal M1 runs.
"""

from .dynamics import M1Parameters, SimulationResult, simulate
from .topology import BipartiteTopology

__all__ = ["BipartiteTopology", "M1Parameters", "SimulationResult", "simulate"]
