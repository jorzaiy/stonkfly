"""Independent dense float64 reference for Q2 graded-only microcircuits.

This module intentionally does not import the Q2 wrapper or call C++ code.
It implements only the frozen graded membrane/synapse/delay equations needed
for A1-A5 protocol-repair evidence.
"""
from __future__ import annotations

import math
import numpy as np


def signed_deviation(v, rest):
    return np.clip((np.asarray(v, dtype=np.float64) - np.asarray(rest, dtype=np.float64)) / 7.0,
                   -1.0, 1.0)


def simulate_graded(ptr, post, weight, *, dt, steps, drive=None, initial_v=None,
                    rest=None):
    ptr = np.asarray(ptr, dtype=np.int64)
    post = np.asarray(post, dtype=np.int32)
    weight = np.asarray(weight, dtype=np.float64)
    n = len(ptr) - 1
    if ptr.shape != (n + 1,) or ptr[0] != 0 or ptr[-1] != len(post):
        raise ValueError("invalid reference CSR")
    rest = np.full(n, -52.0, np.float64) if rest is None else np.asarray(rest, np.float64).copy()
    drive = np.zeros(n, np.float64) if drive is None else np.asarray(drive, np.float64).copy()
    v = rest.copy() if initial_v is None else np.asarray(initial_v, np.float64).copy()
    g = np.zeros(n, np.float64)
    graded_drive = np.zeros(n, np.float64)
    delay = round(1.8 / dt); slots = delay + 1
    queue = np.zeros((slots, n), np.float64)
    a = math.exp(-dt / 20.0); b = math.exp(-dt / 5.0)
    edge_integral = np.zeros(len(post), np.float64)
    first_arrival = np.full(len(post), -1, np.int64)
    trace_v = np.empty((steps, n), np.float64)
    trace_g = np.empty((steps, n), np.float64)
    trace_u = np.empty((steps, n), np.float64)
    trace_r = np.empty((steps, n), np.float64)
    trace_edge = np.empty((steps, len(post)), np.float64)
    for clock in range(steps):
        gss = graded_drive * 5.0
        oldg = g.copy(); oldv = v.copy()
        g = gss + (oldg - gss) * b
        v = (rest + drive + gss + (oldv - rest - drive - gss) * a
             + (oldg - gss) * (5.0 / 15.0) * (a - b))
        release = signed_deviation(v, rest)
        future = (clock + delay) % slots
        for source in range(n):
            if release[source] == 0.0:
                continue
            for edge in range(int(ptr[source]), int(ptr[source + 1])):
                q = weight[edge] * release[source] / 2.2
                queue[future, post[edge]] += q
                edge_integral[edge] += q * dt
                if first_arrival[edge] < 0:
                    first_arrival[edge] = clock + delay
        slot = clock % slots
        graded_drive = queue[slot].copy(); queue[slot].fill(0.0)
        trace_v[clock] = v; trace_g[clock] = g; trace_u[clock] = graded_drive
        trace_r[clock] = release; trace_edge[clock] = edge_integral
    return {
        "v": trace_v, "g_graded": trace_g, "graded_drive": trace_u,
        "release": trace_r, "edge_signed_delivery": trace_edge,
        "edge_first_arrival_clock": first_arrival,
    }


def first_nonzero_clock(trace, *, threshold=1e-12):
    trace = np.asarray(trace)
    hit = np.flatnonzero(np.abs(trace) > threshold)
    return None if not len(hit) else int(hit[0] + 1)
