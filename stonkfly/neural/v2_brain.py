"""Isolated V2-Q1 hybrid brain and audit helpers.

The module is deliberately self-contained: importing it does not import the
v1 brain, controller, plasticity, decoder, market, or reward paths.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("v2_kernel_qualification.cpp")
BUILD_DIR = ROOT / "runs/whole_brain_v2/Q1/.build"
LIBRARY = BUILD_DIR / ("libv2_hybrid.dylib" if sys.platform == "darwin" else "libv2_hybrid.so")
TAU_M_MS = 20.0
TAU_S_MS = 5.0
V_TH_MV = -45.0
V_REST_MV = -52.0
DELAY_MS = 1.8
REFRACTORY_MS = 2.2
TAU_EMIT_MS = 2.2
ALLOWED_DT = (0.1, 0.05, 0.025)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build() -> dict[str, object]:
    """Build only the isolated V2 translation unit and record provenance."""
    source_sha = sha256_file(SOURCE)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    meta = LIBRARY.with_suffix(LIBRARY.suffix + ".json")
    if LIBRARY.exists() and meta.exists():
        old = json.loads(meta.read_text())
        if old.get("source_sha256") == source_sha and old.get("binary_sha256") == sha256_file(LIBRARY):
            return old
    partial = LIBRARY.with_suffix(LIBRARY.suffix + ".partial")
    cmd = ["c++", "-O3", "-std=c++17", "-shared", "-fPIC", str(SOURCE), "-o", str(partial)]
    subprocess.run(cmd, check=True, cwd=ROOT)
    partial.replace(LIBRARY)
    record = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": source_sha,
        "binary": str(LIBRARY.relative_to(ROOT)),
        "binary_sha256": sha256_file(LIBRARY),
        "compiler": subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
        "flags": cmd[1:5],
    }
    meta.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def release_from_voltage(v: np.ndarray | float, rest: float = V_REST_MV) -> np.ndarray | float:
    """Frozen Q1 release clamp, with no threshold/reset side effects."""
    x = np.clip((np.asarray(v) - rest) / (V_TH_MV - rest), 0.0, 1.0)
    return float(x) if np.ndim(v) == 0 else x


def exact_segment(v, g_event, g_graded, drive, graded_rate, dt, rest=V_REST_MV):
    """Return the frozen analytic non-refractory segment update."""
    a = math.exp(-dt / TAU_M_MS)
    b = math.exp(-dt / TAU_S_MS)
    gss = graded_rate * TAU_S_MS
    vp = (rest + drive + gss + (v - rest - drive - gss) * a
          + (g_graded - gss) * TAU_S_MS / (TAU_M_MS - TAU_S_MS) * (a - b)
          + g_event * (a - b) / 3.0)
    return vp, gss + (g_graded - gss) * b, g_event * b


def classify_nodes(cell_type, neurotransmitter):
    """Apply the frozen P0 mutually exclusive declaration priority."""
    ct = np.asarray(["" if x is None else str(x) for x in cell_type], dtype=object)
    nt = np.asarray(["" if x is None else str(x) for x in neurotransmitter], dtype=object)
    names = ("modulatory_declared", "graded_candidate_photoreceptor", "graded_candidate_lamina",
             "graded_candidate_apl", "adaptive_spiking_kc", "generic_spiking_declared")
    layer = np.full(len(ct), 5, dtype=np.int8)
    mod = np.isin(nt, ("dopamine", "octopamine", "serotonin"))
    layer[mod] = 0
    m = ~mod
    layer[m & ((ct == "R1-R6") | np.char.startswith(ct.astype(str), "R8"))] = 1
    layer[m & np.isin(ct, ("L1", "L2", "L3", "L5"))] = 2
    layer[m & (ct == "APL")] = 3
    layer[m & np.char.startswith(ct.astype(str), "KC")] = 4
    return layer, names


class HybridBrain:
    """Retained-graph V2 kernel state with learning permanently disabled."""

    def __init__(self, path=ROOT / "data/graph.npz", *, dt=0.1, graded_mask=None,
                 rest=None, compile=True):
        if dt not in ALLOWED_DT:
            raise ValueError("dt must be one of 0.1, 0.05, 0.025 ms")
        a = np.load(path, allow_pickle=False)
        self.ptr = np.ascontiguousarray(a["ptr"], dtype=np.int64)
        self.post = np.ascontiguousarray(a["post"], dtype=np.int32)
        self.weight = np.ascontiguousarray(a["weight"], dtype=np.float32)
        self.ids = np.ascontiguousarray(a["ids"], dtype=np.int64)
        self.n = len(self.ids)
        if (
            self.ptr.shape != (self.n + 1,)
            or self.ptr[0] != 0
            or np.any(np.diff(self.ptr) < 0)
            or self.ptr[-1] != len(self.post)
            or len(self.weight) != len(self.post)
            or np.any(self.post < 0)
            or np.any(self.post >= self.n)
            or not np.isfinite(self.weight).all()
        ):
            raise ValueError("invalid CSR graph")
        self.dt = float(dt)
        self.delay_steps = round(DELAY_MS / self.dt)
        self.refractory_steps = round(REFRACTORY_MS / self.dt)
        slots = self.delay_steps + 1
        if graded_mask is None:
            graded_mask = np.zeros(self.n, dtype=np.uint8)
        self.graded_mask = np.ascontiguousarray(graded_mask, dtype=np.uint8)
        if self.graded_mask.shape != (self.n,):
            raise ValueError("graded_mask shape mismatch")
        self.rest = np.full(self.n, V_REST_MV, dtype=np.float32) if rest is None else np.ascontiguousarray(rest, dtype=np.float32)
        if self.rest.shape != (self.n,):
            raise ValueError("rest shape mismatch")
        self.kc_mask = np.zeros(self.n, dtype=np.uint8)
        # v1 state is retained in full even when Q1 learning is off.  The
        # runner fills KC/modulatory masks from the frozen P0 classification.
        self.last = np.full(self.n, -1, dtype=np.int64)
        self.eligibility = np.zeros(self.n, dtype=np.float64)
        self.eligibility_last = np.zeros(self.n, dtype=np.int64)
        self.modulation = np.zeros(self.n, dtype=np.float32)
        self.modulation_last = np.zeros(self.n, dtype=np.int64)
        self.modulation_mask = np.zeros(self.n, dtype=np.uint8)
        self.luminance = np.zeros(len(a["retina"]) if "retina" in a.files else 0, dtype=np.float32)
        self.v = self.rest.copy()
        self.g_event = np.zeros(self.n, dtype=np.float32)
        self.g_graded = np.zeros(self.n, dtype=np.float32)
        self.refractory = np.zeros(self.n, dtype=np.int16)
        self.drive = np.zeros(self.n, dtype=np.float32)
        self.previous_drive = np.zeros(self.n, dtype=np.float32)
        self.spike_queue = np.zeros((slots, self.n), dtype=np.int32)
        self.spike_queue_count = np.zeros(slots, dtype=np.int32)
        self.graded_queue = np.zeros((slots, self.n), dtype=np.float32)
        self.graded_drive = np.zeros(self.n, dtype=np.float32)
        self.clock = np.asarray([0], dtype=np.int64)
        self.counts = np.zeros(self.n, dtype=np.int32)
        # Initialize unused active slots too: they are part of the audited
        # state/hash boundary and must not contain allocator-dependent bytes.
        self.active = np.zeros(self.n, dtype=np.int32)
        self.active_flag = np.zeros(self.n, dtype=np.uint8)
        self.nactive = np.asarray([0], dtype=np.int32)
        initial_parts = [a[k] for k in ("retina", "lamina", "sugar") if k in a.files]
        initial = np.unique(np.concatenate(initial_parts)) if initial_parts else np.empty(0, dtype=np.int32)
        self.active[:len(initial)] = initial
        self.active_flag[initial] = 1
        self.nactive[0] = len(initial)
        self.adaptation = np.zeros(self.n, dtype=np.float32)
        self.adaptation_jump = 8.0
        self.adaptation_tau = 200.0
        self.release_integral = np.zeros(1, dtype=np.float64)
        self.signed_delivery = np.zeros(1, dtype=np.float64)
        self.absolute_delivery = np.zeros(1, dtype=np.float64)
        self.overflow = np.zeros(1, dtype=np.int32)
        self.edge_signed_delivery = None
        self.edge_event_count = None
        self.edge_first_arrival_clock = None
        self.total_spikes = 0
        self.build = build() if compile else None
        self._load_library()

    def _load_library(self):
        if self.build is None:
            self.build = build()
        self.lib = C.CDLL(str(LIBRARY))
        self.advance = self.lib.v2_hybrid_advance
        self.advance.argtypes = [C.c_int, C.POINTER(C.c_int64), C.POINTER(C.c_int32), C.POINTER(C.c_float),
            C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_int16),
            C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_int32), C.POINTER(C.c_int32),
            C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_int64), C.c_int, C.c_float,
            C.POINTER(C.c_uint8), C.POINTER(C.c_uint8), C.POINTER(C.c_float), C.POINTER(C.c_float),
            C.c_float, C.c_float, C.POINTER(C.c_int32), C.POINTER(C.c_int32), C.POINTER(C.c_uint8),
            C.POINTER(C.c_int32), C.POINTER(C.c_int64), C.POINTER(C.c_double), C.POINTER(C.c_int64),
            C.POINTER(C.c_float), C.POINTER(C.c_int64), C.POINTER(C.c_uint8), C.POINTER(C.c_double),
            C.POINTER(C.c_double), C.POINTER(C.c_double), C.POINTER(C.c_int32),
            C.POINTER(C.c_double), C.POINTER(C.c_int64), C.POINTER(C.c_int64)]
        self.advance.restype = None
        self.compat_advance = self.lib.v2_compat_advance
        self.compat_advance.argtypes = [C.c_int, C.POINTER(C.c_int64), C.POINTER(C.c_int32), C.POINTER(C.c_float),
            C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_int16), C.POINTER(C.c_float),
            C.POINTER(C.c_float), C.POINTER(C.c_int32), C.POINTER(C.c_int32), C.POINTER(C.c_int64),
            C.c_int, C.c_float, C.POINTER(C.c_int32), C.POINTER(C.c_int32), C.POINTER(C.c_uint8),
            C.POINTER(C.c_int32), C.POINTER(C.c_int64), C.POINTER(C.c_uint8), C.POINTER(C.c_int8),
            C.POINTER(C.c_double), C.POINTER(C.c_int64), C.c_int, C.POINTER(C.c_int64), C.POINTER(C.c_int32),
            C.POINTER(C.c_float), C.POINTER(C.c_float), C.c_float, C.c_float, C.c_float, C.c_int,
            C.POINTER(C.c_float), C.POINTER(C.c_int64), C.POINTER(C.c_uint8), C.POINTER(C.c_float),
            C.POINTER(C.c_float), C.c_float, C.c_float] + [C.c_void_p] * 8
        self.compat_advance.restype = None

    def reset(self):
        self.v[:] = self.rest; self.g_event.fill(0); self.g_graded.fill(0)
        self.refractory.fill(0); self.drive.fill(0); self.previous_drive.fill(0)
        self.spike_queue.fill(0); self.spike_queue_count.fill(0); self.graded_queue.fill(0)
        self.graded_drive.fill(0); self.clock[0] = 0; self.counts.fill(0)
        self.active_flag.fill(0); self.nactive[0] = 0; self.adaptation.fill(0)
        self.last.fill(-1); self.eligibility.fill(0); self.eligibility_last.fill(0)
        self.modulation.fill(0); self.modulation_last.fill(0); self.luminance.fill(0)
        self.release_integral.fill(0); self.signed_delivery.fill(0); self.absolute_delivery.fill(0); self.overflow.fill(0); self.total_spikes = 0

    def run(self, duration_ms: float, drive=None, *, observer=False, ledger=True,
            edge_trace=False):
        steps = round(float(duration_ms) / self.dt)
        if steps < 1 or not math.isfinite(duration_ms):
            raise ValueError("duration must contain at least one dt")
        if drive is not None:
            x = np.asarray(drive, dtype=np.float32)
            if x.shape != (self.n,) or not np.isfinite(x).all():
                raise ValueError("drive shape/nonfinite")
            self.drive[:] = x
        # v1 exposes spike counts per public segment; total_spikes is the
        # cumulative convenience counter maintained by the wrapper.
        self.counts.fill(0)
        before = self.counts.copy()
        # The empty-mask path is the actual v2 compatibility entry point. It
        # uses the frozen v1 arithmetic embedded in the isolated v2 binary;
        # graded counters remain zero and the complete v1 state is retained.
        compatibility = not np.any(self.graded_mask)
        if compatibility:
            self._run_compat(steps)
            self.total_spikes += int(np.sum(self.counts - before))
            return self.counts.copy()
        release_ptr = self.release_integral if ledger else np.zeros(1, dtype=np.float64)
        signed_ptr = self.signed_delivery if ledger else np.zeros(1, dtype=np.float64)
        absolute_ptr = self.absolute_delivery if ledger else np.zeros(1, dtype=np.float64)
        if edge_trace and self.edge_signed_delivery is None:
            m = len(self.weight)
            self.edge_signed_delivery = np.zeros(m, dtype=np.float64)
            self.edge_event_count = np.zeros(m, dtype=np.int64)
            self.edge_first_arrival_clock = np.full(m, -1, dtype=np.int64)
        edge_signed_ptr = self.edge_signed_delivery if edge_trace else None
        edge_count_ptr = self.edge_event_count if edge_trace else None
        edge_clock_ptr = self.edge_first_arrival_clock if edge_trace else None
        self.advance(self.n, self.ptr.ctypes.data_as(C.POINTER(C.c_int64)), self.post.ctypes.data_as(C.POINTER(C.c_int32)), self.weight.ctypes.data_as(C.POINTER(C.c_float)),
                     self.v.ctypes.data_as(C.POINTER(C.c_float)), self.g_event.ctypes.data_as(C.POINTER(C.c_float)), self.g_graded.ctypes.data_as(C.POINTER(C.c_float)), self.refractory.ctypes.data_as(C.POINTER(C.c_int16)),
                     self.drive.ctypes.data_as(C.POINTER(C.c_float)), self.previous_drive.ctypes.data_as(C.POINTER(C.c_float)), self.spike_queue.ctypes.data_as(C.POINTER(C.c_int32)), self.spike_queue_count.ctypes.data_as(C.POINTER(C.c_int32)),
                     self.graded_queue.ctypes.data_as(C.POINTER(C.c_float)), self.graded_drive.ctypes.data_as(C.POINTER(C.c_float)), self.clock.ctypes.data_as(C.POINTER(C.c_int64)), steps, self.dt,
                     self.graded_mask.ctypes.data_as(C.POINTER(C.c_uint8)), self.kc_mask.ctypes.data_as(C.POINTER(C.c_uint8)), self.rest.ctypes.data_as(C.POINTER(C.c_float)), self.adaptation.ctypes.data_as(C.POINTER(C.c_float)),
                     self.adaptation_jump, self.adaptation_tau, self.counts.ctypes.data_as(C.POINTER(C.c_int32)), self.active.ctypes.data_as(C.POINTER(C.c_int32)), self.active_flag.ctypes.data_as(C.POINTER(C.c_uint8)), self.nactive.ctypes.data_as(C.POINTER(C.c_int32)),
                     self.last.ctypes.data_as(C.POINTER(C.c_int64)), self.eligibility.ctypes.data_as(C.POINTER(C.c_double)), self.eligibility_last.ctypes.data_as(C.POINTER(C.c_int64)), self.modulation.ctypes.data_as(C.POINTER(C.c_float)), self.modulation_last.ctypes.data_as(C.POINTER(C.c_int64)), self.modulation_mask.ctypes.data_as(C.POINTER(C.c_uint8)),
                     release_ptr.ctypes.data_as(C.POINTER(C.c_double)), signed_ptr.ctypes.data_as(C.POINTER(C.c_double)), absolute_ptr.ctypes.data_as(C.POINTER(C.c_double)), self.overflow.ctypes.data_as(C.POINTER(C.c_int32)),
                     edge_signed_ptr.ctypes.data_as(C.POINTER(C.c_double)) if edge_signed_ptr is not None else C.POINTER(C.c_double)(),
                     edge_count_ptr.ctypes.data_as(C.POINTER(C.c_int64)) if edge_count_ptr is not None else C.POINTER(C.c_int64)(),
                     edge_clock_ptr.ctypes.data_as(C.POINTER(C.c_int64)) if edge_clock_ptr is not None else C.POINTER(C.c_int64)())
        self.total_spikes += int(np.sum(self.counts - before))
        if self.overflow[0]:
            raise RuntimeError("v2 ring-buffer overflow")
        return self.counts.copy()

    def _run_compat(self, steps):
        """Call the frozen v1 ABI through v2_compat_advance."""
        null = C.c_void_p(0)
        empty_i64 = np.empty(0, dtype=np.int64)
        empty_i32 = np.empty(0, dtype=np.int32)
        empty_f32 = np.empty(0, dtype=np.float32)
        dan = np.full(self.n, -1, dtype=np.int8)
        self.compat_advance(self.n, self.ptr.ctypes.data_as(C.POINTER(C.c_int64)), self.post.ctypes.data_as(C.POINTER(C.c_int32)), self.weight.ctypes.data_as(C.POINTER(C.c_float)), self.v.ctypes.data_as(C.POINTER(C.c_float)), self.g_event.ctypes.data_as(C.POINTER(C.c_float)), self.refractory.ctypes.data_as(C.POINTER(C.c_int16)), self.drive.ctypes.data_as(C.POINTER(C.c_float)), self.previous_drive.ctypes.data_as(C.POINTER(C.c_float)), self.spike_queue.ctypes.data_as(C.POINTER(C.c_int32)), self.spike_queue_count.ctypes.data_as(C.POINTER(C.c_int32)), self.clock.ctypes.data_as(C.POINTER(C.c_int64)), steps, self.dt, self.counts.ctypes.data_as(C.POINTER(C.c_int32)), self.active.ctypes.data_as(C.POINTER(C.c_int32)), self.active_flag.ctypes.data_as(C.POINTER(C.c_uint8)), self.nactive.ctypes.data_as(C.POINTER(C.c_int32)), self.last.ctypes.data_as(C.POINTER(C.c_int64)), self.kc_mask.ctypes.data_as(C.POINTER(C.c_uint8)), dan.ctypes.data_as(C.POINTER(C.c_int8)), self.eligibility.ctypes.data_as(C.POINTER(C.c_double)), self.eligibility_last.ctypes.data_as(C.POINTER(C.c_int64)), 0, empty_i64.ctypes.data_as(C.POINTER(C.c_int64)), empty_i32.ctypes.data_as(C.POINTER(C.c_int32)), empty_f32.ctypes.data_as(C.POINTER(C.c_float)), empty_f32.ctypes.data_as(C.POINTER(C.c_float)), 0.0, 100.0, 0.0, 0, self.modulation.ctypes.data_as(C.POINTER(C.c_float)), self.modulation_last.ctypes.data_as(C.POINTER(C.c_int64)), self.modulation_mask.ctypes.data_as(C.POINTER(C.c_uint8)), self.rest.ctypes.data_as(C.POINTER(C.c_float)), self.adaptation.ctypes.data_as(C.POINTER(C.c_float)), self.adaptation_jump, self.adaptation_tau, null, null, null, null, null, null, null, null)

    def audit_state(self):
        return {"v": self.v.copy(), "g_event": self.g_event.copy(), "g_graded": self.g_graded.copy(),
                "refractory": self.refractory.copy(), "drive": self.drive.copy(), "previous_drive": self.previous_drive.copy(),
                "queue": self.spike_queue.copy(), "queue_count": self.spike_queue_count.copy(), "counts": self.counts.copy(),
                "active": self.active.copy(), "active_flag": self.active_flag.copy(), "nactive": self.nactive.copy(),
                "last": self.last.copy(), "adaptation": self.adaptation.copy(), "eligibility": self.eligibility.copy(),
                "eligibility_last": self.eligibility_last.copy(), "modulation": self.modulation.copy(), "modulation_last": self.modulation_last.copy(),
                "graded_queue": self.graded_queue.copy(), "graded_drive": self.graded_drive.copy(), "overflow": int(self.overflow[0]),
                "clock": int(self.clock[0]), "total_spikes": int(self.total_spikes), "release_integral": float(self.release_integral[0]),
                "signed_delivery": float(self.signed_delivery[0]), "absolute_delivery": float(self.absolute_delivery[0])}
