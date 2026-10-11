"""Preallocated read-only native propagation trace buffers."""
import ctypes as C
import numpy as np


class TraceObserver(C.Structure):
    _fields_ = [("capacity", C.c_int64), ("used", C.c_int64), ("spike_capacity", C.c_int64), ("spike_used", C.c_int64), ("overflow", C.c_int32), ("steps", C.c_int32), ("candidate_mask", C.c_void_p), ("cb_mask", C.c_void_p), ("kc_mask_trace", C.c_void_p), ("delivery_step", C.c_void_p), ("delivery_source", C.c_void_p), ("delivery_target", C.c_void_p), ("delivery_increment", C.c_void_p), ("spike_step", C.c_void_p), ("spike_id", C.c_void_p), ("spike_kind", C.c_void_p), ("step_spike_counts", C.c_void_p), ("start_clock", C.c_int64)]


class TraceBuffers:
    def __init__(self, n, steps, candidate, cb, kc, capacity=2_000_000, spike_capacity=500_000):
        self.candidate_mask = np.asarray(candidate, dtype=np.uint8)
        self.cb_mask = np.asarray(cb, dtype=np.uint8)
        self.kc_mask = np.asarray(kc, dtype=np.uint8)
        if any(x.shape != (n,) for x in (self.candidate_mask, self.cb_mask, self.kc_mask)):
            raise ValueError("trace masks must cover the complete graph")
        self.delivery_step = np.empty(capacity, dtype=np.int64); self.delivery_source = np.empty(capacity, dtype=np.int32); self.delivery_target = np.empty(capacity, dtype=np.int32); self.delivery_increment = np.empty(capacity, dtype=np.float32)
        self.spike_step = np.empty(spike_capacity, dtype=np.int64); self.spike_id = np.empty(spike_capacity, dtype=np.int32); self.spike_kind = np.empty(spike_capacity, dtype=np.uint8); self.step_spike_counts = np.zeros((steps, 3), dtype=np.int32)
        self.native = TraceObserver(capacity, 0, spike_capacity, 0, 0, steps, self.candidate_mask.ctypes.data, self.cb_mask.ctypes.data, self.kc_mask.ctypes.data, self.delivery_step.ctypes.data, self.delivery_source.ctypes.data, self.delivery_target.ctypes.data, self.delivery_increment.ctypes.data, self.spike_step.ctypes.data, self.spike_id.ctypes.data, self.spike_kind.ctypes.data, self.step_spike_counts.ctypes.data, 0)

    def reset(self, start_clock=0):
        self.native.used = self.native.spike_used = self.native.overflow = 0; self.native.start_clock = int(start_clock); self.step_spike_counts.fill(0)

    def fail_if_overflow(self):
        if self.native.overflow:
            raise RuntimeError("native trace buffer overflow; no partial trace is valid")

    def delivery_rows(self):
        n = int(self.native.used); return [{"step": int(self.delivery_step[i]), "source_index": int(self.delivery_source[i]), "target_index": int(self.delivery_target[i]), "signed_g_increment": float(self.delivery_increment[i])} for i in range(n)]

    def spike_rows(self):
        n = int(self.native.spike_used); return [{"step": int(self.spike_step[i]), "neuron_index": int(self.spike_id[i]), "kind": int(self.spike_kind[i])} for i in range(n)]
