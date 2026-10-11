"""S2l bounded KC delivery audit; records only accepted kernel deliveries."""
import ctypes as C
import numpy as np

class TraceObserverS2L(C.Structure):
    _fields_ = [
        ('magic', C.c_uint64), ('abi_version', C.c_int32), ('size', C.c_int32),
        ('source_n', C.c_int32), ('kc_n', C.c_int32), ('overflow', C.c_int32),
        ('source_class', C.c_void_p), ('kc_slot', C.c_void_p), ('kc_first', C.c_void_p),
        ('kc_unique', C.c_void_p), ('kc1_step', C.c_void_p), ('kc10_step', C.c_void_p), ('phase_events', C.c_void_p), ('phase_signed', C.c_void_p),
        ('kc_order_id', C.c_void_p), ('kc_order_step', C.c_void_p), ('kc_order_used', C.c_void_p),
        ('phase_positive', C.c_void_p), ('phase_negative', C.c_void_p),
        ('phase_unique_source', C.c_void_p), ('phase_unique_target', C.c_void_p),
        ('phase_first', C.c_void_p), ('phase_last', C.c_void_p), ('all_kc_delivery_count', C.c_void_p),
        ('source_seen', C.c_void_p), ('target_seen', C.c_void_p),
        ('kc_class_events', C.c_void_p), ('kc_class_signed', C.c_void_p),
        ('kc_class_positive', C.c_void_p), ('kc_class_negative', C.c_void_p),
    ]

class TraceBuffersS2L:
    MAGIC = 0x53324C5452414346
    def __init__(self, source_class, kc_slot):
        self.source_class = np.asarray(source_class, dtype=np.int8)
        self.kc_slot = np.asarray(kc_slot, dtype=np.int32)
        if self.source_class.ndim != 1 or self.kc_slot.shape != self.source_class.shape:
            raise ValueError('S2l slots must cover complete graph')
        if np.any((self.source_class < -1) | (self.source_class >= 6)):
            raise ValueError('invalid source class')
        self.source_n = len(self.source_class)
        self.kc_n = int(self.kc_slot.max() + 1) if np.any(self.kc_slot >= 0) else 0
        self.kc_first = np.full(self.kc_n, -1, np.int32)
        self.kc_unique = np.zeros(1, np.int32)
        self.kc1_step = np.full(1, -1, np.int32)
        self.kc10_step = np.full(1, -1, np.int32)
        self.kc_order_id = np.full(self.kc_n, -1, np.int32)
        self.kc_order_step = np.full(self.kc_n, -1, np.int32)
        self.kc_order_used = np.zeros(1, np.int32)
        self.phase_events = np.zeros((6, 3), np.int64)
        self.phase_signed = np.zeros((6, 3), np.float64)
        self.phase_positive = np.zeros((6, 3), np.float64)
        self.phase_negative = np.zeros((6, 3), np.float64)
        self.phase_unique_source = np.zeros((6, 3), np.int32)
        self.phase_unique_target = np.zeros((6, 3), np.int32)
        self.source_seen = np.zeros((self.source_n, 3), np.uint8)
        self.target_seen = np.zeros((self.kc_n, 6, 3), np.uint8)
        self.phase_first = np.full((6, 3), -1, np.int32)
        self.phase_last = np.full((6, 3), -1, np.int32)
        self.all_kc_delivery_count = np.zeros(1, np.int64)
        self.kc_class_events = np.zeros((self.kc_n, 6), np.int64)
        self.kc_class_signed = np.zeros((self.kc_n, 6), np.float64)
        self.kc_class_positive = np.zeros((self.kc_n, 6), np.float64)
        self.kc_class_negative = np.zeros((self.kc_n, 6), np.float64)
        vals = [self.source_class, self.kc_slot, self.kc_first, self.kc_unique,
                self.kc1_step, self.kc10_step, self.kc_order_id, self.kc_order_step, self.kc_order_used,
                self.phase_events, self.phase_signed, self.phase_positive, self.phase_negative,
                self.phase_unique_source, self.phase_unique_target, self.phase_first, self.phase_last,
                self.all_kc_delivery_count, self.source_seen,
                self.target_seen, self.kc_class_events, self.kc_class_signed,
                self.kc_class_positive, self.kc_class_negative]
        self.native = TraceObserverS2L(self.MAGIC, 1, C.sizeof(TraceObserverS2L),
            self.source_n, self.kc_n, 0, *[x.ctypes.data for x in vals])
    def reset(self):
        self.native.overflow = 0
        for x in (self.kc_unique, self.phase_events, self.phase_signed, self.phase_positive,
                  self.phase_negative, self.phase_unique_source, self.phase_unique_target,
                  self.all_kc_delivery_count, self.source_seen, self.target_seen,
                  self.kc_class_events, self.kc_class_signed, self.kc_class_positive,
                  self.kc_class_negative): x.fill(0)
        for x in (self.kc_first, self.kc1_step, self.kc10_step, self.kc_order_id, self.kc_order_step,
                  self.phase_first, self.phase_last): x.fill(-1)
        self.kc_order_used.fill(0)
    def fail_if_overflow(self):
        if self.native.overflow: raise RuntimeError('S2l trace overflow/ABI failure')
