"""Independent S2n intervention ABI and buffers.

The kernel receives this observer as a nullable pointer.  It only records
accepted KC deliveries and transiently scales conductance; persistent weights
are never exposed as writable intervention state.
"""
import ctypes as C
import numpy as np

WINDOW_BOUNDS=((0,64),(64,128),(128,256),(256,512),(512,1024))

class TraceObserverS2N(C.Structure):
    _fields_=[
        ('magic',C.c_uint64),('abi_version',C.c_int32),('size',C.c_int32),
        ('source_n',C.c_int32),('kc_n',C.c_int32),('overflow',C.c_int32),
        ('source_group',C.c_void_p),('kc_slot',C.c_void_p),('scale',C.c_void_p),
        ('kc1_step',C.c_void_p),('eligible_count',C.c_void_p),('applied_count',C.c_void_p),
        ('eligible_events',C.c_void_p),('applied_events',C.c_void_p),
        ('original_signed',C.c_void_p),('original_positive',C.c_void_p),('original_negative',C.c_void_p),
        ('applied_signed',C.c_void_p),('applied_positive',C.c_void_p),('applied_negative',C.c_void_p),
        ('selected_outside_count',C.c_void_p),
    ]

class TraceBuffersS2N:
    MAGIC=0x53324E5452414346
    ABI_VERSION=2
    def __init__(self,source_group,kc_slot,scale,*,source_body_ids=None):
        # ABI v2 carries the source group explicitly: 0=none, 1=visual, and
        # 2=frozen top-20 other_cb.  A single group is intentional; BOTH is
        # represented by the disjoint union of the two source classes.
        self.source_group=np.ascontiguousarray(source_group,dtype=np.int8)
        self.kc_slot=np.ascontiguousarray(kc_slot,dtype=np.int32)
        self.scale=np.ascontiguousarray(scale,dtype=np.float64)
        if self.source_group.ndim!=1 or self.kc_slot.shape!=self.source_group.shape:
            raise ValueError('S2n source/KC slots must cover complete graph')
        if self.scale.shape!=(5,) or not np.isfinite(self.scale).all() or np.any(self.scale<0):
            raise ValueError('S2n scale must be finite nonnegative length-5')
        if np.any((self.source_group < 0) | (self.source_group > 2)) or np.any(self.kc_slot<-1): raise ValueError('invalid S2n source groups/slots')
        self.source_n=len(self.source_group); self.kc_n=int(self.kc_slot.max()+1) if np.any(self.kc_slot>=0) else 0
        self.source_body_ids=(np.arange(self.source_n,dtype=np.int64) if source_body_ids is None else np.ascontiguousarray(source_body_ids,dtype=np.int64))
        if self.source_body_ids.shape!=(self.source_n,): raise ValueError('source_body_ids shape mismatch')
        self.kc1_step=np.full(1,-1,np.int32); self.eligible_count=np.zeros(1,np.int64); self.applied_count=np.zeros(1,np.int64)
        self.eligible_events=np.zeros((2,5),np.int64); self.applied_events=np.zeros((2,5),np.int64)
        self.original_signed=np.zeros((2,5)); self.original_positive=np.zeros((2,5)); self.original_negative=np.zeros((2,5))
        self.applied_signed=np.zeros((2,5)); self.applied_positive=np.zeros((2,5)); self.applied_negative=np.zeros((2,5))
        self.selected_outside_count=np.zeros(1,np.int64)
        self._arrays=[self.source_group,self.kc_slot,self.scale,self.kc1_step,self.eligible_count,self.applied_count,self.eligible_events,self.applied_events,self.original_signed,self.original_positive,self.original_negative,self.applied_signed,self.applied_positive,self.applied_negative,self.selected_outside_count]
        self.native=TraceObserverS2N(self.MAGIC,self.ABI_VERSION,C.sizeof(TraceObserverS2N),self.source_n,self.kc_n,0,*[x.ctypes.data for x in self._arrays])
    def reset(self):
        self.native.overflow=0
        for x in (self.kc1_step,): x.fill(-1)
        for x in (self.eligible_count,self.applied_count,self.eligible_events,self.applied_events,self.original_signed,self.original_positive,self.original_negative,self.applied_signed,self.applied_positive,self.applied_negative,self.selected_outside_count): x.fill(0)
    def fail(self):
        if self.native.overflow: raise RuntimeError('S2n trace overflow/ABI failure')
    fail_if_overflow=fail
    def source_mapping(self):
        return [{'source_index':int(i),'body_id':int(self.source_body_ids[i])} for i in np.flatnonzero(self.source_group)]
