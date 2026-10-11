import ctypes as C
import numpy as np

class TraceObserverS2G(C.Structure):
    _fields_=[('magic',C.c_uint64),('abi_version',C.c_int32),('size',C.c_int32),('source_n',C.c_int32),('target_n',C.c_int32),('overflow',C.c_int32),('source_slot',C.c_void_p),('target_slot',C.c_void_p),('target_first',C.c_void_p),('pair_events',C.c_void_p),('pair_signed',C.c_void_p),('pair_positive',C.c_void_p),('pair_negative',C.c_void_p),('pair_first',C.c_void_p),('pair_last',C.c_void_p)]

class TraceS2G:
    MAGIC=0x5332475452414346
    def __init__(self, source_slot, target_slot, steps=5000):
        self.source_slot=np.asarray(source_slot,dtype=np.int32)
        self.target_slot=np.asarray(target_slot,dtype=np.int32)
        self.source_n=int(self.source_slot.max(initial=-1)+1); self.target_n=int(self.target_slot.max(initial=-1)+1)
        self.steps=int(steps)
        self.target_first=np.full(self.target_n,-1,np.int32)
        shape=(self.source_n,self.target_n)
        self.pair_events=np.zeros(shape,np.int32); self.pair_signed=np.zeros(shape); self.pair_positive=np.zeros(shape); self.pair_negative=np.zeros(shape)
        self.pair_first=np.full(shape,-1,np.int32); self.pair_last=np.full(shape,-1,np.int32)
        fields=[self.source_slot,self.target_slot,self.target_first,self.pair_events,self.pair_signed,self.pair_positive,self.pair_negative,self.pair_first,self.pair_last]
        self.native=TraceObserverS2G(self.MAGIC,1,C.sizeof(TraceObserverS2G),self.source_n,self.target_n,0,*[x.ctypes.data for x in fields])
    def reset(self):
        self.native.overflow=0; self.target_first.fill(-1); self.pair_events.fill(0); self.pair_signed.fill(0); self.pair_positive.fill(0); self.pair_negative.fill(0); self.pair_first.fill(-1); self.pair_last.fill(-1)
    def fail(self):
        if self.native.overflow: raise RuntimeError('S2G trace overflow or ABI failure')
