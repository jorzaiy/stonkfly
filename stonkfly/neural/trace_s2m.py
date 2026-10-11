"""S2m KC1-relative accepted-delivery observer."""
import ctypes as C
import numpy as np

S2M_CLASSES=("KC","APL","DPM","PN/visual_projection","other_cb","other")
WINDOW_BOUNDS=((0,64),(64,128),(128,256),(256,512),(512,1024))
FIXED_KC_BODY_IDS=(18540,21778,44069,520206)

class TraceObserverS2M(C.Structure):
 _fields_=[('magic',C.c_uint64),('abi_version',C.c_int32),('size',C.c_int32),('source_n',C.c_int32),('kc_n',C.c_int32),('other_cb_n',C.c_int32),('fixed_kc_n',C.c_int32),('overflow',C.c_int32),('source_class',C.c_void_p),('kc_slot',C.c_void_p),('other_cb_slot',C.c_void_p),('kc_fixed_slot',C.c_void_p),('kc_first',C.c_void_p),('kc1_step',C.c_void_p),('events',C.c_void_p),('signed',C.c_void_p),('positive',C.c_void_p),('negative',C.c_void_p),('unique_source',C.c_void_p),('unique_target',C.c_void_p),('first',C.c_void_p),('last',C.c_void_p),('source_seen',C.c_void_p),('target_seen',C.c_void_p),('other_cb_events',C.c_void_p),('other_cb_signed',C.c_void_p),('other_cb_positive',C.c_void_p),('other_cb_negative',C.c_void_p),('other_cb_first',C.c_void_p),('other_cb_last',C.c_void_p),('other_cb_source_seen',C.c_void_p),('fixed_events',C.c_void_p),('fixed_signed',C.c_void_p),('fixed_positive',C.c_void_p),('fixed_negative',C.c_void_p),('fixed_first',C.c_void_p),('fixed_last',C.c_void_p),('audit',C.c_void_p)]

class TraceBuffersS2M:
 MAGIC=0x53324D5452414346; ABI_VERSION=2
 def __init__(self,source_class,kc_slot,*,source_body_ids=None,kc_body_ids=None,fixed_kc_body_ids=FIXED_KC_BODY_IDS):
  self.source_class=np.ascontiguousarray(source_class,np.int8);self.kc_slot=np.ascontiguousarray(kc_slot,np.int32)
  if self.source_class.ndim!=1 or self.kc_slot.shape!=self.source_class.shape: raise ValueError('S2m slots must cover complete graph')
  if np.any((self.source_class < -1)|(self.source_class >= 6)) or np.any(self.kc_slot < -1): raise ValueError('invalid S2m slots')
  self.source_n=len(self.source_class);self.kc_n=int(self.kc_slot.max()+1) if np.any(self.kc_slot>=0) else 0
  self.other_cb_source_indices=np.flatnonzero(self.source_class==4).astype(np.int32);self.other_cb_n=len(self.other_cb_source_indices);self.other_cb_slot=np.full(self.source_n,-1,np.int32);self.other_cb_slot[self.other_cb_source_indices]=np.arange(self.other_cb_n,dtype=np.int32)
  self.source_body_ids=np.arange(self.source_n,dtype=np.int64) if source_body_ids is None else np.ascontiguousarray(source_body_ids,dtype=np.int64);self.kc_body_ids=np.arange(self.kc_n,dtype=np.int64) if kc_body_ids is None else np.ascontiguousarray(kc_body_ids,dtype=np.int64)
  if self.source_body_ids.shape!=(self.source_n,) or self.kc_body_ids.shape!=(self.kc_n,): raise ValueError('S2m body ID shape mismatch')
  self.fixed_kc_body_ids=np.asarray(tuple(fixed_kc_body_ids),dtype=np.int64);self.fixed_kc_n=len(self.fixed_kc_body_ids);self.kc_fixed_slot=np.full(self.kc_n,-1,np.int32)
  for slot,body_id in enumerate(self.fixed_kc_body_ids):
   hits=np.flatnonzero(self.kc_body_ids==body_id)
   if len(hits)==1:self.kc_fixed_slot[hits[0]]=slot
  self.kc_first=np.full(self.kc_n,-1,np.int32);self.kc1_step=np.full(1,-1,np.int32);self.events=np.zeros((6,5),np.int64);self.signed=np.zeros((6,5));self.positive=np.zeros((6,5));self.negative=np.zeros((6,5));self.unique_source=np.zeros((6,5),np.int32);self.unique_target=np.zeros((6,5),np.int32);self.first=np.full((6,5),-1,np.int32);self.last=np.full((6,5),-1,np.int32);self.source_seen=np.zeros((self.source_n,5),np.uint8);self.target_seen=np.zeros((self.kc_n,6,5),np.uint8)
  shape=(self.other_cb_n,5);self.other_cb_events=np.zeros(shape,np.int64);self.other_cb_signed=np.zeros(shape);self.other_cb_positive=np.zeros(shape);self.other_cb_negative=np.zeros(shape);self.other_cb_first=np.full(shape,-1,np.int32);self.other_cb_last=np.full(shape,-1,np.int32);self.other_cb_source_seen=np.zeros(shape,np.uint8)
  fshape=(self.fixed_kc_n,6,5);self.fixed_events=np.zeros(fshape,np.int64);self.fixed_signed=np.zeros(fshape);self.fixed_positive=np.zeros(fshape);self.fixed_negative=np.zeros(fshape);self.fixed_first=np.full(fshape,-1,np.int32);self.fixed_last=np.full(fshape,-1,np.int32);self.audit=np.zeros(1,np.int64)
  self._arrays=[self.source_class,self.kc_slot,self.other_cb_slot,self.kc_fixed_slot,self.kc_first,self.kc1_step,self.events,self.signed,self.positive,self.negative,self.unique_source,self.unique_target,self.first,self.last,self.source_seen,self.target_seen,self.other_cb_events,self.other_cb_signed,self.other_cb_positive,self.other_cb_negative,self.other_cb_first,self.other_cb_last,self.other_cb_source_seen,self.fixed_events,self.fixed_signed,self.fixed_positive,self.fixed_negative,self.fixed_first,self.fixed_last,self.audit]
  self.native=TraceObserverS2M(self.MAGIC,self.ABI_VERSION,C.sizeof(TraceObserverS2M),self.source_n,self.kc_n,self.other_cb_n,self.fixed_kc_n,0,*[x.ctypes.data for x in self._arrays])
 @property
 def compact_other_cb_events(self): return self.other_cb_events
 @property
 def compact_other_cb_signed(self): return self.other_cb_signed
 @property
 def fixed4_events(self): return self.fixed_events
 @property
 def fixed4_signed(self): return self.fixed_signed
 @property
 def fixed4_positive(self): return self.fixed_positive
 @property
 def fixed4_negative(self): return self.fixed_negative
 @property
 def fixed4_first(self): return self.fixed_first
 @property
 def fixed4_last(self): return self.fixed_last
 @property
 def fixed4_target_events(self): return self.fixed_events
 @property
 def fixed4_target_signed(self): return self.fixed_signed
 def other_cb_source_mapping(self): return [{'source_index':int(i),'body_id':int(self.source_body_ids[i]),'slot':int(slot)} for slot,i in enumerate(self.other_cb_source_indices)]
 def other_cb_sparse_coo(self):
  rows=[]
  for slot,source_index in enumerate(self.other_cb_source_indices):
   for window in range(5):
    if self.other_cb_events[slot,window]==0: continue
    rows.append({'source_slot':int(slot),'source_index':int(source_index),'body_id':int(self.source_body_ids[source_index]),'window':int(window),'events':int(self.other_cb_events[slot,window]),'signed':float(self.other_cb_signed[slot,window]),'positive':float(self.other_cb_positive[slot,window]),'negative':float(self.other_cb_negative[slot,window]),'first':int(self.other_cb_first[slot,window]),'last':int(self.other_cb_last[slot,window])})
  return rows
 def reset(self):
  self.native.overflow=0
  for x in (self.events,self.signed,self.positive,self.negative,self.unique_source,self.unique_target,self.source_seen,self.target_seen,self.other_cb_events,self.other_cb_signed,self.other_cb_positive,self.other_cb_negative,self.other_cb_source_seen,self.fixed_events,self.fixed_signed,self.fixed_positive,self.fixed_negative,self.audit):x.fill(0)
  for x in (self.kc_first,self.kc1_step,self.first,self.last,self.other_cb_first,self.other_cb_last,self.fixed_first,self.fixed_last):x.fill(-1)
 def fail(self):
  if self.native.overflow: raise RuntimeError('S2m trace overflow/ABI failure')
 fail_if_overflow=fail
