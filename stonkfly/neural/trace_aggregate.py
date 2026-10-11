"""Bounded native propagation aggregates."""
import ctypes as C
import numpy as np
class TraceObserver(C.Structure):
 _fields_=[('steps',C.c_int32),('candidate_n',C.c_int32),('kc_n',C.c_int32),('overflow',C.c_int32),('candidate_slot',C.c_void_p),('cb_mask',C.c_void_p),('kc_slot',C.c_void_p),('step_spike_counts',C.c_void_p),('candidate_spike_counts',C.c_void_p),('candidate_event_counts',C.c_void_p),('candidate_signed',C.c_void_p),('candidate_positive',C.c_void_p),('candidate_negative',C.c_void_p),('cbkc_event_counts',C.c_void_p),('cbkc_signed',C.c_void_p),('cbkc_positive',C.c_void_p),('cbkc_negative',C.c_void_p),('kc_first_spike_step',C.c_void_p),('start_clock',C.c_int64)]
class TraceBuffers:
 def __init__(self,n,steps,candidate_slot,cb_mask,kc_slot):
  self.steps=int(steps); self.candidate_slot=np.asarray(candidate_slot,np.int32); self.cb_mask=np.asarray(cb_mask,np.uint8); self.kc_slot=np.asarray(kc_slot,np.int32)
  if any(x.shape!=(n,) for x in (self.candidate_slot,self.cb_mask,self.kc_slot)): raise ValueError('trace masks must cover graph')
  self.candidate_n=int(self.candidate_slot.max()+1) if np.any(self.candidate_slot>=0) else 0; self.kc_n=int(self.kc_slot.max()+1) if np.any(self.kc_slot>=0) else 0; shape=(self.candidate_n,self.steps)
  self.step_spike_counts=np.zeros((self.steps,3),np.int32); self.candidate_spike_counts=np.zeros(shape,np.int32); self.candidate_event_counts=np.zeros(shape,np.int32); self.candidate_signed=np.zeros(shape,np.float64); self.candidate_positive=np.zeros(shape,np.float64); self.candidate_negative=np.zeros(shape,np.float64); self.cbkc_event_counts=np.zeros(self.steps,np.int32); self.cbkc_signed=np.zeros(self.steps,np.float64); self.cbkc_positive=np.zeros(self.steps,np.float64); self.cbkc_negative=np.zeros(self.steps,np.float64); self.kc_first_spike_step=np.full(self.kc_n,-1,np.int64)
  self.native=TraceObserver(self.steps,self.candidate_n,self.kc_n,0,*[x.ctypes.data for x in (self.candidate_slot,self.cb_mask,self.kc_slot,self.step_spike_counts,self.candidate_spike_counts,self.candidate_event_counts,self.candidate_signed,self.candidate_positive,self.candidate_negative,self.cbkc_event_counts,self.cbkc_signed,self.cbkc_positive,self.cbkc_negative,self.kc_first_spike_step)],0)
 def reset(self,start_clock=0):
  self.native.overflow=0; self.native.start_clock=int(start_clock); [x.fill(0) for x in (self.step_spike_counts,self.candidate_spike_counts,self.candidate_event_counts,self.candidate_signed,self.candidate_positive,self.candidate_negative,self.cbkc_event_counts,self.cbkc_signed,self.cbkc_positive,self.cbkc_negative)]; self.kc_first_spike_step.fill(-1)
 def fail_if_overflow(self):
  if self.native.overflow: raise RuntimeError('native trace aggregate overflow or step/slot out of bounds')
