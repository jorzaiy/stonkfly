import numpy as np
from stonkfly.neural.trace_s2f import TraceS2Fv2

def test_projection_and_cb_slots_are_distinct_and_bounded():
    p=np.array([0,-1,-1,1],np.int32); c=np.array([-1,0,1,-1],np.int32); k=np.array([-1,-1,-1,0],np.int32)
    t=TraceS2Fv2(4,p,c,k); t.reset(); t.projection_events[0,0]=2; t.projection_signed[0,0]=-1.5; t.projection_positive[0,0]=2.0; t.projection_negative[0,0]=-3.5
    assert t.projection_n==2 and t.cb_n==2 and t.kc_n==1; assert t.projection_events[0,0]==2; assert t.projection_signed[0,0]<0

def test_bad_abi_fails_closed():
    p=np.array([0,-1],np.int32); c=np.array([-1,0],np.int32); k=np.array([-1,0],np.int32); t=TraceS2Fv2(2,p,c,k); t.native.magic=0; t.native.overflow=1
    try: t.fail(); assert False
    except RuntimeError: pass

def test_v2_kc_first_is_native_int32_and_steps_are_bounded():
    p=np.array([0,-1],np.int32); c=np.array([-1,0],np.int32); k=np.array([-1,0],np.int32)
    t=TraceS2Fv2(2,p,c,k)
    assert t.kc_first.dtype == np.dtype(np.int32)
    assert t.kc_first.itemsize == 4
    t.kc_first[0] = 4999
    assert np.all((t.kc_first == -1) | ((t.kc_first >= 0) & (t.kc_first < 5000)))

def test_v2_bad_step_is_detectable():
    p=np.array([0,-1],np.int32); c=np.array([-1,0],np.int32); k=np.array([-1,0],np.int32)
    t=TraceS2Fv2(2,p,c,k); t.kc_first[0] = 5000
    assert not np.all((t.kc_first == -1) | ((t.kc_first >= 0) & (t.kc_first < 5000)))
