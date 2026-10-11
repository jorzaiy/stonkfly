import ctypes as C
import numpy as np
import pytest
from stonkfly.neural.trace_s2l import TraceBuffersS2L

def toy():
    sc=np.array([0,1,2,3,4,5],np.int8)
    ks=np.array([0,0,1,-1,-1,-1],np.int32)
    return TraceBuffersS2L(sc,ks)

def test_reset_sentinels_twice_and_shapes():
    t=toy(); t.kc_first[:]=4; t.kc1_step[:]=5; t.kc10_step[:]=6; t.phase_first[:]=7; t.phase_last[:]=8
    t.reset(); t.reset()
    assert np.all(t.kc_first==-1) and np.all(t.kc1_step==-1) and np.all(t.kc10_step==-1)
    assert np.all(t.phase_first==-1) and np.all(t.phase_last==-1)
    assert t.target_seen.shape==(2,6,3)

def test_cross_class_target_seen_is_independent():
    t=toy(); t.target_seen[0,0,1]=1; t.target_seen[0,1,1]=1
    assert int(t.target_seen[:, :, 1].sum())==2

def test_abi_fail_closed():
    t=toy(); t.native.magic=0; t.native.overflow=1
    with pytest.raises(RuntimeError): t.fail_if_overflow()

def test_class_phase_conservation_fixture():
    t=toy(); t.phase_events[0,0]=3; t.phase_events[1,1]=4; t.all_kc_delivery_count[0]=7
    assert int(t.phase_events.sum())==int(t.all_kc_delivery_count[0])

def test_missing_kc10_sentinel():
    t=toy(); t.kc_unique[0]=4
    assert int(t.kc10_step[0])==-1
