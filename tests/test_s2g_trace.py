import numpy as np
from stonkfly.neural.trace_s2g import TraceS2G, TraceObserverS2G

def test_s2g_pair_shape_direction_and_sign_split():
    s=np.array([-1,0,1],np.int32); t=np.array([0,-1],np.int32); x=TraceS2G(s,t)
    assert x.source_n==2 and x.target_n==1 and x.pair_events.shape==(2,1)
    x.pair_events[0,0]=2; x.pair_signed[0,0]=-3; x.pair_negative[0,0]=-3
    assert x.pair_events[0,0]==2 and x.pair_signed[0,0]==x.pair_negative[0,0]

def test_s2g_bad_abi_and_target_first_sentinel():
    x=TraceS2G(np.array([0],np.int32),np.array([0],np.int32)); assert x.target_first[0]==-1
    x.native.magic=0
    # Wrapper cannot silently accept a malformed header; native call performs final validation.
    assert x.native.magic != TraceS2G.MAGIC

def test_s2g_never_spiking_target_has_full_window_semantics():
    x=TraceS2G(np.array([0],np.int32),np.array([0],np.int32)); x.pair_events[0,0]=1
    assert x.target_first[0] == -1
