import numpy as np
from stonkfly.neural.trace_s2h import TraceS2H

def test_body_id_is_not_graph_node_index():
    z=np.load('data/graph.npz'); node=int(np.flatnonzero(z['ids']==13334)[0]); assert node==3120 and node!=13334
    t=TraceS2H(np.array([-1,0],np.int32),np.array([-1],np.int32),np.array([-1,0],np.int32),node,out_capacity=4)
    assert t.native.target_node==3120

def test_s2h_abi_header_and_overflow_default():
    t=TraceS2H(np.array([0],np.int32),np.array([0],np.int32),np.array([0],np.int32),3120,out_capacity=2)
    assert t.native.abi_version==3 and t.native.magic==TraceS2H.MAGIC and t.native.overflow==0

def test_v3_timestamp_sentinels_and_unique_sequence_capacity():
    t=TraceS2H(np.array([0],np.int32),np.array([0],np.int32),np.array([0],np.int32),3120,out_capacity=2)
    assert t.source_first_delivery[0] == -1 and t.source_pre_first[0] == -1
    assert t.kc_order_id.shape == t.kc_order_step.shape
