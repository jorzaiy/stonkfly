from experiments.a_share.run_s2j_fixed_vector_assay import V3_SCALES

def test_v3_preregistered_scales_are_distributed_actual_list():
    assert list(V3_SCALES) == [1.0,2.0,3.0,4.0,6.0,9.0,12.0,18.0]
