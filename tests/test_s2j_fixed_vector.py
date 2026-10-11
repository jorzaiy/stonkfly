import json
from experiments.a_share.run_s2j_fixed_vector_assay import SCALES, OFFSET, stimuli

def test_candidate_enumeration_and_zero_current_missing_semantics():
    assert SCALES == (3.0,6.0,9.0,12.0,18.0,24.0,30.0) and OFFSET == 0.0
    rows={x[0]:x for x in stimuli()}; assert len(rows)==56
    for i in range(8):
        name,v,m=rows[f'ch{i}_missing']; assert m[i] and v[i]==1.0

def test_stimulus_vectors_are_fixed_and_json_serializable():
    for name,v,m in stimuli():
        assert len(v)==len(m)==8 and all(0<=float(x)<=1 for x in v)
    json.dumps([{'name':n,'values':v,'missing':m} for n,v,m in stimuli()],allow_nan=False)
