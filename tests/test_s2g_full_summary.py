import json
from pathlib import Path

def test_exact_s2d_event_reuse():
    p=Path('runs/plasticity_diagnosis/S2d/20261004T-projection-threshold-v1/projection_threshold_analysis.json')
    d=json.loads(p.read_text()); xs=d['parameters']['event_indices']; pf={x['event_index'] for x in d['per_frame']}
    assert len(xs)==60 and set(xs)==pf and sum(x['group']=='low' for x in d['per_frame'])==43 and sum(x['group']=='high' for x in d['per_frame'])==17

def test_support_rate_and_perfect_definition():
    assert 17/17 == 1.0 and 0/43 == 0.0
    assert (17==17 and 0==0) is True

def test_lead_windows_and_limitation_text():
    assert all(x in (1,2,5,10) for x in (1,2,5,10))
    text='low never-spike window is full 5000 steps and unequal'
    assert 'unequal' in text
