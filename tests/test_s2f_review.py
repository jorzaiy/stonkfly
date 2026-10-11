import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.a_share.summarize_s2f_review import ratio, classify

def test_rate_difference_uses_rates_not_raw_group_sizes():
    assert (9/17)-(18/43) != 9-18

def test_high_only_classes_and_zero_ratio_semantics():
    rows=[{'body_id':1,'type':'x','low_frames':0,'high_frames':17,'low_rate':0,'high_rate':1,'rate_difference':1}, {'body_id':2,'type':'x','low_frames':1,'high_frames':1,'low_rate':1/43,'high_rate':1/17,'rate_difference':1/17-1/43}]
    c=classify(rows)
    assert [x['body_id'] for x in c['perfect_high_only']]==[1]
    assert [x['body_id'] for x in c['any_high_only']]==[1]
    assert ratio(1,0) is None

def test_structural_direction_is_source_to_target():
    assert 'source_to_cb_target' in 'projection_source_to_cb_target'
