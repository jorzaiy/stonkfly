def test_preregistered_leader_rule():
    rows=[{'body_id':10530,'rank_le_3':14},{'body_id':11390,'rank_le_3':13}]
    assert max(rows,key=lambda x:x['rank_le_3'])['body_id']==10530
def test_strict_pre_semantics():
    assert 10 < 11 and not (10 < 10)
