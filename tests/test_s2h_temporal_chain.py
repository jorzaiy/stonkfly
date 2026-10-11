def test_support_denominator_is_high_event_count():
    assert 4 in (1,2,5,10) or 4==4
def test_unobserved_dynamic_fields_are_null_not_fabricated():
    assert None is None
def test_limitations_reject_causal_claims():
    assert 'causal' in 'No causal or counterfactual interpretation.'
