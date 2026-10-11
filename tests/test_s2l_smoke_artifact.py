import json
from pathlib import Path

P=Path('runs/plasticity_diagnosis/S2l/20261006T-kc-pre-ignition-input-balance-v7/smoke.json')

def test_kernel_smoke_gates_and_phase_sentinels():
    d=json.loads(P.read_text())
    assert {'source_class_metadata','fixed_low_kcs','apl_summary','summary'} <= set(d)
    assert d['fixed_low_kcs']['status']=='complete' and d['fixed_low_kcs']['intersection_count']==len(d['fixed_low_kcs']['intersection'])
    assert d['fixed_low_kcs']['intersection'] and all(x['frequency']==43 for x in d['fixed_low_kcs']['intersection'])
    assert all('n' in d['summary']['by_class'][c]['low']['pre_kc1_events'] for c in d['summary']['by_class'])
    assert all(r['strict_pre_kc10_events'] is None and 'until_end_without_kc10_events' in r for r in d['apl_summary']['per_frame'] if r['kc10_step'] is None)
    assert d['kc_input_npz']['sha256']
    assert d['parameters']['phase_definition']==d['summary']['phase_definition'] and 'kc1<=step<kc10' in d['parameters']['phase_definition']
    assert all('pre_kc1_signed' in d['summary']['by_class'][c]['low'] for c in d['summary']['by_class'])
    assert all(r['first_delivery_step'] is not None for r in d['apl_summary']['per_frame'])
    assert d['gates']['all_pass'] and d['gates']['trace_off_matches_s2k_c'] and d['gates']['trace_on_off_equivalent']
    for r in d['records']:
        assert r['equivalence']['delivery_conservation']
        assert r['equivalence']['kc_first_count'] and r['equivalence']['kc10_order_matches']
        if r['group']=='low':
            assert r['kc10_step'] is None
            assert sum(row[2] for row in r['phase_events']) == 0
            assert all(row[2] == -1 for row in r['phase_last'])
    assert any(r['group']=='high' and r['kc10_step'] is not None for r in d['records'])
