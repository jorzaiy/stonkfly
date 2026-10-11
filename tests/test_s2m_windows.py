import sys
from pathlib import Path
import numpy as np
from stonkfly.neural.trace_s2m import TraceBuffersS2M
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.a_share.smoke_s2m import (
    FORMAL_RAW_KEYS, COMPACT_RECORD_KEYS, build_formal_payload, compact_formal_records,
    formal_window_summary, json_manifest_digest,
    rank_other_cb, verify_manifest_pin, atomic_npz, atomic_json,
)

def classify(d):
    if d<0:return -1
    if 0<=d<64:return 0
    if d<128:return 1
    if d<256:return 2
    if d<512:return 3
    if d<1024:return 4
    return -1
def test_window_boundaries():
    assert [classify(x) for x in (-1,0,63,64,127,128,255,256,511,512,1023,1024)]==[-1,0,0,1,1,2,2,3,3,4,4,-1]
def test_reset_and_cross_class_seen():
    t=TraceBuffersS2M(np.array([0,1],np.int8),np.array([0,0],np.int32));t.events[:]=2;t.reset();assert np.all(t.first==-1) and np.all(t.last==-1);t.target_seen[0,0,0]=1;t.target_seen[0,1,0]=1;assert t.target_seen.sum()==2
def test_window_conservation_fixture():
    t=TraceBuffersS2M(np.array([0],np.int8),np.array([0],np.int32));t.events[0,0]=3;t.events[1,1]=4;t.audit[0]=7;assert t.events.sum()==t.audit[0]

def test_compact_other_cb_mapping_and_sparse_coo():
    t=TraceBuffersS2M(np.array([4, 0, 4], np.int8), np.array([0, 1, 0], np.int32),
                      source_body_ids=np.array([101, 102, 103]),
                      kc_body_ids=np.array([18540, 999]))
    assert t.other_cb_source_mapping()==[
        {'source_index': 0, 'body_id': 101, 'slot': 0},
        {'source_index': 2, 'body_id': 103, 'slot': 1},
    ]
    t.other_cb_events[1, 4]=2; t.other_cb_signed[1, 4]=-0.5
    t.other_cb_negative[1, 4]=-0.5; t.other_cb_first[1, 4]=511; t.other_cb_last[1, 4]=700
    rows=t.other_cb_sparse_coo()
    assert rows==[{'source_slot':1,'source_index':2,'body_id':103,'window':4,'events':2,
                   'signed':-0.5,'positive':0.0,'negative':-0.5,'first':511,'last':700}]

def test_fixed4_shape_mapping_and_signed_split():
    ids=np.array([18540, 21778, 44069, 520206, 999], np.int64)
    t=TraceBuffersS2M(np.zeros(5,np.int8), np.arange(5,dtype=np.int32), kc_body_ids=ids)
    assert t.fixed_events.shape==(4,6,5)
    assert t.kc_fixed_slot[:4].tolist()==[0,1,2,3]
    t.fixed_events[2, 4, 3]=1; t.fixed_signed[2,4,3]=-1.25; t.fixed_negative[2,4,3]=-1.25
    assert t.fixed4_events[2,4,3]==1 and t.fixed4_negative[2,4,3]==-1.25

def _formal_row(group, signed, body=100):
    row={'group':group,'other_cb_source_mapping':[{'source_index':0,'body_id':200,'slot':0},{'source_index':1,'body_id':100,'slot':1}],
         'events':np.zeros((6,5),np.int64).tolist(),'signed':np.zeros((6,5)).tolist(),
         'positive':np.zeros((6,5)).tolist(),'negative':np.zeros((6,5)).tolist(),
         'unique_source':np.zeros((6,5),np.int32).tolist(),'unique_target':np.zeros((6,5),np.int32).tolist(),
         'first':np.full((6,5),-1,np.int32).tolist(),'last':np.full((6,5),-1,np.int32).tolist(),
         'fixed4_events':np.zeros((4,6,5),np.int64).tolist(),'fixed4_signed':np.zeros((4,6,5)).tolist(),
         'fixed4_positive':np.zeros((4,6,5)).tolist(),'fixed4_negative':np.zeros((4,6,5)).tolist(),
         'fixed4_first':np.full((4,6,5),-1,np.int32).tolist(),'fixed4_last':np.full((4,6,5),-1,np.int32).tolist(),
         'other_cb_events':np.zeros((2,5),np.int64).tolist(),'other_cb_signed':np.zeros((2,5)).tolist()}
    row['signed'][4][2]=signed; row['other_cb_signed'][0][2]=signed; row['other_cb_signed'][1][2]=-signed; row['other_cb_events'][0][2]=1; row['other_cb_events'][1][2]=1
    for k in ('positive','negative','other_cb_positive','other_cb_negative'): row[k]=np.zeros((2,5)).tolist() if k.startswith('other') else np.zeros((6,5)).tolist()
    row['other_cb_first']=np.full((2,5),-1,np.int32).tolist();row['other_cb_last']=np.full((2,5),-1,np.int32).tolist()
    row.update(event_index=0,expected_frame_sha256='f',frame_sha256='f',active_kc_count=1,
               expected_active_kc_count=1,kc_spike_count=1,expected_kc_spike_count=1,
               spike_sha256='s',expected_spike_sha256='s',audit=1,kc1_step=0,equivalence={'all':True})
    return row

def test_formal_summary_and_frozen_ranking_tie_break():
    rows=[_formal_row('low',-2,200),_formal_row('high',1,200),_formal_row('low',-1,100),_formal_row('high',2,100)]
    rows[0]['fixed4_events'][1][5][4]=11
    rows[1]['fixed4_events'][1][5][4]=13
    rows[2]['fixed4_events'][1][5][4]=17
    rows[3]['fixed4_events'][1][5][4]=19
    rows[0]['other_cb_events'][0][0]=1
    summary=formal_window_summary(rows)
    assert summary['by_class']['other_cb']['signed']['2']['low']['n']==2
    assert summary['by_class']['other_cb']['signed']['2']['high_low_diff']['mean']==3.0
    assert summary['fixed4_by_target']['21778']['by_class']['other']['events']['4']['all']['mean']==15.0
    assert summary['fixed4_by_target']['21778']['by_class']['KC']['events']['4']['all']['mean']==0.0
    ranked=rank_other_cb(rows)
    assert [x['body_id'] for x in ranked[:2]]==[100,200]
    assert ranked[0]['winning_window']==2 and ranked[0]['support']=={'all':4,'low':2,'high':2}
    assert ranked[1]['support_by_window']['0']=={'all':1,'low':1,'high':0}
    assert ranked[0]['direction']=='negative'

def test_formal_manifest_pin_and_payload_schema(tmp_path):
    rows=[_formal_row('low',-1)]
    payload=build_formal_payload(rows)
    assert set(payload)|{'json_manifest_sha256'} == set(FORMAL_RAW_KEYS)
    raw=tmp_path/'raw.npz'; atomic_npz(raw,x=np.asarray([1]))
    manifest={'raw_npz_sha256':None,'json_manifest_sha256':None,'payload_keys':list(FORMAL_RAW_KEYS)}
    manifest['json_manifest_sha256']=json_manifest_digest(manifest)
    atomic_npz(raw,x=np.asarray([1]),json_manifest_sha256=np.asarray(manifest['json_manifest_sha256']))
    manifest['raw_npz_sha256']=__import__('hashlib').sha256(raw.read_bytes()).hexdigest()
    doc=tmp_path/'manifest.json';atomic_json(doc,manifest)
    assert verify_manifest_pin(doc,raw)['all']

def test_rank_other_cb_vectorized_matches_reference_small():
    rows=[_formal_row('low',-2,200),_formal_row('high',1,200),_formal_row('low',-1,100),_formal_row('high',2,100)]
    got=rank_other_cb(rows)
    signed=np.asarray([r['other_cb_signed'] for r in rows]); events=np.asarray([r['other_cb_events'] for r in rows])
    expected=[]
    for slot,meta in enumerate(rows[0]['other_cb_source_mapping']):
        diffs=signed[[1,3],slot].mean(0)-signed[[0,2],slot].mean(0); win=int(np.argmax(np.abs(diffs)))
        sb={str(w):{'all':int((events[:,slot,w]>0).sum()),'low':int((events[[0,2],slot,w]>0).sum()),'high':int((events[[1,3],slot,w]>0).sum())} for w in range(5)}
        expected.append((meta['body_id'],float(np.abs(diffs).max()),win,diffs.tolist(),sb[str(win)]))
    expected.sort(key=lambda x:(-x[1],x[0]))
    assert [(x['body_id'],x['score'],x['winning_window'],x['window_diff_mean_signed'],x['winning_window_support']) for x in got]==expected

def test_formal_json_records_are_compact():
    compact=compact_formal_records([_formal_row('low',-1)])
    assert set(compact[0])==set(COMPACT_RECORD_KEYS)
    assert not any(k.startswith('other_cb_') or k.startswith('fixed4_') or k in ('events','signed','positive','negative','unique_source','unique_target','first','last') for k in compact[0])
