"""S2m smoke and (uninvoked) 60-frame formal runner.

This module keeps observation, propagation, and fixed decoding separate.  It
never enables learning or execution.  The smoke command is intentionally
small; ``run_formal`` is a separate explicit API and is not called here.
"""
import argparse
import ctypes as C
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from stonkfly.config import Settings
from stonkfly.neural.v51_controller import V51Controller
from stonkfly.neural.trace_s2m import (FIXED_KC_BODY_IDS, S2M_CLASSES,
    TraceBuffersS2M, WINDOW_BOUNDS)
from experiments.a_share.analyze_kc_overlap import render_frame_from_observation
from experiments.a_share.analyze_s2l_kc_balance import events, classify

ROOT=Path(__file__).resolve().parents[2]
S2D=ROOT/'runs/plasticity_diagnosis/S2d/20261004T-projection-threshold-v1/projection_threshold_analysis.json'
S2K=ROOT/'runs/plasticity_diagnosis/S2k/20261006T-kc-recurrence-counterfactual-v1/s2k_counterfactual.json'
HIST=ROOT/'runs/wc006_purged_fly_training/20261004T-balanced-reward-v2/brain-clean.npz'
PREREG=ROOT/'docs/20261006-s2m-kc1-window-balance.md'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ah(value): return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False,sort_keys=True)+'\n'); tmp.replace(path)
def atomic_npz(path, **arrays):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix('.partial.npz')
    with tmp.open('wb') as h: np.savez_compressed(h,**arrays)
    tmp.replace(path)

def _snapshot(b):
    out={'weight':b.weight.copy()}
    for k in b.fields: out[k]=getattr(b,k).copy()
    for k in ('cursor','sim_ms','total_spikes','weights_frozen'): out[k]=getattr(b,k)
    return out
def _equal(a,b):
    return np.array_equal(a,b) if isinstance(a,np.ndarray) else a==b
def _bridge(fresh, historical):
    fields=[]
    with np.load(historical,allow_pickle=False) as old, np.load(fresh,allow_pickle=False) as new:
        for key in ('weight',*('v','g','refractory','drive','previous_drive','queue','queue_count','counts','luminance','active','active_flag','nactive','last','eligibility','eligibility_last','modulation','modulation_last','adaptation')):
            ok=key in old.files and key in new.files and old[key].shape==new[key].shape and old[key].dtype==new[key].dtype and np.array_equal(old[key],new[key])
            fields.append({'field':key,'bitwise_equal':bool(ok)})
    return {'all_persistent_arrays_bitwise_equal':all(x['bitwise_equal'] for x in fields),'fields':fields}

def _one(c, clean, frame, sc, ks, body_ids):
    b=c.brain
    c.restore(clean); b.trace_observer_s2m=None; off_result=c.observe(frame,'none'); off_counts=b.counts.copy(); off_state=_snapshot(b)
    c.restore(clean); t=TraceBuffersS2M(sc,ks,source_body_ids=body_ids,kc_body_ids=body_ids[ks>=0] if np.any(ks>=0) else np.empty(0,np.int64)); t.reset(); b.trace_observer_s2m=C.addressof(t.native)
    on_result=c.observe(frame,'none'); b.trace_observer_s2m=None; t.fail(); on_state=_snapshot(b)
    fields_equal={k:_equal(off_state[k],on_state[k]) for k in off_state}
    class_events=t.events.sum(axis=1).astype(np.int64)
    other_events=t.other_cb_events.sum(axis=1).astype(np.int64)
    fixed_events=t.fixed_events.sum(axis=(0,1)).astype(np.int64)
    # Every compact source is class-4 by construction; fixed-four is a
    # target decomposition and therefore cannot exceed its class totals.
    gates={
      'trace_off_frame_hash': off_result['input_sha256']==on_result['input_sha256'],
      'trace_off_active_kc_count': int(np.count_nonzero(off_counts[ks>=0]))==int(np.count_nonzero(on_result and b.counts[ks>=0])),
      'trace_off_kc_spike_count': int(off_counts[ks>=0].sum())==int(b.counts[ks>=0].sum()),
      'trace_off_spike_hash': off_result['spike_sha256']==on_result['spike_sha256'],
      'trace_on_off_all_fields': all(fields_equal.values()),
      'delivery_conservation': int(t.events.sum())==int(t.audit[0]),
      'source_class_totals': bool(np.array_equal(t.other_cb_events.sum(axis=0),t.events[4])),
      'fixed4_decomposition': bool(np.all(t.fixed_events.sum(axis=0)<=t.events)),
      'other_cb_sparse_reconstructs': int(sum(x['events'] for x in t.other_cb_sparse_coo()))==int(other_events.sum()),
    }
    gates['all']=all(gates.values())
    return {'frame_sha256':ah(frame),'active_kc_count':int(np.count_nonzero(off_counts[ks>=0])),'kc_spike_count':int(off_counts[ks>=0].sum()),'spike_sha256':off_result['spike_sha256'],'kc1_step':int(t.kc1_step[0]),'events':t.events.tolist(),'signed':t.signed.tolist(),'positive':t.positive.tolist(),'negative':t.negative.tolist(),'first':t.first.tolist(),'last':t.last.tolist(),'unique_source':t.unique_source.tolist(),'unique_target':t.unique_target.tolist(),'audit':int(t.audit[0]),'other_cb_source_mapping':t.other_cb_source_mapping(),'other_cb_sparse_coo':t.other_cb_sparse_coo(),'other_cb_events':t.other_cb_events.tolist(),'other_cb_signed':t.other_cb_signed.tolist(),'other_cb_positive':t.other_cb_positive.tolist(),'other_cb_negative':t.other_cb_negative.tolist(),'other_cb_first':t.other_cb_first.tolist(),'other_cb_last':t.other_cb_last.tolist(),'fixed4_body_ids':list(map(int,FIXED_KC_BODY_IDS)),'fixed4_events':t.fixed_events.tolist(),'fixed4_signed':t.fixed_signed.tolist(),'fixed4_positive':t.fixed_positive.tolist(),'fixed4_negative':t.fixed_negative.tolist(),'fixed4_first':t.fixed_first.tolist(),'fixed4_last':t.fixed_last.tolist(),'equivalence':gates}

def run_smoke(out=None):
    out=Path(out or ROOT/'runs/plasticity_diagnosis/S2m/20261006T-kc1-window-balance-smoke/s2m_smoke.json')
    c=V51Controller(Settings(neural_ms=500,pulse_ms=200,learning=False)); b=c.brain
    sc,typ,sup,kc=classify(b); ks=np.full(b.n,-1,np.int32); ks[kc]=np.arange(len(kc),dtype=np.int32); body_ids=np.asarray(b.ids,np.int64)
    clean=out.parent/'brain-clean-observer-build.npz'; c.save(clean); bridge=_bridge(clean,HIST) if HIST.exists() else {'all_persistent_arrays_bitwise_equal':False,'reason':'historical checkpoint missing'}
    frames=events(); selected=[x for x in frames if x[2]['active_kc_count']<=10][:2]+[x for x in frames if x[2]['active_kc_count']>=1000][:2]
    refs={int(x['event_index']):x for x in json.loads(S2K.read_text()).get('records',[]) if x.get('arm')=='C'} if S2K.exists() else {}
    records=[]
    for ei,ev,expected in selected:
        row=_one(c,clean,render_frame_from_observation(ev['observation']),sc,ks,body_ids); row.update(event_index=int(ei),group='low' if expected['active_kc_count']<=10 else 'high',expected_active_kc_count=int(expected['active_kc_count']))
        ref=refs.get(int(ei))
        if ref:
            row['equivalence'].update(trace_off_s2k_frame_hash=row['frame_sha256']==ref['frame_sha256'],trace_off_s2k_active_kc_count=row['active_kc_count']==int(ref['active_kc_count']),trace_off_s2k_kc_spike_count=row['kc_spike_count']==int(ref['kc_spike_count']),trace_off_s2k_spike_hash=row['spike_sha256']==ref['spike_sha256'])
            row['equivalence']['all']=all(row['equivalence'].values())
        records.append(row)
    gates={'fresh_observer_checkpoint':bool(bridge['all_persistent_arrays_bitwise_equal']),'records':len(records)==4,'low_n':sum(x['group']=='low' for x in records)==2,'high_n':sum(x['group']=='high' for x in records)==2,'all_records':all(x['equivalence']['all'] for x in records)}
    gates['all']=all(gates.values())
    result={'observer':'S2m','preregistered':True,'counterfactual':False,'learning':False,'diagnostic_only':True,'production_eligible':False,'parameters':{'classes':list(S2M_CLASSES),'window_bounds':[list(x) for x in WINDOW_BOUNDS],'fixed_kc_body_ids':list(FIXED_KC_BODY_IDS),'selected_count':len(records)},'provenance':{'s2d':str(S2D),'s2d_sha256':sha(S2D) if S2D.exists() else None,'kernel_source_sha256':b.build['source_sha256'],'kernel_binary_sha256':b.build['binary_sha256'],'observer_checkpoint':str(clean),'historical_checkpoint':str(HIST),'historical_checkpoint_sha256':sha(HIST) if HIST.exists() else None,'script_sha256':sha(Path(__file__))},'checkpoint_bridge':bridge,'records':records,'gates':gates}
    atomic_json(out,result); return result

FORMAL_METRICS=('events','signed','positive','negative','unique_source','unique_target')
FIXED4_METRICS=('events','signed','positive','negative','first','last')
FORMAL_RAW_KEYS=('event_indices','groups','frame_sha256','expected_frame_sha256','active_kc_count','expected_active_kc_count','kc_spike_count','expected_kc_spike_count','spike_sha256','expected_spike_sha256','events','signed','positive','negative','unique_source','unique_target','first','last','audit','other_cb_events','other_cb_signed','other_cb_positive','other_cb_negative','other_cb_first','other_cb_last','other_cb_source_index','other_cb_body_id','fixed4_body_id','fixed4_events','fixed4_signed','fixed4_positive','fixed4_negative','fixed4_first','fixed4_last','json_manifest_sha256')
COMPACT_RECORD_KEYS=('event_index','group','frame_sha256','expected_frame_sha256','active_kc_count','expected_active_kc_count','kc_spike_count','expected_kc_spike_count','spike_sha256','expected_spike_sha256','kc1_step','audit','equivalence')

def summary_stats(values):
    """Stable descriptive summary used by the preregistered report."""
    a=np.asarray(values,dtype=np.float64).reshape(-1)
    return {'n':int(a.size),'min':float(a.min()) if a.size else None,
            'median':float(np.median(a)) if a.size else None,
            'mean':float(a.mean()) if a.size else None,
            'max':float(a.max()) if a.size else None}

def _window_summary(rows, names, fields):
    result={}
    for name,field in zip(names,fields):
        windows={}
        for w in range(5):
            by_group={}
            for group in ('all','low','high'):
                rr=rows if group=='all' else [r for r in rows if r['group']==group]
                by_group[group]=summary_stats([np.asarray(r[field])[name,w] for r in rr])
            low=by_group['low']; high=by_group['high']
            by_group['high_low_diff']={'mean':None if low['mean'] is None or high['mean'] is None else high['mean']-low['mean'],
                                       'median':None if low['median'] is None or high['median'] is None else high['median']-low['median']}
            windows[str(w)]=by_group
        result[name]=windows
    return result

def _fixed_window_summary(rows, target_slot, class_slot, field):
    windows={}
    for w in range(5):
        by_group={}
        for group in ('all','low','high'):
            rr=rows if group=='all' else [r for r in rows if r['group']==group]
            by_group[group]=summary_stats([np.asarray(r[field])[target_slot,class_slot,w] for r in rr])
        low=by_group['low']; high=by_group['high']
        by_group['high_low_diff']={'mean':None if low['mean'] is None or high['mean'] is None else high['mean']-low['mean'],
                                   'median':None if low['median'] is None or high['median'] is None else high['median']-low['median']}
        windows[str(w)]=by_group
    return windows

def formal_window_summary(rows):
    """Return all/low/high summaries for classes and fixed-four targets."""
    classes={}
    for ci,name in enumerate(S2M_CLASSES):
        classes[name]={}
        for metric in FORMAL_METRICS:
            classes[name][metric]=_window_summary(rows,[ci],[metric])[ci]
    fixed={}
    for fi,body in enumerate(FIXED_KC_BODY_IDS):
        fixed[str(body)]={'by_class':{}}
        for ci,class_name in enumerate(S2M_CLASSES):
            fixed[str(body)]['by_class'][class_name]={}
            for metric in FIXED4_METRICS:
                fixed[str(body)]['by_class'][class_name][metric]=_fixed_window_summary(
                    rows,fi,ci,'fixed4_'+metric)
    return {'low_n':sum(r['group']=='low' for r in rows),'high_n':sum(r['group']=='high' for r in rows),
            'window_bounds':[list(x) for x in WINDOW_BOUNDS],'by_class':classes,'fixed4_by_target':fixed}

def rank_other_cb(rows):
    """Frozen score: max window abs(high-low mean signed), body-id tie break."""
    if not rows: return []
    # Materialize each dense array once.  The old implementation converted
    # the full frame×source×window matrix once per source slot, which made the
    # 28k-source formal report unnecessarily quadratic in Python overhead.
    signed=np.asarray([r['other_cb_signed'] for r in rows],dtype=np.float64)
    events_arr=np.asarray([r['other_cb_events'] for r in rows],dtype=np.int64)
    groups=np.asarray([r['group'] for r in rows])
    low_mask=groups=='low'; high_mask=groups=='high'
    if not low_mask.any() or not high_mask.any(): raise ValueError('ranking requires low and high frames')
    diffs=signed[high_mask].mean(axis=0)-signed[low_mask].mean(axis=0)
    scores=np.max(np.abs(diffs),axis=1); winners=np.argmax(np.abs(diffs),axis=1)
    support_all=(events_arr>0).sum(axis=0); support_low=(events_arr[low_mask]>0).sum(axis=0); support_high=(events_arr[high_mask]>0).sum(axis=0)
    mappings=rows[0]['other_cb_source_mapping']; body_ids=np.asarray([m['body_id'] for m in mappings],dtype=np.int64)
    order=np.lexsort((body_ids,-scores))[:20]
    ranking=[]
    for slot in order:
        slot=int(slot); win=int(winners[slot]); support_by_window={str(w):{'all':int(support_all[slot,w]),'low':int(support_low[slot,w]),'high':int(support_high[slot,w])} for w in range(5)}
        winning_support=support_by_window[str(win)]; d=diffs[slot]
        ranking.append({'body_id':int(mappings[slot]['body_id']),'source_index':int(mappings[slot]['source_index']),'source_slot':slot,'score':float(scores[slot]),'winning_window':win,'window_diff_mean_signed':d.tolist(),'support_by_window':support_by_window,'winning_window_support':winning_support,'support':winning_support,'direction':'positive' if d[win]>0 else 'negative' if d[win]<0 else 'zero'})
    return ranking

def json_manifest_digest(manifest):
    """Digest the canonical JSON manifest with hash pins nulled."""
    probe=dict(manifest); probe['raw_npz_sha256']=None; probe['json_manifest_sha256']=None
    return hashlib.sha256(json.dumps(probe,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()

def verify_manifest_pin(json_path,npz_path):
    with Path(json_path).open() as h: manifest=json.load(h)
    with np.load(npz_path,allow_pickle=False) as z: pinned=str(np.asarray(z['json_manifest_sha256']).item())
    return {'json_manifest_sha256':manifest.get('json_manifest_sha256')==json_manifest_digest(manifest),
            'npz_manifest_sha256':pinned==manifest.get('json_manifest_sha256'),
            'raw_npz_sha256':sha(npz_path)==manifest.get('raw_npz_sha256'),
            'all':manifest.get('json_manifest_sha256')==json_manifest_digest(manifest) and pinned==manifest.get('json_manifest_sha256') and sha(npz_path)==manifest.get('raw_npz_sha256')}

def build_formal_payload(rows):
    """Materialize the complete raw NPZ schema, excluding the manifest pin."""
    if not rows: raise ValueError('formal payload requires non-empty rows')
    return {
      'event_indices':np.asarray([r['event_index'] for r in rows],np.int32),
      'groups':np.asarray([r['group'] for r in rows]),
      'frame_sha256':np.asarray([r['frame_sha256'] for r in rows]),
      'expected_frame_sha256':np.asarray([r['expected_frame_sha256'] for r in rows]),
      'active_kc_count':np.asarray([r['active_kc_count'] for r in rows],np.int32),
      'expected_active_kc_count':np.asarray([r['expected_active_kc_count'] for r in rows],np.int32),
      'kc_spike_count':np.asarray([r['kc_spike_count'] for r in rows],np.int64),
      'expected_kc_spike_count':np.asarray([r['expected_kc_spike_count'] for r in rows],np.int64),
      'spike_sha256':np.asarray([r['spike_sha256'] for r in rows]),
      'expected_spike_sha256':np.asarray([r['expected_spike_sha256'] for r in rows]),
      'events':np.asarray([r['events'] for r in rows],np.int64),
      'signed':np.asarray([r['signed'] for r in rows]),
      'positive':np.asarray([r['positive'] for r in rows]),
      'negative':np.asarray([r['negative'] for r in rows]),
      'unique_source':np.asarray([r['unique_source'] for r in rows],np.int32),
      'unique_target':np.asarray([r['unique_target'] for r in rows],np.int32),
      'first':np.asarray([r['first'] for r in rows],np.int32),
      'last':np.asarray([r['last'] for r in rows],np.int32),
      'audit':np.asarray([r['audit'] for r in rows],np.int64),
      'other_cb_events':np.asarray([r['other_cb_events'] for r in rows],np.int64),
      'other_cb_signed':np.asarray([r['other_cb_signed'] for r in rows]),
      'other_cb_positive':np.asarray([r['other_cb_positive'] for r in rows]),
      'other_cb_negative':np.asarray([r['other_cb_negative'] for r in rows]),
      'other_cb_first':np.asarray([r['other_cb_first'] for r in rows],np.int32),
      'other_cb_last':np.asarray([r['other_cb_last'] for r in rows],np.int32),
      'other_cb_source_index':np.asarray([m['source_index'] for m in rows[0]['other_cb_source_mapping']],np.int32),
      'other_cb_body_id':np.asarray([m['body_id'] for m in rows[0]['other_cb_source_mapping']],np.int64),
      'fixed4_body_id':np.asarray(FIXED_KC_BODY_IDS,np.int64),
      'fixed4_events':np.asarray([r['fixed4_events'] for r in rows],np.int64),
      'fixed4_signed':np.asarray([r['fixed4_signed'] for r in rows]),
      'fixed4_positive':np.asarray([r['fixed4_positive'] for r in rows]),
      'fixed4_negative':np.asarray([r['fixed4_negative'] for r in rows]),
      'fixed4_first':np.asarray([r['fixed4_first'] for r in rows],np.int32),
      'fixed4_last':np.asarray([r['fixed4_last'] for r in rows],np.int32),
    }

def compact_formal_records(rows):
    """Keep per-frame JSON metadata/gates; dense arrays remain NPZ-only."""
    return [{key:row[key] for key in COMPACT_RECORD_KEYS} for row in rows]

def _formal_failure(output, error):
    success=Path(output)/'s2m_formal.json'
    if success.exists(): success.unlink()
    atomic_json(Path(output)/'s2m_formal_failure.json',{'observer':'S2m','formal':True,'success':False,'diagnostic_only':True,'counterfactual':False,'production_eligible':False,'error':str(error)})

def run_formal(output):
    """Frozen 60-frame runner; every gate must pass before success output."""
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    try:
        frames=events(); refs={int(x['event_index']):x for x in json.loads(S2K.read_text()).get('records',[]) if x.get('arm')=='C'}
        if len(frames)!=60 or len(refs)!=60 or {int(x[0]) for x in frames}!=set(refs): raise RuntimeError('exact 60 S2d/S2k-C events and references required')
        c=V51Controller(Settings(neural_ms=500,pulse_ms=200,learning=False)); b=c.brain; sc,typ,sup,kc=classify(b); ks=np.full(b.n,-1,np.int32);ks[kc]=np.arange(len(kc),dtype=np.int32); ids=np.asarray(b.ids,np.int64); clean=output/'brain-clean-observer-build.npz'; c.save(clean); bridge=_bridge(clean,HIST)
        if not bridge['all_persistent_arrays_bitwise_equal']: raise RuntimeError('fresh observer checkpoint bridge failed')
        rows=[]
        for ei,ev,expected in frames:
            ref=refs[int(ei)]; row=_one(c,clean,render_frame_from_observation(ev['observation']),sc,ks,ids)
            row.update(event_index=int(ei),group='low' if int(expected['active_kc_count'])<=10 else 'high',expected_frame_sha256=ref['frame_sha256'],expected_active_kc_count=int(ref['active_kc_count']),expected_kc_spike_count=int(ref['kc_spike_count']),expected_spike_sha256=ref['spike_sha256'])
            row['equivalence'].update(trace_off_s2k_frame_hash=row['frame_sha256']==ref['frame_sha256'],trace_off_s2k_active_kc_count=row['active_kc_count']==int(ref['active_kc_count']),trace_off_s2k_kc_spike_count=row['kc_spike_count']==int(ref['kc_spike_count']),trace_off_s2k_spike_hash=row['spike_sha256']==ref['spike_sha256'],kc1_present=row['kc1_step']>=0)
            row['equivalence']['all']=all(row['equivalence'].values()); rows.append(row)
        if sum(r['group']=='low' for r in rows)!=43 or sum(r['group']=='high' for r in rows)!=17: raise RuntimeError('formal groups must be low43/high17')
        if not all(r['equivalence']['all'] for r in rows): raise RuntimeError('formal per-frame equivalence gate failed')
        raw=output/'s2m_raw.npz'; payload=build_formal_payload(rows); atomic_npz(raw,**payload);
        config_bytes=json.dumps(b.configuration_signature(),sort_keys=True,allow_nan=False,separators=(',',':')).encode(); summary=formal_window_summary(rows); result={'observer':'S2m','formal':True,'success':True,'diagnostic_only':True,'counterfactual':False,'production_eligible':False,'window_summary':summary,'top20_other_cb':rank_other_cb(rows),'source_mapping':rows[0]['other_cb_source_mapping'],'class_mapping':{str(i):name for i,name in enumerate(S2M_CLASSES)},'fixed4_body_ids':list(FIXED_KC_BODY_IDS),'raw_npz':str(raw),'raw_npz_sha256':sha(raw),'json_manifest_sha256':None,'records':compact_formal_records(rows),'checkpoint_bridge':bridge,'provenance':{'pre_registration':str(PREREG),'pre_registration_sha256':sha(PREREG),'s2d':str(S2D),'s2d_sha256':sha(S2D),'s2k':str(S2K),'s2k_sha256':sha(S2K),'kernel_source_sha256':b.build['source_sha256'],'kernel_binary_sha256':b.build['binary_sha256'],'script_sha256':sha(Path(__file__)),'configuration_sha256':hashlib.sha256(config_bytes).hexdigest(),'historical_checkpoint':str(HIST),'historical_checkpoint_sha256':sha(HIST),'fresh_checkpoint':str(clean),'fresh_checkpoint_sha256':sha(clean)}}
        result['json_manifest_sha256']=json_manifest_digest(result); payload['json_manifest_sha256']=np.asarray(result['json_manifest_sha256']); atomic_npz(raw,**payload); result['raw_npz_sha256']=sha(raw)
        candidate=output/'s2m_formal.partial.json'; atomic_json(candidate,result)
        if not verify_manifest_pin(candidate,raw)['all']: raise RuntimeError('manifest pin self-check failed')
        candidate.replace(output/'s2m_formal.json')
        return result
    except Exception as exc:
        _formal_failure(output,exc); raise

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path); ap.add_argument('--formal-output',type=Path); a=ap.parse_args()
    if a.formal_output is not None: raise RuntimeError('formal runner is implemented but disabled in smoke mode; obtain explicit approval first')
    r=run_smoke(a.output); print(json.dumps({'path':str(a.output or ROOT/'runs/plasticity_diagnosis/S2m/20261006T-kc1-window-balance-smoke/s2m_smoke.json'),'gates':r['gates']}))
if __name__=='__main__': main()
