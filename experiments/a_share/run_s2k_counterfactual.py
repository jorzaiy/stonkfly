"""S2k bounded counterfactual diagnostic; never modifies graph/checkpoint on disk."""
import argparse, hashlib, json, tempfile
from pathlib import Path
import numpy as np
from experiments.a_share.analyze_kc_overlap import render_frame_from_observation
from stonkfly.config import Settings
from stonkfly.neural.v51_controller import V51Controller

RUN=Path('runs/wc006_purged_fly_training/20261004T-balanced-reward-v2')
S2D=Path('runs/plasticity_diagnosis/S2d/20261004T-projection-threshold-v1/projection_threshold_analysis.json')
ARMS=('C','A','B2','B4','AB2','AB4')
def sha(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def file_sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_events():
    d=json.loads(S2D.read_text()); idx=[int(x) for x in d['parameters']['event_indices']]
    lines=[json.loads(x) for x in (RUN/'events.jsonl').read_text().splitlines()]
    expected={int(x['event_index']):x for x in d['per_frame']}
    if len(idx)!=60 or len(set(idx))!=60: raise RuntimeError('event list is not exact 60')
    out=[]
    for i in idx:
        ev=lines[i]; frame=render_frame_from_observation(ev['observation']); fh=sha(frame)
        if fh != ev['neural']['input_sha256'] or fh != expected[i]['frame_sha256']: raise RuntimeError(f'frame hash mismatch {i}')
        out.append((i,ev,frame,expected[i]))
    return out
def masks(b):
    pre=np.searchsorted(b.ptr,np.arange(len(b.post)),side='right')-1; post=np.asarray(b.post)
    ann_types=np.asarray(b.superclass).astype(str)
    # KC mask is authoritative circuit identification; APL uses annotation-backed body IDs/types.
    kc=np.zeros(b.n,bool); kc[np.asarray(b.circuit['kc'],dtype=int)]=True
    from stonkfly.neural.common import annotations
    a=annotations(b.ids); types=a.type.fillna('').astype(str).to_numpy()
    apl=np.flatnonzero(types=='APL'); apl_edge=np.isin(pre,apl)&kc[post]
    kcedge=kc[pre]&kc[post]
    return kcedge,apl_edge,{'kc_kc_edges':int(kcedge.sum()),'apl_kc_edges':int(apl_edge.sum()),'apl_body_ids':[int(b.ids[i]) for i in apl]}
def overlay(base,kcedge,apl_edge,arm):
    w=base.copy();
    if arm in ('A','AB2','AB4'): w[kcedge]=0.0
    if arm in ('B2','AB2'): w[apl_edge]*=2.0
    if arm in ('B4','AB4'): w[apl_edge]*=4.0
    return w
def run(output, smoke=False, isolated_source=None, isolated_library=None):
    if isolated_source and isolated_library:
        from stonkfly.neural import brain as brain_module
        brain_module.SOURCE=Path(isolated_source)
        brain_module.LIBRARY=Path(isolated_library)
    events=load_events(); selected=events[:2] if smoke else events
    c=V51Controller(Settings(neural_ms=500,pulse_ms=200,learning=False)); b=c.brain
    clean=RUN/'brain-clean.npz';
    if file_sha(clean)!='a92f547e720e7d9830ae519f26c0bacdf4f76c59761fb2be4adb6f34cac2a225': raise RuntimeError('clean checkpoint hash mismatch')
    clean_local=clean
    base=b.weight.copy(); kcedge,apl_edge,edge_meta=masks(b); kc=np.asarray(b.circuit['kc'],dtype=int)
    records=[]; gates=[]
    for arm in ARMS:
      for event_index,ev,frame,expected in selected:
        c.restore(clean_local); b.weight[:]=overlay(base,kcedge,apl_edge,arm); before=b.weight.copy()
        counts,wall=b.rgb_step(frame,500.0,learning=False); b.counts[:]=counts
        active=np.flatnonzero(counts[kc]>0); h=sha(counts.astype(np.int32));
        # Repeat same clean restore/overlay as the observer on/off/state-isolation gate.
        c.restore(clean_local); b.weight[:]=overlay(base,kcedge,apl_edge,arm); counts2,_=b.rgb_step(frame,500.0,learning=False)
        repeat=bool(np.array_equal(counts,counts2) and h==sha(counts2.astype(np.int32)))
        c_gate=(arm=='C' and int(len(active))==int(expected['active_kc_count']))
        gates.append({'arm':arm,'event_index':event_index,'repeat_equal':repeat,'C_matches_S2_count':c_gate if arm=='C' else True,'weights_restored_before_run':bool(np.array_equal(before,overlay(base,kcedge,apl_edge,arm)))})
        records.append({'arm':arm,'event_index':event_index,'date':ev['date'],'category':expected['category'],'frame_sha256':sha(frame),'active_kc_count':int(len(active)),'kc_spike_count':int(counts[kc].sum()),'spike_sha256':h,'kc_set_hash':sha(kc[active]),'total_spikes':int(counts.sum()),'wall_seconds':float(wall),'counterfactual':arm!='C','production_eligible':arm=='C','learning':False})
    # full runs use compact records; smoke gates are retained and formal launch is only allowed after these pass.
    all_gates=all(x['repeat_equal'] and x['C_matches_S2_count'] for x in gates)
    result={'counterfactual':True,'production_eligible':False,'arms':list(ARMS),'events':{'count':len(events),'selected_for_run':len(selected),'source_path':str(RUN/'events.jsonl'),'events_sha256':file_sha(RUN/'events.jsonl')},'clean_checkpoint':{'path':str(clean),'sha256':file_sha(clean)},'graph':{'path':'data/graph.npz','sha256':'346b8af85a11af13b8324e18669812c1924569e7d1adcb4e6f45cc461a2c344b'},'edge_overlay':edge_meta|{'kc_kc_baseline_signs':{'positive':int((base[kcedge]>0).sum()),'negative':int((base[kcedge]<0).sum()),'zero':int((base[kcedge]==0).sum())},'apl_kc_baseline_signs':{'positive':int((base[apl_edge]>0).sum()),'negative':int((base[apl_edge]<0).sum()),'zero':int((base[apl_edge]==0).sum())},'kcedge_signs':'all retained KC→KC weights set to zero in A arms','apl_multiplier':{'C':1,'A':1,'B2':2,'B4':4,'AB2':2,'AB4':4}},'equivalence_limits':{'C_spike_hash_comparison':None,'reason':'S2 artifact stores verified frame hashes and KC counts, but not authoritative full spike hashes; C KC-count equality is checked.'},'gates':{'all_pass':all_gates,'details':gates},'records':records,'limitations':['Counterfactual diagnostic only; no graph topology deletion and no persistent weight mutation.','No market, reward, label, decoder, trade, or performance selection.']}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2,allow_nan=False));
    if not all_gates: raise RuntimeError('S2k gate failure; formal launch prohibited')
    return result
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--isolated-source',type=Path);ap.add_argument('--isolated-library',type=Path);a=ap.parse_args();r=run(a.output,a.smoke,a.isolated_source,a.isolated_library);print(json.dumps({'records':len(r['records']),'gates':r['gates']['all_pass'],'path':str(a.output)}))
if __name__=='__main__': main()
