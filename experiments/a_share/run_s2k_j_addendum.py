"""S2k-J lossless active-KC bitsets and pre-registered Jaccard addendum."""
import argparse, hashlib, json, os, tempfile
from pathlib import Path
import numpy as np
from experiments.a_share.run_s2k_counterfactual import ARMS, RUN, load_events, masks, overlay
from stonkfly.config import Settings
from stonkfly.neural.v51_controller import V51Controller

ORIG=Path('runs/plasticity_diagnosis/S2k/20261006T-kc-recurrence-counterfactual-v1/s2k_counterfactual.json')
SOURCE_SHA='e2d4d584f4633613277bb1bf6ea7ba20855c01a332ae2e9344ed7782604c8509'; BINARY_SHA='985213feba8cce34bb1c88187e673f6c77dea541116a6c44671037f2d8675224'; CHECKPOINT_SHA='a92f547e720e7d9830ae519f26c0bacdf4f76c59761fb2be4adb6f34cac2a225'
def sha(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def fsha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def qstats(x):
    a=np.asarray(x,dtype=float)
    return {'n':int(a.size),'min':float(np.min(a)),'q25':float(np.quantile(a,.25,method='linear')),'median':float(np.quantile(a,.5,method='linear')),'mean':float(np.mean(a)),'q75':float(np.quantile(a,.75,method='linear')),'max':float(np.max(a))}
def jac(a,b):
    inter=np.count_nonzero(a & b); union=np.count_nonzero(a | b)
    return 1.0 if union==0 else float(inter/union)
def summarize(bitsets, arms, events):
    out={}; cats=[x[1] for x in events]
    for ai,arm in enumerate(arms):
        x=bitsets[ai]; vals=[]; same=[]; cross=[]
        for i in range(len(x)):
            for j in range(i+1,len(x)):
                z=jac(x[i],x[j]); vals.append(z); (same if cats[i]==cats[j] else cross).append(z)
        out[arm]={'within_arm_all_pairs':qstats(vals),'same_category_pairs':qstats(same) if same else None,'cross_category_pairs':qstats(cross) if cross else None}
    c=bitsets[0];
    for ai,arm in enumerate(arms[1:],start=1): out[arm]['same_event_vs_C']=qstats([jac(c[i],bitsets[ai,i]) for i in range(len(c))])
    return out
def run(output, bitset_output, smoke=False, isolated_source=None, isolated_library=None):
    if not isolated_source or not isolated_library: raise RuntimeError('isolated historical runtime is required')
    if fsha(isolated_source)!=SOURCE_SHA or fsha(isolated_library)!=BINARY_SHA: raise RuntimeError('isolated kernel SHA mismatch')
    from stonkfly.neural import brain as brain_module
    brain_module.SOURCE=Path(isolated_source); brain_module.LIBRARY=Path(isolated_library)
    original=json.loads(ORIG.read_text()); expected={(r['arm'],int(r['event_index'])):r for r in original['records']}
    if len(expected)!=360: raise RuntimeError('original S2k result is not 360 records')
    events=load_events(); selected=events[:2] if smoke else events; c=V51Controller(Settings(neural_ms=500,pulse_ms=200,learning=False)); b=c.brain; kc=np.asarray(b.circuit['kc'],dtype=int); clean=RUN/'brain-clean.npz'
    if fsha(clean)!=CHECKPOINT_SHA: raise RuntimeError('checkpoint SHA mismatch')
    base=b.weight.copy(); kcedge,apl_edge,edge_meta=masks(b); records=[]; rows=[]; gate=[]
    for arm in ARMS:
      for event_index,ev,frame,_ in selected:
        exp=expected[(arm,event_index)]; c.restore(clean); b.weight[:]=overlay(base,kcedge,apl_edge,arm); counts,wall=b.rgb_step(frame,500.0,learning=False); b.counts[:]=counts; active=np.zeros(len(kc),dtype=bool); active[np.flatnonzero(counts[kc]>0)]=True; packed=np.packbits(active,bitorder='little'); h=sha(counts.astype(np.int32)); kh=sha(kc[np.flatnonzero(active)])
        unpack=np.unpackbits(packed,bitorder='little')[:len(kc)].astype(bool); valid=bool(np.array_equal(unpack,active) and int(unpack.sum())==int(active.sum()) and sha(kc[np.flatnonzero(unpack)])==kh)
        match=bool(sha(frame)==exp['frame_sha256'] and int(active.sum())==exp['active_kc_count'] and int(counts[kc].sum())==exp['kc_spike_count'] and h==exp['spike_sha256'] and kh==exp['kc_set_hash'] and valid)
        gate.append({'arm':arm,'event_index':event_index,'match_original':match,'bitset_valid':valid})
        if not match: raise RuntimeError(f'S2k-J fail-closed mismatch arm={arm} event={event_index}')
        records.append({'arm':arm,'event_index':event_index,'frame_sha256':sha(frame),'active_kc_count':int(active.sum()),'kc_spike_count':int(counts[kc].sum()),'spike_sha256':h,'kc_set_hash':kh,'bitset_row':len(rows),'wall_seconds':float(wall)})
        rows.append(packed)
    # Rows are arm-major then event-major and lossless; summary only after every selected record passed.
    arr=np.asarray(rows,dtype=np.uint8); npz_tmp=Path(str(bitset_output)+'.partial.npz'); np.savez_compressed(npz_tmp,active_kc_bitsets=arr,arms=np.asarray(ARMS),event_indices=np.asarray([r[0] for r in events[:len(selected)]],dtype=np.int32),kc_n=np.int32(len(kc)),bitorder=np.asarray('little'),kc_order_hash=np.asarray(sha(kc))) ; os.replace(npz_tmp,bitset_output)
    # For smoke, records are two events per arm and Jaccard is only a gate preview.
    nsel=len(selected); cube=arr.reshape(len(ARMS),nsel,-1); boolcube=np.unpackbits(cube,axis=2,bitorder='little')[:,:,:len(kc)].astype(bool); summary=summarize(boolcube,ARMS,[(x[0],x[3]['category']) for x in selected])
    result={'addendum':'S2k-J','preregistered_doc':'docs/20261006-s2k-j-jaccard-addendum.md','counterfactual':True,'production_eligible':False,'learning':False,'arms':list(ARMS),'events':{'count':60,'selected':nsel,'source_path':str(RUN/'events.jsonl'),'sha256':fsha(RUN/'events.jsonl')},'original_s2k':{'path':str(ORIG),'sha256':fsha(ORIG),'records':len(expected)},'checkpoint':{'path':str(clean),'sha256':fsha(clean)},'kernel':{'source':str(isolated_source),'source_sha256':fsha(isolated_source),'binary':str(isolated_library),'binary_sha256':fsha(isolated_library)},'bitset_artifact':{'path':str(bitset_output),'kc_n':len(kc),'bitorder':'little','kc_order_hash':sha(kc),'encoding':'np.packbits bool active mask; unpack length 4064'},'edge_overlay':edge_meta,'gate':{'all_match':all(x['match_original'] and x['bitset_valid'] for x in gate),'details':gate},'records':records,'jaccard_summary':summary,'limitations':['Descriptive active-set similarity only; no reward/decoder/trade/performance selection.','S2k-J does not alter graph, checkpoint, overlay, or learning settings.']}
    tmp=Path(str(output)+'.partial'); tmp.write_text(json.dumps(result,indent=2,allow_nan=False)); os.replace(tmp,output)
    return result
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--bitsets',type=Path,required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--isolated-source',type=Path,required=True);ap.add_argument('--isolated-library',type=Path,required=True);a=ap.parse_args();r=run(a.output,a.bitsets,a.smoke,a.isolated_source,a.isolated_library);print(json.dumps({'selected':r['events']['selected'],'gate':r['gate']['all_match'],'output':str(a.output),'bitsets':str(a.bitsets)}))
if __name__=='__main__': main()
