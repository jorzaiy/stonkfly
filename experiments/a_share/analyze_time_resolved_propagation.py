"""Targeted, read-only native propagation trace for S2e boundary frames."""
import argparse, hashlib, json, ctypes as C, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from stonkfly.config import Settings
from stonkfly.neural.v51_controller import V51Controller
from stonkfly.neural.common import annotations
from stonkfly.neural.trace_aggregate import TraceBuffers
from experiments.a_share.analyze_kc_overlap import render_frame_from_observation


def sha256_file(path):
    h=hashlib.sha256();
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def aggregate(rows):
    out={}
    for row in rows:
        step=int(row['step']); item=out.setdefault(step, {'step':step,'count':0,'signed_g_increment':0.0,'abs_g_increment':0.0})
        item['count']+=1; item['signed_g_increment']+=float(row['signed_g_increment']); item['abs_g_increment']+=abs(float(row['signed_g_increment']))
    return [out[k] for k in sorted(out)]


def aggregate_arrays(steps, increments):
    if len(steps) == 0: return []
    out=[]
    for step in np.unique(steps):
        values=increments[steps==step]
        out.append({'step':int(step),'event_count':int(len(values)),'signed_g_increment':float(values.sum(dtype=np.float64)),'positive_g_increment':float(values[values>0].sum(dtype=np.float64)),'negative_g_increment':float(values[values<0].sum(dtype=np.float64))})
    return out


def first_unique_crossings(steps, ids, thresholds):
    seen=set(); result={int(t):None for t in thresholds}
    for step, neuron in sorted(zip(steps, ids), key=lambda x:int(x[0])):
        seen.add(int(neuron))
        for threshold in thresholds:
            if result[int(threshold)] is None and len(seen) >= threshold: result[int(threshold)]=int(step)
    return result


def distribution(values):
    present=[float(x) for x in values if x is not None]
    return {'n':len(values),'mean':float(np.mean(present)) if present else None,'median':float(np.median(present)) if present else None,'min':float(min(present)) if present else None,'max':float(max(present)) if present else None,'missing':int(len(values)-len(present))}


def run_event(controller, frame, candidate_mask, cb_mask, kc_mask, checkpoint):
    controller.brain.restore(checkpoint)
    off = controller.observe(frame, reinforcement='none')
    off_counts=controller.brain.counts.copy(); off_v=controller.brain.v.copy(); off_g=controller.brain.g.copy(); off_ref=controller.brain.refractory.copy()
    controller.brain.restore(checkpoint)
    trace=TraceBuffers(controller.brain.n, round(controller.s.neural_ms/controller.brain.dt), candidate_mask, cb_mask, kc_mask)
    trace.reset(controller.brain.cursor); controller.brain.trace_observer=C.addressof(trace.native)
    on=controller.observe(frame, reinforcement='none'); controller.brain.trace_observer=None; trace.fail_if_overflow()
    on_counts=controller.brain.counts.copy()
    equivalent={'spike_sha256_match':off['spike_sha256']==on['spike_sha256'],'all_counts_equal':bool(np.array_equal(off_counts,on_counts)),'kc_counts_equal':bool(np.array_equal(off_counts[controller.brain.circuit['kc']],on_counts[controller.brain.circuit['kc']])), 'v_equal':bool(np.array_equal(off_v,controller.brain.v)), 'g_equal':bool(np.array_equal(off_g,controller.brain.g)), 'refractory_equal':bool(np.array_equal(off_ref,controller.brain.refractory))}
    if not all(equivalent.values()): raise RuntimeError(f'trace equivalence failed: {equivalent}')
    return trace, equivalent, on


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('run_path',type=Path); ap.add_argument('--result',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--event-indices',type=int,nargs='+'); ap.add_argument('--all-s2d-events',action='store_true'); args=ap.parse_args()
    source=json.loads(args.result.read_text()); main_result=args.result.parent/'projection_threshold_analysis.json'
    if args.all_s2d_events:
        main=json.loads(main_result.read_text()); indices=[int(r['event_index']) for r in main['per_frame']]; expected=set(main.get('parameters',{}).get('event_indices',indices));
        if len(indices)!=60 or set(indices)!=expected: raise RuntimeError('S2d event set mismatch; refusing to run')
    elif args.event_indices: indices=[int(x) for x in args.event_indices]
    else: raise RuntimeError('choose --event-indices or --all-s2d-events')
    events=[json.loads(x) for x in (args.run_path/'events.jsonl').read_text().splitlines()]
    settings=Settings(neural_ms=500,pulse_ms=200,learning=False); controller=V51Controller(settings); brain=controller.brain; ann=annotations(brain.ids); types=ann.type.fillna('').astype(str).to_numpy(); superclass=np.asarray(brain.superclass).astype(str)
    projection=np.flatnonzero(superclass=='visual_projection'); cb=np.flatnonzero(superclass=='cb_intrinsic'); kc=np.asarray(brain.circuit['kc'],dtype=np.int64); candidate_bodies={int(x) for x in source['candidate_sets']['perfect_presence_candidates'] for x in [x['body_id']]}; candidate_slot=np.full(brain.n,-1,np.int32); [candidate_slot.__setitem__(int(np.flatnonzero(brain.ids==body)[0]),slot) for slot,body in enumerate(sorted(candidate_bodies))]; cb_mask=np.zeros(brain.n,dtype=np.uint8); cb_mask[cb]=1; kc_slot=np.full(brain.n,-1,np.int32); [kc_slot.__setitem__(int(idx),slot) for slot,idx in enumerate(kc)]; kc_mask=np.zeros(brain.n,dtype=np.uint8); kc_mask[kc]=1
    with __import__('tempfile').TemporaryDirectory(prefix='s2e-') as td:
        checkpoint=Path(td)/'clean.npz'; controller.save(checkpoint); records=[]
        for event_index in indices:
            event=events[event_index]; frame=render_frame_from_observation(event['observation']); actual=hashlib.sha256(frame.tobytes()).hexdigest();
            if actual!=event['neural']['input_sha256']: raise RuntimeError(f'frame hash mismatch {event_index}')
            trace,equivalent,on=run_event(controller,frame,candidate_slot,cb_mask,kc_slot,checkpoint)
            cb_index_set=set(map(int,cb)); kc_index_set=set(map(int,kc)); cand_rows=[]
            for body in sorted(candidate_bodies):
                idx=int(np.flatnonzero(brain.ids==body)[0]); slot=int(candidate_slot[idx]); csteps=np.flatnonzero(trace.candidate_spike_counts[slot]>0) if slot>=0 else np.array([],dtype=int); mask=trace.candidate_event_counts[slot]>0
                rows=[{'step':int(s),'event_count':int(trace.candidate_event_counts[slot,s]),'signed_g_increment':float(trace.candidate_signed[slot,s]),'positive_g_increment':float(trace.candidate_positive[slot,s]),'negative_g_increment':float(trace.candidate_negative[slot,s])} for s in np.flatnonzero(mask)]
                cand_rows.append({'body_id':body,'type':str(types[idx]),'spike_steps':[int(x) for x in csteps],'spike_summary':{'total':int(trace.candidate_spike_counts[slot].sum()),'first_step':int(csteps[0]) if len(csteps) else None,'last_step':int(csteps[-1]) if len(csteps) else None},'candidate_to_cb_by_step':rows})
            step_counts=[{'step':i,'candidate_spikes':int(x[0]),'cb_spikes':int(x[1]),'kc_spikes':int(x[2])} for i,x in enumerate(trace.step_spike_counts) if int(x.sum())]
            candidate_spike_steps=np.flatnonzero(trace.candidate_spike_counts.sum(axis=0)>0); cbkc_steps=np.flatnonzero(trace.cbkc_event_counts>0); kc_steps=np.flatnonzero(trace.kc_first_spike_step>=0); cb_steps=np.flatnonzero(trace.step_spike_counts[:,1]>0)
            cbkc_rows=[{'step':int(s),'event_count':int(trace.cbkc_event_counts[s]),'signed_g_increment':float(trace.cbkc_signed[s]),'positive_g_increment':float(trace.cbkc_positive[s]),'negative_g_increment':float(trace.cbkc_negative[s])} for s in cbkc_steps]
            records.append({'event_index':int(event_index),'date':str(event['date']),'frame_sha256':actual,'kc_active_count':int(np.count_nonzero(brain.counts[kc]>0)),'group': 'low' if int(np.count_nonzero(brain.counts[kc]>0))<=10 else 'high' if int(np.count_nonzero(brain.counts[kc]>0))>=1000 else 'mid','equivalence':equivalent,'step_spike_counts':step_counts,'candidates':cand_rows,'cb_to_kc_by_step':cbkc_rows,'layer_spike_summary':{'cb':{'total':int(trace.step_spike_counts[:,1].sum()),'first_step':int(cb_steps[0]) if len(cb_steps) else None,'last_step':int(cb_steps[-1]) if len(cb_steps) else None},'kc':{'total':int(trace.step_spike_counts[:,2].sum()),'first_step':int(kc_steps[0]) if len(kc_steps) else None,'last_step':int(kc_steps[-1]) if len(kc_steps) else None}},'kc_unique_crossings':{str(t):int(np.sort(trace.kc_first_spike_step[trace.kc_first_spike_step>=0])[t-1]) if np.count_nonzero(trace.kc_first_spike_step>=0)>=t else None for t in (1,10,100,1000)},'first_candidate_spike_step':int(candidate_spike_steps[0]) if len(candidate_spike_steps) else None,'first_candidate_to_cb_delivery_step':int(next((r['step'] for c in cand_rows for r in c['candidate_to_cb_by_step']),-1)) if any(c['candidate_to_cb_by_step'] for c in cand_rows) else None,'first_cb_to_kc_delivery_step':int(cbkc_steps[0]) if len(cbkc_steps) else None,'buffer_usage':{'aggregate_shapes':{'candidate_x_step':[int(trace.candidate_n,int(trace.steps))] if False else [int(trace.candidate_n),int(trace.steps)],'cbkc_step':[int(trace.steps)]},'overflow':bool(trace.native.overflow)}})
    first={}
    if len(records)==2:
        a,b=records
        for key in ('candidate_spikes','cb_spikes','kc_spikes'):
            aa={x['step']:x[key] for x in a['step_spike_counts']}; bb={x['step']:x[key] for x in b['step_spike_counts']}; first[key]=next((s for s in sorted(set(aa)|set(bb)) if aa.get(s,0)!=bb.get(s,0)),None)
    main_result=args.result.parent/'projection_threshold_analysis.json'
    if not main_result.exists(): raise FileNotFoundError(f'main S2d result missing: {main_result}')
    for r in records:
        candidate_delivery_steps=[x['step'] for c in r['candidates'] for x in c['candidate_to_cb_by_step']]
        r['first_candidate_to_cb_delivery_step']=min(candidate_delivery_steps) if candidate_delivery_steps else None
        r['kc_unique_crossings']={int(k):v for k,v in r['kc_unique_crossings'].items()}
        r['buffer_usage']['delivery_used']=int(sum(x['event_count'] for c in r['candidates'] for x in c['candidate_to_cb_by_step'])+sum(x['event_count'] for x in r['cb_to_kc_by_step']))
        r['buffer_usage']['spike_used']=int(sum(x['candidate_spikes']+x['cb_spikes']+x['kc_spikes'] for x in r['step_spike_counts']))
    group_summary={}
    for group in ('low','high'):
        subset=[r for r in records if r['group']==group]
        group_summary[group]={'n':len(subset),'kc_unique_crossings':{str(t):distribution([r['kc_unique_crossings'][t] for r in subset]) for t in (1,10,100,1000)},'first_candidate_spike_step':distribution([r['first_candidate_spike_step'] for r in subset]),'first_candidate_to_cb_delivery_step':distribution([r['first_candidate_to_cb_delivery_step'] for r in subset]),'first_cb_to_kc_delivery_step':distribution([r['first_cb_to_kc_delivery_step'] for r in subset]),'candidate_before_kc_crossing':{str(t):sum(r['first_candidate_spike_step'] is not None and r['kc_unique_crossings'][t] is not None and r['first_candidate_spike_step']<r['kc_unique_crossings'][t] for r in subset) for t in (10,100,1000)},'candidate_after_kc_crossing':{str(t):sum(r['first_candidate_spike_step'] is not None and r['kc_unique_crossings'][t] is not None and r['first_candidate_spike_step']>r['kc_unique_crossings'][t] for r in subset) for t in (10,100,1000)},'candidate_spike_total':sum(x['layer_spike_summary']['cb']['total']*0+x['step_spike_counts'][0]['candidate_spikes'] if x['step_spike_counts'] else 0 for x in subset),'cb_total_spikes':sum(r['layer_spike_summary']['cb']['total'] for r in subset),'kc_total_spikes':sum(r['layer_spike_summary']['kc']['total'] for r in subset),'max_delivery_used':max((r['buffer_usage']['delivery_used'] for r in subset),default=0),'max_spike_used':max((r['buffer_usage']['spike_used'] for r in subset),default=0)}
    for group in group_summary:
        group_summary[group]['candidate_spike_total']=sum(sum(s['candidate_spikes'] for s in r['step_spike_counts']) for r in records if r['group']==group)
    if args.all_s2d_events: first=None
    out={'provenance':{'candidate_review_sha256':sha256_file(args.result),'main_result_sha256':sha256_file(main_result) if main_result.exists() else None,'graph_path':str(Path('data/graph.npz').resolve()),'graph_sha256':sha256_file(Path('data/graph.npz')),'script_sha256':sha256_file(Path(__file__))},'settings':{'neural_ms':500,'dt_ms':0.1,'delay_ms':1.8,'refractory_ms':2.2,'update_order':'evolve active -> emit/enqueue -> deliver due queue g+=weight; observer writes after g update','trace_default_off':True,'weight_semantics':'signed synapse_count*0.275 kernel g increment'},'events':records,'group_summary':group_summary,'earliest_low_high_divergence_step':first,'limitations':['Trace is observational dynamic decomposition, not counterfactual causality.','Only candidate->CB and CB->KC deliveries are retained; all other graph edges still execute.','Output is targeted and compact; non-target deliveries are not recorded.','Full all-S2d mode does not compare first divergence across different frames.']}
    out['file_size_bytes']=0
    for _ in range(4):
        args.output.write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        size=int(args.output.stat().st_size)
        if out['file_size_bytes']==size: break
        out['file_size_bytes']=size
    args.output.write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'output':str(args.output),'events':len(records),'file_size_bytes':int(args.output.stat().st_size)}))

if __name__=='__main__': main()
