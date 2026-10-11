import argparse, ctypes as C, hashlib, json
from pathlib import Path
import numpy as np
from stonkfly.config import Settings
from stonkfly.neural.v51_controller import V51Controller
from stonkfly.neural.common import annotations
from stonkfly.neural.trace_s2g import TraceS2G
from experiments.a_share.analyze_kc_overlap import render_frame_from_observation

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('run_path',type=Path);ap.add_argument('--review',type=Path,required=True);ap.add_argument('--s2d-main',type=Path);ap.add_argument('--event-indices',type=int,nargs='+');ap.add_argument('--all-s2d-events',action='store_true');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.all_s2d_events:
  if not a.s2d_main: raise RuntimeError('--s2d-main required with --all-s2d-events')
  s2d=json.loads(a.s2d_main.read_text()); a.event_indices=[int(x) for x in s2d['parameters']['event_indices']]; labels={int(x['event_index']):x['group'] for x in s2d['per_frame']}
  if len(a.event_indices)!=60 or set(a.event_indices)!=set(labels): raise RuntimeError('S2d event set is not exactly 60')
 else:
  if not a.event_indices: raise RuntimeError('--event-indices or --all-s2d-events required')
  labels={int(x):None for x in a.event_indices}
 review=json.loads(a.review.read_text()); stable=review.get('stable_cb_sources',[])
 if len(stable)!=14: raise RuntimeError(f'expected exactly 14 stable CB sources, got {len(stable)}')
 stable_ids=[int(x['body_id']) for x in stable]; ev=[json.loads(x) for x in (a.run_path/'events.jsonl').read_text().splitlines()]
 c=V51Controller(Settings(neural_ms=500,pulse_ms=200,learning=False)); b=c.brain; z=np.load('data/graph.npz',allow_pickle=False); ids=z['ids'];ptr=z['ptr'];post=z['post'];weight=z['weight'];sup=z['superclass']; idnode={int(x):i for i,x in enumerate(ids)}
 targets=[idnode[x] for x in stable_ids if x in idnode]
 if len(targets)!=14: raise RuntimeError('stable CB IDs missing from graph')
 target_slot=np.full(b.n,-1,np.int32)
 for k,i in enumerate(targets): target_slot[i]=k
 source_nodes=set()
 for i in range(b.n):
  if np.intersect1d(post[ptr[i]:ptr[i+1]],targets).size: source_nodes.add(i)
 source_slot=np.full(b.n,-1,np.int32)
 for k,i in enumerate(sorted(source_nodes)): source_slot[i]=k
 ann=annotations(ids); typ=ann.type.fillna('').astype(str).to_numpy(); ck=Path('/tmp/s2g_clean.npz'); c.save(ck); out=[]
 with np.errstate(all='raise'):
  for ix in a.event_indices:
   frame=render_frame_from_observation(ev[ix]['observation']); c.brain.restore(ck);c.brain.trace_observer=None;c.brain.trace_observer_s2f=None;c.brain.trace_observer_s2g=None; off=c.observe(frame,'none'); counts= b.counts.copy();v=b.v.copy();g=b.g.copy();r=b.refractory.copy()
   c.brain.restore(ck); t=TraceS2G(source_slot,target_slot); t.reset(); c.brain.trace_observer_s2g=C.addressof(t.native); on=c.observe(frame,'none'); c.brain.trace_observer_s2g=None;t.fail()
   eq={'spike_sha256':off['spike_sha256']==on['spike_sha256'],'counts':bool(np.array_equal(counts,b.counts)),'kc_counts':bool(np.array_equal(counts[np.asarray(b.circuit['kc'])],b.counts[np.asarray(b.circuit['kc'])])),'v':bool(np.array_equal(v,b.v)),'g':bool(np.array_equal(g,b.g)),'refractory':bool(np.array_equal(r,b.refractory))}
   if not all(eq.values()): raise RuntimeError(eq)
   if np.any((t.target_first < -1)|(t.target_first>=5000)): raise RuntimeError('target first step out of range')
   rows=[]
   for s in range(t.source_n):
    for q in range(t.target_n):
     if t.pair_events[s,q]<=0: continue
     ni=sorted(source_nodes)[s]; ti=targets[q]; lead=int(t.target_first[q]-t.pair_last[s,q]) if t.target_first[q]>=0 else None
     rows.append({'source_body_id':int(ids[ni]),'source_type':str(typ[ni]),'source_superclass':str(sup[ni]),'target_body_id':int(ids[ti]),'target_type':str(typ[ti]),'target_slot':q,'events':int(t.pair_events[s,q]),'signed':float(t.pair_signed[s,q]),'positive':float(t.pair_positive[s,q]),'negative':float(t.pair_negative[s,q]),'first_step':int(t.pair_first[s,q]),'last_step':int(t.pair_last[s,q]),'lead_steps_to_target_spike':lead,'actual_delivery_source_to_target':True,'structural_not_causal':True})
   nearest=[]
   for q in range(14):
    rr=[x for x in rows if x['target_slot']==q and x['lead_steps_to_target_spike'] is not None]
    nearest.extend(sorted(rr,key=lambda x:x['lead_steps_to_target_spike'])[:5])
   by_super={}
   for x in rows:
    z=by_super.setdefault(x['source_superclass'],{'events':0,'signed':0.0,'positive':0.0,'negative':0.0}); z['events']+=x['events'];z['signed']+=x['signed'];z['positive']+=x['positive'];z['negative']+=x['negative']
   out.append({'event_index':int(ix),'group':labels.get(int(ix)),'stable_targets':[{'body_id':stable_ids[q],'first_spike_step':int(t.target_first[q]) if t.target_first[q]>=0 else None} for q in range(14)],'nearest_inputs_to_target_spike':nearest,'source_superclass_summary':by_super,'pre_spike_pair_rows':rows,'source_slot_total':t.source_n,'pair_slot_total':t.source_n*14,'nonzero_pair_count':len(rows),'equivalence':eq,'overflow':bool(t.native.overflow)})
 groups={g:[e for e in out if e.get('group')==g] for g in ('low','high')}; pair_stats={}; source_stats={}; lead_stats={}
 for g,es in groups.items():
  for e in es:
   seen=set()
   for r in e['pre_spike_pair_rows']:
    key=(r['source_body_id'],r['target_body_id']); seen.add(key); z=pair_stats.setdefault(key,{'source_body_id':key[0],'target_body_id':key[1],'source_type':r['source_type'],'source_superclass':r['source_superclass'],'target_type':r['target_type'],'low_frames':0,'high_frames':0,'low_events':0,'high_events':0,'signed':0.0,'positive':0.0,'negative':0.0,'first_steps':[]}); z[f'{g}_events']+=r['events']; z[f'{g}_frames']+=1;z['signed']+=r['signed'];z['positive']+=r['positive'];z['negative']+=r['negative'];z['first_steps'].append(r['first_step'])
    for key in {r['source_body_id'] for r in e['pre_spike_pair_rows']}:
     r=next(r for r in e['pre_spike_pair_rows'] if r['source_body_id']==key); z=source_stats.setdefault(key,{'source_body_id':key,'source_type':r['source_type'],'source_superclass':r['source_superclass'],'low_frames':0,'high_frames':0}); z[f'{g}_frames']+=1
   for q,tg in enumerate(e['stable_targets']):
    leads=[r['lead_steps_to_target_spike'] for r in e['pre_spike_pair_rows'] if r['target_slot']==q and r['lead_steps_to_target_spike'] is not None]
    lead_stats.setdefault(g,[]).append({'event_index':e['event_index'],'target_body_id':tg['body_id'],'lead_le_1':bool(leads and min(leads)<=1),'lead_le_2':bool(leads and min(leads)<=2),'lead_le_5':bool(leads and min(leads)<=5),'lead_le_10':bool(leads and min(leads)<=10),'min_lead':min(leads) if leads else None})
 for z in pair_stats.values():
  z['low_rate']=z['low_frames']/43;z['high_rate']=z['high_frames']/17;z['rate_difference']=z['high_rate']-z['low_rate'];del z['first_steps']
 for z in source_stats.values(): z['low_rate']=z['low_frames']/43;z['high_rate']=z['high_frames']/17;z['rate_difference']=z['high_rate']-z['low_rate']
 perfect=[z for z in pair_stats.values() if z['high_frames']==17 and z['low_frames']==0]
 lead_summary={g:{k:sum(bool(x[k]) for x in xs)/max(1,len(xs)) for k in ('lead_le_1','lead_le_2','lead_le_5','lead_le_10')} for g,xs in lead_stats.items()}
 graph=Path('data/graph.npz'); result={'provenance':{'s2f_review_path':str(a.review.resolve()),'s2f_review_sha256':sha(a.review),'s2d_main_path':str(a.s2d_main.resolve()) if a.s2d_main else None,'s2d_main_sha256':sha(a.s2d_main) if a.s2d_main else None,'graph_path':str(graph.resolve()),'graph_sha256':sha(graph),'kernel_path':str(Path('stonkfly/neural/kernel.cpp').resolve()),'kernel_sha256':sha(Path('stonkfly/neural/kernel.cpp')),'script_path':str(Path(__file__).resolve()),'script_sha256':sha(Path(__file__))},'stable_cb_count':14,'events':out,'group_summary':{'n_low':len(groups['low']),'n_high':len(groups['high']),'pair_support':list(pair_stats.values()),'source_support':list(source_stats.values()),'perfect_high_only':perfect,'lead_window_rates':lead_summary,'lead_window_definition':'per target and event, minimum actual pre-spike delivery lead; high denominator 17*14, low denominator 43*14'},'limitations':['Pre-spike window is full 5000 steps when a target never spikes; low/high quantities then have unequal observed windows.','Delivery attribution is observational dynamic decomposition, not counterfactual causality.','Rows are actual retained source->stable-CB deliveries after g[j]+=weight; support rates and perfect-high-only are sample-only.']};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False));result['file_size_bytes']=a.output.stat().st_size;a.output.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({'source_n':int(sorted(source_nodes).__len__()),'events':len(out),'perfect_high_only':len(perfect),'file_size_bytes':a.output.stat().st_size}))
if __name__=='__main__':main()
