import argparse,json,hashlib,ctypes as C
from pathlib import Path
import numpy as np
from stonkfly.config import Settings
from stonkfly.neural.v51_controller import V51Controller
from stonkfly.neural.common import annotations
from stonkfly.neural.trace_s2f import TraceS2Fv2
from experiments.a_share.analyze_kc_overlap import render_frame_from_observation
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def summarize_sources(events, group_key, source_key):
 stats={}
 for e in events:
  g=group_key.get(int(e['event_index']))
  if g not in ('low','high'): continue
  for row in e[source_key]['rows']:
   if source_key=='projection_sources': active=bool(row['phase_events'][0])
   else: active=bool(row['phase_events'][0])
   s=stats.setdefault(int(row['body_id']),{'body_id':int(row['body_id']),'type':row['type'],'low':{'frames':0,'events':0,'signed':0.0,'positive':0.0,'negative':0.0,'first_steps':[]},'high':{'frames':0,'events':0,'signed':0.0,'positive':0.0,'negative':0.0,'first_steps':[]}})
   if active:
    z=s[g]; z['frames']+=1; z['events']+=int(row['phase_events'][0]); z['signed']+=float(row['phase_signed'][0]); z['positive']+=float(row['phase_positive'][0]); z['negative']+=float(row['phase_negative'][0])
    if row.get('first_delivery_step') is not None: z['first_steps'].append(int(row['first_delivery_step']))
 out=[]
 for s in stats.values():
  for g in ('low','high'):
   z=s[g]; z['rate']=z['frames']/max(1,sum(1 for x in group_key.values() if x==g)); z['first_delivery_step_summary']={'min':min(z['first_steps']) if z['first_steps'] else None,'max':max(z['first_steps']) if z['first_steps'] else None}
   del z['first_steps']
  s['frame_difference']=s['high']['frames']-s['low']['frames']; s['sample_only']=True; out.append(s)
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('run_path',type=Path);ap.add_argument('--result',type=Path,required=True);ap.add_argument('--candidate-review',type=Path);ap.add_argument('--event-indices',type=int,nargs='+');ap.add_argument('--all-s2d-events',action='store_true');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args(); src=json.loads(a.result.read_text()); main_result=(a.result if 'parameters' in src else a.result.parent/'projection_threshold_analysis.json'); review_path=(a.candidate_review or (a.result if 'parameters' not in src else a.result.parent/'candidate_connectivity_review.json')); src=json.loads(main_result.read_text()); ev=[json.loads(x) for x in (a.run_path/'events.jsonl').read_text().splitlines()]; pf=src.get('per_frame',[]); s2d_indices=[int(x) for x in src.get('parameters',{}).get('event_indices',[])]; group_key={int(x['event_index']):x.get('group') for x in pf};
 if a.all_s2d_events:
  if len(s2d_indices)!=60 or set(s2d_indices)!=set(group_key): raise RuntimeError('S2d event set is not exactly 60 per_frame events')
  a.event_indices=s2d_indices
 elif not a.event_indices: raise RuntimeError('--event-indices or --all-s2d-events required')
 c=V51Controller(Settings(neural_ms=500,pulse_ms=200,learning=False));b=c.brain; ann=annotations(b.ids); typ=ann.type.fillna('').astype(str).to_numpy(); sup=np.asarray(b.superclass).astype(str); proj=np.flatnonzero(sup=='visual_projection'); cb=np.flatnonzero(sup=='cb_intrinsic'); kc=np.asarray(b.circuit['kc']); ps=np.full(b.n,-1,np.int32); [ps.__setitem__(int(i),j) for j,i in enumerate(proj)]; cs=np.full(b.n,-1,np.int32); [cs.__setitem__(int(i),j) for j,i in enumerate(cb)]; ks=np.full(b.n,-1,np.int32); [ks.__setitem__(int(i),j) for j,i in enumerate(kc)]; cbmask=np.zeros(b.n,np.uint8);cbmask[cb]=1; out=[]
 import tempfile
 with tempfile.TemporaryDirectory() as td:
  ck=Path(td)/'c.npz';c.save(ck)
  for ix in a.event_indices:
   frame=render_frame_from_observation(ev[ix]['observation']); c.brain.restore(ck); c.brain.trace_observer=None;c.brain.trace_observer_s2f=None; o=c.observe(frame,'none'); oc=c.brain.counts.copy(); ov=b.v.copy();og=b.g.copy();orr=b.refractory.copy(); c.brain.restore(ck); t=TraceS2Fv2(b.n,ps,cs,ks);t.reset();c.brain.trace_observer_s2f=C.addressof(t.native); on=c.observe(frame,'none');c.brain.trace_observer_s2f=None;t.fail(); eq={'spike_sha256':o['spike_sha256']==on['spike_sha256'],'counts':bool(np.array_equal(oc,b.counts)),'kc_counts':bool(np.array_equal(oc[kc],b.counts[kc])),'v':bool(np.array_equal(ov,b.v)),'g':bool(np.array_equal(og,b.g)),'refractory':bool(np.array_equal(orr,b.refractory))};
   if not all(eq.values()): raise RuntimeError(eq)
   if t.kc_first.dtype != np.int32 or t.kc_first.itemsize != 4:
    raise RuntimeError(f'kc_first dtype/itemsize mismatch: {t.kc_first.dtype}/{t.kc_first.itemsize}')
   bad=t.kc_first[(t.kc_first != -1) & ((t.kc_first < 0) | (t.kc_first >= 5000))]
   if bad.size: raise RuntimeError(f'kc_first step out of range: {bad[:8].tolist()}')
   kc_steps=np.sort(t.kc_first[t.kc_first >= 0])
   kc_crossings={str(k): (int(kc_steps[k-1]) if kc_steps.size >= k else None) for k in (1,10,100,1000)}
   rows=[]
   for j,i in enumerate(proj):
    if t.projection_spikes[j] or np.any(t.projection_events[j]): rows.append({'body_id':int(b.ids[i]),'type':str(typ[i]),'spikes':int(t.projection_spikes[j]),'first_step':int(t.projection_first[j]) if t.projection_first[j]>=0 else None,'last_step':int(t.projection_last[j]) if t.projection_last[j]>=0 else None,'first_delivery_step':int(t.projection_first_delivery[j]) if t.projection_first_delivery[j]>=0 else None,'phase_events':[int(x) for x in t.projection_events[j]],'phase_signed':[float(x) for x in t.projection_signed[j]],'phase_positive':[float(x) for x in t.projection_positive[j]],'phase_negative':[float(x) for x in t.projection_negative[j]]})
   cb_rows=[]
   for j,i in enumerate(cb):
    if np.any(t.cb_events[j] > 0):
     cb_rows.append({'body_id':int(b.ids[i]),'type':str(typ[i]),'first_delivery_step':int(t.cb_first[j]) if t.cb_first[j]>=0 else None,'phase_events':[int(x) for x in t.cb_events[j]],'phase_signed':[float(x) for x in t.cb_signed[j]],'phase_positive':[float(x) for x in t.cb_positive[j]],'phase_negative':[float(x) for x in t.cb_negative[j]]})
   out.append({'event_index':int(ix),'group':group_key.get(int(ix)),'equivalence':eq,'kc_crossings':kc_crossings,'projection_sources':{'slot_total':int(len(proj)),'nonzero_count':int(len(rows)),'rows':rows},'cb_sources':{'slot_total':int(len(cb)),'nonzero_count':int(len(cb_rows)),'rows':cb_rows},'overflow':bool(t.native.overflow)})
 group_summary={'n_low':sum(1 for x in group_key.values() if x=='low'),'n_high':sum(1 for x in group_key.values() if x=='high'),'projection_sources':summarize_sources(out,group_key,'projection_sources'),'cb_sources':summarize_sources(out,group_key,'cb_sources'),'boundary_sources':{str(body):next((x for x in summarize_sources(out,group_key,'projection_sources') if x['body_id']==body),None) for body in (10164,11670,11953,13285)},'limitations':['Occurrence, enrichment, rankings, and thresholds are sample-only descriptive statistics.','Source/CB overlap is observational and structural, not a causal attribution.']}
 psumm=group_summary['projection_sources']; cbsumm=group_summary['cb_sources']; group_summary['high_only']=[x for x in psumm if x['high']['frames']>0 and x['low']['frames']==0]; group_summary['high_enriched_top']=sorted(psumm,key=lambda x:(x['frame_difference'],x['high']['events']),reverse=True)[:25]; group_summary['cb_high_enriched_top']=sorted(cbsumm,key=lambda x:(x['frame_difference'],x['high']['events']),reverse=True)[:25]; group_summary['type_summary']={}
 for x in psumm:
  z=group_summary['type_summary'].setdefault(x['type'],{'projection_sources':0,'low_frames':0,'high_frames':0}); z['projection_sources']+=1; z['low_frames']+=x['low']['frames']; z['high_frames']+=x['high']['frames']
 s2e_result=main_result.parent.parent.parent/'S2e/20261004T-time-resolved-v1/time_resolved_propagation.json'; graph=Path('data/graph.npz'); result={'provenance':{'candidate_review_path':str(review_path.resolve()),'candidate_review_sha256':sha(review_path),'s2d_main_path':str(main_result.resolve()),'s2d_main_sha256':sha(main_result),'s2e_path':str(s2e_result.resolve()),'s2e_sha256':sha(s2e_result) if s2e_result.exists() else None,'graph_path':str(graph.resolve()),'graph_sha256':sha(graph),'kernel_path':str(Path('stonkfly/neural/kernel.cpp').resolve()),'kernel_sha256':sha(Path('stonkfly/neural/kernel.cpp')),'script_path':str(Path(__file__).resolve()),'script_sha256':sha(Path(__file__))},'phase_definition':['<10','10-99','100-999','>=1000'],'events':out,'group_summary':group_summary,'limitations':['Phase aggregates are observational dynamic decomposition, not causality.','KC output is crossing summary only; raw first-step arrays are intentionally omitted.','All group occurrence/ranking thresholds are sample-only descriptive results.']};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)); result['file_size_bytes']=a.output.stat().st_size; a.output.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({'events':len(out),'projections':len(proj),'file_size_bytes':a.output.stat().st_size}))
if __name__=='__main__':main()
