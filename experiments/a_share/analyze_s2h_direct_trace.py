import argparse,ctypes as C,hashlib,json
from pathlib import Path
import numpy as np
from stonkfly.config import Settings
from stonkfly.neural.v51_controller import V51Controller
from stonkfly.neural.common import annotations
from stonkfly.neural.trace_s2h import TraceS2H
from experiments.a_share.analyze_kc_overlap import render_frame_from_observation
SRC=(10530,11930,14437,19845);TARGET=13334
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('run_path',type=Path);ap.add_argument('--review',type=Path,required=True);ap.add_argument('--s2d-main',type=Path,required=True);ap.add_argument('--event-indices',type=int,nargs='+');ap.add_argument('--all-s2d-events',action='store_true');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();d=json.loads(a.s2d_main.read_text());labels={int(x['event_index']):x['group'] for x in d['per_frame']};idx=[int(x) for x in d['parameters']['event_indices']] if a.all_s2d_events else a.event_indices
 if a.all_s2d_events and (len(idx)!=60 or set(idx)!=set(labels)):raise RuntimeError('exact S2d event set required')
 ev=[json.loads(x) for x in (a.run_path/'events.jsonl').read_text().splitlines()]; c=V51Controller(Settings(neural_ms=500,pulse_ms=200,learning=False));b=c.brain;z=np.load('data/graph.npz');ids=z['ids'];ptr=z['ptr'];post=z['post'];wt=z['weight'];sup=z['superclass'];ann=annotations(ids);typ=ann.type.fillna('').astype(str).to_numpy();idn={int(x):i for i,x in enumerate(ids)};ss=np.full(b.n,-1,np.int32)
 for k,x in enumerate(SRC):ss[idn[x]]=k
 stable=json.loads(a.review.read_text())['stable_cb_sources'];stable_ids=[int(x['body_id']) for x in stable];st=np.full(b.n,-1,np.int32)
 for k,x in enumerate(stable_ids):st[idn[x]]=k
 kc=np.asarray(b.circuit['kc']);ks=np.full(b.n,-1,np.int32)
 for k,x in enumerate(kc):ks[int(x)]=k
 ck=Path('/tmp/s2h_clean.npz');c.save(ck);out=[]
 for ix in idx:
  frame=render_frame_from_observation(ev[ix]['observation']);c.brain.restore(ck);c.brain.trace_observer_s2h=None;o=c.observe(frame,'none');counts=b.counts.copy();v=b.v.copy();g=b.g.copy();r=b.refractory.copy();c.brain.restore(ck);t=TraceS2H(ss,st,ks,idn[TARGET]);t.reset();c.brain.trace_observer_s2h=C.addressof(t.native);on=c.observe(frame,'none');c.brain.trace_observer_s2h=None;t.fail();eq={'spike_sha256':o['spike_sha256']==on['spike_sha256'],'counts':bool(np.array_equal(counts,b.counts)),'kc_counts':bool(np.array_equal(counts[kc],b.counts[kc])),'v':bool(np.array_equal(v,b.v)),'g':bool(np.array_equal(g,b.g)),'refractory':bool(np.array_equal(r,b.refractory))}
  if not all(eq.values()):raise RuntimeError(eq)
  targets=[{'body_id':x,'type':str(typ[idn[x]]),'first_spike_step':int(t.stable_first[k]) if t.stable_first[k]>=0 else None} for k,x in enumerate(stable_ids)];rows=[]
  for q in range(int(t.native.out_used)):rows.append({'step':int(t.out_step[q]),'target_body_id':int(ids[t.out_target[q]]),'target_type':str(typ[t.out_target[q]]),'target_superclass':str(sup[t.out_target[q]]),'signed_weight':float(t.out_weight[q]),'source_target': '13334_outgoing'})
  srcrows=[{'source_body_id':x,'source_type':str(typ[idn[x]]),'first_spike_step':int(t.source_first[k]) if t.source_first[k]>=0 else None,'spike_count':int(t.source_spikes[k]),'delivery_events_to_13334':int(t.source_events[k]),'delivery_signed':float(t.source_signed[k]),'delivery_positive':float(t.source_positive[k]),'delivery_negative':float(t.source_negative[k]),'first_delivery_step':int(t.source_first_delivery[k]) if t.source_first_delivery[k]>=0 else None,'last_delivery_step':int(t.source_last_delivery[k]) if t.source_last_delivery[k]>=0 else None,'pre_target_count':int(t.source_pre_count[k]),'pre_target_first_step':int(t.source_pre_first[k]) if t.source_pre_first[k]>=0 else None,'pre_target_last_step':int(t.source_pre_last[k]) if t.source_pre_last[k]>=0 else None} for k,x in enumerate(SRC)]
  kcseq=[{'body_id':int(ids[t.kc_order_id[k]]),'step':int(t.kc_order_step[k])} for k in range(int(t.native.kc_order_used))]
  out.append({'event_index':ix,'group':labels.get(ix),'sources':srcrows,'target_13334_first_spike_step':int(t.target_first[0]) if t.target_first[0]>=0 else None,'stable_cb_first_spikes':targets,'kc_unique_spike_sequence':kcseq,'kc1':kcseq[0] if kcseq else None,'kc10':kcseq[9] if len(kcseq)>=10 else None,'target_outgoing_deliveries':rows,'equivalence':eq,'overflow':bool(t.native.overflow),'out_used':int(t.native.out_used)})
 graph=Path('data/graph.npz');result={'data_source':'direct_kernel_trace','provenance':{'s2d_path':str(a.s2d_main.resolve()),'s2d_sha256':sha(a.s2d_main),'s2f_review_path':str(a.review.resolve()),'s2f_review_sha256':sha(a.review),'graph_path':str(graph.resolve()),'graph_sha256':sha(graph),'kernel_path':str(Path('stonkfly/neural/kernel.cpp').resolve()),'kernel_sha256':sha(Path('stonkfly/neural/kernel.cpp')),'script_path':str(Path(__file__).resolve()),'script_sha256':sha(Path(__file__))},'events':out,'limitations':['All events are direct kernel observations after g[j]+=weight[e]; no causal or counterfactual claim.']};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False));result['file_size_bytes']=a.output.stat().st_size;a.output.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({'events':len(out),'file_size_bytes':a.output.stat().st_size}))
if __name__=='__main__':main()
