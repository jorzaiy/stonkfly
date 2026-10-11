import argparse,ctypes as C,hashlib,json
from pathlib import Path
import numpy as np
from stonkfly.config import Settings
from stonkfly.neural.v51_controller import V51Controller
from stonkfly.neural.common import annotations
from stonkfly.neural.trace_s2i import TraceS2I
from experiments.a_share.analyze_kc_overlap import render_frame_from_observation
LEADER=10530
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('run_path',type=Path);ap.add_argument('--s2d-main',type=Path,required=True);ap.add_argument('--event-indices',type=int,nargs='+');ap.add_argument('--all-s2d-events',action='store_true');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();d=json.loads(a.s2d_main.read_text());labels={int(x['event_index']):x['group'] for x in d['per_frame']};idx=[int(x) for x in d['parameters']['event_indices']] if a.all_s2d_events else a.event_indices
 if a.all_s2d_events and (len(idx)!=60 or set(idx)!=set(labels)):raise RuntimeError('exact S2d set required')
 ev=[json.loads(x) for x in (a.run_path/'events.jsonl').read_text().splitlines()];c=V51Controller(Settings(neural_ms=500,pulse_ms=200,learning=False));b=c.brain;z=np.load('data/graph.npz');ids=z['ids'];ptr=z['ptr'];post=z['post'];sup=z['superclass'];ann=annotations(ids);typ=ann.type.fillna('').astype(str).to_numpy();idn={int(x):i for i,x in enumerate(ids)};leader=idn[LEADER]; source_nodes=sorted({i for i in range(b.n) if leader in post[ptr[i]:ptr[i+1]]});slot=np.full(b.n,-1,np.int32)
 for k,i in enumerate(source_nodes):slot[i]=k
 ck=Path('/tmp/s2i_clean.npz');c.save(ck);out=[]
 for ix in idx:
  frame=render_frame_from_observation(ev[ix]['observation']);c.brain.restore(ck);c.brain.trace_observer_s2i=None;o=c.observe(frame,'none');counts=b.counts.copy();v=b.v.copy();g=b.g.copy();r=b.refractory.copy();c.brain.restore(ck);t=TraceS2I(slot,leader);t.reset();c.brain.trace_observer_s2i=C.addressof(t.native);on=c.observe(frame,'none');c.brain.trace_observer_s2i=None;t.fail();eq={'spike_sha256':o['spike_sha256']==on['spike_sha256'],'counts':bool(np.array_equal(counts,b.counts)),'kc_counts':bool(np.array_equal(counts[np.asarray(b.circuit['kc'])],b.counts[np.asarray(b.circuit['kc'])])),'v':bool(np.array_equal(v,b.v)),'g':bool(np.array_equal(g,b.g)),'refractory':bool(np.array_equal(r,b.refractory))}
  rows=[]
  for k,i in enumerate(source_nodes):
   if t.source_events[k]:rows.append({'source_body_id':int(ids[i]),'source_type':str(typ[i]),'source_superclass':str(sup[i]),'spike_first_step':int(t.source_first[k]) if t.source_first[k]>=0 else None,'delivery_events':int(t.source_events[k]),'signed':float(t.source_signed[k]),'positive':float(t.source_positive[k]),'negative':float(t.source_negative[k]),'first_delivery_step':int(t.source_first_delivery[k]),'last_delivery_step':int(t.source_last_delivery[k]),'pre_count':int(t.source_pre_count[k]),'pre_first_step':int(t.source_pre_first[k]) if t.source_pre_first[k]>=0 else None,'pre_last_step':int(t.source_pre_last[k]) if t.source_pre_last[k]>=0 else None})
  out.append({'event_index':ix,'group':labels.get(ix),'leader_body_id':LEADER,'leader_first_spike_step':int(t.leader_first[0]) if t.leader_first[0]>=0 else None,'incoming_rows':rows,'source_slot_total':len(source_nodes),'equivalence':eq,'overflow':bool(t.native.overflow)})
 result={'data_source':'direct_kernel_trace','leader_body_id':LEADER,'events':out,'provenance':{'s2d_path':str(a.s2d_main.resolve()),'s2d_sha256':sha(a.s2d_main),'kernel_path':str(Path('stonkfly/neural/kernel.cpp').resolve()),'kernel_sha256':sha(Path('stonkfly/neural/kernel.cpp')),'script_path':str(Path(__file__).resolve()),'script_sha256':sha(Path(__file__))},'limitations':['Strict pre means delivery_step < leader_first_spike_step; same-step delivery is excluded.','All outputs are observational, not causal.']};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({'source_n':len(source_nodes),'events':len(out),'file_size_bytes':a.output.stat().st_size}))
if __name__=='__main__':main()
