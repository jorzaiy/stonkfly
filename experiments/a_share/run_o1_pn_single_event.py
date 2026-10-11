#!/usr/bin/env python3
"""O1 full-graph single engineered PN spike-event qualification."""
from __future__ import annotations
import argparse,hashlib,json,platform,sys,time,traceback
from pathlib import Path
import numpy as np,pyarrow.feather as feather
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from stonkfly.neural.v2_brain import HybridBrain
OUT=ROOT/'runs/whole_brain_olfactory/O1/20261010T-pn-single-event-v1';PREREG=ROOT/'docs/20261010-o1-olfactory-pn-single-event-preregistered.md';TESTS=ROOT/'tests/test_o1_pn_single_event.py';O0=ROOT/'runs/whole_brain_olfactory/O0/20261010T-pn-proxy-inventory-v1'
PINS={'data/graph.npz':'346b8af85a11af13b8324e18669812c1924569e7d1adcb4e6f45cc461a2c344b','data/normalized/neurons.feather':'2103b5203d2ba5502c06fb70a4e0b23edcbcf79ed0c344722101941fc289ba2d','o0_raw':'d94f9815a061c7bd533193f24335933aeb699d948e8af02ca199cdd982420492','o0_formal':'8c09ccfccc9946bf95892cb1c77aa80ce60331e79394deb0ebb9f5372602f69b','o0_manifest':'71a81418d40d7cb152cbf1a45bd87f7afd50c7973c181da6995788160de15c95','q1_wrapper':'2673de4e7c4aa4ee0fd20636b17bf2c2828103bacfbbb05b2ff0f559979434e6','q1_binary':'9df8110f1e2134f8a6d88cde46d3a8a6278590f291ad87b49161926d2fd53f41'}
PIN_PATH={'data/graph.npz':ROOT/'data/graph.npz','data/normalized/neurons.feather':ROOT/'data/normalized/neurons.feather','o0_raw':O0/'o0_raw.npz','o0_formal':O0/'o0_formal.json','o0_manifest':O0/'o0_manifest.json','q1_wrapper':ROOT/'stonkfly/neural/v2_brain.py','q1_binary':ROOT/'runs/whole_brain_v2/Q1/.build/libv2_hybrid.so'}
DT=(.1,.05,.025);SEGMENTS=(2.,48.,450.,500.,500.,500.);PANEL_SHA='cc011844e8ed4e415353e8a835bca04a0e091d0179aaccef94e86d3e426d5301'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def fixed():
 actual={k:sha(v) for k,v in PIN_PATH.items()};bad={k:(PINS[k],actual[k]) for k in PINS if PINS[k]!=actual[k]}
 f=json.loads((O0/'o0_formal.json').read_text())
 if f.get('verdict')!='O0_PN_PROXY_INVENTORY_QUALIFIED' or f.get('panel',{}).get('panel_sha256')!=PANEL_SHA:bad['o0_verdict_panel']='changed'
 if bad:raise RuntimeError('O1 pin mismatch '+json.dumps(bad,sort_keys=True))
 return actual,f
def inject_single_event(brain,source):
 source=int(source)
 if not 0<=source<brain.n:raise ValueError('source outside graph')
 if int(brain.clock[0])!=0 or np.any(brain.spike_queue_count):raise RuntimeError('intent requires clean clock/queue')
 slot=brain.delay_steps%(brain.delay_steps+1);brain.spike_queue[slot,0]=source;brain.spike_queue_count[slot]=1
 return {'count':1,'source_node_index':source,'emission_clock':0,'arrival_clock':brain.delay_steps,'arrival_slot':slot}
def state_hash(b):
 h=hashlib.sha256()
 for name in ('v','g_event','refractory','spike_queue','spike_queue_count','active','active_flag','nactive','adaptation','counts'):
  h.update(name.encode());h.update(np.ascontiguousarray(getattr(b,name)).tobytes())
 h.update(str(int(b.clock[0])).encode());h.update(str(int(b.total_spikes)).encode());return h.hexdigest()
def trial(dt,source,panel_meta,repeat,segments=SEGMENTS):
 b=HybridBrain(ROOT/'data/graph.npz',dt=dt,graded_mask=np.zeros(166700,np.uint8));nodes=feather.read_table(ROOT/'data/normalized/neurons.feather').to_pydict();ct=np.asarray(nodes['cell_type'],str);kc=np.char.startswith(ct.astype(str),'KC');b.kc_mask[:]=kc
 weights_before=arrsha(b.weight);intent={'count':0}
 direct=b.post[b.ptr[source]:b.ptr[source+1]] if source is not None else np.empty(0,np.int32);direct_kc=direct[kc[direct]]
 if source is not None:intent=inject_single_event(b,source)
 segment_rows=[];direct_nonzero=False;direct_kc_nonzero=False;finite=True;bounds=True
 for si,duration in enumerate(segments):
  counts=b.run(duration);finite &= bool(np.isfinite(b.v).all() and np.isfinite(b.g_event).all() and np.isfinite(b.adaptation).all());bounds &= bool(np.max(np.abs(b.v))<1e6 and np.max(np.abs(b.g_event))<1e9)
  if si==0 and source is not None:
   direct_nonzero=bool(len(direct) and np.any(b.g_event[direct]!=0));direct_kc_nonzero=bool(len(direct_kc) and np.any(b.g_event[direct_kc]!=0))
  segment_rows.append({'duration_ms':duration,'spikes':int(counts.sum()),'kc_spikes':int(counts[kc].sum()),'active_nodes':int(b.nactive[0]),'queue_count':int(b.spike_queue_count.sum()),'v_min':float(b.v.min()),'v_max':float(b.v.max()),'g_abs_max':float(np.max(np.abs(b.g_event)))})
 ulp=float(abs(np.spacing(np.float32(-52.))));final_linf=float(np.max(np.abs(b.v-b.rest)));last_spikes=segment_rows[-1]['spikes'];queue_empty=bool(np.all(b.spike_queue_count==0));g_quiet=bool(np.max(np.abs(b.g_event))<=1e-7);adapt_quiet=bool(np.max(np.abs(b.adaptation))<=1e-4);stable=bool(finite and bounds and not b.overflow[0] and last_spikes==0 and queue_empty and g_quiet and adapt_quiet and final_linf<=4*ulp)
 zero_ok=bool(source is not None or (b.total_spikes==0 and max(r['g_abs_max'] for r in segment_rows)==0 and final_linf==0))
 propagation=bool(source is None or (direct_nonzero and direct_kc_nonzero))
 return {'name':'ZERO' if source is None else panel_meta['cell_type'],'source_node_index':None if source is None else int(source),'source_id':None if source is None else int(panel_meta['source_id']),'dt_ms':dt,'repeat':repeat,'intent':intent,'direct_target_count':len(direct),'direct_kc_count':len(direct_kc),'direct_target_nonzero_at_2ms':direct_nonzero,'direct_kc_nonzero_at_2ms':direct_kc_nonzero,'segments':segment_rows,'total_spikes':int(b.total_spikes),'finite':finite,'bounds':bounds,'overflow':int(b.overflow[0]),'final_linf_mv':final_linf,'four_ulp_mv':4*ulp,'queue_empty':queue_empty,'g_quiet':g_quiet,'adaptation_quiet':adapt_quiet,'last_500ms_spikes':last_spikes,'stable_exit':stable,'zero_ok':zero_ok,'propagation_ok':propagation,'weights_unchanged':weights_before==arrsha(b.weight),'final_state_sha256':state_hash(b)}
def evaluate(rows):
 by={}
 for r in rows:by.setdefault((r['dt_ms'],r['source_node_index']),[]).append(r)
 deterministic=all(len(v)==2 and v[0]['final_state_sha256']==v[1]['final_state_sha256'] and v[0]['segments']==v[1]['segments'] and v[0]['intent']==v[1]['intent'] for v in by.values())
 classes={}
 for r in rows:classes.setdefault(r['source_node_index'],set()).add(r['stable_exit'])
 dt_consistent=all(len(v)==1 for v in classes.values());implementation=all(r['weights_unchanged'] and r['finite'] and r['overflow']==0 for r in rows);propagation=all(r['propagation_ok'] and r['zero_ok'] for r in rows);stability=all(r['stable_exit'] for r in rows)
 return {'implementation':implementation,'propagation':propagation,'stability':stability,'deterministic_repeats':deterministic,'dt_class_consistent':dt_consistent,'trial_count':len(rows)}
def snapshot():return {'prereg':sha(PREREG),'runner':sha(Path(__file__)),'tests':sha(TESTS) if TESTS.exists() else None,'q1_wrapper':sha(PIN_PATH['q1_wrapper']),'q1_binary':sha(PIN_PATH['q1_binary'])}
def prepare(out):
 out=Path(out)
 if out.exists() and any(out.iterdir()):raise RuntimeError(f'refusing to overwrite non-empty output: {out}')
 out.mkdir(parents=True,exist_ok=True);return out
def write(out,f,raw,start,started):
 rp=out/'o1_raw.npz';np.savez_compressed(rp,**raw);f['wall_time_seconds']=time.time()-started;fp=out/'o1_formal.json';fp.write_text(json.dumps(f,indent=2,sort_keys=True)+'\n');report=out/'o1_report.md';report.write_text(f"# O1 PN single-event qualification\n\nVerdict: `{f['verdict']}`\n\nEngineered proxy events only; no market, learning, decoder, or execution.\n");lp=out/'launch.txt';lp.write_text(' '.join(sys.argv)+'\n');sp=out/'stdout.txt';sp.write_text(json.dumps({'verdict':f['verdict']},sort_keys=True)+'\n');m={'schema':'o1-manifest-v1','hashes_start':start,'hashes_end':snapshot(),'python':sys.version,'platform':platform.platform(),'artifacts':{'raw':sha(rp),'formal':sha(fp),'report':sha(report),'launch':sha(lp),'stdout':sha(sp)}};(out/'o1_manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n')
def run(out=OUT,smoke=False):
 started=time.time();out=prepare(out);pins,o0=fixed();start=snapshot();panel=o0['panel']['nodes'];dts=(.1,) if smoke else DT;selected=panel[:1] if smoke else panel;segments=(2.,8.) if smoke else SEGMENTS
 rows=[]
 for dt in dts:
  for rep in (1,2):
   rows.append(trial(dt,None,{},rep,segments))
   for meta in selected:rows.append(trial(dt,int(meta['node_index']),meta,rep,segments))
 gates=evaluate(rows)
 if smoke:verdict='O1_SMOKE_OK' if gates['implementation'] and gates['propagation'] and gates['deterministic_repeats'] else 'O1_IMPLEMENTATION_FAILURE'
 elif not gates['implementation'] or not gates['deterministic_repeats'] or not gates['dt_class_consistent']:verdict='O1_IMPLEMENTATION_FAILURE'
 elif not gates['propagation']:verdict='O1_PROPAGATION_FAILURE'
 elif not gates['stability']:verdict='O1_SINGLE_EVENT_INSTABILITY'
 else:verdict='O1_SINGLE_EVENT_QUALIFIED'
 formal={'schema':'o1-pn-single-event-v1','verdict':verdict,'smoke':smoke,'engineered_proxy':True,'production_eligible':False,'market_used':False,'learning_used':False,'pins':pins,'preregistration_sha256':sha(PREREG),'panel_sha256':PANEL_SHA,'segments_ms':list(segments),'gates':gates,'trials':rows}
 if snapshot()!=start:raise RuntimeError('O1 sources changed during run')
 raw={'trial_json':np.asarray(json.dumps(rows,sort_keys=True)),'source_node_index':np.asarray([-1 if r['source_node_index'] is None else r['source_node_index'] for r in rows]),'dt_ms':np.asarray([r['dt_ms'] for r in rows]),'total_spikes':np.asarray([r['total_spikes'] for r in rows]),'final_state_sha256':np.asarray([r['final_state_sha256'] for r in rows])};write(out,formal,raw,start,started);return formal
def failure(out,e):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);f={'schema':'o1-pn-single-event-v1','verdict':'O1_IMPLEMENTATION_FAILURE','error':repr(e),'traceback':traceback.format_exc(),'production_eligible':False};write(out,f,{'failure':np.asarray(repr(e))},snapshot(),time.time());return f
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);p.add_argument('--smoke',action='store_true');a=p.parse_args(argv)
 try:r=run(a.out,a.smoke)
 except Exception as e:r=failure(a.out,e);print(json.dumps(r,indent=2));raise
 print('O1_SMOKE_OK' if a.smoke and r['verdict']=='O1_SMOKE_OK' else json.dumps({'verdict':r['verdict'],'gates':r['gates'],'wall_time_seconds':r['wall_time_seconds']},indent=2,sort_keys=True))
if __name__=='__main__':main()
