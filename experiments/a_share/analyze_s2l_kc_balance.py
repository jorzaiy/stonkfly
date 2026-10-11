"""S2l KC pre-ignition input balance audit (observation only)."""
import argparse, ctypes as C, hashlib, json
from pathlib import Path
import numpy as np
from experiments.a_share.analyze_kc_overlap import render_frame_from_observation
from stonkfly.config import Settings
from stonkfly.neural.v51_controller import V51Controller
from stonkfly.neural.common import annotations, GRAPH
from stonkfly.neural.trace_s2l import TraceBuffersS2L

RUN=Path('runs/wc006_purged_fly_training/20261004T-balanced-reward-v2')
S2D=Path('runs/plasticity_diagnosis/S2d/20261004T-projection-threshold-v1/projection_threshold_analysis.json')
S2K=Path('runs/plasticity_diagnosis/S2k/20261006T-kc-recurrence-counterfactual-v1/s2k_counterfactual.json')
CLASSES=('KC','APL','DPM','PN/visual_projection','other_cb','other')
def fsha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def events():
 d=json.loads(S2D.read_text()); idx=[int(x) for x in d['parameters']['event_indices']]; lines=[json.loads(x) for x in (RUN/'events.jsonl').read_text().splitlines()]; return [(i,lines[i],d['per_frame'][k]) for k,i in enumerate(idx)]
def classify(b):
 a=annotations(b.ids); typ=a.type.fillna('').astype(str).to_numpy(); sup=np.asarray(b.superclass).astype(str); out=np.full(b.n,-1,np.int8); kc=np.asarray(b.circuit['kc']); out[kc]=0
 for i,x in enumerate(typ):
  if out[i]>=0: continue
  if int(b.ids[i]) in (10540,10977): out[i]=1
  elif x=='DPM': out[i]=2
  elif sup[i]=='visual_projection': out[i]=3
  elif sup[i]=='cb_intrinsic': out[i]=4
  else: out[i]=5
 return out,typ,sup,kc
def run(out, smoke=False):
 evs=events(); selected=evs[:2]+evs[-2:] if smoke else evs
 c=V51Controller(Settings(neural_ms=500,pulse_ms=200,learning=False)); b=c.brain
 tmp=out.parent/'s2l_clean_observer_build.npz'; c.save(tmp)
 # Array-level bridge audit: read historical checkpoint, never restore it into this build.
 old=RUN/'brain-clean.npz'; arrays=[]
 with np.load(old,allow_pickle=False) as a, np.load(tmp,allow_pickle=False) as n:
  for k in ('weight',*b.fields): arrays.append({'field':k,'shape_equal':a[k].shape==n[k].shape,'dtype_equal':str(a[k].dtype)==str(n[k].dtype),'bitwise_equal':bool(np.array_equal(a[k],n[k]))})
 bridge=all(x['shape_equal'] and x['dtype_equal'] and x['bitwise_equal'] for x in arrays)
 sc,typ,sup,kc=classify(b); ks=np.full(b.n,-1,np.int32);ks[kc]=np.arange(len(kc),dtype=np.int32)
 s2k={int(r['event_index']):r for r in json.loads(S2K.read_text())['records'] if r['arm']=='C'} if S2K.exists() else {}
 rec=[]; npz_rows=[]
 for ei,ev,expected in selected:
  frame=render_frame_from_observation(ev['observation']); c.restore(tmp); b.trace_observer_s2l=None
  off=c.observe(frame,'none'); off_counts=b.counts.copy(); off_state={k:(getattr(b,k).copy() if hasattr(getattr(b,k),'copy') else getattr(b,k)) for k in ['weight',*b.fields,'counts','queue','queue_count','cursor','sim_ms','total_spikes','weights_frozen'] if hasattr(b,k)}
  c.restore(tmp); t=TraceBuffersS2L(sc,ks); t.reset(); b.trace_observer_s2l=C.addressof(t.native); on=c.observe(frame,'none'); b.trace_observer_s2l=None; t.fail_if_overflow()
  eq={'frame_hash':ah(frame)==expected['frame_sha256'],'active_kc_count':int((off_counts[kc]>0).sum())==int(expected['active_kc_count']),'kc_spike_count':int(off_counts[kc].sum())==int(s2k.get(ei,{}).get('kc_spike_count',-1)),'spike_sha256':ah(off_counts)==s2k.get(ei,{}).get('spike_sha256'),'trace_counts':bool(np.array_equal(off_counts,b.counts))}
  eq['state']=all((np.array_equal(v,getattr(b,k)) if hasattr(v,'shape') else v==getattr(b,k)) for k,v in off_state.items())
  phase_total=int(t.phase_events.sum()); audit=int(t.all_kc_delivery_count[0]); eq['delivery_conservation']=(phase_total==audit)
  eq['source_class_coverage']=bool(np.all(sc>=0) and np.all(sc<6)); eq['all']=all(eq.values())
  first_count=int(np.count_nonzero(t.kc_first>=0)); first_min=int(t.kc_first[t.kc_first>=0].min()) if first_count else -1
  eq['kc_first_count']=(first_count==int(t.kc_unique[0])==int((off_counts[kc]>0).sum())); eq['kc1_min_matches']=(first_min==int(t.kc1_step[0]) if first_count else int(t.kc1_step[0])==-1); eq['kc_order_used']=(int(t.kc_order_used[0])==int(t.kc_unique[0])); eq['kc10_order_matches']=(int(t.kc10_step[0])==-1 or int(t.kc_order_step[9])==int(t.kc10_step[0])); eq['all']=all(eq.values())
  npz_rows.append((ei,t.kc_class_events.copy(),t.kc_class_signed.copy(),t.kc_class_positive.copy(),t.kc_class_negative.copy(),t.kc_first.copy(),t.kc_order_id.copy(),t.kc_order_step.copy()))
  rec.append({'event_index':ei,'category':expected['category'],'group':'low' if int(expected['active_kc_count'])<=10 else 'high' if int(expected['active_kc_count'])>=1000 else 'middle','frame_sha256':ah(frame),'kc1_step':int(t.kc1_step[0]),'kc10_step':None if int(t.kc10_step[0])<0 else int(t.kc10_step[0]),'kc_unique':int(t.kc_unique[0]),'phase_schema':'phase0 step<kc1; phase1 kc1<=step<kc10; phase2 step>=kc10; if KC10 absent phase1 continues','phase_events':t.phase_events.tolist(),'phase_signed':t.phase_signed.tolist(),'phase_positive':t.phase_positive.tolist(),'phase_negative':t.phase_negative.tolist(),'phase_first':t.phase_first.tolist(),'phase_last':t.phase_last.tolist(),'phase_unique_source':t.phase_unique_source.tolist(),'phase_unique_target':t.phase_unique_target.tolist(),'all_kc_delivery_count':int(t.all_kc_delivery_count[0]),'equivalence':eq})
 allg=bridge and all(r['equivalence']['all'] for r in rec)
 npz=out.with_name('kc_pre_spike_inputs.npz'); kc_body=np.asarray(b.ids)[kc]; tmp_npz=npz.with_suffix('.partial.npz'); np.savez_compressed(tmp_npz,event_indices=np.array([x[0] for x in npz_rows],np.int32),kc_class_events=np.stack([x[1] for x in npz_rows]),kc_class_signed=np.stack([x[2] for x in npz_rows]),kc_class_positive=np.stack([x[3] for x in npz_rows]),kc_class_negative=np.stack([x[4] for x in npz_rows]),kc_first_step=np.stack([x[5] for x in npz_rows]),kc_order_id=np.stack([x[6] for x in npz_rows]),kc_order_step=np.stack([x[7] for x in npz_rows]),kc_global_indices=kc,kc_body_ids=kc_body,class_names=np.array(CLASSES),kc_order_hash=ah(kc_body)); tmp_npz.replace(npz)
 result={'preregistered':True,'counterfactual':False,'learning':False,'production_eligible':False,'parameters':{'event_indices':[int(x[0]) for x in evs],'selected_count':len(selected),'classes':list(CLASSES),'phase_definition':'before KC1: step<KC1; KC1-to-before-KC10: KC1<step<KC10; after: step>=KC10; missing KC10 is null'},'provenance':{'s2d':str(S2D),'s2d_sha256':fsha(S2D),'historical_checkpoint':str(old),'historical_checkpoint_sha256':fsha(old),'observer_checkpoint':str(tmp),'observer_kernel_source_sha256':b.build['source_sha256'],'observer_kernel_binary_sha256':b.build['binary_sha256'],'graph':str(GRAPH.resolve()),'graph_sha256':fsha(GRAPH),'s2k_c':str(S2K),'s2k_c_sha256':fsha(S2K) if S2K.exists() else None},'checkpoint_bridge':{'all_persistent_arrays_bitwise_equal':bridge,'fields':arrays,'metadata_build_may_differ':True},'records':rec,'kc_input_npz':str(npz),'gates':{'all_pass':allg,'trace_off_matches_s2k_c':all(r['equivalence']['spike_sha256'] for r in rec),'trace_on_off_equivalent':all(r['equivalence']['trace_counts'] and r['equivalence']['state'] for r in rec)},'limitations':['Observation of accepted kernel g increment only; no counterfactual or causal claim.','Formal launch is prohibited unless 60/60 behavior gates pass.','KC10 summary is null until unique KC ordering is extended; phase after KC10 is empty when absent.']}
 ids=np.asarray(b.ids); source_meta=[]
 for ci,name in enumerate(CLASSES):
  body=np.sort(ids[sc==ci]).astype(np.int64); source_meta.append({'class':name,'priority':ci,'node_count':int(len(body)),'body_id_sha256':ah(body)})
 typ_arr=typ; vp=np.flatnonzero(sup=='visual_projection'); pn=np.flatnonzero(np.char.find(typ_arr.astype(str),'PN')>=0)
 source_class_metadata={'classes':source_meta,'visual_projection_vs_pn_annotation':{'visual_projection_count':int(len(vp)),'pn_annotation_count':int(len(pn)),'intersection_count':int(len(np.intersect1d(vp,pn))),'visual_only_count':int(len(np.setdiff1d(vp,pn))),'pn_only_count':int(len(np.setdiff1d(pn,vp))),'visual_body_id_sha256':ah(np.sort(ids[vp])),'pn_body_id_sha256':ah(np.sort(ids[pn]))}}
 apl_rows=[]
 for r in rec:
  pe=np.asarray(r['phase_events'])[1]; ps=np.asarray(r['phase_signed'])[1]; pnneg=np.asarray(r['phase_negative'])[1]; af=[int(x) for x in r['phase_first'][1] if int(x)>=0]; first=min(af) if af else -1; k10=r['kc10_step']; strict_ok=k10 is not None
  apl_rows.append({'event_index':r['event_index'],'kc1_step':r['kc1_step'],'kc10_step':k10,'first_delivery_step':None if first<0 else first,'delta_to_kc1':None if first<0 else first-r['kc1_step'],'delta_to_kc10':None if first<0 or k10 is None else first-k10,'phase0_events':int(pe[0]),'strict_pre_kc10_events':int(pe[0]+pe[1]) if strict_ok else None,'until_end_without_kc10_events':int(pe[0]+pe[1]) if not strict_ok else None,'phase0_signed':float(ps[0]),'strict_pre_kc10_signed':float(ps[0]+ps[1]) if strict_ok else None,'until_end_without_kc10_signed':float(ps[0]+ps[1]) if not strict_ok else None,'phase0_inhibition':float(pnneg[0]),'strict_pre_kc10_inhibition':float(pnneg[0]+pnneg[1]) if strict_ok else None,'until_end_without_kc10_inhibition':float(pnneg[0]+pnneg[1]) if not strict_ok else None,'phase0_unique_target':int(r['phase_unique_target'][1][0]),'strict_pre_kc10_unique_target':int(r['phase_unique_target'][1][0]+r['phase_unique_target'][1][1]) if strict_ok else None,'until_end_without_kc10_unique_target':int(r['phase_unique_target'][1][0]+r['phase_unique_target'][1][1]) if not strict_ok else None})
 result['source_class_metadata']=source_class_metadata
 s2j=Path('runs/plasticity_diagnosis/S2k/20261006T-kc-recurrence-counterfactual-v1-jaccard/active_kc_bitsets.npz'); fixed={'status':'unavailable','artifact':str(s2j)}
 if s2j.exists():
  class _EventTuple(tuple):
   def __getitem__(self,k):
    if k=='event_index': return tuple.__getitem__(self,0)
    return tuple.__getitem__(self,k)
  evs=[_EventTuple(x) for x in evs]
  _zcheck=np.load(s2j,allow_pickle=False)
  assert _zcheck['active_kc_bitsets'].shape==(360,508) and int(_zcheck['kc_n'])==4064 and str(_zcheck['bitorder'])=='little'
  assert [str(x) for x in _zcheck['arms']]==['C','A','B2','B4','AB2','AB4']
  assert [int(x) for x in _zcheck['event_indices']]==[int(x[0]) for x in events()]
  assert str(_zcheck['kc_order_hash'])==ah(kc.astype(np.int64))
  z=np.load(s2j,allow_pickle=False); bits=z['active_kc_bitsets']; arms=[str(x) for x in z['arms']]; eis=[int(x) for x in z['event_indices']]; low_ids=[int(r['event_index']) for r in evs if r[2]['active_kc_count']<=10]; pos=[eis.index(e) for e in low_ids]; cpos=arms.index('C'); masks=np.unpackbits(bits[cpos*60+np.asarray(pos)],axis=1,bitorder='little')[:,:len(kc)]; freq=masks.sum(axis=0); inter=np.flatnonzero(freq==len(pos)); fixed={'status':'complete','artifact':str(s2j),'arms':arms,'event_indices':eis,'kc_n':int(z['kc_n']),'bitorder':str(z['bitorder']),'kc_order_hash':str(z['kc_order_hash']),'low_event_indices':low_ids,'intersection_count':int(len(inter)),'intersection':[{'slot':int(q),'global_index':int(kc[q]),'body_id':int(ids[kc[q]]),'frequency':int(freq[q])} for q in inter],'frequency_distribution':{str(int(v)):int((freq==v).sum()) for v in np.unique(freq)}}
 result['fixed_low_kcs']=fixed
 if fixed.get('status')=='complete':
  slots=[int(x['slot']) for x in fixed['intersection']]; by_e={int(x[0]):x for x in npz_rows}; fixed['selected_input_summary']=[{'event_index':int(ei),'group':next(r['group'] for r in rec if r['event_index']==ei),'kc_slots':slots,'pre_own_spike_by_class':{'events':np.asarray(by_e[ei][1])[slots].sum(axis=0).astype(int).tolist(),'signed':np.asarray(by_e[ei][2])[slots].sum(axis=0).astype(float).tolist(),'positive':np.asarray(by_e[ei][3])[slots].sum(axis=0).astype(float).tolist(),'negative':np.asarray(by_e[ei][4])[slots].sum(axis=0).astype(float).tolist()}} for ei in by_e]
 def stats(vals):
  vals=np.asarray(vals,float); return {'n':int(len(vals)),'min':float(vals.min()) if len(vals) else None,'median':float(np.median(vals)) if len(vals) else None,'mean':float(vals.mean()) if len(vals) else None,'max':float(vals.max()) if len(vals) else None}
 summary={}
 for ci,name in enumerate(CLASSES):
  rows={}
  for group in ('all','low','high'):
   rr=[r for r in rec if group=='all' or r['group']==group]; p0=[r['phase_events'][ci][0] for r in rr]; u0=[r['phase_unique_target'][ci][0] for r in rr]; f0=[r['phase_first'][ci][0] for r in rr if r['phase_first'][ci][0]>=0]; eligible=[r for r in rr if r['kc10_step'] is not None]; strict=[r['phase_events'][ci][0]+r['phase_events'][ci][1] for r in eligible]; rows[group]={'pre_kc1_events':stats(p0),'pre_kc1_unique_target':stats(u0),'pre_kc1_first_step':stats(f0),'strict_pre_kc10_events':stats(strict),'strict_pre_kc10_eligible_n':len(eligible),'until_end_without_kc10':sum(r['kc10_step'] is None for r in rr)}
  summary[name]=rows
 result['summary']={'by_class':summary,'low_high_descriptive_only':True,'raw_total_ranking_forbidden':True,'phase_definition':result['parameters']['phase_definition']}
 for ci,name in enumerate(CLASSES):
  for group in ('all','low','high'):
   rr=[r for r in rec if group=='all' or r['group']==group]; summary[name][group]['pre_kc1_signed']=stats([r['phase_signed'][ci][0] for r in rr]); elig=[r for r in rr if r['kc10_step'] is not None]; summary[name][group]['strict_pre_kc10_signed']=stats([r['phase_signed'][ci][0]+r['phase_signed'][ci][1] for r in elig])
 def timing(rows,key): return stats([r[key] for r in rows if r[key] is not None])
 apl_by_group={g:{'first_delivery_step':timing([r for r in apl_rows if g=='all' or next(x['group'] for x in rec if x['event_index']==r['event_index'])==g],'first_delivery_step'),'delta_to_kc1':timing([r for r in apl_rows if g=='all' or next(x['group'] for x in rec if x['event_index']==r['event_index'])==g],'delta_to_kc1'),'delta_to_kc10':timing([r for r in apl_rows if g=='all' or next(x['group'] for x in rec if x['event_index']==r['event_index'])==g],'delta_to_kc10')} for g in ('all','low','high')}
 result['apl_summary']={'per_frame':apl_rows,'by_group':apl_by_group,'eligible_kc10_n':sum(r['kc10_step'] is not None for r in rec),'until_end_without_kc10_n':sum(r['kc10_step'] is None for r in rec),'interpretation':'APL timing/coverage observational; unequal raw windows are not ranked.'}
 result['parameters']['phase_definition']='phase0 step<kc1; phase1 kc1<=step<kc10; phase2 step>=kc10; missing kc10 leaves phase1 open'
 result['summary']['phase_definition']=result['parameters']['phase_definition']
 result['kc_input_npz']={'path':str(npz),'sha256':fsha(npz),'schema':['event_indices','kc_class_events','kc_class_signed','kc_class_positive','kc_class_negative','kc_first_step','kc_order_id','kc_order_step','kc_global_indices','kc_body_ids','class_names','kc_order_hash']}
 result['limitations']=['Observation of accepted kernel g increment only; no counterfactual or causal claim.','Low frames may not reach KC10; strict-pre-KC10 statistics use eligible frames and separately report until-end-without-KC10.','No unequal-window raw-total ranking.']
 out.parent.mkdir(parents=True,exist_ok=True); tmp_json=out.with_suffix('.partial'); tmp_json.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n'); tmp_json.replace(out); return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();r=run(a.output,a.smoke);print(json.dumps({'records':len(r['records']),'gates':r['gates']}))
if __name__=='__main__': main()
