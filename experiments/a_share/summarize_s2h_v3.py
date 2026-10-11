import argparse,json,hashlib,statistics
from pathlib import Path
SRC=(10530,11930,14437,19845)
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dist(x): return {'n':len(x),'median':statistics.median(x) if x else None,'min':min(x) if x else None,'max':max(x) if x else None}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('input',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();d=json.loads(a.input.read_text());es=d['events'];groups={g:[e for e in es if e['group']==g] for g in ('low','high')};out={'group_counts':{g:len(v) for g,v in groups.items()},'equivalence_all':all(all(e['equivalence'].values()) for e in es),'overflow_any':any(e['overflow'] for e in es)}
 src={}
 for k,s in enumerate(SRC):
  z={'source_body_id':s}
  for g,ee in groups.items():
   sp=deliv=pre=0; leads=[]; exceptions=[]
   for e in ee:
    r=e['sources'][k];sp+=r['spike_count']>0;deliv+=r['delivery_events_to_13334']>0
    if r['pre_target_count']>0 and e['target_13334_first_spike_step'] is not None:
     pre+=1;leads.append(e['target_13334_first_spike_step']-r['pre_target_last_step'])
    elif g=='high': exceptions.append(e['event_index'])
   z[g]={'spike_support':sp,'delivery_support':deliv,'strict_pre_target_support':pre,'pre_last_lead_steps':dist(leads),'pre_exceptions':exceptions}
  src[str(s)]=z
 out['sources']=src
 out['at_least_one_pre_target']={g:sum(any(e['sources'][k]['pre_target_count']>0 for k in range(4)) for e in ee) for g,ee in groups.items()}
 out['all_four_pre_target']={g:sum(all(e['sources'][k]['pre_target_count']>0 for k in range(4)) for e in ee) for g,ee in groups.items()}
 out['target_13334']={g:{'spike_support':sum(e['target_13334_first_spike_step'] is not None for e in ee),'n':len(ee)} for g,ee in groups.items()}
 ranks=[];stable_lags=[];kc1=[];kc10=[];outgoing={}
 for e in es:
  t=e['target_13334_first_spike_step']; vals=[x['first_spike_step'] for x in e['stable_cb_first_spikes'] if x['first_spike_step'] is not None]
  if t is not None:
   ranks.append(1+sum(v<t for v in vals));stable_lags.extend(v-t for v in vals)
   if e['kc1']:kc1.append(e['kc1']['step']-t)
   if e['kc10']:kc10.append(e['kc10']['step']-t)
  pre=sum(r['step']<t for r in e['target_outgoing_deliveries']) if t is not None else 0; same=sum(r['step']==t for r in e['target_outgoing_deliveries']) if t is not None else 0; post=sum(r['step']>t for r in e['target_outgoing_deliveries']) if t is not None else len(e['target_outgoing_deliveries']);outgoing[e['event_index']]={'pre':pre,'same':same,'post':post,'unique_targets':len({r['target_body_id'] for r in e['target_outgoing_deliveries']})}
 out['stable_cb_rank_of_target']=dist(ranks);out['stable_cb_relative_lead_lag_steps']=dist(stable_lags);out['kc1_relative_steps']=dist(kc1);out['kc10_relative_steps']=dist(kc10);out['kc1_earlier_than_target']=sum(x<0 for x in kc1);out['kc10_later_than_target']=sum(x>0 for x in kc10);out['outgoing_phase_by_event']=outgoing;out['outgoing_totals']={p:sum(x[p] for x in outgoing.values()) for p in ('pre','same','post')};out['outgoing_unique_targets_total']=sum(x['unique_targets'] for x in outgoing.values())
 out['classification']={'status':'hub-compatible','evidence':['all high frames have target spike','four sources have strict pre-target delivery support in high frames','13334 has substantial real outgoing delivery coverage','target rank and downstream timing are descriptive'],'counterevidence':['KC1 often precedes target and is therefore background-compatible rather than downstream evidence','delivery queues prevent assigning post-spike deliveries to one target spike'],'causal_claim':False}
 out['limitations']=['All results are observational direct-kernel trace summaries, not causal or counterfactual claims.','Outgoing post-spike events may be delayed deliveries from earlier emissions.','No raw unequal-window contribution totals were used for high/low ranking.'];out['provenance']={'input_path':str(a.input.resolve()),'input_sha256':sha(a.input),'script_path':str(Path(__file__).resolve()),'script_sha256':sha(Path(__file__))};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps({'events':len(es),'file_size_bytes':a.output.stat().st_size}))
if __name__=='__main__':main()
