import argparse,json,hashlib,statistics
from pathlib import Path
import numpy as np
SRC=(10530,11930,14437,19845)
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def medrange(xs):
 return {'n':len(xs),'median':statistics.median(xs) if xs else None,'min':min(xs) if xs else None,'max':max(xs) if xs else None}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('input',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();d=json.loads(a.input.read_text());es=d['events']; groups={g:[e for e in es if e['group']==g] for g in ('low','high')}
 out={'event_count':len(es),'group_counts':{g:len(v) for g,v in groups.items()},'equivalence_all':all(all(e['equivalence'].values()) for e in es),'overflow_any':any(e['overflow'] for e in es)}
 srcsum={}
 for s in SRC:
  z={'source_body_id':s}
  for g,ee in groups.items():
   spikes=[e['sources'][SRC.index(s)]['first_spike_step'] for e in ee]; dels=[e['sources'][SRC.index(s)] for e in ee]; target=[e['target_13334_first_spike_step'] for e in ee]; pre=[]; leads=[];lags=[]
   for r,t in zip(dels,target):
    if r['delivery_events_to_13334']>0 and t is not None and 'delivery_first_step' in r:
     pre.append(r['delivery_first_step']<t); leads.append(t-r['delivery_first_step']); lags.append(t-r['delivery_last_step'])
   z[g]={'spike_support':sum(x is not None for x in spikes),'delivery_support':sum(r['delivery_events_to_13334']>0 for r in dels),'delivery_before_target_frames':sum(pre),'first_delivery_lead':medrange([x for x in leads if x is not None]),'last_delivery_lead':medrange([x for x in lags if x is not None])}
  srcsum[str(s)]=z
 out['sources']=srcsum
 for g,ee in groups.items():
  target_sp=[e['target_13334_first_spike_step'] for e in ee]; out.setdefault('target_13334',{})[g]={'spike_support':sum(x is not None for x in target_sp),'n':len(ee)}
 out['at_least_one_source_before_target']={g:None for g in groups}
 out['all_four_sources_before_target']={g:None for g in groups}
 # Outgoing delivery event/target summaries split around target spike.
 outgoing={}
 for g,ee in groups.items():
  rows=[]
  for e in ee:
   t=e['target_13334_first_spike_step']
   for r in e['target_outgoing_deliveries']:
    phase='before_target_spike' if t is not None and r['step']<t else ('after_or_no_target_spike' if t is None or r['step']>=t else 'unknown'); key=(r['target_superclass'],r['target_type'],phase);z=outgoing.setdefault(key,{'events':0,'frames':0,'superclass':key[0],'type':key[1],'phase':phase});z['events']+=1
   rows.extend(e['target_outgoing_deliveries'])
  out['outgoing_'+g]={'delivery_events':len(rows),'unique_targets':len({r['target_body_id'] for r in rows}),'by_superclass_type_phase':list(outgoing.values())}
 # stable CB rank relative to target
 stable={g:[] for g in ('low','high')}; kc={g:[] for g in ('low','high')}
 for g,ee in groups.items():
  for e in ee:
   t=e['target_13334_first_spike_step']; vals=[x['first_spike_step'] for x in e['stable_cb_first_spikes']]; valid=[x for x in vals if x is not None]
   if t is not None:
    stable[g].extend([x-t for x in valid]);
   # kc_first_10_steps is a node-slot prefix, not a unique-spike sequence.
   kc[g].extend([x-t for x in e['kc_first_10_steps'] if x is not None and t is not None])
 out['stable_cb_relative_to_target']={g:medrange(v) for g,v in stable.items()};out['kc_prefix_relative_to_target']={g:medrange(v) for g,v in kc.items()}
 out['kc10_assessment']={'available':False,'reason':'kc_first_10_steps is the first ten KC node slots, with nulls; it is not a sorted unique-KC emission sequence, so KC10 cannot be inferred from this JSON.'}
 out['classification']={'status':'undetermined','reason':'direct outgoing coverage exists, but this artifact lacks a valid unique-KC10 sequence and does not encode downstream target spike ordering sufficiently for the preregistered hub/companion decision.'}
 out['limitations']=['Delivery after target spike is not attributed to that spike; delay queue may contain earlier emissions.','No raw unequal-window contribution totals are compared.','KC10 is intentionally not interpreted from node-slot prefix data.'];out['provenance']={'input_path':str(a.input.resolve()),'input_sha256':sha(a.input),'script_path':str(Path(__file__).resolve()),'script_sha256':sha(Path(__file__))}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps({'events':len(es),'file_size_bytes':a.output.stat().st_size}))
if __name__=='__main__':main()
