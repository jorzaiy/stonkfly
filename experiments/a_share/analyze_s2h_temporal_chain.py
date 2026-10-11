import argparse,hashlib,json
from pathlib import Path
import numpy as np

SOURCES=(10530,11930,14437,19845); TARGET=13334
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--s2g',type=Path,required=True);ap.add_argument('--s2d',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--event-indices',type=int,nargs='+');ap.add_argument('--all-s2d-events',action='store_true');a=ap.parse_args()
 g=json.loads(a.s2g.read_text()); d=json.loads(a.s2d.read_text()); labels={int(x['event_index']):x['group'] for x in d['per_frame']}; indices=[int(x) for x in d['parameters']['event_indices']] if a.all_s2d_events else a.event_indices
 if a.all_s2d_events and (len(indices)!=60 or set(indices)!=set(labels)): raise RuntimeError('S2d event set mismatch')
 by={int(e['event_index']):e for e in g['events']}; out=[]
 for ix in indices:
  e=by[ix]; rows=[r for r in e['pre_spike_pair_rows'] if r['target_body_id']==TARGET and r['source_body_id'] in SOURCES]
  target=next((x for x in e['stable_targets'] if x['body_id']==TARGET),None)
  out.append({'event_index':ix,'group':labels.get(ix),'source_to_13334':[{'source_body_id':r['source_body_id'],'source_type':r['source_type'],'source_superclass':r['source_superclass'],'delivery_first_step':r['first_step'],'delivery_last_step':r['last_step'],'events':r['events'],'signed':r['signed'],'positive':r['positive'],'negative':r['negative'],'source_spike_first_step':None,'delivery_before_target_spike':target and target['first_spike_step'] is not None and r['first_step']<target['first_spike_step']} for r in rows],'target_13334_first_spike_step':target['first_spike_step'] if target else None,'downstream_actual_delivery_observed':False,'downstream_note':'S2g observer targeted the 14 stable CBs and did not record 13334 output deliveries; graph paths are not substituted for dynamic delivery.','equivalence':all(e['equivalence'].values()),'overflow':e['overflow']})
 high=[x for x in out if x['group']=='high']; low=[x for x in out if x['group']=='low']; support={s:sum(any(r['source_body_id']==s and r['delivery_before_target_spike'] for r in x['source_to_13334']) for x in high) for s in SOURCES}; any_support=sum(any(r['delivery_before_target_spike'] for r in x['source_to_13334']) for x in high)
 result={'provenance':{'s2g_path':str(a.s2g.resolve()),'s2g_sha256':sha(a.s2g),'s2d_path':str(a.s2d.resolve()),'s2d_sha256':sha(a.s2d),'script_path':str(Path(__file__).resolve()),'script_sha256':sha(Path(__file__))},'sources':list(SOURCES),'target':TARGET,'events':out,'high_support':{'any_source_delivery_before_target':any_support,'by_source':support,'denominator':len(high)},'hub_companion_preregistration':{'hub_compatible':'target follows source delivery and precedes multiple observed downstream dynamic events, with actual output coverage','companion_compatible':'near-synchronous target/downstream behavior with weak leading/output coverage','current_status':'undetermined: 13334 output deliveries and source spike emission were not recorded by S2g'},'limitations':['This is read-only postprocessing of S2g actual source->stable-CB deliveries. Source spike emission and 13334 downstream dynamic deliveries are unavailable and remain null; retained graph edges are not substituted.','No causal or counterfactual interpretation.','Low/high comparisons use event support and lead ordering only, not unequal-window raw totals.']};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False));result['file_size_bytes']=a.output.stat().st_size;a.output.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({'events':len(out),'high':len(high),'low':len(low),'file_size_bytes':a.output.stat().st_size}))
if __name__=='__main__':main()
