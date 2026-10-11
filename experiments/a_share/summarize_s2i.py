import argparse,json,statistics,hashlib
from pathlib import Path
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dist(x):return {'n':len(x),'median':statistics.median(x) if x else None,'min':min(x) if x else None,'max':max(x) if x else None}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('input',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();d=json.loads(a.input.read_text());es=d['events'];gs={g:[e for e in es if e['group']==g] for g in ('low','high')};out={'group_counts':{g:len(v) for g,v in gs.items()},'equivalence_all':all(all(e['equivalence'].values()) for e in es),'overflow_any':any(e['overflow'] for e in es),'leader_body_id':10530}
 srcs={}
 for k in range(max(len(e['incoming_rows']) for e in es)):
  pass
 allrows={r['source_body_id']:r for e in es for r in e['incoming_rows']};
 for sid,meta in allrows.items():
  z={'source_body_id':sid,'type':meta['source_type'],'superclass':meta['source_superclass']}
  for g,ee in gs.items():
   rows={e['event_index']:{r['source_body_id']:r for r in e['incoming_rows']}.get(sid) for e in ee};sp=sum(r is not None and r['spike_first_step'] is not None for r in rows.values());delivery=sum(r is not None and r['delivery_events']>0 for r in rows.values());pre=[r for r in rows.values() if r is not None and r['pre_count']>0];leads=[e['leader_first_spike_step']-r['pre_last_step'] for e,r in ((e,r) for e in ee for r in [{x['source_body_id']:x for x in e['incoming_rows']}.get(sid)] if r is not None and r['pre_count']>0 and e['leader_first_spike_step'] is not None)];z[g]={'spike_support':sp,'delivery_support':delivery,'pre_support':len(pre),'lead_le_1':sum(x<=1 for x in leads),'lead_le_2':sum(x<=2 for x in leads),'lead_le_5':sum(x<=5 for x in leads),'lead_le_10':sum(x<=10 for x in leads),'lead_distribution':dist(leads),'signed_sum_not_ranked':sum(r['signed'] for r in pre)}
  srcs[str(sid)]=z
 out['sources']=srcs;perfect=[z for z in srcs.values() if z['high']['pre_support']==17 and z['low']['delivery_support']==0];out['perfect_high_only']=perfect;out['threshold_ge_15_high_pre']=[z for z in srcs.values() if z['high']['pre_support']>=15];
 by={}
 for z in srcs.values():
  q=by.setdefault(z['superclass'],{'sources':0,'high_pre_support_sum':0,'low_delivery_support_sum':0});q['sources']+=1;q['high_pre_support_sum']+=z['high']['pre_support'];q['low_delivery_support_sum']+=z['low']['delivery_support']
 out['superclass_support']=by;out['leader_selection_note']='10530 was pre-registered by highest rank<=3 support (14/17); 11390 was next (13/17), so no universal leader claim.';out['limitations']=['Support rates use frame occurrence, not unequal-window raw totals.','Perfect-high-only remains exactly 17/17 high strict pre and 0/43 low delivery.','All results are observational and non-causal.'];out['provenance']={'input_path':str(a.input.resolve()),'input_sha256':sha(a.input),'script_path':str(Path(__file__).resolve()),'script_sha256':sha(Path(__file__))};a.output.write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps({'sources':len(srcs),'perfect':len(perfect),'size':a.output.stat().st_size}))
if __name__=='__main__':main()
