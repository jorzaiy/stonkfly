"""Read-only review of the completed S2f JSON; never runs the simulator."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def ratio(high, low):
    if low == 0:
        return None
    return high / low

def rate_stats(rows, key, groups=('low','high')):
    out=[]
    for row in rows:
        lo,hi=row['low'],row['high']
        x={'body_id':row['body_id'],'type':row['type'],
           'low_frames':lo['frames'],'high_frames':hi['frames'],
           'low_rate':lo['rate'],'high_rate':hi['rate'],
           'rate_difference':hi['rate']-lo['rate'],
           'rate_ratio':ratio(hi['rate'],lo['rate']),
           'rate_ratio_semantics':'infinite_low_zero' if lo['rate']==0 else 'finite',
           'low_support':lo['frames'],'high_support':hi['frames'],
           'low_contribution':{k:lo[k] for k in ('events','signed','positive','negative')},
           'high_contribution':{k:hi[k] for k in ('events','signed','positive','negative')},
           'first_delivery_step_range':hi.get('first_delivery_step_summary',{}),
           'sample_only':True}
        out.append(x)
    return sorted(out,key=lambda x:(x['rate_difference'],x['high_support']),reverse=True)

def classify(rows, nlow=43, nhigh=17):
    perfect=[x for x in rows if x['high_frames']==nhigh and x['low_frames']==0]
    any_only=[x for x in rows if x['high_frames']>0 and x['low_frames']==0]
    return {'perfect_high_only':perfect,'any_high_only':any_only,
            'high_enriched_min_high_9':[x for x in rows if x['high_frames']>=9 and x['rate_difference']>0],
            'high_enriched_min_high_17':[x for x in rows if x['high_frames']==17 and x['rate_difference']>0]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('s2f',type=Path); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    d=json.loads(a.s2f.read_text()); events=d['events']; gs=d['group_summary']; nlow,nhi=gs['n_low'],gs['n_high']
    if len(events)!=60 or any(e.get('overflow') for e in events): raise RuntimeError('S2f must contain 60 events and no overflow')
    if not all(all(bool(v) for v in e['equivalence'].values()) for e in events): raise RuntimeError('S2f equivalence failure')
    labels={int(e['event_index']):e['group'] for e in events}
    proj=rate_stats(gs['projection_sources'],'projection_sources'); cb=rate_stats(gs['cb_sources'],'cb_sources')
    review={'source_summary':{'projection':classify(proj,nlow,nhi),'cb':classify(cb,nlow,nhi)},
            'projection_rate_sorted':proj,'cb_rate_sorted':cb,
            'boundary_sources':{str(i):next((x for x in proj if x['body_id']==i),None) for i in (10164,11670,11953,13285)},
            'all_events_equivalence':True,'overflow_false':True}
    stable={x['body_id']:x for x in cb if x['high_frames']==nhi and x['low_frames']==0}
    high_events=[]
    for e in events:
        if e['group']!='high': continue
        active=[r for r in e['cb_sources']['rows'] if r['phase_events'][0]>0 and r['body_id'] in stable]
        k10=e['kc_crossings']['10']; deliveries=[r['first_delivery_step'] for r in active if r['first_delivery_step'] is not None]
        high_events.append({'event_index':e['event_index'],'stable_cb_count':len(active),'earliest_stable_cb_delivery':min(deliveries) if deliveries else None,'kc10_step':k10,'delivery_to_kc10_margin':(k10-min(deliveries)) if deliveries and k10 is not None else None})
    review['stable_cb_high_frame_review']=high_events
    # Real retained CSR source -> post inspection. IDs/types are recovered from the graph and annotations.
    z=np.load('data/graph.npz',allow_pickle=False); ids=z['ids']; ptr=z['ptr']; post=z['post']; weights=z['weight']; sup=z['superclass']
    from stonkfly.neural.common import annotations
    typ=annotations(ids).type.fillna('').astype(str).to_numpy()
    proj_nodes=set(np.flatnonzero(sup=='visual_projection').tolist()); cb_nodes=set(np.flatnonzero(sup=='cb_intrinsic').tolist()); id_to_node={int(v):i for i,v in enumerate(ids)}
    stable_edges={int(body):[] for body in stable}
    stable_nodes={id_to_node[b] for b in stable if b in id_to_node}
    for i in proj_nodes:
        lo,hi=int(ptr[i]),int(ptr[i+1])
        for edge,j in enumerate(post[lo:hi],lo):
            if int(j) in stable_nodes:
                body=int(ids[i]); stable_edges[int(ids[j])].append({'body_id':body,'type':str(typ[i]),'weight':float(weights[edge]),'edge_direction':'projection_source_to_cb_target','structural_only':True})
    projection_by_body={x['body_id']:x for x in proj}
    for body, s in stable.items():
        for edge in stable_edges.get(body,[]):
            px=projection_by_body.get(edge['body_id'])
            edge['occurrence_class']='common' if px and px['low_rate']>0 else ('enriched' if px else 'unobserved_in_projection_summary')
        s['direct_projection_upstream']=sorted(stable_edges.get(body,[]),key=lambda x:abs(x['weight']),reverse=True)
        s['direct_upstream_count']=len(s['direct_projection_upstream'])
    review['stable_cb_direct_projection_edges']={str(k):v['direct_projection_upstream'] for k,v in stable.items()}
    review['stable_cb_sources']=[{'body_id':x['body_id'],'type':x['type'],'high_frames':x['high_frames'],'low_frames':x['low_frames'],'high_rate':x['high_rate'],'low_rate':x['low_rate'],'high_contribution':x['high_contribution'],'first_delivery_step_range':x['first_delivery_step_range']} for x in sorted(cb,key=lambda x:x['body_id']) if x['high_frames']==nhi and x['low_frames']==0]
    review['limitations']=['All rates, enrichment, perfect-presence labels, and margins are sample-only descriptive statistics.','CSR edges are structural retained source->target edges; an edge is not dynamic causation or effective current.','Stable CB phase0 means observed delivery in all 17 high frames and no low frame; it is not a causal switch.']
    review['provenance']={'s2f_path':str(a.s2f.resolve()),'s2f_sha256':sha(a.s2f),'graph_path':str(Path('data/graph.npz').resolve()),'graph_sha256':sha(Path('data/graph.npz')),'script_path':str(Path(__file__).resolve()),'script_sha256':sha(Path(__file__))}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(review,indent=2,allow_nan=False)); review['file_size_bytes']=a.output.stat().st_size; a.output.write_text(json.dumps(review,indent=2,allow_nan=False))
    print(json.dumps({'stable_cb':len(stable),'file_size_bytes':a.output.stat().st_size}))
if __name__=='__main__': main()
