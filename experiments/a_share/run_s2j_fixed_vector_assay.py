"""S2j synthetic fixed-vector sensory mechanism assay (observation only)."""
import argparse, hashlib, json, math, tempfile
from pathlib import Path
import numpy as np
from stonkfly.config import Settings
from stonkfly.neural.v51_controller import V51Controller
from stonkfly.neural.common import annotations
from stonkfly.neural.sensory_adapter import FixedVectorSensoryAdapter

SCALES=(3.0,6.0,9.0,12.0,18.0,24.0,30.0); OFFSET=0.0
V3_SCALES=(1.0,2.0,3.0,4.0,6.0,9.0,12.0,18.0)

def sha(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def mapping(brain, include_r8=False):
    a=annotations(brain.ids); ret=np.asarray(brain.retina,dtype=np.int64); side=np.asarray(a.iloc[ret].rootSide)
    uv=np.asarray(brain.uv,dtype=np.float64); groups=[]; names=[]
    for s in ('L','R'):
        ix=np.flatnonzero(side==s)
        if not len(ix): raise RuntimeError(f'empty eye {s}')
        xm=float(np.median(uv[ix,0])); ym=float(np.median(uv[ix,1]))
        for qx,qy,label in ((0,0,'xlo_ylo'),(0,1,'xlo_yhi'),(1,0,'xhi_ylo'),(1,1,'xhi_yhi')):
            pick=ix[((uv[ix,0] < xm) if qx==0 else (uv[ix,0] >= xm)) & ((uv[ix,1] < ym) if qy==0 else (uv[ix,1] >= ym))]
            if not len(pick): raise RuntimeError(f'empty group {s}_{label}')
            groups.append(ret[pick].astype(int).tolist()); names.append(f'{s}_{label}')
    r1_groups=[list(g) for g in groups]; r8_groups=[[] for _ in range(8)]
    if include_r8:
        r8=np.asarray(brain.r8,dtype=np.int64); rs=np.asarray(a.iloc[r8].rootSide)
        for node,u,s in zip(r8,np.asarray(brain.r8_uv,dtype=np.float64),rs):
            eye=0 if s=='L' else 1; xm,ym=[float(np.median(uv[np.flatnonzero(side==('L' if eye==0 else 'R')),k])) for k in (0,1)]
            qx=1 if u[0]>=xm else 0; qy=1 if u[1]>=ym else 0; r8_groups[eye*4+qx*2+qy].append(int(node))
        if set(sum(r8_groups,[]))!=set(map(int,r8.tolist())) or len(sum(r8_groups,[]))!=len(set(sum(r8_groups,[]))): raise RuntimeError('R8 mapping invalid')
        groups=[r1_groups[i]+r8_groups[i] for i in range(8)]
    flat=[i for g in groups for i in g]
    expected=set(ret.tolist()) | (set(map(int,np.asarray(brain.r8).tolist())) if include_r8 else set())
    if len(flat)!=len(set(flat)) or set(flat)!=expected: raise RuntimeError('retina mapping not mutually exclusive/complete')
    ids=[[int(brain.ids[i]) for i in g] for g in groups]
    return names,groups,ids,{'retina_n':len(ret),'r8_n':len(getattr(brain,'r8',[])) if include_r8 else 0,'include_r8':include_r8,'r1_group_sizes':[len(g) for g in r1_groups],'r8_group_sizes':[len(g) for g in r8_groups],'group_sizes':[len(g) for g in groups],'body_ids':ids,'r8_body_ids':[[int(brain.ids[i]) for i in g] for g in r8_groups],'uv_medians':{s:[float(np.median(uv[np.flatnonzero(side==s),k])) for k in (0,1)] for s in ('L','R')}}

def mapping_v3(brain):
    a=annotations(brain.ids); nodes=np.unique(np.r_[brain.retina,brain.r8]).astype(np.int64); uv=np.zeros((len(nodes),2),dtype=float); side=np.asarray(a.iloc[nodes].rootSide)
    r1map={int(n):u for n,u in zip(brain.retina,brain.uv)}; r8map={int(n):u for n,u in zip(brain.r8,brain.r8_uv)}
    for i,n in enumerate(nodes): uv[i]=r1map[int(n)] if int(n) in r1map else r8map[int(n)]
    groups=[]; bounds={}; names=[]
    for eye in ('L','R'):
        ix=np.flatnonzero(side==eye); xm=float(np.median(uv[ix,0])); qs=np.quantile(uv[ix,1],[.25,.5,.75]); bounds[eye]={'x_median':xm,'y_quartiles':[float(x) for x in qs]}
        for xb in range(2):
            for yb in range(4):
                ymask=(uv[ix,1] < qs[0]) if yb==0 else ((uv[ix,1] >= qs[yb-1]) & ((uv[ix,1] < qs[yb]) if yb<3 else True))
                pick=ix[((uv[ix,0]<xm) if xb==0 else (uv[ix,0]>=xm)) & ymask]
                if not len(pick): raise RuntimeError('empty v3 bin')
                groups.append(nodes[pick].astype(int).tolist()); names.append(f'{eye}_x{xb}_y{yb}')
    if set(sum(groups,[]))!=set(map(int,nodes.tolist())) or len(sum(groups,[]))!=len(set(sum(groups,[]))): raise RuntimeError('v3 bins invalid')
    H=np.array([[1]],dtype=int)
    while len(H)<16:H=np.block([[H,H],[H,-H]])
    rows=H[1:9]; ch=[]
    for row in rows: ch.append([i for i,v in enumerate(row) if v>0])
    masks=[]; cgroups=[]
    for slots in ch:
        cgroups.append(sum((groups[i] for i in slots),[])); masks.append(slots)
    ids=[[int(brain.ids[i]) for i in g] for g in cgroups]
    overlaps=[[len(set(ch[i])&set(ch[j])) for j in range(8)] for i in range(8)]
    return [f'ch{i}' for i in range(8)],cgroups,ids,{'version':'v3_distributed_16bin','nodes_n':len(nodes),'bin_names':names,'bin_sizes':[len(g) for g in groups],'bin_body_ids':[[int(brain.ids[i]) for i in g] for g in groups],'bounds':bounds,'channel_masks':masks,'channel_group_sizes':[len(g) for g in cgroups],'theoretical_bin_overlap':overlaps,'mapping_hash':sha(np.asarray(nodes,dtype=np.int64))}
def stimuli():
    out=[('zero',[0.0]*8,[False]*8)]
    for i in range(8):
        v=[0.0]*8;v[i]=1.0;out.append((f'onehot_{i}',v,[False]*8))
        for a in (.25,.5,.75,1.0):
            w=[0.0]*8;w[i]=a;out.append((f'ch{i}_amp_{a:g}',w,[False]*8))
        miss=[False]*8;miss[i]=True;out.append((f'ch{i}_missing', [1.0 if j==i else 0.0 for j in range(8)],miss))
    for n,v in [('all_025',[.25]*8),('all_05',[.5]*8),('all_075',[.75]*8),('all_1',[1.0]*8),('dense_A',[1,0,1,0,1,0,1,0]),('dense_B',[0,1,0,1,0,1,0,1]),('dense_C',[1,.75,.5,.25,.25,.5,.75,1])]: out.append((n,v,[False]*8))
    d={};
    for x in out:d[x[0]]=x
    return list(d.values())
def run(output,scales,names=None,include_r8=False,distributed=False):
    c=V51Controller(Settings(learning=False));b=c.brain; names8,groups,ids,mp=(mapping_v3(b) if distributed else mapping(b,include_r8=include_r8))
    adapter_by={s:FixedVectorSensoryAdapter(names8,groups,neuron_ids=ids,current_scale=s,current_offset=OFFSET) for s in scales}
    fs=stimuli(); selected=fs if names is None else [x for x in fs if x[0] in names]; black=np.zeros((64,64,3),dtype=np.uint8)
    clean=Path(tempfile.gettempdir())/'s2j_clean.npz';c.save(clean);kc=np.asarray(b.circuit['kc'],dtype=np.int64); rec=[]
    baseline=None
    for scale in scales:
      ad=adapter_by[scale]
      for name,v,missing in selected:
        c.restore(clean); stim=ad.encode({'values':v,'missing_mask':missing}); counts,wall=b.rgb_step(black,500.0,learning=False,stimulation=stim.brain_pulses());b.counts[:]=counts
        active=np.flatnonzero(np.asarray(counts)[kc]>0); layers={}
        for s in np.unique(b.superclass):
          ix=np.flatnonzero(np.asarray(b.superclass)==s);layers[str(s)]={'spikes':int(counts[ix].sum()),'active_cells':int(np.count_nonzero(counts[ix]))}
        rec.append({'scale':scale,'name':name,'values':v,'missing_mask':missing,'currents':stim.currents,'frame_sha256':sha(black),'stimulation_hash':sha(np.asarray(stim.brain_pulses()[0][1],dtype=np.float32)),'kc_count':int(len(active)),'kc_ids':[int(kc[i]) for i in active],'kc_set_hash':sha(kc[active]),'spike_sha256':sha(np.asarray(counts,dtype=np.int32)),'layers':layers,'wall_seconds':float(wall),'learning':False,'reinforcement':'none','mapping_hash':stim.mapping_hash,'normalization_hash':stim.normalization_hash})
    by={s:[r for r in rec if r['scale']==s] for s in scales}; kc_total=len(kc)
    def assess(s):
      rr={r['name']:r for r in by[s]}
      if not all(f'onehot_{i}' in rr for i in range(8)):
        return {'complete':False,'available_names':sorted(rr),'zero_active':rr.get('zero',{}).get('kc_count')}
      ones=[rr[f'onehot_{i}'] for i in range(8)]; amp=[[rr[f'ch{i}_amp_{a:g}']['kc_set_hash'] for a in (.25,.5,.75,1.0)] for i in range(8)]
      pairs=[len(set(x['kc_ids'])&set(y['kc_ids']))/max(1,len(set(x['kc_ids'])|set(y['kc_ids']))) for i,x in enumerate(ones) for y in ones[i+1:]]
      return {'complete':True,'zero_active':rr['zero']['kc_count'],'onehot_counts':[x['kc_count'] for x in ones],'onehot_pairwise_max_jaccard':max(pairs),'onehot_exact_distinct':len({x['kc_set_hash'] for x in ones}),'channel_amp_distinct_min':min(len(set(x)) for x in amp),'dense_exact_distinct':len({rr[n]['kc_set_hash'] for n in ('dense_A','dense_B','dense_C')}),'dense_max_fraction':max(rr[n]['kc_count'] for n in ('dense_A','dense_B','dense_C'))/kc_total}
    assessments={str(s):assess(s) for s in scales}
    def passed(s):
      a=assessments[str(s)];return a.get('complete',False) and all(0.01*kc_total<=x<=0.2*kc_total for x in a['onehot_counts']) and a['onehot_exact_distinct']==8 and a['onehot_pairwise_max_jaccard']<=.8 and a['channel_amp_distinct_min']>=3 and a['dense_exact_distinct']==3 and a['dense_max_fraction']<.5
    selected_scale=next((s for s in scales if passed(s)),None)
    result={'data_source':'synthetic_fixed_vector_sensory_adapter','adapter_mapping':mp|{'feature_names':names8,'mapping_hash':adapter_by[scales[0]].mapping_hash,'normalization_hash':adapter_by[scales[0]].normalization_hash,'current_offset':OFFSET},'preregistered':{'scales':list(scales) if distributed else list(SCALES),'offset':OFFSET,'acceptance':'minimum scale passing all frozen criteria; no score optimization','onehot_fraction':[.01,.2],'onehot_pairwise_jaccard_max':.8,'channel_amp_min_distinct':3,'dense_exact_distinct':3,'dense_max_fraction':.5},'scales':list(scales),'stimulus_count':len(selected),'runs':len(rec),'records':rec,'assessments':assessments,'selection':{'status':'selected','scale':selected_scale} if selected_scale is not None else {'status':'no_acceptable_mapping_scale','scale':None},'graph_hashes':{'ids':sha(b.ids),'ptr':sha(b.ptr),'post':sha(b.post),'weight':sha(b.weight)},'limitations':['Synthetic mechanism assay only; no market data, labels, reward, decoder, or performance gate.','Full retained graph and weights are unchanged; learning=False and reinforcement none.']}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,allow_nan=False));return result
def main():
  ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--include-r8',action='store_true');ap.add_argument('--distributed',action='store_true');a=ap.parse_args(); names=['zero','onehot_0','onehot_1','all_1'] if a.smoke else None;sc=[6.0,12.0,18.0] if a.smoke else (list(V3_SCALES) if a.distributed else list(SCALES));r=run(a.output,sc,names,include_r8=a.include_r8,distributed=a.distributed);print(json.dumps({'runs':r['runs'],'scales':sc,'path':str(a.output),'distributed':a.distributed}))
if __name__=='__main__':main()
