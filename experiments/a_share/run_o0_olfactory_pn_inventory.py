#!/usr/bin/env python3
"""Read-only O0 olfactory-PN proxy inventory."""
from __future__ import annotations
import argparse,hashlib,json,platform,sys,time,traceback
from pathlib import Path
import numpy as np
import pyarrow.feather as feather
import pyarrow as pa
import pyarrow.ipc as ipc

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"runs/whole_brain_olfactory/O0/20261010T-pn-proxy-inventory-v1"
PREREG=ROOT/"docs/20261010-o0-olfactory-pn-inventory-preregistered.md"
STOP=ROOT/"docs/20261010-q2-visual-route-stop-scope.md"
TESTS=ROOT/"tests/test_o0_olfactory_pn_inventory.py"
PINS={
 "data/graph.npz":"346b8af85a11af13b8324e18669812c1924569e7d1adcb4e6f45cc461a2c344b",
 "data/normalized/neurons.feather":"2103b5203d2ba5502c06fb70a4e0b23edcbcf79ed0c344722101941fc289ba2d",
 "data/normalized/edges.arrow":"842b762c9427c794eaedd453178aa034727a5084850a91fdcc4d3fffe09f5b7e",
 "docs/20261010-q2-visual-route-stop-scope.md":"dfa853f64051ea7ea5af4987602d46933efe03ba13381f1a75671789d80ee6dc",
}
EXPECTED={"orn":2635,"orn_direct_pn":594,"eligible":290}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def edge_hash(source,target,weight):
 h=hashlib.sha256()
 for name,a in (("source",np.asarray(source,np.int32)),("target",np.asarray(target,np.int32)),("weight",np.asarray(weight,np.float32))):h.update(name.encode());h.update(np.ascontiguousarray(a).tobytes())
 return h.hexdigest()

def fixed_inputs():
 actual={k:sha(ROOT/k) for k in PINS};bad={k:{"expected":v,"actual":actual[k]} for k,v in PINS.items() if actual[k]!=v}
 if bad:raise RuntimeError("O0 pin mismatch: "+json.dumps(bad,sort_keys=True))
 return actual

def set_record(indices,ids):
 x=np.asarray(indices,np.int64);sid=np.asarray(ids[x],dtype='<u8')
 return {"count":len(x),"node_index_sha256":ah(x.astype('<i8')),"source_id_sha256":ah(sid)}

def edge_record(source,target,weight):
 source=np.asarray(source,np.int32);target=np.asarray(target,np.int32);weight=np.asarray(weight,np.float32)
 return {"count":len(source),"positive_count":int((weight>0).sum()),"negative_count":int((weight<0).sum()),"zero_count":int((weight==0).sum()),"weight_sum":float(weight.astype(np.float64).sum()),"absolute_weight_sum":float(np.abs(weight.astype(np.float64)).sum()),"edge_sha256":edge_hash(source,target,weight)}

def inventory():
 g=np.load(ROOT/"data/graph.npz",allow_pickle=False);ptr=np.asarray(g['ptr'],np.int64);post=np.asarray(g['post'],np.int32);weight=np.asarray(g['weight'],np.float32);ids=np.asarray(g['ids'],np.int64)
 nodes=feather.read_table(ROOT/"data/normalized/neurons.feather").to_pydict();source_ids=np.asarray(nodes['source_id'],np.uint64).astype(np.int64);ct=np.asarray(['' if x is None else str(x) for x in nodes['cell_type']]);sc=np.asarray(nodes['superclass'],str);nt=np.asarray(nodes['neurotransmitter'],str)
 graph_ok=bool(len(ids)==166700 and len(post)==25582938 and ptr[0]==0 and ptr[-1]==len(post) and np.all(np.diff(ptr)>=0) and np.array_equal(ids,source_ids) and np.isfinite(weight).all())
 orn=np.char.startswith(ct.astype(str),'ORN_');kc=np.char.startswith(ct.astype(str),'KC');orn_target=np.zeros(len(ids),bool);orn_in_edges=np.zeros(len(ids),np.int64)
 os=[];ot=[];ow=[]
 for s in np.flatnonzero(orn):
  lo,hi=int(ptr[s]),int(ptr[s+1]);t=post[lo:hi];orn_target[t]=True;np.add.at(orn_in_edges,t,1)
  keep=np.array(['PN' in x for x in ct[t]])&(sc[t]=='cb_intrinsic')
  if np.any(keep):os.extend([int(s)]*int(keep.sum()));ot.extend(t[keep].tolist());ow.extend(weight[lo:hi][keep].tolist())
 pn_label=np.array(['PN' in x for x in ct]);direct=orn_target&pn_label&(sc=='cb_intrinsic');to_kc=np.zeros(len(ids),bool);kc_out_edges=np.zeros(len(ids),np.int64);es=[];et=[];ew=[]
 for s in np.flatnonzero(direct):
  lo,hi=int(ptr[s]),int(ptr[s+1]);t=post[lo:hi];keep=kc[t]
  if np.any(keep):to_kc[s]=True;kc_out_edges[s]=int(keep.sum())
 eligible=direct&(nt=='acetylcholine')&to_kc;gaba=direct&(nt=='gaba')
 for s in np.flatnonzero(eligible):
  lo,hi=int(ptr[s]),int(ptr[s+1]);t=post[lo:hi];keep=kc[t]
  es.extend([int(s)]*int(keep.sum()));et.extend(t[keep].tolist());ew.extend(weight[lo:hi][keep].tolist())
 eligible_idx=np.flatnonzero(eligible);covered=np.unique(np.asarray(et,np.int32))
 ranked=sorted(((hashlib.sha256(np.asarray(int(ids[i]),dtype='<u8').tobytes()).digest(),int(ids[i]),int(i)) for i in eligible_idx),key=lambda x:(x[0],x[1]));panel=np.asarray([x[2] for x in ranked[:8]],np.int64);panel_digest=[x[0].hex() for x in ranked[:8]]
 panel_payload=[{"node_index":int(i),"source_id":int(ids[i]),"cell_type":str(ct[i]),"neurotransmitter":str(nt[i]),"individual_digest":d} for i,d in zip(panel,panel_digest)]
 panel_hash=hashlib.sha256(json.dumps(panel_payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 with pa.memory_map(str(ROOT/"data/normalized/edges.arrow"),'r') as f:table=ipc.open_file(f).read_all()
 contacts=int(table.column('synapse_count').to_numpy(zero_copy_only=False).astype(np.uint64).sum(dtype=np.uint64))
 sets={"orn":set_record(np.flatnonzero(orn),ids),"orn_direct_pn":set_record(np.flatnonzero(direct),ids),"eligible_excitatory_pn":set_record(eligible_idx,ids),"gaba_orn_direct_pn":set_record(np.flatnonzero(gaba),ids),"kc_covered":set_record(covered,ids)}
 edges={"orn_to_direct_pn":edge_record(os,ot,ow),"eligible_pn_to_kc":edge_record(es,et,ew)}
 expected_match=sets['orn']['count']==EXPECTED['orn'] and sets['orn_direct_pn']['count']==EXPECTED['orn_direct_pn'] and sets['eligible_excitatory_pn']['count']==EXPECTED['eligible']
 gates={"graph_integrity":graph_ok and contacts==124177617,"expected_precheck_match":expected_match,"eligible_minimum":len(eligible_idx)>=32,"direct_kc_edges":len(es)>0,"kc_coverage":len(covered)>=1000,"panel_complete":len(panel)==8 and len(set(panel.tolist()))==8}
 rows=[{"node_index":int(i),"source_id":int(ids[i]),"cell_type":str(ct[i]),"neurotransmitter":str(nt[i]),"orn_incoming_edge_count":int(orn_in_edges[i]),"kc_outgoing_edge_count":int(kc_out_edges[i])} for i in eligible_idx]
 raw={"eligible_node_index":eligible_idx.astype(np.int64),"eligible_source_id":ids[eligible_idx].astype(np.int64),"eligible_cell_type":ct[eligible_idx].astype(str),"eligible_orn_incoming_edge_count":orn_in_edges[eligible_idx],"eligible_kc_outgoing_edge_count":kc_out_edges[eligible_idx],"covered_kc_node_index":covered.astype(np.int64),"panel_node_index":panel,"panel_source_id":ids[panel].astype(np.int64),"orn_pn_source":np.asarray(os,np.int32),"orn_pn_target":np.asarray(ot,np.int32),"orn_pn_weight":np.asarray(ow,np.float32),"pn_kc_source":np.asarray(es,np.int32),"pn_kc_target":np.asarray(et,np.int32),"pn_kc_weight":np.asarray(ew,np.float32)}
 result={"graph":{"nodes":len(ids),"edges":len(post),"contacts":contacts,"ids_aligned":bool(np.array_equal(ids,source_ids))},"sets":sets,"edges":edges,"eligible_rows":rows,"panel":{"rule":"sha256(little_endian_u64_source_id), then source_id","nodes":panel_payload,"panel_sha256":panel_hash},"gates":gates}
 return result,raw

def hash_snapshot():return {"prereg":sha(PREREG),"stop_scope":sha(STOP),"runner":sha(Path(__file__)),"tests":sha(TESTS) if TESTS.exists() else None}
def prepare(out):
 out=Path(out)
 if out.exists() and any(out.iterdir()):raise RuntimeError(f"refusing to overwrite non-empty output: {out}")
 out.mkdir(parents=True,exist_ok=True);return out
def write(out,formal,raw,start,started):
 rp=out/'o0_raw.npz';np.savez_compressed(rp,**raw);formal['wall_time_seconds']=time.time()-started;fp=out/'o0_formal.json';fp.write_text(json.dumps(formal,indent=2,sort_keys=True)+'\n');report=out/'o0_report.md';report.write_text(f"# O0 olfactory-PN proxy inventory\n\nVerdict: `{formal['verdict']}`\n\nO0 is read-only and does not qualify dynamics, learning, or trading.\n");manifest={"schema":"o0-manifest-v1","hashes_start":start,"hashes_end":hash_snapshot(),"python":sys.version,"platform":platform.platform(),"artifacts":{"raw":sha(rp),"formal":sha(fp),"report":sha(report)}};(out/'o0_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
def run(out=OUT):
 started=time.time();out=prepare(out);pins=fixed_inputs();start=hash_snapshot();result,raw=inventory();g=result['gates']
 if not g['graph_integrity'] or not g['expected_precheck_match'] or not g['panel_complete']:verdict='O0_IMPLEMENTATION_FAILURE'
 elif not g['eligible_minimum'] or not g['direct_kc_edges'] or not g['kc_coverage']:verdict='O0_NO_QUALIFYING_PN_SET'
 else:verdict='O0_PN_PROXY_INVENTORY_QUALIFIED'
 formal={"schema":"o0-olfactory-pn-inventory-v1","verdict":verdict,"read_only":True,"market_used":False,"learning_used":False,"production_eligible":False,"pins":pins,"preregistration_sha256":sha(PREREG),**result}
 if hash_snapshot()!=start:raise RuntimeError('O0 sources changed during run')
 write(out,formal,raw,start,started);return formal
def failure(out,exc):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);f={"schema":"o0-olfactory-pn-inventory-v1","verdict":"O0_IMPLEMENTATION_FAILURE","error":repr(exc),"traceback":traceback.format_exc(),"production_eligible":False};write(out,f,{"failure":np.asarray(repr(exc))},hash_snapshot(),time.time());return f
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);a=p.parse_args(argv)
 try:r=run(a.out)
 except Exception as exc:r=failure(a.out,exc);print(json.dumps(r,indent=2));raise
 print(json.dumps({"verdict":r['verdict'],"sets":r.get('sets'),"panel":r.get('panel')},indent=2,sort_keys=True))
if __name__=='__main__':main()
