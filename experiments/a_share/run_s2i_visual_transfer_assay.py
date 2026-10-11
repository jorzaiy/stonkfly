import argparse,hashlib,json,tempfile
from pathlib import Path
import numpy as np
from stonkfly.config import Settings
from stonkfly.neural.v51_controller import V51Controller
def digest(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def frames(n=64):
 out=[]; yy,xx=np.indices((n,n))
 def rgb(a):
  # Construct in a wide/float domain first: uint8 arithmetic would wrap
  # 256 to zero for the mean=0.5 patterns.
  a=np.asarray(a,dtype=np.float64); assert a.shape==(n,n)
  a=np.asarray(np.rint(np.clip(a,0.0,255.0)),dtype=np.uint8)
  return np.ascontiguousarray(np.repeat(a[...,None],3,axis=2))
 for x in [0,1,2,5,10,20,40,80,160,255]: out.append((f'uniform_{x}',rgb(np.full((n,n),x,np.uint8))))
 for mean in (.25,.5):
  val=int(round(mean*255));
  out += [(f'horizontal_mean_{mean}',rgb(np.where(yy < n//2,0,2*val))),(f'vertical_mean_{mean}',rgb(np.where(xx < n//2,0,2*val)))]
  for cell in (2,8):
   q=(yy//cell+xx//cell)%2;out.append((f'checker_{cell}_mean_{mean}',rgb(np.where(q,0,2*val))))
  grad=np.linspace(0,2*val,n,dtype=np.float64);out += [(f'lr_gradient_mean_{mean}',rgb(np.tile(grad,(n,1)))),(f'ud_gradient_mean_{mean}',rgb(np.tile(grad[:,None],(1,n))))]
 base=np.full((n,n,3),64,np.uint8);out += [('rgb_swap_rgb',base.copy()),('rgb_swap_bgr',base[...,::-1].copy())]
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--only',default=None,help='comma-separated stimulus names for targeted reruns');a=ap.parse_args();fs=frames();
 if a.only: fs=[x for x in fs if x[0] in set(a.only.split(','))]
 c=V51Controller(Settings(neural_ms=500,pulse_ms=200,learning=False));b=c.brain;ck=Path('/tmp/s2i_visual_clean.npz');c.save(ck);records=[]
 for name,f in fs:
  assert f.ndim==3 and f.shape[2]==3 and f.dtype==np.uint8 and f.flags.c_contiguous
  c.brain.restore(ck);r=c.observe(f,'none');kc=np.asarray(b.circuit['kc']);active=np.flatnonzero(b.counts[kc]>0);layers={}
  for s in np.unique(b.superclass):
   ix=np.flatnonzero(np.asarray(b.superclass)==s);layers[str(s)]={'spikes':int(b.counts[ix].sum()),'active_cells':int(np.count_nonzero(b.counts[ix]))}
  records.append({'name':name,'frame_sha256':digest(f),'actual_mean':float(f.mean()/255),'channel_means':[float(x/255) for x in f.mean(axis=(0,1))],'retina_luminance':{'mean':float(b.luminance.mean()),'min':float(b.luminance.min()),'max':float(b.luminance.max()),'sha256':digest(b.luminance)},'drive':{'mean':float(b.drive.mean()),'min':float(b.drive.min()),'max':float(b.drive.max())},'layers':layers,'kc_count':int(len(active)),'kc_ids':[int(x) for x in kc[active]],'spike_sha256':r['spike_sha256'],'equivalence':True,'reinforcement':'none','learning':False})
 for i,x in enumerate(records): x['kc_jaccard_to_previous']=None if i==0 else len(set(x['kc_ids'])&set(records[i-1]['kc_ids']))/max(1,len(set(x['kc_ids'])|set(records[i-1]['kc_ids'])))
 result={'data_source':'current_rgb_to_retina_luminance_visual_path','preregistered_stimuli':{'uniform_levels':[0,1,2,5,10,20,40,80,160,255],'patterns':['horizontal/vertical half split','checker cell 2/8','left-right/up-down gradients'],'means':[.25,.5],'channel_controls':['RGB/BGR constant-luminance controls']},'records':records,'limitations':['No FixedVectorSensoryAdapter; no labels/market/reinforcement/parameter changes.','RGB channel controls are identical under the current constant gray frame; production retina uses luminance conversion.']};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({'frames':len(records),'path':str(a.output)}))
if __name__=='__main__':main()
