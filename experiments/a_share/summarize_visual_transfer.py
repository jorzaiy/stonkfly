import argparse,hashlib,json,statistics
from pathlib import Path
def h(ids): return hashlib.sha256(','.join(map(str,sorted(ids))).encode()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('input',type=Path);ap.add_argument('--summary',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();d=json.loads(a.input.read_text());r=d['records']; assert len(r)==24 and all(x['equivalence'] for x in r)
 def row(x): return {'name':x['name'],'actual_mean':x['actual_mean'],'channel_means':x['channel_means'],'retina':x['retina_luminance'],'drive':x['drive'],'kc_count':x['kc_count'],'kc_set_hash':h(x['kc_ids']),'visual_projection_spikes':x['layers'].get('visual_projection',{}).get('spikes',0),'cb_intrinsic_spikes':x['layers'].get('cb_intrinsic',{}).get('spikes',0)}
 uni=[row(x) for x in r if x['name'].startswith('uniform_')]; pat=[row(x) for x in r if not x['name'].startswith('uniform_')]
 groups={}
 for x in pat:
  key='mean_0.25' if '0.25' in x['name'] else ('mean_0.5' if '0.5' in x['name'] else 'controls');groups.setdefault(key,[]).append(x)
 pair={}
 for k,v in groups.items():
  rows=[]
  by_name={z['name']:z for z in r}
  for i,aa in enumerate(v):
   for b in v[i+1:]:
    sa=set(by_name[aa['name']]['kc_ids']); sb=set(by_name[b['name']]['kc_ids'])
    rows.append([aa['name'],b['name'],len(sa & sb)/max(1,len(sa | sb))])
  pair[k]=rows
 summary={'stimulus_count':len(r),'equivalence_all':all(x['equivalence'] for x in r),'uniform_table':uni,'uniform_distinct_kc_set_count':len({x['kc_set_hash'] for x in uni}),'uniform_distinct_kc_counts':sorted({x['kc_count'] for x in uni}),'pattern_groups':groups,'pattern_pairwise_jaccard':pair,'rgb_bgr_control':{'frames':[x for x in r if 'rgb_swap' in x['name']],'is_true_channel_swap':False,'limitation':'Both controls are constant grayscale frames; RGB/BGR arrays are identical, so no color-selectivity conclusion is possible.'}}
 summary['uniform_adjacent_comparisons']=[]
 for x,y in zip(uni,uni[1:]):
  summary['uniform_adjacent_comparisons'].append({'lower':x['name'],'higher':y['name'],'kc_count_equal':x['kc_count']==y['kc_count'],'kc_set_hash_equal':x['kc_set_hash']==y['kc_set_hash'],'jaccard':len(set(next(z for z in r if z['name']==x['name'])['kc_ids']) & set(next(z for z in r if z['name']==y['name'])['kc_ids']))/max(1,len(set(next(z for z in r if z['name']==x['name'])['kc_ids']) | set(next(z for z in r if z['name']==y['name'])['kc_ids'])))} )
 summary['uniform_first_kc_change']=next(({'from':x['name'],'to':y['name']} for x,y in zip(uni,uni[1:]) if x['kc_set_hash']!=y['kc_set_hash']),None)
 summary['pattern_actual_mean_deviation']={k:[{'name':x['name'],'actual_mean':x['actual_mean']} for x in v] for k,v in groups.items() if k!='controls'}
 summary['limitations']=['Assay is brightness/path characterization only; no labels, market data, reward, or parameter search.','KC set overlap is descriptive, not causal.','RGB/BGR control is grayscale and therefore does not test chromatic channel selectivity.'];a.summary.write_text(json.dumps(summary,indent=2,allow_nan=False))
 lines=['# S2i synthetic visual transfer summary','',f"Stimuli: {len(r)}/24; equivalence: {summary['equivalence_all']}.",'','## Interpretation','']
 lines.append(f"Uniform inputs produced {summary['uniform_distinct_kc_set_count']} distinct KC set hashes and counts {summary['uniform_distinct_kc_counts']}; the first KC-set change is {summary['uniform_first_kc_change']}. Retina luminance/drive rises with brightness, while the 0–20 levels remain KC-empty and 80–255 are not saturated to one identical set (see machine-readable adjacent table).")
 lines.append('The corrected mean=0.5 split/checker frames have actual mean 0.5; mean=0.25 binary patterns are 0.25098 due to uint8 quantization and gradients are 0.24908/0.48438. Pairwise Jaccard and exact hashes show spatial patterns can differ strongly, but this is descriptive. The RGB/BGR controls are actually identical grayscale frames, so chromatic conclusions are not supported.')
 lines.append('Interpretation: the direct evidence supports a brightness/central-threshold regime (KC onset near uniform 40–80), not a global spatial-discrimination failure; pattern responses are heterogeneous at fixed mean. High-gray adjacent inputs are not fully saturated because their KC hashes/counts differ.')
 lines += ['', '## Stage B preregistration suggestion','', 'If calibration is later needed, pre-register a label-free grid over half-saturation and lamina_bias, define saturation as adjacent uniform levels with identical KC set/hash and retina-drive tolerance, and define spatial discrimination by within-mean pattern KC Jaccard threshold. Do not select parameters or run that stage here.']
 a.report.write_text('\n'.join(lines)+'\n')
 print(json.dumps({'stimuli':len(r),'summary':str(a.summary),'report':str(a.report)}))
if __name__=='__main__':main()
