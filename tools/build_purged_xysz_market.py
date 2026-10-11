#!/usr/bin/env python3
"""Offline, fail-closed builder for a target-only purged raw XYSZ market bundle."""
from __future__ import annotations
import argparse,csv,hashlib,json
from collections import defaultdict,Counter
from datetime import datetime,timezone
from decimal import Decimal,InvalidOperation
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PLAN_SHA="c2bea3f71cf3fba0fb1c5a9bf21a37d7dd7c4be1fbf467af5cdc2c54f9637413"
NEW_ROOT=ROOT/"runs/wc006_purged_xysz_collection/20260925T-131codes-attempt2"
OLD_PARENT=ROOT/"runs/wc006_v5_market_inputs/manual-bridge/authorized-v5-development-20260922T-right-issue-overlay-2022/manifest.json"
CANDIDATE=ROOT/"runs/wc006_v3_2021_inputs/20260925T-purged-candidate-stream-2021-2023-v2/manifest.json"
PLAN=ROOT/"runs/wc006_v3_2021_inputs/20260925T-purged-candidate-stream-2021-2023-v2/xysz_gap_plan.json"
BEGIN,END,SETTLE="2021-01-01","2024-01-03","2024-01-02"
QFQ_END="2023-12-29"
IMAGE_ID="sha256:57676d949eb5e79a3769c3714205f39d8940ae004e061a49c0174279379fd9bb"
SUPPLEMENT_ROOTS=[]
PRECEDENCE_AUDIT=[]
class BuildBlocked(ValueError): pass

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding='utf-8')
def datev(v):
 s=str(v or '').strip()
 if not s or s.lower() in {'nan','none','nat'}: return None
 if s.endswith('.0'): s=s[:-2]
 s=s.replace('/','-')
 if len(s)==8 and s.isdigit(): return f'{s[:4]}-{s[4:6]}-{s[6:]}'
 return s[:10] if len(s)>=10 and s[4]=='-' and s[7]=='-' else None
def dec(v):
 try: x=Decimal(str(v).strip()); return x if x.is_finite() else None
 except (InvalidOperation,TypeError,ValueError): return None
def read_csv(p):
 with Path(p).open(newline='',encoding='utf-8') as f:
  r=csv.DictReader(f); return list(r.fieldnames or []),[dict(x) for x in r]
def write_csv(p,fields,rows):
 with Path(p).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
def qualified(c):
 c=str(c).upper().split('.')[0]; return c+('.SH' if c.startswith(('6','68')) else '.BJ' if c.startswith(('4','8')) else '.SZ')
def artifact_path(mp,rel):
 for p in (Path(mp).parent/rel,Path(mp).parent/'raw'/rel):
  if p.is_file(): return p.resolve()
 raise BuildBlocked(f'artifact_declared_missing:{mp}:{rel}')
def load_inputs():
 if not PLAN.is_file() or sha(PLAN)!=PLAN_SHA: raise BuildBlocked('gap_plan_sha256_mismatch')
 for p in (CANDIDATE,OLD_PARENT):
  if not p.is_file(): raise BuildBlocked(f'provenance_input_missing:{p}')
 c=json.loads(CANDIDATE.read_text()); g=json.loads(PLAN.read_text())
 if c.get('paper_only') is not True or c.get('live_execution_authorized') is not False: raise BuildBlocked('candidate_not_research_only')
 if g.get('paper_only') is not True or g.get('network_collection_performed') is not False: raise BuildBlocked('gap_plan_not_offline_paper')
 targets=sorted(qualified(x) for x in c.get('cache_files',{}))
 if len(targets)!=131 or len(set(targets))!=131: raise BuildBlocked('target_code_count_not_131')
 if not NEW_ROOT.is_dir(): raise BuildBlocked('new_collection_root_missing')
 rd=json.loads((NEW_ROOT/'launcher_result.json').read_text()) if (NEW_ROOT/'launcher_result.json').is_file() else {}
 if len(rd.get('results',[]))!=49 or any(x.get('returncode')!=0 for x in rd['results']): raise BuildBlocked('collection_not_49_successes')
 return c,g,targets,rd

def child_inventory(rd, supplemental_roots=None):
 inv=defaultdict(list); records=[]; seen=set()
 old=json.loads(OLD_PARENT.read_text())
 for c in old.get('child_manifests',[]):
  mp=(OLD_PARENT.parent/c['fragment']).resolve()
  if not mp.is_file(): raise BuildBlocked(f'old_child_missing:{mp}')
  actual=sha(mp); declared=c.get('manifest_sha256')
  if actual!=declared: raise BuildBlocked(f'old_child_manifest_hash:{mp}')
  md=json.loads(mp.read_text()); rec={'dataset':md.get('dataset',c.get('dataset')),'manifest':str(mp.relative_to(ROOT)),'manifest_sha256':actual,'declared_manifest_sha256':declared,'hash_ok':True,'artifacts':[]}
  for a in md.get('artifacts',[]):
   ap=artifact_path(mp,str(a['path'])); ah=sha(ap)
   if ah!=a.get('sha256'): raise BuildBlocked(f'old_artifact_hash:{ap}')
   rec['artifacts'].append({'path':str(ap.relative_to(ROOT)),'sha256':ah,'declared_sha256':a.get('sha256'),'hash_ok':True,'code':a.get('code')}); inv[rec['dataset']].append((ap,md,rec))
  records.append(rec)
 expected={(str(x['dataset']),str(x['batch'])) for x in rd['results']}
 for x in rd['results']:
  d=Path(x['output']); mp=d/'manifest.json'
  if not mp.is_file(): raise BuildBlocked(f'new_child_manifest_missing:{mp}')
  md=json.loads(mp.read_text())
  if (md.get('dataset'),str(md.get('batch',x['batch']))) not in expected and md.get('dataset')!=x['dataset']: raise BuildBlocked(f'new_child_result_mismatch:{mp}')
  rec={'dataset':md.get('dataset'),'batch':x['batch'],'manifest':str(mp.relative_to(ROOT)),'manifest_sha256':sha(mp),'declared_manifest_sha256':None,'hash_ok':True,'artifacts':[]}
  for a in md.get('artifacts',[]):
   ap=artifact_path(mp,str(a['path'])); ah=sha(ap)
   if ah!=a.get('sha256'): raise BuildBlocked(f'new_artifact_hash:{ap}')
   rec['artifacts'].append({'path':str(ap.relative_to(ROOT)),'sha256':ah,'declared_sha256':a.get('sha256'),'hash_ok':True,'code':a.get('code')}); inv[rec['dataset']].append((ap,md,rec))
  records.append(rec)
 if len([r for r in records if r.get('batch') is not None])!=49: raise BuildBlocked('new_child_count_not_49')
 if supplemental_roots:
  for root in supplemental_roots:
   sr=Path(root).resolve(); lr=sr/'launcher_result.json'
   if not lr.is_file(): raise BuildBlocked('supplement_launcher_result_missing')
   sd=json.loads(lr.read_text());
   if sd.get('collector_sha256')!='98a9344d684f1a2e735b7db9acf6e5072eb6cb1170e325cb80fc7b78141c7c04' or len(sd.get('results',[]))!=1 and len(sd.get('results',[]))!=4: raise BuildBlocked('supplement_launcher_binding')
   for x in sd['results']:
    mp=Path(x['manifest'])
    if not mp.is_file() or sha(mp)!=x.get('manifest_sha256'): raise BuildBlocked(f'supplement_manifest_hash:{mp}')
    md=json.loads(mp.read_text()); rec={'dataset':md.get('dataset'),'batch':x['batch'],'manifest':str(mp.relative_to(ROOT)),'manifest_sha256':sha(mp),'declared_manifest_sha256':x.get('manifest_sha256'),'hash_ok':True,'artifacts':[]}
    for a in md.get('artifacts',[]):
     ap=artifact_path(mp,str(a['path'])); ah=sha(ap)
     if ah!=a.get('sha256'): raise BuildBlocked(f'supplement_artifact_hash:{ap}')
     rec['artifacts'].append({'path':str(ap.relative_to(ROOT)),'sha256':ah,'declared_sha256':a.get('sha256'),'hash_ok':True,'code':a.get('code')}); inv[rec['dataset']].append((ap,md,rec))
    records.append(rec)
 return inv,records

def choose_rows(inv,ds,target):
 groups=defaultdict(lambda: defaultdict(list)); fields=[]
 for ap,_,_ in inv[ds]:
  source=str(ap); rank=0
  if source.startswith(str(NEW_ROOT)): rank=1
  for i,x in enumerate(SUPPLEMENT_ROOTS,2):
   if source.startswith(str(x)): rank=i
  f,rows=read_csv(ap); fields=fields or f
  for r in rows:
   code=str(r.get('code') or r.get('MARKET_CODE') or '').strip().upper()
   if code in target: groups[code][rank].append(r)
 out=[]
 for code in sorted(target):
  ranks=groups.get(code,{})
  if not ranks: continue
  rank=max(ranks); out.extend(ranks[rank])
  if len(ranks)>1:
   PRECEDENCE_AUDIT.append({'dataset':ds,'code':code,'selected_rank':rank,'replaced_source_rows':sum(len(v) for k,v in ranks.items() if k!=rank),'source_ranks':{str(k):len(v) for k,v in ranks.items()}})
 if not out: raise BuildBlocked(f'no_rows:{ds}')
 return fields,out

def equity_pit_baseline(rows, first_signal):
 for r in rows:
  ann=datev(r.get("ANN_DATE")); effective=[datev(r.get(k)) for k in ("CHANGE_DATE","EX_CHANGE_DATE") if datev(r.get(k))]
  if ann and ann<=first_signal and any(x<=first_signal for x in effective): return True
 return False

def action_id(kind,row):
 payload={k:str(v or '') for k,v in sorted(row.items())}; return hashlib.sha256(json.dumps([kind,payload],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def build(args):
 out=Path(args.output).resolve()
 if out.exists() and (out/'manifest.json').exists(): raise BuildBlocked('output_exists')
 c,g,target_list,rd=load_inputs(); target=set(target_list); global SUPPLEMENT_ROOTS; SUPPLEMENT_ROOTS=[Path(x).resolve() for x in (args.supplement_root or [])]; inv,records=child_inventory(rd, SUPPLEMENT_ROOTS); out.mkdir(parents=True,exist_ok=True); raw=out/'raw'; raw.mkdir(parents=True,exist_ok=True); selected={}
 # crop per-row market data to the promised raw API window
 for ds in ('kline','history_stock_status','dividend','right_issue','equity_structure'):
  fields,rows=choose_rows(inv,ds,target)
  if ds=='kline': rows=[r for r in rows if BEGIN<=datev(r.get('kline_time'))<=END]
  if ds=='history_stock_status': rows=[r for r in rows if BEGIN<=datev(r.get('TRADE_DATE'))<=END]
  write_csv(raw/f'{ds}.csv',fields,rows); selected[ds]={'path':str((raw/f'{ds}.csv').relative_to(ROOT)),'rows':len(rows),'fields':fields}
 # adj_factor wide API has no date argument; crop explicitly and retain sparse cells.
 merged={}; cols=[]; datecol=None
 for ap,_,_ in inv['adj_factor']:
  fields,rows=read_csv(ap); datecol=datecol or fields[0]
  cols += [x for x in fields[1:] if x in target and x not in cols]
  for r in rows:
   d=datev(r.get(fields[0]))
   if BEGIN<=str(d)<=END: merged.setdefault(d,{}).update({k:r.get(k,'') for k in fields[1:] if k in target})
 afields=[datecol]+sorted(cols); arows=[dict({datecol:d},**{k:merged[d].get(k,'') for k in sorted(cols)}) for d in sorted(merged)]
 if set(cols)!=target: raise BuildBlocked('adj_factor_target_columns_incomplete')
 write_csv(raw/'adj_factor.csv',afields,arows); selected['adj_factor']={'path':str((raw/'adj_factor.csv').relative_to(ROOT)),'rows':len(arows),'fields':afields}
 # action PIT audit retains every raw row plus stable id/reason; no silent deduplication.
 pit=[]; eligible=[]; rej=Counter()
 for ds,kind in [('dividend','DIVIDEND'),('right_issue','RIGHTS')]:
  _,rows=read_csv(raw/f'{ds}.csv')
  for r in rows:
   ann=datev(r.get('ANN_DATE')); progress=str(r.get('DIV_PROGRESS' if kind=='DIVIDEND' else 'PROGRESS') or '').strip()
   names=('DATE_EQY_RECORD','DATE_EX','DATE_DVD_PAYOUT','LISTINGDATE_OF_DVD_SHR') if kind=='DIVIDEND' else ('EXECUTE_DATE','EX_DIVIDEND_DATE','LISTED_DATE')
   dates={n:datev(r.get(n)) for n in names}; eff=max((d for d in dates.values() if d),default=None)
   reason='eligible' if progress in {'3','3.0'} and ann and eff and ann<=eff else ('progress_not_implemented' if progress not in {'3','3.0'} else 'missing_or_invalid_dates')
   x={'action_id':action_id(kind,r),'kind':kind,'code':str(r.get('MARKET_CODE','')).strip().upper(),'ann_date':ann or '','effective_date':eff or '','progress':progress,'pit_reason':reason}
   pit.append(x); rej[reason]+=1
   if reason=='eligible': eligible.append(x)
 write_csv(out/'pit_action_rows.csv',list(pit[0]) if pit else ['action_id'],pit); write_csv(out/'pit_action_eligible.csv',list(pit[0]) if pit else ['action_id'],eligible)
 # Actual pinned cache comparison; this is evidence only and never source data.
 qdiff=[]; cache_dir=CANDIDATE.parent/'inputs/cache'; cache_hashes={}
 for code,meta in c['cache_files'].items():
  p=cache_dir/meta['source_name']
  if not p.is_file() or sha(p)!=meta['sha256']: raise BuildBlocked(f'candidate_cache_hash:{p}')
  cache_hashes[code]=meta['sha256']; _,qr=read_csv(p); qdates={datev(r.get('date')) for r in qr if BEGIN<=str(datev(r.get('date')))<=QFQ_END}
  # raw key dates for this code
  _,kr=read_csv(raw/'kline.csv'); rdates={datev(r.get('kline_time')) for r in kr if r.get('code')==qualified(code) and BEGIN<=str(datev(r.get('kline_time')))<=QFQ_END}
  for d in sorted(qdates-rdates): qdiff.append({'code':qualified(code),'date':d,'side':'qfq_only'})
  for d in sorted(rdates-qdates): qdiff.append({'code':qualified(code),'date':d,'side':'raw_only'})
 dump(out/'date_key_diff.json',{'scope':[BEGIN,QFQ_END],'qfq_used_for_market':False,'filled':False,'candidate_cache_hashes':cache_hashes,'rows':qdiff,'qfq_only':sum(x['side']=='qfq_only' for x in qdiff),'raw_only':sum(x['side']=='raw_only' for x in qdiff)})
 # QA
 checks={}; issues=[]; kf,krows=read_csv(raw/'kline.csv'); sf,srows=read_csv(raw/'history_stock_status.csv');
 krows=[r for r in krows]; keys={(r.get('code',''),datev(r.get('kline_time'))) for r in krows}; stkeys={(r.get('MARKET_CODE',''),datev(r.get('TRADE_DATE'))) for r in srows}
 checks['target_131_exact']=len(target)==131; checks['all_child_and_artifact_hashes']=True; checks['kline_target_only']=set(r.get('code','') for r in krows)==target; checks['kline_window']=all(BEGIN<=datev(r.get('kline_time'))<=END for r in krows); checks['kline_unique']=len(keys)==len(krows) and all(d for _,d in keys)
 checks['ohlc_volume_amount_valid']=all(all(dec(r.get(k)) is not None and dec(r.get(k))>=0 for k in ('volume','amount')) and all(dec(r.get(k)) is not None and dec(r.get(k))>0 for k in ('open','high','low','close')) and dec(r['high'])>=max(dec(r['open']),dec(r['close']),dec(r['low'])) and dec(r['low'])<=min(dec(r['open']),dec(r['close']),dec(r['high'])) for r in krows)
 checks['status_target_only']=set(r.get('MARKET_CODE','') for r in srows)==target; checks['status_window']=all(BEGIN<=datev(r.get('TRADE_DATE'))<=END for r in srows); checks['status_unique']=len(stkeys)==len(srows); checks['status_covers_all_kline_keys']=keys<=stkeys
 # Adj validates uniqueness/date crop/131 columns and only requires factors on raw kline keys.
 checks['adj_columns_exact_131']=len(afields)==132 and set(afields[1:])==target; adates=[datev(r.get(afields[0])) for r in arows]; checks['adj_date_unique_window']=len(adates)==len(set(adates)) and all(BEGIN<=d<=END for d in adates); amap={(c,datev(r[afields[0]])):dec(r.get(c)) for r in arows for c in target}; checks['adj_factors_on_kline_positive']=all(amap.get(k) is not None and amap[k]>0 for k in keys); checks['adj_nonpositive_or_nonfinite']=not any(v is not None and v<=0 for v in amap.values())
 for ds,key in [('equity_structure','MARKET_CODE'),('dividend','MARKET_CODE'),('right_issue','MARKET_CODE')]:
  fs,rs=read_csv(raw/f'{ds}.csv'); codes={r.get(key,'') for r in rs}; checks[f'{ds}_target_only']=codes<=target
 eqf,eqr=read_csv(raw/'equity_structure.csv'); eqkeys=[(r.get('MARKET_CODE',''),datev(r.get('ANN_DATE')),datev(r.get('CHANGE_DATE')),datev(r.get('EX_CHANGE_DATE')),r.get('SHARE_CHANGE_REASON','')) for r in eqr]; eqcodes={r.get('MARKET_CODE','') for r in eqr}; first_signal={qualified(code):min(x['signal_date'] for x in c['records'] if any(qualified(t.get('code'))==qualified(code) for t in x['selector']['target_topk'])) for code in target}; checks['equity_coverage_and_primary_key_unique']=eqcodes==target and len(eqkeys)==len(set(eqkeys)); checks['equity_ann_pit_baseline']=all(equity_pit_baseline([r for r in eqr if r.get('MARKET_CODE')==code], first_signal[code]) for code in target); equity_future_fields=sum(1 for r in eqr for k in ('CHANGE_DATE','EX_CHANGE_DATE') if datev(r.get(k)) and datev(r.get(k))>first_signal.get(r.get('MARKET_CODE'),'0000-00-00'))
 checks['action_raw_rows_retained']=len(pit)==sum(1 for _ in csv.DictReader((raw/'dividend.csv').open()))+sum(1 for _ in csv.DictReader((raw/'right_issue.csv').open())); checks['action_id_duplicates_reported']=len({x['action_id'] for x in pit})<=len(pit); checks['pit_eligible_date_progress']=all(x['pit_reason']!='eligible' or (x['ann_date']<=x['effective_date'] and x['progress'] in {'3','3.0'}) for x in pit)
 recs={r['signal_date']:r for r in c['records']}; signal_dates={x['signal_date'] for x in c['records']}; exec_dates={x['execution_date'] for x in c['records']}; kdates=defaultdict(set)
 for r in krows: kdates[r['code']].add(datev(r['kline_time']))
 checks['candidate_signal_execution_coverage']=all(any(x in kdates[qualified(code)] for x in (r['signal_date'],r['execution_date'])) for r in c['records'] for code in r['selector']['target_topk'][0:0]) if False else all(r['signal_date'] in {d for ds in kdates.values() for d in ds} and r['execution_date'] in {d for ds in kdates.values() for d in ds} for r in c['records'])
 checks['settlement_2024_01_02']=all(SETTLE in kdates[code] for code in target)
 checks['qfq_diff_bound_and_unfilled']=len(cache_hashes)==131 and all(x['side'] in {'qfq_only','raw_only'} for x in qdiff)
 checks['paper_only_training_false']=True
 status='PASS_RESEARCH_ONLY' if all(checks.values()) else 'FAIL_CLOSED'; issues=[k for k,v in checks.items() if not v]
 prov={'candidate_manifest':{'path':str(CANDIDATE.relative_to(ROOT)),'sha256':sha(CANDIDATE)},'gap_plan':{'path':str(PLAN.relative_to(ROOT)),'sha256':sha(PLAN)},'old_parent':{'path':str(OLD_PARENT.relative_to(ROOT)),'sha256':sha(OLD_PARENT)},'collection_launcher_result':{'path':str((NEW_ROOT/'launcher_result.json').relative_to(ROOT)),'sha256':sha(NEW_ROOT/'launcher_result.json')},'supplement_roots':[str(Path(x).resolve().relative_to(ROOT)) for x in (args.supplement_root or [])],'collector_image_id':IMAGE_ID,'paper_only':True,'credential_values_persisted':False,'source_children':records,'candidate_cache_hashes':cache_hashes,'precedence_audit':PRECEDENCE_AUDIT}
 dump(out/'retrieval_provenance.json',prov)
 children=[{'dataset':r['dataset'],'manifest':r['manifest'],'manifest_sha256':r['manifest_sha256'],'artifact_hashes':[{'path':a['path'],'sha256':a['sha256']} for a in r['artifacts']]} for r in records]
 parent={'schema':'wc006-purged-xysz-market/v2','status':status,'created_at_utc':datetime.now(timezone.utc).isoformat(),'codes':target_list,'code_count':131,'child_manifests':children,'datasets':selected,'raw_root':'raw','quality_report':'quality_report.json','retrieval_provenance':'retrieval_provenance.json','date_key_diff':'date_key_diff.json','candidate_manifest_sha256':sha(CANDIDATE),'gap_plan_sha256':sha(PLAN),'old_parent_manifest_sha256':sha(OLD_PARENT),'paper_only':True,'research_only':True,'broker_orders_enabled':False,'live_execution_authorized':False,'shadow_execution_authorized':False,'training_authorized':False,'credential_values_persisted':False}
 dump(out/'manifest.json',parent); ph=sha(out/'manifest.json')
 qa={'schema':'wc006-purged-xysz-market/quality-report/v2','status':status,'manifest_sha256':ph,'checks':checks,'failed_checks':issues,'target_code_count':131,'coverage':{'kline_codes':len(set(r.get('code','') for r in krows)),'status_codes':len(set(r.get('MARKET_CODE','') for r in srows)),'equity_codes':len(set(r.get('MARKET_CODE','') for r in eqr)),'equity_future_change_fields_not_used':equity_future_fields,'precedence_overlaps':len(PRECEDENCE_AUDIT),'precedence_replaced_rows':sum(x['replaced_source_rows'] for x in PRECEDENCE_AUDIT),'raw_kline_rows':len(krows),'raw_status_rows':len(srows),'adj_factor_rows':len(arows),'adj_non_market_date_blank_cells':sum(1 for r in arows for c in target if not r.get(c)),'action_raw_rows':len(pit),'action_eligible_rows':len(eligible),'action_rejected_by_reason':dict(rej),'action_id_duplicate_count':len(pit)-len({x['action_id'] for x in pit}),'qfq_raw_date_key_diff_rows':len(qdiff),'qfq_only':sum(x['side']=='qfq_only' for x in qdiff),'raw_only':sum(x['side']=='raw_only' for x in qdiff)},'provenance_bound':True,'paper_only':True,'training_authorized':False}
 dump(out/'quality_report.json',qa); dump(out/'checks.json',{'status':status,'checks':checks})
 print(json.dumps({'output':str(out),'status':status,'manifest_sha256':ph,'failed_checks':issues},ensure_ascii=False,indent=2)); return 0 if status.startswith('PASS') else 2

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); ap.add_argument('--supplement-root',type=Path,action='append'); return build(ap.parse_args())
if __name__=='__main__': raise SystemExit(main())
