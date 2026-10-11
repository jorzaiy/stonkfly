#!/usr/bin/env python3
"""Create deterministic summary metrics for a completed 2021 V5.1 paired replay."""
from __future__ import annotations
import hashlib, json, math, sqlite3
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/"runs/wc006_v5_1_2021_baseline/replay"
OUT=ROOT/"runs/wc006_v5_1_2021_baseline"

def sha(p):
 h=hashlib.sha256();
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
 return h.hexdigest()

def curve(returns, final_equity):
 eq=100000.0; vals=[eq]
 for r in returns: eq*=1.0+float(r); vals.append(eq)
 if final_equity is not None and vals[-1] and abs(final_equity-vals[-1])>1e-7:
  eq=float(final_equity); vals.append(eq)
 daily=[vals[i]/vals[i-1]-1.0 for i in range(1,len(vals))]
 mean=sum(daily)/len(daily) if daily else 0.0
 var=sum((x-mean)**2 for x in daily)/(len(daily)-1) if len(daily)>1 else 0.0
 peak=vals[0]; mdd=0.0
 for x in vals:
  peak=max(peak,x); mdd=min(mdd,x/peak-1.0)
 return {"initial_equity":100000.0,"final_equity":eq,"return":eq/100000.0-1.0,"daily_mean":mean,"sharpe_annualized":mean/math.sqrt(var)*math.sqrt(243) if var>0 else None,"max_drawdown":mdd,"daily_positive_fraction":sum(x>0 for x in daily)/len(daily) if daily else None,"daily_observations":len(daily)}

def main():
 events=[json.loads(x) for x in (RUN/"events.jsonl").read_text().splitlines() if x.strip()]
 if len(events)!=243: raise SystemExit(f"expected 243 events, got {len(events)}")
 db=sqlite3.connect(RUN/"ledger.sqlite"); bdb=sqlite3.connect(RUN/"paired_base.sqlite")
 meta=dict(db.execute("select key,value from meta")); bmeta=dict(bdb.execute("select key,value from meta"))
 final=json.loads(meta["final_settlement"]); bfinal={"base_equity": final["base_equity"]}
 fly_r=[float(e["reward"]["portfolio_return"]) for e in events]; base_r=[float(e["reward"]["base_return"]) for e in events]
 fly=curve(fly_r,float(final["equity"])); base=curve(base_r,float(bfinal["base_equity"]))
 fills=[f for e in events for f in (e["execution"]["settled_prior_observation_outcome"].get("fills") or [])]
 veto=[v for e in events for v in (e["execution"]["settled_prior_observation_outcome"].get("vetoes") or [])]
 fees=sum(float(e.get("fees") or 0) for e in events)+float(final.get("cost") or 0)
 turnover=sum(float(e.get("turnover") or 0) for e in events)+float(final.get("turnover") or 0)
 base_fills=bdb.execute("select count(*) from fills").fetchone()[0]
 base_veto=bdb.execute("select count(*) from intents where status='VETO'").fetchone()[0]
 base_fees=sum(float(r[0]) for r in bdb.execute("select fee from fills"))
 base_turnover=sum(float(r[0]) for r in bdb.execute("select notional from fills"))
 fly.update({"fees":fees,"turnover_notional":turnover,"turnover_on_initial_equity":turnover/100000.0,"trade_count_fills":len(fills),"veto_count":len(veto),"win_rate_daily":fly.pop("daily_positive_fraction"),"average_holding_period":"unavailable: lot schema has no acquisition date"})
 base.update({"fees":base_fees,"turnover_notional":base_turnover,"turnover_on_initial_equity":base_turnover/100000.0,"trade_count_fills":base_fills,"veto_count":base_veto,"win_rate_daily":base.pop("daily_positive_fraction"),"average_holding_period":"unavailable: lot schema has no acquisition date"})
 metrics={"schema":"wc006-v5.1-2021-metrics/v1","scope":["2021-01-04","2021-12-31"],"settlement_date":"2022-01-04","settlement_semantic":"settlement-only; no observation/reward","observations":len(events),"seed":17,"plasticity_mode":"frozen","sensory":"RGB-only NullSensoryAdapter","paper_only":True,"fly":fly,"direct_quant_base":base,"notes":["Sharpe uses 243 observation-session returns, annualized sqrt(243), and includes final settlement mark as an additional terminal point only for cumulative return/drawdown.","win_rate_daily is fraction of positive portfolio-return days; trade-level win rate is not inferred.","average holding period unavailable from current lots schema."]}
 (OUT/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
 comparison={"schema":"wc006-v5.1-2021-comparison/v1","fly_return":fly["return"],"direct_quant_base_return":base["return"],"fly_minus_base_return":fly["return"]-base["return"],"fly_final_equity":fly["final_equity"],"base_final_equity":base["final_equity"],"action_counts":{},"neural_events":len(events),"fill_count":len(fills),"veto_count":len(veto)}
 for e in events:
  a=e.get("decoder",{}).get("action");comparison["action_counts"][a]=comparison["action_counts"].get(a,0)+1
 (OUT/"comparison.json").write_text(json.dumps(comparison,indent=2)+"\n")
 # Close SQLite connections before final hashing. SQLite may create transient
 # -wal/-shm siblings while a connection is open; those are deliberately not
 # reproducibility artifacts and must never be pinned in final_manifest.json.
 for conn in (db, bdb):
  conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
  conn.commit()
  conn.close()
 stable_names={"brain-clean.npz","brain-day-0.npz","brain-day-1.npz","events.jsonl","forward_outcomes.jsonl","ledger.sqlite","observation_outcomes.jsonl","paired_base.sqlite","provenance.json","runtime_metrics.jsonl"}
 artifacts={name:sha(RUN/name) for name in sorted(stable_names) if (RUN/name).is_file()}
 artifacts.update({"metrics.json":sha(OUT/"metrics.json"),"comparison.json":sha(OUT/"comparison.json")})
 manifest={"schema":"wc006-v5.1-2021-replay-manifest/v1","run_dir":str(RUN.relative_to(ROOT)),"input_manifest":str((OUT/"inputs/candidate_stream/manifest.json").relative_to(ROOT)),"market_manifest":str((OUT/"inputs/market_inputs/manifest.json").relative_to(ROOT)),"artifacts":artifacts,"artifact_policy":{"stable_only":True,"sqlite_databases":["ledger.sqlite","paired_base.sqlite"],"wal_checkpoint":"PRAGMA wal_checkpoint(TRUNCATE) before hashing","connections_closed_before_hash":True,"transient_excluded":["*-wal","*-shm"]},"chunk_manifests":{p.name:sha(p) for p in sorted((OUT/"chunks").glob("chunk_2021_*.json"))},"logs":{p.name:sha(p) for p in sorted((OUT/"replay_logs").glob("*.log"))},"status":"PASS_COMPLETE_243_SESSIONS"}
 (OUT/"final_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
 print(json.dumps(metrics,indent=2))
if __name__=="__main__": main()
