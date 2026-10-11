"""Label-free, preregistered sensory calibration grid (no market/reward signal)."""
import argparse, hashlib, json, math, tempfile
from pathlib import Path
import numpy as np
from stonkfly.config import Settings
from stonkfly.neural.v51_controller import V51Controller
from experiments.a_share.run_s2i_visual_transfer_assay import frames

GRID = [(h, b) for h in (0.02, 0.05, 0.1, 0.2) for b in (6.0, 9.0, 12.0)]
BASELINE = (0.02, 12.0)
UNIFORM = tuple(f"uniform_{x}" for x in [0,1,2,5,10,20,40,80,160,255])
PATTERN_GROUPS = ("mean_0.25", "mean_0.5")

def digest_ids(ids):
    return hashlib.sha256(",".join(map(str, sorted(map(int, ids)))).encode()).hexdigest()

def metrics(records, kc_total):
    by = {r["name"]: r for r in records}
    uni = [by[n] for n in UNIFORM if n in by]
    distinct = len({r["kc_set_hash"] for r in uni})
    jumps = [abs(a["kc_count"] - b["kc_count"]) for a,b in zip(uni, uni[1:])]
    max_jump_norm = (max(jumps) / kc_total) if jumps else 0.0
    collisions = 0
    for group in PATTERN_GROUPS:
        p = [r for r in records if group in r["name"]]
        for i,a in enumerate(p):
            collisions += sum(b["kc_set_hash"] == a["kc_set_hash"] for b in p[i+1:])
    return {"uniform_distinct_kc_set_count": distinct,
            "uniform_max_abs_kc_count_jump_over_total_kc": max_jump_norm,
            "pattern_exact_kc_set_collision_pairs": collisions}

def select(config_metrics, baseline):
    acceptable = []
    for cfg, m in config_metrics.items():
        strict = (m["uniform_distinct_kc_set_count"] > baseline["uniform_distinct_kc_set_count"] or
                  m["uniform_max_abs_kc_count_jump_over_total_kc"] < baseline["uniform_max_abs_kc_count_jump_over_total_kc"] or
                  m["pattern_exact_kc_set_collision_pairs"] < baseline["pattern_exact_kc_set_collision_pairs"])
        no_worse = (m["uniform_distinct_kc_set_count"] >= baseline["uniform_distinct_kc_set_count"] and
                    m["uniform_max_abs_kc_count_jump_over_total_kc"] <= baseline["uniform_max_abs_kc_count_jump_over_total_kc"] and
                    m["pattern_exact_kc_set_collision_pairs"] <= baseline["pattern_exact_kc_set_collision_pairs"])
        if no_worse and strict: acceptable.append((cfg, m))
    if not acceptable: return {"status":"no_acceptable_calibration","selected":None}
    def key(item):
        (h,b),m=item
        dist=abs(h-BASELINE[0])/BASELINE[0]+abs(b-BASELINE[1])/BASELINE[1]
        return (-m["uniform_distinct_kc_set_count"],m["uniform_max_abs_kc_count_jump_over_total_kc"],m["pattern_exact_kc_set_collision_pairs"],dist,h,b)
    cfg,m=min(acceptable,key=key)
    return {"status":"selected_by_preregistered_lexicographic_rule","selected":{"retina_half_saturation":cfg[0],"lamina_bias":cfg[1],"metrics":m},"acceptable_count":len(acceptable)}

def run(output, selected_configs, selected_frames):
    fs = dict(frames())
    missing=set(selected_frames)-set(fs)
    if missing: raise ValueError(f"Unknown stimuli: {sorted(missing)}")
    controller=V51Controller(Settings(neural_ms=500,pulse_ms=200,learning=False)); brain=controller.brain
    kc=np.asarray(brain.circuit["kc"],dtype=np.int64); kc_total=len(kc)
    clean=Path(tempfile.gettempdir())/"s2i_calibration_clean.npz"; controller.save(clean)
    records=[]
    for hs,lb in selected_configs:
        if not math.isfinite(hs) or hs<=0 or not math.isfinite(lb): raise ValueError("invalid calibration grid")
        for name in selected_frames:
            frame=fs[name]
            if frame.ndim!=3 or frame.shape[2]!=3 or frame.dtype!=np.uint8 or not frame.flags.c_contiguous: raise ValueError("invalid RGB frame")
            controller.restore(clean)
            counts,wall=brain.rgb_step(frame,500.0,learning=False,retina_half_saturation=hs,lamina_bias=lb)
            active=np.flatnonzero(np.asarray(counts)[kc]>0)
            layers={}
            for s in np.unique(brain.superclass):
                ix=np.flatnonzero(np.asarray(brain.superclass)==s)
                layers[str(s)]={"spikes":int(np.asarray(counts)[ix].sum()),"active_cells":int(np.count_nonzero(np.asarray(counts)[ix]))}
            records.append({"retina_half_saturation":hs,"lamina_bias":lb,"name":name,"actual_mean":float(frame.mean()/255),"frame_sha256":hashlib.sha256(frame.tobytes()).hexdigest(),"kc_count":int(len(active)),"kc_set_hash":digest_ids(kc[active]),"spike_sha256":hashlib.sha256(np.asarray(counts,dtype=np.int32).tobytes()).hexdigest(),"layers":layers,"retina_luminance_mean":float(brain.luminance.mean()),"retinal_drive_mean":float(brain.drive[brain.retina].mean()),"wall_seconds":float(wall),"learning":False,"reinforcement":"none"})
    grouped={}
    for r in records: grouped.setdefault((r["retina_half_saturation"],r["lamina_bias"]),[]).append(r)
    summaries={f"{h:g}/{b:g}":metrics(v,kc_total) for (h,b),v in grouped.items()}
    baseline=summaries.get("0.02/12",metrics(grouped.get(BASELINE,[]),kc_total))
    result={"data_source":"current_rgb_to_retina_luminance_visual_path","preregistered_selection_rule":{"metrics":{"a":"uniform distinct KC set count maximize","b":"uniform adjacent max absolute KC-count jump / total KC minimize","c":"mean 0.25/0.5 exact KC-set collision pairs minimize"},"acceptable":"no metric worse than baseline and at least one strictly improves; tie normalized distance then parameters","grid":[{"retina_half_saturation":h,"lamina_bias":b} for h,b in GRID],"baseline":{"retina_half_saturation":BASELINE[0],"lamina_bias":BASELINE[1]}},"parameter_validation":"finite positive retina_half_saturation; finite lamina_bias; production defaults remain 0.02 and 12.0","stimulus_count":len(records),"records":records,"configuration_summaries":summaries,"baseline_metrics":baseline,"selection":select(summaries,baseline),"kc_total":kc_total,"limitations":["No labels, market data, reward, profit, or trading output; learning=False/reinforcement=none.","This is a label-free proxy calibration assay, not parameter validation of fly vision."]}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2,allow_nan=False)); return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--smoke",action="store_true"); a=ap.parse_args()
    configs=[BASELINE, GRID[0]] if a.smoke else GRID; names=list(UNIFORM[:2]) if a.smoke else [n for n,_ in frames()]
    r=run(a.output,configs,names); print(json.dumps({"records":r["stimulus_count"],"configs":len(configs),"path":str(a.output)}))
if __name__=='__main__': main()
