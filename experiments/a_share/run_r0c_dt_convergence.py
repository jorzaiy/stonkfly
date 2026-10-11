"""Frozen R0c dt-convergence audit.

This runner is diagnostic-only.  It reuses the R0b fixed stimuli/seeds and
changes only the Python dt constructor argument; the native kernel source is
intentionally not modified.
"""

import argparse
import hashlib
import json
import math
import time
import gc
from pathlib import Path

import numpy as np

from stonkfly.config import Settings
from stonkfly.neural.common import annotations, digest
from stonkfly.neural.v51_controller import V51Controller

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "docs/20261007-r0c-dt-convergence-preregistered.md"
GRAPH = ROOT / "data/graph.npz"
HIST = ROOT / "runs/wc006_purged_fly_training/20261004T-balanced-reward-v2/brain-clean.npz"
R0B_JSON = ROOT / "runs/plasticity_diagnosis/R0b/20261007T-absorbing-state-v1/r0b_formal.json"
R0B_RAW = ROOT / "runs/plasticity_diagnosis/R0b/20261007T-absorbing-state-v1/r0b_formal_raw.npz"
KERNEL = ROOT / "stonkfly/neural/kernel.cpp"
PREREG_SHA256 = "a4d6baeedf7944aa1968d678d80a47c17dd4ecb50cb48d60aa927f8d65efb9f0"
GRAPH_SHA256 = "346b8af85a11af13b8324e18669812c1924569e7d1adcb4e6f45cc461a2c344b"
HIST_SHA256 = "a92f547e720e7d9830ae519f26c0bacdf4f76c59761fb2be4adb6f34cac2a225"
R0B_JSON_SHA256 = "5f40904f57098802d28733f1b7929d5de5a0a8254969cf9900a32dc43f71c38b"
R0B_RAW_SHA256 = "3f510d9fc840e502d5b9d612a74aea69ddaa25bf6256561ef45bdad9953fea85"
KERNEL_SHA256 = "aea2b97d2f9d78c57787f40f42a9d7dd777c3059d8c7f453e61c4d94398438d4"
KERNEL_BINARY_SHA256 = "9fa6e5cac39a5d865cf0ca50909cc3527c4ff697fd7c302f4d3c657dbfdde344"

DT_VALUES = (0.1, 0.05, 0.025)
BIN_MS = 10.0
STIMULUS_MS = 2000.0
OFF_MS = 1000.0
HALF_SATURATION = 0.02
LAMINA_BIAS = 12.0
STIMULI = ("uniform_50", "uniform_60", "uniform_70", "contrast_15", "contrast_30")
SEEDS = tuple(range(8))
GROUP_NAMES = ("retina", "lamina", "ol_sensory", "ol_intrinsic", "visual_projection", "cb_intrinsic", "KC", "APL", "MBON07", "MBON11")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_bytes(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def ticks(ms, dt):
    q = float(ms) / float(dt)
    if not math.isfinite(q) or q < 0 or not math.isclose(q, round(q), rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"duration {ms} is not exactly divisible by dt {dt}")
    return int(round(q))


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial.npz")
    with tmp.open("wb") as h:
        np.savez_compressed(h, **arrays)
    tmp.replace(path)


def manifest_digest(manifest):
    probe = dict(manifest); probe["raw_npz_sha256"] = None; probe["json_manifest_sha256"] = None
    return hashlib.sha256(json.dumps(probe, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def uniform_frame(level):
    return np.ascontiguousarray(np.full((64, 64, 3), int(level), dtype=np.uint8))


def contrast_frame(amplitude):
    yy, xx = np.indices((64, 64)); q = (yy // 8 + xx // 8) % 2
    v = np.where(q == 0, 60 - int(amplitude), 60 + int(amplitude)).astype(np.uint8)
    return np.ascontiguousarray(np.repeat(v[..., None], 3, axis=2))


def stimuli():
    out = {}
    for name, level in (("uniform_50", 50), ("uniform_60", 60), ("uniform_70", 70)):
        out[name] = {"family": "uniform", "level": level, "name": name, "frame": uniform_frame(level)}
    for name, level in (("contrast_15", 15), ("contrast_30", 30)):
        out[name] = {"family": "contrast", "level": level, "name": name, "frame": contrast_frame(level)}
    return out


def frame_descriptor(item):
    f = item["frame"]
    means = f.mean(axis=(0, 1)).astype(np.float64)
    return {"family": item["family"], "name": item["name"], "level": item["level"], "shape": list(f.shape), "dtype": str(f.dtype), "frame_sha256": sha_bytes(f), "channel_means_uint8": means.tolist(), "channel_mean": float(means.mean()), "pixel_min": int(f.min()), "pixel_max": int(f.max())}


def groups_for(brain):
    kc = np.asarray(brain.circuit["kc"], dtype=np.int64); mask = np.zeros(brain.n, dtype=bool); mask[kc] = True
    sc = np.asarray(brain.superclass); ann = annotations(brain.ids)
    return {
        "retina": np.asarray(brain.retina), "lamina": np.asarray(brain.lamina),
        "ol_sensory": np.flatnonzero((sc == "ol_sensory") & ~mask),
        "ol_intrinsic": np.flatnonzero((sc == "ol_intrinsic") & ~mask),
        "visual_projection": np.flatnonzero((sc == "visual_projection") & ~mask),
        "cb_intrinsic": np.flatnonzero((sc == "cb_intrinsic") & ~mask),
        "KC": kc, "APL": np.flatnonzero(ann.type.fillna("").astype(str).to_numpy() == "APL"),
        "MBON07": np.asarray(brain.circuit["reward_mb"]), "MBON11": np.asarray(brain.circuit["aversive_mb"]),
    }


def state_boundary_hash(brain):
    return hashlib.sha256(b";".join(n.encode() + b"=" + sha_bytes(getattr(brain, n)).encode() for n in ["weight", *brain.fields])).hexdigest()


def snapshot(brain):
    return {"weight": brain.weight.copy(), **{k: getattr(brain, k).copy() for k in brain.fields}}


def restore_snapshot(brain, snap):
    for k, value in snap.items():
        getattr(brain, k)[:] = value
    brain.cursor = 0; brain.sim_ms = 0.0; brain.total_spikes = 0


def hist(v):
    under = int((v < -80).sum()); over = int((v >= -44).sum()); inside = (v >= -80) & (v < -44)
    return {"bin_width_mV": 1.0, "range_min_mV": -80.0, "range_max_mV": -44.0, "num_bins": 36, "counts": np.bincount(np.floor(v[inside] + 80).astype(int), minlength=36).astype(int).tolist(), "underflow": under, "overflow": over, "total": int(len(v))}


def ignition_2s(t_ign):
    """R0b-compatible ignition field: only ignition before the 2 s boundary."""
    return t_ign if t_ign is not None and t_ign < 2000 else None


def trial(controller, clean, item, seed, groups):
    b = controller.brain; restore_snapshot(b, clean); delta = np.random.default_rng(seed).uniform(-.5, .5, size=b.n).astype(np.float32); b.v += delta
    total_bins = ticks(STIMULUS_MS + OFF_MS, BIN_MS); stim_bins = ticks(STIMULUS_MS, BIN_MS)
    gs = [groups[n] for n in GROUP_NAMES]; spikes = np.zeros((total_bins, len(gs)), np.int32); active = np.zeros_like(spikes); hasher = hashlib.sha256(); hist_2s = None; boundary_2s = None
    for bi in range(total_bins):
        frame = item["frame"] if bi < stim_bins else uniform_frame(0)
        c, _ = b.rgb_step(frame, BIN_MS, learning=False, retina_half_saturation=HALF_SATURATION, lamina_bias=LAMINA_BIAS)
        hasher.update(c.tobytes()); on = c > 0
        for gi, ix in enumerate(gs): spikes[bi, gi] = int(c[ix].sum()); active[bi, gi] = int(on[ix].sum())
        if bi == stim_bins - 1:
            hist_2s = {"cb": hist(b.v[groups["cb_intrinsic"]]), "kc": hist(b.v[groups["KC"]])}; boundary_2s = state_boundary_hash(b)
    kc_i = GROUP_NAMES.index("KC"); cb_i = GROUP_NAMES.index("cb_intrinsic")
    ign = np.flatnonzero(spikes[:, kc_i] >= 50); t_ign = int(ign[0] * BIN_MS) if len(ign) else None
    return ({"stimulus": frame_descriptor(item), "seed": int(seed), "delta_v_sha256": sha_bytes(delta), "dt_ms": b.dt, "delay_ticks": ticks(1.8, b.dt), "refractory_ticks": ticks(2.2, b.dt), "T_ign": t_ign, "T_ign_2s": ignition_2s(t_ign), "ignited_by_2s": bool(ignition_2s(t_ign) is not None), "pre_CB": int(spikes[190:200, cb_i].sum()), "pre_KC": int(spikes[190:200, kc_i].sum()), "post_CB_end": int(spikes[290:300, cb_i].sum()), "post_KC_end": int(spikes[290:300, kc_i].sum()), "post_100ms_cb": [int(spikes[200+w*10:210+w*10, cb_i].sum()) for w in range(10)], "post_100ms_kc": [int(spikes[200+w*10:210+w*10, kc_i].sum()) for w in range(10)], "all_bins_sha256": hasher.hexdigest(), "state_boundary_2s_sha256": boundary_2s, "state_boundary_sha256": state_boundary_hash(b), "weight_sha256_before": digest(clean["weight"]), "weight_sha256_after": digest(b.weight), "v_histogram_cb": hist(b.v[groups["cb_intrinsic"]]), "v_histogram_kc": hist(b.v[groups["KC"]]), "v_histogram_2s_cb": hist_2s["cb"], "v_histogram_2s_kc": hist_2s["kc"], "learning": False}, spikes, active)


def checkpoint_bridge(clean_path):
    out = {"all_persistent_arrays_bitwise_equal": True, "fields": []}
    with np.load(clean_path, allow_pickle=False) as a, np.load(HIST, allow_pickle=False) as old:
        for k in old.files:
            if k == "metadata": continue
            eq = k in a and a[k].shape == old[k].shape and a[k].dtype == old[k].dtype and np.array_equal(a[k], old[k])
            out["fields"].append({"field": k, "bitwise_equal": bool(eq)}); out["all_persistent_arrays_bitwise_equal"] &= bool(eq)
    return out


def finite_arrays(value):
    if isinstance(value, dict): return all(finite_arrays(v) for v in value.values())
    if isinstance(value, (list, tuple)): return all(finite_arrays(v) for v in value)
    if isinstance(value, np.ndarray) and value.dtype.kind in "fc": return bool(np.isfinite(value).all())
    if isinstance(value, (float, np.floating)): return bool(np.isfinite(value))
    return True


def verify_manifest(json_path, raw_path):
    m = json.loads(Path(json_path).read_text())
    with np.load(raw_path, allow_pickle=False) as z: pinned = str(np.asarray(z["json_manifest_sha256"]).item())
    return {"json_manifest_sha256": m["json_manifest_sha256"] == manifest_digest(m), "npz_manifest_sha256": pinned == m["json_manifest_sha256"], "raw_npz_sha256": file_sha(raw_path) == m["raw_npz_sha256"]}


def metrics(rows):
    out = {"by_dt": {}}
    for dt in DT_VALUES:
        rr = [r for r in rows if r["dt_ms"] == dt]; events = [r["T_ign"] for r in rr if r["ignited_by_2s"]]
        bins = []
        for k in range(20):
            at = sum(r["T_ign"] is None or r["T_ign"] >= k*100 for r in rr); ev = sum(r["T_ign"] is not None and k*100 <= r["T_ign"] < (k+1)*100 for r in rr); bins.append({"at_risk": at, "event": ev, "rate": ev/at if at else None})
        ign = [r for r in rr if r["ignited_by_2s"]]; persistent = [r for r in ign if r["post_CB_end"] >= .5*r["pre_CB"] and r["post_KC_end"] >= .5*r["pre_KC"]]
        out["by_dt"][str(dt)] = {"H_dt": len(events)/sum(x["at_risk"] for x in bins), "hazard_bins": bins, "ignition_fraction": len(ign)/40, "unignited_fraction": {str(t): sum(r["T_ign"] is None or r["T_ign"] >= t for r in rr)/40 for t in (500,1000,1500,2000)}, "persistent_both_count": len(persistent), "persistent_both_fraction": len(persistent)/len(ign) if ign else 0.0, "T_ign": [r["T_ign"] for r in rr]}
    return out


def run(output, smoke=False):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    expected = {PREREG: PREREG_SHA256, GRAPH: GRAPH_SHA256, HIST: HIST_SHA256, R0B_JSON: R0B_JSON_SHA256, R0B_RAW: R0B_RAW_SHA256, KERNEL: KERNEL_SHA256}
    for p, expected_sha in expected.items():
        if not p.exists(): raise RuntimeError(f"missing provenance: {p}")
        if file_sha(p) != expected_sha: raise RuntimeError(f"frozen SHA mismatch: {p} {file_sha(p)} != {expected_sha}")
    for ms in (BIN_MS, STIMULUS_MS, OFF_MS):
        for dt in DT_VALUES: ticks(ms, dt)
    settings = Settings(neural_ms=500.0, pulse_ms=200, learning=False)
    all_rows = []; all_spikes = {}; all_active = {}; clean_meta = {}; clean_snapshots = {}; bridges = {}; config_sigs = {}; execution_order = []
    stim = stimuli()
    for dt in DT_VALUES:
        controller = V51Controller(settings, diagnostic_dt_ms=dt); b = controller.brain
        config_sigs[str(dt)] = hashlib.sha256(json.dumps(b.configuration_signature(), sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()
        if b.build.get("binary_sha256") != KERNEL_BINARY_SHA256: raise RuntimeError("kernel binary SHA mismatch")
        if dt == .1 and b.dt != .1: raise RuntimeError("default/diagnostic dt=0.1 regression failed")
        if dt == .1:
            clean_path = output / "brain-clean-observer-build.npz"; controller.save(clean_path); bridges["0.1"] = checkpoint_bridge(clean_path)
            if not bridges["0.1"]["all_persistent_arrays_bitwise_equal"]: raise RuntimeError("0.1 fresh clean checkpoint failed historical bridge")
        clean = snapshot(b)
        if dt == .1: clean_snapshots["0.1"] = clean
        clean_meta[str(dt)] = {"queue_shape": list(b.queue.shape), "queue_zero": bool(not b.queue.any()), "queue_count_zero": bool(not b.queue_count.any()), "refractory_zero": bool(not b.refractory.any()), "clock": int(b.cursor), "sim_ms": float(b.sim_ms), "state_boundary_sha256": state_boundary_hash(b), "configuration_sha256": config_sigs[str(dt)]}
        groups = groups_for(b)
        if len(groups["KC"]) != 4064 or any(len(x) == 0 for x in groups.values()): raise RuntimeError("group integrity failure")
        plan = [(name, seed) for name in STIMULI for seed in SEEDS]
        if smoke: plan = [("uniform_50", 0)]
        for name, seed in plan:
            row, sp, ac = trial(controller, clean, stim[name], seed, groups); row["protocol"] = "main"; all_rows.append(row); execution_order.append([dt, "main", name, seed]); all_spikes.setdefault(str(dt), []).append(sp); all_active.setdefault(str(dt), []).append(ac)
        if not smoke:
            # One deterministic rerun per frozen stimulus; it is not a main trial.
            for name in ("uniform_50", "contrast_15"):
                main_index = STIMULI.index(name) * len(SEEDS)
                row, sp, ac = trial(controller, clean, stim[name], 0, groups)
                # Compare with the already-run main row for the same stimulus/seed.
                main_rows = [r for r in all_rows if r["dt_ms"] == dt and r.get("protocol", "main") == "main"]
                base_row = main_rows[main_index]
                base_sp = all_spikes[str(dt)][main_index]
                base_ac = all_active[str(dt)][main_index]
                if row["all_bins_sha256"] != base_row["all_bins_sha256"] or not np.array_equal(sp, base_sp) or not np.array_equal(ac, base_ac): raise RuntimeError(f"deterministic repeat failed dt={dt} {name}")
                row["protocol"] = "deterministic_repeat"; row["repeat_equal"] = True
                all_rows.append(row); execution_order.append([dt, "deterministic_repeat", name, 0]); all_spikes[str(dt)].append(sp); all_active[str(dt)].append(ac)
        # dt-specific clean-state white list: only queue shape/count metadata may differ.
        if dt != .1:
            base = clean_snapshots["0.1"]
            for key, value in clean.items():
                if key in ("queue", "queue_count"): continue
                if value.shape != base[key].shape or value.dtype != base[key].dtype or not np.array_equal(value, base[key]):
                    raise RuntimeError(f"dt-specific clean state differs outside queue whitelist: dt={dt}, field={key}")
        clean_meta[str(dt)]["non_queue_equal_to_0.1"] = bool(dt == .1 or all(np.array_equal(clean[k], clean_snapshots["0.1"][k]) for k in clean if k not in ("queue", "queue_count")))
        if not clean_meta[str(dt)]["non_queue_equal_to_0.1"]: raise RuntimeError(f"clean state whitelist failed for dt={dt}")
        del controller; gc.collect()
    if smoke: return {"smoke": True, "rows": len(all_rows), "dt": list(DT_VALUES), "clean_state": clean_meta}
    # R0b 0.1 gate: first 200 bins of the 40 main rows, in fixed order.
    with np.load(R0B_RAW, allow_pickle=False) as raw:
        old_spikes = raw["perturb_bin_spikes"]; old_active = raw["perturb_bin_actives"]
    old_json = json.loads(R0B_JSON.read_text()); old_records = [r for r in old_json["records"] if r["protocol"] == "perturbation_c"]
    base_rows = [r for r in all_rows if r["dt_ms"] == .1 and r.get("protocol", "main") != "deterministic_repeat"]
    base_sp = all_spikes["0.1"][:40]
    gate_rows = []
    if len(base_rows) != 40 or old_spikes.shape[0] != 40 or old_active.shape[0] != 40: raise RuntimeError("0.1 R0b reproduction row count failed")
    for i, (r, sp, ac, oldr) in enumerate(zip(base_rows, base_sp, all_active["0.1"][:40], old_records)):
        match = (np.array_equal(sp[:200], old_spikes[i]) and np.array_equal(ac[:200], old_active[i]) and r["T_ign_2s"] == oldr["T_ign"] and r["ignited_by_2s"] == oldr["ignited"] and r["v_histogram_2s_cb"] == oldr["v_histogram_cb"] and r["v_histogram_2s_kc"] == oldr["v_histogram_kc"] and r["state_boundary_2s_sha256"] == oldr["state_boundary_sha256"] and r["weight_sha256_before"] == oldr["weight_sha256_before"] and r["weight_sha256_after"] == oldr["weight_sha256_after"])
        gate_rows.append(bool(match))
    if not all(gate_rows): raise RuntimeError(f"0.1 R0b 2000ms reproduction failed rows={[i for i,x in enumerate(gate_rows) if not x]}")
    main_only = [r for r in all_rows if r.get("protocol", "main") == "main"]
    m = metrics(main_only)
    # I_common and frozen decision checks.
    by = {(r["dt_ms"], r["stimulus"]["name"], r["seed"]): r for r in all_rows if r.get("protocol", "main") != "deterministic_repeat"}
    common = [(s, seed) for s in STIMULI for seed in SEEDS if all(by[(dt,s,seed)]["ignited_by_2s"] for dt in (.05,.025))]
    for dt in (.05,.025):
        m["by_dt"][str(dt)]["I_common"] = len(common)
    m["I_common"] = [{"stimulus": s, "seed": seed} for s, seed in common]
    rate_stats = {}
    for dt in DT_VALUES:
        cb_pre = [by[(dt,s,seed)]["pre_CB"] / .1 for s, seed in common]
        kc_pre = [by[(dt,s,seed)]["pre_KC"] / .1 for s, seed in common]
        cb_post = [by[(dt,s,seed)]["post_CB_end"] / .1 for s, seed in common]
        kc_post = [by[(dt,s,seed)]["post_KC_end"] / .1 for s, seed in common]
        rate_stats[str(dt)] = {"cb_pre_median_hz": float(np.median(cb_pre)) if cb_pre else None, "kc_pre_median_hz": float(np.median(kc_pre)) if kc_pre else None, "cb_post_median_hz": float(np.median(cb_post)) if cb_post else None, "kc_post_median_hz": float(np.median(kc_post)) if kc_post else None, "cb_pre_rates_hz": cb_pre, "kc_pre_rates_hz": kc_pre, "cb_post_rates_hz": cb_post, "kc_post_rates_hz": kc_post}
    m["common_rate_stats"] = rate_stats
    def rel(a, b):
        if a == 0 and b == 0: return 0.0
        if a == 0 or b == 0: return 2.0
        return 2 * abs(a-b) / (abs(a)+abs(b))
    h0 = m["by_dt"]["0.1"]["H_dt"]
    if h0 == 0: verdict = "R0C_INCONCLUSIVE"
    else:
        rr = [m["by_dt"][str(dt)]["H_dt"]/h0 for dt in (.05,.025)]
        hdiff = 2*abs(m["by_dt"]["0.05"]["H_dt"]-m["by_dt"]["0.025"]["H_dt"])/(m["by_dt"]["0.05"]["H_dt"]+m["by_dt"]["0.025"]["H_dt"] or 1)
        pers = [m["by_dt"][str(dt)]["persistent_both_fraction"] for dt in DT_VALUES]
        rate_diffs = {k: rel(rate_stats["0.05"][k], rate_stats["0.025"][k]) for k in ("cb_pre_median_hz", "kc_pre_median_hz", "cb_post_median_hz", "kc_post_median_hz")}
        converged = all(.5 <= x <= 2 for x in rr) and hdiff <= .25 and min(pers) >= .9 and max(pers)-min(pers) <= .1 and bool(common) and all(x <= .2 for x in rate_diffs.values()) and all(m["by_dt"][str(dt)]["ignition_fraction"] > 0 and m["by_dt"][str(dt)]["persistent_both_count"] > 0 for dt in DT_VALUES)
        notconv = any(x < .5 or x > 2 for x in rr) or hdiff > .25 or any(m["by_dt"][str(dt)]["persistent_both_count"] == 0 for dt in DT_VALUES) or not common
        verdict = "DT_CONVERGED_FOR_CURRENT_AUDIT" if converged else "DT_NOT_CONVERGED" if notconv else "R0C_INCONCLUSIVE"
        m["comparison"] = {"risk_ratio_0.05": rr[0], "risk_ratio_0.025": rr[1], "H_fine_symdiff": hdiff, "rate_symdiff_0.05_vs_0.025": rate_diffs, "verdict": verdict}
    raw_path = output / "r0c_formal_raw.npz"; atomic_npz(raw_path, **{f"spikes_{dt}": np.asarray(all_spikes[str(dt)], np.int32) for dt in DT_VALUES}, **{f"active_{dt}": np.asarray(all_active[str(dt)], np.int32) for dt in DT_VALUES}, json_manifest_sha256=np.asarray("pending"))
    expected_order = []
    for dt in DT_VALUES:
        expected_order.extend([[dt, "main", name, seed] for name in STIMULI for seed in SEEDS])
        expected_order.extend([[dt, "deterministic_repeat", name, 0] for name in ("uniform_50", "contrast_15")])
    exact_gate = len(all_rows) == 126 and execution_order == expected_order and all(sum(r["dt_ms"] == dt and r.get("protocol") == "main" for r in all_rows) == 40 and sum(r["dt_ms"] == dt and r.get("protocol") == "deterministic_repeat" for r in all_rows) == 2 for dt in DT_VALUES)
    delta_by_key = {(r["stimulus"]["name"], r["seed"]): r["delta_v_sha256"] for r in all_rows if r.get("protocol") == "main"}
    delta_gate = all(len({next(r["delta_v_sha256"] for r in all_rows if r.get("protocol") == "main" and r["dt_ms"] == dt and r["stimulus"]["name"] == name and r["seed"] == seed) for dt in DT_VALUES}) == 1 for name in STIMULI for seed in SEEDS)
    finite_gate = finite_arrays(all_rows) and all(r["weight_sha256_before"] == r["weight_sha256_after"] for r in all_rows) and all(len(r["stimulus"]["shape"]) == 3 for r in all_rows)
    if not exact_gate: raise RuntimeError("formal trial count/order gate failed")
    if not delta_gate: raise RuntimeError("cross-dt delta-v SHA gate failed")
    if not finite_gate: raise RuntimeError("formal finite/weight/group gate failed")
    result = {"observer": "R0c", "formal": True, "success": True, "diagnostic_only": True, "counterfactual": True, "production_eligible": False, "learning": False, "dt_values_ms": list(DT_VALUES), "execution_order": execution_order, "trial_gate": {"exact_126": exact_gate, "per_dt_main_40_repeat_2": exact_gate, "delta_v_sha256_equal_across_dt": delta_gate}, "records": all_rows, "metrics": m, "clean_state": clean_meta, "checkpoint_bridges": bridges, "provenance": {"preregistration": str(PREREG), "preregistration_sha256": file_sha(PREREG), "graph_sha256": file_sha(GRAPH), "historical_checkpoint_sha256": file_sha(HIST), "r0b_json_sha256": file_sha(R0B_JSON), "r0b_raw_sha256": file_sha(R0B_RAW), "kernel_source_sha256": file_sha(KERNEL), "kernel_binary_sha256": KERNEL_BINARY_SHA256, "configuration_signatures": config_sigs, "runner_sha256": file_sha(Path(__file__)), "modified_sources_sha256": {str(p.relative_to(ROOT)): file_sha(p) for p in [ROOT/"stonkfly/neural/state.py", ROOT/"stonkfly/neural/brain.py", ROOT/"stonkfly/neural/controller.py", ROOT/"stonkfly/neural/v51_controller.py", Path(__file__)]}}, "raw_npz": str(raw_path), "raw_npz_sha256": None, "json_manifest_sha256": None}
    result["gates"] = {"r0b_2000ms_reproduction": all(gate_rows), "finite_arrays": finite_gate, "weights_frozen": True, "manifest_pin": True}
    result["json_manifest_sha256"] = manifest_digest(result); atomic_npz(raw_path, **{f"spikes_{dt}": np.asarray(all_spikes[str(dt)], np.int32) for dt in DT_VALUES}, **{f"active_{dt}": np.asarray(all_active[str(dt)], np.int32) for dt in DT_VALUES}, json_manifest_sha256=np.asarray(result["json_manifest_sha256"])); result["raw_npz_sha256"] = file_sha(raw_path); json_path = output / "r0c_formal.json"; atomic_json(json_path, result)
    pin = verify_manifest(json_path, raw_path)
    if not all(pin.values()):
        json_path.unlink(missing_ok=True); raw_path.unlink(missing_ok=True); raise RuntimeError(f"manifest pin failed: {pin}")
    atomic_json(json_path, result)
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    try:
        r = run(a.output, a.smoke); print(json.dumps({"success": True, "formal": not a.smoke, "rows": r.get("rows", len(r.get("records", []))), "output": str(a.output), "verdict": r.get("metrics", {}).get("comparison", {}).get("verdict")}))
    except Exception as exc:
        for name in ("r0c_formal.json", "r0c_formal_raw.npz"):
            p = Path(a.output) / name
            if p.exists(): p.unlink()
        failure = {"observer": "R0c", "formal": not a.smoke, "success": False, "diagnostic_only": True, "counterfactual": True, "production_eligible": False, "error_type": type(exc).__name__, "error_message": str(exc)}
        atomic_json(Path(a.output) / ("r0c_smoke_failure.json" if a.smoke else "r0c_formal_failure.json"), failure); raise


if __name__ == "__main__": main()
