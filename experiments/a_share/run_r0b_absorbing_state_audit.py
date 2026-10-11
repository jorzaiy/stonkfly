"""Frozen R0b first-passage / absorbing state audit runner.

Authoritative specification:
  docs/20261007-r0b-absorbing-state-audit-preregistered.md
  SHA256: de986a67b829e64c06656145d77b0256a7a3ea6df3980d4d15944affa7a1d373
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from stonkfly.config import Settings
from stonkfly.neural.common import annotations, digest
from stonkfly.neural.v51_controller import V51Controller

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "docs/20261007-r0b-absorbing-state-audit-preregistered.md"
GRAPH = ROOT / "data/graph.npz"
HIST = ROOT / "runs/wc006_purged_fly_training/20261004T-balanced-reward-v2/brain-clean.npz"
R0_JSON = ROOT / "runs/plasticity_diagnosis/R0/20261007T-synthetic-dose-response-v2/r0_formal.json"
R0_RAW = ROOT / "runs/plasticity_diagnosis/R0/20261007T-synthetic-dose-response-v2/r0_formal_raw.npz"

PREREG_SHA256 = "de986a67b829e64c06656145d77b0256a7a3ea6df3980d4d15944affa7a1d373"
GRAPH_SHA256 = "346b8af85a11af13b8324e18669812c1924569e7d1adcb4e6f45cc461a2c344b"
HIST_SHA256 = "a92f547e720e7d9830ae519f26c0bacdf4f76c59761fb2be4adb6f34cac2a225"
R0_JSON_SHA256 = "d5f294f5c84aebfafa22a3539315791a56827a76e907ebadcfd8743e809c4e21"
R0_RAW_SHA256 = "7f3a228f5e880368a4937b63c1bde6ff9ff7650b438cfdf7a276de929fb8253b"

HALF_SATURATION = 0.02
LAMINA_BIAS = 12.0
BIN_MS = 10.0

UNIFORM_LEVELS = (40, 42, 45, 48, 50, 52, 55, 58, 60, 62, 65, 68, 70, 72, 75, 78, 80)
CONTRAST_AMPLITUDES = (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60)
C_PERTURB_STIMULI = ("uniform_50", "uniform_60", "uniform_70", "contrast_15", "contrast_30")
C_PERTURB_SEEDS = tuple(range(8))
REPEAT_STIMULI = ("uniform_50", "contrast_15")

GROUP_NAMES = (
    "retina",
    "lamina",
    "ol_sensory",
    "ol_intrinsic",
    "visual_projection",
    "cb_intrinsic",
    "KC",
    "APL",
    "MBON07",
    "MBON11",
)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_bytes(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def state_boundary_hash(brain):
    parts = []
    for name in ["weight", *brain.fields]:
        parts.append(name.encode() + b"=" + sha_bytes(getattr(brain, name)).encode())
    return hashlib.sha256(b";".join(parts)).hexdigest()


def all_finite(value):
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial.npz")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


def manifest_digest(manifest):
    probe = dict(manifest)
    probe["raw_npz_sha256"] = None
    probe["json_manifest_sha256"] = None
    return hashlib.sha256(
        json.dumps(probe, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def verify_manifest_pin(json_path, npz_path):
    manifest = json.loads(Path(json_path).read_text())
    with np.load(npz_path, allow_pickle=False) as arrays:
        pinned = str(np.asarray(arrays["json_manifest_sha256"]).item())
    return {
        "json_manifest_sha256": manifest.get("json_manifest_sha256") == manifest_digest(manifest),
        "npz_manifest_sha256": pinned == manifest.get("json_manifest_sha256"),
        "raw_npz_sha256": file_sha(npz_path) == manifest.get("raw_npz_sha256"),
    }


def uniform_frame(level):
    return np.ascontiguousarray(np.full((64, 64, 3), int(level), dtype=np.uint8))


def contrast_frame(amplitude):
    yy, xx = np.indices((64, 64))
    q = (yy // 8 + xx // 8) % 2
    values = np.where(q == 0, 60 - int(amplitude), 60 + int(amplitude)).astype(np.uint8)
    return np.ascontiguousarray(np.repeat(values[..., None], 3, axis=2))


def build_stimuli():
    out = []
    for level in UNIFORM_LEVELS:
        out.append(
            {
                "family": "uniform",
                "level": int(level),
                "name": f"uniform_{level}",
                "frame": uniform_frame(level),
            }
        )
    for amplitude in CONTRAST_AMPLITUDES:
        out.append(
            {
                "family": "contrast",
                "level": int(amplitude),
                "name": f"contrast_{amplitude}",
                "frame": contrast_frame(amplitude),
            }
        )
    return out


def frame_descriptor(item):
    frame = item["frame"]
    channel_means = frame.mean(axis=(0, 1)).astype(np.float64)
    return {
        "family": item["family"],
        "name": item["name"],
        "level": item["level"],
        "shape": list(frame.shape),
        "dtype": str(frame.dtype),
        "contiguous": bool(frame.flags.c_contiguous),
        "frame_sha256": sha_bytes(frame),
        "channel_means_uint8": channel_means.tolist(),
        "channel_mean": float(channel_means.mean()),
        "pixel_min": int(frame.min()),
        "pixel_max": int(frame.max()),
    }


def checkpoint_bridge(fresh, historical):
    fields = []
    all_equal = True
    with np.load(fresh, allow_pickle=False) as current, np.load(historical, allow_pickle=False) as old:
        current_keys = [key for key in current.files if key != "metadata"]
        old_keys = [key for key in old.files if key != "metadata"]
        all_equal = current_keys == old_keys
        for key in old_keys:
            if key not in current:
                fields.append({"field": key, "bitwise_equal": False, "reason": "missing"})
                all_equal = False
                continue
            left, right = np.asarray(current[key]), np.asarray(old[key])
            equal = (
                left.shape == right.shape
                and left.dtype == right.dtype
                and np.array_equal(left, right, equal_nan=False)
            )
            fields.append({"field": key, "bitwise_equal": bool(equal)})
            all_equal = all_equal and bool(equal)
    return {"all_persistent_arrays_bitwise_equal": bool(all_equal), "fields": fields}


def compute_v_histogram(v):
    underflow = int((v < -80.0).sum())
    overflow = int((v >= -44.0).sum())
    in_range = (v >= -80.0) & (v < -44.0)
    bin_indices = np.floor(v[in_range] - (-80.0)).astype(int)
    counts = np.bincount(bin_indices, minlength=36).astype(int).tolist()
    return {
        "bin_width_mV": 1.0,
        "range_min_mV": -80.0,
        "range_max_mV": -44.0,
        "num_bins": 36,
        "counts": counts,
        "underflow": underflow,
        "overflow": overflow,
        "total": int(len(v)),
    }


def identify_groups(brain):
    kc_indices = np.asarray(brain.circuit["kc"], dtype=np.int64)
    kc_mask = np.zeros(brain.n, dtype=bool)
    kc_mask[kc_indices] = True

    retina_indices = np.asarray(brain.retina, dtype=np.int64)
    lamina_indices = np.asarray(brain.lamina, dtype=np.int64)

    sc = np.asarray(brain.superclass)
    ol_sensory = np.flatnonzero((sc == "ol_sensory") & ~kc_mask)
    ol_intrinsic = np.flatnonzero((sc == "ol_intrinsic") & ~kc_mask)
    visual_projection = np.flatnonzero((sc == "visual_projection") & ~kc_mask)
    cb_intrinsic = np.flatnonzero((sc == "cb_intrinsic") & ~kc_mask)

    ann = annotations(brain.ids)
    apl = np.flatnonzero(ann.type.fillna("").astype(str).to_numpy() == "APL")
    mbon07 = np.asarray(brain.circuit["reward_mb"], dtype=np.int64)
    mbon11 = np.asarray(brain.circuit["aversive_mb"], dtype=np.int64)

    groups = {
        "retina": retina_indices,
        "lamina": lamina_indices,
        "ol_sensory": ol_sensory,
        "ol_intrinsic": ol_intrinsic,
        "visual_projection": visual_projection,
        "cb_intrinsic": cb_intrinsic,
        "KC": kc_indices,
        "APL": apl,
        "MBON07": mbon07,
        "MBON11": mbon11,
    }
    return groups


def run_single_trial(
    controller,
    clean,
    stimulus_item,
    stimulus_ms,
    off_stimulus_ms,
    groups,
    perturb_seed=None,
    cum_target_ms=500.0,
):
    controller.restore(clean)
    b = controller.brain
    weight_before = digest(b.weight)

    if perturb_seed is not None:
        rng = np.random.default_rng(perturb_seed)
        delta_v = rng.uniform(-0.5, 0.5, size=b.n).astype(np.float32)
        b.v += delta_v

    stim_bins = round(stimulus_ms / BIN_MS)
    off_bins = round(off_stimulus_ms / BIN_MS)
    total_bins = stim_bins + off_bins
    cum_bins = min(stim_bins, round(cum_target_ms / BIN_MS))

    stim_frame = stimulus_item["frame"]
    off_frame = uniform_frame(0)

    group_list = [groups[name] for name in GROUP_NAMES]
    num_groups = len(GROUP_NAMES)

    bin_spike_counts = np.zeros((total_bins, num_groups), dtype=np.int32)
    bin_active_cells = np.zeros((total_bins, num_groups), dtype=np.int32)
    cum_counts = np.zeros(b.n, dtype=np.int32)

    hasher = hashlib.sha256()
    wall_total = 0.0

    v_hist_cb = None
    v_hist_kc = None

    cb_group_idx = GROUP_NAMES.index("cb_intrinsic")
    kc_group_idx = GROUP_NAMES.index("KC")
    cb_indices = groups["cb_intrinsic"]
    kc_indices = groups["KC"]

    for b_idx in range(total_bins):
        frame = stim_frame if b_idx < stim_bins else off_frame
        counts, wall = b.rgb_step(
            frame,
            BIN_MS,
            learning=False,
            retina_half_saturation=HALF_SATURATION,
            lamina_bias=LAMINA_BIAS,
        )
        wall_total += wall
        hasher.update(counts.tobytes())

        if b_idx < cum_bins:
            cum_counts += counts

        active = counts > 0
        for g_idx, g_cells in enumerate(group_list):
            bin_spike_counts[b_idx, g_idx] = int(counts[g_cells].sum())
            bin_active_cells[b_idx, g_idx] = int(active[g_cells].sum())

        if b_idx == stim_bins - 1:
            v_hist_cb = compute_v_histogram(b.v[cb_indices])
            v_hist_kc = compute_v_histogram(b.v[kc_indices])

    weight_after = digest(b.weight)
    if weight_after != weight_before:
        raise RuntimeError("Synaptic weights changed while learning=false")

    kc_active_cum = cum_counts[kc_indices] > 0
    kc_bitset = np.packbits(kc_active_cum.astype(np.uint8), bitorder="little")

    kc_spikes_per_bin = bin_spike_counts[:, kc_group_idx]
    ign_bins = np.flatnonzero(kc_spikes_per_bin >= 50)
    t_ign = int(ign_bins[0] * BIN_MS) if len(ign_bins) > 0 else None
    ignited = (t_ign is not None) and (t_ign < stimulus_ms)

    cb_spikes_per_bin = bin_spike_counts[:, cb_group_idx]

    # Pre/post CB windows
    pre_cb = None
    post_cb_300 = None
    post_cb_end = None
    off_100ms_cb = None
    off_100ms_kc = None

    if stimulus_ms == 2000.0 and off_stimulus_ms == 1000.0:
        pre_cb = int(cb_spikes_per_bin[190:200].sum())
        post_cb_300 = int(cb_spikes_per_bin[220:230].sum())
        post_cb_end = int(cb_spikes_per_bin[290:300].sum())

        off_100ms_cb = [
            int(cb_spikes_per_bin[200 + w * 10 : 200 + (w + 1) * 10].sum())
            for w in range(10)
        ]
        off_100ms_kc = [
            int(kc_spikes_per_bin[200 + w * 10 : 200 + (w + 1) * 10].sum())
            for w in range(10)
        ]
    elif perturb_seed is not None and stimulus_ms == 2000.0:
        pre_cb = int(cb_spikes_per_bin[190:200].sum())
    elif stimulus_ms == 300.0 and off_stimulus_ms == 100.0:
        # Smoke timing
        pre_cb = int(cb_spikes_per_bin[20:30].sum())
        post_cb_end = int(cb_spikes_per_bin[30:40].sum())

    record = {
        "stimulus": frame_descriptor(stimulus_item),
        "perturb_seed": perturb_seed,
        "stimulus_ms": float(stimulus_ms),
        "off_stimulus_ms": float(off_stimulus_ms),
        "total_ms": float(stimulus_ms + off_stimulus_ms),
        "total_bins": int(total_bins),
        "wall_seconds": float(wall_total),
        "T_ign": t_ign,
        "ignited": bool(ignited),
        "pre_CB": pre_cb,
        "post_CB_300": post_cb_300,
        "post_CB_end": post_cb_end,
        "off_100ms_cb": off_100ms_cb,
        "off_100ms_kc": off_100ms_kc,
        "v_histogram_cb": v_hist_cb,
        "v_histogram_kc": v_hist_kc,
        "weight_sha256_before": weight_before,
        "weight_sha256_after": weight_after,
        "state_boundary_sha256": state_boundary_hash(b),
        "all_bins_sha256": hasher.hexdigest(),
        "counterfactual": perturb_seed is not None,
        "learning": False,
    }

    raw_arrays = {
        "cum_counts": cum_counts,
        "kc_bitset": kc_bitset,
        "bin_spike_counts": bin_spike_counts,
        "bin_active_cells": bin_active_cells,
    }
    return record, raw_arrays


def evaluate_metrics_and_hypotheses(main_records, perturb_records, smoke=False):
    if smoke:
        # Scaled-down evaluation for smoke verification
        ignited_count = sum(r["ignited"] for r in main_records)
        return {
            "smoke": True,
            "main_sequence_ignited_count": int(ignited_count),
            "main_sequence_total": len(main_records),
            "perturbation_runs_total": len(perturb_records),
            "verdict": "smoke_check_passed",
        }

    total_main = len(main_records)
    ignited_main = [r for r in main_records if r["ignited"]]
    ignited_count = len(ignited_main)

    # Main sequence per-stimulus stats
    main_per_stimulus = []
    for r in main_records:
        pre = r["pre_CB"]
        p_end = r["post_CB_end"]
        p_300 = r["post_CB_300"]
        main_per_stimulus.append(
            {
                "name": r["stimulus"]["name"],
                "family": r["stimulus"]["family"],
                "level": r["stimulus"]["level"],
                "T_ign": r["T_ign"],
                "ignited": r["ignited"],
                "pre_CB": pre,
                "post_CB_300": p_300,
                "post_CB_end": p_end,
                "post_CB_end_ratio_to_pre": float(p_end / pre) if (pre and pre > 0) else None,
                "post_CB_300_ratio_to_pre": float(p_300 / pre) if (pre and pre > 0) else None,
            }
        )

    # 1. H_abs condition 1: >= 27/30 ignited
    h_abs_cond1 = ignited_count >= 27

    # 2. H_abs condition 2: of ignited stimuli, >= 90% have post_CB_end >= 0.5 * pre_CB
    if ignited_count > 0:
        cond2_pass_count = sum(r["post_CB_end"] >= 0.5 * r["pre_CB"] for r in ignited_main)
        h_abs_cond2_ratio = cond2_pass_count / ignited_count
        h_abs_cond2 = h_abs_cond2_ratio >= 0.90
    else:
        cond2_pass_count = 0
        h_abs_cond2_ratio = 0.0
        h_abs_cond2 = False

    # 3. H_abs condition 3:
    # Standard deviation of T_ign across 8 seeds for each of 5 stimuli in (c) (censored at 2000)
    # averaged across 5 stimuli >= 0.5 * std of T_ign across 30 main stimuli (censored at 2000)
    perturb_by_stim = {s: [] for s in C_PERTURB_STIMULI}
    for r in perturb_records:
        perturb_by_stim[r["stimulus"]["name"]].append(r)

    c_censored_t = {}
    std_c_ddof0 = {}
    std_c_ddof1 = {}
    perturb_stim_details = {}

    for s in C_PERTURB_STIMULI:
        runs = sorted(perturb_by_stim[s], key=lambda x: x["perturb_seed"])
        vals = [r["T_ign"] if r["ignited"] else 2000.0 for r in runs]
        c_censored_t[s] = vals
        std_c_ddof0[s] = float(np.std(vals, ddof=0))
        std_c_ddof1[s] = float(np.std(vals, ddof=1))
        perturb_stim_details[s] = [
            {
                "seed": r["perturb_seed"],
                "T_ign": r["T_ign"],
                "ignited": r["ignited"],
                "censored_T_ign": val,
            }
            for r, val in zip(runs, vals)
        ]

    mean_std_c_ddof0 = float(np.mean(list(std_c_ddof0.values())))
    mean_std_c_ddof1 = float(np.mean(list(std_c_ddof1.values())))

    main_censored_t = [r["T_ign"] if r["ignited"] else 2000.0 for r in main_records]
    std_main_ddof0 = float(np.std(main_censored_t, ddof=0))
    std_main_ddof1 = float(np.std(main_censored_t, ddof=1))

    std_ratio_ddof0 = float(mean_std_c_ddof0 / std_main_ddof0) if std_main_ddof0 > 0 else 0.0
    std_ratio_ddof1 = float(mean_std_c_ddof1 / std_main_ddof1) if std_main_ddof1 > 0 else 0.0

    # Section 5 specifies std ratio >= 0.5
    h_abs_cond3 = bool(mean_std_c_ddof0 >= 0.5 * std_main_ddof0)
    h_abs = bool(h_abs_cond1 and h_abs_cond2 and h_abs_cond3)

    # H_thr
    # 1. Monotonicity: uniform and contrast sorted by level each change state at most once
    uniform_runs = sorted(
        [r for r in main_records if r["stimulus"]["family"] == "uniform"],
        key=lambda r: r["stimulus"]["level"],
    )
    contrast_runs = sorted(
        [r for r in main_records if r["stimulus"]["family"] == "contrast"],
        key=lambda r: r["stimulus"]["level"],
    )

    u_ign = [r["ignited"] for r in uniform_runs]
    c_ign = [r["ignited"] for r in contrast_runs]

    u_transitions = sum(u_ign[i] != u_ign[i + 1] for i in range(len(u_ign) - 1))
    c_transitions = sum(c_ign[i] != c_ign[i + 1] for i in range(len(c_ign) - 1))
    h_thr_cond1 = (u_transitions <= 1) and (c_transitions <= 1)

    # 2. Perturbation seed consistency: for each of the 5 stimuli, all 8 seeds have identical ignition outcome
    c_seed_consistency = {}
    for s in C_PERTURB_STIMULI:
        seeds_ign = [r["ignited"] for r in perturb_by_stim[s]]
        c_seed_consistency[s] = len(set(seeds_ign)) == 1
    h_thr_cond2 = all(c_seed_consistency.values())
    h_thr = bool(h_thr_cond1 and h_thr_cond2)

    # H_stim: of ignited stimuli, >= 90% have post_CB_300 < 0.2 * pre_CB
    if ignited_count > 0:
        stim_pass_count = sum(r["post_CB_300"] < 0.2 * r["pre_CB"] for r in ignited_main)
        h_stim_ratio = stim_pass_count / ignited_count
        h_stim = bool(h_stim_ratio >= 0.90)
    else:
        stim_pass_count = 0
        h_stim_ratio = 0.0
        h_stim = False

    # Verdict
    if h_stim:
        verdict = "H_stim"
    elif h_abs and not h_thr:
        verdict = "H_abs"
    elif h_thr and not h_abs:
        verdict = "H_thr"
    else:
        verdict = "inconclusive"

    return {
        "definitions": {
            "ignited": "T_ign is not null and T_ign < 2000 ms",
            "pre_CB": "CB spike count in [1900, 2000) ms",
            "post_CB_300": "CB spike count in [2200, 2300) ms",
            "post_CB_end": "CB spike count in [2900, 3000) ms",
        },
        "main_sequence": {
            "total_stimuli": total_main,
            "ignited_count": ignited_count,
            "ignited_fraction": float(ignited_count / total_main) if total_main else 0.0,
            "per_stimulus": main_per_stimulus,
        },
        "perturbation_c": {
            "stimuli_tested": list(C_PERTURB_STIMULI),
            "seeds_tested": list(C_PERTURB_SEEDS),
            "per_stimulus_seed_runs": perturb_stim_details,
            "per_stimulus_std_T_ign_ddof0": std_c_ddof0,
            "per_stimulus_std_T_ign_ddof1": std_c_ddof1,
            "mean_perturbation_std_ddof0": mean_std_c_ddof0,
            "mean_perturbation_std_ddof1": mean_std_c_ddof1,
        },
        "main_sequence_std_ddof0": std_main_ddof0,
        "main_sequence_std_ddof1": std_main_ddof1,
        "h_abs": {
            "cond1_ignited_ge_27": bool(h_abs_cond1),
            "cond1_val": int(ignited_count),
            "cond2_post_cb_end_ratio_ge_0_5": bool(h_abs_cond2),
            "cond2_val_ratio": float(h_abs_cond2_ratio),
            "cond2_count": int(cond2_pass_count),
            "cond3_std_ratio_ge_0_5": bool(h_abs_cond3),
            "cond3_ratio_ddof0": float(std_ratio_ddof0),
            "cond3_ratio_ddof1": float(std_ratio_ddof1),
            "H_abs_boolean": bool(h_abs),
        },
        "h_thr": {
            "cond1_monotonicity": bool(h_thr_cond1),
            "uniform_state_transitions": int(u_transitions),
            "contrast_state_transitions": int(c_transitions),
            "cond2_perturb_seed_consistency": bool(h_thr_cond2),
            "per_stimulus_consistency": c_seed_consistency,
            "H_thr_boolean": bool(h_thr),
        },
        "h_stim": {
            "pass_count": int(stim_pass_count),
            "pass_ratio": float(h_stim_ratio),
            "H_stim_boolean": bool(h_stim),
        },
        "verdict": verdict,
    }


def run_experiment(output_dir, smoke=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify provenance
    provenance_files = [
        (PREREG, PREREG_SHA256, "preregistration"),
        (GRAPH, GRAPH_SHA256, "graph"),
        (HIST, HIST_SHA256, "historical_clean_checkpoint"),
        (R0_JSON, R0_JSON_SHA256, "r0_formal_json"),
        (R0_RAW, R0_RAW_SHA256, "r0_formal_raw"),
    ]
    for path, expected_sha, name in provenance_files:
        if not path.exists():
            raise RuntimeError(f"Missing provenance file: {path} ({name})")
        actual_sha = file_sha(path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Provenance SHA256 mismatch for {name}: {actual_sha} != {expected_sha}"
            )

    # 2. Build controller and observer fresh clean checkpoint
    controller = V51Controller(Settings(neural_ms=500.0, pulse_ms=200, learning=False))
    b = controller.brain
    clean = output_dir / "brain-clean-observer-build.npz"
    controller.save(clean)

    bridge = checkpoint_bridge(clean, HIST)
    if not bridge["all_persistent_arrays_bitwise_equal"]:
        raise RuntimeError(
            f"Fresh observer-build checkpoint failed bitwise bridge to historical checkpoint: {bridge}"
        )

    # 3. Identify anatomical cell groups
    groups = identify_groups(b)

    # Gate 4 checks: group nonempty, KC count == 4064
    if len(groups["KC"]) != 4064:
        raise RuntimeError(f"KC count mismatch: {len(groups['KC'])} != 4064")
    for g_name, g_idx in groups.items():
        if len(g_idx) == 0:
            raise RuntimeError(f"Cell group {g_name} is empty")

    all_stimuli = build_stimuli()
    stimuli_by_name = {x["name"]: x for x in all_stimuli}

    # Setup execution plans
    if smoke:
        stimulus_ms = 300.0
        off_stimulus_ms = 100.0
        cum_target_ms = 300.0

        main_plan = [stimuli_by_name["uniform_50"]]
        perturb_plan = [(stimuli_by_name["uniform_50"], 0)]
        repeat_plan = [stimuli_by_name["uniform_50"]]
    else:
        stimulus_ms = 2000.0
        off_stimulus_ms = 1000.0
        cum_target_ms = 500.0

        main_plan = all_stimuli
        perturb_plan = [
            (stimuli_by_name[s_name], seed)
            for s_name in C_PERTURB_STIMULI
            for seed in C_PERTURB_SEEDS
        ]
        repeat_plan = [stimuli_by_name[s_name] for s_name in REPEAT_STIMULI]

    # Reference run for smoke Gate 1
    smoke_ref_counts = None
    if smoke:
        controller.restore(clean)
        smoke_ref_counts = np.zeros(b.n, dtype=np.int32)
        ref_frame = uniform_frame(50)
        for _ in range(30):
            c_bin, _ = b.rgb_step(
                ref_frame,
                BIN_MS,
                learning=False,
                retina_half_saturation=HALF_SATURATION,
                lamina_bias=LAMINA_BIAS,
            )
            smoke_ref_counts += c_bin

    # Execute main sequence
    main_records = []
    main_raw_cum_counts = []
    main_raw_kc_bitsets = []
    main_bin_spikes = []
    main_bin_actives = []

    print(f"Starting main sequence ({len(main_plan)} runs)...", flush=True)
    for idx, item in enumerate(main_plan):
        t0 = time.time()
        rec, raw = run_single_trial(
            controller,
            clean,
            item,
            stimulus_ms,
            off_stimulus_ms,
            groups,
            perturb_seed=None,
            cum_target_ms=cum_target_ms,
        )
        rec["arm"] = "C"
        rec["protocol"] = "main_sequence"
        main_records.append(rec)
        main_raw_cum_counts.append(raw["cum_counts"])
        main_raw_kc_bitsets.append(raw["kc_bitset"])
        main_bin_spikes.append(raw["bin_spike_counts"])
        main_bin_actives.append(raw["bin_active_cells"])
        dt = time.time() - t0
        print(
            f"[{idx+1}/{len(main_plan)}] main: {item['name']} "
            f"(wall={dt:.1f}s, T_ign={rec['T_ign']}, ignited={rec['ignited']}, pre_CB={rec['pre_CB']})",
            flush=True,
        )

    # Gate 1 check
    gate1_passed = True
    gate1_details = []

    if smoke:
        # Check first 300 ms against identical repeated 30x10ms bins
        act_cum = main_raw_cum_counts[0]
        eq = np.array_equal(act_cum, smoke_ref_counts)
        gate1_passed = bool(eq)
        gate1_details.append(
            {"stimulus": "uniform_50", "match": bool(eq), "target": "smoke_reference_300ms"}
        )
        if not gate1_passed:
            raise RuntimeError("Smoke Gate 1 failed: first 300ms counts mismatch reference run")
        print("Smoke Gate 1 passed.", flush=True)
    else:
        # Formal Gate 1: Check against R0 C-arm independent repeat 1 raw counts
        print("Checking Gate 1 (R0 C independent repeat 1 match for 30 stimuli)...", flush=True)
        r0_json_data = json.loads(R0_JSON.read_text())
        r0_records = r0_json_data["records"]
        r0_raw_data = np.load(R0_RAW, allow_pickle=False)
        r0_counts = r0_raw_data["counts"]

        r0_row_map = {}
        for row_idx, r_item in enumerate(r0_records):
            if (
                r_item["arm"] == "C"
                and r_item["protocol"] == "independent"
                and r_item["repeat_index"] == 1
            ):
                r0_row_map[r_item["stimulus"]["name"]] = row_idx

        for idx, item in enumerate(main_plan):
            s_name = item["name"]
            if s_name not in r0_row_map:
                raise RuntimeError(f"Gate 1 error: Stimulus {s_name} not found in R0 reference map")
            r0_row = r0_row_map[s_name]
            expected = r0_counts[r0_row]
            actual = main_raw_cum_counts[idx]
            is_equal = np.array_equal(actual, expected)
            gate1_details.append(
                {
                    "stimulus": s_name,
                    "r0_row": int(r0_row),
                    "bitwise_equal": bool(is_equal),
                }
            )
            if not is_equal:
                gate1_passed = False

        if not gate1_passed:
            raise RuntimeError(
                f"Gate 1 failed closed: 500 ms cumulative counts mismatch R0 reference: {gate1_details}"
            )
        print("Gate 1 passed: all 30 stimuli bitwise match R0 C arm repeat 1.", flush=True)

    # Execute perturbation runs (c)
    perturb_records = []
    perturb_bin_spikes = []
    perturb_bin_actives = []

    print(f"Starting perturbation (c) ({len(perturb_plan)} runs)...", flush=True)
    for idx, (item, seed) in enumerate(perturb_plan):
        t0 = time.time()
        rec, raw = run_single_trial(
            controller,
            clean,
            item,
            stimulus_ms,
            0.0,  # No off-stimulus phase for perturbation runs
            groups,
            perturb_seed=seed,
            cum_target_ms=cum_target_ms,
        )
        rec["arm"] = "C"
        rec["protocol"] = "perturbation_c"
        perturb_records.append(rec)
        perturb_bin_spikes.append(raw["bin_spike_counts"])
        perturb_bin_actives.append(raw["bin_active_cells"])
        dt = time.time() - t0
        print(
            f"[{idx+1}/{len(perturb_plan)}] perturb: {item['name']} seed={seed} "
            f"(wall={dt:.1f}s, T_ign={rec['T_ign']}, ignited={rec['ignited']})",
            flush=True,
        )

    # Execute deterministic repeats
    repeat_records = []
    repeat_bin_spikes = []
    repeat_bin_actives = []
    gate2_details = []
    gate2_passed = True

    main_records_by_name = {r["stimulus"]["name"]: r for r in main_records}
    main_spikes_by_name = {
        item["name"]: spikes for item, spikes in zip(main_plan, main_bin_spikes)
    }
    main_actives_by_name = {
        item["name"]: actives for item, actives in zip(main_plan, main_bin_actives)
    }

    print(f"Starting deterministic repeats ({len(repeat_plan)} runs)...", flush=True)
    for idx, item in enumerate(repeat_plan):
        t0 = time.time()
        rec, raw = run_single_trial(
            controller,
            clean,
            item,
            stimulus_ms,
            off_stimulus_ms,
            groups,
            perturb_seed=None,
            cum_target_ms=cum_target_ms,
        )
        rec["arm"] = "C"
        rec["protocol"] = "deterministic_repeat"
        repeat_records.append(rec)
        repeat_bin_spikes.append(raw["bin_spike_counts"])
        repeat_bin_actives.append(raw["bin_active_cells"])

        s_name = item["name"]
        orig_rec = main_records_by_name[s_name]
        orig_spikes = main_spikes_by_name[s_name]
        orig_actives = main_actives_by_name[s_name]

        hash_equal = rec["all_bins_sha256"] == orig_rec["all_bins_sha256"]
        spikes_equal = np.array_equal(raw["bin_spike_counts"], orig_spikes)
        actives_equal = np.array_equal(raw["bin_active_cells"], orig_actives)
        state_equal = rec["state_boundary_sha256"] == orig_rec["state_boundary_sha256"]
        t_ign_equal = rec["T_ign"] == orig_rec["T_ign"]

        chk = {
            "stimulus": s_name,
            "all_bins_sha256_equal": bool(hash_equal),
            "group_spikes_equal": bool(spikes_equal),
            "group_actives_equal": bool(actives_equal),
            "state_boundary_equal": bool(state_equal),
            "T_ign_equal": bool(t_ign_equal),
            "bitwise_deterministic": bool(
                hash_equal and spikes_equal and actives_equal and state_equal and t_ign_equal
            ),
        }
        gate2_details.append(chk)
        if not chk["bitwise_deterministic"]:
            gate2_passed = False
        dt = time.time() - t0
        print(
            f"[{idx+1}/{len(repeat_plan)}] repeat: {s_name} (wall={dt:.1f}s, deterministic={chk['bitwise_deterministic']})",
            flush=True,
        )

    if not gate2_passed:
        raise RuntimeError(f"Gate 2 failed: deterministic repeat mismatch: {gate2_details}")
    print("Gate 2 passed: all repeats bitwise deterministic.", flush=True)

    # Combine all records
    all_records = main_records + perturb_records + repeat_records

    # Gate 3: Check weights invariant in all runs
    gate3_passed = all(
        r["weight_sha256_before"] == r["weight_sha256_after"] for r in all_records
    )
    if not gate3_passed:
        raise RuntimeError("Gate 3 failed: synaptic weights changed during simulation")

    # Gate 4: all finite
    gate4_passed = all_finite(all_records)
    if not gate4_passed:
        raise RuntimeError("Gate 4 failed: nonfinite values in records")

    # Evaluate metrics and hypotheses
    metrics = evaluate_metrics_and_hypotheses(main_records, perturb_records, smoke=smoke)

    # Prepare raw NPZ data
    raw_npz_path = output_dir / ("r0b_smoke_raw.npz" if smoke else "r0b_formal_raw.npz")

    v_hist_cb_counts = np.array(
        [r["v_histogram_cb"]["counts"] for r in all_records], dtype=np.int32
    )
    v_hist_cb_underflow = np.array(
        [r["v_histogram_cb"]["underflow"] for r in all_records], dtype=np.int32
    )
    v_hist_cb_overflow = np.array(
        [r["v_histogram_cb"]["overflow"] for r in all_records], dtype=np.int32
    )

    v_hist_kc_counts = np.array(
        [r["v_histogram_kc"]["counts"] for r in all_records], dtype=np.int32
    )
    v_hist_kc_underflow = np.array(
        [r["v_histogram_kc"]["underflow"] for r in all_records], dtype=np.int32
    )
    v_hist_kc_overflow = np.array(
        [r["v_histogram_kc"]["overflow"] for r in all_records], dtype=np.int32
    )

    atomic_npz(
        raw_npz_path,
        main_cum_counts=np.asarray(main_raw_cum_counts, dtype=np.int32),
        main_kc_bitsets=np.asarray(main_raw_kc_bitsets, dtype=np.uint8),
        main_bin_spikes=np.asarray(main_bin_spikes, dtype=np.int32),
        main_bin_actives=np.asarray(main_bin_actives, dtype=np.int32),
        perturb_bin_spikes=np.asarray(perturb_bin_spikes, dtype=np.int32),
        perturb_bin_actives=np.asarray(perturb_bin_actives, dtype=np.int32),
        repeat_bin_spikes=np.asarray(repeat_bin_spikes, dtype=np.int32),
        repeat_bin_actives=np.asarray(repeat_bin_actives, dtype=np.int32),
        v_hist_cb_counts=v_hist_cb_counts,
        v_hist_cb_underflow=v_hist_cb_underflow,
        v_hist_cb_overflow=v_hist_cb_overflow,
        v_hist_kc_counts=v_hist_kc_counts,
        v_hist_kc_underflow=v_hist_kc_underflow,
        v_hist_kc_overflow=v_hist_kc_overflow,
        json_manifest_sha256=np.asarray("pending"),
    )

    # Build manifest result
    result = {
        "observer": "R0b",
        "formal": not smoke,
        "success": True,
        "diagnostic_only": True,
        "counterfactual": True,
        "production_eligible": False,
        "preregistered": True,
        "learning": False,
        "timing": {
            "stimulus_ms": float(stimulus_ms),
            "off_stimulus_ms": float(off_stimulus_ms),
            "bin_ms": float(BIN_MS),
            "cum_target_ms": float(cum_target_ms),
        },
        "cell_groups": {name: int(len(groups[name])) for name in GROUP_NAMES},
        "gates": {
            "gate1_r0_match": {
                "passed": bool(gate1_passed),
                "details": gate1_details,
            },
            "gate2_determinism": {
                "passed": bool(gate2_passed),
                "details": gate2_details,
            },
            "gate3_weight_frozen": {
                "passed": bool(gate3_passed),
                "runs_checked": len(all_records),
            },
            "gate4_finite_and_groups": {
                "passed": bool(gate4_passed),
                "kc_count": int(len(groups["KC"])),
            },
            "gate5_manifest_pin": {
                "passed": True,  # Will be verified below
            },
            "all_gates_passed": bool(
                gate1_passed and gate2_passed and gate3_passed and gate4_passed
            ),
        },
        "metrics_and_hypotheses": metrics,
        "records": all_records,
        "total_runs": len(all_records),
        "provenance": {
            "preregistration": str(PREREG),
            "preregistration_sha256": file_sha(PREREG),
            "graph": str(GRAPH),
            "graph_sha256": file_sha(GRAPH),
            "historical_checkpoint": str(HIST),
            "historical_checkpoint_sha256": file_sha(HIST),
            "fresh_checkpoint": str(clean),
            "fresh_checkpoint_sha256": file_sha(clean),
            "r0_json": str(R0_JSON),
            "r0_json_sha256": file_sha(R0_JSON),
            "r0_raw": str(R0_RAW),
            "r0_raw_sha256": file_sha(R0_RAW),
            "kernel_source_sha256": b.build["source_sha256"],
            "kernel_binary_sha256": b.build["binary_sha256"],
            "script_sha256": file_sha(Path(__file__)),
            "configuration_sha256": hashlib.sha256(
                json.dumps(
                    b.configuration_signature(),
                    sort_keys=True,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        },
        "checkpoint_bridge": bridge,
        "raw_npz": str(raw_npz_path),
        "raw_npz_sha256": None,
        "json_manifest_sha256": None,
    }

    # Pin manifest
    result["json_manifest_sha256"] = manifest_digest(result)

    atomic_npz(
        raw_npz_path,
        main_cum_counts=np.asarray(main_raw_cum_counts, dtype=np.int32),
        main_kc_bitsets=np.asarray(main_raw_kc_bitsets, dtype=np.uint8),
        main_bin_spikes=np.asarray(main_bin_spikes, dtype=np.int32),
        main_bin_actives=np.asarray(main_bin_actives, dtype=np.int32),
        perturb_bin_spikes=np.asarray(perturb_bin_spikes, dtype=np.int32),
        perturb_bin_actives=np.asarray(perturb_bin_actives, dtype=np.int32),
        repeat_bin_spikes=np.asarray(repeat_bin_spikes, dtype=np.int32),
        repeat_bin_actives=np.asarray(repeat_bin_actives, dtype=np.int32),
        v_hist_cb_counts=v_hist_cb_counts,
        v_hist_cb_underflow=v_hist_cb_underflow,
        v_hist_cb_overflow=v_hist_cb_overflow,
        v_hist_kc_counts=v_hist_kc_counts,
        v_hist_kc_underflow=v_hist_kc_underflow,
        v_hist_kc_overflow=v_hist_kc_overflow,
        json_manifest_sha256=np.asarray(result["json_manifest_sha256"]),
    )

    result["raw_npz_sha256"] = file_sha(raw_npz_path)

    json_name = "r0b_smoke.json" if smoke else "r0b_formal.json"
    json_path = output_dir / json_name
    candidate = output_dir / (json_name + ".partial")

    atomic_json(candidate, result)

    pin_verify = verify_manifest_pin(candidate, raw_npz_path)
    if not all(pin_verify.values()):
        raise RuntimeError(f"Gate 5 failed: manifest pin check failed: {pin_verify}")

    candidate.replace(json_path)
    return result


def main():
    parser = argparse.ArgumentParser(description="R0b absorbing state audit runner")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test")
    args = parser.parse_args()

    try:
        res = run_experiment(args.output, smoke=args.smoke)
        print(
            json.dumps(
                {
                    "formal": res["formal"],
                    "success": res["success"],
                    "runs": res["total_runs"],
                    "output": str(args.output),
                    "verdict": res["metrics_and_hypotheses"].get("verdict"),
                }
            )
        )
    except Exception as exc:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        failure_name = "r0b_smoke_failure.json" if args.smoke else "r0b_formal_failure.json"
        failure = {
            "observer": "R0b",
            "formal": not args.smoke,
            "success": False,
            "diagnostic_only": True,
            "counterfactual": True,
            "production_eligible": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        atomic_json(output_dir / failure_name, failure)
        print(
            json.dumps(
                {
                    "formal": not args.smoke,
                    "success": False,
                    "output": str(args.output),
                    "error": str(exc),
                }
            ),
            file=sys.stderr,
        )
        raise


if __name__ == "__main__":
    main()
