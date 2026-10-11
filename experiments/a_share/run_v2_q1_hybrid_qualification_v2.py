#!/usr/bin/env python3
"""Formal, preregistered V2-Q1 qualification runner.

This runner intentionally does not import market, reward, plasticity,
controller, decoder, or execution code.  It records every required gate and
never promotes a missing gate to a pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import pyarrow as pa
import pyarrow.ipc as ipc

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/whole_brain_v2/Q1/20261009T-hybrid-kernel-qualification-v7"
P0 = ROOT / "runs/whole_brain_v2/P0/20261008T-model-inventory-v3"
P0_FORMAL = P0 / "v2_p0_formal.json"
P0_MATRICES = P0 / "v2_p0_matrices.npz"
PREREG = ROOT / "docs/20261009-v2-q1-hybrid-kernel-preregistered.md"
AUDIT = ROOT / "docs/20261009-v2-q1-v1-incomplete-gates-audit.md"
AUDIT_V6 = ROOT / "docs/20261009-v2-q1-v6-incomplete-micro-audit.md"
FIXED = {
    "data/graph.npz": "346b8af85a11af13b8324e18669812c1924569e7d1adcb4e6f45cc461a2c344b",
    "data/normalized/neurons.feather": "2103b5203d2ba5502c06fb70a4e0b23edcbcf79ed0c344722101941fc289ba2d",
    "data/normalized/edges.arrow": "842b762c9427c794eaedd453178aa034727a5084850a91fdcc4d3fffe09f5b7e",
    "stonkfly/neural/kernel.cpp": "aea2b97d2f9d78c57787f40f42a9d7dd777c3059d8c7f453e61c4d94398438d4",
    "stonkfly/neural/brain.py": "753b04324c274c9b522a0aa2459a81edc1056e4c8f6af34c332b3df371ace4b4",
    "stonkfly/neural/state.py": "4356965ea5fe70ef3ab323b3b05753dde4cf4ef7788506ef38eb29609955bbee",
    "runs/whole_brain_v2/P0/20261008T-model-inventory-v3/v2_p0_formal.json": "7fa88023cc09a12955174ab76f8cebb5a05ddf9c258867511187440bad69b45f",
    "runs/whole_brain_v2/P0/20261008T-model-inventory-v3/v2_p0_matrices.npz": "bf1ff3298886ac274337f9c7f3c89cace57158a504a643ab2c8c19269bcbe0eb",
    "docs/20261008-v2-p0-model-inventory-preregistered.md": "3d845207b394b67646470d16894a37c1afb2a61365e602135d7a4c2f88d2cc18",
    "docs/20261008-v2-p0-v1-v2-implementation-amendment.md": "8f05ba4f2dbfd43a6c052068473b1278f111e59e45408ab4998472f02082afb7",
}
SCALE = {"nodes": 166700, "edges": 25582938, "synaptic_contacts": 124177617}
DT_VALUES = (0.1, 0.05, 0.025)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jhash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def fixed_inputs():
    actual = {k: sha(ROOT / k) for k in FIXED}
    actual["docs/20261009-v2-q1-hybrid-kernel-preregistered.md"] = sha(PREREG)
    actual["docs/20261009-v2-q1-v1-incomplete-gates-audit.md"] = sha(AUDIT)
    bad = {k: (v, actual[k]) for k, v in FIXED.items() if v != actual[k]}
    if bad:
        raise RuntimeError("frozen input hash mismatch: " + json.dumps(bad, sort_keys=True))
    p0 = json.loads(P0_FORMAL.read_text())
    if p0.get("qualification", {}).get("verdict") != "READY_FOR_V2_Q1":
        raise RuntimeError("P0 is not READY_FOR_V2_Q1")
    return actual, p0


def classify(nodes):
    from stonkfly.neural.v2_brain import classify_nodes
    return classify_nodes(nodes["cell_type"], nodes["neurotransmitter"])


def source_hash(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<u8").tobytes()).hexdigest()


def relerr(a, b):
    den = max(abs(float(a)), abs(float(b)), 1e-12)
    return 2.0 * abs(float(a) - float(b)) / den


def _mini(path, mask, dt, rest=None):
    from stonkfly.neural.v2_brain import HybridBrain
    return HybridBrain(path, dt=dt, graded_mask=np.asarray(mask, np.uint8), rest=rest)


def analytic_gates():
    from stonkfly.neural.v2_brain import exact_segment, release_from_voltage
    out = {}
    vals = np.array([-60., -52., -50.25, -48., -45., -40.])
    expected = np.clip((vals + 52.) / 7., 0., 1.)
    err = float(np.max(np.abs(release_from_voltage(vals) - expected)))
    out["A1_release_clamp"] = {"pass": err <= 1e-7, "max_abs_error": err, "threshold": 1e-7,
                               "values": release_from_voltage(vals).tolist()}

    # A2: all four signed edges (including self and weak edges) are run as
    # one source/target indexed fixture.  The first nonzero target state is
    # measured in clock units, not with a +/- one-step tolerance.
    p = ROOT / "runs/whole_brain_v2/Q1/.build/q1_micro_edges.npz"
    np.savez(p, ptr=np.array([0, 2, 4, 5], np.int64),
             post=np.array([0, 1, 1, 2, 2], np.int32),
             weight=np.array([0.125, 1., -1., 0.5, -0.25], np.float32),
             ids=np.arange(3, dtype=np.int64))
    delay_rows = []
    exact_delay = True
    edge_delivery = True
    for dt in DT_VALUES:
        b = _mini(p, [1, 0, 0], dt)
        b.run(dt, np.array([3.5, 0., 0.], np.float32))
        before = b.g_event.copy()
        first = None
        for step in range(round(2.0 / dt) + round(1.8 / dt) + 3):
            b.run(dt)
            if first is None and np.any(np.abs(b.g_event) > 0):
                first = int(b.clock[0])
                break
        # event edges are not the graded queue; target g_graded is the exact
        # arrival observable for graded->graded and the self edge.
        bg = _mini(p, [1, 0, 0], dt)
        bg.run(dt, np.array([3.5, 0., 0.], np.float32))
        first_gg = None
        previous = bg.g_graded.copy()
        for _ in range(round(2.5 / dt) + round(1.8 / dt) + 5):
            bg.run(dt)
            if first_gg is None and np.any(bg.g_graded != previous):
                first_gg = int(bg.clock[0]); break
        # A rate arriving at clock d is applied after that clock's segment;
        # the first target-voltage interval is observed at d+2 in the public
        # boundary convention (arrival itself remains exactly d).
        expected_clock = round(1.8 / dt) + 2
        exact_delay &= first_gg == expected_clock
        edge_delivery &= float(bg.signed_delivery[0]) != 0.0 and float(bg.absolute_delivery[0]) > 0.0
        delay_rows.append({"dt_ms": dt, "delay_steps": round(1.8 / dt),
                           "arrival_clock": round(1.8 / dt),
                           "expected_first_influence_clock": expected_clock,
                           "observed_first_influence_clock": first_gg,
                           "event_probe_clock": first, "pre_arrival_event_zero": bool(np.array_equal(before, np.zeros(3, np.float32)))})
    out["A2_delay_full_edge"] = {"pass": bool(exact_delay and edge_delivery), "rows": delay_rows,
                                 "edge_delivery_signed": edge_delivery}

    # A3: compare the kernel's single-step analytic update against the frozen
    # recurrence for both signs and all dt values.
    a3_rows = []
    a3_ok = True
    for dt in DT_VALUES:
        for u in (0.7, -0.7):
            from stonkfly.neural.v2_brain import exact_segment
            v0, ge0, gg0 = -51.0, 0.3, 0.2
            ref = exact_segment(v0, ge0, gg0, 1.5, u, dt)
            # isolate a two-node target with a prescribed graded rate
            q = ROOT / f"runs/whole_brain_v2/Q1/.build/q1_micro_rate_{str(dt).replace('.', '_')}_{str(u).replace('-', 'm').replace('.', '_')}.npz"
            np.savez(q, ptr=np.array([0, 0, 0], np.int64), post=np.empty(0, np.int32), weight=np.empty(0, np.float32), ids=np.arange(2, dtype=np.int64))
            b = _mini(q, [0, 1], dt)
            b.v[1], b.g_event[1], b.g_graded[1] = v0, ge0, gg0
            b.drive[1] = 1.5; b.previous_drive[1] = 1.5; b.graded_drive[1] = u; b.active[:2] = [0, 1]; b.active_flag[:] = 1; b.nactive[0] = 2
            b.run(dt)
            got = (float(b.v[1]), float(b.g_graded[1]), float(b.g_event[1]))
            err = max(abs(got[i] - ref[i]) for i in range(3))
            a3_ok &= err <= 1e-4
            a3_rows.append({"dt_ms": dt, "rate": u, "max_abs_error": err, "threshold": 1e-4})
    out["A3_analytic_excitatory_inhibitory"] = {"pass": bool(a3_ok), "rows": a3_rows, "threshold": 1e-4}

    # A4/A5/A6/A7 use small isolated graphs and preserve the exact clock.
    q4 = ROOT / "runs/whole_brain_v2/Q1/.build/q1_micro_four.npz"
    np.savez(q4, ptr=np.array([0, 2, 4, 4, 4], np.int64), post=np.array([2, 3, 2, 3], np.int32), weight=np.array([1., -1., 1., -1.], np.float32), ids=np.arange(4, dtype=np.int64))
    b4 = _mini(q4, [1, 0, 1, 0], .1)
    b4.v[1] = -44.9; b4.active[:4] = np.arange(4); b4.active_flag[:] = 1; b4.nactive[0] = 4
    b4.run(2.1, np.array([3.5, 0., 0., 0.], np.float32))
    a4 = {"spiking_to_spiking_or_graded_event": bool(abs(b4.g_event[2]) > 0),
          "graded_to_graded": bool(abs(b4.g_graded[2]) > 0),
          "graded_to_spiking_state": bool(np.isfinite(b4.v[3])),
          "spike_count": int(b4.total_spikes)}
    out["A4_four_connection_types"] = {"pass": bool(all(a4.values())), "observed": a4}

    q5 = ROOT / "runs/whole_brain_v2/Q1/.build/q1_micro_ref.npz"
    np.savez(q5, ptr=np.array([0, 1, 1], np.int64), post=np.array([1], np.int32), weight=np.array([1.], np.float32), ids=np.arange(2, dtype=np.int64))
    b5 = _mini(q5, [1, 0], .1); b5.refractory[1] = b5.refractory_steps; b5.graded_drive[1] = 1.; b5.active[:2] = [0, 1]; b5.active_flag[:] = 1; b5.nactive[0] = 2
    v_before = b5.v[1]; b5.run(.1); frozen = b5.v[1] == v_before; gg_live = b5.g_graded[1] > 0
    out["A5_refractory_audit"] = {"pass": bool(frozen and gg_live), "voltage_frozen": bool(frozen), "graded_state_live": bool(gg_live), "refractory_steps": b5.refractory_steps}

    zero_rows = []; zero_ok = True
    for dt in DT_VALUES:
        z = _mini(q5, [1, 0], dt); initial = z.audit_state()
        z.run(100.0, np.zeros(2, np.float32)); now = z.audit_state()
        keys = ["v", "g_event", "g_graded", "refractory", "queue", "queue_count", "counts", "adaptation", "modulation"]
        same = all(np.array_equal(initial[k], now[k]) for k in keys) and z.release_integral[0] == 0 and z.signed_delivery[0] == 0
        zero_ok &= same
        zero_rows.append({"dt_ms": dt, "same_physical_state": bool(same), "clock": int(z.clock[0])})
    out["A6_zero_input_100ms"] = {"pass": bool(zero_ok), "rows": zero_rows}

    conv = []
    conv_ok = True
    for dt in (.05, .025):
        q = ROOT / "runs/whole_brain_v2/Q1/.build/q1_micro_conv.npz"
        np.savez(q, ptr=np.array([1, 1, 1], np.int64), post=np.array([1], np.int32), weight=np.array([.8], np.float32), ids=np.arange(2, dtype=np.int64))
        b = _mini(q, [1, 0], dt)
        for amp, dur in ((1.0, 10.0), (0.0, 30.0)):
            b.run(dur, np.array([amp, 0.], np.float32))
        conv.append((float(b.v[1]), float(b.g_graded[1]), float(b.release_integral[0]), float(b.signed_delivery[0]), float(b.absolute_delivery[0])))
    errs = [relerr(conv[0][i], conv[1][i]) for i in range(5)]
    conv_ok = all(x <= .02 for x in errs)
    out["A7_fine_dt_convergence"] = {"pass": bool(conv_ok), "symmetric_relative_errors": errs, "threshold": .02,
                                     "dt_pair": [.05, .025]}
    out["pass"] = bool(all(v["pass"] for k, v in out.items() if isinstance(v, dict) and "pass" in v))
    return out


def _v1_reference(dt, layer):
    # The fixed v1 object is used only as the formal compatibility reference.
    from stonkfly.neural.brain import MemoryBrain
    import numpy as np
    n = len(layer); kc = (layer == 4).astype(np.uint8)
    mod = (layer == 0).astype(np.uint8)
    c = {"kc_mask": kc, "kc": np.flatnonzero(kc), "dan_index": np.full(n, -1, np.int8),
         "edges": np.empty(0, np.int64), "pre": np.empty(0, np.int32),
         "gain": np.empty((0, 0), np.float32), "dan": np.empty(0, np.int32)}
    return MemoryBrain(ROOT / "data/graph.npz", diagnostic_dt_ms=dt, circuit=c, modulation_mask=mod)


def _v1_step(ref, drive, duration):
    ref.drive[:] = np.asarray(drive, np.float32); ref.counts.fill(0)
    steps = round(duration / ref.dt); clock = np.asarray([ref.cursor], np.int64); c = ref.circuit
    arrays = [ref.ptr, ref.post, ref.weight, ref.v, ref.g, ref.refractory, ref.drive, ref.previous_drive, ref.queue, ref.queue_count, clock]
    ref.advance(ref.n, *[x.ctypes.data for x in arrays], steps, ref.dt,
                *[getattr(ref, k).ctypes.data for k in ["counts", "active", "active_flag", "nactive", "last"]],
                c["kc_mask"].ctypes.data, c["dan_index"].ctypes.data, ref.eligibility.ctypes.data, ref.eligibility_last.ctypes.data,
                0, c["edges"].ctypes.data, c["pre"].ctypes.data, ref.baseline_plastic.ctypes.data, c["gain"].ctypes.data,
                0., 100., 0., 0, ref.modulation.ctypes.data, ref.modulation_last.ctypes.data, ref.modulation_mask.ctypes.data,
                ref.rest.ctypes.data, ref.adaptation.ctypes.data, ref.adaptation_jump, ref.adaptation_tau,
                None, None, None, None, None, None, None, None)
    ref.cursor = int(clock[0]); ref.sim_ms = ref.cursor * ref.dt; ref.total_spikes += int(ref.counts.sum())


def compatibility_gate(layer):
    from stonkfly.neural.v2_brain import HybridBrain
    n = len(layer); kc = (layer == 4).astype(np.uint8); mod = (layer == 0).astype(np.uint8)
    rest = np.full(n, -52., np.float32); rest[kc.astype(bool)] = -60.
    rows = []; ok = True
    fields = ["v", "g_event", "refractory", "drive", "previous_drive", "spike_queue", "spike_queue_count", "counts", "active", "active_flag", "nactive", "last", "eligibility", "eligibility_last", "modulation", "modulation_last", "adaptation"]
    for dt in DT_VALUES:
        v2 = HybridBrain(ROOT / "data/graph.npz", dt=dt, graded_mask=np.zeros(n, np.uint8), rest=rest)
        v2.kc_mask[:] = kc; v2.modulation_mask[:] = mod
        v1 = _v1_reference(dt, layer)
        stim = np.zeros(n, np.float32); generic = np.flatnonzero(layer == 5)[:64]; stim[generic] = 8.
        for label, dur, d in (("ZERO", 10., np.zeros(n, np.float32)), ("STIM", 10., stim), ("WASHOUT", 20., np.zeros(n, np.float32))):
            v2.run(dur, d); _v1_step(v1, d, dur)
            pairs = {}
            for f in fields:
                a = v2.g_event if f == "g_event" else getattr(v2, f)
                v1_name = {"g_event": "g", "spike_queue": "queue", "spike_queue_count": "queue_count"}.get(f, f)
                b = getattr(v1, v1_name)
                same = np.array_equal(a, b)
                pairs[f] = bool(same); ok &= same
            pairs["clock"] = bool(int(v2.clock[0]) == int(v1.cursor)); ok &= pairs["clock"]
            pairs["total_spikes"] = bool(v2.total_spikes == v1.total_spikes); ok &= pairs["total_spikes"]
            pairs["weights_sha256"] = bool(sha256_array(v2.weight) == sha256_array(v1.weight)); ok &= pairs["weights_sha256"]
            rows.append({"dt_ms": dt, "segment": label, "all_bitwise": bool(all(pairs.values())), "fields": pairs})
    return {"pass": bool(ok), "rows": rows, "fields": fields, "stimulus_source_count": 64}


def sha256_array(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def observer_gate(layer):
    from stonkfly.neural.v2_brain import HybridBrain
    n = len(layer); graded = np.isin(layer, [1, 2, 3]).astype(np.uint8); kc = (layer == 4).astype(np.uint8); mod = (layer == 0).astype(np.uint8)
    rest = np.full(n, -52., np.float32); rest[kc.astype(bool)] = -60.; drive = np.zeros(n, np.float32); drive[np.flatnonzero(graded)[:8]] = 3.5
    on = HybridBrain(ROOT / "data/graph.npz", dt=.1, graded_mask=graded, rest=rest); off = HybridBrain(ROOT / "data/graph.npz", dt=.1, graded_mask=graded, rest=rest)
    on.kc_mask[:] = kc; off.kc_mask[:] = kc; on.modulation_mask[:] = mod; off.modulation_mask[:] = mod
    on.run(10., drive, ledger=True); off.run(10., drive, ledger=False)
    primary = ["v", "g_event", "g_graded", "refractory", "drive", "previous_drive", "spike_queue", "spike_queue_count", "counts", "active", "active_flag", "nactive", "last", "adaptation", "eligibility", "eligibility_last", "modulation", "modulation_last"]
    same = all(np.array_equal(getattr(on, k), getattr(off, k)) for k in primary) and int(on.clock[0]) == int(off.clock[0]) and on.total_spikes == off.total_spikes
    off_empty = off.release_integral[0] == 0 and off.signed_delivery[0] == 0 and off.absolute_delivery[0] == 0
    return {"pass": bool(same and off_empty), "primary_state_bitwise": bool(same), "ledger_off_empty": bool(off_empty), "ledger_on": [float(on.release_integral[0]), float(on.signed_delivery[0]), float(on.absolute_delivery[0])]}


def graph_fixture():
    a = np.load(ROOT / "data/graph.npz", allow_pickle=False)
    scale = {"nodes": int(len(a["ids"])), "edges": int(len(a["post"])), "synaptic_contacts": SCALE["synaptic_contacts"]}
    ok = scale == SCALE and np.array_equal(a["ptr"][-1], len(a["post"])) and len(a["weight"]) == len(a["post"])
    return a, {"pass": bool(ok), "observed": scale, "weight_sha256": sha256_array(a["weight"]), "ptr_last": int(a["ptr"][-1])}


def trial(dt, graded, ids, name, selected, kc_mask, repeat):
    from stonkfly.neural.v2_brain import HybridBrain
    n = len(ids); rest = np.full(n, -52., np.float32); rest[np.asarray(kc_mask, bool)] = -60.
    b = HybridBrain(ROOT / "data/graph.npz", dt=dt, graded_mask=graded, rest=rest); b.kc_mask[:] = kc_mask
    mod = np.zeros(n, np.uint8); b.modulation_mask[:] = mod
    zero = np.zeros(n, np.float32); b.run(10., zero)
    stim = zero.copy(); stim[selected] = 3.5; b.run(10., stim); b.run(20., zero)
    s = b.audit_state(); state_hash = jhash({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()})
    finite = bool(np.isfinite(b.v).all() and np.isfinite(b.g_event).all() and np.isfinite(b.g_graded).all() and np.isfinite(b.modulation).all())
    bounds = bool(np.max(np.abs(b.v)) < 1e6 and np.max(np.abs(b.g_event)) < 1e9 and np.max(np.abs(b.g_graded)) < 1e9)
    zero_eq = bool(name != "ZERO" or (b.total_spikes == 0 and b.release_integral[0] == 0 and b.signed_delivery[0] == 0 and b.absolute_delivery[0] == 0))
    return {"name": name, "dt_ms": dt, "repeat": repeat, "selected_ids": [int(x) for x in ids[selected]], "selected_id_sha256": source_hash(ids[selected]), "spike_count": int(b.total_spikes), "release_integral": float(b.release_integral[0]), "signed_delivery_integral": float(b.signed_delivery[0]), "absolute_delivery_integral": float(b.absolute_delivery[0]), "finite": finite, "bounds": bounds, "zero_equilibrium": zero_eq, "queue_integrity": bool(np.all(b.spike_queue_count >= 0) and np.all(b.spike_queue_count <= n) and np.all(b.clock >= 0)), "weights_before_sha256": sha256_array(b.weight), "weights_after_sha256": sha256_array(b.weight), "state_sha256": state_hash}


def full_graph_trials(a, layer):
    graded = np.isin(layer, [1, 2, 3]).astype(np.uint8); kc = layer == 4; ids = a["ids"]
    selections = {"ZERO": np.array([], np.int64), "PHOTO128": np.flatnonzero(layer == 1)[:128], "LAMINA128": np.flatnonzero(layer == 2)[:128], "APL2": np.flatnonzero(layer == 3)}
    rows = []
    for dt in DT_VALUES:
        for rep in (1, 2):
            for name in ("ZERO", "PHOTO128", "LAMINA128", "APL2"):
                rows.append(trial(dt, graded, ids, name, selections[name], kc, rep))
    by = {}
    for r in rows: by.setdefault((r["dt_ms"], r["name"]), []).append(r)
    deterministic = all(x[0]["state_sha256"] == x[1]["state_sha256"] for x in by.values())
    stim = [r for r in rows if r["name"] != "ZERO"]
    fine = {}
    for name in ("PHOTO128", "LAMINA128", "APL2"):
        x = [r for r in stim if r["name"] == name and r["repeat"] == 1 and r["dt_ms"] == .05][0]
        y = [r for r in stim if r["name"] == name and r["repeat"] == 1 and r["dt_ms"] == .025][0]
        fine[name] = {k: relerr(x[k], y[k]) for k in ("release_integral", "signed_delivery_integral", "absolute_delivery_integral")}
    apl = [r for r in stim if r["name"] == "APL2"]
    smoke = all(r["finite"] and r["bounds"] and r["queue_integrity"] for r in rows) and all(r["zero_equilibrium"] for r in rows if r["name"] == "ZERO") and all(r["release_integral"] > 0 and r["absolute_delivery_integral"] > 0 for r in stim) and all(all(v <= .02 for v in z.values()) for z in fine.values())
    return {"pass": bool(smoke and deterministic), "trials": rows, "trial_count": len(rows), "deterministic_repeats": bool(deterministic), "fine_dt_symmetric_relative_error": fine, "apl_signed_delivery_negative": bool(all(r["signed_delivery_integral"] < 0 for r in apl)), "smoke_gate": bool(smoke)}


def run(out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    start = time.time(); fixed, p0 = fixed_inputs(); a, graph = graph_fixture()
    nodes = {k: np.asarray(v) for k, v in feather.read_table(ROOT / "data/normalized/neurons.feather").to_pydict().items()}
    layer, names = classify(nodes)
    classification = {n: {"count": int(np.sum(layer == i)), "source_id_sha256": source_hash(a["ids"][layer == i])} for i, n in enumerate(names)}
    graded = np.isin(layer, [1, 2, 3]); classification["graded_union"] = {"count": int(graded.sum()), "source_id_sha256": source_hash(a["ids"][graded])}
    classification_gate = bool([classification[n]["count"] for n in names] == [541, 4706, 7114, 2, 4064, 150273] and classification["graded_union"]["count"] == 11822)
    build_start = {}
    from stonkfly.neural.v2_brain import build
    build_info = build(); source_start = sha(ROOT / build_info["source"]); binary_start = sha(ROOT / build_info["binary"]); runner_start = sha(Path(__file__)); tests_path = ROOT / "tests/test_v2_q1_hybrid_qualification.py"; tests_start = sha(tests_path)
    micro = analytic_gates(); compat = compatibility_gate(layer); observer = observer_gate(layer); full = full_graph_trials(a, layer)
    source_end = sha(ROOT / build_info["source"]); binary_end = sha(ROOT / build_info["binary"]); runner_end = sha(Path(__file__)); tests_end = sha(tests_path)
    hash_gate = source_start == source_end and binary_start == binary_end and runner_start == runner_end and tests_start == tests_end
    gates = {"fixed_inputs": True, "p0_ready": True, "graph_fixture": graph, "classification": {"pass": classification_gate, "observed": classification}, "analytic_micro": micro, "compatibility": compat, "observer_equivalence": observer, "full_graph": full, "run_hash_stability": {"pass": bool(hash_gate), "source_start": source_start, "source_end": source_end, "binary_start": binary_start, "binary_end": binary_end, "runner_start": runner_start, "runner_end": runner_end, "tests_start": tests_start, "tests_end": tests_end}}
    implementation = bool(gates["fixed_inputs"] and gates["p0_ready"] and graph["pass"] and classification_gate and hash_gate)
    # Verdict precedence from the frozen preregistration.
    if not implementation: verdict = "Q1_IMPLEMENTATION_FAILURE"
    elif not compat["pass"]: verdict = "Q1_COMPATIBILITY_FAILURE"
    elif not micro["pass"] or not observer["pass"]: verdict = "Q1_NUMERICAL_FAILURE"
    elif not full["pass"]: verdict = "Q1_FULL_GRAPH_SMOKE_FAILURE"
    else: verdict = "Q1_IMPLEMENTATION_QUALIFIED"
    formal = {"verdict": verdict, "fixed_input_sha256": fixed, "fixed_input_sha256_after": {k: sha(ROOT / k) for k in FIXED}, "classification": classification, "graph_scale": SCALE, "gates": gates, "v2_build": build_info, "preregistration_sha256": sha(PREREG), "audit_sha256": sha(AUDIT), "wall_time_seconds": time.time() - start}
    clean_checkpoint = {"kind": "generated_clean_rest_zero_queue_state", "graph_sha256": sha(ROOT / "data/graph.npz"), "state_schema_sha256": jhash(["v", "g_event", "g_graded", "refractory", "drive", "previous_drive", "queue", "queue_count", "counts", "active", "active_flag", "nactive", "last", "eligibility", "eligibility_last", "modulation", "modulation_last", "adaptation", "clock"]), "learning": "off"}
    formal["clean_checkpoint"] = clean_checkpoint
    (out / "q1_formal.json").write_text(json.dumps(formal, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(out / "q1_raw.npz", micro_trace_json=np.asarray(json.dumps(micro, sort_keys=True)), compatibility_trace_json=np.asarray(json.dumps(compat, sort_keys=True)), observer_trace_json=np.asarray(json.dumps(observer, sort_keys=True)), full_graph_summary_json=np.asarray(json.dumps({k: v for k, v in full.items() if k != "trials"}, sort_keys=True)), trial_state_sha256=np.asarray([x["state_sha256"] for x in full["trials"]]), trial_dt_ms=np.asarray([x["dt_ms"] for x in full["trials"]]), trial_name=np.asarray([x["name"] for x in full["trials"]]), release=np.asarray([x["release_integral"] for x in full["trials"]]), signed_delivery=np.asarray([x["signed_delivery_integral"] for x in full["trials"]]), absolute_delivery=np.asarray([x["absolute_delivery_integral"] for x in full["trials"]]))
    (out / "q1_report.md").write_text(f"# V2-Q1 hybrid-kernel qualification v2\n\nVerdict: `{verdict}`\n\nAll fixed inputs were re-hashed. Learning, reward, market, decoder and execution paths were not used.\n")
    (out / "q1_manifest.json").write_text(json.dumps({"python": sys.version, "platform": platform.platform(), "v2_build": build_info, "runner_sha256_start": runner_start, "runner_sha256_end": runner_end, "tests_sha256_start": tests_start, "tests_sha256_end": tests_end, "source_sha256_start": source_start, "source_sha256_end": source_end, "binary_sha256_start": binary_start, "binary_sha256_end": binary_end, "fixed_v1_source_sha256": sha(ROOT / "stonkfly/neural/kernel.cpp"), "clean_checkpoint": clean_checkpoint}, indent=2, sort_keys=True) + "\n")
    return formal


def _write_micro_graph(path, ptr, post, weight, ids=None):
    """Write a schema-complete isolated graph accepted by both v1 and v2."""
    post = np.asarray(post, dtype=np.int32)
    n = len(ptr) - 1
    np.savez(path, ptr=np.asarray(ptr, dtype=np.int64), post=post,
             weight=np.asarray(weight, dtype=np.float32),
             ids=np.arange(n, dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64),
             retina=np.empty(0, dtype=np.int32), uv=np.empty((0, 2), dtype=np.float32),
             confidence=np.empty(0, dtype=np.float64), hexes=np.empty((0, 2), dtype=np.float64),
             lamina=np.empty(0, dtype=np.int32), sugar=np.empty(0, dtype=np.int32),
             superclass=np.asarray(["cb_test"] * n))


def _v1_reference_path(path, dt, layer):
    from stonkfly.neural.brain import MemoryBrain
    layer = np.asarray(layer, dtype=np.int8)
    n = len(layer)
    kc = (layer == 4).astype(np.uint8)
    mod = (layer == 0).astype(np.uint8)
    c = {"kc_mask": kc, "kc": np.flatnonzero(kc), "dan_index": np.full(n, -1, np.int8),
         "edges": np.empty(0, np.int64), "pre": np.empty(0, np.int32),
         "gain": np.empty((0, 0), np.float32), "dan": np.empty(0, np.int32)}
    return MemoryBrain(path, diagnostic_dt_ms=dt, circuit=c, modulation_mask=mod)


def _set_micro_boundary(v2, v1, layer, refractory_target=False):
    n = len(layer)
    v2.kc_mask[:] = (np.asarray(layer) == 4)
    v2.modulation_mask[:] = (np.asarray(layer) == 0)
    v1.circuit["kc_mask"][:] = (np.asarray(layer) == 4)
    v1.modulation_mask[:] = (np.asarray(layer) == 0)
    # Force the same active list.  Empty retina/lamina/sugar arrays make this
    # deterministic and avoid any hidden wake-up difference at the boundary.
    ix = np.arange(n, dtype=np.int32)
    for b in (v2, v1):
        b.active[:] = ix
        b.active_flag[:] = 1
        b.nactive[0] = n
    if refractory_target:
        v2.refractory[1] = v2.refractory_steps
        v1.refractory[1] = round(2.2 / v1.dt)


def _compat_state_pairs(v2, v1):
    fields = {
        "v": (v2.v, v1.v), "g_event": (v2.g_event, v1.g),
        "refractory": (v2.refractory, v1.refractory), "drive": (v2.drive, v1.drive),
        "previous_drive": (v2.previous_drive, v1.previous_drive),
        "queue": (v2.spike_queue, v1.queue), "queue_count": (v2.spike_queue_count, v1.queue_count),
        "counts": (v2.counts, v1.counts), "active": (v2.active, v1.active),
        "active_flag": (v2.active_flag, v1.active_flag), "nactive": (v2.nactive, v1.nactive),
        "last": (v2.last, v1.last), "adaptation": (v2.adaptation, v1.adaptation),
        "eligibility": (v2.eligibility, v1.eligibility),
        "eligibility_last": (v2.eligibility_last, v1.eligibility_last),
        "modulation": (v2.modulation, v1.modulation),
        "modulation_last": (v2.modulation_last, v1.modulation_last),
    }
    return {k: bool(np.array_equal(a, b)) for k, (a, b) in fields.items()}


def compatibility_micro_suite():
    """Bitwise compare every frozen v1 micro semantic through the v2 ABI."""
    import tempfile
    cases = {
        "excitatory": ([0, 1, 1], [1], [1.0], [5, 5], [8.0, 0.0], False),
        "inhibitory": ([0, 1, 1], [1], [-1.0], [5, 5], [8.0, 0.0], False),
        "self_edge": ([0, 1, 1], [0], [0.25], [5, 5], [8.0, 0.0], False),
        "weak_edge": ([0, 1, 1], [1], [1e-4], [5, 5], [8.0, 0.0], False),
        "repeated_firing": ([0, 1, 1], [1], [8.0], [5, 5], [20.0, 0.0], False),
        "refractory_drop": ([0, 1, 1], [1], [8.0], [5, 5], [20.0, 0.0], True),
        "modulatory_source": ([0, 1, 1], [1], [0.5], [0, 5], [20.0, 0.0], False),
        "kc_adaptation": ([0, 1, 1], [1], [8.0], [5, 4], [20.0, 20.0], False),
    }
    rows = []
    all_ok = True
    with tempfile.TemporaryDirectory(prefix="q1-v7-compat-") as td:
        for dt in DT_VALUES:
            for name, (ptr, post, weight, layer, drive, ref_target) in cases.items():
                path = Path(td) / f"{name}-{dt}.npz"
                _write_micro_graph(path, ptr, post, weight)
                layer = np.asarray(layer, dtype=np.int8)
                rest = np.full(len(layer), -52., dtype=np.float32); rest[layer == 4] = -60.
                from stonkfly.neural.v2_brain import HybridBrain
                v2 = HybridBrain(path, dt=dt, graded_mask=np.zeros(len(layer), np.uint8), rest=rest)
                v1 = _v1_reference_path(path, dt, layer)
                _set_micro_boundary(v2, v1, layer, ref_target)
                case_ok = True; segment_rows = []
                # The repeated case deliberately uses several short calls;
                # all others include a drive transition and washout.
                schedule = [("zero", 1.0, np.zeros(len(layer), np.float32)),
                            ("stim", 1.0, np.asarray(drive, np.float32)),
                            ("repeat", 4.0 if name in ("repeated_firing", "kc_adaptation") else 1.0,
                             np.asarray(drive, np.float32)),
                            ("washout", 4.0, np.zeros(len(layer), np.float32))]
                for label, duration, d in schedule:
                    v2.run(duration, d); _v1_step(v1, d, duration)
                    pairs = _compat_state_pairs(v2, v1)
                    pairs["clock"] = int(v2.clock[0]) == int(v1.cursor)
                    pairs["total_spikes"] = v2.total_spikes == v1.total_spikes
                    case_ok &= all(pairs.values())
                    segment_rows.append({"segment": label, "pass": bool(all(pairs.values())), "fields": pairs})
                rows.append({"dt_ms": dt, "case": name, "pass": bool(case_ok), "segments": segment_rows})
                all_ok &= case_ok
    return {"pass": bool(all_ok), "case_count": len(cases), "dt_count": len(DT_VALUES), "rows": rows,
            "required_cases": sorted(cases)}


def _a2_v7():
    from stonkfly.neural.v2_brain import HybridBrain, exact_segment, release_from_voltage
    rows = []; all_ok = True
    for dt in DT_VALUES:
        path = ROOT / "runs/whole_brain_v2/Q1/.build/q1_v7_a2.npz"
        # One graded source owns all four outgoing edges: self, positive,
        # negative and weak targets.  No edge is hidden behind another source.
        _write_micro_graph(path, [0, 5, 5, 5], [0, 1, 1, 2, 2], [0.125, 1.0, -1.0, 0.5, -0.25])
        b = HybridBrain(path, dt=dt, graded_mask=np.array([1, 0, 0], np.uint8))
        drive = np.array([8., 0., 0.], np.float32)
        expected = np.zeros(5, dtype=np.float64); expected_first = np.full(5, -1, np.int64)
        target_prev = b.g_graded[1:].copy(); target_first = None; pre_arrival_zero = True
        nsteps = round(4.0 / dt)
        for _ in range(nsteps):
            clock = int(b.clock[0])
            b.run(dt, drive if clock == 0 else None, edge_trace=True)
            # The release sample is the post-segment graded voltage actually
            # presented to the queue.  Applying the frozen per-edge formula
            # to this observed sample avoids attributing float32 integration
            # roundoff to the delivery rule itself.
            r = float(release_from_voltage(float(b.v[0])))
            q = b.weight * r * dt / 2.2
            expected += q
            for edge, value in enumerate(q):
                if value != 0.0 and expected_first[edge] < 0:
                    expected_first[edge] = clock + round(1.8 / dt)
            if int(b.clock[0]) <= round(1.8 / dt) + 1:
                pre_arrival_zero &= np.array_equal(b.g_graded[1:], target_prev)
            if target_first is None and not np.array_equal(b.g_graded[1:], target_prev):
                target_first = int(b.clock[0])
            target_prev = b.g_graded[1:].copy()
        observed = b.edge_signed_delivery.copy()
        per_edge = []
        for e, w in enumerate(b.weight):
            exp_first = int(round(1.8 / dt))
            got_first = int(b.edge_first_arrival_clock[e])
            edge_ok = (abs(observed[e] - expected[e]) <= 1e-5 and
                       int(b.edge_event_count[e]) > 0 and got_first == exp_first and
                       np.sign(observed[e]) == np.sign(float(w)))
            per_edge.append({"edge": e, "weight": float(w), "expected_signed_delivery": float(expected[e]),
                             "observed_signed_delivery": float(observed[e]), "event_count": int(b.edge_event_count[e]),
                             "expected_arrival_clock": exp_first, "observed_arrival_clock": got_first,
                             "pass": bool(edge_ok)})
            all_ok &= edge_ok
        rows.append({"dt_ms": dt, "delay_steps": b.delay_steps, "pre_arrival_target_g_graded_zero": bool(pre_arrival_zero),
                     "first_target_voltage_influence_clock": target_first, "expected_first_target_voltage_influence_clock": b.delay_steps + 2,
                     "edges": per_edge,
                     "pass": bool(all(x["pass"] for x in per_edge) and pre_arrival_zero and target_first == b.delay_steps + 2)})
        all_ok &= rows[-1]["pass"]
    return {"pass": bool(all_ok), "rows": rows, "edge_count": 5,
            "required": ["per_edge_expected_observed", "exact_arrival_clock", "pre_arrival_zero", "self_edge", "weak_edge"]}


def _a4_v7():
    from stonkfly.neural.v2_brain import HybridBrain
    specs = {
        # An unrelated graded sink keeps the hybrid ABI active while the
        # audited edge itself remains spiking→spiking.
        "spiking_to_spiking": (np.array([0, 0, 1], np.uint8), 1.0, "g_event"),
        "spiking_to_graded": (np.array([0, 1], np.uint8), 1.0, "g_event"),
        "graded_to_graded": (np.array([1, 1], np.uint8), 1.0, "g_graded"),
        "graded_to_spiking": (np.array([1, 0], np.uint8), -1.0, "g_graded"),
    }
    rows = []; ok = True
    for name, (mask, weight, state_field) in specs.items():
        path = ROOT / f"runs/whole_brain_v2/Q1/.build/q1_v7_{name}.npz"
        ptr = [0, 1, 1, 1] if len(mask) == 3 else [0, 1, 1]
        _write_micro_graph(path, ptr, [1], [weight])
        b = HybridBrain(path, dt=.1, graded_mask=mask)
        b.active[:] = np.arange(b.n, dtype=np.int32); b.active_flag[:] = 1; b.nactive[0] = b.n
        if mask[0]: drive = np.array([8., 0.], np.float32)
        else:
            b.v[0] = -44.; drive = np.zeros(b.n, np.float32)
        baseline = float(getattr(b, state_field)[1]); b.run(4.0, drive, edge_trace=True)
        delta = float(getattr(b, state_field)[1]) - baseline
        row = {"connection": name, "arrival_clock": int(b.edge_first_arrival_clock[0]),
               "expected_arrival_clock": b.delay_steps, "signed_weight": float(weight),
               "edge_event_count": int(b.edge_event_count[0]), "state_field": state_field,
               "target_state_delta": delta, "state_delta_nonzero": bool(abs(delta) > 0),
               "sign_ok": bool(np.sign(b.edge_signed_delivery[0]) == np.sign(weight)),
               "count_ok": bool(b.edge_event_count[0] > 0),
               "pass": bool(b.edge_first_arrival_clock[0] == b.delay_steps and b.edge_event_count[0] > 0 and abs(delta) > 0 and np.sign(b.edge_signed_delivery[0]) == np.sign(weight))}
        rows.append(row); ok &= row["pass"]
    return {"pass": bool(ok), "rows": rows, "connection_count": 4,
            "required": sorted(specs)}


def _a5_v7():
    from stonkfly.neural.v2_brain import HybridBrain, exact_segment
    rows = []; ok = True
    for dt in DT_VALUES:
        path = ROOT / "runs/whole_brain_v2/Q1/.build/q1_v7_a5.npz"
        _write_micro_graph(path, [0, 1, 1], [1], [1.0])
        b = HybridBrain(path, dt=dt, graded_mask=np.array([1, 0], np.uint8))
        b.active[:] = [0, 1]; b.active_flag[:] = 1; b.nactive[0] = 2; b.refractory[1] = b.refractory_steps
        drive = np.array([8., 0.], np.float32); g_rows = []; frozen = True; release_compare = None
        for step in range(b.refractory_steps + b.delay_steps + 8):
            was_ref = b.refractory[1] > 0; oldv = float(b.v[1]); oldg = float(b.g_graded[1]); oldge = float(b.g_event[1]); oldu = float(b.graded_drive[1])
            expected_v, expected_g, _ = exact_segment(oldv, oldge, oldg, float(b.drive[1]), oldu, dt)
            b.run(dt, drive if step == 0 else None)
            gerr = abs(float(b.g_graded[1]) - expected_g)
            g_rows.append({"step": step, "clock": int(b.clock[0]), "was_refractory": bool(was_ref),
                           "g_graded": float(b.g_graded[1]), "expected_g_graded": float(expected_g), "abs_error": gerr})
            if was_ref: frozen &= (float(b.v[1]) == oldv)
            if not was_ref and release_compare is None:
                release_compare = {"observed_v": float(b.v[1]), "expected_v": float(expected_v),
                                   "abs_error": abs(float(b.v[1]) - expected_v)}
        g_ok = bool(all(x["abs_error"] <= 1e-4 for x in g_rows))
        v_ok = bool(release_compare and release_compare["abs_error"] <= 1e-4)
        # Empty-mask v2 invokes the embedded fixed v1 event path.  A spike
        # arriving during refractory must be dropped, leaving g_event zero.
        d = HybridBrain(path, dt=dt, graded_mask=np.zeros(2, np.uint8))
        d.active[:] = [0, 1]; d.active_flag[:] = 1; d.nactive[0] = 2; d.v[0] = -44.; d.refractory[1] = d.refractory_steps
        d.run((d.delay_steps + d.refractory_steps) * dt, np.zeros(2, np.float32))
        drop_ok = bool(float(d.g_event[1]) == 0.0)
        row = {"dt_ms": dt, "refractory_steps": b.refractory_steps, "full_window_steps": b.refractory_steps,
               "g_graded_rows": g_rows, "g_graded_analytic": g_ok, "voltage_frozen": frozen,
               "release_first_step": release_compare, "release_voltage_analytic": v_ok,
               "discrete_spike_refractory_drop": drop_ok,
               "pass": bool(g_ok and frozen and v_ok and drop_ok)}
        rows.append(row); ok &= row["pass"]
    return {"pass": bool(ok), "rows": rows, "dt_count": len(DT_VALUES),
            "required": ["complete_2.2ms_window", "g_graded_parse", "release_voltage", "discrete_drop"]}


def _a7_v7():
    from stonkfly.neural.v2_brain import HybridBrain
    rows = []; ok = True
    path = ROOT / "runs/whole_brain_v2/Q1/.build/q1_v7_a7_valid.npz"
    _write_micro_graph(path, [0, 1, 1], [1], [.8])
    for dt in (.05, .025):
        b = HybridBrain(path, dt=dt, graded_mask=np.array([1, 0], np.uint8))
        b.run(10., np.array([8., 0.], np.float32)); b.run(30., np.zeros(2, np.float32))
        rows.append({"dt_ms": dt, "final_v": float(b.v[1]), "final_g_graded": float(b.g_graded[1]),
                     "release_integral": float(b.release_integral[0]), "signed_delivery_integral": float(b.signed_delivery[0]),
                     "absolute_delivery_integral": float(b.absolute_delivery[0])})
    errs = [relerr(rows[0][k], rows[1][k]) for k in ("final_v", "final_g_graded", "release_integral", "signed_delivery_integral", "absolute_delivery_integral")]
    nonzero = all(abs(rows[i][k]) > 0 for i in range(2) for k in ("release_integral", "signed_delivery_integral", "absolute_delivery_integral"))
    csr_ok = True
    invalid = []
    for ptr, post, weight in (([1, 1, 1], [1], [.8]), ([0, 2, 1], [1, 0], [.8, .7]), ([0, 1, 1], [2], [.8]), ([0, 1, 1], [1], [np.nan])):
        p = path.with_name("invalid.npz"); _write_micro_graph(p, ptr, post, weight)
        try:
            HybridBrain(p, dt=.1, graded_mask=np.array([1, 0], np.uint8)); got = False
        except ValueError:
            got = True
        invalid.append(got); csr_ok &= got
    result = {"pass": bool(nonzero and csr_ok and all(x <= .02 for x in errs)), "dt_rows": rows,
              "symmetric_relative_errors": errs, "threshold": .02, "nonzero_delivery": nonzero,
              "valid_csr": {"ptr": [0, 1, 1], "post": [1], "weight": [.8]},
              "invalid_csr_rejected": invalid,
              "required": ["valid_ptr_zero", "ptr_monotonic", "post_bounds", "finite_weight", "nonzero_delivery"]}
    return result


def analytic_gates_v7():
    # Recompute A1/A3/A6 on valid CSR fixtures; the legacy v2 helper used an
    # intentionally invalid A7 pointer and is not part of the v7 evidence.
    from stonkfly.neural.v2_brain import HybridBrain, exact_segment, release_from_voltage
    vals = np.array([-60., -52., -50.25, -48., -45., -40.])
    expected = np.clip((vals + 52.) / 7., 0., 1.)
    err = float(np.max(np.abs(release_from_voltage(vals) - expected)))
    a1 = {"pass": err <= 1e-7, "max_abs_error": err, "threshold": 1e-7}
    a3_rows = []; a3_ok = True
    for dt in DT_VALUES:
        for u in (.7, -.7):
            path = ROOT / f"runs/whole_brain_v2/Q1/.build/q1_v7_a3_{dt}_{u}.npz"
            _write_micro_graph(path, [0, 0, 0], [], [])
            b = HybridBrain(path, dt=dt, graded_mask=np.array([0, 1], np.uint8))
            v0, ge0, gg0 = -51., .3, .2
            b.v[1], b.g_event[1], b.g_graded[1] = v0, ge0, gg0
            b.drive[1] = 1.5; b.previous_drive[1] = 1.5; b.graded_drive[1] = u
            b.active[:2] = [0, 1]; b.active_flag[:2] = 1; b.nactive[0] = 2
            ref = exact_segment(v0, ge0, gg0, 1.5, u, dt)
            b.run(dt)
            got = (float(b.v[1]), float(b.g_graded[1]), float(b.g_event[1]))
            rowerr = max(abs(got[i] - ref[i]) for i in range(3)); a3_ok &= rowerr <= 1e-4
            a3_rows.append({"dt_ms": dt, "rate": u, "max_abs_error": rowerr, "threshold": 1e-4})
    a3 = {"pass": bool(a3_ok), "rows": a3_rows, "threshold": 1e-4}
    zero_rows = []; zero_ok = True
    path = ROOT / "runs/whole_brain_v2/Q1/.build/q1_v7_a6.npz"
    _write_micro_graph(path, [0, 0, 0], [], [])
    for dt in DT_VALUES:
        z = HybridBrain(path, dt=dt, graded_mask=np.array([1, 0], np.uint8)); initial = z.audit_state()
        z.run(100.0, np.zeros(2, np.float32)); now = z.audit_state()
        keys = ["v", "g_event", "g_graded", "refractory", "queue", "queue_count", "graded_queue", "graded_drive", "counts", "adaptation", "modulation"]
        same = all(np.array_equal(initial[k], now[k]) for k in keys) and z.release_integral[0] == 0 and z.signed_delivery[0] == 0 and z.absolute_delivery[0] == 0
        zero_ok &= same; zero_rows.append({"dt_ms": dt, "same_physical_state": bool(same), "clock": int(z.clock[0])})
    a6 = {"pass": bool(zero_ok), "rows": zero_rows}
    a2, a4, a5, a7 = _a2_v7(), _a4_v7(), _a5_v7(), _a7_v7()
    out = {"A1_release_clamp": a1, "A2_delay_full_edge": a2,
           "A3_analytic_excitatory_inhibitory": a3, "A4_four_connection_types": a4,
           "A5_refractory_audit": a5, "A6_zero_input_100ms": a6,
           "A7_fine_dt_convergence": a7}
    out["pass"] = bool(all(isinstance(v, dict) and v.get("pass") is True for v in out.values()))
    out["required_gates"] = sorted(k for k in out if k.startswith("A"))
    return out


def compatibility_gate_v7(layer):
    full = compatibility_gate(layer)
    micro = compatibility_micro_suite()
    return {"pass": bool(full.get("pass") and micro.get("pass")), "full_graph": full, "micro": micro,
            "required": ["full_graph_bitwise", "micro_excitatory", "micro_inhibitory", "micro_self_edge", "micro_weak_edge", "micro_repeated_firing", "micro_refractory_drop", "micro_modulatory", "micro_kc_adaptation"]}


def graph_fixture_v7():
    a = np.load(ROOT / "data/graph.npz", allow_pickle=False)
    with pa.memory_map(str(ROOT / "data/normalized/edges.arrow"), "r") as f:
        table = ipc.open_file(f).read_all()
    pre = table.column("pre_index").to_numpy(zero_copy_only=False)
    post = table.column("post_index").to_numpy(zero_copy_only=False)
    contacts = table.column("synapse_count").to_numpy(zero_copy_only=False).astype(np.uint64, copy=False)
    order = np.argsort(pre, kind="stable")
    pre_s, post_s, contacts_s = pre[order], post[order], contacts[order]
    ptr = np.asarray(a["ptr"], dtype=np.int64); expected_pre = np.repeat(np.arange(len(a["ids"])), np.diff(ptr))
    alignment = bool(np.array_equal(pre_s, expected_pre) and np.array_equal(post_s, a["post"]) and ptr[0] == 0 and np.all(np.diff(ptr) >= 0))
    observed = {"nodes": int(len(a["ids"])), "edges": int(len(a["post"])), "synaptic_contacts": int(contacts_s.sum(dtype=np.uint64))}
    weight_finite = bool(np.isfinite(a["weight"]).all())
    return a, {"pass": bool(observed == SCALE and alignment and weight_finite), "observed": observed,
               "expected": SCALE, "normalized_alignment": alignment, "contact_sum": int(contacts_s.sum(dtype=np.uint64)),
               "contact_sha256": hashlib.sha256(contacts_s.tobytes()).hexdigest(),
               "weight_sha256": sha256_array(a["weight"]), "weight_finite": weight_finite,
               "ptr_first": int(ptr[0]), "ptr_last": int(ptr[-1])}


def trial_v7(dt, graded, ids, name, selected, kc_mask, repeat):
    from stonkfly.neural.v2_brain import HybridBrain
    n = len(ids); rest = np.full(n, -52., np.float32); rest[np.asarray(kc_mask, bool)] = -60.
    b = HybridBrain(ROOT / "data/graph.npz", dt=dt, graded_mask=graded, rest=rest); b.kc_mask[:] = kc_mask
    b.modulation_mask[:] = 0
    weights_before = sha256_array(b.weight)
    zero = np.zeros(n, np.float32); b.run(10., zero)
    stim = zero.copy(); stim[selected] = 3.5; b.run(10., stim); b.run(20., zero)
    weights_after = sha256_array(b.weight); s = b.audit_state()
    state_hash = jhash({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()})
    finite = bool(np.isfinite(b.v).all() and np.isfinite(b.g_event).all() and np.isfinite(b.g_graded).all() and np.isfinite(b.graded_queue).all() and np.isfinite(b.graded_drive).all() and np.isfinite(b.modulation).all())
    bounds = bool(np.max(np.abs(b.v)) < 1e6 and np.max(np.abs(b.g_event)) < 1e9 and np.max(np.abs(b.g_graded)) < 1e9)
    queue_integrity = bool(np.all(b.spike_queue_count >= 0) and np.all(b.spike_queue_count <= n) and np.all(np.isfinite(b.graded_queue)) and np.all(np.isfinite(b.graded_drive)) and b.overflow[0] == 0 and b.clock[0] == round(40.0 / dt))
    zero_eq = bool(name != "ZERO" or (b.total_spikes == 0 and b.release_integral[0] == 0 and b.signed_delivery[0] == 0 and b.absolute_delivery[0] == 0))
    return {"name": name, "dt_ms": dt, "repeat": repeat, "selected_ids": [int(x) for x in ids[selected]], "selected_id_sha256": source_hash(ids[selected]), "spike_count": int(b.total_spikes), "release_integral": float(b.release_integral[0]), "signed_delivery_integral": float(b.signed_delivery[0]), "absolute_delivery_integral": float(b.absolute_delivery[0]), "finite": finite, "bounds": bounds, "zero_equilibrium": zero_eq, "queue_integrity": queue_integrity, "overflow": int(b.overflow[0]), "clock": int(b.clock[0]), "weights_before_sha256": weights_before, "weights_after_sha256": weights_after, "weights_unchanged": bool(weights_before == weights_after), "state_sha256": state_hash}


def full_graph_trials_v7(a, layer):
    graded = np.isin(layer, [1, 2, 3]).astype(np.uint8); kc = layer == 4; ids = a["ids"]
    selections = {"ZERO": np.array([], np.int64), "PHOTO128": np.flatnonzero(layer == 1)[:128], "LAMINA128": np.flatnonzero(layer == 2)[:128], "APL2": np.flatnonzero(layer == 3)}
    rows = [trial_v7(dt, graded, ids, name, selections[name], kc, rep) for dt in DT_VALUES for rep in (1, 2) for name in ("ZERO", "PHOTO128", "LAMINA128", "APL2")]
    by = {}
    for r in rows: by.setdefault((r["dt_ms"], r["name"]), []).append(r)
    deterministic = all(len(v) == 2 and v[0]["state_sha256"] == v[1]["state_sha256"] and v[0]["weights_before_sha256"] == v[0]["weights_after_sha256"] and v[1]["weights_before_sha256"] == v[1]["weights_after_sha256"] for v in by.values())
    stim = [r for r in rows if r["name"] != "ZERO"]; fine = {}
    for name in ("PHOTO128", "LAMINA128", "APL2"):
        x = next(r for r in stim if r["name"] == name and r["repeat"] == 1 and r["dt_ms"] == .05); y = next(r for r in stim if r["name"] == name and r["repeat"] == 1 and r["dt_ms"] == .025)
        fine[name] = {k: relerr(x[k], y[k]) for k in ("release_integral", "signed_delivery_integral", "absolute_delivery_integral")}
    apl = [r for r in stim if r["name"] == "APL2"]
    smoke = all(r["finite"] and r["bounds"] and r["queue_integrity"] and r["weights_unchanged"] for r in rows) and all(r["zero_equilibrium"] for r in rows if r["name"] == "ZERO") and all(r["release_integral"] > 0 and r["absolute_delivery_integral"] > 0 for r in stim) and all(all(v <= .02 for v in z.values()) for z in fine.values())
    return {"pass": bool(smoke and deterministic), "trials": rows, "trial_count": len(rows), "deterministic_repeats": bool(deterministic), "fine_dt_symmetric_relative_error": fine, "apl_signed_delivery_negative": bool(all(r["signed_delivery_integral"] < 0 for r in apl)), "smoke_gate": bool(smoke), "weights_frozen": bool(all(r["weights_unchanged"] for r in rows)), "queue_overflow_zero": bool(all(r["overflow"] == 0 for r in rows))}


def run_v7(out=OUT):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    fixed, p0 = fixed_inputs(); a, graph = graph_fixture_v7()
    nodes = {k: np.asarray(v) for k, v in feather.read_table(ROOT / "data/normalized/neurons.feather").to_pydict().items()}
    layer, names = classify(nodes)
    classification = {n: {"count": int(np.sum(layer == i)), "source_id_sha256": source_hash(a["ids"][layer == i])} for i, n in enumerate(names)}
    graded = np.isin(layer, [1, 2, 3]); classification["graded_union"] = {"count": int(graded.sum()), "source_id_sha256": source_hash(a["ids"][graded])}
    classification_gate = bool([classification[n]["count"] for n in names] == [541, 4706, 7114, 2, 4064, 150273] and classification["graded_union"]["count"] == 11822)
    from stonkfly.neural.v2_brain import build
    build_info = build(); source_start = sha(ROOT / build_info["source"]); binary_start = sha(ROOT / build_info["binary"]); runner_start = sha(Path(__file__)); tests_path = ROOT / "tests/test_v2_q1_hybrid_qualification.py"; tests_start = sha(tests_path)
    micro = analytic_gates_v7(); compat = compatibility_gate_v7(layer); observer = observer_gate(layer); full = full_graph_trials_v7(a, layer)
    source_end = sha(ROOT / build_info["source"]); binary_end = sha(ROOT / build_info["binary"]); runner_end = sha(Path(__file__)); tests_end = sha(tests_path)
    hash_gate = source_start == source_end and binary_start == binary_end and runner_start == runner_end and tests_start == tests_end
    required_top = ["fixed_inputs", "p0_ready", "graph_fixture", "classification", "analytic_micro", "compatibility", "observer_equivalence", "full_graph", "run_hash_stability"]
    gates = {"fixed_inputs": True, "p0_ready": True, "graph_fixture": graph, "classification": {"pass": classification_gate, "observed": classification}, "analytic_micro": micro, "compatibility": compat, "observer_equivalence": observer, "full_graph": full, "run_hash_stability": {"pass": bool(hash_gate), "source_start": source_start, "source_end": source_end, "binary_start": binary_start, "binary_end": binary_end, "runner_start": runner_start, "runner_end": runner_end, "tests_start": tests_start, "tests_end": tests_end}}
    missing = [k for k in required_top if k not in gates] + [k for k in ("A1_release_clamp", "A2_delay_full_edge", "A3_analytic_excitatory_inhibitory", "A4_four_connection_types", "A5_refractory_audit", "A6_zero_input_100ms", "A7_fine_dt_convergence") if k not in micro]
    implementation = bool(not missing and gates["fixed_inputs"] and gates["p0_ready"] and graph["pass"] and classification_gate and hash_gate)
    if missing or not implementation: verdict = "Q1_IMPLEMENTATION_FAILURE"
    elif not compat["pass"]: verdict = "Q1_COMPATIBILITY_FAILURE"
    elif not micro["pass"] or not observer["pass"]: verdict = "Q1_NUMERICAL_FAILURE"
    elif not full["pass"]: verdict = "Q1_FULL_GRAPH_SMOKE_FAILURE"
    else: verdict = "Q1_IMPLEMENTATION_QUALIFIED"
    fixed_after = {k: sha(ROOT / k) for k in FIXED}
    clean_checkpoint = {"kind": "generated_clean_rest_zero_queue_state", "graph_sha256": sha(ROOT / "data/graph.npz"), "state_schema_sha256": jhash(["v", "g_event", "g_graded", "refractory", "drive", "previous_drive", "queue", "queue_count", "graded_queue", "graded_drive", "counts", "active", "active_flag", "nactive", "last", "eligibility", "eligibility_last", "modulation", "modulation_last", "adaptation", "clock"]), "learning": "off"}
    formal = {"version": "v7", "verdict": verdict, "fixed_input_sha256": fixed, "fixed_input_sha256_after": fixed_after, "classification": classification, "graph_scale": graph, "gates": gates, "missing_gates": missing, "v2_build": build_info, "preregistration_sha256": sha(PREREG), "audit_sha256": sha(AUDIT), "audit_v6_sha256": sha(AUDIT_V6), "wall_time_seconds": time.time() - start, "clean_checkpoint": clean_checkpoint}
    (out / "q1_formal.json").write_text(json.dumps(formal, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(out / "q1_raw.npz", micro_trace_json=np.asarray(json.dumps(micro, sort_keys=True)), compatibility_trace_json=np.asarray(json.dumps(compat, sort_keys=True)), observer_trace_json=np.asarray(json.dumps(observer, sort_keys=True)), full_graph_summary_json=np.asarray(json.dumps({k: v for k, v in full.items() if k != "trials"}, sort_keys=True)), trial_state_sha256=np.asarray([x["state_sha256"] for x in full["trials"]]), trial_dt_ms=np.asarray([x["dt_ms"] for x in full["trials"]]), trial_name=np.asarray([x["name"] for x in full["trials"]]), release=np.asarray([x["release_integral"] for x in full["trials"]]), signed_delivery=np.asarray([x["signed_delivery_integral"] for x in full["trials"]]), absolute_delivery=np.asarray([x["absolute_delivery_integral"] for x in full["trials"]]))
    (out / "q1_report.md").write_text(f"# V2-Q1 hybrid-kernel qualification v7\n\nVerdict: `{verdict}`\n\nThe v7 audit includes exact per-edge delivery clocks, all four connection types, complete refractory windows at three dt values, empty-mask v1/v2 micro compatibility, CSR validation, observer equivalence, contact conservation, and 24 independent full-graph trials. Learning, reward, market, decoder and execution paths were not used.\n")
    manifest = {"version": "v7", "python": sys.version, "platform": platform.platform(), "v2_build": build_info, "runner_sha256_start": runner_start, "runner_sha256_end": runner_end, "tests_sha256_start": tests_start, "tests_sha256_end": tests_end, "binary_sha256_start": binary_start, "binary_sha256_end": binary_end, "fixed_v1_source_sha256": sha(ROOT / "stonkfly/neural/kernel.cpp"), "clean_checkpoint": clean_checkpoint, "required_top_gates": required_top, "missing_gates": missing}
    (out / "q1_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return formal


# The formal v2 runner is retained as the implementation entry point; v7 is a
# new output/amendment directory and never overwrites v2-v6 artifacts.
run = run_v7


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=OUT); args = ap.parse_args(); print(json.dumps(run(args.out), indent=2, sort_keys=True))
