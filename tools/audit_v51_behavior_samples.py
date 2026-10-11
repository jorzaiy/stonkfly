#!/usr/bin/env python3
"""Read-only extraction of representative V5.1 behavior samples.

No model is imported or run.  The forward values are descriptive portfolio
mark-to-market paths: for a decision at event index i, k-session performance
is the Decimal compound of reward.portfolio_return on event rows i+1..i+k.
They are not per-stock returns, counterfactual attribution, or trade advice.
"""
from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/wc006_v5_1_v6_development/behavior-samples-audit-v1"
RUNS = {
    "2022_enabled": ROOT / "runs/wc006_v5_1_v6_development/annual-2022-enabled-uninterrupted-v3/events.jsonl",
    "2022_frozen": ROOT / "runs/wc006_v5_1_v6_development/annual-2022-frozen-uninterrupted-v3/events.jsonl",
    "2023_enabled": ROOT / "runs/wc006_v5_1_v6_development/annual-2023-uninterrupted-v1/events.jsonl",
    "2023_frozen": ROOT / "runs/wc006_v5_1_v6_development/annual-2023-frozen-uninterrupted-v1/events.jsonl",
}

SAMPLES = [
    ("2022_enabled", "2022-01-04", "initial exposure; clean checkpoint"),
    ("2022_enabled", "2022-02-17", "defer before a large positive forward path (descriptive miss-upside candidate)"),
    ("2022_enabled", "2022-07-05", "expose followed by negative forward path (descriptive falling-knife candidate)"),
    ("2022_frozen", "2022-07-05", "same-date frozen contrast"),
    ("2022_enabled", "2022-09-07", "repeated candidate day before extreme path"),
    ("2022_enabled", "2022-09-08", "repeated candidate day with extreme negative path"),
    ("2022_frozen", "2022-12-22", "extreme negative next-event path"),
    ("2022_frozen", "2022-12-26", "expose after volatile path"),
    ("2023_enabled", "2023-01-03", "initial exposure; clean checkpoint"),
    ("2023_enabled", "2023-02-27", "defer before a positive forward path (descriptive miss-upside candidate)"),
    ("2023_enabled", "2023-05-11", "defer with negative forward path"),
    ("2023_enabled", "2023-06-28", "expose with volatile forward path"),
    ("2023_enabled", "2023-08-24", "requested/effective multiplier mismatch; hesitation/guard-boundary candidate"),
    ("2023_enabled", "2023-11-01", "expose followed by positive forward path"),
    ("2023_frozen", "2023-10-19", "frozen expose followed by negative forward path"),
    ("2023_frozen", "2023-10-23", "repeated frozen expose with extreme negative next-event path"),
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dec(value: object) -> Decimal:
    return Decimal(str(value))


def future_path(events: list[dict[str, Any]], index: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in (1, 3, 5):
        rows = events[index + 1:index + 1 + k]
        if len(rows) < k:
            out[str(k)] = {"status": "unavailable_insufficient_future_event_rows", "return": None, "dates": [x.get("date") for x in rows]}
            continue
        value = Decimal("1")
        for row in rows:
            value *= Decimal("1") + dec(row.get("reward", {}).get("portfolio_return", "0"))
        out[str(k)] = {
            "status": "available_descriptive_portfolio_path",
            "return": str(value - Decimal("1")),
            "dates": [x.get("date") for x in rows],
            "source_field": "reward.portfolio_return",
        }
    return out


def load_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract(label: str, events: list[dict[str, Any]], date: str, reason: str) -> dict[str, Any]:
    matches = [i for i, e in enumerate(events) if e.get("date") == date]
    if len(matches) != 1:
        raise ValueError(f"{label}:{date}:expected_one_event_got_{len(matches)}")
    i = matches[0]
    e = events[i]
    neural = e.get("neural", {})
    batch = e.get("batch")
    # The event has no raw OHLC/sensory vector. Preserve every recorded Fly-side
    # input/proxy field and explicitly mark the unavailable layers.
    fly_marks = e.get("mark_provenance", {}).get("fly", {})
    return {
        "sample_reason": reason,
        "run_label": label,
        "source_event_index": i,
        "date": date,
        "candidate_stock_input": {
            "codes": e.get("candidate_selection", {}).get("codes"),
            "epoch": e.get("candidate_selection", {}).get("epoch"),
            "updated": e.get("candidate_selection", {}).get("updated"),
            "fly_mark_provenance": fly_marks,
            "signal_eligible": e.get("signal_eligible"),
            "requested_target": e.get("requested_target"),
            "effective_target": e.get("effective_target"),
            "actual_exposure_before_event": e.get("actual_exposure"),
            "input_sha256": neural.get("input_sha256"),
            "raw_ohlc_or_sensory_vector": {"status": "unavailable_in_events_jsonl"},
            "per_stock_forward_return": {"status": "unavailable_in_events_jsonl"},
        },
        "neural_output": {
            "decoder_action": e.get("decoder", {}).get("action"),
            "decoder_side": e.get("decoder", {}).get("side"),
            "decoder_requested_multiplier": e.get("decoder", {}).get("requested_multiplier"),
            "decoder_effective_multiplier": e.get("decoder", {}).get("effective_multiplier"),
            "left_hz": neural.get("left_hz"),
            "right_hz": neural.get("right_hz"),
            "difference_hz": neural.get("difference_hz"),
            "gate_spikes": neural.get("gate_spikes"),
            "KC_spikes": neural.get("KC_spikes"),
            "reward_spikes": neural.get("reward_spikes"),
            "aversive_spikes": neural.get("aversive_spikes"),
            "total_spikes": neural.get("total_spikes"),
            "stimulus": neural.get("stimulus"),
            "stimulus_ms": neural.get("stimulus_ms"),
            "candidate_day": neural.get("v51_candidate_day"),
            "candidate_epoch": neural.get("v51_candidate_epoch"),
            "seed": neural.get("v51_seed"),
            "spike_sha256": neural.get("spike_sha256"),
            "memory_summary": neural.get("memory"),
        },
        "execution_action": {
            "decoder": e.get("decoder"),
            "guard_reason": e.get("guard_reason"),
            "intent_statuses": e.get("intent_statuses"),
            "batch": batch,
            "resolved_batches": e.get("resolved_batches"),
            "turnover": e.get("turnover"),
            "fees": e.get("fees"),
            "paper_only_source": True,
        },
        "forward_performance": future_path(events, i),
        "activation_summary": {
            "scalar_summary_available": all(neural.get(k) is not None for k in ("left_hz", "right_hz", "difference_hz", "gate_spikes", "KC_spikes", "total_spikes")),
            "full_neuron_activation_vector": "unavailable_in_events_jsonl",
            "checkpoint_file": e.get("checkpoint", {}).get("file"),
            "checkpoint_sha256": e.get("checkpoint", {}).get("sha256"),
            "memory_summary_available": isinstance(neural.get("memory"), dict),
        },
        "event_hash": e.get("event_hash"),
    }


def main() -> int:
    loaded = {label: load_events(path) for label, path in RUNS.items()}
    samples = [extract(label, loaded[label], date, reason) for label, date, reason in SAMPLES]
    source_pins = {label: {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "event_count": len(loaded[label])} for label, path in RUNS.items()}
    report = {
        "schema": "wc006-v5.1-behavior-samples-readonly-audit/v1",
        "scope": {"read_only": True, "model_started": False, "data_modified": False, "paper_only": True, "sample_count": len(samples), "years": [2022, 2023]},
        "source_pins": source_pins,
        "extraction_definition": {
            "candidate_inputs": "all Fly-side fields retained in events.jsonl: candidate_selection, mark_provenance.fly, eligibility, targets, actual exposure, and neural.input_sha256; raw OHLC/sensory vectors are absent.",
            "neural_output": "decoder plus recorded scalar Hz/spike/stimulus/memory summary; full neuron activation vectors are absent.",
            "execution": "decoder, guard_reason, intent_statuses, batch/resolved_batches, turnover, fees from the same event.",
            "forward_1_3_5": "Decimal compound of reward.portfolio_return on the next 1/3/5 event rows after the decision row; descriptive portfolio path, not stock-level return or counterfactual attribution.",
        },
        "samples": samples,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / "behavior-samples.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# V5.1 behavior samples (read-only)", "", "Forward performance is descriptive compounded event reward.portfolio_return; per-stock forward return and full activation vectors are unavailable.", "", "| run | date | candidate codes | action/side | 1-event | 3-event | 5-event | reason |", "|---|---|---|---|---:|---:|---:|---|"]
    for s in samples:
        f = s["forward_performance"]
        vals = [f[str(k)]["return"] if f[str(k)]["return"] is not None else "NA" for k in (1, 3, 5)]
        n = s["neural_output"]
        lines.append(f"| {s['run_label']} | {s['date']} | {','.join(s['candidate_stock_input']['codes'] or [])} | {n['decoder_action']}/{n['decoder_side']} | {vals[0]} | {vals[1]} | {vals[2]} | {s['sample_reason']} |")
    md_path = OUT / "behavior-samples.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "json": str(json_path), "markdown": str(md_path), "sample_count": len(samples), "json_sha256": sha(json_path), "markdown_sha256": sha(md_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
