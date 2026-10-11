"""No-market V5.1 sensory-only reward eligibility coverage audit.

This is a post-failure diagnostic.  It reuses the annual draft stream and the
V4 adapter's deterministic ``_frame`` renderer, carries the frozen active
candidate basket across all 242 dates, and calls only ``observe(frame,
"none")``.  It never settles a market, creates an order, or delivers a
reinforcement pulse.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from stonkfly.config import D, Settings
from stonkfly.neural.v51_controller import V51Controller
from experiments.a_share.wc006_male_cns_adapter import StreamRecord, ValidatedStream, _code
from experiments.a_share.wc006_male_cns_v4_adapter import V4Stream, _frame, _load_price_rows

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_SOURCE = ROOT / "stonkfly/neural/controller.py"
V51_CONTROLLER_SOURCE = ROOT / "stonkfly/neural/v51_controller.py"
FRAME_RENDERER_SOURCE = ROOT / "experiments/a_share/wc006_male_cns_v4_adapter.py"
PROTOCOL = ROOT / "docs/experiments/wc006_v5_1_reward_eligibility_coverage_preregister.md"
DEFAULT_MANIFEST = ROOT / "runs/wc006_v5_1_inputs/annual-2023-manifest-draft/manifest.json"
START_DATE = "2023-01-03"
END_DATE = "2023-12-29"
EXPECTED_DATES = 242
EXPECTED_SIGNALS = 17
NEURAL_MS = 500.0
EXPECTED_SOURCE_MANIFEST_SHA256 = "5874b6cefab5f1c4b1ee04db819fd76430cdf67495db67db5242f5d83f42cb52"
EXPECTED_FIRST_FRAME_RAW_SHA256 = "3ab3e37867cbb9b2a860ad8d2760cf5f5b707001c5357da58f4c1c45e0867971"


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class AppendOnlyLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size:
            raise FileExistsError(self.path)
        self.previous = "GENESIS"
        self.sequence = 0

    def append(self, event: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = {"seq": self.sequence, "event": event, "payload": payload, "prev_hash": self.previous}
        event_hash = hashlib.sha256(self.previous.encode() + canonical(body)).hexdigest()
        record = {**body, "event_hash": event_hash}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.previous = event_hash
        self.sequence += 1
        return record


def verify_append_only_log(path: Path) -> dict[str, Any]:
    previous = "GENESIS"
    records = 0
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        record = json.loads(line)
        body = {key: record[key] for key in ("seq", "event", "payload", "prev_hash")}
        if record["seq"] != records or record["prev_hash"] != previous:
            raise ValueError(f"coverage event chain mismatch at line {line_number}")
        actual = hashlib.sha256(previous.encode() + canonical(body)).hexdigest()
        if record.get("event_hash") != actual:
            raise ValueError(f"coverage event hash mismatch at line {line_number}")
        previous = actual
        records += 1
    return {"records": records, "head": previous}


class CleanCashLedger:
    """Read-only frame state: clean cash and no holdings, no DB/execution."""

    def get_meta(self, key: str) -> str | None:
        return {"cash": "100000"}.get(key)

    def holdings(self, code: str) -> Decimal:
        return D("0")

    @property
    def db(self):
        return self

    def execute(self, query: str):
        if "FROM lots" in query:
            return []
        raise RuntimeError(f"coverage ledger rejects database query: {query}")


def _default_stream_loader(manifest: Path):
    """Strictly load the annual-slice schema into the existing V4 stream type."""
    path = Path(manifest).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent
    if payload.get("schema") != "wc006-v5.1-annual-input-slice/v1" or payload.get("draft") is not True:
        raise ValueError("annual draft schema/provenance is missing")
    if payload.get("paper_only") is not True or payload.get("broker_orders_enabled") is not False or payload.get("live_execution_authorized") is not False or payload.get("authorization_artifact") is not False or payload.get("candidate_selection_frozen") is not True:
        raise ValueError("annual draft is not a frozen paper-only non-authorizing input")
    if payload.get("year") != 2023 or payload.get("observation_window") != [START_DATE, END_DATE] or payload.get("settlement_boundary") != "2024-01-02":
        raise ValueError("annual draft year/window/settlement boundary is not pinned")
    if payload.get("source_manifest_sha256") != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise ValueError("annual draft source manifest hash is not pinned")
    if payload.get("observation_count") != EXPECTED_DATES or payload.get("candidate_record_count") != EXPECTED_SIGNALS:
        raise ValueError("annual draft counts are not pinned")
    source_hashes = payload.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise ValueError("annual draft source hashes are missing")
    for relative, key in (("inputs/active_config.json", "active_config"), ("inputs/universe.txt", "universe"), ("inputs/calendar.json", "calendar")):
        source = (root / relative).resolve()
        if not source.is_file() or sha256(source) != source_hashes.get(key):
            raise ValueError(f"annual draft input hash mismatch: {relative}")
    calendar_payload = json.loads((root / "inputs/calendar.json").read_text(encoding="utf-8"))
    calendar = tuple(str(item) for item in calendar_payload.get("dates", []))
    if list(calendar) != sorted(set(calendar)) or len([item for item in calendar if START_DATE <= item <= END_DATE]) != EXPECTED_DATES:
        raise ValueError("annual draft calendar does not contain 242 sessions")
    cache_files = payload.get("cache_files")
    if not isinstance(cache_files, dict) or not cache_files:
        raise ValueError("annual draft cache provenance is missing")
    cache_paths: dict[str, Path] = {}
    for raw_code, item in cache_files.items():
        code = _code(raw_code)
        if not isinstance(item, dict):
            raise ValueError("annual draft cache item is invalid")
        copied = (root / str(item.get("copied_path", ""))).resolve()
        try:
            copied.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"annual draft cache path escapes manifest root: {code}") from exc
        if not copied.is_file() or sha256(copied) != item.get("sha256"):
            raise ValueError(f"annual draft cache hash mismatch: {code}")
        cache_paths[code] = copied
    artifacts = payload.get("selector_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != EXPECTED_SIGNALS:
        raise ValueError("annual draft selector artifacts are incomplete")
    for item in artifacts:
        copied = (root / str(item.get("copied_path", ""))).resolve()
        try:
            copied.relative_to(root)
        except ValueError as exc:
            raise ValueError("annual draft selector path escapes manifest root") from exc
        if not copied.is_file() or sha256(copied) != item.get("sha256"):
            raise ValueError("annual draft selector artifact hash mismatch")
    records: list[StreamRecord] = []
    for raw in payload.get("records", []):
        selector = raw.get("selector")
        targets = tuple({"code": _code(row["code"]), "weight": float(row["target_weight"]), "score": row.get("score")} for row in selector.get("target_topk", []))
        if len(targets) != 3:
            raise ValueError("annual draft target basket is not three symbols")
        signal_date = str(raw["signal_date"])
        execution_date = str(raw["execution_date"])
        later = [item for item in calendar if item > signal_date]
        if not later or execution_date != min(later) or abs(sum(item["weight"] for item in targets) - 1.0) > 1e-9:
            raise ValueError(f"annual draft selector timing/weights invalid at {signal_date}")
        records.append(StreamRecord(signal_date, execution_date, targets))
    if len(records) != EXPECTED_SIGNALS or [item.signal_date for item in records] != sorted(item.signal_date for item in records):
        raise ValueError("annual draft selector records are not chronological")
    base = ValidatedStream(path, payload, tuple(records), calendar, cache_paths, sha256(path))
    prices = _load_price_rows(base)
    for record in records:
        for target in record.targets:
            if len([date for date in prices[target["code"]] if date <= record.signal_date]) < 61:
                raise ValueError(f"insufficient causal lookback for {target['code']} on {record.signal_date}")
    spec = ROOT / "docs/experiments/wc006_v4_2021_2025_preregister.md"
    gate = {"status": "PASS", "candidate_manifest_sha256": sha256(path), "source_manifest_sha256": payload["source_manifest_sha256"], "calendar_days": len(calendar), "records": len(records), "paper_only": True, "broker_orders_enabled": False, "live_execution_authorized": False, "market_execution": False}
    return V4Stream(base, prices, sha256(spec)), gate


def preflight_annual_inputs(manifest: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load the real annual draft and render only its first fixed frame."""
    stream, gate = _default_stream_loader(Path(manifest))
    first = next(record for record in stream.base.records if record.signal_date == START_DATE)
    frame = _frame(first, stream.prices, START_DATE, stream.base.calendar, CleanCashLedger(), D("1.0"), D("0"))
    frame_hash = hashlib.sha256(np.asarray(frame).tobytes()).hexdigest()
    if frame_hash != EXPECTED_FIRST_FRAME_RAW_SHA256:
        raise ValueError(f"first annual frame hash mismatch: {frame_hash}")
    return {"manifest_sha256": sha256(Path(manifest)), "source_manifest_sha256": gate["source_manifest_sha256"], "frame_sha256": frame_hash, "shape": list(frame.shape), "dtype": frame.dtype.str, "records": len(stream.base.records), "calendar_days": len(stream.base.calendar)}


def _eligible_mask(controller: Any, stimulus: str) -> np.ndarray:
    method = getattr(controller, "_linked_eligibility_mask", None)
    if callable(method):
        return np.asarray(method(stimulus), dtype=bool)
    brain = controller.brain
    circuit = brain.circuit
    rows = np.asarray(circuit["gain"])
    edges = np.asarray(circuit["edges"], dtype=np.int64)
    pre = np.asarray(circuit["pre"], dtype=np.int64)
    groups = np.asarray(circuit[stimulus], dtype=np.int64)
    reward_count = len(np.asarray(circuit["reward"]))
    start = 0 if stimulus == "reward" else reward_count
    return np.any(rows[start:start + len(groups)] != 0, axis=0) & (np.abs(np.asarray(brain.eligibility)[pre]) > 0)


def _active_payload(record: Any, epoch: int) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "codes": sorted(str(item["code"]) for item in record.targets),
        "targets": [
            {"code": str(item["code"]), "weight": str(item["weight"]), "score": item.get("score")}
            for item in record.targets
        ],
    }


def _plastic_state_digests(brain: Any) -> dict[str, str]:
    circuit = brain.circuit
    edges = np.asarray(circuit["edges"], dtype=np.int64)
    required = ("weight", "memory_u", "memory_w")
    if any(not hasattr(brain, name) for name in required):
        raise ValueError("coverage brain lacks plastic state arrays")
    return {
        "edge_weights_sha256": digest(np.asarray(brain.weight)[edges]),
        "memory_u_sha256": digest(np.asarray(brain.memory_u)),
        "memory_w_sha256": digest(np.asarray(brain.memory_w)),
    }


def run_coverage_audit(
    output: Path,
    manifest: Path = DEFAULT_MANIFEST,
    *,
    controller_factory: Callable[[Settings], Any] = V51Controller,
    stream_loader: Callable[[Path], tuple[Any, dict[str, Any]]] = _default_stream_loader,
    frame_factory: Callable[..., np.ndarray] | None = None,
    expected_dates: int = EXPECTED_DATES,
    expected_signals: int = EXPECTED_SIGNALS,
) -> dict[str, Any]:
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"coverage output must be new: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    log = AppendOnlyLog(output / "coverage-events.jsonl")
    summary: dict[str, Any] = {
        "schema": "wc006-v5.1-reward-eligibility-coverage/v1",
        "status": "FAILED",
        "protocol": {"file": str(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "manifest": {"file": str(Path(manifest).resolve()), "sha256": sha256(Path(manifest))},
        "paper_only": True,
        "market_execution": False,
        "reinforcement_delivered": False,
        "provenance": {
            "audit_code_sha256": sha256(Path(__file__).resolve()),
            "v51_controller_sha256": sha256(V51_CONTROLLER_SOURCE),
            "frozen_controller_sha256": sha256(CONTROLLER_SOURCE),
            "frame_renderer_sha256": sha256(FRAME_RENDERER_SOURCE),
            "protocol_sha256": sha256(PROTOCOL),
        },
    }
    try:
        manifest_payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
        if manifest_payload.get("source_manifest_sha256") != EXPECTED_SOURCE_MANIFEST_SHA256:
            raise ValueError("annual draft does not bind the pinned V5 source manifest")
        stream, input_gate = stream_loader(Path(manifest))
        if input_gate.get("candidate_manifest_sha256") != sha256(Path(manifest)):
            raise ValueError("validated candidate manifest hash does not match the annual draft")
        dates = tuple(date for date in stream.base.calendar if START_DATE <= date <= END_DATE)
        if len(dates) != expected_dates or dates[0] != START_DATE or dates[-1] != END_DATE:
            raise ValueError("annual observation calendar is not the pinned 242-session window")
        records = {record.signal_date: record for record in stream.base.records if START_DATE <= record.signal_date <= END_DATE}
        if len(records) != expected_signals:
            raise ValueError("annual candidate stream does not contain exactly 17 selector dates")
        if sorted(records) != list(records):
            raise ValueError("candidate selector dates are not chronological")
        settings = Settings(neural_ms=NEURAL_MS, learning=True)
        controller = controller_factory(settings)
        ledger = CleanCashLedger()
        clean = output / "brain-clean.npz"
        controller.save(clean)
        clean_hash = sha256(clean)
        plastic_start = _plastic_state_digests(controller.brain)
        log.append("coverage_start", {
            "schema": summary["schema"], "protocol_sha256": summary["protocol"]["sha256"],
            "manifest_sha256": summary["manifest"]["sha256"], "input_gate": input_gate,
            "dates": [dates[0], dates[-1]], "observation_count": len(dates),
            "signal_count": len(records), "settings": {"neural_ms": NEURAL_MS, "learning": True},
            "provenance": summary["provenance"],
            "clean_checkpoint_sha256": clean_hash, "reinforcement": "none",
            "market_execution": False, "reinforcement_delivered": False,
        })
        active = None
        epoch = 0
        events: list[dict[str, Any]] = []
        for index, date in enumerate(dates):
            if date in records:
                active = records[date]
                epoch += 1
            if active is None:
                raise ValueError(f"no carried candidate basket for {date}")
            frame = (frame_factory or _frame)(active, stream.prices, date, stream.base.calendar, ledger, D("1.0"), D("0"))
            frame = np.asarray(frame)
            if frame.shape != (180, 320, 3) or frame.dtype != np.uint8:
                raise ValueError(f"invalid generated frame at {date}")
            frame_hash = hashlib.sha256(frame.tobytes()).hexdigest()
            neural = controller.observe(frame, "none")
            if str(neural.get("input_sha256")) != frame_hash:
                raise ValueError(f"neural input hash mismatch at {date}")
            brain = controller.brain
            counts = np.asarray(brain.counts, dtype=np.int32)
            reward_indices = np.asarray(brain.circuit["reward"], dtype=np.int64)
            aversive_indices = np.asarray(brain.circuit["aversive"], dtype=np.int64)
            kc_indices = np.asarray(brain.circuit["kc"], dtype=np.int64)
            reward_mask = _eligible_mask(controller, "reward")
            aversive_mask = _eligible_mask(controller, "aversive")
            event = {
                "schema": summary["schema"], "index": index, "date": date,
                "candidate_day": date in records, "active_candidate": _active_payload(active, epoch),
                "frame_sha256": frame_hash, "neural_input_sha256": str(neural["input_sha256"]),
                "spike_sha256": str(neural["spike_sha256"]), "KC_spikes": int(counts[kc_indices].sum()),
                "reward_spikes": int(counts[reward_indices].sum()), "aversive_spikes": int(counts[aversive_indices].sum()),
                "KC_cell_counts": counts[kc_indices].tolist(), "reward_cell_counts": counts[reward_indices].tolist(),
                "aversive_cell_counts": counts[aversive_indices].tolist(),
                "reward_eligible_edge_count": int(np.count_nonzero(reward_mask)),
                "aversive_eligible_edge_count": int(np.count_nonzero(aversive_mask)),
                "reinforcement": "none", "reinforcement_delivered": False, "market_execution": False,
            }
            log.append("observation", event)
            events.append(event)
        end = output / "brain-end.npz"
        controller.save(end)
        end_hash = sha256(end)
        plastic_end = _plastic_state_digests(controller.brain)
        reward_counts = [int(item["reward_eligible_edge_count"]) for item in events]
        aversive_counts = [int(item["aversive_eligible_edge_count"]) for item in events]
        candidate_events = [item for item in events if item["candidate_day"]]
        checks = {
            "observation_count_expected": len(events) == expected_dates,
            "candidate_day_count_expected": len(candidate_events) == expected_signals,
            "chronological": [item["date"] for item in events] == list(dates),
            "all_reinforcement_none": all(item["reinforcement"] == "none" and not item["reinforcement_delivered"] for item in events),
            "no_market_execution": all(item["market_execution"] is False for item in events),
            "frame_hashes_match_neural": all(item["frame_sha256"] == item["neural_input_sha256"] for item in events),
            "plastic_edge_weights_unchanged": plastic_start["edge_weights_sha256"] == plastic_end["edge_weights_sha256"],
            "memory_u_unchanged": plastic_start["memory_u_sha256"] == plastic_end["memory_u_sha256"],
            "memory_w_unchanged": plastic_start["memory_w_sha256"] == plastic_end["memory_w_sha256"],
        }
        summary.update({
            "input_gate": input_gate, "observation_window": [dates[0], dates[-1]],
            "observation_count": len(events), "candidate_day_count": len(candidate_events),
            "expected_counts": {"observations": expected_dates, "candidate_days": expected_signals},
            "candidate_dates": [item["date"] for item in candidate_events],
            "start_checkpoint_sha256": clean_hash, "end_checkpoint_sha256": end_hash,
            "plastic_state_start": plastic_start, "plastic_state_end": plastic_end,
            "reward_eligibility": {"min": min(reward_counts), "max": max(reward_counts), "days_nonzero": sum(value > 0 for value in reward_counts), "candidate_days_nonzero": sum(item["reward_eligible_edge_count"] > 0 for item in candidate_events), "classification": "ZERO_ALL_242" if not any(reward_counts) else "NONZERO_COVERAGE"},
            "aversive_eligibility": {"min": min(aversive_counts), "max": max(aversive_counts), "days_nonzero": sum(value > 0 for value in aversive_counts), "candidate_days_nonzero": sum(item["aversive_eligible_edge_count"] > 0 for item in candidate_events)},
            "checks": checks, "status": "COMPLETE" if all(checks.values()) else "FAILED",
        })
        log.append("coverage_complete", {"status": summary["status"], "checks": checks, "start_checkpoint_sha256": clean_hash, "end_checkpoint_sha256": end_hash, "plastic_state_start": plastic_start, "plastic_state_end": plastic_end, "reward_eligibility": summary["reward_eligibility"], "aversive_eligibility": summary["aversive_eligibility"]})
    except Exception as exc:
        summary.update({"error_type": type(exc).__name__, "error": str(exc), "status": "FAILED"})
        log.append("coverage_failure", {"error_type": type(exc).__name__, "error": str(exc)})
    summary["runtime_seconds"] = round(time.monotonic() - started, 3)
    summary["log"] = {"file": "coverage-events.jsonl", **verify_append_only_log(output / "coverage-events.jsonl")}
    _atomic_json(output / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_coverage_audit(args.out, args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
