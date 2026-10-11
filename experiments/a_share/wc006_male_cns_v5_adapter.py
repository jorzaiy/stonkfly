"""WC006 + Fly V5 paper adapter.

V5 is intentionally separate from the completed V4 adapter.  It keeps the
full MaleCNS graph and WC006 decoder, while making signal-to-fill attribution
and requested/effective exposure state explicit and resumable.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import json
import re
import sqlite3
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from stonkfly.config import Settings
from stonkfly.neural.controller import FlyController, deterministic_neural_payload

from experiments.a_share.replay_wc006 import CostConfig, D, WC006Ledger, incremental_feedback, map_decoder_to_multiplier, sha256
from experiments.a_share.wc006_male_cns_v4_adapter import (
    V4Stream,
    _equity,
    _frame,
    _prepare,
    _settle_v4,
    _adapter_sha256 as _v4_adapter_sha256,
    assess_action_coverage,
    validate_v4_stream,
)
from experiments.a_share.replay_wc006 import source_manifest_hashes
from experiments.a_share.wc006_v4_market import quarantine_run

V5_SCHEMA = "wc006-male-cns-paper/v5"
RISK_BUDGET = D("0.85")
TRAIN_END = "2023-12-29"
V5_ARMS = ("attribution-only", "state-only", "combined")
_ATTRIBUTION_ARMS = frozenset(("attribution-only", "combined"))
_STATE_ARMS = frozenset(("state-only", "combined"))
_COMPARISON_ARM = "WC006+Fly"
_PAIRED_BASE_ARM = "WC006-only"
V5_FORMAL_AUTHORIZATION_SHA256 = "1d957c2244460b0d6fdebf71a7720047bcb7e6767637d676e39ec8d67f55eaec"
V5_PREREG_SHA256 = "6ab3f8d024a798371abb48b40ac751f2cea6038cf011bcbf26173d2c82563670"

def _canonical_adapter_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if path.resolve() == Path(__file__).resolve():
        text = re.sub(
            r'V5_FORMAL_AUTHORIZATION_SHA256 = "[0-9a-f]{64}"',
            'V5_FORMAL_AUTHORIZATION_SHA256 = "' + ('0' * 64) + '"',
            text,
        )
    return hashlib.sha256(text.encode()).hexdigest()



def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("V5 seed must be a non-negative integer")
    return seed


def _assert_no_unknown(ledger: WC006Ledger, label: str) -> None:
    unresolved = ledger.unresolved()
    if unresolved:
        raise RuntimeError(f"{label}: unresolved UNKNOWN intents block V5 progress: {unresolved}")


def _runtime_compute_seconds(neural: dict[str, Any], controller: Any) -> float:
    """Read wall-clock timing for the non-hash telemetry sidecar only."""
    raw = neural.get("compute_seconds", getattr(controller, "last_compute_seconds", 0.0))
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("V5 runtime telemetry compute_seconds is not numeric") from exc
    if not math.isfinite(value) or value < 0:
        raise RuntimeError("V5 runtime telemetry compute_seconds must be finite and non-negative")
    return value


def _load_runtime_metrics(path: Path, *, arm: str, seed: int, dates: list[str], last_day: int) -> list[dict[str, Any]]:
    """Validate the append-only sidecar before continuing a resumed run."""
    if not path.exists():
        if last_day >= 0:
            raise RuntimeError("V5 runtime telemetry missing for completed days")
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise RuntimeError(f"V5 runtime telemetry blank line at {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"V5 runtime telemetry invalid JSON at line {line_number}") from exc
        if not isinstance(row, dict) or set(row) != {"arm", "seed", "date", "index", "compute_seconds"}:
            raise RuntimeError(f"V5 runtime telemetry schema mismatch at line {line_number}")
        index = row["index"]
        if isinstance(index, bool) or not isinstance(index, int) or index != len(rows) or index < 0 or index >= len(dates):
            raise RuntimeError(f"V5 runtime telemetry index is non-contiguous at line {line_number}")
        if row["arm"] != arm or row["seed"] != seed or row["date"] != dates[index]:
            raise RuntimeError(f"V5 runtime telemetry identity mismatch at line {line_number}")
        try:
            seconds = float(row["compute_seconds"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"V5 runtime telemetry timing is not numeric at line {line_number}") from exc
        if not math.isfinite(seconds) or seconds < 0:
            raise RuntimeError(f"V5 runtime telemetry timing is invalid at line {line_number}")
        rows.append(row)
    if len(rows) != last_day + 1:
        raise RuntimeError("V5 runtime telemetry length does not match completed ledger days")
    return rows


def _append_runtime_metric(path: Path, *, arm: str, seed: int, date: str, index: int, compute_seconds: float) -> None:
    row = {"arm": arm, "seed": seed, "date": date, "index": index, "compute_seconds": compute_seconds}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _graph_source_lock(fly: Any) -> dict[str, Any]:
    signature = fly.brain.configuration_signature()
    report = getattr(getattr(fly, "brain", None), "circuit", {}).get("report", {})
    complete = (
        report.get("release") == "MaleCNS v1.0"
        and report.get("selection") == "All existing KC-to-MBON07/11 connections; no graph cropping or added edges."
        and bool(report.get("plastic_edges"))
    )
    return {
        "full_retained_graph": complete,
        "release": report.get("release"),
        "selection": report.get("selection"),
        "configuration_sha256": hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest(),
        "plastic_edges": report.get("plastic_edges"),
        "plastic_edges_sha256": report.get("plastic_edges_sha256"),
        "source_hashes": source_manifest_hashes(),
    }


def _guard_reasons(ledger: WC006Ledger, date: str) -> list[str]:
    return sorted({str(row[0]) for row in ledger.db.execute(
        "SELECT veto_reason FROM intents WHERE execution_date=? AND status='VETO' AND veto_reason IS NOT NULL", (date,)
    )})


def _assert_no_pending_linkage(ledger: WC006Ledger, label: str, *, require_rewards: bool = True) -> None:
    if not _table_exists(ledger, "v5_signal_links"):
        return
    pending = ledger.db.execute("SELECT signal_id FROM v5_signal_links WHERE outcome='PENDING'").fetchall()
    unrewarded = ledger.db.execute("SELECT signal_id FROM v5_signal_links WHERE outcome!='PENDING' AND reward_json IS NULL").fetchall()
    if pending or (require_rewards and unrewarded):
        raise RuntimeError(f"{label}: unresolved V5 signal/state/reward linkage")


def _table_exists(ledger: WC006Ledger, name: str) -> bool:
    return ledger.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _adapter_sha256() -> str:
    return sha256(Path(__file__).resolve())


def signal_id(signal_date: str, record_index: int) -> str:
    return hashlib.sha256(json.dumps({"signal_date": signal_date, "record_index": record_index}, sort_keys=True).encode()).hexdigest()


def _validate_arm(arm: str) -> str:
    arm = str(arm)
    if arm not in V5_ARMS:
        raise ValueError(f"unsupported V5 arm: {arm!r}; expected one of {', '.join(V5_ARMS)}")
    return arm


def _formal_output_scope(output: Path, arm: str) -> dict[str, str]:
    """Require formal runs to stay under a new, arm-specific paper directory."""
    root = Path(__file__).resolve().parents[2]
    runs = root / "runs"
    output = Path(output).resolve()
    try:
        relative = output.relative_to(runs)
    except ValueError as exc:
        raise RuntimeError("formal V5 output must be below runs/wc006_v5_<arm>") from exc
    expected = f"wc006_v5_{arm}"
    if len(relative.parts) < 2 or relative.parts[0] != expected:
        raise RuntimeError(f"formal V5 output must be below runs/{expected}/<run>")
    return {"root": str(root), "relative": str(relative), "directory": expected}


def validate_v5_inputs(
    stream_path: Path,
    *,
    expected_records: int = 46,
    training_end: str = TRAIN_END,
    market_manifest: Path | None = None,
    formal: bool = False,
) -> tuple[V4Stream, dict[str, Any]]:
    """Validate frozen causal inputs and expose auditable gate evidence."""
    market_pin: dict[str, str] = {}
    authorization_path: Path | None = None
    if formal:
        if market_manifest is None:
            raise RuntimeError("V5 input gate: formal authorization artifact is required")
        market_path = Path(market_manifest).resolve()
        market_payload = json.loads(market_path.read_text(encoding="utf-8"))
        auth_rel = market_payload.get("authorization_artifact")
        if not isinstance(auth_rel, str) or not auth_rel:
            raise RuntimeError("V5 input gate: training_authorized is not true; development authorization artifact is missing from market manifest")
        authorization_path = (market_path.parent / auth_rel).resolve()
        if sha256(authorization_path) != V5_FORMAL_AUTHORIZATION_SHA256:
            raise RuntimeError("V5 input gate: authorization artifact hash is not pinned")
        authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
        if authorization.get("training_authorized") is not True or authorization.get("status") != "APPROVED_DEVELOPMENT_ONLY":
            raise RuntimeError("V5 input gate: authorization decision is not approved")
        if authorization.get("paper_only") is not True or authorization.get("live_execution_authorized") is not False or authorization.get("broker_orders_enabled") is not False:
            raise RuntimeError("V5 input gate: authorization is not paper-only")
        if authorization.get("scope", {}).get("development_only") is not True:
            raise RuntimeError("V5 input gate: authorization is not development-only")
        code_lock = authorization.get("code_lock")
        if not isinstance(code_lock, dict):
            raise RuntimeError("V5 input gate: code lock is missing")
        root = Path(__file__).resolve().parents[2]
        prereg = root / "docs/experiments/wc006_v5_preregister.md"
        if sha256(prereg) != code_lock.get("v5_preregister_sha256") or code_lock.get("v5_preregister_sha256") != V5_PREREG_SHA256:
            raise RuntimeError("V5 input gate: V5 preregistration hash mismatch")
        if _canonical_adapter_sha256(Path(__file__).resolve()) != code_lock.get("v5_adapter_sha256"):
            raise RuntimeError("V5 input gate: V5 adapter hash mismatch")
        v4_path = root / "experiments/a_share/wc006_male_cns_v4_adapter.py"
        if sha256(v4_path) != code_lock.get("v4_adapter_sha256"):
            raise RuntimeError("V5 input gate: V4 adapter hash mismatch")
        controller_path = root / "stonkfly/neural/controller.py"
        if sha256(controller_path) != code_lock.get("controller_sha256"):
            raise RuntimeError("V5 input gate: controller hash mismatch")
        if source_manifest_hashes() != code_lock.get("graph_source_lock", {}).get("source_hashes"):
            raise RuntimeError("V5 input gate: graph source lock hash mismatch")
        pinned_manifest = str(authorization.get("market_manifest", {}).get("sha256", ""))
        pinned_qa = str(authorization.get("quality_report", {}).get("sha256", ""))
        if pinned_manifest != sha256(market_path):
            raise RuntimeError("V5 input gate: authorization/market manifest hash mismatch")
        qa_path = market_path.parent / str(authorization.get("quality_report", {}).get("path", ""))
        if not qa_path.is_file() or pinned_qa != sha256(qa_path):
            raise RuntimeError("V5 input gate: authorization/QA hash mismatch")
        market_pin = {"parent": pinned_manifest, "qa": pinned_qa}
        authorization_code_lock = code_lock
        fixed_run_plan = authorization.get("fixed_run_plan", {})
        if fixed_run_plan.get("seed") != 17 or fixed_run_plan.get("arms_in_order") != list(V5_ARMS):
            raise RuntimeError("V5 input gate: fixed seed/arm order mismatch")
    stream = validate_v4_stream(
        stream_path,
        expected_records=expected_records,
        training_end=training_end,
        market_manifest=market_manifest,
        market_expected_parent_sha256=market_pin.get("parent"),
        market_expected_qa_sha256=market_pin.get("qa"),
    )
    candidate = json.loads(Path(stream_path).read_text(encoding="utf-8"))
    if candidate.get("paper_only") is not True or candidate.get("broker_orders_enabled") is not False or candidate.get("live_execution_authorized") is not False:
        raise RuntimeError("V5 input gate: candidate stream is not paper-only")
    market_hashes: dict[str, str] = {}
    training_authorized = False
    if market_manifest is not None:
        market_path = Path(market_manifest).resolve()
        market_payload = json.loads(market_path.read_text(encoding="utf-8"))
        if market_payload.get("broker_orders_enabled") is not False or market_payload.get("live_execution_authorized") is not False:
            raise RuntimeError("V5 input gate: market manifest is not paper-only")
        training_authorized = market_payload.get("training_authorized") is True
        if formal and not training_authorized:
            raise RuntimeError("V5 input gate: training_authorized is not true")
        market_hashes = {
            "manifest": sha256(market_path),
            "qa": str(stream.market.qa_sha256) if stream.market is not None else "",
        }
        if authorization_path is not None:
            market_hashes["authorization"] = sha256(authorization_path)
        if stream.market is not None:
            market_hashes.update({str(k): str(v) for k, v in stream.market.artifact_hashes.items()})
    return stream, {
        "status": "PASS",
        "candidate_manifest_sha256": sha256(Path(stream_path)),
        "market_manifest_sha256": market_hashes.get("manifest"),
        "market_qa_sha256": market_hashes.get("qa"),
        "market_artifact_hashes": market_hashes,
        "training_authorized": training_authorized,
        "formal": formal,
        "records": len(stream.base.records),
        "calendar_days": len(stream.base.calendar),
        "training_end": training_end,
        "paper_only": True,
        "broker_orders_enabled": False,
        "live_execution_authorized": False,
        "authorization_code_lock": locals().get("authorization_code_lock"),
        "fixed_run_plan": locals().get("fixed_run_plan"),
        "input_data_window": (locals().get("fixed_run_plan") or {}).get("input_data_window"),
        "training_window": [stream.base.records[0].signal_date, training_end],
    }


def _ensure_links(ledger: WC006Ledger) -> None:
    ledger.db.execute(
        """CREATE TABLE IF NOT EXISTS v5_signal_links(
          signal_id TEXT PRIMARY KEY, signal_date TEXT NOT NULL,
          execution_date TEXT NOT NULL, requested_multiplier TEXT NOT NULL,
          effective_multiplier TEXT NOT NULL, intent_ids_json TEXT NOT NULL,
          outcome TEXT NOT NULL, pending_age INTEGER NOT NULL,
          reward_json TEXT, UNIQUE(signal_date))"""
    )
    ledger.db.commit()


def _link_signal(ledger: WC006Ledger, *, signal: str, signal_date: str, execution_date: str, requested: Decimal, intent_ids: list[str], pending_age: int) -> None:
    _ensure_links(ledger)
    ledger.db.execute(
        "INSERT OR REPLACE INTO v5_signal_links VALUES (?,?,?,?,?,?,?, ?, NULL)",
        (signal, signal_date, execution_date, str(requested), str(requested), json.dumps(intent_ids, sort_keys=True), "PENDING", pending_age),
    )
    ledger.db.commit()


def _intent_statuses(ledger: WC006Ledger, intent_ids: list[str]) -> list[str]:
    if not intent_ids:
        return []
    marks = ",".join("?" for _ in intent_ids)
    return [str(row[0]) for row in ledger.db.execute(f"SELECT status FROM intents WHERE id IN ({marks})", tuple(intent_ids))]


def _ensure_state_batches(ledger: WC006Ledger) -> None:
    ledger.db.execute(
        """CREATE TABLE IF NOT EXISTS v5_state_batches(
          signal_date TEXT PRIMARY KEY, execution_date TEXT NOT NULL,
          requested_multiplier TEXT NOT NULL, intent_ids_json TEXT NOT NULL,
          outcome TEXT NOT NULL, pending_age INTEGER NOT NULL)"""
    )
    ledger.db.commit()


def _link_state_batch(ledger: WC006Ledger, *, signal_date: str, execution_date: str, requested: Decimal, intent_ids: list[str], pending_age: int) -> None:
    _ensure_state_batches(ledger)
    ledger.db.execute(
        "INSERT OR REPLACE INTO v5_state_batches VALUES (?,?,?,?,?,?)",
        (signal_date, execution_date, str(requested), json.dumps(intent_ids, sort_keys=True), "PENDING", pending_age),
    )
    ledger.db.commit()


def resolve_state_batches(ledger: WC006Ledger, execution_date: str, current_effective: Decimal) -> tuple[Decimal, list[dict[str, Any]]]:
    """Commit requested state only after an exact T+1 batch has a fill."""
    _ensure_state_batches(ledger)
    rows = ledger.db.execute(
        "SELECT signal_date,execution_date,requested_multiplier,intent_ids_json,pending_age FROM v5_state_batches WHERE execution_date=? AND outcome='PENDING' ORDER BY signal_date",
        (execution_date,),
    ).fetchall()
    effective = current_effective
    resolved: list[dict[str, Any]] = []
    for signal_date, execution, requested_raw, ids_raw, age in rows:
        ids = list(json.loads(ids_raw))
        statuses = _intent_statuses(ledger, ids)
        outcome = "FILLED" if "FILLED" in statuses else ("VETO" if statuses else "EMPTY")
        requested = D(requested_raw)
        if outcome == "FILLED":
            effective = requested
        ledger.db.execute("UPDATE v5_state_batches SET outcome=? WHERE signal_date=?", (outcome, signal_date))
        resolved.append({"signal_id": None, "signal_date": signal_date, "execution_date": execution, "outcome": outcome, "intent_statuses": statuses, "pending_age": int(age)})
    ledger.put_meta("effective_multiplier", str(effective))
    ledger.db.commit()
    return effective, resolved


def resolve_signal_outcomes(
    ledger: WC006Ledger,
    execution_date: str,
    current_effective: Decimal,
    *,
    commit_effective: bool = True,
) -> tuple[Decimal, list[dict[str, Any]]]:
    """Resolve exact T+1 batches; optionally leave V4 state untouched."""
    _ensure_links(ledger)
    rows = ledger.db.execute(
        "SELECT signal_id,signal_date,requested_multiplier,intent_ids_json,pending_age FROM v5_signal_links WHERE execution_date=? AND outcome='PENDING' ORDER BY signal_id",
        (execution_date,),
    ).fetchall()
    effective = current_effective
    resolved: list[dict[str, Any]] = []
    for sid, signal_date, requested_raw, ids_raw, age in rows:
        ids = list(json.loads(ids_raw))
        statuses = _intent_statuses(ledger, ids)
        outcome = "FILLED" if "FILLED" in statuses else ("VETO" if statuses else "EMPTY")
        requested = D(requested_raw)
        if outcome == "FILLED":
            effective = requested
        ledger.db.execute("UPDATE v5_signal_links SET effective_multiplier=?,outcome=? WHERE signal_id=?", (str(effective), outcome, sid))
        resolved.append({"signal_id": sid, "signal_date": signal_date, "execution_date": execution_date, "outcome": outcome, "intent_statuses": statuses, "pending_age": int(age)})
    if commit_effective:
        ledger.put_meta("effective_multiplier", str(effective))
    ledger.db.commit()
    return effective, resolved


def attach_settled_reward(ledger: WC006Ledger, execution_date: str, feedback: dict[str, Any]) -> list[str]:
    """Attach the post-settlement, same-day reward to exact signal links."""
    _ensure_links(ledger)
    reward = {key: str(value) if isinstance(value, Decimal) else value for key, value in feedback.items()}
    rows = ledger.db.execute("SELECT signal_id FROM v5_signal_links WHERE execution_date=? AND reward_json IS NULL AND outcome!='PENDING' ORDER BY signal_id", (execution_date,)).fetchall()
    ids = [str(row[0]) for row in rows]
    for sid in ids:
        ledger.db.execute("UPDATE v5_signal_links SET reward_json=? WHERE signal_id=?", (json.dumps(reward, sort_keys=True), sid))
    ledger.db.commit()
    return ids


def _gross_exposure(ledger: WC006Ledger, stream: V4Stream, date: str, active: set[str], equity: Decimal) -> Decimal:
    if equity <= 0:
        return D("0")
    total = D("0")
    held = {str(row[0]) for row in ledger.db.execute("SELECT DISTINCT symbol FROM lots WHERE CAST(quantity AS REAL)>0")}
    if stream.market is not None:
        from experiments.a_share.wc006_v4_market import raw_mark
        for code in sorted(set(active) | held):
            total += ledger.holdings(code) * raw_mark(stream.market, code, date)[1]
    else:
        for code in sorted(set(active) | held):
            available = [item for item in stream.prices.get(code, {}) if item <= date]
            if available:
                total += ledger.holdings(code) * D(stream.prices[code][max(available)]["close"])
    return total / equity


def _fees_on_date(ledger: WC006Ledger, date: str) -> Decimal:
    return sum((D(str(row[0])) for row in ledger.db.execute("SELECT fee FROM fills WHERE execution_date=?", (date,))), D("0"))


def _sqlite_dump(path: Path) -> bytes:
    db = sqlite3.connect(path)
    try:
        return "\n".join(db.iterdump()).encode("utf-8")
    finally:
        db.close()


def _canonical_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _linkage_snapshot(output: Path) -> dict[str, Any]:
    db = sqlite3.connect(output / "ledger.sqlite")
    try:
        unknown = [row[0] for row in db.execute("SELECT id FROM intents WHERE status='UNKNOWN' ORDER BY id")]
        links = []
        if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v5_signal_links'").fetchone():
            for row in db.execute("SELECT signal_id,signal_date,execution_date,intent_ids_json,outcome,reward_json FROM v5_signal_links ORDER BY signal_id"):
                ids = list(json.loads(row[3]))
                existing = [item[0] for item in db.execute("SELECT id FROM intents WHERE id IN (%s)" % ",".join("?" for _ in ids), tuple(ids))] if ids else []
                links.append({"signal_id": row[0], "signal_date": row[1], "execution_date": row[2], "intent_ids": ids, "intent_ids_exist": sorted(existing) == sorted(ids), "outcome": row[4], "reward": row[5]})
        events = []
        for row in db.execute("SELECT payload_json FROM events ORDER BY rowid"):
            payload = json.loads(row[0])
            if payload.get("signal_eligible"):
                events.append({"date": payload.get("date"), "signal_id": payload.get("signal_id"), "neural": bool(payload.get("neural")), "guard_reason": payload.get("guard_reason")})
        provenance = _canonical_json(output / "provenance.json")
        ablation = provenance.get("ablation") or provenance.get("arm")
        requires_reward = ablation in _ATTRIBUTION_ARMS
        valid = not unknown and all(item["intent_ids_exist"] and item["outcome"] != "PENDING" and (not requires_reward or item["reward"] is not None) for item in links)
        valid = valid and all(item["neural"] and isinstance(item["guard_reason"], list) for item in events)
        return {"links": links, "events": events, "unknown": unknown, "valid": valid}
    finally:
        db.close()


def _normalized_provenance(output: Path) -> dict[str, Any]:
    data = _canonical_json(output / "provenance.json")
    data.pop("output_scope", None)
    return data


def _normalized_quarantine(path: Path) -> dict[str, Any]:
    data = _canonical_json(path)
    # quarantine_run records the run directory for operator context; resume
    # parity compares the immutable semantic decision, not that path.
    data.pop("run", None)
    return data


def audit_v5_resume_parity(full_output: Path, resumed_output: Path) -> dict[str, Any]:
    """Compare state, clean brain, gates, provenance and linkage on resume."""
    full_output, resumed_output = Path(full_output), Path(resumed_output)
    checks: dict[str, bool] = {}
    for name in ("ledger.sqlite", "paired_base.sqlite"):
        checks[f"{name}:ledger"] = _sqlite_dump(full_output / name) == _sqlite_dump(resumed_output / name)
    checks["events.jsonl"] = (full_output / "events.jsonl").read_bytes() == (resumed_output / "events.jsonl").read_bytes()
    for name in ("brain-clean.npz", "brain-day-0.npz", "brain-day-1.npz"):
        checks[name] = (full_output / name).read_bytes() == (resumed_output / name).read_bytes()
    full_prov, resumed_prov = _normalized_provenance(full_output), _normalized_provenance(resumed_output)
    checks["provenance"] = full_prov == resumed_prov
    checks["gate_metadata"] = all(full_prov.get(key) == resumed_prov.get(key) for key in ("input_gate", "graph_source_lock", "seed", "arm", "comparison_arm", "ablation", "paired_base_arm", "final_coverage", "final_settlement"))
    full_q, resumed_q = full_output / "QUARANTINED.json", resumed_output / "QUARANTINED.json"
    checks["quarantine"] = (full_q.exists() == resumed_q.exists()) and (not full_q.exists() or _normalized_quarantine(full_q) == _normalized_quarantine(resumed_q))
    full_link, resumed_link = _linkage_snapshot(full_output), _linkage_snapshot(resumed_output)
    checks["linkage"] = full_link == resumed_link and full_link["valid"]
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def audit_v5_isolation(output: Path, *, arm: str | None = None, formal: bool = True) -> dict[str, Any]:
    """Check provenance and output scope without opening any broker interface."""
    output = Path(output).resolve()
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    expected_arm = _validate_arm(arm or provenance.get("arm"))
    if formal:
        scope = _formal_output_scope(output, expected_arm)
        scope_ok = scope["directory"] == f"wc006_v5_{expected_arm}"
    else:
        scope = provenance.get("output_scope", {"formal": False})
        scope_ok = scope.get("formal") is False
    input_gate = provenance.get("input_gate") or {}
    training_authorized = provenance.get("training_authorized")
    checks = {
        "paper_only": provenance.get("paper_only") is True,
        "broker_orders_disabled": provenance.get("broker_orders_enabled") is False,
        "live_execution_disabled": provenance.get("live_execution_authorized") is False,
        "clean_brain": provenance.get("initial_brain") == "clean/new; no historical checkpoint",
        "arm_bound": provenance.get("arm") == expected_arm and provenance.get("ablation", expected_arm) == expected_arm,
        "comparison_arm": provenance.get("comparison_arm") == _COMPARISON_ARM,
        "paired_base_arm": provenance.get("paired_base_arm") == _PAIRED_BASE_ARM,
        "training_authorized_mode": training_authorized is (True if formal else False),
        "training_authorized_gate_consistent": input_gate.get("training_authorized") is training_authorized,
        "output_scope": scope_ok,
        "no_quarantine": not (output / "QUARANTINED.json").exists(),
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "scope": scope}

def _run_v5(
    stream_path: Path,
    output: Path,
    *,
    formal: bool,
    max_days: int,
    expected_records: int,
    training_end: str,
    controller_factory: Callable[[Settings], Any],
    market_manifest: Path | None,
    arm: str,
    seed: int,
) -> dict[str, Any]:
    arm = _validate_arm(arm)
    seed = _validate_seed(seed)
    if formal and (expected_records != 46 or training_end != TRAIN_END or market_manifest is None):
        raise RuntimeError("formal V5 requires frozen 46-record input, training_end and market manifest")
    output = Path(output).resolve()
    if formal and (output / "QUARANTINED.json").exists():
        raise RuntimeError("formal V5 output is quarantined")
    stream, input_gate = validate_v5_inputs(
        stream_path, expected_records=expected_records, training_end=training_end,
        market_manifest=market_manifest, formal=formal,
    )
    if formal and seed != (input_gate.get("fixed_run_plan") or {}).get("seed"):
        raise RuntimeError("formal V5 fixed development seed must be 17")
    scope = _formal_output_scope(output, arm) if formal else {"formal": False, "relative": str(output)}
    if not formal and expected_records == 46:
        raise RuntimeError("fixture API cannot run formal 46-record output")
    if output.exists() and not (output / "ledger.sqlite").exists():
        existing = {item.name for item in output.iterdir()}
        if existing - {"training.log"}:
            raise RuntimeError("V5 output must be new or launcher-log-only")
    output.mkdir(parents=True, exist_ok=True)
    fly = controller_factory(Settings(learning=True))
    graph_lock = _graph_source_lock(fly)
    if formal and not graph_lock["full_retained_graph"]:
        raise RuntimeError("formal V5 requires complete retained MaleCNS v1.0 graph")
    graph_hash = graph_lock["configuration_sha256"]
    if formal:
        expected_graph_lock = (input_gate.get("authorization_code_lock") or {}).get("graph_source_lock") or {}
        for key in ("full_retained_graph", "release", "selection", "configuration_sha256", "plastic_edges", "plastic_edges_sha256", "source_hashes"):
            if graph_lock.get(key) != expected_graph_lock.get(key):
                raise RuntimeError(f"V5 graph source lock mismatch: {key}")
    costs = CostConfig()
    adapter_hash = _adapter_sha256()
    signature_payload = {
        "schema": V5_SCHEMA, "arm": arm, "comparison_arm": _COMPARISON_ARM, "ablation": arm,
        "paired_base_arm": _PAIRED_BASE_ARM, "seed": seed,
        "stream": stream.base.stream_sha256, "spec": stream.spec_sha256,
        "graph_source_lock": graph_lock, "v4_adapter": _v4_adapter_sha256(),
        "adapter": adapter_hash, "training_end": training_end,
        "risk_budget": str(RISK_BUDGET), "costs": costs.as_json(),
        "market_artifact_hashes": input_gate.get("market_artifact_hashes", {}),
    }
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
    ledger = WC006Ledger(output / "ledger.sqlite", initial_state={"cash": "100000", "lots": []}, signature=signature, authoritative_trading_state=stream.market is not None)
    base = WC006Ledger(output / "paired_base.sqlite", initial_state={"cash": "100000", "lots": []}, signature=signature + ":paired-base", authoritative_trading_state=stream.market is not None)
    use_attribution = arm in _ATTRIBUTION_ARMS
    use_state = arm in _STATE_ARMS
    try:
        _assert_no_unknown(ledger, "V5 resume primary")
        _assert_no_unknown(base, "V5 resume paired")
        last_raw = ledger.get_meta("last_completed_index")
        base_last_raw = base.get_meta("last_completed_index")
        last_day = -1 if last_raw is None else int(last_raw)
        if last_day != (-1 if base_last_raw is None else int(base_last_raw)):
            raise RuntimeError("V5 primary/paired indices diverge")
        clean = output / "brain-clean.npz"
        checkpoint = ledger.get_meta("checkpoint")
        if checkpoint:
            if not clean.is_file() or sha256(clean) != ledger.get_meta("initial_brain_sha256"):
                raise RuntimeError("V5 initial brain hash mismatch on resume")
            checkpoint_path = output / checkpoint["file"]
            if not checkpoint_path.is_file() or sha256(checkpoint_path) != checkpoint["sha256"]:
                raise RuntimeError("V5 checkpoint mismatch")
            fly.restore(checkpoint_path)
        else:
            fly.save(clean)
            ledger.put_meta("initial_brain_sha256", sha256(clean))
            ledger.put_meta("initial_brain_source", "new FlyController; no historical checkpoint")
            base.put_meta("initial_brain_sha256", None)
            if use_state:
                ledger.put_meta("effective_multiplier", "1.0")
            ledger.db.commit(); base.db.commit()
        initial_brain_hash = str(ledger.get_meta("initial_brain_sha256"))
        if not initial_brain_hash or sha256(clean) != initial_brain_hash:
            raise RuntimeError("V5 initial brain hash is missing or changed")
        provenance_path = output / "provenance.json"
        static_provenance = {
            "schema": V5_SCHEMA, "arm": arm, "comparison_arm": _COMPARISON_ARM, "ablation": arm,
            "comparison_arms": [_PAIRED_BASE_ARM, _COMPARISON_ARM], "ablation_arms": list(V5_ARMS),
            "paired_base_arm": _PAIRED_BASE_ARM, "descriptive_arms": {"Fly-only": "not executed"}, "seed": seed,
            "stream_sha256": stream.base.stream_sha256, "spec_sha256": stream.spec_sha256,
            "graph_manifest_sha256": graph_hash, "graph_source_lock": graph_lock,
            "adapter_sha256": adapter_hash, "v4_adapter_sha256": _v4_adapter_sha256(),
            "initial_brain": "clean/new; no historical checkpoint", "initial_brain_sha256": initial_brain_hash,
            "paper_only": True, "broker_orders_enabled": False, "live_execution_authorized": False,
            "training_authorized": input_gate["training_authorized"],
            "input_data_window": input_gate.get("input_data_window"),
            "training_window": input_gate.get("training_window"),
            "fixed_run_plan": input_gate.get("fixed_run_plan"),
            "price_layers": {"observation": "pinned qfq cache", "execution_mark_reward": "xysz raw Kline"},
            "training_window": [stream.base.records[0].signal_date, training_end],
            "input_gate": input_gate, "output_scope": scope,
            "risk_budget": str(RISK_BUDGET), "costs": costs.as_json(),
            "final_coverage": None, "final_settlement": None,
            "runtime_telemetry": {"path": "runtime_metrics.jsonl", "schema": "wc006-v5-runtime-metrics/v1", "hash_bound": False},
        }
        if not provenance_path.exists():
            provenance_path.write_text(json.dumps(static_provenance, indent=2, sort_keys=True) + "\n")
        else:
            existing = json.loads(provenance_path.read_text(encoding="utf-8"))
            for key, expected in static_provenance.items():
                if key in {"final_coverage", "final_settlement", "output_scope"}:
                    continue
                if existing.get(key) != expected:
                    raise RuntimeError(f"V5 provenance mismatch: {key}")
            if existing.get("final_coverage") is not None and existing.get("final_settlement") is None:
                raise RuntimeError("V5 provenance final coverage/settlement linkage is incomplete")
            if existing.get("final_coverage") is not None and ledger.get_meta("action_coverage") != existing.get("final_coverage"):
                raise RuntimeError("V5 provenance final coverage mismatch on resume")
            if existing.get("final_settlement") is not None and ledger.get_meta("final_settlement") != existing.get("final_settlement"):
                raise RuntimeError("V5 provenance final settlement mismatch on resume")
        records = {item.signal_date: (idx, item) for idx, item in enumerate(stream.base.records)}
        active = None
        dates = [item for item in stream.base.calendar if stream.base.records[0].signal_date <= item <= training_end]
        settlement_date = stream.base.records[-1].execution_date if stream.base.records[-1].execution_date > training_end else None
        events_path = output / "events.jsonl"
        started = 0
        runtime_metrics_path = output / "runtime_metrics.jsonl"
        runtime_metrics = _load_runtime_metrics(runtime_metrics_path, arm=arm, seed=seed, dates=dates, last_day=last_day)
        for day_index, date in enumerate(dates):
            if day_index <= last_day:
                if date in records:
                    active = records[date][1]
                continue
            _assert_no_unknown(ledger, f"V5 before {date} primary")
            _assert_no_unknown(base, f"V5 before {date} paired")
            prices = stream.market.raw_prices if stream.market is not None else stream.prices
            if stream.market is not None:
                from experiments.a_share.wc006_v4_market import apply_corporate_actions
                apply_corporate_actions(ledger, stream.market, date, stream.base.calendar, stage="start_of_day")
                apply_corporate_actions(base, stream.market, date, stream.base.calendar, stage="start_of_day")
            settled = _settle_v4(ledger, date, stream.base.calendar, prices, costs, stream.market)
            _settle_v4(base, date, stream.base.calendar, prices, costs, stream.market)
            if stream.market is not None:
                from experiments.a_share.wc006_v4_market import apply_corporate_actions
                # Record-date eligibility is based on holdings after open settlement.
                apply_corporate_actions(ledger, stream.market, date, stream.base.calendar, stage="after_open_settlement")
                apply_corporate_actions(base, stream.market, date, stream.base.calendar, stage="after_open_settlement")
            _assert_no_unknown(ledger, f"V5 after {date} settle primary")
            _assert_no_unknown(base, f"V5 after {date} settle paired")
            state_before = D(ledger.get_meta("effective_multiplier") or ledger.get_meta("multiplier") or "1.0")
            resolved_effective, resolved = resolve_signal_outcomes(ledger, date, state_before, commit_effective=use_state)
            # Resolve T-1 first, then re-read the persisted effective state for this observation.
            state_for_observation = D(ledger.get_meta("effective_multiplier") or ledger.get_meta("multiplier") or "1.0")
            if date in records:
                active = records[date][1]
            if active is None:
                raise RuntimeError("V5 observed before first signal")
            active_codes = {item["code"] for item in active.targets}
            previous_equity = D(ledger.get_meta("previous_equity") or "100000")
            previous_base = D(base.get_meta("previous_equity") or "100000")
            if stream.market is not None:
                from experiments.a_share.wc006_v4_market import raw_equity
                equity, fly_marks = raw_equity(ledger, stream.market, date, active_codes)
                base_equity, base_marks = raw_equity(base, stream.market, date, active_codes)
            else:
                equity = _equity(ledger, stream.prices, date, "close", active_codes)
                base_equity = _equity(base, stream.prices, date, "close", active_codes)
                fly_marks = base_marks = {}
            peak = max(previous_equity, D(ledger.get_meta("peak_equity") or previous_equity), equity)
            prior_dd = D(ledger.get_meta("drawdown") or "0")
            drawdown = equity / peak - D("1") if peak else D("0")
            feedback = incremental_feedback(equity=equity, previous_equity=previous_equity, base_equity=base_equity, previous_base_equity=previous_base, drawdown=drawdown, previous_drawdown=prior_dd, turnover_ratio=settled / previous_equity if previous_equity else D("0"))
            rewarded = attach_settled_reward(ledger, date, feedback) if use_attribution else []
            frame = _frame(active, stream.prices, date, stream.base.calendar, ledger, state_for_observation, drawdown)
            neural = fly.observe(frame, str(feedback["stimulus"]))
            compute_seconds = _runtime_compute_seconds(neural, fly)
            action, requested = map_decoder_to_multiplier(str(neural["side"]), state_for_observation)
            v4_next = requested if action != "DEFER" else state_for_observation
            event_effective = resolved_effective if use_state else v4_next
            plan_multiplier = requested
            if use_state:
                ledger.put_meta("effective_multiplier", str(resolved_effective))
            neural.update({"wc006_action": action, "requested_multiplier": str(requested), "effective_multiplier": str(event_effective), "v5_arm": arm, "seed": seed})
            neural_payload = deterministic_neural_payload(neural)
            cp = output / f"brain-day-{day_index % 2}.npz"
            fly.save(cp)
            cp_info = {"file": cp.name, "sha256": sha256(cp)}
            observation_id = ledger.record_observation(date=date, input_sha256=str(neural["input_sha256"]), reinforcement=str(feedback["stimulus"]), neural=neural_payload, checkpoint=cp_info)
            eligible = date in records
            intent_ids: list[str] = []
            signal = None
            fly_plan: list[dict[str, Any]] = []
            base_plan: list[dict[str, Any]] = []
            pending_age = 0
            if eligible:
                index, record = records[date]
                signal = signal_id(date, index)
                pending_age = 1
                if action != "DEFER":
                    fly_plan = _prepare(ledger, record, prices, date, plan_multiplier, str(neural["side"]), observation_id, cp_info["sha256"], stream.base.calendar, stream.market)
                    intent_ids = [str(item["intent_id"]) for item in fly_plan if item.get("intent_id")]
                base_plan = _prepare(base, record, prices, date, D("1.0"), "BUY", 0, "paired-base", stream.base.calendar, stream.market)
                _link_signal(ledger, signal=signal, signal_date=date, execution_date=record.execution_date, requested=requested, intent_ids=intent_ids, pending_age=pending_age)
            target_exposure = sum((D(str(item["weight"])) * RISK_BUDGET for item in active.targets), D("0")) * event_effective
            actual_exposure = _gross_exposure(ledger, stream, date, active_codes, equity)
            event = {"date": date, "arm": arm, "signal_eligible": eligible, "signal_id": signal, "signal_date": date if eligible else None, "execution_date": active.execution_date if eligible else None, "action": action, "neural": neural_payload, "guard_reason": _guard_reasons(ledger, date), "requested_multiplier": str(requested), "effective_multiplier": str(event_effective), "target_exposure": str(target_exposure), "actual_exposure": str(actual_exposure), "pending_age": pending_age, "settled_signal_ids": rewarded, "settled_outcomes": resolved, "reward": {key: str(value) if isinstance(value, Decimal) else value for key, value in feedback.items()}, "fees": str(_fees_on_date(ledger, date)), "turnover": str(settled), "paired_plan": {"fly": fly_plan, "base": base_plan}, "checkpoint": cp_info, "mark_provenance": {"fly": fly_marks, "base": base_marks}}
            event_hash, created = ledger.record_event(date=date, payload=event)
            event["event_hash"] = event_hash
            if created:
                with events_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            base.record_event(date=date, payload={"date": date, "arm": arm, "signal_id": signal, "signal_eligible": eligible, "paired_plan": {"base": base_plan}, "reward": {key: str(value) if isinstance(value, Decimal) else value for key, value in feedback.items()}})
            ledger.complete_date(date=date, index=day_index, multiplier=event_effective, equity=equity, drawdown=drawdown, checkpoint=cp_info)
            base.complete_date(date=date, index=day_index, multiplier=D("1.0"), equity=base_equity, drawdown=D("0"), checkpoint={"paired": hashlib.sha256(json.dumps(base_plan, sort_keys=True).encode()).hexdigest()})
            ledger.db.commit(); base.db.commit(); started += 1
            if day_index != len(runtime_metrics):
                raise RuntimeError("V5 runtime telemetry append index mismatch")
            _append_runtime_metric(runtime_metrics_path, arm=arm, seed=seed, date=date, index=day_index, compute_seconds=compute_seconds)
            runtime_metrics.append({"arm": arm, "seed": seed, "date": date, "index": day_index, "compute_seconds": compute_seconds})
            _assert_no_unknown(ledger, f"V5 after {date} primary")
            _assert_no_unknown(base, f"V5 after {date} paired")
            if max_days and started >= max_days:
                break
        current_raw = ledger.get_meta("last_completed_index")
        training_complete = current_raw is not None and int(current_raw) == len(dates) - 1
        final_settlement: dict[str, Any] | None = None
        if training_complete and settlement_date and ledger.get_meta("final_settlement_date") is None:
            prices = stream.market.raw_prices if stream.market is not None else stream.prices
            settled = _settle_v4(ledger, settlement_date, stream.base.calendar, prices, costs, stream.market)
            _settle_v4(base, settlement_date, stream.base.calendar, prices, costs, stream.market)
            _assert_no_unknown(ledger, "V5 final settlement primary")
            _assert_no_unknown(base, "V5 final settlement paired")
            state_before = D(ledger.get_meta("effective_multiplier") or ledger.get_meta("multiplier") or "1.0")
            effective, final_resolved = resolve_signal_outcomes(ledger, settlement_date, state_before, commit_effective=use_state)
            if stream.market is not None:
                from experiments.a_share.wc006_v4_market import raw_equity
                final_equity, _ = raw_equity(ledger, stream.market, settlement_date, set())
                final_base, _ = raw_equity(base, stream.market, settlement_date, set())
            else:
                final_equity = _equity(ledger, stream.prices, settlement_date, "close", set())
                final_base = _equity(base, stream.prices, settlement_date, "close", set())
            prev = D(ledger.get_meta("previous_equity") or "100000")
            prev_base = D(base.get_meta("previous_equity") or "100000")
            peak = max(prev, D(ledger.get_meta("peak_equity") or prev), final_equity)
            dd = final_equity / peak - D("1") if peak else D("0")
            final_feedback = incremental_feedback(equity=final_equity, previous_equity=prev, base_equity=final_base, previous_base_equity=prev_base, drawdown=dd, previous_drawdown=D(ledger.get_meta("drawdown") or "0"), turnover_ratio=settled / prev if prev else D("0"))
            rewarded = attach_settled_reward(ledger, settlement_date, final_feedback) if use_attribution else []
            _assert_no_pending_linkage(ledger, "V5 final settlement", require_rewards=use_attribution)
            _assert_no_unknown(ledger, "V5 final settlement complete")
            final_settlement = {"date": settlement_date, "effective_multiplier": str(effective), "resolved": final_resolved, "rewarded": rewarded, "turnover": str(settled), "cost": str(_fees_on_date(ledger, settlement_date)), "risk_budget": str(RISK_BUDGET)}
            ledger.put_meta("final_settlement_date", settlement_date); base.put_meta("final_settlement_date", settlement_date)
            ledger.put_meta("final_settlement", final_settlement); ledger.db.commit(); base.db.commit()
        _assert_no_unknown(ledger, "V5 before complete primary")
        _assert_no_unknown(base, "V5 before complete paired")
        complete = training_complete and not ledger.pending() and not base.pending() and not ledger.unresolved() and not base.unresolved()
        if complete:
            _assert_no_pending_linkage(ledger, "V5 completion", require_rewards=use_attribution)
        coverage = None
        if complete:
            events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
            coverage = assess_action_coverage(events, expected_records=expected_records)
            ledger.put_meta("action_coverage", coverage); ledger.db.commit()
            existing = json.loads(provenance_path.read_text(encoding="utf-8"))
            existing["final_coverage"] = coverage
            existing["final_settlement"] = final_settlement or ledger.get_meta("final_settlement")
            provenance_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
        quarantined = bool(coverage and coverage["quarantined_action_collapse"])
        if quarantined:
            quarantine_run(output, reason="action_collapse", manifest_sha256=input_gate.get("market_manifest_sha256") or input_gate.get("candidate_manifest_sha256"))
        return {"status": "quarantined_action_collapse" if quarantined else ("complete" if complete else "paused"), "arm": arm, "comparison_arm": _COMPARISON_ARM, "ablation": arm, "paired_base_arm": _PAIRED_BASE_ARM, "seed": seed, "days": started, "calendar_days": len(dates), "signal_records": len(stream.base.records), "action_coverage": coverage, "input_gate": input_gate, "output_scope": scope, "signature": signature, "graph_manifest_sha256": graph_hash}
    finally:
        ledger.close(); base.close()


def run_v5_fixture(
    stream_path: Path,
    output: Path,
    *,
    max_days: int = 0,
    expected_records: int = 3,
    training_end: str = TRAIN_END,
    controller_factory: Callable[[Settings], Any] = FlyController,
    seed: int = 0,
    arm: str = "combined",
    market_manifest: Path | None = None,
) -> dict[str, Any]:
    if expected_records == 46 or market_manifest is not None:
        raise RuntimeError("fixture API cannot bypass formal market/gates")
    return _run_v5(stream_path, output, formal=False, max_days=max_days, expected_records=expected_records, training_end=training_end, controller_factory=controller_factory, market_manifest=None, arm=arm, seed=seed)


def run_v5_formal(
    stream_path: Path,
    output: Path,
    market_manifest: Path,
    *,
    arm: str,
    seed: int,
    max_days: int = 0,
    controller_factory: Callable[[Settings], Any] = FlyController,
) -> dict[str, Any]:
    return _run_v5(stream_path, output, formal=True, max_days=max_days, expected_records=46, training_end=TRAIN_END, controller_factory=controller_factory, market_manifest=market_manifest, arm=arm, seed=seed)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument("--market-manifest", type=Path, required=True)
    parser.add_argument("--arm", choices=V5_ARMS, required=True, help="prespecified V5 ablation arm")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(run_v5_formal(args.manifest, args.out, args.market_manifest, max_days=args.max_days, arm=args.arm, seed=args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
