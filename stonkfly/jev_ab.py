"""Reproducible, non-invasive Jev sensory A/B experiment scaffolding.

This module only plans/validates/compares runs.  It deliberately does not
construct a controller or start a performance experiment.  A future driver
may consume the emitted plan only after the explicit corrected Branch B
lineage/hash gate is allowed.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


SCHEMA = "stonkfly-jev-ab-plan/v1"
ARMS = ("A", "B", "C", "D")
FORBIDDEN_FEATURE_KEY = re.compile(
    r"(?:^|_)(?:future|forward|reward|aversive|reinforcement|pnl|profit|return)(?:$|_)",
    re.IGNORECASE,
)


def canonical_hash(value: Any) -> str:
    """Hash JSON-compatible values deterministically."""
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def path_hash(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    if p.is_file():
        return file_hash(p)
    entries = []
    for child in sorted(x for x in p.rglob("*") if x.is_file()):
        entries.append((str(child.relative_to(p)), file_hash(child)))
    return canonical_hash(entries)


def _walk_keys(value: Any, prefix: str = "") -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield name
            yield from _walk_keys(child, name)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from _walk_keys(child, f"{prefix}[{i}]")


def validate_feature_payload(payload: Any) -> None:
    """Reject future/reward leakage from a sensory feature payload."""
    bad = [key for key in _walk_keys(payload) if any(FORBIDDEN_FEATURE_KEY.search(part) for part in key.split("."))]
    if bad:
        raise ValueError("feature dataset contains forbidden future/reward field(s): " + ", ".join(bad[:5]))


def validate_feature_dataset(path: str | Path | None, feature_names: Iterable[str]) -> tuple[int, str | None]:
    names = list(feature_names)
    if not 8 <= len(names) <= 32:
        raise ValueError("Jev feature count must be between 8 and 32")
    if len(set(names)) != len(names) or any(not isinstance(x, str) or not x for x in names):
        raise ValueError("feature_names must be unique, non-empty strings")
    if any(FORBIDDEN_FEATURE_KEY.search(name) for name in names):
        raise ValueError("feature_names may not contain future/reward fields")
    if path is None:
        return len(names), None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}:
        if p.suffix.lower() == ".json":
            validate_feature_payload(json.loads(p.read_text(encoding="utf-8")))
        else:
            for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if line.strip():
                    try:
                        validate_feature_payload(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid feature JSONL at line {line_no}") from exc
    elif p.is_file() and p.suffix.lower() == ".csv":
        header = p.read_text(encoding="utf-8").splitlines()[0] if p.read_text(encoding="utf-8").splitlines() else ""
        validate_feature_payload({name: None for name in header.split(",") if name})
    return len(names), path_hash(p)


def read_audit_status(path: str | Path | None) -> str:
    if path is None:
        return "MISSING"
    p = Path(path)
    if not p.exists():
        return "MISSING"
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "UNREADABLE"
    if isinstance(obj, Mapping):
        for key in ("status", "audit_status", "baseline_audit_status"):
            if isinstance(obj.get(key), str):
                return obj[key]
    return "UNKNOWN"


def _read_audit(path: str | Path | None) -> tuple[Path | None, Mapping[str, Any] | None, str]:
    if path is None:
        return None, None, "MISSING"
    audit_path = Path(path).resolve()
    if not audit_path.is_file():
        return audit_path, None, "MISSING"
    try:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return audit_path, None, "UNREADABLE"
    if not isinstance(payload, Mapping):
        return audit_path, None, "UNKNOWN"
    return audit_path, payload, str(payload.get("status", "UNKNOWN"))


def _resolve_audit_artifact(audit_path: Path, declared: Any) -> Path | None:
    if not isinstance(declared, str) or not declared:
        return None
    path = Path(declared)
    if path.is_absolute():
        return path.resolve()
    candidates = [Path.cwd() / path]
    candidates.extend(parent / path for parent in audit_path.parents)
    return next((candidate.resolve() for candidate in candidates if candidate.exists()), candidates[0].resolve())


def _verify_file(path: Path | None, expected_hash: Any) -> tuple[bool, str | None]:
    if path is None or not path.is_file():
        return False, None
    actual = file_hash(path)
    return isinstance(expected_hash, str) and actual == expected_hash, actual


def validate_corrected_baseline(config: "ABConfig") -> dict[str, Any]:
    """Fail-closed validation for the explicitly approved corrected Branch-B lineage."""
    audit_path, audit, audit_status = _read_audit(config.baseline_audit)
    result: dict[str, Any] = {
        "allowed": False,
        "status": "BLOCKED_CORRECTED_B_LINEAGE_INVALID",
        "reason": "corrected Branch B baseline provenance is incomplete",
        "lineage_id": config.baseline_lineage_id,
        "auto_start": False,
        "paper_only": True,
        "verified_artifacts": {},
    }
    if audit_path is None or audit is None:
        result["reason"] = f"baseline audit is {audit_status.lower()}"
        return result

    expected_status = "PASS_CORRECTED_BASELINE_ESTABLISHED_WITH_HISTORICAL_A_INVALIDATED"
    baseline = audit.get("corrected_branch_b_baseline")
    historical = audit.get("historical_branch_a_performance_claim")
    gate = audit.get("jev_performance_gate")
    if audit_status != expected_status or not isinstance(baseline, Mapping) or not isinstance(historical, Mapping) or not isinstance(gate, Mapping):
        result["reason"] = "baseline audit is not the approved corrected-Branch-B audit"
        return result
    if historical.get("status") != "BLOCKED_NONINVARIANT":
        result["reason"] = "historical Branch A is not blocked as non-invariant"
        return result
    if baseline.get("status") != "PASS_TRUSTWORTHY" or gate.get("status") != "UNBLOCKED_CORRECTED_B_LINEAGE_ONLY":
        result["reason"] = "corrected Branch B lineage is not explicitly unlocked"
        return result
    expected_lineage = baseline.get("lineage_id")
    if config.baseline_lineage_id != expected_lineage or expected_lineage != "corrected-branch-b-2022-v5.1":
        result["reason"] = "baseline lineage id does not match corrected Branch B"
        return result
    if gate.get("auto_start") is not False:
        result["reason"] = "audit gate must remain non-automatic"
        return result
    if config.seed != baseline.get("config_seed"):
        result["reason"] = "seed does not match corrected Branch B"
        return result
    if config.config.get("paper_only") is not True:
        result["reason"] = "corrected Branch B unlock is paper-only"
        return result
    if config.config.get("graph_manifest_sha256") != baseline.get("graph_manifest_sha256"):
        result["reason"] = "MaleCNS graph hash does not match corrected Branch B"
        return result
    if config.config.get("baseline_plasticity_mode") != baseline.get("plasticity_mode"):
        result["reason"] = "baseline provenance plasticity mode does not match corrected Branch B"
        return result
    if config.config.get("candidate_cache_002703_sha256") != baseline.get("candidate_cache_002703_sha256"):
        result["reason"] = "corrected candidate cache hash does not match Branch B"
        return result
    if config.config.get("lineage_id") != expected_lineage:
        result["reason"] = "config lineage_id does not match corrected Branch B"
        return result
    if config.config.get("broker_orders_enabled") is not False or config.config.get("live_execution_authorized") is not False:
        result["reason"] = "corrected Branch B requires paper-only execution flags"
        return result

    artifact_specs = {
        "replay_manifest": (baseline.get("manifest_path"), baseline.get("manifest_path"), baseline.get("manifest_sha256")),
        "comparison": (baseline.get("comparison_path"), baseline.get("comparison_path"), baseline.get("comparison_sha256")),
        "candidate_stream": (config.candidate_stream, baseline.get("candidate_stream_manifest"), baseline.get("candidate_stream_manifest_sha256")),
        "market_data": (config.market_data, baseline.get("market_manifest"), baseline.get("market_manifest_sha256")),
        "brain_checkpoint": (config.brain_checkpoint, baseline.get("initial_brain_checkpoint"), baseline.get("initial_brain_sha256")),
        "direct_quant_baseline": (config.direct_quant_baseline, baseline.get("direct_quant_baseline"), baseline.get("direct_quant_baseline_sha256")),
        "branch_provenance": (baseline.get("branch_provenance_path"), baseline.get("branch_provenance_path"), baseline.get("branch_provenance_sha256")),
    }
    for name, (supplied, declared, expected_hash) in artifact_specs.items():
        path = _resolve_audit_artifact(audit_path, supplied)
        declared_path = _resolve_audit_artifact(audit_path, declared)
        path_matches = path is not None and declared_path is not None and path == declared_path
        valid, actual_hash = _verify_file(path, expected_hash)
        valid = valid and path_matches
        result["verified_artifacts"][name] = {
            "path": str(path) if path is not None else None,
            "declared_path": str(declared_path) if declared_path is not None else None,
            "path_matches": path_matches,
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "valid": valid,
        }
        if not valid:
            result["reason"] = f"{name} artifact is missing or hash-mismatched"
            return result

    result["allowed"] = True
    result["status"] = "UNBLOCKED_CORRECTED_B_LINEAGE_ONLY"
    result["reason"] = None
    result["lineage_id"] = expected_lineage
    result["auto_start"] = False
    result["paper_only"] = True
    return result


@dataclass(frozen=True)
class ABConfig:
    output_root: str
    baseline_audit: str | None
    candidate_stream: str | None
    market_data: str | None
    feature_dataset: str | None
    feature_names: tuple[str, ...]
    mapping_hash: str
    normalization_hash: str
    brain_checkpoint: str | None = None
    seed: int = 0
    direct_quant_baseline: str | None = None
    baseline_lineage_id: str | None = None
    config: Mapping[str, Any] = field(default_factory=dict)


def make_plan(config: ABConfig) -> dict[str, Any]:
    """Validate inputs and return a four-arm plan without starting a model."""
    _, feature_hash = validate_feature_dataset(config.feature_dataset, config.feature_names)
    root = Path(config.output_root)
    root.mkdir(parents=True, exist_ok=True)
    arms = {}
    for arm, plasticity, sensory in (("A", "frozen", False), ("B", "frozen", True), ("C", "enabled", False), ("D", "enabled", True)):
        arm_root = root / f"arm-{arm}"
        arms[arm] = {
            "arm": arm,
            "plasticity": plasticity,
            "rgb_only": not sensory,
            "sensory_enabled": sensory,
            "output_root": str(arm_root),
            "checkpoint_root": str(arm_root / "checkpoints"),
        }
    audit_status = read_audit_status(config.baseline_audit)
    lineage_gate = validate_corrected_baseline(config)
    hashes = {
        "candidate_stream": path_hash(config.candidate_stream),
        "market_data": path_hash(config.market_data),
        "feature_dataset": feature_hash,
        "mapping": config.mapping_hash,
        "normalization": config.normalization_hash,
        "brain_checkpoint": path_hash(config.brain_checkpoint),
        "direct_quant_baseline": path_hash(config.direct_quant_baseline),
    }
    payload = {
        "schema": SCHEMA,
        "seed": config.seed,
        "baseline_audit": {"path": config.baseline_audit, "status": audit_status},
        "performance_gate": lineage_gate,
        "hashes": hashes,
        "feature_names": list(config.feature_names),
        "mapping_hash": config.mapping_hash,
        "normalization_hash": config.normalization_hash,
        "direct_quant_baseline": config.direct_quant_baseline,
        "arms": arms,
        "config": dict(config.config),
    }
    payload["config_hash"] = canonical_hash(payload)
    (root / "plan.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def performance_gate(plan: Mapping[str, Any]) -> None:
    if not bool(plan.get("performance_gate", {}).get("allowed")):
        reason = plan.get("performance_gate", {}).get("reason", "baseline audit not PASS")
        raise RuntimeError(f"Jev performance run blocked: {reason}")


def _load_rows(source: str | Path | Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(source, (str, Path)):
        p = Path(source)
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [dict(row) for row in source]


def _get(row: Mapping[str, Any], *paths: str, default: Any = None) -> Any:
    for path in paths:
        value: Any = row
        for part in path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                value = None
                break
            value = value[part]
        if value is not None:
            return value
    return default


def _num(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal("0")



def _event_key(row: Mapping[str, Any], *, arm: str, index: int) -> tuple[str, str]:
    """Stable cross-arm identity; positional alignment is intentionally forbidden."""
    event_id = _get(row, "event_id", "observation.event_id", default=None)
    date = _get(row, "date", "observation.date", "global_state.date", "execution.execution_date", "execution_date", default=None)
    if event_id is None or date is None:
        missing = "event_id" if event_id is None else "date"
        raise ValueError(f"arm {arm!r} row {index} is missing required {missing}; A/B comparison requires event_id/date")
    return str(event_id), str(date)


def _indexed_events(rows: list[dict[str, Any]], *, arm: str, use_cross_arm_key: bool = False) -> dict[str, dict[str, Any]]:
    """Validate local identities and index an arm for the requested comparison."""
    result: dict[str, dict[str, Any]] = {}
    event_ids: set[str] = set()
    dates: set[str] = set()
    for index, row in enumerate(rows):
        event_id, date = _event_key(row, arm=arm, index=index)
        if event_id in event_ids or date in dates:
            duplicate = "event_id" if event_id in event_ids else "date"
            raise ValueError(f"arm {arm!r} repeats {duplicate} {event_id if duplicate == 'event_id' else date!r}")
        event_ids.add(event_id)
        dates.add(date)
        cross_arm_key = _get(row, "cross_arm_key", "observation.cross_arm_key", default=None) if use_cross_arm_key else None
        key = str(cross_arm_key) if cross_arm_key is not None else date
        if key in result:
            raise ValueError(f"arm {arm!r} repeats cross-arm alignment key {key!r}")
        result[key] = {"event_id": event_id, "date": date, "row": row}
    return result


def _decimal_summary(values: Iterable[Decimal]) -> dict[str, Any]:
    """Compact deterministic distribution summary, preserving Decimal precision."""
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "mean": None, "min": None, "max": None, "p05": None, "p50": None, "p95": None}

    def percentile(p: Decimal) -> Decimal:
        position = p * Decimal(len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {"count": len(ordered), "mean": str(sum(ordered, Decimal("0")) / len(ordered)), "min": str(ordered[0]), "max": str(ordered[-1]), "p05": str(percentile(Decimal("0.05"))), "p50": str(percentile(Decimal("0.50"))), "p95": str(percentile(Decimal("0.95")))}


def _first_numeric(row: Mapping[str, Any], *paths: str) -> Decimal | None:
    value = _get(row, *paths, default=None)
    try:
        return Decimal(str(value)) if value is not None else None
    except Exception:
        return None


def _classification(row: Mapping[str, Any]) -> str:
    value = _get(row, "reinforcement.classification", "reward.stimulus", "reinforcement", "reward.reinforcement_classification", default="none")
    return str(value).lower() if not isinstance(value, Mapping) else "none"


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actions = [str(_get(r, "decoder.action", "action", "decoder_action", default="UNKNOWN")) for r in rows]
    returns = [_num(_get(r, "reward.portfolio_return", "portfolio_return", "return", default=0)) for r in rows]
    equity, peak, max_dd = Decimal("1"), Decimal("1"), Decimal("0")
    for ret in returns:
        equity *= Decimal("1") + ret
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak) if peak else max_dd
    differences: list[Decimal] = []
    gates: list[Decimal] = []
    changed_edges: list[Decimal] = []
    efficacy = {"mean": [], "min": [], "max": []}
    requested_path, effective_path, action_path = [], [], []
    for index, row in enumerate(rows):
        event_id, date = _event_key(row, arm="metrics", index=index)
        action_path.append({"event_id": event_id, "date": date, "action": actions[index]})
        difference = _first_numeric(row, "neural.difference_hz", "difference_hz")
        if difference is None:
            left, right = _first_numeric(row, "neural.left_hz", "left_hz"), _first_numeric(row, "neural.right_hz", "right_hz")
            difference = right - left if left is not None and right is not None else None
        if difference is not None:
            differences.append(difference)
        gate = _first_numeric(row, "neural.gate_spikes", "gate_spikes")
        if gate is not None:
            gates.append(gate)
        for paths, target in ((("execution.requested_multiplier", "decoder.requested_multiplier", "requested_multiplier"), requested_path), (("execution.effective_multiplier", "decoder.effective_multiplier", "effective_multiplier", "global_state.multiplier", "multiplier"), effective_path)):
            value = _first_numeric(row, *paths)
            if value is not None:
                target.append({"event_id": event_id, "date": date, "value": str(value)})
        changed = _first_numeric(row, "neural.memory.changed_edges", "memory.changed_edges", "plasticity.changed_edges", "changed_edges")
        if changed is not None:
            changed_edges.append(changed)
        for name, paths in (("mean", ("neural.memory.mean_efficacy", "memory.mean_efficacy", "plasticity.mean_efficacy")), ("min", ("neural.memory.minimum_efficacy", "memory.minimum_efficacy", "plasticity.minimum_efficacy")), ("max", ("neural.memory.maximum_efficacy", "memory.maximum_efficacy", "plasticity.maximum_efficacy"))):
            value = _first_numeric(row, *paths)
            if value is not None:
                efficacy[name].append(value)
    veto_count = sum(bool(_get(r, "execution.vetoes", "vetoes", "guard_reason", default=False)) for r in rows)
    gate_summary = _decimal_summary(gates)
    return {
        "event_count": len(rows), "action_distribution": dict(sorted(Counter(actions).items())), "action_path": action_path,
        "left_right_difference_hz": _decimal_summary(differences),
        "gate_activation": {**gate_summary, "active_event_count": sum(value > 0 for value in gates), "active_rate": sum(value > 0 for value in gates) / len(gates) if gates else None},
        "multiplier_path": {"requested": requested_path, "effective": effective_path},
        "multiplier_path_summary": {"requested": _decimal_summary(_num(item["value"]) for item in requested_path), "effective": _decimal_summary(_num(item["value"]) for item in effective_path)},
        "return": str(equity - Decimal("1")), "max_drawdown": str(max_dd),
        "turnover": str(sum((_num(_get(r, "execution.turnover", "turnover", default=0)) for r in rows), Decimal("0"))),
        "fees": str(sum((_num(_get(r, "execution.fees", "fees", default=0)) for r in rows), Decimal("0"))),
        "veto_count": veto_count, "veto_rate": veto_count / len(rows) if rows else None,
        "reward_count": sum(_classification(r) == "reward" for r in rows), "aversive_count": sum(_classification(r) == "aversive" for r in rows),
        "changed_plastic_edges": _decimal_summary(changed_edges),
        "memory_efficacy": {name: {"available": bool(values), **_decimal_summary(values)} for name, values in efficacy.items()},
    }


def compare_arms(arms: Mapping[str, str | Path | Iterable[Mapping[str, Any]]], *, pairs: tuple[tuple[str, str], ...] = (("B", "A"), ("D", "C"), ("D", "B"))) -> dict[str, Any]:
    """Compare arms on explicit ``cross_arm_key`` or, by default, unique date.

    event_id includes run provenance and is a local identity: B's sensory
    provenance means it need not equal A's event_id.
    """
    loaded = {name: _load_rows(source) for name, source in arms.items()}
    indexed = {name: _indexed_events(rows, arm=name, use_cross_arm_key=True) for name, rows in loaded.items()}
    report: dict[str, Any] = {"schema": "stonkfly-jev-ab-compare/v2", "arms": {name: _metrics(rows) for name, rows in loaded.items()}, "pairs": {}}
    for left, right in pairs:
        if left not in loaded or right not in loaded:
            continue
        left_events, right_events = indexed[left], indexed[right]
        common = sorted(left_events.keys() & right_events.keys())
        left_only, right_only = sorted(left_events.keys() - right_events.keys()), sorted(right_events.keys() - left_events.keys())
        aligned = not left_only and not right_only
        disagreements = []
        if aligned:
            for key in common:
                left_event, right_event = left_events[key], right_events[key]
                left_action = _get(left_event["row"], "decoder.action", "action", "decoder_action", default="UNKNOWN")
                right_action = _get(right_event["row"], "decoder.action", "action", "decoder_action", default="UNKNOWN")
                if left_action != right_action:
                    disagreements.append({"cross_arm_key": key, "date": left_event["date"], "left_event_id": left_event["event_id"], "right_event_id": right_event["event_id"], "left_action": left_action, "right_action": right_action})
        describe = lambda key, event: {"cross_arm_key": key, "event_id": event["event_id"], "date": event["date"]}
        report["pairs"][f"{left}-{right}"] = {
            "alignment": {"status": "ALIGNED" if aligned else "MISALIGNED", "key": "cross_arm_key_or_date", "matched_event_count": len(common), "left_only": [describe(key, left_events[key]) for key in left_only], "right_only": [describe(key, right_events[key]) for key in right_only]},
            "action_disagreement_count": len(disagreements) if aligned else None,
            "action_disagreement_rate": len(disagreements) / len(common) if aligned and common else (0.0 if aligned else None),
            "disagreement_events": disagreements, "disagreement_dates": [item["date"] for item in disagreements],
            "left_minus_right": {"return": str(_num(report["arms"][left]["return"]) - _num(report["arms"][right]["return"])), "max_drawdown": str(_num(report["arms"][left]["max_drawdown"]) - _num(report["arms"][right]["max_drawdown"])), "turnover": str(_num(report["arms"][left]["turnover"]) - _num(report["arms"][right]["turnover"])), "fees": str(_num(report["arms"][left]["fees"]) - _num(report["arms"][right]["fees"]))},
        }
    return report


def regression_compare(original: Iterable[Mapping[str, Any]], null: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Fail closed unless NullSensoryAdapter output matches V5.1 date-by-date.

    Each run has its own provenance-bearing event_id, so identity equality is
    not expected across runs. Local event_id/date uniqueness is still enforced.
    """
    original_rows, null_rows = list(original), list(null)
    original_events = _indexed_events(original_rows, arm="original")
    null_events = _indexed_events(null_rows, arm="null")
    original_keys, null_keys = set(original_events), set(null_events)
    alignment_mismatches = [{"side": "original_only", "event_id": original_events[date]["event_id"], "date": date} for date in sorted(original_keys - null_keys)]
    alignment_mismatches += [{"side": "null_only", "event_id": null_events[date]["event_id"], "date": date} for date in sorted(null_keys - original_keys)]
    fields = (
        ("frame_hash", ("visual.frame_sha256", "frame_sha256", "frame_hash")),
        ("left_hz", ("neural.left_hz", "left_hz")), ("right_hz", ("neural.right_hz", "right_hz")),
        ("gate_spikes", ("neural.gate_spikes", "gate_spikes")), ("decoder_action", ("decoder.action", "action", "decoder_action")),
        ("requested_multiplier", ("execution.requested_multiplier", "decoder.requested_multiplier", "requested_multiplier")),
        ("fills", ("execution.fills", "fills")), ("vetoes", ("execution.vetoes", "vetoes", "guard_reason")),
        ("reinforcement", ("reinforcement.classification", "reward.stimulus", "reinforcement", "reward.reinforcement_classification")),
    )
    mismatches = []
    for date in sorted(original_keys & null_keys):
        original_event, null_event = original_events[date], null_events[date]
        for field_name, paths in fields:
            if _get(original_event["row"], *paths, default=None) != _get(null_event["row"], *paths, default=None):
                mismatches.append({"date": date, "original_event_id": original_event["event_id"], "null_event_id": null_event["event_id"], "field": field_name})
    return {"status": "PASS" if not alignment_mismatches and not mismatches else "FAIL", "event_count_original": len(original_rows), "event_count_null": len(null_rows), "alignment_key": "date", "alignment_mismatches": alignment_mismatches, "mismatches": mismatches}


__all__ = ["ABConfig", "ARMS", "canonical_hash", "compare_arms", "make_plan", "performance_gate", "regression_compare", "validate_corrected_baseline", "validate_feature_dataset", "validate_feature_payload"]
