"""WC006 + Fly V5.1 paper adapter.

V5.1 keeps the V5 input and graph contract in a new run namespace while
making two state machines explicit:

* selector records update a frozen candidate basket only on selector dates;
* the decoder requests an exposure multiplier independently.  A non-selector
  day may create an ``exposure_rebalance`` batch when the requested multiplier
  differs from the effective multiplier.

All orders are paper intents written before settlement.  The guard can only
veto an intent; it never manufactures a replacement order.  This module is
also the annual-slice entry point: ``--start-date`` and ``--end-date`` bound
the observation window without changing the frozen V5 input manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import sqlite3
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

import subprocess

from stonkfly.config import Settings
from stonkfly.neural.controller import deterministic_neural_payload
from stonkfly.neural import rule as plasticity_rule
from stonkfly.neural.sensory_adapter import (
    NullSensoryAdapter,
    SensoryAdapter,
    SensoryStimulation,
)
from stonkfly.neural.v51_controller import V51Controller

from experiments.a_share.replay_wc006 import (
    CostConfig,
    D,
    WC006Ledger,
    incremental_feedback,
    map_decoder_to_multiplier,
    sha256,
)
from experiments.a_share.wc006_male_cns_v4_adapter import (
    LOOKBACK,
    LOOKBACK_CLOSES,
    V4Stream,
    _equity,
    _frame,
    _stats,
    _settle_v4,
    _adapter_sha256 as _v4_adapter_sha256,
    validate_v4_stream,
)
from experiments.a_share.wc006_male_cns_v5_adapter import (
    TRAIN_END,
    _fees_on_date,
    _formal_output_scope as _v5_formal_output_scope,
    _graph_source_lock,
    _gross_exposure,
    _guard_reasons,
    _runtime_compute_seconds,
    _load_runtime_metrics,
    _append_runtime_metric,
    validate_v5_inputs,
)

V51_SCHEMA = "wc006-male-cns-paper/v5.1"
V51_RUN_PREFIX = "wc006_v51_"
V51_ARMS = ("combined",)
V51_PROTOCOL_ARM = "WC006+Fly"
V51_COMPARISON_ARMS = ("WC006-only", "WC006+Fly")
V51_PLASTICITY_MODES = ("enabled", "frozen")
REWARD_MECHANISM = {
    "name": "mechanism_b",
    "status": "NOT_PASSED",
    "formal_blocked": True,
    "adapter_status": "IMPLEMENTED_FIXTURE_AND_LEDGER_LINKAGE",
    "controller_status": "INTERFACE_PRESENT_BUT_FORMAL_NOT_PASSED",
    "note": "fixture linkage/delivery is implemented; no formal mechanism-B pass is claimed without V5.1 authorization and end-to-end evidence",
}
RISK_BUDGET = D("0.85")
LOT = D("100")
_ALLOWED_BATCH_KINDS = frozenset(("candidate_selection", "exposure_rebalance"))
_ALLOWED_OUTCOMES = frozenset(("PENDING", "FILLED", "ALREADY_AT_TARGET", "EMPTY", "VETO"))
_REWARD_ELIGIBLE_OUTCOMES = frozenset(("FILLED", "ALREADY_AT_TARGET"))
_REWARD_STATES = frozenset(("UNLINKED", "PENDING", "IN_FLIGHT", "DELIVERED", "INCONCLUSIVE_NO_ELIGIBILITY", "NONE", "SETTLEMENT_ONLY"))
_ANNUAL_SCHEMA = "wc006-v5.1-annual-input-slice/v1"
_ANNUAL_SOURCE_SHA256 = "5874b6cefab5f1c4b1ee04db819fd76430cdf67495db67db5242f5d83f42cb52"
_ANNUAL_OBSERVATION_WINDOW = ("2023-01-03", "2023-12-29")
_ANNUAL_SETTLEMENT_BOUNDARY = "2024-01-02"
_CORRECTED_MARKET_SHA256 = "e4798d60cf286df35170da6d4f9be20f6e36e18792adc8e5439e11cfe555ddab"
_CORRECTED_MARKET_QA_SHA256 = "75881a19c313fb663c6e35eeb4326f0974358c5c0c2f7d31cddbf6766822ad88"

# The visual renderer is intentionally left unchanged. This schema describes
# the values supplied to it, so a replay can audit the RGB input without
# relying on a frame hash alone.
V51_FRAME_SCHEMA = "wc006-rgb-frame/v1"
V51_FORWARD_OUTCOME_SCHEMA = "wc006-v5.1-forward-basket-outcome/v1"


def _sensory_adapter_provenance(
    sensory_adapter: SensoryAdapter | None,
) -> dict[str, Any]:
    """Return stable, non-secret adapter identity for replay binding."""
    if sensory_adapter is None:
        payload = {"mode": "none", "adapter": None}
    else:
        if not callable(getattr(sensory_adapter, "encode", None)):
            raise TypeError("sensory_adapter must provide encode(observation)")
        payload = {
            "mode": "adapter",
            "adapter_type": f"{type(sensory_adapter).__module__}.{type(sensory_adapter).__qualname__}",
            "adapter_name": str(getattr(sensory_adapter, "adapter_name", "unavailable")),
            "adapter_version": str(getattr(sensory_adapter, "adapter_version", "unavailable")),
            "mapping_hash": str(getattr(sensory_adapter, "mapping_hash", "unavailable")),
            "normalization_hash": str(getattr(sensory_adapter, "normalization_hash", "unavailable")),
        }
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "configuration_sha256": hashlib.sha256(encoded).hexdigest()}


def _decoder_provenance(controller: Any, settings: Settings | None = None) -> dict[str, Any]:
    """Extract decoder configuration from controller for provenance tracking."""
    s = settings if settings is not None else controller.s
    decoder = getattr(controller, "decoder", None)
    payload = {
        "decoder_type": s.decoder_type,
        "decoder_threshold_hz": s.decoder_threshold_hz,
        "decoder_class": (
            f"{type(decoder).__module__}.{type(decoder).__qualname__}"
            if decoder is not None
            else "fixture-unavailable"
        ),
        "normalize_mbon": s.normalize_mbon,
    }
    # Include ratio thresholds if using mbon_ratio decoder
    if s.decoder_type == "mbon_ratio":
        payload["mbon_buy_threshold"] = s.mbon_buy_threshold
        payload["mbon_sell_threshold"] = s.mbon_sell_threshold
    return payload


def _mechanism_provenance(
    controller: Any, settings: Settings | None = None
) -> dict[str, Any]:
    """Bind all settings and feedback defaults that can change learning."""
    s = settings if settings is not None else controller.s
    feedback = {}
    for name, parameter in inspect.signature(incremental_feedback).parameters.items():
        if parameter.default is inspect.Parameter.empty:
            continue
        value = parameter.default
        feedback[name] = str(value) if isinstance(value, Decimal) else value
    settings_payload = json.loads(
        json.dumps(vars(s), allow_nan=False, sort_keys=True)
    )
    return {
        "settings": {
            **settings_payload,
            "signature_sha256": s.signature(),
        },
        "decoder": _decoder_provenance(controller, s),
        "rule_parameters": plasticity_rule.PARAMETERS.copy(),
        "incremental_feedback_defaults": feedback,
        "linked_pulse": {
            "duration_ms": s.pulse_ms,
            "current": s.pulse_current,
            "binary_label_delivery": True,
            "note": (
                "feedback weights change label classification only; they do not "
                "scale linked DAN pulse duration or current"
            ),
        },
    }


def _git_provenance() -> dict[str, Any]:
    """Get git commit and working directory status for provenance."""
    project_root = Path(__file__).resolve().parents[2]
    relevant_sources = (
        "stonkfly/config.py",
        "stonkfly/neural/brain.py",
        "stonkfly/neural/circuit.py",
        "stonkfly/neural/controller.py",
        "stonkfly/neural/rule.py",
        "stonkfly/neural/v51_controller.py",
        "experiments/a_share/replay_wc006.py",
        "experiments/a_share/run_purged_fly_training.py",
        "experiments/a_share/wc006_male_cns_v51_adapter.py",
    )
    source_hashes = {
        relative: hashlib.sha256((project_root / relative).read_bytes()).hexdigest()
        for relative in relevant_sources
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=project_root, timeout=5
        ).stdout.strip()
        status_result = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            capture_output=True, text=True, cwd=project_root, timeout=5
        )
        # Check tracked and untracked state. The repository intentionally has
        # many untracked research files, so record rather than reject it.
        diff_result = subprocess.run(
            ["git", "diff", "--stat"],
            capture_output=True, text=True, cwd=project_root, timeout=5
        )
        diff_staged = subprocess.run(
            ["git", "diff", "--staged", "--stat"],
            capture_output=True, text=True, cwd=project_root, timeout=5
        )
        status_text = status_result.stdout
        is_clean = not status_text.strip()
        # If not clean, compute hash of uncommitted changes
        if not is_clean:
            diff_full = subprocess.run(
                ["git", "diff", "HEAD"],
                capture_output=True, text=True, cwd=project_root, timeout=10
            )
            uncommitted_sha256 = hashlib.sha256(diff_full.stdout.encode()).hexdigest()
        else:
            uncommitted_sha256 = None
        return {
            "git_commit": commit,
            "working_directory_clean": is_clean,
            "uncommitted_changes_sha256": uncommitted_sha256,
            "status_porcelain_sha256": hashlib.sha256(status_text.encode()).hexdigest(),
            "relevant_source_sha256": source_hashes,
        }
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return {
            "git_commit": "unavailable",
            "working_directory_clean": None,
            "uncommitted_changes_sha256": None,
            "status_porcelain_sha256": None,
            "relevant_source_sha256": source_hashes,
        }


def _authorized_annual_path(base: Path, value: object, label: str) -> Path:
    """Resolve an annual authorization path without allowing arbitrary files."""
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise RuntimeError(f"annual market authorization: invalid {label} path")
    project_root = Path(__file__).resolve().parents[2]
    runs_root = (project_root / "runs").resolve()
    path = (base / value).resolve()
    try:
        path.relative_to(runs_root)
    except ValueError as exc:
        raise RuntimeError(f"annual market authorization: {label} escapes runs") from exc
    if not path.is_file():
        raise RuntimeError(f"annual market authorization: {label} missing")
    return path


def _hash_bound_file(path: Path, expected: object, label: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64 or sha256(path) != expected:
        raise RuntimeError(f"annual market authorization: {label} hash mismatch")
    return expected


def _validate_supplemental_right_issue(
    *,
    annual_root: Path,
    authorization: dict[str, Any],
    market_path: Path,
    market_payload: dict[str, Any],
) -> dict[str, Any]:
    supplement_path = _authorized_annual_path(
        annual_root, authorization.get("supplemental_audit_path"), "supplemental audit"
    )
    supplement_sha = _hash_bound_file(
        supplement_path, authorization.get("supplemental_audit_sha256"), "supplemental audit"
    )
    try:
        supplement = json.loads(supplement_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("annual market authorization: supplemental audit malformed") from exc
    if (
        not isinstance(supplement, dict)
        or supplement.get("schema") != "wc006-v5.1-2022-right-issue-code-complete-supplemental-audit/v1"
        or supplement.get("status") != "PASS"
    ):
        raise RuntimeError("annual market authorization: supplemental audit is not PASS")
    query = supplement.get("query")
    coverage = supplement.get("coverage")
    if not isinstance(query, dict) or int(query.get("code_count", 0)) != 55 or not isinstance(coverage, dict):
        raise RuntimeError("annual market authorization: supplemental query coverage mismatch")
    batches = coverage.get("successful_batches")
    declared = authorization.get("right_issue_query_manifests")
    if not isinstance(batches, list) or not isinstance(declared, list) or len(batches) != len(declared) or len(batches) != 3:
        raise RuntimeError("annual market authorization: completed query manifests incomplete")
    seen_codes: set[str] = set()
    for batch, item in zip(batches, declared):
        if not isinstance(batch, dict) or not isinstance(item, dict):
            raise RuntimeError("annual market authorization: query manifest entry malformed")
        batch_manifest = Path(str(batch.get("manifest")))
        if batch_manifest.is_absolute():
            raise RuntimeError("annual market authorization: query manifest path must be relative")
        project_root = Path(__file__).resolve().parents[2]
        manifest_path = (project_root / batch_manifest).resolve()
        try:
            manifest_path.relative_to((project_root / "runs").resolve())
        except ValueError as exc:
            raise RuntimeError("annual market authorization: query manifest escapes runs") from exc
        if not manifest_path.is_file() or sha256(manifest_path) != item.get("sha256"):
            raise RuntimeError("annual market authorization: query manifest hash mismatch")
        import os
        expected_query_path = Path(os.path.relpath(manifest_path, annual_root)).as_posix()
        if item.get("path") != expected_query_path:
            raise RuntimeError("annual market authorization: query manifest binding mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "wc006-xysz-v4-dataset-fragment/v1" or manifest.get("dataset") != "right_issue" or manifest.get("status") != "complete":
            raise RuntimeError("annual market authorization: query manifest schema/status mismatch")
        codes = {str(code).upper() for code in manifest.get("codes", [])}
        if len(codes) != int(item.get("code_count", -1)) or codes & seen_codes:
            raise RuntimeError("annual market authorization: query code partition mismatch")
        seen_codes |= codes
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list) or len(artifacts) != 1:
            raise RuntimeError("annual market authorization: query artifact set mismatch")
        artifact = artifacts[0]
        artifact_path = manifest_path.parent / "raw" / str(artifact.get("path"))
        if not artifact_path.is_file() or sha256(artifact_path) != artifact.get("sha256") or sha256(artifact_path) != batch.get("actual_sha256"):
            raise RuntimeError("annual market authorization: query artifact hash mismatch")
    query_codes = {str(code).upper() for code in query.get("codes", [])}
    if len(query_codes) != 55 or seen_codes != query_codes:
        raise RuntimeError("annual market authorization: supplemental code coverage mismatch")
    market_supplement = market_payload.get("supplemental_audit")
    if (
        not isinstance(market_supplement, dict)
        or market_supplement.get("sha256") != supplement_sha
        or market_supplement.get("status") != "PASS"
    ):
        raise RuntimeError("annual market authorization: market/supplement binding mismatch")
    market_queries = market_payload.get("right_issue_query_manifests")
    if not isinstance(market_queries, list) or [item.get("sha256") for item in market_queries] != [item.get("sha256") for item in declared]:
        raise RuntimeError("annual market authorization: market query binding mismatch")
    return {"path": str(supplement_path), "sha256": supplement_sha, "status": "PASS", "code_count": len(query_codes)}


def _resolve_annual_market_binding(
    annual_manifest: Path,
    payload: dict[str, Any],
    supplied_market: Path,
) -> tuple[Path, str, str, dict[str, Any] | None]:
    authorization = payload.get("market_authorization")
    if authorization is None:
        market_path = Path(supplied_market).resolve()
        if not market_path.is_file() or sha256(market_path) != _CORRECTED_MARKET_SHA256:
            raise RuntimeError("annual input gate: corrected market manifest missing or hash mismatch")
        qa_path = market_path.parent / "quality_report.json"
        if not qa_path.is_file() or sha256(qa_path) != _CORRECTED_MARKET_QA_SHA256:
            raise RuntimeError("annual input gate: corrected market QA missing or hash mismatch")
        return market_path, _CORRECTED_MARKET_SHA256, _CORRECTED_MARKET_QA_SHA256, None
    if not isinstance(authorization, dict) or authorization.get("schema") != "wc006-v5.1-annual-market-binding/v1" or authorization.get("status") != "PASS":
        raise RuntimeError("annual market authorization: malformed or non-PASS binding")
    annual_root = Path(annual_manifest).resolve().parent
    market_path = _authorized_annual_path(annual_root, authorization.get("market_manifest_path"), "market manifest")
    supplied = Path(supplied_market).resolve()
    if supplied != market_path:
        raise RuntimeError("annual market authorization: supplied market path differs from bound path")
    market_sha = _hash_bound_file(market_path, authorization.get("market_manifest_sha256"), "market manifest")
    qa_path = _authorized_annual_path(annual_root, authorization.get("market_quality_report_path"), "market QA")
    qa_sha = _hash_bound_file(qa_path, authorization.get("market_quality_report_sha256"), "market QA")
    authorization_path = _authorized_annual_path(annual_root, authorization.get("market_authorization_path"), "market authorization")
    _hash_bound_file(authorization_path, authorization.get("market_authorization_sha256"), "market authorization")
    try:
        market_payload = json.loads(market_path.read_text(encoding="utf-8"))
        qa_payload = json.loads(qa_path.read_text(encoding="utf-8"))
        market_authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("annual market authorization: market binding JSON malformed") from exc
    if (
        market_payload.get("schema") != "wc006-v5-authorized-development-market-inputs/v1"
        or market_payload.get("authoritative_trading_state") is not True
        or market_payload.get("paper_only") is not True
        or market_payload.get("broker_orders_enabled") is not False
        or market_payload.get("live_execution_authorized") is not False
        or qa_payload.get("manifest_sha256") != market_sha
        or market_authorization.get("status") != "PASS"
        or market_authorization.get("market_manifest", {}).get("sha256") != market_sha
        or market_authorization.get("quality_report", {}).get("sha256") != qa_sha
        or market_authorization.get("supplemental_audit", {}).get("sha256") != authorization.get("supplemental_audit_sha256")
    ):
        raise RuntimeError("annual market authorization: market/QA authorization mismatch")
    code_bindings = authorization.get("code_bindings")
    if not isinstance(code_bindings, dict):
        raise RuntimeError("annual market authorization: code bindings missing")
    project_root = Path(__file__).resolve().parents[2]
    expected_code_paths = {
        "adapter": project_root / "experiments/a_share/wc006_male_cns_v51_adapter.py",
        "controller": project_root / "stonkfly/neural/v51_controller.py",
        "brain": project_root / "stonkfly/neural/brain.py",
        "focused_tests": project_root / "tests/test_wc006_male_cns_v51_adapter.py",
    }
    for name, expected_path in expected_code_paths.items():
        item = code_bindings.get(name)
        if (
            not isinstance(item, dict)
            or item.get("path") != str(expected_path.relative_to(project_root).as_posix())
            or not expected_path.is_file()
            or sha256(expected_path) != item.get("sha256")
        ):
            raise RuntimeError(f"annual market authorization: code binding mismatch: {name}")
    supplement = _validate_supplemental_right_issue(
        annual_root=annual_root,
        authorization=authorization,
        market_path=market_path,
        market_payload=market_payload,
    )
    return market_path, market_sha, qa_sha, supplement


def _validate_annual_manifest(path: Path, source_manifest: Path | None = None) -> dict[str, Any]:
    """Validate a hash-pinned annual slice against its canonical source stream."""
    path = Path(path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("annual input gate: annual manifest is invalid") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("annual input gate: manifest must be an object")
    required_flags = {
        "schema": _ANNUAL_SCHEMA,
        "draft": True,
        "paper_only": True,
        "broker_orders_enabled": False,
        "live_execution_authorized": False,
        "candidate_selection_frozen": True,
    }
    for key, expected in required_flags.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"annual input gate: {key} mismatch")
    has_market_authorization = isinstance(payload.get("market_authorization"), dict)
    if payload.get("authorization_artifact") is not has_market_authorization:
        raise RuntimeError("annual input gate: authorization_artifact mismatch")
    try:
        year = int(payload["year"])
        observation_window = payload["observation_window"]
        observation_start, observation_end = str(observation_window[0]), str(observation_window[1])
        observation_count = int(payload["observation_count"])
        settlement_boundary = str(payload["settlement_boundary"])
        source_record_count = int(payload["source_manifest_records"])
        candidate_record_count = int(payload["candidate_record_count"])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise RuntimeError("annual input gate: explicit annual fields are malformed") from exc
    if observation_count != 242:
        raise RuntimeError("annual input gate: observation_count must be 242")
    if source_record_count <= 0 or candidate_record_count <= 0:
        raise RuntimeError("annual input gate: record counts must be positive")
    if (
        not isinstance(observation_window, list)
        or len(observation_window) != 2
        or len(observation_start) != 10
        or len(observation_end) != 10
        or observation_start > observation_end
    ):
        raise RuntimeError("annual input gate: observation window malformed")
    if (
        not observation_start.startswith(f"{year}-")
        or not observation_end.startswith(f"{year}-")
        or len(settlement_boundary) != 10
        or not settlement_boundary.startswith(f"{year + 1}-")
    ):
        raise RuntimeError("annual input gate: year/window mismatch")
    source_sha = payload.get("source_manifest_sha256")
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        raise RuntimeError("annual input gate: source manifest hash missing")

    root = path.parent
    source = (
        Path(source_manifest).resolve()
        if source_manifest is not None
        else Path(__file__).resolve().parents[2]
        / "runs/wc006_v3_2021_inputs/20260918T-candidate-full-2021-2023-stateful/manifest.json"
    )
    if not source.is_file() or sha256(source) != source_sha:
        raise RuntimeError("annual input gate: pinned source manifest missing or hash mismatch")
    try:
        source_payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("annual input gate: source manifest is invalid") from exc
    if not isinstance(source_payload, dict) or source_payload.get("schema") != "wc006-candidate-stream/v1":
        raise RuntimeError("annual input gate: source manifest schema mismatch")

    source_hashes = payload.get("source_hashes")
    source_payload_hashes = source_payload.get("source_hashes")
    if not isinstance(source_hashes, dict) or not isinstance(source_payload_hashes, dict):
        raise RuntimeError("annual input gate: source_hashes missing")
    if not {"active_config", "universe", "calendar", "source_calendar"} <= set(source_hashes):
        raise RuntimeError("annual input gate: source hash set incomplete")
    if source_hashes.get("source_calendar") != source_payload_hashes.get("calendar"):
        raise RuntimeError("annual input gate: source calendar hash mismatch")
    for name in ("active_config", "universe"):
        if source_hashes.get(name) != source_payload_hashes.get(name):
            raise RuntimeError(f"annual input gate: source input hash mismatch: {name}")
    source_root = source.parent
    for name, filename in (
        ("active_config", "active_config.json"),
        ("universe", "universe.txt"),
        ("calendar", "calendar.json"),
    ):
        source_copy = source_root / "inputs" / filename
        if not source_copy.is_file() or sha256(source_copy) != source_payload_hashes.get(name):
            raise RuntimeError(f"annual input gate: canonical source input hash mismatch: {name}")
    for name in ("active_config", "universe", "calendar"):
        expected_hash = source_hashes.get(name)
        copied = root / "inputs" / {
            "active_config": "active_config.json",
            "universe": "universe.txt",
            "calendar": "calendar.json",
        }[name]
        if not isinstance(expected_hash, str) or not copied.is_file() or sha256(copied) != expected_hash:
            raise RuntimeError(f"annual input gate: source input hash mismatch: {name}")

    try:
        calendar_payload = json.loads(
            (root / "inputs" / "calendar.json").read_text(encoding="utf-8")
        )
        calendar_dates = calendar_payload["dates"]
    except (OSError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError("annual input gate: calendar malformed") from exc
    if not isinstance(calendar_dates, list) or len(calendar_dates) != len(set(calendar_dates)):
        raise RuntimeError("annual input gate: calendar dates are not unique")
    if (
        any(not isinstance(date, str) or len(date) != 10 for date in calendar_dates)
        or calendar_dates != sorted(calendar_dates)
    ):
        raise RuntimeError("annual input gate: calendar dates are not canonical")
    annual_sessions = [date for date in calendar_dates if date.startswith(f"{year}-")]
    if (
        len(annual_sessions) != observation_count
        or annual_sessions[0] != observation_start
        or annual_sessions[-1] != observation_end
        or settlement_boundary not in calendar_dates
        or settlement_boundary in annual_sessions
    ):
        raise RuntimeError("annual input gate: calendar session window mismatch")
    if calendar_dates.index(settlement_boundary) != calendar_dates.index(observation_end) + 1:
        raise RuntimeError("annual input gate: settlement boundary is not the next valid session")

    source_all_records = source_payload.get("records")
    if not isinstance(source_all_records, list) or len(source_all_records) != source_record_count:
        raise RuntimeError("annual input gate: source record count mismatch")
    source_records = [
        item for item in source_all_records
        if isinstance(item, dict) and str(item.get("signal_date", "")).startswith(f"{year}-")
    ]
    if len(source_records) != candidate_record_count:
        raise RuntimeError("annual input gate: source annual record count mismatch")
    annual_records = payload.get("records")
    if not isinstance(annual_records, list) or len(annual_records) != candidate_record_count:
        raise RuntimeError("annual input gate: frozen record count mismatch")
    canonical = lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if [canonical(item) for item in annual_records] != [canonical(item) for item in source_records]:
        raise RuntimeError("annual input gate: annual records differ from source records")

    previous_signal = None
    for item in annual_records:
        if not isinstance(item, dict):
            raise RuntimeError("annual input gate: invalid annual record")
        signal = str(item.get("signal_date", ""))
        execution = str(item.get("execution_date", ""))
        if previous_signal is not None and signal <= previous_signal:
            raise RuntimeError("annual input gate: annual records are not ordered")
        previous_signal = signal
        if (
            not signal.startswith(f"{year}-")
            or not observation_start <= signal <= observation_end
            or signal == settlement_boundary
            or signal not in calendar_dates
            or execution not in calendar_dates
        ):
            raise RuntimeError("annual input gate: signal/execution outside observation window")
        signal_index = calendar_dates.index(signal)
        if signal_index + 1 >= len(calendar_dates) or calendar_dates[signal_index + 1] != execution:
            raise RuntimeError("annual input gate: signal-to-execution boundary mismatch")

    for section in ("cache_files", "selector_artifacts"):
        items = payload.get(section)
        if not isinstance(items, dict if section == "cache_files" else list):
            raise RuntimeError(f"annual input gate: {section} malformed")
        iterable = items.values() if isinstance(items, dict) else items
        for item in iterable:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("copied_path"), str)
                or not isinstance(item.get("sha256"), str)
            ):
                raise RuntimeError(f"annual input gate: {section} entry malformed")
            copied = (root / item["copied_path"]).resolve()
            try:
                copied.relative_to(root)
            except ValueError as exc:
                raise RuntimeError("annual input gate: copied input escapes manifest root") from exc
            if not copied.is_file() or sha256(copied) != item["sha256"]:
                raise RuntimeError(f"annual input gate: copied input hash mismatch: {item['copied_path']}")

    source_cache = source_payload.get("cache_files")
    source_selector = source_payload.get("selector_artifacts")
    annual_cache = payload.get("cache_files")
    annual_selector = payload.get("selector_artifacts")
    if (
        not isinstance(source_cache, dict)
        or not isinstance(source_selector, list)
        or not isinstance(annual_cache, dict)
        or not isinstance(annual_selector, list)
        or len(annual_selector) != candidate_record_count
    ):
        raise RuntimeError("annual input gate: source provenance is malformed")
    source_cache_hashes = {
        str(code): str(item.get("sha256"))
        for code, item in source_cache.items()
        if isinstance(item, dict)
    }
    for code, item in annual_cache.items():
        if (
            str(code) not in source_cache_hashes
            or not isinstance(item, dict)
            or item.get("sha256") != source_cache_hashes[str(code)]
        ):
            raise RuntimeError(f"annual input gate: cache provenance differs from source: {code}")
    source_selector_hashes = {
        str(item.get("sha256")) for item in source_selector if isinstance(item, dict)
    }
    annual_selector_hashes = {
        str(item.get("sha256")) for item in annual_selector if isinstance(item, dict)
    }
    if not annual_selector_hashes <= source_selector_hashes:
        raise RuntimeError("annual input gate: selector provenance differs from source")
    annual_record_selector_hashes = {
        str(item.get("selector", {}).get("artifact_sha256")) for item in annual_records
    }
    if annual_record_selector_hashes != annual_selector_hashes:
        raise RuntimeError("annual input gate: selector record/artifact hash set mismatch")
    annual_record_cache_hashes = {
        str(cache_hash)
        for item in annual_records
        for cache_hash in (item.get("selector", {}).get("cache_sha256", {}) or {}).values()
    }
    if not annual_record_cache_hashes <= {
        str(item.get("sha256")) for item in annual_cache.values() if isinstance(item, dict)
    }:
        raise RuntimeError("annual input gate: selector/cache hash set mismatch")

    cache_dates: dict[str, set[str]] = {}
    for code, item in annual_cache.items():
        copied = (root / str(item.get("copied_path", ""))).resolve()
        try:
            copied.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("annual input gate: cache path escapes manifest root") from exc
        with copied.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "date" not in (reader.fieldnames or ()):
                raise RuntimeError(f"annual input gate: cache date column missing: {code}")
            dates = [str(row.get("date", "")) for row in reader]
        if (
            any(len(date) != 10 or date[4] != "-" or date[7] != "-" for date in dates)
            or dates != sorted(set(dates))
        ):
            raise RuntimeError(f"annual input gate: cache dates are not canonical: {code}")
        cache_dates[str(code)] = set(dates)
    for item in annual_records:
        signal = str(item["signal_date"])
        execution = str(item["execution_date"])
        timing = item.get("timing_proof") or {}
        if timing.get("signal_lag_one_day") is not True or timing.get("execution_is_first_next_valid_session") is not True:
            raise RuntimeError("annual input gate: timing proof missing")
        for target in item.get("selector", {}).get("target_topk", []):
            code = str(target.get("code"))
            if (
                code not in cache_dates
                or len([date for date in cache_dates[code] if date <= signal]) < LOOKBACK_CLOSES
                or signal not in cache_dates[code]
                or execution not in cache_dates[code]
            ):
                raise RuntimeError(f"annual input gate: cache causal coverage failed on {signal}: {code}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "payload": payload,
        "source_path": source,
        "source_sha256": source_sha,
    }


def _assert_annual_source_match(annual: dict[str, Any], stream: V4Stream) -> None:
    payload = annual["payload"]
    year = int(payload["year"])
    expected = {str(item["signal_date"]): item for item in payload["records"]}
    observed = {
        str(item.signal_date): item
        for item in stream.base.records
        if str(item.signal_date).startswith(f"{year}-")
    }
    if set(expected) != set(observed):
        raise RuntimeError("annual input gate: frozen dates differ from source stream")
    for date, frozen in expected.items():
        source_item = observed[date]
        frozen_targets = frozen.get("selector", {}).get("target_topk", [])
        if len(frozen_targets) != len(source_item.targets):
            raise RuntimeError(f"annual input gate: target count mismatch on {date}")
        for left, right in zip(frozen_targets, source_item.targets):
            if str(left.get("code")) != str(right.get("code")) or not math.isclose(float(left.get("target_weight")), float(right.get("weight")), rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"annual input gate: target mismatch on {date}")


def _formal_v51_output_scope(output: Path, arm: str) -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    runs = root / "runs"
    output = Path(output).resolve()
    try:
        relative = output.relative_to(runs)
    except ValueError as exc:
        raise RuntimeError("formal V5.1 output must be below runs/wc006_v51_<arm>") from exc
    expected = f"wc006_v51_{arm}"
    if len(relative.parts) < 2 or relative.parts[0] != expected:
        raise RuntimeError(f"formal V5.1 output must be below runs/{expected}/<run>")
    return {"formal": True, "root": str(root), "relative": str(relative), "directory": expected}


def _validate_plasticity_mode(mode: str) -> str:
    if mode not in V51_PLASTICITY_MODES:
        raise ValueError(f"unsupported V5.1 plasticity mode: {mode}")
    return mode


def _assert_controller_plasticity_mode(controller: Any, mode: str, *, context: str) -> None:
    """Fail closed if a controller exposes a checkpoint freeze mismatch."""
    brain = getattr(controller, "brain", None)
    if brain is None or not hasattr(brain, "weights_frozen"):
        if mode == "frozen":
            raise RuntimeError(f"V5.1 frozen mode requires brain.weights_frozen ({context})")
        return
    expected = mode == "frozen"
    if bool(brain.weights_frozen) is not expected:
        raise RuntimeError(f"V5.1 plasticity mode checkpoint/controller mismatch ({context})")


def _annual_boundary_settlement_only(
    annual_payload: dict[str, Any] | None,
    *,
    settlement_date: str | None,
    date: str,
    end_date: str,
) -> bool:
    """Identify a final observation that must not create a boundary order."""
    if annual_payload is None or settlement_date is None or date != end_date:
        return False
    records = annual_payload.get("records") or []
    if not records:
        return False
    return str(records[-1].get("execution_date")) == end_date


def _scope_dates(
    stream: V4Stream,
    start_date: str,
    end_date: str,
    settlement_boundary: str | None = None,
) -> tuple[list[str], str | None]:
    start_date, end_date = str(start_date), str(end_date)
    if len(start_date) != 10 or len(end_date) != 10 or start_date > end_date:
        raise ValueError("V5.1 scope requires ISO start-date <= end-date")
    calendar = list(stream.base.calendar)
    dates = [date for date in calendar if start_date <= date <= end_date]
    if not dates or dates[0] != start_date or dates[-1] != end_date:
        raise ValueError("V5.1 scope boundaries must be exact calendar sessions")
    records = stream.base.records
    if not any(record.signal_date <= start_date for record in records):
        raise ValueError("V5.1 scope starts before the first available candidate record")
    if settlement_boundary is not None:
        settlement_boundary = str(settlement_boundary)
        if settlement_boundary not in calendar or settlement_boundary <= end_date:
            raise ValueError("V5.1 settlement boundary must be after END_DATE")
        if calendar.index(settlement_boundary) != calendar.index(end_date) + 1:
            raise ValueError("V5.1 settlement boundary must be the next valid session")
        return dates, settlement_boundary
    settlement = next(
        (
            record.execution_date
            for record in reversed(records)
            if start_date <= record.signal_date <= end_date and record.execution_date > end_date
        ),
        None,
    )
    return dates, settlement


def _ensure_v51_batches(ledger: WC006Ledger) -> None:
    ledger.db.execute(
        """CREATE TABLE IF NOT EXISTS v51_batches(
          batch_id TEXT PRIMARY KEY,
          batch_kind TEXT NOT NULL,
          request_date TEXT NOT NULL,
          execution_date TEXT NOT NULL,
          candidate_epoch INTEGER NOT NULL,
          requested_multiplier TEXT NOT NULL,
          effective_multiplier_before TEXT NOT NULL,
          requested_target_exposure TEXT NOT NULL,
          effective_target_exposure TEXT NOT NULL,
          intent_ids_json TEXT NOT NULL,
          outcome TEXT NOT NULL,
          empty_outcome TEXT NOT NULL,
          effective_multiplier_after TEXT NOT NULL,
          reward_linkage_id TEXT,
          reward_delivery_status TEXT NOT NULL DEFAULT 'UNLINKED',
          reward_label TEXT NOT NULL DEFAULT 'none',
          reward_delivery_date TEXT,
          reward_trace_before_digest TEXT,
          reward_trace_after_digest TEXT,
          reward_weight_before_digest TEXT,
          reward_weight_after_digest TEXT,
          UNIQUE(request_date,batch_kind))"""
    )
    existing = {str(row[1]) for row in ledger.db.execute("PRAGMA table_info(v51_batches)").fetchall()}
    additions = (
        ("reward_linkage_id", "TEXT"),
        ("reward_delivery_status", "TEXT NOT NULL DEFAULT 'UNLINKED'"),
        ("reward_label", "TEXT NOT NULL DEFAULT 'none'"),
        ("reward_delivery_date", "TEXT"),
        ("reward_trace_before_digest", "TEXT"),
        ("reward_trace_after_digest", "TEXT"),
        ("reward_weight_before_digest", "TEXT"),
        ("reward_weight_after_digest", "TEXT"),
    )
    for name, definition in additions:
        if name not in existing:
            ledger.db.execute(f"ALTER TABLE v51_batches ADD COLUMN {name} {definition}")
    ledger.db.execute("CREATE UNIQUE INDEX IF NOT EXISTS v51_reward_linkage_uq ON v51_batches(reward_linkage_id) WHERE reward_linkage_id IS NOT NULL")
    ledger.db.commit()


def _reward_linkage_id(batch_id: str, execution_date: str) -> str:
    payload = {"schema": V51_SCHEMA, "batch_id": batch_id, "execution_date": execution_date, "kind": "mechanism_b"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _batch_id(*, request_date: str, kind: str, candidate_epoch: int, requested: Decimal, intent_ids: list[str]) -> str:
    payload = {"schema": V51_SCHEMA, "request_date": request_date, "kind": kind, "candidate_epoch": candidate_epoch, "requested": str(requested), "intent_ids": sorted(intent_ids)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _checkpoint_path(output: Path, checkpoint: dict[str, Any]) -> Path:
    """Validate and resolve a checkpoint reference stored in the ledger."""
    if not isinstance(checkpoint, dict):
        raise RuntimeError("V5.1 checkpoint metadata is invalid")
    filename = checkpoint.get("file")
    expected = checkpoint.get("sha256")
    if not isinstance(filename, str) or not filename or not isinstance(expected, str):
        raise RuntimeError("V5.1 checkpoint metadata lacks file/sha256")
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected.lower()):
        raise RuntimeError("V5.1 checkpoint metadata has invalid sha256")
    root = Path(output).resolve()
    path = (root / filename).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("V5.1 checkpoint escapes output directory") from exc
    if not path.is_file():
        raise RuntimeError(f"V5.1 checkpoint file missing: {filename}")
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"V5.1 checkpoint hash mismatch: {filename}")
    return path


def _authoritative_events(ledger: WC006Ledger) -> list[dict[str, Any]]:
    """Read and validate the SQLite event chain before materializing JSONL."""
    rows = ledger.db.execute(
        "SELECT event_id,trade_date,phase,payload_json,prev_hash,event_hash FROM events ORDER BY rowid"
    ).fetchall()
    expected_previous = "GENESIS"
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for event_id, date, phase, payload_json, prev_hash, event_hash in rows:
        if str(event_id) in seen_ids:
            raise RuntimeError("V5.1 duplicate SQLite event id")
        seen_ids.add(str(event_id))
        if str(prev_hash) != expected_previous:
            raise RuntimeError("V5.1 SQLite event chain link mismatch")
        try:
            payload = json.loads(payload_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("V5.1 SQLite event payload is invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("V5.1 SQLite event payload is not an object")
        canonical = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)
        if canonical != str(payload_json):
            raise RuntimeError("V5.1 SQLite event payload is not canonical")
        actual_hash = hashlib.sha256((str(prev_hash) + str(payload_json)).encode()).hexdigest()
        if actual_hash != str(event_hash):
            raise RuntimeError("V5.1 SQLite event hash mismatch")
        expected_previous = str(event_hash)
        materialized = dict(payload)
        materialized["event_hash"] = str(event_hash)
        records.append(materialized)
    return records


def _sync_event_jsonl(events_path: Path, ledger: WC006Ledger) -> None:
    """Repair a truncated materialized view, rejecting extra or changed data."""
    expected = _authoritative_events(ledger)
    expected_lines = [json.dumps(item, ensure_ascii=False, allow_nan=False, sort_keys=True) for item in expected]
    if not events_path.exists():
        events_path.write_text("" if not expected_lines else "\n".join(expected_lines) + "\n", encoding="utf-8")
        return
    raw_lines = events_path.read_text(encoding="utf-8").splitlines()
    observed: list[dict[str, Any]] = []
    for line in raw_lines:
        if not line.strip():
            raise RuntimeError("V5.1 events.jsonl contains a blank line")
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError("V5.1 events.jsonl contains invalid JSON") from exc
        if not isinstance(item, dict):
            raise RuntimeError("V5.1 events.jsonl line is not an object")
        observed.append(item)
    if len(observed) > len(expected):
        raise RuntimeError("V5.1 events.jsonl contains extra events")
    # Existing records must be an exact ordered subsequence: a missing line is
    # recoverable, while a mutation or reordering is a hard integrity failure.
    cursor = 0
    for item in observed:
        canonical = json.dumps(item, ensure_ascii=False, allow_nan=False, sort_keys=True)
        while cursor < len(expected_lines) and canonical != expected_lines[cursor]:
            cursor += 1
        if cursor == len(expected_lines):
            raise RuntimeError("V5.1 events.jsonl differs from authoritative SQLite chain")
        cursor += 1
    expected_text = "" if not expected_lines else "\n".join(expected_lines) + "\n"
    if len(observed) == len(expected):
        if events_path.read_text(encoding="utf-8") != expected_text:
            raise RuntimeError("V5.1 events.jsonl formatting or content mismatch")
        return
    events_path.write_text(expected_text, encoding="utf-8")


def _intent_statuses(ledger: WC006Ledger, intent_ids: list[str]) -> list[str]:
    if not intent_ids:
        return []
    marks = ",".join("?" for _ in intent_ids)
    rows = ledger.db.execute(f"SELECT status FROM intents WHERE id IN ({marks}) ORDER BY id", tuple(intent_ids)).fetchall()
    return [str(row[0]) for row in rows]


def _batch_outcome(statuses: list[str], empty_outcome: str) -> str:
    if "FILLED" in statuses:
        return "FILLED"
    if statuses and all(status == "VETO" for status in statuses):
        return "VETO"
    if statuses:
        raise RuntimeError(f"unresolved V5.1 intent statuses: {statuses}")
    if empty_outcome not in {"ALREADY_AT_TARGET", "EMPTY"}:
        raise ValueError(f"invalid empty batch outcome: {empty_outcome}")
    return empty_outcome


def _resolve_batches(ledger: WC006Ledger, execution_date: str, current_effective: Decimal) -> list[dict[str, Any]]:
    """Resolve only exact execution-date batches and commit filled state."""
    _ensure_v51_batches(ledger)
    rows = ledger.db.execute(
        "SELECT batch_id,batch_kind,request_date,execution_date,requested_multiplier,intent_ids_json,outcome,empty_outcome,effective_multiplier_before,reward_linkage_id,reward_delivery_status FROM v51_batches WHERE execution_date=? AND outcome='PENDING' ORDER BY request_date,batch_id",
        (execution_date,),
    ).fetchall()
    effective = current_effective
    resolved: list[dict[str, Any]] = []
    for batch_id, kind, request_date, batch_execution_date, requested_raw, ids_raw, prior_outcome, empty_outcome, before_raw, prior_reward_id, prior_reward_status in rows:
        if batch_execution_date != execution_date:
            raise RuntimeError("V5.1 reward linkage crossed execution date")
        expected_reward_id = _reward_linkage_id(str(batch_id), execution_date)
        if prior_reward_id not in (None, expected_reward_id) or str(prior_reward_status) != "UNLINKED":
            raise RuntimeError(f"V5.1 duplicate reward linkage state for batch {batch_id}")
        intent_ids = list(json.loads(ids_raw))
        statuses = _intent_statuses(ledger, intent_ids)
        outcome = _batch_outcome(statuses, str(empty_outcome))
        requested = D(requested_raw)
        if outcome in {"FILLED", "ALREADY_AT_TARGET"}:
            effective = requested
        reward_id = _reward_linkage_id(str(batch_id), execution_date)
        reward_status = "PENDING" if outcome in _REWARD_ELIGIBLE_OUTCOMES else "NONE"
        ledger.db.execute(
            "UPDATE v51_batches SET outcome=?,effective_multiplier_after=?,reward_linkage_id=?,reward_delivery_status=?,reward_label=?,reward_delivery_date=? WHERE batch_id=?",
            (outcome, str(effective), reward_id, reward_status, "none", execution_date, batch_id),
        )
        resolved.append({
            "batch_id": batch_id,
            "batch_kind": kind,
            "request_date": request_date,
            "execution_date": execution_date,
            "requested_multiplier": str(requested),
            "effective_multiplier_before": str(before_raw),
            "effective_multiplier_after": str(effective),
            "intent_ids": intent_ids,
            "intent_statuses": statuses,
            "outcome": outcome,
            "reward_linkage_id": reward_id,
            "reward_delivery_status": reward_status,
            "delivered_label": None,
        })
    ledger.put_meta("effective_multiplier", str(effective))
    ledger.db.commit()
    return resolved


def _classify_empty_batch(plans: list[dict[str, Any]], intent_ids: list[str]) -> str:
    if intent_ids:
        return "PENDING"
    if all(D(str(plan["quantity"])) == 0 for plan in plans):
        # A delta below one executable lot is already at target for paper
        # execution, even when Decimal delta is not exactly zero.
        return "ALREADY_AT_TARGET"
    return "EMPTY"


def _normalise_digest_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalise_digest_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalise_digest_value(item) for item in value]
    if hasattr(value, "tobytes") and hasattr(value, "shape"):
        return {"sha256": hashlib.sha256(value.tobytes()).hexdigest(), "shape": list(value.shape)}
    return str(value)


def _controller_digests(controller: Any) -> tuple[str, str]:
    brain = getattr(controller, "brain", None)
    memory = brain.memory() if brain is not None and callable(getattr(brain, "memory", None)) else None
    controller_state = {key: value for key, value in vars(controller).items() if key not in {"brain", "s", "decoder", "reward_calls", "linked_reinforcement_records", "last_linked_reinforcement", "last_compute_seconds"}} if hasattr(controller, "__dict__") else {}
    trace_payload = {
        "memory": memory,
        "eligibility": getattr(brain, "eligibility", None),
        "eligibility_last": getattr(brain, "eligibility_last", None),
        "controller": controller_state,
    }
    if isinstance(memory, dict) and memory.get("sha256") is not None:
        weight_payload = {"weight_sha256": memory["sha256"]}
    else:
        weight_payload = {"brain": vars(brain) if brain is not None and hasattr(brain, "__dict__") else brain, "controller": controller_state}
    trace_digest = hashlib.sha256(json.dumps(_normalise_digest_value(trace_payload), sort_keys=True).encode()).hexdigest()
    weight_digest = hashlib.sha256(json.dumps(_normalise_digest_value(weight_payload), sort_keys=True).encode()).hexdigest()
    return trace_digest, weight_digest


def _deliver_batch_rewards(ledger: WC006Ledger, resolved: list[dict[str, Any]], feedback: dict[str, Any], controller: Any, execution_date: str) -> list[dict[str, Any]]:
    pending = ledger.db.execute("SELECT batch_id,reward_delivery_status FROM v51_batches WHERE reward_delivery_status IN ('PENDING','IN_FLIGHT') ORDER BY batch_id").fetchall()
    resolved_ids = {str(item["batch_id"]) for item in resolved}
    stale = [str(row[0]) for row in pending if str(row[0]) not in resolved_ids]
    if stale:
        raise RuntimeError(f"V5.1 orphan or duplicate pending reward linkage: {stale}")
    eligible = [item for item in resolved if item["outcome"] in _REWARD_ELIGIBLE_OUTCOMES]
    if len(eligible) > 1:
        raise RuntimeError("V5.1 hard-stop: more than one batch awaits reward delivery")
    deliveries: list[dict[str, Any]] = []
    for item in resolved:
        batch_id = str(item["batch_id"])
        if item["outcome"] not in _REWARD_ELIGIBLE_OUTCOMES:
            item.update({"reward_delivery_status": "NONE", "delivered_label": None})
            deliveries.append({"batch_id": batch_id, "status": "NONE", "delivered_label": None})
            continue
        if execution_date != str(item["execution_date"]):
            raise RuntimeError("V5.1 reward delivery crossed execution date")
        state_row = ledger.db.execute("SELECT reward_linkage_id,reward_delivery_status FROM v51_batches WHERE batch_id=?", (batch_id,)).fetchone()
        if state_row is None or str(state_row[0]) != str(item.get("reward_linkage_id")):
            raise RuntimeError(f"V5.1 orphan reward linkage for batch {batch_id}")
        if str(state_row[1]) != "PENDING":
            raise RuntimeError(f"V5.1 duplicate reward delivery for batch {batch_id}")
        label = str(feedback.get("stimulus", "none"))
        if label not in {"reward", "aversive", "none"}:
            raise RuntimeError(f"V5.1 invalid categorical reward label: {label}")
        linked = getattr(controller, "reinforce_linked", None)
        only = getattr(controller, "reinforce_only", None)
        if not callable(linked) or not callable(only):
            raise RuntimeError("V5.1 mechanism B NOT_PASSED: controller lacks reinforce_only/reinforce_linked")
        before_trace, before_weight = _controller_digests(controller)
        ledger.db.execute("UPDATE v51_batches SET reward_delivery_status='IN_FLIGHT',reward_label=? WHERE batch_id=?", (label, batch_id))
        ledger.db.commit()
        mechanism = linked(label, batch_id)
        if not isinstance(mechanism, dict) or not isinstance(mechanism.get("status"), str):
            raise RuntimeError("V5.1 mechanism B returned no structured reinforcement record")
        mechanism_status = str(mechanism["status"])
        if mechanism_status not in {"APPLIED", "NO_OP_NONE", "INCONCLUSIVE_NO_ELIGIBILITY"}:
            raise RuntimeError(f"V5.1 mechanism B returned invalid status: {mechanism_status}")
        delivery_status = "INCONCLUSIVE_NO_ELIGIBILITY" if mechanism_status == "INCONCLUSIVE_NO_ELIGIBILITY" else "DELIVERED"
        after_trace, after_weight = _controller_digests(controller)
        ledger.db.execute("UPDATE v51_batches SET reward_delivery_status=?,reward_label=?,reward_trace_before_digest=?,reward_trace_after_digest=?,reward_weight_before_digest=?,reward_weight_after_digest=? WHERE batch_id=?", (delivery_status, label, before_trace, after_trace, before_weight, after_weight, batch_id))
        ledger.db.commit()
        item.update({"reward_delivery_status": delivery_status, "delivered_label": label, "trace_before_digest": before_trace, "trace_after_digest": after_trace, "weight_before_digest": before_weight, "weight_after_digest": after_weight, "mechanism": mechanism})
        deliveries.append({"batch_id": batch_id, "status": delivery_status, "delivered_label": label, "trace_before_digest": before_trace, "trace_after_digest": after_trace, "weight_before_digest": before_weight, "weight_after_digest": after_weight, "mechanism": mechanism})
    return deliveries


def _record_settlement_only_rewards(
    ledger: WC006Ledger,
    resolved: list[dict[str, Any]],
    feedback: dict[str, Any],
    execution_date: str,
) -> list[dict[str, Any]]:
    """Close boundary reward linkages without opening a neural observation.

    The annual protocol makes the T+1 boundary settlement-only: it may close
    an intent, but creates no signal, observation, or learning pulse.  We
    therefore persist the exact feedback that would have selected a
    categorical label and mark the linkage ``SETTLEMENT_ONLY``.
    """
    label = str(feedback.get("stimulus", "none"))
    if label not in {"reward", "aversive", "none"}:
        raise RuntimeError(f"V5.1 invalid boundary reward label: {label}")
    feedback_payload = {
        key: str(value) if isinstance(value, Decimal) else value
        for key, value in feedback.items()
    }
    deliveries: list[dict[str, Any]] = []
    for item in resolved:
        batch_id = str(item["batch_id"])
        row = ledger.db.execute(
            "SELECT reward_linkage_id,reward_delivery_status FROM v51_batches WHERE batch_id=?",
            (batch_id,),
        ).fetchone()
        if row is None or str(row[0]) != str(item.get("reward_linkage_id")):
            raise RuntimeError(f"V5.1 orphan boundary reward linkage for batch {batch_id}")
        status = str(row[1])
        if item["outcome"] in _REWARD_ELIGIBLE_OUTCOMES:
            if status not in {"PENDING", "IN_FLIGHT", "SETTLEMENT_ONLY"}:
                raise RuntimeError(f"V5.1 invalid boundary reward state {status} for {batch_id}")
            ledger.db.execute(
                "UPDATE v51_batches SET reward_delivery_status='SETTLEMENT_ONLY',reward_label=?,reward_delivery_date=? WHERE batch_id=?",
                (label, execution_date, batch_id),
            )
            item.update({"reward_delivery_status": "SETTLEMENT_ONLY", "delivered_label": label})
            deliveries.append({
                "batch_id": batch_id,
                "status": "SETTLEMENT_ONLY",
                "delivered_label": label,
                "feedback": feedback_payload,
            })
        else:
            if status not in {"NONE", "SETTLEMENT_ONLY"}:
                ledger.db.execute(
                    "UPDATE v51_batches SET reward_delivery_status='NONE',reward_label='none',reward_delivery_date=? WHERE batch_id=?",
                    (execution_date, batch_id),
                )
            item.update({"reward_delivery_status": "NONE", "delivered_label": None})
            deliveries.append({"batch_id": batch_id, "status": "NONE", "delivered_label": None})
    ledger.db.commit()
    return deliveries


def _v51_batch_completion_issues(ledger: WC006Ledger) -> list[str]:
    """Return unresolved V5.1 batch/reward states that forbid completion."""
    _ensure_v51_batches(ledger)
    rows = ledger.db.execute(
        "SELECT batch_id,outcome,reward_delivery_status FROM v51_batches ORDER BY request_date,batch_id"
    ).fetchall()
    issues: list[str] = []
    for batch_id, outcome, reward_status in rows:
        if str(outcome) == "PENDING":
            issues.append(f"{batch_id}:PENDING")
        elif str(outcome) not in _ALLOWED_OUTCOMES:
            issues.append(f"{batch_id}:invalid-outcome:{outcome}")
        if str(reward_status) in {"UNLINKED", "PENDING", "IN_FLIGHT"}:
            issues.append(f"{batch_id}:{reward_status}")
        elif str(reward_status) not in _REWARD_STATES:
            issues.append(f"{batch_id}:invalid-reward:{reward_status}")
    return issues


def _prepare_target_intents(
    ledger: WC006Ledger,
    record: Any,
    *,
    date: str,
    multiplier: Decimal,
    neural_side: str,
    observation_id: int,
    checkpoint_sha256: str,
    stream: V4Stream,
    target_codes: set[str],
    intent_kind: str,
) -> tuple[list[dict[str, Any]], list[str], str]:
    """Prepare a target batch without selecting a replacement instrument."""
    prices = stream.market.raw_prices if stream.market is not None else stream.prices
    active = {str(item["code"]): item for item in record.targets}
    held = {str(row[0]) for row in ledger.db.execute("SELECT DISTINCT symbol FROM lots WHERE CAST(quantity AS REAL)>0")}
    all_codes = sorted(set(active) | held)
    plans: list[dict[str, Any]] = []
    intent_ids: list[str] = []
    for code in all_codes:
        target = active.get(code, {"code": code, "weight": D("0"), "score": None})
        close = (stream.market and __import__("experiments.a_share.wc006_v4_market", fromlist=["raw_mark"]).raw_mark(stream.market, code, date)[1]) if stream.market is not None else prices[code][max(item for item in prices[code] if item <= date)]["close"]
        equity = (__import__("experiments.a_share.wc006_v4_market", fromlist=["raw_equity"]).raw_equity(ledger, stream.market, date, set(all_codes))[0] if stream.market is not None else _equity(ledger, stream.prices, date, "close", set(all_codes)))
        target_exposure = D(str(target["weight"])) * RISK_BUDGET * multiplier
        current_value = ledger.holdings(code) * close
        delta = equity * target_exposure - current_value
        quantity = (abs(delta) / close / LOT).to_integral_value(rounding="ROUND_DOWN") * LOT if delta != 0 else D("0")
        side = "BUY" if delta > 0 else "SELL"
        item: dict[str, Any] = {"code": code, "weight": target["weight"], "target_exposure": str(target_exposure), "side": side, "quantity": str(quantity), "delta": str(delta)}
        if quantity > 0:
            intent = ledger.prepare_intent(signal_date=date, execution_date=next((d for d in stream.base.calendar if d > date), "END"), symbol=code, neural_side=neural_side, order_side=side, intent_kind=intent_kind, target_exposure=target_exposure, requested_qty=quantity, observation_id=observation_id, checkpoint_sha256=checkpoint_sha256)
            item["intent_id"] = intent
            intent_ids.append(intent)
        plans.append(item)
    return plans, intent_ids, _classify_empty_batch(plans, intent_ids)


def _create_batch(
    ledger: WC006Ledger,
    *,
    kind: str,
    request_date: str,
    execution_date: str,
    candidate_epoch: int,
    requested: Decimal,
    effective: Decimal,
    requested_target_exposure: Decimal,
    effective_target_exposure: Decimal,
    intent_ids: list[str],
    empty_outcome: str,
) -> dict[str, Any]:
    if kind not in _ALLOWED_BATCH_KINDS:
        raise ValueError(f"invalid V5.1 batch kind: {kind}")
    _ensure_v51_batches(ledger)
    batch_id = _batch_id(request_date=request_date, kind=kind, candidate_epoch=candidate_epoch, requested=requested, intent_ids=intent_ids)
    reward_id = _reward_linkage_id(batch_id, execution_date)
    ledger.db.execute(
        "INSERT OR IGNORE INTO v51_batches(batch_id,batch_kind,request_date,execution_date,candidate_epoch,requested_multiplier,effective_multiplier_before,requested_target_exposure,effective_target_exposure,intent_ids_json,outcome,empty_outcome,effective_multiplier_after,reward_linkage_id,reward_delivery_status,reward_label,reward_delivery_date,reward_trace_before_digest,reward_trace_after_digest,reward_weight_before_digest,reward_weight_after_digest) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (batch_id, kind, request_date, execution_date, candidate_epoch, str(requested), str(effective), str(requested_target_exposure), str(effective_target_exposure), json.dumps(intent_ids, sort_keys=True), "PENDING", empty_outcome, str(effective), reward_id, "UNLINKED", "none", None, None, None, None, None),
    )
    ledger.db.commit()
    return {"batch_id": batch_id, "batch_kind": kind, "request_date": request_date, "execution_date": execution_date, "requested_multiplier": str(requested), "effective_multiplier_before": str(effective), "requested_target_exposure": str(requested_target_exposure), "effective_target_exposure": str(effective_target_exposure), "intent_ids": intent_ids, "outcome": empty_outcome, "reward_linkage_id": reward_id, "reward_delivery_status": "UNLINKED", "delivered_label": None}



def _v51_event_id(ledger: WC006Ledger, date: str) -> str:
    """Return the same stable CLOSE event id used by WC006Ledger.record_event."""
    identity = {"signature": ledger.get_meta("signature"), "date": str(date), "phase": "CLOSE"}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _frame_observation_payload(
    record: Any,
    prices: dict[str, dict[str, dict[str, Decimal]]],
    date: str,
    calendar: tuple[str, ...],
    ledger: WC006Ledger,
    multiplier: Decimal,
    drawdown: Decimal,
    equity: Decimal,
    base_equity: Decimal,
    portfolio_exposure: Decimal,
    frame: Any,
) -> dict[str, Any]:
    """Capture the causal values used by ``_frame`` without changing it.

    This deliberately mirrors the renderer's calculations (including its
    61-bar source and 60-point plotted paths). It is a side-channel audit
    record, never an input to the controller or decoder.
    """
    history_dates = [item for item in calendar if item <= date][-LOOKBACK_CLOSES:]
    frame_equity = _equity(ledger, prices, date, "close", {str(item["code"]) for item in record.targets})
    cash = D(ledger.get_meta("cash"))
    cash_fraction = cash / frame_equity if frame_equity else D("0")
    rows: list[dict[str, Any]] = []
    row_height = max(1, 150 // len(record.targets))
    for index, target in enumerate(record.targets):
        code = str(target["code"])
        available_bars = sorted((bar_date, row) for bar_date, row in prices.get(code, {}).items() if bar_date <= date)[-LOOKBACK_CLOSES:]
        if len(available_bars) < LOOKBACK_CLOSES:
            raise RuntimeError(f"{code}: frame log lookback incomplete at {date}")
        feature_closes = [row["close"] for _, row in available_bars]
        feature_volumes = [row.get("volume") or D("0") for _, row in available_bars]
        normalized = [float(value / feature_closes[0] - D("1")) for value in feature_closes[-LOOKBACK:]]
        vmax = max(feature_volumes) or D("1")
        volume_norm = [float(value / vmax) for value in feature_volumes[-LOOKBACK:]]
        r5, v5 = _stats(feature_closes, 5)
        r20, v20 = _stats(feature_closes, 20)
        r60, v60 = _stats(feature_closes, 60)
        current_exposure = ledger.holdings(code) * feature_closes[-1] / frame_equity if frame_equity else D("0")
        target_exposure = D(str(target["weight"])) * RISK_BUDGET * multiplier
        stale = date not in prices.get(code, {})
        top = 24 + index * row_height
        bottom = min(176, top + row_height - 4)
        rows.append({
            "code": code,
            "selector_target_weight": str(target.get("weight")),
            # Never reconstruct rank from target position: these values come
            # only from candidate-stream metadata.
            "selector_score": (target.get("score") if target.get("score") is not None else "unavailable"),
            "selector_score_status": "available" if target.get("score") is not None else "unavailable",
            "selector_rank": (target.get("rank") if target.get("rank") is not None else "unavailable"),
            "selector_rank_status": "available" if target.get("rank") is not None else "unavailable",
            "target_exposure": str(target_exposure),
            "current_exposure": str(current_exposure),
            "stale": bool(stale),
            "missing_bar": bool(stale),
            "return_5": float(r5),
            "return_20": float(r20),
            "return_60": float(r60),
            "volatility_5": float(v5),
            "volatility_20": float(v20),
            "volatility_60": float(v60),
            # Raw paths are retained for quantitative replay. The normalized
            # paths are the exact values passed to the line drawing code.
            "path_semantics": "close_path_60/volume_path_60 are raw causal values; plot_* are normalized renderer values",
            "close_path_60": [str(value) for value in feature_closes[-LOOKBACK:]],
            "volume_path_60": [str(value) for value in feature_volumes[-LOOKBACK:]],
            "plot_close_path_60": normalized,
            "plot_volume_path_60": volume_norm,
            "plot_geometry": {"row_index": index, "top": top, "bottom": bottom},
        })
    return {
        "date": str(date),
        "multiplier": str(multiplier),
        "cash_fraction": str(cash_fraction),
        "drawdown": str(drawdown),
        "equity": str(equity),
        "base_equity": str(base_equity),
        "portfolio_exposure": str(portfolio_exposure),
        "visual": {
            "frame_schema": V51_FRAME_SCHEMA,
            "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
            "shape": list(frame.shape),
            "dtype": str(frame.dtype),
            "header": {
                "date": str(date),
                "multiplier": float(multiplier),
                "cash_fraction": float(cash_fraction),
                "drawdown": float(drawdown),
            },
            "history_dates": history_dates,
            "candidates": rows,
        },
        "candidates": rows,
    }


def _fills_for_date(ledger: WC006Ledger, date: str) -> list[dict[str, Any]]:
    rows = ledger.db.execute(
        "SELECT f.intent_id,f.execution_date,f.symbol,f.quantity,f.price,f.notional,f.fee,i.observation_id,i.order_side,i.neural_side,i.intent_kind "
        "FROM fills f JOIN intents i ON i.id=f.intent_id WHERE f.execution_date=? ORDER BY f.intent_id",
        (date,),
    ).fetchall()
    columns = ("intent_id", "execution_date", "code", "quantity", "price", "notional", "fee", "observation_id", "order_side", "neural_side", "intent_kind")
    return [
        {key: (str(value) if key in {"quantity", "price", "notional", "fee"} else value) for key, value in zip(columns, row)}
        for row in rows
    ]


def _vetoes_for_date(ledger: WC006Ledger, date: str) -> list[dict[str, Any]]:
    rows = ledger.db.execute(
        "SELECT id,symbol,veto_reason,observation_id,neural_side,order_side,intent_kind FROM intents "
        "WHERE execution_date=? AND status='VETO' ORDER BY id", (date,)
    ).fetchall()
    columns = ("intent_id", "code", "reason", "observation_id", "neural_side", "order_side", "intent_kind")
    return [dict(zip(columns, row)) for row in rows]



def _write_observation_outcomes(output: Path, events_path: Path) -> None:
    """Materialize execution/reward outcomes keyed by source observation.

    A CLOSE observation requests a future execution. The settlement event is
    separate, so this view associates fills, vetoes, batches, and
    reinforcement with the originating observation, not the next observation.
    """
    if not events_path.exists():
        return
    events: list[dict[str, Any]] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if isinstance(event, dict) and isinstance(event.get("observation"), dict):
            events.append(event)
    by_observation_id: dict[int, dict[str, Any]] = {}
    by_date: dict[str, dict[str, Any]] = {}
    records: dict[int, dict[str, Any]] = {}
    for event in events:
        observation = event["observation"]
        raw_id = observation.get("observation_id")
        if raw_id is None:
            continue
        observation_id = int(raw_id)
        by_observation_id[observation_id] = event
        by_date[str(event.get("date"))] = event
        execution = event.get("execution") or {}
        records[observation_id] = {
            "schema": "wc006-v5.1-observation-outcome/v2",
            "event_id": event.get("event_id"),
            "observation_id": observation_id,
            "observation_date": event.get("date"),
            "source_event_id": event.get("event_id"),
            "source_observation_id": observation_id,
            "source_observation_date": event.get("date"),
            "next_execution_request": execution.get("next_execution_request"),
            "execution_outcomes": [],
            "reinforcement_outcomes": [],
        }
    for settlement_event in events:
        execution = settlement_event.get("execution") or {}
        settled = execution.get("settled_prior_observation_outcome") or {}
        fills = settled.get("fills") or []
        vetoes = settled.get("vetoes") or []
        resolved = settled.get("resolved_batches") or []
        deliveries = settled.get("reward_delivery") or []
        source_ids = {int(value) for value in execution.get("source_observation_ids", []) if value is not None}
        # Already-at-target batches have no intent/fill; request_date is their
        # authoritative source-observation linkage.
        for batch in resolved:
            if isinstance(batch, dict):
                source = by_date.get(str(batch.get("request_date")))
                source_id = ((source or {}).get("observation") or {}).get("observation_id")
                if source_id is not None:
                    source_ids.add(int(source_id))
        for source_id in sorted(source_ids):
            record = records.get(source_id)
            if record is None:
                # The source can be outside a truncated slice; do not invent
                # an event id or associate its result with another event.
                continue
            source_date = str(by_observation_id[source_id].get("date"))
            source_batches = [
                item for item in resolved
                if isinstance(item, dict) and str(item.get("request_date")) == source_date
            ]
            source_batch_ids = {str(item.get("batch_id")) for item in source_batches}
            source_fills = [item for item in fills if isinstance(item, dict) and item.get("observation_id") == source_id]
            source_vetoes = [item for item in vetoes if isinstance(item, dict) and item.get("observation_id") == source_id]
            source_deliveries = [
                item for item in deliveries
                if isinstance(item, dict) and str(item.get("batch_id")) in source_batch_ids
            ]
            source_turnover = sum((D(str(item.get("notional") or "0")) for item in source_fills), D("0"))
            source_fees = sum((D(str(item.get("fee") or "0")) for item in source_fills), D("0"))
            settlement_global = {
                "portfolio_return": settled.get("portfolio_return"),
                "base_return": settled.get("base_return"),
                "reward_score": settled.get("reward_score"),
                "reinforcement_classification": settled.get("reinforcement_classification", "none"),
            }
            outcome = {
                "settlement_event_id": settlement_event.get("event_id"),
                "execution_date": settlement_event.get("date"),
                "fills": source_fills,
                "vetoes": source_vetoes,
                "guard_reasons": [item.get("reason") for item in source_vetoes if item.get("reason") is not None],
                "turnover": str(source_turnover),
                "fees": str(source_fees),
                "settlement_global": settlement_global,
                "resolved_batches": source_batches,
                "reward_delivery": source_deliveries,
            }
            record["execution_outcomes"].append(outcome)
            record["reinforcement_outcomes"].append({
                "settlement_event_id": settlement_event.get("event_id"),
                "execution_date": settlement_event.get("date"),
                "settlement_global": settlement_global,
                "reward_delivery": source_deliveries,
            })
    path = output / "observation_outcomes.jsonl"
    rows = [records[key] for key in sorted(records)]
    text = "" if not rows else "\n".join(json.dumps(item, ensure_ascii=False, allow_nan=False, sort_keys=True) for item in rows) + "\n"
    path.write_text(text, encoding="utf-8")


def _write_forward_outcomes(
    output: Path,
    events_path: Path,
    prices: dict[str, dict[str, dict[str, Decimal]]],
    calendar: tuple[str, ...],
) -> None:
    """Materialize causal-independent, post-observation basket outcomes.

    The file is intentionally separate from the immutable event hash chain.
    A basket return is a weighted close-to-close target-basket path; it is not
    presented as a single-stock return or fed back into the strategy.
    """
    if not events_path.exists():
        return
    events: list[dict[str, Any]] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if isinstance(item, dict) and item.get("event_id") and item.get("observation"):
                events.append(item)
    index = {str(value): i for i, value in enumerate(calendar)}
    event_by_date = {str(item.get("date")): item for item in events}
    records: list[dict[str, Any]] = []
    for event in events:
        observation = event["observation"]
        date = str(event["date"])
        targets = observation.get("candidates", [])
        start_idx = index.get(date)
        result: dict[str, Any] = {
            "schema": V51_FORWARD_OUTCOME_SCHEMA,
            "event_id": str(event["event_id"]),
            "observation_id": observation.get("observation_id"),
            "observation_date": date,
            "path_type": "target_basket_close",
            "basket_path_semantics": "weighted target-basket close-to-close return; not a single-stock forward return",
            "portfolio_path_semantics": "realized fly portfolio return compounded from subsequent event reward.portfolio_return values",
            "status": "available",
        }
        for horizon in (1, 3, 5):
            key = f"forward_basket_return_{horizon}d"
            future_idx = None if start_idx is None else start_idx + horizon
            if future_idx is None or future_idx >= len(calendar):
                result[key] = None
                result[f"{key}_status"] = "unavailable"
                continue
            future_date = calendar[future_idx]
            weighted = D("0")
            valid = True
            for target in targets:
                code = str(target["code"])
                try:
                    start_row = prices[code][date]
                    future_row = prices[code][future_date]
                    start_close, future_close = start_row.get("close"), future_row.get("close")
                    if start_close is None or future_close is None:
                        valid = False
                        break
                    weighted += D(str(target["selector_target_weight"])) * (future_close / start_close - D("1"))
                except (KeyError, TypeError, ValueError):
                    valid = False
                    break
            result[key] = str(weighted.normalize()) if valid else None
            result[f"{key}_status"] = "available" if valid else "unavailable"
            portfolio_key = f"forward_portfolio_return_{horizon}d"
            portfolio_value = D("1")
            portfolio_valid = True
            for step in range(1, horizon + 1):
                step_date = calendar[start_idx + step]
                step_event = event_by_date.get(step_date)
                try:
                    portfolio_value *= D("1") + D(str((step_event or {}).get("reward", {}).get("portfolio_return")))
                except (TypeError, ValueError, ArithmeticError):
                    portfolio_valid = False
                    break
            result[portfolio_key] = str(portfolio_value - D("1")) if portfolio_valid else None
            result[f"{portfolio_key}_status"] = "available" if portfolio_valid else "unavailable"
            result[f"forward_date_{horizon}d"] = future_date
        if any(result[f"forward_basket_return_{horizon}d"] is None or result[f"forward_portfolio_return_{horizon}d"] is None for horizon in (1, 3, 5)):
            result["status"] = "partially_available"
        records.append(result)
    outcome_path = output / "forward_outcomes.jsonl"
    text = "" if not records else "\n".join(json.dumps(item, ensure_ascii=False, allow_nan=False, sort_keys=True) for item in records) + "\n"
    outcome_path.write_text(text, encoding="utf-8")


def _run_v51(
    stream_path: Path,
    output: Path,
    *,
    start_date: str,
    end_date: str,
    formal: bool,
    market_manifest: Path | None,
    max_days: int,
    expected_records: int,
    training_end: str,
    controller_factory: Callable[[Settings], Any],
    arm: str,
    seed: int,
    plasticity_mode: str,
    sensory_adapter: SensoryAdapter | None = None,
    annual_input_gate: dict[str, Any] | None = None,
    annual_payload: dict[str, Any] | None = None,
    market_expected_parent_sha256: str | None = None,
    market_expected_qa_sha256: str | None = None,
) -> dict[str, Any]:
    if arm not in V51_ARMS:
        raise ValueError(f"unsupported V5.1 arm: {arm}")
    if formal:
        # No V5 authorization/code-lock can authorize this independent adapter.
        raise RuntimeError("V5.1 formal blocked: new V5.1 authorization artifact is missing")
    if seed < 0:
        raise ValueError("V5.1 seed must be non-negative")
    plasticity_mode = _validate_plasticity_mode(plasticity_mode)
    sensory_adapter_provenance = _sensory_adapter_provenance(sensory_adapter)
    if annual_input_gate is not None:
        if market_manifest is None or market_expected_parent_sha256 is None or market_expected_qa_sha256 is None:
            raise RuntimeError("annual input gate: corrected market pins are required")
        stream = validate_v4_stream(
            stream_path,
            expected_records=expected_records,
            # The source stream is a full 2021-2023 candidate stream.  V4's
            # formal training-end pin is retained for source validation; the
            # annual scope itself is bound below from the annual manifest.
            training_end=TRAIN_END,
            market_manifest=market_manifest,
            market_expected_parent_sha256=market_expected_parent_sha256,
            market_expected_qa_sha256=market_expected_qa_sha256,
        )
        if stream.market is None:
            raise RuntimeError("annual input gate: xysz raw execution market was not loaded")
        candidate = json.loads(Path(stream_path).read_text(encoding="utf-8"))
        if candidate.get("paper_only") is not True or candidate.get("broker_orders_enabled") is not False or candidate.get("live_execution_authorized") is not False:
            raise RuntimeError("V5.1 input gate: candidate stream is not paper-only")
        artifact_hashes = {str(k): str(v) for k, v in stream.market.artifact_hashes.items()}
        if annual_payload is not None:
            _assert_annual_source_match({"payload": annual_payload}, stream)
        input_gate = {
            "status": "PASS", "candidate_manifest_sha256": sha256(Path(stream_path)),
            "market_manifest_sha256": market_expected_parent_sha256,
            "market_qa_sha256": market_expected_qa_sha256,
            "market_artifact_hashes": {"manifest": market_expected_parent_sha256, "qa": market_expected_qa_sha256, **artifact_hashes},
            "training_authorized": False, "formal": False, "records": len(stream.base.records),
            "calendar_days": len(stream.base.calendar), "training_end": training_end,
            "paper_only": True, "broker_orders_enabled": False, "live_execution_authorized": False,
            "training_window": [stream.base.records[0].signal_date, training_end],
        }
    else:
        stream, input_gate = validate_v5_inputs(
            stream_path, expected_records=expected_records, training_end=training_end,
            market_manifest=market_manifest, formal=False
        )
    if annual_input_gate is not None:
        input_gate["annual_input"] = annual_input_gate
    input_gate["plasticity_mode"] = plasticity_mode
    scope = {"formal": False, "relative": str(Path(output).resolve())}
    settlement_boundary = annual_payload.get("settlement_boundary") if annual_payload is not None else None
    dates, settlement_date = _scope_dates(
        stream, start_date, end_date, settlement_boundary=settlement_boundary
    )
    records = {item.signal_date: (idx, item) for idx, item in enumerate(stream.base.records) if start_date <= item.signal_date <= end_date}
    prior_records = [item for item in stream.base.records if item.signal_date <= start_date]
    if not prior_records:
        raise ValueError("no candidate basket available at V5.1 scope start")
    output = Path(output).resolve()
    if output.exists() and not (output / "ledger.sqlite").exists():
        existing = {item.name for item in output.iterdir()}
        if existing - {"training.log"}:
            raise RuntimeError("V5.1 output must be new or resumable")
    output.mkdir(parents=True, exist_ok=True)
    controller_settings = Settings(learning=plasticity_mode == "enabled")
    fly = controller_factory(controller_settings)
    _assert_controller_plasticity_mode(fly, plasticity_mode, context="initial controller")
    if sensory_adapter is not None and not isinstance(sensory_adapter, NullSensoryAdapter):
        # Validate static mapping immediately after the fly is created, before
        # any checkpoint, neural state, or market state can advance.
        validator = getattr(sensory_adapter, "validate_against_graph", None)
        if not callable(validator):
            raise TypeError(
                "non-Null sensory_adapter must provide validate_against_graph(ids)"
            )
        try:
            validator(fly.brain.ids)
        except AttributeError as exc:
            raise RuntimeError(
                "controller brain must expose graph ids for non-Null sensory_adapter"
            ) from exc
    graph_lock = _graph_source_lock(fly)
    graph_hash = graph_lock["configuration_sha256"]
    mechanism_provenance = _mechanism_provenance(
        fly, getattr(fly, "s", controller_settings)
    )
    signature_payload = {"schema": V51_SCHEMA, "arm": arm, "seed": seed, "plasticity_mode": plasticity_mode, "scope": [start_date, end_date], "stream": stream.base.stream_sha256, "spec": stream.spec_sha256, "graph": graph_lock, "v4_adapter": _v4_adapter_sha256(), "adapter": sha256(Path(__file__).resolve()), "market": input_gate.get("market_artifact_hashes", {}), "annual_input": input_gate.get("annual_input")}
    signature_payload["sensory_adapter"] = sensory_adapter_provenance
    signature_payload["mechanism"] = mechanism_provenance
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
    ledger = WC006Ledger(output / "ledger.sqlite", initial_state={"cash": "100000", "lots": []}, signature=signature, authoritative_trading_state=stream.market is not None)
    base = WC006Ledger(output / "paired_base.sqlite", initial_state={"cash": "100000", "lots": []}, signature=signature + ":paired-base", authoritative_trading_state=stream.market is not None)
    try:
        _ensure_v51_batches(ledger)
        last_raw = ledger.get_meta("last_completed_index")
        base_last_raw = base.get_meta("last_completed_index")
        last_day = -1 if last_raw is None else int(last_raw)
        if last_day != (-1 if base_last_raw is None else int(base_last_raw)):
            raise RuntimeError("V5.1 primary/paired indices diverge")
        clean = output / "brain-clean.npz"
        checkpoint = ledger.get_meta("checkpoint")
        if checkpoint:
            if not clean.is_file() or sha256(clean) != ledger.get_meta("initial_brain_sha256"):
                raise RuntimeError("V5.1 initial brain hash mismatch")
            checkpoint_path = _checkpoint_path(output, checkpoint)
            fly.restore(checkpoint_path)
            _assert_controller_plasticity_mode(fly, plasticity_mode, context="restored checkpoint")
        else:
            fly.save(clean)
            ledger.put_meta("initial_brain_sha256", sha256(clean)); ledger.put_meta("effective_multiplier", "1.0"); ledger.db.commit(); base.db.commit()
        provenance_path = output / "provenance.json"
        static = {"schema": V51_SCHEMA, "arm": V51_PROTOCOL_ARM, "protocol_arm": V51_PROTOCOL_ARM, "comparison_arms": list(V51_COMPARISON_ARMS), "seed": seed, "plasticity_mode": plasticity_mode, "stream_sha256": stream.base.stream_sha256, "spec_sha256": stream.spec_sha256, "graph_manifest_sha256": graph_hash, "adapter_sha256": sha256(Path(__file__).resolve()), "initial_brain": "clean/new; no historical checkpoint", "initial_brain_sha256": ledger.get_meta("initial_brain_sha256"), "paper_only": True, "broker_orders_enabled": False, "live_execution_authorized": False, "training_authorized": input_gate.get("training_authorized", False), "training_window": [start_date, end_date], "settlement_date": settlement_date, "candidate_selection": "frozen stream basket; updates only on selector dates", "exposure_transition": "non-selector requested/effective mismatch creates independent paper batch", "reward_mechanism": REWARD_MECHANISM, "formal_status": "BLOCKED_NO_V51_AUTHORIZATION_AND_MECHANISM_B", "input_gate": input_gate, "output_scope": scope, "risk_budget": str(RISK_BUDGET), "costs": CostConfig().as_json()}
        static["sensory_adapter"] = sensory_adapter_provenance
        # Bind the complete neural/feedback mechanism, not only the decoder.
        static["mechanism"] = mechanism_provenance
        # Retain these top-level aliases for readers of earlier research runs.
        static["decoder"] = mechanism_provenance["decoder"]
        static["rule_parameters"] = mechanism_provenance["rule_parameters"]
        static["git"] = _git_provenance()
        if not provenance_path.exists():
            provenance_path.write_text(json.dumps(static, indent=2, sort_keys=True) + "\n")
        else:
            existing = json.loads(provenance_path.read_text())
            for key, value in static.items():
                if existing.get(key) != value:
                    raise RuntimeError(f"V5.1 provenance mismatch: {key}")
        events_path = output / "events.jsonl"
        # SQLite is authoritative during resume.  Repair only a truncated
        # materialized view (the commit-before-append crash window), and fail
        # closed on extra, reordered, or modified JSONL records.
        _sync_event_jsonl(events_path, ledger)
        runtime_path = output / "runtime_metrics.jsonl"
        runtime = _load_runtime_metrics(runtime_path, arm=arm, seed=seed, dates=dates, last_day=last_day)
        active = prior_records[-1]
        candidate_epoch = 0
        costs = CostConfig()
        for day_index, date in enumerate(dates):
            if day_index <= last_day:
                if date in records:
                    active = records[date][1]; candidate_epoch += 1
                continue
            if stream.market is not None:
                from experiments.a_share.wc006_v4_market import apply_corporate_actions
                apply_corporate_actions(ledger, stream.market, date, stream.base.calendar, stage="start_of_day")
                apply_corporate_actions(base, stream.market, date, stream.base.calendar, stage="start_of_day")
            settled = _settle_v4(ledger, date, stream.base.calendar, stream.market.raw_prices if stream.market is not None else stream.prices, costs, stream.market)
            _settle_v4(base, date, stream.base.calendar, stream.market.raw_prices if stream.market is not None else stream.prices, costs, stream.market)
            if stream.market is not None:
                from experiments.a_share.wc006_v4_market import apply_corporate_actions
                apply_corporate_actions(ledger, stream.market, date, stream.base.calendar, stage="after_open_settlement")
                apply_corporate_actions(base, stream.market, date, stream.base.calendar, stage="after_open_settlement")
            resolved = _resolve_batches(ledger, date, D(ledger.get_meta("effective_multiplier") or "1.0"))
            effective = D(ledger.get_meta("effective_multiplier") or "1.0")
            is_candidate = date in records
            if is_candidate:
                active = records[date][1]; candidate_epoch += 1
            active_codes = {str(item["code"]) for item in active.targets}
            previous_equity = D(ledger.get_meta("previous_equity") or "100000")
            previous_base = D(base.get_meta("previous_equity") or "100000")
            if stream.market is not None:
                from experiments.a_share.wc006_v4_market import raw_equity
                equity, fly_marks = raw_equity(ledger, stream.market, date, active_codes); base_equity, base_marks = raw_equity(base, stream.market, date, active_codes)
            else:
                equity, base_equity = _equity(ledger, stream.prices, date, "close", active_codes), _equity(base, stream.prices, date, "close", active_codes); fly_marks = base_marks = {}
            drawdown = equity / max(previous_equity, D(ledger.get_meta("peak_equity") or previous_equity)) - D("1")
            feedback = incremental_feedback(equity=equity, previous_equity=previous_equity, base_equity=base_equity, previous_base_equity=previous_base, drawdown=drawdown, previous_drawdown=D(ledger.get_meta("drawdown") or "0"), turnover_ratio=settled / previous_equity if previous_equity else D("0"))
            reward_deliveries = _deliver_batch_rewards(ledger, resolved, feedback, fly, date)
            frame = _frame(active, stream.prices, date, stream.base.calendar, ledger, effective, drawdown)
            portfolio_exposure = _gross_exposure(ledger, stream, date, active_codes, equity)
            frame_log = _frame_observation_payload(
                active, stream.prices, date, stream.base.calendar, ledger, effective,
                drawdown, equity, base_equity, portfolio_exposure, frame,
            )
            event_id = _v51_event_id(ledger, date)
            stimulation = None
            if sensory_adapter is not None:
                sensory_observation = {
                    **frame_log,
                    "event_id": event_id,
                    "observation_schema": V51_SCHEMA,
                    "sensory_adapter": sensory_adapter_provenance,
                }
                stimulation = sensory_adapter.encode(sensory_observation)
                if stimulation is not None and not isinstance(stimulation, SensoryStimulation):
                    raise TypeError("sensory_adapter.encode must return SensoryStimulation or None")
                if stimulation is not None:
                    stimulation.validate_against_graph(fly.brain.ids)
            # None and adapters that encode to None deliberately retain the
            # exact V5.1 observe call (including omitted keyword arguments).
            stimulation_payload = None if stimulation is None else stimulation.to_dict()
            stimulation_sha256 = None if stimulation_payload is None else hashlib.sha256(json.dumps(stimulation_payload, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            if stimulation is None:
                neural = fly.observe(frame, "none")
            else:
                neural = fly.observe(frame, "none", sensory_stimulation=stimulation)
            compute_seconds = _runtime_compute_seconds(neural, fly)
            action, requested = map_decoder_to_multiplier(str(neural["side"]), effective)
            requested_target = sum((D(str(item["weight"])) * RISK_BUDGET for item in active.targets), D("0")) * requested
            effective_target = sum((D(str(item["weight"])) * RISK_BUDGET for item in active.targets), D("0")) * effective
            cp = output / f"brain-day-{day_index % 2}.npz"; fly.save(cp); cp_info = {"file": cp.name, "sha256": sha256(cp)}
            observation_id = ledger.record_observation(date=date, input_sha256=str(neural["input_sha256"]), reinforcement="none", neural=deterministic_neural_payload(neural), checkpoint=cp_info)
            batch = None; fly_plan: list[dict[str, Any]] = []
            boundary_settlement_only = _annual_boundary_settlement_only(
                annual_payload,
                settlement_date=settlement_date,
                date=date,
                end_date=end_date,
            )
            if is_candidate and not boundary_settlement_only:
                # Candidate composition changes are independent of decoder
                # HOLD: HOLD blocks only an extra exposure transition.
                plan_multiplier = requested if action != "DEFER" else effective
                fly_plan, intent_ids, empty_outcome = _prepare_target_intents(ledger, active, date=date, multiplier=plan_multiplier, neural_side=str(neural["side"]), observation_id=observation_id, checkpoint_sha256=cp_info["sha256"], stream=stream, target_codes=active_codes, intent_kind="EXPOSE" if plan_multiplier >= effective else "DEEXPOSE")
                execution_date = next((d for d in stream.base.calendar if d > date), "END")
                batch = _create_batch(ledger, kind="candidate_selection", request_date=date, execution_date=execution_date, candidate_epoch=candidate_epoch, requested=plan_multiplier, effective=effective, requested_target_exposure=requested_target if action != "DEFER" else effective_target, effective_target_exposure=effective_target, intent_ids=intent_ids, empty_outcome=empty_outcome)
            elif not boundary_settlement_only and action != "DEFER" and requested != effective:
                fly_plan, intent_ids, empty_outcome = _prepare_target_intents(ledger, active, date=date, multiplier=requested, neural_side=str(neural["side"]), observation_id=observation_id, checkpoint_sha256=cp_info["sha256"], stream=stream, target_codes=active_codes, intent_kind="EXPOSE" if requested >= effective else "DEEXPOSE")
                execution_date = next((d for d in stream.base.calendar if d > date), "END")
                batch = _create_batch(ledger, kind="exposure_rebalance", request_date=date, execution_date=execution_date, candidate_epoch=candidate_epoch, requested=requested, effective=effective, requested_target_exposure=requested_target, effective_target_exposure=effective_target, intent_ids=intent_ids, empty_outcome=empty_outcome)
            elif action == "DEFER":
                batch = {"batch_kind": "none", "outcome": "HOLD_NO_ORDER", "intent_ids": []}
            if is_candidate:
                # The paired deterministic base remains an independent arm.
                base_record = active
                _prepare_target_intents(base, base_record, date=date, multiplier=D("1.0"), neural_side="BUY", observation_id=observation_id, checkpoint_sha256="paired-base", stream=stream, target_codes=active_codes, intent_kind="EXPOSE")
            neural.update({"v51_action": action, "v51_requested_multiplier": str(requested), "v51_effective_multiplier": str(effective), "v51_candidate_day": is_candidate, "v51_candidate_epoch": candidate_epoch, "v51_seed": seed})
            neural_payload = deterministic_neural_payload(neural)
            requested_execution_date = next((d for d in stream.base.calendar if d > date), None)
            feedback_json = {key: str(value) if isinstance(value, Decimal) else value for key, value in feedback.items()}
            settled_fills = _fills_for_date(ledger, date)
            settled_vetoes = _vetoes_for_date(ledger, date)
            source_observation_ids = {int(item["observation_id"]) for item in settled_fills + settled_vetoes if item.get("observation_id") is not None}
            for resolved_batch in resolved:
                intent_ids = list(resolved_batch.get("intent_ids", []))
                if intent_ids:
                    marks = ",".join("?" for _ in intent_ids)
                    for row in ledger.db.execute(f"SELECT observation_id FROM intents WHERE id IN ({marks})", tuple(intent_ids)).fetchall():
                        source_observation_ids.add(int(row[0]))
            execution_payload = {
                "observation_id": observation_id,
                "source_observation_ids": sorted(source_observation_ids),
                "next_execution_request": {
                    "requested_multiplier": str(requested),
                    "effective_multiplier_before": str(effective),
                    "requested_execution_date": requested_execution_date,
                    "requested_target_exposure": str(requested_target),
                    "batch": batch,
                },
                "settled_prior_observation_outcome": {
                    "execution_date": str(date),
                    "fills": settled_fills,
                    "vetoes": settled_vetoes,
                    "guard_reasons": _guard_reasons(ledger, date),
                    "turnover": str(settled),
                    "fees": str(_fees_on_date(ledger, date)),
                    "portfolio_return": str(feedback.get("portfolio_return")),
                    "base_return": str(feedback.get("base_return")),
                    "reward_score": str(feedback.get("score")),
                    "reinforcement_classification": str(feedback.get("stimulus", "none")),
                    "resolved_batches": resolved,
                    "reward_delivery": reward_deliveries,
                },
            }
            event = {
                "schema": V51_SCHEMA, "event_id": event_id, "date": date,
                "arm": V51_PROTOCOL_ARM, "signal_eligible": is_candidate,
                "candidate_selection": {"updated": is_candidate, "epoch": candidate_epoch, "codes": sorted(active_codes)},
                "decoder": {"side": str(neural["side"]), "action": action, "requested_multiplier": str(requested), "effective_multiplier": str(effective)},
                "requested_target": {"multiplier": str(requested), "exposure": str(requested_target)},
                "effective_target": {"multiplier": str(effective), "exposure": str(effective_target)},
                "batch": batch, "paired_base": {"candidate_day": is_candidate}, "resolved_batches": resolved,
                "intent_statuses": {str(item.get("intent_id")): "PREPARED" for item in fly_plan if item.get("intent_id")},
                "guard_reason": _guard_reasons(ledger, date), "reward": feedback_json,
                "settled_signal_ids": [], "reward_delivery": reward_deliveries,
                "fees": str(_fees_on_date(ledger, date)), "turnover": str(settled),
                "actual_exposure": str(portfolio_exposure), "neural": neural_payload,
                "neural_output": {**neural_payload, "decoder_action": action}, "execution": execution_payload,
                "observation": {"observation_id": observation_id, "event_id": event_id, **frame_log, "sensory": {"adapter": sensory_adapter_provenance, "stimulation": stimulation_payload, "stimulation_sha256": stimulation_sha256}},
                "forward_outcome": {"status": "pending", "file": "forward_outcomes.jsonl", "event_id": event_id},
                "checkpoint": cp_info, "mark_provenance": {"fly": fly_marks, "base": base_marks},
            }
            event_hash, created = ledger.record_event(date=date, payload=event); event["event_hash"] = event_hash
            if created:
                with events_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            base.record_event(date=date, payload={"schema": V51_SCHEMA, "date": date, "candidate_day": is_candidate})
            ledger.complete_date(date=date, index=day_index, multiplier=effective, equity=equity, drawdown=drawdown, checkpoint=cp_info); base.complete_date(date=date, index=day_index, multiplier=D("1.0"), equity=base_equity, drawdown=D("0"), checkpoint={"paired": True})
            ledger.db.commit(); base.db.commit(); _append_runtime_metric(runtime_path, arm=arm, seed=seed, date=date, index=day_index, compute_seconds=compute_seconds); runtime.append({"arm": arm, "seed": seed, "date": date, "index": day_index, "compute_seconds": compute_seconds})
            if max_days and day_index + 1 >= max_days:
                break
        training_complete = ledger.get_meta("last_completed_index") is not None and int(ledger.get_meta("last_completed_index")) == len(dates) - 1
        if training_complete and settlement_date:
            recorded_final_date = ledger.get_meta("final_settlement_date")
            if recorded_final_date not in (None, settlement_date):
                raise RuntimeError("V5.1 final settlement date mismatch")
            if recorded_final_date is not None and ledger.get_meta("final_settlement") is None:
                raise RuntimeError("V5.1 final settlement audit record is missing")
        if training_complete and settlement_date and ledger.get_meta("final_settlement_date") is None:
            prices = stream.market.raw_prices if stream.market is not None else stream.prices
            settled = _settle_v4(ledger, settlement_date, stream.base.calendar, prices, costs, stream.market)
            _settle_v4(base, settlement_date, stream.base.calendar, prices, costs, stream.market)
            final_resolved = _resolve_batches(ledger, settlement_date, D(ledger.get_meta("effective_multiplier") or "1.0"))
            # The preregistered boundary creates no observation or learning
            # pulse.  Still compute the same auditable feedback and close any
            # resolved linkage with explicit settlement-only semantics.
            if stream.market is not None:
                from experiments.a_share.wc006_v4_market import raw_equity
                final_equity, final_marks = raw_equity(ledger, stream.market, settlement_date, set())
                final_base_equity, final_base_marks = raw_equity(base, stream.market, settlement_date, set())
            else:
                final_equity = _equity(ledger, stream.prices, settlement_date, "close", set())
                final_base_equity = _equity(base, stream.prices, settlement_date, "close", set())
                final_marks = final_base_marks = {}
            previous_equity = D(ledger.get_meta("previous_equity") or "100000")
            previous_base = D(base.get_meta("previous_equity") or "100000")
            peak = max(previous_equity, D(ledger.get_meta("peak_equity") or previous_equity), final_equity)
            final_drawdown = final_equity / peak - D("1") if peak else D("0")
            final_feedback = incremental_feedback(
                equity=final_equity,
                previous_equity=previous_equity,
                base_equity=final_base_equity,
                previous_base_equity=previous_base,
                drawdown=final_drawdown,
                previous_drawdown=D(ledger.get_meta("drawdown") or "0"),
                turnover_ratio=settled / previous_equity if previous_equity else D("0"),
            )
            final_rewards = _record_settlement_only_rewards(ledger, final_resolved, final_feedback, settlement_date)
            final_feedback_json = {
                key: str(value) if isinstance(value, Decimal) else value
                for key, value in final_feedback.items()
            }
            final_settlement = {
                "date": settlement_date,
                "semantic": "settlement-only; no boundary observation, signal, or learning pulse",
                "effective_multiplier": ledger.get_meta("effective_multiplier") or "1.0",
                "resolved": final_resolved,
                "reward_delivery": final_rewards,
                "feedback": final_feedback_json,
                "equity": str(final_equity),
                "base_equity": str(final_base_equity),
                "turnover": str(settled),
                "cost": str(_fees_on_date(ledger, settlement_date)),
                "mark_provenance": {"fly": final_marks, "base": final_base_marks},
                "risk_budget": str(RISK_BUDGET),
            }
            ledger.put_meta("final_settlement_date", settlement_date); base.put_meta("final_settlement_date", settlement_date)
            ledger.put_meta("final_settlement", final_settlement); ledger.db.commit(); base.db.commit()
        # Outcome materializers are post-observation audit views. They never
        # feed back into neural propagation, plasticity, reward, or execution.
        _write_observation_outcomes(output, events_path)
        _write_forward_outcomes(output, events_path, stream.prices, stream.base.calendar)
        complete = (
            training_complete
            and not ledger.pending()
            and not base.pending()
            and not ledger.unresolved()
            and not base.unresolved()
            and not _v51_batch_completion_issues(ledger)
        )
        return {"status": "complete" if complete else "paused", "schema": V51_SCHEMA, "arm": V51_PROTOCOL_ARM, "protocol_arm": V51_PROTOCOL_ARM, "seed": seed, "plasticity_mode": plasticity_mode, "days": len(runtime), "calendar_days": len(dates), "signal_records": len(records), "scope": [start_date, end_date], "settlement_date": settlement_date, "output_scope": scope}
    finally:
        ledger.close(); base.close()


def run_v51_annual(
    annual_manifest: Path,
    source_manifest: Path,
    market_manifest: Path | None,
    output: Path,
    *,
    max_days: int = 0,
    controller_factory: Callable[[Settings], Any] = V51Controller,
    arm: str = "combined",
    seed: int = 17,
    plasticity_mode: str = "enabled",
    sensory_adapter: SensoryAdapter | None = None,
) -> dict[str, Any]:
    """Run the development-only 2023 annual slice against pinned inputs.

    This entry point is deliberately separate from both fixture and formal
    APIs.  It accepts the draft annual slice, its full candidate source, and
    the corrected paper market bundle; no market bundle means immediate gate
    failure.
    """
    if market_manifest is None:
        raise RuntimeError("annual input gate: corrected market manifest is required")
    annual_manifest = Path(annual_manifest).resolve()
    annual = _validate_annual_manifest(annual_manifest, Path(source_manifest))
    payload = annual["payload"]
    market_path, market_sha, market_qa_sha, supplemental = _resolve_annual_market_binding(
        annual_manifest, payload, Path(market_manifest)
    )
    annual_evidence = {
        "schema": payload["schema"],
        "manifest_sha256": annual["sha256"],
        "source_manifest_sha256": annual["source_sha256"],
        "source_manifest_path": str(annual["source_path"]),
        "year": payload["year"],
        "observation_window": list(payload["observation_window"]),
        "observation_count": payload["observation_count"],
        "candidate_record_count": payload["candidate_record_count"],
        "settlement_boundary": payload["settlement_boundary"],
        "market_manifest_sha256": market_sha,
        "market_qa_sha256": market_qa_sha,
    }
    if supplemental is not None:
        annual_evidence["supplemental_audit_sha256"] = supplemental["sha256"]
    return _run_v51(
        Path(source_manifest),
        Path(output),
        start_date=str(payload["observation_window"][0]),
        end_date=str(payload["observation_window"][1]),
        formal=False,
        market_manifest=market_path,
        max_days=max_days,
        expected_records=int(payload["source_manifest_records"]),
        training_end=str(payload["observation_window"][1]),
        controller_factory=controller_factory,
        arm=arm,
        seed=seed,
        plasticity_mode=plasticity_mode,
        sensory_adapter=sensory_adapter,
        annual_input_gate=annual_evidence,
        annual_payload=payload,
        market_expected_parent_sha256=market_sha,
        market_expected_qa_sha256=market_qa_sha,
    )


# Descriptive alias retained for callers that name the bridge explicitly.
run_v51_annual_slice = run_v51_annual

def run_v51_fixture(stream_path: Path, output: Path, *, start_date: str, end_date: str, max_days: int = 0, expected_records: int = 3, training_end: str | None = None, controller_factory: Callable[[Settings], Any] = V51Controller, arm: str = "combined", seed: int = 0, plasticity_mode: str = "enabled", sensory_adapter: SensoryAdapter | None = None) -> dict[str, Any]:
    end = training_end or end_date
    return _run_v51(stream_path, output, start_date=start_date, end_date=end_date, formal=False, market_manifest=None, max_days=max_days, expected_records=expected_records, training_end=end, controller_factory=controller_factory, arm=arm, seed=seed, plasticity_mode=plasticity_mode, sensory_adapter=sensory_adapter)


def run_v51_formal(stream_path: Path, output: Path, market_manifest: Path, *, start_date: str, end_date: str = TRAIN_END, max_days: int = 0, controller_factory: Callable[[Settings], Any] = V51Controller, arm: str = "combined", seed: int = 17, plasticity_mode: str = "enabled", sensory_adapter: SensoryAdapter | None = None) -> dict[str, Any]:
    return _run_v51(stream_path, output, start_date=start_date, end_date=end_date, formal=True, market_manifest=market_manifest, max_days=max_days, expected_records=46, training_end=TRAIN_END, controller_factory=controller_factory, arm=arm, seed=seed, plasticity_mode=plasticity_mode, sensory_adapter=sensory_adapter)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="candidate source manifest (or fixture stream)")
    parser.add_argument("--annual-manifest", type=Path, help="development-only annual V5.1 slice manifest")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date", default=TRAIN_END)
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument("--market-manifest", type=Path)
    parser.add_argument("--plasticity-mode", choices=V51_PLASTICITY_MODES, default="enabled")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--arm", choices=V51_ARMS, default="combined")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.annual_manifest is not None:
        if args.formal:
            parser.error("--annual-manifest is development-only and cannot be combined with --formal")
        if args.market_manifest is None:
            parser.error("--annual-manifest requires --market-manifest")
        result = run_v51_annual(args.annual_manifest, args.manifest, args.market_manifest, args.out, max_days=args.max_days, arm=args.arm, seed=args.seed, plasticity_mode=args.plasticity_mode)
    elif args.formal:
        if args.market_manifest is None:
            parser.error("--formal requires --market-manifest")
        if args.start_date is None:
            parser.error("--formal requires --start-date")
        result = run_v51_formal(args.manifest, args.out, args.market_manifest, start_date=args.start_date, end_date=args.end_date, max_days=args.max_days, arm=args.arm, seed=args.seed, plasticity_mode=args.plasticity_mode)
    else:
        if args.start_date is None:
            parser.error("fixture mode requires --start-date")
        result = run_v51_fixture(args.manifest, args.out, start_date=args.start_date, end_date=args.end_date, max_days=args.max_days, arm=args.arm, seed=args.seed, plasticity_mode=args.plasticity_mode)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
