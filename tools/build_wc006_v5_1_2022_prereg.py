#!/usr/bin/env python3
"""Build and audit the input-only 2022 V5.1 paired preregistration draft.

This tool is deliberately limited to causal input slicing and metadata audits.
It never imports the adapter, opens a ledger, runs a model, or reads a result
artifact.  It also does not calculate return, equity, P/L, fee, turnover, or
drawdown values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any


YEAR = 2022
START_DATE = "2022-01-04"
END_DATE = "2022-12-30"
SETTLEMENT_BOUNDARY = "2023-01-03"
EXPECTED_OBSERVATIONS = 242
EXPECTED_RECORDS = 19
EXPECTED_SOURCE_RECORDS = 46
EXPECTED_SOURCE_MANIFEST_SHA256 = "5874b6cefab5f1c4b1ee04db819fd76430cdf67495db67db5242f5d83f42cb52"
ADAPTER_PATH = "experiments/a_share/wc006_male_cns_v51_adapter.py"
MARKET_MANIFEST = "runs/wc006_v5_market_inputs/manual-bridge/authorized-v5-development-20260920T122237Z-controller-lock/manifest.json"
MARKET_QA = "runs/wc006_v5_market_inputs/manual-bridge/authorized-v5-development-20260920T122237Z-controller-lock/quality_report.json"


class AuditBlocked(ValueError):
    """Raised when the canonical inputs cannot support a safe draft."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditBlocked(f"cannot read JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise AuditBlocked(f"JSON object required: {path}")
    return payload


def iso_date(value: object) -> str:
    raw = str(value).strip().replace("/", "-")
    if len(raw) == 8 and raw.isdigit():
        raw = f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    try:
        date.fromisoformat(raw)
    except ValueError as exc:
        raise AuditBlocked(f"invalid ISO date: {value!r}") from exc
    return raw


def source_file(root: Path, relative: object) -> Path:
    rel = Path(str(relative))
    if rel.is_absolute() or ".." in rel.parts:
        raise AuditBlocked(f"source path escapes root: {relative!r}")
    result = (root / rel).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError as exc:
        raise AuditBlocked(f"source path escapes root: {relative!r}") from exc
    return result


def copy_checked(source: Path, destination: Path, expected: str) -> str:
    if not source.is_file():
        raise AuditBlocked(f"missing input: {source}")
    if sha256(source) != expected:
        raise AuditBlocked(f"source hash mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    actual = sha256(destination)
    if actual != expected:
        raise AuditBlocked(f"copied hash mismatch: {destination}")
    return actual


def selected_records(source: dict[str, Any], calendar: list[str]) -> list[dict[str, Any]]:
    records = source.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_SOURCE_RECORDS:
        raise AuditBlocked(f"source records must contain {EXPECTED_SOURCE_RECORDS} entries")
    allowed = set(calendar)
    result: list[dict[str, Any]] = []
    previous = ""
    for record in records:
        if not isinstance(record, dict):
            raise AuditBlocked("source record is not an object")
        signal = iso_date(record.get("signal_date"))
        execution = iso_date(record.get("execution_date"))
        if previous and signal <= previous:
            raise AuditBlocked("source signals are not strictly chronological")
        previous = signal
        if not (START_DATE <= signal <= END_DATE):
            continue
        if signal not in allowed or execution not in allowed:
            raise AuditBlocked(f"record date outside annual calendar: {signal}")
        if execution != SETTLEMENT_BOUNDARY and execution <= signal:
            raise AuditBlocked(f"execution is not next-session causal: {signal}")
        if execution == SETTLEMENT_BOUNDARY and signal != END_DATE:
            raise AuditBlocked("only final annual record may use settlement boundary")
        timing = record.get("timing_proof") or {}
        if timing.get("signal_lag_one_day") is not True or timing.get("execution_is_first_next_valid_session") is not True:
            raise AuditBlocked(f"timing proof missing: {signal}")
        result.append(record)
    if len(result) != EXPECTED_RECORDS:
        raise AuditBlocked(f"2022 record count must be {EXPECTED_RECORDS}")
    # 2022 final signal is 2022-12-29 and executes on final observed session 2022-12-30.
    # 2023-01-03 remains a declared post-year settlement boundary with no fabricated signal.
    if result[-1]["execution_date"] != END_DATE or result[-1]["signal_date"] >= END_DATE:
        raise AuditBlocked("final annual record must execute on the final observed session")
    return result


def build(*, repo: Path, source_manifest: Path, output: Path) -> Path:
    source_manifest = source_manifest.resolve()
    source_root = source_manifest.parent
    if sha256(source_manifest) != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise AuditBlocked("source manifest hash is not the pinned canonical hash")
    source = read_json(source_manifest)
    if source.get("schema") != "wc006-candidate-stream/v1":
        raise AuditBlocked("unsupported source manifest schema")
    source_hashes = source.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise AuditBlocked("source hashes are missing")
    calendar_path = source_file(source_root, "inputs/calendar.json")
    if sha256(calendar_path) != source_hashes.get("calendar"):
        raise AuditBlocked("canonical calendar hash mismatch")
    calendar_payload = read_json(calendar_path)
    raw_dates = calendar_payload.get("dates")
    if not isinstance(raw_dates, list):
        raise AuditBlocked("canonical calendar dates are missing")
    dates = [iso_date(value) for value in raw_dates]
    if dates != sorted(set(dates)):
        raise AuditBlocked("canonical calendar is not strictly chronological")
    annual = [value for value in dates if START_DATE <= value <= END_DATE]
    if len(annual) != EXPECTED_OBSERVATIONS or annual[0] != START_DATE or annual[-1] != END_DATE:
        raise AuditBlocked("2022 calendar does not contain the pinned 242 observations")
    if SETTLEMENT_BOUNDARY not in dates:
        raise AuditBlocked("2023-01-03 settlement boundary is missing")
    scoped_calendar = annual + [SETTLEMENT_BOUNDARY]
    records = selected_records(source, scoped_calendar)
    cache_files = source.get("cache_files")
    selector_artifacts = source.get("selector_artifacts")
    if not isinstance(cache_files, dict) or not isinstance(selector_artifacts, list):
        raise AuditBlocked("source input hash sections are incomplete")
    selected_codes = sorted({str(item["code"]) for record in records for item in record["selector"]["target_topk"]})
    if len(selected_codes) != 55:
        raise AuditBlocked(f"unexpected selected code count: {len(selected_codes)}")
    if output.exists():
        raise AuditBlocked(f"output must be new: {output}")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent)))
    try:
        copied_cache: dict[str, dict[str, str]] = {}
        for code in selected_codes:
            item = cache_files.get(code)
            if not isinstance(item, dict) or not isinstance(item.get("sha256"), str):
                raise AuditBlocked(f"missing cache hash: {code}")
            source_path = source_file(source_root, item.get("copied_path"))
            destination = staging / "inputs" / "cache" / source_path.name
            copied_cache[code] = {
                "source_name": source_path.name,
                "sha256": copy_checked(source_path, destination, item["sha256"]),
                "copied_path": str(destination.relative_to(staging)),
            }
        selected_hashes = {str(record["selector"]["artifact_sha256"]) for record in records}
        copied_artifacts: list[dict[str, str]] = []
        for index, item in enumerate(selector_artifacts):
            if not isinstance(item, dict) or item.get("sha256") not in selected_hashes:
                continue
            source_path = source_file(source_root, item.get("copied_path"))
            destination = staging / "inputs" / "selector" / f"{index:03d}_{source_path.name}"
            copied_artifacts.append({
                "source_name": source_path.name,
                "sha256": copy_checked(source_path, destination, str(item["sha256"])),
                "copied_path": str(destination.relative_to(staging)),
            })
        if len(copied_artifacts) != EXPECTED_RECORDS:
            raise AuditBlocked("not all annual selector artifacts were copied")
        for name, key in (("active_config.json", "active_config"), ("universe.txt", "universe")):
            expected = source_hashes.get(key)
            if not isinstance(expected, str):
                raise AuditBlocked(f"missing source hash: {key}")
            copy_checked(source_file(source_root, f"inputs/{name}"), staging / "inputs" / name, expected)
        calendar_copy = {"dates": scoped_calendar, "source_calendar_sha256": source_hashes["calendar"]}
        (staging / "inputs" / "calendar.json").write_text(json.dumps(calendar_copy, indent=2) + "\n", encoding="utf-8")
        annual_manifest = {
            "schema": "wc006-v5.1-annual-input-slice/v1",
            "draft": True,
            "authorization_artifact": False,
            "paper_only": True,
            "broker_orders_enabled": False,
            "live_execution_authorized": False,
            "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
            "source_manifest_records": EXPECTED_SOURCE_RECORDS,
            "year": YEAR,
            "observation_window": [START_DATE, END_DATE],
            "observation_count": EXPECTED_OBSERVATIONS,
            "settlement_boundary": SETTLEMENT_BOUNDARY,
            "candidate_record_count": EXPECTED_RECORDS,
            "candidate_selection_frozen": True,
            "records": records,
            "source_hashes": {
                "active_config": sha256(staging / "inputs/active_config.json"),
                "universe": sha256(staging / "inputs/universe.txt"),
                "calendar": sha256(staging / "inputs/calendar.json"),
                "source_calendar": source_hashes["calendar"],
            },
            "cache_files": copied_cache,
            "selector_artifacts": copied_artifacts,
            "limitations": [
                "input-only draft; no authorization artifact and no replay created",
                "2023-01-03 is a settlement-only boundary and never an observation or signal",
                "source manifest and copied causal inputs are hash-pinned",
                "current V5.1 adapter is still hard-coded to the 2023 annual gate; a separate adapter update and re-hash is required before execution",
            ],
        }
        manifest_path = staging / "annual-manifest.json"
        manifest_path.write_text(json.dumps(annual_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # Keep a conventional name as well; the plan points to this stable name.
        (staging / "manifest.json").write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / "manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(build(repo=args.repo.resolve(), source_manifest=args.source_manifest, output=args.output.resolve()))
    except AuditBlocked as exc:
        print(f"AUDIT_BLOCKED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
