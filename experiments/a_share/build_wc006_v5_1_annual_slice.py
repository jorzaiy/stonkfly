#!/usr/bin/env python3
"""Build the frozen 2023 WC006 V5.1 input-slice manifest.

This is a read-only input adapter.  It validates the existing 2021--2023
candidate-stream manifest and copies only its causal inputs into a new draft
directory.  It does not create an authorization artifact, run a replay, or
modify the frozen V5 adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any


class AnnualSliceBlocked(ValueError):
    """Raised when the source stream cannot be sliced safely."""


YEAR = 2023
START_DATE = "2023-01-03"
END_DATE = "2023-12-29"
SETTLEMENT_BOUNDARY = "2024-01-02"
EXPECTED_OBSERVATIONS = 242
EXPECTED_RECORDS = 17
EXPECTED_SOURCE_RECORDS = 46
EXPECTED_SOURCE_MANIFEST_SHA256 = "5874b6cefab5f1c4b1ee04db819fd76430cdf67495db67db5242f5d83f42cb52"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnnualSliceBlocked(f"cannot read JSON source: {path}") from exc
    if not isinstance(payload, dict):
        raise AnnualSliceBlocked(f"JSON object required: {path}")
    return payload


def _date(value: object) -> str:
    raw = str(value).strip().replace("/", "-")
    if len(raw) == 8 and raw.isdigit():
        raw = f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    if len(raw) != 10 or raw[4] != "-" or raw[7] != "-":
        raise AnnualSliceBlocked(f"invalid ISO date: {value!r}")
    return raw


def _source_file(source_root: Path, copied_path: object) -> Path:
    rel = Path(str(copied_path))
    if rel.is_absolute() or ".." in rel.parts:
        raise AnnualSliceBlocked(f"source path escapes manifest root: {copied_path!r}")
    path = (source_root / rel).resolve()
    try:
        path.relative_to(source_root.resolve())
    except ValueError as exc:
        raise AnnualSliceBlocked(f"source path escapes manifest root: {copied_path!r}") from exc
    return path


def _copy_checked(source: Path, destination: Path, expected: str) -> str:
    if not source.is_file():
        raise AnnualSliceBlocked(f"missing source input: {source}")
    actual = sha256(source)
    if actual != expected:
        raise AnnualSliceBlocked(f"source hash mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    copied = sha256(destination)
    if copied != expected:
        raise AnnualSliceBlocked(f"copied source hash mismatch: {destination}")
    return copied


def _validate_calendar(source_root: Path, manifest: dict[str, Any]) -> tuple[list[str], Path]:
    source_hashes = manifest.get("source_hashes")
    if not isinstance(source_hashes, dict) or not isinstance(source_hashes.get("calendar"), str):
        raise AnnualSliceBlocked("source manifest calendar hash is missing")
    calendar_path = _source_file(source_root, "inputs/calendar.json")
    if sha256(calendar_path) != source_hashes["calendar"]:
        raise AnnualSliceBlocked("source calendar hash does not match source manifest")
    payload = _json(calendar_path)
    raw_dates = payload.get("dates")
    if not isinstance(raw_dates, list):
        raise AnnualSliceBlocked("source calendar dates are missing")
    dates = [_date(item) for item in raw_dates]
    if dates != sorted(set(dates)):
        raise AnnualSliceBlocked("source calendar is not strictly chronological")
    annual = [item for item in dates if START_DATE <= item <= END_DATE]
    if len(annual) != EXPECTED_OBSERVATIONS or annual[0] != START_DATE or annual[-1] != END_DATE:
        raise AnnualSliceBlocked("2023 calendar does not contain the pinned 242 observations")
    if SETTLEMENT_BOUNDARY not in dates:
        raise AnnualSliceBlocked("2024-01-02 settlement boundary is missing")
    if any(item > END_DATE and item != SETTLEMENT_BOUNDARY for item in dates if item <= SETTLEMENT_BOUNDARY):
        raise AnnualSliceBlocked("unexpected post-year date before settlement boundary")
    return annual + [SETTLEMENT_BOUNDARY], calendar_path


def _validate_records(source_root: Path, manifest: dict[str, Any], calendar: list[str]) -> list[dict[str, Any]]:
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_SOURCE_RECORDS:
        raise AnnualSliceBlocked(f"source record count must be {EXPECTED_SOURCE_RECORDS}")
    allowed = set(calendar)
    selected: list[dict[str, Any]] = []
    previous: str | None = None
    for record in records:
        if not isinstance(record, dict):
            raise AnnualSliceBlocked("source record is not an object")
        signal = _date(record.get("signal_date"))
        execution = _date(record.get("execution_date"))
        if previous is not None and signal <= previous:
            raise AnnualSliceBlocked("source records are not strictly chronological")
        previous = signal
        if signal < START_DATE or signal > END_DATE:
            continue
        if signal not in allowed or execution not in allowed:
            raise AnnualSliceBlocked(f"record date is outside annual calendar: {signal}")
        if execution == SETTLEMENT_BOUNDARY and signal != END_DATE:
            raise AnnualSliceBlocked("only the final 2023 signal may use 2024-01-02 settlement")
        if execution != SETTLEMENT_BOUNDARY and execution <= signal:
            raise AnnualSliceBlocked(f"record execution is not next-session causal: {signal}")
        selected.append(record)
    if len(selected) != EXPECTED_RECORDS:
        raise AnnualSliceBlocked(f"annual record count must be {EXPECTED_RECORDS}")
    if selected[-1]["signal_date"] != END_DATE or selected[-1]["execution_date"] != SETTLEMENT_BOUNDARY:
        raise AnnualSliceBlocked("final annual record lacks the exact settlement boundary")
    return selected


def build_annual_slice(*, source_manifest: Path, output_dir: Path) -> Path:
    """Create a standalone, hash-pinned 2023 input manifest draft."""
    source_manifest = Path(source_manifest).resolve()
    source_root = source_manifest.parent
    if sha256(source_manifest) != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise AnnualSliceBlocked("source manifest hash is not the pinned V5 manifest")
    manifest = _json(source_manifest)
    if manifest.get("schema") != "wc006-candidate-stream/v1":
        raise AnnualSliceBlocked("unsupported source manifest schema")
    calendar, calendar_path = _validate_calendar(source_root, manifest)
    selected = _validate_records(source_root, manifest, calendar)
    if Path(output_dir).exists():
        raise AnnualSliceBlocked("output directory must be new and unique")
    output = Path(output_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent)))
    try:
        source_hashes = manifest.get("source_hashes")
        cache_files = manifest.get("cache_files")
        selector_artifacts = manifest.get("selector_artifacts")
        if not isinstance(cache_files, dict) or not isinstance(selector_artifacts, list):
            raise AnnualSliceBlocked("source manifest input hashes are incomplete")
        selected_codes = {
            str(row["code"])
            for record in selected
            for row in record["selector"]["target_topk"]
        }
        copied_cache: dict[str, dict[str, str]] = {}
        for code in sorted(selected_codes):
            item = cache_files.get(code)
            if not isinstance(item, dict) or not isinstance(item.get("sha256"), str):
                raise AnnualSliceBlocked(f"missing cache hash for selected code: {code}")
            source = _source_file(source_root, item.get("copied_path"))
            destination = staging / "inputs" / "cache" / source.name
            copied_cache[code] = {
                "source_name": source.name,
                "sha256": _copy_checked(source, destination, item["sha256"]),
                "copied_path": str(destination.relative_to(staging)),
            }
        selected_hashes = {str(record["selector"]["artifact_sha256"]) for record in selected}
        copied_artifacts: list[dict[str, str]] = []
        for index, item in enumerate(selector_artifacts):
            if not isinstance(item, dict) or item.get("sha256") not in selected_hashes:
                continue
            source = _source_file(source_root, item.get("copied_path"))
            destination = staging / "inputs" / "selector" / f"{index:03d}_{source.name}"
            copied_artifacts.append({
                "source_name": source.name,
                "sha256": _copy_checked(source, destination, str(item["sha256"])),
                "copied_path": str(destination.relative_to(staging)),
            })
        if len(copied_artifacts) != EXPECTED_RECORDS:
            raise AnnualSliceBlocked("not all annual selector artifacts were found")
        for name, hash_key in (("active_config.json", "active_config"), ("universe.txt", "universe")):
            expected = source_hashes.get(hash_key) if isinstance(source_hashes, dict) else None
            if not isinstance(expected, str):
                raise AnnualSliceBlocked(f"missing source hash for {name}")
            source = _source_file(source_root, f"inputs/{name}")
            _copy_checked(source, staging / "inputs" / name, expected)
        calendar_payload = {"dates": calendar, "source_calendar_sha256": source_hashes["calendar"]}
        (staging / "inputs" / "calendar.json").parent.mkdir(parents=True, exist_ok=True)
        (staging / "inputs" / "calendar.json").write_text(json.dumps(calendar_payload, indent=2) + "\n", encoding="utf-8")
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
            "records": selected,
            "source_hashes": {
                "active_config": sha256(staging / "inputs/active_config.json"),
                "universe": sha256(staging / "inputs/universe.txt"),
                "calendar": sha256(staging / "inputs/calendar.json"),
                "source_calendar": source_hashes["calendar"],
            },
            "cache_files": copied_cache,
            "selector_artifacts": copied_artifacts,
            "limitations": [
                "draft input slice only; no authorization or replay is created",
                "2024-01-02 is a settlement boundary and never an observation",
                "source manifest and all copied inputs are hash-pinned",
            ],
        }
        (staging / "manifest.json").write_text(json.dumps(annual_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / "manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(build_annual_slice(source_manifest=args.source_manifest, output_dir=args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
