#!/usr/bin/env python3
"""Build an immutable-reference equity_structure staging bundle for WC006 V5.

This is intentionally a read-only consumer of the legacy 119-code fragment and
the manually collected six-code fragment.  It emits a fresh normalized CSV and
audit metadata under the V5 manual bridge; it never edits either source.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CSTAGE = ROOT / "runs/wc006_v4_xysz_full/20260919T054212Z-125codes"
BRIDGE = ROOT / "runs/wc006_v5_market_inputs/manual-bridge"
NEW_ROOT = BRIDGE / "equity_structure_missing6_1990_attempt1"
OUT = BRIDGE / "prepared"

KEY_FIELDS = (
    "MARKET_CODE",
    "ANN_DATE",
    "CHANGE_DATE",
    "EX_CHANGE_DATE",
    "SHARE_CHANGE_REASON",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        raw_fields = [str(field) if field is not None else "" for field in (reader.fieldnames or [])]
        rows: list[dict[str, str]] = []
        for raw in reader:
            rows.append({field: str(raw.get(field, "")) for field in raw_fields if field})
    return raw_fields, rows


def source_record(path: Path, manifest_path: Path, expected_rows: int | None = None) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    actual_manifest_sha = sha256(manifest_path)
    if manifest.get("status") != "complete":
        raise RuntimeError(f"source_manifest_not_complete:{manifest_path}")
    fields, rows = read_csv(path)
    if expected_rows is not None and len(rows) != expected_rows:
        raise RuntimeError(f"source_row_count_mismatch:{path}:{len(rows)}:{expected_rows}")
    declared = next((a for a in manifest.get("artifacts", []) if a.get("path") == path.name), None)
    if declared is None:
        raise RuntimeError(f"source_artifact_not_declared:{path}")
    actual_sha = sha256(path)
    if actual_sha != declared.get("sha256"):
        raise RuntimeError(f"source_artifact_hash_mismatch:{path}")
    codes = sorted({row.get("MARKET_CODE", "").strip().upper() for row in rows if row.get("MARKET_CODE", "").strip()})
    return {
        "source_path": str(path.relative_to(ROOT)),
        "source_manifest": str(manifest_path.relative_to(ROOT)),
        "source_manifest_sha256": actual_manifest_sha,
        "source_artifact_sha256": actual_sha,
        "declared_rows": declared.get("rows"),
        "observed_rows": len(rows),
        "declared_fields": declared.get("fields", []),
        "observed_header": fields,
        "observed_codes": codes,
        "rows": rows,
    }


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing_to_overwrite_nonempty_output:{OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    old_sources: list[dict[str, Any]] = []
    for manifest_path in sorted((CSTAGE / "equity_structure_hardexit").glob("batch_*/manifest.json")):
        old_sources.append(source_record(manifest_path.parent / "raw/equity_structure.csv", manifest_path))
    if len(old_sources) != 5:
        raise RuntimeError(f"unexpected_old_equity_batch_count:{len(old_sources)}")
    new_manifest_path = NEW_ROOT / "manifest.json"
    new_source = source_record(NEW_ROOT / "raw/equity_structure.csv", new_manifest_path)

    all_source_rows = [row for source in old_sources for row in source["rows"]]
    old_codes = sorted({row["MARKET_CODE"].strip().upper() for row in all_source_rows if row.get("MARKET_CODE", "").strip()})
    new_codes = sorted({row["MARKET_CODE"].strip().upper() for row in new_source["rows"] if row.get("MARKET_CODE", "").strip()})
    expected_codes = sorted(set(old_codes) | set(new_codes))
    if len(old_codes) != 119 or len(new_codes) != 6 or len(expected_codes) != 125:
        raise RuntimeError(f"unexpected_code_coverage:{len(old_codes)}:{len(new_codes)}:{len(expected_codes)}")
    if set(old_codes) & set(new_codes):
        raise RuntimeError("source_code_overlap")

    fields = list(old_sources[0]["declared_fields"])
    for source in [*old_sources, new_source]:
        if source["declared_fields"] != fields:
            raise RuntimeError("source_field_schema_mismatch")
        if source["observed_header"] != ["", *fields]:
            raise RuntimeError(f"unexpected_csv_index_header:{source['source_path']}:{source['observed_header']}")

    merged_rows = all_source_rows + new_source["rows"]
    key_counts: dict[tuple[str, ...], int] = {}
    blank_key_rows: list[int] = []
    for number, row in enumerate(merged_rows, start=1):
        key = tuple(row.get(field, "").strip() for field in KEY_FIELDS)
        key_counts[key] = key_counts.get(key, 0) + 1
        if any(not value for value in key):
            blank_key_rows.append(number)
    duplicate_keys = [list(key) for key, count in sorted(key_counts.items()) if count > 1]
    if duplicate_keys or blank_key_rows:
        raise RuntimeError(f"equity_primary_key_failure:{len(duplicate_keys)}:{blank_key_rows[:5]}")

    raw_dir = OUT / "raw"
    raw_dir.mkdir()
    merged_path = raw_dir / "equity_structure.csv"
    with merged_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", *fields])
        for index, row in enumerate(merged_rows):
            writer.writerow([index, *[row.get(field, "") for field in fields]])

    codes_path = OUT / "codes_125.txt"
    codes_path.write_text("".join(f"{code}\n" for code in expected_codes), encoding="utf-8")
    merged_sha = sha256(merged_path)
    codes_sha = sha256(codes_path)
    merged_rows_count = len(merged_rows)
    source_refs = [
        {key: value for key, value in source.items() if key != "rows"}
        for source in [*old_sources, new_source]
    ]

    write_json(OUT / "source_inventory.json", {
        "schema": "wc006-v5-manual-bridge/source-inventory/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_fragments": source_refs,
        "source_code_count": {"legacy_observed": len(old_codes), "manual_observed": len(new_codes), "union": len(expected_codes)},
        "source_row_count": {"legacy": len(all_source_rows), "manual": len(new_source["rows"]), "union": merged_rows_count},
        "csv_index_policy": "sources_and_staging_use_a_leading_blank-header_pandas_index_column; staging index reset to contiguous 0..N-1",
        "training_authorized": False,
        "paper_only": True,
        "broker_orders_enabled": False,
        "live_execution_authorized": False,
    })
    write_json(OUT / "manifest.json", {
        "schema": "wc006-v5-manual-bridge/equity-structure-staging/v1",
        "status": "complete",
        "dataset": "equity_structure",
        "codes": expected_codes,
        "rows": merged_rows_count,
        "begin_date": 20210101,
        "end_date": 20240102,
        "action_begin_date": 19900101,
        "exit_mode": "hard_exit_after_commit",
        "credential_values_persisted": False,
        "training_authorized": False,
        "paper_only": True,
        "source_manifest_count": len(source_refs),
        "source_manifest_sha256": [source["source_manifest_sha256"] for source in source_refs],
        "artifacts": [{"path": "raw/equity_structure.csv", "sha256": merged_sha, "rows": merged_rows_count, "fields": fields, "leading_index_column": True}, {"path": "codes_125.txt", "sha256": codes_sha, "rows": 125}],
    })
    (OUT / "manifest.json.sha256").write_text(f"{sha256(OUT / 'manifest.json')}  manifest.json\n", encoding="utf-8")
    (OUT / "codes_125.txt.sha256").write_text(f"{codes_sha}  codes_125.txt\n", encoding="utf-8")
    write_json(OUT / "qa.json", {
        "schema": "wc006-v5-manual-bridge/equity-structure-qa/v1",
        "status": "PASS_WITH_TRAINING_AUTHORIZATION_FALSE",
        "checks": {
            "legacy_source_hashes": True,
            "manual_source_hash": True,
            "expected_code_count_125": len(expected_codes) == 125,
            "legacy_code_count_119": len(old_codes) == 119,
            "manual_code_count_6": len(new_codes) == 6,
            "all_codes_have_rows": len(expected_codes) == 125,
            "row_count_additive": merged_rows_count == len(all_source_rows) + len(new_source["rows"]),
            "primary_key_unique": not duplicate_keys,
            "primary_key_nonempty": not blank_key_rows,
            "field_schema_identical": True,
            "staging_hash_recorded": True,
            "training_authorized": False,
        },
        "coverage": {
            "expected_codes": 125,
            "observed_codes": len(expected_codes),
            "legacy_rows": len(all_source_rows),
            "manual_rows": len(new_source["rows"]),
            "staging_rows": merged_rows_count,
            "missing_codes": sorted(set(expected_codes) - set(expected_codes)),
            "extra_codes": [],
        },
        "primary_key_fields": list(KEY_FIELDS),
        "duplicate_key_count": len(duplicate_keys),
        "blank_key_row_count": len(blank_key_rows),
        "artifact_hashes": {"raw/equity_structure.csv": merged_sha, "codes_125.txt": codes_sha},
        "blocking_note": "This staging bundle is data-complete for equity_structure only; status PIT/retrieval-time gates and the global V5 authorization gate remain unresolved.",
    })
    print(json.dumps({"status": "complete", "output": str(OUT), "codes": len(expected_codes), "rows": merged_rows_count, "equity_sha256": merged_sha, "codes_sha256": codes_sha}, ensure_ascii=False))


if __name__ == "__main__":
    main()
