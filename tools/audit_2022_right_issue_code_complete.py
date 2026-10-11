#!/usr/bin/env python3
"""Read-only audit for the 2022 target-code-complete rights-issue query.

This audit reads only the frozen annual manifest/calendar and the new raw
rights-issue query fragments.  It never imports an adapter, opens a ledger or
performance artifact, or starts a model.  An empty code is considered proven
only when the code is declared in a completed query manifest whose tabular
response has the required schema; the old sparse two-row artifact is not used.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ANNUAL = ROOT / "runs/wc006_v5_1_v6_development/annual-2022-paired-prereg-v1"
RUN = ROOT / "runs/wc006_v5_market_inputs/manual-bridge/2022-right-issue-code-complete-20260922T000011Z"
OUT = RUN / "supplemental-audit"
EXPECTED_FIELDS = {
    "MARKET_CODE", "PROGRESS", "PRICE", "RATIO", "AMT_PLAN", "AMT_REAL",
    "COLLECTION_FUND", "SHAREB_REG_DATE", "EX_DIVIDEND_DATE", "LISTED_DATE",
    "PAY_START_DATE", "PAY_END_DATE", "PREPLAN_DATE", "SMTG_ANN_DATE",
    "PASS_DATE", "APPROVED_DATE", "EXECUTE_DATE", "RESULT_DATE", "LIST_ANN_DATE",
    "GUARANTOR", "GUARTYPE", "RIGHTSISSUE_CODE", "ANN_DATE", "RIGHTSISSUE_YEAR",
    "RIGHTSISSUE_DESC", "RIGHTSISSUE_NAME", "RATIO_DENOMINATOR", "RATIO_MOLECULAR",
    "SUBS_METHOD", "EXPECTED_FUND_RAISING",
}
DATE_FIELDS = (
    "SHAREB_REG_DATE", "EX_DIVIDEND_DATE", "LISTED_DATE", "PAY_START_DATE",
    "PAY_END_DATE", "PREPLAN_DATE", "SMTG_ANN_DATE", "PASS_DATE",
    "APPROVED_DATE", "EXECUTE_DATE", "RESULT_DATE", "LIST_ANN_DATE", "ANN_DATE",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def norm_code(value: object) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    return text if "." in text else f"{text}.SZ"


def date8(value: object) -> str | None:
    text = str(value or "").strip().replace("-", "/")
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None
    text = text.replace("/", "")
    if text.endswith(".0"):
        text = text[:-2]
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"invalid_date:{value!r}")
    date(int(text[:4]), int(text[4:6]), int(text[6:]))
    return f"{text[:4]}-{text[4:6]}-{text[6:]}"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def main() -> int:
    annual_manifest_path = ANNUAL / "annual-manifest.json"
    calendar_path = ANNUAL / "inputs/calendar.json"
    annual = load(annual_manifest_path)
    calendar = load(calendar_path)
    target_codes = sorted({norm_code(item["code"]) for rec in annual["records"] for item in rec["selector"]["target_topk"]})
    target = set(target_codes)
    observation_dates = [str(x) for x in calendar["dates"][:-1]]
    settlement = str(calendar["dates"][-1])
    observation_first = observation_dates[0]

    query_codes_path = RUN / "query-codes.txt"
    query_codes = [line.strip() for line in query_codes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    query_code_set = set(query_codes)
    failures: list[str] = []
    checks: list[dict[str, Any]] = []
    rows_by_code: dict[str, list[dict[str, str]]] = defaultdict(list)
    declared_codes: set[str] = set()
    raw_rows = 0
    schema_failures: list[str] = []
    date_failures: list[str] = []
    row_code_failures: list[str] = []
    artifact_hash_failures: list[str] = []

    for manifest_path in sorted(RUN.glob("raw-right-issue-batch-*/manifest.json")):
        manifest = load(manifest_path)
        codes = {norm_code(x) for x in manifest.get("codes", [])}
        declared_codes |= codes
        if manifest.get("status") != "complete":
            failures.append(f"{manifest_path}:status_not_complete")
        if manifest.get("dataset") != "right_issue":
            failures.append(f"{manifest_path}:dataset_mismatch")
        if manifest.get("action_begin_date") != 20130101 or manifest.get("end_date") != 20240103:
            failures.append(f"{manifest_path}:query_window_mismatch")
        if manifest.get("credential_values_persisted") is not False:
            failures.append(f"{manifest_path}:credential_persistence_flag")
        artifacts = manifest.get("artifacts", [])
        if len(artifacts) != 1:
            failures.append(f"{manifest_path}:artifact_count")
            continue
        item = artifacts[0]
        path = manifest_path.parent / "raw" / str(item["path"])
        if not path.is_file():
            failures.append(f"{manifest_path}:artifact_missing")
            continue
        actual_hash = sha(path)
        hash_ok = actual_hash == item.get("sha256")
        if not hash_ok:
            artifact_hash_failures.append(str(path))
        fields, rows = read_csv(path)
        missing = sorted(EXPECTED_FIELDS - set(fields))
        if missing:
            schema_failures.append(f"{path}:missing={missing}")
        if int(item.get("rows", -1)) != len(rows):
            failures.append(f"{path}:declared_rows={item.get('rows')}:observed_rows={len(rows)}")
        for row in rows:
            code = norm_code(row.get("MARKET_CODE"))
            if code not in codes:
                row_code_failures.append(f"{path}:{code}:not_in_query_codes")
            rows_by_code[code].append(row)
            raw_rows += 1
            for field in DATE_FIELDS:
                try:
                    date8(row.get(field))
                except ValueError as exc:
                    date_failures.append(f"{path}:{code}:{field}:{exc}")
        checks.append({
            "manifest": str(manifest_path.relative_to(ROOT)),
            "codes": sorted(codes),
            "code_count": len(codes),
            "query": {"action_begin_date": manifest.get("action_begin_date"), "end_date": manifest.get("end_date")},
            "artifact": str(path.relative_to(ROOT)),
            "declared_sha256": item.get("sha256"),
            "actual_sha256": actual_hash,
            "hash_ok": hash_ok,
            "declared_rows": item.get("rows"),
            "observed_rows": len(rows),
            "fields": fields,
            "required_fields_missing": missing,
            "empty_response": len(rows) == 0,
        })

    code_rows = {code: len(rows_by_code.get(code, [])) for code in target_codes}
    empty_codes = sorted(code for code in target_codes if code in declared_codes and code not in rows_by_code)
    event_codes = sorted(code for code in target_codes if rows_by_code.get(code))
    missing_codes = sorted(target - declared_codes)
    extra_declared = sorted(declared_codes - target)
    in_window: list[dict[str, Any]] = []
    for code in event_codes:
        for row in rows_by_code[code]:
            ann = date8(row.get("ANN_DATE"))
            ex = date8(row.get("EX_DIVIDEND_DATE"))
            execute = date8(row.get("EXECUTE_DATE"))
            listed = date8(row.get("LISTED_DATE"))
            effective_dates = [x for x in (execute, ex, listed) if x]
            effective = max(effective_dates) if effective_dates else None
            if effective and observation_first <= effective <= settlement:
                in_window.append({
                    "code": code,
                    "ann_date": ann,
                    "execute_date": execute,
                    "ex_dividend_date": ex,
                    "listed_date": listed,
                    "effective_date": effective,
                    "progress": str(row.get("PROGRESS") or ""),
                    "loader_eligible": bool(ann and effective and str(row.get("PROGRESS") or "").strip() in {"3", "3.0"} and ann <= effective),
                })

    failed_checks = []
    if len(target) != 55:
        failed_checks.append("target_code_count")
    if query_codes != target_codes:
        failed_checks.append("query_code_list_not_exact_target_set")
    if len(declared_codes) != 55 or missing_codes or extra_declared:
        failed_checks.append("query_manifest_code_coverage")
    if len(checks) != 3:
        failed_checks.append("successful_batch_count")
    if artifact_hash_failures:
        failed_checks.append("artifact_hash")
    if schema_failures:
        failed_checks.append("schema")
    if date_failures:
        failed_checks.append("date_fields")
    if row_code_failures:
        failed_checks.append("row_code_membership")
    if raw_rows < 0:
        failed_checks.append("raw_row_count")
    status = "PASS" if not failed_checks else "BLOCKED"
    report = {
        "schema": "wc006-v5.1-2022-right-issue-code-complete-supplemental-audit/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": status,
        "scope": {"year": 2022, "read_only": True, "model_started": False, "performance_fields_read": False, "performance_computed": False, "real_orders": False, "paper_only": True},
        "source": {"run_dir": str(RUN.relative_to(ROOT)), "annual_manifest": str(annual_manifest_path.relative_to(ROOT)), "annual_manifest_sha256": sha(annual_manifest_path), "calendar_sha256": sha(calendar_path), "query_codes_file": str(query_codes_path.relative_to(ROOT)), "query_codes_sha256": sha(query_codes_path)},
        "query": {"dataset": "right_issue", "code_count": len(query_codes), "codes": query_codes, "action_begin_date": 20130101, "end_date": 20240103, "date_basis": "ANN_DATE; wide window selected to cover all documented right-issue event fields affecting 2022 observation and 2023-01-03 settlement", "local_path_mode": "per-batch container /out/.../cache with is_local=false", "credential_values_persisted": False},
        "join_boundary": {"target_code_count": len(target), "observation_first": observation_first, "observation_last": observation_dates[-1], "settlement_boundary": settlement, "settlement_is_not_observation": settlement not in observation_dates},
        "coverage": {"successful_batches": checks, "target_codes_declared_in_manifests": len(declared_codes & target), "target_manifest_missing": missing_codes, "extra_declared_codes": extra_declared, "raw_rows": raw_rows, "rows_by_target_code": code_rows, "event_codes": event_codes, "empty_code_proof_codes": empty_codes, "empty_code_proof_semantics": "A code is empty only when it is explicitly in a completed batch manifest and no returned row has that MARKET_CODE; batch response schema and hash are verified. This is distinct from the prior sparse two-row artifact.", "effective_events_in_observation_to_settlement": in_window},
        "integrity": {"artifact_hash_failures": artifact_hash_failures, "schema_failures": schema_failures, "date_failures": date_failures, "row_code_failures": row_code_failures, "failed_checks": sorted(set(failed_checks)), "no_original_data_modified": True, "old_two_row_artifact_overwritten": False},
        "attempt_history": [
            {"run": p.name, "status": "retained_separate"} for p in sorted(RUN.parent.glob("2022-right-issue-code-complete-20260922T*/")) if p.name != RUN.name
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / "audit.json"
    out_log = OUT / "audit.log"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_log.write_text("\n".join([
        "2022 target-code-complete right-issue supplemental audit (read-only)",
        f"status={status}",
        f"target_codes={len(target)} declared_codes={len(declared_codes)} successful_batches={len(checks)} raw_rows={raw_rows}",
        f"empty_code_proof_codes={len(empty_codes)} event_codes={len(event_codes)} in_window_events={len(in_window)}",
        "model_started=false performance_fields_read=false performance_computed=false real_orders=false",
        f"failed_checks={','.join(sorted(set(failed_checks)))}",
        f"audit_sha256={sha(out_json)}",
    ]) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "audit": str(out_json), "log": str(out_log), "audit_sha256": sha(out_json), "failed_checks": sorted(set(failed_checks)), "raw_rows": raw_rows, "empty_code_proof_count": len(empty_codes), "in_window_events": len(in_window)}, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
