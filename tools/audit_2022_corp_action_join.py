#!/usr/bin/env python3
"""Read-only audit for the preregistered 2022 corporate-action join.

This audit deliberately does not import or run an adapter/model and does not
read any ledger, checkpoint, event, equity, return, fee, or turnover output.
It validates the frozen annual slice, the authorized market manifest and the
raw action/status/equity artifacts, then mirrors only the V4 action-loader
eligibility and action-id rules.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ANNUAL = ROOT / "runs/wc006_v5_1_v6_development/annual-2022-paired-prereg-v1"
AUTH = ROOT / "runs/wc006_v5_market_inputs/manual-bridge/authorized-v5-development-20260920T122237Z-controller-lock"
OUT = ANNUAL
ANNUAL_MANIFEST = ANNUAL / "annual-manifest.json"
INPUT_AUDIT = ANNUAL / "input-audit.json"
CALENDAR = ANNUAL / "inputs/calendar.json"
AUTH_MANIFEST = AUTH / "manifest.json"
AUTH_QA = AUTH / "quality_report.json"

DIV_FIELDS = {
    "MARKET_CODE", "DIV_PROGRESS", "DVD_PER_SHARE_STK",
    "DVD_PER_SHARE_PRE_TAX_CASH", "DVD_PER_SHARE_AFTER_TAX_CASH",
    "DATE_EQY_RECORD", "DATE_EX", "DATE_DVD_PAYOUT",
    "LISTINGDATE_OF_DVD_SHR", "DIV_PRELANDATE", "DIV_SMTGDATE",
    "DATE_DVD_ANN", "DIV_BASEDATE", "DIV_BASESHARE", "CURRENCY_CODE",
    "ANN_DATE", "IS_CHANGED", "REPORT_PERIOD", "DIV_CHANGE",
    "DIV_BONUSRATE", "DIV_CONVERSEDRATE", "REMARK", "DIV_PREANN_DATE",
    "DIV_TARGET",
}
RIGHT_FIELDS = {
    "MARKET_CODE", "PROGRESS", "PRICE", "RATIO", "AMT_PLAN", "AMT_REAL",
    "COLLECTION_FUND", "SHAREB_REG_DATE", "EX_DIVIDEND_DATE", "LISTED_DATE",
    "PAY_START_DATE", "PAY_END_DATE", "PREPLAN_DATE", "SMTG_ANN_DATE",
    "PASS_DATE", "APPROVED_DATE", "EXECUTE_DATE", "RESULT_DATE",
    "LIST_ANN_DATE", "GUARANTOR", "GUARTYPE", "RIGHTSISSUE_CODE", "ANN_DATE",
    "RIGHTSISSUE_YEAR", "RIGHTSISSUE_DESC", "RIGHTSISSUE_NAME",
    "RATIO_DENOMINATOR", "RATIO_MOLECULAR", "SUBS_METHOD",
    "EXPECTED_FUND_RAISING",
}
EQ_FIELDS = {
    "MARKET_CODE", "ANN_DATE", "CHANGE_DATE", "SHARE_CHANGE_REASON_STR",
    "EX_CHANGE_DATE", "CURRENT_SIGN", "IS_VALID", "TOT_SHARE", "FLOAT_SHARE",
    "FLOAT_A_SHARE", "FLOAT_B_SHARE", "FLOAT_HK_SHARE", "FLOAT_OS_SHARE",
    "TOT_TRADABLE_SHARE", "RTD_A_SHARE_INST", "RTD_A_SHARE_DOMESNP",
    "RTD_SHARE_SENIOR", "RTD_A_SHARE_FOREIGN", "RTD_A_SHARE_FORJUR",
    "RTD_A_SHARE_FORNP", "RESTRICTED_B_SHARE", "OTHER_RTD_SHARE",
    "NON_TRADABLE_SHARE", "NTRD_SHARE_STATE_PCT", "NTRD_SHARE_STATE",
    "NTRD_SHARE_STATEJUR", "NTRD_SHARE_DOMESJUR", "NTRD_SHARE_DOMES_INITIATOR",
    "NTRD_SHARE_IPOJURIS", "NTRD_SHARE_GENJURIS", "NTRD_SHARE_STRA_INVESTOR",
    "NTRD_SHARE_FUND", "NTRD_SHARE_NAT", "TRAN_SHARE", "FLOAT_SHARE_SENIOR",
    "SHARE_INEMP", "PREFERRED_SHARE", "NTRD_SHARE_NLIST_FRGN", "STAQ_SHARE",
    "NET_SHARE", "SHARE_CHANGE_REASON", "TOT_A_SHARE", "TOT_B_SHARE",
    "OTCA_SHARE", "OTCB_SHARE", "TOT_OTC_SHARE", "SHARE_HK",
    "PRE_NON_TRADABLE_SHARE", "RESTRICTED_A_SHARE", "RTD_A_SHARE_STATE",
    "RTD_A_SHARE_STATEJUR", "RTD_A_SHARE_OTHER_DOMES",
    "RTD_A_SHARE_OTHER_DOMESJUR", "TOT_RESTRICTED_SHARE",
}
STATUS_FIELDS = {
    "MARKET_CODE", "TRADE_DATE", "PRECLOSE", "HIGH_LIMITED", "LOW_LIMITED",
    "PRICE_HIGH_LMT_RATE", "PRICE_LOW_LMT_RATE", "IS_ST_SEC", "IS_SUSP_SEC",
    "IS_WD_SEC", "IS_XR_SEC",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def norm_code(value: object) -> str:
    text = str(value or "").strip().upper()
    return text if "." in text else f"{text}.SZ"


def code_alias(value: object) -> str:
    return norm_code(value).split(".", 1)[0]


def norm_date(value: object) -> str | None:
    text = str(value or "").strip().replace("-", "")
    if text.endswith(".0"):
        text = text[:-2]
    if not text or text.lower() in {"nan", "none"}:
        return None
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"invalid date {value!r}")
    # Validate calendar dates as well as the YYYYMMDD shape.
    date(int(text[:4]), int(text[4:6]), int(text[6:]))
    return f"{text[:4]}-{text[4:6]}-{text[6:]}"


def action_id(kind: str, code: str, record: str | None, ex: str | None,
              payout: str | None, listing: str | None) -> str:
    identity = json.dumps([kind, code, record, ex, payout, listing], separators=(",", ":"))
    return hashlib.sha256(identity.encode()).hexdigest()


def child_entries(payload: dict[str, Any], dataset: str) -> list[tuple[Path, dict[str, Any]]]:
    out: list[tuple[Path, dict[str, Any]]] = []
    for child in payload.get("child_manifests", []):
        if child.get("dataset") != dataset:
            continue
        manifest = (AUTH / str(child["fragment"])).resolve()
        out.append((manifest, child))
    return out


def artifact_path(manifest_path: Path, rel: str) -> Path:
    direct = manifest_path.parent / rel
    raw = manifest_path.parent / "raw" / rel
    if direct.is_file():
        return direct
    if raw.is_file():
        return raw
    raise FileNotFoundError(f"artifact missing: {manifest_path}:{rel}")


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def manifest_artifacts(payload: dict[str, Any], dataset: str, expected_fields: set[str],
                       target_codes: set[str], required_dates_by_code: dict[str, list[str]], default_required_dates: list[str]) -> dict[str, Any]:
    entries = child_entries(payload, dataset)
    child_hashes_ok = True
    manifest_codes: set[str] = set()
    artifact_records: list[dict[str, Any]] = []
    rows_by_code: dict[str, list[dict[str, str]]] = defaultdict(list)
    all_rows = 0
    schema_failures: list[str] = []
    date_failures: list[str] = []
    duplicate_keys: list[str] = []
    key_seen: set[tuple[str, str]] = set()
    for manifest_path, child in entries:
        if not manifest_path.is_file():
            child_hashes_ok = False
            artifact_records.append({"manifest": str(manifest_path), "exists": False})
            continue
        declared_mh = child.get("manifest_sha256")
        actual_mh = sha256(manifest_path)
        child_hashes_ok &= actual_mh == declared_mh
        manifest = load(manifest_path)
        manifest_codes |= {norm_code(c) for c in manifest.get("codes", [])}
        for item in manifest.get("artifacts", []):
            p = artifact_path(manifest_path, str(item["path"]))
            actual = sha256(p)
            hash_ok = actual == item.get("sha256")
            fields, rows = read_rows(p)
            schema_check = dataset != "equity_structure" or "MARKET_CODE" in fields
            missing_fields = sorted(expected_fields - set(fields))
            if missing_fields and schema_check:
                schema_failures.append(f"{p}: missing {missing_fields}")
            artifact_records.append({
                "dataset": dataset,
                "manifest": str(manifest_path.relative_to(ROOT)),
                "artifact": str(p.relative_to(ROOT)),
                "manifest_sha256_declared": declared_mh,
                "manifest_sha256_actual": actual_mh,
                "manifest_hash_ok": actual_mh == declared_mh,
                "artifact_sha256_declared": item.get("sha256"),
                "artifact_sha256_actual": actual,
                "artifact_hash_ok": hash_ok,
                "declared_rows": item.get("rows"),
                "observed_rows": len(rows),
                "fields": fields,
            })
            all_rows += len(rows)
            for row in rows:
                code = norm_code(row.get("MARKET_CODE"))
                if code == ".SZ":
                    continue
                rows_by_code[code].append(row)
                if dataset == "history_stock_status":
                    d = norm_date(row.get("TRADE_DATE"))
                    if d is None:
                        date_failures.append(f"{p}: blank TRADE_DATE for {code}")
                    else:
                        k = (code, d)
                        if k in key_seen:
                            duplicate_keys.append(f"{code}|{d}")
                        key_seen.add(k)
                elif dataset == "equity_structure" and "MARKET_CODE" in fields:
                    for field in ("ANN_DATE", "CHANGE_DATE", "EX_CHANGE_DATE"):
                        try:
                            norm_date(row.get(field))
                        except ValueError as exc:
                            date_failures.append(f"{p}: {code} {field}: {exc}")
    target: dict[str, Any] = {}
    for code in sorted(target_codes):
        rows = rows_by_code.get(code, [])
        if dataset == "history_stock_status":
            dates = {norm_date(r.get("TRADE_DATE")) for r in rows}
            required_dates = required_dates_by_code.get(code, default_required_dates)
            missing = [d for d in required_dates if d not in dates]
            target[code] = {
                "manifest_declares_code": code in manifest_codes,
                "raw_rows": len(rows),
                "date_rows": len(dates),
                "required_date_count": len(required_dates),
                "missing_required_dates": missing,
                "status": "complete" if code in manifest_codes and not missing else "missing_date_rows",
            }
        else:
            dates = []
            for r in rows:
                for field in ("ANN_DATE", "CHANGE_DATE", "EX_CHANGE_DATE"):
                    d = norm_date(r.get(field))
                    if d:
                        dates.append(d)
            target[code] = {
                "manifest_declares_code": code in manifest_codes,
                "raw_rows": len(rows),
                "date_min": min(dates) if dates else None,
                "date_max": max(dates) if dates else None,
                "status": "complete" if code in manifest_codes and rows else "missing_raw_code_rows",
            }
    return {
        "date_keys_by_code": {code: sorted({norm_date(r.get("TRADE_DATE")) for r in rows if norm_date(r.get("TRADE_DATE"))}) for code, rows in rows_by_code.items()},
        "child_manifest_count": len(entries),
        "manifest_codes": len(manifest_codes),
        "target_codes": len(target_codes),
        "target_manifest_missing": sorted(target_codes - manifest_codes),
        "rows": all_rows,
        "child_manifest_hashes_ok": child_hashes_ok,
        "artifact_hashes_ok": all(x.get("artifact_hash_ok", False) for x in artifact_records),
        "schema_failures": schema_failures,
        "date_failures": date_failures,
        "duplicate_keys": duplicate_keys,
        "per_target_code": target,
        "artifacts": artifact_records,
    }


def action_dataset(payload: dict[str, Any], dataset: str, fields: set[str], target_codes: set[str],
                   observation_first: str, settlement: str) -> dict[str, Any]:
    entries = child_entries(payload, dataset)
    manifests_ok = True
    artifacts_ok = True
    schema_failures: list[str] = []
    date_failures: list[str] = []
    raw_rows: list[dict[str, Any]] = []
    manifest_codes: set[str] = set()
    artifacts: list[dict[str, Any]] = []
    seen_identity: dict[str, dict[str, Any]] = {}
    duplicate_identity: list[str] = []
    conflicting_identity: list[str] = []
    for manifest_path, child in entries:
        if not manifest_path.is_file():
            manifests_ok = False
            continue
        actual_mh = sha256(manifest_path)
        manifests_ok &= actual_mh == child.get("manifest_sha256")
        manifest = load(manifest_path)
        manifest_codes |= {norm_code(c) for c in manifest.get("codes", [])}
        for item in manifest.get("artifacts", []):
            p = artifact_path(manifest_path, str(item["path"]))
            actual = sha256(p)
            hash_ok = actual == item.get("sha256")
            artifacts_ok &= hash_ok
            fields_seen, rows = read_rows(p)
            missing = sorted(fields - set(fields_seen))
            if missing:
                schema_failures.append(f"{p}: missing {missing}")
            artifacts.append({
                "manifest": str(manifest_path.relative_to(ROOT)),
                "artifact": str(p.relative_to(ROOT)),
                "manifest_sha256_declared": child.get("manifest_sha256"),
                "manifest_sha256_actual": actual_mh,
                "manifest_hash_ok": actual_mh == child.get("manifest_sha256"),
                "artifact_sha256_declared": item.get("sha256"),
                "artifact_sha256_actual": actual,
                "artifact_hash_ok": hash_ok,
                "declared_rows": item.get("rows"),
                "observed_rows": len(rows),
                "fields": fields_seen,
            })
            for row in rows:
                code = norm_code(row.get("MARKET_CODE"))
                try:
                    ann = norm_date(row.get("ANN_DATE"))
                    if ann is None:
                        date_failures.append(f"{p}: {code} blank ANN_DATE")
                except ValueError as exc:
                    date_failures.append(f"{p}: {code} ANN_DATE: {exc}")
                    ann = None
                if dataset == "dividend":
                    record = norm_date(row.get("DATE_EQY_RECORD"))
                    ex = norm_date(row.get("DATE_EX"))
                    payout = norm_date(row.get("DATE_DVD_PAYOUT"))
                    listing = norm_date(row.get("LISTINGDATE_OF_DVD_SHR"))
                    effective_dates = [x for x in (record, ex, payout, listing) if x]
                    effective = max(effective_dates) if effective_dates else None
                    implemented = str(row.get("DIV_PROGRESS") or "").strip() in {"3", "3.0"}
                    eligible = bool(ann and effective and implemented and ann <= effective)
                    aid = action_id("DIVIDEND", code, record, ex, payout, listing)
                    action = {"kind": "DIVIDEND", "code": code, "ann_date": ann,
                              "record_date": record, "ex_date": ex, "payout_date": payout,
                              "listing_date": listing, "effective_date": effective,
                              "implemented": implemented, "loader_eligible": eligible,
                              "cash_per_share": str(row.get("DVD_PER_SHARE_PRE_TAX_CASH") or "0"),
                              "stock_ratio": str(row.get("DVD_PER_SHARE_STK") or "0"),
                              "bonus_ratio": str(row.get("DIV_BONUSRATE") or "0"),
                              "action_id": aid}
                    ident = aid
                else:
                    execute = norm_date(row.get("EXECUTE_DATE"))
                    ex = norm_date(row.get("EX_DIVIDEND_DATE"))
                    listed = norm_date(row.get("LISTED_DATE"))
                    effective_dates = [x for x in (execute, ex, listed) if x]
                    effective = max(effective_dates) if effective_dates else None
                    implemented = str(row.get("PROGRESS") or "").strip() in {"3", "3.0"}
                    eligible = bool(ann and effective and implemented and ann <= effective)
                    aid = action_id("RIGHTS", code, None, ex, None, listed)
                    action = {"kind": "RIGHTS", "code": code, "ann_date": ann,
                              "execute_date": execute, "ex_date": ex, "listing_date": listed,
                              "effective_date": effective, "implemented": implemented,
                              "loader_eligible": eligible, "rights_ratio": str(row.get("RATIO") or "0"),
                              "rights_price": str(row.get("PRICE") or "0"), "action_id": aid}
                    ident = aid
                prior = seen_identity.get(ident)
                if prior is not None:
                    duplicate_identity.append(ident)
                    if action.get("ann_date") and prior.get("ann_date") and action["ann_date"] > prior["ann_date"]:
                        seen_identity[ident] = action
                    elif action.get("ann_date") == prior.get("ann_date") and prior != action:
                        conflicting_identity.append(ident)
                else:
                    seen_identity[ident] = action
                raw_rows.append(action)
    eligible = [x for x in seen_identity.values() if x["loader_eligible"]]
    by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in eligible:
        by_code[item["code"]].append(item)
    per_code: dict[str, Any] = {}
    for code in sorted(target_codes):
        raw_for_code = [x for x in raw_rows if x["code"] == code]
        eligible_for_code = by_code.get(code, [])
        in_window = [x for x in eligible_for_code if x.get("effective_date") and observation_first <= x["effective_date"] <= settlement]
        if dataset == "right_issue":
            # The source query is sparse and its two rows do not provide a
            # per-code negative assertion.  Keep this distinct from proven
            # no-event evidence even where the child manifest lists the code.
            status = "event_found" if in_window else "unresolved_sparse_right_issue_state"
        else:
            status = "event_found" if in_window else ("no_event_evidence" if code in manifest_codes else "missing_query_code")
        per_code[code] = {
            "manifest_declares_code": code in manifest_codes,
            "raw_rows": len(raw_for_code),
            "loader_eligible_rows": len(eligible_for_code),
            "eligible_in_observation_to_settlement": len(in_window),
            "status": status,
            "actions": sorted(in_window, key=lambda x: (x.get("effective_date") or "", x["action_id"])),
        }
    return {
        "child_manifest_count": len(entries),
        "manifest_codes": len(manifest_codes),
        "target_codes": len(target_codes),
        "target_manifest_missing": sorted(target_codes - manifest_codes),
        "raw_rows": len(raw_rows),
        "unique_loader_actions": len(seen_identity),
        "loader_eligible_unique": len(eligible),
        "child_manifest_hashes_ok": manifests_ok,
        "artifact_hashes_ok": artifacts_ok,
        "schema_failures": schema_failures,
        "date_failures": date_failures,
        "duplicate_identity_count": len(duplicate_identity),
        "conflicting_identity_count": len(conflicting_identity),
        "duplicate_identity_ids": sorted(set(duplicate_identity)),
        "conflicting_identity_ids": sorted(set(conflicting_identity)),
        "per_target_code": per_code,
        "eligible_actions": sorted(eligible, key=lambda x: (x["code"], x.get("effective_date") or "", x["action_id"])),
        "artifacts": artifacts,
    }


def main() -> int:
    annual = load(ANNUAL_MANIFEST)
    input_audit = load(INPUT_AUDIT)
    calendar = load(CALENDAR)
    auth = load(AUTH_MANIFEST)
    qa = load(AUTH_QA)
    obs_dates = list(calendar["dates"][:-1])
    settlement = str(calendar["dates"][-1])
    target_codes = {code_alias(item["code"]) for rec in annual["records"] for item in rec["selector"]["target_topk"]}
    target_codes_full = {f"{c}.SZ" for c in target_codes}
    signal_execution = [[rec["signal_date"], rec["execution_date"]] for rec in annual["records"]]
    required_dates = obs_dates + [settlement]
    required_dates_by_code = {}
    for code in target_codes_full:
        first_signal = min(rec["signal_date"] for rec in annual["records"] if any(code_alias(x["code"]) == code_alias(code) for x in rec["selector"]["target_topk"]))
        required_dates_by_code[code] = [d for d in obs_dates if d >= first_signal] + [settlement]

    # Provenance/hash checks are intentionally limited to input artifacts.
    provenance = {
        "annual_manifest": {"path": str(ANNUAL_MANIFEST.relative_to(ROOT)), "sha256": sha256(ANNUAL_MANIFEST)},
        "input_audit": {"path": str(INPUT_AUDIT.relative_to(ROOT)), "sha256": sha256(INPUT_AUDIT)},
        "calendar": {"path": str(CALENDAR.relative_to(ROOT)), "sha256": sha256(CALENDAR)},
        "authorized_market_manifest": {"path": str(AUTH_MANIFEST.relative_to(ROOT)), "sha256": sha256(AUTH_MANIFEST)},
        "authorized_quality_report": {"path": str(AUTH_QA.relative_to(ROOT)), "sha256": sha256(AUTH_QA)},
        "declared_input_bindings": {
            "annual_manifest_sha256": input_audit.get("input_binding", {}).get("annual_manifest_sha256"),
            "market_manifest_sha256": input_audit.get("input_binding", {}).get("market_manifest_sha256"),
            "market_quality_report_sha256": input_audit.get("input_binding", {}).get("market_quality_report_sha256"),
        },
        "authorization_quality_status": qa.get("status"),
        "authorization_quality_remaining_blockers": qa.get("remaining_blockers"),
    }
    provenance["binding_hash_checks"] = {
        "annual_manifest": provenance["annual_manifest"]["sha256"] == input_audit.get("input_binding", {}).get("annual_manifest_sha256"),
        "market_manifest": provenance["authorized_market_manifest"]["sha256"] == input_audit.get("input_binding", {}).get("market_manifest_sha256"),
        "quality_report": provenance["authorized_quality_report"]["sha256"] == input_audit.get("input_binding", {}).get("market_quality_report_sha256"),
    }

    dividend = action_dataset(auth, "dividend", DIV_FIELDS, target_codes_full, obs_dates[0], settlement)
    rights = action_dataset(auth, "right_issue", RIGHT_FIELDS, target_codes_full, obs_dates[0], settlement)
    equity = manifest_artifacts(auth, "equity_structure", EQ_FIELDS, target_codes_full, required_dates_by_code, obs_dates + [settlement])
    status = manifest_artifacts(auth, "history_stock_status", STATUS_FIELDS, target_codes_full, required_dates_by_code, obs_dates + [settlement])

    # Supporting kline date-key join: no OHLC values are opened or retained.
    kline = {"dataset": "kline", "child_manifest_count": 0, "target_codes": len(target_codes_full),
             "target_manifest_missing": [], "per_target_code": {}, "date_key_failures": [],
             "artifacts": []}
    kline_entries = child_entries(auth, "kline")
    kline["child_manifest_count"] = len(kline_entries)
    kline_codes: set[str] = set()
    for mp, child in kline_entries:
        if not mp.is_file():
            kline["date_key_failures"].append(f"missing child manifest: {mp}")
            continue
        kline_codes |= {norm_code(c) for c in load(mp).get("codes", [])}
        for item in load(mp).get("artifacts", []):
            p = artifact_path(mp, str(item["path"]))
            actual = sha256(p)
            fields, rows = read_rows(p)
            dates = []
            for row in rows:
                try:
                    dates.append(norm_date(row.get("kline_time")))
                except ValueError as exc:
                    kline["date_key_failures"].append(f"{p}: {exc}")
            code = norm_code(item.get("code"))
            kline["artifacts"].append({"artifact": str(p.relative_to(ROOT)), "code": code,
                                        "declared_sha256": item.get("sha256"), "actual_sha256": actual,
                                        "hash_ok": actual == item.get("sha256"), "declared_rows": item.get("rows"),
                                        "observed_rows": len(rows), "fields": fields})
            if code in target_codes_full:
                required_for_code = required_dates_by_code.get(code, obs_dates + [settlement])
                missing = [d for d in required_for_code if d not in set(dates)]
                status_dates = set(status.get("date_keys_by_code", {}).get(code, []))
                gap_is_documented = bool(missing) and all(d in status_dates for d in missing)
                kline["per_target_code"][code] = {"manifest_declares_code": code in kline_codes,
                                                  "observed_rows": len(rows), "required_date_count": len(required_for_code), "missing_required_dates": missing,
                                                  "status": "complete" if not missing else ("documented_raw_date_gap" if gap_is_documented else "missing_official_status_date")}
    kline["manifest_codes"] = len(kline_codes)
    kline["target_manifest_missing"] = sorted(target_codes_full - kline_codes)
    kline["artifact_hashes_ok"] = all(x["hash_ok"] for x in kline["artifacts"])
    kline["documented_raw_gap_note"] = "Raw Kline date-key gaps with official status rows are retained under the accepted qfq/raw suspension semantic split; no raw rows are filled."
    status_date_keys = status.pop("date_keys_by_code", {})

    unresolved: list[dict[str, Any]] = []
    for code, item in rights["per_target_code"].items():
        if item["status"] != "event_found":
            unresolved.append({"dataset": "right_issue", "code": code,
                               "reason": "two raw right_issue rows cannot prove a code-complete negative result",
                               "classification": item["status"]})
    for dataset_name, result in (("dividend", dividend), ("equity_structure", equity), ("history_stock_status", status), ("kline", kline)):
        for code in result.get("target_manifest_missing", []):
            unresolved.append({"dataset": dataset_name, "code": code, "reason": "target code absent from query manifest"})
        for code, item in result.get("per_target_code", {}).items():
            if item.get("status") in {"missing_date_rows", "missing_raw_code_rows", "missing_official_status_date"}:
                unresolved.append({"dataset": dataset_name, "code": code, "reason": item["status"],
                                   "missing_required_dates": item.get("missing_required_dates", [])})
    failed_checks = []
    if not provenance["binding_hash_checks"]["annual_manifest"]:
        failed_checks.append("annual_manifest_binding_hash")
    if not provenance["binding_hash_checks"]["market_manifest"]:
        failed_checks.append("market_manifest_binding_hash")
    if not provenance["binding_hash_checks"]["quality_report"]:
        failed_checks.append("quality_report_binding_hash")
    for name, result in (("dividend", dividend), ("right_issue", rights), ("equity_structure", equity), ("history_stock_status", status), ("kline", kline)):
        if not result.get("child_manifest_hashes_ok", True):
            failed_checks.append(f"{name}_child_manifest_hash")
        if not result.get("artifact_hashes_ok", False):
            failed_checks.append(f"{name}_artifact_hash")
        if result.get("schema_failures"):
            failed_checks.append(f"{name}_schema")
        if result.get("date_failures") or result.get("date_key_failures"):
            failed_checks.append(f"{name}_date_keys")
        if result.get("duplicate_keys") or result.get("conflicting_identity_count"):
            failed_checks.append(f"{name}_duplicates_or_conflicts")
    failed_checks.append("right_issue_code_complete")
    result_status = "BLOCKED" if failed_checks or unresolved else "PASS"
    report = {
        "schema": "wc006-v5.1-2022-corp-action-join-audit/v1",
        "created_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": result_status,
        "scope": {"year": 2022, "read_only": True, "model_started": False,
                  "performance_fields_read": False, "performance_computed": False,
                  "real_orders": False, "paper_only": True},
        "frozen_join_boundary": {"target_code_count": len(target_codes_full),
                                 "target_codes": sorted(target_codes_full),
                                 "observation_count": len(obs_dates),
                                 "observation_first": obs_dates[0], "observation_last": obs_dates[-1],
                                 "settlement_boundary": settlement, "settlement_is_not_observation": settlement not in obs_dates,
                                 "signal_execution_pairs": signal_execution,
                                 "required_status_price_dates_count": len(obs_dates + [settlement]),
                                 "required_dates_basis": "per-code from first selector signal through settlement boundary"},
        "query_boundaries": {"action_begin_date": "2020-01-01", "action_end_date": "2024-01-02",
                             "market_begin_date": "2021-01-01", "market_end_date": "2023-12-29",
                             "settlement_inside_market_window": True},
        "provenance": provenance,
        "loader_contract": {"loader_path": "experiments/a_share/wc006_v4_market.py",
                            "action_filter": "ANN_DATE <= max(applicable effective dates) and DIV_PROGRESS/PROGRESS == 3",
                            "dividend_identity": "SHA256([DIVIDEND, code, record, ex, payout, listing])",
                            "rights_identity": "SHA256([RIGHTS, code, null, ex, null, listed])",
                            "rights_policy": "DECLINE_NO_SUBSCRIPTION",
                            "entitlement_note": "Actual held-position entitlement is not assessed in this input-only audit; no model holdings or performance state was opened.",
                            "loader_parse_result": {"dividend_unique_eligible": dividend["loader_eligible_unique"],
                                                    "right_issue_unique_eligible": rights["loader_eligible_unique"]}},
        "coverage": {"dividend": dividend, "right_issue": rights, "equity_structure": equity,
                     "history_stock_status": status, "kline_date_keys_only": kline},
        "unresolved": unresolved,
        "failed_checks": sorted(set(failed_checks)),
        "integrity": {"no_performance_fields_read": True, "no_model_run": True,
                       "no_original_data_modified": True, "no_manifest_modified": True},
    }
    out_json = OUT / "corp-action-join-audit.json"
    out_log = OUT / "corp-action-join-audit.log"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log_lines = [
        "2022 corporate-action target-code/date join audit (read-only)",
        f"status={result_status}",
        f"target_codes={len(target_codes_full)} observation_dates={len(obs_dates)} settlement={settlement}",
        "model_started=false performance_fields_read=false performance_computed=false real_orders=false",
        f"dividend.raw_rows={dividend['raw_rows']} dividend.loader_eligible_unique={dividend['loader_eligible_unique']}",
        f"right_issue.raw_rows={rights['raw_rows']} right_issue.loader_eligible_unique={rights['loader_eligible_unique']}",
        "right_issue_code_complete=false: two raw rows are not treated as code-complete",
        f"equity_structure.raw_rows={equity['rows']} history_stock_status.raw_rows={status['rows']}",
        f"failed_checks={','.join(sorted(set(failed_checks)))}",
        f"unresolved_count={len(unresolved)}",
        f"audit_sha256={sha256(out_json)}",
    ]
    out_log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result_status, "audit": str(out_json), "log": str(out_log),
                      "audit_sha256": sha256(out_json), "failed_checks": sorted(set(failed_checks)),
                      "unresolved_count": len(unresolved)}, indent=2))
    return 0 if result_status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
