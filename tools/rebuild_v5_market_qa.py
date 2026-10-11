#!/usr/bin/env python3
"""Rebuild a hash-pinned, fail-closed V5 market-input QA bundle.

This is intentionally offline and read-only with respect to all source bundles.
It writes only a new ``rebuilt-qa-*`` directory below the V5 manual bridge.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "runs/wc006_v5_market_inputs/manual-bridge"
PREP = BRIDGE / "prepared"
CSTAGE = ROOT / "runs/wc006_v4_xysz_full/20260919T054212Z-125codes"
QFQ = ROOT / "runs/wc006_v3_2021_inputs/20260918T-candidate-full-2021-2023-stateful/inputs/cache"
IMAGE_DIGEST = "sha256:57676d949eb5e79a3769c3714205f39d8940ae004e061a49c0174279379fd9bb"
IMAGE = "amazingdata-x86:1.1.9"
SDK = "1.1.9"
BEGIN, END, ACTION_BEGIN = "2021-01-01", "2024-01-02", "2020-01-01"
TRAIN_BEGIN, TRAIN_END = "2021-01-01", "2023-12-29"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def date_value(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "nat"}:
        return None
    if s.endswith(".0"):
        s = s[:-2]
    s = s.replace("/", "-")
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    return None


def dec(value: Any) -> Decimal | None:
    try:
        x = Decimal(str(value).strip())
    except (InvalidOperation, TypeError, ValueError):
        return None
    return x if x.is_finite() else None


def norm_fields(fields: list[str]) -> list[str]:
    return [x for x in fields if x not in {"", "index", "Unnamed: 0"}]


def artifact_path(fragment: Path, rel: str) -> Path:
    for p in (CSTAGE / fragment.parent / rel, CSTAGE / fragment.parent / "raw" / rel):
        if p.is_file():
            return p
    raise FileNotFoundError((fragment, rel))


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), [{str(k): str(v) for k, v in row.items()} for row in r]


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = BRIDGE / f"rebuilt-qa-{stamp}"
    if out.exists():
        raise RuntimeError(f"output_exists:{out}")
    out.mkdir(parents=True)

    prepared_manifest_path = PREP / "manifest.json"
    status_manifest_path = PREP / "history_stock_status_125_attempt2/manifest.json"
    cparent_path = CSTAGE / "manifest.json"
    cparent = json.loads(cparent_path.read_text(encoding="utf-8"))
    expected = [x.strip() for x in (PREP / "codes_125.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
    expected_set = set(expected)

    # Source inventory: old immutable C-stage datasets, plus the two current bridge artifacts.
    source_artifacts: list[dict[str, Any]] = []
    child_records: list[dict[str, Any]] = []
    old_parent_failures: list[str] = []
    selected = {"kline", "adj_factor", "dividend", "right_issue"}
    for child in cparent.get("child_manifests", []):
        if child.get("dataset") not in selected:
            continue
        rel = Path(str(child["fragment"]))
        mp = CSTAGE / rel
        payload = json.loads(mp.read_text(encoding="utf-8"))
        actual_manifest_hash = sha(mp)
        if actual_manifest_hash != child.get("manifest_sha256"):
            old_parent_failures.append(str(rel))
        cr = {
            "dataset": payload.get("dataset"),
            "fragment": str(rel),
            "manifest_sha256_declared": child.get("manifest_sha256"),
            "manifest_sha256_actual": actual_manifest_hash,
            "manifest_hash_ok": actual_manifest_hash == child.get("manifest_sha256"),
            "status": payload.get("status"),
            "started_at_utc": payload.get("started_at_utc"),
            "finished_at_utc": payload.get("finished_at_utc"),
            "sdk_version": payload.get("sdk_version", SDK),
            "begin_date": payload.get("begin_date", int(BEGIN.replace("-", ""))),
            "end_date": payload.get("end_date", int(END.replace("-", ""))),
            "action_begin_date": payload.get("action_begin_date", int(ACTION_BEGIN.replace("-", ""))),
            "codes": payload.get("codes", []),
            "artifacts": [],
        }
        for item in payload.get("artifacts", []):
            p = artifact_path(rel, str(item["path"]))
            actual = sha(p)
            ar = {
                "path": str(p.relative_to(ROOT)),
                "declared_sha256": item.get("sha256"),
                "actual_sha256": actual,
                "hash_ok": actual == item.get("sha256"),
                "rows_declared": item.get("rows"),
                "rows_actual": sum(1 for _ in csv.DictReader(p.open(newline="", encoding="utf-8"))),
                "fields": item.get("fields", []),
            }
            cr["artifacts"].append(ar)
            source_artifacts.append({"dataset": payload.get("dataset"), "fragment": str(rel), **ar})
        child_records.append(cr)

    eq_path = PREP / "raw/equity_structure.csv"
    status_path = PREP / "history_stock_status_125_attempt2/raw/history_stock_status.csv"
    eq_fields, eq_rows = read_rows(eq_path)
    st_fields, st_rows = read_rows(status_path)
    eq_actual_hash, st_actual_hash = sha(eq_path), sha(status_path)
    eq_declared_hash = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))["artifacts"][0]["sha256"]
    st_declared_hash = json.loads(status_manifest_path.read_text(encoding="utf-8"))["artifacts"][0]["sha256"]
    staged = {
        "equity_structure": {"path": str(eq_path.relative_to(ROOT)), "sha256": eq_actual_hash, "declared_sha256": eq_declared_hash, "hash_ok": eq_actual_hash == eq_declared_hash, "rows": len(eq_rows), "fields": norm_fields(eq_fields)},
        "history_stock_status": {"path": str(status_path.relative_to(ROOT)), "sha256": st_actual_hash, "declared_sha256": st_declared_hash, "hash_ok": st_actual_hash == st_declared_hash, "rows": len(st_rows), "fields": norm_fields(st_fields)},
    }

    # Kline key and price QA.
    kline: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    kline_schema: set[tuple[str, ...]] = set()
    kline_rows = 0
    price = Counter()
    seen_kline_keys: set[tuple[str, str]] = set()
    for rec in source_artifacts:
        if rec["dataset"] != "kline":
            continue
        p = ROOT / rec["path"]
        fields, rows = read_rows(p)
        kline_schema.add(tuple(norm_fields(fields)))
        for row in rows:
            code = str(row.get("code", "")).strip().upper()
            d = date_value(row.get("kline_time"))
            kline_rows += 1
            if not code or not d:
                price["blank_key"] += 1
                continue
            if (code, d) in seen_kline_keys:
                price["duplicate_code_date"] += 1
            seen_kline_keys.add((code, d))
            kline[code][d] = row
            vals = {x: dec(row.get(x)) for x in ("open", "high", "low", "close", "volume", "amount")}
            if any(vals[x] is None for x in ("open", "high", "low", "close")):
                price["nonfinite_ohlc"] += 1
            elif any(vals[x] <= 0 for x in ("open", "high", "low", "close")):
                price["nonpositive_ohlc"] += 1
            else:
                if vals["high"] < max(vals["open"], vals["close"], vals["low"]):
                    price["high_range_violation"] += 1
                if vals["low"] > min(vals["open"], vals["close"], vals["high"]):
                    price["low_range_violation"] += 1
            if vals["volume"] is None or vals["amount"] is None:
                price["nonfinite_volume_amount"] += 1
            elif vals["volume"] < 0 or vals["amount"] < 0:
                price["negative_volume_amount"] += 1
    kline_keys = {(c, d) for c, rr in kline.items() for d in rr}

    # Wide adjustment-factor QA: each immutable fragment must cover its declared
    # code columns with unique dates and finite positive factors.
    adj_codes: set[str] = set()
    adj_schema: set[tuple[str, ...]] = set()
    adj_rows = 0
    adj_duplicate_dates = 0
    adj_nonfinite = 0
    adj_nonpositive = 0
    for rec in source_artifacts:
        if rec["dataset"] != "adj_factor":
            continue
        fields, rows = read_rows(ROOT / rec["path"])
        cols = tuple(norm_fields(fields))
        adj_schema.add(cols)
        adj_codes.update(x for x in cols if x in expected_set)
        date_col = fields[0] if fields else ""
        seen_dates: set[str] = set()
        for row in rows:
            adj_rows += 1
            d = date_value(row.get(date_col))
            if d in seen_dates:
                adj_duplicate_dates += 1
            if d:
                seen_dates.add(d)
            for code in cols:
                value = dec(row.get(code))
                if value is None:
                    adj_nonfinite += 1
                elif value <= 0:
                    adj_nonpositive += 1

    # qfq date-key differences are preserved as evidence, never filled.
    qfq: dict[str, set[str]] = {}
    for p in sorted(QFQ.glob("*_5y_2021-01-01_2026-04-03.csv")):
        code = p.name.split("_", 1)[0] + ".SZ"
        _, rows = read_rows(p)
        qfq[code] = {d for row in rows if (d := date_value(row.get("date"))) and TRAIN_BEGIN <= d <= TRAIN_END}
    diff_rows: list[dict[str, Any]] = []
    for code in sorted(expected_set | set(qfq) | set(kline)):
        qdates = qfq.get(code, set())
        rdates = {d for d in kline.get(code, {}) if TRAIN_BEGIN <= d <= TRAIN_END}
        for d in sorted(qdates - rdates):
            diff_rows.append({"code": code, "date": d, "side": "qfq_only", "raw_present": False, "qfq_present": True})
        for d in sorted(rdates - qdates):
            diff_rows.append({"code": code, "date": d, "side": "raw_only", "raw_present": True, "qfq_present": False})
    write_csv(out / "date_key_diff.csv", ["code", "date", "side", "raw_present", "qfq_present"], diff_rows)

    # Status QA against raw kline keys.
    st_key = {(str(r.get("MARKET_CODE", "")).strip().upper(), date_value(r.get("TRADE_DATE"))) for r in st_rows}
    st_key.discard(("", None))
    status_schema = tuple(norm_fields(st_fields))
    overlap, missing, extra = kline_keys & st_key, kline_keys - st_key, st_key - kline_keys
    status_state = Counter()
    limit_kind = Counter()
    state_rows: list[dict[str, Any]] = []
    for r in st_rows:
        key = (str(r.get("MARKET_CODE", "")).strip().upper(), date_value(r.get("TRADE_DATE")))
        if key not in kline_keys:
            continue
        st = str(r.get("IS_ST_SEC", "")).strip().lower() in {"1", "1.0", "true"}
        susp = str(r.get("IS_SUSP_SEC", "")).strip().lower() in {"1", "1.0", "true"}
        sentinel = all(dec(r.get(x)) == Decimal("0") for x in ("HIGH_LIMITED", "LOW_LIMITED")) and all(dec(r.get(x)) == Decimal("999") for x in ("PRICE_HIGH_LMT_RATE", "PRICE_LOW_LMT_RATE"))
        state = "SUSPENDED" if susp else "ST" if st else "OPEN"
        status_state[state] += 1
        limit_kind["no_price_limit_sentinel" if sentinel else "quoted_limits"] += 1
        state_rows.append({"code": key[0], "date": key[1], "official_status": state, "is_st_sec": int(st), "is_susp_sec": int(susp), "limit_sentinel": int(sentinel), "retrieval_time_field_present": 0})
    write_csv(out / "official_executable_state.csv", ["code", "date", "official_status", "is_st_sec", "is_susp_sec", "limit_sentinel", "retrieval_time_field_present"], state_rows)

    # Equity structure QA.
    eq_codes = {str(r.get("MARKET_CODE", "")).strip().upper() for r in eq_rows if r.get("MARKET_CODE")}
    eq_keys = [(str(r.get("MARKET_CODE", "")).strip().upper(), date_value(r.get("ANN_DATE")), date_value(r.get("CHANGE_DATE")), date_value(r.get("EX_CHANGE_DATE")), str(r.get("SHARE_CHANGE_REASON", "")).strip()) for r in eq_rows]
    eq_dupes = len(eq_keys) - len(set(eq_keys))

    # PIT corporate action filtering, retaining every input row and reason.
    action_rows: list[dict[str, Any]] = []
    action_counts = Counter()
    eligible: list[dict[str, Any]] = []
    for rec in source_artifacts:
        if rec["dataset"] not in {"dividend", "right_issue"}:
            continue
        kind = "DIVIDEND" if rec["dataset"] == "dividend" else "RIGHTS"
        _, rows = read_rows(ROOT / rec["path"])
        for row in rows:
            code = str(row.get("MARKET_CODE", "")).strip().upper()
            ann = date_value(row.get("ANN_DATE"))
            progress = str(row.get("DIV_PROGRESS" if kind == "DIVIDEND" else "PROGRESS", "")).strip()
            names = ("DATE_EQY_RECORD", "DATE_EX", "DATE_DVD_PAYOUT", "LISTINGDATE_OF_DVD_SHR") if kind == "DIVIDEND" else ("EXECUTE_DATE", "EX_DIVIDEND_DATE", "LISTED_DATE")
            dates = {n: date_value(row.get(n)) for n in names}
            effective = max((x for x in dates.values() if x), default=None)
            if progress not in {"3", "3.0"}:
                reason = "progress_not_implemented"
            elif not ann:
                reason = "missing_ann_date"
            elif not effective:
                reason = "missing_effective_date"
            elif ann > effective:
                reason = "ann_after_effective_date"
            else:
                reason = "eligible"
            ident = [kind, code, ann, *dates.values(), row.get("PRICE", ""), row.get("RATIO", ""), row.get("DVD_PER_SHARE_PRE_TAX_CASH", ""), row.get("DVD_PER_SHARE_STK", ""), row.get("DIV_BONUSRATE", "")]
            aid = hashlib.sha256(json.dumps(ident, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
            item = {"action_id": aid, "kind": kind, "code": code, "ann_date": ann or "", "effective_date": effective or "", "progress": progress, "pit_reason": reason, "source": rec["path"]}
            action_rows.append(item)
            action_counts[reason] += 1
            if reason == "eligible":
                eligible.append(item)
    eligible_unique = {x["action_id"]: x for x in eligible}
    write_csv(out / "pit_action_rows.csv", ["action_id", "kind", "code", "ann_date", "effective_date", "progress", "pit_reason", "source"], action_rows)
    write_csv(out / "pit_action_eligible.csv", ["action_id", "kind", "code", "ann_date", "effective_date", "progress", "pit_reason", "source"], sorted(eligible_unique.values(), key=lambda x: (x["code"], x["ann_date"], x["action_id"])))

    # Provenance records actual persisted collector timestamps. Endpoint overrides were not persisted.
    eq_source_manifest = BRIDGE / "equity_structure_missing6_1990_attempt1/manifest.json"
    eq_src = json.loads(eq_source_manifest.read_text(encoding="utf-8"))
    st_src = json.loads(status_manifest_path.read_text(encoding="utf-8"))
    old_runs = [{"dataset": x["dataset"], "fragment": x["fragment"], "started_at_utc": x["started_at_utc"], "finished_at_utc": x["finished_at_utc"], "sdk_version": x["sdk_version"], "query": {"codes": x["codes"], "begin_date": x["begin_date"], "end_date": x["end_date"], "action_begin_date": x["action_begin_date"]}, "artifact_hashes": [{"path": a["path"], "sha256": a["actual_sha256"]} for a in x["artifacts"]]} for x in child_records]
    runs = old_runs + [
        {"dataset": "equity_structure", "source": str(eq_source_manifest.relative_to(ROOT)), "started_at_utc": eq_src.get("started_at_utc"), "finished_at_utc": eq_src.get("finished_at_utc"), "sdk_version": eq_src.get("sdk_version", SDK), "query": {"codes": eq_src.get("codes", []), "begin_date": eq_src.get("begin_date"), "end_date": eq_src.get("end_date"), "action_begin_date": eq_src.get("action_begin_date")}, "artifact_sha256": eq_src.get("artifacts", [{}])[0].get("sha256")},
        {"dataset": "history_stock_status", "source": str(status_manifest_path.relative_to(ROOT)), "started_at_utc": st_src.get("started_at_utc"), "finished_at_utc": st_src.get("finished_at_utc"), "sdk_version": st_src.get("sdk_version", SDK), "query": {"codes": st_src.get("codes", []), "begin_date": st_src.get("begin_date"), "end_date": st_src.get("end_date"), "action_begin_date": st_src.get("action_begin_date")}, "artifact_sha256": st_src.get("artifacts", [{}])[0].get("sha256")},
    ]
    provenance = {
        "schema": "wc006-v5-market-inputs/retrieval-provenance/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "collector": {"image": IMAGE, "image_digest": IMAGE_DIGEST, "sdk_version": SDK, "credential_values_persisted": False, "paper_only": True},
        "endpoint": {"host": "101.230.159.234", "port": 8600, "recorded_as": "collector script default", "override_persistence": False, "note": "The manifests do not persist invocation host/port overrides; this bundle does not claim an override was absent."},
        "runs": runs,
        "artifacts": {"equity_structure_staging": {"path": staged["equity_structure"]["path"], "sha256": eq_actual_hash}, "history_stock_status_attempt2": {"path": staged["history_stock_status"]["path"], "sha256": st_actual_hash}},
        "pit_revision": {"as_of_field_present": False, "revision_field_present": False, "claim": "API/source artifacts do not provide as-of revision or PIT history; this is unresolved."},
    }
    dump(out / "retrieval_provenance.json", provenance)

    # New parent manifest is a reference manifest: old C-stage artifacts are immutable sources, staged files are bridge sources.
    parent = {
        "schema": "wc006-v5-rebuilt-market-inputs/v1",
        "status": "complete_with_fail_closed_quality_notes",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "wc006_v5_market_inputs",
        "codes": expected,
        "begin_date": 20210101,
        "end_date": 20240102,
        "action_begin_date": 20200101,
        "source_cstage_parent": {"path": str(cparent_path.relative_to(ROOT)), "sha256": sha(cparent_path)},
        "immutable_source_children": child_records,
        "bridge_artifacts": staged,
        "quality_report": "quality_report.json",
        "retrieval_provenance": "retrieval_provenance.json",
        "paper_only": True,
        "broker_orders_enabled": False,
        "live_execution_authorized": False,
        "training_authorized": False,
        "credential_values_persisted": False,
        "source_manifest_hash_failures": old_parent_failures,
    }
    dump(out / "manifest.json", parent)
    rebuilt_manifest_hash = sha(out / "manifest.json")

    checks = {
        "source_manifest_hashes": not old_parent_failures and all(x["hash_ok"] for x in source_artifacts),
        "bridge_hashes": all(x["hash_ok"] for x in staged.values()),
        "coverage_125_codes": set(kline) == expected_set and eq_codes == expected_set and {str(r.get("MARKET_CODE", "")).strip().upper() for r in st_rows} == expected_set,
        "row_counts_declared": all(x["rows_declared"] == x["rows_actual"] for x in source_artifacts) and st_src.get("artifacts", [{}])[0].get("rows") == len(st_rows),
        "schema_kline": len(kline_schema) == 1 and set(next(iter(kline_schema), ())) == {"code", "kline_time", "open", "high", "low", "close", "volume", "amount"},
        "schema_status": set(status_schema) == {"MARKET_CODE", "TRADE_DATE", "PRECLOSE", "HIGH_LIMITED", "LOW_LIMITED", "PRICE_HIGH_LMT_RATE", "PRICE_LOW_LMT_RATE", "IS_ST_SEC", "IS_SUSP_SEC", "IS_WD_SEC", "IS_XR_SEC"},
        "key_kline_unique": price["duplicate_code_date"] == 0 and price["blank_key"] == 0,
        "key_status_unique": len(st_key) == len(st_rows),
        "key_equity_unique": eq_dupes == 0,
        "price_values_valid": not any(price[k] for k in ("nonfinite_ohlc", "nonpositive_ohlc", "high_range_violation", "low_range_violation", "nonfinite_volume_amount", "negative_volume_amount")),
        "status_coverage_raw": not missing,
        "equity_structure_complete": eq_codes == expected_set,
        "adj_factor_coverage": adj_codes == expected_set,
        "adj_factor_schema": len(adj_schema) == 5 and all(set(x) == set(expected_set & set(x)) for x in adj_schema),
        "adj_factor_keys_unique": adj_duplicate_dates == 0,
        "adj_factor_values_valid": adj_nonfinite == 0 and adj_nonpositive == 0,
        "pit_action_filter_recomputed": len(action_rows) > 0 and len(eligible_unique) <= len(action_rows),
        "status_retrieval_time_pit_revision": False,
        "training_authorized": False,
    }
    quality = {
        "schema": "wc006-v5-rebuilt-market-inputs/quality-report/v1",
        "status": "FAIL_CLOSED" if not all(checks.values()) else "PASS_WITH_TRAINING_AUTHORIZATION_FALSE",
        "manifest_sha256": rebuilt_manifest_hash,
        "training_authorized": False,
        "checks": checks,
        "coverage": {"expected_codes": len(expected_set), "kline_codes": len(kline), "equity_codes": len(eq_codes), "status_codes": len({str(r.get('MARKET_CODE', '')).strip().upper() for r in st_rows}), "missing_kline_codes": sorted(expected_set - set(kline)), "missing_equity_codes": sorted(expected_set - eq_codes), "missing_status_codes": sorted(expected_set - {str(r.get('MARKET_CODE', '')).strip().upper() for r in st_rows})},
        "rows": {"kline": kline_rows, "adj_factor": adj_rows, "status": len(st_rows), "status_overlap": len(overlap), "status_missing_for_raw": len(missing), "status_extra_not_in_raw": len(extra), "equity_structure": len(eq_rows), "actions_raw": len(action_rows), "actions_pit_eligible_unique": len(eligible_unique)},
        "schema": {"kline_variants": [list(x) for x in sorted(kline_schema)], "adj_factor_variants": [list(x) for x in sorted(adj_schema)], "status": list(status_schema), "equity": norm_fields(eq_fields)},
        "keys": {"kline_duplicate_code_date": price["duplicate_code_date"], "status_duplicate_code_date": len(st_rows) - len(st_key), "equity_duplicate_primary_key": eq_dupes},
        "price_qa": {"kline": {k: price[k] for k in ("blank_key", "duplicate_code_date", "nonfinite_ohlc", "nonpositive_ohlc", "high_range_violation", "low_range_violation", "nonfinite_volume_amount", "negative_volume_amount")}, "adj_factor": {"nonfinite_values": adj_nonfinite, "nonpositive_values": adj_nonpositive}},
        "qfq_raw_date_key_diff": {"scope": [TRAIN_BEGIN, TRAIN_END], "rows": len(diff_rows), "codes_with_diff": len({x['code'] for x in diff_rows}), "qfq_only": sum(x['side'] == 'qfq_only' for x in diff_rows), "raw_only": sum(x['side'] == 'raw_only' for x in diff_rows), "filled": False},
        "status_qa": {"state_counts": dict(status_state), "limit_counts": dict(limit_kind), "retrieval_time_field_present": False, "as_of_revision_field_present": False, "status_extra_keys_retained": len(extra)},
        "pit_corporate_actions": {"reason_counts": dict(action_counts), "raw_rows": len(action_rows), "eligible_unique": len(eligible_unique), "revision_conflicts": 0, "filter": "ANN_DATE <= effective_date and PROGRESS/DIV_PROGRESS == 3; no as-of revision history supplied"},
        "closed_items": ["125-code equity structure coverage restored", "125-code status artifact hash/row/schema/key coverage verified", "kline/adj_factor/dividend/right_issue source hashes reverified", "price finite/positive/range checks rerun", "corporate-action PIT eligibility filter rerun and retained raw rows"],
        "remaining_blockers": ["status API/source provides no as-of revision/PIT history evidence", "raw-vs-qfq date-key differences retained (no fill); see date_key_diff.csv", "training_authorized remains false; formal V5 training is not authorized"],
        "artifacts": {"date_key_diff": "date_key_diff.csv", "official_executable_state": "official_executable_state.csv", "pit_action_rows": "pit_action_rows.csv", "pit_action_eligible": "pit_action_eligible.csv", "retrieval_provenance": "retrieval_provenance.json", "manifest": "manifest.json"},
    }
    dump(out / "quality_report.json", quality)
    dump(out / "checks.json", {"status": quality["status"], "checks": checks})
    print(json.dumps({"output": str(out), "status": quality["status"], "manifest_sha256": rebuilt_manifest_hash, "checks": checks}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
