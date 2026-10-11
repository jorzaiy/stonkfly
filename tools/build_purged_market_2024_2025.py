#!/usr/bin/env python3
"""Audit raw 2024–2025 XYSZ fragments and seal a research-only market bundle."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

from tools.collect_purged_xysz_2024_2025 import (
    BEGIN, END, SCHEMA, CollectionBlocked, plan, sha, verify_fragment, write_json,
)
from tools.run_purged_xysz_collection import DATASET_BATCH_SIZE, DATASETS

ROOT = Path(__file__).resolve().parents[1]
MARKET_SCHEMA = "fly-purged-xysz-market-2024-2025/v1"


def date(value: object) -> str:
    s = str(value or "").strip().replace("-", "")
    if s.endswith(".0"):
        s = s[:-2]
    if len(s) != 8 or not s.isdigit():
        raise CollectionBlocked("invalid_market_date")
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def positive(value: object, *, allow_zero: bool = False) -> bool:
    try:
        d = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return False
    return d.is_finite() and (d >= 0 if allow_zero else d > 0)


def rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def build(candidate_path: Path, root: Path) -> dict:
    root = root.resolve()
    spec = plan(candidate_path)
    plan_path = root / "collection_plan.json"
    result_path = root / "collection_result.json"
    if not plan_path.is_file() or json.loads(plan_path.read_text()) != spec:
        raise CollectionBlocked("collection_plan_mismatch")
    result = json.loads(result_path.read_text())
    if result.get("schema") != SCHEMA or result.get("plan_sha256") != sha(plan_path):
        raise CollectionBlocked("collection_result_plan_mismatch")
    fragments = []
    for dataset in DATASETS:
        size = DATASET_BATCH_SIZE[dataset]
        for index, start in enumerate(range(0, len(spec["codes"]), size), 1):
            child = root / dataset / f"batch_{index:03d}"
            item = verify_fragment(child, dataset, spec["codes"][start:start + size])
            if item not in result.get("fragments", []):
                raise CollectionBlocked("collection_result_fragment_mismatch")
            fragments.append(item)
    if len(result.get("fragments", [])) != len(fragments):
        raise CollectionBlocked("collection_result_fragment_count_mismatch")
    if (root / "manifest.json").exists() or (root / "quality_report.json").exists():
        raise CollectionBlocked("sealed_market_already_exists")

    codes = set(spec["codes"])
    prices: dict[str, set[str]] = defaultdict(set)
    status: dict[str, set[str]] = defaultdict(set)
    action_rows = defaultdict(int)
    equity_codes: set[str] = set()
    adj_coverage: set[str] = set()
    adj_price_keys: dict[str, set[str]] = defaultdict(set)
    issues: list[str] = []
    for item in fragments:
        dataset = item["dataset"]
        child_manifest = Path(item["manifest"])
        payload = json.loads(child_manifest.read_text())
        for artifact in payload["artifacts"]:
            artifact_path = child_manifest.parent / "raw" / artifact["path"]
            if dataset == "adj_factor":
                with artifact_path.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    fields = list(reader.fieldnames or [])
                    if fields:
                        for row in reader:
                            d = date(row.get(fields[0]))
                            for code in payload["codes"]:
                                if d in prices[code]:
                                    if positive(row.get(code)):
                                        adj_price_keys[code].add(d)
                                    else:
                                        issues.append(f"adj_factor_missing_or_invalid:{code}:{d}")
                data = []
            else:
                fields, data = rows(artifact_path)
            if dataset == "kline":
                code = artifact["code"]
                if code not in codes or not data:
                    issues.append(f"kline_missing_or_extra:{code}")
                for row in data:
                    d = date(row.get("kline_time"))
                    if not "2024-01-01" <= d <= "2026-01-02" or d in prices[code]:
                        issues.append(f"kline_date_or_duplicate:{code}:{d}")
                    prices[code].add(d)
                    if not all(positive(row.get(x)) for x in ("open", "high", "low", "close")) or not all(positive(row.get(x), allow_zero=True) for x in ("volume", "amount")):
                        issues.append(f"kline_numeric:{code}:{d}")
                    else:
                        o, h, l, c = (Decimal(row[x]) for x in ("open", "high", "low", "close"))
                        if h < max(o, c, l) or l > min(o, c, h):
                            issues.append(f"kline_ohlc:{code}:{d}")
            elif dataset == "history_stock_status":
                required = {"MARKET_CODE", "TRADE_DATE", "PRECLOSE", "HIGH_LIMITED", "LOW_LIMITED", "IS_ST_SEC", "IS_SUSP_SEC", "IS_WD_SEC", "IS_XR_SEC"}
                if not required <= set(fields):
                    issues.append("status_fields_missing")
                for row in data:
                    code = str(row.get("MARKET_CODE", "")).upper()
                    d = date(row.get("TRADE_DATE"))
                    if code not in codes or d in status[code] or not "2024-01-01" <= d <= "2026-01-02":
                        issues.append(f"status_code_date_or_duplicate:{code}:{d}")
                    status[code].add(d)
            elif dataset == "adj_factor":
                if not fields:
                    issues.append("adj_factor_fields_missing")
                else:
                    adj_coverage.update(set(fields[1:]) & codes)
                    if not set(payload["codes"]) <= set(fields[1:]):
                        issues.append("adj_factor_code_columns_missing")
            else:
                action_rows[dataset] += len(data)
                if dataset == "equity_structure":
                    equity_codes.update(str(row.get("MARKET_CODE", "")).upper() for row in data)
                if data and "MARKET_CODE" in fields and any(str(row.get("MARKET_CODE", "")).upper() not in codes for row in data):
                    issues.append(f"{dataset}_code_outside_target")
    if set(prices) != codes:
        issues.append("kline_code_coverage")
    if set(status) != codes:
        issues.append("status_code_coverage")
    if adj_coverage != codes:
        issues.append("adj_factor_code_coverage")
    if equity_codes != codes:
        issues.append("equity_structure_code_coverage")
    if any(adj_price_keys[code] != prices[code] for code in codes):
        issues.append("adj_factor_kline_key_coverage")
    for code in codes:
        if not prices[code] <= status[code]:
            issues.append(f"status_missing_kline_dates:{code}")
    candidate = json.loads(Path(candidate_path).read_text())
    for record in candidate["records"]:
        signal, execution = record["signal_date"], record["execution_date"]
        for target in record["selector"]["target_topk"]:
            code = next(x for x in codes if x.startswith(target["code"] + "."))
            if not any(d <= signal for d in prices[code]):
                issues.append(f"no_causal_mark_at_signal:{code}:{signal}")
            if execution not in status[code]:
                issues.append(f"no_official_status_at_execution:{code}:{execution}")
    if not all(any(d <= "2025-12-30" for d in prices[code]) and "2025-12-30" in status[code] for code in codes):
        issues.append("settlement_causal_mark_and_status_coverage")
    checks = {"all_fragment_hashes": True, "code_coverage": set(prices) == codes and set(status) == codes,
              "kline_status_key_coverage": all(prices[code] <= status[code] for code in codes),
              "adj_factor_code_coverage": adj_coverage == codes,
              "equity_structure_code_coverage": equity_codes == codes,
              "adj_factor_kline_key_coverage": all(adj_price_keys[code] == prices[code] for code in codes),
              "candidate_signal_execution_coverage": not any(x.startswith(("no_causal_mark", "no_official_status")) for x in issues),
              "settlement_causal_mark_and_status_coverage": "settlement_causal_mark_and_status_coverage" not in issues,
              "raw_row_validity": not any(x.startswith(("kline_numeric", "kline_ohlc", "kline_date", "status_code")) for x in issues)}
    status_label = "PASS_RESEARCH_ONLY" if not issues and all(checks.values()) else "FAIL_CLOSED"
    children = [{"dataset": item["dataset"], "fragment": str(Path(item["manifest"]).relative_to(root)),
                 "manifest_sha256": item["manifest_sha256"]} for item in fragments]
    parent = {"schema": MARKET_SCHEMA, "status": status_label, "paper_only": True,
              "research_only": True, "broker_orders_enabled": False, "live_execution_authorized": False,
              "shadow_execution_authorized": False, "training_authorized": False,
              "authoritative_trading_state": True,
              "candidate_manifest_sha256": spec["candidate_manifest_sha256"],
              "collection_plan_sha256": sha(plan_path), "collection_result_sha256": sha(result_path),
              "source_manifest": {"codes": sorted(codes), "code_count": len(codes)},
              "child_manifests": children, "quality_report": "quality_report.json"}
    write_json(root / "manifest.json", parent)
    qa = {"schema": MARKET_SCHEMA + "/quality-report", "status": status_label,
          "manifest_sha256": sha(root / "manifest.json"), "checks": checks,
          "issues": sorted(set(issues)), "code_count": len(codes),
          "row_counts": {"kline": sum(map(len, prices.values())), "history_stock_status": sum(map(len, status.values())), **dict(action_rows)}}
    write_json(root / "quality_report.json", qa)
    return {"status": status_label, "manifest_sha256": qa["manifest_sha256"],
            "qa_sha256": sha(root / "quality_report.json"), "issues": qa["issues"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--collection-root", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.candidate, args.collection_root)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "PASS_RESEARCH_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
