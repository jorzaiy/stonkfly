#!/usr/bin/env python3
"""Plan (but never perform) XYSZ collection for a purged candidate gap.

The planner is deliberately offline.  It reads candidate-stream manifests and
existing XYSZ parent/child manifests, then writes a new JSON query plan.  It
does not import AmazingData, log in, modify an old market bundle, or create an
authorization artifact.

The resulting plan is an input to a separately reviewed collector.  The
collector must write a *new* parent bundle and must run the existing QA before
any V5.1 replay is considered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


DATASETS = (
    "kline",
    "history_stock_status",
    "dividend",
    "right_issue",
    "equity_structure",
    "adj_factor",
)
PRICE_BEGIN = 20210101
PRICE_END = 20240103
ACTION_BEGIN = 20200101


class PlanBlocked(ValueError):
    """Raised when provenance is incomplete or ambiguous."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanBlocked(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PlanBlocked(f"JSON object required: {path}")
    return value


def code6(value: object) -> str:
    raw = str(value or "").strip().upper()
    raw = re.sub(r"\.(?:SH|SZ|BJ)$", "", raw)
    if not raw.isdigit() or len(raw) != 6:
        raise PlanBlocked(f"invalid stock code: {value!r}")
    return raw


def qualified(code: str) -> str:
    """Map a six-digit A-share code to the SDK MARKET_CODE form."""
    if code.startswith(("6", "68")):
        return f"{code}.SH"
    if code.startswith(("4", "8")):
        return f"{code}.BJ"
    return f"{code}.SZ"


def candidate_codes(path: Path) -> set[str]:
    payload = load_object(path)
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise PlanBlocked(f"candidate manifest has no records: {path}")
    found: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise PlanBlocked(f"candidate record is not an object: {path}")
        selector = record.get("selector")
        if not isinstance(selector, dict):
            raise PlanBlocked(f"candidate selector missing: {path}")
        rows = selector.get("target_topk")
        if not isinstance(rows, list) or not rows:
            raise PlanBlocked(f"candidate target_topk missing: {path}")
        for row in rows:
            if not isinstance(row, dict) or "code" not in row:
                raise PlanBlocked(f"candidate target row malformed: {path}")
            found.add(code6(row["code"]))
    return found


def _resolve_child(root: Path, raw: object) -> Path:
    value = Path(str(raw))
    path = value if value.is_absolute() else (root / value).resolve()
    if not path.is_file():
        raise PlanBlocked(f"child manifest missing: {path}")
    return path


def _artifact_codes(dataset: str, child: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for item in child.get("codes", []) or []:
        result.add(code6(item))
    for item in child.get("artifacts", []) or []:
        if isinstance(item, dict) and item.get("code"):
            result.add(code6(item["code"]))
        raw_path = str(item.get("path", "")) if isinstance(item, dict) else ""
        match = re.search(r"(?:kline_)?(\d{6})_(?:SH|SZ|BJ)(?:\.csv)?$", raw_path, re.I)
        if dataset == "kline" and match:
            result.add(match.group(1))
    return result


def market_coverage(path: Path) -> dict[str, set[str]]:
    payload = load_object(path)
    root = path.resolve().parent
    result = {dataset: set() for dataset in DATASETS}
    children = payload.get("child_manifests")
    if not isinstance(children, list) or not children:
        raise PlanBlocked(f"market manifest has no child_manifests: {path}")
    for item in children:
        if not isinstance(item, dict) or item.get("dataset") not in DATASETS:
            continue
        child_path = _resolve_child(root, item.get("fragment"))
        expected = item.get("manifest_sha256")
        if not isinstance(expected, str) or sha256(child_path) != expected:
            raise PlanBlocked(f"child manifest hash mismatch: {child_path}")
        child = load_object(child_path)
        dataset = str(item["dataset"])
        result[dataset].update(_artifact_codes(dataset, child))
    return result


def build_plan(candidate_manifests: Iterable[Path], market_manifests: Iterable[Path]) -> dict[str, Any]:
    candidates = [Path(item).resolve() for item in candidate_manifests]
    markets = [Path(item).resolve() for item in market_manifests]
    if not candidates or not markets:
        raise PlanBlocked("at least one candidate and one market manifest are required")
    target: set[str] = set()
    for path in candidates:
        target.update(candidate_codes(path))
    coverage = {dataset: set() for dataset in DATASETS}
    for path in markets:
        current = market_coverage(path)
        for dataset in DATASETS:
            coverage[dataset].update(current[dataset])

    missing = {dataset: sorted(target - coverage[dataset]) for dataset in DATASETS}
    qualified_target = [qualified(code) for code in sorted(target)]
    query_specs: dict[str, dict[str, Any]] = {
        "kline": {
            "client": "AmazingData.MarketData.query_kline",
            "codes": qualified_target,
            "begin_date": PRICE_BEGIN,
            "end_date": PRICE_END,
            "period": "day",
            "required_for": "raw execution, marks and rewards",
        },
        "history_stock_status": {
            "client": "AmazingData.InfoData.get_history_stock_status",
            "codes": qualified_target,
            "begin_date": PRICE_BEGIN,
            "end_date": PRICE_END,
            "required_for": "official suspension, ST and limit guards",
        },
        "adj_factor": {
            "client": "AmazingData.BaseData.get_adj_factor",
            "codes": qualified_target,
            "api_date_arguments": "not exposed by the pinned SDK; retain/filter rows for the required window",
            "required_window": ["2021-01-01", "2024-01-03"],
            "required_for": "hash-pinned corporate-action/factor evidence",
        },
        "dividend": {
            "client": "AmazingData.InfoData.get_dividend",
            "codes": qualified_target,
            "begin_date": ACTION_BEGIN,
            "end_date": PRICE_END,
            "required_for": "PIT cash/stock dividend settlement evidence",
        },
        "right_issue": {
            "client": "AmazingData.InfoData.get_right_issue",
            "codes": qualified_target,
            "begin_date": ACTION_BEGIN,
            "end_date": PRICE_END,
            "required_for": "PIT rights lifecycle evidence",
        },
        "equity_structure": {
            "client": "AmazingData.InfoData.get_equity_structure",
            "codes": qualified_target,
            "begin_date": ACTION_BEGIN,
            "end_date": PRICE_END,
            "required_for": "share-structure/corporate-action evidence",
        },
    }
    return {
        "schema": "wc006-purged-xysz-gap-plan/v1",
        "offline_plan_only": True,
        "network_collection_performed": False,
        "paper_only": True,
        "candidate_manifests": [{"path": str(p), "sha256": sha256(p)} for p in candidates],
        "existing_market_manifests": [{"path": str(p), "sha256": sha256(p)} for p in markets],
        "target_codes": sorted(target),
        "target_code_count": len(target),
        "target_codes_qualified": qualified_target,
        "coverage_by_dataset": {dataset: sorted(coverage[dataset]) for dataset in DATASETS},
        "missing_by_dataset": missing,
        "missing_code_count_by_dataset": {dataset: len(missing[dataset]) for dataset in DATASETS},
        "query_plan": query_specs,
        "window_policy": {
            "raw_price_and_status": ["2021-01-01", "2024-01-03"],
            "corporate_action_queries": ["2020-01-01", "2024-01-03"],
            "action_begin_rationale": "retain pre-2021 announcements that can become effective during the replay window",
        },
        "rebuild_policy": {
            "write_new_parent_only": True,
            "never_modify_existing_parent_or_authorization": True,
            "parent_source_manifest_codes": "exactly target_codes, with updated code_count",
            "child_fragment_paths": "relative to the new parent manifest; hash every child and raw artifact",
            "quality_report": "new quality_report.json; its sha256 is bound by the new authorization artifact",
            "authorization": "new development-only paper artifact required; old V5 authorization is not reusable",
        },
        "qa_requirements": [
            "every target code has raw kline rows covering signal/execution/settlement dates",
            "raw kline code-date keys are unique and OHLC are finite, positive and range-consistent",
            "history_stock_status covers every raw code-date key with required fields and valid flags",
            "adj_factor has every target code column, unique dates and finite positive values",
            "equity_structure covers every target code; duplicate primary keys are zero",
            "dividend/right_issue raw rows are retained and PIT eligibility is recomputed, never silently deduplicated",
            "all child manifests and artifacts are hash-verified before parent construction",
            "raw-vs-QFQ date-key differences are reported and never filled with QFQ rows",
        ],
        "status": "BLOCKED_PENDING_COLLECTION_AND_QA" if any(missing.values()) else "NO_GAP_FOUND_REQUIRES_FRESH_QA",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", action="append", required=True, type=Path)
    parser.add_argument("--market-manifest", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output_exists:{output}")
    try:
        plan = build_plan(args.candidate_manifest, args.market_manifest)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except PlanBlocked as exc:
        raise SystemExit(f"plan_blocked:{exc}") from exc
    print(json.dumps({"status": plan["status"], "output": str(output), "target_code_count": plan["target_code_count"], "missing_code_count_by_dataset": plan["missing_code_count_by_dataset"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
