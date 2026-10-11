#!/usr/bin/env python3
"""Create the immutable, development-only V5 market authorization bundle.

The old C-stage parent and the rebuilt QA bundle are never modified.  This
script creates a new loader-compatible parent manifest which references the
immutable C-stage children plus the completed bridge equity/status children.
The only accepted failed QA condition is the documented absence of supplier
as-of/revision evidence; it is recorded as a limitation, not hidden.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "runs/wc006_v5_market_inputs/manual-bridge"
QA_DIR = BRIDGE / "rebuilt-qa-20260919T144824Z"
QFQ_DIR = ROOT / "runs/wc006_v3_2021_inputs/20260918T-candidate-full-2021-2023-stateful/inputs/cache"
CSTAGE = ROOT / "runs/wc006_v4_xysz_full/20260919T054212Z-125codes"
PREP = BRIDGE / "prepared"
DEFAULT_OUT = BRIDGE / "authorized-v5-development-20260919T194757Z-controller-lock"
PREREG = ROOT / "docs/experiments/wc006_v5_preregister.md"
V5_ADAPTER = ROOT / "experiments/a_share/wc006_male_cns_v5_adapter.py"
V4_ADAPTER = ROOT / "experiments/a_share/wc006_male_cns_v4_adapter.py"
CONTROLLER = ROOT / "stonkfly/neural/controller.py"
FIXED_SEED = 17
FIXED_ARMS = ["attribution-only", "state-only", "combined"]
PREREG_SHA256 = "6ab3f8d024a798371abb48b40ac751f2cea6038cf011bcbf26173d2c82563670"
GRAPH_SOURCE_LOCK = {
    "full_retained_graph": True,
    "release": "MaleCNS v1.0",
    "selection": "All existing KC-to-MBON07/11 connections; no graph cropping or added edges.",
    "configuration_sha256": "70b253b10eaecd2492ced3b143b3b391e10b49b8a62ced7a0ec105930b68cd7f",
    "plastic_edges": 7835,
    "plastic_edges_sha256": "153a1a6192bc7056b90bbfc757b56cb9580ca786a187583cc2cc652e1e3abe8f",
    "source_hashes": {
        "arrays.lock.json": "39de6a2e46be552b734b822c8452e07ed05b5e8af906c5f79db7d63c2d68515e",
        "datasets.json": "4532ef8b2f08d819cb31dee3a40f21685112971a05697cf0fa7fa634cf9035b8",
        "neurons.lock.json": "89ab067db991c674f1fbe4fcaf2d8d5dabab9337e8cb6a24270ad8e8b5c582c7",
        "sources.lock.json": "a518a766aa46b14b7b9230a108a63b902acc1e3386f123f45d5c52311342efd8",
    },
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rel(path: Path, output: Path) -> str:
    return os.path.relpath(path, output).replace(os.sep, "/")


def canonical_v5_adapter_sha256() -> str:
    text = V5_ADAPTER.read_text(encoding="utf-8")
    marker = 'V5_FORMAL_AUTHORIZATION_SHA256 = '
    lines = []
    for line in text.splitlines(keepends=True):
        if line.startswith(marker):
            line = marker + '"' + ('0' * 64) + '"\n'
        lines.append(line)
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def source_hashes() -> dict[str, str]:
    paths: set[Path] = set()
    for path in QA_DIR.iterdir():
        if path.is_file():
            paths.add(path)
    for path in (BRIDGE / "qa_analysis").iterdir():
        if path.is_file() and path.suffix in {".csv", ".json"}:
            paths.add(path)
    for path in (
        PREP / "manifest.json",
        PREP / "manifest.json.sha256",
        PREP / "source_inventory.json",
        PREP / "qa.json",
        PREP / "codes_125.txt",
        PREP / "codes_125.txt.sha256",
        PREP / "raw/equity_structure.csv",
        PREP / "history_stock_status_125_attempt2/manifest.json",
        PREP / "history_stock_status_125_attempt2/raw/history_stock_status.csv",
        CSTAGE / "manifest.json",
    ):
        paths.add(path)
    for child in json.loads((CSTAGE / "manifest.json").read_text(encoding="utf-8")).get("child_manifests", []):
        manifest = CSTAGE / str(child["fragment"])
        paths.add(manifest)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        for item in payload.get("artifacts", []):
            direct = manifest.parent / str(item["path"])
            raw = manifest.parent / "raw" / str(item["path"])
            paths.add(direct if direct.is_file() else raw)
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(paths) if path.is_file()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="new sibling authorization directory; existing directories are never overwritten",
    )
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if output.parent != BRIDGE.resolve():
        raise RuntimeError(f"output_must_be_manual_bridge_sibling:{output}")
    if output.exists():
        raise RuntimeError(f"output_exists:{output}")
    created_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    v4_adapter_sha256 = sha(V4_ADAPTER)
    old_quality = json.loads((QA_DIR / "quality_report.json").read_text(encoding="utf-8"))
    old_checks = dict(old_quality["checks"])
    accepted_checks = {"status_retrieval_time_pit_revision", "training_authorized"}
    failed = {name for name, passed in old_checks.items() if passed is not True}
    if failed != accepted_checks:
        raise RuntimeError(f"unexpected_failed_checks:{sorted(failed)}")

    qfq = json.loads((BRIDGE / "qa_analysis/qfq_gap_summary.json").read_text(encoding="utf-8"))
    if qfq["qfq_semantics"]["bounded_by_raw_bars"] != 162 or qfq["qfq_semantics"]["close_same_as_previous_qfq"] != 162:
        raise RuntimeError("qfq_gap_not_fully_bounded")
    if qfq["qfq_semantics"]["delisting_boundary_evidence"] != 0:
        raise RuntimeError("qfq_gap_has_delisting_boundary_evidence")
    if qfq["status_classification"]["suspended"] != 162:
        raise RuntimeError("qfq_gap_not_classified_as_suspension")
    if qfq["usage"]["execution_fill_occurrences"] != 0 or qfq["usage"]["qfq_mark_source_occurrences"] != 0:
        raise RuntimeError("qfq_gap_used_as_execution_or_mark_source")

    output.mkdir(parents=True)
    old_parent_path = CSTAGE / "manifest.json"
    old_parent = json.loads(old_parent_path.read_text(encoding="utf-8"))
    parent = copy.deepcopy(old_parent)
    children = []
    for child in old_parent.get("child_manifests", []):
        # The rebuilt bridge supplies the complete status/equity datasets.
        # Do not append them to the old C-stage fragments, which would create
        # duplicate code-date keys in the loader.
        if child.get("dataset") in {"equity_structure", "history_stock_status"}:
            continue
        item = dict(child)
        item["fragment"] = rel(CSTAGE / str(child["fragment"]), output)
        children.append(item)
    bridge_children = [
        ("equity_structure", PREP / "manifest.json"),
        ("history_stock_status", PREP / "history_stock_status_125_attempt2/manifest.json"),
    ]
    for dataset, path in bridge_children:
        payload = json.loads(path.read_text(encoding="utf-8"))
        children.append({
            "dataset": dataset,
            "fragment": rel(path, output),
            "manifest_sha256": sha(path),
            "status": payload.get("status"),
            "exit_mode": payload.get("exit_mode"),
        })
    parent.update({
        "schema": "wc006-v5-authorized-development-market-inputs/v1",
        "status": "complete_with_accepted_development_limitation",
        "created_at_utc": created_at_utc,
        "child_manifests": children,
        "quality_report": "quality_report.json",
        "authorization_artifact": "authorization.json",
        "authoritative_trading_state": True,
        "paper_only": True,
        "broker_orders_enabled": False,
        "live_execution_authorized": False,
        "training_authorized": True,
        "development_only": True,
        "input_data_window": ["2021-01-01", "2023-12-29"],
        "training_window": ["2021-07-01", "2023-12-29"],
        "source_parent_manifest": {"path": str(old_parent_path.relative_to(ROOT)), "sha256": sha(old_parent_path)},
        "limitations": [
            "Development-only paper training for 2021-2023; not strict PIT.",
            "The status source has no retrieval-time/as-of revision history; supplier revisions cannot be ruled out.",
            "The 162 qfq-only keys are retained as a documented observation semantic split on official suspension dates; no raw bars were filled and qfq is never used for execution, marks, or rewards.",
        ],
        "prohibitions": [
            "No 2024 validation or 2025 exam may use this authorization.",
            "No live execution or broker orders.",
            "No claim of strict PIT, profitable learning, or validated profitability.",
        ],
    })
    parent["datasets"] = dict(parent.get("datasets", {}))
    parent["datasets"]["equity_structure"] = {
        "artifact_count": 1, "codes_seen": 125, "rows": 1742,
        "missing_codes": [], "duplicate_key_rows": 0,
        "source": "prepared/raw/equity_structure.csv",
    }
    parent["datasets"]["history_stock_status"] = {
        "artifact_count": 1, "codes_seen": 125, "rows": 89534,
        "missing_codes": [], "duplicate_code_date_rows": 0,
        "source": "prepared/history_stock_status_125_attempt2/raw/history_stock_status.csv",
    }
    dump(output / "manifest.json", parent)
    parent_hash = sha(output / "manifest.json")

    quality = copy.deepcopy(old_quality)
    quality.update({
        "schema": "wc006-v5-authorized-development-market-inputs/quality-report/v1",
        "status": "PASS_WITH_ACCEPTED_DEVELOPMENT_LIMITATION",
        "training_authorized": True,
        "manifest_sha256": parent_hash,
        "checks": {**old_checks, "training_authorized": True},
        "remaining_blockers": [],
        "accepted_failed_checks": ["status_retrieval_time_pit_revision"],
        "accepted_limitations": [
            "No status retrieval-time/as-of revision field or PIT revision history was supplied by the API/source.",
            "The qfq/raw date-key split is accepted only as a documented suspension semantic split; no rows are filled or dropped.",
        ],
        "authorization_scope": {
            "development_only": True,
            "paper_only": True,
            "input_data_window": ["2021-01-01", "2023-12-29"],
            "training_window": ["2021-07-01", "2023-12-29"],
            "validation_2024": False,
            "exam_2025": False,
            "live_execution": False,
        },
    })
    dump(output / "quality_report.json", quality)
    quality_hash = sha(output / "quality_report.json")

    hashes = source_hashes()
    hashes.update({
        "authorized_v5_development/manifest.json": parent_hash,
        "authorized_v5_development/quality_report.json": quality_hash,
    })
    authorization = {
        "schema": "wc006-v5-development-authorization/v1",
        "status": "APPROVED_DEVELOPMENT_ONLY",
        "training_authorized": True,
        "paper_only": True,
        "broker_orders_enabled": False,
        "live_execution_authorized": False,
        "review": {
            "decision": "USER_ACCEPTED_DEVELOPMENT_LIMITATION",
            "recorded_at_utc": "2026-09-19T14:48:24Z",
            "basis": "User accepted the non-strict-PIT limitation for development-only paper training.",
            "review_scope": "Only verifiable data-integrity gates passed; the missing status as-of/revision evidence remains an explicit limitation.",
        },
        "scope": {
            "development_only": True,
            "input_data_window": ["2021-01-01", "2023-12-29"],
            "training_window": ["2021-07-01", "2023-12-29"],
            "allowed_use": "V5 clean-brain paper training and mechanism/audit evaluation",
            "validation_2024": False,
            "exam_2025": False,
            "live_execution": False,
            "strict_pit_claim": False,
            "profitability_claim": False,
        },
        "market_manifest": {"path": "manifest.json", "sha256": parent_hash},
        "code_lock": {
            "v5_preregister_sha256": PREREG_SHA256,
            "v5_adapter_sha256": canonical_v5_adapter_sha256(),
            "v4_adapter_sha256": v4_adapter_sha256,
            "controller_sha256": sha(CONTROLLER),
            "graph_source_lock": GRAPH_SOURCE_LOCK,
        },
        "fixed_run_plan": {
            "seed": FIXED_SEED,
            "arms_in_order": FIXED_ARMS,
            "input_data_window": ["2021-01-01", "2023-12-29"],
            "training_window": ["2021-07-01", "2023-12-29"],
        },
        "quality_report": {"path": "quality_report.json", "sha256": quality_hash, "source_rebuilt_qa_sha256": sha(QA_DIR / "quality_report.json")},
        "qfq_raw_disposition": {
            "source": "../qa_analysis/qfq_gap_summary.json",
            "source_sha256": sha(BRIDGE / "qa_analysis/qfq_gap_summary.json"),
            "rows": 162,
            "classification": "all bounded suspension dates; 88 suspension-only and 74 suspension+ST",
            "formal_observation_lookback_unique_keys": 27,
            "raw_fill": False,
            "qfq_execution_mark_reward": False,
        },
        "status_pit_limitation": {
            "retrieval_time_field_present": False,
            "as_of_revision_field_present": False,
            "accepted_as": "non-strict-PIT development limitation",
        },
        "hashes": hashes,
        "self_hash_excluded": "This JSON records every source and pinned output hash; a self-hash would be circular.",
    }
    dump(output / "authorization.json", authorization)
    print(json.dumps({
        "output": str(output),
        "manifest": parent_hash,
        "quality_report": quality_hash,
        "authorization": sha(output / "authorization.json"),
        "training_authorized": True,
        "status": quality["status"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
