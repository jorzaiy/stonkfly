"""Build an auditable parent view with an isolated equity supplement.

This builder never edits the original FAIL_CLOSED collection.  It binds the
original parent/QA and one separately collected equity fragment, then seals a
new research-only manifest only when every original non-equity check remains
true and all target codes have equity rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET_CODES = [
    "000678.SZ", "000753.SZ", "002094.SZ", "002232.SZ",
    "002246.SZ", "002545.SZ", "002730.SZ", "300175.SZ",
]
SCHEMA = "fly-purged-xysz-2024-2025-supplemented-market/v1"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_artifact(manifest_path: Path) -> tuple[dict, set[str], int]:
    manifest = load(manifest_path)
    if manifest.get("status") != "complete" or manifest.get("dataset") != "equity_structure":
        raise ValueError("supplement_manifest_scope_mismatch")
    codes = manifest.get("codes")
    if codes != TARGET_CODES:
        raise ValueError("supplement_code_list_mismatch")
    rows_total = 0
    observed: set[str] = set()
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise ValueError("supplement_artifact_count_mismatch")
    for item in artifacts:
        rel = Path(str(item.get("path", "")))
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("unsafe_supplement_artifact_path")
        artifact = manifest_path.parent / "raw" / rel
        if not artifact.is_file() or sha(artifact) != item.get("sha256"):
            raise ValueError("supplement_artifact_hash_mismatch")
        with artifact.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "MARKET_CODE" not in (reader.fieldnames or []):
                raise ValueError("supplement_market_code_missing")
            for row in reader:
                rows_total += 1
                observed.add(str(row.get("MARKET_CODE", "")).upper())
    if observed != set(TARGET_CODES):
        raise ValueError("supplement_row_code_coverage_mismatch")
    return manifest, observed, rows_total


def build(original: Path, supplement: Path, output: Path) -> dict:
    original = original.resolve()
    supplement = supplement.resolve()
    output = output.resolve()
    if output.exists():
        raise ValueError("output_already_exists")
    old_manifest_path = original / "manifest.json"
    old_qa_path = original / "quality_report.json"
    old_result_path = original / "collection_result.json"
    for path in (old_manifest_path, old_qa_path, old_result_path):
        if not path.is_file():
            raise ValueError(f"original_artifact_missing:{path.name}")
    old_manifest, old_qa = load(old_manifest_path), load(old_qa_path)
    if old_manifest.get("schema") != "fly-purged-xysz-market-2024-2025/v1":
        raise ValueError("original_manifest_schema_mismatch")
    if old_qa.get("status") != "FAIL_CLOSED":
        raise ValueError("original_qa_status_unexpected")
    if old_qa.get("issues") != ["equity_structure_code_coverage"]:
        raise ValueError("original_qa_issue_set_unexpected")
    checks = old_qa.get("checks", {})
    non_equity = {k: v for k, v in checks.items() if k != "equity_structure_code_coverage"}
    if not non_equity or not all(non_equity.values()):
        raise ValueError("original_non_equity_checks_not_all_pass")
    codes = old_manifest.get("source_manifest", {}).get("codes")
    if codes != sorted(codes) or len(codes) != 101:
        raise ValueError("original_target_code_set_mismatch")
    supplement_manifest_path = supplement / "equity_structure" / "batch_001" / "manifest.json"
    supplement_manifest, supplement_codes, supplement_rows = verify_artifact(supplement_manifest_path)
    # The original QA is the authoritative evidence for the eight missing codes;
    # derive the actual gap from the old raw rows before binding the supplement.
    old_equity_codes: set[str] = set()
    for child in old_manifest.get("child_manifests", []):
        if child.get("dataset") != "equity_structure":
            continue
        p = original / child["fragment"]
        payload = load(p)
        for item in payload.get("artifacts", []):
            with (p.parent / "raw" / item["path"]).open(newline="", encoding="utf-8") as f:
                old_equity_codes.update(str(r.get("MARKET_CODE", "")).upper() for r in csv.DictReader(f))
    gap = sorted(set(codes) - old_equity_codes)
    if gap != TARGET_CODES or supplement_codes != set(gap):
        raise ValueError("supplement_does_not_exactly_fill_original_gap")
    output.mkdir(parents=True)
    parent = {
        "schema": SCHEMA,
        "status": "PASS_RESEARCH_ONLY",
        "paper_only": True,
        "research_only": True,
        "broker_orders_enabled": False,
        "live_execution_authorized": False,
        "shadow_execution_authorized": False,
        "training_authorized": False,
        "authoritative_trading_state": True,
        "source_manifest": {"codes": codes, "code_count": len(codes)},
        "original_source": {
            "root": str(original),
            "manifest": str(old_manifest_path),
            "manifest_sha256": sha(old_manifest_path),
            "quality_report": str(old_qa_path),
            "quality_report_sha256": sha(old_qa_path),
            "collection_result_sha256": sha(old_result_path),
        },
        "supplement_source": {
            "root": str(supplement),
            "manifest": str(supplement_manifest_path),
            "manifest_sha256": sha(supplement_manifest_path),
            "artifact_sha256": supplement_manifest["artifacts"][0]["sha256"],
            "window": [19900101, 20260102],
            "codes": TARGET_CODES,
        },
        "candidate_manifest_sha256": old_manifest.get("candidate_manifest_sha256"),
        "child_manifest_count_original": len(old_manifest.get("child_manifests", [])),
        "supplement_rows": supplement_rows,
        "quality_report": "quality_report.json",
    }
    (output / "manifest.json").write_text(json.dumps(parent, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    qa = {
        "schema": SCHEMA + "/quality-report",
        "status": "PASS_RESEARCH_ONLY",
        "manifest_sha256": sha(output / "manifest.json"),
        "checks": {**non_equity, "equity_structure_code_coverage": True,
                    "original_non_equity_checks": True, "supplement_artifact_hash": True,
                    "supplement_exact_gap": True},
        "issues": [],
        "code_count": len(codes),
        "original_equity_code_count": len(old_equity_codes),
        "supplement_equity_code_count": len(supplement_codes),
        "equity_rows_supplement": supplement_rows,
        "original_qa_sha256": sha(old_qa_path),
        "supplement_manifest_sha256": sha(supplement_manifest_path),
        "paper_only": True,
    }
    (output / "quality_report.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return {"status": qa["status"], "manifest_sha256": qa["manifest_sha256"],
            "qa_sha256": sha(output / "quality_report.json"), "gap": gap,
            "supplement_rows": supplement_rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", type=Path, required=True)
    ap.add_argument("--supplement", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(build(args.original, args.supplement, args.output), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
