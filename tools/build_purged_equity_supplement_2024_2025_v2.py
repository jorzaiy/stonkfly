"""Build a loader-compatible v2 parent for the isolated equity supplement."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

TARGET = {"000678.SZ", "000753.SZ", "002094.SZ", "002232.SZ", "002246.SZ", "002545.SZ", "002730.SZ", "300175.SZ"}
SCHEMA = "fly-purged-xysz-2024-2025-supplemented-market/v2"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_child(root: Path, rel: str) -> Path:
    p = root / rel
    rel_path = Path(rel)
    if rel_path.is_absolute() or ".." in rel_path.parts or not p.is_file():
        raise ValueError(f"unsafe_or_missing_child:{rel}")
    return p


def verify_children(root: Path, children: list[dict]) -> None:
    for child in children:
        path = safe_child(root, child["fragment"])
        if sha(path) != child.get("manifest_sha256"):
            raise ValueError(f"child_manifest_hash_mismatch:{child['fragment']}")
        payload = read(path)
        for item in payload.get("artifacts", []):
            rel = Path(str(item.get("path", "")))
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("unsafe_child_artifact")
            artifact = path.parent / "raw" / rel
            if not artifact.is_file() or sha(artifact) != item.get("sha256"):
                raise ValueError(f"child_artifact_hash_mismatch:{child['fragment']}:{rel}")


def supplement_check(path: Path) -> tuple[dict, set[str], int]:
    payload = read(path)
    if payload.get("dataset") != "equity_structure" or payload.get("status") != "complete":
        raise ValueError("supplement_scope_mismatch")
    if set(payload.get("codes", [])) != TARGET:
        raise ValueError("supplement_target_mismatch")
    codes, count = set(), 0
    for item in payload.get("artifacts", []):
        rel = Path(item["path"])
        artifact = path.parent / "raw" / rel
        if not artifact.is_file() or sha(artifact) != item.get("sha256"):
            raise ValueError("supplement_artifact_hash_mismatch")
        with artifact.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                count += 1
                codes.add(str(row.get("MARKET_CODE", "")).upper())
    if codes != TARGET:
        raise ValueError("supplement_row_coverage_mismatch")
    return payload, codes, count


def build(original: Path, supplement: Path, output: Path) -> dict:
    original, supplement, output = original.resolve(), supplement.resolve(), output.resolve()
    if output.exists():
        raise ValueError("output_already_exists")
    old_parent = original / "manifest.json"
    old_qa = original / "quality_report.json"
    old_result = original / "collection_result.json"
    parent, qa = read(old_parent), read(old_qa)
    if qa.get("status") != "FAIL_CLOSED" or qa.get("issues") != ["equity_structure_code_coverage"]:
        raise ValueError("original_qa_scope_mismatch")
    non_equity = {k: v for k, v in qa["checks"].items() if k != "equity_structure_code_coverage"}
    if not all(non_equity.values()):
        raise ValueError("original_non_equity_check_failed")
    original_children = parent.get("child_manifests", [])
    if len(original_children) != 62:
        raise ValueError("original_child_count_mismatch")
    # The symlink roots are inside the new parent and are constrained to these
    # two exact read-only data roots. No data is copied or rewritten.
    output.mkdir(parents=True)
    os.symlink(original, output / "original", target_is_directory=True)
    os.symlink(supplement, output / "supplement", target_is_directory=True)
    prefixed = [{**child, "fragment": f"original/{child['fragment']}"} for child in original_children]
    verify_children(output, prefixed)
    supplement_path = output / "supplement/equity_structure/batch_001/manifest.json"
    supp, supp_codes, supp_rows = supplement_check(supplement_path)
    children = prefixed + [{"dataset": "equity_structure", "fragment": "supplement/equity_structure/batch_001/manifest.json", "manifest_sha256": sha(supplement_path)}]
    if len(children) != 63:
        raise ValueError("combined_child_count_mismatch")
    source_codes = parent.get("source_manifest", {}).get("codes", [])
    if len(source_codes) != 101 or not set(source_codes):
        raise ValueError("source_code_set_mismatch")
    original_equity = set()
    for child in original_children:
        if child["dataset"] != "equity_structure":
            continue
        p = safe_child(output, f"original/{child['fragment']}")
        for item in read(p).get("artifacts", []):
            with (p.parent / "raw" / item["path"]).open(newline="", encoding="utf-8") as f:
                original_equity.update(str(r.get("MARKET_CODE", "")).upper() for r in csv.DictReader(f))
    if set(source_codes) - original_equity != TARGET:
        raise ValueError("supplement_does_not_fill_exact_gap")
    manifest = {
        "schema": SCHEMA, "status": "PASS_RESEARCH_ONLY", "paper_only": True,
        "research_only": True, "broker_orders_enabled": False,
        "live_execution_authorized": False, "shadow_execution_authorized": False,
        "training_authorized": False, "authoritative_trading_state": True,
        "source_manifest": {"codes": source_codes, "code_count": len(source_codes)},
        "candidate_manifest_sha256": parent.get("candidate_manifest_sha256"),
        "original_source": {"root": str(original), "manifest_sha256": sha(old_parent), "quality_report_sha256": sha(old_qa), "collection_result_sha256": sha(old_result)},
        "supplement_source": {"root": str(supplement), "manifest_sha256": sha(supplement_path), "artifact_sha256": supp["artifacts"][0]["sha256"], "codes": sorted(TARGET), "window": [19900101, 20260102]},
        "child_manifests": children, "quality_report": "quality_report.json",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    report = {"schema": SCHEMA + "/quality-report", "status": "PASS_RESEARCH_ONLY", "manifest_sha256": sha(output / "manifest.json"), "checks": {**non_equity, "equity_structure_code_coverage": True, "all_child_manifest_hashes": True, "all_child_artifact_hashes": True, "supplement_exact_gap": True}, "issues": [], "code_count": len(source_codes), "original_equity_code_count": len(original_equity), "supplement_equity_code_count": len(supp_codes), "supplement_equity_rows": supp_rows, "paper_only": True}
    (output / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return {"status": report["status"], "manifest_sha256": report["manifest_sha256"], "qa_sha256": sha(output / "quality_report.json"), "child_count": len(children), "supplement_rows": supp_rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", type=Path, required=True)
    ap.add_argument("--supplement", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(build(args.original, args.supplement, args.output), ensure_ascii=False))
