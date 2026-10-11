"""Seal v3 market inputs with the frozen dividend target policy evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from tools.build_purged_equity_supplement_2024_2025_v2 import build as build_v2

ROOT = Path(__file__).resolve().parents[1]
PARSER = ROOT / "experiments/a_share/wc006_v4_market.py"
MANAGER_TARGET = "重整管理人"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def excluded_rows(root: Path, children: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for child in children:
        if child.get("dataset") != "dividend":
            continue
        manifest_path = root / child["fragment"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest.get("artifacts", []):
            path = manifest_path.parent / "raw" / item["path"]
            with path.open("rb") as handle:
                lines = handle.readlines()
            if not lines:
                continue
            fields = lines[0].decode("utf-8").rstrip("\r\n").split(",")
            target_index = fields.index("DIV_TARGET")
            for line in lines[1:]:
                decoded = next(csv.reader([line.decode("utf-8").rstrip("\r\n")]))
                if len(decoded) > target_index and decoded[target_index].strip() == MANAGER_TARGET:
                    rows.append({"artifact": str(path.relative_to(root)), "row_sha256": hashlib.sha256(line).hexdigest()})
    return rows


def build(original: Path, supplement: Path, output: Path) -> dict:
    result = build_v2(original, supplement, output)
    output = output.resolve()
    manifest_path, qa_path = output / "manifest.json", output / "quality_report.json"
    manifest, qa = json.loads(manifest_path.read_text()), json.loads(qa_path.read_text())
    excluded = excluded_rows(output, manifest["child_manifests"])
    if len(excluded) != 1:
        raise ValueError(f"expected_one_reorganization_manager_row:{len(excluded)}")
    manifest["dividend_target_policy"] = {
        "ordinary_target": "普通股股东",
        "excluded_target": MANAGER_TARGET,
        "unknown_nonempty_target": "FAIL_CLOSED",
        "empty_target": "legacy_compatible",
        "parser_source": str(PARSER),
        "parser_source_sha256": sha(PARSER),
        "excluded_source_rows": excluded,
        "excluded_row_count": len(excluded),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    qa["manifest_sha256"] = sha(manifest_path)
    qa["checks"].update({"dividend_target_policy": True, "excluded_source_row_hash": True, "parser_source_hash": True})
    qa["dividend_target_policy"] = manifest["dividend_target_policy"]
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    result.update({"manifest_sha256": sha(manifest_path), "qa_sha256": sha(qa_path), "excluded_rows": excluded, "parser_source_sha256": sha(PARSER)})
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", type=Path, required=True)
    ap.add_argument("--supplement", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(build(args.original, args.supplement, args.output), ensure_ascii=False))
