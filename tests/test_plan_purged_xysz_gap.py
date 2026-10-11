import json
from pathlib import Path

import pytest

from tools.plan_purged_xysz_gap import PlanBlocked, build_plan, sha256


def _write_manifest(root: Path, name: str, dataset: str, codes: list[str]) -> Path:
    child_dir = root / name
    child_dir.mkdir(parents=True)
    payload = {
        "schema": "wc006-xysz-v4-dataset-fragment/v1",
        "status": "complete",
        "dataset": dataset,
        "codes": [f"{code}.SZ" for code in codes],
        "artifacts": [],
    }
    child = child_dir / "manifest.json"
    child.write_text(json.dumps(payload), encoding="utf-8")
    return child


def test_build_plan_unions_candidate_codes_and_reports_dataset_gaps(tmp_path):
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"records": [{"selector": {"target_topk": [{"code": "000007"}, {"code": "600001.SZ"}]}}]}), encoding="utf-8")
    child = _write_manifest(tmp_path, "kline", "kline", ["000007"])
    market = tmp_path / "market.json"
    market.write_text(json.dumps({"child_manifests": [{"dataset": "kline", "fragment": "kline/manifest.json", "manifest_sha256": sha256(child)}]}), encoding="utf-8")
    plan = build_plan([candidate], [market])
    assert plan["target_codes"] == ["000007", "600001"]
    assert plan["missing_by_dataset"]["kline"] == ["600001"]
    assert plan["missing_by_dataset"]["history_stock_status"] == ["000007", "600001"]
    assert plan["query_plan"]["dividend"]["begin_date"] == 20200101
    assert plan["query_plan"]["kline"]["end_date"] == 20240103


def test_plan_rejects_child_hash_drift(tmp_path):
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"records": [{"selector": {"target_topk": [{"code": "000007"}]}}]}), encoding="utf-8")
    child = _write_manifest(tmp_path, "kline", "kline", ["000007"])
    market = tmp_path / "market.json"
    market.write_text(json.dumps({"child_manifests": [{"dataset": "kline", "fragment": "kline/manifest.json", "manifest_sha256": "0" * 64}]}), encoding="utf-8")
    with pytest.raises(PlanBlocked, match="hash mismatch"):
        build_plan([candidate], [market])
