import hashlib
import json
from pathlib import Path

import pytest

from tools.run_purged_xysz_collection import (
    COLLECTOR,
    DATASETS,
    DOCKER_IMAGE,
    DOCKER_IMAGE_DIGEST,
    LauncherBlocked,
    load_gap_plan,
    plan_collection,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan() -> dict:
    codes = ["000007", "600001"]
    qualified = ["000007.SZ", "600001.SH"]
    missing = {dataset: codes[:] for dataset in DATASETS}
    query = {}
    for dataset in DATASETS:
        query[dataset] = {"codes": qualified[:]}
    query["kline"].update({"begin_date": 20210101, "end_date": 20240103})
    query["history_stock_status"].update({"begin_date": 20210101, "end_date": 20240103})
    for dataset in ("dividend", "right_issue", "equity_structure"):
        query[dataset].update({"begin_date": 20200101, "end_date": 20240103})
    query["adj_factor"]["required_window"] = ["2021-01-01", "2024-01-03"]
    return {
        "schema": "wc006-purged-xysz-gap-plan/v1",
        "status": "BLOCKED_PENDING_COLLECTION_AND_QA",
        "offline_plan_only": True,
        "network_collection_performed": False,
        "paper_only": True,
        "target_codes": codes,
        "target_code_count": len(codes),
        "target_codes_qualified": qualified,
        "missing_by_dataset": missing,
        "missing_code_count_by_dataset": {dataset: len(codes) for dataset in DATASETS},
        "query_plan": query,
    }


def test_valid_plan_is_serial_and_does_not_create_output(tmp_path):
    plan_path = tmp_path / "gap-plan.json"
    plan_path.write_text(json.dumps(_plan()), encoding="utf-8")
    output = tmp_path / "new-collection"
    plan = load_gap_plan(plan_path, _sha(plan_path))
    spec = plan_collection(plan, output, Path("/run/secrets/not-read"))

    assert [item["dataset"] for item in spec["commands"]] == list(DATASETS)
    assert all(item["argv"][0:4] == ["sudo", "-n", "docker", "run"] for item in spec["commands"])
    assert all("--pull=never" in item["argv"] for item in spec["commands"])
    assert all(DOCKER_IMAGE_DIGEST in item["argv"] for item in spec["commands"])
    assert all(f"{DOCKER_IMAGE}@{DOCKER_IMAGE_DIGEST}" not in item["argv"] for item in spec["commands"])
    assert all(item["argv"][item["argv"].index("--dataset") + 1] == item["dataset"] for item in spec["commands"])
    assert all("hunter2" not in json.dumps(item["argv"]) for item in spec["commands"])
    assert not output.exists()


def test_plan_hash_and_blocked_status_are_fail_closed(tmp_path):
    plan_path = tmp_path / "gap-plan.json"
    payload = _plan()
    payload["status"] = "NO_GAP_FOUND_REQUIRES_FRESH_QA"
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LauncherBlocked, match="plan_must_be_blocked"):
        load_gap_plan(plan_path, _sha(plan_path))
    with pytest.raises(LauncherBlocked, match="plan_sha256_mismatch"):
        load_gap_plan(plan_path, "0" * 64)


def test_existing_output_and_collector_drift_are_rejected(tmp_path):
    plan_path = tmp_path / "gap-plan.json"
    plan_path.write_text(json.dumps(_plan()), encoding="utf-8")
    output = tmp_path / "already-there"
    output.mkdir()
    plan = load_gap_plan(plan_path, _sha(plan_path))
    with pytest.raises(LauncherBlocked, match="output_root_already_exists"):
        plan_collection(plan, output, Path("/run/secrets/xysz"))
    drifted = tmp_path / "collector.py"
    drifted.write_text(COLLECTOR.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8")
    with pytest.raises(LauncherBlocked, match="collector_hash_mismatch"):
        plan_collection(plan, tmp_path / "new-output", Path("/run/secrets/xysz"), collector=drifted)


def test_deterministic_batch_boundaries_for_79_and_110_code_gaps(tmp_path):
    payload = _plan()
    codes = [f"{value:06d}" for value in range(1, 111)]
    qualified = [f"{code}.SZ" for code in codes]
    payload["target_codes"] = codes
    payload["target_code_count"] = len(codes)
    payload["target_codes_qualified"] = qualified
    counts = {dataset: (110 if dataset == "right_issue" else 79) for dataset in DATASETS}
    payload["missing_by_dataset"] = {dataset: codes[:count] for dataset, count in counts.items()}
    payload["missing_code_count_by_dataset"] = counts
    for query in payload["query_plan"].values():
        query["codes"] = qualified
    plan_path = tmp_path / "gap-plan-79-110.json"
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    plan = load_gap_plan(plan_path, _sha(plan_path))
    spec = plan_collection(plan, tmp_path / "new-collection", Path("/run/secrets/xysz"))

    expected_batches = {
        "kline": (16, 5, 79),
        "history_stock_status": (16, 5, 79),
        "dividend": (4, 25, 79),
        "right_issue": (5, 25, 110),
        "equity_structure": (4, 25, 79),
        "adj_factor": (4, 25, 79),
    }
    assert {dataset: (item["batch_count"], item["batch_size"], len(item["codes"])) for dataset, item in spec["dataset_specs"].items()} == expected_batches
    assert len(spec["commands"]) == sum(item[0] for item in expected_batches.values())
    for dataset in DATASETS:
        rows = [item for item in spec["commands"] if item["dataset"] == dataset]
        assert [item["batch"] for item in rows] == list(range(1, expected_batches[dataset][0] + 1))
        assert rows[-1]["codes"][-1] == f"{expected_batches[dataset][2]:06d}.SZ" and rows[-1]["output"].endswith(f"{dataset}/batch_{expected_batches[dataset][0]:03d}")
    assert not (tmp_path / "new-collection").exists()
