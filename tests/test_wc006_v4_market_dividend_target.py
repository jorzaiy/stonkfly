from __future__ import annotations

import csv
import hashlib
import json
from decimal import Decimal

import pytest

from experiments.a_share.wc006_v4_market import MarketInputsBlocked, _parse_actions


FIELDS = [
    "MARKET_CODE", "ANN_DATE", "DIV_PROGRESS", "DIV_TARGET",
    "DATE_EQY_RECORD", "DATE_EX", "DATE_DVD_PAYOUT", "LISTINGDATE_OF_DVD_SHR",
    "DVD_PER_SHARE_PRE_TAX_CASH", "DVD_PER_SHARE_STK", "DIV_BONUSRATE",
]


def _parse(tmp_path, targets):
    fragment = tmp_path / "dividend" / "batch_001"
    (fragment / "raw").mkdir(parents=True)
    artifact = fragment / "raw" / "dividend.csv"
    with artifact.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for target, stock in targets:
            writer.writerow({
                "MARKET_CODE": "000615.SZ", "ANN_DATE": "20251223", "DIV_PROGRESS": "3",
                "DIV_TARGET": target, "DATE_EQY_RECORD": "20251226", "DATE_EX": "20251229",
                "DATE_DVD_PAYOUT": "", "LISTINGDATE_OF_DVD_SHR": "20251229",
                "DVD_PER_SHARE_PRE_TAX_CASH": "0", "DVD_PER_SHARE_STK": stock, "DIV_BONUSRATE": "",
            })
    manifest = fragment / "manifest.json"
    manifest.write_text(json.dumps({
        "status": "complete", "dataset": "dividend",
        "artifacts": [{"path": "dividend.csv", "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}],
    }) + "\n")
    rights_fragment = tmp_path / "right_issue" / "batch_001"
    (rights_fragment / "raw").mkdir(parents=True)
    rights_artifact = rights_fragment / "raw" / "right_issue.csv"
    rights_artifact.write_text("MARKET_CODE,ANN_DATE,PROGRESS\n", encoding="utf-8")
    rights_manifest = rights_fragment / "manifest.json"
    rights_manifest.write_text(json.dumps({
        "status": "complete", "dataset": "right_issue",
        "artifacts": [{"path": "right_issue.csv", "sha256": hashlib.sha256(rights_artifact.read_bytes()).hexdigest()}],
    }) + "\n")
    payload = {"child_manifests": [
        {"dataset": "dividend", "fragment": "dividend/batch_001/manifest.json", "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()},
        {"dataset": "right_issue", "fragment": "right_issue/batch_001/manifest.json", "manifest_sha256": hashlib.sha256(rights_manifest.read_bytes()).hexdigest()},
    ]}
    return _parse_actions(tmp_path, payload, {})


def test_same_date_manager_row_is_excluded_from_ordinary_entitlement(tmp_path):
    actions = _parse(tmp_path, [("普通股股东", "0.1"), ("重整管理人", "3.91097858")])
    assert len(actions) == 1
    assert actions[0].stock_ratio == Decimal("0.1")


def test_unknown_nonempty_dividend_target_fails_closed(tmp_path):
    with pytest.raises(MarketInputsBlocked, match="unknown dividend target"):
        _parse(tmp_path, [("未知目标", "0.1")])
