import hashlib
import json
import shutil
from pathlib import Path

import pytest

import experiments.a_share.build_wc006_v5_1_annual_slice as annual
from experiments.a_share.build_wc006_v5_1_annual_slice import AnnualSliceBlocked, build_annual_slice


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/wc006_v3_2021_inputs/20260918T-candidate-full-2021-2023-stateful"
SOURCE_MANIFEST = SOURCE / "manifest.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clone_source(tmp_path: Path) -> tuple[Path, dict]:
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "inputs").mkdir()
    for name in ("active_config.json", "universe.txt", "calendar.json"):
        shutil.copyfile(SOURCE / "inputs" / name, source_root / "inputs" / name)
    selected = [record for record in source_manifest["records"] if "2023-01-03" <= record["signal_date"] <= "2023-12-29"]
    codes = {row["code"] for record in selected for row in record["selector"]["target_topk"]}
    (source_root / "inputs/cache").mkdir()
    for code in codes:
        item = source_manifest["cache_files"][code]
        source = SOURCE / item["copied_path"]
        destination = source_root / item["copied_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    (source_root / "inputs/selector").mkdir()
    selected_hashes = {record["selector"]["artifact_sha256"] for record in selected}
    for item in source_manifest["selector_artifacts"]:
        if item["sha256"] not in selected_hashes:
            continue
        source = SOURCE / item["copied_path"]
        destination = source_root / item["copied_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    manifest_path = source_root / "manifest.json"
    manifest_path.write_text(json.dumps(source_manifest, indent=2) + "\n")
    return manifest_path, source_manifest


def test_real_manifest_builds_exact_2023_draft(tmp_path):
    output = tmp_path / "annual-2023"
    result = build_annual_slice(source_manifest=SOURCE_MANIFEST, output_dir=output)
    payload = json.loads(result.read_text())
    assert payload["draft"] is True
    assert payload["authorization_artifact"] is False
    assert payload["observation_window"] == ["2023-01-03", "2023-12-29"]
    assert payload["observation_count"] == 242
    assert payload["candidate_record_count"] == 17
    assert payload["records"][0]["signal_date"] == "2023-01-03"
    assert payload["records"][-1]["execution_date"] == "2024-01-02"
    calendar = json.loads((output / "inputs/calendar.json").read_text())
    assert len(calendar["dates"]) == 243
    assert calendar["dates"][-1] == "2024-01-02"
    assert _sha(output / "inputs/calendar.json") == payload["source_hashes"]["calendar"]
    assert not (output / "authorization.json").exists()


def test_source_manifest_hash_is_fail_closed(tmp_path, monkeypatch):
    manifest, _ = _clone_source(tmp_path)
    monkeypatch.setattr(annual, "EXPECTED_SOURCE_MANIFEST_SHA256", "0" * 64)
    with pytest.raises(AnnualSliceBlocked, match="source manifest hash"):
        build_annual_slice(source_manifest=manifest, output_dir=tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()


def test_source_record_count_is_fail_closed(tmp_path, monkeypatch):
    manifest, payload = _clone_source(tmp_path)
    payload["records"].pop()
    manifest.write_text(json.dumps(payload) + "\n")
    monkeypatch.setattr(annual, "EXPECTED_SOURCE_MANIFEST_SHA256", _sha(manifest))
    with pytest.raises(AnnualSliceBlocked, match="source record count"):
        build_annual_slice(source_manifest=manifest, output_dir=tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()


def test_annual_dates_and_settlement_boundary_are_fail_closed(tmp_path, monkeypatch):
    manifest, payload = _clone_source(tmp_path)
    payload["records"][-1]["execution_date"] = "2024-01-03"
    manifest.write_text(json.dumps(payload) + "\n")
    monkeypatch.setattr(annual, "EXPECTED_SOURCE_MANIFEST_SHA256", _sha(manifest))
    with pytest.raises(AnnualSliceBlocked, match="outside annual calendar"):
        build_annual_slice(source_manifest=manifest, output_dir=tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()
