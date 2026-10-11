import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from experiments.a_share.wc006_male_cns_adapter import (
    AdapterBlocked,
    FlyController,
    run_dry_fixture,
    validate_candidate_stream,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _make_stream(tmp_path):
    root = tmp_path / "candidate"
    inputs = root / "inputs"
    (inputs / "cache").mkdir(parents=True)
    (inputs / "selector").mkdir(parents=True)
    dates = ["2021-07-01", "2021-07-02", "2021-07-05", "2021-07-06"]
    (inputs / "calendar.json").write_text(json.dumps({"dates": dates}))
    (inputs / "active_config.json").write_text(json.dumps({"walkforward": {"signal_lag_one_day": True}}))
    (inputs / "universe.txt").write_text("000001\n000002\n000003\n000004\n")
    codes = ["000001", "000002", "000003", "000004"]
    cache = {}
    for offset, code in enumerate(codes):
        path = inputs / "cache" / f"{code}_tag.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["date", "open", "high", "low", "close"])
            writer.writeheader()
            for index, date in enumerate(dates):
                price = str(10 + offset + index)
                writer.writerow({"date": date, "open": price, "high": price, "low": price, "close": price})
        cache[code] = {"copied_path": str(path.relative_to(root)), "sha256": _sha(path)}
    target_sets = [
        [("000001", 0.4), ("000002", 0.3), ("000003", 0.3)],
        [("000002", 0.4), ("000003", 0.3), ("000004", 0.3)],
        [("000001", 0.4), ("000003", 0.3), ("000004", 0.3)],
    ]
    records = []
    artifacts = []
    for index, (signal, execution, targets) in enumerate(zip(dates[:3], dates[1:], target_sets)):
        artifact = inputs / "selector" / f"{index:03d}.json"
        artifact.write_text(json.dumps({
            "source": "rdlab_leading_rebalance",
            "rebalance_date": signal,
            "rebalance_applied": True,
            "topk": 3,
            "hold": 20,
            "prefer_tag": "tag",
            "effective_target_topk": [{"code": code, "score": 1.0 - pos / 10, "target_weight": weight} for pos, (code, weight) in enumerate(targets)],
        }))
        target_rows = [{"code": code, "score": 1.0 - pos / 10, "target_weight": weight} for pos, (code, weight) in enumerate(targets)]
        artifacts.append({"source_name": artifact.name, "sha256": _sha(artifact), "copied_path": str(artifact.relative_to(root))})
        records.append({
            "signal_date": signal,
            "execution_date": execution,
            "selector": {
                "artifact_sha256": _sha(artifact),
                "prefer_tag": "tag",
                "rebalance_applied": True,
                "target_topk": target_rows,
                "cache_sha256": {code: cache[code]["sha256"] for code, _ in targets},
            },
            "timing_proof": {"signal_lag_one_day": True, "execution_is_first_next_valid_session": True},
        })
    payload = {
        "schema": "wc006-candidate-stream/v1",
        "paper_only": True,
        "broker_orders_enabled": False,
        "live_execution_authorized": False,
        "authoritative_trading_state": False,
        "selector": "repaired_wc006_walkforward_export_artifact",
        "signal_lag_one_day": True,
        "topk": 3,
        "hold": 20,
        "exact_tag": "tag",
        "records": records,
        "source_hashes": {"active_config": _sha(inputs / "active_config.json"), "universe": _sha(inputs / "universe.txt"), "calendar": _sha(inputs / "calendar.json"), "adapter": hashlib.sha256(Path(__file__).parents[1].joinpath("experiments/a_share/build_wc006_candidate_stream.py").read_bytes()).hexdigest()},
        "cache_files": cache,
        "selector_artifacts": artifacts,
        "limitations": ["fixture only; no official market-state evidence", "no returns or broker execution"],
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


class _FakeBrain:
    def __init__(self):
        self.state = 0

    def configuration_signature(self):
        return {"graph": "complete-male-cns-fixture"}

    def memory(self):
        return {"sha256": hashlib.sha256(str(self.state).encode()).hexdigest()}


class _FakeController:
    restores = []

    def __init__(self, settings):
        self.brain = _FakeBrain()

    def observe(self, frame, reinforcement):
        side = ("BUY", "SELL", "HOLD")[self.brain.state % 3]
        self.brain.state += 1
        return {"side": side, "input_sha256": hashlib.sha256(frame.tobytes() + bytes([self.brain.state])).hexdigest(), "memory": self.brain.memory(), "stimulus": reinforcement}

    def save(self, path):
        Path(path).write_text(str(self.brain.state))

    def restore(self, path):
        self.brain.state = int(Path(path).read_text())
        type(self).restores.append(self.brain.state)


def _logical_db(path):
    db = sqlite3.connect(path / "ledger.sqlite")
    result = {}
    for table in ("meta", "observations", "intents", "fills", "lots", "events"):
        result[table] = db.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
    db.close()
    return result


def test_stream_validation_and_fail_closed(tmp_path):
    manifest = _make_stream(tmp_path)
    stream = validate_candidate_stream(manifest, expected_records=3)
    assert len(stream.records) == 3
    assert [item.execution_date for item in stream.records] == ["2021-07-02", "2021-07-05", "2021-07-06"]
    payload = json.loads(manifest.read_text())
    payload["records"][1]["selector"]["prefer_tag"] = "other"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(AdapterBlocked, match="selector provenance"):
        validate_candidate_stream(manifest, expected_records=3)


def test_interrupted_resume_matches_uninterrupted_and_restores_state(tmp_path):
    manifest = _make_stream(tmp_path)
    full = tmp_path / "full"
    resumed = tmp_path / "resumed"
    _FakeController.restores = []
    run_dry_fixture(manifest, full, expected_records=3, controller_factory=_FakeController)
    run_dry_fixture(manifest, resumed, expected_records=3, max_steps=1, controller_factory=_FakeController)
    run_dry_fixture(manifest, resumed, expected_records=3, controller_factory=_FakeController)
    assert _logical_db(full) == _logical_db(resumed)
    # Compare the paired-base ledger independently (same candidates, costs and guards).
    def paired(path):
        db = sqlite3.connect(path / "paired_base.sqlite")
        result = {table: db.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall() for table in ("meta", "intents", "fills", "events")}
        db.close()
        return result
    assert paired(full) == paired(resumed)
    for path in (full, resumed):
        db = sqlite3.connect(path / "ledger.sqlite")
        assert db.execute("SELECT execution_date FROM fills ORDER BY execution_date LIMIT 1").fetchone()[0] == "2021-07-02"
        db.close()
    assert _sha(full / "brain-0.npz") == _sha(resumed / "brain-0.npz")
    assert _FakeController.restores == [1]
    events = [json.loads(line) for line in (resumed / "events.jsonl").read_text().splitlines()]
    assert events[1]["fly_action"] == "DEEXPOSE"  # no restore would repeat BUY/EXPOSE
    assert events[1]["feedback"]["turnover_ratio"] != "0"
    ghost_exit = [item for item in events[1]["paired_plan"]["fly"] if item["code"] == "000001"]
    assert ghost_exit and ghost_exit[0]["side"] == "SELL" and float(ghost_exit[0]["quantity"]) > 0


def test_dry_fixture_uses_complete_male_cns_graph(tmp_path):
    manifest = _make_stream(tmp_path)
    result = run_dry_fixture(manifest, tmp_path / "graph-run", expected_records=3, max_steps=1, controller_factory=FlyController)
    assert result["steps"] == 1
    assert (tmp_path / "graph-run" / "brain-0.npz").is_file()
