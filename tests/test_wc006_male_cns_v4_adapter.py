import csv
import hashlib
import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pytest

from experiments.a_share.replay_wc006 import CostConfig, D, WC006Ledger

import experiments.a_share.wc006_male_cns_v4_adapter as v4
from experiments.a_share.wc006_male_cns_v4_adapter import (
    AdapterBlocked,
    run_v4_fixture,
    validate_v4_stream,
    _frame,
    assess_action_coverage,
    _settle_v4,
    _open_equity,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _make_stream(tmp_path, *, days=67):
    root = tmp_path / "candidate"
    inputs = root / "inputs"
    (inputs / "cache").mkdir(parents=True)
    (inputs / "selector").mkdir(parents=True)
    start = date(2021, 1, 1)
    dates = [(start + timedelta(days=day)).isoformat() for day in range(days)]
    (inputs / "calendar.json").write_text(json.dumps({"dates": dates}))
    (inputs / "active_config.json").write_text(json.dumps({"walkforward": {"signal_lag_one_day": True}}))
    codes = ["000001", "000002", "000003", "000004"]
    (inputs / "universe.txt").write_text("".join(f"{code}\n" for code in codes))
    cache = {}
    for offset, code in enumerate(codes):
        path = inputs / "cache" / f"{code}_tag.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["date", "open", "high", "low", "close", "volume"])
            writer.writeheader()
            for index, day in enumerate(dates):
                value = str(10 + offset + index / 100)
                writer.writerow({"date": day, "open": value, "high": value, "low": value, "close": value, "volume": str(1000 + index)})
        cache[code] = {"copied_path": str(path.relative_to(root)), "sha256": _sha(path)}
    signal_indices = [60, 63, 65]
    target_sets = [
        [("000001", 0.4), ("000002", 0.3), ("000003", 0.3)],
        [("000002", 0.4), ("000003", 0.3), ("000004", 0.3)],
        [("000001", 0.4), ("000003", 0.3), ("000004", 0.3)],
    ]
    records, artifacts = [], []
    for index, (signal_index, targets) in enumerate(zip(signal_indices, target_sets)):
        signal, execution = dates[signal_index], dates[signal_index + 1]
        rows = [{"code": code, "score": 1.0 - position / 10, "target_weight": weight} for position, (code, weight) in enumerate(targets)]
        artifact = inputs / "selector" / f"{index:03d}.json"
        artifact.write_text(json.dumps({"source": "rdlab_leading_rebalance", "rebalance_date": signal, "rebalance_applied": True, "topk": 3, "hold": 20, "prefer_tag": "tag", "effective_target_topk": rows}))
        artifacts.append({"source_name": artifact.name, "sha256": _sha(artifact), "copied_path": str(artifact.relative_to(root))})
        records.append({"signal_date": signal, "execution_date": execution, "selector": {"artifact_sha256": _sha(artifact), "prefer_tag": "tag", "rebalance_applied": True, "target_topk": rows, "cache_sha256": {code: cache[code]["sha256"] for code, _ in targets}}, "timing_proof": {"signal_lag_one_day": True, "execution_is_first_next_valid_session": True}})
    payload = {"schema": "wc006-candidate-stream/v1", "paper_only": True, "broker_orders_enabled": False, "live_execution_authorized": False, "authoritative_trading_state": False, "selector": "repaired_wc006_walkforward_export_artifact", "signal_lag_one_day": True, "topk": 3, "hold": 20, "exact_tag": "tag", "records": records, "source_hashes": {"active_config": _sha(inputs / "active_config.json"), "universe": _sha(inputs / "universe.txt"), "calendar": _sha(inputs / "calendar.json"), "adapter": _sha(Path(__file__).parents[1] / "experiments/a_share/build_wc006_candidate_stream.py")}, "cache_files": cache, "selector_artifacts": artifacts, "limitations": ["fixture only", "official state unavailable"]}
    path = root / "manifest.json"
    path.write_text(json.dumps(payload, indent=2))
    return path, dates[65]


class _FakeBrain:
    def __init__(self):
        self.state = 0

    def configuration_signature(self):
        return {"graph": "complete-male-cns-v4-fixture"}


class _FakeController:
    restores = []

    def __init__(self, settings):
        self.brain = _FakeBrain()
        self.reward_count = 0
        self.reward_calls = []

    def reinforce_only(self, label):
        if label not in ("none", "reward", "aversive"):
            raise ValueError(label)
        self.reward_count += 1
        self.reward_calls.append(("only", label))
        return {"status": "NO_OP_NONE" if label == "none" else "APPLIED", "label": label}

    def reinforce_linked(self, label, batch_id):
        if label not in ("none", "reward", "aversive") or not batch_id:
            raise ValueError((label, batch_id))
        self.reward_count += 1
        self.reward_calls.append(("linked", label, batch_id))
        return {"status": "NO_OP_NONE" if label == "none" else "APPLIED", "label": label, "batch_id": batch_id}

    def observe(self, frame, reinforcement):
        side = ("BUY", "SELL", "HOLD")[self.brain.state % 3]
        self.brain.state += 1
        return {"side": side, "input_sha256": hashlib.sha256(frame.tobytes() + bytes([self.brain.state % 256])).hexdigest(), "memory": {"state": self.brain.state}, "stimulus": reinforcement}

    def save(self, path):
        Path(path).write_text(json.dumps({"state": self.brain.state, "reward_count": self.reward_count}, sort_keys=True))

    def restore(self, path):
        raw = Path(path).read_text()
        try:
            payload = json.loads(raw)
            self.brain.state = int(payload["state"])
            self.reward_count = int(payload.get("reward_count", 0))
        except json.JSONDecodeError:
            self.brain.state = int(raw)
            self.reward_count = 0
        type(self).restores.append(self.brain.state)


class _TimedFakeController(_FakeController):
    def observe(self, frame, reinforcement):
        result = super().observe(frame, reinforcement)
        result["compute_seconds"] = 0.001 * self.brain.state
        return result


class _DifferentNeuralController(_FakeController):
    def observe(self, frame, reinforcement):
        result = super().observe(frame, reinforcement)
        result["memory"] = {**result["memory"], "neural_variant": "changed"}
        return result


def _db(path, name="ledger.sqlite"):
    db = sqlite3.connect(path / name)
    result = {table: db.execute(f"select * from {table} order by rowid").fetchall() for table in ("meta", "observations", "intents", "fills", "events")}
    db.close()
    return result


def test_supplied_market_manifest_is_loaded_for_any_record_count(tmp_path, monkeypatch):
    # Regression: records != 46 used to skip the pinned XYSZ bundle silently
    # and execute on QFQ (2021 annual replay, audit/p0-2021-002432-concentration).
    manifest, train_end = _make_stream(tmp_path)
    sentinel = object()
    calls = []
    monkeypatch.setattr(v4, "load_xysz_market_inputs", lambda path, **kwargs: calls.append((path, kwargs)) or sentinel)
    stream = validate_v4_stream(manifest, expected_records=3, training_end=train_end, market_manifest=tmp_path / "market.json", market_expected_parent_sha256="a" * 64, market_expected_qa_sha256="b" * 64)
    assert stream.market is sentinel
    assert calls == [(tmp_path / "market.json", {"expected_parent_sha256": "a" * 64, "expected_qa_sha256": "b" * 64})]
    assert validate_v4_stream(manifest, expected_records=3, training_end=train_end).market is None


def test_supplied_market_manifest_fails_closed_for_any_record_count(tmp_path):
    manifest, train_end = _make_stream(tmp_path)
    bogus = tmp_path / "market.json"
    bogus.write_text(json.dumps({"authoritative_trading_state": True}))
    with pytest.raises(AdapterBlocked, match="parent manifest hash"):
        validate_v4_stream(manifest, expected_records=3, training_end=train_end, market_manifest=bogus, market_expected_parent_sha256="0" * 64, market_expected_qa_sha256="0" * 64)


def test_v4_requires_61_closes_and_blocks_non_signal_execution(tmp_path):
    manifest, train_end = _make_stream(tmp_path)
    stream = validate_v4_stream(manifest, expected_records=3, training_end=train_end)
    assert len(stream.base.records) == 3
    with pytest.raises(AdapterBlocked, match="lookback"):
        validate_v4_stream(manifest, expected_records=3, lookback=59, training_end=train_end)
    out = tmp_path / "run"
    result = run_v4_fixture(manifest, out, expected_records=3, training_end=train_end, controller_factory=_FakeController)
    assert result["status"] == "complete"
    events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
    assert len(events) == 6
    assert sum(event["schedule_status"] == "schedule_blocked" for event in events) == 3
    assert all(float(plan["target_exposure"]) <= 0.85 for event in events for plan in event["paired_plan"]["fly"])
    signal_dates = {record.signal_date for record in stream.base.records}
    db = sqlite3.connect(out / "ledger.sqlite")
    assert all(row[0] in signal_dates for row in db.execute("select distinct signal_date from intents"))
    for event in events:
        for plan in event["paired_plan"]["fly"]:
            if plan["code"] in {item["code"] for item in event["paired_plan"]["candidates"]}:
                expected = next(item["weight"] for item in event["paired_plan"]["candidates"] if item["code"] == plan["code"]) * 0.85 * float(event["paired_plan"]["fly_multiplier"])
                assert abs(float(plan["target_exposure"]) - expected) < 1e-12
    assert db.execute("select count(*) from observations").fetchone()[0] == 6
    assert db.execute("select count(*) from intents").fetchone()[0] > 0
    db.close()


def test_v4_allows_only_launcher_log_in_empty_run(tmp_path):
    manifest, train_end = _make_stream(tmp_path)
    out = tmp_path / "run"
    out.mkdir()
    (out / "training.log").write_text("first launch failed: empty output guard\nsecond launch pending\n")
    result = run_v4_fixture(manifest, out, expected_records=3, training_end=train_end, controller_factory=_FakeController)
    assert result["status"] == "complete"
    assert (out / "training.log").read_text().startswith("first launch failed")
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "unexpected.txt").write_text("not a launcher log")
    with pytest.raises(AdapterBlocked, match="new or resumable"):
        run_v4_fixture(manifest, blocked, expected_records=3, training_end=train_end, controller_factory=_FakeController)


def test_v4_resume_rejects_adapter_source_hash_change(tmp_path, monkeypatch):
    manifest, train_end = _make_stream(tmp_path)
    out = tmp_path / "run"
    run_v4_fixture(manifest, out, expected_records=3, training_end=train_end, controller_factory=_FakeController)
    provenance = json.loads((out / "provenance.json").read_text())
    assert provenance["adapter_sha256"] == v4._adapter_sha256()
    monkeypatch.setattr(v4, "_adapter_sha256", lambda: "tampered-adapter-source")
    with pytest.raises(RuntimeError, match="signature mismatch"):
        run_v4_fixture(manifest, out, expected_records=3, training_end=train_end, controller_factory=_FakeController)


def test_v4_daily_interruption_resume_matches_and_checkpoint_changes_action(tmp_path):
    manifest, train_end = _make_stream(tmp_path)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    _FakeController.restores = []
    run_v4_fixture(manifest, full, expected_records=3, training_end=train_end, controller_factory=_FakeController)
    run_v4_fixture(manifest, resumed, max_days=2, expected_records=3, training_end=train_end, controller_factory=_FakeController)
    run_v4_fixture(manifest, resumed, expected_records=3, training_end=train_end, controller_factory=_FakeController)
    assert _db(full) == _db(resumed)
    assert _db(full, "paired_base.sqlite") == _db(resumed, "paired_base.sqlite")
    assert (full / "events.jsonl").read_bytes() == (resumed / "events.jsonl").read_bytes()
    assert (full / "brain-day-0.npz").read_bytes() == (resumed / "brain-day-0.npz").read_bytes()
    assert _FakeController.restores == [2]
    events = [json.loads(line) for line in (resumed / "events.jsonl").read_text().splitlines()]
    assert events[2]["action"] != events[0]["action"]  # restored state affects later decision
    assert events[-1]["schedule_status"] == "eligible"
    db = sqlite3.connect(resumed / "ledger.sqlite")
    db.close()



def test_v4_wall_clock_diagnostic_is_not_hash_bound_but_neural_changes_are(tmp_path):
    manifest, train_end = _make_stream(tmp_path)
    timed = tmp_path / "timed"
    baseline = tmp_path / "baseline"
    changed = tmp_path / "changed"
    run_v4_fixture(manifest, timed, expected_records=3, training_end=train_end, controller_factory=_TimedFakeController)
    run_v4_fixture(manifest, baseline, expected_records=3, training_end=train_end, controller_factory=_FakeController)
    run_v4_fixture(manifest, changed, expected_records=3, training_end=train_end, controller_factory=_DifferentNeuralController)
    assert _db(timed) == _db(baseline)
    assert (timed / "events.jsonl").read_bytes() == (baseline / "events.jsonl").read_bytes()
    events = [json.loads(line) for line in (timed / "events.jsonl").read_text().splitlines()]
    assert all("compute_seconds" not in event["neural"] for event in events)
    assert _db(timed) != _db(changed)


def test_v4_frame_is_causal_and_final_boundary_settlement_is_exactly_once(tmp_path):
    manifest, train_end = _make_stream(tmp_path)
    stream = validate_v4_stream(manifest, expected_records=3, training_end=train_end)
    import copy
    from experiments.a_share.replay_wc006 import CostConfig, D, WC006Ledger
    prices = copy.deepcopy(stream.prices)
    ledger = WC006Ledger(tmp_path / "frame.sqlite", initial_state={"cash": "100000", "lots": []}, signature="frame", authoritative_trading_state=False)
    record = stream.base.records[0]
    before = _frame(record, prices, record.signal_date, stream.base.calendar, ledger, __import__("decimal").Decimal("1"), __import__("decimal").Decimal("0"))
    future = next(day for day in stream.base.calendar if day > record.signal_date)
    prices[record.targets[0]["code"]][future]["close"] = prices[record.targets[0]["code"]][future]["close"] * 999
    after = _frame(record, prices, record.signal_date, stream.base.calendar, ledger, __import__("decimal").Decimal("1"), __import__("decimal").Decimal("0"))
    assert before.tobytes() == after.tobytes()
    ledger.close()
    out = tmp_path / "once"
    run_v4_fixture(manifest, out, expected_records=3, training_end=train_end, controller_factory=_FakeController)
    db = sqlite3.connect(out / "ledger.sqlite")
    first_fill = db.execute("select execution_date from fills order by rowid limit 1").fetchone()[0]
    observations = db.execute("select count(*) from observations").fetchone()[0]
    fills = db.execute("select count(*) from fills").fetchone()[0]
    final_settlement = db.execute("select value from meta where key='final_settlement_date'").fetchone()[0]
    db.close()
    assert first_fill == stream.base.records[0].execution_date
    assert final_settlement == json.dumps(stream.base.records[-1].execution_date)
    run_v4_fixture(manifest, out, expected_records=3, training_end=train_end, controller_factory=_FakeController)
    db = sqlite3.connect(out / "ledger.sqlite")
    assert db.execute("select count(*) from observations").fetchone()[0] == observations
    assert db.execute("select count(*) from fills").fetchone()[0] == fills
    db.close()


def test_action_coverage_gate_normal_and_collapsed():
    normal = ([{"signal_eligible": True, "action": action, "neural": {"side": side}} for action, side in (("EXPOSE", "BUY"), ("DEEXPOSE", "SELL"), ("DEFER", "HOLD"))] * 15) + [{"signal_eligible": True, "action": "EXPOSE", "neural": {"side": "BUY"}}]
    collapsed = [{"signal_eligible": True, "action": "DEFER", "neural": {"side": "HOLD"}}] * 46
    assert assess_action_coverage(normal, expected_records=46)["quarantined_action_collapse"] is False
    report = assess_action_coverage(collapsed, expected_records=46)
    assert report["quarantined_action_collapse"] is True
    assert report["action_counts"] == {"EXPOSE": 0, "DEEXPOSE": 0, "DEFER": 46}
    assert report["defer_ratio"] == 1.0


def test_v4_missing_pre_signal_row_fails_after_repin(tmp_path):
    manifest, train_end = _make_stream(tmp_path)
    payload = json.loads(manifest.read_text())
    code = "000001"
    cache_path = manifest.parent / payload["cache_files"][code]["copied_path"]
    rows = list(csv.DictReader(cache_path.open(newline="")))
    cache_path.write_text("date,open,high,low,close,volume\n" + "\n".join(",".join(row[field] for field in ("date", "open", "high", "low", "close", "volume")) for row in rows[1:]) + "\n")
    new_hash = _sha(cache_path)
    payload["cache_files"][code]["sha256"] = new_hash
    for record in payload["records"]:
        if code in record["selector"]["cache_sha256"]:
            record["selector"]["cache_sha256"][code] = new_hash
    manifest.write_text(json.dumps(payload, indent=2))
    with pytest.raises(AdapterBlocked, match="61"):
        validate_v4_stream(manifest, expected_records=3, training_end=train_end)


def test_v4_settlement_sells_before_buys_when_cash_is_tight(tmp_path):
    ledger = WC006Ledger(tmp_path / "order.sqlite", initial_state={"cash": "0", "lots": [{"symbol": "000001", "quantity": "100", "sellable_from": "2021-01-01", "cost_basis": "1000"}]}, signature="order", authoritative_trading_state=False)
    sell = ledger.prepare_intent(signal_date="2021-01-01", execution_date="2021-01-02", symbol="000001", neural_side="SELL", order_side="SELL", intent_kind="DEEXPOSE", target_exposure=D("0"), requested_qty=D("100"), observation_id=1, checkpoint_sha256="x")
    buy = ledger.prepare_intent(signal_date="2021-01-01", execution_date="2021-01-02", symbol="000002", neural_side="BUY", order_side="BUY", intent_kind="EXPOSE", target_exposure=D("0.1"), requested_qty=D("100"), observation_id=1, checkpoint_sha256="x")
    prices = {"000001": {"2021-01-02": {"open": D("10"), "close": D("10"), "volume": D("1")}}, "000002": {"2021-01-02": {"open": D("8"), "close": D("8"), "volume": D("1")}}}
    _settle_v4(ledger, "2021-01-02", ("2021-01-01", "2021-01-02"), prices, CostConfig())
    rows = dict(ledger.db.execute("select id,status from intents").fetchall())
    assert rows[sell] == "FILLED" and rows[buy] == "FILLED"
    ledger.close()


def test_v4_event_hash_chains_are_recomputed(tmp_path):
    manifest, train_end = _make_stream(tmp_path)
    out = tmp_path / "chain"
    run_v4_fixture(manifest, out, expected_records=3, training_end=train_end, controller_factory=_FakeController)
    for name in ("ledger.sqlite", "paired_base.sqlite"):
        db = sqlite3.connect(out / name)
        previous = "GENESIS"
        for _, payload, prev, event_hash in db.execute("select event_id,payload_json,prev_hash,event_hash from events order by rowid"):
            assert prev == previous
            previous = hashlib.sha256((previous + payload).encode()).hexdigest()
            assert event_hash == previous
        db.close()


def test_v4_intermediate_missing_bar_uses_stale_mark_without_pseudo_fill(tmp_path):
    manifest, train_end = _make_stream(tmp_path)
    payload = json.loads(manifest.read_text())
    code = "000001"
    cache_path = manifest.parent / payload["cache_files"][code]["copied_path"]
    rows = list(csv.DictReader(cache_path.open(newline="")))
    missing_date = rows[62]["date"]
    rows = [row for row in rows if row["date"] != missing_date]
    with cache_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "open", "high", "low", "close", "volume"])
        writer.writeheader(); writer.writerows(rows)
    new_hash = _sha(cache_path)
    payload["cache_files"][code]["sha256"] = new_hash
    for record in payload["records"]:
        if code in record["selector"]["cache_sha256"]:
            record["selector"]["cache_sha256"][code] = new_hash
    manifest.write_text(json.dumps(payload, indent=2))
    out = tmp_path / "missing"
    result = run_v4_fixture(manifest, out, expected_records=3, training_end=train_end, controller_factory=_FakeController)
    assert result["status"] == "complete"
    events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
    stale = next(event for event in events if event["date"] == missing_date)
    assert stale["schedule_status"] == "schedule_blocked"
    assert stale["mark_provenance"]["fly"][code]["stale"] is True
    assert stale["mark_provenance"]["fly"][code]["missing_bar"] is True
    db = sqlite3.connect(out / "ledger.sqlite")
    assert db.execute("select count(*) from fills where execution_date=?", (missing_date,)).fetchone()[0] == 0
    db.close()


def test_v4_missing_open_uses_previous_close_for_open_mark(tmp_path):
    ledger = WC006Ledger(tmp_path / "mark.sqlite", initial_state={"cash": "0", "lots": [{"symbol": "000001", "quantity": "100", "sellable_from": "2021-01-01", "cost_basis": "1000"}]}, signature="mark", authoritative_trading_state=False)
    prices = {"000001": {"2021-01-01": {"open": D("10"), "close": D("11"), "volume": D("1")}}}
    assert _open_equity(ledger, prices, "2021-01-02", {"000001"}) == D("1100")
    ledger.close()
