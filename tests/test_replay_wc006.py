from decimal import Decimal as D
import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from experiments.a_share.replay_wc006 import (
    AFTER_DATA_END,
    LOT,
    MarketStateError,
    POLICY_VERSION,
    CostConfig,
    WC006Ledger,
    canonical_signature,
    incremental_feedback,
    load_manifest,
    load_smoke_bundle,
    run_replay,
    _base_curve_qty,
    map_decoder_to_multiplier,
    validate_intent,
)


def write_csv(path, rows, *, official=False):
    fields = ["date", "open", "high", "low", "close"]
    if official:
        fields += ["official_status", "official_limit_up", "official_limit_down"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            item = dict(row)
            if official:
                item.update(
                    official_status="OPEN",
                    official_limit_up="12",
                    official_limit_down="8",
                )
            writer.writerow(item)


def manifest(tmp_path, *, official=False):
    source = tmp_path / "AAA.csv"
    write_csv(
        source,
        [
            {"date": "2026-01-01", "open": "10", "high": "11", "low": "9", "close": "10"},
            {"date": "2026-01-02", "open": "10", "high": "11", "low": "9", "close": "10.5"},
        ],
        official=official,
    )
    payload = {
        "version": POLICY_VERSION,
        "symbols": ["AAA.SH"],
        "base_targets": {"AAA.SH": "0.50"},
        "initial_state": {
            "cash": "2000",
            "positions": {"AAA.SH": "0"},
            "lots": [],
        },
        "ohlc": {"AAA.SH": {"path": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    return path


def subset_manifest(tmp_path):
    symbols = ["AAA.SH", "BBB.SH", "CCC.SH", "DDD.SH", "EEE.SH", "FFF.SH"]
    entries = {}
    rows = [
        {"date": "2026-01-01", "open": "10", "high": "11", "low": "9", "close": "10"},
        {"date": "2026-01-02", "open": "10", "high": "11", "low": "9", "close": "10.5"},
    ]
    for symbol in symbols:
        source = tmp_path / f"{symbol}.csv"
        write_csv(source, rows)
        entries[symbol] = {"path": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    payload = {
        "version": POLICY_VERSION,
        "symbols": symbols,
        "base_targets": {"AAA.SH": "0.20", "BBB.SH": "0.30", "CCC.SH": "0.20"},
        "initial_state": {"cash": "10000", "positions": {symbol: "0" for symbol in symbols}, "lots": []},
        "ohlc": entries,
    }
    path = tmp_path / "subset-manifest.json"
    path.write_text(json.dumps(payload))
    return path


def test_manifest_hash_lock_and_legacy_state(tmp_path):
    loaded = load_manifest(manifest(tmp_path))
    assert loaded.base_targets == {"AAA.SH": D("0.50")}
    assert loaded.data["AAA.SH"].rows[0]["authoritative_trading_state"] is False


def test_six_symbols_with_three_targets_load_as_a_true_subset(tmp_path):
    loaded = load_manifest(subset_manifest(tmp_path), strict_market_state=False)
    assert loaded.symbols == ("AAA.SH", "BBB.SH", "CCC.SH", "DDD.SH", "EEE.SH", "FFF.SH")
    assert set(loaded.base_targets) == {"AAA.SH", "BBB.SH", "CCC.SH"}


def test_subset_targets_only_generate_positive_target_intents(tmp_path, monkeypatch):
    import experiments.a_share.replay_wc006 as wc

    class FakeBrain:
        weights_frozen = False

        def memory(self):
            return {"sha256": "plastic", "changed_edges": 0}

        def configuration_signature(self):
            return {"graph": "fake"}

    class FakeFly:
        calls = 0

        def __init__(self, settings):
            self.brain = FakeBrain()

        def observe(self, frame, stimulus):
            type(self).calls += 1
            return {"side": "HOLD" if self.calls == 1 else "BUY", "input_sha256": f"input-{self.calls}", "memory": self.brain.memory(), "stimulus": stimulus}

        def save(self, path):
            Path(path).write_bytes(b"fake-checkpoint")

        def restore(self, path):
            return None

    monkeypatch.setattr(wc, "FlyController", FakeFly)
    monkeypatch.setattr(wc, "source_manifest_hashes", lambda: {"fake": "source"})
    loaded = load_manifest(subset_manifest(tmp_path))
    out = tmp_path / "subset-run"
    result = run_replay(loaded, out, capital=D("10000"), costs=CostConfig("test", D("0"), D("0"), D("0"), D("0")), max_steps=2)
    assert result["steps"] == 2
    db = sqlite3.connect(out / "ledger.sqlite")
    intents = db.execute("SELECT symbol, requested_qty, status FROM intents ORDER BY symbol").fetchall()
    assert {row[0] for row in intents} == {"AAA.SH", "BBB.SH", "CCC.SH"}
    assert all(D(row[1]) > 0 and row[2] == "UNEXECUTED_END" for row in intents)
    assert not any(row[0] in {"DDD.SH", "EEE.SH", "FFF.SH"} for row in intents)


def test_strict_market_state_fails_closed_without_official_evidence(tmp_path):
    with pytest.raises(MarketStateError):
        load_manifest(manifest(tmp_path), strict_market_state=True)


def test_strict_market_state_accepts_complete_evidence(tmp_path):
    loaded = load_manifest(manifest(tmp_path, official=True), strict_market_state=True)
    assert loaded.data["AAA.SH"].rows[0]["authoritative_trading_state"] is True


def test_missing_pinned_initial_state_fails_closed(tmp_path):
    path = manifest(tmp_path)
    payload = json.loads(path.read_text())
    payload.pop("initial_state")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="initial_state"):
        load_manifest(path)


def test_cost_profile_and_stamp_duty_are_signature_bound(tmp_path):
    loaded = load_manifest(manifest(tmp_path))
    base = canonical_signature(loaded, learning=True, strict_market_state=False, capital=D("2000"), costs=CostConfig("research", D("0.0003"), D("5"), D("0.0005"), D("0.001")))
    changed = canonical_signature(loaded, learning=True, strict_market_state=False, capital=D("2000"), costs=CostConfig("legacy-shadow", D("0.0003"), D("5"), D("0.002"), D("0.001")))
    assert base != changed


@pytest.mark.parametrize(
    "side,prior,kind,multiplier",
    [("BUY", D("0.5"), "EXPOSE", D("1.0")), ("SELL", D("1"), "DEEXPOSE", D("0.5")), ("HOLD", D("0.5"), "DEFER", D("0.5"))],
)
def test_fixed_decoder_mapping_ignores_magnitude(side, prior, kind, multiplier):
    assert map_decoder_to_multiplier(side, prior) == (kind, multiplier)


def test_veto_is_only_validation_and_never_reallocation(tmp_path):
    loaded = load_manifest(manifest(tmp_path))
    reason = validate_intent(
        manifest=loaded,
        symbol="AAA.SH",
        kind="EXPOSE",
        requested_qty=LOT,
        current_qty=D("0"),
        sellable_qty=D("0"),
        cash=D("99"),
        price=D("10"),
        equity=D("2000"),
        target_exposure=D("0.85"),
        max_total_exposure=D("0.85"),
    )
    assert reason == "cash"


def test_intent_is_persisted_before_fill_and_settlement_is_idempotent(tmp_path):
    loaded = load_manifest(manifest(tmp_path))
    signature = canonical_signature(
        loaded,
        learning=True,
        strict_market_state=False,
        capital=D("2000"),
        costs=CostConfig("test", D("0.001"), D("1"), D("0"), D("0")),
    )
    ledger = WC006Ledger(tmp_path / "ledger.sqlite", initial_state={"cash": "2000", "lots": []}, signature=signature, authoritative_trading_state=False)
    intent = ledger.prepare_intent(
        signal_date="2026-01-01",
        execution_date="2026-01-02",
        symbol="AAA.SH",
        neural_side="BUY",
        order_side="BUY",
        intent_kind="EXPOSE",
        target_exposure=D("0.5"),
        requested_qty=LOT,
        observation_id=1,
        checkpoint_sha256="cp",
    )
    assert ledger.pending("2026-01-02")[0]["status"] == "PREPARED"
    ledger.settle(intent, execution_date="2026-01-02", quantity=LOT, price=D("10"), fee=D("1"))
    ledger.settle(intent, execution_date="2026-01-02", quantity=LOT, price=D("10"), fee=D("1"))
    assert ledger.pending() == []
    assert D(ledger.get_meta("cash")) == D("999")


def test_neural_side_is_separate_from_order_side_and_id_is_deterministic(tmp_path):
    loaded = load_manifest(manifest(tmp_path))
    signature = canonical_signature(loaded, learning=True, strict_market_state=False, capital=D("2000"), costs=CostConfig("test", D("0"), D("0"), D("0"), D("0")))
    ledger = WC006Ledger(tmp_path / "ledger.sqlite", initial_state={"cash": "2000", "lots": []}, signature=signature, authoritative_trading_state=False)
    kwargs = dict(signal_date="2026-01-01", execution_date="2026-01-02", symbol="AAA.SH", neural_side="SELL", order_side="BUY", intent_kind="EXPOSE", target_exposure=D("0.5"), requested_qty=LOT, observation_id=1, checkpoint_sha256="cp")
    first = ledger.prepare_intent(**kwargs)
    second = ledger.prepare_intent(**kwargs)
    assert first == second
    pending = ledger.pending("2026-01-02")[0]
    assert pending["neural_side"] == "SELL"
    assert pending["order_side"] == "BUY"


def test_changed_settlement_is_rejected(tmp_path):
    loaded = load_manifest(manifest(tmp_path))
    signature = canonical_signature(loaded, learning=True, strict_market_state=False, capital=D("2000"), costs=CostConfig("test", D("0"), D("0"), D("0"), D("0")))
    ledger = WC006Ledger(tmp_path / "ledger.sqlite", initial_state={"cash": "2000", "lots": []}, signature=signature, authoritative_trading_state=False)
    intent = ledger.prepare_intent(signal_date="2026-01-01", execution_date="2026-01-02", symbol="AAA.SH", neural_side="BUY", order_side="BUY", intent_kind="EXPOSE", target_exposure=D("0.5"), requested_qty=LOT, observation_id=1, checkpoint_sha256="cp")
    ledger.settle(intent, execution_date="2026-01-02", quantity=LOT, price=D("10"), fee=D("0"))
    with pytest.raises(RuntimeError, match="changed"):
        ledger.settle(intent, execution_date="2026-01-02", quantity=LOT, price=D("10.1"), fee=D("0"))


def test_feedback_is_incremental_against_base():
    result = incremental_feedback(
        equity=D("99"), previous_equity=D("100"),
        base_equity=D("98"), previous_base_equity=D("100"),
        drawdown=D("-0.02"), previous_drawdown=D("-0.01"),
        turnover_ratio=D("0.1"),
    )
    assert result["incremental_return"] > 0


def test_legacy_smoke_bundle_without_pinned_state_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="manifest.json"):
        load_smoke_bundle(tmp_path)


def test_resume_restores_date_checkpoint_and_does_not_duplicate_events(tmp_path, monkeypatch):
    import experiments.a_share.replay_wc006 as wc

    class FakeBrain:
        weights_frozen = False

        def memory(self):
            return {"sha256": "plastic", "changed_edges": 0}

        def configuration_signature(self):
            return {"graph": "fake"}

    class FakeFly:
        def __init__(self, settings):
            self.brain = FakeBrain()
            self.restored = None

        def observe(self, frame, stimulus):
            return {"side": "HOLD", "input_sha256": "input", "memory": self.brain.memory(), "stimulus": stimulus}

        def save(self, path):
            Path(path).write_bytes(b"fake-checkpoint")

        def restore(self, path):
            self.restored = Path(path)

    monkeypatch.setattr(wc, "FlyController", FakeFly)
    monkeypatch.setattr(wc, "source_manifest_hashes", lambda: {"fake": "source"})
    out = tmp_path / "run"
    loaded = load_manifest(manifest(tmp_path))
    first = run_replay(loaded, out, capital=D("2000"), costs=CostConfig("test", D("0"), D("0"), D("0"), D("0")), max_steps=1)
    second = run_replay(loaded, out, capital=D("2000"), costs=CostConfig("test", D("0"), D("0"), D("0"), D("0")))
    assert first["steps"] == 1 and second["steps"] == 1
    with (out / "ledger.sqlite").open("rb"):
        pass
    db = sqlite3.connect(out / "ledger.sqlite")
    assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 2
    assert db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 2
    assert db.execute("SELECT COUNT(*) FROM intents WHERE status='UNEXECUTED_END'").fetchone()[0] == 1
    assert len((out / "events.jsonl").read_text().splitlines()) == 2


def _logical_ledger_snapshot(path):
    """Return SQLite business contents without connection/path metadata."""
    tables = ("meta", "observations", "events", "intents", "lots", "fills", "account_snapshots")
    db = sqlite3.connect(path)
    try:
        return {
            table: db.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            for table in tables
        }
    finally:
        db.close()


def _checkpoint_snapshot(out):
    db = sqlite3.connect(out / "ledger.sqlite")
    try:
        meta = {
            key: json.loads(value)
            for key, value in db.execute("SELECT key,value FROM meta")
        }
    finally:
        db.close()
    checkpoint = meta["checkpoint"]
    checkpoint_path = out / checkpoint["file"]
    checkpoint_bytes = checkpoint_path.read_bytes()
    return meta, checkpoint_bytes, hashlib.sha256(checkpoint_bytes).hexdigest()


def test_uninterrupted_and_same_dir_resume_are_logically_identical(tmp_path, monkeypatch):
    """The resume gate must restore the saved brain and produce byte parity."""
    import experiments.a_share.replay_wc006 as wc

    class FakeBrain:
        weights_frozen = False

        def __init__(self, owner):
            self.owner = owner

        def memory(self):
            state = self.owner.state
            return {"sha256": f"plastic-{state}", "changed_edges": state}

        def configuration_signature(self):
            return {"graph": "fake"}

    class FakeFly:
        restore_calls = []

        def __init__(self, settings):
            self.state = 0
            self.brain = FakeBrain(self)

        def observe(self, frame, stimulus):
            self.state += 1
            return {
                "side": "BUY" if self.state % 2 else "SELL",
                "input_sha256": f"input-{self.state}",
                "memory": self.brain.memory(),
                "stimulus": stimulus,
            }

        def save(self, path):
            Path(path).write_text(json.dumps({"state": self.state}, sort_keys=True) + "\n")

        def restore(self, path):
            path = Path(path)
            payload = json.loads(path.read_text())
            self.state = int(payload["state"])
            self.restore_calls.append({
                "file": path.name,
                "bytes": path.read_bytes(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "state": self.state,
            })

    monkeypatch.setattr(wc, "FlyController", FakeFly)
    monkeypatch.setattr(wc, "source_manifest_hashes", lambda: {"fake": "source"})
    loaded = load_manifest(manifest(tmp_path))
    costs = CostConfig("test", D("0"), D("0"), D("0"), D("0"))

    uninterrupted_out = tmp_path / "uninterrupted"
    resumed_out = tmp_path / "resumed"
    uninterrupted = run_replay(loaded, uninterrupted_out, capital=D("2000"), costs=costs)
    interrupted = run_replay(loaded, resumed_out, capital=D("2000"), costs=costs, max_steps=1)
    assert interrupted["steps"] == 1
    interrupted_meta, interrupted_checkpoint, interrupted_checkpoint_hash = _checkpoint_snapshot(resumed_out)
    assert interrupted_meta["last_completed_index"] == 0
    assert interrupted_meta["checkpoint"]["sha256"] == interrupted_checkpoint_hash
    assert json.loads(interrupted_checkpoint)["state"] == 1

    resumed = run_replay(loaded, resumed_out, capital=D("2000"), costs=costs)
    assert resumed["status"] == uninterrupted["status"] == "complete"
    # The uninterrupted call processes two dates; the resume call reports only
    # the one date processed after the saved index.
    assert uninterrupted["steps"] == 2
    assert resumed["steps"] == 1
    assert FakeFly.restore_calls, "resume must restore the saved checkpoint before processing"
    assert FakeFly.restore_calls[0] == {
        "file": interrupted_meta["checkpoint"]["file"],
        "bytes": interrupted_checkpoint,
        "sha256": interrupted_checkpoint_hash,
        "state": 1,
    }

    assert _logical_ledger_snapshot(resumed_out / "ledger.sqlite") == _logical_ledger_snapshot(uninterrupted_out / "ledger.sqlite")
    assert (resumed_out / "events.jsonl").read_bytes() == (uninterrupted_out / "events.jsonl").read_bytes()

    resumed_meta, resumed_checkpoint, resumed_checkpoint_hash = _checkpoint_snapshot(resumed_out)
    uninterrupted_meta, uninterrupted_checkpoint, uninterrupted_checkpoint_hash = _checkpoint_snapshot(uninterrupted_out)
    for key in ("cash", "last_completed_index", "multiplier", "previous_equity", "drawdown", "checkpoint"):
        assert resumed_meta[key] == uninterrupted_meta[key]
    assert resumed_meta["last_completed_index"] == 1
    assert resumed_meta["multiplier"] == "0.5"
    assert resumed_meta["checkpoint"]["sha256"] == resumed_checkpoint_hash == uninterrupted_checkpoint_hash
    assert json.loads(resumed_checkpoint)["state"] == 2
    assert interrupted_checkpoint != resumed_checkpoint
    assert interrupted_checkpoint_hash != resumed_checkpoint_hash
    assert resumed_checkpoint == uninterrupted_checkpoint

    for out in (resumed_out, uninterrupted_out):
        db = sqlite3.connect(out / "ledger.sqlite")
        try:
            assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 2
            assert db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 2
            assert db.execute("SELECT COUNT(*) FROM intents").fetchone()[0] == 2
            assert db.execute("SELECT COUNT(*) FROM lots").fetchone()[0] == 1
            assert db.execute("SELECT COUNT(*) FROM intents WHERE status='UNEXECUTED_END'").fetchone()[0] == 1
            cash_raw = db.execute("SELECT value FROM meta WHERE key='cash'").fetchone()[0]
            assert D(json.loads(cash_raw)) == D("1000")
            assert db.execute("SELECT SUM(CAST(quantity AS REAL)) FROM lots WHERE symbol='AAA.SH'").fetchone()[0] == 100
        finally:
            db.close()


def test_end_of_data_buy_uses_sentinel_and_same_day_sell_is_rejected(tmp_path):
    loaded = load_manifest(manifest(tmp_path))
    signature = canonical_signature(
        loaded,
        learning=True,
        strict_market_state=False,
        capital=D("2000"),
        costs=CostConfig("test", D("0"), D("0"), D("0"), D("0")),
    )
    ledger = WC006Ledger(tmp_path / "ledger.sqlite", initial_state={"cash": "2000", "lots": []}, signature=signature, authoritative_trading_state=False)
    buy = ledger.prepare_intent(signal_date="2026-01-01", execution_date="2026-01-02", symbol="AAA.SH", neural_side="BUY", order_side="BUY", intent_kind="EXPOSE", target_exposure=D("0.5"), requested_qty=LOT, observation_id=1, checkpoint_sha256="cp")
    ledger.settle(buy, execution_date="2026-01-02", quantity=LOT, price=D("10"), fee=D("0"))
    assert ledger.holdings("AAA.SH", sellable_on="2026-01-02") == D("0")
    assert ledger.holdings("AAA.SH", sellable_on="9999-12-30") == D("0")
    assert ledger.db.execute("SELECT sellable_from FROM lots WHERE symbol='AAA.SH'").fetchone()[0] == AFTER_DATA_END
    sell = ledger.prepare_intent(signal_date="2026-01-02", execution_date="2026-01-02", symbol="AAA.SH", neural_side="SELL", order_side="SELL", intent_kind="DEEXPOSE", target_exposure=D("0"), requested_qty=LOT, observation_id=2, checkpoint_sha256="cp")
    with pytest.raises(RuntimeError, match=r"T\+1 or inventory"):
        ledger.settle(sell, execution_date="2026-01-02", quantity=LOT, price=D("10"), fee=D("0"))


def test_nonterminal_buy_is_sellable_from_next_session(tmp_path):
    loaded = load_manifest(manifest(tmp_path))
    signature = canonical_signature(loaded, learning=True, strict_market_state=False, capital=D("2000"), costs=CostConfig("test", D("0"), D("0"), D("0"), D("0")))
    ledger = WC006Ledger(tmp_path / "ledger.sqlite", initial_state={"cash": "2000", "lots": []}, signature=signature, authoritative_trading_state=False)
    intent = ledger.prepare_intent(signal_date="2026-01-01", execution_date="2026-01-02", symbol="AAA.SH", neural_side="BUY", order_side="BUY", intent_kind="EXPOSE", target_exposure=D("0.5"), requested_qty=LOT, observation_id=1, checkpoint_sha256="cp")
    ledger.settle(intent, execution_date="2026-01-02", sellable_from="2026-01-03", quantity=LOT, price=D("10"), fee=D("0"))
    assert ledger.holdings("AAA.SH", sellable_on="2026-01-02") == D("0")
    assert ledger.holdings("AAA.SH", sellable_on="2026-01-03") == LOT


def test_base_curve_end_of_data_lot_does_not_leak_into_iso_sellability():
    lots = [{"symbol": "AAA.SH", "quantity": "100", "sellable_from": AFTER_DATA_END, "cost_basis": "1000"}]
    assert _base_curve_qty(lots, "AAA.SH", "2026-01-02") == D("0")
    assert _base_curve_qty(lots, "AAA.SH") == LOT
