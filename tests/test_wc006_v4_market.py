from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import sqlite3

import pytest

from experiments.a_share.replay_wc006 import D, WC006Ledger
from experiments.a_share.wc006_v4_market import (
    CorporateAction,
    MarketInputsBlocked,
    StatusRow,
    XyszMarketInputs,
    apply_corporate_actions,
    execution_guard,
    load_xysz_market_inputs,
    raw_equity,
    raw_execution_price,
)
from experiments.a_share.wc006_male_cns_v4_adapter import validate_v4_stream


PARENT = Path("/home/ubuntu/stonkfly-lab/runs/wc006_v4_xysz_full/20260919T054212Z-125codes/manifest.json")


def _bundle(*, status: StatusRow | None = None, actions=()):
    status = status or StatusRow("000001.SZ", "2021-01-02", D("10"), D("11"), D("9"), D("0.1"), D("0.1"), False, False, False, False)
    rows = {
        "2021-01-01": {"open": D("10"), "high": D("10"), "low": D("10"), "close": D("10"), "volume": D("1")},
        "2021-01-02": {"open": D("10"), "high": D("10"), "low": D("10"), "close": D("10"), "volume": D("1")},
        "2021-01-03": {"open": D("11"), "high": D("11"), "low": D("11"), "close": D("11"), "volume": D("1")},
        "2021-01-04": {"open": D("12"), "high": D("12"), "low": D("12"), "close": D("12"), "volume": D("1")},
        "2021-01-05": {"open": D("12"), "high": D("12"), "low": D("12"), "close": D("12"), "volume": D("1")},
    }
    status_map = {(status.code, status.date): status}
    for date in rows:
        if date != status.date:
            status_map[(status.code, date)] = StatusRow(status.code, date, D("10"), D("20"), D("1"), D("0.1"), D("0.1"), False, False, False, False)
    return XyszMarketInputs(Path("."), "parent", "qa", {"000001.SZ": rows}, status_map, tuple(actions), {})


def _ledger(tmp_path, cash="100000", lots=None):
    return WC006Ledger(tmp_path / "ledger.sqlite", initial_state={"cash": cash, "lots": lots or []}, signature="market-test", authoritative_trading_state=True)


def test_raw_execution_uses_raw_open_and_official_limits_not_qfq():
    bundle = _bundle()
    assert raw_execution_price(bundle, "000001.SZ", "2021-01-02", "BUY", D("0.0005")) == D("10.0050")
    altered_qfq = D("999")
    assert altered_qfq != bundle.raw_prices["000001.SZ"]["2021-01-02"]["open"]
    limited = _bundle(status=StatusRow("000001.SZ", "2021-01-02", D("10"), D("10.001"), D("9"), D("0.1"), D("0.1"), False, False, False, False))
    with pytest.raises(MarketInputsBlocked, match="official_limit_up"):
        raw_execution_price(limited, "000001.SZ", "2021-01-02", "BUY", D("0.0005"))


def test_suspension_and_st_are_execution_vetoes():
    suspended = _bundle(status=StatusRow("000001.SZ", "2021-01-02", D("10"), D("11"), D("9"), D("0.1"), D("0.1"), False, True, False, False))
    assert execution_guard(suspended, "000001.SZ", "2021-01-02", "BUY", D("10")) == "missing_open_or_unverified_suspension"
    st = _bundle(status=StatusRow("000001.SZ", "2021-01-02", D("10"), D("11"), D("9"), D("0.1"), D("0.1"), True, False, False, False))
    assert execution_guard(st, "000001.SZ", "2021-01-02", "BUY", D("10")) == "official_st_security"


def test_cash_dividend_and_stock_receivable_then_listing_are_pit_and_t1(tmp_path):
    action = CorporateAction("div", "DIVIDEND", "000001.SZ", "2021-01-01", "2021-01-02", "2021-01-03", "2021-01-03", "2021-01-04", D("0.50"), D("1"), status="implemented")
    bundle = _bundle(actions=(action,))
    ledger = _ledger(tmp_path, lots=[{"symbol": "000001.SZ", "quantity": "100", "sellable_from": "2021-01-01", "cost_basis": "1000"}])
    assert apply_corporate_actions(ledger, bundle, "2021-01-02", tuple(bundle.raw_prices["000001.SZ"]), stage="start_of_day") == []
    apply_corporate_actions(ledger, bundle, "2021-01-02", tuple(bundle.raw_prices["000001.SZ"]), stage="after_open_settlement")
    equity_record, _ = raw_equity(ledger, bundle, "2021-01-02", {"000001.SZ"})
    assert equity_record == D("101000")
    # Same-day replay is idempotent and the pre-ex raw close still includes rights.
    assert apply_corporate_actions(ledger, bundle, "2021-01-02", tuple(bundle.raw_prices["000001.SZ"]), stage="after_open_settlement") == []
    apply_corporate_actions(ledger, bundle, "2021-01-03", tuple(bundle.raw_prices["000001.SZ"]), stage="start_of_day")
    assert D(ledger.get_meta("cash")) == D("100050")
    assert raw_equity(ledger, bundle, "2021-01-03", {"000001.SZ"})[0] == D("102250")
    apply_corporate_actions(ledger, bundle, "2021-01-04", tuple(bundle.raw_prices["000001.SZ"]), stage="start_of_day")
    assert ledger.holdings("000001.SZ", sellable_on="2021-01-04") == D("100")
    assert ledger.holdings("000001.SZ", sellable_on="2021-01-05") == D("200")
    ledger.close()


def _record_date_dividend():
    return CorporateAction(
        "record-date-dividend",
        "DIVIDEND",
        "000001.SZ",
        "2021-01-01",
        "2021-01-02",
        "2021-01-03",
        "2021-01-04",
        None,
        D("0.50"),
        D("0"),
        status="implemented",
    )


def _prepare_market_intent(ledger, *, side, quantity):
    return ledger.prepare_intent(
        signal_date="2021-01-01",
        execution_date="2021-01-02",
        symbol="000001.SZ",
        neural_side="BUY" if side == "BUY" else "SELL",
        order_side=side,
        intent_kind="EXPOSE" if side == "BUY" else "DEEXPOSE",
        target_exposure=D("0.85") if side == "BUY" else D("0"),
        requested_qty=D(quantity),
        observation_id=0,
        checkpoint_sha256="test-checkpoint",
    )


def test_record_date_open_sell_does_not_receive_dividend(tmp_path):
    bundle = _bundle(actions=(_record_date_dividend(),))
    ledger = _ledger(tmp_path, lots=[{"symbol": "000001.SZ", "quantity": "100", "sellable_from": "2021-01-01", "cost_basis": "1000"}])
    assert apply_corporate_actions(ledger, bundle, "2021-01-02", tuple(bundle.raw_prices["000001.SZ"]), stage="start_of_day") == []
    intent = _prepare_market_intent(ledger, side="SELL", quantity="100")
    ledger.settle(intent, execution_date="2021-01-02", quantity=D("100"), price=D("10"), fee=D("0"), sellable_from="2021-01-03")
    events = apply_corporate_actions(ledger, bundle, "2021-01-02", tuple(bundle.raw_prices["000001.SZ"]), stage="after_open_settlement")
    assert any(event.get("event") == "ENTITLEMENT" and event.get("quantity") == "0" for event in events)
    assert ledger.db.execute("select cash_receivable from corporate_action_entitlements").fetchone() == ("0.0",)
    ledger.close()


def test_record_date_open_buy_receives_dividend(tmp_path):
    bundle = _bundle(actions=(_record_date_dividend(),))
    ledger = _ledger(tmp_path)
    assert apply_corporate_actions(ledger, bundle, "2021-01-02", tuple(bundle.raw_prices["000001.SZ"]), stage="start_of_day") == []
    intent = _prepare_market_intent(ledger, side="BUY", quantity="100")
    ledger.settle(intent, execution_date="2021-01-02", quantity=D("100"), price=D("10"), fee=D("0"), sellable_from="2021-01-03")
    events = apply_corporate_actions(ledger, bundle, "2021-01-02", tuple(bundle.raw_prices["000001.SZ"]), stage="after_open_settlement")
    assert any(event.get("event") == "ENTITLEMENT" and event.get("quantity") == "100" for event in events)
    assert ledger.db.execute("select cash_receivable from corporate_action_entitlements").fetchone() == ("50.0",)
    ledger.close()


def test_record_date_close_holder_receives_dividend(tmp_path):
    bundle = _bundle(actions=(_record_date_dividend(),))
    ledger = _ledger(tmp_path, lots=[{"symbol": "000001.SZ", "quantity": "100", "sellable_from": "2021-01-01", "cost_basis": "1000"}])
    assert apply_corporate_actions(ledger, bundle, "2021-01-02", tuple(bundle.raw_prices["000001.SZ"]), stage="start_of_day") == []
    events = apply_corporate_actions(ledger, bundle, "2021-01-02", tuple(bundle.raw_prices["000001.SZ"]), stage="after_open_settlement")
    assert any(event.get("event") == "ENTITLEMENT" and event.get("quantity") == "100" for event in events)
    assert ledger.db.execute("select cash_receivable from corporate_action_entitlements").fetchone() == ("50.0",)
    ledger.close()


def test_record_date_registration_is_idempotent_across_stages(tmp_path):
    bundle = _bundle(actions=(_record_date_dividend(),))
    ledger = _ledger(tmp_path, lots=[{"symbol": "000001.SZ", "quantity": "100", "sellable_from": "2021-01-01", "cost_basis": "1000"}])
    assert apply_corporate_actions(ledger, bundle, "2021-01-02", tuple(bundle.raw_prices["000001.SZ"]), stage="start_of_day") == []
    first = apply_corporate_actions(ledger, bundle, "2021-01-02", tuple(bundle.raw_prices["000001.SZ"]), stage="after_open_settlement")
    assert any(event.get("event") == "ENTITLEMENT" for event in first)
    assert apply_corporate_actions(ledger, bundle, "2021-01-02", tuple(bundle.raw_prices["000001.SZ"]), stage="start_of_day") == []
    assert apply_corporate_actions(ledger, bundle, "2021-01-02", tuple(bundle.raw_prices["000001.SZ"]), stage="after_open_settlement") == []
    assert ledger.db.execute("select count(*),cash_receivable from corporate_action_entitlements").fetchone() == (1, "50.0")
    ledger.close()


@pytest.mark.parametrize("stage", ["pre_settlement", "post_settlement"])
def test_legacy_corporate_action_stage_names_are_rejected(tmp_path, stage):
    ledger = _ledger(tmp_path)
    with pytest.raises(ValueError, match="invalid corporate-action stage"):
        apply_corporate_actions(ledger, _bundle(), "2021-01-02", ("2021-01-01", "2021-01-02"), stage=stage)
    ledger.close()


def test_integer_stock_entitlement_is_not_lot_floored(tmp_path):
    action = CorporateAction("whole-share", "DIVIDEND", "000001.SZ", "2021-01-01", "2021-01-02", "2021-01-03", None, "2021-01-04", D("0"), D("0.3"), status="implemented")
    bundle = _bundle(actions=(action,))
    ledger = _ledger(tmp_path, lots=[{"symbol": "000001.SZ", "quantity": "100", "sellable_from": "2021-01-01", "cost_basis": "1000"}])
    apply_corporate_actions(ledger, bundle, "2021-01-02", tuple(bundle.raw_prices["000001.SZ"]), stage="after_open_settlement")
    apply_corporate_actions(ledger, bundle, "2021-01-03", tuple(bundle.raw_prices["000001.SZ"]), stage="start_of_day")
    apply_corporate_actions(ledger, bundle, "2021-01-04", tuple(bundle.raw_prices["000001.SZ"]), stage="start_of_day")
    assert ledger.holdings("000001.SZ", sellable_on="2021-01-05") == D("130")
    ledger.close()


def test_fractional_stock_entitlement_fails_closed(tmp_path):
    action = CorporateAction("fractional", "DIVIDEND", "000001.SZ", "2021-01-01", "2021-01-02", "2021-01-03", None, "2021-01-04", D("0"), D("0.005"), status="implemented")
    ledger = _ledger(tmp_path, lots=[{"symbol": "000001.SZ", "quantity": "100", "sellable_from": "2021-01-01", "cost_basis": "1000"}])
    with pytest.raises(MarketInputsBlocked, match="fractional"):
        apply_corporate_actions(ledger, _bundle(actions=(action,)), "2021-01-02", ("2021-01-01", "2021-01-02", "2021-01-03"), stage="after_open_settlement")
    ledger.close()


def test_future_announcement_is_not_an_entitlement_and_rights_are_declined(tmp_path):
    future = CorporateAction("future", "DIVIDEND", "000001.SZ", "2021-01-03", "2021-01-02", "2021-01-02", "2021-01-03", "2021-01-04", D("1"), D("1"), status="implemented")
    rights = CorporateAction("rights", "RIGHTS", "000001.SZ", "2021-01-01", ex_date="2021-01-02", listing_date="2021-01-04", rights_ratio=D("0.3"), rights_price=D("5"), status="implemented")
    bundle = _bundle(actions=(future, rights)); ledger = _ledger(tmp_path, lots=[{"symbol": "000001.SZ", "quantity": "100", "sellable_from": "2021-01-01", "cost_basis": "1000"}])
    events = apply_corporate_actions(ledger, bundle, "2021-01-02", tuple(bundle.raw_prices["000001.SZ"]), stage="start_of_day")
    assert any(item.get("decision") == "DECLINE_NO_SUBSCRIPTION" for item in events)
    assert not ledger.db.execute("select 1 from corporate_action_entitlements where action_id='future'").fetchone()
    row = ledger.db.execute("select rights_decision from corporate_action_entitlements where action_id='rights'").fetchone()
    assert row == ("DECLINE_NO_SUBSCRIPTION",)
    assert D(ledger.get_meta("cash")) == D("100000")
    ledger.close()


def test_parent_and_qa_hash_tamper_fail_closed():
    bundle = load_xysz_market_inputs(PARENT)
    assert len(bundle.raw_prices) == 125
    assert len(bundle.status) == 89534
    assert len(bundle.artifact_hashes) >= 150
    with pytest.raises(MarketInputsBlocked, match="parent manifest hash"):
        load_xysz_market_inputs(PARENT, expected_parent_sha256="0" * 64)
    with pytest.raises(MarketInputsBlocked, match="QA hash"):
        load_xysz_market_inputs(PARENT, expected_qa_sha256="0" * 64)


def test_formal_46_dates_pin_raw_next_session_and_status():
    stream = validate_v4_stream(
        Path("/home/ubuntu/stonkfly-lab/runs/wc006_v3_2021_inputs/20260918T-candidate-full-2021-2023-stateful/manifest.json"),
        expected_records=46,
        market_manifest=PARENT,
    )
    assert stream.market is not None
    assert len(stream.base.records) == 46
    from experiments.a_share.wc006_v4_market import _price_rows, _status_row
    for record in stream.base.records[:3]:
        assert record.execution_date > record.signal_date
        for target in record.targets:
            assert _price_rows(stream.market, target["code"])
            assert _status_row(stream.market, target["code"], record.execution_date) is not None


def _sqlite_snapshot(path: Path):
    db = sqlite3.connect(path)
    tables = [row[0] for row in db.execute("select name from sqlite_master where type='table' and name not like 'sqlite_%' order by name")]
    snapshot = {}
    for table in tables:
        snapshot[table] = db.execute(f"select * from {table} order by rowid").fetchall()
    db.close()
    return snapshot


def test_formal_market_interruption_resume_is_logically_identical(tmp_path):
    # This is a bounded formal-data parity test, not a training/smoke run.
    from experiments.a_share.wc006_male_cns_v4_adapter import run_v4_fixture
    from tests.test_wc006_male_cns_v4_adapter import _FakeController
    candidate = Path("/home/ubuntu/stonkfly-lab/runs/wc006_v3_2021_inputs/20260918T-candidate-full-2021-2023-stateful/manifest.json")
    market = Path("/home/ubuntu/stonkfly-lab/runs/wc006_v4_xysz_full/20260919T054212Z-125codes/manifest.json")
    full, resumed = tmp_path / "formal-full", tmp_path / "formal-resumed"
    _FakeController.restores = []
    run_v4_fixture(candidate, full, expected_records=46, market_manifest=market, controller_factory=_FakeController)
    run_v4_fixture(candidate, resumed, max_days=3, expected_records=46, market_manifest=market, controller_factory=_FakeController)
    run_v4_fixture(candidate, resumed, expected_records=46, market_manifest=market, controller_factory=_FakeController)
    assert _sqlite_snapshot(full / "ledger.sqlite") == _sqlite_snapshot(resumed / "ledger.sqlite")
    assert _sqlite_snapshot(full / "paired_base.sqlite") == _sqlite_snapshot(resumed / "paired_base.sqlite")
    assert (full / "events.jsonl").read_bytes() == (resumed / "events.jsonl").read_bytes()
    for name in sorted(path.name for path in full.glob("brain-day-*.npz")):
        assert (full / name).read_bytes() == (resumed / name).read_bytes()
    assert _FakeController.restores, "resume must restore a saved formal-data checkpoint"
    assert len({row[0] for row in _sqlite_snapshot(resumed / "ledger.sqlite")["observations"]}) == len(_sqlite_snapshot(resumed / "ledger.sqlite")["observations"])
