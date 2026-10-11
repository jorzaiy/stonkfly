import hashlib
import json
import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest

from experiments.a_share.replay_wc006 import WC006Ledger
import experiments.a_share.wc006_male_cns_v5_adapter as v5
from experiments.a_share.wc006_male_cns_v5_adapter import (
    V5_ARMS,
    _formal_output_scope,
    audit_v5_resume_parity,
    audit_v5_isolation,
    run_v5_fixture,
    run_v5_formal,
    validate_v5_inputs,
)
from test_wc006_male_cns_v4_adapter import _DifferentNeuralController, _FakeController, _TimedFakeController, _make_stream
from experiments.a_share.wc006_male_cns_v5_adapter import (
    _ensure_links,
    _link_signal,
    _link_state_batch,
    attach_settled_reward,
    resolve_state_batches,
    resolve_signal_outcomes,
    signal_id,
)


def _ledger(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    return WC006Ledger(
        tmp_path / "ledger.sqlite",
        initial_state={"cash": "100000", "lots": []},
        signature="v5-test-signature",
        authoritative_trading_state=False,
    )


_CANDIDATE_MANIFEST = Path("/home/ubuntu/stonkfly-lab/runs/wc006_v3_2021_inputs/20260918T-candidate-full-2021-2023-stateful/manifest.json")
_OLD_FORMAL_MARKET = Path("/home/ubuntu/stonkfly-lab/runs/wc006_v5_market_inputs/manual-bridge/authorized-v5-development-20260919T194757Z-controller-lock/manifest.json")
_NEW_FORMAL_MARKET = Path("/home/ubuntu/stonkfly-lab/runs/wc006_v5_market_inputs/manual-bridge/authorized-v5-development-20260920T122237Z-controller-lock/manifest.json")


def test_signal_id_and_fill_reward_link_are_deterministic(tmp_path):
    ledger = _ledger(tmp_path)
    try:
        sid = signal_id("2021-07-01", 0)
        assert sid == signal_id("2021-07-01", 0)
        intent = ledger.prepare_intent(
            signal_date="2021-07-01",
            execution_date="2021-07-02",
            symbol="000001.SZ",
            neural_side="BUY",
            order_side="BUY",
            intent_kind="EXPOSE",
            target_exposure=Decimal("0.85"),
            requested_qty=Decimal("100"),
            observation_id=1,
            checkpoint_sha256="checkpoint",
        )
        _link_signal(
            ledger,
            signal=sid,
            signal_date="2021-07-01",
            execution_date="2021-07-02",
            requested=Decimal("1.0"),
            intent_ids=[intent],
            pending_age=1,
        )
        ledger.settle(intent, execution_date="2021-07-02", quantity=Decimal("100"), price=Decimal("10"), fee=Decimal("5"), sellable_from="2021-07-05")
        effective, resolved = resolve_signal_outcomes(ledger, "2021-07-02", Decimal("0.5"))
        assert effective == Decimal("1.0")
        assert resolved[0]["outcome"] == "FILLED"
        rewarded = attach_settled_reward(ledger, "2021-07-02", {"stimulus": Decimal("0.02"), "turnover_ratio": Decimal("0.01")})
        assert rewarded == [sid]
        row = ledger.db.execute("SELECT outcome,reward_json FROM v5_signal_links WHERE signal_id=?", (sid,)).fetchone()
        assert row[0] == "FILLED"
        assert '"stimulus": "0.02"' in row[1]
    finally:
        ledger.close()


def test_veto_does_not_change_effective_multiplier(tmp_path):
    ledger = _ledger(tmp_path)
    try:
        sid = signal_id("2021-07-01", 0)
        intent = ledger.prepare_intent(
            signal_date="2021-07-01",
            execution_date="2021-07-02",
            symbol="000001.SZ",
            neural_side="SELL",
            order_side="SELL",
            intent_kind="DEEXPOSE",
            target_exposure=Decimal("0.425"),
            requested_qty=Decimal("100"),
            observation_id=1,
            checkpoint_sha256="checkpoint",
        )
        _link_signal(ledger, signal=sid, signal_date="2021-07-01", execution_date="2021-07-02", requested=Decimal("0.5"), intent_ids=[intent], pending_age=1)
        ledger.mark_veto(intent, "official_st_security")
        effective, resolved = resolve_signal_outcomes(ledger, "2021-07-02", Decimal("1.0"))
        assert effective == Decimal("1.0")
        assert resolved[0]["outcome"] == "VETO"
        assert ledger.get_meta("effective_multiplier") == "1.0"
    finally:
        ledger.close()


def test_empty_signal_batch_cannot_commit_requested_state(tmp_path):
    ledger = _ledger(tmp_path)
    try:
        _ensure_links(ledger)
        sid = signal_id("2021-07-01", 0)
        _link_signal(ledger, signal=sid, signal_date="2021-07-01", execution_date="2021-07-02", requested=Decimal("0.5"), intent_ids=[], pending_age=1)
        effective, resolved = resolve_signal_outcomes(ledger, "2021-07-02", Decimal("1.0"))
        assert effective == Decimal("1.0")
        assert resolved[0]["outcome"] == "EMPTY"
    finally:
        ledger.close()


@pytest.mark.parametrize("arm", V5_ARMS)
def test_explicit_arm_is_bound_and_isolated(tmp_path, arm):
    manifest, train_end = _make_stream(tmp_path)
    out = tmp_path / arm
    result = run_v5_fixture(manifest, out, expected_records=3, training_end=train_end, controller_factory=_FakeController, arm=arm)
    assert result["status"] == "complete"
    assert result["arm"] == arm
    provenance = json.loads((out / "provenance.json").read_text())
    assert provenance["arm"] == arm
    assert provenance["ablation"] == arm
    assert provenance["comparison_arm"] == "WC006+Fly"
    assert provenance["paired_base_arm"] == "WC006-only"
    assert provenance["descriptive_arms"]["Fly-only"] == "not executed"
    assert provenance["paper_only"] is True
    assert provenance["broker_orders_enabled"] is False
    assert provenance["live_execution_authorized"] is False
    assert provenance["training_authorized"] is False
    db = sqlite3.connect(out / "ledger.sqlite")
    tables = {row[0] for row in db.execute("select name from sqlite_master where type='table'")}
    events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
    eligible = [event for event in events if event["signal_eligible"]]
    assert "v5_signal_links" in tables
    assert db.execute("select count(*) from v5_signal_links").fetchone()[0] == 3
    expected_rewards = 3 if arm in ("attribution-only", "combined") else 0
    assert db.execute("select count(*) from v5_signal_links where outcome != 'PENDING' and reward_json is not null").fetchone()[0] == expected_rewards
    assert db.execute("select count(*) from v5_signal_links where outcome != 'PENDING'").fetchone()[0] == 3
    assert all(event["signal_id"] for event in eligible)
    assert all(event["neural"] and isinstance(event["guard_reason"], list) for event in events)
    if arm == "state-only":
        assert db.execute("select value from meta where key='effective_multiplier'").fetchone()[0] is not None
        assert all(not event["settled_signal_ids"] for event in eligible)
    db.close()


def test_v5_resume_parity_gate_covers_all_arms(tmp_path):
    manifest, train_end = _make_stream(tmp_path)
    for arm in V5_ARMS:
        full, resumed = tmp_path / f"full-{arm}", tmp_path / f"resumed-{arm}"
        run_v5_fixture(manifest, full, expected_records=3, training_end=train_end, controller_factory=_FakeController, arm=arm)
        run_v5_fixture(manifest, resumed, max_days=2, expected_records=3, training_end=train_end, controller_factory=_FakeController, arm=arm)
        run_v5_fixture(manifest, resumed, expected_records=3, training_end=train_end, controller_factory=_FakeController, arm=arm)
        gate = audit_v5_resume_parity(full, resumed)
        assert gate["status"] == "PASS", gate
        assert all(gate["checks"].values())




def test_v5_wall_clock_diagnostic_is_not_hash_bound_but_neural_changes_are(tmp_path):
    manifest, train_end = _make_stream(tmp_path)
    timed = tmp_path / "timed"
    baseline = tmp_path / "baseline"
    changed = tmp_path / "changed"
    run_v5_fixture(manifest, timed, expected_records=3, training_end=train_end, controller_factory=_TimedFakeController)
    run_v5_fixture(manifest, baseline, expected_records=3, training_end=train_end, controller_factory=_FakeController)
    run_v5_fixture(manifest, changed, expected_records=3, training_end=train_end, controller_factory=_DifferentNeuralController)
    gate = audit_v5_resume_parity(timed, baseline)
    assert gate["status"] == "PASS", gate
    events = [json.loads(line) for line in (timed / "events.jsonl").read_text().splitlines()]
    assert all("compute_seconds" not in event["neural"] for event in events)
    metrics = [json.loads(line) for line in (timed / "runtime_metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == len(events)
    assert [row["index"] for row in metrics] == list(range(len(events)))
    assert [row["date"] for row in metrics] == [event["date"] for event in events]
    assert all(row["arm"] == "combined" and row["seed"] == 0 and row["compute_seconds"] >= 0 for row in metrics)
    baseline_metrics = (baseline / "runtime_metrics.jsonl").read_bytes()
    assert (timed / "runtime_metrics.jsonl").read_bytes() != baseline_metrics
    changed_gate = audit_v5_resume_parity(timed, changed)
    assert changed_gate["status"] == "FAIL", changed_gate
    assert not changed_gate["checks"]["events.jsonl"]


def test_v5_runtime_metrics_resume_is_contiguous_and_not_duplicated(tmp_path):
    manifest, train_end = _make_stream(tmp_path)
    out = tmp_path / "resumed"
    run_v5_fixture(manifest, out, max_days=2, expected_records=3, training_end=train_end, controller_factory=_TimedFakeController)
    partial = [json.loads(line) for line in (out / "runtime_metrics.jsonl").read_text().splitlines()]
    assert len(partial) == 2
    run_v5_fixture(manifest, out, expected_records=3, training_end=train_end, controller_factory=_TimedFakeController)
    final = [json.loads(line) for line in (out / "runtime_metrics.jsonl").read_text().splitlines()]
    assert len(final) == len((out / "events.jsonl").read_text().splitlines())
    assert [row["index"] for row in final] == list(range(len(final)))
    assert len({(row["index"], row["date"]) for row in final}) == len(final)

def test_v5_input_and_output_scope_gates(tmp_path):
    manifest, train_end = _make_stream(tmp_path)
    stream, gate = validate_v5_inputs(manifest, expected_records=3, training_end=train_end)
    assert len(stream.base.records) == 3
    assert gate["status"] == "PASS"
    assert gate["paper_only"] is True
    with pytest.raises(RuntimeError, match="below runs/wc006_v5"):
        _formal_output_scope(tmp_path / "outside", "combined")
    payload = json.loads(manifest.read_text())
    payload["live_execution_authorized"] = True
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload))
    with pytest.raises(Exception, match="paper-only"):
        validate_v5_inputs(bad, expected_records=3, training_end=train_end)


def test_v5_resume_arm_is_signature_bound(tmp_path):
    manifest, train_end = _make_stream(tmp_path)
    out = tmp_path / "combined"
    run_v5_fixture(manifest, out, max_days=1, expected_records=3, training_end=train_end, controller_factory=_FakeController, arm="combined")
    with pytest.raises(RuntimeError, match="signature mismatch"):
        run_v5_fixture(manifest, out, expected_records=3, training_end=train_end, controller_factory=_FakeController, arm="state-only")



def test_v5_fixture_formal_api_and_seed_are_fail_closed(tmp_path):
    manifest, train_end = _make_stream(tmp_path)
    with pytest.raises(RuntimeError, match="fixture API"):
        run_v5_fixture(manifest, tmp_path / "bad", expected_records=46, training_end=train_end, controller_factory=_FakeController)
    out = tmp_path / "seeded"
    run_v5_fixture(manifest, out, expected_records=3, training_end=train_end, controller_factory=_FakeController, seed=17)
    provenance = json.loads((out / "provenance.json").read_text())
    assert provenance["seed"] == 17
    with pytest.raises(RuntimeError, match="signature mismatch"):
        run_v5_fixture(manifest, out, expected_records=3, training_end=train_end, controller_factory=_FakeController, seed=18)
    with pytest.raises(ValueError, match="seed"):
        run_v5_fixture(manifest, tmp_path / "negative", expected_records=3, training_end=train_end, controller_factory=_FakeController, seed=-1)
    assert audit_v5_isolation(out, formal=False)["status"] == "PASS"
    provenance["training_authorized"] = True
    (out / "provenance.json").write_text(json.dumps(provenance))
    isolation = audit_v5_isolation(out, formal=False)
    assert isolation["status"] == "FAIL"
    assert isolation["checks"]["training_authorized_mode"] is False


def test_v5_formal_rejects_unauthorized_market_manifest(tmp_path):
    manifest, _ = _make_stream(tmp_path)
    market = tmp_path / "market.json"
    market.write_text(json.dumps({"training_authorized": False, "broker_orders_enabled": False, "live_execution_authorized": False}))
    with pytest.raises(Exception, match="expected 46|training_authorized|market"):
        run_v5_formal(manifest, tmp_path / "formal", market, arm="combined", seed=0, controller_factory=_FakeController)
    assert not (tmp_path / "formal").exists()


def test_v5_unresolved_unknown_blocks_resume(tmp_path):
    manifest, train_end = _make_stream(tmp_path)
    out = tmp_path / "unknown"
    run_v5_fixture(manifest, out, max_days=61, expected_records=3, training_end=train_end, controller_factory=_FakeController)
    db = sqlite3.connect(out / "ledger.sqlite")
    intent = db.execute("select id from intents limit 1").fetchone()
    assert intent is not None
    db.execute("update intents set status='UNKNOWN' where id=?", intent)
    db.commit(); db.close()
    with pytest.raises(RuntimeError, match="UNKNOWN"):
        run_v5_fixture(manifest, out, expected_records=3, training_end=train_end, controller_factory=_FakeController)



def test_action_collapse_persists_quarantine_and_formal_resume_blocks(tmp_path, monkeypatch):
    manifest, train_end = _make_stream(tmp_path)
    collapsed = {"eligible_count": 3, "side_counts": {"BUY": 0, "SELL": 0, "HOLD": 3}, "action_counts": {"EXPOSE": 0, "DEEXPOSE": 0, "DEFER": 3}, "defer_ratio": 1.0, "quarantined_action_collapse": True}
    monkeypatch.setattr(v5, "assess_action_coverage", lambda events, expected_records: collapsed)
    out = tmp_path / "collapsed"
    result = run_v5_fixture(manifest, out, expected_records=3, training_end=train_end, controller_factory=_FakeController)
    assert result["status"] == "quarantined_action_collapse"
    quarantine = json.loads((out / "QUARANTINED.json").read_text())
    assert quarantine["status"] == "QUARANTINED"
    assert quarantine["reason"] == "action_collapse"
    market = tmp_path / "market.json"
    market.write_text(json.dumps({"training_authorized": True, "broker_orders_enabled": False, "live_execution_authorized": False}))
    with pytest.raises(RuntimeError, match="quarantined"):
        run_v5_formal(manifest, out, market, arm="combined", seed=0, controller_factory=_FakeController)



def test_v5_parity_normalizes_quarantine_run_path(tmp_path, monkeypatch):
    manifest, train_end = _make_stream(tmp_path)
    collapsed = {"eligible_count": 3, "side_counts": {"BUY": 0, "SELL": 0, "HOLD": 3}, "action_counts": {"EXPOSE": 0, "DEEXPOSE": 0, "DEFER": 3}, "defer_ratio": 1.0, "quarantined_action_collapse": True}
    monkeypatch.setattr(v5, "assess_action_coverage", lambda events, expected_records: collapsed)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    run_v5_fixture(manifest, full, expected_records=3, training_end=train_end, controller_factory=_FakeController)
    run_v5_fixture(manifest, resumed, expected_records=3, training_end=train_end, controller_factory=_FakeController)
    gate = audit_v5_resume_parity(full, resumed)
    assert gate["status"] == "PASS", gate
    assert gate["checks"]["quarantine"] is True


def test_v5_formal_accepts_pinned_development_authorization():
    _, gate = validate_v5_inputs(_CANDIDATE_MANIFEST, expected_records=46, training_end="2023-12-29", market_manifest=_NEW_FORMAL_MARKET, formal=True)
    assert gate["status"] == "PASS"
    assert gate["training_authorized"] is True
    assert gate["market_manifest_sha256"] == "e4798d60cf286df35170da6d4f9be20f6e36e18792adc8e5439e11cfe555ddab"
    assert gate["market_qa_sha256"] == "75881a19c313fb663c6e35eeb4326f0974358c5c0c2f7d31cddbf6766822ad88"
    assert gate["market_artifact_hashes"]["authorization"] == "1d957c2244460b0d6fdebf71a7720047bcb7e6767637d676e39ec8d67f55eaec"


def test_v5_formal_rejects_legacy_authorization_after_code_lock_revision():
    with pytest.raises(RuntimeError, match="authorization artifact hash|V4 adapter hash mismatch"):
        validate_v5_inputs(_CANDIDATE_MANIFEST, expected_records=46, training_end="2023-12-29", market_manifest=_OLD_FORMAL_MARKET, formal=True)


def test_v5_formal_rejects_non_fixed_development_seed():
    candidate = _CANDIDATE_MANIFEST
    market = _NEW_FORMAL_MARKET
    with pytest.raises(RuntimeError, match="fixed development seed must be 17"):
        run_v5_formal(candidate, Path("/home/ubuntu/stonkfly-lab/runs/wc006_v5_combined/seed-reject-test"), market, arm="combined", seed=16, controller_factory=_FakeController)


def test_attach_settled_reward_is_bookkeeping_only_checkpoint_unchanged(tmp_path):
    ledger = _ledger(tmp_path)
    try:
        sid = signal_id("2021-07-01", 0)
        intent = ledger.prepare_intent(signal_date="2021-07-01", execution_date="2021-07-02", symbol="000001.SZ", neural_side="BUY", order_side="BUY", intent_kind="EXPOSE", target_exposure=Decimal("0.5"), requested_qty=Decimal("100"), observation_id=1, checkpoint_sha256="checkpoint")
        _link_signal(ledger, signal=sid, signal_date="2021-07-01", execution_date="2021-07-02", requested=Decimal("1.0"), intent_ids=[intent], pending_age=1)
        ledger.settle(intent, execution_date="2021-07-02", quantity=Decimal("100"), price=Decimal("10"), fee=Decimal("0"))
        resolve_signal_outcomes(ledger, "2021-07-02", Decimal("0.5"))
        ledger.record_observation(date="2021-07-02", input_sha256="input", reinforcement="reward", neural={}, checkpoint={"file": "brain.npz", "sha256": "before"})
        before = ledger.get_observation("2021-07-02")["checkpoint"]
        assert attach_settled_reward(ledger, "2021-07-02", {"stimulus": Decimal("0.02")}) == [sid]
        assert ledger.get_observation("2021-07-02")["checkpoint"] == before
        assert ledger.db.execute("SELECT reward_json FROM v5_signal_links WHERE signal_id=?", (sid,)).fetchone()[0] is not None
    finally:
        ledger.close()


def test_state_resolution_outcomes_and_already_target_gap(tmp_path):
    expected = {"filled": "FILLED", "empty": "EMPTY", "veto": "VETO", "already_target": "FILLED"}
    for case, expected_outcome in expected.items():
        ledger = _ledger(tmp_path / case)
        try:
            current = Decimal("0.5") if case == "already_target" else Decimal("1.0")
            ids = []
            if case in {"filled", "veto", "already_target"}:
                intent = ledger.prepare_intent(signal_date="2021-07-01", execution_date="2021-07-02", symbol="000001.SZ", neural_side="BUY", order_side="BUY", intent_kind="EXPOSE", target_exposure=Decimal("0.5"), requested_qty=Decimal("100"), observation_id=1, checkpoint_sha256="checkpoint")
                ids = [intent]
                if case == "veto":
                    ledger.mark_veto(intent, "test veto")
                else:
                    ledger.settle(intent, execution_date="2021-07-02", quantity=Decimal("100"), price=Decimal("10"), fee=Decimal("0"))
            _link_state_batch(ledger, signal_date="2021-07-01", execution_date="2021-07-02", requested=Decimal("0.5"), intent_ids=ids, pending_age=1)
            effective, resolved = resolve_state_batches(ledger, "2021-07-02", current)
            assert resolved[0]["outcome"] == expected_outcome
            assert effective == (Decimal("0.5") if expected_outcome == "FILLED" else current)
        finally:
            ledger.close()
    # There is no distinct ALREADY_AT_TARGET enum: an already-satisfied
    # requested state is currently characterized as FILLED.


class _AlwaysSellController(_FakeController):
    def observe(self, frame, reinforcement):
        result = super().observe(frame, reinforcement)
        result["side"] = "SELL"
        return result


def test_event_target_exposure_characterizes_requested_effective_split(tmp_path):
    manifest, train_end = _make_stream(tmp_path)
    out = tmp_path / "event-split"
    run_v5_fixture(manifest, out, expected_records=3, training_end=train_end, controller_factory=_AlwaysSellController, arm="state-only")
    events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
    event = next(item for item in events if item["signal_eligible"])
    requested = Decimal(event["requested_multiplier"])
    effective = Decimal(event["effective_multiplier"])
    target = Decimal(event["target_exposure"])
    assert requested == Decimal("0.5") and effective == Decimal("1.0")
    assert target == Decimal("0.85") * effective
