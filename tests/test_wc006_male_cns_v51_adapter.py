import hashlib
import json
import os
import numpy as np
import pytest
import sqlite3
from decimal import Decimal
from pathlib import Path

from stonkfly.neural.sensory_adapter import NullSensoryAdapter
from experiments.a_share.replay_wc006 import WC006Ledger
from experiments.a_share.wc006_male_cns_v51_adapter import (
    _annual_boundary_settlement_only,
    _assert_controller_plasticity_mode,
    _validate_annual_manifest,
    _classify_empty_batch,
    _create_batch,
    _deliver_batch_rewards,
    _ensure_v51_batches,
    _formal_v51_output_scope,
    _resolve_batches,
    _resolve_annual_market_binding,
    _sync_event_jsonl,
    _write_forward_outcomes,
    run_v51_fixture,
    run_v51_formal,
    run_v51_annual,
)
from test_wc006_male_cns_v4_adapter import _FakeController, _make_stream


class _ModeAwareFakeController(_FakeController):
    def __init__(self, settings):
        super().__init__(settings)
        self.settings = settings
        self.brain.weights_frozen = not settings.learning


def _mode_factory(seen):
    def factory(settings):
        seen.append(settings)
        return _ModeAwareFakeController(settings)

    return factory


def test_v51_separates_candidate_and_exposure_batches_and_candidate_hold_still_rebalances(tmp_path):
    manifest, end = _make_stream(tmp_path)
    dates = json.loads((manifest.parent / "inputs/calendar.json").read_text())["dates"]
    out = tmp_path / "v51"
    result = run_v51_fixture(manifest, out, start_date=dates[60], end_date=dates[65], expected_records=3, training_end=dates[65], controller_factory=_FakeController)
    assert result["schema"] == "wc006-male-cns-paper/v5.1"
    events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
    assert [event["date"] for event in events] == dates[60:66]
    assert events[0]["batch"]["batch_kind"] == "candidate_selection"
    assert events[1]["batch"]["batch_kind"] == "exposure_rebalance"
    assert events[2]["batch"]["outcome"] == "HOLD_NO_ORDER"
    assert events[3]["batch"]["batch_kind"] == "candidate_selection"
    assert events[4]["batch"]["batch_kind"] == "exposure_rebalance"
    assert events[5]["batch"]["batch_kind"] == "candidate_selection"
    assert all("requested_target" in event and "effective_target" in event for event in events)
    db = sqlite3.connect(out / "ledger.sqlite")
    rows = db.execute("select batch_kind,outcome from v51_batches order by request_date").fetchall()
    db.close()
    assert [row[0] for row in rows] == ["candidate_selection", "exposure_rebalance", "candidate_selection", "exposure_rebalance", "candidate_selection"]
    assert all(row[1] == "FILLED" for row in rows)



def test_v51_null_sensory_adapter_pipeline_parity(tmp_path):
    """Null must retain V5.1's exact observe call and market-side outcomes."""
    manifest, _ = _make_stream(tmp_path)
    dates = json.loads((manifest.parent / "inputs/calendar.json").read_text())["dates"]
    baseline, null = tmp_path / "baseline", tmp_path / "null"
    common = {
        "start_date": dates[60],
        "end_date": dates[65],
        "expected_records": 3,
        "training_end": dates[65],
        "controller_factory": _FakeController,
    }
    run_v51_fixture(manifest, baseline, **common)
    run_v51_fixture(manifest, null, sensory_adapter=NullSensoryAdapter(), **common)
    baseline_events = [json.loads(line) for line in (baseline / "events.jsonl").read_text().splitlines()]
    null_events = [json.loads(line) for line in (null / "events.jsonl").read_text().splitlines()]
    assert len(baseline_events) == len(null_events)
    neural_fields = ("left_hz", "right_hz", "difference_hz", "gate_spikes", "KC_spikes", "reward_spikes", "aversive_spikes", "total_spikes", "side", "input_sha256")
    for left, right in zip(baseline_events, null_events):
        assert left["observation"]["visual"]["frame_sha256"] == right["observation"]["visual"]["frame_sha256"]
        assert {key: left["neural_output"].get(key, "unavailable") for key in neural_fields} == {key: right["neural_output"].get(key, "unavailable") for key in neural_fields}
        assert left["decoder"] == right["decoder"]
        assert left["requested_target"] == right["requested_target"]
        left_outcome = left["execution"]["settled_prior_observation_outcome"]
        right_outcome = right["execution"]["settled_prior_observation_outcome"]
        for key in ("fills", "vetoes", "guard_reasons", "turnover", "fees", "reinforcement_classification"):
            assert left_outcome[key] == right_outcome[key]
        assert left["reward"] == right["reward"]
        sensory = right["observation"]["sensory"]
        assert sensory["adapter"]["adapter_name"] == "null"
        assert sensory["stimulation"] is None
        assert sensory["stimulation_sha256"] is None
    for root in (baseline, null):
        provenance = json.loads((root / "provenance.json").read_text())
        assert "configuration_sha256" in provenance["sensory_adapter"]
    baseline_db = sqlite3.connect(baseline / "ledger.sqlite")
    null_db = sqlite3.connect(null / "ledger.sqlite")
    try:
        baseline_final = json.loads(baseline_db.execute("select value from meta where key='final_settlement'").fetchone()[0])
        null_final = json.loads(null_db.execute("select value from meta where key='final_settlement'").fetchone()[0])
    finally:
        baseline_db.close()
        null_db.close()
    assert baseline_final["equity"] == null_final["equity"]
def test_v51_resume_matches_uninterrupted_fixture(tmp_path):
    manifest, _ = _make_stream(tmp_path)
    dates = json.loads((manifest.parent / "inputs/calendar.json").read_text())["dates"]
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    run_v51_fixture(manifest, full, start_date=dates[60], end_date=dates[65], expected_records=3, training_end=dates[65], controller_factory=_FakeController)
    run_v51_fixture(manifest, resumed, start_date=dates[60], end_date=dates[65], max_days=2, expected_records=3, training_end=dates[65], controller_factory=_FakeController)
    run_v51_fixture(manifest, resumed, start_date=dates[60], end_date=dates[65], expected_records=3, training_end=dates[65], controller_factory=_FakeController)
    assert (full / "events.jsonl").read_bytes() == (resumed / "events.jsonl").read_bytes()
    assert (full / "runtime_metrics.jsonl").read_bytes() == (resumed / "runtime_metrics.jsonl").read_bytes()
    for name in ("ledger.sqlite", "paired_base.sqlite"):
        left = sqlite3.connect(full / name); right = sqlite3.connect(resumed / name)
        assert left.execute("select * from events order by rowid").fetchall() == right.execute("select * from events order by rowid").fetchall()
        assert left.execute("select * from intents order by rowid").fetchall() == right.execute("select * from intents order by rowid").fetchall()
        left.close(); right.close()


def test_v51_batch_outcomes_include_already_empty_and_veto(tmp_path):
    ledger = WC006Ledger(tmp_path / "ledger.sqlite", initial_state={"cash": "100000", "lots": []}, signature="v51", authoritative_trading_state=False)
    try:
        _ensure_v51_batches(ledger)
        batch = _create_batch(ledger, kind="exposure_rebalance", request_date="2021-01-01", execution_date="2021-01-02", candidate_epoch=1, requested=Decimal("0.5"), effective=Decimal("1.0"), requested_target_exposure=Decimal("0.425"), effective_target_exposure=Decimal("0.85"), intent_ids=[], empty_outcome="ALREADY_AT_TARGET")
        already = _resolve_batches(ledger, "2021-01-02", Decimal("1.0"))
        assert already[0]["outcome"] == "ALREADY_AT_TARGET"
        assert ledger.get_meta("effective_multiplier") == "0.5"
        controller = _FakeController(None)
        assert _deliver_batch_rewards(ledger, already, {"stimulus": "reward"}, controller, "2021-01-02")[0]["status"] == "DELIVERED"
        reward_calls = len(controller.reward_calls)
        intent = ledger.prepare_intent(signal_date="2021-01-02", execution_date="2021-01-03", symbol="000001", neural_side="SELL", order_side="SELL", intent_kind="DEEXPOSE", target_exposure=Decimal("0.425"), requested_qty=Decimal("100"), observation_id=1, checkpoint_sha256="x")
        ledger.mark_veto(intent, "official_st_security")
        _create_batch(ledger, kind="exposure_rebalance", request_date="2021-01-02", execution_date="2021-01-03", candidate_epoch=1, requested=Decimal("0.5"), effective=Decimal("1.0"), requested_target_exposure=Decimal("0.425"), effective_target_exposure=Decimal("0.85"), intent_ids=[intent], empty_outcome="EMPTY")
        resolved = _resolve_batches(ledger, "2021-01-03", Decimal("0.5"))
        assert resolved[0]["outcome"] == "VETO"
        assert _deliver_batch_rewards(ledger, resolved, {"stimulus": "aversive"}, controller, "2021-01-03")[0]["status"] == "NONE"
        assert len(controller.reward_calls) == reward_calls
        assert ledger.get_meta("effective_multiplier") == "0.5"
    finally:
        ledger.close()


def test_v51_formal_scope_is_new_namespace(tmp_path):
    scope = _formal_v51_output_scope(Path("/home/ubuntu/stonkfly-lab/runs/wc006_v51_combined/annual"), "combined")
    assert scope["directory"] == "wc006_v51_combined"


def test_v51_already_at_target_uses_lot_executability_not_exact_decimal_zero():
    assert _classify_empty_batch([{"delta": "0.01", "quantity": "0"}], []) == "ALREADY_AT_TARGET"
    assert _classify_empty_batch([{"delta": "100.01", "quantity": "100"}], ["intent"]) == "PENDING"


def test_v51_formal_fails_closed_without_new_authorization(tmp_path):
    with pytest.raises(RuntimeError, match="new V5.1 authorization artifact is missing"):
        run_v51_formal(tmp_path / "missing-stream.json", tmp_path / "formal", tmp_path / "missing-market.json", start_date="2023-01-03")
    assert not (tmp_path / "formal").exists()


def test_v51_provenance_marks_mechanism_b_and_formal_block(tmp_path):
    manifest, _ = _make_stream(tmp_path)
    dates = json.loads((manifest.parent / "inputs/calendar.json").read_text())["dates"]
    out = tmp_path / "v51-provenance"
    run_v51_fixture(manifest, out, start_date=dates[60], end_date=dates[60], expected_records=3, training_end=dates[60], controller_factory=_FakeController)
    provenance = json.loads((out / "provenance.json").read_text())
    assert provenance["reward_mechanism"]["status"] == "NOT_PASSED"
    assert provenance["reward_mechanism"]["formal_blocked"] is True
    assert provenance["formal_status"] == "BLOCKED_NO_V51_AUTHORIZATION_AND_MECHANISM_B"
    assert provenance["comparison_arms"] == ["WC006-only", "WC006+Fly"]
    assert "ablation" not in provenance


def test_v51_reward_linkage_is_delivered_before_none_observe_and_persisted(tmp_path):
    manifest, _ = _make_stream(tmp_path)
    dates = json.loads((manifest.parent / "inputs/calendar.json").read_text())["dates"]
    out = tmp_path / "v51-rewards"
    run_v51_fixture(manifest, out, start_date=dates[60], end_date=dates[65], expected_records=3, training_end=dates[65], controller_factory=_FakeController)
    events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
    delivered = [item for event in events for item in event["reward_delivery"] if item["status"] == "DELIVERED"]
    assert delivered
    assert all(item["delivered_label"] in {"reward", "aversive", "none"} for item in delivered)
    assert all(event["neural"]["stimulus"] == "none" for event in events)
    db = sqlite3.connect(out / "ledger.sqlite")
    rows = db.execute("select reward_linkage_id,reward_delivery_status,reward_label,reward_trace_before_digest,reward_trace_after_digest,reward_weight_before_digest,reward_weight_after_digest from v51_batches").fetchall()
    observations = db.execute("select reinforcement from observations").fetchall()
    db.close()
    delivered_rows = [row for row in rows if row[1] == "DELIVERED"]
    assert delivered_rows and all(row[0] and row[2] in {"reward", "aversive", "none"} for row in delivered_rows)
    assert all(row[3] and row[4] and row[5] and row[6] for row in delivered_rows)
    assert all(row[0] == "none" for row in observations)


def test_v51_reward_delivery_rejects_duplicate_or_cross_date_linkage(tmp_path):
    ledger = WC006Ledger(tmp_path / "ledger.sqlite", initial_state={"cash": "100000", "lots": []}, signature="v51-reward", authoritative_trading_state=False)
    try:
        _ensure_v51_batches(ledger)
        batch = _create_batch(ledger, kind="exposure_rebalance", request_date="2021-01-01", execution_date="2021-01-02", candidate_epoch=1, requested=Decimal("0.5"), effective=Decimal("1.0"), requested_target_exposure=Decimal("0.425"), effective_target_exposure=Decimal("0.85"), intent_ids=[], empty_outcome="ALREADY_AT_TARGET")
        resolved = _resolve_batches(ledger, "2021-01-02", Decimal("1.0"))
        controller = _FakeController(None)
        assert _deliver_batch_rewards(ledger, resolved, {"stimulus": "reward"}, controller, "2021-01-02")[0]["status"] == "DELIVERED"
        with pytest.raises(RuntimeError, match="duplicate reward delivery"):
            _deliver_batch_rewards(ledger, resolved, {"stimulus": "reward"}, controller, "2021-01-02")
        with pytest.raises(RuntimeError, match="crossed execution date"):
            _deliver_batch_rewards(ledger, resolved, {"stimulus": "reward"}, controller, "2021-01-03")
    finally:
        ledger.close()


def test_v51_resume_rejects_tampered_checkpoint_file(tmp_path):
    manifest, _ = _make_stream(tmp_path)
    dates = json.loads((manifest.parent / "inputs/calendar.json").read_text())["dates"]
    out = tmp_path / "checkpoint-tamper"
    run_v51_fixture(manifest, out, start_date=dates[60], end_date=dates[65], max_days=1, expected_records=3, training_end=dates[65], controller_factory=_FakeController)
    db = sqlite3.connect(out / "ledger.sqlite")
    checkpoint = json.loads(db.execute("select value from meta where key='checkpoint'").fetchone()[0])
    db.close()
    path = out / checkpoint["file"]
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="checkpoint hash mismatch"):
        run_v51_fixture(manifest, out, start_date=dates[60], end_date=dates[65], expected_records=3, training_end=dates[65], controller_factory=_FakeController)


def test_v51_final_boundary_records_settlement_only_feedback_and_closes_batches(tmp_path):
    manifest, _ = _make_stream(tmp_path)
    dates = json.loads((manifest.parent / "inputs/calendar.json").read_text())["dates"]
    out = tmp_path / "boundary"
    result = run_v51_fixture(manifest, out, start_date=dates[60], end_date=dates[65], expected_records=3, training_end=dates[65], controller_factory=_FakeController)
    assert result["status"] == "complete"
    db = sqlite3.connect(out / "ledger.sqlite")
    final = json.loads(db.execute("select value from meta where key='final_settlement'").fetchone()[0])
    rows = db.execute("select outcome,reward_delivery_status from v51_batches").fetchall()
    db.close()
    assert final["semantic"].startswith("settlement-only")
    assert isinstance(final["feedback"], dict) and "stimulus" in final["feedback"]
    assert all(outcome != "PENDING" and status not in {"UNLINKED", "PENDING", "IN_FLIGHT"} for outcome, status in rows)


def test_v51_resume_rebuilds_missing_jsonl_and_rejects_extra_or_tampered_lines(tmp_path):
    manifest, _ = _make_stream(tmp_path)
    dates = json.loads((manifest.parent / "inputs/calendar.json").read_text())["dates"]
    out = tmp_path / "jsonl-recovery"
    run_v51_fixture(manifest, out, start_date=dates[60], end_date=dates[65], expected_records=3, training_end=dates[65], controller_factory=_FakeController)
    original = (out / "events.jsonl").read_bytes()
    lines = (out / "events.jsonl").read_text().splitlines()
    (out / "events.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    run_v51_fixture(manifest, out, start_date=dates[60], end_date=dates[65], expected_records=3, training_end=dates[65], controller_factory=_FakeController)
    assert (out / "events.jsonl").read_bytes() == original
    (out / "events.jsonl").write_text((out / "events.jsonl").read_text() + lines[-1] + "\n")
    with pytest.raises(RuntimeError, match="extra events"):
        run_v51_fixture(manifest, out, start_date=dates[60], end_date=dates[65], expected_records=3, training_end=dates[65], controller_factory=_FakeController)
    (out / "events.jsonl").write_bytes(original.replace(b'\"date\": \"2021-03-02\"', b'\"date\": \"2099-01-01\"'))
    with pytest.raises(RuntimeError, match="differs from authoritative|formatting or content"):
        run_v51_fixture(manifest, out, start_date=dates[60], end_date=dates[65], expected_records=3, training_end=dates[65], controller_factory=_FakeController)


ROOT = Path(__file__).resolve().parents[1]
ANNUAL = ROOT / "runs/wc006_v5_1_inputs/annual-2023-manifest-draft/manifest.json"
ANNUAL_2022 = ROOT / "runs/wc006_v5_1_v6_development/annual-2022-paired-prereg-v1/annual-manifest.json"
AUTHORIZED_2022 = ROOT / "runs/wc006_v5_1_v6_development/annual-2022-paired-prereg-v1/annual-authorized-manifest.json"
SOURCE = ROOT / "runs/wc006_v3_2021_inputs/20260918T-candidate-full-2021-2023-stateful/manifest.json"
NEW_MARKET = ROOT / "runs/wc006_v5_market_inputs/manual-bridge/authorized-v5-development-20260922T-right-issue-overlay-2022/manifest.json"
MARKET = ROOT / "runs/wc006_v5_market_inputs/manual-bridge/authorized-v5-development-20260920T122237Z-controller-lock/manifest.json"

def test_v51_annual_manifest_2022_is_data_driven_and_legal():
    annual = _validate_annual_manifest(ANNUAL_2022, SOURCE)
    payload = annual["payload"]
    assert payload["year"] == 2022
    assert payload["observation_window"] == ["2022-01-04", "2022-12-30"]
    assert payload["settlement_boundary"] == "2023-01-03"
    assert payload["candidate_record_count"] == 19


def test_v51_annual_manifest_rejects_tampered_explicit_count(tmp_path):
    import shutil
    root = tmp_path / "annual-2022"
    shutil.copytree(ANNUAL_2022.parent, root)
    path = root / "annual-manifest.json"
    payload = json.loads(path.read_text())
    payload["candidate_record_count"] = 20
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(RuntimeError, match="source annual record count mismatch"):
        _validate_annual_manifest(path, SOURCE)


def test_v51_2022_last_execution_is_not_a_boundary_signal_or_order():
    payload = _validate_annual_manifest(ANNUAL_2022, SOURCE)["payload"]
    assert payload["records"][-1]["signal_date"] == "2022-12-29"
    assert payload["records"][-1]["execution_date"] == "2022-12-30"
    assert all(item["signal_date"] != "2023-01-03" for item in payload["records"])
    assert _annual_boundary_settlement_only(
        payload,
        settlement_date="2023-01-03",
        date="2022-12-30",
        end_date="2022-12-30",
    )
    assert not _annual_boundary_settlement_only(
        payload,
        settlement_date="2023-01-03",
        date="2023-01-03",
        end_date="2022-12-30",
    )


def test_v51_annual_bridge_binds_2022_scope_without_running_model(tmp_path, monkeypatch):
    captured = {}
    def fake_run(stream_path, output, **kwargs):
        captured.update(stream_path=stream_path, output=output, kwargs=kwargs)
        return {"status": "stubbed"}
    monkeypatch.setattr("experiments.a_share.wc006_male_cns_v51_adapter._run_v51", fake_run)
    result = run_v51_annual(ANNUAL_2022, SOURCE, MARKET, tmp_path / "annual-2022")
    assert result == {"status": "stubbed"}
    assert captured["kwargs"]["start_date"] == "2022-01-04"
    assert captured["kwargs"]["end_date"] == "2022-12-30"
    assert captured["kwargs"]["expected_records"] == 46
    assert captured["kwargs"]["annual_payload"]["settlement_boundary"] == "2023-01-03"
    assert captured["kwargs"]["annual_input_gate"]["candidate_record_count"] == 19


def test_v51_annual_authorized_2022_overlay_fails_closed_after_code_change(tmp_path):
    # The historical authorization binds exact source hashes.  Enhancements
    # must not silently reuse it; a new authorization is required.
    with pytest.raises(RuntimeError, match="code binding mismatch: adapter"):
        run_v51_annual(AUTHORIZED_2022, SOURCE, NEW_MARKET, tmp_path / "authorized-2022")


@pytest.mark.parametrize("field", ["market_manifest_sha256", "market_quality_report_sha256", "supplemental_audit_sha256"])
def test_v51_annual_authorization_rejects_tampered_binding(field):
    payload = json.loads(AUTHORIZED_2022.read_text())
    payload["market_authorization"][field] = "0" * 64
    with pytest.raises(RuntimeError, match="hash mismatch|authorization mismatch"):
        _resolve_annual_market_binding(AUTHORIZED_2022, payload, NEW_MARKET)


def test_v51_annual_bridge_validates_and_binds_all_inputs(tmp_path, monkeypatch):
    captured = {}
    def fake_run(stream_path, output, **kwargs):
        captured.update(stream_path=stream_path, output=output, kwargs=kwargs)
        return {"status": "stubbed"}
    monkeypatch.setattr("experiments.a_share.wc006_male_cns_v51_adapter._run_v51", fake_run)
    result = run_v51_annual(ANNUAL, SOURCE, MARKET, tmp_path / "annual")
    assert result == {"status": "stubbed"}
    assert captured["stream_path"] == SOURCE
    assert captured["kwargs"]["market_manifest"] == MARKET.resolve()
    evidence = captured["kwargs"]["annual_input_gate"]
    assert evidence["schema"] == "wc006-v5.1-annual-input-slice/v1"
    assert evidence["observation_count"] == 242
    assert evidence["settlement_boundary"] == "2024-01-02"

def test_v51_annual_bridge_rejects_tampered_annual_record(tmp_path):
    import shutil
    root = tmp_path / "annual-2023-manifest-draft"
    shutil.copytree(ANNUAL.parent, root)
    path = root / "manifest.json"
    payload = json.loads(path.read_text())
    payload["records"][0]["execution_date"] = "2023-01-05"
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(RuntimeError, match="annual records differ from source records"):
        run_v51_annual(path, SOURCE, MARKET, tmp_path / "out")

def test_v51_annual_bridge_requires_corrected_market():
    with pytest.raises(RuntimeError, match="corrected market manifest is required"):
        run_v51_annual(ANNUAL, SOURCE, None, Path("/tmp/v51-annual-missing-market"))


def test_v51_fixture_propagates_unexpected_input_gate_error(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise ValueError("unexpected input-gate failure")
    monkeypatch.setattr("experiments.a_share.wc006_male_cns_v51_adapter.validate_v5_inputs", fail)
    with pytest.raises(ValueError, match="unexpected input-gate failure"):
        run_v51_fixture(tmp_path / "missing.json", tmp_path / "out", start_date="2023-01-03", end_date="2023-01-03")

def test_v51_annual_bridge_rejects_tampered_calendar_hash(tmp_path):
    import shutil
    root = tmp_path / "annual-2023-manifest-draft"
    shutil.copytree(ANNUAL.parent, root)
    calendar = root / "inputs/calendar.json"
    payload = json.loads(calendar.read_text())
    payload["dates"].remove("2023-01-03")
    calendar.write_text(json.dumps(payload) + "\n")
    with pytest.raises(RuntimeError, match="source input hash mismatch"):
        run_v51_annual(root / "manifest.json", SOURCE, MARKET, tmp_path / "out")


def test_v51_default_enabled_mode_is_backward_compatible_and_provenanced(tmp_path):
    manifest, _ = _make_stream(tmp_path)
    dates = json.loads((manifest.parent / "inputs/calendar.json").read_text())["dates"]
    seen = []
    out = tmp_path / "enabled"
    result = run_v51_fixture(manifest, out, start_date=dates[60], end_date=dates[60], expected_records=3, training_end=dates[60], controller_factory=_mode_factory(seen))
    assert result["plasticity_mode"] == "enabled"
    assert seen and seen[0].learning is True
    provenance = json.loads((out / "provenance.json").read_text())
    assert provenance["plasticity_mode"] == "enabled"
    assert provenance["input_gate"]["plasticity_mode"] == "enabled"
    db = sqlite3.connect(out / "ledger.sqlite")
    signature = json.loads(db.execute("select value from meta where key='signature'").fetchone()[0])
    db.close()
    assert isinstance(signature, str) and len(signature) == 64


def test_v51_frozen_mode_wires_settings_and_rejects_enabled_resume(tmp_path):
    manifest, _ = _make_stream(tmp_path)
    dates = json.loads((manifest.parent / "inputs/calendar.json").read_text())["dates"]
    seen = []
    controllers = []
    def factory(settings):
        seen.append(settings)
        controller = _ModeAwareFakeController(settings)
        controllers.append(controller)
        return controller
    out = tmp_path / "frozen"
    result = run_v51_fixture(manifest, out, start_date=dates[60], end_date=dates[65], max_days=1, expected_records=3, training_end=dates[65], controller_factory=factory, plasticity_mode="frozen")
    assert result["plasticity_mode"] == "frozen"
    assert seen and seen[0].learning is False
    assert controllers and controllers[0].brain.weights_frozen is True
    with pytest.raises(RuntimeError, match="signature mismatch"):
        run_v51_fixture(manifest, out, start_date=dates[60], end_date=dates[65], expected_records=3, training_end=dates[65], controller_factory=_mode_factory([]), plasticity_mode="enabled")


@pytest.mark.skipif(os.environ.get("STONKFLY_FULL_TEST") != "1", reason="full MaleCNS integration")
def test_v51_frozen_linked_pulse_keeps_weights_and_propagates_eligibility_dan(tmp_path):
    from stonkfly.config import Settings
    from stonkfly.neural.v51_controller import V51Controller

    controller = V51Controller(Settings(neural_ms=200, pulse_ms=100, learning=False))
    brain = controller.brain
    assert brain.n == 166700
    assert len(brain.post) == 25582938
    assert len(brain.circuit["edges"]) == 7835
    frame = np.full((180, 320, 3), 255, dtype=np.uint8)
    observation = controller.observe(frame, "none")
    assert observation["KC_spikes"] > 0
    edges = brain.circuit["edges"]
    pre = brain.circuit["pre"]
    brain.eligibility[pre] = 1.0
    before_full = hashlib.sha256(brain.weight.tobytes()).hexdigest()
    before_edges = hashlib.sha256(brain.weight[edges].tobytes()).hexdigest()
    delivery = controller.reinforce_linked("reward", "frozen-test")
    assert delivery["status"] == "APPLIED"
    assert 0 < delivery["eligible_edge_count"] < 7835
    assert delivery["dan_spikes"] > 0
    assert delivery["changed_edges"] == 0
    assert delivery["eligible_changed_edges"] == 0
    assert hashlib.sha256(brain.weight.tobytes()).hexdigest() == before_full
    assert hashlib.sha256(brain.weight[edges].tobytes()).hexdigest() == before_edges
    assert delivery["trace_before_digest"] and delivery["trace_after_digest"]


def test_v51_controller_mode_assertion_checks_both_modes():
    class Brain:
        def __init__(self, frozen):
            self.weights_frozen = frozen

    class Controller:
        def __init__(self, frozen):
            self.brain = Brain(frozen)

    _assert_controller_plasticity_mode(Controller(False), "enabled", context="test")
    _assert_controller_plasticity_mode(Controller(True), "frozen", context="test")
    with pytest.raises(RuntimeError, match="mode checkpoint/controller mismatch"):
        _assert_controller_plasticity_mode(Controller(True), "enabled", context="test")
    with pytest.raises(RuntimeError, match="mode checkpoint/controller mismatch"):
        _assert_controller_plasticity_mode(Controller(False), "frozen", context="test")
    _assert_controller_plasticity_mode(object(), "enabled", context="fixture")
    with pytest.raises(RuntimeError, match="requires brain.weights_frozen"):
        _assert_controller_plasticity_mode(object(), "frozen", context="fixture")



def test_v51_forward_outcomes_link_event_and_separate_basket_portfolio_paths(tmp_path):
    events = tmp_path / "events.jsonl"
    event = {
        "event_id": "event-1",
        "date": "2021-01-01",
        "observation": {"observation_id": 7, "candidates": [{"code": "000001", "selector_target_weight": "1.0"}]},
        "reward": {"portfolio_return": "0.01"},
        "execution": {"observation_id": 7, "next_execution_request": {"requested_execution_date": "2021-01-02"}, "settled_prior_observation_outcome": {}},
    }
    future_events = [event]
    for day in range(2, 7):
        future_events.append({"event_id": f"event-{day}", "date": f"2021-01-0{day}", "observation": {"observation_id": day, "candidates": [{"code": "000001", "selector_target_weight": "1.0"}]}, "reward": {"portfolio_return": "0.01"}, "execution": {}})
    events.write_text("\n".join(json.dumps(item) for item in future_events) + "\n")
    prices = {"000001": {date: {"close": Decimal(close), "volume": Decimal("1")} for date, close in (("2021-01-01", "100"), ("2021-01-02", "101"), ("2021-01-03", "102"), ("2021-01-04", "103"), ("2021-01-05", "104"), ("2021-01-06", "105"))}}
    _write_forward_outcomes(tmp_path, events, prices, tuple(prices["000001"]))
    row = json.loads((tmp_path / "forward_outcomes.jsonl").read_text().splitlines()[0])
    assert row["event_id"] == "event-1"
    assert row["path_type"] == "target_basket_close"
    assert row["forward_basket_return_1d"] == "0.01"
    assert row["forward_portfolio_return_1d"] == "0.01"
