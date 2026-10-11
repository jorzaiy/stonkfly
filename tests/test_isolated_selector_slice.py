import hashlib
import importlib
import json
import importlib.util
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tools.extend_purged_selector_2024_2025 import SOURCE as FROZEN_PURGED_SOURCE

from experiments.a_share.run_isolated_selector_slice import (
    CALENDAR_REL,
    CONFIG_REL,
    EXPORTER_REL,
    WALKFORWARD_REL,
    NEW_RENDER_BLOCK,
    OLD_RENDER_BLOCK,
    OVERLAY_PATCH_ID,
    PURGE_NEW_BLOCK,
    PURGE_OLD_BLOCK,
    PURGE_OVERLAY_PATCH_ID,
    TURNOVER_NEW_BLOCK,
    TURNOVER_OLD_BLOCK,
    TURNOVER_OVERLAY_PATCH_ID,
    STATE_INIT_PATCH_ID,
    PRESERVE_NEW_BLOCK,
    PRESERVE_OLD_BLOCK,
    STATE_INIT_NEW_BLOCK,
    STATE_INIT_OLD_BLOCK,
    apply_presentation_overlay,
    apply_purge_overlay,
    apply_stateful_export_overlay,
    apply_turnover_overlay,
    audit_stateful_artifacts,
    _atomic_json_write,
    _checkpoint_payload,
    _validate_checkpoint,
    contiguous_hold_schedule_dates,
    make_mirror,
    presentation_metric,
)


def test_schedule_uses_three_train_months_then_contiguous_hold():
    dates = [
        "2021-01-04", "2021-01-05", "2021-01-06",
        "2021-02-01", "2021-02-02", "2021-02-03",
        "2021-03-01", "2021-03-02", "2021-03-03",
        "2021-04-01", "2021-04-02", "2021-04-06",
        "2021-05-06", "2021-05-07", "2021-05-10",
    ]
    assert contiguous_hold_schedule_dates(dates, train_months=3, hold=2, warmup_sessions=0) == [
        "2021-04-01", "2021-04-06", "2021-05-06", "2021-05-10",
    ]


def test_mirror_source_hashes_and_root_inputs(tmp_path):
    # The external walkforward has changed since this mirror protocol was
    # frozen. A fresh mirror must fail closed rather than apply an overlay to
    # a different source block; the pinned old mirror is tested separately.
    with pytest.raises(RuntimeError, match="turnover overlay expected exactly one"):
        make_mirror(tmp_path / "staging")


def test_presentation_overlay_is_none_safe_and_only_changes_render_block(tmp_path):
    assert presentation_metric(None) == "N/A"
    assert presentation_metric(0.125) == "0.125000"
    source = Path("/home/ubuntu/trading_agent_20260408") / EXPORTER_REL
    destination = tmp_path / "exporter.py"
    destination.write_bytes(source.read_bytes())
    original = destination.read_text(encoding="utf-8")
    overlay = apply_presentation_overlay(destination)
    patched = destination.read_text(encoding="utf-8")
    assert overlay["matched_blocks"] == 1
    assert patched.replace(NEW_RENDER_BLOCK, OLD_RENDER_BLOCK) == original
    assert OLD_RENDER_BLOCK not in patched
    assert "score_text = \"N/A\" if score is None" in patched
    assert patched.split(NEW_RENDER_BLOCK)[0] == original.split(OLD_RENDER_BLOCK)[0]
    assert patched.split(NEW_RENDER_BLOCK)[1] == original.split(OLD_RENDER_BLOCK)[1]


def test_purge_overlay_only_changes_is_train_block(tmp_path):
    source = Path("/home/ubuntu/trading_agent_20260408") / EXPORTER_REL
    destination = tmp_path / "exporter.py"
    destination.write_bytes(source.read_bytes())
    original = destination.read_text(encoding="utf-8")
    overlay = apply_purge_overlay(destination)
    patched = destination.read_text(encoding="utf-8")
    assert overlay["patch_id"] == PURGE_OVERLAY_PATCH_ID
    assert overlay["matched_blocks"] == 1
    assert patched.replace(PURGE_NEW_BLOCK, PURGE_OLD_BLOCK) == original


def test_purge_overlay_keeps_every_training_label_before_train_end():
    dates = pd.bdate_range("2021-01-04", periods=90)
    index = pd.MultiIndex.from_product([dates, ["000001", "000002"]], names=["datetime", "instrument"])
    feat = pd.DataFrame({"x": 1.0}, index=index)
    namespace = {
        "pd": pd,
        "feat": feat,
        "all_dates": pd.DatetimeIndex(dates),
        "train_start": dates[0],
        "train_end": dates[70],
        "y_label": pd.Series(1.0, index=index),
        "cfg": SimpleNamespace(strategy=SimpleNamespace(hold=20)),
    }
    exec(textwrap.dedent(PURGE_NEW_BLOCK), namespace)
    kept = feat[namespace["is_train"]].index.get_level_values(0).unique()
    assert kept.max() == dates[49]
    # The label of the last kept sample ends at dates[69], the session before train_end.
    assert dates[list(dates).index(kept.max()) + 20] < namespace["train_end"]
    assert len(kept) == 50


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_turnover_overlay_excludes_ghosts_and_preserves_source_block_only(tmp_path):
    source = FROZEN_PURGED_SOURCE / "mirror/rdlab/src/walkforward.py"
    destination = tmp_path / "walkforward.py"
    destination.write_text(source.read_text(encoding="utf-8").replace(TURNOVER_NEW_BLOCK, TURNOVER_OLD_BLOCK), encoding="utf-8")
    original = destination.read_text(encoding="utf-8")
    overlay = apply_turnover_overlay(destination)
    patched = destination.read_text(encoding="utf-8")
    assert overlay["patch_id"] == TURNOVER_OVERLAY_PATCH_ID
    assert overlay["matched_blocks"] == 1
    assert patched.replace(TURNOVER_NEW_BLOCK, TURNOVER_OLD_BLOCK) == original

    walk = _load_module(destination, "turnover_overlay_walkforward")
    desired = ["000001", "000002", "000003", "000004"]
    picks = walk.apply_turnover_cap(desired, {"000001", "888888", "999999"}, topk=3, cap=1.0 / 3.0)
    assert picks == ["000001", "000002", "000003"]
    assert len(picks) == 3 and set(picks) <= set(desired)


def test_weighted_selection_and_exporter_rows_have_matching_finite_weights(tmp_path):
    mirror = FROZEN_PURGED_SOURCE / "mirror"
    walk = _load_module(mirror / "rdlab/src/walkforward.py", "weighted_overlay_walkforward")
    date = pd.Timestamp("2021-07-29")
    codes = ["000001", "000002", "000003", "000004"]
    idx = pd.MultiIndex.from_tuples([(date, code) for code in codes], names=["date", "instrument"])
    g = pd.DataFrame({"pred": [0.5, 0.4, 0.3, 0.2]}, index=idx)
    result = walk._build_weighted_selection(
        g, topk=3, cap=1.0 / 3.0, prev={"000001", "888888", "999999"},
        turnover_budget=None, turnover_penalty=0.0, gating_mean_pred_gt=None,
        weight_cap=None, weight_mode="linear", softmax_temp=0.2,
        fixed_rank_weights=None, industry_map=None, industry_max_names=None,
        industry_penalty_lambda=None, industry_penalty_free_quota=None,
    )
    assert result is not None
    picks, weights = result
    weight_codes = {str(item[1]) for item in weights.index}
    assert len(picks) == 3 and set(picks) == weight_codes and weight_codes <= set(codes)
    vals = weights.to_numpy(dtype=float)
    assert np.isfinite(vals).all() and (vals > 0).all() and np.isclose(vals.sum(), 1.0)

    sys.path.insert(0, str(mirror))
    for name in list(sys.modules):
        if name == "rdlab" or name.startswith("rdlab."):
            del sys.modules[name]
    fake_lgbm = types.ModuleType("lightgbm")
    fake_lgbm.LGBMRegressor = object
    fake_xgb = types.ModuleType("xgboost")
    fake_xgb.XGBRegressor = object
    sys.modules["lightgbm"] = fake_lgbm
    sys.modules["xgboost"] = fake_xgb
    exporter = importlib.import_module("rdlab.scripts.export_leading_daily_signal")
    cfg = SimpleNamespace(strategy=SimpleNamespace(
        topk=3, cap=1.0 / 3.0, turnover_budget=None, gate_mode="none",
        weight_mode="linear", softmax_temp=0.2, weight_cap=None,
        industry_penalty_lambda=None, industry_penalty_free_quota=None,
        industry_max_names=None, gating_mean_pred_gt=None,
    ))
    pred_day = pd.Series([0.5, 0.4, 0.3, 0.2], index=idx)
    rows = exporter._select_rebalance_target(cfg, date, pd.DataFrame(index=idx), pred_day, None, None, ["000001", "888888", "999999"])
    assert [row["code"] for row in rows] == ["000001", "000002", "000003"]
    assert all(row["score"] is not None and row["target_weight"] is not None for row in rows)
    assert np.isclose(sum(float(row["target_weight"]) for row in rows), 1.0)
    sys.path.pop(0)


def test_state_overlay_and_july_august_carry_audit(tmp_path):
    source = Path("/home/ubuntu/trading_agent_20260408") / EXPORTER_REL
    destination = tmp_path / "exporter.py"
    destination.write_bytes(source.read_bytes())
    original = destination.read_text(encoding="utf-8")
    overlay = apply_stateful_export_overlay(destination)
    patched = destination.read_text(encoding="utf-8")
    assert overlay["patch_id"] == STATE_INIT_PATCH_ID
    assert overlay["matched_init_blocks"] == 1
    assert overlay["matched_preserve_blocks"] == 1
    assert patched.replace(STATE_INIT_NEW_BLOCK, STATE_INIT_OLD_BLOCK).replace(PRESERVE_NEW_BLOCK, PRESERVE_OLD_BLOCK) == original

    watch = tmp_path / "watch_stateful"
    watch.mkdir()
    july = {
        "rebalance_applied": True,
        "effective_target_topk": [
            {"code": "000001", "score": 0.5, "target_weight": 0.6},
            {"code": "000002", "score": 0.4, "target_weight": 0.4},
        ],
        "previous_target_codes": [],
    }
    august = {
        "rebalance_applied": True,
        "effective_target_topk": [
            {"code": "000003", "score": 0.3, "target_weight": 1.0},
        ],
        "previous_target_codes": ["000001", "000002"],
    }
    (watch / "rebalance_target_20210729.json").write_text(__import__("json").dumps(july), encoding="utf-8")
    (watch / "rebalance_target_20210802.json").write_text(__import__("json").dumps(august), encoding="utf-8")
    audit = audit_stateful_artifacts(watch, ["2021-07-29", "2021-08-02"])
    assert audit[1]["previous_target_codes"] == audit[0]["target_codes"]


def test_checkpoint_resume_validates_prefix_and_rejects_changed_artifact(tmp_path):
    output = tmp_path / "resume-run"
    (output / "mirror/data/watch_stateful").mkdir(parents=True)
    source = output / "mirror/input.txt"
    source.write_text("pinned", encoding="utf-8")
    watch = output / "mirror/data/watch_stateful"
    july = {
        "rebalance_applied": True,
        "effective_target_topk": [{"code": "000001", "score": 0.5, "target_weight": 1.0}],
        "previous_target_codes": [],
    }
    august = {
        "rebalance_applied": True,
        "effective_target_topk": [{"code": "000002", "score": 0.4, "target_weight": 1.0}],
        "previous_target_codes": ["000001"],
    }
    july_path = watch / "rebalance_target_20210701.json"
    august_path = watch / "rebalance_target_20210729.json"
    july_path.write_text(json.dumps(july), encoding="utf-8")
    august_path.write_text(json.dumps(august), encoding="utf-8")
    dates = ["2021-07-01", "2021-07-29", "2021-08-02"]
    copied = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    checkpoint = _checkpoint_payload(
        output, dates, "tag", copied, {}, ["2021-07-01"], [], watch,
        pending_date="2021-07-29",
        pending_prior_hashes={str(july_path.relative_to(output)): hashlib.sha256(july_path.read_bytes()).hexdigest()},
    )
    _atomic_json_write(output / "checkpoint.json", checkpoint)
    loaded, mirror, copied_loaded, _ = _validate_checkpoint(output, dates, "tag")
    assert loaded["pending_date"] == "2021-07-29"
    assert loaded["completed_dates"] == ["2021-07-01"]
    assert mirror == output / "mirror"
    assert copied_loaded[str(source)] == copied[str(source)]
    assert not list(output.glob(".*.tmp"))
    july_path.write_text(json.dumps({**july, "target_topk": [{"code": "999999"}]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact hash changed"):
        _validate_checkpoint(output, dates, "tag")
