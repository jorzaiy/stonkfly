import hashlib
import json
import shutil
from pathlib import Path

import pytest

from experiments.a_share.build_wc006_candidate_stream import StreamBlocked, build_candidate_stream


def _fixture(tmp_path):
    trading = tmp_path / "trading-read-only"
    trading.mkdir()
    config = trading / "active.json"
    config.write_text(json.dumps({
        "walkforward": {"signal_lag_one_day": True},
        "strategy": {"topk": 2, "hold": 20},
    }) + "\n")
    universe = trading / "universe.txt"
    universe.write_text("000001\n000002\n000003\n")
    calendar = trading / "calendar.json"
    calendar.write_text(json.dumps({"dates": ["2021-04-06", "2021-04-07", "2021-05-06", "2021-05-07"]}) + "\n")
    cache = trading / "cache"
    cache.mkdir()
    for code in ("000001", "000002"):
        (cache / f"{code}_tag.csv").write_text(
            "date,open,high,low,close,volume\n"
            "2021-04-06,1,1,1,1,1\n"
            "2021-04-07,1,1,1,1,1\n"
            "2021-05-06,1,1,1,1,1\n"
            "2021-05-07,1,1,1,1,1\n"
        )
    artifacts = []
    for day in ("2021-04-06", "2021-05-06"):
        path = trading / f"rebalance_target_{day.replace('-', '')}.json"
        path.write_text(json.dumps({
            "date": day.replace('-', ''),
            "rebalance_date": day,
            "source": "rdlab_leading_rebalance",
            "config": str(config),
            "signal_seed": 42,
            "prefer_tag": "tag",
            "rebalance_applied": True,
            "effective_target_topk": [
                {"code": "000001", "rank": 1, "score": 0.2, "target_weight": 0.6},
                {"code": "000002", "rank": 2, "score": 0.1, "target_weight": 0.4},
            ],
            "topk": 2,
            "hold": 20,
        }) + "\n")
        artifacts.append(path)
    return trading, config, universe, calendar, cache, artifacts


def test_candidate_stream_copies_hash_pinned_causal_slice(tmp_path):
    trading, config, universe, calendar, cache, artifacts = _fixture(tmp_path)
    out = tmp_path / "stonkfly" / "stream"
    manifest_path = build_candidate_stream(
        artifact_paths=artifacts,
        active_config=config,
        universe=universe,
        calendar=calendar,
        cache_dir=cache,
        output_dir=out,
        trading_root=trading,
    )
    payload = json.loads(manifest_path.read_text())
    assert payload["signal_lag_one_day"] is True
    assert [(r["signal_date"], r["execution_date"]) for r in payload["records"]] == [
        ("2021-04-06", "2021-04-07"),
        ("2021-05-06", "2021-05-07"),
    ]
    assert payload["authoritative_trading_state"] is False
    copied = out / payload["cache_files"]["000001"]["copied_path"]
    assert hashlib.sha256(copied.read_bytes()).hexdigest() == payload["cache_files"]["000001"]["sha256"]
    assert not (trading / "manifest.json").exists()


def test_empty_schedule_placeholder_fails_closed_without_output(tmp_path):
    trading, config, universe, calendar, cache, artifacts = _fixture(tmp_path)
    empty = artifacts[0]
    payload = json.loads(empty.read_text())
    payload["rebalance_applied"] = False
    payload["effective_target_topk"] = []
    empty.write_text(json.dumps(payload) + "\n")
    out = tmp_path / "stonkfly" / "blocked"
    with pytest.raises(StreamBlocked, match="rebalance not applied"):
        build_candidate_stream(
            artifact_paths=artifacts,
            active_config=config,
            universe=universe,
            calendar=calendar,
            cache_dir=cache,
            output_dir=out,
            trading_root=trading,
        )
    assert not out.exists()
    assert not list(out.parent.glob(f".{out.name}.staging-*"))


def test_signal_without_next_session_fails_closed(tmp_path):
    trading, config, universe, calendar, cache, artifacts = _fixture(tmp_path)
    calendar.write_text(json.dumps({"dates": ["2021-04-06"]}) + "\n")
    out = tmp_path / "stonkfly" / "blocked"
    with pytest.raises(StreamBlocked, match="no next valid session"):
        build_candidate_stream(
            artifact_paths=[artifacts[0]],
            active_config=config,
            universe=universe,
            calendar=calendar,
            cache_dir=cache,
            output_dir=out,
            trading_root=trading,
        )
    assert not out.exists()


def test_mixed_exact_tags_fail_closed(tmp_path):
    trading, config, universe, calendar, cache, artifacts = _fixture(tmp_path)
    second = json.loads(artifacts[1].read_text())
    second["prefer_tag"] = "other-tag"
    artifacts[1].write_text(json.dumps(second) + "\n")
    for code in ("000001", "000002"):
        shutil.copyfile(cache / f"{code}_tag.csv", cache / f"{code}_other-tag.csv")
    out = tmp_path / "stonkfly" / "mixed-tags"
    with pytest.raises(StreamBlocked, match="mixes exact cache tags"):
        build_candidate_stream(
            artifact_paths=artifacts,
            active_config=config,
            universe=universe,
            calendar=calendar,
            cache_dir=cache,
            output_dir=out,
            trading_root=trading,
        )
    assert not out.exists()
    assert not list(out.parent.glob(f".{out.name}.staging-*"))


def test_selector_generation_provenance_is_copied_and_hash_pinned(tmp_path):
    trading, config, universe, calendar, cache, artifacts = _fixture(tmp_path)
    checkpoint = trading / "checkpoint.json"
    mirror = trading / "mirror_provenance.json"
    checkpoint.write_text(json.dumps({"schema": "checkpoint/v1", "dates": ["2021-04-06"]}) + "\n")
    mirror.write_text(json.dumps({"schema": "mirror/v1", "purge_overlay": {"patch_id": "purged-forward-labels-v1"}}) + "\n")
    out = tmp_path / "stonkfly" / "stream-with-provenance"
    manifest_path = build_candidate_stream(
        artifact_paths=artifacts,
        active_config=config,
        universe=universe,
        calendar=calendar,
        cache_dir=cache,
        output_dir=out,
        trading_root=trading,
        selector_checkpoint=checkpoint,
        selector_mirror_provenance=mirror,
    )
    payload = json.loads(manifest_path.read_text())
    provenance = payload["selector_generation_provenance"]
    for key, source in (("checkpoint", checkpoint), ("mirror_provenance", mirror)):
        entry = provenance[key]
        copied = out / entry["copied_path"]
        assert entry["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert entry["sha256"] == hashlib.sha256(copied.read_bytes()).hexdigest()
    original_hash = provenance["checkpoint"]["sha256"]
    checkpoint.write_text(checkpoint.read_text() + "drift\n")
    assert hashlib.sha256((out / provenance["checkpoint"]["copied_path"]).read_bytes()).hexdigest() == original_hash
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() != original_hash


def test_selector_generation_provenance_requires_both_files(tmp_path):
    trading, config, universe, calendar, cache, artifacts = _fixture(tmp_path)
    checkpoint = trading / "checkpoint.json"
    checkpoint.write_text("{}\n")
    out = tmp_path / "stonkfly" / "missing-provenance"
    with pytest.raises(StreamBlocked, match="must be provided together"):
        build_candidate_stream(
            artifact_paths=artifacts,
            active_config=config,
            universe=universe,
            calendar=calendar,
            cache_dir=cache,
            output_dir=out,
            trading_root=trading,
            selector_checkpoint=checkpoint,
        )
    assert not out.exists()
