#!/usr/bin/env python3
"""Build the fixed, non-performance V5.1 mechanism-assay cue.

The cue is intentionally locked to the preregistered first 2023 observation
(``2023-01-03``), clean cash/no holdings, and that day's already-frozen
candidate basket.  It calls the existing V4 ``_frame`` renderer and then runs
one sensory-only V5.1 observation.  No market replay, selection, reward, or
authorization is created here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.a_share.replay_wc006 import D, WC006Ledger, sha256, source_manifest_hashes
from experiments.a_share.wc006_male_cns_adapter import StreamRecord, ValidatedStream
from experiments.a_share.wc006_male_cns_v4_adapter import _frame, _load_price_rows
from experiments.a_share.wc006_v5_1_v6_mechanism_assay import _graph_contract
from stonkfly.config import Settings
from stonkfly.neural.v51_controller import V51Controller


ROOT = Path(__file__).resolve().parents[2]
ANNUAL_INPUT = ROOT / "runs/wc006_v5_1_inputs/annual-2023-manifest-draft"
ANNUAL_MANIFEST = ANNUAL_INPUT / "manifest.json"
SOURCE_MANIFEST = ROOT / "runs/wc006_v3_2021_inputs/20260918T-candidate-full-2021-2023-stateful/manifest.json"
START_DATE = "2023-01-03"
EXPECTED_SOURCE_MANIFEST_SHA256 = "5874b6cefab5f1c4b1ee04db819fd76430cdf67495db67db5242f5d83f42cb52"
EXPECTED_CODES = ("001308", "002244", "002455")
SEED = 17
NEURAL_MS = 500.0


def digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _build(output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output must be a new directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    annual = _read_json(ANNUAL_MANIFEST)
    if annual.get("draft") is not True or annual.get("authorization_artifact") is not False:
        raise ValueError("annual input must remain a draft without authorization")
    if annual.get("year") != 2023 or annual.get("observation_window") != [START_DATE, "2023-12-29"]:
        raise ValueError("annual input window is not the preregistered 2023 window")
    if annual.get("candidate_selection_frozen") is not True:
        raise ValueError("candidate selection is not explicitly frozen")
    if sha256(SOURCE_MANIFEST) != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise ValueError("pinned source candidate manifest hash mismatch")
    records = annual.get("records")
    if not isinstance(records, list) or not records or records[0].get("signal_date") != START_DATE:
        raise ValueError("first annual record is not the fixed 2023-01-03 observation")
    first = records[0]
    selector = first.get("selector")
    if not isinstance(selector, dict):
        raise ValueError("first annual selector record is missing")
    rows = selector.get("target_topk")
    if not isinstance(rows, list) or tuple(str(row.get("code")) for row in rows) != EXPECTED_CODES:
        raise ValueError("first annual candidate basket differs from the pinned basket")
    targets = tuple({"code": str(row["code"]), "weight": float(row["target_weight"]), "score": row.get("score")} for row in rows)
    if abs(sum(item["weight"] for item in targets) - 1.0) > 1e-9:
        raise ValueError("fixed candidate weights do not sum to one")

    calendar_payload = _read_json(ANNUAL_INPUT / "inputs/calendar.json")
    calendar = tuple(str(item) for item in calendar_payload.get("dates", []))
    if START_DATE not in calendar:
        raise ValueError("fixed cue date is absent from pinned calendar")
    cache_files = annual.get("cache_files")
    if not isinstance(cache_files, dict):
        raise ValueError("annual cache provenance is missing")
    cache_paths: dict[str, Path] = {}
    cache_hashes: dict[str, str] = {}
    for code in EXPECTED_CODES:
        item = cache_files.get(code)
        if not isinstance(item, dict):
            raise ValueError(f"missing cache provenance for {code}")
        path = ANNUAL_INPUT / str(item["copied_path"])
        if not path.is_file() or sha256(path) != str(item["sha256"]):
            raise ValueError(f"cache hash mismatch for {code}")
        cache_paths[code] = path
        cache_hashes[code] = sha256(path)
    # Reuse the established V4 price-loader and frame renderer.  The minimal
    # validated stream below contains only the fixed day's already-pinned
    # basket; it is not a selector or replay stream.
    record = StreamRecord(START_DATE, str(first["execution_date"]), targets)
    stream = ValidatedStream(SOURCE_MANIFEST, {}, (record,), calendar, cache_paths, EXPECTED_SOURCE_MANIFEST_SHA256)
    prices = _load_price_rows(stream)
    ledger = WC006Ledger(
        output / "clean-cash-ledger.sqlite",
        initial_state={"cash": "100000", "lots": []},
        signature=hashlib.sha256(b"v51-fixed-cue-clean-cash-v1").hexdigest(),
        authoritative_trading_state=False,
    )
    try:
        if str(ledger.get_meta("cash")) != "100000" or ledger.db.execute("SELECT COUNT(*) FROM lots").fetchone()[0] != 0:
            raise ValueError("ledger is not clean cash/no holdings")
        frame = _frame(record, prices, START_DATE, calendar, ledger, D("1.0"), D("0"))
    finally:
        ledger.close()
    frame_path = output / "frame.npy"
    np.save(frame_path, frame, allow_pickle=False)
    png_path = output / "frame.png"
    Image.fromarray(frame, mode="RGB").save(png_path)

    settings = Settings(neural_ms=NEURAL_MS, learning=True)
    controller = V51Controller(settings)
    clean_path = output / "clean-before-sensory.npz"
    controller.save(clean_path)
    sensory = controller.observe(frame, "none")
    reward_mask = controller._linked_eligibility_mask("reward")
    aversive_mask = controller._linked_eligibility_mask("aversive")
    masks_path = output / "eligible-masks.npz"
    np.savez_compressed(masks_path, reward=reward_mask, aversive=aversive_mask)
    after_path = output / "after-sensory-only.npz"
    controller.save(after_path)

    metadata = {
        "schema": "wc006-v5.1-fixed-cue-input/v1",
        "status": "INCONCLUSIVE_NO_ELIGIBILITY" if not np.any(reward_mask) or not np.any(aversive_mask) else "ELIGIBILITY_OBSERVED",
        "purpose": "fixed non-performance mechanism-assay cue input",
        "year": 2023,
        "observation_date": START_DATE,
        "execution_date": str(first["execution_date"]),
        "seed": SEED,
        "seed_role": "reproducibility only; not a cue or selector choice",
        "clean_state": {"cash": "100000", "holdings": [], "multiplier": "1.0", "drawdown": "0"},
        "candidate_basket": [{"code": item["code"], "weight": item["weight"], "score": item["score"]} for item in targets],
        "frame": {"shape": list(frame.shape), "dtype": str(frame.dtype), "raw_sha256": digest(frame), "npy_sha256": sha256(frame_path), "png_sha256": sha256(png_path)},
        "sensory_only": {"reinforcement": "none", "duration_ms": NEURAL_MS, "input_sha256": str(sensory["input_sha256"]), "brain_ms_after": sensory["brain_ms"], "kc_spikes": sensory["KC_spikes"], "reward_spikes": sensory["reward_spikes"], "aversive_spikes": sensory["aversive_spikes"]},
        "eligible_edge_counts": {"reward": int(np.count_nonzero(reward_mask)), "aversive": int(np.count_nonzero(aversive_mask))},
        "graph": _graph_contract(controller.brain),
        "artifacts": {"clean_before_sensory": {"file": clean_path.name, "sha256": sha256(clean_path)}, "after_sensory_only": {"file": after_path.name, "sha256": sha256(after_path)}, "eligible_masks": {"file": masks_path.name, "sha256": sha256(masks_path)}},
        "source_manifest_hashes": {"annual_input_manifest": sha256(ANNUAL_MANIFEST), "candidate_stream_manifest": EXPECTED_SOURCE_MANIFEST_SHA256, "annual_calendar": sha256(ANNUAL_INPUT / "inputs/calendar.json"), "selector_artifact": str(selector["artifact_sha256"]), "candidate_cache": cache_hashes, "neural_source_manifests": source_manifest_hashes()},
        "paper_only": True,
        "market_replay": False,
        "broker_orders_enabled": False,
        "live_execution_authorized": False,
        "authorization_created": False,
    }
    (output / "build-metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(_build(args.output.resolve()), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
