"""Locked, paper-only Fly/Base replay on the 2024–2025 purged continuation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from stonkfly.config import Settings
from stonkfly.neural.sensory_adapter import NullSensoryAdapter
from stonkfly.neural.v51_controller import V51Controller

from experiments.a_share.replay_wc006 import source_manifest_hashes
from experiments.a_share.wc006_male_cns_v4_adapter import (
    LOOKBACK_CLOSES, V4Stream, _load_price_rows, validate_candidate_stream,
)
from experiments.a_share.wc006_male_cns_v51_adapter import _graph_source_lock, _run_v51
import experiments.a_share.wc006_male_cns_v51_adapter as v51_adapter
from experiments.a_share.wc006_v4_market import load_xysz_market_inputs

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "fly-purged-heldout-2024-2025/v1"
START = "2024-01-02"
END = "2025-12-29"
SETTLE = "2025-12-30"
SEED = 17
MARKET_SCHEMA = "fly-purged-xysz-2024-2025-supplemented-market/v2"
MARKET_QA_SCHEMA = MARKET_SCHEMA + "/quality-report"
PARSER_PATH = ROOT / "experiments/a_share/wc006_v4_market.py"


class HeldoutBlocked(ValueError):
    pass


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def locked(path: Path, expected: str, name: str) -> dict[str, Any]:
    if not path.is_file() or sha(path) != expected:
        raise HeldoutBlocked(f"{name}_hash_mismatch")
    return json.loads(path.read_text())


def validate_market_policy(market: dict[str, Any], qa: dict[str, Any]) -> None:
    if market.get("schema") != MARKET_SCHEMA or qa.get("schema") != MARKET_QA_SCHEMA:
        raise HeldoutBlocked("supplemented_market_schema_mismatch")
    policy = market.get("dividend_target_policy")
    if not isinstance(policy, dict) or policy.get("ordinary_target") != "普通股股东" or policy.get("excluded_target") != "重整管理人" or policy.get("unknown_nonempty_target") != "FAIL_CLOSED" or policy.get("empty_target") != "legacy_compatible":
        raise HeldoutBlocked("dividend_target_policy_mismatch")
    if policy.get("parser_source_sha256") != sha(PARSER_PATH):
        raise HeldoutBlocked("dividend_parser_source_hash_mismatch")
    rows = policy.get("excluded_source_rows")
    if policy.get("excluded_row_count") != 1 or not isinstance(rows, list) or len(rows) != 1 or not all(isinstance(x, dict) and len(str(x.get("row_sha256", ""))) == 64 for x in rows):
        raise HeldoutBlocked("dividend_excluded_row_evidence_mismatch")
    if qa.get("dividend_target_policy") != policy:
        raise HeldoutBlocked("dividend_policy_qa_binding_mismatch")


def validate(candidate_path: Path, market_path: Path, qa_path: Path,
             candidate_sha: str, market_sha: str, qa_sha: str) -> tuple[dict, V4Stream, dict]:
    candidate = locked(candidate_path, candidate_sha, "candidate")
    market = locked(market_path, market_sha, "market")
    qa = locked(qa_path, qa_sha, "market_qa")
    required = {"schema": "wc006-candidate-stream/v1", "paper_only": True,
                "broker_orders_enabled": False, "live_execution_authorized": False,
                "authoritative_trading_state": False, "signal_lag_one_day": True,
                "topk": 3, "hold": 20, "exact_tag": "5y_2021-01-01_2026-04-03"}
    if any(candidate.get(k) != v for k, v in required.items()):
        raise HeldoutBlocked("candidate_protocol_mismatch")
    records = candidate.get("records")
    if not isinstance(records, list) or len(records) != 36 or records[0]["signal_date"] != START or records[-1]["signal_date"] != END or records[-1]["execution_date"] != SETTLE:
        raise HeldoutBlocked("candidate_date_schedule_mismatch")
    validate_market_policy(market, qa)
    if market.get("status") != "PASS_RESEARCH_ONLY":
        raise HeldoutBlocked("market_qa_not_pass")
    if market.get("candidate_manifest_sha256") != candidate_sha or qa.get("manifest_sha256") != market_sha or qa.get("status") != "PASS_RESEARCH_ONLY" or qa.get("issues"):
        raise HeldoutBlocked("market_binding_mismatch")
    if any(v is not True for v in qa.get("checks", {}).values()):
        raise HeldoutBlocked("market_checks_not_pass")
    if market.get("paper_only") is not True or market.get("live_execution_authorized") is not False or market.get("broker_orders_enabled") is not False:
        raise HeldoutBlocked("market_not_paper_only")
    codes = {str(t["code"]) for r in records for t in r["selector"]["target_topk"]}
    if codes != {str(x).split(".")[0] for x in market["source_manifest"]["codes"]}:
        raise HeldoutBlocked("candidate_market_code_mismatch")
    base = validate_candidate_stream(candidate_path, expected_records=36,
                                     require_selector_generation_provenance=True)
    prices = _load_price_rows(base)
    dates = tuple(d for d in base.calendar if START <= d <= SETTLE)
    signal_map = {r.signal_date: r for r in base.records}
    current = None
    for d in dates:
        current = signal_map.get(d, current)
        if current is None:
            raise HeldoutBlocked("observation_before_first_signal")
        for target in current.targets:
            if sum(x <= d for x in prices[target["code"]]) < LOOKBACK_CLOSES:
                raise HeldoutBlocked("causal_qfq_lookback_missing")
    raw = load_xysz_market_inputs(market_path, expected_parent_sha256=market_sha, expected_qa_sha256=qa_sha)
    spec = ROOT / "docs/experiments/fly_2024_2025_heldout_protocol.md"
    stream = V4Stream(base, prices, sha(spec), raw)
    return candidate, stream, qa


def run(lock_path: Path, output: Path, max_days: int = 0, *,
        controller_factory: Callable[[Settings], Any] = V51Controller,
        plasticity_mode: str = "frozen",
        initial_brain: Path | None = None) -> dict:
    lock_path = lock_path.resolve()
    lock = json.loads(lock_path.read_text())
    if lock.get("schema") != SCHEMA + "/input-lock" or lock.get("paper_only") is not True:
        raise HeldoutBlocked("invalid_input_lock")
    protocol = ROOT / "docs/experiments/fly_2024_2025_heldout_protocol.md"
    if lock.get("protocol") != "docs/experiments/fly_2024_2025_heldout_protocol.md" or sha(protocol) != lock.get("protocol_sha256"):
        raise HeldoutBlocked("protocol_hash_mismatch")
    candidate_path = Path(lock["candidate_manifest"]).resolve()
    market_path = Path(lock["market_manifest"]).resolve()
    qa_path = market_path.parent / "quality_report.json"
    output = output.resolve()
    allowed = ROOT / "runs/fly_purged_heldout_2024_2025"
    if allowed not in output.parents:
        raise HeldoutBlocked("heldout_output_namespace_required")
    candidate, stream, qa = validate(candidate_path, market_path, qa_path,
                                     lock["candidate_sha256"], lock["market_sha256"], lock["qa_sha256"])
    graph = _graph_source_lock(V51Controller(Settings(learning=False)))
    if graph.get("full_retained_graph") is not True or graph.get("release") != "MaleCNS v1.0" or graph.get("source_hashes") != source_manifest_hashes():
        raise HeldoutBlocked("full_graph_source_lock_mismatch")
    initial_brain_desc = "clean/new; no historical checkpoint restore"
    initial_brain_sha = None
    if initial_brain is not None:
        initial_brain_desc = f"pretrained weights from {initial_brain.name}"
        initial_brain_sha = sha(initial_brain)
    gate = {"schema": SCHEMA, "status": "PASS_RESEARCH_ONLY", "paper_only": True,
            "broker_orders_enabled": False, "live_execution_authorized": False,
            "shadow_execution_authorized": False, "formal": False,
            "candidate_manifest_sha256": lock["candidate_sha256"],
            "market_manifest_sha256": lock["market_sha256"], "market_qa_sha256": lock["qa_sha256"],
            "input_lock_sha256": sha(lock_path), "graph_source_lock": graph,
            "plasticity_mode": plasticity_mode, "seed": SEED,
            "initial_brain": initial_brain_desc,
            "initial_brain_sha256": initial_brain_sha,
            "sensory_mode": "RGB-only; NullSensoryAdapter; no auxiliary stimulation",
            "date_window": [START, END], "settlement_boundary": SETTLE,
            "candidate_record_count": 36, "target_code_count": len(stream.base.cache_paths),
            "market_qa_status": qa["status"],
            "base_hypothesis": "independent fixed selector basket multiplier 1.0",
            "raw_price_contract": {"execution": "XYSZ raw Kline", "valuation": "XYSZ raw Kline", "reward_return": "raw XYSZ marks"}}
    previous = v51_adapter.validate_v4_stream
    def research_stream(path: Path, **kwargs: Any) -> V4Stream:
        if Path(path).resolve() != candidate_path or Path(kwargs.get("market_manifest", "")).resolve() != market_path:
            raise HeldoutBlocked("replay_input_path_mismatch")
        return stream
    v51_adapter.validate_v4_stream = research_stream
    try:
        result = _run_v51(candidate_path, output, start_date=START, end_date=END,
                          formal=False, market_manifest=market_path,
                          max_days=max_days, expected_records=36, training_end=END,
                          controller_factory=controller_factory, arm="combined", seed=SEED,
                          plasticity_mode=plasticity_mode, sensory_adapter=NullSensoryAdapter(),
                          annual_input_gate={**gate, "stream_sha256": stream.base.stream_sha256,
                                             "spec_sha256": stream.spec_sha256},
                          market_expected_parent_sha256=lock["market_sha256"],
                          market_expected_qa_sha256=lock["qa_sha256"])
    finally:
        v51_adapter.validate_v4_stream = previous
    provenance_path = output / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["research_protocol"] = gate
    provenance["report_constraints"] = ["engineered reinforcement only", "no consciousness or felt pain/pleasure claim", "no validated profitable learning claim"]
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    report = {"schema": SCHEMA + "/report", "status": "PASS_RESEARCH_ONLY" if result.get("status") == "complete" else "PAUSED_RESEARCH_ONLY",
              "result": result, "research_protocol": gate,
              "fly_ledger": "ledger.sqlite", "base_ledger": "paired_base.sqlite"}
    (output / "heldout_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return report


def _make_controller_factory(decoder_type: str, threshold_hz: float = 2.0, initial_brain: Path | None = None):
    """Create a controller factory with specified decoder type and threshold.
    
    If initial_brain is provided, weights will be loaded after controller creation.
    """
    def factory(settings: Settings):
        updated = Settings(
            learning=settings.learning,
            decoder_type=decoder_type,
            decoder_threshold_hz=threshold_hz,
        )
        controller = V51Controller(updated)
        if initial_brain is not None:
            controller.restore_weights_only(initial_brain)
        return controller
    return factory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument(
        "--decoder-type",
        type=str,
        default="dnp20",
        choices=["dnp20", "mbon", "mbon_ratio"],
        help="Decoder type: 'dnp20' (original) or 'mbon'/'mbon_ratio' (plasticity-connected)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="MBON decoder threshold in Hz (default: 2.0)"
    )
    parser.add_argument(
        "--plasticity-mode",
        type=str,
        default="frozen",
        choices=["frozen", "enabled"],
        help="Plasticity mode: 'frozen' (no learning) or 'enabled' (continue learning)"
    )
    parser.add_argument(
        "--initial-brain",
        type=Path,
        default=None,
        help="Load pre-trained brain checkpoint (weights only)"
    )
    args = parser.parse_args()
    
    # Create controller factory with initial brain support
    controller_factory: Callable[[Settings], Any]
    if args.decoder_type != "dnp20" or args.initial_brain is not None:
        controller_factory = _make_controller_factory(
            args.decoder_type, args.threshold, args.initial_brain
        )
    else:
        controller_factory = V51Controller
    
    print(json.dumps(run(args.input_lock, args.out, args.max_days, 
                         controller_factory=controller_factory,
                         plasticity_mode=args.plasticity_mode,
                         initial_brain=args.initial_brain), 
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
