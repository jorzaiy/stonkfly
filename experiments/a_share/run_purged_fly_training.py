"""Research-only Fly training with enabled plasticity for the purged WC006 candidate stream.

This entry point enables KC-MBON plasticity so Fly can learn from reward/aversive
signals. It uses the same hash-pinned purged candidate v2 and XYSZ market v6 bundle
as the frozen baseline, but with plasticity_mode="enabled".

Key differences from frozen baseline:
- plasticity_mode="enabled" allows KC-MBON07/11 synaptic weight changes
- PAM11 (reward) and PPL101 (aversive) dopamine signals modify weights
- The centered rate rule updates 7,835 plastic edges based on P&L feedback

This is paper-only research. It does not claim profitable learning or consciousness.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from stonkfly.config import Settings
from stonkfly.neural.sensory_adapter import NullSensoryAdapter
from stonkfly.neural.v51_controller import V51Controller

from experiments.a_share.run_purged_fly_base_baseline import (
    ROOT,
    CANDIDATE_SHA256,
    MARKET_PARENT_SHA256,
    MARKET_QA_SHA256,
    START_DATE,
    END_DATE,
    SETTLEMENT_DATE,
    SEED,
    BASE_HYPOTHESIS,
    PurgedBaselineBlocked,
    sha256,
    _json,
    validate_purged_candidate,
    validate_purged_market,
    _validate_graph_lock,
    _validate_stream,
    _research_market_loader,
)
from experiments.a_share.wc006_male_cns_v51_adapter import _run_v51
import experiments.a_share.wc006_male_cns_v51_adapter as v51_adapter
from experiments.a_share.wc006_male_cns_v4_adapter import V4Stream

TRAINING_SCHEMA = "stonkfly-purged-fly-training/v1"


def _training_gate(
    candidate_path: Path,
    market_path: Path,
    output: Path,
    candidate: dict[str, Any],
    market: dict[str, Any],
    qa: dict[str, Any],
    graph_lock: dict[str, Any],
) -> dict[str, Any]:
    """Build research gate for enabled plasticity training."""
    return {
        "schema": TRAINING_SCHEMA,
        "status": "PASS_RESEARCH_ONLY",
        "candidate_manifest_sha256": CANDIDATE_SHA256,
        "market_manifest_sha256": MARKET_PARENT_SHA256,
        "market_qa_sha256": MARKET_QA_SHA256,
        "selector_generation_provenance_required": True,
        "selector_generation_provenance": candidate["selector_generation_provenance"],
        "date_window": [START_DATE, END_DATE],
        "settlement_boundary": SETTLEMENT_DATE,
        "candidate_record_count": len(candidate["records"]),
        "target_code_count": 131,
        "market_status": market["status"],
        "market_qa_status": qa["status"],
        "graph_source_lock": graph_lock,
        "raw_price_contract": {
            "execution": "XYSZ raw Kline",
            "valuation": "XYSZ raw Kline",
            "reward_return": "raw XYSZ marks",
            "qfq_execution": False,
        },
        # Key difference: enabled plasticity
        "plasticity_mode": "enabled",
        "sensory_mode": "RGB-only; NullSensoryAdapter; no auxiliary stimulation",
        "initial_brain": "clean/new; no historical checkpoint restore",
        "seed": SEED,
        "paper_only": True,
        "broker_orders_enabled": False,
        "live_execution_authorized": False,
        "shadow_execution_authorized": False,
        "formal": False,
        "training_authorized": True,  # Enable training
        "base_hypothesis": BASE_HYPOTHESIS,
        "output_scope": str(Path(output).resolve()),
        "training_note": (
            "KC-MBON plasticity is enabled. PAM11 reward and PPL101 aversive "
            "dopamine signals modify synaptic weights via the centered rate rule. "
            "This is an experimental mechanism assay, not a profitability claim."
        ),
    }


def run_purged_training(
    candidate_manifest: Path,
    market_manifest: Path,
    output: Path,
    *,
    max_days: int = 0,
    controller_factory: Callable[[Settings], Any] = V51Controller,
    seed: int = SEED,
) -> dict[str, Any]:
    """Run Fly training with enabled plasticity.
    
    This allows the KC-MBON connections to be modified by reward/aversive
    feedback, potentially enabling Fly to learn from trading outcomes.
    """
    candidate_manifest = Path(candidate_manifest).resolve()
    market_manifest = Path(market_manifest).resolve()
    output = Path(output).resolve()
    
    runs_root = (ROOT / "runs").resolve()
    try:
        relative_output = output.relative_to(runs_root)
    except ValueError:
        relative_output = None
    
    # Use a separate namespace for training runs
    if relative_output is not None and (
        not relative_output.parts
        or relative_output.parts[0] != "wc006_purged_fly_training"
    ):
        raise PurgedBaselineBlocked(
            "training output must use wc006_purged_fly_training namespace"
        )
    
    candidate = validate_purged_candidate(candidate_manifest)
    market, qa = validate_purged_market(market_manifest, candidate)
    graph_lock = _validate_graph_lock()
    
    # Check resume compatibility - must match enabled plasticity
    if output.exists() and (output / "provenance.json").exists():
        prior = _json(output / "provenance.json")
        prior_protocol = prior.get("research_protocol", {})
        if prior_protocol.get("candidate_manifest_sha256") != CANDIDATE_SHA256:
            raise PurgedBaselineBlocked("resume candidate hash differs")
        if prior_protocol.get("market_manifest_sha256") != MARKET_PARENT_SHA256:
            raise PurgedBaselineBlocked("resume market hash differs")
        if prior_protocol.get("plasticity_mode") != "enabled":
            raise PurgedBaselineBlocked(
                "resume plasticity mode differs (expected enabled)"
            )
    
    stream = _validate_stream(candidate_manifest, market_manifest)
    gate = _training_gate(
        candidate_manifest, market_manifest, output, candidate, market, qa, graph_lock
    )
    
    # Inject research stream validator
    previous_validator = v51_adapter.validate_v4_stream

    def _validate_research_stream(path: Path, **kwargs: Any) -> V4Stream:
        supplied_market = kwargs.get("market_manifest")
        if supplied_market is None:
            raise PurgedBaselineBlocked("training requires raw v6 market")
        return _validate_stream(Path(path), Path(supplied_market))

    v51_adapter.validate_v4_stream = _validate_research_stream
    
    try:
        result = _run_v51(
            candidate_manifest,
            output,
            start_date=START_DATE,
            end_date=END_DATE,
            formal=False,
            market_manifest=market_manifest,
            max_days=max_days,
            expected_records=46,
            training_end=END_DATE,
            controller_factory=controller_factory,
            arm="combined",
            seed=seed,
            # Key difference: enabled plasticity
            plasticity_mode="enabled",
            sensory_adapter=NullSensoryAdapter(),
            annual_input_gate={
                **gate,
                "stream_sha256": stream.base.stream_sha256,
                "spec_sha256": stream.spec_sha256,
            },
            market_expected_parent_sha256=MARKET_PARENT_SHA256,
            market_expected_qa_sha256=MARKET_QA_SHA256,
        )
    finally:
        v51_adapter.validate_v4_stream = previous_validator
    
    # Update provenance
    provenance_path = output / "provenance.json"
    provenance = _json(provenance_path)
    provenance["research_protocol"] = gate
    provenance["base_hypothesis"] = BASE_HYPOTHESIS
    provenance["report_constraints"] = [
        "engineered reinforcement only",
        "no consciousness/pain/pleasure claim",
        "no validated profitable learning claim",
        "plasticity enabled but profitability not demonstrated",
    ]
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    
    # Build report
    report = {
        "schema": TRAINING_SCHEMA + "/report",
        "status": (
            "PASS_RESEARCH_ONLY"
            if result.get("status") == "complete"
            else "PAUSED_RESEARCH_ONLY"
        ),
        "result": result,
        "research_protocol": gate,
        "fly_ledger": "ledger.sqlite",
        "base_ledger": "paired_base.sqlite",
        "base_action_policy": BASE_HYPOTHESIS,
        "plasticity_mode": "enabled",
        "reinforcement_interpretation": (
            "engineered input delivered to identified dopamine cells; "
            "KC-MBON weights may change; no phenomenological or profitability claim"
        ),
    }
    (output / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    
    return report


from dataclasses import replace


def _make_controller_factory(decoder_type: str, threshold_hz: float = 2.0):
    """Create a controller factory with specified decoder type and threshold."""
    def factory(settings: Settings):
        # Preserve every setting supplied by the adapter. Reconstructing a
        # partial Settings object silently resets timing and pulse parameters.
        updated = replace(
            settings,
            decoder_type=decoder_type,
            decoder_threshold_hz=threshold_hz,
        )
        return V51Controller(updated)
    return factory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--market-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument(
        "--decoder-type",
        type=str,
        default="dnp20",
        choices=["dnp20", "mbon"],
        help="Decoder type: 'dnp20' (original, MBON-disconnected) or 'mbon' (plasticity-connected)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED})"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="MBON decoder threshold in Hz (default: 2.0)"
    )
    args = parser.parse_args()
    
    # Use custom controller factory with decoder_type and threshold
    controller_factory = _make_controller_factory(args.decoder_type, args.threshold)
    
    print(
        json.dumps(
            run_purged_training(
                args.candidate_manifest,
                args.market_manifest,
                args.out,
                max_days=args.max_days,
                controller_factory=controller_factory,
                seed=args.seed,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
