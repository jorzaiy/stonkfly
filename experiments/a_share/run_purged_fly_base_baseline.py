"""Research-only Fly/Base baseline for the purged WC006 candidate stream.

This entry point is intentionally a new namespace.  It accepts only the
hash-pinned purged candidate v2 and XYSZ market v6 bundle, runs the frozen
WC006+Fly replay in paper mode, and records the paired fixed-zero Base ledger
produced by the v5.1 engine.  It has no formal/live/shadow authorization path.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from stonkfly.config import Settings
from stonkfly.neural.sensory_adapter import NullSensoryAdapter, SensoryAdapter
from stonkfly.neural.v51_controller import V51Controller

from experiments.a_share.replay_wc006 import source_manifest_hashes
from experiments.a_share.wc006_male_cns_v4_adapter import (
    LOOKBACK,
    LOOKBACK_CLOSES,
    TRAIN_END,
    V4Stream,
    _load_price_rows,
    validate_candidate_stream,
)
from experiments.a_share.wc006_male_cns_v51_adapter import (
    _graph_source_lock,
    _run_v51,
)
import experiments.a_share.wc006_male_cns_v51_adapter as v51_adapter
from experiments.a_share.wc006_v4_market import (
    MarketInputsBlocked,
    XyszMarketInputs,
    load_xysz_market_inputs,
)

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_SHA256 = "9a07849c4b910390a949934f3329bad7288142cce2acbea7ed3835bd5f77d7bc"
MARKET_SHA256 = "bebf4922faef670d665da46d8aece75665a0ec090426997172a71ea26f33c39c"
# The parent market manifest is pinned independently from its QA report.
MARKET_PARENT_SHA256 = "bebf4922faef670d665da46d8aece75665a0ec090426997172a71ea26f33c39c"
MARKET_QA_SHA256 = "4fb438b60da507b443639746fdc11224da14239c5dedc0a8d4977ee358d267fe"
START_DATE = "2021-07-01"
END_DATE = "2023-12-29"
SETTLEMENT_DATE = "2024-01-02"
SEED = 17
RESEARCH_SCHEMA = "stonkfly-purged-fly-base-baseline/v1"
BASE_HYPOTHESIS = {
    "name": "fixed_zero_selector_independent_base",
    "definition": (
        "On each candidate date, Base independently holds the candidate stream's "
        "frozen target basket at multiplier 1.0; on non-candidate dates it emits "
        "no selector or neural action. It never reads Fly decoder output, neural "
        "state, reward, or fills. It shares the same raw market, costs, calendar, "
        "risk vetoes, corporate-action handling, and retry policy."
    ),
    "selector_source": "same hash-pinned candidate stream only",
    "neural_output_read": False,
    "profit_based_selection": False,
    "ledger": "paired_base.sqlite (independent ledger)",
    "fixed_multiplier": "1.0",
}


class PurgedBaselineBlocked(RuntimeError):
    """A research baseline input or resume gate failed closed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PurgedBaselineBlocked(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PurgedBaselineBlocked(f"JSON object required: {path}")
    return value


def _same_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise PurgedBaselineBlocked(f"{label} hash mismatch")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def validate_purged_candidate(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    _same_hash(path, CANDIDATE_SHA256, "purged candidate manifest")
    payload = _json(path)
    required = {
        "schema": "wc006-candidate-stream/v1",
        "paper_only": True,
        "broker_orders_enabled": False,
        "live_execution_authorized": False,
        "authoritative_trading_state": False,
        "signal_lag_one_day": True,
        "topk": 3,
        "hold": 20,
        "exact_tag": "5y_2021-01-01_2026-04-03",
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise PurgedBaselineBlocked(f"candidate {key} mismatch")
    if payload.get("authorization_artifact") or payload.get("market_authorization"):
        raise PurgedBaselineBlocked("legacy authorization cannot be reused")
    provenance = payload.get("selector_generation_provenance")
    if not isinstance(provenance, dict) or not provenance:
        raise PurgedBaselineBlocked("selector generation provenance is required")
    root = path.parent
    for name, item in provenance.items():
        if not isinstance(item, dict) or not isinstance(item.get("copied_path"), str):
            raise PurgedBaselineBlocked(f"selector provenance malformed: {name}")
        copied = (root / item["copied_path"]).resolve()
        try:
            copied.relative_to(root)
        except ValueError as exc:
            raise PurgedBaselineBlocked("selector provenance escapes candidate root") from exc
        if not copied.is_file() or sha256(copied) != item.get("sha256"):
            raise PurgedBaselineBlocked(f"selector provenance hash mismatch: {name}")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 46:
        raise PurgedBaselineBlocked("purged candidate must contain exactly 46 records")
    previous = None
    target_codes: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise PurgedBaselineBlocked(f"candidate record {index} malformed")
        signal, execution = str(record.get("signal_date")), str(record.get("execution_date"))
        if previous is not None and signal <= previous:
            raise PurgedBaselineBlocked("candidate signals are not strictly ordered")
        previous = signal
        if signal < START_DATE or signal > END_DATE:
            raise PurgedBaselineBlocked("candidate signal outside frozen window")
        selector = record.get("selector")
        timing = record.get("timing_proof")
        targets = selector.get("target_topk") if isinstance(selector, dict) else None
        if not isinstance(selector, dict) or not isinstance(targets, list) or len(targets) != 3:
            raise PurgedBaselineBlocked(f"candidate target_topk malformed on {signal}")
        if selector.get("signal_seed") != 42 or selector.get("prefer_tag") != payload["exact_tag"] or selector.get("rebalance_applied") is not True:
            raise PurgedBaselineBlocked(f"candidate selector provenance mismatch on {signal}")
        if not isinstance(timing, dict) or timing.get("signal_lag_one_day") is not True or timing.get("execution_is_first_next_valid_session") is not True:
            raise PurgedBaselineBlocked(f"candidate timing proof missing on {signal}")
        for rank, target in enumerate(targets, 1):
            if not isinstance(target, dict) or target.get("rank") != rank:
                raise PurgedBaselineBlocked(f"candidate rank mismatch on {signal}")
            code = str(target.get("code", ""))
            try:
                weight = Decimal(str(target["target_weight"]))
            except (KeyError, ValueError, ArithmeticError) as exc:
                raise PurgedBaselineBlocked(f"candidate weight malformed on {signal}") from exc
            if len(code) != 6 or not code.isdigit() or not weight.is_finite() or weight < 0:
                raise PurgedBaselineBlocked(f"candidate target malformed on {signal}")
            target_codes.add(code)
    if records[0]["signal_date"] != START_DATE or records[-1]["signal_date"] != END_DATE:
        raise PurgedBaselineBlocked("candidate date window is not frozen 2021-07-01..2023-12-29")
    if records[0]["execution_date"] != "2021-07-02" or records[-1]["execution_date"] != SETTLEMENT_DATE:
        raise PurgedBaselineBlocked("candidate settlement boundary mismatch")
    if len(target_codes) != 131:
        raise PurgedBaselineBlocked("candidate target union must contain 131 codes")
    return payload


def validate_purged_market(path: Path, candidate: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path).resolve()
    _same_hash(path, MARKET_PARENT_SHA256, "purged v6 market manifest")
    payload = _json(path)
    if payload.get("schema") != "wc006-purged-xysz-market/v2" or payload.get("status") != "PASS_RESEARCH_ONLY":
        raise PurgedBaselineBlocked("market is not PASS_RESEARCH_ONLY")
    for key, expected in {
        "paper_only": True,
        "research_only": True,
        "broker_orders_enabled": False,
        "live_execution_authorized": False,
        "shadow_execution_authorized": False,
        "training_authorized": False,
        "candidate_manifest_sha256": CANDIDATE_SHA256,
        "code_count": 131,
    }.items():
        if payload.get(key) != expected:
            raise PurgedBaselineBlocked(f"market {key} mismatch")
    if payload.get("authorization_artifact") or payload.get("market_authorization"):
        raise PurgedBaselineBlocked("market contains reusable authorization")
    market_codes = {str(code).split(".", 1)[0] for code in payload.get("codes", [])}
    candidate_codes = {str(target["code"]) for record in candidate["records"] for target in record["selector"]["target_topk"]}
    if market_codes != candidate_codes:
        raise PurgedBaselineBlocked("market/candidate code union mismatch")
    qa_path = path.parent / "quality_report.json"
    _same_hash(qa_path, MARKET_QA_SHA256, "purged v6 QA")
    qa = _json(qa_path)
    if qa.get("schema") != "wc006-purged-xysz-market/quality-report/v2" or qa.get("status") != "PASS_RESEARCH_ONLY":
        raise PurgedBaselineBlocked("market QA is not PASS_RESEARCH_ONLY")
    if qa.get("manifest_sha256") != MARKET_PARENT_SHA256 or qa.get("target_code_count") != 131 or qa.get("failed_checks"):
        raise PurgedBaselineBlocked("market QA binding or failed checks")
    if any(value is not True for value in (qa.get("checks") or {}).values()):
        raise PurgedBaselineBlocked("market QA contains a non-PASS check")
    if payload.get("raw_root") != "raw":
        raise PurgedBaselineBlocked("market raw_root must be raw")
    return payload, qa


def _research_market_loader(path: Path, *, expected_parent_sha256: str, expected_qa_sha256: str) -> XyszMarketInputs:
    """Load v6 raw artifacts through the legacy parser in an isolated mirror.

    v6 intentionally stores project-relative child manifests and includes the
    prior 125-code fragments as provenance.  The mirror filters only Kline
    artifacts to the 131-code v6 target set; every retained file remains a
    symlink to, and is hash-checked against, the pinned original.
    """
    path = Path(path).resolve()
    payload = _json(path)
    target_codes = {str(code).upper() for code in payload.get("codes", [])}
    with tempfile.TemporaryDirectory(prefix="stonkfly-purged-market-") as temp_name:
        mirror = Path(temp_name)
        children = []
        for child in payload.get("child_manifests", []):
            manifest_rel = child.get("manifest")
            if not isinstance(manifest_rel, str):
                raise MarketInputsBlocked("v6 child manifest path missing")
            original_manifest = (ROOT / manifest_rel).resolve()
            try:
                original_manifest.relative_to(ROOT)
            except ValueError as exc:
                raise MarketInputsBlocked("v6 child manifest escapes project") from exc
            if not original_manifest.is_file() or sha256(original_manifest) != child.get("manifest_sha256"):
                raise MarketInputsBlocked("v6 child manifest hash mismatch")
            original_payload = _json(original_manifest)
            dataset = str(child.get("dataset"))
            # v6 records old fragments for provenance plus fresh purged
            # collection fragments.  Action precedence is represented by the
            # fresh collection; retaining both would create false revisions.
            if dataset in {"dividend", "right_issue"} and "wc006_purged_xysz_collection" not in manifest_rel:
                continue
            retained = []
            temp_manifest = mirror / manifest_rel
            temp_manifest.parent.mkdir(parents=True, exist_ok=True)
            for artifact in original_payload.get("artifacts", []):
                if dataset == "kline" and str(artifact.get("code", "")).upper() not in target_codes:
                    continue
                source = (original_manifest.parent / str(artifact["path"])).resolve()
                if not source.is_file():
                    source = (original_manifest.parent / "raw" / str(artifact["path"])).resolve()
                if not source.is_file() or sha256(source) != artifact.get("sha256"):
                    raise MarketInputsBlocked("v6 artifact hash mismatch")
                target = temp_manifest.parent / str(artifact["path"])
                target.parent.mkdir(parents=True, exist_ok=True)
                copied = dict(artifact)
                if dataset != "kline":
                    with source.open(newline="", encoding="utf-8-sig") as source_handle:
                        reader = csv.DictReader(source_handle)
                        fields = list(reader.fieldnames or [])
                        rows = list(reader)
                    if "MARKET_CODE" in fields:
                        rows = [row for row in rows if str(row.get("MARKET_CODE", "")).strip().upper() in target_codes]
                        if dataset == "dividend":
                            keys = ("MARKET_CODE", "DATE_EQY_RECORD", "DATE_EX", "DATE_DVD_PAYOUT", "LISTINGDATE_OF_DVD_SHR", "ANN_DATE")
                            unique = {}
                            for row in rows:
                                key = tuple(str(row.get(name, "")) for name in keys)
                                unique[key] = max(unique.get(key, row), row, key=lambda item: _canonical(item))
                            rows = list(unique.values())
                        elif dataset == "right_issue":
                            keys = ("MARKET_CODE", "EXECUTE_DATE", "EX_DIVIDEND_DATE", "LISTED_DATE", "ANN_DATE")
                            unique = {}
                            for row in rows:
                                key = tuple(str(row.get(name, "")) for name in keys)
                                unique[key] = max(unique.get(key, row), row, key=lambda item: _canonical(item))
                            rows = list(unique.values())
                        with target.open("w", newline="", encoding="utf-8") as target_handle:
                            writer = csv.DictWriter(target_handle, fieldnames=fields)
                            writer.writeheader()
                            writer.writerows(rows)
                    else:
                        target.symlink_to(source)
                else:
                    target.symlink_to(source)
                copied["sha256"] = sha256(target)
                retained.append(copied)
            child_payload = dict(original_payload)
            child_payload["artifacts"] = retained
            temp_manifest.write_text(json.dumps(child_payload, sort_keys=True), encoding="utf-8")
            children.append({"dataset": dataset, "fragment": manifest_rel, "manifest_sha256": sha256(temp_manifest)})
        qa_source = path.parent / "quality_report.json"
        (mirror / "quality_report.json").symlink_to(qa_source)
        adapted = dict(payload)
        adapted["authoritative_trading_state"] = True
        adapted["source_manifest"] = {"codes": sorted(target_codes), "code_count": len(target_codes)}
        adapted["child_manifests"] = children
        temp_parent = mirror / "manifest.json"
        temp_parent.write_text(json.dumps(adapted, sort_keys=True), encoding="utf-8")
        bundle = load_xysz_market_inputs(temp_parent, expected_parent_sha256=sha256(temp_parent), expected_qa_sha256=expected_qa_sha256)
        return replace(bundle, parent_sha256=expected_parent_sha256, qa_sha256=expected_qa_sha256)


def _validate_stream(candidate_path: Path, market_path: Path) -> V4Stream:
    """Validate candidate lookback/calendar and attach raw research market."""
    base = validate_candidate_stream(candidate_path, expected_records=46)
    prices = _load_price_rows(base)
    first, final = base.records[0].signal_date, base.records[-1].execution_date
    required_dates = tuple(date for date in base.calendar if first <= date <= final)
    current = None
    signal_dates = {record.signal_date for record in base.records}
    for date in required_dates:
        if date in signal_dates:
            current = next(record for record in base.records if record.signal_date == date)
        if current is None:
            raise PurgedBaselineBlocked("observation before first candidate signal")
        history = tuple(item for item in base.calendar if item <= date)
        if len(history) < LOOKBACK_CLOSES:
            raise PurgedBaselineBlocked(f"{date}: missing 61-session lookback")
        for target in current.targets:
            code = target["code"]
            available = [item for item in prices.get(code, {}) if item <= date]
            if len(available) < LOOKBACK_CLOSES:
                raise PurgedBaselineBlocked(f"{code}: causal lookback missing on {date}")
    spec = ROOT / "docs/experiments/wc006_v4_2021_2025_preregister.md"
    if not spec.is_file():
        raise PurgedBaselineBlocked("v4 protocol source is missing")
    market = _research_market_loader(market_path, expected_parent_sha256=MARKET_PARENT_SHA256, expected_qa_sha256=MARKET_QA_SHA256)
    return V4Stream(base, prices, sha256(spec), market)


def _validate_graph_lock() -> dict[str, Any]:
    controller = V51Controller(Settings(learning=False))
    lock = _graph_source_lock(controller)
    if lock.get("full_retained_graph") is not True or lock.get("release") != "MaleCNS v1.0":
        raise PurgedBaselineBlocked("complete retained MaleCNS v1.0 graph is required")
    if "no graph cropping" not in str(lock.get("selection", "")):
        raise PurgedBaselineBlocked("graph pruning/cropping is not allowed")
    if lock.get("source_hashes") != source_manifest_hashes():
        raise PurgedBaselineBlocked("graph source lock drift")
    return lock


def _research_gate(candidate_path: Path, market_path: Path, output: Path, candidate: dict[str, Any], market: dict[str, Any], qa: dict[str, Any], graph_lock: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": RESEARCH_SCHEMA,
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
        "raw_price_contract": {"execution": "XYSZ raw Kline", "valuation": "XYSZ raw Kline", "reward_return": "raw XYSZ marks", "qfq_execution": False},
        "plasticity_mode": "frozen",
        "sensory_mode": "RGB-only; NullSensoryAdapter; no auxiliary stimulation",
        "initial_brain": "clean/new; no historical checkpoint restore",
        "seed": SEED,
        "paper_only": True,
        "broker_orders_enabled": False,
        "live_execution_authorized": False,
        "shadow_execution_authorized": False,
        "formal": False,
        "base_hypothesis": BASE_HYPOTHESIS,
        "output_scope": str(Path(output).resolve()),
    }


def run_purged_baseline(
    candidate_manifest: Path,
    market_manifest: Path,
    output: Path,
    *,
    max_days: int = 0,
    controller_factory: Callable[[Settings], Any] = V51Controller,
) -> dict[str, Any]:
    candidate_manifest, market_manifest, output = Path(candidate_manifest).resolve(), Path(market_manifest).resolve(), Path(output).resolve()
    runs_root = (ROOT / "runs").resolve()
    try:
        relative_output = output.relative_to(runs_root)
    except ValueError:
        relative_output = None
    if relative_output is not None and (not relative_output.parts or relative_output.parts[0] != "wc006_purged_fly_base_baseline_phase1"):
        raise PurgedBaselineBlocked("research output must use the new phase1 namespace")
    candidate = validate_purged_candidate(candidate_manifest)
    market, qa = validate_purged_market(market_manifest, candidate)
    graph_lock = _validate_graph_lock()
    if output.exists() and (output / "provenance.json").exists():
        prior = _json(output / "provenance.json")
        if prior.get("research_protocol", {}).get("candidate_manifest_sha256") != CANDIDATE_SHA256 or prior.get("research_protocol", {}).get("market_manifest_sha256") != MARKET_PARENT_SHA256:
            raise PurgedBaselineBlocked("resume input hashes differ")
        if prior.get("research_protocol", {}).get("plasticity_mode") != "frozen":
            raise PurgedBaselineBlocked("resume plasticity mode differs")
    stream = _validate_stream(candidate_manifest, market_manifest)
    gate = _research_gate(candidate_manifest, market_manifest, output, candidate, market, qa, graph_lock)
    # This call is deliberately hard-coded non-formal and frozen.  It cannot
    # consume the old V5 authorization path or write a formal namespace.
    # v5.1's old annual bridge intentionally knows only the old authorized
    # market schema. Inject this runner's explicit v6 loader for this call;
    # the module-global is restored immediately and no legacy path is widened.
    previous_validator = v51_adapter.validate_v4_stream
    def _validate_research_stream(path: Path, **kwargs: Any) -> V4Stream:
        supplied_market = kwargs.get("market_manifest")
        if supplied_market is None:
            raise PurgedBaselineBlocked("research baseline requires raw v6 market")
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
            seed=SEED,
            plasticity_mode="frozen",
            sensory_adapter=NullSensoryAdapter(),
            annual_input_gate={**gate, "stream_sha256": stream.base.stream_sha256, "spec_sha256": stream.spec_sha256},
            market_expected_parent_sha256=MARKET_PARENT_SHA256,
            market_expected_qa_sha256=MARKET_QA_SHA256,
        )
    finally:
        v51_adapter.validate_v4_stream = previous_validator
    provenance_path = output / "provenance.json"
    provenance = _json(provenance_path)
    provenance["research_protocol"] = gate
    provenance["base_hypothesis"] = BASE_HYPOTHESIS
    provenance["report_constraints"] = ["engineered reinforcement only", "no consciousness/pain/pleasure claim", "no validated profitable learning claim"]
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "schema": RESEARCH_SCHEMA + "/report",
        "status": "PASS_RESEARCH_ONLY" if result.get("status") == "complete" else "PAUSED_RESEARCH_ONLY",
        "result": result,
        "research_protocol": gate,
        "fly_ledger": "ledger.sqlite",
        "base_ledger": "paired_base.sqlite",
        "base_action_policy": BASE_HYPOTHESIS,
        "reinforcement_interpretation": "engineered input delivered to identified dopamine cells; no phenomenological claim",
    }
    (output / "baseline_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--market-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-days", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run_purged_baseline(args.candidate_manifest, args.market_manifest, args.out, max_days=args.max_days), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
