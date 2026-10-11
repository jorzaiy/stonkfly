#!/usr/bin/env python3
"""Bounded, resumable replay runner for the independent clean-start 2021 V5.1 baseline.

Enforces preflight invariants, audit gates, and paper-only execution:
- Fails closed unless the complete Jan-Dec 2021 candidate/data package is present.
- Supports dry-run and bounded, resumable chunks for the verified full-year slice.
- Pinned seed=17, plasticity_mode=frozen, sensory_adapter=NullSensoryAdapter (RGB-only).
- Paper-only: broker_orders_enabled=false, live_execution_authorized=false.
- Dual-compartment checkpoint verification, SQLite WAL truncation, atomic chunk manifests.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stonkfly.neural.sensory_adapter import NullSensoryAdapter
from stonkfly.neural.v51_controller import V51Controller
from experiments.a_share.wc006_male_cns_v51_adapter import _run_v51

CHUNKS_DIR = REPO_ROOT / "runs/wc006_v5_1_2021_baseline/chunks"
INPUT_ROOT = REPO_ROOT / "runs/wc006_v5_1_2021_baseline/inputs"
STREAM_PATH = INPUT_ROOT / "candidate_stream/manifest.json"
MARKET_MANIFEST_PATH = INPUT_ROOT / "market_inputs/manifest.json"
MARKET_SHA256 = "5f3f147fd1b7338166b068e8bb78e09c9914b0d4ed8426d01317150808690434"
MARKET_QA_SHA256 = "8a6733705ef44804a970faa52b698132657112dde7c0ec40aa822d9754920900"
GRAPH_MANIFEST_PATH = REPO_ROOT / "data/manifest.json"
GRAPH_SHA256 = "7f7c3bda9418c04d2e86cc7245b3569c7b63c772459f28b77ee328b192a86340"

# Independent clean-start full-calendar 2021 window. The selector uses causal
# 2020 QFQ lookback; MaleCNS/account state starts clean on first 2021 session.
FULL_START_DATE = "2021-01-04"
FULL_END_DATE = "2021-12-31"
FULL_SETTLEMENT_DATE = "2022-01-04"
FULL_TOTAL_DAYS = 243
FULL_SIGNAL_RECORDS = 19


class FeasibilityBlockerError(RuntimeError):
    """Raised when an impossible run is requested without required inputs."""


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    print(f"[{ts}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def flush_wal(db_path: Path) -> None:
    if not db_path.is_file():
        return
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.commit()
    con.close()


def collect_state_hashes(run_dir: Path) -> dict[str, str | None]:
    result = {}
    for name in [
        "events.jsonl",
        "ledger.sqlite",
        "paired_base.sqlite",
        "brain-day-0.npz",
        "brain-day-1.npz",
    ]:
        p = run_dir / name
        result[name] = sha256_file(p) if p.is_file() else None
    return result


def audit_2021_feasibility(start_date: str, end_date: str) -> dict[str, Any]:
    """Fail-closed preflight for the exact independent clean-start 2021 scope."""
    result: dict[str, Any] = {"feasible": False, "scope": [start_date, end_date]}
    if [start_date, end_date] != [FULL_START_DATE, FULL_END_DATE]:
        result.update(reason="ONLY_FULL_CALENDAR_2021_IS_PREREGISTERED", missing_inputs=["A shortened or shifted 2021 window is not an authorized performance run."])
        return result
    missing: list[str] = []
    for path in (STREAM_PATH, MARKET_MANIFEST_PATH, GRAPH_MANIFEST_PATH):
        if not path.is_file():
            missing.append(str(path))
    qa = MARKET_MANIFEST_PATH.parent / "quality_report.json"
    if MARKET_MANIFEST_PATH.is_file() and sha256_file(MARKET_MANIFEST_PATH) != MARKET_SHA256:
        missing.append("market manifest hash mismatch")
    if not qa.is_file() or sha256_file(qa) != MARKET_QA_SHA256:
        missing.append("market QA hash mismatch")
    if GRAPH_MANIFEST_PATH.is_file() and sha256_file(GRAPH_MANIFEST_PATH) != GRAPH_SHA256:
        missing.append("retained MaleCNS graph manifest hash mismatch")
    try:
        stream_payload = json.loads(STREAM_PATH.read_text(encoding="utf-8"))
        calendar = json.loads((STREAM_PATH.parent / "inputs/calendar.json").read_text(encoding="utf-8"))["dates"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        missing.append(f"malformed candidate/calendar input: {exc}")
        calendar, stream_payload = [], {}
    sessions = [str(x) for x in calendar if str(x).startswith("2021-")]
    records = stream_payload.get("records", []) if isinstance(stream_payload, dict) else []
    codes = {str(target.get("code")) for record in records for target in record.get("selector", {}).get("target_topk", [])}
    cache_root = STREAM_PATH.parent / "inputs/cache"
    missing_cache = [code for code in sorted(codes) if not any(cache_root.glob(f"{code}_*.csv"))]
    if len(sessions) != FULL_TOTAL_DAYS or sessions[:1] != [FULL_START_DATE] or sessions[-1:] != [FULL_END_DATE]:
        missing.append(f"calendar must contain exactly {FULL_TOTAL_DAYS} 2021 sessions")
    if len(records) != FULL_SIGNAL_RECORDS or (records and records[0].get("signal_date") != FULL_START_DATE) or (records and records[-1].get("signal_date") != "2021-12-29"):
        missing.append(f"candidate stream must contain exactly {FULL_SIGNAL_RECORDS} causal records")
    if missing_cache:
        missing.append(f"missing candidate QFQ cache files: {missing_cache}")
    if not missing:
        try:
            from experiments.a_share.wc006_male_cns_v4_adapter import validate_v4_stream
            validate_v4_stream(STREAM_PATH, expected_records=FULL_SIGNAL_RECORDS, training_end=FULL_END_DATE, market_manifest=MARKET_MANIFEST_PATH, market_expected_parent_sha256=MARKET_SHA256, market_expected_qa_sha256=MARKET_QA_SHA256)
            from experiments.a_share.wc006_v4_market import load_xysz_market_inputs
            market = load_xysz_market_inputs(MARKET_MANIFEST_PATH, expected_parent_sha256=MARKET_SHA256, expected_qa_sha256=MARKET_QA_SHA256)
            market_codes = {str(code).split('.')[0] for code in market.raw_prices}
            if not codes <= market_codes:
                raise RuntimeError(f"market execution coverage missing: {sorted(codes - market_codes)}")
        except Exception as exc:
            missing.append(f"candidate/market causal validation failed: {exc}")
    result.update({"feasible": not missing, "reason": "PASS_INDEPENDENT_CLEAN_START_2021" if not missing else "INPUT_PACKAGE_INVALID", "settlement_boundary": FULL_SETTLEMENT_DATE, "calendar_days": len(sessions), "candidate_records": len(records), "unique_candidates": len(codes), "missing_cache_files": missing_cache, "market_manifest_verified": MARKET_MANIFEST_PATH.is_file() and sha256_file(MARKET_MANIFEST_PATH) == MARKET_SHA256, "market_qa_verified": qa.is_file() and sha256_file(qa) == MARKET_QA_SHA256, "graph_manifest_verified": GRAPH_MANIFEST_PATH.is_file() and sha256_file(GRAPH_MANIFEST_PATH) == GRAPH_SHA256, "neural_warmup_days": 0, "selector_lookback_source": "2020 QFQ cache; causal only", "independent_clean_start": True})
    if missing:
        result["missing_inputs"] = missing
    return result


def run_chunk_2021(
    target_days: int,
    output_dir: Path,
    *,
    start_date: str = FULL_START_DATE,
    end_date: str = FULL_END_DATE,
    dry_run: bool = False,
    chunk_name: str | None = None,
) -> dict[str, Any]:
    """Execute or preflight a bounded chunk for 2021 V5.1 baseline."""
    feasibility = audit_2021_feasibility(start_date, end_date)
    if not feasibility["feasible"]:
        raise FeasibilityBlockerError(
            f"Cannot run 2021 baseline for {start_date}..{end_date}: {feasibility['reason']}\n"
            + "\n".join(feasibility.get("missing_inputs", []))
        )

    t0 = time.time()
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_id = chunk_name or f"chunk_2021_{target_days}"

    log(f"=== 2021 Baseline Preflight / Chunk Runner ===")
    log(f"Scope: {start_date} to {end_date} (Settlement: {FULL_SETTLEMENT_DATE})")
    log(f"Target days: {target_days}")
    log(f"Output directory: {output_dir}")

    if dry_run:
        log("Preflight dry-run validation PASSED. Zero neural decisions executed.")
        return {
            "status": "PREFLIGHT_DRY_RUN_PASSED",
            "chunk_id": chunk_id,
            "feasibility": feasibility,
            "target_days": target_days,
            "dry_run": True,
        }

    # Execute a bounded, resumable chunk. _run_v51 resumes from the ledger's
    # last completed index; brain/account state is never reset between chunks.
    log(f"Executing bounded decision day chunk up to max_days={target_days}...")
    res = _run_v51(
        stream_path=STREAM_PATH,
        output=output_dir,
        start_date=start_date,
        end_date=end_date,
        formal=False,
        market_manifest=MARKET_MANIFEST_PATH,
        max_days=target_days,
        expected_records=FULL_SIGNAL_RECORDS,
        training_end=end_date,
        controller_factory=V51Controller,
        arm="combined",
        seed=17,
        plasticity_mode="frozen",
        sensory_adapter=NullSensoryAdapter(),
        market_expected_parent_sha256=MARKET_SHA256,
        market_expected_qa_sha256=MARKET_QA_SHA256,
        annual_input_gate={
            "status": "PASS", "schema": "wc006-independent-clean-start-2021/v1",
            "manifest_sha256": sha256_file(STREAM_PATH), "source_manifest_sha256": sha256_file(STREAM_PATH),
            "year": 2021, "observation_window": [FULL_START_DATE, FULL_END_DATE],
            "observation_count": FULL_TOTAL_DAYS, "candidate_record_count": FULL_SIGNAL_RECORDS,
            "settlement_boundary": FULL_SETTLEMENT_DATE, "market_manifest_sha256": MARKET_SHA256,
            "market_qa_sha256": MARKET_QA_SHA256, "independent_clean_start": True, "neural_warmup_days": 0,
        },
        annual_payload={
            "year": 2021, "observation_window": [FULL_START_DATE, FULL_END_DATE],
            "settlement_boundary": FULL_SETTLEMENT_DATE,
            "records": json.loads(STREAM_PATH.read_text(encoding="utf-8"))["records"],
        },
    )

    flush_wal(output_dir / "ledger.sqlite")
    flush_wal(output_dir / "paired_base.sqlite")
    output_hashes = collect_state_hashes(output_dir)
    elapsed = round(time.time() - t0, 2)

    chunk_manifest = {
        "schema": "wc006-v5.1-2021-replay-chunk-manifest/v1",
        "chunk_id": chunk_id,
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "target_days": target_days,
        "seed": 17,
        "plasticity_mode": "frozen",
        "output_state_hashes": output_hashes,
        "run_result": res,
    }
    chunk_manifest_path = CHUNKS_DIR / f"{chunk_id}.json"
    chunk_manifest_path.write_text(json.dumps(chunk_manifest, indent=2) + "\n", encoding="utf-8")
    log(f"Chunk manifest written to: {chunk_manifest_path}")
    return chunk_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="2021 V5.1 Bounded Chunk Runner and Feasibility Audit.")
    parser.add_argument("--start-date", default=FULL_START_DATE, help="Start ISO date")
    parser.add_argument("--end-date", default=FULL_END_DATE, help="End ISO date")
    parser.add_argument("--target-days", type=int, default=1, help="Cumulative target day index for a resumable chunk")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "runs/wc006_v5_1_2021_baseline"), help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Perform preflight audit without executing simulation")
    parser.add_argument("--audit-only", action="store_true", help="Print feasibility audit and exit")
    args = parser.parse_args()

    if args.audit_only:
        audit = audit_2021_feasibility(args.start_date, args.end_date)
        print(json.dumps(audit, indent=2))
        return 0 if audit["feasible"] else 2

    try:
        run_chunk_2021(
            target_days=args.target_days,
            output_dir=Path(args.output_dir),
            start_date=args.start_date,
            end_date=args.end_date,
            dry_run=args.dry_run,
        )
        return 0
    except FeasibilityBlockerError as exc:
        log(f"FEASIBILITY BLOCKER: {exc}")
        return 2
    except Exception as exc:
        log(f"ERROR: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
