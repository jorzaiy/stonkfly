#!/usr/bin/env python3
"""Fail-closed preflight for the pinned V5 development-only three-arm run."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from experiments.a_share.wc006_male_cns_v5_adapter import (
    V5_ARMS,
    _formal_output_scope,
    audit_v5_isolation,
    audit_v5_resume_parity,
    _graph_source_lock,
    validate_v5_inputs,
)
from stonkfly.config import Settings
from stonkfly.neural.controller import FlyController

TRAIN_END = "2023-12-29"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--market-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.seed != 17:
        raise SystemExit("fixed V5 development seed must be 17")
    if args.run_id in {"", ".", ".."} or "/" in args.run_id or "\\" in args.run_id:
        raise SystemExit("run-id must be a single path component")

    manifest = args.manifest.resolve()
    market = args.market_manifest.resolve()
    root = args.output_root.resolve()
    stream, input_gate = validate_v5_inputs(
        manifest,
        expected_records=46,
        training_end=TRAIN_END,
        market_manifest=market,
        formal=True,
    )
    if input_gate["training_authorized"] is not True:
        raise SystemExit("input gate did not authorize formal V5")
    if input_gate["paper_only"] is not True or input_gate["broker_orders_enabled"] is not False or input_gate["live_execution_authorized"] is not False:
        raise SystemExit("input gate is not paper-only")
    expected_graph = (input_gate.get("authorization_code_lock") or {}).get("graph_source_lock") or {}
    actual_graph = _graph_source_lock(FlyController(Settings(learning=True)))
    graph_keys = ("full_retained_graph", "release", "selection", "configuration_sha256", "plastic_edges", "plastic_edges_sha256", "source_hashes")
    if any(actual_graph.get(key) != expected_graph.get(key) for key in graph_keys):
        raise SystemExit("graph source lock preflight failed")

    outputs = []
    for arm in V5_ARMS:
        output = root / f"wc006_v5_{arm}" / args.run_id
        _formal_output_scope(output, arm)
        if args.resume:
            if not output.is_dir() or not (output / "ledger.sqlite").is_file():
                raise SystemExit(f"resume output is missing ledger: {output}")
            if (output / "QUARANTINED.json").exists():
                raise SystemExit(f"resume blocked by quarantine: {output}")
            provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
            if provenance.get("seed") != args.seed or provenance.get("arm") != arm:
                raise SystemExit(f"resume seed/arm mismatch: {output}")
            isolation = audit_v5_isolation(output, arm=arm, formal=True)
            if isolation.get("status") != "PASS":
                raise SystemExit(f"resume isolation gate failed: {output}: {isolation}")
            with sqlite3.connect(output / "ledger.sqlite") as db:
                unknown = db.execute("SELECT COUNT(*) FROM intents WHERE status='UNKNOWN'").fetchone()[0]
                pending = db.execute("SELECT COUNT(*) FROM intents WHERE status='PENDING'").fetchone()[0]
            if unknown or pending:
                raise SystemExit(f"resume blocked by UNKNOWN/PENDING: {output} UNKNOWN={unknown} PENDING={pending}")
        elif output.exists():
            raise SystemExit(f"refusing existing arm output: {output}")
        outputs.append(str(output))

    # These are the actual APIs used by post-run gates.  Their presence is a
    # preflight requirement; a parity result is required when a resume pair is
    # supplied, while a fresh run cannot truthfully claim parity in advance.
    api_gate = {
        "resume_parity_api": callable(audit_v5_resume_parity),
        "isolation_api": callable(audit_v5_isolation),
        "input_gate": input_gate["status"] == "PASS",
    }
    if not all(api_gate.values()):
        raise SystemExit(f"preflight API gate failed: {api_gate}")
    print(json.dumps({
        "status": "PASS",
        "seed": args.seed,
        "resume": args.resume,
        "arms": list(V5_ARMS),
        "training_end": TRAIN_END,
        "outputs": outputs,
        "input_gate": input_gate,
        "graph_source_lock": actual_graph,
        "resume_parity_gate": {"status": "READY_POSTRUN", "note": "audit_v5_resume_parity requires an interrupted/resumed pair and is fail-closed when invoked"},
        "isolation_gate": {"status": "READY_POSTRUN", "note": "audit_v5_isolation is required after each arm"},
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
