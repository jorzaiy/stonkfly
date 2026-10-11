#!/usr/bin/env python3
"""Bind frozen candidate and passing raw-market QA before reading replay returns."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.collect_purged_xysz_2024_2025 import sha, write_json
from experiments.a_share.run_purged_fly_base_2024_2025 import validate_market_policy

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/experiments/fly_2024_2025_heldout_protocol.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--market", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    candidate, market, out = args.candidate.resolve(), args.market.resolve(), args.out.resolve()
    qa = market.parent / "quality_report.json"
    for p in (candidate, market, qa):
        if not p.is_file():
            raise ValueError(f"input_missing:{p}")
    cm, mm, qm = (json.loads(p.read_text()) for p in (candidate, market, qa))
    validate_market_policy(mm, qm)
    if (len(cm.get("records", [])) != 36 or cm.get("paper_only") is not True or
            mm.get("candidate_manifest_sha256") != sha(candidate) or
            mm.get("status") != "PASS_RESEARCH_ONLY" or
            qm.get("manifest_sha256") != sha(market) or
            qm.get("status") != "PASS_RESEARCH_ONLY" or qm.get("issues") or
            any(v is not True for v in qm.get("checks", {}).values())):
        raise ValueError("candidate_or_market_gate_failed")
    if out.exists():
        raise ValueError("input_lock_already_exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = {"schema": "fly-purged-heldout-2024-2025/v1/input-lock", "paper_only": True,
            "candidate_manifest": str(candidate), "candidate_sha256": sha(candidate),
            "market_manifest": str(market), "market_sha256": sha(market),
            "qa_sha256": sha(qa), "protocol": "docs/experiments/fly_2024_2025_heldout_protocol.md",
            "protocol_sha256": sha(PROTOCOL)}
    write_json(out, lock)
    print(json.dumps({"input_lock": str(out), "sha256": sha(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
