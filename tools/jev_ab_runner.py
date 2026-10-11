#!/usr/bin/env python3
"""Plan/validate/compare Jev A/B arms; performance execution is opt-in and gated."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from stonkfly.jev_ab import ABConfig, compare_arms, make_plan, performance_gate, regression_compare


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "dry-run", "regression", "performance", "compare"))
    parser.add_argument("--out", default="runs/jev-ab")
    parser.add_argument("--baseline-audit")
    parser.add_argument("--candidate-stream")
    parser.add_argument("--market-data")
    parser.add_argument("--feature-dataset")
    parser.add_argument("--feature-names", required=False, help="comma-separated fixed-order names (8-32)")
    parser.add_argument("--mapping-hash", default="unconfigured")
    parser.add_argument("--normalization-hash", default="unconfigured")
    parser.add_argument("--brain-checkpoint")
    parser.add_argument("--direct-quant-baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--a", help="A events.jsonl")
    parser.add_argument("--b", help="B events.jsonl")
    parser.add_argument("--c", help="C events.jsonl")
    parser.add_argument("--d", help="D events.jsonl")
    parser.add_argument("--original", help="original V5.1 events.jsonl for regression")
    parser.add_argument("--null", dest="null_events", help="NullSensoryAdapter events.jsonl for regression")
    parser.add_argument("--diff-report", help="write cross-arm-key/date-aligned A/B diff JSON (default: --out/diff-report.json)")
    args = parser.parse_args(argv)
    if args.mode == "compare":
        paths = {k: v for k, v in zip(("A", "B", "C", "D"), (args.a, args.b, args.c, args.d)) if v}
        if len(paths) < 2:
            parser.error("compare requires at least two of --a/--b/--c/--d")
        report = compare_arms(paths)
        diff_path = Path(args.diff_report) if args.diff_report else Path(args.out) / "diff-report.json"
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        diff_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
        print(json.dumps({"status": "COMPARED", "diff_report": str(diff_path), "report": report}, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.mode == "regression":
        if not args.original or not args.null_events:
            parser.error("regression requires --original and --null")
        original = [json.loads(x) for x in Path(args.original).read_text(encoding="utf-8").splitlines() if x.strip()]
        null = [json.loads(x) for x in Path(args.null_events).read_text(encoding="utf-8").splitlines() if x.strip()]
        result = regression_compare(original, null)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2
    names = tuple(x.strip() for x in (args.feature_names or "f0,f1,f2,f3,f4,f5,f6,f7").split(","))
    plan = make_plan(ABConfig(output_root=args.out, baseline_audit=args.baseline_audit, candidate_stream=args.candidate_stream, market_data=args.market_data, feature_dataset=args.feature_dataset, feature_names=names, mapping_hash=args.mapping_hash, normalization_hash=args.normalization_hash, brain_checkpoint=args.brain_checkpoint, seed=args.seed, direct_quant_baseline=args.direct_quant_baseline))
    if args.mode == "performance":
        try:
            performance_gate(plan)
        except RuntimeError as exc:
            print(json.dumps({"status": "BLOCKED", "reason": str(exc), "performance_started": False}, ensure_ascii=False))
            return 2
        print(json.dumps({"status": "READY_ONLY", "message": "preflight passed; no performance A/B was started", "performance_started": False, "plan": str(Path(args.out) / "plan.json")}, ensure_ascii=False))
        return 0
    print(json.dumps({"status": "PLANNED", "plan": str(Path(args.out) / "plan.json"), "performance_started": False}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())
