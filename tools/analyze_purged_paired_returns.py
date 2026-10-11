"""Exploratory paired-return uncertainty audit for the frozen paper replay."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from tools.summarize_purged_fly_base_performance import RUN_DIR, dec, read_events, sha256

SEED = 20260927
BLOCK = 20
REPLICATES = 10000


def analyze(run_dir: Path) -> dict:
    events_path = Path(run_dir).resolve() / "events.jsonl"
    events = read_events(events_path)
    differences = []
    monthly: dict[str, list[float]] = defaultdict(list)
    for row in events:
        reward = row["reward"]
        fly = dec(reward["portfolio_return"], "Fly daily return")
        base = dec(reward["base_return"], "Base daily return")
        if fly <= Decimal("-1") or base <= Decimal("-1"):
            raise ValueError("daily return cannot be compounded")
        value = math.log1p(float(fly)) - math.log1p(float(base))
        differences.append(value)
        monthly[row["date"][:7]].append(value)
    if len(differences) != 609:
        raise ValueError("frozen paired event count mismatch")
    blocks = [differences[index:index + BLOCK] for index in range(0, len(differences), BLOCK)]
    rng = random.Random(SEED)
    sampled = []
    for _ in range(REPLICATES):
        sequence = []
        while len(sequence) < len(differences):
            sequence.extend(blocks[rng.randrange(len(blocks))])
        sampled.append(sum(sequence[:len(differences)]))
    sampled.sort()
    observed = sum(differences)
    positive_months = sum(sum(values) > 0 for values in monthly.values())
    return {
        "schema": "stonkfly-purged-paired-return-uncertainty/v1",
        "status": "EXPLORATORY_POSTHOC_RESEARCH_ONLY",
        "source_events_sha256": sha256(events_path),
        "method": {
            "statistic": "sum of paired daily log(1+Fly return)-log(1+Base return)",
            "block_sessions": BLOCK, "resampling": "non-overlapping paired blocks with replacement, truncated to 609 sessions",
            "replicates": REPLICATES, "seed": SEED,
        },
        "observed_log_relative_wealth": observed,
        "observed_fly_to_base_wealth_ratio": math.exp(observed),
        "resample_interval_log_relative_wealth_95pct": [sampled[249], sampled[9749]],
        "resample_interval_wealth_ratio_95pct": [math.exp(sampled[249]), math.exp(sampled[9749])],
        "positive_paired_months": positive_months,
        "paired_months": len(monthly),
        "interpretation": "Post hoc descriptive uncertainty under a 20-session block resampling assumption; not a prospective p-value, neural attribution, or proof of profitability.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(args.run_dir.resolve()):
        raise ValueError("output must not alter the frozen run")
    result = analyze(args.run_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
