#!/usr/bin/env python3
"""Grid search for MBON decoder plasticity parameters."""

import json
import subprocess
import sys
from pathlib import Path
from itertools import product

ROOT = Path(__file__).resolve().parent.parent.parent

# 参数网格
GRID = {
    "minimum_fraction": [0.3, 0.5, 0.7],
    "memory_decay_seconds": [600, 900, 1200],
}

RULE_PY = ROOT / "stonkfly" / "neural" / "rule.py"
CANDIDATE = "runs/wc006_v3_2021_inputs/20260925T-purged-candidate-stream-2021-2023-v2/manifest.json"
MARKET = "runs/wc006_purged_xysz_market/20260925T-131codes-v6/manifest.json"


def update_rule_py(min_f: float, decay: float):
    """Update rule.py PARAMETERS."""
    content = RULE_PY.read_text()
    
    # 替换 minimum_fraction
    import re
    content = re.sub(
        r'"minimum_fraction": [\d.]+',
        f'"minimum_fraction": {min_f}',
        content
    )
    content = re.sub(
        r'"memory_decay_seconds": [\d.]+',
        f'"memory_decay_seconds": {decay}',
        content
    )
    
    RULE_PY.write_text(content)
    print(f"  Updated rule.py: min_f={min_f}, decay={decay}")


def run_training(output_name: str):
    """Run training with current rule.py parameters."""
    output_dir = ROOT / "runs" / "wc006_purged_fly_training" / output_name
    
    if output_dir.exists() and (output_dir / "events.jsonl").exists():
        print(f"  Skipping (exists): {output_name}")
        return True
    
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "a_share" / "run_purged_fly_training.py"),
        "--candidate-manifest", str(ROOT / CANDIDATE),
        "--market-manifest", str(ROOT / MARKET),
        "--out", str(output_dir),
        "--decoder-type", "mbon",
        "--max-days", "609",
    ]
    
    print(f"  Running: {output_name}")
    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env={"PYTHONPATH": str(ROOT), **__import__("os").environ},
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}")
        return False
    
    return True


def main():
    combinations = list(product(
        GRID["minimum_fraction"],
        GRID["memory_decay_seconds"],
    ))
    
    print(f"Grid search: {len(combinations)} combinations")
    print("=" * 70)
    
    results = []
    for i, (min_f, decay) in enumerate(combinations, 1):
        run_name = f"20260928T-mbon-minf{int(min_f*10):02d}-decay{int(decay)}-seed17"
        print(f"\n[{i}/{len(combinations)}] min_f={min_f}, decay={decay}")
        
        update_rule_py(min_f, decay)
        success = run_training(run_name)
        results.append({
            "min_f": min_f,
            "decay": decay,
            "run_name": run_name,
            "success": success,
        })
    
    print("\n" + "=" * 70)
    print("Grid search complete!")
    print(f"Success: {sum(r['success'] for r in results)}/{len(results)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
