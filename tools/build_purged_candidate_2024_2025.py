"""Build a hash-pinned 2024–2025 candidate stream from the completed mirror."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.a_share.build_wc006_candidate_stream import build_candidate_stream, sha256
from tools.extend_purged_selector_2024_2025 import NEW_END, NEW_START, ROOT, TAG, scheduled_dates
from experiments.a_share.run_isolated_selector_slice import TRADING_ROOT, _validate_checkpoint


def build(mirror_run: Path, output: Path) -> Path:
    mirror_run, output = Path(mirror_run).resolve(), Path(output).resolve()
    if not mirror_run.is_relative_to((ROOT / "runs/wc006_v3_2024_2025_inputs").resolve()):
        raise RuntimeError("selector mirror must use the new research namespace")
    if not output.is_relative_to((ROOT / "runs/wc006_v3_2024_2025_inputs").resolve()):
        raise RuntimeError("candidate output must use the new research namespace")
    old, new = scheduled_dates()
    checkpoint, mirror, _, overlays = _validate_checkpoint(
        mirror_run, old + new, TAG, allow_original_source_drift=True)
    if checkpoint["completed_dates"] != old + new or checkpoint.get("pending_date") is not None:
        raise RuntimeError("selector continuation is incomplete")
    if "purge" not in overlays:
        raise RuntimeError("selector continuation lacks purge overlay")
    extension = json.loads((mirror_run / "extension_provenance.json").read_text(encoding="utf-8"))
    if extension.get("new_window") != [NEW_START, NEW_END] or extension.get("new_dates") != new:
        raise RuntimeError("selector extension provenance dates mismatch")
    if extension.get("completed_checkpoint_sha256") != sha256(mirror_run / "checkpoint.json"):
        raise RuntimeError("selector completed checkpoint hash mismatch")
    watch = mirror / "data/watch_stateful"
    artifacts = [watch / f"rebalance_target_{date.replace('-', '')}.json" for date in new]
    if any(not path.is_file() for path in artifacts):
        raise RuntimeError("selector applied artifact is missing")
    return build_candidate_stream(
        artifact_paths=artifacts,
        active_config=mirror / "rdlab/configs/wf_ok1867_candidate_return_leading_hold20_contiguous_continuous_wc006.json",
        universe=mirror / "universe_5y_days161.top2000tag.ok1867.unique.txt",
        calendar=mirror / "data/ops/trading_calendar.json",
        cache_dir=TRADING_ROOT / "data/cache_canonical_202603",
        output_dir=output,
        trading_root=TRADING_ROOT,
        selector_checkpoint=mirror_run / "checkpoint.json",
        selector_mirror_provenance=mirror_run / "mirror_provenance.json",
        selector_extension_provenance=mirror_run / "extension_provenance.json",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.mirror_run, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
