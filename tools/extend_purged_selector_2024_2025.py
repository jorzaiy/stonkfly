"""Continue the purged WC006 selector in a new, isolated 2024–2025 mirror."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from experiments.a_share.run_isolated_selector_slice import (
    TRADING_ROOT, _atomic_json_write, _validate_checkpoint,
    contiguous_hold_schedule_dates, run_slice, sha256,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/wc006_v3_2021_inputs/20260925T-purged-full-2021-2023-stateful"
SOURCE_SHA256 = "e9188ffd44ea1296c2759de6d56752965ad0cfa97c13b9bd10c684dfb41d1641"
TAG = "5y_2021-01-01_2026-04-03"
NEW_START, NEW_END = "2024-01-02", "2025-12-29"
DEFAULT_PYTHON = TRADING_ROOT / "project/venv_exp_clean/bin/python"


def scheduled_dates() -> tuple[list[str], list[str]]:
    calendar_path = TRADING_ROOT / "project/data/ops/trading_calendar.json"
    calendar = json.loads(calendar_path.read_text(encoding="utf-8"))["dates"]
    all_dates = contiguous_hold_schedule_dates(calendar)
    old = [date for date in all_dates if "2021-07-01" <= date <= "2023-12-29"]
    new = [date for date in all_dates if NEW_START <= date <= NEW_END]
    if len(old) != 46 or len(new) != 36 or new[0] != NEW_START or new[-1] != NEW_END:
        raise RuntimeError("selector continuation schedule changed")
    return old, new


def extend(output: Path, *, python: Path, resume: bool = False) -> Path:
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT / "runs/wc006_v3_2024_2025_inputs").resolve()):
        raise RuntimeError("selector extension output must use the new research namespace")
    old, new = scheduled_dates()
    dates = old + new
    if not resume:
        if output.exists():
            raise RuntimeError("selector extension output must be new")
        if sha256(SOURCE / "checkpoint.json") != SOURCE_SHA256:
            raise RuntimeError("source purged selector checkpoint hash mismatch")
        source_checkpoint, _, _, overlay = _validate_checkpoint(
            SOURCE, old, TAG, allow_original_source_drift=True)
        if source_checkpoint["completed_dates"] != old or "purge" not in overlay:
            raise RuntimeError("source purged selector is incomplete or unpurged")
        if source_checkpoint.get("pending_date") is not None:
            raise RuntimeError("source selector has pending work")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(SOURCE, output)
        clone_checkpoint, _, _, _ = _validate_checkpoint(
            output, old, TAG, allow_original_source_drift=True)
        if clone_checkpoint.get("artifact_hashes") != source_checkpoint.get("artifact_hashes"):
            raise RuntimeError("copied prior watch artifacts differ")
        clone_checkpoint["dates"] = dates
        clone_checkpoint["extension_source_checkpoint_sha256"] = SOURCE_SHA256
        clone_checkpoint["extension_window"] = [NEW_START, NEW_END]
        _atomic_json_write(output / "checkpoint.json", clone_checkpoint)
        _atomic_json_write(output / "extension_start.json", {
            "schema": "wc006-purged-selector-extension-start/v1",
            "source_checkpoint_sha256": SOURCE_SHA256,
            "old_dates": old,
            "new_dates": new,
            "prefer_tag": TAG,
        })
    else:
        checkpoint, _, _, overlay = _validate_checkpoint(
            output, dates, TAG, allow_original_source_drift=True)
        marker = json.loads((output / "extension_start.json").read_text(encoding="utf-8"))
        if (marker.get("source_checkpoint_sha256") != SOURCE_SHA256
                or marker.get("old_dates") != old
                or marker.get("new_dates") != new
                or marker.get("prefer_tag") != TAG
                or "purge" not in overlay):
            raise RuntimeError("selector extension resume provenance mismatch")
    # Keep the virtualenv executable path: resolve() follows its python
    # symlink to /usr/bin and loses the virtualenv site-packages.
    run_slice(output, dates, python=Path(python).absolute(), prefer_tag=TAG,
              cache_dir=TRADING_ROOT / "data/cache_canonical_202603",
              resume=True, purge_train_labels=True,
              allow_original_source_drift=True)
    final, _, _, _ = _validate_checkpoint(
        output, dates, TAG, allow_original_source_drift=True)
    if final["completed_dates"] != dates:
        raise RuntimeError("selector continuation did not complete all dates")
    interpreter = Path(python).absolute()
    versions = subprocess.check_output(
        [str(interpreter), "-c", "import sys,lightgbm,numpy,pandas,sklearn,json; print(json.dumps({'python':sys.version.split()[0],'lightgbm':lightgbm.__version__,'numpy':numpy.__version__,'pandas':pandas.__version__,'sklearn':sklearn.__version__}))"],
        text=True,
    )
    _atomic_json_write(output / "extension_provenance.json", {
        "schema": "wc006-purged-selector-extension/v1",
        "source_checkpoint_sha256": SOURCE_SHA256,
        "new_window": [NEW_START, NEW_END],
        "new_dates": new,
        "interpreter": str(interpreter),
        "versions": json.loads(versions),
        "completed_checkpoint_sha256": sha256(output / "checkpoint.json"),
        "mirror_provenance_sha256": sha256(output / "mirror_provenance.json"),
    })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(extend(args.out, python=args.python, resume=args.resume))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
