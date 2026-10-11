#!/usr/bin/env python3
"""Run a tiny selector slice inside a hash-pinned Stonkfly mirror."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


TRADING_ROOT = Path("/home/ubuntu/trading_agent_20260408")
CONFIG_REL = Path("project/rdlab/configs/wf_ok1867_candidate_return_leading_hold20_contiguous_continuous_wc006.json")
UNIVERSE_REL = Path("project/universe_5y_days161.top2000tag.ok1867.unique.txt")
CALENDAR_REL = Path("project/data/ops/trading_calendar.json")
CACHE_REL = Path("data/cache_canonical_202603")
EXPORTER_REL = Path("project/rdlab/scripts/export_leading_daily_signal.py")
WALKFORWARD_REL = Path("project/rdlab/src/walkforward.py")
SPEC_REL = Path("docs/experiments/wc006_v3_2021_preregister.v3-turnover-overlay.md")
OVERLAY_PATCH_ID = "none-safe-target-topk-render-v1"
OLD_RENDER_BLOCK = """    for row in target_topk:
        md_lines.append(
            f"{row['rank']:>2}. {row['code']} score={row['score']:.6f} target_weight={row['target_weight']:.6f}"
        )
"""
NEW_RENDER_BLOCK = """    for row in target_topk:
        score = row["score"]
        weight = row["target_weight"]
        score_text = "N/A" if score is None else f"{float(score):.6f}"
        weight_text = "N/A" if weight is None else f"{float(weight):.6f}"
        md_lines.append(
            f"{row['rank']:>2}. {row['code']} score={score_text} target_weight={weight_text}"
        )
"""
TURNOVER_OVERLAY_PATCH_ID = "eligible-prev-turnover-cap-v1"
TURNOVER_OLD_BLOCK = """def apply_turnover_cap(desired: list[str], prev: set[str] | None, topk: int, cap: float) -> list[str]:
    if prev is None:
        return desired[:topk]
    keep_n = int(round(topk * (1.0 - cap)))
    keep = [x for x in desired if x in prev][:keep_n]
    if len(keep) < keep_n:
        for x in sorted(prev):
            if x not in keep:
                keep.append(x)
                if len(keep) >= keep_n:
                    break
    picks = list(keep)
    for x in desired:
        if x not in picks:
            picks.append(x)
            if len(picks) >= topk:
                break
    return picks[:topk]
"""
TURNOVER_NEW_BLOCK = """def apply_turnover_cap(desired: list[str], prev: set[str] | None, topk: int, cap: float) -> list[str]:
    if prev is None:
        return desired[:topk]
    keep_n = int(round(topk * (1.0 - cap)))
    eligible_prev = [x for x in desired if x in prev]
    keep = eligible_prev[:keep_n]
    picks = list(keep)
    for x in desired:
        if x not in picks:
            picks.append(x)
            if len(picks) >= topk:
                break
    return picks[:topk]
"""
STATE_INIT_PATCH_ID = "shared-watch-prior-state-v1"
STATE_INIT_OLD_BLOCK = """    rebalance_exports = _export_rebalance_targets(
"""
STATE_INIT_NEW_BLOCK = """    # Experiment-only shared-watch carry: load the latest completed applied
    # artifact before exporting this date. The runner invokes dates strictly in
    # order and separately verifies that prior applied artifacts are unchanged.
    prior_paths = sorted(watch_dir.glob("rebalance_target_*.json"))
    for prior_path in reversed(prior_paths):
        prior_payload = json.loads(prior_path.read_text(encoding="utf-8"))
        if prior_payload.get("rebalance_applied") and prior_payload.get("effective_target_topk"):
            init_prev_target_codes = [
                str(row["code"]).zfill(6)
                for row in prior_payload["effective_target_topk"]
            ]
            init_prev_effective_target_topk = [
                dict(row) for row in prior_payload["effective_target_topk"]
            ]
            break

    rebalance_exports = _export_rebalance_targets(
"""
PURGE_OVERLAY_PATCH_ID = "purged-forward-labels-v1"
PURGE_OLD_BLOCK = """    is_train = (
        (feat.index.get_level_values(0) >= train_start)
        & (feat.index.get_level_values(0) < train_end)
        & y_label.notna()
    )
"""
PURGE_NEW_BLOCK = """    is_train = (
        (feat.index.get_level_values(0) >= train_start)
        & (feat.index.get_level_values(0) < train_end)
        & y_label.notna()
    )
    # Experiment-only purge: drop the last `hold` sessions before train_end so
    # no training label reaches past the signal month start. Mirrors
    # walkforward.py purge_train_labels (audit/p0-2021-selector-leakage).
    pre_signal_dates = all_dates[all_dates < train_end]
    purge_hold = int(cfg.strategy.hold)
    if len(pre_signal_dates) <= purge_hold:
        raise ValueError("not enough dates to purge forward labels")
    purge_cutoff = pd.Timestamp(pre_signal_dates[-purge_hold])
    is_train = is_train & (feat.index.get_level_values(0) < purge_cutoff)
"""


def presentation_metric(value: object) -> str:
    """Render report-only numeric fields without changing selection semantics."""
    return "N/A" if value is None else f"{float(value):.6f}"""


def contiguous_hold_schedule_dates(all_dates: list[str], *, train_months: int = 3, hold: int = 20, warmup_sessions: int = 60) -> list[str]:
    """Mirror walkforward's month/contiguous-hold schedule, without models."""
    if train_months < 0 or hold <= 0 or warmup_sessions < 0:
        raise ValueError("invalid schedule parameters")
    dates = sorted(set(str(item)[:10] for item in all_dates))[warmup_sessions:]
    months = sorted(set(item[:7] for item in dates))
    if len(months) <= train_months:
        return []
    result: list[str] = []
    for month in months[train_months:]:
        month_dates = [item for item in dates if item[:7] == month]
        result.extend(month_dates[::hold])
    return result


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def snapshot(root: Path) -> dict[str, tuple[int, int, str]]:
    rels = [Path("project/data/watch"), Path("project/data/shadow"), Path("project/rdlab/results"), Path("project/rdlab/reports"), CONFIG_REL]
    out: dict[str, tuple[int, int, str]] = {}
    for rel in rels:
        p = root / rel
        files = [p] if p.is_file() else sorted(p.rglob("*")) if p.exists() else []
        for item in files:
            if item.is_file():
                st = item.stat()
                out[str(item.relative_to(root))] = (st.st_size, st.st_mtime_ns, sha256(item))
    return out


def external_processes(mirror: Path) -> list[str]:
    raw = subprocess.check_output(["ps", "-eo", "args="], text=True)
    needles = ("daily_ops.py", "run_live_signal.py", "replay_wc006.py", "replay_510300_v3.py", "export_leading_daily_signal.py")
    return [
        line.strip()
        for line in raw.splitlines()
        if any(needle in line for needle in needles)
        and str(mirror) not in line
        and "ps -eo" not in line
        and "rg 'daily_ops.py" not in line
    ]


def guard(before: dict[str, tuple[int, int, str]], mirror: Path) -> None:
    if external_processes(mirror):
        raise RuntimeError("new external live/export/replay process detected")
    if snapshot(TRADING_ROOT) != before:
        raise RuntimeError("trading repository changed during isolated selector run")


def copy_checked(source: Path, destination: Path, copied: dict[str, str]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    source_hash = sha256(source)
    if sha256(destination) != source_hash:
        raise RuntimeError(f"mirror hash mismatch: {source}")
    copied[str(destination)] = source_hash


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def apply_presentation_overlay(exporter: Path) -> dict[str, object]:
    original = exporter.read_text(encoding="utf-8")
    if original.count(OLD_RENDER_BLOCK) != 1:
        raise RuntimeError("presentation overlay expected exactly one render block")
    original_sha = sha256(exporter)
    patched = original.replace(OLD_RENDER_BLOCK, NEW_RENDER_BLOCK)
    exporter.write_text(patched, encoding="utf-8")
    return {
        "patch_id": OVERLAY_PATCH_ID,
        "patch_sha256": text_sha256(NEW_RENDER_BLOCK),
        "matched_blocks": 1,
        "original_sha256": original_sha,
        "patched_sha256": sha256(exporter),
    }


PRESERVE_OLD_BLOCK = """        out_path = watch_dir / f"rebalance_target_{payload['date']}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
"""
PRESERVE_NEW_BLOCK = """        out_path = watch_dir / f"rebalance_target_{payload['date']}.json"
        # Never rewrite an existing history artifact during a later as-of
        # export. This preserves both completed applied state and placeholders.
        if out_path.exists():
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            target_topk = [dict(row) for row in payload.get("target_topk", [])]
            effective_target_topk = [dict(row) for row in payload.get("effective_target_topk", [])]
            rebalance_applied = bool(payload.get("rebalance_applied"))
        else:
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
"""


def apply_stateful_export_overlay(exporter: Path) -> dict[str, object]:
    original = exporter.read_text(encoding="utf-8")
    if original.count(STATE_INIT_OLD_BLOCK) != 1:
        raise RuntimeError("state overlay expected exactly one initialization block")
    if original.count(PRESERVE_OLD_BLOCK) != 1:
        raise RuntimeError("state overlay expected exactly one artifact write block")
    original_sha = sha256(exporter)
    patched = original.replace(STATE_INIT_OLD_BLOCK, STATE_INIT_NEW_BLOCK, 1)
    patched = patched.replace(PRESERVE_OLD_BLOCK, PRESERVE_NEW_BLOCK, 1)
    exporter.write_text(patched, encoding="utf-8")
    return {
        "patch_id": STATE_INIT_PATCH_ID,
        "patch_sha256": text_sha256(STATE_INIT_NEW_BLOCK + PRESERVE_NEW_BLOCK),
        "matched_init_blocks": 1,
        "matched_preserve_blocks": 1,
        "original_sha256": original_sha,
        "patched_sha256": sha256(exporter),
    }


def apply_turnover_overlay(walkforward: Path) -> dict[str, object]:
    original = walkforward.read_text(encoding="utf-8")
    if original.count(TURNOVER_OLD_BLOCK) != 1:
        raise RuntimeError("turnover overlay expected exactly one apply_turnover_cap block")
    original_sha = sha256(walkforward)
    patched = original.replace(TURNOVER_OLD_BLOCK, TURNOVER_NEW_BLOCK)
    walkforward.write_text(patched, encoding="utf-8")
    return {
        "patch_id": TURNOVER_OVERLAY_PATCH_ID,
        "patch_sha256": text_sha256(TURNOVER_NEW_BLOCK),
        "matched_blocks": 1,
        "original_sha256": original_sha,
        "patched_sha256": sha256(walkforward),
    }


def apply_purge_overlay(exporter: Path) -> dict[str, object]:
    original = exporter.read_text(encoding="utf-8")
    if original.count(PURGE_OLD_BLOCK) != 1:
        raise RuntimeError("purge overlay expected exactly one is_train block")
    original_sha = sha256(exporter)
    exporter.write_text(original.replace(PURGE_OLD_BLOCK, PURGE_NEW_BLOCK), encoding="utf-8")
    return {
        "patch_id": PURGE_OVERLAY_PATCH_ID,
        "patch_sha256": text_sha256(PURGE_NEW_BLOCK),
        "matched_blocks": 1,
        "original_sha256": original_sha,
        "patched_sha256": sha256(exporter),
    }


def audit_stateful_artifacts(watch_dir: Path, dates: list[str]) -> list[dict[str, object]]:
    prior_codes: list[str] = []
    audit: list[dict[str, object]] = []
    for date in dates:
        path = watch_dir / f"rebalance_target_{date.replace('-', '')}.json"
        if not path.is_file():
            raise RuntimeError(f"missing stateful rebalance artifact: {date}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("rebalance_applied") or not payload.get("effective_target_topk"):
            raise RuntimeError(f"stateful artifact is not applied/nonempty: {date}")
        actual_prev = [str(code).zfill(6) for code in payload.get("previous_target_codes", [])]
        if actual_prev != prior_codes:
            raise RuntimeError(f"state carry mismatch at {date}: expected {prior_codes}, got {actual_prev}")
        rows = [dict(row) for row in payload["effective_target_topk"]]
        codes = [str(row["code"]).zfill(6) for row in rows]
        audit.append({
            "signal_date": date,
            "artifact": path.name,
            "previous_target_codes": actual_prev,
            "target_codes": codes,
            "target_weights": [float(row["target_weight"]) for row in rows],
        })
        prior_codes = codes
    return audit


def make_mirror(staging: Path, *, purge_train_labels: bool = False) -> tuple[Path, dict[str, str], dict[str, object]]:
    mirror = staging / "mirror"
    copied: dict[str, str] = {}
    copy_checked(TRADING_ROOT / EXPORTER_REL, mirror / "rdlab/scripts/export_leading_daily_signal.py", copied)
    copy_checked(TRADING_ROOT / WALKFORWARD_REL, mirror / "rdlab/src/walkforward.py", copied)
    for rel in (Path("project/rdlab/__init__.py"), Path("project/rdlab/src/__init__.py")):
        copy_checked(TRADING_ROOT / rel, mirror / rel.relative_to("project"), copied)
    for rel in (CONFIG_REL, UNIVERSE_REL, CALENDAR_REL):
        copy_checked(TRADING_ROOT / rel, mirror / rel.relative_to("project"), copied)
    (mirror / "data/watch_stateful").mkdir(parents=True)
    (mirror / "rdlab/results").mkdir(parents=True)
    (mirror / "rdlab/reports").mkdir(parents=True)
    exporter = mirror / "rdlab/scripts/export_leading_daily_signal.py"
    walkforward = mirror / "rdlab/src/walkforward.py"
    if sha256(exporter) != copied[str(exporter)]:
        raise RuntimeError("original exporter hash check failed before overlay")
    if sha256(walkforward) != copied[str(walkforward)]:
        raise RuntimeError("original walkforward hash check failed before overlay")
    source_exporter_sha = copied[str(exporter)]
    state_overlay = apply_stateful_export_overlay(exporter)
    exporter_overlay = apply_presentation_overlay(exporter)
    state_overlay["source_sha256"] = source_exporter_sha
    exporter_overlay["source_sha256"] = source_exporter_sha
    exporter_overlay["input_sha256"] = state_overlay["patched_sha256"]
    turnover_overlay = apply_turnover_overlay(walkforward)
    overlays = {"state": state_overlay, "exporter": exporter_overlay, "walkforward": turnover_overlay}
    if purge_train_labels:
        purge_overlay = apply_purge_overlay(exporter)
        purge_overlay["input_sha256"] = exporter_overlay["patched_sha256"]
        overlays["purge"] = purge_overlay
    return mirror, copied, overlays


def _relative_hashes(root: Path, paths: list[Path]) -> dict[str, str]:
    return {str(path.relative_to(root)): sha256(path) for path in paths}


def _atomic_json_write(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _source_hashes_for_checkpoint(output: Path, copied: dict[str, str]) -> dict[str, str]:
    return {str(Path(key).relative_to(output)): value for key, value in copied.items()}


def _checkpoint_payload(
    output: Path,
    dates: list[str],
    prefer_tag: str,
    copied: dict[str, str],
    overlay: dict[str, object],
    completed_dates: list[str],
    state_audit: list[dict[str, object]],
    watch_dir: Path,
    *,
    pending_date: str | None,
    pending_prior_hashes: dict[str, str],
) -> dict[str, object]:
    artifacts = sorted(watch_dir.glob("rebalance_target_*.json"))
    return {
        "schema": "wc006-isolated-selector-checkpoint/v1",
        "dates": dates,
        "prefer_tag": prefer_tag,
        "completed_dates": completed_dates,
        "last_completed_date": completed_dates[-1] if completed_dates else None,
        "pending_date": pending_date,
        "pending_prior_hashes": pending_prior_hashes,
        "artifact_hashes": _relative_hashes(output, artifacts),
        "prior_state": state_audit[-1] if state_audit else None,
        "state_audit": state_audit,
        "source_hashes_original": _source_hashes_for_checkpoint(output, copied),
        "overlays": overlay,
        "spec_hashes": {str(SPEC_REL): sha256(Path(SPEC_REL))},
    }


def _validate_checkpoint(
    output: Path,
    dates: list[str],
    prefer_tag: str,
    *,
    allow_original_source_drift: bool = False,
) -> tuple[dict[str, object], Path, dict[str, str], dict[str, object]]:
    checkpoint_path = output / "checkpoint.json"
    if not checkpoint_path.is_file():
        raise RuntimeError("resume requires checkpoint.json")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint.get("schema") != "wc006-isolated-selector-checkpoint/v1":
        raise RuntimeError("unsupported checkpoint schema")
    if checkpoint.get("dates") != dates or checkpoint.get("prefer_tag") != prefer_tag:
        raise RuntimeError("resume date list or exact tag differs from checkpoint")
    completed = checkpoint.get("completed_dates") or []
    if completed != dates[: len(completed)] or len(set(completed)) != len(completed):
        raise RuntimeError("checkpoint completed dates are not a strict prefix")
    if checkpoint.get("pending_date") is not None and checkpoint["pending_date"] != dates[len(completed)]:
        raise RuntimeError("checkpoint pending date is not the next strict date")
    mirror = output / "mirror"
    if not mirror.is_dir():
        raise RuntimeError("resume mirror is missing")
    copied = {str(output / key): value for key, value in (checkpoint.get("source_hashes_original") or {}).items()}
    overlay = checkpoint.get("overlays") or {}
    patched_expected = {
        output / "mirror/rdlab/scripts/export_leading_daily_signal.py": (overlay.get("purge") or overlay.get("exporter") or {}).get("patched_sha256"),
        output / "mirror/rdlab/src/walkforward.py": (overlay.get("walkforward") or {}).get("patched_sha256"),
    }
    source_pairs = {
        output / "mirror/rdlab/scripts/export_leading_daily_signal.py": TRADING_ROOT / EXPORTER_REL,
        output / "mirror/rdlab/src/walkforward.py": TRADING_ROOT / WALKFORWARD_REL,
    }
    for raw_path, expected in copied.items():
        path = Path(raw_path)
        mirror_expected = patched_expected.get(path) or expected
        if not path.is_file() or sha256(path) != mirror_expected:
            raise RuntimeError(f"resume source/mirror hash changed: {path}")
        if path in source_pairs and not allow_original_source_drift and sha256(source_pairs[path]) != expected:
            raise RuntimeError(f"resume trading source hash changed: {source_pairs[path]}")
    for rel, expected in (checkpoint.get("artifact_hashes") or {}).items():
        path = output / rel
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"resume artifact hash changed: {path}")
    for rel, expected in (checkpoint.get("spec_hashes") or {}).items():
        path = Path(rel)
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"resume prereg hash changed: {path}")
    return checkpoint, mirror, copied, overlay


def run_slice(output: Path, dates: list[str], *, python: Path, prefer_tag: str, cache_dir: Path | None = None, resume: bool = False, purge_train_labels: bool = False, allow_original_source_drift: bool = False) -> Path:
    before = snapshot(TRADING_ROOT)
    output = output.resolve()
    if cache_dir is None:
        cache_dir = TRADING_ROOT / CACHE_REL
    else:
        cache_dir = Path(cache_dir).resolve()
    if resume:
        if not output.is_dir():
            raise RuntimeError("resume output directory does not exist")
        checkpoint, mirror, copied_paths, overlay = _validate_checkpoint(
            output, dates, prefer_tag,
            allow_original_source_drift=allow_original_source_drift,
        )
        if ("purge" in overlay) != purge_train_labels:
            raise RuntimeError("resume purge_train_labels differs from checkpoint overlays")
        copied = {str(Path(key)): value for key, value in copied_paths.items()}
        watch_dir = mirror / "data/watch_stateful"
        completed_dates = list(checkpoint.get("completed_dates") or [])
        state_audit = list(checkpoint.get("state_audit") or [])
        start_index = len(completed_dates)
        pending = checkpoint.get("pending_date")
        if pending is not None:
            pending_prior = checkpoint.get("pending_prior_hashes") or {}
            for raw_path, expected in pending_prior.items():
                path = output / raw_path if not str(raw_path).startswith("/") else Path(raw_path)
                if not path.is_file() or sha256(path) != expected:
                    raise RuntimeError(f"resume prior artifact changed during pending date: {path}")
            try:
                recovered_audit = audit_stateful_artifacts(watch_dir, dates[: start_index + 1])
            except Exception as exc:
                raise RuntimeError("pending date has incomplete artifacts; refusing duplicate resume") from exc
            state_audit = recovered_audit
            completed_dates.append(pending)
            start_index += 1
            _atomic_json_write(output / "checkpoint.json", _checkpoint_payload(
                output, dates, prefer_tag, copied, overlay, completed_dates, state_audit, watch_dir,
                pending_date=None, pending_prior_hashes={},
            ))
    else:
        if output.exists():
            raise RuntimeError("output must be unique")
        output.mkdir(parents=True)
        mirror, copied, overlay = make_mirror(output, purge_train_labels=purge_train_labels)
        watch_dir = mirror / "data/watch_stateful"
        completed_dates = []
        state_audit = []
        _atomic_json_write(output / "checkpoint.json", _checkpoint_payload(
            output, dates, prefer_tag, copied, overlay, completed_dates, state_audit, watch_dir,
            pending_date=None, pending_prior_hashes={},
        ))
        start_index = 0

    config = mirror / CONFIG_REL.relative_to("project")
    env = os.environ.copy()
    env.update({"TA_PROJECT_ROOT": str(mirror), "TA_DATA_ROOT": str(TRADING_ROOT), "PYTHONPATH": str(mirror)})
    for index in range(start_index, len(dates)):
        date = dates[index]
        guard(before, mirror)
        prior_hashes = _relative_hashes(output, sorted(watch_dir.glob("rebalance_target_*.json")))
        _atomic_json_write(output / "checkpoint.json", _checkpoint_payload(
            output, dates, prefer_tag, copied, overlay, completed_dates, state_audit, watch_dir,
            pending_date=date, pending_prior_hashes=prior_hashes,
        ))
        cmd = [str(python), str(mirror / "rdlab/scripts/export_leading_daily_signal.py"), "--config", str(config), "--asof-date", date, "--watch-dir", str(watch_dir), "--cache-dir", str(cache_dir), "--prefer-tag", prefer_tag]
        proc = subprocess.Popen(cmd, cwd=mirror, env=env)
        try:
            while proc.poll() is None:
                time.sleep(1.0)
                guard(before, mirror)
        except Exception:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
        for rel_path, prior_hash in prior_hashes.items():
            path = output / rel_path
            if not path.is_file() or sha256(path) != prior_hash:
                raise RuntimeError(f"completed applied artifact changed during {date}: {path.name}")
        state_audit = audit_stateful_artifacts(watch_dir, dates[: index + 1])
        completed_dates.append(date)
        _atomic_json_write(output / "checkpoint.json", _checkpoint_payload(
            output, dates, prefer_tag, copied, overlay, completed_dates, state_audit, watch_dir,
            pending_date=None, pending_prior_hashes={},
        ))
        guard(before, mirror)

    final_manifest = {
        "schema": "wc006-isolated-selector-mirror/v2-resumable",
        "dates": dates,
        "prefer_tag": prefer_tag,
        "watch_dir_mode": "shared_stateful_strict_date_order",
        "state_audit": state_audit,
        "source_hashes_original": copied,
        "exporter_state_overlay": overlay["state"],
        "exporter_overlay": overlay["exporter"],
        "walkforward_overlay": overlay["walkforward"],
        **({"purge_overlay": overlay["purge"]} if "purge" in overlay else {}),
        "spec_hashes": {str(SPEC_REL): sha256(Path(SPEC_REL))},
        "checkpoint": "checkpoint.json",
    }
    _atomic_json_write(output / "mirror_provenance.json", final_manifest)
    return output

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--prefer-tag", required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--date", action="append", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--purge-train-labels", action="store_true", help="experiment-only: purge forward labels that reach past the signal month start")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(run_slice(args.output.resolve(), args.date, python=args.python, prefer_tag=args.prefer_tag, cache_dir=args.cache_dir, resume=args.resume, purge_train_labels=args.purge_train_labels))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
