#!/usr/bin/env python3
"""Build an isolated, hash-pinned WC006 candidate/weight stream.

This adapter consumes already-produced walk-forward rebalance artifacts.  It
never treats an empty schedule placeholder as a candidate stream and never
writes the trading repository.  A future selector adapter may feed equivalent
artifacts after proving parity with ``export_leading_daily_signal.py``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable


class StreamBlocked(ValueError):
    """Raised when candidate provenance or causal timing is incomplete."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StreamBlocked(f"JSON object required: {path}")
    return value


def _canonical_code(value: object) -> str:
    raw = str(value).strip().upper()
    for suffix in (".SH", ".SZ", ".BJ"):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
    if not raw.isdigit() or len(raw) != 6:
        raise StreamBlocked(f"invalid candidate code: {value!r}")
    return raw


def _date(value: object) -> str:
    raw = str(value).strip().replace("/", "-")
    if len(raw) == 8 and raw.isdigit():
        raw = f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    if len(raw) != 10 or raw[4] != "-" or raw[7] != "-":
        raise StreamBlocked(f"invalid ISO date: {value!r}")
    return raw


def next_session(calendar_dates: Iterable[str], signal_date: str) -> str:
    signal = _date(signal_date)
    later = sorted({_date(item) for item in calendar_dates if _date(item) > signal})
    if not later:
        raise StreamBlocked(f"no next valid session after {signal}")
    return later[0]


def _source_path(raw: str | Path, trading_root: Path) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return (trading_root / path).resolve()


def _copy_hash(source: Path, destination: Path) -> str:
    if not source.is_file():
        raise StreamBlocked(f"missing pinned source: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return sha256(destination)


def _validate_target_rows(payload: dict[str, Any], *, topk: int) -> list[dict[str, Any]]:
    rows = payload.get("effective_target_topk")
    if not isinstance(rows, list) or not rows:
        raise StreamBlocked(
            f"candidate artifact has no effective target weights: {payload.get('rebalance_date', payload.get('date'))}"
        )
    if len(rows) > topk:
        raise StreamBlocked("candidate artifact exceeds configured topk")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise StreamBlocked("candidate target row is not an object")
        code = _canonical_code(row.get("code"))
        if code in seen:
            raise StreamBlocked(f"duplicate candidate code: {code}")
        try:
            weight = float(row["target_weight"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StreamBlocked(f"candidate weight missing: {code}") from exc
        if not (weight > 0.0) or weight != weight:
            raise StreamBlocked(f"candidate weight is not positive finite: {code}")
        result.append({
            "code": code,
            "rank": int(row.get("rank", rank)),
            "score": row.get("score"),
            "target_weight": weight,
        })
        seen.add(code)
    return result


def build_candidate_stream(
    *,
    artifact_paths: Iterable[Path],
    active_config: Path,
    universe: Path,
    calendar: Path,
    cache_dir: Path,
    output_dir: Path,
    trading_root: Path,
    max_records: int | None = None,
    selector_checkpoint: Path | None = None,
    selector_mirror_provenance: Path | None = None,
) -> Path:
    """Copy a small, immutable causal stream into a new Stonkfly directory.

    The source artifacts are the walk-forward selector's output.  The adapter
    copies only target-specific canonical bars needed by this slice; the
    artifact hash and active input hashes remain part of the manifest.  Empty
    historical placeholders fail closed instead of becoming synthetic weights.
    """
    output = Path(output_dir).resolve()
    root = Path(trading_root).resolve()
    if output == root:
        raise StreamBlocked("output must be outside trading root")
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise StreamBlocked("candidate stream output cannot be inside trading repo")
    if output.exists():
        raise StreamBlocked("candidate stream output must be new and unique")
    output.parent.mkdir(parents=True, exist_ok=True)

    if (selector_checkpoint is None) != (selector_mirror_provenance is None):
        raise StreamBlocked(
            "selector checkpoint and mirror provenance must be provided together"
        )
    selector_checkpoint = (
        Path(selector_checkpoint).resolve() if selector_checkpoint is not None else None
    )
    selector_mirror_provenance = (
        Path(selector_mirror_provenance).resolve()
        if selector_mirror_provenance is not None
        else None
    )

    config = _json(active_config)
    wf = config.get("walkforward")
    strategy = config.get("strategy")
    if not isinstance(wf, dict) or wf.get("signal_lag_one_day") is not True:
        raise StreamBlocked("active config must explicitly enable signal_lag_one_day")
    if not isinstance(strategy, dict):
        raise StreamBlocked("active config strategy missing")
    topk = int(strategy.get("topk", 0))
    hold = int(strategy.get("hold", 0))
    if topk <= 0 or hold <= 0:
        raise StreamBlocked("active config topk/hold missing")
    universe_codes = {
        _canonical_code(line)
        for line in Path(universe).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    calendar_payload = _json(calendar)
    calendar_dates = [_date(item) for item in calendar_payload.get("dates", [])]
    if not calendar_dates:
        raise StreamBlocked("calendar has no dates")

    paths = [Path(item).resolve() for item in artifact_paths]
    if max_records is not None:
        paths = paths[: int(max_records)]
    if not paths:
        raise StreamBlocked("candidate artifact slice is empty")
    records: list[dict[str, Any]] = []
    selected_sources: dict[str, Path] = {}
    exact_tag: str | None = None
    previous_signal: str | None = None
    for artifact in paths:
        payload = _json(artifact)
        signal_date = _date(payload.get("rebalance_date", payload.get("date")))
        if previous_signal is not None and signal_date <= previous_signal:
            raise StreamBlocked("candidate artifacts are not strictly chronological")
        previous_signal = signal_date
        if payload.get("source") != "rdlab_leading_rebalance":
            raise StreamBlocked("unsupported candidate artifact source")
        if payload.get("rebalance_applied") is not True:
            raise StreamBlocked(f"rebalance not applied at {signal_date}")
        if int(payload.get("topk", topk)) != topk or int(payload.get("hold", hold)) != hold:
            raise StreamBlocked(f"candidate artifact config mismatch at {signal_date}")
        execution_date = next_session(calendar_dates, signal_date)
        rows = _validate_target_rows(payload, topk=topk)
        for row in rows:
            code = row["code"]
            if code not in universe_codes:
                raise StreamBlocked(f"candidate outside pinned universe: {code}")
        prefer_tag = str(payload.get("prefer_tag") or "")
        if not prefer_tag:
            raise StreamBlocked(f"candidate artifact has no exact cache tag at {signal_date}")
        if exact_tag is None:
            exact_tag = prefer_tag
        elif prefer_tag != exact_tag:
            raise StreamBlocked(
                f"candidate stream mixes exact cache tags: {exact_tag} vs {prefer_tag} at {signal_date}"
            )
        cache_hashes: dict[str, str] = {}
        for row in rows:
            code = row["code"]
            source = Path(cache_dir) / f"{code}_{prefer_tag}.csv"
            if not source.is_file():
                raise StreamBlocked(f"missing exact-tag cache for {code}: {prefer_tag}")
            with source.open(newline="", encoding="utf-8") as handle:
                dates = {str(item.get("date")) for item in csv.DictReader(handle)}
            if signal_date not in dates or execution_date not in dates:
                raise StreamBlocked(f"cache lacks signal/next session for {code} at {signal_date}")
            selected_sources.setdefault(code, source)
            cache_hashes[code] = sha256(source)
        records.append({
            "signal_date": signal_date,
            "execution_date": execution_date,
            "selector": {
                "artifact_sha256": sha256(artifact),
                "config_path": str(payload.get("config", "")),
                "signal_seed": payload.get("signal_seed"),
                "prefer_tag": prefer_tag,
                "rebalance_applied": True,
                "target_topk": rows,
                "cache_sha256": cache_hashes,
            },
            "timing_proof": {
                "signal_lag_one_day": True,
                "execution_is_first_next_valid_session": True,
            },
        })

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent)))
    try:
        inputs = staging / "inputs"
        source_hashes = {
            "active_config": _copy_hash(Path(active_config), inputs / "active_config.json"),
            "universe": _copy_hash(Path(universe), inputs / "universe.txt"),
            "calendar": _copy_hash(Path(calendar), inputs / "calendar.json"),
            "adapter": sha256(Path(__file__)),
        }
        selector_generation_provenance: dict[str, Any] | None = None
        if selector_checkpoint is not None and selector_mirror_provenance is not None:
            provenance_dir = inputs / "selector_generation_provenance"
            checkpoint_destination = provenance_dir / "checkpoint.json"
            mirror_destination = provenance_dir / "mirror_provenance.json"
            selector_generation_provenance = {
                "checkpoint": {
                    "source_name": selector_checkpoint.name,
                    "sha256": _copy_hash(selector_checkpoint, checkpoint_destination),
                    "copied_path": str(checkpoint_destination.relative_to(staging)),
                },
                "mirror_provenance": {
                    "source_name": selector_mirror_provenance.name,
                    "sha256": _copy_hash(selector_mirror_provenance, mirror_destination),
                    "copied_path": str(mirror_destination.relative_to(staging)),
                },
            }
        cache_manifest: dict[str, dict[str, str]] = {}
        for code, source in selected_sources.items():
            destination = inputs / "cache" / source.name
            cache_manifest[code] = {
                "source_name": source.name,
                "sha256": _copy_hash(source, destination),
                "copied_path": str(destination.relative_to(staging)),
            }
        artifact_manifest: list[dict[str, str]] = []
        for index, artifact in enumerate(paths):
            destination = inputs / "selector" / f"{index:03d}_{artifact.name}"
            artifact_manifest.append({
                "source_name": artifact.name,
                "sha256": _copy_hash(artifact, destination),
                "copied_path": str(destination.relative_to(staging)),
            })
        manifest = {
            "schema": "wc006-candidate-stream/v1",
            "paper_only": True,
            "broker_orders_enabled": False,
            "live_execution_authorized": False,
            "authoritative_trading_state": False,
            "selector": "repaired_wc006_walkforward_export_artifact",
            "signal_lag_one_day": True,
            "topk": topk,
            "hold": hold,
            "exact_tag": exact_tag,
            "records": records,
            "source_hashes": source_hashes,
            **({"selector_generation_provenance": selector_generation_provenance}
               if selector_generation_provenance is not None else {}),
            "cache_files": cache_manifest,
            "selector_artifacts": artifact_manifest,
            "limitations": [
                "selector parity is hash-bound to existing artifacts; empty or mismatched artifacts fail closed",
                "official ST/suspension/limit state is not included; this stream is non-authoritative",
                "no returns, PnL, training, or broker execution are part of this adapter",
            ],
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / "manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", required=True, type=Path)
    parser.add_argument("--active-config", required=True, type=Path)
    parser.add_argument("--universe", required=True, type=Path)
    parser.add_argument("--calendar", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--trading-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--selector-checkpoint", type=Path)
    parser.add_argument("--selector-mirror-provenance", type=Path)
    args = parser.parse_args()
    print(build_candidate_stream(
        artifact_paths=args.artifact,
        active_config=args.active_config,
        universe=args.universe,
        calendar=args.calendar,
        cache_dir=args.cache_dir,
        output_dir=args.output,
        trading_root=args.trading_root,
        selector_checkpoint=args.selector_checkpoint,
        selector_mirror_provenance=args.selector_mirror_provenance,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
