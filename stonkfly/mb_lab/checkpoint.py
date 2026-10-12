"""Immutable chunk checkpoints; resume only under identical provenance."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_chunk_once(root: Path, name: str, arrays: dict[str, np.ndarray], metadata: dict[str, object]) -> Path:
    chunks = root / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    target = chunks / f"{name}.npz"
    sidecar = chunks / f"{name}.json"
    if target.exists() or sidecar.exists():
        raise FileExistsError(f"immutable checkpoint exists: {name}")
    temporary = chunks / f".{name}.{os.getpid()}.partial.npz"
    np.savez_compressed(temporary, **arrays)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.link(temporary, target)
    temporary.unlink()
    record = dict(metadata)
    record["chunk_sha256"] = sha256_file(target)
    descriptor = os.open(sidecar, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return target


def validated_chunk(root: Path, name: str, expected: dict[str, object]) -> Path | None:
    target = root / "chunks" / f"{name}.npz"
    sidecar = root / "chunks" / f"{name}.json"
    if not target.exists() and not sidecar.exists():
        return None
    if not target.is_file() or not sidecar.is_file():
        raise RuntimeError(f"incomplete checkpoint pair: {name}")
    record = json.loads(sidecar.read_text())
    for key, value in expected.items():
        if record.get(key) != value:
            raise RuntimeError(f"checkpoint provenance mismatch: {name}:{key}")
    if record.get("chunk_sha256") != sha256_file(target):
        raise RuntimeError(f"checkpoint hash mismatch: {name}")
    return target
