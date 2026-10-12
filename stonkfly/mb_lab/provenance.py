"""Self-contained freeze-v5 result-affecting source bundle verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bundle_sha256(path: Path) -> str:
    return sha256_file(path)


def verify_bundle(root: Path, bundle_path: Path) -> dict[str, object]:
    bundle = json.loads(bundle_path.read_text())
    if bundle.get("schema") != "stonkfly-m1-freeze-v5-provenance-bundle-v1":
        raise RuntimeError("invalid provenance bundle schema")
    for relative, expected in bundle["files"].items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"provenance mismatch: {relative}")
    return bundle
