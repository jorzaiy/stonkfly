"""Fail-closed access policy for M1/M2 data domains."""

from __future__ import annotations

from enum import Enum
import json
from pathlib import Path


class DataDomain(str, Enum):
    BENCHMARK_ONLY = "BENCHMARK_ONLY"
    M1_RQ = "M1_RQ"
    M1_RT = "M1_RT"
    M2_LQ = "M2_LQ"
    M2_LT = "M2_LT"


def require_implementation_prep_access(domain: DataDomain | str) -> DataDomain:
    try:
        parsed = DataDomain(domain)
    except ValueError as exc:
        raise PermissionError(f"unknown data domain: {domain}") from exc
    if parsed is not DataDomain.BENCHMARK_ONLY:
        raise PermissionError(
            f"{parsed.value} is sealed; current authorization permits BENCHMARK_ONLY only"
        )
    return parsed


def public_manifest_path(root: Path, domain: DataDomain | str) -> Path:
    """Resolve only a public commitment manifest after an exact domain check."""
    parsed = DataDomain(domain)
    return root / f"{parsed.value.lower()}-manifest.json"


def read_public_manifest_for_stage(
    root: Path,
    *,
    authorized_domain: DataDomain | str,
    requested_domain: DataDomain | str,
) -> dict[str, object]:
    authorized = DataDomain(authorized_domain)
    requested = DataDomain(requested_domain)
    if authorized is not requested:
        raise PermissionError(
            f"cross-domain read denied: {authorized.value} cannot read {requested.value}"
        )
    manifest = json.loads(public_manifest_path(root, requested).read_text())
    if manifest.get("domain") != requested.value:
        raise ValueError("manifest domain label mismatch")
    return manifest
