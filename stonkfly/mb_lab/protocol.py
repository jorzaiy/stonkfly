"""Frozen-candidate M1 protocol loader and fail-closed budget helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "protocols/mb_lab/m1_v1/frozen_config.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_config() -> dict[str, object]:
    config = json.loads(CONFIG_PATH.read_text())
    if config.get("status") != "CANDIDATE_FROZEN_AWAITING_INDEPENDENT_GO_NO_GO":
        raise RuntimeError("M1 candidate config is not in review-only frozen state")
    return config


def topology_control_seeds(config: dict[str, object]) -> tuple[int, ...]:
    count = int(config["topology_controls"]["count"])
    seeds = tuple(
        int.from_bytes(
            hashlib.sha256(
                f"STONKFLY-V2/M1/TOPOLOGY-C/v1/{index}".encode()
            ).digest()[:8],
            "little",
        )
        for index in range(count)
    )
    encoded = json.dumps(seeds, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != config["topology_controls"]["seed_list_sha256"]:
        raise RuntimeError("topology control seed commitment mismatch")
    return seeds


@dataclass(frozen=True)
class BudgetUsage:
    cpu_hours: float
    peak_ram_gib: float
    artifact_gib: float
    wall_hours: float


def enforce_stage_budget(usage: BudgetUsage, config: dict[str, object]) -> None:
    budget = config["budget"]
    limits = {
        "cpu_hours": float(budget["per_formal_stage_cpu_hours"]),
        "peak_ram_gib": float(budget["peak_ram_gib"]),
        "artifact_gib": float(budget["artifact_gib"]),
        "wall_hours": float(budget["per_stage_wall_hours"]),
    }
    for field, limit in limits.items():
        value = float(getattr(usage, field))
        if value < 0 or value > limit:
            raise RuntimeError(f"M1_BUDGET_EXHAUSTED: {field}={value} limit={limit}")
