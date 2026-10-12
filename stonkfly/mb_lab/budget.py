"""Runtime hard limits and append-only failure evidence for M1-RQ."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import resource
import time
import hashlib
import math
import signal


class BudgetExceeded(RuntimeError):
    pass


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


@dataclass
class RuntimeBudget:
    root: Path
    cpu_limit_seconds: float
    wall_limit_seconds: float
    rss_limit_bytes: int
    artifact_limit_bytes: int
    cpu_start: float = 0.0
    wall_start: float = 0.0
    prior_cpu_seconds: float = 0.0
    prior_wall_seconds: float = 0.0

    def start(self) -> None:
        self.cpu_start = time.process_time()
        self.wall_start = time.monotonic()

    def install_process_limits(self) -> None:
        """Install kernel backstops; directory size remains checked in-process."""
        remaining_cpu = max(1, math.ceil(self.cpu_limit_seconds - self.prior_cpu_seconds))
        resource.setrlimit(resource.RLIMIT_CPU, (remaining_cpu, remaining_cpu + 1))
        resource.setrlimit(resource.RLIMIT_AS, (self.rss_limit_bytes, self.rss_limit_bytes))

        def cpu_exhausted(_signum, _frame):
            raise BudgetExceeded("M1_BUDGET_EXHAUSTED kernel CPU limit")

        signal.signal(signal.SIGXCPU, cpu_exhausted)

    def usage(self) -> dict[str, float | int]:
        return {
            "cpu_seconds": self.prior_cpu_seconds + time.process_time() - self.cpu_start,
            "wall_seconds": self.prior_wall_seconds + time.monotonic() - self.wall_start,
            "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            "artifact_bytes": directory_bytes(self.root),
        }

    def check(self, *, pending_bytes: int = 0) -> dict[str, float | int]:
        usage = self.usage()
        limits = {
            "cpu_seconds": self.cpu_limit_seconds,
            "wall_seconds": self.wall_limit_seconds,
            "max_rss_bytes": self.rss_limit_bytes,
            "artifact_bytes": self.artifact_limit_bytes - pending_bytes,
        }
        for field, limit in limits.items():
            if usage[field] > limit:
                raise BudgetExceeded(f"M1_BUDGET_EXHAUSTED {field}={usage[field]} limit={limit}")
        return usage


def exclusive_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def append_failure(root: Path, value: dict[str, object]) -> Path:
    directory = root / "failures"
    directory.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        attempt = directory / f"attempt-{index:04d}"
        try:
            attempt.mkdir(mode=0o750)
            formal = attempt / "formal.json"
            exclusive_json(formal, value)
            descriptor = os.open(attempt / "report.md", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
            with os.fdopen(descriptor, "w") as handle:
                handle.write("# M1-RQ failed attempt\n\n")
                handle.write(f"Failure: `{value.get('verdict', 'M1_IMPLEMENTATION_FAILURE')}`\n")
            digest = hashlib.sha256(formal.read_bytes()).hexdigest()
            exclusive_json(attempt / "stdout.json", {"status": "FAILED", "formal_sha256": digest})
            files = {
                item.name: hashlib.sha256(item.read_bytes()).hexdigest()
                for item in sorted(attempt.iterdir()) if item.name != "manifest.json"
            }
            exclusive_json(attempt / "manifest.json", {"schema": "stonkfly-m1-failure-manifest-v1", "artifacts": files})
            return attempt
        except FileExistsError:
            index += 1
