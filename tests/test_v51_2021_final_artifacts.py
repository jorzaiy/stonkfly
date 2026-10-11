import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/wc006_v5_1_2021_baseline"
REPLAY = RUN_ROOT / "replay"

def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def test_final_manifest_pins_stable_artifacts_only():
    manifest = json.loads((RUN_ROOT / "final_manifest.json").read_text())
    artifacts = manifest["artifacts"]
    assert manifest["status"] == "PASS_COMPLETE_243_SESSIONS"
    assert manifest["artifact_policy"]["stable_only"] is True
    assert manifest["artifact_policy"]["connections_closed_before_hash"] is True
    assert set(manifest["artifact_policy"]["transient_excluded"]) == {"*-wal", "*-shm"}
    assert not any(name.endswith(("-wal", "-shm")) for name in artifacts)
    for name, expected in artifacts.items():
        path = RUN_ROOT / name if name in {"metrics.json", "comparison.json"} else REPLAY / name
        assert path.is_file(), name
        assert _sha(path) == expected, name
    for name in ("ledger.sqlite-wal", "ledger.sqlite-shm", "paired_base.sqlite-wal", "paired_base.sqlite-shm"):
        assert not (REPLAY / name).exists(), name

def test_final_chunk_and_logs_are_pinned():
    manifest = json.loads((RUN_ROOT / "final_manifest.json").read_text())
    assert set(manifest["chunk_manifests"]) == {
        "chunk_2021_60.json", "chunk_2021_120.json",
        "chunk_2021_180.json", "chunk_2021_243.json",
    }
    assert json.loads((RUN_ROOT / "chunks/chunk_2021_243.json").read_text())["run_result"]["status"] == "complete"
    assert set(manifest["logs"]) == {
        "chunk_060.stdout_stderr.log", "chunk_120.stdout_stderr.log",
        "chunk_180.stdout_stderr.log", "chunk_243.stdout_stderr.log",
    }
