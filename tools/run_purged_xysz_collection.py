#!/usr/bin/env python3
"""Plan (and, only when explicitly requested, run) a purged XYSZ gap fill.

The pinned collector is intentionally kept outside this repository.  This
launcher only passes it a *new* output directory and a read-only credential
mount.  It never imports AmazingData, opens the credential file, or writes to
the trading-agent repository.  The default action is planning; collection
requires ``--execute``.

``collect_xysz_v4_dataset.py`` is a one-dataset/one-login program.  Therefore
the batch invocations below are strictly serial and each batch invocation has exactly
one provider login.  This launcher does not pretend that the current pinned
collector supports a process-wide shared SDK session.  A future global SSO
mode must be separately audited rather than silently added here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable


DATASETS = (
    "kline",
    "history_stock_status",
    "dividend",
    "right_issue",
    "equity_structure",
    "adj_factor",
)
DATASET_BATCH_SIZE = {
    "kline": 5,
    "history_stock_status": 5,
    "dividend": 25,
    "right_issue": 25,
    "equity_structure": 25,
    "adj_factor": 25,
}
PRICE_DATASETS = {"kline", "history_stock_status"}
ACTION_DATASETS = {"dividend", "right_issue", "equity_structure"}
PLAN_SCHEMA = "wc006-purged-xysz-gap-plan/v1"
PLAN_STATUS = "BLOCKED_PENDING_COLLECTION_AND_QA"
COLLECTOR = Path("/home/ubuntu/trading_agent_20260408/project/scripts/collect_xysz_v4_dataset.py")
COLLECTOR_SHA256 = "98a9344d684f1a2e735b7db9acf6e5072eb6cb1170e325cb80fc7b78141c7c04"
DOCKER_IMAGE = "amazingdata-x86:1.1.9"
DOCKER_IMAGE_DIGEST = "sha256:57676d949eb5e79a3769c3714205f39d8940ae004e061a49c0174279379fd9bb"
TRADING_REPO = Path("/home/ubuntu/trading_agent_20260408").resolve()
CONTAINER_COLLECTOR = "/collector/collect_xysz_v4_dataset.py"
CONTAINER_CREDENTIALS = "/run/secrets/xysz"
CONTAINER_OUTPUT = "/out"


class LauncherBlocked(ValueError):
    """Raised when provenance or the destination safety boundary is invalid."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hex_digest(value: object, label: str) -> str:
    text = str(value or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise LauncherBlocked(f"{label}_must_be_sha256")
    return text


def _code6(value: object) -> str:
    text = str(value or "").strip().upper()
    if re.fullmatch(r"\d{6}", text):
        return text
    match = re.fullmatch(r"(\d{6})\.(?:SH|SZ|BJ)", text)
    if match:
        return match.group(1)
    raise LauncherBlocked(f"invalid_target_code:{value!r}")


def _qualified(code: str) -> str:
    if code.startswith(("6", "68")):
        return f"{code}.SH"
    if code.startswith(("4", "8")):
        return f"{code}.BJ"
    return f"{code}.SZ"


def _sorted_unique_codes(values: object, label: str) -> list[str]:
    if not isinstance(values, list):
        raise LauncherBlocked(f"{label}_must_be_list")
    codes = [_code6(value) for value in values]
    if codes != sorted(set(codes)):
        raise LauncherBlocked(f"{label}_must_be_sorted_unique")
    return codes


def load_gap_plan(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Load and fail closed on an immutable, blocked gap plan."""

    path = path.resolve()
    expected = _hex_digest(expected_sha256, "expected_plan_sha256")
    if not path.is_file():
        raise LauncherBlocked(f"plan_missing:{path}")
    observed = sha256(path)
    if observed != expected:
        raise LauncherBlocked("plan_sha256_mismatch")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LauncherBlocked("plan_invalid_json") from exc
    if not isinstance(payload, dict):
        raise LauncherBlocked("plan_must_be_object")
    if payload.get("schema") != PLAN_SCHEMA:
        raise LauncherBlocked("plan_schema_mismatch")
    if payload.get("status") != PLAN_STATUS:
        raise LauncherBlocked("plan_must_be_blocked_pending_collection_and_qa")
    if payload.get("offline_plan_only") is not True:
        raise LauncherBlocked("plan_must_be_offline_only")
    if payload.get("network_collection_performed") is not False:
        raise LauncherBlocked("plan_network_collection_flag_invalid")
    if payload.get("paper_only") is not True:
        raise LauncherBlocked("plan_must_be_paper_only")

    target = _sorted_unique_codes(payload.get("target_codes"), "target_codes")
    if payload.get("target_code_count") != len(target) or not target:
        raise LauncherBlocked("target_code_count_mismatch")
    qualified = payload.get("target_codes_qualified")
    if qualified != [_qualified(code) for code in target]:
        raise LauncherBlocked("target_codes_qualified_mismatch")

    missing = payload.get("missing_by_dataset")
    if not isinstance(missing, dict) or set(missing) != set(DATASETS):
        raise LauncherBlocked("missing_by_dataset_keys_mismatch")
    if not any(missing[dataset] for dataset in DATASETS):
        raise LauncherBlocked("gap_plan_has_no_missing_codes")
    target_set = set(target)
    for dataset in DATASETS:
        values = _sorted_unique_codes(missing[dataset], f"missing_{dataset}")
        if not set(values) <= target_set:
            raise LauncherBlocked(f"missing_{dataset}_outside_target")
        if payload.get("missing_code_count_by_dataset", {}).get(dataset) != len(values):
            raise LauncherBlocked(f"missing_{dataset}_count_mismatch")

    queries = payload.get("query_plan")
    if not isinstance(queries, dict) or set(queries) != set(DATASETS):
        raise LauncherBlocked("query_plan_keys_mismatch")
    for dataset in DATASETS:
        query = queries[dataset]
        if not isinstance(query, dict) or query.get("codes") != qualified:
            raise LauncherBlocked(f"query_plan_codes_mismatch:{dataset}")
        if dataset in PRICE_DATASETS:
            if query.get("begin_date") != 20210101 or query.get("end_date") != 20240103:
                raise LauncherBlocked(f"query_plan_window_mismatch:{dataset}")
        elif dataset in ACTION_DATASETS:
            if query.get("begin_date") != 20200101 or query.get("end_date") != 20240103:
                raise LauncherBlocked(f"query_plan_window_mismatch:{dataset}")
        elif query.get("required_window") != ["2021-01-01", "2024-01-03"]:
            raise LauncherBlocked("query_plan_window_mismatch:adj_factor")
    return payload


def _safe_output_root(output_root: Path) -> Path:
    root = output_root.resolve()
    if root == Path("/") or root == TRADING_REPO or TRADING_REPO in root.parents:
        raise LauncherBlocked("output_root_must_not_be_trading_repo")
    if root.exists():
        raise LauncherBlocked("output_root_already_exists")
    if not root.parent.is_dir():
        raise LauncherBlocked("output_root_parent_must_exist")
    return root


def _safe_credential_path(path: Path) -> Path:
    # The launcher deliberately checks only the path shape.  It never opens,
    # stats, or otherwise reads this file, so no credential value can leak.
    raw = str(path)
    if not raw or not path.is_absolute() or "\x00" in raw or "\n" in raw:
        raise LauncherBlocked("credential_file_must_be_absolute_single_line_path")
    return path


def _collector_path(path: Path) -> Path:
    path = path.resolve()
    if not path.is_file() or sha256(path) != COLLECTOR_SHA256:
        raise LauncherBlocked("collector_hash_mismatch")
    return path


def plan_collection(
    plan: dict[str, Any],
    output_root: Path,
    credential_file: Path,
    *,
    collector: Path = COLLECTOR,
) -> dict[str, Any]:
    """Build six deterministic Docker argv vectors without executing them."""

    root = _safe_output_root(output_root)
    cred = _safe_credential_path(credential_file)
    collector_path = _collector_path(collector)
    commands: list[dict[str, Any]] = []
    dataset_specs: dict[str, dict[str, Any]] = {}
    for index, dataset in enumerate(DATASETS, start=1):
        all_codes = [_qualified(code) for code in plan["missing_by_dataset"][dataset]]
        if not all_codes:
            continue
        batch_size = DATASET_BATCH_SIZE[dataset]
        batches = [all_codes[start:start + batch_size] for start in range(0, len(all_codes), batch_size)]
        dataset_specs[dataset] = {
            "dataset_index": index,
            "batch_size": batch_size,
            "batch_count": len(batches),
            "codes": all_codes,
        }
        for batch_no, codes in enumerate(batches, start=1):
            child = root / dataset / f"batch_{batch_no:03d}"
            # The root itself is required to be new; these checks also make a
            # planned command invalid if a caller races us by creating a child.
            if child.exists():
                raise LauncherBlocked(f"dataset_output_already_exists:{child}")
            argv = [
                "sudo", "-n", "docker", "run", "--rm", "--pull=never",
                "--platform", "linux/amd64", "--network", "bridge",
                "--mount", f"type=bind,src={collector_path},dst={CONTAINER_COLLECTOR},readonly",
                "--mount", f"type=bind,src={cred},dst={CONTAINER_CREDENTIALS},readonly",
                "--mount", f"type=bind,src={root},dst={CONTAINER_OUTPUT}",
                "-w", "/collector", DOCKER_IMAGE_DIGEST,
                "python", CONTAINER_COLLECTOR,
                "--dataset", dataset,
                "--output", f"{CONTAINER_OUTPUT}/{dataset}/batch_{batch_no:03d}",
                "--codes", *codes,
                "--begin-date", "20210101",
                "--end-date", "20240103",
                "--action-begin-date", "20200101",
                "--credential-file", CONTAINER_CREDENTIALS,
                "--hard-exit-after-complete",
            ]
            commands.append({
                "dataset": dataset,
                "batch": batch_no,
                "batch_count": len(batches),
                "output": str(child),
                "codes": codes,
                "argv": argv,
            })
    if not commands:
        raise LauncherBlocked("gap_plan_has_no_collectable_dataset")
    return {
        "schema": "wc006-purged-xysz-collection-launch/v1",
        "paper_only": True,
        "session_policy": "one_login_per_batch_serial",
        "plan_status": PLAN_STATUS,
        "docker_image": DOCKER_IMAGE,
        "docker_image_digest": DOCKER_IMAGE_DIGEST,
        "collector": str(collector_path),
        "collector_sha256": COLLECTOR_SHA256,
        "output_root": str(root),
        "dataset_specs": dataset_specs,
        "commands": commands,
    }


def execute_collection(spec: dict[str, Any], plan_sha256: str) -> int:
    """Run the already validated plan serially, stopping on first failure."""

    root = Path(spec["output_root"])
    if root.exists():
        raise LauncherBlocked("output_root_already_exists_before_execute")
    root.mkdir(mode=0o750)
    results: list[dict[str, Any]] = []
    for command in spec["commands"]:
        child = Path(command["output"])
        if child.exists():
            raise LauncherBlocked(f"dataset_output_already_exists:{child}")
        result = subprocess.run(command["argv"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        results.append({
            "dataset": command["dataset"],
            "batch": command["batch"],
            "batch_count": command["batch_count"],
            "codes": command["codes"],
            "output": command["output"],
            "returncode": result.returncode,
        })
        if result.returncode != 0:
            break
    (root / "launcher_result.json").write_text(
        json.dumps({"schema": "wc006-purged-xysz-collection-result/v1", "paper_only": True, "plan_sha256": plan_sha256, "spec": {k: v for k, v in spec.items() if k != "commands"}, "results": results}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if len(results) == len(spec["commands"]) and all(item["returncode"] == 0 for item in results) else 2


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--credential-file", required=True, type=Path)
    parser.add_argument("--execute", action="store_true", help="run serial Docker collection; default only prints a plan summary")
    args = parser.parse_args(list(argv) if argv is not None else None)
    plan = load_gap_plan(args.plan, args.expected_plan_sha256)
    spec = plan_collection(plan, args.output_root, args.credential_file)
    if not args.execute:
        print(json.dumps({"status": "planned_only", "paper_only": True, "dataset_order": [item["dataset"] for item in spec["commands"]], "command_count": len(spec["commands"]), "output_root": spec["output_root"], "collector_sha256": spec["collector_sha256"], "docker_image_digest": spec["docker_image_digest"]}, ensure_ascii=False, sort_keys=True))
        return 0
    return execute_collection(spec, args.expected_plan_sha256.lower())


if __name__ == "__main__":
    raise SystemExit(main())
