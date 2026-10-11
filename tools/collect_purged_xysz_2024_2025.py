#!/usr/bin/env python3
"""Hash-bound, serial raw XYSZ collection for the frozen 2024–2025 candidate list.

The pinned collector only performs market-data queries. This wrapper never
opens the credential file; Docker mounts it read-only and suppresses SDK output.
Each completed fragment is hash-checked before a resumed run skips it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from tools.run_purged_xysz_collection import (
    COLLECTOR, COLLECTOR_SHA256, CONTAINER_COLLECTOR,
    CONTAINER_CREDENTIALS, DATASET_BATCH_SIZE, DATASETS, DOCKER_IMAGE_DIGEST,
    TRADING_REPO,
)
from experiments.a_share.wc006_male_cns_adapter import validate_candidate_stream

ROOT = Path(__file__).resolve().parents[1]
BEGIN = 20240101
END = 20260102
ACTION_BEGIN = 20230101
SCHEMA = "fly-purged-xysz-2024-2025-collection/v1"


class CollectionBlocked(ValueError):
    pass


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def qualified(value: str) -> str:
    code = str(value).split(".")[0]
    if not re.fullmatch(r"\d{6}", code):
        raise CollectionBlocked("invalid_candidate_code")
    return code + (".SH" if code.startswith("6") else ".BJ" if code.startswith(("4", "8")) else ".SZ")


def plan(candidate_path: Path) -> dict:
    candidate_path = candidate_path.resolve()
    validate_candidate_stream(candidate_path, expected_records=36,
                              require_selector_generation_provenance=True)
    candidate = json.loads(candidate_path.read_text())
    records = candidate.get("records")
    if (candidate.get("paper_only") is not True or
            candidate.get("live_execution_authorized") is not False or
            not isinstance(records, list) or len(records) != 36):
        raise CollectionBlocked("candidate_not_frozen_36_date_paper_stream")
    dates = [x.get("signal_date") for x in records]
    if dates != sorted(set(dates)) or dates[0] != "2024-01-02" or dates[-1] != "2025-12-29":
        raise CollectionBlocked("candidate_signal_schedule_mismatch")
    codes = sorted({qualified(item["code"]) for row in records for item in row["selector"]["target_topk"]})
    if not codes:
        raise CollectionBlocked("candidate_empty")
    if (candidate.get("authorization_artifact") or candidate.get("market_authorization") or
            not isinstance(candidate.get("selector_generation_provenance"), dict) or
            not candidate["selector_generation_provenance"] or
            {x.split(".")[0] for x in codes} != set(candidate.get("cache_files", {}))):
        raise CollectionBlocked("candidate_provenance_or_code_union_mismatch")
    if not COLLECTOR.is_file() or sha(COLLECTOR) != COLLECTOR_SHA256:
        raise CollectionBlocked("pinned_collector_hash_mismatch")
    return {
        "schema": SCHEMA,
        "candidate_manifest": str(candidate_path),
        "candidate_manifest_sha256": sha(candidate_path),
        "collector_sha256": COLLECTOR_SHA256,
        "image_digest": DOCKER_IMAGE_DIGEST,
        "paper_only": True,
        "begin_date": BEGIN,
        "end_date": END,
        "action_begin_date": ACTION_BEGIN,
        "codes": codes,
        "datasets": list(DATASETS),
        "batch_sizes": DATASET_BATCH_SIZE,
    }


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def verify_fragment(child: Path, dataset: str, codes: list[str]) -> dict:
    manifest_path = child / "manifest.json"
    if not manifest_path.is_file():
        raise CollectionBlocked(f"fragment_manifest_missing:{child}")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("status") != "complete" or manifest.get("dataset") != dataset or
            manifest.get("codes") != codes or manifest.get("begin_date") != BEGIN or
            manifest.get("end_date") != END or manifest.get("action_begin_date") != ACTION_BEGIN):
        raise CollectionBlocked(f"fragment_scope_mismatch:{child}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise CollectionBlocked(f"fragment_artifacts_missing:{child}")
    for item in artifacts:
        rel = Path(str(item.get("path", "")))
        if rel.is_absolute() or ".." in rel.parts:
            raise CollectionBlocked("unsafe_fragment_path")
        artifact = child / "raw" / rel
        if not artifact.is_file() or sha(artifact) != item.get("sha256"):
            raise CollectionBlocked(f"fragment_artifact_hash_mismatch:{artifact}")
    return {"dataset": dataset, "codes": codes, "manifest": str(manifest_path),
            "manifest_sha256": sha(manifest_path)}


def collect(spec: dict, output: Path, credential_file: Path) -> int:
    output = output.absolute()
    if TRADING_REPO == output or TRADING_REPO in output.parents or ROOT not in output.parents:
        raise CollectionBlocked("output_must_be_new_stonkfly_run")
    if not credential_file.is_absolute() or not credential_file.is_file():
        raise CollectionBlocked("credential_file_missing")
    if output.exists():
        saved = output / "collection_plan.json"
        if not saved.is_file() or json.loads(saved.read_text()) != spec:
            raise CollectionBlocked("existing_output_plan_mismatch")
    else:
        output.mkdir(parents=True)
        write_json(output / "collection_plan.json", spec)
    results = []
    for dataset in DATASETS:
        size = DATASET_BATCH_SIZE[dataset]
        for index, start in enumerate(range(0, len(spec["codes"]), size), 1):
            codes = spec["codes"][start:start + size]
            child = output / dataset / f"batch_{index:03d}"
            if child.exists():
                result = verify_fragment(child, dataset, codes)
            else:
                command = [
                    "sudo", "-n", "docker", "run", "--rm", "--pull=never",
                    "--platform", "linux/amd64", "--network", "bridge",
                    "--mount", f"type=bind,src={COLLECTOR},dst={CONTAINER_COLLECTOR},readonly",
                    "--mount", f"type=bind,src={credential_file},dst={CONTAINER_CREDENTIALS},readonly",
                    "--mount", f"type=bind,src={output},dst=/out",
                    "-w", "/collector", DOCKER_IMAGE_DIGEST,
                    "python", CONTAINER_COLLECTOR, "--dataset", dataset,
                    "--output", f"/out/{dataset}/batch_{index:03d}",
                    "--codes", *codes,
                    "--begin-date", str(BEGIN), "--end-date", str(END),
                    "--action-begin-date", str(ACTION_BEGIN),
                    "--credential-file", CONTAINER_CREDENTIALS,
                    "--hard-exit-after-complete",
                ]
                proc = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                if proc.returncode:
                    raise CollectionBlocked(f"collector_failed:{dataset}:batch_{index:03d}:rc={proc.returncode}")
                result = verify_fragment(child, dataset, codes)
            results.append(result)
            write_json(output / "collection_progress.json", {"schema": SCHEMA, "plan_sha256": sha(output / "collection_plan.json"), "completed": results})
            print(f"{dataset} {index} complete", flush=True)
    write_json(output / "collection_result.json", {"schema": SCHEMA, "paper_only": True,
               "plan_sha256": sha(output / "collection_plan.json"), "fragments": results})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--credential-file", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    spec = plan(args.candidate)
    if not args.execute:
        print(json.dumps({"status": "planned", "code_count": len(spec["codes"]),
                          "batch_count": sum((len(spec["codes"]) + DATASET_BATCH_SIZE[d] - 1) // DATASET_BATCH_SIZE[d] for d in DATASETS),
                          "candidate_sha256": spec["candidate_manifest_sha256"]}))
        return 0
    return collect(spec, args.output, args.credential_file)


if __name__ == "__main__":
    raise SystemExit(main())
