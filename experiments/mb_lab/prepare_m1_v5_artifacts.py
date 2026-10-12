#!/usr/bin/env python3
"""Build immutable freeze-v5 public evidence and one-domain authorization."""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import sys

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding, NoEncryption, PrivateFormat, PublicFormat, load_pem_private_key,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stonkfly.mb_lab.authorization import AUTH_SCHEMA, canonical_payload
from stonkfly.mb_lab.provenance import sha256_file


RESULT_FILES = (
    "docs/STONKFLY_V2_M1_CHARTER_DRAFT.md",
    "docs/STONKFLY_V2_M1_FREEZE_REVIEW.md",
    "docs/STONKFLY_V2_M1_FREEZE_V5_AMENDMENT_DRAFT.md",
    "docs/STONKFLY_V2_M1_RQ_PREREG_DRAFT.md",
    "docs/STONKFLY_V2_M1_RQ_FREEZE_V5_PREREG.md",
    "protocols/mb_lab/m1_freeze_v5/AMENDMENT_DRAFT.json",
    "protocols/mb_lab/m1_freeze_v5/RQ_APPROVAL.json",
    "protocols/mb_lab/m1_freeze_v5/capacity_benchmark_v3.json",
    "protocols/mb_lab/m1_freeze_v5/topology_manifest.json",
    "protocols/mb_lab/m1_freeze_v5/authorization_public.pem",
    "protocols/mb_lab/m1_v1/frozen_config.json",
    "data/source.lock.json",
    "experiments/mb_lab/run_m1_capacity_benchmark.py",
    "experiments/mb_lab/run_m1_rq_formal.py",
    "experiments/mb_lab/seal_m1_v5_domains.py",
    "experiments/mb_lab/prepare_m1_v5_artifacts.py",
    "stonkfly/mb_lab/authorization.py",
    "stonkfly/mb_lab/budget.py",
    "stonkfly/mb_lab/checkpoint.py",
    "stonkfly/mb_lab/dynamics.py",
    "stonkfly/mb_lab/formal.py",
    "stonkfly/mb_lab/inventory.py",
    "stonkfly/mb_lab/metrics.py",
    "stonkfly/mb_lab/protocol.py",
    "stonkfly/mb_lab/provenance.py",
    "stonkfly/mb_lab/stages.py",
    "stonkfly/mb_lab/stimuli.py",
    "stonkfly/mb_lab/topology.py",
    "tests/mb_lab/test_m1_foundation.py",
    "tests/mb_lab/test_m1_freeze.py",
    "tests/mb_lab/test_m1_v5_infrastructure.py",
)


def write_once(path: Path, data: bytes, mode: int = 0o640) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


def keygen(private: Path, public: Path) -> None:
    key = Ed25519PrivateKey.generate()
    write_once(private, key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()), 0o600)
    write_once(public, key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo))


def topology(capacity: Path, output: Path) -> None:
    evidence = json.loads(capacity.read_text())
    hashes = evidence["network_hashes"]
    if len(hashes) != 40 or len(set(hashes)) != 40 or not evidence["degree_preserved_all"]:
        raise RuntimeError("capacity evidence does not contain valid B+39C topologies")
    document = {
        "schema": "stonkfly-m1-freeze-v5-topology-manifest-v1",
        "network_order": "index 0 is B; indices 1..39 are fixed degree-preserving C controls",
        "network_hashes": hashes,
        "capacity_evidence_sha256": sha256_file(capacity),
    }
    write_once(output, (json.dumps(document, indent=2, sort_keys=True) + "\n").encode())


def bundle(output: Path) -> None:
    files = {name: sha256_file(ROOT / name) for name in RESULT_FILES}
    document = {"schema": "stonkfly-m1-freeze-v5-provenance-bundle-v1", "files": files}
    write_once(output, (json.dumps(document, indent=2, sort_keys=True) + "\n").encode())


def authorize(private: Path, bundle_path: Path, manifest: Path, run_id: str,
              output_path: Path, destination: Path) -> None:
    key = load_pem_private_key(private.read_bytes(), password=None)
    payload = {
        "authorized": True, "domain": "M1_RQ",
        "protocol_bundle_sha256": sha256_file(bundle_path),
        "rq_manifest_sha256": sha256_file(manifest),
        "run_id": run_id, "output": str(output_path.resolve()),
    }
    document = {"schema": AUTH_SCHEMA, "payload": payload,
                "signature_base64": base64.b64encode(key.sign(canonical_payload(payload))).decode()}
    write_once(destination, (json.dumps(document, indent=2, sort_keys=True) + "\n").encode())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    key = sub.add_parser("keygen")
    key.add_argument("--private", type=Path, required=True); key.add_argument("--public", type=Path, required=True)
    top = sub.add_parser("topology")
    top.add_argument("--capacity", type=Path, required=True); top.add_argument("--out", type=Path, required=True)
    bun = sub.add_parser("bundle"); bun.add_argument("--out", type=Path, required=True)
    auth = sub.add_parser("authorize")
    auth.add_argument("--private", type=Path, required=True); auth.add_argument("--bundle", type=Path, required=True)
    auth.add_argument("--manifest", type=Path, required=True); auth.add_argument("--run-id", required=True)
    auth.add_argument("--output", type=Path, required=True); auth.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "keygen": keygen(args.private, args.public)
    elif args.command == "topology": topology(args.capacity, args.out)
    elif args.command == "bundle": bundle(args.out)
    else: authorize(args.private, args.bundle, args.manifest, args.run_id, args.output, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
