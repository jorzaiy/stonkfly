#!/usr/bin/env python3
"""One-shot sealer for four isolated M1/M2 seed domains; never runs dynamics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stonkfly.mb_lab.protocol import CONFIG_PATH, load_frozen_config, sha256_file
from stonkfly.mb_lab.stages import DataDomain
from stonkfly.mb_lab.stimuli import collection_sha256, generate_m1_collection

SCHEMA = "stonkfly-sealed-domain-manifest-v1"
DOMAINS = (DataDomain.M1_RQ, DataDomain.M1_RT, DataDomain.M2_LQ, DataDomain.M2_LT)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def refuse_existing(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise RuntimeError("refusing to overwrite sealed evidence: " + ", ".join(existing))


def seal(public_root: Path, private_root: Path) -> list[dict[str, object]]:
    config = load_frozen_config()
    public_root = public_root.resolve()
    private_root = private_root.resolve()
    public_root.mkdir(parents=True, exist_ok=True)
    private_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    expected = []
    for domain in DOMAINS:
        stem = domain.value.lower()
        expected.extend(
            [
                public_root / f"{stem}-manifest.json",
                private_root / f"{stem}.sealed",
                private_root / f"{stem}.key",
            ]
        )
    refuse_existing(expected)

    manifests = []
    for domain in DOMAINS:
        seed_bytes = os.urandom(32)
        seed_integer = int.from_bytes(seed_bytes[:8], "little")
        if domain in (DataDomain.M1_RQ, DataDomain.M1_RT):
            content_hash = collection_sha256(
                generate_m1_collection(seed_integer, config["stimulus"])
            )
            content_status = "M1_COLLECTION_GENERATED_IN_MEMORY_AND_HASHED_NOT_RUN"
        else:
            reservation = {
                "domain": domain.value,
                "seed_commitment_sha256": sha256_bytes(seed_bytes),
                "status": "M2_RESERVED_NOT_IMPLEMENTED",
            }
            content_hash = sha256_bytes(canonical_json(reservation))
            content_status = "M2_SEED_RESERVATION_ONLY_NO_TASK_CONTENT"
        envelope = canonical_json(
            {
                "domain": domain.value,
                "seed_hex": seed_bytes.hex(),
                "content_sha256": content_hash,
                "config_sha256": sha256_file(CONFIG_PATH),
            }
        )
        key = AESGCM.generate_key(bit_length=256)
        nonce = os.urandom(12)
        ciphertext = nonce + AESGCM(key).encrypt(nonce, envelope, domain.value.encode())
        stem = domain.value.lower()
        sealed_path = private_root / f"{stem}.sealed"
        key_path = private_root / f"{stem}.key"
        sealed_path.write_bytes(ciphertext)
        key_path.write_bytes(key)
        os.chmod(sealed_path, 0o600)
        os.chmod(key_path, 0o600)
        manifest = {
            "schema": SCHEMA,
            "domain": domain.value,
            "access": "SEALED_AWAITING_EXPLICIT_STAGE_AUTHORIZATION",
            "seed_exposed": False,
            "seed_commitment_sha256": sha256_bytes(seed_bytes),
            "content_sha256": content_hash,
            "content_status": content_status,
            "config_sha256": sha256_file(CONFIG_PATH),
            "sealer_sha256": sha256_file(Path(__file__)),
            "stimulus_generator_sha256": sha256_file(ROOT / "stonkfly/mb_lab/stimuli.py"),
            "encryption": "AES-256-GCM with unique 96-bit nonce and domain label as AAD",
            "sealed_payload_sha256": sha256_file(sealed_path),
            "sealed_payload_relative_path": str(sealed_path.relative_to(ROOT)),
            "key_relative_path": str(key_path.relative_to(ROOT)),
            "custodian": "local_manual_review",
        }
        manifest_path = public_root / f"{stem}-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        manifests.append(manifest)
    return manifests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--public-root",
        type=Path,
        default=ROOT / "protocols/mb_lab/m1_v1/domains",
    )
    parser.add_argument(
        "--private-root",
        type=Path,
        default=ROOT / "data/mb_lab/sealed/m1_v1",
    )
    args = parser.parse_args(argv)
    manifests = seal(args.public_root, args.private_root)
    print(json.dumps({"sealed_domains": [item["domain"] for item in manifests]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
