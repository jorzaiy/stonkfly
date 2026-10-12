#!/usr/bin/env python3
"""One-shot freeze-v5 domain sealer. It never runs neural dynamics."""

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

from stonkfly.mb_lab.protocol import load_frozen_config
from stonkfly.mb_lab.provenance import sha256_file
from stonkfly.mb_lab.stages import DataDomain
from stonkfly.mb_lab.stimuli import collection_sha256, generate_m1_collection

DOMAINS = (DataDomain.M1_RQ, DataDomain.M1_RT, DataDomain.M2_LQ, DataDomain.M2_LT)


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def seal(public_root: Path, custody_root: Path, bundle_path: Path, topology_path: Path) -> None:
    public_root, custody_root = public_root.resolve(), custody_root.resolve()
    if public_root.exists() or custody_root.exists():
        raise RuntimeError("refusing to overwrite freeze-v5 seal roots")
    public_root.mkdir(parents=True)
    custody_root.mkdir(parents=True, mode=0o700)
    config = load_frozen_config()
    bundle_hash = sha256_file(bundle_path)
    topology_hash = sha256_file(topology_path)
    for domain in DOMAINS:
        seed = os.urandom(32)
        seed_int = int.from_bytes(seed[:8], "little")
        if domain in (DataDomain.M1_RQ, DataDomain.M1_RT):
            content_hash = collection_sha256(generate_m1_collection(seed_int, config["stimulus"]))
            content_status = "M1_COLLECTION_HASHED_NOT_RUN"
        else:
            content_hash = hashlib.sha256(canonical({"domain": domain.value, "seed_commitment": hashlib.sha256(seed).hexdigest()})).hexdigest()
            content_status = "M2_RESERVATION_ONLY"
        envelope = canonical({
            "domain": domain.value,
            "seed_hex": seed.hex(),
            "content_sha256": content_hash,
            "bundle_sha256": bundle_hash,
        })
        key = AESGCM.generate_key(bit_length=256)
        nonce = os.urandom(12)
        encrypted = nonce + AESGCM(key).encrypt(nonce, envelope, domain.value.encode())
        private = custody_root / domain.value.lower()
        private.mkdir(mode=0o700)
        sealed_path, key_path = private / "payload.sealed", private / "domain.key"
        sealed_path.write_bytes(encrypted)
        key_path.write_bytes(key)
        os.chmod(sealed_path, 0o600)
        os.chmod(key_path, 0o600)
        manifest = {
            "schema": "stonkfly-m1-freeze-v5-domain-manifest-v1",
            "domain": domain.value,
            "access": "SEALED_AWAITING_STAGE_AUTHORIZATION",
            "seed_exposed": False,
            "seed_commitment_sha256": hashlib.sha256(seed).hexdigest(),
            "content_sha256": content_hash,
            "content_status": content_status,
            "protocol_bundle_sha256": bundle_hash,
            "topology_manifest_sha256": topology_hash,
            "sealed_payload_sha256": sha256_file(sealed_path),
            "sealer_sha256": sha256_file(Path(__file__)),
            "stimulus_generator_sha256": sha256_file(ROOT / "stonkfly/mb_lab/stimuli.py"),
            "encryption": "AES-256-GCM; unique 96-bit nonce; domain label AAD",
        }
        (public_root / f"{domain.value.lower()}-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    supersession = {
        "schema": "stonkfly-m1-freeze-v5-supersession-v1",
        "status": "SUPERSEDED_PROTOCOL_NOT_RUN",
        "v4_manifest_hashes": {
            path.name: sha256_file(path)
            for path in sorted((ROOT / "protocols/mb_lab/m1_v1/domains-freeze-v4").glob("*.json"))
        },
    }
    (public_root / "supersedes-freeze-v4.json").write_text(json.dumps(supersession, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--custody-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    args = parser.parse_args()
    seal(args.public_root, args.custody_root, args.bundle, args.topology)
    print(json.dumps({"sealed": [item.value for item in DOMAINS]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
