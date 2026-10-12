"""Signed, domain-exact authorization for freeze-v5 formal execution."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from cryptography.hazmat.primitives.serialization import load_pem_public_key

from .stages import DataDomain

AUTH_SCHEMA = "stonkfly-m1-stage-authorization-v1"


def canonical_payload(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def verify_rq_authorization(
    authorization_path: Path,
    public_key_path: Path,
    *,
    expected_bundle_sha256: str,
    expected_manifest_sha256: str,
    expected_run_id: str,
    expected_output: str,
) -> dict[str, object]:
    document = json.loads(authorization_path.read_text())
    payload = document.get("payload")
    if not isinstance(payload, dict) or document.get("schema") != AUTH_SCHEMA:
        raise PermissionError("invalid authorization schema")
    expected = {
        "authorized": True,
        "domain": DataDomain.M1_RQ.value,
        "protocol_bundle_sha256": expected_bundle_sha256,
        "rq_manifest_sha256": expected_manifest_sha256,
        "run_id": expected_run_id,
        "output": expected_output,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise PermissionError(f"authorization mismatch: {key}")
    if any(name in payload for name in ("M1_RT", "M2_LQ", "M2_LT")):
        raise PermissionError("authorization contains a forbidden domain")
    signature = base64.b64decode(document.get("signature_base64", ""), validate=True)
    key = load_pem_public_key(public_key_path.read_bytes())
    key.verify(signature, canonical_payload(payload))
    return payload
