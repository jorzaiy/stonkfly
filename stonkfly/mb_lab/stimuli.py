"""Canonical synthetic PN fixtures; formal seeds remain sealed."""

from __future__ import annotations

import hashlib
import io

import numpy as np


def generate_m1_collection(seed: int, specification: dict[str, object]) -> dict[str, np.ndarray]:
    """Generate a collection without running neural dynamics or inspecting outputs."""
    rng = np.random.default_rng(seed)
    n_pn = int(specification["pn_count"])
    n_odor = int(specification["odor_count"])
    n_repeat = int(specification["repeat_count"])
    n_active = int(specification["active_pn_count_per_odor"])
    low, high = map(float, specification["base_amplitude_uniform"])
    jitter = float(specification["repeat_jitter_normal_sd"])
    clip_low, clip_high = map(float, specification["clip"])
    if not (0 < n_active < n_pn and 0 <= low < high <= clip_high):
        raise ValueError("invalid frozen stimulus specification")
    base = np.zeros((n_odor, n_pn), dtype=np.float64)
    repeated = np.zeros((n_odor, n_repeat, n_pn), dtype=np.float64)
    for odor in range(n_odor):
        indices = np.sort(rng.choice(n_pn, size=n_active, replace=False))
        base[odor, indices] = rng.uniform(low, high, size=n_active)
        noise = rng.normal(0.0, jitter, size=(n_repeat, n_active))
        repeated[odor, :, indices] = np.clip(
            base[odor, indices][None, :] + noise, clip_low, clip_high
        ).T
    return {"base": base, "repeated": repeated}


def canonical_collection_bytes(collection: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, base=collection["base"], repeated=collection["repeated"])
    return buffer.getvalue()


def collection_sha256(collection: dict[str, np.ndarray]) -> str:
    return hashlib.sha256(canonical_collection_bytes(collection)).hexdigest()
