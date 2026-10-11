from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path("/home/ubuntu/trading_agent_20260408/project/scripts/collect_xysz_v4_raw.py")
SPEC = importlib.util.spec_from_file_location("collect_xysz_v4_raw", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("002802.SZ", "002802.SZ"),
        ("sz.002802", "002802.SZ"),
        ("SH.600519", "600519.SH"),
        ("688981", "688981.SH"),
        ("830799", "830799.BJ"),
    ],
)
def test_normalize_xysz_code(raw: str, expected: str) -> None:
    assert MODULE.normalize_xysz_code(raw) == expected


@pytest.mark.parametrize("raw", ["foo.002802", "002802.US", "123456"])
def test_normalize_xysz_code_fails_closed(raw: str) -> None:
    with pytest.raises(ValueError):
        MODULE.normalize_xysz_code(raw)
