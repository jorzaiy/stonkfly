"""Isolated Q2 signed graded-deviation brain and linear-audit bindings."""
from __future__ import annotations

import ctypes as C
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from stonkfly.neural.v2_brain import HybridBrain, sha256_file

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("v2_q2_kernel.cpp")
LINEAR_SOURCE = Path(__file__).with_name("q2_linear_audit.cpp")
BUILD_DIR = ROOT / "runs/whole_brain_v2/Q2/.build"
SUFFIX = ".dylib" if sys.platform == "darwin" else ".so"
LIBRARY = BUILD_DIR / ("libv2_q2" + SUFFIX)
LINEAR_LIBRARY = BUILD_DIR / ("libq2_linear_audit" + SUFFIX)
V_REST_MV = -52.0
V_TH_MV = -45.0


def release_deviation_from_voltage(v, rest=V_REST_MV):
    x = np.clip((np.asarray(v) - np.asarray(rest)) / (V_TH_MV - np.asarray(rest)), -1.0, 1.0)
    return float(x) if np.ndim(v) == 0 else x


def _build_one(source: Path, library: Path) -> dict[str, object]:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    source_sha = sha256_file(source)
    meta = library.with_suffix(library.suffix + ".json")
    if library.exists() and meta.exists():
        old = json.loads(meta.read_text())
        if old.get("source_sha256") == source_sha and old.get("binary_sha256") == sha256_file(library):
            return old
    partial = library.with_suffix(library.suffix + ".partial")
    cmd = ["c++", "-O3", "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(partial)]
    subprocess.run(cmd, check=True, cwd=ROOT)
    partial.replace(library)
    record = {
        "source": str(source.relative_to(ROOT)), "source_sha256": source_sha,
        "binary": str(library.relative_to(ROOT)), "binary_sha256": sha256_file(library),
        "compiler": subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
        "flags": cmd[1:5],
    }
    meta.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def build() -> dict[str, object]:
    return _build_one(SOURCE, LIBRARY)


def build_linear_audit() -> dict[str, object]:
    return _build_one(LINEAR_SOURCE, LINEAR_LIBRARY)


class Q2HybridBrain(HybridBrain):
    """Q1 state/wrapper contract bound to the isolated Q2 binary."""

    def __init__(self, path=ROOT / "data/graph.npz", *, dt=0.1,
                 graded_mask=None, rest=None, compile=True):
        # The parent constructor's module-level build() names Q1.  Suppress
        # that build; dynamic dispatch below builds/loads only the Q2 binary.
        super().__init__(path, dt=dt, graded_mask=graded_mask, rest=rest,
                         compile=False)

    def _load_library(self):
        if self.build is None:
            self.build = build()
        self.lib = C.CDLL(str(LIBRARY))
        self.advance = self.lib.v2_q2_advance
        self.advance.argtypes = [C.c_int, C.POINTER(C.c_int64), C.POINTER(C.c_int32), C.POINTER(C.c_float),
            C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_int16),
            C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_int32), C.POINTER(C.c_int32),
            C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_int64), C.c_int, C.c_float,
            C.POINTER(C.c_uint8), C.POINTER(C.c_uint8), C.POINTER(C.c_float), C.POINTER(C.c_float),
            C.c_float, C.c_float, C.POINTER(C.c_int32), C.POINTER(C.c_int32), C.POINTER(C.c_uint8),
            C.POINTER(C.c_int32), C.POINTER(C.c_int64), C.POINTER(C.c_double), C.POINTER(C.c_int64),
            C.POINTER(C.c_float), C.POINTER(C.c_int64), C.POINTER(C.c_uint8), C.POINTER(C.c_double),
            C.POINTER(C.c_double), C.POINTER(C.c_double), C.POINTER(C.c_int32),
            C.POINTER(C.c_double), C.POINTER(C.c_int64), C.POINTER(C.c_int64)]
        self.advance.restype = None
        self.compat_advance = self.lib.v2_q2_compat_advance
        self.compat_advance.argtypes = [C.c_int, C.POINTER(C.c_int64), C.POINTER(C.c_int32), C.POINTER(C.c_float),
            C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_int16), C.POINTER(C.c_float),
            C.POINTER(C.c_float), C.POINTER(C.c_int32), C.POINTER(C.c_int32), C.POINTER(C.c_int64),
            C.c_int, C.c_float, C.POINTER(C.c_int32), C.POINTER(C.c_int32), C.POINTER(C.c_uint8),
            C.POINTER(C.c_int32), C.POINTER(C.c_int64), C.POINTER(C.c_uint8), C.POINTER(C.c_int8),
            C.POINTER(C.c_double), C.POINTER(C.c_int64), C.c_int, C.POINTER(C.c_int64), C.POINTER(C.c_int32),
            C.POINTER(C.c_float), C.POINTER(C.c_float), C.c_float, C.c_float, C.c_float, C.c_int,
            C.POINTER(C.c_float), C.POINTER(C.c_int64), C.POINTER(C.c_uint8), C.POINTER(C.c_float),
            C.POINTER(C.c_float), C.c_float, C.c_float] + [C.c_void_p] * 8
        self.compat_advance.restype = None


def arnoldi_hessenberg(ptr, post, weight, start, k=160):
    """Run the independent C++ double Arnoldi helper and return H/tail."""
    ptr = np.ascontiguousarray(ptr, dtype=np.int64)
    post = np.ascontiguousarray(post, dtype=np.int32)
    weight = np.ascontiguousarray(weight, dtype=np.float64)
    start = np.ascontiguousarray(start, dtype=np.float64)
    n = len(start)
    if ptr.shape != (n + 1,) or ptr[0] != 0 or ptr[-1] != len(post) or len(post) != len(weight):
        raise ValueError("invalid linear-audit CSR")
    info = build_linear_audit()
    lib = C.CDLL(str(LINEAR_LIBRARY)); fn = lib.q2_arnoldi
    fn.argtypes = [C.c_int, C.POINTER(C.c_int64), C.POINTER(C.c_int32), C.POINTER(C.c_double),
                   C.POINTER(C.c_double), C.c_int, C.POINTER(C.c_double), C.POINTER(C.c_double),
                   C.POINTER(C.c_double)]
    fn.restype = C.c_int
    h = np.zeros((k, k), dtype=np.float64); last = np.zeros(n, dtype=np.float64); tail = np.zeros(1, dtype=np.float64)
    actual = fn(n, ptr.ctypes.data_as(C.POINTER(C.c_int64)), post.ctypes.data_as(C.POINTER(C.c_int32)),
                weight.ctypes.data_as(C.POINTER(C.c_double)), start.ctypes.data_as(C.POINTER(C.c_double)), k,
                h.ctypes.data_as(C.POINTER(C.c_double)), last.ctypes.data_as(C.POINTER(C.c_double)),
                tail.ctypes.data_as(C.POINTER(C.c_double)))
    if actual <= 0:
        raise RuntimeError(f"Arnoldi helper failed: {actual}")
    return h[:actual, :actual], float(tail[0]), last, info
