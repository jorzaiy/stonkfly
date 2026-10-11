"""S2n ABI and frozen boundary/membership helper tests.

The exact boundary assertions below are a small Python mirror of the kernel's
half-open window predicate; real C++ delivery behavior is exercised by the
S2n smoke runner, which attaches the ABI to the retained graph.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.a_share.smoke_s2n import SCALES, TOP20, manifest_digest, selected_sources, window_index
import experiments.a_share.smoke_s2n as smoke_s2n
from stonkfly.neural.trace_s2n import TraceBuffersS2N, WINDOW_BOUNDS


def test_s2n_window_boundaries_helper_mirror():
    expected = {63: 0, 64: 1, 127: 1, 128: 2, 255: 2, 256: 3,
                511: 3, 512: 4, 1023: 4, 1024: -1, -1: -1}
    assert all(window_index(k) == v for k, v in expected.items())
    assert WINDOW_BOUNDS == ((0, 64), (64, 128), (128, 256), (256, 512), (512, 1024))


def test_s2n_source_membership_kc_only_and_union():
    ids = np.asarray([11023, 999999, 11030, 42], dtype=np.int64)
    classes = np.asarray([4, 3, 4, 5], dtype=np.int8)
    v = selected_sources(classes, ids, "V12")
    cb = selected_sources(classes, ids, "CB4")
    both = selected_sources(classes, ids, "BOTH")
    assert v.tolist() == [0, 1, 0, 0]
    assert cb.tolist() == [1, 0, 1, 0]
    assert both.tolist() == [1, 1, 1, 0]
    assert np.array_equal(both, np.maximum(v, cb))
    assert len(TOP20) == len(set(TOP20)) == 20


def test_s2n_trace_abi_native_buffers_and_scale_half():
    selected = np.asarray([1, 0, 1], dtype=np.uint8)
    slots = np.asarray([-1, 0, 1], dtype=np.int32)
    t = TraceBuffersS2N(selected, slots, SCALES["BOTH"], source_body_ids=np.array([10, 11, 12]))
    assert t.native.magic == t.MAGIC
    assert t.native.abi_version == t.ABI_VERSION
    assert t.native.size > 0
    assert np.allclose(t.scale, [1., .5, .5, 1., .5])
    assert t.source_mapping() == [{"source_index": 0, "body_id": 10}, {"source_index": 2, "body_id": 12}]
    t.eligible_count[0] = 2; t.eligible_events.flat[[1, 4]] = 1
    t.applied_count[0] = 2; t.applied_events.flat[[1, 4]] = 1
    t.fail()


def test_s2n_weight_is_not_observer_storage():
    selected = np.asarray([1, 0], dtype=np.uint8)
    t = TraceBuffersS2N(selected, np.asarray([-1, 0], dtype=np.int32), SCALES["CB4"])
    before = t.scale.copy()
    t.reset()
    assert np.array_equal(before, t.scale)
    assert np.all(t.applied_count == 0)


def test_s2n_manifest_digest_excludes_mutable_file_pins():
    m = {"raw_npz_sha256": "raw-a", "json_manifest_sha256": "json-a", "gates": {"all": True}, "records": []}
    digest = manifest_digest(m)
    m["raw_npz_sha256"] = "raw-b"; m["json_manifest_sha256"] = "json-b"
    assert manifest_digest(m) == digest


def test_formal_failure_cleans_raw_candidate(tmp_path, monkeypatch):
    (tmp_path / "s2n_formal_raw.npz").write_bytes(b"candidate")
    monkeypatch.setattr(smoke_s2n, "_events", lambda: [])
    with pytest.raises(RuntimeError, match="exact 60"):
        smoke_s2n.run_formal(tmp_path)
    assert not (tmp_path / "s2n_formal_raw.npz").exists()
    assert not (tmp_path / "s2n_formal.json").exists()
    assert (tmp_path / "s2n_formal_failure.json").exists()
