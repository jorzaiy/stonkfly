import json
import importlib.util
from pathlib import Path
import numpy as np
import pytest

from stonkfly.neural.trace_aggregate import TraceBuffers
_spec=importlib.util.spec_from_file_location('s2e',Path(__file__).parents[1]/'experiments/a_share/analyze_time_resolved_propagation.py'); _s2e=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_s2e)


def test_trace_defaults_off_and_json_safe():
    t = TraceBuffers(4, 3, np.full(4, -1, dtype=np.int32), np.zeros(4, dtype=np.uint8), np.full(4, -1, dtype=np.int32))
    assert t.native.overflow == 0
    json.dumps({"step": int(t.native.start_clock), "overflow": int(t.native.overflow)}, allow_nan=False)


def test_signed_increment_split_and_overflow_fail_closed():
    t = TraceBuffers(4, 3, np.full(4, -1, dtype=np.int32), np.zeros(4, dtype=np.uint8), np.full(4, -1, dtype=np.int32))
    t.cbkc_signed[2] = -3.5; t.cbkc_negative[2] = -3.5; t.cbkc_event_counts[2] = 1
    assert t.cbkc_signed[2] < 0
    t.native.overflow = 1
    with pytest.raises(RuntimeError, match="overflow"):
        t.fail_if_overflow()


def test_unique_kc_crossings_and_missing_are_explicit():
    assert _s2e.first_unique_crossings([3, 4, 8], [10, 10, 11], [1, 2, 3]) == {1: 3, 2: 8, 3: None}
