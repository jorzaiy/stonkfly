import json

import numpy as np
import pytest

from stonkfly.neural.sensory_adapter import (
    FixedVectorSensoryAdapter,
    NullSensoryAdapter,
    SensoryStimulation,
)


def _adapter(**kwargs):
    kwargs.setdefault("neuron_ids", [[10 + i] for i in range(8)])
    return FixedVectorSensoryAdapter(
        [f"f{i}" for i in range(8)],
        [[10 + i] for i in range(8)],
        **kwargs,
    )


def test_null_adapter_is_a_no_op():
    assert NullSensoryAdapter().encode({"values": [0.5]}) is None


def test_fixed_vector_is_ordered_hashed_and_json_serialisable():
    adapter = _adapter()
    stimulation = adapter.encode(
        {"features": {f"f{i}": i / 8 for i in range(8)}, "missing_mask": [False] * 8}
    )
    assert stimulation.feature_names == [f"f{i}" for i in range(8)]
    assert stimulation.values == [i / 8 for i in range(8)]
    assert stimulation.mapping_hash == adapter.mapping_hash
    assert stimulation.normalization_hash == adapter.normalization_hash
    assert json.loads(stimulation.to_json()) == stimulation.to_dict()
    assert stimulation.brain_pulses()[0][0].dtype == np.int32


def test_missing_is_distinct_from_zero_and_does_not_pulse():
    stimulation = _adapter().encode(
        {"values": [0.0, 0.0] + [0.5] * 6, "missing_mask": [False, True] + [False] * 6}
    )
    assert stimulation.values[:2] == [0.0, 0.0]
    assert stimulation.missing_mask[:2] == [False, True]
    pulses = stimulation.brain_pulses()
    assert [int(item[0][0]) for item in pulses] == [10, 12, 13, 14, 15, 16, 17]
    assert float(pulses[0][1]) == 0.0


@pytest.mark.parametrize("values", [[-0.01] + [0.0] * 7, [1.01] + [0.0] * 7])
def test_values_are_strictly_bounded(values):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _adapter().encode(values)


def test_configuration_is_deterministic_and_feature_count_is_bounded():
    left = _adapter()
    right = _adapter()
    assert left.mapping_hash == right.mapping_hash
    assert left.normalization_hash == right.normalization_hash
    with pytest.raises(ValueError, match="8-32"):
        FixedVectorSensoryAdapter(["f"] * 7, [[]] * 7, neuron_ids=[[]] * 7)
    with pytest.raises(ValueError, match="8-32"):
        FixedVectorSensoryAdapter(
            [f"f{i}" for i in range(33)],
            [[] for _ in range(33)],
            neuron_ids=[[] for _ in range(33)],
        )


def test_stimulation_rejects_mismatched_fields():
    with pytest.raises(ValueError, match="equal feature lengths"):
        SensoryStimulation(
            feature_names=["f"],
            values=[0.0],
            missing_mask=[False],
            neuron_ids=[[]],
            neuron_indices=[[]],
            currents=[],
            adapter_name="x",
            adapter_version="1",
            mapping_hash="m",
            normalization_hash="n",
        )


def test_formal_mapping_rejects_stale_graph_and_non_integer_indices():
    adapter = _adapter()
    adapter.validate_against_graph(list(range(18)))
    with pytest.raises(ValueError, match="mismatch"):
        adapter.validate_against_graph(list(range(100, 118)))
    with pytest.raises(ValueError, match="non-negative integers"):
        FixedVectorSensoryAdapter(
            [f"f{i}" for i in range(8)],
            [[0.5] for _ in range(8)],
            neuron_ids=[[i] for i in range(8)],
        )
