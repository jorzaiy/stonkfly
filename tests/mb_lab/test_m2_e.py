from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stonkfly.mb_lab.dynamics import M1Parameters
from stonkfly.mb_lab.e1 import APLMapping, E1Result
from stonkfly.mb_lab.inventory import load_source, selected_nodes
from stonkfly.mb_lab.m2e import (
    M2EConnectivity,
    PlasticityState,
    SensoryCache,
    build_connectivity,
    normalized_valence,
    raw_readout,
    train_event,
)
from stonkfly.mb_lab.topology import BipartiteTopology, compile_real_topology
from stonkfly.neural.rule import PARAMETERS as RULE_PARAMETERS
from experiments.mb_lab.run_m2e_exploration import (
    analyze,
    build_bases,
    schedule,
    stimulus_content_hash,
)

ROOT = Path(__file__).resolve().parents[2]


def toy_connectivity() -> M2EConnectivity:
    pre = np.tile(np.arange(2, dtype=np.int32), 3)
    target = np.repeat(np.arange(3, dtype=np.int8), 2)
    channels = np.asarray([0, 0, 1], dtype=np.int8)
    edge_channel = channels[target]
    gain = np.zeros((2, len(pre)))
    gain[edge_channel, np.arange(len(pre))] = 1.0
    return M2EConnectivity(
        n_kc=2,
        pre_slot=pre,
        target_slot=target,
        target_node_index=np.asarray([10, 11, 12], dtype=np.int32),
        target_source_id=np.asarray([100, 101, 102], dtype=np.int64),
        target_channel=channels,
        baseline_weight=np.ones(len(pre)),
        edge_channel=edge_channel,
        gain=gain,
    )


def sensory(which: int) -> E1Result:
    trace = np.zeros((200, 2), dtype=np.float64)
    trace[50:100, which] = 0.6
    integral = trace.sum(axis=0) * 10.0
    return E1Result(
        integrated_rate=integral,
        population_trace=trace.mean(axis=1),
        group_trace=np.zeros((200, 4)),
        apl_trace=np.zeros((200, 1)),
        final_u=np.zeros(2),
        final_apl=np.zeros(1),
        finite=True,
        kc_trace=trace,
    )


def test_rule_sign_timing_and_unpaired_attenuation():
    connectivity = toy_connectivity()
    paired = PlasticityState.fresh(connectivity)
    train_event(paired, sensory(0), valence="reward", timing="paired", washout_seconds=0.0)
    reward_edges = (connectivity.edge_channel == 0) & (connectivity.pre_slot == 0)
    assert np.all(paired.memory_w[reward_edges] < 0)
    assert np.array_equal(paired.memory_w[connectivity.edge_channel == 1], np.zeros(2))

    unpaired = PlasticityState.fresh(connectivity)
    train_event(unpaired, sensory(0), valence="reward", timing="unpaired", washout_seconds=0.0)
    assert np.max(np.abs(unpaired.memory_w[reward_edges])) < np.max(np.abs(paired.memory_w[reward_edges]))

    reverse = PlasticityState.fresh(connectivity)
    zero = np.zeros(2)
    for _ in range(10):
        reverse.advance_bin(zero, np.asarray([1.0, 0.0]))
    for _ in range(50):
        reverse.advance_bin(np.asarray([0.6, 0.0]), np.zeros(2))
    assert np.all(reverse.memory_w[reward_edges] > 0)


def test_frozen_and_reward_omission_preserve_weights_and_probe_is_read_only():
    connectivity = toy_connectivity()
    item = sensory(0)
    for kwargs in ({"frozen": True}, {"omitted": True}):
        state = PlasticityState.fresh(connectivity)
        before = state.weight_digest()
        train_event(state, item, valence="reward", washout_seconds=0.0, **kwargs)
        assert state.weight_digest() == before
        snapshot = state.memory_w.copy()
        raw_readout(connectivity, item.integrated_rate / 500.0, state.efficacy)
        np.testing.assert_array_equal(state.memory_w, snapshot)


def test_paired_association_but_balanced_shuffled_reward_has_no_difference():
    connectivity = toy_connectivity()
    odors = [sensory(0), sensory(1)]
    baseline = [
        raw_readout(
            connectivity,
            odor.integrated_rate / 500.0,
            np.ones(len(connectivity.pre_slot)),
        )["channels"]
        for odor in odors
    ]

    paired = PlasticityState.fresh(connectivity)
    train_event(paired, odors[0], valence="reward", washout_seconds=0.0)
    train_event(paired, odors[1], valence="aversive", washout_seconds=0.0)
    values = [normalized_valence(raw_readout(connectivity, odor.integrated_rate / 500.0, paired.efficacy)["channels"], base) for odor, base in zip(odors, baseline)]
    assert values[0] - values[1] > 0
    learned_digest = paired.weight_digest()
    raw_readout(connectivity, odors[0].integrated_rate / 500.0, paired.efficacy)
    assert paired.weight_digest() == learned_digest  # reward-free probe is read-only
    before_decay = np.max(np.abs(paired.memory_w))
    paired.blank(1.0)
    assert 0 < np.max(np.abs(paired.memory_w)) < before_decay

    shuffled = PlasticityState.fresh(connectivity)
    for valence in ("reward", "aversive"):
        for odor in odors:
            train_event(shuffled, odor, valence=valence, washout_seconds=6.0)
    shuffled_values = [normalized_valence(raw_readout(connectivity, odor.integrated_rate / 500.0, shuffled.efficacy)["channels"], base) for odor, base in zip(odors, baseline)]
    assert abs(shuffled_values[0] - shuffled_values[1]) < abs(values[0] - values[1]) * 0.05


def test_real_channels_have_expected_drivable_kc_inputs():
    source = load_source()
    pn, kc = selected_nodes("R", olfactory_covered_only=True)
    topology = compile_real_topology(source["ptr"], source["post"], pn, kc)
    connectivity = build_connectivity(ROOT / "data/graph.npz", source, topology.kc_node_index)
    assert np.bincount(connectivity.edge_channel, minlength=2).tolist() == [1806, 1705]
    assert [len(np.unique(connectivity.pre_slot[connectivity.edge_channel == channel])) for channel in range(2)] == [931, 1705]
    assert connectivity.target_source_id.tolist() == [12859, 515338, 11402]


def test_cache_is_exact_and_learning_states_share_no_memory():
    topology = BipartiteTopology(
        2, 4, np.asarray([0, 0, 1, 1], dtype=np.int32),
        np.arange(4, dtype=np.int32), np.arange(2, dtype=np.int32),
        np.arange(4, dtype=np.int32),
    )
    groups = np.arange(4, dtype=np.int8)
    mapping = APLMapping(99, 10540, groups, ("g", "ab", "apbp", "u"),
                         np.ones(4), np.ones(4), np.ones(4))
    cache = SensoryCache(topology, M1Parameters(10, 50, 1, 1, .35, 25), mapping, dt_ms=.1)
    key1, first = cache.get(np.asarray([1.0, 0.5]))
    key2, second = cache.get(np.asarray([1.0, 0.5]))
    assert key1 == key2 and first is second and cache.builds == 1 and cache.hits == 1
    assert first.kc_trace is not None and not first.kc_trace.flags.writeable
    states = [PlasticityState.fresh(toy_connectivity()) for _ in range(5)]
    for left in states:
        for right in states:
            if left is not right:
                assert not np.shares_memory(left.memory_w, right.memory_w)


def test_config_matches_e1_and_runtime_rule_and_authorizes_one_matrix():
    config = json.loads((ROOT / "protocols/mb_lab/m2_e/config.json").read_text())
    e1 = json.loads((ROOT / "protocols/mb_lab/m1_e1/config.json").read_text())
    assert config["status"] == "AUTHORIZED_EXPLORATION_ONCE"
    for name in ("tau_k_ms", "tau_apl_ms", "input_scale", "apl_gain", "threshold"):
        assert config["sensory"][name] == e1["parameters"][name]
    assert config["plasticity"]["memory_decay_seconds"] == RULE_PARAMETERS["memory_decay_seconds"]
    assert config["plasticity"]["minimum_fraction"] == RULE_PARAMETERS["minimum_fraction"]
    assert config["full_exploration"]["authorized"] is True
    assert config["full_exploration"]["run_limit"] == 1


def test_exploration_stimuli_and_balanced_shuffled_schedules_are_deterministic():
    first = build_bases(8, 145, 29)
    second = build_bases(8, 145, 29)
    np.testing.assert_array_equal(first, second)
    assert np.all(np.count_nonzero(first, axis=-1) == 29)
    assert stimulus_content_hash(first) == stimulus_content_hash(second)
    for pair in range(8):
        for phase in ("acquisition", "reversal"):
            item = schedule(pair, phase)
            assert len(item["orders"]) == 10
            for odor_labels in item["shuffled"]:
                assert odor_labels.count("reward") == odor_labels.count("aversive") == 5


def test_exploration_statistics_recognize_only_complete_engineering_signature():
    chunks = []
    for replicate in range(8):
        d = np.zeros((5, 4), dtype=float)
        d[0] = [0.0, 0.10 + replicate * .001, -0.10 - replicate * .001, -0.09 - replicate * .0009]
        efficacy = np.ones((5, 4, 2), dtype=float)
        efficacy[0, 1:] = .99
        chunks.append({
            "D": d,
            "efficacy": efficacy,
            "finite": np.ones((5, 4), dtype=bool),
            "bound_fraction": np.zeros((5, 4)),
        })
    statistics, raw = analyze(chunks)
    assert statistics["verdict"] == "ENGINEERED_ASSOCIATIVE_UPDATE_OBSERVED"
    assert all(statistics["criteria"].values())
    assert raw["D"].shape == (5, 8, 4)
