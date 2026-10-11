from collections import Counter
from pathlib import Path

from experiments.a_share.run_purged_controls import SEEDS, permuted_actions, source_actions
from tools.audit_purged_controls import audit_controls


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/wc006_purged_controls/20260927T-frozen-protocol-v1"


def test_frozen_null_sequences_preserve_actions_and_change_dates():
    dates, source = source_actions()
    assert len(dates) == 609
    assert Counter(source) == {"EXPOSE": 234, "DEEXPOSE": 24, "DEFER": 351}
    sequences = [permuted_actions(source, seed) for seed in SEEDS]
    assert all(Counter(sequence) == Counter(source) and sequence != source for sequence in sequences)
    assert len({tuple(sequence) for sequence in sequences}) == len(SEEDS)


def test_independent_controls_have_intact_ledgers_and_exact_parity():
    audit = audit_controls(RUN)
    assert audit["status"] == "PASS_RESEARCH_ONLY"
    assert len(audit["arms"]) == 6
    assert next(item for item in audit["arms"] if item["arm"] == "base_with_retry")["retry_intents"] > 0
