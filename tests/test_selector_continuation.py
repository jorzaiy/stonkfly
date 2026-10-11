from experiments.a_share.run_isolated_selector_slice import _validate_checkpoint, sha256
from tools.extend_purged_selector_2024_2025 import SOURCE, SOURCE_SHA256, TAG, scheduled_dates


def test_frozen_selector_source_and_new_schedule_are_exact():
    old, new = scheduled_dates()
    assert len(old) == 46 and len(new) == 36
    assert old[-1] == "2023-12-29"
    assert new[0] == "2024-01-02" and new[-1] == "2025-12-29"
    assert sha256(SOURCE / "checkpoint.json") == SOURCE_SHA256
    checkpoint, _, _, overlays = _validate_checkpoint(
        SOURCE, old, TAG, allow_original_source_drift=True)
    assert checkpoint["completed_dates"] == old
    assert overlays["purge"]["patch_id"] == "purged-forward-labels-v1"
