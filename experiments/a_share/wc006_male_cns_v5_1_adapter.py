"""Compatibility/canonical V5.1 entry point.

The implementation lives in :mod:`wc006_male_cns_v51_adapter`; this module
provides the conventional v5_1 filename without touching the frozen V5
adapter or its run namespace.
"""
from experiments.a_share.wc006_male_cns_v51_adapter import *
from experiments.a_share.wc006_male_cns_v51_adapter import main
from experiments.a_share.wc006_male_cns_v51_adapter import (
    _create_batch, _ensure_v51_batches, _formal_v51_output_scope, _resolve_batches,
)

if __name__ == "__main__":
    raise SystemExit(main())
