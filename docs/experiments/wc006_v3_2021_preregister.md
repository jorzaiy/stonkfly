# WC006 / MaleCNS v3: 2021--2025 preregistered experiment

Status: preregistered specification only; paper/research execution is not authorized by this file.
Version: `2026-09-18.v1`

## Scope and non-negotiable isolation

This specification covers a paper-only experiment in an independent Stonkfly
run directory. It must not write the trading repository's live, shadow, watch,
config, or service state, and it must not place broker orders. Every output,
checkpoint, ledger, provenance record, and log belongs below a unique
`stonkfly-lab/runs/` directory. No external AmazingData query is part of this
specification; any missing official market-state field remains
non-authoritative and is a blocking limitation for an executable/financial
claim.

The 2026 six-symbol smoke bundle is not a historical candidate universe and
must not be reused as the 2021--2025 stream.

## Primary arm: clean MaleCNS v3

The primary arm uses the complete MaleCNS graph and the normal v3
`FlyController`/dual-compartment runtime with a clean, newly initialized brain.
It must not load any existing v2/v3 checkpoint. The causal schedule is:

| phase | dates | learning | use |
|---|---|---:|---|
| train | 2021-01-01 through 2023-12-31 | enabled | one prespecified learning run |
| validation | 2024-01-01 through 2024-12-31 | frozen | one read-only validation |
| exam | 2025-01-01 through 2025-12-31 | frozen | one final frozen exam |

The 2024 validation cannot change code, graph, decoder, costs, seed,
initialization, or hyperparameters. The 2025 exam is run once after the
validation record is sealed. No checkpoint or date may be selected after
looking at exam outcomes.

The v3 input contract is the single ETF RGB portfolio frame and v3 feedback
contract, including benchmark-relative return, drawdown/turnover terms,
veto penalty, and action-advantage feedback. The v3 close observation is
executed at the next valid session open. The v3 domestic-ETF cost assumption
has zero stamp duty; this is a semantic choice that must be stated in the
ledger rather than silently mapped to an equity stamp-duty rule.

## Repaired WC006 baseline

The baseline is the repaired WC006 implementation with the identical
candidate, price, timing, cost, lot, T+1, and ledger contract used by WC006.
It is a comparator, not a source of candidates for the v3 arm. It must use a
causally generated historical candidate/weight stream and hash-pinned
unadjusted canonical prices. It must preserve WC006's next-session fill,
`signal_lag_one_day`, target-subset behavior, end-of-data T+1 sentinel, and
paper-only/non-authoritative market-state labeling where official state is
unavailable.

The current 2026 six-candidate smoke inputs are explicitly excluded. No
baseline result is admissible until a 2021--2025 stream has per-date
provenance, exact source hashes, signal/execution dates, and a no-future-data
check.

## Secondary migration ablation (not the primary arm)

An existing v3 checkpoint may be evaluated only as a secondary migration
ablation beginning no earlier than 2023. The currently audited candidate is
`runs/a-share-510300-v3-train-2018-2022/brain-1.npz`, SHA-256
`5e7a3c52d894810ba4658b10684eed26b9f088faf50471408e967236364ce641`.
Its provenance says it learned online over 2018-01-01 through 2022-12-31.
Consequently it overlaps the 2021--2022 period and is forbidden in the 2021
primary arm: starting that checkpoint in 2021 is direct lookahead/data
leakage, not a clean transfer test.

The migration arm must report the ETF-to-WC006 input and decoder semantic
shift explicitly. It may not be used to claim that WC006 and v3 are the same
policy, and it may not replace the clean-brain primary arm.

## Compatibility audit recorded at preregistration

At the file/runtime level, v3 and WC006 both use the repository's
`FlyController`/`Settings` and MaleCNS graph, so the checkpoint container is
likely load-compatible when the exact package and graph are unchanged. This
is not a sufficient compatibility claim: WC006 hashes its own multi-symbol
manifest, graph/config signature, source files, initial state, costs, and
strict-market-state mode, while v3 has a separate single-ETF signature.

They are not semantically compatible without an explicit adapter audit:

* v3 presents one ETF portfolio frame with an 11-field status strip and uses
  shaped feedback with action advantage; WC006 presents a multi-symbol
  composite frame and uses its incremental/base-curve feedback.
* v3 emits direct BUY/HOLD/SELL decisions; WC006 applies a fixed decoder that
  maps side to EXPOSE/DEEXPOSE/DEFER and intentionally ignores decoder
  magnitude.
* v3's ETF stamp-duty and portfolio assumptions differ from WC006's
  multi-symbol equity ledger and official-state gates.

The MaleCNS graph provenance is fixed by the checked-in `data/manifest.json`
and `data/source.lock.json` (dataset `malecns_v1`, MaleCNS v1.0). A future
run must record those file hashes plus the exact runtime/package signature;
this document does not authorize silently substituting a graph or checkpoint.

## Data and causal gates

Before any training or return calculation, an isolated builder must prove all
of the following and fail closed otherwise:

1. A single exact canonical price tag covers every required symbol and all
   sessions needed for 2021--2025, with explicit unadjusted/adjusted meaning
   and per-file SHA-256. The local cache currently contains many
   `5y_2021-01-01_<2026 date>` files (including 1867-symbol sets), but no
   `5y_2021-01-01_2025-12-31` exact tag was found; slicing a later tag is not
   itself proof of a frozen historical bundle.
2. The active walk-forward/exporter path is copied or invoked into an
   independent output root and yields a complete dated candidate/weight stream
   from 2021 through 2025. It must use prior observations only for each signal
   and prove `execution_date` is the next valid session. Existing exporter
   outputs are point-in-time reports, not yet an accepted complete stream.
3. Universe membership, delistings, missing bars, corporate-action/adjustment
   policy, official ST/suspension/limit state, and calendar coverage are
   recorded per signal date. Missing official state makes the artifact
   non-authoritative and blocks any executable or performance conclusion.
4. Candidate and price hashes, model/config/graph signatures, initial state,
   costs, and checkpoints are immutable and stored outside the trading repo.
5. The exact spec hash is recorded before the first run; changing this file
   creates a new preregistration version and invalidates prior outputs.

These gates intentionally permit a lightweight coverage/compatibility audit,
but do not permit multi-year training, backtesting, return aggregation,
parameter search, or exam inspection in this preregistration step.

## Planned audit record and next command

The next permitted operation after the data gates pass is an isolated,
read-only coverage builder that emits a new manifest under
`stonkfly-lab/runs/wc006_v3_2021_inputs/<unique-id>/`, followed by a pure
checkpoint/config compatibility probe. It must not touch trading-repo
`data/watch`, `data/shadow`, active configs, or services. Only after the
manifest, provenance, exact hashes, causal-date checks, and this spec hash
are independently verified may the primary training schedule be considered.

No training, multi-year replay, or financial conclusion is authorized by this
document alone.
