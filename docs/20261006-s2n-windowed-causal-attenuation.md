# S2n windowed causal-attenuation counterfactual (pre-registered)

S2n is a paper-only, diagnostic counterfactual on the complete retained
MaleCNS v1.0 graph. It uses the exact 60 S2d frames, a fresh observer-build
checkpoint bridged bitwise to the historical clean checkpoint, and
`learning=false`. It never changes the graph, removes edges, chooses trades,
uses returns/P&L, or enters production. `counterfactual=true` and
`production_eligible=false` are fixed before observing results.

The four fixed arms and execution order are:

1. `C`: scale 1 everywhere.
2. `V12`: only `visual_projection` source → KC accepted deliveries in
   KC1-relative W1 `[64,128)` and W2 `[128,256)` are scaled by 0.5.
3. `CB4`: only the frozen top-20 `other_cb` source body IDs below → KC
   accepted deliveries in W4 `[512,1024)` are scaled by 0.5.
4. `BOTH`: V12 and CB4 simultaneously.

The CB4 body-ID set is frozen from S2m v2 before S2n execution:

`11023, 11145, 11030, 12529, 16473, 515015, 513174, 521189, 18170,
11407, 519131, 523987, 125182, 19061, 534301, 522715, 521721, 12650,
20705, 11455`.

The S2m v2 formal JSON SHA256 is
`6408763c04100c6da79505ce26475d366dfc308628666699e629f9d2530ae4a0`,
and its raw NPZ SHA256 is
`a38838b4951c15a3bff6f011eba3ec91bef47738bd57da3b22460ad7c56e580`.
The S2m source mapping and frozen top-20 rule are inherited exactly; no
result-dependent source selection is permitted. APL's observed first input
delta is approximately `+68` after KC1 in the S2m descriptive report; this
does not authorize APL as an S2n intervention source.

The 0.5 scales and windows above are immutable. Intervention is applied only
at the real accepted kernel delivery point immediately before the transient
conductance increment. The persistent `weight` array is never modified.
All other delivery paths, targets, graph edges, schedules, spikes, and fixed
decoding remain unchanged. KC1 is frozen online at the first unique KC spike.
The independent S2n ABI defaults to null and fails closed on magic, version,
size, or overflow errors.

Each arm audits eligible and applied delivery counts, original and applied
signed/positive/negative weights, per-window scale, source membership,
KC-target membership, and conservation. `C` must have zero intervention
counts and match S2k-C frame hash, active-KC count, KC spike count, and spike
hash frame by frame. All arms must preserve persistent weight hash and finite
state. State summaries are active-KC count, KC spike count, active-KC bitset
and hash, low/intermediate/high state (`<=10`, `11..999`, `>=1000`), and
Jaccard to C.

The single frozen interpretation gate is: a non-C arm is said to stably alter
state categories only if at least 20% of the 60 frames cross a C state
category. If V12, CB4, and BOTH all fail this gate, S2 stops and recommends
redesigning the dynamics before S3. If only one arm passes, it is reported
only as that path's counterfactual sensitivity; it is not sufficient causal
evidence and makes no profitability claim.

No parameter, source set, scale, window, stop rule, or output selection may be
changed after viewing S2n results. Formal 60×4 execution remains separately
gated and is not launched by the smoke command.

## ABI and artifact layout clarification (v2)

The observer ABI carries one `int8 source_group` per graph body: `0` none,
`1` visual projection, and `2` frozen top-20 `other_cb`. The kernel computes
eligibility from `(source_group, KC1-relative window)`: group 1 is eligible
only in W1/W2 and group 2 only in W4. `BOTH` is the disjoint union, so one
delivery can be attenuated at most once. Audit arrays are shaped `2×5`
(group × window), including counts and original/applied signed/positive/
negative totals. The source mapping is validated once and stored once at the
JSON top level; per-frame records retain only small audit matrices and scalar
summaries. Full audit vectors and KC bitsets are in NPZ, with bitset length
derived from the recorded KC target count.
