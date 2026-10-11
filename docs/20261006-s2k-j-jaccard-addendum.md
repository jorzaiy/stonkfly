# S2k-J active-set Jaccard addendum (pre-registered)

This addendum supplements the already authorized six-arm S2k counterfactual run. It does not change arms, inputs, graph, checkpoint, historical kernel, overlays, or `learning=False`.

Primary metric: within each arm, all unordered pairs among the same 60 event frames, exactly C(60,2)=1770 Jaccard values. Each set contains the active KC IDs among the 4064 KC nodes, where spike count >0. For sets A and B, J(A,B)=|A∩B|/|A∪B|; two empty sets are defined as 1, and exactly one empty set as 0.

Reports must include n, min, q25, median, mean, q75, max. Quantiles use NumPy's fixed `linear` method. Category-stratified values (same category and cross category) are descriptive only and use the original fixed labels; they cannot select parameters or arms. A secondary comparison computes, for each intervention arm, the same-event Jaccard against C for 60 pairs.

Every arm/event is restored from the same clean checkpoint and run once with the exact historical isolated kernel. Lossless active bitsets use `numpy.packbits(..., bitorder='little')`, length 4064, with KC ordering/body-ID hash recorded. Before any summary is emitted, every rerun record must match the original S2k record's frame hash, active KC count, KC spike count, full spike hash, and KC-set hash; bitset popcount and unpacked-set hash must also match. Any mismatch is fail-closed.

No reward, labels for selection, decoder, trade, P&L, or performance gate is used. This is descriptive counterfactual audit output only; it is not production eligible and cannot justify graph or weight changes.
