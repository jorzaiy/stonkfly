"""Read-only MaleCNS PN/KC/APL inventory for M1 implementation preparation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / "data/graph.npz"
NEURONS = ROOT / "data/normalized/neurons.feather"
ANNOTATIONS = ROOT / "data/annotations.feather"


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_source() -> dict[str, np.ndarray]:
    """Load only the arrays needed for the M1 static audit."""
    with np.load(GRAPH, allow_pickle=False) as graph:
        ptr = np.asarray(graph["ptr"], dtype=np.int64)
        post = np.asarray(graph["post"], dtype=np.int32)
        ids = np.asarray(graph["ids"], dtype=np.int64)
    nodes = feather.read_table(NEURONS).to_pydict()
    annotations = feather.read_table(
        ANNOTATIONS, columns=["bodyId", "somaSide"]
    ).to_pydict()
    node_ids = np.asarray(nodes["source_id"], dtype=np.uint64).astype(np.int64)
    if not np.array_equal(ids, node_ids):
        raise ValueError("graph and normalized neuron IDs are not aligned")
    side_by_id = {
        int(body): "" if side is None else str(side)
        for body, side in zip(annotations["bodyId"], annotations["somaSide"])
    }
    return {
        "ptr": ptr,
        "post": post,
        "ids": ids,
        "cell_type": np.asarray(
            ["" if value is None else str(value) for value in nodes["cell_type"]]
        ),
        "superclass": np.asarray(
            ["" if value is None else str(value) for value in nodes["superclass"]]
        ),
        "neurotransmitter": np.asarray(
            [
                "" if value is None else str(value)
                for value in nodes["neurotransmitter"]
            ]
        ),
        "soma_side": np.asarray([side_by_id.get(int(body), "") for body in ids]),
    }


def candidate_masks(source: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Reproduce the structural O0 PN candidate rule without its fixed panel."""
    ptr = source["ptr"]
    post = source["post"]
    cell_type = source["cell_type"]
    superclass = source["superclass"]
    neurotransmitter = source["neurotransmitter"]
    kc = np.char.startswith(cell_type.astype(str), "KC")
    apl = cell_type == "APL"
    orn = np.char.startswith(cell_type.astype(str), "ORN_")
    orn_target = np.zeros(len(cell_type), dtype=bool)
    for node in np.flatnonzero(orn):
        orn_target[post[ptr[node] : ptr[node + 1]]] = True
    pn_label = np.asarray(["PN" in value for value in cell_type], dtype=bool)
    direct = orn_target & pn_label & (superclass == "cb_intrinsic")
    to_kc = np.zeros(len(cell_type), dtype=bool)
    for node in np.flatnonzero(direct):
        targets = post[ptr[node] : ptr[node + 1]]
        to_kc[node] = bool(np.any(kc[targets]))
    pn = direct & (neurotransmitter == "acetylcholine") & to_kc
    return {"PN": pn, "KC": kc, "APL": apl}


def hemisphere_inventory(side: str = "R") -> dict[str, object]:
    if side not in ("L", "R"):
        raise ValueError("side must be L or R")
    source = load_source()
    masks = candidate_masks(source)
    selected = {
        name: mask & (source["soma_side"] == side) for name, mask in masks.items()
    }
    union = np.logical_or.reduce(list(selected.values()))
    internal = 0
    boundary_out = 0
    categories: dict[str, int] = {}
    for source_name, source_mask in selected.items():
        for node in np.flatnonzero(source_mask):
            targets = source["post"][source["ptr"][node] : source["ptr"][node + 1]]
            internal += int(union[targets].sum())
            boundary_out += int((~union[targets]).sum())
            for target_name, target_mask in selected.items():
                key = f"{source_name}->{target_name}"
                categories[key] = categories.get(key, 0) + int(
                    target_mask[targets].sum()
                )
    boundary_in = 0
    for node in np.flatnonzero(~union):
        targets = source["post"][source["ptr"][node] : source["ptr"][node + 1]]
        boundary_in += int(union[targets].sum())
    sets = {
        name: np.flatnonzero(mask).astype(np.int32) for name, mask in selected.items()
    }
    covered_kc = np.zeros(len(source["ids"]), dtype=bool)
    for node in sets["PN"]:
        targets = source["post"][source["ptr"][node] : source["ptr"][node + 1]]
        covered_kc[targets[selected["KC"][targets]]] = True
    covered_indices = np.flatnonzero(covered_kc & selected["KC"]).astype(np.int32)
    return {
        "schema": "stonkfly-m1-static-inventory-v1",
        "side": side,
        "nodes": {
            name: {
                "count": len(indices),
                "node_index_sha256": array_sha256(indices),
                "source_id_sha256": array_sha256(source["ids"][indices]),
            }
            for name, indices in sets.items()
        },
        "olfactory_covered_kc": {
            "count": len(covered_indices),
            "uncovered_count": len(sets["KC"]) - len(covered_indices),
            "node_index_sha256": array_sha256(covered_indices),
            "source_id_sha256": array_sha256(source["ids"][covered_indices]),
            "selection": "same-side KC with at least one edge from a same-side eligible PN",
        },
        "union_nodes": int(union.sum()),
        "internal_edges": internal,
        "boundary_in_edges": boundary_in,
        "boundary_out_edges": boundary_out,
        "categories": {key: value for key, value in categories.items() if value},
    }


def selected_nodes(
    side: str = "R", *, olfactory_covered_only: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Return global PN and KC node indices for topology compilation."""
    source = load_source()
    masks = candidate_masks(source)
    side_mask = source["soma_side"] == side
    pn = np.flatnonzero(masks["PN"] & side_mask).astype(np.int32)
    kc_mask = masks["KC"] & side_mask
    if olfactory_covered_only:
        covered = np.zeros(len(kc_mask), dtype=bool)
        for node in pn:
            targets = source["post"][source["ptr"][node] : source["ptr"][node + 1]]
            covered[targets[kc_mask[targets]]] = True
        kc_mask &= covered
    return pn, np.flatnonzero(kc_mask).astype(np.int32)
