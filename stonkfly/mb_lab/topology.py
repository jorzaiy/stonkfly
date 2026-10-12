"""B-real and degree-preserving C topology construction for M1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class BipartiteTopology:
    n_pn: int
    n_kc: int
    source: np.ndarray
    target: np.ndarray
    pn_node_index: np.ndarray
    kc_node_index: np.ndarray

    def __post_init__(self) -> None:
        source = np.ascontiguousarray(self.source, dtype=np.int32)
        target = np.ascontiguousarray(self.target, dtype=np.int32)
        pn_nodes = np.ascontiguousarray(self.pn_node_index, dtype=np.int32)
        kc_nodes = np.ascontiguousarray(self.kc_node_index, dtype=np.int32)
        if source.shape != target.shape or source.ndim != 1 or not len(source):
            raise ValueError("topology needs nonempty aligned edge arrays")
        if pn_nodes.shape != (self.n_pn,) or kc_nodes.shape != (self.n_kc,):
            raise ValueError("node arrays do not match declared sizes")
        if (
            np.any(source < 0)
            or np.any(source >= self.n_pn)
            or np.any(target < 0)
            or np.any(target >= self.n_kc)
        ):
            raise ValueError("edge endpoint outside bipartite topology")
        codes = source.astype(np.int64) * self.n_kc + target
        if len(np.unique(codes)) != len(codes):
            raise ValueError("parallel PN-to-KC edges are not allowed")
        for array in (source, target, pn_nodes, kc_nodes):
            array.setflags(write=False)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "pn_node_index", pn_nodes)
        object.__setattr__(self, "kc_node_index", kc_nodes)

    @property
    def edge_count(self) -> int:
        return len(self.source)

    @property
    def pn_degree(self) -> np.ndarray:
        return np.bincount(self.source, minlength=self.n_pn)

    @property
    def kc_degree(self) -> np.ndarray:
        return np.bincount(self.target, minlength=self.n_kc)

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        for name, array in (
            ("source", self.source),
            ("target", self.target),
            ("pn_node_index", self.pn_node_index),
            ("kc_node_index", self.kc_node_index),
        ):
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

    def project(self, pn_rate: np.ndarray) -> np.ndarray:
        values = np.asarray(pn_rate, dtype=np.float64)
        if values.shape != (self.n_pn,) or not np.isfinite(values).all():
            raise ValueError("pn_rate has wrong shape or nonfinite values")
        degree = self.kc_degree
        if np.any(degree == 0):
            raise ValueError("each KC must have at least one PN input")
        drive = np.bincount(
            self.target,
            weights=values[self.source] / degree[self.target],
            minlength=self.n_kc,
        )
        return drive.astype(np.float64, copy=False)

    def rewire(self, seed: int, swaps: int | None = None) -> "BipartiteTopology":
        """Return a simple bipartite graph with both degree sequences fixed."""
        if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
            raise ValueError("seed must be a nonnegative integer")
        swaps = 5 * self.edge_count if swaps is None else int(swaps)
        if swaps < 1:
            raise ValueError("at least one swap is required")
        rng = np.random.default_rng(int(seed))
        source = self.source.copy()
        target = self.target.copy()
        codes = set((source.astype(np.int64) * self.n_kc + target).tolist())
        accepted = 0
        attempts = 0
        maximum_attempts = max(1000, swaps * 100)
        while accepted < swaps and attempts < maximum_attempts:
            attempts += 1
            first, second = rng.integers(0, self.edge_count, size=2)
            if first == second:
                continue
            s1, s2 = int(source[first]), int(source[second])
            t1, t2 = int(target[first]), int(target[second])
            if s1 == s2 or t1 == t2:
                continue
            old1, old2 = s1 * self.n_kc + t1, s2 * self.n_kc + t2
            new1, new2 = s1 * self.n_kc + t2, s2 * self.n_kc + t1
            if new1 in codes or new2 in codes:
                continue
            codes.remove(old1)
            codes.remove(old2)
            codes.add(new1)
            codes.add(new2)
            target[first], target[second] = t2, t1
            accepted += 1
        if accepted != swaps:
            raise RuntimeError(f"only completed {accepted}/{swaps} degree-preserving swaps")
        result = BipartiteTopology(
            self.n_pn,
            self.n_kc,
            source,
            target,
            self.pn_node_index,
            self.kc_node_index,
        )
        if not np.array_equal(result.pn_degree, self.pn_degree) or not np.array_equal(
            result.kc_degree, self.kc_degree
        ):
            raise AssertionError("rewire changed a degree sequence")
        if result.sha256 == self.sha256:
            raise RuntimeError("rewire did not change topology")
        return result


def compile_real_topology(
    ptr: np.ndarray,
    post: np.ndarray,
    pn_node_index: np.ndarray,
    kc_node_index: np.ndarray,
) -> BipartiteTopology:
    ptr = np.asarray(ptr, dtype=np.int64)
    post = np.asarray(post, dtype=np.int32)
    pn_nodes = np.asarray(pn_node_index, dtype=np.int32)
    kc_nodes = np.asarray(kc_node_index, dtype=np.int32)
    kc_slot = np.full(len(ptr) - 1, -1, dtype=np.int32)
    kc_slot[kc_nodes] = np.arange(len(kc_nodes), dtype=np.int32)
    sources: list[int] = []
    targets: list[int] = []
    for slot, node in enumerate(pn_nodes):
        candidate = kc_slot[post[ptr[node] : ptr[node + 1]]]
        candidate = candidate[candidate >= 0]
        sources.extend([slot] * len(candidate))
        targets.extend(candidate.tolist())
    return BipartiteTopology(
        len(pn_nodes),
        len(kc_nodes),
        np.asarray(sources, dtype=np.int32),
        np.asarray(targets, dtype=np.int32),
        pn_nodes,
        kc_nodes,
    )
