"""Deterministic, optional sensory stimulation adapters.

The adapters in this module are deliberately independent of the MaleCNS
connectome.  They produce a small, serialisable description of extra current
injections; :class:`V51Controller` is responsible for passing those
injections to the existing ``MemoryBrain.step`` path.

No adapter in this module calls an external service.  In particular,
``FixedVectorSensoryAdapter`` is intended to be the stable boundary at which
a future Jev (or synthetic) feature producer can be connected.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


from .connectome import exact_ids

def _canonical_hash(payload: Any) -> str:
    """Hash JSON-compatible configuration using one canonical encoding."""

    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_value(value: Any) -> Any:
    """Convert numpy scalar/container values to JSON primitives."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class SensoryStimulation:
    """A fully serialisable description of one optional stimulation.

    ``neuron_indices`` are the integer indices consumed by ``MemoryBrain``;
    ``neuron_ids`` preserve the stable external identity used to construct the
    mapping.  There is one current per feature by default.  A feature mapped
    to multiple neurons applies the same current to each mapped index.
    """

    feature_names: list[str]
    values: list[float]
    missing_mask: list[bool]
    neuron_ids: list[list[Any]]
    neuron_indices: list[list[int]]
    currents: list[float] | list[list[float]]
    adapter_name: str
    adapter_version: str
    mapping_hash: str
    normalization_hash: str

    def __post_init__(self) -> None:
        names = list(self.feature_names)
        values = list(self.values)
        missing = list(self.missing_mask)
        ids = [list(group) for group in self.neuron_ids]
        indices = [list(group) for group in self.neuron_indices]
        currents = list(self.currents)
        if not (len(names) == len(values) == len(missing) == len(ids) == len(indices) == len(currents)):
            raise ValueError("SensoryStimulation fields must have equal feature lengths")
        if len(set(names)) != len(names) or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("feature_names must be unique, non-empty strings")
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                raise ValueError("sensory values must be numeric")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError("sensory values must be finite and in [0, 1]")
        if any(not isinstance(flag, (bool, np.bool_)) for flag in missing):
            raise ValueError("missing_mask must contain booleans")
        for feature_name, id_group, index_group in zip(names, ids, indices):
            if len(id_group) != len(index_group):
                raise ValueError(
                    f"neuron_ids and neuron_indices must have equal lengths for {feature_name}"
                )
            try:
                exact_ids(id_group)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "neuron_ids must be nonnegative integer source IDs"
                ) from exc
        for group in indices:
            if any(isinstance(index, bool) or not isinstance(index, (int, np.integer)) or int(index) < 0 for index in group):
                raise ValueError("neuron_indices must contain non-negative integers")
        for current in currents:
            if isinstance(current, (list, tuple, np.ndarray)):
                if any(not math.isfinite(float(item)) for item in current):
                    raise ValueError("currents must be finite")
            elif not math.isfinite(float(current)):
                raise ValueError("currents must be finite")
        for value in [self.adapter_name, self.adapter_version, self.mapping_hash, self.normalization_hash]:
            if not isinstance(value, str) or not value:
                raise ValueError("adapter metadata must be non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible, complete stimulation record."""

        return {
            "feature_names": _json_value(self.feature_names),
            "values": _json_value(self.values),
            "missing_mask": _json_value(self.missing_mask),
            "neuron_ids": _json_value(self.neuron_ids),
            "neuron_indices": _json_value(self.neuron_indices),
            "currents": _json_value(self.currents),
            "adapter_name": self.adapter_name,
            "adapter_version": self.adapter_version,
            "mapping_hash": self.mapping_hash,
            "normalization_hash": self.normalization_hash,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    def brain_pulses(self) -> list[tuple[np.ndarray, np.ndarray | float]]:
        """Return grouped ``(indices, current)`` pairs for ``MemoryBrain``."""

        pulses: list[tuple[np.ndarray, np.ndarray | float]] = []
        for group, current, missing in zip(self.neuron_indices, self.currents, self.missing_mask):
            if missing or not group:
                continue
            indices = np.asarray(group, dtype=np.int32)
            if isinstance(current, (list, tuple, np.ndarray)):
                curr = np.asarray(current, dtype=np.float32)
                if curr.shape != indices.shape:
                    raise ValueError("per-neuron current length must match neuron_indices")
                pulses.append((indices, curr))
            else:
                pulses.append((indices, float(current)))
        return pulses

    def validate_against_graph(self, ids: Sequence[Any]) -> None:
        """Fail closed unless every index names exactly its graph source ID.

        An in-range index alone is insufficient: graph ordering is part of
        the retained MaleCNS artifact, so a stale index must not silently
        retarget a stimulation channel after a graph change.
        """

        try:
            graph_ids = exact_ids(ids)
        except (TypeError, ValueError) as exc:
            raise ValueError("graph ids must be nonnegative integer source IDs") from exc
        if graph_ids.ndim != 1 or not len(graph_ids):
            raise ValueError("graph ids must be a non-empty one-dimensional sequence")
        if len(np.unique(graph_ids)) != len(graph_ids):
            raise ValueError("graph ids must be unique")
        for feature_name, id_group, index_group in zip(
            self.feature_names, self.neuron_ids, self.neuron_indices
        ):
            expected_ids = exact_ids(id_group)
            for expected_id, index in zip(expected_ids, index_group):
                if index >= len(graph_ids):
                    raise ValueError(
                        f"neuron index {index} for {feature_name} is outside graph bounds"
                    )
                if graph_ids[index] != expected_id:
                    raise ValueError(
                        f"neuron ID/index mismatch for {feature_name}: "
                        f"index {index} has graph ID {int(graph_ids[index])}, "
                        f"not configured ID {int(expected_id)}"
                    )


@runtime_checkable
class SensoryAdapter(Protocol):
    """Protocol for deterministic conversion of an observation to stimulation."""

    def encode(self, observation: dict) -> SensoryStimulation | None:
        """Encode one observation, or return ``None`` for no stimulation."""


class NullSensoryAdapter:
    """Compatibility adapter that produces no extra neural input."""

    adapter_name = "null"
    adapter_version = "1"
    mapping_hash = _canonical_hash({"adapter": adapter_name, "version": adapter_version})
    normalization_hash = _canonical_hash({"transfer": "none"})

    def encode(self, observation: dict) -> None:
        return None


class FixedVectorSensoryAdapter:
    """Encode a precomputed fixed-order vector into deterministic currents.

    ``feature_names`` and neuron mappings are configuration, not inferred from
    an observation.  The normal vector size is 8--32 features, leaving room
    for smaller synthetic vectors only through direct ``SensoryStimulation``
    construction in tests/tools.
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        neuron_indices: Sequence[Sequence[int]],
        *,
        neuron_ids: Sequence[Sequence[Any]],
        adapter_version: str = "1",
        current_scale: float = 1.0,
        current_offset: float = 0.0,
    ) -> None:
        self.feature_names = tuple(str(name) for name in feature_names)
        if not 8 <= len(self.feature_names) <= 32:
            raise ValueError("FixedVectorSensoryAdapter supports 8-32 features")
        if len(set(self.feature_names)) != len(self.feature_names) or any(not name for name in self.feature_names):
            raise ValueError("feature_names must be unique, non-empty strings")
        raw_indices = tuple(tuple(group) for group in neuron_indices)
        for group in raw_indices:
            if any(
                isinstance(index, (bool, np.bool_))
                or not isinstance(index, (int, np.integer))
                or int(index) < 0
                for index in group
            ):
                raise ValueError("neuron_indices must contain non-negative integers")
        self.neuron_indices = tuple(tuple(int(index) for index in group) for group in raw_indices)
        if len(self.neuron_indices) != len(self.feature_names):
            raise ValueError("one neuron_indices group is required per feature")
        if any(index < 0 for group in self.neuron_indices for index in group):
            raise ValueError("neuron_indices must be non-negative")
        self.neuron_ids = tuple(tuple(group) for group in neuron_ids)
        if len(self.neuron_ids) != len(self.feature_names):
            raise ValueError("one neuron_ids group is required per feature")
        if any(
            len(id_group) != len(index_group)
            for id_group, index_group in zip(self.neuron_ids, self.neuron_indices)
        ):
            raise ValueError("each neuron_ids group must match its neuron_indices group")
        try:
            for group in self.neuron_ids:
                exact_ids(group)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "neuron_ids must be nonnegative integer source IDs"
            ) from exc
        if not isinstance(adapter_version, str) or not adapter_version:
            raise ValueError("adapter_version must be non-empty")
        if not math.isfinite(float(current_scale)) or not math.isfinite(float(current_offset)):
            raise ValueError("current transfer parameters must be finite")
        self.adapter_name = "fixed_vector"
        self.adapter_version = adapter_version
        self.current_scale = float(current_scale)
        self.current_offset = float(current_offset)
        self.mapping_hash = _canonical_hash(
            {
                "feature_names": list(self.feature_names),
                "neuron_ids": _json_value(self.neuron_ids),
                "neuron_indices": [list(group) for group in self.neuron_indices],
            }
        )
        self.normalization_hash = _canonical_hash(
            {
                "transfer": "linear",
                "scale": self.current_scale,
                "offset": self.current_offset,
                "missing": "zero_current",
            }
        )

    def _extract(self, observation: dict) -> tuple[list[Any], list[bool]]:
        if isinstance(observation, Mapping):
            values: Any = observation.get("values", observation.get("sensory_values"))
            if values is None and isinstance(observation.get("features"), Mapping):
                feature_map = observation["features"]
                unknown = set(feature_map) - set(self.feature_names)
                missing_names = set(self.feature_names) - set(feature_map)
                if unknown or missing_names:
                    raise ValueError("feature mapping must contain exactly the configured names")
                values = [feature_map[name] for name in self.feature_names]
            if values is None:
                # A direct feature-name mapping is convenient for synthetic
                # producers and still follows the configured order.
                if all(name in observation for name in self.feature_names):
                    values = [observation[name] for name in self.feature_names]
                else:
                    raise ValueError("observation must contain values or features")
            missing = observation.get("missing_mask", observation.get("missing", [False] * len(self.feature_names)))
        else:
            values = observation
            missing = [False] * len(self.feature_names)
        values = list(values)
        missing = list(missing)
        if len(values) != len(self.feature_names) or len(missing) != len(self.feature_names):
            raise ValueError("sensory values and missing_mask must match feature_names")
        return values, missing

    def encode(self, observation: dict) -> SensoryStimulation:
        values, missing = self._extract(observation)
        clean_values: list[float] = []
        clean_missing: list[bool] = []
        currents: list[float] = []
        for value, flag in zip(values, missing):
            if isinstance(flag, (np.bool_, bool)):
                flag = bool(flag)
            else:
                raise ValueError("missing_mask must contain booleans")
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                raise ValueError("sensory values must be numeric")
            value = float(value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("sensory values must be finite and in [0, 1]")
            clean_values.append(value)
            clean_missing.append(flag)
            currents.append(0.0 if flag else self.current_offset + self.current_scale * value)
        return SensoryStimulation(
            feature_names=list(self.feature_names),
            values=clean_values,
            missing_mask=clean_missing,
            neuron_ids=[list(group) for group in self.neuron_ids],
            neuron_indices=[list(group) for group in self.neuron_indices],
            currents=currents,
            adapter_name=self.adapter_name,
            adapter_version=self.adapter_version,
            mapping_hash=self.mapping_hash,
            normalization_hash=self.normalization_hash,
        )

    def validate_against_graph(self, ids: Sequence[Any]) -> None:
        """Validate this fixed mapping before an experiment starts."""

        self.encode({"values": [0.0] * len(self.feature_names)}).validate_against_graph(ids)


__all__ = [
    "SensoryAdapter",
    "SensoryStimulation",
    "NullSensoryAdapter",
    "FixedVectorSensoryAdapter",
]
