"""Pure canonical packet construction for dynamic-landmark SSL.

This module has no filesystem or training side effects.  It accepts already
authorized trajectory arrays, enforces their exact feature/timeline schemas,
and returns bundle-local 30-Hz packets plus a separate private provenance map.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..datasets.dynamic_landmark import (
    DYNAMIC_FEATURE_NAMES,
    DYNAMIC_FEATURE_SCHEMA,
)
from ..preprocessing.semantic_landmarks import (
    SEMANTIC23_FEATURE_NAMES,
    SEMANTIC23_SCHEMA,
    clinical23_v2_to_semantic23,
)


@dataclass(frozen=True)
class BridgePolicy:
    """The preregistered, length-only packet policy."""

    sample_rate_hz: float = 30.0
    window_length: int = 32
    ravdess_packets_per_trial: int = 1
    mayo_packets_per_recording: int = 16
    selection: str = "uniform_floor_v1"

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_rate_hz, (bool, np.bool_))
            or not isinstance(self.sample_rate_hz, (int, float, np.integer, np.floating))
            or not math.isfinite(float(self.sample_rate_hz))
            or float(self.sample_rate_hz) != 30.0
        ):
            raise ValueError("bridge sample rate must be exactly 30 Hz")
        frozen_integers = {
            "window_length": (self.window_length, 32),
            "ravdess_packets_per_trial": (self.ravdess_packets_per_trial, 1),
            "mayo_packets_per_recording": (self.mayo_packets_per_recording, 16),
        }
        for name, (observed, expected) in frozen_integers.items():
            if (
                isinstance(observed, (bool, np.bool_))
                or not isinstance(observed, (int, np.integer))
                or int(observed) != expected
            ):
                raise ValueError(f"bridge {name} must be exactly {expected}")
        if self.selection != "uniform_floor_v1":
            raise ValueError("bridge selection must be uniform_floor_v1")


@dataclass(frozen=True)
class CanonicalPacketBundle:
    """The four public training arrays emitted for one source trajectory."""

    features: np.ndarray
    valid_mask: np.ndarray
    timestamps: np.ndarray
    source_frame_indices: np.ndarray


@dataclass(frozen=True)
class PrivateTrajectoryMapping:
    """Original trajectory coordinates kept outside the public bundle."""

    window_starts: np.ndarray
    original_canonical_frame_indices: np.ndarray
    original_source_frame_indices: np.ndarray
    original_timestamps: np.ndarray


def _exact_integer(value: object, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def uniform_floor_starts(
    length: int,
    *,
    count: int,
    window: int = 32,
) -> np.ndarray:
    """Return ``floor(i * (T-window) / (count-1))`` using integer arithmetic."""
    length = _exact_integer(length, "length")
    count = _exact_integer(count, "count")
    window = _exact_integer(window, "window")
    if window <= 0:
        raise ValueError("window must be positive")
    if count < 2:
        raise ValueError("uniform floor selection requires at least two starts")
    if length < window:
        raise ValueError(
            f"trajectory length {length} is shorter than window {window}"
        )
    final_start = length - window
    starts = np.fromiter(
        (index * final_start // (count - 1) for index in range(count)),
        dtype=np.int64,
        count=count,
    )
    if (
        starts.shape != (count,)
        or starts[0] != 0
        or starts[-1] != final_start
        or np.any(np.diff(starts) < 0)
    ):
        raise RuntimeError("uniform floor selection violated its frozen contract")
    return starts


def _exact_feature_names(
    feature_names: Sequence[str],
    expected: tuple[str, ...],
) -> None:
    if isinstance(feature_names, (str, bytes)):
        raise ValueError("feature names must be an ordered sequence")
    try:
        observed = tuple(feature_names)
    except TypeError as exc:
        raise ValueError("feature names must be an ordered sequence") from exc
    if observed != expected or not all(type(name) is str for name in observed):
        raise ValueError("feature names do not match the registered exact order")


def _validate_trajectory(
    *,
    features: np.ndarray,
    valid_mask: np.ndarray,
    canonical_frame_indices: np.ndarray,
    original_source_frame_indices: np.ndarray,
    original_timestamps: np.ndarray,
    feature_schema: str,
    feature_names: Sequence[str],
    expected_width: int,
    expected_schema: str,
    expected_names: tuple[str, ...],
    policy: BridgePolicy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(policy, BridgePolicy):
        raise ValueError("packet construction requires BridgePolicy")
    # Revalidate even a forged/deserialized object rather than trusting its type.
    policy.__post_init__()
    if type(feature_schema) is not str or feature_schema != expected_schema:
        raise ValueError(f"feature schema must be exactly {expected_schema!r}")
    _exact_feature_names(feature_names, expected_names)

    features = np.asarray(features)
    valid_mask = np.asarray(valid_mask)
    canonical_frame_indices = np.asarray(canonical_frame_indices)
    original_source_frame_indices = np.asarray(original_source_frame_indices)
    original_timestamps = np.asarray(original_timestamps)
    length = features.shape[0] if features.ndim == 2 else -1
    if features.dtype != np.float32 or features.shape != (length, expected_width):
        raise ValueError(
            f"features must have exact shape (T,{expected_width}) and dtype float32"
        )
    if length < policy.window_length:
        raise ValueError("trajectory is shorter than the canonical window")
    if valid_mask.dtype != np.bool_ or valid_mask.shape != (length,):
        raise ValueError("valid mask must have exact shape (T,) and dtype bool")
    if (
        canonical_frame_indices.dtype != np.int64
        or canonical_frame_indices.shape != (length,)
    ):
        raise ValueError("canonical frame indices must have shape (T,) and dtype int64")
    if (
        original_source_frame_indices.dtype != np.int64
        or original_source_frame_indices.shape != (length,)
    ):
        raise ValueError("original source frame indices must have shape (T,) and dtype int64")
    if original_timestamps.dtype != np.float64 or original_timestamps.shape != (length,):
        raise ValueError("original timestamps must have shape (T,) and dtype float64")
    if not np.isfinite(features).all() or not np.isfinite(original_timestamps).all():
        raise ValueError("trajectory arrays must contain only finite values")
    if np.any(canonical_frame_indices < 0) or not np.all(
        np.diff(canonical_frame_indices) == 1
    ):
        raise ValueError(
            "canonical frame indices must retain every fixed-grid 30-Hz position"
        )
    if np.any(original_source_frame_indices < 0) or not np.all(
        np.diff(original_source_frame_indices) > 0
    ):
        raise ValueError("original source frame indices must increase strictly")
    if not np.all(np.diff(original_timestamps) > 0):
        raise ValueError("original timestamps must increase strictly")
    if not bool(valid_mask.any()):
        raise ValueError("trajectory must contain at least one observed frame")

    canonical_features = features.copy()
    canonical_features[~valid_mask] = np.float32(0.0)
    return (
        canonical_features,
        valid_mask.copy(),
        canonical_frame_indices.copy(),
        original_source_frame_indices.copy(),
        original_timestamps.copy(),
    )


def _construct_packets(
    features: np.ndarray,
    valid_mask: np.ndarray,
    canonical_frame_indices: np.ndarray,
    original_source_frame_indices: np.ndarray,
    original_timestamps: np.ndarray,
    *,
    packet_starts: np.ndarray,
    policy: BridgePolicy,
) -> tuple[CanonicalPacketBundle, PrivateTrajectoryMapping]:
    packet_starts = np.asarray(packet_starts)
    if packet_starts.dtype != np.int64 or packet_starts.ndim != 2:
        raise RuntimeError("internal packet starts violated the exact int64 matrix contract")
    local_indices = np.arange(policy.window_length, dtype=np.int64)
    row_indices = packet_starts[..., None] + local_indices
    if np.any(row_indices < 0) or np.any(row_indices >= len(features)):
        raise RuntimeError("internal packet selection exceeded the trajectory")
    packet_features = features[row_indices].astype(np.float32, copy=True)
    packet_valid = valid_mask[row_indices].astype(np.bool_, copy=True)
    packet_features[~packet_valid] = np.float32(0.0)
    local_timestamps = (
        np.arange(policy.window_length, dtype=np.float32)
        / np.float32(policy.sample_rate_hz)
    )
    leading_shape = packet_starts.shape + (policy.window_length,)
    bundle = CanonicalPacketBundle(
        features=packet_features,
        valid_mask=packet_valid,
        timestamps=np.broadcast_to(local_timestamps, leading_shape).copy(),
        source_frame_indices=np.broadcast_to(local_indices, leading_shape).copy(),
    )
    mapping = PrivateTrajectoryMapping(
        window_starts=packet_starts.copy(),
        original_canonical_frame_indices=canonical_frame_indices[row_indices].copy(),
        original_source_frame_indices=original_source_frame_indices[row_indices].copy(),
        original_timestamps=original_timestamps[row_indices].copy(),
    )
    return bundle, mapping


def packetize_ravdess_trajectory(
    features: np.ndarray,
    valid_mask: np.ndarray,
    *,
    canonical_frame_indices: np.ndarray,
    original_source_frame_indices: np.ndarray,
    original_timestamps: np.ndarray,
    feature_schema: str,
    feature_names: Sequence[str],
    policy: BridgePolicy = BridgePolicy(),
) -> tuple[CanonicalPacketBundle, PrivateTrajectoryMapping]:
    """Build one four-window semantic23 packet from one RAVDESS trial."""
    checked = _validate_trajectory(
        features=features,
        valid_mask=valid_mask,
        canonical_frame_indices=canonical_frame_indices,
        original_source_frame_indices=original_source_frame_indices,
        original_timestamps=original_timestamps,
        feature_schema=feature_schema,
        feature_names=feature_names,
        expected_width=23,
        expected_schema=SEMANTIC23_SCHEMA,
        expected_names=SEMANTIC23_FEATURE_NAMES,
        policy=policy,
    )
    starts = uniform_floor_starts(
        len(checked[0]), count=4, window=policy.window_length,
    ).reshape(1, 4)
    return _construct_packets(*checked, packet_starts=starts, policy=policy)


def packetize_mayo_trajectory(
    features: np.ndarray,
    valid_mask: np.ndarray,
    *,
    canonical_frame_indices: np.ndarray,
    original_source_frame_indices: np.ndarray,
    original_timestamps: np.ndarray,
    feature_schema: str,
    feature_names: Sequence[str],
    policy: BridgePolicy = BridgePolicy(),
) -> tuple[CanonicalPacketBundle, PrivateTrajectoryMapping]:
    """Build 16 quartile-interleaved packets from one MediaPipe trajectory."""
    checked = _validate_trajectory(
        features=features,
        valid_mask=valid_mask,
        canonical_frame_indices=canonical_frame_indices,
        original_source_frame_indices=original_source_frame_indices,
        original_timestamps=original_timestamps,
        feature_schema=feature_schema,
        feature_names=feature_names,
        expected_width=95,
        expected_schema=DYNAMIC_FEATURE_SCHEMA,
        expected_names=DYNAMIC_FEATURE_NAMES,
        policy=policy,
    )
    semantic23 = clinical23_v2_to_semantic23(checked[0][..., 72:])
    if (
        semantic23.dtype != np.float32
        or semantic23.shape != (len(checked[0]), 23)
        or not np.isfinite(semantic23).all()
    ):
        raise ValueError("clinical23_v2 adapter violated the semantic23 contract")
    starts = uniform_floor_starts(
        len(checked[0]),
        count=policy.mayo_packets_per_recording * 4,
        window=policy.window_length,
    )
    packet_starts = np.stack(
        tuple(
            starts[offset:offset + policy.mayo_packets_per_recording]
            for offset in range(
                0,
                policy.mayo_packets_per_recording * 4,
                policy.mayo_packets_per_recording,
            )
        ),
        axis=1,
    )
    return _construct_packets(
        *checked, packet_starts=packet_starts, policy=policy,
    )


__all__ = [
    "BridgePolicy",
    "CanonicalPacketBundle",
    "PrivateTrajectoryMapping",
    "packetize_mayo_trajectory",
    "packetize_ravdess_trajectory",
    "uniform_floor_starts",
]
