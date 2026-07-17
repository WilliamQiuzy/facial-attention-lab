"""Canonical packet construction and private publication for landmark SSL.

Pure packet helpers accept already-authorized trajectories and return local
30-Hz packets.  The lower half of this module publishes and verifies the
owner-only, provenance-bound bridge generations used by the trainer.
"""
from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import hmac
import io
import json
import math
import os
import re
import stat
import sys
import uuid
from contextlib import ExitStack
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from types import CodeType
from typing import Callable, Mapping, Sequence

import numpy as np

from ..datasets.dynamic_landmark import (
    DYNAMIC_FEATURE_NAMES,
    DYNAMIC_FEATURE_SCHEMA,
)
from ..preprocessing.semantic_landmarks import (
    CLINICAL23_V2_ADAPTER_METADATA,
    CLINICAL23_V2_TO_SEMANTIC23_INDEX,
    SEMANTIC23_FEATURE_NAMES,
    SEMANTIC23_SCHEMA,
    clinical23_v2_to_semantic23,
)
from ..preprocessing.openface68_semantic import (
    OPENFACE68_ADAPTER_METADATA,
    openface68_to_semantic23,
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


@dataclass(frozen=True)
class _PreparedBridgeStage:
    name: str
    bundle_bytes: bytes
    record: dict[str, object]
    sample_ids: tuple[str, ...]
    source_unit_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    cache_integrity_ids: tuple[str, ...]
    window_starts: tuple[tuple[int, ...], ...]
    original_canonical_frame_indices: tuple[
        tuple[tuple[int, ...], ...], ...
    ]
    original_source_frame_indices: tuple[
        tuple[tuple[int, ...], ...], ...
    ]
    original_timestamps: tuple[tuple[tuple[float, ...], ...], ...]
    canonical_key_identity_sha256: str
    private_key: bytes = dataclass_field(repr=False)


@dataclass(frozen=True)
class _PreparedBridgeGeneration:
    ravdess: _PreparedBridgeStage
    mayo: _PreparedBridgeStage
    generation: dict[str, object]
    generation_bytes: bytes


@dataclass(frozen=True)
class _PreparedFrozenStage:
    name: str
    artifacts: Mapping[str, bytes]
    receipt: Mapping[str, object]
    receipt_bytes: bytes


@dataclass(frozen=True)
class _PreparedFrozenInputs:
    mode: str
    ravdess: _PreparedFrozenStage
    mayo: _PreparedFrozenStage


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
    if type(policy) is not BridgePolicy:
        raise ValueError("packet construction requires BridgePolicy")
    # Revalidate even a forged/deserialized object rather than trusting its type.
    BridgePolicy.__post_init__(policy)
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
    expected_semantic23 = np.take(
        checked[0][..., 72:],
        CLINICAL23_V2_TO_SEMANTIC23_INDEX,
        axis=-1,
    ).astype(np.float32, copy=True)
    if (
        semantic23.dtype != np.float32
        or semantic23.shape != (len(checked[0]), 23)
        or not np.isfinite(semantic23).all()
        or not np.array_equal(semantic23, expected_semantic23)
    ):
        raise ValueError("clinical23_v2 adapter violated the exact semantic23 mapping")
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


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RAVDESS_TRIAL_ID_RE = re.compile(r"^trial_[a-z2-7]{16}$")
_RAVDESS_ACTOR_ID_RE = re.compile(r"^actor_[a-z2-7]{16}$")
_RAVDESS_CACHE_ID_RE = re.compile(r"^cache_[a-z2-7]{16}$")
_MAYO_RECORDING_ID_RE = re.compile(r"^rec_[0-9a-f]{64}$")
_MAYO_GROUP_ID_RE = re.compile(r"^grp_[0-9a-f]{64}$")
_MAYO_CACHE_ID_RE = re.compile(r"^cache_[0-9a-f]{64}$")
_MAYO_AGGREGATE_ID_RE = re.compile(r"^agg_[0-9a-f]{64}$")
_FROZEN_RAVDESS_TRIAL_COUNT = 2_452
_FROZEN_RAVDESS_ACTOR_COUNT = 24
_FROZEN_RAVDESS_SOURCE_FRAMES = 299_854
_FROZEN_RAVDESS_SAMPLE_COUNT = 2_452
_FROZEN_MAYO_MEDIAPIPE_COUNT = 48
_FROZEN_MAYO_ARKIT_COUNT = 8
_FROZEN_MAYO_CACHE_COUNT = 56
_FROZEN_MAYO_SAMPLE_COUNT = 768
_MAYO_V3_COMMITMENT_FIELDS = frozenset({
    "schema",
    "collection_manifest_sha256",
    "exposure_manifest_sha256",
    "mediapipe_file_count",
    "arkit_file_count",
    "cache_file_count",
    "cache_tree_aggregate_sha256",
    "generation_aggregate_sha256",
    "inventory_counts_sha256",
    "collection_classification_integrity_id",
    "exposure_classification_integrity_id",
})
_BUNDLE_FIELDS = frozenset({
    "features",
    "valid_mask",
    "timestamps",
    "source_frame_indices",
    "group_ids",
})
_BRIDGE_GENERATION_SCHEMA = "dynamic_landmark_bridge_generation_v1"
_BRIDGE_STAGE_SCHEMA = "dynamic_landmark_bridge_stage_v1"
_BRIDGE_RECEIPT_SCHEMA = "dynamic_landmark_bridge_receipt_v1"
_SSL_MANIFEST_SCHEMA = "dynamic_landmark_ssl_manifest_v2"
_SSL_CONFIG_SCHEMA = "dynamic_landmark_ssl_config_v2"
_SSL_SPLIT_SCHEMA = "dynamic_landmark_ssl_split_v2"
_SSL_SCALER_SCHEMA = "dynamic_landmark_ssl_scaler_v2"
_MAX_BUNDLE_BYTES = 100 * 1024 * 1024
_MAX_EXACT_TREE_DEPTH = 4
_MAX_EXACT_TREE_ENTRIES = 64
_MAX_EXACT_TREE_TOTAL_BYTES = 128 * 1024 * 1024
_SAFE_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_PRIVATE_JSON_FORBIDDEN = (
    "source_sha256",
    "private_key",
    "patient",
    "session",
    "run_mode",
    "config",
    "split",
    "scaler",
    "receipt",
)


def _require_sha256(value: object, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_private_key(value: object, field: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise ValueError(f"{field} must be an exact 32-byte private key")
    return value


def _authorization_key_identity(value: object, field: str) -> str:
    return _require_sha256(
        getattr(value, "key_file_identity_sha256", None),
        f"{field} canonical key identity",
    )


def _require_frozen_public_authorizations(
    ravdess: object,
    mayo: object,
) -> None:
    ravdess_trials = tuple(getattr(ravdess, "trials", ()))
    ravdess_trial_count = _exact_integer(
        getattr(ravdess, "trial_count", None), "RAVDESS trial count",
    )
    ravdess_expected_trials = _exact_integer(
        getattr(ravdess, "expected_trial_count", None),
        "RAVDESS expected trial count",
    )
    ravdess_actor_count = _exact_integer(
        getattr(ravdess, "actor_count", None), "RAVDESS actor count",
    )
    ravdess_expected_actors = _exact_integer(
        getattr(ravdess, "expected_actor_count", None),
        "RAVDESS expected actor count",
    )
    ravdess_source_frames = _exact_integer(
        getattr(ravdess, "source_frames", None), "RAVDESS source frames",
    )
    if (
        ravdess_trial_count != _FROZEN_RAVDESS_TRIAL_COUNT
        or ravdess_expected_trials != _FROZEN_RAVDESS_TRIAL_COUNT
        or len(ravdess_trials) != _FROZEN_RAVDESS_TRIAL_COUNT
        or ravdess_actor_count != _FROZEN_RAVDESS_ACTOR_COUNT
        or ravdess_expected_actors != _FROZEN_RAVDESS_ACTOR_COUNT
        or ravdess_source_frames != _FROZEN_RAVDESS_SOURCE_FRAMES
        or ravdess_trial_count != _FROZEN_RAVDESS_SAMPLE_COUNT
    ):
        raise ValueError("RAVDESS authorization violates the frozen production counts")
    ravdess_trial_ids: list[str] = []
    ravdess_actor_ids: list[str] = []
    ravdess_cache_ids: list[str] = []
    observed_source_frames = 0
    for trial in ravdess_trials:
        trial_id = getattr(trial, "trial_id", None)
        actor_id = getattr(trial, "actor_id", None)
        cache_id = getattr(trial, "cache_integrity_id", None)
        if (
            type(trial_id) is not str
            or _RAVDESS_TRIAL_ID_RE.fullmatch(trial_id) is None
            or type(actor_id) is not str
            or _RAVDESS_ACTOR_ID_RE.fullmatch(actor_id) is None
            or type(cache_id) is not str
            or _RAVDESS_CACHE_ID_RE.fullmatch(cache_id) is None
        ):
            raise ValueError("RAVDESS authorization contains a noncanonical opaque ID")
        trial_features = np.asarray(getattr(trial, "features", None))
        if trial_features.ndim != 2:
            raise ValueError("RAVDESS authorization source frame shape is noncanonical")
        observed_source_frames += len(trial_features)
        ravdess_trial_ids.append(trial_id)
        ravdess_actor_ids.append(actor_id)
        ravdess_cache_ids.append(cache_id)
    if (
        observed_source_frames != _FROZEN_RAVDESS_SOURCE_FRAMES
        or len(set(ravdess_trial_ids)) != _FROZEN_RAVDESS_TRIAL_COUNT
        or len(set(ravdess_actor_ids)) != _FROZEN_RAVDESS_ACTOR_COUNT
        or len(set(ravdess_cache_ids)) != _FROZEN_RAVDESS_TRIAL_COUNT
    ):
        raise ValueError("RAVDESS authorization identities or source frames do not close")

    mayo_recordings = tuple(getattr(mayo, "recordings", ()))
    mayo_recording_count = _exact_integer(
        getattr(mayo, "recording_count", None), "Mayo recording count",
    )
    mayo_expected_count = _exact_integer(
        getattr(mayo, "expected_recording_count", None),
        "Mayo expected recording count",
    )
    mayo_arkit_count = _exact_integer(
        getattr(mayo, "arkit_count", None), "Mayo ARKit count",
    )
    if (
        mayo_recording_count != _FROZEN_MAYO_MEDIAPIPE_COUNT
        or mayo_expected_count != _FROZEN_MAYO_MEDIAPIPE_COUNT
        or len(mayo_recordings) != _FROZEN_MAYO_MEDIAPIPE_COUNT
        or mayo_arkit_count != _FROZEN_MAYO_ARKIT_COUNT
        or mayo_recording_count * 16 != _FROZEN_MAYO_SAMPLE_COUNT
    ):
        raise ValueError("Mayo authorization violates the frozen production counts")
    mayo_recording_ids: list[str] = []
    mayo_group_ids: list[str] = []
    mayo_cache_ids: list[str] = []
    for recording in mayo_recordings:
        recording_id = getattr(recording, "recording_id", None)
        group_id = getattr(recording, "group_id", None)
        cache_id = getattr(recording, "cache_integrity_id", None)
        if (
            type(recording_id) is not str
            or _MAYO_RECORDING_ID_RE.fullmatch(recording_id) is None
            or type(group_id) is not str
            or _MAYO_GROUP_ID_RE.fullmatch(group_id) is None
            or type(cache_id) is not str
            or _MAYO_CACHE_ID_RE.fullmatch(cache_id) is None
        ):
            raise ValueError("Mayo authorization contains a noncanonical opaque ID")
        mayo_recording_ids.append(recording_id)
        mayo_group_ids.append(group_id)
        mayo_cache_ids.append(cache_id)
    if (
        len(set(mayo_recording_ids)) != _FROZEN_MAYO_MEDIAPIPE_COUNT
        or len(set(mayo_group_ids)) != _FROZEN_MAYO_MEDIAPIPE_COUNT
        or len(set(mayo_cache_ids)) != _FROZEN_MAYO_MEDIAPIPE_COUNT
    ):
        raise ValueError("Mayo authorization opaque identities do not close")

    commitment = getattr(mayo, "commitment", None)
    if not isinstance(commitment, Mapping) or set(commitment) != _MAYO_V3_COMMITMENT_FIELDS:
        raise ValueError("Mayo v3 generation commitment field set is noncanonical")
    if commitment.get("schema") != "mayo_cache_generation_commitment_v3":
        raise ValueError("Mayo v3 generation commitment schema is noncanonical")
    for field in (
        "collection_manifest_sha256",
        "exposure_manifest_sha256",
        "cache_tree_aggregate_sha256",
        "generation_aggregate_sha256",
        "inventory_counts_sha256",
    ):
        _require_sha256(commitment.get(field), f"Mayo commitment {field}")
    for field in (
        "collection_classification_integrity_id",
        "exposure_classification_integrity_id",
    ):
        value = commitment.get(field)
        if type(value) is not str or _MAYO_AGGREGATE_ID_RE.fullmatch(value) is None:
            raise ValueError(f"Mayo commitment {field} is noncanonical")
    commitment_mediapipe = _exact_integer(
        commitment.get("mediapipe_file_count"), "Mayo commitment MediaPipe count",
    )
    commitment_arkit = _exact_integer(
        commitment.get("arkit_file_count"), "Mayo commitment ARKit count",
    )
    commitment_cache = _exact_integer(
        commitment.get("cache_file_count"), "Mayo commitment cache count",
    )
    if (
        commitment_mediapipe != _FROZEN_MAYO_MEDIAPIPE_COUNT
        or commitment_arkit != _FROZEN_MAYO_ARKIT_COUNT
        or commitment_cache != _FROZEN_MAYO_CACHE_COUNT
        or commitment_cache != commitment_mediapipe + commitment_arkit
        or commitment_mediapipe != mayo_recording_count
        or commitment_arkit != mayo_arkit_count
    ):
        raise ValueError("Mayo v3 generation commitment counts do not close")
    collection_sha256 = _require_sha256(
        getattr(mayo, "collection_manifest_sha256", None),
        "Mayo collection manifest",
    )
    exposure_sha256 = _require_sha256(
        getattr(mayo, "exposure_manifest_sha256", None),
        "Mayo exposure manifest",
    )
    if (
        not hmac.compare_digest(
            str(commitment["collection_manifest_sha256"]), collection_sha256,
        )
        or not hmac.compare_digest(
            str(commitment["exposure_manifest_sha256"]), exposure_sha256,
        )
    ):
        raise ValueError("Mayo v3 generation commitment digests disagree with authorization")


def _int_mapping_rows(value: np.ndarray) -> tuple[tuple[tuple[int, ...], ...], ...]:
    array = np.asarray(value)
    if array.dtype != np.int64 or array.ndim != 3 or array.shape[1:] != (4, 32):
        raise ValueError("private integer mapping is not aligned to packet slots")
    return tuple(
        tuple(tuple(int(item) for item in window) for window in packet)
        for packet in array
    )


def _float_mapping_rows(
    value: np.ndarray,
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    array = np.asarray(value)
    if (
        array.dtype != np.float64
        or array.ndim != 3
        or array.shape[1:] != (4, 32)
        or not np.isfinite(array).all()
    ):
        raise ValueError("private timestamp mapping is not aligned to packet slots")
    return tuple(
        tuple(tuple(float(item) for item in window) for window in packet)
        for packet in array
    )


def _window_has_two_mask_spans(value: np.ndarray, *, span_length: int = 4) -> bool:
    mask = np.asarray(value)
    if mask.dtype != np.bool_ or mask.shape != (32,):
        raise ValueError("bridge validity window is noncanonical")
    available = 0
    run = 0
    for observed in (*mask.tolist(), False):
        if observed:
            run += 1
        else:
            available += run // span_length
            run = 0
    return available >= 2


def _require_mask_span_capacity(value: np.ndarray, stage: str) -> None:
    masks = np.asarray(value)
    if masks.dtype != np.bool_ or masks.ndim != 3 or masks.shape[1:] != (4, 32):
        raise ValueError(f"{stage} bridge validity tensor is noncanonical")
    for packet_index, packet in enumerate(masks):
        for window_index, window in enumerate(packet):
            if not _window_has_two_mask_spans(window):
                raise ValueError(
                    f"{stage} packet {packet_index} window {window_index} lacks "
                    "two non-overlapping valid spans of length four"
                )


def _json_bytes(value: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("bridge metadata is not canonical JSON") from exc


def _json_sha256(value: object) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("digest metadata must be a mapping")
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _commit_code_value(digest: "hashlib._Hash", value: object) -> None:
    """Commit semantic code constants without paths or line-table metadata."""
    if value is None:
        digest.update(b"none\0")
    elif value is Ellipsis:
        digest.update(b"ellipsis\0")
    elif type(value) is bool:
        digest.update(b"bool\0" + (b"1" if value else b"0"))
    elif type(value) is int:
        digest.update(b"int\0" + str(value).encode("ascii") + b"\0")
    elif type(value) is float:
        digest.update(b"float\0" + value.hex().encode("ascii") + b"\0")
    elif type(value) is complex:
        digest.update(b"complex\0")
        _commit_code_value(digest, value.real)
        _commit_code_value(digest, value.imag)
    elif type(value) is str:
        payload = value.encode("utf-8")
        digest.update(b"str\0" + len(payload).to_bytes(8, "big") + payload)
    elif type(value) is bytes:
        digest.update(b"bytes\0" + len(value).to_bytes(8, "big") + value)
    elif type(value) is tuple:
        digest.update(b"tuple\0" + len(value).to_bytes(8, "big"))
        for item in value:
            _commit_code_value(digest, item)
    elif type(value) is frozenset:
        item_digests: list[bytes] = []
        for item in value:
            item_digest = hashlib.sha256()
            _commit_code_value(item_digest, item)
            item_digests.append(item_digest.digest())
        digest.update(b"frozenset\0" + len(item_digests).to_bytes(8, "big"))
        for item_digest in sorted(item_digests):
            digest.update(item_digest)
    elif isinstance(value, CodeType):
        _commit_code_object(digest, value)
    else:
        raise TypeError(f"unsupported deterministic code constant: {type(value)!r}")


def _commit_code_object(digest: "hashlib._Hash", code: CodeType) -> None:
    """Bind executable semantics while excluding filename and debug locations."""
    digest.update(b"code-object-v1\0")
    for value in (
        code.co_argcount,
        getattr(code, "co_posonlyargcount", 0),
        code.co_kwonlyargcount,
        code.co_nlocals,
        code.co_stacksize,
        code.co_flags,
    ):
        digest.update(int(value).to_bytes(8, "big", signed=True))
    digest.update(len(code.co_code).to_bytes(8, "big") + code.co_code)
    exception_table = getattr(code, "co_exceptiontable", b"")
    digest.update(len(exception_table).to_bytes(8, "big") + exception_table)
    for name, values in (
        ("names", code.co_names),
        ("varnames", code.co_varnames),
        ("freevars", code.co_freevars),
        ("cellvars", code.co_cellvars),
    ):
        digest.update(name.encode("ascii") + b"\0")
        _commit_code_value(digest, tuple(values))
    digest.update(b"name\0")
    _commit_code_value(digest, code.co_name)
    digest.update(b"constants\0")
    _commit_code_value(digest, code.co_consts)


def _adapter_lineage_sha256(
    metadata: Mapping[str, object],
    adapter: Callable[[np.ndarray], np.ndarray],
    sentinel: np.ndarray,
) -> str:
    """Bind declared metadata, the live callable, defaults, and known output."""
    code = getattr(adapter, "__code__", None)
    if code is None or not callable(adapter):
        raise ValueError("feature adapter must be one inspectable Python callable")
    defaults = getattr(adapter, "__defaults__", None)
    keyword_defaults = getattr(adapter, "__kwdefaults__", None)
    try:
        defaults_bytes = json.dumps(
            {"defaults": defaults, "keyword_defaults": keyword_defaults},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        source = np.ascontiguousarray(sentinel)
        output = np.ascontiguousarray(adapter(source.copy()))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("feature adapter lineage is not deterministic") from exc
    if output.dtype == np.dtype("O") or not np.issubdtype(output.dtype, np.number):
        raise ValueError("feature adapter sentinel output must be numeric")
    if not np.isfinite(output).all():
        raise ValueError("feature adapter sentinel output must be finite")
    digest = hashlib.sha256()
    digest.update(b"dynamic-landmark-feature-adapter-lineage-v2\0")
    digest.update(_json_bytes(metadata))
    _commit_code_object(digest, code)
    digest.update(len(defaults_bytes).to_bytes(8, "big"))
    digest.update(defaults_bytes)
    _array_commitment(digest, "sentinel_input", source)
    _array_commitment(digest, "sentinel_output", output)
    return digest.hexdigest()


def _openface_adapter_lineage_sha256() -> str:
    index = np.arange(68, dtype=np.float64)
    sentinel = np.stack(
        (index, np.sin(index / 5.0) + index / 20.0), axis=1,
    )
    return _adapter_lineage_sha256(
        OPENFACE68_ADAPTER_METADATA, openface68_to_semantic23, sentinel,
    )


def _clinical_adapter_lineage_sha256() -> str:
    sentinel = (
        np.arange(46, dtype=np.float32).reshape(2, 23) / np.float32(47.0)
    )
    return _adapter_lineage_sha256(
        CLINICAL23_V2_ADAPTER_METADATA,
        clinical23_v2_to_semantic23,
        sentinel,
    )


def _array_commitment(digest: "hashlib._Hash", name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(name.encode("ascii") + b"\0")
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii") + b"\0")
    digest.update(array.tobytes(order="C"))


def _mapping_commitment(
    source_unit_id: str,
    mapping: PrivateTrajectoryMapping,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"dynamic-landmark-bridge-original-mapping-v1\0")
    digest.update(source_unit_id.encode("utf-8") + b"\0")
    _array_commitment(digest, "window_starts", mapping.window_starts)
    _array_commitment(
        digest,
        "original_canonical_frame_indices",
        mapping.original_canonical_frame_indices,
    )
    _array_commitment(
        digest,
        "original_source_frame_indices",
        mapping.original_source_frame_indices,
    )
    _array_commitment(digest, "original_timestamps", mapping.original_timestamps)
    return digest.hexdigest()


def _opaque_packet_id(
    *,
    key: bytes,
    stage: str,
    source_unit_id: str,
    packet_index: int,
) -> str:
    material = (
        b"dynamic-landmark-bridge-packet-id-v1\0"
        + stage.encode("ascii")
        + b"\0"
        + source_unit_id.encode("utf-8")
        + b"\0"
        + str(packet_index).encode("ascii")
    )
    return "pkt_" + hmac.new(key, material, hashlib.sha256).hexdigest()


def _npz_bundle_bytes(
    *,
    features: np.ndarray,
    valid_mask: np.ndarray,
    timestamps: np.ndarray,
    source_frame_indices: np.ndarray,
    group_ids: Sequence[str],
) -> bytes:
    group_array = np.asarray(tuple(group_ids), dtype=np.str_)
    buffer = io.BytesIO()
    np.savez(
        buffer,
        features=np.ascontiguousarray(features, dtype=np.float32),
        valid_mask=np.ascontiguousarray(valid_mask, dtype=np.bool_),
        timestamps=np.ascontiguousarray(timestamps, dtype=np.float32),
        source_frame_indices=np.ascontiguousarray(
            source_frame_indices, dtype=np.int64
        ),
        group_ids=group_array,
    )
    payload = buffer.getvalue()
    if not payload or len(payload) > _MAX_BUNDLE_BYTES:
        raise ValueError("bridge bundle size is outside the frozen private bound")
    return payload


def _packet_coverage(window_starts: np.ndarray, window_length: int) -> tuple[int, int]:
    starts = np.asarray(window_starts, dtype=np.int64)
    overlap_pairs = 0
    covered: set[int] = set()
    for packet in starts:
        for first in range(len(packet)):
            left = int(packet[first])
            covered.update(range(left, left + window_length))
            for second in range(first + 1, len(packet)):
                right = int(packet[second])
                if max(left, right) < min(
                    left + window_length, right + window_length
                ):
                    overlap_pairs += 1
    return overlap_pairs, len(covered)


def _stage_record(
    *,
    name: str,
    key: bytes,
    producer_sha256: str,
    source_schema: str,
    upstream_manifest_commitments: Mapping[str, str],
    upstream_generation_closure_hmac: str,
    feature_names: Sequence[str],
    adapter_sha256: str,
    bundle_bytes: bytes,
    sample_ids: tuple[str, ...],
    source_unit_ids: tuple[str, ...],
    group_ids: tuple[str, ...],
    cache_integrity_ids: tuple[str, ...],
    window_starts: tuple[tuple[int, ...], ...],
    mapping_commitments: tuple[str, ...],
    overlap_pairs: int,
    covered_positions: int,
    exclusions: int,
) -> dict[str, object]:
    names_digest = hashlib.sha256(
        _json_bytes({"feature_names": list(feature_names)})
    ).hexdigest()
    mapping_digest = hashlib.sha256()
    mapping_digest.update(b"dynamic-landmark-bridge-mapping-set-v1\0")
    for value in mapping_commitments:
        mapping_digest.update(value.encode("ascii") + b"\n")
    unique_sources = len(set(source_unit_ids))
    unique_groups = len(set(group_ids))
    unique_caches = len(set(cache_integrity_ids))
    record: dict[str, object] = {
        "schema": _BRIDGE_STAGE_SCHEMA,
        "stage": name,
        "producer_sha256": producer_sha256,
        "source_schema": source_schema,
        "upstream_manifest_commitments": dict(upstream_manifest_commitments),
        "upstream_generation_closure_hmac": upstream_generation_closure_hmac,
        "sample_ids": list(sample_ids),
        "source_unit_ids": list(source_unit_ids),
        "group_ids": list(group_ids),
        "cache_integrity_ids": list(cache_integrity_ids),
        "window_starts": [list(item) for item in window_starts],
        "original_mapping_sha256": mapping_digest.hexdigest(),
        "feature_names_sha256": names_digest,
        "adapter_sha256": _require_sha256(adapter_sha256, "feature adapter lineage"),
        "bundle_file_count": 1,
        "sample_count": len(sample_ids),
        "source_unit_count": unique_sources,
        "unique_group_count": unique_groups,
        "upstream_cache_count": unique_caches,
        "packet_policy": {
            "sample_rate_hz": 30.0,
            "window_length": 32,
            "windows_per_packet": 4,
            "selection": "uniform_floor_v1",
        },
        "overlap_pair_count": overlap_pairs,
        "covered_canonical_position_count": covered_positions,
        "exclusion_count": exclusions,
        "bundle_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        "bundle_size_bytes": len(bundle_bytes),
    }
    closure_hmac = hmac.new(
        key,
        b"dynamic-landmark-bridge-stage-closure-v1\0" + _json_bytes(record),
        hashlib.sha256,
    ).hexdigest()
    record["closure_hmac"] = closure_hmac
    return record


def _prepare_ravdess_stage(
    authorization: object,
    *,
    producer_sha256: str,
) -> _PreparedBridgeStage:
    key = _require_private_key(getattr(authorization, "private_key", None), "RAVDESS key")
    key_identity = _authorization_key_identity(authorization, "RAVDESS")
    if getattr(authorization, "schema", None) != SEMANTIC23_SCHEMA:
        raise ValueError("RAVDESS authorization schema is noncanonical")
    trial_count = _exact_integer(getattr(authorization, "trial_count", None), "RAVDESS trial count")
    actor_count = _exact_integer(getattr(authorization, "actor_count", None), "RAVDESS actor count")
    expected_trials = _exact_integer(
        getattr(authorization, "expected_trial_count", None), "RAVDESS expected trial count"
    )
    expected_actors = _exact_integer(
        getattr(authorization, "expected_actor_count", None), "RAVDESS expected actor count"
    )
    trials = tuple(getattr(authorization, "trials", ()))
    if trial_count != expected_trials or actor_count != expected_actors or len(trials) != trial_count:
        raise ValueError("RAVDESS authorization aggregate is incomplete")
    manifest_sha256 = _require_sha256(
        getattr(authorization, "manifest_sha256", None), "RAVDESS manifest"
    )
    generation_closure = _require_sha256(
        getattr(authorization, "generation_closure_hmac", None),
        "RAVDESS generation closure",
    )
    features: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    sample_ids: list[str] = []
    source_ids: list[str] = []
    groups: list[str] = []
    cache_ids: list[str] = []
    starts: list[tuple[int, ...]] = []
    mappings: list[str] = []
    original_canonical: list[tuple[tuple[int, ...], ...]] = []
    original_source: list[tuple[tuple[int, ...], ...]] = []
    original_times: list[tuple[tuple[float, ...], ...]] = []
    overlap_pairs = covered_positions = 0
    seen_units: set[str] = set()
    for trial in trials:
        trial_id = getattr(trial, "trial_id", None)
        actor_id = getattr(trial, "actor_id", None)
        cache_id = getattr(trial, "cache_integrity_id", None)
        if not all(type(value) is str and value for value in (trial_id, actor_id, cache_id)):
            raise ValueError("RAVDESS authorization contains invalid opaque IDs")
        if trial_id in seen_units:
            raise ValueError("RAVDESS authorization repeats a source unit")
        seen_units.add(trial_id)
        trial_features = np.asarray(getattr(trial, "features", None))
        length = len(trial_features) if trial_features.ndim == 2 else 0
        bundle, mapping = packetize_ravdess_trajectory(
            trial_features,
            np.asarray(getattr(trial, "valid_mask", None)),
            canonical_frame_indices=np.arange(length, dtype=np.int64),
            original_source_frame_indices=np.asarray(
                getattr(trial, "frame_indices", None)
            ),
            original_timestamps=np.asarray(getattr(trial, "timestamps", None)),
            feature_schema=SEMANTIC23_SCHEMA,
            feature_names=SEMANTIC23_FEATURE_NAMES,
        )
        _require_mask_span_capacity(bundle.valid_mask, "ravdess")
        features.append(bundle.features)
        masks.append(bundle.valid_mask)
        timestamps.append(bundle.timestamps)
        indices.append(bundle.source_frame_indices)
        sample_ids.append(
            _opaque_packet_id(
                key=key, stage="ravdess", source_unit_id=trial_id, packet_index=0
            )
        )
        source_ids.append(trial_id)
        groups.append(actor_id)
        cache_ids.append(cache_id)
        starts.append(tuple(int(value) for value in mapping.window_starts[0]))
        mappings.append(_mapping_commitment(trial_id, mapping))
        original_canonical.extend(
            _int_mapping_rows(mapping.original_canonical_frame_indices)
        )
        original_source.extend(
            _int_mapping_rows(mapping.original_source_frame_indices)
        )
        original_times.extend(_float_mapping_rows(mapping.original_timestamps))
        overlap, covered = _packet_coverage(mapping.window_starts, 32)
        overlap_pairs += overlap
        covered_positions += covered
    if len(set(groups)) != actor_count:
        raise ValueError("RAVDESS actor grouping does not close")
    bundle_bytes = _npz_bundle_bytes(
        features=np.concatenate(features, axis=0),
        valid_mask=np.concatenate(masks, axis=0),
        timestamps=np.concatenate(timestamps, axis=0),
        source_frame_indices=np.concatenate(indices, axis=0),
        group_ids=groups,
    )
    record = _stage_record(
        name="ravdess",
        key=key,
        producer_sha256=producer_sha256,
        source_schema=SEMANTIC23_SCHEMA,
        upstream_manifest_commitments={"manifest_sha256": manifest_sha256},
        upstream_generation_closure_hmac=generation_closure,
        feature_names=SEMANTIC23_FEATURE_NAMES,
        adapter_sha256=_openface_adapter_lineage_sha256(),
        bundle_bytes=bundle_bytes,
        sample_ids=tuple(sample_ids),
        source_unit_ids=tuple(source_ids),
        group_ids=tuple(groups),
        cache_integrity_ids=tuple(cache_ids),
        window_starts=tuple(starts),
        mapping_commitments=tuple(mappings),
        overlap_pairs=overlap_pairs,
        covered_positions=covered_positions,
        exclusions=0,
    )
    return _PreparedBridgeStage(
        name="ravdess",
        bundle_bytes=bundle_bytes,
        record=record,
        sample_ids=tuple(sample_ids),
        source_unit_ids=tuple(source_ids),
        group_ids=tuple(groups),
        cache_integrity_ids=tuple(cache_ids),
        window_starts=tuple(starts),
        original_canonical_frame_indices=tuple(original_canonical),
        original_source_frame_indices=tuple(original_source),
        original_timestamps=tuple(original_times),
        canonical_key_identity_sha256=key_identity,
        private_key=key,
    )


def _prepare_mayo_stage(
    authorization: object,
    *,
    producer_sha256: str,
) -> _PreparedBridgeStage:
    key = _require_private_key(getattr(authorization, "private_key", None), "Mayo key")
    key_identity = _authorization_key_identity(authorization, "Mayo")
    if getattr(authorization, "schema", None) != "mayo_mediapipe_clinical23_ssl_v2":
        raise ValueError("Mayo authorization schema is noncanonical")
    recording_count = _exact_integer(
        getattr(authorization, "recording_count", None), "Mayo recording count"
    )
    expected_count = _exact_integer(
        getattr(authorization, "expected_recording_count", None),
        "Mayo expected recording count",
    )
    recordings = tuple(getattr(authorization, "recordings", ()))
    if recording_count != expected_count or len(recordings) != recording_count:
        raise ValueError("Mayo authorization aggregate is incomplete")
    collection_sha256 = _require_sha256(
        getattr(authorization, "collection_manifest_sha256", None),
        "Mayo collection manifest",
    )
    exposure_sha256 = _require_sha256(
        getattr(authorization, "exposure_manifest_sha256", None),
        "Mayo exposure manifest",
    )
    generation_closure = _require_sha256(
        getattr(authorization, "generation_closure_hmac", None),
        "Mayo generation closure",
    )
    commitment = getattr(authorization, "commitment", None)
    if not isinstance(commitment, Mapping) or commitment.get("schema") != "mayo_cache_generation_commitment_v3":
        raise ValueError("Mayo generation commitment is noncanonical")
    features: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    sample_ids: list[str] = []
    source_ids: list[str] = []
    groups: list[str] = []
    cache_ids: list[str] = []
    starts: list[tuple[int, ...]] = []
    mappings: list[str] = []
    original_canonical: list[tuple[tuple[int, ...], ...]] = []
    original_source: list[tuple[tuple[int, ...], ...]] = []
    original_times: list[tuple[tuple[float, ...], ...]] = []
    overlap_pairs = covered_positions = 0
    seen_units: set[str] = set()
    for recording in recordings:
        recording_id = getattr(recording, "recording_id", None)
        group_id = getattr(recording, "group_id", None)
        cache_id = getattr(recording, "cache_integrity_id", None)
        if not all(type(value) is str and value for value in (recording_id, group_id, cache_id)):
            raise ValueError("Mayo authorization contains invalid opaque IDs")
        if recording_id in seen_units:
            raise ValueError("Mayo authorization repeats a source unit")
        seen_units.add(recording_id)
        bundle, mapping = packetize_mayo_trajectory(
            np.asarray(getattr(recording, "features_30hz", None)),
            np.asarray(getattr(recording, "valid_mask_30hz", None)),
            canonical_frame_indices=np.asarray(
                getattr(recording, "target_frame_indices_30hz", None)
            ),
            original_source_frame_indices=np.asarray(
                getattr(recording, "source_frame_indices_30hz", None)
            ),
            original_timestamps=np.asarray(
                getattr(recording, "timestamps_30hz", None)
            ),
            feature_schema=DYNAMIC_FEATURE_SCHEMA,
            feature_names=DYNAMIC_FEATURE_NAMES,
        )
        _require_mask_span_capacity(bundle.valid_mask, "mayo")
        features.append(bundle.features)
        masks.append(bundle.valid_mask)
        timestamps.append(bundle.timestamps)
        indices.append(bundle.source_frame_indices)
        mapping_digest = _mapping_commitment(recording_id, mapping)
        original_canonical.extend(
            _int_mapping_rows(mapping.original_canonical_frame_indices)
        )
        original_source.extend(
            _int_mapping_rows(mapping.original_source_frame_indices)
        )
        original_times.extend(_float_mapping_rows(mapping.original_timestamps))
        overlap, covered = _packet_coverage(mapping.window_starts, 32)
        overlap_pairs += overlap
        covered_positions += covered
        for packet_index in range(16):
            sample_ids.append(
                _opaque_packet_id(
                    key=key,
                    stage="mayo",
                    source_unit_id=recording_id,
                    packet_index=packet_index,
                )
            )
            source_ids.append(recording_id)
            groups.append(group_id)
            cache_ids.append(cache_id)
            starts.append(
                tuple(int(value) for value in mapping.window_starts[packet_index])
            )
            mappings.append(
                hashlib.sha256(
                    (
                        "dynamic-landmark-bridge-packet-mapping-v1\0"
                        + mapping_digest
                        + f"\0{packet_index}"
                    ).encode("ascii")
                ).hexdigest()
            )
    if len(set(groups)) != recording_count:
        raise ValueError("Mayo bridge requires one recording group per source unit")
    bundle_bytes = _npz_bundle_bytes(
        features=np.concatenate(features, axis=0),
        valid_mask=np.concatenate(masks, axis=0),
        timestamps=np.concatenate(timestamps, axis=0),
        source_frame_indices=np.concatenate(indices, axis=0),
        group_ids=groups,
    )
    record = _stage_record(
        name="mayo",
        key=key,
        producer_sha256=producer_sha256,
        source_schema=DYNAMIC_FEATURE_SCHEMA,
        upstream_manifest_commitments={
            "collection_manifest_sha256": collection_sha256,
            "exposure_manifest_sha256": exposure_sha256,
            "generation_commitment_sha256": _json_sha256(commitment),
        },
        upstream_generation_closure_hmac=generation_closure,
        feature_names=DYNAMIC_FEATURE_NAMES,
        adapter_sha256=_clinical_adapter_lineage_sha256(),
        bundle_bytes=bundle_bytes,
        sample_ids=tuple(sample_ids),
        source_unit_ids=tuple(source_ids),
        group_ids=tuple(groups),
        cache_integrity_ids=tuple(cache_ids),
        window_starts=tuple(starts),
        mapping_commitments=tuple(mappings),
        overlap_pairs=overlap_pairs,
        covered_positions=covered_positions,
        exclusions=0,
    )
    return _PreparedBridgeStage(
        name="mayo",
        bundle_bytes=bundle_bytes,
        record=record,
        sample_ids=tuple(sample_ids),
        source_unit_ids=tuple(source_ids),
        group_ids=tuple(groups),
        cache_integrity_ids=tuple(cache_ids),
        window_starts=tuple(starts),
        original_canonical_frame_indices=tuple(original_canonical),
        original_source_frame_indices=tuple(original_source),
        original_timestamps=tuple(original_times),
        canonical_key_identity_sha256=key_identity,
        private_key=key,
    )


def _prepare_bridge_generation(
    ravdess_authorization: object,
    mayo_authorization: object,
    *,
    producer_sha256: str,
) -> _PreparedBridgeGeneration:
    producer_sha256 = _require_sha256(producer_sha256, "bridge producer")
    ravdess = _prepare_ravdess_stage(
        ravdess_authorization, producer_sha256=producer_sha256
    )
    mayo = _prepare_mayo_stage(mayo_authorization, producer_sha256=producer_sha256)
    total_bundle_bytes = len(ravdess.bundle_bytes) + len(mayo.bundle_bytes)
    if not 0 < total_bundle_bytes <= _MAX_BUNDLE_BYTES:
        raise ValueError("dual-stage bridge bundle size exceeds the aggregate bound")
    stages = {"ravdess": ravdess.record, "mayo": mayo.record}
    dual_digest = hashlib.sha256()
    dual_digest.update(b"dynamic-landmark-bridge-dual-stage-v1\0")
    for name in ("ravdess", "mayo"):
        dual_digest.update(name.encode("ascii") + b"\0")
        dual_digest.update(str(stages[name]["closure_hmac"]).encode("ascii") + b"\n")
    unsigned_generation: dict[str, object] = {
        "schema": _BRIDGE_GENERATION_SCHEMA,
        "producer_sha256": producer_sha256,
        "stages": stages,
        "dual_stage_closure_sha256": dual_digest.hexdigest(),
    }
    dual_material = (
        b"dynamic-landmark-bridge-dual-stage-keyed-v1\0"
        + _json_bytes(unsigned_generation)
    )
    generation = {
        **unsigned_generation,
        "dual_stage_closure_hmac": {
            "ravdess": hmac.new(
                ravdess.private_key, dual_material, hashlib.sha256
            ).hexdigest(),
            "mayo": hmac.new(
                mayo.private_key, dual_material, hashlib.sha256
            ).hexdigest(),
        },
    }
    generation_bytes = _json_bytes(generation)
    _assert_private_metadata(generation_bytes)
    return _PreparedBridgeGeneration(
        ravdess=ravdess,
        mayo=mayo,
        generation=generation,
        generation_bytes=generation_bytes,
    )


def _assert_private_metadata(payload: bytes) -> None:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("private bridge metadata must be ASCII JSON") from exc
    lowered = text.lower()
    if any(value in lowered for value in _PRIVATE_JSON_FORBIDDEN):
        raise ValueError("private bridge metadata contains a forbidden field")
    if re.search(r"(?:faces|myslate)[_ ]*\d+", text, re.IGNORECASE):
        raise ValueError("private bridge metadata contains a Mayo source identifier")
    if "/" in text or "\\" in text:
        raise ValueError("private bridge metadata contains a path-like value")


def _resolve_existing_private_directory(value: Path, field: str) -> Path:
    lexical = Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{field} cannot be canonicalized") from exc
    if resolved == lexical:
        return resolved
    aliases = (
        (Path("/var"), Path("/private/var")),
        (Path("/tmp"), Path("/private/tmp")),
    ) if sys.platform == "darwin" else ()
    for source, destination in aliases:
        try:
            relative = lexical.relative_to(source)
        except ValueError:
            continue
        if resolved == destination / relative:
            return resolved
    raise ValueError(f"{field} contains a symlink component")


def _lexical_absolute(value: str | Path) -> Path:
    lexical = Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    if lexical.name in {"", ".", ".."} or Path(lexical.name).name != lexical.name:
        raise ValueError("bridge output must be one safe path component")
    parent = _resolve_existing_private_directory(
        lexical.parent, "bridge output parent",
    )
    return parent / lexical.name


@dataclass(frozen=True)
class _DirectoryAnchor:
    path: Path
    descriptor: int = dataclass_field(repr=False)
    identity: tuple[int, int]
    mode: int


@dataclass(frozen=True)
class _DestinationLock:
    name: str
    descriptor: int = dataclass_field(repr=False)
    identity: tuple[int, ...]


@dataclass
class _OwnedTreeLedger:
    root_identity: tuple[int, int]
    root_nlink: int
    entries: dict[str, tuple[str, tuple[int, int], int]]

    @classmethod
    def create(cls, root_stat: os.stat_result) -> "_OwnedTreeLedger":
        if not stat.S_ISDIR(root_stat.st_mode):
            raise RuntimeError("private tree ledger root is not a directory")
        return cls(
            root_identity=_inode_identity(root_stat),
            root_nlink=int(root_stat.st_nlink),
            entries={},
        )

    def record(self, relative: str, kind: str, value: os.stat_result) -> None:
        if kind not in {"file", "directory"} or not relative:
            raise RuntimeError("private tree ledger entry is malformed")
        self.entries[relative] = (
            kind,
            _inode_identity(value),
            int(value.st_nlink),
        )


def _inode_identity(value: os.stat_result) -> tuple[int, int]:
    return int(value.st_dev), int(value.st_ino)


def _full_file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_mode),
        int(value.st_uid), int(value.st_gid), int(value.st_nlink),
        int(value.st_size), int(value.st_mtime_ns), int(value.st_ctime_ns),
    )


def _safe_entry_name(value: str) -> str:
    if type(value) is not str or value in {"", ".", ".."} or Path(value).name != value:
        raise ValueError("private entry name must be one safe component")
    return value


def _require_safe_output_parent(output: Path) -> None:
    parent = output.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("bridge output parent must be a real existing directory")
    parent_status = os.stat(parent, follow_symlinks=False)
    if (
        parent_status.st_uid != os.geteuid()
        or stat.S_IMODE(parent_status.st_mode) != 0o700
    ):
        raise ValueError("bridge output parent must be current-owner mode 0700")
    current = Path(parent.anchor)
    for part in parent.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError("bridge output parent must not contain symlinks")
    if output.name in {"", ".", ".."} or Path(output.name).name != output.name:
        raise ValueError("bridge output must be one safe path component")


def _open_directory_anchor(
    path: Path,
    field: str,
    *,
    expected_mode: int | None = 0o700,
) -> _DirectoryAnchor:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{field} is unavailable or unsafe") from exc
    try:
        info = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        opened_mode = stat.S_IMODE(info.st_mode)
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or (
                expected_mode is not None
                and opened_mode != expected_mode
            )
            or (
                expected_mode is None
                and opened_mode & 0o022
            )
            or _inode_identity(info) != _inode_identity(current)
            or current.st_uid != os.geteuid()
            or stat.S_IMODE(current.st_mode) != opened_mode
        ):
            raise ValueError(f"{field} identity or permissions are unsafe")
        return _DirectoryAnchor(
            path=path,
            descriptor=descriptor,
            identity=_inode_identity(info),
            mode=opened_mode,
        )
    except Exception:
        os.close(descriptor)
        raise


def _close_anchor(anchor: _DirectoryAnchor) -> None:
    os.close(anchor.descriptor)


def _assert_directory_anchor(anchor: _DirectoryAnchor, field: str) -> None:
    try:
        descriptor_stat = os.fstat(anchor.descriptor)
        path_stat = os.stat(anchor.path, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"{field} identity changed") from exc
    if (
        not stat.S_ISDIR(descriptor_stat.st_mode)
        or not stat.S_ISDIR(path_stat.st_mode)
        or descriptor_stat.st_uid != os.geteuid()
        or path_stat.st_uid != os.geteuid()
        or stat.S_IMODE(descriptor_stat.st_mode) != anchor.mode
        or stat.S_IMODE(path_stat.st_mode) != anchor.mode
        or _inode_identity(descriptor_stat) != anchor.identity
        or _inode_identity(path_stat) != anchor.identity
    ):
        raise ValueError(f"{field} identity changed")


def _entry_stat_at(directory_fd: int, name: str) -> os.stat_result | None:
    _safe_entry_name(name)
    try:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _require_entry_absent_at(directory_fd: int, name: str, field: str) -> None:
    if _entry_stat_at(directory_fd, name) is not None:
        raise FileExistsError(f"{field} already exists")


def _reject_transaction_residue_at(
    directory_fd: int,
    prefixes: Sequence[str],
    field: str,
) -> None:
    names = set(os.listdir(directory_fd))
    for name in names:
        _safe_entry_name(name)
    if any(name.startswith(prefix) for name in names for prefix in prefixes):
        raise ValueError(f"{field} contains unresolved transaction state")


def _require_destination_lock_stat(value: os.stat_result, field: str) -> None:
    if (
        not stat.S_ISREG(value.st_mode)
        or value.st_uid != os.geteuid()
        or stat.S_IMODE(value.st_mode) != 0o600
        or int(value.st_nlink) != 1
        or int(value.st_size) != 0
    ):
        raise ValueError(
            f"{field} must be a singly-linked current-owner mode-0600 empty file"
        )


def _acquire_destination_lock_at(
    parent_fd: int,
    name: str,
    *,
    create: bool,
    exclusive: bool,
    field: str,
) -> _DestinationLock:
    _safe_entry_name(name)
    parent_stat = os.fstat(parent_fd)
    if (
        not stat.S_ISDIR(parent_stat.st_mode)
        or parent_stat.st_uid != os.geteuid()
        or stat.S_IMODE(parent_stat.st_mode) != 0o700
    ):
        raise ValueError(f"{field} parent is not exact private storage")
    access = os.O_RDWR if exclusive else os.O_RDONLY
    flags = (
        access
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    created = False
    creation_guard: int | None = None
    try:
        if create:
            creation_guard = os.dup(parent_fd)
            fcntl.flock(creation_guard, fcntl.LOCK_EX)
            try:
                descriptor = os.open(
                    name,
                    flags | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=parent_fd,
                )
                created = True
            except FileExistsError:
                descriptor = os.open(name, flags, dir_fd=parent_fd)
        else:
            descriptor = os.open(name, flags, dir_fd=parent_fd)
    except BaseException as exc:
        if creation_guard is not None:
            try:
                fcntl.flock(creation_guard, fcntl.LOCK_UN)
            finally:
                os.close(creation_guard)
        if isinstance(exc, FileNotFoundError):
            raise ValueError(f"{field} is missing") from exc
        raise
    try:
        if created:
            os.fchmod(descriptor, 0o600)
        opened = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        _require_destination_lock_stat(opened, field)
        _require_destination_lock_stat(current, field)
        if _full_file_identity(opened) != _full_file_identity(current):
            raise ValueError(f"{field} name does not bind its opened storage")
        if created:
            os.fsync(descriptor)
            _fsync_directory_fd(parent_fd)
            opened = os.fstat(descriptor)
            current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            _require_destination_lock_stat(opened, field)
            _require_destination_lock_stat(current, field)
            if _full_file_identity(opened) != _full_file_identity(current):
                raise ValueError(f"{field} changed during creation durability sync")
        if creation_guard is not None:
            fcntl.flock(creation_guard, fcntl.LOCK_UN)
            os.close(creation_guard)
            creation_guard = None
        fcntl.flock(
            descriptor,
            fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH,
        )
        after = os.fstat(descriptor)
        linked = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        _require_destination_lock_stat(after, field)
        _require_destination_lock_stat(linked, field)
        identity = _full_file_identity(after)
        if identity != _full_file_identity(linked):
            raise ValueError(f"{field} changed while its lock was acquired")
        return _DestinationLock(name=name, descriptor=descriptor, identity=identity)
    except BaseException:
        try:
            if creation_guard is not None:
                try:
                    fcntl.flock(creation_guard, fcntl.LOCK_UN)
                finally:
                    os.close(creation_guard)
        finally:
            os.close(descriptor)
        raise


def _assert_destination_lock_at(
    lock: _DestinationLock,
    parent_fd: int,
    field: str,
) -> None:
    opened = os.fstat(lock.descriptor)
    linked = os.stat(lock.name, dir_fd=parent_fd, follow_symlinks=False)
    _require_destination_lock_stat(opened, field)
    _require_destination_lock_stat(linked, field)
    if (
        _full_file_identity(opened) != lock.identity
        or _full_file_identity(linked) != lock.identity
    ):
        raise ValueError(f"{field} identity changed while held")


def _release_destination_lock(lock: _DestinationLock) -> None:
    try:
        fcntl.flock(lock.descriptor, fcntl.LOCK_UN)
    finally:
        os.close(lock.descriptor)


def _open_private_directory_at(
    directory_fd: int,
    name: str,
    field: str,
    *,
    expected_identity: tuple[int, int] | None = None,
) -> tuple[int, os.stat_result]:
    _safe_entry_name(name)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise ValueError(f"{field} is unavailable or unsafe") from exc
    try:
        before = os.fstat(descriptor)
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(before.st_mode)
            or _inode_identity(before) != _inode_identity(current)
            or (expected_identity is not None and _inode_identity(before) != expected_identity)
        ):
            raise ValueError(f"{field} identity is unsafe")
        return descriptor, before
    except Exception:
        os.close(descriptor)
        raise


def _mkdir_private_directory_at(
    directory_fd: int,
    name: str,
    field: str,
) -> tuple[int, os.stat_result]:
    _safe_entry_name(name)
    descriptor: int | None = None
    created = False
    try:
        os.mkdir(name, 0o700, dir_fd=directory_fd)
        created = True
        initial = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISDIR(initial.st_mode) or initial.st_uid != os.geteuid():
            raise ValueError(f"{field} was not created as owner-only storage")
        created_identity = _inode_identity(initial)
        descriptor, _ = _open_private_directory_at(
            directory_fd,
            name,
            field,
            expected_identity=created_identity,
        )
        os.fchmod(descriptor, 0o700)
        after = os.fstat(descriptor)
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            stat.S_IMODE(after.st_mode) != 0o700
            or after.st_uid != os.geteuid()
            or _inode_identity(after) != created_identity
            or _inode_identity(after) != _inode_identity(current)
        ):
            raise ValueError(f"{field} was not created as owner-only storage")
        result = descriptor, after
        descriptor = None
        return result
    except BaseException as primary:
        retained_cause = primary.__cause__ if (
            isinstance(primary, ValueError)
            and isinstance(primary.__cause__, OSError)
        ) else primary
        if descriptor is not None:
            try:
                os.close(descriptor)
            except BaseException as close_error:
                raise RuntimeError(
                    f"{field} is retained as indeterminate private storage"
                ) from close_error
        if created:
            raise RuntimeError(
                f"{field} is retained as indeterminate private storage"
            ) from retained_cause
        raise primary


def _write_private_file_at(
    directory_fd: int,
    name: str,
    payload: bytes,
    *,
    ledger: _OwnedTreeLedger | None = None,
    relative: str | None = None,
) -> os.stat_result:
    _safe_entry_name(name)
    if (ledger is None) != (relative is None):
        raise ValueError("private writer ledger registration is incomplete")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        try:
            created = os.fstat(descriptor)
        except BaseException as primary:
            try:
                recovered = os.fstat(descriptor)
                if ledger is not None and relative is not None:
                    ledger.record(relative, "file", recovered)
            except BaseException as recovery_error:
                raise primary from recovery_error
            raise primary
        if ledger is not None and relative is not None:
            ledger.record(relative, "file", created)
        if (
            not stat.S_ISREG(created.st_mode)
            or created.st_uid != os.geteuid()
            or created.st_nlink != 1
        ):
            raise ValueError("new private file storage is unsafe")
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("private bridge write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        before = os.fstat(descriptor)
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or _full_file_identity(before) != _full_file_identity(current)
        ):
            raise ValueError("private file identity changed during publication")
        return before
    finally:
        os.close(descriptor)


def _fsync_directory_fd(descriptor: int) -> None:
    info = os.fstat(descriptor)
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError("directory durability descriptor is not a directory")
    os.fsync(descriptor)
    after = os.fstat(descriptor)
    if _inode_identity(after) != _inode_identity(info):
        raise ValueError("directory identity changed during durability sync")


def _read_private_file_at(directory_fd: int, name: str, field: str) -> bytes:
    _safe_entry_name(name)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
        ):
            raise ValueError(f"{field} must be owner-only regular storage")
        payload = bytearray()
        while chunk := os.read(descriptor, 1024 * 1024):
            payload.extend(chunk)
            if len(payload) > _MAX_BUNDLE_BYTES:
                raise ValueError(f"{field} exceeds the private size bound")
        after = os.fstat(descriptor)
        path_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            _full_file_identity(before) != _full_file_identity(after)
            or _full_file_identity(after) != _full_file_identity(path_stat)
        ):
            raise ValueError(f"{field} changed while it was read")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _write_private_file(path: Path, payload: bytes) -> None:
    parent = _open_directory_anchor(path.parent, "private file parent")
    try:
        _assert_directory_anchor(parent, "private file parent")
        _write_private_file_at(parent.descriptor, path.name, payload)
        _assert_directory_anchor(parent, "private file parent")
    finally:
        _close_anchor(parent)


def _fsync_directory(path: Path) -> None:
    anchor = _open_directory_anchor(path, "private directory")
    try:
        _fsync_directory_fd(anchor.descriptor)
        _assert_directory_anchor(anchor, "private directory")
    finally:
        _close_anchor(anchor)


def _read_private_file(path: Path, field: str) -> bytes:
    parent = _open_directory_anchor(path.parent, f"{field} parent")
    try:
        _assert_directory_anchor(parent, f"{field} parent")
        result = _read_private_file_at(parent.descriptor, path.name, field)
        _assert_directory_anchor(parent, f"{field} parent")
        return result
    finally:
        _close_anchor(parent)


def _decode_unique_json(payload: bytes) -> dict[str, object]:
    def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("bridge JSON repeats a field")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("ascii"), object_pairs_hook=unique_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("bridge generation JSON is invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("bridge generation JSON must be an object")
    return value


@dataclass(frozen=True)
class _ExactPrivateTreeSnapshot:
    directories: tuple[
        tuple[str, tuple[int, ...], tuple[str, ...]], ...
    ]
    files: tuple[tuple[str, tuple[int, ...], str], ...]


class _ExactTreeChanged(ValueError):
    """The held private tree no longer matches its validated snapshot."""


def _snapshot_exact_private_tree_fd(root_fd: int) -> _ExactPrivateTreeSnapshot:
    directories: list[tuple[str, tuple[int, ...], tuple[str, ...]]] = []
    files: list[tuple[str, tuple[int, ...], str]] = []
    entry_count = 0
    aggregate_bytes = 0

    def walk(directory_fd: int, relative: str, depth: int) -> None:
        nonlocal entry_count, aggregate_bytes
        if depth > _MAX_EXACT_TREE_DEPTH:
            raise ValueError("exact private tree exceeds recursion depth bound")
        before = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(before.st_mode)
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o700
        ):
            raise ValueError(
                "exact private tree directory must be owner-only mode 0700"
            )
        collected_names: list[str] = []
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                entry_count += 1
                if entry_count > _MAX_EXACT_TREE_ENTRIES:
                    raise ValueError("exact private tree exceeds entry count bound")
                collected_names.append(entry.name)
        names = tuple(sorted(collected_names))
        if len(names) != len(set(names)):
            raise ValueError("exact private tree repeats a directory entry")
        for name in names:
            _safe_entry_name(name)
        directories.append((relative, _full_file_identity(before), names))
        for name in names:
            child_relative = f"{relative}/{name}" if relative else name
            observed = os.stat(
                name, dir_fd=directory_fd, follow_symlinks=False,
            )
            if stat.S_ISDIR(observed.st_mode):
                if depth >= _MAX_EXACT_TREE_DEPTH:
                    raise ValueError("exact private tree exceeds recursion depth bound")
                child_fd: int | None = None
                try:
                    child_fd, _ = _open_private_directory_at(
                        directory_fd,
                        name,
                        "exact private tree directory",
                        expected_identity=_inode_identity(observed),
                    )
                    walk(child_fd, child_relative, depth + 1)
                finally:
                    if child_fd is not None:
                        os.close(child_fd)
            elif stat.S_ISREG(observed.st_mode):
                descriptor: int | None = None
                try:
                    descriptor = os.open(
                        name,
                        os.O_RDONLY
                        | getattr(os, "O_NOFOLLOW", 0)
                        | getattr(os, "O_CLOEXEC", 0),
                        dir_fd=directory_fd,
                    )
                    opened = os.fstat(descriptor)
                    if (
                        not stat.S_ISREG(opened.st_mode)
                        or stat.S_IMODE(opened.st_mode) != 0o600
                        or opened.st_uid != os.geteuid()
                        or opened.st_nlink != 1
                        or _full_file_identity(opened)
                        != _full_file_identity(observed)
                    ):
                        raise ValueError("exact private tree file identity is unsafe")
                    if (
                        opened.st_size < 0
                        or opened.st_size > _MAX_BUNDLE_BYTES
                        or aggregate_bytes + opened.st_size
                        > _MAX_EXACT_TREE_TOTAL_BYTES
                    ):
                        raise ValueError("exact private tree exceeds aggregate size bound")
                    digest = hashlib.sha256()
                    total = 0
                    while chunk := os.read(descriptor, 1024 * 1024):
                        total += len(chunk)
                        aggregate_bytes += len(chunk)
                        if total > _MAX_BUNDLE_BYTES:
                            raise ValueError("exact private tree file exceeds size bound")
                        if aggregate_bytes > _MAX_EXACT_TREE_TOTAL_BYTES:
                            raise ValueError(
                                "exact private tree exceeds aggregate size bound"
                            )
                        digest.update(chunk)
                    after = os.fstat(descriptor)
                    current = os.stat(
                        name, dir_fd=directory_fd, follow_symlinks=False,
                    )
                    if (
                        total != opened.st_size
                        or _full_file_identity(opened) != _full_file_identity(after)
                        or _full_file_identity(after) != _full_file_identity(current)
                    ):
                        raise ValueError("exact private tree file changed during snapshot")
                    files.append((
                        child_relative,
                        _full_file_identity(after),
                        digest.hexdigest(),
                    ))
                finally:
                    if descriptor is not None:
                        os.close(descriptor)
            else:
                raise ValueError("exact private tree contains an unsafe entry type")
        after = os.fstat(directory_fd)
        if _full_file_identity(after) != _full_file_identity(before):
            raise ValueError("exact private directory changed during snapshot")

    walk(root_fd, "", 0)
    return _ExactPrivateTreeSnapshot(
        directories=tuple(sorted(directories)),
        files=tuple(sorted(files)),
    )


def _require_exact_tree_unchanged(
    root_fd: int,
    before: _ExactPrivateTreeSnapshot,
    field: str,
) -> None:
    after = _snapshot_exact_private_tree_fd(root_fd)
    if after != before:
        raise _ExactTreeChanged(f"{field} exact tree changed during validation")


def _validate_bundle_payload(
    payload: bytes,
    *,
    stage: str,
    record: Mapping[str, object],
) -> None:
    width = 23 if stage == "ravdess" else 95 if stage == "mayo" else 0
    sample_count = _exact_integer(record.get("sample_count"), f"{stage} sample count")
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as cached:
            if len(cached.files) != len(set(cached.files)) or set(cached.files) != _BUNDLE_FIELDS:
                raise ValueError("bridge bundle field schema is not exact")
            features = np.asarray(cached["features"])
            valid = np.asarray(cached["valid_mask"])
            timestamps = np.asarray(cached["timestamps"])
            indices = np.asarray(cached["source_frame_indices"])
            groups = np.asarray(cached["group_ids"])
    except (OSError, EOFError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("bridge bundle"):
            raise
        raise ValueError("bridge bundle is not a safe exact NPZ") from exc
    leading = (sample_count, 4, 32)
    if features.dtype != np.float32 or features.shape != leading + (width,):
        raise ValueError("bridge bundle feature tensor is noncanonical")
    if valid.dtype != np.bool_ or valid.shape != leading:
        raise ValueError("bridge bundle validity tensor is noncanonical")
    if timestamps.dtype != np.float32 or timestamps.shape != leading:
        raise ValueError("bridge bundle timestamp tensor is noncanonical")
    if indices.dtype != np.int64 or indices.shape != leading:
        raise ValueError("bridge bundle index tensor is noncanonical")
    if groups.dtype.kind != "U" or groups.shape != (sample_count,):
        raise ValueError("bridge bundle group tensor is noncanonical")
    expected_t = np.arange(32, dtype=np.float32) / np.float32(30.0)
    expected_i = np.arange(32, dtype=np.int64)
    if not np.array_equal(timestamps, np.broadcast_to(expected_t, leading)):
        raise ValueError("bridge bundle timestamps are not exact local 30-Hz axes")
    if not np.array_equal(indices, np.broadcast_to(expected_i, leading)):
        raise ValueError("bridge bundle indices are not exact local canonical axes")
    if not np.isfinite(features).all() or np.any(features[~valid] != np.float32(0.0)):
        raise ValueError("bridge bundle invalid rows are not finite canonical zero")
    if groups.tolist() != list(record.get("group_ids", [])):
        raise ValueError("bridge bundle group order disagrees with its closure")
    if hashlib.sha256(payload).hexdigest() != record.get("bundle_sha256"):
        raise ValueError("bridge bundle digest disagrees with its closure")
    if len(payload) != record.get("bundle_size_bytes"):
        raise ValueError("bridge bundle size disagrees with its closure")


def _validate_generation_fd(
    root_fd: int,
    expected: _PreparedBridgeGeneration,
) -> tuple[int, int, bool, int, bool]:
    root_stat = os.fstat(root_fd)
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("bridge generation root is unsafe")
    tree_before = _snapshot_exact_private_tree_fd(root_fd)
    if set(os.listdir(root_fd)) != {"bundles", "bundle_generation.json"}:
        raise ValueError("bridge generation top-level file set is not exact")
    bundles_fd, _bundles_stat = _open_private_directory_at(
        root_fd, "bundles", "bridge bundle directory",
    )
    try:
        if set(os.listdir(bundles_fd)) != {
            "ravdess_bundle.npz", "mayo_bundle.npz",
        }:
            raise ValueError("bridge bundle file set is not exact")
        generation_bytes = _read_private_file_at(
            root_fd, "bundle_generation.json", "bridge generation closure",
        )
        _assert_private_metadata(generation_bytes)
        generation = _decode_unique_json(generation_bytes)
        if generation != expected.generation or generation_bytes != expected.generation_bytes:
            raise ValueError("bridge generation closure changed or is nondeterministic")
        total_bytes = 0
        for stage, prepared in (("ravdess", expected.ravdess), ("mayo", expected.mayo)):
            payload = _read_private_file_at(
                bundles_fd, f"{stage}_bundle.npz", f"{stage} bridge bundle",
            )
            if payload != prepared.bundle_bytes:
                raise ValueError(f"{stage} bridge bundle changed or is nondeterministic")
            _validate_bundle_payload(payload, stage=stage, record=prepared.record)
            total_bytes += len(payload)
        _require_exact_tree_unchanged(
            root_fd, tree_before, "bridge generation",
        )
        non_0600 = sum(
            stat.S_IMODE(identity[2]) != 0o600
            for _relative, identity, _digest in tree_before.files
        )
        directories_ok = all(
            identity[3] == os.geteuid()
            and stat.S_IMODE(identity[2]) == 0o700
            for _relative, identity, _names in tree_before.directories
        )
        files_private = all(
            identity[3] == os.geteuid() and identity[5] == 1
            for _relative, identity, _digest in tree_before.files
        )
        modes_ok = directories_ok and non_0600 == 0
        privacy_ok = modes_ok and files_private
        return 2, total_bytes, modes_ok, non_0600, privacy_ok
    finally:
        os.close(bundles_fd)


def _validate_generation_tree(
    root: Path,
    expected: _PreparedBridgeGeneration,
) -> tuple[int, int, bool, int, bool]:
    anchor = _open_directory_anchor(root, "bridge generation root")
    try:
        result = _validate_generation_fd(anchor.descriptor, expected)
        _assert_directory_anchor(anchor, "bridge generation root")
        return result
    finally:
        _close_anchor(anchor)


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if type(value) is not str or not value:
            raise ValueError("bridge identity order contains an invalid value")
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _split_indices(stage: _PreparedBridgeStage) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    groups = _ordered_unique(stage.group_ids)
    if len(groups) < 2:
        raise ValueError("frozen SSL split requires at least two source groups")
    permutation = np.random.default_rng(0).permutation(len(groups))
    heldout_count = min(len(groups) - 1, max(1, math.ceil(len(groups) * 0.20)))
    heldout_set = {groups[int(index)] for index in permutation[:heldout_count]}
    train_groups = tuple(group for group in groups if group not in heldout_set)
    heldout_groups = tuple(group for group in groups if group in heldout_set)
    train = np.asarray(
        [index for index, group in enumerate(stage.group_ids) if group not in heldout_set],
        dtype=np.int64,
    )
    heldout = np.asarray(
        [index for index, group in enumerate(stage.group_ids) if group in heldout_set],
        dtype=np.int64,
    )
    if (
        len(train) == 0
        or len(heldout) == 0
        or set(train.tolist()) & set(heldout.tolist())
        or sorted((*train.tolist(), *heldout.tolist())) != list(range(len(stage.group_ids)))
    ):
        raise RuntimeError("frozen group split violated exact coverage")
    return train, heldout, train_groups, heldout_groups


def _load_prepared_bundle(stage: _PreparedBridgeStage) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(io.BytesIO(stage.bundle_bytes), allow_pickle=False) as cached:
            features = np.asarray(cached["features"]).copy()
            valid = np.asarray(cached["valid_mask"]).copy()
    except (OSError, EOFError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("prepared bridge bundle cannot be read for freezing") from exc
    _validate_bundle_payload(stage.bundle_bytes, stage=stage.name, record=stage.record)
    return features, valid


def _unique_train_scaler(
    stage: _PreparedBridgeStage,
    train_indices: np.ndarray,
) -> tuple[list[float], list[float], int, tuple[str, ...]]:
    features, valid = _load_prepared_bundle(stage)
    if len(stage.original_canonical_frame_indices) != len(stage.source_unit_ids):
        raise ValueError("private canonical mappings do not align with source units")
    seen: set[tuple[str, int]] = set()
    observations: list[np.ndarray] = []
    fit_sources: list[str] = []
    fit_source_set: set[str] = set()
    for sample_index in train_indices.tolist():
        source_unit = stage.source_unit_ids[sample_index]
        if source_unit not in fit_source_set:
            fit_source_set.add(source_unit)
            fit_sources.append(source_unit)
        canonical = stage.original_canonical_frame_indices[sample_index]
        for window_index in range(4):
            for frame_index in range(32):
                if not bool(valid[sample_index, window_index, frame_index]):
                    continue
                key = (source_unit, int(canonical[window_index][frame_index]))
                if key in seen:
                    continue
                seen.add(key)
                observations.append(
                    features[sample_index, window_index, frame_index].astype(
                        np.float64, copy=True
                    )
                )
    if not observations:
        raise ValueError("train-only unique-frame scaler has no valid observations")
    values = np.stack(observations, axis=0)
    mean = values.mean(axis=0, dtype=np.float64)
    scale = values.std(axis=0, dtype=np.float64)
    scale[scale < np.finfo(np.float32).eps] = 1.0
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("train-only unique-frame scaler is nonfinite")
    return mean.tolist(), scale.tolist(), len(seen), tuple(fit_sources)


def _artifact_core_sha256(value: Mapping[str, object]) -> str:
    if "bridge_receipt_sha256" in value or "receipt_hmac" in value:
        raise ValueError("artifact core cannot contain receipt cross-links")
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _prepare_frozen_stage(
    stage: _PreparedBridgeStage,
    *,
    mode: str,
    bridge_generation_sha256: str,
) -> _PreparedFrozenStage:
    if mode not in {"smoke", "formal"} or type(mode) is not str:
        raise ValueError("bridge freeze mode must be exactly smoke or formal")
    train, heldout, train_groups, heldout_groups = _split_indices(stage)
    mean, scale, unique_frames, fit_sources = _unique_train_scaler(stage, train)
    source = (
        "ravdess_openface_semantic23"
        if stage.name == "ravdess"
        else "mayo_mediapipe_clinical23_development_only"
    )
    manifest: dict[str, object] = {
        "schema_version": _SSL_MANIFEST_SCHEMA,
        "stage": stage.name,
        "mode": mode,
        "source": source,
        "source_schema": stage.record["source_schema"],
        "sample_ids": list(stage.sample_ids),
        "source_unit_ids": list(stage.source_unit_ids),
        "group_ids": list(stage.group_ids),
        "sample_count": stage.record["sample_count"],
        "source_unit_count": stage.record["source_unit_count"],
        "unique_group_count": stage.record["unique_group_count"],
        "upstream_cache_count": stage.record["upstream_cache_count"],
        "bundle_file_count": 1,
        "bundle_sha256": stage.record["bundle_sha256"],
        "bundle_size_bytes": stage.record["bundle_size_bytes"],
        "feature_names_sha256": stage.record["feature_names_sha256"],
        "adapter_sha256": stage.record["adapter_sha256"],
        "temporal_policy_sha256": _json_sha256(
            stage.record["packet_policy"]  # type: ignore[arg-type]
        ),
        "bridge_generation_sha256": bridge_generation_sha256,
        "upstream_manifest_commitments": stage.record[
            "upstream_manifest_commitments"
        ],
        "upstream_generation_closure_hmac": stage.record[
            "upstream_generation_closure_hmac"
        ],
    }
    config: dict[str, object] = {
        "schema_version": _SSL_CONFIG_SCHEMA,
        "stage": stage.name,
        "mode": mode,
        "source": source,
        "objective": "masked_span_smooth_l1_only",
        "sample_rate_hz": 30.0,
        "seeds": [0] if mode == "smoke" else [0, 1, 2],
        "development_only": stage.name == "mayo",
        "optimizer": "adamw",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "epochs": 1 if mode == "smoke" else 30,
        "batch_policy": "full_train_partition",
        "span_length": 4,
        "spans_per_window": 2,
        "device": "cpu",
    }
    split: dict[str, object] = {
        "schema_version": _SSL_SPLIT_SCHEMA,
        "stage": stage.name,
        "mode": mode,
        "source": source,
        "split_seed": 0,
        "heldout_fraction": 0.20,
        "unit": "actor" if stage.name == "ravdess" else "recording",
        "claim_unit": (
            "actor_held_out" if stage.name == "ravdess"
            else "recording_held_out_not_patient_held_out"
        ),
        "train_group_ids": list(train_groups),
        "heldout_group_ids": list(heldout_groups),
        "train_indices": train.tolist(),
        "heldout_indices": heldout.tolist(),
    }
    scaler: dict[str, object] = {
        "schema_version": _SSL_SCALER_SCHEMA,
        "stage": stage.name,
        "mode": mode,
        "source": source,
        "fit_indices": train.tolist(),
        "fit_source_unit_ids": list(fit_sources),
        "unique_frame_key": "source_unit_id_plus_original_canonical_30hz_index",
        "fit_unique_frame_count": unique_frames,
        "mean": mean,
        "scale": scale,
    }
    cores = {
        "manifest": manifest,
        "config": config,
        "split": split,
        "scaler": scaler,
    }
    core_digests = {
        name: _artifact_core_sha256(value) for name, value in cores.items()
    }
    receipt: dict[str, object] = {
        key: value for key, value in stage.record.items()
        if key not in {"schema", "closure_hmac"}
    }
    receipt.update({
        "schema": _BRIDGE_RECEIPT_SCHEMA,
        "mode": mode,
        "bridge_stage_closure_hmac": stage.record["closure_hmac"],
        "bridge_generation_sha256": bridge_generation_sha256,
        "artifact_core_sha256": core_digests,
        "canonical_key_identity_sha256": stage.canonical_key_identity_sha256,
        "original_canonical_frame_indices": [
            [[int(item) for item in window] for window in packet]
            for packet in stage.original_canonical_frame_indices
        ],
        "original_source_frame_indices": [
            [[int(item) for item in window] for window in packet]
            for packet in stage.original_source_frame_indices
        ],
        "original_timestamps": [
            [[float(item) for item in window] for window in packet]
            for packet in stage.original_timestamps
        ],
    })
    receipt["receipt_hmac"] = hmac.new(
        stage.private_key,
        b"dynamic-landmark-bridge-receipt-v1\0" + _json_bytes(receipt),
        hashlib.sha256,
    ).hexdigest()
    receipt_bytes = _json_bytes(receipt)
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    artifacts: dict[str, bytes] = {}
    for name, core in cores.items():
        linked = {
            **core,
            "bridge_receipt_sha256": receipt_sha256,
            "receipt_hmac": receipt["receipt_hmac"],
        }
        artifacts[name] = _json_bytes(linked)
    return _PreparedFrozenStage(
        name=stage.name,
        artifacts=artifacts,
        receipt=receipt,
        receipt_bytes=receipt_bytes,
    )


def _prepare_frozen_inputs(
    generation: _PreparedBridgeGeneration,
    *,
    mode: str,
) -> _PreparedFrozenInputs:
    generation_sha256 = hashlib.sha256(generation.generation_bytes).hexdigest()
    return _PreparedFrozenInputs(
        mode=mode,
        ravdess=_prepare_frozen_stage(
            generation.ravdess,
            mode=mode,
            bridge_generation_sha256=generation_sha256,
        ),
        mayo=_prepare_frozen_stage(
            generation.mayo,
            mode=mode,
            bridge_generation_sha256=generation_sha256,
        ),
    )


def _frozen_inputs_equal(
    first: _PreparedFrozenInputs,
    second: _PreparedFrozenInputs,
) -> bool:
    if first.mode != second.mode:
        return False
    for name in ("ravdess", "mayo"):
        left = getattr(first, name)
        right = getattr(second, name)
        if not hmac.compare_digest(left.receipt_bytes, right.receipt_bytes):
            return False
        if set(left.artifacts) != set(right.artifacts):
            return False
        if any(
            not hmac.compare_digest(left.artifacts[key], right.artifacts[key])
            for key in left.artifacts
        ):
            return False
    return True


def _assert_frozen_json_private(payload: bytes) -> None:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("frozen bridge input must be ASCII JSON") from exc
    if "/" in text or "\\" in text:
        raise ValueError("frozen bridge input contains a path-like value")
    if re.search(r"(?:faces|myslate)[_ ]*\d+", text, re.IGNORECASE):
        raise ValueError("frozen bridge input contains a Mayo source identifier")
    lowered = text.lower()
    if "source_sha256" in lowered or "private_key" in lowered:
        raise ValueError("frozen bridge input contains raw provenance or key material")


def _validate_frozen_inputs_fd(
    root_fd: int,
    expected: _PreparedFrozenInputs,
) -> None:
    if not stat.S_ISDIR(os.fstat(root_fd).st_mode):
        raise ValueError("frozen inputs root is unsafe")
    tree_before = _snapshot_exact_private_tree_fd(root_fd)
    if set(os.listdir(root_fd)) != {"receipts", "artifacts"}:
        raise ValueError("frozen inputs top-level schema is not exact")
    descriptors = ExitStack()
    try:
        receipts_fd, _ = _open_private_directory_at(
            root_fd, "receipts", "frozen receipt directory",
        )
        descriptors.callback(os.close, receipts_fd)
        artifacts_fd, _ = _open_private_directory_at(
            root_fd, "artifacts", "frozen artifact directory",
        )
        descriptors.callback(os.close, artifacts_fd)
        if set(os.listdir(receipts_fd)) != {"ravdess.json", "mayo.json"}:
            raise ValueError("frozen receipt file set is not exact")
        if set(os.listdir(artifacts_fd)) != {"ravdess", "mayo"}:
            raise ValueError("frozen artifact stage set is not exact")
        for stage_name in ("ravdess", "mayo"):
            prepared = getattr(expected, stage_name)
            receipt_bytes = _read_private_file_at(
                receipts_fd, f"{stage_name}.json", f"{stage_name} bridge receipt",
            )
            _assert_frozen_json_private(receipt_bytes)
            if receipt_bytes != prepared.receipt_bytes:
                raise ValueError(f"{stage_name} bridge receipt changed")
            stage_fd, _ = _open_private_directory_at(
                artifacts_fd, stage_name, f"{stage_name} artifact directory",
            )
            try:
                expected_names = {f"{name}.json" for name in prepared.artifacts}
                if set(os.listdir(stage_fd)) != expected_names:
                    raise ValueError(f"{stage_name} artifact file set is not exact")
                for name, expected_bytes in prepared.artifacts.items():
                    payload = _read_private_file_at(
                        stage_fd, f"{name}.json", f"{stage_name} {name} artifact",
                    )
                    _assert_frozen_json_private(payload)
                    if payload != expected_bytes:
                        raise ValueError(f"{stage_name} {name} artifact changed")
            finally:
                os.close(stage_fd)
        _require_exact_tree_unchanged(
            root_fd, tree_before, "frozen inputs",
        )
    finally:
        descriptors.__exit__(*sys.exc_info())


def _validate_frozen_inputs_tree(
    root: Path,
    expected: _PreparedFrozenInputs,
) -> None:
    anchor = _open_directory_anchor(root, "frozen inputs root")
    try:
        _validate_frozen_inputs_fd(anchor.descriptor, expected)
        _assert_directory_anchor(anchor, "frozen inputs root")
    finally:
        _close_anchor(anchor)


def _write_frozen_inputs_fd(
    staging_fd: int,
    prepared: _PreparedFrozenInputs,
    ledger: _OwnedTreeLedger,
) -> None:
    descriptors = ExitStack()
    try:
        receipts_fd, receipts_stat = _mkdir_private_directory_at(
            staging_fd, "receipts", "frozen receipt directory",
        )
        descriptors.callback(os.close, receipts_fd)
        ledger.record("receipts", "directory", receipts_stat)
        artifacts_fd, artifacts_stat = _mkdir_private_directory_at(
            staging_fd, "artifacts", "frozen artifact directory",
        )
        descriptors.callback(os.close, artifacts_fd)
        ledger.record("artifacts", "directory", artifacts_stat)
        for stage_name in ("ravdess", "mayo"):
            stage = getattr(prepared, stage_name)
            stage_fd, stage_stat = _mkdir_private_directory_at(
                artifacts_fd, stage_name, f"{stage_name} artifact directory",
            )
            ledger.record(f"artifacts/{stage_name}", "directory", stage_stat)
            try:
                receipt_name = f"{stage_name}.json"
                _write_private_file_at(
                    receipts_fd,
                    receipt_name,
                    stage.receipt_bytes,
                    ledger=ledger,
                    relative=f"receipts/{receipt_name}",
                )
                for name, payload in stage.artifacts.items():
                    filename = f"{name}.json"
                    _write_private_file_at(
                        stage_fd,
                        filename,
                        payload,
                        ledger=ledger,
                        relative=f"artifacts/{stage_name}/{filename}",
                    )
                _fsync_directory_fd(stage_fd)
            finally:
                os.close(stage_fd)
        _fsync_directory_fd(receipts_fd)
        _fsync_directory_fd(artifacts_fd)
        _fsync_directory_fd(staging_fd)
    finally:
        descriptors.__exit__(*sys.exc_info())


def _write_frozen_inputs_tree(
    staging: Path,
    prepared: _PreparedFrozenInputs,
) -> None:
    anchor = _open_directory_anchor(staging, "frozen inputs staging")
    ledger = _OwnedTreeLedger.create(os.fstat(anchor.descriptor))
    try:
        _write_frozen_inputs_fd(anchor.descriptor, prepared, ledger)
        _assert_directory_anchor(anchor, "frozen inputs staging")
    finally:
        _close_anchor(anchor)


def _atomic_publish_directory_no_replace_at(
    parent_fd: int,
    staging_name: str,
    output_name: str,
) -> None:
    _safe_entry_name(staging_name)
    _safe_entry_name(output_name)
    _require_entry_absent_at(parent_fd, output_name, "committed private generation")
    libc = ctypes.CDLL(None, use_errno=True)
    old = os.fsencode(staging_name)
    new = os.fsencode(output_name)
    result: int
    ctypes.set_errno(0)
    if sys.platform == "darwin" and hasattr(libc, "renameatx_np"):
        rename = libc.renameatx_np
        rename.argtypes = [
            ctypes.c_int, ctypes.c_char_p,
            ctypes.c_int, ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = int(rename(
            parent_fd, old, parent_fd, new, 0x00000004 | 0x00000010,
        ))  # RENAME_EXCL | RENAME_NOFOLLOW_ANY
    elif hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = int(rename(parent_fd, old, parent_fd, new, 1))
    else:
        raise RuntimeError("atomic no-replace directory publication is unavailable")
    if result != 0:
        code = ctypes.get_errno()
        if code in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                f"committed bridge generation already exists: {output_name}"
            )
        raise OSError(code, os.strerror(code), output_name)


def _atomic_publish_directory_no_replace(staging: Path, output: Path) -> None:
    if staging.parent != output.parent:
        raise ValueError("private publication must use one anchored parent")
    parent = _open_directory_anchor(staging.parent, "private publication parent")
    try:
        _assert_directory_anchor(parent, "private publication parent")
        _atomic_publish_directory_no_replace_at(
            parent.descriptor, staging.name, output.name,
        )
        _fsync_directory_fd(parent.descriptor)
        _assert_directory_anchor(parent, "private publication parent")
    finally:
        _close_anchor(parent)


def _classify_owned_publication(
    parent_fd: int,
    staging_name: str,
    output_name: str,
    identity: tuple[int, int],
) -> str:
    source = _entry_stat_at(parent_fd, staging_name)
    destination = _entry_stat_at(parent_fd, output_name)
    if destination is not None and _inode_identity(destination) == identity:
        return "published"
    if source is not None and _inode_identity(source) == identity:
        return "staged"
    if source is None and destination is None:
        return "absent"
    return "indeterminate"


def _prepared_equal(
    first: _PreparedBridgeGeneration,
    second: _PreparedBridgeGeneration,
) -> bool:
    return (
        hmac.compare_digest(first.ravdess.bundle_bytes, second.ravdess.bundle_bytes)
        and hmac.compare_digest(first.mayo.bundle_bytes, second.mayo.bundle_bytes)
        and hmac.compare_digest(first.generation_bytes, second.generation_bytes)
    )


@dataclass(frozen=True)
class _PublishedBridge:
    prepared: _PreparedBridgeGeneration
    output_name: str
    identity: tuple[int, int]
    ledger: _OwnedTreeLedger = dataclass_field(repr=False)


def _prepare_live_generation(
    ravdess_authorizer: Callable[[], object],
    mayo_authorizer: Callable[[], object],
    *,
    producer_sha256: str,
    anchors: Sequence[tuple[_DirectoryAnchor, str]],
) -> _PreparedBridgeGeneration:
    for anchor, field in anchors:
        _assert_directory_anchor(anchor, field)
    ravdess = ravdess_authorizer()
    for anchor, field in anchors:
        _assert_directory_anchor(anchor, field)
    mayo = mayo_authorizer()
    for anchor, field in anchors:
        _assert_directory_anchor(anchor, field)
    _require_frozen_public_authorizations(ravdess, mayo)
    for anchor, field in anchors:
        _assert_directory_anchor(anchor, field)
    prepared = _prepare_bridge_generation(
        ravdess, mayo, producer_sha256=producer_sha256,
    )
    if (
        prepared.ravdess.record.get("sample_count")
        != _FROZEN_RAVDESS_SAMPLE_COUNT
        or prepared.mayo.record.get("sample_count")
        != _FROZEN_MAYO_SAMPLE_COUNT
    ):
        raise ValueError("bridge samples violate the frozen production counts")
    for anchor, field in anchors:
        _assert_directory_anchor(anchor, field)
    return prepared


def _write_generation_fd(
    staging_fd: int,
    prepared: _PreparedBridgeGeneration,
    ledger: _OwnedTreeLedger,
) -> None:
    bundles_fd, bundles_stat = _mkdir_private_directory_at(
        staging_fd, "bundles", "bridge bundle staging directory",
    )
    ledger.record("bundles", "directory", bundles_stat)
    try:
        for stage, payload in (
            ("ravdess", prepared.ravdess.bundle_bytes),
            ("mayo", prepared.mayo.bundle_bytes),
        ):
            filename = f"{stage}_bundle.npz"
            _write_private_file_at(
                bundles_fd,
                filename,
                payload,
                ledger=ledger,
                relative=f"bundles/{filename}",
            )
        _write_private_file_at(
            staging_fd,
            "bundle_generation.json",
            prepared.generation_bytes,
            ledger=ledger,
            relative="bundle_generation.json",
        )
        _fsync_directory_fd(bundles_fd)
        _fsync_directory_fd(staging_fd)
    finally:
        os.close(bundles_fd)


def _linear_cleanup_cause(
    primary: BaseException | None,
    cleanup_errors: Sequence[BaseException],
) -> BaseException | None:
    """Preserve nested cleanup chains and append each one without a cycle."""
    cause = primary
    seen: set[int] = set()
    current = primary
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    for error in cleanup_errors:
        if id(error) in seen:
            continue
        unique_nodes: list[BaseException] = []
        local_seen: set[int] = set()
        current = error
        while (
            current is not None
            and id(current) not in seen
            and id(current) not in local_seen
        ):
            local_seen.add(id(current))
            unique_nodes.append(current)
            current = current.__cause__ or current.__context__
        for node in reversed(unique_nodes):
            node.__cause__ = cause
            node.__context__ = None
            node.__suppress_context__ = True
            cause = node
            seen.add(id(node))
    return cause


def _attach_cleanup_causes(
    outcome: BaseException,
    cleanup_errors: Sequence[BaseException],
) -> BaseException:
    existing = outcome.__cause__ or outcome.__context__
    outcome.__cause__ = _linear_cleanup_cause(existing, cleanup_errors)
    outcome.__context__ = None
    outcome.__suppress_context__ = outcome.__cause__ is not None
    return outcome


def _build_bridge_bundles_at(
    parent: _DirectoryAnchor,
    output_name: str,
    *,
    destination_lock: _DestinationLock,
    ravdess_authorizer: Callable[[], object],
    mayo_authorizer: Callable[[], object],
    producer_sha256: str,
) -> _PublishedBridge:
    _safe_entry_name(output_name)
    _assert_directory_anchor(parent, "bridge output parent")
    _assert_destination_lock_at(
        destination_lock, parent.descriptor, "bridge destination lock",
    )
    _require_entry_absent_at(
        parent.descriptor, output_name, "committed bridge generation",
    )
    first = _prepare_live_generation(
        ravdess_authorizer,
        mayo_authorizer,
        producer_sha256=producer_sha256,
        anchors=((parent, "bridge output parent"),),
    )
    _assert_destination_lock_at(
        destination_lock, parent.descriptor, "bridge destination lock",
    )
    _require_entry_absent_at(
        parent.descriptor, output_name, "committed bridge generation",
    )
    staging_name = f".{output_name}.staging-{uuid.uuid4().hex}"
    staging_fd: int | None = None
    staging_identity: tuple[int, int] | None = None
    ledger: _OwnedTreeLedger | None = None
    pending_error: BaseException | None = None
    try:
        staging_fd, staging_stat = _mkdir_private_directory_at(
            parent.descriptor, staging_name, "bridge staging root",
        )
        staging_identity = _inode_identity(staging_stat)
        ledger = _OwnedTreeLedger.create(staging_stat)
        _assert_directory_anchor(parent, "bridge output parent")
        _write_generation_fd(staging_fd, first, ledger)
        _assert_directory_anchor(parent, "bridge output parent")
        _validate_generation_fd(staging_fd, first)

        second = _prepare_live_generation(
            ravdess_authorizer,
            mayo_authorizer,
            producer_sha256=producer_sha256,
            anchors=((parent, "bridge output parent"),),
        )
        if not _prepared_equal(first, second):
            raise ValueError("upstream authorization changed before bridge publication")
        _validate_generation_fd(staging_fd, second)
        _require_entry_absent_at(
            parent.descriptor, output_name, "committed bridge generation",
        )
        _assert_destination_lock_at(
            destination_lock, parent.descriptor, "bridge destination lock",
        )
        _assert_directory_anchor(parent, "bridge output parent")
        _atomic_publish_directory_no_replace_at(
            parent.descriptor, staging_name, output_name,
        )
        if _classify_owned_publication(
            parent.descriptor, staging_name, output_name, ledger.root_identity,
        ) != "published":
            raise RuntimeError("bridge publication outcome is indeterminate")
        published = _PublishedBridge(
            prepared=second,
            output_name=output_name,
            identity=ledger.root_identity,
            ledger=ledger,
        )
        _fsync_directory_fd(parent.descriptor)
        _assert_directory_anchor(parent, "bridge output parent")
        _validate_generation_fd(staging_fd, second)
        destination = _entry_stat_at(parent.descriptor, output_name)
        if destination is None or _inode_identity(destination) != ledger.root_identity:
            raise RuntimeError("bridge publication name no longer binds staged storage")
        return published
    except BaseException as caught:
        pending_error = caught
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        try:
            if staging_fd is not None:
                os.close(staging_fd)
        except BaseException as caught:
            cleanup_errors.append(caught)
        retained_error: RuntimeError | None = None
        if staging_identity is not None:
            try:
                state = _classify_owned_publication(
                    parent.descriptor,
                    staging_name,
                    output_name,
                    staging_identity,
                )
            except BaseException as caught:
                cleanup_errors.append(caught)
                state = "indeterminate"
            if pending_error is not None and state == "staged":
                retained_error = RuntimeError(
                    "bridge transaction staging is retained as indeterminate"
                )
            elif state == "indeterminate":
                retained_error = RuntimeError(
                    "bridge transaction storage is retained as indeterminate"
                )
        if retained_error is not None:
            cause = _linear_cleanup_cause(pending_error, cleanup_errors)
            if cause is None:
                raise retained_error
            raise retained_error from cause
        if cleanup_errors:
            if pending_error is not None:
                outcome = _attach_cleanup_causes(
                    pending_error, cleanup_errors,
                )
                raise outcome.with_traceback(pending_error.__traceback__)
            cleanup_cause = _linear_cleanup_cause(None, cleanup_errors)
            assert cleanup_cause is not None
            raise cleanup_cause


def build_bridge_bundles(
    output_root: str | Path,
    *,
    ravdess_authorizer: Callable[[], object],
    mayo_authorizer: Callable[[], object],
    producer_sha256: str,
) -> dict[str, object]:
    """Build and atomically publish one immutable mode-neutral bridge generation."""
    if not callable(ravdess_authorizer) or not callable(mayo_authorizer):
        raise ValueError("bridge authorizers must be callable")
    output = _lexical_absolute(output_root)
    _require_safe_output_parent(output)
    parent = _open_directory_anchor(output.parent, "bridge output parent")
    destination_lock: _DestinationLock | None = None
    try:
        destination_lock = _acquire_destination_lock_at(
            parent.descriptor,
            f".{output.name}.lock",
            create=True,
            exclusive=True,
            field="bridge destination lock",
        )
        _assert_directory_anchor(parent, "bridge output parent")
        _reject_transaction_residue_at(
            parent.descriptor,
            (f".{output.name}.staging-", f".{output.name}.verify-"),
            "bridge output parent",
        )
        published = _build_bridge_bundles_at(
            parent,
            output.name,
            destination_lock=destination_lock,
            ravdess_authorizer=ravdess_authorizer,
            mayo_authorizer=mayo_authorizer,
            producer_sha256=producer_sha256,
        )
        _reject_transaction_residue_at(
            parent.descriptor,
            (f".{output.name}.staging-", f".{output.name}.verify-"),
            "bridge output parent",
        )
        _assert_destination_lock_at(
            destination_lock, parent.descriptor, "bridge destination lock",
        )
        _assert_directory_anchor(parent, "bridge output parent")
        return dict(published.prepared.generation["stages"])
    finally:
        try:
            if destination_lock is not None:
                _release_destination_lock(destination_lock)
        finally:
            _close_anchor(parent)


def _read_exact_key_at(
    parent_fd: int,
    name: str,
    *,
    expected_identity: tuple[int, int] | None = None,
    expected_bytes: bytes | None = None,
) -> bytes:
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=parent_fd,
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or before.st_size != 32
            or (
                expected_identity is not None
                and _inode_identity(before) != expected_identity
            )
        ):
            raise ValueError("canonical bridge key is not exact owner-only storage")
        payload = bytearray()
        while chunk := os.read(descriptor, 64):
            payload.extend(chunk)
            if len(payload) > 32:
                raise ValueError("canonical bridge key exceeds exactly 32 bytes")
        after = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            len(payload) != 32
            or _full_file_identity(before) != _full_file_identity(after)
            or _full_file_identity(after) != _full_file_identity(current)
            or (
                expected_identity is not None
                and _inode_identity(after) != expected_identity
            )
            or (
                expected_bytes is not None
                and not hmac.compare_digest(bytes(payload), expected_bytes)
            )
        ):
            raise ValueError("canonical bridge key changed during exact reopen")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _acquire_key_lock_at(parent_fd: int, key_name: str) -> int:
    _safe_entry_name(key_name)
    descriptor = os.dup(parent_fd)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISDIR(before.st_mode):
            raise ValueError("canonical key lock descriptor is unsafe")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        after = os.fstat(descriptor)
        current_parent = os.fstat(parent_fd)
        if (
            _inode_identity(after) != _inode_identity(before)
            or _inode_identity(after) != _inode_identity(current_parent)
        ):
            raise ValueError("canonical key lock identity changed")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _initialize_owner_only_key_at_destination(destination: Path) -> bool:
    _require_safe_output_parent(destination)
    parent = _open_directory_anchor(destination.parent, "canonical key parent")
    lock_fd: int | None = None
    try:
        lock_fd = _acquire_key_lock_at(parent.descriptor, destination.name)
        _assert_directory_anchor(parent, "canonical key parent")
        _reject_transaction_residue_at(
            parent.descriptor,
            (f".{destination.name}.staging-",),
            "canonical key parent",
        )
        if _entry_stat_at(parent.descriptor, destination.name) is not None:
            _read_exact_key_at(parent.descriptor, destination.name)
            _assert_directory_anchor(parent, "canonical key parent")
            return False

        generated = os.urandom(32)
        if type(generated) is not bytes or len(generated) != 32:
            raise RuntimeError("key generator did not return exactly 32 bytes")
        staging_name = f".{destination.name}.staging-{uuid.uuid4().hex}"
        descriptor: int | None = None
        staging_identity: tuple[int, int] | None = None
        staging_created = False
        committed = False
        pending_error: BaseException | None = None
        try:
            descriptor = os.open(
                staging_name,
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=parent.descriptor,
            )
            staging_created = True
            created = os.fstat(descriptor)
            if (
                not stat.S_ISREG(created.st_mode)
                or created.st_uid != os.geteuid()
                or created.st_nlink != 1
            ):
                raise ValueError("new key staging storage is unsafe")
            staging_identity = _inode_identity(created)
            os.fchmod(descriptor, 0o600)
            view = memoryview(generated)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("canonical key write made no progress")
                view = view[written:]
            os.fsync(descriptor)
            complete = os.fstat(descriptor)
            if (
                not stat.S_ISREG(complete.st_mode)
                or stat.S_IMODE(complete.st_mode) != 0o600
                or complete.st_uid != os.geteuid()
                or complete.st_nlink != 1
                or complete.st_size != 32
                or _inode_identity(complete) != staging_identity
            ):
                raise ValueError("new key staging storage is not exact owner-only storage")
            os.lseek(descriptor, 0, os.SEEK_SET)
            observed = bytearray()
            while chunk := os.read(descriptor, 64):
                observed.extend(chunk)
            staged_path = os.stat(
                staging_name, dir_fd=parent.descriptor, follow_symlinks=False,
            )
            if (
                not hmac.compare_digest(bytes(observed), generated)
                or _full_file_identity(os.fstat(descriptor))
                != _full_file_identity(staged_path)
            ):
                raise ValueError("new key staging bytes or identity changed")
            _fsync_directory_fd(parent.descriptor)
            _assert_directory_anchor(parent, "canonical key parent")
            durable_stage = os.stat(
                staging_name, dir_fd=parent.descriptor, follow_symlinks=False,
            )
            if (
                _full_file_identity(os.fstat(descriptor))
                != _full_file_identity(durable_stage)
            ):
                raise ValueError("new key staging identity changed during durability sync")
            _atomic_publish_directory_no_replace_at(
                parent.descriptor, staging_name, destination.name,
            )
            committed = True
            current = os.stat(
                destination.name, dir_fd=parent.descriptor, follow_symlinks=False,
            )
            if (
                _inode_identity(current) != staging_identity
                or _full_file_identity(current)
                != _full_file_identity(os.fstat(descriptor))
            ):
                raise ValueError("canonical key name does not bind the staged inode")
            _fsync_directory_fd(parent.descriptor)
            _assert_directory_anchor(parent, "canonical key parent")
            current = os.stat(
                destination.name, dir_fd=parent.descriptor, follow_symlinks=False,
            )
            if (
                _inode_identity(current) != staging_identity
                or _full_file_identity(current)
                != _full_file_identity(os.fstat(descriptor))
            ):
                raise ValueError("canonical key identity changed during commit sync")
            os.close(descriptor)
            descriptor = None
            _read_exact_key_at(
                parent.descriptor,
                destination.name,
                expected_identity=staging_identity,
                expected_bytes=generated,
            )
            _assert_directory_anchor(parent, "canonical key parent")
            return True
        except BaseException as caught:
            pending_error = caught
            raise
        finally:
            close_error: BaseException | None = None
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except BaseException as caught:
                    close_error = caught
            if pending_error is not None and staging_created and not committed:
                retained_error = RuntimeError(
                    "canonical key staging is retained as indeterminate"
                )
                cause = _linear_cleanup_cause(
                    pending_error,
                    () if close_error is None else (close_error,),
                )
                assert cause is not None
                raise retained_error from cause
            if close_error is not None:
                if pending_error is not None:
                    outcome = _attach_cleanup_causes(
                        pending_error, (close_error,),
                    )
                    raise outcome.with_traceback(pending_error.__traceback__)
                raise close_error
    finally:
        pending = sys.exc_info()
        cleanup_errors: list[BaseException] = []
        if lock_fd is not None:
            try:
                os.close(lock_fd)
            except BaseException as caught:
                cleanup_errors.append(caught)
        try:
            _close_anchor(parent)
        except BaseException as caught:
            cleanup_errors.append(caught)
        if cleanup_errors:
            if pending[1] is not None:
                outcome = _attach_cleanup_causes(
                    pending[1], cleanup_errors,
                )
                raise outcome.with_traceback(pending[2])
            cleanup_cause = _linear_cleanup_cause(None, cleanup_errors)
            assert cleanup_cause is not None
            raise cleanup_cause


def _prepare_owner_only_key_path(
    value: str | Path,
) -> Path:
    lexical = Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    _safe_entry_name(lexical.name)
    canonical_missing_chain = (
        lexical.name == ".mayo_ssl_hmac.key"
        and lexical.parent.name == "pretraining"
        and lexical.parent.parent.name == "dynamic_landmark"
        and lexical.parent.parent.parent.name == "outputs"
    )
    try:
        parent_status = os.stat(lexical.parent, follow_symlinks=False)
    except FileNotFoundError:
        parent_status = None
    if not canonical_missing_chain or parent_status is not None:
        return _prepare_private_run_path(lexical)

    outputs = _resolve_existing_private_directory(
        lexical.parent.parent.parent, "canonical outputs namespace",
    )
    current = _open_directory_anchor(
        outputs,
        "canonical outputs namespace",
        expected_mode=None,
    )
    try:
        outputs_stat = os.fstat(current.descriptor)
        if (
            outputs_stat.st_uid != os.geteuid()
            or stat.S_IMODE(outputs_stat.st_mode) & 0o022
        ):
            raise ValueError("canonical outputs namespace is writable by another user")
        for name in ("dynamic_landmark", "pretraining"):
            existing = _entry_stat_at(current.descriptor, name)
            was_created = existing is None
            if existing is None:
                child_fd, child_stat = _mkdir_private_directory_at(
                    current.descriptor,
                    name,
                    f"canonical key {name} directory",
                )
            else:
                child_fd, child_stat = _open_private_directory_at(
                    current.descriptor,
                    name,
                    f"canonical key {name} directory",
                )
            child = _DirectoryAnchor(
                path=current.path / name,
                descriptor=child_fd,
                identity=_inode_identity(child_stat),
                mode=stat.S_IMODE(child_stat.st_mode),
            )
            try:
                if was_created:
                    _fsync_directory_fd(current.descriptor)
                if (
                    child_stat.st_uid != os.geteuid()
                    or stat.S_IMODE(child_stat.st_mode) != 0o700
                ):
                    raise ValueError(
                        f"canonical key {name} directory must be owner-only 0700"
                    )
                _assert_directory_anchor(current, f"canonical key {name} parent")
                _close_anchor(current)
            except BaseException:
                _close_anchor(child)
                raise
            current = child
        destination = current.path / lexical.name
        _assert_directory_anchor(current, "canonical key pretraining directory")
        _close_anchor(current)
        current = None
        return destination
    except BaseException:
        if current is not None:
            _close_anchor(current)
        raise


def initialize_owner_only_key(key_path: str | Path) -> bool:
    """Atomically create one 32-byte owner-only key, or validate the winner."""
    destination = _prepare_owner_only_key_path(key_path)
    return _initialize_owner_only_key_at_destination(destination)


def _prepare_private_run_path(value: str | Path) -> Path:
    lexical = Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    _safe_entry_name(lexical.name)
    try:
        parent_status = os.stat(lexical.parent, follow_symlinks=False)
    except FileNotFoundError:
        parent_status = None
    if parent_status is not None:
        if (
            not stat.S_ISDIR(parent_status.st_mode)
            or parent_status.st_uid != os.geteuid()
            or stat.S_IMODE(parent_status.st_mode) != 0o700
        ):
            raise ValueError("run namespace parent is unsafe")
        parent = _resolve_existing_private_directory(
            lexical.parent, "run namespace parent",
        )
    else:
        parent_name = _safe_entry_name(lexical.parent.name)
        try:
            grandparent = _resolve_existing_private_directory(
                lexical.parent.parent, "run namespace grandparent",
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError("run namespace may create only one missing private level") from exc
        grandparent_anchor = _open_directory_anchor(
            grandparent, "run namespace grandparent",
        )
        try:
            grandparent_status = os.fstat(grandparent_anchor.descriptor)
            if (
                grandparent_status.st_uid != os.geteuid()
                or stat.S_IMODE(grandparent_status.st_mode) != 0o700
            ):
                raise ValueError(
                    "run namespace grandparent must be current-owner mode 0700"
                )
            existing = _entry_stat_at(grandparent_anchor.descriptor, parent_name)
            if existing is None:
                try:
                    descriptor, parent_stat = _mkdir_private_directory_at(
                        grandparent_anchor.descriptor,
                        parent_name,
                        "run namespace parent",
                    )
                except FileExistsError:
                    descriptor, parent_stat = _open_private_directory_at(
                        grandparent_anchor.descriptor,
                        parent_name,
                        "run namespace parent",
                    )
                try:
                    if (
                        parent_stat.st_uid != os.geteuid()
                        or stat.S_IMODE(parent_stat.st_mode) != 0o700
                    ):
                        raise ValueError("run namespace parent is unsafe")
                finally:
                    os.close(descriptor)
                _fsync_directory_fd(grandparent_anchor.descriptor)
            elif (
                not stat.S_ISDIR(existing.st_mode)
                or existing.st_uid != os.geteuid()
                or stat.S_IMODE(existing.st_mode) != 0o700
            ):
                raise ValueError("run namespace parent is unsafe")
            _assert_directory_anchor(
                grandparent_anchor, "run namespace grandparent",
            )
        finally:
            _close_anchor(grandparent_anchor)
        parent = grandparent / parent_name
    return parent / lexical.name


def freeze_bridge_stage(
    run_root: str | Path,
    bridge_root: str | Path,
    *,
    mode: str,
    ravdess_authorizer: Callable[[], object],
    mayo_authorizer: Callable[[], object],
    producer_sha256: str,
) -> dict[str, object]:
    """Atomically freeze one immutable, mode-bound SSL inputs generation."""
    if not callable(ravdess_authorizer) or not callable(mayo_authorizer):
        raise ValueError("bridge authorizers must be callable")
    if type(mode) is not str or mode not in {"smoke", "formal"}:
        raise ValueError("bridge freeze mode must be exactly smoke or formal")
    bridge_parent: _DirectoryAnchor | None = None
    bridge_anchor: _DirectoryAnchor | None = None
    bridge_lock: _DestinationLock | None = None
    run_parent: _DirectoryAnchor | None = None
    run_anchor: _DirectoryAnchor | None = None
    inputs_lock: _DestinationLock | None = None
    staging_fd: int | None = None
    staging_identity: tuple[int, int] | None = None
    staging_name = f".inputs.staging-{uuid.uuid4().hex}"
    ledger: _OwnedTreeLedger | None = None
    pending_error: BaseException | None = None
    try:
        bridge = _lexical_absolute(bridge_root)
        _require_safe_output_parent(bridge)
        bridge_parent = _open_directory_anchor(
            bridge.parent, "bridge generation parent",
        )
        bridge_lock = _acquire_destination_lock_at(
            bridge_parent.descriptor,
            f".{bridge.name}.lock",
            create=False,
            exclusive=False,
            field="bridge destination lock",
        )
        _assert_directory_anchor(bridge_parent, "bridge generation parent")
        bridge_anchor = _open_directory_anchor(
            bridge, "committed bridge generation",
        )
        bridge_name_stat = _entry_stat_at(bridge_parent.descriptor, bridge.name)
        if (
            bridge_name_stat is None
            or _inode_identity(bridge_name_stat) != bridge_anchor.identity
        ):
            raise ValueError("committed bridge generation identity is unsafe")
        run = _prepare_private_run_path(run_root)
        _require_safe_output_parent(run)
        run_parent = _open_directory_anchor(run.parent, "run root parent")
        run_stat = _entry_stat_at(run_parent.descriptor, run.name)
        if run_stat is None:
            try:
                run_fd, run_stat = _mkdir_private_directory_at(
                    run_parent.descriptor, run.name, "run root",
                )
            except FileExistsError:
                run_fd, run_stat = _open_private_directory_at(
                    run_parent.descriptor, run.name, "run root",
                )
            run_anchor = _DirectoryAnchor(
                path=run,
                descriptor=run_fd,
                identity=_inode_identity(run_stat),
                mode=stat.S_IMODE(run_stat.st_mode),
            )
            _fsync_directory_fd(run_parent.descriptor)
            _assert_directory_anchor(run_parent, "run root parent")
        else:
            run_fd, run_stat = _open_private_directory_at(
                run_parent.descriptor, run.name, "run root",
            )
            run_anchor = _DirectoryAnchor(
                path=run,
                descriptor=run_fd,
                identity=_inode_identity(run_stat),
                mode=stat.S_IMODE(run_stat.st_mode),
            )
        run_opened = os.fstat(run_anchor.descriptor)
        if (
            run_opened.st_uid != os.geteuid()
            or stat.S_IMODE(run_opened.st_mode) != 0o700
            or run_stat.st_uid != os.geteuid()
            or stat.S_IMODE(run_stat.st_mode) != 0o700
        ):
            raise ValueError("run root must be exact owner-only mode 0700")
        _assert_directory_anchor(run_anchor, "run root")
        inputs_lock = _acquire_destination_lock_at(
            run_anchor.descriptor,
            ".inputs.lock",
            create=True,
            exclusive=True,
            field="frozen inputs destination lock",
        )
        _assert_directory_anchor(run_anchor, "run root")
        existing_names = set(os.listdir(run_anchor.descriptor))
        if "inputs" in existing_names:
            raise FileExistsError("frozen inputs generation already exists")
        if any(name.startswith(".inputs.staging-") for name in existing_names):
            raise ValueError("run root contains unresolved frozen-input staging state")

        first_generation = _prepare_live_generation(
            ravdess_authorizer,
            mayo_authorizer,
            producer_sha256=producer_sha256,
            anchors=(
                (bridge_parent, "bridge generation parent"),
                (bridge_anchor, "committed bridge generation"),
                (run_parent, "run root parent"),
                (run_anchor, "run root"),
            ),
        )
        _validate_generation_fd(bridge_anchor.descriptor, first_generation)
        first = _prepare_frozen_inputs(first_generation, mode=mode)
        _assert_destination_lock_at(
            inputs_lock,
            run_anchor.descriptor,
            "frozen inputs destination lock",
        )
        _assert_destination_lock_at(
            bridge_lock,
            bridge_parent.descriptor,
            "bridge destination lock",
        )

        staging_fd, staging_stat = _mkdir_private_directory_at(
            run_anchor.descriptor, staging_name, "frozen inputs staging root",
        )
        staging_identity = _inode_identity(staging_stat)
        ledger = _OwnedTreeLedger.create(staging_stat)
        _assert_directory_anchor(run_anchor, "run root")
        _write_frozen_inputs_fd(staging_fd, first, ledger)
        _validate_frozen_inputs_fd(staging_fd, first)

        second_generation = _prepare_live_generation(
            ravdess_authorizer,
            mayo_authorizer,
            producer_sha256=producer_sha256,
            anchors=(
                (bridge_parent, "bridge generation parent"),
                (bridge_anchor, "committed bridge generation"),
                (run_parent, "run root parent"),
                (run_anchor, "run root"),
            ),
        )
        if not _prepared_equal(first_generation, second_generation):
            raise ValueError("upstream authorization changed during stage freeze")
        _validate_generation_fd(bridge_anchor.descriptor, second_generation)
        second = _prepare_frozen_inputs(second_generation, mode=mode)
        if not _frozen_inputs_equal(first, second):
            raise ValueError("frozen inputs changed during stage freeze")
        _validate_frozen_inputs_fd(staging_fd, second)
        _require_entry_absent_at(
            run_anchor.descriptor, "inputs", "frozen inputs generation",
        )
        _assert_destination_lock_at(
            inputs_lock,
            run_anchor.descriptor,
            "frozen inputs destination lock",
        )
        _assert_destination_lock_at(
            bridge_lock,
            bridge_parent.descriptor,
            "bridge destination lock",
        )
        _assert_directory_anchor(run_anchor, "run root")
        _atomic_publish_directory_no_replace_at(
            run_anchor.descriptor, staging_name, "inputs",
        )
        if _classify_owned_publication(
            run_anchor.descriptor, staging_name, "inputs", ledger.root_identity,
        ) != "published":
            raise RuntimeError("frozen inputs publication outcome is indeterminate")
        _fsync_directory_fd(run_anchor.descriptor)
        _assert_directory_anchor(run_anchor, "run root")
        _assert_directory_anchor(run_parent, "run root parent")
        _validate_frozen_inputs_fd(staging_fd, second)
        destination = _entry_stat_at(run_anchor.descriptor, "inputs")
        if destination is None or _inode_identity(destination) != ledger.root_identity:
            raise RuntimeError("frozen inputs name no longer binds staged storage")
        _reject_transaction_residue_at(
            run_anchor.descriptor,
            (".inputs.staging-",),
            "run root",
        )
        _assert_destination_lock_at(
            inputs_lock,
            run_anchor.descriptor,
            "frozen inputs destination lock",
        )
        _assert_destination_lock_at(
            bridge_lock,
            bridge_parent.descriptor,
            "bridge destination lock",
        )
        _assert_directory_anchor(bridge_parent, "bridge generation parent")
        return {
            "mode": mode,
            "sample_count": int(
                second_generation.ravdess.record["sample_count"]
            ) + int(second_generation.mayo.record["sample_count"]),
            "stage_count": 2,
        }
    except BaseException as caught:
        pending_error = caught
        raise
    finally:
        retained_error: RuntimeError | None = None
        cleanup_errors: list[BaseException] = []
        if staging_fd is not None:
            try:
                os.close(staging_fd)
            except BaseException as caught:
                cleanup_errors.append(caught)
        if staging_identity is not None and run_anchor is not None:
            try:
                state = _classify_owned_publication(
                    run_anchor.descriptor,
                    staging_name,
                    "inputs",
                    staging_identity,
                )
            except BaseException as caught:
                cleanup_errors.append(caught)
                state = "indeterminate"
            if pending_error is not None and state == "staged":
                retained_error = RuntimeError(
                    "frozen input staging is retained as indeterminate"
                )
            elif state == "indeterminate":
                retained_error = RuntimeError(
                    "frozen input transaction is retained as indeterminate"
                )
        for cleanup in (
            (() if inputs_lock is None else (
                lambda: _release_destination_lock(inputs_lock),
            )),
            (() if run_anchor is None else (
                lambda: _close_anchor(run_anchor),
            )),
            (() if run_parent is None else (
                lambda: _close_anchor(run_parent),
            )),
            (() if bridge_anchor is None else (
                lambda: _close_anchor(bridge_anchor),
            )),
            (() if bridge_lock is None else (
                lambda: _release_destination_lock(bridge_lock),
            )),
            (() if bridge_parent is None else (
                lambda: _close_anchor(bridge_parent),
            )),
        ):
            for operation in cleanup:
                try:
                    operation()
                except BaseException as caught:
                    cleanup_errors.append(caught)
        if retained_error is not None:
            cause = _linear_cleanup_cause(pending_error, cleanup_errors)
            if cause is None:
                raise retained_error
            raise retained_error from cause
        if cleanup_errors:
            if pending_error is not None:
                outcome = _attach_cleanup_causes(
                    pending_error, cleanup_errors,
                )
                raise outcome.with_traceback(pending_error.__traceback__)
            cleanup_cause = _linear_cleanup_cause(None, cleanup_errors)
            assert cleanup_cause is not None
            raise cleanup_cause


def verify_frozen_bridge_stage(
    inputs_root: str | Path,
    bridge_root: str | Path,
    *,
    mode: str,
    ravdess_authorizer: Callable[[], object],
    mayo_authorizer: Callable[[], object],
    producer_sha256: str,
    before_authorization: Callable[[], None] | None = None,
    finalize_locked: Callable[[], None] | None = None,
    include_generation_result: bool = False,
) -> dict[str, object]:
    """Reauthorize and byte-verify one committed mode-bound input tree."""
    if not callable(ravdess_authorizer) or not callable(mayo_authorizer):
        raise ValueError("bridge authorizers must be callable")
    if type(mode) is not str or mode not in {"smoke", "formal"}:
        raise ValueError("bridge verification mode must be exactly smoke or formal")
    if (
        before_authorization is not None and not callable(before_authorization)
        or finalize_locked is not None and not callable(finalize_locked)
        or type(include_generation_result) is not bool
    ):
        raise ValueError("bridge verification callbacks are malformed")
    inputs = _lexical_absolute(inputs_root)
    bridge = _lexical_absolute(bridge_root)
    if inputs.name != "inputs":
        raise ValueError("committed frozen inputs must use the exact inputs name")
    _require_safe_output_parent(bridge)
    inputs_anchor: _DirectoryAnchor | None = None
    run_anchor: _DirectoryAnchor | None = None
    inputs_lock: _DestinationLock | None = None
    bridge_anchor: _DirectoryAnchor | None = None
    bridge_parent: _DirectoryAnchor | None = None
    bridge_lock: _DestinationLock | None = None
    try:
        bridge_parent = _open_directory_anchor(
            bridge.parent, "bridge generation parent",
        )
        bridge_lock = _acquire_destination_lock_at(
            bridge_parent.descriptor,
            f".{bridge.name}.lock",
            create=False,
            exclusive=False,
            field="bridge destination lock",
        )
        bridge_anchor = _open_directory_anchor(
            bridge, "committed bridge generation",
        )
        bridge_name_stat = _entry_stat_at(bridge_parent.descriptor, bridge.name)
        if (
            bridge_name_stat is None
            or _inode_identity(bridge_name_stat) != bridge_anchor.identity
        ):
            raise ValueError("committed bridge generation identity is unsafe")
        run_anchor = _open_directory_anchor(inputs.parent, "committed run root")
        inputs_lock = _acquire_destination_lock_at(
            run_anchor.descriptor,
            ".inputs.lock",
            create=False,
            exclusive=False,
            field="frozen inputs destination lock",
        )
        _reject_transaction_residue_at(
            run_anchor.descriptor,
            (".inputs.staging-",),
            "committed run root",
        )
        inputs_anchor = _open_directory_anchor(inputs, "committed frozen inputs")
        inputs_name_stat = _entry_stat_at(run_anchor.descriptor, "inputs")
        if (
            inputs_name_stat is None
            or _inode_identity(inputs_name_stat) != inputs_anchor.identity
        ):
            raise ValueError("committed frozen inputs identity is unsafe")
        anchors = (
            (bridge_parent, "bridge generation parent"),
            (bridge_anchor, "committed bridge generation"),
            (run_anchor, "committed run root"),
            (inputs_anchor, "committed frozen inputs"),
        )
        if before_authorization is not None:
            before_authorization()
        first_generation = _prepare_live_generation(
            ravdess_authorizer,
            mayo_authorizer,
            producer_sha256=producer_sha256,
            anchors=anchors,
        )
        first_validation = _validate_generation_fd(
            bridge_anchor.descriptor, first_generation,
        )
        first = _prepare_frozen_inputs(first_generation, mode=mode)
        _validate_frozen_inputs_fd(inputs_anchor.descriptor, first)
        second_generation = _prepare_live_generation(
            ravdess_authorizer,
            mayo_authorizer,
            producer_sha256=producer_sha256,
            anchors=anchors,
        )
        if not _prepared_equal(first_generation, second_generation):
            raise ValueError("upstream authorization changed during input verification")
        second = _prepare_frozen_inputs(second_generation, mode=mode)
        if not _frozen_inputs_equal(first, second):
            raise ValueError("mode-bound inputs are nondeterministic")
        second_validation = _validate_generation_fd(
            bridge_anchor.descriptor, second_generation,
        )
        if second_validation != first_validation:
            raise ValueError("committed bridge validation facts changed")
        _validate_frozen_inputs_fd(inputs_anchor.descriptor, second)
        if finalize_locked is not None:
            finalize_locked()
        if _validate_generation_fd(
            bridge_anchor.descriptor, second_generation,
        ) != second_validation:
            raise ValueError("committed bridge changed during locked finalization")
        _validate_frozen_inputs_fd(inputs_anchor.descriptor, second)
        for anchor, field in anchors:
            _assert_directory_anchor(anchor, field)
        _reject_transaction_residue_at(
            run_anchor.descriptor,
            (".inputs.staging-",),
            "committed run root",
        )
        _assert_destination_lock_at(
            inputs_lock,
            run_anchor.descriptor,
            "frozen inputs destination lock",
        )
        _assert_destination_lock_at(
            bridge_lock,
            bridge_parent.descriptor,
            "bridge destination lock",
        )
        if include_generation_result:
            bundle_count, total_bytes, modes_ok, non_0600, privacy_ok = (
                second_validation
            )
            result: dict[str, object] = {
                "bundle_count": bundle_count,
                "bundle_total_bytes": total_bytes,
                "deterministic": True,
                "modes_ok": modes_ok,
                "non_0600_private_file_count": non_0600,
                "privacy_ok": privacy_ok,
                "size_ok": 0 < total_bytes <= _MAX_BUNDLE_BYTES,
            }
            if not all(bool(result[name]) for name in (
                "deterministic", "modes_ok", "privacy_ok", "size_ok",
            )):
                raise ValueError("bridge determinism aggregate gate failed")
            return result
        return {
            "mode": mode,
            "sample_count": int(second_generation.ravdess.record["sample_count"])
            + int(second_generation.mayo.record["sample_count"]),
            "stage_count": 2,
        }
    finally:
        try:
            if inputs_anchor is not None:
                _close_anchor(inputs_anchor)
        finally:
            try:
                if inputs_lock is not None:
                    _release_destination_lock(inputs_lock)
            finally:
                try:
                    if run_anchor is not None:
                        _close_anchor(run_anchor)
                finally:
                    try:
                        if bridge_anchor is not None:
                            _close_anchor(bridge_anchor)
                    finally:
                        try:
                            if bridge_lock is not None:
                                _release_destination_lock(bridge_lock)
                        finally:
                            if bridge_parent is not None:
                                _close_anchor(bridge_parent)


def verify_bridge_generation(
    bridge_root: str | Path,
    *,
    ravdess_authorizer: Callable[[], object],
    mayo_authorizer: Callable[[], object],
    producer_sha256: str,
    before_authorization: Callable[[], None] | None = None,
    finalize_locked: Callable[[], None] | None = None,
) -> dict[str, object]:
    """Reauthorize twice in memory and verify one held committed generation."""
    if not callable(ravdess_authorizer) or not callable(mayo_authorizer):
        raise ValueError("bridge authorizers must be callable")
    if (
        before_authorization is not None and not callable(before_authorization)
        or finalize_locked is not None and not callable(finalize_locked)
    ):
        raise ValueError("bridge verification callbacks are malformed")
    root = _lexical_absolute(bridge_root)
    _require_safe_output_parent(root)
    parent: _DirectoryAnchor | None = None
    root_anchor: _DirectoryAnchor | None = None
    destination_lock: _DestinationLock | None = None

    try:
        parent = _open_directory_anchor(root.parent, "bridge verification parent")
        destination_lock = _acquire_destination_lock_at(
            parent.descriptor,
            f".{root.name}.lock",
            create=False,
            exclusive=False,
            field="bridge destination lock",
        )
        _assert_directory_anchor(parent, "bridge verification parent")
        _reject_transaction_residue_at(
            parent.descriptor,
            (f".{root.name}.staging-", f".{root.name}.verify-"),
            "bridge verification parent",
        )
        root_anchor = _open_directory_anchor(root, "committed bridge generation")
        root_name_stat = _entry_stat_at(parent.descriptor, root.name)
        if (
            root_name_stat is None
            or _inode_identity(root_name_stat) != root_anchor.identity
        ):
            raise ValueError("committed bridge generation identity is unsafe")
        anchors = (
            (parent, "bridge verification parent"),
            (root_anchor, "committed bridge generation"),
        )
        if before_authorization is not None:
            before_authorization()
        first = _prepare_live_generation(
            ravdess_authorizer=ravdess_authorizer,
            mayo_authorizer=mayo_authorizer,
            producer_sha256=producer_sha256,
            anchors=anchors,
        )
        first_validation = _validate_generation_fd(
            root_anchor.descriptor, first,
        )
        second = _prepare_live_generation(
            ravdess_authorizer,
            mayo_authorizer,
            producer_sha256=producer_sha256,
            anchors=anchors,
        )
        if not _prepared_equal(first, second):
            raise ValueError("bridge inputs changed during determinism verification")
        second_validation = _validate_generation_fd(root_anchor.descriptor, second)
        if second_validation != first_validation:
            raise ValueError("committed bridge validation facts changed")
        if finalize_locked is not None:
            finalize_locked()
        if _validate_generation_fd(root_anchor.descriptor, second) != second_validation:
            raise ValueError("committed bridge changed during locked finalization")
        for anchor, field in anchors:
            _assert_directory_anchor(anchor, field)
        _assert_destination_lock_at(
            destination_lock, parent.descriptor, "bridge destination lock",
        )
        bundle_count, total_bytes, modes_ok, non_0600, privacy_ok = first_validation
        result: dict[str, object] = {
            "bundle_count": bundle_count,
            "bundle_total_bytes": total_bytes,
            "deterministic": True,
            "modes_ok": modes_ok,
            "non_0600_private_file_count": non_0600,
            "privacy_ok": privacy_ok,
            "size_ok": 0 < total_bytes <= _MAX_BUNDLE_BYTES,
        }
        if not all(bool(result[name]) for name in (
            "deterministic", "modes_ok", "privacy_ok", "size_ok",
        )):
            raise ValueError("bridge determinism aggregate gate failed")
        return result
    finally:
        try:
            if root_anchor is not None:
                _close_anchor(root_anchor)
        finally:
            try:
                if destination_lock is not None:
                    _release_destination_lock(destination_lock)
            finally:
                if parent is not None:
                    _close_anchor(parent)


__all__ = [
    "BridgePolicy",
    "CanonicalPacketBundle",
    "PrivateTrajectoryMapping",
    "build_bridge_bundles",
    "freeze_bridge_stage",
    "initialize_owner_only_key",
    "packetize_mayo_trajectory",
    "packetize_ravdess_trajectory",
    "uniform_floor_starts",
    "verify_bridge_generation",
    "verify_frozen_bridge_stage",
]
