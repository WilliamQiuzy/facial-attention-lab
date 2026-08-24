"""Validated recording-level caches for dynamic landmark experiments.

This module defines only the data and temporal-difference contracts. It does
not extract features, train models, or choose evaluation data.
"""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .feature_schema import MP_FEATURE_NAMES_BY_SCHEMA


DYNAMIC_FEATURE_SCHEMA = "mediapipe_bs_lr_v1+clinical23_v2"
DYNAMIC_FEATURE_NAMES = MP_FEATURE_NAMES_BY_SCHEMA[DYNAMIC_FEATURE_SCHEMA]
DYNAMIC_FEATURE_SHAPE = (4, 32, 95)
DYNAMIC_MASK_SHAPE = DYNAMIC_FEATURE_SHAPE[:2]
MIN_RECORDING_COVERAGE = 0.90

_REQUIRED_CACHE_FIELDS = frozenset({
    "features",
    "valid_mask",
    "timestamps",
    "timestamp_unit",
    "source_frame_indices",
    "source_frame_count",
    "feature_schema",
    "feature_names",
    "recording_id",
    "group_id",
    "label",
    "source_sha256",
})
_RECORDING_ID = re.compile(r"^rec_[0-9a-f]{64}$")
_GROUP_ID = re.compile(r"^grp_[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class DynamicLandmarkRecording:
    """One validated recording and its four fixed temporal windows."""

    features: np.ndarray
    valid_mask: np.ndarray
    timestamps: np.ndarray
    timestamp_unit: str
    source_frame_indices: np.ndarray
    source_frame_count: int
    feature_schema: str
    feature_names: tuple[str, ...]
    recording_id: str
    group_id: str
    label: int
    source_sha256: str
    cache_path: Path

    @property
    def coverage(self) -> float:
        return float(self.valid_mask.mean())


def deterministic_window_starts(
    n_frames: int,
    window_len: int = 32,
    n_windows: int = 4,
) -> tuple[int, ...]:
    """Return evenly spaced, non-overlapping contiguous-window starts.

    The frozen protocol requires at least 128 source frames. The first window
    starts at frame zero and the final window ends at the final source frame.
    """
    values = {
        "n_frames": n_frames,
        "window_len": window_len,
        "n_windows": n_windows,
    }
    for name, value in values.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer, got {value!r}")
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive, got {value!r}")

    n_frames = int(n_frames)
    window_len = int(window_len)
    n_windows = int(n_windows)
    if n_windows < 2:
        raise ValueError("n_windows must be at least 2 to span a recording")
    required = max(128, window_len * n_windows)
    if n_frames < required:
        raise ValueError(
            f"recording has {n_frames} frames; at least {required} are required "
            f"for {n_windows} non-overlapping windows of length {window_len}"
        )

    final_start = n_frames - window_len
    starts = tuple(
        (index * final_start) // (n_windows - 1)
        for index in range(n_windows)
    )
    if len(starts) != n_windows:
        raise AssertionError("window construction did not emit the requested count")
    if starts[0] != 0 or starts[-1] + window_len != n_frames:
        raise AssertionError("window construction did not span the recording")
    if any(right - left < window_len
           for left, right in zip(starts, starts[1:])):
        raise ValueError("requested windows would overlap")
    return starts


def _scalar_text(value: np.ndarray, field: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{field} must be a scalar string")
    item = array.item()
    if isinstance(item, bytes):
        try:
            item = item.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{field} must be valid UTF-8") from exc
    text = str(item)
    if not text:
        raise ValueError(f"{field} must be nonempty")
    if text != text.strip():
        raise ValueError(f"{field} must not contain leading or trailing whitespace")
    return text


def _binary_label(value: np.ndarray) -> int:
    array = np.asarray(value)
    if (array.shape != () or isinstance(array.item(), (bool, np.bool_))
            or not np.issubdtype(array.dtype, np.integer)):
        raise ValueError("label must be a scalar integer")
    label = int(array.item())
    if label not in (0, 1):
        raise ValueError(f"label must be binary (0 or 1), got {label}")
    return label


def _source_frame_count(value: np.ndarray) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"i", "u"}:
        raise ValueError("source_frame_count must be a scalar non-bool integer")
    frame_count = int(array.item())
    if frame_count < 128:
        raise ValueError(
            f"source_frame_count must be at least 128, got {frame_count}"
        )
    return frame_count


def _ordered_feature_names(value: np.ndarray) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.shape != (DYNAMIC_FEATURE_SHAPE[-1],) or array.dtype.kind not in {"U", "S"}:
        raise ValueError(
            "feature_names must be a one-dimensional string array of length 95"
        )
    names = tuple(
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in array.tolist()
    )
    if names != DYNAMIC_FEATURE_NAMES:
        raise ValueError(
            f"feature_names do not match registered schema {DYNAMIC_FEATURE_SCHEMA!r}"
        )
    return names


def _validate_arrays(
    fields: dict[str, np.ndarray],
    source_frame_count: int,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    features = np.asarray(fields["features"])
    valid_mask = np.asarray(fields["valid_mask"])
    timestamps = np.asarray(fields["timestamps"])
    source_indices = np.asarray(fields["source_frame_indices"])

    if features.shape != DYNAMIC_FEATURE_SHAPE or features.dtype != np.dtype(np.float32):
        raise ValueError(
            f"features must have dtype float32 and shape {DYNAMIC_FEATURE_SHAPE}, "
            f"got dtype {features.dtype} and shape {features.shape}"
        )
    if valid_mask.shape != DYNAMIC_MASK_SHAPE or valid_mask.dtype != np.dtype(bool):
        raise ValueError(
            f"valid_mask must have dtype bool and shape {DYNAMIC_MASK_SHAPE}, "
            f"got dtype {valid_mask.dtype} and shape {valid_mask.shape}"
        )
    if (timestamps.shape != DYNAMIC_MASK_SHAPE
            or timestamps.dtype.kind not in {"i", "u", "f"}):
        raise ValueError(
            f"timestamps must be a real numeric array with shape {DYNAMIC_MASK_SHAPE}"
        )
    if not np.isfinite(timestamps).all():
        raise ValueError("timestamps must contain only finite values")
    if not np.all(timestamps[:, 1:] > timestamps[:, :-1]):
        raise ValueError("timestamps must be strictly increasing within every window")
    if (source_indices.shape != DYNAMIC_MASK_SHAPE
            or source_indices.dtype.kind not in {"i", "u"}):
        raise ValueError(
            "source_frame_indices must be an integer array with shape "
            f"{DYNAMIC_MASK_SHAPE}"
        )
    if np.any(source_indices < 0):
        raise ValueError("source_frame_indices must be nonnegative")
    left_indices = source_indices[:, :-1]
    right_indices = source_indices[:, 1:]
    adjacent = (
        (right_indices > left_indices)
        & (right_indices - left_indices == 1)
    )
    if not np.all(adjacent):
        raise ValueError(
            "source_frame_indices must increase by exactly one within each window"
        )
    window_starts = source_indices[:, 0]
    expected_starts = deterministic_window_starts(source_frame_count)
    observed_starts = tuple(int(value) for value in window_starts.tolist())
    if observed_starts != expected_starts:
        raise ValueError(
            f"source window starts {observed_starts} do not match frozen starts "
            f"{expected_starts} for source_frame_count={source_frame_count}"
        )
    if int(source_indices[-1, -1]) != source_frame_count - 1:
        raise ValueError("last source window must end at source_frame_count - 1")
    ordered_nonoverlapping = (
        (window_starts[1:] > window_starts[:-1])
        & (window_starts[1:] - window_starts[:-1] >= DYNAMIC_FEATURE_SHAPE[1])
    )
    if not np.all(ordered_nonoverlapping):
        raise ValueError("source windows must be ordered and non-overlapping")

    if not np.isfinite(features[valid_mask]).all():
        raise ValueError("valid feature values must be finite")
    if np.any(features[~valid_mask] != 0):
        raise ValueError("invalid or padded feature values must be canonical zero")
    coverage = float(valid_mask.mean())
    if coverage < MIN_RECORDING_COVERAGE:
        raise ValueError(
            f"recording coverage {coverage:.3f} is below required "
            f"{MIN_RECORDING_COVERAGE:.0%}"
        )

    return (
        features.copy(),
        valid_mask.copy(),
        timestamps.copy(),
        source_indices.copy(),
    )


def _load_npz_fields(
    source,
    *,
    identity: str,
) -> dict[str, np.ndarray]:
    try:
        if hasattr(source, "seek"):
            source.seek(0)
        expected_members = {
            f"{field}.npy" for field in _REQUIRED_CACHE_FIELDS
        }
        with zipfile.ZipFile(source, "r") as archive:
            member_names = [info.filename for info in archive.infolist()]
        if (
            len(member_names) != len(expected_members)
            or len(set(member_names)) != len(member_names)
            or set(member_names) != expected_members
        ):
            raise ValueError(
                f"dynamic landmark cache {identity} has duplicate or "
                "noncanonical NPZ members"
            )
        if hasattr(source, "seek"):
            source.seek(0)
        with np.load(source, allow_pickle=False) as saved:
            saved_files = list(saved.files)
            if (
                len(saved_files) != len(_REQUIRED_CACHE_FIELDS)
                or len(set(saved_files)) != len(saved_files)
            ):
                raise ValueError(
                    f"dynamic landmark cache {identity} contains duplicate arrays"
                )
            missing = sorted(_REQUIRED_CACHE_FIELDS.difference(saved.files))
            if missing:
                raise ValueError(
                    f"dynamic landmark cache {identity} is missing required fields: {missing}"
                )
            unexpected = sorted(set(saved.files).difference(_REQUIRED_CACHE_FIELDS))
            if unexpected:
                raise ValueError(
                    f"dynamic landmark cache {identity} has unexpected fields: {unexpected}"
                )
            fields = {name: np.asarray(saved[name]) for name in _REQUIRED_CACHE_FIELDS}
    except ValueError:
        raise
    except (
        EOFError,
        KeyError,
        OSError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ) as exc:
        raise ValueError(f"cannot load dynamic landmark cache {identity}: {exc}") from exc
    return fields


def _recording_from_fields(
    fields: dict[str, np.ndarray],
    *,
    cache_path: Path,
) -> DynamicLandmarkRecording:
    source_frame_count = _source_frame_count(fields["source_frame_count"])
    timestamp_unit = _scalar_text(fields["timestamp_unit"], "timestamp_unit")
    if timestamp_unit != "seconds":
        raise ValueError("timestamp_unit must be exactly 'seconds'")
    features, valid_mask, timestamps, source_indices = _validate_arrays(
        fields, source_frame_count
    )
    schema = _scalar_text(fields["feature_schema"], "feature_schema")
    if schema != DYNAMIC_FEATURE_SCHEMA:
        raise ValueError(
            f"feature_schema must be exactly {DYNAMIC_FEATURE_SCHEMA!r}, got {schema!r}"
        )
    feature_names = _ordered_feature_names(fields["feature_names"])

    recording_id = _scalar_text(fields["recording_id"], "recording_id")
    if _RECORDING_ID.fullmatch(recording_id) is None:
        raise ValueError(
            "recording_id must use canonical opaque format rec_ followed by "
            "64 lowercase hexadecimal characters"
        )
    group_id = _scalar_text(fields["group_id"], "group_id")
    if _GROUP_ID.fullmatch(group_id) is None:
        raise ValueError(
            "group_id must use canonical opaque format grp_ followed by "
            "64 lowercase hexadecimal characters"
        )
    label = _binary_label(fields["label"])
    source_sha256 = _scalar_text(fields["source_sha256"], "source_sha256")
    if _SHA256.fullmatch(source_sha256) is None:
        raise ValueError("source_sha256 must contain exactly 64 hexadecimal characters")

    return DynamicLandmarkRecording(
        features=features,
        valid_mask=valid_mask,
        timestamps=timestamps,
        timestamp_unit=timestamp_unit,
        source_frame_indices=source_indices,
        source_frame_count=source_frame_count,
        feature_schema=schema,
        feature_names=feature_names,
        recording_id=recording_id,
        group_id=group_id,
        label=label,
        source_sha256=source_sha256.lower(),
        cache_path=cache_path,
    )


def load_dynamic_landmark_recording(
    cache_path: str | Path,
) -> DynamicLandmarkRecording:
    """Load and fail-closed validate one NPZ cache representing one recording."""
    path = Path(cache_path)
    fields = _load_npz_fields(path, identity=str(path))
    return _recording_from_fields(fields, cache_path=path)


def load_dynamic_landmark_recording_bytes(
    cache_payload: bytes,
) -> DynamicLandmarkRecording:
    """Validate one immutable NPZ byte string without creating or reopening a path."""
    if type(cache_payload) is not bytes or not cache_payload:
        raise ValueError("dynamic landmark cache payload must be nonempty exact bytes")
    fields = _load_npz_fields(
        io.BytesIO(cache_payload),
        identity="<immutable-bytes>",
    )
    return _recording_from_fields(
        fields,
        cache_path=Path("<immutable-bytes>"),
    )


def load_dynamic_landmark_recordings(
    cache_paths: Iterable[str | Path],
) -> tuple[DynamicLandmarkRecording, ...]:
    """Load a deterministic collection and enforce unique recording IDs."""
    records = tuple(
        load_dynamic_landmark_recording(path)
        for path in sorted((Path(path) for path in cache_paths), key=lambda p: str(p))
    )
    seen: set[str] = set()
    duplicates: set[str] = set()
    for record in records:
        if record.recording_id in seen:
            duplicates.add(record.recording_id)
        seen.add(record.recording_id)
    if duplicates:
        raise ValueError(
            f"recording_id values must be unique across caches; duplicates: "
            f"{sorted(duplicates)}"
        )
    return records


def per_second_first_differences(
    features: np.ndarray,
    valid_mask: np.ndarray,
    timestamps: np.ndarray,
    source_frame_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return first differences per second and their endpoint-validity mask.

    A difference is valid only across adjacent source frames with two valid
    detector rows and finite, strictly increasing timestamps. Detector gaps
    are never searched across or bridged. The first timestep is always zero
    and invalid.
    """
    features = np.asarray(features)
    valid_mask = np.asarray(valid_mask)
    timestamps = np.asarray(timestamps)
    source_indices = np.asarray(source_frame_indices)

    if features.ndim < 2 or features.shape[:-1] != valid_mask.shape:
        raise ValueError("features must have shape valid_mask.shape + (n_features,)")
    if timestamps.shape != valid_mask.shape or source_indices.shape != valid_mask.shape:
        raise ValueError(
            "timestamps and source_frame_indices must have the same shape as valid_mask"
        )
    if valid_mask.dtype != np.dtype(bool):
        raise ValueError("valid_mask must have bool dtype")
    if timestamps.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("timestamps must be real numeric values")
    if source_indices.dtype.kind not in {"i", "u"}:
        raise ValueError("source_frame_indices must have integer dtype")
    if features.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("features must be real numeric values")

    result_dtype = np.result_type(features.dtype, np.float32)
    delta = np.zeros(features.shape, dtype=result_dtype)
    delta_valid = np.zeros(valid_mask.shape, dtype=bool)
    if valid_mask.shape[-1] < 2:
        return delta, delta_valid

    left_time = timestamps[..., :-1]
    right_time = timestamps[..., 1:]
    elapsed = (
        right_time.astype(np.float64, copy=False)
        - left_time.astype(np.float64, copy=False)
    )
    left_index = source_indices[..., :-1]
    right_index = source_indices[..., 1:]
    consecutive_source = (
        (right_index > left_index)
        & (right_index - left_index == 1)
    )
    endpoint_valid = (
        valid_mask[..., :-1]
        & valid_mask[..., 1:]
        & consecutive_source
        & np.isfinite(left_time)
        & np.isfinite(right_time)
        & (elapsed > 0)
        & np.isfinite(features[..., :-1, :]).all(axis=-1)
        & np.isfinite(features[..., 1:, :]).all(axis=-1)
    )
    numeric_features = features.astype(np.float64, copy=False)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        candidate = (
            numeric_features[..., 1:, :] - numeric_features[..., :-1, :]
        ) / elapsed[..., None]
    representable = np.isfinite(candidate)
    if np.issubdtype(result_dtype, np.floating):
        representable &= np.abs(candidate) <= np.finfo(result_dtype).max
    pair_valid = endpoint_valid & representable.all(axis=-1)
    delta[..., 1:, :] = np.where(pair_valid[..., None], candidate, 0)
    delta_valid[..., 1:] = pair_valid
    return delta, delta_valid


# Explicit cache terminology aliases for callers that organize files by cache.
load_dynamic_landmark_cache = load_dynamic_landmark_recording
load_dynamic_landmark_caches = load_dynamic_landmark_recordings
