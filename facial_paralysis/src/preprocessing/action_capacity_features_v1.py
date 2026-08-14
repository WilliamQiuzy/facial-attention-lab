"""Frozen dynamic-capacity projection from supported raw landmark dynamics.

The projection keeps only absolute mouth geometry dynamics: IQR, range, and
peak per-second velocity for the two capture-side mouth corners plus mouth
width and opening. The exact 110D summary is built internally from the same
raw support used for eligibility; callers cannot inject a precomputed row.
Static medians and all derived asymmetry/correlation terms are excluded.
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np

from ..datasets.dynamic_landmark import (
    DYNAMIC_FEATURE_SCHEMA,
    DYNAMIC_FEATURE_NAMES,
    deterministic_window_starts,
    load_dynamic_landmark_recording_bytes,
)
from .generalization_110d import LANDMARK_MI_110D, candidate_feature_names
from .generalization_110d import candidate_feature_vector as _candidate_feature_vector
from .script_action_segmentation_v1 import (
    FacesTimeline,
    NeuroFaceTaskBinding,
    segment_faces_action,
    validate_neuroface_task_binding,
)


ACTION_CAPACITY_DIM = 18
_LANDMARK_MI_110D_SOURCE_NAMES = candidate_feature_names(LANDMARK_MI_110D)
FACES_MIN_VALID_SAMPLES = 26
NEUROFACE_MIN_VALID_SAMPLES = 116
PINNED_NEUROFACE_COLLECTION_MANIFEST_SHA256 = (
    "07527c33fe0e35d34a554f7baccd49e9e692c4588b76aa6392ccad71c122bb17"
)
_MAX_COLLECTION_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_CACHE_PAYLOAD_BYTES = 64 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RECORDING_ID = re.compile(r"^rec_[0-9a-f]{64}$")
_GROUP_ID = re.compile(r"^grp_[0-9a-f]{64}$")
_ALLOWED_NEUROFACE_EXCLUSIONS = frozenset({
    "open_failed", "invalid_fps", "invalid_frame_count",
    "nonintegral_frame_count", "insufficient_frame_count", "invalid_width",
    "invalid_height", "nonintegral_dimensions", "seek_failed", "decode_failed",
    "frame_dimensions_changed", "seek_position_mismatch", "no_valid_detections",
    "coverage_below_0_90",
})

_CAPACITY_CHANNELS = (
    "corner_y_mesh61",
    "corner_y_mesh291",
    "corner_x_mesh61",
    "corner_x_mesh291",
    "mouth_width",
    "mouth_open",
)
_CAPACITY_STATS = ("iqr", "range", "max_abs_velocity_per_second")
_ACTION_CAPACITY_NAMES = tuple(
    f"{channel}__{statistic}"
    for channel in _CAPACITY_CHANNELS
    for statistic in _CAPACITY_STATS
)

if _LANDMARK_MI_110D_SOURCE_NAMES != candidate_feature_names(LANDMARK_MI_110D):
    raise AssertionError("Landmark-MI 110D source-name contract drifted")
if (
    len(_LANDMARK_MI_110D_SOURCE_NAMES) != 110
    or len(set(_LANDMARK_MI_110D_SOURCE_NAMES)) != 110
):
    raise AssertionError("Landmark-MI 110D source names must be unique and exact")
if len(_ACTION_CAPACITY_NAMES) != ACTION_CAPACITY_DIM:
    raise AssertionError("action-capacity feature dimension drifted")
if not set(_ACTION_CAPACITY_NAMES).issubset(_LANDMARK_MI_110D_SOURCE_NAMES):
    raise AssertionError("action-capacity names escaped the frozen 110D schema")

# Bind by the frozen names once at import.  No positional magic numbers are
# allowed to determine which clinical signals enter the 18D representation.
_SOURCE_INDICES = tuple(
    _LANDMARK_MI_110D_SOURCE_NAMES.index(name)
    for name in _ACTION_CAPACITY_NAMES
)
_DYNAMIC_NAME_TO_INDEX = {
    name: index for index, name in enumerate(DYNAMIC_FEATURE_NAMES)
}
_CAPACITY_RAW_INDICES = tuple(
    _DYNAMIC_NAME_TO_INDEX[name] for name in _CAPACITY_CHANNELS
)

_MIRROR_CHANNEL = {
    "corner_y_mesh61": "corner_y_mesh291",
    "corner_y_mesh291": "corner_y_mesh61",
    "corner_x_mesh61": "corner_x_mesh291",
    "corner_x_mesh291": "corner_x_mesh61",
    "mouth_width": "mouth_width",
    "mouth_open": "mouth_open",
}
_MIRROR_NAMES = tuple(
    f"{_MIRROR_CHANNEL[channel]}__{statistic}"
    for channel in _CAPACITY_CHANNELS
    for statistic in _CAPACITY_STATS
)
_MIRROR_INDICES = tuple(
    _ACTION_CAPACITY_NAMES.index(name) for name in _MIRROR_NAMES
)
if tuple(_MIRROR_INDICES[index] for index in _MIRROR_INDICES) != tuple(
    range(ACTION_CAPACITY_DIM)
):
    raise AssertionError("action-capacity mirror must be an exact involution")


def action_capacity_feature_names() -> tuple[str, ...]:
    """Return the exact ordered names for the fixed 18D representation."""
    return _ACTION_CAPACITY_NAMES


def _validate_float64_vector(
    vector: np.ndarray,
    *,
    dimension: int,
    label: str,
) -> np.ndarray:
    values = np.asarray(vector)
    if values.shape != (dimension,) or values.dtype != np.dtype(np.float64):
        raise ValueError(f"{label} must be float64 with shape ({dimension},)")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} must be finite")
    return values


def _validated_raw_recording(
    features: np.ndarray,
    valid_mask: np.ndarray,
    timestamps: np.ndarray,
    source_frame_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(features)
    mask = np.asarray(valid_mask)
    times = np.asarray(timestamps)
    source_indices = np.asarray(source_frame_indices)
    if (
        values.ndim != 3
        or values.shape[0] not in {1, 4}
        or values.shape[1:] != (32, 95)
        or values.dtype != np.dtype(np.float32)
    ):
        raise ValueError(
            "features must be frozen float32 with shape (1, 32, 95) or (4, 32, 95)"
        )
    temporal_shape = values.shape[:2]
    if mask.shape != temporal_shape or mask.dtype != np.dtype(bool):
        raise ValueError(f"valid_mask must be bool with shape {temporal_shape}")
    if times.shape != temporal_shape or times.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"timestamps must be numeric with shape {temporal_shape}")
    normalized_times = times.astype(np.float64, copy=False)
    if not np.isfinite(normalized_times).all() or not np.all(
        normalized_times[:, 1:] > normalized_times[:, :-1]
    ):
        raise ValueError("timestamps must be finite and strictly increasing per window")
    if (
        source_indices.shape != temporal_shape
        or source_indices.dtype.kind not in {"i", "u"}
        or np.any(source_indices < 0)
        or not np.all(source_indices[:, 1:] > source_indices[:, :-1])
    ):
        raise ValueError(
            "source_frame_indices must be nonnegative integers increasing per window"
        )
    if values.shape[0] == 4 and not np.all(
        source_indices[:, 1:] - source_indices[:, :-1] == 1
    ):
        raise ValueError("four-window NeuroFace source frames must remain adjacent")

    finite = np.isfinite(values)
    capacity_indices = np.asarray(_CAPACITY_RAW_INDICES, dtype=np.int64)
    capacity_finite_support = np.sum(
        mask[..., None] & finite[..., capacity_indices],
        axis=(0, 1),
    )
    if values.shape[0] == 1:
        minimum = FACES_MIN_VALID_SAMPLES
        per_channel_support = np.sum(mask[..., None] & finite, axis=(0, 1))
        complete_rows = mask & finite.all(axis=-1)
        if (
            int(np.sum(mask)) < minimum
            or int(np.min(per_channel_support)) < minimum
            or int(np.sum(complete_rows)) < minimum
            or int(np.min(capacity_finite_support)) < minimum
        ):
            raise ValueError("one-window action has fewer than 26 supported samples")
        summary_mask = complete_rows
    else:
        minimum = NEUROFACE_MIN_VALID_SAMPLES
        if int(np.sum(mask)) < minimum:
            raise ValueError("four-window recording has fewer than 116 valid samples")
        if not finite[mask].all():
            raise ValueError("every valid four-window row must have 95 finite channels")
        if int(np.min(capacity_finite_support)) < minimum:
            raise ValueError("a capacity source channel has fewer than 116 finite samples")
        summary_mask = mask
    return values, summary_mask, normalized_times, source_indices


def _validate_dynamic_feature_names(feature_names: tuple[str, ...]) -> None:
    if type(feature_names) is not tuple or feature_names != DYNAMIC_FEATURE_NAMES:
        raise ValueError("feature_names must equal the exact frozen 95-column tuple")


def _validated_complete_faces_stream(
    features: np.ndarray,
    valid_mask: np.ndarray,
    frame_timestamps_ms: np.ndarray,
    source_frame_indices: np.ndarray,
    motion_curve: np.ndarray,
    *,
    decoded_duration_ms: int,
) -> np.ndarray:
    values = np.asarray(features)
    valid = np.asarray(valid_mask)
    times = np.asarray(frame_timestamps_ms)
    indices = np.asarray(source_frame_indices)
    motion = np.asarray(motion_curve)
    if (
        values.ndim != 2
        or values.shape[0] < 2
        or values.shape[1] != len(DYNAMIC_FEATURE_NAMES)
        or values.dtype != np.dtype(np.float32)
    ):
        raise ValueError("FACES features must be complete float32 (n_frames, 95)")
    temporal_shape = (values.shape[0],)
    if valid.shape != temporal_shape or valid.dtype != np.dtype(bool):
        raise ValueError("FACES valid_mask must be an aligned bool vector")
    if (
        times.shape != temporal_shape
        or times.dtype.kind not in {"f", "i", "u"}
        or not np.isfinite(times).all()
        or not np.all(times[1:] > times[:-1])
    ):
        raise ValueError("FACES timestamps must be finite and strictly increasing")
    normalized_times = times.astype(np.float64, copy=False)
    if float(normalized_times[0]) != 0.0:
        raise ValueError("complete FACES timestamps must start at recording time zero")
    if (
        indices.shape != temporal_shape
        or indices.dtype.kind not in {"i", "u"}
        or not np.array_equal(
            indices.astype(np.int64, copy=False),
            np.arange(values.shape[0], dtype=np.int64),
        )
    ):
        raise ValueError("complete FACES source indices must equal arange(n_frames)")
    if motion.shape != temporal_shape or motion.dtype.kind not in {"f", "i", "u"}:
        raise ValueError("FACES motion_curve must be an aligned numeric vector")
    if (
        isinstance(decoded_duration_ms, (bool, np.bool_))
        or not isinstance(decoded_duration_ms, (int, np.integer))
        or int(decoded_duration_ms) <= 0
    ):
        raise ValueError("decoded_duration_ms must be a positive integer")
    frame_period = float(np.median(np.diff(normalized_times)))
    tolerance = frame_period / 2.0 + np.finfo(np.float64).eps * 32
    if (
        not np.isfinite(frame_period)
        or frame_period <= 0.0
        or abs(
            float(normalized_times[-1]) + frame_period - int(decoded_duration_ms)
        ) > tolerance
    ):
        raise ValueError("FACES arrays must span the complete decoded duration")
    return values


def faces_action_capacity_feature_vector(
    timeline: FacesTimeline,
    action: str,
    features: np.ndarray,
    valid_mask: np.ndarray,
    frame_timestamps_ms: np.ndarray,
    source_frame_indices: np.ndarray,
    motion_curve: np.ndarray,
    *,
    decoded_recording_sha256: str,
    decoded_duration_ms: int,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    """Build 18D only from an action anchored by a validated FACES timeline."""
    _validate_dynamic_feature_names(feature_names)
    values = _validated_complete_faces_stream(
        features,
        valid_mask,
        frame_timestamps_ms,
        source_frame_indices,
        motion_curve,
        decoded_duration_ms=decoded_duration_ms,
    )
    segment = segment_faces_action(
        timeline,
        action,
        frame_timestamps_ms,
        source_frame_indices,
        valid_mask,
        motion_curve,
        decoded_recording_sha256=decoded_recording_sha256,
        decoded_duration_ms=decoded_duration_ms,
        landmark_features=values[:, 72:],
    )
    if segment.eligible is not True:
        raise ValueError("the authority-anchored FACES action is not eligible")
    selected = values[segment.frame_positions][None, ...]
    selected_mask = segment.valid_mask[None, ...]
    timing = next(item for item in timeline.actions if item.action == action)
    selected_seconds = (
        timing.hold_start_ms
        + (np.arange(32, dtype=np.float64) + 0.5)
        * (timing.hold_end_ms - timing.hold_start_ms)
        / 32.0
    )[None, ...] / 1000.0
    action_clock = np.arange(32, dtype=np.int64)[None, ...]
    raw, summary_mask, summary_times, summary_indices = _validated_raw_recording(
        selected,
        selected_mask,
        selected_seconds,
        action_clock,
    )
    vector_110d = _candidate_feature_vector(
        LANDMARK_MI_110D,
        raw,
        summary_mask,
        summary_times,
        summary_indices,
    )
    if (
        vector_110d.shape != (len(_LANDMARK_MI_110D_SOURCE_NAMES),)
        or vector_110d.dtype != np.dtype(np.float64)
        or not np.isfinite(vector_110d).all()
    ):
        raise AssertionError("internal FACES summary drifted from finite float64 110D")
    result = vector_110d[np.asarray(_SOURCE_INDICES, dtype=np.int64)].copy()
    if result.shape != (ACTION_CAPACITY_DIM,) or not np.isfinite(result).all():
        raise ValueError("FACES action capacity projection is invalid")
    return result


def _unique_json_object(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"collection manifest contains duplicate key {key!r}")
        output[key] = value
    return output


def _bound_collection_row(
    collection_manifest_bytes: bytes,
    cache_payload: bytes,
    binding: NeuroFaceTaskBinding,
) -> dict[str, object]:
    if type(collection_manifest_bytes) is not bytes or not (
        0 < len(collection_manifest_bytes) <= _MAX_COLLECTION_MANIFEST_BYTES
    ):
        raise ValueError("collection manifest must be bounded exact bytes")
    observed_manifest_sha = hashlib.sha256(collection_manifest_bytes).hexdigest()
    if observed_manifest_sha != PINNED_NEUROFACE_COLLECTION_MANIFEST_SHA256:
        raise ValueError("collection manifest differs from the out-of-band pin")
    try:
        decoded = collection_manifest_bytes.decode("utf-8")
        manifest = json.loads(decoded, object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("collection manifest is not strict UTF-8 JSON") from exc
    required_top = {
        "schema_version", "dataset", "claim_unit", "target", "feature_schema",
        "window_shape", "minimum_recording_coverage", "primary_tasks", "counts",
        "provenance", "records",
    }
    if not isinstance(manifest, dict) or set(manifest) != required_top:
        raise ValueError("collection manifest top-level schema is not exact")
    expected_scalars = {
        "schema_version": "neuroface_clinical23_v2_windows_v1",
        "dataset": "Toronto_NeuroFace_v1",
        "claim_unit": "participant",
        "target": "neurological_orofacial_impairment_vs_healthy_control",
        "feature_schema": DYNAMIC_FEATURE_SCHEMA,
        "window_shape": [4, 32, 95],
        "minimum_recording_coverage": 0.9,
        "primary_tasks": ["NSM_KISS", "NSM_OPEN", "NSM_SPREAD"],
        "counts": {
            "source_records": 261,
            "retained": 231,
            "excluded": 30,
            "participants": 36,
            "primary_complete_participants": 36,
        },
    }
    if any(manifest[name] != value for name, value in expected_scalars.items()):
        raise ValueError("collection manifest frozen values differ")
    provenance = manifest["provenance"]
    if (
        not isinstance(provenance, dict)
        or set(provenance) != {
            "private_manifest_sha256", "mediapipe_model_sha256",
            "implementation_sha256",
        }
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in provenance.values()
        )
    ):
        raise ValueError("collection provenance is malformed")
    records = manifest["records"]
    if not isinstance(records, list) or len(records) != 261:
        raise ValueError("collection manifest must contain exactly 261 records")
    seen: set[str] = set()
    retained = 0
    excluded = 0
    matched: list[dict[str, object]] = []
    ordered_ids: list[str] = []
    for raw in records:
        if not isinstance(raw, dict):
            raise ValueError("collection records must be JSON objects")
        status = raw.get("status")
        expected_keys = (
            {
                "recording_id", "participant_id", "video_sha256", "cache_sha256",
                "coverage", "status",
            }
            if status == "retained"
            else {
                "recording_id", "participant_id", "video_sha256", "status",
                "exclusion_reason",
            }
        )
        if set(raw) != expected_keys:
            raise ValueError("collection record fields differ from the closed schema")
        recording_id = raw["recording_id"]
        participant_id = raw["participant_id"]
        video_sha = raw["video_sha256"]
        if (
            not isinstance(recording_id, str)
            or _RECORDING_ID.fullmatch(recording_id) is None
            or recording_id in seen
            or not isinstance(participant_id, str)
            or _GROUP_ID.fullmatch(participant_id) is None
            or not isinstance(video_sha, str)
            or _SHA256.fullmatch(video_sha) is None
        ):
            raise ValueError("collection record identity is malformed or duplicated")
        seen.add(recording_id)
        ordered_ids.append(recording_id)
        if status == "retained":
            cache_sha = raw["cache_sha256"]
            coverage = raw["coverage"]
            if (
                not isinstance(cache_sha, str)
                or _SHA256.fullmatch(cache_sha) is None
                or isinstance(coverage, bool)
                or not isinstance(coverage, (int, float))
                or not 0.9 <= float(coverage) <= 1.0
            ):
                raise ValueError("retained collection record is malformed")
            retained += 1
        elif status == "excluded":
            if raw["exclusion_reason"] not in _ALLOWED_NEUROFACE_EXCLUSIONS:
                raise ValueError("excluded collection record requires a reason")
            excluded += 1
        else:
            raise ValueError("collection record status is invalid")
        if recording_id == binding.recording_id:
            matched.append(raw)
    if ordered_ids != sorted(ordered_ids) or retained != 231 or excluded != 30:
        raise ValueError("collection inventory order or flow counts differ")
    if len(matched) != 1 or matched[0].get("status") != "retained":
        raise ValueError("binding must resolve to one retained collection record")
    expected_cache_sha = matched[0]["cache_sha256"]
    if hashlib.sha256(cache_payload).hexdigest() != expected_cache_sha:
        raise ValueError("cache payload differs from its collection commitment")
    return matched[0]


def neuroface_action_capacity_feature_vector(
    binding: NeuroFaceTaskBinding,
    cache_payload: bytes,
    collection_manifest_bytes: bytes,
    *,
    decoded_recording_sha256: str,
) -> np.ndarray:
    """Build 18D from one collection-pinned immutable NeuroFace cache payload."""
    if not isinstance(binding, NeuroFaceTaskBinding):
        raise ValueError("a validated NeuroFace task binding is required")
    if type(cache_payload) is not bytes or not (
        0 < len(cache_payload) <= _MAX_CACHE_PAYLOAD_BYTES
    ):
        raise ValueError("cache payload must be bounded exact bytes")
    try:
        revalidated = validate_neuroface_task_binding(
            binding.manifest_bytes,
            recording_id=binding.recording_id,
            decoded_recording_sha256=decoded_recording_sha256,
        )
    except AttributeError as exc:
        raise ValueError("malformed NeuroFace task binding") from exc
    if revalidated != binding:
        raise ValueError("NeuroFace task binding changed after validation")
    collection_row = _bound_collection_row(
        collection_manifest_bytes, cache_payload, binding
    )

    recording = load_dynamic_landmark_recording_bytes(cache_payload)
    if (
        recording.recording_id != binding.recording_id
        or recording.recording_id != collection_row["recording_id"]
        or recording.group_id != collection_row["participant_id"]
        or recording.source_sha256 != binding.recording_sha256
        or recording.source_sha256 != decoded_recording_sha256
        or recording.source_sha256 != collection_row["video_sha256"]
    ):
        raise ValueError("NeuroFace cache identity differs from bound authorities")
    if (
        recording.feature_schema != DYNAMIC_FEATURE_SCHEMA
        or type(recording.feature_names) is not tuple
        or recording.feature_names != DYNAMIC_FEATURE_NAMES
        or recording.timestamp_unit != "seconds"
    ):
        raise ValueError("NeuroFace cache schema or timestamp unit differs")
    expected_indices = np.stack([
        np.arange(start, start + 32, dtype=np.int64)
        for start in deterministic_window_starts(recording.source_frame_count)
    ])
    observed_indices = np.asarray(recording.source_frame_indices)
    if not np.array_equal(observed_indices, expected_indices):
        raise ValueError("NeuroFace cache windows differ from the frozen source count")
    times = np.asarray(recording.timestamps, dtype=np.float64)
    periods = np.diff(times, axis=1) / np.diff(
        observed_indices, axis=1
    ).astype(np.float64)
    if not np.isfinite(periods).all() or np.any(periods <= 0.0):
        raise ValueError("NeuroFace per-frame seconds must be finite and positive")
    frame_period = float(np.median(periods))
    tolerance = max(1e-9, abs(frame_period) * 1e-6)
    if (
        not np.all(np.abs(periods - frame_period) <= tolerance)
        or frame_period < 1.0 / 240.0
        or frame_period > 1.0
    ):
        raise ValueError("NeuroFace frame period is not stable plausible seconds")

    raw, summary_mask, summary_times, summary_indices = _validated_raw_recording(
        recording.features,
        recording.valid_mask,
        recording.timestamps,
        observed_indices,
    )
    vector_110d = _candidate_feature_vector(
        LANDMARK_MI_110D,
        raw,
        summary_mask,
        summary_times,
        summary_indices,
    )
    if (
        vector_110d.shape != (len(_LANDMARK_MI_110D_SOURCE_NAMES),)
        or vector_110d.dtype != np.dtype(np.float64)
        or not np.isfinite(vector_110d).all()
    ):
        raise AssertionError(
            "internal NeuroFace summary drifted from finite float64 110D"
        )
    result = vector_110d[np.asarray(_SOURCE_INDICES, dtype=np.int64)].copy()
    if result.shape != (ACTION_CAPACITY_DIM,) or not np.isfinite(result).all():
        raise ValueError("NeuroFace action capacity projection is invalid")
    return result


def mirror_action_capacity_features(features: np.ndarray) -> np.ndarray:
    """Swap capture-side mouth-corner triplets; keep global signals fixed."""
    values = _validate_float64_vector(
        features,
        dimension=ACTION_CAPACITY_DIM,
        label="action_capacity_features",
    )
    result = values[np.asarray(_MIRROR_INDICES, dtype=np.int64)].copy()
    if not np.isfinite(result).all():
        raise ValueError("action-capacity mirror produced nonfinite values")
    return result


__all__ = [
    "ACTION_CAPACITY_DIM",
    "action_capacity_feature_names",
    "faces_action_capacity_feature_vector",
    "mirror_action_capacity_features",
    "neuroface_action_capacity_feature_vector",
]
