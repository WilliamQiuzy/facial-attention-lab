"""Exogenously anchored action segmentation for FACES and NeuroFace v1.

Action identity and timing come from capture metadata, audited annotations, or
an authenticated recording-level task label. Geometry can describe motion
inside an anchored interval, but it never creates, moves, or deletes one.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..datasets.dynamic_landmark import deterministic_window_starts


FACES_TIMELINE_SCHEMA = "faces-action-timeline/v1"
FACES_SCRIPT_VERSION = "faces-script/24-004956-v1"
NEUROFACE_TASK_BINDING_SCHEMA = "neuroface-task-binding/v1"
PINNED_NEUROFACE_PRIVATE_MANIFEST_SHA256 = (
    "235d2af2f3f4507b4ec858ff8dd9ff949d7f19e0d3656cbf5dcc0218648da07b"
)
MAX_NEUROFACE_MANIFEST_BYTES = 8 * 1024 * 1024
FACES_ACTION_ORDER = (
    "neutral_repose",
    "eyebrow_raise",
    "gentle_eye_closure",
    "tight_eye_squeeze",
    "relaxed_smile",
    "lip_pucker",
    "lower_teeth_show",
    "reanimated_smile",
)
FACES_REQUIRED_ACTIONS = FACES_ACTION_ORDER[:7]
NEUROFACE_PRIMARY_TASKS = ("NSM_KISS", "NSM_OPEN", "NSM_SPREAD")
SAMPLES_PER_HOLD = 32
FACES_HOLD_MS = 3000
MIN_VALID_SAMPLES = 26
MAX_REUSE_SOURCE_FPS = 31.0 / 3.0
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_ACTION_STATUS = frozenset({"prompted", "completed"})


class TimingSource(str, Enum):
    CAPTURE_EVENT_LOG = "capture_event_log"
    AUDIO_FORCED_ALIGNMENT = "audio_forced_alignment"
    BLINDED_MANUAL = "blinded_manual"
    RECORDING_TASK_LABEL = "recording_task_label"


@dataclass(frozen=True)
class ActionTiming:
    action: str
    status: str
    prompt_start_ms: int
    hold_start_ms: int
    hold_end_ms: int
    completion_ms: int | None


@dataclass(frozen=True)
class FacesTimeline:
    schema_version: str
    script_version: str
    recording_sha256: str
    timing_source: TimingSource
    recording_duration_ms: int
    actions: tuple[ActionTiming, ...]


@dataclass(frozen=True)
class AnchoredActionSegment:
    action: str
    timing_source: TimingSource
    prompted: bool
    observed_motion: bool
    tracking_adequate: bool
    eligible: bool
    frame_positions: np.ndarray
    source_frame_indices: np.ndarray
    frame_timestamps_ms: np.ndarray
    valid_mask: np.ndarray
    motion_curve: np.ndarray
    landmark_features: np.ndarray
    channel_finite_support: np.ndarray


@dataclass(frozen=True)
class NeuroFaceTaskSegment:
    task_label: str
    timing_source: TimingSource
    scope: str
    prompted: bool
    frame_positions: np.ndarray
    source_frame_indices: np.ndarray
    frame_timestamps_ms: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class NeuroFaceTaskBinding:
    schema_version: str
    recording_id: str
    task_label: str
    recording_sha256: str
    manifest_sha256: str
    manifest_bytes: bytes


def _integer_ms(value: object, name: str, *, positive: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer number of milliseconds")
    result = int(value)
    if result < 0 or (positive and result == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _exact_keys(
    value: object,
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    keys = frozenset(value)
    missing = required - keys
    unknown = keys - required - optional
    if missing or unknown:
        raise ValueError(
            f"{name} fields differ from contract; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    return value


def validate_faces_sidecar(sidecar: Mapping[str, object]) -> FacesTimeline:
    """Validate and freeze one recording-relative FACES action timeline."""
    row = _exact_keys(
        sidecar,
        required=frozenset({
            "schema_version", "script_version", "recording_sha256",
            "timing_source", "recording_duration_ms", "actions",
        }),
        name="FACES sidecar",
    )
    if row["schema_version"] != FACES_TIMELINE_SCHEMA:
        raise ValueError("unsupported FACES timeline schema")
    if row["script_version"] != FACES_SCRIPT_VERSION:
        raise ValueError("unsupported FACES script version")
    digest = row["recording_sha256"]
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError("recording_sha256 must be 64 lowercase hexadecimal characters")
    try:
        source = TimingSource(row["timing_source"])
    except (TypeError, ValueError) as exc:
        raise ValueError("unknown timing_source") from exc
    if source is TimingSource.RECORDING_TASK_LABEL:
        raise ValueError("recording_task_label is only valid for NeuroFace")
    duration = _integer_ms(
        row["recording_duration_ms"], "recording_duration_ms", positive=True
    )
    raw_actions = row["actions"]
    if not isinstance(raw_actions, Sequence) or isinstance(
        raw_actions, (str, bytes, bytearray)
    ):
        raise ValueError("actions must be an ordered sequence")
    if len(raw_actions) not in {len(FACES_REQUIRED_ACTIONS), len(FACES_ACTION_ORDER)}:
        raise ValueError("FACES sidecar requires seven actions and permits one optional action")

    actions: list[ActionTiming] = []
    for index, raw in enumerate(raw_actions):
        item = _exact_keys(
            raw,
            required=frozenset({
                "action", "status", "prompt_start_ms", "hold_start_ms",
                "hold_end_ms",
            }),
            optional=frozenset({"completion_ms"}),
            name=f"actions[{index}]",
        )
        action = item["action"]
        status = item["status"]
        if not isinstance(action, str) or not isinstance(status, str):
            raise ValueError("action and status must be strings")
        if status not in _ALLOWED_ACTION_STATUS:
            raise ValueError("action status must prove that the action was prompted")
        prompt = _integer_ms(item["prompt_start_ms"], "prompt_start_ms")
        hold_start = _integer_ms(item["hold_start_ms"], "hold_start_ms")
        hold_end = _integer_ms(item["hold_end_ms"], "hold_end_ms")
        if not (prompt <= hold_start < hold_end <= duration):
            raise ValueError("action bounds must be ordered inside the recording")
        if hold_end - hold_start != FACES_HOLD_MS:
            raise ValueError("every FACES hold must last exactly 3000 ms")
        completion_raw = item.get("completion_ms")
        completion = None
        if completion_raw is not None:
            completion = _integer_ms(completion_raw, "completion_ms")
            if not hold_end <= completion <= duration:
                raise ValueError("completion_ms must follow the hold inside the recording")
        actions.append(ActionTiming(
            action=action,
            status=status,
            prompt_start_ms=prompt,
            hold_start_ms=hold_start,
            hold_end_ms=hold_end,
            completion_ms=completion,
        ))

    observed_order = tuple(item.action for item in actions)
    expected_order = FACES_ACTION_ORDER[:len(actions)]
    if observed_order != expected_order:
        raise ValueError("FACES actions must be unique and in the locked script order")
    previous_end = 0
    for item in actions:
        if item.prompt_start_ms < previous_end:
            raise ValueError("FACES action intervals must be monotone and non-overlapping")
        previous_end = (
            item.completion_ms if item.completion_ms is not None else item.hold_end_ms
        )
    return FacesTimeline(
        schema_version=FACES_TIMELINE_SCHEMA,
        script_version=FACES_SCRIPT_VERSION,
        recording_sha256=digest,
        timing_source=source,
        recording_duration_ms=duration,
        actions=tuple(actions),
    )


def _revalidate_faces_timeline(timeline: FacesTimeline | None) -> FacesTimeline:
    if not isinstance(timeline, FacesTimeline):
        raise ValueError("a validated external action timeline is required")
    if type(timeline.actions) is not tuple or len(timeline.actions) not in {7, 8}:
        raise ValueError("FACES timeline actions must be an exact bounded tuple")
    if any(type(item) is not ActionTiming for item in timeline.actions):
        raise ValueError("FACES timeline actions must contain exact ActionTiming values")
    try:
        actions = [{
            "action": item.action,
            "status": item.status,
            "prompt_start_ms": item.prompt_start_ms,
            "hold_start_ms": item.hold_start_ms,
            "hold_end_ms": item.hold_end_ms,
            **({"completion_ms": item.completion_ms}
               if item.completion_ms is not None else {}),
        } for item in timeline.actions]
        source = (
            timeline.timing_source.value
            if isinstance(timeline.timing_source, TimingSource)
            else timeline.timing_source
        )
        serialized = {
            "schema_version": timeline.schema_version,
            "script_version": timeline.script_version,
            "recording_sha256": timeline.recording_sha256,
            "timing_source": source,
            "recording_duration_ms": timeline.recording_duration_ms,
            "actions": actions,
        }
    except (AttributeError, TypeError) as exc:
        raise ValueError("malformed FACES timeline object") from exc
    return validate_faces_sidecar(serialized)


def _unique_json_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"NeuroFace manifest contains duplicate JSON key {key!r}")
        output[key] = value
    return output


def _canonical_private_manifest_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _manifest_int(value: object, name: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact JSON integer")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _validated_neuroface_manifest_records(
    manifest: object,
) -> tuple[dict[str, object], ...]:
    top = _exact_keys(
        manifest,
        required=frozenset({
            "schema_version", "dataset", "claim_unit", "target",
            "primary_tasks", "counts", "archives", "slp_workbook_sha256",
            "participants", "records",
        }),
        name="NeuroFace private manifest",
    )
    expected_scalars = {
        "schema_version": "neuroface_external_private_manifest_v1",
        "dataset": "Toronto_NeuroFace_v1",
        "claim_unit": "participant",
        "target": "neurological_orofacial_impairment_vs_healthy_control",
    }
    if any(top[name] != expected for name, expected in expected_scalars.items()):
        raise ValueError("NeuroFace private manifest identity differs from the freeze")
    if top["primary_tasks"] != list(NEUROFACE_PRIMARY_TASKS):
        raise ValueError("NeuroFace primary task order differs from the freeze")
    expected_by_cohort = {
        "als": {"participants": 11, "videos": 76, "annotated_frames": 920},
        "healthy_control": {
            "participants": 11, "videos": 80, "annotated_frames": 1015,
        },
        "post_stroke": {
            "participants": 14, "videos": 105, "annotated_frames": 1371,
        },
    }
    expected_counts = {
        "participants": 36,
        "videos": 261,
        "annotated_frames": 3306,
        "affected_participants": 25,
        "unaffected_participants": 11,
        "primary_complete_participants": 36,
        "by_cohort": expected_by_cohort,
    }
    counts = top["counts"]
    if not isinstance(counts, dict) or frozenset(counts) != frozenset(expected_counts):
        raise ValueError("NeuroFace manifest count fields differ from the freeze")
    for name, expected in expected_counts.items():
        if name == "by_cohort":
            if counts[name] != expected:
                raise ValueError("NeuroFace cohort counts differ from the freeze")
            for cohort_values in counts[name].values():
                if any(type(item) is not int for item in cohort_values.values()):
                    raise ValueError("NeuroFace cohort counts must be exact integers")
        elif _manifest_int(counts[name], f"counts.{name}") != expected:
            raise ValueError("NeuroFace aggregate counts differ from the freeze")
    if not isinstance(top["archives"], dict) or not isinstance(
        top["slp_workbook_sha256"], dict
    ):
        raise ValueError("NeuroFace archive commitments must be JSON objects")

    participants = top["participants"]
    records = top["records"]
    if not isinstance(participants, list) or len(participants) != 36:
        raise ValueError("NeuroFace manifest must contain 36 participants")
    if not isinstance(records, list) or len(records) != 261:
        raise ValueError("NeuroFace manifest must contain 261 recordings")
    participant_ids: set[str] = set()
    for index, raw in enumerate(participants):
        item = _exact_keys(
            raw,
            required=frozenset({"participant_id", "cohort", "binary_label"}),
            name=f"participants[{index}]",
        )
        participant_id = item["participant_id"]
        if (
            not isinstance(participant_id, str)
            or re.fullmatch(r"grp_[0-9a-f]{64}", participant_id) is None
            or participant_id in participant_ids
        ):
            raise ValueError("participant identifiers must be unique opaque hashes")
        if item["cohort"] not in expected_by_cohort or item["binary_label"] not in {
            "affected", "unaffected",
        }:
            raise ValueError("participant cohort/label is invalid")
        participant_ids.add(participant_id)

    record_keys = frozenset({
        "recording_id", "participant_id", "cohort", "binary_label", "session",
        "task", "video_archive_id", "video_sha256", "video_size_bytes",
        "landmark_archive_id", "landmark_sha256", "annotated_frames", "slp_scores",
    })
    recording_ids: set[str] = set()
    video_hashes: set[str] = set()
    checked: list[dict[str, object]] = []
    for index, raw in enumerate(records):
        item = dict(_exact_keys(
            raw, required=record_keys, name=f"records[{index}]"
        ))
        recording_id = item["recording_id"]
        video_sha = item["video_sha256"]
        if (
            not isinstance(recording_id, str)
            or re.fullmatch(r"rec_[0-9a-f]{64}", recording_id) is None
            or recording_id in recording_ids
        ):
            raise ValueError("recording identifiers must be unique opaque hashes")
        if (
            not isinstance(video_sha, str)
            or _SHA256.fullmatch(video_sha) is None
            or video_sha in video_hashes
        ):
            raise ValueError("video SHA-256 values must be canonical and unique")
        if item["participant_id"] not in participant_ids:
            raise ValueError("record references an unknown participant")
        if not isinstance(item["task"], str):
            raise ValueError("record task must be a string")
        _manifest_int(item["video_size_bytes"], "video_size_bytes", positive=True)
        _manifest_int(item["annotated_frames"], "annotated_frames", positive=True)
        if not isinstance(item["slp_scores"], dict):
            raise ValueError("record slp_scores must be an object")
        recording_ids.add(recording_id)
        video_hashes.add(video_sha)
        checked.append(item)
    return tuple(checked)


def validate_neuroface_task_binding(
    manifest_bytes: bytes,
    *,
    recording_id: str,
    decoded_recording_sha256: str,
) -> NeuroFaceTaskBinding:
    """Derive task identity from one exact record in the pinned private manifest."""
    if type(manifest_bytes) is not bytes:
        raise ValueError("NeuroFace binding requires exact private-manifest bytes")
    if not manifest_bytes or len(manifest_bytes) > MAX_NEUROFACE_MANIFEST_BYTES:
        raise ValueError("NeuroFace private manifest is empty or exceeds 8 MiB")
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha != PINNED_NEUROFACE_PRIVATE_MANIFEST_SHA256:
        raise ValueError("NeuroFace private manifest bytes differ from the frozen pin")
    try:
        decoded = manifest_bytes.decode("utf-8")
        manifest = json.loads(decoded, object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("NeuroFace private manifest is not strict UTF-8 JSON") from exc
    if _canonical_private_manifest_bytes(manifest) != manifest_bytes:
        raise ValueError("NeuroFace private manifest bytes are not canonical")
    records = _validated_neuroface_manifest_records(manifest)
    if (
        not isinstance(recording_id, str)
        or re.fullmatch(r"rec_[0-9a-f]{64}", recording_id) is None
    ):
        raise ValueError("recording_id must be an opaque canonical identifier")
    if (
        not isinstance(decoded_recording_sha256, str)
        or _SHA256.fullmatch(decoded_recording_sha256) is None
    ):
        raise ValueError("decoded_recording_sha256 must be canonical lowercase hex")
    matches = [row for row in records if row["recording_id"] == recording_id]
    if len(matches) != 1:
        raise ValueError("recording_id does not resolve uniquely in the pinned manifest")
    row = matches[0]
    if row["video_sha256"] != decoded_recording_sha256:
        raise ValueError("decoded recording bytes differ from the pinned manifest record")
    task_label = row["task"]
    if task_label not in NEUROFACE_PRIMARY_TASKS:
        raise ValueError("manifest-derived task is not a locked NeuroFace primary task")
    return NeuroFaceTaskBinding(
        schema_version=NEUROFACE_TASK_BINDING_SCHEMA,
        recording_id=recording_id,
        task_label=task_label,
        recording_sha256=decoded_recording_sha256,
        manifest_sha256=manifest_sha,
        manifest_bytes=manifest_bytes,
    )


def _revalidate_neuroface_binding(
    binding: NeuroFaceTaskBinding,
) -> NeuroFaceTaskBinding:
    if not isinstance(binding, NeuroFaceTaskBinding):
        raise ValueError("a validated NeuroFace task binding is required")
    try:
        validated = validate_neuroface_task_binding(
            binding.manifest_bytes,
            recording_id=binding.recording_id,
            decoded_recording_sha256=binding.recording_sha256,
        )
    except AttributeError as exc:
        raise ValueError("malformed NeuroFace task binding") from exc
    if validated != binding:
        raise ValueError("NeuroFace task binding was modified after validation")
    return validated


def _stream_arrays(
    frame_timestamps_ms: np.ndarray,
    source_frame_indices: np.ndarray,
    valid_mask: np.ndarray,
    motion_curve: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    timestamps = np.asarray(frame_timestamps_ms)
    indices = np.asarray(source_frame_indices)
    valid = np.asarray(valid_mask)
    if timestamps.ndim != 1 or timestamps.size < 2 or timestamps.dtype.kind not in {
        "f", "i", "u",
    } or not np.isfinite(timestamps).all():
        raise ValueError("frame_timestamps_ms must be a finite numeric vector")
    if not np.all(timestamps[1:] > timestamps[:-1]):
        raise ValueError("frame timestamps must increase strictly")
    if float(timestamps[0]) < 0:
        raise ValueError("recording-relative frame timestamps must be nonnegative")
    if indices.shape != timestamps.shape or indices.dtype.kind not in {"i", "u"}:
        raise ValueError("source_frame_indices must be an aligned integer vector")
    if indices.dtype.kind == "u" and int(np.max(indices)) > np.iinfo(np.int64).max:
        raise ValueError("source frame indices exceed the signed int64 cache contract")
    if not np.all(indices[1:] > indices[:-1]):
        raise ValueError("source frame indices must increase strictly")
    if int(indices[0]) < 0:
        raise ValueError("source frame indices must be nonnegative")
    if valid.shape != timestamps.shape or valid.dtype != np.dtype(bool):
        raise ValueError("valid_mask must be an aligned bool vector")
    motion = None
    if motion_curve is not None:
        motion = np.asarray(motion_curve)
        if motion.shape != timestamps.shape or motion.dtype.kind not in {"f", "i", "u"}:
            raise ValueError("motion_curve must be an aligned real numeric vector")
        if not np.isfinite(motion[valid]).all():
            raise ValueError("motion must be finite wherever landmark tracking is valid")
    return (
        timestamps.astype(np.float64, copy=False),
        indices.astype(np.int64, copy=False),
        valid,
        None if motion is None else motion.astype(np.float64, copy=False),
    )


def _landmark_array(
    landmark_features: np.ndarray | None,
    n_frames: int,
) -> np.ndarray:
    if landmark_features is None:
        raise ValueError("landmark_features are required for channel-support QC")
    features = np.asarray(landmark_features)
    if (
        features.ndim != 2
        or features.shape[0] != n_frames
        or features.shape[1] < 1
        or features.dtype.kind not in {"f", "i", "u"}
    ):
        raise ValueError(
            "landmark_features must be a real numeric (n_frames, n_channels) array"
        )
    return features.astype(np.float64, copy=False)


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value)
    return np.frombuffer(
        contiguous.tobytes(order="C"), dtype=contiguous.dtype
    ).reshape(contiguous.shape)


def _uniform_hold_positions(
    timestamps: np.ndarray,
    hold_start_ms: int,
    hold_end_ms: int,
) -> np.ndarray:
    inside = np.flatnonzero(
        (timestamps >= float(hold_start_ms)) & (timestamps <= float(hold_end_ms))
    )
    if inside.size == 0:
        raise ValueError("no decoded frames lie inside the anchored hold")
    targets = hold_start_ms + (
        np.arange(SAMPLES_PER_HOLD, dtype=np.float64) + 0.5
    ) * (hold_end_ms - hold_start_ms) / SAMPLES_PER_HOLD
    insertion = np.searchsorted(timestamps[inside], targets)
    right = np.clip(insertion, 0, inside.size - 1)
    left = np.clip(insertion - 1, 0, inside.size - 1)
    left_error = np.abs(timestamps[inside[left]] - targets)
    right_error = np.abs(timestamps[inside[right]] - targets)
    selected = inside[np.where(left_error <= right_error, left, right)]

    frame_period = float(np.median(np.diff(timestamps)))
    if not np.isfinite(frame_period) or frame_period <= 0:
        raise ValueError("decoded frame period is invalid")
    timestamp_error = np.abs(timestamps[selected] - targets)
    tolerance = frame_period / 2.0 + np.finfo(np.float64).eps * 32
    if np.any(timestamp_error > tolerance):
        raise ValueError("anchored sampling exceeds half one decoded frame")
    source_fps = 1000.0 / frame_period
    if np.unique(selected).size != SAMPLES_PER_HOLD and source_fps >= MAX_REUSE_SOURCE_FPS:
        raise ValueError("decoded frames may only be reused below 10.34 Hz")
    return selected.astype(np.int64, copy=False)


def segment_faces_action(
    timeline: FacesTimeline | None,
    action: str,
    frame_timestamps_ms: np.ndarray,
    source_frame_indices: np.ndarray,
    valid_mask: np.ndarray,
    motion_curve: np.ndarray,
    *,
    decoded_recording_sha256: str,
    decoded_duration_ms: int,
    landmark_features: np.ndarray | None,
    motion_threshold: float = 1e-6,
) -> AnchoredActionSegment:
    """Sample one externally anchored three-second hold at 32 time positions."""
    timeline = _revalidate_faces_timeline(timeline)
    if (
        not isinstance(decoded_recording_sha256, str)
        or _SHA256.fullmatch(decoded_recording_sha256) is None
    ):
        raise ValueError("decoded_recording_sha256 must be canonical lowercase hex")
    if decoded_recording_sha256 != timeline.recording_sha256:
        raise ValueError("decoded recording SHA-256 does not match the timeline")
    decoded_duration = _integer_ms(
        decoded_duration_ms, "decoded_duration_ms", positive=True
    )
    if decoded_duration != timeline.recording_duration_ms:
        raise ValueError("decoded recording duration does not match the timeline")
    if not isinstance(action, str):
        raise ValueError("action must be a string")
    if not np.isfinite(motion_threshold) or motion_threshold < 0:
        raise ValueError("motion_threshold must be finite and nonnegative")
    by_name = {item.action: item for item in timeline.actions}
    if action not in by_name:
        raise ValueError("action is absent from the validated timeline")
    timestamps, indices, valid, motion = _stream_arrays(
        frame_timestamps_ms, source_frame_indices, valid_mask, motion_curve
    )
    if float(timestamps[-1]) > float(decoded_duration):
        raise ValueError("decoded frame timestamps exceed the recording duration")
    landmark_values = _landmark_array(landmark_features, timestamps.size)
    assert motion is not None
    timing = by_name[action]
    positions = _uniform_hold_positions(
        timestamps, timing.hold_start_ms, timing.hold_end_ms
    )
    selected_valid = valid[positions].copy()
    selected_motion = motion[positions].copy()
    selected_landmarks = landmark_values[positions].copy()
    finite_support = np.sum(
        selected_valid[:, None] & np.isfinite(selected_landmarks), axis=0
    ).astype(np.int64, copy=False)
    finite_motion = selected_valid & np.isfinite(selected_motion)
    tracking_adequate = bool(
        int(selected_valid.sum()) >= MIN_VALID_SAMPLES
        and np.all(finite_support >= MIN_VALID_SAMPLES)
    )
    observed_motion = False
    if int(finite_motion.sum()) >= 2:
        observed_motion = bool(
            np.ptp(selected_motion[finite_motion]) > float(motion_threshold)
        )
    prompted = timing.status in _ALLOWED_ACTION_STATUS
    return AnchoredActionSegment(
        action=action,
        timing_source=timeline.timing_source,
        prompted=prompted,
        observed_motion=observed_motion,
        tracking_adequate=tracking_adequate,
        eligible=bool(prompted and tracking_adequate),
        frame_positions=_readonly_copy(positions),
        source_frame_indices=_readonly_copy(indices[positions]),
        frame_timestamps_ms=_readonly_copy(timestamps[positions]),
        valid_mask=_readonly_copy(selected_valid),
        motion_curve=_readonly_copy(selected_motion),
        landmark_features=_readonly_copy(selected_landmarks),
        channel_finite_support=_readonly_copy(finite_support),
    )


def segment_neuroface_recording_task(
    binding: NeuroFaceTaskBinding,
    frame_timestamps_ms: np.ndarray,
    source_frame_indices: np.ndarray,
    valid_mask: np.ndarray,
    *,
    decoded_recording_sha256: str,
    decoded_source_frame_count: int,
) -> NeuroFaceTaskSegment:
    """Return the frozen four-window cache positions for one named task file."""
    if isinstance(decoded_source_frame_count, (bool, np.bool_)) or not isinstance(
        decoded_source_frame_count, (int, np.integer)
    ):
        raise ValueError("decoded_source_frame_count must be a strict integer")
    source_frame_count = int(decoded_source_frame_count)
    if source_frame_count <= 0:
        raise ValueError("decoded_source_frame_count must be positive")
    binding = _revalidate_neuroface_binding(binding)
    if (
        not isinstance(decoded_recording_sha256, str)
        or _SHA256.fullmatch(decoded_recording_sha256) is None
    ):
        raise ValueError("decoded_recording_sha256 must be canonical lowercase hex")
    if decoded_recording_sha256 != binding.recording_sha256:
        raise ValueError("decoded recording SHA-256 does not match the manifest binding")
    timestamps, indices, valid, _motion = _stream_arrays(
        frame_timestamps_ms, source_frame_indices, valid_mask
    )
    if timestamps.size != source_frame_count:
        raise ValueError("decoded arrays do not cover the trusted source frame count")
    expected_indices = np.arange(source_frame_count, dtype=np.int64)
    if not np.array_equal(indices, expected_indices):
        raise ValueError("NeuroFace source indices must equal arange(source_frame_count)")
    starts = deterministic_window_starts(source_frame_count)
    positions = np.stack([
        np.arange(start, start + SAMPLES_PER_HOLD, dtype=np.int64)
        for start in starts
    ])
    return NeuroFaceTaskSegment(
        task_label=binding.task_label,
        timing_source=TimingSource.RECORDING_TASK_LABEL,
        scope="whole_recording",
        prompted=True,
        frame_positions=_readonly_copy(positions),
        source_frame_indices=_readonly_copy(indices[positions]),
        frame_timestamps_ms=_readonly_copy(timestamps[positions]),
        valid_mask=_readonly_copy(valid[positions]),
    )


__all__ = [
    "ActionTiming",
    "AnchoredActionSegment",
    "FACES_ACTION_ORDER",
    "FACES_HOLD_MS",
    "FACES_REQUIRED_ACTIONS",
    "FACES_SCRIPT_VERSION",
    "FACES_TIMELINE_SCHEMA",
    "FacesTimeline",
    "MIN_VALID_SAMPLES",
    "NEUROFACE_PRIMARY_TASKS",
    "NEUROFACE_TASK_BINDING_SCHEMA",
    "NeuroFaceTaskBinding",
    "NeuroFaceTaskSegment",
    "SAMPLES_PER_HOLD",
    "TimingSource",
    "segment_faces_action",
    "segment_neuroface_recording_task",
    "validate_faces_sidecar",
    "validate_neuroface_task_binding",
]
