"""Fail-closed FACES capture contracts for the Shared V9 video gateway."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping

import cv2
import numpy as np

from .dense_bilateral_action_v1 import normalize_dense_landmarks
from .clinical_landmarks import clinical_landmark_features
from .shared_clinical_tokens_v1 import dense_action_token_bag
from .script_action_segmentation_v1 import (
    FACES_ACTION_ORDER,
    FACES_SCRIPT_VERSION,
    FACES_TIMELINE_SCHEMA,
    FacesTimeline,
    segment_faces_action,
    validate_faces_sidecar,
)


CAPTURE_MANIFEST_SCHEMA = "faces-v9-capture-manifest/v1"
CAPTURE_PROTOCOL_VERSION = "FACES-v0.01"
GATEWAY_RESPONSE_SCHEMA = "facial-paralysis-shared-v9-inference/v2"
PREPROCESSING_VERSION = "faces-to-shared-v9/v1"
SHARED_V9_MODEL_ID = "broad_literature_shared_v9_blv9_009_ensemble"
SHARED_V9_CANDIDATE_ID = "BLV9-009"
SHARED_V9_MANIFEST_SHA256 = (
    "81e396954090a0da6b99519909c1af15b6df5d1585ba27a642539352fe0a0c64"
)
FACES_TO_V9_ACTIONS = (
    ("eyebrow_raise", "BROW_RAISE"),
    ("gentle_eye_closure", "EYE_GENTLE"),
    ("tight_eye_squeeze", "EYE_FORCEFUL"),
    ("relaxed_smile", "SMILE_GENTLE"),
    ("lip_pucker", "LIP_PUCKER"),
    ("lower_teeth_show", "SHOW_BOTTOM_TEETH"),
    ("reanimated_smile", "SMILE_FULL"),
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCES = frozenset({"browser-camera", "livelink-upload"})
_VIDEO_SUFFIXES = frozenset({".mov", ".mp4", ".m4v", ".avi", ".webm"})
MAX_VIDEO_BYTES = 512 * 1024 * 1024
MAX_VIDEO_DURATION_MS = 180_000
MAX_VIDEO_PIXELS = 3840 * 2160
ACTION_VOCAB = (
    "FREE_EARLY", "FREE_MID_EARLY", "FREE_MID_LATE", "FREE_LATE",
    "LIP_PUCKER", "MOUTH_OPEN", "SMILE_SPREAD", "BROW_RAISE",
    "EYE_GENTLE", "EYE_FORCEFUL", "SMILE_GENTLE", "SMILE_FULL",
    "SHOW_BOTTOM_TEETH",
)


class CaptureTrackingError(ValueError):
    """Non-identifying action-level evidence for an insufficient face track."""

    def __init__(
        self,
        action: str,
        *,
        valid_samples: int,
        required_samples: int = 26,
    ):
        if (
            action not in FACES_ACTION_ORDER
            or type(valid_samples) is not int
            or not 0 <= valid_samples <= 32
            or type(required_samples) is not int
            or not 1 <= required_samples <= 32
        ):
            raise ValueError("tracking failure evidence differs from the closed contract")
        self.action = action
        self.valid_samples = valid_samples
        self.required_samples = required_samples
        super().__init__(
            f"FACES action {action} has {valid_samples} of 32 valid paired samples; "
            f"{required_samples} required"
        )


class CaptureTimingError(ValueError):
    """Non-identifying numeric evidence for a decoded/timeline clock mismatch."""

    _REASONS = frozenset({
        "recording_too_long",
        "hold_sampling_incomplete",
        "hold_sampling_precision",
        "incomplete_final_hold",
        "timeline_duration_drift",
    })

    def __init__(
        self,
        reason: str,
        *,
        decoded_duration_ms: int,
        timeline_duration_ms: int,
        last_hold_ms: int,
        source_fps: float,
        decoded_frame_count: int,
        tolerance_ms: int,
    ):
        if (
            reason not in self._REASONS
            or type(decoded_duration_ms) is not int
            or decoded_duration_ms < 1
            or type(timeline_duration_ms) is not int
            or timeline_duration_ms < 1
            or type(last_hold_ms) is not int
            or last_hold_ms < 1
            or not math.isfinite(source_fps)
            or source_fps <= 0
            or type(decoded_frame_count) is not int
            or decoded_frame_count < 1
            or type(tolerance_ms) is not int
            or tolerance_ms < 1
        ):
            raise ValueError("timing failure evidence differs from the closed contract")
        self.reason = reason
        self.decoded_duration_ms = decoded_duration_ms
        self.timeline_duration_ms = timeline_duration_ms
        self.last_hold_ms = last_hold_ms
        self.source_fps = float(source_fps)
        self.decoded_frame_count = decoded_frame_count
        self.tolerance_ms = tolerance_ms
        super().__init__("decoded duration contradicts the FACES timeline")


@dataclass(frozen=True)
class CaptureManifest:
    schema_version: str
    protocol_version: str
    recording_source: str
    video_sha256: str
    reanimated_smile_applicable: bool


@dataclass(frozen=True)
class CaptureEvidence:
    manifest: CaptureManifest
    timeline: FacesTimeline
    video_sha256: str


@dataclass(frozen=True)
class PreparedV9Request:
    arrays: dict[str, np.ndarray]
    valid_samples_per_action: tuple[int, ...]
    descriptive_evidence_per_action: tuple["DescriptiveActionEvidence", ...]


@dataclass(frozen=True)
class DescriptiveObservation:
    metric: str
    value: float


@dataclass(frozen=True)
class DescriptiveActionEvidence:
    action_id: str
    context_frame_ms: int
    observations: tuple[DescriptiveObservation, ...]


@dataclass(frozen=True)
class DecodedCaptureSamples:
    frames: tuple[np.ndarray, ...]
    frame_timestamps_ms: np.ndarray
    source_frame_indices: np.ndarray
    source_fps: float
    decoded_duration_ms: int


@dataclass(frozen=True)
class PairedMeshStream:
    frame_timestamps_ms: np.ndarray
    source_frame_indices: np.ndarray
    original_meshes: np.ndarray
    mirrored_meshes: np.ndarray
    pair_valid_mask: np.ndarray
    source_fps: float
    decoded_duration_ms: int


def _unique_object(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("JSON contains a duplicate key")
        output[key] = value
    return output


def _json_bytes(payload: bytes, name: str) -> dict[str, object]:
    if type(payload) is not bytes or not payload or len(payload) > 256 * 1024:
        raise ValueError(f"{name} must be nonempty bounded exact bytes")
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    return value


def _capture_manifest(payload: bytes) -> CaptureManifest:
    value = _json_bytes(payload, "capture manifest")
    expected = {
        "schema_version", "protocol_version", "recording_source",
        "video_sha256", "reanimated_smile_applicable",
    }
    if set(value) != expected:
        raise ValueError("capture manifest fields differ from the closed schema")
    if (
        value["schema_version"] != CAPTURE_MANIFEST_SCHEMA
        or value["protocol_version"] != CAPTURE_PROTOCOL_VERSION
        or value["recording_source"] not in _SOURCES
        or type(value["video_sha256"]) is not str
        or _SHA256.fullmatch(value["video_sha256"]) is None
        or type(value["reanimated_smile_applicable"]) is not bool
    ):
        raise ValueError("capture manifest values differ from the frozen contract")
    return CaptureManifest(
        schema_version=CAPTURE_MANIFEST_SCHEMA,
        protocol_version=CAPTURE_PROTOCOL_VERSION,
        recording_source=str(value["recording_source"]),
        video_sha256=str(value["video_sha256"]),
        reanimated_smile_applicable=bool(value["reanimated_smile_applicable"]),
    )


def parse_capture_evidence(
    video_payload: bytes,
    manifest_payload: bytes,
    timeline_payload: bytes,
) -> CaptureEvidence:
    """Bind exact video bytes to one capture manifest and external timeline."""
    if type(video_payload) is not bytes or not video_payload:
        raise ValueError("video must be nonempty exact bytes")
    digest = hashlib.sha256(video_payload).hexdigest()
    manifest = _capture_manifest(manifest_payload)
    timeline_value = _json_bytes(timeline_payload, "action timeline")
    timeline = validate_faces_sidecar(timeline_value)
    if manifest.video_sha256 != digest or timeline.recording_sha256 != digest:
        raise ValueError("video digest differs from capture evidence")
    expected_actions = len(FACES_ACTION_ORDER) if manifest.reanimated_smile_applicable else 7
    if len(timeline.actions) != expected_actions:
        raise ValueError("timeline action count contradicts the capture manifest")
    expected_order = (
        FACES_ACTION_ORDER
        if manifest.reanimated_smile_applicable
        else FACES_ACTION_ORDER[:-1]
    )
    if (
        timeline.schema_version != FACES_TIMELINE_SCHEMA
        or timeline.script_version != FACES_SCRIPT_VERSION
        or tuple(item.action for item in timeline.actions) != expected_order
    ):
        raise ValueError("timeline identity differs from the V9 FACES contract")
    return CaptureEvidence(manifest=manifest, timeline=timeline, video_sha256=digest)


def _present_v9_actions(evidence: CaptureEvidence) -> tuple[tuple[str, str], ...]:
    if not isinstance(evidence, CaptureEvidence):
        raise ValueError("validated capture evidence is required")
    return (
        FACES_TO_V9_ACTIONS
        if evidence.manifest.reanimated_smile_applicable
        else FACES_TO_V9_ACTIONS[:-1]
    )


def _immutable(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def _write_exact_file(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        written = 0
        while written < len(payload):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("short write while staging request video")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _timeline_targets(timeline: FacesTimeline) -> np.ndarray:
    targets = np.concatenate([
        timing.hold_start_ms
        + (np.arange(32, dtype=np.float64) + 0.5)
        * (timing.hold_end_ms - timing.hold_start_ms)
        / 32.0
        for timing in timeline.actions
    ])
    if targets.size not in {7 * 32, 8 * 32} or not np.all(targets[1:] > targets[:-1]):
        raise ValueError("FACES target sample times are not strictly ordered")
    return targets


def decode_capture_samples(
    video_payload: bytes,
    timeline: FacesTimeline,
    *,
    filename: str,
) -> DecodedCaptureSamples:
    """Decode only the frames nearest externally anchored hold positions."""
    if (
        type(video_payload) is not bytes
        or not video_payload
        or len(video_payload) > MAX_VIDEO_BYTES
        or type(filename) is not str
        or Path(filename).name != filename
    ):
        raise ValueError("video payload or filename differs from the bounded contract")
    suffix = Path(filename).suffix.casefold()
    if suffix not in _VIDEO_SUFFIXES:
        raise ValueError("video container extension is unsupported")
    if not isinstance(timeline, FacesTimeline):
        raise ValueError("validated FACES timeline is required")
    targets = _timeline_targets(timeline)
    with tempfile.TemporaryDirectory(prefix="faces-v9-video-") as temporary:
        os.chmod(temporary, 0o700)
        path = Path(temporary) / f"capture{suffix}"
        _write_exact_file(path, video_payload)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError("video cannot be decoded")
        try:
            reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
            width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            if width < 2 or height < 2 or width * height > MAX_VIDEO_PIXELS:
                raise ValueError("video dimensions are outside the gateway bounds")
            selected_frames: list[np.ndarray] = []
            selected_times: list[float] = []
            selected_indices: list[int] = []
            target_index = 0
            previous: tuple[int, float, np.ndarray] | None = None
            decoded_count = 0
            clock_mode: str | None = None
            position_origin: float | None = None
            previous_position: float | None = None
            last_timestamp = 0.0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if (
                    type(frame) is not np.ndarray
                    or frame.dtype != np.dtype(np.uint8)
                    or frame.shape != (height, width, 3)
                ):
                    raise ValueError("decoded frame dimensions changed inside the recording")
                raw_position = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                if decoded_count == 0:
                    if math.isfinite(raw_position) and raw_position >= 0:
                        position_origin = raw_position
                        previous_position = raw_position
                    timestamp = 0.0
                elif clock_mode is None:
                    if (
                        position_origin is not None
                        and math.isfinite(raw_position)
                        and raw_position > position_origin
                    ):
                        clock_mode = "container"
                        timestamp = raw_position - position_origin
                        previous_position = raw_position
                    else:
                        if not math.isfinite(reported_fps) or reported_fps <= 0:
                            raise ValueError("video frame rate is below the gateway minimum")
                        clock_mode = "nominal"
                        timestamp = decoded_count * 1000.0 / reported_fps
                elif clock_mode == "container":
                    if (
                        not math.isfinite(raw_position)
                        or previous_position is None
                        or raw_position <= previous_position
                    ):
                        raise ValueError("video container timestamps are not strictly increasing")
                    timestamp = raw_position - position_origin
                    previous_position = raw_position
                else:
                    timestamp = decoded_count * 1000.0 / reported_fps
                current = (decoded_count, timestamp, frame)
                while target_index < targets.size and targets[target_index] <= timestamp:
                    target = float(targets[target_index])
                    choices = (current,) if previous is None else (previous, current)
                    chosen = min(choices, key=lambda item: (abs(item[1] - target), item[0]))
                    if selected_indices and chosen[0] <= selected_indices[-1]:
                        raise ValueError("video frame rate cannot provide unique hold samples")
                    selected_indices.append(chosen[0])
                    selected_times.append(chosen[1])
                    selected_frames.append(_immutable(chosen[2]))
                    target_index += 1
                previous = current
                last_timestamp = timestamp
                decoded_count += 1
        finally:
            capture.release()
    if decoded_count < 1:
        raise ValueError("decoded recording does not cover every FACES hold")
    if decoded_count == 1 or clock_mode != "container":
        fps = reported_fps
    else:
        fps = (decoded_count - 1) * 1000.0 / last_timestamp
    if not math.isfinite(fps) or fps < 32.0 / 3.0:
        raise ValueError("video frame rate is below the gateway minimum")
    if previous is not None:
        while target_index < targets.size:
            target = float(targets[target_index])
            if abs(previous[1] - target) > 500.0 / fps:
                break
            if selected_indices and previous[0] <= selected_indices[-1]:
                break
            selected_indices.append(previous[0])
            selected_times.append(previous[1])
            selected_frames.append(_immutable(previous[2]))
            target_index += 1
    duration_ms = int(round(last_timestamp + 1000.0 / fps))
    last_hold = timeline.actions[-1].hold_end_ms
    tolerance_ms = max(250, int(math.ceil(2000.0 / fps)))
    if target_index != targets.size:
        timing_reason = "hold_sampling_incomplete"
    elif duration_ms > MAX_VIDEO_DURATION_MS:
        timing_reason = "recording_too_long"
    elif last_hold > duration_ms + tolerance_ms:
        timing_reason = "incomplete_final_hold"
    elif abs(duration_ms - timeline.recording_duration_ms) > 1_500:
        timing_reason = "timeline_duration_drift"
    else:
        timing_reason = None
    if timing_reason is not None:
        raise CaptureTimingError(
            timing_reason,
            decoded_duration_ms=duration_ms,
            timeline_duration_ms=timeline.recording_duration_ms,
            last_hold_ms=last_hold,
            source_fps=fps,
            decoded_frame_count=decoded_count,
            tolerance_ms=tolerance_ms,
        )
    sampled_times = np.asarray(selected_times, dtype=np.float64)
    target_error = np.abs(sampled_times - targets)
    target_tolerance_ms = min(
        (timing.hold_end_ms - timing.hold_start_ms) / 64.0
        for timing in timeline.actions
    )
    if np.any(target_error > target_tolerance_ms + np.finfo(np.float64).eps * 32):
        raise CaptureTimingError(
            "hold_sampling_precision",
            decoded_duration_ms=duration_ms,
            timeline_duration_ms=timeline.recording_duration_ms,
            last_hold_ms=last_hold,
            source_fps=fps,
            decoded_frame_count=decoded_count,
            tolerance_ms=max(1, int(math.ceil(target_tolerance_ms))),
        )
    return DecodedCaptureSamples(
        frames=tuple(selected_frames),
        frame_timestamps_ms=_immutable(sampled_times),
        source_frame_indices=_immutable(np.asarray(selected_indices, dtype=np.int64)),
        source_fps=fps,
        decoded_duration_ms=duration_ms,
    )


def extract_paired_meshes(
    decoded: DecodedCaptureSamples,
    detect_mesh,
) -> PairedMeshStream:
    """Run MediaPipe on each original frame and actual horizontal image flip."""
    if not isinstance(decoded, DecodedCaptureSamples) or not callable(detect_mesh):
        raise ValueError("decoded samples and a mesh detector are required")
    original = np.full((len(decoded.frames), 478, 3), np.nan, dtype=np.float64)
    mirrored = np.full_like(original, np.nan)
    valid = np.zeros(len(decoded.frames), dtype=bool)
    for index, frame in enumerate(decoded.frames):
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        flipped_rgb = np.ascontiguousarray(rgb[:, ::-1])
        first_raw = detect_mesh(rgb)
        second_raw = detect_mesh(flipped_rgb)
        if first_raw is None or second_raw is None:
            continue
        original[index] = normalize_dense_landmarks(
            first_raw, image_width=frame.shape[1], image_height=frame.shape[0]
        )
        mirrored[index] = normalize_dense_landmarks(
            second_raw, image_width=frame.shape[1], image_height=frame.shape[0]
        )
        valid[index] = True
    return PairedMeshStream(
        frame_timestamps_ms=decoded.frame_timestamps_ms,
        source_frame_indices=decoded.source_frame_indices,
        original_meshes=_immutable(original),
        mirrored_meshes=_immutable(mirrored),
        pair_valid_mask=_immutable(valid),
        source_fps=decoded.source_fps,
        decoded_duration_ms=decoded.decoded_duration_ms,
    )


def _mesh_stream(
    frame_timestamps_ms: np.ndarray,
    source_frame_indices: np.ndarray,
    original_meshes: np.ndarray,
    mirrored_meshes: np.ndarray,
    pair_valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    timestamps = np.asarray(frame_timestamps_ms)
    indices = np.asarray(source_frame_indices)
    original = np.asarray(original_meshes)
    mirrored = np.asarray(mirrored_meshes)
    valid = np.asarray(pair_valid_mask)
    if (
        timestamps.ndim != 1
        or timestamps.size < 32
        or timestamps.dtype.kind not in {"f", "i", "u"}
        or not np.isfinite(timestamps).all()
        or not np.all(timestamps[1:] > timestamps[:-1])
        or indices.shape != timestamps.shape
        or indices.dtype != np.dtype(np.int64)
        or not np.all(indices[1:] > indices[:-1])
        or original.shape != (timestamps.size, 478, 3)
        or original.dtype != np.dtype(np.float64)
        or mirrored.shape != original.shape
        or mirrored.dtype != np.dtype(np.float64)
        or valid.shape != timestamps.shape
        or valid.dtype != np.dtype(bool)
    ):
        raise ValueError("decoded paired mesh stream differs from the frozen contract")
    if not np.isfinite(original[valid]).all() or not np.isfinite(mirrored[valid]).all():
        raise ValueError("valid paired meshes must be finite")
    return timestamps.astype(np.float64, copy=False), indices, original, mirrored, valid


_ACTION_EVIDENCE_METRICS = {
    "eyebrow_raise": (
        "brow_height_asymmetry_iod",
        "brow_height_change_from_rest_iod",
    ),
    "gentle_eye_closure": (
        "eye_aperture_asymmetry_iod",
        "residual_eye_aperture_iod",
        "eye_closure_change_from_rest_iod",
    ),
    "tight_eye_squeeze": (
        "eye_aperture_asymmetry_iod",
        "residual_eye_aperture_iod",
        "eye_closure_change_from_rest_iod",
    ),
    "relaxed_smile": (
        "mouth_corner_vertical_asymmetry_iod",
        "mouth_corner_vertical_change_from_rest_iod",
    ),
    "lip_pucker": (
        "mouth_corner_horizontal_asymmetry_iod",
        "mouth_width_change_from_rest_iod",
    ),
    "lower_teeth_show": (
        "mouth_corner_vertical_asymmetry_iod",
        "lower_lip_change_from_rest_iod",
        "mouth_open_change_from_rest_iod",
    ),
    "reanimated_smile": (
        "mouth_corner_vertical_asymmetry_iod",
        "mouth_corner_vertical_change_from_rest_iod",
    ),
}


def _clinical_rows(meshes: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = meshes[mask]
    rows = np.stack([
        clinical_landmark_features(mesh, 1.0, 1.0) for mesh in selected
    ]).astype(np.float64, copy=False)
    if rows.ndim != 2 or rows.shape[1] != 23 or not np.isfinite(rows).all():
        raise ValueError("descriptive clinical geometry could not be measured")
    return rows


def _descriptive_action_evidence(
    *,
    action_id: str,
    timing,
    action_meshes: np.ndarray,
    action_valid: np.ndarray,
    baseline_meshes: np.ndarray,
    baseline_valid: np.ndarray,
) -> DescriptiveActionEvidence:
    action_rows = _clinical_rows(action_meshes, action_valid)
    rest_rows = _clinical_rows(baseline_meshes, baseline_valid)
    action_median = np.median(action_rows, axis=0)
    rest_median = np.median(rest_rows, axis=0)
    action_brow = float(np.mean(action_median[[10, 11]]))
    rest_brow = float(np.mean(rest_median[[10, 11]]))
    action_eye = float(np.mean(action_median[[0, 1]]))
    rest_eye = float(np.mean(rest_median[[0, 1]]))
    values = {
        "brow_height_asymmetry_iod": float(action_median[12]),
        "brow_height_change_from_rest_iod": abs(action_brow - rest_brow),
        "eye_aperture_asymmetry_iod": float(action_median[2]),
        "residual_eye_aperture_iod": action_eye,
        "eye_closure_change_from_rest_iod": abs(action_eye - rest_eye),
        "mouth_corner_vertical_asymmetry_iod": float(action_median[16]),
        "mouth_corner_vertical_change_from_rest_iod": float(np.mean(np.abs(
            action_median[[14, 15]] - rest_median[[14, 15]]
        ))),
        "mouth_corner_horizontal_asymmetry_iod": float(action_median[20]),
        "mouth_width_change_from_rest_iod": abs(
            float(action_median[21] - rest_median[21])
        ),
        "lower_lip_change_from_rest_iod": abs(float(
            np.median(action_meshes[action_valid, 14, 1])
            - np.median(baseline_meshes[baseline_valid, 14, 1])
        )),
        "mouth_open_change_from_rest_iod": abs(
            float(action_median[22] - rest_median[22])
        ),
    }
    metrics = _ACTION_EVIDENCE_METRICS.get(action_id)
    if metrics is None:
        raise ValueError("action has no descriptive evidence contract")
    observations = tuple(
        DescriptiveObservation(metric=metric, value=values[metric])
        for metric in metrics
    )
    if any(not math.isfinite(item.value) or item.value < 0.0 for item in observations):
        raise ValueError("descriptive action evidence must be finite and nonnegative")
    midpoint = (timing.hold_start_ms + timing.hold_end_ms) // 2
    return DescriptiveActionEvidence(
        action_id=action_id,
        context_frame_ms=midpoint,
        observations=observations,
    )


def build_v9_action_arrays(
    evidence: CaptureEvidence,
    *,
    frame_timestamps_ms: np.ndarray,
    source_frame_indices: np.ndarray,
    original_meshes: np.ndarray,
    mirrored_meshes: np.ndarray,
    pair_valid_mask: np.ndarray,
    source_fps: float,
) -> PreparedV9Request:
    """Project a paired decoded FACES stream into the exact V9 NPZ tensors."""
    if not isinstance(evidence, CaptureEvidence):
        raise ValueError("validated capture evidence is required")
    if (
        isinstance(source_fps, bool)
        or not isinstance(source_fps, (int, float))
        or not math.isfinite(float(source_fps))
        or float(source_fps) <= 0.0
    ):
        raise ValueError("source_fps must be finite and positive")
    timestamps, indices, original, mirrored, valid = _mesh_stream(
        frame_timestamps_ms,
        source_frame_indices,
        original_meshes,
        mirrored_meshes,
        pair_valid_mask,
    )
    combined = np.concatenate(
        (original.reshape(original.shape[0], -1), mirrored.reshape(mirrored.shape[0], -1)),
        axis=1,
    )
    combined = combined.copy()
    combined[~valid] = np.nan
    reference = np.nanmedian(combined[valid], axis=0)
    motion = np.zeros(timestamps.size, dtype=np.float64)
    motion[valid] = np.nanmean(np.abs(combined[valid] - reference), axis=1)
    motion[~valid] = np.nan

    segments = {}
    for action in tuple(item.action for item in evidence.timeline.actions):
        segment = segment_faces_action(
            evidence.timeline,
            action,
            timestamps,
            indices,
            valid,
            motion,
            decoded_recording_sha256=evidence.video_sha256,
            decoded_duration_ms=evidence.timeline.recording_duration_ms,
            landmark_features=combined,
        )
        if not segment.eligible:
            raise CaptureTrackingError(
                action,
                valid_samples=int(segment.valid_mask.sum()),
            )
        segments[action] = segment

    baseline = segments["neutral_repose"]
    present_actions = _present_v9_actions(evidence)
    by_action = {item.action: item for item in evidence.timeline.actions}
    active = [segments[action] for action, _v9 in present_actions]
    original_actions = np.stack([original[item.frame_positions] for item in active])
    mirrored_actions = np.stack([mirrored[item.frame_positions] for item in active])
    action_valid = np.stack([item.valid_mask for item in active])
    action_indices = np.stack([item.source_frame_indices for item in active])
    original_baseline = original[baseline.frame_positions]
    mirrored_baseline = mirrored[baseline.frame_positions]
    baseline_valid = baseline.valid_mask
    original_baselines = np.repeat(
        original_baseline[None, ...], len(active), axis=0
    )
    mirrored_baselines = np.repeat(
        mirrored_baseline[None, ...], len(active), axis=0
    )
    baseline_masks = np.repeat(baseline_valid[None, ...], len(active), axis=0)
    action_names = tuple(v9 for _faces, v9 in present_actions)
    bag = dense_action_token_bag(
        original_actions,
        mirrored_actions,
        action_valid,
        action_indices,
        original_baselines,
        mirrored_baselines,
        baseline_masks,
        fps=float(source_fps),
        action_names=action_names,
    )
    codes = np.asarray([ACTION_VOCAB.index(name) for name in action_names], dtype=np.int64)
    arrays = {
        "clinical_original": bag.clinical_original.astype(np.float32),
        "clinical_mirrored": bag.clinical_mirrored.astype(np.float32),
        "dense_original": bag.dense_original.astype(np.float32),
        "dense_mirrored": bag.dense_mirrored.astype(np.float32),
        "dense_valid_mask": bag.dense_valid_mask.astype(bool),
        "dense_available": bag.dense_available.astype(bool),
        "dense_timestamps": bag.dense_timestamps.astype(np.float32),
        "action_mask": np.ones(len(active), dtype=bool),
        "action_codes": codes,
    }
    frozen = {name: _immutable(value) for name, value in arrays.items()}
    counts = tuple(int(item.valid_mask.sum()) for item in active)
    descriptive = tuple(
        _descriptive_action_evidence(
            action_id=action_id,
            timing=by_action[action_id],
            action_meshes=original[segment.frame_positions],
            action_valid=segment.valid_mask,
            baseline_meshes=original_baseline,
            baseline_valid=baseline_valid,
        )
        for (action_id, _v9), segment in zip(present_actions, active)
    )
    return PreparedV9Request(
        arrays=frozen,
        valid_samples_per_action=counts,
        descriptive_evidence_per_action=descriptive,
    )


def encode_v9_request_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Serialize the exact tensor mapping without Python object serialization."""
    expected = {
        "clinical_original", "clinical_mirrored", "dense_original",
        "dense_mirrored", "dense_valid_mask", "dense_available",
        "dense_timestamps", "action_mask", "action_codes",
    }
    if type(arrays) is not dict or set(arrays) != expected:
        raise ValueError("V9 request fields differ from the closed schema")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite probability")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite probability")
    return result


def build_gateway_response(
    prediction: Mapping[str, object],
    *,
    evidence: CaptureEvidence,
    valid_samples_per_action: tuple[int, ...],
    descriptive_evidence_per_action: tuple[DescriptiveActionEvidence, ...],
    preprocessing_version: str,
    face_landmarker_sha256: str,
) -> dict[str, object]:
    """Validate one low-level V9 result and expose the browser-safe response."""
    if not isinstance(evidence, CaptureEvidence):
        raise ValueError("validated capture evidence is required")
    if type(prediction) is not dict or set(prediction) != {
        "model_id", "protocol", "probability", "member_probabilities",
        "predicted_class", "threshold",
    }:
        raise ValueError("V9 prediction fields differ from the closed schema")
    if (
        prediction["model_id"] != SHARED_V9_MODEL_ID
        or prediction["protocol"] != "cue_aligned_action"
        or prediction["threshold"] != 0.5
        or prediction["predicted_class"] not in {0, 1}
        or type(prediction["predicted_class"]) is not int
        or preprocessing_version != PREPROCESSING_VERSION
        or type(face_landmarker_sha256) is not str
        or _SHA256.fullmatch(face_landmarker_sha256) is None
    ):
        raise ValueError("V9 prediction or preprocessing provenance drifted")
    probability = _probability(prediction["probability"], "probability")
    raw_members = prediction["member_probabilities"]
    if type(raw_members) not in {list, tuple} or len(raw_members) != 3:
        raise ValueError("V9 prediction requires exactly three ensemble members")
    members = tuple(
        _probability(value, f"member_probabilities[{index}]")
        for index, value in enumerate(raw_members)
    )
    if (
        abs(probability - sum(members) / len(members)) > 1e-7
        or int(probability >= 0.5) != prediction["predicted_class"]
    ):
        raise ValueError("V9 aggregate probability is inconsistent")
    present_actions = _present_v9_actions(evidence)
    if (
        type(valid_samples_per_action) is not tuple
        or len(valid_samples_per_action) != len(present_actions)
        or any(type(value) is not int or not 26 <= value <= 32
               for value in valid_samples_per_action)
    ):
        raise ValueError("every active FACES action requires 26 of 32 valid samples")
    by_action = {item.action: item for item in evidence.timeline.actions}
    if (
        type(descriptive_evidence_per_action) is not tuple
        or len(descriptive_evidence_per_action) != len(present_actions)
    ):
        raise ValueError("descriptive evidence must align with every active action")
    evidence_rows = []
    for (action, _v9_action), item in zip(
        present_actions, descriptive_evidence_per_action
    ):
        timing = by_action[action]
        expected_midpoint = (timing.hold_start_ms + timing.hold_end_ms) // 2
        expected_metrics = _ACTION_EVIDENCE_METRICS[action]
        if (
            not isinstance(item, DescriptiveActionEvidence)
            or item.action_id != action
            or item.context_frame_ms != expected_midpoint
            or type(item.observations) is not tuple
            or tuple(observation.metric for observation in item.observations)
            != expected_metrics
            or any(
                not isinstance(observation, DescriptiveObservation)
                or not math.isfinite(observation.value)
                or observation.value < 0.0
                for observation in item.observations
            )
        ):
            raise ValueError("descriptive evidence differs from the closed action contract")
        evidence_rows.append({
            "id": action,
            "context_frame_ms": item.context_frame_ms,
            "observations": [
                {
                    "metric": observation.metric,
                    "value": round(observation.value, 6),
                    "unit": "interocular_distance",
                }
                for observation in item.observations
            ],
        })
    action_rows = [
        {
            "id": action,
            "v9_action": v9_action,
            "hold_start_ms": by_action[action].hold_start_ms,
            "hold_end_ms": by_action[action].hold_end_ms,
            "valid_samples": valid,
        }
        for (action, v9_action), valid in zip(
            present_actions, valid_samples_per_action
        )
    ]
    return {
        "schema_version": GATEWAY_RESPONSE_SCHEMA,
        "model": {
            "model_id": SHARED_V9_MODEL_ID,
            "candidate_id": SHARED_V9_CANDIDATE_ID,
            "release_manifest_sha256": SHARED_V9_MANIFEST_SHA256,
            "ensemble_members": 3,
        },
        "preprocessing": {
            "version": PREPROCESSING_VERSION,
            "face_landmarker_sha256": face_landmarker_sha256,
            "mirror_method": "horizontal_flip_and_redetect",
            "protocol": "cue_aligned_action",
            "timing_source": evidence.timeline.timing_source.value,
        },
        "quality": {
            "eligible": True,
            "actions_used": len(action_rows),
            "optional_actions_unavailable": (
                [] if evidence.manifest.reanimated_smile_applicable
                else ["reanimated_smile"]
            ),
            "actions": action_rows,
        },
        "prediction": {
            "probability": probability,
            "member_probabilities": list(members),
            "predicted_class": int(prediction["predicted_class"]),
            "threshold": 0.5,
            "interpretation": "class_1_research_score_only",
            "endpoint_semantics": (
                "meei_facial_palsy_vs_healthy_control_development_head"
            ),
            "class_0_label": "meei_healthy_control",
            "class_1_label": "meei_facial_palsy",
        },
        "report_evidence": {
            "normalization": (
                "original_view_centered_eye_axis_aligned_interocular_scaled"
            ),
            "interpretation": (
                "measured_movement_observation_not_causal_or_severity"
            ),
            "context_frame_method": (
                "registered_hold_midpoint_not_model_selected"
            ),
            "actions": evidence_rows,
        },
        "clinical_use_eligible": False,
    }


__all__ = [
    "CAPTURE_MANIFEST_SCHEMA",
    "CaptureTimingError",
    "CaptureTrackingError",
    "CaptureEvidence",
    "CaptureManifest",
    "DescriptiveActionEvidence",
    "DescriptiveObservation",
    "DecodedCaptureSamples",
    "FACES_TO_V9_ACTIONS",
    "GATEWAY_RESPONSE_SCHEMA",
    "PREPROCESSING_VERSION",
    "PairedMeshStream",
    "PreparedV9Request",
    "SHARED_V9_MANIFEST_SHA256",
    "build_gateway_response",
    "build_v9_action_arrays",
    "decode_capture_samples",
    "encode_v9_request_npz",
    "extract_paired_meshes",
    "parse_capture_evidence",
]
