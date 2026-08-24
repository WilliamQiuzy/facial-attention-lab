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
GATEWAY_RESPONSE_SCHEMA = "facial-paralysis-shared-v9-inference/v1"
PREPROCESSING_VERSION = "faces-to-shared-v9/v1"
SHARED_V9_MODEL_ID = "broad_literature_shared_v9_blv9_009_ensemble"
SHARED_V9_CANDIDATE_ID = "BLV9-009"
SHARED_V9_MANIFEST_SHA256 = (
    "c4fdaf054f3076a2e31b0e1ae93d1e91a45212817eb39d1c4a53620a4007b18f"
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
    if not manifest.reanimated_smile_applicable:
        raise ValueError(
            "Shared V9 requires the validated seven active FACES movements; "
            "reanimated smile cannot be imputed"
        )
    if (
        timeline.schema_version != FACES_TIMELINE_SCHEMA
        or timeline.script_version != FACES_SCRIPT_VERSION
        or tuple(item.action for item in timeline.actions) != FACES_ACTION_ORDER
    ):
        raise ValueError("timeline identity differs from the V9 FACES contract")
    return CaptureEvidence(manifest=manifest, timeline=timeline, video_sha256=digest)


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
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            if (
                not math.isfinite(fps)
                or fps < 32.0 / 3.0
                or width < 2
                or height < 2
                or width * height > MAX_VIDEO_PIXELS
            ):
                raise ValueError("video clock or dimensions are outside the gateway bounds")
            selected_frames: list[np.ndarray] = []
            selected_times: list[float] = []
            selected_indices: list[int] = []
            target_index = 0
            previous: tuple[int, float, np.ndarray] | None = None
            decoded_count = 0
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
                timestamp = decoded_count * 1000.0 / fps
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
                decoded_count += 1
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
        finally:
            capture.release()
    if decoded_count < 1 or target_index != targets.size:
        raise ValueError("decoded recording does not cover every FACES hold")
    duration_ms = int(round(decoded_count * 1000.0 / fps))
    last_hold = timeline.actions[-1].hold_end_ms
    tolerance_ms = max(250, int(math.ceil(2000.0 / fps)))
    if (
        duration_ms > MAX_VIDEO_DURATION_MS
        or last_hold > duration_ms + tolerance_ms
        or abs(duration_ms - timeline.recording_duration_ms) > 1_500
    ):
        raise ValueError("decoded duration contradicts the FACES timeline")
    sampled_times = np.asarray(selected_times, dtype=np.float64)
    target_error = np.abs(sampled_times - targets)
    if np.any(target_error > 500.0 / fps + np.finfo(np.float64).eps * 32):
        raise ValueError("decoded hold sampling exceeds half a source frame")
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
    for action in FACES_ACTION_ORDER:
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
            raise ValueError(f"FACES action {action} failed the 26-of-32 tracking gate")
        segments[action] = segment

    baseline = segments["neutral_repose"]
    active = [segments[action] for action, _v9 in FACES_TO_V9_ACTIONS]
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
    action_names = tuple(v9 for _faces, v9 in FACES_TO_V9_ACTIONS)
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
    return PreparedV9Request(arrays=frozen, valid_samples_per_action=counts)


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
    if (
        type(valid_samples_per_action) is not tuple
        or len(valid_samples_per_action) != len(FACES_TO_V9_ACTIONS)
        or any(type(value) is not int or not 26 <= value <= 32
               for value in valid_samples_per_action)
    ):
        raise ValueError("every active FACES action requires 26 of 32 valid samples")
    by_action = {item.action: item for item in evidence.timeline.actions}
    action_rows = [
        {
            "id": action,
            "v9_action": v9_action,
            "hold_start_ms": by_action[action].hold_start_ms,
            "hold_end_ms": by_action[action].hold_end_ms,
            "valid_samples": valid,
        }
        for (action, v9_action), valid in zip(
            FACES_TO_V9_ACTIONS, valid_samples_per_action
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
            "actions": action_rows,
        },
        "prediction": {
            "probability": probability,
            "member_probabilities": list(members),
            "predicted_class": int(prediction["predicted_class"]),
            "threshold": 0.5,
            "interpretation": "research_score_only",
        },
        "clinical_use_eligible": False,
    }


__all__ = [
    "CAPTURE_MANIFEST_SCHEMA",
    "CaptureEvidence",
    "CaptureManifest",
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
