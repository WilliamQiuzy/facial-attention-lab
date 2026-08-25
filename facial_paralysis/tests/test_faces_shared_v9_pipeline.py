"""Contracts for the FACES raw-video to Shared V9 gateway boundary."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
from src.preprocessing.faces_shared_v9_pipeline import (  # noqa: E402
    CAPTURE_MANIFEST_SCHEMA,
    FACES_TO_V9_ACTIONS,
    GATEWAY_RESPONSE_SCHEMA,
    SHARED_V9_MANIFEST_SHA256,
    build_v9_action_arrays,
    build_gateway_response,
    encode_v9_request_npz,
    decode_capture_samples,
    extract_paired_meshes,
    parse_capture_evidence,
)
from src.preprocessing import faces_shared_v9_pipeline as pipeline_module  # noqa: E402
from src.deployment.shared_v8_release import validate_request_arrays  # noqa: E402
from src.models.dense_clinical_shared_encoder_v1 import ACTION_VOCAB  # noqa: E402
from src.preprocessing.script_action_segmentation_v1 import (  # noqa: E402
    FACES_ACTION_ORDER,
    FACES_SCRIPT_VERSION,
    FACES_TIMELINE_SCHEMA,
    segment_faces_action,
)


def _payloads(
    *,
    include_optional: bool = True,
    video: bytes | None = None,
    timing_source: str = "capture_event_log",
):
    video = video or b"small identifier-free video fixture"
    digest = hashlib.sha256(video).hexdigest()
    action_count = 8 if include_optional else 7
    actions = []
    for index, action in enumerate(FACES_ACTION_ORDER[:action_count]):
        prompt = index * 4_000
        actions.append({
            "action": action,
            "status": "completed",
            "prompt_start_ms": prompt,
            "hold_start_ms": prompt + 500,
            "hold_end_ms": prompt + 3_500,
            "completion_ms": prompt + 3_750,
        })
    timeline = {
        "schema_version": FACES_TIMELINE_SCHEMA,
        "script_version": FACES_SCRIPT_VERSION,
        "recording_sha256": digest,
        "timing_source": timing_source,
        "recording_duration_ms": action_count * 4_000,
        "actions": actions,
    }
    manifest = {
        "schema_version": CAPTURE_MANIFEST_SCHEMA,
        "protocol_version": "FACES-v0.01",
        "recording_source": "browser-camera",
        "video_sha256": digest,
        "reanimated_smile_applicable": include_optional,
    }
    return (
        video,
        json.dumps(manifest, separators=(",", ":")).encode("utf-8"),
        json.dumps(timeline, separators=(",", ":")).encode("utf-8"),
    )


def test_capture_evidence_binds_video_timeline_and_medical_action_map(c: Check):
    video, manifest, timeline = _payloads()
    evidence = parse_capture_evidence(video, manifest, timeline)
    c.eq(evidence.video_sha256, hashlib.sha256(video).hexdigest())
    c.eq(evidence.manifest.recording_source, "browser-camera")
    c.eq(tuple(item.action for item in evidence.timeline.actions), FACES_ACTION_ORDER)
    c.eq(FACES_TO_V9_ACTIONS, (
        ("eyebrow_raise", "BROW_RAISE"),
        ("gentle_eye_closure", "EYE_GENTLE"),
        ("tight_eye_squeeze", "EYE_FORCEFUL"),
        ("relaxed_smile", "SMILE_GENTLE"),
        ("lip_pucker", "LIP_PUCKER"),
        ("lower_teeth_show", "SHOW_BOTTOM_TEETH"),
        ("reanimated_smile", "SMILE_FULL"),
    ))


def test_capture_evidence_accepts_both_medically_valid_script_variants(c: Check):
    video, manifest, timeline = _payloads()
    changed = json.loads(manifest)
    changed["video_sha256"] = "0" * 64
    c.raises(
        lambda: parse_capture_evidence(
            video, json.dumps(changed).encode("utf-8"), timeline
        ),
        ValueError,
        "manifest digest cannot differ from exact video bytes",
    )

    video, manifest, timeline = _payloads(include_optional=False)
    evidence = parse_capture_evidence(video, manifest, timeline)
    c.eq(evidence.manifest.reanimated_smile_applicable, False)
    c.eq(
        tuple(item.action for item in evidence.timeline.actions),
        FACES_ACTION_ORDER[:-1],
    )

    video, manifest, timeline = _payloads(
        include_optional=False,
        timing_source="audio_forced_alignment",
    )
    evidence = parse_capture_evidence(video, manifest, timeline)
    c.eq(evidence.timeline.timing_source.value, "audio_forced_alignment")

    duplicate = manifest.replace(
        b'"schema_version":', b'"schema_version":"forged","schema_version":', 1
    )
    c.raises(
        lambda: parse_capture_evidence(video, duplicate, timeline),
        ValueError,
        "duplicate JSON keys are rejected",
    )


def test_gateway_response_exposes_only_v9_binary_research_output(c: Check):
    video, manifest, timeline = _payloads()
    evidence = parse_capture_evidence(video, manifest, timeline)
    stream = _mesh_stream(evidence)
    prepared = build_v9_action_arrays(
        evidence,
        frame_timestamps_ms=stream[0],
        source_frame_indices=stream[1],
        original_meshes=stream[2],
        mirrored_meshes=stream[3],
        pair_valid_mask=stream[4],
        source_fps=stream[5],
    )
    prediction = {
        "model_id": "broad_literature_shared_v9_blv9_009_ensemble",
        "protocol": "cue_aligned_action",
        "probability": 0.73,
        "member_probabilities": [0.71, 0.74, 0.74],
        "predicted_class": 1,
        "threshold": 0.5,
    }
    response = build_gateway_response(
        prediction,
        evidence=evidence,
        valid_samples_per_action=prepared.valid_samples_per_action,
        descriptive_evidence_per_action=prepared.descriptive_evidence_per_action,
        preprocessing_version="faces-to-shared-v9/v1",
        face_landmarker_sha256="6" * 64,
    )
    c.eq(response["schema_version"], GATEWAY_RESPONSE_SCHEMA)
    c.eq(response["model"]["candidate_id"], "BLV9-009")
    c.eq(response["model"]["release_manifest_sha256"], SHARED_V9_MANIFEST_SHA256)
    c.eq(response["quality"]["actions_used"], 7)
    c.eq(response["prediction"]["probability"], 0.73)
    c.eq(response["prediction"]["interpretation"], "class_1_research_score_only")
    c.eq(
        response["report_evidence"]["interpretation"],
        "measured_movement_observation_not_causal_or_severity",
    )
    c.eq(
        response["report_evidence"]["normalization"],
        "original_view_centered_eye_axis_aligned_interocular_scaled",
    )
    c.eq(len(response["report_evidence"]["actions"]), 7)
    c.eq(
        response["report_evidence"]["actions"][0]["context_frame_ms"],
        6_000,
    )
    c.eq(response["clinical_use_eligible"], False)
    serialized = json.dumps(response, sort_keys=True).lower()
    c.true("house-brackmann" not in serialized)
    for forbidden in ("affected_side", "abnormal", "regional_severity", "caused", "contributed"):
        c.true(forbidden not in serialized)


def _mesh_stream(evidence):
    fps = 30.0
    timestamps = np.arange(
        0.0, float(evidence.timeline.recording_duration_ms), 1000.0 / fps,
        dtype=np.float64,
    )
    source_indices = np.arange(timestamps.size, dtype=np.int64)
    base = np.full((478, 3), (0.5, 0.5, 0.0), dtype=np.float64)
    right_eye = {
        33: (0.30, 0.40), 133: (0.40, 0.40),
        159: (0.35, 0.38), 158: (0.37, 0.38), 160: (0.33, 0.38),
        145: (0.35, 0.42), 144: (0.33, 0.42), 153: (0.37, 0.42),
    }
    left_eye = {
        263: (0.70, 0.40), 362: (0.60, 0.40),
        386: (0.65, 0.38), 385: (0.63, 0.38), 387: (0.67, 0.38),
        374: (0.65, 0.42), 380: (0.67, 0.42), 373: (0.63, 0.42),
    }
    for index, xy in {**right_eye, **left_eye}.items():
        base[index, :2] = xy
    for index, x in zip((70, 63, 105, 66, 107), np.linspace(0.30, 0.40, 5)):
        base[index, :2] = (x, 0.30)
    for index, x in zip((300, 293, 334, 296, 336), np.linspace(0.70, 0.60, 5)):
        base[index, :2] = (x, 0.30)
    midline = (168, 6, 197, 195, 5, 4, 1, 19, 2, 164, 0, 17, 152, 10)
    for offset, index in enumerate(midline):
        base[index, :2] = (0.5, 0.25 + 0.04 * offset)
    base[61, :2] = (0.40, 0.70)
    base[291, :2] = (0.60, 0.70)
    base[13, :2] = (0.50, 0.68)
    base[14, :2] = (0.50, 0.72)
    original = np.repeat(base[None, :, :], timestamps.size, axis=0)
    mirrored = original.copy()
    mirrored[:, :, 0] *= -1.0
    valid = np.ones(timestamps.size, dtype=bool)
    return timestamps, source_indices, original, mirrored, valid, fps


def test_action_pipeline_builds_exact_v9_tensor_contract_and_keeps_flat_actions(c: Check):
    video, manifest, timeline = _payloads()
    evidence = parse_capture_evidence(video, manifest, timeline)
    stream = _mesh_stream(evidence)
    prepared = build_v9_action_arrays(
        evidence,
        frame_timestamps_ms=stream[0],
        source_frame_indices=stream[1],
        original_meshes=stream[2],
        mirrored_meshes=stream[3],
        pair_valid_mask=stream[4],
        source_fps=stream[5],
    )
    c.eq(prepared.valid_samples_per_action, (32, 32, 32, 32, 32, 32, 32))
    c.eq(
        tuple(item.action_id for item in prepared.descriptive_evidence_per_action),
        tuple(action for action, _v9 in FACES_TO_V9_ACTIONS),
    )
    expected_metrics = (
        ("brow_height_asymmetry_iod", "brow_height_change_from_rest_iod"),
        ("eye_aperture_asymmetry_iod", "residual_eye_aperture_iod", "eye_closure_change_from_rest_iod"),
        ("eye_aperture_asymmetry_iod", "residual_eye_aperture_iod", "eye_closure_change_from_rest_iod"),
        ("mouth_corner_vertical_asymmetry_iod", "mouth_corner_vertical_change_from_rest_iod"),
        ("mouth_corner_horizontal_asymmetry_iod", "mouth_width_change_from_rest_iod"),
        ("mouth_corner_vertical_asymmetry_iod", "lower_lip_change_from_rest_iod", "mouth_open_change_from_rest_iod"),
        ("mouth_corner_vertical_asymmetry_iod", "mouth_corner_vertical_change_from_rest_iod"),
    )
    for row, metrics in zip(prepared.descriptive_evidence_per_action, expected_metrics):
        c.eq(tuple(item.metric for item in row.observations), metrics)
        c.true(all(np.isfinite(item.value) and item.value >= 0.0 for item in row.observations))
        c.eq(row.context_frame_ms, 6_000 + 4_000 * len([
            prior for prior in prepared.descriptive_evidence_per_action
            if prior.context_frame_ms < row.context_frame_ms
        ]))
    expected_codes = np.asarray(
        [ACTION_VOCAB.index(name) for _faces, name in FACES_TO_V9_ACTIONS],
        dtype=np.int64,
    )
    c.true(np.array_equal(prepared.arrays["action_codes"], expected_codes))
    validated = validate_request_arrays("cue_aligned_action", prepared.arrays)
    c.eq(validated["clinical_original"].shape, (1, 7, 110))
    c.eq(validated["dense_original"].shape, (1, 7, 32, 478, 3))
    payload = encode_v9_request_npz(prepared.arrays)
    with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
        c.eq(set(saved.files), set(prepared.arrays))
        c.eq(saved["clinical_original"].dtype, np.dtype(np.float32))


def test_action_pipeline_omits_optional_action_without_zero_imputation(c: Check):
    video, manifest, timeline = _payloads(include_optional=False)
    evidence = parse_capture_evidence(video, manifest, timeline)
    stream = _mesh_stream(evidence)
    prepared = build_v9_action_arrays(
        evidence,
        frame_timestamps_ms=stream[0],
        source_frame_indices=stream[1],
        original_meshes=stream[2],
        mirrored_meshes=stream[3],
        pair_valid_mask=stream[4],
        source_fps=stream[5],
    )
    c.eq(prepared.valid_samples_per_action, (32, 32, 32, 32, 32, 32))
    expected_names = tuple(name for _faces, name in FACES_TO_V9_ACTIONS[:-1])
    expected_codes = np.asarray(
        [ACTION_VOCAB.index(name) for name in expected_names], dtype=np.int64
    )
    c.true(np.array_equal(prepared.arrays["action_codes"], expected_codes))
    c.true(bool(prepared.arrays["action_mask"].all()))
    c.eq(prepared.arrays["clinical_original"].shape, (6, 110))
    validated = validate_request_arrays("cue_aligned_action", prepared.arrays)
    c.eq(validated["clinical_original"].shape, (1, 6, 110))
    c.true(ACTION_VOCAB.index("SMILE_FULL") not in set(expected_codes.tolist()))

    prediction = {
        "model_id": "broad_literature_shared_v9_blv9_009_ensemble",
        "protocol": "cue_aligned_action",
        "probability": 0.73,
        "member_probabilities": [0.71, 0.74, 0.74],
        "predicted_class": 1,
        "threshold": 0.5,
    }
    response = build_gateway_response(
        prediction,
        evidence=evidence,
        valid_samples_per_action=prepared.valid_samples_per_action,
        descriptive_evidence_per_action=prepared.descriptive_evidence_per_action,
        preprocessing_version="faces-to-shared-v9/v1",
        face_landmarker_sha256="6" * 64,
    )
    c.eq(response["quality"]["actions_used"], 6)
    c.eq(response["quality"]["optional_actions_unavailable"], ["reanimated_smile"])
    c.true(all(row["id"] != "reanimated_smile" for row in response["quality"]["actions"]))


def test_action_pipeline_rejects_under_supported_tracking_without_motion_bias(c: Check):
    video, manifest, timeline = _payloads()
    evidence = parse_capture_evidence(video, manifest, timeline)
    timestamps, indices, original, mirrored, valid, fps = _mesh_stream(evidence)
    target = evidence.timeline.actions[1]
    inside = np.flatnonzero(
        (timestamps >= target.hold_start_ms) & (timestamps <= target.hold_end_ms)
    )
    valid[inside[:40]] = False
    c.raises(
        lambda: build_v9_action_arrays(
            evidence,
            frame_timestamps_ms=timestamps,
            source_frame_indices=indices,
            original_meshes=original,
            mirrored_meshes=mirrored,
            pair_valid_mask=valid,
            source_fps=fps,
        ),
        ValueError,
        "tracking support, not visible movement magnitude, controls eligibility",
    )


def test_tracking_gate_has_typed_safe_action_context_and_exact_boundary(c: Check):
    video, manifest, timeline = _payloads(include_optional=False)
    evidence = parse_capture_evidence(video, manifest, timeline)
    timestamps, indices, original, mirrored, valid, fps = _mesh_stream(evidence)
    combined = np.concatenate(
        (original.reshape(original.shape[0], -1), mirrored.reshape(mirrored.shape[0], -1)),
        axis=1,
    )
    segment = segment_faces_action(
        evidence.timeline,
        "lower_teeth_show",
        timestamps,
        indices,
        valid,
        np.zeros(timestamps.size, dtype=np.float64),
        decoded_recording_sha256=evidence.video_sha256,
        decoded_duration_ms=evidence.timeline.recording_duration_ms,
        landmark_features=combined,
    )
    valid[segment.frame_positions[25:]] = False
    tracking_error_type = getattr(pipeline_module, "CaptureTrackingError")
    caught = None
    try:
        build_v9_action_arrays(
            evidence,
            frame_timestamps_ms=timestamps,
            source_frame_indices=indices,
            original_meshes=original,
            mirrored_meshes=mirrored,
            pair_valid_mask=valid,
            source_fps=fps,
        )
    except tracking_error_type as error:
        caught = error
    c.true(caught is not None, "25 of 32 samples must produce typed tracking evidence")
    c.eq(caught.action, "lower_teeth_show")
    c.eq(caught.valid_samples, 25)
    c.eq(caught.required_samples, 26)
    c.true("/" not in str(caught), "typed tracking evidence must not contain a path")

    valid[segment.frame_positions[25]] = True
    accepted = build_v9_action_arrays(
        evidence,
        frame_timestamps_ms=timestamps,
        source_frame_indices=indices,
        original_meshes=original,
        mirrored_meshes=mirrored,
        pair_valid_mask=valid,
        source_fps=fps,
    )
    c.eq(accepted.valid_samples_per_action[-1], 26)


def _encoded_capture_video(
    *,
    include_optional: bool = True,
    fps: float = 12.0,
    duration_seconds: float | None = None,
) -> bytes:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "capture.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (16, 16)
        )
        if not writer.isOpened():
            raise RuntimeError("test OpenCV MP4 writer is unavailable")
        action_count = 8 if include_optional else 7
        duration = duration_seconds if duration_seconds is not None else action_count * 4
        for frame_index in range(int(round(duration * fps))):
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            frame[:, :8] = (0, 0, frame_index % 251 + 1)
            frame[:, 8:] = (frame_index % 251 + 1, 0, 0)
            writer.write(frame)
        writer.release()
        return path.read_bytes()


def test_video_decoder_rejects_low_fps_incomplete_holds_and_duration_drift(c: Check):
    low_fps = _encoded_capture_video(include_optional=False, fps=10.0)
    raw, manifest, timeline = _payloads(include_optional=False, video=low_fps)
    evidence = parse_capture_evidence(raw, manifest, timeline)
    c.raises(
        lambda: decode_capture_samples(raw, evidence.timeline, filename="capture.mp4"),
        ValueError,
        "a 10 fps capture cannot provide 32 unique samples in three seconds",
    )

    incomplete = _encoded_capture_video(
        include_optional=False,
        fps=12.0,
        duration_seconds=26.0,
    )
    raw, manifest, timeline = _payloads(include_optional=False, video=incomplete)
    evidence = parse_capture_evidence(raw, manifest, timeline)
    timing_error_type = getattr(pipeline_module, "CaptureTimingError")
    caught = None
    try:
        decode_capture_samples(raw, evidence.timeline, filename="capture.mp4")
    except timing_error_type as error:
        caught = error
    c.true(caught is not None, "incomplete hold sampling must expose numeric diagnostics")
    c.eq(caught.reason, "hold_sampling_incomplete")
    c.eq(caught.decoded_duration_ms, 26_000)
    c.eq(caught.timeline_duration_ms, 28_000)
    c.eq(caught.last_hold_ms, 27_500)

    normal = _encoded_capture_video(include_optional=False)
    raw, manifest, timeline = _payloads(include_optional=False, video=normal)
    changed = json.loads(timeline)
    changed["recording_duration_ms"] = 30_000
    evidence = parse_capture_evidence(
        raw,
        manifest,
        json.dumps(changed, separators=(",", ":")).encode("utf-8"),
    )
    timing_error_type = getattr(pipeline_module, "CaptureTimingError")
    caught = None
    try:
        decode_capture_samples(raw, evidence.timeline, filename="capture.mp4")
    except timing_error_type as error:
        caught = error
    c.true(caught is not None, "duration drift must provide typed numeric diagnostics")
    c.eq(caught.reason, "timeline_duration_drift")
    c.eq(caught.decoded_duration_ms, 28_000)
    c.eq(caught.timeline_duration_ms, 30_000)
    c.eq(caught.last_hold_ms, 27_500)
    c.eq(caught.decoded_frame_count, 336)
    c.true(abs(caught.source_fps - 12.0) < 0.01)


def test_video_decoder_uses_monotonic_container_timestamps_when_webm_fps_is_timebase(c: Check):
    class BrowserWebmCapture:
        def __init__(self, _path):
            self._next = 0
            self._last = -1

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 1000.0
            if prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT):
                return 2.0
            if prop == cv2.CAP_PROP_POS_MSEC:
                frame_index = max(self._last, 0)
                jitter_ms = 10.0 if frame_index % 2 == 0 else 0.0
                return frame_index * 1000.0 / 30.0 + jitter_ms
            return 0.0

        def read(self):
            if self._next >= 840:
                return False, None
            self._last = self._next
            self._next += 1
            return True, np.full((2, 2, 3), self._last % 251, dtype=np.uint8)

        def release(self):
            return None

    video = b"synthetic browser MediaRecorder WebM"
    raw, manifest, timeline = _payloads(include_optional=False, video=video)
    evidence = parse_capture_evidence(raw, manifest, timeline)
    original_capture = pipeline_module.cv2.VideoCapture
    pipeline_module.cv2.VideoCapture = BrowserWebmCapture
    try:
        decoded = decode_capture_samples(raw, evidence.timeline, filename="capture.webm")
    finally:
        pipeline_module.cv2.VideoCapture = original_capture
    c.eq(len(decoded.frames), 7 * 32)
    c.true(abs(decoded.source_fps - 30.0) < 0.02)
    c.true(abs(decoded.decoded_duration_ms - 28_000) <= 15)


def test_true_flip_pair_is_invalid_when_either_detection_fails(c: Check):
    video = _encoded_capture_video(include_optional=False)
    raw, manifest, timeline = _payloads(include_optional=False, video=video)
    evidence = parse_capture_evidence(raw, manifest, timeline)
    decoded = decode_capture_samples(raw, evidence.timeline, filename="capture.mp4")
    base = _mesh_stream(evidence)[2][0]
    calls = 0

    def one_sided_detector(_rgb):
        nonlocal calls
        calls += 1
        return np.array(base, copy=True) if calls % 2 else None

    paired = extract_paired_meshes(decoded, one_sided_detector)
    c.true(not bool(paired.pair_valid_mask.any()))
    c.true(bool(np.isnan(paired.original_meshes).all()))
    c.true(bool(np.isnan(paired.mirrored_meshes).all()))


def test_video_decoder_removes_request_scoped_files_on_success_and_failure(c: Check):
    video = _encoded_capture_video(include_optional=False)
    raw_video, manifest, timeline = _payloads(
        include_optional=False,
        video=video,
    )
    evidence = parse_capture_evidence(raw_video, manifest, timeline)
    with tempfile.TemporaryDirectory() as controlled_root:
        original_tempdir = pipeline_module.tempfile.tempdir
        pipeline_module.tempfile.tempdir = controlled_root
        try:
            decoded = decode_capture_samples(
                raw_video,
                evidence.timeline,
                filename="capture.mp4",
            )
            c.eq(len(decoded.frames), 7 * 32)
            c.eq(list(Path(controlled_root).iterdir()), [])
            c.raises(
                lambda: decode_capture_samples(
                    b"not a decodable mp4",
                    evidence.timeline,
                    filename="capture.mp4",
                ),
                ValueError,
                "decode failures also remove request-scoped video bytes",
            )
            c.eq(list(Path(controlled_root).iterdir()), [])
        finally:
            pipeline_module.tempfile.tempdir = original_tempdir


def test_video_decoder_samples_only_external_holds_and_true_flip_redetects(c: Check):
    video = _encoded_capture_video()
    raw_video, manifest, timeline = _payloads(video=video)
    evidence = parse_capture_evidence(raw_video, manifest, timeline)
    decoded = decode_capture_samples(
        raw_video, evidence.timeline, filename="capture.mp4"
    )
    c.eq(len(decoded.frames), 8 * 32)
    c.eq(decoded.frame_timestamps_ms.shape, (8 * 32,))
    c.true(bool(np.all(np.diff(decoded.frame_timestamps_ms) > 0.0)))
    c.true(bool(np.all(np.diff(decoded.source_frame_indices) > 0)))
    c.true(abs(decoded.source_fps - 12.0) < 1e-6)

    seen: list[np.ndarray] = []
    base = _mesh_stream(evidence)[2][0]

    def detector(rgb: np.ndarray) -> np.ndarray:
        seen.append(np.array(rgb, copy=True))
        return np.array(base, copy=True)

    paired = extract_paired_meshes(decoded, detector)
    c.eq(paired.original_meshes.shape, (8 * 32, 478, 3))
    c.true(bool(paired.pair_valid_mask.all()))
    c.eq(len(seen), 2 * 8 * 32)
    c.true(not np.array_equal(seen[0], seen[1]))
    c.true(np.array_equal(seen[0][:, ::-1], seen[1]))
    c.raises(
        lambda: decode_capture_samples(raw_video, evidence.timeline, filename="capture.txt"),
        ValueError,
        "unsupported container names are rejected before decode",
    )


if __name__ == "__main__":
    run_all("test_faces_shared_v9_pipeline", dict(globals()))
