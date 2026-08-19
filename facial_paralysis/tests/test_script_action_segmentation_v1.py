"""Contracts for exogenously anchored FACES and NeuroFace action segments."""
from __future__ import annotations

import hashlib
import json
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
import src.preprocessing.script_action_segmentation_v1 as segmentation  # noqa: E402
from src.preprocessing.script_action_segmentation_v1 import (  # noqa: E402
    FACES_ACTION_ORDER,
    FACES_SCRIPT_VERSION,
    FACES_TIMELINE_SCHEMA,
    NEUROFACE_PRIMARY_TASKS,
    PINNED_NEUROFACE_PRIVATE_MANIFEST_SHA256,
    TimingSource,
    segment_faces_action,
    segment_neuroface_recording_task,
    validate_faces_sidecar,
    validate_neuroface_task_binding,
)


def _sidecar(*, include_optional: bool = False) -> dict[str, object]:
    actions = []
    for index, action in enumerate(FACES_ACTION_ORDER[:7]):
        prompt_start = index * 4000
        actions.append({
            "action": action,
            "status": "completed",
            "prompt_start_ms": prompt_start,
            "hold_start_ms": prompt_start + 500,
            "hold_end_ms": prompt_start + 3500,
            "completion_ms": prompt_start + 3750,
        })
    if include_optional:
        actions.append({
            "action": FACES_ACTION_ORDER[7],
            "status": "prompted",
            "prompt_start_ms": 28000,
            "hold_start_ms": 28500,
            "hold_end_ms": 31500,
        })
    return {
        "schema_version": FACES_TIMELINE_SCHEMA,
        "script_version": FACES_SCRIPT_VERSION,
        "recording_sha256": "a" * 64,
        "timing_source": "capture_event_log",
        "recording_duration_ms": 32000,
        "actions": actions,
    }


def _frames(duration_ms: int = 32000):
    timestamps = np.arange(0.0, float(duration_ms), 1000.0 / 30.0)
    indices = np.arange(timestamps.size, dtype=np.int64)
    valid = np.ones(timestamps.size, dtype=bool)
    motion = np.linspace(0.0, 1.0, timestamps.size, dtype=np.float64)
    return timestamps, indices, valid, motion


def _landmarks(n_frames: int, n_channels: int = 3) -> np.ndarray:
    return np.linspace(
        0.0, 1.0, n_frames * n_channels, dtype=np.float64
    ).reshape(n_frames, n_channels)


def _manifest_fixture(task_label: str = "NSM_KISS") -> tuple[bytes, str, str]:
    cohort_definitions = (
        ("als", 11, 76, 920, "affected"),
        ("healthy_control", 11, 80, 1015, "unaffected"),
        ("post_stroke", 14, 105, 1371, "affected"),
    )
    participants = []
    participant_ids: dict[str, list[str]] = {}
    participant_index = 0
    for cohort, count, _videos, _frames, label in cohort_definitions:
        participant_ids[cohort] = []
        for _ in range(count):
            participant_id = "grp_" + hashlib.sha256(
                f"participant-{participant_index}".encode("ascii")
            ).hexdigest()
            participant_index += 1
            participant_ids[cohort].append(participant_id)
            participants.append({
                "participant_id": participant_id,
                "cohort": cohort,
                "binary_label": label,
            })
    task_cycle = (
        "NSM_KISS", "NSM_OPEN", "NSM_SPREAD", "BBP_NORMAL", "DDK_PA",
        "DDK_PATAKA", "NSM_BIGSMILE", "NSM_BLOW", "NSM_BROW",
    )
    records = []
    record_index = 0
    for cohort, _count, videos, annotated_total, label in cohort_definitions:
        for cohort_index in range(videos):
            recording_id = "rec_" + hashlib.sha256(
                f"recording-{record_index}".encode("ascii")
            ).hexdigest()
            video_sha = hashlib.sha256(
                f"video-{record_index}".encode("ascii")
            ).hexdigest()
            task = task_label if record_index == 0 else task_cycle[record_index % 9]
            annotated = annotated_total - (videos - 1) if cohort_index == 0 else 1
            records.append({
                "recording_id": recording_id,
                "participant_id": participant_ids[cohort][
                    cohort_index % len(participant_ids[cohort])
                ],
                "cohort": cohort,
                "binary_label": label,
                "session": "02",
                "task": task,
                "video_archive_id": f"{cohort}_videos",
                "video_sha256": video_sha,
                "video_size_bytes": 1000 + record_index,
                "landmark_archive_id": f"{cohort}_landmarks",
                "landmark_sha256": hashlib.sha256(
                    f"landmark-{record_index}".encode("ascii")
                ).hexdigest(),
                "annotated_frames": annotated,
                "slp_scores": {
                    "symmetry": 1.0, "rom": 1.0, "speed": 1.0,
                    "variability": 1.0, "fatigue": 1.0, "total": 5.0,
                },
            })
            record_index += 1
    manifest = {
        "schema_version": "neuroface_external_private_manifest_v1",
        "dataset": "Toronto_NeuroFace_v1",
        "claim_unit": "participant",
        "target": "neurological_orofacial_impairment_vs_healthy_control",
        "primary_tasks": list(NEUROFACE_PRIMARY_TASKS),
        "counts": {
            "participants": 36,
            "videos": 261,
            "annotated_frames": 3306,
            "affected_participants": 25,
            "unaffected_participants": 11,
            "primary_complete_participants": 36,
            "by_cohort": {
                cohort: {
                    "participants": count,
                    "videos": videos,
                    "annotated_frames": annotated,
                }
                for cohort, count, videos, annotated, _label in cohort_definitions
            },
        },
        "archives": {},
        "slp_workbook_sha256": {},
        "participants": participants,
        "records": records,
    }
    payload = (
        json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    return payload, records[0]["recording_id"], records[0]["video_sha256"]


@contextmanager
def _binding_context(task_label: str = "NSM_KISS"):
    payload, recording_id, video_sha = _manifest_fixture(task_label)
    fixture_digest = hashlib.sha256(payload).hexdigest()
    original_pin = segmentation.PINNED_NEUROFACE_PRIVATE_MANIFEST_SHA256
    segmentation.PINNED_NEUROFACE_PRIVATE_MANIFEST_SHA256 = fixture_digest
    try:
        yield validate_neuroface_task_binding(
            payload,
            recording_id=recording_id,
            decoded_recording_sha256=video_sha,
        )
    finally:
        segmentation.PINNED_NEUROFACE_PRIVATE_MANIFEST_SHA256 = original_pin


def _segment(
    timeline,
    action: str,
    timestamps: np.ndarray,
    indices: np.ndarray,
    valid: np.ndarray,
    motion: np.ndarray,
    *,
    landmarks: np.ndarray | None = None,
):
    return segment_faces_action(
        timeline,
        action,
        timestamps,
        indices,
        valid,
        motion,
        decoded_recording_sha256=timeline.recording_sha256,
        decoded_duration_ms=timeline.recording_duration_ms,
        landmark_features=(
            _landmarks(timestamps.size) if landmarks is None else landmarks
        ),
    )


def test_registry_and_timing_sources_are_exact(c: Check):
    c.eq(FACES_ACTION_ORDER, (
        "neutral_repose", "eyebrow_raise", "gentle_eye_closure",
        "tight_eye_squeeze", "relaxed_smile", "lip_pucker",
        "lower_teeth_show", "reanimated_smile",
    ))
    c.eq(NEUROFACE_PRIMARY_TASKS, ("NSM_KISS", "NSM_OPEN", "NSM_SPREAD"))
    c.eq(tuple(source.value for source in TimingSource), (
        "capture_event_log", "audio_forced_alignment", "blinded_manual",
        "recording_task_label",
    ))


def test_faces_sidecar_requires_exact_order_and_optional_eighth(c: Check):
    required = validate_faces_sidecar(_sidecar())
    c.eq(tuple(item.action for item in required.actions), FACES_ACTION_ORDER[:7])
    complete = validate_faces_sidecar(_sidecar(include_optional=True))
    c.eq(tuple(item.action for item in complete.actions), FACES_ACTION_ORDER)
    c.eq(complete.timing_source, TimingSource.CAPTURE_EVENT_LOG)

    duplicate = _sidecar()
    duplicate["actions"][2]["action"] = duplicate["actions"][1]["action"]
    c.raises(lambda: validate_faces_sidecar(duplicate), ValueError,
             "duplicate action cannot masquerade as a complete script")
    reordered = _sidecar()
    reordered["actions"][1], reordered["actions"][2] = (
        reordered["actions"][2], reordered["actions"][1]
    )
    c.raises(lambda: validate_faces_sidecar(reordered), ValueError,
             "script order is authoritative")


def test_faces_sidecar_rejects_malformed_identity_time_and_source(c: Check):
    malformed = _sidecar()
    malformed["recording_sha256"] = "A" * 64
    c.raises(lambda: validate_faces_sidecar(malformed), ValueError,
             "recording digest is canonical lowercase hex")
    wrong_source = _sidecar()
    wrong_source["timing_source"] = "recording_task_label"
    c.raises(lambda: validate_faces_sidecar(wrong_source), ValueError,
             "recording task labels are NeuroFace-only")
    noninteger = _sidecar()
    noninteger["actions"][0]["hold_start_ms"] = 500.0
    c.raises(lambda: validate_faces_sidecar(noninteger), ValueError,
             "millisecond bounds are integers, not rounded implicitly")
    wrong_hold = _sidecar()
    wrong_hold["actions"][0]["hold_end_ms"] = 3499
    c.raises(lambda: validate_faces_sidecar(wrong_hold), ValueError,
             "the FACES hold is exactly three seconds")
    overlap = _sidecar()
    overlap["actions"][1]["prompt_start_ms"] = 3700
    c.raises(lambda: validate_faces_sidecar(overlap), ValueError,
             "action intervals do not overlap")
    outside = _sidecar()
    outside["recording_duration_ms"] = 27000
    c.raises(lambda: validate_faces_sidecar(outside), ValueError,
             "every bound lies inside the recording")


def test_anchored_hold_samples_32_positions_without_visual_retiming(c: Check):
    timeline = validate_faces_sidecar(_sidecar())
    timestamps, indices, valid, rising = _frames()
    first = _segment(
        timeline, "gentle_eye_closure", timestamps, indices, valid, rising
    )
    moved_peak = np.zeros_like(rising)
    moved_peak[-1] = 1000.0
    second = _segment(
        timeline, "gentle_eye_closure", timestamps, indices, valid, moved_peak
    )
    c.eq(first.frame_positions.shape, (32,))
    c.true(np.array_equal(first.frame_positions, second.frame_positions),
           "motion cannot move or delete the external anchor")
    c.true(bool(np.all(first.frame_timestamps_ms >= 8500.0)))
    c.true(bool(np.all(first.frame_timestamps_ms <= 11500.0)))
    c.true(first.prompted and first.tracking_adequate and first.eligible)
    for value in (
        first.frame_positions, first.source_frame_indices,
        first.frame_timestamps_ms, first.valid_mask, first.motion_curve,
        first.landmark_features, first.channel_finite_support,
    ):
        c.true(not value.flags.writeable, "returned arrays are immutable evidence")
        c.raises(lambda value=value: value.setflags(write=True), ValueError,
                 "immutable byte backing prevents re-enabling writes")


def test_flat_prompted_action_remains_eligible_but_not_observed(c: Check):
    timeline = validate_faces_sidecar(_sidecar())
    timestamps, indices, valid, _motion = _frames()
    flat = np.full(timestamps.shape, 0.25, dtype=np.float64)
    result = _segment(
        timeline, "eyebrow_raise", timestamps, indices, valid, flat
    )
    c.true(result.prompted, "external timeline proves the attempt")
    c.true(result.tracking_adequate and result.eligible,
           "bilateral low capacity is retained as evidence")
    c.true(not result.observed_motion,
           "flat motion is represented explicitly, not relabeled missing")


def test_invalid_positions_stay_masked_and_control_eligibility(c: Check):
    timeline = validate_faces_sidecar(_sidecar())
    timestamps, indices, valid, motion = _frames()
    landmarks = _landmarks(timestamps.size)
    baseline = _segment(
        timeline, "relaxed_smile", timestamps, indices, valid, motion,
        landmarks=landmarks,
    )
    damaged = valid.copy()
    damaged[baseline.frame_positions[:7]] = False
    motion[baseline.frame_positions[:7]] = np.nan
    landmarks[baseline.frame_positions[:7]] = np.nan
    result = _segment(
        timeline, "relaxed_smile", timestamps, indices, damaged, motion,
        landmarks=landmarks,
    )
    c.eq(int(result.valid_mask.sum()), 25)
    c.true(bool(np.isnan(result.motion_curve[~result.valid_mask]).all()),
           "invalid values remain masked and are not converted to zero")
    c.true(bool(np.isnan(result.landmark_features[~result.valid_mask]).all()),
           "invalid channel values remain missing rather than zero-imputed")
    c.true(not result.tracking_adequate and not result.eligible)


def test_every_landmark_channel_needs_26_finite_samples(c: Check):
    timeline = validate_faces_sidecar(_sidecar())
    timestamps, indices, valid, motion = _frames()
    landmarks = _landmarks(timestamps.size)
    baseline = _segment(
        timeline, "tight_eye_squeeze", timestamps, indices, valid, motion,
        landmarks=landmarks,
    )
    landmarks[baseline.frame_positions[:7], 1] = np.nan
    result = _segment(
        timeline, "tight_eye_squeeze", timestamps, indices, valid, motion,
        landmarks=landmarks,
    )
    c.eq(tuple(result.channel_finite_support), (32, 25, 32))
    c.true(not result.tracking_adequate and not result.eligible,
           "one under-supported channel makes the segment ineligible")
    c.raises(lambda: segment_faces_action(
        timeline, "tight_eye_squeeze", timestamps, indices, valid, motion,
        decoded_recording_sha256=timeline.recording_sha256,
        decoded_duration_ms=timeline.recording_duration_ms,
        landmark_features=None,
    ), ValueError, "missing channel data fails closed")


def test_decoded_recording_identity_and_duration_are_bound(c: Check):
    timeline = validate_faces_sidecar(_sidecar())
    timestamps, indices, valid, motion = _frames()
    landmarks = _landmarks(timestamps.size)
    c.raises(lambda: segment_faces_action(
        timeline, "lip_pucker", timestamps, indices, valid, motion,
        decoded_recording_sha256="b" * 64,
        decoded_duration_ms=timeline.recording_duration_ms,
        landmark_features=landmarks,
    ), ValueError, "a timeline cannot be applied to different recording bytes")
    c.raises(lambda: segment_faces_action(
        timeline, "lip_pucker", timestamps, indices, valid, motion,
        decoded_recording_sha256=timeline.recording_sha256,
        decoded_duration_ms=timeline.recording_duration_ms - 1,
        landmark_features=landmarks,
    ), ValueError, "decoded duration must match the bound timeline")


def test_public_faces_timeline_construction_cannot_bypass_validation(c: Check):
    timeline = validate_faces_sidecar(_sidecar())
    forged = replace(timeline, actions=tuple(reversed(timeline.actions)))
    timestamps, indices, valid, motion = _frames()
    c.raises(lambda: segment_faces_action(
        forged, "eyebrow_raise", timestamps, indices, valid, motion,
        decoded_recording_sha256=forged.recording_sha256,
        decoded_duration_ms=forged.recording_duration_ms,
        landmark_features=_landmarks(timestamps.size),
    ), ValueError, "segment entrypoint revalidates publicly constructible timelines")

    generator = (item for item in timeline.actions)
    c.raises(lambda: segment_faces_action(
        replace(timeline, actions=generator),
        "eyebrow_raise", timestamps, indices, valid, motion,
        decoded_recording_sha256=timeline.recording_sha256,
        decoded_duration_ms=timeline.recording_duration_ms,
        landmark_features=_landmarks(timestamps.size),
    ), ValueError, "actions must be an exact bounded tuple before traversal")

    class TraversalBomb:
        @property
        def action(self):
            raise RuntimeError("oversized actions were traversed")

    oversized = timeline.actions + (TraversalBomb(), TraversalBomb())
    c.raises(lambda: segment_faces_action(
        replace(timeline, actions=oversized),
        "eyebrow_raise", timestamps, indices, valid, motion,
        decoded_recording_sha256=timeline.recording_sha256,
        decoded_duration_ms=timeline.recording_duration_ms,
        landmark_features=_landmarks(timestamps.size),
    ), ValueError, "oversized actions reject before element traversal")


def test_unanchored_and_malformed_frame_streams_fail_closed(c: Check):
    timestamps, indices, valid, motion = _frames()
    c.raises(
        lambda: segment_faces_action(
            None, "lip_pucker", timestamps, indices, valid, motion,
            decoded_recording_sha256="a" * 64,
            decoded_duration_ms=32000,
            landmark_features=_landmarks(timestamps.size),
        ),
        ValueError,
        "visual motion alone cannot establish an attempted action",
    )
    timeline = validate_faces_sidecar(_sidecar())
    duplicate_time = timestamps.copy()
    duplicate_time[10] = duplicate_time[9]
    c.raises(lambda: _segment(
        timeline, "lip_pucker", duplicate_time, indices, valid, motion
    ), ValueError, "decoded timestamps must increase")
    c.raises(lambda: _segment(
        timeline, "unknown", timestamps, indices, valid, motion
    ), ValueError, "unknown or unprompted action is rejected")
    negative_time = timestamps.copy()
    negative_time[0] = -1.0
    c.raises(lambda: _segment(
        timeline, "lip_pucker", negative_time, indices, valid, motion
    ), ValueError, "recording-relative timestamps are nonnegative")
    negative_index = indices.copy()
    negative_index[0] = -1
    c.raises(lambda: _segment(
        timeline, "lip_pucker", timestamps, negative_index, valid, motion
    ), ValueError, "source frame indices are nonnegative")
    overflowing_index = indices.astype(np.uint64)
    overflowing_index[-1] = np.uint64(2**63)
    c.raises(lambda: _segment(
        timeline, "lip_pucker", timestamps, overflowing_index, valid, motion
    ), ValueError, "uint64 frame indices cannot wrap during int64 conversion")


def test_neuroface_task_label_uses_frozen_four_whole_recording_windows(c: Check):
    timestamps = np.arange(160, dtype=np.float64) * 20.0
    indices = np.arange(160, dtype=np.int64)
    valid = np.ones(160, dtype=bool)
    with _binding_context() as binding:
        first = segment_neuroface_recording_task(
            binding, timestamps, indices, valid,
            decoded_recording_sha256=binding.recording_sha256,
            decoded_source_frame_count=160,
        )
        second = segment_neuroface_recording_task(
            binding, timestamps, indices, valid,
            decoded_recording_sha256=binding.recording_sha256,
            decoded_source_frame_count=160,
        )
    c.eq(first.timing_source, TimingSource.RECORDING_TASK_LABEL)
    c.eq(first.scope, "whole_recording")
    c.eq(first.source_frame_indices.shape, (4, 32))
    c.true(np.array_equal(first.source_frame_indices, second.source_frame_indices),
           "task-label segmentation is deterministic")
    c.eq(tuple(first.source_frame_indices[:, 0]), (0, 42, 85, 128))
    c.eq(tuple(first.source_frame_indices[:, -1]), (31, 73, 116, 159))
    for value in (
        first.frame_positions, first.source_frame_indices,
        first.frame_timestamps_ms, first.valid_mask,
    ):
        c.true(not value.flags.writeable, "NeuroFace evidence arrays are immutable")
        c.raises(lambda value=value: value.setflags(write=True), ValueError,
                 "NeuroFace arrays have immutable byte backing")


def test_neuroface_binding_authenticates_task_and_exact_recording(c: Check):
    timestamps = np.arange(160, dtype=np.float64) * 20.0
    indices = np.arange(160, dtype=np.int64)
    valid = np.ones(160, dtype=bool)
    with _binding_context("NSM_OPEN") as binding:
        c.eq(binding.task_label, "NSM_OPEN")
        c.raises(lambda: segment_neuroface_recording_task(
            binding, timestamps, indices, valid,
            decoded_recording_sha256="d" * 64,
            decoded_source_frame_count=160,
        ), ValueError, "binding cannot be applied to different decoded bytes")
        forged = replace(binding, task_label="NSM_SPREAD")
        c.raises(lambda: segment_neuroface_recording_task(
            forged, timestamps, indices, valid,
            decoded_recording_sha256=forged.recording_sha256,
            decoded_source_frame_count=160,
        ), ValueError, "caller cannot replace the manifest-derived task")


def test_neuroface_rejects_arbitrary_manifest_claim_and_duplicate_json(c: Check):
    c.eq(
        PINNED_NEUROFACE_PRIVATE_MANIFEST_SHA256,
        "235d2af2f3f4507b4ec858ff8dd9ff949d7f19e0d3656cbf5dcc0218648da07b",
        "production binding is pinned to the frozen private manifest bytes",
    )
    payload, recording_id, video_sha = _manifest_fixture("NSM_KISS")
    c.raises(lambda: validate_neuroface_task_binding(
        payload,
        recording_id=recording_id,
        decoded_recording_sha256=video_sha,
    ), ValueError, "arbitrary canonical bytes cannot self-declare an e*64 manifest hash")
    legacy_self_declared = {
        "task_label": "NSM_SPREAD",
        "authenticated_manifest_sha256": "e" * 64,
        "recording_sha256": video_sha,
    }
    c.raises(lambda: validate_neuroface_task_binding(
        legacy_self_declared,
        recording_id=recording_id,
        decoded_recording_sha256=video_sha,
    ), ValueError, "legacy self-declared task/digest objects are rejected")
    duplicate = (
        b'{"schema_version":"neuroface_external_private_manifest_v1",'
        b'"schema_version":"neuroface_external_private_manifest_v1"}'
    )
    original_pin = segmentation.PINNED_NEUROFACE_PRIVATE_MANIFEST_SHA256
    segmentation.PINNED_NEUROFACE_PRIVATE_MANIFEST_SHA256 = hashlib.sha256(
        duplicate
    ).hexdigest()
    try:
        c.raises(lambda: validate_neuroface_task_binding(
            duplicate,
            recording_id=recording_id,
            decoded_recording_sha256=video_sha,
        ), ValueError, "duplicate JSON keys fail before last-value-wins parsing")
    finally:
        segmentation.PINNED_NEUROFACE_PRIVATE_MANIFEST_SHA256 = original_pin


def test_neuroface_rejects_unknown_task_and_noncanonical_stream(c: Check):
    timestamps = np.arange(160, dtype=np.float64) * 20.0
    indices = np.arange(160, dtype=np.int64)
    valid = np.ones(160, dtype=bool)
    payload, recording_id, video_sha = _manifest_fixture("DDK_PA")
    original_pin = segmentation.PINNED_NEUROFACE_PRIVATE_MANIFEST_SHA256
    segmentation.PINNED_NEUROFACE_PRIVATE_MANIFEST_SHA256 = hashlib.sha256(
        payload
    ).hexdigest()
    try:
        c.raises(lambda: validate_neuroface_task_binding(
            payload,
            recording_id=recording_id,
            decoded_recording_sha256=video_sha,
        ), ValueError, "only manifest-derived locked primary tasks are eligible")
    finally:
        segmentation.PINNED_NEUROFACE_PRIVATE_MANIFEST_SHA256 = original_pin
    with _binding_context("NSM_OPEN") as binding:
        c.raises(lambda: segment_neuroface_recording_task(
            binding, timestamps[:127], indices[:127], valid[:127],
            decoded_recording_sha256=binding.recording_sha256,
            decoded_source_frame_count=127,
        ), ValueError, "frozen four-window protocol needs at least 128 frames")
        skipped = indices.copy()
        skipped[80:] += 1
        c.raises(lambda: segment_neuroface_recording_task(
            binding, timestamps, skipped, valid,
            decoded_recording_sha256=binding.recording_sha256,
            decoded_source_frame_count=160,
        ), ValueError, "whole-recording source indices must be contiguous")
        cropped_indices = np.arange(100, 260, dtype=np.int64)
        c.raises(lambda: segment_neuroface_recording_task(
            binding, timestamps, cropped_indices, valid,
            decoded_recording_sha256=binding.recording_sha256,
            decoded_source_frame_count=260,
        ), ValueError, "a contiguous 100..259 crop is not a whole recording")
        c.raises(lambda: segment_neuroface_recording_task(
            binding, timestamps, indices, valid,
            decoded_recording_sha256=binding.recording_sha256,
            decoded_source_frame_count=True,
        ), ValueError, "decoded source frame count is a strict positive integer")


if __name__ == "__main__":
    run_all("test_script_action_segmentation_v1", dict(globals()))
