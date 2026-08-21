"""Authority-bound 18D action-capacity feature contracts."""
from __future__ import annotations

import hashlib
import io
import json
import operator
import sys
import warnings
import zipfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
from scripts.run_mirror_invariant_110d import mirror_dynamic_features  # noqa: E402
from src.datasets.dynamic_landmark import (  # noqa: E402
    DYNAMIC_FEATURE_SCHEMA,
    DYNAMIC_FEATURE_NAMES,
    deterministic_window_starts,
    load_dynamic_landmark_recording_bytes,
)
from src.models.dynamic_landmark import horizontal_mirror_features  # noqa: E402
import src.datasets.dynamic_landmark as dynamic_cache  # noqa: E402
import src.preprocessing.action_capacity_features_v1 as capacity  # noqa: E402
from src.preprocessing.action_capacity_features_v1 import (  # noqa: E402
    ACTION_CAPACITY_DIM,
    action_capacity_feature_names,
    faces_action_capacity_feature_vector,
    mirror_action_capacity_features,
    neuroface_action_capacity_feature_vector,
)
from src.preprocessing.script_action_segmentation_v1 import (  # noqa: E402
    FACES_ACTION_ORDER,
    FACES_SCRIPT_VERSION,
    FACES_TIMELINE_SCHEMA,
    segment_faces_action,
    validate_faces_sidecar,
)
from test_script_action_segmentation_v1 import _binding_context  # noqa: E402


EXPECTED_NAMES = tuple(
    f"{channel}__{statistic}"
    for channel in (
        "corner_y_mesh61", "corner_y_mesh291",
        "corner_x_mesh61", "corner_x_mesh291",
        "mouth_width", "mouth_open",
    )
    for statistic in ("iqr", "range", "max_abs_velocity_per_second")
)


def _faces_timeline():
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
    return validate_faces_sidecar({
        "schema_version": FACES_TIMELINE_SCHEMA,
        "script_version": FACES_SCRIPT_VERSION,
        "recording_sha256": "a" * 64,
        "timing_source": "capture_event_log",
        "recording_duration_ms": 32000,
        "actions": actions,
    })


def _faces_stream(*, flat: bool = False, fps: float = 30.0):
    timestamps_ms = np.arange(0.0, 32000.0, 1000.0 / float(fps))
    indices = np.arange(timestamps_ms.size, dtype=np.int64)
    valid = np.ones(timestamps_ms.size, dtype=bool)
    motion = np.full(timestamps_ms.size, 0.25, dtype=np.float64)
    features = np.full((timestamps_ms.size, 95), 0.25, dtype=np.float32)
    if not flat:
        ramp = np.arange(timestamps_ms.size, dtype=np.float32)
        for scale, channel in enumerate((86, 87, 90, 91, 93, 94), start=1):
            features[:, channel] = float(scale) * ramp
        motion = ramp.astype(np.float64)
    return features, valid, timestamps_ms, indices, motion


def _faces_call(timeline, action, arrays, **overrides):
    features, valid, timestamps_ms, indices, motion = arrays
    kwargs = {
        "decoded_recording_sha256": timeline.recording_sha256,
        "decoded_duration_ms": timeline.recording_duration_ms,
        "feature_names": DYNAMIC_FEATURE_NAMES,
    }
    kwargs.update(overrides)
    return faces_action_capacity_feature_vector(
        timeline,
        action,
        features,
        valid,
        timestamps_ms,
        indices,
        motion,
        **kwargs,
    )


def _neuroface_arrays(source_frame_count: int = 200, *, flat: bool = False):
    starts = deterministic_window_starts(source_frame_count)
    indices = np.stack([
        np.arange(start, start + 32, dtype=np.int64) for start in starts
    ])
    timestamps = indices.astype(np.float64) / 30.0
    valid = np.ones((4, 32), dtype=bool)
    features = np.full((4, 32, 95), 0.25, dtype=np.float32)
    if not flat:
        ramp = np.arange(32, dtype=np.float32)
        for scale, channel in enumerate((86, 87, 90, 91, 93, 94), start=1):
            features[..., channel] = float(scale) * ramp
    return features, valid, timestamps, indices


def _neuroface_cache_payload(
    binding,
    arrays=None,
    *,
    source_frame_count: int = 200,
    timestamp_unit: object = "seconds",
    recording_id: str | None = None,
    source_sha256: str | None = None,
):
    if arrays is None:
        arrays = _neuroface_arrays(source_frame_count)
    features, valid, timestamps, indices = arrays
    stream = io.BytesIO()
    np.savez(
        stream,
        features=features,
        valid_mask=valid,
        timestamps=timestamps,
        timestamp_unit=np.asarray(timestamp_unit),
        source_frame_indices=indices,
        source_frame_count=np.asarray(source_frame_count, dtype=np.int64),
        feature_schema=np.asarray(DYNAMIC_FEATURE_SCHEMA),
        feature_names=np.asarray(DYNAMIC_FEATURE_NAMES),
        recording_id=np.asarray(
            binding.recording_id if recording_id is None else recording_id
        ),
        group_id=np.asarray("grp_" + "c" * 64),
        label=np.asarray(1, dtype=np.int64),
        source_sha256=np.asarray(
            binding.recording_sha256 if source_sha256 is None else source_sha256
        ),
    )
    return stream.getvalue()


def _collection_manifest_payload(binding, cache_payload: bytes) -> bytes:
    records = []
    used = {binding.recording_id}
    retained_needed = 230
    excluded_needed = 30
    records.append({
        "recording_id": binding.recording_id,
        "participant_id": "grp_" + "c" * 64,
        "video_sha256": binding.recording_sha256,
        "cache_sha256": hashlib.sha256(cache_payload).hexdigest(),
        "coverage": 1.0,
        "status": "retained",
    })
    index = 0
    while retained_needed or excluded_needed:
        recording_id = f"rec_{index:064x}"
        index += 1
        if recording_id in used:
            continue
        used.add(recording_id)
        base = {
            "recording_id": recording_id,
            "participant_id": f"grp_{index % 36:064x}",
            "video_sha256": f"{index + 1000:064x}",
        }
        if retained_needed:
            base.update({
                "cache_sha256": f"{index + 2000:064x}",
                "coverage": 1.0,
                "status": "retained",
            })
            retained_needed -= 1
        else:
            base.update({
                "status": "excluded",
                "exclusion_reason": "coverage_below_0_90",
            })
            excluded_needed -= 1
        records.append(base)
    records.sort(key=lambda row: row["recording_id"])
    manifest = {
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
        "provenance": {
            "private_manifest_sha256": "1" * 64,
            "mediapipe_model_sha256": "2" * 64,
            "implementation_sha256": "3" * 64,
        },
        "records": records,
    }
    return (
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


@contextmanager
def _collection_pin(payload: bytes):
    original = capacity.PINNED_NEUROFACE_COLLECTION_MANIFEST_SHA256
    capacity.PINNED_NEUROFACE_COLLECTION_MANIFEST_SHA256 = hashlib.sha256(
        payload
    ).hexdigest()
    try:
        yield
    finally:
        capacity.PINNED_NEUROFACE_COLLECTION_MANIFEST_SHA256 = original


def _neuroface_call(binding, cache_payload, collection_payload, **overrides):
    kwargs = {"decoded_recording_sha256": binding.recording_sha256}
    kwargs.update(overrides)
    return neuroface_action_capacity_feature_vector(
        binding, cache_payload, collection_payload, **kwargs
    )


def test_public_surface_is_authority_bound_and_names_are_exact(c: Check):
    c.eq(ACTION_CAPACITY_DIM, 18)
    c.eq(action_capacity_feature_names(), EXPECTED_NAMES)
    c.true("action_capacity_feature_vector" not in capacity.__all__)
    c.true("LANDMARK_MI_110D_SOURCE_NAMES" not in capacity.__all__)
    c.true(not hasattr(capacity, "action_capacity_feature_vector"))
    c.true(not hasattr(capacity, "_project_supported_raw"))
    for name in ("_SOURCE_INDICES", "_CAPACITY_RAW_INDICES", "_MIRROR_INDICES"):
        value = getattr(capacity, name)
        c.true(type(value) is tuple, f"{name} is immutable")
        c.raises(
            lambda value=value: operator.setitem(value, slice(0, 1), (999,)),
            TypeError,
        )
    for name in action_capacity_feature_names():
        c.true(not any(token in name for token in (
            "median", "absdiff", "minus", "correlation", "amplitude_ratio", "lag",
        )))


def test_faces_flat_prompted_action_is_zero_but_25_of_32_is_rejected(c: Check):
    timeline = _faces_timeline()
    arrays = _faces_stream(flat=True)
    flat = _faces_call(timeline, "eyebrow_raise", arrays)
    c.true(np.array_equal(flat, np.zeros(18, dtype=np.float64)))

    features, valid, timestamps, indices, motion = arrays
    segment = segment_faces_action(
        timeline, "eyebrow_raise", timestamps, indices, valid, motion,
        decoded_recording_sha256=timeline.recording_sha256,
        decoded_duration_ms=timeline.recording_duration_ms,
        landmark_features=features[:, 72:],
    )
    valid_26 = valid.copy()
    valid_26[segment.frame_positions[:6]] = False
    c.eq(_faces_call(
        timeline, "eyebrow_raise",
        (features, valid_26, timestamps, indices, motion),
    ).shape, (18,))
    valid_25 = valid.copy()
    valid_25[segment.frame_positions[:7]] = False
    c.raises(lambda: _faces_call(
        timeline, "eyebrow_raise",
        (features, valid_25, timestamps, indices, motion),
    ), ValueError)


def test_faces_schema_identity_and_complete_stream_are_mandatory(c: Check):
    timeline = _faces_timeline()
    arrays = _faces_stream()
    features, valid, timestamps, indices, motion = arrays
    c.raises(lambda: faces_action_capacity_feature_vector(
        timeline, "relaxed_smile", features, valid, timestamps, indices, motion,
        decoded_recording_sha256=timeline.recording_sha256,
        decoded_duration_ms=timeline.recording_duration_ms,
    ), TypeError, "dynamic schema cannot be omitted")
    swapped = list(DYNAMIC_FEATURE_NAMES)
    swapped[86], swapped[87] = swapped[87], swapped[86]
    c.raises(lambda: _faces_call(
        timeline, "relaxed_smile", arrays, feature_names=tuple(swapped)
    ), ValueError)
    c.raises(lambda: _faces_call(
        timeline, "relaxed_smile", arrays,
        decoded_recording_sha256="b" * 64,
    ), ValueError)

    anchored = segment_faces_action(
        timeline, "relaxed_smile", timestamps, indices, valid, motion,
        decoded_recording_sha256=timeline.recording_sha256,
        decoded_duration_ms=timeline.recording_duration_ms,
        landmark_features=features[:, 72:],
    )
    positions = anchored.frame_positions
    c.raises(lambda: _faces_call(
        timeline,
        "relaxed_smile",
        (
            features[positions], valid[positions], timestamps[positions],
            indices[positions], motion[positions],
        ),
    ), ValueError, "caller-selected single window is not a complete recording")


def test_faces_raw_mirror_matches_18d_mirror(c: Check):
    timeline = _faces_timeline()
    arrays = _faces_stream()
    original = _faces_call(timeline, "lip_pucker", arrays)
    features, valid, timestamps, indices, motion = arrays
    mirrored_raw = horizontal_mirror_features(torch.from_numpy(features)).numpy()
    mirrored = _faces_call(
        timeline, "lip_pucker",
        (mirrored_raw, valid, timestamps, indices, motion),
    )
    c.true(np.array_equal(mirrored, mirror_action_capacity_features(original)))
    c.true(np.array_equal(
        mirror_action_capacity_features(mirror_action_capacity_features(original)),
        original,
    ))


def test_faces_low_fps_reuse_uses_uniform_external_action_clock(c: Check):
    timeline = _faces_timeline()
    for fps in (10.0, 10.3):
        arrays = _faces_stream(fps=fps)
        features, valid, timestamps, indices, motion = arrays
        segment = segment_faces_action(
            timeline, "lip_pucker", timestamps, indices, valid, motion,
            decoded_recording_sha256=timeline.recording_sha256,
            decoded_duration_ms=timeline.recording_duration_ms,
            landmark_features=features[:, 72:],
        )
        c.true(np.unique(segment.frame_positions).size < 32)
        result = _faces_call(timeline, "lip_pucker", arrays)
        c.eq(result.shape, (18,))
        c.true(np.isfinite(result).all())

    arrays_12fps = _faces_stream(fps=12.0)
    segment_12fps = segment_faces_action(
        timeline, "lip_pucker",
        arrays_12fps[2], arrays_12fps[3], arrays_12fps[1], arrays_12fps[4],
        decoded_recording_sha256=timeline.recording_sha256,
        decoded_duration_ms=timeline.recording_duration_ms,
        landmark_features=arrays_12fps[0][:, 72:],
    )
    c.eq(np.unique(segment_12fps.frame_positions).size, 32)
    c.eq(_faces_call(timeline, "lip_pucker", arrays_12fps).shape, (18,))


def test_neuroface_qc_flat_and_raw_mirror_are_exact(c: Check):
    with _binding_context() as binding:
        flat_payload = _neuroface_cache_payload(
            binding, _neuroface_arrays(flat=True)
        )
        flat_collection = _collection_manifest_payload(binding, flat_payload)
        with _collection_pin(flat_collection):
            flat = _neuroface_call(binding, flat_payload, flat_collection)
            c.true(np.array_equal(flat, np.zeros(18, dtype=np.float64)))

        features, valid, timestamps, indices = _neuroface_arrays()
        valid_116 = valid.copy()
        valid_116.reshape(-1)[116:] = False
        features_116 = features.copy()
        features_116[~valid_116] = 0.0
        payload_116 = _neuroface_cache_payload(
            binding, (features_116, valid_116, timestamps, indices)
        )
        collection_116 = _collection_manifest_payload(binding, payload_116)
        with _collection_pin(collection_116):
            c.eq(_neuroface_call(
                binding, payload_116, collection_116
            ).shape, (18,))

        valid_115 = valid.copy()
        valid_115.reshape(-1)[115:] = False
        features_115 = features.copy()
        features_115[~valid_115] = 0.0
        payload_115 = _neuroface_cache_payload(
            binding, (features_115, valid_115, timestamps, indices)
        )
        collection_115 = _collection_manifest_payload(binding, payload_115)
        with _collection_pin(collection_115):
            c.raises(lambda: _neuroface_call(
                binding, payload_115, collection_115
            ), ValueError)

        original_payload = _neuroface_cache_payload(binding)
        original_collection = _collection_manifest_payload(
            binding, original_payload
        )
        mirrored_raw = mirror_dynamic_features(features)
        mirrored_payload = _neuroface_cache_payload(
            binding, (mirrored_raw, valid, timestamps, indices)
        )
        mirrored_collection = _collection_manifest_payload(
            binding, mirrored_payload
        )
        with _collection_pin(original_collection):
            original = _neuroface_call(
                binding, original_payload, original_collection
            )
        with _collection_pin(mirrored_collection):
            mirrored = _neuroface_call(
                binding, mirrored_payload, mirrored_collection
            )
        c.true(np.array_equal(
            mirrored, mirror_action_capacity_features(original)
        ))


def test_neuroface_revalidates_binding_source_windows_and_schema(c: Check):
    with _binding_context() as binding:
        payload = _neuroface_cache_payload(binding)
        collection = _collection_manifest_payload(binding, payload)
        with _collection_pin(collection):
            c.raises(lambda: _neuroface_call(
                replace(binding, task_label="NSM_OPEN"), payload, collection
            ), ValueError)
            c.raises(lambda: _neuroface_call(
                binding, payload, collection,
                decoded_recording_sha256="b" * 64,
            ), ValueError)
            c.raises(lambda: _neuroface_call(
                binding, np.asarray(bytearray(payload), dtype=np.uint8), collection
            ), ValueError, "ndarray subclasses and arrays are not cache authority")
            c.raises(lambda: _neuroface_call(
                binding, Path("forged-cache.npz"), collection
            ), ValueError, "paths are never reopened")

            changed = payload[:-1] + bytes((payload[-1] ^ 1,))
            c.raises(lambda: _neuroface_call(
                binding, changed, collection
            ), ValueError, "changed cache payload fails manifest hash binding")

        class ForgedBytes(bytes):
            pass

        class ForgedArray(np.ndarray):
            pass

        with _collection_pin(collection):
            c.raises(lambda: _neuroface_call(
                binding, ForgedBytes(payload), collection
            ), ValueError, "bytes subclasses cannot supply mutable authority")
            forged_array = np.frombuffer(payload, dtype=np.uint8).view(ForgedArray)
            c.raises(lambda: _neuroface_call(
                binding, forged_array, collection
            ), ValueError, "ndarray subclasses cannot replace exact cache bytes")


def test_neuroface_timestamp_unit_and_fixed_seconds_clock_are_bound(c: Check):
    with _binding_context() as binding:
        payload = _neuroface_cache_payload(binding)
        collection = _collection_manifest_payload(binding, payload)
        for unit in (None, "ms", "milliseconds", "unknown"):
            bad_unit_payload = _neuroface_cache_payload(
                binding, timestamp_unit=unit
            )
            bad_unit_collection = _collection_manifest_payload(
                binding, bad_unit_payload
            )
            with _collection_pin(bad_unit_collection):
                c.raises(lambda: _neuroface_call(
                    binding, bad_unit_payload, bad_unit_collection
                ), ValueError)
        features, valid, timestamps, indices = _neuroface_arrays()
        milliseconds_lie = _neuroface_cache_payload(
            binding,
            (features, valid, timestamps * 1000.0, indices),
            timestamp_unit="seconds",
        )
        with _collection_pin(collection):
            c.raises(lambda: _neuroface_call(
                binding, milliseconds_lie, collection
            ), ValueError, "timestamp changes fail the collection cache hash")


def test_neuroface_collection_pin_duplicates_and_inventory_are_closed(c: Check):
    with _binding_context() as binding:
        payload = _neuroface_cache_payload(binding)
        collection = _collection_manifest_payload(binding, payload)
        c.raises(lambda: _neuroface_call(
            binding, payload, collection
        ), ValueError, "the real out-of-band collection pin is mandatory")

        duplicate = b'{"schema_version":"x","schema_version":"y"}'
        with _collection_pin(duplicate):
            c.raises(lambda: _neuroface_call(
                binding, payload, duplicate
            ), ValueError, "duplicate JSON keys fail before last-value-wins")

        decoded = json.loads(collection)
        decoded["records"] = decoded["records"][:-1]
        incomplete = (
            json.dumps(decoded, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        with _collection_pin(incomplete):
            c.raises(lambda: _neuroface_call(
                binding, payload, incomplete
            ), ValueError, "all 261 collection rows are mandatory")


def test_neuroface_committed_bytes_cannot_be_replaced_before_loading(c: Check):
    with _binding_context() as binding:
        payload = _neuroface_cache_payload(binding)
        replacement = _neuroface_cache_payload(
            binding, _neuroface_arrays(flat=True)
        )
        collection = _collection_manifest_payload(binding, payload)
        with _collection_pin(collection):
            expected = _neuroface_call(binding, payload, collection)

        original_loader = dynamic_cache.load_dynamic_landmark_recording
        calls = {"path_loader": 0}

        def replacing_path_loader(path):
            calls["path_loader"] += 1
            Path(path).write_bytes(replacement)
            return original_loader(path)

        dynamic_cache.load_dynamic_landmark_recording = replacing_path_loader
        try:
            with _collection_pin(collection):
                observed = _neuroface_call(binding, payload, collection)
        finally:
            dynamic_cache.load_dynamic_landmark_recording = original_loader
        c.eq(calls["path_loader"], 0,
             "immutable cache bytes never pass through a reopenable path")
        c.true(np.array_equal(observed, expected),
               "the projected bytes are exactly those committed by the manifest")


def test_safe_bytes_loader_rejects_duplicate_and_broken_npz_containers(c: Check):
    with _binding_context() as binding:
        payload = _neuroface_cache_payload(binding)
    source = io.BytesIO(payload)
    duplicate = io.BytesIO()
    conflicting = io.BytesIO()
    np.save(
        conflicting,
        np.full((4, 32, 95), 99.0, dtype=np.float32),
        allow_pickle=False,
    )
    with zipfile.ZipFile(source, "r") as archive:
        with zipfile.ZipFile(duplicate, "w") as output:
            for info in archive.infolist():
                output.writestr(info.filename, archive.read(info))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                output.writestr("features.npy", conflicting.getvalue())
    c.raises(
        lambda: load_dynamic_landmark_recording_bytes(duplicate.getvalue()),
        ValueError,
        "duplicate conflicting features.npy members cannot be last-value-wins",
    )
    c.raises(
        lambda: load_dynamic_landmark_recording_bytes(b"PK\x03\x04"),
        ValueError,
        "a bare ZIP header fails with the public loader exception contract",
    )
    c.raises(
        lambda: load_dynamic_landmark_recording_bytes(payload[: len(payload) // 2]),
        ValueError,
        "a truncated valid NPZ fails with the public loader exception contract",
    )


if __name__ == "__main__":
    run_all("test_action_capacity_features_v1", dict(globals()))
