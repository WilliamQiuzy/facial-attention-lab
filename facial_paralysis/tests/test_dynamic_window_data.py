"""Fail-closed contracts for dynamic landmark recording caches."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.dynamic_landmark import (  # noqa: E402
    DYNAMIC_FEATURE_SCHEMA,
    deterministic_window_starts,
    load_dynamic_landmark_recording,
    load_dynamic_landmark_recordings,
    per_second_first_differences,
)
from src.datasets.patient_multistream import MP_FEATURE_NAMES_BY_SCHEMA  # noqa: E402
from _testlib import Check, run_all  # noqa: E402


SCHEMA = "mediapipe_bs_lr_v1+clinical23_v2"
SHAPE = (4, 32, 95)
RECORDING_ALPHA = f"rec_{1:064x}"
RECORDING_BETA = f"rec_{2:064x}"
GROUP_ALPHA = f"grp_{1:064x}"


def _fields(
    *,
    recording_id: str = RECORDING_ALPHA,
    group_id: str = GROUP_ALPHA,
    label: int = 1,
) -> dict[str, np.ndarray]:
    source_indices = np.stack(
        [np.arange(start, start + 32, dtype=np.int64)
         for start in (0, 40, 80, 120)]
    )
    timestamps = source_indices.astype(np.float64) / 25.0
    return {
        "features": np.zeros(SHAPE, dtype=np.float32),
        "valid_mask": np.ones(SHAPE[:2], dtype=bool),
        "timestamps": timestamps,
        "timestamp_unit": np.asarray("seconds"),
        "source_frame_indices": source_indices,
        "source_frame_count": np.asarray(152, dtype=np.int64),
        "feature_schema": np.asarray(SCHEMA),
        "feature_names": np.asarray(MP_FEATURE_NAMES_BY_SCHEMA[SCHEMA]),
        "recording_id": np.asarray(recording_id),
        "group_id": np.asarray(group_id),
        "label": np.asarray(label, dtype=np.int64),
        "source_sha256": np.asarray("a" * 64),
    }


def _save(root: Path, name: str = "recording.npz", **overrides) -> Path:
    fields = _fields()
    fields.update(overrides)
    path = root / name
    np.savez(path, **fields)
    return path


def test_window_starts_are_deterministic_and_span_recording(c: Check):
    c.eq(deterministic_window_starts(128), (0, 32, 64, 96),
         "minimum-length recording is partitioned without overlap")
    starts = deterministic_window_starts(160)
    c.eq(starts, deterministic_window_starts(160), "sampling is frozen")
    c.eq(len(starts), 4, "protocol emits exactly four windows")
    c.eq(starts[0], 0, "first window anchors recording start")
    c.eq(starts[-1], 128, "last window anchors recording end")
    c.true(all(b - a >= 32 for a, b in zip(starts, starts[1:])),
           "windows do not overlap")


def test_window_starts_reject_invalid_or_short_inputs(c: Check):
    for args in ((127,), (128, 0, 4), (128, 32, 0), (128, 32, 1), (128.0,),
                 (128, 32.0, 4), (128, 32, 4.0), (128, 40, 4)):
        c.raises(lambda args=args: deterministic_window_starts(*args), ValueError,
                 f"invalid window parameters {args!r}")


def test_valid_recording_cache_loads_with_fixed_contract(c: Check):
    with tempfile.TemporaryDirectory() as td:
        path = _save(Path(td))
        record = load_dynamic_landmark_recording(path)
        c.eq(record.features.shape, SHAPE, "fixed feature tensor shape")
        c.eq(record.features.dtype, np.dtype(np.float32), "fixed feature dtype")
        c.eq(record.valid_mask.dtype, np.dtype(bool), "fixed mask dtype")
        c.eq(record.recording_id, RECORDING_ALPHA, "deidentified recording id")
        c.eq(record.group_id, GROUP_ALPHA, "deidentified group id retained")
        c.eq(record.label, 1, "binary label retained")
        c.eq(record.source_frame_count, 152, "source frame count retained")
        c.eq(record.timestamp_unit, "seconds", "timestamp unit retained")
        c.eq(record.feature_schema, DYNAMIC_FEATURE_SCHEMA, "schema retained")
        c.eq(record.feature_names, MP_FEATURE_NAMES_BY_SCHEMA[SCHEMA],
             "exact ordered feature registry retained")


def test_cache_requires_temporal_provenance_metadata(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for missing_field in ("source_frame_count", "timestamp_unit"):
            fields = _fields()
            del fields[missing_field]
            path = root / f"missing_{missing_field}.npz"
            np.savez(path, **fields)
            c.raises(lambda path=path: load_dynamic_landmark_recording(path),
                     ValueError, f"missing {missing_field} is rejected")


def test_cache_requires_complete_exact_metadata(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        base = _fields()
        for field in (
            "feature_schema", "feature_names", "recording_id", "group_id",
            "label", "source_sha256", "source_frame_count", "timestamp_unit",
        ):
            partial = {key: value for key, value in base.items() if key != field}
            path = root / f"missing_{field}.npz"
            np.savez(path, **partial)
            c.raises(lambda path=path: load_dynamic_landmark_recording(path),
                     ValueError, f"missing {field} is rejected")

        invalid = (
            {"feature_schema": np.asarray("mediapipe_bs_lr_v1")},
            {"feature_names": np.asarray(list(reversed(
                MP_FEATURE_NAMES_BY_SCHEMA[SCHEMA])))},
            {"recording_id": np.asarray("")},
            {"recording_id": np.asarray("not deidentified")},
            {"recording_id": np.asarray("12345678")},
            {"group_id": np.asarray("  ")},
            {"group_id": np.asarray("group_alpha")},
            {"label": np.asarray(2, dtype=np.int64)},
            {"source_sha256": np.asarray("not-a-digest")},
        )
        for i, overrides in enumerate(invalid):
            path = _save(root, f"invalid_metadata_{i}.npz", **overrides)
            c.raises(lambda path=path: load_dynamic_landmark_recording(path),
                     ValueError, f"wrong metadata {tuple(overrides)} is rejected")


def test_cache_rejects_noncanonical_or_unexpected_metadata(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        padded_values = (
            {"feature_schema": np.asarray(f" {SCHEMA}")},
            {"recording_id": np.asarray(f" {RECORDING_ALPHA}")},
            {"group_id": np.asarray(f"{GROUP_ALPHA} ")},
            {"source_sha256": np.asarray(f"{'a' * 64} ")},
        )
        for i, overrides in enumerate(padded_values):
            path = _save(root, f"padded_metadata_{i}.npz", **overrides)
            c.raises(lambda path=path: load_dynamic_landmark_recording(path),
                     ValueError, "metadata whitespace cannot be normalized silently")

        path = _save(
            root,
            "unexpected_field.npz",
            unexpected_metadata=np.asarray("value"),
        )
        c.raises(lambda: load_dynamic_landmark_recording(path), ValueError,
                 "unexpected metadata is rejected fail closed")


def test_cache_rejects_wrong_tensor_types_shapes_and_frame_adjacency(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        invalid = (
            {"features": np.zeros((4, 31, 95), dtype=np.float32)},
            {"features": np.zeros(SHAPE, dtype=np.float64)},
            {"valid_mask": np.ones(SHAPE[:2], dtype=np.uint8)},
            {"timestamps": np.zeros((4, 31), dtype=np.float64)},
            {"timestamps": np.full(SHAPE[:2], "0", dtype="U1")},
            {"source_frame_indices": np.zeros(SHAPE[:2], dtype=np.float64)},
        )
        for i, overrides in enumerate(invalid):
            path = _save(root, f"invalid_tensor_{i}.npz", **overrides)
            c.raises(lambda path=path: load_dynamic_landmark_recording(path),
                     ValueError, f"wrong tensor contract {i} is rejected")

        fields = _fields()
        timestamps = fields["timestamps"].copy()
        timestamps[0, 3] = np.nan
        c.raises(lambda: load_dynamic_landmark_recording(
            _save(root, "nonfinite_time.npz", timestamps=timestamps)), ValueError,
            "timestamps must be finite")

        frame_indices = fields["source_frame_indices"].copy()
        frame_indices[2, 10] += 1
        c.raises(lambda: load_dynamic_landmark_recording(
            _save(root, "frame_gap.npz", source_frame_indices=frame_indices)),
            ValueError, "source frames must be adjacent within each window")


def test_cache_freezes_window_starts_to_source_frame_count(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cases = (
            ((0, 39, 81, 120), 152, "irregular starts"),
            ((1, 41, 81, 121), 153, "shifted starts"),
            ((0, 40, 80, 120), 151, "wrong source frame count"),
        )
        for i, (starts, frame_count, message) in enumerate(cases):
            frame_indices = np.stack([
                np.arange(start, start + 32, dtype=np.int64)
                for start in starts
            ])
            timestamps = frame_indices.astype(np.float64) / 25.0
            path = _save(
                root,
                f"invalid_starts_{i}.npz",
                source_frame_indices=frame_indices,
                source_frame_count=np.asarray(frame_count, dtype=np.int64),
                timestamps=timestamps,
            )
            c.raises(lambda path=path: load_dynamic_landmark_recording(path),
                     ValueError, message)


def test_cache_rejects_invalid_source_frame_count_metadata(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        invalid_counts = (
            np.asarray(True),
            np.asarray(152.0, dtype=np.float64),
            np.asarray(127, dtype=np.int64),
            np.asarray([152], dtype=np.int64),
        )
        for i, frame_count in enumerate(invalid_counts):
            path = _save(
                root, f"invalid_frame_count_{i}.npz",
                source_frame_count=frame_count,
            )
            c.raises(lambda path=path: load_dynamic_landmark_recording(path),
                     ValueError, "source frame count must be a scalar integer >=128")


def test_cache_requires_seconds_and_strictly_increasing_timestamps(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for i, unit in enumerate(("milliseconds", "seconds ")):
            path = _save(
                root, f"invalid_timestamp_unit_{i}.npz",
                timestamp_unit=np.asarray(unit),
            )
            c.raises(lambda path=path: load_dynamic_landmark_recording(path),
                     ValueError, "timestamp unit must be exact seconds")

        timestamps = _fields()["timestamps"].copy()
        timestamps[0, 5] = timestamps[0, 4]
        c.raises(lambda: load_dynamic_landmark_recording(
            _save(root, "equal_timestamps.npz", timestamps=timestamps)),
            ValueError, "equal timestamps are rejected")

        timestamps = _fields()["timestamps"].copy()
        timestamps[1, 8] = timestamps[1, 7] - 0.01
        c.raises(lambda: load_dynamic_landmark_recording(
            _save(root, "decreasing_timestamps.npz", timestamps=timestamps)),
            ValueError, "decreasing timestamps are rejected")


def test_cache_rejects_unsigned_source_frame_wraparound(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        frame_indices = _fields()["source_frame_indices"].astype(np.uint64)
        frame_indices[0, 0] = np.iinfo(np.uint64).max
        frame_indices[0, 1:] = np.arange(31, dtype=np.uint64)
        c.raises(lambda: load_dynamic_landmark_recording(
            _save(root, "wrapped_indices.npz",
                  source_frame_indices=frame_indices)), ValueError,
            "unsigned wraparound is not forward adjacency")


def test_cache_rejects_nonfinite_valid_values_nonzero_padding_and_low_coverage(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        features = np.zeros(SHAPE, dtype=np.float32)
        features[0, 0, 0] = np.nan
        c.raises(lambda: load_dynamic_landmark_recording(
            _save(root, "nonfinite_features.npz", features=features)), ValueError,
            "valid feature values must be finite")

        mask = np.ones(SHAPE[:2], dtype=bool)
        mask[0, 0] = False
        features = np.zeros(SHAPE, dtype=np.float32)
        features[0, 0, 0] = 0.25
        c.raises(lambda: load_dynamic_landmark_recording(
            _save(root, "noncanonical_padding.npz", features=features,
                  valid_mask=mask)), ValueError,
            "invalid feature rows must be canonical zero")

        mask = np.ones(SHAPE[:2], dtype=bool)
        mask.reshape(-1)[:13] = False
        c.raises(lambda: load_dynamic_landmark_recording(
            _save(root, "low_coverage.npz", valid_mask=mask)), ValueError,
            "coverage below 90 percent is rejected")


def test_cache_coverage_boundary_is_exactly_ninety_percent(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fail_mask = np.zeros(SHAPE[:2], dtype=bool)
        fail_mask.reshape(-1)[:115] = True
        pass_mask = np.zeros(SHAPE[:2], dtype=bool)
        pass_mask.reshape(-1)[:116] = True
        c.raises(
            lambda: load_dynamic_landmark_recording(
                _save(root, "coverage_115.npz", valid_mask=fail_mask)
            ),
            ValueError,
            "115 of 128 frames is below the 90 percent gate",
        )
        record = load_dynamic_landmark_recording(
            _save(root, "coverage_116.npz", valid_mask=pass_mask)
        )
        c.eq(int(record.valid_mask.sum()), 116,
             "116 of 128 frames clears the exact coverage gate")


def test_collection_requires_unique_recordings_but_allows_shared_groups(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        first = _save(root, "first.npz")
        second = _save(
            root, "second.npz",
            recording_id=np.asarray(RECORDING_BETA),
            group_id=np.asarray(GROUP_ALPHA),
        )
        records = load_dynamic_landmark_recordings((second, first))
        c.eq(tuple(record.recording_id for record in records),
             (RECORDING_ALPHA, RECORDING_BETA),
             "collection order is deterministic")
        duplicate = _save(root, "duplicate.npz")
        c.raises(lambda: load_dynamic_landmark_recordings((first, duplicate)),
                 ValueError, "recording ids must be unique across caches")


def test_per_second_differences_require_valid_consecutive_endpoints(c: Check):
    features = np.asarray([[[0.0, 1.0], [2.0, 5.0], [7.0, 11.0],
                            [8.0, 13.0]]], dtype=np.float32)
    mask = np.asarray([[True, True, False, True]], dtype=bool)
    timestamps = np.asarray([[0.0, 0.5, 1.0, 1.5]], dtype=np.float64)
    frame_indices = np.asarray([[10, 11, 12, 13]], dtype=np.int64)

    delta, delta_valid = per_second_first_differences(
        features, mask, timestamps, frame_indices)
    c.eq(delta.shape, features.shape, "delta preserves feature shape")
    c.eq(delta_valid.tolist(), [[False, True, False, False]],
         "detector gaps are never bridged")
    c.true(np.allclose(delta[0, 1], np.asarray([4.0, 8.0], np.float32)),
           "valid delta is scaled per second")
    c.true(np.all(delta[~delta_valid] == 0), "invalid deltas are canonical zero")


def test_per_second_differences_reject_frame_and_time_gaps(c: Check):
    features = np.arange(15, dtype=np.float32).reshape(1, 5, 3)
    mask = np.ones((1, 5), dtype=bool)
    timestamps = np.asarray([[0.0, 0.5, 0.5, np.nan, 2.0]], dtype=np.float64)
    frame_indices = np.asarray([[20, 22, 23, 24, 25]], dtype=np.int64)
    delta, delta_valid = per_second_first_differences(
        features, mask, timestamps, frame_indices)
    c.eq(delta_valid.tolist(), [[False, False, False, False, False]],
         "nonconsecutive frames and nonincreasing/nonfinite times are invalid")
    c.true(np.all(delta == 0), "no invalid difference leaks a value")


def test_per_second_differences_do_not_accept_unsigned_wraparound(c: Check):
    features = np.asarray([[[0.0], [1.0]]], dtype=np.float32)
    mask = np.ones((1, 2), dtype=bool)
    timestamps = np.asarray([[5, 4]], dtype=np.uint64)
    frame_indices = np.asarray(
        [[np.iinfo(np.uint64).max, 0]], dtype=np.uint64
    )
    delta, delta_valid = per_second_first_differences(
        features, mask, timestamps, frame_indices)
    c.eq(delta_valid.tolist(), [[False, False]],
         "decreasing unsigned metadata cannot wrap into a valid pair")
    c.true(np.all(delta == 0), "wrapped metadata never produces a delta")


def test_per_second_differences_zero_overflowed_finite_pairs(c: Check):
    limit = np.finfo(np.float32).max
    features = np.asarray([[[-limit], [limit]]], dtype=np.float32)
    mask = np.ones((1, 2), dtype=bool)
    timestamps = np.asarray([[0.0, 1.0]], dtype=np.float64)
    frame_indices = np.asarray([[30, 31]], dtype=np.int64)
    delta, delta_valid = per_second_first_differences(
        features, mask, timestamps, frame_indices)
    c.eq(delta_valid.tolist(), [[False, False]],
         "an unrepresentable delta is not marked valid")
    c.true(np.isfinite(delta).all() and np.all(delta == 0),
           "overflowed deltas are canonical finite zero")


if __name__ == "__main__":
    run_all("test_dynamic_window_data", dict(globals()))
