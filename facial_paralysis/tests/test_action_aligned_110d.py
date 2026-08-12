"""Contracts for deterministic Action-Aligned Landmark 110D v1."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
from src.preprocessing.action_aligned_110d import (  # noqa: E402
    ACTION_SLOT_ORDER,
    action_aligned_feature_vector,
    mirror_action_aligned_features,
    mirror_clinical23_features,
    select_action_window_starts,
)
from src.preprocessing.trajectory_features import trajectory_feature_set  # noqa: E402


def _proposal() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.zeros((121, 95), dtype=np.float32)
    valid = np.ones(121, dtype=bool)
    source_indices = np.arange(121, dtype=np.int64) * 5
    peaks = {
        "brow": (10, (3, 4, 5)),
        "gentle_eye": (20, (9, 10)),
        "tight_eye": (30, (7, 8, 19, 20)),
        "smile_first": (40, (44, 45)),
        "pucker": (50, (38,)),
        "lower_teeth": (60, (25, 34, 35)),
        "smile_second": (80, (44, 45)),
    }
    for _name, (row, columns) in peaks.items():
        features[row, list(columns)] = 1.0
    features[80, [44, 45]] = 0.9
    return features, valid, source_indices


def _four_window_recording() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260811)
    features = rng.normal(size=(4, 32, 95)).astype(np.float32)
    valid = np.ones((4, 32), dtype=bool)
    indices = np.stack([
        np.arange(start, start + 32, dtype=np.int64)
        for start in (0, 100, 200, 300)
    ])
    timestamps = indices.astype(np.float64) / 30.0
    return features, valid, timestamps, indices


def test_slot_registry_and_known_peaks_are_exact(c: Check):
    c.eq(ACTION_SLOT_ORDER, (
        "eyebrow_rise", "gentle_eye_closure", "tight_eye_squeeze",
        "relaxed_smile", "lip_pucker", "lower_teeth_show",
        "reanimated_smile",
    ))
    features, valid, indices = _proposal()
    starts = select_action_window_starts(
        features, valid, indices, source_frame_count=640
    )
    c.eq(starts, (34, 84, 134, 184, 234, 284, 384))


def test_two_smile_slots_use_distinct_time_separated_peaks(c: Check):
    features, valid, indices = _proposal()
    features[41, [44, 45]] = 2.0
    starts = select_action_window_starts(
        features, valid, indices, source_frame_count=640
    )
    c.eq(starts[3], 189, "nearby samples collapse to the strongest point in the bout")
    c.eq(starts[6], 384, "second smile comes from a distinct bout")


def test_peak_windows_clamp_to_video_edges(c: Check):
    features, valid, indices = _proposal()
    features[:, [3, 4, 5]] = 0.0
    features[0, [3, 4, 5]] = 1.0
    features[:, [38]] = 0.0
    features[-1, [38]] = 1.0
    starts = select_action_window_starts(
        features, valid, indices, source_frame_count=601
    )
    c.eq(starts[0], 0)
    c.eq(starts[4], 569)


def test_window_timing_scales_with_source_fps(c: Check):
    features, valid, indices = _proposal()
    starts = select_action_window_starts(
        features, valid, indices * 2, source_frame_count=1280, source_fps=60.0
    )
    c.eq(starts, (68, 168, 268, 368, 468, 568, 768))


def test_action_aligned_pooling_is_110d_and_four_window_bytes_do_not_change(c: Check):
    rng = np.random.default_rng(17)
    features = rng.normal(size=(7, 32, 95)).astype(np.float32)
    valid = np.ones((7, 32), dtype=bool)
    indices = np.stack([
        np.arange(start, start + 32, dtype=np.int64)
        for start in (0, 50, 100, 150, 200, 250, 300)
    ])
    timestamps = indices.astype(np.float64) / 30.0
    vector = action_aligned_feature_vector(features, valid, timestamps, indices)
    c.eq(vector.shape, (110,))
    c.true(bool(np.isfinite(vector).all()))

    old = _four_window_recording()
    frozen = trajectory_feature_set("landmark", *old)
    c.eq(
        hashlib.sha256(frozen.tobytes()).hexdigest(),
        "20b5628acae67a69aeb3fae1c312821bddc6364578cecbab584ce23756844ccf",
        "generalizing the window count cannot change frozen 4-window 110D bytes",
    )


def test_action_aligned_pooling_is_invariant_to_native_video_fps(c: Check):
    rng = np.random.default_rng(20260812)
    features = rng.normal(size=(7, 32, 95)).astype(np.float32)
    valid = np.ones((7, 32), dtype=bool)
    starts_30hz = np.arange(7, dtype=np.int64) * 120
    indices_30hz = starts_30hz[:, None] + np.arange(32, dtype=np.int64)
    indices_60hz = (starts_30hz * 2)[:, None] + 2 * np.arange(32, dtype=np.int64)
    timestamps_30hz = indices_30hz.astype(np.float64) / 30.0
    timestamps_60hz = indices_60hz.astype(np.float64) / 60.0

    at_30hz = action_aligned_feature_vector(
        features, valid, timestamps_30hz, indices_30hz
    )
    at_60hz = action_aligned_feature_vector(
        features, valid, timestamps_60hz, indices_60hz
    )
    c.true(
        np.array_equal(at_30hz, at_60hz),
        "the same 30 Hz action samples must produce identical 110D vectors",
    )


def test_action_aligned_mirror_is_an_exact_involution(c: Check):
    rng = np.random.default_rng(29)
    features = rng.normal(size=(7, 32, 95)).astype(np.float32)
    mirrored = mirror_action_aligned_features(features)
    c.eq(mirrored.shape, features.shape)
    c.eq(mirrored.dtype, features.dtype)
    c.true(np.array_equal(mirror_action_aligned_features(mirrored), features))

    four_window = features[:4]
    from src.models.dynamic_landmark import horizontal_mirror_features
    import torch
    expected = horizontal_mirror_features(torch.from_numpy(four_window)).numpy()
    c.true(np.array_equal(mirror_clinical23_features(four_window), expected))


def test_proposals_fail_closed_on_malformed_or_insufficient_input(c: Check):
    features, valid, indices = _proposal()
    c.raises(
        lambda: select_action_window_starts(
            features[:, :-1], valid, indices, source_frame_count=640
        ),
        ValueError,
    )
    c.raises(
        lambda: select_action_window_starts(
            features, valid.astype(np.uint8), indices, source_frame_count=640
        ),
        ValueError,
    )
    bad_indices = indices.copy()
    bad_indices[2] = bad_indices[1]
    c.raises(
        lambda: select_action_window_starts(
            features, valid, bad_indices, source_frame_count=640
        ),
        ValueError,
    )
    sparse = valid.copy()
    sparse[:] = False
    sparse[0] = True
    c.raises(
        lambda: select_action_window_starts(
            features, sparse, indices, source_frame_count=640
        ),
        ValueError,
    )


if __name__ == "__main__":
    run_all("test_action_aligned_110d", dict(globals()))
