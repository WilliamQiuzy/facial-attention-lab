"""Leak-safe trajectory summaries and train-control reference distances."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.trajectory_features import (  # noqa: E402
    BLENDSHAPE_DIM,
    FUSION_DIM,
    LANDMARK_BILATERAL_PAIRS,
    LANDMARK_DIM,
    LANDMARK_REGIONS,
    RAO_FUSION_DIM,
    HealthyReferencePrototype,
    bilateral_dynamics,
    gaussian_wasserstein_distance,
    summarize_trajectory_channels,
    trajectory_feature_names,
    trajectory_feature_set,
)
from _testlib import Check, run_all  # noqa: E402


def _recording(seed: int = 0):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(4, 32, 95)).astype(np.float32)
    mask = np.ones((4, 32), dtype=bool)
    timestamps = np.stack([
        window * 10.0 + np.arange(32, dtype=np.float64) * 0.1
        for window in range(4)
    ])
    source_indices = np.stack([
        window * 100 + np.arange(32, dtype=np.int64)
        for window in range(4)
    ])
    return features, mask, timestamps, source_indices


def test_masked_summaries_use_seconds_and_never_cross_windows(c: Check):
    features, mask, timestamps, source_indices = _recording()
    within_window = np.arange(32, dtype=np.float32)
    features[..., 0] = within_window
    timestamps[:] = np.stack([
        window * 100.0 + np.arange(32, dtype=np.float64) * 0.5
        for window in range(4)
    ])

    # A flattened implementation would create large derivatives at the three
    # window boundaries. Correct per-window derivatives are all zero here.
    for window, level in enumerate((0.0, 100.0, 200.0, 300.0)):
        features[window, :, 1] = level

    # A detector gap must not be bridged. The zero at the invalid frame is
    # canonical cache padding; the valid values on either side differ sharply.
    full_mask = mask.copy()
    features[..., 2] = 100.0
    features[0, 0, 2] = 0.0
    features[0, 1, 2] = 0.0
    mask[0, 1] = False

    got = summarize_trajectory_channels(
        features, full_mask, timestamps, source_indices, (0, 1)
    ).reshape(2, 4)
    gap_summary = summarize_trajectory_channels(
        features, mask, timestamps, source_indices, (2,)
    ).reshape(1, 4)
    c.true(np.allclose(got[0], (15.5, 15.5, 31.0, 2.0)),
           "median, IQR, range, and per-second velocity are exact")
    c.eq(float(got[1, 3]), 0.0,
         "no derivative is formed between independent windows")
    c.eq(float(gap_summary[0, 3]), 0.0,
         "no derivative is formed across a detector gap")


def test_bilateral_statistics_have_explicit_capture_side_semantics(c: Check):
    rng = np.random.default_rng(7)
    first = rng.normal(size=(4, 32))
    second = 0.5 * first + 3.0
    mask = np.ones((4, 32), dtype=bool)
    timestamps = np.stack([
        window * 10.0 + np.arange(32, dtype=np.float64) * 0.1
        for window in range(4)
    ])
    corr, amplitude_ratio, lag_seconds = bilateral_dynamics(
        first, second, mask, timestamps
    )
    c.true(abs(corr - 1.0) < 1e-12, "Pearson correlation")
    c.true(abs(amplitude_ratio - 0.5) < 1e-12,
           "capture-side amplitude ratio is orientation invariant")
    c.eq(lag_seconds, 0.0, "synchronous trajectories have zero lag")

    expected_pairs = (
        ("fissure_h_mesh33_vs_mesh263", 72, 73),
        ("fissure_w_mesh33_vs_mesh263", 76, 77),
        ("eye_area_mesh33_vs_mesh263", 79, 80),
        ("brow_h_mesh33_vs_mesh263", 82, 83),
        ("corner_y_mesh61_vs_mesh291", 86, 87),
        ("corner_x_mesh61_vs_mesh291", 90, 91),
    )
    c.eq(LANDMARK_BILATERAL_PAIRS, expected_pairs,
         "pairs are frozen to capture-side mesh anchors")


def test_lagged_cross_correlation_returns_seconds_without_cross_window_pairs(c: Check):
    rng = np.random.default_rng(31)
    first = rng.normal(size=(4, 32))
    second = np.zeros_like(first)
    second[:, 2:] = first[:, :-2]
    mask = np.ones((4, 32), dtype=bool)
    # The leading values have no delayed counterpart and should not enter the
    # pair calculation for the known two-frame lag.
    mask[:, :2] = False
    timestamps = np.stack([
        window * 20.0 + np.arange(32, dtype=np.float64) * 0.125
        for window in range(4)
    ])
    _, _, lag_seconds = bilateral_dynamics(
        first, second, mask, timestamps, max_lag_frames=5
    )
    c.true(abs(lag_seconds - 0.25) < 1e-12,
           "positive lag means the second capture-side signal occurs later")


def test_feature_registry_has_frozen_dimensions_and_no_cross_window_deltas(c: Check):
    features, mask, timestamps, source_indices = _recording(4)
    blendshape = trajectory_feature_set(
        "blendshape", features, mask, timestamps, source_indices
    )
    landmark = trajectory_feature_set(
        "landmark", features, mask, timestamps, source_indices
    )
    fusion = trajectory_feature_set(
        "fusion", features, mask, timestamps, source_indices
    )
    c.eq(BLENDSHAPE_DIM, 288, "72 channels times four summaries")
    c.eq(LANDMARK_DIM, 110, "23 summaries plus six bilateral triplets")
    c.eq(FUSION_DIM, 398, "blendshape plus landmark")
    c.eq(RAO_FUSION_DIM, 402, "four train-control distances appended")
    c.eq(blendshape.shape, (BLENDSHAPE_DIM,), "blendshape vector shape")
    c.eq(landmark.shape, (LANDMARK_DIM,), "landmark vector shape")
    c.eq(fusion.shape, (FUSION_DIM,), "fusion vector shape")
    c.true(np.array_equal(fusion[:BLENDSHAPE_DIM], blendshape),
           "fusion begins with the exact blendshape vector")
    c.true(np.array_equal(fusion[BLENDSHAPE_DIM:], landmark),
           "fusion ends with the exact landmark vector")
    for feature_set, expected in (
        ("blendshape", BLENDSHAPE_DIM),
        ("landmark", LANDMARK_DIM),
        ("fusion", FUSION_DIM),
        ("rao_fusion", RAO_FUSION_DIM),
    ):
        names = trajectory_feature_names(feature_set)
        c.eq(len(names), expected, f"{feature_set} names match its vector")
        c.eq(len(set(names)), expected, f"{feature_set} names are unique")


def test_region_map_is_frozen_to_eye_brow_and_mouth_columns(c: Check):
    c.eq(LANDMARK_REGIONS, (
        ("eye", tuple(range(72, 82))),
        ("brow", tuple(range(82, 86))),
        ("mouth", tuple(range(86, 95))),
    ), "three Rao-style regions cover each clinical23 column exactly once")


def test_gaussian_wasserstein_distance_is_zero_for_identical_gaussians(c: Check):
    mean = np.asarray([1.0, -2.0])
    covariance = np.asarray([[2.0, 0.3], [0.3, 1.0]])
    same = gaussian_wasserstein_distance(mean, covariance, mean, covariance)
    shifted = gaussian_wasserstein_distance(
        mean, covariance, mean + np.asarray([3.0, 4.0]), covariance
    )
    c.true(abs(same) < 1e-10, "identical Gaussian distance is zero")
    c.true(abs(shifted - 5.0) < 1e-10,
           "equal-covariance distance reduces to mean Euclidean distance")


def test_healthy_reference_fits_only_the_control_rows_it_receives(c: Check):
    records = [_recording(seed) for seed in range(5)]
    features = np.stack([record[0] for record in records])
    masks = np.stack([record[1] for record in records])
    timestamps = np.stack([record[2] for record in records])
    source_indices = np.stack([record[3] for record in records])
    prototype = HealthyReferencePrototype().fit(
        features[:3], masks[:3], timestamps[:3], source_indices[:3],
        record_indices=np.asarray([10, 11, 12]),
    )
    distances = prototype.transform(
        features[3:], masks[3:], timestamps[3:], source_indices[3:]
    )
    c.eq(tuple(prototype.fit_record_indices_.tolist()), (10, 11, 12),
         "prototype records exactly the supplied control indices")
    c.eq(distances.shape, (2, 4),
         "one correlation Mahalanobis plus three regional W2 distances")
    c.true(np.isfinite(distances).all(), "all reference distances are finite")
    c.true(np.all(distances >= 0.0), "all reference distances are nonnegative")


def test_trajectory_features_fail_closed_on_malformed_arrays(c: Check):
    features, mask, timestamps, source_indices = _recording()
    c.raises(lambda: summarize_trajectory_channels(
        features[:, :-1], mask, timestamps, source_indices, (0,)), ValueError,
        "feature shape is frozen")
    c.raises(lambda: summarize_trajectory_channels(
        features, mask.astype(np.uint8), timestamps, source_indices, (0,)), ValueError,
        "mask must be bool")
    bad = features.copy()
    bad[0, 0, 0] = np.nan
    c.raises(lambda: summarize_trajectory_channels(
        bad, mask, timestamps, source_indices, (0,)), ValueError,
        "valid features must be finite")
    c.raises(lambda: trajectory_feature_set(
        "unknown", features, mask, timestamps, source_indices), ValueError,
        "only frozen feature sets are accepted")


if __name__ == "__main__":
    run_all("test_trajectory_features", dict(globals()))
