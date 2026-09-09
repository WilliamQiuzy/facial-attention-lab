from __future__ import annotations

import math

import numpy as np
import pytest

from gaze_compare.metrics import (
    angular_accuracy,
    bland_altman,
    cluster_bootstrap_mean,
    data_loss,
    density_centroid_distance,
    effective_sampling_rate,
    estimate_temporal_lag,
    histogram_intersection,
    hotspot_dice,
    interval_cv,
    jensen_shannon_distance,
    lin_concordance,
    map_correlation,
    paired_equivalence,
    rms_precision,
    total_variation_distance,
)


def test_calibration_quality_metrics_have_known_answers() -> None:
    gaze = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    targets = np.zeros_like(gaze)

    assert angular_accuracy(gaze, targets) == pytest.approx(5.0)
    assert rms_precision(gaze) == pytest.approx(5.0)
    assert data_loss([True, False, True, False]) == pytest.approx(0.5)
    assert effective_sampling_rate([0.0, 10.0, 20.0, 30.0]) == pytest.approx(100.0)
    assert interval_cv([0.0, 10.0, 20.0, 30.0]) == pytest.approx(0.0)


def test_identical_density_maps_have_perfect_agreement() -> None:
    density = np.array([[0.1, 0.2], [0.3, 0.4]])

    assert map_correlation(density, density) == pytest.approx(1.0)
    assert histogram_intersection(density, density) == pytest.approx(1.0)
    assert jensen_shannon_distance(density, density) == pytest.approx(0.0)
    assert hotspot_dice(density, density, quantile=0.5) == pytest.approx(1.0)
    assert density_centroid_distance(density, density) == pytest.approx(0.0)


def test_spatial_and_aoi_disagreement_metrics_have_known_answers() -> None:
    left = np.array([[1.0, 0.0], [0.0, 0.0]])
    right = np.array([[0.0, 1.0], [0.0, 0.0]])

    assert histogram_intersection(left, right) == pytest.approx(0.0)
    assert jensen_shannon_distance(left, right) == pytest.approx(1.0)
    assert hotspot_dice(left, right, quantile=0.5) == pytest.approx(0.0)
    assert density_centroid_distance(left, right) == pytest.approx(1.0)
    assert total_variation_distance([0.8, 0.2], [0.5, 0.5]) == pytest.approx(0.3)


def test_lin_concordance_penalizes_location_shift() -> None:
    reference = np.array([1.0, 2.0, 3.0, 4.0])

    assert lin_concordance(reference, reference) == pytest.approx(1.0)
    assert lin_concordance(reference + 2.0, reference) < 1.0


def test_bland_altman_uses_one_value_per_participant() -> None:
    webcam = np.array([2.0, 4.0, 6.0])
    reference = np.array([1.0, 2.0, 3.0])

    summary = bland_altman(webcam, reference)

    assert summary.n == 3
    assert summary.mean_difference == pytest.approx(2.0)
    expected_sd = np.std([1.0, 2.0, 3.0], ddof=1)
    assert summary.lower_limit == pytest.approx(2.0 - 1.96 * expected_sd)
    assert summary.upper_limit == pytest.approx(2.0 + 1.96 * expected_sd)


def test_cluster_bootstrap_is_deterministic_and_resamples_clusters() -> None:
    values = np.array([1.0, 3.0, 10.0, 14.0])
    clusters = np.array(["P1", "P1", "P2", "P2"])

    first = cluster_bootstrap_mean(values, clusters, n_boot=1000, seed=7)
    second = cluster_bootstrap_mean(values, clusters, n_boot=1000, seed=7)

    assert first == second
    assert first.estimate == pytest.approx(7.0)
    assert first.lower <= first.estimate <= first.upper


@pytest.mark.parametrize(
    ("differences", "margin", "expected"),
    [
        ([0.01, -0.02, 0.0, 0.01, -0.01, 0.0], 0.10, "equivalent"),
        ([0.8, 0.9, 1.0, 0.9, 1.1, 1.0], 0.20, "not_equivalent"),
        ([0.0, 0.3, -0.2, 0.25, -0.1, 0.1], 0.10, "inconclusive"),
    ],
)
def test_equivalence_has_exact_three_way_decision_rule(
    differences: list[float],
    margin: float,
    expected: str,
) -> None:
    result = paired_equivalence(differences, margin=margin)

    assert result.outcome == expected
    assert result.confidence == pytest.approx(0.90)
    assert math.isfinite(result.mean_difference)


def test_metrics_reject_empty_or_non_finite_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        angular_accuracy(np.array([[math.nan, 0.0]]), np.array([[0.0, 0.0]]))
    with pytest.raises(ValueError, match="positive density"):
        histogram_intersection(np.zeros((2, 2)), np.ones((2, 2)))


def test_temporal_lag_estimation_resamples_to_a_common_clock() -> None:
    reference_time = np.arange(0.0, 3000.0, 1000 / 120)
    webcam_time = np.arange(0.0, 3000.0, 1000 / 30)

    def trajectory(time_ms: np.ndarray) -> np.ndarray:
        seconds = time_ms / 1000
        return np.column_stack(
            [np.sin(2 * np.pi * 0.7 * seconds), np.cos(2 * np.pi * 0.43 * seconds)]
        )

    reference_xy = trajectory(reference_time)
    webcam_xy = trajectory(np.maximum(0, webcam_time - 200.0))

    result = estimate_temporal_lag(
        webcam_time,
        webcam_xy,
        reference_time,
        reference_xy,
        resample_hz=30,
        max_lag_ms=500,
    )

    assert result.lag_ms == pytest.approx(200.0, abs=35.0)
    assert result.peak_correlation > 0.95
