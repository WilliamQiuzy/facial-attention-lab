from __future__ import annotations

import numpy as np
import pytest

from gaze_compare.cohort_metrics import (
    independent_equivalence,
    repeated_cross_validated_auc,
    standardized_mean_difference,
)


def test_standardized_mean_difference_has_known_direction() -> None:
    webcam = np.array([3.0, 4.0, 5.0, 6.0])
    professional = np.array([1.0, 2.0, 3.0, 4.0])

    assert standardized_mean_difference(webcam, professional) > 0
    assert standardized_mean_difference(webcam, webcam) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("webcam", "professional", "margin", "expected"),
    [
        ([1.00] * 30, [0.95] * 30, 0.20, "similar_within_margin"),
        ([1.50] * 30, [0.50] * 30, 0.20, "meaningfully_different"),
        (
            [0.0, 2.0] * 15,
            [0.9, 1.1] * 15,
            0.10,
            "inconclusive",
        ),
    ],
)
def test_independent_equivalence_uses_three_way_rule(
    webcam: list[float],
    professional: list[float],
    margin: float,
    expected: str,
) -> None:
    result = independent_equivalence(webcam, professional, margin=margin)

    assert result.outcome == expected
    assert result.confidence == pytest.approx(0.90)
    assert result.n_webcam == 30
    assert result.n_professional == 30


def test_domain_auc_is_near_chance_for_uninformative_features() -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(200, 3))
    labels = np.repeat([0, 1], 100)

    result = repeated_cross_validated_auc(
        features,
        labels,
        n_splits=5,
        n_repeats=4,
        n_boot=200,
        seed=7,
    )

    assert 0.40 <= result.auc <= 0.60
    assert result.lower <= result.auc <= result.upper


def test_domain_auc_detects_separable_features() -> None:
    rng = np.random.default_rng(9)
    labels = np.repeat([0, 1], 100)
    features = rng.normal(loc=labels[:, None] * 2.5, scale=0.7, size=(200, 2))

    result = repeated_cross_validated_auc(
        features,
        labels,
        n_splits=5,
        n_repeats=3,
        n_boot=200,
        seed=8,
    )

    assert result.auc > 0.90
