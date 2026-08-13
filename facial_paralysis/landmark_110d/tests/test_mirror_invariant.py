from __future__ import annotations

import unittest

import numpy as np

from landmark110d import (
    MirrorInvariantLandmark110DEstimator,
    build_110d_features,
    build_mirror_invariant_110d_views,
    mirror_clinical23,
)


SOURCE_FRAME_COUNT = 432


def _recording() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260805)
    clinical23 = rng.normal(size=(4, 32, 23)).astype(np.float32)
    valid = np.ones((4, 32), dtype=bool)
    timestamps = np.stack([
        window * 10.0 + np.arange(32, dtype=np.float64) / 30.0
        for window in range(4)
    ])
    source_indices = np.stack([
        start + np.arange(32, dtype=np.int64)
        for start in (0, 133, 266, 400)
    ])
    return clinical23, valid, timestamps, source_indices


class MirrorInvariantContractTests(unittest.TestCase):
    def test_clinical23_mirror_is_an_exact_involution(self) -> None:
        clinical23 = _recording()[0]
        mirrored = mirror_clinical23(clinical23)
        restored = mirror_clinical23(mirrored)

        np.testing.assert_array_equal(restored, clinical23)
        np.testing.assert_array_equal(mirrored[..., 0], clinical23[..., 1])
        np.testing.assert_array_equal(mirrored[..., 1], clinical23[..., 0])
        np.testing.assert_array_equal(mirrored[..., 3], -clinical23[..., 3])

    def test_view_builder_matches_two_frozen_110d_transforms(self) -> None:
        clinical23, valid, timestamps, source_indices = _recording()
        original, mirrored = build_mirror_invariant_110d_views(
            clinical23,
            valid,
            timestamps,
            source_indices,
            SOURCE_FRAME_COUNT,
        )

        np.testing.assert_array_equal(
            original,
            build_110d_features(
                clinical23,
                valid,
                timestamps,
                source_indices,
                SOURCE_FRAME_COUNT,
            ),
        )
        np.testing.assert_array_equal(
            mirrored,
            build_110d_features(
                mirror_clinical23(clinical23),
                valid,
                timestamps,
                source_indices,
                SOURCE_FRAME_COUNT,
            ),
        )

    def test_estimator_is_exactly_invariant_to_view_order(self) -> None:
        rng = np.random.default_rng(17)
        original = rng.normal(size=(20, 110))
        mirrored = rng.normal(size=(20, 110))
        labels = np.asarray([0, 1] * 10, dtype=np.int64)
        groups = np.asarray([f"group-{index}" for index in range(20)])
        estimator = MirrorInvariantLandmark110DEstimator().fit(
            original, mirrored, labels, groups
        )

        direct = estimator.predict_proba(original, mirrored)
        reversed_views = estimator.predict_proba(mirrored, original)

        np.testing.assert_array_equal(direct, reversed_views)
        np.testing.assert_array_equal(
            estimator.predict(original, mirrored), direct >= 0.5
        )

    def test_mirror_estimator_round_trip_preserves_predictions(self) -> None:
        rng = np.random.default_rng(23)
        original = rng.normal(size=(12, 110))
        mirrored = rng.normal(size=(12, 110))
        labels = np.asarray([0, 1] * 6, dtype=np.int64)
        groups = np.asarray([f"group-{index}" for index in range(12)])
        fitted = MirrorInvariantLandmark110DEstimator().fit(
            original, mirrored, labels, groups
        )
        restored = MirrorInvariantLandmark110DEstimator.from_dict(
            fitted.to_dict()
        )

        np.testing.assert_allclose(
            restored.predict_proba(original, mirrored),
            fitted.predict_proba(original, mirrored),
            rtol=0,
            atol=1e-14,
        )

    def test_rejects_misaligned_views(self) -> None:
        original = np.zeros((4, 110), dtype=np.float64)
        mirrored = np.zeros((3, 110), dtype=np.float64)
        labels = np.asarray((0, 1, 0, 1))
        groups = np.asarray(("a", "b", "c", "d"))

        with self.assertRaisesRegex(ValueError, "align"):
            MirrorInvariantLandmark110DEstimator().fit(
                original, mirrored, labels, groups
            )


if __name__ == "__main__":
    unittest.main()
