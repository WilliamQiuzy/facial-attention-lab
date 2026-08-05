from __future__ import annotations

import unittest

import numpy as np

from landmark110d.estimator import (
    FIXED_C,
    FIXED_THRESHOLD,
    Landmark110DEstimator,
    equal_group_weights,
)


class EstimatorContractTests(unittest.TestCase):
    def test_group_weights_give_each_group_equal_total_weight(self) -> None:
        groups = np.asarray(("a", "a", "b", "c", "c", "c"))
        weights = equal_group_weights(groups)

        totals = {
            group: float(weights[groups == group].sum())
            for group in np.unique(groups)
        }
        np.testing.assert_allclose(tuple(totals.values()), (1.0, 1.0, 1.0))

    def test_fixed_estimator_fits_and_predicts_110d_rows(self) -> None:
        rng = np.random.default_rng(20260805)
        x = rng.normal(size=(20, 110))
        y = np.asarray([0, 1] * 10, dtype=np.int64)
        x[:, 0] += y * 2.0
        groups = np.asarray([f"group-{index}" for index in range(20)])

        estimator = Landmark110DEstimator().fit(x, y, groups)
        probabilities = estimator.predict_proba(x)
        labels = estimator.predict(x)

        self.assertEqual(estimator.c, FIXED_C)
        self.assertEqual(estimator.threshold, FIXED_THRESHOLD)
        self.assertEqual(probabilities.shape, (20,))
        self.assertEqual(labels.shape, (20,))
        self.assertTrue(np.logical_and(probabilities > 0, probabilities < 1).all())
        np.testing.assert_array_equal(labels, probabilities >= FIXED_THRESHOLD)

    def test_json_round_trip_preserves_predictions(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.normal(size=(12, 110))
        y = np.asarray([0, 1] * 6, dtype=np.int64)
        groups = np.asarray([f"group-{index}" for index in range(12)])
        fitted = Landmark110DEstimator().fit(x, y, groups)

        restored = Landmark110DEstimator.from_dict(fitted.to_dict())

        np.testing.assert_allclose(
            restored.predict_proba(x), fitted.predict_proba(x), rtol=0, atol=1e-14
        )

    def test_rejects_non_110d_input(self) -> None:
        x = np.zeros((4, 109), dtype=np.float64)
        y = np.asarray((0, 1, 0, 1))
        groups = np.asarray(("a", "b", "c", "d"))

        with self.assertRaisesRegex(ValueError, "110"):
            Landmark110DEstimator().fit(x, y, groups)

    def test_rejects_a_group_that_crosses_binary_labels(self) -> None:
        x = np.zeros((4, 110), dtype=np.float64)
        y = np.asarray((0, 1, 0, 1))
        groups = np.asarray(("same", "same", "c", "d"))

        with self.assertRaisesRegex(ValueError, "group"):
            Landmark110DEstimator().fit(x, y, groups)

    def test_rejects_missing_group_ids(self) -> None:
        groups = np.asarray(("a", float("nan"), float("nan")), dtype=object)

        with self.assertRaisesRegex(ValueError, "group_ids"):
            equal_group_weights(groups)


if __name__ == "__main__":
    unittest.main()
