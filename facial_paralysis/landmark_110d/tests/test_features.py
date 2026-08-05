from __future__ import annotations

import unittest

import numpy as np

from landmark110d.features import FEATURE_NAMES, build_110d_features


SOURCE_FRAME_COUNT = 432


def _recording() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    clinical23 = np.zeros((4, 32, 23), dtype=np.float32)
    frame = np.arange(32, dtype=np.float32)
    for window in range(4):
        for channel in range(23):
            clinical23[window, :, channel] = (
                (channel + 1) * 0.01 * (frame + window * 0.1)
            )
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


class FeatureContractTests(unittest.TestCase):
    def test_feature_names_freeze_the_110d_contract(self) -> None:
        self.assertEqual(len(FEATURE_NAMES), 110)
        self.assertEqual(len(set(FEATURE_NAMES)), 110)
        self.assertEqual(FEATURE_NAMES[0], "fissure_h_mesh33__median")
        self.assertEqual(
            FEATURE_NAMES[-1],
            "corner_x_mesh61_vs_mesh291__lag_seconds",
        )

    def test_builds_expected_channel_and_bilateral_statistics(self) -> None:
        clinical23, valid, timestamps, source_indices = _recording()
        vector = build_110d_features(
            clinical23, valid, timestamps, source_indices, SOURCE_FRAME_COUNT
        )

        self.assertEqual(vector.shape, (110,))
        self.assertTrue(np.isfinite(vector).all())
        np.testing.assert_allclose(
            vector[:4],
            np.asarray((0.1565, 0.1565, 0.313, 0.3)),
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            vector[92:95],
            np.asarray((1.0, 0.5, 0.0)),
            rtol=0,
            atol=1e-12,
        )

    def test_velocity_never_bridges_window_boundaries(self) -> None:
        clinical23, valid, timestamps, source_indices = _recording()
        clinical23[:, :, 0] = 0.0
        clinical23[1:, :, 0] = 1000.0

        vector = build_110d_features(
            clinical23, valid, timestamps, source_indices, SOURCE_FRAME_COUNT
        )

        self.assertEqual(vector[3], 0.0)

    def test_integer_timestamp_differences_are_computed_in_float64(self) -> None:
        clinical23 = np.zeros((4, 32, 23), dtype=np.float32)
        clinical23[0, 1:, 0] = 1.0
        valid = np.ones((4, 32), dtype=bool)
        timestamps = np.stack([
            np.arange(32, dtype=np.int64) for _ in range(4)
        ])
        timestamps[0, 0] = np.iinfo(np.int64).min + 1
        high_start = np.iinfo(np.int64).max - 31 * 4096
        timestamps[0, 1:] = high_start + np.arange(31, dtype=np.int64) * 4096
        source_indices = _recording()[3]

        vector = build_110d_features(
            clinical23, valid, timestamps, source_indices, SOURCE_FRAME_COUNT
        )

        expected = 1.0 / (
            np.float64(timestamps[0, 1]) - np.float64(timestamps[0, 0])
        )
        self.assertEqual(vector[3], expected)

    def test_rejects_timestamps_that_collapse_in_float64(self) -> None:
        clinical23, valid, timestamps, source_indices = _recording()
        timestamps = np.stack([
            np.arange(32, dtype=np.int64) for _ in range(4)
        ])
        timestamps[0, 0] = np.iinfo(np.int64).min + 1
        timestamps[0, 1:] = np.arange(
            np.iinfo(np.int64).max - 30,
            np.iinfo(np.int64).max + 1,
            dtype=np.int64,
        )

        with self.assertRaisesRegex(ValueError, "timestamps"):
            build_110d_features(
                clinical23, valid, timestamps, source_indices, SOURCE_FRAME_COUNT
            )

    def test_rejects_nonadjacent_source_frames(self) -> None:
        clinical23, valid, timestamps, source_indices = _recording()
        source_indices[0, 10] += 1

        with self.assertRaisesRegex(ValueError, "adjacent"):
            build_110d_features(
                clinical23, valid, timestamps, source_indices, SOURCE_FRAME_COUNT
            )

    def test_rejects_recordings_below_ninety_percent_coverage(self) -> None:
        clinical23, valid, timestamps, source_indices = _recording()
        valid.reshape(-1)[:13] = False
        clinical23[~valid] = 0.0

        with self.assertRaisesRegex(ValueError, "coverage"):
            build_110d_features(
                clinical23, valid, timestamps, source_indices, SOURCE_FRAME_COUNT
            )

    def test_rejects_negative_source_indices(self) -> None:
        clinical23, valid, timestamps, source_indices = _recording()
        source_indices[0] -= 1

        with self.assertRaisesRegex(ValueError, "nonnegative"):
            build_110d_features(
                clinical23, valid, timestamps, source_indices, SOURCE_FRAME_COUNT
            )

    def test_rejects_noncanonical_window_starts(self) -> None:
        clinical23, valid, timestamps, source_indices = _recording()
        source_indices[1] += 1

        with self.assertRaisesRegex(ValueError, "frozen|deterministic"):
            build_110d_features(
                clinical23, valid, timestamps, source_indices, SOURCE_FRAME_COUNT
            )

    def test_rejects_nonzero_values_in_masked_rows(self) -> None:
        clinical23, valid, timestamps, source_indices = _recording()
        valid[0, 0] = False
        clinical23[0, 0, 0] = 1.0

        with self.assertRaisesRegex(ValueError, "canonical zero"):
            build_110d_features(
                clinical23, valid, timestamps, source_indices, SOURCE_FRAME_COUNT
            )

    def test_rejects_windows_that_do_not_span_declared_recording(self) -> None:
        clinical23, valid, timestamps, source_indices = _recording()
        source_indices = np.stack([
            start + np.arange(32, dtype=np.int64)
            for start in (0, 32, 64, 96)
        ])

        with self.assertRaisesRegex(ValueError, "frozen|span"):
            build_110d_features(
                clinical23, valid, timestamps, source_indices, SOURCE_FRAME_COUNT
            )

    def test_rejects_wrong_clinical_dimension(self) -> None:
        clinical23, valid, timestamps, source_indices = _recording()

        with self.assertRaisesRegex(ValueError, "4, 32, 23"):
            build_110d_features(
                clinical23[..., :-1],
                valid,
                timestamps,
                source_indices,
                SOURCE_FRAME_COUNT,
            )


if __name__ == "__main__":
    unittest.main()
