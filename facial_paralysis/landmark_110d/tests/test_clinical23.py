from __future__ import annotations

import unittest

import numpy as np

from landmark110d.clinical23 import CLINICAL23_NAMES, clinical23_from_mediapipe


def _face_mesh() -> np.ndarray:
    mesh = np.zeros((478, 3), dtype=np.float32)
    mesh[:, 0] = 0.5
    mesh[:, 1] = 0.5

    for index in (33, 133, 159, 145, 160, 144, 158, 153):
        mesh[index] = (0.35, 0.42, 0.0)
    mesh[33] = (0.30, 0.42, 0.0)
    mesh[133] = (0.40, 0.42, 0.0)
    for index in (159, 158, 160):
        mesh[index, 1] = 0.40
    for index in (145, 144, 153):
        mesh[index, 1] = 0.44

    for index in (263, 362, 386, 374, 387, 373, 385, 380):
        mesh[index] = (0.65, 0.42, 0.0)
    mesh[263] = (0.70, 0.42, 0.0)
    mesh[362] = (0.60, 0.42, 0.0)
    for index in (386, 385, 387):
        mesh[index, 1] = 0.40
    for index in (374, 380, 373):
        mesh[index, 1] = 0.44

    for index in (70, 63, 105, 66, 107):
        mesh[index] = (0.35, 0.34, 0.0)
    for index in (300, 293, 334, 296, 336):
        mesh[index] = (0.65, 0.34, 0.0)

    mesh[61] = (0.40, 0.62, 0.0)
    mesh[291] = (0.60, 0.62, 0.0)
    mesh[13] = (0.50, 0.60, 0.0)
    mesh[14] = (0.50, 0.64, 0.0)
    return mesh


class Clinical23ContractTests(unittest.TestCase):
    def test_extracts_finite_23d_geometry_from_mediapipe_mesh(self) -> None:
        vector = clinical23_from_mediapipe(_face_mesh(), 640, 480)

        self.assertEqual(len(CLINICAL23_NAMES), 23)
        self.assertEqual(vector.shape, (23,))
        self.assertTrue(np.isfinite(vector).all())
        self.assertGreater(vector[4], 0)
        self.assertGreater(vector[5], 0)
        self.assertGreater(vector[21], 0)
        self.assertGreater(vector[22], 0)

    def test_rejects_missing_mesh_points(self) -> None:
        with self.assertRaisesRegex(ValueError, "landmarks"):
            clinical23_from_mediapipe(np.zeros((100, 3)), 640, 480)


if __name__ == "__main__":
    unittest.main()
