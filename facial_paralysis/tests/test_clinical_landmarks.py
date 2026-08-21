"""Contract tests for pose-normalized clinical MediaPipe landmark features."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.clinical_landmarks import (  # noqa: E402
    CLINICAL_LANDMARK_NAMES,
    clinical_landmark_features,
    legacy_clinical23_v1_features,
)
from _testlib import Check, run_all  # noqa: E402


def _face() -> np.ndarray:
    """Synthetic, bilaterally symmetric 478-point face in normalized coordinates."""
    p = np.full((478, 3), (0.5, 0.5, 0.0), dtype=np.float32)

    # Subject right is image-left.  All measurements are symmetric about x=.5.
    right_eye = {
        33: (0.30, 0.40), 133: (0.40, 0.40),
        159: (0.35, 0.38), 158: (0.37, 0.38), 160: (0.33, 0.38),
        145: (0.35, 0.42), 144: (0.33, 0.42), 153: (0.37, 0.42),
    }
    left_eye = {
        263: (0.70, 0.40), 362: (0.60, 0.40),
        386: (0.65, 0.38), 385: (0.63, 0.38), 387: (0.67, 0.38),
        374: (0.65, 0.42), 380: (0.67, 0.42), 373: (0.63, 0.42),
    }
    for idx, xy in {**right_eye, **left_eye}.items():
        p[idx, :2] = xy
    for idx, x in zip((70, 63, 105, 66, 107), np.linspace(0.30, 0.40, 5)):
        p[idx, :2] = (x, 0.30)
    for idx, x in zip((300, 293, 334, 296, 336), np.linspace(0.70, 0.60, 5)):
        p[idx, :2] = (x, 0.30)

    for i, idx in enumerate((168, 6, 197, 195, 5, 4, 1, 19, 2, 164, 0, 17, 152, 10)):
        p[idx, :2] = (0.5, 0.25 + 0.04 * i)
    p[61, :2] = (0.40, 0.70)
    p[291, :2] = (0.60, 0.70)
    p[13, :2] = (0.50, 0.68)
    p[14, :2] = (0.50, 0.72)
    return p


def _rotate(p: np.ndarray, angle_rad: float) -> np.ndarray:
    out = p.copy()
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    r = np.array([[c, -s], [s, c]], dtype=np.float32)
    out[:, :2] = (out[:, :2] - 0.5) @ r.T + 0.5
    return out


def test_schema_and_dtype(c: Check):
    v = clinical_landmark_features(_face(), 1000, 1000)
    c.eq(len(CLINICAL_LANDMARK_NAMES), 23, "clinical schema is 23-dimensional")
    c.eq(v.shape, (23,), "vector shape")
    c.eq(v.dtype, np.float32, "vector dtype")
    c.true(bool(np.isfinite(v).all()), "all output values finite")


def test_symmetric_face_has_zero_asymmetry(c: Check):
    v = clinical_landmark_features(_face(), 1000, 1000)
    by_name = dict(zip(CLINICAL_LANDMARK_NAMES, v))
    for name in (
        "fissure_h_absdiff", "fissure_h_mesh33_minus_mesh263", "fissure_w_absdiff",
        "eye_area_absdiff", "brow_h_absdiff", "brow_h_mesh33_minus_mesh263",
        "corner_y_absdiff", "corner_y_mesh61_minus_mesh291", "commissure_x_absdiff",
    ):
        c.true(abs(float(by_name[name])) < 1e-5, f"{name} should be zero")


def test_translation_scale_and_roll_invariance(c: Check):
    base = _face()
    ref = clinical_landmark_features(base, 1000, 1000)

    translated = base.copy()
    translated[:, :2] += np.array([0.07, -0.04], np.float32)
    got_translation = clinical_landmark_features(translated, 1000, 1000)

    # Uniform pixel scaling must cancel through interocular normalization.
    got_scale = clinical_landmark_features(base, 2000, 2000)
    got_roll = clinical_landmark_features(_rotate(base, 0.23), 1000, 1000)

    c.true(bool(np.allclose(ref, got_translation, atol=2e-5)), "translation invariant")
    c.true(bool(np.allclose(ref, got_scale, atol=2e-5)), "uniform-scale invariant")
    c.true(bool(np.allclose(ref, got_roll, atol=2e-5)), "in-plane-roll invariant")


def test_mouth_perturbation_has_expected_direction(c: Check):
    p = _face()
    p[291, 1] += 0.04  # mesh-291 corner moves downward in image coordinates
    v = clinical_landmark_features(p, 1000, 1000)
    by_name = dict(zip(CLINICAL_LANDMARK_NAMES, v))
    c.true(float(by_name["corner_y_absdiff"]) > 0.0, "absolute corner asymmetry grows")
    c.true(float(by_name["corner_y_mesh61_minus_mesh291"]) < 0.0,
           "mesh61-minus-mesh291 sign is preserved")


def test_horizontal_mirror_contract(c: Check):
    p = _face()
    p[291, 1] += 0.04
    mirrored = p.copy()
    pairs = list(zip(
        (33, 133, 159, 145, 160, 144, 158, 153),
        (263, 362, 386, 374, 387, 373, 385, 380),
    ))
    pairs += list(zip((70, 63, 105, 66, 107), (300, 293, 334, 296, 336)))
    pairs += [(61, 291)]
    for right, left in pairs:
        mirrored[right, :2] = (1.0 - p[left, 0], p[left, 1])
        mirrored[left, :2] = (1.0 - p[right, 0], p[right, 1])
    for idx in (168, 6, 197, 195, 5, 4, 1, 19, 2, 164, 0, 13, 14, 17, 152, 10):
        mirrored[idx, 0] = 1.0 - p[idx, 0]

    original = dict(zip(CLINICAL_LANDMARK_NAMES,
                        clinical_landmark_features(p, 1000, 1000)))
    flipped = dict(zip(CLINICAL_LANDMARK_NAMES,
                       clinical_landmark_features(mirrored, 1000, 1000)))
    for name in ("fissure_h_absdiff", "fissure_w_absdiff", "eye_area_absdiff",
                 "brow_h_absdiff", "corner_y_absdiff", "commissure_x_absdiff"):
        c.true(abs(float(original[name] - flipped[name])) < 2e-5,
               f"absolute feature {name} is mirror invariant")
    for name in ("fissure_h_mesh33_minus_mesh263",
                 "brow_h_mesh33_minus_mesh263",
                 "corner_y_mesh61_minus_mesh291"):
        c.true(abs(float(original[name] + flipped[name])) < 2e-5,
               f"signed feature {name} changes sign")


def test_invalid_input_fails_closed(c: Check):
    c.raises(lambda: clinical_landmark_features(np.zeros((10, 3)), 100, 100),
             ValueError, "too few landmarks")
    c.raises(lambda: clinical_landmark_features(_face(), 0, 100),
             ValueError, "non-positive image width")
    bad = _face(); bad[33, 0] = np.nan
    c.raises(lambda: clinical_landmark_features(bad, 100, 100),
             ValueError, "non-finite required landmark")


def test_vertical_dimensions_are_nonnegative_and_gross_outliers_fail(c: Check):
    inverted_eye = _face()
    inverted_eye[[145, 144, 153], 1] = inverted_eye[[159, 158, 160], 1] - 0.01
    inverted_mouth = _face()
    inverted_mouth[14, 1] = inverted_mouth[13, 1] - 0.01
    eye = dict(zip(CLINICAL_LANDMARK_NAMES,
                   clinical_landmark_features(inverted_eye, 1000, 1000)))
    mouth = dict(zip(CLINICAL_LANDMARK_NAMES,
                     clinical_landmark_features(inverted_mouth, 1000, 1000)))
    legacy_mouth = legacy_clinical23_v1_features(inverted_mouth, 1000, 1000)
    c.true(eye["fissure_h_mesh33"] >= 0.0,
           "fissure height is a magnitude, not a negative area source")
    c.true(mouth["mouth_open"] >= 0.0,
           "closed-lip landmark crossing remains a nonnegative magnitude")
    c.true(float(legacy_mouth[-1]) < 0.0,
           "frozen V1 retains the historical signed-gap behavior")

    gross = _face(); gross[14, 1] = -10.0
    c.raises(lambda: clinical_landmark_features(gross, 1000, 1000),
             ValueError, "gross but finite anatomical outlier fails closed")


if __name__ == "__main__":
    run_all("test_clinical_landmarks", dict(globals()))
