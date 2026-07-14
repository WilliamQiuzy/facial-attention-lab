"""OpenFace 68-point 2D topology adapter for ``semantic23_v1``.

Side A is the OpenFace 36--41 eye / 17--21 brow / landmark-48 mouth
commissure topology; side B is 42--47 / 22--26 / landmark 54.  These are
capture-side labels, not patient-left/right labels, because mirroring provenance
is not encoded in the RAVDESS tracking CSVs.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .semantic_landmarks import SEMANTIC23_FEATURE_NAMES, SEMANTIC23_SCHEMA


OPENFACE68_SIDE_A_EYE_RING = (36, 37, 38, 39, 40, 41)
OPENFACE68_SIDE_B_EYE_RING = (42, 43, 44, 45, 46, 47)
OPENFACE68_SIDE_A_UPPER = (37, 38)
OPENFACE68_SIDE_A_LOWER = (40, 41)
OPENFACE68_SIDE_B_UPPER = (43, 44)
OPENFACE68_SIDE_B_LOWER = (46, 47)
OPENFACE68_SIDE_A_INNER = 39
OPENFACE68_SIDE_A_OUTER = 36
OPENFACE68_SIDE_B_INNER = 42
OPENFACE68_SIDE_B_OUTER = 45
OPENFACE68_SIDE_A_BROW = (17, 18, 19, 20, 21)
OPENFACE68_SIDE_B_BROW = (22, 23, 24, 25, 26)
OPENFACE68_SIDE_A_CORNER = 48
OPENFACE68_SIDE_B_CORNER = 54
OPENFACE68_MOUTH_TOP = 62
OPENFACE68_MOUTH_BOTTOM = 66

# OpenFace analogues of the centre-line landmarks used by clinical23_v2.  The
# set intentionally includes central lip points to retain V2's historical
# facial-centre semantics.  A non-oral midline would require a new schema.
OPENFACE68_MIDLINE = (27, 28, 29, 30, 33, 51, 62, 66, 57, 8)

OPENFACE68_REQUIRED_INDICES = tuple(sorted(set(
    OPENFACE68_SIDE_A_EYE_RING
    + OPENFACE68_SIDE_B_EYE_RING
    + OPENFACE68_SIDE_A_UPPER
    + OPENFACE68_SIDE_A_LOWER
    + OPENFACE68_SIDE_B_UPPER
    + OPENFACE68_SIDE_B_LOWER
    + (OPENFACE68_SIDE_A_INNER, OPENFACE68_SIDE_A_OUTER,
       OPENFACE68_SIDE_B_INNER, OPENFACE68_SIDE_B_OUTER)
    + OPENFACE68_SIDE_A_BROW
    + OPENFACE68_SIDE_B_BROW
    + (OPENFACE68_SIDE_A_CORNER, OPENFACE68_SIDE_B_CORNER,
       OPENFACE68_MOUTH_TOP, OPENFACE68_MOUTH_BOTTOM)
    + OPENFACE68_MIDLINE
)))

OPENFACE68_ADAPTER_METADATA: dict[str, object] = {
    "adapter_name": "openface68_2d_to_semantic23_v1",
    "source_schema": "openface68_2d_pixels",
    "target_schema": SEMANTIC23_SCHEMA,
    "source_topology": "openface_68_2d",
    "source_side_a": "landmarks36_to_41_brow17_to_21_corner48",
    "source_side_b": "landmarks42_to_47_brow22_to_26_corner54",
    "patient_side_status": "unknown_until_capture_mirror_provenance_is_known",
    "scale_normalization": "interocular_distance",
    "roll_normalization": "eye_centres_horizontal",
    "centre_normalization": "openface_midline_mean",
    "eye_measure": "height_times_width",
    "numeric_compatibility": (
        "semantic_measure_compatible_not_raw_numeric_interchangeable_with_"
        "clinical23_v2_due_to_cross_topology_anchors"
    ),
}

_MIN_IOD = 1e-6


def _as_openface68(points: Sequence | np.ndarray) -> np.ndarray:
    try:
        array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("OpenFace landmarks must be numeric coordinates") from exc
    if array.shape != (68, 2):
        raise ValueError(
            f"OpenFace landmarks must have exact shape (68, 2), got {array.shape}"
        )
    required = array[np.asarray(OPENFACE68_REQUIRED_INDICES)]
    if not np.isfinite(required).all():
        raise ValueError("required OpenFace-68 coordinates contain NaN or infinity")
    return array


def openface68_to_semantic23(points: Sequence | np.ndarray) -> np.ndarray:
    """Compute one source-neutral 23-dimensional vector from OpenFace 2D points.

    Coordinates may be in any linear unit.  Translation, uniform scale, and
    in-plane roll are removed before measurements are taken.  Malformed
    geometry raises ``ValueError``; detector misses belong in the caller's mask.
    """
    xy = _as_openface68(points)
    side_a_eye_center = xy[np.asarray(OPENFACE68_SIDE_A_EYE_RING)].mean(axis=0)
    side_b_eye_center = xy[np.asarray(OPENFACE68_SIDE_B_EYE_RING)].mean(axis=0)
    eye_vector = side_b_eye_center - side_a_eye_center
    iod = float(np.linalg.norm(eye_vector))
    if not np.isfinite(iod) or iod <= _MIN_IOD:
        raise ValueError("OpenFace interocular distance is degenerate")

    theta = float(np.arctan2(eye_vector[1], eye_vector[0]))
    cos_t, sin_t = np.cos(-theta), np.sin(-theta)
    rotation = np.asarray(((cos_t, -sin_t), (sin_t, cos_t)), dtype=np.float64)
    center = xy[np.asarray(OPENFACE68_MIDLINE)].mean(axis=0)
    normalized = (xy - center) @ rotation.T / iod
    x = normalized[:, 0]
    y = normalized[:, 1]
    mid_x = float(x[np.asarray(OPENFACE68_MIDLINE)].mean())
    eye_line_y = 0.5 * (
        float(y[np.asarray(OPENFACE68_SIDE_A_EYE_RING)].mean())
        + float(y[np.asarray(OPENFACE68_SIDE_B_EYE_RING)].mean())
    )

    fissure_h_a = abs(float(
        y[np.asarray(OPENFACE68_SIDE_A_LOWER)].mean()
        - y[np.asarray(OPENFACE68_SIDE_A_UPPER)].mean()
    ))
    fissure_h_b = abs(float(
        y[np.asarray(OPENFACE68_SIDE_B_LOWER)].mean()
        - y[np.asarray(OPENFACE68_SIDE_B_UPPER)].mean()
    ))
    fissure_w_a = abs(float(
        x[OPENFACE68_SIDE_A_OUTER] - x[OPENFACE68_SIDE_A_INNER]
    ))
    fissure_w_b = abs(float(
        x[OPENFACE68_SIDE_B_OUTER] - x[OPENFACE68_SIDE_B_INNER]
    ))
    eye_measure_a = fissure_h_a * fissure_w_a
    eye_measure_b = fissure_h_b * fissure_w_b
    brow_h_a = abs(float(
        y[np.asarray(OPENFACE68_SIDE_A_EYE_RING)].mean()
        - y[np.asarray(OPENFACE68_SIDE_A_BROW)].mean()
    ))
    brow_h_b = abs(float(
        y[np.asarray(OPENFACE68_SIDE_B_EYE_RING)].mean()
        - y[np.asarray(OPENFACE68_SIDE_B_BROW)].mean()
    ))
    corner_y_a = float(y[OPENFACE68_SIDE_A_CORNER] - eye_line_y)
    corner_y_b = float(y[OPENFACE68_SIDE_B_CORNER] - eye_line_y)
    corner_x_a = abs(float(x[OPENFACE68_SIDE_A_CORNER] - mid_x))
    corner_x_b = abs(float(x[OPENFACE68_SIDE_B_CORNER] - mid_x))
    mouth_width = abs(float(
        x[OPENFACE68_SIDE_B_CORNER] - x[OPENFACE68_SIDE_A_CORNER]
    ))
    mouth_open = abs(float(y[OPENFACE68_MOUTH_BOTTOM] - y[OPENFACE68_MOUTH_TOP]))

    dimensions = {
        "fissure_h_side_a": fissure_h_a,
        "fissure_h_side_b": fissure_h_b,
        "fissure_w_side_a": fissure_w_a,
        "fissure_w_side_b": fissure_w_b,
        "brow_h_side_a": brow_h_a,
        "brow_h_side_b": brow_h_b,
        "mouth_width": mouth_width,
        "mouth_open": mouth_open,
    }
    if any((not np.isfinite(value)) or value < 0.0 or value > 5.0
           for value in dimensions.values()):
        raise ValueError(f"implausible normalized OpenFace geometry: {dimensions}")
    if fissure_w_a <= 1e-6 or fissure_w_b <= 1e-6 or mouth_width <= 1e-6:
        raise ValueError(f"degenerate OpenFace geometry: {dimensions}")

    vector = np.asarray((
        fissure_h_a, fissure_h_b,
        abs(fissure_h_a - fissure_h_b), fissure_h_a - fissure_h_b,
        fissure_w_a, fissure_w_b, abs(fissure_w_a - fissure_w_b),
        eye_measure_a, eye_measure_b, abs(eye_measure_a - eye_measure_b),
        brow_h_a, brow_h_b, abs(brow_h_a - brow_h_b), brow_h_a - brow_h_b,
        corner_y_a, corner_y_b, abs(corner_y_a - corner_y_b),
        corner_y_a - corner_y_b,
        corner_x_a, corner_x_b, abs(corner_x_a - corner_x_b),
        mouth_width, mouth_open,
    ), dtype=np.float32)
    if vector.shape != (len(SEMANTIC23_FEATURE_NAMES),) or not np.isfinite(vector).all():
        raise ValueError("OpenFace semantic transform produced invalid values")
    return vector


__all__ = [
    "OPENFACE68_ADAPTER_METADATA",
    "OPENFACE68_MIDLINE",
    "OPENFACE68_MOUTH_BOTTOM",
    "OPENFACE68_MOUTH_TOP",
    "OPENFACE68_REQUIRED_INDICES",
    "OPENFACE68_SIDE_A_BROW",
    "OPENFACE68_SIDE_A_CORNER",
    "OPENFACE68_SIDE_A_EYE_RING",
    "OPENFACE68_SIDE_A_LOWER",
    "OPENFACE68_SIDE_A_UPPER",
    "OPENFACE68_SIDE_B_BROW",
    "OPENFACE68_SIDE_B_CORNER",
    "OPENFACE68_SIDE_B_EYE_RING",
    "OPENFACE68_SIDE_B_LOWER",
    "OPENFACE68_SIDE_B_UPPER",
    "openface68_to_semantic23",
]
