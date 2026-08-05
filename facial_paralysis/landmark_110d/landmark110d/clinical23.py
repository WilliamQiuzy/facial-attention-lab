"""Clinically interpretable geometry from MediaPipe Face Mesh landmarks."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


RIGHT_EYE_RING = (33, 133, 159, 145, 160, 144, 158, 153)
LEFT_EYE_RING = (263, 362, 386, 374, 387, 373, 385, 380)
RIGHT_UPPER, RIGHT_LOWER = (159, 158, 160), (145, 144, 153)
LEFT_UPPER, LEFT_LOWER = (386, 385, 387), (374, 380, 373)
RIGHT_INNER, RIGHT_OUTER = 133, 33
LEFT_INNER, LEFT_OUTER = 362, 263
RIGHT_BROW = (70, 63, 105, 66, 107)
LEFT_BROW = (300, 293, 334, 296, 336)
RIGHT_CORNER, LEFT_CORNER = 61, 291
MIDLINE = (168, 6, 197, 195, 5, 4, 1, 19, 2, 164, 0, 13, 14, 17, 152, 10)
MOUTH_TOP, MOUTH_BOTTOM = 13, 14

CLINICAL23_NAMES: tuple[str, ...] = (
    "fissure_h_mesh33",
    "fissure_h_mesh263",
    "fissure_h_absdiff",
    "fissure_h_mesh33_minus_mesh263",
    "fissure_w_mesh33",
    "fissure_w_mesh263",
    "fissure_w_absdiff",
    "eye_area_mesh33",
    "eye_area_mesh263",
    "eye_area_absdiff",
    "brow_h_mesh33",
    "brow_h_mesh263",
    "brow_h_absdiff",
    "brow_h_mesh33_minus_mesh263",
    "corner_y_mesh61",
    "corner_y_mesh291",
    "corner_y_absdiff",
    "corner_y_mesh61_minus_mesh291",
    "corner_x_mesh61",
    "corner_x_mesh291",
    "commissure_x_absdiff",
    "mouth_width",
    "mouth_open",
)

SIDE_CONVENTION = "mesh33_vs_mesh263_capture_mirror_required"

_REQUIRED_INDICES = tuple(sorted(set(
    RIGHT_EYE_RING
    + LEFT_EYE_RING
    + RIGHT_UPPER
    + RIGHT_LOWER
    + LEFT_UPPER
    + LEFT_LOWER
    + (RIGHT_INNER, RIGHT_OUTER, LEFT_INNER, LEFT_OUTER)
    + RIGHT_BROW
    + LEFT_BROW
    + (RIGHT_CORNER, LEFT_CORNER, MOUTH_TOP, MOUTH_BOTTOM)
    + MIDLINE
)))


def _as_landmark_array(landmarks: Sequence | np.ndarray) -> np.ndarray:
    if isinstance(landmarks, np.ndarray):
        array = np.asarray(landmarks, dtype=np.float32)
    else:
        try:
            array = np.asarray([
                (float(point.x), float(point.y), float(getattr(point, "z", 0.0)))
                for point in landmarks
            ], dtype=np.float32)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "landmarks must be an array or MediaPipe landmark sequence"
            ) from exc
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("landmarks must have shape (N, >=2)")
    if array.shape[0] <= max(_REQUIRED_INDICES):
        raise ValueError(
            f"landmarks contain {array.shape[0]} points; "
            f"need index {max(_REQUIRED_INDICES)}"
        )
    if array.shape[1] == 2:
        array = np.concatenate(
            (array, np.zeros((array.shape[0], 1), dtype=np.float32)), axis=1
        )
    return array[:, :3]


def _positive_dimension(value: float, name: str) -> float:
    checked = float(value)
    if not np.isfinite(checked) or checked <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return checked


def clinical23_from_mediapipe(
    landmarks: Sequence | np.ndarray,
    image_width: float,
    image_height: float,
) -> np.ndarray:
    """Convert one MediaPipe face mesh to the frozen clinical23_v2 vector.

    Side names follow mesh anchors rather than patient anatomy because mirrored
    capture provenance must be established separately.
    """
    xy = _as_landmark_array(landmarks)[:, :2]
    width = _positive_dimension(image_width, "image_width")
    height = _positive_dimension(image_height, "image_height")
    if not np.isfinite(xy[np.asarray(_REQUIRED_INDICES)]).all():
        raise ValueError("required clinical landmarks contain NaN or infinity")

    pixels = xy * np.asarray((width, height), dtype=np.float32)
    right_eye_center = pixels[np.asarray(RIGHT_EYE_RING)].mean(axis=0)
    left_eye_center = pixels[np.asarray(LEFT_EYE_RING)].mean(axis=0)
    eye_vector = left_eye_center - right_eye_center
    interocular_distance = float(np.linalg.norm(eye_vector))
    if not np.isfinite(interocular_distance) or interocular_distance <= 1e-6:
        raise ValueError("interocular distance is degenerate")

    angle = float(np.arctan2(eye_vector[1], eye_vector[0]))
    cosine, sine = np.cos(-angle), np.sin(-angle)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float32)
    center = pixels[np.asarray(MIDLINE)].mean(axis=0)
    normalized = (pixels - center) @ rotation.T / interocular_distance
    x, y = normalized[:, 0], normalized[:, 1]
    midline_x = float(x[np.asarray(MIDLINE)].mean())
    eye_line_y = 0.5 * (
        float(y[np.asarray(RIGHT_EYE_RING)].mean())
        + float(y[np.asarray(LEFT_EYE_RING)].mean())
    )

    fissure_h_right = abs(float(
        y[np.asarray(RIGHT_LOWER)].mean() - y[np.asarray(RIGHT_UPPER)].mean()
    ))
    fissure_h_left = abs(float(
        y[np.asarray(LEFT_LOWER)].mean() - y[np.asarray(LEFT_UPPER)].mean()
    ))
    fissure_w_right = abs(float(x[RIGHT_OUTER] - x[RIGHT_INNER]))
    fissure_w_left = abs(float(x[LEFT_OUTER] - x[LEFT_INNER]))
    eye_area_right = fissure_h_right * fissure_w_right
    eye_area_left = fissure_h_left * fissure_w_left
    brow_h_right = abs(float(
        y[np.asarray(RIGHT_EYE_RING)].mean()
        - y[np.asarray(RIGHT_BROW)].mean()
    ))
    brow_h_left = abs(float(
        y[np.asarray(LEFT_EYE_RING)].mean()
        - y[np.asarray(LEFT_BROW)].mean()
    ))
    corner_y_right = float(y[RIGHT_CORNER] - eye_line_y)
    corner_y_left = float(y[LEFT_CORNER] - eye_line_y)
    corner_x_right = abs(float(x[RIGHT_CORNER] - midline_x))
    corner_x_left = abs(float(x[LEFT_CORNER] - midline_x))
    mouth_width = abs(float(x[LEFT_CORNER] - x[RIGHT_CORNER]))
    mouth_open = abs(float(y[MOUTH_BOTTOM] - y[MOUTH_TOP]))

    positive_dimensions = (
        fissure_h_right,
        fissure_h_left,
        fissure_w_right,
        fissure_w_left,
        brow_h_right,
        brow_h_left,
        mouth_width,
        mouth_open,
    )
    if any(not np.isfinite(value) or value < 0 or value > 5
           for value in positive_dimensions):
        raise ValueError("implausible normalized landmark geometry")
    if fissure_w_right <= 1e-6 or fissure_w_left <= 1e-6 or mouth_width <= 1e-6:
        raise ValueError("degenerate landmark geometry")

    vector = np.asarray((
        fissure_h_right,
        fissure_h_left,
        abs(fissure_h_right - fissure_h_left),
        fissure_h_right - fissure_h_left,
        fissure_w_right,
        fissure_w_left,
        abs(fissure_w_right - fissure_w_left),
        eye_area_right,
        eye_area_left,
        abs(eye_area_right - eye_area_left),
        brow_h_right,
        brow_h_left,
        abs(brow_h_right - brow_h_left),
        brow_h_right - brow_h_left,
        corner_y_right,
        corner_y_left,
        abs(corner_y_right - corner_y_left),
        corner_y_right - corner_y_left,
        corner_x_right,
        corner_x_left,
        abs(corner_x_right - corner_x_left),
        mouth_width,
        mouth_open,
    ), dtype=np.float32)
    if vector.shape != (23,) or not np.isfinite(vector).all():
        raise ValueError("clinical landmark transform produced invalid values")
    return vector


__all__ = [
    "CLINICAL23_NAMES",
    "SIDE_CONVENTION",
    "clinical23_from_mediapipe",
]
