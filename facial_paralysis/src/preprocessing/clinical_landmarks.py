"""Clinically interpretable geometry from MediaPipe face landmarks.

The feature contract follows measurements used by Emotrics/Auto-eFACE-style
facial palsy assessment: palpebral fissure dimensions, brow height, oral
commissure position/excursion, mouth width, and mouth opening. Coordinates are
leveled using the eye centres, centred on the facial midline, and normalized by
interocular distance before any measurement is computed.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# MediaPipe FaceMesh topology.  We intentionally name the two sides by anchor
# index rather than by patient-left/patient-right: a front-camera recording may
# be mirrored before inference, and the current Mayo exports do not freeze that
# provenance.  Signed features therefore become patient-sided only after the
# capture orientation is known.
R_EYE_RING = (33, 133, 159, 145, 160, 144, 158, 153)
L_EYE_RING = (263, 362, 386, 374, 387, 373, 385, 380)
R_UP, R_LO = (159, 158, 160), (145, 144, 153)
L_UP, L_LO = (386, 385, 387), (374, 380, 373)
R_IN, R_OUT = 133, 33
L_IN, L_OUT = 362, 263
R_BROW = (70, 63, 105, 66, 107)
L_BROW = (300, 293, 334, 296, 336)
R_CORNER, L_CORNER = 61, 291
# Preserve the July 10 clinical23 numerical contract used by the historical
# web cache. A stable non-oral midline is reserved for a separately versioned
# future schema rather than silently changing these 23 measurements.
MIDLINE = (168, 6, 197, 195, 5, 4, 1, 19, 2, 164, 0, 13, 14, 17, 152, 10)
MOUTH_TOP, MOUTH_BOTTOM = 13, 14

CLINICAL_LANDMARK_NAMES: tuple[str, ...] = (
    "fissure_h_mesh33", "fissure_h_mesh263", "fissure_h_absdiff",
    "fissure_h_mesh33_minus_mesh263",
    "fissure_w_mesh33", "fissure_w_mesh263", "fissure_w_absdiff",
    "eye_area_mesh33", "eye_area_mesh263", "eye_area_absdiff",
    "brow_h_mesh33", "brow_h_mesh263", "brow_h_absdiff",
    "brow_h_mesh33_minus_mesh263",
    "corner_y_mesh61", "corner_y_mesh291", "corner_y_absdiff",
    "corner_y_mesh61_minus_mesh291",
    "corner_x_mesh61", "corner_x_mesh291", "commissure_x_absdiff",
    "mouth_width", "mouth_open",
)

LEGACY_CLINICAL_LANDMARK_NAMES: tuple[str, ...] = (
    "fissure_h_R", "fissure_h_L", "fissure_h_asym", "fissure_h_sd",
    "fissure_w_R", "fissure_w_L", "fissure_w_asym",
    "eye_area_R", "eye_area_L", "eye_area_asym",
    "brow_h_R", "brow_h_L", "brow_h_asym", "brow_h_sd",
    "corner_y_R", "corner_y_L", "corner_y_asym", "corner_y_sd",
    "corner_x_R", "corner_x_L", "commissure_asym",
    "mouth_width", "mouth_open",
)

CLINICAL_SIDE_CONVENTION = "mesh33_vs_mesh263_capture_mirror_required"

_REQUIRED_INDICES = tuple(sorted(set(
    R_EYE_RING + L_EYE_RING + R_UP + R_LO + L_UP + L_LO
    + (R_IN, R_OUT, L_IN, L_OUT) + R_BROW + L_BROW
    + (R_CORNER, L_CORNER, MOUTH_TOP, MOUTH_BOTTOM) + MIDLINE
)))
_MIN_IOD_PX = 1e-6


def landmarks_to_array(landmarks: Sequence | np.ndarray) -> np.ndarray:
    """Return an ``(N, 3)`` float32 array from MediaPipe objects or an array."""
    if isinstance(landmarks, np.ndarray):
        arr = np.asarray(landmarks, dtype=np.float32)
    else:
        try:
            arr = np.asarray([
                (float(p.x), float(p.y), float(getattr(p, "z", 0.0)))
                for p in landmarks
            ], dtype=np.float32)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("landmarks must be an array or MediaPipe landmark sequence") from exc
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"landmarks must have shape (N, >=2), got {arr.shape}")
    if arr.shape[0] <= max(_REQUIRED_INDICES):
        raise ValueError(
            f"landmarks contain {arr.shape[0]} points; need index {max(_REQUIRED_INDICES)}"
        )
    if arr.shape[1] == 2:
        arr = np.concatenate([arr, np.zeros((arr.shape[0], 1), np.float32)], axis=1)
    return arr[:, :3]


def _positive_dimension(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return value


def clinical_landmark_features(
    landmarks: Sequence | np.ndarray,
    image_width: float,
    image_height: float,
) -> np.ndarray:
    """Compute the stable 23-dimensional clinical landmark feature vector.

    Invalid required coordinates fail closed. A detector miss should be handled
    by the caller's frame mask; returning zeros here would make a malformed face
    indistinguishable from a valid, perfectly neutral measurement.
    """
    xy = landmarks_to_array(landmarks)[:, :2]
    width = _positive_dimension(image_width, "image_width")
    height = _positive_dimension(image_height, "image_height")
    required = xy[np.asarray(_REQUIRED_INDICES)]
    if not np.isfinite(required).all():
        raise ValueError("required clinical landmarks contain NaN or infinity")

    pixels = xy * np.asarray((width, height), dtype=np.float32)
    right_eye_center = pixels[np.asarray(R_EYE_RING)].mean(axis=0)
    left_eye_center = pixels[np.asarray(L_EYE_RING)].mean(axis=0)
    eye_vector = left_eye_center - right_eye_center
    iod = float(np.linalg.norm(eye_vector))
    if not np.isfinite(iod) or iod <= _MIN_IOD_PX:
        raise ValueError("interocular distance is degenerate")

    theta = float(np.arctan2(eye_vector[1], eye_vector[0]))
    cos_t, sin_t = np.cos(-theta), np.sin(-theta)
    rotation = np.asarray(((cos_t, -sin_t), (sin_t, cos_t)), dtype=np.float32)
    center = pixels[np.asarray(MIDLINE)].mean(axis=0)
    normalized = (pixels - center) @ rotation.T / iod
    x, y = normalized[:, 0], normalized[:, 1]
    mid_x = float(x[np.asarray(MIDLINE)].mean())
    eye_line_y = 0.5 * (
        float(y[np.asarray(R_EYE_RING)].mean())
        + float(y[np.asarray(L_EYE_RING)].mean())
    )

    fissure_h_right = abs(float(
        y[np.asarray(R_LO)].mean() - y[np.asarray(R_UP)].mean()))
    fissure_h_left = abs(float(
        y[np.asarray(L_LO)].mean() - y[np.asarray(L_UP)].mean()))
    fissure_w_right = abs(float(x[R_OUT] - x[R_IN]))
    fissure_w_left = abs(float(x[L_OUT] - x[L_IN]))
    eye_area_right = fissure_h_right * fissure_w_right
    eye_area_left = fissure_h_left * fissure_w_left
    brow_h_right = abs(float(
        y[np.asarray(R_EYE_RING)].mean() - y[np.asarray(R_BROW)].mean()
    ))
    brow_h_left = abs(float(
        y[np.asarray(L_EYE_RING)].mean() - y[np.asarray(L_BROW)].mean()
    ))
    corner_y_right = float(y[R_CORNER] - eye_line_y)
    corner_y_left = float(y[L_CORNER] - eye_line_y)
    corner_x_right = abs(float(x[R_CORNER] - mid_x))
    corner_x_left = abs(float(x[L_CORNER] - mid_x))
    mouth_width = abs(float(x[L_CORNER] - x[R_CORNER]))
    mouth_open = abs(float(y[MOUTH_BOTTOM] - y[MOUTH_TOP]))

    dimensions = {
        "fissure_h_mesh33": fissure_h_right,
        "fissure_h_mesh263": fissure_h_left,
        "fissure_w_mesh33": fissure_w_right,
        "fissure_w_mesh263": fissure_w_left,
        "brow_h_mesh33": brow_h_right,
        "brow_h_mesh263": brow_h_left,
        "mouth_width": mouth_width,
        "mouth_open": mouth_open,
    }
    if any((not np.isfinite(value)) or value < 0.0 or value > 5.0
           for value in dimensions.values()):
        raise ValueError(f"implausible normalized landmark geometry: {dimensions}")
    if fissure_w_right <= 1e-6 or fissure_w_left <= 1e-6 or mouth_width <= 1e-6:
        raise ValueError(f"degenerate landmark geometry: {dimensions}")

    vector = np.asarray((
        fissure_h_right, fissure_h_left,
        abs(fissure_h_right - fissure_h_left), fissure_h_right - fissure_h_left,
        fissure_w_right, fissure_w_left, abs(fissure_w_right - fissure_w_left),
        eye_area_right, eye_area_left, abs(eye_area_right - eye_area_left),
        brow_h_right, brow_h_left,
        abs(brow_h_right - brow_h_left), brow_h_right - brow_h_left,
        corner_y_right, corner_y_left,
        abs(corner_y_right - corner_y_left), corner_y_right - corner_y_left,
        corner_x_right, corner_x_left, abs(corner_x_right - corner_x_left),
        mouth_width,
        mouth_open,
    ), dtype=np.float32)
    if vector.shape != (len(CLINICAL_LANDMARK_NAMES),) or not np.isfinite(vector).all():
        raise ValueError("clinical landmark transform produced invalid values")
    return vector


def legacy_clinical23_v1_features(
    landmarks: Sequence | np.ndarray,
    image_width: float,
    image_height: float,
) -> np.ndarray:
    """Frozen July-10 transform used by the historical static-web cache.

    V1 retained signed vertical gaps and the historical ``iod + 1e-6`` scale.
    Keep it solely for exact experiment reproduction; new extraction uses the
    nonnegative clinical-distance V2 transform above.
    """
    # Keep the NumPy scalar operation order byte-for-byte aligned with the
    # original July-10 script. Seemingly harmless ``float(...)`` casts move a
    # few float32 results by one ULP and would make the pinned static cache no
    # longer exactly reproducible.
    xy = landmarks_to_array(landmarks)[:, :2]
    width = _positive_dimension(image_width, "image_width")
    height = _positive_dimension(image_height, "image_height")
    pixels = xy * np.array([width, height], np.float32)
    right_eye_center = pixels[list(R_EYE_RING)].mean(0)
    left_eye_center = pixels[list(L_EYE_RING)].mean(0)
    iod = np.linalg.norm(left_eye_center - right_eye_center) + 1e-6
    theta = np.arctan2(
        left_eye_center[1] - right_eye_center[1],
        left_eye_center[0] - right_eye_center[0],
    )
    cos_t, sin_t = np.cos(-theta), np.sin(-theta)
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]], np.float32)
    center = pixels[list(MIDLINE)].mean(0)
    normalized = (pixels - center) @ rotation.T / iod
    x, y = normalized[:, 0], normalized[:, 1]
    mid_x = normalized[list(MIDLINE), 0].mean()
    eye_line_y = 0.5 * (
        normalized[list(R_EYE_RING), 1].mean()
        + normalized[list(L_EYE_RING), 1].mean()
    )

    fh_r = y[list(R_LO)].mean() - y[list(R_UP)].mean()
    fh_l = y[list(L_LO)].mean() - y[list(L_UP)].mean()
    fw_r = abs(x[R_OUT] - x[R_IN])
    fw_l = abs(x[L_OUT] - x[L_IN])
    area_r, area_l = fh_r * fw_r, fh_l * fw_l
    brow_r = normalized[list(R_EYE_RING), 1].mean() - y[list(R_BROW)].mean()
    brow_l = normalized[list(L_EYE_RING), 1].mean() - y[list(L_BROW)].mean()
    corner_y_r = y[R_CORNER] - eye_line_y
    corner_y_l = y[L_CORNER] - eye_line_y
    corner_x_r = abs(x[R_CORNER] - mid_x)
    corner_x_l = abs(x[L_CORNER] - mid_x)
    vector = np.asarray((
        fh_r, fh_l, abs(fh_r - fh_l), fh_r - fh_l,
        fw_r, fw_l, abs(fw_r - fw_l),
        area_r, area_l, abs(area_r - area_l),
        brow_r, brow_l, abs(brow_r - brow_l), brow_r - brow_l,
        corner_y_r, corner_y_l, abs(corner_y_r - corner_y_l),
        corner_y_r - corner_y_l,
        corner_x_r, corner_x_l, abs(corner_x_r - corner_x_l),
        abs(x[L_CORNER] - x[R_CORNER]),
        y[MOUTH_BOTTOM] - y[MOUTH_TOP],
    ), dtype=np.float32)
    vector[~np.isfinite(vector)] = 0.0
    return vector


__all__ = [
    "CLINICAL_LANDMARK_NAMES",
    "CLINICAL_SIDE_CONVENTION",
    "LEGACY_CLINICAL_LANDMARK_NAMES",
    "clinical_landmark_features",
    "legacy_clinical23_v1_features",
    "landmarks_to_array",
]
