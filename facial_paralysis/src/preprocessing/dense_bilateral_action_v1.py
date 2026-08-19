"""Dense, action-conditioned MediaPipe mesh summaries for Router v6.

The representation is intentionally estimator-independent.  It aligns every
478-point mesh in pixel space, compares each prompted action with its external
baseline, and keeps original and truly mirrored video streams separate.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


DENSE_POINT_COUNT = 478
DENSE_COORD_NAMES = ("x", "y", "z")
DENSE_STAT_NAMES = (
    "action_median",
    "response_median",
    "response_q10",
    "response_q90",
    "response_range",
    "response_max_abs_adjacent_step",
)
BILATERAL_INTERACTION_STAT_NAMES = (
    "action_geometry_asymmetry",
    "response_asymmetry",
    "response_low",
    "response_high",
    "response_ratio",
    "range_asymmetry",
    "range_low",
    "range_high",
    "range_ratio",
    "peak_asymmetry",
    "peak_low",
    "peak_high",
    "peak_ratio",
    "paired_difference_median",
    "paired_difference_q90",
)
_LEFT_EYE_ANCHOR = 33
_RIGHT_EYE_ANCHOR = 263
_MIN_ACTION_SUPPORT = 6
_MIN_BASELINE_SUPPORT = 4


def _immutable_float64(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(
        contiguous.shape
    )


def _positive_dimension(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return numeric


def normalize_dense_landmarks(
    landmarks: np.ndarray,
    image_width: object,
    image_height: object,
) -> np.ndarray:
    """Return translation, roll, and inter-eye-scale normalized 478x3 mesh."""
    if (
        type(landmarks) is not np.ndarray
        or landmarks.dtype != np.dtype(np.float64)
        or landmarks.shape != (DENSE_POINT_COUNT, 3)
    ):
        raise ValueError("landmarks must be an exact float64 array with shape (478, 3)")
    if not np.isfinite(landmarks).all():
        raise ValueError("landmarks must be finite")
    width = _positive_dimension(image_width, "image_width")
    height = _positive_dimension(image_height, "image_height")

    pixels = landmarks.copy()
    pixels[:, 0] *= width
    pixels[:, 1] *= height
    pixels[:, 2] *= width
    eye_a = pixels[_LEFT_EYE_ANCHOR]
    eye_b = pixels[_RIGHT_EYE_ANCHOR]
    eye_vector = eye_b[:2] - eye_a[:2]
    eye_scale = float(np.linalg.norm(eye_vector))
    if not np.isfinite(eye_scale) or eye_scale <= np.finfo(np.float64).eps:
        raise ValueError("outer-eye anchors must define a nondegenerate scale")
    center = 0.5 * (eye_a + eye_b)
    centered = pixels - center
    angle = float(np.arctan2(eye_vector[1], eye_vector[0]))
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    rotate_minus_angle = np.asarray(
        ((cosine, -sine), (sine, cosine)), dtype=np.float64
    )
    centered[:, :2] = centered[:, :2] @ rotate_minus_angle
    normalized = centered / eye_scale
    if not np.isfinite(normalized).all():
        raise ValueError("dense normalization produced nonfinite values")
    return _immutable_float64(normalized)


def _validated_action_names(action_names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(action_names, (str, bytes)):
        raise ValueError("action_names must be a nonempty sequence")
    try:
        names = tuple(action_names)
    except TypeError as exc:
        raise ValueError("action_names must be a nonempty sequence") from exc
    if (
        not names
        or any(type(name) is not str or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("action_names must contain unique nonempty strings")
    return names


def dense_action_feature_names(action_names: Sequence[str]) -> tuple[str, ...]:
    names = _validated_action_names(action_names)
    return tuple(
        f"{action}__mesh{point}__{coordinate}__{statistic}"
        for action in names
        for point in range(DENSE_POINT_COUNT)
        for coordinate in DENSE_COORD_NAMES
        for statistic in DENSE_STAT_NAMES
    )


def bilateral_interaction_feature_names(
    action_names: Sequence[str],
) -> tuple[str, ...]:
    names = _validated_action_names(action_names)
    return tuple(
        f"{action}__mesh{point}__{coordinate}__{statistic}"
        for action in names
        for point in range(DENSE_POINT_COUNT)
        for coordinate in DENSE_COORD_NAMES
        for statistic in BILATERAL_INTERACTION_STAT_NAMES
    )


def _validated_stream(
    actions: np.ndarray,
    action_valid: np.ndarray,
    baselines: np.ndarray,
    baseline_valid: np.ndarray,
    action_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if (
        type(actions) is not np.ndarray
        or actions.dtype != np.dtype(np.float64)
        or actions.ndim != 4
        or actions.shape[0] != action_count
        or actions.shape[2:] != (DENSE_POINT_COUNT, 3)
    ):
        raise ValueError("actions must be exact float64 (actions, frames, 478, 3)")
    if (
        type(baselines) is not np.ndarray
        or baselines.dtype != np.dtype(np.float64)
        or baselines.ndim != 4
        or baselines.shape[0] != action_count
        or baselines.shape[2:] != (DENSE_POINT_COUNT, 3)
    ):
        raise ValueError(
            "baselines must be exact float64 (actions, frames, 478, 3)"
        )
    if (
        type(action_valid) is not np.ndarray
        or action_valid.dtype != np.dtype(bool)
        or action_valid.shape != actions.shape[:2]
    ):
        raise ValueError("action_valid must be an exact bool action-frame mask")
    if (
        type(baseline_valid) is not np.ndarray
        or baseline_valid.dtype != np.dtype(bool)
        or baseline_valid.shape != baselines.shape[:2]
    ):
        raise ValueError("baseline_valid must be an exact bool baseline-frame mask")
    if np.any(action_valid.sum(axis=1) < _MIN_ACTION_SUPPORT):
        raise ValueError("every action requires at least six valid frames")
    if np.any(baseline_valid.sum(axis=1) < _MIN_BASELINE_SUPPORT):
        raise ValueError("every baseline requires at least four valid frames")
    if not np.isfinite(actions[action_valid]).all():
        raise ValueError("valid action meshes must be finite")
    if not np.isfinite(baselines[baseline_valid]).all():
        raise ValueError("valid baseline meshes must be finite")
    return actions, action_valid, baselines, baseline_valid


def _summarize_stream(
    actions: np.ndarray,
    action_valid: np.ndarray,
    baselines: np.ndarray,
    baseline_valid: np.ndarray,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for action_index in range(actions.shape[0]):
        observed = actions[action_index, action_valid[action_index]]
        rest = baselines[action_index, baseline_valid[action_index]]
        rest_median = np.median(rest, axis=0)
        response = observed - rest_median
        quantiles = np.quantile(response, (0.10, 0.90), axis=0)

        full_response = actions[action_index] - rest_median
        adjacent_valid = (
            action_valid[action_index, :-1] & action_valid[action_index, 1:]
        )
        if adjacent_valid.any():
            adjacent_steps = np.abs(np.diff(full_response, axis=0))[adjacent_valid]
            max_step = np.max(adjacent_steps, axis=0)
        else:
            max_step = np.zeros((DENSE_POINT_COUNT, 3), dtype=np.float64)
        statistics = np.stack(
            (
                np.median(observed, axis=0),
                np.median(response, axis=0),
                quantiles[0],
                quantiles[1],
                np.max(response, axis=0) - np.min(response, axis=0),
                max_step,
            ),
            axis=-1,
        )
        rows.append(statistics.reshape(-1))
    vector = np.concatenate(rows)
    if not np.isfinite(vector).all():
        raise ValueError("dense action summary produced nonfinite values")
    return _immutable_float64(vector)


def dense_action_feature_views(
    original_actions: np.ndarray,
    original_action_valid: np.ndarray,
    original_baselines: np.ndarray,
    original_baseline_valid: np.ndarray,
    mirrored_actions: np.ndarray,
    mirrored_action_valid: np.ndarray,
    mirrored_baselines: np.ndarray,
    mirrored_baseline_valid: np.ndarray,
    *,
    action_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Summarize independently extracted original and horizontally flipped streams."""
    names = _validated_action_names(action_names)
    original = _validated_stream(
        original_actions,
        original_action_valid,
        original_baselines,
        original_baseline_valid,
        len(names),
    )
    mirrored = _validated_stream(
        mirrored_actions,
        mirrored_action_valid,
        mirrored_baselines,
        mirrored_baseline_valid,
        len(names),
    )
    return _summarize_stream(*original), _summarize_stream(*mirrored)


def _safe_ratio(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.divide(
        low,
        high,
        out=np.ones_like(low),
        where=high > np.finfo(np.float64).eps,
    )


def bilateral_interaction_feature_vector(
    original_actions: np.ndarray,
    original_action_valid: np.ndarray,
    original_baselines: np.ndarray,
    original_baseline_valid: np.ndarray,
    mirrored_actions: np.ndarray,
    mirrored_action_valid: np.ndarray,
    mirrored_baselines: np.ndarray,
    mirrored_baseline_valid: np.ndarray,
    *,
    action_names: Sequence[str],
) -> np.ndarray:
    """Return swap-invariant, frame-paired bilateral capacity statistics."""
    names = _validated_action_names(action_names)
    _validated_stream(
        original_actions,
        original_action_valid,
        original_baselines,
        original_baseline_valid,
        len(names),
    )
    _validated_stream(
        mirrored_actions,
        mirrored_action_valid,
        mirrored_baselines,
        mirrored_baseline_valid,
        len(names),
    )
    if not np.array_equal(original_action_valid, mirrored_action_valid) or not np.array_equal(
        original_baseline_valid, mirrored_baseline_valid
    ):
        raise ValueError("bilateral interaction requires exactly paired view support")
    outputs = []
    for action in range(len(names)):
        action_mask = original_action_valid[action]
        baseline_mask = original_baseline_valid[action]
        observed_o = original_actions[action, action_mask]
        observed_m = mirrored_actions[action, action_mask]
        rest_o = np.median(original_baselines[action, baseline_mask], axis=0)
        rest_m = np.median(mirrored_baselines[action, baseline_mask], axis=0)
        response_o = observed_o - rest_o
        response_m = observed_m - rest_m
        median_o = np.median(response_o, axis=0)
        median_m = np.median(response_m, axis=0)
        absolute_o = np.abs(median_o)
        absolute_m = np.abs(median_m)
        response_low = np.minimum(absolute_o, absolute_m)
        response_high = np.maximum(absolute_o, absolute_m)
        range_o = np.ptp(response_o, axis=0)
        range_m = np.ptp(response_m, axis=0)
        range_low = np.minimum(range_o, range_m)
        range_high = np.maximum(range_o, range_m)
        peak_o = np.max(np.abs(response_o), axis=0)
        peak_m = np.max(np.abs(response_m), axis=0)
        peak_low = np.minimum(peak_o, peak_m)
        peak_high = np.maximum(peak_o, peak_m)
        paired = np.abs(response_o - response_m)
        statistics = np.stack(
            (
                np.abs(
                    np.median(observed_o, axis=0)
                    - np.median(observed_m, axis=0)
                ),
                np.abs(median_o - median_m),
                response_low,
                response_high,
                _safe_ratio(response_low, response_high),
                np.abs(range_o - range_m),
                range_low,
                range_high,
                _safe_ratio(range_low, range_high),
                np.abs(peak_o - peak_m),
                peak_low,
                peak_high,
                _safe_ratio(peak_low, peak_high),
                np.median(paired, axis=0),
                np.quantile(paired, 0.9, axis=0),
            ),
            axis=-1,
        )
        outputs.append(statistics.reshape(-1))
    vector = np.concatenate(outputs)
    if not np.isfinite(vector).all():
        raise ValueError("bilateral interaction summary produced nonfinite values")
    return _immutable_float64(vector)


__all__ = (
    "BILATERAL_INTERACTION_STAT_NAMES",
    "DENSE_POINT_COUNT",
    "DENSE_STAT_NAMES",
    "bilateral_interaction_feature_names",
    "bilateral_interaction_feature_vector",
    "dense_action_feature_names",
    "dense_action_feature_views",
    "normalize_dense_landmarks",
)
