"""Frozen scale-robust trajectory representations for 110D landmarks."""
from __future__ import annotations

import numpy as np

from ..datasets.dynamic_landmark import (
    DYNAMIC_FEATURE_SHAPE,
    DYNAMIC_MASK_SHAPE,
)
from .generalization_110d import LANDMARK_MI_110D, candidate_feature_vector


RAW_110D = "raw_110d"
EYE_MEDIAN3_110D = "eye_median3_110d"
ALL_LANDMARK_MEDIAN3_110D = "all_landmark_median3_110d"
CANDIDATE_ORDER = (
    RAW_110D,
    EYE_MEDIAN3_110D,
    ALL_LANDMARK_MEDIAN3_110D,
)
EYE_CHANNELS = tuple(range(72, 82))
ALL_LANDMARK_CHANNELS = tuple(range(72, 95))


def _validated_recording(
    features: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features)
    mask = np.asarray(valid_mask)
    if values.shape != DYNAMIC_FEATURE_SHAPE or values.dtype != np.dtype(np.float32):
        raise ValueError("features must be float32 with shape (4, 32, 95)")
    if mask.shape != DYNAMIC_MASK_SHAPE or mask.dtype != np.dtype(bool):
        raise ValueError("valid_mask must be bool with shape (4, 32)")
    if not mask.any() or not np.isfinite(values[mask]).all():
        raise ValueError("valid feature rows must be finite and nonempty")
    if np.any(values[~mask] != 0):
        raise ValueError("invalid feature rows must remain canonical zero")
    return values, mask


def scale_robust_recording(
    candidate: str,
    features: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Return a copy with one frozen, gap-safe temporal filter applied."""
    if not isinstance(candidate, str) or candidate not in CANDIDATE_ORDER:
        raise ValueError(f"unknown scale-robust candidate {candidate!r}")
    values, mask = _validated_recording(features, valid_mask)
    result = values.copy()
    if candidate == RAW_110D:
        return result
    channels = (
        EYE_CHANNELS
        if candidate == EYE_MEDIAN3_110D
        else ALL_LANDMARK_CHANNELS
    )
    channel_array = np.asarray(channels, dtype=np.int64)
    for window in range(DYNAMIC_FEATURE_SHAPE[0]):
        for position in range(1, DYNAMIC_FEATURE_SHAPE[1] - 1):
            if bool(mask[window, position - 1: position + 2].all()):
                result[window, position, channel_array] = np.median(
                    values[window, position - 1: position + 2][:, channel_array],
                    axis=0,
                )
    if result.dtype != np.dtype(np.float32) or not np.isfinite(result[mask]).all():
        raise AssertionError("scale-robust filtering drifted from float32 finite output")
    if np.any(result[~mask] != 0):
        raise AssertionError("scale-robust filtering changed invalid rows")
    return result


def scale_robust_feature_vector(
    candidate: str,
    features: np.ndarray,
    valid_mask: np.ndarray,
    timestamps: np.ndarray,
    source_frame_indices: np.ndarray,
) -> np.ndarray:
    """Build one 110D vector after the candidate's frozen preprocessing."""
    filtered = scale_robust_recording(candidate, features, valid_mask)
    result = candidate_feature_vector(
        LANDMARK_MI_110D,
        filtered,
        valid_mask,
        timestamps,
        source_frame_indices,
    )
    if result.shape != (110,) or result.dtype != np.dtype(np.float64):
        raise AssertionError("scale-robust feature vector drifted from 110D float64")
    if not np.isfinite(result).all():
        raise ValueError("scale-robust feature vector contains nonfinite values")
    return result


__all__ = [
    "ALL_LANDMARK_MEDIAN3_110D",
    "CANDIDATE_ORDER",
    "EYE_MEDIAN3_110D",
    "RAW_110D",
    "scale_robust_feature_vector",
    "scale_robust_recording",
]
