"""Closed feature views for Source-Robust Landmark 110D v1."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .trajectory_features import SUMMARY_STAT_NAMES, trajectory_feature_names


CANDIDATE_ORDER = (
    "landmark_mi_110d",
    "within_video_dynamics_87d",
    "asymmetry_dynamics_93d",
)
CANDIDATE_DIMENSIONS: Mapping[str, int] = {
    "landmark_mi_110d": 110,
    "within_video_dynamics_87d": 87,
    "asymmetry_dynamics_93d": 93,
}
ASYMMETRY_MEDIAN_CHANNEL_NAMES = (
    "fissure_h_absdiff",
    "fissure_w_absdiff",
    "eye_area_absdiff",
    "brow_h_absdiff",
    "corner_y_absdiff",
    "commissure_x_absdiff",
)

_BASE_NAMES = trajectory_feature_names("landmark")
_MEDIAN_SUFFIX = f"__{SUMMARY_STAT_NAMES[0]}"
_DYNAMIC_NAMES = tuple(
    name for name in _BASE_NAMES if not name.endswith(_MEDIAN_SUFFIX)
)
_ASYMMETRY_MEDIAN_NAMES = tuple(
    f"{channel}{_MEDIAN_SUFFIX}" for channel in ASYMMETRY_MEDIAN_CHANNEL_NAMES
)
_CANDIDATE_NAMES = {
    CANDIDATE_ORDER[0]: _BASE_NAMES,
    CANDIDATE_ORDER[1]: _DYNAMIC_NAMES,
    CANDIDATE_ORDER[2]: _ASYMMETRY_MEDIAN_NAMES + _DYNAMIC_NAMES,
}
_BASE_INDEX = {name: index for index, name in enumerate(_BASE_NAMES)}
_CANDIDATE_INDICES = {
    candidate: np.asarray([_BASE_INDEX[name] for name in names], dtype=np.int64)
    for candidate, names in _CANDIDATE_NAMES.items()
}

if tuple(_CANDIDATE_NAMES) != CANDIDATE_ORDER or any(
    len(_CANDIDATE_NAMES[candidate]) != CANDIDATE_DIMENSIONS[candidate]
    or len(set(_CANDIDATE_NAMES[candidate])) != CANDIDATE_DIMENSIONS[candidate]
    for candidate in CANDIDATE_ORDER
):
    raise RuntimeError("source-robust feature registry drifted")


def candidate_feature_names(candidate: str) -> tuple[str, ...]:
    """Return the exact ordered feature names for one locked candidate."""
    try:
        return _CANDIDATE_NAMES[candidate]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unknown source-robust candidate {candidate!r}") from exc


def source_robust_feature_views(
    landmark_110d: np.ndarray,
) -> dict[str, np.ndarray]:
    """Slice one vector or batch into the three preregistered feature views."""
    values = np.asarray(landmark_110d)
    if values.ndim not in {1, 2} or values.shape[-1] != 110:
        raise ValueError("landmark_110d must have shape (110,) or (N, 110)")
    if values.dtype.kind not in {"f", "i", "u"} or not np.isfinite(values).all():
        raise ValueError("landmark_110d must contain finite real numeric values")
    return {
        candidate: np.take(values, indices, axis=-1).copy()
        for candidate, indices in _CANDIDATE_INDICES.items()
    }


__all__ = [
    "ASYMMETRY_MEDIAN_CHANNEL_NAMES",
    "CANDIDATE_DIMENSIONS",
    "CANDIDATE_ORDER",
    "candidate_feature_names",
    "source_robust_feature_views",
]
