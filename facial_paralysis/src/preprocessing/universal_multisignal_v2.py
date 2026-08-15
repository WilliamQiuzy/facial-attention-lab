"""Mirror-paired Landmark, Blendshape, and Fusion summaries for Universal v2."""
from __future__ import annotations

import numpy as np
import torch

from ..models.dynamic_landmark import horizontal_mirror_features
from .trajectory_features import trajectory_feature_set


REPRESENTATIONS = {
    "landmark_110": 110,
    "blendshape_288": 288,
    "fusion_398": 398,
}


def _immutable(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def _one_view(
    features: np.ndarray,
    valid_mask: np.ndarray,
    timestamps: np.ndarray,
    source_frame_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    blendshape = trajectory_feature_set(
        "blendshape", features, valid_mask, timestamps, source_frame_indices
    ).astype(np.float64, copy=False)
    landmark = trajectory_feature_set(
        "landmark", features, valid_mask, timestamps, source_frame_indices
    ).astype(np.float64, copy=False)
    fusion = np.concatenate((blendshape, landmark))
    result = {
        "landmark_110": _immutable(landmark),
        "blendshape_288": _immutable(blendshape),
        "fusion_398": _immutable(fusion),
    }
    if (
        {name: vector.shape[0] for name, vector in result.items()}
        != REPRESENTATIONS
        or not all(np.isfinite(vector).all() for vector in result.values())
    ):
        raise AssertionError("universal multi-signal representation drifted")
    return result


def multisignal_feature_views(
    features: np.ndarray,
    valid_mask: np.ndarray,
    timestamps: np.ndarray,
    source_frame_indices: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build exact original/mirror views from one authenticated 95D cache."""
    values = np.asarray(features)
    if values.dtype != np.dtype(np.float32):
        raise ValueError("multi-signal raw features must be exact float32")
    original = _one_view(
        values, valid_mask, timestamps, source_frame_indices
    )
    mirrored_raw = horizontal_mirror_features(
        torch.from_numpy(np.array(values, copy=True))
    ).numpy()
    remirrored = horizontal_mirror_features(
        torch.from_numpy(np.array(mirrored_raw, copy=True))
    ).numpy()
    if not np.array_equal(values, remirrored):
        raise ValueError("raw horizontal mirror must be an exact involution")
    mirrored = _one_view(
        mirrored_raw, valid_mask, timestamps, source_frame_indices
    )
    return {
        name: (original[name], mirrored[name]) for name in REPRESENTATIONS
    }


__all__ = ("REPRESENTATIONS", "multisignal_feature_views")
