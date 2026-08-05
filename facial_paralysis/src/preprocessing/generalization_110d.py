"""Frozen geometry representations for 110D-Generalization v1.

The 110D and 58D blocks delegate to their existing extractors.  The final
36D block preserves the four deterministic recording windows and summarizes
only direction-free bilateral excursion and velocity within each window.
These slots are recording-position proxies, not clinically labelled phases.
"""
from __future__ import annotations

import numpy as np

from .clinical_dynamics import (
    CLINICAL_DYNAMICS_DIM,
    clinical_dynamics_feature_names,
    clinical_dynamics_feature_vector,
)
from .trajectory_features import (
    LANDMARK_BILATERAL_PAIRS,
    LANDMARK_DIM,
    SUMMARY_STAT_NAMES,
    summarize_trajectory_channels,
    trajectory_feature_names,
    trajectory_feature_set,
)


LANDMARK_MI_110D = "landmark_mi_110d"
ACTION_PROXY_168D = "landmark_mi_110d_action_proxy_168d"
ACTION_PHASE_PROXY_204D = "landmark_mi_110d_action_phase_proxy_204d"

# Dict insertion order is part of the preregistered comparison contract.
CANDIDATE_REGISTRY = {
    LANDMARK_MI_110D: 110,
    ACTION_PROXY_168D: 168,
    ACTION_PHASE_PROXY_204D: 204,
}
CANDIDATE_ORDER = tuple(CANDIDATE_REGISTRY)

PHASE_PROXY_REGION_PAIRS = (
    ("eye", LANDMARK_BILATERAL_PAIRS[:3]),
    ("brow", LANDMARK_BILATERAL_PAIRS[3:4]),
    ("mouth", LANDMARK_BILATERAL_PAIRS[4:]),
)
PHASE_PROXY_STAT_NAMES = (
    "mean_bilateral_excursion",
    "absolute_excursion_asymmetry",
    "mean_bilateral_peak_velocity_per_second",
)
PHASE_PROXY_DIM = (
    4 * len(PHASE_PROXY_REGION_PAIRS) * len(PHASE_PROXY_STAT_NAMES)
)


def phase_proxy_feature_names() -> tuple[str, ...]:
    """Return the exact window-major names for the 36D Phase proxy."""
    names = tuple(
        f"phase_proxy__window_{window}__{region}__{statistic}"
        for window in range(4)
        for region, _ in PHASE_PROXY_REGION_PAIRS
        for statistic in PHASE_PROXY_STAT_NAMES
    )
    if len(names) != PHASE_PROXY_DIM or len(set(names)) != PHASE_PROXY_DIM:
        raise AssertionError("phase-proxy feature-name contract drifted")
    return names


def _action_proxy_feature_names() -> tuple[str, ...]:
    return tuple(
        f"action_proxy__{name}" for name in clinical_dynamics_feature_names()
    )


def candidate_feature_names(candidate: str) -> tuple[str, ...]:
    """Return the exact ordered names for one preregistered candidate."""
    if not isinstance(candidate, str) or candidate not in CANDIDATE_REGISTRY:
        raise ValueError(f"unknown 110D-Generalization candidate {candidate!r}")
    landmark = trajectory_feature_names("landmark")
    if candidate == LANDMARK_MI_110D:
        names = landmark
    else:
        names = landmark + _action_proxy_feature_names()
        if candidate == ACTION_PHASE_PROXY_204D:
            names += phase_proxy_feature_names()
    expected = CANDIDATE_REGISTRY[candidate]
    if len(names) != expected or len(set(names)) != expected:
        raise AssertionError("candidate feature-name contract drifted")
    return names


def phase_proxy_feature_vector(
    features: np.ndarray,
    valid_mask: np.ndarray,
    timestamps: np.ndarray,
    source_frame_indices: np.ndarray,
) -> np.ndarray:
    """Build the fixed 36D window/region geometry proxy.

    Every temporal slot is summarized independently by the existing validated
    trajectory primitive.  A window with no valid frame fails closed rather
    than being imputed, and derivatives cannot cross a window or detector gap.
    """
    mask = np.asarray(valid_mask)
    if mask.shape != (4, 32) or mask.dtype != np.dtype(bool):
        raise ValueError("valid_mask must be bool with shape (4, 32)")

    pair_channels = tuple(
        channel
        for _, first, second in LANDMARK_BILATERAL_PAIRS
        for channel in (first, second)
    )
    range_column = SUMMARY_STAT_NAMES.index("range")
    velocity_column = SUMMARY_STAT_NAMES.index(
        "max_abs_velocity_per_second"
    )
    output: list[float] = []
    for window in range(4):
        window_mask = np.zeros_like(mask)
        window_mask[window] = mask[window]
        summaries = summarize_trajectory_channels(
            features,
            window_mask,
            timestamps,
            source_frame_indices,
            pair_channels,
        ).reshape(len(pair_channels), len(SUMMARY_STAT_NAMES))
        by_channel = {
            channel: summaries[position]
            for position, channel in enumerate(pair_channels)
        }
        for _, pairs in PHASE_PROXY_REGION_PAIRS:
            bilateral_excursions: list[float] = []
            excursion_asymmetries: list[float] = []
            bilateral_velocities: list[float] = []
            for _, first, second in pairs:
                first_range = float(by_channel[first][range_column])
                second_range = float(by_channel[second][range_column])
                first_velocity = float(by_channel[first][velocity_column])
                second_velocity = float(by_channel[second][velocity_column])
                bilateral_excursions.append(
                    0.5 * (first_range + second_range)
                )
                excursion_asymmetries.append(abs(first_range - second_range))
                bilateral_velocities.append(
                    0.5 * (first_velocity + second_velocity)
                )
            output.extend((
                float(np.mean(bilateral_excursions)),
                float(np.mean(excursion_asymmetries)),
                float(np.mean(bilateral_velocities)),
            ))

    result = np.asarray(output, dtype=np.float64)
    if result.shape != (PHASE_PROXY_DIM,):
        raise AssertionError("phase-proxy feature dimension drifted")
    if not np.isfinite(result).all():
        raise ValueError("phase proxy produced nonfinite values")
    return result


def candidate_feature_vector(
    candidate: str,
    features: np.ndarray,
    valid_mask: np.ndarray,
    timestamps: np.ndarray,
    source_frame_indices: np.ndarray,
) -> np.ndarray:
    """Build one frozen 110D-Generalization v1 representation."""
    if not isinstance(candidate, str) or candidate not in CANDIDATE_REGISTRY:
        raise ValueError(f"unknown 110D-Generalization candidate {candidate!r}")

    landmark = trajectory_feature_set(
        "landmark",
        features,
        valid_mask,
        timestamps,
        source_frame_indices,
    )
    if candidate == LANDMARK_MI_110D:
        result = landmark
    else:
        action = clinical_dynamics_feature_vector(
            features,
            valid_mask,
            timestamps,
            source_frame_indices,
        )
        result = np.concatenate((landmark, action))
        if candidate == ACTION_PHASE_PROXY_204D:
            phase = phase_proxy_feature_vector(
                features,
                valid_mask,
                timestamps,
                source_frame_indices,
            )
            result = np.concatenate((result, phase))

    expected = CANDIDATE_REGISTRY[candidate]
    if result.shape != (expected,):
        raise AssertionError("candidate feature dimension drifted")
    if result.dtype != np.dtype(np.float64) or not np.isfinite(result).all():
        raise ValueError("candidate feature vector must be finite float64")
    return result


if LANDMARK_DIM != 110 or CLINICAL_DYNAMICS_DIM != 58 or PHASE_PROXY_DIM != 36:
    raise AssertionError("110D-Generalization component dimensions drifted")


__all__ = [
    "ACTION_PHASE_PROXY_204D",
    "ACTION_PROXY_168D",
    "CANDIDATE_ORDER",
    "CANDIDATE_REGISTRY",
    "LANDMARK_MI_110D",
    "PHASE_PROXY_DIM",
    "PHASE_PROXY_REGION_PAIRS",
    "PHASE_PROXY_STAT_NAMES",
    "candidate_feature_names",
    "candidate_feature_vector",
    "phase_proxy_feature_names",
    "phase_proxy_feature_vector",
]
