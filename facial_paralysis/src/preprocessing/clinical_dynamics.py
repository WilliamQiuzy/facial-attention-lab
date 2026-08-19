"""Compact, direction-free clinical dynamics from the frozen landmark schema.

The feature contract is deliberately invariant to swapping each capture-side
MediaPipe landmark pair. Patient laterality is not inferred because capture
mirroring provenance is unknown in the current PalsyNet cache.
"""
from __future__ import annotations

import numpy as np

from ..datasets.dynamic_landmark import DYNAMIC_FEATURE_NAMES
from .trajectory_features import (
    LANDMARK_BILATERAL_PAIRS,
    SUMMARY_STAT_NAMES,
    bilateral_dynamics,
    summarize_trajectory_channels,
)


CLINICAL_DYNAMICS_PAIR_STAT_NAMES = (
    "correlation",
    "amplitude_ratio",
    "absolute_lag_seconds",
    "mean_range",
    "absolute_range_difference",
    "mean_max_abs_velocity_per_second",
    "absolute_max_abs_velocity_difference_per_second",
)

# These are explicit absolute-difference channels plus two global mouth-shape
# channels in clinical23_v2. They remain meaningful when capture sides swap.
CLINICAL_DYNAMICS_GLOBAL_CHANNELS = (
    ("eye", 74),
    ("eye", 78),
    ("eye", 81),
    ("brow", 84),
    ("mouth", 88),
    ("mouth", 92),
    ("mouth", 93),
    ("mouth", 94),
)
CLINICAL_DYNAMICS_GLOBAL_STAT_NAMES = (
    "range",
    "max_abs_velocity_per_second",
)
CLINICAL_DYNAMICS_DIM = (
    len(LANDMARK_BILATERAL_PAIRS) * len(CLINICAL_DYNAMICS_PAIR_STAT_NAMES)
    + len(CLINICAL_DYNAMICS_GLOBAL_CHANNELS)
    * len(CLINICAL_DYNAMICS_GLOBAL_STAT_NAMES)
)


def clinical_dynamics_feature_names() -> tuple[str, ...]:
    """Return the exact ordered names for the frozen 58-feature contract."""
    pair_names = tuple(
        f"{pair_name}__{statistic}"
        for pair_name, _, _ in LANDMARK_BILATERAL_PAIRS
        for statistic in CLINICAL_DYNAMICS_PAIR_STAT_NAMES
    )
    global_names = tuple(
        f"{DYNAMIC_FEATURE_NAMES[index]}__{statistic}"
        for _, index in CLINICAL_DYNAMICS_GLOBAL_CHANNELS
        for statistic in CLINICAL_DYNAMICS_GLOBAL_STAT_NAMES
    )
    names = pair_names + global_names
    if len(names) != CLINICAL_DYNAMICS_DIM or len(set(names)) != len(names):
        raise AssertionError("clinical dynamics feature-name contract drifted")
    return names


def clinical_dynamics_feature_vector(
    features: np.ndarray,
    valid_mask: np.ndarray,
    timestamps: np.ndarray,
    source_frame_indices: np.ndarray,
) -> np.ndarray:
    """Build one direction-free clinical-dynamics vector.

    Validation, velocity construction, detector-gap handling, and window
    isolation are delegated to the frozen trajectory preprocessing primitives.
    """
    pair_channels = tuple(
        channel
        for _, first, second in LANDMARK_BILATERAL_PAIRS
        for channel in (first, second)
    )
    global_channels = tuple(
        index for _, index in CLINICAL_DYNAMICS_GLOBAL_CHANNELS
    )
    channels = pair_channels + global_channels
    summaries = summarize_trajectory_channels(
        features,
        valid_mask,
        timestamps,
        source_frame_indices,
        channels,
    ).reshape(len(channels), len(SUMMARY_STAT_NAMES))
    range_column = SUMMARY_STAT_NAMES.index("range")
    velocity_column = SUMMARY_STAT_NAMES.index(
        "max_abs_velocity_per_second"
    )
    by_channel = {
        channel: summaries[position]
        for position, channel in enumerate(channels)
    }

    output: list[float] = []
    for _, first, second in LANDMARK_BILATERAL_PAIRS:
        correlation, amplitude_ratio, lag_seconds = bilateral_dynamics(
            np.asarray(features)[..., first],
            np.asarray(features)[..., second],
            valid_mask,
            timestamps,
        )
        first_range = float(by_channel[first][range_column])
        second_range = float(by_channel[second][range_column])
        first_velocity = float(by_channel[first][velocity_column])
        second_velocity = float(by_channel[second][velocity_column])
        output.extend((
            float(correlation),
            float(amplitude_ratio),
            abs(float(lag_seconds)),
            0.5 * (first_range + second_range),
            abs(first_range - second_range),
            0.5 * (first_velocity + second_velocity),
            abs(first_velocity - second_velocity),
        ))

    for _, channel in CLINICAL_DYNAMICS_GLOBAL_CHANNELS:
        output.extend((
            float(by_channel[channel][range_column]),
            float(by_channel[channel][velocity_column]),
        ))

    result = np.asarray(output, dtype=np.float64)
    if result.shape != (CLINICAL_DYNAMICS_DIM,):
        raise AssertionError("clinical dynamics feature dimension drifted")
    if not np.isfinite(result).all():
        raise ValueError("clinical dynamics produced nonfinite values")
    return result


__all__ = [
    "CLINICAL_DYNAMICS_DIM",
    "CLINICAL_DYNAMICS_GLOBAL_CHANNELS",
    "CLINICAL_DYNAMICS_GLOBAL_STAT_NAMES",
    "CLINICAL_DYNAMICS_PAIR_STAT_NAMES",
    "clinical_dynamics_feature_names",
    "clinical_dynamics_feature_vector",
]
