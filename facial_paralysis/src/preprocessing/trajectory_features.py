"""Leak-safe classical summaries for fixed dynamic facial trajectories.

The 95-column input contract is already validated by the dynamic cache loader.
This module preserves the four independent temporal windows: frame derivatives
and lagged pairs are never formed across a window boundary.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.covariance import LedoitWolf

from ..datasets.dynamic_landmark import (
    DYNAMIC_FEATURE_NAMES,
    DYNAMIC_FEATURE_SHAPE,
    DYNAMIC_MASK_SHAPE,
    per_second_first_differences,
)


SUMMARY_STAT_NAMES = ("median", "iqr", "range", "max_abs_velocity_per_second")
BLENDSHAPE_CHANNELS = tuple(range(72))
LANDMARK_CHANNELS = tuple(range(72, 95))

# Side names intentionally follow capture-side MediaPipe mesh anchors. Patient
# left/right is unknowable until mirroring provenance is frozen.
LANDMARK_BILATERAL_PAIRS = (
    ("fissure_h_mesh33_vs_mesh263", 72, 73),
    ("fissure_w_mesh33_vs_mesh263", 76, 77),
    ("eye_area_mesh33_vs_mesh263", 79, 80),
    ("brow_h_mesh33_vs_mesh263", 82, 83),
    ("corner_y_mesh61_vs_mesh291", 86, 87),
    ("corner_x_mesh61_vs_mesh291", 90, 91),
)
LANDMARK_REGIONS = (
    ("eye", tuple(range(72, 82))),
    ("brow", tuple(range(82, 86))),
    ("mouth", tuple(range(86, 95))),
)

BLENDSHAPE_DIM = len(BLENDSHAPE_CHANNELS) * len(SUMMARY_STAT_NAMES)
LANDMARK_DIM = (
    len(LANDMARK_CHANNELS) * len(SUMMARY_STAT_NAMES)
    + len(LANDMARK_BILATERAL_PAIRS) * 3
)
FUSION_DIM = BLENDSHAPE_DIM + LANDMARK_DIM
RAO_FUSION_DIM = FUSION_DIM + 1 + len(LANDMARK_REGIONS)


def _validated_recording_arrays(
    features: np.ndarray,
    valid_mask: np.ndarray,
    timestamps: np.ndarray,
    source_frame_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(features)
    valid_mask = np.asarray(valid_mask)
    timestamps = np.asarray(timestamps)
    source_frame_indices = np.asarray(source_frame_indices)
    if features.shape != DYNAMIC_FEATURE_SHAPE:
        raise ValueError(f"features must have shape {DYNAMIC_FEATURE_SHAPE}")
    if features.dtype.kind not in {"f", "i", "u"}:
        raise ValueError("features must be real numeric values")
    if valid_mask.shape != DYNAMIC_MASK_SHAPE or valid_mask.dtype != np.dtype(bool):
        raise ValueError(f"valid_mask must be bool with shape {DYNAMIC_MASK_SHAPE}")
    if timestamps.shape != DYNAMIC_MASK_SHAPE or timestamps.dtype.kind not in {
        "f", "i", "u",
    }:
        raise ValueError(f"timestamps must be numeric with shape {DYNAMIC_MASK_SHAPE}")
    if source_frame_indices.shape != DYNAMIC_MASK_SHAPE or (
        source_frame_indices.dtype.kind not in {"i", "u"}
    ):
        raise ValueError(
            f"source_frame_indices must be integer with shape {DYNAMIC_MASK_SHAPE}"
        )
    if not np.isfinite(timestamps).all():
        raise ValueError("timestamps must be finite")
    if not np.all(timestamps[:, 1:] > timestamps[:, :-1]):
        raise ValueError("timestamps must increase strictly within each window")
    source_steps = source_frame_indices[:, 1:] - source_frame_indices[:, :-1]
    if not np.all(source_steps == 1):
        raise ValueError("source frames must be adjacent within each fixed window")
    if not valid_mask.any():
        raise ValueError("at least one valid frame is required")
    if not np.isfinite(features[valid_mask]).all():
        raise ValueError("valid feature rows must be finite")
    return features, valid_mask, timestamps, source_frame_indices


def _validated_channels(channel_indices: Sequence[int]) -> tuple[int, ...]:
    try:
        channels = tuple(int(index) for index in channel_indices)
    except (TypeError, ValueError) as exc:
        raise ValueError("channel_indices must be integer indices") from exc
    if not channels or len(set(channels)) != len(channels):
        raise ValueError("channel_indices must be nonempty and unique")
    if any(index < 0 or index >= DYNAMIC_FEATURE_SHAPE[-1] for index in channels):
        raise ValueError("channel index is outside the frozen 95-column schema")
    return channels


def summarize_trajectory_channels(
    features: np.ndarray,
    valid_mask: np.ndarray,
    timestamps: np.ndarray,
    source_frame_indices: np.ndarray,
    channel_indices: Sequence[int],
) -> np.ndarray:
    """Return median, IQR, range, and maximum absolute velocity per channel."""
    features, valid_mask, timestamps, source_frame_indices = (
        _validated_recording_arrays(
            features, valid_mask, timestamps, source_frame_indices
        )
    )
    channels = _validated_channels(channel_indices)
    selected = features[..., channels].astype(np.float64, copy=False)
    derivatives, derivative_mask = per_second_first_differences(
        selected, valid_mask, timestamps, source_frame_indices
    )
    result = np.empty((len(channels), len(SUMMARY_STAT_NAMES)), dtype=np.float64)
    for output_index in range(len(channels)):
        values = selected[..., output_index][valid_mask]
        if values.size == 0:
            raise ValueError("every summarized channel requires a valid frame")
        quartiles = np.quantile(values, (0.25, 0.75))
        velocities = np.abs(derivatives[..., output_index][derivative_mask])
        result[output_index] = (
            float(np.median(values)),
            float(quartiles[1] - quartiles[0]),
            float(np.max(values) - np.min(values)),
            float(np.max(velocities)) if velocities.size else 0.0,
        )
    if not np.isfinite(result).all():
        raise ValueError("trajectory summary produced nonfinite values")
    return result.reshape(-1)


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.size < 2 or first.shape != second.shape:
        return 0.0
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = float(
        np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    )
    if not np.isfinite(denominator) or denominator <= np.finfo(np.float64).eps:
        return 0.0
    value = float(np.dot(first_centered, second_centered) / denominator)
    return float(np.clip(value, -1.0, 1.0))


def bilateral_dynamics(
    capture_side_a: np.ndarray,
    capture_side_b: np.ndarray,
    valid_mask: np.ndarray,
    timestamps: np.ndarray,
    *,
    max_lag_frames: int = 5,
) -> tuple[float, float, float]:
    """Return pooled correlation, invariant amplitude ratio, and best lag.

    Positive lag means capture-side B occurs later than capture-side A. Lagged
    pairs are created independently inside each row/window, then pooled.
    """
    first = np.asarray(capture_side_a, dtype=np.float64)
    second = np.asarray(capture_side_b, dtype=np.float64)
    mask = np.asarray(valid_mask)
    times = np.asarray(timestamps, dtype=np.float64)
    if first.shape != DYNAMIC_MASK_SHAPE or second.shape != DYNAMIC_MASK_SHAPE:
        raise ValueError(f"bilateral trajectories must have shape {DYNAMIC_MASK_SHAPE}")
    if mask.shape != DYNAMIC_MASK_SHAPE or mask.dtype != np.dtype(bool):
        raise ValueError(f"valid_mask must be bool with shape {DYNAMIC_MASK_SHAPE}")
    if times.shape != DYNAMIC_MASK_SHAPE or not np.isfinite(times).all():
        raise ValueError(f"timestamps must be finite with shape {DYNAMIC_MASK_SHAPE}")
    if not np.all(times[:, 1:] > times[:, :-1]):
        raise ValueError("timestamps must increase strictly within each window")
    if isinstance(max_lag_frames, bool) or not isinstance(
        max_lag_frames, (int, np.integer)
    ) or int(max_lag_frames) < 0 or int(max_lag_frames) >= DYNAMIC_MASK_SHAPE[1]:
        raise ValueError("max_lag_frames must be an integer from 0 through 31")
    max_lag_frames = int(max_lag_frames)
    finite = np.isfinite(first) & np.isfinite(second)
    paired = mask & finite
    correlation = _pearson(first[paired], second[paired])
    if paired.any():
        amplitude_a = float(np.ptp(first[paired]))
        amplitude_b = float(np.ptp(second[paired]))
        largest = max(amplitude_a, amplitude_b)
        amplitude_ratio = min(amplitude_a, amplitude_b) / largest if largest > 0 else 0.0
    else:
        amplitude_ratio = 0.0

    # Prefer zero and then smaller absolute lags when correlations tie. This
    # avoids reporting arbitrary latency for constant or periodic signals.
    lag_order = sorted(
        range(-max_lag_frames, max_lag_frames + 1),
        key=lambda lag: (abs(lag), lag),
    )
    best_correlation = -np.inf
    best_lag_seconds = 0.0
    for lag in lag_order:
        if lag > 0:
            first_values = first[:, :-lag]
            second_values = second[:, lag:]
            pair_mask = mask[:, :-lag] & mask[:, lag:]
            elapsed = times[:, lag:] - times[:, :-lag]
        elif lag < 0:
            offset = -lag
            first_values = first[:, offset:]
            second_values = second[:, :-offset]
            pair_mask = mask[:, offset:] & mask[:, :-offset]
            elapsed = times[:, :-offset] - times[:, offset:]
        else:
            first_values = first
            second_values = second
            pair_mask = mask.copy()
            elapsed = np.zeros_like(times)
        pair_mask &= np.isfinite(first_values) & np.isfinite(second_values)
        candidate = _pearson(first_values[pair_mask], second_values[pair_mask])
        if candidate > best_correlation + 1e-12:
            best_correlation = candidate
            best_lag_seconds = (
                float(np.median(elapsed[pair_mask])) if pair_mask.any() else 0.0
            )
    return correlation, float(amplitude_ratio), best_lag_seconds


def _landmark_vector(
    features: np.ndarray,
    valid_mask: np.ndarray,
    timestamps: np.ndarray,
    source_frame_indices: np.ndarray,
) -> np.ndarray:
    summary = summarize_trajectory_channels(
        features, valid_mask, timestamps, source_frame_indices, LANDMARK_CHANNELS
    )
    pair_values = [
        value
        for _, first_index, second_index in LANDMARK_BILATERAL_PAIRS
        for value in bilateral_dynamics(
            features[..., first_index],
            features[..., second_index],
            valid_mask,
            timestamps,
        )
    ]
    result = np.concatenate((summary, np.asarray(pair_values, dtype=np.float64)))
    if result.shape != (LANDMARK_DIM,):
        raise AssertionError("landmark trajectory feature dimension drifted")
    return result


def trajectory_feature_set(
    feature_set: str,
    features: np.ndarray,
    valid_mask: np.ndarray,
    timestamps: np.ndarray,
    source_frame_indices: np.ndarray,
    *,
    reference_distances: np.ndarray | None = None,
) -> np.ndarray:
    """Build one frozen classical feature vector from a validated recording."""
    if feature_set not in {"blendshape", "landmark", "fusion", "rao_fusion"}:
        raise ValueError(f"unknown trajectory feature set {feature_set!r}")
    blendshape = None
    landmark = None
    if feature_set in {"blendshape", "fusion", "rao_fusion"}:
        blendshape = summarize_trajectory_channels(
            features,
            valid_mask,
            timestamps,
            source_frame_indices,
            BLENDSHAPE_CHANNELS,
        )
    if feature_set in {"landmark", "fusion", "rao_fusion"}:
        landmark = _landmark_vector(
            features, valid_mask, timestamps, source_frame_indices
        )
    if feature_set == "blendshape":
        return blendshape
    if feature_set == "landmark":
        return landmark
    fusion = np.concatenate((blendshape, landmark))
    if feature_set == "fusion":
        return fusion
    distances = np.asarray(reference_distances) if reference_distances is not None else None
    if distances is None or distances.shape != (4,) or not np.isfinite(distances).all():
        raise ValueError("rao_fusion requires exactly four finite reference distances")
    return np.concatenate((fusion, distances.astype(np.float64, copy=False)))


def _summary_names(channels: Sequence[int]) -> tuple[str, ...]:
    return tuple(
        f"{DYNAMIC_FEATURE_NAMES[index]}__{statistic}"
        for index in channels
        for statistic in SUMMARY_STAT_NAMES
    )


def trajectory_feature_names(feature_set: str) -> tuple[str, ...]:
    """Return the exact ordered names for a frozen trajectory feature set."""
    blendshape = _summary_names(BLENDSHAPE_CHANNELS)
    landmark = _summary_names(LANDMARK_CHANNELS) + tuple(
        f"{pair_name}__{statistic}"
        for pair_name, _, _ in LANDMARK_BILATERAL_PAIRS
        for statistic in ("correlation", "amplitude_ratio", "lag_seconds")
    )
    if feature_set == "blendshape":
        return blendshape
    if feature_set == "landmark":
        return landmark
    if feature_set == "fusion":
        return blendshape + landmark
    if feature_set == "rao_fusion":
        return blendshape + landmark + (
            "healthy_reference__bilateral_correlation_mahalanobis",
            "healthy_reference__eye_gaussian_w2",
            "healthy_reference__brow_gaussian_w2",
            "healthy_reference__mouth_gaussian_w2",
        )
    raise ValueError(f"unknown trajectory feature set {feature_set!r}")


def _psd_square_root(matrix: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    tolerance = np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(eigenvalues))))
    if np.min(eigenvalues) < -100.0 * tolerance:
        raise ValueError("covariance must be positive semidefinite")
    return (eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))) @ eigenvectors.T


def gaussian_wasserstein_distance(
    first_mean: np.ndarray,
    first_covariance: np.ndarray,
    second_mean: np.ndarray,
    second_covariance: np.ndarray,
) -> float:
    """Return the Gaussian 2-Wasserstein distance (not its square)."""
    first_mean = np.asarray(first_mean, dtype=np.float64)
    second_mean = np.asarray(second_mean, dtype=np.float64)
    first_covariance = np.asarray(first_covariance, dtype=np.float64)
    second_covariance = np.asarray(second_covariance, dtype=np.float64)
    if first_mean.ndim != 1 or first_mean.shape != second_mean.shape:
        raise ValueError("Gaussian means must be same-length vectors")
    expected = (first_mean.size, first_mean.size)
    if first_covariance.shape != expected or second_covariance.shape != expected:
        raise ValueError("Gaussian covariances must be square and match the means")
    if not all(np.isfinite(value).all() for value in (
        first_mean, second_mean, first_covariance, second_covariance
    )):
        raise ValueError("Gaussian parameters must be finite")
    second_root = _psd_square_root(second_covariance)
    middle_root = _psd_square_root(
        second_root @ first_covariance @ second_root
    )
    squared = float(
        np.dot(first_mean - second_mean, first_mean - second_mean)
        + np.trace(first_covariance)
        + np.trace(second_covariance)
        - 2.0 * np.trace(middle_root)
    )
    scale = max(
        1.0,
        float(np.dot(first_mean - second_mean, first_mean - second_mean)),
        float(np.trace(first_covariance) + np.trace(second_covariance)),
    )
    if abs(squared) <= 1e-10 * scale:
        squared = 0.0
    if squared < 0.0 or not np.isfinite(squared):
        raise ValueError("Gaussian Wasserstein distance is numerically invalid")
    return float(np.sqrt(squared))


def _fit_shrinkage(samples: np.ndarray) -> LedoitWolf:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] < 1:
        raise ValueError("shrinkage covariance requires at least two sample rows")
    if not np.isfinite(samples).all():
        raise ValueError("shrinkage covariance samples must be finite")
    return LedoitWolf(assume_centered=False).fit(samples)


class HealthyReferencePrototype:
    """A reference fitted only from control trajectories supplied by the caller."""

    def fit(
        self,
        control_features: np.ndarray,
        control_masks: np.ndarray,
        control_timestamps: np.ndarray,
        control_source_frame_indices: np.ndarray,
        *,
        record_indices: np.ndarray | None = None,
    ) -> "HealthyReferencePrototype":
        arrays = _validated_batch(
            control_features,
            control_masks,
            control_timestamps,
            control_source_frame_indices,
        )
        features, masks, timestamps, source_indices = arrays
        if features.shape[0] < 2:
            raise ValueError("healthy reference requires at least two controls")
        if record_indices is None:
            fit_indices = np.arange(features.shape[0], dtype=np.int64)
        else:
            fit_indices = np.asarray(record_indices)
            if (
                fit_indices.shape != (features.shape[0],)
                or fit_indices.dtype.kind not in {"i", "u"}
                or np.any(fit_indices < 0)
                or np.unique(fit_indices).size != fit_indices.size
            ):
                raise ValueError("record_indices must uniquely identify supplied controls")
            fit_indices = fit_indices.astype(np.int64, copy=True)

        correlations = np.stack([
            _bilateral_correlations(
                features[index], masks[index], timestamps[index]
            )
            for index in range(features.shape[0])
        ])
        self.correlation_model_ = _fit_shrinkage(correlations)
        self.region_models_: dict[str, LedoitWolf] = {}
        for region_name, channels in LANDMARK_REGIONS:
            pooled = np.concatenate([
                features[index][masks[index]][:, channels]
                for index in range(features.shape[0])
            ])
            self.region_models_[region_name] = _fit_shrinkage(pooled)
        self.fit_record_indices_ = fit_indices
        self._fitted = True
        return self

    def transform(
        self,
        features: np.ndarray,
        masks: np.ndarray,
        timestamps: np.ndarray,
        source_frame_indices: np.ndarray,
    ) -> np.ndarray:
        if not getattr(self, "_fitted", False):
            raise ValueError("healthy reference must be fitted before transform")
        features, masks, timestamps, source_indices = _validated_batch(
            features, masks, timestamps, source_frame_indices
        )
        output = np.empty((features.shape[0], 4), dtype=np.float64)
        for index in range(features.shape[0]):
            correlations = _bilateral_correlations(
                features[index], masks[index], timestamps[index]
            )
            difference = correlations - self.correlation_model_.location_
            mahalanobis_squared = float(
                difference @ self.correlation_model_.precision_ @ difference
            )
            output[index, 0] = np.sqrt(max(mahalanobis_squared, 0.0))
            for region_index, (region_name, channels) in enumerate(
                LANDMARK_REGIONS, start=1
            ):
                samples = features[index][masks[index]][:, channels]
                subject = _fit_shrinkage(samples)
                reference = self.region_models_[region_name]
                output[index, region_index] = gaussian_wasserstein_distance(
                    subject.location_,
                    subject.covariance_,
                    reference.location_,
                    reference.covariance_,
                )
        if not np.isfinite(output).all() or np.any(output < 0.0):
            raise ValueError("healthy reference produced invalid distances")
        return output


def _bilateral_correlations(
    features: np.ndarray,
    valid_mask: np.ndarray,
    timestamps: np.ndarray,
) -> np.ndarray:
    return np.asarray([
        bilateral_dynamics(
            features[..., first], features[..., second], valid_mask, timestamps
        )[0]
        for _, first, second in LANDMARK_BILATERAL_PAIRS
    ], dtype=np.float64)


def _validated_batch(
    features: np.ndarray,
    masks: np.ndarray,
    timestamps: np.ndarray,
    source_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(features)
    masks = np.asarray(masks)
    timestamps = np.asarray(timestamps)
    source_indices = np.asarray(source_indices)
    if features.ndim != 4 or features.shape[1:] != DYNAMIC_FEATURE_SHAPE:
        raise ValueError(f"features batch must have shape (N, {DYNAMIC_FEATURE_SHAPE})")
    expected = (features.shape[0],) + DYNAMIC_MASK_SHAPE
    if masks.shape != expected or timestamps.shape != expected or source_indices.shape != expected:
        raise ValueError("mask, timestamp, and source-index batches must align")
    for index in range(features.shape[0]):
        _validated_recording_arrays(
            features[index], masks[index], timestamps[index], source_indices[index]
        )
    return features, masks, timestamps, source_indices


__all__ = [
    "BLENDSHAPE_CHANNELS",
    "BLENDSHAPE_DIM",
    "FUSION_DIM",
    "HealthyReferencePrototype",
    "LANDMARK_BILATERAL_PAIRS",
    "LANDMARK_CHANNELS",
    "LANDMARK_DIM",
    "LANDMARK_REGIONS",
    "RAO_FUSION_DIM",
    "SUMMARY_STAT_NAMES",
    "bilateral_dynamics",
    "gaussian_wasserstein_distance",
    "summarize_trajectory_channels",
    "trajectory_feature_names",
    "trajectory_feature_set",
]
