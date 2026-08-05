"""Frozen 110D summaries for dynamic clinical23_v2 trajectories."""
from __future__ import annotations

import numpy as np

from .clinical23 import CLINICAL23_NAMES


INPUT_SHAPE = (4, 32, 23)
MASK_SHAPE = INPUT_SHAPE[:2]
MIN_RECORDING_COVERAGE = 0.90
SUMMARY_STAT_NAMES = (
    "median",
    "iqr",
    "range",
    "max_abs_velocity_per_second",
)
BILATERAL_PAIRS = (
    ("fissure_h_mesh33_vs_mesh263", 0, 1),
    ("fissure_w_mesh33_vs_mesh263", 4, 5),
    ("eye_area_mesh33_vs_mesh263", 7, 8),
    ("brow_h_mesh33_vs_mesh263", 10, 11),
    ("corner_y_mesh61_vs_mesh291", 14, 15),
    ("corner_x_mesh61_vs_mesh291", 18, 19),
)
FEATURE_NAMES: tuple[str, ...] = tuple(
    f"{channel}__{statistic}"
    for channel in CLINICAL23_NAMES
    for statistic in SUMMARY_STAT_NAMES
) + tuple(
    f"{pair}__{statistic}"
    for pair, _, _ in BILATERAL_PAIRS
    for statistic in ("correlation", "amplitude_ratio", "lag_seconds")
)

if len(FEATURE_NAMES) != 110 or len(set(FEATURE_NAMES)) != 110:
    raise AssertionError("110D feature-name contract drifted")


def _validated_inputs(
    clinical23: np.ndarray,
    valid_mask: np.ndarray,
    timestamps: np.ndarray,
    source_frame_indices: np.ndarray,
    source_frame_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(clinical23)
    mask = np.asarray(valid_mask)
    times = np.asarray(timestamps)
    indices = np.asarray(source_frame_indices)
    if features.shape != INPUT_SHAPE or features.dtype.kind not in {"f", "i", "u"}:
        raise ValueError("clinical23 must be real numeric with shape (4, 32, 23)")
    if mask.shape != MASK_SHAPE or mask.dtype != np.dtype(bool):
        raise ValueError("valid_mask must be bool with shape (4, 32)")
    if times.shape != MASK_SHAPE or times.dtype.kind not in {"f", "i", "u"}:
        raise ValueError("timestamps must be numeric with shape (4, 32)")
    if indices.shape != MASK_SHAPE or indices.dtype.kind not in {"i", "u"}:
        raise ValueError("source_frame_indices must be integer with shape (4, 32)")
    if (
        isinstance(source_frame_count, (bool, np.bool_))
        or not isinstance(source_frame_count, (int, np.integer))
        or int(source_frame_count) < 128
    ):
        raise ValueError("source_frame_count must be an integer of at least 128")
    source_frame_count = int(source_frame_count)
    normalized_times = times.astype(np.float64, copy=False)
    if not np.isfinite(normalized_times).all() or not np.all(
        normalized_times[:, 1:] > normalized_times[:, :-1]
    ):
        raise ValueError("timestamps must be finite and strictly increasing per window")
    if np.any(indices < 0):
        raise ValueError("source_frame_indices must be nonnegative")
    if not np.all(indices[:, 1:] - indices[:, :-1] == 1):
        raise ValueError("source frames must be adjacent within every window")
    final_start = source_frame_count - INPUT_SHAPE[1]
    expected_starts = tuple(
        (window * final_start) // (INPUT_SHAPE[0] - 1)
        for window in range(INPUT_SHAPE[0])
    )
    observed_starts = tuple(int(value) for value in indices[:, 0].tolist())
    if observed_starts != expected_starts:
        raise ValueError(
            "source windows do not match the frozen deterministic starts "
            f"{expected_starts} for {source_frame_count} source frames"
        )
    if int(indices[-1, -1]) != source_frame_count - 1:
        raise ValueError("last source window must span the end of the recording")
    if np.any(features[~mask] != 0):
        raise ValueError("invalid or padded clinical23 rows must be canonical zero")
    coverage = float(mask.mean())
    if coverage < MIN_RECORDING_COVERAGE:
        raise ValueError(
            f"recording coverage {coverage:.3f} is below required "
            f"{MIN_RECORDING_COVERAGE:.0%}"
        )
    if not np.isfinite(features[mask]).all():
        raise ValueError("valid clinical23 rows must be finite")
    return features, mask, times, indices


def _per_second_differences(
    features: np.ndarray,
    mask: np.ndarray,
    timestamps: np.ndarray,
    source_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    differences = np.zeros(features.shape, dtype=np.float64)
    difference_mask = np.zeros(mask.shape, dtype=bool)
    left_time = timestamps[:, :-1].astype(np.float64, copy=False)
    right_time = timestamps[:, 1:].astype(np.float64, copy=False)
    elapsed = right_time - left_time
    pair_valid = (
        mask[:, :-1]
        & mask[:, 1:]
        & (source_indices[:, 1:] - source_indices[:, :-1] == 1)
        & np.isfinite(elapsed)
        & (elapsed > 0)
        & np.isfinite(features[:, :-1]).all(axis=-1)
        & np.isfinite(features[:, 1:]).all(axis=-1)
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        candidate = (
            features[:, 1:].astype(np.float64, copy=False)
            - features[:, :-1].astype(np.float64, copy=False)
        ) / elapsed[..., None]
    pair_valid &= np.isfinite(candidate).all(axis=-1)
    differences[:, 1:] = np.where(pair_valid[..., None], candidate, 0.0)
    difference_mask[:, 1:] = pair_valid
    return differences, difference_mask


def _channel_summaries(
    features: np.ndarray,
    mask: np.ndarray,
    timestamps: np.ndarray,
    source_indices: np.ndarray,
) -> np.ndarray:
    derivatives, derivative_mask = _per_second_differences(
        features, mask, timestamps, source_indices
    )
    output = np.empty((23, 4), dtype=np.float64)
    for channel in range(23):
        values = features[..., channel][mask].astype(np.float64, copy=False)
        quartiles = np.quantile(values, (0.25, 0.75))
        velocities = np.abs(derivatives[..., channel][derivative_mask])
        output[channel] = (
            float(np.median(values)),
            float(quartiles[1] - quartiles[0]),
            float(np.max(values) - np.min(values)),
            float(np.max(velocities)) if velocities.size else 0.0,
        )
    return output.reshape(-1)


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.size < 2 or first.shape != second.shape:
        return 0.0
    first = first - first.mean()
    second = second - second.mean()
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if not np.isfinite(denominator) or denominator <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))


def _bilateral_dynamics(
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray,
    timestamps: np.ndarray,
    max_lag_frames: int = 5,
) -> tuple[float, float, float]:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    mask = np.asarray(mask)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    finite = np.isfinite(first) & np.isfinite(second)
    paired = mask & finite
    correlation = _pearson(first[paired], second[paired])
    if paired.any():
        amplitude_first = float(np.ptp(first[paired]))
        amplitude_second = float(np.ptp(second[paired]))
        largest = max(amplitude_first, amplitude_second)
        amplitude_ratio = (
            min(amplitude_first, amplitude_second) / largest
            if largest > 0 else 0.0
        )
    else:
        amplitude_ratio = 0.0

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
            elapsed = timestamps[:, lag:] - timestamps[:, :-lag]
        elif lag < 0:
            offset = -lag
            first_values = first[:, offset:]
            second_values = second[:, :-offset]
            pair_mask = mask[:, offset:] & mask[:, :-offset]
            elapsed = timestamps[:, :-offset] - timestamps[:, offset:]
        else:
            first_values = first
            second_values = second
            pair_mask = mask.copy()
            elapsed = np.zeros_like(timestamps)
        pair_mask &= np.isfinite(first_values) & np.isfinite(second_values)
        candidate = _pearson(first_values[pair_mask], second_values[pair_mask])
        if candidate > best_correlation + 1e-12:
            best_correlation = candidate
            best_lag_seconds = (
                float(np.median(elapsed[pair_mask])) if pair_mask.any() else 0.0
            )
    return correlation, float(amplitude_ratio), best_lag_seconds


def build_110d_features(
    clinical23: np.ndarray,
    valid_mask: np.ndarray,
    timestamps: np.ndarray,
    source_frame_indices: np.ndarray,
    source_frame_count: int,
) -> np.ndarray:
    """Build the frozen 110D recording vector from four 32-frame windows."""
    features, mask, times, indices = _validated_inputs(
        clinical23,
        valid_mask,
        timestamps,
        source_frame_indices,
        source_frame_count,
    )
    summary = _channel_summaries(features, mask, times, indices)
    paired = np.asarray([
        value
        for _, first_index, second_index in BILATERAL_PAIRS
        for value in _bilateral_dynamics(
            features[..., first_index],
            features[..., second_index],
            mask,
            times,
        )
    ], dtype=np.float64)
    output = np.concatenate((summary, paired))
    if output.shape != (110,) or not np.isfinite(output).all():
        raise ValueError("110D feature construction produced invalid output")
    return output


__all__ = [
    "BILATERAL_PAIRS",
    "FEATURE_NAMES",
    "INPUT_SHAPE",
    "MIN_RECORDING_COVERAGE",
    "SUMMARY_STAT_NAMES",
    "build_110d_features",
]
