from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon


@dataclass(frozen=True)
class IntervalEstimate:
    estimate: float
    lower: float
    upper: float
    confidence: float
    n: int


@dataclass(frozen=True)
class BlandAltmanSummary:
    mean_difference: float
    lower_limit: float
    upper_limit: float
    n: int


@dataclass(frozen=True)
class EquivalenceResult:
    mean_difference: float
    lower: float
    upper: float
    margin: float
    confidence: float
    outcome: str
    n: int


@dataclass(frozen=True)
class TemporalLagEstimate:
    lag_ms: float
    peak_correlation: float
    resample_hz: float
    n_aligned: int


def _finite_vector(values: Iterable[float], *, name: str = "values") -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array


def _finite_points(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] == 0:
        raise ValueError(f"{name} must have shape (n, 2)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite coordinates")
    return array


def angular_accuracy(gaze_degrees: np.ndarray, target_degrees: np.ndarray) -> float:
    gaze = _finite_points(gaze_degrees, name="gaze")
    target = _finite_points(target_degrees, name="target")
    if gaze.shape != target.shape:
        raise ValueError("gaze and target must have matching shapes")
    return float(np.linalg.norm(gaze - target, axis=1).mean())


def rms_precision(gaze_degrees: np.ndarray) -> float:
    gaze = _finite_points(gaze_degrees, name="gaze")
    if len(gaze) < 2:
        raise ValueError("gaze must contain at least two finite points")
    squared_displacement = np.square(np.diff(gaze, axis=0)).sum(axis=1)
    return float(np.sqrt(squared_displacement.mean()))


def data_loss(valid: Iterable[bool]) -> float:
    values = np.asarray(list(valid))
    if values.ndim != 1 or values.size == 0:
        raise ValueError("valid flags must be a non-empty vector")
    return float(1 - values.astype(bool).mean())


def effective_sampling_rate(timestamps_ms: Iterable[float]) -> float:
    timestamps = _finite_vector(timestamps_ms, name="timestamps_ms")
    if timestamps.size < 2:
        raise ValueError("timestamps_ms must contain at least two values")
    ordered = np.sort(timestamps)
    elapsed_seconds = (ordered[-1] - ordered[0]) / 1000
    if elapsed_seconds <= 0:
        raise ValueError("timestamps_ms must span positive elapsed time")
    return float((len(ordered) - 1) / elapsed_seconds)


def interval_cv(timestamps_ms: Iterable[float]) -> float:
    timestamps = _finite_vector(timestamps_ms, name="timestamps_ms")
    intervals = np.diff(np.sort(timestamps))
    if intervals.size == 0 or (intervals <= 0).any():
        raise ValueError("timestamps_ms must have positive unique intervals")
    return float(np.std(intervals, ddof=0) / np.mean(intervals))


def estimate_temporal_lag(
    webcam_timestamps_ms: Iterable[float],
    webcam_xy: np.ndarray,
    reference_timestamps_ms: Iterable[float],
    reference_xy: np.ndarray,
    *,
    resample_hz: float = 30.0,
    max_lag_ms: float = 500.0,
) -> TemporalLagEstimate:
    webcam_time = _finite_vector(webcam_timestamps_ms, name="webcam_timestamps_ms")
    reference_time = _finite_vector(reference_timestamps_ms, name="reference_timestamps_ms")
    webcam_points = _finite_points(webcam_xy, name="webcam_xy")
    reference_points = _finite_points(reference_xy, name="reference_xy")
    if len(webcam_time) != len(webcam_points) or len(reference_time) != len(reference_points):
        raise ValueError("timestamps and gaze points must have matching lengths")
    if (np.diff(webcam_time) <= 0).any() or (np.diff(reference_time) <= 0).any():
        raise ValueError("timestamps must be strictly increasing")
    if resample_hz <= 0 or max_lag_ms <= 0:
        raise ValueError("resample_hz and max_lag_ms must be positive")

    start = max(float(webcam_time[0]), float(reference_time[0]))
    stop = min(float(webcam_time[-1]), float(reference_time[-1]))
    step_ms = 1000 / resample_hz
    common_time = np.arange(start, stop + step_ms * 0.25, step_ms)
    max_lag_steps = int(round(max_lag_ms / step_ms))
    if len(common_time) < max(20, 2 * max_lag_steps + 5):
        raise ValueError("streams do not have enough overlapping samples for lag estimation")

    def interpolate(time: np.ndarray, points: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [np.interp(common_time, time, points[:, axis]) for axis in range(2)]
        )

    webcam_common = interpolate(webcam_time, webcam_points)
    reference_common = interpolate(reference_time, reference_points)

    def standardize(points: np.ndarray) -> np.ndarray:
        scale = points.std(axis=0)
        if (scale == 0).any():
            raise ValueError("gaze trajectories must vary in both axes")
        return (points - points.mean(axis=0)) / scale

    webcam_common = standardize(webcam_common)
    reference_common = standardize(reference_common)
    best_lag = 0
    best_correlation = -np.inf
    best_n = 0
    for lag_steps in range(-max_lag_steps, max_lag_steps + 1):
        if lag_steps > 0:
            reference_slice = reference_common[:-lag_steps]
            webcam_slice = webcam_common[lag_steps:]
        elif lag_steps < 0:
            reference_slice = reference_common[-lag_steps:]
            webcam_slice = webcam_common[:lag_steps]
        else:
            reference_slice = reference_common
            webcam_slice = webcam_common
        correlation = float(
            np.corrcoef(reference_slice.ravel(), webcam_slice.ravel())[0, 1]
        )
        if correlation > best_correlation:
            best_lag = lag_steps
            best_correlation = correlation
            best_n = len(reference_slice)
    return TemporalLagEstimate(
        lag_ms=float(best_lag * step_ms),
        peak_correlation=best_correlation,
        resample_hz=float(resample_hz),
        n_aligned=best_n,
    )


def density_map(
    x_norm: Iterable[float],
    y_norm: Iterable[float],
    *,
    grid_size: int = 48,
) -> np.ndarray:
    x = _finite_vector(x_norm, name="x_norm")
    y = _finite_vector(y_norm, name="y_norm")
    if x.shape != y.shape:
        raise ValueError("x_norm and y_norm must have matching shapes")
    if grid_size < 4:
        raise ValueError("grid_size must be at least 4")
    if ((x < 0) | (x > 1) | (y < 0) | (y > 1)).any():
        raise ValueError("normalized gaze coordinates must be within [0, 1]")
    histogram, _, _ = np.histogram2d(y, x, bins=grid_size, range=[[0, 1], [0, 1]])
    return _normalize_density(histogram)


def _normalize_density(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("density must contain finite non-negative values")
    total = float(array.sum())
    if total <= 0:
        raise ValueError("density must contain positive density")
    return array / total


def map_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_density = _normalize_density(first)
    second_density = _normalize_density(second)
    if first_density.shape != second_density.shape:
        raise ValueError("density maps must have matching shapes")
    if np.allclose(first_density, second_density):
        return 1.0
    if np.std(first_density) == 0 or np.std(second_density) == 0:
        return 0.0
    return float(np.corrcoef(first_density.ravel(), second_density.ravel())[0, 1])


def histogram_intersection(first: np.ndarray, second: np.ndarray) -> float:
    first_density = _normalize_density(first)
    second_density = _normalize_density(second)
    if first_density.shape != second_density.shape:
        raise ValueError("density maps must have matching shapes")
    return float(np.minimum(first_density, second_density).sum())


def jensen_shannon_distance(first: np.ndarray, second: np.ndarray) -> float:
    first_density = _normalize_density(first)
    second_density = _normalize_density(second)
    if first_density.shape != second_density.shape:
        raise ValueError("density maps must have matching shapes")
    return float(jensenshannon(first_density.ravel(), second_density.ravel(), base=2))


def _hotspots(density: np.ndarray, quantile: float) -> np.ndarray:
    normalized = _normalize_density(density)
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    positive = normalized[normalized > 0]
    threshold = np.quantile(positive, quantile)
    return normalized >= threshold


def hotspot_dice(first: np.ndarray, second: np.ndarray, *, quantile: float = 0.9) -> float:
    first_hotspots = _hotspots(first, quantile)
    second_hotspots = _hotspots(second, quantile)
    if first_hotspots.shape != second_hotspots.shape:
        raise ValueError("density maps must have matching shapes")
    denominator = int(first_hotspots.sum() + second_hotspots.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(first_hotspots, second_hotspots).sum() / denominator)


def density_centroid_distance(first: np.ndarray, second: np.ndarray) -> float:
    first_density = _normalize_density(first)
    second_density = _normalize_density(second)
    if first_density.shape != second_density.shape:
        raise ValueError("density maps must have matching shapes")
    row_coordinates, column_coordinates = np.indices(first_density.shape, dtype=float)
    if first_density.shape[0] > 1:
        row_coordinates /= first_density.shape[0] - 1
    if first_density.shape[1] > 1:
        column_coordinates /= first_density.shape[1] - 1

    def centroid(density: np.ndarray) -> np.ndarray:
        return np.array(
            [
                float((row_coordinates * density).sum()),
                float((column_coordinates * density).sum()),
            ]
        )

    return float(np.linalg.norm(centroid(first_density) - centroid(second_density)))


def lin_concordance(first: Iterable[float], second: Iterable[float]) -> float:
    x = _finite_vector(first, name="first")
    y = _finite_vector(second, name="second")
    if x.shape != y.shape:
        raise ValueError("vectors must have matching shapes")
    if np.allclose(x, y):
        return 1.0
    covariance = float(np.mean((x - x.mean()) * (y - y.mean())))
    denominator = float(np.var(x) + np.var(y) + (x.mean() - y.mean()) ** 2)
    return 0.0 if denominator == 0 else float(2 * covariance / denominator)


def total_variation_distance(first: Iterable[float], second: Iterable[float]) -> float:
    first_density = _normalize_density(np.asarray(list(first), dtype=float))
    second_density = _normalize_density(np.asarray(list(second), dtype=float))
    if first_density.shape != second_density.shape:
        raise ValueError("vectors must have matching shapes")
    return float(0.5 * np.abs(first_density - second_density).sum())


def bland_altman(webcam: Iterable[float], reference: Iterable[float]) -> BlandAltmanSummary:
    webcam_values = _finite_vector(webcam, name="webcam")
    reference_values = _finite_vector(reference, name="reference")
    if webcam_values.shape != reference_values.shape or webcam_values.size < 2:
        raise ValueError("paired participant values must have matching length of at least two")
    differences = webcam_values - reference_values
    mean_difference = float(differences.mean())
    standard_deviation = float(np.std(differences, ddof=1))
    return BlandAltmanSummary(
        mean_difference=mean_difference,
        lower_limit=mean_difference - 1.96 * standard_deviation,
        upper_limit=mean_difference + 1.96 * standard_deviation,
        n=len(differences),
    )


def cluster_bootstrap_mean(
    values: Iterable[float],
    clusters: Iterable[object],
    *,
    n_boot: int = 2000,
    seed: int = 20260813,
    confidence: float = 0.95,
) -> IntervalEstimate:
    value_array = _finite_vector(values)
    cluster_array = np.asarray(list(clusters), dtype=object)
    if cluster_array.shape != value_array.shape:
        raise ValueError("values and clusters must have matching shapes")
    unique_clusters = np.unique(cluster_array)
    if len(unique_clusters) < 2 or n_boot < 100:
        raise ValueError("bootstrap requires at least two clusters and 100 replicates")
    rng = np.random.default_rng(seed)
    replicates = np.empty(n_boot, dtype=float)
    cluster_values = {cluster: value_array[cluster_array == cluster] for cluster in unique_clusters}
    for index in range(n_boot):
        selected = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        replicates[index] = np.concatenate([cluster_values[cluster] for cluster in selected]).mean()
    alpha = 1 - confidence
    return IntervalEstimate(
        estimate=float(value_array.mean()),
        lower=float(np.quantile(replicates, alpha / 2)),
        upper=float(np.quantile(replicates, 1 - alpha / 2)),
        confidence=confidence,
        n=len(unique_clusters),
    )


def paired_equivalence(
    differences: Iterable[float],
    *,
    margin: float,
    confidence: float = 0.90,
) -> EquivalenceResult:
    values = _finite_vector(differences, name="differences")
    if values.size < 2:
        raise ValueError("equivalence requires at least two paired differences")
    if margin <= 0 or not 0 < confidence < 1:
        raise ValueError("margin and confidence must be positive and valid")
    mean_difference = float(values.mean())
    standard_error = float(stats.sem(values))
    if standard_error == 0:
        lower = upper = mean_difference
    else:
        lower, upper = stats.t.interval(
            confidence,
            df=len(values) - 1,
            loc=mean_difference,
            scale=standard_error,
        )
        lower, upper = float(lower), float(upper)
    if lower > -margin and upper < margin:
        outcome = "equivalent"
    elif lower > margin or upper < -margin:
        outcome = "not_equivalent"
    else:
        outcome = "inconclusive"
    return EquivalenceResult(
        mean_difference=mean_difference,
        lower=lower,
        upper=upper,
        margin=float(margin),
        confidence=confidence,
        outcome=outcome,
        n=len(values),
    )
