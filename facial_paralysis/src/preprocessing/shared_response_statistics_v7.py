"""Label-free dense response statistics for shared action encoding."""
from __future__ import annotations

import numpy as np


STATISTIC_COUNT = 5
DENSE_RESPONSE_DIM = 478 * 3 * STATISTIC_COUNT * 2


def _immutable(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)


def _stream_statistics(
    stream: np.ndarray, timestamps: np.ndarray, available: np.ndarray
) -> np.ndarray:
    count, actions = available.shape
    output = np.zeros((count, actions, 478, 3, STATISTIC_COUNT), dtype=np.float64)
    for participant, action in np.argwhere(available):
        values = stream[participant, action].astype(np.float64, copy=False)
        times = timestamps[participant, action]
        intervals = np.diff(times)
        if np.any(intervals <= 0.0):
            raise ValueError("dense velocity requires increasing real seconds")
        quantiles = np.quantile(values, (0.10, 0.90), axis=0)
        velocity = np.abs(np.diff(values, axis=0)) / intervals[:, None, None]
        output[participant, action] = np.stack((
            np.median(values, axis=0), quantiles[0], quantiles[1],
            np.max(values, axis=0) - np.min(values, axis=0),
            np.max(velocity, axis=0),
        ), axis=-1)
    return output


def dense_response_statistics_v7(
    original: np.ndarray,
    mirrored: np.ndarray,
    timestamps: np.ndarray,
    available: np.ndarray,
) -> np.ndarray:
    if (
        type(original) is not np.ndarray or original.dtype != np.dtype(np.float32)
        or original.ndim != 5 or original.shape[2:] != (32, 478, 3)
        or type(mirrored) is not np.ndarray or mirrored.dtype != np.dtype(np.float32)
        or mirrored.shape != original.shape
        or type(timestamps) is not np.ndarray or timestamps.dtype != np.dtype(np.float64)
        or timestamps.shape != original.shape[:2] + (32,)
        or type(available) is not np.ndarray or available.dtype != np.dtype(bool)
        or available.shape != original.shape[:2]
        or not np.isfinite(original).all() or not np.isfinite(mirrored).all()
        or not np.isfinite(timestamps).all()
    ):
        raise ValueError("dense response statistics received malformed evidence")
    first = _stream_statistics(original, timestamps, available)
    second = _stream_statistics(mirrored, timestamps, available)
    paired = np.concatenate((0.5 * (first + second), np.abs(first - second)), axis=-1)
    result = paired.reshape(*available.shape, DENSE_RESPONSE_DIM)
    if not np.isfinite(result).all() or np.any(result[~available] != 0.0):
        raise RuntimeError("dense response statistics failed finite missingness QC")
    return _immutable(result)


__all__ = ["DENSE_RESPONSE_DIM", "STATISTIC_COUNT", "dense_response_statistics_v7"]
