"""Common dense-clinical action bags for shared facial-weakness models."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .action_aligned_110d import mirror_clinical23_features
from .clinical_landmarks import CLINICAL_LANDMARK_NAMES, clinical_landmark_features
from .generalization_110d import LANDMARK_MI_110D, candidate_feature_vector


ACTION_TOKEN_FRAMES = 32
CLINICAL_TOKEN_DIM = 110
DENSE_POINT_COUNT = 478

_PALSENET_WINDOW_NAMES = (
    "FREE_EARLY",
    "FREE_MID_EARLY",
    "FREE_MID_LATE",
    "FREE_LATE",
)
_ACTION_NAME_MAP = {
    "NSM_KISS": "LIP_PUCKER",
    "NSM_OPEN": "MOUTH_OPEN",
    "NSM_SPREAD": "SMILE_SPREAD",
    "BROW_RAISE": "BROW_RAISE",
    "EYE_GENTLE": "EYE_GENTLE",
    "EYE_FORCEFUL": "EYE_FORCEFUL",
    "SMILE_GENTLE": "SMILE_GENTLE",
    "SMILE_FULL": "SMILE_FULL",
    "LIP_PUCKER": "LIP_PUCKER",
    "SHOW_BOTTOM_TEETH": "SHOW_BOTTOM_TEETH",
}


def _immutable(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


@dataclass(frozen=True)
class ClinicalActionBag:
    clinical_original: np.ndarray
    clinical_mirrored: np.ndarray
    dense_original: np.ndarray
    dense_mirrored: np.ndarray
    dense_valid_mask: np.ndarray
    dense_available: np.ndarray
    dense_timestamps: np.ndarray
    action_names: tuple[str, ...]


def _build_bag(
    *,
    clinical_original: np.ndarray,
    clinical_mirrored: np.ndarray,
    dense_original: np.ndarray,
    dense_mirrored: np.ndarray,
    dense_valid_mask: np.ndarray,
    dense_available: np.ndarray,
    dense_timestamps: np.ndarray,
    action_names: tuple[str, ...],
) -> ClinicalActionBag:
    action_count = len(action_names)
    expected_dense = (action_count, ACTION_TOKEN_FRAMES, DENSE_POINT_COUNT, 3)
    if (
        type(action_names) is not tuple
        or not action_names
        or len(set(action_names)) != action_count
        or clinical_original.shape != (action_count, CLINICAL_TOKEN_DIM)
        or clinical_mirrored.shape != clinical_original.shape
        or clinical_original.dtype != np.dtype(np.float64)
        or clinical_mirrored.dtype != np.dtype(np.float64)
        or dense_original.shape != expected_dense
        or dense_mirrored.shape != expected_dense
        or dense_original.dtype != np.dtype(np.float32)
        or dense_mirrored.dtype != np.dtype(np.float32)
        or dense_valid_mask.shape != expected_dense[:2]
        or dense_valid_mask.dtype != np.dtype(bool)
        or dense_available.shape != (action_count,)
        or dense_available.dtype != np.dtype(bool)
        or dense_timestamps.shape != expected_dense[:2]
        or dense_timestamps.dtype != np.dtype(np.float64)
        or not np.isfinite(clinical_original).all()
        or not np.isfinite(clinical_mirrored).all()
    ):
        raise ValueError("action bag differs from the dense-clinical contract")
    if np.any(dense_valid_mask & ~dense_available[:, None]):
        raise ValueError("unavailable dense actions cannot contain valid frames")
    if dense_available.any():
        supported = dense_available[:, None] & dense_valid_mask
        if not np.isfinite(dense_original[supported]).all():
            raise ValueError("valid original dense frames must be finite")
        if not np.isfinite(dense_mirrored[supported]).all():
            raise ValueError("valid mirrored dense frames must be finite")
        if not np.isfinite(dense_timestamps[dense_valid_mask]).all():
            raise ValueError("valid dense timestamps must be finite")
    return ClinicalActionBag(
        clinical_original=_immutable(clinical_original),
        clinical_mirrored=_immutable(clinical_mirrored),
        dense_original=_immutable(dense_original),
        dense_mirrored=_immutable(dense_mirrored),
        dense_valid_mask=_immutable(dense_valid_mask),
        dense_available=_immutable(dense_available),
        dense_timestamps=_immutable(dense_timestamps),
        action_names=action_names,
    )


def palsynet_window_token_bag(
    features: np.ndarray,
    valid_mask: np.ndarray,
    timestamps: np.ndarray,
    source_frame_indices: np.ndarray,
) -> ClinicalActionBag:
    """Turn four authenticated PalsyNet windows into clinical-only tokens."""
    values = np.asarray(features)
    mask = np.asarray(valid_mask)
    times = np.asarray(timestamps)
    indices = np.asarray(source_frame_indices)
    if (
        values.shape != (4, ACTION_TOKEN_FRAMES, 95)
        or values.dtype != np.dtype(np.float32)
        or mask.shape != (4, ACTION_TOKEN_FRAMES)
        or mask.dtype != np.dtype(bool)
        or times.shape != (4, ACTION_TOKEN_FRAMES)
        or times.dtype != np.dtype(np.float64)
        or indices.shape != (4, ACTION_TOKEN_FRAMES)
        or indices.dtype != np.dtype(np.int64)
    ):
        raise ValueError("PalsyNet windows must follow the frozen 4x32x95 contract")
    mirrored_values = mirror_clinical23_features(values)
    original_rows: list[np.ndarray] = []
    mirrored_rows: list[np.ndarray] = []
    for window in range(4):
        temporal = (
            mask[window : window + 1],
            times[window : window + 1],
            indices[window : window + 1],
        )
        original_rows.append(candidate_feature_vector(
            LANDMARK_MI_110D, values[window : window + 1], *temporal
        ))
        mirrored_rows.append(candidate_feature_vector(
            LANDMARK_MI_110D, mirrored_values[window : window + 1], *temporal
        ))
    dense_shape = (4, ACTION_TOKEN_FRAMES, DENSE_POINT_COUNT, 3)
    return _build_bag(
        clinical_original=np.stack(original_rows).astype(np.float64, copy=False),
        clinical_mirrored=np.stack(mirrored_rows).astype(np.float64, copy=False),
        dense_original=np.zeros(dense_shape, dtype=np.float32),
        dense_mirrored=np.zeros(dense_shape, dtype=np.float32),
        dense_valid_mask=np.zeros(dense_shape[:2], dtype=bool),
        dense_available=np.zeros(4, dtype=bool),
        dense_timestamps=np.zeros(dense_shape[:2], dtype=np.float64),
        action_names=_PALSENET_WINDOW_NAMES,
    )


def _validated_dense_inputs(
    original_actions: np.ndarray,
    mirrored_actions: np.ndarray,
    action_valid: np.ndarray,
    action_frame_indices: np.ndarray,
    original_baselines: np.ndarray,
    mirrored_baselines: np.ndarray,
    baseline_valid: np.ndarray,
    fps: float,
    action_names: tuple[str, ...],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    tuple[str, ...],
]:
    if type(action_names) is not tuple or not action_names:
        raise ValueError("action_names must be a nonempty exact tuple")
    try:
        canonical = tuple(_ACTION_NAME_MAP[name] for name in action_names)
    except (KeyError, TypeError) as exc:
        raise ValueError("an action name is outside the frozen clinical ontology") from exc
    if len(set(canonical)) != len(canonical):
        raise ValueError("canonical action identities must be unique in one bag")
    if (
        type(original_actions) is not np.ndarray
        or original_actions.dtype != np.dtype(np.float64)
        or original_actions.ndim != 4
        or original_actions.shape[0] != len(action_names)
        or original_actions.shape[2:] != (DENSE_POINT_COUNT, 3)
        or type(mirrored_actions) is not np.ndarray
        or mirrored_actions.dtype != np.dtype(np.float64)
        or mirrored_actions.shape != original_actions.shape
        or type(action_valid) is not np.ndarray
        or action_valid.dtype != np.dtype(bool)
        or action_valid.shape != original_actions.shape[:2]
        or type(action_frame_indices) is not np.ndarray
        or action_frame_indices.dtype != np.dtype(np.int64)
        or action_frame_indices.shape != action_valid.shape
        or np.any(action_frame_indices < 0)
        or np.any(np.diff(action_frame_indices, axis=1) < 0)
        or np.any(action_valid.sum(axis=1) < 6)
        or type(original_baselines) is not np.ndarray
        or original_baselines.dtype != np.dtype(np.float64)
        or original_baselines.ndim != 4
        or original_baselines.shape[0] != len(action_names)
        or original_baselines.shape[2:] != (DENSE_POINT_COUNT, 3)
        or type(mirrored_baselines) is not np.ndarray
        or mirrored_baselines.dtype != np.dtype(np.float64)
        or mirrored_baselines.shape != original_baselines.shape
        or type(baseline_valid) is not np.ndarray
        or baseline_valid.dtype != np.dtype(bool)
        or baseline_valid.shape != original_baselines.shape[:2]
        or np.any(baseline_valid.sum(axis=1) < 4)
        or not np.isfinite(float(fps))
        or isinstance(fps, (bool, np.bool_))
        or float(fps) <= 0.0
    ):
        raise ValueError("dense action evidence differs from the frozen contract")
    if not np.isfinite(original_actions[action_valid]).all():
        raise ValueError("valid original dense rows must be finite")
    if not np.isfinite(mirrored_actions[action_valid]).all():
        raise ValueError("valid mirrored dense rows must be finite")
    if not np.isfinite(original_baselines[baseline_valid]).all():
        raise ValueError("valid original baseline rows must be finite")
    if not np.isfinite(mirrored_baselines[baseline_valid]).all():
        raise ValueError("valid mirrored baseline rows must be finite")
    return (
        original_actions,
        mirrored_actions,
        action_valid,
        action_frame_indices,
        original_baselines,
        mirrored_baselines,
        baseline_valid,
        float(fps),
        canonical,
    )


def _collapse_and_interpolate(
    meshes: np.ndarray,
    indices: np.ndarray,
    fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(indices)
    if unique.size < 2:
        raise ValueError("an action requires at least two distinct source times")
    collapsed = np.stack([
        np.median(meshes[indices == index], axis=0) for index in unique
    ])
    source_times = unique.astype(np.float64) / fps
    target_times = np.linspace(
        float(source_times[0]), float(source_times[-1]), ACTION_TOKEN_FRAMES,
        dtype=np.float64,
    )
    flat = collapsed.reshape(collapsed.shape[0], -1)
    interpolated = np.empty((ACTION_TOKEN_FRAMES, flat.shape[1]), dtype=np.float64)
    for coordinate in range(flat.shape[1]):
        interpolated[:, coordinate] = np.interp(
            target_times, source_times, flat[:, coordinate]
        )
    result = interpolated.reshape(ACTION_TOKEN_FRAMES, DENSE_POINT_COUNT, 3)
    if not np.isfinite(result).all():
        raise ValueError("dense interpolation produced nonfinite values")
    return result.astype(np.float32), target_times


def _clinical_token_from_dense(
    meshes: np.ndarray,
    timestamps: np.ndarray,
) -> np.ndarray:
    clinical = np.stack([
        clinical_landmark_features(mesh, 1.0, 1.0) for mesh in meshes
    ]).astype(np.float32, copy=False)
    if clinical.shape != (ACTION_TOKEN_FRAMES, len(CLINICAL_LANDMARK_NAMES)):
        raise AssertionError("clinical23 dense projection dimension drifted")
    features = np.zeros((1, ACTION_TOKEN_FRAMES, 95), dtype=np.float32)
    features[0, :, 72:] = clinical
    valid = np.ones((1, ACTION_TOKEN_FRAMES), dtype=bool)
    sample_positions = np.arange(ACTION_TOKEN_FRAMES, dtype=np.int64)[None, :]
    return candidate_feature_vector(
        LANDMARK_MI_110D,
        features,
        valid,
        timestamps[None, :],
        sample_positions,
    )


def neutral_clinical_token_pair(
    original_baseline: np.ndarray,
    mirrored_baseline: np.ndarray,
    baseline_valid: np.ndarray,
    baseline_frame_indices: np.ndarray,
    *,
    fps: float,
    action_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the repeated absolute-clinical neutral baseline for attribution.

    The dense response baseline is exactly zero elsewhere.  This helper only
    supplies the absolute 110D clinical geometry measured during the
    authenticated neutral-repose hold, separately for the original and true
    flip-and-redetect views.
    """
    original = np.asarray(original_baseline)
    mirrored = np.asarray(mirrored_baseline)
    valid = np.asarray(baseline_valid)
    indices = np.asarray(baseline_frame_indices)
    if (
        original.shape != (ACTION_TOKEN_FRAMES, DENSE_POINT_COUNT, 3)
        or mirrored.shape != original.shape
        or original.dtype not in {np.dtype(np.float32), np.dtype(np.float64)}
        or mirrored.dtype not in {np.dtype(np.float32), np.dtype(np.float64)}
        or valid.shape != (ACTION_TOKEN_FRAMES,)
        or valid.dtype != np.dtype(bool)
        or indices.shape != (ACTION_TOKEN_FRAMES,)
        or indices.dtype != np.dtype(np.int64)
        or int(valid.sum()) < 26
        or isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not np.isfinite(float(fps))
        or float(fps) <= 0.0
        or type(action_count) is not int
        or action_count < 1
    ):
        raise ValueError("neutral clinical baseline differs from the closed contract")
    if (
        not np.isfinite(original[valid]).all()
        or not np.isfinite(mirrored[valid]).all()
        or np.any(indices[valid] < 0)
    ):
        raise ValueError("neutral clinical baseline contains invalid supported values")
    original_dense, target_times = _collapse_and_interpolate(
        original[valid], indices[valid], float(fps)
    )
    mirrored_dense, mirrored_times = _collapse_and_interpolate(
        mirrored[valid], indices[valid], float(fps)
    )
    if not np.array_equal(target_times, mirrored_times):
        raise AssertionError("paired neutral views changed their time grid")
    original_token = _clinical_token_from_dense(original_dense, target_times)
    mirrored_token = _clinical_token_from_dense(mirrored_dense, target_times)
    return (
        _immutable(np.repeat(original_token[None, :], action_count, axis=0).astype(
            np.float32, copy=False
        )),
        _immutable(np.repeat(mirrored_token[None, :], action_count, axis=0).astype(
            np.float32, copy=False
        )),
    )


def dense_action_token_bag(
    original_actions: np.ndarray,
    mirrored_actions: np.ndarray,
    action_valid: np.ndarray,
    action_frame_indices: np.ndarray,
    original_baselines: np.ndarray,
    mirrored_baselines: np.ndarray,
    baseline_valid: np.ndarray,
    *,
    fps: float,
    action_names: tuple[str, ...],
) -> ClinicalActionBag:
    """Build dense and clinical tokens from authenticated scripted actions."""
    (
        original,
        mirrored,
        valid,
        indices,
        original_rest,
        mirrored_rest,
        rest_valid,
        fps,
        canonical,
    ) = _validated_dense_inputs(
        original_actions,
        mirrored_actions,
        action_valid,
        action_frame_indices,
        original_baselines,
        mirrored_baselines,
        baseline_valid,
        fps,
        action_names,
    )
    dense_original: list[np.ndarray] = []
    dense_mirrored: list[np.ndarray] = []
    dense_times: list[np.ndarray] = []
    clinical_original: list[np.ndarray] = []
    clinical_mirrored: list[np.ndarray] = []
    for action in range(len(canonical)):
        selected = valid[action]
        original_dense, target_times = _collapse_and_interpolate(
            original[action, selected], indices[action, selected], fps
        )
        mirrored_dense, mirrored_times = _collapse_and_interpolate(
            mirrored[action, selected], indices[action, selected], fps
        )
        if not np.array_equal(target_times, mirrored_times):
            raise AssertionError("paired dense views changed their time grid")
        original_baseline = np.median(
            original_rest[action, rest_valid[action]], axis=0
        )
        mirrored_baseline = np.median(
            mirrored_rest[action, rest_valid[action]], axis=0
        )
        original_response = original_dense.astype(np.float64) - original_baseline
        mirrored_response = mirrored_dense.astype(np.float64) - mirrored_baseline
        dense_original.append(original_response.astype(np.float32))
        dense_mirrored.append(mirrored_response.astype(np.float32))
        dense_times.append(target_times)
        # The clinical branch deliberately keeps absolute normalized geometry;
        # only the full-mesh branch is converted to action-minus-rest response.
        clinical_original.append(_clinical_token_from_dense(
            original_dense, target_times
        ))
        clinical_mirrored.append(_clinical_token_from_dense(
            mirrored_dense, target_times
        ))
    action_count = len(canonical)
    return _build_bag(
        clinical_original=np.stack(clinical_original).astype(np.float64, copy=False),
        clinical_mirrored=np.stack(clinical_mirrored).astype(np.float64, copy=False),
        dense_original=np.stack(dense_original).astype(np.float32, copy=False),
        dense_mirrored=np.stack(dense_mirrored).astype(np.float32, copy=False),
        dense_valid_mask=np.ones((action_count, ACTION_TOKEN_FRAMES), dtype=bool),
        dense_available=np.ones(action_count, dtype=bool),
        dense_timestamps=np.stack(dense_times).astype(np.float64, copy=False),
        action_names=canonical,
    )


__all__ = [
    "ACTION_TOKEN_FRAMES",
    "CLINICAL_TOKEN_DIM",
    "ClinicalActionBag",
    "dense_action_token_bag",
    "neutral_clinical_token_pair",
    "palsynet_window_token_bag",
]
