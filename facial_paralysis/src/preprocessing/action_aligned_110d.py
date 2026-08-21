"""Deterministic, label-free action-window proposals for Landmark 110D.

MediaPipe blendshapes are used only to locate movement peaks. The classifier
input remains the exact 110-dimensional clinical landmark summary.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..datasets.dynamic_landmark import DYNAMIC_FEATURE_NAMES


ACTION_SLOT_ORDER = (
    "eyebrow_rise",
    "gentle_eye_closure",
    "tight_eye_squeeze",
    "relaxed_smile",
    "lip_pucker",
    "lower_teeth_show",
    "reanimated_smile",
)
WINDOW_FRAMES = 32
WINDOW_PEAK_OFFSET = WINDOW_FRAMES // 2
MIN_DISTINCT_PEAK_SECONDS = 2.0
REFERENCE_FPS = 30.0
_MIN_ACTIVITY_RANGE = 1e-6
_FEATURE_INDEX = {name: index for index, name in enumerate(DYNAMIC_FEATURE_NAMES)}
_SIGNATURE_NAMES = {
    "brow": ("browInnerUp", "browOuterUpLeft", "browOuterUpRight"),
    "eye_close": ("eyeBlinkLeft", "eyeBlinkRight"),
    "eye_squeeze": (
        "cheekSquintLeft", "cheekSquintRight", "eyeSquintLeft", "eyeSquintRight",
    ),
    "smile": ("mouthSmileLeft", "mouthSmileRight"),
    "pucker": ("mouthPucker",),
    "lower_teeth": ("jawOpen", "mouthLowerDownLeft", "mouthLowerDownRight"),
}
_SIGNATURE_INDICES = {
    name: tuple(_FEATURE_INDEX[column] for column in columns)
    for name, columns in _SIGNATURE_NAMES.items()
}


def _numpy_mirror_contract() -> tuple[np.ndarray, np.ndarray]:
    names = tuple(DYNAMIC_FEATURE_NAMES)
    indices = np.arange(len(names), dtype=np.int64)
    signs = np.ones(len(names), dtype=np.float32)
    by_name = {name: index for index, name in enumerate(names)}
    for index, name in enumerate(names[:52]):
        partner = None
        if name.endswith("Left"):
            partner = name[:-4] + "Right"
        elif name.endswith("Right"):
            partner = name[:-5] + "Left"
        if partner is not None:
            if partner not in by_name or by_name[partner] >= 52:
                raise RuntimeError(f"missing mirror partner for {name!r}")
            indices[index] = by_name[partner]
    for index, name in enumerate(names[52:72], start=52):
        if not name.startswith("delta_left_minus_right_"):
            raise RuntimeError("unexpected blendshape asymmetry column order")
        signs[index] = -1.0
    for first, second in (
        ("fissure_h_mesh33", "fissure_h_mesh263"),
        ("fissure_w_mesh33", "fissure_w_mesh263"),
        ("eye_area_mesh33", "eye_area_mesh263"),
        ("brow_h_mesh33", "brow_h_mesh263"),
        ("corner_y_mesh61", "corner_y_mesh291"),
        ("corner_x_mesh61", "corner_x_mesh291"),
    ):
        left, right = by_name[first], by_name[second]
        indices[left], indices[right] = right, left
    for name in (
        "fissure_h_mesh33_minus_mesh263",
        "brow_h_mesh33_minus_mesh263",
        "corner_y_mesh61_minus_mesh291",
    ):
        signs[by_name[name]] = -1.0
    if any(indices[indices[index]] != index or signs[index] != signs[indices[index]]
           for index in range(len(names))):
        raise RuntimeError("NumPy mirror schema is not an involution")
    return indices, signs


_MIRROR_INDICES, _MIRROR_SIGNS = _numpy_mirror_contract()


def _proposal_arrays(
    features: np.ndarray,
    valid_mask: np.ndarray,
    source_frame_indices: np.ndarray,
    source_frame_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    values = np.asarray(features)
    valid = np.asarray(valid_mask)
    indices = np.asarray(source_frame_indices)
    if values.ndim != 2 or values.shape[0] < 7 or values.shape[1] != 95:
        raise ValueError("proposal features must have shape (T, 95) with T at least 7")
    if values.dtype.kind != "f" or not np.isfinite(values).all():
        raise ValueError("proposal features must be finite floating values")
    if valid.shape != (values.shape[0],) or valid.dtype != np.dtype(bool):
        raise ValueError("proposal valid_mask must be a bool vector aligned to features")
    if indices.shape != valid.shape or indices.dtype.kind not in {"i", "u"}:
        raise ValueError("proposal source_frame_indices must be an aligned integer vector")
    if not np.all(indices[1:] > indices[:-1]):
        raise ValueError("proposal source frame indices must increase strictly")
    if isinstance(source_frame_count, (bool, np.bool_)) or not isinstance(
        source_frame_count, (int, np.integer)
    ):
        raise ValueError("source_frame_count must be an integer")
    count = int(source_frame_count)
    if count < WINDOW_FRAMES or int(indices[-1]) >= count:
        raise ValueError("proposal indices must lie inside a video of at least 32 frames")
    if int(np.sum(valid)) < 7:
        raise ValueError("at least seven valid proposal samples are required")
    return values, valid, indices.astype(np.int64, copy=False), count


def _activation(
    features: np.ndarray,
    valid: np.ndarray,
    signature: str,
) -> np.ndarray:
    curve = features[:, _SIGNATURE_INDICES[signature]].mean(axis=1).astype(np.float64)
    curve[~valid] = -np.inf
    finite = curve[valid]
    if finite.size == 0 or float(np.max(finite) - np.min(finite)) <= _MIN_ACTIVITY_RANGE:
        raise ValueError(f"no measurable {signature} activation")
    return curve


def _strongest_peak(curve: np.ndarray) -> int:
    return int(np.argmax(curve))


def _two_distinct_peaks(
    curve: np.ndarray,
    source_indices: np.ndarray,
    minimum_source_frames: int,
) -> tuple[int, int]:
    candidates = sorted(
        np.flatnonzero(np.isfinite(curve)).tolist(),
        key=lambda index: (-float(curve[index]), int(source_indices[index])),
    )
    selected: list[int] = []
    for index in candidates:
        if all(
            abs(int(source_indices[index]) - int(source_indices[other]))
            >= minimum_source_frames
            for other in selected
        ):
            selected.append(int(index))
        if len(selected) == 2:
            break
    if len(selected) != 2:
        raise ValueError("two time-separated smile peaks are required")
    selected.sort(key=lambda index: int(source_indices[index]))
    return selected[0], selected[1]


def _window_offsets(source_fps: float) -> np.ndarray:
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("source_fps must be finite and positive")
    offsets = np.rint(
        np.arange(WINDOW_FRAMES, dtype=np.float64) * float(source_fps) / REFERENCE_FPS
    ).astype(np.int64)
    if not np.all(offsets[1:] > offsets[:-1]):
        raise ValueError("source_fps is too low for 32 distinct action samples")
    return offsets


def _window_start(
    peak_source_frame: int,
    source_frame_count: int,
    source_fps: float,
) -> int:
    offsets = _window_offsets(source_fps)
    peak_offset = int(round(WINDOW_PEAK_OFFSET * float(source_fps) / REFERENCE_FPS))
    return int(np.clip(
        int(peak_source_frame) - peak_offset,
        0,
        source_frame_count - 1 - int(offsets[-1]),
    ))


def action_window_source_indices(
    starts: Sequence[int],
    *,
    source_fps: float,
    source_frame_count: int,
) -> np.ndarray:
    """Map seven starts to 32 samples at the reference 30 Hz time grid."""
    if len(starts) != len(ACTION_SLOT_ORDER):
        raise ValueError("exactly seven action starts are required")
    offsets = _window_offsets(source_fps)
    indices = np.stack([
        int(start) + offsets for start in starts
    ]).astype(np.int64, copy=False)
    if np.any(indices < 0) or np.any(indices >= int(source_frame_count)):
        raise ValueError("action window source indices escaped the video")
    return indices


def select_action_window_starts(
    proposal_features: np.ndarray,
    proposal_valid_mask: np.ndarray,
    proposal_source_frame_indices: np.ndarray,
    *,
    source_frame_count: int,
    source_fps: float = REFERENCE_FPS,
) -> tuple[int, ...]:
    """Return seven fixed action-slot window starts without using labels."""
    features, valid, indices, count = _proposal_arrays(
        proposal_features,
        proposal_valid_mask,
        proposal_source_frame_indices,
        source_frame_count,
    )
    curves = {
        signature: _activation(features, valid, signature)
        for signature in _SIGNATURE_INDICES
    }
    minimum_peak_frames = int(round(MIN_DISTINCT_PEAK_SECONDS * float(source_fps)))
    first_smile, second_smile = _two_distinct_peaks(
        curves["smile"], indices, minimum_peak_frames
    )
    peak_rows: Sequence[int] = (
        _strongest_peak(curves["brow"]),
        _strongest_peak(curves["eye_close"]),
        _strongest_peak(curves["eye_squeeze"]),
        first_smile,
        _strongest_peak(curves["pucker"]),
        _strongest_peak(curves["lower_teeth"]),
        second_smile,
    )
    starts = tuple(
        _window_start(int(indices[row]), count, source_fps) for row in peak_rows
    )
    if len(starts) != len(ACTION_SLOT_ORDER):
        raise AssertionError("action-slot selection drifted")
    return starts


def action_aligned_feature_vector(
    features: np.ndarray,
    valid_mask: np.ndarray,
    timestamps: np.ndarray,
    source_frame_indices: np.ndarray,
) -> np.ndarray:
    """Pool exactly seven action windows into the frozen Landmark 110D schema."""
    values = np.asarray(features)
    if values.shape != (len(ACTION_SLOT_ORDER), WINDOW_FRAMES, 95):
        raise ValueError("action-aligned features must have shape (7, 32, 95)")
    indices = np.asarray(source_frame_indices)
    expected_temporal_shape = (len(ACTION_SLOT_ORDER), WINDOW_FRAMES)
    if indices.shape != expected_temporal_shape or indices.dtype.kind not in {"i", "u"}:
        raise ValueError("action source_frame_indices must be integer with shape (7, 32)")
    if np.any(indices < 0) or not np.all(indices[:, 1:] > indices[:, :-1]):
        raise ValueError("action source_frame_indices must increase within each window")

    # The action cache samples every source at the same 30 Hz time grid. Native
    # 60 Hz videos therefore advance two source frames per sample. The frozen
    # trajectory summarizer uses integer adjacency only to prevent bridging a
    # detector gap; for this resampled representation that contract belongs to
    # sample positions, while the original timestamps retain the exact elapsed
    # seconds and valid_mask retains detector failures.
    sample_positions = np.broadcast_to(
        np.arange(WINDOW_FRAMES, dtype=np.int64), expected_temporal_shape
    ).copy()
    # Keep the label-free proposal/extraction environment independent of
    # scikit-learn; the classical 110D dependency is needed only at summary time.
    from .trajectory_features import LANDMARK_DIM, trajectory_feature_set

    result = trajectory_feature_set(
        "landmark", values, valid_mask, timestamps, sample_positions
    )
    if result.shape != (LANDMARK_DIM,):
        raise AssertionError("action-aligned Landmark dimension drifted")
    return result


def mirror_action_aligned_features(features: np.ndarray) -> np.ndarray:
    """Apply the frozen 95-channel mirror to all seven action windows."""
    values = np.asarray(features)
    if values.shape != (len(ACTION_SLOT_ORDER), WINDOW_FRAMES, 95):
        raise ValueError("action-aligned features must have shape (7, 32, 95)")
    return mirror_clinical23_features(values)


def mirror_clinical23_features(features: np.ndarray) -> np.ndarray:
    """Apply the exact frozen 95-channel mirror to any window-count tensor."""
    values = np.asarray(features)
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[1:] != (32, 95):
        raise ValueError("clinical23 features must have shape (n_windows, 32, 95)")
    if values.dtype.kind != "f" or not np.isfinite(values).all():
        raise ValueError("clinical23 features must be finite floating values")
    return (values[..., _MIRROR_INDICES] * _MIRROR_SIGNS).astype(
        values.dtype, copy=False
    )


__all__ = [
    "ACTION_SLOT_ORDER",
    "MIN_DISTINCT_PEAK_SECONDS",
    "REFERENCE_FPS",
    "WINDOW_FRAMES",
    "action_window_source_indices",
    "action_aligned_feature_vector",
    "mirror_action_aligned_features",
    "mirror_clinical23_features",
    "select_action_window_starts",
]
