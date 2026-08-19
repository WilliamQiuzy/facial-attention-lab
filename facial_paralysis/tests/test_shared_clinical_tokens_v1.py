from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all

from src.preprocessing.generalization_110d import (
    LANDMARK_MI_110D,
    candidate_feature_names,
)
from src.preprocessing.shared_clinical_tokens_v1 import (
    ACTION_TOKEN_FRAMES,
    CLINICAL_TOKEN_DIM,
    dense_action_token_bag,
    palsynet_window_token_bag,
)


def _clinical_windows() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = np.zeros((4, 32, 95), dtype=np.float32)
    phase = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    for window in range(4):
        features[window, :, 72:] = (
            np.arange(23, dtype=np.float32)[None, :] * 0.01
            + phase[:, None] * (0.02 + 0.01 * window)
        )
    valid = np.ones((4, 32), dtype=bool)
    timestamps = np.stack([
        window * 4.0 + np.arange(32, dtype=np.float64) / 30.0
        for window in range(4)
    ])
    indices = np.stack([
        window * 100 + np.arange(32, dtype=np.int64)
        for window in range(4)
    ])
    return features, valid, timestamps, indices


def _face() -> np.ndarray:
    points = np.full((478, 3), (0.5, 0.5, 0.0), dtype=np.float64)
    right_eye = {
        33: (0.30, 0.40), 133: (0.40, 0.40),
        159: (0.35, 0.38), 158: (0.37, 0.38), 160: (0.33, 0.38),
        145: (0.35, 0.42), 144: (0.33, 0.42), 153: (0.37, 0.42),
    }
    left_eye = {
        263: (0.70, 0.40), 362: (0.60, 0.40),
        386: (0.65, 0.38), 385: (0.63, 0.38), 387: (0.67, 0.38),
        374: (0.65, 0.42), 380: (0.67, 0.42), 373: (0.63, 0.42),
    }
    for index, xy in {**right_eye, **left_eye}.items():
        points[index, :2] = xy
    for index, x in zip((70, 63, 105, 66, 107), np.linspace(0.30, 0.40, 5)):
        points[index, :2] = (x, 0.30)
    for index, x in zip((300, 293, 334, 296, 336), np.linspace(0.70, 0.60, 5)):
        points[index, :2] = (x, 0.30)
    midline = (168, 6, 197, 195, 5, 4, 1, 19, 2, 164, 0, 17, 152, 10)
    for offset, index in enumerate(midline):
        points[index, :2] = (0.5, 0.25 + 0.04 * offset)
    points[61, :2] = (0.40, 0.70)
    points[291, :2] = (0.60, 0.70)
    points[13, :2] = (0.50, 0.68)
    points[14, :2] = (0.50, 0.72)
    return points


def _dense_actions() -> tuple[np.ndarray, ...]:
    action_count, sample_count = 3, 8
    original = np.empty((action_count, sample_count, 478, 3), dtype=np.float64)
    mirrored = np.empty_like(original)
    for action in range(action_count):
        for frame in range(sample_count):
            mesh = _face()
            mesh[61, 1] += 0.002 * (action + 1) * frame
            mesh[291, 1] += 0.001 * (action + 1) * frame
            original[action, frame] = mesh
            flipped = mesh.copy()
            flipped[:, 0] = 1.0 - flipped[:, 0]
            mirrored[action, frame] = flipped
    valid = np.ones((action_count, sample_count), dtype=bool)
    indices = np.stack([
        action * 100 + np.arange(sample_count, dtype=np.int64) * 2
        for action in range(action_count)
    ])
    original_baseline = np.repeat(original[:, :1], 4, axis=1)
    mirrored_baseline = np.repeat(mirrored[:, :1], 4, axis=1)
    baseline_valid = np.ones((action_count, 4), dtype=bool)
    return (
        original, mirrored, valid, indices,
        original_baseline, mirrored_baseline, baseline_valid,
    )


def test_palsynet_adapter_produces_four_clinical_only_tokens(c):
    bag = palsynet_window_token_bag(*_clinical_windows())
    c.eq(bag.clinical_original.shape, (4, CLINICAL_TOKEN_DIM))
    c.eq(bag.clinical_mirrored.shape, bag.clinical_original.shape)
    c.eq(bag.dense_original.shape, (4, ACTION_TOKEN_FRAMES, 478, 3))
    c.true(not bag.dense_available.any())
    c.eq(bag.action_names, (
        "FREE_EARLY", "FREE_MID_EARLY", "FREE_MID_LATE", "FREE_LATE"
    ))
    c.eq(CLINICAL_TOKEN_DIM, len(candidate_feature_names(LANDMARK_MI_110D)))
    c.true(np.isfinite(bag.clinical_original).all())
    c.true(not bag.clinical_original.flags.writeable)


def test_dense_adapter_keeps_full_mesh_and_builds_same_clinical_schema(c):
    original, mirrored, valid, indices, base, mirror_base, base_valid = _dense_actions()
    bag = dense_action_token_bag(
        original,
        mirrored,
        valid,
        indices,
        base,
        mirror_base,
        base_valid,
        fps=30.0,
        action_names=("NSM_KISS", "NSM_OPEN", "NSM_SPREAD"),
    )
    c.eq(bag.clinical_original.shape, (3, CLINICAL_TOKEN_DIM))
    c.eq(bag.dense_original.shape, (3, ACTION_TOKEN_FRAMES, 478, 3))
    c.eq(bag.dense_valid_mask.shape, (3, ACTION_TOKEN_FRAMES))
    c.true(bag.dense_valid_mask.all() and bag.dense_available.all())
    c.eq(bag.action_names, ("LIP_PUCKER", "MOUTH_OPEN", "SMILE_SPREAD"))
    c.true(np.isfinite(bag.dense_original).all())
    c.true(np.any(bag.clinical_original != bag.clinical_mirrored))
    c.true(not bag.dense_original.flags.writeable)


def test_dense_adapter_interpolates_nonadjacent_samples_on_real_time(c):
    original, mirrored, valid, indices, base, mirror_base, base_valid = _dense_actions()
    bag = dense_action_token_bag(
        original,
        mirrored,
        valid,
        indices,
        base,
        mirror_base,
        base_valid,
        fps=10.0,
        action_names=("NSM_KISS", "NSM_OPEN", "NSM_SPREAD"),
    )
    c.true(np.allclose(bag.dense_timestamps[:, 0], indices[:, 0] / 10.0))
    c.true(np.allclose(bag.dense_timestamps[:, -1], indices[:, -1] / 10.0))
    c.true(np.all(np.diff(bag.dense_timestamps, axis=1) > 0.0))


def test_unknown_action_and_insufficient_support_fail_closed(c):
    original, mirrored, valid, indices, base, mirror_base, base_valid = _dense_actions()
    c.raises(
        lambda: dense_action_token_bag(
            original, mirrored, valid, indices, base, mirror_base, base_valid,
            fps=30.0,
            action_names=("NSM_KISS", "UNKNOWN", "NSM_SPREAD"),
        ),
        ValueError,
    )
    valid[0, :3] = False
    c.raises(
        lambda: dense_action_token_bag(
            original, mirrored, valid, indices, base, mirror_base, base_valid,
            fps=30.0,
            action_names=("NSM_KISS", "NSM_OPEN", "NSM_SPREAD"),
        ),
        ValueError,
    )


def test_dense_stream_is_baseline_centered_not_absolute_face_shape(c):
    original, mirrored, valid, indices, base, mirror_base, base_valid = _dense_actions()
    first = dense_action_token_bag(
        original, mirrored, valid, indices, base, mirror_base, base_valid,
        fps=30.0, action_names=("NSM_KISS", "NSM_OPEN", "NSM_SPREAD"),
    )
    shifted_original = original.copy(); shifted_original[..., 2] += 0.25
    shifted_mirrored = mirrored.copy(); shifted_mirrored[..., 2] += 0.25
    shifted_base = base.copy(); shifted_base[..., 2] += 0.25
    shifted_mirror_base = mirror_base.copy(); shifted_mirror_base[..., 2] += 0.25
    shifted = dense_action_token_bag(
        shifted_original, shifted_mirrored, valid, indices,
        shifted_base, shifted_mirror_base, base_valid,
        fps=30.0, action_names=("NSM_KISS", "NSM_OPEN", "NSM_SPREAD"),
    )
    c.true(np.allclose(first.dense_original, shifted.dense_original, atol=1e-7))
    c.true(np.allclose(first.dense_original[:, 0], 0.0, atol=1e-7))
    c.true(np.any(np.abs(first.dense_original[:, -1]) > 0.0))


def test_token_bag_contains_no_source_or_dataset_identity(c):
    bag = palsynet_window_token_bag(*_clinical_windows())
    fields = set(bag.__dataclass_fields__)
    c.true("source" not in fields and "dataset" not in fields and "group_id" not in fields)


if __name__ == "__main__":
    run_all("test_shared_clinical_tokens_v1", dict(globals()))
