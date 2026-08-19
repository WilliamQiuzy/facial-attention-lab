from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all

from src.preprocessing.dense_bilateral_action_v1 import (
    BILATERAL_INTERACTION_STAT_NAMES,
    DENSE_POINT_COUNT,
    DENSE_STAT_NAMES,
    bilateral_interaction_feature_names,
    bilateral_interaction_feature_vector,
    dense_action_feature_names,
    dense_action_feature_views,
    normalize_dense_landmarks,
)


def _face() -> np.ndarray:
    points = np.zeros((DENSE_POINT_COUNT, 3), dtype=np.float64)
    points[:, 0] = np.linspace(0.25, 0.75, DENSE_POINT_COUNT)
    points[:, 1] = np.linspace(0.30, 0.70, DENSE_POINT_COUNT)
    points[:, 2] = np.linspace(-0.05, 0.05, DENSE_POINT_COUNT)
    points[33, :2] = (0.30, 0.40)
    points[263, :2] = (0.70, 0.40)
    return points


def _action_arrays():
    actions = np.stack([
        np.stack([_face() for _ in range(8)]),
        np.stack([_face() for _ in range(8)]),
    ])
    mirror = actions.copy()
    for frame in range(8):
        actions[0, frame, 61, 1] += 0.01 * frame
        actions[1, frame, 159, 1] -= 0.005 * frame
        mirror[0, frame, 291, 1] += 0.01 * frame
        mirror[1, frame, 386, 1] -= 0.005 * frame
    baseline = np.repeat(actions[:, :1], 4, axis=1)
    mirror_baseline = np.repeat(mirror[:, :1], 4, axis=1)
    action_valid = np.ones((2, 8), dtype=bool)
    baseline_valid = np.ones((2, 4), dtype=bool)
    return (
        actions, action_valid, baseline, baseline_valid,
        mirror, action_valid.copy(), mirror_baseline, baseline_valid.copy(),
    )


def test_normalization_is_translation_scale_and_roll_invariant(c):
    face = _face()
    normalized = normalize_dense_landmarks(face, image_width=1000, image_height=800)
    pixels = face.copy()
    pixels[:, 0] *= 1000
    pixels[:, 1] *= 800
    pixels[:, 2] *= 1000
    center = 0.5 * (pixels[33, :2] + pixels[263, :2])
    theta = 0.27
    rotation = np.asarray([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
    transformed = pixels.copy()
    transformed[:, :2] = (pixels[:, :2] - center) @ rotation.T * 1.7
    transformed[:, :2] += np.asarray((413.0, 287.0))
    transformed[:, 2] = pixels[:, 2] * 1.7
    transformed[:, 0] /= 1000
    transformed[:, 1] /= 800
    transformed[:, 2] /= 1000
    observed = normalize_dense_landmarks(
        transformed, image_width=1000, image_height=800
    )
    c.true(np.allclose(observed, normalized, atol=1e-12, rtol=0.0))
    c.eq(normalized.shape, (DENSE_POINT_COUNT, 3))
    c.true(not normalized.flags.writeable)


def test_normalization_rejects_bad_mesh_or_eye_scale(c):
    face = _face()
    for bad in (face[:-1], face.astype(np.float32), face.copy()):
        if bad.shape == face.shape and bad.dtype == face.dtype:
            bad[263, :2] = bad[33, :2]
        c.raises(lambda value=bad: normalize_dense_landmarks(value, 1000, 800), ValueError)
    bad = face.copy()
    bad[12, 0] = np.nan
    c.raises(lambda: normalize_dense_landmarks(bad, 1000, 800), ValueError)


def test_action_feature_contract_is_exact_finite_and_immutable(c):
    arrays = _action_arrays()
    first, second = dense_action_feature_views(
        *arrays, action_names=("SMILE", "EYE"),
    )
    names = dense_action_feature_names(("SMILE", "EYE"))
    expected = 2 * DENSE_POINT_COUNT * 3 * len(DENSE_STAT_NAMES)
    c.eq(first.shape, (expected,))
    c.eq(second.shape, first.shape)
    c.eq(len(names), expected)
    c.eq(len(set(names)), expected)
    c.true(np.isfinite(first).all())
    c.true(np.any(first != second))
    c.true(not first.flags.writeable and not second.flags.writeable)
    c.true(not np.shares_memory(first, second))


def test_action_statistics_preserve_response_and_gap_safe_steps(c):
    arrays = list(_action_arrays())
    action_valid = arrays[1]
    action_valid[0, 3] = False
    arrays[0][0, 3] = np.nan
    first, _ = dense_action_feature_views(
        *arrays, action_names=("SMILE", "EYE"),
    )
    names = dense_action_feature_names(("SMILE", "EYE"))
    median_name = "SMILE__mesh61__y__response_median"
    step_name = "SMILE__mesh61__y__response_max_abs_adjacent_step"
    c.true(first[names.index(median_name)] > 0.0)
    c.true(first[names.index(step_name)] > 0.0)


def test_action_support_schema_and_names_fail_closed(c):
    arrays = list(_action_arrays())
    arrays[1][0, :3] = False
    c.raises(
        lambda: dense_action_feature_views(
            *arrays, action_names=("SMILE", "EYE")
        ),
        ValueError,
    )
    arrays = list(_action_arrays())
    c.raises(
        lambda: dense_action_feature_views(
            *arrays, action_names=("SMILE", "SMILE")
        ),
        ValueError,
    )
    c.raises(lambda: dense_action_feature_names(("",)), ValueError)


def test_bilateral_interaction_is_swap_invariant_finite_and_immutable(c):
    arrays = _action_arrays()
    first = bilateral_interaction_feature_vector(
        *arrays, action_names=("SMILE", "EYE")
    )
    swapped = bilateral_interaction_feature_vector(
        arrays[4], arrays[5], arrays[6], arrays[7],
        arrays[0], arrays[1], arrays[2], arrays[3],
        action_names=("SMILE", "EYE"),
    )
    names = bilateral_interaction_feature_names(("SMILE", "EYE"))
    expected = 2 * DENSE_POINT_COUNT * 3 * len(BILATERAL_INTERACTION_STAT_NAMES)
    c.eq(first.shape, (expected,))
    c.eq(len(names), expected)
    c.true(np.array_equal(first, swapped))
    c.true(np.isfinite(first).all() and not first.flags.writeable)


def test_identical_views_have_zero_asymmetry_statistics(c):
    arrays = _action_arrays()
    value = bilateral_interaction_feature_vector(
        arrays[0], arrays[1], arrays[2], arrays[3],
        arrays[0], arrays[1], arrays[2], arrays[3],
        action_names=("SMILE", "EYE"),
    )
    names = bilateral_interaction_feature_names(("SMILE", "EYE"))
    selected = np.asarray([
        name.endswith("__response_asymmetry")
        or name.endswith("__range_asymmetry")
        or name.endswith("__peak_asymmetry")
        or name.endswith("__paired_difference_median")
        or name.endswith("__paired_difference_q90")
        for name in names
    ])
    c.true(np.all(value[selected] == 0.0))


if __name__ == "__main__":
    run_all("test_dense_bilateral_action_v1", dict(globals()))
