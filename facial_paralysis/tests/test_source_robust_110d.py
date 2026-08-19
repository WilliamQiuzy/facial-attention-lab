"""Frozen feature-view contracts for Source-Robust Landmark 110D v1."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
from src.preprocessing.source_robust_110d import (  # noqa: E402
    ASYMMETRY_MEDIAN_CHANNEL_NAMES,
    CANDIDATE_DIMENSIONS,
    CANDIDATE_ORDER,
    candidate_feature_names,
    source_robust_feature_views,
)
from src.preprocessing.trajectory_features import (  # noqa: E402
    SUMMARY_STAT_NAMES,
    trajectory_feature_names,
)


def test_candidate_registry_and_feature_names_are_exact(c: Check):
    c.eq(CANDIDATE_ORDER, (
        "landmark_mi_110d",
        "within_video_dynamics_87d",
        "asymmetry_dynamics_93d",
    ))
    c.eq(CANDIDATE_DIMENSIONS, {
        "landmark_mi_110d": 110,
        "within_video_dynamics_87d": 87,
        "asymmetry_dynamics_93d": 93,
    })
    c.eq(ASYMMETRY_MEDIAN_CHANNEL_NAMES, (
        "fissure_h_absdiff", "fissure_w_absdiff", "eye_area_absdiff",
        "brow_h_absdiff", "corner_y_absdiff", "commissure_x_absdiff",
    ))
    for candidate in CANDIDATE_ORDER:
        names = candidate_feature_names(candidate)
        c.eq(len(names), CANDIDATE_DIMENSIONS[candidate])
        c.eq(len(set(names)), len(names))


def test_87d_drops_every_median_and_93d_restores_only_asymmetry_medians(c: Check):
    base_names = trajectory_feature_names("landmark")
    base = np.arange(110, dtype=np.float64)
    views = source_robust_feature_views(base)
    c.eq(tuple(views), CANDIDATE_ORDER)
    c.true(np.array_equal(views[CANDIDATE_ORDER[0]], base))

    dynamic_names = candidate_feature_names(CANDIDATE_ORDER[1])
    restored_names = candidate_feature_names(CANDIDATE_ORDER[2])
    median_suffix = f"__{SUMMARY_STAT_NAMES[0]}"
    c.eq(sum(name.endswith(median_suffix) for name in dynamic_names), 0)
    c.eq(tuple(
        name.removesuffix(median_suffix)
        for name in restored_names if name.endswith(median_suffix)
    ), ASYMMETRY_MEDIAN_CHANNEL_NAMES)

    by_name = {name: base[index] for index, name in enumerate(base_names)}
    c.true(np.array_equal(
        views[CANDIDATE_ORDER[1]],
        np.asarray([by_name[name] for name in dynamic_names]),
    ))
    c.true(np.array_equal(
        views[CANDIDATE_ORDER[2]],
        np.asarray([by_name[name] for name in restored_names]),
    ))


def test_feature_views_support_batches_and_fail_closed(c: Check):
    matrix = np.arange(330, dtype=np.float64).reshape(3, 110)
    views = source_robust_feature_views(matrix)
    for candidate in CANDIDATE_ORDER:
        c.eq(views[candidate].shape, (3, CANDIDATE_DIMENSIONS[candidate]))
        c.true(bool(np.isfinite(views[candidate]).all()))
    malformed = matrix.copy()
    malformed[0, 0] = np.nan
    c.raises(lambda: source_robust_feature_views(malformed), ValueError)
    c.raises(lambda: source_robust_feature_views(matrix[:, :-1]), ValueError)
    c.raises(lambda: candidate_feature_names("unknown"), ValueError)


if __name__ == "__main__":
    run_all("test_source_robust_110d", dict(globals()))
