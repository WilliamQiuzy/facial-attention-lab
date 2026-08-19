from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all
from src.evaluation.dense_action_router_v6 import (
    DENSE_ARCHITECTURES,
    DenseRouterDataset,
    run_nested_dense_router,
)


def _dataset() -> DenseRouterDataset:
    rng = np.random.default_rng(20260817)
    labels = np.asarray(([0, 1] * 36), dtype=np.int64)
    original = rng.normal(scale=0.35, size=(72, 48)).astype(np.float64)
    mirrored = original + rng.normal(scale=0.03, size=original.shape)
    for start in (0, 16, 32):
        original[:, start : start + 4] += labels[:, None] * 1.5
        mirrored[:, start : start + 4] += labels[:, None] * 1.5
    return DenseRouterDataset(
        original=original,
        mirrored=mirrored.astype(np.float64),
        labels=labels,
        group_ids=tuple(f"group_{index:03d}" for index in range(72)),
        action_slices=(("A", 0, 16), ("B", 16, 32), ("C", 32, 48)),
        baseline_probability=np.full(72, 0.5, dtype=np.float64),
    )


def _registry():
    return (
        {
            "name": "fixture_sparse",
            "architecture": "dense_sparse_logistic",
            "statistic_family": "all",
            "view": "augment",
            "top_k": 12,
            "c": 0.1,
            "fusion_weight": 1.0,
        },
        {
            "name": "fixture_experts",
            "architecture": "dense_action_experts",
            "statistic_family": "all",
            "view": "augment",
            "top_k": 4,
            "c": 0.1,
            "fusion_weight": 1.0,
        },
    )


def test_registry_covers_the_four_frozen_architectures(c):
    c.eq(
        DENSE_ARCHITECTURES,
        (
            "dense_sparse_logistic",
            "dense_action_experts",
            "dense_rbf",
            "dense_ucr4_fusion",
        ),
    )


def test_nested_oof_is_deterministic_accurate_and_group_disjoint(c):
    first = run_nested_dense_router(
        _dataset(), registry=_registry(), outer_folds=6, inner_folds=5
    )
    second = run_nested_dense_router(
        _dataset(), registry=_registry(), outer_folds=6, inner_folds=5
    )
    c.true(np.array_equal(first.probability, second.probability))
    c.true(np.array_equal(first.prediction, second.prediction))
    c.true(first.metrics["accuracy"] >= 0.90)
    c.true(first.metrics["balanced_accuracy"] >= 0.90)
    c.eq(first.audit["outer_held_group_overlap"], 0)
    c.eq(first.audit["inner_held_group_overlap"], 0)
    c.eq(first.audit["protected_reads"], 0)


def test_each_outer_fit_uses_two_half_weight_views_and_train_only_ranking(c):
    result = run_nested_dense_router(
        _dataset(), registry=_registry(), outer_folds=6, inner_folds=5
    )
    c.eq(len(result.outer_folds), 6)
    for fold in result.outer_folds:
        c.eq(fold["augmented_training_rows"], 2 * fold["training_groups"])
        c.eq(fold["training_weight_sum"], float(fold["training_groups"]))
        c.eq(fold["ranking_scope"], "outer_training_groups_only")


def test_dataset_and_registry_fail_closed(c):
    dataset = _dataset()
    c.raises(
        lambda: DenseRouterDataset(
            original=dataset.original,
            mirrored=dataset.mirrored,
            labels=dataset.labels,
            group_ids=(dataset.group_ids[0],) * len(dataset.group_ids),
            action_slices=dataset.action_slices,
            baseline_probability=dataset.baseline_probability,
        ),
        ValueError,
    )
    bad = (dict(_registry()[0], top_k=10_000),)
    c.raises(lambda: run_nested_dense_router(dataset, registry=bad), ValueError)


if __name__ == "__main__":
    run_all("test_dense_action_router_v6", dict(globals()))
