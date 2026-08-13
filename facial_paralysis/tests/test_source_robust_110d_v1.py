"""Closed PalsyNet-only evaluation contracts for Source-Robust 110D v1."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
from scripts.run_source_robust_110d_v1 import _group_metrics  # noqa: E402
from src.evaluation.source_robust_110d_v1 import (  # noqa: E402
    ACQUISITION_NUISANCE_INDICES,
    FIXED_C,
    FIXED_THRESHOLD,
    build_acquisition_blocked_folds,
    choose_source_robust_candidate,
    run_candidate_oof,
)
from src.preprocessing.source_robust_110d import (  # noqa: E402
    CANDIDATE_DIMENSIONS,
    CANDIDATE_ORDER,
)


def _fold_fixture():
    groups = np.asarray([f"group-{index:02d}" for index in range(16)] + ["group-00"])
    labels = np.asarray([0] * 8 + [1] * 8 + [0], dtype=np.int64)
    nuisance = np.zeros((17, 9), dtype=np.float64)
    for index in range(16):
        nuisance[index] = np.linspace(index, index + 1.0, 9)
    nuisance[-1] = nuisance[0] + 0.1
    return labels, groups, nuisance


def test_acquisition_blocked_folds_are_deterministic_group_disjoint_and_balanced(c: Check):
    c.eq(ACQUISITION_NUISANCE_INDICES, (1, 2, 3, 5, 6, 7, 8))
    labels, groups, nuisance = _fold_fixture()
    first = build_acquisition_blocked_folds(labels, groups, nuisance)
    second = build_acquisition_blocked_folds(labels, groups, nuisance)
    c.true(np.array_equal(first, second))
    c.eq(first.shape, labels.shape)
    c.eq(set(first.tolist()), {0, 1, 2, 3})
    c.eq(int(first[0]), int(first[-1]), "one reviewed group cannot cross folds")
    for fold in range(4):
        c.eq(set(labels[first == fold].tolist()), {0, 1})


def test_fixed_oof_covers_every_row_once_and_has_no_protected_activity(c: Check):
    rng = np.random.default_rng(20260812)
    labels = np.asarray([0, 0, 1, 1] * 4, dtype=np.int64)
    groups = np.asarray([f"group-{index}" for index in range(labels.size)])
    folds = np.repeat(np.arange(4), 4)
    original = {
        candidate: rng.normal(size=(labels.size, dimension))
        for candidate, dimension in CANDIDATE_DIMENSIONS.items()
    }
    mirrored = {candidate: values.copy() for candidate, values in original.items()}
    result = run_candidate_oof(
        labels=labels, group_ids=groups, folds=folds,
        original=original, mirrored=mirrored,
    )
    c.eq(tuple(result.probabilities), CANDIDATE_ORDER)
    c.true(all(values.shape == labels.shape for values in result.probabilities.values()))
    c.true(all(np.isfinite(values).all() for values in result.probabilities.values()))
    c.eq(result.audit, {
        "development_scaler_fits": 12,
        "development_model_fits": 12,
        "development_predictions": 48,
        "protected_feature_reads": 0,
        "protected_fits": 0,
        "protected_predictions": 0,
    })
    c.eq(FIXED_C, 0.01)
    c.eq(FIXED_THRESHOLD, 0.5)


def test_promotion_requires_registered_noninferiority_and_blocked_improvement(c: Check):
    base_registered = {"auroc": 0.98, "balanced_accuracy": 0.95, "specificity": 1.0}
    base_blocked = {"auroc": 0.80, "balanced_accuracy": 0.75, "specificity": 0.8}
    metrics = {
        "registered": {
            CANDIDATE_ORDER[0]: base_registered,
            CANDIDATE_ORDER[1]: {"auroc": 0.97, "balanced_accuracy": 0.94, "specificity": 1.0},
            CANDIDATE_ORDER[2]: {"auroc": 0.975, "balanced_accuracy": 0.94, "specificity": 1.0},
        },
        "acquisition_blocked": {
            CANDIDATE_ORDER[0]: base_blocked,
            CANDIDATE_ORDER[1]: {"auroc": 0.86, "balanced_accuracy": 0.82, "specificity": 0.9},
            CANDIDATE_ORDER[2]: {"auroc": 0.84, "balanced_accuracy": 0.80, "specificity": 0.9},
        },
    }
    decision = choose_source_robust_candidate(metrics)
    c.eq(decision.locked_candidate, CANDIDATE_ORDER[1])
    c.true(decision.gates[CANDIDATE_ORDER[1]]["passed"])

    tied = {
        protocol: {candidate: dict(values) for candidate, values in candidates.items()}
        for protocol, candidates in metrics.items()
    }
    tied["acquisition_blocked"][CANDIDATE_ORDER[2]] = dict(
        tied["acquisition_blocked"][CANDIDATE_ORDER[1]]
    )
    c.eq(
        choose_source_robust_candidate(tied).locked_candidate,
        CANDIDATE_ORDER[0],
        "a blocked-metric tie retains the current 110D",
    )


def test_group_metrics_average_duplicate_recordings_before_scoring(c: Check):
    labels = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
    groups = np.asarray(["normal-a", "normal-a", "normal-b", "case-a", "case-b"])
    probabilities = np.asarray([0.1, 0.3, 0.4, 0.8, 0.9])
    metrics, grouped_labels, grouped_scores = _group_metrics(
        labels, groups, probabilities
    )
    c.eq(grouped_labels.tolist(), [1, 1, 0, 0])
    c.true(np.allclose(grouped_scores, (0.8, 0.9, 0.2, 0.4)))
    c.eq(metrics["balanced_accuracy"], 1.0)
    c.eq(metrics["sensitivity"], 1.0)
    c.eq(metrics["specificity"], 1.0)


if __name__ == "__main__":
    run_all("test_source_robust_110d_v1", dict(globals()))
