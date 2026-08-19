"""Participant-disjoint contracts for the NeuroFace ALS benchmark."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import src.evaluation.neuroface_als_benchmark_v1 as benchmark  # noqa: E402
from src.evaluation.neuroface_als_benchmark_v1 import (  # noqa: E402
    C_GRID,
    PAPER_ACCURACY,
    PAPER_AUROC,
    Candidate,
    build_public_report,
    evaluate_fixed_loso,
    evaluate_nested_loso,
    evaluate_nested_loso_with_threshold,
    evaluate_nested_balanced_logistic,
    evaluate_nested_shrinkage_lda,
    fit_fold_preprocessor,
    participant_stratified_bootstrap,
    recompute_binary_metrics,
    select_balanced_threshold,
    select_oof_candidate_with_threshold,
    select_paper_like_candidate,
)
from _testlib import Check, run_all  # noqa: E402


def _dataset(n: int = 10):
    labels = np.asarray([0, 1] * (n // 2), dtype=np.int64)
    groups = tuple(f"grp_{index:064x}" for index in range(n))
    signal = (labels * 2 - 1).astype(np.float64)
    representations = {
        "min": np.column_stack((signal, signal * 0.8, np.arange(n) % 3)),
        "mean": np.column_stack((signal * 0.3, np.arange(n) % 2, np.ones(n))),
    }
    return representations, labels, groups


def test_published_comparator_and_search_space_are_exact(c: Check):
    c.eq(PAPER_ACCURACY, 0.91, "published SPREAD minimum-AU accuracy is frozen")
    c.eq(PAPER_AUROC, 0.97, "published SPREAD minimum-AU AUROC is frozen")
    c.eq(C_GRID, (0.01, 0.1, 1.0, 10.0, 100.0),
         "paper C grid is exact")


def test_preprocessing_is_fit_only_on_training_rows(c: Check):
    train = np.asarray([
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 1.0],
        [3.0, 3.0, 1.0],
    ])
    held = np.asarray([[1000.0, -1000.0, 1.0]])
    fold = fit_fold_preprocessor(train, held, correlation_threshold=0.7)
    c.true(np.array_equal(fold.kept_indices, np.asarray([0, 2])),
           "correlated feature filtering uses stable first-retained order")
    c.true(np.allclose(fold.train[:, 0].mean(), 0.0),
           "training rows are standardized")
    c.true(abs(float(fold.test[0, 0])) > 100,
           "held row cannot influence the training mean or scale")


def test_fixed_loso_is_group_disjoint_and_exact(c: Check):
    reps, labels, groups = _dataset()
    candidate = Candidate("min", "l2", 0.1)
    result = evaluate_fixed_loso(reps, labels, groups, candidate)
    c.eq(result.n_participants, 10, "each participant contributes one OOF score")
    c.eq(result.candidate, candidate, "candidate is frozen across outer folds")
    c.eq(len(result.probabilities), 10, "there is exactly one probability per person")
    c.true(result.metrics["accuracy"] >= 0.9, "separable signal is recovered")
    duplicate_groups = groups[:-1] + (groups[0],)
    c.raises(lambda: evaluate_fixed_loso(
        reps, labels, duplicate_groups, candidate
    ), ValueError, "duplicate participant rows are rejected")


def test_paper_like_and_nested_search_are_separate(c: Check):
    reps, labels, groups = _dataset()
    paper_like = select_paper_like_candidate(reps, labels, groups)
    nested = evaluate_nested_loso(reps, labels, groups)
    c.true(paper_like.selection_protocol == "same_oof_candidate_search_descriptive",
           "paper-like candidate search is explicitly descriptive")
    c.true(nested.selection_protocol == "nested_participant_loso",
           "strict result selects candidates only inside each outer fold")
    c.eq(len(nested.outer_candidates), len(groups),
         "every held participant has an independently selected candidate")
    c.eq(len(nested.probabilities), len(groups),
         "nested evaluation emits one outer probability per participant")


def test_paper_search_reuses_candidate_independent_fold_preprocessing(c: Check):
    representations, labels, groups = _dataset()
    original = benchmark.fit_fold_preprocessor
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    benchmark.fit_fold_preprocessor = counted
    try:
        select_paper_like_candidate(
            {"paper_au": representations["min"]}, labels, groups
        )
    finally:
        benchmark.fit_fold_preprocessor = original
    c.eq(calls, len(groups),
         "each LOSO fold is preprocessed once rather than once per C and penalty")


def test_nested_threshold_is_selected_inside_each_outer_fold(c: Check):
    reps, labels, groups = _dataset()
    threshold = select_balanced_threshold(
        np.asarray([0, 0, 1, 1], dtype=np.int64),
        np.asarray([0.2, 0.4, 0.45, 0.8], dtype=np.float64),
    )
    c.eq(threshold, 0.45, "inner balanced threshold is deterministic")
    result = evaluate_nested_loso_with_threshold(reps, labels, groups)
    c.eq(len(result.outer_thresholds), len(groups),
         "each held participant gets an independently fitted inner threshold")
    c.true(all(0 <= value <= 1 for value in result.outer_thresholds),
           "all inner thresholds remain valid probabilities")
    c.eq(result.selection_protocol,
         "nested_participant_loso_with_inner_oof_threshold",
         "threshold calibration remains inside participant folds")
    reps, labels, groups = _dataset()
    selection = select_oof_candidate_with_threshold(reps, labels, groups)
    c.true(0 <= selection.threshold <= 1,
           "candidate and threshold are jointly selected from OOF predictions")
    c.eq(selection.selection_protocol,
         "same_oof_candidate_and_threshold_search_descriptive",
         "joint search is descriptive unless nested by the caller")


def test_participant_bootstrap_is_stratified_and_deterministic(c: Check):
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.1, 0.4, 0.6, 0.9], dtype=np.float64)
    first = participant_stratified_bootstrap(
        labels, probabilities, replicates=200, seed=20260814
    )
    second = participant_stratified_bootstrap(
        labels, probabilities, replicates=200, seed=20260814
    )
    c.eq(first, second, "bootstrap evidence is exactly reproducible")
    c.eq(first["valid_replicates"], 200,
         "class-stratified resampling preserves valid AUROC in every replicate")
    c.true(all(value[0] <= value[1] for value in first["interval_95"].values()),
           "all confidence intervals are ordered")


def test_shrinkage_lda_selects_representation_and_threshold_inside_outer_fold(c: Check):
    reps, labels, groups = _dataset()
    result = evaluate_nested_shrinkage_lda(reps, labels, groups)
    c.eq(len(result.outer_representations), len(groups),
         "each held person has an inner-selected LDA representation")
    c.eq(len(result.outer_thresholds), len(groups),
         "each held person has an inner-selected LDA threshold")
    c.true(result.metrics["auroc"] >= 0.9,
           "shrinkage LDA recovers the synthetic participant signal")
    c.eq(result.selection_protocol,
         "nested_participant_loso_shrinkage_lda_with_inner_oof_threshold",
         "LDA selection remains nested")


def test_balanced_logistic_remains_fully_nested(c: Check):
    reps, labels, groups = _dataset()
    result = evaluate_nested_balanced_logistic(reps, labels, groups)
    c.eq(len(result.outer_candidates), len(groups),
         "balanced Logistic selects one candidate inside each outer fold")
    c.eq(result.selection_protocol,
         "nested_participant_loso_balanced_logistic_with_inner_oof_threshold",
         "balanced loss and threshold stay inside participant folds")
    c.true(result.metrics["auroc"] >= 0.9,
           "balanced Logistic recovers the synthetic signal")


def test_metrics_and_public_report_are_recomputed_without_ids(c: Check):
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.1, 0.6, 0.4, 0.9], dtype=np.float64)
    metrics = recompute_binary_metrics(labels, probabilities)
    c.eq(metrics["accuracy"], 0.5, "0.5 threshold accuracy is exact")
    c.eq(metrics["sensitivity"], 0.5, "sensitivity is exact")
    c.eq(metrics["specificity"], 0.5, "specificity is exact")
    report = build_public_report(
        endpoint="als_vs_healthy_spread",
        labels=labels,
        probabilities=probabilities,
        protocol="synthetic_test",
        representation="min",
    )
    encoded = str(report)
    c.true("grp_" not in encoded and "rec_" not in encoded and "/Users/" not in encoded,
           "public report contains no participant, recording, or path identity")
    c.eq(report["metrics"], metrics, "public metrics are independently recomputed")


if __name__ == "__main__":
    run_all("test_neuroface_als_benchmark_v1", dict(globals()))
