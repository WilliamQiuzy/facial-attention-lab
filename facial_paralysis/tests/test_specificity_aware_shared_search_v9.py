from __future__ import annotations

import inspect

import numpy as np

from _testlib import run_all
from test_medically_gated_shared_search_v2 import _dataset

from src.evaluation.residual_shared_search_v8 import evaluate_residual_candidate
from src.evaluation.specificity_aware_shared_search_v9 import (
    SpecificityEvaluationV9,
    calibrated_binary_metrics,
    evaluate_specificity_candidate,
    rank_specificity_results,
    select_training_threshold,
)
from src.models.residual_shared_router_v8 import candidate_registry_v8
from src.models.specificity_aware_candidate_registry_v9 import candidate_registry_v9


def _v8_candidate():
    return next(row for row in candidate_registry_v8() if row.candidate_id == "RSR8-001")


def test_threshold_is_training_only_deterministic_and_sensitivity_constrained(c):
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.1, 0.3, 0.7, 0.65, 0.8, 0.9], dtype=np.float64)
    threshold = select_training_threshold(labels, probabilities, min_sensitivity=0.90)
    c.eq(threshold, 0.65)
    predicted = probabilities >= threshold
    c.eq(float(np.mean(predicted[labels == 1])), 1.0)
    c.eq(float(np.mean(~predicted[labels == 0])), 2.0 / 3.0)
    c.eq(tuple(inspect.signature(select_training_threshold).parameters), (
        "labels", "probabilities", "min_sensitivity",
    ))


def test_calibrated_metrics_preserve_probability_metrics(c):
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.2, 0.7, 0.6, 0.9], dtype=np.float64)
    fixed = calibrated_binary_metrics(labels, probabilities, probabilities >= 0.5)
    shifted = calibrated_binary_metrics(labels, probabilities, probabilities >= 0.7)
    c.eq(fixed["auroc"], shifted["auroc"])
    c.eq(fixed["brier"], shifted["brier"])
    c.true(fixed["sensitivity"] != shifted["sensitivity"])


def test_off_candidate_reproduces_v8_and_evaluation_is_complete(c):
    dataset = _dataset()
    v8 = evaluate_residual_candidate(
        dataset, _v8_candidate(), epochs=1, n_splits=2, seed=0, device="cpu",
    )
    v9 = evaluate_specificity_candidate(
        dataset, candidate_registry_v9()[0], epochs=1, n_splits=2,
        seed=0, device="cpu",
    )
    c.true(np.array_equal(v8.probabilities, v9.probabilities))
    c.eq(v9.probabilities.shape, (12,))
    c.eq(v9.calibrated_predictions.shape, (12,))
    c.eq(v9.model_fits, 2)
    c.eq(v9.shared_gradient_sources, ("palsynet", "neuroface", "meei"))
    c.eq(set(v9.fixed_metrics), {"palsynet", "neuroface", "meei"})
    c.eq(set(v9.calibrated_metrics), {"palsynet", "neuroface", "meei"})
    c.eq({len(value) for value in v9.thresholds_by_source.values()}, {2})


def _evaluation(value: float, *, specificity: float, sensitivity: float = 0.90):
    metrics = {
        source: {
            "accuracy": value,
            "auroc": value + 0.02,
            "balanced_accuracy": 0.5 * (specificity + sensitivity),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "brier": 1.0 - value,
        }
        for source in ("palsynet", "neuroface", "meei")
    }
    probabilities = np.linspace(0.1, 0.9, 12)
    return SpecificityEvaluationV9(
        probabilities=probabilities,
        calibrated_predictions=probabilities >= 0.5,
        fixed_metrics=metrics,
        calibrated_metrics=metrics,
        thresholds_by_source={source: (0.5, 0.5) for source in metrics},
        model_fits=2,
        shared_gradient_sources=("palsynet", "neuroface", "meei"),
    )


def test_ranking_prioritizes_feasible_worst_source_specificity(c):
    registry = candidate_registry_v9()
    results = {
        row.candidate_id: _evaluation(0.91, specificity=0.82)
        for row in registry
    }
    results[registry[0].candidate_id] = _evaluation(0.92, specificity=0.80)
    results[registry[1].candidate_id] = _evaluation(0.91, specificity=0.90)
    ranking = rank_specificity_results(results, comparator_id=registry[0].candidate_id)
    c.eq(ranking[0], registry[1].candidate_id)
    results[registry[1].candidate_id] = _evaluation(
        0.80, specificity=0.95, sensitivity=0.80,
    )
    ranking = rank_specificity_results(results, comparator_id=registry[0].candidate_id)
    c.true(ranking[0] != registry[1].candidate_id)


if __name__ == "__main__":
    run_all("test_specificity_aware_shared_search_v9", dict(globals()))
