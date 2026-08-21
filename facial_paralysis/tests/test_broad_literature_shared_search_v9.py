from __future__ import annotations

import numpy as np

from _testlib import run_all
from test_medically_gated_shared_search_v2 import _dataset

from src.evaluation.broad_literature_shared_search_v9 import (
    candidate_is_promotable,
    evaluate_broad_literature_candidate,
)
from src.evaluation.residual_shared_search_v8 import evaluate_residual_candidate
from src.models.broad_literature_candidate_registry_v9 import candidate_registry_v9
from src.models.residual_shared_router_v8 import candidate_registry_v8


def _candidate(mechanism: str):
    return next(row for row in candidate_registry_v9() if row.mechanism == mechanism)


def _metrics(accuracy=0.91, specificity=0.82, sensitivity=0.90, auroc=0.93):
    return {
        source: {
            "accuracy": accuracy,
            "specificity": specificity,
            "sensitivity": sensitivity,
            "auroc": auroc,
            "balanced_accuracy": 0.5 * (specificity + sensitivity),
            "brier": 0.10,
        }
        for source in ("palsynet", "neuroface", "meei")
    }


def test_comparator_is_byte_semantically_exact_v8(c):
    dataset = _dataset()
    observed = evaluate_broad_literature_candidate(
        dataset, _candidate("exact_v8_comparator"), epochs=1, n_splits=2,
        seed=0, device="cpu",
    )
    baseline = evaluate_residual_candidate(
        dataset,
        next(row for row in candidate_registry_v8() if row.candidate_id == "RSR8-001"),
        epochs=1, n_splits=2, seed=0, device="cpu",
    )
    c.true(np.array_equal(observed.probabilities, baseline.probabilities))
    c.eq(observed.metrics, baseline.metrics)
    c.eq(observed.model_fits, 5)
    c.eq(observed.active_mechanism, "exact_v8_comparator")


def test_architecture_loss_self_supervision_and_sam_use_same_closed_evaluator(c):
    dataset = _dataset()
    for mechanism in (
        "sam",
        "cross_view_vicreg",
        "focal_loss",
        "action_conditioned_film",
    ):
        observed = evaluate_broad_literature_candidate(
            dataset, _candidate(mechanism), epochs=1, n_splits=2,
            seed=0, device="cpu",
        )
        c.eq(observed.probabilities.shape, (12,))
        c.true(np.isfinite(observed.probabilities).all())
        c.eq(set(observed.metrics), {"palsynet", "neuroface", "meei"})
        c.eq(set(observed.loso_metrics), {"palsynet", "neuroface", "meei"})
        c.eq(observed.shared_gradient_sources, ("palsynet", "neuroface", "meei"))
        c.eq(observed.active_mechanism, mechanism)
        c.true(observed.task_specific_parameter_fraction < 0.10)
        for target, training_sources in observed.loso_train_sources:
            c.true(target not in training_sources)


def test_promotion_gate_requires_every_source_and_strict_specificity_gain(c):
    comparator = _metrics(specificity=0.80)
    c.true(candidate_is_promotable(_metrics(specificity=0.82), comparator))
    low_accuracy = _metrics(accuracy=0.89, specificity=0.90)
    c.true(not candidate_is_promotable(low_accuracy, comparator))
    no_specificity_gain = _metrics(specificity=0.80)
    c.true(not candidate_is_promotable(no_specificity_gain, comparator))
    regressed = _metrics(accuracy=0.899, specificity=0.90, auroc=0.919)
    c.true(not candidate_is_promotable(regressed, comparator))


if __name__ == "__main__":
    run_all("test_broad_literature_shared_search_v9", dict(globals()))
