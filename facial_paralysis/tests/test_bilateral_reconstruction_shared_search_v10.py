from __future__ import annotations

import numpy as np

from _testlib import run_all
from test_medically_gated_shared_search_v2 import _dataset

from src.evaluation.bilateral_reconstruction_shared_search_v10 import (
    candidate_is_promotable,
    evaluate_bilateral_reconstruction_candidate,
    uses_sam,
)
from src.evaluation.broad_literature_shared_search_v9 import (
    evaluate_broad_literature_candidate,
)
from src.models.bilateral_reconstruction_candidate_registry_v10 import (
    candidate_registry_v10,
)
from src.models.broad_literature_candidate_registry_v9 import candidate_registry_v9


def _v9_baseline():
    return next(
        row for row in candidate_registry_v9()
        if row.mechanism == "masked_clinical_reconstruction"
    )


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


def test_brv10_000_is_exact_blv9_009_under_same_evaluator(c):
    dataset = _dataset()
    observed = evaluate_bilateral_reconstruction_candidate(
        dataset, candidate_registry_v10()[0], epochs=1, n_splits=2,
        seed=0, device="cpu",
    )
    baseline = evaluate_broad_literature_candidate(
        dataset, _v9_baseline(), epochs=1, n_splits=2,
        seed=0, device="cpu",
    )
    c.true(np.array_equal(observed.probabilities, baseline.probabilities))
    c.eq(observed.metrics, baseline.metrics)
    c.eq(observed.active_candidate_id, "BRV10-000")


def test_all_six_candidates_share_protocol_and_complete_held_out_coverage(c):
    dataset = _dataset()
    for candidate in candidate_registry_v10():
        observed = evaluate_bilateral_reconstruction_candidate(
            dataset, candidate, epochs=1, n_splits=2, seed=0, device="cpu",
        )
        c.eq(observed.probabilities.shape, (12,))
        c.true(np.isfinite(observed.probabilities).all())
        c.eq(set(observed.metrics), {"palsynet", "neuroface", "meei"})
        c.eq(set(observed.loso_metrics), {"palsynet", "neuroface", "meei"})
        c.eq(observed.shared_gradient_sources, ("palsynet", "neuroface", "meei"))
        c.eq(observed.model_fits, 5)
        c.true(observed.task_specific_parameter_fraction < 0.10)
        for target, training_sources in observed.loso_train_sources:
            c.true(target not in training_sources)


def test_only_three_frozen_candidates_use_sam(c):
    observed = tuple(row.candidate_id for row in candidate_registry_v10() if uses_sam(row))
    c.eq(observed, ("BRV10-001", "BRV10-003", "BRV10-005"))


def test_promotion_gate_remains_strict_and_three_source(c):
    comparator = _metrics(specificity=0.80)
    c.true(candidate_is_promotable(_metrics(specificity=0.82), comparator))
    c.true(not candidate_is_promotable(
        _metrics(accuracy=0.89, specificity=0.90), comparator
    ))
    c.true(not candidate_is_promotable(_metrics(specificity=0.80), comparator))


if __name__ == "__main__":
    run_all("test_bilateral_reconstruction_shared_search_v10", dict(globals()))
