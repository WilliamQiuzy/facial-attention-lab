from __future__ import annotations

import numpy as np

from _testlib import run_all
from test_medically_gated_shared_search_v2 import _dataset

from src.evaluation.literature_grounded_shared_search_v9 import (
    candidate_is_non_degrading,
    evaluate_literature_grounded_candidate,
    screen_candidate_ids,
)
from src.evaluation.residual_shared_search_v8 import evaluate_residual_candidate
from src.models.literature_grounded_candidate_registry_v9 import (
    candidate_registry_v9,
)
from src.models.residual_shared_router_v8 import candidate_registry_v8


def _metrics(accuracy=0.92, specificity=0.85, sensitivity=0.90, auroc=0.94):
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


def test_registry_is_small_paper_bound_and_conditional(c):
    registry = candidate_registry_v9()
    c.eq(
        tuple(row.candidate_id for row in registry),
        ("LGS9-000", "LGS9-001", "LGS9-002", "LGS9-003"),
    )
    c.eq(tuple(row.auxiliary_weight for row in registry), (0.0, 0.0, 0.25, 0.25))
    c.eq(tuple(row.relation_enabled for row in registry), (False, True, False, True))
    c.true(all(row.paper_basis and row.medical_rationale for row in registry))


def test_combination_runs_only_after_both_single_mechanisms_are_non_degrading(c):
    comparator = _metrics()
    good = _metrics(accuracy=0.92, specificity=0.87, sensitivity=0.90, auroc=0.94)
    bad = _metrics(accuracy=0.85, specificity=0.90, sensitivity=0.70, auroc=0.88)
    c.true(candidate_is_non_degrading(good, comparator))
    c.true(not candidate_is_non_degrading(bad, comparator))
    c.eq(
        screen_candidate_ids({"LGS9-000": comparator, "LGS9-001": good, "LGS9-002": good}),
        ("LGS9-000", "LGS9-001", "LGS9-002", "LGS9-003"),
    )
    c.eq(
        screen_candidate_ids({"LGS9-000": comparator, "LGS9-001": good, "LGS9-002": bad}),
        ("LGS9-000", "LGS9-001", "LGS9-002"),
    )


def test_comparator_is_exact_v8_and_loso_never_trains_on_target_source(c):
    dataset = _dataset()
    observed = evaluate_literature_grounded_candidate(
        dataset, candidate_registry_v9()[0], epochs=1, n_splits=2,
        seed=0, device="cpu",
    )
    baseline = evaluate_residual_candidate(
        dataset,
        next(row for row in candidate_registry_v8() if row.candidate_id == "RSR8-001"),
        epochs=1, n_splits=2, seed=0, device="cpu",
    )
    c.true(np.array_equal(observed.probabilities, baseline.probabilities))
    c.eq(observed.model_fits, 5)
    c.eq(set(observed.loso_metrics), {"palsynet", "neuroface", "meei"})
    for target, training_sources in observed.loso_train_sources:
        c.true(target not in training_sources)
        c.eq(len(training_sources), 2)


def test_each_single_mechanism_updates_one_genuinely_shared_encoder(c):
    dataset = _dataset()
    for candidate in candidate_registry_v9()[1:3]:
        result = evaluate_literature_grounded_candidate(
            dataset, candidate, epochs=1, n_splits=2,
            seed=0, device="cpu",
        )
        c.eq(result.shared_gradient_sources, ("palsynet", "neuroface", "meei"))
        c.true(result.task_specific_parameter_fraction < 0.10)
        c.eq(result.probabilities.shape, (12,))


if __name__ == "__main__":
    run_all("test_literature_grounded_shared_search_v9", dict(globals()))
