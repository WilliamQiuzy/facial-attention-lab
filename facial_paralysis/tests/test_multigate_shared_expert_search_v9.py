from __future__ import annotations

import numpy as np

from _testlib import run_all
from test_medically_gated_shared_search_v2 import _dataset

from src.evaluation.multigate_shared_expert_search_v9 import (
    evaluate_multigate_shared_expert_candidate,
)
from src.evaluation.residual_shared_search_v8 import evaluate_residual_candidate
from src.models.multigate_shared_expert_router_v9 import candidate_registry_v9
from src.models.residual_shared_router_v8 import candidate_registry_v8


def test_comparator_is_exact_v8_with_loso_target_exclusion(c):
    dataset = _dataset()
    observed = evaluate_multigate_shared_expert_candidate(
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
    for target, training_sources in observed.loso_train_sources:
        c.true(target not in training_sources)


def test_mmoe_receives_gradients_from_all_sources_and_stays_mostly_shared(c):
    result = evaluate_multigate_shared_expert_candidate(
        _dataset(), candidate_registry_v9()[1], epochs=1, n_splits=2,
        seed=0, device="cpu",
    )
    c.eq(result.shared_gradient_sources, ("palsynet", "neuroface", "meei"))
    c.eq(result.shared_expert_gradient_sources, ("palsynet", "neuroface", "meei"))
    c.true(result.task_specific_parameter_fraction < 0.10)
    c.eq(result.probabilities.shape, (12,))


if __name__ == "__main__":
    run_all("test_multigate_shared_expert_search_v9", dict(globals()))
