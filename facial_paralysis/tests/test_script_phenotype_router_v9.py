from __future__ import annotations

import numpy as np
import torch

from _testlib import run_all
from test_medically_gated_shared_search_v2 import _dataset
from test_shared_normal_manifold_router_v4 import _inputs

from src.evaluation.residual_shared_search_v8 import evaluate_residual_candidate
from src.evaluation.script_phenotype_search_v9 import evaluate_phenotype_candidate
from src.models.residual_shared_router_v8 import candidate_registry_v8
from src.models.script_phenotype_router_v9 import (
    ScriptPhenotypeRouterV9,
    candidate_registry_v9,
)


def test_registry_has_exact_comparator_and_24_medical_script_candidates(c):
    registry = candidate_registry_v9()
    c.eq(len(registry), 25)
    c.eq(registry[0].candidate_id, "SAP9-000")
    c.eq(registry[0].script_blend, 0.0)
    c.eq(len({row.candidate_id for row in registry}), 25)
    c.true(all(row.medical_rationale for row in registry))


def test_action_bank_aggregates_repeated_script_actions_without_source_input(c):
    candidate = candidate_registry_v9()[1]
    model = ScriptPhenotypeRouterV9(candidate)
    tokens = torch.randn(2, 4, 64)
    mask = torch.ones(2, 4, dtype=torch.bool)
    codes = torch.tensor([[0, 0, 1, 2], [3, 4, 5, 5]], dtype=torch.long)
    bank, present = model.shared_action_phenotype_bank(tokens, mask, codes)
    c.eq(bank.shape, (2, 13, candidate.phenotype_dim))
    c.eq(present.shape, (2, 13))
    c.true(bool(present[0, 0]) and bool(present[1, 5]))
    c.true(model.task_specific_parameter_fraction() < 0.10)


def test_comparator_logits_are_bit_exact_v8(c):
    inputs = _inputs()
    tasks = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    torch.manual_seed(29)
    from src.models.residual_shared_router_v8 import ResidualSharedRouterV8
    baseline = ResidualSharedRouterV8(candidate_registry_v8()[1]).eval()
    torch.manual_seed(29)
    observed = ScriptPhenotypeRouterV9(candidate_registry_v9()[0]).eval()
    with torch.no_grad():
        first = baseline.shared_action_tokens(*inputs)
        expected = baseline.routed_logits(first, inputs[-2], tasks)
        second = observed.shared_action_tokens(*inputs)
        got = observed.routed_logits(second, inputs[-2], inputs[-1], tasks)
    c.true(torch.equal(first, second) and torch.equal(expected, got))


def test_evaluation_is_shared_complete_and_comparator_exact(c):
    dataset = _dataset()
    observed = evaluate_phenotype_candidate(
        dataset, candidate_registry_v9()[0], epochs=1, n_splits=2,
        seed=0, device="cpu",
    )
    baseline = evaluate_residual_candidate(
        dataset, candidate_registry_v8()[1], epochs=1, n_splits=2,
        seed=0, device="cpu",
    )
    c.true(np.array_equal(observed.probabilities, baseline.probabilities))
    c.eq(observed.shared_gradient_sources, ("palsynet", "neuroface", "meei"))
    c.true(observed.task_specific_parameter_fraction < 0.10)


if __name__ == "__main__":
    run_all("test_script_phenotype_router_v9", dict(globals()))
