from __future__ import annotations

import inspect

import numpy as np
import torch

from _testlib import run_all
from test_medically_gated_shared_search_v2 import _dataset

from src.evaluation.shared_mechanism_search_v9 import (
    evaluate_mechanism_candidate,
    mechanism_action_tensor,
)
from src.models.shared_mechanism_router_v9 import (
    SharedMechanismRouterV9,
    candidate_registry_v9,
)


def test_registry_is_bounded_and_every_candidate_is_genuinely_shared(c):
    registry = candidate_registry_v9()
    c.eq(len(registry), 16)
    c.eq(len({row.candidate_id for row in registry}), 16)
    c.true(all(row.medical_rationale for row in registry))
    for candidate in registry:
        model = SharedMechanismRouterV9(candidate)
        names = tuple(name for name, _ in model.named_parameters())
        c.true(any(name.startswith("action_encoder") for name in names))
        c.true(not any("source_encoder" in name for name in names))
        c.true(model.task_specific_parameter_fraction() < 0.10)


def test_action_tensor_preserves_clinical_and_dense_mechanisms(c):
    evidence = mechanism_action_tensor(_dataset(per_cell=4))
    c.eq(evidence.values.shape, (24, 13, 237))
    c.eq(evidence.action_mask.shape, (24, 13))
    c.eq(evidence.dense_available.shape, (24, 13))
    c.true(np.isfinite(evidence.values).all())
    c.true(not evidence.values.flags.writeable)
    c.true("sources" not in inspect.signature(mechanism_action_tensor).parameters)


def test_forward_uses_one_shared_action_encoder_then_tiny_heads(c):
    candidate = candidate_registry_v9()[0]
    model = SharedMechanismRouterV9(candidate)
    evidence = mechanism_action_tensor(_dataset())
    values = torch.from_numpy(np.array(evidence.values, dtype=np.float32, copy=True))
    mask = torch.from_numpy(np.array(evidence.action_mask, copy=True))
    tasks = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    logits, universal = model(values, mask, tasks)
    c.eq(logits.shape, (12,))
    c.eq(universal.shape, (12,))
    c.true(bool(torch.isfinite(logits).all()))


def test_participant_disjoint_evaluation_is_complete(c):
    result = evaluate_mechanism_candidate(
        _dataset(), candidate_registry_v9()[0], epochs=1, n_splits=2,
        seed=0, device="cpu",
    )
    c.eq(result.probabilities.shape, (12,))
    c.eq(result.model_fits, 2)
    c.eq(result.shared_gradient_sources, ("palsynet", "neuroface", "meei"))
    c.true(result.task_specific_parameter_fraction < 0.10)


if __name__ == "__main__":
    run_all("test_shared_mechanism_router_v9", dict(globals()))
