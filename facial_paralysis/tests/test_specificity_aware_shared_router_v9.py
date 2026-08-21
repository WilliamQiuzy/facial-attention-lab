from __future__ import annotations

import inspect

import torch

from _testlib import run_all
from test_shared_normal_manifold_router_v4 import _inputs

from src.models.residual_shared_router_v8 import (
    ResidualSharedRouterV8,
    candidate_registry_v8,
)
from src.models.specificity_aware_candidate_registry_v9 import candidate_registry_v9
from src.models.specificity_aware_shared_router_v9 import SpecificityAwareSharedRouterV9


def _v8_candidate():
    return next(row for row in candidate_registry_v8() if row.candidate_id == "RSR8-001")


def _v9_candidate(candidate_id: str):
    return next(row for row in candidate_registry_v9() if row.candidate_id == candidate_id)


def test_off_candidate_is_exact_v8_comparator(c):
    inputs = _inputs()
    tasks = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    torch.manual_seed(19)
    v8 = ResidualSharedRouterV8(_v8_candidate()).eval()
    torch.manual_seed(19)
    v9 = SpecificityAwareSharedRouterV9(_v9_candidate("SSR9-000")).eval()
    with torch.no_grad():
        tokens_v8 = v8.shared_action_tokens(*inputs)
        expected = v8.routed_logits(tokens_v8, inputs[-2], tasks)
        tokens_v9 = v9.shared_action_tokens(*inputs)
        observed = v9.routed_logits(tokens_v9, inputs[-2], tasks)
    c.true(torch.equal(tokens_v8, tokens_v9))
    c.true(torch.equal(expected, observed))


def test_healthy_reference_is_shared_finite_and_post_embedding(c):
    model = SpecificityAwareSharedRouterV9(_v9_candidate("SSR9-016")).eval()
    inputs = _inputs()
    tasks = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32)
    weights = torch.full((6,), 1.0 / 6.0)
    with torch.no_grad():
        tokens = model.shared_action_tokens(*inputs)
        endpoint, universal = model.patient_embeddings(tokens, inputs[-2], tasks)
        distances = model.normal_distance(universal)
        logits = model.normality_logits(universal)
        reference_loss = model.normal_reference_loss(universal, labels, weights)
        alignment_loss = model.control_alignment_loss(
            universal, labels, tasks, weights,
        )
    c.eq(endpoint.shape, (6, 64))
    c.eq(universal.shape, (6, 64))
    c.eq(distances.shape, (6,))
    c.eq(logits.shape, (6,))
    c.true(bool(torch.isfinite(distances).all()) and bool((distances >= 0).all()))
    c.true(bool(torch.isfinite(logits).all()))
    c.true(float(reference_loss) >= 0.0 and float(alignment_loss) >= 0.0)
    c.true("task_codes" not in inspect.signature(model.normality_logits).parameters)


def test_task_specific_capacity_remains_small_and_after_shared_embedding(c):
    model = SpecificityAwareSharedRouterV9(_v9_candidate("SSR9-023"))
    task_specific = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("base.adapters")
        or name.startswith("base.base.task_queries")
        or name.startswith("base.base.backbone.task_heads")
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    c.true(task_specific / total < 0.10)
    c.true("tokens" not in model.base.adapt_endpoint.__code__.co_varnames)
    c.true("sources" not in inspect.signature(model.routed_logits).parameters)


if __name__ == "__main__":
    run_all("test_specificity_aware_shared_router_v9", dict(globals()))
