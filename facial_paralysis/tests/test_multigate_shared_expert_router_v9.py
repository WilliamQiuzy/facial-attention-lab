from __future__ import annotations

import torch

from _testlib import run_all

from src.models.multigate_shared_expert_router_v9 import (
    MultiGateSharedExpertRouterV9,
    candidate_registry_v9,
)
from src.models.residual_shared_router_v8 import (
    ResidualSharedRouterV8,
    candidate_registry_v8,
)


def _tokens():
    generator = torch.Generator().manual_seed(19)
    tokens = torch.randn(6, 4, 64, generator=generator)
    mask = torch.ones(6, 4, dtype=torch.bool)
    tasks = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    return tokens, mask, tasks


def test_registry_has_exact_comparator_and_one_mmoe_change(c):
    rows = candidate_registry_v9()
    c.eq(tuple(row.candidate_id for row in rows), ("MSE9-000", "MSE9-001"))
    c.eq(tuple(row.shared_expert_count for row in rows), (0, 3))
    c.eq(tuple(row.expert_rank for row in rows), (0, 16))
    c.true(all(row.paper_basis and row.medical_rationale for row in rows))


def test_comparator_is_exact_v8_and_experts_are_one_shared_bank(c):
    torch.manual_seed(7)
    observed = MultiGateSharedExpertRouterV9(candidate_registry_v9()[0])
    torch.manual_seed(7)
    baseline = ResidualSharedRouterV8(
        next(row for row in candidate_registry_v8() if row.candidate_id == "RSR8-001")
    )
    tokens, mask, tasks = _tokens()
    observed.eval()
    baseline.eval()
    c.true(torch.equal(
        observed.routed_logits(tokens, mask, tasks),
        baseline.routed_logits(tokens, mask, tasks),
    ))

    model = MultiGateSharedExpertRouterV9(candidate_registry_v9()[1])
    c.eq(len(model.shared_experts), 3)
    c.eq(len(model.task_gates), 3)
    c.true(model.task_specific_parameter_fraction() < 0.10)
    names = tuple(name for name, _ in model.named_parameters())
    c.true(any(name.startswith("shared_experts.") for name in names))
    c.true(not any(name.startswith("task_experts.") for name in names))


def test_every_task_updates_the_same_expert_bank(c):
    model = MultiGateSharedExpertRouterV9(candidate_registry_v9()[1])
    tokens, mask, tasks = _tokens()
    for task in range(3):
        model.zero_grad(set_to_none=True)
        selected = tasks == task
        routed, _ = model.routed_and_universal_logits(
            tokens[selected], mask[selected], tasks[selected]
        )
        routed.sum().backward()
        gradients = [
            parameter.grad for parameter in model.shared_experts.parameters()
        ]
        c.true(any(
            gradient is not None and float(gradient.norm()) > 0.0
            for gradient in gradients
        ))


if __name__ == "__main__":
    run_all("test_multigate_shared_expert_router_v9", dict(globals()))
