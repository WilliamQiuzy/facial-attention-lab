from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all
from test_shared_normal_manifold_router_v4 import _inputs

from src.models.script_aware_shared_router_v6 import (
    candidate_registry_v6,
    ScriptAwareSharedRouterV6,
)


def test_candidate_registry_is_exact_four(c):
    registry = candidate_registry_v6()
    c.eq(len(registry), 4)
    c.eq({item.head_mode for item in registry}, {"linear", "small_mlp"})
    c.eq({item.universal_blend for item in registry}, {0.25, 0.5})


def test_task_identity_enters_only_after_shared_action_tokens(c):
    model = ScriptAwareSharedRouterV6(candidate_registry_v6()[0]).eval()
    inputs = _inputs()
    with torch.no_grad():
        tokens = model.shared_action_tokens(*inputs)
        c.eq(tuple(tokens.shape), (6, 3, 64))
        first_tasks = torch.zeros(6, dtype=torch.long)
        second_tasks = torch.ones(6, dtype=torch.long)
        first = model.endpoint_embedding(tokens, inputs[-2], first_tasks)
        second = model.endpoint_embedding(tokens, inputs[-2], second_tasks)
    c.true(not torch.equal(first, second))
    c.true("task_codes" not in model.shared_action_tokens.__code__.co_varnames)


def test_patient_projection_remains_shared_after_script_pooling(c):
    model = ScriptAwareSharedRouterV6(candidate_registry_v6()[-1])
    names = tuple(name for name, _ in model.named_parameters())
    c.eq(sum(name.startswith("backbone.patient_projection") for name in names), 2)
    c.eq(len(model.task_heads), 3)
    task_specific = sum(
        parameter.numel() for name, parameter in model.named_parameters()
        if name.startswith("task_queries") or name.startswith("task_heads")
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    c.true(task_specific / total < 0.05)


def test_logits_use_endpoint_and_shared_universal_paths(c):
    candidate = candidate_registry_v6()[1]
    model = ScriptAwareSharedRouterV6(candidate).eval()
    inputs = _inputs()
    tasks = torch.tensor([0, 1, 2, 0, 1, 2])
    with torch.no_grad():
        tokens = model.shared_action_tokens(*inputs)
        endpoint = model.endpoint_embedding(tokens, inputs[-2], tasks)
        universal = model.universal_embedding(tokens, inputs[-2])
        task_logits = model.task_logits_from_embedding(endpoint, tasks)
        universal_logits = model.universal_head(universal).squeeze(-1)
        expected = (1 - candidate.universal_blend) * task_logits + candidate.universal_blend * universal_logits
        observed = model.routed_logits(tokens, inputs[-2], tasks)
    c.true(torch.equal(expected, observed))


if __name__ == "__main__":
    run_all("test_script_aware_shared_router_v6", dict(globals()))
