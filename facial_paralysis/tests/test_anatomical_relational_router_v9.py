from __future__ import annotations

import inspect

import torch

from _testlib import run_all
from test_shared_normal_manifold_router_v4 import _inputs

from src.models.anatomical_relational_router_v9 import (
    AnatomicalRelationalRouterV9,
    anatomical_region_indices,
    candidate_registry_v9,
)
from src.models.residual_shared_router_v8 import (
    ResidualSharedRouterV8,
    candidate_registry_v8,
)
from src.preprocessing.generalization_110d import (
    LANDMARK_MI_110D,
    candidate_feature_names,
)


def _v8_candidate():
    return next(row for row in candidate_registry_v8() if row.candidate_id == "RSR8-001")


def test_anatomical_partition_is_exact_disjoint_and_name_bound(c):
    names = candidate_feature_names(LANDMARK_MI_110D)
    regions = anatomical_region_indices(names)
    c.eq(tuple(regions), ("eye", "brow", "oral"))
    c.eq(tuple(len(regions[name]) for name in regions), (49, 19, 42))
    flattened = tuple(index for region in regions.values() for index in region)
    c.eq(len(set(flattened)), 110)
    c.eq(set(flattened), set(range(110)))
    reordered = list(names)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    c.raises(lambda: anatomical_region_indices(tuple(reordered)), ValueError)


def test_registry_contains_only_comparator_and_one_paper_supported_change(c):
    registry = candidate_registry_v9()
    c.eq(tuple(row.candidate_id for row in registry), ("ARR9-000", "ARR9-001"))
    c.true(not registry[0].relation_enabled)
    c.true(registry[1].relation_enabled)
    c.true("local" in registry[1].medical_rationale.lower())
    c.true("global" in registry[1].medical_rationale.lower())


def test_comparator_is_bit_exact_v8_and_relation_is_mirror_order_invariant(c):
    inputs = _inputs()
    torch.manual_seed(41)
    baseline = ResidualSharedRouterV8(_v8_candidate()).eval()
    torch.manual_seed(41)
    comparator = AnatomicalRelationalRouterV9(candidate_registry_v9()[0]).eval()
    torch.manual_seed(41)
    relation = AnatomicalRelationalRouterV9(candidate_registry_v9()[1]).eval()
    swapped = list(inputs)
    swapped[0], swapped[1] = inputs[1], inputs[0]
    swapped[2], swapped[3] = inputs[3], inputs[2]
    with torch.no_grad():
        expected = baseline.shared_action_tokens(*inputs)
        observed = comparator.shared_action_tokens(*inputs)
        first = relation.shared_action_tokens(*inputs)
        second = relation.shared_action_tokens(*tuple(swapped))
    c.true(torch.equal(expected, observed))
    c.true(torch.equal(first, second))


def test_relation_is_shared_trainable_and_endpoint_capacity_remains_small(c):
    model = AnatomicalRelationalRouterV9(candidate_registry_v9()[1]).train()
    inputs = _inputs()
    tokens = model.shared_action_tokens(*inputs)
    tasks = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    logits = model.routed_logits(tokens, inputs[-2], tasks)
    logits.sum().backward()
    c.true(float(model.relation_output.weight.grad.norm()) > 0.0)
    c.true(model.task_specific_parameter_fraction() < 0.10)
    c.true("source" not in inspect.signature(model.shared_action_tokens).parameters)
    module_names = tuple(name.lower() for name, _ in model.named_modules())
    c.true(not any(
        source in name
        for source in ("palsynet", "neuroface", "meei")
        for name in module_names
    ))


if __name__ == "__main__":
    run_all("test_anatomical_relational_router_v9", dict(globals()))
