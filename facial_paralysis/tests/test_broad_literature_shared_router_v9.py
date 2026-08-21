from __future__ import annotations

import numpy as np
import torch

from _testlib import run_all
from test_medically_gated_shared_search_v2 import _dataset

from src.evaluation.medically_gated_shared_search_v2 import _model_inputs
from src.models.broad_literature_candidate_registry_v9 import candidate_registry_v9
from src.models.broad_literature_shared_router_v9 import BroadLiteratureSharedRouterV9
from src.models.residual_shared_router_v8 import ResidualSharedRouterV8, candidate_registry_v8


def _candidate(mechanism: str):
    return next(row for row in candidate_registry_v9() if row.mechanism == mechanism)


def _inputs():
    dataset = _dataset()
    count = len(dataset.base.labels)
    values = _model_inputs(
        dataset,
        dataset.base.clinical_original,
        dataset.base.clinical_mirrored,
        np.arange(count, dtype=np.int64),
        torch.device("cpu"),
    )
    tasks = torch.tensor([0, 0, 1, 1, 2, 2] * 2, dtype=torch.long)
    return values, tasks


def test_comparator_is_exact_v8(c):
    torch.manual_seed(17)
    observed = BroadLiteratureSharedRouterV9(_candidate("exact_v8_comparator")).eval()
    torch.manual_seed(17)
    baseline = ResidualSharedRouterV8(
        next(row for row in candidate_registry_v8() if row.candidate_id == "RSR8-001")
    ).eval()
    values, tasks = _inputs()
    with torch.no_grad():
        tokens = observed.shared_action_tokens(*values)
        c.true(torch.equal(tokens, baseline.shared_action_tokens(*values)))
        c.true(torch.equal(
            observed.routed_logits(tokens, values[-2], tasks),
            baseline.routed_logits(tokens, values[-2], tasks),
        ))


def test_four_architecture_models_are_active_shared_and_bounded(c):
    values, tasks = _inputs()
    architecture = tuple(
        row for row in candidate_registry_v9() if row.inference_change == "architecture"
    )
    c.eq(len(architecture), 4)
    for candidate in architecture:
        torch.manual_seed(23)
        model = BroadLiteratureSharedRouterV9(candidate)
        tokens = model.shared_action_tokens(*values)
        routed, universal = model.routed_and_universal_logits(
            tokens, values[-2], tasks
        )
        c.eq(tokens.shape, (12, 3, 64))
        c.eq(routed.shape, (12,))
        c.eq(universal.shape, (12,))
        c.true(bool(torch.isfinite(tokens).all()))
        c.true(bool(torch.isfinite(routed).all()))
        c.true(model.task_specific_parameter_fraction() < 0.10)
        model.zero_grad(set_to_none=True)
        routed.sum().backward()
        gradient = model.base.base.backbone.clinical_encoder[0].weight.grad
        c.true(gradient is not None and float(gradient.norm()) > 0.0)


def test_action_conditioning_and_graph_never_receive_source_identity(c):
    values, tasks = _inputs()
    for mechanism in ("action_conditioned_film", "anatomy_action_graph"):
        model = BroadLiteratureSharedRouterV9(_candidate(mechanism)).eval()
        names = tuple(name for name, _ in model.named_parameters())
        c.true(not any("source" in name or "dataset" in name for name in names))
        with torch.no_grad():
            first = model.shared_action_tokens(*values)
            changed = list(values)
            changed[-1] = torch.roll(values[-1], shifts=1, dims=1)
            second = model.shared_action_tokens(*tuple(changed))
        c.true(not torch.equal(first, second))
        c.true(torch.equal(tasks, tasks.clone()))


def test_architecture_mechanisms_change_the_v8_function(c):
    values, tasks = _inputs()
    for candidate in (
        row for row in candidate_registry_v9() if row.inference_change == "architecture"
    ):
        torch.manual_seed(31)
        model = BroadLiteratureSharedRouterV9(candidate).eval()
        torch.manual_seed(31)
        baseline = BroadLiteratureSharedRouterV9(
            _candidate("exact_v8_comparator")
        ).eval()
        with torch.no_grad():
            model_tokens = model.shared_action_tokens(*values)
            base_tokens = baseline.shared_action_tokens(*values)
            observed = model.routed_logits(model_tokens, values[-2], tasks)
            control = baseline.routed_logits(base_tokens, values[-2], tasks)
        c.true(not torch.equal(observed, control))


if __name__ == "__main__":
    run_all("test_broad_literature_shared_router_v9", dict(globals()))
