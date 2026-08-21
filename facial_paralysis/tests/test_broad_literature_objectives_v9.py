from __future__ import annotations

import numpy as np
import torch
from torch import nn

from _testlib import run_all
from test_medically_gated_shared_search_v2 import _dataset

from src.evaluation.medically_gated_shared_search_v2 import _model_inputs
from src.models.broad_literature_candidate_registry_v9 import candidate_registry_v9
from src.models.broad_literature_shared_router_v9 import BroadLiteratureSharedRouterV9
from src.training.broad_literature_objectives_v9 import (
    RepresentationAuxiliariesV9,
    SWAAccumulatorV9,
    SharpnessAwareControllerV9,
    action_dropout_mask,
    barlow_twins_loss,
    classification_objective,
    focal_binary_loss,
    ldam_binary_loss,
    modality_dropout_mask,
    source_pairwise_auc_loss,
    symmetric_binary_kl,
    vicreg_loss,
)


def _candidate(mechanism: str):
    return next(row for row in candidate_registry_v9() if row.mechanism == mechanism)


def _inputs():
    dataset = _dataset()
    indices = np.arange(len(dataset.base.labels), dtype=np.int64)
    values = _model_inputs(
        dataset, dataset.base.clinical_original, dataset.base.clinical_mirrored,
        indices, torch.device("cpu"),
    )
    tasks = torch.tensor([0, 0, 1, 1, 2, 2] * 2, dtype=torch.long)
    return dataset, values, tasks


def test_sam_asam_and_swa_are_real_weight_mechanisms(c):
    for adaptive in (False, True):
        model = nn.Linear(3, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        inputs = torch.tensor([[1.0, 2.0, -1.0]], dtype=torch.float32)
        model(inputs).sum().backward()
        control = model.weight.detach().clone()
        controller = SharpnessAwareControllerV9(
            model.parameters(), rho=0.5 if adaptive else 0.05,
            adaptive=adaptive, eta=0.01,
        )
        controller.first_step()
        perturbed = model.weight.detach().clone()
        c.true(not torch.equal(control, perturbed))
        c.true(len(controller.perturbations) == 1)
        optimizer.zero_grad(set_to_none=True)
        model(inputs).sum().backward()
        controller.second_step(optimizer)
        c.true(not torch.equal(model.weight.detach(), perturbed))

    model = nn.Linear(2, 1, bias=False)
    average = SWAAccumulatorV9(model)
    with torch.no_grad():
        model.weight.fill_(1.0)
    average.update(model)
    with torch.no_grad():
        model.weight.fill_(3.0)
    average.update(model)
    average.copy_to(model)
    c.true(torch.equal(model.weight, torch.full_like(model.weight, 2.0)))
    c.eq(average.updates, 2)


def test_missing_evidence_masks_are_deterministic_and_never_erase_the_exam(c):
    dense = torch.tensor([
        [True, True, True], [True, False, True], [False, False, False]
    ])
    first = modality_dropout_mask(dense, probability=0.2, seed=19)
    second = modality_dropout_mask(dense, probability=0.2, seed=19)
    c.true(torch.equal(first, second))
    c.true(bool((first <= dense).all()))

    actions = torch.tensor([
        [True, True, True], [True, False, False], [True, True, False]
    ])
    dropped = action_dropout_mask(actions, probability=1.0, seed=23)
    c.true(bool((dropped.sum(dim=1) >= 1).all()))
    c.true(bool((dropped <= actions).all()))
    c.eq(int(dropped[0].sum()), 2)
    c.eq(int(dropped[1].sum()), 1)
    c.eq(int(dropped[2].sum()), 1)


def test_consistency_and_self_supervision_formulas_are_finite_noncollapsed(c):
    logits_a = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    logits_b = torch.tensor([-0.5, 0.1, 1.5], requires_grad=True)
    consistency = symmetric_binary_kl(logits_a, logits_b)
    c.true(float(consistency) > 0.0)
    consistency.backward()
    c.true(float(logits_a.grad.norm()) > 0.0)

    generator = torch.Generator().manual_seed(7)
    first = torch.randn(12, 8, generator=generator, requires_grad=True)
    second = first.detach() + 0.1 * torch.randn(12, 8, generator=generator)
    second.requires_grad_(True)
    vicreg = vicreg_loss(first, second, 25.0, 25.0, 1.0)
    barlow = barlow_twins_loss(first, second, 0.005)
    c.true(float(vicreg) > 0.0)
    c.true(float(barlow) > 0.0)
    (vicreg + barlow).backward()
    c.true(float(first.grad.norm()) > 0.0)


def test_all_five_representation_models_update_the_shared_clinical_trunk(c):
    _dataset_value, values, tasks = _inputs()
    mechanisms = (
        "cross_view_vicreg",
        "cross_view_barlow_twins",
        "masked_clinical_reconstruction",
        "masked_action_reconstruction",
        "clinical_to_dense_reconstruction",
    )
    for mechanism in mechanisms:
        torch.manual_seed(29)
        candidate = _candidate(mechanism)
        model = BroadLiteratureSharedRouterV9(candidate)
        auxiliary = RepresentationAuxiliariesV9(candidate)
        model.zero_grad(set_to_none=True)
        auxiliary.zero_grad(set_to_none=True)
        loss = auxiliary.loss(model, values, tasks, seed=31)
        c.true(bool(torch.isfinite(loss)))
        c.true(float(loss) > 0.0)
        loss.backward()
        gradient = model.base.base.backbone.clinical_encoder[0].weight.grad
        c.true(gradient is not None and float(gradient.norm()) > 0.0)


def test_clinical_losses_have_closed_distinct_formulas(c):
    logits = torch.tensor([-1.2, 0.2, 1.0, -0.3, 0.7, 1.4], requires_grad=True)
    labels = torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    sources = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    weights = torch.full((6,), 1.0 / 6.0)
    focal = focal_binary_loss(logits, labels, gamma=2.0)
    c.eq(focal.shape, labels.shape)
    counts = torch.tensor([[1, 2], [2, 1]], dtype=torch.long)
    ldam = ldam_binary_loss(logits, labels, sources, counts, 0.5, 30.0)
    c.eq(ldam.shape, labels.shape)
    full_auc = source_pairwise_auc_loss(logits, labels, sources, 1.0)
    partial_auc = source_pairwise_auc_loss(logits, labels, sources, 0.5)
    c.true(float(full_auc) > 0.0)
    c.true(float(partial_auc) > 0.0)

    values = []
    for mechanism in (
        "focal_loss", "ldam_loss", "pairwise_auc_loss",
        "high_specificity_partial_auc_loss", "brier_composite_loss",
    ):
        value = classification_objective(
            _candidate(mechanism), logits, logits * 0.8, labels, weights,
            sources, counts,
        )
        c.true(bool(torch.isfinite(value)))
        values.append(round(float(value.detach()), 7))
    c.eq(len(set(values)), 5)
    sum(classification_objective(
        _candidate(mechanism), logits, logits * 0.8, labels, weights,
        sources, counts,
    ) for mechanism in (
        "focal_loss", "ldam_loss", "pairwise_auc_loss",
        "high_specificity_partial_auc_loss", "brier_composite_loss",
    )).backward()
    c.true(float(logits.grad.norm()) > 0.0)


if __name__ == "__main__":
    run_all("test_broad_literature_objectives_v9", dict(globals()))
