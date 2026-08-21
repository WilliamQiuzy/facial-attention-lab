from __future__ import annotations

import inspect

import numpy as np
import torch

from _testlib import run_all
from test_medically_gated_shared_search_v2 import _dataset

from src.evaluation.medically_gated_shared_search_v2 import _model_inputs
from src.models.bilateral_reconstruction_candidate_registry_v10 import (
    candidate_registry_v10,
)
from src.models.broad_literature_candidate_registry_v9 import candidate_registry_v9
from src.models.broad_literature_shared_router_v9 import BroadLiteratureSharedRouterV9
from src.training.bilateral_masked_reconstruction_v10 import (
    BilateralMaskedReconstructionV10,
    bilateral_reconstruction_targets,
    masked_bilateral_reconstruction_loss,
)


def _v9_model():
    candidate = next(
        row for row in candidate_registry_v9()
        if row.mechanism == "masked_clinical_reconstruction"
    )
    return BroadLiteratureSharedRouterV9(candidate)


def _inputs():
    dataset = _dataset()
    indices = np.arange(len(dataset.base.labels), dtype=np.int64)
    return _model_inputs(
        dataset, dataset.base.clinical_original, dataset.base.clinical_mirrored,
        indices, torch.device("cpu"),
    )


def test_bilateral_targets_are_exact_and_view_swap_invariant(c):
    original = torch.tensor([[[1.0, 4.0, -2.0]]])
    mirrored = torch.tensor([[[3.0, 0.0, -1.0]]])
    averaged = bilateral_reconstruction_targets(original, mirrored, "v9_average")
    decomposition = bilateral_reconstruction_targets(
        original, mirrored, "bilateral_decomposition"
    )
    swapped = bilateral_reconstruction_targets(
        mirrored, original, "bilateral_decomposition"
    )
    c.true(torch.equal(averaged, torch.tensor([[[2.0, 2.0, -1.5]]])))
    c.true(torch.equal(
        decomposition,
        torch.tensor([[[2.0, 2.0, -1.5, 2.0, 4.0, 1.0]]]),
    ))
    c.true(torch.equal(decomposition, swapped))


def test_unordered_twin_loss_preserves_views_without_assigning_side(c):
    original = torch.tensor([[[1.0, 2.0, 3.0]]])
    mirrored = torch.tensor([[[3.0, 1.0, 0.0]]])
    indices = torch.tensor([0, 2], dtype=torch.long)
    prediction = torch.cat((original, mirrored), dim=-1)
    swapped = torch.cat((mirrored, original), dim=-1)
    first = masked_bilateral_reconstruction_loss(
        prediction, original, mirrored, indices, "unordered_twin"
    )
    second = masked_bilateral_reconstruction_loss(
        swapped, original, mirrored, indices, "unordered_twin"
    )
    c.eq(float(first), 0.0)
    c.eq(float(second), 0.0)


def test_loss_uses_only_masked_clinical_indices(c):
    original = torch.arange(12, dtype=torch.float32).reshape(1, 1, 12)
    mirrored = torch.flip(original, dims=(-1,))
    indices = torch.tensor([2, 7], dtype=torch.long)
    target = bilateral_reconstruction_targets(
        original, mirrored, "bilateral_decomposition"
    )
    prediction = target.clone()
    prediction[..., 0] = 999.0
    prediction[..., 12] = -999.0
    loss = masked_bilateral_reconstruction_loss(
        prediction, original, mirrored, indices, "bilateral_decomposition"
    )
    c.eq(float(loss), 0.0)


def test_all_modes_send_finite_gradient_into_same_shared_clinical_encoder(c):
    inputs = _inputs()
    for candidate in candidate_registry_v10()[::2]:
        torch.manual_seed(41)
        model = _v9_model()
        auxiliary = BilateralMaskedReconstructionV10(candidate)
        model.zero_grad(set_to_none=True)
        loss = auxiliary.loss(model, inputs, seed=43)
        c.true(bool(torch.isfinite(loss)) and float(loss) > 0.0)
        loss.backward()
        gradient = model.base.base.backbone.clinical_encoder[0].weight.grad
        c.true(gradient is not None and float(gradient.norm()) > 0.0)


def test_auxiliary_has_no_source_or_dataset_interface(c):
    signature = inspect.signature(BilateralMaskedReconstructionV10.loss)
    c.true("source" not in str(signature).lower())
    c.true("dataset" not in str(signature).lower())


if __name__ == "__main__":
    run_all("test_bilateral_masked_reconstruction_v10", dict(globals()))
