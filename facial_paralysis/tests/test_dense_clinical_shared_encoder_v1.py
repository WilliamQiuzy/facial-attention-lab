from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all

from src.models.dense_clinical_shared_encoder_v1 import (
    ACTION_VOCAB,
    DenseClinicalSharedEncoder,
)


def _batch(batch_size: int = 3, action_count: int = 7):
    generator = torch.Generator().manual_seed(17)
    clinical = torch.randn(batch_size, action_count, 110, generator=generator)
    clinical_mirrored = clinical.flip(-1) + 0.02
    dense = torch.randn(
        batch_size, action_count, 32, 478, 3, generator=generator
    ) * 0.05
    dense_mirrored = dense.flip(-2).clone()
    dense_mirrored[..., 0] *= -1.0
    dense_valid = torch.ones(batch_size, action_count, 32, dtype=torch.bool)
    dense_available = torch.ones(batch_size, action_count, dtype=torch.bool)
    action_mask = torch.ones(batch_size, action_count, dtype=torch.bool)
    action_mask[0, 4:] = False
    action_mask[1, 3:] = False
    dense_valid &= action_mask.unsqueeze(-1)
    dense_available &= action_mask
    action_codes = torch.arange(action_count).repeat(batch_size, 1)
    action_codes %= len(ACTION_VOCAB)
    return (
        clinical,
        clinical_mirrored,
        dense,
        dense_mirrored,
        dense_valid,
        dense_available,
        action_mask,
        action_codes,
    )


def test_model_has_one_shared_encoder_with_small_task_heads(c):
    model = DenseClinicalSharedEncoder(use_dense=True)
    parameters = set(inspect.signature(model.forward).parameters)
    c.true("source" not in parameters and "dataset" not in parameters)
    names = tuple(name for name, _ in model.named_modules())
    c.true("universal_head" in names)
    c.eq(len(model.task_heads), 3)
    c.true(not any("palsynet" in name or "neuroface" in name or "meei" in name
                   for name in names))
    head_parameters = sum(p.numel() for p in model.universal_head.parameters())
    head_parameters += sum(p.numel() for p in model.task_heads.parameters())
    total_parameters = sum(p.numel() for p in model.parameters())
    c.true(head_parameters / total_parameters < 0.05)


def test_task_heads_only_route_the_shared_patient_embedding(c):
    model = DenseClinicalSharedEncoder(use_dense=True).eval()
    with torch.no_grad():
        embedding = model.encode(*_batch())
        task_codes = torch.tensor([0, 1, 2], dtype=torch.long)
        logits = model.task_logits_from_embedding(embedding, task_codes)
    c.eq(tuple(logits.shape), (3,))
    c.true(bool(torch.isfinite(logits).all()))
    c.raises(
        lambda: model.task_logits_from_embedding(embedding, torch.tensor([0, 1, 3])),
        ValueError,
    )


def test_forward_returns_one_finite_logit_per_patient(c):
    model = DenseClinicalSharedEncoder(use_dense=True).eval()
    with torch.no_grad():
        logits = model(*_batch())
        embeddings = model.encode(*_batch())
    c.eq(tuple(logits.shape), (3,))
    c.eq(tuple(embeddings.shape), (3, model.patient_dim))
    c.true(bool(torch.isfinite(logits).all()))
    c.true(bool(torch.isfinite(embeddings).all()))


def test_unavailable_dense_stream_is_exactly_inert(c):
    model = DenseClinicalSharedEncoder(use_dense=True).eval()
    batch = list(_batch(batch_size=2, action_count=4))
    batch[5].zero_()
    batch[4].zero_()
    with torch.no_grad():
        first = model(*batch)
        changed = list(batch)
        changed[2] = torch.randn_like(batch[2]) * 1000000.0
        changed[3] = torch.randn_like(batch[3]) * 1000000.0
        second = model(*changed)
    c.true(bool(torch.equal(first, second)))


def test_action_set_is_permutation_invariant_when_codes_move_with_tokens(c):
    model = DenseClinicalSharedEncoder(use_dense=True).eval()
    batch = _batch(batch_size=2, action_count=4)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = tuple(value[:, permutation] for value in batch)
    with torch.no_grad():
        original = model(*batch)
        observed = model(*permuted)
    c.true(bool(torch.allclose(original, observed, atol=1e-6, rtol=0.0)))


def test_original_mirror_pair_is_swap_invariant(c):
    model = DenseClinicalSharedEncoder(use_dense=True).eval()
    batch = list(_batch(batch_size=2, action_count=4))
    swapped = list(batch)
    swapped[0], swapped[1] = batch[1], batch[0]
    swapped[2], swapped[3] = batch[3], batch[2]
    with torch.no_grad():
        original = model(*batch)
        observed = model(*swapped)
    c.true(bool(torch.allclose(original, observed, atol=1e-6, rtol=0.0)))


def test_each_source_loss_updates_the_same_shared_layers(c):
    model = DenseClinicalSharedEncoder(use_dense=True).train()
    loss_function = torch.nn.BCEWithLogitsLoss()
    batch = _batch(batch_size=3, action_count=7)
    labels = torch.tensor([0.0, 1.0, 1.0])
    for source_index in range(3):
        model.zero_grad(set_to_none=True)
        selected = tuple(value[source_index : source_index + 1] for value in batch)
        embedding = model.encode(*selected)
        task_logit = model.task_logits_from_embedding(
            embedding, torch.tensor([source_index], dtype=torch.long)
        )
        universal_logit = model.universal_head(embedding).squeeze(-1)
        loss = loss_function(
            task_logit, labels[source_index : source_index + 1]
        ) + 0.25 * loss_function(
            universal_logit, labels[source_index : source_index + 1]
        )
        loss.backward()
        clinical_gradient = model.clinical_encoder[0].weight.grad
        patient_gradient = model.patient_projection.weight.grad
        c.true(clinical_gradient is not None and float(clinical_gradient.norm()) > 0.0)
        c.true(patient_gradient is not None and float(patient_gradient.norm()) > 0.0)
        if source_index > 0:
            dense_gradient = model.dense_spatial[0].weight.grad
            c.true(dense_gradient is not None and float(dense_gradient.norm()) > 0.0)


def test_malformed_masks_and_action_codes_fail_closed(c):
    model = DenseClinicalSharedEncoder(use_dense=True)
    batch = list(_batch(batch_size=2, action_count=4))
    batch[6][0].zero_()
    c.raises(lambda: model(*batch), ValueError)
    batch = list(_batch(batch_size=2, action_count=4))
    batch[7][0, 0] = len(ACTION_VOCAB)
    c.raises(lambda: model(*batch), ValueError)


if __name__ == "__main__":
    run_all("test_dense_clinical_shared_encoder_v1", dict(globals()))
