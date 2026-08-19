from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all

from src.models.dense_clinical_shared_encoder_v1 import ACTION_VOCAB
from src.models.medical_shared_candidate_registry_v2 import candidate_registry
from src.models.medically_gated_shared_encoder_v2 import (
    BROW_LANDMARKS,
    EYE_LANDMARKS,
    MOUTH_LANDMARKS,
    MedicallyGatedSharedEncoderV2,
)


def _batch(batch_size: int = 3, action_count: int = 7):
    generator = torch.Generator().manual_seed(29)
    clinical = torch.randn(batch_size, action_count, 110, generator=generator)
    clinical_mirror = clinical.flip(-1) + 0.01
    dense = torch.randn(
        batch_size, action_count, 32, 478, 3, generator=generator
    ) * 0.02
    dense_mirror = dense.flip(-2).clone()
    dense_mirror[..., 0] *= -1.0
    valid = torch.ones(batch_size, action_count, 32, dtype=torch.bool)
    available = torch.ones(batch_size, action_count, dtype=torch.bool)
    timestamps = torch.arange(32, dtype=torch.float32)[None, None, :] / 30.0
    timestamps = timestamps.expand(batch_size, action_count, -1).clone()
    action_mask = torch.ones(batch_size, action_count, dtype=torch.bool)
    action_mask[0, 5:] = False
    valid &= action_mask.unsqueeze(-1)
    available &= action_mask
    timestamps *= action_mask.unsqueeze(-1)
    action_codes = torch.arange(action_count).repeat(batch_size, 1)
    action_codes %= len(ACTION_VOCAB)
    return (
        clinical, clinical_mirror, dense, dense_mirror, valid, available,
        timestamps, action_mask, action_codes,
    )


def test_region_indices_are_frozen_official_contours(c):
    c.eq(len(BROW_LANDMARKS), 20)
    c.eq(len(EYE_LANDMARKS), 32)
    c.eq(len(MOUTH_LANDMARKS), 40)
    c.eq(len(set(BROW_LANDMARKS)), len(BROW_LANDMARKS))
    c.eq(len(set(EYE_LANDMARKS)), len(EYE_LANDMARKS))
    c.eq(len(set(MOUTH_LANDMARKS)), len(MOUTH_LANDMARKS))
    c.true(set(BROW_LANDMARKS).isdisjoint(MOUTH_LANDMARKS))
    c.true(set(EYE_LANDMARKS).isdisjoint(MOUTH_LANDMARKS))


def test_action_matched_region_keeps_oral_and_global_for_pucker(c):
    candidate = next(
        item for item in candidate_registry()
        if item.view_mode == "original_only"
        and item.regional_mode == "matched_excursion"
    )
    model = MedicallyGatedSharedEncoderV2(candidate)
    dense = torch.zeros(1, 1, 32, 478, 3)
    dense[:, :, :, list(MOUTH_LANDMARKS), 1] = torch.linspace(0, 1, 32)[None, None, :, None]
    timestamps = torch.arange(32, dtype=torch.float32)[None, None, :] / 30.0
    code = torch.tensor([[ACTION_VOCAB.index("LIP_PUCKER")]], dtype=torch.long)
    evidence = model.regional_evidence(dense, dense, timestamps, code)
    c.eq(tuple(evidence.shape), (1, 1, 32))
    c.true(bool(torch.equal(evidence[..., :16], torch.zeros_like(evidence[..., :16]))))
    c.true(bool((evidence[..., 16:24] != 0).any()))
    c.true(bool((evidence[..., 24:32] != 0).any()))


def test_velocity_uses_real_seconds_and_no_cross_action_step(c):
    candidate = next(
        item for item in candidate_registry()
        if item.view_mode == "original_only"
        and item.regional_mode == "matched_excursion_velocity"
    )
    model = MedicallyGatedSharedEncoderV2(candidate)
    dense = torch.zeros(2, 1, 32, 478, 3)
    dense[:, :, :, list(EYE_LANDMARKS), 1] = torch.linspace(0, 1, 32)[None, None, :, None]
    timestamps = torch.arange(32, dtype=torch.float32)[None, None, :] / 30.0
    timestamps = torch.cat((timestamps, timestamps * 2.0), dim=0)
    code = torch.tensor([[ACTION_VOCAB.index("EYE_GENTLE")]] * 2)
    evidence = model.regional_evidence(dense, dense, timestamps, code)
    # Eye region is the second 8-value block; positions 2/3 are velocities.
    c.true(float(evidence[0, 0, 10]) > float(evidence[1, 0, 10]) * 1.9)
    c.true(float(evidence[0, 0, 11]) > float(evidence[1, 0, 11]) * 1.9)


def test_bilateral_candidate_is_exactly_swap_invariant(c):
    candidate = next(
        item for item in candidate_registry()
        if item.view_mode == "bilateral_invariant"
        and item.regional_mode == "matched_excursion_velocity"
        and item.pooling_mode == "meanmax_set"
        and item.fusion_mode == "masked_concat"
    )
    model = MedicallyGatedSharedEncoderV2(candidate).eval()
    batch = list(_batch(batch_size=2, action_count=5))
    swapped = list(batch)
    swapped[0], swapped[1] = batch[1], batch[0]
    swapped[2], swapped[3] = batch[3], batch[2]
    with torch.no_grad():
        first = model(*batch)
        second = model(*swapped)
    c.true(bool(torch.allclose(first, second, atol=1e-6, rtol=0.0)))


def test_every_registered_candidate_returns_finite_shared_logits(c):
    batch = _batch(batch_size=2, action_count=5)
    for candidate in candidate_registry():
        model = MedicallyGatedSharedEncoderV2(candidate).eval()
        with torch.no_grad():
            embedding = model.encode(*batch)
            logits = model.task_logits_from_embedding(
                embedding, torch.tensor([0, 1], dtype=torch.long)
            )
        c.eq(tuple(embedding.shape), (2, 64))
        c.true(bool(torch.isfinite(logits).all()))


def test_source_identity_is_absent_and_heads_are_small(c):
    candidate = candidate_registry()[0]
    model = MedicallyGatedSharedEncoderV2(candidate)
    parameters = set(inspect.signature(model.encode).parameters)
    c.true("source" not in parameters and "dataset" not in parameters)
    names = tuple(name.lower() for name, _ in model.named_modules())
    c.true(not any("palsynet" in name or "neuroface" in name or "meei" in name
                   for name in names))
    head_parameters = sum(parameter.numel() for parameter in model.task_heads.parameters())
    head_parameters += sum(parameter.numel() for parameter in model.universal_head.parameters())
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    c.true(head_parameters / total_parameters < 0.05)


def test_every_source_loss_updates_the_same_shared_patient_layers(c):
    candidate = candidate_registry()[-1]
    model = MedicallyGatedSharedEncoderV2(candidate).train()
    batch = _batch(batch_size=3, action_count=7)
    labels = torch.tensor([0.0, 1.0, 1.0])
    loss_function = torch.nn.BCEWithLogitsLoss()
    for task in range(3):
        model.zero_grad(set_to_none=True)
        selected = tuple(value[task:task + 1] for value in batch)
        embedding = model.encode(*selected)
        logits = model.task_logits_from_embedding(
            embedding, torch.tensor([task], dtype=torch.long)
        )
        loss_function(logits, labels[task:task + 1]).backward()
        c.true(float(model.clinical_encoder[0].weight.grad.norm()) > 0.0)
        c.true(float(model.patient_projection.weight.grad.norm()) > 0.0)
        if task > 0:
            c.true(float(model.dense_spatial[0].weight.grad.norm()) > 0.0)
            c.true(float(model.regional_encoder[0].weight.grad.norm()) > 0.0)


if __name__ == "__main__":
    run_all("test_medically_gated_shared_encoder_v2", dict(globals()))
