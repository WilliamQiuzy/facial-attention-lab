from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all

from src.models.compact_shared_severity_v3 import (
    CompactSharedSeverityV3,
    compact_candidate_registry,
)
from src.models.dense_clinical_shared_encoder_v1 import ACTION_VOCAB


def _inputs(batch: int = 3, actions: int = 4):
    torch.manual_seed(7)
    clinical = torch.randn(batch, actions, 110)
    dense = torch.randn(batch, actions, 32, 478, 3) * 0.05
    available = torch.ones(batch, actions, dtype=torch.bool)
    timestamps = torch.linspace(0.0, 1.0, 32).repeat(batch, actions, 1)
    mask = torch.ones(batch, actions, dtype=torch.bool)
    codes = torch.tensor([
        ACTION_VOCAB.index("BROW_RAISE"),
        ACTION_VOCAB.index("EYE_GENTLE"),
        ACTION_VOCAB.index("LIP_PUCKER"),
        ACTION_VOCAB.index("FREE_EARLY"),
    ]).repeat(batch, 1)
    return clinical, dense, available, timestamps, mask, codes


def test_registry_is_16_medically_bounded_shared_candidates(c):
    registry = compact_candidate_registry()
    c.eq(len(registry), 16)
    c.eq(len({candidate.candidate_id for candidate in registry}), 16)
    for candidate in registry:
        c.true(candidate.region_scope in {"all_regions", "action_matched"})
        c.true(candidate.dynamic_stats in {"excursion", "excursion_velocity"})
        c.true(candidate.pooling in {"meanmax", "action_weighted"})
        c.true(candidate.head_mode in {"embedding_head", "severity_calibration"})


def test_model_consumes_478_but_has_no_flip_or_source_input(c):
    source = inspect.getsource(CompactSharedSeverityV3.forward).lower()
    c.true("source" not in source and "mirror" not in source and "flip" not in source)
    inputs = _inputs()
    model = CompactSharedSeverityV3(compact_candidate_registry()[0]).eval()
    with torch.no_grad():
        first = model(*inputs)
        changed = list(inputs)
        changed[1] = inputs[1].clone()
        changed[1][:, :, :, 0, :] += 0.5
        second = model(*changed)
    c.true(not torch.equal(first, second))


def test_action_matched_descriptor_respects_brow_eye_oral_anatomy(c):
    candidate = next(
        item for item in compact_candidate_registry()
        if item.region_scope == "action_matched"
        and item.dynamic_stats == "excursion_velocity"
    )
    model = CompactSharedSeverityV3(candidate)
    _, dense, available, timestamps, _, codes = _inputs()
    descriptor = model.regional_descriptor(dense, timestamps, available, codes)
    shaped = descriptor.reshape(3, 4, 4, 4)
    c.true(torch.all(shaped[:, 0, 1:3] == 0.0))
    c.true(torch.any(shaped[:, 0, 0] != 0.0))
    c.true(torch.any(shaped[:, 0, 3] != 0.0))
    c.true(torch.all(shaped[:, 2, :2] == 0.0))
    c.true(torch.any(shaped[:, 2, 2] != 0.0))


def test_excursion_candidate_excludes_velocity_exactly(c):
    candidate = next(
        item for item in compact_candidate_registry()
        if item.dynamic_stats == "excursion" and item.region_scope == "all_regions"
    )
    model = CompactSharedSeverityV3(candidate)
    _, dense, available, timestamps, _, codes = _inputs()
    shaped = model.regional_descriptor(
        dense, timestamps, available, codes
    ).reshape(3, 4, 4, 4)
    c.true(torch.all(shaped[..., 2:] == 0.0))


def test_action_weights_are_masked_and_normalized(c):
    candidate = next(
        item for item in compact_candidate_registry() if item.pooling == "action_weighted"
    )
    model = CompactSharedSeverityV3(candidate)
    inputs = list(_inputs())
    inputs[4] = inputs[4].clone()
    inputs[4][:, -1] = False
    inputs[2] = inputs[2].clone()
    inputs[2][:, -1] = False
    inputs[3] = inputs[3].clone()
    inputs[3][:, -1] = 0.0
    _, weights = model.encode_with_action_weights(*inputs)
    c.true(torch.all(weights[:, -1] == 0.0))
    c.true(np.allclose(weights.sum(dim=1).detach().numpy(), 1.0, atol=1e-6))


def test_severity_calibration_is_monotone_for_every_task(c):
    candidate = next(
        item for item in compact_candidate_registry()
        if item.head_mode == "severity_calibration"
    )
    model = CompactSharedSeverityV3(candidate)
    embedding = torch.randn(7, 64)
    severity = model.shared_severity(embedding).squeeze(-1)
    for task in range(3):
        codes = torch.full((7,), task, dtype=torch.long)
        logits = model.task_logits_from_embedding(embedding, codes)
        ordering = torch.argsort(severity)
        c.true(torch.all(torch.diff(logits[ordering]) >= 0.0))


def test_all_candidates_emit_one_finite_logit_per_participant(c):
    inputs = _inputs()
    tasks = torch.tensor([0, 1, 2], dtype=torch.long)
    for candidate in compact_candidate_registry():
        model = CompactSharedSeverityV3(candidate)
        embedding = model.encode(*inputs)
        logits = model.task_logits_from_embedding(embedding, tasks)
        c.eq(tuple(embedding.shape), (3, 64))
        c.eq(tuple(logits.shape), (3,))
        c.true(torch.isfinite(logits).all())


if __name__ == "__main__":
    run_all("test_compact_shared_severity_v3", dict(globals()))
