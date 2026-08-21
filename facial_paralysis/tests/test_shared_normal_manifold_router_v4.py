from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all

from src.models.normal_manifold_candidate_registry_v4 import candidate_registry_v4
from src.models.shared_normal_manifold_router_v4 import SharedNormalManifoldRouterV4


def _inputs(batch: int = 6, actions: int = 3):
    torch.manual_seed(8)
    clinical = torch.randn(batch, actions, 110)
    mirrored = clinical.clone()
    mirrored[..., :12] *= -1.0
    dense = torch.randn(batch, actions, 32, 478, 3) * 0.01
    dense_mirror = dense.clone()
    dense_mirror[..., 0] *= -1.0
    valid = torch.ones(batch, actions, 32, dtype=torch.bool)
    available = torch.ones(batch, actions, dtype=torch.bool)
    timestamps = torch.arange(32).float()[None, None, :].repeat(batch, actions, 1) / 30.0
    mask = torch.ones(batch, actions, dtype=torch.bool)
    codes = torch.arange(actions, dtype=torch.long)[None, :].repeat(batch, 1)
    return clinical, mirrored, dense, dense_mirror, valid, available, timestamps, mask, codes


def test_one_shared_anchor_and_endpoint_heads_are_after_embedding(c):
    model = SharedNormalManifoldRouterV4(candidate_registry_v4()[1])
    c.eq(tuple(model.normal_anchor.shape), (64,))
    names = tuple(name for name, _ in model.named_parameters())
    c.eq(sum(name.endswith("normal_anchor") for name in names), 1)
    c.eq(len(model.backbone.task_heads), 3)
    embedding = model.encode(*_inputs())
    c.eq(tuple(embedding.shape), (6, 64))
    task_codes = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    logits = model.routed_logits_from_embedding(embedding, task_codes)
    c.eq(tuple(logits.shape), (6,))


def test_manifold_loss_uses_controls_only_and_preserves_affected_geometry(c):
    model = SharedNormalManifoldRouterV4(candidate_registry_v4()[2])
    embedding = torch.zeros(4, 64)
    embedding[0] = 1.0
    embedding[1] = 2.0
    embedding[2] = 10.0
    embedding[3] = -9.0
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    weights = torch.tensor([0.2, 0.3, 0.1, 0.4])
    first = model.normal_manifold_loss(embedding, labels, weights)
    changed = embedding.clone()
    changed[2:] *= 100.0
    second = model.normal_manifold_loss(changed, labels, weights)
    c.eq(float(first), float(second))
    changed[0] += 2.0
    c.true(float(model.normal_manifold_loss(changed, labels, weights)) > float(first))


def test_routed_logit_is_exact_shared_universal_blend(c):
    candidate = candidate_registry_v4()[3]
    model = SharedNormalManifoldRouterV4(candidate)
    embedding = torch.randn(5, 64)
    task_codes = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
    task = model.backbone.task_logits_from_embedding(embedding, task_codes)
    universal = model.backbone.universal_head(embedding).squeeze(-1)
    expected = (1.0 - candidate.universal_blend) * task + candidate.universal_blend * universal
    c.true(torch.equal(model.routed_logits_from_embedding(embedding, task_codes), expected))


def test_missing_dense_evidence_is_masked_not_fabricated(c):
    values = list(_inputs())
    values[2][0] = 999.0
    values[3][0] = -999.0
    values[4][0] = False
    values[5][0] = False
    model = SharedNormalManifoldRouterV4(candidate_registry_v4()[0]).eval()
    with torch.no_grad():
        first = model.encode(*values)
        values[2][0] = -123.0
        values[3][0] = 456.0
        second = model.encode(*values)
    c.true(torch.equal(first[0], second[0]))


def test_model_rejects_forged_candidate(c):
    from src.models.normal_manifold_candidate_registry_v4 import NormalManifoldCandidateV4
    c.raises(
        lambda: SharedNormalManifoldRouterV4(NormalManifoldCandidateV4("fake", 0.1, 0.2)),
        ValueError,
    )


if __name__ == "__main__":
    run_all("test_shared_normal_manifold_router_v4", dict(globals()))
