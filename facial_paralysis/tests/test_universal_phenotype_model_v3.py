"""Architecture contracts for the source-blind universal phenotype mixture."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.models.universal_phenotype_v3 import (  # noqa: E402
    FIXED_FUSION_RULES,
    EXPERT_NAMES,
    MODEL_VARIANTS,
    fixed_source_blind_fusion,
    healthy_control_alignment_loss,
    UniversalPhenotypeMixture,
    group_dro_binary_loss,
)
from _testlib import Check, run_all  # noqa: E402


def _batch(batch: int = 4, instances: int = 9):
    generator = torch.Generator().manual_seed(7)
    return {
        "landmark_original": torch.randn(batch, 110, generator=generator),
        "landmark_mirrored": torch.randn(batch, 110, generator=generator),
        "common_original": torch.randn(batch, instances, 398, generator=generator),
        "common_mirrored": torch.randn(batch, instances, 398, generator=generator),
        "instance_mask": torch.tensor([
            [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0],
        ], dtype=torch.bool),
        "temporal_features": torch.randn(
            batch, instances, 4, 32, 95, generator=generator
        ),
        "temporal_valid_mask": torch.ones(
            batch, instances, 4, 32, dtype=torch.bool
        ),
        "au_instances": torch.randn(batch, instances, 100, generator=generator),
        "au_temporal": torch.randn(
            batch, instances, 64, 20, generator=generator
        ),
        "au_temporal_mask": torch.ones(
            batch, instances, 64, dtype=torch.bool
        ),
        "au_mask": torch.tensor([
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0],
        ], dtype=torch.bool),
        "task_codes": torch.tensor([
            [0, 1, -1, -1, -1, -1, -1, -1, -1],
            [1, 2, 3, -1, -1, -1, -1, -1, -1],
            [0, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 2, 3, 4, -1, -1, -1, -1, -1],
        ], dtype=torch.long),
    }


def test_all_large_architecture_variants_return_expert_and_fused_logits(c: Check):
    c.eq(EXPERT_NAMES, (
        "global_landmark110", "action_landmark110", "common398",
        "mediapipe_temporal", "au_bilateral_capacity", "au_palsy_capacity",
    ), "expert order and clinical roles are frozen")
    c.true("hybrid_set_transformer" in MODEL_VARIANTS,
           "search includes cross-action set attention as a major architecture")
    batch = _batch()
    for variant in MODEL_VARIANTS:
        model = UniversalPhenotypeMixture(variant=variant, width=32, dropout=0.0)
        model.eval()
        output = model(**batch)
        c.eq(output["fused_logit"].shape, (4,), f"{variant} returns participant logits")
        c.eq(output["expert_logits"].shape, (4, 6),
             f"{variant} exposes six auditable expert logits")
        c.eq(output["gate_weights"].shape, (4, 6),
             f"{variant} exposes source-blind reliability weights")
        c.eq(output["expert_embeddings"].shape, (4, 6, 32),
             f"{variant} exposes train-only expert embeddings for domain alignment")
        c.true(torch.allclose(output["gate_weights"].sum(dim=1), torch.ones(4)),
               f"{variant} gate weights sum to one")
        c.true(torch.all(output["gate_weights"][:1, 4:] == 0),
               f"{variant} never assigns weight to either missing AU expert")


def test_source_identity_is_not_a_model_input_and_missing_au_values_are_inert(c: Check):
    model = UniversalPhenotypeMixture(variant="hybrid_tcn_mil", width=24, dropout=0.0)
    model.eval()
    batch = _batch()
    first = model(**batch)["fused_logit"]
    changed = dict(batch)
    changed["au_instances"] = batch["au_instances"].clone()
    changed["au_instances"][~batch["au_mask"]] = 10_000.0
    changed["au_temporal"] = batch["au_temporal"].clone()
    changed["au_temporal"][~batch["au_mask"]] = 10_000.0
    second = model(**changed)["fused_logit"]
    c.true(torch.allclose(first, second, atol=0.0, rtol=0.0),
           "values behind a false modality mask cannot affect prediction")
    c.true("source" not in model.forward.__annotations__,
           "forward contract contains no dataset/source identity")


def test_gate_can_route_distinct_phenotypes_with_equal_modality_reliability(c: Check):
    model = UniversalPhenotypeMixture(
        variant="hybrid_set_transformer", width=24, dropout=0.0
    )
    model.eval()
    c.true(hasattr(model, "action_landmark"),
           "task-specific Landmark-110 has its own encoder")
    source = _batch()
    paired = {
        name: value[:1].repeat((2,) + (1,) * (value.ndim - 1))
        for name, value in source.items()
    }
    paired["landmark_original"][1, :20] += 8.0
    paired["landmark_mirrored"][1, 20:40] -= 8.0
    weights = model(**paired)["gate_weights"]
    c.true(not torch.allclose(weights[0], weights[1]),
           "equal-quality ALS-like and unilateral-like signals can route to different experts")


def test_gate_quality_is_invariant_to_duplicate_file_packaging(c: Check):
    model = UniversalPhenotypeMixture(
        variant="hybrid_set_transformer", width=24, dropout=0.0
    )
    model.eval()
    source = _batch()
    paired = {
        name: value[:1].repeat((2,) + (1,) * (value.ndim - 1))
        for name, value in source.items()
    }
    for name in (
        "common_original", "common_mirrored", "temporal_features",
        "temporal_valid_mask", "au_instances", "au_temporal",
        "au_temporal_mask",
    ):
        paired[name][1, 1] = paired[name][1, 0]
    paired["instance_mask"][:] = False
    paired["instance_mask"][:, 0] = True
    paired["instance_mask"][1, 1] = True
    paired["au_mask"][:] = False
    paired["au_mask"][:, 0] = True
    paired["au_mask"][1, 1] = True
    paired["task_codes"][:] = -1
    paired["task_codes"][:, 0] = 1
    paired["task_codes"][1, 1] = 1
    weights = model(**paired)["gate_weights"]
    c.true(torch.allclose(weights[0], weights[1], atol=1e-6, rtol=1e-6),
           "duplicating identical evidence into two files cannot reveal the dataset")


def test_action_landmark_expert_preserves_task_specific_110d_geometry(c: Check):
    model = UniversalPhenotypeMixture(
        variant="hybrid_set_transformer", width=24, dropout=0.0
    )
    model.eval()
    batch = _batch()
    baseline = model(**batch)["expert_logits"]
    changed = dict(batch)
    changed["common_original"] = batch["common_original"].clone()
    changed["common_mirrored"] = batch["common_mirrored"].clone()
    changed["common_original"][:, :, 288:304] += 5.0
    changed["common_mirrored"][:, :, 304:320] -= 5.0
    observed = model(**changed)["expert_logits"]
    c.true(torch.equal(baseline[:, 0], observed[:, 0]),
           "global participant Landmark-110 remains an independent expert")
    c.true(not torch.allclose(baseline[:, 1], observed[:, 1]),
           "the action-landmark expert consumes per-task Landmark-110 geometry")


def test_group_dro_optimizes_worst_source_class_groups(c: Check):
    logits = torch.tensor((0.0, -2.0, 2.0, 0.5), dtype=torch.float32)
    labels = torch.tensor((0.0, 1.0, 1.0, 0.0), dtype=torch.float32)
    groups = torch.tensor((0, 1, 2, 3), dtype=torch.long)
    weights = torch.full((4,), 0.25, dtype=torch.float32)
    loss, updated = group_dro_binary_loss(
        logits, labels, groups, weights, step_size=0.2
    )
    c.true(bool(torch.isfinite(loss)), "GroupDRO loss is finite")
    c.true(torch.allclose(updated.sum(), torch.tensor(1.0)),
           "updated robust group weights are normalized")
    c.true(updated[1] == updated.max(),
           "the hardest source-class group receives the largest weight")


def test_control_alignment_removes_source_shift_without_using_patient_labels(c: Check):
    generator = torch.Generator().manual_seed(31)
    base = torch.randn(6, 5, 8, generator=generator)
    labels = torch.tensor((0, 0, 0, 0, 1, 1), dtype=torch.float32)
    sources = torch.tensor((0, 0, 1, 1, 0, 1), dtype=torch.long)
    available = torch.ones((6, 5), dtype=torch.bool)
    aligned = base.clone()
    aligned[2:4] = aligned[:2]
    zero = healthy_control_alignment_loss(aligned, labels, sources, available)
    shifted = aligned.clone()
    shifted[2:4, :, :4] += 2.0
    positive = healthy_control_alignment_loss(shifted, labels, sources, available)
    changed_patients = shifted.clone()
    changed_patients[4:] += 1000.0
    same = healthy_control_alignment_loss(
        changed_patients, labels, sources, available
    )
    c.true(torch.allclose(zero, torch.zeros_like(zero), atol=1e-7),
           "matched healthy controls have zero cross-source moment loss")
    c.true(positive > 0.0, "a healthy acquisition shift is penalized")
    c.true(torch.equal(positive, same),
           "affected rows cannot influence healthy-control alignment")


def test_fixed_fusion_is_source_blind_bounded_and_missing_safe(c: Check):
    logits = torch.tensor((
        (-2.0, 3.0, 1000.0, -1000.0),
        (1.0, -1.0, 0.5, -0.5),
    ))
    available = torch.tensor(((1, 1, 0, 0), (1, 1, 1, 1)), dtype=torch.bool)
    reliability = torch.tensor(((1.0, 0.8, 0.0, 0.0), (1.0, 0.8, 0.5, 0.2)))
    for rule in FIXED_FUSION_RULES:
        first = fixed_source_blind_fusion(
            logits, available, reliability, rule=rule
        )
        changed = logits.clone()
        changed[0, 2:] *= -1.0
        second = fixed_source_blind_fusion(
            changed, available, reliability, rule=rule
        )
        c.eq(first.shape, (2,), f"{rule} returns one probability per participant")
        c.true(bool(torch.all((first >= 0.0) & (first <= 1.0))),
               f"{rule} returns bounded probabilities")
        c.true(torch.equal(first[:1], second[:1]),
               f"{rule} ignores unavailable expert values exactly")


def test_fixed_fusion_rules_have_frozen_distinct_semantics(c: Check):
    logits = torch.tensor(((-1.0, 2.0, 0.0, 1.0),))
    available = torch.ones((1, 4), dtype=torch.bool)
    reliability = torch.tensor(((1.0, 0.75, 0.5, 0.25),))
    maximum = fixed_source_blind_fusion(
        logits, available, reliability, rule="max_probability"
    )
    noisy_or = fixed_source_blind_fusion(
        logits, available, reliability, rule="reliability_noisy_or"
    )
    confidence = fixed_source_blind_fusion(
        logits, available, reliability, rule="confidence_weighted"
    )
    c.true(noisy_or > maximum > confidence,
           "the frozen OR, max and confidence-weighted rules are not aliases")


if __name__ == "__main__":
    run_all("test_universal_phenotype_model_v3", dict(globals()))
