"""Contracts for exploratory NeuroFace motion-quality pretraining v1."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
from src.models.neuroface_motion_pretrain_v1 import (  # noqa: E402
    MotionQualityRegressor,
    count_parameters,
    within_window_velocity,
)
from src.training.neuroface_motion_pretrain_v1 import (  # noqa: E402
    MotionPretrainConfig,
    TransferDataset,
    build_aggregate_report,
    build_stratified_participant_folds,
    evaluate_frozen_palsynet_transfer,
)


def _model_batch(n: int = 3):
    generator = torch.Generator().manual_seed(813)
    landmarks = torch.randn(n, 4, 32, 23, generator=generator)
    mask = torch.ones(n, 4, 32, dtype=torch.bool)
    mask[0, 0, -3:] = False
    timestamps = torch.arange(32, dtype=torch.float32).view(1, 1, 32).repeat(n, 4, 1)
    timestamps += torch.arange(4, dtype=torch.float32).view(1, 4, 1) * 100.0
    tasks = torch.arange(n) % 9
    return landmarks, mask, timestamps, tasks


def test_encoder_is_compact_masked_and_emits_fixed_embedding(c: Check):
    landmarks, mask, timestamps, tasks = _model_batch()
    dirty = landmarks.clone()
    dirty[~mask] = 100_000.0
    clean = landmarks.clone()
    clean[~mask] = 0.0
    torch.manual_seed(0)
    model = MotionQualityRegressor().eval()
    c.true(0 < count_parameters(model) < 30_000)
    with torch.no_grad():
        dirty_logits, dirty_embedding = model(dirty, mask, timestamps, tasks)
        clean_logits, clean_embedding = model(clean, mask, timestamps, tasks)
    c.eq(tuple(dirty_logits.shape), (3, 5))
    c.eq(tuple(dirty_embedding.shape), (3, 32))
    c.true(bool(torch.isfinite(dirty_logits).all()))
    c.true(bool(torch.allclose(dirty_logits, clean_logits, atol=1e-6, rtol=1e-6)))
    c.true(bool(torch.allclose(dirty_embedding, clean_embedding, atol=1e-6, rtol=1e-6)))


def test_velocity_never_crosses_the_four_window_gaps(c: Check):
    values = torch.zeros(1, 4, 32, 23)
    values[:, 1] = 10.0
    values[:, 2] = 20.0
    values[:, 3] = 30.0
    mask = torch.ones(1, 4, 32, dtype=torch.bool)
    timestamps = torch.arange(32, dtype=torch.float32).view(1, 1, 32).repeat(1, 4, 1)
    timestamps += torch.arange(4, dtype=torch.float32).view(1, 4, 1) * 1_000.0
    velocity = within_window_velocity(values, mask, timestamps)
    c.true(bool(torch.allclose(velocity, torch.zeros_like(velocity))),
           "constant windows cannot acquire artificial gap velocities")
    values[:, :, 1:, 0] += 2.0
    mask[:, 2, 1] = False
    velocity = within_window_velocity(values, mask, timestamps)
    c.eq(float(velocity[0, 0, 1, 0]), 2.0)
    c.eq(float(velocity[0, 2, 1, 0]), 0.0)
    c.eq(float(velocity[0, 2, 2, 0]), 0.0)


def test_six_folds_are_deterministic_cohort_stratified_and_group_disjoint(c: Check):
    groups = np.asarray([f"g{index}" for index in range(36)], dtype=object)
    cohorts = np.asarray(["als"] * 11 + ["healthy_control"] * 11 + ["post_stroke"] * 14)
    first = build_stratified_participant_folds(groups, cohorts, folds=6, seed=20260813)
    second = build_stratified_participant_folds(groups, cohorts, folds=6, seed=20260813)
    c.true(bool(np.array_equal(first, second)))
    c.eq(set(first.tolist()), set(range(6)))
    for fold in range(6):
        c.eq(set(cohorts[first == fold].tolist()), {"als", "healthy_control", "post_stroke"})
        c.eq(set(cohorts[first != fold].tolist()), {"als", "healthy_control", "post_stroke"})


def test_transfer_comparison_never_observes_protected_rows(c: Check):
    rng = np.random.default_rng(813)
    n_dev, n_protected = 16, 2
    n = n_dev + n_protected
    labels = np.asarray([index % 2 for index in range(n)], dtype=np.int64)
    summary = rng.normal(size=(n, 110))
    motion = rng.normal(size=(n, 32))
    summary[labels == 1, :3] += 2.0
    motion[labels == 1, :2] += 1.0
    protected = np.arange(n_dev, n)
    summary[protected] = np.nan
    motion[protected] = np.nan
    folds = np.full(n, -1, dtype=np.int64)
    folds[:n_dev] = np.repeat(np.arange(4), 4)
    dataset = TransferDataset(
        summary_features=summary,
        mirrored_summary_features=summary.copy(),
        motion_features=motion,
        mirrored_motion_features=motion.copy(),
        labels=labels,
        group_ids=np.asarray([f"p{index}" for index in range(n)], dtype=object),
        development_indices=np.arange(n_dev),
        protected_indices=protected,
        inner_fold_by_index=folds,
    )
    result = evaluate_frozen_palsynet_transfer(dataset)
    c.eq(tuple(result.metrics), ("landmark_110d", "motion_32d", "landmark_110d_plus_motion_32d"))
    c.eq(result.protected_predictions, 0)
    c.eq(result.development_groups, n_dev)
    for values in result.metrics.values():
        c.true(0.0 <= values["auroc"] <= 1.0)
        c.true(0.0 <= values["balanced_accuracy"] <= 1.0)
        c.true(0.0 <= values["brier"] <= 1.0)


def test_config_and_public_report_are_locked_and_identifier_free(c: Check):
    config = MotionPretrainConfig()
    c.eq((config.folds, config.seed, config.epochs, config.patience),
         (6, 20260813, 80, 10))
    c.raises(lambda: MotionPretrainConfig(epochs=81), ValueError)
    result = type("Transfer", (), {
        "metrics": {
            "landmark_110d": {"auroc": 0.98, "balanced_accuracy": 0.95, "brier": 0.06},
            "motion_32d": {"auroc": 0.70, "balanced_accuracy": 0.65, "brier": 0.20},
            "landmark_110d_plus_motion_32d": {
                "auroc": 0.981, "balanced_accuracy": 0.95, "brier": 0.059,
            },
        },
        "development_recordings": 38,
        "development_groups": 38,
        "protected_predictions": 0,
    })()
    report = build_aggregate_report(
        pretrain_metrics={
            "domains": {name: {"spearman": 0.2, "mae": 0.4}
                        for name in ("symmetry", "rom", "speed", "variability", "fatigue")},
            "participant_macro_mae": 0.4,
            "best_epochs": [10, 11, 12, 13, 14, 15],
            "final_epochs": 12,
        },
        transfer=result,
        provenance={name: character * 64 for name, character in {
            "neuroface_private_manifest_sha256": "a",
            "neuroface_cache_collection_sha256": "b",
            "palsynet_cache_collection_sha256": "c",
            "palsynet_reviewed_manifest_sha256": "d",
            "palsynet_review_ledger_sha256": "e",
            "palsynet_split_registry_sha256": "f",
            "implementation_sha256": "1",
            "dependency_lock_sha256": "2",
        }.items()},
        runtime={"host": "nebius-h200", "device": "cpu", "seconds": 1.0},
        parameter_count=7000,
    )
    c.eq(report["claim_scope"], "exploratory_neuroface_pretraining_palsynet_development_oof_only")
    c.eq(report["audit"]["protected_predictions"], 0)
    c.eq(report["decision"]["outer_evaluation_authorized"], False)
    c.eq(report["decision"]["current_model_replaced"], True)
    encoded = str(report).lower()
    c.true("group_id" not in encoded and "recording_id" not in encoded and "grp_" not in encoded)


if __name__ == "__main__":
    run_all("test_neuroface_motion_pretrain_v1", dict(globals()))
