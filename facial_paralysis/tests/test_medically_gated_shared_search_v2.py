from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all

from src.evaluation.medically_gated_shared_search_v2 import (
    MedicalSharedDatasetV2,
    evaluate_medical_candidate,
    rank_candidate_results,
)
from src.evaluation.shared_clinical_encoder_v1 import (
    SOURCES,
    SharedClinicalDataset,
    SharedEvaluation,
)
from src.models.medical_shared_candidate_registry_v2 import candidate_registry


def _dataset(per_cell: int = 2) -> MedicalSharedDatasetV2:
    rng = np.random.default_rng(37)
    labels, sources, groups = [], [], []
    for source in SOURCES:
        for label in (0, 1):
            for _ in range(per_cell):
                sources.append(source)
                labels.append(label)
                groups.append(f"grp_{len(groups):064x}")
    count, actions = len(labels), 3
    label_values = np.asarray(labels, dtype=np.float32)[:, None, None]
    clinical = rng.normal(0, 0.2, (count, actions, 110)).astype(np.float32)
    clinical += label_values * 0.3
    mirrored = clinical.copy()
    mirrored[:, :, 0] *= -1
    dense = rng.normal(0, 0.02, (count, actions, 32, 478, 3)).astype(np.float32)
    dense += label_values[..., None, None] * 0.015
    dense_mirror = dense.copy()
    dense_mirror[..., 0] *= -1
    valid = np.ones((count, actions, 32), dtype=bool)
    available = np.ones((count, actions), dtype=bool)
    action_mask = np.ones((count, actions), dtype=bool)
    action_codes = np.tile(np.arange(actions, dtype=np.int64), (count, 1))
    base = SharedClinicalDataset(
        clinical_original=clinical,
        clinical_mirrored=mirrored,
        dense_original=dense,
        dense_mirrored=dense_mirror,
        dense_valid_mask=valid,
        dense_available=available,
        action_mask=action_mask,
        action_codes=action_codes,
        labels=np.asarray(labels, dtype=np.int64),
        group_ids=tuple(groups),
        sources=tuple(sources),
    )
    timestamps = np.tile(
        np.arange(32, dtype=np.float64)[None, None, :] / 30.0,
        (count, actions, 1),
    )
    return MedicalSharedDatasetV2(base=base, dense_timestamps=timestamps)


def test_timestamp_contract_is_exact_immutable_and_dense_bound(c):
    dataset = _dataset()
    c.eq(dataset.dense_timestamps.dtype, np.dtype(np.float64))
    c.eq(dataset.dense_timestamps.shape, dataset.base.dense_valid_mask.shape)
    c.true(not dataset.dense_timestamps.flags.writeable)
    changed = dataset.dense_timestamps.copy()
    changed[0, 0, 5] = changed[0, 0, 4]
    c.raises(
        lambda: MedicalSharedDatasetV2(dataset.base, changed), ValueError
    )


def test_candidate_evaluation_is_group_disjoint_and_source_complete(c):
    result = evaluate_medical_candidate(
        _dataset(), candidate_registry()[0], epochs=1, n_splits=2,
        seed=0, device="cpu",
    )
    c.eq(result.probabilities.shape, (12,))
    c.true(np.isfinite(result.probabilities).all())
    c.eq(set(result.metrics), set(SOURCES))
    c.eq(result.shared_gradient_sources, SOURCES)
    c.eq(result.model_fits, 2)


def test_ranking_uses_worst_source_accuracy_then_auc_then_mean(c):
    registry = candidate_registry()
    results = {}
    for index, candidate in enumerate(registry):
        accuracy = 0.60 + index * 0.005
        metrics = {
            source: {
                "accuracy": accuracy,
                "auroc": 0.70 + index * 0.002,
                "balanced_accuracy": accuracy,
                "sensitivity": accuracy,
                "specificity": accuracy,
                "brier": 1.0 - accuracy,
            }
            for source in SOURCES
        }
        results[candidate.candidate_id] = SharedEvaluation(
            probabilities=np.full(6, accuracy, dtype=np.float64),
            metrics=metrics,
            model_fits=2,
            threshold=0.5,
            shared_gradient_sources=SOURCES,
        )
    ranked = rank_candidate_results(results)
    c.eq(ranked[:4], tuple(candidate.candidate_id for candidate in registry[-1:-5:-1]))
    c.eq(set(ranked), {candidate.candidate_id for candidate in registry})


if __name__ == "__main__":
    run_all("test_medically_gated_shared_search_v2", dict(globals()))
