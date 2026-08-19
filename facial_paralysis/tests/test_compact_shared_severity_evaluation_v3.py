from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all

from src.evaluation.compact_shared_severity_v3 import (
    evaluate_compact_candidate,
    rank_compact_results,
)
from src.evaluation.medically_gated_shared_search_v2 import MedicalSharedDatasetV2
from src.evaluation.shared_clinical_encoder_v1 import (
    SOURCES,
    SharedClinicalDataset,
    SharedEvaluation,
)
from src.models.compact_shared_severity_v3 import compact_candidate_registry


def _dataset(per_cell: int = 2) -> MedicalSharedDatasetV2:
    rng = np.random.default_rng(83)
    labels, sources, groups = [], [], []
    for source in SOURCES:
        for label in (0, 1):
            for _ in range(per_cell):
                sources.append(source)
                labels.append(label)
                groups.append(f"grp_{len(groups):064x}")
    count, actions = len(labels), 3
    signal = np.asarray(labels, dtype=np.float32)[:, None, None]
    clinical = rng.normal(0, 0.2, (count, actions, 110)).astype(np.float32)
    clinical += signal * 0.3
    dense = rng.normal(0, 0.02, (count, actions, 32, 478, 3)).astype(np.float32)
    dense += signal[..., None, None] * 0.015
    base = SharedClinicalDataset(
        clinical_original=clinical,
        clinical_mirrored=clinical.copy(),
        dense_original=dense,
        dense_mirrored=dense.copy(),
        dense_valid_mask=np.ones((count, actions, 32), dtype=bool),
        dense_available=np.ones((count, actions), dtype=bool),
        action_mask=np.ones((count, actions), dtype=bool),
        action_codes=np.tile(np.arange(actions, dtype=np.int64), (count, 1)),
        labels=np.asarray(labels, dtype=np.int64),
        group_ids=tuple(groups),
        sources=tuple(sources),
    )
    timestamps = np.tile(
        np.arange(32, dtype=np.float64)[None, None, :] / 30.0,
        (count, actions, 1),
    )
    return MedicalSharedDatasetV2(base=base, dense_timestamps=timestamps)


def test_compact_evaluation_is_participant_disjoint_and_shared(c):
    result = evaluate_compact_candidate(
        _dataset(), compact_candidate_registry()[0], epochs=2,
        n_splits=2, seed=0, device="cpu",
    )
    c.eq(result.probabilities.shape, (12,))
    c.true(np.isfinite(result.probabilities).all())
    c.eq(set(result.metrics), set(SOURCES))
    c.eq(result.shared_gradient_sources, SOURCES)
    c.eq(result.model_fits, 2)


def test_ranking_prioritizes_worst_source_not_mean(c):
    registry = compact_candidate_registry()
    results = {}
    for index, candidate in enumerate(registry):
        accuracy = 0.60 + index * 0.01
        metrics = {
            source: {
                "accuracy": accuracy,
                "auroc": 0.65 + index * 0.01,
                "balanced_accuracy": accuracy,
                "sensitivity": accuracy,
                "specificity": accuracy,
                "brier": 1.0 - accuracy,
            }
            for source in SOURCES
        }
        results[candidate.candidate_id] = SharedEvaluation(
            probabilities=np.full(12, accuracy, dtype=np.float64),
            metrics=metrics,
            model_fits=2,
            threshold=0.5,
            shared_gradient_sources=SOURCES,
        )
    ranking = rank_compact_results(results)
    c.eq(ranking[0], registry[-1].candidate_id)
    c.eq(set(ranking), {candidate.candidate_id for candidate in registry})


if __name__ == "__main__":
    run_all("test_compact_shared_severity_evaluation_v3", dict(globals()))
