from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all
from test_medically_gated_shared_search_v2 import _dataset

from src.evaluation.shared_clinical_encoder_v1 import SOURCES
from src.evaluation.shared_normal_manifold_search_v4 import (
    NormalManifoldEvaluationV4,
    evaluate_normal_manifold_candidate,
    rank_normal_manifold_results,
)
from src.models.normal_manifold_candidate_registry_v4 import candidate_registry_v4


def test_evaluation_is_complete_and_audits_pairwise_shared_gradients(c):
    result = evaluate_normal_manifold_candidate(
        _dataset(), candidate_registry_v4()[0], epochs=1, n_splits=2,
        seed=0, device="cpu",
    )
    c.eq(result.probabilities.shape, (12,))
    c.eq(set(result.metrics), set(SOURCES))
    c.eq(set(result.gradient_cosines), {
        "palsynet__neuroface", "palsynet__meei", "neuroface__meei"
    })
    c.true(all(np.isfinite(value) for value in result.gradient_cosines.values()))
    c.eq(result.model_fits, 2)


def test_ranking_prefers_worst_balanced_accuracy_then_specificity(c):
    results = {}
    for index, candidate in enumerate(candidate_registry_v4()):
        score = 0.60 + index * 0.02
        metrics = {
            source: {
                "accuracy": score,
                "balanced_accuracy": score,
                "auroc": score + 0.05,
                "sensitivity": score,
                "specificity": score - 0.02,
                "brier": 1.0 - score,
            }
            for source in SOURCES
        }
        results[candidate.candidate_id] = NormalManifoldEvaluationV4(
            probabilities=np.full(6, score), metrics=metrics, model_fits=2,
            threshold=0.5, gradient_cosines={
                "palsynet__neuroface": 0.1,
                "palsynet__meei": 0.2,
                "neuroface__meei": 0.3,
            },
        )
    ranked = rank_normal_manifold_results(results)
    c.eq(ranked[0], candidate_registry_v4()[-1].candidate_id)
    c.eq(set(ranked), {candidate.candidate_id for candidate in candidate_registry_v4()})


if __name__ == "__main__":
    run_all("test_shared_normal_manifold_search_v4", dict(globals()))
