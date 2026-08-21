from __future__ import annotations

import numpy as np

from _testlib import run_all

from src.evaluation.residual_shared_search_v8 import ResidualEvaluationV8
from src.evaluation.shared_ensemble_search_v9 import (
    EnsembleEvaluationV9,
    ensemble_candidate_registry_v9,
    evaluate_ensemble_candidate,
    rank_ensemble_results,
)


SOURCES = ("palsynet", "neuroface", "meei")


def _base_result(offset: float):
    probabilities = np.asarray([
        0.1, 0.8, 0.2, 0.7,
        0.2, 0.9, 0.3, 0.8,
        0.1, 0.7, 0.4, 0.9,
    ], dtype=np.float64) + offset
    return ResidualEvaluationV8(
        probabilities=probabilities,
        metrics={},
        model_fits=2,
        threshold=0.5,
        shared_gradient_sources=SOURCES,
    )


def _base_results():
    ids = ("RSR8-000", "RSR8-001", "RSR8-002", "RSR8-003", "RSR8-004", "RSR8-005")
    return {
        (candidate_id, seed): _base_result(0.001 * (index + seed))
        for index, candidate_id in enumerate(ids)
        for seed in (0, 1, 2)
    }


def _labels_sources():
    labels = np.asarray([0, 1] * 6, dtype=np.int64)
    sources = tuple(source for source in SOURCES for _ in range(4))
    return labels, sources


def test_registry_is_exact_closed_and_contains_single_v8_comparator(c):
    registry = ensemble_candidate_registry_v9()
    c.eq(len(registry), 16)
    c.eq(tuple(row.candidate_id for row in registry), tuple(
        f"SEN9-{index:03d}" for index in range(16)
    ))
    c.eq({row.aggregation for row in registry}, {"probability_mean", "logit_mean"})
    c.eq({row.seeds for row in registry}, {(0,), (0, 1, 2)})
    c.eq(registry[0].member_candidate_ids, ("RSR8-001",))
    c.eq(registry[0].seeds, (0,))
    c.eq(registry[0].aggregation, "probability_mean")


def test_single_member_comparator_is_exact_and_ensembles_are_finite(c):
    labels, sources = _labels_sources()
    registry = ensemble_candidate_registry_v9()
    base = _base_results()
    comparator = evaluate_ensemble_candidate(labels, sources, base, registry[0])
    c.true(np.array_equal(
        comparator.probabilities, base[("RSR8-001", 0)].probabilities
    ))
    ensemble = evaluate_ensemble_candidate(labels, sources, base, registry[-1])
    c.true(np.isfinite(ensemble.probabilities).all())
    c.eq(set(ensemble.metrics), set(SOURCES))
    c.true(ensemble.member_models > 1)


def test_ranking_requires_full_registry_and_sensitivity_floor(c):
    labels, sources = _labels_sources()
    registry = ensemble_candidate_registry_v9()
    base = _base_results()
    results = {
        row.candidate_id: evaluate_ensemble_candidate(labels, sources, base, row)
        for row in registry
    }
    c.eq(set(rank_ensemble_results(results)), {row.candidate_id for row in registry})
    incomplete = dict(results)
    incomplete.pop(registry[-1].candidate_id)
    c.raises(lambda: rank_ensemble_results(incomplete), ValueError)
    c.true(all(type(value) is EnsembleEvaluationV9 for value in results.values()))


if __name__ == "__main__":
    run_all("test_shared_ensemble_search_v9", dict(globals()))
