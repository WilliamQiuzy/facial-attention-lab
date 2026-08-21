from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all
from test_medically_gated_shared_search_v2 import _dataset

from src.evaluation.conflict_aware_shared_search_v5 import (
    evaluate_conflict_aware_candidate,
    project_conflicting_gradient_vectors,
)
from src.models.conflict_aware_candidate_registry_v5 import candidate_registry_v5


def test_projection_removes_only_negative_component(c):
    first = torch.tensor([1.0, 0.0])
    opposing = torch.tensor([-1.0, 1.0])
    aligned = torch.tensor([1.0, 1.0])
    projected = project_conflicting_gradient_vectors(
        (first, opposing), strength=1.0
    )
    c.true(float(torch.dot(projected[0], opposing)) >= -1e-6)
    unchanged = project_conflicting_gradient_vectors((first, aligned), strength=1.0)
    c.true(torch.equal(unchanged[0], first))
    c.true(torch.equal(unchanged[1], aligned))


def test_half_projection_is_between_original_and_full(c):
    vectors = (torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 1.0]))
    half = project_conflicting_gradient_vectors(vectors, strength=0.5)
    full = project_conflicting_gradient_vectors(vectors, strength=1.0)
    c.true(float(torch.linalg.vector_norm(half[0] - vectors[0])) > 0.0)
    c.true(float(torch.linalg.vector_norm(full[0] - vectors[0])) > float(torch.linalg.vector_norm(half[0] - vectors[0])))


def test_real_evaluation_keeps_participant_disjoint_outputs(c):
    result = evaluate_conflict_aware_candidate(
        _dataset(), candidate_registry_v5()[0], epochs=1, n_splits=2,
        seed=0, device="cpu",
    )
    c.eq(result.probabilities.shape, (12,))
    c.eq(result.model_fits, 2)
    c.eq(len(result.pre_projection_cosines), 3)
    c.eq(len(result.post_projection_cosines), 3)


if __name__ == "__main__":
    run_all("test_conflict_aware_shared_search_v5", dict(globals()))
