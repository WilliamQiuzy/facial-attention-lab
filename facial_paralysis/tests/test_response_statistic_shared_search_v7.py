from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all
from test_medically_gated_shared_search_v2 import _dataset

from src.evaluation.response_statistic_shared_search_v7 import evaluate_response_statistic_candidate
from src.models.response_statistic_shared_router_v7 import candidate_registry_v7


def test_evaluation_is_fold_local_group_disjoint_and_complete(c):
    result = evaluate_response_statistic_candidate(
        _dataset(), candidate_registry_v7()[0], epochs=1, n_splits=2,
        seed=0, device="cpu",
    )
    c.eq(result.probabilities.shape, (12,))
    c.eq(result.model_fits, 2)
    c.eq(result.shared_gradient_sources, ("palsynet", "neuroface", "meei"))


if __name__ == "__main__":
    run_all("test_response_statistic_shared_search_v7", dict(globals()))
