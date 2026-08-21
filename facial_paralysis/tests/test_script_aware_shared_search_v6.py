from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all
from test_medically_gated_shared_search_v2 import _dataset

from src.evaluation.script_aware_shared_search_v6 import evaluate_script_aware_candidate
from src.models.script_aware_shared_router_v6 import candidate_registry_v6


def test_evaluation_is_participant_disjoint_and_shared_gradient_complete(c):
    result = evaluate_script_aware_candidate(
        _dataset(), candidate_registry_v6()[0], epochs=1, n_splits=2,
        seed=0, device="cpu",
    )
    c.eq(result.probabilities.shape, (12,))
    c.eq(result.model_fits, 2)
    c.eq(result.shared_gradient_sources, ("palsynet", "neuroface", "meei"))
    c.eq(len(result.gradient_cosines), 3)


if __name__ == "__main__":
    run_all("test_script_aware_shared_search_v6", dict(globals()))
