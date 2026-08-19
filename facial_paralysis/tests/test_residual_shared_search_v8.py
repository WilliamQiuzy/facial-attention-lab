from __future__ import annotations

import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from _testlib import run_all
from test_medically_gated_shared_search_v2 import _dataset
from src.evaluation.residual_shared_search_v8 import evaluate_residual_candidate
from src.models.residual_shared_router_v8 import candidate_registry_v8


def test_evaluation_is_complete(c):
    result=evaluate_residual_candidate(_dataset(),candidate_registry_v8()[0],epochs=1,n_splits=2,seed=0,device="cpu")
    c.eq(result.probabilities.shape,(12,)); c.eq(result.model_fits,2)
    c.eq(result.shared_gradient_sources,("palsynet","neuroface","meei"))


def test_staged_endpoint_adaptation_preserves_evaluation_contract(c):
    result=evaluate_residual_candidate(_dataset(),candidate_registry_v8()[1],epochs=1,adapter_epochs=1,n_splits=2,seed=0,device="cpu")
    c.eq(result.probabilities.shape,(12,)); c.eq(result.model_fits,2)


if __name__=="__main__": run_all("test_residual_shared_search_v8",dict(globals()))
