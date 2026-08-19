from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all
from test_shared_normal_manifold_router_v4 import _inputs

from src.models.residual_shared_router_v8 import ResidualSharedRouterV8, candidate_registry_v8


def test_registry_and_parameter_sharing_contract(c):
    registry=candidate_registry_v8(); c.eq(len(registry),6)
    c.eq({x.adapter_rank for x in registry},{8,16,32}); c.eq({x.residual_scale for x in registry},{0.25,0.5})
    model=ResidualSharedRouterV8(registry[-1])
    task_specific=sum(p.numel() for n,p in model.named_parameters() if n.startswith("adapters") or n.startswith("base.task_queries") or n.startswith("base.backbone.task_heads"))
    total=sum(p.numel() for p in model.parameters())
    c.true(task_specific/total<0.10)


def test_residual_operates_only_after_shared_patient_embedding(c):
    model=ResidualSharedRouterV8(candidate_registry_v8()[0]).eval(); inputs=_inputs(); tasks=torch.tensor([0,1,2,0,1,2])
    with torch.no_grad():
        tokens=model.shared_action_tokens(*inputs)
        common=model.base.endpoint_embedding(tokens,inputs[-2],tasks)
        adapted=model.adapt_endpoint(common,tasks)
    c.eq(common.shape,adapted.shape); c.true(not torch.equal(common,adapted))
    c.true("tokens" not in model.adapt_endpoint.__code__.co_varnames)


if __name__=="__main__": run_all("test_residual_shared_router_v8",dict(globals()))
