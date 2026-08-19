"""Shared full-mesh core with small post-phenotype endpoint residuals."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .script_aware_shared_router_v6 import ScriptAwareSharedRouterV6, candidate_registry_v6


@dataclass(frozen=True)
class ResidualCandidateV8:
    candidate_id: str
    adapter_rank: int
    residual_scale: float


def candidate_registry_v8():
    rows=[]; index=0
    for rank in (8,16,32):
        for scale in (0.25,0.5):
            rows.append(ResidualCandidateV8(f"RSR8-{index:03d}",rank,scale)); index+=1
    return tuple(rows)


def _base_candidate():
    return next(item for item in candidate_registry_v6() if item.candidate_id=="SAR6-002")


class ResidualSharedRouterV8(nn.Module):
    def __init__(self,candidate:ResidualCandidateV8):
        super().__init__()
        if type(candidate) is not ResidualCandidateV8 or candidate not in candidate_registry_v8(): raise ValueError("v8 requires frozen candidate")
        self.candidate=candidate; self.base=ScriptAwareSharedRouterV6(_base_candidate())
        self.adapters=nn.ModuleList()
        for _ in range(3):
            adapter=nn.Sequential(nn.LayerNorm(64),nn.Linear(64,candidate.adapter_rank),nn.GELU(),nn.Linear(candidate.adapter_rank,64))
            nn.init.normal_(adapter[-1].weight,std=0.01); nn.init.zeros_(adapter[-1].bias)
            self.adapters.append(adapter)

    def shared_action_tokens(self,*inputs): return self.base.shared_action_tokens(*inputs)

    def adapt_endpoint(self,embedding,task_codes):
        if embedding.ndim!=2 or embedding.shape[1]!=64 or task_codes.shape!=(embedding.shape[0],): raise ValueError("adapter requires shared patient embeddings")
        adapted=torch.stack([
            embedding+self.candidate.residual_scale*adapter(embedding)
            for adapter in self.adapters
        ],1)
        return adapted.gather(1,task_codes[:,None,None].expand(-1,1,64)).squeeze(1)

    def routed_logits(self,tokens,action_mask,task_codes):
        common=self.base.endpoint_embedding(tokens,action_mask,task_codes)
        endpoint=self.adapt_endpoint(common,task_codes)
        universal=self.base.universal_embedding(tokens,action_mask)
        return 0.75*self.base.task_logits_from_embedding(endpoint,task_codes)+0.25*self.base.universal_head(universal).squeeze(-1)


__all__=["ResidualCandidateV8","ResidualSharedRouterV8","candidate_registry_v8"]
