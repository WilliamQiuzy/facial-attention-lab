"""MMoE-inspired shared physiological expert bank for heterogeneous scripts."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .residual_shared_router_v8 import (
    ResidualSharedRouterV8,
    candidate_registry_v8,
)


@dataclass(frozen=True)
class MultiGateSharedExpertCandidateV9:
    candidate_id: str
    shared_expert_count: int
    expert_rank: int
    paper_basis: str
    medical_rationale: str


def candidate_registry_v9() -> tuple[MultiGateSharedExpertCandidateV9, ...]:
    return (
        MultiGateSharedExpertCandidateV9(
            candidate_id="MSE9-000",
            shared_expert_count=0,
            expert_rank=0,
            paper_basis="deterministic RSR8-001 comparator",
            medical_rationale="Exact V8 shared-encoder control.",
        ),
        MultiGateSharedExpertCandidateV9(
            candidate_id="MSE9-001",
            shared_expert_count=3,
            expert_rank=16,
            paper_basis="Multi-gate Mixture-of-Experts, KDD 2018",
            medical_rationale=(
                "All scripts reuse one bank of three shared motor experts while "
                "small task gates learn different mixtures for heterogeneous "
                "Bell-palsy, ALS/post-stroke, and peripheral-palsy protocols."
            ),
        ),
    )


def _locked_v8_candidate():
    matched = tuple(
        row for row in candidate_registry_v8() if row.candidate_id == "RSR8-001"
    )
    if len(matched) != 1:
        raise RuntimeError("the exact V8 comparator is unavailable")
    return matched[0]


class MultiGateSharedExpertRouterV9(nn.Module):
    """Keep V8 shared action physiology and mix a shared patient-expert bank."""

    def __init__(self, candidate: MultiGateSharedExpertCandidateV9):
        super().__init__()
        if (
            type(candidate) is not MultiGateSharedExpertCandidateV9
            or candidate not in candidate_registry_v9()
        ):
            raise ValueError("multi-gate V9 requires one frozen candidate")
        self.candidate = candidate
        self.base = ResidualSharedRouterV8(_locked_v8_candidate())
        if candidate.shared_expert_count == 0:
            return
        self.shared_experts = nn.ModuleList()
        for _ in range(candidate.shared_expert_count):
            expert = nn.Sequential(
                nn.LayerNorm(64),
                nn.Linear(64, candidate.expert_rank),
                nn.GELU(),
                nn.Linear(candidate.expert_rank, 64),
            )
            nn.init.normal_(expert[-1].weight, mean=0.0, std=0.01)
            nn.init.zeros_(expert[-1].bias)
            self.shared_experts.append(expert)
        self.task_gates = nn.ModuleList(
            nn.Linear(64, candidate.shared_expert_count) for _ in range(3)
        )

    def shared_action_tokens(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self.base.shared_action_tokens(*inputs)

    def task_specific_parameter_fraction(self) -> float:
        task_specific = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name.startswith("base.adapters")
            or name.startswith("base.base.task_queries")
            or name.startswith("base.base.backbone.task_heads")
            or name.startswith("task_gates")
        )
        return task_specific / sum(parameter.numel() for parameter in self.parameters())

    def _mixed_endpoint(
        self,
        embedding: torch.Tensor,
        task_codes: torch.Tensor,
    ) -> torch.Tensor:
        endpoint = self.base.adapt_endpoint(embedding, task_codes)
        if self.candidate.shared_expert_count == 0:
            return endpoint
        expert_values = torch.stack(
            [expert(endpoint) for expert in self.shared_experts], dim=1
        )
        all_gates = torch.stack(
            [gate(endpoint) for gate in self.task_gates], dim=1
        )
        selected_gates = all_gates.gather(
            1,
            task_codes[:, None, None].expand(
                -1, 1, self.candidate.shared_expert_count
            ),
        ).squeeze(1)
        weights = torch.softmax(selected_gates, dim=-1)
        residual = torch.sum(expert_values * weights.unsqueeze(-1), dim=1)
        return endpoint + 0.5 * residual

    def routed_and_universal_logits(
        self,
        tokens: torch.Tensor,
        action_mask: torch.Tensor,
        task_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        common = self.base.base.endpoint_embedding(tokens, action_mask, task_codes)
        endpoint = self._mixed_endpoint(common, task_codes)
        universal_embedding = self.base.base.universal_embedding(tokens, action_mask)
        task = self.base.base.task_logits_from_embedding(endpoint, task_codes)
        universal = self.base.base.universal_head(universal_embedding).squeeze(-1)
        return 0.75 * task + 0.25 * universal, universal

    def routed_logits(
        self,
        tokens: torch.Tensor,
        action_mask: torch.Tensor,
        task_codes: torch.Tensor,
    ) -> torch.Tensor:
        routed, _ = self.routed_and_universal_logits(tokens, action_mask, task_codes)
        return routed

    def universal_logits(
        self,
        tokens: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.base.base.universal_embedding(tokens, action_mask)
        return self.base.base.universal_head(embedding).squeeze(-1)


__all__ = [
    "MultiGateSharedExpertCandidateV9",
    "MultiGateSharedExpertRouterV9",
    "candidate_registry_v9",
]
