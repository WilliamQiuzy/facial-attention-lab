"""One genuinely shared V8 trunk with four isolated architecture mechanisms."""
from __future__ import annotations

import math

import torch
from torch import nn

from .broad_literature_candidate_registry_v9 import (
    BroadLiteratureCandidateV9,
    candidate_registry_v9,
)
from .dense_clinical_shared_encoder_v1 import ACTION_VOCAB
from .residual_shared_router_v8 import ResidualSharedRouterV8, candidate_registry_v8


_ARCHITECTURE_MECHANISMS = {
    "progressive_layered_extraction",
    "cross_stitch_endpoint_streams",
    "action_conditioned_film",
    "anatomy_action_graph",
}


def _locked_v8_candidate():
    matched = tuple(
        row for row in candidate_registry_v8() if row.candidate_id == "RSR8-001"
    )
    if len(matched) != 1:
        raise RuntimeError("the exact V8 comparator is unavailable")
    return matched[0]


def _low_rank_block(rank: int) -> nn.Sequential:
    block = nn.Sequential(
        nn.LayerNorm(64),
        nn.Linear(64, rank),
        nn.GELU(),
        nn.Linear(rank, 64),
    )
    nn.init.normal_(block[-1].weight, mean=0.0, std=0.01)
    nn.init.zeros_(block[-1].bias)
    return block


def _action_regions() -> tuple[int, ...]:
    oral = {
        "LIP_PUCKER", "MOUTH_OPEN", "SMILE_SPREAD", "SMILE_GENTLE",
        "SMILE_FULL", "SHOW_BOTTOM_TEETH",
    }
    eye = {"EYE_GENTLE", "EYE_FORCEFUL"}
    result = []
    for name in ACTION_VOCAB:
        if name.startswith("FREE_"):
            result.append(0)
        elif name in oral:
            result.append(1)
        elif name == "BROW_RAISE":
            result.append(2)
        elif name in eye:
            result.append(3)
        else:
            raise RuntimeError("the action anatomy ontology is incomplete")
    return tuple(result)


class BroadLiteratureSharedRouterV9(nn.Module):
    """Isolate one architecture change while preserving the exact V8 trunk."""

    def __init__(self, candidate: BroadLiteratureCandidateV9):
        super().__init__()
        if (
            type(candidate) is not BroadLiteratureCandidateV9
            or candidate not in candidate_registry_v9()
        ):
            raise ValueError("broad V9 requires one exact frozen candidate")
        self.candidate = candidate
        self.base = ResidualSharedRouterV8(_locked_v8_candidate())
        mechanism = candidate.mechanism
        if mechanism == "progressive_layered_extraction":
            self.shared_experts = nn.ModuleList(_low_rank_block(8) for _ in range(2))
            self.endpoint_experts = nn.ModuleList(_low_rank_block(4) for _ in range(3))
            self.ple_gates = nn.ModuleList(nn.Linear(64, 3) for _ in range(3))
        elif mechanism == "cross_stitch_endpoint_streams":
            self.shared_stream = _low_rank_block(8)
            self.endpoint_streams = nn.ModuleList(_low_rank_block(8) for _ in range(3))
            self.cross_stitch = nn.Parameter(torch.tensor(
                [[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]], dtype=torch.float32
            ))
        elif mechanism == "action_conditioned_film":
            self.film_gamma = nn.Embedding(len(ACTION_VOCAB), 64)
            self.film_beta = nn.Embedding(len(ACTION_VOCAB), 64)
            nn.init.normal_(self.film_gamma.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.film_beta.weight, mean=0.0, std=0.02)
        elif mechanism == "anatomy_action_graph":
            self.register_buffer(
                "action_region", torch.tensor(_action_regions(), dtype=torch.long),
                persistent=True,
            )
            self.graph_query = nn.Linear(64, 64, bias=False)
            self.graph_key = nn.Linear(64, 64, bias=False)
            self.graph_value = nn.Linear(64, 64, bias=False)
            self.graph_output = nn.Linear(64, 64)
        elif candidate.inference_change == "architecture":
            raise RuntimeError("an architecture candidate has no isolated implementation")

    def task_specific_parameter_fraction(self) -> float:
        task_prefixes = (
            "base.adapters",
            "base.base.task_queries",
            "base.base.backbone.task_heads",
            "endpoint_experts",
            "ple_gates",
            "endpoint_streams",
            "cross_stitch",
        )
        task_specific = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name.startswith(task_prefixes)
        )
        total = sum(parameter.numel() for parameter in self.parameters())
        if total <= 0:
            raise RuntimeError("broad V9 has no trainable parameters")
        return task_specific / total

    def _film_tokens(
        self,
        tokens: torch.Tensor,
        action_codes: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        gamma = self.film_gamma(action_codes)
        beta = self.film_beta(action_codes)
        modulated = tokens + 0.1 * (gamma * tokens + beta)
        return torch.where(action_mask.unsqueeze(-1), modulated, tokens)

    def _graph_tokens(
        self,
        tokens: torch.Tensor,
        action_codes: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, actions, dimension = tokens.shape
        heads = 4
        head_dim = dimension // heads
        regions = self.action_region.index_select(0, action_codes.reshape(-1)).reshape(
            batch, actions
        )
        same_region = regions[:, :, None] == regions[:, None, :]
        free_relation = (regions[:, :, None] == 0) | (regions[:, None, :] == 0)
        allowed = (same_region | free_relation) & action_mask[:, None, :]
        invalid_query = ~action_mask
        identity = torch.eye(actions, dtype=torch.bool, device=tokens.device)[None]
        allowed = torch.where(invalid_query[:, :, None], identity, allowed)

        def split(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(batch, actions, heads, head_dim).transpose(1, 2)

        query = split(self.graph_query(tokens))
        key = split(self.graph_key(tokens))
        value = split(self.graph_value(tokens))
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)
        scores = scores.masked_fill(~allowed[:, None], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        message = torch.matmul(weights, value).transpose(1, 2).reshape(
            batch, actions, dimension
        )
        residual = 0.1 * self.graph_output(message)
        return tokens + residual * action_mask.unsqueeze(-1).to(tokens.dtype)

    def shared_action_tokens(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) != 9:
            raise ValueError("broad V9 requires the exact nine-input action contract")
        tokens = self.base.shared_action_tokens(*inputs)
        mechanism = self.candidate.mechanism
        if mechanism == "action_conditioned_film":
            return self._film_tokens(tokens, inputs[-1], inputs[-2])
        if mechanism == "anatomy_action_graph":
            return self._graph_tokens(tokens, inputs[-1], inputs[-2])
        return tokens

    @staticmethod
    def _selected_task_values(
        modules: nn.ModuleList,
        embedding: torch.Tensor,
        task_codes: torch.Tensor,
    ) -> torch.Tensor:
        values = torch.stack([module(embedding) for module in modules], dim=1)
        return values.gather(
            1, task_codes[:, None, None].expand(-1, 1, embedding.shape[-1])
        ).squeeze(1)

    def _ple_embeddings(
        self,
        common: torch.Tensor,
        endpoint: torch.Tensor,
        task_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared = torch.stack([expert(common) for expert in self.shared_experts], dim=1)
        task = self._selected_task_values(self.endpoint_experts, endpoint, task_codes)
        all_gates = torch.stack([gate(common) for gate in self.ple_gates], dim=1)
        selected = all_gates.gather(
            1, task_codes[:, None, None].expand(-1, 1, 3)
        ).squeeze(1)
        weights = torch.softmax(selected, dim=-1)
        experts = torch.cat((shared, task[:, None, :]), dim=1)
        endpoint = endpoint + 0.5 * torch.sum(experts * weights.unsqueeze(-1), dim=1)
        universal = common + 0.25 * shared.mean(dim=1)
        return endpoint, universal

    def _cross_stitch_embeddings(
        self,
        common: torch.Tensor,
        endpoint: torch.Tensor,
        task_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared_stream(common)
        task = self._selected_task_values(self.endpoint_streams, endpoint, task_codes)
        coefficients = torch.softmax(self.cross_stitch, dim=-1).index_select(0, task_codes)
        endpoint = endpoint + coefficients[:, :1] * shared + coefficients[:, 1:] * task
        return endpoint, common + 0.1 * shared

    def patient_embeddings(
        self,
        tokens: torch.Tensor,
        action_mask: torch.Tensor,
        task_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        common = self.base.base.endpoint_embedding(tokens, action_mask, task_codes)
        endpoint = self.base.adapt_endpoint(common, task_codes)
        universal = self.base.base.universal_embedding(tokens, action_mask)
        mechanism = self.candidate.mechanism
        if mechanism == "progressive_layered_extraction":
            endpoint, universal = self._ple_embeddings(common, endpoint, task_codes)
        elif mechanism == "cross_stitch_endpoint_streams":
            endpoint, universal = self._cross_stitch_embeddings(
                common, endpoint, task_codes
            )
        return endpoint, universal

    def routed_and_universal_logits(
        self,
        tokens: torch.Tensor,
        action_mask: torch.Tensor,
        task_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        endpoint, universal_embedding = self.patient_embeddings(
            tokens, action_mask, task_codes
        )
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
        if self.candidate.mechanism == "progressive_layered_extraction":
            shared = torch.stack(
                [expert(embedding) for expert in self.shared_experts], dim=1
            )
            embedding = embedding + 0.25 * shared.mean(dim=1)
        elif self.candidate.mechanism == "cross_stitch_endpoint_streams":
            embedding = embedding + 0.1 * self.shared_stream(embedding)
        return self.base.base.universal_head(embedding).squeeze(-1)


__all__ = ["BroadLiteratureSharedRouterV9"]
