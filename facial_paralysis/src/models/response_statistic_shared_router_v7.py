"""Shared clinical and response-statistic action router v7."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .dense_clinical_shared_encoder_v1 import ACTION_VOCAB


@dataclass(frozen=True)
class ResponseStatisticCandidateV7:
    candidate_id: str
    pca_dim: int
    head_mode: str


def candidate_registry_v7() -> tuple[ResponseStatisticCandidateV7, ...]:
    candidates = []
    index = 0
    for pca_dim in (64, 128):
        for head_mode in ("linear", "small_mlp"):
            candidates.append(ResponseStatisticCandidateV7(
                candidate_id=f"RSR7-{index:03d}", pca_dim=pca_dim, head_mode=head_mode
            ))
            index += 1
    return tuple(candidates)


class ResponseStatisticSharedRouterV7(nn.Module):
    def __init__(self, candidate: ResponseStatisticCandidateV7):
        super().__init__()
        if type(candidate) is not ResponseStatisticCandidateV7 or candidate not in candidate_registry_v7():
            raise ValueError("v7 requires one exact frozen candidate")
        self.candidate = candidate
        dimension = 64
        self.patient_dim = dimension
        self.clinical_encoder = nn.Sequential(
            nn.Linear(110, dimension), nn.LayerNorm(dimension), nn.GELU(),
            nn.Linear(dimension, dimension),
        )
        self.clinical_pair_projection = nn.Sequential(
            nn.Linear(dimension * 2, dimension), nn.LayerNorm(dimension), nn.GELU()
        )
        self.dense_encoder = nn.Sequential(
            nn.Linear(candidate.pca_dim, dimension), nn.LayerNorm(dimension), nn.GELU(),
            nn.Linear(dimension, dimension),
        )
        self.fusion = nn.Sequential(
            nn.Linear(dimension * 2 + 1, dimension), nn.LayerNorm(dimension), nn.GELU()
        )
        self.action_embedding = nn.Embedding(len(ACTION_VOCAB), dimension)
        layer = nn.TransformerEncoderLayer(
            d_model=dimension, nhead=4, dim_feedforward=128, dropout=0.10,
            activation="gelu", batch_first=True,
        )
        self.action_encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.task_queries = nn.Parameter(torch.empty(3, dimension))
        nn.init.normal_(self.task_queries, std=0.02)
        self.patient_projection = nn.Linear(dimension * 2, dimension)
        self.patient_norm = nn.LayerNorm(dimension)
        if candidate.head_mode == "linear":
            self.task_heads = nn.ModuleList(nn.Linear(dimension, 1) for _ in range(3))
        else:
            self.task_heads = nn.ModuleList(
                nn.Sequential(nn.Linear(dimension, 16), nn.GELU(), nn.Linear(16, 1))
                for _ in range(3)
            )
        self.universal_head = nn.Linear(dimension, 1)

    def shared_action_tokens(
        self, clinical_original, clinical_mirrored, dense_pca,
        dense_available, action_mask, action_codes,
    ):
        if (
            clinical_original.ndim != 3 or clinical_original.shape[-1] != 110
            or clinical_mirrored.shape != clinical_original.shape
            or dense_pca.shape != clinical_original.shape[:2] + (self.candidate.pca_dim,)
            or dense_available.shape != clinical_original.shape[:2] or dense_available.dtype != torch.bool
            or action_mask.shape != clinical_original.shape[:2] or action_mask.dtype != torch.bool
            or action_codes.shape != clinical_original.shape[:2] or action_codes.dtype != torch.long
            or not bool(torch.isfinite(clinical_original).all())
            or not bool(torch.isfinite(clinical_mirrored).all())
            or not bool(torch.isfinite(dense_pca).all())
            or bool((dense_available & ~action_mask).any())
        ):
            raise ValueError("v7 shared action inputs differ from the closed contract")
        first = self.clinical_encoder(clinical_original)
        second = self.clinical_encoder(clinical_mirrored)
        clinical = self.clinical_pair_projection(torch.cat((
            0.5 * (first + second), torch.abs(first - second)
        ), dim=-1))
        dense = self.dense_encoder(dense_pca)
        mask = dense_available.unsqueeze(-1).to(clinical.dtype)
        tokens = self.fusion(torch.cat((clinical, dense * mask, mask), dim=-1))
        tokens = tokens + self.action_embedding(action_codes)
        return self.action_encoder(tokens, src_key_padding_mask=~action_mask)

    def _patient(self, tokens, action_mask, task_codes=None):
        if task_codes is None:
            weights = action_mask.unsqueeze(-1).to(tokens.dtype)
            pooled = (tokens * weights).sum(1) / weights.sum(1)
        else:
            query = self.task_queries.index_select(0, task_codes)
            scores = (tokens * query[:, None]).sum(-1) / math.sqrt(self.patient_dim)
            pooled = (tokens * torch.softmax(scores.masked_fill(~action_mask, float("-inf")), 1).unsqueeze(-1)).sum(1)
        maximum = tokens.masked_fill(~action_mask.unsqueeze(-1), float("-inf")).max(1).values
        return self.patient_norm(self.patient_projection(torch.cat((pooled, maximum), -1)))

    def task_logits_from_embedding(self, embedding, task_codes):
        logits = torch.stack([head(embedding).squeeze(-1) for head in self.task_heads], 1)
        return logits.gather(1, task_codes[:, None]).squeeze(1)

    def routed_logits(self, tokens, action_mask, task_codes):
        endpoint = self._patient(tokens, action_mask, task_codes)
        universal = self._patient(tokens, action_mask)
        return 0.75 * self.task_logits_from_embedding(endpoint, task_codes) + 0.25 * self.universal_head(universal).squeeze(-1)


__all__ = ["ResponseStatisticCandidateV7", "ResponseStatisticSharedRouterV7", "candidate_registry_v7"]
