"""Shared per-action motor encoder with tiny script-aware endpoint pooling."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .medical_shared_candidate_registry_v2 import candidate_registry
from .medically_gated_shared_encoder_v2 import MedicallyGatedSharedEncoderV2


@dataclass(frozen=True)
class ScriptAwareCandidateV6:
    candidate_id: str
    head_mode: str
    universal_blend: float


def candidate_registry_v6() -> tuple[ScriptAwareCandidateV6, ...]:
    candidates = []
    index = 0
    for head_mode in ("linear", "small_mlp"):
        for universal_blend in (0.25, 0.5):
            candidates.append(ScriptAwareCandidateV6(
                candidate_id=f"SAR6-{index:03d}",
                head_mode=head_mode,
                universal_blend=universal_blend,
            ))
            index += 1
    if len(candidates) != 4:
        raise AssertionError("the script-aware candidate registry drifted")
    return tuple(candidates)


def _locked_backbone_candidate():
    matched = tuple(
        candidate for candidate in candidate_registry()
        if candidate.candidate_id == "MSC2-022"
    )
    if len(matched) != 1:
        raise RuntimeError("the locked shared full-mesh backbone is unavailable")
    return matched[0]


class ScriptAwareSharedRouterV6(nn.Module):
    """Share action physiology; adapt only post-token script aggregation and head."""

    def __init__(self, candidate: ScriptAwareCandidateV6):
        super().__init__()
        if (
            type(candidate) is not ScriptAwareCandidateV6
            or candidate not in candidate_registry_v6()
        ):
            raise ValueError("v6 requires one exact frozen script-aware candidate")
        self.candidate = candidate
        self.backbone = MedicallyGatedSharedEncoderV2(_locked_backbone_candidate())
        self.task_queries = nn.Parameter(torch.empty(3, self.backbone.patient_dim))
        nn.init.normal_(self.task_queries, mean=0.0, std=0.02)
        if candidate.head_mode == "small_mlp":
            self.backbone.task_heads = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(self.backbone.patient_dim, 16),
                    nn.GELU(),
                    nn.Linear(16, 1),
                )
                for _ in range(3)
            )

    @property
    def task_heads(self) -> nn.ModuleList:
        return self.backbone.task_heads

    @property
    def universal_head(self) -> nn.Module:
        return self.backbone.universal_head

    @property
    def patient_dim(self) -> int:
        return self.backbone.patient_dim

    def shared_action_tokens(
        self,
        clinical_original: torch.Tensor,
        clinical_mirrored: torch.Tensor,
        dense_original: torch.Tensor,
        dense_mirrored: torch.Tensor,
        dense_valid_mask: torch.Tensor,
        dense_available: torch.Tensor,
        dense_timestamps: torch.Tensor,
        action_mask: torch.Tensor,
        action_codes: torch.Tensor,
    ) -> torch.Tensor:
        self.backbone._validate_inputs(
            clinical_original, clinical_mirrored, dense_original, dense_mirrored,
            dense_valid_mask, dense_available, dense_timestamps, action_mask,
            action_codes,
        )
        clinical = self.backbone._clinical_tokens(clinical_original, clinical_mirrored)
        dense_first = self.backbone._dense_tokens(dense_original)
        dense_second = self.backbone._dense_tokens(dense_mirrored)
        dense = self.backbone.dense_pair_projection(torch.cat((
            0.5 * (dense_first + dense_second), torch.abs(dense_first - dense_second)
        ), dim=-1))
        regional_values = self.backbone.regional_evidence(
            dense_original, dense_mirrored, dense_timestamps, action_codes
        )
        regional = self.backbone.regional_encoder(regional_values)
        tokens = self.backbone._fused_tokens(
            clinical, dense, regional, dense_available
        ) + self.backbone.action_embedding(action_codes)
        return self.backbone.action_encoder(
            tokens, src_key_padding_mask=~action_mask
        )

    @staticmethod
    def _validate_pool_inputs(
        tokens: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> None:
        if (
            not isinstance(tokens, torch.Tensor)
            or tokens.ndim != 3
            or tokens.shape[0] < 1
            or tokens.shape[1] < 1
            or tokens.shape[2] != 64
            or not tokens.is_floating_point()
            or not bool(torch.isfinite(tokens).all())
            or action_mask.shape != tokens.shape[:2]
            or action_mask.dtype != torch.bool
            or action_mask.device != tokens.device
            or bool((action_mask.sum(dim=1) == 0).any())
        ):
            raise ValueError("script pooling requires finite shared action tokens")

    def endpoint_embedding(
        self,
        tokens: torch.Tensor,
        action_mask: torch.Tensor,
        task_codes: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_pool_inputs(tokens, action_mask)
        if (
            task_codes.shape != (tokens.shape[0],)
            or task_codes.dtype != torch.long
            or task_codes.device != tokens.device
            or bool((task_codes < 0).any())
            or bool((task_codes >= 3).any())
        ):
            raise ValueError("endpoint pooling requires one valid task code per person")
        query = self.task_queries.index_select(0, task_codes)
        scores = torch.sum(tokens * query[:, None, :], dim=-1) / math.sqrt(self.patient_dim)
        scores = scores.masked_fill(~action_mask, float("-inf"))
        attention = torch.softmax(scores, dim=1)
        attended = torch.sum(tokens * attention.unsqueeze(-1), dim=1)
        maximum = tokens.masked_fill(
            ~action_mask.unsqueeze(-1), float("-inf")
        ).max(dim=1).values
        patient = self.backbone.patient_projection(torch.cat((attended, maximum), dim=-1))
        return self.backbone.patient_norm(patient)

    def universal_embedding(
        self,
        tokens: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_pool_inputs(tokens, action_mask)
        weights = action_mask.unsqueeze(-1).to(tokens.dtype)
        mean = torch.sum(tokens * weights, dim=1) / torch.sum(weights, dim=1)
        maximum = tokens.masked_fill(
            ~action_mask.unsqueeze(-1), float("-inf")
        ).max(dim=1).values
        patient = self.backbone.patient_projection(torch.cat((mean, maximum), dim=-1))
        return self.backbone.patient_norm(patient)

    def task_logits_from_embedding(
        self,
        patient_embedding: torch.Tensor,
        task_codes: torch.Tensor,
    ) -> torch.Tensor:
        if (
            patient_embedding.ndim != 2
            or patient_embedding.shape[1] != self.patient_dim
            or task_codes.shape != (patient_embedding.shape[0],)
            or task_codes.dtype != torch.long
            or task_codes.device != patient_embedding.device
        ):
            raise ValueError("task logits require aligned patient embeddings")
        logits = torch.stack([
            head(patient_embedding).squeeze(-1) for head in self.task_heads
        ], dim=1)
        return logits.gather(1, task_codes[:, None]).squeeze(1)

    def routed_logits(
        self,
        tokens: torch.Tensor,
        action_mask: torch.Tensor,
        task_codes: torch.Tensor,
    ) -> torch.Tensor:
        endpoint = self.endpoint_embedding(tokens, action_mask, task_codes)
        universal = self.universal_embedding(tokens, action_mask)
        task_logits = self.task_logits_from_embedding(endpoint, task_codes)
        universal_logits = self.universal_head(universal).squeeze(-1)
        blend = self.candidate.universal_blend
        return (1.0 - blend) * task_logits + blend * universal_logits


__all__ = [
    "ScriptAwareCandidateV6",
    "ScriptAwareSharedRouterV6",
    "candidate_registry_v6",
]
