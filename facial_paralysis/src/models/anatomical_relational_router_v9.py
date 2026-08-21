"""Paper-grounded bilateral local-global relation residual for shared V9."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.preprocessing.generalization_110d import (
    LANDMARK_MI_110D,
    candidate_feature_names,
)

from .residual_shared_router_v8 import (
    ResidualSharedRouterV8,
    candidate_registry_v8,
)


_FROZEN_FEATURE_NAMES = candidate_feature_names(LANDMARK_MI_110D)
_REGION_PREFIXES = {
    "eye": ("fissure_h_", "fissure_w_", "eye_area_"),
    "brow": ("brow_h_",),
    "oral": ("corner_y_", "corner_x_", "commissure_x_", "mouth_"),
}


def anatomical_region_indices(
    feature_names: tuple[str, ...],
) -> dict[str, tuple[int, ...]]:
    """Bind every exact 110D feature to one physician-relevant facial region."""
    if type(feature_names) is not tuple or feature_names != _FROZEN_FEATURE_NAMES:
        raise ValueError("anatomical V9 requires the exact name-bound 110D schema")
    regions = {
        region: tuple(
            index for index, name in enumerate(feature_names)
            if name.startswith(prefixes)
        )
        for region, prefixes in _REGION_PREFIXES.items()
    }
    flattened = tuple(index for values in regions.values() for index in values)
    if (
        tuple(len(regions[name]) for name in regions) != (49, 19, 42)
        or len(flattened) != 110
        or len(set(flattened)) != 110
        or set(flattened) != set(range(110))
    ):
        raise RuntimeError("the clinical eye/brow/oral partition is incomplete")
    return regions


_REGION_INDICES = anatomical_region_indices(_FROZEN_FEATURE_NAMES)


@dataclass(frozen=True)
class AnatomicalRelationalCandidateV9:
    candidate_id: str
    relation_enabled: bool
    medical_rationale: str


def candidate_registry_v9() -> tuple[AnatomicalRelationalCandidateV9, ...]:
    return (
        AnatomicalRelationalCandidateV9(
            candidate_id="ARR9-000",
            relation_enabled=False,
            medical_rationale="Exact deterministic RSR8-001 comparator.",
        ),
        AnatomicalRelationalCandidateV9(
            candidate_id="ARR9-001",
            relation_enabled=True,
            medical_rationale=(
                "One source-blind local-global relation block preserves bilateral "
                "eye, brow, and oral physiology before shared action pooling."
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


class AnatomicalRelationalRouterV9(nn.Module):
    """Add one shared anatomy-bounded relation residual to V8 action tokens."""

    def __init__(self, candidate: AnatomicalRelationalCandidateV9):
        super().__init__()
        if (
            type(candidate) is not AnatomicalRelationalCandidateV9
            or candidate not in candidate_registry_v9()
        ):
            raise ValueError("anatomical relational V9 requires a frozen candidate")
        self.candidate = candidate
        self.base = ResidualSharedRouterV8(_locked_v8_candidate())
        if not candidate.relation_enabled:
            return
        self.region_encoders = nn.ModuleDict({
            region: nn.Sequential(
                nn.LayerNorm(len(indices) * 2),
                nn.Linear(len(indices) * 2, 64),
                nn.GELU(),
                nn.Linear(64, 64),
            )
            for region, indices in _REGION_INDICES.items()
        })
        self.anatomical_identity = nn.Parameter(torch.zeros(4, 64))
        nn.init.normal_(self.anatomical_identity, mean=0.0, std=0.02)
        self.relation_block = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.region_attention = nn.Linear(64, 1)
        self.relation_output = nn.Linear(128, 64)
        nn.init.zeros_(self.relation_output.weight)
        nn.init.zeros_(self.relation_output.bias)

    def task_specific_parameter_fraction(self) -> float:
        task_specific = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name.startswith("base.adapters")
            or name.startswith("base.base.task_queries")
            or name.startswith("base.base.backbone.task_heads")
        )
        return task_specific / sum(parameter.numel() for parameter in self.parameters())

    def _anatomical_residual(
        self,
        clinical_original: torch.Tensor,
        clinical_mirrored: torch.Tensor,
        global_tokens: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            clinical_original.ndim != 3
            or clinical_original.shape[-1] != 110
            or clinical_mirrored.shape != clinical_original.shape
            or global_tokens.shape != clinical_original.shape[:2] + (64,)
            or action_mask.shape != clinical_original.shape[:2]
            or action_mask.dtype != torch.bool
            or clinical_original.device != global_tokens.device
            or clinical_mirrored.device != global_tokens.device
            or action_mask.device != global_tokens.device
            or not bool(torch.isfinite(clinical_original).all())
            or not bool(torch.isfinite(clinical_mirrored).all())
            or not bool(torch.isfinite(global_tokens).all())
        ):
            raise ValueError("anatomical relation received malformed action evidence")
        batch, actions = clinical_original.shape[:2]
        local_tokens = []
        for region, indices in _REGION_INDICES.items():
            index = torch.tensor(indices, dtype=torch.long, device=global_tokens.device)
            first = clinical_original.index_select(-1, index)
            second = clinical_mirrored.index_select(-1, index)
            paired = torch.cat((
                0.5 * (first + second), torch.abs(first - second)
            ), dim=-1)
            local_tokens.append(self.region_encoders[region](paired))
        anatomy = torch.stack((*local_tokens, global_tokens), dim=2)
        anatomy = anatomy.reshape(batch * actions, 4, 64)
        anatomy = anatomy + self.anatomical_identity.unsqueeze(0)
        related = self.relation_block(anatomy).reshape(batch, actions, 4, 64)
        local = related[:, :, :3]
        weights = torch.softmax(self.region_attention(local).squeeze(-1), dim=-1)
        local_summary = torch.sum(local * weights.unsqueeze(-1), dim=2)
        global_summary = related[:, :, 3]
        residual = self.relation_output(torch.cat((local_summary, global_summary), dim=-1))
        return residual * action_mask.unsqueeze(-1).to(residual.dtype)

    def shared_action_tokens(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) != 9:
            raise ValueError("anatomical V9 requires the exact shared input contract")
        tokens = self.base.shared_action_tokens(*inputs)
        if not self.candidate.relation_enabled:
            return tokens
        residual = self._anatomical_residual(
            inputs[0], inputs[1], tokens, inputs[-2]
        )
        return tokens + residual

    def routed_logits(
        self,
        tokens: torch.Tensor,
        action_mask: torch.Tensor,
        task_codes: torch.Tensor,
    ) -> torch.Tensor:
        return self.base.routed_logits(tokens, action_mask, task_codes)

    def routed_and_universal_logits(
        self,
        tokens: torch.Tensor,
        action_mask: torch.Tensor,
        task_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        common = self.base.base.endpoint_embedding(tokens, action_mask, task_codes)
        endpoint = self.base.adapt_endpoint(common, task_codes)
        universal_embedding = self.base.base.universal_embedding(tokens, action_mask)
        task = self.base.base.task_logits_from_embedding(endpoint, task_codes)
        universal = self.base.base.universal_head(universal_embedding).squeeze(-1)
        return 0.75 * task + 0.25 * universal, universal

    def universal_logits(
        self,
        tokens: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        universal_embedding = self.base.base.universal_embedding(tokens, action_mask)
        return self.base.base.universal_head(universal_embedding).squeeze(-1)


__all__ = [
    "AnatomicalRelationalCandidateV9",
    "AnatomicalRelationalRouterV9",
    "anatomical_region_indices",
    "candidate_registry_v9",
]
