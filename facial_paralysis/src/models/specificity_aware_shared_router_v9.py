"""Shared V8 trunk with a source-blind healthy-reference mechanism for V9."""
from __future__ import annotations

from itertools import combinations

import torch
from torch import nn
from torch.nn import functional as F

from .residual_shared_router_v8 import ResidualSharedRouterV8, candidate_registry_v8
from .specificity_aware_candidate_registry_v9 import (
    SpecificityCandidateV9,
    candidate_registry_v9,
)


_HEALTHY_BLEND = {"off": 0.0, "compact": 0.15, "compact_margin": 0.25}
_REFERENCE_WEIGHT = {"off": 0.0, "compact": 0.05, "compact_margin": 0.05}
_AFFECTED_MARGIN_WEIGHT = {"off": 0.0, "compact": 0.0, "compact_margin": 0.05}
_AFFECTED_DISTANCE_MARGIN = 1.0


def _locked_v8_candidate():
    rows = tuple(
        row for row in candidate_registry_v8() if row.candidate_id == "RSR8-001"
    )
    if len(rows) != 1:
        raise RuntimeError("the locked RSR8-001 comparator is unavailable")
    return rows[0]


class SpecificityAwareSharedRouterV9(nn.Module):
    """Keep the shared V8 motor encoder and add one shared control reference."""

    def __init__(self, candidate: SpecificityCandidateV9):
        super().__init__()
        if (
            type(candidate) is not SpecificityCandidateV9
            or candidate not in candidate_registry_v9()
        ):
            raise ValueError("V9 requires one exact frozen specificity candidate")
        self.candidate = candidate
        self.base = ResidualSharedRouterV8(_locked_v8_candidate())
        self.normal_anchor = nn.Parameter(torch.zeros(64))
        self.normal_scale_raw = nn.Parameter(torch.tensor(0.54132485))
        self.normal_bias = nn.Parameter(torch.tensor(-1.0))

    def shared_action_tokens(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self.base.shared_action_tokens(*inputs)

    @staticmethod
    def _validate_embeddings(embedding: torch.Tensor) -> None:
        if (
            not isinstance(embedding, torch.Tensor)
            or embedding.ndim != 2
            or embedding.shape[0] < 1
            or embedding.shape[1] != 64
            or not embedding.is_floating_point()
            or not bool(torch.isfinite(embedding).all())
        ):
            raise ValueError("V9 requires finite shared 64D participant embeddings")

    def patient_embeddings(
        self,
        tokens: torch.Tensor,
        action_mask: torch.Tensor,
        task_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        common = self.base.base.endpoint_embedding(tokens, action_mask, task_codes)
        endpoint = self.base.adapt_endpoint(common, task_codes)
        universal = self.base.base.universal_embedding(tokens, action_mask)
        return endpoint, universal

    def normal_distance(self, universal_embedding: torch.Tensor) -> torch.Tensor:
        self._validate_embeddings(universal_embedding)
        return torch.mean(
            torch.square(universal_embedding - self.normal_anchor[None, :]), dim=1
        )

    def normality_logits(self, universal_embedding: torch.Tensor) -> torch.Tensor:
        distance = self.normal_distance(universal_embedding)
        scale = F.softplus(self.normal_scale_raw)
        return scale * distance + self.normal_bias

    @staticmethod
    def _validate_loss_inputs(
        embedding: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        SpecificityAwareSharedRouterV9._validate_embeddings(embedding)
        count = embedding.shape[0]
        if (
            labels.shape != (count,)
            or weights.shape != (count,)
            or labels.device != embedding.device
            or weights.device != embedding.device
            or not labels.is_floating_point()
            or not weights.is_floating_point()
            or not bool(torch.isfinite(labels).all())
            or not bool(torch.isfinite(weights).all())
            or bool(((labels != 0.0) & (labels != 1.0)).any())
            or bool((weights < 0.0).any())
            or float(weights.sum()) <= 0.0
        ):
            raise ValueError("V9 losses require aligned binary participant evidence")

    def normal_reference_loss(
        self,
        universal_embedding: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_loss_inputs(universal_embedding, labels, weights)
        if self.candidate.healthy_mode == "off":
            return universal_embedding.sum() * 0.0
        distances = self.normal_distance(universal_embedding)
        controls = labels == 0.0
        affected = labels == 1.0
        control_mass = weights[controls].sum()
        affected_mass = weights[affected].sum()
        if (
            not bool(controls.any())
            or not bool(affected.any())
            or float(control_mass) <= 0.0
            or float(affected_mass) <= 0.0
        ):
            raise ValueError("V9 reference loss requires both outcome classes")
        compactness = torch.sum(distances[controls] * weights[controls]) / control_mass
        margin = torch.sum(
            torch.square(F.relu(_AFFECTED_DISTANCE_MARGIN - distances[affected]))
            * weights[affected]
        ) / affected_mass
        return (
            _REFERENCE_WEIGHT[self.candidate.healthy_mode] * compactness
            + _AFFECTED_MARGIN_WEIGHT[self.candidate.healthy_mode] * margin
        )

    def control_alignment_loss(
        self,
        universal_embedding: torch.Tensor,
        labels: torch.Tensor,
        task_codes: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_loss_inputs(universal_embedding, labels, weights)
        if (
            task_codes.shape != labels.shape
            or task_codes.dtype != torch.long
            or task_codes.device != universal_embedding.device
            or bool((task_codes < 0).any())
            or bool((task_codes >= 3).any())
        ):
            raise ValueError("control alignment requires the three training protocols")
        if self.candidate.control_alignment_weight == 0.0:
            return universal_embedding.sum() * 0.0
        centroids = []
        for task_code in range(3):
            selected = (labels == 0.0) & (task_codes == task_code)
            mass = weights[selected].sum()
            if not bool(selected.any()) or float(mass) <= 0.0:
                raise ValueError("every protocol requires controls for alignment")
            centroids.append(
                torch.sum(universal_embedding[selected] * weights[selected, None], dim=0)
                / mass
            )
        pair_losses = [
            torch.mean(torch.square(first - second))
            for first, second in combinations(centroids, 2)
        ]
        return self.candidate.control_alignment_weight * torch.stack(pair_losses).mean()

    def routed_logits(
        self,
        tokens: torch.Tensor,
        action_mask: torch.Tensor,
        task_codes: torch.Tensor,
    ) -> torch.Tensor:
        endpoint, universal = self.patient_embeddings(tokens, action_mask, task_codes)
        task_logits = self.base.base.task_logits_from_embedding(endpoint, task_codes)
        universal_logits = self.base.base.universal_head(universal).squeeze(-1)
        blend = self.candidate.universal_blend
        shared_logits = (1.0 - blend) * task_logits + blend * universal_logits
        healthy_blend = _HEALTHY_BLEND[self.candidate.healthy_mode]
        if healthy_blend == 0.0:
            return shared_logits
        normal_logits = self.normality_logits(universal)
        return (1.0 - healthy_blend) * shared_logits + healthy_blend * normal_logits


__all__ = ["SpecificityAwareSharedRouterV9"]
