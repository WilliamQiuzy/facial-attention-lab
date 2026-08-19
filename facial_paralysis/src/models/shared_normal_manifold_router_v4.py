"""Shared normal-manifold router built on the strongest stable v2 encoder."""
from __future__ import annotations

import torch
from torch import nn

from .medical_shared_candidate_registry_v2 import candidate_registry
from .medically_gated_shared_encoder_v2 import MedicallyGatedSharedEncoderV2
from .normal_manifold_candidate_registry_v4 import (
    NormalManifoldCandidateV4,
    candidate_registry_v4,
)


_LOCKED_BACKBONE_ID = "MSC2-022"


def _locked_backbone_candidate():
    matched = tuple(
        candidate for candidate in candidate_registry()
        if candidate.candidate_id == _LOCKED_BACKBONE_ID
    )
    if len(matched) != 1:
        raise RuntimeError("the locked v2 backbone candidate is unavailable")
    return matched[0]


class SharedNormalManifoldRouterV4(nn.Module):
    """One shared motor encoder and anchor; routing begins at endpoint logits."""

    def __init__(self, candidate: NormalManifoldCandidateV4):
        super().__init__()
        if (
            type(candidate) is not NormalManifoldCandidateV4
            or candidate not in candidate_registry_v4()
        ):
            raise ValueError("v4 requires one exact frozen normal-manifold candidate")
        self.candidate = candidate
        self.backbone = MedicallyGatedSharedEncoderV2(_locked_backbone_candidate())
        self.normal_anchor = nn.Parameter(torch.zeros(self.backbone.patient_dim))

    @property
    def patient_dim(self) -> int:
        return self.backbone.patient_dim

    def encode(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode(*inputs)

    def normal_distance(self, patient_embedding: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(patient_embedding, torch.Tensor)
            or patient_embedding.ndim != 2
            or patient_embedding.shape[1] != self.patient_dim
            or not patient_embedding.is_floating_point()
            or not bool(torch.isfinite(patient_embedding).all())
        ):
            raise ValueError("normal distance requires finite shared patient embeddings")
        return torch.mean(
            torch.square(patient_embedding - self.normal_anchor[None, :]), dim=1
        )

    def normal_manifold_loss(
        self,
        patient_embedding: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> torch.Tensor:
        distances = self.normal_distance(patient_embedding)
        if (
            labels.shape != distances.shape
            or sample_weights.shape != distances.shape
            or labels.device != distances.device
            or sample_weights.device != distances.device
            or not labels.is_floating_point()
            or not sample_weights.is_floating_point()
            or not bool(torch.isfinite(labels).all())
            or not bool(torch.isfinite(sample_weights).all())
            or bool(((labels != 0.0) & (labels != 1.0)).any())
            or bool((sample_weights < 0.0).any())
        ):
            raise ValueError("normal-manifold loss requires aligned binary evidence")
        controls = labels == 0.0
        control_mass = sample_weights[controls].sum()
        if not bool(controls.any()) or float(control_mass) <= 0.0:
            raise ValueError("each fold requires positively weighted controls")
        return torch.sum(distances[controls] * sample_weights[controls]) / control_mass

    def universal_logits_from_embedding(
        self, patient_embedding: torch.Tensor
    ) -> torch.Tensor:
        return self.backbone.universal_head(patient_embedding).squeeze(-1)

    def routed_logits_from_embedding(
        self,
        patient_embedding: torch.Tensor,
        task_codes: torch.Tensor,
    ) -> torch.Tensor:
        task = self.backbone.task_logits_from_embedding(patient_embedding, task_codes)
        universal = self.universal_logits_from_embedding(patient_embedding)
        blend = self.candidate.universal_blend
        return (1.0 - blend) * task + blend * universal

    def forward(self, task_codes: torch.Tensor, *inputs: torch.Tensor) -> torch.Tensor:
        return self.routed_logits_from_embedding(self.encode(*inputs), task_codes)


__all__ = ["SharedNormalManifoldRouterV4"]
