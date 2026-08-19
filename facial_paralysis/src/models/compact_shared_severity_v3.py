"""Compact, medically constrained shared severity encoder for three cohorts."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .dense_clinical_shared_encoder_v1 import ACTION_VOCAB, TASK_HEAD_COUNT
from .medically_gated_shared_encoder_v2 import (
    BROW_LANDMARKS,
    EYE_LANDMARKS,
    MOUTH_LANDMARKS,
)


_REGIONS = (
    BROW_LANDMARKS,
    EYE_LANDMARKS,
    MOUTH_LANDMARKS,
    tuple(range(478)),
)

COMPACT_RATIONALES = {
    "all_regions": (
        "Facial dysfunction can be distributed across brow, eye and oral regions; "
        "retain each region plus a global summary."
    ),
    "action_matched": (
        "Brow raise, eye closure and oral movements have anatomically intended "
        "regions; keep the intended region plus global motion."
    ),
    "excursion": (
        "Voluntary movement excursion is a core dynamic facial-function domain."
    ),
    "excursion_velocity": (
        "Paralysis can alter both movement magnitude and movement speed."
    ),
    "meanmax": (
        "Mean and maximum summarize typical and worst action capacity without "
        "claiming recording order is disease progression."
    ),
    "action_weighted": (
        "Different standardized actions can contribute unequally to composite "
        "function; weights are shared across cohorts and remain predictive only."
    ),
    "embedding_head": (
        "Cohorts expose different binary endpoints, so a small endpoint head may "
        "read the same shared patient representation."
    ),
    "severity_calibration": (
        "All positive labels indicate impaired motor function; a single monotone "
        "severity axis may use cohort-specific positive scale and intercept only."
    ),
}


@dataclass(frozen=True)
class CompactCandidateV3:
    candidate_id: str
    region_scope: str
    dynamic_stats: str
    pooling: str
    head_mode: str


def compact_candidate_registry() -> tuple[CompactCandidateV3, ...]:
    candidates = []
    index = 0
    for region_scope in ("all_regions", "action_matched"):
        for dynamic_stats in ("excursion", "excursion_velocity"):
            for pooling in ("meanmax", "action_weighted"):
                for head_mode in ("embedding_head", "severity_calibration"):
                    candidates.append(CompactCandidateV3(
                        candidate_id=f"MSC3-{index:03d}",
                        region_scope=region_scope,
                        dynamic_stats=dynamic_stats,
                        pooling=pooling,
                        head_mode=head_mode,
                    ))
                    index += 1
    if len(candidates) != 16 or any(
        option not in COMPACT_RATIONALES
        for candidate in candidates
        for option in (
            candidate.region_scope, candidate.dynamic_stats,
            candidate.pooling, candidate.head_mode,
        )
    ):
        raise AssertionError("compact medically gated registry drifted")
    return tuple(candidates)


class CompactSharedSeverityV3(nn.Module):
    """Full-mesh regional measurements feed one shared 64D patient encoder."""

    def __init__(self, candidate: CompactCandidateV3):
        super().__init__()
        if (
            type(candidate) is not CompactCandidateV3
            or candidate not in compact_candidate_registry()
        ):
            raise ValueError("v3 requires one frozen compact candidate")
        self.candidate = candidate
        self.clinical_encoder = nn.Sequential(
            nn.Linear(110, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )
        self.regional_encoder = nn.Sequential(
            nn.Linear(16, 16),
            nn.LayerNorm(16),
            nn.GELU(),
        )
        self.action_embedding = nn.Embedding(len(ACTION_VOCAB), 16)
        self.action_fusion = nn.Sequential(
            nn.Linear(32 + 16 + 16 + 1, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )
        self.action_weight = nn.Linear(32, 1)
        self.patient_projection = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.shared_severity = nn.Linear(64, 1)
        self.task_heads = nn.ModuleList(
            nn.Linear(64, 1) for _ in range(TASK_HEAD_COUNT)
        )
        self.calibration_log_scale = nn.Parameter(torch.zeros(TASK_HEAD_COUNT))
        self.calibration_bias = nn.Parameter(torch.zeros(TASK_HEAD_COUNT))

    @staticmethod
    def _validate_inputs(
        clinical: torch.Tensor,
        dense: torch.Tensor,
        dense_available: torch.Tensor,
        timestamps: torch.Tensor,
        action_mask: torch.Tensor,
        action_codes: torch.Tensor,
    ) -> tuple[int, int]:
        if not isinstance(clinical, torch.Tensor) or clinical.ndim != 3:
            raise ValueError("clinical evidence requires batch, action and feature axes")
        batch, actions, dimension = clinical.shape
        device = clinical.device
        if (
            batch < 1
            or actions < 1
            or dimension != 110
            or not clinical.is_floating_point()
            or not bool(torch.isfinite(clinical).all())
            or dense.shape != (batch, actions, 32, 478, 3)
            or not dense.is_floating_point()
            or dense.device != device
            or not bool(torch.isfinite(dense).all())
            or dense_available.shape != (batch, actions)
            or dense_available.dtype != torch.bool
            or dense_available.device != device
            or timestamps.shape != (batch, actions, 32)
            or not timestamps.is_floating_point()
            or timestamps.device != device
            or not bool(torch.isfinite(timestamps).all())
            or action_mask.shape != (batch, actions)
            or action_mask.dtype != torch.bool
            or action_mask.device != device
            or action_codes.shape != (batch, actions)
            or action_codes.dtype != torch.long
            or action_codes.device != device
            or bool((action_codes < 0).any())
            or bool((action_codes >= len(ACTION_VOCAB)).any())
            or bool((action_mask.sum(dim=1) == 0).any())
            or bool((dense_available & ~action_mask).any())
        ):
            raise ValueError("compact shared inputs differ from the closed contract")
        if bool(dense_available.any()) and bool(
            (torch.diff(timestamps, dim=-1)[dense_available] <= 0.0).any()
        ):
            raise ValueError("regional velocity requires increasing real seconds")
        if bool((timestamps[~dense_available] != 0.0).any()):
            raise ValueError("unavailable dense actions cannot carry a clock")
        return batch, actions

    def regional_descriptor(
        self,
        dense: torch.Tensor,
        timestamps: torch.Tensor,
        dense_available: torch.Tensor,
        action_codes: torch.Tensor,
    ) -> torch.Tensor:
        if (
            dense.ndim != 5
            or dense.shape[-3:] != (32, 478, 3)
            or timestamps.shape != dense.shape[:2] + (32,)
            or dense_available.shape != dense.shape[:2]
            or action_codes.shape != dense.shape[:2]
        ):
            raise ValueError("regional descriptor received malformed evidence")
        blocks = []
        intervals = torch.diff(timestamps, dim=-1).clamp_min(1e-6)
        for indices in _REGIONS:
            selected = dense[:, :, :, list(indices), :]
            magnitude = torch.linalg.vector_norm(selected, dim=-1)
            mean_excursion = magnitude.mean(dim=(2, 3))
            peak_excursion = magnitude.amax(dim=(2, 3))
            displacement = torch.linalg.vector_norm(
                selected[:, :, 1:] - selected[:, :, :-1], dim=-1
            )
            velocity = displacement / intervals.unsqueeze(-1)
            mean_velocity = velocity.mean(dim=(2, 3))
            peak_velocity = velocity.amax(dim=(2, 3))
            blocks.append(torch.stack((
                mean_excursion, peak_excursion, mean_velocity, peak_velocity,
            ), dim=-1))
        descriptor = torch.stack(blocks, dim=-2)
        if self.candidate.dynamic_stats == "excursion":
            descriptor = descriptor.clone()
            descriptor[..., 2:] = 0.0
        if self.candidate.region_scope == "action_matched":
            region_mask = torch.zeros(
                *action_codes.shape, 4,
                dtype=descriptor.dtype,
                device=descriptor.device,
            )
            region_mask[..., 3] = 1.0
            region_mask[..., 0] = (
                action_codes == ACTION_VOCAB.index("BROW_RAISE")
            ).to(descriptor.dtype)
            eye = (
                (action_codes == ACTION_VOCAB.index("EYE_GENTLE"))
                | (action_codes == ACTION_VOCAB.index("EYE_FORCEFUL"))
            )
            region_mask[..., 1] = eye.to(descriptor.dtype)
            oral = torch.zeros_like(action_codes, dtype=torch.bool)
            for name in (
                "LIP_PUCKER", "MOUTH_OPEN", "SMILE_SPREAD", "SMILE_GENTLE",
                "SMILE_FULL", "SHOW_BOTTOM_TEETH",
            ):
                oral |= action_codes == ACTION_VOCAB.index(name)
            region_mask[..., 2] = oral.to(descriptor.dtype)
            descriptor = descriptor * region_mask.unsqueeze(-1)
        descriptor = descriptor * dense_available[..., None, None].to(
            descriptor.dtype
        )
        return descriptor.reshape(*dense.shape[:2], 16)

    def encode_with_action_weights(
        self,
        clinical: torch.Tensor,
        dense: torch.Tensor,
        dense_available: torch.Tensor,
        timestamps: torch.Tensor,
        action_mask: torch.Tensor,
        action_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(
            clinical, dense, dense_available, timestamps, action_mask, action_codes
        )
        regional = self.regional_descriptor(
            dense, timestamps, dense_available, action_codes
        )
        token = self.action_fusion(torch.cat((
            self.clinical_encoder(clinical),
            self.regional_encoder(regional),
            self.action_embedding(action_codes),
            dense_available.unsqueeze(-1).to(clinical.dtype),
        ), dim=-1))
        maximum = token.masked_fill(
            ~action_mask.unsqueeze(-1), float("-inf")
        ).max(dim=1).values
        if self.candidate.pooling == "action_weighted":
            scores = self.action_weight(token).squeeze(-1).masked_fill(
                ~action_mask, float("-inf")
            )
            weights = torch.softmax(scores, dim=1)
        else:
            weights = action_mask.to(token.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True)
        mean = (token * weights.unsqueeze(-1)).sum(dim=1)
        embedding = self.patient_projection(torch.cat((mean, maximum), dim=-1))
        return embedding, weights

    def encode(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self.encode_with_action_weights(*inputs)[0]

    def task_logits_from_embedding(
        self,
        patient_embedding: torch.Tensor,
        task_codes: torch.Tensor,
    ) -> torch.Tensor:
        if (
            patient_embedding.ndim != 2
            or patient_embedding.shape[1] != 64
            or task_codes.shape != (patient_embedding.shape[0],)
            or task_codes.dtype != torch.long
            or task_codes.device != patient_embedding.device
            or bool((task_codes < 0).any())
            or bool((task_codes >= TASK_HEAD_COUNT).any())
        ):
            raise ValueError("task calibration requires one valid endpoint code")
        if self.candidate.head_mode == "severity_calibration":
            severity = self.shared_severity(patient_embedding).squeeze(-1)
            scale = F.softplus(self.calibration_log_scale[task_codes]) + 1e-6
            return scale * severity + self.calibration_bias[task_codes]
        logits = torch.stack(
            [head(patient_embedding).squeeze(-1) for head in self.task_heads], dim=1
        )
        return logits.gather(1, task_codes[:, None]).squeeze(1)

    def forward(
        self,
        clinical: torch.Tensor,
        dense: torch.Tensor,
        dense_available: torch.Tensor,
        timestamps: torch.Tensor,
        action_mask: torch.Tensor,
        action_codes: torch.Tensor,
    ) -> torch.Tensor:
        return self.shared_severity(self.encode(
            clinical, dense, dense_available, timestamps, action_mask, action_codes
        )).squeeze(-1)


__all__ = [
    "COMPACT_RATIONALES",
    "CompactCandidateV3",
    "CompactSharedSeverityV3",
    "compact_candidate_registry",
]
