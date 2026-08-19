"""Source-blind missing-modality mixture for universal orofacial phenotypes."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


MODEL_VARIANTS = (
    "linear_expert_mixture",
    "residual_mil",
    "hybrid_tcn_mil",
    "hybrid_set_transformer",
)
FIXED_FUSION_RULES = (
    "max_probability",
    "reliability_noisy_or",
    "confidence_weighted",
)
EXPERT_NAMES = (
    "global_landmark110",
    "action_landmark110",
    "common398",
    "mediapipe_temporal",
    "au_bilateral_capacity",
    "au_palsy_capacity",
)


def fixed_source_blind_fusion(
    expert_logits: torch.Tensor,
    availability: torch.Tensor,
    reliability: torch.Tensor,
    *,
    rule: str,
) -> torch.Tensor:
    """Fuse phenotype experts without dataset identity or fitted parameters."""
    if (
        rule not in FIXED_FUSION_RULES
        or expert_logits.ndim != 2
        or not 2 <= expert_logits.shape[1] <= 16
        or availability.shape != expert_logits.shape
        or availability.dtype != torch.bool
        or reliability.shape != expert_logits.shape
        or not expert_logits.is_floating_point()
        or not reliability.is_floating_point()
        or not bool(torch.isfinite(expert_logits).all())
        or not bool(torch.isfinite(reliability).all())
        or bool(((reliability < 0.0) | (reliability > 1.0)).any())
        or bool((~availability.any(dim=1)).any())
        or bool(((reliability * availability).sum(dim=1) <= 0.0).any())
    ):
        raise ValueError("fixed fusion inputs violate the expert contract")
    probabilities = torch.sigmoid(expert_logits)
    if rule == "max_probability":
        return probabilities.masked_fill(~availability, -torch.inf).max(dim=1).values
    effective_reliability = reliability * availability.to(reliability.dtype)
    if rule == "reliability_noisy_or":
        log_survival = effective_reliability * torch.log1p(
            -probabilities.clamp(max=1.0 - torch.finfo(probabilities.dtype).eps)
        )
        return (-torch.expm1(log_survival.sum(dim=1))).clamp(0.0, 1.0)
    confidence = (probabilities - 0.5).abs()
    weights = effective_reliability * confidence
    fallback = effective_reliability
    empty = weights.sum(dim=1, keepdim=True) <= torch.finfo(weights.dtype).eps
    weights = torch.where(empty, fallback, weights)
    return (weights * probabilities).sum(dim=1) / weights.sum(dim=1)


def healthy_control_alignment_loss(
    expert_embeddings: torch.Tensor,
    labels: torch.Tensor,
    source_indices: torch.Tensor,
    availability: torch.Tensor,
) -> torch.Tensor:
    """Align healthy expert moments across sources using training rows only."""
    if (
        expert_embeddings.ndim != 3
        or labels.shape != expert_embeddings.shape[:1]
        or source_indices.shape != labels.shape
        or source_indices.dtype != torch.long
        or availability.shape != expert_embeddings.shape[:2]
        or availability.dtype != torch.bool
        or not expert_embeddings.is_floating_point()
        or not labels.is_floating_point()
        or not bool(torch.isfinite(expert_embeddings).all())
        or not bool(torch.isfinite(labels).all())
        or bool(((labels != 0.0) & (labels != 1.0)).any())
        or bool((source_indices < 0).any())
    ):
        raise ValueError("healthy-control alignment inputs are invalid")
    terms = []
    healthy = labels == 0.0
    for expert in range(expert_embeddings.shape[1]):
        moments = []
        for source in torch.unique(source_indices, sorted=True):
            selected = healthy & availability[:, expert] & (source_indices == source)
            if int(selected.sum().item()) < 2:
                continue
            values = expert_embeddings[selected, expert]
            moments.append((values.mean(dim=0), values.var(dim=0, unbiased=False)))
        for left in range(len(moments)):
            for right in range(left + 1, len(moments)):
                terms.append(
                    (moments[left][0] - moments[right][0]).square().mean()
                    + (moments[left][1] - moments[right][1]).square().mean()
                )
    if not terms:
        return expert_embeddings.sum() * 0.0
    return torch.stack(terms).mean()


class _ResidualEncoder(nn.Module):
    def __init__(self, input_dim: int, width: int, dropout: float, *, linear: bool):
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width))
        self.residual = None if linear else nn.Sequential(
            nn.GELU(), nn.Dropout(dropout), nn.Linear(width, width),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(width, width),
        )
        self.output = nn.LayerNorm(width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.input(values)
        if self.residual is not None:
            hidden = hidden + self.residual(hidden)
        return self.output(hidden)


class _MaskedAttentionPool(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1, bias=False),
        )

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 3 or mask.shape != values.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("attention pooling requires aligned bag values and bool mask")
        available = mask.any(dim=1)
        safe_mask = mask.clone()
        safe_mask[~available, 0] = True
        safe_values = values * mask.unsqueeze(-1).to(values.dtype)
        scores = self.score(safe_values).squeeze(-1)
        scores = scores.masked_fill(~safe_mask, -torch.inf)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * safe_values, dim=1)
        pooled = pooled * available.unsqueeze(-1).to(pooled.dtype)
        return pooled, available


class _ActionSetContext(nn.Module):
    """Model cross-action complementarity without using action order as time."""

    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=width,
                nhead=4,
                dim_feedforward=2 * width,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=2,
            enable_nested_tensor=False,
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or mask.shape != values.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("action context requires aligned bag values and bool mask")
        available = mask.any(dim=1)
        safe_mask = mask.clone()
        safe_mask[~available, 0] = True
        masked = values * mask.unsqueeze(-1).to(values.dtype)
        contextual = self.encoder(masked, src_key_padding_mask=~safe_mask)
        return contextual * mask.unsqueeze(-1).to(contextual.dtype)


class _TemporalEncoder(nn.Module):
    def __init__(self, width: int, dropout: float, *, use_tcn: bool):
        super().__init__()
        self.use_tcn = use_tcn
        if use_tcn:
            self.input = nn.Conv1d(95, width, kernel_size=1)
            self.blocks = nn.ModuleList((
                nn.Sequential(
                    nn.Conv1d(width, width, kernel_size=3, padding=dilation,
                              dilation=dilation),
                    nn.GELU(), nn.Dropout(dropout),
                    nn.Conv1d(width, width, kernel_size=1),
                )
                for dilation in (1, 2, 4)
            ))
            self.normalizers = nn.ModuleList(nn.GroupNorm(1, width) for _ in range(3))
        else:
            self.summary = _ResidualEncoder(95, width, dropout, linear=False)

    def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if values.ndim != 5 or values.shape[-3:] != (4, 32, 95):
            raise ValueError("temporal expert requires BxNx4x32x95 features")
        if valid.shape != values.shape[:-1] or valid.dtype != torch.bool:
            raise ValueError("temporal mask differs from the feature clock")
        batch, instances = values.shape[:2]
        masked = values * valid.unsqueeze(-1).to(values.dtype)
        if not self.use_tcn:
            support = valid.sum(dim=(-1, -2)).clamp_min(1).unsqueeze(-1)
            summary = masked.sum(dim=(-2, -3)) / support.to(values.dtype)
            return self.summary(summary)
        flat = masked.reshape(batch * instances * 4, 32, 95).transpose(1, 2)
        flat_mask = valid.reshape(batch * instances * 4, 32)
        hidden = self.input(flat)
        for block, normalizer in zip(self.blocks, self.normalizers):
            hidden = normalizer(hidden + block(hidden))
        hidden = hidden * flat_mask.unsqueeze(1).to(hidden.dtype)
        support = flat_mask.sum(dim=1).clamp_min(1).unsqueeze(1)
        windows = (hidden.sum(dim=2) / support.to(hidden.dtype)).reshape(
            batch, instances, 4, -1
        )
        return windows.mean(dim=2)


class _AUTemporalEncoder(nn.Module):
    def __init__(self, width: int, dropout: float, *, use_tcn: bool):
        super().__init__()
        self.use_tcn = use_tcn
        if use_tcn:
            self.input = nn.Conv1d(20, width, kernel_size=1)
            self.blocks = nn.ModuleList((
                nn.Sequential(
                    nn.Conv1d(width, width, kernel_size=3, padding=dilation,
                              dilation=dilation),
                    nn.GELU(), nn.Dropout(dropout),
                    nn.Conv1d(width, width, kernel_size=1),
                )
                for dilation in (1, 2, 4, 8)
            ))
            self.normalizers = nn.ModuleList(nn.GroupNorm(1, width) for _ in range(4))
        else:
            self.summary = _ResidualEncoder(20, width, dropout, linear=False)

    def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4 or values.shape[-2:] != (64, 20):
            raise ValueError("temporal AU expert requires BxNx64x20 features")
        if valid.shape != values.shape[:-1] or valid.dtype != torch.bool:
            raise ValueError("temporal AU mask differs from the feature clock")
        batch, instances = values.shape[:2]
        masked = values * valid.unsqueeze(-1).to(values.dtype)
        if not self.use_tcn:
            support = valid.sum(dim=-1).clamp_min(1).unsqueeze(-1)
            result = self.summary(masked.sum(dim=-2) / support.to(values.dtype))
        else:
            flat = masked.reshape(batch * instances, 64, 20).transpose(1, 2)
            flat_mask = valid.reshape(batch * instances, 64)
            hidden = self.input(flat)
            for block, normalizer in zip(self.blocks, self.normalizers):
                hidden = normalizer(hidden + block(hidden))
            hidden = hidden * flat_mask.unsqueeze(1).to(hidden.dtype)
            support = flat_mask.sum(dim=1).clamp_min(1).unsqueeze(1)
            result = (hidden.sum(dim=2) / support.to(hidden.dtype)).reshape(
                batch, instances, -1
            )
        return result * valid.any(dim=-1).unsqueeze(-1).to(result.dtype)


class UniversalPhenotypeMixture(nn.Module):
    """Six phenotype experts with a content-aware, source-blind gate."""

    def __init__(
        self, *, variant: str = "hybrid_tcn_mil", width: int = 64,
        dropout: float = 0.15,
    ):
        super().__init__()
        if variant not in MODEL_VARIANTS:
            raise ValueError("unknown universal phenotype architecture")
        if isinstance(width, bool) or width < 8 or not math.isfinite(float(dropout)):
            raise ValueError("model width or dropout is invalid")
        if width % 4 != 0:
            raise ValueError("model width must support four-head action attention")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        linear = variant == "linear_expert_mixture"
        self.variant = variant
        self.landmark = _ResidualEncoder(110, width, dropout, linear=linear)
        self.action_landmark = _ResidualEncoder(110, width, dropout, linear=linear)
        self.common = _ResidualEncoder(398, width, dropout, linear=linear)
        self.au = _ResidualEncoder(100, width, dropout, linear=linear)
        self.temporal = _TemporalEncoder(
            width, dropout,
            use_tcn=variant in {"hybrid_tcn_mil", "hybrid_set_transformer"},
        )
        self.au_temporal_encoder = _AUTemporalEncoder(
            width, dropout,
            use_tcn=variant in {"hybrid_tcn_mil", "hybrid_set_transformer"},
        )
        self.task_embedding = nn.Embedding(10, width)
        self.common_pool = _MaskedAttentionPool(width)
        self.action_landmark_pool = _MaskedAttentionPool(width)
        self.temporal_pool = _MaskedAttentionPool(width)
        self.au_pool = _MaskedAttentionPool(width)
        if variant == "hybrid_set_transformer":
            self.common_context = _ActionSetContext(width, dropout)
            self.action_landmark_context = _ActionSetContext(width, dropout)
            self.temporal_context = _ActionSetContext(width, dropout)
            self.au_context = _ActionSetContext(width, dropout)
        else:
            self.common_context = None
            self.action_landmark_context = None
            self.temporal_context = None
            self.au_context = None
        self.expert_heads = nn.ModuleList(nn.Linear(width, 1) for _ in range(6))
        self.gate = nn.Sequential(
            nn.Linear(12, 36), nn.GELU(), nn.Dropout(dropout), nn.Linear(36, 6),
        )

    def forward(
        self,
        landmark_original: torch.Tensor,
        landmark_mirrored: torch.Tensor,
        common_original: torch.Tensor,
        common_mirrored: torch.Tensor,
        instance_mask: torch.Tensor,
        temporal_features: torch.Tensor,
        temporal_valid_mask: torch.Tensor,
        au_instances: torch.Tensor,
        au_mask: torch.Tensor,
        au_temporal: torch.Tensor,
        au_temporal_mask: torch.Tensor,
        task_codes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = landmark_original.shape[0]
        if (
            landmark_original.shape != (batch, 110)
            or landmark_mirrored.shape != (batch, 110)
            or common_original.shape != common_mirrored.shape
            or common_original.ndim != 3
            or common_original.shape[0] != batch
            or common_original.shape[2] != 398
            or instance_mask.shape != common_original.shape[:2]
            or instance_mask.dtype != torch.bool
            or au_instances.shape != (*instance_mask.shape, 100)
            or au_mask.shape != instance_mask.shape
            or au_mask.dtype != torch.bool
            or au_temporal.shape != (*instance_mask.shape, 64, 20)
            or au_temporal_mask.shape != (*instance_mask.shape, 64)
            or au_temporal_mask.dtype != torch.bool
            or task_codes.shape != instance_mask.shape
            or task_codes.dtype != torch.long
            or torch.any(au_mask & ~instance_mask)
            or torch.any(instance_mask.sum(dim=1) == 0)
        ):
            raise ValueError("universal phenotype tensors violate the closed batch schema")
        task_index = task_codes.clamp(min=0, max=9)
        task = self.task_embedding(task_index) * instance_mask.unsqueeze(-1)

        landmark_embedding = 0.5 * (
            self.landmark(landmark_original) + self.landmark(landmark_mirrored)
        )
        common_masked_original = common_original * instance_mask.unsqueeze(-1)
        common_masked_mirrored = common_mirrored * instance_mask.unsqueeze(-1)
        action_landmark_embedding = 0.5 * (
            self.action_landmark(common_masked_original[:, :, 288:])
            + self.action_landmark(common_masked_mirrored[:, :, 288:])
        ) + task
        if self.action_landmark_context is not None:
            action_landmark_embedding = self.action_landmark_context(
                action_landmark_embedding, instance_mask
            )
        action_landmark_embedding, action_landmark_available = (
            self.action_landmark_pool(action_landmark_embedding, instance_mask)
        )
        common_embedding = 0.5 * (
            self.common(common_masked_original)
            + self.common(common_masked_mirrored)
        ) + task
        if self.common_context is not None:
            common_embedding = self.common_context(common_embedding, instance_mask)
        common_embedding, common_available = self.common_pool(
            common_embedding, instance_mask
        )

        temporal_mask = temporal_valid_mask & instance_mask[:, :, None, None]
        temporal_embedding = self.temporal(temporal_features, temporal_mask) + task
        if self.temporal_context is not None:
            temporal_embedding = self.temporal_context(
                temporal_embedding, instance_mask
            )
        temporal_embedding, temporal_available = self.temporal_pool(
            temporal_embedding, instance_mask
        )

        safe_au = au_instances * au_mask.unsqueeze(-1)
        effective_au_temporal_mask = au_temporal_mask & au_mask.unsqueeze(-1)
        safe_au_temporal = au_temporal * effective_au_temporal_mask.unsqueeze(-1)
        au_embedding = (
            self.au(safe_au)
            + self.au_temporal_encoder(
                safe_au_temporal, effective_au_temporal_mask
            )
            + self.task_embedding(task_index) * au_mask.unsqueeze(-1)
        )
        if self.au_context is not None:
            au_embedding = self.au_context(au_embedding, au_mask)
        au_embedding, au_available = self.au_pool(au_embedding, au_mask)

        embeddings = (
            landmark_embedding, action_landmark_embedding, common_embedding,
            temporal_embedding, au_embedding, au_embedding,
        )
        expert_logits = torch.cat([
            head(embedding) for head, embedding in zip(self.expert_heads, embeddings)
        ], dim=1)
        availability = torch.stack((
            torch.ones(batch, dtype=torch.bool, device=instance_mask.device),
            action_landmark_available, common_available, temporal_available,
            au_available, au_available,
        ), dim=1)
        temporal_denominator = (
            instance_mask.sum(dim=1).clamp_min(1).to(landmark_original.dtype)
            * float(4 * 32)
        )
        temporal_reliability = (
            temporal_mask.sum(dim=(1, 2, 3)).to(landmark_original.dtype)
            / temporal_denominator
        )
        au_denominator = (
            au_mask.sum(dim=1).clamp_min(1).to(landmark_original.dtype) * 64.0
        )
        au_reliability = (
            effective_au_temporal_mask.sum(dim=(1, 2)).to(landmark_original.dtype)
            / au_denominator
        )
        reliability = torch.stack((
            torch.ones(batch, dtype=landmark_original.dtype, device=instance_mask.device),
            action_landmark_available.to(landmark_original.dtype),
            common_available.to(landmark_original.dtype), temporal_reliability,
            au_reliability, au_reliability,
        ), dim=1)
        gate_inputs = torch.cat((reliability, expert_logits), dim=1)
        gate_logits = self.gate(gate_inputs).masked_fill(~availability, -torch.inf)
        gate_weights = torch.softmax(gate_logits, dim=1)
        fused = torch.sum(gate_weights * expert_logits, dim=1)
        return {
            "fused_logit": fused,
            "expert_embeddings": torch.stack(embeddings, dim=1),
            "expert_logits": expert_logits,
            "gate_weights": gate_weights,
            "expert_available": availability,
        }


def group_dro_binary_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_indices: torch.Tensor,
    group_weights: torch.Tensor,
    *,
    step_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return robust BCE and an exponential update over observed groups."""
    if (
        logits.ndim != 1
        or labels.shape != logits.shape
        or group_indices.shape != logits.shape
        or group_indices.dtype != torch.long
        or group_weights.ndim != 1
        or not logits.is_floating_point()
        or not labels.is_floating_point()
        or not group_weights.is_floating_point()
        or not bool(torch.isfinite(logits).all())
        or not bool(torch.isfinite(labels).all())
        or not bool(torch.isfinite(group_weights).all())
        or bool((group_weights <= 0).any())
        or not math.isfinite(step_size)
        or step_size <= 0.0
    ):
        raise ValueError("GroupDRO binary inputs are invalid")
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    group_losses = []
    for group in range(group_weights.numel()):
        selected = group_indices == group
        if not bool(selected.any()):
            raise ValueError("every robust group must be represented")
        group_losses.append(losses[selected].mean())
    stacked = torch.stack(group_losses)
    updated = group_weights * torch.exp(float(step_size) * stacked.detach())
    updated = updated / updated.sum()
    return torch.sum(updated * stacked), updated


__all__ = (
    "EXPERT_NAMES",
    "FIXED_FUSION_RULES",
    "MODEL_VARIANTS",
    "UniversalPhenotypeMixture",
    "fixed_source_blind_fusion",
    "group_dro_binary_loss",
    "healthy_control_alignment_loss",
)
