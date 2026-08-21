"""Label-free clinical kinematic auxiliary supervision for shared V9."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from src.models.anatomical_relational_router_v9 import anatomical_region_indices
from src.preprocessing.generalization_110d import (
    LANDMARK_MI_110D,
    candidate_feature_names,
)


KINEMATIC_TARGET_NAMES = (
    "eye_excursion",
    "eye_velocity",
    "eye_bilateral_synchrony",
    "brow_excursion",
    "brow_velocity",
    "brow_bilateral_synchrony",
    "oral_excursion",
    "oral_velocity",
    "oral_bilateral_synchrony",
)
_FROZEN_FEATURE_NAMES = candidate_feature_names(LANDMARK_MI_110D)
_REGIONS = anatomical_region_indices(_FROZEN_FEATURE_NAMES)
_STATISTIC_SUFFIXES = (
    "__range",
    "__max_abs_velocity_per_second",
    "__correlation",
)
_TARGET_INDICES = tuple(
    tuple(
        index for index in _REGIONS[region]
        if _FROZEN_FEATURE_NAMES[index].endswith(suffix)
    )
    for region in ("eye", "brow", "oral")
    for suffix in _STATISTIC_SUFFIXES
)
if tuple(len(row) for row in _TARGET_INDICES) != (10, 10, 3, 4, 4, 1, 9, 9, 2):
    raise RuntimeError("clinical kinematic target binding drifted")


@dataclass(frozen=True)
class KinematicTargetScalerV9:
    mean: torch.Tensor
    scale: torch.Tensor


class ClinicalKinematicAuxiliaryHeadV9(nn.Module):
    """Predict interpretable action kinematics from the shared action token."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, len(KINEMATIC_TARGET_NAMES)),
        )

    def forward(self, action_tokens: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(action_tokens, torch.Tensor)
            or action_tokens.ndim != 3
            or action_tokens.shape[-1] != 64
            or not action_tokens.is_floating_point()
            or not bool(torch.isfinite(action_tokens).all())
        ):
            raise ValueError("kinematic auxiliary head requires finite action tokens")
        return self.network(action_tokens)


def clinical_kinematic_targets(
    clinical_original: torch.Tensor,
    clinical_mirrored: torch.Tensor,
    action_mask: torch.Tensor,
    feature_names: tuple[str, ...],
) -> torch.Tensor:
    """Derive nine label-free eye/brow/oral motion targets from exact 110D."""
    if (
        type(feature_names) is not tuple
        or feature_names != _FROZEN_FEATURE_NAMES
        or not isinstance(clinical_original, torch.Tensor)
        or clinical_original.ndim != 3
        or clinical_original.shape[-1] != 110
        or not clinical_original.is_floating_point()
        or not bool(torch.isfinite(clinical_original).all())
        or clinical_mirrored.shape != clinical_original.shape
        or not clinical_mirrored.is_floating_point()
        or clinical_mirrored.device != clinical_original.device
        or not bool(torch.isfinite(clinical_mirrored).all())
        or action_mask.shape != clinical_original.shape[:2]
        or action_mask.dtype != torch.bool
        or action_mask.device != clinical_original.device
    ):
        raise ValueError("clinical kinematic targets require exact finite 110D evidence")
    paired = 0.5 * (clinical_original + clinical_mirrored)
    columns = []
    for indices in _TARGET_INDICES:
        index = torch.tensor(indices, dtype=torch.long, device=paired.device)
        columns.append(paired.index_select(-1, index).mean(dim=-1))
    targets = torch.stack(columns, dim=-1)
    targets = targets * action_mask.unsqueeze(-1).to(targets.dtype)
    if not bool(torch.isfinite(targets).all()):
        raise RuntimeError("clinical kinematic targets are nonfinite")
    return targets


def fit_kinematic_target_scaler(
    targets: torch.Tensor,
    action_mask: torch.Tensor,
) -> KinematicTargetScalerV9:
    if (
        not isinstance(targets, torch.Tensor)
        or targets.ndim != 3
        or targets.shape[-1] != len(KINEMATIC_TARGET_NAMES)
        or not targets.is_floating_point()
        or not bool(torch.isfinite(targets).all())
        or action_mask.shape != targets.shape[:2]
        or action_mask.dtype != torch.bool
        or action_mask.device != targets.device
        or not bool(action_mask.any())
    ):
        raise ValueError("target scaling requires valid training-fold actions")
    selected = targets[action_mask]
    mean = selected.mean(dim=0)
    observed_scale = selected.std(dim=0, unbiased=False)
    scale = torch.where(
        observed_scale > 1e-6, observed_scale, torch.ones_like(observed_scale)
    )
    return KinematicTargetScalerV9(mean=mean.detach().clone(), scale=scale.detach().clone())


def clinical_kinematic_auxiliary_loss(
    head: ClinicalKinematicAuxiliaryHeadV9,
    action_tokens: torch.Tensor,
    targets: torch.Tensor,
    action_mask: torch.Tensor,
    scaler: KinematicTargetScalerV9,
) -> torch.Tensor:
    if (
        type(head) is not ClinicalKinematicAuxiliaryHeadV9
        or type(scaler) is not KinematicTargetScalerV9
        or targets.shape != action_tokens.shape[:2] + (len(KINEMATIC_TARGET_NAMES),)
        or targets.device != action_tokens.device
        or not targets.is_floating_point()
        or not bool(torch.isfinite(targets).all())
        or action_mask.shape != action_tokens.shape[:2]
        or action_mask.dtype != torch.bool
        or action_mask.device != action_tokens.device
        or not bool(action_mask.any())
        or scaler.mean.shape != (len(KINEMATIC_TARGET_NAMES),)
        or scaler.scale.shape != scaler.mean.shape
        or scaler.mean.device != action_tokens.device
        or scaler.scale.device != action_tokens.device
        or not bool(torch.isfinite(scaler.mean).all())
        or not bool(torch.isfinite(scaler.scale).all())
        or bool((scaler.scale <= 0.0).any())
    ):
        raise ValueError("kinematic auxiliary loss received malformed fold evidence")
    predicted = head(action_tokens)
    standardized = (targets - scaler.mean) / scaler.scale
    return F.smooth_l1_loss(
        predicted[action_mask], standardized[action_mask], reduction="mean"
    )


__all__ = [
    "KINEMATIC_TARGET_NAMES",
    "ClinicalKinematicAuxiliaryHeadV9",
    "KinematicTargetScalerV9",
    "clinical_kinematic_auxiliary_loss",
    "clinical_kinematic_targets",
    "fit_kinematic_target_scaler",
]
