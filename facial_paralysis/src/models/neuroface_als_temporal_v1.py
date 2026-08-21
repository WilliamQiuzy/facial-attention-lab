"""Compact participant-level temporal model for the NeuroFace ALS endpoint.

The model consumes the three frozen primary tasks together and emits one logit
per participant.  Windows and task recordings are therefore never treated as
independent labelled examples.  Invalid detector rows are zeroed before any
learned operation, and derivatives never bridge a missing row or a window
boundary.
"""
from __future__ import annotations

from typing import Final

import torch
from torch import nn
from torch.nn import functional as F


PRIMARY_TASKS: Final[tuple[str, ...]] = (
    "NSM_KISS",
    "NSM_OPEN",
    "NSM_SPREAD",
)
TASK_COUNT: Final[int] = len(PRIMARY_TASKS)
WINDOW_COUNT: Final[int] = 4
FRAME_COUNT: Final[int] = 32
FEATURE_COUNT: Final[int] = 95
PARAMETER_CAP: Final[int] = 25_000


def _validate_inputs(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    timestamps: torch.Tensor,
) -> None:
    if not all(isinstance(value, torch.Tensor) for value in (
        features, valid_mask, timestamps
    )):
        raise ValueError("features, valid_mask, and timestamps must be tensors")
    expected_features = (TASK_COUNT, WINDOW_COUNT, FRAME_COUNT, FEATURE_COUNT)
    if features.ndim != 5 or tuple(features.shape[1:]) != expected_features:
        raise ValueError(
            "features must have shape (batch, 3 tasks, 4 windows, 32 frames, 95 features)"
        )
    expected_rows = tuple(features.shape[:-1])
    if tuple(valid_mask.shape) != expected_rows or tuple(timestamps.shape) != expected_rows:
        raise ValueError("mask and timestamps must match the feature leading dimensions")
    if features.device != valid_mask.device or features.device != timestamps.device:
        raise ValueError("all inputs must share one device")
    if not features.is_floating_point() or not timestamps.is_floating_point():
        raise ValueError("features and timestamps must be floating tensors")
    if valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must have bool dtype")
    if not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(timestamps).all()):
        raise ValueError("features and timestamps must be finite")
    if bool((timestamps[..., 1:] <= timestamps[..., :-1]).any()):
        raise ValueError("timestamps must increase strictly within every window")
    if bool((~valid_mask.any(dim=-1)).any()):
        raise ValueError("every task window must contain at least one valid frame")


def within_window_velocity(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    timestamps: torch.Tensor,
) -> torch.Tensor:
    """Return per-second differences without crossing a window or mask gap."""
    _validate_inputs(features, valid_mask, timestamps)
    velocity = torch.zeros_like(features)
    endpoint_valid = valid_mask[..., 1:] & valid_mask[..., :-1]
    elapsed = timestamps[..., 1:] - timestamps[..., :-1]
    safe_elapsed = torch.where(endpoint_valid, elapsed, torch.ones_like(elapsed))
    differences = (
        features[..., 1:, :] - features[..., :-1, :]
    ) / safe_elapsed.unsqueeze(-1)
    differences = torch.where(
        endpoint_valid.unsqueeze(-1), differences, torch.zeros_like(differences)
    )
    if not bool(torch.isfinite(differences).all()):
        raise ValueError("valid within-window velocities must be finite")
    velocity[..., 1:, :] = differences
    return velocity


def _masked_statistics(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Concatenate masked mean, maximum, and minimum along the frame axis."""
    mask = valid_mask.unsqueeze(1)
    count = mask.sum(dim=-1).clamp_min(1)
    mean = (values * mask).sum(dim=-1) / count
    high = torch.finfo(values.dtype).max
    maximum = values.masked_fill(~mask, -high).max(dim=-1).values
    minimum = values.masked_fill(~mask, high).min(dim=-1).values
    return torch.cat((mean, maximum, minimum), dim=1)


class TaskAwareTemporalALSClassifier(nn.Module):
    """Small task-aware TCN that produces exactly one participant logit."""

    def __init__(self) -> None:
        super().__init__()
        input_channels = FEATURE_COUNT * 2
        self.temporal = nn.Sequential(
            nn.Conv1d(
                input_channels, input_channels, kernel_size=5,
                padding=2, groups=input_channels,
            ),
            nn.Conv1d(input_channels, 48, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(48, 48, kernel_size=3, padding=2, dilation=2, groups=48),
            nn.Conv1d(48, 24, kernel_size=1),
            nn.GELU(),
        )
        self.window_projection = nn.Sequential(
            nn.Linear(24 * 3, 24),
            nn.GELU(),
        )
        self.task_projection = nn.Sequential(
            nn.Linear(24 * 4, 32),
            nn.GELU(),
        )
        self.participant_head = nn.Sequential(
            nn.Linear(32 * TASK_COUNT, 32),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(32, 1),
        )
        if count_parameters(self) > PARAMETER_CAP:
            raise RuntimeError("temporal classifier exceeded its frozen parameter cap")

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        _validate_inputs(features, valid_mask, timestamps)
        clean = torch.where(
            valid_mask.unsqueeze(-1), features, torch.zeros_like(features)
        )
        velocity = within_window_velocity(clean, valid_mask, timestamps)
        joined = torch.cat((clean, velocity), dim=-1)
        batch_size = features.shape[0]
        flat = joined.reshape(
            batch_size * TASK_COUNT * WINDOW_COUNT, FRAME_COUNT, FEATURE_COUNT * 2
        ).transpose(1, 2)
        flat_mask = valid_mask.reshape(
            batch_size * TASK_COUNT * WINDOW_COUNT, FRAME_COUNT
        )
        encoded = self.temporal(flat)
        encoded = encoded * flat_mask.unsqueeze(1)
        window_stats = _masked_statistics(encoded, flat_mask)
        window_embeddings = self.window_projection(window_stats).reshape(
            batch_size, TASK_COUNT, WINDOW_COUNT, 24
        )

        task_mean = window_embeddings.mean(dim=2)
        task_maximum = window_embeddings.max(dim=2).values
        task_minimum = window_embeddings.min(dim=2).values
        task_change = window_embeddings[:, :, -1, :] - window_embeddings[:, :, 0, :]
        task_summary = torch.cat(
            (task_mean, task_maximum, task_minimum, task_change), dim=-1
        )
        task_embeddings = self.task_projection(task_summary)
        logits = self.participant_head(task_embeddings.reshape(batch_size, -1))
        return logits.squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    """Count trainable scalar parameters."""
    if not isinstance(model, nn.Module):
        raise ValueError("model must be a torch module")
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def mirror_mean_probability(
    model: nn.Module,
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    timestamps: torch.Tensor,
) -> torch.Tensor:
    """Average original and mirrored predictions in probability space."""
    from .dynamic_landmark import horizontal_mirror_features

    original = torch.sigmoid(model(features, valid_mask, timestamps))
    mirrored = horizontal_mirror_features(features)
    mirror_probability = torch.sigmoid(model(mirrored, valid_mask, timestamps))
    return 0.5 * (original + mirror_probability)


def participant_balanced_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Binary cross entropy with one equally weighted row per participant."""
    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise ValueError("logits and labels must be tensors")
    if logits.ndim != 1 or labels.ndim != 1 or logits.shape != labels.shape:
        raise ValueError("logits and labels must be equal participant-level vectors")
    if logits.numel() < 2:
        raise ValueError("participant loss requires at least two participants")
    if not logits.is_floating_point() or not labels.is_floating_point():
        raise ValueError("logits and labels must be floating tensors")
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(labels).all()):
        raise ValueError("logits and labels must be finite")
    if not bool(((labels == 0.0) | (labels == 1.0)).all()):
        raise ValueError("participant labels must be binary")
    return F.binary_cross_entropy_with_logits(logits, labels)


__all__ = (
    "PARAMETER_CAP",
    "PRIMARY_TASKS",
    "TaskAwareTemporalALSClassifier",
    "count_parameters",
    "mirror_mean_probability",
    "participant_balanced_bce",
    "within_window_velocity",
)
