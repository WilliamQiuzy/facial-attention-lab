"""Compact motion-quality encoder locked for NeuroFace pretraining v1."""
from __future__ import annotations

import torch
from torch import nn


LANDMARK_DIM = 23
WINDOWS = 4
FRAMES = 32
WINDOW_WIDTH = 24
EMBEDDING_DIM = 32
TASK_COUNT = 9
DOMAIN_COUNT = 5
PARAMETER_CAP = 30_000


def _validate_sequence(
    landmarks: torch.Tensor,
    valid_mask: torch.Tensor,
    timestamps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not all(isinstance(value, torch.Tensor) for value in (
        landmarks, valid_mask, timestamps
    )):
        raise ValueError("motion inputs must be torch tensors")
    if landmarks.ndim != 4 or landmarks.shape[1:] != (WINDOWS, FRAMES, LANDMARK_DIM):
        raise ValueError("landmarks must have shape (batch, 4, 32, 23)")
    if valid_mask.shape != landmarks.shape[:-1] or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool with shape (batch, 4, 32)")
    if timestamps.shape != valid_mask.shape or not timestamps.is_floating_point():
        raise ValueError("timestamps must be floating point with shape (batch, 4, 32)")
    if not landmarks.is_floating_point():
        raise ValueError("landmarks must be floating point")
    if bool((~valid_mask.reshape(valid_mask.shape[0], -1).any(dim=1)).any()):
        raise ValueError("every recording must contain a valid frame")
    if not torch.isfinite(landmarks[valid_mask]).all():
        raise ValueError("valid landmark values must be finite")
    if not torch.isfinite(timestamps[valid_mask]).all():
        raise ValueError("valid timestamps must be finite")
    clean_landmarks = torch.where(
        valid_mask.unsqueeze(-1), landmarks, torch.zeros_like(landmarks)
    )
    clean_timestamps = torch.where(valid_mask, timestamps, torch.zeros_like(timestamps))
    return clean_landmarks, valid_mask, clean_timestamps


def within_window_velocity(
    landmarks: torch.Tensor,
    valid_mask: torch.Tensor,
    timestamps: torch.Tensor,
) -> torch.Tensor:
    """Compute per-second first differences without crossing window boundaries."""
    values, mask, times = _validate_sequence(landmarks, valid_mask, timestamps)
    delta = torch.zeros_like(values)
    pair = mask[..., 1:] & mask[..., :-1]
    dt = times[..., 1:] - times[..., :-1]
    if bool((pair & (dt <= 0)).any()):
        raise ValueError("adjacent valid timestamps must increase within each window")
    safe_dt = torch.where(pair, dt, torch.ones_like(dt))
    observed = (values[..., 1:, :] - values[..., :-1, :]) / safe_dt.unsqueeze(-1)
    delta[..., 1:, :] = torch.where(
        pair.unsqueeze(-1), observed, torch.zeros_like(observed)
    )
    return delta


def _masked_temporal_pool(encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(encoded.dtype).unsqueeze(-1)
    mean = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    floor = torch.finfo(encoded.dtype).min
    maximum = encoded.masked_fill(~mask.unsqueeze(-1), floor).max(dim=1).values
    present = mask.any(dim=1)
    maximum = torch.where(present.unsqueeze(-1), maximum, torch.zeros_like(maximum))
    return torch.where(
        present.unsqueeze(-1), 0.5 * (mean + maximum), torch.zeros_like(mean)
    )


def _window_mean(values: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    weights = present.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class MotionQualityEncoder(nn.Module):
    """Encode overall motion plus fixed early-to-late change into 32 dimensions."""

    def __init__(self):
        super().__init__()
        channels = LANDMARK_DIM * 2
        self.temporal = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2, groups=channels),
            nn.Conv1d(channels, 48, 1),
            nn.GELU(),
            nn.Conv1d(48, 48, 3, padding=2, dilation=2, groups=48),
            nn.Conv1d(48, WINDOW_WIDTH, 1),
            nn.GELU(),
        )
        self.projection = nn.Sequential(
            nn.Linear(WINDOW_WIDTH * 2, EMBEDDING_DIM),
            nn.GELU(),
            nn.LayerNorm(EMBEDDING_DIM),
        )

    def forward(
        self,
        landmarks: torch.Tensor,
        valid_mask: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        values, mask, times = _validate_sequence(landmarks, valid_mask, timestamps)
        velocity = within_window_velocity(values, mask, times)
        motion = torch.cat((values, velocity), dim=-1)
        batch = values.shape[0]
        flat = motion.reshape(batch * WINDOWS, FRAMES, -1)
        flat_mask = mask.reshape(batch * WINDOWS, FRAMES)
        encoded = self.temporal(flat.transpose(1, 2)).transpose(1, 2)
        windows = _masked_temporal_pool(encoded, flat_mask).reshape(
            batch, WINDOWS, WINDOW_WIDTH
        )
        present = mask.any(dim=-1)
        overall = _window_mean(windows, present)
        early = _window_mean(windows[:, :2], present[:, :2])
        late = _window_mean(windows[:, 2:], present[:, 2:])
        if bool((~present[:, :2].any(dim=1)).any() or (~present[:, 2:].any(dim=1)).any()):
            raise ValueError("motion trend requires an early and late observed window")
        embedding = self.projection(torch.cat((overall, late - early), dim=-1))
        if embedding.shape != (batch, EMBEDDING_DIM) or not torch.isfinite(embedding).all():
            raise RuntimeError("motion encoder produced invalid embeddings")
        return embedding


class MotionQualityRegressor(nn.Module):
    """Five-domain SLP head; the task embedding is excluded from transfer."""

    def __init__(self):
        super().__init__()
        self.encoder = MotionQualityEncoder()
        self.task_embedding = nn.Embedding(TASK_COUNT, 8)
        self.head = nn.Sequential(
            nn.Linear(EMBEDDING_DIM + 8, 32),
            nn.GELU(),
            nn.Linear(32, DOMAIN_COUNT),
        )

    def forward(
        self,
        landmarks: torch.Tensor,
        valid_mask: torch.Tensor,
        timestamps: torch.Tensor,
        task_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            not isinstance(task_indices, torch.Tensor)
            or task_indices.shape != (landmarks.shape[0],)
            or task_indices.dtype != torch.long
            or bool(((task_indices < 0) | (task_indices >= TASK_COUNT)).any())
        ):
            raise ValueError("task_indices must contain one valid int64 task per recording")
        embedding = self.encoder(landmarks, valid_mask, timestamps)
        logits = self.head(torch.cat((embedding, self.task_embedding(task_indices)), dim=-1))
        if logits.shape != (landmarks.shape[0], DOMAIN_COUNT) or not torch.isfinite(logits).all():
            raise RuntimeError("motion regressor produced invalid logits")
        return logits, embedding


def count_parameters(model: nn.Module) -> int:
    if not isinstance(model, nn.Module):
        raise ValueError("model must be a torch module")
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


__all__ = [
    "DOMAIN_COUNT", "EMBEDDING_DIM", "LANDMARK_DIM", "MotionQualityEncoder",
    "MotionQualityRegressor", "PARAMETER_CAP", "count_parameters",
    "within_window_velocity",
]
