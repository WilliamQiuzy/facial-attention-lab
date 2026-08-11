"""Compact candidate architectures for development-only landmark research."""
from __future__ import annotations

import math
from typing import Final

import torch
from torch import nn


CANDIDATE_ORDER: Final[tuple[str, ...]] = (
    "logistic_110d",
    "extra_trees_110d",
    "hist_gradient_boosting_110d",
    "mlp_110d",
    "tcn_landmark23",
    "bigru_landmark23",
    "transformer_landmark23",
    "region_tcn_landmark23",
    "hybrid_110d_tcn",
)
CLASSICAL_CANDIDATES: Final[tuple[str, ...]] = CANDIDATE_ORDER[:3]
NEURAL_CANDIDATES: Final[tuple[str, ...]] = CANDIDATE_ORDER[3:]
LANDMARK_OFFSET: Final[int] = 72
LANDMARK_DIM: Final[int] = 23
SUMMARY_DIM: Final[int] = 110
PARAMETER_CAP: Final[int] = 300_000


def _validate_inputs(
    raw_features: torch.Tensor,
    valid_mask: torch.Tensor,
    summary_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not all(isinstance(value, torch.Tensor) for value in (
        raw_features, valid_mask, summary_features
    )):
        raise ValueError("architecture-search inputs must be torch tensors")
    if raw_features.ndim != 4 or raw_features.shape[1:] != (4, 32, 95):
        raise ValueError("raw_features must have shape (batch, 4, 32, 95)")
    if valid_mask.shape != raw_features.shape[:-1] or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool with raw feature leading shape")
    if summary_features.shape != (raw_features.shape[0], SUMMARY_DIM):
        raise ValueError("summary_features must have shape (batch, 110)")
    if not raw_features.is_floating_point() or not summary_features.is_floating_point():
        raise ValueError("architecture-search features must be floating point")
    if not torch.isfinite(raw_features[valid_mask]).all():
        raise ValueError("valid raw feature values must be finite")
    if not torch.isfinite(summary_features).all():
        raise ValueError("summary feature values must be finite")
    if bool((~valid_mask.reshape(valid_mask.shape[0], -1).any(dim=1)).any()):
        raise ValueError("every recording must contain a valid frame")
    landmarks = raw_features[..., LANDMARK_OFFSET:]
    landmarks = torch.where(
        valid_mask.unsqueeze(-1), landmarks, torch.zeros_like(landmarks)
    )
    return landmarks, valid_mask


def _masked_window_pool(encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pool `(batch*window, time, channels)` without crossing window gaps."""
    present = mask.any(dim=1)
    weights = mask.to(encoded.dtype).unsqueeze(-1)
    mean = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    floor = torch.finfo(encoded.dtype).min
    maximum = encoded.masked_fill(~mask.unsqueeze(-1), floor).max(dim=1).values
    maximum = torch.where(present.unsqueeze(-1), maximum, torch.zeros_like(maximum))
    pooled = 0.5 * (mean + maximum)
    return torch.where(present.unsqueeze(-1), pooled, torch.zeros_like(pooled))


def _recording_pool(window_vectors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    batch = mask.shape[0]
    present = mask.any(dim=-1)
    vectors = window_vectors.reshape(batch, 4, -1)
    weights = present.to(vectors.dtype).unsqueeze(-1)
    return (vectors * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class _WindowTCN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 48):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, 5, padding=2, groups=input_dim),
            nn.Conv1d(input_dim, 48, 1),
            nn.GELU(),
            nn.Conv1d(48, 48, 3, padding=2, dilation=2, groups=48),
            nn.Conv1d(48, output_dim, 1),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, windows, frames, channels = values.shape
        flat = values.reshape(batch * windows, frames, channels)
        flat_mask = mask.reshape(batch * windows, frames)
        encoded = self.network(flat.transpose(1, 2)).transpose(1, 2)
        return _masked_window_pool(encoded, flat_mask)


class _BaseCandidate(nn.Module):
    candidate_name: str

    def _finish(self, logits: torch.Tensor, batch_size: int) -> torch.Tensor:
        logits = logits.reshape(-1)
        if logits.shape != (batch_size,) or not torch.isfinite(logits).all():
            raise RuntimeError(f"{self.candidate_name} produced invalid logits")
        return logits


class SummaryMLP(_BaseCandidate):
    candidate_name = "mlp_110d"

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(SUMMARY_DIM, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 24),
            nn.GELU(),
            nn.Linear(24, 1),
        )

    def forward(self, raw_features, valid_mask, summary_features):
        _validate_inputs(raw_features, valid_mask, summary_features)
        return self._finish(self.network(summary_features), raw_features.shape[0])


class TCNCandidate(_BaseCandidate):
    candidate_name = "tcn_landmark23"

    def __init__(self):
        super().__init__()
        self.encoder = _WindowTCN(LANDMARK_DIM, 48)
        self.head = nn.Sequential(nn.Linear(48, 24), nn.GELU(), nn.Linear(24, 1))

    def forward(self, raw_features, valid_mask, summary_features):
        landmarks, mask = _validate_inputs(raw_features, valid_mask, summary_features)
        windows = self.encoder(landmarks, mask)
        recording = _recording_pool(windows, mask)
        return self._finish(self.head(recording), raw_features.shape[0])


class BiGRUCandidate(_BaseCandidate):
    candidate_name = "bigru_landmark23"

    def __init__(self):
        super().__init__()
        self.input_projection = nn.Linear(LANDMARK_DIM, 40)
        self.gru = nn.GRU(40, 32, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(64, 24), nn.GELU(), nn.Linear(24, 1))

    def forward(self, raw_features, valid_mask, summary_features):
        landmarks, mask = _validate_inputs(raw_features, valid_mask, summary_features)
        batch = landmarks.shape[0]
        flat = landmarks.reshape(batch * 4, 32, LANDMARK_DIM)
        flat_mask = mask.reshape(batch * 4, 32)
        encoded, _ = self.gru(self.input_projection(flat))
        recording = _recording_pool(_masked_window_pool(encoded, flat_mask), mask)
        return self._finish(self.head(recording), batch)


class TransformerCandidate(_BaseCandidate):
    candidate_name = "transformer_landmark23"

    def __init__(self):
        super().__init__()
        width = 48
        self.input_projection = nn.Linear(LANDMARK_DIM, width)
        self.position = nn.Parameter(torch.zeros(1, 32, width))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=4, dim_feedforward=96, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, raw_features, valid_mask, summary_features):
        landmarks, mask = _validate_inputs(raw_features, valid_mask, summary_features)
        batch = landmarks.shape[0]
        flat = landmarks.reshape(batch * 4, 32, LANDMARK_DIM)
        flat_mask = mask.reshape(batch * 4, 32)
        safe_mask = flat_mask.clone()
        absent = ~safe_mask.any(dim=1)
        safe_mask[absent, 0] = True
        tokens = self.input_projection(flat) + self.position
        tokens[absent, 0] = 0.0
        encoded = self.encoder(tokens, src_key_padding_mask=~safe_mask)
        window = _masked_window_pool(encoded, flat_mask)
        recording = _recording_pool(window, mask)
        return self._finish(self.head(recording), batch)


class RegionTCNCandidate(_BaseCandidate):
    candidate_name = "region_tcn_landmark23"

    def __init__(self):
        super().__init__()
        self.eye = _WindowTCN(10, 24)
        self.brow = _WindowTCN(4, 16)
        self.mouth = _WindowTCN(9, 24)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, raw_features, valid_mask, summary_features):
        landmarks, mask = _validate_inputs(raw_features, valid_mask, summary_features)
        windows = torch.cat((
            self.eye(landmarks[..., :10], mask),
            self.brow(landmarks[..., 10:14], mask),
            self.mouth(landmarks[..., 14:], mask),
        ), dim=-1)
        recording = _recording_pool(windows, mask)
        return self._finish(self.head(recording), raw_features.shape[0])


class HybridCandidate(_BaseCandidate):
    candidate_name = "hybrid_110d_tcn"

    def __init__(self):
        super().__init__()
        self.temporal = _WindowTCN(LANDMARK_DIM, 48)
        self.summary = nn.Sequential(nn.Linear(SUMMARY_DIM, 48), nn.GELU(), nn.Linear(48, 24))
        self.head = nn.Sequential(nn.Linear(72, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, raw_features, valid_mask, summary_features):
        landmarks, mask = _validate_inputs(raw_features, valid_mask, summary_features)
        temporal = _recording_pool(self.temporal(landmarks, mask), mask)
        summary = self.summary(summary_features)
        return self._finish(self.head(torch.cat((temporal, summary), dim=-1)), raw_features.shape[0])


_BUILDERS = {
    "mlp_110d": SummaryMLP,
    "tcn_landmark23": TCNCandidate,
    "bigru_landmark23": BiGRUCandidate,
    "transformer_landmark23": TransformerCandidate,
    "region_tcn_landmark23": RegionTCNCandidate,
    "hybrid_110d_tcn": HybridCandidate,
}


def count_trainable_parameters(model: nn.Module) -> int:
    if not isinstance(model, nn.Module):
        raise ValueError("model must be a torch module")
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_neural_candidate(name: str) -> nn.Module:
    if not isinstance(name, str) or name not in _BUILDERS:
        raise ValueError(f"unknown neural architecture-search candidate {name!r}")
    model = _BUILDERS[name]()
    count = count_trainable_parameters(model)
    if not 0 < count < PARAMETER_CAP:
        raise RuntimeError(f"{name} parameter count {count} violates the fixed capacity cap")
    return model


__all__ = [
    "CANDIDATE_ORDER", "CLASSICAL_CANDIDATES", "NEURAL_CANDIDATES",
    "PARAMETER_CAP", "build_neural_candidate", "count_trainable_parameters",
]
