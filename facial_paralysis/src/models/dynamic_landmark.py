"""Equal-shape temporal model for frozen 72 + 23 MediaPipe trajectories.

The three ablations instantiate exactly the same parameters.  A candidate is
defined only by deterministic zeroing of its inactive input block, which keeps
architecture and optimization capacity comparable across arms.
"""
from __future__ import annotations

from typing import Final

import torch
from torch import nn

from ..datasets.dynamic_landmark import DYNAMIC_FEATURE_NAMES


ARM_BLENDSHAPE: Final[str] = "blendshape_only"
ARM_LANDMARK: Final[str] = "landmark_only"
ARM_FUSION: Final[str] = "fusion"
DYNAMIC_NEURAL_ARMS: Final[tuple[str, ...]] = (
    ARM_BLENDSHAPE, ARM_LANDMARK, ARM_FUSION,
)
BLENDSHAPE_DIM: Final[int] = 72
LANDMARK_DIM: Final[int] = 23
FRAME_LATENT_DIM: Final[int] = 64


def _mirror_contract() -> tuple[torch.Tensor, torch.Tensor]:
    names = tuple(DYNAMIC_FEATURE_NAMES)
    if len(names) != BLENDSHAPE_DIM + LANDMARK_DIM:
        raise RuntimeError("dynamic feature schema is not the frozen 95-column layout")
    indices = list(range(len(names)))
    signs = [1.0] * len(names)
    by_name = {name: index for index, name in enumerate(names)}

    # MediaPipe's first 52 columns use explicit Left/Right category names.
    for index, name in enumerate(names[:52]):
        if name.endswith("Left"):
            partner = name[:-4] + "Right"
        elif name.endswith("Right"):
            partner = name[:-5] + "Left"
        else:
            continue
        if partner not in by_name or by_name[partner] >= 52:
            raise RuntimeError(f"missing mirror partner for {name!r}")
        indices[index] = by_name[partner]

    # The following 20 columns are registered left-minus-right differences.
    for index, name in enumerate(names[52:BLENDSHAPE_DIM], start=52):
        if not name.startswith("delta_left_minus_right_"):
            raise RuntimeError("unexpected blendshape asymmetry column order")
        signs[index] = -1.0

    clinical_pairs = (
        ("fissure_h_mesh33", "fissure_h_mesh263"),
        ("fissure_w_mesh33", "fissure_w_mesh263"),
        ("eye_area_mesh33", "eye_area_mesh263"),
        ("brow_h_mesh33", "brow_h_mesh263"),
        ("corner_y_mesh61", "corner_y_mesh291"),
        ("corner_x_mesh61", "corner_x_mesh291"),
    )
    for first, second in clinical_pairs:
        left, right = by_name[first], by_name[second]
        indices[left], indices[right] = right, left
    for name in (
        "fissure_h_mesh33_minus_mesh263",
        "brow_h_mesh33_minus_mesh263",
        "corner_y_mesh61_minus_mesh291",
    ):
        signs[by_name[name]] = -1.0

    for index, partner in enumerate(indices):
        if indices[partner] != index or signs[index] != signs[partner]:
            raise RuntimeError("horizontal mirror schema is not an involution")
    return torch.tensor(indices, dtype=torch.int64), torch.tensor(signs)


_MIRROR_INDICES, _MIRROR_SIGNS = _mirror_contract()


def horizontal_mirror_features(features: torch.Tensor) -> torch.Tensor:
    """Mirror a raw clinical23_v2 feature tensor along its final dimension."""
    if not isinstance(features, torch.Tensor):
        raise ValueError("features must be a torch tensor")
    if features.ndim < 1 or features.shape[-1] != 95:
        raise ValueError("horizontal mirror requires exactly 95 feature columns")
    if not features.is_floating_point() or not torch.isfinite(features).all():
        raise ValueError("horizontal mirror features must be finite floating values")
    indices = _MIRROR_INDICES.to(device=features.device)
    signs = _MIRROR_SIGNS.to(device=features.device, dtype=features.dtype)
    return features.index_select(-1, indices) * signs


def _validate_temporal_inputs(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    timestamps: torch.Tensor,
    source_frame_indices: torch.Tensor,
) -> None:
    if not all(isinstance(item, torch.Tensor) for item in (
        features, valid_mask, timestamps, source_frame_indices
    )):
        raise ValueError("dynamic model inputs must be torch tensors")
    if features.ndim < 3:
        raise ValueError("features require at least batch, time, and feature dimensions")
    expected = features.shape[:-1]
    if valid_mask.shape != expected or timestamps.shape != expected or source_frame_indices.shape != expected:
        raise ValueError("mask, timestamps, and source indices must match feature leading dimensions")
    devices = {
        features.device, valid_mask.device, timestamps.device,
        source_frame_indices.device,
    }
    if len(devices) != 1:
        raise ValueError("features, mask, timestamps, and source indices must share one device")
    if valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must have bool dtype")
    if not features.is_floating_point() or not timestamps.is_floating_point():
        raise ValueError("features and timestamps must have floating dtype")
    if source_frame_indices.dtype != torch.int64:
        raise ValueError("source_frame_indices must have int64 dtype")
    if not torch.isfinite(features).all() or not torch.isfinite(timestamps).all():
        raise ValueError("features and timestamps must contain only finite values")
    if bool((source_frame_indices < 0).any()):
        raise ValueError("source frame indices must be nonnegative")
    if bool((source_frame_indices[..., 1:] <= source_frame_indices[..., :-1]).any()):
        raise ValueError("source frame indices must increase strictly within every window")
    if bool((timestamps[..., 1:] <= timestamps[..., :-1]).any()):
        raise ValueError("timestamps must increase strictly within every window")


def gap_safe_per_second_differences(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    timestamps: torch.Tensor,
    source_frame_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiate only adjacent source frames with two valid endpoints."""
    _validate_temporal_inputs(features, valid_mask, timestamps, source_frame_indices)
    differences = torch.zeros_like(features)
    difference_mask = torch.zeros_like(valid_mask)
    if features.shape[-2] < 2:
        return differences, difference_mask
    previous_index = source_frame_indices[..., :-1]
    current_index = source_frame_indices[..., 1:]
    max_int = torch.iinfo(torch.int64).max
    adjacent = (
        (previous_index < max_int)
        & (current_index > previous_index)
        & ((current_index - previous_index) == 1)
    )
    endpoint_valid = valid_mask[..., :-1] & valid_mask[..., 1:]
    valid_difference = adjacent & endpoint_valid
    elapsed = timestamps[..., 1:] - timestamps[..., :-1]
    safe_elapsed = torch.where(valid_difference, elapsed, torch.ones_like(elapsed))
    raw = (features[..., 1:, :] - features[..., :-1, :]) / safe_elapsed.unsqueeze(-1)
    raw = torch.where(valid_difference.unsqueeze(-1), raw, torch.zeros_like(raw))
    if not torch.isfinite(raw).all():
        raise ValueError("valid per-second differences must be finite")
    differences[..., 1:, :] = raw
    difference_mask[..., 1:] = valid_difference
    return differences, difference_mask


class DynamicLandmarkModel(nn.Module):
    """One fixed-capacity binary temporal model for all modality ablations."""

    def __init__(self, arm: str = ARM_FUSION):
        super().__init__()
        if arm not in DYNAMIC_NEURAL_ARMS:
            raise ValueError(f"unknown dynamic neural arm {arm!r}")
        self.arm = arm
        self.proj_bs_x = nn.Linear(BLENDSHAPE_DIM, 32, bias=False)
        self.proj_bs_dx = nn.Linear(BLENDSHAPE_DIM, 32, bias=False)
        self.proj_lm_x = nn.Linear(LANDMARK_DIM, 32, bias=False)
        self.proj_lm_dx = nn.Linear(LANDMARK_DIM, 32, bias=False)
        self.temporal = nn.GRU(
            input_size=FRAME_LATENT_DIM,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention_score = nn.Linear(64, 1)
        self.pool_projection = nn.Linear(128, 32)
        self.binary_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def _effective_blocks(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bs = features[..., :BLENDSHAPE_DIM]
        lm = features[..., BLENDSHAPE_DIM:]
        if self.arm == ARM_LANDMARK:
            bs = torch.zeros_like(bs)
        if self.arm == ARM_BLENDSHAPE:
            lm = torch.zeros_like(lm)
        return bs, lm

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
        timestamps: torch.Tensor,
        source_frame_indices: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1:] != (4, 32, 95):
            raise ValueError("features must have shape (batch, 4, 32, 95)")
        _validate_temporal_inputs(features, valid_mask, timestamps, source_frame_indices)
        if bool((~valid_mask.reshape(valid_mask.shape[0], -1).any(dim=1)).any()):
            raise ValueError("every recording must contain at least one valid frame")

        bs, lm = self._effective_blocks(features)
        bs = bs * valid_mask.unsqueeze(-1)
        lm = lm * valid_mask.unsqueeze(-1)
        bs_dx, _ = gap_safe_per_second_differences(
            bs, valid_mask, timestamps, source_frame_indices
        )
        lm_dx, _ = gap_safe_per_second_differences(
            lm, valid_mask, timestamps, source_frame_indices
        )
        bs_latent = self.proj_bs_x(bs) + self.proj_bs_dx(bs_dx)
        lm_latent = self.proj_lm_x(lm) + self.proj_lm_dx(lm_dx)
        frame_latent = torch.cat((bs_latent, lm_latent), dim=-1)
        if frame_latent.shape[-1] != FRAME_LATENT_DIM:
            raise AssertionError("frame latent width drifted from 64")

        batch_size, window_count, frame_count, _ = frame_latent.shape
        encoded, _ = self.temporal(
            frame_latent.reshape(batch_size * window_count, frame_count, FRAME_LATENT_DIM)
        )
        flat_mask = valid_mask.reshape(batch_size * window_count, frame_count)
        present = flat_mask.any(dim=1)

        masked_encoded = encoded.masked_fill(
            ~flat_mask.unsqueeze(-1), torch.finfo(encoded.dtype).min
        )
        max_pool = masked_encoded.max(dim=1).values
        max_pool = torch.where(present.unsqueeze(-1), max_pool, torch.zeros_like(max_pool))

        scores = self.attention_score(encoded).squeeze(-1)
        scores = scores.masked_fill(~flat_mask, -1e9)
        weights = torch.softmax(scores, dim=1) * flat_mask.to(scores.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).eps
        )
        attention_pool = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        attention_pool = torch.where(
            present.unsqueeze(-1), attention_pool, torch.zeros_like(attention_pool)
        )

        pooled = torch.tanh(self.pool_projection(torch.cat((max_pool, attention_pool), dim=-1)))
        pooled = pooled.reshape(batch_size, window_count, 32)
        present_windows = present.reshape(batch_size, window_count)
        recording = (pooled * present_windows.unsqueeze(-1)).sum(dim=1)
        recording = recording / present_windows.sum(dim=1, keepdim=True).to(
            pooled.dtype
        )
        logits = self.binary_head(recording).squeeze(-1)
        if logits.shape != (batch_size,) or not torch.isfinite(logits).all():
            raise RuntimeError("dynamic landmark model produced invalid logits")
        return logits


__all__ = [
    "ARM_BLENDSHAPE", "ARM_LANDMARK", "ARM_FUSION", "DYNAMIC_NEURAL_ARMS",
    "DynamicLandmarkModel", "gap_safe_per_second_differences",
    "horizontal_mirror_features",
]
