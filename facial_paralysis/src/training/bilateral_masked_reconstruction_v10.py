"""Laterality-safe masked clinical reconstruction for shared V10 research."""
from __future__ import annotations

from collections import OrderedDict
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.models.bilateral_reconstruction_candidate_registry_v10 import (
    BilateralReconstructionCandidateV10,
    candidate_registry_v10,
)
from src.models.broad_literature_shared_router_v9 import BroadLiteratureSharedRouterV9
from src.preprocessing.generalization_110d import (
    LANDMARK_MI_110D,
    candidate_feature_names,
)


_FEATURE_NAMES = candidate_feature_names(LANDMARK_MI_110D)


def _clinical_feature_groups() -> tuple[tuple[int, ...], ...]:
    groups: OrderedDict[str, list[int]] = OrderedDict()
    for index, name in enumerate(_FEATURE_NAMES):
        groups.setdefault(name.split("__", 1)[0], []).append(index)
    result = tuple(tuple(values) for values in groups.values())
    if set(index for group in result for index in group) != set(range(110)):
        raise RuntimeError("V10 clinical groups must cover the exact 110D schema")
    return result


_CLINICAL_GROUPS = _clinical_feature_groups()


def _validate_views(
    original: torch.Tensor,
    mirrored: torch.Tensor,
) -> int:
    if (
        not isinstance(original, torch.Tensor)
        or not isinstance(mirrored, torch.Tensor)
        or original.shape != mirrored.shape
        or original.ndim < 2
        or original.shape[-1] < 1
        or not original.is_floating_point()
        or not mirrored.is_floating_point()
        or original.device != mirrored.device
        or not bool(torch.isfinite(original).all())
        or not bool(torch.isfinite(mirrored).all())
    ):
        raise ValueError("bilateral targets require aligned finite floating views")
    return original.shape[-1]


def bilateral_reconstruction_targets(
    original: torch.Tensor,
    mirrored: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    _validate_views(original, mirrored)
    if mode == "v9_average":
        return 0.5 * (original + mirrored)
    if mode == "bilateral_decomposition":
        return torch.cat((
            0.5 * (original + mirrored), torch.abs(original - mirrored)
        ), dim=-1)
    if mode == "unordered_twin":
        return torch.cat((original, mirrored), dim=-1)
    raise ValueError("unknown bilateral reconstruction mode")


def _validate_indices(indices: torch.Tensor, dimension: int, device: torch.device) -> None:
    if (
        not isinstance(indices, torch.Tensor)
        or indices.ndim != 1
        or indices.dtype != torch.long
        or indices.device != device
        or indices.numel() < 1
        or bool((indices < 0).any())
        or bool((indices >= dimension).any())
        or indices.unique().numel() != indices.numel()
    ):
        raise ValueError("masked reconstruction requires unique in-range indices")


def masked_bilateral_reconstruction_loss(
    prediction: torch.Tensor,
    original: torch.Tensor,
    mirrored: torch.Tensor,
    indices: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    dimension = _validate_views(original, mirrored)
    _validate_indices(indices, dimension, original.device)
    expected = dimension if mode == "v9_average" else dimension * 2
    if (
        not isinstance(prediction, torch.Tensor)
        or prediction.shape != original.shape[:-1] + (expected,)
        or not prediction.is_floating_point()
        or prediction.device != original.device
        or not bool(torch.isfinite(prediction).all())
    ):
        raise ValueError("prediction differs from the bilateral target contract")
    if mode == "v9_average":
        target = 0.5 * (original + mirrored)
        return F.mse_loss(
            prediction.index_select(-1, indices),
            target.index_select(-1, indices),
        )
    first = prediction[..., :dimension].index_select(-1, indices)
    second = prediction[..., dimension:].index_select(-1, indices)
    if mode == "bilateral_decomposition":
        mean = (0.5 * (original + mirrored)).index_select(-1, indices)
        difference = torch.abs(original - mirrored).index_select(-1, indices)
        return 0.5 * (F.mse_loss(first, mean) + F.mse_loss(second, difference))
    if mode == "unordered_twin":
        original = original.index_select(-1, indices)
        mirrored = mirrored.index_select(-1, indices)
        direct = 0.5 * (
            (first - original).square().mean(dim=-1)
            + (second - mirrored).square().mean(dim=-1)
        )
        swapped = 0.5 * (
            (first - mirrored).square().mean(dim=-1)
            + (second - original).square().mean(dim=-1)
        )
        return torch.minimum(direct, swapped).mean()
    raise ValueError("unknown bilateral reconstruction mode")


class BilateralMaskedReconstructionV10(nn.Module):
    """Training-only decoder; all inference remains the exact shared V8 trunk."""

    def __init__(self, candidate: BilateralReconstructionCandidateV10):
        super().__init__()
        if (
            type(candidate) is not BilateralReconstructionCandidateV10
            or candidate not in candidate_registry_v10()
        ):
            raise ValueError("V10 auxiliary requires one frozen candidate")
        self.candidate = candidate
        output_dimension = 110 if candidate.reconstruction_mode == "v9_average" else 220
        self.decoder = nn.Linear(64, output_dimension)

    @staticmethod
    def _selected_indices(seed: int, device: torch.device) -> torch.Tensor:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("masked reconstruction seed must be a nonnegative integer")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        group_count = max(1, math.ceil(len(_CLINICAL_GROUPS) * 0.25))
        selected_groups = torch.randperm(
            len(_CLINICAL_GROUPS), generator=generator
        )[:group_count].tolist()
        selected = tuple(
            index for group in selected_groups for index in _CLINICAL_GROUPS[group]
        )
        return torch.tensor(selected, dtype=torch.long, device=device)

    def loss(
        self,
        model: BroadLiteratureSharedRouterV9,
        inputs: tuple[torch.Tensor, ...],
        *,
        seed: int,
    ) -> torch.Tensor:
        if type(model) is not BroadLiteratureSharedRouterV9 or len(inputs) != 9:
            raise ValueError("V10 reconstruction requires the exact V9 shared model batch")
        original, mirrored = inputs[0], inputs[1]
        if original.shape[-1] != 110 or inputs[-2].shape != original.shape[:2]:
            raise ValueError("V10 reconstruction requires aligned 110D action inputs")
        index = self._selected_indices(seed, original.device)
        changed = list(inputs)
        changed[0] = original.clone()
        changed[1] = mirrored.clone()
        changed[0].index_fill_(-1, index, 0.0)
        changed[1].index_fill_(-1, index, 0.0)
        tokens = model.shared_action_tokens(*tuple(changed))
        selected_actions = inputs[-2]
        if not bool(selected_actions.any()):
            raise ValueError("V10 reconstruction requires a valid action")
        prediction = self.decoder(tokens)[selected_actions]
        value = masked_bilateral_reconstruction_loss(
            prediction,
            original[selected_actions],
            mirrored[selected_actions],
            index,
            self.candidate.reconstruction_mode,
        )
        return 0.25 * value


__all__ = [
    "BilateralMaskedReconstructionV10",
    "bilateral_reconstruction_targets",
    "masked_bilateral_reconstruction_loss",
]
