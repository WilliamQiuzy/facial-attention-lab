"""Closed training mechanisms for the twenty-model shared V9 screen."""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from src.models.broad_literature_candidate_registry_v9 import (
    BroadLiteratureCandidateV9,
    candidate_registry_v9,
)
from src.models.broad_literature_shared_router_v9 import BroadLiteratureSharedRouterV9
from src.preprocessing.generalization_110d import (
    LANDMARK_MI_110D,
    candidate_feature_names,
)


_SELF_SUPERVISION = {
    "cross_view_vicreg",
    "cross_view_barlow_twins",
    "masked_clinical_reconstruction",
    "masked_action_reconstruction",
    "clinical_to_dense_reconstruction",
}
_CLINICAL_NAMES = candidate_feature_names(LANDMARK_MI_110D)


def _setting(candidate: BroadLiteratureCandidateV9, name: str) -> float | int | str:
    values = dict(candidate.settings)
    if name not in values:
        raise ValueError(f"candidate does not define {name}")
    return values[name]


class SharpnessAwareControllerV9:
    """Two-step SAM/ASAM perturbation around one ordinary optimizer step."""

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        *,
        rho: float,
        adaptive: bool,
        eta: float = 0.01,
    ):
        self.parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
        if (
            not self.parameters
            or isinstance(rho, bool) or not math.isfinite(float(rho)) or rho <= 0.0
            or type(adaptive) is not bool
            or isinstance(eta, bool) or not math.isfinite(float(eta)) or eta < 0.0
        ):
            raise ValueError("SAM requires trainable parameters and positive finite settings")
        self.rho = float(rho)
        self.adaptive = adaptive
        self.eta = float(eta)
        self.perturbations: tuple[tuple[nn.Parameter, torch.Tensor], ...] = ()

    @torch.no_grad()
    def first_step(self) -> None:
        if self.perturbations:
            raise RuntimeError("a sharpness perturbation is already active")
        terms = []
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            scale = parameter.abs() + self.eta if self.adaptive else torch.ones_like(parameter)
            terms.append(torch.linalg.vector_norm(scale * parameter.grad, ord=2))
        if not terms:
            raise RuntimeError("SAM first step requires finite gradients")
        norm = torch.linalg.vector_norm(torch.stack(terms), ord=2)
        if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
            raise RuntimeError("SAM gradient norm is not positive and finite")
        step_scale = self.rho / (norm + 1e-12)
        perturbations = []
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            if self.adaptive:
                perturbation = (parameter.abs() + self.eta).square() * parameter.grad
            else:
                perturbation = parameter.grad
            perturbation = perturbation * step_scale
            parameter.add_(perturbation)
            perturbations.append((parameter, perturbation.detach().clone()))
        self.perturbations = tuple(perturbations)

    @torch.no_grad()
    def second_step(self, optimizer: torch.optim.Optimizer) -> None:
        if not self.perturbations or not isinstance(optimizer, torch.optim.Optimizer):
            raise RuntimeError("SAM second step requires an active perturbation and optimizer")
        for parameter, perturbation in self.perturbations:
            parameter.sub_(perturbation)
        self.perturbations = ()
        optimizer.step()


class SWAAccumulatorV9:
    """Equal arithmetic mean of explicitly selected model iterates."""

    def __init__(self, model: nn.Module):
        if not isinstance(model, nn.Module):
            raise ValueError("SWA requires a torch model")
        self.values = OrderedDict(
            (name, torch.zeros_like(parameter.detach()))
            for name, parameter in model.named_parameters()
        )
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        observed = OrderedDict(model.named_parameters())
        if tuple(observed) != tuple(self.values):
            raise ValueError("SWA model parameter schema drifted")
        self.updates += 1
        for name, parameter in observed.items():
            self.values[name].add_((parameter.detach() - self.values[name]) / self.updates)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        if self.updates < 1:
            raise RuntimeError("SWA has no averaged iterate")
        observed = OrderedDict(model.named_parameters())
        if tuple(observed) != tuple(self.values):
            raise ValueError("SWA model parameter schema drifted")
        for name, parameter in observed.items():
            parameter.copy_(self.values[name])


def _generator(seed: int) -> torch.Generator:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("deterministic masks require a nonnegative integer seed")
    return torch.Generator(device="cpu").manual_seed(seed)


def modality_dropout_mask(
    dense_available: torch.Tensor,
    *,
    probability: float,
    seed: int,
) -> torch.Tensor:
    if (
        not isinstance(dense_available, torch.Tensor)
        or dense_available.ndim != 2 or dense_available.dtype != torch.bool
        or isinstance(probability, bool) or not 0.0 <= float(probability) <= 1.0
    ):
        raise ValueError("modality dropout requires a boolean availability matrix")
    drop = torch.rand(
        dense_available.shape[0], generator=_generator(seed), device="cpu"
    ) < float(probability)
    return dense_available & ~drop.to(dense_available.device)[:, None]


def action_dropout_mask(
    action_mask: torch.Tensor,
    *,
    probability: float,
    seed: int,
) -> torch.Tensor:
    if (
        not isinstance(action_mask, torch.Tensor)
        or action_mask.ndim != 2 or action_mask.dtype != torch.bool
        or bool((action_mask.sum(dim=1) < 1).any())
        or isinstance(probability, bool) or not 0.0 <= float(probability) <= 1.0
    ):
        raise ValueError("action dropout requires at least one valid action per row")
    result = action_mask.detach().cpu().clone()
    generator = _generator(seed)
    decisions = torch.rand(result.shape[0], generator=generator) < float(probability)
    for row in range(result.shape[0]):
        valid = torch.nonzero(result[row], as_tuple=False).flatten()
        if not bool(decisions[row]) or len(valid) <= 1:
            continue
        selected = valid[torch.randint(len(valid), (1,), generator=generator).item()]
        result[row, selected] = False
    return result.to(action_mask.device)


def symmetric_binary_kl(first_logits: torch.Tensor, second_logits: torch.Tensor) -> torch.Tensor:
    if (
        first_logits.shape != second_logits.shape
        or first_logits.numel() < 1
        or not first_logits.is_floating_point()
        or not second_logits.is_floating_point()
        or not bool(torch.isfinite(first_logits).all())
        or not bool(torch.isfinite(second_logits).all())
    ):
        raise ValueError("R-Drop requires two aligned finite logit tensors")
    first = torch.sigmoid(first_logits).clamp(1e-6, 1.0 - 1e-6)
    second = torch.sigmoid(second_logits).clamp(1e-6, 1.0 - 1e-6)
    kl_first_second = first * torch.log(first / second) + (
        1.0 - first
    ) * torch.log((1.0 - first) / (1.0 - second))
    kl_second_first = second * torch.log(second / first) + (
        1.0 - second
    ) * torch.log((1.0 - second) / (1.0 - first))
    return 0.5 * (kl_first_second + kl_second_first).mean()


def _off_diagonal(values: torch.Tensor) -> torch.Tensor:
    rows, columns = values.shape
    if rows != columns or rows < 2:
        raise ValueError("redundancy matrices must be square with dimension at least two")
    return values.flatten()[:-1].view(rows - 1, rows + 1)[:, 1:].flatten()


def vicreg_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    invariance_weight: float,
    variance_weight: float,
    covariance_weight: float,
) -> torch.Tensor:
    if (
        first.shape != second.shape or first.ndim != 2 or first.shape[0] < 2
        or first.shape[1] < 2 or not bool(torch.isfinite(first).all())
        or not bool(torch.isfinite(second).all())
    ):
        raise ValueError("VICReg requires two finite batch-by-feature views")
    invariance = F.mse_loss(first, second)
    first_centered = first - first.mean(dim=0)
    second_centered = second - second.mean(dim=0)
    std_first = torch.sqrt(first_centered.var(dim=0, unbiased=True) + 1e-4)
    std_second = torch.sqrt(second_centered.var(dim=0, unbiased=True) + 1e-4)
    variance = 0.5 * (
        F.relu(1.0 - std_first).mean() + F.relu(1.0 - std_second).mean()
    )
    denominator = first.shape[0] - 1
    covariance_first = first_centered.T @ first_centered / denominator
    covariance_second = second_centered.T @ second_centered / denominator
    covariance = (
        _off_diagonal(covariance_first).square().sum()
        + _off_diagonal(covariance_second).square().sum()
    ) / first.shape[1]
    return (
        float(invariance_weight) * invariance
        + float(variance_weight) * variance
        + float(covariance_weight) * covariance
    )


def barlow_twins_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    off_diagonal_weight: float,
) -> torch.Tensor:
    if (
        first.shape != second.shape or first.ndim != 2 or first.shape[0] < 2
        or first.shape[1] < 2 or not bool(torch.isfinite(first).all())
        or not bool(torch.isfinite(second).all())
    ):
        raise ValueError("Barlow Twins requires two finite batch-by-feature views")
    first = (first - first.mean(dim=0)) / (first.std(dim=0, unbiased=True) + 1e-6)
    second = (second - second.mean(dim=0)) / (second.std(dim=0, unbiased=True) + 1e-6)
    cross = first.T @ second / first.shape[0]
    diagonal = torch.diagonal(cross).sub(1.0).square().sum()
    return diagonal + float(off_diagonal_weight) * _off_diagonal(cross).square().sum()


def focal_binary_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    if logits.shape != labels.shape or not bool(torch.isfinite(logits).all()):
        raise ValueError("focal loss requires aligned finite logits and labels")
    base = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    probability = torch.sigmoid(logits)
    target_probability = labels * probability + (1.0 - labels) * (1.0 - probability)
    return (1.0 - target_probability).pow(float(gamma)) * base


def ldam_binary_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    source_codes: torch.Tensor,
    class_counts: torch.Tensor,
    maximum_margin: float,
    logit_scale: float,
) -> torch.Tensor:
    if (
        logits.shape != labels.shape or source_codes.shape != labels.shape
        or source_codes.dtype != torch.long or class_counts.ndim != 2
        or class_counts.shape[1] != 2 or class_counts.dtype != torch.long
        or bool((source_codes < 0).any())
        or bool((source_codes >= class_counts.shape[0]).any())
    ):
        raise ValueError("LDAM requires positive fold-local source-class counts")
    used_counts = class_counts.index_select(0, torch.unique(source_codes, sorted=True))
    if bool((used_counts <= 0).any()):
        raise ValueError("every observed LDAM source requires both classes")
    safe_counts = class_counts.clamp_min(1)
    margins = safe_counts.to(logits.dtype).pow(-0.25)
    margins = margins / used_counts.to(logits.dtype).pow(-0.25).max() * float(maximum_margin)
    two_class = torch.stack((torch.zeros_like(logits), logits), dim=1)
    target = labels.to(torch.long)
    observed_margin = margins[source_codes, target]
    adjusted = two_class.clone()
    adjusted[torch.arange(len(labels), device=labels.device), target] -= observed_margin
    return F.cross_entropy(adjusted * float(logit_scale), target, reduction="none")


def source_pairwise_auc_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    source_codes: torch.Tensor,
    negative_tail_fraction: float,
) -> torch.Tensor:
    if (
        logits.shape != labels.shape or source_codes.shape != labels.shape
        or source_codes.dtype != torch.long
        or isinstance(negative_tail_fraction, bool)
        or not 0.0 < float(negative_tail_fraction) <= 1.0
    ):
        raise ValueError("pairwise AUC requires aligned source-coded binary evidence")
    losses = []
    for source in torch.unique(source_codes, sorted=True):
        local = source_codes == source
        positive = logits[local & (labels == 1)]
        negative = logits[local & (labels == 0)]
        if positive.numel() < 1 or negative.numel() < 1:
            raise ValueError("each source requires positive and negative training participants")
        count = max(1, math.ceil(negative.numel() * float(negative_tail_fraction)))
        hard_negative = torch.topk(negative, k=count, largest=True, sorted=False).values
        losses.append(F.softplus(-(positive[:, None] - hard_negative[None, :])).mean())
    return torch.stack(losses).mean()


def classification_objective(
    candidate: BroadLiteratureCandidateV9,
    routed_logits: torch.Tensor,
    universal_logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    source_codes: torch.Tensor,
    class_counts: torch.Tensor,
) -> torch.Tensor:
    if (
        type(candidate) is not BroadLiteratureCandidateV9
        or candidate not in candidate_registry_v9()
        or routed_logits.shape != labels.shape
        or universal_logits.shape != labels.shape
        or weights.shape != labels.shape
        or source_codes.shape != labels.shape
        or not torch.isclose(weights.sum(), torch.ones((), device=weights.device), atol=1e-5)
    ):
        raise ValueError("classification objective received an invalid frozen batch")
    mechanism = candidate.mechanism
    if mechanism == "focal_loss":
        routed_losses = focal_binary_loss(
            routed_logits, labels, gamma=float(_setting(candidate, "gamma"))
        )
    elif mechanism == "ldam_loss":
        routed_losses = ldam_binary_loss(
            routed_logits, labels, source_codes, class_counts,
            float(_setting(candidate, "maximum_margin")),
            float(_setting(candidate, "logit_scale")),
        )
    else:
        routed_losses = F.binary_cross_entropy_with_logits(
            routed_logits, labels, reduction="none"
        )
    universal_losses = F.binary_cross_entropy_with_logits(
        universal_logits, labels, reduction="none"
    )
    objective = torch.sum((routed_losses + 0.5 * universal_losses) * weights)
    if mechanism == "pairwise_auc_loss":
        objective = objective + float(_setting(candidate, "ranking_weight")) * (
            source_pairwise_auc_loss(routed_logits, labels, source_codes, 1.0)
        )
    elif mechanism == "high_specificity_partial_auc_loss":
        objective = objective + float(_setting(candidate, "ranking_weight")) * (
            source_pairwise_auc_loss(
                routed_logits, labels, source_codes,
                float(_setting(candidate, "negative_tail_fraction")),
            )
        )
    elif mechanism == "brier_composite_loss":
        brier = torch.sum((torch.sigmoid(routed_logits) - labels).square() * weights)
        objective = objective + float(_setting(candidate, "brier_weight")) * brier
    return objective


def _clinical_feature_groups() -> tuple[tuple[int, ...], ...]:
    groups: OrderedDict[str, list[int]] = OrderedDict()
    for index, name in enumerate(_CLINICAL_NAMES):
        key = name.split("__", 1)[0]
        groups.setdefault(key, []).append(index)
    result = tuple(tuple(values) for values in groups.values())
    if set(index for group in result for index in group) != set(range(110)):
        raise RuntimeError("the clinical reconstruction groups do not cover 110D")
    return result


_CLINICAL_GROUPS = _clinical_feature_groups()


class RepresentationAuxiliariesV9(nn.Module):
    """Fold-local auxiliary heads for the five self-supervised candidates."""

    def __init__(self, candidate: BroadLiteratureCandidateV9):
        super().__init__()
        if (
            type(candidate) is not BroadLiteratureCandidateV9
            or candidate not in candidate_registry_v9()
            or candidate.mechanism not in _SELF_SUPERVISION
        ):
            raise ValueError("representation auxiliaries require one self-supervised candidate")
        self.candidate = candidate
        if candidate.mechanism in {"cross_view_vicreg", "cross_view_barlow_twins"}:
            self.clinical_projector = nn.Sequential(
                nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 32)
            )
            self.dense_projector = nn.Sequential(
                nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 32)
            )
        elif candidate.mechanism == "masked_clinical_reconstruction":
            self.decoder = nn.Linear(64, 110)
        elif candidate.mechanism == "masked_action_reconstruction":
            self.decoder = nn.Linear(64, 64)
        elif candidate.mechanism == "clinical_to_dense_reconstruction":
            self.decoder = nn.Linear(64, 64)

    @staticmethod
    def _view_tokens(
        model: BroadLiteratureSharedRouterV9,
        inputs: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        backbone = model.base.base.backbone
        clinical = backbone._clinical_tokens(inputs[0], inputs[1])
        dense_first = backbone._dense_tokens(inputs[2])
        dense_second = backbone._dense_tokens(inputs[3])
        dense = backbone.dense_pair_projection(torch.cat((
            0.5 * (dense_first + dense_second), torch.abs(dense_first - dense_second)
        ), dim=-1))
        return clinical, dense

    @staticmethod
    def _masked_action_inputs(
        inputs: tuple[torch.Tensor, ...],
        dropped: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        changed = [value.clone() for value in inputs]
        changed[0][dropped] = 0.0
        changed[1][dropped] = 0.0
        changed[2][dropped] = 0.0
        changed[3][dropped] = 0.0
        changed[4][dropped] = False
        changed[5][dropped] = False
        changed[6][dropped] = 0.0
        return tuple(changed)

    def loss(
        self,
        model: BroadLiteratureSharedRouterV9,
        inputs: tuple[torch.Tensor, ...],
        task_codes: torch.Tensor,
        *,
        seed: int,
    ) -> torch.Tensor:
        if (
            type(model) is not BroadLiteratureSharedRouterV9
            or len(inputs) != 9
            or task_codes.shape != (inputs[0].shape[0],)
        ):
            raise ValueError("representation loss requires the exact shared model batch")
        mechanism = self.candidate.mechanism
        action_mask = inputs[-2]
        if mechanism in {"cross_view_vicreg", "cross_view_barlow_twins"}:
            clinical, dense = self._view_tokens(model, inputs)
            supported = action_mask & inputs[5]
            if int(supported.sum()) < 2:
                raise RuntimeError("cross-view supervision requires two supported action views")
            first = self.clinical_projector(clinical[supported])
            second = self.dense_projector(dense[supported])
            if mechanism == "cross_view_vicreg":
                return vicreg_loss(
                    first, second,
                    float(_setting(self.candidate, "invariance_weight")),
                    float(_setting(self.candidate, "variance_weight")),
                    float(_setting(self.candidate, "covariance_weight")),
                )
            return barlow_twins_loss(
                first, second, float(_setting(self.candidate, "off_diagonal_weight"))
            )
        if mechanism == "masked_clinical_reconstruction":
            group_count = max(1, math.ceil(
                len(_CLINICAL_GROUPS) * float(_setting(self.candidate, "mask_fraction"))
            ))
            selected_groups = torch.randperm(
                len(_CLINICAL_GROUPS), generator=_generator(seed)
            )[:group_count].tolist()
            selected_indices = tuple(
                index for group in selected_groups for index in _CLINICAL_GROUPS[group]
            )
            index = torch.tensor(selected_indices, dtype=torch.long, device=inputs[0].device)
            changed = list(inputs)
            changed[0] = inputs[0].clone()
            changed[1] = inputs[1].clone()
            changed[0].index_fill_(-1, index, 0.0)
            changed[1].index_fill_(-1, index, 0.0)
            tokens = model.shared_action_tokens(*tuple(changed))
            predicted = self.decoder(tokens).index_select(-1, index)
            target = (0.5 * (inputs[0] + inputs[1])).index_select(-1, index)
            selected = action_mask.unsqueeze(-1).expand_as(predicted)
            return 0.25 * F.mse_loss(predicted[selected], target[selected])
        if mechanism == "masked_action_reconstruction":
            kept = action_dropout_mask(action_mask, probability=1.0, seed=seed)
            dropped = action_mask & ~kept
            if not bool(dropped.any()):
                raise RuntimeError("masked-action supervision requires a multislot examination")
            with torch.no_grad():
                target = model.shared_action_tokens(*inputs).detach()
            masked = self._masked_action_inputs(inputs, dropped)
            predicted = self.decoder(model.shared_action_tokens(*masked))
            return 0.25 * F.mse_loss(predicted[dropped], target[dropped])
        if mechanism == "clinical_to_dense_reconstruction":
            clinical, dense = self._view_tokens(model, inputs)
            supported = action_mask & inputs[5]
            if not bool(supported.any()):
                raise RuntimeError("cross-view reconstruction requires dense evidence")
            return float(_setting(self.candidate, "reconstruction_weight")) * (
                F.mse_loss(self.decoder(clinical[supported]), dense[supported].detach())
            )
        raise RuntimeError("an unimplemented self-supervised candidate escaped validation")


__all__ = [
    "RepresentationAuxiliariesV9",
    "SWAAccumulatorV9",
    "SharpnessAwareControllerV9",
    "action_dropout_mask",
    "barlow_twins_loss",
    "classification_objective",
    "focal_binary_loss",
    "ldam_binary_loss",
    "modality_dropout_mask",
    "source_pairwise_auc_loss",
    "symmetric_binary_kl",
    "vicreg_loss",
]
