"""Fold-train-only diagnostics that authorize literature-grounded V9 methods."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch
from torch.nn import functional as F

from src.evaluation.distilled_shared_search_v9 import (
    configure_deterministic_training_v9,
)
from src.evaluation.medically_gated_shared_search_v2 import (
    MedicalSharedDatasetV2,
    _model_inputs,
    _scaled,
    _tensor,
)
from src.evaluation.shared_clinical_encoder_v1 import (
    SOURCES,
    fit_clinical_scaler,
    participant_disjoint_folds,
    source_class_balanced_weights,
)
from src.models.residual_shared_router_v8 import (
    ResidualSharedRouterV8,
    candidate_registry_v8,
)


_SOURCE_TASK_CODE = {source: index for index, source in enumerate(SOURCES)}
_GRADNORM_AUTHORIZATION_RATIO = 2.0


@dataclass(frozen=True)
class SourceOptimizationSummaryV9:
    source: str
    initial_loss: float
    final_loss: float
    initial_gradient_norm: float
    relative_remaining_loss: float


@dataclass(frozen=True)
class PairwiseGradientSummaryV9:
    pair: str
    median_cosine: float


@dataclass(frozen=True)
class SharedOptimizationDiagnosticV9:
    fold_count: int
    source_summaries: tuple[SourceOptimizationSummaryV9, ...]
    pairwise_cosines: tuple[PairwiseGradientSummaryV9, ...]
    gradient_norm_ratio: float
    relative_training_rate_ratio: float
    cagrad_authorized: bool
    gradnorm_authorized: bool


def _locked_v8_candidate():
    matched = tuple(
        row for row in candidate_registry_v8() if row.candidate_id == "RSR8-001"
    )
    if len(matched) != 1:
        raise RuntimeError("the exact V8 comparator is unavailable")
    return matched[0]


def _shared_parameters(
    model: ResidualSharedRouterV8,
) -> tuple[torch.nn.Parameter, ...]:
    selected = tuple(
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("adapters.")
        and not name.startswith("base.task_queries")
        and not name.startswith("base.backbone.task_heads")
    )
    if not selected:
        raise RuntimeError("the V8 shared-parameter closure is empty")
    return selected


def _source_objective(
    model: ResidualSharedRouterV8,
    inputs: tuple[torch.Tensor, ...],
    labels: torch.Tensor,
    task_codes: torch.Tensor,
    weights: torch.Tensor,
    local: torch.Tensor,
) -> torch.Tensor:
    local_inputs = tuple(value.index_select(0, local) for value in inputs)
    local_labels = labels.index_select(0, local)
    local_tasks = task_codes.index_select(0, local)
    local_weights = weights.index_select(0, local)
    local_weights = local_weights / local_weights.sum()
    tokens = model.shared_action_tokens(*local_inputs)
    common = model.base.endpoint_embedding(tokens, local_inputs[-2], local_tasks)
    endpoint = model.adapt_endpoint(common, local_tasks)
    universal_embedding = model.base.universal_embedding(tokens, local_inputs[-2])
    task_logits = model.base.task_logits_from_embedding(endpoint, local_tasks)
    universal_logits = model.base.universal_head(universal_embedding).squeeze(-1)
    routed = 0.75 * task_logits + 0.25 * universal_logits
    losses = F.binary_cross_entropy_with_logits(
        routed, local_labels, reduction="none"
    ) + 0.5 * F.binary_cross_entropy_with_logits(
        universal_logits, local_labels, reduction="none"
    )
    return torch.sum(losses * local_weights)


def _gradient_vector(
    loss: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
) -> torch.Tensor:
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    vector = torch.cat(tuple(
        (torch.zeros_like(parameter) if gradient is None else gradient)
        .detach().reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ))
    if not bool(torch.isfinite(vector).all()) or float(vector.norm()) <= 0.0:
        raise RuntimeError("shared optimization diagnostic produced invalid gradients")
    return vector


def _pairwise_cosines(
    vectors: dict[str, torch.Tensor],
) -> dict[str, float]:
    output = {}
    for first, second in combinations(SOURCES, 2):
        value = F.cosine_similarity(
            vectors[first][None, :], vectors[second][None, :]
        )
        output[f"{first}__{second}"] = float(value.item())
    return output


def diagnose_v8_shared_optimization(
    dataset: MedicalSharedDatasetV2,
    *,
    audit_epochs: int,
    n_splits: int = 6,
    seed: int = 0,
    device: str = "cpu",
) -> SharedOptimizationDiagnosticV9:
    """Measure optimization evidence without inspecting any held-fold outcome."""
    if (
        type(dataset) is not MedicalSharedDatasetV2
        or isinstance(audit_epochs, bool)
        or not isinstance(audit_epochs, int)
        or audit_epochs < 1
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or type(device) is not str
    ):
        raise ValueError("invalid V9 diagnostic configuration")
    runtime = torch.device(device)
    if runtime.type not in {"cpu", "cuda"} or (
        runtime.type == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError("requested V9 diagnostic runtime is unavailable")
    configure_deterministic_training_v9(runtime)
    base = dataset.base
    folds = participant_disjoint_folds(base, n_splits=n_splits)
    initial_losses = {source: [] for source in SOURCES}
    final_losses = {source: [] for source in SOURCES}
    gradient_norms = {source: [] for source in SOURCES}
    cosine_rows = {
        f"{first}__{second}": [] for first, second in combinations(SOURCES, 2)
    }

    for fold_index, (train, _held) in enumerate(folds):
        local_seed = seed * 1009 + fold_index
        torch.manual_seed(local_seed)
        if runtime.type == "cuda":
            torch.cuda.manual_seed_all(local_seed)
        scaler = fit_clinical_scaler(base, train)
        original = _scaled(base.clinical_original, scaler.mean, scaler.scale)
        mirrored = _scaled(base.clinical_mirrored, scaler.mean, scaler.scale)
        model = ResidualSharedRouterV8(_locked_v8_candidate()).to(runtime)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=3e-4, weight_decay=1e-3
        )
        train_sources = tuple(base.sources[index] for index in train)
        weights = _tensor(
            source_class_balanced_weights(
                base.labels[train], train_sources
            ).astype(np.float32),
            runtime,
        )
        labels = _tensor(base.labels[train].astype(np.float32), runtime)
        tasks = _tensor(np.asarray([
            _SOURCE_TASK_CODE[source] for source in train_sources
        ], dtype=np.int64), runtime)
        inputs = _model_inputs(dataset, original, mirrored, train, runtime)
        source_indices = {
            source: torch.tensor([
                index for index, observed in enumerate(train_sources)
                if observed == source
            ], dtype=torch.long, device=runtime)
            for source in SOURCES
        }
        if any(local.numel() == 0 for local in source_indices.values()):
            raise RuntimeError("every diagnostic fold requires all three sources")
        shared = _shared_parameters(model)
        vectors = {}
        model.train()
        for source in SOURCES:
            loss = _source_objective(
                model, inputs, labels, tasks, weights, source_indices[source]
            )
            vector = _gradient_vector(loss, shared)
            initial_losses[source].append(float(loss.detach().item()))
            gradient_norms[source].append(float(vector.norm().item()))
            vectors[source] = vector
        for pair, value in _pairwise_cosines(vectors).items():
            cosine_rows[pair].append(value)

        for _ in range(audit_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            objective = sum(
                _source_objective(
                    model, inputs, labels, tasks, weights, source_indices[source]
                )
                for source in SOURCES
            ) / len(SOURCES)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            for source in SOURCES:
                loss = _source_objective(
                    model, inputs, labels, tasks, weights, source_indices[source]
                )
                final_losses[source].append(float(loss.item()))
        del model, optimizer

    source_summaries = tuple(
        SourceOptimizationSummaryV9(
            source=source,
            initial_loss=float(np.mean(initial_losses[source])),
            final_loss=float(np.mean(final_losses[source])),
            initial_gradient_norm=float(np.mean(gradient_norms[source])),
            relative_remaining_loss=float(
                np.mean(final_losses[source]) / np.mean(initial_losses[source])
            ),
        )
        for source in SOURCES
    )
    pairwise = tuple(
        PairwiseGradientSummaryV9(
            pair=pair, median_cosine=float(np.median(cosine_rows[pair]))
        )
        for pair in cosine_rows
    )
    norms = [row.initial_gradient_norm for row in source_summaries]
    rates = [row.relative_remaining_loss for row in source_summaries]
    norm_ratio = max(norms) / min(norms)
    rate_ratio = max(rates) / min(rates)
    if not all(np.isfinite(value) and value > 0.0 for value in (*norms, *rates)):
        raise RuntimeError("V9 diagnostic aggregation is nonfinite")
    return SharedOptimizationDiagnosticV9(
        fold_count=len(folds),
        source_summaries=source_summaries,
        pairwise_cosines=pairwise,
        gradient_norm_ratio=float(norm_ratio),
        relative_training_rate_ratio=float(rate_ratio),
        cagrad_authorized=any(row.median_cosine < 0.0 for row in pairwise),
        gradnorm_authorized=(
            norm_ratio >= _GRADNORM_AUTHORIZATION_RATIO
            or rate_ratio >= _GRADNORM_AUTHORIZATION_RATIO
        ),
    )


def diagnostic_to_public_dict(
    diagnostic: SharedOptimizationDiagnosticV9,
) -> dict[str, object]:
    if type(diagnostic) is not SharedOptimizationDiagnosticV9:
        raise ValueError("validated V9 diagnostic required")
    return {
        "schema_version": "literature_grounded_shared_v9_diagnostic_v1",
        "fold_count": diagnostic.fold_count,
        "source_summaries": [
            {
                "source": row.source,
                "initial_loss": row.initial_loss,
                "final_loss": row.final_loss,
                "initial_gradient_norm": row.initial_gradient_norm,
                "relative_remaining_loss": row.relative_remaining_loss,
            }
            for row in diagnostic.source_summaries
        ],
        "pairwise_gradient_cosines": [
            {"pair": row.pair, "median_cosine": row.median_cosine}
            for row in diagnostic.pairwise_cosines
        ],
        "gradient_norm_ratio": diagnostic.gradient_norm_ratio,
        "relative_training_rate_ratio": diagnostic.relative_training_rate_ratio,
        "cagrad_authorized": diagnostic.cagrad_authorized,
        "gradnorm_authorized": diagnostic.gradnorm_authorized,
    }


__all__ = [
    "PairwiseGradientSummaryV9",
    "SharedOptimizationDiagnosticV9",
    "SourceOptimizationSummaryV9",
    "diagnose_v8_shared_optimization",
    "diagnostic_to_public_dict",
]
