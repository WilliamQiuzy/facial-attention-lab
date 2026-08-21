"""Participant-disjoint and LOSO evaluation for the shared MMoE candidate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from src.evaluation.distilled_shared_search_v9 import configure_deterministic_training_v9
from src.evaluation.medically_gated_shared_search_v2 import (
    MedicalSharedDatasetV2,
    _model_inputs,
    _scaled,
    _tensor,
)
from src.evaluation.residual_shared_search_v8 import evaluate_residual_candidate
from src.evaluation.shared_clinical_encoder_v1 import (
    SOURCES,
    fit_clinical_scaler,
    participant_disjoint_folds,
    source_class_balanced_weights,
)
from src.evaluation.universal_orofacial_v1 import binary_metrics
from src.models.multigate_shared_expert_router_v9 import (
    MultiGateSharedExpertCandidateV9,
    MultiGateSharedExpertRouterV9,
    candidate_registry_v9,
)
from src.models.residual_shared_router_v8 import candidate_registry_v8


_SOURCE_TASK_CODE = {source: index for index, source in enumerate(SOURCES)}


def _immutable(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


@dataclass(frozen=True)
class MultiGateSharedExpertEvaluationV9:
    probabilities: np.ndarray
    metrics: dict[str, dict[str, float]]
    loso_metrics: dict[str, dict[str, float]]
    loso_train_sources: tuple[tuple[str, tuple[str, ...]], ...]
    model_fits: int
    shared_gradient_sources: tuple[str, ...]
    shared_expert_gradient_sources: tuple[str, ...]
    task_specific_parameter_fraction: float


def _fit_model(
    dataset: MedicalSharedDatasetV2,
    candidate: MultiGateSharedExpertCandidateV9,
    train: np.ndarray,
    *,
    epochs: int,
    local_seed: int,
    runtime: torch.device,
    audit_sources: bool,
):
    base = dataset.base
    torch.manual_seed(local_seed)
    if runtime.type == "cuda":
        torch.cuda.manual_seed_all(local_seed)
    scaler = fit_clinical_scaler(base, train)
    original = _scaled(base.clinical_original, scaler.mean, scaler.scale)
    mirrored = _scaled(base.clinical_mirrored, scaler.mean, scaler.scale)
    model = MultiGateSharedExpertRouterV9(candidate).to(runtime)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
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

    covered: set[str] = set()
    expert_covered: set[str] = set()
    if audit_sources:
        for source in SOURCES:
            local = torch.tensor([
                index for index, observed in enumerate(train_sources)
                if observed == source
            ], dtype=torch.long, device=runtime)
            model.zero_grad(set_to_none=True)
            local_inputs = tuple(value.index_select(0, local) for value in inputs)
            local_tasks = tasks.index_select(0, local)
            local_labels = labels.index_select(0, local)
            tokens = model.shared_action_tokens(*local_inputs)
            routed, _ = model.routed_and_universal_logits(
                tokens, local_inputs[-2], local_tasks
            )
            F.binary_cross_entropy_with_logits(routed, local_labels).backward()
            clinical_gradient = model.base.base.backbone.clinical_encoder[0].weight.grad
            if clinical_gradient is None or float(clinical_gradient.norm()) <= 0.0:
                raise RuntimeError("a source failed the shared-trunk gradient audit")
            covered.add(source)
            if candidate.shared_expert_count > 0:
                if not any(
                    parameter.grad is not None and float(parameter.grad.norm()) > 0.0
                    for parameter in model.shared_experts.parameters()
                ):
                    raise RuntimeError("a source failed the shared-expert gradient audit")
                expert_covered.add(source)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        tokens = model.shared_action_tokens(*inputs)
        routed, universal = model.routed_and_universal_logits(
            tokens, inputs[-2], tasks
        )
        losses = F.binary_cross_entropy_with_logits(
            routed, labels, reduction="none"
        ) + 0.5 * F.binary_cross_entropy_with_logits(
            universal, labels, reduction="none"
        )
        torch.sum(losses * weights).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model, original, mirrored, covered, expert_covered


def _predict_routed(
    model: MultiGateSharedExpertRouterV9,
    dataset: MedicalSharedDatasetV2,
    original: np.ndarray,
    mirrored: np.ndarray,
    held: np.ndarray,
    runtime: torch.device,
) -> np.ndarray:
    held_inputs = _model_inputs(dataset, original, mirrored, held, runtime)
    tasks = _tensor(np.asarray([
        _SOURCE_TASK_CODE[dataset.base.sources[index]] for index in held
    ], dtype=np.int64), runtime)
    model.eval()
    with torch.no_grad():
        tokens = model.shared_action_tokens(*held_inputs)
        logits = model.routed_logits(tokens, held_inputs[-2], tasks)
    return torch.sigmoid(logits).cpu().numpy()


def _loso_metrics(
    dataset: MedicalSharedDatasetV2,
    candidate: MultiGateSharedExpertCandidateV9,
    *,
    epochs: int,
    seed: int,
    runtime: torch.device,
):
    base = dataset.base
    results = {}
    training_sources = []
    for target_index, target_source in enumerate(SOURCES):
        train = np.asarray([
            index for index, source in enumerate(base.sources)
            if source != target_source
        ], dtype=np.int64)
        held = np.asarray([
            index for index, source in enumerate(base.sources)
            if source == target_source
        ], dtype=np.int64)
        model, original, mirrored, _, _ = _fit_model(
            dataset, candidate, train, epochs=epochs,
            local_seed=seed * 1009 + 100 + target_index,
            runtime=runtime, audit_sources=False,
        )
        held_inputs = _model_inputs(dataset, original, mirrored, held, runtime)
        model.eval()
        with torch.no_grad():
            tokens = model.shared_action_tokens(*held_inputs)
            probabilities = torch.sigmoid(
                model.universal_logits(tokens, held_inputs[-2])
            ).cpu().numpy()
        results[target_source] = binary_metrics(base.labels[held], probabilities)
        training_sources.append((
            target_source,
            tuple(source for source in SOURCES if source != target_source),
        ))
    return results, tuple(training_sources)


def evaluate_multigate_shared_expert_candidate(
    dataset: MedicalSharedDatasetV2,
    candidate: MultiGateSharedExpertCandidateV9,
    *,
    epochs: int,
    n_splits: int = 6,
    seed: int = 0,
    device: str = "cpu",
) -> MultiGateSharedExpertEvaluationV9:
    if (
        type(dataset) is not MedicalSharedDatasetV2
        or type(candidate) is not MultiGateSharedExpertCandidateV9
        or candidate not in candidate_registry_v9()
        or isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1
        or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        or type(device) is not str
    ):
        raise ValueError("invalid shared-expert V9 evaluation")
    runtime = torch.device(device)
    if runtime.type not in {"cpu", "cuda"} or (
        runtime.type == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError("shared-expert V9 runtime is unavailable")
    configure_deterministic_training_v9(runtime)
    base = dataset.base
    folds = participant_disjoint_folds(base, n_splits=n_splits)
    probabilities = np.full(len(base.labels), np.nan, dtype=np.float64)
    covered: set[str] = set()
    expert_covered: set[str] = set()
    if candidate.candidate_id == "MSE9-000":
        residual = evaluate_residual_candidate(
            dataset,
            next(row for row in candidate_registry_v8() if row.candidate_id == "RSR8-001"),
            epochs=epochs, n_splits=n_splits, seed=seed, device=device,
        )
        probabilities[:] = residual.probabilities
        metrics = residual.metrics
        covered.update(residual.shared_gradient_sources)
        fraction = MultiGateSharedExpertRouterV9(
            candidate
        ).task_specific_parameter_fraction()
    else:
        fraction = np.nan
        for fold_index, (train, held) in enumerate(folds):
            model, original, mirrored, audited, expert_audited = _fit_model(
                dataset, candidate, train, epochs=epochs,
                local_seed=seed * 1009 + fold_index,
                runtime=runtime, audit_sources=fold_index == 0,
            )
            probabilities[held] = _predict_routed(
                model, dataset, original, mirrored, held, runtime
            )
            covered.update(audited)
            expert_covered.update(expert_audited)
            fraction = model.task_specific_parameter_fraction()
        metrics = {
            source: binary_metrics(
                base.labels[np.asarray([value == source for value in base.sources])],
                probabilities[np.asarray([value == source for value in base.sources])],
            )
            for source in SOURCES
        }
    loso, loso_sources = _loso_metrics(
        dataset, candidate, epochs=epochs, seed=seed, runtime=runtime
    )
    expected_expert = set(SOURCES) if candidate.shared_expert_count > 0 else set()
    if (
        not np.isfinite(probabilities).all()
        or covered != set(SOURCES)
        or expert_covered != expected_expert
        or not 0.0 <= fraction < 0.10
    ):
        raise RuntimeError("shared-expert V9 evaluation is incomplete")
    return MultiGateSharedExpertEvaluationV9(
        probabilities=_immutable(probabilities),
        metrics=metrics,
        loso_metrics=loso,
        loso_train_sources=loso_sources,
        model_fits=len(folds) + len(SOURCES),
        shared_gradient_sources=SOURCES,
        shared_expert_gradient_sources=tuple(
            source for source in SOURCES if source in expert_covered
        ),
        task_specific_parameter_fraction=float(fraction),
    )


__all__ = [
    "MultiGateSharedExpertEvaluationV9",
    "evaluate_multigate_shared_expert_candidate",
]
