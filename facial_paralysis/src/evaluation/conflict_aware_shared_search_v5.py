"""Conflict-aware training for the locked shared normal-manifold representation."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch
from torch.nn import functional as F

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
from src.evaluation.shared_normal_manifold_search_v4 import (
    UNIVERSAL_AUXILIARY_WEIGHT,
)
from src.evaluation.universal_orofacial_v1 import binary_metrics
from src.models.conflict_aware_candidate_registry_v5 import (
    ConflictAwareCandidateV5,
    candidate_registry_v5,
)
from src.models.normal_manifold_candidate_registry_v4 import candidate_registry_v4
from src.models.shared_normal_manifold_router_v4 import SharedNormalManifoldRouterV4


_SOURCE_TASK_CODE = {source: index for index, source in enumerate(SOURCES)}


def _immutable(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def project_conflicting_gradient_vectors(
    vectors: tuple[torch.Tensor, ...],
    *,
    strength: float,
) -> tuple[torch.Tensor, ...]:
    if (
        type(vectors) is not tuple
        or len(vectors) < 2
        or any(
            not isinstance(vector, torch.Tensor)
            or vector.ndim != 1
            or vector.shape != vectors[0].shape
            or not vector.is_floating_point()
            or vector.device != vectors[0].device
            or not bool(torch.isfinite(vector).all())
            for vector in vectors
        )
        or strength not in (0.5, 1.0)
    ):
        raise ValueError("gradient projection requires frozen finite vectors")
    originals = tuple(vector.detach().clone() for vector in vectors)
    projected = []
    for index, vector in enumerate(originals):
        current = vector.clone()
        for other_index, other in enumerate(originals):
            if index == other_index:
                continue
            dot = torch.dot(current, other)
            denominator = torch.dot(other, other)
            if float(dot) < 0.0 and float(denominator) > 0.0:
                current = current - strength * dot / denominator * other
        projected.append(current)
    return tuple(projected)


def _cosines(vectors: tuple[torch.Tensor, ...]) -> dict[str, float]:
    result = {}
    for (first_index, first), (second_index, second) in combinations(
        enumerate(SOURCES), 2
    ):
        first_vector = vectors[first_index]
        second_vector = vectors[second_index]
        if float(torch.linalg.vector_norm(first_vector)) == 0.0 or float(
            torch.linalg.vector_norm(second_vector)
        ) == 0.0:
            raise RuntimeError("gradient audit encountered a zero source vector")
        result[f"{first}__{second}"] = float(F.cosine_similarity(
            first_vector[None, :], second_vector[None, :]
        ).item())
    return result


def _base_candidate():
    matched = tuple(
        candidate for candidate in candidate_registry_v4()
        if candidate.candidate_id == "NMR4-001"
    )
    if len(matched) != 1:
        raise RuntimeError("the locked v4 base candidate is unavailable")
    return matched[0]


def _projection_parameter_mask(
    names: tuple[str, ...], candidate: ConflictAwareCandidateV5
) -> tuple[bool, ...]:
    if candidate.projection_scope == "patient_block":
        return tuple(
            name.startswith("backbone.patient_projection")
            or name.startswith("backbone.patient_norm")
            for name in names
        )
    return tuple(not name.startswith("backbone.task_heads") for name in names)


def _flatten_selected(
    gradients: tuple[torch.Tensor, ...], selected: tuple[bool, ...]
) -> torch.Tensor:
    return torch.cat([
        gradient.reshape(-1) for gradient, include in zip(gradients, selected)
        if include
    ])


def _assign_projected_gradients(
    parameters: tuple[torch.nn.Parameter, ...],
    source_gradients: tuple[tuple[torch.Tensor, ...], ...],
    selected: tuple[bool, ...],
    projected_vectors: tuple[torch.Tensor, ...],
) -> None:
    offsets = [0 for _ in projected_vectors]
    for parameter_index, parameter in enumerate(parameters):
        if selected[parameter_index]:
            pieces = []
            size = parameter.numel()
            for source_index, vector in enumerate(projected_vectors):
                start = offsets[source_index]
                pieces.append(vector[start:start + size].reshape_as(parameter))
                offsets[source_index] += size
            parameter.grad = torch.stack(pieces).sum(dim=0)
        else:
            parameter.grad = torch.stack([
                gradients[parameter_index] for gradients in source_gradients
            ]).sum(dim=0)
    if any(offset != vector.numel() for offset, vector in zip(offsets, projected_vectors)):
        raise RuntimeError("projected gradient vector was not consumed exactly")


@dataclass(frozen=True)
class ConflictAwareEvaluationV5:
    probabilities: np.ndarray
    metrics: dict[str, dict[str, float]]
    model_fits: int
    threshold: float
    pre_projection_cosines: dict[str, float]
    post_projection_cosines: dict[str, float]


def evaluate_conflict_aware_candidate(
    dataset: MedicalSharedDatasetV2,
    candidate: ConflictAwareCandidateV5,
    *,
    epochs: int,
    n_splits: int = 6,
    seed: int = 0,
    device: str = "cpu",
) -> ConflictAwareEvaluationV5:
    if (
        type(dataset) is not MedicalSharedDatasetV2
        or type(candidate) is not ConflictAwareCandidateV5
        or candidate not in candidate_registry_v5()
        or isinstance(epochs, bool)
        or not isinstance(epochs, int)
        or epochs < 1
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or type(device) is not str
    ):
        raise ValueError("invalid conflict-aware evaluation configuration")
    runtime = torch.device(device)
    if runtime.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    base = dataset.base
    folds = participant_disjoint_folds(base, n_splits=n_splits)
    probabilities = np.full(len(base.labels), np.nan, dtype=np.float64)
    pre_cosines = None
    post_cosines = None
    for fold_index, (train, held) in enumerate(folds):
        local_seed = seed * 1009 + fold_index
        torch.manual_seed(local_seed)
        if runtime.type == "cuda":
            torch.cuda.manual_seed_all(local_seed)
        scaler = fit_clinical_scaler(base, train)
        original = _scaled(base.clinical_original, scaler.mean, scaler.scale)
        mirrored = _scaled(base.clinical_mirrored, scaler.mean, scaler.scale)
        model = SharedNormalManifoldRouterV4(_base_candidate()).to(runtime)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
        named_parameters = tuple(model.named_parameters())
        names = tuple(name for name, _ in named_parameters)
        parameters = tuple(parameter for _, parameter in named_parameters)
        selected = _projection_parameter_mask(names, candidate)
        if not any(selected):
            raise RuntimeError("conflict projection selected no shared parameters")
        train_sources = tuple(base.sources[index] for index in train)
        participant_weights = source_class_balanced_weights(
            base.labels[train], train_sources
        )
        weights = _tensor(participant_weights.astype(np.float32), runtime)
        labels = _tensor(base.labels[train].astype(np.float32), runtime)
        task_codes = _tensor(np.asarray([
            _SOURCE_TASK_CODE[source] for source in train_sources
        ], dtype=np.int64), runtime)
        train_inputs = _model_inputs(dataset, original, mirrored, train, runtime)
        source_indices = tuple(torch.tensor([
            index for index, observed in enumerate(train_sources) if observed == source
        ], dtype=torch.long, device=runtime) for source in SOURCES)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            source_gradients = []
            for local in source_indices:
                local_inputs = tuple(value.index_select(0, local) for value in train_inputs)
                embedding = model.encode(*local_inputs)
                local_labels = labels.index_select(0, local)
                routed = model.routed_logits_from_embedding(
                    embedding, task_codes.index_select(0, local)
                )
                universal = model.universal_logits_from_embedding(embedding)
                losses = F.binary_cross_entropy_with_logits(
                    routed, local_labels, reduction="none"
                ) + UNIVERSAL_AUXILIARY_WEIGHT * F.binary_cross_entropy_with_logits(
                    universal, local_labels, reduction="none"
                )
                source_loss = torch.sum(losses * weights.index_select(0, local))
                gradients = torch.autograd.grad(
                    source_loss, parameters, allow_unused=True
                )
                source_gradients.append(tuple(
                    torch.zeros_like(parameter) if gradient is None else gradient.detach()
                    for parameter, gradient in zip(parameters, gradients)
                ))
            source_gradient_tuple = tuple(source_gradients)
            vectors = tuple(
                _flatten_selected(gradients, selected)
                for gradients in source_gradient_tuple
            )
            projected = project_conflicting_gradient_vectors(
                vectors, strength=candidate.projection_strength
            )
            if fold_index == 0 and epoch == 0:
                pre_cosines = _cosines(vectors)
                post_cosines = _cosines(projected)
            _assign_projected_gradients(
                parameters, source_gradient_tuple, selected, projected
            )
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            held_inputs = _model_inputs(dataset, original, mirrored, held, runtime)
            embedding = model.encode(*held_inputs)
            held_tasks = _tensor(np.asarray([
                _SOURCE_TASK_CODE[base.sources[index]] for index in held
            ], dtype=np.int64), runtime)
            probabilities[held] = torch.sigmoid(
                model.routed_logits_from_embedding(embedding, held_tasks)
            ).cpu().numpy()
        del model, optimizer
    if not np.isfinite(probabilities).all() or pre_cosines is None or post_cosines is None:
        raise RuntimeError("conflict-aware evaluation did not cover all participants")
    metrics = {}
    for source in SOURCES:
        selected_source = np.asarray([observed == source for observed in base.sources])
        metrics[source] = binary_metrics(
            base.labels[selected_source], probabilities[selected_source]
        )
    return ConflictAwareEvaluationV5(
        probabilities=_immutable(probabilities), metrics=metrics,
        model_fits=len(folds), threshold=0.5,
        pre_projection_cosines=pre_cosines,
        post_projection_cosines=post_cosines,
    )


def rank_conflict_aware_results(
    results: dict[str, ConflictAwareEvaluationV5],
) -> tuple[str, ...]:
    expected = {candidate.candidate_id for candidate in candidate_registry_v5()}
    if type(results) is not dict or set(results) != expected or any(
        type(result) is not ConflictAwareEvaluationV5 for result in results.values()
    ):
        raise ValueError("ranking requires every frozen v5 candidate")

    def key(candidate_id: str):
        metrics = results[candidate_id].metrics
        balanced = [float(metrics[source]["balanced_accuracy"]) for source in SOURCES]
        specificity = [float(metrics[source]["specificity"]) for source in SOURCES]
        aurocs = [float(metrics[source]["auroc"]) for source in SOURCES]
        return (
            -min(balanced), -min(specificity), -min(aurocs),
            -float(np.mean(balanced)), candidate_id,
        )

    return tuple(sorted(expected, key=key))


__all__ = [
    "ConflictAwareEvaluationV5",
    "evaluate_conflict_aware_candidate",
    "project_conflicting_gradient_vectors",
    "rank_conflict_aware_results",
]
