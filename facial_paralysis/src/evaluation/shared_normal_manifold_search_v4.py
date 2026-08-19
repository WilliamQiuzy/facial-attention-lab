"""Participant-disjoint evaluation of shared normal-manifold candidates."""
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
from src.evaluation.universal_orofacial_v1 import binary_metrics
from src.models.normal_manifold_candidate_registry_v4 import (
    NormalManifoldCandidateV4,
    candidate_registry_v4,
)
from src.models.shared_normal_manifold_router_v4 import SharedNormalManifoldRouterV4


UNIVERSAL_AUXILIARY_WEIGHT = 0.50
_SOURCE_TASK_CODE = {source: index for index, source in enumerate(SOURCES)}


def _immutable(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


@dataclass(frozen=True)
class NormalManifoldEvaluationV4:
    probabilities: np.ndarray
    metrics: dict[str, dict[str, float]]
    model_fits: int
    threshold: float
    gradient_cosines: dict[str, float]


def _source_gradient_cosines(
    model: SharedNormalManifoldRouterV4,
    train_inputs: tuple[torch.Tensor, ...],
    labels: torch.Tensor,
    task_codes: torch.Tensor,
    train_sources: tuple[str, ...],
) -> dict[str, float]:
    gradients: dict[str, torch.Tensor] = {}
    for source in SOURCES:
        local = torch.tensor(
            [index for index, observed in enumerate(train_sources) if observed == source],
            dtype=torch.long,
            device=labels.device,
        )
        model.zero_grad(set_to_none=True)
        source_inputs = tuple(values.index_select(0, local) for values in train_inputs)
        embedding = model.encode(*source_inputs)
        routed = model.routed_logits_from_embedding(
            embedding, task_codes.index_select(0, local)
        )
        universal = model.universal_logits_from_embedding(embedding)
        local_labels = labels.index_select(0, local)
        loss = F.binary_cross_entropy_with_logits(routed, local_labels)
        loss = loss + UNIVERSAL_AUXILIARY_WEIGHT * F.binary_cross_entropy_with_logits(
            universal, local_labels
        )
        loss.backward()
        gradient = model.backbone.patient_projection.weight.grad
        if gradient is None or float(torch.linalg.vector_norm(gradient)) <= 0.0:
            raise RuntimeError("a source failed the shared patient-gradient audit")
        gradients[source] = gradient.detach().flatten().cpu()
    result = {}
    for first, second in combinations(SOURCES, 2):
        result[f"{first}__{second}"] = float(F.cosine_similarity(
            gradients[first][None, :], gradients[second][None, :]
        ).item())
    return result


def evaluate_normal_manifold_candidate(
    dataset: MedicalSharedDatasetV2,
    candidate: NormalManifoldCandidateV4,
    *,
    epochs: int,
    n_splits: int = 6,
    seed: int = 0,
    device: str = "cpu",
) -> NormalManifoldEvaluationV4:
    if (
        type(dataset) is not MedicalSharedDatasetV2
        or type(candidate) is not NormalManifoldCandidateV4
        or candidate not in candidate_registry_v4()
        or isinstance(epochs, bool)
        or not isinstance(epochs, int)
        or epochs < 1
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or type(device) is not str
    ):
        raise ValueError("invalid normal-manifold evaluation configuration")
    runtime = torch.device(device)
    if runtime.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    base = dataset.base
    folds = participant_disjoint_folds(base, n_splits=n_splits)
    probabilities = np.full(len(base.labels), np.nan, dtype=np.float64)
    gradient_cosines: dict[str, float] | None = None
    for fold_index, (train, held) in enumerate(folds):
        local_seed = seed * 1009 + fold_index
        torch.manual_seed(local_seed)
        if runtime.type == "cuda":
            torch.cuda.manual_seed_all(local_seed)
        scaler = fit_clinical_scaler(base, train)
        original = _scaled(base.clinical_original, scaler.mean, scaler.scale)
        mirrored = _scaled(base.clinical_mirrored, scaler.mean, scaler.scale)
        model = SharedNormalManifoldRouterV4(candidate).to(runtime)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
        train_sources = tuple(base.sources[index] for index in train)
        participant_weights = source_class_balanced_weights(
            base.labels[train], train_sources
        )
        weights = _tensor(participant_weights.astype(np.float32), runtime)
        labels = _tensor(base.labels[train].astype(np.float32), runtime)
        task_codes = _tensor(np.asarray(
            [_SOURCE_TASK_CODE[source] for source in train_sources], dtype=np.int64
        ), runtime)
        train_inputs = _model_inputs(dataset, original, mirrored, train, runtime)
        if fold_index == 0:
            gradient_cosines = _source_gradient_cosines(
                model, train_inputs, labels, task_codes, train_sources
            )
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            embedding = model.encode(*train_inputs)
            routed = model.routed_logits_from_embedding(embedding, task_codes)
            universal = model.universal_logits_from_embedding(embedding)
            prediction_losses = F.binary_cross_entropy_with_logits(
                routed, labels, reduction="none"
            ) + UNIVERSAL_AUXILIARY_WEIGHT * F.binary_cross_entropy_with_logits(
                universal, labels, reduction="none"
            )
            loss = torch.sum(prediction_losses * weights)
            loss = loss + candidate.normal_weight * model.normal_manifold_loss(
                embedding, labels, weights
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    if not np.isfinite(probabilities).all() or gradient_cosines is None:
        raise RuntimeError("normal-manifold evaluation did not cover all participants")
    metrics = {}
    for source in SOURCES:
        selected = np.asarray([observed == source for observed in base.sources])
        metrics[source] = binary_metrics(base.labels[selected], probabilities[selected])
    return NormalManifoldEvaluationV4(
        probabilities=_immutable(probabilities),
        metrics=metrics,
        model_fits=len(folds),
        threshold=0.5,
        gradient_cosines=gradient_cosines,
    )


def rank_normal_manifold_results(
    results: dict[str, NormalManifoldEvaluationV4],
) -> tuple[str, ...]:
    expected = {candidate.candidate_id for candidate in candidate_registry_v4()}
    if type(results) is not dict or set(results) != expected or any(
        type(result) is not NormalManifoldEvaluationV4 for result in results.values()
    ):
        raise ValueError("ranking requires every frozen v4 candidate")

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
    "NormalManifoldEvaluationV4",
    "evaluate_normal_manifold_candidate",
    "rank_normal_manifold_results",
]
