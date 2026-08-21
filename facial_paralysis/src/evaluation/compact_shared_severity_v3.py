"""Participant-disjoint evaluation for compact shared severity v3."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from src.evaluation.medically_gated_shared_search_v2 import MedicalSharedDatasetV2
from src.evaluation.shared_clinical_encoder_v1 import (
    SOURCES,
    ClinicalScaler,
    SharedEvaluation,
    participant_disjoint_folds,
    source_class_balanced_weights,
)
from src.evaluation.universal_orofacial_v1 import binary_metrics
from src.models.compact_shared_severity_v3 import (
    CompactCandidateV3,
    CompactSharedSeverityV3,
    compact_candidate_registry,
)


UNIVERSAL_SEVERITY_WEIGHT = 0.5
PAIRWISE_SEVERITY_WEIGHT = 0.1
_SOURCE_TASK_CODE = {source: index for index, source in enumerate(SOURCES)}


def _immutable(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def _fit_original_scaler(
    dataset: MedicalSharedDatasetV2,
    train_indices: np.ndarray,
) -> ClinicalScaler:
    base = dataset.base
    rows = []
    weights = []
    for participant in train_indices:
        valid = base.action_mask[participant]
        count = int(valid.sum())
        for row in base.clinical_original[participant, valid]:
            rows.append(row.astype(np.float64, copy=False))
            weights.append(1.0 / (len(train_indices) * count))
    matrix = np.stack(rows)
    mass = np.asarray(weights, dtype=np.float64)
    mean = np.average(matrix, axis=0, weights=mass)
    variance = np.average((matrix - mean) ** 2, axis=0, weights=mass)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    return ClinicalScaler(_immutable(mean), _immutable(scale))


def _scaled(
    values: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    result = (values.astype(np.float64, copy=False) - mean[None, None, :]) / (
        scale[None, None, :]
    )
    if not np.isfinite(result).all():
        raise ValueError("original-only fold scaling produced nonfinite evidence")
    return result.astype(np.float32)


def _tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.array(values, copy=True)).to(device)


def _model_inputs(
    dataset: MedicalSharedDatasetV2,
    clinical: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    base = dataset.base
    return (
        _tensor(clinical[indices], device),
        _tensor(base.dense_original[indices], device),
        _tensor(base.dense_available[indices], device),
        _tensor(dataset.dense_timestamps[indices].astype(np.float32), device),
        _tensor(base.action_mask[indices], device),
        _tensor(base.action_codes[indices], device),
    )


def _pairwise_severity_loss(
    severity: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    positive = severity[labels == 1.0]
    negative = severity[labels == 0.0]
    if positive.numel() < 1 or negative.numel() < 1:
        raise ValueError("shared severity ordering requires both binary classes")
    return F.softplus(-(positive[:, None] - negative[None, :])).mean()


def evaluate_compact_candidate(
    dataset: MedicalSharedDatasetV2,
    candidate: CompactCandidateV3,
    *,
    epochs: int,
    n_splits: int = 6,
    seed: int = 0,
    device: str = "cpu",
) -> SharedEvaluation:
    if (
        type(dataset) is not MedicalSharedDatasetV2
        or type(candidate) is not CompactCandidateV3
        or candidate not in compact_candidate_registry()
        or isinstance(epochs, bool)
        or not isinstance(epochs, int)
        or epochs < 1
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or type(device) is not str
    ):
        raise ValueError("invalid compact shared evaluation configuration")
    runtime = torch.device(device)
    if runtime.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    base = dataset.base
    folds = participant_disjoint_folds(base, n_splits=n_splits)
    probabilities = np.full(len(base.labels), np.nan, dtype=np.float64)
    audited_sources: set[str] = set()
    for fold_index, (train, held) in enumerate(folds):
        local_seed = seed * 1009 + fold_index
        torch.manual_seed(local_seed)
        if runtime.type == "cuda":
            torch.cuda.manual_seed_all(local_seed)
        scaler = _fit_original_scaler(dataset, train)
        clinical = _scaled(
            base.clinical_original, scaler.mean, scaler.scale
        )
        model = CompactSharedSeverityV3(candidate).to(runtime)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, weight_decay=2e-2
        )
        train_sources = tuple(base.sources[index] for index in train)
        weights = _tensor(source_class_balanced_weights(
            base.labels[train], train_sources
        ).astype(np.float32), runtime)
        labels = _tensor(base.labels[train].astype(np.float32), runtime)
        task_codes = _tensor(np.asarray([
            _SOURCE_TASK_CODE[source] for source in train_sources
        ], dtype=np.int64), runtime)
        train_inputs = _model_inputs(dataset, clinical, train, runtime)

        if fold_index == 0:
            for source in SOURCES:
                local = np.asarray([
                    index for index, observed in enumerate(train_sources)
                    if observed == source
                ], dtype=np.int64)
                model.zero_grad(set_to_none=True)
                embedding = model.encode(*(values[local] for values in train_inputs))
                local_labels = labels[local]
                source_loss = F.binary_cross_entropy_with_logits(
                    model.task_logits_from_embedding(embedding, task_codes[local]),
                    local_labels,
                ) + UNIVERSAL_SEVERITY_WEIGHT * F.binary_cross_entropy_with_logits(
                    model.shared_severity(embedding).squeeze(-1), local_labels,
                )
                source_loss.backward()
                clinical_gradient = model.clinical_encoder[0].weight.grad
                patient_gradient = model.patient_projection[0].weight.grad
                severity_gradient = model.shared_severity.weight.grad
                if any(
                    gradient is None or float(gradient.norm()) <= 0.0
                    for gradient in (
                        clinical_gradient, patient_gradient, severity_gradient
                    )
                ):
                    raise RuntimeError("a source failed to update the shared severity trunk")
                audited_sources.add(source)

        for _ in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            embedding = model.encode(*train_inputs)
            task_logits = model.task_logits_from_embedding(embedding, task_codes)
            severity = model.shared_severity(embedding).squeeze(-1)
            losses = F.binary_cross_entropy_with_logits(
                task_logits, labels, reduction="none"
            ) + UNIVERSAL_SEVERITY_WEIGHT * F.binary_cross_entropy_with_logits(
                severity, labels, reduction="none"
            )
            loss = torch.sum(losses * weights) + (
                PAIRWISE_SEVERITY_WEIGHT
                * _pairwise_severity_loss(severity, labels)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            held_inputs = _model_inputs(dataset, clinical, held, runtime)
            embedding = model.encode(*held_inputs)
            held_tasks = _tensor(np.asarray([
                _SOURCE_TASK_CODE[base.sources[index]] for index in held
            ], dtype=np.int64), runtime)
            probabilities[held] = torch.sigmoid(
                model.task_logits_from_embedding(embedding, held_tasks)
            ).cpu().numpy()
        del model, optimizer
    if not np.isfinite(probabilities).all() or audited_sources != set(SOURCES):
        raise RuntimeError("compact shared evaluation did not cover all evidence")
    metrics = {}
    for source in SOURCES:
        selected = np.asarray([observed == source for observed in base.sources])
        metrics[source] = binary_metrics(
            base.labels[selected], probabilities[selected]
        )
    return SharedEvaluation(
        probabilities=_immutable(probabilities),
        metrics=metrics,
        model_fits=len(folds),
        threshold=0.5,
        shared_gradient_sources=tuple(
            source for source in SOURCES if source in audited_sources
        ),
    )


def rank_compact_results(
    results: dict[str, SharedEvaluation],
) -> tuple[str, ...]:
    expected = {
        candidate.candidate_id for candidate in compact_candidate_registry()
    }
    if type(results) is not dict or set(results) != expected or any(
        type(result) is not SharedEvaluation for result in results.values()
    ):
        raise ValueError("ranking requires all compact candidates")

    def key(candidate_id: str):
        metrics = results[candidate_id].metrics
        accuracies = [float(metrics[source]["accuracy"]) for source in SOURCES]
        aurocs = [float(metrics[source]["auroc"]) for source in SOURCES]
        return (
            -min(accuracies),
            -min(aurocs),
            -float(np.mean(accuracies)),
            candidate_id,
        )

    return tuple(sorted(expected, key=key))


__all__ = [
    "evaluate_compact_candidate",
    "rank_compact_results",
]
