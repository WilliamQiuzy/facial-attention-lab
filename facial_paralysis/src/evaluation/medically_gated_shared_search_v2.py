"""Participant-disjoint shared-only evaluation for medically gated v2."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from src.evaluation.shared_clinical_encoder_v1 import (
    SOURCES,
    SharedClinicalDataset,
    SharedEvaluation,
    fit_clinical_scaler,
    participant_disjoint_folds,
    source_class_balanced_weights,
)
from src.evaluation.universal_orofacial_v1 import binary_metrics
from src.models.medical_shared_candidate_registry_v2 import (
    SharedCandidateV2,
    candidate_registry,
)
from src.models.medically_gated_shared_encoder_v2 import (
    MedicallyGatedSharedEncoderV2,
)


UNIVERSAL_AUXILIARY_WEIGHT = 0.25
_SOURCE_TASK_CODE = {source: index for index, source in enumerate(SOURCES)}


def _immutable(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


@dataclass(frozen=True)
class MedicalSharedDatasetV2:
    base: SharedClinicalDataset
    dense_timestamps: np.ndarray

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.dense_timestamps)
        if (
            type(self.base) is not SharedClinicalDataset
            or type(self.dense_timestamps) is not np.ndarray
            or timestamps.dtype != np.dtype(np.float64)
            or timestamps.shape != self.base.dense_valid_mask.shape
            or not np.isfinite(timestamps).all()
        ):
            raise ValueError("v2 timestamps differ from the exact dense evidence contract")
        available = self.base.dense_available
        if available.any():
            intervals = np.diff(timestamps, axis=-1)
            if np.any(intervals[available] <= 0.0):
                raise ValueError("available dense frames require increasing real seconds")
            if np.any(self.base.dense_valid_mask[available].sum(axis=-1) != 32):
                raise ValueError("v2 requires each dense action on the full 32-frame grid")
        if np.any(timestamps[~available] != 0.0):
            raise ValueError("unavailable dense actions cannot carry timestamps")
        object.__setattr__(self, "dense_timestamps", _immutable(timestamps))


def _scaled(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    result = (
        values.astype(np.float64, copy=False) - mean[None, None, :]
    ) / scale[None, None, :]
    if not np.isfinite(result).all():
        raise ValueError("fold-local clinical scaling produced nonfinite values")
    return result.astype(np.float32)


def _tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.array(values, copy=True)).to(device)


def _model_inputs(
    dataset: MedicalSharedDatasetV2,
    clinical_original: np.ndarray,
    clinical_mirrored: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    base = dataset.base
    return (
        _tensor(clinical_original[indices], device),
        _tensor(clinical_mirrored[indices], device),
        _tensor(base.dense_original[indices], device),
        _tensor(base.dense_mirrored[indices], device),
        _tensor(base.dense_valid_mask[indices], device),
        _tensor(base.dense_available[indices], device),
        _tensor(dataset.dense_timestamps[indices].astype(np.float32), device),
        _tensor(base.action_mask[indices], device),
        _tensor(base.action_codes[indices], device),
    )


def evaluate_medical_candidate(
    dataset: MedicalSharedDatasetV2,
    candidate: SharedCandidateV2,
    *,
    epochs: int,
    n_splits: int = 6,
    seed: int = 0,
    device: str = "cpu",
) -> SharedEvaluation:
    if (
        type(dataset) is not MedicalSharedDatasetV2
        or type(candidate) is not SharedCandidateV2
        or candidate not in candidate_registry()
        or isinstance(epochs, bool)
        or not isinstance(epochs, int)
        or epochs < 1
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or type(device) is not str
    ):
        raise ValueError("invalid medically gated evaluation configuration")
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
        scaler = fit_clinical_scaler(base, train)
        original = _scaled(base.clinical_original, scaler.mean, scaler.scale)
        mirrored = _scaled(base.clinical_mirrored, scaler.mean, scaler.scale)
        model = MedicallyGatedSharedEncoderV2(candidate).to(runtime)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=3e-4, weight_decay=1e-3
        )
        train_sources = tuple(base.sources[index] for index in train)
        participant_weights = source_class_balanced_weights(
            base.labels[train], train_sources
        )
        weights = _tensor(participant_weights.astype(np.float32), runtime)
        labels = _tensor(base.labels[train].astype(np.float32), runtime)
        task_codes = _tensor(np.asarray(
            [_SOURCE_TASK_CODE[source] for source in train_sources],
            dtype=np.int64,
        ), runtime)
        train_inputs = _model_inputs(
            dataset, original, mirrored, train, runtime
        )

        if fold_index == 0:
            for source in SOURCES:
                local = np.asarray([
                    index for index, observed in enumerate(train_sources)
                    if observed == source
                ], dtype=np.int64)
                model.zero_grad(set_to_none=True)
                source_inputs = tuple(values[local] for values in train_inputs)
                embedding = model.encode(*source_inputs)
                loss = F.binary_cross_entropy_with_logits(
                    model.task_logits_from_embedding(embedding, task_codes[local]),
                    labels[local],
                )
                loss.backward()
                clinical_gradient = model.clinical_encoder[0].weight.grad
                patient_gradient = model.patient_projection.weight.grad
                if (
                    clinical_gradient is None
                    or patient_gradient is None
                    or float(clinical_gradient.norm()) <= 0.0
                    or float(patient_gradient.norm()) <= 0.0
                ):
                    raise RuntimeError("a source failed to update the shared trunk")
                audited_sources.add(source)

        for _ in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            embedding = model.encode(*train_inputs)
            task_logits = model.task_logits_from_embedding(embedding, task_codes)
            universal_logits = model.universal_head(embedding).squeeze(-1)
            losses = F.binary_cross_entropy_with_logits(
                task_logits, labels, reduction="none"
            ) + UNIVERSAL_AUXILIARY_WEIGHT * F.binary_cross_entropy_with_logits(
                universal_logits, labels, reduction="none"
            )
            torch.sum(losses * weights).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            held_inputs = _model_inputs(
                dataset, original, mirrored, held, runtime
            )
            embedding = model.encode(*held_inputs)
            held_tasks = _tensor(np.asarray([
                _SOURCE_TASK_CODE[base.sources[index]] for index in held
            ], dtype=np.int64), runtime)
            probabilities[held] = torch.sigmoid(
                model.task_logits_from_embedding(embedding, held_tasks)
            ).cpu().numpy()
        del model, optimizer
    if not np.isfinite(probabilities).all() or audited_sources != set(SOURCES):
        raise RuntimeError("candidate evaluation did not cover all people and sources")
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


def rank_candidate_results(
    results: dict[str, SharedEvaluation],
) -> tuple[str, ...]:
    expected = {candidate.candidate_id for candidate in candidate_registry()}
    if type(results) is not dict or set(results) != expected or any(
        type(result) is not SharedEvaluation for result in results.values()
    ):
        raise ValueError("ranking requires one validated result per frozen candidate")

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
    "MedicalSharedDatasetV2",
    "evaluate_medical_candidate",
    "rank_candidate_results",
]
