"""Participant-disjoint evaluation for the dense-clinical shared encoder."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from src.evaluation.universal_orofacial_v1 import binary_metrics
from src.models.dense_clinical_shared_encoder_v1 import (
    ACTION_VOCAB,
    DenseClinicalSharedEncoder,
)


SOURCES = ("palsynet", "neuroface", "meei")
UNIVERSAL_AUXILIARY_WEIGHT = 0.25
_SOURCE_TASK_CODE = {source: index for index, source in enumerate(SOURCES)}
_GROUP_ID = re.compile(r"grp_[0-9a-f]{64}\Z")


def _immutable(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


@dataclass(frozen=True)
class SharedClinicalDataset:
    clinical_original: np.ndarray
    clinical_mirrored: np.ndarray
    dense_original: np.ndarray
    dense_mirrored: np.ndarray
    dense_valid_mask: np.ndarray
    dense_available: np.ndarray
    action_mask: np.ndarray
    action_codes: np.ndarray
    labels: np.ndarray
    group_ids: tuple[str, ...]
    sources: tuple[str, ...]

    def __post_init__(self) -> None:
        clinical = np.asarray(self.clinical_original)
        mirrored = np.asarray(self.clinical_mirrored)
        if clinical.ndim != 3:
            raise ValueError("clinical dataset must have batch, action and feature axes")
        count, actions, dimension = clinical.shape
        dense_shape = (count, actions, 32, 478, 3)
        arrays = {
            "clinical_original": (clinical, (count, actions, 110), np.dtype(np.float32)),
            "clinical_mirrored": (mirrored, (count, actions, 110), np.dtype(np.float32)),
            "dense_original": (np.asarray(self.dense_original), dense_shape, np.dtype(np.float32)),
            "dense_mirrored": (np.asarray(self.dense_mirrored), dense_shape, np.dtype(np.float32)),
            "dense_valid_mask": (
                np.asarray(self.dense_valid_mask), dense_shape[:3], np.dtype(bool)
            ),
            "dense_available": (
                np.asarray(self.dense_available), (count, actions), np.dtype(bool)
            ),
            "action_mask": (
                np.asarray(self.action_mask), (count, actions), np.dtype(bool)
            ),
            "action_codes": (
                np.asarray(self.action_codes), (count, actions), np.dtype(np.int64)
            ),
            "labels": (np.asarray(self.labels), (count,), np.dtype(np.int64)),
        }
        if count < 2 or dimension != 110:
            raise ValueError("shared dataset needs at least two 110D participants")
        for name, (values, shape, dtype) in arrays.items():
            if values.shape != shape or values.dtype != dtype:
                raise ValueError(f"{name} differs from the exact dataset contract")
        if (
            not np.isfinite(clinical).all()
            or not np.isfinite(mirrored).all()
            or not np.isfinite(arrays["dense_original"][0]).all()
            or not np.isfinite(arrays["dense_mirrored"][0]).all()
            or not np.isin(arrays["labels"][0], (0, 1)).all()
            or np.any(arrays["action_mask"][0].sum(axis=1) == 0)
            or np.any(arrays["dense_available"][0] & ~arrays["action_mask"][0])
            or np.any(
                arrays["dense_valid_mask"][0]
                & ~arrays["dense_available"][0][..., None]
            )
            or np.any(arrays["action_codes"][0] < 0)
            or np.any(arrays["action_codes"][0] >= len(ACTION_VOCAB))
        ):
            raise ValueError("shared dataset failed finite, label, mask or action QC")
        if (
            type(self.group_ids) is not tuple
            or type(self.sources) is not tuple
            or len(self.group_ids) != count
            or len(self.sources) != count
            or len(set(self.group_ids)) != count
            or any(_GROUP_ID.fullmatch(group) is None for group in self.group_ids)
            or any(source not in SOURCES for source in self.sources)
        ):
            raise ValueError("participant identities or sources differ from the closed registry")
        for name, (values, _, _) in arrays.items():
            object.__setattr__(self, name, _immutable(values))


@dataclass(frozen=True)
class ClinicalScaler:
    mean: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class SharedEvaluation:
    probabilities: np.ndarray
    metrics: dict[str, dict[str, float]]
    model_fits: int
    threshold: float
    shared_gradient_sources: tuple[str, ...]


def participant_disjoint_folds(
    dataset: SharedClinicalDataset,
    *,
    n_splits: int = 6,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    if not isinstance(dataset, SharedClinicalDataset):
        raise ValueError("validated SharedClinicalDataset required")
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least two")
    assignments = np.full(len(dataset.labels), -1, dtype=np.int64)
    for source in SOURCES:
        for label in (0, 1):
            cell = sorted(
                (
                    dataset.group_ids[index],
                    index,
                )
                for index in range(len(dataset.labels))
                if dataset.sources[index] == source
                and int(dataset.labels[index]) == label
            )
            if len(cell) < n_splits:
                raise ValueError("every source-label cell must cover every fold")
            for position, (_, index) in enumerate(cell):
                assignments[index] = position % n_splits
    if np.any(assignments < 0):
        raise ValueError("a participant escaped source-label stratification")
    all_indices = np.arange(len(dataset.labels), dtype=np.int64)
    return tuple(
        (
            _immutable(all_indices[assignments != fold]),
            _immutable(all_indices[assignments == fold]),
        )
        for fold in range(n_splits)
    )


def source_class_balanced_weights(
    labels: np.ndarray,
    sources: Sequence[str],
) -> np.ndarray:
    labels = np.asarray(labels)
    sources = tuple(sources)
    if (
        labels.shape != (len(sources),)
        or labels.dtype != np.dtype(np.int64)
        or not np.isin(labels, (0, 1)).all()
        or any(source not in SOURCES for source in sources)
    ):
        raise ValueError("source weights require aligned frozen-source binary labels")
    cells = [
        (source, label)
        for source in SOURCES
        for label in (0, 1)
        if any(
            observed_source == source and int(observed_label) == label
            for observed_source, observed_label in zip(sources, labels)
        )
    ]
    weights = np.zeros(len(labels), dtype=np.float64)
    for source, label in cells:
        selected = np.asarray([
            observed_source == source and int(observed_label) == label
            for observed_source, observed_label in zip(sources, labels)
        ])
        weights[selected] = 1.0 / (len(cells) * int(selected.sum()))
    if not np.isclose(weights.sum(), 1.0) or np.any(weights <= 0.0):
        raise ValueError("every observed source-label cell must receive positive mass")
    return _immutable(weights)


def fit_clinical_scaler(
    dataset: SharedClinicalDataset,
    train_indices: np.ndarray,
) -> ClinicalScaler:
    indices = np.asarray(train_indices)
    if (
        not isinstance(dataset, SharedClinicalDataset)
        or indices.ndim != 1
        or indices.dtype != np.dtype(np.int64)
        or indices.size < 2
        or np.any(indices < 0)
        or np.any(indices >= len(dataset.labels))
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError("scaler requires unique in-range training participants")
    rows: list[np.ndarray] = []
    row_weights: list[float] = []
    participant_mass = 1.0 / len(indices)
    for participant in indices:
        valid_actions = dataset.action_mask[participant]
        action_count = int(valid_actions.sum())
        for view in (
            dataset.clinical_original[participant],
            dataset.clinical_mirrored[participant],
        ):
            for row in view[valid_actions]:
                rows.append(row.astype(np.float64, copy=False))
                row_weights.append(participant_mass / (2.0 * action_count))
    matrix = np.stack(rows)
    weights = np.asarray(row_weights, dtype=np.float64)
    mean = np.average(matrix, axis=0, weights=weights)
    variance = np.average((matrix - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    return ClinicalScaler(_immutable(mean), _immutable(scale))


def _scaled(values: np.ndarray, scaler: ClinicalScaler) -> np.ndarray:
    result = (
        values.astype(np.float64, copy=False) - scaler.mean[None, None, :]
    ) / scaler.scale[None, None, :]
    if not np.isfinite(result).all():
        raise ValueError("fold-local clinical scaling produced nonfinite values")
    return result.astype(np.float32)


def _tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.array(values, copy=True)).to(device)


def _model_inputs(
    dataset: SharedClinicalDataset,
    clinical_original: np.ndarray,
    clinical_mirrored: np.ndarray,
    indices: np.ndarray,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    return (
        _tensor(clinical_original[indices], device),
        _tensor(clinical_mirrored[indices], device),
        _tensor(dataset.dense_original[indices], device),
        _tensor(dataset.dense_mirrored[indices], device),
        _tensor(dataset.dense_valid_mask[indices], device),
        _tensor(dataset.dense_available[indices], device),
        _tensor(dataset.action_mask[indices], device),
        _tensor(dataset.action_codes[indices], device),
    )


def evaluate_shared_model(
    dataset: SharedClinicalDataset,
    *,
    use_dense: bool,
    epochs: int,
    n_splits: int = 6,
    seed: int = 0,
    device: str = "cpu",
) -> SharedEvaluation:
    if (
        not isinstance(dataset, SharedClinicalDataset)
        or type(use_dense) is not bool
        or isinstance(epochs, bool)
        or not isinstance(epochs, int)
        or epochs < 1
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or type(device) is not str
    ):
        raise ValueError("shared evaluation configuration is invalid")
    runtime = torch.device(device)
    if runtime.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    folds = participant_disjoint_folds(dataset, n_splits=n_splits)
    probabilities = np.full(len(dataset.labels), np.nan, dtype=np.float64)
    audited_sources: set[str] = set()
    for fold_index, (train, held) in enumerate(folds):
        torch.manual_seed(seed * 1009 + fold_index)
        if runtime.type == "cuda":
            torch.cuda.manual_seed_all(seed * 1009 + fold_index)
        scaler = fit_clinical_scaler(dataset, train)
        original = _scaled(dataset.clinical_original, scaler)
        mirrored = _scaled(dataset.clinical_mirrored, scaler)
        model = DenseClinicalSharedEncoder(use_dense=use_dense).to(runtime)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=3e-4, weight_decay=1e-3
        )
        train_sources = tuple(dataset.sources[index] for index in train)
        participant_weights = source_class_balanced_weights(
            dataset.labels[train], train_sources
        )
        weight_tensor = _tensor(participant_weights.astype(np.float32), runtime)
        label_tensor = _tensor(dataset.labels[train].astype(np.float32), runtime)
        task_tensor = _tensor(np.asarray(
            [_SOURCE_TASK_CODE[source] for source in train_sources],
            dtype=np.int64,
        ), runtime)
        train_inputs = _model_inputs(
            dataset, original, mirrored, train, device=runtime
        )

        if fold_index == 0:
            for source in SOURCES:
                local = np.asarray([
                    index for index, observed in enumerate(train_sources)
                    if observed == source
                ], dtype=np.int64)
                model.zero_grad(set_to_none=True)
                source_inputs = tuple(values[local] for values in train_inputs)
                source_embedding = model.encode(*source_inputs)
                source_loss = F.binary_cross_entropy_with_logits(
                    model.task_logits_from_embedding(
                        source_embedding, task_tensor[local]
                    ),
                    label_tensor[local],
                ) + UNIVERSAL_AUXILIARY_WEIGHT * F.binary_cross_entropy_with_logits(
                    model.universal_head(source_embedding).squeeze(-1),
                    label_tensor[local],
                )
                source_loss.backward()
                clinical_gradient = model.clinical_encoder[0].weight.grad
                patient_gradient = model.patient_projection.weight.grad
                if (
                    clinical_gradient is None
                    or patient_gradient is None
                    or float(clinical_gradient.norm()) <= 0.0
                    or float(patient_gradient.norm()) <= 0.0
                ):
                    raise RuntimeError("a source failed to update shared parameters")
                audited_sources.add(source)

        for _ in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            embedding = model.encode(*train_inputs)
            task_logits = model.task_logits_from_embedding(embedding, task_tensor)
            universal_logits = model.universal_head(embedding).squeeze(-1)
            losses = F.binary_cross_entropy_with_logits(
                task_logits, label_tensor, reduction="none"
            ) + UNIVERSAL_AUXILIARY_WEIGHT * F.binary_cross_entropy_with_logits(
                universal_logits, label_tensor, reduction="none"
            )
            loss = torch.sum(losses * weight_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            held_inputs = _model_inputs(
                dataset, original, mirrored, held, device=runtime
            )
            held_embedding = model.encode(*held_inputs)
            held_tasks = _tensor(np.asarray(
                [_SOURCE_TASK_CODE[dataset.sources[index]] for index in held],
                dtype=np.int64,
            ), runtime)
            probabilities[held] = torch.sigmoid(
                model.task_logits_from_embedding(held_embedding, held_tasks)
            ).cpu().numpy()
        del model, optimizer
    if not np.isfinite(probabilities).all() or audited_sources != set(SOURCES):
        raise RuntimeError("shared evaluation did not cover every participant and source")
    metrics = {}
    for source in SOURCES:
        selected = np.asarray([observed == source for observed in dataset.sources])
        metrics[source] = binary_metrics(
            dataset.labels[selected], probabilities[selected]
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


__all__ = [
    "SOURCES",
    "ClinicalScaler",
    "SharedClinicalDataset",
    "SharedEvaluation",
    "evaluate_shared_model",
    "fit_clinical_scaler",
    "participant_disjoint_folds",
    "source_class_balanced_weights",
]
