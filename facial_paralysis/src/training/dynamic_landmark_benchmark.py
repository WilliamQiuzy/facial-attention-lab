"""Leak-safe training primitives for the locked dynamic neural candidates.

This module may select epochs with inner validation folds and may refit a fresh
model on an outer-training fold.  It deliberately contains no outer-test
prediction path; Task 7 must register every candidate before that one-shot run.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from ..evaluation.nested_group_cv import NestedGroupFold, assert_outer_test_isolation
from ..models.dynamic_landmark import (
    ARM_BLENDSHAPE,
    ARM_FUSION,
    ARM_LANDMARK,
    DYNAMIC_NEURAL_ARMS,
    DynamicLandmarkModel,
    horizontal_mirror_features,
)


RANDOM_INIT_SEEDS: tuple[int, ...] = (0, 1, 2)
HISTORICAL_POOLED_AUC_REFERENCE = 0.860
HISTORICAL_CONTEXT = "Historical 0.860 pooled AUC is a non-nested reference only."


class OuterEvaluationLockedError(RuntimeError):
    """Task 7 has not authorized any real outer-test prediction."""


@dataclass(frozen=True)
class BenchmarkConfig:
    max_epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    mirror_probability: float = 0.5

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_epochs, (bool, np.bool_))
            or not isinstance(self.max_epochs, (int, np.integer))
            or self.max_epochs < 1
        ):
            raise ValueError("max_epochs must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not math.isfinite(self.mirror_probability) or not 0 <= self.mirror_probability <= 1:
            raise ValueError("mirror_probability must lie within [0, 1]")


@dataclass(frozen=True)
class CandidateSpec:
    arm: str
    seeds: tuple[int, ...]
    max_epochs: int
    historical_context: str


@dataclass(frozen=True)
class FoldStandardizer:
    mean: torch.Tensor
    scale: torch.Tensor
    fit_indices: tuple[int, ...]

    def transform(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1:] != (4, 32, 95):
            raise ValueError("standardizer features must have shape (N, 4, 32, 95)")
        if valid_mask.shape != features.shape[:-1] or valid_mask.dtype != torch.bool:
            raise ValueError("standardizer mask must be bool with feature leading shape")
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError("standardizer input features must be finite floating values")
        mean = self.mean.to(device=features.device, dtype=features.dtype)
        scale = self.scale.to(device=features.device, dtype=features.dtype)
        transformed = (features - mean) / scale
        return torch.where(
            valid_mask.unsqueeze(-1), transformed, torch.zeros_like(transformed)
        )


@dataclass(frozen=True)
class TrainingTrace:
    epochs_ran: int
    best_epoch: int
    validation_losses: tuple[float, ...]


@dataclass(frozen=True)
class InnerEpochSelection:
    selected_epoch: int
    best_epochs: tuple[int, ...]
    traces: tuple[TrainingTrace, ...]


@dataclass(frozen=True)
class OuterRefitArtifact:
    model: DynamicLandmarkModel
    scaler: FoldStandardizer
    arm: str
    seed: int
    epochs_trained: int
    train_indices: tuple[int, ...]


def build_candidate_registry(
    config: BenchmarkConfig | None = None,
) -> Mapping[str, CandidateSpec]:
    config = config or BenchmarkConfig()
    return {
        arm: CandidateSpec(
            arm=arm,
            seeds=RANDOM_INIT_SEEDS,
            max_epochs=config.max_epochs,
            historical_context=HISTORICAL_CONTEXT,
        )
        for arm in (ARM_BLENDSHAPE, ARM_LANDMARK, ARM_FUSION)
    }


def _indices(values: Sequence[int] | np.ndarray, n_samples: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must be a one-dimensional integer array")
    integers = np.asarray([int(value) for value in array.tolist()], dtype=np.int64)
    if np.unique(integers).size != integers.size:
        raise ValueError(f"{name} cannot contain duplicates")
    if integers.size == 0 or np.any(integers < 0) or np.any(integers >= n_samples):
        raise ValueError(f"{name} must be nonempty and within the dataset")
    return integers


def _subset(tensor: torch.Tensor, indices: np.ndarray) -> torch.Tensor:
    return tensor.index_select(0, torch.as_tensor(indices, dtype=torch.int64, device=tensor.device))


def _validate_outer_partition(
    train: np.ndarray,
    test: np.ndarray,
    n_samples: int,
) -> None:
    train_set = set(train.tolist())
    test_set = set(test.tolist())
    if train_set.intersection(test_set):
        raise ValueError("outer train and test indices must be disjoint")
    if train_set.union(test_set) != set(range(n_samples)):
        raise ValueError("outer train and test must cover every dataset row exactly once")


def fit_fold_standardizer(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    fit_indices: Sequence[int] | np.ndarray,
    outer_test_indices: Sequence[int] | np.ndarray,
) -> FoldStandardizer:
    """Fit 95 independent columns using valid frames from one train fold only."""
    if features.ndim != 4 or features.shape[1:] != (4, 32, 95):
        raise ValueError("features must have shape (N, 4, 32, 95)")
    if valid_mask.shape != features.shape[:-1] or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool with feature leading shape")
    fit = _indices(fit_indices, features.shape[0], "fit_indices")
    outer = _indices(outer_test_indices, features.shape[0], "outer_test_indices")
    assert_outer_test_isolation(outer, scaler_fit_indices=fit)
    fit_features = _subset(features, fit)
    fit_mask = _subset(valid_mask, fit)
    if not fit_features.is_floating_point() or not torch.isfinite(fit_features).all():
        raise ValueError("scaler fit rows must be finite floating values")
    rows = fit_features[fit_mask]
    if rows.shape[0] < 2:
        raise ValueError("scaler fit requires at least two valid frames")
    mean = rows.mean(dim=0)
    scale = rows.std(dim=0, unbiased=False)
    scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
    return FoldStandardizer(
        mean=mean.detach().cpu(),
        scale=scale.detach().cpu(),
        fit_indices=tuple(int(index) for index in fit.tolist()),
    )


def _validate_labels(labels: torch.Tensor, indices: np.ndarray) -> torch.Tensor:
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    selected = _subset(labels, indices).to(torch.float32)
    if not torch.isfinite(selected).all() or not bool(((selected == 0) | (selected == 1)).all()):
        raise ValueError("selected labels must be finite binary values")
    return selected


def _initialize_model(arm: str, seed: int) -> DynamicLandmarkModel:
    if arm not in DYNAMIC_NEURAL_ARMS or seed not in RANDOM_INIT_SEEDS:
        raise ValueError("arm and seed must be registered before training")
    torch.manual_seed(seed)
    return DynamicLandmarkModel(arm)


def _augment_raw(
    features: torch.Tensor,
    probability: float,
    seed: int,
    epoch: int,
) -> torch.Tensor:
    if probability == 0:
        return features
    generator = torch.Generator(device="cpu").manual_seed(seed * 10_000 + epoch)
    decisions = torch.rand(features.shape[0], generator=generator) < probability
    if not bool(decisions.any()):
        return features
    result = features.clone()
    device_decisions = decisions.to(features.device)
    result[device_decisions] = horizontal_mirror_features(result[device_decisions])
    return result


def _train_fixed_epochs(
    model: DynamicLandmarkModel,
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    timestamps: torch.Tensor,
    source_frame_indices: torch.Tensor,
    labels: torch.Tensor,
    *,
    train_indices: np.ndarray,
    scaler: FoldStandardizer,
    epochs: int,
    config: BenchmarkConfig,
    seed: int,
    validation_indices: np.ndarray | None,
) -> TrainingTrace:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    train_raw = _subset(features, train_indices)
    train_mask = _subset(valid_mask, train_indices)
    train_times = _subset(timestamps, train_indices)
    train_sources = _subset(source_frame_indices, train_indices)
    train_labels = _validate_labels(labels, train_indices)
    validation_losses: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        augmented = _augment_raw(
            train_raw, config.mirror_probability, seed=seed, epoch=epoch
        )
        standardized = scaler.transform(augmented, train_mask)
        optimizer.zero_grad(set_to_none=True)
        logits = model(standardized, train_mask, train_times, train_sources)
        loss = criterion(logits, train_labels.to(logits.device))
        if not torch.isfinite(loss):
            raise RuntimeError("training loss became nonfinite")
        loss.backward()
        optimizer.step()

        if validation_indices is not None:
            model.eval()
            with torch.no_grad():
                val_mask = _subset(valid_mask, validation_indices)
                val_features = scaler.transform(
                    _subset(features, validation_indices), val_mask
                )
                val_logits = model(
                    val_features,
                    val_mask,
                    _subset(timestamps, validation_indices),
                    _subset(source_frame_indices, validation_indices),
                )
                val_labels = _validate_labels(labels, validation_indices).to(val_logits.device)
                value = float(criterion(val_logits, val_labels).item())
            if not math.isfinite(value):
                raise RuntimeError("validation loss became nonfinite")
            validation_losses.append(value)

    if validation_indices is None:
        return TrainingTrace(epochs_ran=epochs, best_epoch=epochs, validation_losses=())
    best_epoch = min(range(1, epochs + 1), key=lambda index: validation_losses[index - 1])
    return TrainingTrace(
        epochs_ran=epochs,
        best_epoch=best_epoch,
        validation_losses=tuple(validation_losses),
    )


def _median_epoch(best_epochs: Sequence[int], max_epochs: int) -> int:
    if len(best_epochs) != 4 or any(
        isinstance(value, bool) or not 1 <= int(value) <= max_epochs
        for value in best_epochs
    ):
        raise ValueError("epoch selection requires four valid inner best epochs")
    median = float(np.median(np.asarray(best_epochs, dtype=np.float64)))
    # Explicit half-up conversion avoids platform/banker's-rounding ambiguity.
    return int(math.floor(median + 0.5))


def select_inner_epoch(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    timestamps: torch.Tensor,
    source_frame_indices: torch.Tensor,
    labels: torch.Tensor,
    *,
    fold: NestedGroupFold,
    arm: str,
    seed: int,
    config: BenchmarkConfig | None = None,
) -> InnerEpochSelection:
    """Run all four fixed-budget inner fits and choose their median best epoch."""
    config = config or BenchmarkConfig()
    if len(fold.inner_folds) != 4:
        raise ValueError("locked neural selection requires exactly four inner folds")
    outer_train = _indices(fold.train_indices, features.shape[0], "outer_train_indices")
    outer_test = _indices(fold.test_indices, features.shape[0], "outer_test_indices")
    _validate_outer_partition(outer_train, outer_test, features.shape[0])
    outer_train_set = set(outer_train.tolist())
    validated_inner: list[tuple[np.ndarray, np.ndarray]] = []
    validation_counts = {index: 0 for index in outer_train.tolist()}
    validation_sets: set[frozenset[int]] = set()
    for inner in fold.inner_folds:
        train = _indices(inner.train_indices, features.shape[0], "inner_train_indices")
        validation = _indices(
            inner.validation_indices, features.shape[0], "inner_validation_indices"
        )
        if set(train.tolist()).intersection(validation.tolist()) or set(train.tolist()).union(
            validation.tolist()
        ) != outer_train_set:
            raise ValueError("each inner split must partition the outer train exactly")
        held_out = frozenset(validation.tolist())
        if held_out in validation_sets:
            raise ValueError("inner validation sets must be distinct")
        validation_sets.add(held_out)
        for index in validation.tolist():
            validation_counts[index] += 1
        assert_outer_test_isolation(
            outer_test,
            train_indices=train,
            validation_indices=validation,
            scaler_fit_indices=train,
            selection_indices=validation,
        )
        validated_inner.append((train, validation))
    if any(count != 1 for count in validation_counts.values()):
        raise ValueError(
            "four inner validation sets must cover each outer-train row exactly once"
        )

    traces: list[TrainingTrace] = []
    for train, validation in validated_inner:
        scaler = fit_fold_standardizer(
            features, valid_mask, fit_indices=train, outer_test_indices=outer_test
        )
        model = _initialize_model(arm, seed).to(features.device)
        traces.append(_train_fixed_epochs(
            model, features, valid_mask, timestamps, source_frame_indices, labels,
            train_indices=train,
            scaler=scaler,
            epochs=config.max_epochs,
            config=config,
            seed=seed,
            validation_indices=validation,
        ))
    best_epochs = tuple(trace.best_epoch for trace in traces)
    return InnerEpochSelection(
        selected_epoch=_median_epoch(best_epochs, config.max_epochs),
        best_epochs=best_epochs,
        traces=tuple(traces),
    )


def refit_outer_train(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    timestamps: torch.Tensor,
    source_frame_indices: torch.Tensor,
    labels: torch.Tensor,
    *,
    outer_train_indices: Sequence[int] | np.ndarray,
    outer_test_indices: Sequence[int] | np.ndarray,
    arm: str,
    seed: int,
    selected_epoch: int,
    config: BenchmarkConfig | None = None,
) -> OuterRefitArtifact:
    """Reinitialize and fit outer-train only; there is no validation argument."""
    config = config or BenchmarkConfig()
    if (
        isinstance(selected_epoch, (bool, np.bool_))
        or not isinstance(selected_epoch, (int, np.integer))
        or not 1 <= selected_epoch <= config.max_epochs
    ):
        raise ValueError("selected_epoch must come from the fixed inner budget")
    selected_epoch = int(selected_epoch)
    train = _indices(outer_train_indices, features.shape[0], "outer_train_indices")
    test = _indices(outer_test_indices, features.shape[0], "outer_test_indices")
    _validate_outer_partition(train, test, features.shape[0])
    assert_outer_test_isolation(
        test, train_indices=train, scaler_fit_indices=train
    )
    scaler = fit_fold_standardizer(
        features, valid_mask, fit_indices=train, outer_test_indices=test
    )
    model = _initialize_model(arm, seed).to(features.device)
    trace = _train_fixed_epochs(
        model, features, valid_mask, timestamps, source_frame_indices, labels,
        train_indices=train,
        scaler=scaler,
        epochs=selected_epoch,
        config=config,
        seed=seed,
        validation_indices=None,
    )
    return OuterRefitArtifact(
        model=model,
        scaler=scaler,
        arm=arm,
        seed=seed,
        epochs_trained=trace.epochs_ran,
        train_indices=tuple(int(index) for index in train.tolist()),
    )


def ensemble_seed_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Average predictions only after sigmoid, across the exact three seeds."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(RANDOM_INIT_SEEDS) or values.shape[1] == 0:
        raise ValueError("probabilities must have shape (3, n_samples)")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("seed inputs must be finite probabilities in [0, 1]")
    return values.mean(axis=0)


def require_frozen_outer_registry(_registry_hash: str | None = None) -> None:
    """Unconditionally refuse real outer work until Task 7 lands atomically."""
    raise OuterEvaluationLockedError(
        "real outer evaluation is disabled until Task 7 freezes the complete "
        "all-candidate registry and one-shot execution protocol"
    )


__all__ = [
    "BenchmarkConfig", "CandidateSpec", "FoldStandardizer", "TrainingTrace",
    "InnerEpochSelection", "OuterRefitArtifact", "OuterEvaluationLockedError",
    "RANDOM_INIT_SEEDS", "HISTORICAL_POOLED_AUC_REFERENCE",
    "build_candidate_registry", "fit_fold_standardizer", "select_inner_epoch",
    "refit_outer_train", "ensemble_seed_probabilities",
    "require_frozen_outer_registry",
]
