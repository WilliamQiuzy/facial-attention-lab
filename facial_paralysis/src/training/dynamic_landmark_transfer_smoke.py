"""Development-only inner-OOF smoke for focused Fusion SSL transfer.

This module deliberately exposes no outer-test prediction or refit path.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Optional

import numpy as np
import torch

from ..evaluation.nested_group_cv import NestedGroupFold, assert_outer_test_isolation
from ..models.dynamic_landmark import (
    ARM_FUSION,
    ARM_LANDMARK,
    DynamicLandmarkModel,
)
from ..pretraining.dynamic_landmark_ssl import DynamicLandmarkSSLModel
from .dynamic_landmark_benchmark import (
    BenchmarkConfig,
    _indices,
    _initialize_model,
    _subset,
    _train_fixed_epochs,
    _validate_labels,
    _validate_outer_partition,
    fit_fold_standardizer,
)


LANDMARK_RANDOM: Final[str] = "landmark_random"
FUSION_RANDOM: Final[str] = "fusion_random"
FUSION_SSL_WARMSTART: Final[str] = "fusion_ssl_warmstart"
DEVELOPMENT_CANDIDATES: Final[tuple[str, ...]] = (
    LANDMARK_RANDOM,
    FUSION_RANDOM,
    FUSION_SSL_WARMSTART,
)

TRANSFER_PREFIXES: Final[tuple[str, ...]] = (
    "proj_bs_x.",
    "proj_bs_dx.",
    "proj_lm_x.",
    "proj_lm_dx.",
    "temporal.",
    "attention_score.",
    "pool_projection.",
)


def _model_schema(model: torch.nn.Module) -> tuple[tuple[str, tuple[int, ...], torch.dtype], ...]:
    return tuple(
        (name, tuple(value.shape), value.dtype)
        for name, value in model.state_dict().items()
    )


with torch.random.fork_rng(devices=[]):
    _FOCUSED_SSL_SCHEMA = _model_schema(DynamicLandmarkSSLModel())
    _DOWNSTREAM_SCHEMA = _model_schema(DynamicLandmarkModel(ARM_FUSION))

_FOCUSED_SSL_KEYS: Final[tuple[str, ...]] = tuple(
    name for name, _shape, _dtype in _FOCUSED_SSL_SCHEMA
)
_TRANSFER_KEYS: Final[tuple[str, ...]] = tuple(sorted(
    name for name, _shape, _dtype in _DOWNSTREAM_SCHEMA
    if name.startswith(TRANSFER_PREFIXES)
))

if len(_FOCUSED_SSL_KEYS) != 22 or len(_TRANSFER_KEYS) != 16:
    raise RuntimeError("focused SSL transfer schema drifted")


@dataclass(frozen=True)
class DevelopmentInnerOOFResult:
    candidate: str
    seed: int
    epochs: int
    outer_train_indices: np.ndarray
    labels: np.ndarray
    probabilities: np.ndarray
    transferred_keys_by_fold: tuple[tuple[str, ...], ...]


def _validated_source_snapshot(
    source_state: object,
) -> OrderedDict[str, torch.Tensor]:
    if not isinstance(source_state, Mapping):
        raise ValueError("source_state must be a tensor mapping")
    keys = tuple(source_state.keys())
    if any(not isinstance(name, str) for name in keys):
        raise ValueError("source_state keys must be strings")
    if len(keys) != 22 or set(keys) != set(_FOCUSED_SSL_KEYS):
        raise ValueError("source_state must have the exact 22-tensor focused SSL schema")

    snapshot: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, expected_shape, expected_dtype in _FOCUSED_SSL_SCHEMA:
        value = source_state[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"source_state[{name!r}] must be a torch tensor")
        if value.layout != torch.strided:
            raise ValueError(f"source_state[{name!r}] must be a strided tensor")
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"source_state[{name!r}] has an incompatible shape")
        if value.dtype != expected_dtype or value.dtype != torch.float32:
            raise ValueError(f"source_state[{name!r}] must have float32 dtype")
        try:
            finite = bool(torch.isfinite(value).all().item())
        except (RuntimeError, TypeError) as exc:
            raise ValueError(
                f"source_state[{name!r}] finiteness could not be verified"
            ) from exc
        if not finite:
            raise ValueError(f"source_state[{name!r}] must be finite")
        snapshot[name] = value.detach().clone()
    return snapshot


def _validate_downstream(downstream: object) -> OrderedDict[str, torch.Tensor]:
    if type(downstream) is not DynamicLandmarkModel:
        raise ValueError("downstream must be an exact DynamicLandmarkModel")
    if downstream.arm != ARM_FUSION:
        raise ValueError("downstream must use the fusion arm")
    state = downstream.state_dict()
    if tuple(state) != tuple(name for name, _shape, _dtype in _DOWNSTREAM_SCHEMA):
        raise ValueError("downstream state schema is incompatible")
    for name, expected_shape, expected_dtype in _DOWNSTREAM_SCHEMA:
        value = state[name]
        if tuple(value.shape) != expected_shape or value.dtype != expected_dtype:
            raise ValueError(f"downstream state is incompatible at {name!r}")
    return OrderedDict(
        (name, value.detach().clone()) for name, value in state.items()
    )


def transfer_focused_fusion_encoder(
    source_state: Mapping[str, torch.Tensor],
    downstream: DynamicLandmarkModel,
) -> tuple[str, ...]:
    """Atomically copy the exact focused SSL encoder into a fresh Fusion model."""
    destination_snapshot = _validate_downstream(downstream)
    source_snapshot = _validated_source_snapshot(source_state)

    replacement: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, destination_value in destination_snapshot.items():
        if name in _TRANSFER_KEYS:
            replacement[name] = source_snapshot[name].to(
                device=destination_value.device,
                dtype=destination_value.dtype,
            ).clone()
        else:
            replacement[name] = destination_value

    # Every possible incompatibility has been checked before this sole mutation.
    downstream.load_state_dict(replacement, strict=True)
    return _TRANSFER_KEYS


def _validate_tensor_metadata(
    features: object,
    valid_mask: object,
    timestamps: object,
    source_frame_indices: object,
    labels: object,
) -> int:
    tensors = (features, valid_mask, timestamps, source_frame_indices, labels)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise ValueError("development inputs must be torch tensors")
    if features.ndim != 4 or features.shape[1:] != (4, 32, 95):
        raise ValueError("features must have shape (N, 4, 32, 95)")
    leading_shape = features.shape[:-1]
    if valid_mask.shape != leading_shape or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool with feature leading shape")
    if timestamps.shape != leading_shape or not timestamps.is_floating_point():
        raise ValueError("timestamps must be floating with feature leading shape")
    if source_frame_indices.shape != leading_shape or source_frame_indices.dtype != torch.int64:
        raise ValueError("source_frame_indices must be int64 with feature leading shape")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("labels must have one value per dataset row")
    if not features.is_floating_point():
        raise ValueError("features must have floating dtype")
    if len({value.device for value in tensors}) != 1:
        raise ValueError("development inputs must share one device")
    return int(features.shape[0])


def _validated_inner_partitions(
    fold: NestedGroupFold,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[np.ndarray, np.ndarray], ...]]:
    if not isinstance(fold, NestedGroupFold) or len(fold.inner_folds) != 4:
        raise ValueError("development smoke requires exactly four inner folds")
    outer_train = _indices(fold.train_indices, n_samples, "outer_train_indices")
    outer_test = _indices(fold.test_indices, n_samples, "outer_test_indices")
    _validate_outer_partition(outer_train, outer_test, n_samples)

    outer_train_set = set(outer_train.tolist())
    validation_counts = {index: 0 for index in outer_train.tolist()}
    partitions = []
    for inner in fold.inner_folds:
        train = _indices(inner.train_indices, n_samples, "inner_train_indices")
        validation = _indices(
            inner.validation_indices, n_samples, "inner_validation_indices"
        )
        train_set = set(train.tolist())
        validation_set = set(validation.tolist())
        if train_set.intersection(validation_set):
            raise ValueError("inner train and validation indices must be disjoint")
        if train_set.union(validation_set) != outer_train_set:
            raise ValueError("each inner fold must partition outer train exactly")
        for index in validation.tolist():
            validation_counts[index] += 1
        assert_outer_test_isolation(
            outer_test,
            train_indices=train,
            validation_indices=validation,
            scaler_fit_indices=train,
            selection_indices=validation,
        )
        partitions.append((train, validation))
    if any(count != 1 for count in validation_counts.values()):
        raise ValueError("inner validation folds must cover outer train exactly once")
    return outer_train, outer_test, tuple(partitions)


def run_development_inner_oof(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    timestamps: torch.Tensor,
    source_frame_indices: torch.Tensor,
    labels: torch.Tensor,
    *,
    fold: NestedGroupFold,
    candidate: str,
    seed: int,
    epochs: int = 12,
    config: Optional[BenchmarkConfig] = None,
    source_state: Optional[Mapping[str, torch.Tensor]] = None,
) -> DevelopmentInnerOOFResult:
    """Collect fixed-budget inner OOF probabilities without touching outer test."""
    if candidate not in DEVELOPMENT_CANDIDATES:
        raise ValueError("candidate is not registered for the development smoke")
    if candidate == FUSION_SSL_WARMSTART:
        if source_state is None:
            raise ValueError("fusion_ssl_warmstart requires source_state")
        validated_source = _validated_source_snapshot(source_state)
    else:
        if source_state is not None:
            raise ValueError("random-init candidates forbid source_state")
        validated_source = None
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
    ):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    if (
        isinstance(epochs, (bool, np.bool_))
        or not isinstance(epochs, (int, np.integer))
        or int(epochs) < 1
    ):
        raise ValueError("epochs must be a positive integer")
    epochs = int(epochs)
    if config is None:
        config = BenchmarkConfig()
    if not isinstance(config, BenchmarkConfig):
        raise ValueError("config must be a BenchmarkConfig")

    n_samples = _validate_tensor_metadata(
        features, valid_mask, timestamps, source_frame_indices, labels
    )
    outer_train, outer_test, partitions = _validated_inner_partitions(
        fold, n_samples
    )
    outer_labels = _validate_labels(labels, outer_train).detach().cpu().numpy()
    outer_labels = outer_labels.astype(np.int64, copy=True)
    positions = {index: position for position, index in enumerate(outer_train.tolist())}
    probabilities = np.full(outer_train.size, np.nan, dtype=np.float64)
    transfer_audit = []
    arm = ARM_LANDMARK if candidate == LANDMARK_RANDOM else ARM_FUSION

    for train, validation in partitions:
        scaler = fit_fold_standardizer(
            features,
            valid_mask,
            fit_indices=train,
            outer_test_indices=outer_test,
        )
        model = _initialize_model(arm, seed)
        if validated_source is None:
            transferred = ()
        else:
            transferred = transfer_focused_fusion_encoder(validated_source, model)
        model = model.to(features.device)
        _train_fixed_epochs(
            model,
            features,
            valid_mask,
            timestamps,
            source_frame_indices,
            labels,
            train_indices=train,
            scaler=scaler,
            epochs=epochs,
            config=config,
            seed=seed,
            validation_indices=None,
        )

        model.eval()
        with torch.no_grad():
            validation_mask = _subset(valid_mask, validation)
            validation_features = scaler.transform(
                _subset(features, validation), validation_mask
            )
            logits = model(
                validation_features,
                validation_mask,
                _subset(timestamps, validation),
                _subset(source_frame_indices, validation),
            )
            fold_probabilities = torch.sigmoid(logits).detach().cpu().numpy()
        if fold_probabilities.shape != (validation.size,):
            raise RuntimeError("inner validation prediction shape is invalid")
        if not np.isfinite(fold_probabilities).all() or not np.logical_and(
            fold_probabilities >= 0.0, fold_probabilities <= 1.0
        ).all():
            raise RuntimeError("inner validation probabilities are invalid")
        for index, probability in zip(validation.tolist(), fold_probabilities.tolist()):
            position = positions[index]
            if math.isfinite(float(probabilities[position])):
                raise RuntimeError("outer-train row received duplicate OOF predictions")
            probabilities[position] = float(probability)
        transfer_audit.append(transferred)

    if not np.isfinite(probabilities).all():
        raise RuntimeError("outer-train OOF predictions are incomplete")
    return DevelopmentInnerOOFResult(
        candidate=candidate,
        seed=seed,
        epochs=epochs,
        outer_train_indices=outer_train.copy(),
        labels=outer_labels,
        probabilities=probabilities,
        transferred_keys_by_fold=tuple(transfer_audit),
    )


__all__ = [
    "LANDMARK_RANDOM",
    "FUSION_RANDOM",
    "FUSION_SSL_WARMSTART",
    "DEVELOPMENT_CANDIDATES",
    "DevelopmentInnerOOFResult",
    "transfer_focused_fusion_encoder",
    "run_development_inner_oof",
]
