"""Fixed participant-disjoint training protocol for the NeuroFace temporal TCN."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np
import torch

from ..models.dynamic_landmark import horizontal_mirror_features
from ..models.neuroface_als_temporal_v1 import (
    TaskAwareTemporalALSClassifier,
    mirror_mean_probability,
    participant_balanced_bce,
)
from .neuroface_als_benchmark_v1 import recompute_binary_metrics


FROZEN_SEEDS: Final[tuple[int, ...]] = (17, 43, 79)
FROZEN_EPOCHS: Final[int] = 200
FROZEN_LEARNING_RATE: Final[float] = 5e-4
FROZEN_WEIGHT_DECAY: Final[float] = 1e-2
FROZEN_GRADIENT_NORM: Final[float] = 1.0
_GROUP_ID = re.compile(r"^grp_[0-9a-f]{64}$")


def _frozen(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    result = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype)
    return result.reshape(contiguous.shape)


@dataclass(frozen=True)
class TemporalDataset:
    features: np.ndarray
    valid_mask: np.ndarray
    timestamps: np.ndarray
    labels: np.ndarray
    group_ids: tuple[str, ...]


@dataclass(frozen=True)
class FeatureScaler:
    mean: np.ndarray
    scale: np.ndarray
    valid_rows: int


@dataclass(frozen=True)
class TemporalEvaluation:
    probabilities: np.ndarray
    seed_probabilities: np.ndarray
    metrics: dict[str, float]
    group_count: int
    protocol: str


def validate_temporal_dataset(
    features: np.ndarray,
    valid_mask: np.ndarray,
    timestamps: np.ndarray,
    labels: np.ndarray,
    group_ids: Sequence[str],
) -> TemporalDataset:
    """Validate exactly one three-task tensor and one outcome per participant."""
    features = np.asarray(features)
    valid_mask = np.asarray(valid_mask)
    timestamps = np.asarray(timestamps)
    labels = np.asarray(labels)
    groups = tuple(group_ids)
    if (features.ndim != 5 or features.shape[1:] != (3, 4, 32, 95)
            or features.dtype != np.dtype(np.float32)):
        raise ValueError("features must be float32 participant-by-3-by-4-by-32-by-95")
    row_shape = features.shape[:-1]
    if valid_mask.shape != row_shape or valid_mask.dtype != np.dtype(bool):
        raise ValueError("valid_mask must be bool and match all feature rows")
    if (timestamps.shape != row_shape
            or timestamps.dtype.kind not in {"f", "i", "u"}):
        raise ValueError("timestamps must be real numeric values matching feature rows")
    if (not np.isfinite(features).all() or not np.isfinite(timestamps).all()
            or not np.all(timestamps[..., 1:] > timestamps[..., :-1])):
        raise ValueError("features must be finite and window timestamps strictly ordered")
    if np.any(features[~valid_mask] != 0) or np.any(~valid_mask.any(axis=-1)):
        raise ValueError("invalid feature rows must be zero and every window must be present")
    count = features.shape[0]
    if (labels.shape != (count,) or labels.dtype.kind not in {"i", "u"}
            or not np.isin(labels, (0, 1)).all() or len(np.unique(labels)) != 2):
        raise ValueError("labels must contain both binary classes once per participant")
    if (len(groups) != count or len(set(groups)) != count
            or any(not isinstance(group, str) or _GROUP_ID.fullmatch(group) is None
                   for group in groups)):
        raise ValueError("group_ids must be unique opaque participant identifiers")
    return TemporalDataset(
        features=_frozen(features),
        valid_mask=_frozen(valid_mask),
        timestamps=_frozen(timestamps.astype(np.float32, copy=False)),
        labels=_frozen(labels.astype(np.int64, copy=False)),
        group_ids=groups,
    )


def fit_masked_feature_scaler(
    train_features: np.ndarray,
    train_mask: np.ndarray,
) -> FeatureScaler:
    """Fit one raw-channel scaler from training participants and valid rows only."""
    train_features = np.asarray(train_features)
    train_mask = np.asarray(train_mask)
    if (train_features.ndim != 5 or train_features.shape[1:] != (3, 4, 32, 95)
            or train_features.dtype != np.dtype(np.float32)
            or train_mask.shape != train_features.shape[:-1]
            or train_mask.dtype != np.dtype(bool)
            or train_features.shape[0] < 2):
        raise ValueError("scaler requires at least two valid participant tensors")
    if not np.isfinite(train_features).all() or np.any(train_features[~train_mask] != 0):
        raise ValueError("scaler input must be finite with canonical-zero invalid rows")
    observed = train_features[train_mask].astype(np.float64, copy=False)
    if observed.shape[0] < 2:
        raise ValueError("scaler requires at least two valid training rows")
    mean = observed.mean(axis=0)
    scale = observed.std(axis=0, ddof=0)
    scale[scale == 0.0] = 1.0
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("training scaler statistics are invalid")
    return FeatureScaler(
        mean=_frozen(mean.astype(np.float32)),
        scale=_frozen(scale.astype(np.float32)),
        valid_rows=int(observed.shape[0]),
    )


def apply_feature_scaling(
    features: np.ndarray,
    valid_mask: np.ndarray,
    scaler: FeatureScaler,
) -> np.ndarray:
    """Apply a training-only scaler and preserve canonical missing rows."""
    features = np.asarray(features)
    valid_mask = np.asarray(valid_mask)
    if (not isinstance(scaler, FeatureScaler)
            or features.ndim != 5 or features.shape[1:] != (3, 4, 32, 95)
            or features.dtype != np.dtype(np.float32)
            or valid_mask.shape != features.shape[:-1]
            or valid_mask.dtype != np.dtype(bool)
            or scaler.mean.shape != (95,) or scaler.scale.shape != (95,)):
        raise ValueError("features, mask, or scaler differs from the frozen schema")
    if not np.isfinite(features).all() or np.any(features[~valid_mask] != 0):
        raise ValueError("scaling input must be finite with canonical-zero invalid rows")
    scaled = (features - scaler.mean) / scaler.scale
    scaled = np.where(valid_mask[..., None], scaled, 0.0).astype(np.float32)
    if not np.isfinite(scaled).all():
        raise ValueError("scaled features contain nonfinite values")
    return scaled


def participant_loso_splits(
    labels: np.ndarray,
    group_ids: Sequence[str],
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Create deterministic outer leave-one-participant-out splits."""
    labels = np.asarray(labels)
    groups = tuple(group_ids)
    count = len(groups)
    if (labels.shape != (count,) or labels.dtype.kind not in {"i", "u"}
            or not np.isin(labels, (0, 1)).all() or len(np.unique(labels)) != 2
            or len(set(groups)) != count
            or any(_GROUP_ID.fullmatch(group) is None for group in groups)):
        raise ValueError("LOSO requires unique opaque groups and both binary classes")
    indices = np.arange(count, dtype=np.int64)
    splits = []
    for held in range(count):
        train = indices[indices != held]
        if len(np.unique(labels[train])) != 2:
            raise ValueError("every outer training fold must retain both classes")
        splits.append((_frozen(train), _frozen(np.asarray([held], dtype=np.int64))))
    return tuple(splits)


def _configure_reproducibility(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _fit_seed_fold(
    dataset: TemporalDataset,
    train_indices: np.ndarray,
    held_index: int,
    *,
    seed: int,
    device: torch.device,
) -> float:
    scaler = fit_masked_feature_scaler(
        dataset.features[train_indices], dataset.valid_mask[train_indices]
    )
    train_features = torch.from_numpy(apply_feature_scaling(
        dataset.features[train_indices], dataset.valid_mask[train_indices], scaler
    )).to(device)
    train_mask = torch.from_numpy(
        np.array(dataset.valid_mask[train_indices], copy=True)
    ).to(device)
    train_timestamps = torch.from_numpy(
        np.array(dataset.timestamps[train_indices], copy=True)
    ).to(device)
    train_labels = torch.from_numpy(
        np.array(dataset.labels[train_indices], dtype=np.float32, copy=True)
    ).to(device)
    held_features = torch.from_numpy(apply_feature_scaling(
        dataset.features[[held_index]], dataset.valid_mask[[held_index]], scaler
    )).to(device)
    held_mask = torch.from_numpy(
        np.array(dataset.valid_mask[[held_index]], copy=True)
    ).to(device)
    held_timestamps = torch.from_numpy(
        np.array(dataset.timestamps[[held_index]], copy=True)
    ).to(device)

    _configure_reproducibility(seed, device)
    model = TaskAwareTemporalALSClassifier().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=FROZEN_LEARNING_RATE,
        weight_decay=FROZEN_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FROZEN_EPOCHS,
    )
    mirrored_train = horizontal_mirror_features(train_features)
    for _ in range(FROZEN_EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        original_logits = model(train_features, train_mask, train_timestamps)
        mirror_logits = model(mirrored_train, train_mask, train_timestamps)
        loss = 0.5 * (
            participant_balanced_bce(original_logits, train_labels)
            + participant_balanced_bce(mirror_logits, train_labels)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), FROZEN_GRADIENT_NORM)
        optimizer.step()
        scheduler.step()
    model.eval()
    with torch.inference_mode():
        probability = mirror_mean_probability(
            model, held_features, held_mask, held_timestamps
        )[0]
    value = float(probability.detach().cpu())
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise RuntimeError("temporal model emitted an invalid probability")
    return value


def evaluate_temporal_loso(
    features: np.ndarray,
    valid_mask: np.ndarray,
    timestamps: np.ndarray,
    labels: np.ndarray,
    group_ids: Sequence[str],
    *,
    device: torch.device,
) -> TemporalEvaluation:
    """Run the single frozen three-seed TCN in participant-level outer LOSO."""
    if not isinstance(device, torch.device) or device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be an explicit CPU or CUDA torch device")
    dataset = validate_temporal_dataset(
        features, valid_mask, timestamps, labels, group_ids
    )
    splits = participant_loso_splits(dataset.labels, dataset.group_ids)
    seed_probabilities = np.empty(
        (len(FROZEN_SEEDS), len(dataset.group_ids)), dtype=np.float64
    )
    for seed_index, seed in enumerate(FROZEN_SEEDS):
        for train_indices, held_indices in splits:
            held = int(held_indices[0])
            seed_probabilities[seed_index, held] = _fit_seed_fold(
                dataset, train_indices, held, seed=seed, device=device
            )
    probabilities = seed_probabilities.mean(axis=0)
    return TemporalEvaluation(
        probabilities=_frozen(probabilities),
        seed_probabilities=_frozen(seed_probabilities),
        metrics=recompute_binary_metrics(dataset.labels, probabilities),
        group_count=len(dataset.group_ids),
        protocol=(
            "fixed_three_seed_task_aware_tcn_participant_loso_"
            "train_fold_scaling_no_outer_early_stopping"
        ),
    )


__all__ = (
    "FROZEN_EPOCHS",
    "FROZEN_SEEDS",
    "FeatureScaler",
    "TemporalDataset",
    "TemporalEvaluation",
    "apply_feature_scaling",
    "evaluate_temporal_loso",
    "fit_masked_feature_scaler",
    "participant_loso_splits",
    "validate_temporal_dataset",
)
