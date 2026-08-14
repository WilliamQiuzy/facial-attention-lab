"""Locked exploratory NeuroFace motion pretraining and PalsyNet transfer policy."""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.models.neuroface_motion_pretrain_v1 import (
    DOMAIN_COUNT,
    EMBEDDING_DIM,
    MotionQualityRegressor,
    count_parameters,
)
from src.training.architecture_search_v1 import (
    OuterSearchLockedError,
    group_balanced_weights,
    require_development_only,
)


DOMAINS = ("symmetry", "rom", "speed", "variability", "fatigue")
MODEL_ORDER = ("landmark_110d", "motion_32d", "landmark_110d_plus_motion_32d")
COHORTS = ("als", "healthy_control", "post_stroke")


@dataclass(frozen=True)
class MotionPretrainConfig:
    folds: int = 6
    seed: int = 20260813
    epochs: int = 80
    patience: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    def __post_init__(self) -> None:
        if (self.folds, self.seed, self.epochs, self.patience) != (6, 20260813, 80, 10):
            raise ValueError("motion pretraining fold, seed, and budget are frozen")
        if self.learning_rate != 1e-3 or self.weight_decay != 1e-4:
            raise ValueError("motion pretraining optimizer settings are frozen")


@dataclass(frozen=True)
class MotionDataset:
    landmarks: np.ndarray
    mirrored_landmarks: np.ndarray
    valid_masks: np.ndarray
    timestamps: np.ndarray
    targets: np.ndarray
    task_indices: np.ndarray
    group_ids: np.ndarray
    cohorts: np.ndarray


@dataclass(frozen=True)
class MotionPretrainResult:
    metrics: Mapping[str, object]
    final_model: MotionQualityRegressor
    landmark_mean: np.ndarray
    landmark_scale: np.ndarray
    parameter_count: int


@dataclass(frozen=True)
class TransferDataset:
    summary_features: np.ndarray
    mirrored_summary_features: np.ndarray
    motion_features: np.ndarray
    mirrored_motion_features: np.ndarray
    labels: np.ndarray
    group_ids: np.ndarray
    development_indices: np.ndarray
    protected_indices: np.ndarray
    inner_fold_by_index: np.ndarray


@dataclass(frozen=True)
class TransferResult:
    metrics: Mapping[str, Mapping[str, float]]
    development_recordings: int
    development_groups: int
    protected_predictions: int


def build_stratified_participant_folds(
    group_ids: Sequence[object] | np.ndarray,
    cohorts: Sequence[object] | np.ndarray,
    *,
    folds: int,
    seed: int,
) -> np.ndarray:
    groups = np.asarray(group_ids, dtype=object)
    cohort_array = np.asarray(cohorts, dtype=object)
    if groups.ndim != 1 or cohort_array.shape != groups.shape or groups.size == 0:
        raise ValueError("participant fold inputs must be aligned one-dimensional arrays")
    if folds != 6 or seed != 20260813:
        raise ValueError("NeuroFace participant folds and seed are frozen")
    group_cohort: dict[str, str] = {}
    for raw_group, raw_cohort in zip(groups.tolist(), cohort_array.tolist()):
        group, cohort = str(raw_group), str(raw_cohort)
        if not group or cohort not in COHORTS:
            raise ValueError("participant group or cohort is invalid")
        previous = group_cohort.setdefault(group, cohort)
        if previous != cohort:
            raise ValueError("one participant cannot cross cohorts")
    by_cohort = {
        cohort: np.asarray(sorted(
            group for group, value in group_cohort.items() if value == cohort
        ), dtype=object)
        for cohort in COHORTS
    }
    if any(values.size < folds for values in by_cohort.values()):
        raise ValueError("every cohort needs at least one participant in each fold")
    rng = np.random.default_rng(seed)
    fold_by_group: dict[str, int] = {}
    for cohort in COHORTS:
        shuffled = rng.permutation(by_cohort[cohort])
        for fold, chunk in enumerate(np.array_split(shuffled, folds)):
            for group in chunk.tolist():
                fold_by_group[str(group)] = fold
    assignments = np.asarray(
        [fold_by_group[str(group)] for group in groups.tolist()], dtype=np.int64
    )
    for fold in range(folds):
        if (
            set(cohort_array[assignments == fold].tolist()) != set(COHORTS)
            or set(cohort_array[assignments != fold].tolist()) != set(COHORTS)
        ):
            raise AssertionError("cohort-stratified fold lost one cohort")
        held_groups = set(groups[assignments == fold].tolist())
        train_groups = set(groups[assignments != fold].tolist())
        if held_groups & train_groups:
            raise AssertionError("participant crossed a pretraining fold")
    return assignments


def _validate_motion_dataset(dataset: MotionDataset) -> None:
    n = int(np.asarray(dataset.targets).shape[0])
    expected_sequence = (n, 4, 32, 23)
    if (
        np.asarray(dataset.landmarks).shape != expected_sequence
        or np.asarray(dataset.mirrored_landmarks).shape != expected_sequence
        or np.asarray(dataset.valid_masks).shape != (n, 4, 32)
        or np.asarray(dataset.timestamps).shape != (n, 4, 32)
        or np.asarray(dataset.targets).shape != (n, DOMAIN_COUNT)
        or np.asarray(dataset.task_indices).shape != (n,)
        or np.asarray(dataset.group_ids).shape != (n,)
        or np.asarray(dataset.cohorts).shape != (n,)
    ):
        raise ValueError("motion pretraining arrays are misaligned")
    masks = np.asarray(dataset.valid_masks, dtype=bool)
    if n == 0 or np.any(~masks.reshape(n, -1).any(axis=1)):
        raise ValueError("each motion recording requires a valid frame")
    for values in (dataset.landmarks, dataset.mirrored_landmarks):
        if not np.isfinite(np.asarray(values)[masks]).all():
            raise ValueError("valid motion values must be finite")
    if not np.isfinite(np.asarray(dataset.timestamps)[masks]).all():
        raise ValueError("valid timestamps must be finite")
    targets = np.asarray(dataset.targets, dtype=np.float64)
    if not np.isfinite(targets).all() or np.any((targets < 1.0) | (targets > 5.0)):
        raise ValueError("SLP targets must use the released 1-to-5 scale")
    tasks = np.asarray(dataset.task_indices)
    if tasks.dtype.kind not in {"i", "u"} or np.any((tasks < 0) | (tasks >= 9)):
        raise ValueError("task indices are invalid")


def _fit_standardizer(dataset: MotionDataset, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.concatenate((dataset.valid_masks[indices], dataset.valid_masks[indices]))
    values = np.concatenate((dataset.landmarks[indices], dataset.mirrored_landmarks[indices]))
    rows = np.asarray(values[mask], dtype=np.float64)
    mean = rows.mean(axis=0)
    scale = rows.std(axis=0)
    scale[scale <= 1e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def _standardized(values: np.ndarray, mask: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    output = (np.asarray(values, dtype=np.float32) - mean) / scale
    return np.where(np.asarray(mask, dtype=bool)[..., None], output, 0.0).astype(np.float32)


def _tensor_batch(
    dataset: MotionDataset,
    indices: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    *,
    mirrored: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    values = dataset.mirrored_landmarks if mirrored else dataset.landmarks
    return (
        torch.as_tensor(
            _standardized(values[indices], dataset.valid_masks[indices], mean, scale),
            dtype=torch.float32, device=device,
        ),
        torch.as_tensor(dataset.valid_masks[indices], dtype=torch.bool, device=device),
        torch.as_tensor(dataset.timestamps[indices], dtype=torch.float32, device=device),
        torch.as_tensor(dataset.task_indices[indices], dtype=torch.long, device=device),
    )


def _weighted_regression_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    predictions = torch.sigmoid(logits)
    losses = nn.functional.smooth_l1_loss(predictions, targets, reduction="none").mean(dim=1)
    return (losses * weights).sum() / weights.sum().clamp_min(torch.finfo(losses.dtype).eps)


def _train_model(
    dataset: MotionDataset,
    train: np.ndarray,
    validation: np.ndarray | None,
    mean: np.ndarray,
    scale: np.ndarray,
    *,
    epochs: int,
    config: MotionPretrainConfig,
    device: torch.device,
) -> tuple[MotionQualityRegressor, int, np.ndarray | None]:
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    original = _tensor_batch(dataset, train, mean, scale, mirrored=False, device=device)
    mirrored = _tensor_batch(dataset, train, mean, scale, mirrored=True, device=device)
    train_inputs = tuple(torch.cat((left, right)) for left, right in zip(original, mirrored))
    normalized = (np.asarray(dataset.targets[train], dtype=np.float32) - 1.0) / 4.0
    targets = torch.as_tensor(np.concatenate((normalized, normalized)), device=device)
    weights = torch.as_tensor(group_balanced_weights(np.concatenate((
        dataset.group_ids[train], dataset.group_ids[train]
    ))), dtype=torch.float32, device=device)
    model = MotionQualityRegressor().to(device)
    if not 0 < count_parameters(model) < 30_000:
        raise RuntimeError("motion model violates its fixed capacity cap")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    validation_batches = None
    validation_targets = None
    validation_weights = None
    if validation is not None:
        validation_batches = (
            _tensor_batch(dataset, validation, mean, scale, mirrored=False, device=device),
            _tensor_batch(dataset, validation, mean, scale, mirrored=True, device=device),
        )
        validation_targets = torch.as_tensor(
            (np.asarray(dataset.targets[validation], dtype=np.float32) - 1.0) / 4.0,
            device=device,
        )
        validation_weights = torch.as_tensor(
            group_balanced_weights(dataset.group_ids[validation]),
            dtype=torch.float32, device=device,
        )
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _embedding = model(*train_inputs)
        loss = _weighted_regression_loss(logits, targets, weights)
        if not torch.isfinite(loss):
            raise RuntimeError("motion pretraining loss became nonfinite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if validation_batches is None:
            best_epoch = epoch
            continue
        model.eval()
        with torch.no_grad():
            original_logits, _ = model(*validation_batches[0])
            mirrored_logits, _ = model(*validation_batches[1])
            probability = 0.5 * (
                torch.sigmoid(original_logits) + torch.sigmoid(mirrored_logits)
            )
            eps = torch.finfo(probability.dtype).eps
            mean_logit = torch.logit(probability.clamp(eps, 1 - eps))
            value = float(_weighted_regression_loss(
                mean_logit, validation_targets, validation_weights
            ).item())
        if value < best_loss - 1e-7:
            best_loss = value
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if validation_batches is not None:
        if best_state is None:
            raise RuntimeError("motion fold produced no finite validation state")
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            original_logits, _ = model(*validation_batches[0])
            mirrored_logits, _ = model(*validation_batches[1])
            predictions = 0.5 * (
                torch.sigmoid(original_logits) + torch.sigmoid(mirrored_logits)
            )
        observed = 1.0 + 4.0 * predictions.detach().cpu().numpy().astype(np.float64)
        return model.cpu(), best_epoch, observed
    return model.cpu(), best_epoch, None


def _pretrain_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    groups: np.ndarray,
    best_epochs: list[int],
    final_epochs: int,
) -> dict[str, object]:
    domains: dict[str, dict[str, float]] = {}
    for index, domain in enumerate(DOMAINS):
        rho = spearmanr(targets[:, index], predictions[:, index]).statistic
        domains[domain] = {
            "spearman": float(rho) if np.isfinite(rho) else 0.0,
            "mae": float(np.mean(np.abs(targets[:, index] - predictions[:, index]))),
        }
    participant_errors = []
    for group in sorted(set(groups.tolist())):
        rows = groups == group
        participant_errors.append(float(np.mean(np.abs(targets[rows] - predictions[rows]))))
    return {
        "domains": domains,
        "participant_macro_mae": float(np.mean(participant_errors)),
        "best_epochs": [int(value) for value in best_epochs],
        "final_epochs": int(final_epochs),
    }


def run_motion_pretraining(
    dataset: MotionDataset,
    *,
    config: MotionPretrainConfig | None = None,
    device: str | torch.device = "cpu",
) -> MotionPretrainResult:
    config = config or MotionPretrainConfig()
    _validate_motion_dataset(dataset)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA motion pretraining requested but unavailable")
    assignments = build_stratified_participant_folds(
        dataset.group_ids, dataset.cohorts, folds=config.folds, seed=config.seed
    )
    oof = np.full(dataset.targets.shape, np.nan, dtype=np.float64)
    best_epochs: list[int] = []
    all_indices = np.arange(dataset.targets.shape[0], dtype=np.int64)
    for fold in range(config.folds):
        train = all_indices[assignments != fold]
        validation = all_indices[assignments == fold]
        mean, scale = _fit_standardizer(dataset, train)
        _model, best_epoch, predictions = _train_model(
            dataset, train, validation, mean, scale,
            epochs=config.epochs, config=config, device=torch_device,
        )
        if predictions is None:
            raise AssertionError("held-out motion fold did not return predictions")
        oof[validation] = predictions
        best_epochs.append(best_epoch)
    if not np.isfinite(oof).all():
        raise RuntimeError("participant-held-out motion predictions are incomplete")
    final_epochs = max(1, int(np.median(np.asarray(best_epochs, dtype=np.int64))))
    mean, scale = _fit_standardizer(dataset, all_indices)
    final_model, observed_epochs, _ = _train_model(
        dataset, all_indices, None, mean, scale,
        epochs=final_epochs, config=config, device=torch_device,
    )
    if observed_epochs != final_epochs:
        raise RuntimeError("final motion fit did not use the locked median epoch count")
    return MotionPretrainResult(
        metrics=_pretrain_metrics(
            np.asarray(dataset.targets, dtype=np.float64), oof,
            np.asarray(dataset.group_ids, dtype=object), best_epochs, final_epochs,
        ),
        final_model=final_model,
        landmark_mean=mean,
        landmark_scale=scale,
        parameter_count=count_parameters(final_model),
    )


def frozen_motion_embeddings(
    model: MotionQualityRegressor,
    landmarks: np.ndarray,
    mirrored_landmarks: np.ndarray,
    valid_masks: np.ndarray,
    timestamps: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    *,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    torch_device = torch.device(device)
    encoder = copy.deepcopy(model.encoder).to(torch_device).eval()
    mask_tensor = torch.as_tensor(valid_masks, dtype=torch.bool, device=torch_device)
    time_tensor = torch.as_tensor(timestamps, dtype=torch.float32, device=torch_device)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for values in (landmarks, mirrored_landmarks):
            standardized = _standardized(values, valid_masks, mean, scale)
            embedding = encoder(
                torch.as_tensor(standardized, dtype=torch.float32, device=torch_device),
                mask_tensor,
                time_tensor,
            )
            outputs.append(embedding.cpu().numpy().astype(np.float64))
    if any(value.shape != (len(landmarks), EMBEDDING_DIM) for value in outputs):
        raise RuntimeError("frozen motion embedding shape drifted")
    return outputs[0], outputs[1]


def _aggregate_groups(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    group_labels: list[int] = []
    group_probabilities: list[float] = []
    for group in sorted(set(groups.tolist())):
        rows = groups == group
        values = set(labels[rows].tolist())
        if len(values) != 1:
            raise ValueError("one PalsyNet group crosses binary labels")
        group_labels.append(int(next(iter(values))))
        group_probabilities.append(float(np.mean(probabilities[rows])))
    return np.asarray(group_labels), np.asarray(group_probabilities)


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("binary transfer metrics require both classes")
    predictions = probabilities >= 0.5
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity": float(np.mean(predictions[labels == 1])),
        "specificity": float(np.mean(~predictions[labels == 0])),
    }


def _transfer_features(dataset: TransferDataset, name: str, mirrored: bool) -> np.ndarray:
    summary = dataset.mirrored_summary_features if mirrored else dataset.summary_features
    motion = dataset.mirrored_motion_features if mirrored else dataset.motion_features
    if name == "landmark_110d":
        return summary
    if name == "motion_32d":
        return motion
    if name == "landmark_110d_plus_motion_32d":
        return np.concatenate((summary, motion), axis=1)
    raise ValueError("unknown transfer representation")


def _validate_transfer_dataset(dataset: TransferDataset) -> None:
    labels = np.asarray(dataset.labels)
    n = labels.size
    if (
        labels.ndim != 1 or labels.dtype.kind not in {"i", "u"}
        or np.asarray(dataset.group_ids).shape != (n,)
        or np.asarray(dataset.summary_features).shape != (n, 110)
        or np.asarray(dataset.mirrored_summary_features).shape != (n, 110)
        or np.asarray(dataset.motion_features).shape != (n, EMBEDDING_DIM)
        or np.asarray(dataset.mirrored_motion_features).shape != (n, EMBEDDING_DIM)
        or np.asarray(dataset.inner_fold_by_index).shape != (n,)
    ):
        raise ValueError("PalsyNet transfer arrays are misaligned")
    require_development_only(
        dataset.development_indices, dataset.development_indices,
        dataset.protected_indices, "frozen motion transfer",
    )
    development = np.asarray(dataset.development_indices, dtype=np.int64)
    if set(np.asarray(dataset.inner_fold_by_index)[development].tolist()) != {0, 1, 2, 3}:
        raise ValueError("PalsyNet transfer requires the four reviewed inner folds")
    for name in MODEL_ORDER:
        for mirrored in (False, True):
            if not np.isfinite(_transfer_features(dataset, name, mirrored)[development]).all():
                raise ValueError("development transfer features must be finite")


def evaluate_frozen_palsynet_transfer(dataset: TransferDataset) -> TransferResult:
    _validate_transfer_dataset(dataset)
    development = np.asarray(dataset.development_indices, dtype=np.int64)
    protected = np.asarray(dataset.protected_indices, dtype=np.int64)
    folds = np.asarray(dataset.inner_fold_by_index, dtype=np.int64)
    labels = np.asarray(dataset.labels, dtype=np.int64)
    groups = np.asarray(dataset.group_ids, dtype=object)
    metrics: dict[str, Mapping[str, float]] = {}
    for name in MODEL_ORDER:
        oof = np.full(labels.shape, np.nan, dtype=np.float64)
        for fold in range(4):
            train = development[folds[development] != fold]
            validation = development[folds[development] == fold]
            require_development_only(train, development, protected, f"{name} fit")
            require_development_only(validation, development, protected, f"{name} predict")
            train_features = np.concatenate((
                _transfer_features(dataset, name, False)[train],
                _transfer_features(dataset, name, True)[train],
            ))
            train_labels = np.concatenate((labels[train], labels[train]))
            train_groups = np.concatenate((groups[train], groups[train]))
            scaler = StandardScaler().fit(train_features)
            classifier = LogisticRegression(
                C=0.01, penalty="l2", solver="liblinear", max_iter=2000,
                random_state=0,
            )
            classifier.fit(
                scaler.transform(train_features), train_labels,
                sample_weight=group_balanced_weights(train_groups),
            )
            original = classifier.predict_proba(scaler.transform(
                _transfer_features(dataset, name, False)[validation]
            ))[:, 1]
            mirrored = classifier.predict_proba(scaler.transform(
                _transfer_features(dataset, name, True)[validation]
            ))[:, 1]
            oof[validation] = 0.5 * (original + mirrored)
        if not np.isfinite(oof[development]).all() or np.isfinite(oof[protected]).any():
            raise OuterSearchLockedError("transfer predictions crossed the development gate")
        group_labels, group_probabilities = _aggregate_groups(
            labels[development], oof[development], groups[development]
        )
        metrics[name] = _binary_metrics(group_labels, group_probabilities)
    return TransferResult(
        metrics=metrics,
        development_recordings=int(development.size),
        development_groups=len(set(groups[development].tolist())),
        protected_predictions=0,
    )


def build_aggregate_report(
    *,
    pretrain_metrics: Mapping[str, object],
    transfer: TransferResult,
    provenance: Mapping[str, str],
    runtime: Mapping[str, object],
    parameter_count: int,
) -> dict[str, object]:
    required = {
        "neuroface_private_manifest_sha256",
        "neuroface_cache_collection_sha256",
        "palsynet_cache_collection_sha256",
        "palsynet_reviewed_manifest_sha256",
        "palsynet_review_ledger_sha256",
        "palsynet_split_registry_sha256",
        "implementation_sha256",
        "dependency_lock_sha256",
    }
    if set(provenance) != required or any(
        not isinstance(value, str) or len(value) != 64 for value in provenance.values()
    ):
        raise ValueError("motion report provenance must contain exact SHA-256 fields")
    if transfer.protected_predictions != 0 or not 0 < int(parameter_count) < 30_000:
        raise ValueError("motion report violates its capacity or protected-data boundary")
    baseline = transfer.metrics["landmark_110d"]
    fusion = transfer.metrics["landmark_110d_plus_motion_32d"]
    promoted = (
        fusion["auroc"] > baseline["auroc"]
        and fusion["balanced_accuracy"] >= baseline["balanced_accuracy"] - 0.02
        and fusion["brier"] <= baseline["brier"]
    )
    report = {
        "schema_version": "neuroface_motion_quality_pretraining_v1_report",
        "claim_scope": "exploratory_neuroface_pretraining_palsynet_development_oof_only",
        "target": "binary_affected_vs_unaffected_not_hb_grade",
        "protocol": {
            "pretraining_folds": 6,
            "palsynet_development_folds": 4,
            "seed": 20260813,
            "epochs_max": 80,
            "patience": 10,
            "embedding_dimensions": 32,
            "classifier": "standardized_l2_logistic_C_0.01",
            "threshold": 0.5,
        },
        "counts": {
            "neuroface_retained_videos": 231,
            "neuroface_participants": 36,
            "palsynet_development_recordings": transfer.development_recordings,
            "palsynet_development_groups": transfer.development_groups,
        },
        "pretraining_diagnostics": dict(pretrain_metrics),
        "palsynet_development_metrics": {
            name: dict(transfer.metrics[name]) for name in MODEL_ORDER
        },
        "parameter_count": int(parameter_count),
        "decision": {
            "promotion_criteria_met": bool(promoted),
            "current_model_replaced": bool(promoted),
            "selected_model": (
                "landmark_110d_plus_motion_32d" if promoted else "landmark_110d"
            ),
            "outer_evaluation_authorized": False,
            "mayo_used_for_selection": False,
            "meei_used_for_selection": False,
            "neuroface_is_external_validation_after_pretraining": False,
            "clinical_validation": False,
        },
        "audit": {
            "palsynet_protected_cache_records_loaded": 0,
            "protected_feature_extractions": 0,
            "protected_model_fits": 0,
            "protected_predictions": 0,
        },
        "runtime": dict(runtime),
        "provenance": dict(provenance),
    }
    encoded = str(report).lower()
    if any(token in encoded for token in (
        "group_id", "recording_id", "participant_id", "grp_", "rec_", "/users/", "/home/"
    )):
        raise ValueError("aggregate motion report leaks private identifiers or paths")
    return report


__all__ = [
    "DOMAINS", "MotionDataset", "MotionPretrainConfig", "MotionPretrainResult",
    "TransferDataset", "TransferResult", "build_aggregate_report",
    "build_stratified_participant_folds", "evaluate_frozen_palsynet_transfer",
    "frozen_motion_embeddings", "run_motion_pretraining",
]
