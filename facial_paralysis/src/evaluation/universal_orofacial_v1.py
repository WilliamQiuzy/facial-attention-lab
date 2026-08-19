"""Source-blind participant contracts for Universal Orofacial Model v1."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.nn import functional as torch_functional

from src.models.universal_orofacial_v1 import UniversalLowRankModel


SOURCES = ("palsynet", "neuroface")
FEATURE_DIM = 110
CANDIDATES = (
    "source_balanced_logistic_110d",
    "groupdro_lowrank_110d",
    "multitask_lowrank_110d",
)
FROZEN_NEURAL_CONFIG = {
    "epochs": 120,
    "learning_rate": 0.01,
    "weight_decay": 0.05,
    "group_step": 0.1,
    "gradient_clip": 1.0,
    "seeds": (0, 1, 2),
    "auxiliary_weight": 0.5,
}
_GROUP_ID = re.compile(r"grp_[0-9a-f]{64}")


def _immutable(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


@dataclass(frozen=True)
class UniversalDataset:
    original: np.ndarray
    mirrored: np.ndarray
    labels: np.ndarray
    group_ids: tuple[str, ...]
    sources: tuple[str, ...]


@dataclass(frozen=True)
class MirrorScaler:
    mean: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: str
    protocol: str
    probabilities: np.ndarray
    metrics: dict[str, dict[str, float]]
    model_fits: int


@dataclass(frozen=True)
class LockedCandidate:
    """A fitted source-blind candidate with no participant or source identity."""

    candidate: str
    scaler: MirrorScaler
    models: tuple[dict[str, object], ...]
    model_fits: int


def aggregate_participant_recordings(
    original: np.ndarray,
    mirrored: np.ndarray,
    labels: np.ndarray,
    group_ids: Sequence[str],
    sources: Sequence[str],
) -> UniversalDataset:
    """Mean aggregate mirror-paired 110D recordings once per participant."""
    original = np.asarray(original)
    mirrored = np.asarray(mirrored)
    labels = np.asarray(labels)
    group_ids = tuple(group_ids)
    sources = tuple(sources)
    count = len(group_ids)
    if (
        original.shape != (count, FEATURE_DIM)
        or mirrored.shape != original.shape
        or original.dtype.kind != "f"
        or mirrored.dtype.kind != "f"
        or labels.shape != (count,)
        or labels.dtype.kind not in {"i", "u"}
        or len(sources) != count
        or count < 2
    ):
        raise ValueError("recordings must follow the exact mirror-paired 110D schema")
    if not np.isfinite(original).all() or not np.isfinite(mirrored).all():
        raise ValueError("recording representations must be finite")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("recording labels must be binary")
    if any(not isinstance(group, str) or _GROUP_ID.fullmatch(group) is None
           for group in group_ids):
        raise ValueError("group IDs must be opaque SHA-256 commitments")
    if any(source not in SOURCES for source in sources):
        raise ValueError("recording source differs from the frozen development sources")

    participant_original: list[np.ndarray] = []
    participant_mirrored: list[np.ndarray] = []
    participant_labels: list[int] = []
    participant_sources: list[str] = []
    ordered_groups = tuple(sorted(set(group_ids)))
    for group in ordered_groups:
        indices = np.asarray(
            [index for index, observed in enumerate(group_ids) if observed == group],
            dtype=np.int64,
        )
        group_labels = set(int(labels[index]) for index in indices)
        group_sources = set(sources[index] for index in indices)
        if len(group_labels) != 1 or len(group_sources) != 1:
            raise ValueError("one participant cannot change label or source")
        participant_original.append(original[indices].mean(axis=0, dtype=np.float64))
        participant_mirrored.append(mirrored[indices].mean(axis=0, dtype=np.float64))
        participant_labels.append(group_labels.pop())
        participant_sources.append(group_sources.pop())

    return UniversalDataset(
        original=_immutable(np.asarray(participant_original, dtype=np.float64)),
        mirrored=_immutable(np.asarray(participant_mirrored, dtype=np.float64)),
        labels=_immutable(np.asarray(participant_labels, dtype=np.int64)),
        group_ids=ordered_groups,
        sources=tuple(participant_sources),
    )


def source_class_balanced_weights(
    labels: np.ndarray,
    sources: Sequence[str],
) -> np.ndarray:
    """Give each of the two-source by two-class groups equal total weight."""
    labels = np.asarray(labels)
    sources = tuple(sources)
    if (
        labels.shape != (len(sources),)
        or labels.dtype.kind not in {"i", "u"}
        or not np.isin(labels, (0, 1)).all()
        or any(source not in SOURCES for source in sources)
    ):
        raise ValueError("weights require aligned binary labels and frozen sources")
    weights = np.zeros(labels.shape[0], dtype=np.float64)
    for source in SOURCES:
        for label in (0, 1):
            indices = np.asarray([
                index for index, (observed_source, observed_label)
                in enumerate(zip(sources, labels))
                if observed_source == source and int(observed_label) == label
            ], dtype=np.int64)
            if indices.size == 0:
                raise ValueError("every source-class group must be represented")
            weights[indices] = 0.25 / indices.size
    return _immutable(weights)


def weighted_mirror_scaler(
    original: np.ndarray,
    mirrored: np.ndarray,
    participant_weights: np.ndarray,
) -> MirrorScaler:
    """Fit a fold-local weighted scaler with equal original/mirror mass."""
    original = np.asarray(original)
    mirrored = np.asarray(mirrored)
    participant_weights = np.asarray(participant_weights)
    count = original.shape[0] if original.ndim == 2 else -1
    if (
        original.shape != (count, FEATURE_DIM)
        or mirrored.shape != original.shape
        or participant_weights.shape != (count,)
        or original.dtype.kind != "f"
        or mirrored.dtype.kind != "f"
        or participant_weights.dtype.kind != "f"
        or count < 2
        or not np.isfinite(original).all()
        or not np.isfinite(mirrored).all()
        or not np.isfinite(participant_weights).all()
        or np.any(participant_weights <= 0.0)
        or not np.isclose(participant_weights.sum(), 1.0)
    ):
        raise ValueError("scaler requires finite mirror-paired 110D training rows")
    augmented = np.concatenate((original, mirrored), axis=0).astype(
        np.float64, copy=False
    )
    augmented_weights = np.concatenate(
        (participant_weights * 0.5, participant_weights * 0.5)
    )
    mean = np.average(augmented, axis=0, weights=augmented_weights)
    variance = np.average(
        (augmented - mean) ** 2, axis=0, weights=augmented_weights
    )
    scale = np.sqrt(variance)
    scale[scale == 0.0] = 1.0
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("weighted scaler statistics are invalid")
    return MirrorScaler(_immutable(mean), _immutable(scale))


def apply_scaler(features: np.ndarray, scaler: MirrorScaler) -> np.ndarray:
    """Apply an authenticated fold-local 110D scaler."""
    features = np.asarray(features)
    if (
        not isinstance(scaler, MirrorScaler)
        or features.ndim != 2
        or features.shape[1] != FEATURE_DIM
        or features.dtype.kind != "f"
        or scaler.mean.shape != (FEATURE_DIM,)
        or scaler.scale.shape != (FEATURE_DIM,)
        or not np.isfinite(features).all()
        or not np.isfinite(scaler.mean).all()
        or not np.isfinite(scaler.scale).all()
        or np.any(scaler.scale <= 0.0)
    ):
        raise ValueError("scaling requires finite 110D features and valid statistics")
    scaled = (features.astype(np.float64, copy=False) - scaler.mean) / scaler.scale
    if not np.isfinite(scaled).all():
        raise ValueError("scaled universal features are nonfinite")
    return scaled


def group_dro_objective(
    losses: torch.Tensor,
    group_indices: torch.Tensor,
    group_weights: torch.Tensor,
    *,
    group_step: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update exponential worst-group weights and return the GroupDRO loss."""
    if (
        not isinstance(losses, torch.Tensor)
        or losses.ndim != 1
        or not losses.is_floating_point()
        or not bool(torch.isfinite(losses).all())
        or not isinstance(group_indices, torch.Tensor)
        or group_indices.shape != losses.shape
        or group_indices.dtype != torch.int64
        or group_indices.device != losses.device
        or not isinstance(group_weights, torch.Tensor)
        or group_weights.ndim != 1
        or group_weights.device != losses.device
        or not group_weights.is_floating_point()
        or not bool(torch.isfinite(group_weights).all())
        or bool((group_weights <= 0).any())
        or not np.isfinite(group_step)
        or group_step <= 0.0
    ):
        raise ValueError("GroupDRO inputs are invalid")
    group_losses = []
    for group in range(group_weights.numel()):
        selected = group_indices == group
        if not bool(selected.any()):
            raise ValueError("every frozen GroupDRO group must be represented")
        group_losses.append(losses[selected].mean())
    stacked = torch.stack(group_losses)
    updated = group_weights * torch.exp(group_step * stacked.detach())
    updated = updated / updated.sum()
    return torch.sum(updated * stacked), updated


def stratified_source_class_folds(
    labels: np.ndarray,
    group_ids: Sequence[str],
    sources: Sequence[str],
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Create six deterministic participant-disjoint source×class folds."""
    labels = np.asarray(labels)
    group_ids = tuple(group_ids)
    sources = tuple(sources)
    count = len(group_ids)
    if (
        labels.shape != (count,)
        or labels.dtype.kind not in {"i", "u"}
        or len(sources) != count
        or len(set(group_ids)) != count
        or any(_GROUP_ID.fullmatch(group) is None for group in group_ids)
    ):
        raise ValueError("folds require unique opaque participants and aligned labels")
    assignments = np.full(count, -1, dtype=np.int64)
    for source in SOURCES:
        for label in (0, 1):
            indices = sorted(
                (index for index in range(count)
                 if sources[index] == source and int(labels[index]) == label),
                key=lambda index: group_ids[index],
            )
            if len(indices) < 6:
                raise ValueError("six folds require at least six participants per group")
            for position, index in enumerate(indices):
                assignments[index] = position % 6
    if np.any(assignments < 0):
        raise ValueError("sources and labels differ from the frozen strata")
    all_indices = np.arange(count, dtype=np.int64)
    folds = []
    for fold in range(6):
        held = all_indices[assignments == fold]
        train = all_indices[assignments != fold]
        folds.append((_immutable(train), _immutable(held)))
    return tuple(folds)


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Recompute participant-level fixed-threshold binary metrics."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (
        labels.ndim != 1
        or probabilities.shape != labels.shape
        or labels.dtype.kind not in {"i", "u"}
        or not np.isin(labels, (0, 1)).all()
        or len(np.unique(labels)) != 2
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0.0) | (probabilities > 1.0))
    ):
        raise ValueError("metrics require finite probabilities and both binary classes")
    predicted = probabilities >= 0.5
    positive = labels == 1
    negative = labels == 0
    sensitivity = float(np.mean(predicted[positive]))
    specificity = float(np.mean(~predicted[negative]))
    return {
        "accuracy": float(np.mean(predicted == positive)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "brier": float(np.mean((probabilities - labels) ** 2)),
    }


def _augmented_fold_arrays(
    dataset: UniversalDataset,
    train_indices: np.ndarray,
) -> tuple[MirrorScaler, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = dataset.labels[train_indices]
    sources = tuple(dataset.sources[index] for index in train_indices)
    observed_sources = tuple(source for source in SOURCES if source in set(sources))
    if len(observed_sources) == 2:
        participant_weights = source_class_balanced_weights(labels, sources)
    elif len(observed_sources) == 1:
        participant_weights_array = np.zeros(labels.shape[0], dtype=np.float64)
        for label in (0, 1):
            selected = np.flatnonzero(labels == label)
            if selected.size == 0:
                raise ValueError("single-source training requires both classes")
            participant_weights_array[selected] = 0.5 / selected.size
        participant_weights = _immutable(participant_weights_array)
    else:
        raise ValueError("training rows require one or two frozen sources")
    scaler = weighted_mirror_scaler(
        dataset.original[train_indices], dataset.mirrored[train_indices],
        participant_weights,
    )
    features = np.concatenate((
        apply_scaler(dataset.original[train_indices], scaler),
        apply_scaler(dataset.mirrored[train_indices], scaler),
    )).astype(np.float32)
    augmented_labels = np.concatenate((labels, labels)).astype(np.float32)
    sample_weights = np.concatenate((
        participant_weights * 0.5, participant_weights * 0.5,
    )).astype(np.float32)
    source_indices = np.asarray(
        [SOURCES.index(source) for source in sources], dtype=np.int64
    )
    augmented_sources = np.concatenate((source_indices, source_indices))
    return scaler, features, augmented_labels, sample_weights, augmented_sources


def _held_scaled_pair(
    dataset: UniversalDataset,
    held_indices: np.ndarray,
    scaler: MirrorScaler,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        apply_scaler(dataset.original[held_indices], scaler).astype(np.float32),
        apply_scaler(dataset.mirrored[held_indices], scaler).astype(np.float32),
    )


def _fit_logistic_fold(
    dataset: UniversalDataset,
    train_indices: np.ndarray,
    held_indices: np.ndarray,
) -> np.ndarray:
    scaler, features, labels, sample_weights, _ = _augmented_fold_arrays(
        dataset, train_indices
    )
    model = LogisticRegression(
        C=0.01, penalty="l2", solver="liblinear", max_iter=2000,
        random_state=0,
    )
    model.fit(
        features, labels.astype(np.int64),
        sample_weight=sample_weights * train_indices.size,
    )
    held_original, held_mirrored = _held_scaled_pair(dataset, held_indices, scaler)
    return 0.5 * (
        model.predict_proba(held_original)[:, 1]
        + model.predict_proba(held_mirrored)[:, 1]
    )


def _train_neural_model(
    dataset: UniversalDataset,
    train_indices: np.ndarray,
    *,
    candidate: str,
    seed: int,
    device: str,
) -> tuple[MirrorScaler, UniversalLowRankModel]:
    scaler, features, labels, sample_weights, source_indices = (
        _augmented_fold_arrays(dataset, train_indices)
    )
    torch_device = torch.device(device)
    torch.manual_seed(seed)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    auxiliary = candidate == "multitask_lowrank_110d"
    model = UniversalLowRankModel(auxiliary_heads=auxiliary).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(FROZEN_NEURAL_CONFIG["learning_rate"]),
        weight_decay=float(FROZEN_NEURAL_CONFIG["weight_decay"]),
    )
    feature_tensor = torch.from_numpy(features).to(torch_device)
    label_tensor = torch.from_numpy(labels).to(torch_device)
    weight_tensor = torch.from_numpy(sample_weights).to(torch_device)
    source_tensor = torch.from_numpy(source_indices).to(torch_device)
    raw_group_indices = source_tensor * 2 + label_tensor.to(torch.int64)
    _, group_indices = torch.unique(
        raw_group_indices, sorted=True, return_inverse=True
    )
    group_count = int(group_indices.max().item()) + 1
    dro_weights = torch.full(
        (group_count,), 1.0 / group_count, device=torch_device
    )
    for _ in range(int(FROZEN_NEURAL_CONFIG["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        universal_logits = model(feature_tensor)
        losses = torch_functional.binary_cross_entropy_with_logits(
            universal_logits, label_tensor, reduction="none"
        )
        if candidate == "groupdro_lowrank_110d":
            loss, dro_weights = group_dro_objective(
                losses, group_indices, dro_weights,
                group_step=float(FROZEN_NEURAL_CONFIG["group_step"]),
            )
        elif auxiliary:
            universal_loss = torch.sum(weight_tensor * losses)
            auxiliary_losses = torch_functional.binary_cross_entropy_with_logits(
                model.auxiliary_logits(feature_tensor, source_tensor),
                label_tensor, reduction="none",
            )
            loss = universal_loss + float(
                FROZEN_NEURAL_CONFIG["auxiliary_weight"]
            ) * torch.sum(weight_tensor * auxiliary_losses)
        else:
            raise ValueError("unknown neural universal candidate")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(FROZEN_NEURAL_CONFIG["gradient_clip"])
        )
        optimizer.step()
    return scaler, model


def _predict_neural_pair(
    model: UniversalLowRankModel,
    original: np.ndarray,
    mirrored: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()
    with torch.inference_mode():
        original_probability = torch.sigmoid(model(
            torch.from_numpy(original.astype(np.float32, copy=False)).to(torch_device)
        ))
        mirrored_probability = torch.sigmoid(model(
            torch.from_numpy(mirrored.astype(np.float32, copy=False)).to(torch_device)
        ))
    probability = 0.5 * (original_probability + mirrored_probability)
    return probability.detach().cpu().numpy().astype(np.float64)


def _fit_neural_fold(
    dataset: UniversalDataset,
    train_indices: np.ndarray,
    held_indices: np.ndarray,
    *,
    candidate: str,
    seed: int,
    device: str,
) -> np.ndarray:
    scaler, model = _train_neural_model(
        dataset, train_indices, candidate=candidate, seed=seed, device=device
    )
    held_original, held_mirrored = _held_scaled_pair(dataset, held_indices, scaler)
    return _predict_neural_pair(
        model, held_original, held_mirrored, device=device
    )


def evaluate_candidate_oof(
    dataset: UniversalDataset,
    candidate: str,
    *,
    device: str,
) -> CandidateEvaluation:
    """Evaluate one frozen candidate in six participant-disjoint folds."""
    if not isinstance(dataset, UniversalDataset) or candidate not in CANDIDATES:
        raise ValueError("candidate evaluation requires the frozen universal contract")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    folds = stratified_source_class_folds(
        dataset.labels, dataset.group_ids, dataset.sources
    )
    probabilities = np.full(len(dataset.group_ids), np.nan, dtype=np.float64)
    model_fits = 0
    for train_indices, held_indices in folds:
        if candidate == "source_balanced_logistic_110d":
            probabilities[held_indices] = _fit_logistic_fold(
                dataset, train_indices, held_indices
            )
            model_fits += 1
        else:
            seed_probabilities = []
            for seed in FROZEN_NEURAL_CONFIG["seeds"]:
                seed_probabilities.append(_fit_neural_fold(
                    dataset, train_indices, held_indices,
                    candidate=candidate, seed=int(seed), device=device,
                ))
                model_fits += 1
            probabilities[held_indices] = np.mean(seed_probabilities, axis=0)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("universal OOF evaluation left missing probabilities")
    metrics = {"overall": binary_metrics(dataset.labels, probabilities)}
    for source in SOURCES:
        selected = np.asarray([
            observed == source for observed in dataset.sources
        ], dtype=bool)
        metrics[source] = binary_metrics(
            dataset.labels[selected], probabilities[selected]
        )
    return CandidateEvaluation(
        candidate=candidate,
        protocol="six_fold_source_class_stratified_participant_oof",
        probabilities=_immutable(probabilities),
        metrics=metrics,
        model_fits=model_fits,
    )


def select_universal_candidate(
    summaries: dict[str, dict[str, float]],
) -> str:
    """Select only from universal worst-source metrics with a stable tie break."""
    if set(summaries) != set(CANDIDATES):
        raise ValueError("selection requires exactly the three frozen candidates")
    required = {
        "worst_source_auroc", "worst_source_balanced_accuracy", "overall_brier",
    }
    ranking = []
    for order, candidate in enumerate(CANDIDATES):
        summary = summaries[candidate]
        if not required.issubset(summary) or not all(
            np.isfinite(float(summary[name])) for name in required
        ):
            raise ValueError("candidate summary lacks universal selection metrics")
        ranking.append((
            float(summary["worst_source_auroc"]),
            float(summary["worst_source_balanced_accuracy"]),
            -float(summary["overall_brier"]),
            -order,
            candidate,
        ))
    return max(ranking)[-1]


def evaluate_leave_one_source_out(
    dataset: UniversalDataset,
    candidate: str,
    *,
    device: str,
) -> dict[str, dict[str, object]]:
    """Fit on one complete source and evaluate the universal head on the other."""
    if not isinstance(dataset, UniversalDataset) or candidate not in CANDIDATES:
        raise ValueError("source-held-out evaluation requires a frozen candidate")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    result: dict[str, dict[str, object]] = {}
    for training_source, held_source in (
        ("palsynet", "neuroface"), ("neuroface", "palsynet")
    ):
        train_indices = np.asarray([
            index for index, source in enumerate(dataset.sources)
            if source == training_source
        ], dtype=np.int64)
        held_indices = np.asarray([
            index for index, source in enumerate(dataset.sources)
            if source == held_source
        ], dtype=np.int64)
        if len(np.unique(dataset.labels[train_indices])) != 2 or len(
            np.unique(dataset.labels[held_indices])
        ) != 2:
            raise ValueError("each source must contain both participant classes")
        if candidate == "source_balanced_logistic_110d":
            probabilities = _fit_logistic_fold(
                dataset, train_indices, held_indices
            )
            model_fits = 1
        else:
            probabilities_by_seed = [
                _fit_neural_fold(
                    dataset, train_indices, held_indices,
                    candidate=candidate, seed=int(seed), device=device,
                )
                for seed in FROZEN_NEURAL_CONFIG["seeds"]
            ]
            probabilities = np.mean(probabilities_by_seed, axis=0)
            model_fits = len(probabilities_by_seed)
        key = f"{training_source}_to_{held_source}"
        result[key] = {
            "training_source": training_source,
            "held_source": held_source,
            "training_participants": int(train_indices.size),
            "held_participants": int(held_indices.size),
            "model_fits": int(model_fits),
            "metrics": binary_metrics(dataset.labels[held_indices], probabilities),
        }
    return result


def _stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    result = np.empty(logits.shape, dtype=np.float64)
    nonnegative = logits >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
    exponential = np.exp(logits[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    return result


def _validate_prediction_pair(
    original: np.ndarray,
    mirrored: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    original = np.asarray(original)
    mirrored = np.asarray(mirrored)
    count = original.shape[0] if original.ndim == 2 else -1
    if (
        original.shape != (count, FEATURE_DIM)
        or mirrored.shape != original.shape
        or original.dtype.kind != "f"
        or mirrored.dtype.kind != "f"
        or count < 1
        or not np.isfinite(original).all()
        or not np.isfinite(mirrored).all()
    ):
        raise ValueError("prediction requires finite mirror-paired 110D rows")
    return original, mirrored


def _neural_state(model: UniversalLowRankModel, *, seed: int) -> dict[str, object]:
    return {
        "type": "neural",
        "seed": int(seed),
        "parameters": {
            name: _immutable(tensor.detach().cpu().numpy().astype(np.float32))
            for name, tensor in model.state_dict().items()
        },
    }


def fit_locked_candidate(
    dataset: UniversalDataset,
    candidate: str,
    *,
    device: str,
) -> LockedCandidate:
    """Fit the selected candidate once on all development participants."""
    if not isinstance(dataset, UniversalDataset) or candidate not in CANDIDATES:
        raise ValueError("locking requires one frozen universal candidate")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    train_indices = np.arange(len(dataset.group_ids), dtype=np.int64)
    if candidate == "source_balanced_logistic_110d":
        scaler, features, labels, sample_weights, _ = _augmented_fold_arrays(
            dataset, train_indices
        )
        model = LogisticRegression(
            C=0.01, penalty="l2", solver="liblinear", max_iter=2000,
            random_state=0,
        )
        model.fit(
            features, labels.astype(np.int64),
            sample_weight=sample_weights * train_indices.size,
        )
        states = ({
            "type": "logistic",
            "coef": _immutable(model.coef_[0].astype(np.float64)),
            "intercept": float(model.intercept_[0]),
        },)
    else:
        states_list: list[dict[str, object]] = []
        scaler: MirrorScaler | None = None
        for seed in FROZEN_NEURAL_CONFIG["seeds"]:
            observed_scaler, model = _train_neural_model(
                dataset, train_indices, candidate=candidate,
                seed=int(seed), device=device,
            )
            if scaler is None:
                scaler = observed_scaler
            elif not (
                np.array_equal(scaler.mean, observed_scaler.mean)
                and np.array_equal(scaler.scale, observed_scaler.scale)
            ):
                raise RuntimeError("seed fits changed the frozen training scaler")
            states_list.append(_neural_state(model, seed=int(seed)))
        if scaler is None:
            raise RuntimeError("neural candidate produced no fitted models")
        states = tuple(states_list)
    return LockedCandidate(
        candidate=candidate,
        scaler=scaler,
        models=tuple(states),
        model_fits=len(states),
    )


def predict_locked_candidate(
    locked: LockedCandidate,
    original: np.ndarray,
    mirrored: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    """Predict with a locked universal head without labels, sources, or refit."""
    if not isinstance(locked, LockedCandidate) or locked.candidate not in CANDIDATES:
        raise ValueError("prediction requires a validated locked candidate")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    original, mirrored = _validate_prediction_pair(original, mirrored)
    scaled_original = apply_scaler(original, locked.scaler)
    scaled_mirrored = apply_scaler(mirrored, locked.scaler)
    probabilities: list[np.ndarray] = []
    if locked.candidate == "source_balanced_logistic_110d":
        if len(locked.models) != 1:
            raise ValueError("locked Logistic requires exactly one model")
        state = locked.models[0]
        if set(state) != {"type", "coef", "intercept"} or state["type"] != "logistic":
            raise ValueError("locked Logistic state is invalid")
        coefficient = np.asarray(state["coef"])
        intercept = state["intercept"]
        if (
            coefficient.shape != (FEATURE_DIM,)
            or coefficient.dtype.kind != "f"
            or not np.isfinite(coefficient).all()
            or not isinstance(intercept, float)
            or not np.isfinite(intercept)
        ):
            raise ValueError("locked Logistic parameters are invalid")
        probabilities.append(0.5 * (
            _stable_sigmoid(scaled_original @ coefficient + intercept)
            + _stable_sigmoid(scaled_mirrored @ coefficient + intercept)
        ))
    else:
        if len(locked.models) != len(FROZEN_NEURAL_CONFIG["seeds"]):
            raise ValueError("locked neural candidate requires every frozen seed")
        expected_seeds = tuple(int(seed) for seed in FROZEN_NEURAL_CONFIG["seeds"])
        observed_seeds = tuple(int(state.get("seed", -1)) for state in locked.models)
        if observed_seeds != expected_seeds:
            raise ValueError("locked neural seeds differ from the frozen order")
        for state in locked.models:
            if set(state) != {"type", "seed", "parameters"} or state["type"] != "neural":
                raise ValueError("locked neural state is invalid")
            model = UniversalLowRankModel(
                auxiliary_heads=locked.candidate == "multitask_lowrank_110d"
            )
            expected = model.state_dict()
            parameters = state["parameters"]
            if not isinstance(parameters, Mapping) or set(parameters) != set(expected):
                raise ValueError("locked neural parameter names are invalid")
            restored = {}
            for name, template in expected.items():
                value = np.asarray(parameters[name])
                if (
                    value.shape != tuple(template.shape)
                    or value.dtype.kind != "f"
                    or not np.isfinite(value).all()
                ):
                    raise ValueError("locked neural parameter shape is invalid")
                restored[name] = torch.from_numpy(
                    value.astype(np.float32, copy=True)
                )
            model.load_state_dict(restored, strict=True)
            probabilities.append(_predict_neural_pair(
                model,
                scaled_original.astype(np.float32),
                scaled_mirrored.astype(np.float32),
                device=device,
            ))
    result = np.mean(probabilities, axis=0)
    if result.shape != (original.shape[0],) or not np.isfinite(result).all():
        raise RuntimeError("locked universal prediction is invalid")
    return _immutable(result.astype(np.float64))


def locked_candidate_to_dict(locked: LockedCandidate) -> dict[str, object]:
    """Encode a locked candidate as a closed JSON-compatible private artifact."""
    if not isinstance(locked, LockedCandidate):
        raise ValueError("private artifact requires a locked candidate")
    models: list[dict[str, object]] = []
    for state in locked.models:
        if state.get("type") == "logistic":
            models.append({
                "type": "logistic",
                "coef": np.asarray(state["coef"]).tolist(),
                "intercept": float(state["intercept"]),
            })
        elif state.get("type") == "neural":
            parameters = state.get("parameters")
            if not isinstance(parameters, Mapping):
                raise ValueError("neural parameter state is invalid")
            models.append({
                "type": "neural",
                "seed": int(state["seed"]),
                "parameters": {
                    name: np.asarray(value).tolist()
                    for name, value in parameters.items()
                },
            })
        else:
            raise ValueError("unknown locked model type")
    payload = {
        "schema_version": "universal_orofacial_locked_candidate_v1",
        "candidate": locked.candidate,
        "scaler": {
            "mean": locked.scaler.mean.tolist(),
            "scale": locked.scaler.scale.tolist(),
        },
        "models": models,
    }
    locked_candidate_from_dict(payload)
    return payload


def locked_candidate_from_dict(payload: object) -> LockedCandidate:
    """Validate and restore a closed JSON-compatible private artifact."""
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "candidate", "scaler", "models",
    }:
        raise ValueError("locked candidate artifact schema is invalid")
    if payload["schema_version"] != "universal_orofacial_locked_candidate_v1":
        raise ValueError("locked candidate artifact version is invalid")
    candidate = payload["candidate"]
    if candidate not in CANDIDATES:
        raise ValueError("locked candidate name is invalid")
    scaler_payload = payload["scaler"]
    if not isinstance(scaler_payload, dict) or set(scaler_payload) != {"mean", "scale"}:
        raise ValueError("locked scaler schema is invalid")
    mean = np.asarray(scaler_payload["mean"], dtype=np.float64)
    scale = np.asarray(scaler_payload["scale"], dtype=np.float64)
    if (
        mean.shape != (FEATURE_DIM,)
        or scale.shape != (FEATURE_DIM,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0.0)
    ):
        raise ValueError("locked scaler values are invalid")
    raw_models = payload["models"]
    expected_count = 1 if candidate == "source_balanced_logistic_110d" else 3
    if not isinstance(raw_models, list) or len(raw_models) != expected_count:
        raise ValueError("locked artifact has the wrong model count")
    models: list[dict[str, object]] = []
    if candidate == "source_balanced_logistic_110d":
        raw = raw_models[0]
        if not isinstance(raw, dict) or set(raw) != {"type", "coef", "intercept"}:
            raise ValueError("locked Logistic schema is invalid")
        coefficient = np.asarray(raw["coef"], dtype=np.float64)
        intercept = raw["intercept"]
        if (
            raw["type"] != "logistic"
            or coefficient.shape != (FEATURE_DIM,)
            or not np.isfinite(coefficient).all()
            or not isinstance(intercept, (int, float))
            or isinstance(intercept, bool)
            or not np.isfinite(float(intercept))
        ):
            raise ValueError("locked Logistic values are invalid")
        models.append({
            "type": "logistic",
            "coef": _immutable(coefficient),
            "intercept": float(intercept),
        })
    else:
        auxiliary = candidate == "multitask_lowrank_110d"
        template = UniversalLowRankModel(auxiliary_heads=auxiliary).state_dict()
        expected_seeds = tuple(int(seed) for seed in FROZEN_NEURAL_CONFIG["seeds"])
        for raw, expected_seed in zip(raw_models, expected_seeds):
            if not isinstance(raw, dict) or set(raw) != {"type", "seed", "parameters"}:
                raise ValueError("locked neural schema is invalid")
            parameters = raw["parameters"]
            if (
                raw["type"] != "neural"
                or type(raw["seed"]) is not int
                or raw["seed"] != expected_seed
                or not isinstance(parameters, dict)
                or set(parameters) != set(template)
            ):
                raise ValueError("locked neural metadata is invalid")
            restored_parameters: dict[str, np.ndarray] = {}
            for name, expected in template.items():
                value = np.asarray(parameters[name], dtype=np.float32)
                if value.shape != tuple(expected.shape) or not np.isfinite(value).all():
                    raise ValueError("locked neural parameter values are invalid")
                restored_parameters[name] = _immutable(value)
            models.append({
                "type": "neural",
                "seed": expected_seed,
                "parameters": restored_parameters,
            })
    return LockedCandidate(
        candidate=str(candidate),
        scaler=MirrorScaler(_immutable(mean), _immutable(scale)),
        models=tuple(models),
        model_fits=len(models),
    )
