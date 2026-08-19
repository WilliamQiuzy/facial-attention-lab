"""Participant-disjoint source-balanced evaluation for Universal Multi-Signal v2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..preprocessing.universal_multisignal_v2 import REPRESENTATIONS
from .universal_orofacial_v1 import (
    CandidateEvaluation,
    SOURCES,
    binary_metrics,
    source_class_balanced_weights,
    stratified_source_class_folds,
)


def _immutable(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


@dataclass(frozen=True)
class MultiSignalDataset:
    original: Mapping[str, np.ndarray]
    mirrored: Mapping[str, np.ndarray]
    labels: np.ndarray
    group_ids: tuple[str, ...]
    sources: tuple[str, ...]


@dataclass(frozen=True)
class LockedMultiSignal:
    representation: str
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float


def aggregate_multisignal_recordings(
    rows: Mapping[str, tuple[np.ndarray, np.ndarray]],
    labels: np.ndarray,
    group_ids: Sequence[str],
    sources: Sequence[str],
) -> MultiSignalDataset:
    """Aggregate every representation to the same sorted participant order."""
    if not isinstance(rows, Mapping) or set(rows) != set(REPRESENTATIONS):
        raise ValueError("multi-signal rows require exactly three representations")
    labels = np.asarray(labels)
    groups = tuple(group_ids)
    sources = tuple(sources)
    count = len(groups)
    if (
        labels.shape != (count,)
        or labels.dtype.kind not in {"i", "u"}
        or not np.isin(labels, (0, 1)).all()
        or len(sources) != count
        or any(source not in SOURCES for source in sources)
        or count < 2
    ):
        raise ValueError("multi-signal identity rows are invalid")
    normalized: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, dimension in REPRESENTATIONS.items():
        pair = rows[name]
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("every representation requires original/mirror matrices")
        original, mirrored = np.asarray(pair[0]), np.asarray(pair[1])
        if (
            original.shape != (count, dimension)
            or mirrored.shape != original.shape
            or original.dtype.kind != "f"
            or mirrored.dtype.kind != "f"
            or not np.isfinite(original).all()
            or not np.isfinite(mirrored).all()
        ):
            raise ValueError("multi-signal matrices differ from frozen dimensions")
        normalized[name] = (original, mirrored)
    ordered_groups = tuple(sorted(set(groups)))
    participant_labels: list[int] = []
    participant_sources: list[str] = []
    aggregate_original = {name: [] for name in REPRESENTATIONS}
    aggregate_mirrored = {name: [] for name in REPRESENTATIONS}
    for group in ordered_groups:
        indices = np.asarray([
            index for index, observed in enumerate(groups) if observed == group
        ], dtype=np.int64)
        observed_labels = set(int(labels[index]) for index in indices)
        observed_sources = set(sources[index] for index in indices)
        if len(observed_labels) != 1 or len(observed_sources) != 1:
            raise ValueError("participant label/source changed across recordings")
        participant_labels.append(observed_labels.pop())
        participant_sources.append(observed_sources.pop())
        for name in REPRESENTATIONS:
            aggregate_original[name].append(
                normalized[name][0][indices].mean(axis=0, dtype=np.float64)
            )
            aggregate_mirrored[name].append(
                normalized[name][1][indices].mean(axis=0, dtype=np.float64)
            )
    return MultiSignalDataset(
        original={name: _immutable(np.asarray(values, dtype=np.float64))
                  for name, values in aggregate_original.items()},
        mirrored={name: _immutable(np.asarray(values, dtype=np.float64))
                  for name, values in aggregate_mirrored.items()},
        labels=_immutable(np.asarray(participant_labels, dtype=np.int64)),
        group_ids=ordered_groups,
        sources=tuple(participant_sources),
    )


def _fit_predict(
    dataset: MultiSignalDataset,
    representation: str,
    train: np.ndarray,
    held: np.ndarray,
) -> np.ndarray:
    original = dataset.original[representation]
    mirrored = dataset.mirrored[representation]
    labels = dataset.labels[train]
    sources = tuple(dataset.sources[index] for index in train)
    observed = tuple(source for source in SOURCES if source in set(sources))
    if len(observed) == 2:
        weights = source_class_balanced_weights(labels, sources)
    elif len(observed) == 1:
        weights = np.zeros(train.size, dtype=np.float64)
        for label in (0, 1):
            selected = np.flatnonzero(labels == label)
            if selected.size == 0:
                raise ValueError("one-source fit requires both classes")
            weights[selected] = 0.5 / selected.size
    else:
        raise ValueError("fit requires one or two frozen sources")
    augmented = np.concatenate((original[train], mirrored[train]))
    augmented_weights = np.concatenate((weights * 0.5, weights * 0.5))
    mean = np.average(augmented, axis=0, weights=augmented_weights)
    scale = np.sqrt(np.average(
        (augmented - mean) ** 2, axis=0, weights=augmented_weights
    ))
    scale[scale == 0.0] = 1.0
    x_train = (augmented - mean) / scale
    y_train = np.concatenate((labels, labels))
    model = LogisticRegression(
        C=0.01, penalty="l2", solver="liblinear", max_iter=2000,
        random_state=0,
    )
    model.fit(
        x_train, y_train,
        sample_weight=augmented_weights * train.size,
    )
    held_original = (original[held] - mean) / scale
    held_mirrored = (mirrored[held] - mean) / scale
    return 0.5 * (
        model.predict_proba(held_original)[:, 1]
        + model.predict_proba(held_mirrored)[:, 1]
    )


def evaluate_multisignal_oof(
    dataset: MultiSignalDataset,
    representation: str,
) -> CandidateEvaluation:
    """Evaluate one representation with the same six-fold Logistic estimator."""
    if not isinstance(dataset, MultiSignalDataset) or representation not in REPRESENTATIONS:
        raise ValueError("multi-signal evaluation requires a frozen representation")
    folds = stratified_source_class_folds(
        dataset.labels, dataset.group_ids, dataset.sources
    )
    probabilities = np.full(len(dataset.group_ids), np.nan, dtype=np.float64)
    for train, held in folds:
        probabilities[held] = _fit_predict(dataset, representation, train, held)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("multi-signal OOF probabilities are incomplete")
    metrics = {"overall": binary_metrics(dataset.labels, probabilities)}
    for source in SOURCES:
        selected = np.asarray([
            observed == source for observed in dataset.sources
        ], dtype=bool)
        metrics[source] = binary_metrics(
            dataset.labels[selected], probabilities[selected]
        )
    return CandidateEvaluation(
        candidate=representation,
        protocol="six_fold_source_class_stratified_participant_oof",
        probabilities=_immutable(probabilities),
        metrics=metrics,
        model_fits=6,
    )


def evaluate_multisignal_leave_one_source_out(
    dataset: MultiSignalDataset,
    representation: str,
) -> dict[str, dict[str, object]]:
    """Evaluate both train-one-source/test-the-other transfer directions."""
    if representation not in REPRESENTATIONS:
        raise ValueError("unknown multi-signal representation")
    result = {}
    for training_source, held_source in (
        ("palsynet", "neuroface"), ("neuroface", "palsynet")
    ):
        train = np.asarray([
            index for index, source in enumerate(dataset.sources)
            if source == training_source
        ], dtype=np.int64)
        held = np.asarray([
            index for index, source in enumerate(dataset.sources)
            if source == held_source
        ], dtype=np.int64)
        probabilities = _fit_predict(dataset, representation, train, held)
        result[f"{training_source}_to_{held_source}"] = {
            "training_source": training_source,
            "held_source": held_source,
            "training_participants": int(train.size),
            "held_participants": int(held.size),
            "model_fits": 1,
            "metrics": binary_metrics(dataset.labels[held], probabilities),
        }
    return result


def select_multisignal_representation(
    summaries: Mapping[str, Mapping[str, float]],
) -> str:
    """Select by worst-source AUROC, BA, Brier, then frozen order."""
    if set(summaries) != set(REPRESENTATIONS):
        raise ValueError("selection requires all three representations")
    ranking = []
    for order, name in enumerate(REPRESENTATIONS):
        row = summaries[name]
        required = (
            "worst_source_auroc", "worst_source_balanced_accuracy",
            "overall_brier",
        )
        if any(key not in row or not np.isfinite(float(row[key])) for key in required):
            raise ValueError("representation summary is incomplete")
        ranking.append((
            float(row["worst_source_auroc"]),
            float(row["worst_source_balanced_accuracy"]),
            -float(row["overall_brier"]), -order, name,
        ))
    return max(ranking)[-1]


def fit_locked_multisignal(
    dataset: MultiSignalDataset,
    representation: str,
) -> LockedMultiSignal:
    """Fit one source-balanced Logistic on all development participants."""
    if not isinstance(dataset, MultiSignalDataset) or representation not in REPRESENTATIONS:
        raise ValueError("locking requires one frozen multi-signal representation")
    indices = np.arange(len(dataset.group_ids), dtype=np.int64)
    original = dataset.original[representation]
    mirrored = dataset.mirrored[representation]
    weights = source_class_balanced_weights(dataset.labels, dataset.sources)
    augmented = np.concatenate((original, mirrored))
    augmented_weights = np.concatenate((weights * 0.5, weights * 0.5))
    mean = np.average(augmented, axis=0, weights=augmented_weights)
    scale = np.sqrt(np.average(
        (augmented - mean) ** 2, axis=0, weights=augmented_weights
    ))
    scale[scale == 0.0] = 1.0
    model = LogisticRegression(
        C=0.01, penalty="l2", solver="liblinear", max_iter=2000,
        random_state=0,
    )
    model.fit(
        (augmented - mean) / scale,
        np.concatenate((dataset.labels, dataset.labels)),
        sample_weight=augmented_weights * indices.size,
    )
    return LockedMultiSignal(
        representation=representation,
        mean=_immutable(mean.astype(np.float64)),
        scale=_immutable(scale.astype(np.float64)),
        coefficient=_immutable(model.coef_[0].astype(np.float64)),
        intercept=float(model.intercept_[0]),
    )


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    result = np.empty(logits.shape, dtype=np.float64)
    nonnegative = logits >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
    exponential = np.exp(logits[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    return result


def predict_locked_multisignal(
    locked: LockedMultiSignal,
    original: np.ndarray,
    mirrored: np.ndarray,
) -> np.ndarray:
    """Apply a locked multi-signal model without labels, source, or refit."""
    if not isinstance(locked, LockedMultiSignal) or locked.representation not in REPRESENTATIONS:
        raise ValueError("prediction requires a locked multi-signal model")
    dimension = REPRESENTATIONS[locked.representation]
    original = np.asarray(original)
    mirrored = np.asarray(mirrored)
    count = original.shape[0] if original.ndim == 2 else -1
    if (
        original.shape != (count, dimension)
        or mirrored.shape != original.shape
        or count < 1
        or not np.isfinite(original).all()
        or not np.isfinite(mirrored).all()
        or locked.mean.shape != (dimension,)
        or locked.scale.shape != (dimension,)
        or locked.coefficient.shape != (dimension,)
        or not np.isfinite(locked.mean).all()
        or not np.isfinite(locked.scale).all()
        or not np.isfinite(locked.coefficient).all()
        or np.any(locked.scale <= 0.0)
        or not np.isfinite(locked.intercept)
    ):
        raise ValueError("locked multi-signal prediction inputs are invalid")
    original_logit = ((original - locked.mean) / locked.scale) @ locked.coefficient
    mirrored_logit = ((mirrored - locked.mean) / locked.scale) @ locked.coefficient
    result = 0.5 * (
        _sigmoid(original_logit + locked.intercept)
        + _sigmoid(mirrored_logit + locked.intercept)
    )
    return _immutable(result.astype(np.float64))


def locked_multisignal_to_dict(locked: LockedMultiSignal) -> dict[str, object]:
    """Encode one locked Logistic as strict JSON-compatible private evidence."""
    if not isinstance(locked, LockedMultiSignal):
        raise ValueError("private v2 artifact requires a locked model")
    payload = {
        "schema_version": "universal_multisignal_locked_v2",
        "representation": locked.representation,
        "scaler": {
            "mean": locked.mean.tolist(), "scale": locked.scale.tolist(),
        },
        "model": {
            "type": "l2_logistic_regression", "c": 0.01,
            "solver": "liblinear", "max_iter": 2000, "random_state": 0,
            "coefficient": locked.coefficient.tolist(),
            "intercept": locked.intercept,
        },
    }
    locked_multisignal_from_dict(payload)
    return payload


def locked_multisignal_from_dict(payload: object) -> LockedMultiSignal:
    """Validate and restore the exact v2 private artifact schema."""
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "representation", "scaler", "model",
    }:
        raise ValueError("locked v2 artifact schema is invalid")
    representation = payload["representation"]
    if (
        payload["schema_version"] != "universal_multisignal_locked_v2"
        or representation not in REPRESENTATIONS
    ):
        raise ValueError("locked v2 artifact identity is invalid")
    scaler = payload["scaler"]
    model = payload["model"]
    if (
        not isinstance(scaler, dict)
        or set(scaler) != {"mean", "scale"}
        or not isinstance(model, dict)
        or set(model) != {
            "type", "c", "solver", "max_iter", "random_state",
            "coefficient", "intercept",
        }
        or model["type"] != "l2_logistic_regression"
        or model["c"] != 0.01
        or model["solver"] != "liblinear"
        or model["max_iter"] != 2000
        or model["random_state"] != 0
    ):
        raise ValueError("locked v2 scaler/model metadata is invalid")
    dimension = REPRESENTATIONS[str(representation)]
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    coefficient = np.asarray(model["coefficient"], dtype=np.float64)
    intercept = model["intercept"]
    if (
        any(value.shape != (dimension,) for value in (mean, scale, coefficient))
        or not all(np.isfinite(value).all() for value in (mean, scale, coefficient))
        or np.any(scale <= 0.0)
        or isinstance(intercept, bool)
        or not isinstance(intercept, (int, float))
        or not np.isfinite(float(intercept))
    ):
        raise ValueError("locked v2 numeric values are invalid")
    return LockedMultiSignal(
        representation=str(representation),
        mean=_immutable(mean), scale=_immutable(scale),
        coefficient=_immutable(coefficient), intercept=float(intercept),
    )


__all__ = (
    "MultiSignalDataset", "REPRESENTATIONS",
    "aggregate_multisignal_recordings", "evaluate_multisignal_leave_one_source_out",
    "evaluate_multisignal_oof", "fit_locked_multisignal",
    "locked_multisignal_from_dict", "locked_multisignal_to_dict",
    "predict_locked_multisignal", "select_multisignal_representation",
)
