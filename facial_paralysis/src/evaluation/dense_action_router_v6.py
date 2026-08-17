"""Fully nested participant-disjoint evaluator for dense action Router v6."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DENSE_ARCHITECTURES = (
    "dense_sparse_logistic",
    "dense_action_experts",
    "dense_rbf",
    "dense_ucr4_fusion",
)
_STATISTIC_FAMILIES = {
    "all": (0, 1, 2, 3, 4, 5),
    "response": (1, 2, 3, 4, 5),
    "central": (1, 4, 5),
    "static_response": (1, 2, 3, 4),
    "action_response": (0, 1, 4, 5),
}
_VIEWS = ("augment", "mean", "minmax", "difference", "mean_absdiff")
_REGISTRY_FIELDS = frozenset(
    {
        "name",
        "architecture",
        "statistic_family",
        "view",
        "top_k",
        "c",
        "fusion_weight",
    }
)


def _immutable(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


@dataclass(frozen=True)
class DenseRouterDataset:
    original: np.ndarray
    mirrored: np.ndarray
    labels: np.ndarray
    group_ids: tuple[str, ...]
    action_slices: tuple[tuple[str, int, int], ...]
    baseline_probability: np.ndarray

    def __post_init__(self) -> None:
        if (
            type(self.original) is not np.ndarray
            or self.original.dtype != np.dtype(np.float64)
            or self.original.ndim != 2
            or self.original.shape[0] < 12
            or self.original.shape[1] < 1
            or type(self.mirrored) is not np.ndarray
            or self.mirrored.dtype != np.dtype(np.float64)
            or self.mirrored.shape != self.original.shape
            or not np.isfinite(self.original).all()
            or not np.isfinite(self.mirrored).all()
        ):
            raise ValueError("dense views must be finite exact float64 matrices")
        rows = self.original.shape[0]
        if (
            type(self.labels) is not np.ndarray
            or self.labels.dtype != np.dtype(np.int64)
            or self.labels.shape != (rows,)
            or set(self.labels.tolist()) != {0, 1}
        ):
            raise ValueError("labels must be a nondegenerate exact int64 vector")
        if (
            type(self.group_ids) is not tuple
            or len(self.group_ids) != rows
            or len(set(self.group_ids)) != rows
            or any(type(value) is not str or not value for value in self.group_ids)
        ):
            raise ValueError("every row requires one unique participant group")
        if (
            type(self.baseline_probability) is not np.ndarray
            or self.baseline_probability.dtype != np.dtype(np.float64)
            or self.baseline_probability.shape != (rows,)
            or not np.isfinite(self.baseline_probability).all()
            or np.any((self.baseline_probability < 0) | (self.baseline_probability > 1))
        ):
            raise ValueError("baseline probability must be a finite [0,1] vector")
        if type(self.action_slices) is not tuple or not self.action_slices:
            raise ValueError("action_slices must be a nonempty exact tuple")
        cursor = 0
        names = []
        for item in self.action_slices:
            if (
                type(item) is not tuple
                or len(item) != 3
                or type(item[0]) is not str
                or not item[0]
                or type(item[1]) is not int
                or type(item[2]) is not int
                or item[1] != cursor
                or item[2] <= item[1]
            ):
                raise ValueError("action_slices must be a contiguous exact partition")
            names.append(item[0])
            cursor = item[2]
        if cursor != self.original.shape[1] or len(set(names)) != len(names):
            raise ValueError("action_slices must uniquely cover every feature")
        object.__setattr__(self, "original", _immutable(self.original))
        object.__setattr__(self, "mirrored", _immutable(self.mirrored))
        object.__setattr__(self, "labels", _immutable(self.labels))
        object.__setattr__(
            self, "baseline_probability", _immutable(self.baseline_probability)
        )


@dataclass(frozen=True)
class DenseRouterOOFResult:
    probability: np.ndarray
    prediction: np.ndarray
    metrics: Mapping[str, float | int]
    outer_folds: tuple[Mapping[str, object], ...]
    audit: Mapping[str, int]


def _validate_registry(
    registry: Sequence[Mapping[str, object]], dimension: int
) -> tuple[dict[str, object], ...]:
    if isinstance(registry, (str, bytes)):
        raise ValueError("candidate registry must be a sequence of closed mappings")
    candidates = []
    names = set()
    for raw in registry:
        if type(raw) is not dict or set(raw) != _REGISTRY_FIELDS:
            raise ValueError("candidate registry entry has an open schema")
        candidate = dict(raw)
        name = candidate["name"]
        architecture = candidate["architecture"]
        statistic = candidate["statistic_family"]
        view = candidate["view"]
        top_k = candidate["top_k"]
        c = candidate["c"]
        weight = candidate["fusion_weight"]
        if type(name) is not str or not name or name in names:
            raise ValueError("candidate names must be unique nonempty strings")
        if architecture not in DENSE_ARCHITECTURES:
            raise ValueError("candidate architecture is not frozen")
        if statistic not in _STATISTIC_FAMILIES or view not in _VIEWS:
            raise ValueError("candidate representation option is not frozen")
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, (int, np.integer))
            or int(top_k) < 1
            or int(top_k) > dimension
        ):
            raise ValueError("candidate top_k falls outside the input dimension")
        if (
            isinstance(c, bool)
            or not isinstance(c, (int, float))
            or not np.isfinite(float(c))
            or float(c) <= 0
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not np.isfinite(float(weight))
            or not 0.0 <= float(weight) <= 1.0
        ):
            raise ValueError("candidate estimator values are invalid")
        if architecture != "dense_ucr4_fusion" and float(weight) != 1.0:
            raise ValueError("only the registered fusion architecture may mix UCR4")
        candidate["top_k"] = int(top_k)
        candidate["c"] = float(c)
        candidate["fusion_weight"] = float(weight)
        names.add(name)
        candidates.append(candidate)
    if not candidates:
        raise ValueError("candidate registry cannot be empty")
    return tuple(candidates)


def _stat_columns(start: int, end: int, family: str) -> np.ndarray:
    if family == "all":
        return np.arange(start, end, dtype=np.int64)
    if (end - start) % 6:
        raise ValueError("non-all statistic families require six-stat feature blocks")
    allowed = np.asarray(_STATISTIC_FAMILIES[family], dtype=np.int64)
    local = np.arange(end - start, dtype=np.int64)
    return start + np.flatnonzero(np.isin(local % 6, allowed))


def _view_matrices(original: np.ndarray, mirrored: np.ndarray, view: str):
    if view == "augment":
        return original, mirrored, True
    if view == "mean":
        mean = 0.5 * (original + mirrored)
        return mean, mean, False
    if view == "minmax":
        invariant = np.concatenate(
            (np.minimum(original, mirrored), np.maximum(original, mirrored)), axis=1
        )
        return invariant, invariant, False
    if view == "difference":
        invariant = np.abs(original - mirrored)
        return invariant, invariant, False
    if view == "mean_absdiff":
        invariant = np.concatenate(
            (0.5 * (original + mirrored), np.abs(original - mirrored)), axis=1
        )
        return invariant, invariant, False
    raise ValueError("unknown dense view")


def _rank(matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        scores, _ = f_classif(matrix, labels)
    scores = np.nan_to_num(
        scores, nan=-np.inf, posinf=np.finfo(np.float64).max, neginf=-np.inf
    )
    return np.lexsort((np.arange(scores.size), -scores))


def _estimator(architecture: str, c: float):
    if architecture == "dense_rbf":
        return make_pipeline(
            StandardScaler(),
            SVC(
                C=c,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=20260817,
            ),
        )
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c,
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=3000,
            random_state=20260817,
        ),
    )


def _single_head_probability(
    original: np.ndarray,
    mirrored: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    held: np.ndarray,
    candidate: Mapping[str, object],
) -> tuple[np.ndarray, int, float]:
    first, second, augmented = _view_matrices(
        original, mirrored, str(candidate["view"])
    )
    rank_source = 0.5 * (first[train] + second[train])
    selected = _rank(rank_source, labels[train])[: int(candidate["top_k"])]
    if augmented:
        x_train = np.concatenate((first[train][:, selected], second[train][:, selected]))
        y_train = np.concatenate((labels[train], labels[train]))
        sample_weight = np.full(y_train.size, 0.5, dtype=np.float64)
    else:
        x_train = first[train][:, selected]
        y_train = labels[train]
        sample_weight = np.ones(y_train.size, dtype=np.float64)
    estimator = _estimator(str(candidate["architecture"]), float(candidate["c"]))
    step = "svc" if str(candidate["architecture"]) == "dense_rbf" else "logisticregression"
    estimator.fit(x_train, y_train, **{f"{step}__sample_weight": sample_weight})
    probability = 0.5 * (
        estimator.predict_proba(first[held][:, selected])[:, 1]
        + estimator.predict_proba(second[held][:, selected])[:, 1]
    )
    return probability, int(x_train.shape[0]), float(sample_weight.sum())


def _candidate_probability(
    dataset: DenseRouterDataset,
    train: np.ndarray,
    held: np.ndarray,
    candidate: Mapping[str, object],
) -> tuple[np.ndarray, int, float]:
    architecture = str(candidate["architecture"])
    if architecture == "dense_action_experts":
        outputs = []
        rows = 0
        weight_sum = 0.0
        for _, start, end in dataset.action_slices:
            columns = _stat_columns(start, end, str(candidate["statistic_family"]))
            local = dict(candidate)
            local["top_k"] = min(int(candidate["top_k"]), columns.size)
            probability, local_rows, local_weight = _single_head_probability(
                dataset.original[:, columns],
                dataset.mirrored[:, columns],
                dataset.labels,
                train,
                held,
                local,
            )
            outputs.append(probability)
            rows = max(rows, local_rows)
            weight_sum = max(weight_sum, local_weight)
        dense_probability = np.mean(outputs, axis=0)
    else:
        columns = np.concatenate(
            [
                _stat_columns(start, end, str(candidate["statistic_family"]))
                for _, start, end in dataset.action_slices
            ]
        )
        dense_probability, rows, weight_sum = _single_head_probability(
            dataset.original[:, columns],
            dataset.mirrored[:, columns],
            dataset.labels,
            train,
            held,
            candidate,
        )
    weight = float(candidate["fusion_weight"])
    probability = (
        weight * dense_probability
        + (1.0 - weight) * dataset.baseline_probability[held]
    )
    return probability, rows, weight_sum


def _threshold(labels: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.5], probability)))
    best = None
    for threshold in candidates:
        prediction = probability >= threshold
        key = (
            balanced_accuracy_score(labels, prediction),
            accuracy_score(labels, prediction),
            -abs(float(threshold) - 0.5),
        )
        if best is None or key > best[0]:
            best = (key, float(threshold))
    return best[1]


def _metric_key(labels: np.ndarray, probability: np.ndarray, threshold: float):
    prediction = probability >= threshold
    return (
        float(balanced_accuracy_score(labels, prediction)),
        float(accuracy_score(labels, prediction)),
        float(roc_auc_score(labels, probability)),
        -float(brier_score_loss(labels, probability)),
    )


def _metrics(labels: np.ndarray, probability: np.ndarray, prediction: np.ndarray):
    positive = labels == 1
    negative = labels == 0
    sensitivity = float(np.mean(prediction[positive]))
    specificity = float(np.mean(~prediction[negative]))
    return {
        "participants": int(labels.size),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auroc": float(roc_auc_score(labels, probability)),
        "brier": float(brier_score_loss(labels, probability)),
        "errors": int(np.sum(prediction != labels)),
    }


def run_nested_dense_router(
    dataset: DenseRouterDataset,
    *,
    registry: Sequence[Mapping[str, object]],
    outer_folds: int = 6,
    inner_folds: int = 5,
    random_state: int = 20260817,
) -> DenseRouterOOFResult:
    if type(dataset) is not DenseRouterDataset:
        raise ValueError("a validated DenseRouterDataset is required")
    candidates = _validate_registry(registry, dataset.original.shape[1])
    if (
        isinstance(outer_folds, bool)
        or not isinstance(outer_folds, int)
        or not 2 <= outer_folds <= 10
        or isinstance(inner_folds, bool)
        or not isinstance(inner_folds, int)
        or not 2 <= inner_folds < outer_folds
    ):
        raise ValueError("nested fold counts are invalid")
    outer = tuple(
        StratifiedKFold(
            outer_folds, shuffle=True, random_state=random_state
        ).split(np.zeros(dataset.labels.size), dataset.labels)
    )
    probability = np.full(dataset.labels.size, np.nan, dtype=np.float64)
    prediction = np.zeros(dataset.labels.size, dtype=bool)
    fold_reports = []
    inner_overlap = 0
    for outer_index, (outer_train, outer_held) in enumerate(outer):
        train_groups = {dataset.group_ids[index] for index in outer_train}
        held_groups = {dataset.group_ids[index] for index in outer_held}
        if train_groups & held_groups:
            raise ValueError("outer participant split overlaps")
        inner = tuple(
            StratifiedKFold(
                inner_folds, shuffle=True, random_state=random_state + outer_index + 1
            ).split(np.zeros(outer_train.size), dataset.labels[outer_train])
        )
        best = None
        for registry_index, candidate in enumerate(candidates):
            inner_probability = np.full(outer_train.size, np.nan, dtype=np.float64)
            for inner_train_local, inner_held_local in inner:
                inner_train = outer_train[inner_train_local]
                inner_held = outer_train[inner_held_local]
                first_groups = {dataset.group_ids[index] for index in inner_train}
                second_groups = {dataset.group_ids[index] for index in inner_held}
                inner_overlap += len(first_groups & second_groups)
                values, _, _ = _candidate_probability(
                    dataset, inner_train, inner_held, candidate
                )
                inner_probability[inner_held_local] = values
            if not np.isfinite(inner_probability).all():
                raise ValueError("inner OOF reconstruction is incomplete")
            threshold = _threshold(dataset.labels[outer_train], inner_probability)
            metric_key = _metric_key(
                dataset.labels[outer_train], inner_probability, threshold
            )
            selection_key = (
                *metric_key,
                -int(candidate["top_k"]),
                -registry_index,
            )
            if best is None or selection_key > best[0]:
                best = (selection_key, candidate, threshold)
        _, chosen, threshold = best
        held_probability, training_rows, weight_sum = _candidate_probability(
            dataset, outer_train, outer_held, chosen
        )
        probability[outer_held] = held_probability
        prediction[outer_held] = held_probability >= threshold
        fold_reports.append(
            {
                "fold": outer_index,
                "candidate": str(chosen["name"]),
                "threshold": float(threshold),
                "training_groups": int(outer_train.size),
                "held_groups": int(outer_held.size),
                "augmented_training_rows": training_rows,
                "training_weight_sum": weight_sum,
                "ranking_scope": "outer_training_groups_only",
            }
        )
    if not np.isfinite(probability).all():
        raise ValueError("outer OOF reconstruction is incomplete")
    return DenseRouterOOFResult(
        probability=_immutable(probability),
        prediction=_immutable(prediction),
        metrics=_metrics(dataset.labels, probability, prediction),
        outer_folds=tuple(fold_reports),
        audit={
            "outer_held_group_overlap": 0,
            "inner_held_group_overlap": int(inner_overlap),
            "protected_reads": 0,
        },
    )


__all__ = (
    "DENSE_ARCHITECTURES",
    "DenseRouterDataset",
    "DenseRouterOOFResult",
    "run_nested_dense_router",
)
