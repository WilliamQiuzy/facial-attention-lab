"""Leakage guards and fixed policy for Architecture Autoresearch v1."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn

from ..models.architecture_search_v1 import (
    CANDIDATE_ORDER,
    CLASSICAL_CANDIDATES,
    NEURAL_CANDIDATES,
    build_neural_candidate,
    count_trainable_parameters,
)


class OuterSearchLockedError(RuntimeError):
    """A protected record reached a development-only research operation."""


@dataclass(frozen=True)
class SearchDataset:
    raw_features: np.ndarray
    mirrored_raw_features: np.ndarray
    valid_masks: np.ndarray
    summary_features: np.ndarray
    mirrored_summary_features: np.ndarray
    labels: np.ndarray
    group_ids: np.ndarray
    development_indices: np.ndarray
    protected_indices: np.ndarray
    inner_fold_by_index: np.ndarray


@dataclass(frozen=True)
class ScreeningResult:
    candidate_metrics: Mapping[str, Mapping[str, object]]
    candidate_fold_metrics: Mapping[str, tuple[Mapping[str, float], ...]]
    winner: str
    development_recordings: int
    development_groups: int
    protected_predictions: int
    candidate_oof_probabilities: Mapping[str, np.ndarray] | None = None


@dataclass(frozen=True)
class ConfirmationResult:
    winner: str
    seed_metrics: Mapping[int, Mapping[str, float]]
    ensemble_metrics: Mapping[str, float]
    parameter_count: int
    protected_predictions: int


@dataclass(frozen=True)
class SearchConfig:
    screen_epochs: int = 40
    patience: int = 6
    screen_seed: int = 0
    confirmation_seeds: tuple[int, ...] = (0, 1, 2)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    smoke: bool = False

    def __post_init__(self) -> None:
        if self.screen_epochs != 40:
            raise ValueError("screen_epochs is frozen at 40")
        if self.patience != 6:
            raise ValueError("patience is frozen at 6")
        if self.screen_seed != 0 or self.confirmation_seeds != (0, 1, 2):
            raise ValueError("screen and confirmation seeds are frozen")
        if (
            not math.isfinite(self.learning_rate) or self.learning_rate <= 0
            or not math.isfinite(self.weight_decay) or self.weight_decay < 0
        ):
            raise ValueError("optimizer settings must be finite and valid")
        if not isinstance(self.smoke, bool):
            raise ValueError("smoke must be boolean")

    @property
    def effective_epochs(self) -> int:
        return 2 if self.smoke else self.screen_epochs


def _index_array(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must be a one-dimensional integer array")
    result = array.astype(np.int64, copy=False)
    if np.unique(result).size != result.size or np.any(result < 0):
        raise ValueError(f"{name} must contain unique nonnegative indices")
    return result


def require_development_only(
    requested_indices: Sequence[int] | np.ndarray,
    development_indices: Sequence[int] | np.ndarray,
    protected_indices: Sequence[int] | np.ndarray,
    operation: str,
) -> None:
    """Reject any protected or unregistered row before a research operation."""
    if not isinstance(operation, str) or not operation:
        raise ValueError("operation must be a nonempty string")
    requested = _index_array(requested_indices, "requested_indices")
    development = _index_array(development_indices, "development_indices")
    protected = _index_array(protected_indices, "protected_indices")
    requested_set = set(requested.tolist())
    development_set = set(development.tolist())
    protected_set = set(protected.tolist())
    if development_set & protected_set:
        raise ValueError("development and protected registries overlap")
    if requested_set & protected_set:
        raise OuterSearchLockedError(
            f"protected row reached development-only {operation} operation"
        )
    if not requested_set.issubset(development_set):
        raise ValueError(f"unregistered row reached {operation} operation")


def group_balanced_weights(group_ids: Sequence[object] | np.ndarray) -> np.ndarray:
    groups = np.asarray(group_ids, dtype=object)
    if groups.ndim != 1 or groups.size == 0:
        raise ValueError("group_ids must be a nonempty one-dimensional array")
    values = groups.tolist()
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("group_ids must be nonempty strings")
    counts = {group: values.count(group) for group in set(values)}
    return np.asarray([1.0 / counts[group] for group in values], dtype=np.float64)


def build_repeated_stratified_group_folds(
    labels: Sequence[int] | np.ndarray,
    group_ids: Sequence[object] | np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> tuple[np.ndarray, ...]:
    """Create deterministic four-fold assignments without splitting groups."""
    label_array = np.asarray(labels)
    groups = np.asarray(group_ids, dtype=object)
    if (
        label_array.ndim != 1 or groups.shape != label_array.shape
        or label_array.dtype.kind not in {"i", "u"}
        or not np.isin(label_array, (0, 1)).all()
    ):
        raise ValueError("repeated folds require aligned binary labels and groups")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    group_labels: dict[str, int] = {}
    for label, group in zip(label_array.tolist(), groups.tolist()):
        if not isinstance(group, str) or not group:
            raise ValueError("repeated-fold group IDs must be nonempty strings")
        previous = group_labels.setdefault(group, int(label))
        if previous != int(label):
            raise ValueError("one repeated-fold group cannot cross labels")
    by_label = {
        label: np.asarray(sorted(
            group for group, group_label in group_labels.items()
            if group_label == label
        ), dtype=object)
        for label in (0, 1)
    }
    if any(values.size < 4 for values in by_label.values()):
        raise ValueError("four-fold stratification needs at least four groups per class")
    rng = np.random.default_rng(seed)
    assignments: list[np.ndarray] = []
    for _repeat in range(repeats):
        fold_by_group: dict[str, int] = {}
        for label in (0, 1):
            shuffled = rng.permutation(by_label[label])
            for fold, chunk in enumerate(np.array_split(shuffled, 4)):
                for group in chunk.tolist():
                    fold_by_group[str(group)] = fold
        assignment = np.asarray(
            [fold_by_group[str(group)] for group in groups.tolist()], dtype=np.int64
        )
        for fold in range(4):
            if (
                set(label_array[assignment == fold].tolist()) != {0, 1}
                or set(label_array[assignment != fold].tolist()) != {0, 1}
            ):
                raise AssertionError("stratified group fold lost one class")
        assignments.append(assignment)
    return tuple(assignments)


def _metric(metrics: Mapping[str, object], name: str) -> float:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"candidate metric {name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"candidate metric {name} must be finite")
    return number


def candidate_rank_key(name: str, metrics: Mapping[str, object]) -> tuple[float, ...]:
    if not isinstance(name, str) or name not in CANDIDATE_ORDER:
        raise ValueError(f"unknown candidate {name!r}")
    auroc = _metric(metrics, "auroc")
    balanced = _metric(metrics, "balanced_accuracy")
    brier = _metric(metrics, "brier")
    parameters = _metric(metrics, "parameter_count")
    if not 0 <= auroc <= 1 or not 0 <= balanced <= 1 or not 0 <= brier <= 1:
        raise ValueError("probability metrics must lie within [0, 1]")
    if parameters <= 0 or not parameters.is_integer():
        raise ValueError("parameter_count must be a positive integer")
    return (-auroc, -balanced, brier, parameters, float(CANDIDATE_ORDER.index(name)))


def select_screening_winner(
    metrics_by_candidate: Mapping[str, Mapping[str, object]],
) -> str:
    if not isinstance(metrics_by_candidate, Mapping) or not metrics_by_candidate:
        raise ValueError("screening metrics cannot be empty")
    unknown = set(metrics_by_candidate) - set(CANDIDATE_ORDER)
    if unknown:
        raise ValueError(f"screening metrics contain unknown candidates {sorted(unknown)}")
    return min(
        metrics_by_candidate,
        key=lambda name: candidate_rank_key(name, metrics_by_candidate[name]),
    )


_ENSEMBLE_RECIPES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("logistic_extra_trees_mean", ("logistic_110d", "extra_trees_110d")),
    ("logistic_mlp_mean", ("logistic_110d", "mlp_110d")),
    ("logistic_hybrid_mean", ("logistic_110d", "hybrid_110d_tcn")),
    (
        "logistic_extra_hybrid_mean",
        ("logistic_110d", "extra_trees_110d", "hybrid_110d_tcn"),
    ),
)


def evaluate_fixed_ensembles(
    labels: Sequence[int] | np.ndarray,
    group_ids: Sequence[object] | np.ndarray,
    oof_probabilities: Mapping[str, Sequence[float] | np.ndarray],
    *,
    bootstrap_repeats: int = 5000,
) -> dict[str, dict[str, object]]:
    """Evaluate four fixed arithmetic ensembles in an explicitly adaptive round."""
    label_array = np.asarray(labels)
    groups = np.asarray(group_ids, dtype=object)
    if (
        label_array.ndim != 1 or groups.shape != label_array.shape
        or label_array.dtype.kind not in {"i", "u"}
        or not np.isin(label_array, (0, 1)).all()
    ):
        raise ValueError("ensemble labels and groups must be aligned binary vectors")
    if isinstance(bootstrap_repeats, bool) or not isinstance(bootstrap_repeats, int) or bootstrap_repeats < 1:
        raise ValueError("bootstrap_repeats must be a positive integer")
    required = {candidate for _, recipe in _ENSEMBLE_RECIPES for candidate in recipe}
    if not required.issubset(oof_probabilities):
        raise ValueError("ensemble round is missing a required base candidate")
    aligned: dict[str, np.ndarray] = {}
    for candidate in required:
        values = np.asarray(oof_probabilities[candidate], dtype=np.float64)
        if values.shape != label_array.shape or not np.isfinite(values).all():
            raise ValueError("ensemble OOF probabilities must be finite and aligned")
        if np.any((values < 0) | (values > 1)):
            raise ValueError("ensemble inputs must be probabilities")
        aligned[candidate] = values
    group_labels, logistic_groups = _aggregate_groups(
        label_array, aligned["logistic_110d"], groups
    )
    logistic_point = _binary_metrics(group_labels, logistic_groups)
    class_rows = {label: np.flatnonzero(group_labels == label) for label in (0, 1)}
    if any(rows.size == 0 for rows in class_rows.values()):
        raise ValueError("ensemble bootstrap requires both group-level classes")
    output: dict[str, dict[str, object]] = {}
    for recipe_index, (name, recipe) in enumerate(_ENSEMBLE_RECIPES):
        recording_probabilities = np.mean(
            np.stack([aligned[candidate] for candidate in recipe], axis=0), axis=0
        )
        candidate_labels, group_probabilities = _aggregate_groups(
            label_array, recording_probabilities, groups
        )
        if not np.array_equal(candidate_labels, group_labels):
            raise AssertionError("ensemble group alignment drifted")
        point = _binary_metrics(group_labels, group_probabilities)
        rng = np.random.default_rng(20260811 + recipe_index)
        deltas = {
            "auroc": np.empty(bootstrap_repeats),
            "balanced_accuracy": np.empty(bootstrap_repeats),
            "brier": np.empty(bootstrap_repeats),
        }
        for repeat in range(bootstrap_repeats):
            sampled = np.concatenate([
                rng.choice(rows, size=rows.size, replace=True)
                for rows in class_rows.values()
            ])
            ensemble_metrics = _binary_metrics(
                group_labels[sampled], group_probabilities[sampled]
            )
            baseline_metrics = _binary_metrics(
                group_labels[sampled], logistic_groups[sampled]
            )
            for metric in deltas:
                deltas[metric][repeat] = (
                    ensemble_metrics[metric] - baseline_metrics[metric]
                )
        output[name] = {
            **point,
            "components": list(recipe),
            "adaptive_after_base_screen": True,
            "bootstrap": {
                "repeats": bootstrap_repeats,
                **{
                    f"{metric}_delta_vs_logistic": {
                        "point": point[metric] - logistic_point[metric],
                        "ci95": [float(value) for value in np.quantile(
                            values, (0.025, 0.975)
                        )],
                    }
                    for metric, values in deltas.items()
                },
            },
        }
    return output


def _validate_search_dataset(dataset: SearchDataset) -> None:
    if not isinstance(dataset, SearchDataset):
        raise ValueError("dataset must be SearchDataset")
    raw = np.asarray(dataset.raw_features)
    mirrored = np.asarray(dataset.mirrored_raw_features)
    masks = np.asarray(dataset.valid_masks)
    summaries = np.asarray(dataset.summary_features)
    mirrored_summaries = np.asarray(dataset.mirrored_summary_features)
    labels = np.asarray(dataset.labels)
    groups = np.asarray(dataset.group_ids, dtype=object)
    folds = np.asarray(dataset.inner_fold_by_index)
    n = raw.shape[0] if raw.ndim else 0
    if raw.shape != (n, 4, 32, 95) or mirrored.shape != raw.shape:
        raise ValueError("raw and mirrored features must have shape (N, 4, 32, 95)")
    if raw.dtype != np.float32 or mirrored.dtype != np.float32:
        raise ValueError("raw and mirrored features must be float32")
    if masks.shape != (n, 4, 32) or masks.dtype != np.dtype(bool):
        raise ValueError("valid masks must be bool with shape (N, 4, 32)")
    if summaries.shape != (n, 110) or mirrored_summaries.shape != summaries.shape:
        raise ValueError("summary and mirrored summary features must be (N, 110)")
    if labels.shape != (n,) or labels.dtype.kind not in {"i", "u"}:
        raise ValueError("labels must be one-dimensional integer values")
    if groups.shape != (n,) or folds.shape != (n,) or folds.dtype.kind not in {"i", "u"}:
        raise ValueError("groups and fold registry must align with recordings")
    development = _index_array(dataset.development_indices, "development_indices")
    protected = _index_array(dataset.protected_indices, "protected_indices")
    if set(development.tolist()) & set(protected.tolist()):
        raise ValueError("development and protected indices overlap")
    if set(development.tolist()) | set(protected.tolist()) != set(range(n)):
        raise ValueError("development and protected indices must partition the cache")
    require_development_only(development, development, protected, "validate")
    # Values are deliberately checked only after selecting development rows.
    for values, name in (
        (raw[development][masks[development]], "raw"),
        (mirrored[development][masks[development]], "mirrored raw"),
        (summaries[development], "summary"),
        (mirrored_summaries[development], "mirrored summary"),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"development {name} features must be finite")
    if not np.isin(labels[development], (0, 1)).all():
        raise ValueError("development labels must be binary")
    if set(folds[development].tolist()) != {0, 1, 2, 3}:
        raise ValueError("development records must cover exactly four inner folds")
    if any(not isinstance(group, str) or not group for group in groups[development].tolist()):
        raise ValueError("development groups must be nonempty strings")
    for group in set(groups[development].tolist()):
        group_rows = development[groups[development] == group]
        if len(set(labels[group_rows].tolist())) != 1:
            raise ValueError("one development group cannot cross binary labels")
        if len(set(folds[group_rows].tolist())) != 1:
            raise ValueError("one development group cannot cross inner folds")
    for fold in range(4):
        validation = development[folds[development] == fold]
        training = development[folds[development] != fold]
        if not validation.size or set(labels[training].tolist()) != {0, 1}:
            raise ValueError("each fold needs validation rows and both training classes")
        if set(groups[training].tolist()) & set(groups[validation].tolist()):
            raise ValueError("inner folds must be group disjoint")


def _aggregate_groups(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    group_labels: list[int] = []
    group_probabilities: list[float] = []
    for group in sorted(set(groups.tolist())):
        rows = np.flatnonzero(groups == group)
        values = set(labels[rows].tolist())
        if len(values) != 1:
            raise ValueError("group aggregation found inconsistent labels")
        group_labels.append(int(next(iter(values))))
        group_probabilities.append(float(np.mean(probabilities[rows])))
    return np.asarray(group_labels, dtype=np.int64), np.asarray(group_probabilities)


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("binary metrics require both classes")
    predictions = probabilities >= 0.5
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity": float(np.mean(predictions[labels == 1])),
        "specificity": float(np.mean(~predictions[labels == 0])),
    }


def _classical_model(name: str, seed: int):
    if name == "logistic_110d":
        return LogisticRegression(
            C=0.01, penalty="l2", solver="liblinear", max_iter=2000,
            random_state=seed,
        ), 111
    if name == "extra_trees_110d":
        return ExtraTreesClassifier(
            n_estimators=256, max_depth=4, min_samples_leaf=2,
            max_features="sqrt", class_weight=None, random_state=seed, n_jobs=1,
        ), 256
    if name == "hist_gradient_boosting_110d":
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=100, max_leaf_nodes=15,
            min_samples_leaf=4, l2_regularization=1.0,
            early_stopping=False, random_state=seed,
        ), 1500
    raise ValueError(f"unknown classical candidate {name!r}")


def _fit_classical_fold(
    name: str,
    dataset: SearchDataset,
    train: np.ndarray,
    validation: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, int]:
    require_development_only(
        train, dataset.development_indices, dataset.protected_indices, "classical fit"
    )
    require_development_only(
        validation, dataset.development_indices, dataset.protected_indices,
        "classical predict",
    )
    train_summary = np.concatenate((
        dataset.summary_features[train], dataset.mirrored_summary_features[train]
    ))
    train_labels = np.concatenate((dataset.labels[train], dataset.labels[train]))
    train_groups = np.concatenate((dataset.group_ids[train], dataset.group_ids[train]))
    scaler = StandardScaler().fit(train_summary)
    model, parameter_count = _classical_model(name, seed)
    model.fit(
        scaler.transform(train_summary), train_labels,
        sample_weight=group_balanced_weights(train_groups),
    )
    original = model.predict_proba(
        scaler.transform(dataset.summary_features[validation])
    )[:, 1]
    mirrored = model.predict_proba(
        scaler.transform(dataset.mirrored_summary_features[validation])
    )[:, 1]
    return 0.5 * (original + mirrored), parameter_count


def _logistic_oof_for_assignment(
    dataset: SearchDataset,
    fold_assignment: np.ndarray,
) -> dict[str, float]:
    development = np.asarray(dataset.development_indices, dtype=np.int64)
    assignment = np.asarray(fold_assignment)
    if assignment.shape != (development.size,) or assignment.dtype.kind not in {"i", "u"}:
        raise ValueError("stability fold assignment must align with development rows")
    if set(assignment.tolist()) != {0, 1, 2, 3}:
        raise ValueError("stability audit requires exactly four folds")
    require_development_only(
        development, dataset.development_indices, dataset.protected_indices,
        "stability audit",
    )
    probabilities = np.full(development.size, np.nan, dtype=np.float64)
    for fold in range(4):
        train_local = np.flatnonzero(assignment != fold)
        validation_local = np.flatnonzero(assignment == fold)
        train = development[train_local]
        validation = development[validation_local]
        if set(dataset.labels[train].tolist()) != {0, 1}:
            raise ValueError("stability training fold lost one class")
        fold_probabilities, _parameters = _fit_classical_fold(
            "logistic_110d", dataset, train, validation, seed=0
        )
        probabilities[validation_local] = fold_probabilities
    if not np.isfinite(probabilities).all():
        raise RuntimeError("stability audit did not fill all OOF probabilities")
    group_labels, group_probabilities = _aggregate_groups(
        dataset.labels[development], probabilities, dataset.group_ids[development]
    )
    return _binary_metrics(group_labels, group_probabilities)


def _distribution_summary(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("stability metric distribution must be finite and nonempty")
    quantiles = np.quantile(array, (0.05, 0.5, 0.95))
    return {
        "minimum": float(array.min()),
        "q05": float(quantiles[0]),
        "median": float(quantiles[1]),
        "q95": float(quantiles[2]),
        "maximum": float(array.max()),
    }


def run_logistic_stability_audit(
    dataset: SearchDataset,
    *,
    repeats: int = 50,
    permutations: int = 500,
    seed: int = 20260811,
) -> dict[str, object]:
    """Stress-test the frozen winner across splits and a group-label null."""
    _validate_search_dataset(dataset)
    if isinstance(permutations, bool) or not isinstance(permutations, int) or permutations < 1:
        raise ValueError("permutations must be a positive integer")
    development = np.asarray(dataset.development_indices, dtype=np.int64)
    development_labels = np.asarray(dataset.labels[development], dtype=np.int64)
    development_groups = np.asarray(dataset.group_ids[development], dtype=object)
    repeated_assignments = build_repeated_stratified_group_folds(
        development_labels, development_groups, repeats=repeats, seed=seed
    )
    repeated_metrics = [
        _logistic_oof_for_assignment(dataset, assignment)
        for assignment in repeated_assignments
    ]
    fixed_assignment = np.asarray(
        dataset.inner_fold_by_index[development], dtype=np.int64
    )
    observed = _logistic_oof_for_assignment(dataset, fixed_assignment)

    group_names = sorted(set(development_groups.tolist()))
    label_by_group = {
        group: int(development_labels[development_groups == group][0])
        for group in group_names
    }
    base_group_labels = np.asarray(
        [label_by_group[group] for group in group_names], dtype=np.int64
    )
    rng = np.random.default_rng(seed + 1)
    null_aurocs: list[float] = []
    attempts = 0
    while len(null_aurocs) < permutations:
        attempts += 1
        if attempts > permutations * 20:
            raise RuntimeError("could not construct enough valid group-label permutations")
        shuffled = rng.permutation(base_group_labels)
        permuted_by_group = dict(zip(group_names, shuffled.tolist()))
        permuted_development_labels = np.asarray(
            [permuted_by_group[str(group)] for group in development_groups.tolist()],
            dtype=np.int64,
        )
        if any(
            set(permuted_development_labels[fixed_assignment != fold].tolist()) != {0, 1}
            for fold in range(4)
        ):
            continue
        full_labels = np.asarray(dataset.labels, dtype=np.int64).copy()
        full_labels[development] = permuted_development_labels
        permuted_dataset = replace(dataset, labels=full_labels)
        null_aurocs.append(
            _logistic_oof_for_assignment(permuted_dataset, fixed_assignment)["auroc"]
        )
    null_array = np.asarray(null_aurocs, dtype=np.float64)
    return {
        "schema_version": "logistic_110d_stability_audit_v1",
        "repeated_group_cv": {
            "repeats": repeats,
            "folds": 4,
            "auroc": {
                **_distribution_summary([value["auroc"] for value in repeated_metrics]),
                "fraction_at_least_0_95": float(np.mean([
                    value["auroc"] >= 0.95 for value in repeated_metrics
                ])),
            },
            "balanced_accuracy": _distribution_summary([
                value["balanced_accuracy"] for value in repeated_metrics
            ]),
            "brier": _distribution_summary([
                value["brier"] for value in repeated_metrics
            ]),
        },
        "group_label_permutation": {
            "permutations": permutations,
            "observed_fixed_fold_auroc": observed["auroc"],
            "null_auroc_mean": float(null_array.mean()),
            "null_auroc_q95": float(np.quantile(null_array, 0.95)),
            "p_value": float(
                (1 + np.sum(null_array >= observed["auroc"])) / (permutations + 1)
            ),
        },
        "protected_predictions": 0,
    }


@dataclass(frozen=True)
class _TensorStandardizers:
    raw_mean: torch.Tensor
    raw_scale: torch.Tensor
    summary_mean: torch.Tensor
    summary_scale: torch.Tensor


def _fit_tensor_standardizers(dataset: SearchDataset, train: np.ndarray) -> _TensorStandardizers:
    require_development_only(
        train, dataset.development_indices, dataset.protected_indices, "neural scale"
    )
    raw = np.concatenate((
        dataset.raw_features[train], dataset.mirrored_raw_features[train]
    ))
    mask = np.concatenate((dataset.valid_masks[train], dataset.valid_masks[train]))
    rows = raw[mask].astype(np.float64)
    raw_mean = rows.mean(axis=0)
    raw_scale = rows.std(axis=0)
    raw_scale[raw_scale <= 1e-6] = 1.0
    summary = np.concatenate((
        dataset.summary_features[train], dataset.mirrored_summary_features[train]
    )).astype(np.float64)
    summary_mean = summary.mean(axis=0)
    summary_scale = summary.std(axis=0)
    summary_scale[summary_scale <= 1e-6] = 1.0
    return _TensorStandardizers(*(
        torch.as_tensor(value, dtype=torch.float32)
        for value in (raw_mean, raw_scale, summary_mean, summary_scale)
    ))


def _standardized_tensors(
    dataset: SearchDataset,
    indices: np.ndarray,
    standardizers: _TensorStandardizers,
    *,
    mirrored: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_raw = dataset.mirrored_raw_features if mirrored else dataset.raw_features
    source_summary = (
        dataset.mirrored_summary_features if mirrored else dataset.summary_features
    )
    raw = torch.as_tensor(source_raw[indices], dtype=torch.float32, device=device)
    mask = torch.as_tensor(dataset.valid_masks[indices], dtype=torch.bool, device=device)
    summary = torch.as_tensor(source_summary[indices], dtype=torch.float32, device=device)
    raw_mean = standardizers.raw_mean.to(device)
    raw_scale = standardizers.raw_scale.to(device)
    summary_mean = standardizers.summary_mean.to(device)
    summary_scale = standardizers.summary_scale.to(device)
    raw = (raw - raw_mean) / raw_scale
    raw = torch.where(mask.unsqueeze(-1), raw, torch.zeros_like(raw))
    summary = (summary - summary_mean) / summary_scale
    return raw, mask, summary


def _weighted_bce(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    losses = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (losses * weights).sum() / weights.sum().clamp_min(torch.finfo(losses.dtype).eps)


def _fit_neural_fold(
    name: str,
    dataset: SearchDataset,
    train: np.ndarray,
    validation: np.ndarray,
    *,
    seed: int,
    config: SearchConfig,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    require_development_only(
        train, dataset.development_indices, dataset.protected_indices, "neural fit"
    )
    require_development_only(
        validation, dataset.development_indices, dataset.protected_indices,
        "neural predict",
    )
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    standardizers = _fit_tensor_standardizers(dataset, train)
    original_train = _standardized_tensors(
        dataset, train, standardizers, mirrored=False, device=device
    )
    mirrored_train = _standardized_tensors(
        dataset, train, standardizers, mirrored=True, device=device
    )
    train_raw = torch.cat((original_train[0], mirrored_train[0]))
    train_mask = torch.cat((original_train[1], mirrored_train[1]))
    train_summary = torch.cat((original_train[2], mirrored_train[2]))
    labels = torch.as_tensor(
        np.concatenate((dataset.labels[train], dataset.labels[train])),
        dtype=torch.float32, device=device,
    )
    weights = torch.as_tensor(
        group_balanced_weights(np.concatenate((
            dataset.group_ids[train], dataset.group_ids[train]
        ))), dtype=torch.float32, device=device,
    )
    val_original = _standardized_tensors(
        dataset, validation, standardizers, mirrored=False, device=device
    )
    val_mirrored = _standardized_tensors(
        dataset, validation, standardizers, mirrored=True, device=device
    )
    val_labels = torch.as_tensor(
        dataset.labels[validation], dtype=torch.float32, device=device
    )
    val_weights = torch.as_tensor(
        group_balanced_weights(dataset.group_ids[validation]),
        dtype=torch.float32, device=device,
    )
    model = build_neural_candidate(name).to(device)
    parameter_count = count_trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for _epoch in range(config.effective_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_raw, train_mask, train_summary)
        loss = _weighted_bce(logits, labels, weights)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name} training loss became nonfinite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            original_logits = model(*val_original)
            mirrored_logits = model(*val_mirrored)
            mean_probability = 0.5 * (
                torch.sigmoid(original_logits) + torch.sigmoid(mirrored_logits)
            )
            eps = torch.finfo(mean_probability.dtype).eps
            mean_logit = torch.logit(mean_probability.clamp(eps, 1 - eps))
            val_loss = float(_weighted_bce(mean_logit, val_labels, val_weights).item())
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError(f"{name} did not produce a finite validation state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = 0.5 * (
            torch.sigmoid(model(*val_original)) + torch.sigmoid(model(*val_mirrored))
        )
    result = probabilities.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise RuntimeError(f"{name} produced invalid probabilities")
    return result, parameter_count


def run_screening(
    dataset: SearchDataset,
    *,
    config: SearchConfig | None = None,
    candidates: Sequence[str] | None = None,
    device: str | torch.device = "cpu",
) -> ScreeningResult:
    """Run aligned four-fold OOF screening without exposing protected predictions."""
    config = config or SearchConfig()
    _validate_search_dataset(dataset)
    requested = tuple(candidates) if candidates is not None else CANDIDATE_ORDER
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("screening candidates must be unique and nonempty")
    if any(name not in CANDIDATE_ORDER for name in requested):
        raise ValueError("screening contains an unregistered candidate")
    if not config.smoke and requested != CANDIDATE_ORDER:
        raise ValueError("full screening must run the complete frozen registry")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA screening requested but CUDA is unavailable")
    development = np.asarray(dataset.development_indices, dtype=np.int64)
    folds = np.asarray(dataset.inner_fold_by_index)
    labels = np.asarray(dataset.labels)
    groups = np.asarray(dataset.group_ids, dtype=object)
    candidate_metrics: dict[str, dict[str, object]] = {}
    candidate_fold_metrics: dict[str, tuple[Mapping[str, float], ...]] = {}
    candidate_oof_probabilities: dict[str, np.ndarray] = {}
    for name in requested:
        started = time.perf_counter()
        oof = np.full(labels.shape[0], np.nan, dtype=np.float64)
        fold_metrics: list[Mapping[str, float]] = []
        parameter_count: int | None = None
        for fold in range(4):
            validation = development[folds[development] == fold]
            train = development[folds[development] != fold]
            if name in CLASSICAL_CANDIDATES:
                probabilities, observed_parameters = _fit_classical_fold(
                    name, dataset, train, validation, config.screen_seed
                )
            elif name in NEURAL_CANDIDATES:
                probabilities, observed_parameters = _fit_neural_fold(
                    name, dataset, train, validation, seed=config.screen_seed,
                    config=config, device=torch_device,
                )
            else:
                raise AssertionError("candidate registry dispatch drifted")
            if parameter_count is None:
                parameter_count = observed_parameters
            elif parameter_count != observed_parameters:
                raise RuntimeError("candidate parameter count changed between folds")
            oof[validation] = probabilities
            fold_labels, fold_probabilities = _aggregate_groups(
                labels[validation], probabilities, groups[validation]
            )
            fold_metrics.append(_binary_metrics(fold_labels, fold_probabilities))
        if not np.isfinite(oof[development]).all():
            raise RuntimeError(f"{name} did not cover every development record exactly once")
        if np.isfinite(oof[np.asarray(dataset.protected_indices, dtype=np.int64)]).any():
            raise OuterSearchLockedError("protected prediction appeared in OOF storage")
        group_labels, group_probabilities = _aggregate_groups(
            labels[development], oof[development], groups[development]
        )
        metrics: dict[str, object] = _binary_metrics(group_labels, group_probabilities)
        metrics.update({
            "parameter_count": int(parameter_count or 0),
            "oof_recordings": int(development.size),
            "oof_groups": int(group_labels.size),
            "elapsed_seconds": float(time.perf_counter() - started),
        })
        candidate_metrics[name] = metrics
        candidate_fold_metrics[name] = tuple(fold_metrics)
        candidate_oof_probabilities[name] = oof[development].copy()
    winner = select_screening_winner(candidate_metrics)
    return ScreeningResult(
        candidate_metrics=candidate_metrics,
        candidate_fold_metrics=candidate_fold_metrics,
        winner=winner,
        development_recordings=int(development.size),
        development_groups=len(set(groups[development].tolist())),
        protected_predictions=0,
        candidate_oof_probabilities=candidate_oof_probabilities,
    )


def run_confirmation(
    dataset: SearchDataset,
    *,
    winner: str,
    config: SearchConfig | None = None,
    device: str | torch.device = "cpu",
) -> ConfirmationResult:
    """Re-run only the selected family at the exact three confirmation seeds."""
    config = config or SearchConfig()
    _validate_search_dataset(dataset)
    if not isinstance(winner, str) or winner not in CANDIDATE_ORDER:
        raise ValueError("confirmation winner must be a registered candidate")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA confirmation requested but CUDA is unavailable")
    development = np.asarray(dataset.development_indices, dtype=np.int64)
    folds = np.asarray(dataset.inner_fold_by_index)
    labels = np.asarray(dataset.labels)
    groups = np.asarray(dataset.group_ids, dtype=object)
    probabilities_by_seed: list[np.ndarray] = []
    seed_metrics: dict[int, Mapping[str, float]] = {}
    parameter_count: int | None = None
    for seed in config.confirmation_seeds:
        oof = np.full(labels.shape[0], np.nan, dtype=np.float64)
        for fold in range(4):
            validation = development[folds[development] == fold]
            train = development[folds[development] != fold]
            if winner in CLASSICAL_CANDIDATES:
                probabilities, observed_parameters = _fit_classical_fold(
                    winner, dataset, train, validation, seed
                )
            else:
                probabilities, observed_parameters = _fit_neural_fold(
                    winner, dataset, train, validation, seed=seed,
                    config=config, device=torch_device,
                )
            if parameter_count is None:
                parameter_count = observed_parameters
            elif parameter_count != observed_parameters:
                raise RuntimeError("confirmation parameter count changed across fits")
            oof[validation] = probabilities
        if not np.isfinite(oof[development]).all():
            raise RuntimeError("confirmation did not cover every development record")
        if np.isfinite(oof[np.asarray(dataset.protected_indices, dtype=np.int64)]).any():
            raise OuterSearchLockedError("confirmation produced a protected prediction")
        seed_labels, seed_probabilities = _aggregate_groups(
            labels[development], oof[development], groups[development]
        )
        seed_metrics[seed] = _binary_metrics(seed_labels, seed_probabilities)
        probabilities_by_seed.append(oof[development].copy())
    ensemble_recording_probabilities = np.mean(
        np.stack(probabilities_by_seed, axis=0), axis=0
    )
    ensemble_labels, ensemble_probabilities = _aggregate_groups(
        labels[development], ensemble_recording_probabilities, groups[development]
    )
    return ConfirmationResult(
        winner=winner,
        seed_metrics=seed_metrics,
        ensemble_metrics=_binary_metrics(ensemble_labels, ensemble_probabilities),
        parameter_count=int(parameter_count or 0),
        protected_predictions=0,
    )


__all__ = [
    "ConfirmationResult", "OuterSearchLockedError", "SearchConfig", "SearchDataset",
    "ScreeningResult",
    "candidate_rank_key",
    "evaluate_fixed_ensembles", "group_balanced_weights", "require_development_only",
    "run_confirmation", "run_screening", "select_screening_winner",
]
