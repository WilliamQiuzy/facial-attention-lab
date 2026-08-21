"""Closed PalsyNet development evaluation for Source-Robust 110D v1."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..preprocessing.source_robust_110d import (
    CANDIDATE_DIMENSIONS,
    CANDIDATE_ORDER,
)


FIXED_C = 0.01
FIXED_THRESHOLD = 0.5
INNER_FOLDS = 4
REGISTERED_AUROC_TOLERANCE = 0.02
REGISTERED_BALANCED_ACCURACY_TOLERANCE = 1.0 / (2.0 * 21.0)
# Excludes duration and frame-difference motion, retaining acquisition only.
ACQUISITION_NUISANCE_INDICES = (1, 2, 3, 5, 6, 7, 8)


@dataclass(frozen=True)
class SourceRobustOOFResult:
    probabilities: Mapping[str, np.ndarray]
    audit: Mapping[str, int]


@dataclass(frozen=True)
class SourceRobustDecision:
    locked_candidate: str
    gates: Mapping[str, Mapping[str, bool]]


def _validated_labels_groups(
    labels: np.ndarray, group_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels)
    groups = np.asarray(group_ids)
    if (
        y.ndim != 1 or y.dtype.kind not in {"i", "u"}
        or groups.shape != y.shape or set(y.tolist()) != {0, 1}
    ):
        raise ValueError("binary labels and group IDs must be aligned vectors")
    for group in set(groups.tolist()):
        if np.unique(y[groups == group]).size != 1:
            raise ValueError("one reviewed group cannot cross labels")
    return y.astype(np.int64, copy=False), groups


def build_acquisition_blocked_folds(
    labels: np.ndarray,
    group_ids: np.ndarray,
    nuisance: np.ndarray,
) -> np.ndarray:
    """Create four label-stratified contiguous blocks along acquisition PC1."""
    y, groups = _validated_labels_groups(labels, group_ids)
    values = np.asarray(nuisance, dtype=np.float64)
    if values.shape != (y.size, 9) or not np.isfinite(values).all():
        raise ValueError("nuisance must be a finite aligned (N, 9) matrix")

    ordered_groups = sorted(set(groups.tolist()), key=str)
    group_labels = np.empty(len(ordered_groups), dtype=np.int64)
    group_nuisance = np.empty(
        (len(ordered_groups), len(ACQUISITION_NUISANCE_INDICES)),
        dtype=np.float64,
    )
    for index, group in enumerate(ordered_groups):
        rows = np.flatnonzero(groups == group)
        group_labels[index] = int(y[rows[0]])
        group_nuisance[index] = np.mean(
            values[rows][:, ACQUISITION_NUISANCE_INDICES], axis=0
        )
    means = np.mean(group_nuisance, axis=0)
    scales = np.std(group_nuisance, axis=0, ddof=0)
    scales[scales <= np.finfo(np.float64).eps] = 1.0
    standardized = (group_nuisance - means) / scales
    _left, _singular, right = np.linalg.svd(standardized, full_matrices=False)
    loading = right[0].copy()
    anchor = int(np.argmax(np.abs(loading)))
    if loading[anchor] < 0:
        loading *= -1.0
    scores = standardized @ loading

    group_fold: dict[object, int] = {}
    for label in (0, 1):
        members = np.flatnonzero(group_labels == label)
        if members.size < INNER_FOLDS:
            raise ValueError("each label needs at least four reviewed groups")
        ordered = sorted(
            members.tolist(),
            key=lambda index: (float(scores[index]), str(ordered_groups[index])),
        )
        for fold, block in enumerate(np.array_split(ordered, INNER_FOLDS)):
            if len(block) == 0:
                raise ValueError("acquisition block cannot be empty")
            for index in block:
                group_fold[ordered_groups[int(index)]] = fold
    folds = np.asarray([group_fold[group] for group in groups.tolist()], dtype=np.int64)
    for fold in range(INNER_FOLDS):
        if set(y[folds == fold].tolist()) != {0, 1}:
            raise ValueError("every acquisition block must contain both labels")
    return folds


def _group_balanced_weights(groups: np.ndarray) -> np.ndarray:
    values = groups.tolist()
    counts = {group: values.count(group) for group in set(values)}
    return np.asarray([1.0 / counts[group] for group in values], dtype=np.float64)


def run_candidate_oof(
    *,
    labels: np.ndarray,
    group_ids: np.ndarray,
    folds: np.ndarray,
    original: Mapping[str, np.ndarray],
    mirrored: Mapping[str, np.ndarray],
) -> SourceRobustOOFResult:
    """Fit every locked candidate once per group-disjoint fold."""
    y, groups = _validated_labels_groups(labels, group_ids)
    fold_values = np.asarray(folds)
    if (
        fold_values.shape != y.shape or fold_values.dtype.kind not in {"i", "u"}
        or set(fold_values.tolist()) != set(range(INNER_FOLDS))
        or tuple(original) != CANDIDATE_ORDER or tuple(mirrored) != CANDIDATE_ORDER
    ):
        raise ValueError("folds and candidate registry must match the closed protocol")
    for group in set(groups.tolist()):
        if np.unique(fold_values[groups == group]).size != 1:
            raise ValueError("one reviewed group cannot cross folds")

    checked: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for candidate in CANDIDATE_ORDER:
        first = np.asarray(original[candidate], dtype=np.float64)
        second = np.asarray(mirrored[candidate], dtype=np.float64)
        expected = (y.size, CANDIDATE_DIMENSIONS[candidate])
        if (
            first.shape != expected or second.shape != expected
            or not np.isfinite(first).all() or not np.isfinite(second).all()
        ):
            raise ValueError("candidate matrices differ from the frozen schema")
        checked[candidate] = (first, second)

    probabilities: dict[str, np.ndarray] = {}
    fits = predictions = 0
    for candidate in CANDIDATE_ORDER:
        first, second = checked[candidate]
        oof = np.full(y.size, np.nan, dtype=np.float64)
        counts = np.zeros(y.size, dtype=np.int64)
        for fold in range(INNER_FOLDS):
            train = np.flatnonzero(fold_values != fold)
            valid = np.flatnonzero(fold_values == fold)
            if set(y[train].tolist()) != {0, 1} or not valid.size:
                raise ValueError("every fold needs train classes and validation rows")
            if set(groups[train].tolist()) & set(groups[valid].tolist()):
                raise ValueError("folds are not group disjoint")
            x_train = np.concatenate((first[train], second[train]))
            y_train = np.concatenate((y[train], y[train]))
            train_groups = np.concatenate((groups[train], groups[train]))
            scaler = StandardScaler().fit(x_train)
            model = LogisticRegression(
                C=FIXED_C, penalty="l2", solver="liblinear",
                max_iter=2000, random_state=0,
            )
            model.fit(
                scaler.transform(x_train), y_train,
                sample_weight=_group_balanced_weights(train_groups),
            )
            oof[valid] = 0.5 * (
                model.predict_proba(scaler.transform(first[valid]))[:, 1]
                + model.predict_proba(scaler.transform(second[valid]))[:, 1]
            )
            counts[valid] += 1
            fits += 1
            predictions += int(valid.size)
        if not np.isfinite(oof).all() or not np.all(counts == 1):
            raise RuntimeError("OOF predictions must cover every row exactly once")
        probabilities[candidate] = oof
    return SourceRobustOOFResult(
        probabilities=probabilities,
        audit={
            "development_scaler_fits": fits,
            "development_model_fits": fits,
            "development_predictions": predictions,
            "protected_feature_reads": 0,
            "protected_fits": 0,
            "protected_predictions": 0,
        },
    )


def choose_source_robust_candidate(
    metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> SourceRobustDecision:
    """Apply the preregistered noninferiority plus blocked-improvement gate."""
    protocols = ("registered", "acquisition_blocked")
    if tuple(metrics) != protocols or any(
        tuple(metrics[protocol]) != CANDIDATE_ORDER for protocol in protocols
    ):
        raise ValueError("source-robust metrics differ from the closed registry")
    baseline_registered = metrics[protocols[0]][CANDIDATE_ORDER[0]]
    baseline_blocked = metrics[protocols[1]][CANDIDATE_ORDER[0]]
    required = {"auroc", "balanced_accuracy", "specificity"}
    if any(
        not required.issubset(metrics[protocol][candidate])
        for protocol in protocols for candidate in CANDIDATE_ORDER
    ):
        raise ValueError("source-robust locking metrics are incomplete")

    gates: dict[str, dict[str, bool]] = {}
    passing: list[str] = []
    for candidate in CANDIDATE_ORDER[1:]:
        registered = metrics[protocols[0]][candidate]
        blocked = metrics[protocols[1]][candidate]
        candidate_gates = {
            "registered_auroc_noninferior": bool(
                float(registered["auroc"])
                >= float(baseline_registered["auroc"]) - REGISTERED_AUROC_TOLERANCE
            ),
            "registered_balanced_accuracy_noninferior": bool(
                float(registered["balanced_accuracy"])
                >= float(baseline_registered["balanced_accuracy"])
                - REGISTERED_BALANCED_ACCURACY_TOLERANCE
            ),
            "registered_specificity_one": bool(
                float(registered["specificity"]) == 1.0
            ),
            "blocked_auroc_improved": bool(
                float(blocked["auroc"]) > float(baseline_blocked["auroc"])
            ),
            "blocked_balanced_accuracy_improved": bool(
                float(blocked["balanced_accuracy"])
                > float(baseline_blocked["balanced_accuracy"])
            ),
        }
        candidate_gates["passed"] = all(candidate_gates.values())
        gates[candidate] = candidate_gates
        if candidate_gates["passed"]:
            passing.append(candidate)

    locked = CANDIDATE_ORDER[0]
    if len(passing) == 1:
        locked = passing[0]
    elif len(passing) > 1:
        dominant = []
        for candidate in passing:
            score = metrics[protocols[1]][candidate]
            if all(
                float(score["auroc"]) > float(metrics[protocols[1]][other]["auroc"])
                and float(score["balanced_accuracy"])
                > float(metrics[protocols[1]][other]["balanced_accuracy"])
                for other in passing if other != candidate
            ):
                dominant.append(candidate)
        if len(dominant) == 1:
            locked = dominant[0]
    return SourceRobustDecision(locked_candidate=locked, gates=gates)


__all__ = [
    "ACQUISITION_NUISANCE_INDICES",
    "FIXED_C",
    "FIXED_THRESHOLD",
    "SourceRobustDecision",
    "SourceRobustOOFResult",
    "build_acquisition_blocked_folds",
    "choose_source_robust_candidate",
    "run_candidate_oof",
]
