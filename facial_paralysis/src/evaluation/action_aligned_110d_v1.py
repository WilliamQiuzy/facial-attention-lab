"""Frozen development comparison for four-time versus seven-action 110D."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CANDIDATE_ORDER = ("four_time_window_110d", "seven_action_window_110d")
FIXED_C = 0.01
FIXED_THRESHOLD = 0.5
INNER_FOLDS = 4


@dataclass(frozen=True)
class ActionAlignedOOFResult:
    probabilities: Mapping[str, np.ndarray]
    audit: Mapping[str, int]


def _group_balanced_weights(group_ids: np.ndarray) -> np.ndarray:
    values = np.asarray(group_ids).tolist()
    counts = {group: values.count(group) for group in set(values)}
    return np.asarray([1.0 / counts[group] for group in values], dtype=np.float64)


def _validated_inputs(
    labels: np.ndarray,
    group_ids: np.ndarray,
    inner_folds: np.ndarray,
    original: Mapping[str, np.ndarray],
    mirrored: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    y = np.asarray(labels)
    groups = np.asarray(group_ids)
    folds = np.asarray(inner_folds)
    if (
        y.ndim != 1 or y.dtype.kind not in {"i", "u"}
        or groups.shape != y.shape or folds.shape != y.shape
        or set(y.tolist()) != {0, 1} or set(folds.tolist()) != set(range(INNER_FOLDS))
    ):
        raise ValueError("labels, groups, and four inner folds must align")
    for group in set(groups.tolist()):
        indices = np.flatnonzero(groups == group)
        if np.unique(y[indices]).size != 1 or np.unique(folds[indices]).size != 1:
            raise ValueError("one identity group cannot cross labels or folds")
    if tuple(original) != CANDIDATE_ORDER or tuple(mirrored) != CANDIDATE_ORDER:
        raise ValueError("candidate registry/order drifted")
    checked_original: dict[str, np.ndarray] = {}
    checked_mirrored: dict[str, np.ndarray] = {}
    for candidate in CANDIDATE_ORDER:
        first = np.asarray(original[candidate], dtype=np.float64)
        second = np.asarray(mirrored[candidate], dtype=np.float64)
        if (
            first.shape != (y.size, 110) or second.shape != first.shape
            or not np.isfinite(first).all() or not np.isfinite(second).all()
        ):
            raise ValueError("candidate matrices must align as finite (N, 110)")
        checked_original[candidate] = first
        checked_mirrored[candidate] = second
    return y.astype(np.int64, copy=False), groups, folds, checked_original, checked_mirrored


def run_group_disjoint_oof(
    *,
    labels: np.ndarray,
    group_ids: np.ndarray,
    inner_folds: np.ndarray,
    original: Mapping[str, np.ndarray],
    mirrored: Mapping[str, np.ndarray],
) -> ActionAlignedOOFResult:
    """Run the fixed C=0.01 mirror-trained Logistic in four group folds."""
    y, groups, folds, originals, mirrors = _validated_inputs(
        labels, group_ids, inner_folds, original, mirrored
    )
    probabilities: dict[str, np.ndarray] = {}
    fits = predictions = 0
    for candidate in CANDIDATE_ORDER:
        oof = np.full(y.size, np.nan, dtype=np.float64)
        counts = np.zeros(y.size, dtype=np.int64)
        for fold in range(INNER_FOLDS):
            train = np.flatnonzero(folds != fold)
            valid = np.flatnonzero(folds == fold)
            if not train.size or not valid.size or set(y[train].tolist()) != {0, 1}:
                raise ValueError("every inner fold must have train/valid and both train classes")
            if set(groups[train].tolist()) & set(groups[valid].tolist()):
                raise ValueError("inner folds are not group disjoint")
            x_train = np.concatenate((originals[candidate][train], mirrors[candidate][train]))
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
                model.predict_proba(scaler.transform(originals[candidate][valid]))[:, 1]
                + model.predict_proba(scaler.transform(mirrors[candidate][valid]))[:, 1]
            )
            counts[valid] += 1
            fits += 1
            predictions += int(valid.size)
        if not np.isfinite(oof).all() or not np.all(counts == 1):
            raise RuntimeError("OOF predictions must cover each development row once")
        probabilities[candidate] = oof
    return ActionAlignedOOFResult(
        probabilities=probabilities,
        audit={
            "development_model_fits": fits,
            "development_predictions": predictions,
            "protected_feature_reads": 0,
            "protected_fits": 0,
            "protected_predictions": 0,
        },
    )


def choose_locked_candidate(metrics: Mapping[str, Mapping[str, float]]) -> str:
    """Promote action only under the predeclared non-inferiority/improvement gate."""
    if tuple(metrics) != CANDIDATE_ORDER:
        raise ValueError("locking metrics must preserve candidate order")
    baseline, action = (metrics[name] for name in CANDIDATE_ORDER)
    required = {"auroc", "balanced_accuracy", "brier"}
    if any(not required.issubset(candidate) for candidate in (baseline, action)):
        raise ValueError("locking metrics are incomplete")
    advances = (
        float(action["auroc"]) >= float(baseline["auroc"])
        and float(action["balanced_accuracy"]) >= float(baseline["balanced_accuracy"])
        and (
            float(action["balanced_accuracy"]) > float(baseline["balanced_accuracy"])
            or float(action["brier"]) < float(baseline["brier"])
        )
    )
    return CANDIDATE_ORDER[1] if advances else CANDIDATE_ORDER[0]


__all__ = [
    "ActionAlignedOOFResult", "CANDIDATE_ORDER", "FIXED_C", "FIXED_THRESHOLD",
    "choose_locked_candidate", "run_group_disjoint_oof",
]
