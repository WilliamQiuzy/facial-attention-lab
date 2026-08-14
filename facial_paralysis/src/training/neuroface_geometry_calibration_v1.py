"""Label-free manual68 teacher calibration and sealed 110D comparison."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.preprocessing.semantic_landmarks import clinical23_v2_to_semantic23
from src.training.architecture_search_v1 import group_balanced_weights, require_development_only
from src.training.neuroface_motion_pretrain_v1 import build_stratified_participant_folds


ALPHA = 1.0
MIRROR_PAIRS = ((0, 1), (4, 5), (7, 8), (10, 11), (14, 15), (18, 19))
SIGNED = (3, 13, 17)
MIRROR_INVARIANT = (2, 6, 9, 12, 16, 20, 21, 22)


@dataclass(frozen=True)
class GeometryCalibrationResult:
    scaler: StandardScaler
    model: Ridge
    metrics: Mapping[str, object]


@dataclass(frozen=True)
class CalibratedTransferDataset:
    baseline: np.ndarray
    mirrored_baseline: np.ndarray
    calibrated: np.ndarray
    mirrored_calibrated: np.ndarray
    labels: np.ndarray
    group_ids: np.ndarray
    development_indices: np.ndarray
    protected_indices: np.ndarray
    inner_fold_by_index: np.ndarray


def mirror_semantic23(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim < 1 or array.shape[-1] != 23 or not np.isfinite(array).all():
        raise ValueError("semantic23 mirror requires finite vectors with last dimension 23")
    output = array.copy()
    for left, right in MIRROR_PAIRS:
        output[..., left] = array[..., right]
        output[..., right] = array[..., left]
    for index in SIGNED:
        output[..., index] = -array[..., index]
    return output


def _fit_calibrator(
    inputs: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
) -> tuple[StandardScaler, Ridge]:
    augmented_inputs = np.concatenate((inputs, mirror_semantic23(inputs)))
    augmented_targets = np.concatenate((targets, mirror_semantic23(targets)))
    augmented_groups = np.concatenate((groups, groups))
    scaler = StandardScaler().fit(augmented_inputs)
    model = Ridge(alpha=ALPHA)
    model.fit(
        scaler.transform(augmented_inputs), augmented_targets,
        sample_weight=group_balanced_weights(augmented_groups),
    )
    return scaler, model


def fit_geometry_calibration(
    mediapipe: np.ndarray,
    manual: np.ndarray,
    participant_ids: np.ndarray,
    cohorts: np.ndarray,
) -> GeometryCalibrationResult:
    inputs = np.asarray(mediapipe, dtype=np.float64)
    targets = np.asarray(manual, dtype=np.float64)
    groups = np.asarray(participant_ids, dtype=object)
    cohort_array = np.asarray(cohorts, dtype=object)
    n = inputs.shape[0]
    if (
        inputs.shape != (n, 23) or targets.shape != (n, 23)
        or groups.shape != (n,) or cohort_array.shape != (n,) or n != 3306
        or not np.isfinite(inputs).all() or not np.isfinite(targets).all()
        or len(set(groups.tolist())) != 36
    ):
        raise ValueError("geometry calibration requires all 3,306 paired frames and 36 participants")
    folds = build_stratified_participant_folds(
        groups, cohort_array, folds=6, seed=20260813
    )
    oof = np.full(targets.shape, np.nan, dtype=np.float64)
    indices = np.arange(n)
    for fold in range(6):
        train, validation = indices[folds != fold], indices[folds == fold]
        scaler, model = _fit_calibrator(inputs[train], targets[train], groups[train])
        oof[validation] = model.predict(scaler.transform(inputs[validation]))
    if not np.isfinite(oof).all():
        raise RuntimeError("manual68 calibration OOF predictions are incomplete")
    before: list[float] = []
    after: list[float] = []
    for index in MIRROR_INVARIANT:
        initial = spearmanr(inputs[:, index], targets[:, index]).statistic
        calibrated = spearmanr(oof[:, index], targets[:, index]).statistic
        before.append(abs(float(initial)) if np.isfinite(initial) else 0.0)
        after.append(abs(float(calibrated)) if np.isfinite(calibrated) else 0.0)
    scaler, model = _fit_calibrator(inputs, targets, groups)
    equivariance_error = float(np.max(np.abs(
        model.predict(scaler.transform(mirror_semantic23(inputs[:256])))
        - mirror_semantic23(model.predict(scaler.transform(inputs[:256])))
    )))
    return GeometryCalibrationResult(
        scaler=scaler,
        model=model,
        metrics={
            "participant_disjoint_folds": 6,
            "frames": n,
            "participants": 36,
            "mirror_invariant_median_absolute_spearman_before": float(np.median(before)),
            "mirror_invariant_median_absolute_spearman_after": float(np.median(after)),
            "mirror_equivariance_max_abs_error": equivariance_error,
            "ridge_alpha": ALPHA,
        },
    )


def calibrate_dynamic_features(
    raw_features: np.ndarray,
    valid_masks: np.ndarray,
    calibration: GeometryCalibrationResult,
) -> np.ndarray:
    raw = np.asarray(raw_features, dtype=np.float32)
    mask = np.asarray(valid_masks, dtype=bool)
    if raw.ndim != 4 or raw.shape[1:] != (4, 32, 95) or mask.shape != raw.shape[:-1]:
        raise ValueError("dynamic calibration requires (n,4,32,95) plus aligned masks")
    output = raw.copy()
    semantic = clinical23_v2_to_semantic23(raw[..., -23:])
    rows = semantic[mask]
    output[..., -23:][mask] = calibration.model.predict(
        calibration.scaler.transform(rows)
    ).astype(np.float32)
    output[~mask] = 0.0
    return output


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities >= 0.5
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "sensitivity": float(np.mean(predictions[labels == 1])),
        "specificity": float(np.mean(~predictions[labels == 0])),
    }


def evaluate_calibrated_transfer(dataset: CalibratedTransferDataset) -> dict[str, object]:
    labels = np.asarray(dataset.labels, dtype=np.int64)
    groups = np.asarray(dataset.group_ids, dtype=object)
    development = np.asarray(dataset.development_indices, dtype=np.int64)
    protected = np.asarray(dataset.protected_indices, dtype=np.int64)
    folds = np.asarray(dataset.inner_fold_by_index, dtype=np.int64)
    require_development_only(development, development, protected, "manual68 calibrated transfer")
    candidates = {
        "landmark_110d": (dataset.baseline, dataset.mirrored_baseline),
        "manual68_calibrated_110d": (dataset.calibrated, dataset.mirrored_calibrated),
    }
    metrics: dict[str, dict[str, float]] = {}
    for name, (original, mirrored) in candidates.items():
        original = np.asarray(original, dtype=np.float64)
        mirrored = np.asarray(mirrored, dtype=np.float64)
        if original.shape != (labels.size, 110) or mirrored.shape != original.shape:
            raise ValueError("calibrated transfer summaries must have 110 dimensions")
        if not np.isfinite(original[development]).all() or not np.isfinite(mirrored[development]).all():
            raise ValueError("calibrated development features must be finite")
        oof = np.full(labels.shape, np.nan)
        for fold in range(4):
            train = development[folds[development] != fold]
            validation = development[folds[development] == fold]
            x = np.concatenate((original[train], mirrored[train]))
            y = np.concatenate((labels[train], labels[train]))
            g = np.concatenate((groups[train], groups[train]))
            scaler = StandardScaler().fit(x)
            model = LogisticRegression(
                C=0.01, penalty="l2", solver="liblinear", max_iter=2000, random_state=0
            )
            model.fit(scaler.transform(x), y, sample_weight=group_balanced_weights(g))
            oof[validation] = 0.5 * (
                model.predict_proba(scaler.transform(original[validation]))[:, 1]
                + model.predict_proba(scaler.transform(mirrored[validation]))[:, 1]
            )
        if not np.isfinite(oof[development]).all() or np.isfinite(oof[protected]).any():
            raise RuntimeError("calibrated transfer crossed the protected boundary")
        group_labels: list[int] = []
        group_probabilities: list[float] = []
        for group in sorted(set(groups[development].tolist())):
            rows = development[groups[development] == group]
            values = set(labels[rows].tolist())
            if len(values) != 1:
                raise ValueError("one PalsyNet group crosses labels")
            group_labels.append(int(next(iter(values))))
            group_probabilities.append(float(np.mean(oof[rows])))
        metrics[name] = _metrics(np.asarray(group_labels), np.asarray(group_probabilities))
    baseline, calibrated = metrics["landmark_110d"], metrics["manual68_calibrated_110d"]
    promoted = (
        calibrated["auroc"] > baseline["auroc"]
        and calibrated["balanced_accuracy"] >= baseline["balanced_accuracy"] - 0.02
        and calibrated["brier"] <= baseline["brier"]
    )
    return {
        "metrics": metrics,
        "development_recordings": int(development.size),
        "development_groups": len(set(groups[development].tolist())),
        "protected_predictions": 0,
        "promotion_criteria_met": bool(promoted),
        "selected_model": "manual68_calibrated_110d" if promoted else "landmark_110d",
    }


__all__ = [
    "CalibratedTransferDataset", "GeometryCalibrationResult", "calibrate_dynamic_features",
    "evaluate_calibrated_transfer", "fit_geometry_calibration", "mirror_semantic23",
]
