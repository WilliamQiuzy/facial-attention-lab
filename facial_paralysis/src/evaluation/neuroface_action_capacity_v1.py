"""Frozen participant-disjoint NeuroFace action-capacity experiment.

This module is deliberately limited to the three preregistered oral tasks.  It
derives the binary target from the released cohort, fits one small expert per
task and fold, and keeps participant identifiers only in the private result.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src.training.neuroface_motion_pretrain_v1 import (
    build_stratified_participant_folds,
)


SCHEMA_VERSION = "neuroface_action_capacity_v1"
PRIMARY_TASKS = ("NSM_KISS", "NSM_OPEN", "NSM_SPREAD")
COHORTS = ("als", "healthy_control", "post_stroke")
EXPECTED_COHORT_COUNTS = {"als": 11, "healthy_control": 11, "post_stroke": 14}
FEATURE_DIMENSION = 18
FOLDS = 6
FOLD_SEED = 20260813
BOOTSTRAP_REPEATS = 5000
BOOTSTRAP_SEED = 20260814
MINIMUM_VALID_BOOTSTRAP_FRACTION = 0.95
METRICS = (
    "auroc", "average_precision", "brier", "balanced_accuracy",
    "sensitivity", "specificity",
)
_GROUP_ID = re.compile(r"^grp_[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
FROZEN_110D_EXTERNAL_REPORT_SHA256 = (
    "beda5bee5ed3736a90245e98c165777198aee6d6be2bbcd8a13f0fb2b1a11984"
)
FROZEN_110D_MODEL_ARTIFACT_SHA256 = (
    "cbc49d0aa54b504915bebd00fdbe005458378e5675b57461ce83d3385f9b60f9"
)


def _frozen_110d_descriptive_comparator() -> dict[str, object]:
    return {
        "name": "frozen_landmark_110d_neuroface_primary_three_task",
        "status": "previously_released_descriptive_comparator_not_rerun",
        "used_for_selection": False,
        "public_report_sha256": FROZEN_110D_EXTERNAL_REPORT_SHA256,
        "frozen_model_artifact_sha256": FROZEN_110D_MODEL_ARTIFACT_SHA256,
        "metrics": {
            "auroc": 0.34909090909090906,
            "average_precision": 0.6083866568652084,
            "brier": 0.2832172596186839,
            "balanced_accuracy": 0.4163636363636364,
            "sensitivity": 0.56,
            "specificity": 0.2727272727272727,
        },
    }


FROZEN_110D_DESCRIPTIVE_COMPARATOR = _frozen_110d_descriptive_comparator()


@dataclass(frozen=True)
class ActionCapacityDataset:
    original_features: np.ndarray
    mirrored_features: np.ndarray
    participant_ids: np.ndarray
    tasks: np.ndarray
    cohorts: np.ndarray


@dataclass(frozen=True)
class ExpertFitAudit:
    fold: int
    task: str
    training_participants: int
    training_rows: int
    training_cohorts: tuple[str, ...]
    training_participant_ids: frozenset[str]
    original_weight_sum: float
    mirrored_weight_sum: float
    total_weight: float
    C: float
    solver: str
    max_iter: int
    random_state: int


@dataclass(frozen=True)
class ActionCapacityAudit:
    palsynet_path_accesses: int = 0
    palsynet_cache_reads: int = 0
    palsynet_predictions: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name))
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class ActionCapacityResult:
    participant_ids: np.ndarray
    cohorts: np.ndarray
    labels: np.ndarray
    fold_assignments: np.ndarray
    original_probabilities: np.ndarray
    mirrored_probabilities: np.ndarray
    task_scores: np.ndarray
    participant_scores: np.ndarray
    metrics: Mapping[str, float]
    per_task_metrics: Mapping[str, Mapping[str, float | int]]
    bootstrap: Mapping[str, object]
    fit_audit: tuple[ExpertFitAudit, ...]


@dataclass(frozen=True)
class _FixtureActionCapacityResult:
    """Non-public short-bootstrap result that cannot enter formal reporting."""
    participant_ids: np.ndarray
    cohorts: np.ndarray
    labels: np.ndarray
    fold_assignments: np.ndarray
    original_probabilities: np.ndarray
    mirrored_probabilities: np.ndarray
    task_scores: np.ndarray
    participant_scores: np.ndarray
    metrics: Mapping[str, float]
    per_task_metrics: Mapping[str, Mapping[str, float | int]]
    bootstrap: Mapping[str, object]
    fit_audit: tuple[ExpertFitAudit, ...]


def protocol() -> dict[str, object]:
    """Return the non-selectable v1 analysis contract."""
    return {
        "tasks": list(PRIMARY_TASKS),
        "folds": {"count": FOLDS, "seed": FOLD_SEED},
        "target": {
            "healthy_control": 0, "als": 1, "post_stroke": 1,
        },
        "expert": {
            "type": "standardized_l2_logistic",
            "C": 0.01,
            "solver": "liblinear",
            "max_iter": 2000,
            "random_state": 0,
            "threshold": 0.5,
        },
        "training_mirror_weights": {
            "original": 0.5,
            "horizontal_mirror": 0.5,
            "total_per_participant_per_task": 1.0,
        },
        "task_score": "mean_original_and_horizontal_mirror_probability",
        "participant_score": (
            "mean_of_three_task_scores_after_each_original_mirror_mean"
        ),
        "missing_primary_task": "abstain_fail_closed",
        "bootstrap": {
            "method": "three_cohort_stratified_participant_percentile",
            "draw_sizes": dict(EXPECTED_COHORT_COUNTS),
            "repeats": BOOTSTRAP_REPEATS,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
            "minimum_valid_draws": 4750,
        },
    }


def _validated_dataset(
    dataset: ActionCapacityDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(dataset, ActionCapacityDataset):
        raise ValueError("an ActionCapacityDataset is required")
    original = np.asarray(dataset.original_features)
    mirrored = np.asarray(dataset.mirrored_features)
    groups = np.asarray(dataset.participant_ids, dtype=object)
    tasks = np.asarray(dataset.tasks, dtype=object)
    cohorts = np.asarray(dataset.cohorts, dtype=object)
    n = groups.size
    if (
        original.shape != (n, FEATURE_DIMENSION)
        or mirrored.shape != (n, FEATURE_DIMENSION)
        or original.dtype != np.dtype(np.float64)
        or mirrored.dtype != np.dtype(np.float64)
        or tasks.shape != (n,)
        or cohorts.shape != (n,)
        or groups.shape != (n,)
        or n != 36 * len(PRIMARY_TASKS)
    ):
        raise ValueError("capacity dataset must contain exact aligned 108 x 18 arrays")
    if not np.isfinite(original).all() or not np.isfinite(mirrored).all():
        raise ValueError("capacity features must be finite")

    by_group: dict[str, dict[str, int]] = {}
    cohort_by_group: dict[str, str] = {}
    for row, (raw_group, raw_task, raw_cohort) in enumerate(
        zip(groups.tolist(), tasks.tolist(), cohorts.tolist())
    ):
        if (
            not isinstance(raw_group, str)
            or _GROUP_ID.fullmatch(raw_group) is None
            or raw_task not in PRIMARY_TASKS
            or raw_cohort not in COHORTS
        ):
            raise ValueError("participant, task, or cohort is outside the freeze")
        previous = cohort_by_group.setdefault(raw_group, str(raw_cohort))
        if previous != raw_cohort:
            raise ValueError("one participant crosses cohorts")
        task_rows = by_group.setdefault(raw_group, {})
        if raw_task in task_rows:
            raise ValueError("participant/task rows must be unique")
        task_rows[str(raw_task)] = row
    if len(by_group) != 36 or any(tuple(sorted(rows)) != tuple(sorted(PRIMARY_TASKS))
                                  for rows in by_group.values()):
        raise ValueError("every participant must have all three primary tasks")
    observed_counts = {
        cohort: sum(value == cohort for value in cohort_by_group.values())
        for cohort in COHORTS
    }
    if observed_counts != EXPECTED_COHORT_COUNTS:
        raise ValueError("NeuroFace cohort counts differ from 11/11/14")

    ordered_groups = np.asarray(sorted(by_group), dtype=object)
    ordered_cohorts = np.asarray(
        [cohort_by_group[str(group)] for group in ordered_groups], dtype=object
    )
    ordered_original = np.stack([
        original[by_group[str(group)][task]]
        for group in ordered_groups for task in PRIMARY_TASKS
    ]).reshape(36, 3, FEATURE_DIMENSION)
    ordered_mirrored = np.stack([
        mirrored[by_group[str(group)][task]]
        for group in ordered_groups for task in PRIMARY_TASKS
    ]).reshape(36, 3, FEATURE_DIMENSION)
    labels = np.asarray(
        [int(cohort != "healthy_control") for cohort in ordered_cohorts],
        dtype=np.int64,
    )
    return ordered_original, ordered_mirrored, ordered_groups, ordered_cohorts, labels


def _binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(scores, dtype=np.float64)
    if (
        y.ndim != 1 or probability.shape != y.shape or y.size == 0
        or set(y.tolist()) != {0, 1} or not np.isfinite(probability).all()
        or np.any((probability < 0.0) | (probability > 1.0))
    ):
        raise ValueError("binary metric arrays must be finite, aligned, and two-class")
    predicted = probability >= 0.5
    positive = y == 1
    negative = ~positive
    sensitivity = float(np.mean(predicted[positive]))
    specificity = float(np.mean(~predicted[negative]))
    bounded = lambda value: float(np.clip(float(value), 0.0, 1.0))
    return {
        "auroc": bounded(roc_auc_score(y, probability)),
        "average_precision": bounded(average_precision_score(y, probability)),
        "brier": bounded(brier_score_loss(y, probability)),
        "balanced_accuracy": bounded(balanced_accuracy_score(y, predicted)),
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def _bootstrap_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    cohorts: np.ndarray,
    *,
    repeats: int,
) -> dict[str, object]:
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("bootstrap_repeats must be a positive integer")
    strata = {
        cohort: np.flatnonzero(cohorts == cohort) for cohort in COHORTS
    }
    if {name: int(rows.size) for name, rows in strata.items()} != EXPECTED_COHORT_COUNTS:
        raise ValueError("bootstrap requires exact original 11/11/14 strata")
    draws: dict[str, list[float]] = {name: [] for name in METRICS}
    invalid = 0
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for _ in range(repeats):
        sampled = np.concatenate([
            rng.choice(strata[cohort], size=EXPECTED_COHORT_COUNTS[cohort], replace=True)
            for cohort in COHORTS
        ])
        try:
            values = _binary_metrics(labels[sampled], scores[sampled])
        except ValueError:
            invalid += 1
            continue
        for name in METRICS:
            draws[name].append(values[name])
    valid = repeats - invalid
    required = int(np.ceil(MINIMUM_VALID_BOOTSTRAP_FRACTION * repeats))
    if valid < required:
        raise ValueError("fewer than 95 percent of bootstrap draws are valid")
    intervals = {}
    for name in METRICS:
        values = np.asarray(draws[name], dtype=np.float64)
        lower, upper = np.percentile(values, [2.5, 97.5])
        intervals[name] = {"lower": float(lower), "upper": float(upper)}
    return {
        "method": "three_cohort_stratified_participant_percentile",
        "repeats": repeats,
        "seed": BOOTSTRAP_SEED,
        "draw_sizes": dict(EXPECTED_COHORT_COUNTS),
        "valid_draws": valid,
        "invalid_draws": invalid,
        "minimum_valid_draws": required,
        "confidence_level": 0.95,
        "intervals": intervals,
    }


def _evaluate_action_capacity_oof_core(
    dataset: ActionCapacityDataset,
    *,
    bootstrap_repeats: int,
    result_type,
):
    original, mirrored, groups, cohorts, labels = _validated_dataset(dataset)
    folds = build_stratified_participant_folds(
        groups, cohorts, folds=FOLDS, seed=FOLD_SEED
    )
    original_probability = np.full((36, 3), np.nan, dtype=np.float64)
    mirrored_probability = np.full((36, 3), np.nan, dtype=np.float64)
    fit_audit: list[ExpertFitAudit] = []
    for fold in range(FOLDS):
        train = np.flatnonzero(folds != fold)
        held = np.flatnonzero(folds == fold)
        if set(groups[train].tolist()) & set(groups[held].tolist()):
            raise AssertionError("participant crossed an action expert fold")
        for task_index, task in enumerate(PRIMARY_TASKS):
            train_original = original[train, task_index]
            train_mirrored = mirrored[train, task_index]
            x_train = np.concatenate((train_original, train_mirrored), axis=0)
            y_train = np.concatenate((labels[train], labels[train]), axis=0)
            weights = np.full(x_train.shape[0], 0.5, dtype=np.float64)
            if set(y_train.tolist()) != {0, 1}:
                raise ValueError("a task-specific training fold lacks both classes")
            scaler = StandardScaler()
            transformed = scaler.fit_transform(x_train)
            model = LogisticRegression(
                C=0.01,
                penalty="l2",
                solver="liblinear",
                max_iter=2000,
                random_state=0,
            )
            model.fit(transformed, y_train, sample_weight=weights)
            original_probability[held, task_index] = model.predict_proba(
                scaler.transform(original[held, task_index])
            )[:, 1]
            mirrored_probability[held, task_index] = model.predict_proba(
                scaler.transform(mirrored[held, task_index])
            )[:, 1]
            fit_audit.append(ExpertFitAudit(
                fold=fold,
                task=task,
                training_participants=int(train.size),
                training_rows=int(x_train.shape[0]),
                training_cohorts=tuple(sorted(set(cohorts[train].tolist()))),
                training_participant_ids=frozenset(groups[train].tolist()),
                original_weight_sum=float(weights[:train.size].sum()),
                mirrored_weight_sum=float(weights[train.size:].sum()),
                total_weight=float(weights.sum()),
                C=float(model.C),
                solver=str(model.solver),
                max_iter=int(model.max_iter),
                random_state=int(model.random_state),
            ))
    if (
        not np.isfinite(original_probability).all()
        or not np.isfinite(mirrored_probability).all()
    ):
        raise ValueError("a held-out participant lacks a primary task probability")
    task_scores = (original_probability + mirrored_probability) / 2.0
    participant_scores = task_scores.mean(axis=1)
    metrics = _binary_metrics(labels, participant_scores)
    per_task = {
        task: {
            "auroc": float(roc_auc_score(labels, task_scores[:, index])),
            "coverage_participants": 36,
            "coverage_fraction": 1.0,
        }
        for index, task in enumerate(PRIMARY_TASKS)
    }
    bootstrap = _bootstrap_metrics(
        labels, participant_scores, cohorts, repeats=bootstrap_repeats
    )
    return result_type(
        participant_ids=groups.copy(),
        cohorts=cohorts.copy(),
        labels=labels.copy(),
        fold_assignments=folds.copy(),
        original_probabilities=original_probability,
        mirrored_probabilities=mirrored_probability,
        task_scores=task_scores,
        participant_scores=participant_scores,
        metrics=metrics,
        per_task_metrics=per_task,
        bootstrap=bootstrap,
        fit_audit=tuple(fit_audit),
    )


def evaluate_action_capacity_oof(
    dataset: ActionCapacityDataset,
) -> ActionCapacityResult:
    """Run the formal v1 evaluator with exactly 5,000 bootstrap draws."""
    return _evaluate_action_capacity_oof_core(
        dataset,
        bootstrap_repeats=BOOTSTRAP_REPEATS,
        result_type=ActionCapacityResult,
    )


def _evaluate_action_capacity_oof_fixture(
    dataset: ActionCapacityDataset,
    *,
    bootstrap_repeats: int,
) -> _FixtureActionCapacityResult:
    """Test-only speed helper; its distinct type is refused by formal reporting."""
    return _evaluate_action_capacity_oof_core(
        dataset,
        bootstrap_repeats=bootstrap_repeats,
        result_type=_FixtureActionCapacityResult,
    )


def _validated_runtime(runtime: Mapping[str, object]) -> dict[str, object]:
    if set(runtime) != {"host_class", "device_class", "seconds"}:
        raise ValueError("runtime report fields are closed")
    if runtime["host_class"] != "nebius_h200" or runtime["device_class"] not in {
        "cpu", "cuda",
    }:
        raise ValueError("runtime class is outside the H200 release contract")
    seconds = runtime["seconds"]
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not np.isfinite(
        seconds
    ) or float(seconds) < 0.0:
        raise ValueError("runtime seconds must be finite and nonnegative")
    return {"host_class": runtime["host_class"], "device_class": runtime["device_class"],
            "seconds": float(seconds)}


def _private_oof_commitment_sha256(result: ActionCapacityResult) -> str:
    """Commit to private row order and all probabilities without publishing IDs."""
    hasher = hashlib.sha256()
    identity = {
        "participant_ids": [str(value) for value in result.participant_ids.tolist()],
        "cohorts": [str(value) for value in result.cohorts.tolist()],
        "tasks": list(PRIMARY_TASKS),
    }
    hasher.update((
        json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii"))
    arrays = (
        ("labels", np.asarray(result.labels, dtype="<i8")),
        ("fold_assignments", np.asarray(result.fold_assignments, dtype="<i8")),
        ("original_probabilities", np.asarray(result.original_probabilities, dtype="<f8")),
        ("mirrored_probabilities", np.asarray(result.mirrored_probabilities, dtype="<f8")),
        ("task_scores", np.asarray(result.task_scores, dtype="<f8")),
        ("participant_scores", np.asarray(result.participant_scores, dtype="<f8")),
    )
    for name, values in arrays:
        contiguous = np.ascontiguousarray(values)
        header = {
            "name": name, "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
        }
        hasher.update((
            json.dumps(header, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("ascii"))
        hasher.update(contiguous.tobytes(order="C"))
    return hasher.hexdigest()


def _validate_formal_result(
    result: ActionCapacityResult,
) -> tuple[dict[str, float], dict[str, dict[str, float | int]], dict[str, object]]:
    if type(result) is not ActionCapacityResult:
        raise ValueError("formal reporting requires the exact 5,000-draw result type")
    groups = np.asarray(result.participant_ids, dtype=object)
    cohorts = np.asarray(result.cohorts, dtype=object)
    labels = np.asarray(result.labels)
    folds = np.asarray(result.fold_assignments)
    original = np.asarray(result.original_probabilities)
    mirrored = np.asarray(result.mirrored_probabilities)
    task_scores = np.asarray(result.task_scores)
    participant_scores = np.asarray(result.participant_scores)
    if (
        groups.shape != (36,) or len(set(groups.tolist())) != 36
        or any(not isinstance(group, str) or _GROUP_ID.fullmatch(group) is None
               for group in groups.tolist())
        or cohorts.shape != (36,) or labels.shape != (36,) or folds.shape != (36,)
        or original.shape != (36, 3) or mirrored.shape != (36, 3)
        or task_scores.shape != (36, 3) or participant_scores.shape != (36,)
        or not all(np.isfinite(values).all() for values in (
            original, mirrored, task_scores, participant_scores,
        ))
        or any(np.any((values < 0.0) | (values > 1.0)) for values in (
            original, mirrored, task_scores, participant_scores,
        ))
    ):
        raise ValueError("private formal OOF arrays are malformed")
    cohort_counts = {
        cohort: int(np.sum(cohorts == cohort)) for cohort in COHORTS
    }
    expected_labels = np.asarray(
        [int(cohort != "healthy_control") for cohort in cohorts], dtype=np.int64
    )
    expected_folds = build_stratified_participant_folds(
        groups, cohorts, folds=FOLDS, seed=FOLD_SEED
    )
    if (
        cohort_counts != EXPECTED_COHORT_COUNTS
        or not np.array_equal(labels, expected_labels)
        or not np.array_equal(folds, expected_folds)
        or not np.array_equal(task_scores, (original + mirrored) / 2.0)
        or not np.array_equal(participant_scores, task_scores.mean(axis=1))
    ):
        raise ValueError("private OOF label, fold, or aggregation contract differs")
    expected_fit_keys = {(fold, task) for fold in range(FOLDS) for task in PRIMARY_TASKS}
    observed_fit_keys: set[tuple[int, str]] = set()
    for fit in result.fit_audit:
        if not isinstance(fit, ExpertFitAudit):
            raise ValueError("formal fit audit contains a malformed entry")
        held = set(groups[folds == fit.fold].tolist())
        expected_train = frozenset(groups[folds != fit.fold].tolist())
        key = (fit.fold, fit.task)
        if (
            key not in expected_fit_keys or key in observed_fit_keys
            or fit.training_participant_ids != expected_train
            or held.intersection(fit.training_participant_ids)
            or fit.training_participants != 36 - len(held)
            or fit.training_rows != fit.training_participants * 2
            or fit.original_weight_sum != fit.training_participants * 0.5
            or fit.mirrored_weight_sum != fit.training_participants * 0.5
            or fit.total_weight != float(fit.training_participants)
            or set(fit.training_cohorts) != set(COHORTS)
            or (fit.C, fit.solver, fit.max_iter, fit.random_state)
            != (0.01, "liblinear", 2000, 0)
        ):
            raise ValueError("formal fit audit differs from the fixed expert protocol")
        observed_fit_keys.add(key)
    if observed_fit_keys != expected_fit_keys:
        raise ValueError("formal result does not contain all 18 expert fits")

    expected_metrics = _binary_metrics(labels, participant_scores)
    expected_per_task = {
        task: {
            "auroc": float(roc_auc_score(labels, task_scores[:, index])),
            "coverage_participants": 36,
            "coverage_fraction": 1.0,
        }
        for index, task in enumerate(PRIMARY_TASKS)
    }
    expected_bootstrap = _bootstrap_metrics(
        labels, participant_scores, cohorts, repeats=BOOTSTRAP_REPEATS
    )
    if (
        dict(result.metrics) != expected_metrics
        or dict(result.per_task_metrics) != expected_per_task
        or dict(result.bootstrap) != expected_bootstrap
        or result.bootstrap.get("minimum_valid_draws") != 4750
    ):
        raise ValueError("formal metrics or fixed 5,000-draw bootstrap cannot be reproduced")
    return expected_metrics, expected_per_task, expected_bootstrap


def build_public_report(
    result: ActionCapacityResult,
    *,
    provenance: Mapping[str, object],
    audit: ActionCapacityAudit,
    runtime: Mapping[str, object],
) -> dict[str, object]:
    """Construct an aggregate-only report; private OOF rows are never embedded."""
    recomputed, recomputed_per_task, recomputed_bootstrap = _validate_formal_result(result)
    expected_provenance = {
        "private_manifest_sha256", "collection_manifest_sha256",
        "primary_cache_collection_sha256", "implementation_sha256",
        "dependency_lock_sha256", "mount_attestation_sha256",
    }
    if set(provenance) != expected_provenance or any(
        not isinstance(value, str) or _SHA256.fullmatch(value) is None
        for value in provenance.values()
    ):
        raise ValueError("public provenance requires exact SHA-256 commitments")
    if not isinstance(audit, ActionCapacityAudit) or any(audit.as_dict().values()):
        raise ValueError("all protected-data access counters must remain zero")
    frozen_protocol = protocol()
    metric_report = {
        name: {
            "point": float(recomputed[name]),
            "ci95": dict(recomputed_bootstrap["intervals"][name]),
        }
        for name in METRICS
    }
    auroc_lower = float(recomputed_bootstrap["intervals"]["auroc"]["lower"])
    report_provenance = dict(provenance)
    report_provenance["private_oof_commitment_sha256"] = (
        _private_oof_commitment_sha256(result)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": "exploratory_cross_disease_orofacial_capacity_only",
        "dataset": {
            "name": "Toronto_NeuroFace_v1",
            "participants": 36,
            "affected": 25,
            "unaffected": 11,
            "primary_task_recordings": 108,
            "cohorts": dict(EXPECTED_COHORT_COUNTS),
        },
        "protocol": frozen_protocol,
        "metrics": metric_report,
        "per_task": {
            task: dict(recomputed_per_task[task]) for task in PRIMARY_TASKS
        },
        "descriptive_comparator": _frozen_110d_descriptive_comparator(),
        "mask_diagnostic": {
            "primary_task_mask": [True, True, True],
            "unique_masks": 1,
            "complete_participants": 36,
            "incomplete_participants": 0,
            "mask_only_discrimination_possible": False,
        },
        "bootstrap": {
            name: copy.deepcopy(recomputed_bootstrap[name])
            for name in (
                "method", "repeats", "seed", "draw_sizes", "valid_draws",
                "invalid_draws", "minimum_valid_draws", "confidence_level",
            )
        },
        "audit": audit.as_dict(),
        "runtime": _validated_runtime(runtime),
        "provenance": report_provenance,
        "decision": {
            "capacity_feasibility_signal": bool(auroc_lower > 0.5),
            "criterion": "lower_95_percent_auroc_above_0_50",
            "current_110d_replaced": False,
            "fusion_authorized": False,
            "bell_palsy_transfer_claim_authorized": False,
            "mayo_accuracy_claim_authorized": False,
            "clinical_use_authorized": False,
        },
    }


def validate_public_report(
    report: Mapping[str, object],
    *,
    result: ActionCapacityResult,
    expected_provenance: Mapping[str, object],
    expected_runtime: Mapping[str, object],
    expected_audit: ActionCapacityAudit,
) -> None:
    """Independently rebuild the formal report from private OOF evidence."""
    _reject_sensitive_report_content(report)
    required = {
        "schema_version", "claim_scope", "dataset", "protocol", "metrics",
        "per_task", "descriptive_comparator", "mask_diagnostic", "bootstrap",
        "audit", "runtime", "provenance", "decision",
    }
    if not isinstance(report, Mapping) or set(report) != required:
        raise ValueError("public report top-level schema differs")
    provenance = report["provenance"]
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "private_manifest_sha256", "collection_manifest_sha256",
        "primary_cache_collection_sha256", "implementation_sha256",
        "dependency_lock_sha256", "mount_attestation_sha256",
        "private_oof_commitment_sha256",
    } or any(not isinstance(value, str) or _SHA256.fullmatch(value) is None
             for value in provenance.values()):
        raise ValueError("provenance is malformed")
    expected = build_public_report(
        result,
        provenance=expected_provenance,
        audit=expected_audit,
        runtime=expected_runtime,
    )
    if dict(report) != expected:
        raise ValueError("public report differs from independent private-OOF recomputation")


def _reject_sensitive_report_content(report: object) -> None:
    """Reject identifiers, filesystem paths, and common secret markers recursively."""
    forbidden_fragments = (
        "grp_", "rec_", "participant_id", "recording_id", "/users/", "/home/",
        "aws_access_key", "secret_access_key", "runpod_api", "nvapi-",
        "bearer ", "private_key",
    )

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ValueError("public report keys must be strings")
                visit(key)
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif isinstance(value, str):
            lowered = value.casefold()
            if (
                any(fragment in lowered for fragment in forbidden_fragments)
                or value.startswith("/")
                or re.match(r"^[a-zA-Z]:[\\/]", value) is not None
                or value.startswith("~")
                or "file://" in lowered
            ):
                raise ValueError("public report contains an identifier, path, or secret")

    # Serialization is also required to be finite and plain JSON.
    try:
        json.dumps(report, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("public report is not finite plain JSON") from exc
    visit(report)


__all__ = [
    "BOOTSTRAP_REPEATS", "BOOTSTRAP_SEED", "COHORTS", "PRIMARY_TASKS",
    "FROZEN_110D_DESCRIPTIVE_COMPARATOR",
    "ActionCapacityAudit", "ActionCapacityDataset", "ActionCapacityResult",
    "build_public_report", "evaluate_action_capacity_oof", "protocol",
    "validate_public_report",
]
