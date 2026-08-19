#!/usr/bin/env python3
"""Run the fixed development-only mirror-invariant Landmark 110D screen.

Outer fold 0 remains sealed: this script neither extracts features, fits a
model, nor predicts on that fold.  The candidate differs from the frozen 110D
baseline only by horizontal-mirror augmentation during training and symmetric
probability averaging during validation.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_dynamic_landmark_classical import (  # noqa: E402
    ClassicalDataset,
    binary_group_metrics,
    group_sample_weights,
    load_classical_dataset,
    paired_stratified_group_bootstrap,
)
from src.evaluation.nested_group_cv import (  # noqa: E402
    INNER_FOLDS as _INNER_FOLDS,
    assert_outer_test_isolation,
    build_nested_group_splits,
)
from src.models.dynamic_landmark import horizontal_mirror_features  # noqa: E402
from src.preprocessing.trajectory_features import (  # noqa: E402
    LANDMARK_DIM,
    trajectory_feature_set,
)


BASELINE = "landmark_110d"
CANDIDATE = "mirror_invariant_landmark_110d"
CANDIDATE_REGISTRY = {BASELINE: LANDMARK_DIM, CANDIDATE: LANDMARK_DIM}
FIXED_C = 0.01
FIXED_THRESHOLD = 0.5
FIXED_SOLVER = "liblinear"
FIXED_RANDOM_STATE = 0
FIXED_MAX_ITER = 2000
OUTER_FOLD_NUMBER = 0
INNER_FOLDS = _INNER_FOLDS
BOOTSTRAP_REPEATS = 5000
BOOTSTRAP_SEED = 20260805
MIRROR_TOLERANCE = 1e-12
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "dynamic_landmark"
    / "benchmarks"
    / "development"
    / "mirror-invariant-110d-v1"
    / "report.json"
)

_REPORT_TOP_LEVEL_FIELDS = {
    "schema_version",
    "claim_scope",
    "target",
    "disclosure",
    "dataset",
    "protocol",
    "counts",
    "metrics",
    "primary_comparison",
    "robustness",
    "audit",
    "decision",
}
_CLOSED_REPORT_OBJECT_FIELDS = {
    (): _REPORT_TOP_LEVEL_FIELDS,
    ("disclosure",): {
        "mirror_provenance_known",
        "identity_review_complete",
        "clinical_validation",
        "hyperparameter_search",
    },
    ("dataset",): {
        "name",
        "claim_unit",
        "identity_status",
        "collection_manifest_sha256",
    },
    ("protocol",): {
        "outer_fold_number",
        "inner_folds",
        "candidates",
        "candidate_dimensions",
        "mirror",
        "model",
        "bootstrap",
    },
    ("protocol", "candidate_dimensions"): set(CANDIDATE_REGISTRY),
    ("protocol", "mirror"): {
        "training_augmentation",
        "validation_inference",
        "transform",
        "probability_tolerance",
    },
    ("protocol", "model"): {
        "type",
        "c",
        "solver",
        "random_state",
        "max_iter",
        "threshold",
        "sample_weight",
        "hyperparameter_search",
    },
    ("protocol", "bootstrap"): {
        "paired",
        "unit",
        "stratified_by_binary_label",
        "repeats",
        "seed",
        "interval_scope",
    },
    ("counts",): {
        "dataset_recordings",
        "dataset_groups",
        "development_recordings",
        "development_groups",
        "development_affected_groups",
        "development_unaffected_groups",
        "protected_recordings",
        "protected_groups",
    },
    ("metrics",): set(CANDIDATE_REGISTRY),
    **{
        ("metrics", candidate): {
            "auroc",
            "average_precision",
            "brier",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
        }
        for candidate in CANDIDATE_REGISTRY
    },
    ("primary_comparison",): {
        "baseline",
        "candidate",
        "delta_auroc",
        "ci95",
        "probability_delta_gt_zero",
        "repeats",
        "seed",
        "interval_scope",
    },
    ("robustness",): {"max_mirror_probability_error"},
    ("audit",): {
        "development_candidate_feature_extractions",
        "development_mirror_transforms",
        "development_scaler_fits",
        "development_model_fits",
        "development_prediction_folds",
        "protected_candidate_feature_extractions",
        "protected_fits",
        "protected_predictions",
    },
    ("decision",): {
        "passed",
        "gates",
        "development_champion",
        "outer_evaluation_authorized",
        "hb_claim_authorized",
        "clinical_validation",
        "next_gate",
    },
    ("decision", "gates"): {
        "mirror_probability_error_at_most_1e_12",
        "auroc_not_lower",
        "balanced_accuracy_not_lower",
        "brier_not_higher",
        "zero_protected_candidate_feature_extractions_fits_predictions",
    },
}
_SENSITIVE_REPORT_KEYS = {
    "recording_id",
    "recording_ids",
    "record_ids",
    "group_id",
    "group_ids",
    "patient_id",
    "patient_ids",
    "subject_id",
    "subject_ids",
    "source_path",
    "source_paths",
    "filename",
    "filenames",
    "features",
    "labels",
    "probability",
    "probabilities",
    "prediction",
    "predictions",
    "rows",
    "records",
    "samples",
}
_OPAQUE_IDENTIFIER = re.compile(r"\b(?:rec|grp)_[A-Za-z0-9._-]+\b")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_FILE_LIKE_SUFFIX = re.compile(
    r"\.(?:json|jsonl|csv|tsv|npy|npz|pt|pth|mp4|mov|avi|mkv)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DevelopmentMatrices:
    """Original and mirrored 110D rows aligned to outer-train indices."""

    development_indices: np.ndarray
    protected_indices: np.ndarray
    original: np.ndarray
    mirrored: np.ndarray
    remirrored: np.ndarray
    extraction_indices: tuple[int, ...]
    mirror_transform_indices: tuple[int, ...]


@dataclass(frozen=True)
class IndexAuditEvent:
    """Global recording indices used by one development operation."""

    candidate: str
    inner_fold: int
    operation: str
    indices: tuple[int, ...]


@dataclass(frozen=True)
class DevelopmentOOFResult:
    candidate: str
    probabilities: np.ndarray
    audit_events: tuple[IndexAuditEvent, ...]
    max_mirror_probability_error: float | None


@dataclass(frozen=True)
class DevelopmentScreenResult:
    report: dict[str, object]
    candidate_results: Mapping[str, DevelopmentOOFResult]


MirrorTransform = Callable[[np.ndarray], np.ndarray]


def mirror_dynamic_features(features: np.ndarray) -> np.ndarray:
    """Apply the frozen raw 95-channel horizontal mirror exactly once."""
    array = np.asarray(features)
    if array.shape != (4, 32, 95):
        raise ValueError("features must have shape (4, 32, 95)")
    if array.dtype.kind != "f" or not np.isfinite(array).all():
        raise ValueError("features must be finite floating values")
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    return horizontal_mirror_features(tensor).detach().cpu().numpy().copy()


def build_development_matrices(
    dataset: ClassicalDataset,
    *,
    mirror_transform: MirrorTransform = mirror_dynamic_features,
) -> DevelopmentMatrices:
    """Build both 110D views exclusively from frozen outer-train rows."""
    if not callable(mirror_transform):
        raise ValueError("mirror_transform must be callable")
    outer = build_nested_group_splits(dataset.labels, dataset.group_ids)[
        OUTER_FOLD_NUMBER
    ]
    development = np.asarray(outer.train_indices, dtype=np.int64)
    protected = np.asarray(outer.test_indices, dtype=np.int64)
    assert_outer_test_isolation(
        protected,
        feature_extraction_indices=development,
        mirror_transform_indices=development,
    )

    original_rows: list[np.ndarray] = []
    mirrored_rows: list[np.ndarray] = []
    remirrored_rows: list[np.ndarray] = []
    for global_index in development.tolist():
        raw = dataset.features[global_index]
        temporal = (
            dataset.valid_masks[global_index],
            dataset.timestamps[global_index],
            dataset.source_frame_indices[global_index],
        )
        original_rows.append(trajectory_feature_set("landmark", raw, *temporal))
        mirrored_raw = np.asarray(mirror_transform(raw))
        if mirrored_raw.shape != raw.shape or mirrored_raw.dtype.kind != "f":
            raise ValueError("mirror transform must preserve the raw feature contract")
        if not np.isfinite(mirrored_raw).all():
            raise ValueError("mirror transform produced nonfinite values")
        remirrored_raw = np.asarray(mirror_transform(mirrored_raw))
        if (
            remirrored_raw.shape != raw.shape
            or remirrored_raw.dtype != raw.dtype
            or not np.isfinite(remirrored_raw).all()
            or not np.array_equal(remirrored_raw, raw)
        ):
            raise ValueError("mirror transform must be an exact involution")
        mirrored_rows.append(
            trajectory_feature_set("landmark", mirrored_raw, *temporal)
        )
        remirrored_rows.append(
            trajectory_feature_set("landmark", remirrored_raw, *temporal)
        )

    original = np.stack(original_rows).astype(np.float64, copy=False)
    mirrored = np.stack(mirrored_rows).astype(np.float64, copy=False)
    remirrored = np.stack(remirrored_rows).astype(np.float64, copy=False)
    expected = (development.size, LANDMARK_DIM)
    if (
        original.shape != expected
        or mirrored.shape != expected
        or remirrored.shape != expected
    ):
        raise ValueError(f"development matrices must each have shape {expected}")
    if not all(
        np.isfinite(matrix).all()
        for matrix in (original, mirrored, remirrored)
    ):
        raise ValueError("development matrices contain nonfinite values")
    if not np.array_equal(remirrored, original):
        raise ValueError("remirrored 110D rows must equal original rows")
    indices = tuple(int(index) for index in development)
    return DevelopmentMatrices(
        development_indices=development.copy(),
        protected_indices=protected.copy(),
        original=original,
        mirrored=mirrored,
        remirrored=remirrored,
        extraction_indices=indices,
        mirror_transform_indices=indices,
    )


def _fit_fixed_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_groups: np.ndarray,
) -> LogisticRegression:
    model = LogisticRegression(
        C=FIXED_C,
        penalty="l2",
        solver=FIXED_SOLVER,
        max_iter=FIXED_MAX_ITER,
        random_state=FIXED_RANDOM_STATE,
    )
    model.fit(
        x_train,
        y_train,
        sample_weight=group_sample_weights(train_groups),
    )
    return model


def _validate_prepared_mirror_pairing(
    dataset: ClassicalDataset,
    prepared: DevelopmentMatrices,
) -> None:
    """Re-derive every paired view so reordered or stale rows fail closed."""
    for local_index, global_index in enumerate(
        prepared.development_indices.tolist()
    ):
        raw = dataset.features[global_index]
        temporal = (
            dataset.valid_masks[global_index],
            dataset.timestamps[global_index],
            dataset.source_frame_indices[global_index],
        )
        expected_original = trajectory_feature_set("landmark", raw, *temporal)
        expected_mirrored = trajectory_feature_set(
            "landmark", mirror_dynamic_features(raw), *temporal
        )
        if not np.array_equal(expected_original, prepared.original[local_index]):
            raise ValueError("prepared original 110D row is stale or reordered")
        if not np.array_equal(expected_mirrored, prepared.mirrored[local_index]):
            raise ValueError("prepared mirrored 110D row is stale or reordered")
        if not np.array_equal(
            prepared.remirrored[local_index], prepared.original[local_index]
        ):
            raise ValueError("prepared remirrored 110D row is not paired")


def run_fixed_inner_oof(
    dataset: ClassicalDataset,
    prepared: DevelopmentMatrices,
    candidate: str,
) -> DevelopmentOOFResult:
    """Run one preregistered candidate on grouped inner OOF folds."""
    if candidate not in CANDIDATE_REGISTRY:
        raise ValueError(f"unknown fixed candidate {candidate!r}")
    outer = build_nested_group_splits(dataset.labels, dataset.group_ids)[
        OUTER_FOLD_NUMBER
    ]
    if not np.array_equal(prepared.development_indices, outer.train_indices):
        raise ValueError("development rows differ from the frozen outer train")
    if not np.array_equal(prepared.protected_indices, outer.test_indices):
        raise ValueError("protected rows differ from the frozen outer fold")
    if candidate == CANDIDATE:
        _validate_prepared_mirror_pairing(dataset, prepared)

    local_by_global = {
        int(global_index): local_index
        for local_index, global_index in enumerate(
            prepared.development_indices.tolist()
        )
    }
    probabilities = np.full(prepared.development_indices.size, np.nan)
    prediction_counts = np.zeros(prepared.development_indices.size, dtype=np.int64)
    events: list[IndexAuditEvent] = []
    maximum_mirror_error = 0.0

    for inner_number, inner in enumerate(outer.inner_folds):
        assert_outer_test_isolation(
            outer.test_indices,
            train_indices=inner.train_indices,
            validation_indices=inner.validation_indices,
            scaler_fit_indices=inner.train_indices,
            model_fit_indices=inner.train_indices,
            prediction_indices=inner.validation_indices,
        )
        train_local = np.asarray(
            [local_by_global[int(index)] for index in inner.train_indices],
            dtype=np.int64,
        )
        validation_local = np.asarray(
            [local_by_global[int(index)] for index in inner.validation_indices],
            dtype=np.int64,
        )

        if candidate == BASELINE:
            x_train = prepared.original[train_local]
            y_train = dataset.labels[inner.train_indices]
            train_groups = dataset.group_ids[inner.train_indices]
        else:
            x_train = np.concatenate(
                (prepared.original[train_local], prepared.mirrored[train_local]),
                axis=0,
            )
            y_train = np.concatenate(
                (dataset.labels[inner.train_indices], dataset.labels[inner.train_indices])
            )
            train_groups = np.concatenate(
                (
                    dataset.group_ids[inner.train_indices],
                    dataset.group_ids[inner.train_indices],
                )
            )

        scaler = StandardScaler().fit(x_train)
        events.append(IndexAuditEvent(
            candidate,
            inner_number,
            "scaler_fit",
            tuple(int(index) for index in inner.train_indices),
        ))
        model = _fit_fixed_logistic(
            scaler.transform(x_train), y_train, train_groups
        )
        events.append(IndexAuditEvent(
            candidate,
            inner_number,
            "model_fit",
            tuple(int(index) for index in inner.train_indices),
        ))

        original_probability = model.predict_proba(
            scaler.transform(prepared.original[validation_local])
        )[:, 1]
        if candidate == CANDIDATE:
            mirrored_probability = model.predict_proba(
                scaler.transform(prepared.mirrored[validation_local])
            )[:, 1]
            remirrored_probability = model.predict_proba(
                scaler.transform(prepared.remirrored[validation_local])
            )[:, 1]
            folded_probability = 0.5 * (
                original_probability + mirrored_probability
            )
            mirrored_input_probability = 0.5 * (
                mirrored_probability + remirrored_probability
            )
            maximum_mirror_error = max(
                maximum_mirror_error,
                float(np.max(np.abs(
                    folded_probability - mirrored_input_probability
                ))),
            )
        else:
            folded_probability = original_probability
        probabilities[validation_local] = folded_probability
        prediction_counts[validation_local] += 1
        events.append(IndexAuditEvent(
            candidate,
            inner_number,
            "predict",
            tuple(int(index) for index in inner.validation_indices),
        ))

    if not np.isfinite(probabilities).all() or not np.all(prediction_counts == 1):
        raise AssertionError("inner OOF did not predict every development row once")
    return DevelopmentOOFResult(
        candidate=candidate,
        probabilities=probabilities,
        audit_events=tuple(events),
        max_mirror_probability_error=(
            maximum_mirror_error if candidate == CANDIDATE else None
        ),
    )


def _count_groups(dataset: ClassicalDataset, indices: np.ndarray) -> int:
    return len(set(dataset.group_ids[indices].tolist()))


def _evaluate_gates(
    baseline_metrics: Mapping[str, float],
    candidate_metrics: Mapping[str, float],
    mirror_probability_error: float,
    audit: Mapping[str, int],
) -> dict[str, bool]:
    """Apply the fixed robustness and non-inferiority gates."""
    return {
        "mirror_probability_error_at_most_1e_12": bool(
            np.isfinite(mirror_probability_error)
            and mirror_probability_error <= MIRROR_TOLERANCE
        ),
        "auroc_not_lower": bool(
            float(candidate_metrics["auroc"]) >= float(baseline_metrics["auroc"])
        ),
        "balanced_accuracy_not_lower": bool(
            float(candidate_metrics["balanced_accuracy"])
            >= float(baseline_metrics["balanced_accuracy"])
        ),
        "brier_not_higher": bool(
            float(candidate_metrics["brier"]) <= float(baseline_metrics["brier"])
        ),
        "zero_protected_candidate_feature_extractions_fits_predictions": all(
            int(audit[name]) == 0
            for name in (
                "protected_candidate_feature_extractions",
                "protected_fits",
                "protected_predictions",
            )
        ),
    }


def run_development_screen(dataset: ClassicalDataset) -> DevelopmentScreenResult:
    """Evaluate the frozen pair and construct a deidentified aggregate report."""
    prepared = build_development_matrices(dataset)
    results = {
        candidate: run_fixed_inner_oof(dataset, prepared, candidate)
        for candidate in CANDIDATE_REGISTRY
    }
    development = prepared.development_indices
    protected = prepared.protected_indices
    labels = dataset.labels[development]
    groups = dataset.group_ids[development]
    metrics = {
        candidate: binary_group_metrics(
            labels, groups, results[candidate].probabilities
        )
        for candidate in CANDIDATE_REGISTRY
    }
    comparison = paired_stratified_group_bootstrap(
        labels,
        groups,
        results[BASELINE].probabilities,
        results[CANDIDATE].probabilities,
        repeats=BOOTSTRAP_REPEATS,
        seed=BOOTSTRAP_SEED,
    )
    primary_comparison = {
        "baseline": BASELINE,
        "candidate": CANDIDATE,
        **comparison,
    }

    protected_set = set(protected.tolist())
    extraction_set = set(prepared.extraction_indices)
    mirror_set = set(prepared.mirror_transform_indices)
    fit_events: list[IndexAuditEvent] = []
    prediction_events: list[IndexAuditEvent] = []
    for result in results.values():
        for event in result.audit_events:
            if event.operation in {"scaler_fit", "model_fit"}:
                fit_events.append(event)
            elif event.operation == "predict":
                prediction_events.append(event)
            else:
                raise AssertionError("unknown development audit operation")
    fit_index_set = {index for event in fit_events for index in event.indices}
    prediction_index_set = {
        index for event in prediction_events for index in event.indices
    }
    audit = {
        "development_candidate_feature_extractions": 5 * len(extraction_set),
        "development_mirror_transforms": 3 * len(mirror_set),
        "development_scaler_fits": sum(
            event.operation == "scaler_fit" for event in fit_events
        ),
        "development_model_fits": sum(
            event.operation == "model_fit" for event in fit_events
        ),
        "development_prediction_folds": len(prediction_events),
        "protected_candidate_feature_extractions": len(
            extraction_set & protected_set
        ),
        "protected_fits": len(fit_index_set & protected_set),
        "protected_predictions": len(prediction_index_set & protected_set),
    }
    mirror_error = results[CANDIDATE].max_mirror_probability_error
    if mirror_error is None:
        raise AssertionError("candidate did not provide a mirror robustness audit")
    gates = _evaluate_gates(
        metrics[BASELINE], metrics[CANDIDATE], mirror_error, audit
    )
    passed = all(gates.values())

    report: dict[str, object] = {
        "schema_version": "mirror_invariant_110d_v1_report",
        "claim_scope": "development_inner_oof_robustness_screen_only",
        "target": "binary_affected_vs_unaffected_not_hb_grade",
        "disclosure": {
            "mirror_provenance_known": False,
            "identity_review_complete": False,
            "clinical_validation": False,
            "hyperparameter_search": False,
        },
        "dataset": {
            "name": "PalsyNet",
            "claim_unit": dataset.claim_unit,
            "identity_status": dataset.identity_status,
            "collection_manifest_sha256": dataset.collection_manifest_sha256,
        },
        "protocol": {
            "outer_fold_number": OUTER_FOLD_NUMBER,
            "inner_folds": INNER_FOLDS,
            "candidates": list(CANDIDATE_REGISTRY),
            "candidate_dimensions": dict(CANDIDATE_REGISTRY),
            "mirror": {
                "training_augmentation": "original_plus_horizontal_mirror",
                "validation_inference": "mean_original_and_horizontal_mirror_probability",
                "transform": "frozen_clinical23_v2_horizontal_mirror",
                "probability_tolerance": MIRROR_TOLERANCE,
            },
            "model": {
                "type": "standardized_l2_logistic_regression",
                "c": FIXED_C,
                "solver": FIXED_SOLVER,
                "random_state": FIXED_RANDOM_STATE,
                "max_iter": FIXED_MAX_ITER,
                "threshold": FIXED_THRESHOLD,
                "sample_weight": "equal_total_weight_per_group",
                "hyperparameter_search": False,
            },
            "bootstrap": {
                "paired": True,
                "unit": "group",
                "stratified_by_binary_label": True,
                "repeats": BOOTSTRAP_REPEATS,
                "seed": BOOTSTRAP_SEED,
                "interval_scope": "fixed_oof_predictions_descriptive",
            },
        },
        "counts": {
            "dataset_recordings": int(dataset.labels.size),
            "dataset_groups": _count_groups(
                dataset, np.arange(dataset.labels.size, dtype=np.int64)
            ),
            "development_recordings": int(development.size),
            "development_groups": _count_groups(dataset, development),
            "development_affected_groups": len(set(groups[labels == 1].tolist())),
            "development_unaffected_groups": len(set(groups[labels == 0].tolist())),
            "protected_recordings": int(protected.size),
            "protected_groups": _count_groups(dataset, protected),
        },
        "metrics": metrics,
        "primary_comparison": primary_comparison,
        "robustness": {"max_mirror_probability_error": mirror_error},
        "audit": audit,
        "decision": {
            "passed": passed,
            "gates": gates,
            "development_champion": CANDIDATE if passed else BASELINE,
            "outer_evaluation_authorized": False,
            "hb_claim_authorized": False,
            "clinical_validation": False,
            "next_gate": "identity_review_then_prespecified_patient_disjoint_validation",
        },
    }
    return DevelopmentScreenResult(report=report, candidate_results=results)


def _validate_private_report(payload: Mapping[str, object]) -> None:
    """Fail closed on schema drift, row-level data, IDs, and paths."""
    if not isinstance(payload, Mapping) or set(payload) != _REPORT_TOP_LEVEL_FIELDS:
        raise ValueError("report top-level fields differ from the closed schema")
    if payload.get("schema_version") != "mirror_invariant_110d_v1_report":
        raise ValueError("report schema version is unsupported")

    def visit(value: object, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            expected_fields = _CLOSED_REPORT_OBJECT_FIELDS.get(path)
            if expected_fields is None or set(value) != expected_fields:
                raise ValueError("report nested fields differ from the closed schema")
            for key, child in value.items():
                if not isinstance(key, str) or key in _SENSITIVE_REPORT_KEYS:
                    raise ValueError("report contains a forbidden row-level field")
                visit(child, path + (key,))
        elif isinstance(value, (list, tuple)):
            if path not in {
                ("protocol", "candidates"),
                ("primary_comparison", "ci95"),
            }:
                raise ValueError("report arrays are forbidden outside fixed fields")
            for child in value:
                visit(child, path)
        elif isinstance(value, str):
            if (
                _OPAQUE_IDENTIFIER.search(value) is not None
                or "/" in value
                or "\\" in value
                or ".." in value
                or _WINDOWS_ABSOLUTE_PATH.match(value) is not None
                or _FILE_LIKE_SUFFIX.search(value) is not None
                or "file://" in value.lower()
            ):
                raise ValueError("report contains an identifier or local path")

    visit(payload)
    disclosure = payload.get("disclosure")
    dataset = payload.get("dataset")
    protocol = payload.get("protocol")
    counts = payload.get("counts")
    metrics = payload.get("metrics")
    comparison = payload.get("primary_comparison")
    robustness = payload.get("robustness")
    audit = payload.get("audit")
    decision = payload.get("decision")
    objects = (
        disclosure,
        dataset,
        protocol,
        counts,
        metrics,
        comparison,
        robustness,
        audit,
        decision,
    )
    if not all(isinstance(value, Mapping) for value in objects):
        raise ValueError("report aggregate sections must be objects")

    def finite_scalar(value: object) -> bool:
        return bool(
            not isinstance(value, (bool, np.bool_))
            and isinstance(value, (int, float, np.integer, np.floating))
            and np.isfinite(value)
        )

    if payload.get("claim_scope") != (
        "development_inner_oof_robustness_screen_only"
    ) or payload.get("target") != "binary_affected_vs_unaffected_not_hb_grade":
        raise ValueError("report claim boundary differs from the frozen protocol")
    if any(value is not False for value in disclosure.values()):
        raise ValueError("report disclosures must preserve development-only state")
    manifest_sha = dataset.get("collection_manifest_sha256")
    dataset_state = (
        dataset.get("claim_unit"),
        dataset.get("identity_status"),
        manifest_sha,
    )
    valid_dataset_state = (
        dataset_state[0] == "video_held_out"
        and dataset_state[1] == "unreviewed"
        and isinstance(dataset_state[2], str)
        and re.fullmatch(r"[0-9a-f]{64}", dataset_state[2]) is not None
    ) or dataset_state == ("synthetic_group_held_out", "synthetic", None)
    if (
        dataset.get("name") != "PalsyNet"
        or not valid_dataset_state
    ):
        raise ValueError("report dataset state differs from the frozen protocol")
    candidates = protocol.get("candidates")
    dimensions = protocol.get("candidate_dimensions")
    mirror_protocol = protocol.get("mirror")
    model_protocol = protocol.get("model")
    bootstrap_protocol = protocol.get("bootstrap")
    if (
        candidates != list(CANDIDATE_REGISTRY)
        or dimensions != CANDIDATE_REGISTRY
        or protocol.get("outer_fold_number") != OUTER_FOLD_NUMBER
        or protocol.get("inner_folds") != INNER_FOLDS
        or not isinstance(mirror_protocol, Mapping)
        or mirror_protocol.get("training_augmentation")
        != "original_plus_horizontal_mirror"
        or mirror_protocol.get("validation_inference")
        != "mean_original_and_horizontal_mirror_probability"
        or mirror_protocol.get("transform")
        != "frozen_clinical23_v2_horizontal_mirror"
        or mirror_protocol.get("probability_tolerance") != MIRROR_TOLERANCE
        or not isinstance(model_protocol, Mapping)
        or model_protocol.get("type")
        != "standardized_l2_logistic_regression"
        or model_protocol.get("c") != FIXED_C
        or model_protocol.get("solver") != FIXED_SOLVER
        or model_protocol.get("random_state") != FIXED_RANDOM_STATE
        or model_protocol.get("max_iter") != FIXED_MAX_ITER
        or model_protocol.get("threshold") != FIXED_THRESHOLD
        or model_protocol.get("sample_weight") != "equal_total_weight_per_group"
        or model_protocol.get("hyperparameter_search") is not False
        or not isinstance(bootstrap_protocol, Mapping)
        or bootstrap_protocol.get("paired") is not True
        or bootstrap_protocol.get("unit") != "group"
        or bootstrap_protocol.get("stratified_by_binary_label") is not True
        or bootstrap_protocol.get("repeats") != BOOTSTRAP_REPEATS
        or bootstrap_protocol.get("seed") != BOOTSTRAP_SEED
        or bootstrap_protocol.get("interval_scope")
        != "fixed_oof_predictions_descriptive"
    ):
        raise ValueError("report candidates differ from the frozen registry")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
        for section in (counts, audit)
        for value in section.values()
    ):
        raise ValueError("report counts and audit values must be nonnegative integers")
    for candidate in CANDIDATE_REGISTRY:
        candidate_metrics = metrics.get(candidate)
        if not isinstance(candidate_metrics, Mapping) or not all(
            finite_scalar(value) for value in candidate_metrics.values()
        ):
            raise ValueError("report metrics must be finite aggregate scalars")
    ci95 = comparison.get("ci95")
    if (
        comparison.get("baseline") != BASELINE
        or comparison.get("candidate") != CANDIDATE
        or not isinstance(ci95, (list, tuple))
        or len(ci95) != 2
        or not all(finite_scalar(value) for value in ci95)
        or not all(
            finite_scalar(comparison.get(name))
            for name in ("delta_auroc", "probability_delta_gt_zero")
        )
        or comparison.get("repeats") != BOOTSTRAP_REPEATS
        or comparison.get("seed") != BOOTSTRAP_SEED
        or comparison.get("interval_scope")
        != "fixed_oof_predictions_descriptive"
    ):
        raise ValueError("report comparison differs from the frozen aggregate schema")
    if not finite_scalar(robustness.get("max_mirror_probability_error")):
        raise ValueError("report mirror robustness must be one finite scalar")
    gates = decision.get("gates")
    if not isinstance(gates, Mapping) or not all(
        isinstance(value, (bool, np.bool_)) for value in gates.values()
    ):
        raise ValueError("report gates must contain only boolean scalars")
    if (
        not isinstance(decision.get("passed"), (bool, np.bool_))
        or decision.get("development_champion") not in CANDIDATE_REGISTRY
        or decision.get("next_gate")
        != "identity_review_then_prespecified_patient_disjoint_validation"
    ):
        raise ValueError("report decision fields are invalid")
    if any(
        int(audit.get(name, -1)) != 0
        for name in (
            "protected_candidate_feature_extractions",
            "protected_fits",
            "protected_predictions",
        )
    ):
        raise ValueError("private report cannot publish protected-row use")
    if (
        decision.get("outer_evaluation_authorized") is not False
        or decision.get("hb_claim_authorized") is not False
        or decision.get("clinical_validation") is not False
    ):
        raise ValueError("development report cannot authorize clinical claims")


def _write_private_report(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically publish one owner-only aggregate report without overwrite."""
    path = Path(path)
    _validate_private_report(payload)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing report {path.name!r}")
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite existing report {path.name!r}"
            ) from None
        os.chmod(path, 0o600)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--palsynet-cache-root", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    dataset = load_classical_dataset(args.palsynet_cache_root)
    result = run_development_screen(dataset)
    _write_private_report(DEFAULT_REPORT_PATH, result.report)
    print(json.dumps(result.report, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
