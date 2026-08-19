#!/usr/bin/env python3
"""Run the fixed development-only clinical-geometry nuisance challenge.

This successor protocol is informed by prior exploratory development analysis.
It never extracts candidate features, fits models, or predicts on outer fold 0.
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
from src.preprocessing.clinical_dynamics import (  # noqa: E402
    CLINICAL_DYNAMICS_DIM,
    clinical_dynamics_feature_names,
    clinical_dynamics_feature_vector,
)
from src.preprocessing.trajectory_features import (  # noqa: E402
    LANDMARK_DIM,
    trajectory_feature_set,
)


FIXED_C = 0.01
FIXED_THRESHOLD = 0.5
FIXED_SOLVER = "liblinear"
FIXED_RANDOM_STATE = 0
FIXED_MAX_ITER = 2000
OUTER_FOLD_NUMBER = 0
INNER_FOLDS = _INNER_FOLDS
BOOTSTRAP_REPEATS = 5000
BOOTSTRAP_SEED = 20260722
MINIMUM_INCREMENTAL_AUROC = 0.10
MINIMUM_SENSITIVITY = 0.70
MINIMUM_SPECIFICITY = 0.70
CANDIDATE_REGISTRY = {
    "nuisance": 9,
    "landmark": LANDMARK_DIM,
    "clinical_dynamics": CLINICAL_DYNAMICS_DIM,
    "clinical_dynamics_plus_nuisance": CLINICAL_DYNAMICS_DIM + 9,
}
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
    "audit",
    "decision",
}
_CLOSED_REPORT_OBJECT_FIELDS = {
    (): _REPORT_TOP_LEVEL_FIELDS,
    ("disclosure",): {
        "exploratory_analysis_informed_successor",
        "true_action_labels_available",
        "representation_scope",
        "unrecorded_nuisance_excluded",
        "clinical_validation",
    },
    ("dataset",): {
        "name", "claim_unit", "identity_status", "collection_manifest_sha256",
    },
    ("protocol",): {
        "outer_fold_number", "inner_folds", "candidates",
        "candidate_dimensions", "clinical_dynamics_feature_names",
        "region_feature_counts", "lag", "model", "bootstrap",
    },
    ("protocol", "candidate_dimensions"): set(CANDIDATE_REGISTRY),
    ("protocol", "region_feature_counts"): {"eye", "brow", "mouth"},
    ("protocol", "lag"): {
        "maximum_frames", "unit", "tie_break", "validity",
    },
    ("protocol", "model"): {
        "type", "c", "solver", "random_state", "max_iter", "threshold",
        "sample_weight", "hyperparameter_search",
    },
    ("protocol", "bootstrap"): {
        "paired", "unit", "stratified_by_binary_label", "repeats", "seed",
        "interval_scope",
    },
    ("counts",): {
        "dataset_recordings", "dataset_groups", "development_recordings",
        "development_groups", "development_affected_groups",
        "development_unaffected_groups", "protected_recordings",
        "protected_groups",
    },
    ("metrics",): set(CANDIDATE_REGISTRY),
    **{
        ("metrics", candidate): {
            "auroc", "average_precision", "brier", "balanced_accuracy",
            "sensitivity", "specificity",
        }
        for candidate in CANDIDATE_REGISTRY
    },
    ("primary_comparison",): {
        "baseline", "candidate", "delta_auroc", "ci95",
        "probability_delta_gt_zero", "repeats", "seed", "interval_scope",
    },
    ("audit",): {
        "development_feature_extractions", "development_scaler_fits",
        "development_model_fits", "development_prediction_folds",
        "protected_feature_extractions", "protected_fits",
        "protected_predictions",
    },
    ("decision",): {
        "passed", "gates", "if_passed", "outer_evaluation_authorized",
        "hb_claim_authorized",
    },
    ("decision", "gates"): {
        "incremental_delta_auroc_at_least_0_10",
        "paired_ci95_lower_bound_above_zero",
        "clinical_dynamics_sensitivity_at_least_0_70",
        "clinical_dynamics_specificity_at_least_0_70",
        "zero_protected_feature_extractions_fits_predictions",
    },
}
_SENSITIVE_REPORT_KEYS = {
    "recording_id",
    "recording_ids",
    "record_ids",
    "group_id",
    "group_ids",
    "source_path",
    "source_paths",
    "raw_features",
    "features",
    "probability",
    "probabilities",
    "predictions_by_record",
}
_OPAQUE_IDENTIFIER = re.compile(r"\b(?:rec|grp)_[A-Za-z0-9._-]+\b")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_FILE_LIKE_SUFFIX = re.compile(
    r"\.(?:json|jsonl|csv|tsv|npy|npz|pt|pth|mp4|mov|avi|mkv)$", re.IGNORECASE
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "dynamic_landmark"
    / "benchmarks"
    / "development"
    / "action-clinical-geometry-v1"
    / "report.json"
)


@dataclass(frozen=True)
class IndexAuditEvent:
    """Global record indices used by one development-only operation."""

    candidate: str
    inner_fold: int
    operation: str
    indices: tuple[int, ...]


@dataclass(frozen=True)
class DevelopmentCandidateMatrices:
    """Candidate rows aligned exactly to the frozen outer-train indices."""

    development_indices: np.ndarray
    protected_indices: np.ndarray
    matrices: Mapping[str, np.ndarray]
    extraction_indices: tuple[int, ...]


@dataclass(frozen=True)
class DevelopmentOOFResult:
    candidate: str
    probabilities: np.ndarray
    audit_events: tuple[IndexAuditEvent, ...]


@dataclass(frozen=True)
class DevelopmentScreenResult:
    report: dict[str, object]
    candidate_results: Mapping[str, DevelopmentOOFResult]


ClinicalExtractor = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray
]


def build_development_candidate_matrices(
    dataset: ClassicalDataset,
    *,
    clinical_extractor: ClinicalExtractor = clinical_dynamics_feature_vector,
) -> DevelopmentCandidateMatrices:
    """Extract all candidates only for outer-fold-0 development records."""
    if not callable(clinical_extractor):
        raise ValueError("clinical_extractor must be callable")
    outer = build_nested_group_splits(dataset.labels, dataset.group_ids)[
        OUTER_FOLD_NUMBER
    ]
    development = np.asarray(outer.train_indices, dtype=np.int64)
    protected = np.asarray(outer.test_indices, dtype=np.int64)
    assert_outer_test_isolation(
        protected,
        feature_extraction_indices=development,
    )

    landmark_rows: list[np.ndarray] = []
    clinical_rows: list[np.ndarray] = []
    for global_index in development.tolist():
        arrays = (
            dataset.features[global_index],
            dataset.valid_masks[global_index],
            dataset.timestamps[global_index],
            dataset.source_frame_indices[global_index],
        )
        landmark_rows.append(trajectory_feature_set("landmark", *arrays))
        clinical_rows.append(np.asarray(clinical_extractor(*arrays), dtype=np.float64))
    landmark = np.stack(landmark_rows)
    clinical = np.stack(clinical_rows)
    nuisance = dataset.nuisance[development].astype(np.float64, copy=True)
    matrices = {
        "nuisance": nuisance,
        "landmark": landmark,
        "clinical_dynamics": clinical,
        "clinical_dynamics_plus_nuisance": np.concatenate(
            (clinical, nuisance), axis=1
        ),
    }
    expected_rows = development.size
    for candidate, expected_dimension in CANDIDATE_REGISTRY.items():
        matrix = matrices[candidate]
        if matrix.shape != (expected_rows, expected_dimension):
            raise ValueError(
                f"{candidate} matrix must have shape "
                f"({expected_rows}, {expected_dimension})"
            )
        if not np.isfinite(matrix).all():
            raise ValueError(f"{candidate} matrix contains nonfinite values")
    return DevelopmentCandidateMatrices(
        development_indices=development.copy(),
        protected_indices=protected.copy(),
        matrices=matrices,
        extraction_indices=tuple(int(index) for index in development),
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


def run_fixed_inner_oof(
    dataset: ClassicalDataset,
    prepared: DevelopmentCandidateMatrices,
    candidate: str,
) -> DevelopmentOOFResult:
    """Fit one fixed candidate in four grouped inner folds, without selection."""
    if candidate not in CANDIDATE_REGISTRY:
        raise ValueError(f"unknown fixed candidate {candidate!r}")
    outer = build_nested_group_splits(dataset.labels, dataset.group_ids)[
        OUTER_FOLD_NUMBER
    ]
    if not np.array_equal(prepared.development_indices, outer.train_indices):
        raise ValueError("development matrix rows differ from the frozen outer train")
    if not np.array_equal(prepared.protected_indices, outer.test_indices):
        raise ValueError("protected indices differ from the frozen outer fold")
    matrix = np.asarray(prepared.matrices[candidate], dtype=np.float64)
    if matrix.shape != (
        prepared.development_indices.size,
        CANDIDATE_REGISTRY[candidate],
    ):
        raise ValueError("candidate matrix differs from its frozen dimensions")
    local_by_global = {
        int(global_index): local_index
        for local_index, global_index in enumerate(
            prepared.development_indices.tolist()
        )
    }
    probabilities = np.full(prepared.development_indices.size, np.nan)
    prediction_counts = np.zeros(prepared.development_indices.size, dtype=np.int64)
    events: list[IndexAuditEvent] = []

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
        scaler = StandardScaler().fit(matrix[train_local])
        events.append(IndexAuditEvent(
            candidate,
            inner_number,
            "scaler_fit",
            tuple(int(index) for index in inner.train_indices),
        ))
        model = _fit_fixed_logistic(
            scaler.transform(matrix[train_local]),
            dataset.labels[inner.train_indices],
            dataset.group_ids[inner.train_indices],
        )
        events.append(IndexAuditEvent(
            candidate,
            inner_number,
            "model_fit",
            tuple(int(index) for index in inner.train_indices),
        ))
        probabilities[validation_local] = model.predict_proba(
            scaler.transform(matrix[validation_local])
        )[:, 1]
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
    )


def _count_groups(dataset: ClassicalDataset, indices: np.ndarray) -> int:
    return len(set(dataset.group_ids[indices].tolist()))


def _evaluate_gates(
    metrics: Mapping[str, Mapping[str, float]],
    comparison: Mapping[str, object],
    audit: Mapping[str, int],
) -> dict[str, bool]:
    """Derive every pass/fail gate from unrounded observed evidence."""
    clinical = metrics["clinical_dynamics"]
    ci95 = comparison["ci95"]
    if not isinstance(ci95, (list, tuple)) or len(ci95) != 2:
        raise ValueError("primary comparison must provide a two-sided CI")
    return {
        "incremental_delta_auroc_at_least_0_10": bool(
            float(comparison["delta_auroc"]) >= MINIMUM_INCREMENTAL_AUROC
        ),
        "paired_ci95_lower_bound_above_zero": bool(float(ci95[0]) > 0.0),
        "clinical_dynamics_sensitivity_at_least_0_70": bool(
            float(clinical["sensitivity"]) >= MINIMUM_SENSITIVITY
        ),
        "clinical_dynamics_specificity_at_least_0_70": bool(
            float(clinical["specificity"]) >= MINIMUM_SPECIFICITY
        ),
        "zero_protected_feature_extractions_fits_predictions": all(
            int(audit[name]) == 0
            for name in (
                "protected_feature_extractions",
                "protected_fits",
                "protected_predictions",
            )
        ),
    }


def run_development_screen(dataset: ClassicalDataset) -> DevelopmentScreenResult:
    """Run all frozen candidates and build a deidentified aggregate report."""
    prepared = build_development_candidate_matrices(dataset)
    results = {
        candidate: run_fixed_inner_oof(dataset, prepared, candidate)
        for candidate in CANDIDATE_REGISTRY
    }
    development = prepared.development_indices
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
        results["nuisance"].probabilities,
        results["clinical_dynamics_plus_nuisance"].probabilities,
        repeats=BOOTSTRAP_REPEATS,
        seed=BOOTSTRAP_SEED,
    )
    primary_comparison = {
        "baseline": "nuisance",
        "candidate": "clinical_dynamics_plus_nuisance",
        **comparison,
    }
    protected = prepared.protected_indices
    protected_set = set(protected.tolist())
    extraction_set = set(prepared.extraction_indices)
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
    fit_index_set = {
        index for event in fit_events for index in event.indices
    }
    prediction_index_set = {
        index for event in prediction_events for index in event.indices
    }
    audit = {
        "development_feature_extractions": len(extraction_set),
        "development_scaler_fits": sum(
            event.operation == "scaler_fit" for event in fit_events
        ),
        "development_model_fits": sum(
            event.operation == "model_fit" for event in fit_events
        ),
        "development_prediction_folds": len(prediction_events),
        "protected_feature_extractions": len(extraction_set & protected_set),
        "protected_fits": len(fit_index_set & protected_set),
        "protected_predictions": len(prediction_index_set & protected_set),
    }
    gates = _evaluate_gates(metrics, comparison, audit)
    passed = all(gates.values())

    report: dict[str, object] = {
        "schema_version": "action_clinical_geometry_v1_report",
        "claim_scope": "development_inner_oof_direction_screen_only",
        "target": "binary_affected_vs_unaffected_not_hb_grade",
        "disclosure": {
            "exploratory_analysis_informed_successor": True,
            "true_action_labels_available": False,
            "representation_scope": "clinical_region_action_proxy_only",
            "unrecorded_nuisance_excluded": False,
            "clinical_validation": False,
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
            "clinical_dynamics_feature_names": list(
                clinical_dynamics_feature_names()
            ),
            "region_feature_counts": {"eye": 27, "brow": 9, "mouth": 22},
            "lag": {
                "maximum_frames": 5,
                "unit": "seconds",
                "tie_break": "zero_then_smaller_absolute_lag_then_negative",
                "validity": "within_window_valid_endpoints_intermediate_continuity_not_enforced",
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
            "development_affected_groups": len(set(
                groups[labels == 1].tolist()
            )),
            "development_unaffected_groups": len(set(
                groups[labels == 0].tolist()
            )),
            "protected_recordings": int(protected.size),
            "protected_groups": _count_groups(dataset, protected),
        },
        "metrics": metrics,
        "primary_comparison": primary_comparison,
        "audit": audit,
        "decision": {
            "passed": passed,
            "gates": gates,
            "if_passed": "freeze_successor_then_review_identity_and_action_coverage_before_outer_evaluation",
            "outer_evaluation_authorized": False,
            "hb_claim_authorized": False,
        },
    }
    return DevelopmentScreenResult(report=report, candidate_results=results)


def _validate_private_report(payload: Mapping[str, object]) -> None:
    """Fail closed on schema drift, row-level fields, IDs, and local paths."""
    if not isinstance(payload, Mapping) or set(payload) != _REPORT_TOP_LEVEL_FIELDS:
        raise ValueError("report top-level fields differ from the closed schema")
    if payload.get("schema_version") != "action_clinical_geometry_v1_report":
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
    audit = payload.get("audit")
    decision = payload.get("decision")
    if not isinstance(audit, Mapping) or not isinstance(decision, Mapping):
        raise ValueError("report audit and decision must be objects")
    if any(int(audit.get(name, -1)) != 0 for name in (
        "protected_feature_extractions", "protected_fits", "protected_predictions"
    )):
        raise ValueError("private report cannot publish protected-row use")
    if decision.get("outer_evaluation_authorized") is not False or (
        decision.get("hb_claim_authorized") is not False
    ):
        raise ValueError("development report cannot authorize outer or HB claims")


def _write_private_report(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically publish one owner-only report without overwriting."""
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
