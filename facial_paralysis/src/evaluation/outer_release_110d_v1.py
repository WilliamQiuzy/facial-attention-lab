"""One-shot protected evaluation and release contracts for frozen Landmark 110D.

The public functions in this module operate only after the existing reviewed
identity and deterministic person-split gate has succeeded.  The thin CLI owns
the out-of-band authorization digest and must validate it before opening any
protected NPZ file.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts.run_110d_generalization_v1 import (
    BOOTSTRAP_REPEATS,
    FIXED_C,
    FIXED_MAX_ITER,
    FIXED_RANDOM_STATE,
    FIXED_SOLVER,
    FIXED_THRESHOLD,
    DevelopmentGate,
    _fast_metrics,
    _implementation_fingerprints,
    _validate_report,
    canonical_json_sha256,
)
from scripts.run_dynamic_landmark_classical import (
    ClassicalDataset,
    group_mean_predictions,
    group_sample_weights,
)
from scripts.run_mirror_invariant_110d import mirror_dynamic_features
from src.preprocessing.generalization_110d import (
    LANDMARK_MI_110D,
    candidate_feature_names,
    candidate_feature_vector,
)


LOCKED_CANDIDATE = LANDMARK_MI_110D
LOCKED_DIMENSION = 110
OUTER_BOOTSTRAP_REPEATS = 5000
OUTER_BOOTSTRAP_SEED = 20260805
DEFAULT_OUTER_REPORT_RELATIVE = (
    "outputs/dynamic_landmark/benchmarks/protected/"
    "110d-generalization-v1/report.json"
)
AUTHORIZATION_BASIS = (
    "researcher_instruction_candidate_locked_then_one_protected_outer_test"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_METRICS = (
    "auroc",
    "average_precision",
    "brier",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
)


def _model_protocol() -> dict[str, object]:
    return {
        "type": "standardized_l2_logistic_regression",
        "penalty": "l2",
        "c": FIXED_C,
        "solver": FIXED_SOLVER,
        "max_iter": FIXED_MAX_ITER,
        "random_state": FIXED_RANDOM_STATE,
        "sample_weight": "equal_total_weight_per_reviewed_group",
        "training_augmentation": "original_plus_horizontal_mirror",
        "threshold": FIXED_THRESHOLD,
    }


def release_implementation_fingerprints() -> tuple[dict[str, str], str]:
    """Bind the locked development implementation plus this release core."""
    components, _ = _implementation_fingerprints()
    merged = dict(components)
    merged["outer_release"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    project_root = Path(__file__).resolve().parents[2]
    merged["outer_runner"] = hashlib.sha256(
        (project_root / "scripts" / "run_110d_outer_release_v1.py").read_bytes()
    ).hexdigest()
    merged["artifact_freezer"] = hashlib.sha256(
        (project_root / "scripts" / "freeze_110d_generalization_v1_artifact.py").read_bytes()
    ).hexdigest()
    return merged, canonical_json_sha256(merged)


@dataclass
class OuterReleaseAudit:
    authorization_attempts: int = 0
    authorization_passes: int = 0
    development_cache_records_loaded: int = 0
    protected_cache_records_loaded: int = 0
    development_feature_extractions: int = 0
    protected_feature_extractions: int = 0
    mirror_transforms: int = 0
    scaler_fits: int = 0
    model_fits: int = 0
    protected_predictions: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            field: int(getattr(self, field))
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class AuthorizedOuterState:
    authorization_sha256: str
    release_implementation_components_sha256: Mapping[str, str]
    release_implementation_sha256: str
    development_report_sha256: str


@dataclass(frozen=True)
class LockedViews:
    original: np.ndarray
    mirrored: np.ndarray
    remirrored: np.ndarray


@dataclass(frozen=True)
class ProtectedOuterResult:
    report: dict[str, object]
    group_probabilities: np.ndarray
    group_labels: np.ndarray


@dataclass
class FinalArtifactAudit:
    protected_report_attempts: int = 0
    protected_report_passes: int = 0
    scaler_fits: int = 0
    model_fits: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            field: int(getattr(self, field))
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class SealedOuterState:
    protected_report_sha256: str


@dataclass(frozen=True)
class FrozenArtifactResult:
    artifact: dict[str, object]
    scaler: StandardScaler
    model: LogisticRegression


def load_authorized_cache_records(
    cache_root: Path,
    dataset: ClassicalDataset,
    gate: DevelopmentGate,
    collection_rows: Mapping[str, Mapping[str, object]],
    *,
    state: AuthorizedOuterState,
    audit: OuterReleaseAudit,
    record_loader: Callable[[Path], object],
) -> None:
    """Open every frozen cache row only after exact outer authorization."""
    if not isinstance(state, AuthorizedOuterState) or audit.authorization_passes != 1:
        raise ValueError("outer authorization must pass before cache loading")
    development = set(gate.development_indices.tolist())
    protected = set(gate.protected_indices.tolist())
    if development & protected or development | protected != set(range(dataset.labels.size)):
        raise ValueError("person split does not cover cache rows exactly once")
    for index, recording_id in enumerate(dataset.recording_ids):
        row = collection_rows.get(recording_id)
        if row is None:
            raise ValueError("cache record is absent from collection manifest")
        record = record_loader(Path(cache_root) / f"{recording_id}.npz")
        expected_label = 1 if row.get("label") == "affected" else 0
        if (
            getattr(record, "recording_id", None) != recording_id
            or getattr(record, "group_id", None) != row.get("group_id")
            or getattr(record, "source_sha256", None) != row.get("source_sha256")
            or getattr(record, "label", None) != expected_label
        ):
            raise ValueError("cache NPZ provenance differs from collection manifest")
        dataset.features[index] = np.asarray(record.features)
        dataset.valid_masks[index] = np.asarray(record.valid_mask)
        dataset.timestamps[index] = np.asarray(record.timestamps)
        dataset.source_frame_indices[index] = np.asarray(record.source_frame_indices)
        if index in protected:
            audit.protected_cache_records_loaded += 1
        else:
            audit.development_cache_records_loaded += 1


def build_expected_authorization(
    *,
    gate: DevelopmentGate,
    development_report_sha256: str,
    release_implementation_sha256: str,
) -> dict[str, object]:
    """Return the one exact result-free authorization payload."""
    for name, value in (
        ("development report", development_report_sha256),
        ("release implementation", release_implementation_sha256),
    ):
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"{name} SHA-256 is invalid")
    return {
        "schema_version": "110d_generalization_outer_authorization_v1",
        "dataset": "PalsyNet",
        "target": "binary_affected_vs_unaffected",
        "claim_unit": "person_held_out",
        "identity_status": "reviewed",
        "candidate": LOCKED_CANDIDATE,
        "candidate_dimension": LOCKED_DIMENSION,
        "authorized_once": True,
        "authorization_basis": AUTHORIZATION_BASIS,
        "output_relative_path": DEFAULT_OUTER_REPORT_RELATIVE,
        "reviewed_identity_manifest_sha256": gate.reviewed_manifest_sha256,
        "review_ledger_sha256": gate.review_ledger_sha256,
        "person_split_registry_sha256": gate.split_registry_sha256,
        "source_collection_sha256": gate.source_collection_sha256,
        "development_report_sha256": development_report_sha256,
        "release_implementation_sha256": release_implementation_sha256,
        "model_protocol": _model_protocol(),
        "partition_protocol": {
            "fit": "all_frozen_development_reviewed_groups_only",
            "score": "frozen_outer_zero_protected_reviewed_groups_once",
            "candidate_change_after_result": False,
            "threshold_change_after_result": False,
        },
    }


def validate_outer_authorization(
    authorization: Mapping[str, object],
    *,
    authorization_sha256: str,
    pinned_authorization_sha256: str | None,
    gate: DevelopmentGate,
    development_report: Mapping[str, object],
    development_report_sha256: str,
    audit: OuterReleaseAudit,
    expected_development_bootstrap_repeats: int = BOOTSTRAP_REPEATS,
) -> AuthorizedOuterState:
    """Authenticate the out-of-band pin before any protected operation."""
    if not isinstance(audit, OuterReleaseAudit):
        raise TypeError("audit must be OuterReleaseAudit")
    audit.authorization_attempts += 1
    if (
        not isinstance(pinned_authorization_sha256, str)
        or _SHA256.fullmatch(pinned_authorization_sha256) is None
    ):
        raise ValueError("outer authorization is not pinned out of band")
    if (
        not isinstance(authorization_sha256, str)
        or _SHA256.fullmatch(authorization_sha256) is None
        or authorization_sha256 != pinned_authorization_sha256
        or canonical_json_sha256(authorization) != authorization_sha256
    ):
        raise ValueError("outer authorization bytes differ from the pinned digest")
    if (
        not isinstance(development_report_sha256, str)
        or _SHA256.fullmatch(development_report_sha256) is None
    ):
        raise ValueError("development report digest is invalid")
    _validate_report(
        development_report,
        expected_bootstrap_repeats=expected_development_bootstrap_repeats,
    )
    decision = development_report.get("decision")
    if not isinstance(decision, Mapping) or (
        decision.get("locked_candidate") != LOCKED_CANDIDATE
        or decision.get("outer_evaluation_authorized") is not False
    ):
        raise ValueError("development report does not lock the expected 110D candidate")
    components, release_sha = release_implementation_fingerprints()
    expected = build_expected_authorization(
        gate=gate,
        development_report_sha256=development_report_sha256,
        release_implementation_sha256=release_sha,
    )
    if dict(authorization) != expected:
        raise ValueError("outer authorization fields differ from the frozen contract")
    audit.authorization_passes += 1
    return AuthorizedOuterState(
        authorization_sha256=authorization_sha256,
        release_implementation_components_sha256=components,
        release_implementation_sha256=release_sha,
        development_report_sha256=development_report_sha256,
    )


def prepare_locked_views(
    dataset: ClassicalDataset,
    gate: DevelopmentGate,
    *,
    state: AuthorizedOuterState,
    audit: OuterReleaseAudit,
) -> LockedViews:
    """Extract aligned original/mirror 110D rows after authorization."""
    if not isinstance(state, AuthorizedOuterState) or audit.authorization_passes != 1:
        raise ValueError("authorization must pass exactly once before feature extraction")
    if (
        audit.development_cache_records_loaded != gate.development_indices.size
        or audit.protected_cache_records_loaded != gate.protected_indices.size
    ):
        raise ValueError("every authenticated cache row must load before extraction")
    eligible = np.concatenate((gate.development_indices, gate.protected_indices))
    if eligible.size != dataset.labels.size or len(set(eligible.tolist())) != eligible.size:
        raise ValueError("frozen split must cover every cache row exactly once")
    original = np.zeros((dataset.labels.size, LOCKED_DIMENSION), dtype=np.float64)
    mirrored = np.zeros_like(original)
    remirrored = np.zeros_like(original)
    protected_set = set(gate.protected_indices.tolist())
    for index in eligible.tolist():
        raw = dataset.features[index]
        mirrored_raw = mirror_dynamic_features(raw)
        remirrored_raw = mirror_dynamic_features(mirrored_raw)
        audit.mirror_transforms += 2
        if not np.array_equal(raw, remirrored_raw):
            raise ValueError("horizontal mirror must remain an exact involution")
        temporal = (
            dataset.valid_masks[index],
            dataset.timestamps[index],
            dataset.source_frame_indices[index],
        )
        original[index] = candidate_feature_vector(
            LOCKED_CANDIDATE, raw, *temporal
        )
        mirrored[index] = candidate_feature_vector(
            LOCKED_CANDIDATE, mirrored_raw, *temporal
        )
        remirrored[index] = candidate_feature_vector(
            LOCKED_CANDIDATE, remirrored_raw, *temporal
        )
        if not np.array_equal(original[index], remirrored[index]):
            raise ValueError("remirrored 110D row differs from original")
        if index in protected_set:
            audit.protected_feature_extractions += 1
        else:
            audit.development_feature_extractions += 1
    for matrix in (original, mirrored, remirrored):
        if matrix.shape != (dataset.labels.size, LOCKED_DIMENSION) or not np.isfinite(matrix).all():
            raise ValueError("locked 110D view matrix is invalid")
    return LockedViews(original=original, mirrored=mirrored, remirrored=remirrored)


def _metric_report(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    repeats: int,
) -> dict[str, dict[str, object]]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != probabilities.shape or labels.ndim != 1:
        raise ValueError("group labels and probabilities must align")
    if set(labels.tolist()) != {0, 1} or not np.isfinite(probabilities).all():
        raise ValueError("protected metrics require finite scores and both classes")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("protected probabilities must be in [0, 1]")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("bootstrap repeats must be a positive integer")
    point = _fast_metrics(labels, probabilities)
    by_class = {label: np.flatnonzero(labels == label) for label in (0, 1)}
    rng = np.random.default_rng(OUTER_BOOTSTRAP_SEED)
    draws = {metric: np.empty(repeats, dtype=np.float64) for metric in _METRICS}
    for repeat in range(repeats):
        sampled = np.concatenate((
            rng.choice(by_class[0], size=by_class[0].size, replace=True),
            rng.choice(by_class[1], size=by_class[1].size, replace=True),
        ))
        values = _fast_metrics(labels[sampled], probabilities[sampled])
        for metric in _METRICS:
            draws[metric][repeat] = values[metric]
    return {
        metric: {
            "point": float(point[metric]),
            "ci95": [
                float(np.quantile(draws[metric], 0.025)),
                float(np.quantile(draws[metric], 0.975)),
            ],
        }
        for metric in _METRICS
    }


def _counts(
    dataset: ClassicalDataset,
    gate: DevelopmentGate,
    group_labels: np.ndarray,
) -> dict[str, int]:
    development_groups = set(gate.group_ids[gate.development_indices].tolist())
    protected_groups = set(gate.group_ids[gate.protected_indices].tolist())
    return {
        "eligible_recordings": int(dataset.labels.size),
        "eligible_groups": len(development_groups | protected_groups),
        "training_recordings": int(gate.development_indices.size),
        "training_groups": len(development_groups),
        "protected_recordings": int(gate.protected_indices.size),
        "protected_groups": len(protected_groups),
        "protected_affected_groups": int(np.sum(group_labels == 1)),
        "protected_unaffected_groups": int(np.sum(group_labels == 0)),
    }


def _protocol(repeats: int) -> dict[str, object]:
    return {
        "candidate": LOCKED_CANDIDATE,
        "candidate_dimension": LOCKED_DIMENSION,
        "model": _model_protocol(),
        "validation_inference": "mean_original_and_horizontal_mirror_probability",
        "aggregation": "mean_recording_probability_once_per_reviewed_group",
        "bootstrap": {
            "repeats": repeats,
            "seed": OUTER_BOOTSTRAP_SEED,
            "unit": "reviewed_group",
            "class_stratified": True,
            "interval": "percentile_95_descriptive",
        },
    }


def run_protected_outer(
    dataset: ClassicalDataset,
    gate: DevelopmentGate,
    views: LockedViews,
    *,
    state: AuthorizedOuterState,
    audit: OuterReleaseAudit,
    bootstrap_repeats: int = OUTER_BOOTSTRAP_REPEATS,
) -> ProtectedOuterResult:
    """Fit once on development groups and score protected groups once."""
    if audit.authorization_passes != 1 or not isinstance(state, AuthorizedOuterState):
        raise ValueError("authorized state is required for protected evaluation")
    development = np.asarray(gate.development_indices, dtype=np.int64)
    protected = np.asarray(gate.protected_indices, dtype=np.int64)
    if set(gate.group_ids[development]) & set(gate.group_ids[protected]):
        raise ValueError("development and protected reviewed groups overlap")
    x_train = np.concatenate((views.original[development], views.mirrored[development]))
    y_train = np.concatenate((dataset.labels[development], dataset.labels[development]))
    train_groups = np.concatenate((gate.group_ids[development], gate.group_ids[development]))
    scaler = StandardScaler().fit(x_train)
    audit.scaler_fits += 1
    model = LogisticRegression(
        C=FIXED_C,
        penalty="l2",
        solver=FIXED_SOLVER,
        max_iter=FIXED_MAX_ITER,
        random_state=FIXED_RANDOM_STATE,
    )
    model.fit(
        scaler.transform(x_train),
        y_train,
        sample_weight=group_sample_weights(train_groups),
    )
    audit.model_fits += 1
    original_probability = model.predict_proba(
        scaler.transform(views.original[protected])
    )[:, 1]
    mirrored_probability = model.predict_proba(
        scaler.transform(views.mirrored[protected])
    )[:, 1]
    probabilities = 0.5 * (original_probability + mirrored_probability)
    audit.protected_predictions += int(protected.size)
    group_labels, _, group_probabilities = group_mean_predictions(
        dataset.labels[protected], gate.group_ids[protected], probabilities
    )
    metrics = _metric_report(
        group_labels, group_probabilities, repeats=bootstrap_repeats
    )
    report: dict[str, object] = {
        "schema_version": "110d_generalization_protected_outer_v1",
        "claim_scope": "reviewed_person_group_protected_outer_once",
        "dataset": "PalsyNet",
        "target": "binary_affected_vs_unaffected",
        "counts": _counts(dataset, gate, group_labels),
        "protocol": _protocol(bootstrap_repeats),
        "metrics": metrics,
        "decision": {
            "candidate": LOCKED_CANDIDATE,
            "candidate_changed_after_result": False,
            "threshold_changed_after_result": False,
            "sealed": True,
            "clinical_validation": False,
            "hb_claim_authorized": False,
            "external_validation": False,
        },
        "audit": audit.as_dict(),
        "provenance": {
            "authorization_sha256": state.authorization_sha256,
            "development_report_sha256": state.development_report_sha256,
            "reviewed_identity_manifest_sha256": gate.reviewed_manifest_sha256,
            "review_ledger_sha256": gate.review_ledger_sha256,
            "person_split_registry_sha256": gate.split_registry_sha256,
            "source_collection_sha256": gate.source_collection_sha256,
            "release_implementation_components_sha256": dict(
                state.release_implementation_components_sha256
            ),
            "release_implementation_sha256": state.release_implementation_sha256,
        },
    }
    validate_outer_report_against_predictions(
        report,
        dataset=dataset,
        gate=gate,
        probabilities=group_probabilities,
        group_labels=group_labels,
        state=state,
        expected_bootstrap_repeats=bootstrap_repeats,
    )
    return ProtectedOuterResult(
        report=report,
        group_probabilities=group_probabilities,
        group_labels=group_labels,
    )


def validate_outer_report_against_predictions(
    report: Mapping[str, object],
    *,
    dataset: ClassicalDataset,
    gate: DevelopmentGate,
    probabilities: np.ndarray,
    group_labels: np.ndarray,
    state: AuthorizedOuterState,
    expected_bootstrap_repeats: int = OUTER_BOOTSTRAP_REPEATS,
) -> None:
    """Independently recompute all outcome-derived aggregate report fields."""
    expected_top = {
        "schema_version", "claim_scope", "dataset", "target", "counts",
        "protocol", "metrics", "decision", "audit", "provenance",
    }
    if not isinstance(report, Mapping) or set(report) != expected_top:
        raise ValueError("protected report fields differ from the closed schema")
    if (
        report["schema_version"] != "110d_generalization_protected_outer_v1"
        or report["claim_scope"] != "reviewed_person_group_protected_outer_once"
        or report["dataset"] != "PalsyNet"
        or report["target"] != "binary_affected_vs_unaffected"
        or report["counts"] != _counts(dataset, gate, np.asarray(group_labels))
        or report["protocol"] != _protocol(expected_bootstrap_repeats)
        or report["metrics"] != _metric_report(
            np.asarray(group_labels), np.asarray(probabilities),
            repeats=expected_bootstrap_repeats,
        )
    ):
        raise ValueError("protected report outcome fields failed recomputation")
    expected_decision = {
        "candidate": LOCKED_CANDIDATE,
        "candidate_changed_after_result": False,
        "threshold_changed_after_result": False,
        "sealed": True,
        "clinical_validation": False,
        "hb_claim_authorized": False,
        "external_validation": False,
    }
    if report["decision"] != expected_decision:
        raise ValueError("protected report decision fields are invalid")
    expected_provenance = {
        "authorization_sha256": state.authorization_sha256,
        "development_report_sha256": state.development_report_sha256,
        "reviewed_identity_manifest_sha256": gate.reviewed_manifest_sha256,
        "review_ledger_sha256": gate.review_ledger_sha256,
        "person_split_registry_sha256": gate.split_registry_sha256,
        "source_collection_sha256": gate.source_collection_sha256,
        "release_implementation_components_sha256": dict(
            state.release_implementation_components_sha256
        ),
        "release_implementation_sha256": state.release_implementation_sha256,
    }
    if report["provenance"] != expected_provenance:
        raise ValueError("protected report provenance is invalid")
    encoded = json.dumps(report, sort_keys=True, allow_nan=False)
    forbidden = (
        "rec_", "grp_", '"recording_ids"', '"group_ids"',
        '"probabilities"', '"predictions"', "/Users/", "\\",
    )
    if any(value in encoded for value in forbidden):
        raise ValueError("protected report leaks row outcomes, identifiers, or paths")
    audit = report["audit"]
    if not isinstance(audit, Mapping) or set(audit) != set(OuterReleaseAudit().__dict__):
        raise ValueError("protected report audit fields are invalid")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in audit.values()):
        raise ValueError("protected report audit counters must be nonnegative integers")
    expected_counts = report["counts"]
    if (
        audit["authorization_attempts"] != 1
        or audit["authorization_passes"] != 1
        or audit["development_cache_records_loaded"] != expected_counts["training_recordings"]
        or audit["protected_cache_records_loaded"] != expected_counts["protected_recordings"]
        or audit["development_feature_extractions"] != expected_counts["training_recordings"]
        or audit["protected_feature_extractions"] != expected_counts["protected_recordings"]
        or audit["mirror_transforms"] != 2 * expected_counts["eligible_recordings"]
        or audit["scaler_fits"] != 1
        or audit["model_fits"] != 1
        or audit["protected_predictions"] != expected_counts["protected_recordings"]
    ):
        raise ValueError("protected report audit counts are incoherent")


def authorize_final_artifact(
    protected_report: Mapping[str, object],
    *,
    protected_report_sha256: str,
    pinned_protected_report_sha256: str | None,
    gate: DevelopmentGate,
    state: AuthorizedOuterState,
    audit: FinalArtifactAudit,
    expected_bootstrap_repeats: int = OUTER_BOOTSTRAP_REPEATS,
) -> SealedOuterState:
    """Authenticate the exact sealed protected report before final fitting."""
    if not isinstance(audit, FinalArtifactAudit):
        raise TypeError("audit must be FinalArtifactAudit")
    audit.protected_report_attempts += 1
    if (
        not isinstance(protected_report_sha256, str)
        or _SHA256.fullmatch(protected_report_sha256) is None
        or not isinstance(pinned_protected_report_sha256, str)
        or _SHA256.fullmatch(pinned_protected_report_sha256) is None
        or protected_report_sha256 != pinned_protected_report_sha256
    ):
        raise ValueError("protected report differs from its out-of-band commitment")
    expected_top = {
        "schema_version", "claim_scope", "dataset", "target", "counts",
        "protocol", "metrics", "decision", "audit", "provenance",
    }
    if not isinstance(protected_report, Mapping) or set(protected_report) != expected_top:
        raise ValueError("protected report fields differ from the closed schema")
    counts = protected_report["counts"]
    expected_count_fields = {
        "eligible_recordings", "eligible_groups", "training_recordings",
        "training_groups", "protected_recordings", "protected_groups",
        "protected_affected_groups", "protected_unaffected_groups",
    }
    if not isinstance(counts, Mapping) or set(counts) != expected_count_fields:
        raise ValueError("protected report counts are invalid")
    development_groups = set(gate.group_ids[gate.development_indices].tolist())
    protected_groups = set(gate.group_ids[gate.protected_indices].tolist())
    if (
        protected_report["schema_version"] != "110d_generalization_protected_outer_v1"
        or protected_report["claim_scope"] != "reviewed_person_group_protected_outer_once"
        or protected_report["dataset"] != "PalsyNet"
        or protected_report["target"] != "binary_affected_vs_unaffected"
        or protected_report["protocol"] != _protocol(expected_bootstrap_repeats)
        or counts["eligible_recordings"] != gate.development_indices.size + gate.protected_indices.size
        or counts["eligible_groups"] != len(development_groups | protected_groups)
        or counts["training_recordings"] != gate.development_indices.size
        or counts["training_groups"] != len(development_groups)
        or counts["protected_recordings"] != gate.protected_indices.size
        or counts["protected_groups"] != len(protected_groups)
        or counts["protected_affected_groups"] + counts["protected_unaffected_groups"] != len(protected_groups)
    ):
        raise ValueError("protected report protocol/counts differ from the frozen split")
    metrics = protected_report["metrics"]
    if not isinstance(metrics, Mapping) or set(metrics) != set(_METRICS):
        raise ValueError("protected report metrics are invalid")
    for metric in _METRICS:
        value = metrics[metric]
        if not isinstance(value, Mapping) or set(value) != {"point", "ci95"}:
            raise ValueError("protected metric fields are invalid")
        point, interval = value["point"], value["ci95"]
        if (
            isinstance(point, bool)
            or not isinstance(point, (int, float))
            or not np.isfinite(point)
            or point < 0.0
            or point > 1.0
            or not isinstance(interval, list)
            or len(interval) != 2
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not np.isfinite(item)
                or item < 0.0
                or item > 1.0
                for item in interval
            )
            or interval[0] > interval[1]
        ):
            raise ValueError("protected metric values are invalid")
    if protected_report["decision"] != {
        "candidate": LOCKED_CANDIDATE,
        "candidate_changed_after_result": False,
        "threshold_changed_after_result": False,
        "sealed": True,
        "clinical_validation": False,
        "hb_claim_authorized": False,
        "external_validation": False,
    }:
        raise ValueError("protected report is not sealed to the locked candidate")
    if protected_report["provenance"] != {
        "authorization_sha256": state.authorization_sha256,
        "development_report_sha256": state.development_report_sha256,
        "reviewed_identity_manifest_sha256": gate.reviewed_manifest_sha256,
        "review_ledger_sha256": gate.review_ledger_sha256,
        "person_split_registry_sha256": gate.split_registry_sha256,
        "source_collection_sha256": gate.source_collection_sha256,
        "release_implementation_components_sha256": dict(
            state.release_implementation_components_sha256
        ),
        "release_implementation_sha256": state.release_implementation_sha256,
    }:
        raise ValueError("protected report provenance differs from the release state")
    encoded = json.dumps(protected_report, sort_keys=True, allow_nan=False)
    if any(value in encoded for value in ("rec_", "grp_", "/Users/", "\\")):
        raise ValueError("protected report contains identifiers or paths")
    audit.protected_report_passes += 1
    return SealedOuterState(protected_report_sha256=protected_report_sha256)


def _group_class_counts(dataset: ClassicalDataset, gate: DevelopmentGate) -> tuple[int, int]:
    labels_by_group: dict[str, set[int]] = {}
    eligible = np.concatenate((gate.development_indices, gate.protected_indices))
    for index in eligible.tolist():
        labels_by_group.setdefault(str(gate.group_ids[index]), set()).add(
            int(dataset.labels[index])
        )
    if any(len(values) != 1 for values in labels_by_group.values()):
        raise ValueError("eligible reviewed group crosses binary labels")
    affected = sum(next(iter(values)) == 1 for values in labels_by_group.values())
    return affected, len(labels_by_group) - affected


def freeze_final_artifact(
    dataset: ClassicalDataset,
    gate: DevelopmentGate,
    views: LockedViews,
    *,
    state: AuthorizedOuterState,
    sealed_outer: SealedOuterState,
    audit: FinalArtifactAudit,
) -> FrozenArtifactResult:
    """Fit the locked model once on all eligible reviewed PalsyNet groups."""
    if (
        not isinstance(state, AuthorizedOuterState)
        or not isinstance(sealed_outer, SealedOuterState)
        or audit.protected_report_passes != 1
    ):
        raise ValueError("sealed protected report is required before final fitting")
    eligible = np.concatenate((gate.development_indices, gate.protected_indices))
    if eligible.size != dataset.labels.size or len(set(eligible.tolist())) != eligible.size:
        raise ValueError("final fit requires every eligible cache row exactly once")
    x_train = np.concatenate((views.original[eligible], views.mirrored[eligible]))
    y_train = np.concatenate((dataset.labels[eligible], dataset.labels[eligible]))
    groups = np.concatenate((gate.group_ids[eligible], gate.group_ids[eligible]))
    scaler = StandardScaler().fit(x_train)
    audit.scaler_fits += 1
    model = LogisticRegression(
        C=FIXED_C,
        penalty="l2",
        solver=FIXED_SOLVER,
        max_iter=FIXED_MAX_ITER,
        random_state=FIXED_RANDOM_STATE,
    )
    model.fit(
        scaler.transform(x_train),
        y_train,
        sample_weight=group_sample_weights(groups),
    )
    audit.model_fits += 1
    if (
        scaler.mean_.shape != (LOCKED_DIMENSION,)
        or scaler.scale_.shape != (LOCKED_DIMENSION,)
        or model.coef_.shape != (1, LOCKED_DIMENSION)
        or model.intercept_.shape != (1,)
        or not all(np.isfinite(value).all() for value in (
            scaler.mean_, scaler.scale_, model.coef_, model.intercept_
        ))
        or np.any(scaler.scale_ <= 0.0)
    ):
        raise ValueError("final scaler/model parameters are invalid")
    affected_groups, unaffected_groups = _group_class_counts(dataset, gate)
    artifact: dict[str, object] = {
        "schema_version": "110d_generalization_final_artifact_v1",
        "claim_scope": "palsynet_all_eligible_refit_for_frozen_external_inference",
        "dataset": "PalsyNet",
        "target": "binary_affected_vs_unaffected",
        "representation": {
            "name": LOCKED_CANDIDATE,
            "dimension": LOCKED_DIMENSION,
            "feature_names": list(candidate_feature_names(LOCKED_CANDIDATE)),
            "frame_source": "mediapipe_facemesh_478_to_clinical23_v2",
            "video_summary": "clinical23_trajectory_92_plus_bilateral_dynamics_18",
            "mirror_inference": "mean_original_and_horizontal_mirror_probability",
        },
        "scaler": {
            "type": "standard_scaler",
            "mean": scaler.mean_.astype(float).tolist(),
            "scale": scaler.scale_.astype(float).tolist(),
        },
        "model": {
            "type": "l2_logistic_regression",
            "classes": [0, 1],
            "coefficient": model.coef_[0].astype(float).tolist(),
            "intercept": float(model.intercept_[0]),
            "c": FIXED_C,
            "solver": FIXED_SOLVER,
            "max_iter": FIXED_MAX_ITER,
            "random_state": FIXED_RANDOM_STATE,
            "sample_weight": "equal_total_weight_per_reviewed_group",
        },
        "threshold": FIXED_THRESHOLD,
        "training": {
            "eligible_recordings": int(eligible.size),
            "eligible_groups": affected_groups + unaffected_groups,
            "affected_groups": affected_groups,
            "unaffected_groups": unaffected_groups,
            "augmentation_rows_per_recording": 2,
        },
        "audit": audit.as_dict(),
        "provenance": {
            "authorization_sha256": state.authorization_sha256,
            "development_report_sha256": state.development_report_sha256,
            "protected_report_sha256": sealed_outer.protected_report_sha256,
            "reviewed_identity_manifest_sha256": gate.reviewed_manifest_sha256,
            "review_ledger_sha256": gate.review_ledger_sha256,
            "person_split_registry_sha256": gate.split_registry_sha256,
            "source_collection_sha256": gate.source_collection_sha256,
            "release_implementation_components_sha256": dict(
                state.release_implementation_components_sha256
            ),
            "release_implementation_sha256": state.release_implementation_sha256,
        },
    }
    validate_frozen_artifact(
        artifact, gate=gate, state=state, sealed_outer=sealed_outer
    )
    return FrozenArtifactResult(artifact=artifact, scaler=scaler, model=model)


def validate_frozen_artifact(
    artifact: Mapping[str, object],
    *,
    gate: DevelopmentGate,
    state: AuthorizedOuterState,
    sealed_outer: SealedOuterState,
) -> None:
    """Validate the closed, identifier-free inference artifact."""
    expected_top = {
        "schema_version", "claim_scope", "dataset", "target",
        "representation", "scaler", "model", "threshold", "training",
        "audit", "provenance",
    }
    if not isinstance(artifact, Mapping) or set(artifact) != expected_top:
        raise ValueError("final artifact fields differ from the closed schema")
    representation = artifact["representation"]
    scaler = artifact["scaler"]
    model = artifact["model"]
    if (
        artifact["schema_version"] != "110d_generalization_final_artifact_v1"
        or artifact["dataset"] != "PalsyNet"
        or artifact["target"] != "binary_affected_vs_unaffected"
        or not isinstance(representation, Mapping)
        or representation.get("name") != LOCKED_CANDIDATE
        or representation.get("dimension") != LOCKED_DIMENSION
        or representation.get("feature_names") != list(candidate_feature_names(LOCKED_CANDIDATE))
        or artifact["threshold"] != FIXED_THRESHOLD
        or not isinstance(scaler, Mapping)
        or set(scaler) != {"type", "mean", "scale"}
        or scaler["type"] != "standard_scaler"
        or not isinstance(model, Mapping)
        or set(model) != {
            "type", "classes", "coefficient", "intercept", "c", "solver",
            "max_iter", "random_state", "sample_weight",
        }
        or model["type"] != "l2_logistic_regression"
        or model["classes"] != [0, 1]
        or model["c"] != FIXED_C
        or model["solver"] != FIXED_SOLVER
        or model["max_iter"] != FIXED_MAX_ITER
        or model["random_state"] != FIXED_RANDOM_STATE
    ):
        raise ValueError("final artifact model/representation protocol is invalid")
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    coefficient = np.asarray(model["coefficient"], dtype=np.float64)
    intercept = model["intercept"]
    if (
        mean.shape != (LOCKED_DIMENSION,)
        or scale.shape != (LOCKED_DIMENSION,)
        or coefficient.shape != (LOCKED_DIMENSION,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or not np.isfinite(coefficient).all()
        or np.any(scale <= 0.0)
        or isinstance(intercept, bool)
        or not isinstance(intercept, (int, float))
        or not np.isfinite(intercept)
    ):
        raise ValueError("final artifact numeric parameters are invalid")
    training = artifact["training"]
    expected_groups = len(set(gate.group_ids.tolist()))
    if (
        not isinstance(training, Mapping)
        or set(training) != {
            "eligible_recordings", "eligible_groups", "affected_groups",
            "unaffected_groups", "augmentation_rows_per_recording",
        }
        or training["eligible_recordings"] != gate.development_indices.size + gate.protected_indices.size
        or training["eligible_groups"] != expected_groups
        or training["affected_groups"] + training["unaffected_groups"] != expected_groups
        or training["augmentation_rows_per_recording"] != 2
        or artifact["audit"] != {
            "protected_report_attempts": 1,
            "protected_report_passes": 1,
            "scaler_fits": 1,
            "model_fits": 1,
        }
    ):
        raise ValueError("final artifact training/audit counts are invalid")
    expected_provenance = {
        "authorization_sha256": state.authorization_sha256,
        "development_report_sha256": state.development_report_sha256,
        "protected_report_sha256": sealed_outer.protected_report_sha256,
        "reviewed_identity_manifest_sha256": gate.reviewed_manifest_sha256,
        "review_ledger_sha256": gate.review_ledger_sha256,
        "person_split_registry_sha256": gate.split_registry_sha256,
        "source_collection_sha256": gate.source_collection_sha256,
        "release_implementation_components_sha256": dict(
            state.release_implementation_components_sha256
        ),
        "release_implementation_sha256": state.release_implementation_sha256,
    }
    if artifact["provenance"] != expected_provenance:
        raise ValueError("final artifact provenance is invalid")
    encoded = json.dumps(artifact, sort_keys=True, allow_nan=False)
    if any(value in encoded for value in ("rec_", "grp_", "/Users/", "\\")):
        raise ValueError("final artifact contains identifiers or paths")


def predict_from_frozen_artifact(
    artifact: Mapping[str, object],
    original_110d: np.ndarray,
    mirrored_110d: np.ndarray,
) -> float:
    """Apply only serialized scaler/model parameters to one paired 110D row."""
    scaler = artifact.get("scaler")
    model = artifact.get("model")
    if not isinstance(scaler, Mapping) or not isinstance(model, Mapping):
        raise ValueError("artifact lacks scaler/model mappings")
    mean = np.asarray(scaler.get("mean"), dtype=np.float64)
    scale = np.asarray(scaler.get("scale"), dtype=np.float64)
    coefficient = np.asarray(model.get("coefficient"), dtype=np.float64)
    intercept = model.get("intercept")
    original = np.asarray(original_110d, dtype=np.float64)
    mirrored = np.asarray(mirrored_110d, dtype=np.float64)
    if any(value.shape != (LOCKED_DIMENSION,) for value in (
        mean, scale, coefficient, original, mirrored
    )) or not all(np.isfinite(value).all() for value in (
        mean, scale, coefficient, original, mirrored
    )) or np.any(scale <= 0.0) or not isinstance(intercept, (int, float)):
        raise ValueError("serialized inference inputs are invalid")

    def probability(row: np.ndarray) -> float:
        score = float(((row - mean) / scale) @ coefficient + float(intercept))
        if score >= 0.0:
            return float(1.0 / (1.0 + np.exp(-score)))
        exponential = float(np.exp(score))
        return exponential / (1.0 + exponential)

    return 0.5 * (probability(original) + probability(mirrored))


def write_private_no_overwrite_json(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically publish one owner-private JSON artifact without overwrite."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(target.parent, 0o700)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite {target.name}")
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
        os.chmod(target, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "AUTHORIZATION_BASIS",
    "AuthorizedOuterState",
    "DEFAULT_OUTER_REPORT_RELATIVE",
    "FinalArtifactAudit",
    "FrozenArtifactResult",
    "LOCKED_CANDIDATE",
    "LOCKED_DIMENSION",
    "LockedViews",
    "OUTER_BOOTSTRAP_REPEATS",
    "OUTER_BOOTSTRAP_SEED",
    "OuterReleaseAudit",
    "ProtectedOuterResult",
    "SealedOuterState",
    "authorize_final_artifact",
    "build_expected_authorization",
    "freeze_final_artifact",
    "load_authorized_cache_records",
    "prepare_locked_views",
    "predict_from_frozen_artifact",
    "release_implementation_fingerprints",
    "run_protected_outer",
    "validate_outer_authorization",
    "validate_outer_report_against_predictions",
    "validate_frozen_artifact",
    "write_private_no_overwrite_json",
]
