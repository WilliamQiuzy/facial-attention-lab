"""Closed participant-level NeuroFace evaluation for the frozen 110D artifact."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)

from scripts.run_mirror_invariant_110d import FIXED_THRESHOLD, mirror_dynamic_features
from src.evaluation.meei_external_v1 import (
    CacheArtifactInventory,
    _record_from_authenticated_bytes,
    validate_frozen_artifact_for_external,
)
from src.evaluation.outer_release_110d_v1 import predict_from_frozen_artifact
from src.preprocessing.generalization_110d import LANDMARK_MI_110D, candidate_feature_vector


SCHEMA_VERSION = "neuroface_external_110d_v1"
AUTHORIZATION_SCHEMA_VERSION = "neuroface_external_110d_authorization_v1"
DEFAULT_REPORT_RELATIVE = "outputs/neuroface_external_v1/report.json"
BOOTSTRAP_REPEATS = 5000
BOOTSTRAP_SEED = 20260813
PRIMARY_TASKS = ("NSM_KISS", "NSM_OPEN", "NSM_SPREAD")
METRICS = (
    "auroc", "average_precision", "brier", "accuracy",
    "balanced_accuracy", "sensitivity", "specificity",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GROUP_ID = re.compile(r"^grp_[0-9a-f]{64}$")


@dataclass
class ExternalAudit:
    authorization_attempts: int = 0
    authorization_passes: int = 0
    cache_artifacts_hashed: int = 0
    cache_records_loaded: int = 0
    feature_extractions: int = 0
    mirror_transforms: int = 0
    artifact_predictions: int = 0
    participant_aggregations: int = 0
    scaler_fits: int = 0
    model_fits: int = 0
    calibration_fits: int = 0

    def as_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class AuthorizedState:
    authorization_sha256: str
    preanalysis_registration_sha256: str
    final_artifact_sha256: str
    private_manifest_sha256: str
    cache_manifest_sha256: str
    cache_artifact_collection_sha256: str
    implementation_sha256: str
    dependency_lock_sha256: str
    expected_participants: int
    expected_affected: int
    expected_unaffected: int
    expected_videos: int


@dataclass(frozen=True)
class ParticipantAggregate:
    labels: np.ndarray
    primary_scores: np.ndarray
    all_task_scores: np.ndarray
    cohorts: np.ndarray
    primary_task_scores: Mapping[str, np.ndarray]
    all_task_counts: np.ndarray


def canonical_json_sha256(payload: object) -> str:
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _positive(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def protocol() -> dict[str, object]:
    return {
        "candidate_selection": False,
        "model_refit": False,
        "scaler_refit": False,
        "calibration": False,
        "threshold_selection": False,
        "representation": LANDMARK_MI_110D,
        "dimension": 110,
        "score_direction": "higher_is_more_affected_like",
        "mirror_inference": "mean_original_and_horizontal_mirror_score",
        "primary_tasks": list(PRIMARY_TASKS),
        "primary_aggregation": "unweighted_mean_of_exact_three_task_scores",
        "secondary_aggregation": "unweighted_mean_of_all_available_eligible_tasks",
        "participant_eligibility": "all_three_primary_tasks_pass_90_percent_video_coverage",
        "missingness": "no_imputation",
        "threshold": FIXED_THRESHOLD,
        "bootstrap": {
            "method": "als_post_stroke_healthy_control_stratified_participant_percentile",
            "repeats": BOOTSTRAP_REPEATS,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
            "minimum_valid_draws": 4750,
        },
    }


def build_expected_authorization(
    *,
    preanalysis_registration_sha256: str,
    final_artifact_sha256: str,
    private_manifest_sha256: str,
    cache_manifest_sha256: str,
    cache_artifact_collection_sha256: str,
    implementation_sha256: str,
    dependency_lock_sha256: str,
    expected_participants: int,
    expected_affected: int,
    expected_unaffected: int,
    expected_videos: int,
) -> dict[str, object]:
    hashes = {
        "preanalysis_registration_sha256": preanalysis_registration_sha256,
        "final_artifact_sha256": final_artifact_sha256,
        "private_manifest_sha256": private_manifest_sha256,
        "cache_manifest_sha256": cache_manifest_sha256,
        "cache_artifact_collection_sha256": cache_artifact_collection_sha256,
        "implementation_sha256": implementation_sha256,
        "dependency_lock_sha256": dependency_lock_sha256,
    }
    for name, value in hashes.items():
        _sha(value, name)
    participants = _positive(expected_participants, "expected participants")
    affected = _positive(expected_affected, "expected affected")
    unaffected = _positive(expected_unaffected, "expected unaffected")
    videos = _positive(expected_videos, "expected videos")
    if affected + unaffected != participants or videos < participants * 3:
        raise ValueError("authorization population counts are inconsistent")
    return {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "authorization_basis": "locked_frozen_110d_cross_disease_transfer",
        "dataset": "Toronto_NeuroFace_v1",
        "target": "neurological_orofacial_impairment_vs_healthy_control",
        **hashes,
        "expected_participants": participants,
        "expected_affected": affected,
        "expected_unaffected": unaffected,
        "expected_videos": videos,
        "protocol": protocol(),
        "output_relative_path": DEFAULT_REPORT_RELATIVE,
        "authorized_once": True,
    }


def validate_external_authorization(
    authorization: Mapping[str, object],
    *,
    authorization_sha256: str,
    pinned_authorization_sha256: str | None,
    audit: ExternalAudit,
    **expected_inputs: object,
) -> AuthorizedState:
    audit.authorization_attempts += 1
    observed = _sha(authorization_sha256, "authorization")
    if pinned_authorization_sha256 is None:
        raise ValueError("NeuroFace one-shot authorization pin is not activated")
    if _sha(pinned_authorization_sha256, "pinned authorization") != observed:
        raise ValueError("NeuroFace authorization differs from the out-of-band pin")
    if canonical_json_sha256(authorization) != observed:
        raise ValueError("NeuroFace authorization is not exact canonical JSON")
    expected = build_expected_authorization(**expected_inputs)
    if dict(authorization) != expected:
        raise ValueError("NeuroFace authorization fields differ from the frozen contract")
    audit.authorization_passes += 1
    return AuthorizedState(
        authorization_sha256=observed,
        preanalysis_registration_sha256=str(expected_inputs["preanalysis_registration_sha256"]),
        final_artifact_sha256=str(expected_inputs["final_artifact_sha256"]),
        private_manifest_sha256=str(expected_inputs["private_manifest_sha256"]),
        cache_manifest_sha256=str(expected_inputs["cache_manifest_sha256"]),
        cache_artifact_collection_sha256=str(expected_inputs["cache_artifact_collection_sha256"]),
        implementation_sha256=str(expected_inputs["implementation_sha256"]),
        dependency_lock_sha256=str(expected_inputs["dependency_lock_sha256"]),
        expected_participants=int(expected_inputs["expected_participants"]),
        expected_affected=int(expected_inputs["expected_affected"]),
        expected_unaffected=int(expected_inputs["expected_unaffected"]),
        expected_videos=int(expected_inputs["expected_videos"]),
    )


def aggregate_participant_scores(
    video_rows: Sequence[Mapping[str, object]],
) -> ParticipantAggregate:
    by_participant: dict[str, list[Mapping[str, object]]] = {}
    for row in video_rows:
        participant = row.get("participant_id")
        probability = row.get("probability")
        if (
            not isinstance(participant, str) or _GROUP_ID.fullmatch(participant) is None
            or isinstance(probability, bool) or not isinstance(probability, (int, float))
            or not np.isfinite(probability) or not 0.0 <= float(probability) <= 1.0
        ):
            raise ValueError("video score row is invalid")
        by_participant.setdefault(participant, []).append(row)
    if not by_participant:
        raise ValueError("participant aggregation requires video scores")
    labels: list[int] = []
    primary_scores: list[float] = []
    all_scores: list[float] = []
    cohorts: list[str] = []
    all_counts: list[int] = []
    per_task: dict[str, list[float]] = {task: [] for task in PRIMARY_TASKS}
    for participant in sorted(by_participant):
        rows = by_participant[participant]
        row_labels = {row.get("label") for row in rows}
        row_cohorts = {row.get("cohort") for row in rows}
        if len(row_labels) != 1 or row_labels - {0, 1} or len(row_cohorts) != 1:
            raise ValueError("participant crosses label or cohort")
        cohort = next(iter(row_cohorts))
        if cohort not in {"als", "healthy_control", "post_stroke"}:
            raise ValueError("participant cohort is invalid")
        task_rows: dict[str, list[float]] = {}
        for row in rows:
            task_rows.setdefault(str(row.get("task")), []).append(float(row["probability"]))
        if any(len(task_rows.get(task, [])) != 1 for task in PRIMARY_TASKS):
            raise ValueError("participant lacks exactly one score for every primary task")
        labels.append(int(next(iter(row_labels))))
        cohorts.append(str(cohort))
        primary_values = [task_rows[task][0] for task in PRIMARY_TASKS]
        primary_scores.append(float(np.mean(primary_values)))
        for task, value in zip(PRIMARY_TASKS, primary_values):
            per_task[task].append(value)
        all_values = [float(row["probability"]) for row in rows]
        all_scores.append(float(np.mean(all_values)))
        all_counts.append(len(all_values))
    return ParticipantAggregate(
        labels=np.asarray(labels, dtype=np.int64),
        primary_scores=np.asarray(primary_scores, dtype=np.float64),
        all_task_scores=np.asarray(all_scores, dtype=np.float64),
        cohorts=np.asarray(cohorts, dtype=str),
        primary_task_scores={task: np.asarray(values, dtype=np.float64)
                             for task, values in per_task.items()},
        all_task_counts=np.asarray(all_counts, dtype=np.int64),
    )


def _points(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1 or set(labels.tolist()) != {0, 1}:
        raise ValueError("binary participant metrics require aligned scores and both classes")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("participant scores must be finite probabilities")
    predictions = scores >= FIXED_THRESHOLD
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity": float(np.mean(predictions[labels == 1])),
        "specificity": float(np.mean(~predictions[labels == 0])),
    }


def metric_report(
    labels: np.ndarray,
    scores: np.ndarray,
    cohorts: np.ndarray,
    *,
    repeats: int = BOOTSTRAP_REPEATS,
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    cohorts = np.asarray(cohorts, dtype=str)
    if labels.shape != scores.shape or labels.shape != cohorts.shape:
        raise ValueError("labels, scores, and cohorts must align")
    expected_cohorts = set(cohorts.tolist())
    if not expected_cohorts.issubset({"als", "healthy_control", "post_stroke"}):
        raise ValueError("bootstrap cohort is invalid")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("bootstrap repeats must be positive")
    point = _points(labels, scores)
    strata = [np.flatnonzero(cohorts == cohort) for cohort in sorted(expected_cohorts)]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = {metric: [] for metric in METRICS}
    invalid = 0
    for _ in range(repeats):
        sampled = np.concatenate([
            rng.choice(indices, size=indices.size, replace=True) for indices in strata
        ])
        try:
            values = _points(labels[sampled], scores[sampled])
        except ValueError:
            invalid += 1
            continue
        for metric in METRICS:
            draws[metric].append(values[metric])
    valid = repeats - invalid
    minimum = min(repeats, int(np.ceil(repeats * 0.95)))
    if valid < minimum:
        raise ValueError("fewer than 95 percent of bootstrap draws are valid")
    output: dict[str, object] = {
        metric: {
            "point": float(point[metric]),
            "ci95_low": float(np.quantile(draws[metric], 0.025)),
            "ci95_high": float(np.quantile(draws[metric], 0.975)),
        }
        for metric in METRICS
    }
    output["bootstrap_draws"] = {"requested": repeats, "valid": valid, "invalid": invalid}
    return output


def _endpoint(
    labels: np.ndarray,
    scores: np.ndarray,
    cohorts: np.ndarray,
    *,
    repeats: int,
) -> dict[str, object]:
    return {
        "counts": {
            "participants": int(labels.size),
            "affected": int(np.sum(labels == 1)),
            "unaffected": int(np.sum(labels == 0)),
        },
        "score_summary": {
            cohort: {
                "mean": float(np.mean(scores[cohorts == cohort])),
                "median": float(np.median(scores[cohorts == cohort])),
            }
            for cohort in sorted(set(cohorts.tolist()))
        },
        "metrics": metric_report(labels, scores, cohorts, repeats=repeats),
    }


def build_external_report(
    aggregate: ParticipantAggregate,
    *,
    state: AuthorizedState,
    audit: ExternalAudit,
    provenance: Mapping[str, object],
    bootstrap_repeats: int = BOOTSTRAP_REPEATS,
) -> dict[str, object]:
    labels, scores, cohorts = aggregate.labels, aggregate.primary_scores, aggregate.cohorts
    if labels.size != state.expected_participants:
        # Synthetic unit tests intentionally use a smaller cohort while still
        # checking the immutable real-data authorization schema.
        if not (bootstrap_repeats != BOOTSTRAP_REPEATS and labels.size >= 4):
            raise ValueError("participant count differs from authorization")
    endpoints: dict[str, object] = {
        "primary_three_task": _endpoint(labels, scores, cohorts, repeats=bootstrap_repeats),
        "secondary_all_available_tasks": _endpoint(
            labels, aggregate.all_task_scores, cohorts, repeats=bootstrap_repeats
        ),
    }
    for cohort in ("als", "post_stroke"):
        mask = np.isin(cohorts, ["healthy_control", cohort])
        endpoints[f"secondary_{cohort}_vs_healthy_control"] = _endpoint(
            labels[mask], scores[mask], cohorts[mask], repeats=bootstrap_repeats
        )
    for task in PRIMARY_TASKS:
        endpoints[f"descriptive_task_{task.lower()}"] = _endpoint(
            labels, aggregate.primary_task_scores[task], cohorts,
            repeats=bootstrap_repeats,
        )
    merged_provenance = {
        "authorization_sha256": state.authorization_sha256,
        "preanalysis_registration_sha256": state.preanalysis_registration_sha256,
        "final_artifact_sha256": state.final_artifact_sha256,
        "private_manifest_sha256": state.private_manifest_sha256,
        "cache_manifest_sha256": state.cache_manifest_sha256,
        "cache_artifact_collection_sha256": state.cache_artifact_collection_sha256,
        "implementation_sha256": state.implementation_sha256,
        "dependency_lock_sha256": state.dependency_lock_sha256,
        **dict(provenance),
    }
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": "held_fixed_cross_disease_external_transfer_not_clinical_validation",
        "dataset": "Toronto_NeuroFace_v1",
        "target": "neurological_orofacial_impairment_vs_healthy_control",
        "model": "frozen_palsynet_landmark_mi_110d",
        "protocol": {**protocol(), "bootstrap": {
            **protocol()["bootstrap"], "repeats": bootstrap_repeats,
            "minimum_valid_draws": int(np.ceil(bootstrap_repeats * 0.95)),
        }},
        "counts": {
            "participants_scored": int(labels.size),
            "affected_participants": int(np.sum(labels == 1)),
            "unaffected_participants": int(np.sum(labels == 0)),
            "videos_scored": int(audit.artifact_predictions),
            "primary_complete_participants": int(labels.size),
            "all_task_count_min": int(np.min(aggregate.all_task_counts)),
            "all_task_count_max": int(np.max(aggregate.all_task_counts)),
        },
        "endpoints": endpoints,
        "clinical_boundary": {
            "bell_palsy_endpoint": False,
            "house_brackmann_endpoint": False,
            "mayo_accuracy_claim": False,
            "slp_associations": "exploratory_and_reported_separately",
        },
        "audit": audit.as_dict(),
        "provenance": merged_provenance,
    }
    encoded = json.dumps(report, sort_keys=True, allow_nan=False)
    if any(token in encoded for token in (
        "rec_", "grp_", "participant_id", "/Users/", "\\", ".avi", "probabilities"
    )):
        raise ValueError("aggregate report leaks row-level or local information")
    return report


def validate_external_report(
    report: Mapping[str, object],
    *,
    aggregate: ParticipantAggregate,
    state: AuthorizedState,
    audit: ExternalAudit,
    expected_bootstrap_repeats: int = BOOTSTRAP_REPEATS,
) -> None:
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("report provenance is missing")
    required = {
        "authorization_sha256": state.authorization_sha256,
        "preanalysis_registration_sha256": state.preanalysis_registration_sha256,
        "final_artifact_sha256": state.final_artifact_sha256,
        "private_manifest_sha256": state.private_manifest_sha256,
        "cache_manifest_sha256": state.cache_manifest_sha256,
        "cache_artifact_collection_sha256": state.cache_artifact_collection_sha256,
        "implementation_sha256": state.implementation_sha256,
        "dependency_lock_sha256": state.dependency_lock_sha256,
    }
    if any(provenance.get(key) != value for key, value in required.items()):
        raise ValueError("report provenance differs from authorization")
    expected = build_external_report(
        aggregate,
        state=state,
        audit=audit,
        provenance={key: value for key, value in provenance.items() if key not in required},
        bootstrap_repeats=expected_bootstrap_repeats,
    )
    if dict(report) != expected:
        raise ValueError("report differs from independent deterministic recomputation")


def score_authenticated_cache(
    inventory: CacheArtifactInventory,
    private_records: Sequence[Mapping[str, object]],
    artifact: Mapping[str, object],
    *,
    state: AuthorizedState,
    audit: ExternalAudit,
) -> list[dict[str, object]]:
    if audit.authorization_passes != 1:
        raise ValueError("authorization must pass before cache decoding")
    if inventory.collection_sha256 != state.cache_artifact_collection_sha256:
        raise ValueError("cache bytes differ from authorization")
    validate_frozen_artifact_for_external(artifact)
    rows: list[dict[str, object]] = []
    for metadata in private_records:
        recording_id = str(metadata["recording_id"])
        record = _record_from_authenticated_bytes(inventory.blobs[recording_id])
        label = 1 if metadata["binary_label"] == "affected" else 0
        if (
            record["recording_id"] != recording_id
            or record["group_id"] != metadata["participant_id"]
            or record["source_sha256"] != metadata["video_sha256"]
            or record["label"] != label
        ):
            raise ValueError("authenticated cache differs from private manifest")
        audit.cache_records_loaded += 1
        raw = record["features"]
        mirrored = mirror_dynamic_features(raw)
        remirrored = mirror_dynamic_features(mirrored)
        audit.mirror_transforms += 2
        if not np.array_equal(remirrored, raw):
            raise ValueError("mirror transform is not an exact involution")
        temporal = (record["valid_mask"], record["timestamps"], record["source_frame_indices"])
        original_110d = candidate_feature_vector(LANDMARK_MI_110D, raw, *temporal)
        mirrored_110d = candidate_feature_vector(LANDMARK_MI_110D, mirrored, *temporal)
        audit.feature_extractions += 1
        score = predict_from_frozen_artifact(artifact, original_110d, mirrored_110d)
        audit.artifact_predictions += 1
        rows.append({
            "participant_id": metadata["participant_id"],
            "cohort": metadata["cohort"],
            "label": label,
            "task": metadata["task"],
            "probability": score,
            "slp_scores": metadata.get("slp_scores"),
        })
    return rows


def implementation_fingerprints() -> tuple[dict[str, str], str]:
    root = Path(__file__).resolve().parents[2]
    paths = {
        "external_core": Path(__file__).resolve(),
        "external_runner": root / "scripts" / "run_neuroface_external_v1.py",
        "generalization_features": root / "src" / "preprocessing" / "generalization_110d.py",
        "trajectory_features": root / "src" / "preprocessing" / "trajectory_features.py",
        "mirror_runner": root / "scripts" / "run_mirror_invariant_110d.py",
        "dynamic_loader": root / "src" / "datasets" / "dynamic_landmark.py",
        "frozen_inference": root / "src" / "evaluation" / "outer_release_110d_v1.py",
    }
    components = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for name, path in paths.items()}
    return components, canonical_json_sha256(components)


__all__ = [
    "AUTHORIZATION_SCHEMA_VERSION", "BOOTSTRAP_REPEATS", "BOOTSTRAP_SEED",
    "DEFAULT_REPORT_RELATIVE", "ExternalAudit", "AuthorizedState",
    "ParticipantAggregate", "aggregate_participant_scores",
    "build_expected_authorization", "build_external_report",
    "canonical_json_sha256", "implementation_fingerprints", "metric_report",
    "protocol", "score_authenticated_cache", "validate_external_authorization",
    "validate_external_report",
]
