"""Source-blind selective evaluation over frozen UCR4 expert probabilities."""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

import numpy as np


CANDIDATE_ORDER = (
    "probability_margin",
    "range_penalized_margin",
    "unanimous_min_margin",
    "dispersion_normalized_margin",
)
COVERAGES = (0.60, 0.70, 0.75, 0.80, 0.90, 1.00)
PRIMARY_COVERAGE = 0.70
EVIDENCE_PROFILES = (
    "free_asymmetry",
    "scripted_multimechanism",
    "cue_aligned_upper",
)
_PROFILE_FIELDS = {
    "schema_version",
    "evidence_profile",
    "anonymous_groups",
    "labels",
    "final_probability",
    "component_probability",
    "decision_threshold",
}


def _immutable_float64(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(
        contiguous.shape
    )


def _probability_vector(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        type(value) is not np.ndarray
        or array.dtype != np.dtype(np.float64)
        or array.ndim != 1
        or array.size == 0
        or not np.isfinite(array).all()
        or np.any((array < 0.0) | (array > 1.0))
    ):
        raise ValueError(f"{name} must be one finite float64 probability vector")
    return array


def confidence_scores(
    final_probability: np.ndarray,
    component_probability: np.ndarray,
    *,
    decision_threshold: np.ndarray,
) -> Mapping[str, np.ndarray]:
    """Compute the four frozen, label-free selective-confidence scores."""
    final = _probability_vector(final_probability, "final probability")
    components = np.asarray(component_probability)
    thresholds = np.asarray(decision_threshold)
    if (
        type(component_probability) is not np.ndarray
        or components.dtype != np.dtype(np.float64)
        or components.ndim != 2
        or components.shape[0] != final.size
        or components.shape[1] < 2
        or not np.isfinite(components).all()
        or np.any((components < 0.0) | (components > 1.0))
        or type(decision_threshold) is not np.ndarray
        or thresholds.dtype != np.dtype(np.float64)
        or thresholds.ndim != 1
        or thresholds.shape != final.shape
        or not np.isfinite(thresholds).all()
        or np.any((thresholds <= 0.0) | (thresholds >= 1.0))
    ):
        raise ValueError("component probabilities violate the frozen schema")
    margin = np.abs(final - thresholds)
    dispersion = np.std(components, axis=1, ddof=0)
    component_range = np.max(components, axis=1) - np.min(components, axis=1)
    final_class = final >= thresholds
    component_class = components >= thresholds[:, None]
    unanimous = np.all(component_class == final_class[:, None], axis=1)
    minimum_margin = np.min(np.abs(components - thresholds[:, None]), axis=1)
    unanimous_score = np.where(unanimous, minimum_margin, -minimum_margin)
    return {
        "probability_margin": _immutable_float64(margin),
        "range_penalized_margin": _immutable_float64(
            margin - 0.5 * component_range
        ),
        "unanimous_min_margin": _immutable_float64(unanimous_score),
        "dispersion_normalized_margin": _immutable_float64(
            margin / (1.0 + 4.0 * dispersion)
        ),
    }


def _validate_profile(document: object) -> tuple[
    str, tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    if not isinstance(document, Mapping) or set(document) != _PROFILE_FIELDS:
        raise ValueError("private selective profile has an open or incomplete schema")
    if document["schema_version"] != "selective_router_v5_private_profile":
        raise ValueError("private selective profile schema version differs")
    profile = document["evidence_profile"]
    if profile not in EVIDENCE_PROFILES:
        raise ValueError("selective evidence profile is not registered")
    groups = document["anonymous_groups"]
    if (
        type(groups) is not tuple
        or not groups
        or any(type(value) is not str or not value for value in groups)
        or len(set(groups)) != len(groups)
    ):
        raise ValueError("anonymous participant groups are not unique canonical strings")
    labels = np.asarray(document["labels"])
    if (
        type(document["labels"]) is not np.ndarray
        or labels.dtype != np.dtype(np.int64)
        or labels.ndim != 1
        or labels.size != len(groups)
        or set(labels.tolist()) != {0, 1}
    ):
        raise ValueError("profile labels must be aligned binary int64 values")
    final = _probability_vector(document["final_probability"], "final probability")
    components = np.asarray(document["component_probability"])
    threshold = np.asarray(document["decision_threshold"])
    confidence_scores(
        final, components, decision_threshold=document["decision_threshold"]
    )
    if final.size != labels.size:
        raise ValueError("profile probability and participant counts differ")
    return profile, groups, labels, final, components, threshold


def _binary_metrics(
    labels: np.ndarray,
    probability: np.ndarray,
    threshold: np.ndarray,
    retained: np.ndarray,
) -> dict[str, object]:
    selected_labels = labels[retained]
    selected_probability = probability[retained]
    prediction = selected_probability >= threshold[retained]
    positive = selected_labels == 1
    negative = selected_labels == 0
    retained_positive = int(np.sum(positive))
    retained_negative = int(np.sum(negative))
    sensitivity = (
        float(np.mean(prediction[positive])) if retained_positive else None
    )
    specificity = (
        float(np.mean(~prediction[negative])) if retained_negative else None
    )
    balanced_accuracy = (
        0.5 * (sensitivity + specificity)
        if sensitivity is not None and specificity is not None else None
    )
    return {
        "accuracy": float(np.mean(prediction == positive)),
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "errors": int(np.sum(prediction != positive)),
        "retained_positive": retained_positive,
        "retained_negative": retained_negative,
    }


def _selection_sha256(indices: np.ndarray) -> str:
    payload = ",".join(str(int(index)) for index in np.sort(indices)) + "\n"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def evaluate_profile(document: object) -> dict[str, object]:
    """Evaluate every frozen confidence ranking for one evidence profile."""
    profile, groups, labels, final, components, threshold = _validate_profile(document)
    scores = confidence_scores(
        final, components, decision_threshold=threshold
    )
    total_positive = int(np.sum(labels == 1))
    total_negative = int(np.sum(labels == 0))
    all_indices = np.arange(labels.size, dtype=np.int64)
    baseline_metrics = _binary_metrics(
        labels, final, threshold, all_indices
    )
    candidates: dict[str, dict[str, object]] = {}
    for name in CANDIDATE_ORDER:
        ranking = np.lexsort((all_indices, -scores[name]))
        points: dict[str, object] = {}
        for requested in COVERAGES:
            retained_count = int(math.ceil(requested * labels.size))
            retained = np.sort(ranking[:retained_count])
            metrics = _binary_metrics(labels, final, threshold, retained)
            points[f"{requested:.2f}"] = {
                "requested_coverage": requested,
                "coverage": retained_count / labels.size,
                "retained": retained_count,
                "abstained": int(labels.size - retained_count),
                "negative_coverage": (
                    metrics["retained_negative"] / total_negative
                ),
                "positive_coverage": (
                    metrics["retained_positive"] / total_positive
                ),
                "selection_sha256": _selection_sha256(retained),
                **metrics,
            }
        candidates[name] = points
    return {
        "schema_version": "selective_router_v5_profile_evaluation",
        "evidence_profile": profile,
        "participants": len(groups),
        "class_counts": {
            "unaffected": total_negative,
            "affected": total_positive,
        },
        "decision_threshold_scope": "per_participant_oof",
        "decision_threshold_range": [
            float(np.min(threshold)), float(np.max(threshold))
        ],
        "baseline": baseline_metrics,
        "candidates": candidates,
    }


def select_candidate(
    profiles: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Apply the frozen all-profile 0.70 development-candidate gate."""
    if not isinstance(profiles, Mapping) or set(profiles) != set(EVIDENCE_PROFILES):
        raise ValueError("candidate decision requires all evidence profiles exactly once")
    eligible: list[tuple[tuple[float, float, float, int], str]] = []
    reasons: dict[str, list[str]] = {}
    for candidate_index, candidate in enumerate(CANDIDATE_ORDER):
        failures = []
        primary_balanced = []
        primary_accuracy = []
        utility = []
        for profile in EVIDENCE_PROFILES:
            report = profiles[profile]
            if (
                not isinstance(report, Mapping)
                or report.get("schema_version")
                != "selective_router_v5_profile_evaluation"
                or report.get("evidence_profile") != profile
            ):
                raise ValueError("profile evaluation is not canonical or aligned")
            candidate_points = report["candidates"][candidate]
            point = candidate_points[f"{PRIMARY_COVERAGE:.2f}"]
            balanced = point["balanced_accuracy"]
            baseline_accuracy = report["baseline"]["accuracy"]
            if point["coverage"] < PRIMARY_COVERAGE:
                failures.append(f"{profile}:coverage")
            if point["retained_negative"] < 5 or point["retained_positive"] < 5:
                failures.append(f"{profile}:class_support")
            if point["accuracy"] < 0.95:
                failures.append(f"{profile}:accuracy")
            if balanced is None or balanced < 0.95:
                failures.append(f"{profile}:balanced_accuracy")
            if point["accuracy"] < baseline_accuracy:
                failures.append(f"{profile}:below_full_cohort_accuracy")
            primary_accuracy.append(float(point["accuracy"]))
            primary_balanced.append(-math.inf if balanced is None else float(balanced))
            for coverage in COVERAGES:
                secondary = candidate_points[f"{coverage:.2f}"]
                utility.append(
                    float(secondary["coverage"]) * float(secondary["accuracy"])
                )
        reasons[candidate] = failures
        if not failures:
            key = (
                min(primary_balanced),
                min(primary_accuracy),
                float(np.mean(utility)),
                -candidate_index,
            )
            eligible.append((key, candidate))
    selected = max(eligible)[1] if eligible else None
    return {
        "schema_version": "selective_router_v5_candidate_decision",
        "primary_coverage": PRIMARY_COVERAGE,
        "passed": selected is not None,
        "selected": selected,
        "failures": reasons,
        "promotion_authorized": False,
        "next_gate": (
            "untouched_participant_disjoint_external_confirmation"
            if selected is not None else "retain_universal_clinical_router_v4"
        ),
    }


__all__ = (
    "CANDIDATE_ORDER",
    "COVERAGES",
    "EVIDENCE_PROFILES",
    "PRIMARY_COVERAGE",
    "confidence_scores",
    "evaluate_profile",
    "select_candidate",
)
