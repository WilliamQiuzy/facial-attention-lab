"""Aggregate-only Mayo diagnostics and PalsyNet-locked mirror aggregation.

Mayo is an assumed-positive challenge cohort.  This module deliberately does
not expose record identifiers, fit models, select thresholds, or define
accuracy.  Aggregator selection is performed from caller-supplied PalsyNet
development metrics only.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from ..datasets.dynamic_landmark import DYNAMIC_FEATURE_NAMES
from ..preprocessing.generalization_110d import (
    LANDMARK_MI_110D,
    candidate_feature_names,
)
from ..preprocessing.trajectory_features import (
    LANDMARK_BILATERAL_PAIRS,
    SUMMARY_STAT_NAMES,
)


AGGREGATOR_ORDER = (
    "mirror_mean",
    "mirror_logit_mean",
    "mirror_conservative_max",
)
REGION_ORDER = ("eye", "brow", "mouth")
_EPSILON = np.finfo(np.float64).eps
_METRIC_TOLERANCE = 1e-12


def _probability_pair(
    original: Sequence[float] | np.ndarray,
    mirrored: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(original, dtype=np.float64)
    second = np.asarray(mirrored, dtype=np.float64)
    if first.ndim != 1 or first.size == 0 or second.shape != first.shape:
        raise ValueError("mirror probabilities must be aligned nonempty vectors")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("mirror probabilities must be finite")
    if np.any((first < 0.0) | (first > 1.0)) or np.any(
        (second < 0.0) | (second > 1.0)
    ):
        raise ValueError("mirror probabilities must lie within [0, 1]")
    return first, second


def aggregate_mirror_probabilities(
    original: Sequence[float] | np.ndarray,
    mirrored: Sequence[float] | np.ndarray,
    method: str,
) -> np.ndarray:
    """Combine paired mirror-view probabilities by one frozen rule."""
    if not isinstance(method, str) or method not in AGGREGATOR_ORDER:
        raise ValueError(f"unknown mirror aggregation method {method!r}")
    first, second = _probability_pair(original, mirrored)
    if method == "mirror_mean":
        result = 0.5 * (first + second)
    elif method == "mirror_logit_mean":
        clipped_first = np.clip(first, _EPSILON, 1.0 - _EPSILON)
        clipped_second = np.clip(second, _EPSILON, 1.0 - _EPSILON)
        logits = 0.5 * (
            np.log(clipped_first / (1.0 - clipped_first))
            + np.log(clipped_second / (1.0 - clipped_second))
        )
        result = 1.0 / (1.0 + np.exp(-np.clip(logits, -700.0, 700.0)))
    else:
        result = np.maximum(first, second)
    if not np.isfinite(result).all() or np.any((result < 0.0) | (result > 1.0)):
        raise RuntimeError("mirror aggregation produced invalid probabilities")
    return result.astype(np.float64, copy=False)


def _frozen_region_by_feature() -> dict[str, str]:
    mapping: dict[str, str] = {}
    channel_regions = (
        ("eye", range(72, 82)),
        ("brow", range(82, 86)),
        ("mouth", range(86, 95)),
    )
    for region, channels in channel_regions:
        for channel in channels:
            for statistic in SUMMARY_STAT_NAMES:
                mapping[f"{DYNAMIC_FEATURE_NAMES[channel]}__{statistic}"] = region
    for pair_index, (pair_name, _first, _second) in enumerate(
        LANDMARK_BILATERAL_PAIRS
    ):
        region = "eye" if pair_index < 3 else "brow" if pair_index == 3 else "mouth"
        for statistic in ("correlation", "amplitude_ratio", "lag_seconds"):
            mapping[f"{pair_name}__{statistic}"] = region
    if len(mapping) != 110:
        raise AssertionError("frozen 110D region map drifted")
    return mapping


def feature_region_assignments(feature_names: Sequence[str]) -> tuple[str, ...]:
    """Assign every exact frozen 110D feature to eye, brow, or mouth."""
    try:
        names = tuple(feature_names)
    except TypeError as exc:
        raise ValueError("feature_names must be a sequence") from exc
    expected = candidate_feature_names(LANDMARK_MI_110D)
    if names != expected:
        raise ValueError("feature names differ from the exact frozen 110D schema")
    mapping = _frozen_region_by_feature()
    return tuple(mapping[name] for name in names)


def _validated_metric_row(values: Mapping[str, object], name: str) -> dict[str, float]:
    if not isinstance(values, Mapping) or set(values) != {
        "auroc", "balanced_accuracy", "brier"
    }:
        raise ValueError(f"{name} metrics must have the frozen three fields")
    output: dict[str, float] = {}
    for metric, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name}.{metric} must be numeric")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{name}.{metric} must lie within [0, 1]")
        output[str(metric)] = number
    return output


def select_palsynet_aggregator(
    metrics_by_method: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Apply the frozen PalsyNet-only non-inferiority promotion rule."""
    if not isinstance(metrics_by_method, Mapping) or tuple(metrics_by_method) != AGGREGATOR_ORDER:
        raise ValueError("aggregator metrics must follow the frozen candidate order")
    metrics = {
        name: _validated_metric_row(metrics_by_method[name], name)
        for name in AGGREGATOR_ORDER
    }
    baseline = metrics["mirror_mean"]
    eligibility = {"mirror_mean": True}
    for name in AGGREGATOR_ORDER[1:]:
        candidate = metrics[name]
        noninferior = (
            candidate["auroc"] + _METRIC_TOLERANCE >= baseline["auroc"]
            and candidate["balanced_accuracy"] + _METRIC_TOLERANCE
            >= baseline["balanced_accuracy"]
            and candidate["brier"] <= baseline["brier"] + 0.01 + _METRIC_TOLERANCE
        )
        improved = (
            candidate["balanced_accuracy"]
            > baseline["balanced_accuracy"] + _METRIC_TOLERANCE
            or candidate["brier"] < baseline["brier"] - _METRIC_TOLERANCE
        )
        eligibility[name] = bool(noninferior and improved)
    eligible_candidates = [
        name for name in AGGREGATOR_ORDER[1:] if eligibility[name]
    ]
    if eligible_candidates:
        selected = min(
            eligible_candidates,
            key=lambda name: (
                -metrics[name]["balanced_accuracy"],
                metrics[name]["brier"],
                -metrics[name]["auroc"],
                AGGREGATOR_ORDER.index(name),
            ),
        )
    else:
        selected = "mirror_mean"
    return {
        "selected": selected,
        "promoted": selected != "mirror_mean",
        "baseline": "mirror_mean",
        "eligibility": eligibility,
        "selection_source": "identity_reviewed_palsynet_development_oof_only",
    }


def _finite_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite nonempty vector")
    return result


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def _standardized_shift(failed: np.ndarray, called: np.ndarray, all_values: np.ndarray) -> float:
    scale = float(np.std(all_values, ddof=1)) if all_values.size > 1 else 0.0
    if not math.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        return 0.0
    return float((np.mean(failed) - np.mean(called)) / scale)


def build_failure_summary(
    probabilities: Sequence[float] | np.ndarray,
    coverages: Sequence[float] | np.ndarray,
    nuisance: Sequence[Sequence[float]] | np.ndarray,
    nuisance_names: Sequence[str],
    region_logit_contributions: Sequence[Sequence[float]] | np.ndarray,
) -> dict[str, object]:
    """Summarize Mayo misses without emitting record-level information."""
    scores = _finite_vector(probabilities, "probabilities")
    coverage = _finite_vector(coverages, "coverages")
    nuisance_matrix = np.asarray(nuisance, dtype=np.float64)
    contributions = np.asarray(region_logit_contributions, dtype=np.float64)
    names = tuple(nuisance_names)
    n = scores.size
    if coverage.shape != (n,) or nuisance_matrix.shape != (n, len(names)):
        raise ValueError("Mayo diagnostic arrays must align by record")
    if contributions.shape != (n, len(REGION_ORDER)):
        raise ValueError("region contributions must align with eye/brow/mouth")
    if (
        not names or len(set(names)) != len(names)
        or any(not isinstance(name, str) or not name for name in names)
        or not np.isfinite(nuisance_matrix).all()
        or not np.isfinite(contributions).all()
        or np.any((scores < 0.0) | (scores > 1.0))
        or np.any((coverage < 0.0) | (coverage > 1.0))
    ):
        raise ValueError("Mayo diagnostic values violate the aggregate contract")
    failed = scores < 0.5
    called = ~failed
    if not failed.any() or not called.any():
        raise ValueError("failure analysis requires both below- and above-threshold records")

    nuisance_shift = {}
    for column, name in enumerate(names):
        values = nuisance_matrix[:, column]
        nuisance_shift[name] = {
            "below_threshold_mean": float(np.mean(values[failed])),
            "positive_call_mean": float(np.mean(values[called])),
            "standardized_mean_shift": _standardized_shift(
                values[failed], values[called], values
            ),
        }
    region_shift = {}
    for column, region in enumerate(REGION_ORDER):
        values = contributions[:, column]
        region_shift[region] = {
            "below_threshold_mean": float(np.mean(values[failed])),
            "positive_call_mean": float(np.mean(values[called])),
            "mean_shift": float(np.mean(values[failed]) - np.mean(values[called])),
        }
    return {
        "schema_version": "mayo_failure_analysis_v1_aggregate",
        "claim_scope": "mayo_assumed_positive_aggregate_diagnostic",
        "threshold": 0.5,
        "counts": {
            "scored": int(n),
            "below_threshold": int(np.sum(failed)),
            "positive_calls": int(np.sum(called)),
        },
        "below_threshold_confidence": _distribution(scores[failed]),
        "positive_call_confidence": _distribution(scores[called]),
        "below_threshold_coverage": _distribution(coverage[failed]),
        "positive_call_coverage": _distribution(coverage[called]),
        "nuisance_standardized_shift": nuisance_shift,
        "region_logit_contribution_shift": region_shift,
        "accuracy_defined": False,
        "mayo_used_for_model_selection": False,
    }


def build_robust_inference_report(
    palsynet_metrics: Mapping[str, Mapping[str, object]],
    decision: Mapping[str, object],
    failure_analysis: Mapping[str, object],
    mayo_current: Mapping[str, object],
    mayo_selected: Mapping[str, object],
    *,
    development_recordings: int,
    development_groups: int,
    mayo_records: int,
    provenance: Mapping[str, str],
    protected_cache_records_loaded: int,
) -> dict[str, object]:
    """Build the identifier-free public report after PalsyNet-only selection."""
    metrics = {
        name: _validated_metric_row(palsynet_metrics[name], name)
        for name in AGGREGATOR_ORDER
    } if isinstance(palsynet_metrics, Mapping) and tuple(palsynet_metrics) == AGGREGATOR_ORDER else None
    if metrics is None:
        raise ValueError("PalsyNet metrics differ from the frozen aggregator registry")
    expected_decision = select_palsynet_aggregator(metrics)
    if not isinstance(decision, Mapping) or dict(decision) != expected_decision:
        raise ValueError("aggregator decision differs from the PalsyNet-only rule")
    if (
        not isinstance(failure_analysis, Mapping)
        or failure_analysis.get("claim_scope")
        != "mayo_assumed_positive_aggregate_diagnostic"
        or failure_analysis.get("accuracy_defined") is not False
        or failure_analysis.get("mayo_used_for_model_selection") is not False
    ):
        raise ValueError("failure analysis does not preserve the one-class boundary")
    for name, summary in (("current", mayo_current), ("selected", mayo_selected)):
        if (
            not isinstance(summary, Mapping)
            or summary.get("accuracy_defined") is not False
            or int(summary.get("records", -1)) != int(mayo_records)
        ):
            raise ValueError(f"Mayo {name} summary violates the one-class contract")
    required_provenance = {
        "palsynet_source_collection_sha256",
        "palsynet_reviewed_manifest_sha256",
        "palsynet_review_ledger_sha256",
        "palsynet_split_registry_sha256",
        "mayo_cache_manifest_sha256",
        "implementation_sha256",
    }
    if set(provenance) != required_provenance or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in provenance.values()
    ):
        raise ValueError("report provenance must contain six lowercase SHA-256 digests")
    counts = (development_recordings, development_groups, mayo_records)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in counts):
        raise ValueError("report counts must be positive integers")
    if protected_cache_records_loaded != 0:
        raise ValueError("protected cache access invalidates robust-inference reporting")
    report = {
        "schema_version": "mayo_failure_analysis_robust_inference_v1_report",
        "target": "binary_affected_vs_unaffected_not_hb_grade",
        "claim_scope": {
            "selection": "identity_reviewed_palsynet_development_oof_only",
            "mayo": "assumed_positive_confidence_challenge_only",
        },
        "protocol": {
            "threshold": 0.5,
            "aggregator_order": list(AGGREGATOR_ORDER),
            "logistic_c": 0.01,
            "promotion_rule": (
                "no_lower_auroc_or_balanced_accuracy_and_brier_increase_at_most_0.01"
                "_with_balanced_accuracy_or_brier_improvement"
            ),
        },
        "counts": {
            "palsynet_development_recordings": int(development_recordings),
            "palsynet_development_groups": int(development_groups),
            "mayo_scored_contents": int(mayo_records),
        },
        "palsynet_development_metrics": metrics,
        "decision": {
            **expected_decision,
            "mayo_used_for_model_selection": False,
            "threshold_tuned_on_mayo": False,
            "outer_evaluation_authorized": False,
            "clinical_validation": False,
        },
        "mayo_current_mirror_mean": dict(mayo_current),
        "mayo_post_lock_selected": dict(mayo_selected),
        "mayo_failure_analysis": dict(failure_analysis),
        "audit": {
            "protected_cache_records_loaded": 0,
            "protected_feature_extractions": 0,
            "protected_model_fits": 0,
            "protected_predictions": 0,
        },
        "provenance": dict(provenance),
    }
    encoded = repr(report).lower()
    if any(token in encoded for token in (
        "recording_id", "group_id", "source_sha256", ".mov", "/users/"
    )):
        raise ValueError("private record metadata reached the aggregate report")
    return report


__all__ = [
    "AGGREGATOR_ORDER",
    "REGION_ORDER",
    "aggregate_mirror_probabilities",
    "build_failure_summary",
    "build_robust_inference_report",
    "feature_region_assignments",
    "select_palsynet_aggregator",
]
