"""Selection and aggregate sampling audits for scale-robust geometry v1."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from ..preprocessing.scale_robust_geometry_v1 import (
    CANDIDATE_ORDER,
    RAW_110D,
)


METRIC_FIELDS = (
    "overall_auroc",
    "overall_balanced_accuracy",
    "overall_brier",
    "low_scale_auroc",
    "low_scale_balanced_accuracy",
    "low_scale_brier",
)
_TOLERANCE = 1e-12


def select_low_scale_groups(
    labels: Sequence[int] | np.ndarray,
    group_ids: Sequence[object] | np.ndarray,
    face_scales: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Select the lowest-scale half of reviewed groups within each class."""
    label_array = np.asarray(labels)
    groups = np.asarray(group_ids, dtype=object)
    scales = np.asarray(face_scales, dtype=np.float64)
    if (
        label_array.ndim != 1
        or groups.shape != label_array.shape
        or scales.shape != label_array.shape
        or label_array.size == 0
        or label_array.dtype.kind not in {"i", "u"}
        or set(label_array.tolist()) != {0, 1}
        or not np.isfinite(scales).all()
        or np.any(scales <= 0)
    ):
        raise ValueError("low-scale selection requires aligned binary finite inputs")
    group_rows: dict[str, list[int]] = {}
    for index, group in enumerate(groups.tolist()):
        if not isinstance(group, str) or not group:
            raise ValueError("low-scale group IDs must be nonempty strings")
        group_rows.setdefault(group, []).append(index)
    group_label: dict[str, int] = {}
    group_scale: dict[str, float] = {}
    for group, rows in group_rows.items():
        observed = set(label_array[rows].tolist())
        if len(observed) != 1:
            raise ValueError("one low-scale group cannot cross labels")
        group_label[group] = int(next(iter(observed)))
        group_scale[group] = float(np.mean(scales[rows]))
    selected_groups: set[str] = set()
    for label in (0, 1):
        candidates = sorted(
            (group for group in group_rows if group_label[group] == label),
            key=lambda group: (group_scale[group], group),
        )
        if len(candidates) < 2:
            raise ValueError("low-scale stress needs at least two groups per class")
        count = int(math.ceil(len(candidates) / 2.0))
        selected_groups.update(candidates[:count])
    selected = np.asarray(
        [str(group) in selected_groups for group in groups.tolist()], dtype=bool
    )
    if set(label_array[selected].tolist()) != {0, 1}:
        raise AssertionError("label-stratified low-scale selection lost a class")
    return selected


def _validated_metrics(
    metrics_by_candidate: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, float]]:
    if (
        not isinstance(metrics_by_candidate, Mapping)
        or tuple(metrics_by_candidate) != CANDIDATE_ORDER
    ):
        raise ValueError("metrics must follow the frozen scale-robust registry")
    output: dict[str, dict[str, float]] = {}
    for candidate in CANDIDATE_ORDER:
        row = metrics_by_candidate[candidate]
        if not isinstance(row, Mapping) or tuple(row) != METRIC_FIELDS:
            raise ValueError("scale-robust metrics differ from the frozen fields")
        converted: dict[str, float] = {}
        for name in METRIC_FIELDS:
            value = row[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("scale-robust metrics must be numeric")
            number = float(value)
            if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                raise ValueError("scale-robust metrics must lie within [0, 1]")
            converted[name] = number
        output[candidate] = converted
    return output


def select_scale_robust_candidate(
    metrics_by_candidate: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Apply the PalsyNet-only overall and low-scale promotion gate."""
    metrics = _validated_metrics(metrics_by_candidate)
    baseline = metrics[RAW_110D]
    eligibility = {RAW_110D: True}
    for candidate in CANDIDATE_ORDER[1:]:
        row = metrics[candidate]
        noninferior = (
            row["overall_auroc"] + _TOLERANCE >= baseline["overall_auroc"]
            and row["overall_balanced_accuracy"] + _TOLERANCE
            >= baseline["overall_balanced_accuracy"]
            and row["overall_brier"] <= baseline["overall_brier"] + 0.01 + _TOLERANCE
            and row["low_scale_auroc"] + _TOLERANCE
            >= baseline["low_scale_auroc"]
        )
        low_scale_improvement = (
            row["low_scale_balanced_accuracy"]
            > baseline["low_scale_balanced_accuracy"] + _TOLERANCE
            or row["low_scale_brier"] < baseline["low_scale_brier"] - _TOLERANCE
        )
        eligibility[candidate] = bool(noninferior and low_scale_improvement)
    eligible = [
        candidate for candidate in CANDIDATE_ORDER[1:]
        if eligibility[candidate]
    ]
    if eligible:
        selected = min(
            eligible,
            key=lambda candidate: (
                -metrics[candidate]["overall_balanced_accuracy"],
                -metrics[candidate]["overall_auroc"],
                metrics[candidate]["low_scale_brier"],
                CANDIDATE_ORDER.index(candidate),
            ),
        )
    else:
        selected = RAW_110D
    return {
        "selected": selected,
        "promoted": selected != RAW_110D,
        "baseline": RAW_110D,
        "eligibility": eligibility,
        "selection_source": "identity_reviewed_palsynet_development_oof_only",
    }


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
    }


def build_action_coverage_summary(
    *,
    source_frame_counts: Sequence[int] | np.ndarray,
    fps: Sequence[float] | np.ndarray,
    window_starts: Sequence[Sequence[int]] | np.ndarray,
) -> dict[str, object]:
    """Quantify frame use without pretending four windows are action segments."""
    counts = np.asarray(source_frame_counts)
    frame_rates = np.asarray(fps, dtype=np.float64)
    starts = np.asarray(window_starts)
    if (
        counts.ndim != 1
        or counts.size == 0
        or counts.dtype.kind not in {"i", "u"}
        or frame_rates.shape != counts.shape
        or starts.shape != (counts.size, 4)
        or starts.dtype.kind not in {"i", "u"}
        or not np.isfinite(frame_rates).all()
        or np.any(frame_rates <= 0)
        or np.any(counts < 128)
        or np.any(starts < 0)
        or np.any(np.diff(starts, axis=1) < 32)
        or np.any(starts[:, -1] + 32 > counts)
    ):
        raise ValueError("four-window coverage inputs violate the frozen contract")
    sampled_fraction = 128.0 / counts.astype(np.float64)
    gap_seconds = np.diff(starts.astype(np.float64), axis=1) / frame_rates[:, None]
    return {
        "videos": int(counts.size),
        "windows_per_video": 4,
        "frames_per_window": 32,
        "frames_sampled_per_video": 128,
        "sampling": "four_time_spread_nonoverlapping_windows_not_action_aligned",
        "sampled_frame_fraction": _summary(sampled_fraction),
        "window_start_gap_seconds": _summary(gap_seconds.reshape(-1)),
        "action_segments_defined": False,
        "eight_action_coverage_defined": False,
        "action_recognition_accuracy_defined": False,
    }


def build_scale_robust_report(
    metrics_by_candidate: Mapping[str, Mapping[str, object]],
    decision: Mapping[str, object],
    mayo_current: Mapping[str, object],
    mayo_selected: Mapping[str, object],
    action_coverage: Mapping[str, object],
    *,
    development_recordings: int,
    development_groups: int,
    low_scale_groups: int,
    mayo_records: int,
    provenance: Mapping[str, str],
    protected_cache_records_loaded: int,
) -> dict[str, object]:
    """Build an identifier-free report after PalsyNet-only candidate lock."""
    metrics = _validated_metrics(metrics_by_candidate)
    expected_decision = select_scale_robust_candidate(metrics)
    if not isinstance(decision, Mapping) or dict(decision) != expected_decision:
        raise ValueError("candidate decision differs from the frozen PalsyNet-only rule")
    for name, summary in (("current", mayo_current), ("selected", mayo_selected)):
        if (
            not isinstance(summary, Mapping)
            or summary.get("accuracy_defined") is not False
            or int(summary.get("records", -1)) != int(mayo_records)
        ):
            raise ValueError(f"Mayo {name} result violates the one-class contract")
    if (
        not isinstance(action_coverage, Mapping)
        or action_coverage.get("action_segments_defined") is not False
        or action_coverage.get("eight_action_coverage_defined") is not False
        or int(action_coverage.get("windows_per_video", -1)) != 4
    ):
        raise ValueError("action coverage must preserve the four-window limitation")
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
    counts = (
        development_recordings,
        development_groups,
        low_scale_groups,
        mayo_records,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in counts
    ):
        raise ValueError("scale-robust report counts must be positive integers")
    if low_scale_groups > development_groups:
        raise ValueError("low-scale groups cannot exceed all development groups")
    if protected_cache_records_loaded != 0:
        raise ValueError("protected cache access invalidates scale-robust reporting")
    report = {
        "schema_version": "scale_robust_eye_geometry_v1_report",
        "target": "binary_affected_vs_unaffected_not_hb_grade",
        "claim_scope": {
            "selection": "identity_reviewed_palsynet_development_oof_only",
            "mayo": "assumed_positive_confidence_challenge_only",
        },
        "protocol": {
            "candidate_order": list(CANDIDATE_ORDER),
            "filter": "window_local_complete_valid_triplet_median3",
            "classifier": "standardized_l2_logistic_c_0_01",
            "threshold": 0.5,
            "low_scale_stress": "lowest_mean_face_scale_half_within_each_label",
        },
        "counts": {
            "palsynet_development_recordings": int(development_recordings),
            "palsynet_development_groups": int(development_groups),
            "palsynet_low_scale_groups": int(low_scale_groups),
            "mayo_scored_contents": int(mayo_records),
        },
        "palsynet_development_metrics": metrics,
        "decision": {
            **expected_decision,
            "mayo_used_for_model_selection": False,
            "threshold_tuned_on_mayo": False,
            "outer_evaluation_authorized": False,
            "current_model_replaced": bool(expected_decision["promoted"]),
            "clinical_validation": False,
        },
        "mayo_current_raw_110d": dict(mayo_current),
        "mayo_post_lock_selected": dict(mayo_selected),
        "mayo_four_window_action_coverage": dict(action_coverage),
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
        raise ValueError("private record metadata reached the scale-robust report")
    return report


__all__ = [
    "METRIC_FIELDS",
    "build_action_coverage_summary",
    "build_scale_robust_report",
    "select_low_scale_groups",
    "select_scale_robust_candidate",
]
