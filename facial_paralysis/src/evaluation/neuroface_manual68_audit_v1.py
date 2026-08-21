"""Aggregate manual-68 versus MediaPipe semantic geometry audit for NeuroFace."""
from __future__ import annotations

import json
from collections import Counter
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from src.preprocessing.semantic_landmarks import SEMANTIC23_FEATURE_NAMES


SCHEMA_VERSION = "neuroface_manual68_mediapipe_audit_v1"
REGIONS = {
    "eye": tuple(range(0, 10)),
    "brow": tuple(range(10, 14)),
    "mouth": tuple(range(14, 23)),
}
MIRROR_INVARIANT = (2, 6, 9, 12, 16, 20, 21, 22)


def _correlation(left: np.ndarray, right: np.ndarray, *, rank: bool) -> float | None:
    if left.size < 3 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    value = spearmanr(left, right).statistic if rank else np.corrcoef(left, right)[0, 1]
    return float(value) if np.isfinite(value) else None


def _within_recording_correlation(
    manual: np.ndarray,
    mediapipe: np.ndarray,
    recordings: np.ndarray,
) -> float | None:
    manual_residuals: list[np.ndarray] = []
    mediapipe_residuals: list[np.ndarray] = []
    for recording in sorted(set(recordings.tolist())):
        rows = recordings == recording
        left, right = manual[rows], mediapipe[rows]
        if left.size < 3 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
            continue
        manual_residuals.append((left - left.mean()) / left.std())
        mediapipe_residuals.append((right - right.mean()) / right.std())
    if not manual_residuals:
        return None
    return _correlation(
        np.concatenate(manual_residuals), np.concatenate(mediapipe_residuals), rank=False
    )


def _median(values: Sequence[float | None]) -> float | None:
    observed = [float(value) for value in values if value is not None]
    return float(np.median(observed)) if observed else None


def build_manual68_audit_report(
    manual: np.ndarray,
    mediapipe: np.ndarray,
    detected: np.ndarray,
    *,
    participant_ids: Sequence[object] | np.ndarray,
    recording_ids: Sequence[object] | np.ndarray,
    cohorts: Sequence[object] | np.ndarray,
    tasks: Sequence[object] | np.ndarray,
    provenance: Mapping[str, str],
    runtime: Mapping[str, object],
) -> dict[str, object]:
    manual_array = np.asarray(manual, dtype=np.float64)
    mp_array = np.asarray(mediapipe, dtype=np.float64)
    present = np.asarray(detected, dtype=bool)
    n = manual_array.shape[0]
    participants = np.asarray(participant_ids, dtype=object)
    recordings = np.asarray(recording_ids, dtype=object)
    cohort_array = np.asarray(cohorts, dtype=object)
    task_array = np.asarray(tasks, dtype=object)
    if (
        manual_array.shape != (n, 23) or mp_array.shape != (n, 23)
        or present.shape != (n,) or participants.shape != (n,)
        or recordings.shape != (n,) or cohort_array.shape != (n,)
        or task_array.shape != (n,) or n == 0
    ):
        raise ValueError("manual68 audit arrays must align by annotated frame")
    if not np.isfinite(manual_array).all():
        raise ValueError("manual semantic geometry must be finite")
    if not np.isfinite(mp_array[present]).all() or np.isfinite(mp_array[~present]).any():
        raise ValueError("MediaPipe geometry must be finite exactly on detected frames")
    required_provenance = {
        "private_manifest_sha256", "manual_landmark_collection_sha256",
        "video_collection_sha256", "mediapipe_model_sha256",
        "implementation_sha256", "dependency_lock_sha256",
    }
    if set(provenance) != required_provenance or any(
        not isinstance(value, str) or len(value) != 64 for value in provenance.values()
    ):
        raise ValueError("manual68 audit provenance fields are incomplete")
    detected_manual = manual_array[present]
    detected_mp = mp_array[present]
    detected_recordings = recordings[present]
    feature_metrics: dict[str, dict[str, object]] = {}
    for index, name in enumerate(SEMANTIC23_FEATURE_NAMES):
        rho = _correlation(detected_manual[:, index], detected_mp[:, index], rank=True)
        feature_metrics[name] = {
            "spearman": rho,
            "absolute_spearman": None if rho is None else abs(rho),
            "within_recording_pearson": _within_recording_correlation(
                detected_manual[:, index], detected_mp[:, index], detected_recordings
            ),
            "detected_frames": int(present.sum()),
        }
    regions = {
        name: {
            "median_absolute_spearman": _median([
                feature_metrics[SEMANTIC23_FEATURE_NAMES[index]]["absolute_spearman"]
                for index in indices
            ]),
            "median_absolute_within_recording_pearson": _median([
                None if feature_metrics[SEMANTIC23_FEATURE_NAMES[index]][
                    "within_recording_pearson"
                ] is None else abs(float(feature_metrics[SEMANTIC23_FEATURE_NAMES[index]][
                    "within_recording_pearson"
                ]))
                for index in indices
            ]),
        }
        for name, indices in REGIONS.items()
    }
    invariant_spearman = _median([
        feature_metrics[SEMANTIC23_FEATURE_NAMES[index]]["absolute_spearman"]
        for index in MIRROR_INVARIANT
    ])
    invariant_within = _median([
        None if feature_metrics[SEMANTIC23_FEATURE_NAMES[index]][
            "within_recording_pearson"
        ] is None else abs(float(feature_metrics[SEMANTIC23_FEATURE_NAMES[index]][
            "within_recording_pearson"
        ]))
        for index in MIRROR_INVARIANT
    ])
    detection_rate = float(present.mean())
    gate = (
        detection_rate >= 0.95
        and invariant_spearman is not None and invariant_spearman >= 0.70
        and invariant_within is not None and invariant_within >= 0.70
    )
    def detection_breakdown(values: np.ndarray) -> dict[str, dict[str, object]]:
        output: dict[str, dict[str, object]] = {}
        for value in sorted(set(values.tolist())):
            rows = values == value
            output[str(value)] = {
                "annotated_frames": int(rows.sum()),
                "detected_frames": int(present[rows].sum()),
                "detection_rate": float(present[rows].mean()),
            }
        return output
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": "measurement_audit_not_external_model_validation",
        "dataset": "Toronto_NeuroFace_v1",
        "comparison": "manual_68_point_to_mediapipe_semantic23_measure_compatibility",
        "numeric_boundary": (
            "cross_topology_rank_and_within_recording_motion_agreement_not_raw_coordinate_"
            "interchangeability"
        ),
        "counts": {
            "participants": len(set(participants.tolist())),
            "recordings": len(set(recordings.tolist())),
            "annotated_frames": int(n),
            "detected_frames": int(present.sum()),
        },
        "detection": {
            "overall_rate": detection_rate,
            "by_cohort": detection_breakdown(cohort_array),
            "by_task": detection_breakdown(task_array),
        },
        "feature_metrics": feature_metrics,
        "region_summary": regions,
        "mirror_invariant_summary": {
            "feature_count": len(MIRROR_INVARIANT),
            "median_absolute_spearman": invariant_spearman,
            "median_absolute_within_recording_pearson": invariant_within,
        },
        "decision": {
            "measurement_gate_passed": bool(gate),
            "thresholds": {
                "detection_rate_min": 0.95,
                "median_absolute_spearman_min": 0.70,
                "median_absolute_within_recording_pearson_min": 0.70,
            },
            "model_retraining_authorized": False,
            "clinical_validation": False,
        },
        "runtime": dict(runtime),
        "provenance": dict(provenance),
    }
    encoded = json.dumps(report, sort_keys=True, allow_nan=False).lower()
    if any(token in encoded for token in (
        "grp_", "rec_", "participant_id", "recording_id", "/users/", "/home/",
        ".avi", ".txt",
    )):
        raise ValueError("manual68 aggregate report leaks private identifiers or paths")
    return report


__all__ = ["MIRROR_INVARIANT", "REGIONS", "build_manual68_audit_report"]
