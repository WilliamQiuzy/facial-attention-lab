"""Contracts for Scale-Robust Eye Geometry v1."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
from scripts.run_mirror_invariant_110d import mirror_dynamic_features  # noqa: E402
from src.evaluation.scale_robust_geometry_v1 import (  # noqa: E402
    build_action_coverage_summary,
    build_scale_robust_report,
    select_low_scale_groups,
    select_scale_robust_candidate,
)
from scripts.run_scale_robust_geometry_v1 import _write_no_overwrite  # noqa: E402
from src.preprocessing.scale_robust_geometry_v1 import (  # noqa: E402
    CANDIDATE_ORDER,
    ALL_LANDMARK_MEDIAN3_110D,
    EYE_MEDIAN3_110D,
    RAW_110D,
    scale_robust_feature_vector,
    scale_robust_recording,
)


def _recording() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = np.zeros((4, 32, 95), dtype=np.float32)
    mask = np.ones((4, 32), dtype=bool)
    indices = np.stack([
        np.arange(start, start + 32, dtype=np.int64)
        for start in (0, 64, 128, 192)
    ])
    timestamps = indices.astype(np.float64) / 30.0
    time = np.arange(32, dtype=np.float32)
    for window in range(4):
        for channel in range(72, 95):
            features[window, :, channel] = (
                0.01 * channel + 0.001 * window + 0.0001 * time
            )
    return features, mask, timestamps, indices


def test_candidate_registry_is_exact_small_and_ordered(c: Check):
    c.eq(CANDIDATE_ORDER, (
        RAW_110D, EYE_MEDIAN3_110D, ALL_LANDMARK_MEDIAN3_110D,
    ))
    c.eq(CANDIDATE_ORDER, (
        "raw_110d", "eye_median3_110d", "all_landmark_median3_110d",
    ))


def test_median3_suppresses_eye_spike_without_changing_blendshapes(c: Check):
    features, mask, _timestamps, _indices = _recording()
    features[0, 10, 72] = 100.0
    features[0, 10, 86] = 200.0
    features[0, 10, 5] = 300.0
    eye = scale_robust_recording(EYE_MEDIAN3_110D, features, mask)
    all_landmark = scale_robust_recording(
        ALL_LANDMARK_MEDIAN3_110D, features, mask
    )
    c.true(eye[0, 10, 72] < 1.0, "eye spike is suppressed")
    c.eq(float(eye[0, 10, 86]), 200.0, "eye-only candidate leaves mouth raw")
    c.true(all_landmark[0, 10, 86] < 2.0, "all-landmark candidate filters mouth")
    c.eq(float(eye[0, 10, 5]), 300.0, "blendshape stream is never filtered")
    c.true(np.array_equal(features, scale_robust_recording(RAW_110D, features, mask)))


def test_filter_never_crosses_window_or_detector_gap(c: Check):
    features, mask, _timestamps, _indices = _recording()
    features[0, 9, 72] = 0.0
    features[0, 10, 72] = 50.0
    features[0, 11, 72] = 100.0
    mask[0, 9] = False
    features[0, 9] = 0.0
    filtered = scale_robust_recording(EYE_MEDIAN3_110D, features, mask)
    c.eq(float(filtered[0, 10, 72]), 50.0, "gap-adjacent centre remains unchanged")
    c.true(bool(np.all(filtered[~mask] == 0)), "invalid rows stay canonical zero")
    c.eq(float(filtered[1, 0, 72]), float(features[1, 0, 72]))
    c.eq(float(filtered[0, -1, 72]), float(features[0, -1, 72]))


def test_filter_commutes_with_mirror_and_vectors_remain_110d(c: Check):
    features, mask, timestamps, indices = _recording()
    features[2, 15, 72] += 10.0
    for candidate in CANDIDATE_ORDER:
        vector = scale_robust_feature_vector(
            candidate, features, mask, timestamps, indices
        )
        c.eq(vector.shape, (110,))
        c.true(bool(np.isfinite(vector).all()))
    first = mirror_dynamic_features(
        scale_robust_recording(ALL_LANDMARK_MEDIAN3_110D, features, mask)
    )
    second = scale_robust_recording(
        ALL_LANDMARK_MEDIAN3_110D,
        mirror_dynamic_features(features),
        mask,
    )
    c.true(np.array_equal(first, second), "filter must commute with mirror")


def test_low_scale_groups_are_label_stratified_deterministic_and_disjoint(c: Check):
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    groups = np.asarray(["d", "b", "a", "c", "h", "f", "e", "g"], dtype=object)
    scales = np.asarray([0.4, 0.2, 0.1, 0.3, 0.8, 0.6, 0.5, 0.7])
    selected = select_low_scale_groups(labels, groups, scales)
    c.eq(selected.tolist(), [False, True, True, False, False, True, True, False])
    c.eq(set(labels[selected].tolist()), {0, 1})
    inconsistent_groups = groups.copy()
    inconsistent_groups[4] = "d"
    c.raises(
        lambda: select_low_scale_groups(labels, inconsistent_groups, scales),
        ValueError,
        "one group cannot cross labels",
    )


def test_selection_retains_ties_and_requires_low_scale_improvement(c: Check):
    row = {
        "overall_auroc": 0.95,
        "overall_balanced_accuracy": 0.90,
        "overall_brier": 0.12,
        "low_scale_auroc": 0.90,
        "low_scale_balanced_accuracy": 0.85,
        "low_scale_brier": 0.14,
    }
    tied = {name: dict(row) for name in CANDIDATE_ORDER}
    c.eq(select_scale_robust_candidate(tied)["selected"], RAW_110D)
    improved = {name: dict(row) for name in CANDIDATE_ORDER}
    improved[EYE_MEDIAN3_110D]["low_scale_brier"] = 0.13
    improved[ALL_LANDMARK_MEDIAN3_110D]["overall_auroc"] = 0.94
    decision = select_scale_robust_candidate(improved)
    c.eq(decision["selected"], EYE_MEDIAN3_110D)
    c.eq(decision["promoted"], True)
    c.eq(decision["eligibility"][ALL_LANDMARK_MEDIAN3_110D], False)


def test_action_coverage_summary_is_aggregate_and_names_limitation(c: Check):
    summary = build_action_coverage_summary(
        source_frame_counts=np.asarray([3000, 6000]),
        fps=np.asarray([30.0, 60.0]),
        window_starts=np.asarray([[0, 1000, 2000, 2968], [10, 2000, 4000, 5968]]),
    )
    c.eq(summary["windows_per_video"], 4)
    c.eq(summary["frames_per_window"], 32)
    c.eq(summary["frames_sampled_per_video"], 128)
    c.eq(summary["action_segments_defined"], False)
    c.eq(summary["eight_action_coverage_defined"], False)
    c.true(summary["sampled_frame_fraction"]["maximum"] < 0.05)
    encoded = str(summary).lower()
    c.true("recording_id" not in encoded and "group_id" not in encoded)


def test_report_is_identifier_free_one_class_and_protected_sealed(c: Check):
    row = {
        "overall_auroc": 0.95,
        "overall_balanced_accuracy": 0.90,
        "overall_brier": 0.12,
        "low_scale_auroc": 0.90,
        "low_scale_balanced_accuracy": 0.85,
        "low_scale_brier": 0.14,
    }
    metrics = {name: dict(row) for name in CANDIDATE_ORDER}
    decision = select_scale_robust_candidate(metrics)
    positive = {
        "records": 47,
        "positive_calls": 45,
        "positive_call_rate": 45 / 47,
        "accuracy_defined": False,
    }
    coverage = build_action_coverage_summary(
        source_frame_counts=np.asarray([3000, 6000]),
        fps=np.asarray([30.0, 60.0]),
        window_starts=np.asarray([[0, 1000, 2000, 2968], [10, 2000, 4000, 5968]]),
    )
    provenance = {name: character * 64 for name, character in zip((
        "palsynet_source_collection_sha256",
        "palsynet_reviewed_manifest_sha256",
        "palsynet_review_ledger_sha256",
        "palsynet_split_registry_sha256",
        "mayo_cache_manifest_sha256",
        "implementation_sha256",
    ), "abcdef")}
    report = build_scale_robust_report(
        metrics,
        decision,
        positive,
        positive,
        coverage,
        development_recordings=39,
        development_groups=38,
        low_scale_groups=20,
        mayo_records=47,
        provenance=provenance,
        protected_cache_records_loaded=0,
    )
    encoded = json.dumps(report, sort_keys=True).lower()
    c.eq(report["decision"]["mayo_used_for_model_selection"], False)
    c.eq(report["audit"]["protected_predictions"], 0)
    for forbidden in ("recording_id", "group_id", "source_sha256", ".mov", "/users/"):
        c.true(forbidden not in encoded)
    c.raises(
        lambda: build_scale_robust_report(
            metrics,
            decision,
            positive,
            positive,
            coverage,
            development_recordings=39,
            development_groups=38,
            low_scale_groups=20,
            mayo_records=47,
            provenance=provenance,
            protected_cache_records_loaded=1,
        ),
        ValueError,
    )


def test_report_writer_is_private_and_no_overwrite(c: Check):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "report.json"
        _write_no_overwrite(path, {"ok": True})
        c.eq(os.stat(path).st_mode & 0o777, 0o600)
        c.raises(lambda: _write_no_overwrite(path, {"ok": False}), FileExistsError)


if __name__ == "__main__":
    run_all("test_scale_robust_geometry_v1", dict(globals()))
