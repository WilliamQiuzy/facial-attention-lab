"""Contracts for the NeuroFace manual-68 versus MediaPipe geometry audit."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
from src.evaluation.neuroface_manual68_audit_v1 import (  # noqa: E402
    build_manual68_audit_report,
)


def _report(detected=None):
    rng = np.random.default_rng(813)
    n = 72
    manual = rng.normal(size=(n, 23))
    mp = 1.5 * manual + rng.normal(scale=0.01, size=(n, 23))
    present = np.ones(n, dtype=bool) if detected is None else np.asarray(detected, dtype=bool)
    mp[~present] = np.nan
    return build_manual68_audit_report(
        manual, mp, present,
        participant_ids=np.repeat([f"private-{i}" for i in range(12)], 6),
        recording_ids=np.repeat([f"video-{i}" for i in range(24)], 3),
        cohorts=np.tile(["als", "healthy_control", "post_stroke"], 24),
        tasks=np.tile(["NSM_KISS", "NSM_OPEN", "NSM_SPREAD"], 24),
        provenance={name: value * 64 for name, value in {
            "private_manifest_sha256": "a",
            "manual_landmark_collection_sha256": "b",
            "video_collection_sha256": "c",
            "mediapipe_model_sha256": "d",
            "implementation_sha256": "e",
            "dependency_lock_sha256": "f",
        }.items()},
        runtime={"host": "nebius-h200", "seconds": 1.0},
    )


def test_high_agreement_passes_locked_measurement_gate(c: Check):
    report = _report()
    c.eq(report["counts"], {
        "participants": 12, "recordings": 24,
        "annotated_frames": 72, "detected_frames": 72,
    })
    c.eq(report["decision"]["measurement_gate_passed"], True)
    c.true(report["mirror_invariant_summary"]["median_absolute_spearman"] > 0.99)
    c.true(report["mirror_invariant_summary"]["median_absolute_within_recording_pearson"] > 0.99)
    encoded = str(report).lower()
    c.true("private-" not in encoded and "video-" not in encoded)


def test_missed_frames_are_counted_and_never_treated_as_neutral(c: Check):
    detected = np.ones(72, dtype=bool)
    detected[:8] = False
    report = _report(detected)
    c.eq(report["counts"]["detected_frames"], 64)
    c.true(report["detection"]["overall_rate"] < 0.95)
    c.eq(report["decision"]["measurement_gate_passed"], False)


def test_malformed_detection_contract_fails_closed(c: Check):
    rng = np.random.default_rng(1)
    manual = rng.normal(size=(4, 23))
    mp = manual.copy()
    detected = np.asarray([True, True, False, False])
    c.raises(lambda: build_manual68_audit_report(
        manual, mp, detected,
        participant_ids=["a"] * 4, recording_ids=["b"] * 4,
        cohorts=["als"] * 4, tasks=["NSM_KISS"] * 4,
        provenance={name: "a" * 64 for name in (
            "private_manifest_sha256", "manual_landmark_collection_sha256",
            "video_collection_sha256", "mediapipe_model_sha256",
            "implementation_sha256", "dependency_lock_sha256",
        )}, runtime={"host": "test"},
    ), ValueError, "missed frames must carry NaN rather than fake neutral geometry")


if __name__ == "__main__":
    run_all("test_neuroface_manual68_audit_v1", dict(globals()))
