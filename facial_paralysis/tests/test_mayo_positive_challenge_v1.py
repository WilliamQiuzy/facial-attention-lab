"""Contracts for the local-only Mayo positive-cohort challenge."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
from src.evaluation.mayo_positive_challenge_v1 import (  # noqa: E402
    ChallengeRecord,
    build_aggregate_challenge_report,
    fit_frozen_110d_champion,
    inventory_content_deduplicated_videos,
    positive_cohort_summary,
    predict_mirror_mean,
)
from scripts.build_mayo_positive_challenge_v1 import (  # noqa: E402
    _assert_no_private_locations,
    configure_capture_orientation,
    select_face_anchored_starts,
)


def test_inventory_is_content_deduplicated_and_opaque(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "patient-name-a.mov").write_bytes(b"same-content")
        (root / "nested").mkdir()
        (root / "nested" / "patient-name-b.mov").write_bytes(b"same-content")
        (root / "nested" / "different.mov").write_bytes(b"different-content")
        (root / "ignored.mp4").write_bytes(b"not-in-frozen-extension")
        inventory = inventory_content_deduplicated_videos(root)
    c.eq(inventory.source_files, 3)
    c.eq(inventory.unique_contents, 2)
    c.eq(inventory.exact_duplicate_files, 1)
    c.eq(len(inventory.records), 2)
    for record in inventory.records:
        c.true(record.recording_id.startswith("rec_") and len(record.recording_id) == 68)
        c.true(record.group_id.startswith("grp_") and len(record.group_id) == 68)
        c.true("patient-name" not in record.recording_id)
        c.true(record.path.is_absolute())


def test_frozen_champion_is_mirror_mean_and_group_weighted(c: Check):
    rng = np.random.default_rng(811)
    original = rng.normal(size=(8, 110))
    mirrored = original.copy()
    mirrored[:, 0] *= -1
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    groups = np.asarray(["a", "a", "b", "c", "d", "e", "f", "f"], dtype=object)
    champion = fit_frozen_110d_champion(original, mirrored, labels, groups)
    probabilities = predict_mirror_mean(champion, original, mirrored)
    swapped = predict_mirror_mean(champion, mirrored, original)
    c.eq(probabilities.shape, (8,))
    c.true(bool(np.all((0 <= probabilities) & (probabilities <= 1))))
    c.true(bool(np.array_equal(probabilities, swapped)), "mirror order is invariant")
    c.eq(champion.model.get_params()["C"], 0.01)


def test_face_anchored_windows_are_spread_and_nonoverlapping(c: Check):
    starts = select_face_anchored_starts(
        [16, 48, 120, 220, 320, 480], frame_count=512
    )
    c.eq(len(starts), 4)
    c.true(all(0 <= value <= 480 for value in starts))
    c.true(all(right - left >= 32 for left, right in zip(starts, starts[1:])))
    c.raises(
        lambda: select_face_anchored_starts([16, 20, 24], frame_count=512),
        ValueError,
        "four time-separated face anchors are mandatory",
    )


def test_capture_orientation_metadata_is_enabled_before_decode(c: Check):
    class FakeCapture:
        def __init__(self):
            self.auto = 0.0

        def set(self, prop, value):
            self.auto = float(value)
            return True

        def get(self, prop):
            return self.auto

    capture = FakeCapture()
    configure_capture_orientation(capture)
    c.eq(capture.auto, 1.0)


def test_manifest_protocol_name_does_not_look_like_a_raw_filename(c: Check):
    _assert_no_private_locations({"video_container": "quicktime_mov_container"})
    c.raises(
        lambda: _assert_no_private_locations({"video_container": "example.mov"}),
        ValueError,
        "raw-looking filenames must fail closed",
    )


def test_positive_summary_names_the_one_class_limitation(c: Check):
    probabilities = np.asarray([0.2, 0.51, 0.8, 0.99], dtype=np.float64)
    coverage = np.asarray([0.9, 1.0, 0.95, 1.0], dtype=np.float64)
    summary = positive_cohort_summary(probabilities, coverage)
    c.eq(summary["records"], 4)
    c.eq(summary["positive_calls"], 3)
    c.eq(summary["positive_call_rate"], 0.75)
    c.eq(summary["verified_negative_records"], 0)
    c.eq(summary["accuracy_defined"], False)
    c.true(0 <= summary["positive_call_rate_wilson95"][0] < 0.75)
    c.true(0.75 < summary["positive_call_rate_wilson95"][1] <= 1)


def test_report_is_aggregate_identifier_free_and_not_model_selection(c: Check):
    report = build_aggregate_challenge_report(
        positive_cohort_summary(
            np.asarray([0.8, 0.9]), np.asarray([1.0, 0.95])
        ),
        source_files=3,
        unique_contents=2,
        exact_duplicate_files=1,
        excluded_records=0,
        provenance={
            "palsynet_source_collection_sha256": "a" * 64,
            "palsynet_reviewed_manifest_sha256": "b" * 64,
            "palsynet_review_ledger_sha256": "c" * 64,
            "palsynet_split_registry_sha256": "d" * 64,
            "mayo_cache_manifest_sha256": "e" * 64,
            "implementation_sha256": "f" * 64,
        },
    )
    encoded = str(report).lower()
    c.true("recording_id" not in encoded and "group_id" not in encoded)
    c.true("patient" not in encoded and ".mov" not in encoded)
    c.eq(report["decision"]["mayo_used_for_model_selection"], False)
    c.eq(report["decision"]["accuracy_claimed"], False)
    c.eq(report["audit"]["palsynet_protected_predictions"], 0)


if __name__ == "__main__":
    run_all("test_mayo_positive_challenge_v1", dict(globals()))
