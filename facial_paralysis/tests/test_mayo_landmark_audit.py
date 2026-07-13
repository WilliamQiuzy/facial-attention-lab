"""Tests for streaming Mayo landmark trajectory auditing."""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.audit_mayo_landmark_trajectories as audit_module  # noqa: E402
from scripts.audit_mayo_landmark_trajectories import (  # noqa: E402
    _feature_summary,
    audit_directory,
    audit_landmark_csv,
)
from _testlib import Check, run_all  # noqa: E402
from test_clinical_landmarks import _face  # noqa: E402


def _write_csv(
    path: Path,
    frame_ids: tuple[int, int, int] = (0, 1, 2),
    paired_corner_motion: bool = False,
) -> None:
    frames = [_face(), _face(), _face()]
    frames[1][291, 1] += 0.02
    frames[2][291, 1] += 0.04
    if paired_corner_motion:
        frames[1][61, 1] += 0.02
        frames[2][61, 1] += 0.04
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("frame", "point_idx", "x", "y", "z"))
        for frame_i, points in zip(frame_ids, frames):
            for point_i, (x, y, z) in enumerate(points):
                writer.writerow((frame_i, point_i, x, y, z))


def _write_video(path: Path, frame_count: int) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 32)
    )
    if not writer.isOpened():
        raise RuntimeError(f"test video writer did not open: {path}")
    try:
        for frame_i in range(frame_count):
            frame = np.full((32, 32, 3), frame_i, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_video_frame_count_exposes_trailing_csv_misses(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        take = root / "take_01"
        take.mkdir()
        _write_csv(take / "landmarks.csv")
        _write_video(take / "raw.mp4", frame_count=5)
        result = audit_directory(root, stride=1)
        record = result["records"][0]
    c.eq(record.get("video_frame_count"), 5, "video supplies the denominator")
    c.eq(record["frames_total"], 3, "CSV groups remain separately counted")
    c.eq(record.get("frames_missing_from_csv"), 2, "trailing misses are visible")
    c.eq(record.get("csv_group_coverage"), 0.6, "CSV group coverage uses video count")
    c.eq(
        record.get("valid_landmark_coverage"), 0.6,
        "valid landmark coverage uses video count",
    )
    c.true("coverage" not in record, "ambiguous legacy coverage is not reported")


def test_streaming_audit_extracts_dynamic_signal(c: Check):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "landmarks.csv"
        _write_csv(path)
        result = audit_landmark_csv(
            path, image_width=1000, image_height=1000, stride=1,
            video_frame_count=3,
        )
    c.eq(result["frames_total"], 3, "all frame groups counted")
    c.eq(result["frames_analyzed"], 3, "all frames analyzed")
    c.eq(result["points_per_frame"], 478, "MediaPipe mesh size")
    c.true(result["csv_group_coverage"] == 1.0, "contiguous CSV group coverage")
    c.true(result["valid_landmark_coverage"] == 1.0, "all landmarks valid")
    corner = result["features"]["corner_y_mesh61_minus_mesh291"]
    c.true(corner["peak_abs_early_delta"] > 0.0, "mouth motion captured")
    c.true(corner["max_abs_velocity_contiguous"] > 0.0,
           "temporal derivative captured")
    c.true(0.0 <= corner["time_to_global_peak_fraction"] <= 1.0,
           "normalized global peak time")
    c.true("end_recovery_ratio" not in corner,
           "whole-video endpoint is not mislabeled as action recovery")


def test_stride_is_explicit(c: Check):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "landmarks.csv"
        _write_csv(path)
        result = audit_landmark_csv(
            path, image_width=1000, image_height=1000, stride=2,
            video_frame_count=3,
        )
    c.eq(result["frames_total"], 3, "stride does not hide total frames")
    c.eq(result["frames_analyzed"], 2, "stride controls analyzed frames")
    c.eq(result["stride"], 2, "stride recorded for provenance")


def test_topology_pair_correlations_use_explicit_mesh_anchors(c: Check):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "landmarks.csv"
        _write_csv(path, paired_corner_motion=True)
        result = audit_landmark_csv(
            path, image_width=1000, image_height=1000, stride=1,
            video_frame_count=3,
        )
    correlations = result.get("topology_pair_trajectory_correlation")
    c.true(correlations is not None, "topology-pair field is reported")
    c.eq(
        set(correlations),
        {
            "fissure_h_mesh33_vs_mesh263",
            "fissure_w_mesh33_vs_mesh263",
            "eye_area_mesh33_vs_mesh263",
            "brow_h_mesh33_vs_mesh263",
            "corner_y_mesh61_vs_mesh291",
            "corner_x_mesh61_vs_mesh291",
        },
        "all explicit topology pairs are retained",
    )
    c.true(
        correlations["corner_y_mesh61_vs_mesh291"] > 0.99,
        "correlated mesh-anchor motion is measured",
    )
    c.true(
        "left_right_trajectory_correlation" not in result,
        "capture topology is not mislabeled as patient left/right",
    )


def test_sequence_artifact_is_schema_versioned(c: Check):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "landmarks.csv"
        video_path = Path(td) / "raw.mp4"
        output = Path(td) / "clinical23.npz"
        _write_csv(path)
        _write_video(video_path, frame_count=3)
        audit_landmark_csv(
            path, image_width=1000, image_height=1000, stride=1,
            sequence_output=output, video_frame_count=3,
            source_video=video_path,
        )
        saved = np.load(output)
        c.eq(saved["clinical23_seq"].shape, (3, 23), "per-frame sequence persisted")
        c.eq(saved["frame_ids"].tolist(), [0, 1, 2], "timestamps persisted")
        c.eq(str(saved["feature_schema"].item()), "clinical23_v2", "schema persisted")
        c.eq(len(saved["feature_names"]), 23, "feature order persisted")


def test_sequence_artifact_has_regular_timeline_mask_and_provenance(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        take = root / "take_01"
        derived = root / "derived"
        take.mkdir()
        csv_path = take / "landmarks.csv"
        video_path = take / "raw.mp4"
        _write_csv(csv_path)
        rows = csv_path.read_text().splitlines()
        del rows[1 + 478 + 100]  # frame 1 is present but invalid
        csv_path.write_text("\n".join(rows) + "\n")
        _write_video(video_path, frame_count=5)  # frames 3 and 4 are absent in CSV

        result = audit_directory(root, stride=1, derived_root=derived)
        record = result["records"][0]
        with np.load(derived / "take_01.npz") as saved:
            sequence = saved["clinical23_seq"]
            valid_mask = saved.get("valid_mask")
            c.eq(sequence.shape, (5, 23), "sequence follows the full video timeline")
            c.true(valid_mask is not None, "artifact has an explicit validity mask")
            c.eq(valid_mask.tolist(), [True, False, True, False, False], "invalid and missing frames stay visible")
            c.eq(saved["frame_ids"].tolist(), [0, 1, 2, 3, 4], "regular source frame ids persisted")
            c.true(np.isfinite(sequence[valid_mask]).all(), "valid rows contain features")
            c.true(np.isnan(sequence[~valid_mask]).all(), "invalid rows are NaN, never silent zeros")
            c.eq(int(saved["stride"].item()), 1, "stride persisted")
            c.eq(int(saved["video_frame_count"].item()), 5, "video denominator persisted")
            c.eq(str(saved["source_csv"].item()), str(csv_path.resolve()), "CSV path persisted")
            c.eq(str(saved["source_video"].item()), str(video_path.resolve()), "video path persisted")
            c.eq(str(saved["source_csv_sha256"].item()), record["csv_sha256"], "CSV digest persisted")


def test_malformed_frame_is_reported_not_zero_filled(c: Check):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "landmarks.csv"
        _write_csv(path)
        rows = path.read_text().splitlines()
        # Remove one point from the middle frame.
        del rows[1 + 478 + 100]
        path.write_text("\n".join(rows) + "\n")
        result = audit_landmark_csv(
            path, image_width=1000, image_height=1000, stride=1,
            video_frame_count=3,
        )
    c.eq(result["frames_total"], 3, "malformed group still counted")
    c.eq(result["frames_analyzed"], 2, "malformed frame excluded")
    c.eq(result["frames_invalid"], 1, "invalidity made visible")


def test_direct_audit_fails_closed_without_video_frame_count(c: Check):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "landmarks.csv"
        _write_csv(path)
        c.raises(
            lambda: audit_landmark_csv(
                path, image_width=1000, image_height=1000, stride=1,
            ),
            ValueError,
            "coverage denominator must come from video metadata",
        )


def test_sequence_artifact_fails_closed_without_video_provenance(c: Check):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "landmarks.csv"
        output = Path(td) / "clinical23.npz"
        _write_csv(path)
        c.raises(
            lambda: audit_landmark_csv(
                path, image_width=1000, image_height=1000, stride=1,
                sequence_output=output, video_frame_count=3,
            ),
            ValueError,
            "derived sequences require source-video provenance",
        )
        c.true(not output.exists(), "incomplete-provenance artifact is not written")


def test_sequence_artifact_rejects_mismatched_video_frame_count(c: Check):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "landmarks.csv"
        video_path = Path(td) / "raw.mp4"
        output = Path(td) / "clinical23.npz"
        _write_csv(path)
        _write_video(video_path, frame_count=4)
        c.raises(
            lambda: audit_landmark_csv(
                path, image_width=1000, image_height=1000, stride=1,
                sequence_output=output, video_frame_count=3,
                source_video=video_path,
            ),
            ValueError,
            "stored denominator must match source-video metadata",
        )
        c.true(not output.exists(), "mismatched-provenance artifact is not written")


def test_out_of_range_frame_id_fails_before_writing_sequence(c: Check):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "landmarks.csv"
        video_path = Path(td) / "raw.mp4"
        output = Path(td) / "clinical23.npz"
        _write_csv(path, frame_ids=(0, 1, 3))
        _write_video(video_path, frame_count=3)
        c.raises(
            lambda: audit_landmark_csv(
                path, image_width=1000, image_height=1000, stride=1,
                sequence_output=output, video_frame_count=3,
                source_video=video_path,
            ),
            ValueError,
            "CSV frame ids must fit the video timeline",
        )
        c.true(not output.exists(), "invalid-timeline artifact is not written")


def test_out_of_order_frame_ids_fail_before_writing_sequence(c: Check):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "landmarks.csv"
        video_path = Path(td) / "raw.mp4"
        output = Path(td) / "clinical23.npz"
        _write_csv(path, frame_ids=(0, 2, 1))
        _write_video(video_path, frame_count=3)
        c.raises(
            lambda: audit_landmark_csv(
                path, image_width=1000, image_height=1000, stride=1,
                sequence_output=output, video_frame_count=3,
                source_video=video_path,
            ),
            ValueError,
            "CSV frame ids must be strictly increasing",
        )
        c.true(not output.exists(), "out-of-order artifact is not written")


def test_summary_does_not_bridge_missing_frame_gaps(c: Check):
    summary = _feature_summary(
        np.asarray((0.0, 1.0, 10.0), np.float32),
        np.asarray((0, 1, 5), np.int64),
        expected_step=1,
    )
    c.eq(summary["auc_abs_early_delta_contiguous"], 0.5,
         "AUC only integrates contiguous observed intervals")
    c.eq(summary["max_abs_velocity_contiguous"], 1.0,
         "velocity never divides across a missing-frame gap")
    c.eq(summary["reference_semantics"],
         "provisional_early_valid_frames_not_action_rest",
         "early frames are not mislabeled as a clinical rest cue")
    c.eq(summary["contiguous_interval_count"], 1,
         "the report exposes how many derivative intervals exist")


def test_summary_marks_dynamics_undefined_without_contiguous_intervals(c: Check):
    summary = _feature_summary(
        np.asarray((0.0, 1.0, 2.0), np.float32),
        np.asarray((0, 2, 4), np.int64),
        expected_step=1,
    )
    c.eq(summary["contiguous_interval_count"], 0,
         "all observed points are separated by gaps")
    c.eq(summary["auc_abs_early_delta_contiguous"], None,
         "undefined contiguous AUC is not reported as zero motion")
    c.eq(summary["max_abs_velocity_contiguous"], None,
         "undefined velocity is not reported as zero motion")


def test_empty_or_missing_input_root_fails_closed(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        c.raises(lambda: audit_directory(root, stride=1), ValueError,
                 "empty audit input is not success")
        c.raises(lambda: audit_directory(root / "missing", stride=1),
                 FileNotFoundError, "missing audit input is not success")


def test_partial_audit_requires_explicit_opt_in(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        good = root / "good"
        bad = root / "bad"
        good.mkdir(); bad.mkdir()
        _write_csv(good / "landmarks.csv")
        _write_video(good / "raw.mp4", frame_count=3)
        _write_csv(bad / "landmarks.csv")  # no matching video
        c.raises(lambda: audit_directory(root, stride=1), RuntimeError,
                 "partial audit fails by default")
        result = audit_directory(root, stride=1, allow_partial=True)
        c.eq(result["record_count"], 1, "valid records retained after explicit opt-in")
        c.eq(result["failure_count"], 1, "partial failures remain visible")


def test_failed_directory_audit_does_not_publish_partial_sequences(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "input"
        derived = Path(td) / "derived"
        good = root / "good"
        bad = root / "bad"
        good.mkdir(parents=True); bad.mkdir()
        derived.mkdir()
        _write_csv(good / "landmarks.csv")
        _write_video(good / "raw.mp4", frame_count=3)
        _write_csv(bad / "landmarks.csv")  # missing source video
        sentinel = derived / "good.npz"
        sentinel.write_bytes(b"previous-complete-audit")
        c.raises(
            lambda: audit_directory(root, stride=1, derived_root=derived),
            RuntimeError,
            "default partial failure aborts before publishing any sequence",
        )
        c.eq(sentinel.read_bytes(), b"previous-complete-audit",
             "previous complete artifact remains untouched")
        c.eq(sorted(p.name for p in derived.iterdir()), ["good.npz"],
             "no staged success leaks into the canonical directory")


def test_collection_promotion_failure_rolls_back_complete_directory(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "input"
        derived = Path(td) / "derived"
        take = root / "take"
        take.mkdir(parents=True)
        derived.mkdir()
        _write_csv(take / "landmarks.csv")
        _write_video(take / "raw.mp4", frame_count=3)
        sentinel = derived / "previous.npz"
        sentinel.write_bytes(b"previous-complete-collection")

        real_replace = audit_module.os.replace

        def fail_new_collection_promotion(src, dst):
            src_path, dst_path = Path(src), Path(dst)
            if dst_path == derived and ".derived.staging-" in src_path.name:
                raise OSError("injected directory promotion failure")
            return real_replace(src, dst)

        audit_module.os.replace = fail_new_collection_promotion
        try:
            c.raises(
                lambda: audit_directory(root, stride=1, derived_root=derived),
                OSError,
                "failed new-directory promotion is surfaced",
            )
        finally:
            audit_module.os.replace = real_replace

        c.eq(sentinel.read_bytes(), b"previous-complete-collection",
             "the prior complete directory is restored")
        c.eq(sorted(p.name for p in derived.iterdir()), ["previous.npz"],
             "no new/old artifact mixture remains after rollback")


if __name__ == "__main__":
    run_all("test_mayo_landmark_audit", dict(globals()))
