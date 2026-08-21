"""Contracts for the paper-comparable full-frame NeuroFace AU cache."""
from __future__ import annotations

import io
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.datasets.neuroface_au_v1 import (  # noqa: E402
    AU_NAMES,
    PAPER_PYFEAT_VERSION,
    PAPER_TASKS,
    NeuroFaceAURecording,
    build_au_recording,
    load_au_recording_bytes,
    publish_au_cache,
    serialize_au_recording,
    summarize_au_recording,
)
from _testlib import Check, run_all  # noqa: E402


def _recording(*, missing: int = 0) -> NeuroFaceAURecording:
    frames = 40
    values = np.arange(frames * 20, dtype=np.float32).reshape(frames, 20) / 100.0
    valid = np.ones(frames, dtype=bool)
    valid[:missing] = False
    values[~valid] = 0.0
    return build_au_recording(
        recording_id="rec_" + "1" * 64,
        group_id="grp_" + "2" * 64,
        task="NSM_SPREAD",
        source_sha256="3" * 64,
        source_frame_count=frames,
        fps=50.0,
        frame_indices=np.arange(frames, dtype=np.int64),
        timestamps=np.arange(frames, dtype=np.float64) / 50.0,
        au_values=values,
        valid_mask=valid,
        selected_face_count=np.where(valid, 1, 0).astype(np.int16),
        selected_face_score=np.where(valid, 0.99, 0.0).astype(np.float32),
    )


def test_schema_matches_paper_detector(c: Check):
    c.eq(PAPER_PYFEAT_VERSION, "0.6.2", "paper-era Py-Feat is pinned")
    c.eq(PAPER_TASKS, ("NSM_KISS", "NSM_OPEN", "NSM_SPREAD"),
         "three complete ALS/healthy tasks are ordered")
    c.eq(AU_NAMES, (
        "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10",
        "AU11", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24",
        "AU25", "AU26", "AU28", "AU43",
    ), "the exact 20 XGBoost AU outputs are name-bound")


def test_build_validates_identity_time_and_missingness(c: Check):
    recording = _recording(missing=2)
    c.eq(recording.coverage, 0.95, "face-detection coverage remains explicit")
    c.true(all(not value.flags.writeable for value in (
        recording.frame_indices,
        recording.timestamps,
        recording.au_values,
        recording.valid_mask,
        recording.selected_face_count,
        recording.selected_face_score,
    )), "cache arrays are immutable")

    kwargs = dict(
        recording_id="rec_" + "1" * 64,
        group_id="grp_" + "2" * 64,
        task="NSM_SPREAD",
        source_sha256="3" * 64,
        source_frame_count=3,
        fps=50.0,
        frame_indices=np.arange(3, dtype=np.int64),
        timestamps=np.arange(3, dtype=np.float64) / 50.0,
        au_values=np.zeros((3, 20), dtype=np.float32),
        valid_mask=np.ones(3, dtype=bool),
        selected_face_count=np.ones(3, dtype=np.int16),
        selected_face_score=np.ones(3, dtype=np.float32),
    )
    bad = dict(kwargs)
    bad["task"] = "NSM_BROW"
    c.raises(lambda: build_au_recording(**bad), ValueError,
             "unlocked tasks are rejected")
    bad = dict(kwargs)
    bad["frame_indices"] = np.asarray([0, 2, 1], dtype=np.int64)
    c.raises(lambda: build_au_recording(**bad), ValueError,
             "frame order cannot drift")
    bad = dict(kwargs)
    bad["au_values"] = np.full((3, 20), np.nan, dtype=np.float32)
    c.raises(lambda: build_au_recording(**bad), ValueError,
             "valid AU rows must be finite")


def test_summary_is_exact_and_excludes_missing_rows(c: Check):
    recording = _recording(missing=2)
    summary = summarize_au_recording(recording)
    observed = recording.au_values[recording.valid_mask].astype(np.float64)
    c.eq(summary.feature_names[:3], ("mean_AU01", "mean_AU02", "mean_AU04"),
         "summary order is statistic-major and AU-name-bound")
    c.eq(summary.feature_names[20:23], ("min_AU01", "min_AU02", "min_AU04"),
         "paper minimum feature block is stable")
    expected = np.concatenate((
        observed.mean(axis=0), observed.min(axis=0), observed.max(axis=0),
        observed.std(axis=0, ddof=0), observed.var(axis=0, ddof=0),
    ))
    c.true(np.array_equal(summary.values, expected),
           "mean/min/max/std/variance exactly match valid frames")
    c.eq(summary.valid_frames, 38, "missing detections are not imputed")


def test_roundtrip_is_closed_and_duplicate_npz_members_fail(c: Check):
    recording = _recording()
    payload = serialize_au_recording(recording)
    loaded = load_au_recording_bytes(payload)
    c.eq(loaded.recording_id, recording.recording_id, "opaque identity round-trips")
    c.true(np.array_equal(loaded.au_values, recording.au_values),
           "AU values round-trip exactly")

    duplicate = io.BytesIO()
    with zipfile.ZipFile(duplicate, "w") as archive:
        archive.writestr("au_values.npy", b"first")
        archive.writestr("au_values.npy", b"second")
    c.raises(lambda: load_au_recording_bytes(duplicate.getvalue()), ValueError,
             "ambiguous duplicate NPZ members fail closed")
    c.raises(lambda: load_au_recording_bytes(b"PK\x03\x04"), ValueError,
             "malformed ZIP errors are normalized")


def test_publication_is_no_overwrite(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary) / "cache.npz"
        digest = publish_au_cache(target, serialize_au_recording(_recording()))
        c.eq(len(digest), 64, "publication returns a SHA-256 commitment")
        c.eq(target.stat().st_mode & 0o777, 0o600, "private cache is owner-only")
        c.raises(lambda: publish_au_cache(target, b"replacement"), FileExistsError,
                 "existing evidence is never overwritten")


if __name__ == "__main__":
    run_all("test_neuroface_au_v1", dict(globals()))
