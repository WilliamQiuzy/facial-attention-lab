"""Contracts for the closed nine-action NeuroFace AU cache."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.datasets.neuroface_au_v1 import (  # noqa: E402
    build_au_recording as build_v1_recording,
    serialize_au_recording as serialize_v1_recording,
)
from src.datasets.neuroface_au_v2 import (  # noqa: E402
    ALL_TASKS,
    SCHEMA_VERSION,
    TEMPORAL_SAMPLES,
    build_full_au_recording,
    load_full_au_recording_bytes,
    serialize_full_au_recording,
    summarize_full_au_recording,
    temporal_full_au_view,
)
from _testlib import Check, run_all  # noqa: E402


def _id(prefix: str, value: str) -> str:
    return prefix + hashlib.sha256(value.encode("ascii")).hexdigest()


def _record(task: str):
    frames = 8
    sampled = np.arange(0, frames, 3, dtype=np.int64)
    values = np.arange(sampled.size * 20, dtype=np.float32).reshape(sampled.size, 20) / 100.0
    return build_full_au_recording(
        recording_id=_id("rec_", f"recording-{task}"),
        group_id=_id("grp_", "participant"),
        task=task,
        source_sha256=_id("", f"source-{task}"),
        source_frame_count=frames,
        fps=20.0,
        sampling_stride=3,
        frame_indices=sampled,
        timestamps=sampled.astype(np.float64) / 20.0,
        au_values=values,
        valid_mask=np.ones(sampled.size, dtype=bool),
        selected_face_count=np.ones(sampled.size, dtype=np.int16),
        selected_face_score=np.full(sampled.size, 0.9, dtype=np.float32),
    )


def test_all_nine_actions_round_trip_under_one_v2_schema(c: Check):
    c.eq(len(ALL_TASKS), 9, "full cache freezes all nine released actions")
    for task in ALL_TASKS:
        record = _record(task)
        loaded = load_full_au_recording_bytes(serialize_full_au_recording(record))
        c.eq(loaded.task, task, f"{task} survives the closed round trip")
        c.eq(loaded.au_values.shape, (3, 20), "sampled AU evidence is retained")
        c.eq(loaded.source_frame_count, 8, "original decoded frame count is retained")
        c.eq(loaded.frame_indices.tolist(), [0, 3, 6],
             "sampled rows remain bound to original frame indices")


def test_v2_rejects_historical_v1_payload_and_unknown_task(c: Check):
    v1 = build_v1_recording(
        recording_id=_id("rec_", "v1"), group_id=_id("grp_", "v1"),
        task="NSM_SPREAD", source_sha256=_id("", "v1"),
        source_frame_count=2, fps=20.0,
        frame_indices=np.arange(2, dtype=np.int64),
        timestamps=np.arange(2, dtype=np.float64) / 20.0,
        au_values=np.ones((2, 20), dtype=np.float32),
        valid_mask=np.ones(2, dtype=bool),
        selected_face_count=np.ones(2, dtype=np.int16),
        selected_face_score=np.full(2, 0.9, dtype=np.float32),
    )
    c.raises(
        lambda: load_full_au_recording_bytes(serialize_v1_recording(v1)),
        ValueError,
        "the full collection cannot silently mix v1 and v2 cache schemas",
    )
    c.raises(lambda: _record("UNKNOWN_ACTION"), ValueError,
             "an uncommitted action cannot enter the full cache")


def test_summary_is_exact_100d_and_does_not_hide_missing_frames(c: Check):
    record = _record("NSM_BROW")
    summary = summarize_full_au_recording(record)
    c.eq(summary.values.shape, (100,), "five statistics by twenty AUs is 100D")
    c.eq(summary.valid_frames, 3, "summary reports actual valid support")
    c.eq(summary.total_frames, 8, "summary reports decoded frame count")
    c.eq(SCHEMA_VERSION, "neuroface_pyfeat_xgb_au_temporal_sample_v2",
         "sampled cache owns a distinct schema")
    sequence, mask = temporal_full_au_view(record)
    c.eq(sequence.shape, (TEMPORAL_SAMPLES, 20),
         "the neural AU expert gets a fixed temporal sequence")
    c.eq(mask.shape, (TEMPORAL_SAMPLES,),
         "temporal AU missingness remains explicit")
    c.true(np.all(sequence[~mask] == 0),
           "invalid temporal AU positions are canonical zero")

    c.raises(lambda: build_full_au_recording(
        recording_id=record.recording_id, group_id=record.group_id,
        task=record.task, source_sha256=record.source_sha256,
        source_frame_count=8, fps=20.0, sampling_stride=3,
        frame_indices=np.asarray([0, 2, 6], dtype=np.int64),
        timestamps=np.asarray([0.0, 0.1, 0.3], dtype=np.float64),
        au_values=record.au_values, valid_mask=record.valid_mask,
        selected_face_count=record.selected_face_count,
        selected_face_score=record.selected_face_score,
    ), ValueError, "sample membership must be the frozen regular source clock")


if __name__ == "__main__":
    run_all("test_neuroface_au_v2", dict(globals()))
