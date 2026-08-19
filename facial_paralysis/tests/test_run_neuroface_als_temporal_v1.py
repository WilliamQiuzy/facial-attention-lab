"""Runner contracts for the fixed NeuroFace participant-level TCN."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from scripts.run_neuroface_als_temporal_v1 import (  # noqa: E402
    assemble_temporal_dataset,
    build_public_report,
    parser,
)
from src.datasets.dynamic_landmark import DynamicLandmarkRecording  # noqa: E402
from _testlib import Check, run_all  # noqa: E402


TASKS = ("NSM_KISS", "NSM_OPEN", "NSM_SPREAD")


def _fixture():
    selected = []
    dynamic_rows = []
    recordings = {}
    for participant_index in range(22):
        group = f"grp_{participant_index:064x}"
        cohort = "als" if participant_index < 11 else "healthy_control"
        label = 1 if cohort == "als" else 0
        for task_index, task in enumerate(TASKS):
            ordinal = participant_index * 3 + task_index
            recording_id = f"rec_{ordinal:064x}"
            source_sha = f"{ordinal + 1000:064x}"
            cache_sha = f"{ordinal + 2000:064x}"
            selected.append({
                "recording_id": recording_id,
                "participant_id": group,
                "video_sha256": source_sha,
                "task": task,
                "cohort": cohort,
            })
            dynamic_rows.append({
                "recording_id": recording_id,
                "participant_id": group,
                "video_sha256": source_sha,
                "cache_sha256": cache_sha,
                "status": "retained",
            })
            features = np.full((4, 32, 95), label, dtype=np.float32)
            mask = np.ones((4, 32), dtype=bool)
            timestamps = np.tile(np.arange(32, dtype=np.float64) / 30.0, (4, 1))
            recordings[recording_id] = DynamicLandmarkRecording(
                features=features,
                valid_mask=mask,
                timestamps=timestamps,
                timestamp_unit="seconds",
                source_frame_indices=np.tile(np.arange(32, dtype=np.int64), (4, 1)),
                source_frame_count=128,
                feature_schema="mediapipe_bs_lr_v1+clinical23_v2",
                feature_names=tuple(f"feature_{i}" for i in range(95)),
                recording_id=recording_id,
                group_id=group,
                label=label,
                source_sha256=source_sha,
                cache_path=Path("<fixture>"),
            )
    return selected, dynamic_rows, recordings


def test_assembler_emits_one_row_per_person_and_three_tasks(c: Check):
    selected, dynamic_rows, recordings = _fixture()
    dataset = assemble_temporal_dataset(selected, dynamic_rows, recordings)
    c.eq(dataset.features.shape, (22, 3, 4, 32, 95),
         "three tasks stay inside one participant row")
    c.eq(dataset.labels.tolist(), [1] * 11 + [0] * 11,
         "ALS and healthy labels remain participant-level")
    c.eq(len(set(dataset.group_ids)), 22, "all outer groups are unique")
    c.raises(lambda: assemble_temporal_dataset(
        selected, dynamic_rows, {key: value for key, value in list(recordings.items())[:-1]}
    ), ValueError, "a missing task cache fails closed")


def test_public_report_recomputes_metrics_and_excludes_ids(c: Check):
    labels = np.asarray([1] * 11 + [0] * 11, dtype=np.int64)
    probabilities = np.linspace(0.95, 0.55, 11).tolist() + np.linspace(0.45, 0.05, 11).tolist()
    report = build_public_report(labels, np.asarray(probabilities, dtype=np.float64))
    c.eq(report["counts"], {"participants": 22, "als": 11, "healthy": 11},
         "public report exposes only aggregate counts")
    c.eq(report["metrics"]["accuracy"], 1.0, "metrics are recomputed")
    rendered = repr(report).casefold()
    c.true("grp_" not in rendered and "rec_" not in rendered,
           "public report contains no participant or recording identifier")


def test_cli_has_no_training_or_candidate_controls(c: Check):
    options = {action.dest for action in parser()._actions}
    forbidden = {"epochs", "seed", "learning_rate", "architecture", "candidate", "task"}
    c.true(options.isdisjoint(forbidden), "CLI cannot tune on the endpoint")
    source = inspect.getsource(parser)
    c.true("--private-manifest" in source and "--dynamic-cache-root" in source,
           "runner receives only authoritative inputs and output destination")


if __name__ == "__main__":
    run_all("test_run_neuroface_als_temporal_v1", dict(globals()))
