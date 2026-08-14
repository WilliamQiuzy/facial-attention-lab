"""Contracts for participant-level KISS/OPEN/SPREAD AU evaluation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from scripts.run_neuroface_als_multitask_au_v1 import (  # noqa: E402
    ALL_STATS_REPRESENTATION,
    TASK_STRUCTURED_REPRESENTATIONS,
    build_multitask_representations,
    parser,
)
from src.datasets.neuroface_au_v1 import PAPER_TASKS, build_au_recording  # noqa: E402
from _testlib import Check, run_all  # noqa: E402


def _fixture():
    selected = []
    recordings = {}
    for participant in range(22):
        cohort = "als" if participant < 11 else "healthy_control"
        label = 1 if cohort == "als" else 0
        group = f"grp_{participant:064x}"
        for task_index, task in enumerate(PAPER_TASKS):
            ordinal = participant * 3 + task_index
            recording_id = f"rec_{ordinal:064x}"
            source = f"{ordinal + 700:064x}"
            selected.append({
                "recording_id": recording_id,
                "participant_id": group,
                "video_sha256": source,
                "task": task,
                "cohort": cohort,
            })
            frames = 8
            values = np.full((frames, 20), label + task_index / 10.0, dtype=np.float32)
            recordings[recording_id] = build_au_recording(
                recording_id=recording_id,
                group_id=group,
                task=task,
                source_sha256=source,
                source_frame_count=frames,
                fps=30.0,
                frame_indices=np.arange(frames, dtype=np.int64),
                timestamps=np.arange(frames, dtype=np.float64) / 30.0,
                au_values=values,
                valid_mask=np.ones(frames, dtype=bool),
                selected_face_count=np.ones(frames, dtype=np.int16),
                selected_face_score=np.full(frames, 0.95, dtype=np.float32),
            )
    return selected, recordings


def test_three_tasks_form_one_participant_row(c: Check):
    selected, recordings = _fixture()
    matrices, labels, groups, coverage = build_multitask_representations(
        selected, recordings
    )
    c.eq(TASK_STRUCTURED_REPRESENTATIONS, (
        "three_task_mean_au_60d",
        "three_task_min_au_60d",
        "three_task_max_au_60d",
        "three_task_std_au_60d",
        "three_task_var_au_60d",
    ), "five statistic candidates are frozen before real evaluation")
    c.true(all(matrices[name].shape == (22, 60)
               for name in TASK_STRUCTURED_REPRESENTATIONS),
           "each statistic concatenates three named 20D task blocks")
    c.eq(matrices[ALL_STATS_REPRESENTATION].shape, (22, 300),
         "all-statistics ablation is task-major 300D")
    c.eq(labels.tolist(), [1] * 11 + [0] * 11,
         "labels occur once per participant")
    c.eq(len(set(groups)), 22, "participant identities are unique")
    c.true(np.all(coverage == 1.0), "per-task AU coverage remains explicit")


def test_incomplete_task_cartesian_product_fails(c: Check):
    selected, recordings = _fixture()
    recordings.pop(next(iter(recordings)))
    c.raises(lambda: build_multitask_representations(selected, recordings),
             ValueError, "a missing task cannot be imputed")


def test_cli_has_no_model_selection_controls(c: Check):
    options = {action.dest for action in parser()._actions}
    c.true(options.isdisjoint({
        "task", "statistic", "candidate", "threshold", "c", "penalty"
    }), "all selection stays inside frozen participant folds")


if __name__ == "__main__":
    run_all("test_run_neuroface_als_multitask_au_v1", dict(globals()))
