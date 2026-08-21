"""Contracts for the NeuroFace AU, 110D, and fusion comparison runner."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from scripts.run_neuroface_als_au_geometry_v1 import (  # noqa: E402
    REPRESENTATIONS,
    build_spread_representations,
    parser,
)
from src.datasets.dynamic_landmark import DynamicLandmarkRecording  # noqa: E402
from src.datasets.neuroface_au_v1 import build_au_recording  # noqa: E402
from _testlib import Check, run_all  # noqa: E402


def _fixture():
    selected = []
    dynamic = {}
    au = {}
    starts = np.asarray([0, 64, 128, 192], dtype=np.int64)
    source_indices = starts[:, None] + np.arange(32, dtype=np.int64)[None, :]
    timestamps = source_indices.astype(np.float64) / 30.0
    for participant in range(22):
        cohort = "als" if participant < 11 else "healthy_control"
        label = 1 if cohort == "als" else 0
        group = f"grp_{participant:064x}"
        recording_id = f"rec_{participant:064x}"
        source = f"{participant + 500:064x}"
        selected.append({
            "recording_id": recording_id,
            "participant_id": group,
            "video_sha256": source,
            "task": "NSM_SPREAD",
            "cohort": cohort,
        })
        features = np.zeros((4, 32, 95), dtype=np.float32)
        features[..., 72:] = label + np.arange(23, dtype=np.float32) / 100.0
        dynamic[recording_id] = DynamicLandmarkRecording(
            features=features,
            valid_mask=np.ones((4, 32), dtype=bool),
            timestamps=timestamps,
            timestamp_unit="seconds",
            source_frame_indices=source_indices,
            source_frame_count=224,
            feature_schema="mediapipe_bs_lr_v1+clinical23_v2",
            feature_names=tuple(f"feature_{index}" for index in range(95)),
            recording_id=recording_id,
            group_id=group,
            label=label,
            source_sha256=source,
            cache_path=Path("<fixture>"),
        )
        frame_count = 6
        au[recording_id] = build_au_recording(
            recording_id=recording_id,
            group_id=group,
            task="NSM_SPREAD",
            source_sha256=source,
            source_frame_count=frame_count,
            fps=30.0,
            frame_indices=np.arange(frame_count, dtype=np.int64),
            timestamps=np.arange(frame_count, dtype=np.float64) / 30.0,
            au_values=np.full((frame_count, 20), label, dtype=np.float32),
            valid_mask=np.ones(frame_count, dtype=bool),
            selected_face_count=np.ones(frame_count, dtype=np.int16),
            selected_face_score=np.full(frame_count, 0.9, dtype=np.float32),
        )
    return selected, dynamic, au


def test_representation_shapes_and_participant_alignment(c: Check):
    selected, dynamic, au = _fixture()
    matrices, labels, groups, coverage = build_spread_representations(
        selected, dynamic, au
    )
    c.eq(REPRESENTATIONS, (
        "paper_pyfeat_min_au_20d",
        "paper_pyfeat_all_stats_100d",
        "landmark_110d",
        "min_au_110d_fusion_130d",
        "all_stats_au_110d_fusion_210d",
    ),
         "comparison is frozen before real evaluation")
    c.eq({name: value.shape for name, value in matrices.items()}, {
        "paper_pyfeat_min_au_20d": (22, 20),
        "paper_pyfeat_all_stats_100d": (22, 100),
        "landmark_110d": (22, 110),
        "min_au_110d_fusion_130d": (22, 130),
        "all_stats_au_110d_fusion_210d": (22, 210),
    }, "all representations contain one aligned row per participant")
    c.eq(labels.tolist(), [1] * 11 + [0] * 11, "participant labels remain aligned")
    c.eq(len(set(groups)), 22, "participant groups are unique")
    c.true(np.all(coverage == 1.0), "AU detector coverage is retained")


def test_identity_mismatch_and_missing_participant_fail(c: Check):
    selected, dynamic, au = _fixture()
    broken = dict(au)
    broken.pop(next(iter(broken)))
    c.raises(lambda: build_spread_representations(selected, dynamic, broken),
             ValueError, "missing AU evidence fails closed")


def test_cli_cannot_select_candidate_or_hyperparameters(c: Check):
    options = {action.dest for action in parser()._actions}
    c.true(options.isdisjoint({
        "c", "penalty", "candidate", "representation", "task", "threshold"
    }), "endpoint cannot be tuned from the CLI")
    c.true("--au-cache-root" in inspect.getsource(parser),
           "runner accepts the authenticated AU cache root")
    from scripts.run_neuroface_als_au_geometry_v1 import main
    main_source = inspect.getsource(main)
    c.true("evaluate_nested_loso(paper_statistic_matrices" in main_source
           and "strict_nested_paper_statistic_search" in main_source,
           "five AU statistics are selected only inside strict participant folds")


if __name__ == "__main__":
    run_all("test_run_neuroface_als_au_geometry_v1", dict(globals()))
