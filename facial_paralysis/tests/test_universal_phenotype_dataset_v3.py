"""Participant-level contracts for Universal Phenotype Mixture v3."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.datasets.universal_phenotype_v3 import (  # noqa: E402
    AU_DIM,
    AU_TEMPORAL_SAMPLES,
    COMMON_DIM,
    LANDMARK_DIM,
    MAX_INSTANCES,
    RecordingEvidence,
    build_phenotype_dataset,
)
from _testlib import Check, run_all  # noqa: E402


def _hex(index: int) -> str:
    return f"{index:064x}"


def _row(
    recording: int,
    group: int,
    *,
    label: int,
    source: str,
    phenotype: str,
    task: str,
    au: bool,
) -> RecordingEvidence:
    rng = np.random.default_rng(recording)
    temporal = rng.normal(size=(4, 32, 95)).astype(np.float32)
    valid = np.ones((4, 32), dtype=bool)
    timestamps = np.stack([
        np.arange(32, dtype=np.float64) / 30.0 + window * 10.0
        for window in range(4)
    ])
    indices = np.stack([
        np.arange(32, dtype=np.int64) + window * 300
        for window in range(4)
    ])
    return RecordingEvidence(
        recording_id="rec_" + _hex(recording),
        group_id="grp_" + _hex(group),
        label=label,
        evaluation_source=source,
        phenotype=phenotype,
        task=task,
        landmark_original=np.full(LANDMARK_DIM, recording, dtype=np.float64),
        landmark_mirrored=np.full(LANDMARK_DIM, -recording, dtype=np.float64),
        common_original=np.full(COMMON_DIM, recording + 0.25, dtype=np.float64),
        common_mirrored=np.full(COMMON_DIM, -recording - 0.25, dtype=np.float64),
        temporal_features=temporal,
        temporal_valid_mask=valid,
        temporal_timestamps=timestamps,
        temporal_source_frame_indices=indices,
        au_summary=(
            np.full(AU_DIM, recording + 0.5, dtype=np.float64) if au else None
        ),
        au_temporal=(
            np.full((AU_TEMPORAL_SAMPLES, 20), recording + 0.75,
                    dtype=np.float32) if au else None
        ),
        au_temporal_mask=(
            np.ones(AU_TEMPORAL_SAMPLES, dtype=bool) if au else None
        ),
    )


def test_builder_retains_bags_and_aggregates_landmark_per_participant(c: Check):
    rows = (
        _row(1, 101, label=1, source="neuroface", phenotype="als",
             task="NSM_SPREAD", au=True),
        _row(2, 101, label=1, source="neuroface", phenotype="als",
             task="NSM_OPEN", au=True),
        _row(3, 102, label=0, source="palsynet", phenotype="healthy",
             task="FREE_RECORDING", au=False),
    )
    dataset = build_phenotype_dataset(rows)
    c.eq(dataset.group_ids, ("grp_" + _hex(101), "grp_" + _hex(102)),
         "participants have deterministic opaque order")
    c.eq(dataset.instance_counts.tolist(), [2, 1],
         "recordings remain separate MIL instances")
    c.eq(float(dataset.landmark_original[0, 0]), 1.5,
         "landmark evidence averages only within participant")
    c.true(dataset.instance_mask[0, :2].all() and not dataset.instance_mask[0, 2:].any(),
           "instance missingness is explicit")
    c.true(dataset.au_mask[0, :2].all() and not dataset.au_mask[1].any(),
           "AU availability is explicit rather than inferred from zeros")
    c.true(np.all(dataset.au_instances[1] == 0),
           "unavailable AU storage is canonical zero behind a false mask")
    c.eq(dataset.au_temporal.shape, (2, MAX_INSTANCES, AU_TEMPORAL_SAMPLES, 20),
         "frame-level AU dynamics remain available to the temporal AU expert")
    c.true(np.all(dataset.au_temporal[~dataset.au_temporal_mask] == 0),
           "missing temporal AU samples are canonical zero")
    c.eq(dataset.temporal_features.shape, (2, MAX_INSTANCES, 4, 32, 95),
         "raw common temporal evidence remains available to the temporal expert")


def test_dataset_arrays_are_immutable_and_exactly_typed(c: Check):
    dataset = build_phenotype_dataset((
        _row(4, 103, label=1, source="palsynet", phenotype="palsy",
             task="FREE_RECORDING", au=False),
        _row(5, 104, label=0, source="neuroface", phenotype="healthy",
             task="NSM_KISS", au=True),
    ))
    for array in (
        dataset.landmark_original, dataset.landmark_mirrored,
        dataset.common_original, dataset.common_mirrored,
        dataset.instance_mask, dataset.au_instances, dataset.au_mask,
        dataset.temporal_features, dataset.temporal_valid_mask,
        dataset.temporal_timestamps, dataset.temporal_source_frame_indices,
        dataset.labels, dataset.task_codes, dataset.instance_counts,
        dataset.au_temporal, dataset.au_temporal_mask,
    ):
        c.true(not array.flags.writeable, "every dataset tensor is immutable")
        c.raises(lambda array=array: array.setflags(write=True), ValueError,
                 "immutable backing cannot be re-enabled")
    c.eq(dataset.temporal_features.dtype, np.dtype(np.float32),
         "raw temporal features preserve float32 cache semantics")
    c.eq(dataset.landmark_original.dtype, np.dtype(np.float64),
         "aggregated geometric evidence uses float64")


def test_builder_rejects_identity_label_source_and_duplicate_drift(c: Check):
    first = _row(6, 105, label=1, source="palsynet", phenotype="palsy",
                 task="FREE_RECORDING", au=False)
    c.raises(lambda: build_phenotype_dataset((
        first,
        replace(first, recording_id="rec_" + _hex(7), label=0),
    )), ValueError, "one participant cannot cross labels")
    c.raises(lambda: build_phenotype_dataset((
        first,
        replace(first, recording_id="rec_" + _hex(8), evaluation_source="neuroface"),
    )), ValueError, "one participant cannot cross evaluation sources")
    c.raises(lambda: build_phenotype_dataset((first, first)), ValueError,
             "one recording cannot be duplicated")


def test_repeated_free_recordings_are_retained_but_script_actions_stay_unique(c: Check):
    first = _row(10, 107, label=1, source="palsynet", phenotype="palsy",
                 task="FREE_RECORDING", au=False)
    second = replace(first, recording_id="rec_" + _hex(11))
    dataset = build_phenotype_dataset((first, second))
    c.eq(dataset.instance_counts.tolist(), [2],
         "multiple unscripted recordings from one patient remain separate evidence")

    scripted = _row(12, 108, label=1, source="neuroface", phenotype="als",
                    task="NSM_OPEN", au=True)
    c.raises(lambda: build_phenotype_dataset((
        scripted, replace(scripted, recording_id="rec_" + _hex(13)),
    )), ValueError, "one scripted action cannot be duplicated for a participant")


def test_invalid_rows_and_missing_au_never_become_observed_zero(c: Check):
    row = _row(9, 106, label=1, source="neuroface", phenotype="post_stroke",
               task="NSM_BLOW", au=False)
    invalid_temporal = np.array(row.temporal_features, copy=True)
    invalid_mask = np.array(row.temporal_valid_mask, copy=True)
    invalid_mask[0, 0] = False
    invalid_temporal[0, 0, 0] = 1.0
    c.raises(lambda: build_phenotype_dataset((
        replace(row, temporal_features=invalid_temporal,
                temporal_valid_mask=invalid_mask),
    )), ValueError, "invalid temporal rows must be canonical zero")
    c.raises(lambda: build_phenotype_dataset((
        replace(row, au_summary=np.zeros(AU_DIM - 1, dtype=np.float64)),
    )), ValueError, "present AU evidence must use the exact schema")


def test_temporal_support_uses_the_authenticated_recording_level_gate(c: Check):
    row = _row(14, 109, label=1, source="neuroface", phenotype="post_stroke",
               task="DDK_PA", au=True)
    mask = np.array(row.temporal_valid_mask, copy=True)
    features = np.array(row.temporal_features, copy=True)
    mask[0, :10] = False
    features[~mask] = 0.0
    dataset = build_phenotype_dataset((replace(
        row, temporal_features=features, temporal_valid_mask=mask,
    ),))
    c.eq(int(dataset.temporal_valid_mask.sum()), 118,
         "uneven windows remain valid when authenticated total coverage exceeds 90%")

    mask[0, 10:16] = False
    features[~mask] = 0.0
    c.raises(lambda: build_phenotype_dataset((replace(
        row, temporal_features=features, temporal_valid_mask=mask,
    ),)), ValueError, "recording-level temporal coverage below 90% fails closed")


if __name__ == "__main__":
    run_all("test_universal_phenotype_dataset_v3", dict(globals()))
