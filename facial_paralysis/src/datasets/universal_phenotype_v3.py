"""Strict participant evidence contract for Universal Phenotype Mixture v3."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.datasets.dynamic_landmark import MIN_RECORDING_COVERAGE


LANDMARK_DIM = 110
COMMON_DIM = 398
AU_DIM = 100
AU_TEMPORAL_SAMPLES = 64
AU_CHANNELS = 20
MAX_INSTANCES = 9
TASKS = (
    "FREE_RECORDING",
    "NSM_SPREAD",
    "NSM_KISS",
    "NSM_OPEN",
    "NSM_BLOW",
    "NSM_BROW",
    "NSM_BIGSMILE",
    "DDK_PA",
    "DDK_PATAKA",
    "BBP_NORMAL",
)
EVALUATION_SOURCES = ("palsynet", "neuroface", "meei")
PHENOTYPES = ("healthy", "palsy", "als", "post_stroke")

_REC_ID = re.compile(r"^rec_[0-9a-f]{64}$")
_GROUP_ID = re.compile(r"^grp_[0-9a-f]{64}$")
_TASK_CODE = {task: index for index, task in enumerate(TASKS)}


def _immutable(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


@dataclass(frozen=True)
class RecordingEvidence:
    recording_id: str
    group_id: str
    label: int
    evaluation_source: str
    phenotype: str
    task: str
    landmark_original: np.ndarray
    landmark_mirrored: np.ndarray
    common_original: np.ndarray
    common_mirrored: np.ndarray
    temporal_features: np.ndarray
    temporal_valid_mask: np.ndarray
    temporal_timestamps: np.ndarray
    temporal_source_frame_indices: np.ndarray
    au_summary: np.ndarray | None
    au_temporal: np.ndarray | None
    au_temporal_mask: np.ndarray | None


@dataclass(frozen=True)
class PhenotypeDataset:
    landmark_original: np.ndarray
    landmark_mirrored: np.ndarray
    common_original: np.ndarray
    common_mirrored: np.ndarray
    instance_mask: np.ndarray
    temporal_features: np.ndarray
    temporal_valid_mask: np.ndarray
    temporal_timestamps: np.ndarray
    temporal_source_frame_indices: np.ndarray
    au_instances: np.ndarray
    au_mask: np.ndarray
    au_temporal: np.ndarray
    au_temporal_mask: np.ndarray
    task_codes: np.ndarray
    instance_counts: np.ndarray
    labels: np.ndarray
    group_ids: tuple[str, ...]
    evaluation_sources: tuple[str, ...]
    phenotypes: tuple[str, ...]


def _exact_float_vector(value: object, dimension: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.shape != (dimension,)
        or array.dtype != np.dtype(np.float64)
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be exact finite float64 {dimension}D")
    return array


def _validate_recording(row: RecordingEvidence) -> None:
    if not isinstance(row, RecordingEvidence):
        raise ValueError("evidence rows must use RecordingEvidence")
    if not isinstance(row.recording_id, str) or _REC_ID.fullmatch(row.recording_id) is None:
        raise ValueError("recording_id is not a canonical opaque identifier")
    if not isinstance(row.group_id, str) or _GROUP_ID.fullmatch(row.group_id) is None:
        raise ValueError("group_id is not a canonical opaque identifier")
    if isinstance(row.label, bool) or not isinstance(row.label, int) or row.label not in (0, 1):
        raise ValueError("label must be an exact binary integer")
    if row.evaluation_source not in EVALUATION_SOURCES:
        raise ValueError("evaluation source differs from the closed registry")
    if row.phenotype not in PHENOTYPES:
        raise ValueError("phenotype differs from the closed registry")
    if row.task not in _TASK_CODE:
        raise ValueError("task differs from the closed registry")
    if (row.phenotype == "healthy") != (row.label == 0):
        raise ValueError("phenotype and affected label disagree")
    _exact_float_vector(row.landmark_original, LANDMARK_DIM, "landmark_original")
    _exact_float_vector(row.landmark_mirrored, LANDMARK_DIM, "landmark_mirrored")
    _exact_float_vector(row.common_original, COMMON_DIM, "common_original")
    _exact_float_vector(row.common_mirrored, COMMON_DIM, "common_mirrored")

    features = np.asarray(row.temporal_features)
    valid = np.asarray(row.temporal_valid_mask)
    timestamps = np.asarray(row.temporal_timestamps)
    indices = np.asarray(row.temporal_source_frame_indices)
    if features.shape != (4, 32, 95) or features.dtype != np.dtype(np.float32):
        raise ValueError("temporal features must be exact float32 4x32x95")
    if valid.shape != (4, 32) or valid.dtype != np.dtype(bool):
        raise ValueError("temporal valid mask must be exact bool 4x32")
    if timestamps.shape != (4, 32) or timestamps.dtype != np.dtype(np.float64):
        raise ValueError("temporal timestamps must be exact float64 4x32")
    if indices.shape != (4, 32) or indices.dtype != np.dtype(np.int64):
        raise ValueError("temporal source indices must be exact int64 4x32")
    if (
        not np.isfinite(features[valid]).all()
        or not np.isfinite(timestamps).all()
        or np.any(features[~valid] != 0)
        or float(valid.mean()) < MIN_RECORDING_COVERAGE
        or np.any(np.diff(timestamps, axis=1) <= 0)
        or np.any(np.diff(indices, axis=1) <= 0)
        or np.any(indices < 0)
    ):
        raise ValueError("temporal evidence fails finite, support, zero or time QC")
    au_parts = (
        row.au_summary is not None,
        row.au_temporal is not None,
        row.au_temporal_mask is not None,
    )
    if len(set(au_parts)) != 1:
        raise ValueError("AU summary, temporal values and temporal mask are atomic")
    if row.au_summary is not None:
        _exact_float_vector(row.au_summary, AU_DIM, "au_summary")
        temporal_au = np.asarray(row.au_temporal)
        temporal_au_mask = np.asarray(row.au_temporal_mask)
        if (
            temporal_au.shape != (AU_TEMPORAL_SAMPLES, AU_CHANNELS)
            or temporal_au.dtype != np.dtype(np.float32)
            or temporal_au_mask.shape != (AU_TEMPORAL_SAMPLES,)
            or temporal_au_mask.dtype != np.dtype(bool)
            or not temporal_au_mask.any()
            or not np.isfinite(temporal_au[temporal_au_mask]).all()
            or np.any(temporal_au[~temporal_au_mask] != 0)
        ):
            raise ValueError("temporal AU evidence fails shape, support or zero QC")


def build_phenotype_dataset(
    rows: Sequence[RecordingEvidence],
) -> PhenotypeDataset:
    """Aggregate participants while retaining each action recording as one bag item."""
    values = tuple(rows)
    if not values:
        raise ValueError("phenotype dataset requires at least one recording")
    seen_recordings: set[str] = set()
    grouped: dict[str, list[RecordingEvidence]] = {}
    for row in values:
        _validate_recording(row)
        if row.recording_id in seen_recordings:
            raise ValueError("recording evidence cannot be duplicated")
        seen_recordings.add(row.recording_id)
        grouped.setdefault(row.group_id, []).append(row)
    ordered_groups = tuple(sorted(grouped))
    participant_count = len(ordered_groups)

    landmark_original = np.zeros((participant_count, LANDMARK_DIM), dtype=np.float64)
    landmark_mirrored = np.zeros_like(landmark_original)
    common_original = np.zeros(
        (participant_count, MAX_INSTANCES, COMMON_DIM), dtype=np.float64
    )
    common_mirrored = np.zeros_like(common_original)
    instance_mask = np.zeros((participant_count, MAX_INSTANCES), dtype=bool)
    temporal_features = np.zeros(
        (participant_count, MAX_INSTANCES, 4, 32, 95), dtype=np.float32
    )
    temporal_valid = np.zeros(
        (participant_count, MAX_INSTANCES, 4, 32), dtype=bool
    )
    temporal_timestamps = np.zeros(
        (participant_count, MAX_INSTANCES, 4, 32), dtype=np.float64
    )
    temporal_indices = np.full(
        (participant_count, MAX_INSTANCES, 4, 32), -1, dtype=np.int64
    )
    au_instances = np.zeros((participant_count, MAX_INSTANCES, AU_DIM), dtype=np.float64)
    au_mask = np.zeros((participant_count, MAX_INSTANCES), dtype=bool)
    au_temporal = np.zeros(
        (participant_count, MAX_INSTANCES, AU_TEMPORAL_SAMPLES, AU_CHANNELS),
        dtype=np.float32,
    )
    au_temporal_mask = np.zeros(
        (participant_count, MAX_INSTANCES, AU_TEMPORAL_SAMPLES), dtype=bool
    )
    task_codes = np.full((participant_count, MAX_INSTANCES), -1, dtype=np.int8)
    instance_counts = np.zeros(participant_count, dtype=np.int8)
    labels = np.zeros(participant_count, dtype=np.int64)
    sources: list[str] = []
    phenotypes: list[str] = []

    for participant_index, group_id in enumerate(ordered_groups):
        participant_rows = sorted(
            grouped[group_id], key=lambda row: (_TASK_CODE[row.task], row.recording_id)
        )
        if len(participant_rows) > MAX_INSTANCES:
            raise ValueError("participant exceeds the closed nine-recording bag")
        observed_labels = {row.label for row in participant_rows}
        observed_sources = {row.evaluation_source for row in participant_rows}
        observed_phenotypes = {row.phenotype for row in participant_rows}
        scripted_tasks = [
            row.task for row in participant_rows if row.task != "FREE_RECORDING"
        ]
        if len(observed_labels) != 1:
            raise ValueError("participant label changed across recordings")
        if len(observed_sources) != 1:
            raise ValueError("participant crossed evaluation sources")
        if len(observed_phenotypes) != 1:
            raise ValueError("participant phenotype changed across recordings")
        if len(set(scripted_tasks)) != len(scripted_tasks):
            raise ValueError("participant repeated a scripted task recording")
        labels[participant_index] = observed_labels.pop()
        sources.append(observed_sources.pop())
        phenotypes.append(observed_phenotypes.pop())
        landmark_original[participant_index] = np.mean(
            [row.landmark_original for row in participant_rows], axis=0,
            dtype=np.float64,
        )
        landmark_mirrored[participant_index] = np.mean(
            [row.landmark_mirrored for row in participant_rows], axis=0,
            dtype=np.float64,
        )
        instance_counts[participant_index] = len(participant_rows)
        for instance_index, row in enumerate(participant_rows):
            instance_mask[participant_index, instance_index] = True
            common_original[participant_index, instance_index] = row.common_original
            common_mirrored[participant_index, instance_index] = row.common_mirrored
            temporal_features[participant_index, instance_index] = row.temporal_features
            temporal_valid[participant_index, instance_index] = row.temporal_valid_mask
            temporal_timestamps[participant_index, instance_index] = row.temporal_timestamps
            temporal_indices[participant_index, instance_index] = (
                row.temporal_source_frame_indices
            )
            task_codes[participant_index, instance_index] = _TASK_CODE[row.task]
            if row.au_summary is not None:
                au_instances[participant_index, instance_index] = row.au_summary
                au_mask[participant_index, instance_index] = True
                au_temporal[participant_index, instance_index] = row.au_temporal
                au_temporal_mask[participant_index, instance_index] = (
                    row.au_temporal_mask
                )

    return PhenotypeDataset(
        landmark_original=_immutable(landmark_original),
        landmark_mirrored=_immutable(landmark_mirrored),
        common_original=_immutable(common_original),
        common_mirrored=_immutable(common_mirrored),
        instance_mask=_immutable(instance_mask),
        temporal_features=_immutable(temporal_features),
        temporal_valid_mask=_immutable(temporal_valid),
        temporal_timestamps=_immutable(temporal_timestamps),
        temporal_source_frame_indices=_immutable(temporal_indices),
        au_instances=_immutable(au_instances),
        au_mask=_immutable(au_mask),
        au_temporal=_immutable(au_temporal),
        au_temporal_mask=_immutable(au_temporal_mask),
        task_codes=_immutable(task_codes),
        instance_counts=_immutable(instance_counts),
        labels=_immutable(labels),
        group_ids=ordered_groups,
        evaluation_sources=tuple(sources),
        phenotypes=tuple(phenotypes),
    )


__all__ = (
    "AU_DIM",
    "AU_CHANNELS",
    "AU_TEMPORAL_SAMPLES",
    "COMMON_DIM",
    "LANDMARK_DIM",
    "MAX_INSTANCES",
    "PHENOTYPES",
    "TASKS",
    "PhenotypeDataset",
    "RecordingEvidence",
    "build_phenotype_dataset",
)
