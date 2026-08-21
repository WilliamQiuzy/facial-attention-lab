"""Closed full-frame Py-Feat AU cache for all nine NeuroFace actions."""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass

import numpy as np

from src.datasets.neuroface_au_v1 import (
    AU_NAMES,
    PAPER_AU_MODEL,
    PAPER_PYFEAT_VERSION,
    build_au_recording,
    publish_au_cache,
)


SCHEMA_VERSION = "neuroface_pyfeat_xgb_au_temporal_sample_v2"
ALL_TASKS = (
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
SUMMARY_STATISTICS = ("mean", "min", "max", "std", "var")
TEMPORAL_SAMPLES = 64
_FIELDS = frozenset({
    "schema_version", "pyfeat_version", "au_model", "au_names",
    "recording_id", "group_id", "task", "source_sha256",
    "source_frame_count", "fps", "frame_indices", "timestamps",
    "sampling_stride", "timestamp_unit", "au_values", "valid_mask", "selected_face_count",
    "selected_face_score",
})


def _immutable(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


@dataclass(frozen=True)
class FullNeuroFaceAURecording:
    recording_id: str
    group_id: str
    task: str
    source_sha256: str
    source_frame_count: int
    fps: float
    sampling_stride: int
    frame_indices: np.ndarray
    timestamps: np.ndarray
    au_values: np.ndarray
    valid_mask: np.ndarray
    selected_face_count: np.ndarray
    selected_face_score: np.ndarray

    @property
    def coverage(self) -> float:
        return float(self.valid_mask.mean())


@dataclass(frozen=True)
class FullAUSummary:
    feature_names: tuple[str, ...]
    values: np.ndarray
    valid_frames: int
    total_frames: int


def build_full_au_recording(
    *,
    recording_id: str,
    group_id: str,
    task: str,
    source_sha256: str,
    source_frame_count: int,
    fps: float,
    sampling_stride: int,
    frame_indices: np.ndarray,
    timestamps: np.ndarray,
    au_values: np.ndarray,
    valid_mask: np.ndarray,
    selected_face_count: np.ndarray,
    selected_face_score: np.ndarray,
) -> FullNeuroFaceAURecording:
    """Validate a full-cohort row while reusing the frozen frame-level QC."""
    if task not in ALL_TASKS:
        raise ValueError("task differs from the frozen nine-action registry")
    if (
        isinstance(sampling_stride, (bool, np.bool_))
        or not isinstance(sampling_stride, (int, np.integer))
        or int(sampling_stride) < 1 or int(sampling_stride) > 60
    ):
        raise ValueError("sampling_stride must be a bounded positive integer")
    sampling_stride = int(sampling_stride)
    source_indices = np.asarray(frame_indices)
    if (
        isinstance(source_frame_count, (bool, np.bool_))
        or not isinstance(source_frame_count, (int, np.integer))
        or int(source_frame_count) <= 0
        or source_indices.dtype != np.dtype(np.int64)
        or source_indices.ndim != 1
        or not np.array_equal(
            source_indices,
            np.arange(0, int(source_frame_count), sampling_stride, dtype=np.int64),
        )
    ):
        raise ValueError("sampled rows must follow the frozen regular source clock")
    processed = int(source_indices.size)
    validated = build_au_recording(
        recording_id=recording_id,
        group_id=group_id,
        task="NSM_SPREAD",
        source_sha256=source_sha256,
        source_frame_count=processed,
        fps=float(fps) / sampling_stride,
        frame_indices=np.arange(processed, dtype=np.int64),
        timestamps=timestamps,
        au_values=au_values,
        valid_mask=valid_mask,
        selected_face_count=selected_face_count,
        selected_face_score=selected_face_score,
    )
    return FullNeuroFaceAURecording(
        recording_id=validated.recording_id,
        group_id=validated.group_id,
        task=task,
        source_sha256=validated.source_sha256,
        source_frame_count=int(source_frame_count),
        fps=float(fps),
        sampling_stride=sampling_stride,
        frame_indices=_immutable(source_indices),
        timestamps=validated.timestamps,
        au_values=validated.au_values,
        valid_mask=validated.valid_mask,
        selected_face_count=validated.selected_face_count,
        selected_face_score=validated.selected_face_score,
    )


def summarize_full_au_recording(
    recording: FullNeuroFaceAURecording,
) -> FullAUSummary:
    if not isinstance(recording, FullNeuroFaceAURecording):
        raise ValueError("a validated full-cohort AU recording is required")
    observed = recording.au_values[recording.valid_mask].astype(np.float64)
    values = np.concatenate((
        observed.mean(axis=0), observed.min(axis=0), observed.max(axis=0),
        observed.std(axis=0, ddof=0), observed.var(axis=0, ddof=0),
    )).astype(np.float64, copy=False)
    names = tuple(
        f"{statistic}_{au_name}"
        for statistic in SUMMARY_STATISTICS
        for au_name in AU_NAMES
    )
    return FullAUSummary(
        feature_names=names,
        values=_immutable(values),
        valid_frames=int(recording.valid_mask.sum()),
        total_frames=recording.source_frame_count,
    )


def temporal_full_au_view(
    recording: FullNeuroFaceAURecording,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic fixed-length AU sequence with explicit support."""
    if not isinstance(recording, FullNeuroFaceAURecording):
        raise ValueError("a validated full-cohort AU recording is required")
    observed_rows = recording.au_values.shape[0]
    retained = min(TEMPORAL_SAMPLES, observed_rows)
    if retained == 1:
        positions = np.zeros(1, dtype=np.int64)
    else:
        positions = np.rint(
            np.linspace(0, observed_rows - 1, retained, dtype=np.float64)
        ).astype(np.int64)
    if len(np.unique(positions)) != retained:
        raise AssertionError("fixed AU temporal sampling produced duplicate rows")
    values = np.zeros((TEMPORAL_SAMPLES, len(AU_NAMES)), dtype=np.float32)
    mask = np.zeros(TEMPORAL_SAMPLES, dtype=bool)
    valid = recording.valid_mask[positions]
    values[:retained][valid] = recording.au_values[positions][valid]
    mask[:retained] = valid
    return _immutable(values), _immutable(mask)


def serialize_full_au_recording(recording: FullNeuroFaceAURecording) -> bytes:
    if not isinstance(recording, FullNeuroFaceAURecording):
        raise ValueError("a validated full-cohort AU recording is required")
    payload = io.BytesIO()
    np.savez_compressed(
        payload,
        schema_version=np.asarray(SCHEMA_VERSION),
        pyfeat_version=np.asarray(PAPER_PYFEAT_VERSION),
        au_model=np.asarray(PAPER_AU_MODEL),
        au_names=np.asarray(AU_NAMES),
        recording_id=np.asarray(recording.recording_id),
        group_id=np.asarray(recording.group_id),
        task=np.asarray(recording.task),
        source_sha256=np.asarray(recording.source_sha256),
        source_frame_count=np.asarray(recording.source_frame_count, dtype=np.int64),
        fps=np.asarray(recording.fps, dtype=np.float64),
        sampling_stride=np.asarray(recording.sampling_stride, dtype=np.int64),
        frame_indices=recording.frame_indices,
        timestamps=recording.timestamps,
        timestamp_unit=np.asarray("seconds"),
        au_values=recording.au_values,
        valid_mask=recording.valid_mask,
        selected_face_count=recording.selected_face_count,
        selected_face_score=recording.selected_face_score,
    )
    return payload.getvalue()


def _scalar(array: np.ndarray, name: str) -> object:
    value = np.asarray(array)
    if value.shape != ():
        raise ValueError(f"{name} must be a scalar")
    return value.item()


def load_full_au_recording_bytes(payload: bytes) -> FullNeuroFaceAURecording:
    if type(payload) is not bytes or not payload or len(payload) > 512 * 1024 * 1024:
        raise ValueError("full AU cache must be bounded exact bytes")
    source = io.BytesIO(payload)
    expected_members = {f"{field}.npy" for field in _FIELDS}
    try:
        with zipfile.ZipFile(source, "r") as archive:
            members = [info.filename for info in archive.infolist()]
        if (
            len(members) != len(expected_members)
            or len(set(members)) != len(members)
            or set(members) != expected_members
        ):
            raise ValueError("full AU cache has duplicate or noncanonical members")
        source.seek(0)
        with np.load(source, allow_pickle=False) as saved:
            names = list(saved.files)
            if (
                len(names) != len(_FIELDS)
                or len(set(names)) != len(names)
                or set(names) != _FIELDS
            ):
                raise ValueError("full AU cache fields differ from the closed schema")
            fields = {name: np.asarray(saved[name]) for name in _FIELDS}
    except ValueError:
        raise
    except (EOFError, KeyError, OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ValueError(f"cannot parse full AU cache: {exc}") from exc
    if _scalar(fields["schema_version"], "schema_version") != SCHEMA_VERSION:
        raise ValueError("full AU cache schema version differs")
    if _scalar(fields["pyfeat_version"], "pyfeat_version") != PAPER_PYFEAT_VERSION:
        raise ValueError("full AU cache Py-Feat version differs")
    if _scalar(fields["au_model"], "au_model") != PAPER_AU_MODEL:
        raise ValueError("full AU detector differs from the freeze")
    if _scalar(fields["timestamp_unit"], "timestamp_unit") != "seconds":
        raise ValueError("full AU timestamps must be seconds")
    au_names = tuple(str(value) for value in np.asarray(fields["au_names"]).tolist())
    if au_names != AU_NAMES:
        raise ValueError("full AU name order differs from the freeze")
    return build_full_au_recording(
        recording_id=str(_scalar(fields["recording_id"], "recording_id")),
        group_id=str(_scalar(fields["group_id"], "group_id")),
        task=str(_scalar(fields["task"], "task")),
        source_sha256=str(_scalar(fields["source_sha256"], "source_sha256")),
        source_frame_count=int(_scalar(fields["source_frame_count"], "source_frame_count")),
        fps=float(_scalar(fields["fps"], "fps")),
        sampling_stride=int(_scalar(fields["sampling_stride"], "sampling_stride")),
        frame_indices=fields["frame_indices"],
        timestamps=fields["timestamps"],
        au_values=fields["au_values"],
        valid_mask=fields["valid_mask"],
        selected_face_count=fields["selected_face_count"],
        selected_face_score=fields["selected_face_score"],
    )


__all__ = (
    "ALL_TASKS",
    "AU_NAMES",
    "FullAUSummary",
    "FullNeuroFaceAURecording",
    "SCHEMA_VERSION",
    "TEMPORAL_SAMPLES",
    "build_full_au_recording",
    "load_full_au_recording_bytes",
    "publish_au_cache",
    "serialize_full_au_recording",
    "summarize_full_au_recording",
    "temporal_full_au_view",
)
