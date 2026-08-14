"""Validated full-frame Py-Feat AU evidence for the NeuroFace ALS benchmark.

This module owns only the immutable cache and summary contracts. It does not
choose participants, fit a classifier, or access any other dataset.
"""
from __future__ import annotations

import hashlib
import io
import math
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "neuroface_pyfeat_xgb_au_fullframe_v1"
PAPER_PYFEAT_VERSION = "0.6.2"
PAPER_AU_MODEL = "xgb"
PAPER_TASKS = ("NSM_KISS", "NSM_OPEN", "NSM_SPREAD")
AU_NAMES = (
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10",
    "AU11", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24",
    "AU25", "AU26", "AU28", "AU43",
)
SUMMARY_STATISTICS = ("mean", "min", "max", "std", "var")

_RECORDING_ID = re.compile(r"^rec_[0-9a-f]{64}$")
_GROUP_ID = re.compile(r"^grp_[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FIELDS = frozenset({
    "schema_version", "pyfeat_version", "au_model", "au_names",
    "recording_id", "group_id", "task", "source_sha256",
    "source_frame_count", "fps", "frame_indices", "timestamps",
    "timestamp_unit", "au_values", "valid_mask", "selected_face_count",
    "selected_face_score",
})


def _immutable(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    frozen = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype)
    return frozen.reshape(contiguous.shape)


@dataclass(frozen=True)
class NeuroFaceAURecording:
    recording_id: str
    group_id: str
    task: str
    source_sha256: str
    source_frame_count: int
    fps: float
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
class AUSummary:
    feature_names: tuple[str, ...]
    values: np.ndarray
    valid_frames: int
    total_frames: int


def _canonical_text(value: object, *, name: str, pattern=None) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty canonical string")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise ValueError(f"{name} does not match its frozen format")
    return value


def build_au_recording(
    *,
    recording_id: str,
    group_id: str,
    task: str,
    source_sha256: str,
    source_frame_count: int,
    fps: float,
    frame_indices: np.ndarray,
    timestamps: np.ndarray,
    au_values: np.ndarray,
    valid_mask: np.ndarray,
    selected_face_count: np.ndarray,
    selected_face_score: np.ndarray,
) -> NeuroFaceAURecording:
    """Validate and freeze one complete, frame-indexed AU recording."""
    recording_id = _canonical_text(
        recording_id, name="recording_id", pattern=_RECORDING_ID
    )
    group_id = _canonical_text(group_id, name="group_id", pattern=_GROUP_ID)
    source_sha256 = _canonical_text(
        source_sha256, name="source_sha256", pattern=_SHA256
    )
    if task not in PAPER_TASKS:
        raise ValueError(f"task must be one of {PAPER_TASKS}")
    if (isinstance(source_frame_count, (bool, np.bool_))
            or not isinstance(source_frame_count, (int, np.integer))
            or int(source_frame_count) <= 0):
        raise ValueError("source_frame_count must be a positive integer")
    source_frame_count = int(source_frame_count)
    if isinstance(fps, (bool, np.bool_)) or not isinstance(
        fps, (int, float, np.integer, np.floating)
    ) or not math.isfinite(float(fps)) or float(fps) <= 0:
        raise ValueError("fps must be a finite positive number")
    fps = float(fps)

    frame_indices = np.asarray(frame_indices)
    timestamps = np.asarray(timestamps)
    au_values = np.asarray(au_values)
    valid_mask = np.asarray(valid_mask)
    selected_face_count = np.asarray(selected_face_count)
    selected_face_score = np.asarray(selected_face_score)
    length = source_frame_count
    if frame_indices.dtype != np.dtype(np.int64) or frame_indices.shape != (length,):
        raise ValueError("frame_indices must be int64 with one row per source frame")
    if not np.array_equal(frame_indices, np.arange(length, dtype=np.int64)):
        raise ValueError("frame_indices must cover the complete source in order")
    if timestamps.dtype != np.dtype(np.float64) or timestamps.shape != (length,):
        raise ValueError("timestamps must be float64 with one row per source frame")
    if not np.isfinite(timestamps).all() or not np.all(timestamps[1:] > timestamps[:-1]):
        raise ValueError("timestamps must be finite and strictly increasing")
    if not np.allclose(timestamps, frame_indices.astype(np.float64) / fps,
                       rtol=0.0, atol=1e-9):
        raise ValueError("timestamps must equal frame_indices divided by fps")
    if au_values.dtype != np.dtype(np.float32) or au_values.shape != (length, len(AU_NAMES)):
        raise ValueError("au_values must be float32 with shape (frames, 20)")
    if valid_mask.dtype != np.dtype(bool) or valid_mask.shape != (length,):
        raise ValueError("valid_mask must be bool with one value per source frame")
    if selected_face_count.dtype != np.dtype(np.int16) or selected_face_count.shape != (length,):
        raise ValueError("selected_face_count must be int16 with one value per frame")
    if selected_face_score.dtype != np.dtype(np.float32) or selected_face_score.shape != (length,):
        raise ValueError("selected_face_score must be float32 with one value per frame")
    if not valid_mask.any():
        raise ValueError("at least one valid AU frame is required")
    if not np.isfinite(au_values[valid_mask]).all():
        raise ValueError("valid AU values must be finite")
    if np.any(au_values[~valid_mask] != 0):
        raise ValueError("invalid AU rows must be canonical zero")
    if np.any(selected_face_count[valid_mask] < 1) or np.any(selected_face_count[~valid_mask] != 0):
        raise ValueError("face counts must agree with AU validity")
    if (not np.isfinite(selected_face_score).all()
            or np.any(selected_face_score[valid_mask] <= 0)
            or np.any(selected_face_score[valid_mask] > 1)
            or np.any(selected_face_score[~valid_mask] != 0)):
        raise ValueError("face scores must be probabilities consistent with validity")

    return NeuroFaceAURecording(
        recording_id=recording_id,
        group_id=group_id,
        task=task,
        source_sha256=source_sha256,
        source_frame_count=source_frame_count,
        fps=fps,
        frame_indices=_immutable(frame_indices),
        timestamps=_immutable(timestamps),
        au_values=_immutable(au_values),
        valid_mask=_immutable(valid_mask),
        selected_face_count=_immutable(selected_face_count),
        selected_face_score=_immutable(selected_face_score),
    )


def summarize_au_recording(recording: NeuroFaceAURecording) -> AUSummary:
    if not isinstance(recording, NeuroFaceAURecording):
        raise ValueError("a validated NeuroFaceAURecording is required")
    observed = recording.au_values[recording.valid_mask].astype(np.float64)
    blocks = (
        observed.mean(axis=0),
        observed.min(axis=0),
        observed.max(axis=0),
        observed.std(axis=0, ddof=0),
        observed.var(axis=0, ddof=0),
    )
    values = np.concatenate(blocks).astype(np.float64, copy=False)
    names = tuple(
        f"{statistic}_{au_name}"
        for statistic in SUMMARY_STATISTICS
        for au_name in AU_NAMES
    )
    return AUSummary(
        feature_names=names,
        values=_immutable(values),
        valid_frames=int(recording.valid_mask.sum()),
        total_frames=recording.source_frame_count,
    )


def serialize_au_recording(recording: NeuroFaceAURecording) -> bytes:
    if not isinstance(recording, NeuroFaceAURecording):
        raise ValueError("a validated NeuroFaceAURecording is required")
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


def load_au_recording_bytes(payload: bytes) -> NeuroFaceAURecording:
    if type(payload) is not bytes or not payload or len(payload) > 512 * 1024 * 1024:
        raise ValueError("AU cache must be exact nonempty bytes no larger than 512 MiB")
    source = io.BytesIO(payload)
    expected_members = {f"{field}.npy" for field in _FIELDS}
    try:
        with zipfile.ZipFile(source, "r") as archive:
            members = [info.filename for info in archive.infolist()]
        if (len(members) != len(expected_members)
                or len(set(members)) != len(members)
                or set(members) != expected_members):
            raise ValueError("AU cache has duplicate or noncanonical NPZ members")
        source.seek(0)
        with np.load(source, allow_pickle=False) as saved:
            names = list(saved.files)
            if (len(names) != len(_FIELDS) or len(set(names)) != len(names)
                    or set(names) != _FIELDS):
                raise ValueError("AU cache fields differ from the closed schema")
            fields = {name: np.asarray(saved[name]) for name in _FIELDS}
    except ValueError:
        raise
    except (EOFError, KeyError, OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ValueError(f"cannot parse AU cache: {exc}") from exc

    if _scalar(fields["schema_version"], "schema_version") != SCHEMA_VERSION:
        raise ValueError("AU cache schema version differs")
    if _scalar(fields["pyfeat_version"], "pyfeat_version") != PAPER_PYFEAT_VERSION:
        raise ValueError("Py-Feat version differs from the freeze")
    if _scalar(fields["au_model"], "au_model") != PAPER_AU_MODEL:
        raise ValueError("AU detector differs from the freeze")
    if _scalar(fields["timestamp_unit"], "timestamp_unit") != "seconds":
        raise ValueError("timestamps must be expressed in seconds")
    au_names = tuple(str(value) for value in np.asarray(fields["au_names"]).tolist())
    if au_names != AU_NAMES:
        raise ValueError("AU name order differs from the freeze")
    return build_au_recording(
        recording_id=str(_scalar(fields["recording_id"], "recording_id")),
        group_id=str(_scalar(fields["group_id"], "group_id")),
        task=str(_scalar(fields["task"], "task")),
        source_sha256=str(_scalar(fields["source_sha256"], "source_sha256")),
        source_frame_count=int(_scalar(fields["source_frame_count"], "source_frame_count")),
        fps=float(_scalar(fields["fps"], "fps")),
        frame_indices=fields["frame_indices"],
        timestamps=fields["timestamps"],
        au_values=fields["au_values"],
        valid_mask=fields["valid_mask"],
        selected_face_count=fields["selected_face_count"],
        selected_face_score=fields["selected_face_score"],
    )


def publish_au_cache(target: Path, payload: bytes) -> str:
    if not isinstance(target, Path) or type(payload) is not bytes or not payload:
        raise ValueError("publication requires a Path and exact nonempty bytes")
    parent = target.parent
    if not parent.is_dir():
        raise ValueError("cache parent directory must already exist")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite AU cache {target}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            written = handle.write(payload)
            if written != len(payload):
                raise OSError("short AU cache write")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "AU_NAMES", "AUSummary", "NeuroFaceAURecording", "PAPER_AU_MODEL",
    "PAPER_PYFEAT_VERSION", "PAPER_TASKS", "SCHEMA_VERSION",
    "build_au_recording", "load_au_recording_bytes", "publish_au_cache",
    "serialize_au_recording", "summarize_au_recording",
]
