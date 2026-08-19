#!/usr/bin/env python3
"""Private full-mesh action cache contract for Universal Router v6.

The production H200 driver supplies authenticated frame indices.  This module
contains the closed cache format and the frame-level original/flip extractor;
it deliberately contains no cohort labels or dataset-dependent inference.
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from src.preprocessing.dense_bilateral_action_v1 import (
    DENSE_POINT_COUNT,
    normalize_dense_landmarks,
)


DENSE_CACHE_SCHEMA = "dense_bilateral_action_mesh_v1"
_MAX_CACHE_BYTES = 128 * 1024 * 1024
_FIELDS = frozenset(
    {
        "schema_version",
        "recording_id",
        "group_id",
        "source_sha256",
        "timing_sha256",
        "face_landmarker_sha256",
        "action_names",
        "source_frame_count",
        "fps",
        "action_frame_indices",
        "baseline_frame_indices",
        "original_actions",
        "mirrored_actions",
        "original_baselines",
        "mirrored_baselines",
        "action_valid",
        "baseline_valid",
    }
)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _immutable(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


@dataclass(frozen=True)
class DenseActionMeshCache:
    recording_id: str
    group_id: str
    source_sha256: str
    timing_sha256: str
    face_landmarker_sha256: str
    action_names: tuple[str, ...]
    source_frame_count: int
    fps: float
    action_frame_indices: np.ndarray
    baseline_frame_indices: np.ndarray
    original_actions: np.ndarray
    mirrored_actions: np.ndarray
    original_baselines: np.ndarray
    mirrored_baselines: np.ndarray
    action_valid: np.ndarray
    baseline_valid: np.ndarray

    @property
    def schema_version(self) -> str:
        return DENSE_CACHE_SCHEMA


def _identifier(value: object, prefix: str, name: str) -> str:
    if type(value) is not str or not value.startswith(prefix) or not _HEX64.fullmatch(
        value[len(prefix) :]
    ):
        raise ValueError(f"{name} must be an opaque {prefix}<sha256> identifier")
    return value


def _digest(value: object, name: str) -> str:
    if type(value) is not str or not _HEX64.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def build_dense_action_cache(
    *,
    recording_id: str,
    group_id: str,
    source_sha256: str,
    timing_sha256: str,
    face_landmarker_sha256: str,
    action_names: tuple[str, ...],
    source_frame_count: int,
    fps: float,
    action_frame_indices: np.ndarray,
    baseline_frame_indices: np.ndarray,
    original_actions: np.ndarray,
    mirrored_actions: np.ndarray,
    original_baselines: np.ndarray,
    mirrored_baselines: np.ndarray,
    action_valid: np.ndarray,
    baseline_valid: np.ndarray,
) -> DenseActionMeshCache:
    recording_id = _identifier(recording_id, "rec_", "recording_id")
    group_id = _identifier(group_id, "grp_", "group_id")
    source_sha256 = _digest(source_sha256, "source_sha256")
    timing_sha256 = _digest(timing_sha256, "timing_sha256")
    face_landmarker_sha256 = _digest(
        face_landmarker_sha256, "face_landmarker_sha256"
    )
    if (
        type(action_names) is not tuple
        or not action_names
        or any(type(name) is not str or not name for name in action_names)
        or len(set(action_names)) != len(action_names)
    ):
        raise ValueError("action_names must be an exact tuple of unique names")
    if (
        isinstance(source_frame_count, (bool, np.bool_))
        or not isinstance(source_frame_count, (int, np.integer))
        or int(source_frame_count) < 1
    ):
        raise ValueError("source_frame_count must be a positive integer")
    source_frame_count = int(source_frame_count)
    if isinstance(fps, (bool, np.bool_)) or not np.isfinite(float(fps)) or float(fps) <= 0:
        raise ValueError("fps must be positive and finite")
    fps = float(fps)
    action_count = len(action_names)

    arrays = {
        "action_frame_indices": (action_frame_indices, 2, np.dtype(np.int64)),
        "baseline_frame_indices": (baseline_frame_indices, 2, np.dtype(np.int64)),
        "original_actions": (original_actions, 4, np.dtype(np.float64)),
        "mirrored_actions": (mirrored_actions, 4, np.dtype(np.float64)),
        "original_baselines": (original_baselines, 4, np.dtype(np.float64)),
        "mirrored_baselines": (mirrored_baselines, 4, np.dtype(np.float64)),
        "action_valid": (action_valid, 2, np.dtype(bool)),
        "baseline_valid": (baseline_valid, 2, np.dtype(bool)),
    }
    for name, (array, ndim, dtype) in arrays.items():
        if type(array) is not np.ndarray or array.ndim != ndim or array.dtype != dtype:
            raise ValueError(f"{name} has a noncanonical array contract")
        if array.shape[0] != action_count:
            raise ValueError(f"{name} action dimension differs from action_names")
    if action_frame_indices.shape != action_valid.shape:
        raise ValueError("action frame indices and mask shapes differ")
    if baseline_frame_indices.shape != baseline_valid.shape:
        raise ValueError("baseline frame indices and mask shapes differ")
    expected_action = action_frame_indices.shape + (DENSE_POINT_COUNT, 3)
    expected_baseline = baseline_frame_indices.shape + (DENSE_POINT_COUNT, 3)
    if original_actions.shape != expected_action or mirrored_actions.shape != expected_action:
        raise ValueError("action meshes differ from their frozen frame grid")
    if (
        original_baselines.shape != expected_baseline
        or mirrored_baselines.shape != expected_baseline
    ):
        raise ValueError("baseline meshes differ from their frozen frame grid")
    for indices in (action_frame_indices, baseline_frame_indices):
        if np.any(indices < 0) or np.any(indices >= source_frame_count):
            raise ValueError("source frame indices fall outside the decoded recording")
        if np.any(np.diff(indices, axis=1) < 0):
            raise ValueError("source frame grids must be nondecreasing")
    if np.any(action_valid.sum(axis=1) < 6):
        raise ValueError("every action requires at least six paired detections")
    if np.any(baseline_valid.sum(axis=1) < 4):
        raise ValueError("every baseline requires at least four paired detections")
    for values, mask, name in (
        (original_actions, action_valid, "original_actions"),
        (mirrored_actions, action_valid, "mirrored_actions"),
        (original_baselines, baseline_valid, "original_baselines"),
        (mirrored_baselines, baseline_valid, "mirrored_baselines"),
    ):
        if not np.isfinite(values[mask]).all():
            raise ValueError(f"valid {name} rows must be finite")
        if not np.isnan(values[~mask]).all():
            raise ValueError(f"invalid {name} rows must remain explicit NaN")

    return DenseActionMeshCache(
        recording_id=recording_id,
        group_id=group_id,
        source_sha256=source_sha256,
        timing_sha256=timing_sha256,
        face_landmarker_sha256=face_landmarker_sha256,
        action_names=action_names,
        source_frame_count=source_frame_count,
        fps=fps,
        action_frame_indices=_immutable(action_frame_indices),
        baseline_frame_indices=_immutable(baseline_frame_indices),
        original_actions=_immutable(original_actions),
        mirrored_actions=_immutable(mirrored_actions),
        original_baselines=_immutable(original_baselines),
        mirrored_baselines=_immutable(mirrored_baselines),
        action_valid=_immutable(action_valid),
        baseline_valid=_immutable(baseline_valid),
    )


def serialize_dense_action_cache(cache: DenseActionMeshCache) -> bytes:
    if type(cache) is not DenseActionMeshCache:
        raise ValueError("a validated DenseActionMeshCache is required")
    # Revalidation makes a forged dataclass fail closed.
    cache = build_dense_action_cache(**cache.__dict__)
    output = io.BytesIO()
    np.savez_compressed(
        output,
        schema_version=np.asarray(DENSE_CACHE_SCHEMA),
        recording_id=np.asarray(cache.recording_id),
        group_id=np.asarray(cache.group_id),
        source_sha256=np.asarray(cache.source_sha256),
        timing_sha256=np.asarray(cache.timing_sha256),
        face_landmarker_sha256=np.asarray(cache.face_landmarker_sha256),
        action_names=np.asarray(cache.action_names),
        source_frame_count=np.asarray(cache.source_frame_count, dtype=np.int64),
        fps=np.asarray(cache.fps, dtype=np.float64),
        action_frame_indices=cache.action_frame_indices,
        baseline_frame_indices=cache.baseline_frame_indices,
        original_actions=cache.original_actions,
        mirrored_actions=cache.mirrored_actions,
        original_baselines=cache.original_baselines,
        mirrored_baselines=cache.mirrored_baselines,
        action_valid=cache.action_valid,
        baseline_valid=cache.baseline_valid,
    )
    return output.getvalue()


def _scalar(array: np.ndarray, name: str) -> object:
    value = np.asarray(array)
    if value.shape != ():
        raise ValueError(f"{name} must be scalar")
    return value.item()


def load_dense_action_cache_bytes(payload: bytes) -> DenseActionMeshCache:
    if type(payload) is not bytes or not payload or len(payload) > _MAX_CACHE_BYTES:
        raise ValueError("dense cache must be bounded exact bytes")
    source = io.BytesIO(payload)
    expected_members = {f"{name}.npy" for name in _FIELDS}
    try:
        with zipfile.ZipFile(source, "r") as archive:
            members = [entry.filename for entry in archive.infolist()]
        if (
            len(members) != len(expected_members)
            or len(set(members)) != len(members)
            or set(members) != expected_members
        ):
            raise ValueError("dense cache has duplicate or noncanonical members")
        source.seek(0)
        with np.load(source, allow_pickle=False) as saved:
            fields = list(saved.files)
            if (
                len(fields) != len(_FIELDS)
                or len(set(fields)) != len(fields)
                or set(fields) != _FIELDS
            ):
                raise ValueError("dense cache fields differ from its closed schema")
            values = {name: np.asarray(saved[name]) for name in fields}
    except ValueError:
        raise
    except (EOFError, KeyError, OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ValueError(f"cannot parse dense cache: {exc}") from exc
    if _scalar(values["schema_version"], "schema_version") != DENSE_CACHE_SCHEMA:
        raise ValueError("dense cache schema differs")
    names_array = values["action_names"]
    if names_array.ndim != 1 or names_array.dtype.kind not in {"U", "S"}:
        raise ValueError("action_names serialization is noncanonical")
    return build_dense_action_cache(
        recording_id=str(_scalar(values["recording_id"], "recording_id")),
        group_id=str(_scalar(values["group_id"], "group_id")),
        source_sha256=str(_scalar(values["source_sha256"], "source_sha256")),
        timing_sha256=str(_scalar(values["timing_sha256"], "timing_sha256")),
        face_landmarker_sha256=str(
            _scalar(values["face_landmarker_sha256"], "face_landmarker_sha256")
        ),
        action_names=tuple(str(value) for value in names_array.tolist()),
        source_frame_count=int(
            _scalar(values["source_frame_count"], "source_frame_count")
        ),
        fps=float(_scalar(values["fps"], "fps")),
        action_frame_indices=values["action_frame_indices"],
        baseline_frame_indices=values["baseline_frame_indices"],
        original_actions=values["original_actions"],
        mirrored_actions=values["mirrored_actions"],
        original_baselines=values["original_baselines"],
        mirrored_baselines=values["mirrored_baselines"],
        action_valid=values["action_valid"],
        baseline_valid=values["baseline_valid"],
    )


def publish_dense_action_cache(path: Path, payload: bytes) -> str:
    if not isinstance(path, Path) or type(payload) is not bytes or not payload:
        raise ValueError("publication requires an exact Path and nonempty bytes")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        written = 0
        while written < len(payload):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("short write while publishing dense cache")
            written += count
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        try:
            path.unlink()
        except OSError:
            pass
        raise
    else:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def extract_normalized_pair(
    frame_bgr: np.ndarray,
    detect_mesh: Callable[[np.ndarray], np.ndarray | None],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Run the detector on an actual frame and actual horizontal image flip."""
    if (
        type(frame_bgr) is not np.ndarray
        or frame_bgr.dtype != np.dtype(np.uint8)
        or frame_bgr.ndim != 3
        or frame_bgr.shape[2] != 3
        or frame_bgr.shape[0] < 2
        or frame_bgr.shape[1] < 2
    ):
        raise ValueError("frame_bgr must be an exact uint8 HxWx3 image")
    if not callable(detect_mesh):
        raise ValueError("detect_mesh must be callable")
    rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
    flipped_rgb = np.ascontiguousarray(rgb[:, ::-1])
    outputs: list[np.ndarray | None] = []
    for image in (rgb, flipped_rgb):
        mesh = detect_mesh(image)
        if mesh is None:
            outputs.append(None)
            continue
        outputs.append(
            normalize_dense_landmarks(
                mesh, image_width=image.shape[1], image_height=image.shape[0]
            )
        )
    return outputs[0], outputs[1]


__all__ = (
    "DENSE_CACHE_SCHEMA",
    "DenseActionMeshCache",
    "build_dense_action_cache",
    "extract_normalized_pair",
    "load_dense_action_cache_bytes",
    "publish_dense_action_cache",
    "serialize_dense_action_cache",
)
