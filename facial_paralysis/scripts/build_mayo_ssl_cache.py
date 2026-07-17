"""Build a transactional, deidentified Mayo development-only SSL cache.

The source collection remains read-only.  MediaPipe recordings are stored as
compact source-rate clinical23_v2 trajectories plus exact-source 30-Hz views; the
ARKit-only 52-blendshape trajectories remain a separate modality.  Generated
metadata uses opaque HMAC identifiers and never serializes a session name or
filesystem location.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import ctypes
import csv
import errno
import fcntl
import hashlib
import hmac
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import secrets
import shutil
import stat
import sys
import tempfile
import zipfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field as dataclass_field
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PINNED_MEDIAPIPE_PYTHON = Path(
    "/Users/williamqiu/.cache/facial-paralysis/mediapipe-py310/bin/python"
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dynamic_landmark import (  # noqa: E402
    DYNAMIC_FEATURE_NAMES,
    DYNAMIC_FEATURE_SCHEMA,
)
from src.preprocessing.action_bundle import MediaPipeFeatureExtractor  # noqa: E402
from src.preprocessing.clinical_landmarks import (  # noqa: E402
    CLINICAL_SIDE_CONVENTION,
)


VIDEO_EXTENSIONS = frozenset({".mov", ".mp4", ".m4v"})
ARKIT_TIMECODE_FPS = 60.0
TARGET_SSL_FPS = 30.0
SHORT_QC_MAX_SECONDS = 2.0
MEDIAPIPE_CACHE_SCHEMA = "mayo_mediapipe_clinical23_ssl_v2"
ARKIT_CACHE_SCHEMA = "mayo_arkit_blendshapes_ssl_v1"
COLLECTION_SCHEMA = "mayo_development_ssl_collection_v1"
EXPOSURE_SCHEMA = "mayo_development_exposure_v1"
TRANSFORM_NORMALIZATION = "clinical23_v2_roll_level_midline_center_interocular_scale"
VIDEO_PRODUCER_PROTOCOL = "mediapipe_face_landmarker_running_mode_video_v1"
VIDEO_ADAPTER_VERSION = "mayo_clinical23_same_detection_adapter_v1"
COLLECTION_DATASET = "Mayo_development_only_unlabeled"
EXPOSURE_DATASET = "Mayo_current_cohort"
UNKNOWN_IDENTITY = "unknown_patient_identity"
RECORDING_HELD_OUT = "recording_held_out"
EXPOSURE_POLICY = (
    "permanently development-only after method development or SSL exposure; "
    "future independent HB evidence requires new people"
)
EXISTING_EXPORT_FILES = (
    "done.json",
    "landmarks.csv",
    "blendshapes_wide.csv",
    "transform_matrices.npy",
)
OPENCV_DISTRIBUTIONS = (
    "opencv-python",
    "opencv-contrib-python",
    "opencv-python-headless",
    "opencv-contrib-python-headless",
)

FROZEN_INVENTORY: dict[str, int] = {
    "total_sessions": 65,
    "video_bearing_sessions": 50,
    "without_video_sessions": 15,
    "exact_duplicate_copies_excluded": 1,
    "short_qc_clips_excluded": 1,
    "long_unique_videos": 48,
    "existing_complete_v2_exports": 13,
    "remaining_long_videos": 35,
    "remaining_long_video_frames": 221_121,
    "arkit_only_sessions": 7,
    "arkit_trajectories": 8,
    "arkit_rows": 58_054,
    "arkit_timecode_gaps": 24,
    "metadata_only_sessions": 8,
}

ARKIT_BLENDSHAPE_NAMES: tuple[str, ...] = (
    "EyeBlinkLeft", "EyeLookDownLeft", "EyeLookInLeft", "EyeLookOutLeft",
    "EyeLookUpLeft", "EyeSquintLeft", "EyeWideLeft", "EyeBlinkRight",
    "EyeLookDownRight", "EyeLookInRight", "EyeLookOutRight", "EyeLookUpRight",
    "EyeSquintRight", "EyeWideRight", "JawForward", "JawRight", "JawLeft",
    "JawOpen", "MouthClose", "MouthFunnel", "MouthPucker", "MouthRight",
    "MouthLeft", "MouthSmileLeft", "MouthSmileRight", "MouthFrownLeft",
    "MouthFrownRight", "MouthDimpleLeft", "MouthDimpleRight", "MouthStretchLeft",
    "MouthStretchRight", "MouthRollLower", "MouthRollUpper", "MouthShrugLower",
    "MouthShrugUpper", "MouthPressLeft", "MouthPressRight", "MouthLowerDownLeft",
    "MouthLowerDownRight", "MouthUpperUpLeft", "MouthUpperUpRight", "BrowDownLeft",
    "BrowDownRight", "BrowInnerUp", "BrowOuterUpLeft", "BrowOuterUpRight",
    "CheekPuff", "CheekSquintLeft", "CheekSquintRight", "NoseSneerLeft",
    "NoseSneerRight", "TongueOut",
)
ARKIT_ROTATION_NAMES: tuple[str, ...] = (
    "HeadYaw", "HeadPitch", "HeadRoll", "LeftEyeYaw", "LeftEyePitch",
    "LeftEyeRoll", "RightEyeYaw", "RightEyePitch", "RightEyeRoll",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RECORDING_ID = re.compile(r"^rec_[0-9a-f]{64}$")
_GROUP_ID = re.compile(r"^grp_[0-9a-f]{64}$")
_FINGERPRINT = re.compile(r"^fp_[0-9a-f]{64}$")
_INSTANCE_ID = re.compile(r"^inst_[0-9a-f]{64}$")
_INTEGRITY_ID = re.compile(r"^(?:src|cache|agg)_[0-9a-f]{64}$")
_RAW_MAYO_NAME = re.compile(r"(?:faces|myslate)[_ ]*\d+", re.IGNORECASE)
_ARKIT_TIMECODE = re.compile(
    r"^(?P<hour>(?:[01]\d|2[0-3])):(?P<minute>[0-5]\d):"
    r"(?P<second>[0-5]\d):(?P<frame>[0-5]\d)\.(?P<subframe>\d{3})$"
)
_HELD_OUTPUT_LOCKS: set[Path] = set()

_MEDIAPIPE_CACHE_FIELDS = frozenset({
    "features_source_rate", "valid_mask_source_rate", "timestamps_source_rate",
    "source_frame_indices_source_rate", "facial_transforms_source_rate",
    "facial_transform_mask_source_rate", "features_30hz", "valid_mask_30hz",
    "timestamps_30hz", "source_frame_indices_30hz", "target_frame_indices_30hz",
    "contiguous_from_previous_30hz", "facial_transforms_30hz",
    "facial_transform_mask_30hz", "feature_schema", "feature_names",
    "side_convention", "capture_mirrored", "normalization_transform",
    "facial_transform_source", "timestamp_unit", "timestamp_source", "source_fps",
    "producer_protocol", "producer_adapter_version", "recording_id", "group_id",
    "source_integrity_id", "source_fingerprint", "cache_schema",
    "development_only", "patient_identity", "split_unit",
})
_ARKIT_CACHE_FIELDS = frozenset({
    "features_60hz", "valid_mask_60hz", "timestamps_60hz",
    "source_frame_indices_60hz", "features_30hz", "valid_mask_30hz",
    "timestamps_30hz", "source_frame_indices_30hz", "target_frame_indices_30hz",
    "contiguous_from_previous_30hz", "feature_schema", "feature_names",
    "timestamp_unit", "timestamp_source", "recording_id", "group_id",
    "source_integrity_id", "source_fingerprint", "cache_schema",
    "development_only", "patient_identity", "split_unit",
})
_MAX_MAYO_CACHE_RAW_BYTES = 256 * 1024 * 1024
_MAX_MAYO_NPZ_COMPRESSED_BYTES = 256 * 1024 * 1024
_MAX_MAYO_NPZ_EXPANDED_BYTES = 512 * 1024 * 1024
_MAX_MAYO_CACHE_ROWS = 1_000_000
_MAX_EXACT_PRIVATE_TREE_REGULAR_BYTES = 128 * 1024 * 1024
_MAX_NPY_HEADER_BYTES = 4096
_MAX_MAYO_NPZ_CENTRAL_RECORD_BYTES = 1024
_MAX_MAYO_MANIFEST_BYTES = 4 * 1024 * 1024


class InventoryDriftError(ValueError):
    """The live Mayo source tree differs from the frozen inventory."""


@dataclass(frozen=True)
class VideoMetadata:
    frame_count: int
    fps: float
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps


@dataclass(frozen=True)
class ARKitInspection:
    row_count: int
    feature_names: tuple[str, ...]
    missing_source_frames: int


@dataclass(frozen=True)
class VideoAsset:
    session_path: Path
    path: Path
    metadata: VideoMetadata
    source_sha256: str
    export_dir: Path | None


@dataclass(frozen=True)
class ARKitAsset:
    session_path: Path
    path: Path
    row_count: int
    feature_names: tuple[str, ...]
    source_sha256: str
    missing_source_frames: int


@dataclass(frozen=True)
class MayoInventory:
    data_root: Path
    export_root: Path
    counts: dict[str, int]
    video_instances: tuple[VideoAsset, ...]
    long_unique_videos: tuple[VideoAsset, ...]
    existing_export_videos: tuple[VideoAsset, ...]
    pending_videos: tuple[VideoAsset, ...]
    duplicate_videos: tuple[VideoAsset, ...]
    short_videos: tuple[VideoAsset, ...]
    arkit_sessions: tuple[Path, ...]
    arkit_trajectories: tuple[ARKitAsset, ...]
    metadata_only_sessions: tuple[Path, ...]


@dataclass(frozen=True)
class MayoMediaSequence:
    features: np.ndarray
    valid_mask: np.ndarray
    timestamps: np.ndarray
    source_frame_indices: np.ndarray
    facial_transforms: np.ndarray
    facial_transform_mask: np.ndarray
    transform_source: str
    source_fps: float = 60.0
    timestamp_source: str = "source_frame_index_divided_by_audited_fps"


@dataclass(frozen=True)
class MayoSSLView:
    features: np.ndarray
    valid_mask: np.ndarray
    timestamps: np.ndarray
    source_frame_indices: np.ndarray
    facial_transforms: np.ndarray
    facial_transform_mask: np.ndarray
    contiguous_from_previous: np.ndarray
    target_frame_indices: np.ndarray


@dataclass(frozen=True)
class ARKitSequence:
    features: np.ndarray
    valid_mask: np.ndarray
    timestamps: np.ndarray
    source_frame_indices: np.ndarray
    source_fps: float = ARKIT_TIMECODE_FPS
    timestamp_source: str = "arkit_original_timecode_relative_seconds"


@dataclass(frozen=True)
class ARKitSSLView:
    features: np.ndarray
    valid_mask: np.ndarray
    timestamps: np.ndarray
    source_frame_indices: np.ndarray
    target_frame_indices: np.ndarray
    contiguous_from_previous: np.ndarray


@dataclass(frozen=True)
class CompactCacheSummary:
    source_rows: int
    missing_source_frames: int


@dataclass(frozen=True)
class AuthorizedMayoRecording:
    """One retained MediaPipe cache parsed from its authorized byte snapshot."""

    recording_id: str
    group_id: str
    cache_integrity_id: str
    cache_sha256: str
    cache_size_bytes: int
    features_30hz: np.ndarray
    valid_mask_30hz: np.ndarray
    timestamps_30hz: np.ndarray
    source_frame_indices_30hz: np.ndarray
    target_frame_indices_30hz: np.ndarray


@dataclass(frozen=True)
class AuthorizedMayoGeneration:
    """Private live authorization of the coupled cache/exposure generation."""

    schema: str
    collection_manifest_sha256: str
    exposure_manifest_sha256: str
    generation_closure_hmac: str
    recording_count: int
    arkit_count: int
    expected_recording_count: int
    commitment: dict[str, object]
    recordings: tuple[AuthorizedMayoRecording, ...]
    private_key: bytes = dataclass_field(repr=False)


@dataclass(frozen=True)
class DependencyFileSnapshot:
    logical_name: str
    distribution: str
    record_name: str
    path: Path
    sha256: str
    device: int
    inode: int
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class ProvenanceSnapshot:
    source_files: tuple[tuple[Path, str], ...]
    model_file: tuple[Path, str]
    producer_files: tuple[tuple[str, Path, str], ...]
    dependencies: dict[str, str]
    dependency_distributions: dict[str, str]
    dependency_files: tuple[DependencyFileSnapshot, ...]
    dependency_aggregate_sha256: str
    producer_aggregate_sha256: str
    source_aggregate_sha256: str

    @property
    def source_sha256(self) -> tuple[str, ...]:
        return tuple(digest for _path, digest in self.source_files)

    @property
    def model_sha256(self) -> str:
        return self.model_file[1]

    @property
    def producer_sha256(self) -> dict[str, str]:
        return {name: digest for name, _path, digest in self.producer_files}

    @property
    def dependency_sha256(self) -> dict[str, str]:
        return {
            item.logical_name: item.sha256 for item in self.dependency_files
        }

    @property
    def dependency_file_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.dependency_files:
            counts[item.distribution] = counts.get(item.distribution, 0) + 1
        return counts


@dataclass(frozen=True)
class PinnedSourceSnapshot:
    original_path: Path
    pinned_path: Path
    sha256: str
    device: int
    inode: int
    size: int
    mtime_ns: int


class MayoVideoClinical23Extractor(MediaPipeFeatureExtractor):
    """One homogeneous FaceLandmarker VIDEO-mode producer for Mayo SSL.

    Blendshapes, clinical landmarks, nuisance values, and the facial transform
    are derived from the same ``detect_for_video`` result.  The legacy IMAGE
    mode extractor is intentionally not used by this cache builder.
    """

    producer_protocol = VIDEO_PRODUCER_PROTOCOL
    adapter_version = VIDEO_ADAPTER_VERSION

    def __init__(
        self,
        model_path: str | Path,
        *,
        runtime_factory: Callable[[Path], tuple[object, object]] | None = None,
    ):
        model = _require_regular_file(model_path, "MediaPipe model")
        self.landmark_features = "clinical23"
        self.capture_mirrored = None
        self.with_geometry = True
        self._closed = False
        self._bs_names = None
        self._pairs = None
        self._last_timestamp_ms: int | None = None
        if runtime_factory is not None:
            self._mp, self._landmarker = runtime_factory(model)
            return

        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

    def extract_video_frame(
        self,
        bgr: np.ndarray,
        timestamp_ms: int,
    ) -> tuple[np.ndarray | None, dict[str, float] | None, np.ndarray | None]:
        if (
            not isinstance(timestamp_ms, int) or isinstance(timestamp_ms, bool)
            or timestamp_ms < 0
            or (self._last_timestamp_ms is not None
                and timestamp_ms <= self._last_timestamp_ms)
        ):
            raise ValueError("VIDEO-mode timestamps must be strictly increasing milliseconds")
        if not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("VIDEO-mode input must be one BGR frame")
        self._last_timestamp_ms = timestamp_ms
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        )
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_blendshapes or not result.face_landmarks:
            return None, None, None
        categories = result.face_blendshapes[0]
        self._ensure_layout([category.category_name for category in categories])
        scores = np.asarray([category.score for category in categories], dtype=np.float32)
        landmarks = result.face_landmarks[0]
        try:
            features = self._assemble_features(
                scores, landmarks, image_width=bgr.shape[1], image_height=bgr.shape[0]
            )
            nuisance = self._nuisance_geometry(
                landmarks, image_width=bgr.shape[1], image_height=bgr.shape[0]
            )
        except ValueError:
            return None, None, None
        transforms = getattr(result, "facial_transformation_matrixes", None)
        transform = None
        if transforms:
            candidate = np.asarray(transforms[0], dtype=np.float32)
            if candidate.shape == (4, 4) and np.isfinite(candidate).all():
                transform = candidate
        return features, nuisance, transform


def _lexical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_symlink(path: Path) -> bool:
    try:
        return stat.S_ISLNK(os.lstat(path).st_mode)
    except FileNotFoundError:
        return False


def _assert_no_symlink_components(path: str | Path) -> Path:
    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        candidate = current / part
        try:
            info = os.lstat(candidate)
        except FileNotFoundError:
            break
        if stat.S_ISLNK(info.st_mode):
            if current == Path(absolute.anchor) and info.st_uid == 0:
                current = Path(os.path.realpath(candidate))
                continue
            raise ValueError("filesystem path component must not be a symlink")
        current = candidate
    return absolute


def _require_directory(path: str | Path, field: str) -> Path:
    checked = _assert_no_symlink_components(path)
    try:
        info = os.lstat(checked)
    except FileNotFoundError as exc:
        raise ValueError(f"{field} is missing") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{field} must be a real directory")
    return checked


def _require_regular_file(path: str | Path, field: str) -> Path:
    checked = _assert_no_symlink_components(path)
    try:
        info = os.lstat(checked)
    except FileNotFoundError as exc:
        raise ValueError(f"{field} is missing") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{field} must be a regular file")
    return checked


def _require_private_directory_stat(info: os.stat_result, field: str) -> None:
    if (
        not stat.S_ISDIR(info.st_mode)
        or int(info.st_uid) != os.geteuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise ValueError(f"{field} must be a current-owner mode-0700 directory")


def _require_private_regular_stat(info: os.stat_result, field: str) -> None:
    if (
        not stat.S_ISREG(info.st_mode)
        or int(info.st_uid) != os.geteuid()
        or stat.S_IMODE(info.st_mode) != 0o600
        or int(info.st_nlink) != 1
    ):
        raise ValueError(
            f"{field} must be a singly-linked current-owner mode-0600 regular file"
        )


def _movement_stable_regular_snapshot(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_mode),
        int(value.st_uid), int(value.st_gid), int(value.st_nlink),
        int(value.st_size),
    )


def _require_private_directory(path: str | Path, field: str) -> Path:
    checked = _require_directory(path, field)
    _require_private_directory_stat(os.lstat(checked), field)
    return checked


def _private_generation_storage_ledger(
    path: str | Path,
    field: str,
) -> tuple[Path, tuple[tuple[str, tuple[str, ...], tuple[int, ...]], ...]]:
    root = _require_private_directory(path, field)
    entries = 0
    total_bytes = 0
    pending: list[tuple[Path, int]] = [(root, 0)]
    records: list[tuple[str, tuple[str, ...], tuple[int, ...]]] = [
        ("directory", (), _directory_snapshot(os.lstat(root)))
    ]
    while pending:
        directory, depth = pending.pop()
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            entries += 1
            if entries > 64 or depth + 1 > 4:
                raise ValueError(f"{field} exceeds its exact structural budget")
            info = os.lstat(child)
            if stat.S_ISDIR(info.st_mode):
                _require_private_directory_stat(info, field)
                records.append((
                    "directory", child.relative_to(root).parts,
                    _directory_snapshot(info),
                ))
                pending.append((child, depth + 1))
            elif stat.S_ISREG(info.st_mode):
                _require_private_regular_stat(info, field)
                records.append((
                    "file", child.relative_to(root).parts,
                    _regular_snapshot(info),
                ))
                total_bytes += int(info.st_size)
                if total_bytes > _MAX_EXACT_PRIVATE_TREE_REGULAR_BYTES:
                    raise ValueError(
                        f"{field} exceeds its exact regular-payload budget"
                    )
            else:
                raise ValueError(f"{field} contains unsafe storage")
    return root, tuple(sorted(records, key=lambda item: (item[1], item[0])))


def _private_generation_storage_commitment(
    path: str | Path,
    field: str,
) -> tuple[
    Path,
    tuple[tuple[str, tuple[str, ...], tuple[int, ...]], ...],
    str,
]:
    with _hold_private_storage_tree(Path(path), field) as held:
        ledger, commitment = _held_private_generation_storage_commitment(
            held, field,
        )
        return held.root, ledger, commitment


def _held_private_generation_storage_commitment(
    held: _HeldPrivateStorageTree,
    field: str,
) -> tuple[
    tuple[tuple[str, tuple[str, ...], tuple[int, ...]], ...],
    str,
]:
    ledger = tuple(
        (entry.kind, entry.parts, entry.identity)
        for entry in held.entries
    )
    file_digests: list[tuple[tuple[str, ...], str]] = []
    remaining_bytes = _MAX_EXACT_PRIVATE_TREE_REGULAR_BYTES
    for entry in held.entries:
        if entry.kind != "file":
            continue
        digest, size = _sha256_held_private_regular_file(
            entry,
            field,
            max_bytes=remaining_bytes,
        )
        remaining_bytes -= size
        file_digests.append((entry.parts, digest))
    _assert_held_private_storage_tree(held, field)
    encoded = json.dumps(
        {
            "ledger": ledger,
            "file_sha256": tuple(file_digests),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return ledger, hashlib.sha256(encoded).hexdigest()


def _sha256_private_regular_file(
    path: Path,
    field: str,
    *,
    max_bytes: int,
) -> tuple[str, int]:
    if (
        not isinstance(max_bytes, int)
        or isinstance(max_bytes, bool)
        or max_bytes < 0
    ):
        raise ValueError(f"{field} digest byte limit is invalid")
    checked = _require_regular_file(path, field)
    descriptor = os.open(
        checked,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        linked_before = os.lstat(checked)
        _require_private_regular_stat(before, field)
        _require_private_regular_stat(linked_before, field)
        identity = _regular_snapshot(before)
        if _regular_snapshot(linked_before) != identity:
            raise ValueError(f"{field} changed while opened for digest")
        if int(before.st_size) > max_bytes:
            raise ValueError(f"{field} exceeds its shared digest budget")
        digest = hashlib.sha256()
        total = 0
        while block := os.read(
            descriptor, min(1024 * 1024, max_bytes - total + 1),
        ):
            total += len(block)
            if total > max_bytes:
                raise ValueError(f"{field} exceeds its shared digest budget")
            digest.update(block)
        after = os.fstat(descriptor)
        linked_after = os.lstat(checked)
        _require_private_regular_stat(after, field)
        _require_private_regular_stat(linked_after, field)
        if (
            _regular_snapshot(after) != identity
            or _regular_snapshot(linked_after) != identity
            or total != int(after.st_size)
        ):
            raise ValueError(f"{field} changed while its digest was computed")
        return digest.hexdigest(), total
    finally:
        os.close(descriptor)


def _private_regular_storage_commitment(
    stable_identity: tuple[int, ...],
    digest: str,
) -> str:
    encoded = json.dumps(
        {"identity": stable_identity, "sha256": digest},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _publish_private_path_no_replace(
    source: Path,
    destination: Path,
    field: str,
    *,
    expected_identity: tuple[int, ...] | None = None,
) -> None:
    if source.parent != destination.parent or source.name == destination.name:
        raise ValueError(f"{field} publication paths are inconsistent")
    parent_descriptor = _open_nofollow_directory(
        source.parent, f"{field} publication parent",
    )
    try:
        staged = os.stat(
            source.name, dir_fd=parent_descriptor, follow_symlinks=False,
        )
        if stat.S_ISDIR(staged.st_mode):
            _require_private_directory_stat(staged, field)
            staged_identity = _directory_snapshot(staged)
            snapshot = _directory_snapshot
        elif stat.S_ISREG(staged.st_mode):
            _require_private_regular_stat(staged, field)
            staged_identity = _movement_stable_regular_snapshot(staged)
            snapshot = _movement_stable_regular_snapshot
        else:
            raise ValueError(f"{field} staged object is unsafe")
        if (
            expected_identity is not None
            and staged_identity != expected_identity
        ):
            raise ValueError(f"{field} source identity changed before publication")
        try:
            os.stat(
                destination.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"{field} destination already exists")
        library = ctypes.CDLL(None, use_errno=True)
        old = os.fsencode(source.name)
        new = os.fsencode(destination.name)
        ctypes.set_errno(0)
        if sys.platform == "darwin" and hasattr(library, "renameatx_np"):
            operation = library.renameatx_np
            flag = 0x00000004 | 0x00000010
        elif hasattr(library, "renameat2"):
            operation = library.renameat2
            flag = 0x00000001
        else:
            raise OSError(f"{field} atomic no-replace publication is unavailable")
        operation.argtypes = (
            ctypes.c_int, ctypes.c_char_p,
            ctypes.c_int, ctypes.c_char_p,
            ctypes.c_uint,
        )
        operation.restype = ctypes.c_int
        if operation(
            parent_descriptor, old,
            parent_descriptor, new,
            flag,
        ) != 0:
            error = ctypes.get_errno()
            if error in {errno.EEXIST, errno.ENOTEMPTY}:
                raise FileExistsError(f"{field} destination already exists")
            raise OSError(error, os.strerror(error), destination.name)
        published = os.stat(
            destination.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if snapshot(published) != staged_identity:
            raise ValueError(f"{field} changed during no-replace publication")
        try:
            os.stat(
                source.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise ValueError(f"{field} staging name remains after publication")
    finally:
        os.close(parent_descriptor)


@dataclass(frozen=True)
class _HeldPrivateStorageEntry:
    kind: str
    parts: tuple[str, ...]
    descriptor: int = dataclass_field(repr=False)
    identity: tuple[int, ...]


def _sha256_held_private_regular_file(
    entry: _HeldPrivateStorageEntry,
    field: str,
    *,
    max_bytes: int,
) -> tuple[str, int]:
    if entry.kind != "file" or max_bytes < 0:
        raise ValueError(f"{field} held digest inputs are invalid")
    before = os.fstat(entry.descriptor)
    _require_private_regular_stat(before, field)
    if _regular_snapshot(before) != entry.identity:
        raise ValueError(f"{field} held file changed before digest")
    if int(before.st_size) > max_bytes:
        raise ValueError(f"{field} exceeds its shared digest budget")
    os.lseek(entry.descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    total = 0
    while block := os.read(
        entry.descriptor, min(1024 * 1024, max_bytes - total + 1),
    ):
        total += len(block)
        if total > max_bytes:
            raise ValueError(f"{field} exceeds its shared digest budget")
        digest.update(block)
    after = os.fstat(entry.descriptor)
    _require_private_regular_stat(after, field)
    if _regular_snapshot(after) != entry.identity or total != int(after.st_size):
        raise ValueError(f"{field} held file changed during digest")
    return digest.hexdigest(), total


@dataclass(frozen=True)
class _HeldPrivateStorageTree:
    root: Path
    entries: tuple[_HeldPrivateStorageEntry, ...]


def _assert_held_private_storage_tree(
    held: _HeldPrivateStorageTree,
    field: str,
) -> None:
    for entry in held.entries:
        opened = os.fstat(entry.descriptor)
        linked = os.lstat(held.root.joinpath(*entry.parts))
        snapshot = (
            _directory_snapshot if entry.kind == "directory"
            else _regular_snapshot
        )
        if entry.kind == "directory":
            _require_private_directory_stat(opened, field)
            _require_private_directory_stat(linked, field)
        else:
            _require_private_regular_stat(opened, field)
            _require_private_regular_stat(linked, field)
        if snapshot(opened) != entry.identity or snapshot(linked) != entry.identity:
            raise ValueError(f"{field} changed while held")


@contextmanager
def _hold_private_storage_tree(path: Path, field: str):
    root, ledger = _private_generation_storage_ledger(path, field)
    descriptors = ExitStack()
    entries: list[_HeldPrivateStorageEntry] = []
    try:
        for kind, parts, identity in ledger:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            if kind == "directory":
                flags |= getattr(os, "O_DIRECTORY", 0)
            descriptor = os.open(root.joinpath(*parts), flags)
            descriptors.callback(os.close, descriptor)
            entries.append(_HeldPrivateStorageEntry(
                kind=kind,
                parts=parts,
                descriptor=descriptor,
                identity=identity,
            ))
        held = _HeldPrivateStorageTree(root=root, entries=tuple(entries))
        _assert_held_private_storage_tree(held, field)
        yield held
    finally:
        descriptors.__exit__(*sys.exc_info())


@dataclass(frozen=True)
class _HeldPrivateRegularStorage:
    path: Path
    descriptor: int = dataclass_field(repr=False)
    identity: tuple[int, ...]


def _assert_held_private_regular_storage(
    held: _HeldPrivateRegularStorage,
    field: str,
) -> None:
    opened = os.fstat(held.descriptor)
    linked = os.lstat(held.path)
    _require_private_regular_stat(opened, field)
    _require_private_regular_stat(linked, field)
    if (
        _regular_snapshot(opened) != held.identity
        or _regular_snapshot(linked) != held.identity
    ):
        raise ValueError(f"{field} changed while held")


@contextmanager
def _hold_private_regular_storage(path: Path, field: str):
    checked = _require_regular_file(path, field)
    descriptor = os.open(
        checked,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        identity = _regular_snapshot(os.fstat(descriptor))
        held = _HeldPrivateRegularStorage(
            path=checked, descriptor=descriptor, identity=identity,
        )
        _assert_held_private_regular_storage(held, field)
        yield held
    finally:
        os.close(descriptor)


def _held_private_regular_storage_commitment(
    held: _HeldPrivateRegularStorage,
    field: str,
) -> str:
    entry = _HeldPrivateStorageEntry(
        kind="file",
        parts=(held.path.name,),
        descriptor=held.descriptor,
        identity=held.identity,
    )
    digest, _size = _sha256_held_private_regular_file(
        entry, field, max_bytes=_MAX_MAYO_MANIFEST_BYTES,
    )
    _assert_held_private_regular_storage(held, field)
    return _private_regular_storage_commitment(held.identity[:7], digest)


def _require_private_generation_storage_tree(
    path: str | Path,
    field: str,
) -> Path:
    root, _ledger = _private_generation_storage_ledger(path, field)
    return root


def _require_owned_nonwritable_directory(path: str | Path, field: str) -> Path:
    checked = _require_directory(path, field)
    info = os.lstat(checked)
    if int(info.st_uid) != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o022:
        raise ValueError(
            f"{field} must be current-owner and not group/world writable"
        )
    return checked


def _make_private_directory(path: Path, field: str) -> Path:
    os.mkdir(path, 0o700)
    os.chmod(path, 0o700, follow_symlinks=False)
    return _require_private_directory(path, field)


def _open_exclusive_private_file(path: Path, field: str) -> int:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        opened = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        _require_private_regular_stat(opened, field)
        if _regular_snapshot(opened) != _regular_snapshot(current):
            raise ValueError(f"{field} changed identity during creation")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def sha256_file(path: str | Path) -> str:
    checked = _require_regular_file(path, "hashed file")
    digest = hashlib.sha256()
    with checked.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pin_source_file(
    source_path: str | Path,
    snapshot_directory: str | Path,
    opaque_name: str,
    *,
    expected_sha256: str,
) -> PinnedSourceSnapshot:
    """Hard-link one audited inode so decoding uses the bytes that were hashed."""
    if _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("pinned source expected SHA-256 is not canonical")
    if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", opaque_name) is None:
        raise ValueError("pinned source name must be opaque and path-free")
    original = _require_regular_file(source_path, "source to pin")
    directory = _require_directory(snapshot_directory, "source snapshot directory")
    pinned = directory / opaque_name
    if pinned.exists() or _is_symlink(pinned):
        raise FileExistsError("pinned source destination already exists")
    before = os.lstat(original)
    try:
        os.link(original, pinned, follow_symlinks=False)
        pinned_info = os.lstat(pinned)
        after = os.lstat(original)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        identity_pinned = (
            pinned_info.st_dev, pinned_info.st_ino,
            pinned_info.st_size, pinned_info.st_mtime_ns,
        )
        if identity_before != identity_after or identity_after != identity_pinned:
            raise ValueError("source inode changed while creating its pinned snapshot")
        observed = sha256_file(pinned)
        if not hmac.compare_digest(observed, expected_sha256):
            raise ValueError("pinned bytes differ from the inventory SHA-256")
        final = os.lstat(original)
        if (
            final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns
        ) != identity_after:
            raise ValueError("source inode changed while hashing its pinned snapshot")
        return PinnedSourceSnapshot(
            original_path=original,
            pinned_path=pinned,
            sha256=observed,
            device=after.st_dev,
            inode=after.st_ino,
            size=after.st_size,
            mtime_ns=after.st_mtime_ns,
        )
    except BaseException:
        if pinned.exists() and not _is_symlink(pinned):
            pinned.unlink()
        raise


def assert_pinned_source_unchanged(snapshot: PinnedSourceSnapshot) -> None:
    if not isinstance(snapshot, PinnedSourceSnapshot):
        raise ValueError("pinned source snapshot has the wrong type")
    original = _require_regular_file(snapshot.original_path, "original pinned source")
    info = os.lstat(original)
    if (
        info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns
    ) != (snapshot.device, snapshot.inode, snapshot.size, snapshot.mtime_ns):
        raise ValueError("original source path or inode changed after pinning")
    if not hmac.compare_digest(sha256_file(original), snapshot.sha256):
        raise ValueError("original source bytes changed after pinning")
    pinned = _require_regular_file(snapshot.pinned_path, "pinned source")
    pinned_info = os.lstat(pinned)
    if (pinned_info.st_dev, pinned_info.st_ino) != (snapshot.device, snapshot.inode):
        raise ValueError("pinned decoder path no longer names the audited inode")


def _probe_video(path: Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("video cannot be opened")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        raw_frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        raw_width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if (
        not math.isfinite(fps) or fps <= 0
        or not math.isfinite(raw_frames) or raw_frames <= 0 or not raw_frames.is_integer()
        or not math.isfinite(raw_width) or raw_width <= 0 or not raw_width.is_integer()
        or not math.isfinite(raw_height) or raw_height <= 0 or not raw_height.is_integer()
    ):
        raise ValueError("video metadata must be finite positive integral dimensions/frames")
    return VideoMetadata(int(raw_frames), fps, int(raw_width), int(raw_height))


def _parse_arkit_timecode_milliframes(value: str) -> int:
    match = _ARKIT_TIMECODE.fullmatch(value)
    if match is None:
        raise ValueError("ARKit Timecode must be HH:MM:SS:FF.subframe at 60 fps")
    whole_frames = (
        (int(match["hour"]) * 3600
         + int(match["minute"]) * 60
         + int(match["second"])) * 60
        + int(match["frame"])
    )
    return whole_frames * 1000 + int(match["subframe"])


def _arkit_timeline(timecodes: Sequence[str]) -> tuple[np.ndarray, np.ndarray, int]:
    if not timecodes:
        raise ValueError("ARKit trajectory has no Timecode rows")
    clocks = [_parse_arkit_timecode_milliframes(value) for value in timecodes]
    indices = np.zeros(len(clocks), dtype=np.int64)
    missing = 0
    for row in range(1, len(clocks)):
        difference = clocks[row] - clocks[row - 1]
        if difference <= 0:
            raise ValueError("ARKit Timecode must increase strictly without duplicates")
        source_step = (difference + 500) // 1000
        if source_step < 1 or abs(difference - source_step * 1000) > 10:
            raise ValueError("ARKit Timecode increment is not an integral 60-fps step")
        indices[row] = indices[row - 1] + source_step
        missing += source_step - 1
    timestamps = (
        np.asarray(clocks, dtype=np.float64) - float(clocks[0])
    ) / (ARKIT_TIMECODE_FPS * 1000.0)
    return indices, timestamps.astype(np.float64, copy=False), int(missing)


def inspect_arkit_csv(path: str | Path) -> ARKitInspection:
    checked = _require_regular_file(path, "ARKit trajectory")
    with checked.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            raise ValueError("ARKit trajectory is empty") from exc
        expected = ("Timecode", "BlendshapeCount", *ARKIT_BLENDSHAPE_NAMES,
                    *ARKIT_ROTATION_NAMES)
        if header != expected:
            raise ValueError("ARKit trajectory has a noncanonical 52-blendshape layout")
        timecodes: list[str] = []
        for row in reader:
            if len(row) != len(expected):
                raise ValueError("ARKit trajectory has a malformed-width data row")
            timecodes.append(row[0])
        count = len(timecodes)
    if count <= 0:
        raise ValueError("ARKit trajectory has no data rows")
    _indices, _timestamps, missing = _arkit_timeline(timecodes)
    return ARKitInspection(count, ARKIT_BLENDSHAPE_NAMES, missing)


def _canonical_video(session: Path) -> Path | None:
    candidates = sorted(
        (item for item in session.iterdir()
         if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda item: item.name,
    )
    if not candidates:
        return None
    mov = [item for item in candidates if item.suffix.lower() == ".mov"]
    if len(mov) == 1:
        if any(item.stem != mov[0].stem for item in candidates if item != mov[0]):
            raise ValueError(
                "a Mayo session contains an unrelated second clip, not a container proxy"
            )
        return mov[0]
    if len(mov) > 1 or len(candidates) > 1:
        raise ValueError("a Mayo session has multiple ambiguous canonical videos")
    return candidates[0]


def _complete_export(export_root: Path, session_name: str) -> Path | None:
    candidate = export_root / session_name
    if not candidate.is_dir() or _is_symlink(candidate):
        return None
    for filename in EXISTING_EXPORT_FILES:
        path = candidate / filename
        if not path.is_file() or _is_symlink(path):
            return None
    return candidate


def _validate_video_metadata(metadata: VideoMetadata) -> None:
    if (
        isinstance(metadata.frame_count, bool) or metadata.frame_count <= 0
        or not math.isfinite(float(metadata.fps))
        or float(metadata.fps) < TARGET_SSL_FPS or float(metadata.fps) > 240.0
        or isinstance(metadata.width, bool) or metadata.width <= 0
        or isinstance(metadata.height, bool) or metadata.height <= 0
    ):
        raise ValueError("Mayo video metadata must have a finite plausible source FPS")


def inventory_mayo_sources(
    data_root: str | Path,
    export_root: str | Path,
    *,
    probe_video: Callable[[Path], VideoMetadata] = _probe_video,
    inspect_arkit: Callable[[Path], ARKitInspection] = inspect_arkit_csv,
    enforce_frozen: bool = True,
) -> MayoInventory:
    """Inspect the live source tree; frozen counts are a gate, never a substitute."""
    data = _require_directory(data_root, "Mayo data root")
    exports = _require_directory(export_root, "Mayo existing export root")
    sessions = sorted(
        (item for item in data.iterdir() if item.is_dir() and not _is_symlink(item)),
        key=lambda item: item.name,
    )
    video_assets: list[VideoAsset] = []
    without_video: list[Path] = []
    for session in sessions:
        video = _canonical_video(session)
        if video is None:
            without_video.append(session)
            continue
        metadata = probe_video(video)
        if not isinstance(metadata, VideoMetadata):
            raise ValueError("video probe must return VideoMetadata")
        _validate_video_metadata(metadata)
        video_assets.append(VideoAsset(
            session_path=session,
            path=video,
            metadata=metadata,
            source_sha256=sha256_file(video),
            export_dir=_complete_export(exports, session.name),
        ))

    by_digest: dict[str, list[VideoAsset]] = {}
    for asset in video_assets:
        by_digest.setdefault(asset.source_sha256, []).append(asset)
    unique: list[VideoAsset] = []
    duplicates: list[VideoAsset] = []
    for digest in sorted(by_digest):
        members = sorted(by_digest[digest], key=lambda item: item.session_path.name)
        first = members[0]
        for other in members[1:]:
            if other.metadata != first.metadata:
                raise ValueError("equal source hashes have inconsistent video metadata")
        unique.append(first)
        duplicates.extend(members[1:])
    unique.sort(key=lambda item: item.session_path.name)
    duplicates.sort(key=lambda item: item.session_path.name)

    short = tuple(
        asset for asset in unique
        if asset.metadata.duration_seconds <= SHORT_QC_MAX_SECONDS
    )
    short_paths = {asset.path for asset in short}
    long_unique = tuple(asset for asset in unique if asset.path not in short_paths)
    existing = tuple(asset for asset in long_unique if asset.export_dir is not None)
    pending = tuple(asset for asset in long_unique if asset.export_dir is None)

    arkit_assets: list[ARKitAsset] = []
    arkit_sessions: list[Path] = []
    metadata_only: list[Path] = []
    for session in without_video:
        csv_files = sorted(
            (item for item in session.rglob("*_iPhone.csv")
             if item.is_file() and not _is_symlink(item)),
            key=lambda item: str(item.relative_to(session)),
        )
        if not csv_files:
            metadata_only.append(session)
            continue
        arkit_sessions.append(session)
        for path in csv_files:
            inspection = inspect_arkit(path)
            if (
                not isinstance(inspection, ARKitInspection)
                or inspection.row_count <= 0
                or tuple(inspection.feature_names) != ARKIT_BLENDSHAPE_NAMES
                or isinstance(inspection.missing_source_frames, bool)
                or inspection.missing_source_frames < 0
            ):
                raise ValueError("ARKit inspection differs from the frozen 52-column schema")
            arkit_assets.append(ARKitAsset(
                session_path=session,
                path=path,
                row_count=inspection.row_count,
                feature_names=tuple(inspection.feature_names),
                source_sha256=sha256_file(path),
                missing_source_frames=inspection.missing_source_frames,
            ))
    if len({asset.source_sha256 for asset in arkit_assets}) != len(arkit_assets):
        raise ValueError("ARKit auxiliary pool contains duplicate trajectory content")

    counts = {
        "total_sessions": len(sessions),
        "video_bearing_sessions": len(video_assets),
        "without_video_sessions": len(without_video),
        "exact_duplicate_copies_excluded": len(duplicates),
        "short_qc_clips_excluded": len(short),
        "long_unique_videos": len(long_unique),
        "existing_complete_v2_exports": len(existing),
        "remaining_long_videos": len(pending),
        "remaining_long_video_frames": sum(asset.metadata.frame_count for asset in pending),
        "arkit_only_sessions": len(arkit_sessions),
        "arkit_trajectories": len(arkit_assets),
        "arkit_rows": sum(asset.row_count for asset in arkit_assets),
        "arkit_timecode_gaps": sum(
            asset.missing_source_frames for asset in arkit_assets
        ),
        "metadata_only_sessions": len(metadata_only),
    }
    if enforce_frozen and counts != FROZEN_INVENTORY:
        differences = {
            key: {"expected": FROZEN_INVENTORY[key], "observed": counts.get(key)}
            for key in FROZEN_INVENTORY if counts.get(key) != FROZEN_INVENTORY[key]
        }
        raise InventoryDriftError(
            "live Mayo inventory drifted from the frozen contract: "
            + json.dumps(differences, sort_keys=True)
        )
    return MayoInventory(
        data_root=data,
        export_root=exports,
        counts=counts,
        video_instances=tuple(video_assets),
        long_unique_videos=long_unique,
        existing_export_videos=existing,
        pending_videos=pending,
        duplicate_videos=tuple(duplicates),
        short_videos=short,
        arkit_sessions=tuple(arkit_sessions),
        arkit_trajectories=tuple(arkit_assets),
        metadata_only_sessions=tuple(metadata_only),
    )


def _require_salt(salt: bytes) -> bytes:
    if not isinstance(salt, bytes) or len(salt) < 32:
        raise ValueError("local HMAC salt must contain at least 32 bytes")
    return salt


def hmac_identifier(prefix: str, salt: bytes, context: str, material: str) -> str:
    if prefix not in {"rec", "grp", "fp", "inst", "src", "cache", "agg"}:
        raise ValueError("opaque identifier prefix is invalid")
    if not context or not material:
        raise ValueError("opaque identifier inputs must be nonempty")
    digest = hmac.new(
        _require_salt(salt),
        f"{context}\0{material}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{prefix}_{digest}"


def exposure_classification_integrity_id(
    video_rows: Sequence[Mapping[str, object]], salt: bytes,
) -> str:
    """Bind opaque source instances to their frozen exposure classifications."""
    fields = (
        "instance_id", "recording_id", "group_id", "source_integrity_id",
        "source_fingerprint", "status",
    )
    canonical: list[dict[str, str]] = []
    for raw_row in video_rows:
        if not isinstance(raw_row, Mapping):
            raise ValueError("exposure classification row must be an object")
        row: dict[str, str] = {}
        for field in fields:
            value = raw_row.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError("exposure classification identity is incomplete")
            row[field] = value
        canonical.append(row)
    canonical.sort(key=lambda row: row["instance_id"])
    if len({row["instance_id"] for row in canonical}) != len(canonical):
        raise ValueError("exposure classification repeats an instance")
    material = hashlib.sha256(json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    return hmac_identifier(
        "agg", salt, "mayo-exposure-classification-v1", material
    )


def collection_classification_integrity_id(
    media_rows: Sequence[Mapping[str, object]], salt: bytes,
) -> str:
    """Bind retained MediaPipe sources to their frozen legacy classifications."""
    fields = (
        "recording_id", "group_id", "source_integrity_id", "source_fingerprint",
        "legacy_export_audit_status",
    )
    canonical: list[dict[str, str]] = []
    for raw_row in media_rows:
        if not isinstance(raw_row, Mapping):
            raise ValueError("collection classification row must be an object")
        row: dict[str, str] = {}
        for field in fields:
            value = raw_row.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError("collection classification identity is incomplete")
            row[field] = value
        canonical.append(row)
    canonical.sort(key=lambda row: row["recording_id"])
    if len({row["recording_id"] for row in canonical}) != len(canonical):
        raise ValueError("collection classification repeats a recording")
    material = hashlib.sha256(json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    return hmac_identifier(
        "agg", salt, "mayo-collection-classification-v1", material
    )


def _video_public_row(
    asset: VideoAsset,
    salt: bytes,
    legacy_export_audit_status: str,
) -> dict[str, object]:
    digest = asset.source_sha256
    return {
        "recording_id": hmac_identifier("rec", salt, "mayo-mediapipe-recording", digest),
        "group_id": hmac_identifier("grp", salt, "mayo-proven-source-group", digest),
        "source_integrity_id": hmac_identifier(
            "src", salt, "mayo-mediapipe-source-integrity", digest
        ),
        "source_fingerprint": hmac_identifier("fp", salt, "mayo-source-fingerprint", digest),
        "cache_source": "raw_video_reextracted_homogeneous_video_mode",
        "producer_protocol": VIDEO_PRODUCER_PROTOCOL,
        "producer_adapter_version": VIDEO_ADAPTER_VERSION,
        "legacy_export_audit_status": legacy_export_audit_status,
        "identity_status": "unknown_patient_identity",
        "split_unit": "recording_held_out",
        "development_only": True,
        "ssl_exposed": True,
        "independent_evaluation_eligible": False,
    }


def _arkit_public_row(asset: ARKitAsset, salt: bytes) -> dict[str, object]:
    digest = asset.source_sha256
    return {
        "recording_id": hmac_identifier("rec", salt, "mayo-arkit-recording", digest),
        "group_id": hmac_identifier("grp", salt, "mayo-arkit-recording-group", digest),
        "source_integrity_id": hmac_identifier(
            "src", salt, "mayo-arkit-source-integrity", digest
        ),
        "source_fingerprint": hmac_identifier("fp", salt, "mayo-source-fingerprint", digest),
        "feature_schema": "arkit_blendshapes_52_v1",
        "identity_status": "unknown_patient_identity",
        "split_unit": "recording_held_out",
        "development_only": True,
        "ssl_exposed": True,
        "independent_evaluation_eligible": False,
    }


def build_public_manifests(
    inventory: MayoInventory,
    salt: bytes,
) -> tuple[dict[str, object], dict[str, object]]:
    salt = _require_salt(salt)
    existing_paths = {asset.path for asset in inventory.existing_export_videos}
    long_paths = {asset.path for asset in inventory.long_unique_videos}
    duplicate_paths = {asset.path for asset in inventory.duplicate_videos}
    short_paths = {asset.path for asset in inventory.short_videos}
    media_rows = sorted(
        (_video_public_row(
            asset,
            salt,
            ("not_reused_unverifiable_source_binding"
             if asset.path in existing_paths else "no_complete_legacy_export"),
        ) for asset in inventory.long_unique_videos),
        key=lambda row: str(row["recording_id"]),
    )
    arkit_rows = sorted(
        (_arkit_public_row(asset, salt) for asset in inventory.arkit_trajectories),
        key=lambda row: str(row["recording_id"]),
    )
    collection: dict[str, object] = {
        "schema_version": COLLECTION_SCHEMA,
        "dataset": COLLECTION_DATASET,
        "identity_status": UNKNOWN_IDENTITY,
        "split_unit": RECORDING_HELD_OUT,
        "feature_schema": DYNAMIC_FEATURE_SCHEMA,
        "feature_names": list(DYNAMIC_FEATURE_NAMES),
        "capture_mirrored": "unknown",
        "normalization_transform": TRANSFORM_NORMALIZATION,
        "temporal_protocol": {
            "source_timeline": "per_recording_audited_fps_and_monotonic_source_index",
            "ssl_view_hz": 30,
            "resampling": "exact_target_source_index_selection_no_interpolation_or_nearest_fill",
        },
        "modality_boundary": (
            "ARKit 52-blendshape trajectories are auxiliary-only and are never "
            "concatenated with or promoted to MediaPipe landmarks"
        ),
        "counts": dict(inventory.counts),
        "mediapipe_records": media_rows,
        "arkit_records": arkit_rows,
        "classification_integrity_id": collection_classification_integrity_id(
            media_rows, salt
        ),
        "metadata_only_exclusions": {
            "index_or_depth_metadata_only_no_video_or_arkit_trajectory": (
                len(inventory.metadata_only_sessions)
            )
        },
    }
    video_rows: list[dict[str, object]] = []
    for asset in inventory.video_instances:
        if asset.path in duplicate_paths:
            status = "exact_duplicate_excluded"
        elif asset.path in short_paths:
            status = "qc_only_short_clip_excluded"
        elif asset.path in long_paths:
            status = "mediapipe_ssl"
        else:
            raise AssertionError("video instance was not classified")
        digest = asset.source_sha256
        private_instance_material = str(asset.path.relative_to(inventory.data_root))
        video_rows.append({
            "instance_id": hmac_identifier(
                "inst", salt, "mayo-private-source-instance",
                f"{digest}\0{private_instance_material}",
            ),
            "recording_id": hmac_identifier(
                "rec", salt, "mayo-mediapipe-recording", digest
            ),
            "group_id": hmac_identifier(
                "grp", salt, "mayo-proven-source-group", digest
            ),
            "source_integrity_id": hmac_identifier(
                "src", salt, "mayo-mediapipe-source-integrity", digest
            ),
            "source_fingerprint": hmac_identifier(
                "fp", salt, "mayo-source-fingerprint", digest
            ),
            "status": status,
            "identity_status": UNKNOWN_IDENTITY,
            "split_unit": RECORDING_HELD_OUT,
            "development_only": True,
            "ssl_exposed": True,
            "independent_evaluation_eligible": False,
        })
    exposure: dict[str, object] = {
        "schema_version": EXPOSURE_SCHEMA,
        "dataset": EXPOSURE_DATASET,
        "policy": EXPOSURE_POLICY,
        "identity_status": UNKNOWN_IDENTITY,
        "videos": sorted(video_rows, key=lambda row: str(row["instance_id"])),
        "arkit_trajectories": arkit_rows,
        "classification_integrity_id": exposure_classification_integrity_id(
            video_rows, salt
        ),
        "counts": {
            "videos": len(video_rows),
            "arkit_trajectories": len(arkit_rows),
        },
    }
    validate_public_manifest(collection)
    validate_public_manifest(exposure)
    return collection, exposure


def validate_public_manifest(value: object) -> None:
    """Reject raw locations, names, non-finite values, and JSON-unsafe objects."""
    forbidden_keys = {"path", "filename", "session", "session_name", "stem", "dirname"}
    forbidden_suffixes = ("_path", "_filename", "_session_name", "_stem", "_dirname")

    def walk(item: object, key: str | None = None) -> None:
        if key is not None:
            lowered = key.lower()
            if lowered in forbidden_keys or lowered.endswith(forbidden_suffixes):
                raise ValueError("public manifest contains a private-location field")
        if item is None or isinstance(item, (bool, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("public manifest contains a non-finite number")
            return
        if isinstance(item, str):
            lower = item.lower()
            if item.startswith(("/", "~")) or lower.endswith(
                (".mov", ".mp4", ".m4v", ".csv")
            ) or _RAW_MAYO_NAME.search(item) is not None:
                raise ValueError("public manifest contains a raw location or filename")
            return
        if isinstance(item, list):
            for child in item:
                walk(child)
            return
        if isinstance(item, dict):
            for child_key, child in item.items():
                if not isinstance(child_key, str):
                    raise ValueError("public manifest keys must be strings")
                walk(child, child_key)
            return
        raise ValueError("public manifest contains a non-JSON value")

    walk(value)
    json.dumps(value, sort_keys=True, allow_nan=False)


def _validate_media_sequence(sequence: MayoMediaSequence) -> MayoMediaSequence:
    if not isinstance(sequence, MayoMediaSequence):
        raise ValueError("MediaPipe sequence has the wrong type")
    features = np.asarray(sequence.features)
    mask = np.asarray(sequence.valid_mask)
    timestamps = np.asarray(sequence.timestamps)
    indices = np.asarray(sequence.source_frame_indices)
    transforms = np.asarray(sequence.facial_transforms)
    transform_mask = np.asarray(sequence.facial_transform_mask)
    n = features.shape[0] if features.ndim == 2 else -1
    if features.dtype != np.float32 or features.shape != (n, 95) or n <= 0:
        raise ValueError("MediaPipe features must have shape (T,95) float32")
    if mask.dtype != np.bool_ or mask.shape != (n,):
        raise ValueError("MediaPipe valid mask must be bool with shape (T,)")
    if not mask.any():
        raise ValueError("MediaPipe sequence must contain a valid detected frame")
    if timestamps.dtype != np.float64 or timestamps.shape != (n,):
        raise ValueError("MediaPipe timestamps must be float64 with shape (T,)")
    if indices.dtype != np.int64 or indices.shape != (n,):
        raise ValueError("MediaPipe source indices must be int64 with shape (T,)")
    if transforms.dtype != np.float32 or transforms.shape != (n, 4, 4):
        raise ValueError("facial transform tensor must have shape (T,4,4) float32")
    if transform_mask.dtype != np.bool_ or transform_mask.shape != (n,):
        raise ValueError("facial transform mask must be bool with shape (T,)")
    if not isinstance(sequence.transform_source, str) or not sequence.transform_source:
        raise ValueError("facial transform provenance must be explicit")
    if (
        not isinstance(sequence.source_fps, (int, float))
        or isinstance(sequence.source_fps, bool)
        or not math.isfinite(float(sequence.source_fps))
        or float(sequence.source_fps) < TARGET_SSL_FPS
        or float(sequence.source_fps) > 240.0
        or sequence.timestamp_source
        != "source_frame_index_divided_by_audited_fps"
    ):
        raise ValueError("source FPS and timestamp provenance must be explicit")
    if (
        not np.isfinite(features).all() or not np.isfinite(timestamps).all()
        or not np.isfinite(transforms).all()
    ):
        raise ValueError("MediaPipe cache arrays must be finite")
    if (indices < 0).any() or (np.diff(indices) <= 0).any():
        raise ValueError("source frame indices must increase strictly")
    if (np.diff(timestamps) <= 0).any():
        raise ValueError("timestamps must increase strictly")
    expected_timestamps = indices.astype(np.float64) / float(sequence.source_fps)
    if not np.allclose(timestamps, expected_timestamps, rtol=0.0, atol=1e-12):
        raise ValueError("timestamps must preserve the audited source-FPS timeline")
    if np.any(features[~mask] != 0.0):
        raise ValueError("invalid MediaPipe feature rows must be canonical zero")
    if np.any(transform_mask & ~mask):
        raise ValueError("facial transforms cannot outlive their same-frame detection")
    if np.any(transforms[~transform_mask] != 0.0):
        raise ValueError("invalid facial transform rows must be canonical zero")
    return sequence


def downsample_to_30hz(sequence: MayoMediaSequence) -> MayoSSLView:
    sequence = _validate_media_sequence(sequence)
    selected_rows: list[int] = []
    target_indices: list[int] = []
    fps = float(sequence.source_fps)
    fps_fraction = Fraction(str(fps))
    fps_numerator = fps_fraction.numerator
    fps_denominator = fps_fraction.denominator
    target_rate = int(TARGET_SSL_FPS)
    target_denominator = fps_denominator * target_rate
    for row, raw_source_index in enumerate(sequence.source_frame_indices):
        source_index = int(raw_source_index)
        lower_numerator = (2 * source_index - 1) * target_denominator
        target_index = max(
            0, -(-lower_numerator // (2 * fps_numerator))
        )
        expected_source = (
            2 * target_index * fps_numerator + target_denominator
        ) // (2 * target_denominator)
        if expected_source == source_index:
            selected_rows.append(row)
            target_indices.append(target_index)
    selected = np.asarray(selected_rows, dtype=np.int64)
    targets = np.asarray(target_indices, dtype=np.int64)
    if selected.size == 0:
        raise ValueError("30-Hz source-frame selection produced no rows")
    if len(targets) > 1 and (np.diff(targets) <= 0).any():
        raise ValueError("source FPS cannot support a unique exact 30-Hz view")
    indices = sequence.source_frame_indices[selected].copy()
    mask = sequence.valid_mask[selected].copy()
    contiguous = np.zeros(len(selected), dtype=bool)
    if len(selected) > 1:
        contiguous[1:] = (
            (np.diff(targets) == 1)
            & mask[:-1]
            & mask[1:]
        )
    return MayoSSLView(
        features=sequence.features[selected].copy(),
        valid_mask=mask,
        timestamps=sequence.timestamps[selected].copy(),
        source_frame_indices=indices,
        facial_transforms=sequence.facial_transforms[selected].copy(),
        facial_transform_mask=sequence.facial_transform_mask[selected].copy(),
        contiguous_from_previous=contiguous,
        target_frame_indices=targets,
    )


def downsample_60hz_to_30hz(sequence: MayoMediaSequence) -> MayoSSLView:
    """Compatibility alias; the implementation now honors each source FPS."""
    return downsample_to_30hz(sequence)


def _capture_metadata(capture) -> VideoMetadata:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    raw_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (
        not math.isfinite(raw_count) or raw_count <= 0 or not raw_count.is_integer()
        or not math.isfinite(raw_width) or raw_width <= 0 or not raw_width.is_integer()
        or not math.isfinite(raw_height) or raw_height <= 0 or not raw_height.is_integer()
    ):
        raise ValueError("video capture metadata is invalid")
    metadata = VideoMetadata(int(raw_count), fps, int(raw_width), int(raw_height))
    _validate_video_metadata(metadata)
    return metadata


def extract_video_sequence(
    path: str | Path,
    extractor: MayoVideoClinical23Extractor,
    *,
    capture_factory: Callable[[str], object] = cv2.VideoCapture,
    expected_metadata: VideoMetadata | None = None,
) -> MayoMediaSequence:
    """Stream one full video through the shared public clinical23 extractor."""
    capture = capture_factory(str(path))
    features: list[np.ndarray | None] = []
    masks: list[bool] = []
    transforms: list[np.ndarray | None] = []
    try:
        if not capture.isOpened():
            raise ValueError("source video cannot be opened")
        metadata = _capture_metadata(capture)
        if expected_metadata is not None:
            _validate_video_metadata(expected_metadata)
            if (
                metadata.frame_count != expected_metadata.frame_count
                or metadata.width != expected_metadata.width
                or metadata.height != expected_metadata.height
                or abs(metadata.fps - expected_metadata.fps) > 1e-6
            ):
                raise ValueError("decode metadata differs from the audited source metadata")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            source_index = len(features)
            timestamp_ms = int(round(source_index * 1000.0 / metadata.fps))
            vector, _nuisance, transform = extractor.extract_video_frame(
                frame, timestamp_ms
            )
            if vector is None:
                features.append(None)
                masks.append(False)
                transforms.append(None)
                continue
            array = np.asarray(vector, dtype=np.float32)
            if array.shape != (95,) or not np.isfinite(array).all():
                raise ValueError("public clinical23 extractor returned a malformed frame")
            features.append(array)
            masks.append(True)
            if transform is None:
                transforms.append(None)
            else:
                transform_array = np.asarray(transform, dtype=np.float32)
                if transform_array.shape != (4, 4) or not np.isfinite(transform_array).all():
                    raise ValueError("VIDEO extractor returned a malformed facial transform")
                transforms.append(transform_array)
    finally:
        capture.release()
    if len(features) != metadata.frame_count:
        raise ValueError("decoded frame count differs from frozen video metadata")
    if getattr(extractor, "feature_schema", None) != DYNAMIC_FEATURE_SCHEMA:
        raise ValueError("public extractor schema differs from clinical23_v2")
    if tuple(getattr(extractor, "feature_names", ())) != tuple(DYNAMIC_FEATURE_NAMES):
        raise ValueError("public extractor feature order differs from the registered 95 columns")
    matrix = np.zeros((len(features), 95), dtype=np.float32)
    valid = np.asarray(masks, dtype=bool)
    transform_matrix = np.zeros((len(features), 4, 4), dtype=np.float32)
    transform_valid = np.zeros(len(features), dtype=bool)
    for index, vector in enumerate(features):
        if vector is not None:
            matrix[index] = vector
        if transforms[index] is not None:
            transform_matrix[index] = transforms[index]
            transform_valid[index] = True
    if not valid.any():
        raise ValueError("MediaPipe detected no valid clinical23 frames")
    source_indices = np.arange(len(features), dtype=np.int64)
    sequence = MayoMediaSequence(
        features=matrix,
        valid_mask=valid,
        timestamps=source_indices.astype(np.float64) / metadata.fps,
        source_frame_indices=source_indices,
        facial_transforms=transform_matrix,
        facial_transform_mask=transform_valid,
        transform_source="same_detection_mediapipe_video_mode",
        source_fps=float(metadata.fps),
        timestamp_source="source_frame_index_divided_by_audited_fps",
    )
    return _validate_media_sequence(sequence)


def extract_homogeneous_video_sequences(
    assets: Sequence[VideoAsset],
    extractor_factory: Callable[..., MayoVideoClinical23Extractor],
    *,
    model_path: str | Path,
    capture_factory: Callable[[str], object] = cv2.VideoCapture,
    source_paths: Mapping[Path, Path] | None = None,
):
    """Yield every long video through the same VIDEO-mode producer.

    ``export_dir`` is deliberately ignored: existing exports have no
    cryptographically verifiable binding to their source video.
    """
    remapped = dict(source_paths or {})
    for asset in assets:
        if not isinstance(asset, VideoAsset):
            raise ValueError("homogeneous extraction received a non-video asset")
        decode_path = remapped.get(asset.path, asset.path)
        with managed_extractor(
            extractor_factory, model_path=model_path
        ) as extractor:
            sequence = extract_video_sequence(
                decode_path, extractor, capture_factory=capture_factory,
                expected_metadata=asset.metadata,
            )
        yield asset, sequence


def load_arkit_trajectory(path: str | Path) -> ARKitSequence:
    checked = _require_regular_file(path, "ARKit trajectory")
    features: list[list[float]] = []
    timecodes: list[str] = []
    with checked.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = ("Timecode", "BlendshapeCount", *ARKIT_BLENDSHAPE_NAMES,
                    *ARKIT_ROTATION_NAMES)
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError("ARKit trajectory has a noncanonical ordered schema")
        for row in reader:
            try:
                declared = int(row["BlendshapeCount"])
                vector = [float(row[name]) for name in ARKIT_BLENDSHAPE_NAMES]
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError("ARKit trajectory has a malformed data row") from exc
            if declared < len(ARKIT_BLENDSHAPE_NAMES) or not np.isfinite(vector).all():
                raise ValueError("ARKit trajectory row is incomplete or non-finite")
            features.append(vector)
            timecodes.append(row["Timecode"])
    if not features:
        raise ValueError("ARKit trajectory has no usable rows")
    matrix = np.asarray(features, dtype=np.float32)
    indices, timestamps, _missing = _arkit_timeline(timecodes)
    return _validate_arkit_sequence(ARKitSequence(
        features=matrix,
        valid_mask=np.ones(len(matrix), dtype=bool),
        timestamps=timestamps,
        source_frame_indices=indices,
    ))


def _validate_arkit_sequence(sequence: ARKitSequence) -> ARKitSequence:
    if not isinstance(sequence, ARKitSequence):
        raise ValueError("ARKit sequence has the wrong type")
    features = np.asarray(sequence.features)
    mask = np.asarray(sequence.valid_mask)
    timestamps = np.asarray(sequence.timestamps)
    indices = np.asarray(sequence.source_frame_indices)
    n = features.shape[0] if features.ndim == 2 else -1
    if (
        features.dtype != np.float32 or features.shape != (n, 52) or n <= 0
        or mask.dtype != np.bool_ or mask.shape != (n,) or not mask.all()
        or timestamps.dtype != np.float64 or timestamps.shape != (n,)
        or indices.dtype != np.int64 or indices.shape != (n,)
        or not np.isfinite(features).all() or not np.isfinite(timestamps).all()
        or indices[0] != 0 or timestamps[0] != 0.0
        or (np.diff(indices) <= 0).any() or (np.diff(timestamps) <= 0).any()
        or sequence.source_fps != ARKIT_TIMECODE_FPS
        or sequence.timestamp_source != "arkit_original_timecode_relative_seconds"
    ):
        raise ValueError("ARKit cache does not satisfy its original-Timecode contract")
    if n > 1:
        expected_delta = np.diff(indices).astype(np.float64) / ARKIT_TIMECODE_FPS
        if not np.allclose(np.diff(timestamps), expected_delta, rtol=0.0,
                           atol=10.0 / (ARKIT_TIMECODE_FPS * 1000.0)):
            raise ValueError("ARKit timestamps disagree with their source-frame gaps")
    return sequence


def downsample_arkit_to_30hz(sequence: ARKitSequence) -> ARKitSSLView:
    sequence = _validate_arkit_sequence(sequence)
    selected_rows: list[int] = []
    target_indices: list[int] = []
    for row, raw_source_index in enumerate(sequence.source_frame_indices):
        source_index = int(raw_source_index)
        if source_index % 2 == 0:
            selected_rows.append(row)
            target_indices.append(source_index // 2)
    selected = np.asarray(selected_rows, dtype=np.int64)
    targets = np.asarray(target_indices, dtype=np.int64)
    if selected.size == 0:
        raise ValueError("ARKit 30-Hz exact-source selection produced no rows")
    mask = sequence.valid_mask[selected].copy()
    contiguous = np.zeros(len(selected), dtype=bool)
    if len(selected) > 1:
        contiguous[1:] = (np.diff(targets) == 1) & mask[:-1] & mask[1:]
    return ARKitSSLView(
        features=sequence.features[selected].copy(),
        valid_mask=mask,
        timestamps=sequence.timestamps[selected].copy(),
        source_frame_indices=sequence.source_frame_indices[selected].copy(),
        target_frame_indices=targets,
        contiguous_from_previous=contiguous,
    )


def _require_cache_identity(
    path: Path,
    recording_id: str,
    group_id: str,
    source_integrity_id: str,
    source_fingerprint: str,
) -> None:
    if _RECORDING_ID.fullmatch(recording_id) is None:
        raise ValueError("recording ID is not canonical")
    if _GROUP_ID.fullmatch(group_id) is None:
        raise ValueError("group ID is not canonical")
    if _INTEGRITY_ID.fullmatch(source_integrity_id) is None:
        raise ValueError("source integrity ID is not canonical")
    if _FINGERPRINT.fullmatch(source_fingerprint) is None:
        raise ValueError("source fingerprint is not canonical")
    if path.name != f"{recording_id}.npz":
        raise ValueError("cache filename must contain only the opaque recording ID")


def _write_npz_atomic(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    path = _lexical_absolute(path)
    _require_private_directory(path.parent, "compact cache parent")
    _assert_no_symlink_components(path.parent)
    temporary = path.parent / f".{path.stem}.tmp-{secrets.token_hex(8)}.npz"
    try:
        descriptor = _open_exclusive_private_file(
            temporary, "temporary compact cache",
        )
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        with np.load(temporary, allow_pickle=False) as loaded:
            if set(loaded.files) != set(payload):
                raise ValueError("reread compact cache has a different field set")
        os.replace(temporary, path)
        _require_private_regular_stat(
            os.stat(path, follow_symlinks=False), "committed compact cache",
        )
    finally:
        if temporary.exists():
            temporary.unlink()


def write_mediapipe_cache(
    path: str | Path,
    sequence: MayoMediaSequence,
    *,
    recording_id: str,
    group_id: str,
    source_integrity_id: str,
    source_fingerprint: str,
) -> None:
    path = _lexical_absolute(path)
    _require_cache_identity(
        path, recording_id, group_id, source_integrity_id, source_fingerprint
    )
    sequence = _validate_media_sequence(sequence)
    view = downsample_to_30hz(sequence)
    payload = {
        "features_source_rate": sequence.features,
        "valid_mask_source_rate": sequence.valid_mask,
        "timestamps_source_rate": sequence.timestamps,
        "source_frame_indices_source_rate": sequence.source_frame_indices,
        "facial_transforms_source_rate": sequence.facial_transforms,
        "facial_transform_mask_source_rate": sequence.facial_transform_mask,
        "features_30hz": view.features,
        "valid_mask_30hz": view.valid_mask,
        "timestamps_30hz": view.timestamps,
        "source_frame_indices_30hz": view.source_frame_indices,
        "target_frame_indices_30hz": view.target_frame_indices,
        "contiguous_from_previous_30hz": view.contiguous_from_previous,
        "facial_transforms_30hz": view.facial_transforms,
        "facial_transform_mask_30hz": view.facial_transform_mask,
        "feature_schema": np.asarray(DYNAMIC_FEATURE_SCHEMA),
        "feature_names": np.asarray(DYNAMIC_FEATURE_NAMES),
        "side_convention": np.asarray(CLINICAL_SIDE_CONVENTION),
        "capture_mirrored": np.asarray("unknown"),
        "normalization_transform": np.asarray(TRANSFORM_NORMALIZATION),
        "facial_transform_source": np.asarray(sequence.transform_source),
        "timestamp_unit": np.asarray("seconds"),
        "timestamp_source": np.asarray(sequence.timestamp_source),
        "source_fps": np.asarray(sequence.source_fps, dtype=np.float64),
        "producer_protocol": np.asarray(VIDEO_PRODUCER_PROTOCOL),
        "producer_adapter_version": np.asarray(VIDEO_ADAPTER_VERSION),
        "recording_id": np.asarray(recording_id),
        "group_id": np.asarray(group_id),
        "source_integrity_id": np.asarray(source_integrity_id),
        "source_fingerprint": np.asarray(source_fingerprint),
        "cache_schema": np.asarray(MEDIAPIPE_CACHE_SCHEMA),
        "development_only": np.asarray(True),
        "patient_identity": np.asarray("unknown"),
        "split_unit": np.asarray("recording"),
    }
    _write_npz_atomic(path, payload)


def write_arkit_cache(
    path: str | Path,
    sequence: ARKitSequence,
    *,
    recording_id: str,
    group_id: str,
    source_integrity_id: str,
    source_fingerprint: str,
) -> None:
    path = _lexical_absolute(path)
    _require_cache_identity(
        path, recording_id, group_id, source_integrity_id, source_fingerprint
    )
    sequence = _validate_arkit_sequence(sequence)
    view = downsample_arkit_to_30hz(sequence)
    payload = {
        "features_60hz": sequence.features,
        "valid_mask_60hz": sequence.valid_mask,
        "timestamps_60hz": sequence.timestamps,
        "source_frame_indices_60hz": sequence.source_frame_indices,
        "features_30hz": view.features,
        "valid_mask_30hz": view.valid_mask,
        "timestamps_30hz": view.timestamps,
        "source_frame_indices_30hz": view.source_frame_indices,
        "target_frame_indices_30hz": view.target_frame_indices,
        "contiguous_from_previous_30hz": view.contiguous_from_previous,
        "feature_schema": np.asarray("arkit_blendshapes_52_v1"),
        "feature_names": np.asarray(ARKIT_BLENDSHAPE_NAMES),
        "timestamp_unit": np.asarray("seconds"),
        "timestamp_source": np.asarray(sequence.timestamp_source),
        "recording_id": np.asarray(recording_id),
        "group_id": np.asarray(group_id),
        "source_integrity_id": np.asarray(source_integrity_id),
        "source_fingerprint": np.asarray(source_fingerprint),
        "cache_schema": np.asarray(ARKIT_CACHE_SCHEMA),
        "development_only": np.asarray(True),
        "patient_identity": np.asarray("unknown"),
        "split_unit": np.asarray("recording"),
    }
    _write_npz_atomic(path, payload)


def _dependency_contract(
    *,
    version_resolver: Callable[[str], str] = importlib.metadata.version,
) -> tuple[dict[str, str], dict[str, str]]:
    python_version = platform.python_version()
    if not python_version or python_version.lower() == "unknown":
        raise ValueError("Python runtime version is unavailable")

    def required(distribution: str) -> str:
        try:
            version = version_resolver(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError(f"required distribution {distribution} is unavailable") from exc
        if not isinstance(version, str) or not version or version.lower() == "unknown":
            raise ValueError(f"distribution {distribution} has no exact version")
        return version

    versions = {
        "python": f"python=={python_version}",
        "numpy": f"numpy=={required('numpy')}",
        "mediapipe": f"mediapipe=={required('mediapipe')}",
    }
    opencv: list[tuple[str, str]] = []
    for distribution in OPENCV_DISTRIBUTIONS:
        try:
            version = version_resolver(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
        if not isinstance(version, str) or not version or version.lower() == "unknown":
            raise ValueError("OpenCV distribution has no exact version")
        opencv.append((distribution, version))
    if len(opencv) != 1:
        raise ValueError("exactly one OpenCV wheel distribution must be installed")
    versions["opencv"] = f"{opencv[0][0]}=={opencv[0][1]}"
    distributions = {
        "numpy": "numpy",
        "mediapipe": "mediapipe",
        "opencv": opencv[0][0],
    }
    return versions, distributions


def _dependency_versions(
    *,
    version_resolver: Callable[[str], str] = importlib.metadata.version,
) -> dict[str, str]:
    return _dependency_contract(version_resolver=version_resolver)[0]


def _default_dependency_artifact_resolver(
    distribution_name: str,
) -> tuple[Path, Path]:
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError(
            f"required distribution {distribution_name} is unavailable"
        ) from exc
    metadata_root = getattr(distribution, "_path", None)
    if metadata_root is None:
        raise ValueError("distribution metadata location is unavailable")
    root = Path(metadata_root)
    return root / "METADATA", root / "RECORD"


def _dependency_artifact_rows(
    distributions: Mapping[str, str],
    *,
    dependency_artifact_resolver: Callable[[str], tuple[str | Path, str | Path]],
    python_executable: str | Path,
) -> tuple[DependencyFileSnapshot, ...]:
    configured_executable = _lexical_absolute(python_executable)
    executable_parent = configured_executable.parent
    runtime_prefix = (
        executable_parent.parent
        if executable_parent.name.lower() in {"bin", "scripts"}
        else executable_parent
    )
    runtime_prefix = _require_directory(runtime_prefix, "Python runtime prefix")
    executable = Path(python_executable).resolve(strict=True)
    executable = _require_regular_file(executable, "Python executable")
    executable_snapshot, _payload = _snapshot_dependency_file(
        executable,
        logical_name="python_executable",
        distribution="python",
        record_name="<resolved-python-executable>",
        capture_bytes=False,
    )
    rows: list[DependencyFileSnapshot] = [executable_snapshot]
    seen_targets: set[Path] = set()
    for logical_name, distribution_name in sorted(distributions.items()):
        artifacts = dependency_artifact_resolver(distribution_name)
        if not isinstance(artifacts, tuple) or len(artifacts) != 2:
            raise ValueError("dependency artifact resolver must return METADATA and RECORD")
        metadata = _require_regular_file(artifacts[0], f"{distribution_name} metadata")
        record = _require_regular_file(artifacts[1], f"{distribution_name} RECORD")
        if metadata.parent != record.parent:
            raise ValueError("distribution METADATA and RECORD must share one dist-info root")
        try:
            record_name = record.relative_to(runtime_prefix).as_posix()
        except ValueError as exc:
            raise ValueError("distribution RECORD escapes the exact Python runtime") from exc
        record_snapshot, record_payload = _snapshot_dependency_file(
            record,
            logical_name=f"{logical_name}_record",
            distribution=distribution_name,
            record_name=record_name,
            capture_bytes=True,
        )
        assert record_payload is not None
        entries = _parse_distribution_record(
            record_payload,
            record=record,
            runtime_prefix=runtime_prefix,
        )
        targets = {target for _name, target, _hash, _size in entries}
        if metadata not in targets or record not in targets:
            raise ValueError("distribution RECORD must close over METADATA and itself")
        for index, (listed_name, target, recorded_hash, recorded_size) in enumerate(entries):
            if target in seen_targets:
                raise ValueError("dependency RECORD closure contains a duplicate target")
            seen_targets.add(target)
            if target == record:
                snapshot = record_snapshot
            else:
                suffix = (
                    "metadata" if target == metadata else f"file_{index:05d}"
                )
                snapshot, _payload = _snapshot_dependency_file(
                    target,
                    logical_name=f"{logical_name}_{suffix}",
                    distribution=distribution_name,
                    record_name=listed_name,
                    capture_bytes=False,
                )
            if recorded_size is not None and snapshot.size != recorded_size:
                raise ValueError("installed dependency size disagrees with RECORD")
            if recorded_hash is not None and not hmac.compare_digest(
                snapshot.sha256, recorded_hash
            ):
                raise ValueError("installed dependency bytes disagree with RECORD")
            rows.append(snapshot)
    return tuple(rows)


def _snapshot_dependency_file(
    path: Path,
    *,
    logical_name: str,
    distribution: str,
    record_name: str,
    capture_bytes: bool,
) -> tuple[DependencyFileSnapshot, bytes | None]:
    """Hash one exact regular-file descriptor and bind it back to its path."""
    checked = _require_regular_file(path, f"{distribution} installed file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(checked, flags)
    collected = bytearray() if capture_bytes else None
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("installed dependency must be a regular file")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            if collected is not None:
                collected.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
    )
    identity_after = (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    )
    current = os.lstat(checked)
    identity_path = (
        current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns
    )
    if identity_before != identity_after or identity_after != identity_path:
        raise ValueError("installed dependency changed while it was fingerprinted")
    return DependencyFileSnapshot(
        logical_name=logical_name,
        distribution=distribution,
        record_name=record_name,
        path=checked,
        sha256=digest.hexdigest(),
        device=before.st_dev,
        inode=before.st_ino,
        size=before.st_size,
        mtime_ns=before.st_mtime_ns,
    ), (bytes(collected) if collected is not None else None)


def _parse_distribution_record(
    payload: bytes,
    *,
    record: Path,
    runtime_prefix: Path,
) -> tuple[tuple[str, Path, str | None, int | None], ...]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("dependency RECORD must be UTF-8") from exc
    base = record.parent.parent
    rows_by_target: dict[Path, tuple[str, Path, str | None, int | None]] = {}
    seen: dict[Path, tuple[str, str, str]] = {}
    try:
        reader = csv.reader(io.StringIO(text, newline=""))
        for row in reader:
            if len(row) != 3:
                raise ValueError("dependency RECORD row must have exactly three fields")
            listed_name, recorded_hash_text, recorded_size_text = row
            pure = PurePosixPath(listed_name)
            if (
                not listed_name or "\\" in listed_name or "\x00" in listed_name
                or pure.is_absolute() or any(part in {"", "."} for part in pure.parts)
            ):
                raise ValueError("dependency RECORD contains an unsafe installed path")
            target = _lexical_absolute(base.joinpath(*pure.parts))
            try:
                target.relative_to(runtime_prefix)
            except ValueError as exc:
                raise ValueError(
                    "dependency RECORD path escapes the exact Python runtime"
                ) from exc
            target = _require_regular_file(target, "dependency RECORD target")
            if bool(recorded_hash_text) != bool(recorded_size_text):
                raise ValueError("dependency RECORD hash and size must both be present or absent")
            recorded_hash: str | None = None
            recorded_size: int | None = None
            if recorded_hash_text:
                try:
                    algorithm, encoded = recorded_hash_text.split("=", 1)
                    if algorithm != "sha256" or not encoded:
                        raise ValueError
                    padding = "=" * ((4 - len(encoded) % 4) % 4)
                    decoded = base64.b64decode(
                        encoded + padding, altchars=b"-_", validate=True
                    )
                except (ValueError, binascii.Error) as exc:
                    raise ValueError("dependency RECORD hash is not canonical SHA-256") from exc
                if len(decoded) != hashlib.sha256().digest_size:
                    raise ValueError("dependency RECORD hash has the wrong length")
                if (
                    not recorded_size_text.isdigit()
                    or str(int(recorded_size_text)) != recorded_size_text
                ):
                    raise ValueError("dependency RECORD size is not canonical")
                recorded_hash = decoded.hex()
                recorded_size = int(recorded_size_text)
            prior = seen.get(target)
            raw_row = (listed_name, recorded_hash_text, recorded_size_text)
            if prior is not None:
                prior_name, prior_hash, prior_size = prior
                current_committed = bool(recorded_hash_text and recorded_size_text)
                prior_committed = bool(prior_hash and prior_size)
                # numpy 1.26.4 lists one identical canonical pycache name first
                # as an unhashed placeholder and then with a SHA/size.  Merge
                # only that strict strengthening pattern; every other repeated
                # target remains ambiguous and is rejected.
                mutable_bytecode = (
                    "__pycache__" in pure.parts and pure.suffix == ".pyc"
                )
                if (
                    prior_name == listed_name
                    and prior_committed != current_committed
                    and (not prior_committed or (not recorded_hash_text and not recorded_size_text))
                    and mutable_bytecode
                ):
                    # Installed bytecode can be regenerated by Python.  If the
                    # wheel explicitly lists both a blank mutable row and an
                    # install-time hash for the same pyc, retain the blank
                    # RECORD semantics but still snapshot the exact current
                    # bytes for the extraction lifetime.
                    if not prior_committed:
                        continue
                    # Replace a prior hashed row with the later blank row.
                else:
                    raise ValueError(
                        "dependency RECORD contains a duplicate normalized target"
                    )
            seen[target] = raw_row
            rows_by_target[target] = (
                listed_name, target, recorded_hash, recorded_size
            )
    except csv.Error as exc:
        raise ValueError("dependency RECORD is malformed") from exc
    if not rows_by_target:
        raise ValueError("dependency RECORD closure must be nonempty")
    return tuple(sorted(rows_by_target.values(), key=lambda item: item[0]))


def _artifact_aggregate(rows: Sequence[DependencyFileSnapshot]) -> str:
    aggregate = hashlib.sha256()
    for item in rows:
        aggregate.update(
            (
                f"{item.logical_name}:{item.distribution}:"
                f"{item.record_name}:{item.sha256}:{item.size}\n"
            ).encode("utf-8")
        )
    return aggregate.hexdigest()


def _public_dependency_provenance(
    snapshot: ProvenanceSnapshot,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for logical_name, requirement in sorted(snapshot.dependencies.items()):
        distribution = (
            "python" if logical_name == "python"
            else snapshot.dependency_distributions[logical_name]
        )
        prefix = (
            "python==" if logical_name == "python" else f"{distribution}=="
        )
        if not requirement.startswith(prefix) or len(requirement) == len(prefix):
            raise ValueError("dependency version requirement is noncanonical")
        files = tuple(
            item for item in snapshot.dependency_files
            if item.distribution == distribution
        )
        if not files:
            raise ValueError("dependency provenance has an empty installed-file closure")
        rows.append({
            "distribution": distribution,
            "version": requirement[len(prefix):],
            "installed_file_count": len(files),
            "installed_file_aggregate_sha256": _artifact_aggregate(files),
        })
    return rows


def snapshot_provenance(
    source_paths: Sequence[str | Path],
    model_path: str | Path,
    producer_paths: Mapping[str, str | Path],
    *,
    version_resolver: Callable[[str], str] = importlib.metadata.version,
    expected_source_hashes: Mapping[str | Path, str] | None = None,
    dependency_artifact_resolver: Callable[
        [str], tuple[str | Path, str | Path]
    ] = _default_dependency_artifact_resolver,
    python_executable: str | Path | None = None,
) -> ProvenanceSnapshot:
    if not source_paths or not producer_paths:
        raise ValueError("source and producer hash closures must be nonempty")
    expected = {
        _lexical_absolute(path): digest
        for path, digest in (expected_source_hashes or {}).items()
    }
    for digest in expected.values():
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError("expected source hash is not canonical")
    source_rows: list[tuple[Path, str]] = []
    for path in source_paths:
        checked = _require_regular_file(path, "source artifact")
        observed = sha256_file(checked)
        frozen = expected.get(checked)
        if frozen is not None and not hmac.compare_digest(observed, frozen):
            raise ValueError("source changed between inventory and provenance snapshot")
        source_rows.append((checked, observed))
    sources = tuple(sorted(source_rows, key=lambda item: str(item[0])))
    if set(expected) - {path for path, _digest in sources}:
        raise ValueError("expected source hash references an artifact outside the source closure")
    model = _require_regular_file(model_path, "MediaPipe model")
    producers: list[tuple[str, Path, str]] = []
    for name, path in sorted(producer_paths.items()):
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError("producer logical name is invalid")
        checked = _require_regular_file(path, "producer source")
        producers.append((name, checked, sha256_file(checked)))
    producer_aggregate = hashlib.sha256()
    for name, _path, digest in producers:
        producer_aggregate.update(f"{name}:{digest}\n".encode("ascii"))
    source_aggregate = hashlib.sha256()
    for _path, digest in sorted(sources, key=lambda item: item[1]):
        source_aggregate.update(f"{digest}\n".encode("ascii"))
    dependencies, dependency_distributions = _dependency_contract(
        version_resolver=version_resolver
    )
    dependency_files = _dependency_artifact_rows(
        dependency_distributions,
        dependency_artifact_resolver=dependency_artifact_resolver,
        python_executable=(sys.executable if python_executable is None else python_executable),
    )
    return ProvenanceSnapshot(
        source_files=sources,
        model_file=(model, sha256_file(model)),
        producer_files=tuple(producers),
        dependencies=dependencies,
        dependency_distributions=dependency_distributions,
        dependency_files=dependency_files,
        dependency_aggregate_sha256=_artifact_aggregate(dependency_files),
        producer_aggregate_sha256=producer_aggregate.hexdigest(),
        source_aggregate_sha256=source_aggregate.hexdigest(),
    )


def assert_provenance_unchanged(
    snapshot: ProvenanceSnapshot,
    *,
    version_resolver: Callable[[str], str] = importlib.metadata.version,
    dependency_artifact_resolver: Callable[
        [str], tuple[str | Path, str | Path]
    ] = _default_dependency_artifact_resolver,
    python_executable: str | Path | None = None,
) -> None:
    if not isinstance(snapshot, ProvenanceSnapshot):
        raise ValueError("provenance snapshot has the wrong type")
    observed_versions, observed_distributions = _dependency_contract(
        version_resolver=version_resolver
    )
    if (
        observed_versions != snapshot.dependencies
        or observed_distributions != snapshot.dependency_distributions
    ):
        raise ValueError("dependency versions or selected distributions changed")
    dependency_files = _dependency_artifact_rows(
        observed_distributions,
        dependency_artifact_resolver=dependency_artifact_resolver,
        python_executable=(sys.executable if python_executable is None else python_executable),
    )
    if dependency_files != snapshot.dependency_files:
        raise ValueError("dependency artifact path or fingerprint changed")
    if not hmac.compare_digest(
        _artifact_aggregate(dependency_files), snapshot.dependency_aggregate_sha256
    ):
        raise ValueError("dependency artifact aggregate changed")
    for path, expected in (*snapshot.source_files, snapshot.model_file):
        if not hmac.compare_digest(sha256_file(path), expected):
            raise ValueError("source or model changed before cache promotion")
    aggregate = hashlib.sha256()
    for name, path, expected in snapshot.producer_files:
        observed = sha256_file(path)
        if not hmac.compare_digest(observed, expected):
            raise ValueError("producer source changed before cache promotion")
        aggregate.update(f"{name}:{expected}\n".encode("ascii"))
    if not hmac.compare_digest(aggregate.hexdigest(), snapshot.producer_aggregate_sha256):
        raise ValueError("producer aggregate changed before cache promotion")
    source_aggregate = hashlib.sha256()
    for _path, digest in sorted(snapshot.source_files, key=lambda item: item[1]):
        source_aggregate.update(f"{digest}\n".encode("ascii"))
    if not hmac.compare_digest(
        source_aggregate.hexdigest(), snapshot.source_aggregate_sha256
    ):
        raise ValueError("source aggregate changed before cache promotion")


@contextmanager
def managed_extractor(
    extractor_factory: Callable[..., MayoVideoClinical23Extractor],
    *,
    model_path: str | Path,
):
    extractor = extractor_factory(model_path=model_path)
    try:
        yield extractor
    finally:
        extractor.close()


@contextmanager
def output_parent_lock(
    output_root: str | Path,
    *,
    create_if_missing: bool = True,
):
    output = _lexical_absolute(output_root)
    parent = _require_private_directory(
        _assert_no_symlink_components(output.parent), "output parent",
    )
    parent_info = os.lstat(parent)
    parent_identity = (
        int(parent_info.st_dev), int(parent_info.st_ino), int(parent_info.st_mode),
        int(parent_info.st_uid), int(parent_info.st_gid),
    )
    lock_path = parent / f".{output.name}.lock"
    base_flags = (
        os.O_RDWR if create_if_missing else os.O_RDONLY
    ) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    created = False
    if create_if_missing:
        try:
            fd = os.open(lock_path, base_flags | os.O_CREAT | os.O_EXCL, 0o600)
            created = True
        except FileExistsError:
            fd = os.open(lock_path, base_flags)
    else:
        try:
            fd = os.open(lock_path, base_flags)
        except FileNotFoundError as exc:
            raise ValueError(
                "committed authorization requires the existing output lock"
            ) from exc
    acquired = registered = False
    try:
        if created:
            os.fchmod(fd, 0o600)
        info = os.fstat(fd)
        _require_private_regular_stat(info, "output lock")
        current = os.stat(lock_path, follow_symlinks=False)
        lock_identity = (
            info.st_dev, info.st_ino, info.st_mode, info.st_uid,
            info.st_nlink,
        )
        if lock_identity != (
            current.st_dev, current.st_ino, current.st_mode, current.st_uid,
            current.st_nlink,
        ):
            raise ValueError("output lock path identity changed during open")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another Mayo SSL builder holds the output lock") from exc
        acquired = True
        after_acquire = os.fstat(fd)
        linked_after_acquire = os.stat(lock_path, follow_symlinks=False)
        _require_private_regular_stat(after_acquire, "output lock")
        _require_private_regular_stat(linked_after_acquire, "output lock")
        if lock_identity != (
            after_acquire.st_dev, after_acquire.st_ino, after_acquire.st_mode,
            after_acquire.st_uid, after_acquire.st_nlink,
        ) or lock_identity != (
            linked_after_acquire.st_dev, linked_after_acquire.st_ino,
            linked_after_acquire.st_mode, linked_after_acquire.st_uid,
            linked_after_acquire.st_nlink,
        ):
            raise ValueError("output lock changed while flock was acquired")
        parent_after_acquire = os.lstat(parent)
        _require_private_directory_stat(parent_after_acquire, "output parent")
        if (
            int(parent_after_acquire.st_dev), int(parent_after_acquire.st_ino),
            int(parent_after_acquire.st_mode), int(parent_after_acquire.st_uid),
            int(parent_after_acquire.st_gid),
        ) != parent_identity:
            raise ValueError("output parent changed while flock was acquired")
        if output in _HELD_OUTPUT_LOCKS:
            raise RuntimeError("output lock is already held in this process")
        _HELD_OUTPUT_LOCKS.add(output)
        registered = True
        yield
        parent_after = os.lstat(parent)
        _require_private_directory_stat(parent_after, "output parent")
        if (
            int(parent_after.st_dev), int(parent_after.st_ino),
            int(parent_after.st_mode), int(parent_after.st_uid),
            int(parent_after.st_gid),
        ) != parent_identity:
            raise ValueError("output parent changed while its lock was held")
        after = os.fstat(fd)
        current = os.stat(lock_path, follow_symlinks=False)
        _require_private_regular_stat(after, "output lock")
        _require_private_regular_stat(current, "output lock")
        if lock_identity != (
            after.st_dev, after.st_ino, after.st_mode, after.st_uid,
            after.st_nlink,
        ) or lock_identity != (
            current.st_dev, current.st_ino, current.st_mode, current.st_uid,
            current.st_nlink,
        ):
            raise ValueError("output lock changed while held")
    finally:
        try:
            if registered:
                _HELD_OUTPUT_LOCKS.discard(output)
        finally:
            try:
                if acquired:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


def _require_output_lock(output: Path) -> None:
    if output not in _HELD_OUTPUT_LOCKS:
        raise RuntimeError("output lifecycle mutation requires the exclusive lock")


def _remove_real_tree(path: Path) -> None:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("generation cleanup target must be a real directory")
    shutil.rmtree(path)


def _fsync_directory(path: Path) -> None:
    checked = _require_directory(path, "directory to fsync")
    flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(checked, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_generation_tree(staging: Path) -> None:
    for dirname in ("mediapipe", "arkit"):
        _fsync_directory(staging / dirname)
    _fsync_directory(staging)


def _journal_path(output: Path) -> Path:
    return output.parent / f".{output.name}.transaction.json"


def _decode_unique_json_object(payload: bytes, field: str) -> dict[str, object]:
    """Decode one UTF-8 JSON object while rejecting duplicate keys recursively."""
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{field} repeats JSON key {key!r}")
            value[key] = item
        return value

    try:
        decoded = payload.decode("utf-8")
        value = json.loads(decoded, object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field} must contain a JSON object")
    return value


def _write_transaction_journal(path: Path, payload: Mapping[str, object]) -> None:
    _require_private_directory(path.parent, "Mayo transaction journal parent")
    temporary = path.parent / f".{path.name}.tmp-{secrets.token_hex(8)}"
    serialized = json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists() and not _is_symlink(temporary):
            temporary.unlink()


def _validate_transaction_journal_payload(
    raw_payload: bytes,
) -> dict[str, object]:
    try:
        payload = _decode_unique_json_object(
            raw_payload, "Mayo transaction journal"
        )
    except (OSError, ValueError) as exc:
        raise ValueError("Mayo transaction journal is invalid") from exc
    required = {
        "schema", "token", "staging_name", "exposure_name",
        "had_output", "had_exposure", "phase", "generation_commitment",
        "indeterminate", "previous_output_storage_commitment",
        "previous_exposure_storage_commitment",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("Mayo transaction journal has a noncanonical schema")
    if (
        payload["schema"] != "mayo_cache_exposure_transaction_v3"
        or not isinstance(payload["token"], str)
        or re.fullmatch(r"[0-9a-f]{16}", payload["token"]) is None
        or not isinstance(payload["staging_name"], str)
        or not isinstance(payload["exposure_name"], str)
        or not isinstance(payload["had_output"], bool)
        or not isinstance(payload["had_exposure"], bool)
        or not isinstance(payload["indeterminate"], bool)
        or payload["phase"] not in {
            "prepared", "moving_old_output", "old_output_moved",
            "moving_old_exposure", "old_exposure_moved",
            "installing_new_output", "new_output_installed",
            "installing_new_exposure", "new_exposure_installed", "committed",
        }
    ):
        raise ValueError("Mayo transaction journal contains invalid values")
    output_commitment = payload["previous_output_storage_commitment"]
    exposure_commitment = payload["previous_exposure_storage_commitment"]
    if (
        (payload["had_output"] is True) != (
            isinstance(output_commitment, str)
            and re.fullmatch(r"[0-9a-f]{64}", output_commitment) is not None
        )
        or (payload["had_exposure"] is True) != (
            isinstance(exposure_commitment, str)
            and re.fullmatch(r"[0-9a-f]{64}", exposure_commitment) is not None
        )
        or (payload["had_output"] is False and output_commitment is not None)
        or (payload["had_exposure"] is False and exposure_commitment is not None)
    ):
        raise ValueError("Mayo transaction journal storage closure is invalid")
    payload["generation_commitment"] = _validate_generation_commitment(
        payload["generation_commitment"]
    )
    return payload


def _assert_held_transaction_journal(
    descriptor: int,
    path: Path,
    identity: tuple[int, ...],
) -> None:
    opened = os.fstat(descriptor)
    linked = os.lstat(path)
    _require_private_regular_stat(opened, "Mayo transaction journal")
    _require_private_regular_stat(linked, "Mayo transaction journal")
    if (
        _regular_snapshot(opened) != identity
        or _regular_snapshot(linked) != identity
    ):
        raise ValueError("Mayo transaction journal changed while held")


def _open_transaction_journal(
    path: Path,
) -> tuple[int, tuple[int, ...], dict[str, object]]:
    checked = _require_regular_file(path, "Mayo transaction journal")
    descriptor = os.open(
        checked,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        identity = _regular_snapshot(os.fstat(descriptor))
        _assert_held_transaction_journal(descriptor, checked, identity)
        size = int(os.fstat(descriptor).st_size)
        if size < 1 or size > _MAX_MAYO_MANIFEST_BYTES:
            raise ValueError("Mayo transaction journal exceeds its byte limit")
        payload = bytearray()
        while len(payload) < size:
            chunk = os.read(descriptor, min(1024 * 1024, size - len(payload)))
            if not chunk:
                raise ValueError("Mayo transaction journal is truncated")
            payload.extend(chunk)
        if os.read(descriptor, 1):
            raise ValueError("Mayo transaction journal exceeds its byte limit")
        _assert_held_transaction_journal(descriptor, checked, identity)
        return descriptor, identity, _validate_transaction_journal_payload(
            bytes(payload)
        )
    except BaseException:
        os.close(descriptor)
        raise


def _assert_unlinked_transaction_journal(
    descriptor: int,
    identity: tuple[int, ...],
) -> None:
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISREG(opened.st_mode)
        or stat.S_IMODE(opened.st_mode) != 0o600
        or int(opened.st_uid) != os.geteuid()
        or int(opened.st_nlink) != 0
        or (
            int(opened.st_dev), int(opened.st_ino), int(opened.st_mode),
            int(opened.st_uid), int(opened.st_gid), int(opened.st_size),
            int(opened.st_mtime_ns),
        ) != (
            identity[0], identity[1], identity[2], identity[3],
            identity[4], identity[6], identity[7],
        )
    ):
        raise ValueError("the held Mayo transaction journal was not unlinked")


@dataclass
class _JournalCleanupState:
    compensation_attempted: bool = False


def _unlink_held_transaction_journal_durably(
    *,
    descriptor: int,
    identity: tuple[int, ...],
    path: Path,
    journal: Mapping[str, object],
    validate_final_state: Callable[[], None],
    fsync_directories: tuple[Path, ...],
    cleanup_state: _JournalCleanupState,
) -> None:
    if (
        not isinstance(cleanup_state, _JournalCleanupState)
        or cleanup_state.compensation_attempted
    ):
        raise ValueError("Mayo journal cleanup state is invalid")
    _assert_held_transaction_journal(descriptor, path, identity)
    validate_final_state()
    unlinked = False

    def require_journal_absent() -> None:
        if path.exists() or _is_symlink(path):
            raise ValueError("Mayo transaction journal path reappeared after unlink")

    try:
        for directory in fsync_directories:
            _fsync_directory(directory)
            _assert_held_transaction_journal(descriptor, path, identity)
            validate_final_state()
        os.unlink(path)
        unlinked = True
        _assert_unlinked_transaction_journal(descriptor, identity)
        require_journal_absent()
        validate_final_state()
        require_journal_absent()
        _fsync_directory(path.parent)
        require_journal_absent()
        validate_final_state()
        require_journal_absent()
    except BaseException as primary:
        if unlinked:
            cleanup_state.compensation_attempted = True
            try:
                if not path.exists() and not _is_symlink(path):
                    _write_transaction_journal(path, journal)
            except BaseException as restoration_error:
                raise primary from restoration_error
        raise


def _load_transaction_journal(path: Path) -> dict[str, object]:
    descriptor, _identity, payload = _open_transaction_journal(path)
    try:
        return payload
    finally:
        os.close(descriptor)


def _unlink_regular_if_present(path: Path, field: str) -> None:
    if path.exists() or _is_symlink(path):
        _require_regular_file(path, field).unlink()


def _recover_cache_exposure_transaction_held(
    output: Path,
    exposure: Path,
    *,
    journal_path: Path,
    journal_descriptor: int,
    journal_identity: tuple[int, ...],
    journal: dict[str, object],
    cleanup_state: _JournalCleanupState,
    salt: bytes | None = None,
    expected_inventory_counts: Mapping[str, object] | None = None,
    expected_collection_classification_integrity_id: str | None = None,
    expected_classification_integrity_id: str | None = None,
    allow_indeterminate: bool = False,
) -> None:
    if bool(journal["indeterminate"]) and not allow_indeterminate:
        raise RuntimeError(
            "Mayo transaction is retained for explicit offline review"
        )
    if journal["exposure_name"] != exposure.name:
        raise ValueError("transaction journal targets a different exposure manifest")
    staging_name = str(journal["staging_name"])
    if not staging_name.startswith(f".{output.name}.staging-") or "/" in staging_name:
        raise ValueError("transaction journal staging name is unsafe")
    token = str(journal["token"])
    staging = output.parent / staging_name
    output_backup = output.parent / f".{output.name}.backup-{token}"
    exposure_backup = exposure.parent / f".{exposure.name}.backup-{token}"
    exposure_temporary = exposure.parent / f".{exposure.name}.tmp-{token}"
    committed = journal["phase"] == "committed"
    _assert_held_transaction_journal(
        journal_descriptor, journal_path, journal_identity,
    )

    def present(path: Path) -> bool:
        return path.exists() or _is_symlink(path)

    def private_regular_snapshot(
        path: Path,
        field: str,
    ) -> tuple[tuple[int, ...], tuple[int, ...], str]:
        checked = _require_regular_file(path, field)
        before = os.lstat(checked)
        _require_private_regular_stat(before, field)
        _payload, digest, _size = _read_regular_bytes(
            checked, field, max_bytes=_MAX_MAYO_MANIFEST_BYTES,
        )
        after = os.lstat(checked)
        _require_private_regular_stat(after, field)
        if _regular_snapshot(before) != _regular_snapshot(after):
            raise ValueError(f"{field} changed while it was snapshotted")
        return (
            _regular_snapshot(after),
            _movement_stable_regular_snapshot(after),
            digest,
        )

    def require_private_tree_snapshot(
        path: Path,
        expected: tuple[tuple[str, tuple[str, ...], tuple[int, ...]], ...],
        field: str,
    ) -> None:
        _root, observed = _private_generation_storage_ledger(path, field)
        if observed != expected:
            raise ValueError(f"{field} changed after recovery preflight")

    def require_private_regular_snapshot(
        path: Path,
        expected_stable: tuple[int, ...],
        expected_digest: str,
        field: str,
        *,
        require_full: tuple[int, ...] | None = None,
    ) -> None:
        observed_full, observed_stable, observed_digest = (
            private_regular_snapshot(path, field)
        )
        if (
            observed_stable != expected_stable
            or not hmac.compare_digest(observed_digest, expected_digest)
            or (require_full is not None and observed_full != require_full)
        ):
            raise ValueError(f"{field} changed after recovery preflight")

    had_output = bool(journal["had_output"])
    had_exposure = bool(journal["had_exposure"])
    if committed:
        _assert_held_transaction_journal(
            journal_descriptor, journal_path, journal_identity,
        )
        _require_directory(output, "committed output generation")
        _require_regular_file(exposure, "committed exposure manifest")
        if present(staging) or present(exposure_temporary):
            raise ValueError("committed recovery has unexpected staging residue")
    else:
        phase = str(journal["phase"])
        observed = (
            present(output), present(output_backup),
            present(exposure), present(exposure_backup), present(staging),
        )
        prepared = (had_output, False, had_exposure, False, True)
        old_output_moved = (False, True, had_exposure, False, True)
        before_exposure_move = (False, had_output, had_exposure, False, True)
        old_exposure_moved = (False, had_output, False, True, True)
        before_output_install = (False, had_output, False, had_exposure, True)
        new_output_installed = (True, had_output, False, had_exposure, False)
        new_exposure_installed = (True, had_output, True, had_exposure, False)
        allowed_topologies: dict[str, tuple[tuple[bool, ...], ...]] = {
            "prepared": (prepared,),
            "moving_old_output": (
                (prepared, old_output_moved) if had_output else ()
            ),
            "old_output_moved": (
                (old_output_moved,) if had_output else ()
            ),
            "moving_old_exposure": (
                (before_exposure_move, old_exposure_moved)
                if had_exposure else ()
            ),
            "old_exposure_moved": (
                (old_exposure_moved,) if had_exposure else ()
            ),
            "installing_new_output": (
                before_output_install, new_output_installed,
            ),
            "new_output_installed": (new_output_installed,),
            "installing_new_exposure": (
                new_output_installed, new_exposure_installed,
            ),
            "new_exposure_installed": (new_exposure_installed,),
        }
        expected = allowed_topologies.get(phase, ())
        if observed not in expected:
            raise ValueError("transaction storage topology does not match its journal")
        temporary_present = present(exposure_temporary)
        if phase == "prepared":
            pass
        elif phase == "installing_new_exposure":
            if temporary_present == present(exposure):
                raise ValueError(
                    "transaction exposure topology does not match its journal"
                )
        elif phase == "new_exposure_installed":
            if temporary_present:
                raise ValueError(
                    "transaction exposure topology does not match its journal"
                )
        elif not temporary_present:
            raise ValueError(
                "transaction exposure topology does not match its journal"
            )

        output_source = output_backup if present(output_backup) else output
        exposure_source = (
            exposure_backup if present(exposure_backup) else exposure
        )
        output_ledger = None
        output_storage_commitment = None
        if had_output:
            (
                _output_source,
                output_ledger,
                output_storage_commitment,
            ) = _private_generation_storage_commitment(
                output_source, "recoverable previous output generation",
            )
            if not hmac.compare_digest(
                output_storage_commitment,
                str(journal["previous_output_storage_commitment"]),
            ):
                raise ValueError(
                    "recoverable previous output does not match its journal"
                )
        exposure_full = None
        exposure_stable = None
        exposure_digest = None
        if had_exposure:
            exposure_full, exposure_stable, exposure_digest = (
                private_regular_snapshot(
                    exposure_source, "recoverable previous exposure manifest",
                )
            )
            exposure_storage_commitment = _private_regular_storage_commitment(
                exposure_stable, exposure_digest,
            )
            if not hmac.compare_digest(
                exposure_storage_commitment,
                str(journal["previous_exposure_storage_commitment"]),
            ):
                raise ValueError(
                    "recoverable previous exposure does not match its journal"
                )

        if output_ledger is not None:
            require_private_tree_snapshot(
                output_source, output_ledger,
                "recoverable previous output generation",
            )
        if exposure_stable is not None and exposure_digest is not None:
            require_private_regular_snapshot(
                exposure_source, exposure_stable, exposure_digest,
                "recoverable previous exposure manifest",
                require_full=exposure_full,
            )
        _assert_held_transaction_journal(
            journal_descriptor, journal_path, journal_identity,
        )

        new_output_is_canonical = (
            phase in {
                "installing_new_output", "new_output_installed",
                "installing_new_exposure", "new_exposure_installed",
            }
            and present(output)
            and not present(staging)
        )
        new_exposure_is_canonical = (
            phase in {"installing_new_exposure", "new_exposure_installed"}
            and present(exposure)
            and not present(exposure_temporary)
        )
        new_output_path = output if new_output_is_canonical else staging
        if new_exposure_is_canonical:
            new_exposure_path = exposure
        elif present(exposure_temporary):
            new_exposure_path = exposure_temporary
        else:
            new_exposure_path = (
                new_output_path / "mayo_exposure_manifest.json"
            )
        expected_generation = _validate_generation_commitment(
            journal["generation_commitment"]
        )
        with _hold_committed_mayo_generation(
            new_output_path,
            new_exposure_path,
            media_count=int(expected_generation["mediapipe_file_count"]),
            arkit_count=int(expected_generation["arkit_file_count"]),
        ) as held_new_generation:
            _assert_committed_generation(
                new_output_path,
                new_exposure_path,
                expected_generation,
                salt=salt,
                expected_inventory_counts=expected_inventory_counts,
                expected_collection_classification_integrity_id=(
                    expected_collection_classification_integrity_id
                ),
                expected_classification_integrity_id=(
                    expected_classification_integrity_id
                ),
                _held=held_new_generation,
            )
            new_output_identity = held_new_generation.output_identity
            new_exposure_identity = (
                held_new_generation.external_exposure_identity[:7]
            )

        if new_output_is_canonical:
            _publish_private_path_no_replace(
                output,
                staging,
                "interrupted new Mayo output generation",
                expected_identity=new_output_identity,
            )
        if new_exposure_is_canonical:
            _publish_private_path_no_replace(
                exposure,
                exposure_temporary,
                "interrupted new Mayo exposure manifest",
                expected_identity=new_exposure_identity,
            )

        if had_output:
            if present(output_backup):
                _require_directory(output_backup, "interrupted output backup")
                if present(output):
                    raise ValueError(
                        "output destination is occupied before recovery restore"
                    )
                _publish_private_path_no_replace(
                    output_backup,
                    output,
                    "restored previous output generation",
                    expected_identity=output_ledger[0][2],
                )
            else:
                _require_directory(output, "unmoved previous output generation")
            require_private_tree_snapshot(
                output, output_ledger,
                "restored previous output generation",
            )
        elif present(output):
            _remove_real_tree(output)

        if had_exposure:
            if present(exposure_backup):
                _require_regular_file(exposure_backup, "interrupted exposure backup")
                if present(exposure):
                    raise ValueError(
                        "exposure destination is occupied before recovery restore"
                    )
                _publish_private_path_no_replace(
                    exposure_backup,
                    exposure,
                    "restored previous exposure manifest",
                    expected_identity=exposure_stable,
                )
            else:
                _require_regular_file(exposure, "unmoved previous exposure manifest")
            require_private_regular_snapshot(
                exposure, exposure_stable, exposure_digest,
                "restored previous exposure manifest",
            )
        else:
            _unlink_regular_if_present(exposure, "interrupted new exposure")

        if output_ledger is not None:
            require_private_tree_snapshot(
                output, output_ledger,
                "restored previous output generation",
            )
        elif present(output):
            raise ValueError("recovery created an unexpected output generation")
        if exposure_stable is not None and exposure_digest is not None:
            require_private_regular_snapshot(
                exposure, exposure_stable, exposure_digest,
                "restored previous exposure manifest",
            )
        elif present(exposure):
            raise ValueError("recovery created an unexpected exposure manifest")

    final_holds = ExitStack()
    try:
        held_committed = None
        held_output = None
        held_exposure = None
        if committed:
            expected_generation = _validate_generation_commitment(
                journal["generation_commitment"]
            )
            held_committed = final_holds.enter_context(
                _hold_committed_mayo_generation(
                    output,
                    exposure,
                    media_count=int(
                        expected_generation["mediapipe_file_count"]
                    ),
                    arkit_count=int(expected_generation["arkit_file_count"]),
                )
            )
            _assert_committed_generation(
                output,
                exposure,
                expected_generation,
                salt=salt,
                expected_inventory_counts=expected_inventory_counts,
                expected_collection_classification_integrity_id=(
                    expected_collection_classification_integrity_id
                ),
                expected_classification_integrity_id=(
                    expected_classification_integrity_id
                ),
                _held=held_committed,
            )
            held_output_backup = None
            if present(output_backup):
                if not had_output:
                    raise ValueError(
                        "committed recovery has an unexpected output backup"
                    )
                held_output_backup = final_holds.enter_context(
                    _hold_private_storage_tree(
                        output_backup, "committed recovery output backup",
                    )
                )
                _ledger, observed_commitment = (
                    _held_private_generation_storage_commitment(
                        held_output_backup,
                        "committed recovery output backup",
                    )
                )
                if not hmac.compare_digest(
                    observed_commitment,
                    str(journal["previous_output_storage_commitment"]),
                ):
                    raise ValueError(
                        "committed recovery output backup changed"
                    )
            held_exposure_backup = None
            if present(exposure_backup):
                if not had_exposure:
                    raise ValueError(
                        "committed recovery has an unexpected exposure backup"
                    )
                held_exposure_backup = final_holds.enter_context(
                    _hold_private_regular_storage(
                        exposure_backup, "committed recovery exposure backup",
                    )
                )
                if not hmac.compare_digest(
                    _held_private_regular_storage_commitment(
                        held_exposure_backup,
                        "committed recovery exposure backup",
                    ),
                    str(journal["previous_exposure_storage_commitment"]),
                ):
                    raise ValueError(
                        "committed recovery exposure backup changed"
                    )
        else:
            final_new_exposure_path = (
                exposure_temporary
                if present(exposure_temporary)
                else staging / "mayo_exposure_manifest.json"
            )
            held_interrupted_generation = final_holds.enter_context(
                _hold_committed_mayo_generation(
                    staging,
                    final_new_exposure_path,
                    media_count=int(
                        expected_generation["mediapipe_file_count"]
                    ),
                    arkit_count=int(expected_generation["arkit_file_count"]),
                    assert_on_exit=False,
                )
            )
            _assert_committed_generation(
                staging,
                final_new_exposure_path,
                expected_generation,
                salt=salt,
                expected_inventory_counts=expected_inventory_counts,
                expected_collection_classification_integrity_id=(
                    expected_collection_classification_integrity_id
                ),
                expected_classification_integrity_id=(
                    expected_classification_integrity_id
                ),
                _held=held_interrupted_generation,
            )
            if output_ledger is not None:
                held_output = final_holds.enter_context(
                    _hold_private_storage_tree(
                        output, "final recovered output generation",
                    )
                )
            if exposure_stable is not None and exposure_digest is not None:
                held_exposure = final_holds.enter_context(
                    _hold_private_regular_storage(
                        exposure, "final recovered exposure manifest",
                    )
                )

        if not committed:
            _assert_held_mayo_generation(held_interrupted_generation)
        _unlink_regular_if_present(
            exposure_temporary, "interrupted exposure temporary",
        )
        if staging.exists() or _is_symlink(staging):
            _remove_real_tree(staging)
        _remove_real_tree(output_backup)
        _unlink_regular_if_present(exposure_backup, "stale exposure backup")

        def validate_final_state() -> None:
            if not committed:
                if output_ledger is not None:
                    require_private_tree_snapshot(
                        output, output_ledger,
                        "final recovered output generation",
                    )
                    _assert_held_private_storage_tree(
                        held_output, "final recovered output generation",
                    )
                elif present(output):
                    raise ValueError(
                        "recovery retained an unexpected output generation"
                    )
                if exposure_stable is not None and exposure_digest is not None:
                    require_private_regular_snapshot(
                        exposure, exposure_stable, exposure_digest,
                        "final recovered exposure manifest",
                    )
                    _assert_held_private_regular_storage(
                        held_exposure, "final recovered exposure manifest",
                    )
                elif present(exposure):
                    raise ValueError(
                        "recovery retained an unexpected exposure manifest"
                    )
            else:
                _assert_held_mayo_generation(held_committed)
                if (
                    present(staging) or present(output_backup)
                    or present(exposure_backup) or present(exposure_temporary)
                ):
                    raise ValueError(
                        "committed recovery cleanup residue reappeared"
                    )

        fsync_directories = tuple(dict.fromkeys((
            output.parent, exposure.parent,
        )))
        _unlink_held_transaction_journal_durably(
            descriptor=journal_descriptor,
            identity=journal_identity,
            path=journal_path,
            journal=journal,
            validate_final_state=validate_final_state,
            fsync_directories=fsync_directories,
            cleanup_state=cleanup_state,
        )
    finally:
        final_holds.__exit__(*sys.exc_info())


def _recover_cache_exposure_transaction(
    output: Path,
    exposure: Path,
    *,
    salt: bytes | None = None,
    expected_inventory_counts: Mapping[str, object] | None = None,
    expected_collection_classification_integrity_id: str | None = None,
    expected_classification_integrity_id: str | None = None,
    allow_indeterminate: bool = False,
) -> None:
    if type(allow_indeterminate) is not bool:
        raise ValueError("indeterminate recovery authority must be boolean")
    journal_path = _journal_path(output)
    if not journal_path.exists() and not _is_symlink(journal_path):
        return
    descriptor, identity, journal = _open_transaction_journal(journal_path)
    cleanup_state = _JournalCleanupState()
    primary: BaseException | None = None
    try:
        _recover_cache_exposure_transaction_held(
            output,
            exposure,
            journal_path=journal_path,
            journal_descriptor=descriptor,
            journal_identity=identity,
            journal=journal,
            cleanup_state=cleanup_state,
            salt=salt,
            expected_inventory_counts=expected_inventory_counts,
            expected_collection_classification_integrity_id=(
                expected_collection_classification_integrity_id
            ),
            expected_classification_integrity_id=(
                expected_classification_integrity_id
            ),
            allow_indeterminate=allow_indeterminate,
        )
    except BaseException as exc:
        primary = exc
    close_error: BaseException | None = None
    try:
        os.close(descriptor)
    except BaseException as exc:
        close_error = exc
    if primary is not None or close_error is not None:
        compensation_failed = cleanup_state.compensation_attempted
        if (
            not compensation_failed
            and not journal_path.exists()
            and not _is_symlink(journal_path)
        ):
            try:
                _write_transaction_journal(journal_path, journal)
            except BaseException as restoration_error:
                if primary is not None:
                    raise primary.with_traceback(primary.__traceback__) from restoration_error
                raise close_error from restoration_error
        if primary is not None:
            if close_error is not None:
                raise primary.with_traceback(primary.__traceback__) from close_error
            raise primary.with_traceback(primary.__traceback__)
        raise close_error


def recover_interrupted_generations(
    output_root: str | Path,
    *,
    exposure_manifest_path: str | Path | None = None,
    salt: bytes | None = None,
    expected_inventory_counts: Mapping[str, object] | None = None,
    expected_collection_classification_integrity_id: str | None = None,
    expected_classification_integrity_id: str | None = None,
) -> None:
    output = _lexical_absolute(output_root)
    _require_output_lock(output)
    parent = _assert_no_symlink_components(output.parent)
    journal = _journal_path(output)
    if journal.exists() or _is_symlink(journal):
        if exposure_manifest_path is None:
            raise ValueError("exposure path is required to recover a coupled transaction")
        _recover_cache_exposure_transaction(
            output, _lexical_absolute(exposure_manifest_path),
            salt=salt,
            expected_inventory_counts=expected_inventory_counts,
            expected_collection_classification_integrity_id=(
                expected_collection_classification_integrity_id
            ),
            expected_classification_integrity_id=(
                expected_classification_integrity_id
            ),
        )
    staging = sorted(parent.glob(f".{output.name}.staging-*"), key=lambda item: item.name)
    backups = sorted(parent.glob(f".{output.name}.backup-*"), key=lambda item: item.name)
    for candidate in (*staging, *backups):
        if _is_symlink(candidate) or not candidate.is_dir():
            raise ValueError("interrupted generation candidate must be a real directory")
    for candidate in staging:
        _remove_real_tree(candidate)
    if output.exists():
        _require_directory(output, "existing output generation")
        for backup in backups:
            _remove_real_tree(backup)
    elif len(backups) == 1:
        os.replace(backups[0], output)
    elif len(backups) > 1:
        raise ValueError("multiple interrupted backups are ambiguous")


def promote_generation(
    staging_root: str | Path,
    output_root: str | Path,
    *,
    exposure_manifest_path: str | Path | None = None,
    replace_func: Callable[[str | Path, str | Path], None] = os.replace,
    phase_hook: Callable[[str], None] | None = None,
    continuity_validator: Callable[[], None] | None = None,
    salt: bytes | None = None,
    expected_inventory_counts: Mapping[str, object] | None = None,
    expected_collection_classification_integrity_id: str | None = None,
    expected_classification_integrity_id: str | None = None,
) -> None:
    staging = _require_directory(staging_root, "staging generation")
    output = _lexical_absolute(output_root)
    _require_output_lock(output)
    if staging.parent != output.parent:
        raise ValueError("staging generation must be a sibling of output")
    if exposure_manifest_path is not None:
        _promote_generation_with_exposure(
            staging,
            output,
            _lexical_absolute(exposure_manifest_path),
            replace_func=replace_func,
            phase_hook=phase_hook,
            continuity_validator=continuity_validator,
            salt=salt,
            expected_inventory_counts=expected_inventory_counts,
            expected_collection_classification_integrity_id=(
                expected_collection_classification_integrity_id
            ),
            expected_classification_integrity_id=(
                expected_classification_integrity_id
            ),
        )
        return
    backup = output.parent / f".{output.name}.backup-{secrets.token_hex(8)}"
    old_moved = False
    new_installed = False
    try:
        if continuity_validator is not None:
            continuity_validator()
        if output.exists():
            _require_directory(output, "previous output generation")
            replace_func(output, backup)
            old_moved = True
        try:
            replace_func(staging, output)
            new_installed = True
            if continuity_validator is not None:
                continuity_validator()
        except BaseException:
            if new_installed and (output.exists() or _is_symlink(output)):
                _remove_real_tree(output)
                new_installed = False
            if old_moved:
                if output.exists() or _is_symlink(output):
                    raise RuntimeError("cannot restore previous output generation")
                replace_func(backup, output)
                old_moved = False
            raise
        if old_moved:
            _remove_real_tree(backup)
            old_moved = False
    finally:
        if old_moved and backup.exists() and not output.exists():
            replace_func(backup, output)


def _promote_generation_with_exposure(
    staging: Path,
    output: Path,
    exposure: Path,
    *,
    replace_func: Callable[[str | Path, str | Path], None],
    phase_hook: Callable[[str], None] | None,
    continuity_validator: Callable[[], None] | None,
    salt: bytes | None,
    expected_inventory_counts: Mapping[str, object] | None,
    expected_collection_classification_integrity_id: str | None,
    expected_classification_integrity_id: str | None,
) -> None:
    """Promote cache + exposure with a fsynced crash-recovery journal."""
    staged_exposure = _require_regular_file(
        staging / "mayo_exposure_manifest.json", "staged exposure manifest"
    )
    generation_commitment = _validate_staging(
        staging,
        salt=salt,
        expected_inventory_counts=expected_inventory_counts,
        expected_collection_classification_integrity_id=(
            expected_collection_classification_integrity_id
        ),
        expected_classification_integrity_id=expected_classification_integrity_id,
    )
    _fsync_generation_tree(staging)
    try:
        exposure.relative_to(output)
    except ValueError:
        pass
    else:
        raise ValueError("external exposure manifest must not live inside output generation")
    _require_private_directory(exposure.parent, "external exposure parent")
    _assert_no_symlink_components(exposure.parent)
    previous_output_ledger = None
    previous_output_storage_commitment = None
    if output.exists() or _is_symlink(output):
        (
            _previous_output,
            previous_output_ledger,
            previous_output_storage_commitment,
        ) = _private_generation_storage_commitment(
                output, "previous output generation",
        )
    previous_exposure_identity = None
    previous_exposure_stable_identity = None
    previous_exposure_sha256 = None
    previous_exposure_storage_commitment = None
    if exposure.exists() or _is_symlink(exposure):
        checked_exposure = _require_regular_file(
            exposure, "previous exposure manifest",
        )
        _require_private_regular_stat(
            os.lstat(checked_exposure), "previous exposure manifest",
        )
        previous_exposure_identity = _regular_snapshot(
            os.lstat(checked_exposure)
        )
        previous_exposure_stable_identity = _movement_stable_regular_snapshot(
            os.lstat(checked_exposure)
        )
        _payload, previous_exposure_sha256, _size = _read_regular_bytes(
            checked_exposure,
            "previous exposure manifest",
            max_bytes=_MAX_MAYO_MANIFEST_BYTES,
        )
        previous_exposure_storage_commitment = (
            _private_regular_storage_commitment(
                previous_exposure_stable_identity,
                previous_exposure_sha256,
            )
        )
    had_output = previous_output_storage_commitment is not None
    had_exposure = previous_exposure_storage_commitment is not None

    def assert_previous_output_unchanged() -> None:
        if (output.exists() or _is_symlink(output)) != had_output:
            raise ValueError("previous output topology changed before move")
        if not had_output:
            return
        (
            _current_output,
            current_output_ledger,
            current_output_storage_commitment,
        ) = _private_generation_storage_commitment(
            output, "previous output generation",
        )
        if (
            current_output_ledger != previous_output_ledger
            or current_output_storage_commitment
            != previous_output_storage_commitment
        ):
            raise ValueError("previous output generation changed before move")

    def assert_previous_exposure_unchanged() -> None:
        if (exposure.exists() or _is_symlink(exposure)) != had_exposure:
            raise ValueError("previous exposure topology changed before move")
        if not had_exposure:
            return
        live_exposure = os.lstat(exposure)
        _require_private_regular_stat(
            live_exposure, "previous exposure manifest",
        )
        if _regular_snapshot(live_exposure) != previous_exposure_identity:
            raise ValueError("previous exposure manifest changed before move")
        _payload, live_digest, _size = _read_regular_bytes(
            exposure,
            "previous exposure manifest",
            max_bytes=_MAX_MAYO_MANIFEST_BYTES,
        )
        live_commitment = _private_regular_storage_commitment(
            _movement_stable_regular_snapshot(os.lstat(exposure)),
            live_digest,
        )
        if not hmac.compare_digest(
            live_commitment, previous_exposure_storage_commitment,
        ):
            raise ValueError("previous exposure manifest changed before move")

    journal_path = _journal_path(output)
    if journal_path.exists() or _is_symlink(journal_path):
        raise RuntimeError("an interrupted Mayo cache transaction requires recovery")
    token = secrets.token_hex(8)
    output_backup = output.parent / f".{output.name}.backup-{token}"
    exposure_backup = exposure.parent / f".{exposure.name}.backup-{token}"
    exposure_temporary = exposure.parent / f".{exposure.name}.tmp-{token}"
    journal: dict[str, object] = {
        "schema": "mayo_cache_exposure_transaction_v3",
        "token": token,
        "staging_name": staging.name,
        "exposure_name": exposure.name,
        "had_output": had_output,
        "had_exposure": had_exposure,
        "generation_commitment": generation_commitment,
        "previous_output_storage_commitment": (
            previous_output_storage_commitment
        ),
        "previous_exposure_storage_commitment": (
            previous_exposure_storage_commitment
        ),
        "phase": "prepared",
        "indeterminate": False,
    }

    def set_phase(phase: str, *, invoke_hook: bool = True) -> None:
        journal["phase"] = phase
        _write_transaction_journal(journal_path, journal)
        if invoke_hook and phase_hook is not None:
            phase_hook(phase)

    def retain_indeterminate_move(phase: str) -> None:
        nonlocal rollback_is_durable
        rollback_is_durable = False
        journal["phase"] = phase
        journal["indeterminate"] = True
        _write_transaction_journal(journal_path, journal)

    rollback_is_durable = True
    allow_indeterminate_recovery = False
    committed_boundary_started = False
    cleanup_state = _JournalCleanupState()
    if continuity_validator is not None:
        continuity_validator()
    assert_previous_output_unchanged()
    assert_previous_exposure_unchanged()
    set_phase("prepared")
    try:
        temporary_descriptor = _open_exclusive_private_file(
            exposure_temporary, "external exposure temporary",
        )
        with staged_exposure.open("rb") as source, os.fdopen(
            temporary_descriptor, "wb",
        ) as target:
            shutil.copyfileobj(source, target)
            target.flush()
            os.fsync(target.fileno())
        _fsync_directory(exposure.parent)
        if had_output:
            assert_previous_output_unchanged()
            set_phase("moving_old_output", invoke_hook=False)
            _publish_private_path_no_replace(
                output,
                output_backup,
                "previous Mayo output backup",
                expected_identity=previous_output_ledger[0][2],
            )
            try:
                (
                    _moved_output,
                    moved_output_ledger,
                    moved_output_storage_commitment,
                ) = _private_generation_storage_commitment(
                        output_backup, "moved previous output generation",
                )
                if (
                    moved_output_ledger != previous_output_ledger
                    or moved_output_storage_commitment
                    != previous_output_storage_commitment
                ):
                    raise ValueError(
                        "previous output generation changed during move"
                    )
            except Exception as primary_error:
                try:
                    retain_indeterminate_move("old_output_moved")
                except Exception as evidence_error:
                    raise primary_error from evidence_error
                raise
            _fsync_directory(output.parent)
            set_phase("old_output_moved")
        if had_exposure:
            assert_previous_exposure_unchanged()
            set_phase("moving_old_exposure", invoke_hook=False)
            _publish_private_path_no_replace(
                exposure,
                exposure_backup,
                "previous Mayo exposure backup",
                expected_identity=previous_exposure_stable_identity,
            )
            try:
                moved_exposure = os.lstat(exposure_backup)
                _require_private_regular_stat(
                    moved_exposure, "moved previous exposure manifest",
                )
                if (
                    _movement_stable_regular_snapshot(moved_exposure)
                    != previous_exposure_stable_identity
                ):
                    raise ValueError(
                        "previous exposure manifest changed during move"
                    )
                _payload, moved_exposure_sha256, _size = _read_regular_bytes(
                    exposure_backup,
                    "moved previous exposure manifest",
                    max_bytes=_MAX_MAYO_MANIFEST_BYTES,
                )
                if not hmac.compare_digest(
                    moved_exposure_sha256, previous_exposure_sha256,
                ):
                    raise ValueError(
                        "previous exposure manifest changed during move"
                    )
            except Exception as primary_error:
                try:
                    retain_indeterminate_move("old_exposure_moved")
                except Exception as evidence_error:
                    raise primary_error from evidence_error
                raise
            _fsync_directory(exposure.parent)
            set_phase("old_exposure_moved")
        set_phase("installing_new_output", invoke_hook=False)
        _publish_private_path_no_replace(
            staging, output, "Mayo cache generation",
        )
        _fsync_directory(output.parent)
        set_phase("new_output_installed")
        set_phase("installing_new_exposure", invoke_hook=False)
        _publish_private_path_no_replace(
            exposure_temporary, exposure, "Mayo exposure manifest",
        )
        _fsync_directory(exposure.parent)
        set_phase("new_exposure_installed")
        committed_holds = ExitStack()
        committed_journal_descriptor = None
        try:
            held_generation = committed_holds.enter_context(
                _hold_committed_mayo_generation(
                    output,
                    exposure,
                    media_count=int(generation_commitment["mediapipe_file_count"]),
                    arkit_count=int(generation_commitment["arkit_file_count"]),
                )
            )
            _assert_committed_generation(
                output, exposure, generation_commitment,
                salt=salt,
                expected_inventory_counts=expected_inventory_counts,
                expected_collection_classification_integrity_id=(
                    expected_collection_classification_integrity_id
                ),
                expected_classification_integrity_id=(
                    expected_classification_integrity_id
                ),
                _held=held_generation,
            )
            held_output_backup = None
            if had_output:
                held_output_backup = committed_holds.enter_context(
                    _hold_private_storage_tree(
                        output_backup, "committed previous output backup",
                    )
                )
                backup_ledger, backup_commitment = (
                    _held_private_generation_storage_commitment(
                        held_output_backup,
                        "committed previous output backup",
                    )
                )
                if (
                    backup_ledger != previous_output_ledger
                    or not hmac.compare_digest(
                        backup_commitment,
                        previous_output_storage_commitment,
                    )
                ):
                    raise ValueError(
                        "committed previous output backup changed"
                    )
            held_exposure_backup = None
            if had_exposure:
                held_exposure_backup = committed_holds.enter_context(
                    _hold_private_regular_storage(
                        exposure_backup,
                        "committed previous exposure backup",
                    )
                )
                if not hmac.compare_digest(
                    _held_private_regular_storage_commitment(
                        held_exposure_backup,
                        "committed previous exposure backup",
                    ),
                    previous_exposure_storage_commitment,
                ):
                    raise ValueError(
                        "committed previous exposure backup changed"
                    )
            if exposure_temporary.exists() or _is_symlink(exposure_temporary):
                raise ValueError(
                    "committed exposure temporary unexpectedly exists"
                )
            if continuity_validator is not None:
                continuity_validator()
            _assert_held_mayo_generation(held_generation)
            set_phase("committed", invoke_hook=False)
            committed_boundary_started = True
            (
                committed_journal_descriptor,
                committed_journal_identity,
                committed_journal,
            ) = _open_transaction_journal(journal_path)
            if committed_journal != journal:
                raise ValueError("committed journal bytes changed after write")
            _assert_held_mayo_generation(held_generation)
            if phase_hook is not None:
                phase_hook("committed")
            _assert_held_mayo_generation(held_generation)
            if held_output_backup is not None:
                _assert_held_private_storage_tree(
                    held_output_backup, "committed previous output backup",
                )
            if held_exposure_backup is not None:
                _assert_held_private_regular_storage(
                    held_exposure_backup, "committed previous exposure backup",
                )
            if exposure_temporary.exists() or _is_symlink(exposure_temporary):
                raise ValueError(
                    "committed exposure temporary unexpectedly reappeared"
                )
            if continuity_validator is not None:
                try:
                    continuity_validator()
                except Exception as primary_error:
                    # If the committed hook exposed key drift, durably downgrade
                    # before the common recovery path rolls back both outputs.
                    os.close(committed_journal_descriptor)
                    committed_journal_descriptor = None
                    journal["phase"] = "new_exposure_installed"
                    journal["indeterminate"] = True
                    try:
                        _write_transaction_journal(journal_path, journal)
                    except Exception as downgrade_error:
                        rollback_is_durable = False
                        raise primary_error from downgrade_error
                    allow_indeterminate_recovery = True
                    committed_boundary_started = False
                    raise

            _remove_real_tree(output_backup)
            _unlink_regular_if_present(
                exposure_backup, "committed exposure backup",
            )
            _unlink_regular_if_present(
                exposure_temporary, "committed exposure temporary",
            )

            def validate_committed_final_state() -> None:
                _assert_held_mayo_generation(held_generation)
                if (
                    output_backup.exists() or _is_symlink(output_backup)
                    or exposure_backup.exists() or _is_symlink(exposure_backup)
                    or exposure_temporary.exists()
                    or _is_symlink(exposure_temporary)
                ):
                    raise ValueError(
                        "committed Mayo cleanup residue reappeared"
                    )
                if continuity_validator is not None:
                    continuity_validator()

            fsync_directories = tuple(dict.fromkeys((
                output.parent, exposure.parent,
            )))
            _unlink_held_transaction_journal_durably(
                descriptor=committed_journal_descriptor,
                identity=committed_journal_identity,
                path=journal_path,
                journal=journal,
                validate_final_state=validate_committed_final_state,
                fsync_directories=fsync_directories,
                cleanup_state=cleanup_state,
            )
        finally:
            try:
                committed_holds.__exit__(*sys.exc_info())
            finally:
                if committed_journal_descriptor is not None:
                    os.close(committed_journal_descriptor)
    except Exception as primary_error:
        if (
            committed_boundary_started
            and not cleanup_state.compensation_attempted
            and not journal_path.exists()
            and not _is_symlink(journal_path)
        ):
            try:
                _write_transaction_journal(journal_path, journal)
            except Exception as restoration_error:
                raise primary_error from restoration_error
        if rollback_is_durable and not committed_boundary_started:
            _recover_cache_exposure_transaction(
                output, exposure,
                salt=salt,
                expected_inventory_counts=expected_inventory_counts,
                expected_collection_classification_integrity_id=(
                    expected_collection_classification_integrity_id
                ),
                expected_classification_integrity_id=(
                    expected_classification_integrity_id
                ),
                allow_indeterminate=allow_indeterminate_recovery,
            )
        raise


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    validate_public_manifest(dict(payload))
    _require_private_directory(path.parent, "private manifest parent")
    serialized = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    descriptor = _open_exclusive_private_file(path, "private manifest")
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())


def _open_nofollow_directory(path: Path, field: str) -> int:
    checked = _require_private_directory(path, field)
    descriptor = os.open(
        checked,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened = os.fstat(descriptor)
        current = os.lstat(checked)
        _require_private_directory_stat(opened, field)
        _require_private_directory_stat(current, field)
        identity = (opened.st_dev, opened.st_ino, opened.st_mode)
        if not stat.S_ISDIR(opened.st_mode) or identity != (
            current.st_dev, current.st_ino, current.st_mode
        ):
            raise ValueError(f"{field} changed identity while it was opened")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _directory_snapshot(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_mode),
        int(value.st_uid), int(value.st_gid), int(value.st_nlink),
    )


def _regular_snapshot(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_mode),
        int(value.st_uid), int(value.st_gid), int(value.st_nlink),
        int(value.st_size), int(value.st_mtime_ns), int(value.st_ctime_ns),
    )


def _open_nofollow_directory_at(
    parent_descriptor: int,
    name: str,
    field: str,
) -> tuple[int, tuple[int, ...]]:
    if type(name) is not str or name in {"", ".", ".."} or Path(name).name != name:
        raise ValueError(f"{field} anchored directory name is unsafe")
    before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        dir_fd=parent_descriptor,
    )
    try:
        opened = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        _require_private_directory_stat(before, field)
        _require_private_directory_stat(opened, field)
        _require_private_directory_stat(current, field)
        identity = _directory_snapshot(opened)
        if (
            not stat.S_ISDIR(before.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or _directory_snapshot(before) != identity
            or _directory_snapshot(current) != identity
        ):
            raise ValueError(f"{field} changed identity while it was opened")
        return descriptor, identity
    except BaseException:
        os.close(descriptor)
        raise


def _open_regular_at(
    parent_descriptor: int,
    name: str,
    field: str,
) -> tuple[int, tuple[int, ...]]:
    if type(name) is not str or name in {"", ".", ".."} or Path(name).name != name:
        raise ValueError(f"{field} anchored filename is unsafe")
    try:
        before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        descriptor = os.open(
            name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
    except FileNotFoundError as exc:
        raise ValueError(f"{field} is missing") from exc
    try:
        opened = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        _require_private_regular_stat(before, field)
        _require_private_regular_stat(opened, field)
        _require_private_regular_stat(current, field)
        identity = _regular_snapshot(opened)
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or int(opened.st_nlink) != 1
            or _regular_snapshot(before) != identity
            or _regular_snapshot(current) != identity
        ):
            raise ValueError(f"{field} changed identity while it was opened")
        return descriptor, identity
    except BaseException:
        os.close(descriptor)
        raise


def _read_regular_descriptor(
    descriptor: int,
    *,
    parent_descriptor: int,
    name: str,
    field: str,
    expected_identity: tuple[int, ...],
    max_bytes: int | None = None,
) -> tuple[bytes, str, int]:
    if max_bytes is not None and (
        not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1
    ):
        raise ValueError(f"{field} byte limit must be a positive integer")
    before = os.fstat(descriptor)
    _require_private_regular_stat(before, field)
    if (
        not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
        or _regular_snapshot(before) != expected_identity
    ):
        raise ValueError(f"{field} held descriptor changed")
    if max_bytes is not None and int(before.st_size) > max_bytes:
        raise ValueError(f"{field} exceeds its raw byte limit")
    os.lseek(descriptor, 0, os.SEEK_SET)
    payload = bytearray()
    digest = hashlib.sha256()
    while block := os.read(descriptor, 1024 * 1024):
        payload.extend(block)
        if max_bytes is not None and len(payload) > max_bytes:
            raise ValueError(f"{field} exceeds its raw byte limit")
        digest.update(block)
    after = os.fstat(descriptor)
    current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    _require_private_regular_stat(after, field)
    _require_private_regular_stat(current, field)
    if (
        _regular_snapshot(after) != expected_identity
        or _regular_snapshot(current) != expected_identity
    ):
        raise ValueError(f"{field} changed while its held descriptor was read")
    return bytes(payload), digest.hexdigest(), int(after.st_size)


def _read_regular_bytes(
    path: Path,
    field: str,
    *,
    max_bytes: int | None = None,
    parent_descriptor: int | None = None,
) -> tuple[bytes, str, int]:
    if max_bytes is not None and (
        not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1
    ):
        raise ValueError(f"{field} byte limit must be a positive integer")
    open_kwargs: dict[str, int] = {}
    if parent_descriptor is None:
        target: str | Path = _require_regular_file(path, field)
    else:
        if path.name != str(path) or len(path.parts) != 1:
            raise ValueError(f"{field} anchored filename is unsafe")
        target = path.name
        open_kwargs["dir_fd"] = parent_descriptor
    descriptor = os.open(
        target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0), **open_kwargs
    )
    payload = bytearray()
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        _require_private_regular_stat(before, field)
        if max_bytes is not None and int(before.st_size) > max_bytes:
            raise ValueError(f"{field} exceeds its raw byte limit")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            payload.extend(block)
            if max_bytes is not None and len(payload) > max_bytes:
                raise ValueError(f"{field} exceeds its raw byte limit")
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = (
        os.lstat(target)
        if parent_descriptor is None
        else os.stat(target, dir_fd=parent_descriptor, follow_symlinks=False)
    )
    _require_private_regular_stat(after, field)
    _require_private_regular_stat(current, field)
    before_identity = _regular_snapshot(before)
    if (
        before_identity != _regular_snapshot(after)
        or before_identity != _regular_snapshot(current)
    ):
        raise ValueError(f"{field} changed while it was read")
    return bytes(payload), digest.hexdigest(), before.st_size


def _load_public_json(path: Path, field: str) -> tuple[dict[str, object], str]:
    payload, digest, _size = _read_regular_bytes(
        path, field, max_bytes=_MAX_MAYO_MANIFEST_BYTES
    )
    value = _decode_unique_json_object(payload, field)
    validate_public_manifest(value)
    return value, digest


def _load_public_json_descriptor(
    descriptor: int,
    *,
    parent_descriptor: int,
    name: str,
    field: str,
    expected_identity: tuple[int, ...],
) -> tuple[dict[str, object], str]:
    payload, digest, _size = _read_regular_descriptor(
        descriptor,
        parent_descriptor=parent_descriptor,
        name=name,
        field=field,
        expected_identity=expected_identity,
        max_bytes=_MAX_MAYO_MANIFEST_BYTES,
    )
    value = _decode_unique_json_object(payload, field)
    validate_public_manifest(value)
    return value, digest


_COLLECTION_TOP_FIELDS = frozenset({
    "schema_version", "dataset", "identity_status", "split_unit",
    "feature_schema", "feature_names", "capture_mirrored",
    "normalization_transform", "temporal_protocol", "modality_boundary",
    "counts", "mediapipe_records", "arkit_records",
    "classification_integrity_id", "metadata_only_exclusions", "provenance",
})
_EXPOSURE_TOP_FIELDS = frozenset({
    "schema_version", "dataset", "policy", "identity_status", "videos",
    "arkit_trajectories", "classification_integrity_id", "counts",
})
_MEDIA_COLLECTION_FIELDS = frozenset({
    "recording_id", "group_id", "source_integrity_id", "source_fingerprint",
    "cache_integrity_id", "cache_source", "producer_protocol",
    "producer_adapter_version", "legacy_export_audit_status", "identity_status",
    "split_unit", "development_only", "ssl_exposed",
    "independent_evaluation_eligible",
})
_ARKIT_PUBLIC_FIELDS = frozenset({
    "recording_id", "group_id", "source_integrity_id", "source_fingerprint",
    "cache_integrity_id", "feature_schema", "identity_status", "split_unit",
    "development_only", "ssl_exposed", "independent_evaluation_eligible",
})
_EXPOSURE_MEDIA_FIELDS = frozenset({
    "instance_id", "recording_id", "group_id", "source_integrity_id",
    "source_fingerprint", "cache_integrity_id", "status", "identity_status",
    "split_unit", "development_only", "ssl_exposed",
    "independent_evaluation_eligible",
})
_EXPOSURE_EXCLUDED_FIELDS = _EXPOSURE_MEDIA_FIELDS - {"cache_integrity_id"}
_PROVENANCE_FIELDS = frozenset({
    "runtime_dependencies", "dependency_aggregate_sha256", "model_sha256",
    "source_collection_integrity_id", "producer_sha256",
    "producer_aggregate_sha256",
})
_DEPENDENCY_ROW_FIELDS = frozenset({
    "distribution", "version", "installed_file_count",
    "installed_file_aggregate_sha256",
})
_PRODUCER_NAMES = frozenset({
    "builder", "action_bundle", "clinical_landmarks",
    "dynamic_landmark_schema", "feature_registry",
})


def _require_exact_object(
    value: object, fields: frozenset[str], label: str,
) -> Mapping[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} field schema is not exact")
    return value


def _require_public_governance(row: Mapping[str, object], label: str) -> None:
    if (
        row.get("identity_status") != UNKNOWN_IDENTITY
        or row.get("split_unit") != RECORDING_HELD_OUT
        or row.get("development_only") is not True
        or row.get("ssl_exposed") is not True
        or row.get("independent_evaluation_eligible") is not False
    ):
        raise ValueError(f"{label} governance is not development-only")


def _require_public_identity(row: Mapping[str, object], label: str) -> None:
    recording = row.get("recording_id")
    group = row.get("group_id")
    source_integrity = row.get("source_integrity_id")
    fingerprint = row.get("source_fingerprint")
    cache_integrity = row.get("cache_integrity_id")
    if (
        not isinstance(recording, str) or _RECORDING_ID.fullmatch(recording) is None
        or not isinstance(group, str) or _GROUP_ID.fullmatch(group) is None
        or not isinstance(source_integrity, str)
        or _INTEGRITY_ID.fullmatch(source_integrity) is None
        or not source_integrity.startswith("src_")
        or not isinstance(fingerprint, str) or _FINGERPRINT.fullmatch(fingerprint) is None
        or not isinstance(cache_integrity, str)
        or _INTEGRITY_ID.fullmatch(cache_integrity) is None
        or not cache_integrity.startswith("cache_")
    ):
        raise ValueError(f"{label} opaque identity is missing or noncanonical")


def _validate_public_provenance(value: object) -> None:
    provenance = _require_exact_object(value, _PROVENANCE_FIELDS, "public provenance")
    for key in (
        "dependency_aggregate_sha256", "model_sha256",
        "producer_aggregate_sha256",
    ):
        if not isinstance(provenance[key], str) or _SHA256.fullmatch(provenance[key]) is None:
            raise ValueError("public provenance contains a noncanonical hash")
    aggregate = provenance["source_collection_integrity_id"]
    if (
        not isinstance(aggregate, str) or _INTEGRITY_ID.fullmatch(aggregate) is None
        or not aggregate.startswith("agg_")
    ):
        raise ValueError("public source collection integrity ID is noncanonical")
    producers = _require_exact_object(
        provenance["producer_sha256"], _PRODUCER_NAMES, "producer provenance"
    )
    if any(not isinstance(value, str) or _SHA256.fullmatch(value) is None
           for value in producers.values()):
        raise ValueError("producer provenance hash is noncanonical")
    dependencies = provenance["runtime_dependencies"]
    if not isinstance(dependencies, list) or len(dependencies) != 4:
        raise ValueError("runtime dependency provenance is incomplete")
    names: list[str] = []
    for value in dependencies:
        row = _require_exact_object(value, _DEPENDENCY_ROW_FIELDS,
                                    "runtime dependency provenance")
        distribution = row["distribution"]
        if not isinstance(distribution, str) or not distribution:
            raise ValueError("runtime dependency distribution is invalid")
        names.append(distribution)
        if (
            not isinstance(row["version"], str) or not row["version"]
            or not isinstance(row["installed_file_count"], int)
            or isinstance(row["installed_file_count"], bool)
            or row["installed_file_count"] <= 0
            or not isinstance(row["installed_file_aggregate_sha256"], str)
            or _SHA256.fullmatch(row["installed_file_aggregate_sha256"]) is None
        ):
            raise ValueError("runtime dependency provenance is noncanonical")
    if (
        names != sorted(names) or len(names) != len(set(names))
        or set(names) - ({"python", "numpy", "mediapipe"} | set(OPENCV_DISTRIBUTIONS))
        or not {"python", "numpy", "mediapipe"}.issubset(names)
        or len(set(names) & set(OPENCV_DISTRIBUTIONS)) != 1
    ):
        raise ValueError("runtime dependency set is noncanonical")


def _validate_collection_top(manifest: Mapping[str, object]) -> dict[str, int]:
    _require_exact_object(manifest, _COLLECTION_TOP_FIELDS, "collection manifest")
    expected_temporal = {
        "source_timeline": "per_recording_audited_fps_and_monotonic_source_index",
        "ssl_view_hz": 30,
        "resampling": "exact_target_source_index_selection_no_interpolation_or_nearest_fill",
    }
    expected_modality = (
        "ARKit 52-blendshape trajectories are auxiliary-only and are never "
        "concatenated with or promoted to MediaPipe landmarks"
    )
    if (
        manifest["schema_version"] != COLLECTION_SCHEMA
        or manifest["dataset"] != COLLECTION_DATASET
        or manifest["identity_status"] != UNKNOWN_IDENTITY
        or manifest["split_unit"] != RECORDING_HELD_OUT
        or manifest["feature_schema"] != DYNAMIC_FEATURE_SCHEMA
        or manifest["feature_names"] != list(DYNAMIC_FEATURE_NAMES)
        or manifest["capture_mirrored"] != "unknown"
        or manifest["normalization_transform"] != TRANSFORM_NORMALIZATION
        or manifest["temporal_protocol"] != expected_temporal
        or manifest["modality_boundary"] != expected_modality
    ):
        raise ValueError("collection manifest top-level policy is noncanonical")
    counts = _require_exact_object(
        manifest["counts"], frozenset(FROZEN_INVENTORY), "collection counts"
    )
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
           for value in counts.values()):
        raise ValueError("collection count is invalid")
    exclusion = _require_exact_object(
        manifest["metadata_only_exclusions"],
        frozenset({"index_or_depth_metadata_only_no_video_or_arkit_trajectory"}),
        "metadata-only exclusion",
    )
    if exclusion["index_or_depth_metadata_only_no_video_or_arkit_trajectory"] != counts[
        "metadata_only_sessions"
    ]:
        raise ValueError("metadata-only exclusion count disagrees")
    _validate_public_provenance(manifest["provenance"])
    return dict(counts)


def _manifest_cache_rows(
    manifest: Mapping[str, object], key: str, modality: str,
) -> dict[str, Mapping[str, object]]:
    value = manifest.get(key)
    if not isinstance(value, list):
        raise ValueError(f"collection manifest {key} must be a list")
    rows: dict[str, Mapping[str, object]] = {}
    for item in value:
        fields = _MEDIA_COLLECTION_FIELDS if modality == "mediapipe" else _ARKIT_PUBLIC_FIELDS
        row = _require_exact_object(item, fields, f"{modality} collection record")
        _require_public_identity(row, f"{modality} collection record")
        _require_public_governance(row, f"{modality} collection record")
        recording_id = row["recording_id"]
        if modality == "mediapipe":
            if (
                row["cache_source"] != "raw_video_reextracted_homogeneous_video_mode"
                or row["producer_protocol"] != VIDEO_PRODUCER_PROTOCOL
                or row["producer_adapter_version"] != VIDEO_ADAPTER_VERSION
                or row["legacy_export_audit_status"] not in {
                    "not_reused_unverifiable_source_binding", "no_complete_legacy_export"
                }
            ):
                raise ValueError("MediaPipe collection producer contract is noncanonical")
        elif row["feature_schema"] != "arkit_blendshapes_52_v1":
            raise ValueError("ARKit collection feature schema is noncanonical")
        if recording_id in rows:
            raise ValueError("collection manifest repeats a cache recording")
        rows[str(recording_id)] = row
    return rows


def _require_cached_exact(
    cached: object, name: str, expected: object,
) -> None:
    observed = np.asarray(cached[name])
    canonical = np.asarray(expected)
    if (
        observed.dtype != canonical.dtype
        or observed.shape != canonical.shape
        or not np.array_equal(observed, canonical)
    ):
        raise ValueError(f"compact cache {name} is not canonical")


def _require_cached_view(
    cached: object,
    mapping: Mapping[str, np.ndarray],
) -> None:
    for name, expected in mapping.items():
        observed = np.asarray(cached[name])
        if (
            observed.dtype != expected.dtype
            or observed.shape != expected.shape
            or not np.array_equal(observed, expected)
        ):
            raise ValueError(f"compact cache {name} is not the exact 30-Hz view")


def _validate_mediapipe_cache_payload(
    cached: object,
    *,
    recording_id: str,
    group_id: str,
    source_integrity_id: str,
    source_fingerprint: str,
) -> CompactCacheSummary:
    sequence = _validate_media_sequence(MayoMediaSequence(
        features=np.asarray(cached["features_source_rate"]),
        valid_mask=np.asarray(cached["valid_mask_source_rate"]),
        timestamps=np.asarray(cached["timestamps_source_rate"]),
        source_frame_indices=np.asarray(cached["source_frame_indices_source_rate"]),
        facial_transforms=np.asarray(cached["facial_transforms_source_rate"]),
        facial_transform_mask=np.asarray(cached["facial_transform_mask_source_rate"]),
        transform_source=str(np.asarray(cached["facial_transform_source"]).item()),
        source_fps=float(np.asarray(cached["source_fps"]).item()),
        timestamp_source=str(np.asarray(cached["timestamp_source"]).item()),
    ))
    view = downsample_to_30hz(sequence)
    _require_cached_view(cached, {
        "features_30hz": view.features,
        "valid_mask_30hz": view.valid_mask,
        "timestamps_30hz": view.timestamps,
        "source_frame_indices_30hz": view.source_frame_indices,
        "target_frame_indices_30hz": view.target_frame_indices,
        "contiguous_from_previous_30hz": view.contiguous_from_previous,
        "facial_transforms_30hz": view.facial_transforms,
        "facial_transform_mask_30hz": view.facial_transform_mask,
    })
    for name, expected in (
        ("feature_schema", DYNAMIC_FEATURE_SCHEMA),
        ("feature_names", np.asarray(DYNAMIC_FEATURE_NAMES)),
        ("side_convention", CLINICAL_SIDE_CONVENTION),
        ("capture_mirrored", "unknown"),
        ("normalization_transform", TRANSFORM_NORMALIZATION),
        ("facial_transform_source", "same_detection_mediapipe_video_mode"),
        ("timestamp_unit", "seconds"),
        ("timestamp_source", "source_frame_index_divided_by_audited_fps"),
        ("source_fps", np.asarray(sequence.source_fps, dtype=np.float64)),
        ("producer_protocol", VIDEO_PRODUCER_PROTOCOL),
        ("producer_adapter_version", VIDEO_ADAPTER_VERSION),
        ("recording_id", recording_id),
        ("group_id", group_id),
        ("source_integrity_id", source_integrity_id),
        ("source_fingerprint", source_fingerprint),
        ("cache_schema", MEDIAPIPE_CACHE_SCHEMA),
        ("development_only", np.asarray(True)),
        ("patient_identity", "unknown"),
        ("split_unit", "recording"),
    ):
        _require_cached_exact(cached, name, expected)
    return CompactCacheSummary(len(sequence.features), 0)


def _validate_arkit_cache_payload(
    cached: object,
    *,
    recording_id: str,
    group_id: str,
    source_integrity_id: str,
    source_fingerprint: str,
) -> CompactCacheSummary:
    sequence = _validate_arkit_sequence(ARKitSequence(
        features=np.asarray(cached["features_60hz"]),
        valid_mask=np.asarray(cached["valid_mask_60hz"]),
        timestamps=np.asarray(cached["timestamps_60hz"]),
        source_frame_indices=np.asarray(cached["source_frame_indices_60hz"]),
        timestamp_source=str(np.asarray(cached["timestamp_source"]).item()),
    ))
    view = downsample_arkit_to_30hz(sequence)
    _require_cached_view(cached, {
        "features_30hz": view.features,
        "valid_mask_30hz": view.valid_mask,
        "timestamps_30hz": view.timestamps,
        "source_frame_indices_30hz": view.source_frame_indices,
        "target_frame_indices_30hz": view.target_frame_indices,
        "contiguous_from_previous_30hz": view.contiguous_from_previous,
    })
    for name, expected in (
        ("feature_schema", "arkit_blendshapes_52_v1"),
        ("feature_names", np.asarray(ARKIT_BLENDSHAPE_NAMES)),
        ("timestamp_unit", "seconds"),
        ("timestamp_source", "arkit_original_timecode_relative_seconds"),
        ("recording_id", recording_id),
        ("group_id", group_id),
        ("source_integrity_id", source_integrity_id),
        ("source_fingerprint", source_fingerprint),
        ("cache_schema", ARKIT_CACHE_SCHEMA),
        ("development_only", np.asarray(True)),
        ("patient_identity", "unknown"),
        ("split_unit", "recording"),
    ):
        _require_cached_exact(cached, name, expected)
    missing = sum(
        int(current) - int(previous) - 1
        for previous, current in zip(
            sequence.source_frame_indices[:-1], sequence.source_frame_indices[1:]
        )
    )
    return CompactCacheSummary(len(sequence.features), missing)


def _bounded_npy_header(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    field: str,
) -> tuple[np.dtype, tuple[int, ...], bool]:
    try:
        with archive.open(info, "r") as member:
            version = np.lib.format.read_magic(member)
            if version not in {(1, 0), (2, 0), (3, 0)}:
                raise ValueError(f"{field} has an unsupported NPY version")
            shape, fortran_order, dtype = np.lib.format._read_array_header(
                member, version, max_header_size=_MAX_NPY_HEADER_BYTES
            )
            header_bytes = member.tell()
    except (OSError, EOFError, UnicodeError, ValueError, RuntimeError,
            zipfile.BadZipFile) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(field):
            raise
        raise ValueError(f"{field} has an invalid bounded NPY header") from exc
    if (
        not isinstance(shape, tuple)
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 0
               for item in shape)
        or not isinstance(fortran_order, bool)
    ):
        raise ValueError(f"{field} has a noncanonical NPY header")
    canonical_dtype = np.dtype(dtype)
    if canonical_dtype.hasobject or canonical_dtype.fields is not None:
        raise ValueError(f"{field} uses an unsafe NPY dtype")
    element_count = 1
    for dimension in shape:
        element_count *= dimension
    if header_bytes + element_count * canonical_dtype.itemsize != int(info.file_size):
        raise ValueError(f"{field} NPY header does not match its member size")
    return canonical_dtype, shape, fortran_order


def _require_exact_zip_eocd(payload: bytes, *, member_count: int, field: str) -> None:
    """Validate a bounded, single-disk central directory before ``zipfile``."""
    if (
        not isinstance(member_count, int)
        or isinstance(member_count, bool)
        or not 0 < member_count < 0xFFFF
    ):
        raise ValueError(f"{field} has an invalid fixed member count")

    offset = len(payload) - 22
    if offset < 0 or payload[offset:offset + 4] != b"PK\x05\x06":
        raise ValueError(f"{field} has no canonical ZIP end record")
    record = memoryview(payload)[offset:offset + 22]
    disk = int.from_bytes(record[4:6], "little")
    central_disk = int.from_bytes(record[6:8], "little")
    disk_members = int.from_bytes(record[8:10], "little")
    total_members = int.from_bytes(record[10:12], "little")
    central_size = int.from_bytes(record[12:16], "little")
    central_offset = int.from_bytes(record[16:20], "little")
    comment_size = int.from_bytes(record[20:22], "little")
    if (
        disk_members == 0xFFFF
        or total_members == 0xFFFF
        or central_size == 0xFFFF_FFFF
        or central_offset == 0xFFFF_FFFF
        or disk != 0
        or central_disk != 0
        or disk_members != member_count
        or total_members != member_count
        or comment_size != 0
        or central_size < 47 * member_count
        or central_size > _MAX_MAYO_NPZ_CENTRAL_RECORD_BYTES * member_count
        or central_offset <= 0
        or central_offset + central_size != offset
    ):
        raise ValueError(f"{field} ZIP end record is noncanonical")
    if offset >= 20 and payload[offset - 20:offset - 16] == b"PK\x06\x07":
        raise ValueError(f"{field} ZIP64 end record is unsupported")

    central = memoryview(payload)[central_offset:offset]
    cursor = 0
    actual_members = 0
    compressed_total = 0
    expanded_total = 0
    member_names: set[bytes] = set()
    local_offsets: set[int] = set()
    while cursor < central_size:
        remaining = central_size - cursor
        if remaining < 46 or central[cursor:cursor + 4].tobytes() != b"PK\x01\x02":
            raise ValueError(f"{field} central directory is noncanonical")

        actual_members += 1
        if actual_members > member_count:
            raise ValueError(f"{field} central directory member count is not exact")
        version_needed = int.from_bytes(central[cursor + 6:cursor + 8], "little")
        flags = int.from_bytes(central[cursor + 8:cursor + 10], "little")
        compression = int.from_bytes(central[cursor + 10:cursor + 12], "little")
        compressed_size = int.from_bytes(central[cursor + 20:cursor + 24], "little")
        expanded_size = int.from_bytes(central[cursor + 24:cursor + 28], "little")
        name_size = int.from_bytes(central[cursor + 28:cursor + 30], "little")
        extra_size = int.from_bytes(central[cursor + 30:cursor + 32], "little")
        member_comment_size = int.from_bytes(
            central[cursor + 32:cursor + 34], "little"
        )
        member_disk = int.from_bytes(central[cursor + 34:cursor + 36], "little")
        local_offset = int.from_bytes(central[cursor + 42:cursor + 46], "little")
        record_size = 46 + name_size + extra_size + member_comment_size
        if (
            version_needed != 20
            or flags != 0
            or compression != zipfile.ZIP_DEFLATED
            or compressed_size == 0xFFFF_FFFF
            or expanded_size == 0xFFFF_FFFF
            or name_size <= 0
            or extra_size != 0
            or member_comment_size != 0
            or member_disk != 0
            or local_offset == 0xFFFF_FFFF
            or local_offset >= central_offset
            or record_size > _MAX_MAYO_NPZ_CENTRAL_RECORD_BYTES
            or record_size > remaining
        ):
            raise ValueError(f"{field} central directory metadata is noncanonical")

        name_start = cursor + 46
        member_name = central[name_start:name_start + name_size].tobytes()
        if (
            member_name in member_names
            or local_offset in local_offsets
            or payload[local_offset:local_offset + 4] != b"PK\x03\x04"
        ):
            raise ValueError(f"{field} central directory references are noncanonical")
        member_names.add(member_name)
        local_offsets.add(local_offset)

        compressed_total += compressed_size
        expanded_total += expanded_size
        if (
            compressed_total > _MAX_MAYO_NPZ_COMPRESSED_BYTES
            or expanded_total > _MAX_MAYO_NPZ_EXPANDED_BYTES
        ):
            raise ValueError(f"{field} central directory size declarations are excessive")
        cursor += record_size

    if cursor != central_size or actual_members != member_count:
        raise ValueError(f"{field} central directory member count is not exact")


def _require_mayo_npz_headers(
    payload: bytes,
    *,
    recording_id: str,
    group_id: str,
    source_integrity_id: str,
    source_fingerprint: str,
    expected_schema: str,
) -> None:
    """Inspect bounded ZIP metadata and NPY headers before ``np.load``."""
    if len(payload) > _MAX_MAYO_CACHE_RAW_BYTES:
        raise ValueError("compact cache exceeds its raw byte limit")
    expected_fields = (
        _MEDIAPIPE_CACHE_FIELDS
        if expected_schema == MEDIAPIPE_CACHE_SCHEMA
        else _ARKIT_CACHE_FIELDS
        if expected_schema == ARKIT_CACHE_SCHEMA
        else None
    )
    if expected_fields is None:
        raise ValueError("compact cache schema is unsupported")
    expected_names = {f"{name}.npy" for name in expected_fields}
    _require_exact_zip_eocd(
        payload, member_count=len(expected_names), field="compact cache"
    )
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            infos = tuple(archive.infolist())
            names = tuple(info.filename for info in infos)
            if (
                len(infos) != len(expected_names)
                or len(names) != len(set(names))
                or set(names) != expected_names
                or any(info.is_dir() for info in infos)
            ):
                raise ValueError("compact cache ZIP member schema is not exact")
            if any(
                info.flag_bits & 0x1
                or info.compress_type != zipfile.ZIP_DEFLATED
                or info.compress_size < 0
                or info.file_size < 0
                for info in infos
            ):
                raise ValueError("compact cache ZIP metadata is noncanonical")
            if sum(int(info.compress_size) for info in infos) > (
                _MAX_MAYO_NPZ_COMPRESSED_BYTES
            ):
                raise ValueError("compact cache exceeds its compressed byte limit")
            if sum(int(info.file_size) for info in infos) > (
                _MAX_MAYO_NPZ_EXPANDED_BYTES
            ):
                raise ValueError("compact cache exceeds its expanded byte limit")
            headers = {
                info.filename[:-4]: _bounded_npy_header(
                    archive, info, field=f"compact cache {info.filename[:-4]}"
                )
                for info in infos
            }
    except (OSError, EOFError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("compact cache"):
            raise
        raise ValueError("compact cache is not a bounded exact NPZ") from exc

    if expected_schema == MEDIAPIPE_CACHE_SCHEMA:
        source_dtype, source_shape, _ = headers["features_source_rate"]
        target_dtype, target_shape, _ = headers["features_30hz"]
        if (
            source_dtype != np.dtype(np.float32)
            or len(source_shape) != 2
            or source_shape[1] != len(DYNAMIC_FEATURE_NAMES)
            or not 1 <= source_shape[0] <= _MAX_MAYO_CACHE_ROWS
            or target_dtype != np.dtype(np.float32)
            or len(target_shape) != 2
            or target_shape[1] != len(DYNAMIC_FEATURE_NAMES)
            or not 1 <= target_shape[0] <= source_shape[0]
        ):
            raise ValueError("compact cache feature NPY headers are noncanonical")
        source_rows = source_shape[0]
        target_rows = target_shape[0]
        expected_arrays = {
            "features_source_rate": (np.dtype(np.float32), (source_rows, 95)),
            "valid_mask_source_rate": (np.dtype(np.bool_), (source_rows,)),
            "timestamps_source_rate": (np.dtype(np.float64), (source_rows,)),
            "source_frame_indices_source_rate": (np.dtype(np.int64), (source_rows,)),
            "facial_transforms_source_rate": (
                np.dtype(np.float32), (source_rows, 4, 4),
            ),
            "facial_transform_mask_source_rate": (
                np.dtype(np.bool_), (source_rows,),
            ),
            "features_30hz": (np.dtype(np.float32), (target_rows, 95)),
            "valid_mask_30hz": (np.dtype(np.bool_), (target_rows,)),
            "timestamps_30hz": (np.dtype(np.float64), (target_rows,)),
            "source_frame_indices_30hz": (np.dtype(np.int64), (target_rows,)),
            "target_frame_indices_30hz": (np.dtype(np.int64), (target_rows,)),
            "contiguous_from_previous_30hz": (
                np.dtype(np.bool_), (target_rows,),
            ),
            "facial_transforms_30hz": (
                np.dtype(np.float32), (target_rows, 4, 4),
            ),
            "facial_transform_mask_30hz": (
                np.dtype(np.bool_), (target_rows,),
            ),
        }
        metadata = {
            "feature_schema": DYNAMIC_FEATURE_SCHEMA,
            "feature_names": np.asarray(DYNAMIC_FEATURE_NAMES),
            "side_convention": CLINICAL_SIDE_CONVENTION,
            "capture_mirrored": "unknown",
            "normalization_transform": TRANSFORM_NORMALIZATION,
            "facial_transform_source": "same_detection_mediapipe_video_mode",
            "timestamp_unit": "seconds",
            "timestamp_source": "source_frame_index_divided_by_audited_fps",
            "source_fps": np.asarray(0.0, dtype=np.float64),
            "producer_protocol": VIDEO_PRODUCER_PROTOCOL,
            "producer_adapter_version": VIDEO_ADAPTER_VERSION,
            "recording_id": recording_id,
            "group_id": group_id,
            "source_integrity_id": source_integrity_id,
            "source_fingerprint": source_fingerprint,
            "cache_schema": MEDIAPIPE_CACHE_SCHEMA,
            "development_only": np.asarray(True),
            "patient_identity": "unknown",
            "split_unit": "recording",
        }
    else:
        source_dtype, source_shape, _ = headers["features_60hz"]
        target_dtype, target_shape, _ = headers["features_30hz"]
        if (
            source_dtype != np.dtype(np.float32)
            or len(source_shape) != 2
            or source_shape[1] != len(ARKIT_BLENDSHAPE_NAMES)
            or not 1 <= source_shape[0] <= _MAX_MAYO_CACHE_ROWS
            or target_dtype != np.dtype(np.float32)
            or len(target_shape) != 2
            or target_shape[1] != len(ARKIT_BLENDSHAPE_NAMES)
            or not 1 <= target_shape[0] <= source_shape[0]
        ):
            raise ValueError("compact cache feature NPY headers are noncanonical")
        source_rows = source_shape[0]
        target_rows = target_shape[0]
        expected_arrays = {
            "features_60hz": (np.dtype(np.float32), (source_rows, 52)),
            "valid_mask_60hz": (np.dtype(np.bool_), (source_rows,)),
            "timestamps_60hz": (np.dtype(np.float64), (source_rows,)),
            "source_frame_indices_60hz": (np.dtype(np.int64), (source_rows,)),
            "features_30hz": (np.dtype(np.float32), (target_rows, 52)),
            "valid_mask_30hz": (np.dtype(np.bool_), (target_rows,)),
            "timestamps_30hz": (np.dtype(np.float64), (target_rows,)),
            "source_frame_indices_30hz": (np.dtype(np.int64), (target_rows,)),
            "target_frame_indices_30hz": (np.dtype(np.int64), (target_rows,)),
            "contiguous_from_previous_30hz": (
                np.dtype(np.bool_), (target_rows,),
            ),
        }
        metadata = {
            "feature_schema": "arkit_blendshapes_52_v1",
            "feature_names": np.asarray(ARKIT_BLENDSHAPE_NAMES),
            "timestamp_unit": "seconds",
            "timestamp_source": "arkit_original_timecode_relative_seconds",
            "recording_id": recording_id,
            "group_id": group_id,
            "source_integrity_id": source_integrity_id,
            "source_fingerprint": source_fingerprint,
            "cache_schema": ARKIT_CACHE_SCHEMA,
            "development_only": np.asarray(True),
            "patient_identity": "unknown",
            "split_unit": "recording",
        }
    expected_headers = dict(expected_arrays)
    expected_headers.update({
        name: (np.asarray(value).dtype, np.asarray(value).shape)
        for name, value in metadata.items()
    })
    for name, (expected_dtype, expected_shape) in expected_headers.items():
        dtype, shape, fortran_order = headers[name]
        if dtype != expected_dtype or shape != expected_shape or fortran_order:
            raise ValueError(f"compact cache {name} NPY header is noncanonical")


def _validate_compact_cache(
    path: Path,
    *,
    recording_id: str,
    group_id: str,
    source_integrity_id: str,
    source_fingerprint: str,
    expected_schema: str,
    salt: bytes | None = None,
    integrity_context: str | None = None,
    expected_integrity_id: str | None = None,
    parent_descriptor: int | None = None,
    held_descriptor: int | None = None,
    held_identity: tuple[int, ...] | None = None,
) -> tuple[str, int, CompactCacheSummary]:
    if (held_descriptor is None) != (held_identity is None):
        raise ValueError("compact cache held identity is incomplete")
    if held_descriptor is not None:
        if parent_descriptor is None or path.name != str(path):
            raise ValueError("compact cache held path is malformed")
        payload, digest, size = _read_regular_descriptor(
            held_descriptor,
            parent_descriptor=parent_descriptor,
            name=path.name,
            field="compact cache",
            expected_identity=held_identity,
            max_bytes=_MAX_MAYO_CACHE_RAW_BYTES,
        )
    else:
        payload, digest, size = _read_regular_bytes(
            path, "compact cache", max_bytes=_MAX_MAYO_CACHE_RAW_BYTES,
            parent_descriptor=parent_descriptor,
        )
    if salt is not None:
        if integrity_context is None or expected_integrity_id is None:
            raise ValueError("compact cache integrity inputs are incomplete")
        observed_integrity = hmac_identifier(
            "cache", salt, integrity_context, digest
        )
        if not hmac.compare_digest(observed_integrity, expected_integrity_id):
            raise ValueError("cache integrity ID does not bind the staged bytes")
    _require_mayo_npz_headers(
        payload,
        recording_id=recording_id,
        group_id=group_id,
        source_integrity_id=source_integrity_id,
        source_fingerprint=source_fingerprint,
        expected_schema=expected_schema,
    )
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as cached:
            observed_fields = tuple(cached.files)
            expected_fields = (
                _MEDIAPIPE_CACHE_FIELDS
                if expected_schema == MEDIAPIPE_CACHE_SCHEMA
                else _ARKIT_CACHE_FIELDS
                if expected_schema == ARKIT_CACHE_SCHEMA
                else None
            )
            if (
                expected_fields is None
                or len(observed_fields) != len(set(observed_fields))
                or set(observed_fields) != expected_fields
            ):
                raise ValueError("compact cache field schema is not exact")
            if expected_schema == MEDIAPIPE_CACHE_SCHEMA:
                summary = _validate_mediapipe_cache_payload(
                    cached, recording_id=recording_id, group_id=group_id,
                    source_integrity_id=source_integrity_id,
                    source_fingerprint=source_fingerprint,
                )
            else:
                summary = _validate_arkit_cache_payload(
                    cached, recording_id=recording_id, group_id=group_id,
                    source_integrity_id=source_integrity_id,
                    source_fingerprint=source_fingerprint,
                )
    except (OSError, ValueError, KeyError, TypeError, OverflowError, EOFError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("compact cache"):
            raise
        raise ValueError("compact cache cannot be validated") from exc
    return digest, size, summary


@dataclass(frozen=True)
class _HeldCommittedMayoCache:
    name: str
    descriptor: int = dataclass_field(repr=False)
    identity: tuple[int, ...]


@dataclass(frozen=True)
class _HeldCommittedMayoGeneration:
    output: Path
    exposure: Path
    output_parent_descriptor: int = dataclass_field(repr=False)
    output_parent_identity: tuple[int, ...]
    output_descriptor: int = dataclass_field(repr=False)
    output_identity: tuple[int, ...]
    collection_descriptor: int = dataclass_field(repr=False)
    collection_identity: tuple[int, ...]
    internal_exposure_descriptor: int = dataclass_field(repr=False)
    internal_exposure_identity: tuple[int, ...]
    media_descriptor: int = dataclass_field(repr=False)
    media_identity: tuple[int, ...]
    media_files: tuple[_HeldCommittedMayoCache, ...]
    arkit_descriptor: int = dataclass_field(repr=False)
    arkit_identity: tuple[int, ...]
    arkit_files: tuple[_HeldCommittedMayoCache, ...]
    external_parent_descriptor: int = dataclass_field(repr=False)
    external_parent_identity: tuple[int, ...]
    external_exposure_descriptor: int = dataclass_field(repr=False)
    external_exposure_identity: tuple[int, ...]


@contextmanager
def _hold_committed_mayo_generation(
    output: Path,
    exposure: Path,
    *,
    media_count: int | None = None,
    arkit_count: int | None = None,
    assert_on_exit: bool = True,
):
    if type(assert_on_exit) is not bool:
        raise ValueError("committed Mayo exit assertion flag is invalid")
    expected_media_count = (
        int(FROZEN_INVENTORY["long_unique_videos"])
        if media_count is None else media_count
    )
    expected_arkit_count = (
        int(FROZEN_INVENTORY["arkit_trajectories"])
        if arkit_count is None else arkit_count
    )
    if (
        not isinstance(expected_media_count, int)
        or isinstance(expected_media_count, bool)
        or expected_media_count < 0
        or not isinstance(expected_arkit_count, int)
        or isinstance(expected_arkit_count, bool)
        or expected_arkit_count < 0
    ):
        raise ValueError("committed Mayo held counts are invalid")
    descriptors = ExitStack()
    try:
        output_parent_descriptor = _open_nofollow_directory(
            output.parent, "committed Mayo output parent",
        )
        descriptors.callback(os.close, output_parent_descriptor)
        output_parent_identity = _directory_snapshot(
            os.fstat(output_parent_descriptor)
        )
        output_descriptor, output_identity = _open_nofollow_directory_at(
            output_parent_descriptor, output.name, "committed Mayo generation",
        )
        descriptors.callback(os.close, output_descriptor)
        collection_descriptor, collection_identity = _open_regular_at(
            output_descriptor,
            "collection_manifest.json",
            "committed collection manifest",
        )
        descriptors.callback(os.close, collection_descriptor)
        internal_exposure_descriptor, internal_exposure_identity = _open_regular_at(
            output_descriptor,
            "mayo_exposure_manifest.json",
            "committed internal exposure manifest",
        )
        descriptors.callback(os.close, internal_exposure_descriptor)
        media_descriptor, media_identity = _open_nofollow_directory_at(
            output_descriptor, "mediapipe", "committed Mayo MediaPipe cache",
        )
        descriptors.callback(os.close, media_descriptor)
        arkit_descriptor, arkit_identity = _open_nofollow_directory_at(
            output_descriptor, "arkit", "committed Mayo ARKit cache",
        )
        descriptors.callback(os.close, arkit_descriptor)

        def hold_cache_files(
            parent_descriptor: int,
            field: str,
            expected_count: int,
        ) -> tuple[_HeldCommittedMayoCache, ...]:
            names = sorted(os.listdir(parent_descriptor))
            if (
                len(names) != expected_count
                or any(Path(name).name != name or Path(name).suffix != ".npz"
                       for name in names)
            ):
                raise ValueError(f"{field} file set is incomplete or unsafe")
            held_files: list[_HeldCommittedMayoCache] = []
            for name in names:
                descriptor, identity = _open_regular_at(
                    parent_descriptor, name, field,
                )
                descriptors.callback(os.close, descriptor)
                held_files.append(_HeldCommittedMayoCache(
                    name=name,
                    descriptor=descriptor,
                    identity=identity,
                ))
            return tuple(held_files)

        media_files = hold_cache_files(
            media_descriptor,
            "committed Mayo MediaPipe cache",
            expected_media_count,
        )
        arkit_files = hold_cache_files(
            arkit_descriptor,
            "committed Mayo ARKit cache",
            expected_arkit_count,
        )

        external_parent_descriptor = _open_nofollow_directory(
            exposure.parent, "external exposure parent",
        )
        descriptors.callback(os.close, external_parent_descriptor)
        external_parent_identity = _directory_snapshot(
            os.fstat(external_parent_descriptor)
        )
        external_exposure_descriptor, external_exposure_identity = _open_regular_at(
            external_parent_descriptor,
            exposure.name,
            "committed external exposure manifest",
        )
        descriptors.callback(os.close, external_exposure_descriptor)
        regular_identities = (
            collection_identity,
            internal_exposure_identity,
            *(item.identity for item in media_files),
            *(item.identity for item in arkit_files),
            external_exposure_identity,
        )
        if (
            len(regular_identities) + 2 > 64
            or sum(int(identity[6]) for identity in regular_identities)
            > _MAX_EXACT_PRIVATE_TREE_REGULAR_BYTES
        ):
            raise ValueError(
                "committed Mayo generation exceeds its exact-tree budget"
            )
        held = _HeldCommittedMayoGeneration(
            output=output,
            exposure=exposure,
            output_parent_descriptor=output_parent_descriptor,
            output_parent_identity=output_parent_identity,
            output_descriptor=output_descriptor,
            output_identity=output_identity,
            collection_descriptor=collection_descriptor,
            collection_identity=collection_identity,
            internal_exposure_descriptor=internal_exposure_descriptor,
            internal_exposure_identity=internal_exposure_identity,
            media_descriptor=media_descriptor,
            media_identity=media_identity,
            media_files=media_files,
            arkit_descriptor=arkit_descriptor,
            arkit_identity=arkit_identity,
            arkit_files=arkit_files,
            external_parent_descriptor=external_parent_descriptor,
            external_parent_identity=external_parent_identity,
            external_exposure_descriptor=external_exposure_descriptor,
            external_exposure_identity=external_exposure_identity,
        )
        _assert_held_mayo_generation(held)
        yield held
        if assert_on_exit:
            _assert_held_mayo_generation(held)
    finally:
        descriptors.__exit__(*sys.exc_info())


def _assert_held_mayo_generation(held: _HeldCommittedMayoGeneration) -> None:
    def require_directory(
        descriptor: int,
        expected: tuple[int, ...],
        field: str,
    ) -> None:
        observed = os.fstat(descriptor)
        _require_private_directory_stat(observed, field)
        if (
            not stat.S_ISDIR(observed.st_mode)
            or _directory_snapshot(observed) != expected
        ):
            raise ValueError(f"{field} held descriptor changed")

    def require_directory_name(
        parent_descriptor: int,
        name: str,
        expected: tuple[int, ...],
        field: str,
    ) -> None:
        observed = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        _require_private_directory_stat(observed, field)
        if (
            not stat.S_ISDIR(observed.st_mode)
            or _directory_snapshot(observed) != expected
        ):
            raise ValueError(f"{field} name no longer binds its held directory")

    def require_parent_directory(
        descriptor: int,
        expected: tuple[int, ...],
        path: Path,
        field: str,
    ) -> None:
        opened = os.fstat(descriptor)
        linked = os.lstat(path)
        _require_private_directory_stat(opened, field)
        _require_private_directory_stat(linked, field)
        if (
            _directory_snapshot(opened)[:5] != expected[:5]
            or _directory_snapshot(linked)[:5] != expected[:5]
        ):
            raise ValueError(f"{field} identity changed")

    def require_file(
        descriptor: int,
        parent_descriptor: int,
        name: str,
        expected: tuple[int, ...],
        field: str,
    ) -> None:
        opened = os.fstat(descriptor)
        linked = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        _require_private_regular_stat(opened, field)
        _require_private_regular_stat(linked, field)
        if (
            not stat.S_ISREG(opened.st_mode)
            or int(opened.st_nlink) != 1
            or _regular_snapshot(opened) != expected
            or _regular_snapshot(linked) != expected
        ):
            raise ValueError(f"{field} name no longer binds its held file")

    require_parent_directory(
        held.output_parent_descriptor,
        held.output_parent_identity,
        held.output.parent,
        "committed Mayo output parent",
    )
    require_directory(
        held.output_descriptor, held.output_identity, "committed Mayo generation",
    )
    require_directory_name(
        held.output_parent_descriptor,
        held.output.name,
        held.output_identity,
        "committed Mayo generation",
    )
    require_file(
        held.collection_descriptor,
        held.output_descriptor,
        "collection_manifest.json",
        held.collection_identity,
        "committed collection manifest",
    )
    require_file(
        held.internal_exposure_descriptor,
        held.output_descriptor,
        "mayo_exposure_manifest.json",
        held.internal_exposure_identity,
        "committed internal exposure manifest",
    )
    require_directory(
        held.media_descriptor, held.media_identity, "committed Mayo MediaPipe cache",
    )
    require_directory_name(
        held.output_descriptor,
        "mediapipe",
        held.media_identity,
        "committed Mayo MediaPipe cache",
    )
    require_directory(
        held.arkit_descriptor, held.arkit_identity, "committed Mayo ARKit cache",
    )
    require_directory_name(
        held.output_descriptor,
        "arkit",
        held.arkit_identity,
        "committed Mayo ARKit cache",
    )
    for item in held.media_files:
        require_file(
            item.descriptor,
            held.media_descriptor,
            item.name,
            item.identity,
            "committed Mayo MediaPipe cache",
        )
    for item in held.arkit_files:
        require_file(
            item.descriptor,
            held.arkit_descriptor,
            item.name,
            item.identity,
            "committed Mayo ARKit cache",
        )
    require_parent_directory(
        held.external_parent_descriptor,
        held.external_parent_identity,
        held.exposure.parent,
        "external exposure parent",
    )
    require_file(
        held.external_exposure_descriptor,
        held.external_parent_descriptor,
        held.exposure.name,
        held.external_exposure_identity,
        "committed external exposure manifest",
    )


def _inventory_counts_sha256(value: Mapping[str, object]) -> str:
    if (
        not isinstance(value, Mapping)
        or set(value) != set(FROZEN_INVENTORY)
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 0
               for item in value.values())
    ):
        raise ValueError("inventory count commitment is noncanonical")
    return hashlib.sha256(json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("ascii")).hexdigest()


def _validate_staging(
    staging: Path,
    media_count: int | None = None,
    arkit_count: int | None = None,
    *,
    salt: bytes | None = None,
    expected_inventory_counts: Mapping[str, object] | None = None,
    expected_collection_classification_integrity_id: str | None = None,
    expected_classification_integrity_id: str | None = None,
    _held: _HeldCommittedMayoGeneration | None = None,
) -> dict[str, object]:
    allowed_top = {"collection_manifest.json", "mayo_exposure_manifest.json",
                   "mediapipe", "arkit"}
    if _held is None:
        staging = _require_private_generation_storage_tree(
            staging, "staging generation",
        )
        observed_top = {item.name for item in staging.iterdir()}
        if observed_top != allowed_top:
            raise ValueError(
                "staging generation has a stale, missing, or unexpected top-level file"
            )
        collection, collection_digest = _load_public_json(
            staging / "collection_manifest.json", "collection manifest"
        )
        exposure, exposure_digest = _load_public_json(
            staging / "mayo_exposure_manifest.json", "exposure manifest"
        )
    else:
        if Path(staging) != _held.output:
            raise ValueError("held Mayo generation path is inconsistent")
        _assert_held_mayo_generation(_held)
        observed_top = set(os.listdir(_held.output_descriptor))
        if observed_top != allowed_top:
            raise ValueError(
                "staging generation has a stale, missing, or unexpected top-level file"
            )
        collection, collection_digest = _load_public_json_descriptor(
            _held.collection_descriptor,
            parent_descriptor=_held.output_descriptor,
            name="collection_manifest.json",
            field="collection manifest",
            expected_identity=_held.collection_identity,
        )
        exposure, exposure_digest = _load_public_json_descriptor(
            _held.internal_exposure_descriptor,
            parent_descriptor=_held.output_descriptor,
            name="mayo_exposure_manifest.json",
            field="exposure manifest",
            expected_identity=_held.internal_exposure_identity,
        )
    collection_counts = _validate_collection_top(collection)
    inventory_counts_sha256 = _inventory_counts_sha256(collection_counts)
    if expected_inventory_counts is not None:
        expected_counts = dict(expected_inventory_counts)
        _inventory_counts_sha256(expected_counts)
        if collection_counts != expected_counts:
            raise ValueError("collection counts disagree with the caller inventory")
    _require_exact_object(exposure, _EXPOSURE_TOP_FIELDS, "exposure manifest")
    if (
        exposure["schema_version"] != EXPOSURE_SCHEMA
        or exposure["dataset"] != EXPOSURE_DATASET
        or exposure["policy"] != EXPOSURE_POLICY
        or exposure["identity_status"] != UNKNOWN_IDENTITY
    ):
        raise ValueError("exposure manifest top-level policy is noncanonical")
    media_rows = _manifest_cache_rows(
        collection, "mediapipe_records", "mediapipe"
    )
    arkit_rows = _manifest_cache_rows(collection, "arkit_records", "arkit")
    collection_classification_integrity = collection[
        "classification_integrity_id"
    ]
    if (
        not isinstance(collection_classification_integrity, str)
        or _INTEGRITY_ID.fullmatch(collection_classification_integrity) is None
        or not collection_classification_integrity.startswith("agg_")
    ):
        raise ValueError("collection classification integrity ID is noncanonical")
    if expected_collection_classification_integrity_id is not None:
        if (
            not isinstance(expected_collection_classification_integrity_id, str)
            or _INTEGRITY_ID.fullmatch(
                expected_collection_classification_integrity_id
            ) is None
            or not expected_collection_classification_integrity_id.startswith("agg_")
        ):
            raise ValueError("caller collection classification is noncanonical")
        if not hmac.compare_digest(
            collection_classification_integrity,
            expected_collection_classification_integrity_id,
        ):
            raise ValueError("collection classification disagrees with caller inventory")
    if salt is not None:
        recomputed_collection_classification = (
            collection_classification_integrity_id(
                list(media_rows.values()), salt
            )
        )
        if not hmac.compare_digest(
            collection_classification_integrity,
            recomputed_collection_classification,
        ):
            raise ValueError("collection classification HMAC is invalid")
    exposure_videos = exposure.get("videos")
    exposure_arkit = exposure.get("arkit_trajectories")
    exposure_counts = exposure.get("counts")
    if (
        not isinstance(exposure_videos, list)
        or not isinstance(exposure_arkit, list)
        or not isinstance(exposure_counts, dict)
        or set(exposure_counts) != {"videos", "arkit_trajectories"}
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 0
               for value in exposure_counts.values())
        or exposure_counts != {
            "videos": len(exposure_videos),
            "arkit_trajectories": len(exposure_arkit),
        }
    ):
        raise ValueError("exposure manifest counts or record lists are noncanonical")
    classification_integrity = exposure["classification_integrity_id"]
    if (
        not isinstance(classification_integrity, str)
        or _INTEGRITY_ID.fullmatch(classification_integrity) is None
        or not classification_integrity.startswith("agg_")
    ):
        raise ValueError("exposure classification integrity ID is noncanonical")
    if expected_classification_integrity_id is not None:
        if (
            not isinstance(expected_classification_integrity_id, str)
            or _INTEGRITY_ID.fullmatch(expected_classification_integrity_id) is None
            or not expected_classification_integrity_id.startswith("agg_")
        ):
            raise ValueError("caller classification commitment is noncanonical")
        if not hmac.compare_digest(
            classification_integrity, expected_classification_integrity_id
        ):
            raise ValueError("exposure classification disagrees with the caller inventory")
    if salt is not None:
        recomputed_classification = exposure_classification_integrity_id(
            exposure_videos, salt
        )
        if not hmac.compare_digest(classification_integrity,
                                   recomputed_classification):
            raise ValueError("exposure classification HMAC is invalid")
    exposure_media_rows: dict[str, Mapping[str, object]] = {}
    exposure_instance_ids: set[str] = set()
    duplicate_rows: list[Mapping[str, object]] = []
    short_rows: list[Mapping[str, object]] = []
    for item in exposure_videos:
        if not isinstance(item, dict) or item.get("status") not in {
            "mediapipe_ssl", "exact_duplicate_excluded", "qc_only_short_clip_excluded"
        }:
            raise ValueError("exposure video status is noncanonical")
        status = str(item["status"])
        fields = (
            _EXPOSURE_MEDIA_FIELDS if status == "mediapipe_ssl"
            else _EXPOSURE_EXCLUDED_FIELDS
        )
        row = _require_exact_object(item, fields, "exposure video record")
        identity_fields = dict(row)
        if status != "mediapipe_ssl":
            identity_fields["cache_integrity_id"] = "cache_" + "0" * 64
        _require_public_identity(identity_fields, "exposure video record")
        _require_public_governance(row, "exposure video record")
        instance_id = row["instance_id"]
        if (
            not isinstance(instance_id, str) or _INSTANCE_ID.fullmatch(instance_id) is None
            or instance_id in exposure_instance_ids
        ):
            raise ValueError("exposure video instance ID is missing or repeated")
        exposure_instance_ids.add(instance_id)
        recording_id = str(row["recording_id"])
        if status == "mediapipe_ssl":
            if recording_id in exposure_media_rows:
                raise ValueError("exposure manifest repeats a MediaPipe cache record")
            exposure_media_rows[recording_id] = row
        elif status == "exact_duplicate_excluded":
            duplicate_rows.append(row)
        else:
            short_rows.append(row)
    exposure_arkit_rows = _manifest_cache_rows(
        {"arkit_trajectories": exposure_arkit}, "arkit_trajectories", "arkit"
    )
    if set(exposure_media_rows) != set(media_rows) or set(exposure_arkit_rows) != set(arkit_rows):
        raise ValueError("collection and exposure cache record sets disagree")
    shared_fields = {
        "recording_id", "group_id", "source_integrity_id", "source_fingerprint",
        "cache_integrity_id", "identity_status", "split_unit", "development_only",
        "ssl_exposed", "independent_evaluation_eligible",
    }
    for recording_id, row in media_rows.items():
        exposed = exposure_media_rows[recording_id]
        if any(exposed[field] != row[field] for field in shared_fields):
            raise ValueError("MediaPipe collection and exposure records disagree")
    for recording_id, row in arkit_rows.items():
        if exposure_arkit_rows[recording_id] != row:
            raise ValueError("ARKit collection and exposure records disagree")
    media_group_bindings = {
        (row["recording_id"], row["group_id"], row["source_integrity_id"],
         row["source_fingerprint"])
        for row in exposure_media_rows.values()
    }
    if any(
        (row["recording_id"], row["group_id"], row["source_integrity_id"],
         row["source_fingerprint"]) not in media_group_bindings
        for row in duplicate_rows
    ):
        raise ValueError("excluded duplicate does not bind a retained MediaPipe source")
    protected_rows = (*exposure_media_rows.values(), *duplicate_rows)
    for field in (
        "recording_id", "group_id", "source_integrity_id", "source_fingerprint",
    ):
        protected = {row[field] for row in protected_rows}
        short_values = [row[field] for row in short_rows]
        if protected.intersection(short_values) or len(short_values) != len(
            set(short_values)
        ):
            raise ValueError("short-clip exposure identities must be unique and disjoint")
    observed_media_count = len(media_rows)
    observed_arkit_count = len(arkit_rows)
    if (
        collection_counts["total_sessions"]
        != collection_counts["video_bearing_sessions"]
        + collection_counts["without_video_sessions"]
        or collection_counts["without_video_sessions"]
        != collection_counts["arkit_only_sessions"]
        + collection_counts["metadata_only_sessions"]
        or collection_counts["video_bearing_sessions"] != len(exposure_videos)
        or collection_counts["long_unique_videos"] != observed_media_count
        or collection_counts["exact_duplicate_copies_excluded"] != len(duplicate_rows)
        or collection_counts["short_qc_clips_excluded"] != len(short_rows)
        or collection_counts["existing_complete_v2_exports"]
        != sum(row["legacy_export_audit_status"]
               == "not_reused_unverifiable_source_binding"
               for row in media_rows.values())
        or collection_counts["remaining_long_videos"]
        != sum(row["legacy_export_audit_status"] == "no_complete_legacy_export"
               for row in media_rows.values())
        or collection_counts["existing_complete_v2_exports"]
        + collection_counts["remaining_long_videos"]
        != collection_counts["long_unique_videos"]
        or collection_counts["arkit_trajectories"] != observed_arkit_count
    ):
        raise ValueError("collection counts disagree with exact staged records")
    if media_count is not None and observed_media_count != media_count:
        raise ValueError("staged MediaPipe manifest count is incomplete")
    if arkit_count is not None and observed_arkit_count != arkit_count:
        raise ValueError("staged ARKit manifest count is incomplete")
    cache_commitments: list[tuple[str, str, int]] = []
    remaining_media_rows = 0
    observed_arkit_rows = 0
    observed_arkit_gaps = 0
    for dirname, rows, expected_schema, context in (
        ("mediapipe", media_rows, MEDIAPIPE_CACHE_SCHEMA,
         "mayo-mediapipe-cache-integrity"),
        ("arkit", arkit_rows, ARKIT_CACHE_SCHEMA,
         "mayo-arkit-cache-integrity"),
    ):
        close_directory = _held is None
        if _held is None:
            directory = _require_directory(
                staging / dirname, f"staged {dirname} cache",
            )
            directory_descriptor = _open_nofollow_directory(
                directory, f"staged {dirname} cache",
            )
        else:
            directory_descriptor = (
                _held.media_descriptor
                if dirname == "mediapipe"
                else _held.arkit_descriptor
            )
            held_files = {
                item.name: item for item in (
                    _held.media_files
                    if dirname == "mediapipe"
                    else _held.arkit_files
                )
            }
            _assert_held_mayo_generation(_held)
        try:
            names = sorted(os.listdir(directory_descriptor))
            expected_names = {f"{recording_id}.npz" for recording_id in rows}
            if set(names) != expected_names:
                raise ValueError(
                    f"staged {dirname} cache file set is incomplete or unsafe"
                )
            for name in names:
                info = os.stat(
                    name, dir_fd=directory_descriptor, follow_symlinks=False
                )
                _require_private_regular_stat(
                    info, f"staged {dirname} compact cache",
                )
                if Path(name).suffix != ".npz":
                    raise ValueError(
                        f"staged {dirname} cache file set is incomplete or unsafe"
                    )
                path = Path(name)
                recording_id = path.stem
                row = rows[recording_id]
                held_file = None if _held is None else held_files.get(name)
                if _held is not None and held_file is None:
                    raise ValueError("held compact cache file set is incomplete")
                digest, size, summary = _validate_compact_cache(
                    path,
                    recording_id=recording_id,
                    group_id=str(row["group_id"]),
                    source_integrity_id=str(row["source_integrity_id"]),
                    source_fingerprint=str(row["source_fingerprint"]),
                    expected_schema=expected_schema,
                    salt=salt,
                    integrity_context=context,
                    expected_integrity_id=str(row["cache_integrity_id"]),
                    parent_descriptor=directory_descriptor,
                    held_descriptor=(
                        None if held_file is None else held_file.descriptor
                    ),
                    held_identity=(
                        None if held_file is None else held_file.identity
                    ),
                )
                if dirname == "mediapipe":
                    if row["legacy_export_audit_status"] == "no_complete_legacy_export":
                        remaining_media_rows += summary.source_rows
                else:
                    observed_arkit_rows += summary.source_rows
                    observed_arkit_gaps += summary.missing_source_frames
                cache_commitments.append((f"{dirname}/{path.name}", digest, size))
        finally:
            if close_directory:
                os.close(directory_descriptor)
    if (
        collection_counts["remaining_long_video_frames"] != remaining_media_rows
        or collection_counts["arkit_rows"] != observed_arkit_rows
        or collection_counts["arkit_timecode_gaps"] != observed_arkit_gaps
    ):
        raise ValueError("collection temporal totals disagree with private cache arrays")
    if _held is None and (
        list(staging.rglob("*.csv"))
        or list(staging.rglob("*.mp4"))
        or list(staging.rglob("*.mov"))
    ):
        raise ValueError("staged generation must not contain raw CSV or preview video")
    cache_aggregate = hashlib.sha256()
    for relative_name, digest, size in cache_commitments:
        cache_aggregate.update(
            f"{relative_name}:{digest}:{size}\n".encode("ascii")
        )
    generation_aggregate = hashlib.sha256()
    generation_aggregate.update(f"collection:{collection_digest}\n".encode("ascii"))
    generation_aggregate.update(f"exposure:{exposure_digest}\n".encode("ascii"))
    generation_aggregate.update(
        f"caches:{cache_aggregate.hexdigest()}\n".encode("ascii")
    )
    result = {
        "schema": "mayo_cache_generation_commitment_v3",
        "collection_manifest_sha256": collection_digest,
        "exposure_manifest_sha256": exposure_digest,
        "mediapipe_file_count": observed_media_count,
        "arkit_file_count": observed_arkit_count,
        "cache_file_count": len(cache_commitments),
        "cache_tree_aggregate_sha256": cache_aggregate.hexdigest(),
        "generation_aggregate_sha256": generation_aggregate.hexdigest(),
        "inventory_counts_sha256": inventory_counts_sha256,
        "collection_classification_integrity_id": (
            collection_classification_integrity
        ),
        "exposure_classification_integrity_id": classification_integrity,
    }
    if _held is not None:
        _assert_held_mayo_generation(_held)
    return result


def _validate_generation_commitment(value: object) -> dict[str, object]:
    required = {
        "schema", "collection_manifest_sha256", "exposure_manifest_sha256",
        "mediapipe_file_count", "arkit_file_count", "cache_file_count",
        "cache_tree_aggregate_sha256", "generation_aggregate_sha256",
        "inventory_counts_sha256", "collection_classification_integrity_id",
        "exposure_classification_integrity_id",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("transaction generation commitment has a noncanonical schema")
    if value["schema"] != "mayo_cache_generation_commitment_v3":
        raise ValueError("transaction generation commitment has the wrong version")
    for key in (
        "collection_manifest_sha256", "exposure_manifest_sha256",
        "cache_tree_aggregate_sha256", "generation_aggregate_sha256",
        "inventory_counts_sha256",
    ):
        if not isinstance(value[key], str) or _SHA256.fullmatch(value[key]) is None:
            raise ValueError("transaction generation commitment digest is invalid")
    for key in (
        "collection_classification_integrity_id",
        "exposure_classification_integrity_id",
    ):
        classification = value[key]
        if (
            not isinstance(classification, str)
            or _INTEGRITY_ID.fullmatch(classification) is None
            or not classification.startswith("agg_")
        ):
            raise ValueError("transaction classification commitment is invalid")
    for key in ("mediapipe_file_count", "arkit_file_count", "cache_file_count"):
        if not isinstance(value[key], int) or isinstance(value[key], bool) or value[key] < 0:
            raise ValueError("transaction generation commitment count is invalid")
    if value["cache_file_count"] != (
        value["mediapipe_file_count"] + value["arkit_file_count"]
    ):
        raise ValueError("transaction generation commitment counts disagree")
    return dict(value)


def _assert_committed_generation(
    output: Path,
    exposure: Path,
    commitment: Mapping[str, object],
    *,
    salt: bytes | None = None,
    expected_inventory_counts: Mapping[str, object] | None = None,
    expected_collection_classification_integrity_id: str | None = None,
    expected_classification_integrity_id: str | None = None,
    _held: _HeldCommittedMayoGeneration | None = None,
) -> None:
    _require_private_directory(output.parent, "committed Mayo output parent")
    _require_private_directory(exposure.parent, "external exposure parent")
    expected = _validate_generation_commitment(dict(commitment))
    observed = _validate_staging(
        output,
        int(expected["mediapipe_file_count"]),
        int(expected["arkit_file_count"]),
        salt=salt,
        expected_inventory_counts=expected_inventory_counts,
        expected_collection_classification_integrity_id=(
            expected_collection_classification_integrity_id
        ),
        expected_classification_integrity_id=expected_classification_integrity_id,
        _held=_held,
    )
    if observed != expected:
        raise ValueError("committed cache generation no longer matches its journal")
    if _held is None:
        _exposure, exposure_digest = _load_public_json(
            exposure, "committed external exposure manifest"
        )
    else:
        if _held.output != output or _held.exposure != exposure:
            raise ValueError("held committed Mayo paths are inconsistent")
        _assert_held_mayo_generation(_held)
        _exposure, exposure_digest = _load_public_json_descriptor(
            _held.external_exposure_descriptor,
            parent_descriptor=_held.external_parent_descriptor,
            name=exposure.name,
            field="committed external exposure manifest",
            expected_identity=_held.external_exposure_identity,
        )
    if not hmac.compare_digest(
        exposure_digest, str(expected["exposure_manifest_sha256"])
    ):
        raise ValueError("committed external exposure manifest changed")
    if _held is not None:
        _assert_held_mayo_generation(_held)


def _assert_no_unresolved_generation_state(output: Path, exposure: Path) -> None:
    candidates = [
        _journal_path(output),
        *output.parent.glob(f".{output.name}.staging-*"),
        *output.parent.glob(f".{output.name}.backup-*"),
        *exposure.parent.glob(f".{exposure.name}.backup-*"),
        *exposure.parent.glob(f".{exposure.name}.tmp-*"),
    ]
    if any(path.exists() or _is_symlink(path) for path in candidates):
        raise RuntimeError(
            "Mayo committed authorization rejects unresolved transaction state"
        )


def _key_file_identity(path: Path) -> tuple[int, ...]:
    info = os.stat(path, follow_symlinks=False)
    return (
        info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
        info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
    )


def _immutable_array(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy()
    result.flags.writeable = False
    return result


def authorize_committed_mayo_ssl_generation(
    data_root: str | Path,
    existing_export_root: str | Path,
    salt_file: str | Path,
    output_root: str | Path,
    exposure_manifest: str | Path,
) -> AuthorizedMayoGeneration:
    """Live-authorize the coupled Mayo cache/exposure generation read-only.

    The commitment is always recomputed from live bytes and the canonical key;
    callers cannot provide or recover a transaction commitment.
    """
    data = _require_directory(data_root, "Mayo data root")
    exports = _require_directory(existing_export_root, "existing MediaPipe export root")
    output, exposure = validate_output_locations(
        output_root, exposure_manifest, project_root=PROJECT_ROOT
    )
    validate_source_output_separation(data, exports, output, exposure)
    key_path = _assert_no_symlink_components(salt_file)
    expected_key = (
        PROJECT_ROOT / "outputs" / "dynamic_landmark" / "pretraining"
        / ".mayo_ssl_hmac.key"
    )
    if key_path != expected_key:
        raise ValueError("Mayo authorization requires the canonical private key")
    before_key_identity = _key_file_identity(key_path)
    salt = read_canonical_salt(key_path, project_root=PROJECT_ROOT)
    if len(salt) != 32:
        raise ValueError("Mayo authorization key must contain exactly 32 bytes")
    if _key_file_identity(key_path) != before_key_identity:
        raise ValueError("Mayo private key changed while it was read")
    inventory = inventory_mayo_sources(data, exports, enforce_frozen=True)
    if not isinstance(inventory, MayoInventory) or inventory.counts != FROZEN_INVENTORY:
        raise ValueError("Mayo live inventory does not match the frozen contract")
    expected_collection, expected_exposure = build_public_manifests(inventory, salt)
    expected_collection_classification = str(
        expected_collection["classification_integrity_id"]
    )
    expected_exposure_classification = str(
        expected_exposure["classification_integrity_id"]
    )
    with output_parent_lock(output, create_if_missing=False), \
            _hold_committed_mayo_generation(output, exposure) as held:
        _assert_no_unresolved_generation_state(output, exposure)
        commitment = _validate_staging(
            output,
            int(FROZEN_INVENTORY["long_unique_videos"]),
            int(FROZEN_INVENTORY["arkit_trajectories"]),
            salt=salt,
            expected_inventory_counts=inventory.counts,
            expected_collection_classification_integrity_id=(
                expected_collection_classification
            ),
            expected_classification_integrity_id=expected_exposure_classification,
            _held=held,
        )
        _external_exposure, external_digest = _load_public_json_descriptor(
            held.external_exposure_descriptor,
            parent_descriptor=held.external_parent_descriptor,
            name=held.exposure.name,
            field="committed external exposure manifest",
            expected_identity=held.external_exposure_identity,
        )
        if not hmac.compare_digest(
            external_digest, str(commitment["exposure_manifest_sha256"])
        ):
            raise ValueError("external Mayo exposure manifest is not coupled to cache")
        collection, collection_digest = _load_public_json_descriptor(
            held.collection_descriptor,
            parent_descriptor=held.output_descriptor,
            name="collection_manifest.json",
            field="committed collection manifest",
            expected_identity=held.collection_identity,
        )
        if not hmac.compare_digest(
            collection_digest, str(commitment["collection_manifest_sha256"])
        ):
            raise ValueError("committed collection manifest changed after validation")
        media_rows = _manifest_cache_rows(
            collection, "mediapipe_records", "mediapipe"
        )
        if len(media_rows) != int(FROZEN_INVENTORY["long_unique_videos"]):
            raise ValueError("Mayo main bridge cache count is not exact")
        expected_names = {f"{recording_id}.npz" for recording_id in media_rows}
        recordings: list[AuthorizedMayoRecording] = []
        closure_rows: list[dict[str, object]] = []
        media_descriptor = held.media_descriptor
        try:
            _assert_held_mayo_generation(held)
            names = sorted(os.listdir(media_descriptor))
            if set(names) != expected_names:
                raise ValueError("Mayo main bridge cache filename set is incomplete")
            held_media = {item.name: item for item in held.media_files}
            if set(held_media) != expected_names:
                raise ValueError("held Mayo MediaPipe cache set is incomplete")
            for name in names:
                path = Path(name)
                row = media_rows[path.stem]
                held_file = held_media[name]
                payload, digest, size = _read_regular_descriptor(
                    held_file.descriptor,
                    parent_descriptor=media_descriptor,
                    name=name,
                    field="committed Mayo MediaPipe cache",
                    expected_identity=held_file.identity,
                    max_bytes=_MAX_MAYO_CACHE_RAW_BYTES,
                )
                expected_integrity = hmac_identifier(
                    "cache", salt, "mayo-mediapipe-cache-integrity", digest
                )
                if not hmac.compare_digest(
                    expected_integrity, str(row["cache_integrity_id"])
                ):
                    raise ValueError("Mayo cache integrity ID does not bind cache bytes")
                _require_mayo_npz_headers(
                    payload,
                    recording_id=path.stem,
                    group_id=str(row["group_id"]),
                    source_integrity_id=str(row["source_integrity_id"]),
                    source_fingerprint=str(row["source_fingerprint"]),
                    expected_schema=MEDIAPIPE_CACHE_SCHEMA,
                )
                try:
                    with np.load(io.BytesIO(payload), allow_pickle=False) as cached:
                        if (
                            len(cached.files) != len(set(cached.files))
                            or set(cached.files) != _MEDIAPIPE_CACHE_FIELDS
                        ):
                            raise ValueError("Mayo MediaPipe cache field schema is not exact")
                        _validate_mediapipe_cache_payload(
                            cached,
                            recording_id=path.stem,
                            group_id=str(row["group_id"]),
                            source_integrity_id=str(row["source_integrity_id"]),
                            source_fingerprint=str(row["source_fingerprint"]),
                        )
                        recording = AuthorizedMayoRecording(
                            recording_id=path.stem,
                            group_id=str(row["group_id"]),
                            cache_integrity_id=str(row["cache_integrity_id"]),
                            cache_sha256=digest,
                            cache_size_bytes=size,
                            features_30hz=_immutable_array(cached["features_30hz"]),
                            valid_mask_30hz=_immutable_array(cached["valid_mask_30hz"]),
                            timestamps_30hz=_immutable_array(cached["timestamps_30hz"]),
                            source_frame_indices_30hz=_immutable_array(
                                cached["source_frame_indices_30hz"]
                            ),
                            target_frame_indices_30hz=_immutable_array(
                                cached["target_frame_indices_30hz"]
                            ),
                        )
                except (OSError, EOFError, KeyError, TypeError, ValueError) as exc:
                    if isinstance(exc, ValueError) and str(exc).startswith("Mayo MediaPipe"):
                        raise
                    raise ValueError(
                        "Mayo MediaPipe cache is not a safe exact NPZ"
                    ) from exc
                recordings.append(recording)
                closure_rows.append({
                    "recording_id": recording.recording_id,
                    "group_id": recording.group_id,
                    "cache_integrity_id": recording.cache_integrity_id,
                    "cache_sha256": recording.cache_sha256,
                    "cache_size_bytes": recording.cache_size_bytes,
                })
        finally:
            _assert_held_mayo_generation(held)
        if len({recording.recording_id for recording in recordings}) != len(recordings):
            raise ValueError("Mayo main bridge repeats a recording")
        if len({recording.group_id for recording in recordings}) != len(recordings):
            raise ValueError("Mayo main bridge requires one exact recording group each")
        closure_material = json.dumps({
            "commitment": commitment,
            "collection_manifest_sha256": collection_digest,
            "recordings": closure_rows,
        }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
        closure_hmac = hmac.new(
            salt,
            b"mayo-ssl-committed-generation-v1\0" + closure_material,
            hashlib.sha256,
        ).hexdigest()

        # Revalidate all live commitments immediately before returning.  This
        # deliberately does not invoke transaction recovery or mutate output.
        _assert_no_unresolved_generation_state(output, exposure)
        repeated = _validate_staging(
            output,
            int(FROZEN_INVENTORY["long_unique_videos"]),
            int(FROZEN_INVENTORY["arkit_trajectories"]),
            salt=salt,
            expected_inventory_counts=inventory.counts,
            expected_collection_classification_integrity_id=(
                expected_collection_classification
            ),
            expected_classification_integrity_id=expected_exposure_classification,
            _held=held,
        )
        if repeated != commitment:
            raise ValueError("Mayo cache generation changed during authorization")
        _repeated_exposure, repeated_exposure_digest = _load_public_json_descriptor(
            held.external_exposure_descriptor,
            parent_descriptor=held.external_parent_descriptor,
            name=held.exposure.name,
            field="committed external exposure manifest",
            expected_identity=held.external_exposure_identity,
        )
        if not hmac.compare_digest(repeated_exposure_digest, external_digest):
            raise ValueError("Mayo exposure manifest changed during authorization")
        repeated_inventory = inventory_mayo_sources(data, exports, enforce_frozen=True)
        if (
            not isinstance(repeated_inventory, MayoInventory)
            or repeated_inventory.counts != inventory.counts
        ):
            raise ValueError("Mayo live inventory changed during authorization")
        repeated_collection, repeated_exposure = build_public_manifests(
            repeated_inventory, salt
        )
        if (
            not hmac.compare_digest(
                str(repeated_collection["classification_integrity_id"]),
                expected_collection_classification,
            )
            or not hmac.compare_digest(
                str(repeated_exposure["classification_integrity_id"]),
                expected_exposure_classification,
            )
        ):
            raise ValueError("Mayo live classification changed during authorization")
        final_salt = read_canonical_salt(key_path, project_root=PROJECT_ROOT)
        if (
            not hmac.compare_digest(final_salt, salt)
            or _key_file_identity(key_path) != before_key_identity
        ):
            raise ValueError("Mayo private key changed during authorization")
        _assert_held_mayo_generation(held)
        return AuthorizedMayoGeneration(
            schema=MEDIAPIPE_CACHE_SCHEMA,
            collection_manifest_sha256=collection_digest,
            exposure_manifest_sha256=external_digest,
            generation_closure_hmac=closure_hmac,
            recording_count=len(recordings),
            arkit_count=int(commitment["arkit_file_count"]),
            expected_recording_count=int(FROZEN_INVENTORY["long_unique_videos"]),
            commitment=dict(commitment),
            recordings=tuple(recordings),
            private_key=salt,
        )


def validate_output_locations(
    output_root: str | Path,
    exposure_manifest: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> tuple[Path, Path]:
    """Confine biometric artifacts to the two exact repository-ignored layouts."""
    output = _assert_no_symlink_components(output_root)
    exposure = _assert_no_symlink_components(exposure_manifest)
    root = _assert_no_symlink_components(project_root)
    expected_output = (
        root / "outputs" / "dynamic_landmark" / "pretraining" / "mayo_ssl_cache"
    )
    expected_exposure = (
        root / "outputs" / "dynamic_landmark" / "mayo_exposure_manifest.json"
    )
    if output != expected_output:
        raise ValueError(
            "Mayo SSL cache path must exactly equal PROJECT_ROOT/outputs/"
            "dynamic_landmark/pretraining/mayo_ssl_cache"
        )
    if exposure != expected_exposure:
        raise ValueError(
            "exposure manifest path must exactly equal PROJECT_ROOT/outputs/"
            "dynamic_landmark/mayo_exposure_manifest.json"
        )
    return output, exposure


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def validate_source_output_separation(
    data_root: str | Path,
    existing_export_root: str | Path,
    output_root: str | Path,
    exposure_manifest: str | Path,
) -> None:
    data = _lexical_absolute(data_root)
    exports = _lexical_absolute(existing_export_root)
    output = _lexical_absolute(output_root)
    exposure = _lexical_absolute(exposure_manifest)
    for protected in (data, exports):
        if _paths_overlap(protected, output) or _paths_overlap(protected, exposure):
            raise ValueError("derived cache/exposure paths must not overlap source roots")


@dataclass(frozen=True)
class _HeldCanonicalMayoKey:
    path: Path
    descriptor: int = dataclass_field(repr=False)
    identity: tuple[int, ...]
    key_bytes: bytes = dataclass_field(repr=False)
    owner_uid: int

    def assert_unchanged(self) -> None:
        _assert_canonical_mayo_key_unchanged(self)


def _canonical_mayo_key_path(
    salt_file: str | Path,
    *,
    project_root: str | Path,
) -> Path:
    root = _assert_no_symlink_components(project_root)
    expected = (
        root / "outputs" / "dynamic_landmark" / "pretraining" / ".mayo_ssl_hmac.key"
    )
    path = _assert_no_symlink_components(salt_file)
    if path != expected:
        raise ValueError("HMAC salt path must exactly equal the canonical ignored key path")
    _require_owned_nonwritable_directory(root / "outputs", "outputs directory")
    _require_private_directory(
        root / "outputs" / "dynamic_landmark", "dynamic-landmark output directory",
    )
    _require_private_directory(
        root / "outputs" / "dynamic_landmark" / "pretraining",
        "pretraining output directory",
    )
    return path


def _open_canonical_mayo_key(path: Path) -> int:
    return os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _require_canonical_mayo_key_stat(
    info: os.stat_result,
    *,
    owner_uid: int,
) -> None:
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("HMAC salt must be a regular file")
    if int(info.st_uid) != owner_uid:
        raise ValueError("HMAC salt must be owned by the current user")
    if stat.S_IMODE(info.st_mode) != 0o600:
        raise ValueError("HMAC salt must have exact mode 0600")
    if int(info.st_nlink) != 1:
        raise ValueError("HMAC salt must have exactly one hard link")
    if int(info.st_size) != 32:
        raise ValueError("canonical Mayo HMAC key must contain exactly 32 bytes")


def _read_canonical_mayo_key_descriptor(
    descriptor: int,
    *,
    path: Path,
    owner_uid: int,
) -> tuple[bytes, tuple[int, ...]]:
    before = os.fstat(descriptor)
    linked_before = os.stat(path, follow_symlinks=False)
    _require_canonical_mayo_key_stat(before, owner_uid=owner_uid)
    _require_canonical_mayo_key_stat(linked_before, owner_uid=owner_uid)
    identity = _regular_snapshot(before)
    if _regular_snapshot(linked_before) != identity:
        raise ValueError("canonical Mayo HMAC key path changed while it was opened")
    os.lseek(descriptor, 0, os.SEEK_SET)
    payload = bytearray()
    while len(payload) <= 32:
        block = os.read(descriptor, 33 - len(payload))
        if not block:
            break
        payload.extend(block)
    after = os.fstat(descriptor)
    linked_after = os.stat(path, follow_symlinks=False)
    _require_canonical_mayo_key_stat(after, owner_uid=owner_uid)
    _require_canonical_mayo_key_stat(linked_after, owner_uid=owner_uid)
    if (
        _regular_snapshot(after) != identity
        or _regular_snapshot(linked_after) != identity
    ):
        raise ValueError("canonical Mayo HMAC key changed while it was read")
    if len(payload) != 32:
        raise ValueError("canonical Mayo HMAC key must contain exactly 32 bytes")
    return bytes(payload), identity


def _assert_canonical_mayo_key_unchanged(
    held: _HeldCanonicalMayoKey,
) -> None:
    payload, identity = _read_canonical_mayo_key_descriptor(
        held.descriptor,
        path=held.path,
        owner_uid=held.owner_uid,
    )
    if (
        identity != held.identity
        or not hmac.compare_digest(payload, held.key_bytes)
    ):
        raise ValueError("canonical Mayo HMAC key changed while held")
    reopened = _open_canonical_mayo_key(held.path)
    try:
        reopened_payload, reopened_identity = _read_canonical_mayo_key_descriptor(
            reopened,
            path=held.path,
            owner_uid=held.owner_uid,
        )
    finally:
        os.close(reopened)
    if (
        reopened_identity != held.identity
        or not hmac.compare_digest(reopened_payload, held.key_bytes)
    ):
        raise ValueError("canonical Mayo HMAC key changed at its live path")


@contextmanager
def _hold_canonical_mayo_key(
    salt_file: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
    owner_uid: int | None = None,
):
    path = _canonical_mayo_key_path(salt_file, project_root=project_root)
    expected_owner = os.geteuid() if owner_uid is None else owner_uid
    descriptors = ExitStack()
    try:
        descriptor = _open_canonical_mayo_key(path)
        descriptors.callback(os.close, descriptor)
        key_bytes, identity = _read_canonical_mayo_key_descriptor(
            descriptor,
            path=path,
            owner_uid=expected_owner,
        )
        held = _HeldCanonicalMayoKey(
            path=path,
            descriptor=descriptor,
            identity=identity,
            key_bytes=key_bytes,
            owner_uid=expected_owner,
        )
        held.assert_unchanged()
        yield held
        held.assert_unchanged()
    finally:
        descriptors.__exit__(*sys.exc_info())


def read_canonical_salt(
    salt_file: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
    owner_uid: int | None = None,
) -> bytes:
    with _hold_canonical_mayo_key(
        salt_file,
        project_root=project_root,
        owner_uid=owner_uid,
    ) as held:
        return held.key_bytes


def validate_extraction_runtime(
    current_executable: str | Path,
    *,
    expected_executable: str | Path = PINNED_MEDIAPIPE_PYTHON,
) -> Path:
    try:
        current = Path(current_executable).resolve(strict=True)
        expected = Path(expected_executable).resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("pinned isolated MediaPipe Python is unavailable") from exc
    if current != expected:
        raise RuntimeError(
            "Mayo extraction must run with the isolated MediaPipe runtime: "
            f"{expected_executable}"
        )
    return current


def _run_builder_impl(
    data_root: str | Path,
    existing_export_root: str | Path,
    model_path: str | Path,
    salt_file: str | Path,
    output_root: str | Path,
    exposure_manifest: str | Path,
    *,
    extractor_factory: Callable[
        ..., MayoVideoClinical23Extractor
    ] = MayoVideoClinical23Extractor,
    capture_factory: Callable[[str], object] = cv2.VideoCapture,
    inventory_factory: Callable[..., MayoInventory] = inventory_mayo_sources,
    project_root: str | Path = PROJECT_ROOT,
    current_executable: str | Path = sys.executable,
    expected_executable: str | Path = PINNED_MEDIAPIPE_PYTHON,
    version_resolver: Callable[[str], str] = importlib.metadata.version,
    dependency_artifact_resolver: Callable[
        [str], tuple[str | Path, str | Path]
    ] = _default_dependency_artifact_resolver,
    provenance_python_executable: str | Path | None = None,
    _key_guard: _HeldCanonicalMayoKey | None = None,
) -> dict[str, object]:
    """Build and atomically promote one complete cache generation."""
    validate_extraction_runtime(
        current_executable, expected_executable=expected_executable
    )
    data = _require_directory(data_root, "Mayo data root")
    exports = _require_directory(existing_export_root, "existing MediaPipe export root")
    model = _require_regular_file(model_path, "MediaPipe model")
    if _key_guard is None:
        with _hold_canonical_mayo_key(
            salt_file, project_root=project_root,
        ) as held_key:
            return _run_builder_impl(
                data_root,
                existing_export_root,
                model_path,
                salt_file,
                output_root,
                exposure_manifest,
                extractor_factory=extractor_factory,
                capture_factory=capture_factory,
                inventory_factory=inventory_factory,
                project_root=project_root,
                current_executable=current_executable,
                expected_executable=expected_executable,
                version_resolver=version_resolver,
                dependency_artifact_resolver=dependency_artifact_resolver,
                provenance_python_executable=provenance_python_executable,
                _key_guard=held_key,
            )
    _key_guard.assert_unchanged()
    salt = _key_guard.key_bytes
    output, exposure_path = validate_output_locations(
        output_root, exposure_manifest, project_root=project_root
    )
    validate_source_output_separation(data, exports, output, exposure_path)
    inventory = inventory_factory(data, exports, enforce_frozen=True)
    if not isinstance(inventory, MayoInventory):
        raise ValueError("Mayo inventory factory returned the wrong type")
    expected_inventory_counts = dict(inventory.counts)
    _inventory_counts_sha256(expected_inventory_counts)

    source_paths: list[Path] = [asset.path for asset in inventory.video_instances]
    source_paths.extend(asset.path for asset in inventory.arkit_trajectories)
    producer_paths = {
        "builder": Path(__file__),
        "action_bundle": PROJECT_ROOT / "src" / "preprocessing" / "action_bundle.py",
        "clinical_landmarks": PROJECT_ROOT / "src" / "preprocessing" / "clinical_landmarks.py",
        "dynamic_landmark_schema": PROJECT_ROOT / "src" / "datasets" / "dynamic_landmark.py",
        "feature_registry": PROJECT_ROOT / "src" / "datasets" / "patient_multistream.py",
    }
    inventory_source_hashes = {
        asset.path: asset.source_sha256 for asset in inventory.video_instances
    }
    inventory_source_hashes.update({
        asset.path: asset.source_sha256 for asset in inventory.arkit_trajectories
    })
    provenance = snapshot_provenance(
        source_paths,
        model,
        producer_paths,
        expected_source_hashes=inventory_source_hashes,
        version_resolver=version_resolver,
        dependency_artifact_resolver=dependency_artifact_resolver,
        python_executable=(
            current_executable
            if provenance_python_executable is None
            else provenance_python_executable
        ),
    )
    collection, exposure = build_public_manifests(inventory, salt)
    expected_collection_classification_integrity_id = str(
        collection["classification_integrity_id"]
    )
    expected_classification_integrity_id = str(
        exposure["classification_integrity_id"]
    )
    collection["provenance"] = {
        "runtime_dependencies": _public_dependency_provenance(provenance),
        "dependency_aggregate_sha256": provenance.dependency_aggregate_sha256,
        "model_sha256": provenance.model_sha256,
        "source_collection_integrity_id": hmac_identifier(
            "agg", salt, "mayo-source-collection-integrity",
            provenance.source_aggregate_sha256,
        ),
        "producer_sha256": provenance.producer_sha256,
        "producer_aggregate_sha256": provenance.producer_aggregate_sha256,
    }
    validate_public_manifest(collection)

    media_by_recording_id = {
        str(row["recording_id"]): row for row in collection["mediapipe_records"]
    }
    arkit_by_recording_id = {
        str(row["recording_id"]): row for row in collection["arkit_records"]
    }
    exposure_media_by_recording_id = {
        str(row["recording_id"]): row
        for row in exposure["videos"] if row["status"] == "mediapipe_ssl"
    }
    _require_private_directory(output.parent, "Mayo output parent")
    _require_private_directory(exposure_path.parent, "Mayo exposure parent")
    _assert_no_symlink_components(output.parent)
    with output_parent_lock(output):
        recover_interrupted_generations(
            output, exposure_manifest_path=exposure_path,
            salt=salt,
            expected_inventory_counts=expected_inventory_counts,
            expected_collection_classification_integrity_id=(
                expected_collection_classification_integrity_id
            ),
            expected_classification_integrity_id=(
                expected_classification_integrity_id
            ),
        )
        staging = Path(tempfile.mkdtemp(
            prefix=f".{output.name}.staging-", dir=output.parent
        ))
        try:
            os.chmod(staging, 0o700, follow_symlinks=False)
            _require_private_directory(staging, "Mayo staging generation")
            snapshot_dir = staging / ".source_snapshots"
            _make_private_directory(snapshot_dir, "Mayo source snapshot directory")
            pinned: list[PinnedSourceSnapshot] = []
            video_decode_paths: dict[Path, Path] = {}
            for index, asset in enumerate(inventory.long_unique_videos):
                item = pin_source_file(
                    asset.path,
                    snapshot_dir,
                    f"video-{index:03d}{asset.path.suffix.lower()}",
                    expected_sha256=asset.source_sha256,
                )
                pinned.append(item)
                video_decode_paths[asset.path] = item.pinned_path
            arkit_decode_paths: dict[Path, Path] = {}
            for index, asset in enumerate(inventory.arkit_trajectories):
                item = pin_source_file(
                    asset.path, snapshot_dir, f"arkit-{index:03d}.csv",
                    expected_sha256=asset.source_sha256,
                )
                pinned.append(item)
                arkit_decode_paths[asset.path] = item.pinned_path
            pinned_model = pin_source_file(
                model, snapshot_dir, "model.task",
                expected_sha256=provenance.model_sha256,
            )
            pinned.append(pinned_model)

            media_dir = staging / "mediapipe"
            arkit_dir = staging / "arkit"
            _make_private_directory(media_dir, "Mayo MediaPipe cache directory")
            _make_private_directory(arkit_dir, "Mayo ARKit cache directory")
            for asset, sequence in extract_homogeneous_video_sequences(
                inventory.long_unique_videos,
                extractor_factory,
                model_path=pinned_model.pinned_path,
                capture_factory=capture_factory,
                source_paths=video_decode_paths,
            ):
                recording_id = hmac_identifier(
                    "rec", salt, "mayo-mediapipe-recording", asset.source_sha256
                )
                row = media_by_recording_id[recording_id]
                cache_path = media_dir / f"{row['recording_id']}.npz"
                write_mediapipe_cache(
                    cache_path,
                    sequence,
                    recording_id=str(row["recording_id"]),
                    group_id=str(row["group_id"]),
                    source_integrity_id=str(row["source_integrity_id"]),
                    source_fingerprint=str(row["source_fingerprint"]),
                )
                cache_integrity_id = hmac_identifier(
                    "cache", salt, "mayo-mediapipe-cache-integrity",
                    sha256_file(cache_path),
                )
                row["cache_integrity_id"] = cache_integrity_id
                exposure_media_by_recording_id[recording_id][
                    "cache_integrity_id"
                ] = cache_integrity_id
            for asset in inventory.arkit_trajectories:
                recording_id = hmac_identifier(
                    "rec", salt, "mayo-arkit-recording", asset.source_sha256
                )
                row = arkit_by_recording_id[recording_id]
                sequence = load_arkit_trajectory(arkit_decode_paths[asset.path])
                if len(sequence.features) != asset.row_count:
                    raise ValueError("ARKit row count changed after inventory")
                cache_path = arkit_dir / f"{row['recording_id']}.npz"
                write_arkit_cache(
                    cache_path,
                    sequence,
                    recording_id=str(row["recording_id"]),
                    group_id=str(row["group_id"]),
                    source_integrity_id=str(row["source_integrity_id"]),
                    source_fingerprint=str(row["source_fingerprint"]),
                )
                row["cache_integrity_id"] = hmac_identifier(
                    "cache", salt, "mayo-arkit-cache-integrity",
                    sha256_file(cache_path),
                )
            validate_public_manifest(collection)
            validate_public_manifest(exposure)
            _write_json_exclusive(staging / "collection_manifest.json", collection)
            _write_json_exclusive(staging / "mayo_exposure_manifest.json", exposure)
            for item in pinned:
                assert_pinned_source_unchanged(item)
            assert_provenance_unchanged(
                provenance,
                version_resolver=version_resolver,
                dependency_artifact_resolver=dependency_artifact_resolver,
                python_executable=(
                    current_executable
                    if provenance_python_executable is None
                    else provenance_python_executable
                ),
            )
            _key_guard.assert_unchanged()
            _remove_real_tree(snapshot_dir)
            _validate_staging(
                staging, len(inventory.long_unique_videos),
                len(inventory.arkit_trajectories),
                salt=salt,
                expected_inventory_counts=expected_inventory_counts,
                expected_collection_classification_integrity_id=(
                    expected_collection_classification_integrity_id
                ),
                expected_classification_integrity_id=(
                    expected_classification_integrity_id
                ),
            )
            promote_generation(
                staging,
                output,
                exposure_manifest_path=exposure_path,
                salt=salt,
                expected_inventory_counts=expected_inventory_counts,
                expected_collection_classification_integrity_id=(
                    expected_collection_classification_integrity_id
                ),
                expected_classification_integrity_id=(
                    expected_classification_integrity_id
                ),
                continuity_validator=_key_guard.assert_unchanged,
            )
            _key_guard.assert_unchanged()
        finally:
            if staging.exists():
                _remove_real_tree(staging)
    return collection


def run_builder(
    data_root: str | Path,
    existing_export_root: str | Path,
    model_path: str | Path,
    salt_file: str | Path,
    output_root: str | Path,
    exposure_manifest: str | Path,
) -> dict[str, object]:
    """Run the frozen production policy without caller-overridable safeguards."""
    return _run_builder_impl(
        data_root,
        existing_export_root,
        model_path,
        salt_file,
        output_root,
        exposure_manifest,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Extraction runtime (required for a build): "
            f"{PINNED_MEDIAPIPE_PYTHON} with mediapipe==0.10.35. "
            "--inventory-only requires only the two source-root arguments."
        ),
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--existing-export-root", required=True, type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--salt-file", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--exposure-manifest", type=Path)
    parser.add_argument("--inventory-only", action="store_true")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.inventory_only:
        inventory = inventory_mayo_sources(
            args.data_root, args.existing_export_root, enforce_frozen=True
        )
        print(json.dumps({"counts": inventory.counts}, sort_keys=True))
        return
    missing = [
        option for option, value in (
            ("--model-path", args.model_path),
            ("--salt-file", args.salt_file),
            ("--output-root", args.output_root),
            ("--exposure-manifest", args.exposure_manifest),
        ) if value is None
    ]
    if missing:
        parser.error(
            "build mode requires " + ", ".join(missing)
            + f" and must run under {PINNED_MEDIAPIPE_PYTHON}"
        )
    validate_extraction_runtime(sys.executable)
    manifest = run_builder(
        data_root=args.data_root,
        existing_export_root=args.existing_export_root,
        model_path=args.model_path,
        salt_file=args.salt_file,
        output_root=args.output_root,
        exposure_manifest=args.exposure_manifest,
    )
    print(json.dumps({"counts": manifest["counts"], "output": "deidentified_cache"},
                     sort_keys=True))


if __name__ == "__main__":
    main()
