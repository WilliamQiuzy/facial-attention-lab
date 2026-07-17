#!/usr/bin/env python3
"""Audit and transactionally adapt RAVDESS OpenFace CSVs to semantic23.

The default invocation is audit-only.  Pass ``--execute`` to create
``derived_semantic23/`` directly from member bytes in the exact frozen archive.
Invalid detector frames remain in the timeline with ``valid_mask=0``; features
are never interpolated.
"""
from __future__ import annotations

import argparse
import base64
import csv
import ctypes
import errno
import fcntl
import hashlib
import hmac
import io
import json
import os
import re
import secrets
import stat
import sys
import uuid
import zipfile
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.openface68_semantic import (  # noqa: E402
    OPENFACE68_ADAPTER_METADATA,
    OPENFACE68_REQUIRED_INDICES,
    openface68_to_semantic23,
)
from src.preprocessing.semantic_landmarks import (  # noqa: E402
    SEMANTIC23_DEFINITIONS,
    SEMANTIC23_FEATURE_NAMES,
    SEMANTIC23_SCHEMA,
)


RAVDESS_ARCHIVE_RELATIVE_PATH = Path("raw/FacialTracking_Actors_01-24.zip")
RAVDESS_ID_KEY_RELATIVE_PATH = Path(".semantic23_private_id_key")
DEFAULT_CONFIDENCE_THRESHOLD = 0.80
PRIVATE_ID_KEY_BYTES = 32
_MAX_RAVDESS_CACHE_RAW_BYTES = 16 * 1024 * 1024
_MAX_RAVDESS_AGGREGATE_REGULAR_PAYLOAD_BYTES = 128 * 1024 * 1024
_MAX_RAVDESS_NPZ_COMPRESSED_BYTES = 16 * 1024 * 1024
_MAX_RAVDESS_NPZ_EXPANDED_BYTES = 64 * 1024 * 1024
_MAX_RAVDESS_NPZ_CENTRAL_BYTES = 4096
_MAX_RAVDESS_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_NPY_HEADER_BYTES = 4096
_CANONICAL_NPY_MEMBER_HEADER_BYTES = 128


@dataclass(frozen=True)
class RavdessInventoryExpectation:
    archive_size: int
    archive_md5: str
    csv_files: int
    actors: int
    frames: int
    header_sha256: str
    unique_archive_member_names: int
    unique_source_content_sha256s: int
    duplicate_content_groups: int
    members_beyond_unique_content: int
    max_content_multiplicity: int
    cross_actor_duplicate_content_groups: int
    empty_trials: int = 0
    repeated_headers: int = 0


@dataclass(frozen=True)
class RavdessInventory:
    archive_size: int
    archive_md5: str
    csv_files: int
    actors: int
    frames: int
    header_sha256: str
    empty_trials: int
    repeated_headers: int
    unique_archive_member_names: int
    unique_source_content_sha256s: int
    duplicate_content_groups: int
    members_beyond_unique_content: int
    max_content_multiplicity: int
    cross_actor_duplicate_content_groups: int
    archive_device: int
    archive_inode: int
    archive_mtime_ns: int
    archive_ctime_ns: int
    member_sha256: dict[str, str]


FROZEN_RAVDESS_INVENTORY = RavdessInventoryExpectation(
    archive_size=417_163_019,
    archive_md5="5753bbc64a9a790f8a8d3e03cba526ee",
    csv_files=2_452,
    actors=24,
    frames=299_854,
    header_sha256="d89e2164e4c4e8d60393f88365ef0e87a10bef227dc90dc1d431117a74991b4e",
    unique_archive_member_names=2_452,
    unique_source_content_sha256s=2_451,
    duplicate_content_groups=1,
    members_beyond_unique_content=1,
    max_content_multiplicity=2,
    cross_actor_duplicate_content_groups=0,
    empty_trials=0,
    repeated_headers=0,
)


@dataclass(frozen=True)
class SemanticTrial:
    """One source trial with its original timeline and detector gap mask."""

    frame_indices: np.ndarray
    timestamps: np.ndarray
    detector_confidence: np.ndarray
    features: np.ndarray
    valid_mask: np.ndarray
    source_sha256: str


@dataclass(frozen=True)
class AuthorizedRavdessTrial:
    """One cache snapshot authorized from the exact bytes that were parsed."""

    trial_id: str
    actor_id: str
    cache_integrity_id: str
    cache_sha256: str
    cache_size_bytes: int
    features: np.ndarray
    valid_mask: np.ndarray
    timestamps: np.ndarray
    frame_indices: np.ndarray
    detector_confidence: np.ndarray


@dataclass(frozen=True)
class AuthorizedRavdessGeneration:
    """Private, in-memory authorization for one immutable source generation."""

    schema: str
    manifest_sha256: str
    generation_closure_hmac: str
    trial_count: int
    actor_count: int
    source_frames: int
    valid_frames: int
    expected_trial_count: int
    expected_actor_count: int
    trials: tuple[AuthorizedRavdessTrial, ...]
    private_key: bytes = dataclass_field(repr=False)


@dataclass(frozen=True)
class _ArchiveSnapshot:
    size: int
    md5: str
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int


def _stream_file_digest(handle, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    handle.seek(0)
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _archive_stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_size),
        int(value.st_mtime_ns), int(value.st_ctime_ns),
    )


def _snapshot_from_file(handle) -> _ArchiveSnapshot:
    before = os.fstat(handle.fileno())
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("RAVDESS archive must be a regular non-symlink file")
    digest = _stream_file_digest(
        handle, "md5"  # noqa: S324 - exact frozen provenance digest
    )
    after = os.fstat(handle.fileno())
    if _archive_stat_identity(before) != _archive_stat_identity(after):
        raise ValueError("RAVDESS archive changed while its digest was computed")
    return _ArchiveSnapshot(
        size=int(after.st_size), md5=digest,
        device=int(after.st_dev), inode=int(after.st_ino),
        mtime_ns=int(after.st_mtime_ns), ctime_ns=int(after.st_ctime_ns),
    )


def _assert_archive_expectation(
    snapshot: _ArchiveSnapshot,
    expectation: RavdessInventoryExpectation,
) -> None:
    drift: dict[str, object] = {}
    if snapshot.size != expectation.archive_size:
        drift["archive_size"] = {
            "expected": expectation.archive_size, "observed": snapshot.size,
        }
    if snapshot.md5 != expectation.archive_md5:
        drift["archive_md5"] = {
            "expected": expectation.archive_md5, "observed": snapshot.md5,
        }
    if drift:
        raise ValueError(
            "RAVDESS archive drift; generation is blocked: "
            + json.dumps(drift, sort_keys=True)
        )


def _assert_path_names_snapshot(path: Path, snapshot: _ArchiveSnapshot) -> None:
    try:
        current = os.stat(path, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ValueError("RAVDESS archive lexical path disappeared") from exc
    if (not stat.S_ISREG(current.st_mode)
            or int(current.st_dev) != snapshot.device
            or int(current.st_ino) != snapshot.inode):
        raise ValueError("RAVDESS archive lexical path changed identity")


@contextmanager
def _open_verified_archive(
    archive: Path,
    expectation: RavdessInventoryExpectation,
):
    """Hold one verified no-follow archive fd through every member read."""
    try:
        descriptor = os.open(archive, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        raise FileNotFoundError(f"RAVDESS archive is missing: {archive}") from None
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError("RAVDESS archive must not be a symlink") from exc
        raise
    with os.fdopen(descriptor, "rb") as handle:
        initial = _snapshot_from_file(handle)
        _assert_archive_expectation(initial, expectation)
        _assert_path_names_snapshot(archive, initial)
        handle.seek(0)
        try:
            yield handle, initial
        finally:
            final = _snapshot_from_file(handle)
            if final != initial:
                raise ValueError("RAVDESS archive producer changed while open")
            _assert_path_names_snapshot(archive, initial)


def _archive_csv_infos(archive: zipfile.ZipFile) -> tuple[zipfile.ZipInfo, ...]:
    files = [item for item in archive.infolist() if not item.is_dir()]
    names = [item.filename for item in files]
    if len(names) != len(set(names)):
        raise ValueError("RAVDESS archive contains duplicate member names")
    if any(Path(name).name != name or not name.endswith(".csv") for name in names):
        raise ValueError("RAVDESS archive must contain flat CSV members only")
    return tuple(sorted(files, key=lambda item: item.filename))


def _read_ravdess_member_bytes(
    archive: zipfile.ZipFile, member: zipfile.ZipInfo,
) -> bytes:
    """Read one source member without exposing archive diagnostics."""
    read_failed = False
    member_bytes: bytes | None = None
    try:
        member_bytes = archive.read(member)
    except (
        OSError,
        EOFError,
        KeyError,
        RuntimeError,
        ValueError,
        zipfile.BadZipFile,
    ):
        read_failed = True
    if read_failed or not isinstance(member_bytes, bytes):
        raise ValueError("RAVDESS archive member could not be read safely")
    return member_bytes


def _actor_token_from_name(path: Path) -> str:
    fields = path.stem.split("-")
    if len(fields) != 7 or any(len(field) != 2 or not field.isdigit()
                               for field in fields):
        raise ValueError("RAVDESS CSV filename does not match the seven-field contract")
    return fields[-1]


def _inventory_values(inventory: RavdessInventory) -> dict[str, Any]:
    return asdict(inventory)


def audit_ravdess_inventory(
    data_root: str | Path,
    *,
    expectation: RavdessInventoryExpectation = FROZEN_RAVDESS_INVENTORY,
) -> RavdessInventory:
    """Read-only exact audit of the verified archive and its CSV members.

    Every CSV must have one common header, at least one data row, and a valid
    seven-field RAVDESS name.  Any mismatch raises before a staging directory is
    created.
    """
    root = Path(data_root).expanduser().resolve()
    archive = root / RAVDESS_ARCHIVE_RELATIVE_PATH
    header_hashes: set[str] = set()
    member_sha256: dict[str, str] = {}
    repeated_headers = 0
    empty_trials = 0
    frames = 0
    actors: set[str] = set()
    actor_token_by_member_name: dict[str, str] = {}

    with _open_verified_archive(archive, expectation) as (archive_file, snapshot):
        with zipfile.ZipFile(archive_file, "r") as source_zip:
            members = _archive_csv_infos(source_zip)
            for member in members:
                path = Path(member.filename)
                actor_token = _actor_token_from_name(path)
                actors.add(actor_token)
                actor_token_by_member_name[member.filename] = actor_token
                member_bytes = _read_ravdess_member_bytes(source_zip, member)
                member_sha256[member.filename] = hashlib.sha256(member_bytes).hexdigest()
                lines = member_bytes.splitlines()
                header = lines[0] if lines else b""
                if not header:
                    raise ValueError("RAVDESS CSV has an empty header")
                header_hashes.add(hashlib.sha256(header).hexdigest())
                trial_frames = 0
                for line in lines[1:]:
                    if line == header:
                        repeated_headers += 1
                    elif line.strip():
                        trial_frames += 1
                if trial_frames == 0:
                    empty_trials += 1
                frames += trial_frames

    if len(header_hashes) != 1:
        raise ValueError(
            "RAVDESS header drift: expected exactly one common header, "
            f"found {len(header_hashes)}"
        )
    expected_actor_tokens = {f"{index:02d}" for index in range(1, expectation.actors + 1)}
    if actors != expected_actor_tokens:
        raise ValueError(
            "RAVDESS actor-token drift; generation is blocked: "
            f"expected {sorted(expected_actor_tokens)}, observed {sorted(actors)}"
        )
    members_by_content: dict[str, list[str]] = {}
    for member_name, source_content_sha256 in member_sha256.items():
        members_by_content.setdefault(source_content_sha256, []).append(member_name)
    content_multiplicities = tuple(
        len(member_names) for member_names in members_by_content.values()
    )
    inventory = RavdessInventory(
        archive_size=snapshot.size,
        archive_md5=snapshot.md5,
        csv_files=len(members),
        actors=len(actors),
        frames=frames,
        header_sha256=next(iter(header_hashes)),
        empty_trials=empty_trials,
        repeated_headers=repeated_headers,
        unique_archive_member_names=len(member_sha256),
        unique_source_content_sha256s=len(members_by_content),
        duplicate_content_groups=sum(
            multiplicity > 1 for multiplicity in content_multiplicities
        ),
        members_beyond_unique_content=(
            len(member_sha256) - len(members_by_content)
        ),
        max_content_multiplicity=max(content_multiplicities, default=0),
        cross_actor_duplicate_content_groups=sum(
            len({
                actor_token_by_member_name[member_name]
                for member_name in member_names
            }) > 1
            for member_names in members_by_content.values()
            if len(member_names) > 1
        ),
        archive_device=snapshot.device,
        archive_inode=snapshot.inode,
        archive_mtime_ns=snapshot.mtime_ns,
        archive_ctime_ns=snapshot.ctime_ns,
        member_sha256=dict(sorted(member_sha256.items())),
    )
    wanted = asdict(expectation)
    got = _inventory_values(inventory)
    drift = {key: {"expected": wanted[key], "observed": got[key]}
             for key in wanted if wanted[key] != got[key]}
    if drift:
        raise ValueError(
            "RAVDESS inventory drift; generation is blocked: "
            + json.dumps(drift, sort_keys=True)
        )
    return inventory


def _parse_openface_csv_bytes_impl(
    source_bytes: bytes,
    *,
    source_name: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> SemanticTrial:
    """Parse and hash one immutable CSV byte snapshot exactly once."""
    if not isinstance(source_bytes, bytes):
        raise ValueError("OpenFace CSV source must be immutable bytes")
    if not isinstance(source_name, str) or not source_name:
        raise ValueError("OpenFace CSV source name must be nonempty")
    if not np.isfinite(confidence_threshold) or not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be finite and in [0, 1]")
    required_columns = (
        "frame", "timestamp", "confidence",
        *tuple(f"x_{index}" for index in OPENFACE68_REQUIRED_INDICES),
        *tuple(f"y_{index}" for index in OPENFACE68_REQUIRED_INDICES),
    )

    frames: list[int] = []
    timestamps: list[float] = []
    confidences: list[float] = []
    features: list[np.ndarray] = []
    valid: list[bool] = []
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    with io.TextIOWrapper(
        io.BytesIO(source_bytes), encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError("OpenFace CSV contains duplicate column names")
        missing = [name for name in required_columns if name not in fieldnames]
        if missing:
            raise ValueError(f"OpenFace CSV is missing required columns: {missing[:8]}")

        for row_number, row in enumerate(reader, start=2):
            metadata_failed = False
            try:
                frame = int(row["frame"])
                timestamp = float(row["timestamp"])
                confidence = float(row["confidence"])
            except (TypeError, ValueError):
                metadata_failed = True
            if metadata_failed:
                raise ValueError(f"invalid frame metadata at CSV row {row_number}")
            if not np.isfinite(timestamp) or not np.isfinite(confidence):
                raise ValueError(f"non-finite frame metadata at CSV row {row_number}")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"confidence outside [0, 1] at CSV row {row_number}")
            if frames and frame <= frames[-1]:
                raise ValueError("OpenFace frame indices must be strictly increasing")
            if timestamps and timestamp < timestamps[-1]:
                raise ValueError("OpenFace timestamps must be nondecreasing")

            vector = np.zeros(len(SEMANTIC23_FEATURE_NAMES), dtype=np.float32)
            is_valid = False
            if confidence >= confidence_threshold:
                points = np.zeros((68, 2), dtype=np.float64)
                try:
                    for index in OPENFACE68_REQUIRED_INDICES:
                        points[index, 0] = float(row[f"x_{index}"])
                        points[index, 1] = float(row[f"y_{index}"])
                    vector = openface68_to_semantic23(points)
                    is_valid = True
                except (TypeError, ValueError):
                    # A confident detector row with malformed required geometry
                    # is retained as a gap; it is never converted to a neutral
                    # face and never interpolated from adjacent frames.
                    vector = np.zeros(len(SEMANTIC23_FEATURE_NAMES), dtype=np.float32)
                    is_valid = False
            frames.append(frame)
            timestamps.append(timestamp)
            confidences.append(confidence)
            features.append(vector)
            valid.append(is_valid)

    if not frames:
        raise ValueError("OpenFace CSV contains no data rows")
    return SemanticTrial(
        frame_indices=np.asarray(frames, dtype=np.int64),
        timestamps=np.asarray(timestamps, dtype=np.float64),
        detector_confidence=np.asarray(confidences, dtype=np.float32),
        features=np.stack(features).astype(np.float32, copy=False),
        valid_mask=np.asarray(valid, dtype=np.bool_),
        source_sha256=source_sha256,
    )


def parse_openface_csv_bytes(
    source_bytes: bytes,
    *,
    source_name: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> SemanticTrial:
    """Parse one immutable CSV snapshot with sanitized decoder failures."""
    decoding_failed = False
    try:
        trial = _parse_openface_csv_bytes_impl(
            source_bytes,
            source_name=source_name,
            confidence_threshold=confidence_threshold,
        )
    except (UnicodeError, csv.Error):
        decoding_failed = True
    if decoding_failed:
        raise ValueError("OpenFace CSV is not valid UTF-8 CSV")
    return trial


def _parse_ravdess_member_csv(
    source_bytes: bytes, *, source_name: str,
) -> SemanticTrial:
    """Parse one source member without exposing parser diagnostics."""
    parse_failed = False
    try:
        trial = parse_openface_csv_bytes(
            source_bytes, source_name=source_name
        )
    except (csv.Error, UnicodeError, KeyError, TypeError, ValueError):
        parse_failed = True
    if parse_failed:
        raise ValueError("RAVDESS CSV could not be parsed safely")
    return trial


def parse_openface_csv(
    source_csv: str | Path,
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> SemanticTrial:
    """Read one extracted CSV once, then parse and hash that exact byte snapshot."""
    path = Path(source_csv)
    return parse_openface_csv_bytes(
        path.read_bytes(), source_name=path.name,
        confidence_threshold=confidence_threshold,
    )


def _private_id_key(value: bytes) -> bytes:
    if not isinstance(value, bytes) or len(value) < PRIVATE_ID_KEY_BYTES:
        raise ValueError("private ID key must contain at least 32 bytes")
    return value


def _private_key_stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_mode),
        int(value.st_uid), int(value.st_gid), int(value.st_nlink),
        int(value.st_size), int(value.st_mtime_ns), int(value.st_ctime_ns),
    )


def _validate_private_key_stat(
    value: os.stat_result,
    *,
    exact_size: int | None = None,
) -> None:
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("private ID key must be a regular non-symlink file")
    if int(value.st_uid) != os.geteuid():
        raise ValueError("private ID key must be owned by the current user")
    if stat.S_IMODE(value.st_mode) != 0o600:
        raise ValueError("private ID key permissions must be exactly 0600")
    if int(value.st_nlink) != 1:
        raise ValueError("private ID key must have exactly one hard link")
    if exact_size is not None and int(value.st_size) != exact_size:
        raise ValueError(f"private ID key must contain exactly {exact_size} bytes")


def _assert_private_key_path_identity(
    path: Path,
    descriptor_stat: os.stat_result,
) -> None:
    try:
        path_stat = os.stat(path, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ValueError("private ID key path disappeared during access") from exc
    if _private_key_stat_identity(path_stat) != _private_key_stat_identity(
        descriptor_stat
    ):
        raise ValueError("private ID key path identity or stat changed during access")


def _load_private_id_key(path: Path) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError("private ID key must not be a symlink") from exc
        raise
    try:
        before = os.fstat(descriptor)
        _validate_private_key_stat(before, exact_size=PRIVATE_ID_KEY_BYTES)
        payload = bytearray()
        while len(payload) <= PRIVATE_ID_KEY_BYTES:
            chunk = os.read(descriptor, PRIVATE_ID_KEY_BYTES + 1 - len(payload))
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) != PRIVATE_ID_KEY_BYTES:
            raise ValueError("private ID key must contain exactly 32 bytes")
        after = os.fstat(descriptor)
        _validate_private_key_stat(after, exact_size=PRIVATE_ID_KEY_BYTES)
        if _private_key_stat_identity(before) != _private_key_stat_identity(after):
            raise ValueError("private ID key identity or stat changed while reading")
        _assert_private_key_path_identity(path, after)
        return bytes(payload)
    finally:
        os.close(descriptor)


def _private_key_staging_prefix(destination_name: str) -> str:
    return f".{_safe_entry_name(destination_name)}.staging-"


def _assert_no_private_key_staging(
    parent_descriptor: int,
    destination_name: str,
) -> None:
    prefix = _private_key_staging_prefix(destination_name)
    if any(name.startswith(prefix) for name in os.listdir(parent_descriptor)):
        raise RuntimeError("private ID key has unresolved staging state")


def _publish_private_key_no_replace(
    parent_descriptor: int,
    staging_name: str,
    destination_name: str,
    expected_identity: tuple[int, int],
) -> None:
    """Move one verified private-key inode to its absent canonical name."""
    staging_name = _safe_entry_name(staging_name)
    destination_name = _safe_entry_name(destination_name)
    staged = _entry_stat(parent_descriptor, staging_name)
    if (
        staged is None
        or not stat.S_ISREG(staged.st_mode)
        or (int(staged.st_dev), int(staged.st_ino)) != expected_identity
    ):
        raise ValueError("private ID key staging identity changed before publication")
    _validate_private_key_stat(staged, exact_size=PRIVATE_ID_KEY_BYTES)
    if _entry_stat(parent_descriptor, destination_name) is not None:
        raise FileExistsError("canonical private ID key already exists")

    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(staging_name)
    destination_bytes = os.fsencode(destination_name)
    if sys.platform == "darwin":
        operation = library.renameatx_np
        no_replace_flag = 0x00000004 | 0x00000010
    elif sys.platform.startswith("linux"):
        operation = library.renameat2
        no_replace_flag = 0x00000001
    else:
        raise OSError("atomic private-key publication is unsupported")
    operation.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    )
    operation.restype = ctypes.c_int
    ctypes.set_errno(0)
    if operation(
        parent_descriptor,
        source_bytes,
        parent_descriptor,
        destination_bytes,
        no_replace_flag,
    ) != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError("canonical private ID key already exists")
        raise OSError(error, os.strerror(error), destination_name)
    published = _entry_stat(parent_descriptor, destination_name)
    if (
        published is None
        or (int(published.st_dev), int(published.st_ino)) != expected_identity
    ):
        raise ValueError("published private ID key changed identity")
    _validate_private_key_stat(published, exact_size=PRIVATE_ID_KEY_BYTES)
    os.fsync(parent_descriptor)


def load_or_create_private_id_key(path: str | Path) -> bytes:
    """Load or atomically create one owner-only stable HMAC key."""
    destination = _absolute_lexical_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent_descriptor, parent_identity = _open_output_parent(destination.parent)
    locked = False
    descriptor: int | None = None
    staging_name = (
        _private_key_staging_prefix(destination.name) + uuid.uuid4().hex
    )
    created_identity: tuple[int, int] | None = None
    try:
        fcntl.flock(parent_descriptor, fcntl.LOCK_EX)
        locked = True
        _assert_output_parent_identity(
            destination.parent, parent_descriptor, parent_identity,
        )
        _assert_no_private_key_staging(parent_descriptor, destination.name)
        if _entry_stat(parent_descriptor, destination.name) is not None:
            result = _load_private_id_key(destination)
            _assert_output_parent_identity(
                destination.parent, parent_descriptor, parent_identity,
            )
            return result

        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        try:
            descriptor = os.open(
                staging_name, flags, 0o600, dir_fd=parent_descriptor,
            )
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ValueError("private ID key staging must not be a symlink") from exc
            raise
        os.fchmod(descriptor, 0o600)
        initial = os.fstat(descriptor)
        created_identity = (int(initial.st_dev), int(initial.st_ino))
        _validate_private_key_stat(initial)
        payload = secrets.token_bytes(PRIVATE_ID_KEY_BYTES)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        written = os.fstat(descriptor)
        _validate_private_key_stat(written, exact_size=PRIVATE_ID_KEY_BYTES)
        if (int(written.st_dev), int(written.st_ino)) != created_identity:
            raise ValueError("private ID key identity changed during creation")
        staged = os.stat(
            staging_name, dir_fd=parent_descriptor, follow_symlinks=False,
        )
        if _private_key_stat_identity(staged) != _private_key_stat_identity(written):
            raise ValueError("private ID key staging path changed during creation")
        os.lseek(descriptor, 0, os.SEEK_SET)
        observed = bytearray()
        while len(observed) <= PRIVATE_ID_KEY_BYTES:
            chunk = os.read(
                descriptor, PRIVATE_ID_KEY_BYTES + 1 - len(observed),
            )
            if not chunk:
                break
            observed.extend(chunk)
        verified = os.fstat(descriptor)
        if (
            len(observed) != PRIVATE_ID_KEY_BYTES
            or not hmac.compare_digest(bytes(observed), payload)
            or _private_key_stat_identity(verified)
            != _private_key_stat_identity(written)
        ):
            raise ValueError("private ID key staging bytes changed before publication")
        staged = os.stat(
            staging_name, dir_fd=parent_descriptor, follow_symlinks=False,
        )
        if _private_key_stat_identity(staged) != _private_key_stat_identity(verified):
            raise ValueError("private ID key staging identity changed before publication")
        _assert_output_parent_identity(
            destination.parent, parent_descriptor, parent_identity,
        )
        _publish_private_key_no_replace(
            parent_descriptor,
            staging_name,
            destination.name,
            created_identity,
        )

        final = os.fstat(descriptor)
        canonical = os.stat(
            destination.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        _validate_private_key_stat(final, exact_size=PRIVATE_ID_KEY_BYTES)
        if (
            _private_key_stat_identity(final)
            != _private_key_stat_identity(canonical)
            or (int(final.st_dev), int(final.st_ino)) != created_identity
        ):
            raise ValueError("canonical private ID key changed after publication")
        os.lseek(descriptor, 0, os.SEEK_SET)
        final_payload = bytearray()
        while len(final_payload) <= PRIVATE_ID_KEY_BYTES:
            chunk = os.read(
                descriptor, PRIVATE_ID_KEY_BYTES + 1 - len(final_payload),
            )
            if not chunk:
                break
            final_payload.extend(chunk)
        if (
            len(final_payload) != PRIVATE_ID_KEY_BYTES
            or not hmac.compare_digest(bytes(final_payload), payload)
        ):
            raise ValueError("canonical private ID key bytes changed after publication")
        _assert_output_parent_identity(
            destination.parent, parent_descriptor, parent_identity,
        )
        return payload
    finally:
        try:
            if descriptor is not None:
                os.close(descriptor)
        finally:
            try:
                if locked:
                    fcntl.flock(parent_descriptor, fcntl.LOCK_UN)
            finally:
                os.close(parent_descriptor)


def _opaque_id(namespace: str, source_value: str, prefix: str, *, key: bytes) -> str:
    if not isinstance(source_value, str) or not source_value:
        raise ValueError("opaque provenance input must be a non-empty string")
    digest = hmac.new(
        _private_id_key(key),
        f"ravdess-semantic23-v1:{namespace}:{source_value}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    token = base64.b32encode(digest).decode("ascii").lower().rstrip("=")[:16]
    return f"{prefix}_{token}"


def opaque_actor_id(actor_token: str, *, key: bytes) -> str:
    """Stable private-key pseudonym linking trials from one RAVDESS actor."""
    return _opaque_id("actor", actor_token, "actor", key=key)


def opaque_trial_id(
    archive_member_name: str,
    source_content_sha256: str,
    *,
    key: bytes,
) -> str:
    """Stable v2 pseudonym binding one exact member name and byte digest."""
    if (
        type(archive_member_name) is not str
        or re.fullmatch(
            r"[0-9]{2}(?:-[0-9]{2}){6}\.csv",
            archive_member_name,
            flags=re.ASCII,
        ) is None
    ):
        raise ValueError("RAVDESS archive member name is noncanonical")
    if (
        type(source_content_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", source_content_sha256, flags=re.ASCII)
        is None
    ):
        raise ValueError("RAVDESS source-content digest is noncanonical")
    if type(key) is not bytes or len(key) != PRIVATE_ID_KEY_BYTES:
        raise ValueError("RAVDESS trial-ID key must be exactly 32 bytes")
    binding = json.dumps(
        {
            "archive_member_name": archive_member_name,
            "source_content_sha256": source_content_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    digest = hmac.new(
        key,
        b"ravdess-semantic23-trial-id-v2\0" + binding,
        hashlib.sha256,
    ).digest()
    token = base64.b32encode(digest).decode("ascii").lower().rstrip("=")[:16]
    return f"trial_{token}"


def _opaque_cache_integrity_id(
    cache_sha256: str,
    *,
    trial_id: str,
    actor_id: str,
    key: bytes,
) -> str:
    """Bind exact cache bytes to their keyed trial and actor identities."""
    binding = json.dumps(
        {
            "actor_id": actor_id,
            "cache_sha256": cache_sha256,
            "trial_id": trial_id,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return _opaque_id("cache-integrity-v2", binding, "cache", key=key)


def _manifest_inventory(inventory: RavdessInventory) -> dict[str, Any]:
    return {
        "archive_size_bytes": inventory.archive_size,
        "archive_md5": inventory.archive_md5,
        "csv_trials": inventory.csv_files,
        "actors": inventory.actors,
        "source_frames": inventory.frames,
        "header_sha256": inventory.header_sha256,
        "empty_trials": inventory.empty_trials,
        "repeated_headers": inventory.repeated_headers,
        "unique_archive_member_names": inventory.unique_archive_member_names,
        "unique_source_content_sha256s": inventory.unique_source_content_sha256s,
        "duplicate_content_groups": inventory.duplicate_content_groups,
        "members_beyond_unique_content": inventory.members_beyond_unique_content,
        "max_content_multiplicity": inventory.max_content_multiplicity,
        "cross_actor_duplicate_content_groups": (
            inventory.cross_actor_duplicate_content_groups
        ),
    }


def _private_provenance_representations(
    *,
    source_paths: Sequence[Path],
    raw_source_sha256s: set[str] | frozenset[str],
    raw_cache_sha256s: set[str] | frozenset[str] = frozenset(),
) -> frozenset[bytes]:
    representations: set[bytes] = set()
    for path in source_paths:
        name = path.name.encode("utf-8")
        if not name:
            raise ValueError("source filename privacy input is empty")
        representations.update({
            name,
            name.hex().encode("ascii"),
            name.hex().upper().encode("ascii"),
            base64.b64encode(name),
        })
    for digest in raw_source_sha256s | set(raw_cache_sha256s):
        if (
            type(digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", digest, flags=re.ASCII) is None
        ):
            raise ValueError("source digest privacy input is noncanonical")
        digest_text = digest.encode("ascii")
        digest_bytes = bytes.fromhex(digest)
        representations.update({
            digest_text,
            digest_text.upper(),
            digest_text.hex().encode("ascii"),
            digest_text.hex().upper().encode("ascii"),
            base64.b64encode(digest_text),
            digest_bytes,
            digest_bytes.hex().encode("ascii"),
            digest_bytes.hex().upper().encode("ascii"),
            base64.b64encode(digest_bytes),
        })
    for representation in tuple(representations):
        try:
            text = representation.decode("ascii")
        except UnicodeDecodeError:
            continue
        representations.update({
            text.encode("utf-16-le"),
            text.encode("utf-16-be"),
            text.encode("utf-32-le"),
            text.encode("utf-32-be"),
        })
    return frozenset(representations)


_PRIVATE_PATTERN_HASH_BASE = 257
_PRIVATE_PATTERN_HASH_MASK = (1 << 64) - 1
_PRIVATE_PATTERN_SCAN_CHUNK_BYTES = 1024 * 1024
_PrivatePatternIndex = tuple[
    tuple[int, frozenset[bytes], np.ndarray], ...
]


def _private_pattern_hash(pattern: bytes) -> int:
    value = 0
    factor = 1
    for byte in pattern:
        value = (
            value + int(byte) * factor
        ) & _PRIVATE_PATTERN_HASH_MASK
        factor = (
            factor * _PRIVATE_PATTERN_HASH_BASE
        ) & _PRIVATE_PATTERN_HASH_MASK
    return value


def _compile_private_pattern_index(
    representations: set[bytes] | frozenset[bytes],
) -> _PrivatePatternIndex:
    grouped: dict[int, set[bytes]] = {}
    for representation in representations:
        if representation:
            grouped.setdefault(len(representation), set()).add(representation)
    compiled: list[tuple[int, frozenset[bytes], np.ndarray]] = []
    for length, patterns in sorted(grouped.items()):
        hashes = np.asarray(
            sorted({_private_pattern_hash(pattern) for pattern in patterns}),
            dtype=np.uint64,
        )
        hashes.setflags(write=False)
        compiled.append((length, frozenset(patterns), hashes))
    return tuple(compiled)


def _contains_indexed_private_pattern(
    blobs: Sequence[bytes],
    *indexes: _PrivatePatternIndex,
) -> bool:
    """Exact chunked multi-pattern scan with bounded vectorized workspaces."""
    grouped: dict[int, list[tuple[frozenset[bytes], np.ndarray]]] = {}
    for index in indexes:
        for length, patterns, hashes in index:
            grouped.setdefault(length, []).append((patterns, hashes))
    if not grouped:
        return False
    max_pattern_length = max(grouped)
    inverse_base = pow(_PRIVATE_PATTERN_HASH_BASE, -1, 1 << 64)

    for blob in blobs:
        if not blob:
            continue
        for chunk_start in range(
            0, len(blob), _PRIVATE_PATTERN_SCAN_CHUNK_BYTES
        ):
            assigned_starts = min(
                _PRIVATE_PATTERN_SCAN_CHUNK_BYTES,
                len(blob) - chunk_start,
            )
            chunk_end = min(
                len(blob),
                chunk_start + assigned_starts + max_pattern_length - 1,
            )
            chunk = blob[chunk_start:chunk_end]
            size = len(chunk)
            byte_values = np.frombuffer(chunk, dtype=np.uint8).astype(
                np.uint64, copy=False
            )
            powers = np.empty(size, dtype=np.uint64)
            inverse_powers = np.empty(size, dtype=np.uint64)
            powers[0] = np.uint64(1)
            inverse_powers[0] = np.uint64(1)
            if size > 1:
                powers[1:] = np.uint64(_PRIVATE_PATTERN_HASH_BASE)
                np.multiply.accumulate(powers[1:], out=powers[1:])
                inverse_powers[1:] = np.uint64(inverse_base)
                np.multiply.accumulate(
                    inverse_powers[1:], out=inverse_powers[1:]
                )
            weighted = byte_values * powers
            prefix = np.empty(size + 1, dtype=np.uint64)
            prefix[0] = np.uint64(0)
            np.cumsum(weighted, dtype=np.uint64, out=prefix[1:])

            for length, pattern_groups in grouped.items():
                if length > size:
                    continue
                raw_hashes = prefix[length:] - prefix[:-length]
                window_hashes = (
                    raw_hashes * inverse_powers[:size - length + 1]
                )
                for patterns, pattern_hashes in pattern_groups:
                    indices = np.searchsorted(pattern_hashes, window_hashes)
                    possible = np.flatnonzero(
                        indices < len(pattern_hashes)
                    )
                    possible = possible[possible < assigned_starts]
                    if possible.size == 0:
                        continue
                    matching = possible[
                        pattern_hashes[indices[possible]]
                        == window_hashes[possible]
                    ]
                    for position_value in matching:
                        position = int(position_value)
                        if chunk[position:position + length] in patterns:
                            return True
    return False


def _assert_manifest_deidentified(
    manifest_text: str,
    *,
    source_root: Path,
    source_paths: list[Path],
    raw_source_sha256s: set[str],
    raw_cache_sha256s: set[str] | frozenset[str] = frozenset(),
) -> None:
    manifest_bytes = manifest_text.encode("utf-8")
    if str(source_root) in manifest_text:
        raise ValueError("aggregate manifest contains the raw source root")
    representations = _private_provenance_representations(
        source_paths=source_paths,
        raw_source_sha256s=raw_source_sha256s,
        raw_cache_sha256s=raw_cache_sha256s,
    )
    if _contains_indexed_private_pattern(
        (manifest_bytes,), _compile_private_pattern_index(representations)
    ):
        raise ValueError(
            "aggregate manifest contains raw or reversibly encoded provenance"
        )


_AUTHORIZED_MANIFEST_FIELDS = frozenset({
    "format_version", "schema", "feature_names", "feature_definitions",
    "source_dataset", "license", "permitted_role", "clinical_claim",
    "adapter", "filter", "timeline_policy", "provenance_policy",
    "inventory", "quality_control", "trials",
})
_AUTHORIZED_TRIAL_FIELDS = frozenset({
    "trial_id", "actor_id", "cache_integrity_id",
})
_AUTHORIZED_CACHE_FIELDS = (
    "features", "valid_mask", "timestamps", "frame_indices",
    "detector_confidence", "feature_names", "schema", "adapter_name",
    "scale_normalization", "confidence_threshold",
)
_TRIAL_ID = re.compile(r"^trial_[a-z2-7]{16}$")
_ACTOR_ID = re.compile(r"^actor_[a-z2-7]{16}$")
_CACHE_ID = re.compile(r"^cache_[a-z2-7]{16}$")


def _owner_regular_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_mode),
        int(value.st_uid), int(value.st_gid), int(value.st_nlink),
        int(value.st_size), int(value.st_mtime_ns), int(value.st_ctime_ns),
    )


def _owner_directory_stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_mode),
        int(value.st_uid), int(value.st_gid), int(value.st_nlink),
        int(value.st_size), int(value.st_mtime_ns), int(value.st_ctime_ns),
    )


def _snapshot_exact_owner_directory(
    descriptor: int,
    expected_names: Sequence[str],
    field: str,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    expected = tuple(expected_names)
    if (
        not expected
        or len(set(expected)) != len(expected)
        or any(type(name) is not str or not name or Path(name).name != name
               for name in expected)
    ):
        raise ValueError(f"{field} expected entry set is malformed")
    before = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(before.st_mode)
        or int(before.st_uid) != os.geteuid()
        or stat.S_IMODE(before.st_mode) != 0o700
    ):
        raise ValueError(f"{field} is not an owner-only directory")
    names = tuple(sorted(os.listdir(descriptor)))
    after = os.fstat(descriptor)
    identity = _owner_directory_stat_identity(before)
    if _owner_directory_stat_identity(after) != identity:
        raise ValueError(f"{field} changed while its entries were listed")
    if len(names) != len(expected) or set(names) != set(expected):
        raise ValueError(f"{field} entry set is not exact")
    return identity, names


def _read_owner_only_regular(
    path: Path,
    field: str,
    *,
    max_bytes: int | None = None,
    parent_descriptor: int | None = None,
) -> tuple[bytes, str, tuple[int, ...]]:
    """Read and hash one owner-only file from one no-follow descriptor."""
    if max_bytes is not None and (
        not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1
    ):
        raise ValueError(f"{field} byte limit must be a positive integer")
    target: str | Path = path
    open_kwargs: dict[str, int] = {}
    if parent_descriptor is not None:
        if str(path) != path.name or len(path.parts) != 1:
            raise ValueError(f"{field} anchored filename is unsafe")
        target = _safe_entry_name(path.name)
        open_kwargs["dir_fd"] = parent_descriptor
    try:
        descriptor = os.open(
            target, os.O_RDONLY | os.O_NOFOLLOW, **open_kwargs
        )
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(f"{field} must not be a symlink") from exc
        raise
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or int(before.st_uid) != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o600
            or int(before.st_nlink) != 1
        ):
            raise ValueError(f"{field} must be a singly-linked owner-only regular file")
        if max_bytes is not None and int(before.st_size) > max_bytes:
            raise ValueError(f"{field} exceeds its raw byte limit")
        digest = hashlib.sha256()
        payload = bytearray()
        while chunk := os.read(descriptor, 1024 * 1024):
            payload.extend(chunk)
            if max_bytes is not None and len(payload) > max_bytes:
                raise ValueError(f"{field} exceeds its raw byte limit")
            digest.update(chunk)
        after = os.fstat(descriptor)
        if _owner_regular_identity(before) != _owner_regular_identity(after):
            raise ValueError(f"{field} changed while it was read")
        try:
            current = os.stat(
                target, follow_symlinks=False, **open_kwargs
            )
        except FileNotFoundError as exc:
            raise ValueError(f"{field} disappeared while it was read") from exc
        if _owner_regular_identity(after) != _owner_regular_identity(current):
            raise ValueError(f"{field} path identity or stat changed while it was read")
        return bytes(payload), digest.hexdigest(), _owner_regular_identity(after)
    finally:
        os.close(descriptor)


def _assert_owner_snapshot(path: Path, identity: tuple[int, ...], field: str) -> None:
    try:
        current = os.stat(path, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ValueError(f"{field} disappeared before authorization returned") from exc
    if _owner_regular_identity(current) != identity:
        raise ValueError(f"{field} changed before authorization returned")


def _assert_owner_snapshot_at(
    parent_descriptor: int,
    name: str,
    identity: tuple[int, ...],
    field: str,
) -> None:
    try:
        current = os.stat(
            _safe_entry_name(name),
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError as exc:
        raise ValueError(f"{field} disappeared before authorization returned") from exc
    if _owner_regular_identity(current) != identity:
        raise ValueError(f"{field} changed before authorization returned")


def _revalidate_ctime_only_owner_snapshot_at(
    parent_descriptor: int,
    name: str,
    identity: tuple[int, ...],
    expected_payload: bytes,
    field: str,
    *,
    max_bytes: int,
) -> tuple[int, ...]:
    """Rebind one newly written file after a content-neutral ctime update.

    macOS may asynchronously attach ``com.apple.provenance`` to a new file,
    changing only ctime after the first held read.  Extended attributes are not
    part of the RAVDESS artifact contract, but every contracted stat field and
    the complete file bytes remain fail-closed here before the refreshed
    identity can be used.
    """
    name = _safe_entry_name(name)
    try:
        current = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError as exc:
        raise ValueError(
            f"{field} disappeared before authorization returned"
        ) from exc
    observed_identity = _owner_regular_identity(current)
    if observed_identity == identity:
        return identity
    if observed_identity[:-1] != identity[:-1]:
        raise ValueError(f"{field} changed before authorization returned")
    payload, _, refreshed_identity = _read_owner_only_regular(
        Path(name),
        field,
        max_bytes=max_bytes,
        parent_descriptor=parent_descriptor,
    )
    if (
        refreshed_identity[:-1] != identity[:-1]
        or not hmac.compare_digest(payload, expected_payload)
    ):
        raise ValueError(f"{field} changed before authorization returned")
    return refreshed_identity


def _load_unique_json_object(payload: bytes, field: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{field} repeats a JSON field")
            result[key] = value
        return result

    decoding_failed = False
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError):
        decoding_failed = True
    if decoding_failed:
        raise ValueError(f"{field} is not valid UTF-8 JSON")
    if not isinstance(value, dict):
        raise ValueError(f"{field} must contain an object")
    return value


def _json_exact_equal(observed: object, expected: object) -> bool:
    if type(observed) is not type(expected):
        return False
    if type(expected) is dict:
        observed_mapping = observed
        expected_mapping = expected
        return (
            set(observed_mapping) == set(expected_mapping)
            and all(
                _json_exact_equal(observed_mapping[key], expected_mapping[key])
                for key in expected_mapping
            )
        )
    if type(expected) is list:
        observed_items = observed
        expected_items = expected
        return (
            len(observed_items) == len(expected_items)
            and all(
                _json_exact_equal(left, right)
                for left, right in zip(observed_items, expected_items)
            )
        )
    return bool(observed == expected)


def _readonly_array(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy()
    result.flags.writeable = False
    return result


def _npy_header(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    field: str,
) -> tuple[np.dtype, tuple[int, ...], bool]:
    header_failure: str | None = None
    try:
        with archive.open(info, "r") as member:
            version = np.lib.format.read_magic(member)
            if version not in {(1, 0), (2, 0), (3, 0)}:
                header_failure = f"{field} has an unsupported NPY version"
            else:
                shape, fortran_order, dtype = np.lib.format._read_array_header(
                    member, version, max_header_size=_MAX_NPY_HEADER_BYTES
                )
                header_bytes = member.tell()
    except (OSError, EOFError, UnicodeError, ValueError, zipfile.BadZipFile):
        header_failure = f"{field} has an invalid bounded NPY header"
    if header_failure is not None:
        raise ValueError(header_failure)
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
    expected_size = header_bytes + element_count * canonical_dtype.itemsize
    if expected_size != int(info.file_size):
        raise ValueError(f"{field} NPY header does not match its declared member size")
    return canonical_dtype, shape, fortran_order


def _require_exact_zip_eocd(payload: bytes, *, member_count: int, field: str) -> None:
    """Validate a small, single-disk central directory before ``zipfile``."""
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
        or central_size <= 0
        or central_size > _MAX_RAVDESS_NPZ_CENTRAL_BYTES
        or central_offset <= 0
        or central_offset + central_size != offset
    ):
        raise ValueError(f"{field} ZIP end record is noncanonical")

    central = memoryview(payload)[central_offset:offset]
    cursor = 0
    actual_members = 0
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
            or record_size > remaining
        ):
            raise ValueError(f"{field} central directory metadata is noncanonical")
        cursor += record_size

    if cursor != central_size or actual_members != member_count:
        raise ValueError(f"{field} central directory member count is not exact")


def _require_ravdess_npz_headers(payload: bytes) -> tuple[int, int]:
    """Bound the ZIP and validate every NPY header before array materialization."""
    if len(payload) > _MAX_RAVDESS_CACHE_RAW_BYTES:
        raise ValueError("RAVDESS cache exceeds its raw byte limit")
    expected_names = tuple(f"{name}.npy" for name in _AUTHORIZED_CACHE_FIELDS)
    _require_exact_zip_eocd(
        payload, member_count=len(expected_names), field="RAVDESS cache"
    )
    validation_failure: str | None = None
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            infos = tuple(archive.infolist())
            names = tuple(info.filename for info in infos)
            if (
                len(infos) != len(expected_names)
                or len(names) != len(set(names))
                or names != expected_names
                or any(info.is_dir() for info in infos)
            ):
                raise ValueError("RAVDESS cache ZIP member schema is not exact")
            if any(
                info.flag_bits & 0x1
                or info.compress_type != zipfile.ZIP_DEFLATED
                or info.compress_size < 0
                or info.file_size < 0
                for info in infos
            ):
                raise ValueError("RAVDESS cache ZIP metadata is noncanonical")
            if sum(int(info.compress_size) for info in infos) > (
                _MAX_RAVDESS_NPZ_COMPRESSED_BYTES
            ):
                raise ValueError("RAVDESS cache exceeds its compressed byte limit")
            expanded_bytes = sum(int(info.file_size) for info in infos)
            if expanded_bytes > _MAX_RAVDESS_NPZ_EXPANDED_BYTES:
                raise ValueError("RAVDESS cache exceeds its expanded byte limit")
            headers = {
                info.filename[:-4]: _npy_header(
                    archive, info, field=f"RAVDESS cache {info.filename[:-4]}"
                )
                for info in infos
            }
    except (OSError, EOFError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("RAVDESS cache"):
            validation_failure = str(exc)
        else:
            validation_failure = "RAVDESS cache is not a bounded exact NPZ"
    if validation_failure is not None:
        raise ValueError(validation_failure)

    features_dtype, features_shape, _ = headers["features"]
    if (
        features_dtype != np.dtype(np.float32)
        or len(features_shape) != 2
        or features_shape[1] != len(SEMANTIC23_FEATURE_NAMES)
        or not 1 <= features_shape[0] <= FROZEN_RAVDESS_INVENTORY.frames
    ):
        raise ValueError("RAVDESS cache features NPY header is noncanonical")
    length = features_shape[0]
    expected = {
        "features": (np.dtype(np.float32), (length, 23)),
        "valid_mask": (np.dtype(np.bool_), (length,)),
        "timestamps": (np.dtype(np.float64), (length,)),
        "frame_indices": (np.dtype(np.int64), (length,)),
        "detector_confidence": (np.dtype(np.float32), (length,)),
        "feature_names": (
            np.asarray(SEMANTIC23_FEATURE_NAMES).dtype,
            (len(SEMANTIC23_FEATURE_NAMES),),
        ),
        "schema": (np.asarray(SEMANTIC23_SCHEMA).dtype, ()),
        "adapter_name": (
            np.asarray(OPENFACE68_ADAPTER_METADATA["adapter_name"]).dtype, (),
        ),
        "scale_normalization": (
            np.asarray(OPENFACE68_ADAPTER_METADATA["scale_normalization"]).dtype, (),
        ),
        "confidence_threshold": (np.dtype(np.float32), ()),
    }
    for name, (expected_dtype, expected_shape) in expected.items():
        dtype, shape, fortran_order = headers[name]
        if dtype != expected_dtype or shape != expected_shape or fortran_order:
            raise ValueError(f"RAVDESS cache {name} NPY header is noncanonical")
    return length, expanded_bytes


def _ravdess_aggregate_expanded_budget(
    expectation: RavdessInventoryExpectation,
) -> int:
    per_frame_bytes = (
        23 * np.dtype(np.float32).itemsize
        + np.dtype(np.bool_).itemsize
        + np.dtype(np.float64).itemsize
        + np.dtype(np.int64).itemsize
        + np.dtype(np.float32).itemsize
    )
    fixed_payload_bytes = sum(
        int(value.nbytes)
        for value in (
            np.asarray(SEMANTIC23_FEATURE_NAMES),
            np.asarray(SEMANTIC23_SCHEMA),
            np.asarray(OPENFACE68_ADAPTER_METADATA["adapter_name"]),
            np.asarray(OPENFACE68_ADAPTER_METADATA["scale_normalization"]),
            np.asarray(DEFAULT_CONFIDENCE_THRESHOLD, dtype=np.float32),
        )
    )
    max_header_bytes = (
        len(_AUTHORIZED_CACHE_FIELDS) * _CANONICAL_NPY_MEMBER_HEADER_BYTES
    )
    return (
        expectation.frames * per_frame_bytes
        + expectation.csv_files * (fixed_payload_bytes + max_header_bytes)
    )


def _assert_ravdess_cache_deidentified(
    payload: bytes,
    *,
    cache_name: str,
    source_pattern_index: _PrivatePatternIndex,
    raw_cache_sha256: str,
) -> tuple[int, int]:
    """Scan one exact cache and its expanded NPZ members for private inputs."""
    cache_pattern_index = _compile_private_pattern_index(
        _private_provenance_representations(
            source_paths=(),
            raw_source_sha256s=set(),
            raw_cache_sha256s={raw_cache_sha256},
        )
    )
    if _contains_indexed_private_pattern(
        (cache_name.encode("utf-8"), payload),
        source_pattern_index,
        cache_pattern_index,
    ):
        raise ValueError("RAVDESS cache contains private provenance")
    declared_frames, expanded_bytes = _require_ravdess_npz_headers(payload)
    blobs: list[bytes] = []
    privacy_scan_failed = False
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            blobs.append(archive.comment)
            for info in archive.infolist():
                blobs.extend((
                    info.filename.encode("utf-8"),
                    info.comment,
                    info.extra,
                    archive.read(info),
                ))
    except (OSError, EOFError, ValueError, zipfile.BadZipFile):
        privacy_scan_failed = True
    if privacy_scan_failed:
        raise ValueError("RAVDESS cache privacy scan failed")
    if _contains_indexed_private_pattern(
        blobs, source_pattern_index, cache_pattern_index
    ):
        raise ValueError("RAVDESS cache contains private provenance")
    return declared_frames, expanded_bytes


def _source_private_pattern_index(
    member_sha256: dict[str, str],
) -> _PrivatePatternIndex:
    return _compile_private_pattern_index(
        _private_provenance_representations(
            source_paths=tuple(Path(name) for name in member_sha256),
            raw_source_sha256s=set(member_sha256.values()),
        )
    )


def _scan_staged_ravdess_caches_for_private_provenance(
    parent_descriptor: int,
    cache_sha256_by_name: dict[str, str],
    source_pattern_index: _PrivatePatternIndex,
    *,
    expectation: RavdessInventoryExpectation,
) -> tuple[tuple[int, ...], dict[str, tuple[int, ...]], int, int, int]:
    """Scan exact staged cache bytes and expanded NPZ members before publish."""
    if not source_pattern_index:
        raise ValueError("staged RAVDESS source privacy index is empty")
    expected_names = tuple(sorted(cache_sha256_by_name))
    tree_identity, cache_names = _snapshot_exact_owner_directory(
        parent_descriptor,
        expected_names,
        "staged RAVDESS trial cache",
    )
    identities: dict[str, tuple[int, ...]] = {}
    aggregate_regular_payload_bytes = 0
    aggregate_declared_frames = 0
    aggregate_expanded_bytes = 0
    aggregate_expanded_budget = _ravdess_aggregate_expanded_budget(expectation)
    for cache_name in cache_names:
        payload, observed_sha256, identity = _read_owner_only_regular(
            Path(cache_name),
            "staged RAVDESS semantic23 cache",
            max_bytes=_MAX_RAVDESS_CACHE_RAW_BYTES,
            parent_descriptor=parent_descriptor,
        )
        if not hmac.compare_digest(
            observed_sha256, cache_sha256_by_name[cache_name]
        ):
            raise ValueError("staged RAVDESS cache integrity changed")
        declared_frames, expanded_bytes = _assert_ravdess_cache_deidentified(
            payload,
            cache_name=cache_name,
            source_pattern_index=source_pattern_index,
            raw_cache_sha256=observed_sha256,
        )
        aggregate_regular_payload_bytes += len(payload)
        aggregate_declared_frames += declared_frames
        aggregate_expanded_bytes += expanded_bytes
        if (
            aggregate_regular_payload_bytes
            > _MAX_RAVDESS_AGGREGATE_REGULAR_PAYLOAD_BYTES
            or aggregate_declared_frames > expectation.frames
            or aggregate_expanded_bytes > aggregate_expanded_budget
        ):
            raise ValueError(
                "staged RAVDESS cumulative cache resource budget is exceeded"
            )
        identities[cache_name] = identity
    if aggregate_declared_frames != expectation.frames:
        raise ValueError("staged RAVDESS declared frame count is not exact")
    return (
        tree_identity,
        identities,
        aggregate_regular_payload_bytes,
        aggregate_declared_frames,
        aggregate_expanded_bytes,
    )


def _validate_authorized_ravdess_cache(
    payload: bytes,
    *,
    trial_id: str,
    actor_id: str,
    cache_integrity_id: str,
    cache_sha256: str,
) -> AuthorizedRavdessTrial:
    _require_ravdess_npz_headers(payload)
    validation_failure: str | None = None
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as cached:
            if tuple(cached.files) != _AUTHORIZED_CACHE_FIELDS:
                raise ValueError("RAVDESS cache field schema is not exact")
            features = np.asarray(cached["features"])
            valid = np.asarray(cached["valid_mask"])
            timestamps = np.asarray(cached["timestamps"])
            frame_indices = np.asarray(cached["frame_indices"])
            confidence = np.asarray(cached["detector_confidence"])
            length = features.shape[0] if features.ndim == 2 else -1
            if features.dtype != np.float32 or features.shape != (length, 23) or length < 1:
                raise ValueError("RAVDESS cache feature array is noncanonical")
            if valid.dtype != np.bool_ or valid.shape != (length,):
                raise ValueError("RAVDESS cache validity mask is noncanonical")
            if timestamps.dtype != np.float64 or timestamps.shape != (length,):
                raise ValueError("RAVDESS cache timestamps are noncanonical")
            if frame_indices.dtype != np.int64 or frame_indices.shape != (length,):
                raise ValueError("RAVDESS cache frame indices are noncanonical")
            if confidence.dtype != np.float32 or confidence.shape != (length,):
                raise ValueError("RAVDESS detector confidence is noncanonical")
            if (
                not np.isfinite(features).all()
                or not np.isfinite(timestamps).all()
                or not np.isfinite(confidence).all()
                or np.any(confidence < 0.0)
                or np.any(confidence > 1.0)
                or np.any(frame_indices < 0)
                or (length > 1 and not np.all(np.diff(frame_indices) == 1))
                or (length > 1 and not np.all(np.diff(timestamps) > 0))
                or np.any(features[~valid] != np.float32(0.0))
                or np.any(confidence[valid] < np.float32(DEFAULT_CONFIDENCE_THRESHOLD))
            ):
                raise ValueError("RAVDESS cache timeline, mask, or values are invalid")
            if tuple(str(item) for item in np.asarray(cached["feature_names"]).tolist()) != SEMANTIC23_FEATURE_NAMES:
                raise ValueError("RAVDESS cache semantic23 names are noncanonical")
            exact_scalars = {
                "schema": SEMANTIC23_SCHEMA,
                "adapter_name": OPENFACE68_ADAPTER_METADATA["adapter_name"],
                "scale_normalization": OPENFACE68_ADAPTER_METADATA["scale_normalization"],
            }
            for name, expected in exact_scalars.items():
                if str(np.asarray(cached[name]).item()) != expected:
                    raise ValueError(f"RAVDESS cache {name} is noncanonical")
            threshold = np.asarray(cached["confidence_threshold"])
            if threshold.dtype != np.float32 or threshold.shape != () or threshold.item() != np.float32(DEFAULT_CONFIDENCE_THRESHOLD):
                raise ValueError("RAVDESS cache confidence threshold is noncanonical")
    except (OSError, EOFError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("RAVDESS cache"):
            validation_failure = str(exc)
        else:
            validation_failure = "RAVDESS cache is not a safe exact NPZ"
    if validation_failure is not None:
        raise ValueError(validation_failure)
    return AuthorizedRavdessTrial(
        trial_id=trial_id,
        actor_id=actor_id,
        cache_integrity_id=cache_integrity_id,
        cache_sha256=cache_sha256,
        cache_size_bytes=len(payload),
        features=_readonly_array(features),
        valid_mask=_readonly_array(valid),
        timestamps=_readonly_array(timestamps),
        frame_indices=_readonly_array(frame_indices),
        detector_confidence=_readonly_array(confidence),
    )


def authorize_committed_ravdess_semantic23(
    data_root: str | Path,
    *,
    id_key_path: str | Path | None = None,
) -> AuthorizedRavdessGeneration:
    """Authorize the exact committed RAVDESS generation without modifying it."""
    root = Path(data_root).expanduser().resolve()
    output = root / "derived_semantic23"
    key_path = root / RAVDESS_ID_KEY_RELATIVE_PATH
    if id_key_path is not None and _absolute_lexical_path(id_key_path) != key_path:
        raise ValueError("RAVDESS authorization requires the canonical private key")
    if not output.is_dir() or output.is_symlink():
        raise ValueError("committed RAVDESS semantic23 generation is missing or unsafe")
    trials_root = output / "trials"
    if not trials_root.is_dir() or trials_root.is_symlink():
        raise ValueError("committed RAVDESS trial directory is missing or unsafe")
    parent_descriptor, parent_identity = _open_output_parent(output.parent)
    descriptors = ExitStack()
    descriptors.callback(os.close, parent_descriptor)
    lock_name = f".{output.name}.lock"
    lock_descriptor: int | None = None
    lock_identity: tuple[int, int] | None = None
    output_descriptor: int | None = None
    trials_descriptor: int | None = None
    snapshots: list[
        tuple[int, str, Path, tuple[int, ...], str]
    ] = []
    try:
        lock_descriptor, lock_identity = _acquire_output_lock(
            parent_descriptor, lock_name, create_if_missing=False
        )
        _assert_no_unresolved_ravdess_state(parent_descriptor, output.name)
        output_descriptor = _open_directory_at(
            parent_descriptor, output.name, "committed RAVDESS generation"
        )
        descriptors.callback(os.close, output_descriptor)
        output_tree_identity, _ = _snapshot_exact_owner_directory(
            output_descriptor,
            ("manifest.json", "trials"),
            "committed RAVDESS generation",
        )
        trials_descriptor = _open_directory_at(
            output_descriptor, "trials", "committed RAVDESS trial cache"
        )
        descriptors.callback(os.close, trials_descriptor)
        private_key = _load_private_id_key(key_path)
        key_status = os.stat(key_path, follow_symlinks=False)
        key_snapshot = _private_key_stat_identity(key_status)
        live_inventory = audit_ravdess_inventory(
            root, expectation=FROZEN_RAVDESS_INVENTORY
        )
        source_pattern_index = _source_private_pattern_index(
            live_inventory.member_sha256
        )
        expected_actor_by_trial: dict[str, str] = {}
        for member_name, source_sha256 in live_inventory.member_sha256.items():
            trial_id = opaque_trial_id(
                member_name, source_sha256, key=private_key
            )
            actor_id = opaque_actor_id(
                _actor_token_from_name(Path(member_name)), key=private_key
            )
            if trial_id in expected_actor_by_trial:
                raise ValueError("RAVDESS live archive repeats an opaque trial ID")
            expected_actor_by_trial[trial_id] = actor_id
        manifest_path = output / "manifest.json"
        manifest_bytes, manifest_sha256, manifest_identity = (
            _read_owner_only_regular(
                Path("manifest.json"), "RAVDESS manifest",
                max_bytes=_MAX_RAVDESS_MANIFEST_BYTES,
                parent_descriptor=output_descriptor,
            )
        )
        snapshots.append((
            output_descriptor, "manifest.json", manifest_path,
            manifest_identity, "RAVDESS manifest",
        ))
        manifest = _load_unique_json_object(manifest_bytes, "RAVDESS manifest")
        _assert_manifest_deidentified(
            manifest_bytes.decode("utf-8"),
            source_root=root,
            source_paths=[
                Path(name) for name in live_inventory.member_sha256
            ],
            raw_source_sha256s=set(live_inventory.member_sha256.values()),
        )
        if set(manifest) != _AUTHORIZED_MANIFEST_FIELDS:
            raise ValueError("RAVDESS manifest field schema is not exact")
        expectation = FROZEN_RAVDESS_INVENTORY
        expected_inventory = _manifest_inventory(live_inventory)
        exact_values = {
            "format_version": 2,
            "schema": SEMANTIC23_SCHEMA,
            "feature_names": list(SEMANTIC23_FEATURE_NAMES),
            "feature_definitions": [asdict(item) for item in SEMANTIC23_DEFINITIONS],
            "source_dataset": "RAVDESS Facial Landmark Tracking",
            "license": "CC BY-NC-SA 4.0",
            "permitted_role": "non_clinical_healthy_motion_pretraining_only",
            "clinical_claim": "not_a_facial_palsy_or_house_brackmann_validation_cohort",
            "adapter": dict(OPENFACE68_ADAPTER_METADATA),
            "filter": {
                "confidence_operator": ">=",
                "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
                "required_coordinates": "finite_openface68_semantic_anchor_coordinates",
            },
            "timeline_policy": {
                "source_rows_preserved": True,
                "timestamps_preserved": True,
                "detector_gaps_preserved_in_valid_mask": True,
                "interpolation": "none",
                "masked_feature_storage": "zeros_with_required_valid_mask",
            },
            "provenance_policy": {
                "actor_id": "private_hmac_sha256_base32",
                "trial_id": (
                    "private_hmac_archive_member_name_"
                    "source_content_sha256_base32_v2"
                ),
                "cache_integrity_id": (
                    "private_hmac_trial_id_actor_id_cache_sha256_base32"
                ),
                "source_binding": (
                    "verified_archive_member_name_and_bytes_single_read"
                ),
                "raw_paths_or_filenames_in_manifest": False,
                "raw_source_content_sha256_in_manifest": False,
            },
            "inventory": expected_inventory,
        }
        if any(
            name not in manifest
            or not _json_exact_equal(manifest[name], value)
            for name, value in exact_values.items()
        ):
            raise ValueError("RAVDESS manifest policy or frozen inventory is noncanonical")
        raw_rows = manifest.get("trials")
        if not isinstance(raw_rows, list) or len(raw_rows) != expectation.csv_files:
            raise ValueError("RAVDESS manifest trial set is incomplete")
        rows: list[dict[str, str]] = []
        for value in raw_rows:
            if not isinstance(value, dict) or set(value) != _AUTHORIZED_TRIAL_FIELDS:
                raise ValueError("RAVDESS trial row field schema is not exact")
            row = {name: value[name] for name in _AUTHORIZED_TRIAL_FIELDS}
            if (
                not isinstance(row["trial_id"], str)
                or _TRIAL_ID.fullmatch(row["trial_id"]) is None
                or not isinstance(row["actor_id"], str)
                or _ACTOR_ID.fullmatch(row["actor_id"]) is None
                or not isinstance(row["cache_integrity_id"], str)
                or _CACHE_ID.fullmatch(row["cache_integrity_id"]) is None
            ):
                raise ValueError("RAVDESS trial row contains a noncanonical opaque ID")
            rows.append(row)
        if rows != sorted(rows, key=lambda item: item["trial_id"]):
            raise ValueError("RAVDESS trial rows are not in canonical keyed order")
        trial_ids = [row["trial_id"] for row in rows]
        if len(set(trial_ids)) != len(trial_ids):
            raise ValueError("RAVDESS manifest repeats a trial")
        if set(trial_ids) != set(expected_actor_by_trial):
            raise ValueError("RAVDESS manifest trial IDs do not match the live archive")
        if any(
            not hmac.compare_digest(
                row["actor_id"], expected_actor_by_trial[row["trial_id"]]
            )
            for row in rows
        ):
            raise ValueError("RAVDESS trial/group join does not match the live archive")
        if len({row["actor_id"] for row in rows}) != expectation.actors:
            raise ValueError("RAVDESS actor grouping is incomplete")
        expected_names = {f"{trial_id}.npz" for trial_id in trial_ids}
        trials_tree_identity, cache_name_tuple = _snapshot_exact_owner_directory(
            trials_descriptor,
            tuple(sorted(expected_names)),
            "committed RAVDESS trial cache",
        )
        cache_names = list(cache_name_tuple)
        authorized_trials: list[AuthorizedRavdessTrial] = []
        closure_rows: list[dict[str, object]] = []
        source_frames = valid_frames = 0
        rows_by_trial = {row["trial_id"]: row for row in rows}
        aggregate_declared_frames = 0
        aggregate_regular_payload_bytes = len(manifest_bytes)
        aggregate_expanded_bytes = 0
        aggregate_expanded_budget = _ravdess_aggregate_expanded_budget(
            expectation
        )
        preflight_cache_snapshots: dict[
            str, tuple[str, tuple[int, ...]]
        ] = {}
        for name in cache_names:
            row = rows_by_trial[Path(name).stem]
            cache_bytes, cache_sha256, cache_identity = _read_owner_only_regular(
                Path(name),
                "RAVDESS semantic23 cache preflight",
                max_bytes=_MAX_RAVDESS_CACHE_RAW_BYTES,
                parent_descriptor=trials_descriptor,
            )
            expected_cache_id = _opaque_cache_integrity_id(
                cache_sha256,
                trial_id=row["trial_id"],
                actor_id=row["actor_id"],
                key=private_key,
            )
            if not hmac.compare_digest(
                expected_cache_id, row["cache_integrity_id"]
            ):
                raise ValueError(
                    "RAVDESS cache integrity ID does not bind cache bytes"
                )
            declared_frames, expanded_bytes = _require_ravdess_npz_headers(
                cache_bytes
            )
            aggregate_declared_frames += declared_frames
            aggregate_regular_payload_bytes += len(cache_bytes)
            aggregate_expanded_bytes += expanded_bytes
            if (
                aggregate_declared_frames > expectation.frames
                or aggregate_regular_payload_bytes
                > _MAX_RAVDESS_AGGREGATE_REGULAR_PAYLOAD_BYTES
                or aggregate_expanded_bytes > aggregate_expanded_budget
            ):
                raise ValueError(
                    "RAVDESS cumulative cache resource budget is exceeded"
                )
            preflight_cache_snapshots[name] = (
                cache_sha256, cache_identity
            )
        if aggregate_declared_frames != expectation.frames:
            raise ValueError(
                "RAVDESS cumulative declared frame count is not exact"
            )
        for name in cache_names:
            path = trials_root / name
            row = rows_by_trial[path.stem]
            cache_bytes, cache_sha256, cache_identity = _read_owner_only_regular(
                Path(name), "RAVDESS semantic23 cache",
                max_bytes=_MAX_RAVDESS_CACHE_RAW_BYTES,
                parent_descriptor=trials_descriptor,
            )
            preflight_sha256, preflight_identity = (
                preflight_cache_snapshots[name]
            )
            if (
                not hmac.compare_digest(cache_sha256, preflight_sha256)
                or cache_identity != preflight_identity
            ):
                raise ValueError(
                    "RAVDESS cache changed after resource preflight"
                )
            snapshots.append((
                trials_descriptor, name, path, cache_identity,
                "RAVDESS semantic23 cache",
            ))
            expected_cache_id = _opaque_cache_integrity_id(
                cache_sha256,
                trial_id=row["trial_id"],
                actor_id=row["actor_id"],
                key=private_key,
            )
            if not hmac.compare_digest(expected_cache_id, row["cache_integrity_id"]):
                raise ValueError("RAVDESS cache integrity ID does not bind cache bytes")
            _assert_ravdess_cache_deidentified(
                cache_bytes,
                cache_name=name,
                source_pattern_index=source_pattern_index,
                raw_cache_sha256=cache_sha256,
            )
            trial = _validate_authorized_ravdess_cache(
                cache_bytes,
                trial_id=row["trial_id"], actor_id=row["actor_id"],
                cache_integrity_id=row["cache_integrity_id"],
                cache_sha256=cache_sha256,
            )
            authorized_trials.append(trial)
            source_frames += len(trial.features)
            valid_frames += int(trial.valid_mask.sum())
            closure_rows.append({
                "trial_id": trial.trial_id,
                "actor_id": trial.actor_id,
                "cache_integrity_id": trial.cache_integrity_id,
                "cache_sha256": cache_sha256,
                "cache_size_bytes": len(cache_bytes),
            })
        quality = manifest.get("quality_control")
        if not _json_exact_equal(quality, {
            "source_frames": source_frames,
            "valid_frames": valid_frames,
            "invalid_frames": source_frames - valid_frames,
        }) or source_frames != expectation.frames:
            raise ValueError("RAVDESS aggregate frame counts do not close")
        material = json.dumps({
            "manifest_sha256": manifest_sha256,
            "trials": closure_rows,
        }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
        closure_hmac = hmac.new(
            private_key,
            b"ravdess-semantic23-committed-generation-v1\0" + material,
            hashlib.sha256,
        ).hexdigest()
        repeated_inventory = audit_ravdess_inventory(
            root, expectation=FROZEN_RAVDESS_INVENTORY
        )
        if repeated_inventory != live_inventory:
            raise ValueError("RAVDESS live archive changed during authorization")
        final_key = _load_private_id_key(key_path)
        final_key_status = os.stat(key_path, follow_symlinks=False)
        if (
            not hmac.compare_digest(final_key, private_key)
            or _private_key_stat_identity(final_key_status) != key_snapshot
        ):
            raise ValueError("RAVDESS private key changed before authorization returned")
        _assert_no_unresolved_ravdess_state(parent_descriptor, output.name)
        for descriptor, name, path, identity, field in snapshots:
            _assert_owner_snapshot_at(descriptor, name, identity, field)
            _assert_owner_snapshot(path, identity, field)
        final_trials_tree_identity, _ = _snapshot_exact_owner_directory(
            trials_descriptor,
            tuple(sorted(expected_names)),
            "committed RAVDESS trial cache",
        )
        if final_trials_tree_identity != trials_tree_identity:
            raise ValueError("committed RAVDESS trial directory changed during authorization")
        final_output_tree_identity, _ = _snapshot_exact_owner_directory(
            output_descriptor,
            ("manifest.json", "trials"),
            "committed RAVDESS generation",
        )
        if final_output_tree_identity != output_tree_identity:
            raise ValueError("committed RAVDESS generation changed during authorization")
        return AuthorizedRavdessGeneration(
            schema=SEMANTIC23_SCHEMA,
            manifest_sha256=manifest_sha256,
            generation_closure_hmac=closure_hmac,
            trial_count=len(authorized_trials),
            actor_count=len({trial.actor_id for trial in authorized_trials}),
            source_frames=source_frames,
            valid_frames=valid_frames,
            expected_trial_count=expectation.csv_files,
            expected_actor_count=expectation.actors,
            trials=tuple(authorized_trials),
            private_key=private_key,
        )
    finally:
        try:
            if lock_descriptor is not None and lock_identity is not None:
                _release_output_lock(
                    output.parent, parent_descriptor, parent_identity,
                    lock_name, lock_descriptor, lock_identity,
                )
        finally:
            descriptors.__exit__(*sys.exc_info())


def _safe_entry_name(value: str) -> str:
    if (not isinstance(value, str) or not value or value in {".", ".."}
            or Path(value).name != value):
        raise ValueError("anchored output entry name must be one path component")
    return value


def _directory_identity(value: os.stat_result) -> tuple[int, int]:
    return int(value.st_dev), int(value.st_ino)


def _is_safe_output_parent(value: os.stat_result) -> bool:
    return (
        stat.S_ISDIR(value.st_mode)
        and int(value.st_uid) == os.geteuid()
        and stat.S_IMODE(value.st_mode) & 0o022 == 0
    )


def _assert_output_parent_identity(
    parent_path: Path,
    parent_descriptor: int,
    identity: tuple[int, int],
) -> None:
    opened = os.fstat(parent_descriptor)
    if (
        not _is_safe_output_parent(opened)
        or _directory_identity(opened) != identity
    ):
        raise ValueError("held output parent directory changed identity")
    try:
        current = os.stat(parent_path, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ValueError("output parent lexical path disappeared") from exc
    if (
        not _is_safe_output_parent(current)
        or _directory_identity(current) != identity
    ):
        raise ValueError("output parent lexical path changed identity")


def _open_output_parent(parent_path: Path) -> tuple[int, tuple[int, int]]:
    try:
        before = os.stat(parent_path, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ValueError("trusted output parent directory must already exist") from exc
    if not _is_safe_output_parent(before):
        raise ValueError(
            "trusted output parent must be current-owner and not group/world-writable"
        )
    try:
        descriptor = os.open(
            parent_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError(
                "trusted output parent must be a non-symlink directory"
            ) from exc
        raise
    try:
        opened = os.fstat(descriptor)
        identity = _directory_identity(opened)
        if (not _is_safe_output_parent(opened)
                or identity != _directory_identity(before)):
            raise ValueError("output parent changed while its directory fd was opened")
        _assert_output_parent_identity(parent_path, descriptor, identity)
        return descriptor, identity
    except BaseException:
        os.close(descriptor)
        raise


def _entry_stat(parent_descriptor: int, name: str) -> os.stat_result | None:
    name = _safe_entry_name(name)
    try:
        return os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _open_directory_at(parent_descriptor: int, name: str, field: str) -> int:
    name = _safe_entry_name(name)
    descriptor = os.open(
        name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=parent_descriptor,
    )
    try:
        opened = os.fstat(descriptor)
        current = os.stat(
            name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if (
            not stat.S_ISDIR(opened.st_mode)
            or int(opened.st_uid) != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != 0o700
            or not stat.S_ISDIR(current.st_mode)
            or int(current.st_uid) != os.geteuid()
            or stat.S_IMODE(current.st_mode) != 0o700
            or _directory_identity(opened) != _directory_identity(current)
        ):
            raise ValueError(f"{field} is not an anchored owner-only directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _create_directory_at(
    parent_descriptor: int,
    name: str,
    *,
    mode: int = 0o700,
) -> int:
    name = _safe_entry_name(name)
    os.mkdir(name, mode, dir_fd=parent_descriptor)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        os.fchmod(descriptor, mode)
        opened = os.fstat(descriptor)
        current = os.stat(
            name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if (not stat.S_ISDIR(opened.st_mode)
                or _directory_identity(opened) != _directory_identity(current)):
            raise ValueError("anchored staging directory changed identity")
        os.fsync(parent_descriptor)
        return descriptor
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        raise


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("anchored output write was incomplete")
        offset += written


def _write_bytes_at(
    parent_descriptor: int,
    name: str,
    payload: bytes,
) -> None:
    name = _safe_entry_name(name)
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=parent_descriptor,
    )
    try:
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        current = os.stat(
            name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if (not stat.S_ISREG(opened.st_mode)
                or _directory_identity(opened) != _directory_identity(current)):
            raise ValueError("anchored output file changed identity")
    finally:
        os.close(descriptor)


def _write_cache_at(
    parent_descriptor: int,
    name: str,
    trial: SemanticTrial,
) -> str:
    name = _safe_entry_name(name)
    descriptor = os.open(
        name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=parent_descriptor,
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w+b", closefd=False) as cache_handle:
            np.savez_compressed(
                cache_handle,
                features=trial.features,
                valid_mask=trial.valid_mask,
                timestamps=trial.timestamps,
                frame_indices=trial.frame_indices,
                detector_confidence=trial.detector_confidence,
                feature_names=np.asarray(SEMANTIC23_FEATURE_NAMES),
                schema=np.asarray(SEMANTIC23_SCHEMA),
                adapter_name=np.asarray(OPENFACE68_ADAPTER_METADATA["adapter_name"]),
                scale_normalization=np.asarray(
                    OPENFACE68_ADAPTER_METADATA["scale_normalization"]
                ),
                confidence_threshold=np.asarray(
                    DEFAULT_CONFIDENCE_THRESHOLD, dtype=np.float32
                ),
            )
            cache_handle.flush()
            os.fsync(descriptor)
            cache_sha256 = _stream_file_digest(cache_handle, "sha256")
        opened = os.fstat(descriptor)
        current = os.stat(
            name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if (not stat.S_ISREG(opened.st_mode)
                or _directory_identity(opened) != _directory_identity(current)):
            raise ValueError("anchored cache file changed identity")
        return cache_sha256
    finally:
        os.close(descriptor)


def _assert_no_unresolved_ravdess_state(
    parent_descriptor: int,
    output_name: str,
) -> None:
    output_name = _safe_entry_name(output_name)
    residue_prefixes = (
        f".{output_name}.staging-",
        f".{output_name}.backup-",
        f".{output_name}.tmp-",
    )
    residue_names = {
        f".{output_name}.transaction.json",
        f".{output_name}.journal.json",
    }
    if any(
        name in residue_names or name.startswith(residue_prefixes)
        for name in os.listdir(parent_descriptor)
    ):
        raise RuntimeError(
            "RAVDESS authorization rejects unresolved transaction state"
        )


def _acquire_output_lock(
    parent_descriptor: int,
    lock_name: str,
    *,
    create_if_missing: bool = True,
) -> tuple[int, tuple[int, int]]:
    lock_name = _safe_entry_name(lock_name)
    created = False
    if not create_if_missing:
        try:
            descriptor = os.open(
                lock_name, os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=parent_descriptor,
            )
        except FileNotFoundError as exc:
            raise ValueError(
                "committed authorization requires the existing producer lock"
            ) from exc
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ValueError("output lock must not be a symlink") from exc
            raise
    else:
        try:
            descriptor = os.open(
                lock_name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent_descriptor,
            )
            created = True
        except FileExistsError:
            try:
                descriptor = os.open(
                    lock_name,
                    os.O_RDWR | os.O_NOFOLLOW,
                    dir_fd=parent_descriptor,
                )
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise ValueError("output lock must not be a symlink") from exc
                raise
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ValueError("output lock must not be a symlink") from exc
            raise
    try:
        if created:
            os.fchmod(descriptor, 0o600)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("output lock must be a regular file")
        if int(info.st_uid) != os.geteuid():
            raise ValueError("output lock must be owned by the current user")
        if stat.S_IMODE(info.st_mode) != 0o600:
            raise ValueError("output lock permissions must be exactly 0600")
        if int(info.st_nlink) != 1:
            raise ValueError("output lock must have exactly one hard link")
        identity = (int(info.st_dev), int(info.st_ino))
        current = os.stat(
            lock_name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if (not stat.S_ISREG(current.st_mode)
                or int(current.st_uid) != os.geteuid()
                or stat.S_IMODE(current.st_mode) != 0o600
                or int(current.st_nlink) != 1
                or (int(current.st_dev), int(current.st_ino)) != identity):
            raise ValueError("output lock path identity or stat changed during open")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise BlockingIOError(
                    errno.EWOULDBLOCK,
                    f"derived output producer lock is held: {lock_name}",
                ) from exc
            raise
        after = os.fstat(descriptor)
        if (not stat.S_ISREG(after.st_mode)
                or int(after.st_uid) != os.geteuid()
                or stat.S_IMODE(after.st_mode) != 0o600
                or int(after.st_nlink) != 1
                or (int(after.st_dev), int(after.st_ino)) != identity):
            raise ValueError("output lock identity or stat changed during acquisition")
        current = os.stat(
            lock_name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if (not stat.S_ISREG(current.st_mode)
                or int(current.st_uid) != os.geteuid()
                or stat.S_IMODE(current.st_mode) != 0o600
                or int(current.st_nlink) != 1
                or (int(current.st_dev), int(current.st_ino)) != identity):
            raise ValueError("output lock path identity or stat changed during acquisition")
        if created:
            os.fsync(descriptor)
            os.fsync(parent_descriptor)
        return descriptor, identity
    except BaseException:
        os.close(descriptor)
        raise


def _release_output_lock(
    parent_path: Path,
    parent_descriptor: int,
    parent_identity: tuple[int, int],
    lock_name: str,
    descriptor: int,
    identity: tuple[int, int],
) -> None:
    lock_name = _safe_entry_name(lock_name)
    try:
        _assert_output_parent_identity(
            parent_path, parent_descriptor, parent_identity
        )
        current = os.stat(
            lock_name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if (not stat.S_ISREG(current.st_mode)
                or int(current.st_uid) != os.geteuid()
                or stat.S_IMODE(current.st_mode) != 0o600
                or int(current.st_nlink) != 1
                or (int(current.st_dev), int(current.st_ino)) != identity):
            raise ValueError("output lock path identity or stat changed before release")
        _assert_output_parent_identity(
            parent_path, parent_descriptor, parent_identity
        )
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _publish_directory_no_replace(
    parent_descriptor: int,
    stage_name: str,
    destination_name: str,
    expected_stage_identity: tuple[int, int],
) -> None:
    """Atomically publish two entries anchored to one trusted parent fd."""
    stage_name = _safe_entry_name(stage_name)
    destination_name = _safe_entry_name(destination_name)
    if (
        type(expected_stage_identity) is not tuple
        or len(expected_stage_identity) != 2
        or any(type(item) is not int for item in expected_stage_identity)
    ):
        raise ValueError("expected staging identity is malformed")
    staged = _entry_stat(parent_descriptor, stage_name)
    if (
        staged is None
        or not stat.S_ISDIR(staged.st_mode)
        or int(staged.st_uid) != os.geteuid()
        or stat.S_IMODE(staged.st_mode) != 0o700
        or _directory_identity(staged) != expected_stage_identity
    ):
        raise ValueError("staged generation changed identity before publication")
    if _entry_stat(parent_descriptor, destination_name) is not None:
        raise FileExistsError(
            f"derived semantic23 output already exists: {destination_name}"
        )
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(stage_name)
    destination_bytes = os.fsencode(destination_name)
    if sys.platform == "darwin":
        operation = library.renameatx_np
        no_replace_flag = 0x00000004
    elif sys.platform.startswith("linux"):
        operation = library.renameat2
        no_replace_flag = 0x00000001
    else:
        raise OSError("atomic no-replace directory publication is unsupported")
    operation.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    )
    operation.restype = ctypes.c_int
    if operation(
        parent_descriptor,
        source_bytes,
        parent_descriptor,
        destination_bytes,
        no_replace_flag,
    ) != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                f"derived semantic23 output already exists: {destination_name}"
            )
        raise OSError(error, os.strerror(error), destination_name)
    published = _entry_stat(parent_descriptor, destination_name)
    if (
        published is None
        or not stat.S_ISDIR(published.st_mode)
        or int(published.st_uid) != os.geteuid()
        or stat.S_IMODE(published.st_mode) != 0o700
        or _directory_identity(published) != expected_stage_identity
    ):
        raise ValueError("published output changed identity during anchored rename")
    os.fsync(parent_descriptor)


def _absolute_lexical_path(value: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(value).expanduser())))


def _output_path_preserving_descendants(
    data_root: str | Path,
    trusted_root: Path,
    output_root: str | Path,
) -> Path:
    """Canonicalize only the trusted root, never output descendants."""
    lexical_root = _absolute_lexical_path(data_root)
    lexical_output = _absolute_lexical_path(output_root)
    try:
        relative = lexical_output.relative_to(lexical_root)
    except ValueError:
        return lexical_output
    return trusted_root / relative


def _assert_output_path_safe_under_root(
    trusted_root: Path,
    output: Path,
) -> None:
    """Reject a present final path and symlinked descendants of the data root."""
    if os.path.lexists(output):
        raise FileExistsError(f"derived semantic23 output already exists: {output}")
    try:
        relative_parent = output.parent.relative_to(trusted_root)
    except ValueError:
        return
    current = trusted_root
    for component in relative_parent.parts:
        current /= component
        if not os.path.lexists(current):
            continue
        info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(
                f"derived output parent must not be a symlink: {current}"
            )
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(
                f"derived output parent must be a directory: {current}"
            )


def _linear_cleanup_cause(
    primary: BaseException | None,
    cleanup_errors: Sequence[BaseException],
    *,
    forbidden: Sequence[BaseException] = (),
) -> BaseException | None:
    """Preserve nested cleanup chains and append each one without a cycle."""
    cause = primary
    seen: set[int] = {id(error) for error in forbidden}
    current = primary
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    for error in cleanup_errors:
        if id(error) in seen:
            continue
        unique_nodes: list[BaseException] = []
        local_seen: set[int] = set()
        current = error
        while (
            current is not None
            and id(current) not in seen
            and id(current) not in local_seen
        ):
            local_seen.add(id(current))
            unique_nodes.append(current)
            current = current.__cause__ or current.__context__
        for node in reversed(unique_nodes):
            node.__cause__ = cause
            node.__context__ = None
            node.__suppress_context__ = True
            cause = node
            seen.add(id(node))
    return cause


def _attach_cleanup_causes(
    outcome: BaseException,
    cleanup_errors: Sequence[BaseException],
) -> BaseException:
    existing = outcome.__cause__ or outcome.__context__
    outcome.__cause__ = _linear_cleanup_cause(
        existing, cleanup_errors, forbidden=(outcome,),
    )
    outcome.__context__ = None
    outcome.__suppress_context__ = outcome.__cause__ is not None
    return outcome


def build_generation_from_audited_sources(
    data_root: str | Path,
    output_root: str | Path,
    inventory: RavdessInventory,
    *,
    expectation: RavdessInventoryExpectation = FROZEN_RAVDESS_INVENTORY,
    id_key: bytes | None = None,
    canonical_key_path: Path | None = None,
) -> dict[str, Any]:
    """Build a staged generation bound to one previously audited inventory.

    This lower-level entry point accepts an expectation so synthetic tests can
    exercise the full transaction.  Production callers use
    :func:`prepare_ravdess_semantic23`, which always pins the frozen inventory.
    Existing outputs are never overwritten implicitly.
    """
    source_root = Path(data_root).expanduser().resolve()
    output = _output_path_preserving_descendants(
        data_root, source_root, output_root
    )
    if (id_key is None) == (canonical_key_path is None):
        raise ValueError(
            "generation requires exactly one private-key input mode"
        )
    private_key: bytes | None = None
    canonical_key_identity: tuple[int, ...] | None = None
    if canonical_key_path is None:
        private_key = _private_id_key(id_key)
    else:
        expected_key_path = source_root / RAVDESS_ID_KEY_RELATIVE_PATH
        requested_key_path = _output_path_preserving_descendants(
            data_root, source_root, canonical_key_path
        )
        if requested_key_path != expected_key_path:
            raise ValueError(
                "generation requires the canonical private-key path"
            )
        canonical_key_path = expected_key_path
    _assert_output_path_safe_under_root(source_root, output)
    parent_descriptor, parent_identity = _open_output_parent(output.parent)
    descriptors = ExitStack()
    descriptors.callback(os.close, parent_descriptor)
    lock_name = f".{output.name}.lock"
    stage_name = f".{output.name}.staging-{uuid.uuid4().hex}"
    lock_descriptor: int | None = None
    lock_identity: tuple[int, int] | None = None
    stage_descriptor: int | None = None
    trials_descriptor: int | None = None
    stage_identity: tuple[int, int] | None = None
    committed = False
    pending_error: BaseException | None = None
    try:
        _assert_output_parent_identity(
            output.parent, parent_descriptor, parent_identity
        )
        if _entry_stat(parent_descriptor, output.name) is not None:
            raise FileExistsError(f"derived semantic23 output already exists: {output}")
        lock_descriptor, lock_identity = _acquire_output_lock(
            parent_descriptor, lock_name
        )
        _assert_output_parent_identity(
            output.parent, parent_descriptor, parent_identity
        )
        if _entry_stat(parent_descriptor, output.name) is not None:
            raise FileExistsError(f"derived semantic23 output already exists: {output}")
        _assert_no_unresolved_ravdess_state(parent_descriptor, output.name)
        expected_inventory = asdict(expectation)
        observed_inventory = _inventory_values(inventory)
        if any(
            name not in observed_inventory
            or not _json_exact_equal(observed_inventory[name], value)
            for name, value in expected_inventory.items()
        ):
            raise ValueError("audited inventory does not match the required archive")
        if canonical_key_path is not None:
            private_key = load_or_create_private_id_key(canonical_key_path)
            canonical_key_identity = _private_key_stat_identity(
                os.stat(canonical_key_path, follow_symlinks=False)
            )
        if private_key is None:
            raise RuntimeError("private-key mode did not produce key bytes")
        archive_path = source_root / RAVDESS_ARCHIVE_RELATIVE_PATH
        trial_id_by_member = {
            member_name: opaque_trial_id(
                member_name, source_sha256, key=private_key
            )
            for member_name, source_sha256 in inventory.member_sha256.items()
        }
        if len(set(trial_id_by_member.values())) != len(trial_id_by_member):
            raise ValueError("opaque trial ID collision detected")
        source_pattern_index = _source_private_pattern_index(
            inventory.member_sha256
        )
        inventory_snapshot = (
            inventory.archive_device, inventory.archive_inode,
            inventory.archive_size, inventory.archive_mtime_ns,
            inventory.archive_ctime_ns, inventory.archive_md5,
        )
        try:
            stage_descriptor = _create_directory_at(
                parent_descriptor, stage_name
            )
        except FileExistsError as exc:
            raise FileExistsError(
                f"staging path already exists: {stage_name}"
            ) from exc
        stage_identity = _directory_identity(os.fstat(stage_descriptor))
        descriptors.callback(os.close, stage_descriptor)
        trials_descriptor = _create_directory_at(stage_descriptor, "trials")
        descriptors.callback(os.close, trials_descriptor)
        _assert_output_parent_identity(
            output.parent, parent_descriptor, parent_identity
        )

        records: list[dict[str, Any]] = []
        raw_cache_sha256s: set[str] = set()
        cache_sha256_by_name: dict[str, str] = {}
        total_source_frames = 0
        total_valid_frames = 0
        with _open_verified_archive(
            archive_path, expectation
        ) as (archive_file, generation_snapshot):
            observed_snapshot = (
                generation_snapshot.device, generation_snapshot.inode,
                generation_snapshot.size, generation_snapshot.mtime_ns,
                generation_snapshot.ctime_ns, generation_snapshot.md5,
            )
            if observed_snapshot != inventory_snapshot:
                raise ValueError("RAVDESS archive snapshot changed before generation")
            with zipfile.ZipFile(archive_file, "r") as source_zip:
                members = _archive_csv_infos(source_zip)
                source_paths = [Path(member.filename) for member in members]
                member_names = [member.filename for member in members]
                if member_names != list(inventory.member_sha256):
                    raise ValueError("RAVDESS member names changed after inventory audit")
                for member in members:
                    member_bytes = _read_ravdess_member_bytes(
                        source_zip, member
                    )
                    observed_member_sha256 = hashlib.sha256(member_bytes).hexdigest()
                    if observed_member_sha256 != inventory.member_sha256[member.filename]:
                        raise ValueError(
                            "RAVDESS member bytes changed after inventory audit"
                        )
                    trial = _parse_ravdess_member_csv(
                        member_bytes, source_name=member.filename
                    )
                    source_path = Path(member.filename)
                    actor_id = opaque_actor_id(
                        _actor_token_from_name(source_path), key=private_key
                    )
                    trial_id = opaque_trial_id(
                        member.filename, observed_member_sha256, key=private_key
                    )
                    if trial_id != trial_id_by_member[member.filename]:
                        raise ValueError("opaque trial ID changed after preflight")
                    trial_source_frames = int(trial.features.shape[0])
                    trial_valid_frames = int(trial.valid_mask.sum())
                    total_source_frames += trial_source_frames
                    total_valid_frames += trial_valid_frames
                    cache_name = f"{trial_id}.npz"
                    try:
                        cache_sha256 = _write_cache_at(
                            trials_descriptor, cache_name, trial
                        )
                    except FileExistsError as exc:
                        raise ValueError("opaque trial ID collision detected") from exc
                    raw_cache_sha256s.add(cache_sha256)
                    cache_sha256_by_name[cache_name] = cache_sha256
                    records.append({
                        "trial_id": trial_id,
                        "actor_id": actor_id,
                        "cache_integrity_id": _opaque_cache_integrity_id(
                            cache_sha256,
                            trial_id=trial_id,
                            actor_id=actor_id,
                            key=private_key,
                        ),
                    })
        os.fsync(trials_descriptor)
        (
            staged_trials_identity,
            staged_cache_identities,
            staged_cache_regular_payload_bytes,
            _,
            _,
        ) = _scan_staged_ravdess_caches_for_private_provenance(
            trials_descriptor,
            cache_sha256_by_name,
            source_pattern_index,
            expectation=expectation,
        )
        _assert_output_parent_identity(
            output.parent, parent_descriptor, parent_identity
        )

        if len(records) != inventory.csv_files:
            raise ValueError("parsed trial count drifted after the frozen archive audit")
        if total_source_frames != inventory.frames:
            raise ValueError("parsed frame count drifted after the frozen inventory audit")
        records.sort(key=lambda item: str(item["trial_id"]))
        manifest: dict[str, Any] = {
            "format_version": 2,
            "schema": SEMANTIC23_SCHEMA,
            "feature_names": list(SEMANTIC23_FEATURE_NAMES),
            "feature_definitions": [asdict(item) for item in SEMANTIC23_DEFINITIONS],
            "source_dataset": "RAVDESS Facial Landmark Tracking",
            "license": "CC BY-NC-SA 4.0",
            "permitted_role": "non_clinical_healthy_motion_pretraining_only",
            "clinical_claim": "not_a_facial_palsy_or_house_brackmann_validation_cohort",
            "adapter": dict(OPENFACE68_ADAPTER_METADATA),
            "filter": {
                "confidence_operator": ">=",
                "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
                "required_coordinates": "finite_openface68_semantic_anchor_coordinates",
            },
            "timeline_policy": {
                "source_rows_preserved": True,
                "timestamps_preserved": True,
                "detector_gaps_preserved_in_valid_mask": True,
                "interpolation": "none",
                "masked_feature_storage": "zeros_with_required_valid_mask",
            },
            "provenance_policy": {
                "actor_id": "private_hmac_sha256_base32",
                "trial_id": (
                    "private_hmac_archive_member_name_"
                    "source_content_sha256_base32_v2"
                ),
                "cache_integrity_id": (
                    "private_hmac_trial_id_actor_id_cache_sha256_base32"
                ),
                "source_binding": (
                    "verified_archive_member_name_and_bytes_single_read"
                ),
                "raw_paths_or_filenames_in_manifest": False,
                "raw_source_content_sha256_in_manifest": False,
            },
            "inventory": _manifest_inventory(inventory),
            "quality_control": {
                "source_frames": total_source_frames,
                "valid_frames": total_valid_frames,
                "invalid_frames": total_source_frames - total_valid_frames,
            },
            "trials": records,
        }
        manifest_text = json.dumps(
            manifest, sort_keys=True, indent=2, ensure_ascii=True
        ) + "\n"
        _assert_manifest_deidentified(
            manifest_text,
            source_root=source_root,
            source_paths=source_paths,
            raw_source_sha256s=set(inventory.member_sha256.values()),
            raw_cache_sha256s=raw_cache_sha256s,
        )
        _assert_output_parent_identity(
            output.parent, parent_descriptor, parent_identity
        )
        manifest_payload = manifest_text.encode("utf-8")
        if (
            len(manifest_payload) + staged_cache_regular_payload_bytes
            > _MAX_RAVDESS_AGGREGATE_REGULAR_PAYLOAD_BYTES
        ):
            raise ValueError(
                "staged RAVDESS aggregate regular payload budget is exceeded"
            )
        _write_bytes_at(stage_descriptor, "manifest.json", manifest_payload)
        os.fsync(stage_descriptor)
        (
            staged_manifest_payload,
            _,
            staged_manifest_identity,
        ) = _read_owner_only_regular(
            Path("manifest.json"),
            "staged RAVDESS manifest",
            max_bytes=_MAX_RAVDESS_MANIFEST_BYTES,
            parent_descriptor=stage_descriptor,
        )
        if not hmac.compare_digest(staged_manifest_payload, manifest_payload):
            raise ValueError("staged RAVDESS manifest changed after write")
        _assert_manifest_deidentified(
            staged_manifest_payload.decode("utf-8"),
            source_root=source_root,
            source_paths=source_paths,
            raw_source_sha256s=set(inventory.member_sha256.values()),
            raw_cache_sha256s=raw_cache_sha256s,
        )
        staged_output_identity, _ = _snapshot_exact_owner_directory(
            stage_descriptor,
            ("manifest.json", "trials"),
            "staged RAVDESS generation",
        )
        _assert_output_parent_identity(
            output.parent, parent_descriptor, parent_identity
        )
        current_trials_identity, _ = _snapshot_exact_owner_directory(
            trials_descriptor,
            tuple(sorted(cache_sha256_by_name)),
            "staged RAVDESS trial cache",
        )
        if current_trials_identity != staged_trials_identity:
            raise ValueError("staged RAVDESS trial cache changed before publication")
        for cache_name, cache_identity in staged_cache_identities.items():
            _assert_owner_snapshot_at(
                trials_descriptor,
                cache_name,
                cache_identity,
                "staged RAVDESS semantic23 cache",
            )
        staged_manifest_identity = _revalidate_ctime_only_owner_snapshot_at(
            stage_descriptor,
            "manifest.json",
            staged_manifest_identity,
            manifest_payload,
            "staged RAVDESS manifest",
            max_bytes=_MAX_RAVDESS_MANIFEST_BYTES,
        )
        current_output_identity, _ = _snapshot_exact_owner_directory(
            stage_descriptor,
            ("manifest.json", "trials"),
            "staged RAVDESS generation",
        )
        if current_output_identity != staged_output_identity:
            raise ValueError("staged RAVDESS generation changed before publication")
        _publish_directory_no_replace(
            parent_descriptor, stage_name, output.name, stage_identity
        )
        _assert_output_parent_identity(
            output.parent, parent_descriptor, parent_identity
        )
        canonical_output_descriptor = _open_directory_at(
            parent_descriptor,
            output.name,
            "published RAVDESS generation",
        )
        descriptors.callback(os.close, canonical_output_descriptor)
        if (
            _directory_identity(os.fstat(canonical_output_descriptor))
            != stage_identity
            or _directory_identity(os.fstat(stage_descriptor))
            != stage_identity
        ):
            raise ValueError(
                "published RAVDESS generation is not the held staging inode"
            )
        canonical_trials_descriptor = _open_directory_at(
            canonical_output_descriptor,
            "trials",
            "published RAVDESS trial cache",
        )
        descriptors.callback(os.close, canonical_trials_descriptor)
        if _directory_identity(os.fstat(canonical_trials_descriptor)) != (
            _directory_identity(os.fstat(trials_descriptor))
        ):
            raise ValueError(
                "published RAVDESS trial cache is not the held staging inode"
            )

        held_output_identity, _ = _snapshot_exact_owner_directory(
            stage_descriptor,
            ("manifest.json", "trials"),
            "held published RAVDESS generation",
        )
        canonical_output_identity, _ = _snapshot_exact_owner_directory(
            canonical_output_descriptor,
            ("manifest.json", "trials"),
            "published RAVDESS generation",
        )
        if held_output_identity != canonical_output_identity:
            raise ValueError("published RAVDESS generation tree changed")
        postpublish_output_identity = held_output_identity
        expected_cache_names = tuple(sorted(cache_sha256_by_name))
        held_trials_identity, _ = _snapshot_exact_owner_directory(
            trials_descriptor,
            expected_cache_names,
            "held published RAVDESS trial cache",
        )
        canonical_trials_identity, _ = _snapshot_exact_owner_directory(
            canonical_trials_descriptor,
            expected_cache_names,
            "published RAVDESS trial cache",
        )
        if (
            held_trials_identity != staged_trials_identity
            or canonical_trials_identity != staged_trials_identity
        ):
            raise ValueError("published RAVDESS trial cache tree changed")

        held_manifest, _, held_manifest_identity = _read_owner_only_regular(
            Path("manifest.json"),
            "held published RAVDESS manifest",
            max_bytes=_MAX_RAVDESS_MANIFEST_BYTES,
            parent_descriptor=stage_descriptor,
        )
        canonical_manifest, _, canonical_manifest_identity = (
            _read_owner_only_regular(
                Path("manifest.json"),
                "published RAVDESS manifest",
                max_bytes=_MAX_RAVDESS_MANIFEST_BYTES,
                parent_descriptor=canonical_output_descriptor,
            )
        )
        if (
            not hmac.compare_digest(held_manifest, manifest_payload)
            or not hmac.compare_digest(canonical_manifest, manifest_payload)
            or held_manifest_identity != staged_manifest_identity
            or canonical_manifest_identity != staged_manifest_identity
        ):
            raise ValueError("published RAVDESS manifest changed")
        canonical_manifest_object = _load_unique_json_object(
            canonical_manifest, "published RAVDESS manifest"
        )
        if not _json_exact_equal(canonical_manifest_object, manifest):
            raise ValueError("published RAVDESS manifest schema changed")
        _assert_manifest_deidentified(
            canonical_manifest.decode("utf-8"),
            source_root=source_root,
            source_paths=source_paths,
            raw_source_sha256s=set(inventory.member_sha256.values()),
            raw_cache_sha256s=raw_cache_sha256s,
        )

        rows_by_cache = {
            f"{row['trial_id']}.npz": row for row in records
        }
        for cache_name in expected_cache_names:
            held_cache, held_sha256, held_identity = _read_owner_only_regular(
                Path(cache_name),
                "held published RAVDESS semantic23 cache",
                max_bytes=_MAX_RAVDESS_CACHE_RAW_BYTES,
                parent_descriptor=trials_descriptor,
            )
            canonical_cache, canonical_sha256, canonical_identity = (
                _read_owner_only_regular(
                    Path(cache_name),
                    "published RAVDESS semantic23 cache",
                    max_bytes=_MAX_RAVDESS_CACHE_RAW_BYTES,
                    parent_descriptor=canonical_trials_descriptor,
                )
            )
            expected_sha256 = cache_sha256_by_name[cache_name]
            expected_identity = staged_cache_identities[cache_name]
            if (
                not hmac.compare_digest(held_cache, canonical_cache)
                or not hmac.compare_digest(held_sha256, expected_sha256)
                or not hmac.compare_digest(canonical_sha256, expected_sha256)
                or held_identity != expected_identity
                or canonical_identity != expected_identity
            ):
                raise ValueError("published RAVDESS semantic23 cache changed")
            row = rows_by_cache[cache_name]
            expected_cache_id = _opaque_cache_integrity_id(
                canonical_sha256,
                trial_id=str(row["trial_id"]),
                actor_id=str(row["actor_id"]),
                key=private_key,
            )
            if not hmac.compare_digest(
                expected_cache_id, str(row["cache_integrity_id"])
            ):
                raise ValueError(
                    "published RAVDESS cache integrity ID changed"
                )
            _assert_ravdess_cache_deidentified(
                canonical_cache,
                cache_name=cache_name,
                source_pattern_index=source_pattern_index,
                raw_cache_sha256=canonical_sha256,
            )
            _validate_authorized_ravdess_cache(
                canonical_cache,
                trial_id=str(row["trial_id"]),
                actor_id=str(row["actor_id"]),
                cache_integrity_id=str(row["cache_integrity_id"]),
                cache_sha256=canonical_sha256,
            )

        repeated_inventory = audit_ravdess_inventory(
            source_root, expectation=expectation
        )
        if repeated_inventory != inventory:
            raise ValueError(
                "RAVDESS source inventory changed after publication"
            )
        if canonical_key_path is not None:
            final_key = _load_private_id_key(canonical_key_path)
            final_key_identity = _private_key_stat_identity(
                os.stat(canonical_key_path, follow_symlinks=False)
            )
            if (
                not hmac.compare_digest(final_key, private_key)
                or final_key_identity != canonical_key_identity
            ):
                raise ValueError(
                    "canonical private key changed after publication"
                )
        _assert_output_parent_identity(
            output.parent, parent_descriptor, parent_identity
        )
        opened_lock = os.fstat(lock_descriptor)
        current_lock = os.stat(
            lock_name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(opened_lock.st_mode)
            or not stat.S_ISREG(current_lock.st_mode)
            or _directory_identity(opened_lock) != lock_identity
            or _directory_identity(current_lock) != lock_identity
        ):
            raise ValueError("output lock changed during publication validation")
        final_held_trials_identity, _ = _snapshot_exact_owner_directory(
            trials_descriptor,
            expected_cache_names,
            "held published RAVDESS trial cache",
        )
        final_canonical_trials_identity, _ = _snapshot_exact_owner_directory(
            canonical_trials_descriptor,
            expected_cache_names,
            "published RAVDESS trial cache",
        )
        final_held_output_identity, _ = _snapshot_exact_owner_directory(
            stage_descriptor,
            ("manifest.json", "trials"),
            "held published RAVDESS generation",
        )
        final_canonical_output_identity, _ = _snapshot_exact_owner_directory(
            canonical_output_descriptor,
            ("manifest.json", "trials"),
            "published RAVDESS generation",
        )
        if (
            final_held_trials_identity != staged_trials_identity
            or final_canonical_trials_identity != staged_trials_identity
            or final_held_output_identity != postpublish_output_identity
            or final_canonical_output_identity != postpublish_output_identity
        ):
            raise ValueError(
                "published RAVDESS generation changed during validation"
            )
        try:
            _release_output_lock(
                output.parent,
                parent_descriptor,
                parent_identity,
                lock_name,
                lock_descriptor,
                lock_identity,
            )
        finally:
            # _release_output_lock always closes this descriptor, including
            # when its final lexical-parent validation fails.
            lock_descriptor = None
        _assert_output_parent_identity(
            output.parent, parent_descriptor, parent_identity
        )
        committed = True
        return manifest
    except BaseException as caught:
        pending_error = caught
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        try:
            if lock_descriptor is not None and lock_identity is not None:
                _release_output_lock(
                    output.parent,
                    parent_descriptor,
                    parent_identity,
                    lock_name,
                    lock_descriptor,
                    lock_identity,
                )
        except BaseException as caught:
            cleanup_errors.append(caught)
        try:
            descriptors.close()
        except BaseException as caught:
            cleanup_errors.append(caught)
        if pending_error is not None and stage_identity is not None and not committed:
            retained_cause = _linear_cleanup_cause(
                pending_error, cleanup_errors,
            )
            assert retained_cause is not None
            raise RuntimeError(
                "RAVDESS generation storage is retained as indeterminate"
            ) from retained_cause
        if cleanup_errors:
            if pending_error is not None:
                outcome = _attach_cleanup_causes(
                    pending_error, cleanup_errors,
                )
                raise outcome.with_traceback(pending_error.__traceback__)
            cleanup_cause = _linear_cleanup_cause(None, cleanup_errors)
            assert cleanup_cause is not None
            raise cleanup_cause


def prepare_ravdess_semantic23(
    data_root: str | Path,
    *,
    output_root: str | Path | None = None,
    id_key_path: str | Path | None = None,
) -> dict[str, Any]:
    """Production entry point pinned to the frozen RAVDESS inventory."""
    root = Path(data_root).expanduser().resolve()
    output = root / "derived_semantic23"
    key_path = root / RAVDESS_ID_KEY_RELATIVE_PATH
    if id_key_path is not None:
        requested_key = _output_path_preserving_descendants(
            data_root, root, id_key_path
        )
        if requested_key != key_path:
            raise ValueError(
                "production RAVDESS requires the canonical private key"
            )
    if output_root is not None:
        requested_output = _output_path_preserving_descendants(
            data_root, root, output_root
        )
        if requested_output != output:
            raise ValueError(
                "production RAVDESS output must be the exact lexical canonical "
                f"<data-root>/derived_semantic23 directory: {output}"
            )
    _assert_output_path_safe_under_root(root, output)
    inventory = audit_ravdess_inventory(root, expectation=FROZEN_RAVDESS_INVENTORY)
    return build_generation_from_audited_sources(
        root, output, inventory, expectation=FROZEN_RAVDESS_INVENTORY,
        canonical_key_path=key_path,
    )


class _PathRedactingArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        self.exit(2, '{"error":"RAVDESS command arguments invalid","status":"error"}\n')


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_root = ROOT / "data" / "external" / "ravdess_facial_tracking"
    parser = _PathRedactingArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_root)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--execute", action="store_true",
        help="create the derived generation; without this flag the command is read-only",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.execute:
        manifest = prepare_ravdess_semantic23(
            args.data_root, output_root=args.output_root
        )
        print(json.dumps({
            "status": "generated",
            "schema": manifest["schema"],
            "trials": len(manifest["trials"]),
            "source_frames": manifest["inventory"]["source_frames"],
        }, sort_keys=True))
    else:
        inventory = audit_ravdess_inventory(
            args.data_root, expectation=FROZEN_RAVDESS_INVENTORY
        )
        print(json.dumps({
            "status": "audit_ok",
            **_manifest_inventory(inventory),
        }, sort_keys=True))
    return 0


def _run_cli(argv: Sequence[str] | None = None) -> int:
    try:
        return main(argv)
    except Exception:  # noqa: BLE001 - public CLI emits one fixed safe failure
        print(
            json.dumps(
                {"error": "RAVDESS command failed", "status": "error"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(_run_cli())


__all__ = [
    "AuthorizedRavdessGeneration",
    "AuthorizedRavdessTrial",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "FROZEN_RAVDESS_INVENTORY",
    "RAVDESS_ARCHIVE_RELATIVE_PATH",
    "RAVDESS_ID_KEY_RELATIVE_PATH",
    "RavdessInventory",
    "RavdessInventoryExpectation",
    "SemanticTrial",
    "audit_ravdess_inventory",
    "authorize_committed_ravdess_semantic23",
    "build_generation_from_audited_sources",
    "load_or_create_private_id_key",
    "opaque_actor_id",
    "opaque_trial_id",
    "parse_openface_csv",
    "parse_openface_csv_bytes",
    "prepare_ravdess_semantic23",
]
