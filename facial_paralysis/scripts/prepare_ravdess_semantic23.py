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
import secrets
import stat
import sys
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class RavdessInventoryExpectation:
    archive_size: int
    archive_md5: str
    csv_files: int
    actors: int
    frames: int
    header_sha256: str
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

    with _open_verified_archive(archive, expectation) as (archive_file, snapshot):
        with zipfile.ZipFile(archive_file, "r") as source_zip:
            members = _archive_csv_infos(source_zip)
            for member in members:
                path = Path(member.filename)
                actors.add(_actor_token_from_name(path))
                member_bytes = source_zip.read(member)
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
    inventory = RavdessInventory(
        archive_size=snapshot.size,
        archive_md5=snapshot.md5,
        csv_files=len(members),
        actors=len(actors),
        frames=frames,
        header_sha256=next(iter(header_hashes)),
        empty_trials=empty_trials,
        repeated_headers=repeated_headers,
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


def parse_openface_csv_bytes(
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
            try:
                frame = int(row["frame"])
                timestamp = float(row["timestamp"])
                confidence = float(row["confidence"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid frame metadata at CSV row {row_number}") from exc
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


def load_or_create_private_id_key(path: str | Path) -> bytes:
    """Load or atomically create one owner-only stable HMAC key."""
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        return _load_private_id_key(destination)

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    try:
        descriptor = os.open(destination, flags, 0o600)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError("private ID key must not be a symlink") from exc
        raise
    created_identity: tuple[int, int] | None = None
    try:
        os.fchmod(descriptor, 0o600)
        initial = os.fstat(descriptor)
        created_identity = (int(initial.st_dev), int(initial.st_ino))
        _validate_private_key_stat(initial)
        payload = secrets.token_bytes(PRIVATE_ID_KEY_BYTES)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("private ID key write was incomplete")
            offset += written
        os.fsync(descriptor)
        final = os.fstat(descriptor)
        _validate_private_key_stat(final, exact_size=PRIVATE_ID_KEY_BYTES)
        if (int(final.st_dev), int(final.st_ino)) != created_identity:
            raise ValueError("private ID key identity changed during creation")
        _assert_private_key_path_identity(destination, final)
        return payload
    except BaseException:
        try:
            current = os.stat(destination, follow_symlinks=False)
        except FileNotFoundError:
            current = None
        if (current is not None and created_identity is not None
                and (int(current.st_dev), int(current.st_ino)) == created_identity):
            os.unlink(destination)
        raise
    finally:
        os.close(descriptor)


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


def opaque_trial_id(source_sha256: str, *, key: bytes) -> str:
    """Stable private-key pseudonym for one source-content digest."""
    return _opaque_id("trial", source_sha256, "trial", key=key)


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
    }


def _assert_manifest_deidentified(
    manifest_text: str,
    *,
    source_root: Path,
    source_paths: list[Path],
    raw_source_sha256s: set[str],
    raw_cache_sha256s: set[str] | frozenset[str] = frozenset(),
) -> None:
    if str(source_root) in manifest_text:
        raise ValueError("aggregate manifest contains the raw source root")
    leaked = [path.name for path in source_paths if path.name in manifest_text]
    if leaked:
        raise ValueError("aggregate manifest contains raw source filenames")
    leaked_digests = [digest for digest in raw_source_sha256s | set(raw_cache_sha256s)
                      if digest in manifest_text]
    if leaked_digests:
        raise ValueError("aggregate manifest contains raw source or cache digests")


def _safe_entry_name(value: str) -> str:
    if (not isinstance(value, str) or not value or value in {".", ".."}
            or Path(value).name != value):
        raise ValueError("anchored output entry name must be one path component")
    return value


def _directory_identity(value: os.stat_result) -> tuple[int, int]:
    return int(value.st_dev), int(value.st_ino)


def _assert_output_parent_identity(
    parent_path: Path,
    parent_descriptor: int,
    identity: tuple[int, int],
) -> None:
    opened = os.fstat(parent_descriptor)
    if not stat.S_ISDIR(opened.st_mode) or _directory_identity(opened) != identity:
        raise ValueError("held output parent directory changed identity")
    try:
        current = os.stat(parent_path, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ValueError("output parent lexical path disappeared") from exc
    if not stat.S_ISDIR(current.st_mode) or _directory_identity(current) != identity:
        raise ValueError("output parent lexical path changed identity")


def _open_output_parent(parent_path: Path) -> tuple[int, tuple[int, int]]:
    try:
        before = os.stat(parent_path, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ValueError("trusted output parent directory must already exist") from exc
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError("trusted output parent must be a non-symlink directory")
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
        if (not stat.S_ISDIR(opened.st_mode)
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


def _remove_tree_at(parent_descriptor: int, name: str) -> bool:
    name = _safe_entry_name(name)
    info = _entry_stat(parent_descriptor, name)
    if info is None:
        return False
    if stat.S_ISDIR(info.st_mode):
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        try:
            for child in os.listdir(descriptor):
                _remove_tree_at(descriptor, child)
        finally:
            os.close(descriptor)
        os.rmdir(name, dir_fd=parent_descriptor)
    else:
        os.unlink(name, dir_fd=parent_descriptor)
    os.fsync(parent_descriptor)
    return True


def _acquire_output_lock(
    parent_descriptor: int,
    lock_name: str,
) -> tuple[int, tuple[int, int]]:
    lock_name = _safe_entry_name(lock_name)
    created = False
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
        identity = (int(info.st_dev), int(info.st_ino))
        current = os.stat(
            lock_name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if (not stat.S_ISREG(current.st_mode)
                or int(current.st_uid) != os.geteuid()
                or stat.S_IMODE(current.st_mode) != 0o600
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
                or (int(after.st_dev), int(after.st_ino)) != identity):
            raise ValueError("output lock identity or stat changed during acquisition")
        current = os.stat(
            lock_name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if (not stat.S_ISREG(current.st_mode)
                or int(current.st_uid) != os.geteuid()
                or stat.S_IMODE(current.st_mode) != 0o600
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
) -> None:
    """Atomically publish two entries anchored to one trusted parent fd."""
    stage_name = _safe_entry_name(stage_name)
    destination_name = _safe_entry_name(destination_name)
    staged = _entry_stat(parent_descriptor, stage_name)
    if staged is None or not stat.S_ISDIR(staged.st_mode):
        raise ValueError("staged generation must be a directory")
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
    if published is None or _directory_identity(published) != _directory_identity(staged):
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


def build_generation_from_audited_sources(
    data_root: str | Path,
    output_root: str | Path,
    inventory: RavdessInventory,
    *,
    expectation: RavdessInventoryExpectation = FROZEN_RAVDESS_INVENTORY,
    id_key: bytes,
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
    private_key = _private_id_key(id_key)
    _assert_output_path_safe_under_root(source_root, output)
    parent_descriptor, parent_identity = _open_output_parent(output.parent)
    lock_name = f".{output.name}.lock"
    stage_name = f".{output.name}.staging-{uuid.uuid4().hex}"
    lock_descriptor: int | None = None
    lock_identity: tuple[int, int] | None = None
    stage_descriptor: int | None = None
    trials_descriptor: int | None = None
    stage_identity: tuple[int, int] | None = None
    committed = False
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
        archive_path = source_root / RAVDESS_ARCHIVE_RELATIVE_PATH
        expected_inventory = asdict(expectation)
        observed_inventory = _inventory_values(inventory)
        if any(observed_inventory[name] != value
               for name, value in expected_inventory.items()):
            raise ValueError("audited inventory does not match the required archive")
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
        trials_descriptor = _create_directory_at(stage_descriptor, "trials")
        _assert_output_parent_identity(
            output.parent, parent_descriptor, parent_identity
        )

        records: list[dict[str, Any]] = []
        raw_cache_sha256s: set[str] = set()
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
                    member_bytes = source_zip.read(member)
                    observed_member_sha256 = hashlib.sha256(member_bytes).hexdigest()
                    if observed_member_sha256 != inventory.member_sha256[member.filename]:
                        raise ValueError(
                            "RAVDESS member bytes changed after inventory audit: "
                            f"{member.filename}"
                        )
                    trial = parse_openface_csv_bytes(
                        member_bytes, source_name=member.filename
                    )
                    source_path = Path(member.filename)
                    actor_id = opaque_actor_id(
                        _actor_token_from_name(source_path), key=private_key
                    )
                    trial_id = opaque_trial_id(trial.source_sha256, key=private_key)
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
                    records.append({
                        "trial_id": trial_id,
                        "actor_id": actor_id,
                        "cache_integrity_id": _opaque_id(
                            "cache-integrity", cache_sha256, "cache",
                            key=private_key,
                        ),
                    })
        os.fsync(trials_descriptor)
        _assert_output_parent_identity(
            output.parent, parent_descriptor, parent_identity
        )

        if len(records) != inventory.csv_files:
            raise ValueError("parsed trial count drifted after the frozen archive audit")
        if total_source_frames != inventory.frames:
            raise ValueError("parsed frame count drifted after the frozen inventory audit")
        records.sort(key=lambda item: str(item["trial_id"]))
        manifest: dict[str, Any] = {
            "format_version": 1,
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
                "trial_id": "private_hmac_source_content_sha256_base32",
                "cache_integrity_id": "private_hmac_cache_sha256_base32",
                "source_binding": "verified_archive_member_bytes_single_read",
                "raw_paths_or_filenames_in_manifest": False,
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
        _write_bytes_at(
            stage_descriptor, "manifest.json", manifest_text.encode("utf-8")
        )
        os.fsync(stage_descriptor)
        _assert_output_parent_identity(
            output.parent, parent_descriptor, parent_identity
        )
        _publish_directory_no_replace(
            parent_descriptor, stage_name, output.name
        )
        _assert_output_parent_identity(
            output.parent, parent_descriptor, parent_identity
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
    finally:
        if trials_descriptor is not None:
            os.close(trials_descriptor)
        if stage_descriptor is not None:
            os.close(stage_descriptor)
        try:
            if not committed and stage_identity is not None:
                published = _entry_stat(parent_descriptor, output.name)
                if (published is not None
                        and _directory_identity(published) == stage_identity):
                    _remove_tree_at(parent_descriptor, output.name)
                staged = _entry_stat(parent_descriptor, stage_name)
                if staged is not None:
                    if _directory_identity(staged) != stage_identity:
                        raise ValueError(
                            "staging entry changed identity before cleanup"
                        )
                    _remove_tree_at(parent_descriptor, stage_name)
        finally:
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
            finally:
                os.close(parent_descriptor)


def prepare_ravdess_semantic23(
    data_root: str | Path,
    *,
    output_root: str | Path | None = None,
    id_key_path: str | Path | None = None,
) -> dict[str, Any]:
    """Production entry point pinned to the frozen RAVDESS inventory."""
    root = Path(data_root).expanduser().resolve()
    output = root / "derived_semantic23"
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
    key_path = (
        Path(id_key_path).expanduser()
        if id_key_path is not None
        else root / RAVDESS_ID_KEY_RELATIVE_PATH
    )
    private_key = load_or_create_private_id_key(key_path)
    return build_generation_from_audited_sources(
        root, output, inventory, expectation=FROZEN_RAVDESS_INVENTORY,
        id_key=private_key,
    )


def _parse_args() -> argparse.Namespace:
    default_root = ROOT / "data" / "external" / "ravdess_facial_tracking"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_root)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--execute", action="store_true",
        help="create the derived generation; without this flag the command is read-only",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
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
        inventory = audit_ravdess_inventory(args.data_root)
        print(json.dumps({
            "status": "audit_ok",
            **_manifest_inventory(inventory),
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "FROZEN_RAVDESS_INVENTORY",
    "RAVDESS_ARCHIVE_RELATIVE_PATH",
    "RAVDESS_ID_KEY_RELATIVE_PATH",
    "RavdessInventory",
    "RavdessInventoryExpectation",
    "SemanticTrial",
    "audit_ravdess_inventory",
    "build_generation_from_audited_sources",
    "load_or_create_private_id_key",
    "opaque_actor_id",
    "opaque_trial_id",
    "parse_openface_csv",
    "parse_openface_csv_bytes",
    "prepare_ravdess_semantic23",
]
