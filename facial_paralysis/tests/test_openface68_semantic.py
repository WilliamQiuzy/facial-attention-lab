"""Contracts for source-neutral semantic23 and the RAVDESS OpenFace adapter.

All fixtures are synthetic.  The production RAVDESS corpus is audited by the
preparation script, but this test module never reads or writes that corpus.
"""
from __future__ import annotations

import base64
import contextlib
import csv
import fcntl
import hashlib
import hmac
import io
import json
import logging
import os
import stat
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.prepare_ravdess_semantic23 as prep  # noqa: E402
from scripts.prepare_ravdess_semantic23 import (  # noqa: E402
    RAVDESS_ARCHIVE_RELATIVE_PATH,
    RavdessInventoryExpectation,
    audit_ravdess_inventory,
    build_generation_from_audited_sources,
    opaque_actor_id,
    opaque_trial_id,
    parse_openface_csv,
)
from src.preprocessing.openface68_semantic import (  # noqa: E402
    OPENFACE68_ADAPTER_METADATA,
    OPENFACE68_MIDLINE,
    OPENFACE68_MOUTH_BOTTOM,
    OPENFACE68_MOUTH_TOP,
    OPENFACE68_REQUIRED_INDICES,
    OPENFACE68_SIDE_A_BROW,
    OPENFACE68_SIDE_A_CORNER,
    OPENFACE68_SIDE_A_EYE_RING,
    OPENFACE68_SIDE_A_LOWER,
    OPENFACE68_SIDE_A_UPPER,
    OPENFACE68_SIDE_B_BROW,
    OPENFACE68_SIDE_B_CORNER,
    OPENFACE68_SIDE_B_EYE_RING,
    OPENFACE68_SIDE_B_LOWER,
    OPENFACE68_SIDE_B_UPPER,
    openface68_to_semantic23,
)
from src.preprocessing.semantic_landmarks import (  # noqa: E402
    CLINICAL23_V2_ADAPTER_METADATA,
    SEMANTIC23_DEFINITIONS,
    SEMANTIC23_FEATURE_NAMES,
    SEMANTIC23_SCHEMA,
    clinical23_v2_to_semantic23,
)
from _testlib import Check, run_all  # noqa: E402


EXPECTED_NAMES = (
    "fissure_h_side_a", "fissure_h_side_b", "fissure_h_absdiff",
    "fissure_h_side_a_minus_side_b",
    "fissure_w_side_a", "fissure_w_side_b", "fissure_w_absdiff",
    "eye_measure_side_a", "eye_measure_side_b", "eye_measure_absdiff",
    "brow_h_side_a", "brow_h_side_b", "brow_h_absdiff",
    "brow_h_side_a_minus_side_b",
    "corner_y_side_a", "corner_y_side_b", "corner_y_absdiff",
    "corner_y_side_a_minus_side_b",
    "corner_x_side_a", "corner_x_side_b", "corner_x_absdiff",
    "mouth_width", "mouth_open",
)

TEST_ID_KEY = b"k" * 32

RAVDESS_TOPOLOGY_FIELDS = (
    "unique_archive_member_names",
    "unique_source_content_sha256s",
    "duplicate_content_groups",
    "members_beyond_unique_content",
    "max_content_multiplicity",
    "cross_actor_duplicate_content_groups",
)

RAVDESS_V2_PROVENANCE_POLICY = {
    "actor_id": "private_hmac_sha256_base32",
    "cache_integrity_id": (
        "private_hmac_trial_id_actor_id_cache_sha256_base32"
    ),
    "raw_paths_or_filenames_in_manifest": False,
    "raw_source_content_sha256_in_manifest": False,
    "source_binding": "verified_archive_member_name_and_bytes_single_read",
    "trial_id": (
        "private_hmac_archive_member_name_source_content_sha256_base32_v2"
    ),
}


def _ravdess_cache_bytes(
    *, features_dtype: np.dtype = np.dtype(np.float32), feature_width: int = 23,
) -> bytes:
    payload = io.BytesIO()
    np.savez_compressed(
        payload,
        features=np.zeros((2, feature_width), dtype=features_dtype),
        valid_mask=np.ones(2, dtype=np.bool_),
        timestamps=np.asarray([0.0, 0.033], dtype=np.float64),
        frame_indices=np.asarray([1, 2], dtype=np.int64),
        detector_confidence=np.asarray([0.99, 0.99], dtype=np.float32),
        feature_names=np.asarray(SEMANTIC23_FEATURE_NAMES),
        schema=np.asarray(SEMANTIC23_SCHEMA),
        adapter_name=np.asarray(OPENFACE68_ADAPTER_METADATA["adapter_name"]),
        scale_normalization=np.asarray(
            OPENFACE68_ADAPTER_METADATA["scale_normalization"]
        ),
        confidence_threshold=np.asarray(
            prep.DEFAULT_CONFIDENCE_THRESHOLD, dtype=np.float32
        ),
    )
    return payload.getvalue()


def _patch_first_central_size(payload: bytes, *, field_offset: int) -> bytes:
    """Change only one ZIP central-directory size declaration."""
    changed = bytearray(payload)
    central = changed.index(b"PK\x01\x02")
    changed[central + field_offset:central + field_offset + 4] = (
        0x7FFF_FFFF
    ).to_bytes(4, "little")
    return bytes(changed)


def _with_repeated_central_records_and_declared_count(
    payload: bytes, *, actual_record_count: int
) -> bytes:
    """Build central-directory bytes whose record count disagrees with the EOCD."""
    eocd = payload.rfind(b"PK\x05\x06")
    central_size = int.from_bytes(payload[eocd + 12:eocd + 16], "little")
    central_offset = int.from_bytes(payload[eocd + 16:eocd + 20], "little")
    declared_record_count = int.from_bytes(payload[eocd + 10:eocd + 12], "little")
    central = payload[central_offset:central_offset + central_size]
    first_name_size = int.from_bytes(central[28:30], "little")
    first_extra_size = int.from_bytes(central[30:32], "little")
    first_comment_size = int.from_bytes(central[32:34], "little")
    first_record_size = 46 + first_name_size + first_extra_size + first_comment_size
    first_record = central[:first_record_size]
    expanded = central + first_record * (actual_record_count - declared_record_count)
    forged_eocd = bytearray(payload[eocd:eocd + 22])
    forged_eocd[12:16] = len(expanded).to_bytes(4, "little")
    return payload[:central_offset] + expanded + bytes(forged_eocd)


def _ravdess_validate_without_materializing(payload: bytes) -> None:
    prep._validate_authorized_ravdess_cache(
        payload,
        trial_id="trial_aaaaaaaaaaaaaaaa",
        actor_id="actor_aaaaaaaaaaaaaaaa",
        cache_integrity_id="cache_aaaaaaaaaaaaaaaa",
        cache_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _face() -> np.ndarray:
    """Return a symmetric synthetic OpenFace-68 face in pixel coordinates."""
    p = np.full((68, 2), (50.0, 50.0), dtype=np.float32)
    eye_a = {
        36: (30, 40), 37: (32, 38), 38: (38, 38),
        39: (40, 40), 40: (38, 42), 41: (32, 42),
    }
    eye_b = {
        42: (60, 40), 43: (62, 38), 44: (68, 38),
        45: (70, 40), 46: (68, 42), 47: (62, 42),
    }
    for idx, xy in {**eye_a, **eye_b}.items():
        p[idx] = xy
    for idx, x in zip(range(17, 22), np.linspace(30, 40, 5)):
        p[idx] = (x, 30)
    for idx, x in zip(range(22, 27), np.linspace(60, 70, 5)):
        p[idx] = (x, 30)
    for i, idx in enumerate(OPENFACE68_MIDLINE):
        p[idx] = (50, 25 + 4 * i)
    p[48] = (40, 70)
    p[54] = (60, 70)
    p[62] = (50, 68)
    p[66] = (50, 72)
    return p


def _rotate(points: np.ndarray, angle: float) -> np.ndarray:
    out = points.copy()
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.asarray(((c, -s), (s, c)), dtype=np.float32)
    out[:] = (out - np.asarray((50, 50), np.float32)) @ rotation.T + 50
    return out


def _mirror_and_swap(points: np.ndarray) -> np.ndarray:
    """Horizontal reflection with exact OpenFace left/right topology swap."""
    out = points.copy()
    out[:, 0] = 100 - points[:, 0]
    pairs = (
        (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
        (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
        (48, 54), (49, 53), (50, 52), (55, 59), (56, 58),
        (60, 64), (61, 63), (65, 67),
    )
    for a, b in pairs:
        out[a] = (100 - points[b, 0], points[b, 1])
        out[b] = (100 - points[a, 0], points[a, 1])
    return out


def _csv_header() -> list[str]:
    return ["frame", "timestamp", "confidence", *[f"x_{i}" for i in range(68)],
            *[f"y_{i}" for i in range(68)]]


def _csv_row(frame: int, timestamp: float, confidence: float,
             points: np.ndarray) -> list[object]:
    return [frame, timestamp, confidence, *points[:, 0], *points[:, 1]]


def _write_csv(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(_csv_header())
        writer.writerows(rows)


def _synthetic_tree(
    root: Path, *, duplicate_first_frame: bool = False
) -> tuple[RavdessInventoryExpectation, list[Path]]:
    first = root / "extracted" / "01-01-01-01-01-01-01.csv"
    second = root / "extracted" / "01-01-01-01-01-01-02.csv"
    _write_csv(first, [
        _csv_row(1, 0.000, 0.99, _face()),
        _csv_row(1 if duplicate_first_frame else 2, 0.033, 0.50, _face()),
    ])
    _write_csv(second, [_csv_row(1, 0.000, 0.80, _face())])
    archive = root / RAVDESS_ARCHIVE_RELATIVE_PATH
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.write(first, arcname=first.name)
        handle.write(second, arcname=second.name)
    header_bytes = (",".join(_csv_header())).encode("utf-8")
    expected = RavdessInventoryExpectation(
        archive_size=archive.stat().st_size,
        archive_md5=hashlib.md5(archive.read_bytes()).hexdigest(),  # noqa: S324
        csv_files=2,
        actors=2,
        frames=3,
        header_sha256=hashlib.sha256(header_bytes).hexdigest(),
        empty_trials=0,
        repeated_headers=0,
        unique_archive_member_names=2,
        unique_source_content_sha256s=2,
        duplicate_content_groups=0,
        members_beyond_unique_content=0,
        max_content_multiplicity=1,
        cross_actor_duplicate_content_groups=0,
    )
    return expected, [first, second]


def _synthetic_duplicate_content_tree(
    root: Path,
) -> tuple[RavdessInventoryExpectation, tuple[str, str], bytes]:
    first = root / "extracted" / "01-01-01-01-01-01-01.csv"
    second = root / "extracted" / "01-01-01-01-01-02-01.csv"
    _write_csv(first, [
        _csv_row(1, 0.000, 0.99, _face()),
        _csv_row(2, 0.033, 0.80, _face()),
    ])
    member_bytes = first.read_bytes()
    second.write_bytes(member_bytes)
    archive = root / RAVDESS_ARCHIVE_RELATIVE_PATH
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.write(first, arcname=first.name)
        handle.write(second, arcname=second.name)
    header_bytes = (",".join(_csv_header())).encode("utf-8")
    expected = RavdessInventoryExpectation(
        archive_size=archive.stat().st_size,
        archive_md5=hashlib.md5(archive.read_bytes()).hexdigest(),  # noqa: S324
        csv_files=2,
        actors=1,
        frames=4,
        header_sha256=hashlib.sha256(header_bytes).hexdigest(),
        empty_trials=0,
        repeated_headers=0,
        unique_archive_member_names=2,
        unique_source_content_sha256s=1,
        duplicate_content_groups=1,
        members_beyond_unique_content=1,
        max_content_multiplicity=2,
        cross_actor_duplicate_content_groups=0,
    )
    return expected, (first.name, second.name), member_bytes


def _source_private_representations(
    member_name: str, source_content_sha256: str,
) -> tuple[bytes, ...]:
    name_bytes = member_name.encode("ascii")
    digest_text = source_content_sha256.encode("ascii")
    digest_bytes = bytes.fromhex(source_content_sha256)
    representations = {
        name_bytes,
        name_bytes.hex().encode("ascii"),
        name_bytes.hex().upper().encode("ascii"),
        base64.b64encode(name_bytes),
        digest_text,
        digest_text.upper(),
        digest_text.hex().encode("ascii"),
        digest_text.hex().upper().encode("ascii"),
        base64.b64encode(digest_text),
        digest_bytes,
        digest_bytes.hex().encode("ascii"),
        digest_bytes.hex().upper().encode("ascii"),
        base64.b64encode(digest_bytes),
    }
    return tuple(sorted(representations))


def _all_source_private_representations(
    member_sha256: dict[str, str],
) -> tuple[bytes, ...]:
    representations = {
        representation
        for member_name, source_digest in member_sha256.items()
        for representation in _source_private_representations(
            member_name, source_digest
        )
    }
    return tuple(sorted(representations))


def _contains_any_private_representation(
    captured: bytes, representations: tuple[bytes, ...],
) -> bool:
    return any(representation in captured for representation in representations)


def _exception_chain_bytes(error: BaseException | None) -> bytes:
    max_depth = 16
    max_nodes = 4096
    max_bytes = 4 * 1024 * 1024
    fragments: list[bytes] = []
    total_bytes = 0
    seen: set[int] = set()
    pending: list[tuple[object, int]] = [(error, 0)]

    def append_fragment(fragment: bytes) -> None:
        nonlocal total_bytes
        if not fragment or total_bytes >= max_bytes:
            return
        bounded = fragment[:max_bytes - total_bytes]
        fragments.append(bounded)
        total_bytes += len(bounded)

    def append_text(fragment: str) -> None:
        remaining = max_bytes - total_bytes
        if remaining > 0:
            append_fragment(
                fragment[:remaining].encode("utf-8", errors="replace")
            )

    visited = 0
    while pending and visited < max_nodes and total_bytes < max_bytes:
        value, depth = pending.pop()
        if value is None or depth > max_depth:
            continue
        visited += 1
        if isinstance(value, str):
            append_text(value)
            continue
        if isinstance(value, bytes):
            append_fragment(value)
            continue
        if isinstance(value, (bytearray, memoryview)):
            append_fragment(bytes(value[:max_bytes - total_bytes]))
            continue
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        if isinstance(value, BaseException):
            append_text(str(value))
            children: list[object] = [
                value.__cause__,
                value.__context__,
                value.args,
                getattr(value, "__dict__", {}),
                getattr(value, "__notes__", ()),
            ]
            if isinstance(value, json.JSONDecodeError):
                children.append(value.doc)
            if isinstance(value, UnicodeDecodeError):
                children.append(value.object)
            available = max_nodes - visited - len(pending)
            pending.extend(
                (child, depth + 1) for child in children[:max(0, available)]
            )
        elif isinstance(value, dict):
            for key, item in value.items():
                if len(pending) + visited + 2 > max_nodes:
                    break
                pending.append((key, depth + 1))
                pending.append((item, depth + 1))
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                if len(pending) + visited + 1 > max_nodes:
                    break
                pending.append((item, depth + 1))
    return b"\n".join(fragments)


def _capture_process_output(callback):
    stdout = io.StringIO()
    stderr = io.StringIO()
    log_output = io.StringIO()
    log_handler = logging.StreamHandler(log_output)
    root_logger = logging.getLogger()
    original_root_level = root_logger.level
    original_handler_level = log_handler.level
    root_logger.setLevel(logging.NOTSET)
    log_handler.setLevel(logging.NOTSET)
    root_logger.addHandler(log_handler)
    with tempfile.TemporaryFile() as fd_stdout, tempfile.TemporaryFile() as fd_stderr:
        original_fd1 = os.dup(1)
        original_fd2 = os.dup(2)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(fd_stdout.fileno(), 1)
            os.dup2(fd_stderr.fileno(), 2)
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = callback()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(original_fd1, 1)
            os.dup2(original_fd2, 2)
            os.close(original_fd1)
            os.close(original_fd2)
            root_logger.removeHandler(log_handler)
            log_handler.setLevel(original_handler_level)
            root_logger.setLevel(original_root_level)
        fd_stdout.seek(0)
        fd1_output = fd_stdout.read()
        fd_stderr.seek(0)
        fd2_output = fd_stderr.read()
    captured = (
        stdout.getvalue().encode("utf-8")
        + stderr.getvalue().encode("utf-8")
        + fd1_output
        + fd2_output
        + log_output.getvalue().encode("utf-8")
    )
    return result, captured


def _capture_failure(callback) -> tuple[BaseException | None, bytes]:
    observed: BaseException | None = None

    def invoke() -> None:
        nonlocal observed
        try:
            callback()
        except BaseException as exc:  # noqa: BLE001 - inspect full rejection
            observed = exc

    _, captured = _capture_process_output(invoke)
    return observed, captured


def _assert_deidentified_failure(
    c: Check, callback, sentinel: bytes, label: str,
) -> None:
    observed, captured = _capture_failure(callback)
    c.true(isinstance(observed, ValueError), f"{label} fails closed")
    c.true(
        sentinel not in _exception_chain_bytes(observed) + captured,
        f"{label} exception graph and output are deidentified",
    )


def _generation_artifact_blobs(output: Path) -> tuple[bytes, ...]:
    blobs: list[bytes] = []
    for path in sorted(output.rglob("*")):
        blobs.append(path.name.encode("utf-8"))
        if not path.is_file():
            continue
        payload = path.read_bytes()
        blobs.append(payload)
        if path.suffix == ".npz":
            with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
                blobs.append(archive.comment)
                for info in archive.infolist():
                    blobs.extend((
                        info.filename.encode("utf-8"),
                        info.comment,
                        info.extra,
                        archive.read(info),
                    ))
    return tuple(blobs)


def _rewrite_cache_with_feature_prefix(
    parent_descriptor: int,
    cache_name: str,
    prefix: bytes,
) -> str:
    """Fault-inject bytes into a schema-valid staged cache feature payload."""
    descriptor = os.open(
        cache_name,
        os.O_RDWR | os.O_NOFOLLOW,
        dir_fd=parent_descriptor,
    )
    try:
        with os.fdopen(descriptor, "r+b", closefd=False) as handle:
            original = handle.read()
            with np.load(io.BytesIO(original), allow_pickle=False) as cached:
                arrays = {
                    name: np.array(cached[name], copy=True)
                    for name in cached.files
                }
            feature_bytes = arrays["features"].view(np.uint8).reshape(-1)
            if len(prefix) > feature_bytes.size:
                raise AssertionError("privacy sentinel does not fit cache fixture")
            feature_bytes[:len(prefix)] = np.frombuffer(prefix, dtype=np.uint8)
            payload = io.BytesIO()
            np.savez_compressed(payload, **arrays)
            rewritten = payload.getvalue()
            handle.seek(0)
            handle.truncate(0)
            handle.write(rewritten)
            handle.flush()
            os.fsync(descriptor)
        return hashlib.sha256(rewritten).hexdigest()
    finally:
        os.close(descriptor)


def _rewrite_first_local_zip_name(
    payload: bytes, private_name: bytes,
) -> bytes:
    """Keep the central name but fault-inject a different first local name."""
    rewritten = bytearray(payload)
    eocd = len(rewritten) - 22
    if rewritten[eocd:eocd + 4] != b"PK\x05\x06":
        raise AssertionError("cache fixture has a noncanonical EOCD")
    central = int.from_bytes(rewritten[eocd + 16:eocd + 20], "little")
    first_local = int.from_bytes(
        rewritten[central + 42:central + 46], "little"
    )
    if rewritten[first_local:first_local + 4] != b"PK\x03\x04":
        raise AssertionError("cache fixture has a noncanonical local header")
    old_length = int.from_bytes(
        rewritten[first_local + 26:first_local + 28], "little"
    )
    name_start = first_local + 30
    delta = len(private_name) - old_length
    rewritten[name_start:name_start + old_length] = private_name
    rewritten[first_local + 26:first_local + 28] = len(private_name).to_bytes(
        2, "little"
    )
    new_eocd = eocd + delta
    new_central = central + delta
    rewritten[new_eocd + 16:new_eocd + 20] = new_central.to_bytes(
        4, "little"
    )
    cursor = new_central
    while cursor < new_eocd:
        if rewritten[cursor:cursor + 4] != b"PK\x01\x02":
            raise AssertionError("cache fixture has a noncanonical central record")
        old_offset = int.from_bytes(
            rewritten[cursor + 42:cursor + 46], "little"
        )
        if old_offset > first_local:
            rewritten[cursor + 42:cursor + 46] = (
                old_offset + delta
            ).to_bytes(4, "little")
        name_length = int.from_bytes(
            rewritten[cursor + 28:cursor + 30], "little"
        )
        extra_length = int.from_bytes(
            rewritten[cursor + 30:cursor + 32], "little"
        )
        comment_length = int.from_bytes(
            rewritten[cursor + 32:cursor + 34], "little"
        )
        cursor += 46 + name_length + extra_length + comment_length
    if cursor != new_eocd:
        raise AssertionError("cache fixture central directory does not close")
    return bytes(rewritten)


def _rewrite_cache_zip_surface(
    payload: bytes, surface: str, sentinel: bytes,
) -> bytes:
    if surface == "local_name":
        return _rewrite_first_local_zip_name(payload, sentinel)
    if surface in {"central_name", "central_extra"}:
        rewritten = bytearray(payload)
        eocd = len(rewritten) - 22
        central = int.from_bytes(
            rewritten[eocd + 16:eocd + 20], "little"
        )
        central_size = int.from_bytes(
            rewritten[eocd + 12:eocd + 16], "little"
        )
        name_length = int.from_bytes(
            rewritten[central + 28:central + 30], "little"
        )
        extra_length = int.from_bytes(
            rewritten[central + 30:central + 32], "little"
        )
        if surface == "central_name":
            start = central + 46
            replacement = sentinel
            rewritten[central + 28:central + 30] = len(replacement).to_bytes(
                2, "little"
            )
            old_length = name_length
        else:
            start = central + 46 + name_length
            replacement = (
                b"\xfe\xca" + len(sentinel).to_bytes(2, "little") + sentinel
            )
            rewritten[central + 30:central + 32] = len(replacement).to_bytes(
                2, "little"
            )
            old_length = extra_length
        rewritten[start:start + old_length] = replacement
        delta = len(replacement) - old_length
        new_eocd = eocd + delta
        rewritten[new_eocd + 12:new_eocd + 16] = (
            central_size + delta
        ).to_bytes(4, "little")
        return bytes(rewritten)
    if surface == "local_extra":
        rewritten = bytearray(payload)
        eocd = len(rewritten) - 22
        central = int.from_bytes(
            rewritten[eocd + 16:eocd + 20], "little"
        )
        first_local = int.from_bytes(
            rewritten[central + 42:central + 46], "little"
        )
        name_length = int.from_bytes(
            rewritten[first_local + 26:first_local + 28], "little"
        )
        extra_length = int.from_bytes(
            rewritten[first_local + 28:first_local + 30], "little"
        )
        replacement = (
            b"\xfe\xca" + len(sentinel).to_bytes(2, "little") + sentinel
        )
        start = first_local + 30 + name_length
        rewritten[start:start + extra_length] = replacement
        rewritten[first_local + 28:first_local + 30] = len(replacement).to_bytes(
            2, "little"
        )
        delta = len(replacement) - extra_length
        new_eocd = eocd + delta
        new_central = central + delta
        rewritten[new_eocd + 16:new_eocd + 20] = new_central.to_bytes(
            4, "little"
        )
        cursor = new_central
        while cursor < new_eocd:
            old_offset = int.from_bytes(
                rewritten[cursor + 42:cursor + 46], "little"
            )
            if old_offset > first_local:
                rewritten[cursor + 42:cursor + 46] = (
                    old_offset + delta
                ).to_bytes(4, "little")
            current_name = int.from_bytes(
                rewritten[cursor + 28:cursor + 30], "little"
            )
            current_extra = int.from_bytes(
                rewritten[cursor + 30:cursor + 32], "little"
            )
            current_comment = int.from_bytes(
                rewritten[cursor + 32:cursor + 34], "little"
            )
            cursor += 46 + current_name + current_extra + current_comment
        return bytes(rewritten)
    if surface == "archive_comment":
        rewritten = bytearray(payload)
        eocd = len(rewritten) - 22
        rewritten[eocd + 20:eocd + 22] = len(sentinel).to_bytes(2, "little")
        rewritten.extend(sentinel)
        return bytes(rewritten)
    with zipfile.ZipFile(io.BytesIO(payload), "r") as source:
        members = [
            (info, source.read(info)) for info in source.infolist()
        ]
    rewritten = io.BytesIO()
    with zipfile.ZipFile(
        rewritten, "w", compression=zipfile.ZIP_DEFLATED
    ) as destination:
        for index, (info, member_payload) in enumerate(members):
            name = info.filename
            if index == 0 and surface == "both_names":
                name = sentinel.decode("ascii")
            changed = zipfile.ZipInfo(name)
            changed.compress_type = zipfile.ZIP_DEFLATED
            changed.external_attr = info.external_attr
            changed.create_system = info.create_system
            if index == 0 and surface == "member_comment":
                changed.comment = sentinel
            destination.writestr(changed, member_payload)
        if surface in {"directory_name", "directory_payload"}:
            directory_name = "private-carrier/"
            directory_payload = sentinel
            if surface == "directory_name":
                directory_name = sentinel.decode("ascii") + "/"
                directory_payload = b""
            directory = zipfile.ZipInfo(directory_name)
            directory.compress_type = zipfile.ZIP_DEFLATED
            directory.external_attr = (stat.S_IFDIR | 0o700) << 16
            destination.writestr(directory, directory_payload)
    return rewritten.getvalue()


def _rewrite_first_npy_header(payload: bytes, sentinel: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(payload), "r") as source:
        members = [
            (info.filename, source.read(info)) for info in source.infolist()
        ]
    header = b'"' + sentinel + b'"\n'
    invalid_npy = (
        b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header
    )
    rewritten = io.BytesIO()
    with zipfile.ZipFile(
        rewritten, "w", compression=zipfile.ZIP_DEFLATED
    ) as destination:
        for index, (name, member_payload) in enumerate(members):
            destination.writestr(
                name, invalid_npy if index == 0 else member_payload
            )
    return rewritten.getvalue()


def _pad_first_npy_header(payload: bytes, padding_bytes: int = 64) -> bytes:
    with zipfile.ZipFile(io.BytesIO(payload), "r") as source:
        members = [
            (info.filename, source.read(info)) for info in source.infolist()
        ]
    first_name, first_payload = members[0]
    if first_payload[:8] != b"\x93NUMPY\x01\x00":
        raise AssertionError("cache fixture does not use NPY v1")
    header_length = int.from_bytes(first_payload[8:10], "little")
    header = first_payload[10:10 + header_length]
    if not header.endswith(b"\n"):
        raise AssertionError("cache fixture NPY header has no newline")
    padded_header = header[:-1] + b" " * padding_bytes + b"\n"
    padded_first = (
        first_payload[:8]
        + len(padded_header).to_bytes(2, "little")
        + padded_header
        + first_payload[10 + header_length:]
    )
    rewritten = io.BytesIO()
    with zipfile.ZipFile(
        rewritten, "w", compression=zipfile.ZIP_DEFLATED
    ) as destination:
        for name, member_payload in members:
            destination.writestr(
                name, padded_first if name == first_name else member_payload
            )
    return rewritten.getvalue()


def _zip_member_bytes(archive: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(archive, "r") as handle:
        return {name: handle.read(name) for name in sorted(handle.namelist())}


def _write_zip_bytes(destination: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for name, payload in sorted(members.items()):
            handle.writestr(name, payload)


def _transient_zipfile_path_swap(
    archive_path: Path, replacement_path: Path
):
    """Return a ZipFile wrapper that swaps only the archive's lexical path."""
    original_zipfile = prep.zipfile.ZipFile
    parked_original = archive_path.with_name(".parked-original.zip")

    def wrapped(file, *args, **kwargs):
        wrapped.opened_arguments.append(file)
        os.replace(archive_path, parked_original)
        os.replace(replacement_path, archive_path)
        try:
            opened = original_zipfile(file, *args, **kwargs)
        finally:
            os.replace(archive_path, replacement_path)
            os.replace(parked_original, archive_path)
        return opened

    wrapped.opened_arguments = []
    return original_zipfile, wrapped


def _assert_lock_reacquirable(c: Check, lock_path: Path, message: str) -> None:
    descriptor = os.open(lock_path, os.O_RDWR | os.O_NOFOLLOW)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)
    c.eq(stat.S_IMODE(lock_path.stat().st_mode), 0o600, message)


def test_semantic23_schema_is_exact_and_self_describing(c: Check):
    c.eq(SEMANTIC23_SCHEMA, "semantic23_v1", "generic schema version")
    c.eq(SEMANTIC23_FEATURE_NAMES, EXPECTED_NAMES, "source-neutral feature order")
    c.eq(len(SEMANTIC23_DEFINITIONS), 23, "one definition per feature")
    c.eq(tuple(item.name for item in SEMANTIC23_DEFINITIONS), EXPECTED_NAMES,
         "definition order matches vector order")
    for item in SEMANTIC23_DEFINITIONS:
        c.true(item.unit in {"interocular_distance", "interocular_distance_squared"},
               f"explicit unit for {item.name}")
        c.true(bool(item.definition), f"definition for {item.name}")
        c.true(item.sign in {"nonnegative", "side_a_minus_side_b", "eye_line_relative"},
               f"sign convention for {item.name}")


def test_clinical23_adapter_is_explicit_not_length_inference(c: Check):
    source = np.arange(23, dtype=np.float32)
    got = clinical23_v2_to_semantic23(source)
    c.true(bool(np.array_equal(got, source)), "clinical V2 reorder is currently identity")
    c.true(got is not source, "adapter returns an owned array")
    c.eq(CLINICAL23_V2_ADAPTER_METADATA["source_schema"], "clinical23_v2",
         "source schema is explicit")
    c.eq(CLINICAL23_V2_ADAPTER_METADATA["target_schema"], SEMANTIC23_SCHEMA,
         "target schema is explicit")
    c.eq(CLINICAL23_V2_ADAPTER_METADATA["eye_measure"], "height_times_width",
         "eye measure is not polygon area")
    c.raises(lambda: clinical23_v2_to_semantic23(np.zeros(22, np.float32)),
             ValueError, "same contract cannot be inferred from arbitrary length")
    bad = source.copy()
    bad[3] = np.nan
    c.raises(lambda: clinical23_v2_to_semantic23(bad), ValueError,
             "non-finite source vector fails closed")


def test_openface_topology_schema_eye_measure_and_dtype(c: Check):
    c.eq(OPENFACE68_SIDE_A_EYE_RING, (36, 37, 38, 39, 40, 41),
         "exact OpenFace side-A eye ring")
    c.eq(OPENFACE68_SIDE_B_EYE_RING, (42, 43, 44, 45, 46, 47),
         "exact OpenFace side-B eye ring")
    c.eq(OPENFACE68_SIDE_A_UPPER, (37, 38), "exact side-A upper lid")
    c.eq(OPENFACE68_SIDE_A_LOWER, (40, 41), "exact side-A lower lid")
    c.eq(OPENFACE68_SIDE_B_UPPER, (43, 44), "exact side-B upper lid")
    c.eq(OPENFACE68_SIDE_B_LOWER, (46, 47), "exact side-B lower lid")
    c.eq(OPENFACE68_SIDE_A_BROW, (17, 18, 19, 20, 21), "exact side-A brow")
    c.eq(OPENFACE68_SIDE_B_BROW, (22, 23, 24, 25, 26), "exact side-B brow")
    c.eq((OPENFACE68_SIDE_A_CORNER, OPENFACE68_SIDE_B_CORNER), (48, 54),
         "exact commissures")
    c.eq((OPENFACE68_MOUTH_TOP, OPENFACE68_MOUTH_BOTTOM), (62, 66),
         "exact inner central lip points")
    c.eq(OPENFACE68_REQUIRED_INDICES, tuple(sorted(set(OPENFACE68_REQUIRED_INDICES))),
         "required topology is exact, sorted and unique")
    c.true(all(0 <= i < 68 for i in OPENFACE68_REQUIRED_INDICES),
           "topology uses exact 0-based OpenFace-68 indices")
    vector = openface68_to_semantic23(_face())
    by_name = dict(zip(SEMANTIC23_FEATURE_NAMES, vector))
    c.eq(vector.shape, (23,), "vector shape")
    c.eq(vector.dtype, np.float32, "vector dtype")
    c.true(bool(np.isfinite(vector).all()), "finite vector")
    for side in ("a", "b"):
        want = by_name[f"fissure_h_side_{side}"] * by_name[f"fissure_w_side_{side}"]
        c.true(abs(float(by_name[f"eye_measure_side_{side}"] - want)) < 1e-7,
               f"side {side} eye measure equals height times width")
    c.eq(OPENFACE68_ADAPTER_METADATA["source_topology"], "openface_68_2d",
         "source topology metadata")
    c.eq(OPENFACE68_ADAPTER_METADATA["scale_normalization"], "interocular_distance",
         "source scaling metadata")


def test_openface_translation_scale_and_roll_invariance(c: Check):
    base = _face()
    reference = openface68_to_semantic23(base)
    translated = base + np.asarray((17.5, -9.25), np.float32)
    scaled = base * np.float32(3.7)
    rolled = _rotate(base, 0.31)
    for name, transformed in (("translation", translated), ("scale", scaled),
                              ("roll", rolled)):
        c.true(bool(np.allclose(reference, openface68_to_semantic23(transformed),
                                atol=2e-5)), f"{name} invariant")


def test_openface_symmetry_signed_and_absolute_perturbations(c: Check):
    symmetric = dict(zip(SEMANTIC23_FEATURE_NAMES,
                         openface68_to_semantic23(_face())))
    for name in ("fissure_h_absdiff", "fissure_h_side_a_minus_side_b",
                 "fissure_w_absdiff", "eye_measure_absdiff", "brow_h_absdiff",
                 "brow_h_side_a_minus_side_b", "corner_y_absdiff",
                 "corner_y_side_a_minus_side_b", "corner_x_absdiff"):
        c.true(abs(float(symmetric[name])) < 1e-6, f"symmetric {name} is zero")

    changed = _face()
    changed[54, 1] += 4
    original = dict(zip(SEMANTIC23_FEATURE_NAMES,
                        openface68_to_semantic23(changed)))
    mirrored = dict(zip(SEMANTIC23_FEATURE_NAMES,
                        openface68_to_semantic23(_mirror_and_swap(changed))))
    c.true(float(original["corner_y_absdiff"]) > 0, "absolute asymmetry grows")
    c.true(float(original["corner_y_side_a_minus_side_b"]) < 0,
           "side-A minus side-B sign is retained")
    for name in ("fissure_h_absdiff", "fissure_w_absdiff", "eye_measure_absdiff",
                 "brow_h_absdiff", "corner_y_absdiff", "corner_x_absdiff"):
        c.true(abs(float(original[name] - mirrored[name])) < 2e-5,
               f"mirror keeps absolute feature {name}")
    for name in ("fissure_h_side_a_minus_side_b", "brow_h_side_a_minus_side_b",
                 "corner_y_side_a_minus_side_b"):
        c.true(abs(float(original[name] + mirrored[name])) < 2e-5,
               f"mirror changes sign for {name}")


def test_openface_malformed_geometry_fails_closed(c: Check):
    c.raises(lambda: openface68_to_semantic23(np.zeros((67, 2), np.float32)),
             ValueError, "requires exact 68-point topology")
    c.raises(lambda: openface68_to_semantic23(np.zeros((68, 3), np.float32)),
             ValueError, "requires exact 2D coordinates")
    bad = _face()
    bad[36, 0] = np.nan
    c.raises(lambda: openface68_to_semantic23(bad), ValueError,
             "non-finite required coordinate")
    unused_nan = _face()
    unused_nan[0, 0] = np.nan
    c.true(bool(np.isfinite(openface68_to_semantic23(unused_nan)).all()),
           "only coordinates required by the semantic adapter are mandatory")
    degenerate = _face()
    degenerate[42:48] = degenerate[36:42]
    c.raises(lambda: openface68_to_semantic23(degenerate), ValueError,
             "degenerate interocular distance")


def test_csv_parser_keeps_timestamps_and_detector_gaps(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "01-01-01-01-01-01-01.csv"
        invalid_geometry = _face()
        invalid_geometry[36, 0] = np.nan
        _write_csv(path, [
            _csv_row(10, 1.000, 0.80, _face()),
            _csv_row(11, 1.033, 0.79, _face()),
            _csv_row(12, 1.066, 0.99, invalid_geometry),
            _csv_row(13, 1.099, 0.95, _face()),
        ])
        trial = parse_openface_csv(path)
    c.eq(trial.features.shape, (4, 23), "no frame is dropped")
    c.true(bool(np.array_equal(trial.frame_indices, [10, 11, 12, 13])),
           "source frame indices preserved")
    c.true(bool(np.allclose(trial.timestamps, [1.000, 1.033, 1.066, 1.099])),
           "source timestamps preserved")
    c.true(bool(np.array_equal(trial.valid_mask, [True, False, False, True])),
           "threshold is inclusive and malformed geometry remains a gap")
    c.true(bool(np.all(trial.features[~trial.valid_mask] == 0)),
           "masked gaps use neutral storage, never interpolation")

    for confidence in (-0.01, 1.01):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "01-01-01-01-01-01-01.csv"
            _write_csv(path, [_csv_row(1, 0.0, confidence, _face())])
            c.raises(lambda: parse_openface_csv(path), ValueError,
                     "detector confidence outside [0, 1] fails closed")


def test_opaque_provenance_is_stable_and_does_not_expose_names(c: Check):
    actor = opaque_actor_id("07", key=TEST_ID_KEY)
    same_actor = opaque_actor_id("07", key=TEST_ID_KEY)
    other_actor = opaque_actor_id("08", key=TEST_ID_KEY)
    other_key_actor = opaque_actor_id("07", key=b"z" * 32)
    trial = opaque_trial_id(
        "01-01-01-01-01-01-01.csv", "0" * 64, key=TEST_ID_KEY
    )
    c.eq(actor, same_actor, "actor ID stable")
    c.true(actor != other_actor, "actors remain distinguishable")
    c.true(actor != other_key_actor, "private HMAC key prevents public enumeration")
    c.true(actor.startswith("actor_") and len(actor) == 22, "bounded opaque actor ID")
    c.true(trial.startswith("trial_") and len(trial) == 22, "bounded opaque trial ID")
    c.true("07" not in actor and "source" not in trial, "raw provenance not exposed")
    c.raises(lambda: opaque_actor_id("07", key=b"short"), ValueError,
             "HMAC pseudonyms require a private 256-bit key")

    with tempfile.TemporaryDirectory() as temporary:
        key_path = Path(temporary) / "private-id.key"
        first = prep.load_or_create_private_id_key(key_path)
        second = prep.load_or_create_private_id_key(key_path)
        c.eq(first, second, "private ID key is stable across runs")
        c.eq(len(first), 32, "private ID key has 256 bits")
        mode = stat.S_IMODE(key_path.stat().st_mode)
        c.eq(mode, 0o600, "private ID key is owner-only")


def test_opaque_trial_id_v2_is_strict_and_binds_name_content_and_key(c: Check):
    member_name = "01-01-01-01-01-01-01.csv"
    source_content_sha256 = "0" * 64
    expected = "trial_o457alx6gmxoxyak"
    c.eq(
        opaque_trial_id(
            member_name, source_content_sha256, key=TEST_ID_KEY
        ),
        expected,
        "frozen RAVDESS v2 trial-ID known-answer vector",
    )
    c.eq(
        opaque_trial_id(
            member_name, source_content_sha256, key=TEST_ID_KEY
        ),
        expected,
        "the exact v2 source binding is stable",
    )
    c.true(
        opaque_trial_id(
            "01-01-01-01-01-02-01.csv",
            source_content_sha256,
            key=TEST_ID_KEY,
        ) != expected,
        "the exact archive member name participates in trial identity",
    )
    c.true(
        opaque_trial_id(member_name, "1" * 64, key=TEST_ID_KEY) != expected,
        "the source-content digest participates in trial identity",
    )
    c.true(
        opaque_trial_id(
            member_name, source_content_sha256, key=b"z" * 32
        ) != expected,
        "trial identity is private-key scoped",
    )

    invalid_names = (
        Path(member_name),
        f"nested/{member_name}",
        member_name.replace("01", "０１", 1),
        "1-01-01-01-01-01-01.csv",
        "01-01-01-01-01-01-01.CSV",
    )
    for invalid_name in invalid_names:
        c.raises(
            lambda invalid_name=invalid_name: opaque_trial_id(
                invalid_name, source_content_sha256, key=TEST_ID_KEY
            ),
            ValueError,
            f"noncanonical archive member name is rejected: {invalid_name!r}",
        )

    invalid_digests = (
        "A" * 64,
        "0" * 63,
        "0" * 65,
        b"0" * 64,
        "g" * 64,
    )
    for invalid_digest in invalid_digests:
        c.raises(
            lambda invalid_digest=invalid_digest: opaque_trial_id(
                member_name, invalid_digest, key=TEST_ID_KEY
            ),
            ValueError,
            f"noncanonical source-content digest is rejected: {invalid_digest!r}",
        )

    invalid_keys = (bytearray(TEST_ID_KEY), "k" * 32, b"k" * 31, b"k" * 33)
    for invalid_key in invalid_keys:
        c.raises(
            lambda invalid_key=invalid_key: opaque_trial_id(
                member_name, source_content_sha256, key=invalid_key
            ),
            ValueError,
            "trial-ID key must be type-exact canonical 256-bit bytes",
        )


def test_manifest_guard_rejects_reversible_source_identity_encodings(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "ravdess"
        expected, files = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        member_name = files[1].name
        source_digest = inventory.member_sha256[member_name]
        checked = 0
        for representation in _source_private_representations(
            member_name, source_digest
        ):
            try:
                encoded = representation.decode("ascii")
            except UnicodeDecodeError:
                continue
            checked += 1
            c.raises(
                lambda encoded=encoded: prep._assert_manifest_deidentified(
                    encoded,
                    source_root=data_root,
                    source_paths=[Path(member_name)],
                    raw_source_sha256s={source_digest},
                    raw_cache_sha256s=set(),
                ),
                ValueError,
                "manifest guard rejects every reversible ASCII source identity",
            )
        c.eq(checked, 9, "all unique reversible ASCII manifest forms are exercised")


def test_authorizer_rejects_valid_json_private_keys_and_values(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "ravdess"
        expected, files = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        key_path = data_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        key_path.write_bytes(TEST_ID_KEY)
        key_path.chmod(0o600)
        output = data_root / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root,
            output,
            inventory,
            expectation=expected,
            id_key=TEST_ID_KEY,
        )
        manifest_path = output / "manifest.json"
        original_manifest = manifest_path.read_bytes()
        representations = _all_source_private_representations(
            inventory.member_sha256
        )
        observations: list[tuple[str, str, str | None, bool]] = []
        original_expectation = prep.FROZEN_RAVDESS_INVENTORY
        prep.FROZEN_RAVDESS_INVENTORY = expected
        try:
            for representation in representations:
                try:
                    encoded = representation.decode("ascii")
                except UnicodeDecodeError:
                    continue
                for surface in ("key", "value"):
                    manifest = json.loads(original_manifest.decode("utf-8"))
                    if surface == "key":
                        manifest[encoded] = "public-placeholder"
                    else:
                        manifest["license"] = encoded
                    manifest_path.write_text(
                        json.dumps(manifest, sort_keys=True), encoding="utf-8"
                    )
                    manifest_path.chmod(0o600)
                    observed, process_output = _capture_failure(
                        lambda: prep.authorize_committed_ravdess_semantic23(
                            data_root
                        )
                    )
                    captured = _exception_chain_bytes(observed) + process_output
                    observations.append((
                        surface,
                        type(observed).__name__ if observed else "none",
                        str(observed) if observed else None,
                        _contains_any_private_representation(
                            captured, representations
                        ),
                    ))
        finally:
            manifest_path.write_bytes(original_manifest)
            manifest_path.chmod(0o600)
            prep.FROZEN_RAVDESS_INVENTORY = original_expectation
        c.true(len(observations) >= 18, "all applicable ASCII representations run")
        c.true(
            all(
                kind == ValueError.__name__
                and message == (
                    "aggregate manifest contains raw or reversibly encoded provenance"
                )
                and not leaked
                for _, kind, message, leaked in observations
            ),
            "valid JSON key/value carriers fail at the privacy gate without leakage",
        )


def test_authorizer_rejects_schema_shaped_unicode_npy_private_value(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "ravdess"
        expected, files = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        key_path = data_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        key_path.write_bytes(TEST_ID_KEY)
        key_path.chmod(0o600)
        output = data_root / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root,
            output,
            inventory,
            expectation=expected,
            id_key=TEST_ID_KEY,
        )
        manifest_path = output / "manifest.json"
        original_manifest = manifest_path.read_bytes()
        manifest_template = json.loads(original_manifest.decode("utf-8"))
        row_template = manifest_template["trials"][0]
        cache_path = output / "trials" / f"{row_template['trial_id']}.npz"
        original_cache = cache_path.read_bytes()
        with np.load(cache_path, allow_pickle=False) as cached:
            original_arrays = {
                name: np.array(cached[name], copy=True)
                for name in cached.files
            }
        representations = _all_source_private_representations(
            inventory.member_sha256
        )
        observations: list[tuple[bytes, str, str | None, bool]] = []
        original_expectation = prep.FROZEN_RAVDESS_INVENTORY
        prep.FROZEN_RAVDESS_INVENTORY = expected
        try:
            for representation in representations:
                try:
                    text = representation.decode("ascii")
                except UnicodeDecodeError:
                    continue
                arrays = {
                    name: np.array(value, copy=True)
                    for name, value in original_arrays.items()
                }
                values = arrays["feature_names"]
                width = values.dtype.itemsize // 4
                c.true(
                    len(text) <= width * values.size,
                    "ASCII private representation fits schema-shaped Unicode array",
                )
                values[:] = ""
                for index, start in enumerate(range(0, len(text), width)):
                    values[index] = text[start:start + width]
                payload = io.BytesIO()
                np.savez_compressed(payload, **arrays)
                rewritten = payload.getvalue()
                with zipfile.ZipFile(io.BytesIO(rewritten), "r") as archive:
                    expanded_feature_names = archive.read("feature_names.npy")
                c.true(
                    text.encode("utf-32-le") in expanded_feature_names,
                    "Unicode carrier contains the exact private representation",
                )
                cache_path.write_bytes(rewritten)
                cache_path.chmod(0o600)
                manifest = json.loads(original_manifest.decode("utf-8"))
                row = manifest["trials"][0]
                cache_sha256 = hashlib.sha256(rewritten).hexdigest()
                row["cache_integrity_id"] = prep._opaque_cache_integrity_id(
                    cache_sha256,
                    trial_id=row["trial_id"],
                    actor_id=row["actor_id"],
                    key=TEST_ID_KEY,
                )
                manifest_path.write_text(
                    json.dumps(manifest, sort_keys=True), encoding="utf-8"
                )
                manifest_path.chmod(0o600)
                observed, process_output = _capture_failure(
                    lambda: prep.authorize_committed_ravdess_semantic23(
                        data_root
                    )
                )
                captured = _exception_chain_bytes(observed) + process_output
                observations.append((
                    representation,
                    type(observed).__name__ if observed else "none",
                    str(observed) if observed else None,
                    _contains_any_private_representation(
                        captured, representations
                    ),
                ))
        finally:
            cache_path.write_bytes(original_cache)
            cache_path.chmod(0o600)
            manifest_path.write_bytes(original_manifest)
            manifest_path.chmod(0o600)
            prep.FROZEN_RAVDESS_INVENTORY = original_expectation
        c.true(
            len(observations) >= 9,
            "all ASCII-decodable private representations use Unicode carriers",
        )
        c.true(
            all(
                kind == ValueError.__name__
                and message == "RAVDESS cache contains private provenance"
                and not leaked
                for _, kind, message, leaked in observations
            ),
            "all schema-shaped Unicode carriers hit privacy gate without leakage",
        )


def test_authorizer_zip_private_surface_matrix_is_deidentified(c: Check):
    byte_surfaces = (
        "local_extra",
        "central_extra",
        "member_comment",
        "archive_comment",
        "directory_payload",
    )
    ascii_surfaces = (
        "local_name",
        "central_name",
        "both_names",
        "directory_name",
    )
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "ravdess"
        expected, files = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        key_path = data_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        key_path.write_bytes(TEST_ID_KEY)
        key_path.chmod(0o600)
        output = data_root / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root,
            output,
            inventory,
            expectation=expected,
            id_key=TEST_ID_KEY,
        )
        manifest_path = output / "manifest.json"
        original_manifest = manifest_path.read_bytes()
        manifest_template = json.loads(original_manifest.decode("utf-8"))
        row_template = manifest_template["trials"][0]
        cache_path = output / "trials" / f"{row_template['trial_id']}.npz"
        original_cache = cache_path.read_bytes()
        representations = _all_source_private_representations(
            inventory.member_sha256
        )
        ascii_representation_count = sum(
            1
            for representation in representations
            if all(byte < 128 for byte in representation)
        )
        observations: list[tuple[str, bytes, str, bool]] = []
        original_expectation = prep.FROZEN_RAVDESS_INVENTORY
        prep.FROZEN_RAVDESS_INVENTORY = expected
        try:
            for representation in representations:
                surfaces = list(byte_surfaces)
                try:
                    representation.decode("ascii")
                except UnicodeDecodeError:
                    pass
                else:
                    surfaces.extend(ascii_surfaces)
                for surface in surfaces:
                    manifest = json.loads(original_manifest.decode("utf-8"))
                    row = manifest["trials"][0]
                    rewritten = _rewrite_cache_zip_surface(
                        original_cache, surface, representation
                    )
                    cache_path.write_bytes(rewritten)
                    cache_path.chmod(0o600)
                    cache_sha256 = hashlib.sha256(rewritten).hexdigest()
                    row["cache_integrity_id"] = prep._opaque_cache_integrity_id(
                        cache_sha256,
                        trial_id=row["trial_id"],
                        actor_id=row["actor_id"],
                        key=TEST_ID_KEY,
                    )
                    manifest_path.write_text(
                        json.dumps(manifest, sort_keys=True), encoding="utf-8"
                    )
                    manifest_path.chmod(0o600)
                    observed, process_output = _capture_failure(
                        lambda: prep.authorize_committed_ravdess_semantic23(
                            data_root
                        )
                    )
                    captured = _exception_chain_bytes(observed) + process_output
                    observations.append((
                        surface,
                        representation,
                        type(observed).__name__ if observed else "none",
                        _contains_any_private_representation(
                            captured, representations
                        ),
                    ))
        finally:
            cache_path.write_bytes(original_cache)
            cache_path.chmod(0o600)
            manifest_path.write_bytes(original_manifest)
            manifest_path.chmod(0o600)
            prep.FROZEN_RAVDESS_INVENTORY = original_expectation
        c.eq(
            len(observations),
            len(representations) * len(byte_surfaces)
            + ascii_representation_count * len(ascii_surfaces),
            "every applicable representation and ZIP surface is exercised",
        )
        c.true(
            all(kind == ValueError.__name__ and not leaked
                for _, _, kind, leaked in observations),
            "all ZIP metadata/name/directory carriers reject without leakage",
        )


def test_authorizer_npy_header_private_matrix_is_deidentified(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "ravdess"
        expected, files = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        key_path = data_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        key_path.write_bytes(TEST_ID_KEY)
        key_path.chmod(0o600)
        output = data_root / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root,
            output,
            inventory,
            expectation=expected,
            id_key=TEST_ID_KEY,
        )
        manifest_path = output / "manifest.json"
        original_manifest = manifest_path.read_bytes()
        manifest_template = json.loads(original_manifest.decode("utf-8"))
        row_template = manifest_template["trials"][0]
        cache_path = output / "trials" / f"{row_template['trial_id']}.npz"
        original_cache = cache_path.read_bytes()
        representations = _all_source_private_representations(
            inventory.member_sha256
        )
        observations: list[tuple[bytes, str, bool]] = []
        original_expectation = prep.FROZEN_RAVDESS_INVENTORY
        prep.FROZEN_RAVDESS_INVENTORY = expected
        try:
            for representation in representations:
                rewritten = _rewrite_first_npy_header(
                    original_cache, representation
                )
                cache_path.write_bytes(rewritten)
                cache_path.chmod(0o600)
                manifest = json.loads(original_manifest.decode("utf-8"))
                row = manifest["trials"][0]
                cache_sha256 = hashlib.sha256(rewritten).hexdigest()
                row["cache_integrity_id"] = prep._opaque_cache_integrity_id(
                    cache_sha256,
                    trial_id=row["trial_id"],
                    actor_id=row["actor_id"],
                    key=TEST_ID_KEY,
                )
                manifest_path.write_text(
                    json.dumps(manifest, sort_keys=True), encoding="utf-8"
                )
                manifest_path.chmod(0o600)
                observed, process_output = _capture_failure(
                    lambda: prep.authorize_committed_ravdess_semantic23(
                        data_root
                    )
                )
                captured = _exception_chain_bytes(observed) + process_output
                observations.append((
                    representation,
                    type(observed).__name__ if observed else "none",
                    _contains_any_private_representation(
                        captured, representations
                    ),
                ))
        finally:
            cache_path.write_bytes(original_cache)
            cache_path.chmod(0o600)
            manifest_path.write_bytes(original_manifest)
            manifest_path.chmod(0o600)
            prep.FROZEN_RAVDESS_INVENTORY = original_expectation
        c.eq(
            len(observations),
            len(representations),
            "every private representation is injected into an NPY header",
        )
        c.true(
            all(kind == ValueError.__name__ and not leaked
                for _, kind, leaked in observations),
            "all NPY header carriers reject without exception/output leakage",
        )


def test_private_pattern_index_is_exact_chunked_and_blob_bounded(c: Check):
    pattern = b"private-provenance-pattern-across-chunk-boundary"
    index = prep._compile_private_pattern_index({pattern})
    chunk_size = prep._PRIVATE_PATTERN_SCAN_CHUNK_BYTES
    near_boundary = b"x" * (chunk_size - 5) + pattern + b"tail"
    c.true(
        prep._contains_indexed_private_pattern((near_boundary,), index),
        "pattern starting before a chunk boundary is found through overlap",
    )
    at_boundary = b"x" * chunk_size + pattern
    c.true(
        prep._contains_indexed_private_pattern((at_boundary,), index),
        "pattern starting at a chunk boundary is found",
    )
    split = len(pattern) // 2
    c.true(
        not prep._contains_indexed_private_pattern(
            (pattern[:split], pattern[split:]), index
        ),
        "a representation split across distinct artifact blobs is not invented",
    )
    c.true(
        not prep._contains_indexed_private_pattern((b"public-only",), index),
        "nonmatching public bytes remain accepted",
    )


def test_private_local_zip_name_rejection_has_no_diagnostic_leak(c: Check):
    member_name = "01-01-01-01-01-01-01.csv"
    source_digest = hashlib.sha256(b"synthetic private source").hexdigest()
    representations = _source_private_representations(
        member_name, source_digest
    )
    payload = _rewrite_first_local_zip_name(
        _ravdess_cache_bytes(), member_name.encode("ascii")
    )
    source_index = prep._source_private_pattern_index(
        {member_name: source_digest}
    )
    cache_sha256 = hashlib.sha256(payload).hexdigest()
    callbacks = (
        lambda: prep._assert_ravdess_cache_deidentified(
            payload,
            cache_name="trial_aaaaaaaaaaaaaaaa.npz",
            source_pattern_index=source_index,
            raw_cache_sha256=cache_sha256,
        ),
        lambda: prep._require_ravdess_npz_headers(payload),
    )
    for callback in callbacks:
        observed, process_output = _capture_failure(callback)
        c.true(
            isinstance(observed, ValueError),
            "private local ZIP header name is rejected",
        )
        captured = _exception_chain_bytes(observed) + process_output
        for representation in representations:
            c.true(
                representation not in captured,
                "ZIP/NPY rejection diagnostics contain no private representation",
            )


def test_exception_graph_scan_is_bounded_and_follows_both_branches(c: Check):
    cause_sentinel = b"private-cause-branch-sentinel"
    context_sentinel = b"private-context-branch-sentinel"
    cause = ValueError(({"nested": [cause_sentinel]},))
    context = ValueError("public context")
    context.private_state = {"nested": [context_sentinel]}
    cyclic: list[object] = []
    cyclic.append(cyclic)
    context.cyclic = cyclic
    root = RuntimeError("public root")
    note_sentinel = b"private-exception-note-sentinel"
    root.__notes__ = [note_sentinel.decode("ascii")]
    root.__cause__ = cause
    root.__context__ = context
    captured = _exception_chain_bytes(root)
    c.true(cause_sentinel in captured, "cause args are recursively scanned")
    c.true(context_sentinel in captured, "context __dict__ is recursively scanned")
    c.true(note_sentinel in captured, "exception notes are recursively scanned")
    c.true(len(captured) <= 4 * 1024 * 1024, "exception scanning is byte bounded")


def test_private_oracle_detects_cross_identity_leakage(c: Check):
    member_sha256 = {
        "01-01-01-01-01-01-01.csv": hashlib.sha256(b"first").hexdigest(),
        "01-01-01-01-01-01-02.csv": hashlib.sha256(b"second").hexdigest(),
    }
    all_private = _all_source_private_representations(member_sha256)
    second_identity = _source_private_representations(
        "01-01-01-01-01-01-02.csv",
        member_sha256["01-01-01-01-01-01-02.csv"],
    )[0]
    c.true(
        _contains_any_private_representation(
            b"public failure\n" + second_identity, all_private
        ),
        "oracle detects a non-current source identity in captured output",
    )
    c.true(
        not _contains_any_private_representation(b"public only", all_private),
        "oracle does not invent a private leak",
    )


def test_process_capture_includes_python_fds_and_all_log_levels(c: Check):
    sentinels = {
        "stdout": b"capture-python-stdout",
        "stderr": b"capture-python-stderr",
        "fd1": b"capture-native-fd1",
        "fd2": b"capture-native-fd2",
        "debug": b"capture-log-debug",
        "info": b"capture-log-info",
        "warning": b"capture-log-warning",
    }

    def emit() -> None:
        print(sentinels["stdout"].decode("ascii"))
        print(sentinels["stderr"].decode("ascii"), file=sys.stderr)
        os.write(1, sentinels["fd1"] + b"\n")
        os.write(2, sentinels["fd2"] + b"\n")
        logging.getLogger().debug(sentinels["debug"].decode("ascii"))
        logging.getLogger().info(sentinels["info"].decode("ascii"))
        logging.getLogger().warning(sentinels["warning"].decode("ascii"))

    _, captured = _capture_process_output(emit)
    for channel, sentinel in sentinels.items():
        c.true(sentinel in captured, f"capture includes {channel}")


def test_malformed_json_has_no_private_exception_state(c: Check):
    sentinel = b"private-json-document-sentinel"
    payload = b'{"value":"' + sentinel + b'",}'
    _assert_deidentified_failure(
        c,
        lambda: prep._load_unique_json_object(payload, "RAVDESS manifest"),
        sentinel,
        "malformed JSON",
    )


def test_invalid_utf8_json_has_no_private_exception_state(c: Check):
    sentinel = b"private-utf8-document-sentinel"
    payload = b'{"value":"' + sentinel + b'\xff"}'
    _assert_deidentified_failure(
        c,
        lambda: prep._load_unique_json_object(payload, "RAVDESS manifest"),
        sentinel,
        "invalid UTF-8 JSON",
    )


def test_npy_header_failure_has_no_private_exception_state(c: Check):
    npy_sentinel = b"private_npy_header_sentinel"
    header = b'"' + npy_sentinel + b'"\n'
    member_bytes = (
        b"\x93NUMPY\x01\x00"
        + len(header).to_bytes(2, "little")
        + header
    )
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("features.npy", member_bytes)

    def parse_npy_header() -> None:
        with zipfile.ZipFile(io.BytesIO(payload.getvalue()), "r") as archive:
            prep._npy_header(
                archive,
                archive.infolist()[0],
                field="RAVDESS cache features",
            )

    _assert_deidentified_failure(
        c, parse_npy_header, npy_sentinel, "invalid bounded NPY header"
    )


def test_csv_frame_failure_has_no_private_exception_state(c: Check):
    csv_sentinel = b"private-frame-metadata-sentinel"
    row = _csv_row(1, 0.0, 0.99, _face())
    row[0] = csv_sentinel.decode("ascii")
    source = io.StringIO()
    writer = csv.writer(source, lineterminator="\n")
    writer.writerow(_csv_header())
    writer.writerow(row)
    _assert_deidentified_failure(
        c,
        lambda: prep.parse_openface_csv_bytes(
            source.getvalue().encode("utf-8"), source_name="synthetic.csv"
        ),
        csv_sentinel,
        "invalid OpenFace frame metadata",
    )


def test_invalid_utf8_csv_has_no_private_exception_state(c: Check):
    sentinel = b"private-lazy-utf8-csv-sentinel"
    payload = (
        (",".join(_csv_header()) + "\n").encode("utf-8")
        + sentinel
        + b"\xff\n"
    )
    _assert_deidentified_failure(
        c,
        lambda: prep.parse_openface_csv_bytes(
            payload, source_name="synthetic.csv"
        ),
        sentinel,
        "invalid UTF-8 OpenFace CSV",
    )


def test_generation_parser_failure_has_no_private_exception_state(c: Check):
    csv_sentinel = b"private-generation-parser-sentinel"
    original_parser = prep.parse_openface_csv_bytes

    def private_parser_failure(*args, **kwargs):
        raise ValueError({"private": [csv_sentinel]})

    prep.parse_openface_csv_bytes = private_parser_failure
    try:
        _assert_deidentified_failure(
            c,
            lambda: prep._parse_ravdess_member_csv(
                b"synthetic", source_name="synthetic.csv"
            ),
            csv_sentinel,
            "RAVDESS generation parser boundary",
        )
    finally:
        prep.parse_openface_csv_bytes = original_parser


def test_zip_member_read_failure_has_no_private_exception_state(c: Check):
    zip_sentinel = b"private-ravdess-member-name.csv"

    class FailingArchive:
        def read(self, member):
            raise zipfile.BadZipFile(
                "corrupt RAVDESS member " + member.filename
            )

    _assert_deidentified_failure(
        c,
        lambda: prep._read_ravdess_member_bytes(
            FailingArchive(), zipfile.ZipInfo(zip_sentinel.decode("ascii"))
        ),
        zip_sentinel,
        "RAVDESS ZIP member read",
    )


def test_duplicate_content_members_remain_distinct_v2_trials_end_to_end(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "ravdess"
        expected, member_names, member_bytes = _synthetic_duplicate_content_tree(
            data_root
        )
        shared_digest = hashlib.sha256(member_bytes).hexdigest()
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        c.eq(
            {name: getattr(inventory, name) for name in RAVDESS_TOPOLOGY_FIELDS},
            {
                "unique_archive_member_names": 2,
                "unique_source_content_sha256s": 1,
                "duplicate_content_groups": 1,
                "members_beyond_unique_content": 1,
                "max_content_multiplicity": 2,
                "cross_actor_duplicate_content_groups": 0,
            },
            "same-actor duplicate bytes retain the frozen member/content topology",
        )

        key_path = data_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        key_path.write_bytes(TEST_ID_KEY)
        key_path.chmod(0o600)
        output = data_root / "derived_semantic23"
        _, build_output = _capture_process_output(
            lambda: build_generation_from_audited_sources(
                data_root,
                output,
                inventory,
                expectation=expected,
                id_key=TEST_ID_KEY,
            )
        )
        manifest_bytes = (output / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes.decode("ascii"))
        c.true(
            type(manifest["format_version"]) is int,
            "RAVDESS manifest version rejects the bool/int alias",
        )
        c.eq(manifest["format_version"], 2, "RAVDESS manifest v2")
        c.eq(
            manifest["provenance_policy"],
            RAVDESS_V2_PROVENANCE_POLICY,
            "RAVDESS v2 provenance policy is exact",
        )
        expected_inventory = {
            "archive_size_bytes": expected.archive_size,
            "archive_md5": expected.archive_md5,
            "csv_trials": expected.csv_files,
            "actors": expected.actors,
            "source_frames": expected.frames,
            "header_sha256": expected.header_sha256,
            "empty_trials": expected.empty_trials,
            "repeated_headers": expected.repeated_headers,
            **{name: getattr(expected, name) for name in RAVDESS_TOPOLOGY_FIELDS},
        }
        c.eq(
            manifest["inventory"], expected_inventory,
            "manifest inventory has exactly the eight aggregates and six topology fields",
        )
        for name, value in expected_inventory.items():
            if name not in {"archive_md5", "header_sha256"}:
                c.true(type(value) is int, f"inventory integer is type-exact: {name}")

        rows = manifest["trials"]
        expected_trial_ids = {
            opaque_trial_id(name, shared_digest, key=TEST_ID_KEY)
            for name in member_names
        }
        c.eq(
            {row["trial_id"] for row in rows},
            expected_trial_ids,
            "identical bytes under distinct exact member names remain distinct trials",
        )
        c.eq(len({row["cache_integrity_id"] for row in rows}), 2,
             "duplicate-content trials retain distinct cache integrity IDs")
        c.eq(len({row["actor_id"] for row in rows}), 1,
             "same-actor duplicate content retains one actor group")
        cache_paths = sorted((output / "trials").glob("*.npz"))
        c.eq(
            {path.name for path in cache_paths},
            {f"{trial_id}.npz" for trial_id in expected_trial_ids},
            "one exact opaque cache filename is emitted per archive member",
        )

        original_expectation = prep.FROZEN_RAVDESS_INVENTORY
        prep.FROZEN_RAVDESS_INVENTORY = expected
        try:
            authorized, authorizer_output = _capture_process_output(
                lambda: prep.authorize_committed_ravdess_semantic23(data_root)
            )
        finally:
            prep.FROZEN_RAVDESS_INVENTORY = original_expectation
        c.eq(authorized.trial_count, 2, "committed authorizer retains both trials")
        c.eq(authorized.actor_count, 1, "committed authorizer retains one actor")

        public_blobs = _generation_artifact_blobs(output)
        private_representations = {
            representation
            for member_name in member_names
            for representation in _source_private_representations(
                member_name, shared_digest
            )
        }
        for representation in private_representations:
            c.true(
                all(representation not in blob for blob in public_blobs),
                "raw and reversibly encoded source identity never persists",
            )
            c.true(
                representation not in build_output + authorizer_output,
                "successful build, authorization, stdout, stderr, and logs stay clean",
            )


def test_generation_rejects_private_source_representations_in_staged_cache(
    c: Check,
):
    with tempfile.TemporaryDirectory() as fixture_root:
        fixture_data_root = Path(fixture_root) / "ravdess"
        expected, files = _synthetic_tree(fixture_data_root)
        inventory = audit_ravdess_inventory(
            fixture_data_root, expectation=expected
        )
        member_name = files[1].name
        source_digest = inventory.member_sha256[member_name]
        representations = _source_private_representations(
            member_name, source_digest
        )

    for representation in representations:
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "ravdess"
            expected, _ = _synthetic_tree(data_root)
            inventory = audit_ravdess_inventory(
                data_root, expectation=expected
            )
            output = data_root / "derived_semantic23"
            original_write_cache = prep._write_cache_at
            injected = False

            def injecting_write_cache(
                parent_descriptor: int,
                cache_name: str,
                trial: prep.SemanticTrial,
            ) -> str:
                nonlocal injected
                cache_sha256 = original_write_cache(
                    parent_descriptor, cache_name, trial
                )
                if injected:
                    return cache_sha256
                injected = True
                return _rewrite_cache_with_feature_prefix(
                    parent_descriptor, cache_name, representation
                )

            stdout = io.StringIO()
            stderr = io.StringIO()
            observed: BaseException | None = None
            prep._write_cache_at = injecting_write_cache
            try:
                try:
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                        stderr
                    ):
                        build_generation_from_audited_sources(
                            data_root,
                            output,
                            inventory,
                            expectation=expected,
                            id_key=TEST_ID_KEY,
                        )
                except BaseException as exc:  # noqa: BLE001 - inspect rejection
                    observed = exc
            finally:
                prep._write_cache_at = original_write_cache

            c.true(injected, "privacy representation is injected into staged NPZ")
            c.true(
                isinstance(observed, RuntimeError),
                "privacy-contaminated staged cache fails before publication",
            )
            c.true(
                not output.exists(),
                "privacy-contaminated cache is never canonical",
            )
            c.eq(
                len(list(data_root.glob(f".{output.name}.staging-*"))),
                1,
                "failed private staging remains as indeterminate evidence",
            )
            captured = (
                _exception_chain_bytes(observed)
                + stdout.getvalue().encode("utf-8")
                + stderr.getvalue().encode("utf-8")
            )
            for private_value in representations:
                c.true(
                    private_value not in captured,
                    "cache-privacy rejection and captured output stay deidentified",
                )


def test_generator_rejects_post_write_private_manifest_injection(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "ravdess"
        expected, files = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        member_name = files[0].name
        source_digest = inventory.member_sha256[member_name]
        foreign_member_name = files[1].name
        foreign_source_digest = inventory.member_sha256[foreign_member_name]
        sentinel = base64.b64encode(bytes.fromhex(foreign_source_digest))
        output = data_root / "derived_semantic23"
        original_write_bytes = prep._write_bytes_at
        injected = False

        def injecting_write_bytes(
            parent_descriptor: int, name: str, payload: bytes,
        ) -> None:
            nonlocal injected
            original_write_bytes(parent_descriptor, name, payload)
            if name != "manifest.json":
                return
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW,
                dir_fd=parent_descriptor,
            )
            try:
                os.write(descriptor, b"privacy_probe=" + sentinel)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            injected = True

        stdout = io.StringIO()
        stderr = io.StringIO()
        observed: BaseException | None = None
        prep._write_bytes_at = injecting_write_bytes
        try:
            try:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                    stderr
                ):
                    build_generation_from_audited_sources(
                        data_root,
                        output,
                        inventory,
                        expectation=expected,
                        id_key=TEST_ID_KEY,
                    )
            except BaseException as exc:  # noqa: BLE001 - inspect rejection
                observed = exc
        finally:
            prep._write_bytes_at = original_write_bytes

        c.true(injected, "private sentinel is injected after manifest write")
        c.true(
            isinstance(observed, RuntimeError),
            "post-write manifest mutation fails before publication",
        )
        c.true(not output.exists(), "mutated staged manifest is never canonical")
        c.eq(
            len(list(data_root.glob(f".{output.name}.staging-*"))),
            1,
            "mutated manifest stage remains as indeterminate evidence",
        )
        captured = (
            _exception_chain_bytes(observed)
            + stdout.getvalue().encode("utf-8")
            + stderr.getvalue().encode("utf-8")
        )
        private_representations = {
            *_source_private_representations(member_name, source_digest),
            *_source_private_representations(
                foreign_member_name, foreign_source_digest
            ),
        }
        for private_value in private_representations:
            c.true(
                private_value not in captured,
                "manifest mutation rejection and output stay deidentified",
            )


def test_generator_rejects_private_extra_stage_artifact(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "ravdess"
        expected, files = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        member_name = files[0].name
        source_digest = inventory.member_sha256[member_name]
        sentinel = base64.b64encode(bytes.fromhex(source_digest))
        output = data_root / "derived_semantic23"
        original_write_bytes = prep._write_bytes_at
        injected = False

        def injecting_extra_artifact(
            parent_descriptor: int, name: str, payload: bytes,
        ) -> None:
            nonlocal injected
            original_write_bytes(parent_descriptor, name, payload)
            if name != "manifest.json":
                return
            descriptor = os.open(
                member_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent_descriptor,
            )
            try:
                os.fchmod(descriptor, 0o600)
                os.write(descriptor, sentinel)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            injected = True

        stdout = io.StringIO()
        stderr = io.StringIO()
        observed: BaseException | None = None
        prep._write_bytes_at = injecting_extra_artifact
        try:
            try:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                    stderr
                ):
                    build_generation_from_audited_sources(
                        data_root,
                        output,
                        inventory,
                        expectation=expected,
                        id_key=TEST_ID_KEY,
                    )
            except BaseException as exc:  # noqa: BLE001 - inspect rejection
                observed = exc
        finally:
            prep._write_bytes_at = original_write_bytes

        c.true(injected, "private extra artifact is injected into stage root")
        c.true(
            isinstance(observed, RuntimeError),
            "unexpected private stage artifact fails before publication",
        )
        c.true(not output.exists(), "stage-root extra artifact is never canonical")
        c.eq(
            len(list(data_root.glob(f".{output.name}.staging-*"))),
            1,
            "stage-root fault remains as private indeterminate evidence",
        )
        captured = (
            _exception_chain_bytes(observed)
            + stdout.getvalue().encode("utf-8")
            + stderr.getvalue().encode("utf-8")
        )
        for private_value in _source_private_representations(
            member_name, source_digest
        ):
            c.true(
                private_value not in captured,
                "extra-artifact rejection and output stay deidentified",
            )


def test_authorizer_rejects_coordinated_private_cache_and_manifest_forgery(
    c: Check,
):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "ravdess"
        expected, files = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        key_path = data_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        key_path.write_bytes(TEST_ID_KEY)
        key_path.chmod(0o600)
        output = data_root / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root,
            output,
            inventory,
            expectation=expected,
            id_key=TEST_ID_KEY,
        )

        member_name = files[0].name
        source_digest = inventory.member_sha256[member_name]
        foreign_member_name = files[1].name
        foreign_source_digest = inventory.member_sha256[foreign_member_name]
        sentinel = base64.b64encode(bytes.fromhex(foreign_source_digest))
        trial_id = opaque_trial_id(
            member_name, source_digest, key=TEST_ID_KEY
        )
        trials_root = output / "trials"
        trials_descriptor = os.open(
            trials_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        try:
            cache_sha256 = _rewrite_cache_with_feature_prefix(
                trials_descriptor, f"{trial_id}.npz", sentinel
            )
        finally:
            os.close(trials_descriptor)

        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text("ascii"))
        row = next(
            item for item in manifest["trials"]
            if item["trial_id"] == trial_id
        )
        row["cache_integrity_id"] = prep._opaque_cache_integrity_id(
            cache_sha256,
            trial_id=trial_id,
            actor_id=row["actor_id"],
            key=TEST_ID_KEY,
        )
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")),
            encoding="ascii",
        )
        manifest_path.chmod(0o600)
        forged_manifest = manifest_path.read_bytes()
        forged_cache = (trials_root / f"{trial_id}.npz").read_bytes()

        stdout = io.StringIO()
        stderr = io.StringIO()
        observed: BaseException | None = None
        original_expectation = prep.FROZEN_RAVDESS_INVENTORY
        prep.FROZEN_RAVDESS_INVENTORY = expected
        try:
            try:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                    stderr
                ):
                    prep.authorize_committed_ravdess_semantic23(data_root)
            except BaseException as exc:  # noqa: BLE001 - inspect rejection
                observed = exc
        finally:
            prep.FROZEN_RAVDESS_INVENTORY = original_expectation

        c.true(
            isinstance(observed, ValueError),
            "coordinated privacy-bearing cache/HMAC forgery is rejected",
        )
        c.eq(
            manifest_path.read_bytes(),
            forged_manifest,
            "read-only authorizer never mutates forged manifest evidence",
        )
        c.eq(
            (trials_root / f"{trial_id}.npz").read_bytes(),
            forged_cache,
            "read-only authorizer never mutates forged cache evidence",
        )
        captured = (
            _exception_chain_bytes(observed)
            + stdout.getvalue().encode("utf-8")
            + stderr.getvalue().encode("utf-8")
        )
        private_representations = {
            *_source_private_representations(member_name, source_digest),
            *_source_private_representations(
                foreign_member_name, foreign_source_digest
            ),
        }
        for private_value in private_representations:
            c.true(
                private_value not in captured,
                "coordinated forgery rejection and output stay deidentified",
            )


def test_committed_authorizer_rejects_alternate_trial_identity_constructions(
    c: Check,
):
    cases = (
        "coherent_v1",
        "content_only_ids",
        "wrong_v2_serialization",
        "wrong_v2_prefix",
        "wrong_v2_policy",
    )
    for case in cases:
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "ravdess"
            expected, _ = _synthetic_tree(data_root)
            inventory = audit_ravdess_inventory(data_root, expectation=expected)
            key_path = data_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
            key_path.write_bytes(TEST_ID_KEY)
            key_path.chmod(0o600)
            output = data_root / "derived_semantic23"
            build_generation_from_audited_sources(
                data_root,
                output,
                inventory,
                expectation=expected,
                id_key=TEST_ID_KEY,
            )
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text("ascii"))

            def alternate_trial_id(member_name: str, digest: str) -> str:
                if case in {"coherent_v1", "content_only_ids"}:
                    return prep._opaque_id(
                        "trial", digest, "trial", key=TEST_ID_KEY
                    )
                binding_object = {
                    "archive_member_name": member_name,
                    "source_content_sha256": digest,
                }
                if case == "wrong_v2_serialization":
                    binding = json.dumps(
                        binding_object,
                        sort_keys=True,
                        ensure_ascii=True,
                        allow_nan=False,
                    ).encode("ascii")
                    prefix = b"ravdess-semantic23-trial-id-v2\0"
                else:
                    binding = json.dumps(
                        binding_object,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        allow_nan=False,
                    ).encode("ascii")
                    prefix = b"ravdess-semantic23-trial-id-v2:"
                digest_bytes = hmac.new(
                    TEST_ID_KEY,
                    prefix + binding,
                    hashlib.sha256,
                ).digest()
                token = base64.b32encode(digest_bytes).decode("ascii")
                return "trial_" + token.lower().rstrip("=")[:16]

            if case != "wrong_v2_policy":
                alternate_by_v2 = {
                    opaque_trial_id(name, digest, key=TEST_ID_KEY): (
                        alternate_trial_id(name, digest)
                    )
                    for name, digest in inventory.member_sha256.items()
                }
                for row in manifest["trials"]:
                    old_trial_id = row["trial_id"]
                    alternate_id = alternate_by_v2[old_trial_id]
                    old_cache = output / "trials" / f"{old_trial_id}.npz"
                    new_cache = output / "trials" / f"{alternate_id}.npz"
                    old_cache.rename(new_cache)
                    cache_sha256 = hashlib.sha256(
                        new_cache.read_bytes()
                    ).hexdigest()
                    row["trial_id"] = alternate_id
                    row["cache_integrity_id"] = prep._opaque_cache_integrity_id(
                        cache_sha256,
                        trial_id=alternate_id,
                        actor_id=row["actor_id"],
                        key=TEST_ID_KEY,
                    )
                manifest["trials"].sort(key=lambda row: row["trial_id"])

            if case == "coherent_v1":
                manifest["format_version"] = 1
                manifest["provenance_policy"] = {
                    "actor_id": "private_hmac_sha256_base32",
                    "trial_id": "private_hmac_source_content_sha256_base32",
                    "cache_integrity_id": (
                        "private_hmac_trial_id_actor_id_cache_sha256_base32"
                    ),
                    "source_binding": "verified_archive_member_bytes_single_read",
                    "raw_paths_or_filenames_in_manifest": False,
                }
                for field in RAVDESS_TOPOLOGY_FIELDS:
                    manifest["inventory"].pop(field)
            elif case == "wrong_v2_policy":
                manifest["provenance_policy"]["trial_id"] = (
                    "private_hmac_wrong_serialization_base32_v2"
                )
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")),
                encoding="ascii",
            )
            manifest_path.chmod(0o600)

            original_expectation = prep.FROZEN_RAVDESS_INVENTORY
            prep.FROZEN_RAVDESS_INVENTORY = expected
            try:
                observed, process_output = _capture_failure(
                    lambda: prep.authorize_committed_ravdess_semantic23(data_root)
                )
            finally:
                prep.FROZEN_RAVDESS_INVENTORY = original_expectation
            c.true(
                isinstance(observed, ValueError),
                f"{case} is rejected by committed v2 authorization",
            )
            failure_bytes = _exception_chain_bytes(observed) + process_output
            for member_name, digest in inventory.member_sha256.items():
                for representation in _source_private_representations(
                    member_name, digest
                ):
                    c.true(
                        representation not in failure_bytes,
                        f"{case} rejection remains deidentified",
                    )


def test_v2_trial_id_collision_is_rejected_before_staging_or_cache_open(c: Check):
    collision_id = "trial_aaaaaaaaaaaaaaaa"

    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "ravdess"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        output = data_root / "derived_semantic23"
        expected_helper_inputs = [
            (name, digest, TEST_ID_KEY)
            for name, digest in inventory.member_sha256.items()
        ]
        helper_inputs: list[tuple[str, str, bytes]] = []
        create_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        cache_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        original_trial_id = prep.opaque_trial_id
        original_create = prep._create_directory_at
        original_write_cache = prep._write_cache_at

        def colliding_trial_id(
            member_name: str, source_digest: str, *, key: bytes,
        ) -> str:
            helper_inputs.append((member_name, source_digest, key))
            return collision_id

        def tracked_create(*args, **kwargs):
            create_calls.append((args, kwargs))
            return original_create(*args, **kwargs)

        def tracked_write_cache(*args, **kwargs):
            cache_calls.append((args, kwargs))
            return original_write_cache(*args, **kwargs)

        observed: BaseException | None = None
        prep.opaque_trial_id = colliding_trial_id
        prep._create_directory_at = tracked_create
        prep._write_cache_at = tracked_write_cache
        try:
            try:
                build_generation_from_audited_sources(
                    data_root,
                    output,
                    inventory,
                    expectation=expected,
                    id_key=TEST_ID_KEY,
                )
            except BaseException as exc:  # noqa: BLE001 - inspect exact failure
                observed = exc
        finally:
            prep.opaque_trial_id = original_trial_id
            prep._create_directory_at = original_create
            prep._write_cache_at = original_write_cache

        raw_bindings = tuple(inventory.member_sha256.items())
        collision_residues = list(
            data_root.glob(f".{output.name}.staging-*")
        )
        collision_observation = {
            "exception": (
                type(observed).__name__ if observed is not None else None,
                str(observed) if observed is not None else None,
            ),
            "generic_message": observed is not None and not any(
                raw in str(observed)
                for member_name, digest in raw_bindings
                for raw in (member_name, digest)
            ),
            "all_distinct_bindings_evaluated": (
                helper_inputs == expected_helper_inputs
                and len({item[:2] for item in helper_inputs}) == 2
            ),
            "staging_create_calls": len(create_calls),
            "cache_open_calls": len(cache_calls),
            "canonical_exists": output.exists(),
            "staging_residue_count": len(collision_residues),
        }

    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "ravdess"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        output = data_root / "derived_semantic23"
        residue = data_root / f".{output.name}.staging-existing"
        residue.mkdir(mode=0o700)
        marker = residue / "marker"
        marker.write_bytes(b"owner-only retained transaction evidence")
        marker.chmod(0o600)

        def residue_snapshot() -> tuple[tuple[object, ...], ...]:
            snapshot: list[tuple[object, ...]] = []
            for path in (residue, *sorted(residue.rglob("*"))):
                info = path.lstat()
                snapshot.append((
                    path.relative_to(residue),
                    stat.S_IMODE(info.st_mode),
                    info.st_uid,
                    info.st_gid,
                    info.st_nlink,
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                    path.read_bytes() if path.is_file() else None,
                ))
            return tuple(snapshot)

        residue_before = residue_snapshot()
        helper_inputs: list[tuple[str, str, bytes]] = []
        create_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        cache_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        original_trial_id = prep.opaque_trial_id
        original_create = prep._create_directory_at
        original_write_cache = prep._write_cache_at

        def tracked_trial_id(
            member_name: str, source_digest: str, *, key: bytes,
        ) -> str:
            helper_inputs.append((member_name, source_digest, key))
            return collision_id

        def tracked_create(*args, **kwargs):
            create_calls.append((args, kwargs))
            return original_create(*args, **kwargs)

        def tracked_write_cache(*args, **kwargs):
            cache_calls.append((args, kwargs))
            return original_write_cache(*args, **kwargs)

        observed: BaseException | None = None
        prep.opaque_trial_id = tracked_trial_id
        prep._create_directory_at = tracked_create
        prep._write_cache_at = tracked_write_cache
        try:
            try:
                build_generation_from_audited_sources(
                    data_root,
                    output,
                    inventory,
                    expectation=expected,
                    id_key=TEST_ID_KEY,
                )
            except BaseException as exc:  # noqa: BLE001 - inspect exact failure
                observed = exc
        finally:
            prep.opaque_trial_id = original_trial_id
            prep._create_directory_at = original_create
            prep._write_cache_at = original_write_cache

        retry_residues = list(data_root.glob(f".{output.name}.staging-*"))
        retry_observation = {
            "exception": (
                type(observed).__name__ if observed is not None else None,
                str(observed) if observed is not None else None,
            ),
            "residue_owner_only": (
                stat.S_IMODE(residue.stat().st_mode) == 0o700
                and stat.S_IMODE(marker.stat().st_mode) == 0o600
            ),
            "residue_unchanged": residue_snapshot() == residue_before,
            "helper_calls": len(helper_inputs),
            "staging_create_calls": len(create_calls),
            "cache_open_calls": len(cache_calls),
            "canonical_exists": output.exists(),
            "only_existing_residue": retry_residues == [residue],
        }

    c.eq(
        {
            "collision": collision_observation,
            "unresolved_retry": retry_observation,
        },
        {
            "collision": {
                "exception": (ValueError.__name__, "opaque trial ID collision detected"),
                "generic_message": True,
                "all_distinct_bindings_evaluated": True,
                "staging_create_calls": 0,
                "cache_open_calls": 0,
                "canonical_exists": False,
                "staging_residue_count": 0,
            },
            "unresolved_retry": {
                "exception": (
                    RuntimeError.__name__,
                    "RAVDESS authorization rejects unresolved transaction state",
                ),
                "residue_owner_only": True,
                "residue_unchanged": True,
                "helper_calls": 0,
                "staging_create_calls": 0,
                "cache_open_calls": 0,
                "canonical_exists": False,
                "only_existing_residue": True,
            },
        },
        "trial-ID collisions fail before staging/cache while unresolved residue blocks retry",
    )


def test_inventory_cli_emits_exact_v2_topology_json(c: Check):
    frozen_inventory = prep.RavdessInventory(
        archive_size=417_163_019,
        archive_md5="5753bbc64a9a790f8a8d3e03cba526ee",
        csv_files=2_452,
        actors=24,
        frames=299_854,
        header_sha256=(
            "d89e2164e4c4e8d60393f88365ef0e87a10bef227dc90dc1d431117a74991b4e"
        ),
        empty_trials=0,
        repeated_headers=0,
        unique_archive_member_names=2_452,
        unique_source_content_sha256s=2_451,
        duplicate_content_groups=1,
        members_beyond_unique_content=1,
        max_content_multiplicity=2,
        cross_actor_duplicate_content_groups=0,
        archive_device=1,
        archive_inode=2,
        archive_mtime_ns=3,
        archive_ctime_ns=4,
        member_sha256={},
    )
    expected_stdout = {
        "status": "audit_ok",
        "archive_size_bytes": 417_163_019,
        "archive_md5": "5753bbc64a9a790f8a8d3e03cba526ee",
        "csv_trials": 2_452,
        "actors": 24,
        "source_frames": 299_854,
        "header_sha256": (
            "d89e2164e4c4e8d60393f88365ef0e87a10bef227dc90dc1d431117a74991b4e"
        ),
        "empty_trials": 0,
        "repeated_headers": 0,
        "unique_archive_member_names": 2_452,
        "unique_source_content_sha256s": 2_451,
        "duplicate_content_groups": 1,
        "members_beyond_unique_content": 1,
        "max_content_multiplicity": 2,
        "cross_actor_duplicate_content_groups": 0,
    }
    original_argv = sys.argv
    original_audit = prep.audit_ravdess_inventory
    sys.argv = ["prepare_ravdess_semantic23.py", "--data-root", "/unused"]
    prep.audit_ravdess_inventory = lambda *_args, **_kwargs: frozen_inventory
    try:
        result, captured = _capture_process_output(prep.main)
    finally:
        prep.audit_ravdess_inventory = original_audit
        sys.argv = original_argv
    c.eq(result, 0, "read-only inventory CLI succeeds")
    c.eq(
        captured,
        (json.dumps(expected_stdout, sort_keys=True) + "\n").encode("utf-8"),
        "CLI emits the literal frozen 14-field inventory plus status",
    )


def test_cli_subprocess_stderr_never_echoes_private_input(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        sentinels = (
            "private-cli-stderr-sentinel",
            "private-cli-unknown-argument-sentinel",
        )
        missing_root = Path(temporary) / sentinels[0]
        invocations = (
            ["--data-root", str(missing_root)],
            [
                "--data-root", str(missing_root),
                str(Path(temporary) / sentinels[1]),
            ],
        )
        for arguments in invocations:
            completed = subprocess.run(
                [
                    "/Users/williamqiu/opt/anaconda3/bin/python3",
                    str(ROOT / "scripts" / "prepare_ravdess_semantic23.py"),
                    *arguments,
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            captured = completed.stdout + completed.stderr
            c.true(completed.returncode != 0, "invalid CLI input fails closed")
            c.true(
                all(sentinel.encode("utf-8") not in captured for sentinel in sentinels),
                "real subprocess fd stdout/stderr never echo private input",
            )
            c.true(b"Traceback" not in captured, "CLI failure emits no traceback")


def test_archive_audit_uses_one_nofollow_fd_across_transient_path_swap(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "source"
        expected, _ = _synthetic_tree(data_root)
        archive_path = data_root / RAVDESS_ARCHIVE_RELATIVE_PATH
        original_members = _zip_member_bytes(archive_path)
        replacement_members = dict(original_members)
        first_name = sorted(replacement_members)[0]
        lines = replacement_members[first_name].splitlines(keepends=True)
        replacement_members[first_name] = lines[0] + lines[0] + b"".join(lines[1:])
        replacement_path = archive_path.with_name("transient-replacement.zip")
        _write_zip_bytes(replacement_path, replacement_members)

        archive_opens: list[int] = []
        original_open = prep.os.open
        original_zipfile, wrapped_zipfile = _transient_zipfile_path_swap(
            archive_path, replacement_path
        )

        def tracked_open(path, flags, *args, **kwargs):
            if Path(path).resolve() == archive_path.resolve():
                archive_opens.append(flags)
            return original_open(path, flags, *args, **kwargs)

        prep.os.open = tracked_open
        prep.zipfile.ZipFile = wrapped_zipfile
        try:
            c.raises(lambda: audit_ravdess_inventory(
                data_root, expectation=expected
            ), ValueError, "transient archive path replacement fails closed")
        finally:
            prep.zipfile.ZipFile = original_zipfile
            prep.os.open = original_open

        c.eq(len(archive_opens), 1, "audit opens the archive exactly once")
        c.true(bool(archive_opens[0] & os.O_NOFOLLOW),
               "audit archive fd rejects a final symlink")
        c.true(hasattr(wrapped_zipfile.opened_arguments[0], "fileno"),
               "ZipFile receives the already verified archive file object")


def test_audited_inventory_retains_deterministic_member_digests(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "source"
        expected, _ = _synthetic_tree(data_root)
        archive_path = data_root / RAVDESS_ARCHIVE_RELATIVE_PATH
        original_members = _zip_member_bytes(archive_path)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        c.eq(
            inventory.member_sha256,
            {name: hashlib.sha256(payload).hexdigest()
             for name, payload in sorted(original_members.items())},
            "audited inventory retains deterministic member byte digests",
        )


def test_generation_uses_one_verified_fd_and_audited_member_digests(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        data_root = base / "source"
        expected, extracted = _synthetic_tree(data_root)
        archive_path = data_root / RAVDESS_ARCHIVE_RELATIVE_PATH
        original_members = _zip_member_bytes(archive_path)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)

        changed = _face()
        changed[54, 1] += 7.0
        changed_csv = base / extracted[0].name
        _write_csv(changed_csv, [
            _csv_row(1, 0.000, 0.99, changed),
            _csv_row(2, 0.033, 0.50, changed),
        ])
        replacement_members = dict(original_members)
        replacement_members[extracted[0].name] = changed_csv.read_bytes()
        replacement_path = archive_path.with_name("transient-replacement.zip")
        _write_zip_bytes(replacement_path, replacement_members)

        archive_opens: list[int] = []
        original_open = prep.os.open
        original_zipfile, wrapped_zipfile = _transient_zipfile_path_swap(
            archive_path, replacement_path
        )

        def tracked_open(path, flags, *args, **kwargs):
            if Path(path).resolve() == archive_path.resolve():
                archive_opens.append(flags)
            return original_open(path, flags, *args, **kwargs)

        prep.os.open = tracked_open
        prep.zipfile.ZipFile = wrapped_zipfile
        output = base / "derived_semantic23"
        try:
            c.raises(lambda: build_generation_from_audited_sources(
                data_root, output, inventory, expectation=expected,
                id_key=TEST_ID_KEY,
            ), RuntimeError,
            "transient archive path replacement retains an indeterminate generation")
        finally:
            prep.zipfile.ZipFile = original_zipfile
            prep.os.open = original_open

        c.true(not output.exists(),
               "transient path replacement cannot publish consumed bytes")
        c.eq(len(archive_opens), 1, "generation opens the archive exactly once")
        c.true(bool(archive_opens[0] & os.O_NOFOLLOW),
               "generation archive fd rejects a final symlink")
        c.true(hasattr(wrapped_zipfile.opened_arguments[0], "fileno"),
               "generation ZipFile consumes the already verified file object")
        c.eq(len(list(base.glob(f".{output.name}.staging-*"))), 1,
             "transient replacement retains one auditable private stage")


def test_generation_rejects_member_digest_outside_audited_inventory(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        data_root = base / "source"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        tampered = dict(inventory.member_sha256)
        tampered[sorted(tampered)[0]] = "0" * 64
        mismatched_inventory = replace(inventory, member_sha256=tampered)
        output = base / "derived_semantic23"
        c.raises(lambda: build_generation_from_audited_sources(
            data_root, output, mismatched_inventory, expectation=expected,
            id_key=TEST_ID_KEY), RuntimeError,
            "single-read member bytes must match the audited digest map")
        c.true(not output.exists(), "member digest mismatch publishes nothing")
        c.eq(len(list(base.glob(f".{output.name}.staging-*"))), 1,
             "member mismatch retains one auditable private stage")


def test_output_paths_reject_lexical_symlink_bypasses(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        data_root = base / "source"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)

        external_parent = base / "external-parent"
        external_parent.mkdir()
        linked_parent = data_root / "linked-parent"
        linked_parent.symlink_to(external_parent, target_is_directory=True)
        escaped_output = linked_parent / "derived_semantic23"
        c.raises(lambda: build_generation_from_audited_sources(
            data_root, escaped_output, inventory, expectation=expected,
            id_key=TEST_ID_KEY), ValueError,
            "symlinked descendant output parent is rejected")
        c.true(not (external_parent / "derived_semantic23").exists(),
               "descendant symlink cannot redirect derived output")

        canonical = data_root / "derived_semantic23"
        external_output = base / "external-output"
        canonical.symlink_to(external_output, target_is_directory=True)
        key_path = data_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        original = prep.FROZEN_RAVDESS_INVENTORY
        prep.FROZEN_RAVDESS_INVENTORY = expected
        try:
            c.raises(lambda: prep.prepare_ravdess_semantic23(
                data_root, id_key_path=key_path
            ), FileExistsError, "canonical final symlink is rejected lexically")
        finally:
            prep.FROZEN_RAVDESS_INVENTORY = original
        c.true(canonical.is_symlink() and not external_output.exists(),
               "final symlink is neither followed nor replaced")
        c.true(not key_path.exists(),
               "invalid canonical output fails before private key creation")


def test_production_rejects_resolved_but_noncanonical_output_alias(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "source"
        expected, _ = _synthetic_tree(data_root)
        alias = data_root / "alias"
        alias.symlink_to(data_root, target_is_directory=True)
        key_path = data_root / ".must-not-exist.key"
        original = prep.FROZEN_RAVDESS_INVENTORY
        prep.FROZEN_RAVDESS_INVENTORY = expected
        try:
            c.raises(lambda: prep.prepare_ravdess_semantic23(
                data_root, output_root=alias / "derived_semantic23",
                id_key_path=key_path,
            ), ValueError, "production output must be the exact lexical canonical path")
        finally:
            prep.FROZEN_RAVDESS_INVENTORY = original
        c.true(not key_path.exists(), "noncanonical output fails without key creation")


def test_private_key_read_is_same_fd_nofollow_and_identity_bound(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        key_path = base / "private-id.key"
        key_path.write_bytes(b"a" * 32)
        key_path.chmod(0o600)
        original_open = prep.os.open
        read_flags: list[int] = []

        def tracked_open(path, flags, *args, **kwargs):
            if Path(path) == key_path:
                read_flags.append(flags)
            return original_open(path, flags, *args, **kwargs)

        prep.os.open = tracked_open
        try:
            c.eq(prep.load_or_create_private_id_key(key_path), b"a" * 32,
                 "existing key bytes")
        finally:
            prep.os.open = original_open
        c.eq(len(read_flags), 1, "existing key is opened exactly once")
        c.true(bool(read_flags[0] & os.O_NOFOLLOW),
               "existing key read rejects a final symlink")
        c.eq(read_flags[0] & os.O_ACCMODE, os.O_RDONLY,
             "existing key is read through an O_RDONLY fd")

        replacement = base / "replacement.key"
        replacement.write_bytes(b"b" * 32)
        replacement.chmod(0o600)
        parked = base / "parked.key"
        swapped = False

        def swap_after_open(path, flags, *args, **kwargs):
            nonlocal swapped
            descriptor = original_open(path, flags, *args, **kwargs)
            if Path(path) == key_path and not swapped:
                os.replace(key_path, parked)
                os.replace(replacement, key_path)
                swapped = True
            return descriptor

        prep.os.open = swap_after_open
        try:
            c.raises(lambda: prep.load_or_create_private_id_key(key_path),
                     ValueError, "key lexical identity change after open is rejected")
        finally:
            prep.os.open = original_open


def test_private_key_creation_is_exclusive_nofollow_and_owner_only(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        key_path = Path(temporary) / "private-id.key"
        original_open = prep.os.open
        original_read = prep.os.read
        original_fsync = prep.os.fsync
        staging_identity: list[tuple[int, int]] = []
        staging_descriptor: list[int] = []
        events: list[str] = []

        def tracked_open(path, flags, mode=0o777, *args, **kwargs):
            descriptor = original_open(path, flags, mode, *args, **kwargs)
            if (
                isinstance(path, str)
                and path.startswith(f".{key_path.name}.staging-")
            ):
                info = os.fstat(descriptor)
                staging_identity.append((int(info.st_dev), int(info.st_ino)))
                staging_descriptor.append(descriptor)
                events.append("open-staging")
                c.true(bool(flags & os.O_EXCL) and bool(flags & os.O_NOFOLLOW),
                       "key staging is exclusive and nofollow")
                c.eq(mode, 0o600, "key staging requests exact owner-only mode")
            return descriptor

        def tracked_fsync(descriptor):
            if staging_descriptor and descriptor == staging_descriptor[0]:
                info = os.fstat(descriptor)
                c.eq(stat.S_IMODE(info.st_mode), 0o600,
                     "staging is owner-only before durability sync")
                c.eq(info.st_nlink, 1, "staging has one link before publication")
                c.eq(info.st_size, 32, "staging is complete before durability sync")
                c.true(not key_path.exists(),
                       "canonical key is absent while staging bytes are synced")
                events.append("fsync-staging")
            return original_fsync(descriptor)

        def tracked_read(descriptor, count):
            if staging_descriptor and descriptor == staging_descriptor[0]:
                events.append("readback-staging")
            return original_read(descriptor, count)

        prep.os.open = tracked_open
        prep.os.fsync = tracked_fsync
        prep.os.read = tracked_read
        try:
            payload = prep.load_or_create_private_id_key(key_path)
        finally:
            prep.os.read = original_read
            prep.os.fsync = original_fsync
            prep.os.open = original_open
        c.eq(len(payload), 32, "created key size")
        c.eq(len(staging_identity), 1,
             "new key bytes are created in one private staging inode")
        c.true(events.index("open-staging") < events.index("fsync-staging")
               < events.index("readback-staging"),
               "staging is written, synced, then verified through the same fd")
        final = key_path.stat()
        c.eq((int(final.st_dev), int(final.st_ino)), staging_identity[0],
             "anchored no-replace publication moves the verified staging inode")
        c.eq(stat.S_IMODE(key_path.stat().st_mode), 0o600,
             "created key remains exact owner-only mode")
        c.eq(list(key_path.parent.glob(f".{key_path.name}.staging-*")), [],
             "successful publication leaves no staging name")


def test_private_key_crash_never_exposes_partial_canonical_file(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        key_path = Path(temporary) / "private-id.key"
        child = os.fork()
        if child == 0:
            original_write = prep.os.write

            def partial_then_exit(descriptor, payload):
                original_write(descriptor, bytes(payload[:1]))
                os._exit(73)

            prep.os.write = partial_then_exit
            prep.load_or_create_private_id_key(key_path)
            os._exit(0)
        waited, status = os.waitpid(child, 0)
        c.eq(waited, child)
        c.true(os.WIFEXITED(status) and os.WEXITSTATUS(status) == 73,
               "synthetic child stops immediately after one partial write")
        c.true(not os.path.lexists(key_path),
               "a partial staging write never becomes the canonical key")
        residue = list(key_path.parent.glob(f".{key_path.name}.staging-*"))
        c.eq(len(residue), 1, "interrupted private staging is visible for audit")


def test_private_key_unknown_staging_residue_fails_without_mutation(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        key_path = Path(temporary) / "private-id.key"
        residue = key_path.parent / f".{key_path.name}.staging-unknown"
        residue.write_bytes(b"partial")
        residue.chmod(0o600)
        before = (
            residue.read_bytes(), residue.stat().st_ino,
            stat.S_IMODE(residue.stat().st_mode),
        )
        c.raises(lambda: prep.load_or_create_private_id_key(key_path),
                 RuntimeError, "unknown matching staging residue fails closed")
        after = (
            residue.read_bytes(), residue.stat().st_ino,
            stat.S_IMODE(residue.stat().st_mode),
        )
        c.eq(after, before, "unknown staging inode is never deleted or changed")
        c.true(not os.path.lexists(key_path),
               "residue rejection cannot create a canonical key")


def test_private_key_concurrent_creators_share_one_committed_winner(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        key_path = base / "private-id.key"
        winner_ready_read, winner_ready_write = os.pipe()
        winner_release_read, winner_release_write = os.pipe()
        loser_opened_read, loser_opened_write = os.pipe()
        candidates = (b"a" * 32, b"b" * 32)
        winner = os.fork()
        if winner == 0:
            os.close(winner_ready_read)
            os.close(winner_release_write)
            os.close(loser_opened_read)
            os.close(loser_opened_write)
            original_write = prep.os.write
            original_read = prep.os.read
            paused = False

            def pause_first_key_write(descriptor, payload):
                nonlocal paused
                if not paused:
                    paused = True
                    original_write(winner_ready_write, b"1")
                    original_read(winner_release_read, 1)
                return original_write(descriptor, payload)

            prep.os.write = pause_first_key_write
            prep.secrets.token_bytes = lambda count: candidates[0]
            try:
                result = prep.load_or_create_private_id_key(key_path)
                (base / "child-0.result").write_bytes(result)
                os._exit(0)
            except BaseException:
                os._exit(91)
        os.close(winner_ready_write)
        os.close(winner_release_read)
        c.eq(os.read(winner_ready_read, 1), b"1",
             "winner pauses after opening private storage but before full write")

        loser = os.fork()
        if loser == 0:
            os.close(winner_ready_read)
            os.close(winner_release_write)
            os.close(loser_opened_read)
            original_open = prep.os.open
            signalled = False

            def signal_first_open(path, flags, *args, **kwargs):
                nonlocal signalled
                descriptor = original_open(path, flags, *args, **kwargs)
                if not signalled:
                    signalled = True
                    os.write(loser_opened_write, b"1")
                return descriptor

            prep.os.open = signal_first_open
            prep.secrets.token_bytes = lambda count: candidates[1]
            try:
                result = prep.load_or_create_private_id_key(key_path)
                (base / "child-1.result").write_bytes(result)
                os._exit(0)
            except BaseException:
                os._exit(91)
        os.close(loser_opened_write)
        c.eq(os.read(loser_opened_read, 1), b"1",
             "loser reaches the same creation lifecycle while winner is paused")
        os.write(winner_release_write, b"1")
        os.close(winner_release_write)
        os.close(winner_ready_read)
        os.close(loser_opened_read)
        children = [winner, loser]
        statuses = [os.waitpid(child, 0)[1] for child in children]
        c.true(all(os.WIFEXITED(value) and os.WEXITSTATUS(value) == 0
                   for value in statuses),
               "both concurrent callers complete through winner-or-validated-loser")
        results = tuple((base / f"child-{index}.result").read_bytes()
                        for index in range(2))
        c.eq(results[0], results[1], "both callers return the canonical key bytes")
        c.true(results[0] in candidates, "one generated candidate wins exactly once")
        c.eq(sum(result == candidate
                 for result, candidate in zip(results, candidates)), 1,
             "only the winning child observes its own generated candidate")
        c.eq(key_path.read_bytes(), results[0],
             "the loser validates rather than replacing the committed key")
        c.eq(stat.S_IMODE(key_path.stat().st_mode), 0o600)
        c.eq(key_path.stat().st_nlink, 1)
        c.eq(list(base.glob(f".{key_path.name}.staging-*")), [],
             "concurrent completion leaves no staging residue")


def test_private_key_failure_closes_fds_and_never_deletes_replacement(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        key_path = base / "private-id.key"
        parked = base / "owned-staging-parked"
        original_fsync = prep.os.fsync
        replaced: list[Path] = []
        fd_root = Path("/dev/fd") if Path("/dev/fd").is_dir() else Path("/proc/self/fd")
        before_fds = len(list(fd_root.iterdir()))

        def replace_staging_then_fail(descriptor):
            info = os.fstat(descriptor)
            if stat.S_ISREG(info.st_mode) and not replaced:
                original_fsync(descriptor)
                matches = list(base.glob(f".{key_path.name}.staging-*"))
                if not matches:
                    raise OSError("private key creation did not use staging")
                staging = matches[0]
                staging.rename(parked)
                staging.write_bytes(b"foreign-residue")
                staging.chmod(0o600)
                replaced.append(staging)
                raise OSError("synthetic staging fsync failure")
            return original_fsync(descriptor)

        prep.os.fsync = replace_staging_then_fail
        try:
            c.raises(lambda: prep.load_or_create_private_id_key(key_path), OSError,
                     "staging failure is surfaced")
        finally:
            prep.os.fsync = original_fsync
        c.eq(len(list(fd_root.iterdir())), before_fds,
             "key creation closes every descriptor on exception")
        c.true(not os.path.lexists(key_path),
               "failed staging never publishes a canonical key")
        c.true(parked.is_file() and len(parked.read_bytes()) == 32,
               "the originally owned inode is not confused with its replacement")
        c.true(bool(replaced), "failure occurs against a private staging inode")
        c.eq(replaced[0].read_bytes(), b"foreign-residue",
             "identity-mismatched replacement is never deleted")


def test_private_key_failure_never_unlinks_a_staging_name_after_identity_check(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        key_path = base / "private-id.key"
        foreign_marker = base / "foreign-marker"
        foreign_marker.write_bytes(b"foreign-marker-must-survive")
        foreign_marker.chmod(0o600)
        parked = base / "owned-staging-parked"
        original_fsync = prep.os.fsync
        original_unlink = prep.os.unlink
        failures: list[int] = []
        unlink_calls: list[str] = []

        def fail_first_staging_fsync(descriptor):
            info = os.fstat(descriptor)
            if stat.S_ISREG(info.st_mode) and not failures:
                original_fsync(descriptor)
                failures.append(descriptor)
                raise OSError("synthetic staging durability failure")
            return original_fsync(descriptor)

        def replace_at_unlink(path, *args, **kwargs):
            name = os.fspath(path)
            if isinstance(name, str) and name.startswith(
                f".{key_path.name}.staging-"
            ):
                staging = base / name
                staging.rename(parked)
                foreign_marker.rename(staging)
                unlink_calls.append(name)
            return original_unlink(path, *args, **kwargs)

        prep.os.fsync = fail_first_staging_fsync
        prep.os.unlink = replace_at_unlink
        try:
            c.raises(
                lambda: prep.load_or_create_private_id_key(key_path),
                OSError,
                "a staging durability failure is surfaced",
            )
        finally:
            prep.os.unlink = original_unlink
            prep.os.fsync = original_fsync
        c.true(
            foreign_marker.is_file()
            and foreign_marker.read_bytes() == b"foreign-marker-must-survive",
            "a marker replacing staging at unlink time is never deleted",
        )
        c.eq(unlink_calls, [], "failure cleanup never unlinks a staging pathname")
        residue = list(base.glob(f".{key_path.name}.staging-*"))
        c.eq(len(residue), 1, "the owned failed staging inode remains auditable")
        c.eq(len(residue[0].read_bytes()), 32, "auditable staging retains complete bytes")
        c.true(not os.path.lexists(key_path), "failure publishes no canonical key")


def test_generation_is_bound_to_verified_archive_member_bytes(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        data_root = base / "source"
        expected, extracted = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        archive_path = data_root / RAVDESS_ARCHIVE_RELATIVE_PATH
        with zipfile.ZipFile(archive_path, "r") as archive:
            member_bytes = archive.read(extracted[0].name)

        changed = _face()
        changed[54, 1] += 4.0
        _write_csv(extracted[0], [
            _csv_row(1, 0.000, 0.99, changed),
            _csv_row(2, 0.033, 0.50, changed),
        ])
        c.true(hashlib.sha256(extracted[0].read_bytes()).hexdigest()
               != hashlib.sha256(member_bytes).hexdigest(),
               "synthetic extracted copy is value-tampered")

        output = base / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root, output, inventory, expectation=expected,
            id_key=TEST_ID_KEY,
        )
        manifest = json.loads((output / "manifest.json").read_text("utf-8"))
        source_digest = hashlib.sha256(member_bytes).hexdigest()
        expected_trial_id = opaque_trial_id(
            extracted[0].name, source_digest, key=TEST_ID_KEY
        )
        two_frame = next(item for item in manifest["trials"]
                         if item["trial_id"] == expected_trial_id)
        c.eq(two_frame["trial_id"], opaque_trial_id(
            extracted[0].name, source_digest, key=TEST_ID_KEY
        ),
             "keyed trial identity is computed from verified ZIP member bytes")
        archive_trial = prep.parse_openface_csv_bytes(
            member_bytes, source_name=extracted[0].name
        )
        with np.load(output / "trials" / f"{two_frame['trial_id']}.npz") as cache:
            c.true(bool(np.array_equal(cache["features"], archive_trial.features)),
                   "cache values come from the verified ZIP, not extracted copy")


def test_public_manifest_never_exposes_audited_raw_member_digests(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        data_root = base / "source"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        output = base / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root, output, inventory, expectation=expected,
            id_key=TEST_ID_KEY,
        )
        manifest_text = (output / "manifest.json").read_text("utf-8")
        manifest = json.loads(manifest_text)
        for raw_digest in inventory.member_sha256.values():
            c.true(raw_digest not in manifest_text,
                   "public manifest cannot expose an enumerable raw member digest")
        c.true(all("source_sha256" not in record for record in manifest["trials"]),
               "public trial records omit unkeyed source digests")


def test_public_cache_integrity_ids_are_private_key_scoped(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        data_root = base / "source"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        outputs = (base / "derived-a", base / "derived-b")
        keys = (b"a" * 32, b"b" * 32)
        manifests: list[dict] = []

        for output, key in zip(outputs, keys):
            build_generation_from_audited_sources(
                data_root, output, inventory, expectation=expected,
                id_key=key,
            )
            text = (output / "manifest.json").read_text("utf-8")
            manifests.append(json.loads(text))
            raw_cache_digests = {
                hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (output / "trials").glob("*.npz")
            }
            for raw_digest in raw_cache_digests:
                c.true(raw_digest not in text,
                       "public manifest cannot expose a raw cache SHA-256")

        for manifest in manifests:
            c.true(all(set(record) == {
                "trial_id", "actor_id", "cache_integrity_id",
            } for record in manifest["trials"]),
                "public trial records expose only keyed cache integrity IDs")
        first_ids = {
            record["cache_integrity_id"] for record in manifests[0]["trials"]
        }
        second_ids = {
            record["cache_integrity_id"] for record in manifests[1]["trials"]
        }
        c.true(first_ids.isdisjoint(second_ids),
               "different private keys cannot yield intersectable cache fingerprints")


def test_public_manifest_order_is_keyed_not_raw_name_order(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        data_root = base / "source"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        output = base / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root, output, inventory, expectation=expected,
            id_key=TEST_ID_KEY,
        )
        manifest = json.loads((output / "manifest.json").read_text("utf-8"))
        public_order = [record["trial_id"] for record in manifest["trials"]]
        raw_name_order = [
            opaque_trial_id(name, digest, key=TEST_ID_KEY)
            for name, digest in inventory.member_sha256.items()
        ]
        c.eq(set(public_order), set(raw_name_order),
             "public records contain the exact keyed v2 trial set")
        c.eq(public_order, sorted(public_order),
             "keyed opaque trial ID determines public record order")


def test_public_manifest_keeps_only_aggregate_frame_qc(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        data_root = base / "source"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        output = base / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root, output, inventory, expectation=expected,
            id_key=TEST_ID_KEY,
        )
        manifest = json.loads((output / "manifest.json").read_text("utf-8"))
        records = manifest["trials"]
        c.true(all("source_frames" not in record and "valid_frames" not in record
                   for record in records),
               "per-trial frame counts cannot fingerprint public source members")
        c.eq(manifest["quality_control"], {
            "source_frames": 3,
            "valid_frames": 2,
            "invalid_frames": 1,
        }, "frame quality control remains available only in aggregate")
        c.true(all(record["actor_id"].startswith("actor_") for record in records),
               "keyed actor grouping remains available for training splits")


def test_manifest_deidentification_guard_rejects_raw_member_digest(c: Check):
    raw_digest = hashlib.sha256(b"public-archive-member").hexdigest()
    c.raises(lambda: prep._assert_manifest_deidentified(
        json.dumps({"leaked_digest": raw_digest}),
        source_root=Path("/private/raw/ravdess"),
        source_paths=[Path("01-01-01-01-01-01-01.csv")],
        raw_source_sha256s={raw_digest},
    ), ValueError, "aggregate deidentification guard rejects raw source digests")


def test_output_parent_swap_after_validation_fails_closed(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        data_root = base / "source"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        safe_parent = data_root / "safe-output-parent"
        safe_parent.mkdir()
        parked_parent = data_root / "parked-safe-output-parent"
        attack_target = base / "attack-target"
        attack_target.mkdir()
        output = safe_parent / "derived_semantic23"

        original_acquire = prep._acquire_output_lock
        swapped = False

        def swap_parent_then_acquire(*args, **kwargs):
            nonlocal swapped
            if not swapped:
                os.replace(safe_parent, parked_parent)
                safe_parent.symlink_to(attack_target, target_is_directory=True)
                swapped = True
            return original_acquire(*args, **kwargs)

        prep._acquire_output_lock = swap_parent_then_acquire
        try:
            c.raises(lambda: build_generation_from_audited_sources(
                data_root, output, inventory, expectation=expected,
                id_key=TEST_ID_KEY,
            ), ValueError, "output parent identity swap after validation fails closed")
        finally:
            prep._acquire_output_lock = original_acquire
            if safe_parent.is_symlink():
                safe_parent.unlink()
            if parked_parent.exists():
                os.replace(parked_parent, safe_parent)

        c.true(not any(attack_target.iterdir()),
               "parent swap creates no lock, staging, or output in attack target")
        c.true(not any("staging" in path.name for path in safe_parent.iterdir()),
               "failed anchored transaction removes sensitive staging")
        c.true(not output.exists(), "parent swap publishes no derived output")


def test_ravdess_parent_permissions_and_original_stage_identity_are_required(
    c: Check,
):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        data_root = base / "source"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        unsafe_parent = data_root / "unsafe-output-parent"
        unsafe_parent.mkdir(mode=0o700)
        unsafe_parent.chmod(0o777)
        output = unsafe_parent / "derived_semantic23"

        c.raises(lambda: build_generation_from_audited_sources(
            data_root, output, inventory, expectation=expected,
            id_key=TEST_ID_KEY,
        ), ValueError,
            "RAVDESS generation rejects a group/world-writable output parent")
        c.eq(tuple(unsafe_parent.iterdir()), (),
             "unsafe parent rejection precedes lock or staging creation")

    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        data_root = base / "source"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        output = base / "derived_semantic23"
        parked_name = ".parked-original-stage"
        original_publish = prep._publish_directory_no_replace
        replacement_created = False

        def replace_stage_then_publish(
            parent_descriptor,
            stage_name,
            destination_name,
            *args,
            **kwargs,
        ):
            nonlocal replacement_created
            os.rename(
                stage_name, parked_name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            os.mkdir(stage_name, 0o700, dir_fd=parent_descriptor)
            replacement_created = True
            return original_publish(
                parent_descriptor,
                stage_name,
                destination_name,
                *args,
                **kwargs,
            )

        prep._publish_directory_no_replace = replace_stage_then_publish
        try:
            c.raises(lambda: build_generation_from_audited_sources(
                data_root, output, inventory, expectation=expected,
                id_key=TEST_ID_KEY,
            ), RuntimeError,
                "publication rejects a same-name directory replacing the held stage")
        finally:
            prep._publish_directory_no_replace = original_publish

        c.true(replacement_created, "hostile stage replacement reached publication")
        c.true(not output.exists(), "replacement staging directory is never published")
        c.true((base / parked_name).is_dir(),
               "the original held generation remains as indeterminate evidence")


def test_generation_holds_nofollow_output_parent_directory_fd(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        data_root = base / "source"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        output = base / "derived_semantic23"
        original_open = prep.os.open
        parent_flags: list[int] = []

        def tracked_open(path, flags, *args, **kwargs):
            if Path(path) == output.parent and kwargs.get("dir_fd") is None:
                parent_flags.append(flags)
            return original_open(path, flags, *args, **kwargs)

        prep.os.open = tracked_open
        try:
            build_generation_from_audited_sources(
                data_root, output, inventory, expectation=expected,
                id_key=TEST_ID_KEY,
            )
        finally:
            prep.os.open = original_open
        c.eq(len(parent_flags), 1, "generation opens the output parent exactly once")
        c.true(bool(parent_flags[0] & os.O_DIRECTORY)
               and bool(parent_flags[0] & os.O_NOFOLLOW),
               "output parent is held with O_DIRECTORY and O_NOFOLLOW")


def test_output_parent_swap_during_staging_fails_closed(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        data_root = base / "source"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        safe_parent = data_root / "safe-output-parent"
        safe_parent.mkdir()
        parked_parent = data_root / "parked-safe-output-parent"
        attack_target = base / "attack-target"
        attack_target.mkdir()
        output = safe_parent / "derived_semantic23"

        original_guard = prep._assert_manifest_deidentified
        swapped = False

        def swap_parent_after_staging(*args, **kwargs):
            nonlocal swapped
            result = original_guard(*args, **kwargs)
            if not swapped:
                stage_name = next(
                    path.name for path in safe_parent.iterdir()
                    if "staging" in path.name
                )
                os.replace(safe_parent, parked_parent)
                safe_parent.symlink_to(attack_target, target_is_directory=True)
                (attack_target / stage_name).mkdir()
                swapped = True
            return result

        prep._assert_manifest_deidentified = swap_parent_after_staging
        try:
            c.raises(lambda: build_generation_from_audited_sources(
                data_root, output, inventory, expectation=expected,
                id_key=TEST_ID_KEY,
            ), RuntimeError,
            "output parent swap during staging retains indeterminate storage")
        finally:
            prep._assert_manifest_deidentified = original_guard
            if safe_parent.is_symlink():
                safe_parent.unlink()
            if parked_parent.exists():
                os.replace(parked_parent, safe_parent)

        c.true(not (attack_target / "derived_semantic23").exists(),
               "staging-time swap publishes no attacker output")
        c.true(not any((path / "manifest.json").exists()
                       for path in attack_target.iterdir() if path.is_dir()),
               "staging-time swap leaks no manifest into attacker staging")
        c.eq(len([
            path for path in safe_parent.iterdir() if "staging" in path.name
        ]), 1, "staging-time swap retains the anchored sensitive stage")
        c.true(not output.exists(), "staging-time swap publishes no trusted output")


def test_output_parent_swap_at_publish_fails_closed(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        data_root = base / "source"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        safe_parent = data_root / "safe-output-parent"
        safe_parent.mkdir()
        parked_parent = data_root / "parked-safe-output-parent"
        attack_target = base / "attack-target"
        attack_target.mkdir()
        output = safe_parent / "derived_semantic23"

        original_publish = prep._publish_directory_no_replace
        swapped = False

        def swap_parent_then_publish(*args, **kwargs):
            nonlocal swapped
            if not swapped:
                if isinstance(args[0], Path):
                    stage_name = args[0].name
                else:
                    stage_name = str(args[1])
                (attack_target / stage_name).mkdir()
                os.replace(safe_parent, parked_parent)
                safe_parent.symlink_to(attack_target, target_is_directory=True)
                swapped = True
            return original_publish(*args, **kwargs)

        prep._publish_directory_no_replace = swap_parent_then_publish
        try:
            c.raises(lambda: build_generation_from_audited_sources(
                data_root, output, inventory, expectation=expected,
                id_key=TEST_ID_KEY,
            ), RuntimeError,
            "output parent swap at publish retains indeterminate storage")
        finally:
            prep._publish_directory_no_replace = original_publish
            if safe_parent.is_symlink():
                safe_parent.unlink()
            if parked_parent.exists():
                os.replace(parked_parent, safe_parent)

        c.true(not (attack_target / "derived_semantic23").exists(),
               "publish-time swap creates no attacker output")
        c.true(not any("staging" in path.name for path in safe_parent.iterdir()),
               "published real stage no longer remains under its staging name")
        c.true(output.is_dir(),
               "post-publish failure retains the canonical generation as evidence")


def test_postpublication_canonical_mutation_never_returns_success(c: Check):
    observations: dict[str, dict[str, object]] = {}
    for surface in ("manifest", "cache"):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "source"
            expected, _ = _synthetic_tree(data_root)
            inventory = audit_ravdess_inventory(data_root, expectation=expected)
            output = data_root / "derived_semantic23"
            original_publish = prep._publish_directory_no_replace
            mutated = False

            def publish_then_mutate(*args, **kwargs):
                nonlocal mutated
                result = original_publish(*args, **kwargs)
                target = output / "manifest.json"
                if surface == "cache":
                    target = sorted((output / "trials").glob("*.npz"))[0]
                with target.open("ab") as handle:
                    handle.write(b"post-publication mutation")
                    handle.flush()
                    os.fsync(handle.fileno())
                mutated = True
                return result

            observed: BaseException | None = None
            prep._publish_directory_no_replace = publish_then_mutate
            try:
                try:
                    build_generation_from_audited_sources(
                        data_root,
                        output,
                        inventory,
                        expectation=expected,
                        id_key=TEST_ID_KEY,
                    )
                except BaseException as exc:  # noqa: BLE001 - inspect fail-closed state
                    observed = exc
            finally:
                prep._publish_directory_no_replace = original_publish
            observations[surface] = {
                "mutated": mutated,
                "exception": type(observed).__name__ if observed else None,
                "canonical_retained": output.is_dir(),
                "staging_count": len(list(
                    output.parent.glob(f".{output.name}.staging-*")
                )),
            }

    c.eq(
        observations,
        {
            surface: {
                "mutated": True,
                "exception": RuntimeError.__name__,
                "canonical_retained": True,
                "staging_count": 0,
            }
            for surface in ("manifest", "cache")
        },
        "canonical manifest/cache are reauthorized before success and retained on failure",
    )


def test_output_parent_swap_at_lock_release_retains_generation(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        data_root = base / "source"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        safe_parent = data_root / "safe-output-parent"
        safe_parent.mkdir()
        parked_parent = data_root / "parked-safe-output-parent"
        attack_target = base / "attack-target"
        attack_target.mkdir()
        output = safe_parent / "derived_semantic23"

        original_release = prep._release_output_lock
        swapped = False

        def swap_parent_then_release(*args, **kwargs):
            nonlocal swapped
            result = original_release(*args, **kwargs)
            if not swapped:
                os.replace(safe_parent, parked_parent)
                safe_parent.symlink_to(attack_target, target_is_directory=True)
                swapped = True
            return result

        prep._release_output_lock = swap_parent_then_release
        caught: BaseException | None = None
        try:
            try:
                build_generation_from_audited_sources(
                    data_root, output, inventory, expectation=expected,
                    id_key=TEST_ID_KEY,
                )
            except BaseException as exc:  # noqa: BLE001 - assert exact fail-closed state
                caught = exc
            c.true(isinstance(caught, RuntimeError),
                   "lock-release parent swap must fail the transaction")
            c.true(not (attack_target / output.name).exists(),
                   "release-time swap publishes no attacker generation")
            c.true((parked_parent / output.name).is_dir(),
                   "release-time swap retains the anchored canonical generation")
            c.true(not any("staging" in path.name for path in attack_target.iterdir()),
                   "release-time swap leaves no attacker staging")
            c.true(not any("staging" in path.name for path in parked_parent.iterdir()),
                   "release-time swap removes anchored staging")
        finally:
            prep._release_output_lock = original_release
            if safe_parent.is_symlink():
                safe_parent.unlink()
            if parked_parent.exists():
                os.replace(parked_parent, safe_parent)

        c.true(output.is_dir(),
               "failed release retains the canonical generation as indeterminate")
        lock = safe_parent / f".{output.name}.lock"
        c.true(lock.is_file() and not lock.is_symlink(),
               "failed release preserves the persistent owner-only lock")
        _assert_lock_reacquirable(c, lock, "failed release lock is reacquirable")


def test_archive_postcheck_output_lock_and_no_replace_fail_closed(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        data_root = base / "source"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        output = base / "derived_semantic23"
        lock = output.parent / f".{output.name}.lock"
        lock.touch(mode=0o600)
        lock.chmod(0o600)
        foreign_descriptor = os.open(lock, os.O_RDWR | os.O_NOFOLLOW)
        fcntl.flock(foreign_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            c.raises(lambda: build_generation_from_audited_sources(
                data_root, output, inventory, expectation=expected,
                id_key=TEST_ID_KEY), BlockingIOError,
                "held advisory output lock blocks concurrent producer")
        finally:
            fcntl.flock(foreign_descriptor, fcntl.LOCK_UN)
            os.close(foreign_descriptor)
        c.true(lock.is_file() and not lock.is_symlink(),
               "foreign persistent lock is never removed")
        c.true(not output.exists(), "locked transaction publishes nothing")
        _assert_lock_reacquirable(c, lock, "foreign lock is safely reacquirable")

        original_parser = prep.parse_openface_csv_bytes
        archive_path = data_root / RAVDESS_ARCHIVE_RELATIVE_PATH
        mutated = False

        def mutate_archive_after_first_read(data: bytes, *, source_name: str):
            nonlocal mutated
            trial = original_parser(data, source_name=source_name)
            if not mutated:
                replacement = archive_path.with_suffix(".replacement.zip")
                replacement.write_bytes(archive_path.read_bytes() + b"post-audit mutation")
                os.replace(replacement, archive_path)
                mutated = True
            return trial

        prep.parse_openface_csv_bytes = mutate_archive_after_first_read
        try:
            c.raises(lambda: build_generation_from_audited_sources(
                data_root, output, inventory, expectation=expected,
                id_key=TEST_ID_KEY), RuntimeError,
                "archive mutation during generation fails the pre-promotion postcheck")
        finally:
            prep.parse_openface_csv_bytes = original_parser
        c.true(not output.exists(), "mutated archive publishes no generation")
        c.true(lock.is_file() and not lock.is_symlink(),
               "failed transaction preserves its safe persistent lock")
        _assert_lock_reacquirable(c, lock, "failed transaction lock is reacquirable")
        c.eq(len(list(base.glob(f".{output.name}.staging-*"))), 1,
             "archive mutation retains one auditable staging directory")

        stage = base / ".manual-stage"
        destination = base / "manual-output"
        stage.mkdir(mode=0o700)
        destination.mkdir()
        parent_descriptor = os.open(
            base, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        try:
            stage_identity = prep._directory_identity(os.stat(stage))
            c.raises(lambda: prep._publish_directory_no_replace(
                parent_descriptor, stage.name, destination.name, stage_identity
            ), FileExistsError,
                "publication never replaces an existing empty path")
        finally:
            os.close(parent_descriptor)
        c.true(stage.is_dir() and destination.is_dir(),
               "failed no-replace publication preserves both paths")


def test_production_entrypoint_uses_private_key_and_canonical_output(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        invalid_root = base / "invalid-source"
        invalid_expected, _ = _synthetic_tree(invalid_root)
        invalid_archive = invalid_root / RAVDESS_ARCHIVE_RELATIVE_PATH
        invalid_archive.write_bytes(invalid_archive.read_bytes() + b"drift")
        invalid_key = invalid_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        original = prep.FROZEN_RAVDESS_INVENTORY
        prep.FROZEN_RAVDESS_INVENTORY = invalid_expected
        try:
            c.raises(lambda: prep.prepare_ravdess_semantic23(
                invalid_root, id_key_path=invalid_key
            ), ValueError, "invalid archive fails before private state is created")
        finally:
            prep.FROZEN_RAVDESS_INVENTORY = original
        c.true(not invalid_key.exists(),
               "failed source audit leaves no private key side effect")

        mismatch_root = base / "mismatched-key-source"
        mismatch_expected, _ = _synthetic_tree(mismatch_root)
        alternate_key = mismatch_root / ".alternate-private-id-key"
        canonical_key = mismatch_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        original = prep.FROZEN_RAVDESS_INVENTORY
        prep.FROZEN_RAVDESS_INVENTORY = mismatch_expected
        try:
            c.raises(
                lambda: prep.prepare_ravdess_semantic23(
                    mismatch_root, id_key_path=alternate_key
                ),
                ValueError,
                "production rejects a noncanonical private-key path",
            )
        finally:
            prep.FROZEN_RAVDESS_INVENTORY = original
        c.true(
            not alternate_key.exists() and not canonical_key.exists(),
            "key-path mismatch creates no alternate or canonical private state",
        )
        c.true(
            not (mismatch_root / "derived_semantic23").exists(),
            "key-path mismatch publishes no output",
        )

        data_root = base / "source"
        expected, _ = _synthetic_tree(data_root)
        key_path = data_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        original = prep.FROZEN_RAVDESS_INVENTORY
        prep.FROZEN_RAVDESS_INVENTORY = expected
        try:
            manifest = prep.prepare_ravdess_semantic23(
                data_root, id_key_path=key_path
            )
        finally:
            prep.FROZEN_RAVDESS_INVENTORY = original
        c.eq(manifest["schema"], SEMANTIC23_SCHEMA)
        c.true((data_root / "derived_semantic23" / "manifest.json").is_file(),
               "production entrypoint publishes only at canonical output")
        c.eq(stat.S_IMODE(key_path.stat().st_mode), 0o600,
             "production entrypoint uses an owner-only persistent key")


def test_production_residue_rejection_creates_no_canonical_key_or_stage(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "source"
        expected, _ = _synthetic_tree(data_root)
        output = data_root / "derived_semantic23"
        residue = data_root / f".{output.name}.staging-existing"
        residue.mkdir(mode=0o700)
        marker = residue / "marker"
        marker.write_bytes(b"retained transaction evidence")
        marker.chmod(0o600)

        def residue_snapshot() -> tuple[tuple[object, ...], ...]:
            snapshot: list[tuple[object, ...]] = []
            for path in (residue, marker):
                info = path.lstat()
                snapshot.append((
                    path.name,
                    stat.S_IMODE(info.st_mode),
                    info.st_dev,
                    info.st_ino,
                    info.st_nlink,
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                    path.read_bytes() if path.is_file() else None,
                ))
            return tuple(snapshot)

        residue_before = residue_snapshot()
        canonical_key = data_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        original_expectation = prep.FROZEN_RAVDESS_INVENTORY
        original_create = prep._create_directory_at
        original_write_cache = prep._write_cache_at
        stage_calls = 0
        cache_calls = 0

        def tracked_create(*args, **kwargs):
            nonlocal stage_calls
            stage_calls += 1
            return original_create(*args, **kwargs)

        def tracked_cache(*args, **kwargs):
            nonlocal cache_calls
            cache_calls += 1
            return original_write_cache(*args, **kwargs)

        observed: BaseException | None = None
        prep.FROZEN_RAVDESS_INVENTORY = expected
        prep._create_directory_at = tracked_create
        prep._write_cache_at = tracked_cache
        try:
            try:
                prep.prepare_ravdess_semantic23(data_root)
            except BaseException as exc:  # noqa: BLE001 - inspect zero-side-effect gate
                observed = exc
        finally:
            prep._write_cache_at = original_write_cache
            prep._create_directory_at = original_create
            prep.FROZEN_RAVDESS_INVENTORY = original_expectation

        c.true(
            isinstance(observed, RuntimeError),
            "preexisting producer residue fails closed",
        )
        c.true(
            not canonical_key.exists(),
            "residue rejection creates no canonical private key",
        )
        c.eq(
            list(data_root.glob(f".{canonical_key.name}.staging-*")),
            [],
            "residue rejection creates no private-key staging state",
        )
        c.eq(
            residue_snapshot(), residue_before,
            "residue directory and marker remain byte/stat identical",
        )
        c.eq(stage_calls, 0, "residue rejection creates no generation stage")
        c.eq(cache_calls, 0, "residue rejection opens no generation cache")
        c.true(not output.exists(), "residue rejection publishes no generation")


def test_generator_rejects_inventory_integer_type_aliases_before_key_or_stage(
    c: Check,
):
    integer_fields = (
        "archive_size",
        "csv_files",
        "actors",
        "frames",
        "unique_archive_member_names",
        "unique_source_content_sha256s",
        "duplicate_content_groups",
        "members_beyond_unique_content",
        "max_content_multiplicity",
        "cross_actor_duplicate_content_groups",
        "empty_trials",
        "repeated_headers",
    )
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "source"
        expected, _ = _synthetic_tree(data_root)
        audited = audit_ravdess_inventory(data_root, expectation=expected)
        output = data_root / "derived_semantic23"
        canonical_key = data_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        original_expectation = prep.FROZEN_RAVDESS_INVENTORY
        original_audit = prep.audit_ravdess_inventory
        original_key_loader = prep.load_or_create_private_id_key
        original_create = prep._create_directory_at
        original_write_cache = prep._write_cache_at
        key_calls = stage_calls = cache_calls = 0

        def unexpected_key_loader(*args, **kwargs):
            nonlocal key_calls
            key_calls += 1
            raise AssertionError("private-key loader reached after inventory type drift")

        def tracked_create(*args, **kwargs):
            nonlocal stage_calls
            stage_calls += 1
            return original_create(*args, **kwargs)

        def tracked_cache(*args, **kwargs):
            nonlocal cache_calls
            cache_calls += 1
            return original_write_cache(*args, **kwargs)

        mutations: list[tuple[str, object]] = []
        for field in integer_fields:
            value = getattr(audited, field)
            mutations.append((field, float(value)))
            if value in {0, 1}:
                mutations.append((field, bool(value)))

        observations: list[tuple[str, str, str]] = []
        prep.FROZEN_RAVDESS_INVENTORY = expected
        prep.load_or_create_private_id_key = unexpected_key_loader
        prep._create_directory_at = tracked_create
        prep._write_cache_at = tracked_cache
        try:
            for field, alias in mutations:
                mutated = replace(audited, **{field: alias})
                prep.audit_ravdess_inventory = (
                    lambda *_args, mutated=mutated, **_kwargs: mutated
                )
                observed: BaseException | None = None
                try:
                    prep.prepare_ravdess_semantic23(data_root)
                except BaseException as exc:  # noqa: BLE001 - inspect fail-closed type gate
                    observed = exc
                observations.append((
                    field,
                    type(alias).__name__,
                    type(observed).__name__ if observed else "none",
                ))
        finally:
            prep._write_cache_at = original_write_cache
            prep._create_directory_at = original_create
            prep.load_or_create_private_id_key = original_key_loader
            prep.audit_ravdess_inventory = original_audit
            prep.FROZEN_RAVDESS_INVENTORY = original_expectation

        c.eq(
            len(observations),
            len(mutations),
            "every frozen integer field exercises each equal-valued type alias",
        )
        c.true(
            all(kind == ValueError.__name__ for _, _, kind in observations),
            "generator rejects all float and bool inventory aliases type-exactly",
        )
        c.eq(key_calls, 0, "inventory type drift fails before key load/create")
        c.eq(stage_calls, 0, "inventory type drift creates no generation stage")
        c.eq(cache_calls, 0, "inventory type drift opens no cache")
        c.true(not canonical_key.exists(), "inventory type drift creates no key")
        c.true(not output.exists(), "inventory type drift publishes no generation")


def test_production_rechecks_canonical_key_before_return(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "source"
        expected, _ = _synthetic_tree(data_root)
        output = data_root / "derived_semantic23"
        key_path = data_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        original_expectation = prep.FROZEN_RAVDESS_INVENTORY
        original_publish = prep._publish_directory_no_replace
        mutated = False

        def publish_then_mutate_key(*args, **kwargs):
            nonlocal mutated
            result = original_publish(*args, **kwargs)
            key_path.write_bytes(b"z" * prep.PRIVATE_ID_KEY_BYTES)
            key_path.chmod(0o600)
            mutated = True
            return result

        observed: BaseException | None = None
        prep.FROZEN_RAVDESS_INVENTORY = expected
        prep._publish_directory_no_replace = publish_then_mutate_key
        try:
            try:
                prep.prepare_ravdess_semantic23(data_root)
            except BaseException as exc:  # noqa: BLE001 - inspect retained state
                observed = exc
        finally:
            prep._publish_directory_no_replace = original_publish
            prep.FROZEN_RAVDESS_INVENTORY = original_expectation
        c.true(mutated, "canonical private key is mutated after publication")
        c.true(
            isinstance(observed, RuntimeError),
            "post-publication canonical key mutation blocks success",
        )
        c.true(output.is_dir(), "failed post-publication key check retains output")


def test_inventory_and_transactional_generation_fail_closed(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)

        failing_source = base / "failing_source"
        failing_expected, _ = _synthetic_tree(
            failing_source, duplicate_first_frame=True
        )
        failing_inventory = audit_ravdess_inventory(
            failing_source, expectation=failing_expected
        )
        failed_output = base / "failed_generation"
        c.raises(lambda: build_generation_from_audited_sources(
            failing_source, failed_output, failing_inventory,
            expectation=failing_expected, id_key=TEST_ID_KEY), RuntimeError,
            "parse failure aborts the staged transaction")
        c.true(not failed_output.exists(), "failed transaction publishes no output")
        c.eq(len(list(base.glob(f".{failed_output.name}.staging-*"))), 1,
             "failed transaction retains its private staging directory")

        data_root = base / "source"
        expected, files = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        c.eq(inventory.csv_files, 2, "audited CSV count")
        c.eq(inventory.frames, 3, "audited frame count")

        output = base / "derived_semantic23"
        build_generation_from_audited_sources(data_root, output, inventory,
                                              expectation=expected,
                                              id_key=TEST_ID_KEY)
        manifest_path = output / "manifest.json"
        manifest_text = manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(manifest_text)
        c.eq(manifest["schema"], SEMANTIC23_SCHEMA, "manifest target schema")
        c.eq(manifest["adapter"]["source_topology"], "openface_68_2d",
             "manifest carries explicit adapter metadata")
        c.eq(manifest["adapter"]["scale_normalization"], "interocular_distance",
             "manifest carries source scaling")
        c.eq(len(manifest["trials"]), 2, "all audited trials emitted")
        c.true(all(item["trial_id"].startswith("trial_") for item in manifest["trials"]),
               "trial provenance is opaque")
        c.eq(manifest["provenance_policy"]["actor_id"],
             "private_hmac_sha256_base32",
             "manifest identifies keyed pseudonymization")
        c.true(TEST_ID_KEY.hex() not in manifest_text,
               "private HMAC key is never serialized")
        for source in files:
            c.true(source.name not in manifest_text and str(source) not in manifest_text,
                   "aggregate manifest contains no raw path or filename")
        c.eq(len(list((output / "trials").glob("*.npz"))), 2,
             "one cache per trial")
        two_frame_trial_id = opaque_trial_id(
            files[0].name,
            inventory.member_sha256[files[0].name],
            key=TEST_ID_KEY,
        )
        two_frame_record = next(item for item in manifest["trials"]
                                if item["trial_id"] == two_frame_trial_id)
        with np.load(output / "trials" / f"{two_frame_record['trial_id']}.npz") as cache:
            c.true(bool(np.array_equal(cache["valid_mask"], [True, False])),
                   "published cache retains the low-confidence detector gap")
            c.true(bool(np.allclose(cache["timestamps"], [0.000, 0.033])),
                   "published cache retains source timestamps")
            c.eq(str(cache["schema"]), SEMANTIC23_SCHEMA,
                 "published cache names the exact target schema")
        c.true(not any(
            p.name.startswith(f".{output.name}.staging-") for p in base.iterdir()
        ), "successful transaction leaves no staging for its own output name")
        lock = output.parent / f".{output.name}.lock"
        c.true(lock.is_file() and not lock.is_symlink(),
               "successful transaction preserves a safe persistent lock")
        _assert_lock_reacquirable(c, lock, "successful transaction lock is reacquirable")

        c.raises(lambda: build_generation_from_audited_sources(
            data_root, output, inventory, expectation=expected,
            id_key=TEST_ID_KEY), FileExistsError,
            "existing generation is not silently replaced")
        c.true(manifest_path.exists(), "failed replacement leaves old generation intact")

        archive = data_root / RAVDESS_ARCHIVE_RELATIVE_PATH
        with archive.open("ab") as handle:
            handle.write(b"unexpected archive mutation")
        c.raises(lambda: audit_ravdess_inventory(data_root, expectation=expected),
                 ValueError, "inventory drift fails closed before generation")


def test_failed_ravdess_generation_retains_residue_without_delete_and_blocks_retry(
    c: Check,
):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary).resolve()
        data_root = base / "failing-source"
        expected, _ = _synthetic_tree(data_root, duplicate_first_frame=True)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        output = base / "derived_semantic23"
        original_unlink = prep.os.unlink
        original_rmdir = prep.os.rmdir
        delete_calls: list[str] = []

        def tracked_unlink(*args, **kwargs):
            delete_calls.append("unlink")
            return original_unlink(*args, **kwargs)

        def tracked_rmdir(*args, **kwargs):
            delete_calls.append("rmdir")
            return original_rmdir(*args, **kwargs)

        prep.os.unlink = tracked_unlink
        prep.os.rmdir = tracked_rmdir
        try:
            try:
                build_generation_from_audited_sources(
                    data_root, output, inventory, expectation=expected,
                    id_key=TEST_ID_KEY,
                )
            except BaseException as exc:  # noqa: BLE001 - inspect fail-closed state
                observed = exc
            else:
                observed = None
        finally:
            prep.os.unlink = original_unlink
            prep.os.rmdir = original_rmdir
        c.true(isinstance(observed, RuntimeError),
               "failed producer reports retained indeterminate storage")
        c.eq(delete_calls, [],
             "failed producer never performs pathname-recursive deletion")
        residues = list(base.glob(f".{output.name}.staging-*"))
        c.eq(len(residues), 1, "failed producer retains exactly one private residue")
        c.eq(stat.S_IMODE(residues[0].stat().st_mode), 0o700)
        c.true(not output.exists(), "prepublication failure has no canonical output")

        original_create = prep._create_directory_at
        create_calls = 0

        def tracked_create(*args, **kwargs):
            nonlocal create_calls
            create_calls += 1
            return original_create(*args, **kwargs)

        prep._create_directory_at = tracked_create
        try:
            c.raises(
                lambda: build_generation_from_audited_sources(
                    data_root, output, inventory, expectation=expected,
                    id_key=TEST_ID_KEY,
                ),
                RuntimeError,
                "retained RAVDESS residue blocks every retry",
            )
        finally:
            prep._create_directory_at = original_create
        c.eq(create_calls, 0, "retry is blocked before any new staging directory")
        c.true(residues[0].is_dir(), "retry never mutates retained evidence")


def test_retained_generation_preserves_primary_and_cleanup_failures(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "source"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        output = data_root / "derived_semantic23"
        original_scan = prep._scan_staged_ravdess_caches_for_private_provenance
        original_release = prep._release_output_lock
        release_calls = 0

        def fail_primary(*args, **kwargs):
            original_scan(*args, **kwargs)
            raise ValueError("synthetic primary generation validation failure")

        def fail_cleanup(*args, **kwargs):
            nonlocal release_calls
            release_calls += 1
            original_release(*args, **kwargs)
            raise OSError("synthetic generation cleanup failure")

        prep._scan_staged_ravdess_caches_for_private_provenance = fail_primary
        prep._release_output_lock = fail_cleanup
        observed: BaseException | None = None
        try:
            try:
                build_generation_from_audited_sources(
                    data_root,
                    output,
                    inventory,
                    expectation=expected,
                    id_key=TEST_ID_KEY,
                )
            except BaseException as exc:  # noqa: BLE001 - inspect combined graph
                observed = exc
        finally:
            prep._release_output_lock = original_release
            prep._scan_staged_ravdess_caches_for_private_provenance = original_scan

        graph = _exception_chain_bytes(observed)
        c.true(
            isinstance(observed, RuntimeError)
            and b"RAVDESS generation storage is retained as indeterminate" in graph,
            "retained-storage wrapper remains the top-level failure",
        )
        c.true(
            b"synthetic primary generation validation failure" in graph,
            "retained-storage graph preserves the primary validation failure",
        )
        c.true(
            b"synthetic generation cleanup failure" in graph,
            "retained-storage graph preserves the cleanup failure",
        )
        linear: list[str] = []
        current = observed
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            linear.append(str(current))
            current = current.__cause__ or current.__context__
        c.true(
            any("synthetic primary generation validation failure" in item
                for item in linear)
            and any("synthetic generation cleanup failure" in item
                    for item in linear),
            "ordinary exception chaining retains primary and cleanup failures",
        )
        c.eq(release_calls, 1, "cleanup release is attempted exactly once")


def test_ravdess_cleanup_attachment_breaks_implicit_primary_cycle(c: Check):
    primary = ValueError("RAVDESS implicit primary")
    observed = None

    def fail_during_cleanup():
        try:
            raise primary
        finally:
            try:
                raise OSError("RAVDESS implicit cleanup")
            except OSError as cleanup_error:
                c.true(cleanup_error.__context__ is primary)
                outcome = prep._attach_cleanup_causes(
                    primary, (cleanup_error,),
                )
                raise outcome.with_traceback(primary.__traceback__)

    try:
        fail_during_cleanup()
    except BaseException as exc:
        observed = exc
    chain: list[BaseException] = []
    current = observed
    while current is not None and len(chain) < 8:
        chain.append(current)
        current = current.__cause__ or current.__context__
    c.eq(len(chain), len({id(error) for error in chain}))
    c.true(
        any(
            isinstance(error, ValueError)
            and "RAVDESS implicit primary" in str(error)
            for error in chain
        )
        and any(
            isinstance(error, OSError)
            and "RAVDESS implicit cleanup" in str(error)
            for error in chain
        ),
        "RAVDESS attachment retains one acyclic primary-cleanup chain",
    )


def test_committed_ravdess_generation_exposes_narrow_read_only_authorizer(c: Check):
    c.true(
        hasattr(prep, "authorize_committed_ravdess_semantic23"),
        "the bridge requires a public committed-generation authorizer",
    )
    c.true(
        "authorize_committed_ravdess_semantic23" in prep.__all__,
        "the committed-generation authorizer is part of the explicit public API",
    )


def test_generator_rejects_staged_aggregate_resources_before_publication_and_numpy(
    c: Check,
):
    observations: list[tuple[str, str, bool, int, int]] = []
    for mode in ("declared_frames", "expanded_bytes", "regular_payload"):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "ravdess"
            expected, _ = _synthetic_tree(data_root)
            inventory = audit_ravdess_inventory(data_root, expectation=expected)
            output = data_root / "derived_semantic23"
            original_write = prep._write_cache_at
            original_load = prep.np.load
            original_regular_limit = (
                prep._MAX_RAVDESS_AGGREGATE_REGULAR_PAYLOAD_BYTES
            )
            injected = 0
            load_calls = 0

            def fault_injected_write(
                parent_descriptor: int,
                cache_name: str,
                trial: prep.SemanticTrial,
            ) -> str:
                nonlocal injected
                written_trial = trial
                if mode == "declared_frames" and injected == 0:
                    written_trial = prep.SemanticTrial(
                        frame_indices=np.concatenate((
                            trial.frame_indices,
                            np.asarray([int(trial.frame_indices[-1]) + 1], dtype=np.int64),
                        )),
                        timestamps=np.concatenate((
                            trial.timestamps,
                            np.asarray([float(trial.timestamps[-1]) + 0.033], dtype=np.float64),
                        )),
                        detector_confidence=np.concatenate((
                            trial.detector_confidence,
                            trial.detector_confidence[-1:],
                        )),
                        features=np.concatenate((trial.features, trial.features[-1:]), axis=0),
                        valid_mask=np.concatenate((trial.valid_mask, trial.valid_mask[-1:])),
                        source_sha256=trial.source_sha256,
                    )
                cache_sha256 = original_write(
                    parent_descriptor, cache_name, written_trial
                )
                if mode == "expanded_bytes" and injected == 0:
                    descriptor = os.open(
                        cache_name,
                        os.O_RDWR | os.O_NOFOLLOW,
                        dir_fd=parent_descriptor,
                    )
                    try:
                        with os.fdopen(descriptor, "r+b", closefd=False) as handle:
                            rewritten = _pad_first_npy_header(handle.read())
                            handle.seek(0)
                            handle.truncate(0)
                            handle.write(rewritten)
                            handle.flush()
                            os.fsync(descriptor)
                        cache_sha256 = hashlib.sha256(rewritten).hexdigest()
                    finally:
                        os.close(descriptor)
                injected += 1
                return cache_sha256

            def materialized_before_staged_aggregate_gate(*args, **kwargs):
                nonlocal load_calls
                load_calls += 1
                raise AssertionError(
                    "np.load reached before staged aggregate resource gate"
                )

            if mode == "regular_payload":
                prep._MAX_RAVDESS_AGGREGATE_REGULAR_PAYLOAD_BYTES = 1
            prep._write_cache_at = fault_injected_write
            prep.np.load = materialized_before_staged_aggregate_gate
            observed: BaseException | None = None
            try:
                try:
                    build_generation_from_audited_sources(
                        data_root,
                        output,
                        inventory,
                        expectation=expected,
                        id_key=TEST_ID_KEY,
                    )
                except BaseException as exc:  # noqa: BLE001 - inspect gate ordering
                    observed = exc
            finally:
                prep.np.load = original_load
                prep._write_cache_at = original_write
                prep._MAX_RAVDESS_AGGREGATE_REGULAR_PAYLOAD_BYTES = (
                    original_regular_limit
                )
            observations.append((
                mode,
                type(observed).__name__ if observed else "none",
                output.exists(),
                injected,
                load_calls,
            ))

    c.eq(
        observations,
        [
            (mode, RuntimeError.__name__, False, 2, 0)
            for mode in ("declared_frames", "expanded_bytes", "regular_payload")
        ],
        "staged aggregate frame/expanded/regular budgets fail before publication and NumPy",
    )


def test_ravdess_npz_resource_metadata_is_rejected_before_numpy_load(c: Check):
    valid = _ravdess_cache_bytes()
    extra = io.BytesIO(valid)
    member = io.BytesIO()
    np.save(member, np.asarray(1, dtype=np.int64), allow_pickle=False)
    with zipfile.ZipFile(extra, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("unexpected.npy", member.getvalue())
    attacks = (
        ("declared compressed bytes", _patch_first_central_size(valid, field_offset=20)),
        ("declared expanded bytes", _patch_first_central_size(valid, field_offset=24)),
        ("excessive member count", extra.getvalue()),
    )
    original_load = prep.np.load

    def materialized_too_early(*_args, **_kwargs):
        raise RuntimeError("np.load reached before bounded ZIP inspection")

    prep.np.load = materialized_too_early
    try:
        for label, payload in attacks:
            c.raises(
                lambda value=payload: _ravdess_validate_without_materializing(value),
                ValueError,
                f"RAVDESS {label} is rejected before NumPy materialization",
            )
    finally:
        prep.np.load = original_load


def test_authorizer_rejects_cumulative_cache_budget_before_numpy_load(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "ravdess"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        key_path = data_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        key_path.write_bytes(TEST_ID_KEY)
        key_path.chmod(0o600)
        output = data_root / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root,
            output,
            inventory,
            expectation=expected,
            id_key=TEST_ID_KEY,
        )
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        inflated_payload = _ravdess_cache_bytes()
        for row in manifest["trials"]:
            cache_path = output / "trials" / f"{row['trial_id']}.npz"
            cache_path.write_bytes(inflated_payload)
            cache_path.chmod(0o600)
            cache_sha256 = hashlib.sha256(inflated_payload).hexdigest()
            row["cache_integrity_id"] = prep._opaque_cache_integrity_id(
                cache_sha256,
                trial_id=row["trial_id"],
                actor_id=row["actor_id"],
                key=TEST_ID_KEY,
            )
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8"
        )
        manifest_path.chmod(0o600)

        original_expectation = prep.FROZEN_RAVDESS_INVENTORY
        original_load = prep.np.load
        load_calls = 0

        def materialized_before_aggregate_gate(*args, **kwargs):
            nonlocal load_calls
            load_calls += 1
            raise AssertionError("np.load reached before cumulative cache gate")

        observed: BaseException | None = None
        prep.FROZEN_RAVDESS_INVENTORY = expected
        prep.np.load = materialized_before_aggregate_gate
        try:
            try:
                prep.authorize_committed_ravdess_semantic23(data_root)
            except BaseException as exc:  # noqa: BLE001 - inspect gate ordering
                observed = exc
        finally:
            prep.np.load = original_load
            prep.FROZEN_RAVDESS_INVENTORY = original_expectation
        c.true(
            isinstance(observed, ValueError),
            "coordinated valid-HMAC cumulative frame inflation fails closed",
        )
        c.eq(load_calls, 0, "aggregate frames/bytes are bounded before np.load")


def test_authorizer_rejects_cumulative_expanded_bytes_before_numpy_load(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "ravdess"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        key_path = data_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        key_path.write_bytes(TEST_ID_KEY)
        key_path.chmod(0o600)
        output = data_root / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root,
            output,
            inventory,
            expectation=expected,
            id_key=TEST_ID_KEY,
        )
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        row = manifest["trials"][0]
        cache_path = output / "trials" / f"{row['trial_id']}.npz"
        padded = _pad_first_npy_header(cache_path.read_bytes())
        cache_path.write_bytes(padded)
        cache_path.chmod(0o600)
        cache_sha256 = hashlib.sha256(padded).hexdigest()
        row["cache_integrity_id"] = prep._opaque_cache_integrity_id(
            cache_sha256,
            trial_id=row["trial_id"],
            actor_id=row["actor_id"],
            key=TEST_ID_KEY,
        )
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8"
        )
        manifest_path.chmod(0o600)

        original_expectation = prep.FROZEN_RAVDESS_INVENTORY
        original_load = prep.np.load
        load_calls = 0

        def materialized_before_byte_gate(*args, **kwargs):
            nonlocal load_calls
            load_calls += 1
            raise AssertionError("np.load reached before cumulative byte gate")

        observed: BaseException | None = None
        prep.FROZEN_RAVDESS_INVENTORY = expected
        prep.np.load = materialized_before_byte_gate
        try:
            try:
                prep.authorize_committed_ravdess_semantic23(data_root)
            except BaseException as exc:  # noqa: BLE001 - inspect gate ordering
                observed = exc
        finally:
            prep.np.load = original_load
            prep.FROZEN_RAVDESS_INVENTORY = original_expectation
        c.true(
            isinstance(observed, ValueError),
            "valid-HMAC cumulative NPY header padding fails closed",
        )
        c.eq(load_calls, 0, "aggregate expanded bytes are bounded before np.load")


def test_authorizer_rejects_cumulative_regular_payload_before_numpy_load(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "ravdess"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        key_path = data_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        key_path.write_bytes(TEST_ID_KEY)
        key_path.chmod(0o600)
        output = data_root / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root,
            output,
            inventory,
            expectation=expected,
            id_key=TEST_ID_KEY,
        )
        cache_paths = sorted((output / "trials").glob("*.npz"))
        manifest_path = output / "manifest.json"
        total_regular_payload = (
            manifest_path.stat().st_size
            + sum(path.stat().st_size for path in cache_paths)
        )
        limit_name = "_MAX_RAVDESS_AGGREGATE_REGULAR_PAYLOAD_BYTES"
        original_limit = getattr(prep, limit_name, None)
        original_expectation = prep.FROZEN_RAVDESS_INVENTORY
        original_load = prep.np.load
        load_calls = 0

        def materialized_before_raw_gate(*args, **kwargs):
            nonlocal load_calls
            load_calls += 1
            raise AssertionError("np.load reached before cumulative raw-byte gate")

        setattr(prep, limit_name, total_regular_payload - 1)
        prep.FROZEN_RAVDESS_INVENTORY = expected
        prep.np.load = materialized_before_raw_gate
        observed: BaseException | None = None
        try:
            try:
                prep.authorize_committed_ravdess_semantic23(data_root)
            except BaseException as exc:  # noqa: BLE001 - inspect gate ordering
                observed = exc
        finally:
            prep.np.load = original_load
            prep.FROZEN_RAVDESS_INVENTORY = original_expectation
            if original_limit is None:
                delattr(prep, limit_name)
            else:
                setattr(prep, limit_name, original_limit)
        c.true(
            isinstance(observed, ValueError),
            "coordinated cumulative regular-file payload overflow fails closed",
        )
        c.eq(
            load_calls, 0,
            "manifest plus cache payload bytes are bounded before np.load",
        )
        c.eq(
            original_limit,
            128 * 1024 * 1024,
            "production aggregate regular-file payload contract is exactly 128 MiB",
        )


def test_ravdess_actual_central_record_count_is_bounded_before_zipfile(c: Check):
    payloads = (
        _with_repeated_central_records_and_declared_count(
            _ravdess_cache_bytes(), actual_record_count=11
        ),
        _with_repeated_central_records_and_declared_count(
            _ravdess_cache_bytes(), actual_record_count=5_000
        ),
    )
    original_zip_file = prep.zipfile.ZipFile
    zipfile_calls: list[str] = []

    def zipfile_reached_too_early(*_args, **_kwargs):
        zipfile_calls.append("ZipFile/infolist")
        raise AssertionError("ZipFile/infolist reached before bounded central parsing")

    prep.zipfile.ZipFile = zipfile_reached_too_early
    try:
        for payload in payloads:
            c.raises(
                lambda value=payload: _ravdess_validate_without_materializing(value),
                ValueError,
                "the actual central-record count must match the fixed schema",
            )
    finally:
        prep.zipfile.ZipFile = original_zip_file
    c.eq(
        zipfile_calls,
        [],
        "crafted central directories are rejected before ZipFile or infolist",
    )


def test_ravdess_npy_dtype_and_shape_are_rejected_before_numpy_load(c: Check):
    attacks = (
        ("wrong feature dtype", _ravdess_cache_bytes(features_dtype=np.float64)),
        ("wrong feature shape", _ravdess_cache_bytes(feature_width=22)),
    )
    original_load = prep.np.load

    def materialized_too_early(*_args, **_kwargs):
        raise RuntimeError("np.load reached before NPY header validation")

    prep.np.load = materialized_too_early
    try:
        for label, payload in attacks:
            c.raises(
                lambda value=payload: _ravdess_validate_without_materializing(value),
                ValueError,
                f"RAVDESS {label} is rejected from its NPY header",
            )
    finally:
        prep.np.load = original_load


def test_ravdess_raw_cache_limit_is_checked_before_read(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "oversized.npz"
        path.write_bytes(b"x" * 65)
        path.chmod(0o600)
        original_read = prep.os.read

        def read_must_not_run(*_args, **_kwargs):
            raise RuntimeError("oversized raw file was read")

        prep.os.read = read_must_not_run
        try:
            c.raises(
                lambda: prep._read_owner_only_regular(
                    path, "RAVDESS semantic23 cache", max_bytes=64
                ),
                ValueError,
                "RAVDESS raw cache size is gated from fstat before reading",
            )
        finally:
            prep.os.read = original_read


def test_ravdess_manifest_limit_is_checked_before_read(c: Check):
    c.eq(
        getattr(prep, "_MAX_RAVDESS_MANIFEST_BYTES", None),
        4 * 1024 * 1024,
        "RAVDESS committed manifest has an exact 4 MiB raw-byte cap",
    )
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "ravdess"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        key_path = data_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        key_path.write_bytes(TEST_ID_KEY)
        key_path.chmod(0o600)
        output = data_root / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root, output, inventory, expectation=expected,
            id_key=TEST_ID_KEY,
        )
        manifest_path = output / "manifest.json"
        manifest_path.write_bytes(b"{" + b" " * (4 * 1024 * 1024))
        manifest_path.chmod(0o600)
        original_expectation = prep.FROZEN_RAVDESS_INVENTORY
        original_regular = prep._read_owner_only_regular
        observed_limits: list[int | None] = []
        manifest_reads = 0

        def tracked_regular(path, field, **kwargs):
            nonlocal manifest_reads
            if field != "RAVDESS manifest":
                return original_regular(path, field, **kwargs)
            observed_limits.append(kwargs.get("max_bytes"))
            original_read = prep.os.read

            def tracked_read(*args, **read_kwargs):
                nonlocal manifest_reads
                manifest_reads += 1
                return original_read(*args, **read_kwargs)

            prep.os.read = tracked_read
            try:
                return original_regular(path, field, **kwargs)
            finally:
                prep.os.read = original_read

        prep.FROZEN_RAVDESS_INVENTORY = expected
        prep._read_owner_only_regular = tracked_regular
        try:
            c.raises(
                lambda: prep.authorize_committed_ravdess_semantic23(data_root),
                ValueError,
                "oversized RAVDESS manifest fails closed",
            )
        finally:
            prep._read_owner_only_regular = original_regular
            prep.FROZEN_RAVDESS_INVENTORY = original_expectation
        c.eq(observed_limits, [4 * 1024 * 1024])
        c.eq(manifest_reads, 0, "oversized manifest is rejected from fstat before read")


def test_ravdess_authorizer_reads_npz_members_from_held_directory_fd(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "ravdess"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        key_path = data_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        key_path.write_bytes(TEST_ID_KEY)
        key_path.chmod(0o600)
        output = data_root / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root, output, inventory, expectation=expected,
            id_key=TEST_ID_KEY,
        )
        original_expectation = prep.FROZEN_RAVDESS_INVENTORY
        original_read = prep._read_owner_only_regular
        cache_reads: list[tuple[Path, int | None]] = []

        def tracked_read(path, field, **kwargs):
            if field == "RAVDESS semantic23 cache":
                cache_reads.append((Path(path), kwargs.get("parent_descriptor")))
            return original_read(path, field, **kwargs)

        prep.FROZEN_RAVDESS_INVENTORY = expected
        prep._read_owner_only_regular = tracked_read
        try:
            prep.authorize_committed_ravdess_semantic23(data_root)
        finally:
            prep._read_owner_only_regular = original_read
            prep.FROZEN_RAVDESS_INVENTORY = original_expectation
        c.eq(len(cache_reads), 2, "both committed trial caches were read")
        c.true(
            all(path.name == str(path) and isinstance(descriptor, int)
                for path, descriptor in cache_reads),
            "every RAVDESS NPZ read uses a basename plus held trials directory FD",
        )


def test_ravdess_authorizer_attempts_every_fd_close_after_one_failure(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "ravdess"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        key_path = data_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        key_path.write_bytes(TEST_ID_KEY)
        key_path.chmod(0o600)
        output = data_root / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root, output, inventory, expectation=expected,
            id_key=TEST_ID_KEY,
        )
        original_expectation = prep.FROZEN_RAVDESS_INVENTORY
        original_parent = prep._open_output_parent
        original_directory = prep._open_directory_at
        original_snapshot = prep._assert_owner_snapshot_at
        original_close = prep.os.close
        expected_descriptors: list[int] = []
        close_attempts: list[int] = []
        fail_descriptor: int | None = None
        injected_close = False

        def tracked_parent(path):
            descriptor, identity = original_parent(path)
            expected_descriptors.append(descriptor)
            return descriptor, identity

        def tracked_directory(parent_descriptor, name, field):
            nonlocal fail_descriptor
            descriptor = original_directory(parent_descriptor, name, field)
            expected_descriptors.append(descriptor)
            if field == "committed RAVDESS trial cache":
                fail_descriptor = descriptor
            return descriptor

        def fail_primary(*_args, **_kwargs):
            raise RuntimeError("primary RAVDESS authorization failure")

        def tracked_close(descriptor):
            nonlocal injected_close
            close_attempts.append(descriptor)
            original_close(descriptor)
            if descriptor == fail_descriptor and not injected_close:
                injected_close = True
                raise OSError("synthetic RAVDESS descriptor close failure")

        prep.FROZEN_RAVDESS_INVENTORY = expected
        prep._open_output_parent = tracked_parent
        prep._open_directory_at = tracked_directory
        prep._assert_owner_snapshot_at = fail_primary
        prep.os.close = tracked_close
        caught: BaseException | None = None
        try:
            try:
                prep.authorize_committed_ravdess_semantic23(data_root)
            except BaseException as exc:  # noqa: BLE001 - inspect cleanup chain
                caught = exc
        finally:
            prep.os.close = original_close
            prep._assert_owner_snapshot_at = original_snapshot
            prep._open_directory_at = original_directory
            prep._open_output_parent = original_parent
            prep.FROZEN_RAVDESS_INVENTORY = original_expectation
            for descriptor in expected_descriptors:
                if descriptor not in close_attempts:
                    try:
                        original_close(descriptor)
                    except OSError:
                        pass

        def chain_contains(error: BaseException | None, text: str) -> bool:
            seen: set[int] = set()
            while error is not None and id(error) not in seen:
                seen.add(id(error))
                if text in str(error):
                    return True
                error = error.__cause__ or error.__context__
            return False

        c.true(injected_close, "one RAVDESS held-descriptor close failure was injected")
        c.true(
            set(expected_descriptors).issubset(set(close_attempts)),
            "every RAVDESS held descriptor close is attempted after one failure",
        )
        c.true(
            chain_contains(caught, "primary RAVDESS authorization failure"),
            "RAVDESS descriptor cleanup preserves the primary exception chain",
        )


def test_committed_ravdess_authorizer_recomputes_keyed_closure_and_fails_closed(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "ravdess"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        key_path = data_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        key_path.write_bytes(TEST_ID_KEY)
        key_path.chmod(0o600)
        output = data_root / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root, output, inventory, expectation=expected,
            id_key=TEST_ID_KEY,
        )
        before = {
            path.relative_to(data_root): (
                path.read_bytes(), stat.S_IMODE(path.stat().st_mode),
                path.stat().st_mtime_ns,
            )
            for path in (key_path, output / "manifest.json", *(output / "trials").glob("*.npz"))
        }
        original = prep.FROZEN_RAVDESS_INVENTORY
        prep.FROZEN_RAVDESS_INVENTORY = expected
        try:
            authorized = prep.authorize_committed_ravdess_semantic23(data_root)
            c.eq(authorized.trial_count, 2)
            c.eq(authorized.actor_count, 2)
            c.eq(authorized.source_frames, 3)
            c.eq(len(authorized.trials), 2)
            c.true(bool(authorized.generation_closure_hmac))
            c.eq(
                {item.relative_to(data_root): (
                    item.read_bytes(), stat.S_IMODE(item.stat().st_mode),
                    item.stat().st_mtime_ns,
                ) for item in (
                    key_path, output / "manifest.json", *(output / "trials").glob("*.npz")
                )},
                before,
                "authorization is read-only",
            )

            cache = sorted((output / "trials").glob("*.npz"))[0]
            original_cache = cache.read_bytes()
            cache.write_bytes(original_cache[:-1] + bytes([original_cache[-1] ^ 1]))
            cache.chmod(0o600)
            c.raises(
                lambda: prep.authorize_committed_ravdess_semantic23(data_root),
                ValueError,
                "a changed cache byte invalidates the private keyed closure",
            )
            cache.write_bytes(original_cache)
            cache.chmod(0o600)

            key_path.chmod(0o640)
            c.raises(
                lambda: prep.authorize_committed_ravdess_semantic23(data_root),
                ValueError,
                "the canonical key must remain owner-only",
            )
            key_path.chmod(0o600)

            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text("utf-8"))
            manifest["raw_path"] = "/private/source.csv"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            manifest_path.chmod(0o600)
            c.raises(
                lambda: prep.authorize_committed_ravdess_semantic23(data_root),
                ValueError,
                "manifest fields and privacy are exact",
            )
        finally:
            prep.FROZEN_RAVDESS_INVENTORY = original


def test_committed_ravdess_authorizer_requires_exact_generation_tree(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "ravdess"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        key_path = data_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        key_path.write_bytes(TEST_ID_KEY)
        key_path.chmod(0o600)
        output = data_root / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root, output, inventory, expectation=expected,
            id_key=TEST_ID_KEY,
        )
        unexpected = output / "unexpected-private.bin"
        original_expectation = prep.FROZEN_RAVDESS_INVENTORY
        prep.FROZEN_RAVDESS_INVENTORY = expected
        try:
            unexpected.write_bytes(b"private residue")
            unexpected.chmod(0o600)
            c.raises(
                lambda: prep.authorize_committed_ravdess_semantic23(data_root),
                ValueError,
                "a preexisting extra generation-root file fails closed",
            )
            unexpected.unlink()

            original_snapshot = prep._assert_owner_snapshot_at
            injected = False

            def add_root_residue_during_final_recheck(*args, **kwargs):
                nonlocal injected
                result = original_snapshot(*args, **kwargs)
                if not injected:
                    unexpected.write_bytes(b"late private residue")
                    unexpected.chmod(0o600)
                    injected = True
                return result

            prep._assert_owner_snapshot_at = add_root_residue_during_final_recheck
            try:
                c.raises(
                    lambda: prep.authorize_committed_ravdess_semantic23(data_root),
                    ValueError,
                    "a late extra generation-root file fails before authorization",
                )
            finally:
                prep._assert_owner_snapshot_at = original_snapshot
            c.true(injected, "late generation-root residue was injected")
        finally:
            prep.FROZEN_RAVDESS_INVENTORY = original_expectation


def test_committed_ravdess_authorizer_rechecks_cache_after_archive_reaudit(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "ravdess"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        key_path = data_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        key_path.write_bytes(TEST_ID_KEY)
        key_path.chmod(0o600)
        output = data_root / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root, output, inventory, expectation=expected,
            id_key=TEST_ID_KEY,
        )
        cache = sorted((output / "trials").glob("*.npz"))[0]
        original_cache = cache.read_bytes()
        original_expectation = prep.FROZEN_RAVDESS_INVENTORY
        original_audit = prep.audit_ravdess_inventory
        audit_calls = 0

        def mutate_cache_during_second_archive_audit(*args, **kwargs):
            nonlocal audit_calls
            result = original_audit(*args, **kwargs)
            audit_calls += 1
            if audit_calls == 2:
                cache.write_bytes(
                    original_cache[:-1] + bytes([original_cache[-1] ^ 1])
                )
                cache.chmod(0o600)
            return result

        prep.FROZEN_RAVDESS_INVENTORY = expected
        prep.audit_ravdess_inventory = mutate_cache_during_second_archive_audit
        try:
            c.raises(
                lambda: prep.authorize_committed_ravdess_semantic23(data_root),
                ValueError,
                "cache mutation during the final raw-archive audit fails closed",
            )
        finally:
            prep.audit_ravdess_inventory = original_audit
            prep.FROZEN_RAVDESS_INVENTORY = original_expectation
        c.eq(audit_calls, 2, "authorization performed both archive audits")


def test_committed_ravdess_manifest_rejects_bool_int_type_aliases(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "ravdess"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        key_path = data_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        key_path.write_bytes(TEST_ID_KEY)
        key_path.chmod(0o600)
        output = data_root / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root, output, inventory, expectation=expected,
            id_key=TEST_ID_KEY,
        )
        manifest_path = output / "manifest.json"
        original_bytes = manifest_path.read_bytes()
        cases = (
            (("format_version",), True),
            (("inventory", "empty_trials"), False),
            (("timeline_policy", "source_rows_preserved"), 1),
            (("provenance_policy", "raw_paths_or_filenames_in_manifest"), 0),
        )
        original_expectation = prep.FROZEN_RAVDESS_INVENTORY
        prep.FROZEN_RAVDESS_INVENTORY = expected
        try:
            for path, replacement in cases:
                manifest = json.loads(original_bytes.decode("utf-8"))
                target = manifest
                for component in path[:-1]:
                    target = target[component]
                target[path[-1]] = replacement
                manifest_path.write_text(
                    json.dumps(manifest, sort_keys=True), encoding="utf-8",
                )
                manifest_path.chmod(0o600)
                c.raises(
                    lambda: prep.authorize_committed_ravdess_semantic23(data_root),
                    ValueError,
                    f"RAVDESS manifest type alias is rejected at {'.'.join(path)}",
                )
        finally:
            manifest_path.write_bytes(original_bytes)
            manifest_path.chmod(0o600)
            prep.FROZEN_RAVDESS_INVENTORY = original_expectation


def test_all_v2_inventory_topology_fields_are_individually_exact(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "ravdess"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        key_path = data_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        key_path.write_bytes(TEST_ID_KEY)
        key_path.chmod(0o600)
        output = data_root / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root,
            output,
            inventory,
            expectation=expected,
            id_key=TEST_ID_KEY,
        )
        manifest_path = output / "manifest.json"
        original_bytes = manifest_path.read_bytes()
        observations: list[tuple[str, str, str]] = []
        original_expectation = prep.FROZEN_RAVDESS_INVENTORY
        prep.FROZEN_RAVDESS_INVENTORY = expected
        try:
            for field in RAVDESS_TOPOLOGY_FIELDS:
                original = json.loads(original_bytes.decode("utf-8"))
                expected_value = original["inventory"][field]
                for mutation in (
                    "missing", "extra", "bool", "wrong_type", "value_drift"
                ):
                    manifest = json.loads(original_bytes.decode("utf-8"))
                    topology = manifest["inventory"]
                    if mutation == "missing":
                        topology.pop(field)
                    elif mutation == "extra":
                        topology[f"unexpected_{field}"] = expected_value
                    elif mutation == "bool":
                        topology[field] = True
                    elif mutation == "wrong_type":
                        topology[field] = str(expected_value)
                    else:
                        topology[field] = expected_value + 1
                    manifest_path.write_text(
                        json.dumps(manifest, sort_keys=True), encoding="utf-8"
                    )
                    manifest_path.chmod(0o600)
                    observed, _ = _capture_failure(
                        lambda: prep.authorize_committed_ravdess_semantic23(
                            data_root
                        )
                    )
                    observations.append((
                        field,
                        mutation,
                        type(observed).__name__ if observed else "none",
                    ))
        finally:
            manifest_path.write_bytes(original_bytes)
            manifest_path.chmod(0o600)
            prep.FROZEN_RAVDESS_INVENTORY = original_expectation
        c.eq(
            len(observations),
            len(RAVDESS_TOPOLOGY_FIELDS) * 5,
            "all six topology keys run every exactness mutation",
        )
        c.true(
            all(kind == ValueError.__name__ for _, _, kind in observations),
            "every missing/extra/bool/type/value topology mutation is rejected",
        )


def test_committed_ravdess_authorizer_rebuilds_exact_archive_actor_join(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "ravdess"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        key_path = data_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        key_path.write_bytes(TEST_ID_KEY)
        key_path.chmod(0o600)
        output = data_root / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root, output, inventory, expectation=expected,
            id_key=TEST_ID_KEY,
        )
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        c.eq(len(manifest["trials"]), 2)
        first_actor = manifest["trials"][0]["actor_id"]
        second_actor = manifest["trials"][1]["actor_id"]
        c.true(first_actor != second_actor)
        manifest["trials"][0]["actor_id"] = second_actor
        manifest["trials"][1]["actor_id"] = first_actor
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")),
            encoding="ascii",
        )
        manifest_path.chmod(0o600)
        original = prep.FROZEN_RAVDESS_INVENTORY
        prep.FROZEN_RAVDESS_INVENTORY = expected
        try:
            c.raises(
                lambda: prep.authorize_committed_ravdess_semantic23(data_root),
                ValueError,
                "self-consistent manifest actor swaps cannot rewrite the live archive join",
            )
        finally:
            prep.FROZEN_RAVDESS_INVENTORY = original


def test_committed_ravdess_authorizer_rejects_coordinated_cache_and_hmac_swap(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "ravdess"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        key_path = data_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        key_path.write_bytes(TEST_ID_KEY)
        key_path.chmod(0o600)
        output = data_root / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root, output, inventory, expectation=expected,
            id_key=TEST_ID_KEY,
        )
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        first, second = manifest["trials"]
        first_cache = output / "trials" / f"{first['trial_id']}.npz"
        second_cache = output / "trials" / f"{second['trial_id']}.npz"
        first_bytes = first_cache.read_bytes()
        second_bytes = second_cache.read_bytes()
        first_cache.write_bytes(second_bytes)
        second_cache.write_bytes(first_bytes)
        first["cache_integrity_id"], second["cache_integrity_id"] = (
            second["cache_integrity_id"], first["cache_integrity_id"]
        )
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")),
            encoding="ascii",
        )
        manifest_path.chmod(0o600)
        original = prep.FROZEN_RAVDESS_INVENTORY
        prep.FROZEN_RAVDESS_INVENTORY = expected
        try:
            c.raises(
                lambda: prep.authorize_committed_ravdess_semantic23(data_root),
                ValueError,
                "cache HMACs bind bytes to the exact live trial and actor",
            )
        finally:
            prep.FROZEN_RAVDESS_INVENTORY = original


def test_committed_ravdess_authorizer_rejects_gap_even_with_valid_cache_hmac(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "ravdess"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        key_path = data_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        key_path.write_bytes(TEST_ID_KEY)
        key_path.chmod(0o600)
        output = data_root / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root, output, inventory, expectation=expected,
            id_key=TEST_ID_KEY,
        )
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        row = None
        for item in manifest["trials"]:
            with np.load(
                output / "trials" / f"{item['trial_id']}.npz",
                allow_pickle=False,
            ) as candidate:
                if candidate["frame_indices"].shape[0] == 2:
                    row = item
                    break
        if row is None:
            raise AssertionError("synthetic fixture must contain a two-frame trial")
        cache_path = output / "trials" / f"{row['trial_id']}.npz"
        with np.load(cache_path, allow_pickle=False) as cached:
            arrays = {name: np.asarray(cached[name]) for name in cached.files}
        arrays["frame_indices"] = np.asarray([0, 2], dtype=np.int64)
        np.savez(cache_path, **arrays)
        cache_path.chmod(0o600)
        cache_sha256 = hashlib.sha256(cache_path.read_bytes()).hexdigest()
        row["cache_integrity_id"] = prep._opaque_cache_integrity_id(
            cache_sha256,
            trial_id=row["trial_id"],
            actor_id=row["actor_id"],
            key=TEST_ID_KEY,
        )
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")),
            encoding="ascii",
        )
        manifest_path.chmod(0o600)
        original = prep.FROZEN_RAVDESS_INVENTORY
        prep.FROZEN_RAVDESS_INVENTORY = expected
        try:
            c.raises(
                lambda: prep.authorize_committed_ravdess_semantic23(data_root),
                ValueError,
                "RAVDESS frame indices must be contiguous before 30 Hz bridging",
            )
        finally:
            prep.FROZEN_RAVDESS_INVENTORY = original


def test_committed_ravdess_authorizer_requires_existing_lock_without_mutation(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "ravdess"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        key_path = data_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        key_path.write_bytes(TEST_ID_KEY)
        key_path.chmod(0o600)
        output = data_root / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root, output, inventory, expectation=expected,
            id_key=TEST_ID_KEY,
        )
        lock = output.parent / f".{output.name}.lock"
        c.true(lock.is_file())
        original = prep.FROZEN_RAVDESS_INVENTORY
        prep.FROZEN_RAVDESS_INVENTORY = expected
        try:
            lock.unlink()
            before = {
                path.relative_to(data_root): (
                    path.read_bytes(), stat.S_IMODE(path.stat().st_mode),
                    path.stat().st_mtime_ns,
                )
                for path in data_root.rglob("*") if path.is_file()
            }
            c.raises(
                lambda: prep.authorize_committed_ravdess_semantic23(data_root),
                ValueError,
                "read-only authorization rejects a missing producer lock",
            )
            c.true(not lock.exists(), "read-only authorization never O_CREATs a lock")
            c.eq({
                path.relative_to(data_root): (
                    path.read_bytes(), stat.S_IMODE(path.stat().st_mode),
                    path.stat().st_mtime_ns,
                )
                for path in data_root.rglob("*") if path.is_file()
            }, before, "a rejected read-only authorization leaves every file unchanged")
        finally:
            prep.FROZEN_RAVDESS_INVENTORY = original


def test_committed_ravdess_authorizer_requires_single_link_key(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "ravdess"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        key_path = data_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        key_path.write_bytes(TEST_ID_KEY)
        key_path.chmod(0o600)
        output = data_root / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root, output, inventory, expectation=expected,
            id_key=TEST_ID_KEY,
        )
        original = prep.FROZEN_RAVDESS_INVENTORY
        prep.FROZEN_RAVDESS_INVENTORY = expected
        try:
            hardlink = data_root / ".hardlinked-private-key"
            os.link(key_path, hardlink)
            c.eq(key_path.stat().st_nlink, 2)
            c.raises(
                lambda: prep.authorize_committed_ravdess_semantic23(data_root),
                ValueError,
                "a multiply-linked private key is never authorized",
            )
        finally:
            prep.FROZEN_RAVDESS_INVENTORY = original


def test_committed_ravdess_authorizer_rejects_transaction_residue(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "ravdess"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        key_path = data_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        key_path.write_bytes(TEST_ID_KEY)
        key_path.chmod(0o600)
        output = data_root / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root, output, inventory, expectation=expected,
            id_key=TEST_ID_KEY,
        )
        residue = output.parent / f".{output.name}.staging-interrupted"
        residue.mkdir()
        original = prep.FROZEN_RAVDESS_INVENTORY
        prep.FROZEN_RAVDESS_INVENTORY = expected
        try:
            c.raises(
                lambda: prep.authorize_committed_ravdess_semantic23(data_root),
                RuntimeError,
                "an unresolved producer transaction cannot authorize a generation",
            )
            c.true(residue.is_dir(), "read-only authorization never cleans residue")
        finally:
            prep.FROZEN_RAVDESS_INVENTORY = original


def test_committed_ravdess_authorizer_rejects_live_archive_drift(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "ravdess"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        key_path = data_root / prep.RAVDESS_ID_KEY_RELATIVE_PATH
        key_path.write_bytes(TEST_ID_KEY)
        key_path.chmod(0o600)
        output = data_root / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root, output, inventory, expectation=expected,
            id_key=TEST_ID_KEY,
        )
        archive = data_root / RAVDESS_ARCHIVE_RELATIVE_PATH
        with archive.open("ab") as handle:
            handle.write(b"live-root-drift")
        original = prep.FROZEN_RAVDESS_INVENTORY
        prep.FROZEN_RAVDESS_INVENTORY = expected
        try:
            c.raises(
                lambda: prep.authorize_committed_ravdess_semantic23(data_root),
                ValueError,
                "the archive behind the live root must still match the frozen inventory",
            )
        finally:
            prep.FROZEN_RAVDESS_INVENTORY = original


if __name__ == "__main__":
    run_all("test_openface68_semantic", dict(globals()))
