#!/usr/bin/env python3
"""Build a read-only, aggregate-only Mayo action-anchor feasibility audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "mayo_action_anchor_feasibility_v1"
MEDIA_EXTENSIONS = frozenset({".mov", ".mp4"})
ACTIONS = (
    "EYEBROW_RAISE",
    "GENTLE_EYE_CLOSURE",
    "TIGHT_EYE_SQUEEZE",
    "RELAXED_SMILE",
    "LIP_PUCKER",
    "LOWER_TEETH_SHOW",
)
_SIDECAR_BASENAMES = {
    "capture_event_log.json": "capture_event_log",
    "audio_forced_alignment.json": "audio_forced_alignment",
    "blinded_manual_action_timing.json": "blinded_manual",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class MediaRecord:
    path: Path
    source_sha256: str
    source_file_count: int
    size_bytes: int
    duration_seconds: float | None
    has_audio: bool | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.path, Path) or not self.path.is_absolute()
            or not isinstance(self.source_sha256, str)
            or _SHA256.fullmatch(self.source_sha256) is None
            or isinstance(self.source_file_count, bool)
            or not isinstance(self.source_file_count, int)
            or self.source_file_count <= 0
            or isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int) or self.size_bytes <= 0
            or not (self.has_audio is None or type(self.has_audio) is bool)
        ):
            raise ValueError("media record identity or metadata is malformed")
        if self.duration_seconds is not None and (
            isinstance(self.duration_seconds, bool)
            or not isinstance(self.duration_seconds, (int, float))
            or not np.isfinite(float(self.duration_seconds))
            or float(self.duration_seconds) <= 0.0
        ):
            raise ValueError("media duration must be positive finite seconds")


@dataclass(frozen=True)
class MayoMediaInventory:
    source_files: int
    unique_contents: int
    exact_duplicate_files: int
    audio_bearing_source_files: int
    audio_free_source_files: int
    audio_unknown_source_files: int
    audio_bearing_unique_contents: int
    records: tuple[MediaRecord, ...]


@dataclass(frozen=True)
class TimingEvent:
    recording_slot: int
    source_sha256: str
    action: str
    start_ms: int
    end_ms: int
    prompted_flat: bool
    prompted_flat_manually_verified: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.recording_slot, bool)
            or not isinstance(self.recording_slot, int)
            or not 0 <= self.recording_slot < 12
            or not isinstance(self.source_sha256, str)
            or _SHA256.fullmatch(self.source_sha256) is None
            or self.action not in ACTIONS
            or isinstance(self.start_ms, bool) or not isinstance(self.start_ms, int)
            or isinstance(self.end_ms, bool) or not isinstance(self.end_ms, int)
            or self.start_ms < 0 or self.end_ms <= self.start_ms
            or type(self.prompted_flat) is not bool
            or type(self.prompted_flat_manually_verified) is not bool
            or self.prompted_flat and not self.prompted_flat_manually_verified
        ):
            raise ValueError("timing event differs from the frozen 12-by-6 contract")


@dataclass(frozen=True)
class TimingGateEvidence:
    registry_payload: bytes
    annotation_audit_payload: bytes
    reference_events: tuple[TimingEvent, ...]
    predicted_events: tuple[TimingEvent, ...]
    summary: Mapping[str, object]


def _read_and_hash_regular_file(path: Path) -> tuple[str, int]:
    candidate = Path(path)
    try:
        descriptor = os.open(candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ValueError("media inventory member is not a readable regular file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size <= 0:
            raise ValueError("media inventory member has an unsafe file identity")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        leaf = os.lstat(candidate)
        identity = lambda value: (
            value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
            value.st_size, value.st_mtime_ns, value.st_ctime_ns,
        )
        if total != before.st_size or identity(before) != identity(after) or identity(after) != identity(leaf):
            raise ValueError("media inventory member changed while hashing")
        return digest.hexdigest(), total
    finally:
        os.close(descriptor)


def _probe_committed_regular_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    probe: Callable[[Path], Mapping[str, object]],
) -> dict[str, object]:
    candidate = Path(path)
    try:
        descriptor = os.open(candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ValueError("committed media probe input is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or before.st_size != expected_size
        ):
            raise ValueError("committed media probe identity differs from inventory")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("media content changed between inventory and probe")
        os.lseek(descriptor, 0, os.SEEK_SET)
        metadata = dict(probe(Path(f"/dev/fd/{descriptor}")))
        after = os.fstat(descriptor)
        leaf = os.lstat(candidate)
        identity = lambda value: (
            value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
            value.st_size, value.st_mtime_ns, value.st_ctime_ns,
        )
        if identity(before) != identity(after) or identity(after) != identity(leaf):
            raise ValueError("media identity changed during committed ffprobe")
        return metadata
    finally:
        os.close(descriptor)


def _ffprobe(path: Path) -> dict[str, object]:
    try:
        descriptor = int(path.name) if path.parent == Path("/dev/fd") else None
        completed = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration:stream=codec_type", "-of", "json", os.fspath(path),
            ],
            check=True, capture_output=True, text=True, timeout=30.0, shell=False,
            pass_fds=() if descriptor is None else (descriptor,),
        )
        value = json.loads(completed.stdout)
        duration = float(value["format"]["duration"])
        streams = value.get("streams")
        if not isinstance(streams, list) or not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("ffprobe returned malformed media metadata")
        return {
            "duration_seconds": duration,
            "has_audio": any(
                isinstance(stream, dict) and stream.get("codec_type") == "audio"
                for stream in streams
            ),
        }
    except (OSError, subprocess.SubprocessError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return {"duration_seconds": None, "has_audio": None}


def inventory_media(
    root: Path,
    *,
    probe: Callable[[Path], Mapping[str, object]] = _ffprobe,
) -> MayoMediaInventory:
    media_root = Path(root)
    try:
        root_metadata = os.lstat(media_root)
    except OSError as exc:
        raise ValueError("Mayo media root is unavailable") from exc
    if not media_root.is_absolute() or not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError("Mayo media root must be an absolute non-symlink directory")
    paths = sorted(
        path for path in media_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in MEDIA_EXTENSIONS
    )
    if not paths:
        raise ValueError("Mayo media root contains no supported videos")
    grouped: dict[str, list[tuple[Path, int]]] = {}
    for path in paths:
        digest, size = _read_and_hash_regular_file(path)
        grouped.setdefault(digest, []).append((path.resolve(strict=True), size))
    records = []
    for digest in sorted(grouped):
        members = grouped[digest]
        sizes = {size for _, size in members}
        if len(sizes) != 1:
            raise ValueError("equal content digests unexpectedly disagree on byte length")
        metadata = _probe_committed_regular_file(
            members[0][0], expected_sha256=digest,
            expected_size=next(iter(sizes)), probe=probe,
        )
        duration = metadata.get("duration_seconds")
        has_audio = metadata.get("has_audio")
        record = MediaRecord(
            path=members[0][0], source_sha256=digest,
            source_file_count=len(members), size_bytes=next(iter(sizes)),
            duration_seconds=None if duration is None else float(duration),
            has_audio=has_audio if type(has_audio) is bool else None,
        )
        records.append(record)
    for expected_digest, members in grouped.items():
        for member_path, expected_size in members:
            observed_sha, observed_size = _read_and_hash_regular_file(member_path)
            if observed_sha != expected_digest or observed_size != expected_size:
                raise ValueError("a media member changed before inventory finalization")
    audio_files = sum(r.source_file_count for r in records if r.has_audio is True)
    audio_free = sum(r.source_file_count for r in records if r.has_audio is False)
    audio_unknown = sum(r.source_file_count for r in records if r.has_audio is None)
    return MayoMediaInventory(
        source_files=len(paths), unique_contents=len(records),
        exact_duplicate_files=len(paths) - len(records),
        audio_bearing_source_files=audio_files,
        audio_free_source_files=audio_free,
        audio_unknown_source_files=audio_unknown,
        audio_bearing_unique_contents=sum(r.has_audio is True for r in records),
        records=tuple(records),
    )


def discover_timing_sidecars(root: Path) -> dict[str, int]:
    counts = {value: 0 for value in _SIDECAR_BASENAMES.values()}
    for path in Path(root).rglob("*"):
        kind = _SIDECAR_BASENAMES.get(path.name)
        if kind is not None and path.is_file() and not path.is_symlink():
            counts[kind] += 1
    return counts


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return (json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False,
    ) + "\n").encode("ascii")


def _unique_object(pairs):
    value = {}
    for key, child in pairs:
        if key in value:
            raise ValueError("audit JSON contains a duplicate key")
        value[key] = child
    return value


def _strict_canonical_json(payload: bytes, *, label: str) -> dict[str, object]:
    if type(payload) is not bytes or not payload or len(payload) > 8 * 1024 * 1024:
        raise ValueError(f"{label} must be bounded exact bytes")
    try:
        value = json.loads(
            payload.decode("ascii"), object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict canonical JSON") from exc
    if not isinstance(value, dict) or _canonical_json(value) != payload:
        raise ValueError(f"{label} differs from its canonical representation")
    return value


def _validated_private_registry(payload: bytes) -> tuple[dict[str, object], tuple[str, ...]]:
    registry = _strict_canonical_json(payload, label="private audit registry")
    expected_keys = {
        "schema_version", "selection_rule",
        "candidate_audio_bearing_unique_contents", "selected_count",
        "selected_source_sha256",
        "transcription_or_visual_inspection_performed_before_selection",
    }
    selected = registry.get("selected_source_sha256")
    candidates = registry.get("candidate_audio_bearing_unique_contents")
    if (
        set(registry) != expected_keys
        or registry.get("schema_version") != "mayo_action_anchor_private_registry_v1"
        or registry.get("selection_rule")
        != "12_lexicographically_smallest_audio_bearing_deduplicated_source_sha256"
        or type(candidates) is not int or candidates < 12
        or registry.get("selected_count") != 12
        or not isinstance(selected, list) or len(selected) != 12
        or any(not isinstance(value, str) or _SHA256.fullmatch(value) is None for value in selected)
        or selected != sorted(set(selected))
        or registry.get("transcription_or_visual_inspection_performed_before_selection") is not False
    ):
        raise ValueError("private audit registry differs from the frozen selection contract")
    return registry, tuple(selected)


def build_private_audit_registry(
    records: Sequence[MediaRecord],
) -> tuple[dict[str, object], bytes]:
    normalized = tuple(records)
    if not normalized or any(not isinstance(record, MediaRecord) for record in normalized):
        raise ValueError("private audit registry requires validated media records")
    if any(record.has_audio is None for record in normalized):
        raise ValueError("unknown audio metadata makes audit selection ambiguous")
    audio_hashes = sorted({
        record.source_sha256 for record in normalized if record.has_audio is True
    })
    if len(audio_hashes) < 12:
        raise ValueError("fewer than 12 audio-bearing unique Mayo recordings exist")
    registry = {
        "schema_version": "mayo_action_anchor_private_registry_v1",
        "selection_rule": "12_lexicographically_smallest_audio_bearing_deduplicated_source_sha256",
        "candidate_audio_bearing_unique_contents": len(audio_hashes),
        "selected_count": 12,
        "selected_source_sha256": audio_hashes[:12],
        "transcription_or_visual_inspection_performed_before_selection": False,
    }
    return registry, _canonical_json(registry)


def _stage_bytes_at(directory_fd: int, payload: bytes, *, mode: int) -> str:
    temporary = f".mayo-anchor-{secrets.token_hex(16)}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode, dir_fd=directory_fd,
    )
    succeeded = False
    try:
        os.fchmod(descriptor, mode)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing audit evidence")
            view = view[written:]
        os.fsync(descriptor)
        succeeded = True
    finally:
        os.close(descriptor)
        if not succeeded:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
    return temporary


def _open_output_parent(path: Path) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    return os.open(
        output.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )


def _unlink_if_same_as_staged(
    directory_fd: int,
    *,
    final_name: str,
    staged_name: str,
) -> None:
    try:
        final = os.stat(final_name, dir_fd=directory_fd, follow_symlinks=False)
        staged = os.stat(staged_name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if (final.st_dev, final.st_ino) == (staged.st_dev, staged.st_ino):
        os.unlink(final_name, dir_fd=directory_fd)


def _verify_published_leaf(
    descriptor: int,
    directory_fd: int,
    *,
    final_name: str,
    staged_name: str,
    payload: bytes,
    mode: int,
) -> None:
    opened = os.fstat(descriptor)
    final = os.stat(final_name, dir_fd=directory_fd, follow_symlinks=False)
    staged = os.stat(staged_name, dir_fd=directory_fd, follow_symlinks=False)
    identities = {
        (opened.st_dev, opened.st_ino),
        (final.st_dev, final.st_ino),
        (staged.st_dev, staged.st_ino),
    }
    if (
        len(identities) != 1 or opened.st_nlink != 2
        or final.st_nlink != 2 or staged.st_nlink != 2
        or opened.st_size != len(payload)
        or stat.S_IMODE(opened.st_mode) != mode
    ):
        raise ValueError("published audit artifact identity changed before finalization")
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    if digest.hexdigest() != hashlib.sha256(payload).hexdigest():
        raise ValueError("published audit artifact bytes changed before finalization")


def _write_bytes_no_overwrite(path: Path, payload: bytes, *, mode: int) -> str:
    if type(payload) is not bytes or not payload:
        raise ValueError("audit artifact payload must be exact nonempty bytes")
    output = Path(path)
    directory_fd = _open_output_parent(output)
    temporary = None
    published = False
    published_fd = None
    try:
        temporary = _stage_bytes_at(directory_fd, payload, mode=mode)
        os.link(
            temporary, output.name,
            src_dir_fd=directory_fd, dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        published = True
        published_fd = os.open(
            output.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
        _verify_published_leaf(
            published_fd, directory_fd,
            final_name=output.name, staged_name=temporary,
            payload=payload, mode=mode,
        )
        return hashlib.sha256(payload).hexdigest()
    except BaseException:
        if published and temporary is not None:
            _unlink_if_same_as_staged(
                directory_fd, final_name=output.name, staged_name=temporary,
            )
            published = False
        raise
    finally:
        if published_fd is not None:
            os.close(published_fd)
        if temporary is not None:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


def write_private_registry_no_overwrite(path: Path, payload: bytes) -> str:
    if type(payload) is not bytes or not payload:
        raise ValueError("private registry payload must be exact nonempty bytes")
    return _write_bytes_no_overwrite(path, payload, mode=0o600)


def write_audit_release_pair_no_overwrite(
    private_path: Path,
    private_payload: bytes,
    public_path: Path,
    public_payload: bytes,
) -> tuple[str, str]:
    if (
        type(private_payload) is not bytes or not private_payload
        or type(public_payload) is not bytes or not public_payload
    ):
        raise ValueError("paired audit publication requires exact nonempty bytes")
    private = Path(private_path)
    public = Path(public_path)
    private_fd = _open_output_parent(private)
    public_fd = _open_output_parent(public)
    private_tmp = public_tmp = None
    private_published = public_published = False
    private_leaf_fd = public_leaf_fd = None
    try:
        private_tmp = _stage_bytes_at(private_fd, private_payload, mode=0o600)
        public_tmp = _stage_bytes_at(public_fd, public_payload, mode=0o600)
        os.link(
            private_tmp, private.name,
            src_dir_fd=private_fd, dst_dir_fd=private_fd, follow_symlinks=False,
        )
        private_published = True
        private_leaf_fd = os.open(
            private.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=private_fd,
        )
        os.link(
            public_tmp, public.name,
            src_dir_fd=public_fd, dst_dir_fd=public_fd, follow_symlinks=False,
        )
        public_published = True
        public_leaf_fd = os.open(
            public.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=public_fd,
        )
        os.fsync(private_fd)
        os.fsync(public_fd)
        _verify_published_leaf(
            private_leaf_fd, private_fd,
            final_name=private.name, staged_name=private_tmp,
            payload=private_payload, mode=0o600,
        )
        _verify_published_leaf(
            public_leaf_fd, public_fd,
            final_name=public.name, staged_name=public_tmp,
            payload=public_payload, mode=0o600,
        )
        return (
            hashlib.sha256(private_payload).hexdigest(),
            hashlib.sha256(public_payload).hexdigest(),
        )
    except BaseException:
        if private_published and private_tmp is not None:
            _unlink_if_same_as_staged(
                private_fd, final_name=private.name, staged_name=private_tmp,
            )
            private_published = False
        if public_published and public_tmp is not None:
            _unlink_if_same_as_staged(
                public_fd, final_name=public.name, staged_name=public_tmp,
            )
            public_published = False
        raise
    finally:
        if private_leaf_fd is not None:
            os.close(private_leaf_fd)
        if public_leaf_fd is not None:
            os.close(public_leaf_fd)
        for directory_fd, temporary in (
            (private_fd, private_tmp), (public_fd, public_tmp),
        ):
            if temporary is not None:
                try:
                    os.unlink(temporary, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
        os.close(private_fd)
        os.close(public_fd)


def _temporal_iou(left: TimingEvent, right: TimingEvent) -> float:
    intersection = max(0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))
    union = max(left.end_ms, right.end_ms) - min(left.start_ms, right.start_ms)
    return float(intersection / union) if union > 0 else 0.0


def reference_events_sha256(reference_events: Sequence[TimingEvent]) -> str:
    references = tuple(reference_events)
    if len(references) != 72 or any(not isinstance(event, TimingEvent) for event in references):
        raise ValueError("reference commitment requires all 72 typed events")
    encoded = [{
        "recording_slot": event.recording_slot,
        "source_sha256": event.source_sha256,
        "action": event.action,
        "start_ms": event.start_ms,
        "end_ms": event.end_ms,
        "prompted_flat": event.prompted_flat,
        "prompted_flat_manually_verified": event.prompted_flat_manually_verified,
    } for event in references]
    return hashlib.sha256(_canonical_json({"reference_events": encoded})).hexdigest()


def evaluate_timing_gate(
    private_registry_payload: bytes,
    annotation_audit_payload: bytes,
    reference_events: Sequence[TimingEvent],
    predicted_events: Sequence[TimingEvent],
) -> TimingGateEvidence:
    _, selected_hashes = _validated_private_registry(private_registry_payload)
    registry_sha = hashlib.sha256(private_registry_payload).hexdigest()
    references = tuple(reference_events)
    predictions = tuple(predicted_events)
    if (
        len(references) != 72
        or any(not isinstance(event, TimingEvent) for event in references + predictions)
        or len({(event.recording_slot, event.action) for event in references}) != 72
    ):
        raise ValueError("timing audit requires the exact 12 recordings by six references")
    for event in references + predictions:
        if event.source_sha256 != selected_hashes[event.recording_slot]:
            raise ValueError("timing event is not bound to its hash-selected recording")
    if any(
        event.prompted_flat or event.prompted_flat_manually_verified
        for event in predictions
    ):
        raise ValueError("predicted events cannot assert manual prompted-flat evidence")
    reference_sha = reference_events_sha256(references)
    annotation_audit = _strict_canonical_json(
        annotation_audit_payload, label="blinded reference audit",
    )
    manually_verified_flat = sum(
        event.prompted_flat and event.prompted_flat_manually_verified
        for event in references
    )
    if (
        set(annotation_audit) != {
            "schema_version", "registry_sha256", "annotator_count",
            "annotators_blinded", "adjudication_complete",
            "boundary_difference_adjudication_threshold_ms",
            "reference_events_sha256", "prompted_flat_manually_verified_count",
        }
        or annotation_audit.get("schema_version")
        != "mayo_action_anchor_blinded_reference_audit_v1"
        or annotation_audit.get("registry_sha256") != registry_sha
        or annotation_audit.get("annotator_count") != 2
        or annotation_audit.get("annotators_blinded") is not True
        or annotation_audit.get("adjudication_complete") is not True
        or annotation_audit.get("boundary_difference_adjudication_threshold_ms") != 500
        or annotation_audit.get("reference_events_sha256") != reference_sha
        or annotation_audit.get("prompted_flat_manually_verified_count")
        != manually_verified_flat
    ):
        raise ValueError("blinded reference audit does not authenticate these 72 events")
    unused = set(range(len(predictions)))
    matched = 0
    ious = []
    for reference in references:
        candidates = [
            index for index in unused
            if predictions[index].recording_slot == reference.recording_slot
            and predictions[index].action == reference.action
            and _temporal_iou(reference, predictions[index]) > 0.0
        ]
        if not candidates:
            ious.append(0.0)
            continue
        selected = max(candidates, key=lambda index: _temporal_iou(reference, predictions[index]))
        unused.remove(selected)
        matched += 1
        ious.append(_temporal_iou(reference, predictions[selected]))
    precision = float(matched / len(predictions)) if predictions else 0.0
    recall = float(matched / len(references))
    median_iou = float(np.median(np.asarray(ious, dtype=np.float64)))
    prompted_flat = manually_verified_flat
    eligible = bool(
        precision >= 0.95 and recall >= 0.95 and median_iou >= 0.80
        and prompted_flat >= 2
    )
    summary = {
        "reference_events": 72,
        "predicted_events": len(predictions),
        "matched_events": matched,
        "precision": precision,
        "recall": recall,
        "median_temporal_iou": median_iou,
        "prompted_flat_attempts": prompted_flat,
        "registry_sha256": registry_sha,
        "blinded_reference_audit_sha256": hashlib.sha256(
            annotation_audit_payload
        ).hexdigest(),
        "annotator_count": 2,
        "adjudication_complete": True,
        "eligible": eligible,
    }
    return TimingGateEvidence(
        registry_payload=private_registry_payload,
        annotation_audit_payload=annotation_audit_payload,
        reference_events=references,
        predicted_events=predictions,
        summary=summary,
    )


def build_aggregate_report(
    *,
    source_files: int,
    records: Sequence[MediaRecord],
    exact_duplicate_files: int,
    sidecar_counts: Mapping[str, int],
    registry_sha256: str,
    timing_gate: TimingGateEvidence | None,
) -> dict[str, object]:
    normalized = tuple(records)
    if (
        isinstance(source_files, bool) or not isinstance(source_files, int) or source_files <= 0
        or source_files != sum(record.source_file_count for record in normalized)
        or exact_duplicate_files != source_files - len(normalized)
        or set(sidecar_counts) != set(_SIDECAR_BASENAMES.values())
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in sidecar_counts.values())
        or _SHA256.fullmatch(str(registry_sha256)) is None
    ):
        raise ValueError("aggregate Mayo feasibility inputs are inconsistent")
    durations = np.asarray([
        float(record.duration_seconds) for record in normalized
        if record.duration_seconds is not None
    ], dtype=np.float64)
    if timing_gate is not None and not isinstance(timing_gate, TimingGateEvidence):
        raise ValueError("timing gate must come from authenticated evidence")
    if timing_gate is not None:
        recomputed = evaluate_timing_gate(
            timing_gate.registry_payload,
            timing_gate.annotation_audit_payload,
            timing_gate.reference_events,
            timing_gate.predicted_events,
        )
        if (
            dict(recomputed.summary) != dict(timing_gate.summary)
            or recomputed.summary.get("registry_sha256") != registry_sha256
        ):
            raise ValueError("timing gate evidence or registry commitment drifted")
        gate = dict(recomputed.summary)
    else:
        gate = {
        "reference_events": 0,
        "predicted_events": 0,
        "matched_events": 0,
        "precision": None,
        "recall": None,
        "median_temporal_iou": None,
        "prompted_flat_attempts": 0,
        "eligible": False,
        "reason": "no_locked_two_annotator_72_event_timing_audit",
        }
    if gate.get("eligible") is True and gate.get("reference_events") != 72:
        raise ValueError("eligible timing evidence must cover all 72 references")
    audio_sources = sum(r.source_file_count for r in normalized if r.has_audio is True)
    free_sources = sum(r.source_file_count for r in normalized if r.has_audio is False)
    unknown_sources = sum(r.source_file_count for r in normalized if r.has_audio is None)
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "historical_mayo_action_timing_feasibility_only",
        "media_inventory": {
            "source_files": source_files,
            "unique_contents": len(normalized),
            "exact_duplicate_files": exact_duplicate_files,
            "audio_bearing_source_files": audio_sources,
            "audio_free_source_files": free_sources,
            "audio_unknown_source_files": unknown_sources,
            "audio_bearing_unique_contents": sum(r.has_audio is True for r in normalized),
            "audio_free_unique_contents": sum(r.has_audio is False for r in normalized),
            "probe_failed_unique_contents": sum(r.has_audio is None for r in normalized),
            "duration_seconds_unique_contents": {
                "count": int(durations.size),
                "minimum": None if not durations.size else float(durations.min()),
                "median": None if not durations.size else float(np.median(durations)),
                "maximum": None if not durations.size else float(durations.max()),
                "total": float(durations.sum()),
            },
        },
        "discovered_sidecar_files": dict(sidecar_counts),
        "audit_registry": {
            "selection_rule": "12_smallest_audio_bearing_deduplicated_content_hashes_before_transcription",
            "selected_recordings": 12,
            "registry_sha256": registry_sha256,
            "row_level_registry_public": False,
        },
        "timing_gate": gate,
        "scoring": {
            "mayo_action_expert_predictions": 0,
            "mayo_accuracy_defined": False,
            "allowed": False,
            "reason": "timing_gate_not_eligible" if not gate.get("eligible") else "requires_separate_locked_scoring_release",
        },
        "claim_boundary": {
            "mayo_accuracy_claim_authorized": False,
            "house_brackmann_claim_authorized": False,
            "clinical_use_authorized": False,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-root", required=True, type=Path)
    parser.add_argument("--private-registry", required=True, type=Path)
    parser.add_argument("--public-report", required=True, type=Path)
    return parser


def _reject_symlink_parent_components(path: Path) -> None:
    candidate = Path(path)
    current = Path(candidate.anchor)
    for part in candidate.parent.parts[1:]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("audit output parent contains a symlink or non-directory")


def validate_output_boundaries(
    media_root: Path,
    private_path: Path,
    public_path: Path,
) -> None:
    media = Path(media_root)
    private = Path(private_path)
    public = Path(public_path)
    if (
        not media.is_absolute() or not private.is_absolute() or not public.is_absolute()
        or Path(os.path.abspath(media)) != media
        or Path(os.path.abspath(private)) != private
        or Path(os.path.abspath(public)) != public
        or private == public
    ):
        raise ValueError("audit input/output paths must be distinct canonical absolutes")
    try:
        media_metadata = os.lstat(media)
    except OSError as exc:
        raise ValueError("Mayo media root is unavailable") from exc
    if not stat.S_ISDIR(media_metadata.st_mode) or stat.S_ISLNK(media_metadata.st_mode):
        raise ValueError("Mayo media root must be a real directory")
    if media.resolve(strict=True) != media:
        raise ValueError("Mayo media root must not traverse symlink components")
    for output in (private, public):
        try:
            output.relative_to(media)
        except ValueError:
            pass
        else:
            raise ValueError("audit output may not mutate the read-only media tree")
        _reject_symlink_parent_components(output)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    validate_output_boundaries(
        args.media_root, args.private_registry, args.public_report,
    )
    inventory = inventory_media(args.media_root)
    _, registry_bytes = build_private_audit_registry(inventory.records)
    registry_sha = hashlib.sha256(registry_bytes).hexdigest()
    report = build_aggregate_report(
        source_files=inventory.source_files,
        records=inventory.records,
        exact_duplicate_files=inventory.exact_duplicate_files,
        sidecar_counts=discover_timing_sidecars(args.media_root),
        registry_sha256=registry_sha,
        timing_gate=None,
    )
    report_bytes = _canonical_json(report)
    validate_output_boundaries(
        args.media_root, args.private_registry, args.public_report,
    )
    published_registry_sha, report_sha = write_audit_release_pair_no_overwrite(
        args.private_registry, registry_bytes, args.public_report, report_bytes,
    )
    if published_registry_sha != registry_sha:
        raise ValueError("published private registry commitment changed")
    print(json.dumps({
        "schema_version": "mayo_action_anchor_feasibility_receipt_v1",
        "public_report_sha256": report_sha,
        "private_registry_sha256": registry_sha,
        "source_files": inventory.source_files,
        "unique_contents": inventory.unique_contents,
        "mayo_action_expert_predictions": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
