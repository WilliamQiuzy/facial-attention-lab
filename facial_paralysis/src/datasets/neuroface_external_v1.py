"""Strict, deidentified inventory for Toronto NeuroFace external validation.

Raw archives and publisher participant IDs are used only in memory.  The
serialized manifest contains opaque IDs, exact byte digests, cohort labels,
task labels, and averaged SLP ratings, but no archive member names or paths.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import secrets
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

import numpy as np
import openpyxl


DATASET = "Toronto_NeuroFace_v1"
PRIMARY_TASKS = ("NSM_KISS", "NSM_OPEN", "NSM_SPREAD")
TASKS = (
    "BBP_NORMAL",
    "DDK_PA",
    "DDK_PATAKA",
    "NSM_BIGSMILE",
    "NSM_BLOW",
    "NSM_BROW",
    "NSM_KISS",
    "NSM_OPEN",
    "NSM_SPREAD",
)
COHORTS = ("als", "healthy_control", "post_stroke")
SCORE_NAMES = ("symmetry", "rom", "speed", "variability", "fatigue", "total")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VIDEO = re.compile(
    r"^(?P<subject>[A-Z0-9]+)_(?P<session>[0-9]{2})_"
    r"(?P<task>BBP_NORMAL|DDK_PA|DDK_PATAKA|NSM_BIGSMILE|NSM_BLOW|"
    r"NSM_BROW|NSM_KISS|NSM_OPEN|NSM_SPREAD)_color\.avi$"
)


@dataclass(frozen=True)
class ArchiveBinding:
    archive_id: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class CohortSource:
    cohort: str
    binary_label: str
    video_archive_id: str
    landmark_archive_id: str
    slp_workbook_path: Path | None = None
    slp_archive_id: str | None = None
    slp_member: str | None = None


@dataclass(frozen=True)
class InventoryExpectation:
    participants: Mapping[str, int]
    videos: Mapping[str, int]
    annotated_frames: Mapping[str, int]


@dataclass(frozen=True)
class ParsedVideoName:
    subject_id: str
    session: str
    task: str


@dataclass(frozen=True)
class Participant:
    raw_subject_id: str
    participant_id: str
    cohort: str
    binary_label: str


@dataclass(frozen=True)
class NeuroFaceRecord:
    raw_subject_id: str
    participant_id: str
    recording_id: str
    cohort: str
    binary_label: str
    session: str
    task: str
    video_archive_id: str
    video_member: str
    video_sha256: str
    video_size_bytes: int
    landmark_archive_id: str
    landmark_member: str
    landmark_sha256: str
    annotated_frames: int
    slp_scores: Mapping[str, float]


@dataclass(frozen=True)
class NeuroFaceInventory:
    archives: Mapping[str, ArchiveBinding]
    participants: tuple[Participant, ...]
    records: tuple[NeuroFaceRecord, ...]
    slp_workbook_sha256: Mapping[str, str]
    expected: InventoryExpectation


REAL_ARCHIVE_SHA256 = {
    "als_videos": "0f484cc915ba248e5b66319a12030fcaa729044a4c2904ec8d601b546736051c",
    "als_landmarks": "63b4b500b2856141d897eea1093d0a73ef5fd4924eda5091edb343650ca119bf",
    "healthy_control_videos": "ea720fb904cf498645e388cf14a9b110e8435cdaa3d78aebfa3ed8a3e7343d6f",
    "healthy_control_landmarks": "dd46cc7a70ba2f0a713a6f0efcc408bb96661905a26a1094f43fb2e2d85b058b",
    "post_stroke_videos": "3f86e5202097e5a976555d7d12b3d3d8a0c29503ef89d51c9444b4a06a9b3bed",
    "post_stroke_landmarks": "25b9a6e094d777cfb7c9d36c562b813575c283880dc41ab71658a9a87e9ca979",
}
REAL_EXPECTATION = InventoryExpectation(
    participants={"als": 11, "healthy_control": 11, "post_stroke": 14},
    videos={"als": 76, "healthy_control": 80, "post_stroke": 105},
    annotated_frames={"als": 920, "healthy_control": 1015, "post_stroke": 1371},
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError("source artifacts must be regular non-symlink files")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _opaque(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def parse_video_filename(filename: str) -> ParsedVideoName:
    if not isinstance(filename, str):
        raise ValueError("video filename must be text")
    match = _VIDEO.fullmatch(filename)
    if match is None:
        raise ValueError("video filename differs from the frozen NeuroFace convention")
    return ParsedVideoName(
        subject_id=match.group("subject"),
        session=match.group("session"),
        task=match.group("task"),
    )


def parse_landmark_text(payload: bytes) -> dict[int, np.ndarray]:
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("landmark payload must be nonempty bytes")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeError as exc:
        raise ValueError("landmark payload must be UTF-8 text") from exc
    rows = list(csv.reader(io.StringIO(text), skipinitialspace=True))
    expected_header = ["Frame"] + [
        part for index in range(1, 69) for part in (f"x{index}", f"y{index}")
    ]
    if not rows or [value.strip() for value in rows[0]] != expected_header:
        raise ValueError("landmark header differs from the exact 68-point schema")
    output: dict[int, np.ndarray] = {}
    for row in rows[1:]:
        if len(row) != 137:
            raise ValueError("landmark row must contain frame plus 68 coordinate pairs")
        try:
            frame_value = float(row[0].strip())
            values = np.asarray([float(value.strip()) for value in row[1:]], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("landmark row must be numeric") from exc
        if (
            not frame_value.is_integer()
            or frame_value < 0
            or not np.isfinite(values).all()
        ):
            raise ValueError("landmark frame/coordinates must be finite and nonnegative-indexed")
        frame = int(frame_value)
        if frame in output:
            raise ValueError("landmark frame indices must be unique")
        output[frame] = values.reshape(68, 2)
    if not output:
        raise ValueError("landmark file must contain at least one annotated frame")
    return dict(sorted(output.items()))


def _validate_member_name(name: str) -> None:
    if not name or "\\" in name or "\x00" in name:
        raise ValueError("archive member path is invalid")
    pure = PurePosixPath(name)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("archive member path escapes its archive")


def _validated_zip(binding: ArchiveBinding) -> zipfile.ZipFile:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", binding.archive_id):
        raise ValueError("archive IDs must be stable identifiers")
    if _SHA256.fullmatch(binding.sha256) is None:
        raise ValueError("archive SHA-256 is malformed")
    path = Path(binding.path)
    observed = _sha256_file(path)
    if not secrets.compare_digest(observed, binding.sha256):
        raise ValueError("archive bytes differ from their pinned SHA-256")
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("source archive is not a valid ZIP") from exc
    infos = archive.infolist()
    names: set[str] = set()
    try:
        if not infos or len(infos) > 20_000:
            raise ValueError("archive entry count is invalid")
        for info in infos:
            _validate_member_name(info.filename)
            if info.filename in names:
                raise ValueError("archive contains duplicate member names")
            names.add(info.filename)
            if info.flag_bits & 0x1:
                raise ValueError("encrypted archive members are not accepted")
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise ValueError("archive symlinks are not accepted")
            if info.file_size < 0 or info.file_size > 2_000_000_000:
                raise ValueError("archive member size is outside the strict bound")
    except BaseException:
        archive.close()
        raise
    return archive


def _read_slp_workbook(payload: bytes) -> tuple[dict[str, dict[str, float]], str]:
    digest = _sha256_bytes(payload)
    try:
        workbook = openpyxl.load_workbook(io.BytesIO(payload), read_only=True, data_only=True)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError("SLP workbook is invalid") from exc
    if len(workbook.worksheets) != 1:
        raise ValueError("SLP workbook must contain exactly one worksheet")
    sheet = workbook.worksheets[0]
    iterator = sheet.iter_rows(values_only=True)
    try:
        header = list(next(iterator))
    except StopIteration as exc:
        raise ValueError("SLP workbook is empty") from exc
    expected = [
        "File Name", "Subject ID", "Symmetry (SLP1)", "ROM (SLP1)",
        "Speed (SLP1)", "Variability (SLP1)", "Fatigue (SLP1)", "Tot (SLP1)",
        None, "Symmetry (SLP2)", "ROM (SLP2)", "Speed (SLP2)",
        "Variability (SLP2)", "Fatigue (SLP2)", "Tot (SLP2)",
    ]
    if header[:15] != expected or any(value is not None for value in header[15:]):
        raise ValueError("SLP workbook header differs from the frozen schema")
    output: dict[str, dict[str, float]] = {}
    for values in iterator:
        row = list(values)
        if not row or all(value is None for value in row):
            continue
        row += [None] * max(0, 15 - len(row))
        filename, subject = row[0], row[1]
        if not isinstance(filename, str) or not isinstance(subject, str):
            raise ValueError("SLP rows require filename and subject text")
        parsed = parse_video_filename(filename)
        if parsed.subject_id != subject or filename in output:
            raise ValueError("SLP filename/subject join is invalid or duplicated")
        first = row[2:8]
        second = row[9:15]
        if len(first) != 6 or len(second) != 6:
            raise ValueError("SLP rows require two complete rater blocks")
        try:
            first_values = [float(value) for value in first]
            second_values = [float(value) for value in second]
        except (TypeError, ValueError) as exc:
            raise ValueError("SLP ratings must be numeric") from exc
        if (
            not all(math.isfinite(value) for value in first_values + second_values)
            or not all(1.0 <= value <= 5.0 for value in first_values[:5] + second_values[:5])
            or abs(first_values[5] - sum(first_values[:5])) > 1e-9
            or abs(second_values[5] - sum(second_values[:5])) > 1e-9
        ):
            raise ValueError("SLP ratings or totals are invalid")
        output[filename] = {
            name: float((first_values[index] + second_values[index]) / 2.0)
            for index, name in enumerate(SCORE_NAMES)
        }
    if not output:
        raise ValueError("SLP workbook contains no scored videos")
    return output, digest


def _slp_bytes(
    source: CohortSource,
    archives: Mapping[str, zipfile.ZipFile],
) -> bytes:
    path = source.slp_workbook_path
    archive_id = source.slp_archive_id
    member = source.slp_member
    if path is not None and archive_id is None and member is None:
        checked = Path(path)
        _sha256_file(checked)
        return checked.read_bytes()
    if path is None and archive_id is not None and member is not None:
        if archive_id not in archives:
            raise ValueError("SLP archive reference is unknown")
        _validate_member_name(member)
        try:
            return archives[archive_id].read(member)
        except KeyError as exc:
            raise ValueError("SLP workbook member is missing") from exc
    raise ValueError("SLP workbook must have exactly one source binding")


def _collect_members(
    archive: zipfile.ZipFile,
    suffix: str,
) -> dict[str, tuple[str, bytes]]:
    output: dict[str, tuple[str, bytes]] = {}
    for info in archive.infolist():
        if info.is_dir() or not info.filename.lower().endswith(suffix):
            continue
        basename = PurePosixPath(info.filename).name
        if basename in output:
            raise ValueError("archive contains duplicate relevant basenames")
        try:
            payload = archive.read(info)
        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
            raise ValueError("archive member failed CRC/read validation") from exc
        if len(payload) != info.file_size:
            raise ValueError("archive member size differs after read")
        output[basename] = (info.filename, payload)
    return output


def audit_neuroface_sources(
    archive_bindings: Mapping[str, ArchiveBinding],
    cohort_sources: Sequence[CohortSource],
    *,
    expected: InventoryExpectation,
) -> NeuroFaceInventory:
    if set(source.cohort for source in cohort_sources) != set(COHORTS):
        raise ValueError("NeuroFace requires exactly the three frozen cohorts")
    if set(expected.participants) != set(COHORTS) or set(expected.videos) != set(COHORTS):
        raise ValueError("inventory expectation must cover every cohort")
    if set(expected.annotated_frames) != set(COHORTS):
        raise ValueError("annotated-frame expectation must cover every cohort")
    if set(archive_bindings) != {binding.archive_id for binding in archive_bindings.values()}:
        raise ValueError("archive mapping keys must equal binding IDs")
    archives: dict[str, zipfile.ZipFile] = {}
    try:
        for archive_id, binding in sorted(archive_bindings.items()):
            archives[archive_id] = _validated_zip(binding)
        participants: dict[tuple[str, str], Participant] = {}
        records: list[NeuroFaceRecord] = []
        slp_hashes: dict[str, str] = {}
        for source in sorted(cohort_sources, key=lambda value: value.cohort):
            if source.cohort not in COHORTS or source.binary_label not in {"affected", "unaffected"}:
                raise ValueError("cohort source label is invalid")
            if source.video_archive_id not in archives or source.landmark_archive_id not in archives:
                raise ValueError("cohort archive reference is unknown")
            videos = _collect_members(archives[source.video_archive_id], ".avi")
            landmarks = _collect_members(archives[source.landmark_archive_id], ".txt")
            ratings, slp_sha = _read_slp_workbook(_slp_bytes(source, archives))
            slp_hashes[source.cohort] = slp_sha
            video_stems = {Path(name).stem for name in videos}
            landmark_stems = {Path(name).stem for name in landmarks}
            if video_stems != landmark_stems or set(videos) != set(ratings):
                raise ValueError("video, landmark, and SLP inventories do not match exactly")
            for filename in sorted(videos):
                parsed = parse_video_filename(filename)
                landmark_name = f"{Path(filename).stem}.txt"
                video_member, video_payload = videos[filename]
                landmark_member, landmark_payload = landmarks[landmark_name]
                landmark_rows = parse_landmark_text(landmark_payload)
                participant_key = (source.cohort, parsed.subject_id)
                participant_id = _opaque(
                    "grp", f"{DATASET}:participant:{source.cohort}:{parsed.subject_id}"
                )
                participant = Participant(
                    raw_subject_id=parsed.subject_id,
                    participant_id=participant_id,
                    cohort=source.cohort,
                    binary_label=source.binary_label,
                )
                prior = participants.setdefault(participant_key, participant)
                if prior != participant:
                    raise ValueError("participant crosses cohort or binary label")
                video_sha = _sha256_bytes(video_payload)
                records.append(NeuroFaceRecord(
                    raw_subject_id=parsed.subject_id,
                    participant_id=participant_id,
                    recording_id=_opaque("rec", f"{DATASET}:video:{video_sha}"),
                    cohort=source.cohort,
                    binary_label=source.binary_label,
                    session=parsed.session,
                    task=parsed.task,
                    video_archive_id=source.video_archive_id,
                    video_member=video_member,
                    video_sha256=video_sha,
                    video_size_bytes=len(video_payload),
                    landmark_archive_id=source.landmark_archive_id,
                    landmark_member=landmark_member,
                    landmark_sha256=_sha256_bytes(landmark_payload),
                    annotated_frames=len(landmark_rows),
                    slp_scores=ratings[filename],
                ))
        ordered_participants = tuple(sorted(
            participants.values(), key=lambda value: (value.cohort, value.raw_subject_id)
        ))
        ordered_records = tuple(sorted(
            records,
            key=lambda value: (value.cohort, value.raw_subject_id, value.task),
        ))
        for cohort in COHORTS:
            participant_count = sum(p.cohort == cohort for p in ordered_participants)
            cohort_records = [record for record in ordered_records if record.cohort == cohort]
            if (
                participant_count != int(expected.participants[cohort])
                or len(cohort_records) != int(expected.videos[cohort])
                or sum(record.annotated_frames for record in cohort_records)
                != int(expected.annotated_frames[cohort])
            ):
                raise ValueError("NeuroFace inventory differs from frozen counts")
        by_participant: dict[str, list[NeuroFaceRecord]] = {}
        for record in ordered_records:
            by_participant.setdefault(record.participant_id, []).append(record)
        for participant_id, participant_records in by_participant.items():
            tasks = [record.task for record in participant_records]
            if len(tasks) != len(set(tasks)):
                raise ValueError("participant contains duplicate task recordings")
            if any(tasks.count(task) != 1 for task in PRIMARY_TASKS):
                raise ValueError("participant lacks the exact three primary recordings")
        if len({record.video_sha256 for record in ordered_records}) != len(ordered_records):
            raise ValueError("NeuroFace contains duplicate video content")
        return NeuroFaceInventory(
            archives=dict(archive_bindings),
            participants=ordered_participants,
            records=ordered_records,
            slp_workbook_sha256=slp_hashes,
            expected=expected,
        )
    finally:
        for archive in archives.values():
            archive.close()


def build_private_manifest(inventory: NeuroFaceInventory) -> dict[str, object]:
    participants = [
        {
            "participant_id": participant.participant_id,
            "cohort": participant.cohort,
            "binary_label": participant.binary_label,
        }
        for participant in inventory.participants
    ]
    records = [
        {
            "recording_id": record.recording_id,
            "participant_id": record.participant_id,
            "cohort": record.cohort,
            "binary_label": record.binary_label,
            "session": record.session,
            "task": record.task,
            "video_archive_id": record.video_archive_id,
            "video_sha256": record.video_sha256,
            "video_size_bytes": record.video_size_bytes,
            "landmark_archive_id": record.landmark_archive_id,
            "landmark_sha256": record.landmark_sha256,
            "annotated_frames": record.annotated_frames,
            "slp_scores": {name: float(record.slp_scores[name]) for name in SCORE_NAMES},
        }
        for record in inventory.records
    ]
    counts_by_cohort = {
        cohort: {
            "participants": sum(row["cohort"] == cohort for row in participants),
            "videos": sum(row["cohort"] == cohort for row in records),
            "annotated_frames": sum(
                int(row["annotated_frames"]) for row in records if row["cohort"] == cohort
            ),
        }
        for cohort in COHORTS
    }
    manifest: dict[str, object] = {
        "schema_version": "neuroface_external_private_manifest_v1",
        "dataset": DATASET,
        "claim_unit": "participant",
        "target": "neurological_orofacial_impairment_vs_healthy_control",
        "primary_tasks": list(PRIMARY_TASKS),
        "counts": {
            "participants": len(participants),
            "videos": len(records),
            "annotated_frames": sum(int(row["annotated_frames"]) for row in records),
            "affected_participants": sum(
                row["binary_label"] == "affected" for row in participants
            ),
            "unaffected_participants": sum(
                row["binary_label"] == "unaffected" for row in participants
            ),
            "primary_complete_participants": len(participants),
            "by_cohort": counts_by_cohort,
        },
        "archives": {
            archive_id: {
                "sha256": binding.sha256,
                "size_bytes": int(os.lstat(binding.path).st_size),
            }
            for archive_id, binding in sorted(inventory.archives.items())
        },
        "slp_workbook_sha256": dict(sorted(inventory.slp_workbook_sha256.items())),
        "participants": participants,
        "records": records,
    }
    encoded = json.dumps(manifest, sort_keys=True, allow_nan=False)
    if any(token in encoded for token in ("/Users/", "Videos/", "Landmarks_gt/", ".avi", ".txt")):
        raise ValueError("private manifest unexpectedly contains source locations")
    return manifest


def real_source_configuration(data_root: Path) -> tuple[
    dict[str, ArchiveBinding], tuple[CohortSource, ...], InventoryExpectation
]:
    root = Path(data_root).expanduser().absolute()
    archives = root / "archive"
    extracted = root / "extracted_non_depth" / "NeuroFace Open Access Data"
    paths = {
        "als_videos": archives / "als" / "Videos.zip",
        "als_landmarks": archives / "als" / "Landmarks_gt.zip",
        "healthy_control_videos": archives / "healthy_controls" / "Videos.zip",
        "healthy_control_landmarks": (
            archives / "healthy_controls" / "Landmarks_gt_and_VideoInfoFile_HC.zip"
        ),
        "post_stroke_videos": archives / "stroke" / "Videos_and_metadata.zip",
        "post_stroke_landmarks": archives / "stroke" / "Landmarks_gt.zip",
    }
    bindings = {
        archive_id: ArchiveBinding(archive_id, path, REAL_ARCHIVE_SHA256[archive_id])
        for archive_id, path in paths.items()
    }
    sources = (
        CohortSource(
            cohort="als",
            binary_label="affected",
            video_archive_id="als_videos",
            landmark_archive_id="als_landmarks",
            slp_workbook_path=extracted / "ALS" / "SLP_Assessment_ALS.xlsx",
        ),
        CohortSource(
            cohort="healthy_control",
            binary_label="unaffected",
            video_archive_id="healthy_control_videos",
            landmark_archive_id="healthy_control_landmarks",
            slp_workbook_path=(
                extracted / "Healthy controls" / "SLP_Assessment_HC.xlsx"
            ),
        ),
        CohortSource(
            cohort="post_stroke",
            binary_label="affected",
            video_archive_id="post_stroke_videos",
            landmark_archive_id="post_stroke_landmarks",
            slp_archive_id="post_stroke_videos",
            slp_member="SLP_Assessment_PS.xlsx",
        ),
    )
    return bindings, sources, REAL_EXPECTATION


__all__ = [
    "ArchiveBinding",
    "CohortSource",
    "InventoryExpectation",
    "NeuroFaceInventory",
    "NeuroFaceRecord",
    "PRIMARY_TASKS",
    "REAL_EXPECTATION",
    "SCORE_NAMES",
    "TASKS",
    "audit_neuroface_sources",
    "build_private_manifest",
    "parse_landmark_text",
    "parse_video_filename",
    "real_source_configuration",
]
