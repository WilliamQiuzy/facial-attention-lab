#!/usr/bin/env python3
"""Build the frozen NeuroFace clinical23_v2 cache without unpacking the corpus.

The command authenticates each ZIP member, copies one video at a time into an
owner-only decoder directory, applies the existing four-by-32 PalsyNet feature
contract, validates any resumable cache, and deletes the temporary source on
every normal or exceptional exit.  Raw member names never enter the generated
cache or collection manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_palsynet_v2_windows import (  # noqa: E402
    IdentityBinding,
    RecordingExtractionError,
    SourceVideo,
    extract_source_video,
    managed_extractor,
    validate_retained_recording,
    write_validated_recording_cache,
)
from src.datasets.dynamic_landmark import (  # noqa: E402
    DYNAMIC_FEATURE_SCHEMA,
    DYNAMIC_FEATURE_SHAPE,
    load_dynamic_landmark_recording,
)
from src.datasets.neuroface_external_v1 import (  # noqa: E402
    PRIMARY_TASKS,
    REAL_ARCHIVE_SHA256,
    real_source_configuration,
)
from src.preprocessing.action_bundle import MediaPipeFeatureExtractor  # noqa: E402


FROZEN_MEDIAPIPE_MODEL_SHA256 = (
    "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
)
COLLECTION_SCHEMA = "neuroface_clinical23_v2_windows_v1"
PRIVATE_SCHEMA = "neuroface_external_private_manifest_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REC_ID = re.compile(r"^rec_[0-9a-f]{64}$")
_GROUP_ID = re.compile(r"^grp_[0-9a-f]{64}$")
_FORBIDDEN_SERIALIZED_TOKENS = (
    "/Users/", "/home/", "\\", ".avi", "Videos/", "Landmarks_gt/",
)
_ALLOWED_EXTRACTION_EXCLUSIONS = {
    "open_failed",
    "invalid_fps",
    "invalid_frame_count",
    "nonintegral_frame_count",
    "insufficient_frame_count",
    "invalid_width",
    "invalid_height",
    "nonintegral_dimensions",
    "seek_failed",
    "decode_failed",
    "frame_dimensions_changed",
    "seek_position_mismatch",
    "no_valid_detections",
    "coverage_below_0_90",
}


def _sha256_file(path: Path) -> str:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError("authenticated input must be a regular non-symlink file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    if stat.S_ISLNK(os.lstat(path).st_mode):
        raise ValueError("manifest must not be a symlink")
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("manifest must contain one JSON object")
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON contains a duplicate key")
        result[key] = value
    return result


def validate_model_lock(
    model_path: str | Path,
    *,
    expected_sha256: str = FROZEN_MEDIAPIPE_MODEL_SHA256,
) -> str:
    if _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("expected model SHA-256 is malformed")
    observed = _sha256_file(Path(model_path))
    if not secrets.compare_digest(observed, expected_sha256):
        raise ValueError("MediaPipe model differs from the frozen asset")
    return observed


def copy_authenticated_zip_member(
    archive: zipfile.ZipFile,
    member: str,
    expected_sha256: str,
    decoder_directory: str | Path,
) -> Path:
    """Copy and hash exactly the bytes later consumed by OpenCV."""
    if _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("expected member SHA-256 is malformed")
    pure = PurePosixPath(member)
    if (
        not member or pure.is_absolute() or "\\" in member or "\x00" in member
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError("archive member path is unsafe")
    root = Path(decoder_directory)
    root_info = os.lstat(root)
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise ValueError("decoder directory must be a real directory")
    output = root / f"source-{secrets.token_hex(16)}.avi"
    digest = hashlib.sha256()
    try:
        with archive.open(member, "r") as source, open(output, "xb", opener=_private_opener) as sink:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                sink.write(chunk)
            sink.flush()
            os.fsync(sink.fileno())
        if not secrets.compare_digest(digest.hexdigest(), expected_sha256):
            raise ValueError("ZIP member differs from the private manifest")
        output.chmod(0o400)
        return output
    except BaseException:
        if output.exists():
            output.chmod(0o600)
            output.unlink()
        raise


def _private_opener(path: str, flags: int) -> int:
    return os.open(path, flags, 0o600)


def validate_existing_cache(
    cache_path: str | Path,
    private_record: Mapping[str, object],
) -> dict[str, object]:
    path = Path(cache_path)
    recording = load_dynamic_landmark_recording(path)
    expected_label = 1 if private_record.get("binary_label") == "affected" else 0
    exact = (
        recording.recording_id == private_record.get("recording_id")
        and recording.group_id == private_record.get("participant_id")
        and recording.source_sha256 == private_record.get("video_sha256")
        and recording.label == expected_label
        and recording.feature_schema == DYNAMIC_FEATURE_SCHEMA
        and recording.features.shape == DYNAMIC_FEATURE_SHAPE
    )
    if not exact:
        raise ValueError("existing cache is not bound to the private source record")
    return {
        "recording_id": recording.recording_id,
        "participant_id": recording.group_id,
        "video_sha256": recording.source_sha256,
        "cache_sha256": _sha256_file(path),
        "coverage": float(recording.valid_mask.mean()),
        "status": "retained",
    }


def _validate_private_record(row: Mapping[str, object]) -> None:
    if _REC_ID.fullmatch(str(row.get("recording_id", ""))) is None:
        raise ValueError("private recording ID is malformed")
    if _GROUP_ID.fullmatch(str(row.get("participant_id", ""))) is None:
        raise ValueError("private participant ID is malformed")
    if _SHA256.fullmatch(str(row.get("video_sha256", ""))) is None:
        raise ValueError("private video SHA-256 is malformed")
    if row.get("binary_label") not in {"affected", "unaffected"}:
        raise ValueError("private binary label is invalid")
    if row.get("cohort") not in {"als", "healthy_control", "post_stroke"}:
        raise ValueError("private cohort is invalid")


def build_collection_manifest(
    private_manifest: Mapping[str, object],
    cache_rows: Sequence[Mapping[str, object]],
    *,
    private_manifest_sha256: str,
    model_sha256: str,
    implementation_sha256: str,
) -> dict[str, object]:
    if private_manifest.get("schema_version") != PRIVATE_SCHEMA:
        raise ValueError("private manifest schema is not authorized")
    for digest in (private_manifest_sha256, model_sha256, implementation_sha256):
        if _SHA256.fullmatch(digest) is None:
            raise ValueError("collection provenance SHA-256 is malformed")
    records = private_manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("private manifest records are missing")
    expected: dict[str, Mapping[str, object]] = {}
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("private record must be an object")
        _validate_private_record(row)
        record_id = str(row["recording_id"])
        if record_id in expected:
            raise ValueError("private recording IDs must be unique")
        expected[record_id] = row
    observed: dict[str, dict[str, object]] = {}
    retained_ids: set[str] = set()
    for raw in cache_rows:
        row = dict(raw)
        record_id = str(row.get("recording_id", ""))
        if record_id not in expected or record_id in observed:
            raise ValueError("cache rows must exactly match private recording IDs")
        if (
            row.get("participant_id") != expected[record_id]["participant_id"]
            or row.get("video_sha256") != expected[record_id]["video_sha256"]
        ):
            raise ValueError("cache/QC row is not bound to its private record")
        status = row.get("status", "retained")
        if status == "retained":
            if _SHA256.fullmatch(str(row.get("cache_sha256", ""))) is None:
                raise ValueError("retained cache SHA-256 is malformed")
            coverage = float(row.get("coverage", -1.0))
            if not (0.9 <= coverage <= 1.0):
                raise ValueError("retained NeuroFace cache must meet 90 percent coverage")
            observed[record_id] = {
                "recording_id": record_id,
                "participant_id": row["participant_id"],
                "video_sha256": row["video_sha256"],
                "cache_sha256": row["cache_sha256"],
                "coverage": coverage,
                "status": "retained",
            }
            retained_ids.add(record_id)
        elif status == "excluded":
            reason = row.get("exclusion_reason")
            if reason not in _ALLOWED_EXTRACTION_EXCLUSIONS or set(row) != {
                "recording_id", "participant_id", "video_sha256",
                "status", "exclusion_reason",
            }:
                raise ValueError("QC exclusion is not an exact allowed technical reason")
            observed[record_id] = {
                "recording_id": record_id,
                "participant_id": row["participant_id"],
                "video_sha256": row["video_sha256"],
                "status": "excluded",
                "exclusion_reason": reason,
            }
        else:
            raise ValueError("cache/QC row status is invalid")
    if set(observed) != set(expected):
        raise ValueError("collection manifest requires every private recording")
    primary = set(private_manifest.get("primary_tasks", []))
    eligible_participants: dict[str, set[str]] = {}
    for record_id, source in expected.items():
        if record_id not in retained_ids:
            continue
        if source.get("task") in primary:
            eligible_participants.setdefault(str(source["participant_id"]), set()).add(
                str(source["task"])
            )
    payload: dict[str, object] = {
        "schema_version": COLLECTION_SCHEMA,
        "dataset": private_manifest.get("dataset"),
        "claim_unit": "participant",
        "target": private_manifest.get("target"),
        "feature_schema": DYNAMIC_FEATURE_SCHEMA,
        "window_shape": list(DYNAMIC_FEATURE_SHAPE),
        "minimum_recording_coverage": 0.9,
        "primary_tasks": sorted(primary),
        "counts": {
            "source_records": len(expected),
            "retained": len(retained_ids),
            "excluded": len(observed) - len(retained_ids),
            "participants": len({str(row["participant_id"]) for row in expected.values()}),
            "primary_complete_participants": sum(
                tasks == primary for tasks in eligible_participants.values()
            ),
        },
        "provenance": {
            "private_manifest_sha256": private_manifest_sha256,
            "mediapipe_model_sha256": model_sha256,
            "implementation_sha256": implementation_sha256,
        },
        "records": [observed[key] for key in sorted(observed)],
    }
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
    if any(token in encoded for token in _FORBIDDEN_SERIALIZED_TOKENS):
        raise ValueError("collection manifest contains a raw identifier or location")
    return payload


def _technical_exclusion_reason(error: BaseException) -> str:
    if isinstance(error, RecordingExtractionError):
        reason = str(error)
        if reason in _ALLOWED_EXTRACTION_EXCLUSIONS:
            return reason
        raise error
    if isinstance(error, ValueError) and str(error) == "recording coverage is below 90 percent":
        return "coverage_below_0_90"
    raise error


def _implementation_sha256() -> str:
    paths = (
        Path(__file__),
        PROJECT_ROOT / "scripts" / "build_palsynet_v2_windows.py",
        PROJECT_ROOT / "src" / "preprocessing" / "action_bundle.py",
        PROJECT_ROOT / "src" / "preprocessing" / "clinical_landmarks.py",
        PROJECT_ROOT / "src" / "datasets" / "dynamic_landmark.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _archive_member_index(
    archive: zipfile.ZipFile,
    expected_hashes: set[str],
) -> dict[str, str]:
    index: dict[str, str] = {}
    for info in archive.infolist():
        if info.is_dir() or not info.filename.lower().endswith(".avi"):
            continue
        digest = hashlib.sha256()
        with archive.open(info, "r") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        observed = digest.hexdigest()
        if observed in expected_hashes:
            if observed in index:
                raise ValueError("source archive contains duplicate expected video bytes")
            index[observed] = info.filename
    if set(index) != expected_hashes:
        raise ValueError("source archive does not contain the exact private videos")
    return index


def run_extraction(
    *,
    data_root: Path,
    private_manifest_path: Path,
    model_path: Path,
    output_root: Path,
) -> dict[str, object]:
    private_sha = _sha256_file(private_manifest_path)
    private = _load_json(private_manifest_path)
    model_sha = validate_model_lock(model_path)
    records = private.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("private manifest has no source records")
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("private records must be objects")
        _validate_private_record(row)
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_info = os.lstat(output_root)
    if stat.S_ISLNK(output_info.st_mode) or not stat.S_ISDIR(output_info.st_mode):
        raise ValueError("output root must be a real directory")
    collection_path = output_root / "collection_manifest.json"
    if collection_path.exists():
        raise ValueError("collection manifest already exists; frozen extraction is complete")

    bindings, _, _ = real_source_configuration(data_root)
    archive_rows: dict[str, list[Mapping[str, object]]] = {}
    for row in records:
        archive_rows.setdefault(str(row["video_archive_id"]), []).append(row)
    archives: dict[str, zipfile.ZipFile] = {}
    member_by_recording: dict[str, tuple[zipfile.ZipFile, str]] = {}
    try:
        for archive_id, rows in sorted(archive_rows.items()):
            if archive_id not in bindings:
                raise ValueError("private manifest references an unknown video archive")
            binding = bindings[archive_id]
            expected_archive_sha = str(private["archives"][archive_id]["sha256"])
            if expected_archive_sha != REAL_ARCHIVE_SHA256[archive_id]:
                raise ValueError("private archive digest differs from the frozen source lock")
            if _sha256_file(binding.path) != expected_archive_sha:
                raise ValueError("raw archive differs from the frozen source lock")
            archive = zipfile.ZipFile(binding.path)
            archives[archive_id] = archive
            hashes = {str(row["video_sha256"]) for row in rows}
            index = _archive_member_index(archive, hashes)
            for row in rows:
                member_by_recording[str(row["recording_id"])] = (
                    archive, index[str(row["video_sha256"])]
                )

        cache_rows: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory(prefix="neuroface-decoder-") as temporary:
            decoder = Path(temporary)
            decoder.chmod(0o700)
            with managed_extractor(MediaPipeFeatureExtractor, model_path=model_path) as extractor:
                for row in records:
                    cache_path = output_root / f"{row['recording_id']}.npz"
                    if cache_path.exists():
                        cache_rows.append(validate_existing_cache(cache_path, row))
                        continue
                    archive, member = member_by_recording[str(row["recording_id"])]
                    source_path = copy_authenticated_zip_member(
                        archive, member, str(row["video_sha256"]), decoder
                    )
                    try:
                        binding = IdentityBinding(
                            source_sha256=str(row["video_sha256"]),
                            recording_id=str(row["recording_id"]),
                            group_id=str(row["participant_id"]),
                            label=str(row["binary_label"]),
                            identity_status="publisher_participant",
                            claim_unit="participant",
                        )
                        result = extract_source_video(
                            SourceVideo(source_path, str(row["video_sha256"]), binding),
                            extractor,
                        )
                        validate_retained_recording(result)
                        write_validated_recording_cache(cache_path, result)
                        cache_rows.append(validate_existing_cache(cache_path, row))
                    except (RecordingExtractionError, ValueError) as exc:
                        reason = _technical_exclusion_reason(exc)
                        cache_rows.append({
                            "recording_id": row["recording_id"],
                            "participant_id": row["participant_id"],
                            "video_sha256": row["video_sha256"],
                            "status": "excluded",
                            "exclusion_reason": reason,
                        })
                    finally:
                        if source_path.exists():
                            source_path.chmod(0o600)
                            source_path.unlink()
        manifest = build_collection_manifest(
            private,
            cache_rows,
            private_manifest_sha256=private_sha,
            model_sha256=model_sha,
            implementation_sha256=_implementation_sha256(),
        )
        serialized = json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
        with open(collection_path, "x", encoding="utf-8", opener=_private_opener) as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        return manifest
    finally:
        for archive in archives.values():
            archive.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        manifest = run_extraction(
            data_root=args.data_root,
            private_manifest_path=args.private_manifest,
            model_path=args.model_path,
            output_root=args.output_root,
        )
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"NeuroFace extraction refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"counts": manifest["counts"], "status": "complete"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
