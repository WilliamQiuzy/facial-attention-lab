#!/usr/bin/env python3
"""Extract the frozen PalsyNet dynamic feature contract from every MEEI video."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_meei_participant_manifest import (  # noqa: E402
    FROZEN_EXPECTED,
    ExpectedInventory,
    validate_participant_manifest,
)


SCHEMA_VERSION = "meei_clinical23_v2_windows_v1"
FEATURE_SCHEMA = "mediapipe_bs_lr_v1+clinical23_v2"
FEATURE_SHAPE = [4, 32, 95]
MINIMUM_COVERAGE = 0.90


@dataclass(frozen=True)
class PalsyNetExtractorLock:
    model_sha256: str
    feature_schema: str
    feature_shape: tuple[int, int, int]


@dataclass(frozen=True)
class ExternalBinding:
    source_sha256: str
    recording_id: str
    group_id: str
    label: str
    identity_status: str = "publisher_participant_directory_one_video_each"
    claim_unit: str = "participant"


@dataclass(frozen=True)
class ExternalSource:
    path: Path
    source_sha256: str
    binding: ExternalBinding


class SourceAuthenticationError(RuntimeError):
    """Governed source bytes differ from the authenticated manifest."""


def _sha256_file(path: Path) -> str:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError("authenticated input must be a regular non-symlink file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_unique_json(path: Path) -> tuple[dict[str, object], str]:
    checked = Path(path)
    raw = checked.read_bytes()

    def unique(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError("JSON input contains a duplicate key")
            output[key] = value
        return output

    try:
        payload = json.loads(raw, object_pairs_hook=unique)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("JSON input is not valid unique-key JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON input root must be an object")
    return payload, hashlib.sha256(raw).hexdigest()


def validate_palsynet_extractor_lock(
    manifest: Mapping[str, object], model_path: Path
) -> PalsyNetExtractorLock:
    """Require the exact schema/window/model contract used by PalsyNet."""
    protocol = manifest.get("protocol")
    provenance = manifest.get("provenance")
    if (
        manifest.get("schema_version") != "palsynet_clinical23_v2_windows_v1"
        or manifest.get("feature_schema") != FEATURE_SCHEMA
        or manifest.get("feature_shape") != FEATURE_SHAPE
        or manifest.get("capture_mirrored") is not None
        or not isinstance(protocol, Mapping)
        or protocol.get("windows_per_recording") != 4
        or protocol.get("frames_per_window") != 32
        or not isinstance(provenance, Mapping)
    ):
        raise ValueError("PalsyNet extractor contract differs from the frozen protocol")
    expected = provenance.get("model_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("PalsyNet model digest is invalid")
    observed = _sha256_file(Path(model_path))
    if not secrets.compare_digest(observed, expected):
        raise ValueError("MediaPipe model bytes differ from the PalsyNet model")
    return PalsyNetExtractorLock(
        model_sha256=observed,
        feature_schema=FEATURE_SCHEMA,
        feature_shape=tuple(FEATURE_SHAPE),
    )


def enumerate_video_sources(
    data_root: Path,
    participant_manifest: Mapping[str, object],
    *,
    expected: ExpectedInventory = FROZEN_EXPECTED,
) -> tuple[ExternalSource, ...]:
    """Join all and only MP4 files back to opaque manifest rows by SHA-256."""
    validate_participant_manifest(participant_manifest, expected=expected)
    root = Path(data_root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("MEEI data root must be a real directory")
    participant_labels = {
        str(row["participant_id"]): str(row["binary_label"])
        for row in participant_manifest["participants"]
    }
    manifest_videos = {
        str(row["source_sha256"]): row
        for row in participant_manifest["media"]
        if row["dynamic_binary_eligible"] is True
    }
    if len(manifest_videos) != expected.videos:
        raise ValueError("MEEI manifest video set is incomplete or duplicated")
    observed: dict[str, Path] = {}
    paths = sorted(root.rglob("*.mp4"), key=lambda item: item.as_posix())
    if len(paths) != expected.videos:
        raise ValueError("MEEI source tree MP4 count differs from the manifest")
    for path in paths:
        digest = _sha256_file(path)
        if digest in observed:
            raise ValueError("MEEI MP4 content is duplicated")
        observed[digest] = path
    if set(observed) != set(manifest_videos):
        raise ValueError("MEEI source MP4 set differs from the authenticated manifest")
    output: list[ExternalSource] = []
    for digest, row in sorted(manifest_videos.items()):
        participant_id = str(row["participant_id"])
        label = participant_labels.get(participant_id)
        if label not in {"affected", "unaffected"}:
            raise ValueError("MEEI participant label join is invalid")
        output.append(ExternalSource(
            path=observed[digest],
            source_sha256=digest,
            binding=ExternalBinding(
                source_sha256=digest,
                recording_id=str(row["recording_id"]),
                group_id=participant_id,
                label=label,
            ),
        ))
    return tuple(output)


def copy_authenticated_source(
    source_path: Path, expected_sha256: str, private_directory: Path
) -> Path:
    """Copy and hash one source through the same descriptor used for bytes."""
    source = Path(source_path)
    target_root = Path(private_directory)
    if not target_root.is_dir() or target_root.is_symlink():
        raise ValueError("private decoder directory must be a real directory")
    info = os.lstat(source)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise SourceAuthenticationError("source video is not a regular file")
    target = target_root / f"source-{secrets.token_hex(16)}.mp4"
    digest = hashlib.sha256()
    try:
        with source.open("rb", buffering=0) as reader, target.open("xb", buffering=0) as writer:
            for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                digest.update(chunk)
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        if not secrets.compare_digest(digest.hexdigest(), expected_sha256):
            raise SourceAuthenticationError("source video differs from the manifest")
        os.chmod(target, 0o400)
        return target
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def _implementation_components() -> dict[str, str]:
    paths = {
        "meei_extractor": Path(__file__),
        "meei_manifest_builder": PROJECT_ROOT / "scripts" / "build_meei_participant_manifest.py",
        "palsynet_extractor": PROJECT_ROOT / "scripts" / "build_palsynet_v2_windows.py",
        "action_bundle": PROJECT_ROOT / "src" / "preprocessing" / "action_bundle.py",
        "clinical_landmarks": PROJECT_ROOT / "src" / "preprocessing" / "clinical_landmarks.py",
        "dynamic_landmark_loader": PROJECT_ROOT / "src" / "datasets" / "dynamic_landmark.py",
    }
    return {name: _sha256_file(path) for name, path in sorted(paths.items())}


def _aggregate_components(components: Mapping[str, str]) -> str:
    encoded = json.dumps(dict(components), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _contains_private_location(value: object, key: str | None = None) -> bool:
    if key is not None and any(token in key.lower() for token in (
        "path", "filename", "stem", "public_key",
    )):
        return True
    if isinstance(value, Mapping):
        return any(
            _contains_private_location(child, str(child_key))
            for child_key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_private_location(child) for child in value)
    if isinstance(value, str):
        return value.startswith(("/", "~")) or value.lower().endswith((".mp4", ".jpg"))
    return False


def _build_collection_manifest(
    participant_manifest: Mapping[str, object],
    retained: Sequence[object],
    exclusions: Sequence[Mapping[str, str]],
    *,
    participant_manifest_sha256: str,
    palsynet_cache_manifest_sha256: str,
    model_sha256: str,
    implementation: Mapping[str, str],
) -> dict[str, object]:
    from scripts.build_palsynet_v2_windows import _record_manifest

    records = sorted(
        (_record_manifest(result) for result in retained),
        key=lambda row: str(row["recording_id"]),
    )
    all_videos = {
        str(row["recording_id"]): row
        for row in participant_manifest["media"]
        if row["dynamic_binary_eligible"] is True
    }
    observed_ids = {str(row["recording_id"]) for row in records} | {
        str(row["recording_id"]) for row in exclusions
    }
    if observed_ids != set(all_videos) or len(records) + len(exclusions) != len(all_videos):
        raise ValueError("MEEI cache accounting omits or duplicates a manifest video")
    retained_labels = {"affected": 0, "unaffected": 0}
    for row in records:
        retained_labels[str(row["label"])] += 1
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "MEEI_Facial_Palsy_Standard_Set",
        "claim_unit": "participant",
        "identity_status": "publisher_participant_directory_one_video_each",
        "feature_schema": FEATURE_SCHEMA,
        "feature_shape": FEATURE_SHAPE,
        "capture_mirrored": None,
        "counts": {
            "participants": len(all_videos),
            "videos": len(all_videos),
            "retained": len(records),
            "excluded_label_blind_qc": len(exclusions),
            "retained_affected": retained_labels["affected"],
            "retained_unaffected": retained_labels["unaffected"],
        },
        "protocol": {
            "windows_per_recording": 4,
            "frames_per_window": 32,
            "window_placement": "palsynet_deterministic_full_recording_span",
            "minimum_coverage": MINIMUM_COVERAGE,
            "quality_exclusion_uses_label": False,
            "source_consumption": "same_descriptor_hashed_private_copy",
            "photo_decodes": 0,
        },
        "records": records,
        "excluded": sorted(
            (dict(row) for row in exclusions), key=lambda row: str(row["recording_id"])
        ),
        "provenance": {
            "participant_manifest_sha256": participant_manifest_sha256,
            "participant_media_collection_sha256": participant_manifest["provenance"][
                "participant_media_collection_sha256"
            ],
            "palsynet_cache_manifest_sha256": palsynet_cache_manifest_sha256,
            "model_sha256": model_sha256,
            "producer_sources": {
                "components": dict(implementation),
                "aggregate_sha256": _aggregate_components(implementation),
            },
        },
    }
    if _contains_private_location(manifest):
        raise ValueError("MEEI cache manifest contains a private source location")
    return manifest


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o600)


def run_extraction(
    data_root: Path,
    participant_manifest_path: Path,
    palsynet_cache_manifest_path: Path,
    model_path: Path,
    output_root: Path,
) -> dict[str, object]:
    from scripts.build_palsynet_v2_windows import (
        RecordingExtractionError,
        SourceVideo,
        extract_source_video,
        managed_extractor,
        validate_retained_recording,
        validate_staged_file_set,
        write_validated_recording_cache,
    )
    from src.preprocessing.action_bundle import MediaPipeFeatureExtractor

    root = Path(data_root).expanduser().absolute()
    output = Path(output_root).expanduser().absolute()
    canonical = root / "derived" / "clinical23_v2_windows"
    if output != canonical:
        raise ValueError("MEEI output root differs from the canonical derived cache")
    if output.exists() or output.is_symlink():
        raise FileExistsError("refusing to overwrite the MEEI dynamic cache")
    participant_manifest, participant_sha = _read_unique_json(participant_manifest_path)
    validate_participant_manifest(participant_manifest)
    palsynet_manifest, palsynet_sha = _read_unique_json(palsynet_cache_manifest_path)
    lock = validate_palsynet_extractor_lock(palsynet_manifest, model_path)
    sources = enumerate_video_sources(root, participant_manifest)
    implementation = _implementation_components()

    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(tempfile.mkdtemp(prefix=".clinical23_v2_windows.staging-", dir=output.parent))
    decoder_root = Path(tempfile.mkdtemp(prefix=".meei-decoder-", dir=output.parent))
    os.chmod(staging, 0o700)
    os.chmod(decoder_root, 0o700)
    retained: list[object] = []
    exclusions: list[dict[str, str]] = []
    try:
        with managed_extractor(MediaPipeFeatureExtractor, model_path=Path(model_path)) as extractor:
            for index, source in enumerate(sources, start=1):
                copied = copy_authenticated_source(
                    source.path, source.source_sha256, decoder_root
                )
                try:
                    decoded_source = SourceVideo(
                        path=copied,
                        source_sha256=source.source_sha256,
                        binding=source.binding,
                    )
                    try:
                        result = extract_source_video(decoded_source, extractor)
                        validate_retained_recording(result)
                    except (RecordingExtractionError, ValueError) as exc:
                        reason = str(exc) or type(exc).__name__
                        if any(token in reason.lower() for token in ("/users/", ".mp4")):
                            reason = type(exc).__name__
                        exclusions.append({
                            "recording_id": source.binding.recording_id,
                            "source_sha256": source.source_sha256,
                            "reason": reason[:160],
                        })
                        print(json.dumps({
                            "processed": index, "status": "excluded", "reason": reason[:80]
                        }), flush=True)
                        continue
                    write_validated_recording_cache(
                        staging / f"{source.binding.recording_id}.npz", result
                    )
                    retained.append(result)
                    print(json.dumps({
                        "processed": index, "status": "retained",
                        "coverage": result.coverage,
                    }), flush=True)
                finally:
                    copied.chmod(0o600)
                    copied.unlink(missing_ok=True)
        manifest = _build_collection_manifest(
            participant_manifest,
            retained,
            exclusions,
            participant_manifest_sha256=participant_sha,
            palsynet_cache_manifest_sha256=palsynet_sha,
            model_sha256=lock.model_sha256,
            implementation=implementation,
        )
        _write_json_exclusive(staging / "collection_manifest.json", manifest)
        validate_staged_file_set(staging, manifest)
        if _implementation_components() != implementation:
            raise ValueError("MEEI producer source changed during extraction")
        os.rename(staging, output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if decoder_root.exists():
            shutil.rmtree(decoder_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--participant-manifest", required=True, type=Path)
    parser.add_argument("--palsynet-cache-manifest", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    manifest = run_extraction(
        args.data_root,
        args.participant_manifest,
        args.palsynet_cache_manifest,
        args.model_path,
        args.output_root,
    )
    print(json.dumps({
        "schema_version": manifest["schema_version"],
        "counts": manifest["counts"],
        "output_created": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
