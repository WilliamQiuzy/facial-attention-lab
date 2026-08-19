#!/usr/bin/env python3
"""Build a local-only, content-deduplicated Mayo clinical23_v2 cache.

Raw paths exist only in process memory.  Generated metadata contains opaque
identifiers and SHA-256 content provenance, never names or filesystem paths.
The positive-cohort assertion is retained as an unverified challenge-cohort
assumption and does not make this cache eligible for model selection.
"""
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
from pathlib import Path
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_palsynet_v2_windows import (  # noqa: E402
    ExtractionResult,
    IdentityBinding,
    RecordingExtractionError,
    SourceVideo,
    _dependency_versions,
    _record_manifest,
    landmark_variation,
    managed_extractor,
    sha256_file,
)
from src.datasets.dynamic_landmark import (  # noqa: E402
    DYNAMIC_FEATURE_NAMES,
    DYNAMIC_FEATURE_SCHEMA,
    DYNAMIC_FEATURE_SHAPE,
)
from src.evaluation.mayo_positive_challenge_v1 import (  # noqa: E402
    ContentInventory,
    inventory_content_deduplicated_videos,
)
from src.preprocessing.action_bundle import MediaPipeFeatureExtractor  # noqa: E402


SCHEMA_VERSION = "mayo_positive_clinical23_v2_windows_v1"
EXPECTED_SOURCE_FILES = 50
EXPECTED_UNIQUE_CONTENTS = 49
EXPECTED_RETAINED_CONTENTS = 47
COARSE_FACE_SAMPLES = 96
MIN_CHALLENGE_COVERAGE = 0.75
SEEK_TOLERANCE_FRAMES = 0.25
_IMPLEMENTATION_FILES = (
    Path(__file__).resolve(),
    PROJECT_ROOT / "scripts" / "build_palsynet_v2_windows.py",
    PROJECT_ROOT / "src" / "preprocessing" / "action_bundle.py",
    PROJECT_ROOT / "src" / "preprocessing" / "clinical_landmarks.py",
    PROJECT_ROOT / "src" / "datasets" / "dynamic_landmark.py",
    PROJECT_ROOT / "src" / "evaluation" / "mayo_positive_challenge_v1.py",
)


def _implementation_components() -> dict[str, str]:
    return {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in _IMPLEMENTATION_FILES
    }


def _aggregate_digest(components: Mapping[str, str]) -> str:
    encoded = json.dumps(dict(components), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_collection_digest(inventory: ContentInventory) -> str:
    digest = hashlib.sha256()
    for record in inventory.records:
        digest.update(f"assumed_affected:{record.source_sha256}\n".encode("ascii"))
    return digest.hexdigest()


def _assert_no_private_locations(value: object, key: str | None = None) -> None:
    if key is not None and any(token in key.lower() for token in ("path", "filename", "stem")):
        raise ValueError("Mayo manifest contains a private-location key")
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            _assert_no_private_locations(child, str(child_key))
    elif isinstance(value, list):
        for child in value:
            _assert_no_private_locations(child)
    elif isinstance(value, str):
        if value.startswith(("/", "~")) or value.lower().endswith((".mov", ".mp4")):
            raise ValueError("Mayo manifest contains a raw location")


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o600)


def _binding(record) -> IdentityBinding:
    return IdentityBinding(
        source_sha256=record.source_sha256,
        recording_id=record.recording_id,
        group_id=record.group_id,
        label="affected",
        identity_status="positive_cohort_assumption_only",
        claim_unit="deduplicated_video_content",
    )


def select_face_anchored_starts(
    successful_centres: Sequence[int],
    *,
    frame_count: int,
) -> tuple[int, ...]:
    """Choose four time-spread windows using face presence only."""
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 128:
        raise ValueError("face-anchored sampling requires at least 128 frames")
    centres = sorted({
        int(value) for value in successful_centres
        if not isinstance(value, bool) and isinstance(value, (int, np.integer))
        and 16 <= int(value) <= frame_count - 17
    })
    starts = sorted({min(max(centre - 16, 0), frame_count - 32) for centre in centres})
    if len(starts) < 4 or starts[-1] - starts[0] < 96:
        raise ValueError("fewer_than_four_time_separated_face_anchors")
    targets = np.linspace(starts[0], starts[-1], 4)
    selected: list[int] = []
    for index, target in enumerate(targets):
        remaining = 3 - index
        minimum = 0 if not selected else selected[-1] + 32
        maximum = frame_count - 32 - remaining * 32
        options = [value for value in starts if minimum <= value <= maximum]
        if not options:
            raise ValueError("fewer_than_four_time_separated_face_anchors")
        selected.append(min(options, key=lambda value: (abs(value - target), value)))
    if len(set(selected)) != 4 or any(
        right - left < 32 for left, right in zip(selected, selected[1:])
    ):
        raise ValueError("fewer_than_four_time_separated_face_anchors")
    return tuple(selected)


def _finite_positive_capture_value(capture, prop: int, name: str) -> float:
    value = float(capture.get(prop))
    if not np.isfinite(value) or value <= 0:
        raise RecordingExtractionError(f"invalid_{name}")
    return value


def configure_capture_orientation(capture) -> None:
    """Apply container rotation metadata before dimensions or frames are read."""
    if not capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1):
        raise RecordingExtractionError("orientation_auto_enable_failed")
    observed = float(capture.get(cv2.CAP_PROP_ORIENTATION_AUTO))
    if not np.isfinite(observed) or observed != 1.0:
        raise RecordingExtractionError("orientation_auto_not_active")


def extract_face_anchored_video(
    source: SourceVideo,
    extractor: MediaPipeFeatureExtractor,
    *,
    capture_factory: Callable[[str], object] = cv2.VideoCapture,
) -> ExtractionResult:
    """Extract four time-spread, face-present windows without label/model scores."""
    capture = capture_factory(str(source.path))
    try:
        if not capture.isOpened():
            raise RecordingExtractionError("open_failed")
        configure_capture_orientation(capture)
        fps = _finite_positive_capture_value(capture, cv2.CAP_PROP_FPS, "fps")
        raw_count = _finite_positive_capture_value(
            capture, cv2.CAP_PROP_FRAME_COUNT, "frame_count"
        )
        if not raw_count.is_integer():
            raise RecordingExtractionError("nonintegral_frame_count")
        frame_count = int(raw_count)
        if frame_count < 128:
            raise RecordingExtractionError("insufficient_frame_count")
        raw_width = _finite_positive_capture_value(capture, cv2.CAP_PROP_FRAME_WIDTH, "width")
        raw_height = _finite_positive_capture_value(capture, cv2.CAP_PROP_FRAME_HEIGHT, "height")
        if not raw_width.is_integer() or not raw_height.is_integer():
            raise RecordingExtractionError("nonintegral_dimensions")
        width, height = int(raw_width), int(raw_height)

        lower, upper = 16, frame_count - 17
        candidate_count = min(COARSE_FACE_SAMPLES, upper - lower + 1)
        candidate_centres = np.unique(
            np.linspace(lower, upper, candidate_count).round().astype(np.int64)
        )
        successful_centres: list[int] = []
        for centre_value in candidate_centres.tolist():
            centre = int(centre_value)
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, centre):
                continue
            ok, frame = capture.read()
            if not ok or frame is None or frame.ndim != 3 or frame.size == 0:
                continue
            vector, _audit = extractor.extract_frame_with_nuisance(frame)
            if vector is not None:
                successful_centres.append(centre)
        try:
            starts = select_face_anchored_starts(
                successful_centres, frame_count=frame_count
            )
        except ValueError as exc:
            raise RecordingExtractionError(str(exc)) from exc

        source_indices = np.stack([
            np.arange(start, start + 32, dtype=np.int64) for start in starts
        ])
        timestamps = source_indices.astype(np.float64) / fps
        features = np.zeros(DYNAMIC_FEATURE_SHAPE, dtype=np.float32)
        valid_mask = np.zeros(DYNAMIC_FEATURE_SHAPE[:2], dtype=bool)
        luminance: list[float] = []
        face_scale: list[float] = []
        roll: list[float] = []
        frame_differences: list[float] = []
        for window_index, start in enumerate(starts):
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, start):
                raise RecordingExtractionError("seek_failed")
            previous_gray: np.ndarray | None = None
            for offset in range(32):
                source_index = start + offset
                ok, frame = capture.read()
                if not ok or frame is None or frame.ndim != 3 or frame.size == 0:
                    raise RecordingExtractionError("decode_failed")
                if frame.shape[:2] != (height, width):
                    raise RecordingExtractionError("frame_dimensions_changed")
                reported = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
                if not np.isfinite(reported) or abs(reported - (source_index + 1)) > SEEK_TOLERANCE_FRAMES:
                    raise RecordingExtractionError("seek_position_mismatch")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                luminance.append(float(gray.mean()))
                if previous_gray is not None:
                    frame_differences.append(float(np.abs(
                        gray.astype(np.float32) - previous_gray.astype(np.float32)
                    ).mean()))
                previous_gray = gray
                vector, audit = extractor.extract_frame_with_nuisance(frame)
                if vector is None:
                    continue
                vector = np.asarray(vector)
                if vector.shape != (95,) or vector.dtype != np.float32 or not np.isfinite(vector).all():
                    raise RecordingExtractionError("invalid_feature_vector")
                if not isinstance(audit, dict) or set(audit) != {
                    "face_scale", "eye_line_roll_degrees"
                }:
                    raise RecordingExtractionError("invalid_nuisance_audit")
                features[window_index, offset] = vector
                valid_mask[window_index, offset] = True
                face_scale.append(float(audit["face_scale"]))
                roll.append(float(audit["eye_line_roll_degrees"]))
        if float(valid_mask.mean()) < MIN_CHALLENGE_COVERAGE:
            raise RecordingExtractionError("face_anchored_coverage_below_75_percent")
        if tuple(extractor.feature_names) != DYNAMIC_FEATURE_NAMES:
            raise RecordingExtractionError("feature_name_order_mismatch")
        if extractor.feature_schema != DYNAMIC_FEATURE_SCHEMA:
            raise RecordingExtractionError("feature_schema_mismatch")
        varied, variation_stat = landmark_variation(features, valid_mask)

        def summary(values: Sequence[float], prefix: str) -> dict[str, float]:
            array = np.asarray(values, dtype=np.float64)
            return {
                f"{prefix}_mean": float(array.mean()) if array.size else 0.0,
                f"{prefix}_std": float(array.std()) if array.size else 0.0,
            }

        duration = frame_count / fps
        file_size = int(os.lstat(source.path).st_size)
        nuisance = {
            "duration_seconds": float(duration),
            "bitrate_proxy_bytes_per_second": float(file_size / duration),
            "detection_rate": float(valid_mask.mean()),
            "luminance_mean": float(np.mean(luminance)),
            "frame_difference_mean": float(np.mean(frame_differences)) if frame_differences else 0.0,
            **summary(face_scale, "face_scale"),
            **summary(roll, "eye_line_roll_degrees"),
        }
        if not all(np.isfinite(value) for value in nuisance.values()):
            raise RecordingExtractionError("nonfinite_nuisance_summary")
        return ExtractionResult(
            binding=source.binding,
            source_sha256=source.source_sha256,
            features=features,
            valid_mask=valid_mask,
            timestamps=timestamps,
            source_frame_indices=source_indices,
            source_frame_count=frame_count,
            fps=fps,
            frame_width=width,
            frame_height=height,
            file_size_bytes=file_size,
            nuisance=nuisance,
            landmark_varied=varied,
            landmark_variation_stat=variation_stat,
        )
    finally:
        capture.release()


def _challenge_cache_payload(result: ExtractionResult) -> dict[str, np.ndarray]:
    return {
        "features": result.features.astype(np.float32, copy=False),
        "valid_mask": result.valid_mask.astype(bool, copy=False),
        "timestamps": result.timestamps.astype(np.float64, copy=False),
        "timestamp_unit": np.asarray("seconds"),
        "source_frame_indices": result.source_frame_indices.astype(np.int64, copy=False),
        "source_frame_count": np.asarray(result.source_frame_count, dtype=np.int64),
        "feature_schema": np.asarray(DYNAMIC_FEATURE_SCHEMA),
        "feature_names": np.asarray(DYNAMIC_FEATURE_NAMES),
        "recording_id": np.asarray(result.binding.recording_id),
        "group_id": np.asarray(result.binding.group_id),
        "label": np.asarray(1, dtype=np.int64),
        "source_sha256": np.asarray(result.source_sha256),
    }


def write_challenge_cache(path: Path, result: ExtractionResult) -> None:
    if result.coverage < MIN_CHALLENGE_COVERAGE:
        raise ValueError("challenge cache coverage is below 75 percent")
    if path.name != f"{result.binding.recording_id}.npz":
        raise ValueError("challenge cache filename must be the opaque recording ID")
    temporary = path.parent / f".{result.binding.recording_id}.tmp-{secrets.token_hex(8)}.npz"
    try:
        with temporary.open("xb") as stream:
            np.savez(stream, **_challenge_cache_payload(result))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _build_manifest(
    inventory: ContentInventory,
    retained: list[object],
    exclusions: list[dict[str, str]],
    *,
    model_sha256: str,
    implementation_components: Mapping[str, str],
) -> dict[str, object]:
    retained_records = [_record_manifest(result) for result in retained]
    retained_digest = hashlib.sha256()
    for row in sorted(retained_records, key=lambda item: str(item["source_sha256"])):
        retained_digest.update(f"assumed_affected:{row['source_sha256']}\n".encode("ascii"))
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "Mayo_positive_challenge_v1",
        "claim_unit": "deduplicated_video_content",
        "cohort_assumption": {
            "label": "affected",
            "source": "collaborator_assertion",
            "independently_verified": False,
        },
        "eligibility": {
            "model_selection": False,
            "external_accuracy": False,
            "hb_training": False,
            "positive_confidence_challenge": True,
        },
        "inventory": {
            "source_video_files": inventory.source_files,
            "unique_video_contents": inventory.unique_contents,
            "exact_duplicate_files": inventory.exact_duplicate_files,
            "retained_unique_contents": len(retained_records),
            "quality_excluded_unique_contents": len(exclusions),
        },
        "protocol": {
            "video_container": "quicktime_mov_container",
            "deduplication": "exact_file_sha256",
            "windows_per_recording": 4,
            "frames_per_window": 32,
            "window_placement": "face_presence_only_time_spread",
            "coarse_face_samples": COARSE_FACE_SAMPLES,
            "container_orientation": "opencv_orientation_auto_enabled",
            "feature_schema": "mediapipe_bs_lr_v1+clinical23_v2",
            "minimum_detection_coverage": MIN_CHALLENGE_COVERAGE,
            "capture_mirroring": "unknown",
        },
        "records": sorted(retained_records, key=lambda item: str(item["recording_id"])),
        "exclusions": sorted(exclusions, key=lambda item: item["source_sha256"]),
        "provenance": {
            "source_collection_sha256": _source_collection_digest(inventory),
            "retained_source_collection_sha256": retained_digest.hexdigest(),
            "model_sha256": model_sha256,
            "dependency_versions": _dependency_versions(),
            "producer_sources": {
                "aggregate_sha256": _aggregate_digest(implementation_components),
                "components": dict(implementation_components),
            },
        },
    }


def run_builder(
    data_root: Path,
    model_path: Path,
    output_root: Path,
    *,
    expected_source_files: int = EXPECTED_SOURCE_FILES,
    expected_unique_contents: int = EXPECTED_UNIQUE_CONTENTS,
    expected_retained_contents: int = EXPECTED_RETAINED_CONTENTS,
) -> dict[str, object]:
    output = output_root.expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite Mayo cache {output}")
    model = model_path.expanduser().absolute()
    model_info = os.lstat(model)
    if stat.S_ISLNK(model_info.st_mode) or not stat.S_ISREG(model_info.st_mode):
        raise ValueError("MediaPipe model must be a regular file")
    inventory = inventory_content_deduplicated_videos(data_root)
    if inventory.source_files != expected_source_files:
        raise ValueError("Mayo source-file inventory differs from the frozen run")
    if inventory.unique_contents != expected_unique_contents:
        raise ValueError("Mayo exact-content inventory differs from the frozen run")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    os.chmod(staging, 0o700)
    retained: list[object] = []
    exclusions: list[dict[str, str]] = []
    try:
        with managed_extractor(MediaPipeFeatureExtractor, model_path=model) as extractor:
            for index, record in enumerate(inventory.records, start=1):
                try:
                    result = extract_face_anchored_video(
                        SourceVideo(
                            path=record.path,
                            source_sha256=record.source_sha256,
                            binding=_binding(record),
                        ),
                        extractor,
                    )
                    if not secrets.compare_digest(sha256_file(record.path), record.source_sha256):
                        raise ValueError("source_changed_during_extraction")
                    write_challenge_cache(
                        staging / f"{record.recording_id}.npz", result
                    )
                except Exception as exc:  # one bad source is an audited whole-record exclusion
                    reason = str(exc) or type(exc).__name__
                    if any(token in reason.lower() for token in ("/users/", ".mov", ".mp4")):
                        reason = type(exc).__name__
                    exclusions.append({
                        "source_sha256": record.source_sha256,
                        "reason": reason[:160],
                    })
                    print(json.dumps({"processed": index, "status": "excluded", "reason": reason[:80]}), flush=True)
                    continue
                retained.append(result)
                print(json.dumps({"processed": index, "status": "retained", "coverage": result.coverage}), flush=True)
        if len(retained) != expected_retained_contents:
            raise ValueError("Mayo retained-content count differs from the frozen run")
        if len(retained) + len(exclusions) != inventory.unique_contents:
            raise ValueError("Mayo extraction accounting does not reconcile")
        implementation = _implementation_components()
        manifest = _build_manifest(
            inventory,
            retained,
            exclusions,
            model_sha256=sha256_file(model),
            implementation_components=implementation,
        )
        _assert_no_private_locations(manifest)
        _write_json_exclusive(staging / "collection_manifest.json", manifest)
        observed_files = {path.name for path in staging.iterdir()}
        expected_files = {"collection_manifest.json"} | {
            f"{result.binding.recording_id}.npz" for result in retained
        }
        if observed_files != expected_files:
            raise ValueError("Mayo staged cache file set is incomplete or stale")
        os.rename(staging, output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-source-files", type=int, default=EXPECTED_SOURCE_FILES)
    parser.add_argument("--expected-unique-contents", type=int, default=EXPECTED_UNIQUE_CONTENTS)
    parser.add_argument("--expected-retained-contents", type=int, default=EXPECTED_RETAINED_CONTENTS)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    manifest = run_builder(
        args.data_root,
        args.model_path,
        args.output_root,
        expected_source_files=args.expected_source_files,
        expected_unique_contents=args.expected_unique_contents,
        expected_retained_contents=args.expected_retained_contents,
    )
    print(json.dumps({
        "schema_version": manifest["schema_version"],
        "inventory": manifest["inventory"],
        "output_created": True,
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
