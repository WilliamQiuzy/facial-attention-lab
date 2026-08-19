"""Build a transactional, deidentified PalsyNet clinical23_v2 window cache.

This command implements a frozen extraction protocol: exactly four 32-frame
windows per recording, identity joined only by source SHA-256, and no raw path
or filename in generated metadata.  Extraction failures exclude a complete
recording; collection-level gates then fail closed before atomic promotion.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
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
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dynamic_landmark import (  # noqa: E402
    DYNAMIC_FEATURE_NAMES,
    DYNAMIC_FEATURE_SCHEMA,
    DYNAMIC_FEATURE_SHAPE,
    MIN_RECORDING_COVERAGE,
    deterministic_window_starts,
    load_dynamic_landmark_recording,
)
from src.preprocessing.action_bundle import MediaPipeFeatureExtractor  # noqa: E402


EXPECTED_LABEL_COUNTS = {"affected": 27, "unaffected": 22}
EXPECTED_TOTAL = 49
EXPECTED_IDENTITY_GROUPS = 48
EXPECTED_FPS = 30.0
FPS_TOLERANCE = 1e-6
EXPECTED_TOTAL_FRAMES = 177_511
EXPECTED_MINIMUM_FRAMES = 172
EXPECTED_DURATION_MINUTES = 98.61722222222221
DURATION_TOLERANCE_MINUTES = 1e-12
OPENCV_DISTRIBUTIONS = (
    "opencv-python",
    "opencv-contrib-python",
    "opencv-python-headless",
    "opencv-contrib-python-headless",
)
MIN_RETAINED = 47
MIN_VARIATION_FRACTION = 0.95
SEEK_TOLERANCE_FRAMES = 0.25
COLLECTION_SCHEMA = "palsynet_clinical23_v2_windows_v1"
DYNAMIC_CACHE_FIELDS = {
    "features", "valid_mask", "timestamps", "timestamp_unit",
    "source_frame_indices", "source_frame_count", "feature_schema",
    "feature_names", "recording_id", "group_id", "label", "source_sha256",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REC_ID = re.compile(r"^rec_[0-9a-f]{64}$")
_GROUP_ID = re.compile(r"^grp_[0-9a-f]{64}$")
_IDENTITY_TOP_FIELDS = {
    "claim_unit", "contact_sheet_sampling", "contact_sheets", "counts",
    "dataset", "fingerprints", "identity_review", "ranked_pairs",
    "recordings", "schema_version",
}
_IDENTITY_RECORD_FIELDS = {
    "claim_unit", "group_id", "identity_status", "label", "recording_id",
    "source_sha256",
}
_HELD_OUTPUT_LOCKS: set[Path] = set()


class RecordingExtractionError(RuntimeError):
    """One recording cannot satisfy the frozen extraction contract."""


@dataclass(frozen=True)
class IdentityBinding:
    source_sha256: str
    recording_id: str
    group_id: str
    label: str
    identity_status: str
    claim_unit: str


@dataclass(frozen=True)
class IdentityManifest:
    by_source_sha256: Mapping[str, IdentityBinding]
    claim_unit: str
    identity_status: str
    manifest_sha256: str
    fingerprints: Mapping[str, str]


@dataclass(frozen=True)
class SourceVideo:
    """Private in-memory source join.  ``path`` is never serialized."""

    path: Path
    source_sha256: str
    binding: IdentityBinding


@dataclass(frozen=True)
class ExtractionResult:
    binding: IdentityBinding
    source_sha256: str
    features: np.ndarray
    valid_mask: np.ndarray
    timestamps: np.ndarray
    source_frame_indices: np.ndarray
    source_frame_count: int
    fps: float
    frame_width: int
    frame_height: int
    file_size_bytes: int
    nuisance: Mapping[str, float]
    landmark_varied: bool
    landmark_variation_stat: float

    @property
    def coverage(self) -> float:
        return float(self.valid_mask.mean())


@dataclass(frozen=True)
class ProvenanceSnapshot:
    source_files: tuple[tuple[Path, str], ...]
    model_file: tuple[Path, str]
    identity_manifest: tuple[Path, str]
    producer_files: tuple[tuple[str, Path, str], ...]
    producer_aggregate_sha256: str


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON object contains a duplicate key")
        result[key] = value
    return result


def _lexical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_symlink(path: Path) -> bool:
    try:
        return stat.S_ISLNK(os.lstat(path).st_mode)
    except FileNotFoundError:
        return False


def _assert_no_symlink_components(path: str | Path) -> Path:
    """Reject existing intermediate links, except root-owned top-level aliases.

    macOS exposes system-managed aliases such as ``/var -> /private/var``.
    Those root-owned top-level links are resolved once; every descendant link
    remains forbidden so a user-controlled intermediate component cannot
    escape a trusted tree.
    """
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


def _require_regular_file(path: str | Path, field: str) -> Path:
    checked = _assert_no_symlink_components(path)
    try:
        info = os.lstat(checked)
    except FileNotFoundError as exc:
        raise ValueError(f"{field} is missing") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{field} must be a regular file")
    return checked


def _require_directory(path: str | Path, field: str) -> Path:
    checked = _assert_no_symlink_components(path)
    try:
        info = os.lstat(checked)
    except FileNotFoundError as exc:
        raise ValueError(f"{field} is missing") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{field} must be a real directory")
    return checked


def sha256_file(path: str | Path) -> str:
    checked = _require_regular_file(path, "hashed file")
    digest = hashlib.sha256()
    with checked.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_cli_paths(
    data_root: str | Path,
    model_path: str | Path,
    identity_manifest: str | Path,
    output_root: str | Path,
) -> tuple[Path, Path, Path, Path]:
    data = _require_directory(data_root, "data root")
    model = _require_regular_file(model_path, "MediaPipe model")
    identity = _require_regular_file(identity_manifest, "identity manifest")
    output = _assert_no_symlink_components(output_root)
    _assert_no_symlink_components(output.parent)
    canonical = data.parent / "derived" / "clinical23_v2_windows"
    if output != canonical:
        raise ValueError(
            "output root must be exactly data-root.parent/derived/clinical23_v2_windows"
        )
    if output.exists() and not output.is_dir():
        raise ValueError("output root must be a real directory when it exists")
    return data, model, identity, output


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be 64 lowercase hexadecimal characters")
    return value


def _require_opaque(value: object, pattern: re.Pattern[str], field: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{field} is not a canonical opaque identifier")
    return value


def source_collection_fingerprint(
    label_hash_pairs: Sequence[tuple[str, str]],
) -> str:
    """Match the identity audit's canonical ``label:sha256\n`` fingerprint."""
    normalized: list[tuple[str, str]] = []
    for label, digest in label_hash_pairs:
        if label not in EXPECTED_LABEL_COUNTS:
            raise ValueError("source fingerprint label is invalid")
        normalized.append((label, _require_sha256(digest, "source_sha256")))
    fingerprint = hashlib.sha256()
    for label, digest in sorted(normalized):
        fingerprint.update(f"{label}:{digest}\n".encode("ascii"))
    return fingerprint.hexdigest()


def load_identity_manifest(path: str | Path) -> IdentityManifest:
    """Load the exact audit schema and return a source-hash join table."""
    checked = _require_regular_file(path, "identity manifest")
    raw = checked.read_bytes()
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_json_object)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("identity manifest is not valid unique-key JSON") from exc
    if not isinstance(payload, dict) or set(payload) != _IDENTITY_TOP_FIELDS:
        raise ValueError("identity manifest has an unexpected top-level schema")
    if payload["schema_version"] != "palsynet_identity_audit_v1":
        raise ValueError("identity manifest schema version is unsupported")
    if payload["dataset"] != "PalsyNet":
        raise ValueError("identity manifest dataset must be PalsyNet")

    claim_unit = payload["claim_unit"]
    review = payload["identity_review"]
    if not isinstance(review, dict) or set(review) != {
        "group_override_applied", "manual_review_required",
        "reviewer_evidence_sha256", "status",
    }:
        raise ValueError("identity review block has an unexpected schema")
    identity_status = review["status"]
    valid_review = (
        identity_status == "unreviewed"
        and claim_unit == "video_held_out"
        and review["manual_review_required"] is True
        and review["reviewer_evidence_sha256"] is None
    ) or (
        identity_status == "reviewed"
        and claim_unit == "person_held_out"
        and review["manual_review_required"] is False
        and isinstance(review["reviewer_evidence_sha256"], str)
        and _SHA256.fullmatch(review["reviewer_evidence_sha256"]) is not None
    )
    if not valid_review or review["group_override_applied"] is not True:
        raise ValueError("identity status and claim unit are inconsistent")

    counts = payload["counts"]
    if not isinstance(counts, dict) or set(counts) != {
        "affected", "ranked_pairs", "total", "unaffected",
    }:
        raise ValueError("identity counts block has an unexpected schema")
    if any(counts[label] != expected for label, expected in EXPECTED_LABEL_COUNTS.items()):
        raise ValueError("identity manifest must contain exactly 27/22 labels")
    if counts["total"] != EXPECTED_TOTAL or counts["ranked_pairs"] != 1176:
        raise ValueError("identity manifest counts are not the frozen 49-video audit")

    fingerprints = payload["fingerprints"]
    if not isinstance(fingerprints, dict) or set(fingerprints) != {
        "bundle_provenance_sha256", "embedding_collection_sha256",
        "source_collection_sha256",
    }:
        raise ValueError("identity fingerprint block has an unexpected schema")
    fingerprints = {
        key: _require_sha256(value, f"identity fingerprint {key}")
        for key, value in fingerprints.items()
    }

    sampling = payload["contact_sheet_sampling"]
    if sampling != {
        "raw_filename_text_burned_in": False,
        "representative_frame_offset": 16,
        "window_size_frames": 32,
        "windows_per_video": 4,
    }:
        raise ValueError("identity contact-sheet sampling contract changed")
    contact = payload["contact_sheets"]
    if contact != {
        "filenames": "opaque_ids_or_ranks_only",
        "ranked_pairs": 25,
        "recordings": 49,
        "storage": "local_ignored_output",
    }:
        raise ValueError("identity contact-sheet generation contract changed")

    rows = payload["recordings"]
    if not isinstance(rows, list) or len(rows) != EXPECTED_TOTAL:
        raise ValueError("identity manifest must contain exactly 49 recording rows")
    bindings: dict[str, IdentityBinding] = {}
    recording_ids: set[str] = set()
    group_labels: dict[str, str] = {}
    observed_counts = {label: 0 for label in EXPECTED_LABEL_COUNTS}
    for row in rows:
        if not isinstance(row, dict) or set(row) != _IDENTITY_RECORD_FIELDS:
            raise ValueError("identity recording row has an unexpected schema")
        digest = _require_sha256(row["source_sha256"], "source_sha256")
        recording_id = _require_opaque(row["recording_id"], _REC_ID, "recording_id")
        group_id = _require_opaque(row["group_id"], _GROUP_ID, "group_id")
        label = row["label"]
        if label not in EXPECTED_LABEL_COUNTS:
            raise ValueError("identity label must be affected or unaffected")
        if row["identity_status"] != identity_status or row["claim_unit"] != claim_unit:
            raise ValueError("recording identity claim differs from manifest claim")
        if digest in bindings or recording_id in recording_ids:
            raise ValueError("source hashes and recording IDs must be unique")
        previous_label = group_labels.setdefault(group_id, label)
        if previous_label != label:
            raise ValueError("one identity group cannot cross labels")
        observed_counts[label] += 1
        recording_ids.add(recording_id)
        bindings[digest] = IdentityBinding(
            source_sha256=digest,
            recording_id=recording_id,
            group_id=group_id,
            label=label,
            identity_status=identity_status,
            claim_unit=claim_unit,
        )
    if observed_counts != EXPECTED_LABEL_COUNTS:
        raise ValueError("identity recording labels do not match the declared 27/22 counts")
    if len(group_labels) != EXPECTED_IDENTITY_GROUPS:
        raise ValueError("identity manifest must preserve the confirmed 48 groups")
    observed_source_fingerprint = source_collection_fingerprint([
        (binding.label, binding.source_sha256) for binding in bindings.values()
    ])
    if observed_source_fingerprint != fingerprints["source_collection_sha256"]:
        raise ValueError("identity source collection fingerprint does not match its rows")

    pairs = payload["ranked_pairs"]
    if not isinstance(pairs, list) or len(pairs) != 1176:
        raise ValueError("identity manifest must retain all ranked pairs")
    seen_pairs: set[tuple[str, str]] = set()
    for expected_rank, pair in enumerate(pairs, start=1):
        if not isinstance(pair, dict) or set(pair) != {
            "cosine", "rank", "recording_id_a", "recording_id_b",
        }:
            raise ValueError("ranked identity pair has an unexpected schema")
        if pair["rank"] != expected_rank:
            raise ValueError("ranked identity pairs must be contiguous")
        if pair["recording_id_a"] not in recording_ids or pair["recording_id_b"] not in recording_ids:
            raise ValueError("ranked identity pair references an unknown recording")
        if pair["recording_id_a"] == pair["recording_id_b"]:
            raise ValueError("ranked identity pair cannot compare a recording to itself")
        pair_key = tuple(sorted((pair["recording_id_a"], pair["recording_id_b"])))
        if pair_key in seen_pairs:
            raise ValueError("ranked identity pairs must be unique")
        seen_pairs.add(pair_key)
        cosine = pair["cosine"]
        if (
            isinstance(cosine, bool)
            or not isinstance(cosine, (int, float))
            or not math.isfinite(cosine)
            or not -1.0 <= float(cosine) <= 1.0
        ):
            raise ValueError("ranked pair cosine must be finite and within [-1, 1]")

    return IdentityManifest(
        by_source_sha256=bindings,
        claim_unit=claim_unit,
        identity_status=identity_status,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        fingerprints=fingerprints,
    )


def _source_label(path: Path, data_root: Path) -> str:
    try:
        relative = path.relative_to(data_root)
    except ValueError as exc:
        raise ValueError("source video escaped data root") from exc
    if len(relative.parts) != 2 or relative.parts[0] not in EXPECTED_LABEL_COUNTS:
        raise ValueError("source videos must be direct children of affected/unaffected")
    return relative.parts[0]


def enumerate_source_videos(
    data_root: str | Path,
    identity: IdentityManifest,
) -> tuple[SourceVideo, ...]:
    """Hash all MP4s and join to identity strictly by content digest."""
    root = _require_directory(data_root, "data root")
    videos: list[Path] = []
    for label in EXPECTED_LABEL_COUNTS:
        label_root = _require_directory(root / label, f"{label} source directory")
        for child in sorted(label_root.iterdir(), key=lambda item: item.name):
            info = os.lstat(child)
            if stat.S_ISLNK(info.st_mode):
                raise ValueError("source collection must not contain symlinks")
            if stat.S_ISDIR(info.st_mode):
                raise ValueError("source label directories must not contain subdirectories")
            if stat.S_ISREG(info.st_mode) and child.suffix.lower() == ".mp4":
                videos.append(child)
    if len(videos) != EXPECTED_TOTAL:
        raise ValueError("source collection must contain exactly 49 MP4 files")

    joined: list[SourceVideo] = []
    observed_hashes: set[str] = set()
    for path in videos:
        digest = sha256_file(path)
        if digest in observed_hashes:
            raise ValueError("source collection contains duplicate file content")
        observed_hashes.add(digest)
        binding = identity.by_source_sha256.get(digest)
        if binding is None:
            raise ValueError("source hash is absent from the identity manifest")
        if binding.label != _source_label(path, root):
            raise ValueError("source directory label differs from identity audit label")
        joined.append(SourceVideo(path=path, source_sha256=digest, binding=binding))
    expected_hashes = set(identity.by_source_sha256)
    if observed_hashes != expected_hashes:
        raise ValueError("source hash coverage differs from the identity manifest")
    return tuple(sorted(joined, key=lambda item: item.binding.recording_id))


def preflight_corpus_metadata(
    sources: Sequence[SourceVideo],
    capture_factory: Callable[[str], object] = cv2.VideoCapture,
) -> dict[str, float | int]:
    """Verify the frozen whole-corpus inventory before MediaPipe is created.

    OpenCV may report FPS with harmless floating representation noise, so the
    audited 30-Hz value uses a strict absolute tolerance of ``1e-6``.  Duration
    is recomputed from every integral frame count and its verified FPS.
    """
    if len(sources) != EXPECTED_TOTAL:
        raise ValueError("metadata preflight requires exactly 49 sources")
    frame_counts: list[int] = []
    duration_seconds = 0.0
    for source in sources:
        capture = capture_factory(str(source.path))
        try:
            if not capture.isOpened():
                raise ValueError("corpus metadata video could not be opened")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            raw_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if (
                not math.isfinite(fps)
                or abs(fps - EXPECTED_FPS) > FPS_TOLERANCE
            ):
                raise ValueError("every PalsyNet source must match the audited 30-Hz FPS")
            if (
                not math.isfinite(raw_count)
                or raw_count <= 0
                or not raw_count.is_integer()
            ):
                raise ValueError("every PalsyNet frame count must be a positive integer")
            frame_count = int(raw_count)
            frame_counts.append(frame_count)
            duration_seconds += frame_count / fps
        finally:
            capture.release()

    total_frames = sum(frame_counts)
    minimum_frames = min(frame_counts)
    duration_minutes = duration_seconds / 60.0
    if total_frames != EXPECTED_TOTAL_FRAMES:
        raise ValueError("PalsyNet total frame count differs from the audited 177,511")
    if minimum_frames != EXPECTED_MINIMUM_FRAMES:
        raise ValueError("PalsyNet minimum frame count differs from the audited 172")
    if abs(duration_minutes - EXPECTED_DURATION_MINUTES) > DURATION_TOLERANCE_MINUTES:
        raise ValueError("PalsyNet derived duration differs from the audited inventory")
    return {
        "recordings": len(sources),
        "fps": EXPECTED_FPS,
        "total_frames": total_frames,
        "minimum_frames": minimum_frames,
        "duration_minutes": duration_minutes,
    }


def landmark_variation(
    features: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[bool, float]:
    features = np.asarray(features)
    mask = np.asarray(valid_mask)
    if features.shape != DYNAMIC_FEATURE_SHAPE or mask.shape != DYNAMIC_FEATURE_SHAPE[:2]:
        raise ValueError("variation input must use the fixed dynamic cache shape")
    statistic = 0.0
    for window_index in range(DYNAMIC_FEATURE_SHAPE[0]):
        valid_landmarks = features[window_index, :, -23:][mask[window_index]]
        if valid_landmarks.shape[0] < 2:
            continue
        ranges = np.ptp(valid_landmarks.astype(np.float64, copy=False), axis=0)
        statistic = max(statistic, float(np.max(ranges)))
    if not math.isfinite(statistic):
        raise ValueError("landmark variation must be finite")
    return statistic > 0.0, statistic


def _finite_positive_metadata(capture, prop: int, name: str) -> float:
    raw = float(capture.get(prop))
    if not math.isfinite(raw) or raw <= 0:
        raise RecordingExtractionError(f"invalid_{name}")
    return raw


def extract_source_video(
    source: SourceVideo,
    extractor: MediaPipeFeatureExtractor,
    capture_factory: Callable[[str], object] = cv2.VideoCapture,
) -> ExtractionResult:
    """Decode the exact frozen indices and extract one recording atomically."""
    capture = capture_factory(str(source.path))
    try:
        if not capture.isOpened():
            raise RecordingExtractionError("open_failed")
        fps = _finite_positive_metadata(capture, cv2.CAP_PROP_FPS, "fps")
        raw_count = _finite_positive_metadata(
            capture, cv2.CAP_PROP_FRAME_COUNT, "frame_count"
        )
        if not raw_count.is_integer():
            raise RecordingExtractionError("nonintegral_frame_count")
        frame_count = int(raw_count)
        try:
            starts = deterministic_window_starts(frame_count)
        except ValueError as exc:
            raise RecordingExtractionError("insufficient_frame_count") from exc
        raw_width = _finite_positive_metadata(capture, cv2.CAP_PROP_FRAME_WIDTH, "width")
        raw_height = _finite_positive_metadata(capture, cv2.CAP_PROP_FRAME_HEIGHT, "height")
        if not raw_width.is_integer() or not raw_height.is_integer():
            raise RecordingExtractionError("nonintegral_dimensions")
        width, height = int(raw_width), int(raw_height)

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
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, int(start)):
                raise RecordingExtractionError("seek_failed")
            previous_gray: np.ndarray | None = None
            for offset in range(32):
                source_index = int(start + offset)
                ok, frame = capture.read()
                if (
                    not ok or frame is None or not isinstance(frame, np.ndarray)
                    or frame.ndim != 3 or frame.size == 0
                ):
                    raise RecordingExtractionError("decode_failed")
                if frame.shape[1] != width or frame.shape[0] != height:
                    raise RecordingExtractionError("frame_dimensions_changed")
                reported = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
                if (
                    not math.isfinite(reported)
                    or abs(reported - float(source_index + 1)) > SEEK_TOLERANCE_FRAMES
                ):
                    raise RecordingExtractionError("seek_position_mismatch")

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                luminance.append(float(gray.mean()))
                if previous_gray is not None:
                    delta = np.abs(
                        gray.astype(np.float32) - previous_gray.astype(np.float32)
                    )
                    frame_differences.append(float(delta.mean()))
                previous_gray = gray

                vector, audit = extractor.extract_frame_with_nuisance(frame)
                if vector is None:
                    continue
                vector = np.asarray(vector)
                if (
                    vector.shape != (95,) or vector.dtype != np.float32
                    or not np.isfinite(vector).all()
                ):
                    raise RecordingExtractionError("invalid_feature_vector")
                if not isinstance(audit, dict) or set(audit) != {
                    "face_scale", "eye_line_roll_degrees",
                }:
                    raise RecordingExtractionError("invalid_nuisance_audit")
                scale_value = float(audit["face_scale"])
                roll_value = float(audit["eye_line_roll_degrees"])
                if not math.isfinite(scale_value) or scale_value <= 0 or not math.isfinite(roll_value):
                    raise RecordingExtractionError("invalid_nuisance_audit")
                features[window_index, offset] = vector
                valid_mask[window_index, offset] = True
                face_scale.append(scale_value)
                roll.append(roll_value)

        if not valid_mask.any():
            raise RecordingExtractionError("no_valid_detections")
        if tuple(extractor.feature_names) != DYNAMIC_FEATURE_NAMES:
            raise RecordingExtractionError("feature_name_order_mismatch")
        if extractor.feature_schema != DYNAMIC_FEATURE_SCHEMA:
            raise RecordingExtractionError("feature_schema_mismatch")
        if getattr(extractor, "capture_mirrored", None) is not None:
            raise RecordingExtractionError("capture_mirroring_must_be_unknown")

        varied, variation_stat = landmark_variation(features, valid_mask)
        duration = frame_count / fps
        file_size = int(os.lstat(source.path).st_size)

        def summary(values: Sequence[float], prefix: str) -> dict[str, float]:
            array = np.asarray(values, dtype=np.float64)
            if array.size == 0:
                return {f"{prefix}_mean": 0.0, f"{prefix}_std": 0.0}
            return {
                f"{prefix}_mean": float(array.mean()),
                f"{prefix}_std": float(array.std()),
            }

        nuisance = {
            "duration_seconds": float(duration),
            "bitrate_proxy_bytes_per_second": float(file_size / duration),
            "detection_rate": float(valid_mask.mean()),
            "luminance_mean": float(np.mean(luminance)),
            "frame_difference_mean": (
                float(np.mean(frame_differences)) if frame_differences else 0.0
            ),
            **summary(face_scale, "face_scale"),
            **summary(roll, "eye_line_roll_degrees"),
        }
        if not all(math.isfinite(value) for value in nuisance.values()):
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
    except RecordingExtractionError:
        raise
    except (OSError, ValueError, cv2.error) as exc:
        raise RecordingExtractionError("unexpected_decode_or_feature_error") from exc
    finally:
        capture.release()


def validate_retained_recording(result: ExtractionResult) -> None:
    if result.coverage < MIN_RECORDING_COVERAGE:
        raise ValueError("recording coverage is below 90 percent")
    if result.features.shape != DYNAMIC_FEATURE_SHAPE or result.features.dtype != np.float32:
        raise ValueError("recording feature tensor violates the fixed contract")
    if not np.isfinite(result.features[result.valid_mask]).all():
        raise ValueError("valid recording features must be finite")
    if np.any(result.features[~result.valid_mask] != 0):
        raise ValueError("detector misses must remain canonical zero")


def _cache_payload(result: ExtractionResult) -> dict[str, np.ndarray]:
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
        "label": np.asarray(1 if result.binding.label == "affected" else 0, dtype=np.int64),
        "source_sha256": np.asarray(result.source_sha256),
    }


def write_validated_recording_cache(path: str | Path, result: ExtractionResult) -> None:
    validate_retained_recording(result)
    path = _lexical_absolute(path)
    if path.name != f"{result.binding.recording_id}.npz":
        raise ValueError("cache filename must be the opaque recording ID only")
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(path.parent)
    temporary = path.parent / f".{result.binding.recording_id}.tmp-{secrets.token_hex(8)}.npz"
    try:
        with temporary.open("xb") as handle:
            np.savez(handle, **_cache_payload(result))
            handle.flush()
            os.fsync(handle.fileno())
        loaded = load_dynamic_landmark_recording(temporary)
        if (
            loaded.recording_id != result.binding.recording_id
            or loaded.group_id != result.binding.group_id
            or loaded.source_sha256 != result.source_sha256
        ):
            raise ValueError("reread cache metadata differs from extraction result")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_collection_gate(
    records: Sequence[object],
    expected_total: int = EXPECTED_TOTAL,
) -> None:
    if len(records) < MIN_RETAINED or len(records) > expected_total:
        raise ValueError("collection must retain between 47 and 49 recordings")
    recording_ids: set[str] = set()
    group_labels: dict[str, str] = {}
    groups_by_label = {label: set() for label in EXPECTED_LABEL_COUNTS}
    varied = 0
    for result in records:
        binding = result.binding
        if binding.recording_id in recording_ids:
            raise ValueError("retained recording IDs must be unique")
        recording_ids.add(binding.recording_id)
        if result.coverage < MIN_RECORDING_COVERAGE:
            raise ValueError("every retained recording must meet 90 percent coverage")
        if binding.label not in groups_by_label:
            raise ValueError("retained label is invalid")
        previous = group_labels.setdefault(binding.group_id, binding.label)
        if previous != binding.label:
            raise ValueError("retained group crosses labels")
        groups_by_label[binding.label].add(binding.group_id)
        varied += int(bool(result.landmark_varied))
    if any(len(groups) < 5 for groups in groups_by_label.values()):
        raise ValueError("each label requires at least five groups for five-fold CV")
    if varied / len(records) < MIN_VARIATION_FRACTION:
        raise ValueError("fewer than 95 percent of retained videos have landmark variation")


def snapshot_provenance(
    sources: Sequence[SourceVideo],
    model_path: str | Path,
    identity_manifest: str | Path,
    *,
    producer_paths: Mapping[str, str | Path] | None = None,
) -> ProvenanceSnapshot:
    source_files: list[tuple[Path, str]] = []
    for source in sources:
        path = _require_regular_file(source.path, "source video")
        observed = sha256_file(path)
        expected = _require_sha256(source.source_sha256, "expected source_sha256")
        if not secrets.compare_digest(observed, expected):
            raise ValueError("source content changed after identity enumeration")
        source_files.append((path, expected))
    source_files.sort(key=lambda item: str(item[0]))

    model = _require_regular_file(model_path, "MediaPipe model")
    identity = _require_regular_file(identity_manifest, "identity manifest")
    if producer_paths is None:
        producer_paths = {
            "builder": Path(__file__),
            "action_bundle": PROJECT_ROOT / "src" / "preprocessing" / "action_bundle.py",
            "clinical_landmarks": PROJECT_ROOT / "src" / "preprocessing" / "clinical_landmarks.py",
            "dynamic_landmark_loader": PROJECT_ROOT / "src" / "datasets" / "dynamic_landmark.py",
            "feature_registry": PROJECT_ROOT / "src" / "datasets" / "patient_multistream.py",
        }
    if not producer_paths:
        raise ValueError("producer source closure must not be empty")
    producer_files: list[tuple[str, Path, str]] = []
    for logical_name, producer_path in sorted(producer_paths.items()):
        if (
            not isinstance(logical_name, str)
            or not logical_name
            or not re.fullmatch(r"[a-z][a-z0-9_]*", logical_name)
        ):
            raise ValueError("producer source logical names must be stable identifiers")
        checked = _require_regular_file(producer_path, "producer source")
        producer_files.append((logical_name, checked, sha256_file(checked)))
    aggregate = hashlib.sha256()
    for logical_name, _path, digest in producer_files:
        aggregate.update(f"{logical_name}:{digest}\n".encode("ascii"))
    return ProvenanceSnapshot(
        source_files=tuple(source_files),
        model_file=(model, sha256_file(model)),
        identity_manifest=(identity, sha256_file(identity)),
        producer_files=tuple(producer_files),
        producer_aggregate_sha256=aggregate.hexdigest(),
    )


def assert_provenance_unchanged(snapshot: ProvenanceSnapshot) -> None:
    for path, expected in (
        *snapshot.source_files,
        snapshot.model_file,
        snapshot.identity_manifest,
    ):
        if sha256_file(path) != expected:
            raise ValueError("source, model, or identity manifest changed before promotion")
    aggregate = hashlib.sha256()
    for logical_name, path, expected in snapshot.producer_files:
        observed = sha256_file(path)
        if not secrets.compare_digest(observed, expected):
            raise ValueError("producer source closure changed before promotion")
        aggregate.update(f"{logical_name}:{expected}\n".encode("ascii"))
    if not secrets.compare_digest(
        aggregate.hexdigest(), snapshot.producer_aggregate_sha256
    ):
        raise ValueError("producer source aggregate changed before promotion")


def _manifest_contains_private_location(value: object, key: str | None = None) -> bool:
    if key is not None and any(token in key.lower() for token in ("path", "filename", "stem")):
        return True
    if isinstance(value, dict):
        return any(_manifest_contains_private_location(item, str(item_key))
                   for item_key, item in value.items())
    if isinstance(value, list):
        return any(_manifest_contains_private_location(item) for item in value)
    if isinstance(value, str):
        lower = value.lower()
        return value.startswith(("/", "~")) or lower.endswith((".mp4", ".mov"))
    return False


def validate_staged_file_set(staging_root: str | Path, manifest: Mapping[str, object]) -> None:
    staging = _require_directory(staging_root, "staging root")
    if _manifest_contains_private_location(manifest):
        raise ValueError("collection manifest contains a raw path or filename")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("collection manifest records must be a list")
    expected = {Path("collection_manifest.json")}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("collection record must be an object")
        recording_id = _require_opaque(record.get("recording_id"), _REC_ID, "recording_id")
        _require_sha256(record.get("source_sha256"), "source_sha256")
        expected.add(Path(f"{recording_id}.npz"))

    observed: set[Path] = set()
    for current, directories, files in os.walk(staging, followlinks=False):
        if directories:
            raise ValueError("staged generation must not contain subdirectories")
        for filename in files:
            candidate = Path(current) / filename
            info = os.lstat(candidate)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise ValueError("staged generation contains an unsafe file")
            observed.add(candidate.relative_to(staging))
    if observed != expected:
        raise ValueError("staged cache file set is stale, missing, or mixed")
    try:
        disk_manifest = json.loads(
            (staging / "collection_manifest.json").read_text(),
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("staged collection manifest is invalid") from exc
    if disk_manifest != manifest:
        raise ValueError("staged collection manifest differs from validated memory")
    for record in records:
        recording_id = str(record["recording_id"])
        loaded = load_dynamic_landmark_recording(staging / f"{recording_id}.npz")
        expected_label = 1 if record.get("label") == "affected" else (
            0 if record.get("label") == "unaffected" else None
        )
        if expected_label is None:
            raise ValueError("collection record label is invalid")
        if (
            loaded.recording_id != recording_id
            or loaded.group_id != record.get("group_id")
            or loaded.source_sha256 != record.get("source_sha256")
            or loaded.label != expected_label
            or loaded.source_frame_count != record.get("source_frame_count")
        ):
            raise ValueError("collection record identity/source metadata differs from NPZ")
        starts = record.get("window_starts")
        if (
            not isinstance(starts, list)
            or len(starts) != DYNAMIC_FEATURE_SHAPE[0]
            or any(isinstance(value, bool) or not isinstance(value, int) for value in starts)
            or starts != loaded.source_frame_indices[:, 0].tolist()
        ):
            raise ValueError("collection window starts differ from NPZ")
        if record.get("frames_per_window") != DYNAMIC_FEATURE_SHAPE[1]:
            raise ValueError("collection frames-per-window differs from NPZ contract")
        if record.get("timestamp_unit") != loaded.timestamp_unit:
            raise ValueError("collection timestamp unit differs from NPZ")
        fps = record.get("fps")
        if (
            isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or not math.isfinite(float(fps))
            or float(fps) <= 0
        ):
            raise ValueError("collection FPS must be finite and positive")
        expected_timestamps = loaded.source_frame_indices.astype(np.float64) / float(fps)
        if not np.array_equal(loaded.timestamps, expected_timestamps):
            raise ValueError("collection FPS cannot reconstruct the exact NPZ timestamps")


def _remove_real_tree(path: Path) -> None:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("generation cleanup target must be a real directory")
    shutil.rmtree(path)


@contextmanager
def output_parent_lock(output_root: str | Path):
    """Hold a nonblocking process lock for one output generation lifecycle."""
    output = _lexical_absolute(output_root)
    parent = _assert_no_symlink_components(output.parent)
    if not parent.is_dir():
        raise ValueError("output parent must exist as a real directory before locking")
    lock_path = parent / f".{output.name}.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(lock_path, flags, 0o600)
    acquired = False
    registered = False
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("output lock must be a regular file")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another PalsyNet builder holds the output lock") from exc
        acquired = True
        if output in _HELD_OUTPUT_LOCKS:
            raise RuntimeError("output lock is already held in this process")
        _HELD_OUTPUT_LOCKS.add(output)
        registered = True
        yield
    finally:
        if registered:
            _HELD_OUTPUT_LOCKS.discard(output)
        if acquired:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _require_output_lock(output: Path) -> None:
    if output not in _HELD_OUTPUT_LOCKS:
        raise RuntimeError("output lifecycle mutation requires the exclusive lock")


def _validated_generation_directories(paths: Sequence[Path]) -> tuple[Path, ...]:
    checked: list[Path] = []
    for path in paths:
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("generation recovery candidate must be a real directory")
        checked.append(path)
    return tuple(checked)


def recover_interrupted_generations(output_root: str | Path) -> None:
    """Under the lock, remove staging and deterministically recover backups."""
    output = _lexical_absolute(output_root)
    _require_output_lock(output)
    parent = _assert_no_symlink_components(output.parent)
    staging = _validated_generation_directories(sorted(
        parent.glob(f".{output.name}.staging-*"), key=lambda item: item.name
    ))
    backups = _validated_generation_directories(sorted(
        parent.glob(f".{output.name}.backup-*"), key=lambda item: item.name
    ))
    for candidate in staging:
        _remove_real_tree(candidate)
    if output.exists() or _is_symlink(output):
        _require_directory(output, "existing output generation")
        for backup in backups:
            _remove_real_tree(backup)
        return
    if len(backups) == 1:
        os.replace(backups[0], output)
    elif len(backups) > 1:
        raise ValueError("multiple interrupted backups are ambiguous; refusing recovery")


def promote_generation(staging_root: str | Path, output_root: str | Path) -> None:
    staging = _require_directory(staging_root, "staging root")
    output = _lexical_absolute(output_root)
    _require_output_lock(output)
    if staging.parent != output.parent:
        raise ValueError("staging generation must be a sibling of output")
    _assert_no_symlink_components(output)
    backup = output.parent / f".{output.name}.backup-{secrets.token_hex(8)}"
    old_moved = False
    try:
        if output.exists():
            _require_directory(output, "previous output")
            os.replace(output, backup)
            old_moved = True
        try:
            os.replace(staging, output)
        except BaseException:
            if old_moved:
                if output.exists() or _is_symlink(output):
                    raise RuntimeError("cannot restore prior generation because output reappeared")
                os.replace(backup, output)
                old_moved = False
            raise
        if old_moved:
            _remove_real_tree(backup)
            old_moved = False
    finally:
        if old_moved and backup.exists() and not output.exists():
            os.replace(backup, output)


def _dependency_versions(
    *,
    version_resolver: Callable[[str], str] = importlib.metadata.version,
) -> dict[str, str]:
    """Return exact installed distributions, rejecting ambiguous OpenCV wheels."""
    python_version = platform.python_version()
    if not python_version or python_version.lower() == "unknown":
        raise ValueError("Python runtime version is unavailable")

    def required(package: str) -> str:
        try:
            version = version_resolver(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError(f"required distribution {package} is not installed") from exc
        if not isinstance(version, str) or not version or version.lower() == "unknown":
            raise ValueError(f"required distribution {package} has no exact version")
        return version

    numpy_version = required("numpy")
    mediapipe_version = required("mediapipe")
    torch_version = required("torch")
    installed_opencv: list[tuple[str, str]] = []
    for distribution in OPENCV_DISTRIBUTIONS:
        try:
            version = version_resolver(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
        if not isinstance(version, str) or not version or version.lower() == "unknown":
            raise ValueError("installed OpenCV distribution has no exact version")
        installed_opencv.append((distribution, version))
    if len(installed_opencv) != 1:
        raise ValueError(
            "exactly one OpenCV wheel distribution must be installed; "
            f"found {len(installed_opencv)}"
        )
    opencv_name, opencv_version = installed_opencv[0]
    return {
        "python": f"python=={python_version}",
        "numpy": f"numpy=={numpy_version}",
        "mediapipe": f"mediapipe=={mediapipe_version}",
        "torch": f"torch=={torch_version}",
        "opencv": f"{opencv_name}=={opencv_version}",
    }


def _record_manifest(result: ExtractionResult) -> dict[str, object]:
    return {
        "recording_id": result.binding.recording_id,
        "group_id": result.binding.group_id,
        "source_sha256": result.source_sha256,
        "label": result.binding.label,
        "source_frame_count": result.source_frame_count,
        "fps": result.fps,
        "window_starts": result.source_frame_indices[:, 0].tolist(),
        "frames_per_window": DYNAMIC_FEATURE_SHAPE[1],
        "timestamp_unit": "seconds",
        "frame_width": result.frame_width,
        "frame_height": result.frame_height,
        "file_size_bytes": result.file_size_bytes,
        "coverage": result.coverage,
        "landmark_varied": result.landmark_varied,
        "landmark_variation_stat": result.landmark_variation_stat,
        "nuisance": dict(result.nuisance),
    }


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    serialized = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def managed_extractor(
    extractor_factory: Callable[..., MediaPipeFeatureExtractor],
    *,
    model_path: str | Path,
):
    """Construct once and close the actual FaceLandmarker on every exit."""
    extractor = extractor_factory(
        model_path=model_path,
        landmark_features="clinical23",
        capture_mirrored=None,
    )
    try:
        yield extractor
    finally:
        extractor.close()


def run_builder(
    data_root: str | Path,
    model_path: str | Path,
    identity_manifest: str | Path,
    output_root: str | Path,
    *,
    extractor_factory: Callable[..., MediaPipeFeatureExtractor] = MediaPipeFeatureExtractor,
    capture_factory: Callable[[str], object] = cv2.VideoCapture,
) -> dict[str, object]:
    data, model, identity_path, output = validate_cli_paths(
        data_root, model_path, identity_manifest, output_root
    )
    identity = load_identity_manifest(identity_path)
    sources = enumerate_source_videos(data, identity)
    inventory = preflight_corpus_metadata(sources, capture_factory=capture_factory)
    observed_source_fingerprint = source_collection_fingerprint([
        (source.binding.label, source.source_sha256) for source in sources
    ])
    if observed_source_fingerprint != identity.fingerprints["source_collection_sha256"]:
        raise ValueError("enumerated source collection differs from identity fingerprint")
    provenance = snapshot_provenance(sources, model, identity_path)
    if provenance.identity_manifest[1] != identity.manifest_sha256:
        raise ValueError("identity manifest changed while it was being loaded")

    output.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(output.parent)
    with output_parent_lock(output):
        recover_interrupted_generations(output)
        with managed_extractor(extractor_factory, model_path=model) as extractor:
            # Construction happens before staging so a model/runtime failure
            # cannot leave a partial generation behind.
            staging = Path(tempfile.mkdtemp(
                prefix=f".{output.name}.staging-", dir=output.parent
            ))
            retained: list[ExtractionResult] = []
            excluded: list[dict[str, object]] = []
            try:
                for source in sources:
                    try:
                        result = extract_source_video(
                            source, extractor, capture_factory=capture_factory
                        )
                        validate_retained_recording(result)
                    except (RecordingExtractionError, ValueError) as exc:
                        excluded.append({
                            "recording_id": source.binding.recording_id,
                            "group_id": source.binding.group_id,
                            "source_sha256": source.source_sha256,
                            "label": source.binding.label,
                            "reason": str(exc),
                        })
                        continue
                    write_validated_recording_cache(
                        staging / f"{source.binding.recording_id}.npz", result
                    )
                    retained.append(result)

                validate_collection_gate(retained)
                retained_label_counts = {
                    label: sum(result.binding.label == label for result in retained)
                    for label in EXPECTED_LABEL_COUNTS
                }
                producer_components = {
                    logical_name: digest
                    for logical_name, _path, digest in provenance.producer_files
                }
                manifest: dict[str, object] = {
                    "schema_version": COLLECTION_SCHEMA,
                    "dataset": "PalsyNet",
                    "feature_schema": DYNAMIC_FEATURE_SCHEMA,
                    "feature_shape": list(DYNAMIC_FEATURE_SHAPE),
                    "capture_mirrored": None,
                    "claim_unit": identity.claim_unit,
                    "identity_status": identity.identity_status,
                    "protocol": {
                        "windows_per_recording": 4,
                        "frames_per_window": 32,
                        "minimum_coverage": MIN_RECORDING_COVERAGE,
                        "minimum_retained": MIN_RETAINED,
                        "minimum_landmark_variation_fraction": MIN_VARIATION_FRACTION,
                    },
                    "provenance": {
                        "model_sha256": provenance.model_file[1],
                        "identity_manifest_sha256": identity.manifest_sha256,
                        "identity_fingerprints": dict(identity.fingerprints),
                        "source_collection_sha256": observed_source_fingerprint,
                        "corpus_inventory": dict(inventory),
                        "dependency_versions": _dependency_versions(),
                        "producer_sources": {
                            "components": producer_components,
                            "aggregate_sha256": provenance.producer_aggregate_sha256,
                        },
                    },
                    "counts": {
                        "discovered": EXPECTED_TOTAL,
                        "retained": len(retained),
                        "excluded": len(excluded),
                        "retained_affected": retained_label_counts["affected"],
                        "retained_unaffected": retained_label_counts["unaffected"],
                        "retained_groups": len({result.binding.group_id for result in retained}),
                    },
                    "records": [
                        _record_manifest(result)
                        for result in sorted(retained, key=lambda item: item.binding.recording_id)
                    ],
                    "excluded": sorted(excluded, key=lambda item: str(item["recording_id"])),
                }
                _write_json_exclusive(staging / "collection_manifest.json", manifest)
                validate_staged_file_set(staging, manifest)
                assert_provenance_unchanged(provenance)
                promote_generation(staging, output)
                return manifest
            finally:
                if staging.exists():
                    _remove_real_tree(staging)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--identity-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = run_builder(
        data_root=args.data_root,
        model_path=args.model_path,
        identity_manifest=args.identity_manifest,
        output_root=args.output_root,
    )
    print(json.dumps({"counts": manifest["counts"], "output_root": "deidentified_cache"},
                     sort_keys=True))


if __name__ == "__main__":
    main()
