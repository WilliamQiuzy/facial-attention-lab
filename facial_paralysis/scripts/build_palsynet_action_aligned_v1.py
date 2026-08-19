#!/usr/bin/env python3
"""Build a local identity-bound PalsyNet Action-Aligned clinical23 cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_palsynet_v2_windows import (  # noqa: E402
    IdentityBinding,
    SourceVideo,
    managed_extractor,
)
from scripts.freeze_palsynet_person_split_registry import (  # noqa: E402
    validate_person_split_registry,
)
from src.datasets.dynamic_landmark import (  # noqa: E402
    DYNAMIC_FEATURE_NAMES,
    DYNAMIC_FEATURE_SCHEMA,
)
from src.preprocessing.action_aligned_110d import (  # noqa: E402
    ACTION_SLOT_ORDER,
    WINDOW_FRAMES,
    action_window_source_indices,
    select_action_window_starts,
)
from src.preprocessing.action_bundle import MediaPipeFeatureExtractor  # noqa: E402


ACTION_CACHE_SCHEMA = "palsynet_action_aligned_clinical23_v1"
PROPOSAL_RATE_HZ = 6.0
MIN_CACHE_COVERAGE = 0.75
DEFAULT_EXTRACTION_MIN_COVERAGE = 0.90
_RECORDING_ID = re.compile(r"^rec_[0-9a-f]{64}$")
_GROUP_ID = re.compile(r"^grp_[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CACHE_FIELDS = {
    "features",
    "valid_mask",
    "timestamps",
    "timestamp_unit",
    "source_frame_indices",
    "source_frame_count",
    "fps",
    "feature_schema",
    "feature_names",
    "recording_id",
    "group_id",
    "label",
    "source_sha256",
    "identity_status",
    "claim_unit",
    "action_slots",
    "schema_version",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enumerate_reviewed_sources(
    data_root: str | Path,
    reviewed_identity_manifest: str | Path,
) -> tuple[SourceVideo, ...]:
    """Bind the 49 raw PalsyNet videos to reviewed groups by SHA-256 only."""
    root = Path(data_root)
    manifest_path = Path(reviewed_identity_manifest)
    if not root.is_dir() or not manifest_path.is_file():
        raise ValueError("PalsyNet data root and reviewed manifest must exist")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("reviewed identity manifest is invalid") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "palsynet_identity_reviewed_v1"
        or payload.get("dataset") != "PalsyNet"
        or payload.get("claim_unit") != "person_held_out"
        or payload.get("identity_review", {}).get("status") != "reviewed"
    ):
        raise ValueError("reviewed identity manifest contract drifted")
    rows = payload.get("recordings")
    if not isinstance(rows, list) or len(rows) != 49:
        raise ValueError("reviewed identity manifest must contain 49 rows")
    by_hash: dict[str, IdentityBinding] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("training_eligible") is not True:
            raise ValueError("every reviewed identity row must be training eligible")
        digest = str(row.get("source_sha256", ""))
        binding = IdentityBinding(
            source_sha256=digest,
            recording_id=str(row.get("recording_id", "")),
            group_id=str(row.get("group_id", "")),
            label=str(row.get("label", "")),
            identity_status=str(row.get("identity_status", "")),
            claim_unit=str(row.get("claim_unit", "")),
        )
        if (
            _SHA256.fullmatch(digest) is None
            or _RECORDING_ID.fullmatch(binding.recording_id) is None
            or _GROUP_ID.fullmatch(binding.group_id) is None
            or binding.label not in {"affected", "unaffected"}
            or binding.identity_status != "reviewed"
            or binding.claim_unit != "person_held_out"
            or digest in by_hash
        ):
            raise ValueError("reviewed identity row is invalid or duplicated")
        by_hash[digest] = binding

    sources: list[SourceVideo] = []
    seen: set[str] = set()
    for label in ("affected", "unaffected"):
        label_root = root / label
        if not label_root.is_dir():
            raise ValueError("PalsyNet label directory is missing")
        for path in sorted(label_root.iterdir(), key=lambda item: item.name):
            info = os.lstat(path)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise ValueError("PalsyNet label directories allow regular files only")
            if path.suffix.lower() != ".mp4":
                raise ValueError("PalsyNet label directories allow MP4 files only")
            digest = _sha256_file(path)
            binding = by_hash.get(digest)
            if binding is None or binding.label != label or digest in seen:
                raise ValueError("PalsyNet source bytes differ from reviewed identity rows")
            seen.add(digest)
            sources.append(
                SourceVideo(path=path, source_sha256=digest, binding=binding)
            )
    if len(sources) != 49 or seen != set(by_hash):
        raise ValueError("PalsyNet source collection differs from reviewed identity coverage")
    return tuple(sorted(sources, key=lambda item: item.binding.recording_id))


def select_development_sources(
    sources: tuple[SourceVideo, ...],
    split_registry: object,
) -> tuple[SourceVideo, ...]:
    """Return only development sources from an already authenticated registry."""
    if not isinstance(split_registry, dict) or (
        split_registry.get("schema_version")
        != "palsynet_person_split_registry_v1"
        or split_registry.get("dataset") != "PalsyNet"
        or split_registry.get("claim_unit") != "person_held_out"
        or split_registry.get("identity_status") != "reviewed"
        or split_registry.get("outer_fold_number") != 0
    ):
        raise ValueError("person split registry contract drifted")
    assignments = split_registry.get("assignments")
    counts = split_registry.get("counts")
    if not isinstance(assignments, list) or not isinstance(counts, dict):
        raise ValueError("person split registry assignments/counts are invalid")
    source_by_recording = {
        source.binding.recording_id: source for source in sources
    }
    if len(source_by_recording) != len(sources):
        raise ValueError("source recording IDs must be unique")
    assignment_by_recording: dict[str, dict[str, object]] = {}
    development_ids: list[str] = []
    for row in assignments:
        if not isinstance(row, dict) or set(row) != {
            "recording_id", "group_id", "semantic_group_key_sha256",
            "partition", "outer_fold", "inner_fold",
        }:
            raise ValueError("split assignment schema drifted")
        recording_id = str(row.get("recording_id", ""))
        if recording_id in assignment_by_recording or recording_id not in source_by_recording:
            raise ValueError("split assignments differ from reviewed sources")
        source = source_by_recording[recording_id]
        if row.get("group_id") != source.binding.group_id:
            raise ValueError("split group differs from reviewed source binding")
        partition = row.get("partition")
        if partition == "development":
            if row.get("outer_fold") == 0 or row.get("inner_fold") not in range(4):
                raise ValueError("development split coordinates are invalid")
            development_ids.append(recording_id)
        elif partition == "protected":
            if row.get("outer_fold") != 0 or row.get("inner_fold") is not None:
                raise ValueError("protected split coordinates are invalid")
        else:
            raise ValueError("split partition must be development or protected")
        assignment_by_recording[recording_id] = row
    if set(assignment_by_recording) != set(source_by_recording):
        raise ValueError("split registry does not cover every reviewed source")
    expected_development = counts.get("development_recordings")
    expected_protected = counts.get("protected_recordings")
    if (
        isinstance(expected_development, bool)
        or not isinstance(expected_development, int)
        or isinstance(expected_protected, bool)
        or not isinstance(expected_protected, int)
        or len(development_ids) != expected_development
        or len(sources) - len(development_ids) != expected_protected
        or expected_development + expected_protected != len(sources)
    ):
        raise ValueError("split recording counts are incoherent")
    return tuple(source_by_recording[recording_id] for recording_id in development_ids)


def _read_json_artifact(path: Path, name: str) -> tuple[dict[str, object], str]:
    try:
        raw = path.read_bytes()
        def unique(pairs):
            output = {}
            for key, value in pairs:
                if key in output:
                    raise ValueError(f"{name} contains duplicate JSON keys")
                output[key] = value
            return output
        payload = json.loads(raw, object_pairs_hook=unique)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} root must be an object")
    return payload, hashlib.sha256(raw).hexdigest()


def authenticated_development_sources(
    sources: tuple[SourceVideo, ...],
    reviewed_identity_manifest: str | Path,
    review_ledger: str | Path,
    split_registry: str | Path,
) -> tuple[SourceVideo, ...]:
    """Authenticate identity evidence and return the development partition only."""
    reviewed, reviewed_sha = _read_json_artifact(
        Path(reviewed_identity_manifest), "reviewed identity manifest"
    )
    ledger, ledger_sha = _read_json_artifact(Path(review_ledger), "review ledger")
    registry, _registry_sha = _read_json_artifact(
        Path(split_registry), "person split registry"
    )
    if (
        registry.get("reviewed_manifest_sha256") != reviewed_sha
        or registry.get("review_ledger_sha256") != ledger_sha
    ):
        raise ValueError("person split registry does not bind the supplied identity files")
    validate_person_split_registry(registry, reviewed, ledger)
    return select_development_sources(sources, registry)


@dataclass(frozen=True)
class ActionAlignedExtraction:
    features: np.ndarray
    valid_mask: np.ndarray
    timestamps: np.ndarray
    source_frame_indices: np.ndarray
    source_frame_count: int
    fps: float
    binding: IdentityBinding
    action_slots: tuple[str, ...]
    schema_version: str = ACTION_CACHE_SCHEMA

    @property
    def feature_names(self) -> tuple[str, ...]:
        return DYNAMIC_FEATURE_NAMES

    @property
    def coverage(self) -> float:
        return float(np.asarray(self.valid_mask).mean())


def _metadata(capture, prop: int, name: str) -> float:
    value = float(capture.get(prop))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"invalid_{name}")
    return value


def _extract_vector(extractor, frame: np.ndarray) -> np.ndarray | None:
    vector, _nuisance = extractor.extract_frame_with_nuisance(frame)
    if vector is None:
        return None
    result = np.asarray(vector)
    if result.shape != (95,) or result.dtype != np.dtype(np.float32):
        raise ValueError("MediaPipe action feature vector must be float32 with 95 columns")
    if not np.isfinite(result).all():
        raise ValueError("MediaPipe action feature vector must be finite")
    return result


def _seek_read(capture, frame_index: int) -> np.ndarray:
    if not capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index)):
        raise ValueError("seek_failed")
    ok, frame = capture.read()
    if not ok or frame is None or frame.ndim != 3 or frame.size == 0:
        raise ValueError("decode_failed")
    return frame


def _proposal_frames(capture, frame_indices: np.ndarray):
    """Yield exact proposal frames sequentially when the decoder supports grab()."""
    if not hasattr(capture, "grab"):
        for frame_index in frame_indices.tolist():
            yield frame_index, _seek_read(capture, frame_index)
        return
    if not capture.set(cv2.CAP_PROP_POS_FRAMES, 0):
        raise ValueError("seek_failed")
    position = 0
    for frame_index in frame_indices.tolist():
        while position < frame_index:
            if not capture.grab():
                raise ValueError("decode_failed")
            position += 1
        ok, frame = capture.read()
        if not ok or frame is None or frame.ndim != 3 or frame.size == 0:
            raise ValueError("decode_failed")
        position += 1
        yield frame_index, frame


def extract_action_aligned_source(
    source: SourceVideo,
    extractor,
    *,
    capture_factory=cv2.VideoCapture,
    capture_configurator=None,
    minimum_window_coverage: float = DEFAULT_EXTRACTION_MIN_COVERAGE,
) -> ActionAlignedExtraction:
    """Extract one video using a label-free sparse proposal and seven windows."""
    capture = capture_factory(str(source.path))
    try:
        if not capture.isOpened():
            raise ValueError("open_failed")
        if capture_configurator is not None:
            capture_configurator(capture)
        fps = _metadata(capture, cv2.CAP_PROP_FPS, "fps")
        raw_count = _metadata(capture, cv2.CAP_PROP_FRAME_COUNT, "frame_count")
        if not raw_count.is_integer():
            raise ValueError("nonintegral_frame_count")
        frame_count = int(raw_count)
        if frame_count < WINDOW_FRAMES:
            raise ValueError("insufficient_frame_count")

        proposal_stride_frames = max(1, int(round(fps / PROPOSAL_RATE_HZ)))
        proposal_indices = np.arange(
            0, frame_count, proposal_stride_frames, dtype=np.int64
        )
        proposal = np.zeros((proposal_indices.size, 95), dtype=np.float32)
        proposal_valid = np.zeros(proposal_indices.size, dtype=bool)
        if (
            not np.isfinite(minimum_window_coverage)
            or not MIN_CACHE_COVERAGE <= minimum_window_coverage <= 1.0
        ):
            raise ValueError("minimum window coverage must lie within [0.75, 1]")
        for row, (_source_index, frame) in enumerate(
            _proposal_frames(capture, proposal_indices)
        ):
            vector = _extract_vector(extractor, frame)
            if vector is not None:
                proposal[row] = vector
                proposal_valid[row] = True
        starts = select_action_window_starts(
            proposal,
            proposal_valid,
            proposal_indices,
            source_frame_count=frame_count,
            source_fps=fps,
        )
        source_indices = action_window_source_indices(
            starts, source_fps=fps, source_frame_count=frame_count
        )
        features = np.zeros((len(ACTION_SLOT_ORDER), WINDOW_FRAMES, 95), dtype=np.float32)
        valid = np.zeros((len(ACTION_SLOT_ORDER), WINDOW_FRAMES), dtype=bool)
        for slot, slot_indices in enumerate(source_indices):
            previous = None
            for offset, source_index in enumerate(slot_indices.tolist()):
                if previous is None or source_index != previous + 1:
                    frame = _seek_read(capture, source_index)
                else:
                    ok, frame = capture.read()
                    if not ok or frame is None or frame.ndim != 3 or frame.size == 0:
                        raise ValueError("decode_failed")
                vector = _extract_vector(extractor, frame)
                if vector is not None:
                    features[slot, offset] = vector
                    valid[slot, offset] = True
                previous = source_index
        if float(valid.mean()) < minimum_window_coverage:
            raise ValueError("action_window_coverage_below_required_minimum")
        return ActionAlignedExtraction(
            features=features,
            valid_mask=valid,
            timestamps=source_indices.astype(np.float64) / fps,
            source_frame_indices=source_indices,
            source_frame_count=frame_count,
            fps=fps,
            binding=source.binding,
            action_slots=ACTION_SLOT_ORDER,
        )
    finally:
        capture.release()


def _validate(result: ActionAlignedExtraction) -> None:
    if result.schema_version != ACTION_CACHE_SCHEMA:
        raise ValueError("action cache schema drifted")
    if tuple(result.action_slots) != ACTION_SLOT_ORDER:
        raise ValueError("action slot order drifted")
    features = np.asarray(result.features)
    valid = np.asarray(result.valid_mask)
    timestamps = np.asarray(result.timestamps)
    indices = np.asarray(result.source_frame_indices)
    expected = (len(ACTION_SLOT_ORDER), WINDOW_FRAMES)
    if features.shape != expected + (95,) or features.dtype != np.dtype(np.float32):
        raise ValueError("action features must be float32 with shape (7, 32, 95)")
    if valid.shape != expected or valid.dtype != np.dtype(bool):
        raise ValueError("action mask must be bool with shape (7, 32)")
    if timestamps.shape != expected or not np.isfinite(timestamps).all():
        raise ValueError("action timestamps must be finite with shape (7, 32)")
    if indices.shape != expected or indices.dtype.kind not in {"i", "u"}:
        raise ValueError("action indices must be integer with shape (7, 32)")
    expected_indices = action_window_source_indices(
        indices[:, 0].tolist(), source_fps=float(result.fps),
        source_frame_count=int(result.source_frame_count),
    )
    if not np.array_equal(indices, expected_indices):
        raise ValueError("action windows must follow the reference 30 Hz time grid")
    if np.any(indices < 0) or np.any(indices >= int(result.source_frame_count)):
        raise ValueError("action source indices escaped the video")
    if not np.array_equal(timestamps, indices.astype(np.float64) / float(result.fps)):
        raise ValueError("timestamps must derive exactly from source frame indices and FPS")
    if not np.isfinite(features[valid]).all() or np.any(features[~valid] != 0):
        raise ValueError("action features violate finite/zero-padding contract")
    if float(valid.mean()) < MIN_CACHE_COVERAGE:
        raise ValueError("action cache coverage is below 0.75")
    binding = result.binding
    if (
        _RECORDING_ID.fullmatch(binding.recording_id) is None
        or _GROUP_ID.fullmatch(binding.group_id) is None
        or _SHA256.fullmatch(binding.source_sha256) is None
        or binding.label not in {"affected", "unaffected"}
        or not binding.identity_status
        or not binding.claim_unit
    ):
        raise ValueError("action cache identity binding is invalid")


def _payload(result: ActionAlignedExtraction) -> dict[str, np.ndarray]:
    _validate(result)
    return {
        "features": np.asarray(result.features),
        "valid_mask": np.asarray(result.valid_mask),
        "timestamps": np.asarray(result.timestamps),
        "timestamp_unit": np.asarray("seconds"),
        "source_frame_indices": np.asarray(result.source_frame_indices),
        "source_frame_count": np.asarray(result.source_frame_count, dtype=np.int64),
        "fps": np.asarray(result.fps, dtype=np.float64),
        "feature_schema": np.asarray(DYNAMIC_FEATURE_SCHEMA),
        "feature_names": np.asarray(DYNAMIC_FEATURE_NAMES),
        "recording_id": np.asarray(result.binding.recording_id),
        "group_id": np.asarray(result.binding.group_id),
        "label": np.asarray(1 if result.binding.label == "affected" else 0, dtype=np.int64),
        "source_sha256": np.asarray(result.binding.source_sha256),
        "identity_status": np.asarray(result.binding.identity_status),
        "claim_unit": np.asarray(result.binding.claim_unit),
        "action_slots": np.asarray(ACTION_SLOT_ORDER),
        "schema_version": np.asarray(ACTION_CACHE_SCHEMA),
    }


def write_action_aligned_cache(path: str | Path, result: ActionAlignedExtraction) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite action cache {output}")
    payload = _payload(result)
    fd, temporary_name = tempfile.mkstemp(
        prefix=".action-aligned-", suffix=".npz", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            np.savez(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
        os.chmod(output, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def load_action_aligned_cache(path: str | Path) -> ActionAlignedExtraction:
    try:
        with np.load(Path(path), allow_pickle=False) as saved:
            if set(saved.files) != _CACHE_FIELDS:
                raise ValueError("action cache fields differ from the closed schema")
            values = {name: np.asarray(saved[name]) for name in _CACHE_FIELDS}
    except (OSError, ValueError) as exc:
        raise ValueError("action cache is invalid") from exc
    scalar = lambda name: np.asarray(values[name]).item()
    if (
        scalar("schema_version") != ACTION_CACHE_SCHEMA
        or scalar("feature_schema") != DYNAMIC_FEATURE_SCHEMA
        or scalar("timestamp_unit") != "seconds"
        or tuple(values["feature_names"].tolist()) != DYNAMIC_FEATURE_NAMES
        or tuple(values["action_slots"].tolist()) != ACTION_SLOT_ORDER
    ):
        raise ValueError("action cache schema metadata drifted")
    label_value = int(scalar("label"))
    if label_value not in {0, 1}:
        raise ValueError("action cache label must be binary")
    result = ActionAlignedExtraction(
        features=values["features"],
        valid_mask=values["valid_mask"],
        timestamps=values["timestamps"],
        source_frame_indices=values["source_frame_indices"],
        source_frame_count=int(scalar("source_frame_count")),
        fps=float(scalar("fps")),
        binding=IdentityBinding(
            source_sha256=str(scalar("source_sha256")),
            recording_id=str(scalar("recording_id")),
            group_id=str(scalar("group_id")),
            label="affected" if label_value == 1 else "unaffected",
            identity_status=str(scalar("identity_status")),
            claim_unit=str(scalar("claim_unit")),
        ),
        action_slots=tuple(str(value) for value in values["action_slots"].tolist()),
    )
    _validate(result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--palsynet-data-root", required=True, type=Path)
    parser.add_argument("--reviewed-identity-manifest", required=True, type=Path)
    parser.add_argument("--review-ledger", required=True, type=Path)
    parser.add_argument("--split-registry", required=True, type=Path)
    parser.add_argument("--mediapipe-model", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    sources = enumerate_reviewed_sources(
        args.palsynet_data_root, args.reviewed_identity_manifest
    )
    sources = authenticated_development_sources(
        sources,
        args.reviewed_identity_manifest,
        args.review_ledger,
        args.split_registry,
    )
    args.output_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    with managed_extractor(
        MediaPipeFeatureExtractor,
        model_path=args.mediapipe_model,
    ) as extractor:
        for source in sources:
            result = extract_action_aligned_source(source, extractor)
            write_action_aligned_cache(
                args.output_root / f"{source.binding.recording_id}.npz", result
            )
            print(source.binding.recording_id, flush=True)


if __name__ == "__main__":
    main()
