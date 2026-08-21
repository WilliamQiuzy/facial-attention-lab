"""One-shot external MEEI evaluation of the frozen PalsyNet 110D artifact.

This module has no fitting or tuning path.  It authenticates the final model,
participant manifest, collection manifest, every cache byte artifact, and the
execution implementation before any NPZ array is decoded.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from scripts.run_110d_generalization_v1 import _fast_metrics
from scripts.run_mirror_invariant_110d import (
    FIXED_C,
    FIXED_MAX_ITER,
    FIXED_RANDOM_STATE,
    FIXED_SOLVER,
    FIXED_THRESHOLD,
    mirror_dynamic_features,
)
from src.datasets import dynamic_landmark as dynamic_cache
from src.evaluation.outer_release_110d_v1 import predict_from_frozen_artifact
from src.preprocessing.generalization_110d import (
    LANDMARK_MI_110D,
    candidate_feature_names,
    candidate_feature_vector,
)


SCHEMA_VERSION = "meei_external_110d_v1"
AUTHORIZATION_SCHEMA_VERSION = "meei_external_110d_authorization_v1"
AUTHORIZATION_BASIS = "researcher_instruction_frozen_110d_true_external_validation"
DEFAULT_REPORT_RELATIVE = "outputs/meei_external_v1/report.json"
BOOTSTRAP_REPEATS = 5000
BOOTSTRAP_SEED = 20260805
LOCKED_DIMENSION = 110
METRICS = (
    "auroc",
    "average_precision",
    "brier",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REC_ID = re.compile(r"^rec_[0-9a-f]{64}$")
_GROUP_ID = re.compile(r"^grp_[0-9a-f]{64}$")


@dataclass
class ExternalAudit:
    authorization_attempts: int = 0
    authorization_passes: int = 0
    cache_artifacts_hashed: int = 0
    cache_records_loaded: int = 0
    feature_extractions: int = 0
    mirror_transforms: int = 0
    artifact_predictions: int = 0
    participant_aggregations: int = 0
    scaler_fits: int = 0
    model_fits: int = 0
    calibration_fits: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name))
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class CacheArtifactInventory:
    blobs: Mapping[str, bytes]
    sha256_by_recording: Mapping[str, str]
    size_by_recording: Mapping[str, int]
    collection_sha256: str


@dataclass(frozen=True)
class AuthorizedExternalState:
    authorization_sha256: str
    final_artifact_sha256: str
    participant_manifest_sha256: str
    cache_manifest_sha256: str
    cache_artifact_collection_sha256: str
    implementation_sha256: str
    expected_participants: int
    expected_affected: int
    expected_unaffected: int
    expected_eligible_videos: int


@dataclass(frozen=True)
class ExternalMetadata:
    rows: tuple[Mapping[str, object], ...]
    participants_total: int
    affected: int
    unaffected: int
    eligible_videos: int


def canonical_json_sha256(payload: object) -> str:
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _exact_mapping(value: object, fields: set[str], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} fields differ from the frozen schema")
    return value


def implementation_fingerprints() -> tuple[dict[str, str], str]:
    project_root = Path(__file__).resolve().parents[2]
    paths = {
        "external_core": Path(__file__).resolve(),
        "external_runner": project_root / "scripts/run_meei_external_v1.py",
        "generalization_features": project_root / "src/preprocessing/generalization_110d.py",
        "trajectory_features": project_root / "src/preprocessing/trajectory_features.py",
        "mirror_runner": project_root / "scripts/run_mirror_invariant_110d.py",
        "dynamic_landmark_model": project_root / "src/models/dynamic_landmark.py",
        "dynamic_landmark_loader": project_root / "src/datasets/dynamic_landmark.py",
        "frozen_artifact_inference": project_root / "src/evaluation/outer_release_110d_v1.py",
    }
    components: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"implementation component {name!r} is unavailable")
        components[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return components, canonical_json_sha256(components)


def _protocol() -> dict[str, object]:
    return {
        "candidate_selection": False,
        "model_refit": False,
        "scaler_refit": False,
        "calibration": False,
        "threshold_selection": False,
        "representation": LANDMARK_MI_110D,
        "dimension": LOCKED_DIMENSION,
        "mirror_inference": "mean_original_and_horizontal_mirror_probability",
        "video_aggregation": "mean_authenticated_video_probability_once_per_participant",
        "metric_unit": "participant",
        "threshold": FIXED_THRESHOLD,
        "primary_metric": "auroc",
        "secondary_metrics": list(METRICS[1:]),
        "bootstrap": {
            "method": "affected_unaffected_stratified_participant_percentile",
            "repeats": BOOTSTRAP_REPEATS,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
        },
    }


def build_expected_authorization(
    *,
    final_artifact_sha256: str,
    participant_manifest_sha256: str,
    cache_manifest_sha256: str,
    cache_artifact_collection_sha256: str,
    implementation_sha256: str,
    expected_participants: int,
    expected_affected: int,
    expected_unaffected: int,
    expected_eligible_videos: int,
) -> dict[str, object]:
    for name, value in (
        ("final artifact", final_artifact_sha256),
        ("participant manifest", participant_manifest_sha256),
        ("cache manifest", cache_manifest_sha256),
        ("cache artifact collection", cache_artifact_collection_sha256),
        ("implementation", implementation_sha256),
    ):
        _sha(value, name)
    participants = _positive_integer(expected_participants, "expected participants")
    affected = _positive_integer(expected_affected, "expected affected")
    unaffected = _positive_integer(expected_unaffected, "expected unaffected")
    videos = _positive_integer(expected_eligible_videos, "expected eligible videos")
    if affected + unaffected != participants or videos < participants:
        raise ValueError("authorization population counts are inconsistent")
    return {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "authorization_basis": AUTHORIZATION_BASIS,
        "dataset": "MEEI_Facial_Palsy_Standard_Set",
        "target": "normal_vs_facial_palsy",
        "final_artifact_sha256": final_artifact_sha256,
        "participant_manifest_sha256": participant_manifest_sha256,
        "cache_manifest_sha256": cache_manifest_sha256,
        "cache_artifact_collection_sha256": cache_artifact_collection_sha256,
        "implementation_sha256": implementation_sha256,
        "expected_participants": participants,
        "expected_affected": affected,
        "expected_unaffected": unaffected,
        "expected_eligible_videos": videos,
        "protocol": _protocol(),
        "output_relative_path": DEFAULT_REPORT_RELATIVE,
        "authorized_once": True,
    }


def validate_external_authorization(
    authorization: Mapping[str, object],
    *,
    authorization_sha256: str,
    pinned_authorization_sha256: str | None,
    audit: ExternalAudit,
    **expected_inputs: object,
) -> AuthorizedExternalState:
    audit.authorization_attempts += 1
    authorization_sha = _sha(authorization_sha256, "authorization")
    if pinned_authorization_sha256 is None:
        raise ValueError("MEEI one-shot authorization pin is not activated")
    if _sha(pinned_authorization_sha256, "pinned authorization") != authorization_sha:
        raise ValueError("MEEI authorization differs from its out-of-band pin")
    if canonical_json_sha256(authorization) != authorization_sha:
        raise ValueError("MEEI authorization is not exact canonical JSON")
    expected = build_expected_authorization(**expected_inputs)
    if dict(authorization) != expected:
        raise ValueError("MEEI authorization fields differ from the frozen contract")
    audit.authorization_passes += 1
    return AuthorizedExternalState(
        authorization_sha256=authorization_sha,
        final_artifact_sha256=str(expected_inputs["final_artifact_sha256"]),
        participant_manifest_sha256=str(expected_inputs["participant_manifest_sha256"]),
        cache_manifest_sha256=str(expected_inputs["cache_manifest_sha256"]),
        cache_artifact_collection_sha256=str(
            expected_inputs["cache_artifact_collection_sha256"]
        ),
        implementation_sha256=str(expected_inputs["implementation_sha256"]),
        expected_participants=int(expected_inputs["expected_participants"]),
        expected_affected=int(expected_inputs["expected_affected"]),
        expected_unaffected=int(expected_inputs["expected_unaffected"]),
        expected_eligible_videos=int(expected_inputs["expected_eligible_videos"]),
    )


def _read_regular_file_same_descriptor(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot securely open cache artifact {path.name}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("cache artifact must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("cache artifact changed during authenticated read")
        data = b"".join(chunks)
        if len(data) != before.st_size:
            raise ValueError("cache artifact read is incomplete")
        return data
    finally:
        os.close(descriptor)


def cache_artifact_inventory(
    cache_root: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    audit: ExternalAudit,
) -> CacheArtifactInventory:
    root = Path(cache_root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("feature cache root must be a real directory")
    recording_ids = [row.get("recording_id") for row in rows]
    if (
        not recording_ids
        or any(not isinstance(value, str) or _REC_ID.fullmatch(value) is None
               for value in recording_ids)
        or len(set(recording_ids)) != len(recording_ids)
    ):
        raise ValueError("cache manifest recording IDs are invalid or duplicated")
    expected = {root / f"{recording_id}.npz" for recording_id in recording_ids}
    observed = set(root.glob("*.npz"))
    if observed != expected:
        raise ValueError("cache NPZ names differ from the authenticated manifest")
    blobs: dict[str, bytes] = {}
    digests: dict[str, str] = {}
    sizes: dict[str, int] = {}
    collection = hashlib.sha256()
    for recording_id in sorted(recording_ids):
        blob = _read_regular_file_same_descriptor(root / f"{recording_id}.npz")
        digest = hashlib.sha256(blob).hexdigest()
        blobs[recording_id] = blob
        digests[recording_id] = digest
        sizes[recording_id] = len(blob)
        collection.update(f"{recording_id}:{digest}:{len(blob)}\n".encode("ascii"))
        audit.cache_artifacts_hashed += 1
    return CacheArtifactInventory(
        blobs=blobs,
        sha256_by_recording=digests,
        size_by_recording=sizes,
        collection_sha256=collection.hexdigest(),
    )


def validate_external_metadata(
    participant_manifest: Mapping[str, object],
    cache_manifest: Mapping[str, object],
    *,
    participant_manifest_sha256: str,
) -> ExternalMetadata:
    participant_manifest_sha = _sha(
        participant_manifest_sha256, "participant manifest"
    )
    if (
        participant_manifest.get("schema_version") != "meei_participant_manifest_v1"
        or participant_manifest.get("dataset") != "MEEI_Facial_Palsy_Standard_Set"
        or participant_manifest.get("claim_unit") != "participant"
    ):
        raise ValueError("MEEI participant manifest identity is invalid")
    endpoint = participant_manifest.get("endpoint")
    if not isinstance(endpoint, Mapping) or (
        endpoint.get("target") != "normal_vs_facial_palsy"
        or endpoint.get("candidate_selection_eligible") is not False
        or endpoint.get("photo_decoding_allowed") is not False
    ):
        raise ValueError("MEEI participant endpoint contract is invalid")
    participants = participant_manifest.get("participants")
    media = participant_manifest.get("media")
    counts = participant_manifest.get("counts")
    if not isinstance(participants, list) or not isinstance(media, list) or not isinstance(counts, Mapping):
        raise ValueError("MEEI participant inventory is malformed")
    labels_by_group: dict[str, int] = {}
    for row in participants:
        if not isinstance(row, Mapping):
            raise ValueError("MEEI participant row must be an object")
        group_id = row.get("participant_id")
        label_name = row.get("binary_label")
        if (
            not isinstance(group_id, str)
            or _GROUP_ID.fullmatch(group_id) is None
            or label_name not in {"affected", "unaffected"}
            or group_id in labels_by_group
        ):
            raise ValueError("MEEI participant identity or label is invalid")
        media_counts = row.get("media_counts")
        if not isinstance(media_counts, Mapping) or media_counts.get("video") != 1:
            raise ValueError("every MEEI participant must have exactly one video")
        labels_by_group[group_id] = 1 if label_name == "affected" else 0
    dynamic_media: dict[str, Mapping[str, object]] = {}
    for row in media:
        if not isinstance(row, Mapping) or row.get("dynamic_binary_eligible") is not True:
            continue
        recording_id = row.get("recording_id")
        group_id = row.get("participant_id")
        source_sha = row.get("source_sha256")
        if (
            row.get("media_type") != "video"
            or not isinstance(recording_id, str)
            or _REC_ID.fullmatch(recording_id) is None
            or not isinstance(group_id, str)
            or group_id not in labels_by_group
            or not isinstance(source_sha, str)
            or _SHA256.fullmatch(source_sha) is None
            or recording_id in dynamic_media
        ):
            raise ValueError("eligible MEEI video join is invalid")
        dynamic_media[recording_id] = row
    if (
        len(participants) != len(labels_by_group)
        or len(dynamic_media) != len(participants)
        or {str(row["participant_id"]) for row in dynamic_media.values()}
        != set(labels_by_group)
        or counts.get("participants") != len(participants)
        or counts.get("dynamic_binary_eligible_videos") != len(dynamic_media)
    ):
        raise ValueError("MEEI participant/video coverage is incomplete")

    if (
        cache_manifest.get("schema_version") != "meei_clinical23_v2_windows_v1"
        or cache_manifest.get("dataset") != "MEEI_Facial_Palsy_Standard_Set"
        or cache_manifest.get("claim_unit") != "participant"
        or cache_manifest.get("feature_schema") != dynamic_cache.DYNAMIC_FEATURE_SCHEMA
        or cache_manifest.get("feature_shape") != list(dynamic_cache.DYNAMIC_FEATURE_SHAPE)
        or cache_manifest.get("capture_mirrored") is not None
        or cache_manifest.get("excluded") != []
    ):
        raise ValueError("MEEI dynamic cache manifest contract is invalid")
    cache_counts = cache_manifest.get("counts")
    cache_provenance = cache_manifest.get("provenance")
    rows = cache_manifest.get("records")
    if not isinstance(cache_counts, Mapping) or not isinstance(cache_provenance, Mapping) or not isinstance(rows, list):
        raise ValueError("MEEI cache counts/provenance/records are malformed")
    if (
        cache_provenance.get("participant_manifest_sha256") != participant_manifest_sha
        or cache_counts.get("participants") != len(participants)
        or cache_counts.get("videos") != len(dynamic_media)
        or cache_counts.get("retained") != len(dynamic_media)
        or cache_counts.get("excluded_label_blind_qc") != 0
        or len(rows) != len(dynamic_media)
    ):
        raise ValueError("MEEI cache provenance or coverage differs from participants")
    seen: set[str] = set()
    normalized: list[Mapping[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("MEEI cache record must be an object")
        recording_id = row.get("recording_id")
        group_id = row.get("group_id")
        source_sha = row.get("source_sha256")
        label_name = row.get("label")
        media_row = dynamic_media.get(str(recording_id))
        if (
            media_row is None
            or recording_id in seen
            or group_id != media_row.get("participant_id")
            or source_sha != media_row.get("source_sha256")
            or label_name != ("affected" if labels_by_group[str(group_id)] else "unaffected")
        ):
            raise ValueError("MEEI cache record differs from participant media join")
        seen.add(str(recording_id))
        normalized.append(row)
    if seen != set(dynamic_media):
        raise ValueError("MEEI cache omits or adds eligible videos")
    affected = sum(labels_by_group.values())
    unaffected = len(labels_by_group) - affected
    if affected <= 0 or unaffected <= 0:
        raise ValueError("MEEI external endpoint requires both classes")
    return ExternalMetadata(
        rows=tuple(normalized),
        participants_total=len(labels_by_group),
        affected=affected,
        unaffected=unaffected,
        eligible_videos=len(dynamic_media),
    )


def validate_frozen_artifact_for_external(artifact: Mapping[str, object]) -> None:
    expected_top = {
        "schema_version", "claim_scope", "dataset", "target", "representation",
        "scaler", "model", "threshold", "training", "audit", "provenance",
    }
    if not isinstance(artifact, Mapping) or set(artifact) != expected_top:
        raise ValueError("final artifact fields differ from the closed schema")
    representation = artifact.get("representation")
    scaler = artifact.get("scaler")
    model = artifact.get("model")
    training = artifact.get("training")
    if (
        artifact.get("schema_version") != "110d_generalization_final_artifact_v1"
        or artifact.get("claim_scope") != "palsynet_all_eligible_refit_for_frozen_external_inference"
        or artifact.get("dataset") != "PalsyNet"
        or artifact.get("target") != "binary_affected_vs_unaffected"
        or artifact.get("threshold") != FIXED_THRESHOLD
        or not isinstance(representation, Mapping)
        or representation.get("name") != LANDMARK_MI_110D
        or representation.get("dimension") != LOCKED_DIMENSION
        or representation.get("feature_names") != list(candidate_feature_names(LANDMARK_MI_110D))
        or representation.get("mirror_inference") != "mean_original_and_horizontal_mirror_probability"
        or not isinstance(scaler, Mapping)
        or set(scaler) != {"type", "mean", "scale"}
        or scaler.get("type") != "standard_scaler"
        or not isinstance(model, Mapping)
        or model.get("type") != "l2_logistic_regression"
        or model.get("classes") != [0, 1]
        or model.get("c") != FIXED_C
        or model.get("solver") != FIXED_SOLVER
        or model.get("max_iter") != FIXED_MAX_ITER
        or model.get("random_state") != FIXED_RANDOM_STATE
        or not isinstance(training, Mapping)
        or training.get("augmentation_rows_per_recording") != 2
    ):
        raise ValueError("final artifact protocol is not the locked PalsyNet 110D model")
    mean = np.asarray(scaler.get("mean"), dtype=np.float64)
    scale = np.asarray(scaler.get("scale"), dtype=np.float64)
    coefficient = np.asarray(model.get("coefficient"), dtype=np.float64)
    intercept = model.get("intercept")
    if (
        any(value.shape != (LOCKED_DIMENSION,) for value in (mean, scale, coefficient))
        or not all(np.isfinite(value).all() for value in (mean, scale, coefficient))
        or np.any(scale <= 0.0)
        or isinstance(intercept, bool)
        or not isinstance(intercept, (int, float))
        or not np.isfinite(intercept)
    ):
        raise ValueError("final artifact numeric parameters are invalid")
    encoded = json.dumps(artifact, sort_keys=True, allow_nan=False)
    if any(token in encoded for token in ("rec_", "grp_", "/Users/", "\\")):
        raise ValueError("final artifact contains identifiers or paths")


def _record_from_authenticated_bytes(blob: bytes):
    try:
        with np.load(io.BytesIO(blob), allow_pickle=False) as saved:
            missing = dynamic_cache._REQUIRED_CACHE_FIELDS.difference(saved.files)
            unexpected = set(saved.files).difference(dynamic_cache._REQUIRED_CACHE_FIELDS)
            if missing or unexpected:
                raise ValueError("dynamic cache fields differ from the frozen schema")
            fields = {
                name: np.asarray(saved[name])
                for name in dynamic_cache._REQUIRED_CACHE_FIELDS
            }
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError("authenticated cache bytes are not a valid dynamic NPZ") from exc
    source_frame_count = dynamic_cache._source_frame_count(fields["source_frame_count"])
    features, valid_mask, timestamps, source_indices = dynamic_cache._validate_arrays(
        fields, source_frame_count
    )
    timestamp_unit = dynamic_cache._scalar_text(fields["timestamp_unit"], "timestamp_unit")
    schema = dynamic_cache._scalar_text(fields["feature_schema"], "feature_schema")
    feature_names = dynamic_cache._ordered_feature_names(fields["feature_names"])
    recording_id = dynamic_cache._scalar_text(fields["recording_id"], "recording_id")
    group_id = dynamic_cache._scalar_text(fields["group_id"], "group_id")
    label = dynamic_cache._binary_label(fields["label"])
    source_sha = dynamic_cache._scalar_text(fields["source_sha256"], "source_sha256")
    if (
        timestamp_unit != "seconds"
        or schema != dynamic_cache.DYNAMIC_FEATURE_SCHEMA
        or _REC_ID.fullmatch(recording_id) is None
        or _GROUP_ID.fullmatch(group_id) is None
        or _SHA256.fullmatch(source_sha) is None
    ):
        raise ValueError("authenticated dynamic cache metadata is invalid")
    return {
        "features": features,
        "valid_mask": valid_mask,
        "timestamps": timestamps,
        "source_frame_indices": source_indices,
        "source_frame_count": source_frame_count,
        "feature_names": feature_names,
        "recording_id": recording_id,
        "group_id": group_id,
        "label": label,
        "source_sha256": source_sha,
    }


def score_authenticated_cache(
    inventory: CacheArtifactInventory,
    rows: Sequence[Mapping[str, object]],
    artifact: Mapping[str, object],
    *,
    state: AuthorizedExternalState,
    audit: ExternalAudit,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(state, AuthorizedExternalState) or audit.authorization_passes != 1:
        raise ValueError("MEEI authorization must pass before cache decoding")
    if inventory.collection_sha256 != state.cache_artifact_collection_sha256:
        raise ValueError("cache artifact bytes differ from the authorized collection")
    validate_frozen_artifact_for_external(artifact)
    probabilities_by_group: dict[str, list[float]] = defaultdict(list)
    labels_by_group: dict[str, int] = {}
    for row in rows:
        recording_id = str(row["recording_id"])
        record = _record_from_authenticated_bytes(inventory.blobs[recording_id])
        expected_label = 1 if row["label"] == "affected" else 0
        if (
            record["recording_id"] != recording_id
            or record["group_id"] != row["group_id"]
            or record["source_sha256"] != row["source_sha256"]
            or record["label"] != expected_label
        ):
            raise ValueError("authenticated NPZ provenance differs from cache manifest")
        audit.cache_records_loaded += 1
        raw = record["features"]
        mirrored_raw = mirror_dynamic_features(raw)
        remirrored_raw = mirror_dynamic_features(mirrored_raw)
        audit.mirror_transforms += 2
        if not np.array_equal(remirrored_raw, raw):
            raise ValueError("horizontal mirror is not an exact involution")
        temporal = (
            record["valid_mask"],
            record["timestamps"],
            record["source_frame_indices"],
        )
        original = candidate_feature_vector(LANDMARK_MI_110D, raw, *temporal)
        mirrored = candidate_feature_vector(
            LANDMARK_MI_110D, mirrored_raw, *temporal
        )
        remirrored = candidate_feature_vector(
            LANDMARK_MI_110D, remirrored_raw, *temporal
        )
        audit.feature_extractions += 1
        if not np.array_equal(original, remirrored):
            raise ValueError("remirrored 110D features differ from original")
        probability = predict_from_frozen_artifact(artifact, original, mirrored)
        audit.artifact_predictions += 1
        group_id = str(row["group_id"])
        prior = labels_by_group.setdefault(group_id, expected_label)
        if prior != expected_label:
            raise ValueError("participant crosses external binary labels")
        probabilities_by_group[group_id].append(probability)
    if len(probabilities_by_group) != state.expected_participants:
        raise ValueError("external score coverage differs from authorization")
    labels: list[int] = []
    probabilities: list[float] = []
    for group_id in sorted(probabilities_by_group):
        labels.append(labels_by_group[group_id])
        probabilities.append(float(np.mean(probabilities_by_group[group_id])))
        audit.participant_aggregations += 1
    return np.asarray(labels, dtype=np.int64), np.asarray(probabilities, dtype=np.float64)


def _metric_report(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    repeats: int,
) -> dict[str, dict[str, float]]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != probabilities.shape or labels.ndim != 1:
        raise ValueError("participant labels and scores must align")
    if set(labels.tolist()) != {0, 1} or not np.isfinite(probabilities).all():
        raise ValueError("external metrics require finite scores and both classes")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("external scores must lie in [0, 1]")
    repeats = _positive_integer(repeats, "bootstrap repeats")
    point = _fast_metrics(labels, probabilities)
    class_indices = {label: np.flatnonzero(labels == label) for label in (0, 1)}
    draws = {metric: np.empty(repeats, dtype=np.float64) for metric in METRICS}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for repeat in range(repeats):
        sampled = np.concatenate(tuple(
            rng.choice(indices, size=indices.size, replace=True)
            for indices in class_indices.values()
        ))
        values = _fast_metrics(labels[sampled], probabilities[sampled])
        for metric in METRICS:
            draws[metric][repeat] = values[metric]
    return {
        metric: {
            "point": float(point[metric]),
            "ci95_low": float(np.quantile(draws[metric], 0.025)),
            "ci95_high": float(np.quantile(draws[metric], 0.975)),
        }
        for metric in METRICS
    }


def build_external_report(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    state: AuthorizedExternalState,
    audit: ExternalAudit,
    final_artifact_sha256: str,
    participant_manifest_sha256: str,
    cache_manifest_sha256: str,
    cache_artifact_collection_sha256: str,
    implementation_components_sha256: Mapping[str, str],
    implementation_sha256: str,
    participants_total: int,
    eligible_videos: int,
    bootstrap_repeats: int = BOOTSTRAP_REPEATS,
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    affected = int(np.sum(labels == 1))
    unaffected = int(np.sum(labels == 0))
    if (
        audit.authorization_passes != 1
        or participants_total != state.expected_participants
        or labels.size != participants_total
        or affected != state.expected_affected
        or unaffected != state.expected_unaffected
        or eligible_videos != state.expected_eligible_videos
        or audit.cache_artifacts_hashed != eligible_videos
        or audit.cache_records_loaded != eligible_videos
        or audit.feature_extractions != eligible_videos
        or audit.mirror_transforms != 2 * eligible_videos
        or audit.artifact_predictions != eligible_videos
        or audit.participant_aggregations != participants_total
        or audit.scaler_fits != 0
        or audit.model_fits != 0
        or audit.calibration_fits != 0
    ):
        raise ValueError("MEEI external audit or coverage counts are invalid")
    provenance_values = (
        final_artifact_sha256,
        participant_manifest_sha256,
        cache_manifest_sha256,
        cache_artifact_collection_sha256,
        implementation_sha256,
    )
    if provenance_values != (
        state.final_artifact_sha256,
        state.participant_manifest_sha256,
        state.cache_manifest_sha256,
        state.cache_artifact_collection_sha256,
        state.implementation_sha256,
    ):
        raise ValueError("external report provenance differs from authorization")
    components = dict(implementation_components_sha256)
    if canonical_json_sha256(components) != implementation_sha256:
        raise ValueError("external implementation components differ from aggregate")
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": "independent_cross_dataset_external_validation",
        "target": "normal_vs_facial_palsy",
        "dataset": {
            "name": "MEEI_Facial_Palsy_Standard_Set",
            "claim_unit": "participant",
            "identity_status": "publisher_participant_directory_one_video_each",
        },
        "model": {
            "training_dataset": "PalsyNet",
            "artifact": "frozen_landmark_mi_110d",
            "dimension": LOCKED_DIMENSION,
            "refit_on_meei": False,
            "recalibrated_on_meei": False,
            "threshold_changed_on_meei": False,
        },
        "protocol": {
            **_protocol(),
            "bootstrap": {
                **_protocol()["bootstrap"],
                "repeats": bootstrap_repeats,
            },
        },
        "counts": {
            "participants_total": participants_total,
            "participants_scored": int(labels.size),
            "affected_participants": affected,
            "unaffected_participants": unaffected,
            "eligible_videos": eligible_videos,
            "videos_scored": audit.artifact_predictions,
            "photos_scored": 0,
            "label_blind_qc_exclusions": participants_total - int(labels.size),
        },
        "metrics": _metric_report(labels, probabilities, repeats=bootstrap_repeats),
        "audit": audit.as_dict(),
        "provenance": {
            "authorization_sha256": state.authorization_sha256,
            "final_artifact_sha256": final_artifact_sha256,
            "participant_manifest_sha256": participant_manifest_sha256,
            "cache_manifest_sha256": cache_manifest_sha256,
            "cache_artifact_collection_sha256": cache_artifact_collection_sha256,
            "implementation_components_sha256": components,
            "implementation_sha256": implementation_sha256,
        },
        "decision": {
            "candidate_selection_performed": False,
            "external_validation_completed": True,
            "clinical_validation": False,
            "artifact_remains_frozen": True,
        },
    }
    encoded = json.dumps(report, sort_keys=True, allow_nan=False)
    if any(token in encoded for token in (
        "rec_", "grp_", "/Users/", "\\", "participant_id", "probabilities"
    )):
        raise ValueError("external report contains identifiers, paths, or row scores")
    return report


def validate_external_report(
    report: Mapping[str, object],
    *,
    labels: np.ndarray,
    probabilities: np.ndarray,
    state: AuthorizedExternalState,
    audit: ExternalAudit,
    expected_bootstrap_repeats: int = BOOTSTRAP_REPEATS,
) -> None:
    expected_top = {
        "schema_version", "claim_scope", "target", "dataset", "model",
        "protocol", "counts", "metrics", "audit", "provenance", "decision",
    }
    if not isinstance(report, Mapping) or set(report) != expected_top:
        raise ValueError("external report fields differ from the closed schema")
    protocol = report.get("protocol")
    expected_protocol = _protocol()
    expected_protocol["bootstrap"] = {
        **expected_protocol["bootstrap"],
        "repeats": expected_bootstrap_repeats,
    }
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("claim_scope") != "independent_cross_dataset_external_validation"
        or report.get("target") != "normal_vs_facial_palsy"
        or protocol != expected_protocol
        or report.get("audit") != audit.as_dict()
        or report.get("metrics") != _metric_report(
            labels, probabilities, repeats=expected_bootstrap_repeats
        )
    ):
        raise ValueError("external report protocol, audit, or metrics are invalid")
    counts = report.get("counts")
    provenance = report.get("provenance")
    if not isinstance(counts, Mapping) or (
        counts.get("participants_total") != state.expected_participants
        or counts.get("participants_scored") != len(labels)
        or counts.get("affected_participants") != state.expected_affected
        or counts.get("unaffected_participants") != state.expected_unaffected
        or counts.get("eligible_videos") != state.expected_eligible_videos
        or counts.get("photos_scored") != 0
    ):
        raise ValueError("external report counts are invalid")
    if not isinstance(provenance, Mapping) or (
        provenance.get("authorization_sha256") != state.authorization_sha256
        or provenance.get("final_artifact_sha256") != state.final_artifact_sha256
        or provenance.get("participant_manifest_sha256") != state.participant_manifest_sha256
        or provenance.get("cache_manifest_sha256") != state.cache_manifest_sha256
        or provenance.get("cache_artifact_collection_sha256") != state.cache_artifact_collection_sha256
        or provenance.get("implementation_sha256") != state.implementation_sha256
    ):
        raise ValueError("external report provenance is invalid")
    encoded = json.dumps(report, sort_keys=True, allow_nan=False)
    if any(token in encoded for token in (
        "rec_", "grp_", "/Users/", "\\", "participant_id", "probabilities"
    )):
        raise ValueError("external report leaks row-level or local information")


def write_private_no_overwrite_json(path: Path, payload: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(target.parent, 0o700)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite {target.name}")
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
        os.chmod(target, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "AUTHORIZATION_BASIS",
    "BOOTSTRAP_REPEATS",
    "BOOTSTRAP_SEED",
    "CacheArtifactInventory",
    "DEFAULT_REPORT_RELATIVE",
    "ExternalAudit",
    "ExternalMetadata",
    "AuthorizedExternalState",
    "build_expected_authorization",
    "build_external_report",
    "cache_artifact_inventory",
    "canonical_json_sha256",
    "implementation_fingerprints",
    "score_authenticated_cache",
    "validate_external_authorization",
    "validate_external_metadata",
    "validate_external_report",
    "validate_frozen_artifact_for_external",
    "write_private_no_overwrite_json",
]
