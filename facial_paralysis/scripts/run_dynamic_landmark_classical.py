#!/usr/bin/env python3
"""Run leak-safe classical candidates on dynamic clinical23_v2 caches.

Inner-fold smoke selection is available immediately. Outer scoring is gated by
the exact SHA-256 of a frozen experiment registry whose classical protocol must
match the constants in this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dynamic_landmark import (  # noqa: E402
    DYNAMIC_FEATURE_SCHEMA,
    DYNAMIC_FEATURE_SHAPE,
    DYNAMIC_MASK_SHAPE,
    load_dynamic_landmark_recordings,
)
from src.evaluation.nested_group_cv import (  # noqa: E402
    assert_outer_test_isolation,
    build_nested_group_splits,
)
from src.preprocessing.trajectory_features import (  # noqa: E402
    BLENDSHAPE_DIM,
    FUSION_DIM,
    LANDMARK_DIM,
    RAO_FUSION_DIM,
    HealthyReferencePrototype,
    trajectory_feature_set,
)


C_GRID = (0.01, 0.1, 1.0, 10.0)
BOOTSTRAP_REPEATS = 5000
# Task 7 must replace this sentinel with the committed, one-shot experiment
# registry digest before any real outer evaluation can run.
PINNED_TASK7_REGISTRY_SHA256: str | None = None
NUISANCE_FEATURE_NAMES = (
    "duration_seconds",
    "bitrate_proxy_bytes_per_second",
    "detection_rate",
    "luminance_mean",
    "frame_difference_mean",
    "face_scale_mean",
    "face_scale_std",
    "eye_line_roll_degrees_mean",
    "eye_line_roll_degrees_std",
)
CANDIDATE_REGISTRY = {
    "nuisance": len(NUISANCE_FEATURE_NAMES),
    "blendshape": BLENDSHAPE_DIM,
    "landmark": LANDMARK_DIM,
    "fusion": FUSION_DIM,
    "rao_fusion": RAO_FUSION_DIM,
}
COLLECTION_SCHEMA = "palsynet_clinical23_v2_windows_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REC_ID = re.compile(r"^rec_[0-9a-f]{64}$")
_GROUP_ID = re.compile(r"^grp_[0-9a-f]{64}$")
_C_TIE_TOLERANCE = 1e-12
_ALLOWED_CLAIM_STATES = {
    ("video_held_out", "unreviewed"),
    ("person_held_out", "reviewed"),
    ("synthetic_group_held_out", "synthetic"),
}
_TASK2_TOP_FIELDS = {
    "schema_version", "dataset", "feature_schema", "feature_shape",
    "capture_mirrored", "claim_unit", "identity_status", "protocol",
    "provenance", "counts", "records", "excluded",
}
_TASK2_PROTOCOL = {
    "windows_per_recording": 4,
    "frames_per_window": 32,
    "minimum_coverage": 0.90,
    "minimum_retained": 47,
    "minimum_landmark_variation_fraction": 0.95,
}
_TASK2_PROVENANCE_FIELDS = {
    "model_sha256", "identity_manifest_sha256", "identity_fingerprints",
    "source_collection_sha256", "corpus_inventory", "dependency_versions",
    "producer_sources",
}
_TASK2_IDENTITY_FINGERPRINT_FIELDS = {
    "bundle_provenance_sha256",
    "embedding_collection_sha256",
    "source_collection_sha256",
}
_TASK2_CORPUS_INVENTORY = {
    "recordings": 49,
    "fps": 30.0,
    "total_frames": 177_511,
    "minimum_frames": 172,
    "duration_minutes": 98.61722222222221,
}
_TASK2_DEPENDENCY_FIELDS = {"python", "numpy", "mediapipe", "opencv", "torch"}
_TASK2_DEPENDENCY_DISTRIBUTIONS = {
    "python": {"python"},
    "numpy": {"numpy"},
    "mediapipe": {"mediapipe"},
    "torch": {"torch"},
    "opencv": {
        "opencv-python",
        "opencv-contrib-python",
        "opencv-python-headless",
        "opencv-contrib-python-headless",
    },
}
_TASK2_PRODUCER_COMPONENTS = {
    "builder", "action_bundle", "clinical_landmarks",
    "dynamic_landmark_loader", "feature_registry",
}
_TASK2_COUNT_FIELDS = {
    "discovered", "retained", "excluded", "retained_affected",
    "retained_unaffected", "retained_groups",
}
_TASK2_RECORD_FIELDS = {
    "recording_id", "group_id", "source_sha256", "label",
    "source_frame_count", "fps", "window_starts", "frames_per_window",
    "timestamp_unit", "frame_width", "frame_height", "file_size_bytes",
    "coverage", "landmark_varied", "landmark_variation_stat", "nuisance",
}
_TASK2_EXCLUDED_FIELDS = {
    "recording_id", "group_id", "source_sha256", "label", "reason",
}


@dataclass(frozen=True)
class ClassicalDataset:
    """In-memory validated arrays and deidentified recording metadata."""

    features: np.ndarray
    valid_masks: np.ndarray
    timestamps: np.ndarray
    source_frame_indices: np.ndarray
    nuisance: np.ndarray
    labels: np.ndarray
    group_ids: np.ndarray
    recording_ids: tuple[str, ...]
    claim_unit: str = "synthetic_group_held_out"
    identity_status: str = "synthetic"
    collection_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        features = np.asarray(self.features)
        masks = np.asarray(self.valid_masks)
        timestamps = np.asarray(self.timestamps)
        source_indices = np.asarray(self.source_frame_indices)
        nuisance = np.asarray(self.nuisance)
        labels = np.asarray(self.labels)
        groups = np.asarray(self.group_ids)
        count = features.shape[0] if features.ndim else 0
        if features.ndim != 4 or features.shape[1:] != DYNAMIC_FEATURE_SHAPE:
            raise ValueError(
                f"features must have shape (N, {DYNAMIC_FEATURE_SHAPE})"
            )
        temporal_shape = (count,) + DYNAMIC_MASK_SHAPE
        if masks.shape != temporal_shape or masks.dtype != np.dtype(bool):
            raise ValueError(f"valid_masks must be bool with shape {temporal_shape}")
        if timestamps.shape != temporal_shape or timestamps.dtype.kind not in {
            "f", "i", "u",
        }:
            raise ValueError("timestamps must be numeric and align with masks")
        if source_indices.shape != temporal_shape or source_indices.dtype.kind not in {
            "i", "u",
        }:
            raise ValueError("source_frame_indices must be integer and align with masks")
        if nuisance.shape != (count, len(NUISANCE_FEATURE_NAMES)):
            raise ValueError("nuisance matrix does not match its frozen nine-field schema")
        if not np.isfinite(nuisance).all():
            raise ValueError("nuisance values must be finite")
        if labels.shape != (count,) or labels.dtype.kind not in {"i", "u", "b"}:
            raise ValueError("labels must be a one-dimensional integer array")
        if not np.isin(labels, (0, 1)).all():
            raise ValueError("labels must be binary")
        if groups.shape != (count,):
            raise ValueError("group_ids must align with records")
        group_values = groups.tolist()
        if any(not isinstance(group, str) or _GROUP_ID.fullmatch(group) is None
               for group in group_values):
            raise ValueError("group_ids must be canonical opaque IDs")
        if len(self.recording_ids) != count or len(set(self.recording_ids)) != count:
            raise ValueError("recording_ids must be unique and align with records")
        if any(_REC_ID.fullmatch(value) is None for value in self.recording_ids):
            raise ValueError("recording_ids must be canonical opaque IDs")
        if (self.claim_unit, self.identity_status) not in _ALLOWED_CLAIM_STATES:
            raise ValueError("claim unit and identity status are not a frozen combination")
        if (
            self.claim_unit != "synthetic_group_held_out"
            and (
                self.collection_manifest_sha256 is None
                or _SHA256.fullmatch(self.collection_manifest_sha256) is None
            )
        ):
            raise ValueError("real cohorts require the validated collection manifest hash")
        if not np.isfinite(timestamps).all() or not np.all(
            timestamps[:, :, 1:] > timestamps[:, :, :-1]
        ):
            raise ValueError("timestamps must be finite and increase within each window")
        if not np.all(
            source_indices[:, :, 1:] - source_indices[:, :, :-1] == 1
        ):
            raise ValueError("source frames must remain adjacent within windows")
        if not np.isfinite(features[masks]).all():
            raise ValueError("valid feature rows must be finite")
        group_labels: dict[str, int] = {}
        for group, label in zip(group_values, labels.astype(int).tolist()):
            previous = group_labels.setdefault(group, label)
            if previous != label:
                raise ValueError("a group cannot cross binary labels")


@dataclass(frozen=True)
class FitAuditEvent:
    outer_fold: int
    inner_fold: int | None
    candidate: str
    fit_kind: str
    fit_indices: tuple[int, ...]


@dataclass(frozen=True)
class InnerSelectionResult:
    outer_fold_number: int
    candidate: str
    selected_c: float
    c_scores: dict[float, float]
    outer_train_indices: tuple[int, ...]
    outer_test_indices: tuple[int, ...]
    oof_probabilities: np.ndarray
    audit_events: tuple[FitAuditEvent, ...]


@dataclass(frozen=True)
class OuterCandidateResult:
    candidate: str
    metrics: dict[str, float]
    fold_metrics: tuple[dict[str, float], ...]
    probabilities: np.ndarray
    outer_fold_by_record: np.ndarray
    selected_c_by_fold: tuple[float, ...]
    audit_events: tuple[FitAuditEvent, ...]


class _SyntheticOuterAuthorization:
    """Unforgeable-by-API marker for synthetic mechanics tests only."""

    __slots__ = ("_secret",)

    def __init__(self, secret: object) -> None:
        self._secret = secret


_SYNTHETIC_OUTER_SECRET = object()
_SYNTHETIC_OUTER_AUTHORIZATION = _SyntheticOuterAuthorization(
    _SYNTHETIC_OUTER_SECRET
)


def frozen_classical_protocol() -> dict[str, object]:
    """Return the JSON-serializable protocol that an outer registry must pin."""
    return {
        "schema_version": "dynamic_landmark_classical_v1",
        "candidate_dimensions": dict(CANDIDATE_REGISTRY),
        "candidates": list(CANDIDATE_REGISTRY),
        "c_grid": list(C_GRID),
        "outer_folds": 5,
        "inner_folds": 4,
        "primary_metric": "pooled_group_auroc",
        "secondary_metrics": [
            "average_precision",
            "brier",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
        ],
        "probability_threshold": 0.5,
        "group_probability_aggregation": "mean",
        "training_group_weight": "inverse_recordings_per_group",
        "c_tie_break": "smaller_c_within_1e-12",
        "bootstrap": {
            "unit": "group",
            "paired": True,
            "stratified_by_binary_label": True,
            "repeats": BOOTSTRAP_REPEATS,
            "interval_scope": "fixed_oof_predictions_descriptive",
        },
    }


def _unique_json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _read_json(path: Path) -> tuple[dict[str, object], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read JSON artifact {path.name!r}") from exc
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_json_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid unique-key JSON artifact {path.name!r}") from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON artifact must contain an object")
    return payload, raw


def validate_outer_registry(
    registry_path: str | Path,
    supplied_sha256: str | None,
) -> str:
    """Authorize outer scoring only for exact bytes and the frozen protocol."""
    if PINNED_TASK7_REGISTRY_SHA256 is None:
        raise ValueError(
            "real outer scoring is disabled until Task 7 pins the registry SHA-256"
        )
    if supplied_sha256 is None or _SHA256.fullmatch(supplied_sha256) is None:
        raise ValueError("outer scoring requires a lowercase frozen registry SHA-256")
    if supplied_sha256 != PINNED_TASK7_REGISTRY_SHA256:
        raise ValueError("supplied registry SHA-256 is not the Task 7 pinned digest")
    path = Path(registry_path)
    if not path.is_file():
        raise ValueError("experiment registry must be a regular file")
    payload, raw = _read_json(path)
    observed = hashlib.sha256(raw).hexdigest()
    if observed != supplied_sha256:
        raise ValueError("experiment registry SHA-256 does not match supplied freeze")
    if payload.get("schema_version") != "dynamic_landmark_experiment_registry_v1":
        raise ValueError("experiment registry schema is unsupported")
    if payload.get("classical_protocol") != frozen_classical_protocol():
        raise ValueError("experiment registry classical protocol has drifted")
    return observed


def _exact_object(value: object, fields: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} fields differ from the frozen Task2 contract")
    return value


def _manifest_integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


def _manifest_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _manifest_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _task2_source_fingerprint(rows: Sequence[Mapping[str, object]]) -> str:
    fingerprint = hashlib.sha256()
    for row in sorted(rows, key=lambda item: (str(item["label"]), str(item["source_sha256"]))):
        fingerprint.update(
            f"{row['label']}:{row['source_sha256']}\n".encode("ascii")
        )
    return fingerprint.hexdigest()


def _validate_task2_provenance(
    value: object,
    discovered_rows: Sequence[Mapping[str, object]],
) -> None:
    provenance = _exact_object(
        value, _TASK2_PROVENANCE_FIELDS, "collection provenance"
    )
    _manifest_sha256(provenance["model_sha256"], "model_sha256")
    _manifest_sha256(
        provenance["identity_manifest_sha256"], "identity_manifest_sha256"
    )
    identity = _exact_object(
        provenance["identity_fingerprints"],
        _TASK2_IDENTITY_FINGERPRINT_FIELDS,
        "identity fingerprints",
    )
    for name, digest in identity.items():
        _manifest_sha256(digest, f"identity_fingerprints.{name}")
    source_fingerprint = _manifest_sha256(
        provenance["source_collection_sha256"], "source_collection_sha256"
    )
    observed_fingerprint = _task2_source_fingerprint(discovered_rows)
    if source_fingerprint != observed_fingerprint:
        raise ValueError("source collection fingerprint differs from discovered rows")
    if identity["source_collection_sha256"] != source_fingerprint:
        raise ValueError("identity and collection source fingerprints differ")

    inventory = _exact_object(
        provenance["corpus_inventory"],
        set(_TASK2_CORPUS_INVENTORY),
        "corpus inventory",
    )
    for name, expected in _TASK2_CORPUS_INVENTORY.items():
        observed = inventory[name]
        if isinstance(expected, int):
            observed = _manifest_integer(observed, f"corpus_inventory.{name}")
            if observed != expected:
                raise ValueError("corpus inventory differs from the frozen PalsyNet audit")
        else:
            observed_number = _manifest_number(
                observed, f"corpus_inventory.{name}"
            )
            if abs(observed_number - expected) > 1e-12:
                raise ValueError("corpus inventory differs from the frozen PalsyNet audit")

    dependencies = _exact_object(
        provenance["dependency_versions"],
        _TASK2_DEPENDENCY_FIELDS,
        "dependency versions",
    )
    for name, version in dependencies.items():
        if not isinstance(version, str):
            raise ValueError(f"dependency_versions.{name} is not exact")
        distribution, separator, release = version.partition("==")
        if (
            separator != "=="
            or distribution not in _TASK2_DEPENDENCY_DISTRIBUTIONS[name]
            or not release
            or re.search(r"[=\s]", release) is not None
            or "unknown" in release.lower()
        ):
            raise ValueError(f"dependency_versions.{name} is not exact")

    producer_sources = _exact_object(
        provenance["producer_sources"],
        {"components", "aggregate_sha256"},
        "producer sources",
    )
    components = _exact_object(
        producer_sources["components"],
        _TASK2_PRODUCER_COMPONENTS,
        "producer source components",
    )
    aggregate = hashlib.sha256()
    for name, digest in sorted(components.items()):
        checked = _manifest_sha256(digest, f"producer_sources.components.{name}")
        aggregate.update(f"{name}:{checked}\n".encode("ascii"))
    expected_aggregate = _manifest_sha256(
        producer_sources["aggregate_sha256"], "producer source aggregate"
    )
    if aggregate.hexdigest() != expected_aggregate:
        raise ValueError("producer source aggregate differs from its components")


def _validate_task2_collection_manifest(
    payload: dict[str, object],
) -> tuple[list[dict[str, object]], str, str]:
    if set(payload) != _TASK2_TOP_FIELDS:
        raise ValueError("collection top-level fields differ from Task2")
    if payload["schema_version"] != COLLECTION_SCHEMA:
        raise ValueError("collection manifest schema is unsupported")
    if payload["dataset"] != "PalsyNet":
        raise ValueError("collection manifest dataset must be PalsyNet")
    if payload["feature_schema"] != DYNAMIC_FEATURE_SCHEMA:
        raise ValueError("collection feature schema is not clinical23_v2")
    if payload["feature_shape"] != list(DYNAMIC_FEATURE_SHAPE):
        raise ValueError("collection feature shape has drifted")
    if payload["capture_mirrored"] is not None:
        raise ValueError("Task2 capture mirroring must remain unknown")
    claim_unit = payload["claim_unit"]
    identity_status = payload["identity_status"]
    if (claim_unit, identity_status) not in {
        ("video_held_out", "unreviewed"),
        ("person_held_out", "reviewed"),
    }:
        raise ValueError("collection claim metadata is not a frozen reviewed state")
    if payload["protocol"] != _TASK2_PROTOCOL:
        raise ValueError("collection protocol differs from the frozen Task2 gates")

    rows = payload["records"]
    excluded = payload["excluded"]
    if not isinstance(rows, list) or not isinstance(excluded, list):
        raise ValueError("collection records and excluded rows must be lists")
    if not 47 <= len(rows) <= 49 or len(rows) + len(excluded) != 49:
        raise ValueError("Task2 must retain 47 through 49 of exactly 49 recordings")
    if rows != sorted(rows, key=lambda row: str(row.get("recording_id"))):
        raise ValueError("retained rows must use deterministic recording order")
    if excluded != sorted(excluded, key=lambda row: str(row.get("recording_id"))):
        raise ValueError("excluded rows must use deterministic recording order")

    retained_label_counts = {"affected": 0, "unaffected": 0}
    discovered_label_counts = {"affected": 0, "unaffected": 0}
    seen_recordings: set[str] = set()
    seen_sources: set[str] = set()
    retained_groups: set[str] = set()
    groups_by_label: dict[str, set[str]] = {"affected": set(), "unaffected": set()}
    group_labels: dict[str, str] = {}
    varied = 0
    checked_rows: list[dict[str, object]] = []
    for row in rows:
        checked = _exact_object(row, _TASK2_RECORD_FIELDS, "retained record")
        recording_id = checked["recording_id"]
        group_id = checked["group_id"]
        source_sha = checked["source_sha256"]
        label = checked["label"]
        if not isinstance(recording_id, str) or _REC_ID.fullmatch(recording_id) is None:
            raise ValueError("retained recording ID is not canonical")
        if not isinstance(group_id, str) or _GROUP_ID.fullmatch(group_id) is None:
            raise ValueError("retained group ID is not canonical")
        _manifest_sha256(source_sha, "retained source_sha256")
        if label not in retained_label_counts:
            raise ValueError("retained label must be affected or unaffected")
        if recording_id in seen_recordings or source_sha in seen_sources:
            raise ValueError("discovered recording IDs and source hashes must be unique")
        seen_recordings.add(recording_id)
        seen_sources.add(source_sha)
        previous = group_labels.setdefault(group_id, label)
        if previous != label:
            raise ValueError("a Task2 group cannot cross labels")
        retained_groups.add(group_id)
        groups_by_label[label].add(group_id)
        retained_label_counts[label] += 1
        discovered_label_counts[label] += 1
        _manifest_integer(checked["source_frame_count"], "source_frame_count", minimum=128)
        if abs(_manifest_number(checked["fps"], "fps") - 30.0) > 1e-6:
            raise ValueError("retained recording FPS differs from audited 30 Hz")
        starts = checked["window_starts"]
        if (
            not isinstance(starts, list)
            or len(starts) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                   for value in starts)
        ):
            raise ValueError("window_starts must contain four nonnegative integers")
        if checked["frames_per_window"] != 32 or checked["timestamp_unit"] != "seconds":
            raise ValueError("retained temporal schema differs from Task2")
        for name in ("frame_width", "frame_height", "file_size_bytes"):
            _manifest_integer(checked[name], name, minimum=1)
        coverage = _manifest_number(checked["coverage"], "coverage")
        if not 0.90 <= coverage <= 1.0:
            raise ValueError("retained coverage is below the Task2 gate")
        if not isinstance(checked["landmark_varied"], bool):
            raise ValueError("landmark_varied must be bool")
        variation = _manifest_number(
            checked["landmark_variation_stat"], "landmark_variation_stat"
        )
        if variation < 0.0 or checked["landmark_varied"] != (variation > 0.0):
            raise ValueError("landmark variation metadata is inconsistent")
        varied += int(checked["landmark_varied"])
        nuisance = _exact_object(
            checked["nuisance"], set(NUISANCE_FEATURE_NAMES), "recording nuisance"
        )
        for name, value in nuisance.items():
            _manifest_number(value, f"nuisance.{name}")
        checked_rows.append(checked)

    checked_excluded: list[dict[str, object]] = []
    for row in excluded:
        checked = _exact_object(row, _TASK2_EXCLUDED_FIELDS, "excluded record")
        recording_id = checked["recording_id"]
        group_id = checked["group_id"]
        source_sha = checked["source_sha256"]
        label = checked["label"]
        if not isinstance(recording_id, str) or _REC_ID.fullmatch(recording_id) is None:
            raise ValueError("excluded recording ID is not canonical")
        if not isinstance(group_id, str) or _GROUP_ID.fullmatch(group_id) is None:
            raise ValueError("excluded group ID is not canonical")
        _manifest_sha256(source_sha, "excluded source_sha256")
        if label not in discovered_label_counts:
            raise ValueError("excluded label must be affected or unaffected")
        if recording_id in seen_recordings or source_sha in seen_sources:
            raise ValueError("discovered recording IDs and source hashes must be unique")
        if not isinstance(checked["reason"], str) or not checked["reason"]:
            raise ValueError("excluded record requires a nonempty reason")
        seen_recordings.add(recording_id)
        seen_sources.add(source_sha)
        previous = group_labels.setdefault(group_id, label)
        if previous != label:
            raise ValueError("a Task2 group cannot cross labels")
        discovered_label_counts[label] += 1
        checked_excluded.append(checked)

    if discovered_label_counts != {"affected": 27, "unaffected": 22}:
        raise ValueError("Task2 discovered labels differ from frozen 27/22 counts")
    if any(len(group_set) < 5 for group_set in groups_by_label.values()):
        raise ValueError("each retained class needs at least five identity groups")
    if varied / len(checked_rows) < 0.95:
        raise ValueError("Task2 landmark variation collection gate failed")

    counts = _exact_object(payload["counts"], _TASK2_COUNT_FIELDS, "collection counts")
    checked_counts = {
        name: _manifest_integer(value, f"counts.{name}")
        for name, value in counts.items()
    }
    expected_counts = {
        "discovered": 49,
        "retained": len(checked_rows),
        "excluded": len(checked_excluded),
        "retained_affected": retained_label_counts["affected"],
        "retained_unaffected": retained_label_counts["unaffected"],
        "retained_groups": len(retained_groups),
    }
    if checked_counts != expected_counts:
        raise ValueError("collection counts differ from validated rows")
    _validate_task2_provenance(
        payload["provenance"], [*checked_rows, *checked_excluded]
    )
    return checked_rows, str(claim_unit), str(identity_status)


def _record_landmark_variation(
    features: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[bool, float]:
    statistic = 0.0
    for window_index in range(4):
        valid = features[window_index, :, -23:][valid_mask[window_index]]
        if valid.shape[0] >= 2:
            statistic = max(
                statistic,
                float(np.max(np.ptp(valid.astype(np.float64, copy=False), axis=0))),
            )
    return statistic > 0.0, statistic


def load_classical_dataset(cache_root: str | Path) -> ClassicalDataset:
    """Load the Task-2 manifest plus NPZs through the public validating loader."""
    root = Path(cache_root)
    if not root.is_dir():
        raise ValueError("cache root must be a directory")
    manifest_path = root / "collection_manifest.json"
    payload, raw_manifest = _read_json(manifest_path)
    sorted_rows, claim_unit, identity_status = _validate_task2_collection_manifest(
        payload
    )
    expected_paths = {
        root / f"{row['recording_id']}.npz"
        for row in sorted_rows
    }
    observed_paths = set(root.glob("*.npz"))
    if observed_paths != expected_paths:
        raise ValueError("cache NPZ set differs from collection manifest")
    records = load_dynamic_landmark_recordings(expected_paths)
    by_recording = {record.recording_id: record for record in records}

    nuisance_rows: list[np.ndarray] = []
    ordered_records = []
    for row in sorted_rows:
        recording_id = str(row["recording_id"])
        record = by_recording.get(recording_id)
        if record is None:
            raise ValueError("manifest recording is absent from validated caches")
        manifest_label = {"unaffected": 0, "affected": 1}.get(row.get("label"))
        if manifest_label is None:
            raise ValueError("collection manifest label must be affected or unaffected")
        if (
            row.get("group_id") != record.group_id
            or row.get("source_sha256") != record.source_sha256
            or manifest_label != record.label
        ):
            raise ValueError("collection manifest/cache provenance disagreement")
        observed_varied, observed_variation = _record_landmark_variation(
            record.features, record.valid_mask
        )
        if (
            row["source_frame_count"] != record.source_frame_count
            or row["window_starts"]
            != record.source_frame_indices[:, 0].astype(int).tolist()
            or row["frames_per_window"] != DYNAMIC_FEATURE_SHAPE[1]
            or row["timestamp_unit"] != record.timestamp_unit
            or abs(float(row["coverage"]) - record.coverage) > 1e-12
            or row["landmark_varied"] != observed_varied
            or abs(float(row["landmark_variation_stat"]) - observed_variation) > 1e-12
        ):
            raise ValueError("collection temporal or variation metadata differs from NPZ")
        nuisance = row.get("nuisance")
        if not isinstance(nuisance, dict) or set(nuisance) != set(NUISANCE_FEATURE_NAMES):
            raise ValueError("recording nuisance fields differ from frozen schema")
        try:
            nuisance_vector = np.asarray(
                [float(nuisance[name]) for name in NUISANCE_FEATURE_NAMES],
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("recording nuisance values must be numeric") from exc
        if not np.isfinite(nuisance_vector).all():
            raise ValueError("recording nuisance values must be finite")
        expected_duration = record.source_frame_count / float(row["fps"])
        if (
            abs(float(nuisance["duration_seconds"]) - expected_duration) > 1e-12
            or abs(float(nuisance["detection_rate"]) - record.coverage) > 1e-12
        ):
            raise ValueError("recording nuisance timing/coverage differs from NPZ")
        nuisance_rows.append(nuisance_vector)
        ordered_records.append(record)

    return ClassicalDataset(
        features=np.stack([record.features for record in ordered_records]),
        valid_masks=np.stack([record.valid_mask for record in ordered_records]),
        timestamps=np.stack([record.timestamps for record in ordered_records]),
        source_frame_indices=np.stack([
            record.source_frame_indices for record in ordered_records
        ]),
        nuisance=np.stack(nuisance_rows),
        labels=np.asarray([record.label for record in ordered_records], dtype=np.int64),
        group_ids=np.asarray([record.group_id for record in ordered_records]),
        recording_ids=tuple(record.recording_id for record in ordered_records),
        claim_unit=claim_unit,
        identity_status=identity_status,
        collection_manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
    )


def group_sample_weights(group_ids: Sequence[object]) -> np.ndarray:
    """Give every group total training weight one, split across its recordings."""
    groups = np.asarray(group_ids)
    if groups.ndim != 1 or groups.size == 0:
        raise ValueError("group_ids must be a nonempty one-dimensional array")
    values = groups.tolist()
    counts = {group: values.count(group) for group in set(values)}
    return np.asarray([1.0 / counts[group] for group in values], dtype=np.float64)


def group_mean_predictions(
    labels: Sequence[int],
    group_ids: Sequence[object],
    probabilities: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse recording probabilities to one deterministic mean per group."""
    labels = np.asarray(labels)
    groups = np.asarray(group_ids)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or groups.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("labels, groups, and probabilities must be one-dimensional")
    if labels.shape != groups.shape or labels.shape != probabilities.shape or labels.size == 0:
        raise ValueError("labels, groups, and probabilities must have equal nonzero length")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("probabilities must be finite values from zero through one")
    ordered_groups = sorted(set(groups.tolist()), key=str)
    group_labels = []
    group_probabilities = []
    for group in ordered_groups:
        indices = np.flatnonzero(groups == group)
        observed_labels = np.unique(labels[indices])
        if observed_labels.size != 1:
            raise ValueError("a group cannot cross labels")
        group_labels.append(int(observed_labels[0]))
        group_probabilities.append(float(probabilities[indices].mean()))
    return (
        np.asarray(group_labels, dtype=np.int64),
        np.asarray(ordered_groups),
        np.asarray(group_probabilities, dtype=np.float64),
    )


def binary_group_metrics(
    labels: Sequence[int],
    group_ids: Sequence[object],
    probabilities: Sequence[float],
) -> dict[str, float]:
    """Evaluate pooled group probabilities with a fixed threshold of 0.5."""
    group_labels, _, group_probabilities = group_mean_predictions(
        labels, group_ids, probabilities
    )
    if set(group_labels.tolist()) != {0, 1}:
        raise ValueError("binary group metrics require both classes")
    predictions = (group_probabilities >= 0.5).astype(np.int64)
    positive = group_labels == 1
    negative = ~positive
    sensitivity = float(np.mean(predictions[positive] == 1))
    specificity = float(np.mean(predictions[negative] == 0))
    return {
        "auroc": float(roc_auc_score(group_labels, group_probabilities)),
        "average_precision": float(
            average_precision_score(group_labels, group_probabilities)
        ),
        "brier": float(brier_score_loss(group_labels, group_probabilities)),
        "balanced_accuracy": float(
            balanced_accuracy_score(group_labels, predictions)
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def choose_regularization_c(c_scores: Mapping[float, float]) -> float:
    """Choose maximum pooled inner OOF AUROC; numerical ties use smaller C."""
    if set(c_scores) != set(C_GRID):
        raise ValueError("C scores must cover the complete frozen grid")
    scores = {float(c_value): float(c_scores[c_value]) for c_value in C_GRID}
    if not np.isfinite(tuple(scores.values())).all():
        raise ValueError("C scores must be finite")
    maximum = max(scores.values())
    tied = [c_value for c_value in C_GRID
            if maximum - scores[c_value] <= _C_TIE_TOLERANCE]
    return float(min(tied))


def _base_feature_matrices(dataset: ClassicalDataset) -> dict[str, np.ndarray]:
    fusion_rows = np.stack([
        trajectory_feature_set(
            "fusion",
            dataset.features[index],
            dataset.valid_masks[index],
            dataset.timestamps[index],
            dataset.source_frame_indices[index],
        )
        for index in range(dataset.labels.size)
    ])
    if fusion_rows.shape[1] != FUSION_DIM:
        raise AssertionError("fusion feature dimension drifted")
    return {
        "nuisance": dataset.nuisance.astype(np.float64, copy=True),
        "blendshape": fusion_rows[:, :BLENDSHAPE_DIM],
        "landmark": fusion_rows[:, BLENDSHAPE_DIM:],
        "fusion": fusion_rows,
    }


def _candidate_fold_features(
    dataset: ClassicalDataset,
    candidate: str,
    fit_indices: np.ndarray,
    apply_indices: np.ndarray,
    base_matrices: Mapping[str, np.ndarray],
    *,
    outer_fold: int,
    inner_fold: int | None,
    audit_events: list[FitAuditEvent],
) -> tuple[np.ndarray, np.ndarray]:
    if candidate != "rao_fusion":
        matrix = base_matrices[candidate]
        return matrix[fit_indices], matrix[apply_indices]
    control_indices = fit_indices[dataset.labels[fit_indices] == 0]
    if control_indices.size < 2:
        raise ValueError("Rao reference requires at least two training controls")
    prototype = HealthyReferencePrototype().fit(
        dataset.features[control_indices],
        dataset.valid_masks[control_indices],
        dataset.timestamps[control_indices],
        dataset.source_frame_indices[control_indices],
        record_indices=control_indices,
    )
    audit_events.append(FitAuditEvent(
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        candidate=candidate,
        fit_kind="healthy_reference",
        fit_indices=tuple(int(index) for index in control_indices),
    ))
    train_distances = prototype.transform(
        dataset.features[fit_indices],
        dataset.valid_masks[fit_indices],
        dataset.timestamps[fit_indices],
        dataset.source_frame_indices[fit_indices],
    )
    apply_distances = prototype.transform(
        dataset.features[apply_indices],
        dataset.valid_masks[apply_indices],
        dataset.timestamps[apply_indices],
        dataset.source_frame_indices[apply_indices],
    )
    return (
        np.concatenate((base_matrices["fusion"][fit_indices], train_distances), axis=1),
        np.concatenate((base_matrices["fusion"][apply_indices], apply_distances), axis=1),
    )


def _fit_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_groups: np.ndarray,
    c_value: float,
) -> LogisticRegression:
    model = LogisticRegression(
        C=float(c_value),
        penalty="l2",
        solver="liblinear",
        max_iter=2000,
        random_state=0,
    )
    model.fit(
        x_train,
        y_train,
        sample_weight=group_sample_weights(train_groups),
    )
    return model


def run_inner_candidate_selection(
    dataset: ClassicalDataset,
    candidate: str,
    *,
    outer_fold_number: int,
) -> InnerSelectionResult:
    """Select C using pooled group OOF predictions inside one outer train set."""
    if candidate not in CANDIDATE_REGISTRY:
        raise ValueError(f"unknown classical candidate {candidate!r}")
    folds = build_nested_group_splits(dataset.labels, dataset.group_ids)
    if not 0 <= outer_fold_number < len(folds):
        raise ValueError("outer_fold_number is outside the frozen five folds")
    outer = folds[outer_fold_number]
    base_matrices = _base_feature_matrices(dataset)
    audit_events: list[FitAuditEvent] = []
    prepared = []
    for inner_number, inner in enumerate(outer.inner_folds):
        assert_outer_test_isolation(
            outer.test_indices,
            train_indices=inner.train_indices,
            validation_indices=inner.validation_indices,
            scaler_fit_indices=inner.train_indices,
            selection_indices=inner.validation_indices,
        )
        x_train, x_validation = _candidate_fold_features(
            dataset,
            candidate,
            inner.train_indices,
            inner.validation_indices,
            base_matrices,
            outer_fold=outer_fold_number,
            inner_fold=inner_number,
            audit_events=audit_events,
        )
        scaler = StandardScaler().fit(x_train)
        audit_events.append(FitAuditEvent(
            outer_fold=outer_fold_number,
            inner_fold=inner_number,
            candidate=candidate,
            fit_kind="standard_scaler",
            fit_indices=tuple(int(index) for index in inner.train_indices),
        ))
        prepared.append((
            inner,
            scaler.transform(x_train),
            scaler.transform(x_validation),
        ))

    score_by_c: dict[float, float] = {}
    oof_by_c: dict[float, np.ndarray] = {}
    for c_value in C_GRID:
        full_oof = np.full(dataset.labels.shape[0], np.nan, dtype=np.float64)
        for inner_number, (inner, x_train, x_validation) in enumerate(prepared):
            model = _fit_logistic(
                x_train,
                dataset.labels[inner.train_indices],
                dataset.group_ids[inner.train_indices],
                c_value,
            )
            audit_events.append(FitAuditEvent(
                outer_fold=outer_fold_number,
                inner_fold=inner_number,
                candidate=candidate,
                fit_kind=f"logistic_regression_C={c_value:g}",
                fit_indices=tuple(int(index) for index in inner.train_indices),
            ))
            full_oof[inner.validation_indices] = model.predict_proba(x_validation)[:, 1]
        oof = full_oof[outer.train_indices]
        if not np.isfinite(oof).all():
            raise AssertionError("inner validation folds did not fill pooled OOF predictions")
        score_by_c[c_value] = binary_group_metrics(
            dataset.labels[outer.train_indices],
            dataset.group_ids[outer.train_indices],
            oof,
        )["auroc"]
        oof_by_c[c_value] = oof
    selected = choose_regularization_c(score_by_c)
    return InnerSelectionResult(
        outer_fold_number=outer_fold_number,
        candidate=candidate,
        selected_c=selected,
        c_scores=score_by_c,
        outer_train_indices=tuple(int(index) for index in outer.train_indices),
        outer_test_indices=tuple(int(index) for index in outer.test_indices),
        oof_probabilities=oof_by_c[selected],
        audit_events=tuple(audit_events),
    )


def _run_outer_candidate_unlocked(
    dataset: ClassicalDataset,
    candidate: str,
    *,
    authorization: _SyntheticOuterAuthorization | None = None,
) -> OuterCandidateResult:
    if (
        not isinstance(authorization, _SyntheticOuterAuthorization)
        or authorization._secret is not _SYNTHETIC_OUTER_SECRET
    ):
        raise ValueError("outer mechanics require the private synthetic authorization")
    if (dataset.claim_unit, dataset.identity_status) != (
        "synthetic_group_held_out", "synthetic"
    ):
        raise ValueError("outer mechanics cannot evaluate a real cohort before Task7")
    folds = build_nested_group_splits(dataset.labels, dataset.group_ids)
    base_matrices = _base_feature_matrices(dataset)
    probabilities = np.full(dataset.labels.shape[0], np.nan, dtype=np.float64)
    outer_fold_by_record = np.full(dataset.labels.shape[0], -1, dtype=np.int64)
    fold_metrics: list[dict[str, float]] = []
    selected_values: list[float] = []
    all_events: list[FitAuditEvent] = []
    for outer_number, outer in enumerate(folds):
        selection = run_inner_candidate_selection(
            dataset, candidate, outer_fold_number=outer_number
        )
        all_events.extend(selection.audit_events)
        selected_values.append(selection.selected_c)
        assert_outer_test_isolation(
            outer.test_indices,
            train_indices=outer.train_indices,
            scaler_fit_indices=outer.train_indices,
            prototype_fit_indices=outer.train_indices[
                dataset.labels[outer.train_indices] == 0
            ] if candidate == "rao_fusion" else np.asarray([], dtype=np.int64),
        )
        x_train, x_test = _candidate_fold_features(
            dataset,
            candidate,
            outer.train_indices,
            outer.test_indices,
            base_matrices,
            outer_fold=outer_number,
            inner_fold=None,
            audit_events=all_events,
        )
        scaler = StandardScaler().fit(x_train)
        all_events.append(FitAuditEvent(
            outer_fold=outer_number,
            inner_fold=None,
            candidate=candidate,
            fit_kind="standard_scaler",
            fit_indices=tuple(int(index) for index in outer.train_indices),
        ))
        scaled_train = scaler.transform(x_train)
        scaled_test = scaler.transform(x_test)
        model = _fit_logistic(
            scaled_train,
            dataset.labels[outer.train_indices],
            dataset.group_ids[outer.train_indices],
            selection.selected_c,
        )
        all_events.append(FitAuditEvent(
            outer_fold=outer_number,
            inner_fold=None,
            candidate=candidate,
            fit_kind=f"logistic_regression_C={selection.selected_c:g}",
            fit_indices=tuple(int(index) for index in outer.train_indices),
        ))
        fold_probabilities = model.predict_proba(scaled_test)[:, 1]
        probabilities[outer.test_indices] = fold_probabilities
        outer_fold_by_record[outer.test_indices] = outer_number
        fold_metrics.append(binary_group_metrics(
            dataset.labels[outer.test_indices],
            dataset.group_ids[outer.test_indices],
            fold_probabilities,
        ))
    if not np.isfinite(probabilities).all():
        raise AssertionError("outer test folds did not produce one prediction per record")
    if not np.all(outer_fold_by_record >= 0):
        raise AssertionError("outer fold assignment did not cover every record")
    return OuterCandidateResult(
        candidate=candidate,
        metrics=binary_group_metrics(dataset.labels, dataset.group_ids, probabilities),
        fold_metrics=tuple(fold_metrics),
        probabilities=probabilities,
        outer_fold_by_record=outer_fold_by_record,
        selected_c_by_fold=tuple(selected_values),
        audit_events=tuple(all_events),
    )


def _run_synthetic_outer_candidate(
    dataset: ClassicalDataset,
    candidate: str,
) -> OuterCandidateResult:
    """Exercise outer mechanics only on explicitly synthetic unit-test data."""
    if (dataset.claim_unit, dataset.identity_status) != (
        "synthetic_group_held_out", "synthetic"
    ):
        raise ValueError("synthetic outer harness cannot evaluate a real cohort")
    if candidate not in CANDIDATE_REGISTRY:
        raise ValueError(f"unknown classical candidate {candidate!r}")
    return _run_outer_candidate_unlocked(
        dataset,
        candidate,
        authorization=_SYNTHETIC_OUTER_AUTHORIZATION,
    )


def paired_stratified_group_bootstrap(
    labels: Sequence[int],
    group_ids: Sequence[object],
    baseline_probabilities: Sequence[float],
    candidate_probabilities: Sequence[float],
    *,
    repeats: int = BOOTSTRAP_REPEATS,
    seed: int = 20260713,
) -> dict[str, object]:
    """Paired class-stratified group bootstrap on fixed pooled OOF predictions."""
    if isinstance(repeats, bool) or not isinstance(repeats, (int, np.integer)) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    base_labels, base_groups, base = group_mean_predictions(
        labels, group_ids, baseline_probabilities
    )
    candidate_labels, candidate_groups, candidate = group_mean_predictions(
        labels, group_ids, candidate_probabilities
    )
    if not np.array_equal(base_labels, candidate_labels) or not np.array_equal(
        base_groups, candidate_groups
    ):
        raise ValueError("paired candidates must contain the same labeled groups")
    if set(base_labels.tolist()) != {0, 1}:
        raise ValueError("bootstrap requires both binary classes")
    point = float(
        roc_auc_score(base_labels, candidate) - roc_auc_score(base_labels, base)
    )
    rng = np.random.default_rng(seed)
    class_indices = {
        label: np.flatnonzero(base_labels == label)
        for label in (0, 1)
    }
    deltas = np.empty(int(repeats), dtype=np.float64)
    for repeat in range(int(repeats)):
        sampled = np.concatenate([
            rng.choice(indices, size=indices.size, replace=True)
            for indices in class_indices.values()
        ])
        sampled_labels = base_labels[sampled]
        deltas[repeat] = (
            roc_auc_score(sampled_labels, candidate[sampled])
            - roc_auc_score(sampled_labels, base[sampled])
        )
    return {
        "delta_auroc": point,
        "ci95": [float(value) for value in np.quantile(deltas, (0.025, 0.975))],
        "probability_delta_gt_zero": float(np.mean(deltas > 0.0)),
        "repeats": int(repeats),
        "seed": int(seed),
        "interval_scope": "fixed_oof_predictions_descriptive",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--candidate", choices=tuple(CANDIDATE_REGISTRY), default="fusion")
    parser.add_argument("--outer-fold", type=int, default=0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    dataset = load_classical_dataset(args.cache_root)
    selection = run_inner_candidate_selection(
        dataset, args.candidate, outer_fold_number=args.outer_fold
    )
    print(json.dumps({
        "mode": "inner-smoke",
        "candidate": selection.candidate,
        "outer_fold": selection.outer_fold_number,
        "selected_c": selection.selected_c,
        "c_scores": selection.c_scores,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
