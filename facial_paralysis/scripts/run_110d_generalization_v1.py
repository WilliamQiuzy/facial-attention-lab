#!/usr/bin/env python3
"""Run the sealed-development comparison for 110D-Generalization v1.

This command has no protected-test path.  It authenticates the finalized
identity review and frozen person split before extracting any candidate row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

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

from scripts.run_dynamic_landmark_classical import (  # noqa: E402
    ClassicalDataset,
    NUISANCE_FEATURE_NAMES,
    _read_json as _read_collection_json,
    _validate_task2_collection_manifest,
    group_mean_predictions,
    group_sample_weights,
)
from scripts.freeze_palsynet_person_split_registry import (  # noqa: E402
    validate_person_split_registry,
)
from scripts.run_mirror_invariant_110d import mirror_dynamic_features  # noqa: E402
from src.datasets.dynamic_landmark import load_dynamic_landmark_recordings  # noqa: E402
from src.preprocessing.generalization_110d import (  # noqa: E402
    CANDIDATE_ORDER,
    CANDIDATE_REGISTRY,
    candidate_feature_names,
    candidate_feature_vector,
)


FIXED_C = 0.01
FIXED_THRESHOLD = 0.5
FIXED_SOLVER = "liblinear"
FIXED_RANDOM_STATE = 0
FIXED_MAX_ITER = 2000
INNER_FOLDS = 4
OUTER_FOLD_NUMBER = 0
BOOTSTRAP_REPEATS = 5000
BOOTSTRAP_SEED = 20260805
DOMAIN_SEPARATOR = "110d-generalization-v1-person-split"
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "outputs" / "dynamic_landmark" / "benchmarks"
    / "development" / "110d-generalization-v1" / "report.json"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REC_ID = re.compile(r"^rec_[0-9a-f]{64}$")
_GROUP_ID = re.compile(r"^grp_[0-9a-f]{64}$")
_METRICS = (
    "auroc", "average_precision", "brier", "balanced_accuracy",
    "sensitivity", "specificity",
)
_PAIR_KEYS = tuple(
    f"{right}_minus_{left}"
    for left_index, left in enumerate(CANDIDATE_ORDER)
    for right in CANDIDATE_ORDER[left_index + 1:]
)
_IMPLEMENTATION_COMPONENT_PATHS = {
    "runner": Path(__file__).resolve(),
    "generalization_features": PROJECT_ROOT / "src/preprocessing/generalization_110d.py",
    "trajectory_features": PROJECT_ROOT / "src/preprocessing/trajectory_features.py",
    "clinical_dynamics": PROJECT_ROOT / "src/preprocessing/clinical_dynamics.py",
    "mirror_runner": PROJECT_ROOT / "scripts/run_mirror_invariant_110d.py",
    "dynamic_landmark_model": PROJECT_ROOT / "src/models/dynamic_landmark.py",
    "dynamic_landmark_loader": PROJECT_ROOT / "src/datasets/dynamic_landmark.py",
    "classical_group_evaluation": PROJECT_ROOT / "scripts/run_dynamic_landmark_classical.py",
    "person_split_registry": PROJECT_ROOT / "scripts/freeze_palsynet_person_split_registry.py",
}


@dataclass
class GateAudit:
    gate_attempts: int = 0
    gate_passes: int = 0
    development_feature_extractions: int = 0
    development_mirror_transforms: int = 0
    development_cache_records_loaded: int = 0
    development_scaler_fits: int = 0
    development_model_fits: int = 0
    development_predictions: int = 0
    protected_feature_extractions: int = 0
    protected_cache_records_loaded: int = 0
    protected_scaler_fits: int = 0
    protected_model_fits: int = 0
    protected_predictions: int = 0

    def as_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass
class DevelopmentGate:
    development_indices: np.ndarray
    protected_indices: np.ndarray
    group_ids: np.ndarray
    inner_fold_by_index: np.ndarray
    reviewed_manifest_sha256: str
    review_ledger_sha256: str
    split_registry_sha256: str
    source_collection_sha256: str


@dataclass
class PreparedCandidates:
    development_indices: np.ndarray
    protected_indices: np.ndarray
    original: dict[str, np.ndarray]
    mirrored: dict[str, np.ndarray]
    remirrored: dict[str, np.ndarray]


@dataclass(frozen=True)
class DevelopmentResult:
    report: dict[str, object]
    probabilities: Mapping[str, np.ndarray]


def canonical_json_sha256(payload: object) -> str:
    encoded = (
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _implementation_fingerprints() -> tuple[dict[str, str], str]:
    """Bind every execution-affecting local implementation component."""
    components: dict[str, str] = {}
    for name, path in _IMPLEMENTATION_COMPONENT_PATHS.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"implementation component {name!r} is unavailable")
        components[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return components, canonical_json_sha256(components)


def _exact_object(value: object, fields: set[str], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} fields differ from the frozen schema")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _semantic_group_key(member_sha256: Sequence[str]) -> str:
    joined = ",".join(sorted(member_sha256))
    return hashlib.sha256(f"{DOMAIN_SEPARATOR}:{joined}".encode("ascii")).hexdigest()


def validate_development_gate(
    dataset: ClassicalDataset,
    reviewed_manifest: Mapping[str, object],
    review_ledger: Mapping[str, object],
    split_registry: Mapping[str, object],
    *,
    reviewed_manifest_sha256: str,
    review_ledger_sha256: str,
    audit: GateAudit,
    split_registry_sha256: str | None = None,
    cache_source_sha256_by_recording_id: Mapping[str, str] | None = None,
    cache_source_collection_sha256: str | None = None,
) -> DevelopmentGate:
    """Authenticate identity and person split without touching feature arrays."""
    if not isinstance(audit, GateAudit):
        raise TypeError("audit must be GateAudit")
    audit.gate_attempts += 1
    validate_person_split_registry(
        split_registry, reviewed_manifest, review_ledger
    )
    manifest_sha = _sha(reviewed_manifest_sha256, "reviewed manifest digest")
    ledger_sha = _sha(review_ledger_sha256, "review ledger digest")
    registry_sha = (
        canonical_json_sha256(split_registry)
        if split_registry_sha256 is None
        else _sha(split_registry_sha256, "split registry digest")
    )
    manifest = _exact_object(reviewed_manifest, {
        "schema_version", "dataset", "claim_unit", "identity_review",
        "counts", "fingerprints", "recordings",
    }, "reviewed identity manifest")
    if (
        manifest["schema_version"] != "palsynet_identity_reviewed_v1"
        or manifest["dataset"] != "PalsyNet"
        or manifest["claim_unit"] != "person_held_out"
    ):
        raise ValueError("identity manifest is not finalized person-held-out PalsyNet")
    identity = _exact_object(manifest["identity_review"], {
        "status", "label_blinded", "exhaustive_pair_review",
        "uncertainties_resolved",
    }, "identity review state")
    if identity != {
        "status": "reviewed", "label_blinded": True,
        "exhaustive_pair_review": True, "uncertainties_resolved": True,
    }:
        raise ValueError("identity review is incomplete")
    fingerprints = _exact_object(manifest["fingerprints"], {
        "source_collection_sha256", "generated_manifest_sha256",
        "contact_inventory_sha256", "review_ledger_sha256",
        "reviewer_evidence_sha256", "cross_label_adjudication_sha256",
    }, "identity fingerprints")
    source_sha = _sha(fingerprints["source_collection_sha256"], "source digest")
    for name, value in fingerprints.items():
        _sha(value, name)
    if fingerprints["review_ledger_sha256"] != ledger_sha:
        raise ValueError("review ledger digest differs from finalized manifest")
    if cache_source_collection_sha256 is not None and (
        _sha(cache_source_collection_sha256, "cache source collection digest")
        != source_sha
    ):
        raise ValueError("feature cache and identity review bind different source bytes")

    if (
        review_ledger.get("schema_version") != "palsynet_identity_review_ledger_v1"
        or review_ledger.get("dataset") != "PalsyNet"
        or review_ledger.get("uncertainty_status") != "resolved"
        or not isinstance(review_ledger.get("recording_to_group"), list)
        or not isinstance(review_ledger.get("pair_decisions"), list)
    ):
        raise ValueError("review ledger is not the resolved structured ledger")

    rows = manifest["recordings"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("reviewed manifest needs recording rows")
    manifest_by_record: dict[str, Mapping[str, object]] = {}
    members_by_group: dict[str, list[str]] = {}
    labels_by_group: dict[str, set[int]] = {}
    eligible_ids: set[str] = set()
    allowed_row_fields = {
        "recording_id", "group_id", "source_sha256", "source_label", "label",
        "identity_status", "claim_unit", "training_eligible",
        "adjudication_outcome", "adjudication_evidence_sha256",
    }
    for value in rows:
        row = _exact_object(value, allowed_row_fields, "reviewed recording")
        recording_id, group_id = row["recording_id"], row["group_id"]
        if (
            not isinstance(recording_id, str) or _REC_ID.fullmatch(recording_id) is None
            or not isinstance(group_id, str) or _GROUP_ID.fullmatch(group_id) is None
            or recording_id in manifest_by_record
            or row["identity_status"] != "reviewed"
            or row["claim_unit"] != "person_held_out"
            or not isinstance(row["training_eligible"], bool)
        ):
            raise ValueError("reviewed recording identity fields are invalid")
        source_member = _sha(row["source_sha256"], "recording source digest")
        label = {"unaffected": 0, "affected": 1}.get(row["label"])
        if label is None:
            raise ValueError("reviewed recording label is invalid")
        manifest_by_record[recording_id] = row
        members_by_group.setdefault(group_id, []).append(source_member)
        if row["training_eligible"]:
            eligible_ids.add(recording_id)
            labels_by_group.setdefault(group_id, set()).add(label)
    if any(len(labels) != 1 for labels in labels_by_group.values()):
        raise ValueError("eligible reviewed group crosses labels without exclusion")

    ledger_mapping: dict[str, str] = {}
    for value in review_ledger["recording_to_group"]:
        if not isinstance(value, Mapping) or set(value) != {"recording_id", "group_id"}:
            raise ValueError("ledger mapping fields are invalid")
        recording_id, group_id = value["recording_id"], value["group_id"]
        if recording_id in ledger_mapping or not isinstance(group_id, str):
            raise ValueError("ledger mapping must cover each recording once")
        ledger_mapping[str(recording_id)] = group_id
    if ledger_mapping != {
        recording_id: str(row["group_id"])
        for recording_id, row in manifest_by_record.items()
    }:
        raise ValueError("review ledger mapping differs from finalized manifest")
    expected_pairs = {
        frozenset((left, right))
        for left_index, left in enumerate(sorted(ledger_mapping))
        for right in sorted(ledger_mapping)[left_index + 1:]
    }
    observed_pairs: set[frozenset[str]] = set()
    for value in review_ledger["pair_decisions"]:
        if not isinstance(value, Mapping) or set(value) != {
            "recording_id_a", "recording_id_b", "decision"
        }:
            raise ValueError("ledger pair-decision fields are invalid")
        first, second = value["recording_id_a"], value["recording_id_b"]
        pair = frozenset((str(first), str(second)))
        if (
            len(pair) != 2 or pair not in expected_pairs or pair in observed_pairs
            or value["decision"] not in {"same", "different"}
        ):
            raise ValueError("ledger pair coverage/decision is invalid")
        expected_decision = (
            "same" if ledger_mapping[str(first)] == ledger_mapping[str(second)]
            else "different"
        )
        if value["decision"] != expected_decision:
            raise ValueError("ledger pair decision differs from final identity groups")
        observed_pairs.add(pair)
    if observed_pairs != expected_pairs:
        raise ValueError("review ledger must cover every unordered recording pair")

    registry = _exact_object(split_registry, {
        "schema_version", "dataset", "claim_unit", "identity_status",
        "source_collection_sha256", "reviewed_manifest_sha256",
        "review_ledger_sha256", "outer_fold_number", "protocol", "counts",
        "assignments",
    }, "person split registry")
    if (
        registry["schema_version"] != "palsynet_person_split_registry_v1"
        or registry["dataset"] != "PalsyNet"
        or registry["claim_unit"] != "person_held_out"
        or registry["identity_status"] != "reviewed"
        or registry["source_collection_sha256"] != source_sha
        or registry["reviewed_manifest_sha256"] != manifest_sha
        or registry["review_ledger_sha256"] != ledger_sha
        or registry["outer_fold_number"] != OUTER_FOLD_NUMBER
    ):
        raise ValueError("person split registry provenance/state mismatch")
    protocol = _exact_object(registry["protocol"], {
        "domain_separator", "outer_folds", "inner_folds",
        "semantic_group_key", "stratification",
    }, "split protocol")
    if protocol != {
        "domain_separator": DOMAIN_SEPARATOR,
        "outer_folds": 5,
        "inner_folds": INNER_FOLDS,
        "semantic_group_key": "sha256(domain_separator + ':' + comma_join(sorted_member_source_sha256))",
        "stratification": "binary_label_then_group_size_then_semantic_key",
    }:
        raise ValueError("person split protocol drifted")
    assignments = registry["assignments"]
    if not isinstance(assignments, list):
        raise ValueError("split assignments must be a list")
    assignment_by_record: dict[str, Mapping[str, object]] = {}
    group_assignment: dict[str, tuple[object, object, object]] = {}
    for value in assignments:
        assignment = _exact_object(value, {
            "recording_id", "group_id", "semantic_group_key_sha256",
            "partition", "outer_fold", "inner_fold",
        }, "split assignment")
        recording_id, group_id = assignment["recording_id"], assignment["group_id"]
        if recording_id in assignment_by_record or recording_id not in eligible_ids:
            raise ValueError("split assignment coverage is duplicate or ineligible")
        row = manifest_by_record[str(recording_id)]
        if group_id != row["group_id"]:
            raise ValueError("split assignment group differs from reviewed identity")
        if assignment["semantic_group_key_sha256"] != _semantic_group_key(
            members_by_group[str(group_id)]
        ):
            raise ValueError("semantic group split key is invalid")
        partition = assignment["partition"]
        outer_fold, inner_fold = assignment["outer_fold"], assignment["inner_fold"]
        if partition == "protected":
            if outer_fold != 0 or inner_fold is not None:
                raise ValueError("protected assignment must be outer fold zero only")
        elif partition == "development":
            if outer_fold not in {1, 2, 3, 4} or inner_fold not in {0, 1, 2, 3}:
                raise ValueError("development assignment fold is invalid")
        else:
            raise ValueError("split partition must be development or protected")
        state = (partition, outer_fold, inner_fold)
        previous = group_assignment.setdefault(str(group_id), state)
        if previous != state:
            raise ValueError("one reviewed group cannot cross split assignments")
        assignment_by_record[str(recording_id)] = assignment
    if set(assignment_by_record) != eligible_ids:
        raise ValueError("split registry must cover every eligible recording exactly once")

    dataset_by_record = {
        recording_id: index for index, recording_id in enumerate(dataset.recording_ids)
    }
    if cache_source_sha256_by_recording_id is None:
        identity_to_dataset_index = {
            recording_id: dataset_by_record[recording_id]
            for recording_id in eligible_ids if recording_id in dataset_by_record
        }
    else:
        if set(cache_source_sha256_by_recording_id) != set(dataset.recording_ids):
            raise ValueError("cache source map must cover every cache recording exactly once")
        cache_index_by_source: dict[str, int] = {}
        for recording_id, member_sha in cache_source_sha256_by_recording_id.items():
            source_member = _sha(member_sha, "cache recording source digest")
            if source_member in cache_index_by_source:
                raise ValueError("cache source digests must be unique")
            cache_index_by_source[source_member] = dataset_by_record[recording_id]
        identity_to_dataset_index = {
            recording_id: cache_index_by_source[str(manifest_by_record[recording_id]["source_sha256"])]
            for recording_id in eligible_ids
            if str(manifest_by_record[recording_id]["source_sha256"]) in cache_index_by_source
        }
    if set(identity_to_dataset_index) != eligible_ids:
        raise ValueError("eligible reviewed source bytes are missing from feature cache")
    development: list[int] = []
    protected: list[int] = []
    folds = np.full(dataset.labels.size, -1, dtype=np.int64)
    reviewed_groups = dataset.group_ids.astype(object, copy=True)
    for recording_id, assignment in assignment_by_record.items():
        index = identity_to_dataset_index[recording_id]
        row = manifest_by_record[recording_id]
        expected_label = 1 if row["label"] == "affected" else 0
        if int(dataset.labels[index]) != expected_label:
            raise ValueError("reviewed final label differs from feature cache label")
        reviewed_groups[index] = str(row["group_id"])
        if assignment["partition"] == "development":
            development.append(index)
            folds[index] = int(assignment["inner_fold"])
        else:
            protected.append(index)
    development_array = np.asarray(sorted(development), dtype=np.int64)
    protected_array = np.asarray(sorted(protected), dtype=np.int64)
    if set(development_array) & set(protected_array):
        raise ValueError("development and protected rows overlap")
    for fold in range(INNER_FOLDS):
        validation = development_array[folds[development_array] == fold]
        training = development_array[folds[development_array] != fold]
        if validation.size == 0 or training.size == 0:
            raise ValueError("all four inner folds must be nonempty")
        if set(dataset.labels[training].tolist()) != {0, 1}:
            raise ValueError("every inner training fold needs both classes")
        if set(reviewed_groups[training]) & set(reviewed_groups[validation]):
            raise ValueError("inner folds are not group disjoint")

    counts = _exact_object(registry["counts"], {
        "eligible_recordings", "eligible_groups", "development_recordings",
        "development_groups", "protected_recordings", "protected_groups",
    }, "split counts")
    expected_counts = {
        "eligible_recordings": len(eligible_ids),
        "eligible_groups": len({manifest_by_record[r]["group_id"] for r in eligible_ids}),
        "development_recordings": int(development_array.size),
        "development_groups": len(set(reviewed_groups[development_array].tolist())),
        "protected_recordings": int(protected_array.size),
        "protected_groups": len(set(reviewed_groups[protected_array].tolist())),
    }
    if counts != expected_counts:
        raise ValueError("split counts differ from assignments")
    manifest_counts = manifest["counts"]
    if (
        not isinstance(manifest_counts, Mapping)
        or manifest_counts.get("eligible_recordings") != len(eligible_ids)
        or manifest_counts.get("eligible_groups") != expected_counts["eligible_groups"]
    ):
        raise ValueError("reviewed identity counts differ from rows")
    audit.gate_passes += 1
    return DevelopmentGate(
        development_indices=development_array,
        protected_indices=protected_array,
        group_ids=reviewed_groups,
        inner_fold_by_index=folds,
        reviewed_manifest_sha256=manifest_sha,
        review_ledger_sha256=ledger_sha,
        split_registry_sha256=registry_sha,
        source_collection_sha256=source_sha,
    )


def _assert_development_indices(gate: DevelopmentGate, indices: np.ndarray, operation: str, audit: GateAudit) -> None:
    values = set(np.asarray(indices, dtype=np.int64).tolist())
    protected = values & set(gate.protected_indices.tolist())
    if protected:
        field = {
            "feature": "protected_feature_extractions",
            "scaler": "protected_scaler_fits",
            "model": "protected_model_fits",
            "predict": "protected_predictions",
        }[operation]
        setattr(audit, field, getattr(audit, field) + len(protected))
        raise ValueError(f"protected row reached {operation} operation")
    if not values.issubset(set(gate.development_indices.tolist())):
        raise ValueError(f"non-development row reached {operation} operation")


def prepare_development_candidates(
    dataset: ClassicalDataset,
    gate: DevelopmentGate,
    *,
    audit: GateAudit,
) -> PreparedCandidates:
    """Extract all three paired views only after a successful gate."""
    if audit.gate_passes != 1:
        raise ValueError("identity/split gate must pass before extraction")
    development = np.asarray(gate.development_indices, dtype=np.int64)
    _assert_development_indices(gate, development, "feature", audit)
    original = {candidate: [] for candidate in CANDIDATE_ORDER}
    mirrored = {candidate: [] for candidate in CANDIDATE_ORDER}
    remirrored = {candidate: [] for candidate in CANDIDATE_ORDER}
    for index in development.tolist():
        raw = dataset.features[index]
        mirrored_raw = mirror_dynamic_features(raw)
        remirrored_raw = mirror_dynamic_features(mirrored_raw)
        audit.development_mirror_transforms += 2
        if not np.array_equal(raw, remirrored_raw):
            raise ValueError("raw mirror transform must be an exact involution")
        temporal = (
            dataset.valid_masks[index], dataset.timestamps[index],
            dataset.source_frame_indices[index],
        )
        for candidate in CANDIDATE_ORDER:
            for target, view in (
                (original, raw), (mirrored, mirrored_raw),
                (remirrored, remirrored_raw),
            ):
                target[candidate].append(candidate_feature_vector(candidate, view, *temporal))
                audit.development_feature_extractions += 1
    converted: list[dict[str, np.ndarray]] = []
    for source in (original, mirrored, remirrored):
        converted.append({
            candidate: np.stack(source[candidate]).astype(np.float64, copy=False)
            for candidate in CANDIDATE_ORDER
        })
    return PreparedCandidates(
        development_indices=development.copy(),
        protected_indices=np.asarray(gate.protected_indices, dtype=np.int64).copy(),
        original=converted[0], mirrored=converted[1], remirrored=converted[2],
    )


def _validate_pairing(dataset: ClassicalDataset, gate: DevelopmentGate, prepared: PreparedCandidates) -> None:
    if (
        not np.array_equal(prepared.development_indices, gate.development_indices)
        or not np.array_equal(prepared.protected_indices, gate.protected_indices)
    ):
        raise ValueError("prepared rows differ from authenticated split")
    for local, index in enumerate(gate.development_indices.tolist()):
        raw = dataset.features[index]
        mirrored_raw = mirror_dynamic_features(raw)
        temporal = (
            dataset.valid_masks[index], dataset.timestamps[index],
            dataset.source_frame_indices[index],
        )
        for candidate, dimension in CANDIDATE_REGISTRY.items():
            matrices = (
                prepared.original[candidate], prepared.mirrored[candidate],
                prepared.remirrored[candidate],
            )
            if any(matrix.shape != (gate.development_indices.size, dimension) for matrix in matrices):
                raise ValueError("candidate matrix dimension/alignment drifted")
            expected_original = candidate_feature_vector(candidate, raw, *temporal)
            expected_mirror = candidate_feature_vector(candidate, mirrored_raw, *temporal)
            if (
                not np.array_equal(matrices[0][local], expected_original)
                or not np.array_equal(matrices[1][local], expected_mirror)
                or not np.array_equal(matrices[2][local], expected_original)
            ):
                raise ValueError(f"{candidate} mirror pairing is stale or reordered")


def _fast_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positive = probabilities[labels == 1]
    negative = probabilities[labels == 0]
    if positive.size == 0 or negative.size == 0:
        raise ValueError("binary metrics require both classes")
    predictions = probabilities >= FIXED_THRESHOLD
    sensitivity = float(np.mean(predictions[labels == 1]))
    specificity = float(np.mean(~predictions[labels == 0]))
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def _paired_bootstrap(
    labels: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    *,
    repeats: int,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("bootstrap repeats must be a positive integer")
    class_indices = {label: np.flatnonzero(labels == label) for label in (0, 1)}
    if any(indices.size == 0 for indices in class_indices.values()):
        raise ValueError("bootstrap needs affected and unaffected groups")
    points = {candidate: _fast_metrics(labels, values) for candidate, values in probabilities.items()}
    distributions = {
        candidate: {metric: np.empty(repeats) for metric in _METRICS}
        for candidate in CANDIDATE_ORDER
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for repeat in range(repeats):
        sampled = np.concatenate([
            rng.choice(indices, size=indices.size, replace=True)
            for indices in class_indices.values()
        ])
        for candidate in CANDIDATE_ORDER:
            metrics = _fast_metrics(labels[sampled], probabilities[candidate][sampled])
            for metric in _METRICS:
                distributions[candidate][metric][repeat] = metrics[metric]
    candidate_output: dict[str, dict[str, object]] = {}
    for candidate in CANDIDATE_ORDER:
        candidate_output[candidate] = {
            metric: {
                "point": points[candidate][metric],
                "ci95": [float(value) for value in np.quantile(
                    distributions[candidate][metric], (0.025, 0.975)
                )],
            }
            for metric in _METRICS
        }
    delta_output: dict[str, dict[str, object]] = {}
    for left_index, left in enumerate(CANDIDATE_ORDER):
        for right in CANDIDATE_ORDER[left_index + 1:]:
            key = f"{right}_minus_{left}"
            delta_output[key] = {}
            for metric in _METRICS:
                delta = distributions[right][metric] - distributions[left][metric]
                delta_output[key][metric] = {
                    "point": points[right][metric] - points[left][metric],
                    "ci95": [float(value) for value in np.quantile(delta, (0.025, 0.975))],
                }
    return candidate_output, delta_output


def _independent_expected_aggregates(
    labels: np.ndarray,
    group_ids: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    *,
    repeats: int,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]], np.ndarray]:
    """Recompute group aggregation, metrics, and paired draws independently."""
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(group_ids)
    if tuple(probabilities) != tuple(CANDIDATE_ORDER):
        raise ValueError("verification probabilities must cover candidates in order")
    ordered_groups = sorted(set(groups.tolist()), key=str)
    grouped_labels: list[int] = []
    grouped_probabilities = {
        candidate: [] for candidate in CANDIDATE_ORDER
    }
    for group in ordered_groups:
        indices = np.flatnonzero(groups == group)
        observed = np.unique(labels[indices])
        if observed.size != 1:
            raise ValueError("verification group crosses labels")
        grouped_labels.append(int(observed[0]))
        for candidate in CANDIDATE_ORDER:
            values = np.asarray(probabilities[candidate], dtype=np.float64)
            if values.shape != labels.shape or not np.isfinite(values).all() or np.any(
                (values < 0.0) | (values > 1.0)
            ):
                raise ValueError("verification probabilities are invalid or unaligned")
            grouped_probabilities[candidate].append(float(np.mean(values[indices])))
    group_labels = np.asarray(grouped_labels, dtype=np.int64)
    group_probabilities = {
        candidate: np.asarray(values, dtype=np.float64)
        for candidate, values in grouped_probabilities.items()
    }

    def metrics_for(values: np.ndarray) -> dict[str, float]:
        predictions = (values >= FIXED_THRESHOLD).astype(np.int64)
        positive = group_labels == 1
        negative = group_labels == 0
        return {
            "auroc": float(roc_auc_score(group_labels, values)),
            "average_precision": float(
                average_precision_score(group_labels, values)
            ),
            "brier": float(brier_score_loss(group_labels, values)),
            "balanced_accuracy": float(
                balanced_accuracy_score(group_labels, predictions)
            ),
            "sensitivity": float(np.mean(predictions[positive] == 1)),
            "specificity": float(np.mean(predictions[negative] == 0)),
        }

    points = {
        candidate: metrics_for(group_probabilities[candidate])
        for candidate in CANDIDATE_ORDER
    }
    distributions = {
        candidate: {metric: np.empty(repeats, dtype=np.float64) for metric in _METRICS}
        for candidate in CANDIDATE_ORDER
    }
    class_indices = {
        label: np.flatnonzero(group_labels == label) for label in (0, 1)
    }
    if any(indices.size == 0 for indices in class_indices.values()):
        raise ValueError("verification bootstrap requires both classes")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for repeat in range(repeats):
        draw = np.concatenate(tuple(
            rng.choice(indices, size=indices.size, replace=True)
            for indices in (class_indices[0], class_indices[1])
        ))
        sampled_labels = group_labels[draw]
        for candidate in CANDIDATE_ORDER:
            values = group_probabilities[candidate][draw]
            predictions = (values >= FIXED_THRESHOLD).astype(np.int64)
            sampled_positive = sampled_labels == 1
            sampled_negative = sampled_labels == 0
            independently_computed = {
                "auroc": float(roc_auc_score(sampled_labels, values)),
                "average_precision": float(
                    average_precision_score(sampled_labels, values)
                ),
                "brier": float(brier_score_loss(sampled_labels, values)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(sampled_labels, predictions)
                ),
                "sensitivity": float(np.mean(
                    predictions[sampled_positive] == 1
                )),
                "specificity": float(np.mean(
                    predictions[sampled_negative] == 0
                )),
            }
            for metric in _METRICS:
                distributions[candidate][metric][repeat] = (
                    independently_computed[metric]
                )
    metric_report = {
        candidate: {
            metric: {
                "point": points[candidate][metric],
                "ci95": [float(value) for value in np.quantile(
                    distributions[candidate][metric], (0.025, 0.975)
                )],
            }
            for metric in _METRICS
        }
        for candidate in CANDIDATE_ORDER
    }
    delta_report: dict[str, dict[str, object]] = {}
    for left_index, left in enumerate(CANDIDATE_ORDER):
        for right in CANDIDATE_ORDER[left_index + 1:]:
            key = f"{right}_minus_{left}"
            delta_report[key] = {}
            for metric in _METRICS:
                distribution = (
                    distributions[right][metric] - distributions[left][metric]
                )
                delta_report[key][metric] = {
                    "point": points[right][metric] - points[left][metric],
                    "ci95": [float(value) for value in np.quantile(
                        distribution, (0.025, 0.975)
                    )],
                }
    return metric_report, delta_report, group_labels


def select_locked_candidate(metrics: Mapping[str, Mapping[str, float]]) -> tuple[str, dict[str, bool]]:
    if tuple(metrics) != tuple(CANDIDATE_ORDER):
        raise ValueError("locking metrics must preserve candidate registry order")
    base, action, phase = (metrics[candidate] for candidate in CANDIDATE_ORDER)
    action_advances = bool(
        action["auroc"] > base["auroc"]
        and action["balanced_accuracy"] >= base["balanced_accuracy"]
        and action["brier"] <= base["brier"]
    )
    phase_advances = bool(all((
        phase["auroc"] > base["auroc"],
        phase["auroc"] > action["auroc"],
        phase["balanced_accuracy"] >= base["balanced_accuracy"],
        phase["balanced_accuracy"] >= action["balanced_accuracy"],
        phase["brier"] <= base["brier"],
        phase["brier"] <= action["brier"],
    )))
    locked = CANDIDATE_ORDER[2] if phase_advances else (
        CANDIDATE_ORDER[1] if action_advances else CANDIDATE_ORDER[0]
    )
    return locked, {
        "action_proxy_advances": action_advances,
        "phase_proxy_advances_against_both_simpler_candidates": phase_advances,
    }


def run_development_comparison(
    dataset: ClassicalDataset,
    gate: DevelopmentGate,
    prepared: PreparedCandidates,
    *,
    audit: GateAudit,
    bootstrap_repeats: int = BOOTSTRAP_REPEATS,
) -> DevelopmentResult:
    """Run aligned four-fold person/group-disjoint OOF on development only."""
    if audit.gate_passes != 1:
        raise ValueError("authenticated gate is required")
    _validate_pairing(dataset, gate, prepared)
    development = gate.development_indices
    local_by_global = {int(index): local for local, index in enumerate(development)}
    probabilities: dict[str, np.ndarray] = {}
    for candidate in CANDIDATE_ORDER:
        oof = np.full(development.size, np.nan)
        counts = np.zeros(development.size, dtype=np.int64)
        for fold in range(INNER_FOLDS):
            validation_global = development[gate.inner_fold_by_index[development] == fold]
            training_global = development[gate.inner_fold_by_index[development] != fold]
            _assert_development_indices(gate, training_global, "scaler", audit)
            _assert_development_indices(gate, training_global, "model", audit)
            _assert_development_indices(gate, validation_global, "predict", audit)
            train_local = np.asarray([local_by_global[int(i)] for i in training_global])
            validation_local = np.asarray([local_by_global[int(i)] for i in validation_global])
            x_train = np.concatenate((
                prepared.original[candidate][train_local],
                prepared.mirrored[candidate][train_local],
            ))
            y_train = np.concatenate((dataset.labels[training_global], dataset.labels[training_global]))
            train_groups = np.concatenate((gate.group_ids[training_global], gate.group_ids[training_global]))
            scaler = StandardScaler().fit(x_train)
            audit.development_scaler_fits += 1
            model = LogisticRegression(
                C=FIXED_C, penalty="l2", solver=FIXED_SOLVER,
                max_iter=FIXED_MAX_ITER, random_state=FIXED_RANDOM_STATE,
            )
            model.fit(
                scaler.transform(x_train), y_train,
                sample_weight=group_sample_weights(train_groups),
            )
            audit.development_model_fits += 1
            original_probability = model.predict_proba(
                scaler.transform(prepared.original[candidate][validation_local])
            )[:, 1]
            mirrored_probability = model.predict_proba(
                scaler.transform(prepared.mirrored[candidate][validation_local])
            )[:, 1]
            folded = 0.5 * (original_probability + mirrored_probability)
            oof[validation_local] = folded
            counts[validation_local] += 1
            audit.development_predictions += validation_local.size
        if not np.isfinite(oof).all() or not np.all(counts == 1):
            raise ValueError("inner OOF must align one probability per development row")
        probabilities[candidate] = oof

    labels = dataset.labels[development]
    groups = gate.group_ids[development]
    grouped_probabilities: dict[str, np.ndarray] = {}
    grouped_labels = None
    grouped_ids = None
    for candidate in CANDIDATE_ORDER:
        candidate_labels, candidate_groups, candidate_probs = group_mean_predictions(
            labels, groups, probabilities[candidate]
        )
        if grouped_labels is None:
            grouped_labels, grouped_ids = candidate_labels, candidate_groups
        elif not np.array_equal(grouped_labels, candidate_labels) or not np.array_equal(grouped_ids, candidate_groups):
            raise ValueError("candidate group rows are not aligned")
        grouped_probabilities[candidate] = candidate_probs
    assert grouped_labels is not None
    metrics, deltas = _paired_bootstrap(
        grouped_labels, grouped_probabilities, repeats=bootstrap_repeats
    )
    point_metrics = {
        candidate: {
            metric: float(metrics[candidate][metric]["point"])
            for metric in _METRICS
        }
        for candidate in CANDIDATE_ORDER
    }
    locked, gates = select_locked_candidate(point_metrics)
    implementation_components, implementation_aggregate = (
        _implementation_fingerprints()
    )
    report: dict[str, object] = {
        "schema_version": "110d_generalization_v1_development_report",
        "claim_scope": "identity_reviewed_palsynet_development_inner_oof_only",
        "target": "binary_affected_vs_unaffected_not_hb_grade",
        "dataset": {
            "name": "PalsyNet", "claim_unit": "person_held_out",
            "identity_status": "reviewed",
        },
        "provenance": {
            "reviewed_identity_manifest_sha256": gate.reviewed_manifest_sha256,
            "review_ledger_sha256": gate.review_ledger_sha256,
            "person_split_registry_sha256": gate.split_registry_sha256,
            "source_collection_sha256": gate.source_collection_sha256,
            "implementation_components_sha256": implementation_components,
            "implementation_aggregate_sha256": implementation_aggregate,
        },
        "protocol": {
            "candidates": list(CANDIDATE_ORDER),
            "candidate_dimensions": dict(CANDIDATE_REGISTRY),
            "candidate_feature_names": {
                candidate: list(candidate_feature_names(candidate))
                for candidate in CANDIDATE_ORDER
            },
            "inner_folds": INNER_FOLDS,
            "model": {
                "type": "standardized_l2_logistic_regression", "c": FIXED_C,
                "penalty": "l2", "solver": FIXED_SOLVER,
                "max_iter": FIXED_MAX_ITER, "random_state": FIXED_RANDOM_STATE,
                "threshold": FIXED_THRESHOLD,
                "sample_weight": "equal_total_weight_per_group",
                "training_augmentation": "original_plus_horizontal_mirror",
                "validation_inference": "mean_original_and_horizontal_mirror_probability",
                "hyperparameter_search": False,
            },
            "bootstrap": {
                "repeats": bootstrap_repeats, "seed": BOOTSTRAP_SEED,
                "paired": True, "unit": "reviewed_group",
                "class_stratified": True, "interval": "percentile_95",
            },
        },
        "counts": {
            "eligible_recordings": int(development.size + gate.protected_indices.size),
            "eligible_groups": len(set(gate.group_ids[np.concatenate((development, gate.protected_indices))].tolist())),
            "development_recordings": int(development.size),
            "development_groups": len(set(groups.tolist())),
            "development_affected_groups": int(np.sum(grouped_labels == 1)),
            "development_unaffected_groups": int(np.sum(grouped_labels == 0)),
            "protected_recordings": int(gate.protected_indices.size),
            "protected_groups": len(set(gate.group_ids[gate.protected_indices].tolist())),
        },
        "metrics": metrics,
        "pairwise_deltas": deltas,
        "audit": audit.as_dict(),
        "decision": {
            "gates": gates, "passed": locked != CANDIDATE_ORDER[0],
            "locked_candidate": locked,
            "outer_evaluation_authorized": False,
            "hb_claim_authorized": False,
            "clinical_validation": False,
            "next_gate": "freeze_candidate_then_explicit_one_shot_outer_authorization",
        },
    }
    _validate_report(report, expected_bootstrap_repeats=bootstrap_repeats)
    _validate_report_against_oof(
        report, dataset, gate, probabilities,
        expected_bootstrap_repeats=bootstrap_repeats,
    )
    return DevelopmentResult(report=report, probabilities=probabilities)


def _validate_report(payload: Mapping[str, object], *, expected_bootstrap_repeats: int = BOOTSTRAP_REPEATS) -> None:
    """Independently reject schema, range, count, leakage, and lock drift."""
    top = _exact_object(payload, {
        "schema_version", "claim_scope", "target", "dataset", "provenance",
        "protocol", "counts", "metrics", "pairwise_deltas", "audit", "decision",
    }, "development report")
    if (
        top["schema_version"] != "110d_generalization_v1_development_report"
        or top["claim_scope"] != "identity_reviewed_palsynet_development_inner_oof_only"
        or top["target"] != "binary_affected_vs_unaffected_not_hb_grade"
    ):
        raise ValueError("report claim/schema drifted")
    forbidden_keys = {
        "recording_id", "recording_ids", "group_id", "group_ids", "labels",
        "probability", "probabilities", "prediction", "predictions", "path",
        "paths", "filename", "filenames", "rows", "records",
    }
    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            if any(key in forbidden_keys for key in value):
                raise ValueError("report contains row-level identifiers/outcomes")
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif isinstance(value, str):
            if _REC_ID.fullmatch(value) or _GROUP_ID.fullmatch(value) or value.startswith("/"):
                raise ValueError("report contains identifier or path")
    visit(payload)
    if top["dataset"] != {
        "name": "PalsyNet", "claim_unit": "person_held_out", "identity_status": "reviewed"
    }:
        raise ValueError("report dataset state drifted")
    provenance = _exact_object(top["provenance"], {
        "reviewed_identity_manifest_sha256", "review_ledger_sha256",
        "person_split_registry_sha256", "source_collection_sha256",
        "implementation_components_sha256", "implementation_aggregate_sha256",
    }, "report provenance")
    for name in (
        "reviewed_identity_manifest_sha256", "review_ledger_sha256",
        "person_split_registry_sha256", "source_collection_sha256",
        "implementation_aggregate_sha256",
    ):
        _sha(provenance[name], name)
    expected_components, expected_aggregate = _implementation_fingerprints()
    if (
        provenance["implementation_components_sha256"] != expected_components
        or provenance["implementation_aggregate_sha256"] != expected_aggregate
    ):
        raise ValueError("report implementation component fingerprints drifted")
    protocol = _exact_object(top["protocol"], {
        "candidates", "candidate_dimensions", "candidate_feature_names",
        "inner_folds", "model", "bootstrap",
    }, "report protocol")
    if (
        protocol["candidates"] != list(CANDIDATE_ORDER)
        or protocol["candidate_dimensions"] != dict(CANDIDATE_REGISTRY)
        or protocol["candidate_feature_names"] != {
            candidate: list(candidate_feature_names(candidate))
            for candidate in CANDIDATE_ORDER
        }
        or protocol["inner_folds"] != INNER_FOLDS
    ):
        raise ValueError("candidate registry/features drifted")
    if protocol["model"] != {
        "type": "standardized_l2_logistic_regression", "c": FIXED_C,
        "penalty": "l2", "solver": FIXED_SOLVER, "max_iter": FIXED_MAX_ITER,
        "random_state": FIXED_RANDOM_STATE, "threshold": FIXED_THRESHOLD,
        "sample_weight": "equal_total_weight_per_group",
        "training_augmentation": "original_plus_horizontal_mirror",
        "validation_inference": "mean_original_and_horizontal_mirror_probability",
        "hyperparameter_search": False,
    }:
        raise ValueError("fixed model protocol drifted")
    if protocol["bootstrap"] != {
        "repeats": expected_bootstrap_repeats, "seed": BOOTSTRAP_SEED,
        "paired": True, "unit": "reviewed_group", "class_stratified": True,
        "interval": "percentile_95",
    }:
        raise ValueError("bootstrap protocol drifted")
    counts = _exact_object(top["counts"], {
        "eligible_recordings", "eligible_groups", "development_recordings",
        "development_groups", "development_affected_groups",
        "development_unaffected_groups", "protected_recordings", "protected_groups",
    }, "report counts")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts.values()):
        raise ValueError("report counts must be nonnegative integers")
    if (
        counts["development_recordings"] + counts["protected_recordings"] != counts["eligible_recordings"]
        or counts["development_affected_groups"] + counts["development_unaffected_groups"] != counts["development_groups"]
        or counts["development_groups"] + counts["protected_groups"] != counts["eligible_groups"]
    ):
        raise ValueError("report counts are incoherent")
    metrics = _exact_object(top["metrics"], set(CANDIDATE_ORDER), "candidate metrics")
    point_metrics: dict[str, dict[str, float]] = {}
    for candidate in CANDIDATE_ORDER:
        metric_map = _exact_object(metrics[candidate], set(_METRICS), "candidate metric map")
        point_metrics[candidate] = {}
        for metric in _METRICS:
            summary = _exact_object(metric_map[metric], {"point", "ci95"}, "metric summary")
            point = summary["point"]
            ci = summary["ci95"]
            if (
                isinstance(point, bool) or not isinstance(point, (int, float))
                or not np.isfinite(point) or not 0.0 <= point <= 1.0
                or not isinstance(ci, list) or len(ci) != 2
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in ci)
                or ci[0] > ci[1]
            ):
                raise ValueError("candidate metric value/range is invalid")
            point_metrics[candidate][metric] = float(point)
    deltas = _exact_object(top["pairwise_deltas"], set(_PAIR_KEYS), "pairwise deltas")
    for key in _PAIR_KEYS:
        left, right = None, None
        for candidate_left in CANDIDATE_ORDER:
            for candidate_right in CANDIDATE_ORDER:
                if key == f"{candidate_right}_minus_{candidate_left}":
                    left, right = candidate_left, candidate_right
        delta_map = _exact_object(deltas[key], set(_METRICS), "delta metric map")
        for metric in _METRICS:
            summary = _exact_object(delta_map[metric], {"point", "ci95"}, "delta summary")
            point, ci = summary["point"], summary["ci95"]
            if (
                left is None or right is None
                or not isinstance(point, (int, float)) or isinstance(point, bool)
                or not np.isfinite(point) or not -1.0 <= point <= 1.0
                or abs(float(point) - (point_metrics[right][metric] - point_metrics[left][metric])) > 1e-12
                or not isinstance(ci, list) or len(ci) != 2
                or any(not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value) or not -1.0 <= value <= 1.0 for value in ci)
                or ci[0] > ci[1]
            ):
                raise ValueError("pairwise metric delta is invalid")
    audit = _exact_object(top["audit"], set(GateAudit.__dataclass_fields__), "report audit")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in audit.values()):
        raise ValueError("audit counters must be nonnegative integers")
    if (
        audit["gate_attempts"] != 1 or audit["gate_passes"] != 1
        or audit["development_cache_records_loaded"] != counts["development_recordings"]
        or audit["development_feature_extractions"] != 9 * counts["development_recordings"]
        or audit["development_mirror_transforms"] != 2 * counts["development_recordings"]
        or audit["development_scaler_fits"] != 3 * INNER_FOLDS
        or audit["development_model_fits"] != 3 * INNER_FOLDS
        or audit["development_predictions"] != 3 * counts["development_recordings"]
        or any(audit[name] != 0 for name in (
            "protected_feature_extractions", "protected_cache_records_loaded",
            "protected_scaler_fits",
            "protected_model_fits", "protected_predictions",
        ))
    ):
        raise ValueError("audit counters do not prove one clean development run")
    locked, gates = select_locked_candidate(point_metrics)
    decision = _exact_object(top["decision"], {
        "gates", "passed", "locked_candidate", "outer_evaluation_authorized",
        "hb_claim_authorized", "clinical_validation", "next_gate",
    }, "report decision")
    if (
        decision["gates"] != gates or decision["passed"] is not (locked != CANDIDATE_ORDER[0])
        or decision["locked_candidate"] != locked
        or decision["outer_evaluation_authorized"] is not False
        or decision["hb_claim_authorized"] is not False
        or decision["clinical_validation"] is not False
        or decision["next_gate"] != "freeze_candidate_then_explicit_one_shot_outer_authorization"
    ):
        raise ValueError("candidate lock/claim decision is invalid")


def _validate_report_against_oof(
    payload: Mapping[str, object],
    dataset: ClassicalDataset,
    gate: DevelopmentGate,
    probabilities: Mapping[str, np.ndarray],
    *,
    expected_bootstrap_repeats: int = BOOTSTRAP_REPEATS,
) -> None:
    """Rebuild every published result from aligned OOF rows before writing."""
    _validate_report(
        payload, expected_bootstrap_repeats=expected_bootstrap_repeats
    )
    development = np.asarray(gate.development_indices, dtype=np.int64)
    protected = np.asarray(gate.protected_indices, dtype=np.int64)
    if set(development.tolist()) & set(protected.tolist()):
        raise ValueError("verification split overlaps protected rows")
    expected_metrics, expected_deltas, grouped_labels = (
        _independent_expected_aggregates(
            dataset.labels[development], gate.group_ids[development],
            probabilities, repeats=expected_bootstrap_repeats,
        )
    )
    if payload["metrics"] != expected_metrics:
        raise ValueError("published metrics/intervals differ from independent OOF recomputation")
    if payload["pairwise_deltas"] != expected_deltas:
        raise ValueError("published paired deltas differ from independent resample plan")
    development_groups = gate.group_ids[development]
    protected_groups = gate.group_ids[protected]
    eligible_indices = np.concatenate((development, protected))
    expected_counts = {
        "eligible_recordings": int(eligible_indices.size),
        "eligible_groups": len(set(gate.group_ids[eligible_indices].tolist())),
        "development_recordings": int(development.size),
        "development_groups": len(set(development_groups.tolist())),
        "development_affected_groups": int(np.sum(grouped_labels == 1)),
        "development_unaffected_groups": int(np.sum(grouped_labels == 0)),
        "protected_recordings": int(protected.size),
        "protected_groups": len(set(protected_groups.tolist())),
    }
    if payload["counts"] != expected_counts:
        raise ValueError("published counts differ from authenticated split/OOF rows")
    point_metrics = {
        candidate: {
            metric: float(expected_metrics[candidate][metric]["point"])
            for metric in _METRICS
        }
        for candidate in CANDIDATE_ORDER
    }
    locked, gates = select_locked_candidate(point_metrics)
    decision = payload["decision"]
    if (
        decision["gates"] != gates
        or decision["locked_candidate"] != locked
        or decision["passed"] is not (locked != CANDIDATE_ORDER[0])
    ):
        raise ValueError("published lock differs from independently recomputed point metrics")


def _read_json(path: Path) -> tuple[dict[str, object], str]:
    raw = path.read_bytes()
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError("JSON artifact contains duplicate keys")
            output[key] = value
        return output
    payload = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(payload, dict):
        raise ValueError("JSON artifact root must be an object")
    return payload, hashlib.sha256(raw).hexdigest()


def _build_cache_metadata_dataset(
    cache_root: Path,
) -> tuple[ClassicalDataset, dict[str, Mapping[str, object]], str]:
    """Validate collection metadata without opening any recording NPZ."""
    root = Path(cache_root)
    if not root.is_dir():
        raise ValueError("cache root must be a directory")
    payload, raw_manifest = _read_collection_json(root / "collection_manifest.json")
    rows, claim_unit, identity_status = _validate_task2_collection_manifest(payload)
    expected_paths = {root / f"{row['recording_id']}.npz" for row in rows}
    if set(root.glob("*.npz")) != expected_paths:
        raise ValueError("cache NPZ names differ from the validated collection manifest")
    count = len(rows)
    timestamps = np.tile(
        np.stack([window * 10.0 + np.arange(32) / 30.0 for window in range(4)]),
        (count, 1, 1),
    )
    source_indices = np.tile(
        np.stack([window * 100 + np.arange(32) for window in range(4)]),
        (count, 1, 1),
    ).astype(np.int64)
    labels = np.asarray([
        1 if row["label"] == "affected" else 0 for row in rows
    ], dtype=np.int64)
    nuisance = np.asarray([
        [float(row["nuisance"][name]) for name in NUISANCE_FEATURE_NAMES]
        for row in rows
    ], dtype=np.float64)
    dataset = ClassicalDataset(
        features=np.zeros((count, 4, 32, 95), dtype=np.float32),
        valid_masks=np.ones((count, 4, 32), dtype=bool),
        timestamps=timestamps,
        source_frame_indices=source_indices,
        nuisance=nuisance,
        labels=labels,
        group_ids=np.asarray([row["group_id"] for row in rows]),
        recording_ids=tuple(str(row["recording_id"]) for row in rows),
        claim_unit=claim_unit,
        identity_status=identity_status,
        collection_manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
    )
    source_collection_sha256 = _sha(
        payload["provenance"]["source_collection_sha256"],
        "cache source collection digest",
    )
    return (
        dataset,
        {str(row["recording_id"]): row for row in rows},
        source_collection_sha256,
    )


def _load_one_dynamic_record(path: Path):
    records = load_dynamic_landmark_recordings([path])
    if len(records) != 1:
        raise ValueError("one development cache path must load exactly one record")
    return records[0]


def load_development_cache_records(
    cache_root: Path,
    dataset: ClassicalDataset,
    gate: DevelopmentGate,
    collection_rows: Mapping[str, Mapping[str, object]],
    *,
    audit: GateAudit,
    record_loader: Callable[[Path], object] = _load_one_dynamic_record,
) -> None:
    """Open only authenticated development NPZs; protected paths are forbidden."""
    if audit.gate_passes != 1:
        raise ValueError("identity/split gate must pass before cache loading")
    protected_ids = {
        dataset.recording_ids[int(index)] for index in gate.protected_indices
    }
    for index in gate.development_indices.tolist():
        recording_id = dataset.recording_ids[index]
        if recording_id in protected_ids:
            audit.protected_cache_records_loaded += 1
            raise ValueError("protected cache path reached development loader")
        path = Path(cache_root) / f"{recording_id}.npz"
        record = record_loader(path)
        audit.development_cache_records_loaded += 1
        row = collection_rows.get(recording_id)
        if row is None:
            raise ValueError("development cache record is absent from collection manifest")
        expected_label = 1 if row["label"] == "affected" else 0
        if (
            getattr(record, "recording_id", None) != recording_id
            or getattr(record, "group_id", None) != row["group_id"]
            or getattr(record, "source_sha256", None) != row["source_sha256"]
            or getattr(record, "label", None) != expected_label
        ):
            raise ValueError("development NPZ provenance differs from collection manifest")
        dataset.features[index] = np.asarray(record.features)
        dataset.valid_masks[index] = np.asarray(record.valid_mask)
        dataset.timestamps[index] = np.asarray(record.timestamps)
        dataset.source_frame_indices[index] = np.asarray(record.source_frame_indices)
    if audit.protected_cache_records_loaded != 0:
        raise AssertionError("protected cache load audit must remain zero")


def _write_private_report(
    path: Path,
    payload: Mapping[str, object],
    *,
    dataset: ClassicalDataset,
    gate: DevelopmentGate,
    probabilities: Mapping[str, np.ndarray],
) -> None:
    _validate_report_against_oof(payload, dataset, gate, probabilities)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite development report")
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd, temporary_name = tempfile.mkstemp(prefix=".report.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--palsynet-cache-root", required=True, type=Path)
    parser.add_argument("--reviewed-identity-manifest", required=True, type=Path)
    parser.add_argument("--review-ledger", required=True, type=Path)
    parser.add_argument("--split-registry", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest, manifest_sha = _read_json(args.reviewed_identity_manifest)
    ledger, ledger_sha = _read_json(args.review_ledger)
    registry, registry_sha = _read_json(args.split_registry)
    dataset, collection_rows, cache_source_collection_sha256 = (
        _build_cache_metadata_dataset(args.palsynet_cache_root)
    )
    audit = GateAudit()
    gate = validate_development_gate(
        dataset, manifest, ledger, registry,
        reviewed_manifest_sha256=manifest_sha,
        review_ledger_sha256=ledger_sha,
        split_registry_sha256=registry_sha,
        cache_source_sha256_by_recording_id={
            recording_id: str(row["source_sha256"])
            for recording_id, row in collection_rows.items()
        },
        cache_source_collection_sha256=cache_source_collection_sha256,
        audit=audit,
    )
    load_development_cache_records(
        args.palsynet_cache_root, dataset, gate, collection_rows, audit=audit
    )
    prepared = prepare_development_candidates(dataset, gate, audit=audit)
    result = run_development_comparison(dataset, gate, prepared, audit=audit)
    _write_private_report(
        DEFAULT_REPORT_PATH, result.report, dataset=dataset, gate=gate,
        probabilities=result.probabilities,
    )
    print(json.dumps(result.report, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
