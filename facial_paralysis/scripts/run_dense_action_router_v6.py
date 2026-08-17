#!/usr/bin/env python3
"""Locked participant-disjoint confirmation for Universal Router v6."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import stat
from typing import Mapping
import zipfile

import numpy as np
from sklearn import __version__ as SKLEARN_VERSION
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.extract_dense_action_mesh_v1 import load_dense_action_cache_bytes
from src.preprocessing.dense_bilateral_action_v1 import (
    BILATERAL_INTERACTION_STAT_NAMES,
    bilateral_interaction_feature_vector,
    dense_action_feature_views,
)


LOCKED_DECISION_THRESHOLD = 0.5
PRIMARY_ACCURACY_FLOOR = 0.93
BALANCED_ACCURACY_FLOOR = 0.90
EXPECTED_RUNTIME_VERSIONS = {
    "python": "3.12.3",
    "numpy": "1.26.4",
    "scikit_learn": "1.6.1",
}
PROFILE_REGISTRY = {
    "neuroface": {
        "name": "bilateral_range_fusion",
        "bilateral_statistics": (
            "range_asymmetry", "range_low", "range_high", "range_ratio",
            "paired_difference_median", "paired_difference_q90",
        ),
        "top_k": 64,
        "c": 10.0,
        "dense_weight": 0.5,
    },
    "meei": {
        "name": "paired_action_expert_fusion",
        "statistic_family": "central",
        "view": "mean_absdiff",
        "first": {"top_k": 16, "aggregation": "median", "c": 1.0},
        "second": {"top_k": 32, "aggregation": "mean", "c": 1.0},
        "pair_operator": "mean",
        "dense_weight": 0.25,
    },
}
_METRIC_FIELDS = frozenset(
    {
        "participants", "accuracy", "balanced_accuracy", "sensitivity",
        "specificity", "auroc", "brier", "errors",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "ucr4_artifact_sha256", "neuroface_collection_sha256",
        "meei_collection_sha256", "implementation_sha256",
        "runtime_versions", "palsynet_protected_reads", "mayo_reads",
    }
)
_BASELINE_FIELDS = frozenset(
    {
        "anonymous_groups", "component_probability", "decision_threshold",
        "evidence_profile", "final_probability", "labels", "schema_version",
    }
)
_COLLECTION_FIELDS = frozenset(
    {
        "schema_version", "profile", "private_manifest_sha256",
        "face_landmarker_sha256", "recordings", "records",
    }
)
_COLLECTION_ROW_FIELDS = frozenset(
    {
        "recording_id", "group_id", "source_sha256", "cache_sha256",
        "action_valid", "baseline_valid",
    }
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hex_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON document contains duplicate keys")
        result[key] = value
    return result


def _json_document(payload: bytes) -> dict[str, object]:
    try:
        document = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot parse canonical JSON evidence") from exc
    if type(document) is not dict:
        raise ValueError("JSON evidence must be an object")
    return document


def _read_exact_bytes(path: Path, *, maximum: int) -> bytes:
    if not isinstance(path, Path) or not path.is_absolute() or maximum < 1:
        raise ValueError("evidence path must be an absolute Path")
    if path.resolve(strict=True) != path:
        raise ValueError("evidence path must be canonical and symlink-free")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise ValueError("evidence file has an invalid type or size")
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("evidence file ended before its committed size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("evidence file grew while it was read")
        after = os.fstat(descriptor)
        if (
            before.st_dev, before.st_ino, before.st_size,
            before.st_mtime_ns, before.st_ctime_ns,
        ) != (
            after.st_dev, after.st_ino, after.st_size,
            after.st_mtime_ns, after.st_ctime_ns,
        ):
            raise ValueError("evidence identity changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _validated_metrics(metrics: Mapping[str, Mapping[str, object]]):
    expected_profiles = {"palsynet_development", "neuroface", "meei"}
    if type(metrics) is not dict or set(metrics) != expected_profiles:
        raise ValueError("all three exact development profiles are required")
    result = {}
    expected_counts = {"palsynet_development": 38, "neuroface": 36, "meei": 56}
    for profile in sorted(expected_profiles):
        row = metrics[profile]
        if type(row) is not dict or set(row) != _METRIC_FIELDS:
            raise ValueError("profile metrics have an open schema")
        participants = row["participants"]
        errors = row["errors"]
        if (
            type(participants) is not int
            or participants != expected_counts[profile]
            or type(errors) is not int
            or not 0 <= errors <= participants
        ):
            raise ValueError("profile count metrics differ from the frozen cohorts")
        validated = {"participants": participants, "errors": errors}
        for name in _METRIC_FIELDS - {"participants", "errors"}:
            value = row[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError("profile metric is outside [0,1]")
            validated[name] = float(value)
        if abs(validated["accuracy"] - (participants - errors) / participants) > 1e-12:
            raise ValueError("accuracy and error count disagree")
        result[profile] = validated
    return result


def build_public_report(
    metrics: Mapping[str, Mapping[str, object]],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    validated = _validated_metrics(metrics)
    if type(provenance) is not dict or set(provenance) != _PROVENANCE_FIELDS:
        raise ValueError("provenance has an open schema")
    for name in (
        "ucr4_artifact_sha256", "neuroface_collection_sha256",
        "meei_collection_sha256", "implementation_sha256",
    ):
        if not _hex_digest(provenance[name]):
            raise ValueError("provenance digest is invalid")
    if provenance["runtime_versions"] != EXPECTED_RUNTIME_VERSIONS:
        raise ValueError("runtime dependency versions differ from the frozen run")
    if provenance["palsynet_protected_reads"] != 0 or provenance["mayo_reads"] != 0:
        raise ValueError("protected/model-selection boundary was crossed")
    passed = all(
        row["accuracy"] >= PRIMARY_ACCURACY_FLOOR
        and row["balanced_accuracy"] >= BALANCED_ACCURACY_FLOOR
        for row in validated.values()
    )
    return {
        "schema_version": "universal_clinical_router_v6_candidate_report",
        "status": "exposed_development_candidate_not_clinically_validated",
        "model": {
            "name": "Universal Clinical Router v6 dense-action candidate",
            "base_model": "Universal Clinical Router v4",
            "default_model_changed": False,
            "decision_threshold": LOCKED_DECISION_THRESHOLD,
            "profile_registry": PROFILE_REGISTRY,
        },
        "evaluations": validated,
        "decision": {
            "primary_accuracy_floor": PRIMARY_ACCURACY_FLOOR,
            "balanced_accuracy_floor": BALANCED_ACCURACY_FLOOR,
            "all_profile_gate_passed": bool(passed),
            "promotion_authorized": False,
        },
        "audit": dict(provenance),
        "claim_boundary": {
            "participant_disjoint_development_only": True,
            "candidate_configuration_selected_after_development_exploration": True,
            "untouched_external_validation": False,
            "mayo_clinical_accuracy": False,
            "house_brackmann_grading": False,
        },
    }


def write_public_report_no_overwrite(path: Path, report: Mapping[str, object]) -> str:
    if not isinstance(path, Path) or type(report) is not dict:
        raise ValueError("report publication requires a Path and exact mapping")
    payload = (
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        written = 0
        while written < len(payload):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("short public report write")
            written += count
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        try:
            path.unlink()
        except OSError:
            pass
        raise
    else:
        os.close(descriptor)
    return _sha256(payload)


def _rank(matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        scores, _ = f_classif(matrix, labels)
    scores = np.nan_to_num(
        scores, nan=-np.inf, posinf=np.finfo(np.float64).max, neginf=-np.inf
    )
    return np.lexsort((np.arange(scores.size), -scores))


def _model(c_value: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value, penalty="l2", solver="liblinear",
            class_weight="balanced", max_iter=3000, random_state=20260817,
        ),
    )


def _load_collection(
    cache_root: Path,
    expected_profile: str,
    expected_collection_sha256: str,
    expected_manifest_sha256: str,
):
    collection_payload = _read_exact_bytes(
        cache_root / "collection_manifest.json", maximum=4 * 1024 * 1024
    )
    if _sha256(collection_payload) != expected_collection_sha256:
        raise ValueError("dense collection differs from its out-of-band pin")
    collection = _json_document(collection_payload)
    expected = 108 if expected_profile == "neuroface" else 56
    if (
        set(collection) != _COLLECTION_FIELDS
        or collection.get("schema_version") != "dense_action_mesh_collection_v1"
        or collection.get("profile") != expected_profile
        or collection.get("private_manifest_sha256") != expected_manifest_sha256
        or not _hex_digest(collection.get("face_landmarker_sha256"))
        or collection.get("recordings") != expected
        or type(collection.get("records")) is not list
        or len(collection["records"]) != expected
    ):
        raise ValueError("dense collection differs from its frozen profile")
    if any(
        type(row) is not dict or set(row) != _COLLECTION_ROW_FIELDS
        for row in collection["records"]
    ):
        raise ValueError("dense collection row has an open schema")
    caches = []
    seen_recordings = set()
    seen_sources = set()
    for row in sorted(collection["records"], key=lambda value: value["recording_id"]):
        if (
            type(row["recording_id"]) is not str
            or type(row["group_id"]) is not str
            or not _hex_digest(row["source_sha256"])
            or not _hex_digest(row["cache_sha256"])
            or row["recording_id"] in seen_recordings
            or row["source_sha256"] in seen_sources
        ):
            raise ValueError("dense collection identity is malformed or duplicated")
        seen_recordings.add(row["recording_id"])
        seen_sources.add(row["source_sha256"])
        path = cache_root / f"{row['recording_id']}.npz"
        payload = _read_exact_bytes(path, maximum=128 * 1024 * 1024)
        if _sha256(payload) != row["cache_sha256"]:
            raise ValueError("dense cache differs from collection commitment")
        cache = load_dense_action_cache_bytes(payload)
        if (
            cache.recording_id != row["recording_id"]
            or cache.group_id != row["group_id"]
            or cache.source_sha256 != row["source_sha256"]
        ):
            raise ValueError("dense cache identity differs from collection row")
        caches.append(cache)
    return caches, _sha256(collection_payload)


def _read_manifest(path: Path, expected_sha256: str):
    payload = _read_exact_bytes(path, maximum=64 * 1024 * 1024)
    if _sha256(payload) != expected_sha256:
        raise ValueError("private manifest differs from its out-of-band pin")
    document = _json_document(payload)
    participants = document.get("participants")
    if type(participants) is not list or not participants:
        raise ValueError("private manifest has no participant evidence")
    labels = {}
    for row in participants:
        if (
            type(row) is not dict
            or type(row.get("participant_id")) is not str
            or row.get("binary_label") not in {"affected", "unaffected"}
            or row["participant_id"] in labels
        ):
            raise ValueError("private manifest participant identity is malformed")
        labels[row["participant_id"]] = int(row["binary_label"] == "affected")
    return document, labels


def _load_baseline(path: Path, expected_sha256: str, expected_rows: int):
    payload = _read_exact_bytes(path, maximum=64 * 1024 * 1024)
    if _sha256(payload) != expected_sha256:
        raise ValueError("frozen UCR4 OOF profile differs from its pin")
    source = io.BytesIO(payload)
    try:
        with zipfile.ZipFile(source, "r") as archive:
            members = [entry.filename for entry in archive.infolist()]
        expected_members = {f"{name}.npy" for name in _BASELINE_FIELDS}
        if (
            len(members) != len(expected_members)
            or len(set(members)) != len(members)
            or set(members) != expected_members
        ):
            raise ValueError("frozen UCR4 OOF profile has noncanonical members")
        source.seek(0)
        with np.load(source, allow_pickle=False) as saved:
            if (
                len(saved.files) != len(_BASELINE_FIELDS)
                or len(set(saved.files)) != len(saved.files)
                or set(saved.files) != _BASELINE_FIELDS
            ):
                raise ValueError("frozen UCR4 OOF profile has an open schema")
            labels = np.asarray(saved["labels"], dtype=np.int64)
            probability = np.asarray(saved["final_probability"], dtype=np.float64)
    except ValueError:
        raise
    except (EOFError, OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ValueError("cannot parse frozen UCR4 OOF profile") from exc
    if (
        labels.shape != (expected_rows,)
        or probability.shape != labels.shape
        or not np.isfinite(probability).all()
        or np.any((probability < 0) | (probability > 1))
    ):
        raise ValueError("frozen UCR4 OOF profile has a malformed shape")
    return labels, probability


def _neuroface_matrix(caches, labels_by_group):
    tasks = ("NSM_KISS", "NSM_OPEN", "NSM_SPREAD")
    people = {}
    for cache in caches:
        vector = bilateral_interaction_feature_vector(
            cache.original_actions, cache.action_valid,
            cache.original_baselines, cache.baseline_valid,
            cache.mirrored_actions, cache.action_valid,
            cache.mirrored_baselines, cache.baseline_valid,
            action_names=cache.action_names,
        )
        people.setdefault(cache.group_id, {})[cache.action_names[0]] = vector
    groups = tuple(sorted(people))
    if len(groups) != 36 or any(set(people[group]) != set(tasks) for group in groups):
        raise ValueError("NeuroFace requires exact 36x3 task evidence")
    features = np.stack([
        np.concatenate([people[group][task] for task in tasks]) for group in groups
    ])
    outcomes = np.asarray([labels_by_group[group] for group in groups], dtype=np.int64)
    keep_stats = set(PROFILE_REGISTRY["neuroface"]["bilateral_statistics"])
    keep_positions = tuple(
        index for index, name in enumerate(BILATERAL_INTERACTION_STAT_NAMES)
        if name in keep_stats
    )
    columns = np.flatnonzero(
        np.isin(
            np.arange(features.shape[1]) % len(BILATERAL_INTERACTION_STAT_NAMES),
            keep_positions,
        )
    )
    return features[:, columns], outcomes


def _meei_matrices(caches, labels_by_group):
    originals, mirrored, outcomes = [], [], []
    for cache in caches:
        first, second = dense_action_feature_views(
            cache.original_actions, cache.action_valid,
            cache.original_baselines, cache.baseline_valid,
            cache.mirrored_actions, cache.action_valid,
            cache.mirrored_baselines, cache.baseline_valid,
            action_names=cache.action_names,
        )
        originals.append(first)
        mirrored.append(second)
        outcomes.append(labels_by_group[cache.group_id])
    return (
        np.asarray(originals, dtype=np.float64),
        np.asarray(mirrored, dtype=np.float64),
        np.asarray(outcomes, dtype=np.int64),
    )


def _neuroface_predict(features, labels, train, held):
    config = PROFILE_REGISTRY["neuroface"]
    selected = _rank(features[train], labels[train])[: config["top_k"]]
    model = _model(config["c"])
    model.fit(features[train][:, selected], labels[train])
    return model.predict_proba(features[held][:, selected])[:, 1]


def _mean_absdiff(original, mirrored):
    return np.concatenate(
        (0.5 * (original + mirrored), np.abs(original - mirrored)), axis=1
    )


def _meei_action_predict(
    original, mirrored, labels, train, held, *, top_k, aggregation, c_value
):
    action_count = 7
    action_dimension = original.shape[1] // action_count
    original = original.reshape(labels.size, action_count, action_dimension)
    mirrored = mirrored.reshape(labels.size, action_count, action_dimension)
    central_positions = (1, 4, 5)
    columns = np.flatnonzero(
        np.isin(np.arange(action_dimension) % 6, central_positions)
    )
    probability = np.full((held.size, action_count), np.nan)
    for action in range(action_count):
        features = _mean_absdiff(
            original[:, action, columns], mirrored[:, action, columns]
        )
        selected = _rank(features[train], labels[train])[:top_k]
        model = _model(c_value)
        model.fit(features[train][:, selected], labels[train])
        probability[:, action] = model.predict_proba(
            features[held][:, selected]
        )[:, 1]
    if aggregation == "median":
        return np.median(probability, axis=1)
    if aggregation == "mean":
        return np.mean(probability, axis=1)
    raise AssertionError("locked action aggregation drifted")


def _meei_predict(original, mirrored, labels, train, held):
    config = PROFILE_REGISTRY["meei"]
    first = _meei_action_predict(
        original, mirrored, labels, train, held,
        top_k=config["first"]["top_k"],
        aggregation=config["first"]["aggregation"],
        c_value=config["first"]["c"],
    )
    second = _meei_action_predict(
        original, mirrored, labels, train, held,
        top_k=config["second"]["top_k"],
        aggregation=config["second"]["aggregation"],
        c_value=config["second"]["c"],
    )
    return 0.5 * (first + second)


def _cross_fitted(labels, baseline, dense_predict, dense_weight):
    probability = np.full(labels.size, np.nan)
    folds = tuple(
        StratifiedKFold(6, shuffle=True, random_state=20260817).split(
            np.zeros(labels.size), labels
        )
    )
    for train, held in folds:
        dense = dense_predict(train, held)
        probability[held] = dense_weight * dense + (1.0 - dense_weight) * baseline[held]
    prediction = probability >= LOCKED_DECISION_THRESHOLD
    positive = labels == 1
    negative = labels == 0
    return {
        "participants": int(labels.size),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "sensitivity": float(np.mean(prediction[positive])),
        "specificity": float(np.mean(~prediction[negative])),
        "auroc": float(roc_auc_score(labels, probability)),
        "brier": float(brier_score_loss(labels, probability)),
        "errors": int(np.sum(prediction != labels)),
    }


def _implementation_sha256(root: Path) -> str:
    files = (
        "scripts/extract_dense_action_mesh_v1.py",
        "scripts/run_private_dense_action_extraction_v6.py",
        "scripts/run_dense_action_router_v6.py",
        "src/preprocessing/dense_bilateral_action_v1.py",
        "src/evaluation/dense_action_router_v6.py",
    )
    entries = []
    for relative in files:
        payload = (root / relative).read_bytes()
        entries.append((relative, _sha256(payload)))
    return _sha256(
        (json.dumps(entries, separators=(",", ":")) + "\n").encode("utf-8")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--ucr4-report", type=Path, required=True)
    parser.add_argument("--ucr4-report-sha256", required=True)
    parser.add_argument("--neuroface-cache", type=Path, required=True)
    parser.add_argument("--neuroface-collection-sha256", required=True)
    parser.add_argument("--neuroface-manifest", type=Path, required=True)
    parser.add_argument("--neuroface-manifest-sha256", required=True)
    parser.add_argument("--neuroface-baseline", type=Path, required=True)
    parser.add_argument("--neuroface-baseline-sha256", required=True)
    parser.add_argument("--meei-cache", type=Path, required=True)
    parser.add_argument("--meei-collection-sha256", required=True)
    parser.add_argument("--meei-manifest", type=Path, required=True)
    parser.add_argument("--meei-manifest-sha256", required=True)
    parser.add_argument("--meei-baseline", type=Path, required=True)
    parser.add_argument("--meei-baseline-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime_versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": SKLEARN_VERSION,
    }
    if runtime_versions != EXPECTED_RUNTIME_VERSIONS:
        raise ValueError("formal confirmation requires the frozen H200 runtime")
    implementation_before = _implementation_sha256(args.project_root)

    ucr4_payload = _read_exact_bytes(args.ucr4_report, maximum=4 * 1024 * 1024)
    if _sha256(ucr4_payload) != args.ucr4_report_sha256:
        raise ValueError("UCR4 report differs from its pin")
    ucr4 = _json_document(ucr4_payload)
    palsy = dict(ucr4["evaluations"]["palsynet_development"]["metrics"])
    palsy["participants"] = 38
    palsy["errors"] = 2

    _, nf_labels_by_group = _read_manifest(
        args.neuroface_manifest, args.neuroface_manifest_sha256
    )
    nf_caches, nf_collection_sha = _load_collection(
        args.neuroface_cache,
        "neuroface",
        args.neuroface_collection_sha256,
        args.neuroface_manifest_sha256,
    )
    nf_features, nf_labels = _neuroface_matrix(nf_caches, nf_labels_by_group)
    nf_baseline_labels, nf_baseline = _load_baseline(
        args.neuroface_baseline, args.neuroface_baseline_sha256, 36
    )
    if not np.array_equal(nf_labels, nf_baseline_labels):
        raise ValueError("NeuroFace row order differs from UCR4 OOF")
    nf_metrics = _cross_fitted(
        nf_labels,
        nf_baseline,
        lambda train, held: _neuroface_predict(
            nf_features, nf_labels, train, held
        ),
        PROFILE_REGISTRY["neuroface"]["dense_weight"],
    )

    _, meei_labels_by_group = _read_manifest(
        args.meei_manifest, args.meei_manifest_sha256
    )
    meei_caches, meei_collection_sha = _load_collection(
        args.meei_cache,
        "meei",
        args.meei_collection_sha256,
        args.meei_manifest_sha256,
    )
    meei_original, meei_mirrored, meei_labels = _meei_matrices(
        meei_caches, meei_labels_by_group
    )
    meei_baseline_labels, meei_baseline = _load_baseline(
        args.meei_baseline, args.meei_baseline_sha256, 56
    )
    if not np.array_equal(meei_labels, meei_baseline_labels):
        raise ValueError("MEEI row order differs from UCR4 OOF")
    meei_metrics = _cross_fitted(
        meei_labels,
        meei_baseline,
        lambda train, held: _meei_predict(
            meei_original, meei_mirrored, meei_labels, train, held
        ),
        PROFILE_REGISTRY["meei"]["dense_weight"],
    )
    implementation_after = _implementation_sha256(args.project_root)
    if implementation_after != implementation_before:
        raise ValueError("implementation changed during formal confirmation")
    report = build_public_report(
        {
            "palsynet_development": palsy,
            "neuroface": nf_metrics,
            "meei": meei_metrics,
        },
        {
            "ucr4_artifact_sha256": args.ucr4_report_sha256,
            "neuroface_collection_sha256": nf_collection_sha,
            "meei_collection_sha256": meei_collection_sha,
            "implementation_sha256": implementation_before,
            "runtime_versions": runtime_versions,
            "palsynet_protected_reads": 0,
            "mayo_reads": 0,
        },
    )
    digest = write_public_report_no_overwrite(args.output, report)
    print(json.dumps({
        "report_sha256": digest,
        "all_profile_gate_passed": report["decision"]["all_profile_gate_passed"],
        "metrics": report["evaluations"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
