"""Frozen-model evaluation helpers for a one-class Mayo challenge cohort.

The cohort can estimate positive-call consistency and confidence distribution.
It cannot estimate specificity, balanced accuracy, AUROC, or ordinary accuracy
because it has no verified negative class.  Architecture selection is performed
upstream on the identity-reviewed PalsyNet development folds only.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VIDEO_SUFFIX = ".mov"


@dataclass(frozen=True)
class ChallengeRecord:
    """A private in-memory path paired with deidentified content identity."""

    path: Path
    source_sha256: str
    recording_id: str
    group_id: str


@dataclass(frozen=True)
class ContentInventory:
    source_files: int
    unique_contents: int
    exact_duplicate_files: int
    records: tuple[ChallengeRecord, ...]


@dataclass(frozen=True)
class FrozenChampion:
    scaler: Any
    model: Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _opaque(prefix: str, namespace: str, source_sha256: str) -> str:
    encoded = f"mayo-positive-challenge-v1:{namespace}:{source_sha256}".encode("ascii")
    return prefix + hashlib.sha256(encoded).hexdigest()


def inventory_content_deduplicated_videos(
    data_root: str | Path,
) -> ContentInventory:
    """Hash every regular MOV and keep one private path per exact content hash."""
    root = Path(data_root).expanduser().absolute()
    try:
        info = os.lstat(root)
    except FileNotFoundError as exc:
        raise ValueError("Mayo video root is missing") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("Mayo video root must be a real directory")
    paths: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: str(item)):
        if path.suffix.lower() != _VIDEO_SUFFIX:
            continue
        item_info = os.lstat(path)
        if stat.S_ISLNK(item_info.st_mode) or not stat.S_ISREG(item_info.st_mode):
            raise ValueError("Mayo inventory contains a non-regular MOV")
        paths.append(path.absolute())
    if not paths:
        raise ValueError("Mayo inventory contains no MOV videos")
    by_digest: dict[str, list[Path]] = {}
    for path in paths:
        by_digest.setdefault(_sha256_file(path), []).append(path)
    records = tuple(
        ChallengeRecord(
            path=sorted(duplicates, key=lambda item: str(item))[0],
            source_sha256=digest,
            recording_id=_opaque("rec_", "recording", digest),
            group_id=_opaque("grp_", "group", digest),
        )
        for digest, duplicates in sorted(by_digest.items())
    )
    return ContentInventory(
        source_files=len(paths),
        unique_contents=len(records),
        exact_duplicate_files=len(paths) - len(records),
        records=records,
    )


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1:] != (110,) or matrix.shape[0] == 0:
        raise ValueError(f"{name} must have shape (N, 110)")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite")
    return matrix


def fit_frozen_110d_champion(
    original: Sequence[Sequence[float]] | np.ndarray,
    mirrored: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    group_ids: Sequence[object] | np.ndarray,
) -> FrozenChampion:
    """Fit the already-selected C=0.01 mirror-invariant Logistic champion."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from ..training.architecture_search_v1 import group_balanced_weights

    original_matrix = _feature_matrix(original, "original")
    mirrored_matrix = _feature_matrix(mirrored, "mirrored")
    label_array = np.asarray(labels)
    groups = np.asarray(group_ids, dtype=object)
    n = original_matrix.shape[0]
    if mirrored_matrix.shape != original_matrix.shape:
        raise ValueError("original and mirrored development features must align")
    if (
        label_array.shape != (n,) or label_array.dtype.kind not in {"i", "u"}
        or set(label_array.tolist()) != {0, 1}
        or groups.shape != (n,)
    ):
        raise ValueError("development labels/groups must align and contain both classes")
    if any(not isinstance(group, str) or not group for group in groups.tolist()):
        raise ValueError("development group IDs must be nonempty strings")
    for group in set(groups.tolist()):
        if len(set(label_array[groups == group].tolist())) != 1:
            raise ValueError("one development group cannot cross labels")
    training = np.concatenate((original_matrix, mirrored_matrix), axis=0)
    training_labels = np.concatenate((label_array, label_array), axis=0)
    training_groups = np.concatenate((groups, groups), axis=0)
    scaler = StandardScaler().fit(training)
    model = LogisticRegression(
        C=0.01,
        penalty="l2",
        solver="liblinear",
        max_iter=2000,
        random_state=0,
    )
    model.fit(
        scaler.transform(training),
        training_labels,
        sample_weight=group_balanced_weights(training_groups),
    )
    return FrozenChampion(scaler=scaler, model=model)


def predict_mirror_mean(
    champion: FrozenChampion,
    original: Sequence[Sequence[float]] | np.ndarray,
    mirrored: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    if not isinstance(champion, FrozenChampion):
        raise ValueError("champion must be a fitted FrozenChampion")
    original_matrix = _feature_matrix(original, "original")
    mirrored_matrix = _feature_matrix(mirrored, "mirrored")
    if mirrored_matrix.shape != original_matrix.shape:
        raise ValueError("original and mirrored challenge features must align")
    original_probability = champion.model.predict_proba(
        champion.scaler.transform(original_matrix)
    )[:, 1]
    mirrored_probability = champion.model.predict_proba(
        champion.scaler.transform(mirrored_matrix)
    )[:, 1]
    probabilities = 0.5 * (original_probability + mirrored_probability)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise RuntimeError("champion produced invalid probabilities")
    return probabilities.astype(np.float64, copy=False)


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total < 1 or successes < 0 or successes > total:
        raise ValueError("Wilson inputs are invalid")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def positive_cohort_summary(
    probabilities: Sequence[float] | np.ndarray,
    coverages: Sequence[float] | np.ndarray,
) -> dict[str, object]:
    """Summarize one-class challenge behavior without inventing accuracy."""
    scores = np.asarray(probabilities, dtype=np.float64)
    coverage = np.asarray(coverages, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0 or coverage.shape != scores.shape:
        raise ValueError("challenge scores and coverage must be aligned nonempty vectors")
    if (
        not np.isfinite(scores).all() or not np.isfinite(coverage).all()
        or np.any((scores < 0) | (scores > 1))
        or np.any((coverage < 0) | (coverage > 1))
    ):
        raise ValueError("challenge scores and coverage must lie within [0, 1]")
    positive_calls = int(np.sum(scores >= 0.5))
    quantiles = np.quantile(scores, (0.0, 0.25, 0.5, 0.75, 1.0))
    coverage_quantiles = np.quantile(coverage, (0.0, 0.5, 1.0))
    return {
        "records": int(scores.size),
        "assumed_positive_records": int(scores.size),
        "verified_negative_records": 0,
        "positive_calls": positive_calls,
        "positive_call_rate": float(positive_calls / scores.size),
        "positive_call_rate_wilson95": _wilson_interval(positive_calls, int(scores.size)),
        "confidence": {
            "mean": float(scores.mean()),
            "minimum": float(quantiles[0]),
            "q25": float(quantiles[1]),
            "median": float(quantiles[2]),
            "q75": float(quantiles[3]),
            "maximum": float(quantiles[4]),
            "at_least_0_80": float(np.mean(scores >= 0.80)),
            "at_least_0_90": float(np.mean(scores >= 0.90)),
            "at_least_0_95": float(np.mean(scores >= 0.95)),
        },
        "extraction_coverage": {
            "minimum": float(coverage_quantiles[0]),
            "median": float(coverage_quantiles[1]),
            "maximum": float(coverage_quantiles[2]),
        },
        "accuracy_defined": False,
        "specificity_defined": False,
        "auroc_defined": False,
    }


def build_aggregate_challenge_report(
    summary: Mapping[str, object],
    *,
    source_files: int,
    unique_contents: int,
    exact_duplicate_files: int,
    excluded_records: int,
    provenance: Mapping[str, str],
) -> dict[str, object]:
    required = {
        "palsynet_source_collection_sha256",
        "palsynet_reviewed_manifest_sha256",
        "palsynet_review_ledger_sha256",
        "palsynet_split_registry_sha256",
        "mayo_cache_manifest_sha256",
        "implementation_sha256",
    }
    if set(provenance) != required or any(
        not isinstance(value, str) or _SHA256.fullmatch(value) is None
        for value in provenance.values()
    ):
        raise ValueError("challenge provenance must contain six SHA-256 digests")
    counts = (source_files, unique_contents, exact_duplicate_files, excluded_records)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
        raise ValueError("challenge inventory counts must be nonnegative integers")
    if unique_contents + exact_duplicate_files != source_files:
        raise ValueError("challenge inventory counts do not reconcile")
    if summary.get("records") != unique_contents - excluded_records:
        raise ValueError("challenge score count does not reconcile with exclusions")
    return {
        "schema_version": "mayo_positive_challenge_v1_report",
        "claim_scope": "known_positive_recording_cohort_challenge_only",
        "target": "binary_affected_probability_not_hb_grade",
        "inventory": {
            "source_video_files": source_files,
            "unique_video_contents": unique_contents,
            "exact_duplicate_files": exact_duplicate_files,
            "quality_excluded_unique_contents": excluded_records,
        },
        "summary": dict(summary),
        "protocol": {
            "training_source": "identity_reviewed_palsynet_development_only",
            "model": "frozen_mirror_invariant_landmark_110d_logistic_c_0_01",
            "threshold": 0.5,
            "challenge_source": "mayo_known_positive_local_video_cohort",
            "deduplication": "exact_file_sha256",
        },
        "decision": {
            "mayo_used_for_model_selection": False,
            "current_model_replaced": False,
            "accuracy_claimed": False,
            "clinical_validation": False,
        },
        "audit": {
            "palsynet_protected_cache_records_loaded": 0,
            "palsynet_protected_predictions": 0,
            "raw_mayo_videos_uploaded": 0,
            "per_record_probabilities_exported": 0,
        },
        "provenance": dict(provenance),
    }
