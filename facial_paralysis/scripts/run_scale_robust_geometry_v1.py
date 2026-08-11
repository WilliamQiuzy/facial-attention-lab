#!/usr/bin/env python3
"""Run PalsyNet-only scale-robust selection, then one Mayo challenge."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_110d_generalization_v1 import (  # noqa: E402
    GateAudit,
    _build_cache_metadata_dataset,
    _read_json,
    load_development_cache_records,
    validate_development_gate,
)
from scripts.run_dynamic_landmark_classical import (  # noqa: E402
    NUISANCE_FEATURE_NAMES,
    binary_group_metrics,
)
from scripts.run_mayo_positive_challenge_v1 import _load_mayo_features  # noqa: E402
from scripts.run_mirror_invariant_110d import mirror_dynamic_features  # noqa: E402
from src.datasets.dynamic_landmark import (  # noqa: E402
    DYNAMIC_FEATURE_NAMES,
    DYNAMIC_FEATURE_SCHEMA,
    DYNAMIC_FEATURE_SHAPE,
)
from src.evaluation.mayo_positive_challenge_v1 import (  # noqa: E402
    fit_frozen_110d_champion,
    positive_cohort_summary,
)
from src.evaluation.scale_robust_geometry_v1 import (  # noqa: E402
    METRIC_FIELDS,
    build_action_coverage_summary,
    build_scale_robust_report,
    select_low_scale_groups,
    select_scale_robust_candidate,
)
from src.preprocessing.scale_robust_geometry_v1 import (  # noqa: E402
    CANDIDATE_ORDER,
    RAW_110D,
    scale_robust_feature_vector,
)


DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "outputs" / "dynamic_landmark" / "benchmarks" / "external"
    / "scale-robust-eye-geometry-v1" / "report.json"
)
_MAYO_CACHE_FIELDS = {
    "features", "valid_mask", "timestamps", "timestamp_unit",
    "source_frame_indices", "source_frame_count", "feature_schema",
    "feature_names", "recording_id", "group_id", "label", "source_sha256",
}
_IMPLEMENTATION_FILES = (
    Path(__file__).resolve(),
    PROJECT_ROOT / "src" / "preprocessing" / "scale_robust_geometry_v1.py",
    PROJECT_ROOT / "src" / "evaluation" / "scale_robust_geometry_v1.py",
    PROJECT_ROOT / "src" / "evaluation" / "mayo_positive_challenge_v1.py",
    PROJECT_ROOT / "scripts" / "run_mayo_positive_challenge_v1.py",
    PROJECT_ROOT / "scripts" / "run_110d_generalization_v1.py",
    PROJECT_ROOT / "scripts" / "run_mirror_invariant_110d.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_sha256() -> str:
    components = {
        str(path.relative_to(PROJECT_ROOT)): _sha256(path)
        for path in _IMPLEMENTATION_FILES
    }
    encoded = json.dumps(components, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_no_overwrite(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite scale-robust report {path}")
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd, temporary_name = tempfile.mkstemp(
        prefix=".scale-robust.", suffix=".tmp", dir=path.parent
    )
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


def _development_matrices(dataset, gate, audit: GateAudit):
    if audit.gate_passes != 1:
        raise ValueError("identity gate must pass before scale-robust extraction")
    development = np.asarray(gate.development_indices, dtype=np.int64)
    if set(development.tolist()).intersection(set(gate.protected_indices.tolist())):
        raise ValueError("development and protected rows overlap")
    original = {candidate: [] for candidate in CANDIDATE_ORDER}
    mirrored = {candidate: [] for candidate in CANDIDATE_ORDER}
    for index in development.tolist():
        raw = np.asarray(dataset.features[index], dtype=np.float32)
        mirrored_raw = np.asarray(mirror_dynamic_features(raw), dtype=np.float32)
        audit.development_mirror_transforms += 1
        temporal = (
            dataset.valid_masks[index],
            dataset.timestamps[index],
            dataset.source_frame_indices[index],
        )
        for candidate in CANDIDATE_ORDER:
            original[candidate].append(
                scale_robust_feature_vector(candidate, raw, *temporal)
            )
            mirrored[candidate].append(
                scale_robust_feature_vector(candidate, mirrored_raw, *temporal)
            )
            audit.development_feature_extractions += 2
    return (
        development,
        {
            candidate: np.stack(original[candidate]).astype(np.float64, copy=False)
            for candidate in CANDIDATE_ORDER
        },
        {
            candidate: np.stack(mirrored[candidate]).astype(np.float64, copy=False)
            for candidate in CANDIDATE_ORDER
        },
    )


def _view_probabilities(champion, original: np.ndarray, mirrored: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first = champion.model.predict_proba(
        champion.scaler.transform(original)
    )[:, 1]
    second = champion.model.predict_proba(
        champion.scaler.transform(mirrored)
    )[:, 1]
    return first.astype(np.float64, copy=False), second.astype(np.float64, copy=False)


def _mirror_mean(champion, original: np.ndarray, mirrored: np.ndarray) -> np.ndarray:
    first, second = _view_probabilities(champion, original, mirrored)
    probabilities = 0.5 * (first + second)
    if not np.isfinite(probabilities).all():
        raise ValueError("scale-robust model produced nonfinite probabilities")
    return probabilities


def _oof_metrics(
    dataset,
    gate,
    development: np.ndarray,
    original: Mapping[str, np.ndarray],
    mirrored: Mapping[str, np.ndarray],
    audit: GateAudit,
) -> tuple[dict[str, dict[str, float]], np.ndarray]:
    labels = np.asarray(dataset.labels[development], dtype=np.int64)
    groups = np.asarray(gate.group_ids[development], dtype=object)
    face_scale_column = NUISANCE_FEATURE_NAMES.index("face_scale_mean")
    face_scales = np.asarray(
        dataset.nuisance[development, face_scale_column], dtype=np.float64
    )
    low_scale = select_low_scale_groups(labels, groups, face_scales)
    fold_by_local = np.asarray(gate.inner_fold_by_index[development], dtype=np.int64)
    if tuple(sorted(set(fold_by_local.tolist()))) != tuple(range(4)):
        raise ValueError("PalsyNet registry must contain four development folds")
    metrics: dict[str, dict[str, float]] = {}
    for candidate in CANDIDATE_ORDER:
        probabilities = np.full(development.size, np.nan, dtype=np.float64)
        counts = np.zeros(development.size, dtype=np.int64)
        for fold in range(4):
            train = np.flatnonzero(fold_by_local != fold)
            validation = np.flatnonzero(fold_by_local == fold)
            if not set(groups[train].tolist()).isdisjoint(set(groups[validation].tolist())):
                raise AssertionError("scale-robust OOF fold splits a group")
            champion = fit_frozen_110d_champion(
                original[candidate][train],
                mirrored[candidate][train],
                labels[train],
                groups[train],
            )
            audit.development_scaler_fits += 1
            audit.development_model_fits += 1
            probabilities[validation] = _mirror_mean(
                champion,
                original[candidate][validation],
                mirrored[candidate][validation],
            )
            audit.development_predictions += int(validation.size)
            counts[validation] += 1
        if not np.isfinite(probabilities).all() or not np.all(counts == 1):
            raise AssertionError("scale-robust OOF did not predict each row once")
        overall = binary_group_metrics(labels, groups, probabilities)
        stress = binary_group_metrics(
            labels[low_scale], groups[low_scale], probabilities[low_scale]
        )
        metrics[candidate] = {
            "overall_auroc": float(overall["auroc"]),
            "overall_balanced_accuracy": float(overall["balanced_accuracy"]),
            "overall_brier": float(overall["brier"]),
            "low_scale_auroc": float(stress["auroc"]),
            "low_scale_balanced_accuracy": float(stress["balanced_accuracy"]),
            "low_scale_brier": float(stress["brier"]),
        }
        if tuple(metrics[candidate]) != METRIC_FIELDS:
            raise AssertionError("scale-robust metric field order drifted")
    return metrics, low_scale


def _mayo_selected_matrices(
    cache_root: Path,
    candidate: str,
    baseline_original: np.ndarray,
    baseline_mirrored: np.ndarray,
    expected_manifest_sha: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    manifest, manifest_sha = _read_json(cache_root / "collection_manifest.json")
    if manifest_sha != expected_manifest_sha:
        raise ValueError("Mayo manifest changed after baseline validation")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != baseline_original.shape[0]:
        raise ValueError("Mayo record order differs from validated baseline")
    frame_counts: list[int] = []
    frame_rates: list[float] = []
    starts: list[list[int]] = []
    if candidate == RAW_110D:
        selected_original = baseline_original
        selected_mirrored = baseline_mirrored
    else:
        original_rows: list[np.ndarray] = []
        mirrored_rows: list[np.ndarray] = []
        for row in records:
            path = cache_root / f"{row['recording_id']}.npz"
            with np.load(path, allow_pickle=False) as saved:
                if set(saved.files) != _MAYO_CACHE_FIELDS:
                    raise ValueError("Mayo selected-candidate cache fields drifted")
                values = {name: np.asarray(saved[name]) for name in _MAYO_CACHE_FIELDS}
            scalar = lambda name: np.asarray(values[name]).item()
            if (
                scalar("recording_id") != row.get("recording_id")
                or scalar("group_id") != row.get("group_id")
                or scalar("source_sha256") != row.get("source_sha256")
                or scalar("label") != 1
                or scalar("timestamp_unit") != "seconds"
                or scalar("feature_schema") != DYNAMIC_FEATURE_SCHEMA
                or tuple(values["feature_names"].tolist()) != DYNAMIC_FEATURE_NAMES
            ):
                raise ValueError("Mayo selected-candidate metadata drifted")
            features = values["features"]
            mask = values["valid_mask"]
            timestamps = values["timestamps"]
            indices = values["source_frame_indices"]
            if features.shape != DYNAMIC_FEATURE_SHAPE or features.dtype != np.float32:
                raise ValueError("Mayo selected-candidate feature shape drifted")
            original_rows.append(
                scale_robust_feature_vector(
                    candidate, features, mask, timestamps, indices
                )
            )
            mirrored_rows.append(
                scale_robust_feature_vector(
                    candidate,
                    np.asarray(mirror_dynamic_features(features), dtype=np.float32),
                    mask,
                    timestamps,
                    indices,
                )
            )
        selected_original = np.stack(original_rows)
        selected_mirrored = np.stack(mirrored_rows)
    for row in records:
        frame_counts.append(int(row["source_frame_count"]))
        frame_rates.append(float(row["fps"]))
        row_starts = row.get("window_starts")
        if not isinstance(row_starts, list) or len(row_starts) != 4:
            raise ValueError("Mayo window starts differ from four-window contract")
        starts.append([int(value) for value in row_starts])
    action_coverage = build_action_coverage_summary(
        source_frame_counts=np.asarray(frame_counts, dtype=np.int64),
        fps=np.asarray(frame_rates, dtype=np.float64),
        window_starts=np.asarray(starts, dtype=np.int64),
    )
    return selected_original, selected_mirrored, action_coverage


def run_experiment(args) -> dict[str, object]:
    manifest, manifest_sha = _read_json(args.reviewed_identity_manifest)
    ledger, ledger_sha = _read_json(args.review_ledger)
    registry, registry_sha = _read_json(args.split_registry)
    dataset, collection_rows, collection_sha = _build_cache_metadata_dataset(
        args.palsynet_cache_root
    )
    audit = GateAudit()
    gate = validate_development_gate(
        dataset,
        manifest,
        ledger,
        registry,
        reviewed_manifest_sha256=manifest_sha,
        review_ledger_sha256=ledger_sha,
        split_registry_sha256=registry_sha,
        cache_source_sha256_by_recording_id={
            recording_id: str(row["source_sha256"])
            for recording_id, row in collection_rows.items()
        },
        cache_source_collection_sha256=collection_sha,
        audit=audit,
    )
    load_development_cache_records(
        args.palsynet_cache_root, dataset, gate, collection_rows, audit=audit
    )
    development, original, mirrored = _development_matrices(dataset, gate, audit)
    metrics, low_scale = _oof_metrics(
        dataset, gate, development, original, mirrored, audit
    )
    decision = select_scale_robust_candidate(metrics)
    selected = str(decision["selected"])
    labels = dataset.labels[development]
    groups = gate.group_ids[development]
    champion = fit_frozen_110d_champion(
        original[selected], mirrored[selected], labels, groups
    )
    audit.development_scaler_fits += 1
    audit.development_model_fits += 1

    mayo_raw_original, mayo_raw_mirrored, coverage, mayo_manifest_sha, _counts = (
        _load_mayo_features(args.mayo_cache_root)
    )
    current_probabilities = _mirror_mean(
        fit_frozen_110d_champion(
            original[RAW_110D], mirrored[RAW_110D], labels, groups
        ),
        mayo_raw_original,
        mayo_raw_mirrored,
    )
    audit.development_scaler_fits += 1
    audit.development_model_fits += 1
    mayo_selected_original, mayo_selected_mirrored, action_coverage = (
        _mayo_selected_matrices(
            args.mayo_cache_root,
            selected,
            mayo_raw_original,
            mayo_raw_mirrored,
            mayo_manifest_sha,
        )
    )
    selected_probabilities = _mirror_mean(
        champion, mayo_selected_original, mayo_selected_mirrored
    )
    if any((
        audit.protected_cache_records_loaded,
        audit.protected_feature_extractions,
        audit.protected_scaler_fits,
        audit.protected_model_fits,
        audit.protected_predictions,
    )):
        raise AssertionError("scale-robust experiment touched protected PalsyNet state")
    report = build_scale_robust_report(
        metrics,
        decision,
        positive_cohort_summary(current_probabilities, coverage),
        positive_cohort_summary(selected_probabilities, coverage),
        action_coverage,
        development_recordings=int(development.size),
        development_groups=len(set(groups.tolist())),
        low_scale_groups=len(set(groups[low_scale].tolist())),
        mayo_records=int(current_probabilities.size),
        provenance={
            "palsynet_source_collection_sha256": gate.source_collection_sha256,
            "palsynet_reviewed_manifest_sha256": gate.reviewed_manifest_sha256,
            "palsynet_review_ledger_sha256": gate.review_ledger_sha256,
            "palsynet_split_registry_sha256": gate.split_registry_sha256,
            "mayo_cache_manifest_sha256": mayo_manifest_sha,
            "implementation_sha256": _implementation_sha256(),
        },
        protected_cache_records_loaded=audit.protected_cache_records_loaded,
    )
    _write_no_overwrite(args.output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--palsynet-cache-root", required=True, type=Path)
    parser.add_argument("--reviewed-identity-manifest", required=True, type=Path)
    parser.add_argument("--review-ledger", required=True, type=Path)
    parser.add_argument("--split-registry", required=True, type=Path)
    parser.add_argument("--mayo-cache-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    return parser


def main(argv: list[str] | None = None) -> None:
    report = run_experiment(_parser().parse_args(argv))
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
