#!/usr/bin/env python3
"""Run PalsyNet-locked mirror aggregation and aggregate Mayo failure analysis."""
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
from scripts.run_architecture_search_v1 import _prepare_search_dataset  # noqa: E402
from scripts.run_dynamic_landmark_classical import (  # noqa: E402
    NUISANCE_FEATURE_NAMES,
    binary_group_metrics,
)
from scripts.run_mayo_positive_challenge_v1 import _load_mayo_features  # noqa: E402
from src.evaluation.mayo_failure_analysis_v1 import (  # noqa: E402
    AGGREGATOR_ORDER,
    REGION_ORDER,
    aggregate_mirror_probabilities,
    build_failure_summary,
    build_robust_inference_report,
    feature_region_assignments,
    select_palsynet_aggregator,
)
from src.evaluation.mayo_positive_challenge_v1 import (  # noqa: E402
    fit_frozen_110d_champion,
    positive_cohort_summary,
)
from src.preprocessing.generalization_110d import (  # noqa: E402
    LANDMARK_MI_110D,
    candidate_feature_names,
)


DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "outputs" / "dynamic_landmark" / "benchmarks" / "external"
    / "mayo-failure-analysis-robust-inference-v1" / "report.json"
)
_IMPLEMENTATION_FILES = (
    Path(__file__).resolve(),
    PROJECT_ROOT / "src" / "evaluation" / "mayo_failure_analysis_v1.py",
    PROJECT_ROOT / "src" / "evaluation" / "mayo_positive_challenge_v1.py",
    PROJECT_ROOT / "scripts" / "run_mayo_positive_challenge_v1.py",
    PROJECT_ROOT / "scripts" / "run_architecture_search_v1.py",
    PROJECT_ROOT / "scripts" / "run_110d_generalization_v1.py",
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
        raise FileExistsError(f"refusing to overwrite robust-inference report {path}")
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd, temporary_name = tempfile.mkstemp(
        prefix=".mayo-failure-analysis.", suffix=".tmp", dir=path.parent
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


def _view_probabilities(champion, original: np.ndarray, mirrored: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first = champion.model.predict_proba(
        champion.scaler.transform(original)
    )[:, 1]
    second = champion.model.predict_proba(
        champion.scaler.transform(mirrored)
    )[:, 1]
    return first.astype(np.float64, copy=False), second.astype(np.float64, copy=False)


def _palsynet_oof_metrics(prepared) -> dict[str, dict[str, float]]:
    development = prepared.development_indices
    fold_by_local = prepared.inner_fold_by_index[development]
    folds = tuple(sorted(set(fold_by_local.tolist())))
    if folds != tuple(range(4)):
        raise ValueError("PalsyNet development registry must contain four frozen folds")
    original_probability = np.full(development.size, np.nan, dtype=np.float64)
    mirrored_probability = np.full(development.size, np.nan, dtype=np.float64)
    counts = np.zeros(development.size, dtype=np.int64)
    labels = prepared.labels[development]
    groups = prepared.group_ids[development]
    for fold in folds:
        train_local = np.flatnonzero(fold_by_local != fold)
        validation_local = np.flatnonzero(fold_by_local == fold)
        if not set(groups[train_local].tolist()).isdisjoint(
            set(groups[validation_local].tolist())
        ):
            raise AssertionError("PalsyNet frozen fold splits a group")
        champion = fit_frozen_110d_champion(
            prepared.summary_features[development[train_local]],
            prepared.mirrored_summary_features[development[train_local]],
            labels[train_local],
            groups[train_local],
        )
        first, second = _view_probabilities(
            champion,
            prepared.summary_features[development[validation_local]],
            prepared.mirrored_summary_features[development[validation_local]],
        )
        original_probability[validation_local] = first
        mirrored_probability[validation_local] = second
        counts[validation_local] += 1
    if (
        not np.isfinite(original_probability).all()
        or not np.isfinite(mirrored_probability).all()
        or not np.all(counts == 1)
    ):
        raise AssertionError("PalsyNet OOF mirror views were not scored exactly once")
    metrics: dict[str, dict[str, float]] = {}
    for method in AGGREGATOR_ORDER:
        probabilities = aggregate_mirror_probabilities(
            original_probability, mirrored_probability, method
        )
        measured = binary_group_metrics(labels, groups, probabilities)
        metrics[method] = {
            key: float(measured[key])
            for key in ("auroc", "balanced_accuracy", "brier")
        }
    return metrics


def _load_mayo_nuisance(cache_root: Path, expected_manifest_sha: str) -> np.ndarray:
    manifest, manifest_sha = _read_json(cache_root / "collection_manifest.json")
    if manifest_sha != expected_manifest_sha:
        raise ValueError("Mayo manifest changed between feature and nuisance reads")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Mayo manifest has no diagnostic records")
    rows: list[np.ndarray] = []
    for row in records:
        nuisance = row.get("nuisance") if isinstance(row, dict) else None
        if not isinstance(nuisance, dict) or set(nuisance) != set(NUISANCE_FEATURE_NAMES):
            raise ValueError("Mayo nuisance fields differ from the frozen schema")
        values = np.asarray(
            [float(nuisance[name]) for name in NUISANCE_FEATURE_NAMES],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError("Mayo nuisance values must be finite")
        rows.append(values)
    return np.stack(rows)


def _region_contributions(champion, original: np.ndarray, mirrored: np.ndarray) -> np.ndarray:
    names = candidate_feature_names(LANDMARK_MI_110D)
    regions = feature_region_assignments(names)
    coefficients = np.asarray(champion.model.coef_, dtype=np.float64)
    if coefficients.shape != (1, 110):
        raise ValueError("frozen champion coefficient shape drifted")
    standardized = 0.5 * (
        champion.scaler.transform(original) + champion.scaler.transform(mirrored)
    )
    feature_contributions = standardized * coefficients[0]
    output = np.zeros((original.shape[0], len(REGION_ORDER)), dtype=np.float64)
    for region_index, region in enumerate(REGION_ORDER):
        indices = np.asarray(
            [index for index, value in enumerate(regions) if value == region],
            dtype=np.int64,
        )
        output[:, region_index] = feature_contributions[:, indices].sum(axis=1)
    if not np.isfinite(output).all():
        raise ValueError("region logit contribution produced nonfinite values")
    return output


def run_analysis(args) -> dict[str, object]:
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
    prepared = _prepare_search_dataset(dataset, gate)
    metrics = _palsynet_oof_metrics(prepared)
    decision = select_palsynet_aggregator(metrics)

    development = prepared.development_indices
    champion = fit_frozen_110d_champion(
        prepared.summary_features[development],
        prepared.mirrored_summary_features[development],
        prepared.labels[development],
        prepared.group_ids[development],
    )
    mayo_original, mayo_mirrored, coverage, mayo_manifest_sha, _counts = (
        _load_mayo_features(args.mayo_cache_root)
    )
    nuisance = _load_mayo_nuisance(args.mayo_cache_root, mayo_manifest_sha)
    first, second = _view_probabilities(champion, mayo_original, mayo_mirrored)
    current_probabilities = aggregate_mirror_probabilities(
        first, second, "mirror_mean"
    )
    failure_analysis = build_failure_summary(
        current_probabilities,
        coverage,
        nuisance,
        NUISANCE_FEATURE_NAMES,
        _region_contributions(champion, mayo_original, mayo_mirrored),
    )
    selected_probabilities = aggregate_mirror_probabilities(
        first, second, str(decision["selected"])
    )
    if audit.protected_cache_records_loaded != 0:
        raise AssertionError("robust inference opened protected PalsyNet cache records")
    report = build_robust_inference_report(
        metrics,
        decision,
        failure_analysis,
        positive_cohort_summary(current_probabilities, coverage),
        positive_cohort_summary(selected_probabilities, coverage),
        development_recordings=int(development.size),
        development_groups=len(set(prepared.group_ids[development].tolist())),
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
    report = run_analysis(_parser().parse_args(argv))
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
