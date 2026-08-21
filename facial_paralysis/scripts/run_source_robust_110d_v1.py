#!/usr/bin/env python3
"""Run the PalsyNet-only Source-Robust Landmark 110D v1 comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)

import sys
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
from src.evaluation.source_robust_110d_v1 import (  # noqa: E402
    ACQUISITION_NUISANCE_INDICES,
    FIXED_C,
    FIXED_THRESHOLD,
    REGISTERED_AUROC_TOLERANCE,
    REGISTERED_BALANCED_ACCURACY_TOLERANCE,
    build_acquisition_blocked_folds,
    choose_source_robust_candidate,
    run_candidate_oof,
)
from src.preprocessing.source_robust_110d import (  # noqa: E402
    CANDIDATE_DIMENSIONS,
    CANDIDATE_ORDER,
    candidate_feature_names,
    source_robust_feature_views,
)


REPORT_SCHEMA = "source_robust_110d_v1_development_report"
BOOTSTRAP_REPEATS = 5000
BOOTSTRAP_SEED = 20260812
_IMPLEMENTATION_FILES = (
    Path(__file__).resolve(),
    PROJECT_ROOT / "src/evaluation/source_robust_110d_v1.py",
    PROJECT_ROOT / "src/preprocessing/source_robust_110d.py",
    PROJECT_ROOT / "src/preprocessing/trajectory_features.py",
    PROJECT_ROOT / "scripts/run_110d_generalization_v1.py",
    PROJECT_ROOT / "scripts/run_architecture_search_v1.py",
)


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in _IMPLEMENTATION_FILES:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _group_metrics(
    labels: np.ndarray,
    groups: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int64)
    group_values = np.asarray(groups)
    scores = np.asarray(probabilities, dtype=np.float64)
    if y.shape != group_values.shape or y.shape != scores.shape or y.ndim != 1:
        raise ValueError("labels, groups, and probabilities must align")
    grouped_labels, grouped_scores = [], []
    for group in sorted(set(group_values.tolist()), key=str):
        rows = np.flatnonzero(group_values == group)
        observed = np.unique(y[rows])
        if observed.size != 1:
            raise ValueError("one reviewed group cannot cross labels")
        grouped_labels.append(int(observed[0]))
        grouped_scores.append(float(np.mean(scores[rows])))
    group_y = np.asarray(grouped_labels, dtype=np.int64)
    group_scores = np.asarray(grouped_scores, dtype=np.float64)
    predictions = group_scores >= FIXED_THRESHOLD
    return {
        "auroc": float(roc_auc_score(group_y, group_scores)),
        "average_precision": float(average_precision_score(group_y, group_scores)),
        "balanced_accuracy": float(balanced_accuracy_score(group_y, predictions)),
        "sensitivity": float(np.mean(predictions[group_y == 1])),
        "specificity": float(np.mean(~predictions[group_y == 0])),
        "brier": float(brier_score_loss(group_y, group_scores)),
    }, group_y, group_scores


def _paired_bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, object]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    classes = {label: np.flatnonzero(labels == label) for label in (0, 1)}
    samples = {
        "auroc": np.empty(BOOTSTRAP_REPEATS),
        "balanced_accuracy": np.empty(BOOTSTRAP_REPEATS),
        "brier": np.empty(BOOTSTRAP_REPEATS),
    }
    for repeat in range(BOOTSTRAP_REPEATS):
        draw = np.concatenate([
            rng.choice(rows, size=rows.size, replace=True)
            for rows in classes.values()
        ])
        y, first, second = labels[draw], baseline[draw], candidate[draw]
        samples["auroc"][repeat] = roc_auc_score(y, second) - roc_auc_score(y, first)
        samples["balanced_accuracy"][repeat] = (
            balanced_accuracy_score(y, second >= FIXED_THRESHOLD)
            - balanced_accuracy_score(y, first >= FIXED_THRESHOLD)
        )
        samples["brier"][repeat] = brier_score_loss(y, second) - brier_score_loss(y, first)
    points = {
        "auroc": roc_auc_score(labels, candidate) - roc_auc_score(labels, baseline),
        "balanced_accuracy": (
            balanced_accuracy_score(labels, candidate >= FIXED_THRESHOLD)
            - balanced_accuracy_score(labels, baseline >= FIXED_THRESHOLD)
        ),
        "brier": brier_score_loss(labels, candidate) - brier_score_loss(labels, baseline),
    }
    return {
        metric: {
            "point": float(points[metric]),
            "ci95": [float(value) for value in np.quantile(values, (0.025, 0.975))],
        }
        for metric, values in samples.items()
    }


def run(args) -> dict[str, object]:
    reviewed, reviewed_sha = _read_json(args.reviewed_identity_manifest)
    ledger, ledger_sha = _read_json(args.review_ledger)
    registry, registry_sha = _read_json(args.split_registry)
    dataset, collection_rows, collection_sha = _build_cache_metadata_dataset(
        args.palsynet_cache_root
    )
    gate_audit = GateAudit()
    gate = validate_development_gate(
        dataset, reviewed, ledger, registry,
        reviewed_manifest_sha256=reviewed_sha,
        review_ledger_sha256=ledger_sha,
        split_registry_sha256=registry_sha,
        cache_source_sha256_by_recording_id={
            recording_id: str(row["source_sha256"])
            for recording_id, row in collection_rows.items()
        },
        cache_source_collection_sha256=collection_sha,
        audit=gate_audit,
    )
    load_development_cache_records(
        args.palsynet_cache_root, dataset, gate, collection_rows, audit=gate_audit
    )
    prepared = _prepare_search_dataset(dataset, gate)
    development = prepared.development_indices
    original = source_robust_feature_views(prepared.summary_features[development])
    mirrored = source_robust_feature_views(
        prepared.mirrored_summary_features[development]
    )
    labels = prepared.labels[development]
    groups = prepared.group_ids[development]
    registered_folds = prepared.inner_fold_by_index[development]
    acquisition_folds = build_acquisition_blocked_folds(
        labels, groups, dataset.nuisance[development]
    )
    fold_registry = {
        "registered": registered_folds,
        "acquisition_blocked": acquisition_folds,
    }
    results = {
        protocol: run_candidate_oof(
            labels=labels, group_ids=groups, folds=folds,
            original=original, mirrored=mirrored,
        )
        for protocol, folds in fold_registry.items()
    }

    metrics: dict[str, dict[str, dict[str, float]]] = {}
    grouped: dict[str, dict[str, np.ndarray]] = {}
    grouped_labels: np.ndarray | None = None
    for protocol, result in results.items():
        metrics[protocol], grouped[protocol] = {}, {}
        for candidate in CANDIDATE_ORDER:
            metric, candidate_labels, scores = _group_metrics(
                labels, groups, result.probabilities[candidate]
            )
            metrics[protocol][candidate] = metric
            grouped[protocol][candidate] = scores
            if grouped_labels is None:
                grouped_labels = candidate_labels
            elif not np.array_equal(grouped_labels, candidate_labels):
                raise RuntimeError("group-level candidate ordering drifted")
    assert grouped_labels is not None
    decision = choose_source_robust_candidate(metrics)
    comparisons = {
        protocol: {
            candidate: _paired_bootstrap(
                grouped_labels,
                grouped[protocol][CANDIDATE_ORDER[0]],
                grouped[protocol][candidate],
            )
            for candidate in CANDIDATE_ORDER[1:]
        }
        for protocol in fold_registry
    }
    audit = {
        "development_cache_records_loaded": gate_audit.development_cache_records_loaded,
        "development_feature_extractions": int(development.size) * 2,
        "development_mirror_transforms": int(development.size),
        "development_scaler_fits": sum(
            result.audit["development_scaler_fits"] for result in results.values()
        ),
        "development_model_fits": sum(
            result.audit["development_model_fits"] for result in results.values()
        ),
        "development_predictions": sum(
            result.audit["development_predictions"] for result in results.values()
        ),
        "protected_cache_records_loaded": gate_audit.protected_cache_records_loaded,
        "protected_feature_reads": 0,
        "protected_fits": 0,
        "protected_predictions": 0,
        "mayo_reads": 0,
        "meei_reads": 0,
    }
    if any(audit[key] != 0 for key in (
        "protected_cache_records_loaded", "protected_feature_reads",
        "protected_fits", "protected_predictions", "mayo_reads", "meei_reads",
    )):
        raise AssertionError("evaluation boundary was violated")
    return {
        "schema_version": REPORT_SCHEMA,
        "claim_scope": "identity_reviewed_palsynet_development_only",
        "target": "binary_affected_vs_unaffected_not_hb_grade",
        "counts": {
            "development_recordings": int(development.size),
            "development_groups": len(set(groups.tolist())),
            "development_affected_groups": len(set(groups[labels == 1].tolist())),
            "development_unaffected_groups": len(set(groups[labels == 0].tolist())),
            "protected_recordings": int(gate.protected_indices.size),
            "protected_groups": len(set(gate.group_ids[gate.protected_indices].tolist())),
        },
        "protocol": {
            "candidates": list(CANDIDATE_ORDER),
            "candidate_dimensions": dict(CANDIDATE_DIMENSIONS),
            "candidate_feature_names": {
                candidate: list(candidate_feature_names(candidate))
                for candidate in CANDIDATE_ORDER
            },
            "split_families": ["registered", "acquisition_blocked"],
            "acquisition_nuisance_indices": list(ACQUISITION_NUISANCE_INDICES),
            "acquisition_blocking": "development_group_mean_standardize_pca1_label_stratified_contiguous_quartiles",
            "model": {
                "type": "standardized_l2_logistic_regression",
                "c": FIXED_C,
                "threshold": FIXED_THRESHOLD,
                "training": "original_plus_horizontal_mirror",
                "inference": "mean_original_and_horizontal_mirror_probability",
                "hyperparameter_search": False,
            },
            "promotion_tolerances": {
                "registered_auroc": REGISTERED_AUROC_TOLERANCE,
                "registered_balanced_accuracy": REGISTERED_BALANCED_ACCURACY_TOLERANCE,
            },
            "bootstrap": {"repeats": BOOTSTRAP_REPEATS, "seed": BOOTSTRAP_SEED},
        },
        "metrics": metrics,
        "paired_candidate_minus_baseline": comparisons,
        "decision": {
            "locked_candidate": decision.locked_candidate,
            "current_model_replaced": decision.locked_candidate != CANDIDATE_ORDER[0],
            "candidate_gates": decision.gates,
            "outer_evaluation_authorized": False,
            "meei_rescore_authorized": False,
            "mayo_used_for_selection": False,
            "clinical_validation": False,
            "hb_claim_authorized": False,
        },
        "audit": audit,
        "provenance": {
            "source_collection_sha256": collection_sha,
            "reviewed_manifest_sha256": reviewed_sha,
            "review_ledger_sha256": ledger_sha,
            "split_registry_sha256": registry_sha,
            "implementation_sha256": _implementation_sha256(),
        },
    }


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite source-robust report")
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    descriptor, name = tempfile.mkstemp(prefix=".source-robust-", dir=path.parent)
    temporary = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--palsynet-cache-root", required=True, type=Path)
    parser.add_argument("--reviewed-identity-manifest", required=True, type=Path)
    parser.add_argument("--review-ledger", required=True, type=Path)
    parser.add_argument("--split-registry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main():
    args = _parser().parse_args()
    report = run(args)
    _write(args.output, report)
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
