#!/usr/bin/env python3
"""Compare four-time and seven-action Landmark 110D on PalsyNet development."""
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

from scripts.build_palsynet_action_aligned_v1 import (  # noqa: E402
    load_action_aligned_cache,
)
from scripts.freeze_palsynet_person_split_registry import (  # noqa: E402
    validate_person_split_registry,
)
from src.datasets.dynamic_landmark import load_dynamic_landmark_recordings  # noqa: E402
from src.evaluation.action_aligned_110d_v1 import (  # noqa: E402
    CANDIDATE_ORDER,
    FIXED_C,
    FIXED_THRESHOLD,
    choose_locked_candidate,
    run_group_disjoint_oof,
)
from src.preprocessing.action_aligned_110d import (  # noqa: E402
    action_aligned_feature_vector,
    mirror_action_aligned_features,
    mirror_clinical23_features,
)
from src.preprocessing.trajectory_features import trajectory_feature_set  # noqa: E402


BOOTSTRAP_REPEATS = 5000
BOOTSTRAP_SEED = 20260811
REPORT_SCHEMA = "action_aligned_110d_v1_development_report"
_IMPLEMENTATION_FILES = (
    Path(__file__).resolve(),
    PROJECT_ROOT / "src/evaluation/action_aligned_110d_v1.py",
    PROJECT_ROOT / "src/preprocessing/action_aligned_110d.py",
    PROJECT_ROOT / "src/preprocessing/trajectory_features.py",
    PROJECT_ROOT / "scripts/build_palsynet_action_aligned_v1.py",
)


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in _IMPLEMENTATION_FILES:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _read_json(path: Path, name: str) -> tuple[dict[str, object], str]:
    raw = path.read_bytes()
    def unique(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"{name} contains duplicate JSON keys")
            output[key] = value
        return output
    payload = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload, hashlib.sha256(raw).hexdigest()


def _one_baseline(path: Path):
    records = load_dynamic_landmark_recordings([path])
    if len(records) != 1:
        raise ValueError("baseline cache must contain one recording")
    return records[0]


def _metrics(labels: np.ndarray, groups: np.ndarray, probabilities: np.ndarray):
    ordered = sorted(set(groups.tolist()), key=str)
    grouped_labels = []
    grouped_scores = []
    for group in ordered:
        indices = np.flatnonzero(groups == group)
        observed = np.unique(labels[indices])
        if observed.size != 1:
            raise ValueError("one group cannot cross labels")
        grouped_labels.append(int(observed[0]))
        grouped_scores.append(float(np.mean(probabilities[indices])))
    y = np.asarray(grouped_labels, dtype=np.int64)
    scores = np.asarray(grouped_scores, dtype=np.float64)
    prediction = scores >= FIXED_THRESHOLD
    return {
        "auroc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "brier": float(brier_score_loss(y, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "sensitivity": float(np.mean(prediction[y == 1])),
        "specificity": float(np.mean(~prediction[y == 0])),
    }, y, scores


def _bootstrap(labels: np.ndarray, baseline: np.ndarray, action: np.ndarray):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    class_indices = {value: np.flatnonzero(labels == value) for value in (0, 1)}
    deltas = {name: np.empty(BOOTSTRAP_REPEATS) for name in (
        "auroc", "balanced_accuracy", "brier",
    )}
    for repeat in range(BOOTSTRAP_REPEATS):
        draw = np.concatenate([
            rng.choice(indices, size=indices.size, replace=True)
            for indices in class_indices.values()
        ])
        y = labels[draw]
        b = baseline[draw]
        a = action[draw]
        deltas["auroc"][repeat] = roc_auc_score(y, a) - roc_auc_score(y, b)
        deltas["balanced_accuracy"][repeat] = (
            balanced_accuracy_score(y, a >= 0.5)
            - balanced_accuracy_score(y, b >= 0.5)
        )
        deltas["brier"][repeat] = brier_score_loss(y, a) - brier_score_loss(y, b)
    point_metrics = {
        "auroc": float(roc_auc_score(labels, action) - roc_auc_score(labels, baseline)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, action >= 0.5)
            - balanced_accuracy_score(labels, baseline >= 0.5)
        ),
        "brier": float(
            brier_score_loss(labels, action) - brier_score_loss(labels, baseline)
        ),
    }
    return {
        name: {
            "point": point_metrics[name],
            "ci95": [float(x) for x in np.quantile(values, (0.025, 0.975))],
        }
        for name, values in deltas.items()
    }


def run(args) -> dict[str, object]:
    reviewed, reviewed_sha = _read_json(args.reviewed_identity_manifest, "reviewed manifest")
    ledger, ledger_sha = _read_json(args.review_ledger, "review ledger")
    registry, registry_sha = _read_json(args.split_registry, "split registry")
    if (
        registry.get("reviewed_manifest_sha256") != reviewed_sha
        or registry.get("review_ledger_sha256") != ledger_sha
    ):
        raise ValueError("split registry does not bind supplied identity artifacts")
    validate_person_split_registry(registry, reviewed, ledger)
    reviewed_rows = {
        row["recording_id"]: row for row in reviewed["recordings"]
        if row.get("training_eligible") is True
    }
    development = [
        row for row in registry["assignments"] if row["partition"] == "development"
    ]
    protected = [row for row in registry["assignments"] if row["partition"] == "protected"]
    baseline_manifest, _baseline_manifest_sha = _read_json(
        args.baseline_cache_root / "collection_manifest.json", "baseline collection manifest"
    )
    baseline_rows = baseline_manifest.get("records")
    if (
        baseline_manifest.get("schema_version") != "palsynet_clinical23_v2_windows_v1"
        or not isinstance(baseline_rows, list) or len(baseline_rows) != 49
    ):
        raise ValueError("baseline collection manifest contract drifted")
    baseline_by_source: dict[str, Path] = {}
    expected_baseline_files: set[Path] = set()
    for row in baseline_rows:
        if not isinstance(row, dict):
            raise ValueError("baseline collection row is invalid")
        source_sha = str(row.get("source_sha256", ""))
        recording_id = str(row.get("recording_id", ""))
        path = args.baseline_cache_root / f"{recording_id}.npz"
        if source_sha in baseline_by_source or not path.is_file():
            raise ValueError("baseline collection hash/path coverage is invalid")
        baseline_by_source[source_sha] = path
        expected_baseline_files.add(path)
    if set(args.baseline_cache_root.glob("*.npz")) != expected_baseline_files:
        raise ValueError("baseline cache files differ from collection manifest")
    action_files = set(args.action_cache_root.glob("*.npz"))
    expected_action_files = {
        args.action_cache_root / f"{row['recording_id']}.npz" for row in development
    }
    if action_files != expected_action_files:
        raise ValueError("action cache must contain the exact development partition only")

    originals = {candidate: [] for candidate in CANDIDATE_ORDER}
    mirrors = {candidate: [] for candidate in CANDIDATE_ORDER}
    labels, groups, folds, coverages = [], [], [], []
    cache_digest = hashlib.sha256()
    for assignment in development:
        recording_id = assignment["recording_id"]
        identity = reviewed_rows.get(recording_id)
        if identity is None or identity["group_id"] != assignment["group_id"]:
            raise ValueError("development assignment differs from reviewed identity")
        baseline_path = baseline_by_source.get(str(identity["source_sha256"]))
        if baseline_path is None:
            raise ValueError("reviewed source bytes are absent from baseline cache")
        action_path = args.action_cache_root / f"{recording_id}.npz"
        baseline = _one_baseline(baseline_path)
        action = load_action_aligned_cache(action_path)
        expected_label = 1 if identity["label"] == "affected" else 0
        if (
            baseline.source_sha256 != identity["source_sha256"]
            or baseline.label != expected_label
            or action.binding.recording_id != recording_id
            or action.binding.group_id != identity["group_id"]
            or action.binding.source_sha256 != identity["source_sha256"]
            or (1 if action.binding.label == "affected" else 0) != expected_label
        ):
            raise ValueError("cache provenance differs from reviewed identity")

        base_temporal = (
            baseline.valid_mask, baseline.timestamps, baseline.source_frame_indices,
        )
        originals[CANDIDATE_ORDER[0]].append(
            trajectory_feature_set("landmark", baseline.features, *base_temporal)
        )
        mirrors[CANDIDATE_ORDER[0]].append(trajectory_feature_set(
            "landmark", mirror_clinical23_features(baseline.features), *base_temporal
        ))
        originals[CANDIDATE_ORDER[1]].append(action_aligned_feature_vector(
            action.features, action.valid_mask, action.timestamps,
            action.source_frame_indices,
        ))
        mirrors[CANDIDATE_ORDER[1]].append(action_aligned_feature_vector(
            mirror_action_aligned_features(action.features), action.valid_mask,
            action.timestamps, action.source_frame_indices,
        ))
        labels.append(expected_label)
        groups.append(identity["group_id"])
        folds.append(assignment["inner_fold"])
        coverages.append(action.coverage)
        cache_digest.update(hashlib.sha256(action_path.read_bytes()).digest())

    original_matrices = {
        key: np.stack(value).astype(np.float64, copy=False)
        for key, value in originals.items()
    }
    mirror_matrices = {
        key: np.stack(value).astype(np.float64, copy=False)
        for key, value in mirrors.items()
    }
    y = np.asarray(labels, dtype=np.int64)
    group_array = np.asarray(groups)
    fold_array = np.asarray(folds, dtype=np.int64)
    oof = run_group_disjoint_oof(
        labels=y, group_ids=group_array, inner_folds=fold_array,
        original=original_matrices, mirrored=mirror_matrices,
    )
    metric_report = {}
    grouped = {}
    grouped_labels = None
    for candidate in CANDIDATE_ORDER:
        metric_report[candidate], candidate_labels, candidate_scores = _metrics(
            y, group_array, oof.probabilities[candidate]
        )
        if grouped_labels is None:
            grouped_labels = candidate_labels
        elif not np.array_equal(grouped_labels, candidate_labels):
            raise RuntimeError("candidate group labels are unaligned")
        grouped[candidate] = candidate_scores
    assert grouped_labels is not None
    locked = choose_locked_candidate(metric_report)
    return {
        "schema_version": REPORT_SCHEMA,
        "claim_scope": "identity_reviewed_palsynet_development_group_oof_only",
        "target": "binary_affected_vs_unaffected_not_hb_grade",
        "protocol": {
            "candidates": list(CANDIDATE_ORDER),
            "candidate_dimensions": {name: 110 for name in CANDIDATE_ORDER},
            "inner_folds": 4,
            "model": {
                "type": "standardized_l2_logistic_regression",
                "c": FIXED_C,
                "threshold": FIXED_THRESHOLD,
                "training_augmentation": "original_plus_horizontal_mirror",
                "validation_inference": "mean_original_and_horizontal_mirror_probability",
                "hyperparameter_search": False,
            },
            "action_windows": {
                "proposal_rate_hz": 6.0,
                "slots": 7,
                "frames_per_slot": 32,
                "selection_uses_labels_or_classifier_scores": False,
            },
        },
        "provenance": {
            "reviewed_manifest_sha256": reviewed_sha,
            "review_ledger_sha256": ledger_sha,
            "split_registry_sha256": registry_sha,
            "action_development_cache_aggregate_sha256": cache_digest.hexdigest(),
            "implementation_sha256": _implementation_sha256(),
        },
        "counts": {
            "development_recordings": len(development),
            "development_groups": len(set(groups)),
            "protected_recordings": len(protected),
            "protected_groups": len({row["group_id"] for row in protected}),
        },
        "action_cache": {
            "minimum_coverage": float(np.min(coverages)),
            "mean_coverage": float(np.mean(coverages)),
        },
        "metrics": metric_report,
        "paired_action_minus_baseline": _bootstrap(
            grouped_labels, grouped[CANDIDATE_ORDER[0]], grouped[CANDIDATE_ORDER[1]]
        ),
        "audit": {
            **oof.audit,
            "development_baseline_cache_reads": len(development),
            "development_action_cache_reads": len(development),
        },
        "decision": {
            "locked_candidate": locked,
            "action_promoted": locked == CANDIDATE_ORDER[1],
            "mayo_evaluation_authorized": locked == CANDIDATE_ORDER[1],
            "outer_test_access_authorized": False,
            "hb_claim_authorized": False,
            "clinical_validation": False,
        },
    }


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite action-aligned report")
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd, name = tempfile.mkstemp(prefix=".action-report-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
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


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-cache-root", required=True, type=Path)
    parser.add_argument("--action-cache-root", required=True, type=Path)
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
