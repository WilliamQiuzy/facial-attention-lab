#!/usr/bin/env python3
"""Run the post-lock Mayo challenge for seven-action Landmark 110D."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_palsynet_action_aligned_v1 import (  # noqa: E402
    load_action_aligned_cache,
)
from scripts.run_110d_generalization_v1 import (  # noqa: E402
    GateAudit,
    _build_cache_metadata_dataset,
    _read_json,
    load_development_cache_records,
    validate_development_gate,
)
from scripts.run_architecture_search_v1 import _prepare_search_dataset  # noqa: E402
from scripts.run_mayo_positive_challenge_v1 import _load_mayo_features  # noqa: E402
from src.evaluation.mayo_positive_challenge_v1 import (  # noqa: E402
    fit_frozen_110d_champion,
    positive_cohort_summary,
    predict_mirror_mean,
)
from src.preprocessing.action_aligned_110d import (  # noqa: E402
    action_aligned_feature_vector,
    mirror_action_aligned_features,
)


SCHEMA = "mayo_action_aligned_110d_v1_report"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _action_matrices(
    action_root: Path,
    ordered_rows: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected_files = {
        action_root / f"{row['recording_id']}.npz" for row in ordered_rows
    }
    if set(action_root.glob("*.npz")) != expected_files:
        raise ValueError("action cache files differ from the locked ordered rows")
    original, mirrored, coverage = [], [], []
    for row in ordered_rows:
        recording_id = str(row["recording_id"])
        record = load_action_aligned_cache(action_root / f"{recording_id}.npz")
        expected_label = str(row.get("label"))
        if (
            record.binding.recording_id != recording_id
            or record.binding.group_id != row.get("group_id")
            or record.binding.source_sha256 != row.get("source_sha256")
            or record.binding.label != expected_label
        ):
            raise ValueError("action cache provenance differs from locked row")
        temporal = (
            record.valid_mask, record.timestamps, record.source_frame_indices,
        )
        original.append(action_aligned_feature_vector(record.features, *temporal))
        mirrored.append(action_aligned_feature_vector(
            mirror_action_aligned_features(record.features), *temporal
        ))
        coverage.append(record.coverage)
    return (
        np.stack(original).astype(np.float64, copy=False),
        np.stack(mirrored).astype(np.float64, copy=False),
        np.asarray(coverage, dtype=np.float64),
    )


def _paired_transition(
    baseline: np.ndarray,
    action: np.ndarray,
) -> dict[str, object]:
    first = np.asarray(baseline, dtype=np.float64)
    second = np.asarray(action, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1 or first.size != 47:
        raise ValueError("Mayo paired scores must align for 47 records")
    base_call = first >= 0.5
    action_call = second >= 0.5
    delta = second - first
    return {
        "both_positive": int(np.sum(base_call & action_call)),
        "baseline_negative_to_action_positive": int(np.sum(~base_call & action_call)),
        "baseline_positive_to_action_negative": int(np.sum(base_call & ~action_call)),
        "both_negative": int(np.sum(~base_call & ~action_call)),
        "score_delta": {
            "minimum": float(np.min(delta)),
            "q25": float(np.quantile(delta, 0.25)),
            "median": float(np.median(delta)),
            "mean": float(np.mean(delta)),
            "q75": float(np.quantile(delta, 0.75)),
            "maximum": float(np.max(delta)),
        },
    }


def run(args) -> dict[str, object]:
    development_report = json.loads(args.development_report.read_text())
    if (
        development_report.get("schema_version")
        != "action_aligned_110d_v1_development_report"
        or development_report.get("decision", {}).get("locked_candidate")
        != "seven_action_window_110d"
        or development_report.get("decision", {}).get("mayo_evaluation_authorized")
        is not True
        or development_report.get("audit", {}).get("protected_feature_reads") != 0
    ):
        raise ValueError("Mayo action challenge is not authorized by the locked development report")

    reviewed, reviewed_sha = _read_json(args.reviewed_identity_manifest)
    ledger, ledger_sha = _read_json(args.review_ledger)
    registry, registry_sha = _read_json(args.split_registry)
    dataset, collection_rows, collection_sha = _build_cache_metadata_dataset(
        args.palsynet_baseline_cache_root
    )
    audit = GateAudit()
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
        audit=audit,
    )
    load_development_cache_records(
        args.palsynet_baseline_cache_root, dataset, gate, collection_rows,
        audit=audit,
    )
    prepared = _prepare_search_dataset(dataset, gate)
    development = prepared.development_indices
    baseline_champion = fit_frozen_110d_champion(
        prepared.summary_features[development],
        prepared.mirrored_summary_features[development],
        prepared.labels[development],
        prepared.group_ids[development],
    )

    reviewed_by_id = {
        row["recording_id"]: row for row in reviewed["recordings"]
        if row.get("training_eligible") is True
    }
    development_assignments = [
        row for row in registry["assignments"] if row["partition"] == "development"
    ]
    development_rows = [reviewed_by_id[row["recording_id"]]
                        for row in development_assignments]
    action_train_original, action_train_mirror, _ = _action_matrices(
        args.palsynet_action_cache_root, development_rows
    )
    action_labels = np.asarray([
        1 if row["label"] == "affected" else 0 for row in development_rows
    ], dtype=np.int64)
    action_groups = np.asarray([row["group_id"] for row in development_rows])
    action_champion = fit_frozen_110d_champion(
        action_train_original, action_train_mirror, action_labels, action_groups
    )

    mayo_original, mayo_mirror, baseline_coverage, mayo_baseline_sha, counts = (
        _load_mayo_features(args.mayo_baseline_cache_root)
    )
    baseline_probabilities = predict_mirror_mean(
        baseline_champion, mayo_original, mayo_mirror
    )
    mayo_manifest = json.loads(
        (args.mayo_baseline_cache_root / "collection_manifest.json").read_text()
    )
    mayo_rows = mayo_manifest.get("records")
    if not isinstance(mayo_rows, list) or len(mayo_rows) != 47:
        raise ValueError("Mayo baseline cache must contain 47 ordered rows")
    action_original, action_mirror, action_coverage = _action_matrices(
        args.mayo_action_cache_root, mayo_rows
    )
    action_probabilities = predict_mirror_mean(
        action_champion, action_original, action_mirror
    )
    baseline_summary = positive_cohort_summary(
        baseline_probabilities, baseline_coverage
    )
    action_summary = positive_cohort_summary(action_probabilities, action_coverage)
    if audit.protected_cache_records_loaded != 0:
        raise AssertionError("protected PalsyNet cache was opened")
    return {
        "schema_version": SCHEMA,
        "claim_scope": {
            "selection": "identity_reviewed_palsynet_development_oof_only",
            "mayo": "assumed_positive_confidence_challenge_only",
        },
        "target": "binary_affected_vs_unaffected_not_hb_grade",
        "counts": {
            "palsynet_development_recordings": 39,
            "palsynet_development_groups": 38,
            "mayo_scored_contents": 47,
            "mayo_verified_negative_records": 0,
        },
        "protocol": {
            "baseline": "four_time_window_landmark_110d",
            "selected": "seven_action_window_landmark_110d",
            "classifier": "standardized_l2_logistic_c_0_01",
            "threshold": 0.5,
            "mayo_used_for_model_selection": False,
            "threshold_tuned_on_mayo": False,
        },
        "mayo_four_time_window_baseline": baseline_summary,
        "mayo_seven_action_window_selected": action_summary,
        "paired_transition": _paired_transition(
            baseline_probabilities, action_probabilities
        ),
        "audit": {
            "palsynet_protected_cache_records_loaded": 0,
            "palsynet_protected_predictions": 0,
            "per_record_probabilities_exported": 0,
            "raw_mayo_videos_uploaded": 0,
        },
        "decision": {
            "development_champion": "seven_action_window_110d",
            "mayo_used_for_model_selection": False,
            "accuracy_defined": False,
            "auroc_defined_on_mayo": False,
            "clinical_validation": False,
            "hb_claim_authorized": False,
            "outer_evaluation_authorized": False,
        },
        "provenance": {
            "development_report_sha256": _sha256(args.development_report),
            "palsynet_reviewed_manifest_sha256": reviewed_sha,
            "palsynet_review_ledger_sha256": ledger_sha,
            "palsynet_split_registry_sha256": registry_sha,
            "mayo_baseline_manifest_sha256": mayo_baseline_sha,
            "mayo_action_manifest_sha256": _sha256(
                args.mayo_action_cache_root / "collection_manifest.json"
            ),
        },
    }


def _write(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite Mayo action report")
    encoded = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd, name = tempfile.mkstemp(prefix=".mayo-action-report-", dir=path.parent)
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
    parser.add_argument("--development-report", required=True, type=Path)
    parser.add_argument("--palsynet-baseline-cache-root", required=True, type=Path)
    parser.add_argument("--palsynet-action-cache-root", required=True, type=Path)
    parser.add_argument("--reviewed-identity-manifest", required=True, type=Path)
    parser.add_argument("--review-ledger", required=True, type=Path)
    parser.add_argument("--split-registry", required=True, type=Path)
    parser.add_argument("--mayo-baseline-cache-root", required=True, type=Path)
    parser.add_argument("--mayo-action-cache-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main():
    args = _parser().parse_args()
    report = run(args)
    _write(args.output, report)
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
