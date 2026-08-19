#!/usr/bin/env python3
"""Compare paper-like AU, frozen 110D, and AU+110D on NeuroFace SPREAD."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_neuroface_au_v1 import (  # noqa: E402
    COLLECTION_SCHEMA,
    select_paper_records,
)
from scripts.run_neuroface_als_temporal_v1 import (  # noqa: E402
    _json_bytes,
    _read_regular_bytes,
    _unique_rows,
)
from src.datasets.dynamic_landmark import (  # noqa: E402
    DynamicLandmarkRecording,
    load_dynamic_landmark_recording_bytes,
)
from src.datasets.neuroface_au_v1 import (  # noqa: E402
    AU_NAMES,
    NeuroFaceAURecording,
    SUMMARY_STATISTICS,
    load_au_recording_bytes,
    summarize_au_recording,
)
from src.evaluation.neuroface_als_benchmark_v1 import (  # noqa: E402
    PAPER_ACCURACY,
    PAPER_AUROC,
    Candidate,
    evaluate_nested_balanced_logistic,
    evaluate_fixed_loso,
    evaluate_nested_loso,
    evaluate_nested_loso_with_threshold,
    participant_stratified_bootstrap,
    select_paper_like_candidate,
)
from src.preprocessing.action_capacity_features_v1 import (  # noqa: E402
    PINNED_NEUROFACE_COLLECTION_MANIFEST_SHA256,
)
from src.preprocessing.generalization_110d import (  # noqa: E402
    LANDMARK_MI_110D,
    candidate_feature_vector,
)
from src.preprocessing.script_action_segmentation_v1 import (  # noqa: E402
    PINNED_NEUROFACE_PRIVATE_MANIFEST_SHA256,
)


REPRESENTATIONS = (
    "paper_pyfeat_min_au_20d",
    "paper_pyfeat_all_stats_100d",
    "landmark_110d",
    "min_au_110d_fusion_130d",
    "all_stats_au_110d_fusion_210d",
)
PRIMARY_REPRESENTATIONS = (
    REPRESENTATIONS[0], REPRESENTATIONS[2], REPRESENTATIONS[3],
)
REPORT_SCHEMA = "neuroface_als_au_geometry_public_report_v1"
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_CACHE_BYTES = 512 * 1024 * 1024


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--private-manifest", required=True, type=Path)
    result.add_argument("--dynamic-collection", required=True, type=Path)
    result.add_argument("--dynamic-cache-root", required=True, type=Path)
    result.add_argument("--au-cache-root", required=True, type=Path)
    result.add_argument("--output-root", required=True, type=Path)
    return result


def build_spread_representations(
    selected_rows: Sequence[Mapping[str, object]],
    dynamic_recordings: Mapping[str, DynamicLandmarkRecording],
    au_recordings: Mapping[str, NeuroFaceAURecording],
) -> tuple[dict[str, np.ndarray], np.ndarray, tuple[str, ...], np.ndarray]:
    """Build aligned 100D, 110D, and 210D rows for the 22-person endpoint."""
    if len(selected_rows) != 22 or set(dynamic_recordings) != set(au_recordings):
        raise ValueError("SPREAD comparison requires 22 paired AU and dynamic caches")
    if set(dynamic_recordings) != {str(row.get("recording_id")) for row in selected_rows}:
        raise ValueError("cache identities differ from selected SPREAD recordings")
    au_rows = []
    landmark_rows = []
    labels = []
    groups = []
    coverage = []
    for row in selected_rows:
        recording_id = str(row.get("recording_id"))
        group_id = row.get("participant_id")
        source_sha = row.get("video_sha256")
        cohort = row.get("cohort")
        if (row.get("task") != "NSM_SPREAD" or not isinstance(group_id, str)
                or not isinstance(source_sha, str)
                or cohort not in {"als", "healthy_control"}):
            raise ValueError("selected row differs from the frozen SPREAD endpoint")
        label = 1 if cohort == "als" else 0
        dynamic = dynamic_recordings[recording_id]
        au = au_recordings[recording_id]
        expected_identity = (recording_id, group_id, source_sha)
        if ((dynamic.recording_id, dynamic.group_id, dynamic.source_sha256) != expected_identity
                or (au.recording_id, au.group_id, au.source_sha256) != expected_identity
                or dynamic.label != label or au.task != "NSM_SPREAD"):
            raise ValueError("AU, dynamic, and private identities disagree")
        summary = summarize_au_recording(au)
        landmark = candidate_feature_vector(
            LANDMARK_MI_110D,
            dynamic.features,
            dynamic.valid_mask,
            dynamic.timestamps,
            dynamic.source_frame_indices,
        )
        au_rows.append(summary.values)
        landmark_rows.append(landmark)
        labels.append(label)
        groups.append(group_id)
        coverage.append(au.coverage)
    if len(set(groups)) != 22 or sum(labels) != 11:
        raise ValueError("SPREAD endpoint must contain 11 ALS and 11 healthy participants")
    au_matrix = np.stack(au_rows).astype(np.float64, copy=False)
    landmark_matrix = np.stack(landmark_rows).astype(np.float64, copy=False)
    statistic_width = len(AU_NAMES)
    minimum_offset = SUMMARY_STATISTICS.index("min") * statistic_width
    minimum_matrix = au_matrix[:, minimum_offset:minimum_offset + statistic_width]
    matrices = {
        REPRESENTATIONS[0]: minimum_matrix,
        REPRESENTATIONS[1]: au_matrix,
        REPRESENTATIONS[2]: landmark_matrix,
        REPRESENTATIONS[3]: np.concatenate((minimum_matrix, landmark_matrix), axis=1),
        REPRESENTATIONS[4]: np.concatenate((au_matrix, landmark_matrix), axis=1),
    }
    return (
        matrices,
        np.asarray(labels, dtype=np.int64),
        tuple(groups),
        np.asarray(coverage, dtype=np.float64),
    )


def _candidate_document(candidate) -> dict[str, object]:
    return {
        "representation": candidate.representation,
        "penalty": candidate.penalty,
        "c": candidate.c,
    }


def _write_release(
    output_root: Path,
    report: dict[str, object],
    *,
    labels: np.ndarray,
    groups: Sequence[str],
    paper_probabilities: np.ndarray,
    paper_nested_probabilities: np.ndarray,
    paper_thresholded_predictions: np.ndarray,
    paper_outer_thresholds: np.ndarray,
    balanced_probabilities: np.ndarray,
    balanced_predictions: np.ndarray,
    balanced_outer_thresholds: np.ndarray,
    nested_probabilities: Mapping[str, np.ndarray],
) -> None:
    if not output_root.is_absolute() or output_root.exists() or output_root.is_symlink():
        raise ValueError("output root must be a new absolute directory")
    output_root.mkdir(mode=0o700, parents=False)
    private = output_root / "private_oof.npz"
    np.savez_compressed(
        private,
        schema_version=np.asarray("neuroface_als_au_geometry_private_oof_v1"),
        group_ids=np.asarray(tuple(groups)),
        labels=labels,
        paper_like_au_probabilities=paper_probabilities,
        strict_nested_paper_statistic_probabilities=paper_nested_probabilities,
        strict_nested_paper_statistic_threshold_predictions=paper_thresholded_predictions,
        strict_nested_paper_statistic_outer_thresholds=paper_outer_thresholds,
        outcome_informed_balanced_probabilities=balanced_probabilities,
        outcome_informed_balanced_predictions=balanced_predictions,
        outcome_informed_balanced_outer_thresholds=balanced_outer_thresholds,
        **{f"nested__{name}": values for name, values in nested_probabilities.items()},
    )
    os.chmod(private, 0o600)
    report_payload = (json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n").encode("utf-8")
    report_path = output_root / "report.json"
    descriptor = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        if os.write(descriptor, report_payload) != len(report_payload):
            raise OSError("short report write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__).resolve(),
        PROJECT_ROOT / "scripts" / "run_neuroface_als_temporal_v1.py",
        PROJECT_ROOT / "scripts" / "extract_neuroface_au_v1.py",
        PROJECT_ROOT / "src" / "datasets" / "neuroface_au_v1.py",
        PROJECT_ROOT / "src" / "datasets" / "dynamic_landmark.py",
        PROJECT_ROOT / "src" / "evaluation" / "neuroface_als_benchmark_v1.py",
        PROJECT_ROOT / "src" / "preprocessing" / "generalization_110d.py",
        PROJECT_ROOT / "src" / "preprocessing" / "trajectory_features.py",
    ):
        payload = path.read_bytes()
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(len(payload).to_bytes(8, "big") + payload)
    return digest.hexdigest()


def main() -> int:
    args = parser().parse_args()
    for path in (
        args.private_manifest, args.dynamic_collection,
        args.dynamic_cache_root, args.au_cache_root,
    ):
        if any(token in os.fspath(path).casefold() for token in ("palsynet", "mayo", "meei")):
            raise ValueError("non-NeuroFace data are prohibited during candidate development")
    private_payload = _read_regular_bytes(args.private_manifest, maximum=_MAX_JSON_BYTES)
    dynamic_payload = _read_regular_bytes(args.dynamic_collection, maximum=_MAX_JSON_BYTES)
    if hashlib.sha256(private_payload).hexdigest() != PINNED_NEUROFACE_PRIVATE_MANIFEST_SHA256:
        raise ValueError("private manifest differs from the frozen NeuroFace inventory")
    if hashlib.sha256(dynamic_payload).hexdigest() != PINNED_NEUROFACE_COLLECTION_MANIFEST_SHA256:
        raise ValueError("dynamic collection differs from the frozen inventory")
    private = _json_bytes(private_payload)
    dynamic_collection = _json_bytes(dynamic_payload)
    selected_all = select_paper_records(private)
    selected = tuple(row for row in selected_all if row["task"] == "NSM_SPREAD")
    dynamic_rows = dynamic_collection.get("records")
    if (dynamic_collection.get("schema_version") != "neuroface_clinical23_v2_windows_v1"
            or not isinstance(dynamic_rows, list) or len(dynamic_rows) != 261):
        raise ValueError("dynamic collection is incomplete")
    dynamic_by_id = _unique_rows(dynamic_rows, name="dynamic collection")

    au_collection_path = args.au_cache_root.parent / "collection_manifest.json"
    au_collection_sha = None
    au_collection_rows = None
    if au_collection_path.exists():
        au_collection_payload = _read_regular_bytes(
            au_collection_path, maximum=_MAX_JSON_BYTES
        )
        au_collection = _json_bytes(au_collection_payload)
        if (au_collection.get("schema_version") != COLLECTION_SCHEMA
                or au_collection.get("counts") != {
                    "participants": 22,
                    "recordings": 66,
                    "als_participants": 11,
                    "healthy_participants": 11,
                    "tasks": {task: 22 for task in ("NSM_KISS", "NSM_OPEN", "NSM_SPREAD")},
                }
                or not isinstance(au_collection.get("records"), list)
                or len(au_collection["records"]) != 66):
            raise ValueError("AU collection manifest is incomplete")
        au_collection_sha = hashlib.sha256(au_collection_payload).hexdigest()
        au_collection_rows = _unique_rows(au_collection["records"], name="AU collection")

    dynamic_recordings = {}
    au_recordings = {}
    au_cache_digests = []
    for row in selected:
        recording_id = str(row["recording_id"])
        dynamic_row = dynamic_by_id.get(recording_id)
        if dynamic_row is None or dynamic_row.get("status") != "retained":
            raise ValueError("SPREAD dynamic cache is not retained")
        dynamic_cache = _read_regular_bytes(
            args.dynamic_cache_root / f"{recording_id}.npz", maximum=64 * 1024 * 1024
        )
        if hashlib.sha256(dynamic_cache).hexdigest() != dynamic_row.get("cache_sha256"):
            raise ValueError("dynamic cache differs from its collection commitment")
        dynamic_recordings[recording_id] = load_dynamic_landmark_recording_bytes(dynamic_cache)

        au_cache = _read_regular_bytes(
            args.au_cache_root / f"{recording_id}.npz", maximum=_MAX_CACHE_BYTES
        )
        au_digest = hashlib.sha256(au_cache).hexdigest()
        au_cache_digests.append(au_digest)
        if au_collection_rows is not None:
            au_row = au_collection_rows.get(recording_id)
            if au_row is None or au_row.get("cache_sha256") != au_digest:
                raise ValueError("AU cache differs from the complete collection commitment")
        au_recordings[recording_id] = load_au_recording_bytes(au_cache)

    matrices, labels, groups, coverage = build_spread_representations(
        selected, dynamic_recordings, au_recordings
    )
    all_statistics = matrices[REPRESENTATIONS[1]]
    paper_statistic_matrices = {
        f"paper_pyfeat_{statistic}_au_20d": all_statistics[
            :, index * len(AU_NAMES):(index + 1) * len(AU_NAMES)
        ]
        for index, statistic in enumerate(SUMMARY_STATISTICS)
    }
    paper_like = select_paper_like_candidate(
        paper_statistic_matrices, labels, groups
    )
    paper_nested = evaluate_nested_loso(paper_statistic_matrices, labels, groups)
    paper_thresholded = evaluate_nested_loso_with_threshold(
        paper_statistic_matrices, labels, groups
    )
    paper_balanced = evaluate_nested_balanced_logistic(
        paper_statistic_matrices, labels, groups
    )
    published_configuration = evaluate_fixed_loso(
        {REPRESENTATIONS[0]: matrices[REPRESENTATIONS[0]]},
        labels,
        groups,
        Candidate(REPRESENTATIONS[0], "l1", 1.0),
    )
    nested = {
        name: evaluate_nested_loso({name: matrices[name]}, labels, groups)
        for name in REPRESENTATIONS
    }
    strict_pipeline_metrics = {
        "paper_statistic_nested_selector_fixed_0_5": paper_nested.metrics,
        "paper_statistic_nested_selector_with_inner_threshold": paper_thresholded.metrics,
        "outcome_informed_balanced_logistic": paper_balanced.metrics,
        **{name: nested[name].metrics for name in PRIMARY_REPRESENTATIONS},
    }
    strict_pipeline_order = (
        "paper_statistic_nested_selector_fixed_0_5",
        "paper_statistic_nested_selector_with_inner_threshold",
        "outcome_informed_balanced_logistic",
        *PRIMARY_REPRESENTATIONS,
    )
    strict_best = max(
        strict_pipeline_order,
        key=lambda name: (
            strict_pipeline_metrics[name]["auroc"],
            strict_pipeline_metrics[name]["accuracy"],
            -strict_pipeline_order.index(name),
        ),
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "endpoint": "neuroface_als_vs_healthy_spread_22_participants",
        "counts": {"participants": 22, "als": 11, "healthy": 11},
        "evidence_state": (
            "complete_66_recording_au_collection"
            if au_collection_rows is not None else
            "provisional_22_spread_caches_source_and_schema_authenticated"
        ),
        "au_coverage": {
            "minimum": float(coverage.min()),
            "median": float(np.median(coverage)),
            "maximum": float(coverage.max()),
        },
        "paper_like_same_oof_descriptive": {
            "candidate": _candidate_document(paper_like.candidate),
            "metrics": paper_like.metrics,
            "selection_protocol": paper_like.selection_protocol,
        },
        "published_min_au_configuration_descriptive": {
            "candidate": _candidate_document(published_configuration.candidate),
            "metrics": published_configuration.metrics,
            "selection_protocol": published_configuration.selection_protocol,
        },
        "strict_nested_paper_statistic_search": {
            "metrics": paper_nested.metrics,
            "bootstrap": participant_stratified_bootstrap(
                labels, paper_nested.probabilities
            ),
            "selection_protocol": paper_nested.selection_protocol,
            "outer_candidate_counts": {
                json.dumps(_candidate_document(candidate), sort_keys=True): count
                for candidate, count in Counter(paper_nested.outer_candidates).items()
            },
        },
        "strict_nested_paper_statistic_search_with_inner_threshold": {
            "metrics": paper_thresholded.metrics,
            "bootstrap": participant_stratified_bootstrap(
                labels,
                paper_thresholded.probabilities,
                predictions=paper_thresholded.predictions,
            ),
            "selection_protocol": paper_thresholded.selection_protocol,
            "outer_threshold_summary": {
                "minimum": float(paper_thresholded.outer_thresholds.min()),
                "median": float(np.median(paper_thresholded.outer_thresholds)),
                "maximum": float(paper_thresholded.outer_thresholds.max()),
            },
            "outer_candidate_counts": {
                json.dumps(_candidate_document(candidate), sort_keys=True): count
                for candidate, count in Counter(paper_thresholded.outer_candidates).items()
            },
        },
        "outcome_informed_balanced_logistic": {
            "metrics": paper_balanced.metrics,
            "bootstrap": participant_stratified_bootstrap(
                labels,
                paper_balanced.probabilities,
                predictions=paper_balanced.predictions,
            ),
            "selection_protocol": paper_balanced.selection_protocol,
            "outer_candidate_counts": {
                json.dumps(_candidate_document(candidate), sort_keys=True): count
                for candidate, count in Counter(paper_balanced.outer_candidates).items()
            },
            "outer_threshold_summary": {
                "minimum": float(paper_balanced.outer_thresholds.min()),
                "median": float(np.median(paper_balanced.outer_thresholds)),
                "maximum": float(paper_balanced.outer_thresholds.max()),
            },
            "research_process_status": "outcome_informed_final_ablation",
        },
        "strict_nested_participant_loso": {
            name: {
                "metrics": nested[name].metrics,
                "selection_protocol": nested[name].selection_protocol,
                "outer_candidate_counts": {
                    json.dumps(_candidate_document(candidate), sort_keys=True): count
                    for candidate, count in Counter(nested[name].outer_candidates).items()
                },
            }
            for name in REPRESENTATIONS
        },
        "exploratory_best_locked_candidate": strict_best,
        "development_milestones": {
            "best_nested_auroc_above_0_90": bool(
                strict_pipeline_metrics[strict_best]["auroc"] > 0.90
            ),
            "best_nested_accuracy_above_paper_0_91": bool(
                strict_pipeline_metrics[strict_best]["accuracy"] > PAPER_ACCURACY
            ),
            "best_nested_auroc_above_paper_0_97": bool(
                strict_pipeline_metrics[strict_best]["auroc"] > PAPER_AUROC
            ),
        },
        "published_descriptive_comparator": {
            "accuracy": PAPER_ACCURACY,
            "auroc": PAPER_AUROC,
            "paper_candidate_selection_not_documented_as_nested": True,
        },
        "claim_boundary": {
            "internal_development_only": True,
            "post_candidate_external_validation_required": True,
            "causal_representation_comparison": False,
            "clinical_deployment_claim": False,
            "palsynet_reads": 0,
            "mayo_reads": 0,
            "meei_reads": 0,
        },
        "provenance": {
            "private_manifest_sha256": hashlib.sha256(private_payload).hexdigest(),
            "dynamic_collection_sha256": hashlib.sha256(dynamic_payload).hexdigest(),
            "au_collection_sha256": au_collection_sha,
            "spread_au_cache_set_sha256": hashlib.sha256(
                "\n".join(sorted(au_cache_digests)).encode("ascii")
            ).hexdigest(),
            "implementation_sha256": _implementation_sha256(),
        },
    }
    _write_release(
        args.output_root,
        report,
        labels=labels,
        groups=groups,
        paper_probabilities=paper_like.probabilities,
        paper_nested_probabilities=paper_nested.probabilities,
        paper_thresholded_predictions=paper_thresholded.predictions,
        paper_outer_thresholds=paper_thresholded.outer_thresholds,
        balanced_probabilities=paper_balanced.probabilities,
        balanced_predictions=paper_balanced.predictions,
        balanced_outer_thresholds=paper_balanced.outer_thresholds,
        nested_probabilities={name: value.probabilities for name, value in nested.items()},
    )
    print(json.dumps({
        "schema_version": "neuroface_als_au_geometry_receipt_v1",
        "report_sha256": hashlib.sha256((args.output_root / "report.json").read_bytes()).hexdigest(),
        "evidence_state": report["evidence_state"],
        "best": strict_best,
        "metrics": strict_pipeline_metrics[strict_best],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
