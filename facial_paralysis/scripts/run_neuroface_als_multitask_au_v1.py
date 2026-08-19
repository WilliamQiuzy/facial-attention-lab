#!/usr/bin/env python3
"""Evaluate participant-level KISS/OPEN/SPREAD AU representations on NeuroFace."""
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
from src.datasets.neuroface_au_v1 import (  # noqa: E402
    AU_NAMES,
    PAPER_TASKS,
    SUMMARY_STATISTICS,
    NeuroFaceAURecording,
    load_au_recording_bytes,
    summarize_au_recording,
)
from src.evaluation.neuroface_als_benchmark_v1 import (  # noqa: E402
    PAPER_ACCURACY,
    PAPER_AUROC,
    evaluate_nested_loso,
    evaluate_nested_loso_with_threshold,
    evaluate_nested_shrinkage_lda,
    participant_stratified_bootstrap,
)
from src.preprocessing.script_action_segmentation_v1 import (  # noqa: E402
    PINNED_NEUROFACE_PRIVATE_MANIFEST_SHA256,
)


TASK_STRUCTURED_REPRESENTATIONS = tuple(
    f"three_task_{statistic}_au_60d" for statistic in SUMMARY_STATISTICS
)
ALL_STATS_REPRESENTATION = "three_task_all_stats_au_300d"
REPORT_SCHEMA = "neuroface_als_multitask_au_public_report_v1"
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_CACHE_BYTES = 512 * 1024 * 1024


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--private-manifest", required=True, type=Path)
    result.add_argument("--au-cache-root", required=True, type=Path)
    result.add_argument("--output-root", required=True, type=Path)
    return result


def build_multitask_representations(
    selected_rows: Sequence[Mapping[str, object]],
    recordings: Mapping[str, NeuroFaceAURecording],
) -> tuple[dict[str, np.ndarray], np.ndarray, tuple[str, ...], np.ndarray]:
    """Concatenate three named task summaries while retaining one row per person."""
    if (len(selected_rows) != 66 or not isinstance(recordings, Mapping)
            or set(recordings) != {str(row.get("recording_id")) for row in selected_rows}):
        raise ValueError("multi-task endpoint requires the exact 66 authenticated recordings")
    participant_order = []
    summaries: dict[str, dict[str, np.ndarray]] = {}
    labels = {}
    coverage: dict[str, dict[str, float]] = {}
    for row in selected_rows:
        recording_id = str(row.get("recording_id"))
        group_id = row.get("participant_id")
        source_sha = row.get("video_sha256")
        cohort = row.get("cohort")
        task = row.get("task")
        if (not isinstance(group_id, str) or not isinstance(source_sha, str)
                or cohort not in {"als", "healthy_control"} or task not in PAPER_TASKS):
            raise ValueError("private row differs from the frozen multi-task endpoint")
        label = 1 if cohort == "als" else 0
        recording = recordings[recording_id]
        if ((recording.recording_id, recording.group_id, recording.source_sha256,
             recording.task) != (recording_id, group_id, source_sha, task)):
            raise ValueError("AU cache identity differs from the private endpoint")
        if group_id not in summaries:
            participant_order.append(group_id)
            summaries[group_id] = {}
            coverage[group_id] = {}
            labels[group_id] = label
        if labels[group_id] != label or task in summaries[group_id]:
            raise ValueError("participant cohort or task identity is inconsistent")
        summaries[group_id][str(task)] = summarize_au_recording(recording).values
        coverage[group_id][str(task)] = recording.coverage
    if (len(participant_order) != 22 or sum(labels.values()) != 11
            or any(set(value) != set(PAPER_TASKS) for value in summaries.values())):
        raise ValueError("endpoint must contain 11 ALS and 11 healthy complete-task people")
    matrices = {}
    width = len(AU_NAMES)
    for statistic_index, representation in enumerate(TASK_STRUCTURED_REPRESENTATIONS):
        start = statistic_index * width
        stop = start + width
        matrices[representation] = np.stack([
            np.concatenate([
                summaries[group][task][start:stop] for task in PAPER_TASKS
            ])
            for group in participant_order
        ]).astype(np.float64, copy=False)
    matrices[ALL_STATS_REPRESENTATION] = np.stack([
        np.concatenate([summaries[group][task] for task in PAPER_TASKS])
        for group in participant_order
    ]).astype(np.float64, copy=False)
    coverage_matrix = np.asarray([
        [coverage[group][task] for task in PAPER_TASKS]
        for group in participant_order
    ], dtype=np.float64)
    return (
        matrices,
        np.asarray([labels[group] for group in participant_order], dtype=np.int64),
        tuple(participant_order),
        coverage_matrix,
    )


def _candidate_counts(candidates) -> dict[str, int]:
    return {
        json.dumps({
            "representation": candidate.representation,
            "penalty": candidate.penalty,
            "c": candidate.c,
        }, sort_keys=True): count
        for candidate, count in Counter(candidates).items()
    }


def _write_release(output_root: Path, report: dict[str, object], *, labels,
                   groups, statistic_result, all_stats_result, lda_result) -> None:
    if not output_root.is_absolute() or output_root.exists() or output_root.is_symlink():
        raise ValueError("output root must be a new absolute directory")
    output_root.mkdir(mode=0o700, parents=False)
    private = output_root / "private_oof.npz"
    np.savez_compressed(
        private,
        schema_version=np.asarray("neuroface_als_multitask_au_private_oof_v1"),
        group_ids=np.asarray(tuple(groups)),
        labels=labels,
        statistic_probabilities=statistic_result.probabilities,
        statistic_predictions=statistic_result.predictions,
        statistic_outer_thresholds=statistic_result.outer_thresholds,
        all_stats_probabilities=all_stats_result.probabilities,
        all_stats_predictions=all_stats_result.predictions,
        all_stats_outer_thresholds=all_stats_result.outer_thresholds,
        shrinkage_lda_probabilities=lda_result.probabilities,
        shrinkage_lda_predictions=lda_result.predictions,
        shrinkage_lda_outer_thresholds=lda_result.outer_thresholds,
    )
    os.chmod(private, 0o600)
    payload = (json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n").encode("utf-8")
    report_path = output_root / "report.json"
    descriptor = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        if os.write(descriptor, payload) != len(payload):
            raise OSError("short report write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__).resolve(),
        PROJECT_ROOT / "scripts" / "extract_neuroface_au_v1.py",
        PROJECT_ROOT / "src" / "datasets" / "neuroface_au_v1.py",
        PROJECT_ROOT / "src" / "evaluation" / "neuroface_als_benchmark_v1.py",
    ):
        payload = path.read_bytes()
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(len(payload).to_bytes(8, "big") + payload)
    return digest.hexdigest()


def main() -> int:
    args = parser().parse_args()
    for path in (args.private_manifest, args.au_cache_root):
        if any(token in os.fspath(path).casefold() for token in ("palsynet", "mayo", "meei")):
            raise ValueError("non-NeuroFace data are prohibited during candidate development")
    private_payload = _read_regular_bytes(args.private_manifest, maximum=_MAX_JSON_BYTES)
    if hashlib.sha256(private_payload).hexdigest() != PINNED_NEUROFACE_PRIVATE_MANIFEST_SHA256:
        raise ValueError("private manifest differs from the frozen NeuroFace inventory")
    selected = select_paper_records(_json_bytes(private_payload))
    collection_path = args.au_cache_root.parent / "collection_manifest.json"
    collection_payload = _read_regular_bytes(collection_path, maximum=_MAX_JSON_BYTES)
    collection = _json_bytes(collection_payload)
    expected_counts = {
        "participants": 22,
        "recordings": 66,
        "als_participants": 11,
        "healthy_participants": 11,
        "tasks": {task: 22 for task in PAPER_TASKS},
    }
    if (collection.get("schema_version") != COLLECTION_SCHEMA
            or collection.get("private_manifest_sha256") != PINNED_NEUROFACE_PRIVATE_MANIFEST_SHA256
            or collection.get("counts") != expected_counts
            or not isinstance(collection.get("records"), list)
            or len(collection["records"]) != 66):
        raise ValueError("complete AU collection manifest is required")
    collection_rows = _unique_rows(collection["records"], name="AU collection")
    recordings = {}
    for row in selected:
        recording_id = str(row["recording_id"])
        collection_row = collection_rows.get(recording_id)
        if collection_row is None or not isinstance(collection_row.get("cache_sha256"), str):
            raise ValueError("selected recording is absent from the AU collection")
        cache_payload = _read_regular_bytes(
            args.au_cache_root / f"{recording_id}.npz", maximum=_MAX_CACHE_BYTES
        )
        if hashlib.sha256(cache_payload).hexdigest() != collection_row["cache_sha256"]:
            raise ValueError("AU cache differs from the complete collection commitment")
        recordings[recording_id] = load_au_recording_bytes(cache_payload)
    matrices, labels, groups, coverage = build_multitask_representations(
        selected, recordings
    )
    statistic_matrices = {
        name: matrices[name] for name in TASK_STRUCTURED_REPRESENTATIONS
    }
    statistic_fixed = evaluate_nested_loso(statistic_matrices, labels, groups)
    statistic_thresholded = evaluate_nested_loso_with_threshold(
        statistic_matrices, labels, groups
    )
    all_stats_thresholded = evaluate_nested_loso_with_threshold(
        {ALL_STATS_REPRESENTATION: matrices[ALL_STATS_REPRESENTATION]}, labels, groups
    )
    shrinkage_lda = evaluate_nested_shrinkage_lda(
        statistic_matrices, labels, groups
    )
    best_name, best_metrics = max((
        ("nested_task_statistic_selector_with_inner_threshold", statistic_thresholded.metrics),
        ("nested_all_stats_300d_with_inner_threshold", all_stats_thresholded.metrics),
        ("nested_task_statistic_shrinkage_lda", shrinkage_lda.metrics),
    ), key=lambda item: (item[1]["auroc"], item[1]["accuracy"], item[0]))
    report = {
        "schema_version": REPORT_SCHEMA,
        "endpoint": "neuroface_als_vs_healthy_three_tasks_22_participants",
        "counts": {"participants": 22, "als": 11, "healthy": 11, "recordings": 66},
        "au_coverage": {
            "minimum": float(coverage.min()),
            "median": float(np.median(coverage)),
            "maximum": float(coverage.max()),
        },
        "nested_task_statistic_selector_fixed_0_5": {
            "metrics": statistic_fixed.metrics,
            "selection_protocol": statistic_fixed.selection_protocol,
            "outer_candidate_counts": _candidate_counts(statistic_fixed.outer_candidates),
        },
        "nested_task_statistic_selector_with_inner_threshold": {
            "metrics": statistic_thresholded.metrics,
            "bootstrap": participant_stratified_bootstrap(
                labels,
                statistic_thresholded.probabilities,
                predictions=statistic_thresholded.predictions,
            ),
            "selection_protocol": statistic_thresholded.selection_protocol,
            "outer_candidate_counts": _candidate_counts(statistic_thresholded.outer_candidates),
            "outer_threshold_summary": {
                "minimum": float(statistic_thresholded.outer_thresholds.min()),
                "median": float(np.median(statistic_thresholded.outer_thresholds)),
                "maximum": float(statistic_thresholded.outer_thresholds.max()),
            },
        },
        "nested_all_stats_300d_with_inner_threshold": {
            "metrics": all_stats_thresholded.metrics,
            "bootstrap": participant_stratified_bootstrap(
                labels,
                all_stats_thresholded.probabilities,
                predictions=all_stats_thresholded.predictions,
            ),
            "selection_protocol": all_stats_thresholded.selection_protocol,
            "outer_candidate_counts": _candidate_counts(all_stats_thresholded.outer_candidates),
            "outer_threshold_summary": {
                "minimum": float(all_stats_thresholded.outer_thresholds.min()),
                "median": float(np.median(all_stats_thresholded.outer_thresholds)),
                "maximum": float(all_stats_thresholded.outer_thresholds.max()),
            },
        },
        "nested_task_statistic_shrinkage_lda": {
            "metrics": shrinkage_lda.metrics,
            "bootstrap": participant_stratified_bootstrap(
                labels,
                shrinkage_lda.probabilities,
                predictions=shrinkage_lda.predictions,
            ),
            "selection_protocol": shrinkage_lda.selection_protocol,
            "outer_representation_counts": dict(Counter(
                shrinkage_lda.outer_representations
            )),
            "outer_threshold_summary": {
                "minimum": float(shrinkage_lda.outer_thresholds.min()),
                "median": float(np.median(shrinkage_lda.outer_thresholds)),
                "maximum": float(shrinkage_lda.outer_thresholds.max()),
            },
        },
        "locked_best_pipeline": best_name,
        "development_milestones": {
            "best_accuracy_above_0_90": bool(best_metrics["accuracy"] > 0.90),
            "best_accuracy_above_paper_0_91": bool(best_metrics["accuracy"] > PAPER_ACCURACY),
            "best_auroc_above_paper_0_97": bool(best_metrics["auroc"] > PAPER_AUROC),
        },
        "published_descriptive_comparator": {
            "single_spread_input": True,
            "accuracy": PAPER_ACCURACY,
            "auroc": PAPER_AUROC,
            "protocol_identical_to_three_task_pipeline": False,
        },
        "claim_boundary": {
            "internal_development_only": True,
            "external_validation": False,
            "same_participants_previously_explored": True,
            "clinical_deployment_claim": False,
            "palsynet_reads": 0,
            "mayo_reads": 0,
            "meei_reads": 0,
        },
        "provenance": {
            "private_manifest_sha256": hashlib.sha256(private_payload).hexdigest(),
            "au_collection_sha256": hashlib.sha256(collection_payload).hexdigest(),
            "implementation_sha256": _implementation_sha256(),
        },
    }
    _write_release(
        args.output_root, report, labels=labels, groups=groups,
        statistic_result=statistic_thresholded,
        all_stats_result=all_stats_thresholded,
        lda_result=shrinkage_lda,
    )
    print(json.dumps({
        "schema_version": "neuroface_als_multitask_au_receipt_v1",
        "report_sha256": hashlib.sha256((args.output_root / "report.json").read_bytes()).hexdigest(),
        "best": best_name,
        "metrics": best_metrics,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
