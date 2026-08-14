#!/usr/bin/env python3
"""Run the frozen participant-level NeuroFace ALS task-aware TCN on H200."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_neuroface_au_v1 import select_paper_records  # noqa: E402
from src.datasets.dynamic_landmark import (  # noqa: E402
    DynamicLandmarkRecording,
    load_dynamic_landmark_recording_bytes,
)
from src.evaluation.neuroface_als_benchmark_v1 import (  # noqa: E402
    PAPER_ACCURACY,
    PAPER_AUROC,
    recompute_binary_metrics,
)
from src.evaluation.neuroface_als_temporal_v1 import (  # noqa: E402
    FROZEN_EPOCHS,
    FROZEN_SEEDS,
    TemporalDataset,
    evaluate_temporal_loso,
    validate_temporal_dataset,
)
from src.models.neuroface_als_temporal_v1 import (  # noqa: E402
    PRIMARY_TASKS,
    count_parameters,
    TaskAwareTemporalALSClassifier,
)
from src.preprocessing.action_capacity_features_v1 import (  # noqa: E402
    PINNED_NEUROFACE_COLLECTION_MANIFEST_SHA256,
)
from src.preprocessing.script_action_segmentation_v1 import (  # noqa: E402
    PINNED_NEUROFACE_PRIVATE_MANIFEST_SHA256,
)


REPORT_SCHEMA = "neuroface_als_temporal_tcn_public_report_v1"
PRIVATE_SCHEMA = "neuroface_als_temporal_tcn_private_oof_v1"
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_CACHE_BYTES = 64 * 1024 * 1024


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--private-manifest", required=True, type=Path)
    result.add_argument("--dynamic-collection", required=True, type=Path)
    result.add_argument("--dynamic-cache-root", required=True, type=Path)
    result.add_argument("--output-root", required=True, type=Path)
    return result


def _strict_object(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("authenticated JSON contains a duplicate key")
        output[key] = value
    return output


def _read_regular_bytes(path: Path, *, maximum: int) -> bytes:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("authenticated paths must be absolute")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if (not stat.S_ISREG(before.st_mode) or before.st_size <= 0
                or before.st_size > maximum):
            raise ValueError("authenticated input is not a bounded regular file")
        chunks = []
        while True:
            block = os.read(descriptor, min(1024 * 1024, maximum + 1))
            if not block:
                break
            chunks.append(block)
            if sum(map(len, chunks)) > maximum:
                raise ValueError("authenticated input exceeded its size bound")
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise ValueError("authenticated input changed during the read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _json_bytes(payload: bytes) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("authenticated input is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("authenticated JSON must contain an object")
    return value


def _unique_rows(rows: Sequence[Mapping[str, object]], *, name: str):
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError(f"{name} must be a row sequence")
    output = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("recording_id"), str):
            raise ValueError(f"{name} contains an invalid row")
        recording_id = str(row["recording_id"])
        if recording_id in output:
            raise ValueError(f"{name} contains a duplicate recording")
        output[recording_id] = row
    return output


def assemble_temporal_dataset(
    selected_rows: Sequence[Mapping[str, object]],
    dynamic_rows: Sequence[Mapping[str, object]],
    recordings: Mapping[str, DynamicLandmarkRecording],
) -> TemporalDataset:
    """Bind 66 authenticated task recordings into 22 participant tensors."""
    if len(selected_rows) != 66 or not isinstance(recordings, Mapping):
        raise ValueError("temporal endpoint requires exactly 66 selected recordings")
    selected = _unique_rows(selected_rows, name="private endpoint")
    dynamic = _unique_rows(dynamic_rows, name="dynamic collection")
    if set(recordings) != set(selected):
        raise ValueError("authenticated recording payloads differ from the endpoint")
    participant_order = []
    participant_slots: dict[str, dict[str, DynamicLandmarkRecording]] = {}
    participant_labels = {}
    for row in selected_rows:
        recording_id = str(row["recording_id"])
        group_id = row.get("participant_id")
        source_sha = row.get("video_sha256")
        task = row.get("task")
        cohort = row.get("cohort")
        if (not isinstance(group_id, str) or task not in PRIMARY_TASKS
                or cohort not in {"als", "healthy_control"}
                or not isinstance(source_sha, str)):
            raise ValueError("private endpoint row differs from the frozen schema")
        expected_label = 1 if cohort == "als" else 0
        dynamic_row = dynamic.get(recording_id)
        recording = recordings[recording_id]
        if (dynamic_row is None or dynamic_row.get("status") != "retained"
                or dynamic_row.get("participant_id") != group_id
                or dynamic_row.get("video_sha256") != source_sha
                or recording.recording_id != recording_id
                or recording.group_id != group_id
                or recording.source_sha256 != source_sha
                or recording.label != expected_label):
            raise ValueError("private, collection, and cache identities disagree")
        if group_id not in participant_slots:
            participant_order.append(group_id)
            participant_slots[group_id] = {}
            participant_labels[group_id] = expected_label
        if participant_labels[group_id] != expected_label or task in participant_slots[group_id]:
            raise ValueError("participant cohort or task identity is inconsistent")
        participant_slots[group_id][str(task)] = recording
    if (len(participant_order) != 22
            or any(set(slots) != set(PRIMARY_TASKS) for slots in participant_slots.values())
            or sum(participant_labels.values()) != 11):
        raise ValueError("endpoint must be 11 ALS and 11 healthy with all three tasks")
    features = np.stack([
        np.stack([participant_slots[group][task].features for task in PRIMARY_TASKS])
        for group in participant_order
    ]).astype(np.float32, copy=False)
    mask = np.stack([
        np.stack([participant_slots[group][task].valid_mask for task in PRIMARY_TASKS])
        for group in participant_order
    ])
    timestamps = np.stack([
        np.stack([participant_slots[group][task].timestamps for task in PRIMARY_TASKS])
        for group in participant_order
    ]).astype(np.float32)
    labels = np.asarray([participant_labels[group] for group in participant_order], dtype=np.int64)
    return validate_temporal_dataset(features, mask, timestamps, labels, participant_order)


def build_public_report(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    metrics = recompute_binary_metrics(np.asarray(labels), np.asarray(probabilities))
    return {
        "schema_version": REPORT_SCHEMA,
        "endpoint": "neuroface_als_vs_healthy_three_primary_tasks",
        "counts": {"participants": 22, "als": 11, "healthy": 11},
        "representation": "mediapipe_95d_raw_plus_gap_safe_velocity",
        "architecture": "compact_task_aware_temporal_tcn",
        "parameter_count": count_parameters(TaskAwareTemporalALSClassifier()),
        "training": {
            "outer_protocol": "participant_loso",
            "seeds": list(FROZEN_SEEDS),
            "epochs": FROZEN_EPOCHS,
            "train_fold_only_scaling": True,
            "outer_early_stopping": False,
            "mirror_aggregation": "probability_mean",
        },
        "metrics": metrics,
        "published_descriptive_comparator": {
            "paper_accuracy": PAPER_ACCURACY,
            "paper_auroc": PAPER_AUROC,
            "same_protocol_or_representation": False,
        },
        "development_milestones": {
            "auroc_above_0_90": bool(metrics["auroc"] > 0.90),
            "accuracy_above_paper_0_91": bool(metrics["accuracy"] > PAPER_ACCURACY),
            "auroc_above_paper_0_97": bool(metrics["auroc"] > PAPER_AUROC),
        },
        "claim_boundary": {
            "internal_development_only": True,
            "external_validation": False,
            "clinical_deployment_claim": False,
            "palsynet_reads": 0,
            "mayo_reads": 0,
            "meei_reads": 0,
        },
    }


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src" / "models" / "neuroface_als_temporal_v1.py",
        PROJECT_ROOT / "src" / "evaluation" / "neuroface_als_temporal_v1.py",
        PROJECT_ROOT / "src" / "evaluation" / "neuroface_als_benchmark_v1.py",
        PROJECT_ROOT / "src" / "models" / "dynamic_landmark.py",
        PROJECT_ROOT / "src" / "datasets" / "dynamic_landmark.py",
    ):
        payload = path.read_bytes()
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(len(payload).to_bytes(8, "big") + payload)
    return digest.hexdigest()


def _write_release(output_root: Path, report: dict[str, object], dataset: TemporalDataset,
                   probabilities: np.ndarray, seed_probabilities: np.ndarray) -> None:
    if not output_root.is_absolute() or output_root.exists() or output_root.is_symlink():
        raise ValueError("output root must be a new absolute directory")
    output_root.mkdir(mode=0o700, parents=False)
    private_path = output_root / "private_oof.npz"
    report_path = output_root / "report.json"
    np.savez_compressed(
        private_path,
        schema_version=np.asarray(PRIVATE_SCHEMA),
        group_ids=np.asarray(dataset.group_ids),
        labels=dataset.labels,
        probabilities=probabilities,
        seed_probabilities=seed_probabilities,
    )
    os.chmod(private_path, 0o600)
    report_payload = (json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n").encode("utf-8")
    descriptor = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        if os.write(descriptor, report_payload) != len(report_payload):
            raise OSError("short report write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    args = parser().parse_args()
    for path in (args.private_manifest, args.dynamic_collection, args.dynamic_cache_root):
        if any(token in os.fspath(path).casefold() for token in ("palsynet", "mayo", "meei")):
            raise ValueError("non-NeuroFace data are prohibited during candidate development")
    private_payload = _read_regular_bytes(args.private_manifest, maximum=_MAX_JSON_BYTES)
    collection_payload = _read_regular_bytes(args.dynamic_collection, maximum=_MAX_JSON_BYTES)
    if hashlib.sha256(private_payload).hexdigest() != PINNED_NEUROFACE_PRIVATE_MANIFEST_SHA256:
        raise ValueError("private NeuroFace manifest differs from the frozen inventory")
    if hashlib.sha256(collection_payload).hexdigest() != PINNED_NEUROFACE_COLLECTION_MANIFEST_SHA256:
        raise ValueError("dynamic collection differs from the frozen inventory")
    private = _json_bytes(private_payload)
    collection = _json_bytes(collection_payload)
    selected = select_paper_records(private)
    dynamic_rows = collection.get("records")
    if (collection.get("schema_version") != "neuroface_clinical23_v2_windows_v1"
            or not isinstance(dynamic_rows, list) or len(dynamic_rows) != 261):
        raise ValueError("dynamic collection is incomplete")
    dynamic_by_id = _unique_rows(dynamic_rows, name="dynamic collection")
    recordings = {}
    for row in selected:
        recording_id = str(row["recording_id"])
        dynamic_row = dynamic_by_id.get(recording_id)
        if dynamic_row is None or not isinstance(dynamic_row.get("cache_sha256"), str):
            raise ValueError("selected recording is absent from the dynamic collection")
        cache_path = args.dynamic_cache_root / f"{recording_id}.npz"
        payload = _read_regular_bytes(cache_path, maximum=_MAX_CACHE_BYTES)
        if hashlib.sha256(payload).hexdigest() != dynamic_row["cache_sha256"]:
            raise ValueError("dynamic cache differs from its collection commitment")
        recordings[recording_id] = load_dynamic_landmark_recording_bytes(payload)
    dataset = assemble_temporal_dataset(selected, dynamic_rows, recordings)
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) not in {
        "NVIDIA H200", "NVIDIA H200 NVL"
    }:
        raise ValueError("formal temporal development run requires the verified H200")
    evaluation = evaluate_temporal_loso(
        dataset.features, dataset.valid_mask, dataset.timestamps,
        dataset.labels, dataset.group_ids, device=torch.device("cuda"),
    )
    report = build_public_report(dataset.labels, evaluation.probabilities)
    report["provenance"] = {
        "private_manifest_sha256": hashlib.sha256(private_payload).hexdigest(),
        "dynamic_collection_sha256": hashlib.sha256(collection_payload).hexdigest(),
        "implementation_sha256": _implementation_sha256(),
    }
    _write_release(
        args.output_root, report, dataset,
        evaluation.probabilities, evaluation.seed_probabilities,
    )
    print(json.dumps({
        "schema_version": "neuroface_als_temporal_tcn_receipt_v1",
        "report_sha256": hashlib.sha256((args.output_root / "report.json").read_bytes()).hexdigest(),
        "metrics": report["metrics"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
