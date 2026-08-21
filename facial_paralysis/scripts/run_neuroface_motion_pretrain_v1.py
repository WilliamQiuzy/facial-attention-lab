#!/usr/bin/env python3
"""Run locked NeuroFace SLP motion pretraining and sealed PalsyNet dev transfer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

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
from scripts.run_mirror_invariant_110d import mirror_dynamic_features  # noqa: E402
from scripts.run_neuroface_external_v1 import _json, _read_same_descriptor  # noqa: E402
from src.datasets.dynamic_landmark import load_dynamic_landmark_recording  # noqa: E402
from src.preprocessing.generalization_110d import (  # noqa: E402
    LANDMARK_MI_110D,
    candidate_feature_vector,
)
from src.training.neuroface_motion_pretrain_v1 import (  # noqa: E402
    DOMAINS,
    MotionDataset,
    MotionPretrainConfig,
    TransferDataset,
    build_aggregate_report,
    evaluate_frozen_palsynet_transfer,
    frozen_motion_embeddings,
    run_motion_pretraining,
)


TASKS = (
    "BBP_NORMAL", "DDK_PA", "DDK_PATAKA", "NSM_BIGSMILE", "NSM_BLOW",
    "NSM_BROW", "NSM_KISS", "NSM_OPEN", "NSM_SPREAD",
)
LANDMARK_OFFSET = 72
OUTPUT_RELATIVE = "outputs/neuroface_motion_pretraining_v1"
_IMPLEMENTATION_FILES = (
    Path(__file__).resolve(),
    PROJECT_ROOT / "src" / "models" / "neuroface_motion_pretrain_v1.py",
    PROJECT_ROOT / "src" / "training" / "neuroface_motion_pretrain_v1.py",
    PROJECT_ROOT / "src" / "datasets" / "dynamic_landmark.py",
    PROJECT_ROOT / "src" / "preprocessing" / "generalization_110d.py",
    PROJECT_ROOT / "scripts" / "run_110d_generalization_v1.py",
    PROJECT_ROOT / "scripts" / "run_mirror_invariant_110d.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _implementation_sha256() -> str:
    components = {
        str(path.relative_to(PROJECT_ROOT)): _sha256(path)
        for path in _IMPLEMENTATION_FILES
    }
    encoded = json.dumps(components, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_record_same_bytes(path: Path, expected_sha256: str):
    payload, observed = _read_same_descriptor(path)
    if observed != expected_sha256:
        raise ValueError("dynamic cache bytes differ from the collection manifest")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".motion-cache.", suffix=".npz")
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return load_dynamic_landmark_recording(temporary)
    finally:
        temporary.unlink(missing_ok=True)


def _build_neuroface_dataset(
    private: Mapping[str, object],
    collection: Mapping[str, object],
    cache_root: Path,
) -> MotionDataset:
    private_rows = private.get("records")
    cache_rows = collection.get("records")
    if (
        private.get("schema_version") != "neuroface_external_private_manifest_v1"
        or not isinstance(private_rows, list) or len(private_rows) != 261
        or collection.get("schema_version") != "neuroface_clinical23_v2_windows_v1"
        or not isinstance(cache_rows, list) or len(cache_rows) != 261
    ):
        raise ValueError("NeuroFace private or cache manifest differs from the frozen cohort")
    private_by_id = {str(row["recording_id"]): row for row in private_rows}
    retained = [row for row in cache_rows if row.get("status") == "retained"]
    if len(private_by_id) != 261 or len(retained) != 231:
        raise ValueError("NeuroFace training requires the frozen 231-record QC cohort")
    features: list[np.ndarray] = []
    mirrors: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    targets: list[list[float]] = []
    task_indices: list[int] = []
    groups: list[str] = []
    cohorts: list[str] = []
    for cache_row in sorted(retained, key=lambda row: str(row["recording_id"])):
        recording_id = str(cache_row["recording_id"])
        source = private_by_id.get(recording_id)
        if source is None:
            raise ValueError("retained cache record is missing from the private manifest")
        cache_path = cache_root / f"{recording_id}.npz"
        record = _load_record_same_bytes(cache_path, str(cache_row["cache_sha256"]))
        if (
            record.recording_id != recording_id
            or record.group_id != source.get("participant_id")
            or record.source_sha256 != source.get("video_sha256")
            or cache_row.get("participant_id") != source.get("participant_id")
            or cache_row.get("video_sha256") != source.get("video_sha256")
        ):
            raise ValueError("NeuroFace cache/private provenance is not exact")
        task = source.get("task")
        slp = source.get("slp_scores")
        cohort = source.get("cohort")
        if task not in TASKS or not isinstance(slp, Mapping) or cohort not in {
            "als", "healthy_control", "post_stroke"
        }:
            raise ValueError("NeuroFace task, SLP, or cohort schema is invalid")
        values = np.asarray(record.features, dtype=np.float32)
        mirrored = np.asarray(mirror_dynamic_features(values), dtype=np.float32)
        features.append(values[..., LANDMARK_OFFSET:])
        mirrors.append(mirrored[..., LANDMARK_OFFSET:])
        masks.append(np.asarray(record.valid_mask, dtype=bool))
        timestamps.append(np.asarray(record.timestamps, dtype=np.float32))
        targets.append([float(slp[name]) for name in DOMAINS])
        task_indices.append(TASKS.index(str(task)))
        groups.append(str(source["participant_id"]))
        cohorts.append(str(cohort))
    if len(set(groups)) != 36:
        raise ValueError("NeuroFace motion pretraining requires all 36 participants")
    return MotionDataset(
        landmarks=np.stack(features),
        mirrored_landmarks=np.stack(mirrors),
        valid_masks=np.stack(masks),
        timestamps=np.stack(timestamps),
        targets=np.asarray(targets, dtype=np.float32),
        task_indices=np.asarray(task_indices, dtype=np.int64),
        group_ids=np.asarray(groups, dtype=object),
        cohorts=np.asarray(cohorts, dtype=object),
    )


def _build_palsynet_transfer(
    *,
    cache_root: Path,
    reviewed_manifest: Mapping[str, object],
    reviewed_manifest_sha256: str,
    review_ledger: Mapping[str, object],
    review_ledger_sha256: str,
    split_registry: Mapping[str, object],
    split_registry_sha256: str,
    motion_result,
    device: str,
):
    dataset, collection_rows, source_collection_sha = _build_cache_metadata_dataset(cache_root)
    audit = GateAudit()
    gate = validate_development_gate(
        dataset, reviewed_manifest, review_ledger, split_registry,
        reviewed_manifest_sha256=reviewed_manifest_sha256,
        review_ledger_sha256=review_ledger_sha256,
        split_registry_sha256=split_registry_sha256,
        cache_source_sha256_by_recording_id={
            recording_id: str(row["source_sha256"])
            for recording_id, row in collection_rows.items()
        },
        cache_source_collection_sha256=source_collection_sha,
        audit=audit,
    )
    load_development_cache_records(
        cache_root, dataset, gate, collection_rows, audit=audit
    )
    n = int(dataset.labels.size)
    summaries = np.full((n, 110), np.nan, dtype=np.float64)
    mirrored_summaries = np.full((n, 110), np.nan, dtype=np.float64)
    original_motion = np.full((n, 32), np.nan, dtype=np.float64)
    mirrored_motion = np.full((n, 32), np.nan, dtype=np.float64)
    development = np.asarray(gate.development_indices, dtype=np.int64)
    raw_development = np.asarray(dataset.features[development], dtype=np.float32)
    mirrored_development = np.stack([
        np.asarray(mirror_dynamic_features(values), dtype=np.float32)
        for values in raw_development
    ])
    motion_original, motion_mirrored = frozen_motion_embeddings(
        motion_result.final_model,
        raw_development[..., LANDMARK_OFFSET:],
        mirrored_development[..., LANDMARK_OFFSET:],
        dataset.valid_masks[development],
        dataset.timestamps[development],
        motion_result.landmark_mean,
        motion_result.landmark_scale,
        device=device,
    )
    original_motion[development] = motion_original
    mirrored_motion[development] = motion_mirrored
    for local, index in enumerate(development.tolist()):
        temporal = (
            dataset.valid_masks[index], dataset.timestamps[index],
            dataset.source_frame_indices[index],
        )
        summaries[index] = candidate_feature_vector(
            LANDMARK_MI_110D, raw_development[local], *temporal
        )
        mirrored_summaries[index] = candidate_feature_vector(
            LANDMARK_MI_110D, mirrored_development[local], *temporal
        )
    transfer = evaluate_frozen_palsynet_transfer(TransferDataset(
        summary_features=summaries,
        mirrored_summary_features=mirrored_summaries,
        motion_features=original_motion,
        mirrored_motion_features=mirrored_motion,
        labels=np.asarray(dataset.labels, dtype=np.int64),
        group_ids=np.asarray(gate.group_ids, dtype=object),
        development_indices=development,
        protected_indices=np.asarray(gate.protected_indices, dtype=np.int64),
        inner_fold_by_index=np.asarray(gate.inner_fold_by_index, dtype=np.int64),
    ))
    if audit.protected_cache_records_loaded != 0:
        raise AssertionError("PalsyNet protected cache read occurred")
    return transfer, gate


def _write_no_overwrite_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".motion-report.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
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


def _write_no_overwrite_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".motion-model.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        os.chmod(temporary, 0o600)
        torch.save(dict(payload), temporary)
        os.link(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neuroface-private-manifest", required=True, type=Path)
    parser.add_argument("--neuroface-cache-root", required=True, type=Path)
    parser.add_argument("--palsynet-cache-root", required=True, type=Path)
    parser.add_argument("--reviewed-identity-manifest", required=True, type=Path)
    parser.add_argument("--review-ledger", required=True, type=Path)
    parser.add_argument("--split-registry", required=True, type=Path)
    parser.add_argument("--dependency-lock", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.perf_counter()
    output_root = PROJECT_ROOT / OUTPUT_RELATIVE
    report_path = output_root / "report.json"
    checkpoint_path = output_root / "motion_encoder_private.pt"
    if report_path.exists() or checkpoint_path.exists():
        raise FileExistsError("motion pretraining outputs already exist")
    private, private_sha = _json(args.neuroface_private_manifest)
    collection, neuroface_cache_sha = _json(
        args.neuroface_cache_root / "collection_manifest.json"
    )
    reviewed, reviewed_sha = _read_json(args.reviewed_identity_manifest)
    ledger, ledger_sha = _read_json(args.review_ledger)
    registry, registry_sha = _read_json(args.split_registry)
    _, palsynet_cache_sha = _read_same_descriptor(
        args.palsynet_cache_root / "collection_manifest.json"
    )
    _, dependency_sha = _read_same_descriptor(args.dependency_lock)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    neuroface = _build_neuroface_dataset(private, collection, args.neuroface_cache_root)
    motion = run_motion_pretraining(
        neuroface, config=MotionPretrainConfig(), device=device
    )
    transfer, gate = _build_palsynet_transfer(
        cache_root=args.palsynet_cache_root,
        reviewed_manifest=reviewed,
        reviewed_manifest_sha256=reviewed_sha,
        review_ledger=ledger,
        review_ledger_sha256=ledger_sha,
        split_registry=registry,
        split_registry_sha256=registry_sha,
        motion_result=motion,
        device=device,
    )
    report = build_aggregate_report(
        pretrain_metrics=motion.metrics,
        transfer=transfer,
        provenance={
            "neuroface_private_manifest_sha256": private_sha,
            "neuroface_cache_collection_sha256": neuroface_cache_sha,
            "palsynet_cache_collection_sha256": palsynet_cache_sha,
            "palsynet_reviewed_manifest_sha256": reviewed_sha,
            "palsynet_review_ledger_sha256": ledger_sha,
            "palsynet_split_registry_sha256": registry_sha,
            "implementation_sha256": _implementation_sha256(),
            "dependency_lock_sha256": dependency_sha,
        },
        runtime={
            "host": "nebius-h200",
            "device": device,
            "seconds": float(time.perf_counter() - started),
        },
        parameter_count=motion.parameter_count,
    )
    _write_no_overwrite_checkpoint(checkpoint_path, {
        "schema_version": "neuroface_motion_encoder_private_v1",
        "model_state_dict": motion.final_model.state_dict(),
        "landmark_mean": motion.landmark_mean,
        "landmark_scale": motion.landmark_scale,
        "final_epochs": motion.metrics["final_epochs"],
        "parameter_count": motion.parameter_count,
        "private_manifest_sha256": private_sha,
        "neuroface_cache_collection_sha256": neuroface_cache_sha,
    })
    _write_no_overwrite_json(report_path, report)
    receipt = {
        "schema_version": "neuroface_motion_pretraining_v1_receipt",
        "report_sha256": _sha256(report_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "selected_model": report["decision"]["selected_model"],
        "protected_predictions": report["audit"]["protected_predictions"],
        "palsynet_protected_rows": int(gate.protected_indices.size),
    }
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
