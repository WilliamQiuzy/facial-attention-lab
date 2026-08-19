#!/usr/bin/env python3
"""Fit fixed manual68 geometry calibration and compare calibrated 110D on PalsyNet dev."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_110d_generalization_v1 import (  # noqa: E402
    GateAudit, _build_cache_metadata_dataset, _read_json,
    load_development_cache_records, validate_development_gate,
)
from scripts.run_mirror_invariant_110d import mirror_dynamic_features  # noqa: E402
from scripts.run_neuroface_external_v1 import _read_same_descriptor  # noqa: E402
from scripts.run_neuroface_motion_pretrain_v1 import _write_no_overwrite_json  # noqa: E402
from src.preprocessing.generalization_110d import (  # noqa: E402
    LANDMARK_MI_110D, candidate_feature_vector,
)
from src.training.neuroface_geometry_calibration_v1 import (  # noqa: E402
    CalibratedTransferDataset, calibrate_dynamic_features,
    evaluate_calibrated_transfer, fit_geometry_calibration,
)


OUTPUT_RELATIVE = "outputs/neuroface_geometry_calibration_v1/report.json"
PAIR_FIELDS = {
    "schema_version", "manual_semantic23", "mediapipe_semantic23", "detected",
    "participant_ids", "recording_ids", "cohorts", "tasks",
    "private_manifest_sha256", "mediapipe_model_sha256",
}
_IMPLEMENTATION_FILES = (
    Path(__file__).resolve(),
    PROJECT_ROOT / "src" / "training" / "neuroface_geometry_calibration_v1.py",
    PROJECT_ROOT / "src" / "preprocessing" / "semantic_landmarks.py",
    PROJECT_ROOT / "src" / "preprocessing" / "generalization_110d.py",
    PROJECT_ROOT / "scripts" / "run_110d_generalization_v1.py",
    PROJECT_ROOT / "scripts" / "run_mirror_invariant_110d.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _implementation_sha256() -> str:
    values = {
        str(path.relative_to(PROJECT_ROOT)): _sha256(path)
        for path in _IMPLEMENTATION_FILES
    }
    return hashlib.sha256(json.dumps(
        values, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def _scalar_text(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name} must be scalar text")
    item = array.item()
    return item.decode() if isinstance(item, bytes) else str(item)


def _load_pairs(path: Path) -> dict[str, np.ndarray]:
    payload, _digest = _read_same_descriptor(path)
    import io
    with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
        if set(saved.files) != PAIR_FIELDS:
            raise ValueError("private manual68 pair fields differ from the frozen schema")
        values = {name: np.asarray(saved[name]) for name in saved.files}
    if (
        _scalar_text(values["schema_version"], "schema_version")
        != "neuroface_manual68_pairs_private_v1"
        or values["manual_semantic23"].shape != (3306, 23)
        or values["mediapipe_semantic23"].shape != (3306, 23)
        or values["detected"].shape != (3306,)
        or values["detected"].dtype != np.dtype(bool)
        or not values["detected"].all()
    ):
        raise ValueError("private manual68 pair cache is incomplete")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual68-pairs", required=True, type=Path)
    parser.add_argument("--palsynet-cache-root", required=True, type=Path)
    parser.add_argument("--reviewed-identity-manifest", required=True, type=Path)
    parser.add_argument("--review-ledger", required=True, type=Path)
    parser.add_argument("--split-registry", required=True, type=Path)
    parser.add_argument("--dependency-lock", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.perf_counter()
    output = PROJECT_ROOT / OUTPUT_RELATIVE
    if output.exists() or output.is_symlink():
        raise FileExistsError("geometry calibration report already exists")
    pairs = _load_pairs(args.manual68_pairs)
    pairs_sha = _sha256(args.manual68_pairs)
    calibration = fit_geometry_calibration(
        pairs["mediapipe_semantic23"], pairs["manual_semantic23"],
        pairs["participant_ids"].astype(object), pairs["cohorts"].astype(object),
    )
    reviewed, reviewed_sha = _read_json(args.reviewed_identity_manifest)
    ledger, ledger_sha = _read_json(args.review_ledger)
    registry, registry_sha = _read_json(args.split_registry)
    dataset, collection_rows, source_collection_sha = _build_cache_metadata_dataset(
        args.palsynet_cache_root
    )
    _, palsynet_cache_sha = _read_same_descriptor(
        args.palsynet_cache_root / "collection_manifest.json"
    )
    _, dependency_sha = _read_same_descriptor(args.dependency_lock)
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
        cache_source_collection_sha256=source_collection_sha,
        audit=audit,
    )
    load_development_cache_records(
        args.palsynet_cache_root, dataset, gate, collection_rows, audit=audit
    )
    n = dataset.labels.size
    development = np.asarray(gate.development_indices, dtype=np.int64)
    baseline = np.full((n, 110), np.nan)
    mirrored_baseline = np.full((n, 110), np.nan)
    calibrated = np.full((n, 110), np.nan)
    mirrored_calibrated = np.full((n, 110), np.nan)
    raw = np.asarray(dataset.features[development], dtype=np.float32)
    mirrored_raw = np.stack([
        np.asarray(mirror_dynamic_features(value), dtype=np.float32) for value in raw
    ])
    calibrated_raw = calibrate_dynamic_features(
        raw, dataset.valid_masks[development], calibration
    )
    calibrated_mirrored_raw = calibrate_dynamic_features(
        mirrored_raw, dataset.valid_masks[development], calibration
    )
    for local, index in enumerate(development.tolist()):
        temporal = (
            dataset.valid_masks[index], dataset.timestamps[index],
            dataset.source_frame_indices[index],
        )
        baseline[index] = candidate_feature_vector(
            LANDMARK_MI_110D, raw[local], *temporal
        )
        mirrored_baseline[index] = candidate_feature_vector(
            LANDMARK_MI_110D, mirrored_raw[local], *temporal
        )
        calibrated[index] = candidate_feature_vector(
            LANDMARK_MI_110D, calibrated_raw[local], *temporal
        )
        mirrored_calibrated[index] = candidate_feature_vector(
            LANDMARK_MI_110D, calibrated_mirrored_raw[local], *temporal
        )
    transfer = evaluate_calibrated_transfer(CalibratedTransferDataset(
        baseline=baseline,
        mirrored_baseline=mirrored_baseline,
        calibrated=calibrated,
        mirrored_calibrated=mirrored_calibrated,
        labels=np.asarray(dataset.labels, dtype=np.int64),
        group_ids=np.asarray(gate.group_ids, dtype=object),
        development_indices=development,
        protected_indices=np.asarray(gate.protected_indices, dtype=np.int64),
        inner_fold_by_index=np.asarray(gate.inner_fold_by_index, dtype=np.int64),
    ))
    report = {
        "schema_version": "neuroface_geometry_calibration_v1_report",
        "claim_scope": "label_free_measurement_transfer_palsynet_development_oof_only",
        "target": "binary_affected_vs_unaffected_not_hb_grade",
        "calibration": dict(calibration.metrics),
        "counts": {
            "neuroface_manual_frames": 3306,
            "neuroface_participants": 36,
            "palsynet_development_recordings": transfer["development_recordings"],
            "palsynet_development_groups": transfer["development_groups"],
        },
        "palsynet_development_metrics": transfer["metrics"],
        "decision": {
            "promotion_criteria_met": transfer["promotion_criteria_met"],
            "current_model_replaced": transfer["promotion_criteria_met"],
            "selected_model": transfer["selected_model"],
            "outer_evaluation_authorized": False,
            "clinical_validation": False,
        },
        "audit": {
            "palsynet_protected_cache_records_loaded": 0,
            "protected_feature_extractions": 0,
            "protected_model_fits": 0,
            "protected_predictions": transfer["protected_predictions"],
        },
        "runtime": {"host": "nebius-h200", "seconds": time.perf_counter() - started},
        "provenance": {
            "manual68_pairs_sha256": pairs_sha,
            "private_manifest_sha256": _scalar_text(
                pairs["private_manifest_sha256"], "private_manifest_sha256"
            ),
            "mediapipe_model_sha256": _scalar_text(
                pairs["mediapipe_model_sha256"], "mediapipe_model_sha256"
            ),
            "palsynet_cache_collection_sha256": palsynet_cache_sha,
            "palsynet_reviewed_manifest_sha256": reviewed_sha,
            "palsynet_review_ledger_sha256": ledger_sha,
            "palsynet_split_registry_sha256": registry_sha,
            "implementation_sha256": _implementation_sha256(),
            "dependency_lock_sha256": dependency_sha,
        },
    }
    encoded = json.dumps(report, sort_keys=True, allow_nan=False).lower()
    if any(token in encoded for token in (
        "grp_", "rec_", "participant_id", "recording_id", "/home/", "/users/"
    )):
        raise ValueError("calibration aggregate report leaks private identifiers or paths")
    _write_no_overwrite_json(output, report)
    print(json.dumps({
        "schema_version": "neuroface_geometry_calibration_v1_receipt",
        "report_sha256": _sha256(output),
        "selected_model": transfer["selected_model"],
        "protected_predictions": transfer["protected_predictions"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
