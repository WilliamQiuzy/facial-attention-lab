#!/usr/bin/env python3
"""Run the protected-test-sealed Architecture Autoresearch v1 screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
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
from src.models.architecture_search_v1 import CANDIDATE_ORDER  # noqa: E402
from src.preprocessing.generalization_110d import (  # noqa: E402
    LANDMARK_MI_110D,
    candidate_feature_vector,
)
from src.training.architecture_search_v1 import (  # noqa: E402
    ConfirmationResult,
    SearchConfig,
    SearchDataset,
    ScreeningResult,
    evaluate_fixed_ensembles,
    run_confirmation,
    run_logistic_stability_audit,
    run_screening,
)


DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "outputs" / "dynamic_landmark" / "benchmarks" / "development"
    / "architecture-autoresearch-v1" / "report.json"
)
SMOKE_CANDIDATES = CANDIDATE_ORDER[:4]
_IMPLEMENTATION_FILES = (
    Path(__file__).resolve(),
    PROJECT_ROOT / "src" / "models" / "architecture_search_v1.py",
    PROJECT_ROOT / "src" / "training" / "architecture_search_v1.py",
    PROJECT_ROOT / "scripts" / "run_110d_generalization_v1.py",
    PROJECT_ROOT / "src" / "preprocessing" / "generalization_110d.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_sha256() -> str:
    components = {str(path.relative_to(PROJECT_ROOT)): _sha256(path)
                  for path in _IMPLEMENTATION_FILES}
    encoded = json.dumps(components, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_aggregate_report(
    result: ScreeningResult,
    *,
    smoke: bool,
    device: str,
    provenance: Mapping[str, str],
    protected_groups: int,
    protected_recordings: int,
    confirmation: ConfirmationResult | None = None,
    adaptive_ensembles: Mapping[str, Mapping[str, object]] | None = None,
    stability_audit: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Create an identifier-free aggregate report for independent audit."""
    required_provenance = {
        "source_collection_sha256", "reviewed_manifest_sha256",
        "review_ledger_sha256", "split_registry_sha256", "implementation_sha256",
    }
    if set(provenance) != required_provenance or any(
        not isinstance(value, str) or len(value) != 64
        for value in provenance.values()
    ):
        raise ValueError("report provenance must contain five SHA-256 digests")
    if result.protected_predictions != 0:
        raise ValueError("protected predictions cannot enter an aggregate report")
    if confirmation is not None and (
        confirmation.winner != result.winner
        or confirmation.protected_predictions != 0
    ):
        raise ValueError("confirmation must match the sealed screening winner")
    return {
        "schema_version": "architecture_autoresearch_v1_report",
        "claim_scope": "identity_reviewed_palsynet_development_oof_only",
        "target": "binary_affected_vs_unaffected_not_hb_grade",
        "mode": "smoke" if smoke else "full_screening",
        "protocol": {
            "development_folds": 4,
            "screen_seed": 0,
            "confirmation_seeds": [0, 1, 2],
            "screen_epochs": 40,
            "patience": 6,
            "threshold": 0.5,
            "candidate_order": list(result.candidate_metrics),
            "ranking": ["auroc", "balanced_accuracy", "brier", "simplicity"],
            "device": device,
        },
        "counts": {
            "development_recordings": result.development_recordings,
            "development_groups": result.development_groups,
            "protected_recordings": int(protected_recordings),
            "protected_groups": int(protected_groups),
        },
        "metrics": {name: dict(values)
                    for name, values in result.candidate_metrics.items()},
        "fold_metrics": {
            name: [dict(fold) for fold in folds]
            for name, folds in result.candidate_fold_metrics.items()
        },
        "confirmation": None if confirmation is None else {
            "winner": confirmation.winner,
            "seeds": {
                str(seed): dict(values)
                for seed, values in confirmation.seed_metrics.items()
            },
            "ensemble_metrics": dict(confirmation.ensemble_metrics),
            "parameter_count": confirmation.parameter_count,
        },
        "adaptive_ensemble_round": None if adaptive_ensembles is None else {
            name: dict(values) for name, values in adaptive_ensembles.items()
        },
        "validation_stress_test": None if stability_audit is None else dict(stability_audit),
        "decision": {
            "screening_winner": result.winner,
            "current_model_replaced": False,
            "outer_evaluation_authorized": False,
            "mayo_used_for_selection": False,
            "meei_used_for_selection": False,
            "clinical_validation": False,
        },
        "audit": {
            "protected_cache_records_loaded": 0,
            "protected_feature_extractions": 0,
            "protected_model_fits": 0,
            "protected_predictions": result.protected_predictions,
        },
        "provenance": dict(provenance),
    }


def _prepare_search_dataset(dataset, gate) -> SearchDataset:
    n = int(dataset.labels.size)
    mirrored_raw = np.zeros_like(dataset.features)
    summaries = np.zeros((n, 110), dtype=np.float64)
    mirrored_summaries = np.zeros_like(summaries)
    for index in gate.development_indices.tolist():
        raw = np.asarray(dataset.features[index], dtype=np.float32)
        mirrored = np.asarray(mirror_dynamic_features(raw), dtype=np.float32)
        mirrored_raw[index] = mirrored
        temporal = (
            dataset.valid_masks[index], dataset.timestamps[index],
            dataset.source_frame_indices[index],
        )
        summaries[index] = candidate_feature_vector(
            LANDMARK_MI_110D, raw, *temporal
        )
        mirrored_summaries[index] = candidate_feature_vector(
            LANDMARK_MI_110D, mirrored, *temporal
        )
    return SearchDataset(
        raw_features=np.asarray(dataset.features, dtype=np.float32),
        mirrored_raw_features=mirrored_raw,
        valid_masks=np.asarray(dataset.valid_masks, dtype=bool),
        summary_features=summaries,
        mirrored_summary_features=mirrored_summaries,
        labels=np.asarray(dataset.labels, dtype=np.int64),
        group_ids=np.asarray(gate.group_ids, dtype=object),
        development_indices=np.asarray(gate.development_indices, dtype=np.int64),
        protected_indices=np.asarray(gate.protected_indices, dtype=np.int64),
        inner_fold_by_index=np.asarray(gate.inner_fold_by_index, dtype=np.int64),
    )


def _write_no_overwrite(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite architecture report {path}")
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd, temporary_name = tempfile.mkstemp(prefix=".architecture.", suffix=".tmp", dir=path.parent)
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--palsynet-cache-root", required=True, type=Path)
    parser.add_argument("--reviewed-identity-manifest", required=True, type=Path)
    parser.add_argument("--review-ledger", required=True, type=Path)
    parser.add_argument("--split-registry", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    manifest, manifest_sha = _read_json(args.reviewed_identity_manifest)
    ledger, ledger_sha = _read_json(args.review_ledger)
    registry, registry_sha = _read_json(args.split_registry)
    dataset, collection_rows, collection_sha = _build_cache_metadata_dataset(
        args.palsynet_cache_root
    )
    audit = GateAudit()
    gate = validate_development_gate(
        dataset, manifest, ledger, registry,
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
    search_dataset = _prepare_search_dataset(dataset, gate)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    config = SearchConfig(smoke=args.smoke)
    result = run_screening(
        search_dataset,
        config=config,
        candidates=SMOKE_CANDIDATES if args.smoke else CANDIDATE_ORDER,
        device=device,
    )
    confirmation = None
    adaptive_ensembles = None
    stability_audit = None
    if not args.smoke:
        confirmation = run_confirmation(
            search_dataset,
            winner=result.winner,
            config=config,
            device=device,
        )
        if result.candidate_oof_probabilities is None:
            raise RuntimeError("full screening did not retain aligned private OOF values")
        development = search_dataset.development_indices
        adaptive_ensembles = evaluate_fixed_ensembles(
            search_dataset.labels[development],
            search_dataset.group_ids[development],
            result.candidate_oof_probabilities,
        )
        stability_audit = run_logistic_stability_audit(search_dataset)
    report = build_aggregate_report(
        result,
        smoke=args.smoke,
        device=device,
        provenance={
            "source_collection_sha256": gate.source_collection_sha256,
            "reviewed_manifest_sha256": gate.reviewed_manifest_sha256,
            "review_ledger_sha256": gate.review_ledger_sha256,
            "split_registry_sha256": gate.split_registry_sha256,
            "implementation_sha256": _implementation_sha256(),
        },
        protected_groups=len(set(gate.group_ids[gate.protected_indices].tolist())),
        protected_recordings=int(gate.protected_indices.size),
        confirmation=confirmation,
        adaptive_ensembles=adaptive_ensembles,
        stability_audit=stability_audit,
    )
    if not args.smoke:
        _write_no_overwrite(DEFAULT_REPORT_PATH, report)
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
