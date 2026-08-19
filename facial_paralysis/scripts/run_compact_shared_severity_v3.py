#!/usr/bin/env python3
"""Run the compact medically gated shared-severity search on H200."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_dense_clinical_shared_encoder_v1 as v1_runner  # noqa: E402
from scripts import run_medically_gated_shared_search_v2 as v2_runner  # noqa: E402
from src.evaluation.compact_shared_severity_v3 import (  # noqa: E402
    evaluate_compact_candidate,
    rank_compact_results,
)
from src.evaluation.shared_clinical_encoder_v1 import SOURCES  # noqa: E402
from src.models.compact_shared_severity_v3 import (  # noqa: E402
    COMPACT_RATIONALES,
    compact_candidate_registry,
)


def validate_candidate_phase(phase: str, candidate_ids: tuple[str, ...]) -> None:
    all_ids = tuple(
        candidate.candidate_id for candidate in compact_candidate_registry()
    )
    if (
        type(candidate_ids) is not tuple
        or len(candidate_ids) != len(set(candidate_ids))
        or any(type(candidate_id) is not str for candidate_id in candidate_ids)
        or any(candidate_id not in all_ids for candidate_id in candidate_ids)
    ):
        raise ValueError("candidate identifiers differ from the compact registry")
    if phase == "screen":
        valid = candidate_ids == all_ids
    elif phase == "confirm":
        valid = len(candidate_ids) == 4
    else:
        raise ValueError("phase must be screen or confirm")
    if not valid:
        raise ValueError("candidate set differs from the frozen v3 phase")


def _ranking_key(
    candidate_id: str,
    evaluations: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> tuple[float, float, float, str]:
    metrics = evaluations[candidate_id]
    accuracy = [float(metrics[source]["accuracy"]) for source in SOURCES]
    auroc = [float(metrics[source]["auroc"]) for source in SOURCES]
    return (
        -min(accuracy), -min(auroc), -float(np.mean(accuracy)), candidate_id
    )


def build_report(
    *,
    phase: str,
    evaluations: Mapping[str, Mapping[str, Mapping[str, float]]],
    ranking: tuple[str, ...],
    counts: Mapping[str, int],
    runtime: Mapping[str, object],
    commitments: Mapping[str, str],
) -> dict[str, object]:
    if type(evaluations) is not dict:
        raise ValueError("evaluations must be an exact dictionary")
    candidate_ids = tuple(evaluations)
    validate_candidate_phase(phase, candidate_ids)
    if (
        type(ranking) is not tuple
        or len(ranking) != len(candidate_ids)
        or set(ranking) != set(candidate_ids)
        or dict(counts) != {"palsynet": 38, "neuroface": 36, "meei": 56}
        or ranking != tuple(sorted(
            candidate_ids,
            key=lambda candidate_id: _ranking_key(candidate_id, evaluations),
        ))
    ):
        raise ValueError("v3 report ranking or participant counts drifted")
    required = {
        "accuracy", "balanced_accuracy", "auroc", "sensitivity",
        "specificity", "brier",
    }
    for metrics in evaluations.values():
        if set(metrics) != set(SOURCES) or any(
            set(metrics[source]) != required
            or any(not np.isfinite(float(value)) for value in metrics[source].values())
            for source in SOURCES
        ):
            raise ValueError("v3 metrics differ from the closed schema")
    return {
        "schema_version": "compact_shared_severity_v3_search",
        "status": "exposed_development_candidate_search_not_clinically_validated",
        "phase": phase,
        "model": {
            "name": "Compact Shared Clinical Severity v3",
            "inputs": ["clinical_110d", "dense_mediapipe_478x3"],
            "dense_transform": "fixed_regional_excursion_velocity",
            "shared_patient_embedding_dim": 64,
            "shared_severity_axis": True,
            "source_specific_layers": ["final_binary_calibration_or_head"],
            "source_identifier_input": False,
            "flip_used": False,
        },
        "medical_gate": {
            "rationales": COMPACT_RATIONALES,
            "random_visual_augmentation": False,
            "affected_side_claim_allowed": False,
            "hb_grade_claim_allowed": False,
        },
        "candidate_registry": [
            {
                "candidate_id": candidate.candidate_id,
                "region_scope": candidate.region_scope,
                "dynamic_stats": candidate.dynamic_stats,
                "pooling": candidate.pooling,
                "head_mode": candidate.head_mode,
            }
            for candidate in compact_candidate_registry()
        ],
        "candidate_ids": list(candidate_ids),
        "counts": dict(counts),
        "evaluations": {
            candidate_id: {
                source: {
                    metric: float(value)
                    for metric, value in evaluations[candidate_id][source].items()
                }
                for source in SOURCES
            }
            for candidate_id in candidate_ids
        },
        "selection": {
            "primary_metric": "minimum_source_accuracy",
            "secondary_metric": "minimum_source_auroc",
            "tertiary_metric": "mean_source_accuracy",
            "ranking": list(ranking),
        },
        "runtime": dict(runtime),
        "commitments": dict(commitments),
        "audit": {
            "palsynet_protected_reads": 0,
            "mayo_reads": 0,
            "mayo_predictions": 0,
        },
        "decision": {
            "promotion_authorized": False,
            "clinical_claim_authorized": False,
        },
    }


def _implementation_sha256() -> str:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/evaluation/compact_shared_severity_v3.py",
        PROJECT_ROOT / "src/models/compact_shared_severity_v3.py",
        PROJECT_ROOT / "scripts/run_medically_gated_shared_search_v2.py",
        PROJECT_ROOT / "src/evaluation/medically_gated_shared_search_v2.py",
        PROJECT_ROOT / "scripts/run_dense_clinical_shared_encoder_v1.py",
        PROJECT_ROOT / "src/evaluation/shared_clinical_encoder_v1.py",
        PROJECT_ROOT / "src/preprocessing/shared_clinical_tokens_v1.py",
        PROJECT_ROOT / "src/preprocessing/clinical_landmarks.py",
        PROJECT_ROOT / "src/preprocessing/generalization_110d.py",
        PROJECT_ROOT / "src/preprocessing/trajectory_features.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        payload = path.read_bytes()
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--palsynet-cache-root", type=Path, required=True)
    parser.add_argument("--reviewed-identity-manifest", type=Path, required=True)
    parser.add_argument("--review-ledger", type=Path, required=True)
    parser.add_argument("--split-registry", type=Path, required=True)
    parser.add_argument("--neuroface-cache", type=Path, required=True)
    parser.add_argument("--neuroface-collection-sha256", required=True)
    parser.add_argument("--neuroface-manifest", type=Path, required=True)
    parser.add_argument("--neuroface-manifest-sha256", required=True)
    parser.add_argument("--meei-cache", type=Path, required=True)
    parser.add_argument("--meei-collection-sha256", required=True)
    parser.add_argument("--meei-manifest", type=Path, required=True)
    parser.add_argument("--meei-manifest-sha256", required=True)
    parser.add_argument("--phase", choices=("screen", "confirm"), required=True)
    parser.add_argument("--candidate-ids", nargs="*", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, default=6)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA H200":
        raise RuntimeError("v3 search requires the verified NVIDIA H200")
    registry = compact_candidate_registry()
    lookup = {candidate.candidate_id: candidate for candidate in registry}
    candidate_ids = (
        tuple(candidate.candidate_id for candidate in registry)
        if args.candidate_ids is None else tuple(args.candidate_ids)
    )
    validate_candidate_phase(args.phase, candidate_ids)
    if args.epochs != 40 or args.folds != 6:
        raise ValueError("the frozen v3 search requires 40 updates and six folds")
    if (args.phase == "screen" and args.seed != 0) or (
        args.phase == "confirm" and args.seed not in (1, 2)
    ):
        raise ValueError("screen uses seed 0; confirmation uses seeds 1 and 2")

    started = time.monotonic()
    palsy, palsy_commitments = v1_runner._load_palsynet(args)
    neuroface, neuroface_collection = v1_runner._load_dense_profile(
        profile="neuroface",
        cache_root=args.neuroface_cache,
        collection_sha256=args.neuroface_collection_sha256,
        manifest_path=args.neuroface_manifest,
        manifest_sha256=args.neuroface_manifest_sha256,
    )
    meei, meei_collection = v1_runner._load_dense_profile(
        profile="meei",
        cache_root=args.meei_cache,
        collection_sha256=args.meei_collection_sha256,
        manifest_path=args.meei_manifest,
        manifest_sha256=args.meei_manifest_sha256,
    )
    dataset = v2_runner.pack_participant_bags_v2((*palsy, *neuroface, *meei))
    counts = {
        source: sum(observed == source for observed in dataset.base.sources)
        for source in SOURCES
    }
    if counts != {"palsynet": 38, "neuroface": 36, "meei": 56}:
        raise ValueError("shared v3 participant counts drifted")
    results = {}
    evaluations = {}
    for candidate_id in candidate_ids:
        result = evaluate_compact_candidate(
            dataset, lookup[candidate_id], epochs=args.epochs,
            n_splits=args.folds, seed=args.seed, device="cuda",
        )
        results[candidate_id] = result
        evaluations[candidate_id] = result.metrics
    ranking = (
        rank_compact_results(results)
        if args.phase == "screen" else tuple(sorted(
            candidate_ids,
            key=lambda candidate_id: _ranking_key(candidate_id, evaluations),
        ))
    )
    report = build_report(
        phase=args.phase,
        evaluations=evaluations,
        ranking=ranking,
        counts=counts,
        runtime={
            "gpu": torch.cuda.get_device_name(0),
            "epochs": args.epochs,
            "seed": args.seed,
            "folds": args.folds,
            "elapsed_seconds": time.monotonic() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        commitments={
            **palsy_commitments,
            "neuroface_collection_sha256": neuroface_collection,
            "meei_collection_sha256": meei_collection,
            "neuroface_manifest_sha256": args.neuroface_manifest_sha256,
            "meei_manifest_sha256": args.meei_manifest_sha256,
            "implementation_sha256": _implementation_sha256(),
        },
    )
    v1_runner.write_report_no_overwrite(args.output, report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
