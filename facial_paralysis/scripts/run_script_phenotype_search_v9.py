#!/usr/bin/env python3
"""Run the deterministic full-mesh shared action-phenotype V9 search."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from src.evaluation.script_phenotype_search_v9 import (  # noqa: E402
    evaluate_phenotype_candidate,
    rank_phenotype_results,
)
from src.evaluation.shared_clinical_encoder_v1 import SOURCES  # noqa: E402
from src.models.script_phenotype_router_v9 import candidate_registry_v9  # noqa: E402


_COUNTS = {"palsynet": 38, "neuroface": 36, "meei": 56}
_METRICS = {
    "accuracy", "auroc", "balanced_accuracy", "sensitivity", "specificity", "brier"
}


def _is_sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _ranking_key(candidate_id: str, evaluations: Mapping[str, Mapping[str, object]]):
    metrics = evaluations[candidate_id]["metrics"]
    comparator = evaluations["SAP9-000"]["metrics"]
    feasible = all(
        float(metrics[source]["sensitivity"]) >= 0.85
        and float(metrics[source]["accuracy"]) + 0.01
        >= float(comparator[source]["accuracy"])
        and float(metrics[source]["auroc"]) + 0.01
        >= float(comparator[source]["auroc"])
        for source in SOURCES
    )
    return (
        not feasible,
        -min(float(metrics[source]["specificity"]) for source in SOURCES),
        -min(float(metrics[source]["auroc"]) for source in SOURCES),
        -min(float(metrics[source]["accuracy"]) for source in SOURCES),
        -float(np.mean([float(metrics[source]["accuracy"]) for source in SOURCES])),
        candidate_id,
    )


def build_report(
    *, evaluations: Mapping[str, Mapping[str, object]], ranking: tuple[str, ...],
    counts: Mapping[str, int], runtime: Mapping[str, object],
    commitments: Mapping[str, str],
) -> dict[str, object]:
    registry = candidate_registry_v9()
    ids = tuple(row.candidate_id for row in registry)
    if (
        type(evaluations) is not dict or set(evaluations) != set(ids)
        or type(ranking) is not tuple or set(ranking) != set(ids)
        or len(ranking) != len(ids) or dict(counts) != _COUNTS
        or type(commitments) is not dict or not commitments
        or any(not _is_sha(value) for value in commitments.values())
    ):
        raise ValueError("script phenotype report registry or commitments drifted")
    for candidate_id in ids:
        row = evaluations[candidate_id]
        if (
            set(row) != {"metrics", "model_fits", "task_specific_parameter_fraction"}
            or int(row["model_fits"]) != 6
            or not 0 <= float(row["task_specific_parameter_fraction"]) < 0.10
            or set(row["metrics"]) != set(SOURCES)
        ):
            raise ValueError("script phenotype evaluation schema drifted")
        for source in SOURCES:
            metrics = row["metrics"][source]
            if set(metrics) != _METRICS or any(
                not np.isfinite(float(value)) for value in metrics.values()
            ):
                raise ValueError("script phenotype metrics are incomplete")
    expected = tuple(sorted(ids, key=lambda value: _ranking_key(value, evaluations)))
    if ranking != expected:
        raise ValueError("script phenotype ranking differs from its frozen objective")
    return {
        "schema_version": "script_phenotype_router_v9_search",
        "status": "participant_disjoint_development_search_not_clinically_validated",
        "candidate_registry": [
            {
                "candidate_id": row.candidate_id,
                "phenotype_dim": row.phenotype_dim,
                "head_mode": row.head_mode,
                "universal_blend": row.universal_blend,
                "script_blend": row.script_blend,
                "medical_rationale": row.medical_rationale,
            }
            for row in registry
        ],
        "model": {
            "full_478d_plus_110d_shared_encoder": True,
            "shared_middle_layer": "RSR8_action_tokens_then_shared_motor_phenotypes",
            "task_specific_surface": "registered_script_weighting_and_binary_head_only",
            "source_identifier_input_to_shared_encoder": False,
            "task_specific_parameter_fraction_maximum": 0.10,
        },
        "counts": dict(counts), "evaluations": dict(evaluations),
        "ranking": list(ranking),
        "selection": {
            "primary_metric": "minimum_source_specificity",
            "secondary_metric": "minimum_source_auroc",
            "tertiary_metric": "minimum_source_accuracy",
            "minimum_sensitivity": 0.85, "comparator_id": "SAP9-000",
        },
        "promotion_gate": {
            "minimum_accuracy": 0.90, "minimum_specificity": 0.80,
            "minimum_auroc": 0.92, "minimum_sensitivity": 0.85,
        },
        "runtime": dict(runtime), "commitments": dict(commitments),
        "audit": {
            "palsynet_protected_reads": 0, "mayo_reads": 0,
            "mayo_predictions": 0, "patient_level_rows_published": 0,
        },
        "decision": {"promotion_authorized": False, "clinical_claim_authorized": False},
    }


def _implementation_sha256() -> str:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/evaluation/script_phenotype_search_v9.py",
        PROJECT_ROOT / "src/models/script_phenotype_router_v9.py",
        PROJECT_ROOT / "src/evaluation/residual_shared_search_v8.py",
        PROJECT_ROOT / "src/models/residual_shared_router_v8.py",
        PROJECT_ROOT / "src/models/script_aware_shared_router_v6.py",
        PROJECT_ROOT / "src/models/medically_gated_shared_encoder_v2.py",
        PROJECT_ROOT / "src/evaluation/distilled_shared_search_v9.py",
        PROJECT_ROOT / "src/evaluation/medically_gated_shared_search_v2.py",
        PROJECT_ROOT / "src/evaluation/shared_clinical_encoder_v1.py",
        PROJECT_ROOT / "src/preprocessing/shared_clinical_tokens_v1.py",
        PROJECT_ROOT / "src/preprocessing/clinical_landmarks.py",
        PROJECT_ROOT / "src/preprocessing/generalization_110d.py",
        PROJECT_ROOT / "src/preprocessing/trajectory_features.py",
        PROJECT_ROOT / "scripts/run_dense_clinical_shared_encoder_v1.py",
        PROJECT_ROOT / "scripts/run_medically_gated_shared_search_v2.py",
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if (
        not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA H200"
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        or args.epochs != 20 or args.folds != 6
    ):
        raise RuntimeError("script phenotype V9 requires the deterministic H200 schedule")
    started = time.monotonic()
    palsy, palsy_commitments = v1_runner._load_palsynet(args)
    if palsy_commitments.get("palsynet_protected_reads") != 0:
        raise RuntimeError("protected PalsyNet data were accessed")
    neuroface, neuroface_collection = v1_runner._load_dense_profile(
        profile="neuroface", cache_root=args.neuroface_cache,
        collection_sha256=args.neuroface_collection_sha256,
        manifest_path=args.neuroface_manifest,
        manifest_sha256=args.neuroface_manifest_sha256,
    )
    meei, meei_collection = v1_runner._load_dense_profile(
        profile="meei", cache_root=args.meei_cache,
        collection_sha256=args.meei_collection_sha256,
        manifest_path=args.meei_manifest,
        manifest_sha256=args.meei_manifest_sha256,
    )
    dataset = v2_runner.pack_participant_bags_v2((*palsy, *neuroface, *meei))
    counts = {
        source: sum(observed == source for observed in dataset.base.sources)
        for source in SOURCES
    }
    if counts != _COUNTS:
        raise ValueError("script phenotype participant counts drifted")
    results = {
        candidate.candidate_id: evaluate_phenotype_candidate(
            dataset, candidate, epochs=args.epochs, n_splits=args.folds,
            seed=0, device="cuda",
        )
        for candidate in candidate_registry_v9()
    }
    evaluations = {
        candidate_id: {
            "metrics": result.metrics, "model_fits": result.model_fits,
            "task_specific_parameter_fraction": result.task_specific_parameter_fraction,
        }
        for candidate_id, result in results.items()
    }
    report = build_report(
        evaluations=evaluations, ranking=rank_phenotype_results(results), counts=counts,
        runtime={
            "gpu": torch.cuda.get_device_name(0), "epochs": args.epochs,
            "folds": args.folds, "seed": 0,
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "elapsed_seconds": time.monotonic() - started,
            "python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__,
        },
        commitments={
            "palsynet_collection_sha256": str(
                palsy_commitments["palsynet_collection_sha256"]
            ),
            "palsynet_reviewed_manifest_sha256": str(
                palsy_commitments["palsynet_reviewed_manifest_sha256"]
            ),
            "palsynet_split_registry_sha256": str(
                palsy_commitments["palsynet_split_registry_sha256"]
            ),
            "neuroface_collection_sha256": neuroface_collection,
            "neuroface_manifest_sha256": args.neuroface_manifest_sha256,
            "meei_collection_sha256": meei_collection,
            "meei_manifest_sha256": args.meei_manifest_sha256,
            "implementation_sha256": _implementation_sha256(),
        },
    )
    v1_runner.write_report_no_overwrite(args.output, report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
