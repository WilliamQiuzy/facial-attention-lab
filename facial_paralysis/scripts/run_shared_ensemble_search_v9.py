#!/usr/bin/env python3
"""Evaluate deterministic deep ensembles of shared V8 encoders on H200."""
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
from src.evaluation.residual_shared_search_v8 import evaluate_residual_candidate  # noqa: E402
from src.evaluation.shared_clinical_encoder_v1 import SOURCES  # noqa: E402
from src.evaluation.shared_ensemble_search_v9 import (  # noqa: E402
    ensemble_candidate_registry_v9,
    evaluate_ensemble_candidate,
    rank_ensemble_results,
)
from src.models.residual_shared_router_v8 import candidate_registry_v8  # noqa: E402


_METRIC_KEYS = {
    "accuracy", "auroc", "balanced_accuracy", "sensitivity", "specificity", "brier"
}
_COUNTS = {"palsynet": 38, "neuroface": 36, "meei": 56}


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _ranking_key(
    candidate_id: str,
    evaluations: Mapping[str, Mapping[str, object]],
):
    metrics = evaluations[candidate_id]["metrics"]
    comparator = evaluations["SEN9-000"]["metrics"]
    feasible = all(
        float(metrics[source]["sensitivity"]) + 1e-12 >= 0.85
        and float(metrics[source]["accuracy"]) + 0.01 + 1e-12
        >= float(comparator[source]["accuracy"])
        and float(metrics[source]["auroc"]) + 0.01 + 1e-12
        >= float(comparator[source]["auroc"])
        for source in SOURCES
    )
    specificity = [float(metrics[source]["specificity"]) for source in SOURCES]
    auroc = [float(metrics[source]["auroc"]) for source in SOURCES]
    accuracy = [float(metrics[source]["accuracy"]) for source in SOURCES]
    balanced = [float(metrics[source]["balanced_accuracy"]) for source in SOURCES]
    return (
        not feasible,
        -min(specificity),
        -min(auroc),
        -min(accuracy),
        -min(balanced),
        -float(np.mean(accuracy)),
        candidate_id,
    )


def build_report(
    *,
    evaluations: Mapping[str, Mapping[str, object]],
    ranking: tuple[str, ...],
    counts: Mapping[str, int],
    runtime: Mapping[str, object],
    commitments: Mapping[str, str],
) -> dict[str, object]:
    registry = ensemble_candidate_registry_v9()
    ids = tuple(row.candidate_id for row in registry)
    if (
        type(evaluations) is not dict
        or set(evaluations) != set(ids)
        or type(ranking) is not tuple
        or set(ranking) != set(ids)
        or len(ranking) != len(ids)
        or dict(counts) != _COUNTS
        or type(commitments) is not dict
        or not commitments
        or any(not _is_sha256(value) for value in commitments.values())
    ):
        raise ValueError("ensemble report registry, counts, or commitments drifted")
    for row in registry:
        evaluation = evaluations[row.candidate_id]
        expected_members = len(row.member_candidate_ids) * len(row.seeds)
        if (
            set(evaluation) != {"metrics", "member_models", "model_fits"}
            or int(evaluation["member_models"]) != expected_members
            or int(evaluation["model_fits"]) != expected_members * 6
            or set(evaluation["metrics"]) != set(SOURCES)
        ):
            raise ValueError("ensemble evaluation schema drifted")
        for source in SOURCES:
            metrics = evaluation["metrics"][source]
            if (
                set(metrics) != _METRIC_KEYS
                or any(not np.isfinite(float(value)) for value in metrics.values())
            ):
                raise ValueError("ensemble metrics are incomplete or nonfinite")
    expected_ranking = tuple(sorted(
        ids, key=lambda candidate_id: _ranking_key(candidate_id, evaluations)
    ))
    if ranking != expected_ranking:
        raise ValueError("ensemble ranking differs from the frozen objectives")
    return {
        "schema_version": "shared_deep_ensemble_v9_search",
        "status": "participant_disjoint_development_search_not_clinically_validated",
        "candidate_registry": [
            {
                "candidate_id": row.candidate_id,
                "member_candidate_ids": list(row.member_candidate_ids),
                "seeds": list(row.seeds),
                "aggregation": row.aggregation,
            }
            for row in registry
        ],
        "medical_gate": {
            "phenomenon": (
                "Equal-weight deep ensembles reduce small-cohort initialization and "
                "adapter variance without using label-derived ensemble weights."
            ),
            "shared_encoder_in_every_member": True,
            "source_identifier_input": False,
            "random_visual_augmentation": False,
            "contralateral_assumed_normal": False,
            "hb_grade_claim_allowed": False,
        },
        "counts": dict(counts),
        "evaluations": dict(evaluations),
        "ranking": list(ranking),
        "selection": {
            "primary_metric": "minimum_source_specificity",
            "secondary_metric": "minimum_source_auroc",
            "tertiary_metric": "minimum_source_accuracy",
            "minimum_sensitivity": 0.85,
            "comparator_id": "SEN9-000",
            "ensemble_weights": "equal_and_label_independent",
        },
        "promotion_gate": {
            "minimum_accuracy": 0.90,
            "minimum_specificity": 0.80,
            "minimum_auroc": 0.92,
            "minimum_sensitivity": 0.85,
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
        PROJECT_ROOT / "src/evaluation/shared_ensemble_search_v9.py",
        PROJECT_ROOT / "src/evaluation/residual_shared_search_v8.py",
        PROJECT_ROOT / "src/models/residual_shared_router_v8.py",
        PROJECT_ROOT / "src/models/script_aware_shared_router_v6.py",
        PROJECT_ROOT / "src/models/medically_gated_shared_encoder_v2.py",
        PROJECT_ROOT / "src/models/medical_shared_candidate_registry_v2.py",
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
        not torch.cuda.is_available()
        or torch.cuda.get_device_name(0) != "NVIDIA H200"
        or args.epochs != 20
        or args.folds != 6
    ):
        raise RuntimeError("ensemble search requires the verified H200 schedule")
    started = time.monotonic()
    palsy, palsy_commitments = v1_runner._load_palsynet(args)
    if palsy_commitments.get("palsynet_protected_reads") != 0:
        raise RuntimeError("protected PalsyNet data were accessed")
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
    if counts != _COUNTS:
        raise ValueError("ensemble participant counts drifted")

    base_lookup = {row.candidate_id: row for row in candidate_registry_v8()}
    base_results = {}
    for candidate_id in tuple(base_lookup):
        for seed in (0, 1, 2):
            base_results[(candidate_id, seed)] = evaluate_residual_candidate(
                dataset,
                base_lookup[candidate_id],
                epochs=args.epochs,
                n_splits=args.folds,
                seed=seed,
                device="cuda",
            )
    ensemble_results = {
        row.candidate_id: evaluate_ensemble_candidate(
            dataset.base.labels, dataset.base.sources, base_results, row
        )
        for row in ensemble_candidate_registry_v9()
    }
    evaluations = {
        candidate_id: {
            "metrics": result.metrics,
            "member_models": result.member_models,
            "model_fits": result.model_fits,
        }
        for candidate_id, result in ensemble_results.items()
    }
    ranking = rank_ensemble_results(ensemble_results)
    report = build_report(
        evaluations=evaluations,
        ranking=ranking,
        counts=counts,
        runtime={
            "gpu": torch.cuda.get_device_name(0),
            "epochs": args.epochs,
            "folds": args.folds,
            "base_seeds": [0, 1, 2],
            "elapsed_seconds": time.monotonic() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
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
