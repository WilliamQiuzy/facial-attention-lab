#!/usr/bin/env python3
"""Run the frozen six-candidate bilateral reconstruction V10 screen on H200."""
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

from scripts import run_broad_literature_shared_search_v9 as v9_runner  # noqa: E402
from scripts import run_dense_clinical_shared_encoder_v1 as v1_runner  # noqa: E402
from scripts import run_medically_gated_shared_search_v2 as v2_runner  # noqa: E402
from src.evaluation.bilateral_reconstruction_shared_search_v10 import (  # noqa: E402
    candidate_is_promotable,
    evaluate_bilateral_reconstruction_candidate,
)
from src.evaluation.shared_clinical_encoder_v1 import SOURCES  # noqa: E402
from src.evaluation.universal_orofacial_v1 import binary_metrics  # noqa: E402
from src.models.bilateral_reconstruction_candidate_registry_v10 import (  # noqa: E402
    candidate_registry_v10,
)


SEEDS = (0, 1, 2)
_COUNTS = {"palsynet": 38, "neuroface": 36, "meei": 56}
_METRIC_KEYS = {
    "accuracy", "auroc", "balanced_accuracy", "sensitivity", "specificity",
    "brier",
}


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_source_metrics(metrics: object) -> None:
    if type(metrics) is not dict or set(metrics) != set(SOURCES):
        raise ValueError("V10 metrics require the exact three sources")
    for source in SOURCES:
        row = metrics[source]
        if (
            type(row) is not dict or set(row) != _METRIC_KEYS
            or any(not np.isfinite(float(value)) for value in row.values())
        ):
            raise ValueError("V10 metric rows must be closed and finite")


def _validate_evaluations(evaluations: object) -> None:
    expected = tuple(row.candidate_id for row in candidate_registry_v10())
    if type(evaluations) is not dict or tuple(evaluations) != expected:
        raise ValueError("every frozen bilateral reconstruction candidate is required")
    for candidate_id, seed_rows in evaluations.items():
        if type(seed_rows) is not dict or tuple(seed_rows) != tuple(
            str(seed) for seed in SEEDS
        ):
            raise ValueError("every V10 candidate requires the exact three seeds")
        for row in seed_rows.values():
            if type(row) is not dict or set(row) != {
                "within_source", "leave_one_source_out", "model_fits",
                "task_specific_parameter_fraction", "active_candidate_id",
            }:
                raise ValueError("V10 evaluation schema drifted")
            _validate_source_metrics(row["within_source"])
            _validate_source_metrics(row["leave_one_source_out"])
            if (
                row["model_fits"] != 9
                or row["active_candidate_id"] != candidate_id
                or not np.isfinite(float(row["task_specific_parameter_fraction"]))
                or not 0.0 <= float(row["task_specific_parameter_fraction"]) < 0.10
            ):
                raise ValueError("V10 fit, identity, or sharing contract drifted")


def _mean_source_metrics(seed_rows: Mapping[str, Mapping[str, object]], key: str):
    return {
        source: {
            metric: float(np.mean([
                float(seed_rows[str(seed)][key][source][metric])
                for seed in SEEDS
            ]))
            for metric in sorted(_METRIC_KEYS)
        }
        for source in SOURCES
    }


def build_report(
    *,
    evaluations: Mapping[str, Mapping[str, Mapping[str, object]]],
    ensemble_metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    counts: Mapping[str, int],
    runtime: Mapping[str, object],
    commitments: Mapping[str, str],
) -> dict[str, object]:
    _validate_evaluations(evaluations)
    registry = candidate_registry_v10()
    expected = tuple(row.candidate_id for row in registry)
    if type(ensemble_metrics) is not dict or tuple(ensemble_metrics) != expected:
        raise ValueError("ensemble metrics must match all V10 candidates")
    for metrics in ensemble_metrics.values():
        _validate_source_metrics(metrics)
    if (
        dict(counts) != _COUNTS or type(runtime) is not dict or not runtime
        or type(commitments) is not dict or not commitments
        or any(not _is_sha256(value) for value in commitments.values())
    ):
        raise ValueError("V10 counts, runtime, or commitments drifted")

    comparator = dict(ensemble_metrics["BRV10-000"])
    summaries = {}
    for candidate_id in expected:
        seed_rows = evaluations[candidate_id]
        observed = dict(ensemble_metrics[candidate_id])
        summaries[candidate_id] = {
            "within_source_three_seed_mean": _mean_source_metrics(
                seed_rows, "within_source"
            ),
            "leave_one_source_out_three_seed_mean": _mean_source_metrics(
                seed_rows, "leave_one_source_out"
            ),
            "three_seed_probability_ensemble": observed,
            "promotion_gate_passed": (
                candidate_id != "BRV10-000"
                and candidate_is_promotable(observed, comparator)
            ),
        }

    def ranking_key(candidate_id: str):
        metrics = summaries[candidate_id]["three_seed_probability_ensemble"]
        return (
            not summaries[candidate_id]["promotion_gate_passed"],
            -min(float(metrics[source]["specificity"]) for source in SOURCES),
            -min(float(metrics[source]["auroc"]) for source in SOURCES),
            -min(float(metrics[source]["accuracy"]) for source in SOURCES),
            candidate_id,
        )

    ranking = tuple(sorted(expected, key=ranking_key))
    promoted = next((
        candidate_id for candidate_id in ranking
        if summaries[candidate_id]["promotion_gate_passed"]
    ), None)
    return {
        "schema_version": "bilateral_reconstruction_shared_v10_search",
        "status": "participant_disjoint_development_research_not_clinically_validated",
        "candidate_registry": [
            {
                "candidate_id": row.candidate_id,
                "reconstruction_mode": row.reconstruction_mode,
                "optimizer_mode": row.optimizer_mode,
                "medical_rationale": row.medical_rationale,
                "contraindication": row.contraindication,
            }
            for row in registry
        ],
        "evaluations": dict(evaluations),
        "summaries": summaries,
        "selection": {
            "research_baseline": "BLV9-009/BRV10-000",
            "seeds": list(SEEDS),
            "primary_metric": "minimum_source_specificity",
            "secondary_metric": "minimum_source_auroc",
            "tertiary_metric": "minimum_source_accuracy",
            "promotion_estimator": "prespecified_three_seed_mean_probability",
            "ranking": list(ranking),
            "leave_one_source_out_role": "descriptive_transfer_stress_test_only",
        },
        "promotion_gate": {
            "minimum_accuracy_every_source": 0.90,
            "minimum_specificity_every_source": 0.80,
            "minimum_sensitivity_every_source": 0.85,
            "minimum_auroc_every_source": 0.92,
            "maximum_accuracy_or_auroc_regression": 0.01,
            "requires_strict_worst_source_specificity_improvement": True,
        },
        "counts": dict(counts),
        "runtime": dict(runtime),
        "commitments": dict(commitments),
        "audit": {
            "palsynet_protected_reads": 0,
            "mayo_reads": 0,
            "mayo_predictions": 0,
        },
        "decision": {
            "promoted_research_candidate_id": promoted,
            "research_promotion_authorized": promoted is not None,
            "deployment_model_changed": False,
            "clinical_claim_authorized": False,
        },
    }


write_release_atomic = v9_runner.write_release_atomic


def _implementation_sha256() -> str:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/evaluation/bilateral_reconstruction_shared_search_v10.py",
        PROJECT_ROOT / "src/models/bilateral_reconstruction_candidate_registry_v10.py",
        PROJECT_ROOT / "src/training/bilateral_masked_reconstruction_v10.py",
        PROJECT_ROOT / "src/evaluation/broad_literature_shared_search_v9.py",
        PROJECT_ROOT / "src/models/broad_literature_candidate_registry_v9.py",
        PROJECT_ROOT / "src/models/broad_literature_shared_router_v9.py",
        PROJECT_ROOT / "src/training/broad_literature_objectives_v9.py",
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
        PROJECT_ROOT / "scripts/run_broad_literature_shared_search_v9.py",
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


def _serialize(result) -> dict[str, object]:
    return {
        "within_source": result.metrics,
        "leave_one_source_out": result.loso_metrics,
        "model_fits": result.model_fits,
        "task_specific_parameter_fraction": result.task_specific_parameter_fraction,
        "active_candidate_id": result.active_candidate_id,
    }


def main() -> None:
    args = _parser().parse_args()
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_name(0) not in {"NVIDIA H200", "NVIDIA H200 NVL"}
        or args.epochs != 20 or args.folds != 6
    ):
        raise RuntimeError("the frozen bilateral V10 search requires the verified H200")
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
        raise ValueError("bilateral V10 participant counts drifted")

    evaluations: dict[str, dict[str, object]] = {}
    raw_probabilities: dict[str, list[np.ndarray]] = {}
    for candidate in candidate_registry_v10():
        evaluations[candidate.candidate_id] = {}
        raw_probabilities[candidate.candidate_id] = []
        for seed in SEEDS:
            print(
                f"START {candidate.candidate_id} "
                f"{candidate.reconstruction_mode}+{candidate.optimizer_mode} seed={seed}",
                flush=True,
            )
            result = evaluate_bilateral_reconstruction_candidate(
                dataset, candidate, epochs=args.epochs, n_splits=args.folds,
                seed=seed, device="cuda",
            )
            evaluations[candidate.candidate_id][str(seed)] = _serialize(result)
            raw_probabilities[candidate.candidate_id].append(result.probabilities)
            print(f"DONE {candidate.candidate_id} seed={seed}", flush=True)

    ensemble_metrics = {}
    for candidate_id, rows in raw_probabilities.items():
        probabilities = np.mean(np.stack(rows, axis=0), axis=0)
        ensemble_metrics[candidate_id] = {
            source: binary_metrics(
                dataset.base.labels[np.asarray([
                    observed == source for observed in dataset.base.sources
                ])],
                probabilities[np.asarray([
                    observed == source for observed in dataset.base.sources
                ])],
            )
            for source in SOURCES
        }
    report = build_report(
        evaluations=evaluations,
        ensemble_metrics=ensemble_metrics,
        counts=counts,
        runtime={
            "gpu": torch.cuda.get_device_name(0),
            "epochs": args.epochs,
            "folds": args.folds,
            "seeds": list(SEEDS),
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
    digest = write_release_atomic(args.output, report)
    print(json.dumps({
        "release": str(args.output),
        "report_sha256": digest,
        "decision": report["decision"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
