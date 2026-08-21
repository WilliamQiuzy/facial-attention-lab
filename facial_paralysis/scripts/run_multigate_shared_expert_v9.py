#!/usr/bin/env python3
"""Run the frozen paper-supported shared-expert V9 evaluation on H200."""
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
from src.evaluation.literature_grounded_shared_search_v9 import (  # noqa: E402
    candidate_is_non_degrading,
)
from src.evaluation.multigate_shared_expert_search_v9 import (  # noqa: E402
    evaluate_multigate_shared_expert_candidate,
)
from src.evaluation.shared_clinical_encoder_v1 import SOURCES  # noqa: E402
from src.evaluation.universal_orofacial_v1 import binary_metrics  # noqa: E402
from src.models.multigate_shared_expert_router_v9 import (  # noqa: E402
    candidate_registry_v9,
)


SEEDS = (0, 1, 2)
_COUNTS = {"palsynet": 38, "neuroface": 36, "meei": 56}
_IDS = ("MSE9-000", "MSE9-001")
_METRIC_KEYS = {
    "accuracy", "auroc", "balanced_accuracy", "sensitivity", "specificity",
    "brier",
}


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_metrics(metrics: object) -> None:
    if type(metrics) is not dict or set(metrics) != set(SOURCES):
        raise ValueError("shared-expert metrics require the exact three sources")
    for source in SOURCES:
        if (
            type(metrics[source]) is not dict
            or set(metrics[source]) != _METRIC_KEYS
            or any(not np.isfinite(float(value)) for value in metrics[source].values())
        ):
            raise ValueError("shared-expert metric rows must be closed and finite")


def _validate_evaluations(evaluations: object) -> None:
    if type(evaluations) is not dict or tuple(evaluations) != _IDS:
        raise ValueError("shared-expert V9 requires comparator and MMoE")
    for seed_rows in evaluations.values():
        if type(seed_rows) is not dict or set(seed_rows) != {
            str(seed) for seed in SEEDS
        }:
            raise ValueError("shared-expert V9 requires the exact three seeds")
        for row in seed_rows.values():
            if type(row) is not dict or set(row) != {
                "within_source", "leave_one_source_out", "model_fits",
                "task_specific_parameter_fraction",
            }:
                raise ValueError("shared-expert evaluation schema drifted")
            _validate_metrics(row["within_source"])
            _validate_metrics(row["leave_one_source_out"])
            if (
                row["model_fits"] != 9
                or not 0.0 <= float(row["task_specific_parameter_fraction"]) < 0.10
            ):
                raise ValueError("shared-expert fit count or capacity drifted")


def _mean_metrics(seed_rows: Mapping[str, Mapping[str, object]], key: str):
    return {
        source: {
            metric: float(np.mean([
                float(seed_rows[str(seed)][key][source][metric]) for seed in SEEDS
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
    if type(ensemble_metrics) is not dict or tuple(ensemble_metrics) != _IDS:
        raise ValueError("shared-expert ensembles must match the two candidates")
    for metrics in ensemble_metrics.values():
        _validate_metrics(metrics)
    if (
        dict(counts) != _COUNTS or type(runtime) is not dict or not runtime
        or type(commitments) is not dict or not commitments
        or any(not _is_sha256(value) for value in commitments.values())
    ):
        raise ValueError("shared-expert counts, runtime, or commitments drifted")
    summaries = {
        candidate_id: {
            "within_source_three_seed_mean": _mean_metrics(rows, "within_source"),
            "leave_one_source_out_three_seed_mean": _mean_metrics(
                rows, "leave_one_source_out"
            ),
            "three_seed_probability_ensemble": dict(ensemble_metrics[candidate_id]),
        }
        for candidate_id, rows in evaluations.items()
    }
    comparator = ensemble_metrics["MSE9-000"]
    observed = ensemble_metrics["MSE9-001"]
    promotion = (
        candidate_is_non_degrading(dict(observed), dict(comparator))
        and min(float(observed[source]["accuracy"]) for source in SOURCES) >= 0.90
        and min(float(observed[source]["specificity"]) for source in SOURCES) >= 0.80
        and min(float(observed[source]["auroc"]) for source in SOURCES) >= 0.92
        and min(float(observed[source]["specificity"]) for source in SOURCES)
        > min(float(comparator[source]["specificity"]) for source in SOURCES)
    )
    return {
        "schema_version": "multigate_shared_expert_v9_search",
        "status": "participant_disjoint_development_research_not_clinically_validated",
        "model": {
            "shared_experts": 3,
            "expert_rank": 16,
            "task_specific_component": "small gates and existing endpoint heads only",
            "paper_basis": "Multi-gate Mixture-of-Experts, KDD 2018",
            "all_experts_receive_all_three_source_gradients": True,
        },
        "candidate_registry": [row.__dict__ for row in candidate_registry_v9()],
        "evaluations": dict(evaluations),
        "summaries": summaries,
        "selection": {
            "primary_metric": "minimum_source_specificity",
            "secondary_metric": "minimum_source_auroc",
            "tertiary_metric": "minimum_source_accuracy",
            "promotion_estimator": "mean_probability_deep_ensemble",
            "seeds": list(SEEDS),
            "leave_one_source_out_role": "descriptive_transfer_stress_test_only",
        },
        "promotion_gate": {
            "minimum_accuracy_every_source": 0.90,
            "minimum_specificity_every_source": 0.80,
            "minimum_auroc_every_source": 0.92,
            "minimum_sensitivity_every_source": 0.85,
            "maximum_source_accuracy_or_auroc_regression": 0.01,
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
            "promoted_candidate_id": "MSE9-001" if promotion else None,
            "promotion_authorized": promotion,
            "clinical_claim_authorized": False,
        },
    }


def _implementation_sha256() -> str:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/evaluation/multigate_shared_expert_search_v9.py",
        PROJECT_ROOT / "src/models/multigate_shared_expert_router_v9.py",
        PROJECT_ROOT / "src/evaluation/residual_shared_search_v8.py",
        PROJECT_ROOT / "src/models/residual_shared_router_v8.py",
        PROJECT_ROOT / "src/models/script_aware_shared_router_v6.py",
        PROJECT_ROOT / "src/models/medically_gated_shared_encoder_v2.py",
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


def _serialize(result) -> dict[str, object]:
    return {
        "within_source": result.metrics,
        "leave_one_source_out": result.loso_metrics,
        "model_fits": result.model_fits,
        "task_specific_parameter_fraction": result.task_specific_parameter_fraction,
    }


def main() -> None:
    args = _parser().parse_args()
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_name(0) != "NVIDIA H200"
        or args.epochs != 20 or args.folds != 6
    ):
        raise RuntimeError("shared-expert V9 requires the verified H200")
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
        source: sum(value == source for value in dataset.base.sources)
        for source in SOURCES
    }
    if counts != _COUNTS:
        raise ValueError("shared-expert participant counts drifted")
    evaluations = {}
    raw_probabilities = {}
    for candidate in candidate_registry_v9():
        evaluations[candidate.candidate_id] = {}
        raw_probabilities[candidate.candidate_id] = []
        for seed in SEEDS:
            result = evaluate_multigate_shared_expert_candidate(
                dataset, candidate, epochs=args.epochs, n_splits=args.folds,
                seed=seed, device="cuda",
            )
            evaluations[candidate.candidate_id][str(seed)] = _serialize(result)
            raw_probabilities[candidate.candidate_id].append(result.probabilities)
    ensemble_metrics = {}
    for candidate_id, rows in raw_probabilities.items():
        probabilities = np.mean(np.stack(rows, axis=0), axis=0)
        ensemble_metrics[candidate_id] = {
            source: binary_metrics(
                dataset.base.labels[np.asarray([
                    value == source for value in dataset.base.sources
                ])],
                probabilities[np.asarray([
                    value == source for value in dataset.base.sources
                ])],
            )
            for source in SOURCES
        }
    report = build_report(
        evaluations=evaluations,
        ensemble_metrics=ensemble_metrics,
        counts=counts,
        runtime={
            "gpu": torch.cuda.get_device_name(0), "epochs": args.epochs,
            "folds": args.folds, "seeds": list(SEEDS),
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
