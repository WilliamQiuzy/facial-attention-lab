#!/usr/bin/env python3
"""Run the frozen literature-grounded shared V9 evaluation on H200."""
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
    evaluate_literature_grounded_candidate,
)
from src.evaluation.shared_clinical_encoder_v1 import SOURCES  # noqa: E402
from src.models.literature_grounded_candidate_registry_v9 import (  # noqa: E402
    candidate_registry_v9,
)


SEEDS = (0, 1, 2)
_COUNTS = {"palsynet": 38, "neuroface": 36, "meei": 56}
_BASE_IDS = ("LGS9-000", "LGS9-001", "LGS9-002")
_COMBINED_ID = "LGS9-003"
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
        raise ValueError("V9 metrics must contain the exact three sources")
    for source in SOURCES:
        row = metrics[source]
        if (
            type(row) is not dict or set(row) != _METRIC_KEYS
            or any(not np.isfinite(float(value)) for value in row.values())
        ):
            raise ValueError("V9 metric rows must be closed and finite")


def _validate_evaluations(evaluations: object) -> None:
    registry_ids = tuple(row.candidate_id for row in candidate_registry_v9())
    if (
        type(evaluations) is not dict
        or tuple(evaluations) not in (_BASE_IDS, registry_ids)
    ):
        raise ValueError("V9 evaluations must be base candidates plus optional combination")
    for candidate_evaluations in evaluations.values():
        if type(candidate_evaluations) is not dict or set(candidate_evaluations) != {
            str(seed) for seed in SEEDS
        }:
            raise ValueError("every V9 candidate requires the exact three seeds")
        for row in candidate_evaluations.values():
            if type(row) is not dict or set(row) != {
                "within_source", "leave_one_source_out", "model_fits",
                "task_specific_parameter_fraction",
            }:
                raise ValueError("V9 evaluation schema drifted")
            _validate_source_metrics(row["within_source"])
            _validate_source_metrics(row["leave_one_source_out"])
            if (
                row["model_fits"] != 9
                or not np.isfinite(float(row["task_specific_parameter_fraction"]))
                or not 0.0 <= float(row["task_specific_parameter_fraction"]) < 0.10
            ):
                raise ValueError("V9 fit count or shared-capacity contract drifted")


def combination_is_authorized(
    evaluations: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> bool:
    if type(evaluations) is not dict or tuple(evaluations) != _BASE_IDS:
        raise ValueError("combination authorization requires the three base candidates")
    _validate_evaluations(evaluations)
    for seed in SEEDS:
        comparator = evaluations["LGS9-000"][str(seed)]["within_source"]
        for candidate_id in ("LGS9-001", "LGS9-002"):
            observed = evaluations[candidate_id][str(seed)]["within_source"]
            if not candidate_is_non_degrading(observed, comparator):
                return False
    return True


def _mean_source_metrics(seed_evaluations: Mapping[str, Mapping[str, object]], key: str):
    return {
        source: {
            metric: float(np.mean([
                float(seed_evaluations[str(seed)][key][source][metric])
                for seed in SEEDS
            ]))
            for metric in sorted(_METRIC_KEYS)
        }
        for source in SOURCES
    }


def _passes_promotion(
    observed: Mapping[str, Mapping[str, float]],
    comparator: Mapping[str, Mapping[str, float]],
) -> bool:
    return (
        candidate_is_non_degrading(dict(observed), dict(comparator))
        and min(float(observed[source]["accuracy"]) for source in SOURCES) >= 0.90
        and min(float(observed[source]["specificity"]) for source in SOURCES) >= 0.80
        and min(float(observed[source]["auroc"]) for source in SOURCES) >= 0.92
        and min(float(observed[source]["specificity"]) for source in SOURCES)
        > min(float(comparator[source]["specificity"]) for source in SOURCES)
    )


def build_report(
    *,
    evaluations: Mapping[str, Mapping[str, Mapping[str, object]]],
    counts: Mapping[str, int],
    runtime: Mapping[str, object],
    commitments: Mapping[str, str],
) -> dict[str, object]:
    _validate_evaluations(evaluations)
    authorized = combination_is_authorized({
        candidate_id: evaluations[candidate_id] for candidate_id in _BASE_IDS
    })
    if (_COMBINED_ID in evaluations) != authorized:
        raise ValueError("combined V9 candidate differs from its evidence authorization")
    if (
        dict(counts) != _COUNTS or type(runtime) is not dict or not runtime
        or type(commitments) is not dict or not commitments
        or any(not _is_sha256(value) for value in commitments.values())
    ):
        raise ValueError("V9 counts, runtime, or commitments drifted")

    summaries = {}
    for candidate_id, seed_rows in evaluations.items():
        summaries[candidate_id] = {
            "within_source_three_seed_mean": _mean_source_metrics(
                seed_rows, "within_source"
            ),
            "leave_one_source_out_three_seed_mean": _mean_source_metrics(
                seed_rows, "leave_one_source_out"
            ),
        }
    comparator = summaries["LGS9-000"]["within_source_three_seed_mean"]
    for candidate_id in summaries:
        summaries[candidate_id]["promotion_gate_passed"] = (
            candidate_id != "LGS9-000"
            and _passes_promotion(
                summaries[candidate_id]["within_source_three_seed_mean"], comparator
            )
        )

    def ranking_key(candidate_id: str):
        metrics = summaries[candidate_id]["within_source_three_seed_mean"]
        return (
            not summaries[candidate_id]["promotion_gate_passed"],
            -min(float(metrics[source]["specificity"]) for source in SOURCES),
            -min(float(metrics[source]["auroc"]) for source in SOURCES),
            -min(float(metrics[source]["accuracy"]) for source in SOURCES),
            candidate_id,
        )

    ranking = tuple(sorted(evaluations, key=ranking_key))
    promoted = next(
        (candidate_id for candidate_id in ranking
         if summaries[candidate_id]["promotion_gate_passed"]),
        None,
    )
    registry = candidate_registry_v9()
    return {
        "schema_version": "literature_grounded_shared_v9_search",
        "status": "participant_disjoint_development_research_not_clinically_validated",
        "candidate_registry": [
            {
                "candidate_id": row.candidate_id,
                "relation_enabled": row.relation_enabled,
                "auxiliary_weight": row.auxiliary_weight,
                "paper_basis": row.paper_basis,
                "medical_rationale": row.medical_rationale,
            }
            for row in registry
        ],
        "evaluations": dict(evaluations),
        "summaries": summaries,
        "combination_authorized": authorized,
        "selection": {
            "primary_metric": "minimum_source_specificity",
            "secondary_metric": "minimum_source_auroc",
            "tertiary_metric": "minimum_source_accuracy",
            "minimum_sensitivity": 0.85,
            "maximum_source_accuracy_or_auroc_regression": 0.01,
            "seeds": list(SEEDS),
            "ranking": list(ranking),
            "leave_one_source_out_role": "descriptive_transfer_stress_test_only",
        },
        "promotion_gate": {
            "minimum_accuracy_every_source": 0.90,
            "minimum_specificity_every_source": 0.80,
            "minimum_auroc_every_source": 0.92,
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
            "promoted_candidate_id": promoted,
            "promotion_authorized": promoted is not None,
            "clinical_claim_authorized": False,
        },
    }


def _implementation_sha256() -> str:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/evaluation/literature_grounded_shared_search_v9.py",
        PROJECT_ROOT / "src/models/literature_grounded_candidate_registry_v9.py",
        PROJECT_ROOT / "src/models/anatomical_relational_router_v9.py",
        PROJECT_ROOT / "src/training/clinical_kinematic_auxiliary_v9.py",
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
        raise RuntimeError("the frozen V9 search requires the verified H200")
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
        raise ValueError("V9 participant counts drifted")

    lookup = {row.candidate_id: row for row in candidate_registry_v9()}
    evaluations: dict[str, dict[str, object]] = {}
    for candidate_id in _BASE_IDS:
        evaluations[candidate_id] = {
            str(seed): _serialize(evaluate_literature_grounded_candidate(
                dataset, lookup[candidate_id], epochs=args.epochs,
                n_splits=args.folds, seed=seed, device="cuda",
            ))
            for seed in SEEDS
        }
    if combination_is_authorized(evaluations):
        evaluations[_COMBINED_ID] = {
            str(seed): _serialize(evaluate_literature_grounded_candidate(
                dataset, lookup[_COMBINED_ID], epochs=args.epochs,
                n_splits=args.folds, seed=seed, device="cuda",
            ))
            for seed in SEEDS
        }

    report = build_report(
        evaluations=evaluations,
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
    v1_runner.write_report_no_overwrite(args.output, report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
