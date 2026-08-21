#!/usr/bin/env python3
"""Run the bounded specificity-aware shared V9 search on the verified H200."""
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
from src.evaluation.shared_clinical_encoder_v1 import SOURCES  # noqa: E402
from src.evaluation.specificity_aware_shared_search_v9 import (  # noqa: E402
    SpecificityEvaluationV9,
    evaluate_specificity_candidate,
)
from src.models.specificity_aware_candidate_registry_v9 import (  # noqa: E402
    COMPONENT_RATIONALES_V9,
    candidate_registry_v9,
)


_COMPARATOR_ID = "SSR9-000"
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


def _evaluation_key(
    candidate_id: str,
    evaluations: Mapping[str, Mapping[str, object]],
):
    observed = evaluations[candidate_id]["calibrated"]
    comparator = evaluations[_COMPARATOR_ID]["calibrated"]
    feasible = True
    for source in SOURCES:
        metrics = observed[source]
        baseline = comparator[source]
        if (
            float(metrics["sensitivity"]) + 1e-12 < 0.85
            or float(metrics["accuracy"]) + 0.01 + 1e-12
            < float(baseline["accuracy"])
            or float(metrics["auroc"]) + 0.01 + 1e-12
            < float(baseline["auroc"])
        ):
            feasible = False
    specificity = [float(observed[source]["specificity"]) for source in SOURCES]
    auroc = [float(observed[source]["auroc"]) for source in SOURCES]
    accuracy = [float(observed[source]["accuracy"]) for source in SOURCES]
    balanced = [float(observed[source]["balanced_accuracy"]) for source in SOURCES]
    return (
        not feasible,
        -min(specificity),
        -min(auroc),
        -min(accuracy),
        -min(balanced),
        -float(np.mean(accuracy)),
        candidate_id,
    )


def _validate_evaluations(
    evaluations: Mapping[str, Mapping[str, object]],
    expected_ids: set[str],
) -> None:
    if type(evaluations) is not dict or set(evaluations) != expected_ids:
        raise ValueError("V9 evaluations differ from the frozen phase")
    for candidate_id, evaluation in evaluations.items():
        if type(candidate_id) is not str or set(evaluation) != {
            "fixed", "calibrated", "thresholds"
        }:
            raise ValueError("V9 evaluation schema drifted")
        for operating_point in ("fixed", "calibrated"):
            metrics = evaluation[operating_point]
            if type(metrics) is not dict or set(metrics) != set(SOURCES):
                raise ValueError("V9 source metrics drifted")
            for source in SOURCES:
                if (
                    set(metrics[source]) != _METRIC_KEYS
                    or any(
                        not np.isfinite(float(value))
                        for value in metrics[source].values()
                    )
                ):
                    raise ValueError("V9 metric values are not closed and finite")
        thresholds = evaluation["thresholds"]
        if (
            type(thresholds) is not dict
            or set(thresholds) != set(SOURCES)
            or any(
                type(thresholds[source]) is not list
                or len(thresholds[source]) != 6
                or any(
                    not np.isfinite(float(value))
                    or float(value) < 0.0
                    or float(value) > 1.0
                    for value in thresholds[source]
                )
                for source in SOURCES
            )
        ):
            raise ValueError("V9 fold thresholds drifted")


def candidate_ids_for_phase(
    phase: str,
    seed: int,
    screen_report: Mapping[str, object] | None,
) -> tuple[str, ...]:
    registry_ids = tuple(row.candidate_id for row in candidate_registry_v9())
    if phase == "screen":
        if seed != 0 or screen_report is not None:
            raise ValueError("V9 screening is the full registry at seed 0")
        return registry_ids
    if phase != "confirm" or seed not in (1, 2) or type(screen_report) is not dict:
        raise ValueError("V9 confirmation requires the locked screen and seed 1 or 2")
    required = {
        "schema_version", "status", "phase", "seed", "candidate_ids",
        "candidate_registry", "medical_gate", "counts", "evaluations", "ranking",
        "selection", "promotion_gate", "runtime", "commitments", "audit", "decision",
        "screen_report_sha256",
    }
    if (
        set(screen_report) != required
        or screen_report["schema_version"] != "specificity_aware_shared_router_v9_search"
        or screen_report["phase"] != "screen"
        or screen_report["seed"] != 0
        or tuple(screen_report["candidate_ids"]) != registry_ids
        or dict(screen_report["counts"]) != _COUNTS
        or dict(screen_report["audit"]) != {
            "palsynet_protected_reads": 0,
            "mayo_reads": 0,
            "mayo_predictions": 0,
        }
    ):
        raise ValueError("V9 screen report is not the frozen aggregate evidence")
    evaluations = screen_report["evaluations"]
    _validate_evaluations(evaluations, set(registry_ids))
    expected_ranking = tuple(sorted(
        registry_ids, key=lambda candidate_id: _evaluation_key(candidate_id, evaluations)
    ))
    if tuple(screen_report["ranking"]) != expected_ranking:
        raise ValueError("V9 screen ranking cannot be changed before confirmation")
    return expected_ranking[:3]


def build_report(
    *,
    phase: str,
    seed: int,
    candidate_ids: tuple[str, ...],
    evaluations: Mapping[str, Mapping[str, object]],
    ranking: tuple[str, ...],
    counts: Mapping[str, int],
    runtime: Mapping[str, object],
    commitments: Mapping[str, str],
    screen_report_sha256: str | None,
) -> dict[str, object]:
    registry = candidate_registry_v9()
    registry_ids = tuple(row.candidate_id for row in registry)
    if phase == "screen":
        valid_phase = (
            seed == 0
            and candidate_ids == registry_ids
            and screen_report_sha256 is None
            and set(evaluations) == set(registry_ids)
        )
    elif phase == "confirm":
        valid_phase = (
            seed in (1, 2)
            and len(candidate_ids) == 3
            and len(set(candidate_ids)) == 3
            and all(candidate_id in registry_ids for candidate_id in candidate_ids)
            and _is_sha256(screen_report_sha256)
            and set(evaluations) == set(candidate_ids) | {_COMPARATOR_ID}
        )
    else:
        valid_phase = False
    if (
        not valid_phase
        or dict(counts) != _COUNTS
        or type(ranking) is not tuple
        or set(ranking) != set(candidate_ids)
        or len(ranking) != len(candidate_ids)
        or type(commitments) is not dict
        or not commitments
        or any(not _is_sha256(value) for value in commitments.values())
    ):
        raise ValueError("V9 report phase, counts, ranking, or commitments drifted")
    _validate_evaluations(evaluations, set(evaluations))
    expected_ranking = tuple(sorted(
        candidate_ids,
        key=lambda candidate_id: _evaluation_key(candidate_id, evaluations),
    ))
    if ranking != expected_ranking:
        raise ValueError("V9 report ranking differs from the frozen objectives")
    return {
        "schema_version": "specificity_aware_shared_router_v9_search",
        "status": "participant_disjoint_development_search_not_clinically_validated",
        "phase": phase,
        "seed": seed,
        "candidate_ids": list(candidate_ids),
        "candidate_registry": [
            {
                "candidate_id": row.candidate_id,
                "healthy_mode": row.healthy_mode,
                "control_cost": row.control_cost,
                "universal_blend": row.universal_blend,
                "control_alignment_weight": row.control_alignment_weight,
            }
            for row in registry
        ],
        "medical_gate": {
            "rationales": COMPONENT_RATIONALES_V9,
            "shared_control_reference_only": True,
            "affected_centroid_alignment": False,
            "contralateral_assumed_normal": False,
            "source_identifier_input": False,
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
            "calibration": "outer_training_predictions_only_sensitivity_at_least_0.90",
            "comparator_id": _COMPARATOR_ID,
        },
        "promotion_gate": {
            "minimum_accuracy": 0.90,
            "minimum_specificity": 0.80,
            "minimum_auroc": 0.92,
            "minimum_sensitivity": 0.85,
            "maximum_same_seed_comparator_regression": 0.01,
            "requires_seeds": [0, 1, 2],
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
        "screen_report_sha256": screen_report_sha256,
    }


def _implementation_sha256() -> str:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/evaluation/specificity_aware_shared_search_v9.py",
        PROJECT_ROOT / "src/models/specificity_aware_shared_router_v9.py",
        PROJECT_ROOT / "src/models/specificity_aware_candidate_registry_v9.py",
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


def _read_report(path: Path) -> tuple[dict[str, object], str]:
    payload = path.read_bytes()
    try:
        report = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("screen report is not canonical JSON") from exc
    if type(report) is not dict:
        raise ValueError("screen report must be a JSON object")
    return report, hashlib.sha256(payload).hexdigest()


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
    parser.add_argument("--screen-report", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _serialized_evaluation(result: SpecificityEvaluationV9) -> dict[str, object]:
    return {
        "fixed": result.fixed_metrics,
        "calibrated": result.calibrated_metrics,
        "thresholds": {
            source: list(result.thresholds_by_source[source]) for source in SOURCES
        },
    }


def main() -> None:
    args = _parser().parse_args()
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_name(0) != "NVIDIA H200"
        or args.epochs != 20
        or args.folds != 6
    ):
        raise RuntimeError("V9 search requires the verified H200 and frozen schedule")
    screen_report = None
    screen_report_sha256 = None
    if args.screen_report is not None:
        screen_report, screen_report_sha256 = _read_report(args.screen_report)
    candidate_ids = candidate_ids_for_phase(args.phase, args.seed, screen_report)
    if (args.phase == "screen") != (args.screen_report is None):
        raise ValueError("screen report is required only for confirmation")

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
        raise ValueError("V9 participant counts drifted")

    lookup = {row.candidate_id: row for row in candidate_registry_v9()}
    evaluation_ids = tuple(dict.fromkeys((*candidate_ids, _COMPARATOR_ID)))
    raw_results = {
        candidate_id: evaluate_specificity_candidate(
            dataset,
            lookup[candidate_id],
            epochs=args.epochs,
            n_splits=args.folds,
            seed=args.seed,
            device="cuda",
        )
        for candidate_id in evaluation_ids
    }
    evaluations = {
        candidate_id: _serialized_evaluation(result)
        for candidate_id, result in raw_results.items()
    }
    ranking = tuple(sorted(
        candidate_ids,
        key=lambda candidate_id: _evaluation_key(candidate_id, evaluations),
    ))
    report = build_report(
        phase=args.phase,
        seed=args.seed,
        candidate_ids=candidate_ids,
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
        screen_report_sha256=screen_report_sha256,
    )
    v1_runner.write_report_no_overwrite(args.output, report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
