#!/usr/bin/env python3
"""Run the frozen conflict-aware shared-router search on H200."""
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
from src.evaluation.conflict_aware_shared_search_v5 import (  # noqa: E402
    evaluate_conflict_aware_candidate,
    rank_conflict_aware_results,
)
from src.evaluation.shared_clinical_encoder_v1 import SOURCES  # noqa: E402
from src.models.conflict_aware_candidate_registry_v5 import (  # noqa: E402
    candidate_registry_v5,
)


_METRICS = {"accuracy", "balanced_accuracy", "auroc", "sensitivity", "specificity", "brier"}
_COSINES = {"palsynet__neuroface", "palsynet__meei", "neuroface__meei"}


def validate_candidate_phase(phase: str, candidate_ids: tuple[str, ...]) -> None:
    all_ids = tuple(item.candidate_id for item in candidate_registry_v5())
    if (
        type(candidate_ids) is not tuple
        or len(candidate_ids) != len(set(candidate_ids))
        or any(type(item) is not str or item not in all_ids for item in candidate_ids)
    ):
        raise ValueError("candidate identifiers differ from the frozen v5 registry")
    if phase == "screen":
        valid = candidate_ids == all_ids
    elif phase == "confirm":
        valid = len(candidate_ids) == 2
    else:
        raise ValueError("phase must be screen or confirm")
    if not valid:
        raise ValueError("candidate set differs from the frozen v5 phase")


def _ranking_key(candidate_id: str, evaluations: Mapping[str, Mapping[str, Mapping[str, float]]]):
    metrics = evaluations[candidate_id]
    balanced = [float(metrics[source]["balanced_accuracy"]) for source in SOURCES]
    specificity = [float(metrics[source]["specificity"]) for source in SOURCES]
    aurocs = [float(metrics[source]["auroc"]) for source in SOURCES]
    return (-min(balanced), -min(specificity), -min(aurocs), -float(np.mean(balanced)), candidate_id)


def build_report(
    *, phase: str,
    evaluations: Mapping[str, Mapping[str, Mapping[str, float]]],
    pre_cosines: Mapping[str, Mapping[str, float]],
    post_cosines: Mapping[str, Mapping[str, float]],
    ranking: tuple[str, ...], counts: Mapping[str, int],
    runtime: Mapping[str, object], commitments: Mapping[str, str],
) -> dict[str, object]:
    if type(evaluations) is not dict or type(pre_cosines) is not dict or type(post_cosines) is not dict:
        raise ValueError("v5 evidence must use exact dictionaries")
    ids = tuple(evaluations)
    validate_candidate_phase(phase, ids)
    if (
        set(pre_cosines) != set(ids) or set(post_cosines) != set(ids)
        or type(ranking) is not tuple or set(ranking) != set(ids) or len(ranking) != len(ids)
        or dict(counts) != {"palsynet": 38, "neuroface": 36, "meei": 56}
        or ranking != tuple(sorted(ids, key=lambda item: _ranking_key(item, evaluations)))
    ):
        raise ValueError("v5 ranking, counts, or gradient audit drifted")
    for item in ids:
        if set(evaluations[item]) != set(SOURCES):
            raise ValueError("v5 source metrics are incomplete")
        for source in SOURCES:
            values = evaluations[item][source]
            if set(values) != _METRICS or any(not np.isfinite(float(value)) for value in values.values()):
                raise ValueError("v5 metric schema drifted")
        for evidence in (pre_cosines[item], post_cosines[item]):
            if set(evidence) != _COSINES or any(
                not np.isfinite(float(value)) or not -1.0 <= float(value) <= 1.0
                for value in evidence.values()
            ):
                raise ValueError("v5 gradient cosine schema drifted")
    return {
        "schema_version": "conflict_aware_shared_router_v5_search",
        "status": "exposed_development_candidate_search_not_clinically_validated",
        "phase": phase,
        "model": {
            "name": "Conflict-Aware Shared Clinical Router v5",
            "locked_base_candidate": "NMR4-001",
            "representation_changed_from_v4": False,
            "shared_patient_embedding_dim": 64,
            "source_specific_layers": ["final_binary_head"],
            "task_head_gradients_projected": False,
            "source_identifier_input": False,
        },
        "candidate_registry": [{
            "candidate_id": item.candidate_id,
            "base_candidate_id": item.base_candidate_id,
            "projection_scope": item.projection_scope,
            "projection_strength": item.projection_strength,
        } for item in candidate_registry_v5()],
        "candidate_ids": list(ids),
        "counts": dict(counts),
        "evaluations": {item: {
            source: {metric: float(value) for metric, value in evaluations[item][source].items()}
            for source in SOURCES
        } for item in ids},
        "pre_projection_cosines": {item: dict(pre_cosines[item]) for item in ids},
        "post_projection_cosines": {item: dict(post_cosines[item]) for item in ids},
        "selection": {
            "primary_metric": "minimum_source_balanced_accuracy",
            "secondary_metric": "minimum_source_specificity",
            "tertiary_metric": "minimum_source_auroc",
            "quaternary_metric": "mean_source_balanced_accuracy",
            "ranking": list(ranking),
            "pass_gate": {"three_seed_minimum_balanced_accuracy": 0.90, "three_seed_minimum_specificity": 0.85},
        },
        "runtime": dict(runtime),
        "commitments": dict(commitments),
        "audit": {"palsynet_protected_reads": 0, "mayo_reads": 0, "mayo_predictions": 0},
        "decision": {"promotion_authorized": False, "clinical_claim_authorized": False},
    }


def _implementation_sha256() -> str:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/evaluation/conflict_aware_shared_search_v5.py",
        PROJECT_ROOT / "src/models/conflict_aware_candidate_registry_v5.py",
        PROJECT_ROOT / "scripts/run_shared_normal_manifold_search_v4.py",
        PROJECT_ROOT / "src/evaluation/shared_normal_manifold_search_v4.py",
        PROJECT_ROOT / "src/models/shared_normal_manifold_router_v4.py",
        PROJECT_ROOT / "src/models/normal_manifold_candidate_registry_v4.py",
        PROJECT_ROOT / "scripts/run_medically_gated_shared_search_v2.py",
        PROJECT_ROOT / "src/evaluation/medically_gated_shared_search_v2.py",
        PROJECT_ROOT / "src/models/medically_gated_shared_encoder_v2.py",
        PROJECT_ROOT / "src/models/medical_shared_candidate_registry_v2.py",
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
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode())
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, default=6)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA H200":
        raise RuntimeError("v5 search requires the verified NVIDIA H200")
    registry = candidate_registry_v5()
    lookup = {item.candidate_id: item for item in registry}
    ids = tuple(item.candidate_id for item in registry) if args.candidate_ids is None else tuple(args.candidate_ids)
    validate_candidate_phase(args.phase, ids)
    if args.epochs != 20 or args.folds != 6:
        raise ValueError("the frozen v5 search requires 20 updates and six folds")
    if (args.phase == "screen" and args.seed != 0) or (args.phase == "confirm" and args.seed not in (1, 2)):
        raise ValueError("screen uses seed 0; confirmation uses seeds 1 and 2")
    started = time.monotonic()
    palsy, palsy_commitments = v1_runner._load_palsynet(args)
    neuroface, neuroface_collection = v1_runner._load_dense_profile(
        profile="neuroface", cache_root=args.neuroface_cache,
        collection_sha256=args.neuroface_collection_sha256,
        manifest_path=args.neuroface_manifest, manifest_sha256=args.neuroface_manifest_sha256,
    )
    meei, meei_collection = v1_runner._load_dense_profile(
        profile="meei", cache_root=args.meei_cache,
        collection_sha256=args.meei_collection_sha256,
        manifest_path=args.meei_manifest, manifest_sha256=args.meei_manifest_sha256,
    )
    dataset = v2_runner.pack_participant_bags_v2((*palsy, *neuroface, *meei))
    counts = {source: sum(observed == source for observed in dataset.base.sources) for source in SOURCES}
    if counts != {"palsynet": 38, "neuroface": 36, "meei": 56}:
        raise ValueError("shared v5 participant counts drifted")
    results, evaluations, pre, post = {}, {}, {}, {}
    for candidate_id in ids:
        result = evaluate_conflict_aware_candidate(
            dataset, lookup[candidate_id], epochs=args.epochs,
            n_splits=args.folds, seed=args.seed, device="cuda",
        )
        results[candidate_id] = result
        evaluations[candidate_id] = result.metrics
        pre[candidate_id] = result.pre_projection_cosines
        post[candidate_id] = result.post_projection_cosines
    ranking = rank_conflict_aware_results(results) if args.phase == "screen" else tuple(
        sorted(ids, key=lambda item: _ranking_key(item, evaluations))
    )
    report = build_report(
        phase=args.phase, evaluations=evaluations, pre_cosines=pre,
        post_cosines=post, ranking=ranking, counts=counts,
        runtime={
            "gpu": torch.cuda.get_device_name(0), "epochs": args.epochs,
            "seed": args.seed, "folds": args.folds,
            "elapsed_seconds": time.monotonic() - started,
            "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
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
