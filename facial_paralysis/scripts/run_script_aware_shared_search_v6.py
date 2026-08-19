#!/usr/bin/env python3
"""Run the frozen script-aware shared-router search on H200."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_dense_clinical_shared_encoder_v1 as v1_runner  # noqa: E402
from scripts import run_medically_gated_shared_search_v2 as v2_runner  # noqa: E402
from src.evaluation.script_aware_shared_search_v6 import (  # noqa: E402
    evaluate_script_aware_candidate, rank_script_aware_results,
)
from src.evaluation.shared_clinical_encoder_v1 import SOURCES  # noqa: E402
from src.models.script_aware_shared_router_v6 import candidate_registry_v6  # noqa: E402


def validate_candidate_phase(phase: str, ids: tuple[str, ...]) -> None:
    expected = tuple(item.candidate_id for item in candidate_registry_v6())
    if type(ids) is not tuple or len(ids) != len(set(ids)) or any(item not in expected for item in ids):
        raise ValueError("candidate identifiers differ from the frozen v6 registry")
    if phase == "screen": valid = ids == expected
    elif phase == "confirm": valid = len(ids) == 2
    else: raise ValueError("phase must be screen or confirm")
    if not valid: raise ValueError("candidate set differs from the frozen v6 phase")


def _rank(ids, evaluations):
    def key(item):
        metrics = evaluations[item]
        balanced = [float(metrics[source]["balanced_accuracy"]) for source in SOURCES]
        specificity = [float(metrics[source]["specificity"]) for source in SOURCES]
        aurocs = [float(metrics[source]["auroc"]) for source in SOURCES]
        return (-min(balanced), -min(specificity), -min(aurocs), -float(np.mean(balanced)), item)
    return tuple(sorted(ids, key=key))


def _implementation_sha256() -> str:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/evaluation/script_aware_shared_search_v6.py",
        PROJECT_ROOT / "src/models/script_aware_shared_router_v6.py",
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
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode()); digest.update(b"\0")
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
        raise RuntimeError("v6 search requires the verified NVIDIA H200")
    registry = candidate_registry_v6(); lookup = {item.candidate_id: item for item in registry}
    ids = tuple(item.candidate_id for item in registry) if args.candidate_ids is None else tuple(args.candidate_ids)
    validate_candidate_phase(args.phase, ids)
    if args.epochs != 20 or args.folds != 6: raise ValueError("v6 requires 20 updates and six folds")
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
    counts = {source: sum(value == source for value in dataset.base.sources) for source in SOURCES}
    if counts != {"palsynet": 38, "neuroface": 36, "meei": 56}: raise ValueError("v6 counts drifted")
    results = {}; evaluations = {}; cosines = {}
    for item in ids:
        result = evaluate_script_aware_candidate(
            dataset, lookup[item], epochs=args.epochs, n_splits=args.folds,
            seed=args.seed, device="cuda",
        )
        results[item] = result; evaluations[item] = result.metrics; cosines[item] = result.gradient_cosines
    ranking = rank_script_aware_results(results) if args.phase == "screen" else _rank(ids, evaluations)
    report = {
        "schema_version": "script_aware_shared_router_v6_search",
        "status": "exposed_development_candidate_search_not_clinically_validated",
        "phase": args.phase,
        "model": {
            "name": "Script-Aware Shared Clinical Router v6",
            "shared_layers": ["clinical_110d_encoder", "full_478_encoder", "regional_encoder", "cross_action_encoder", "patient_projection"],
            "endpoint_specific_layers": ["script_query", "binary_head"],
            "source_identifier_in_shared_encoder": False,
        },
        "candidate_registry": [item.__dict__ for item in registry],
        "candidate_ids": list(ids), "counts": counts,
        "evaluations": evaluations, "gradient_cosines": cosines,
        "selection": {
            "primary_metric": "minimum_source_balanced_accuracy",
            "secondary_metric": "minimum_source_specificity",
            "tertiary_metric": "minimum_source_auroc",
            "ranking": list(ranking),
        },
        "runtime": {
            "gpu": torch.cuda.get_device_name(0), "epochs": args.epochs,
            "seed": args.seed, "folds": args.folds,
            "elapsed_seconds": time.monotonic() - started,
            "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
        },
        "commitments": {
            **palsy_commitments, "neuroface_collection_sha256": neuroface_collection,
            "meei_collection_sha256": meei_collection,
            "neuroface_manifest_sha256": args.neuroface_manifest_sha256,
            "meei_manifest_sha256": args.meei_manifest_sha256,
            "implementation_sha256": _implementation_sha256(),
        },
        "audit": {"palsynet_protected_reads": 0, "mayo_reads": 0, "mayo_predictions": 0},
        "decision": {"promotion_authorized": False, "clinical_claim_authorized": False},
    }
    v1_runner.write_report_no_overwrite(args.output, report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__": main()
