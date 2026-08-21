#!/usr/bin/env python3
"""Fit and export the exact three-seed Shared V9 research ensemble on H200."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_dense_clinical_shared_encoder_v1 as v1_runner  # noqa: E402
from scripts import run_medically_gated_shared_search_v2 as v2_runner  # noqa: E402
from src.deployment.shared_v9_research_release import (  # noqa: E402
    RESEARCH_SEEDS,
    write_release,
)
from src.evaluation.broad_literature_shared_search_v9 import _fit_model  # noqa: E402
from src.evaluation.distilled_shared_search_v9 import (  # noqa: E402
    configure_deterministic_training_v9,
)
from src.evaluation.shared_clinical_encoder_v1 import (  # noqa: E402
    SOURCES,
    fit_clinical_scaler,
)
from src.models.broad_literature_candidate_registry_v9 import (  # noqa: E402
    candidate_registry_v9,
)


COUNTS = {"palsynet": 38, "neuroface": 36, "meei": 56}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def selected_candidate():
    rows = tuple(
        row for row in candidate_registry_v9() if row.candidate_id == "BLV9-009"
    )
    if (
        len(rows) != 1
        or rows[0].mechanism != "masked_clinical_reconstruction"
        or rows[0].inference_change != "training_only"
    ):
        raise RuntimeError("the locked V9 candidate drifted")
    return rows[0]


def build_public_provenance(
    *,
    git_commit: str,
    source_commitments: dict[str, str],
) -> dict[str, object]:
    if (
        type(git_commit) is not str
        or _COMMIT.fullmatch(git_commit) is None
        or type(source_commitments) is not dict
        or set(source_commitments) != set(SOURCES)
        or any(
            type(value) is not str or _SHA256.fullmatch(value) is None
            for value in source_commitments.values()
        )
    ):
        raise ValueError("public V9 provenance commitments are invalid")
    return {
        "git_commit": git_commit,
        "training_seeds": list(RESEARCH_SEEDS),
        "training_epochs": 20,
        "training_device": "NVIDIA H200",
        "source_counts": dict(COUNTS),
        "source_commitments": dict(source_commitments),
    }


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
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_name(0) not in {"NVIDIA H200", "NVIDIA H200 NVL"}
    ):
        raise RuntimeError("Shared V9 export requires the verified NVIDIA H200")
    runtime = torch.device("cuda")
    configure_deterministic_training_v9(runtime)

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
    if counts != COUNTS:
        raise ValueError("Shared V9 participant counts drifted")
    all_indices = np.arange(len(dataset.base.labels), dtype=np.int64)
    scaler = fit_clinical_scaler(dataset.base, all_indices)
    candidate = selected_candidate()
    models = []
    for seed in RESEARCH_SEEDS:
        print(f"START full-data Shared V9 seed={seed}", flush=True)
        model, _, _, covered = _fit_model(
            dataset,
            candidate,
            all_indices,
            epochs=20,
            local_seed=seed * 1009,
            runtime=runtime,
            audit_sources=True,
        )
        if covered != set(SOURCES):
            raise RuntimeError("a source failed the full-data shared-gradient audit")
        models.append(model.eval())
        print(f"DONE full-data Shared V9 seed={seed}", flush=True)

    provenance = build_public_provenance(
        git_commit=args.git_commit,
        source_commitments={
            "palsynet": str(palsy_commitments["palsynet_collection_sha256"]),
            "neuroface": neuroface_collection,
            "meei": meei_collection,
        },
    )
    manifest = write_release(
        args.output,
        models=tuple(models),
        scaler_mean=scaler.mean,
        scaler_scale=scaler.scale,
        provenance=provenance,
    )
    print(json.dumps({
        "release": str(args.output),
        "model_id": manifest["model_id"],
        "candidate_id": manifest["candidate_id"],
        "weights": manifest["weights"],
        "protected_palsynet_reads": 0,
        "mayo_reads": 0,
        "mayo_predictions": 0,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
