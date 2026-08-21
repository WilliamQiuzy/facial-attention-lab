#!/usr/bin/env python3
"""Fit and publish the locked shared V8 / RSR8-001 research deployment model."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_dense_clinical_shared_encoder_v1 as v1_runner  # noqa: E402
from scripts import run_medically_gated_shared_search_v2 as v2_runner  # noqa: E402
from src.deployment.shared_v8_release import write_release  # noqa: E402
from src.evaluation.medically_gated_shared_search_v2 import (  # noqa: E402
    MedicalSharedDatasetV2,
    _model_inputs,
    _scaled,
    _tensor,
)
from src.evaluation.shared_clinical_encoder_v1 import (  # noqa: E402
    SOURCES,
    fit_clinical_scaler,
    source_class_balanced_weights,
)
from src.models.residual_shared_router_v8 import (  # noqa: E402
    ResidualSharedRouterV8,
    candidate_registry_v8,
)


FROZEN_CANDIDATE_ID = "RSR8-001"
FROZEN_EPOCHS = 20
FROZEN_SEED = 0
_SOURCE_TASK_CODE = {source: index for index, source in enumerate(SOURCES)}
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


@dataclass(frozen=True)
class FrozenFitV8:
    model: ResidualSharedRouterV8
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    shared_gradient_sources: tuple[str, ...]
    training_examples: int
    epochs: int
    final_loss: float


def _candidate():
    rows = tuple(
        row for row in candidate_registry_v8()
        if row.candidate_id == FROZEN_CANDIDATE_ID
    )
    if len(rows) != 1:
        raise RuntimeError("the frozen V8 candidate is unavailable")
    return rows[0]


def fit_full_dataset(
    dataset: MedicalSharedDatasetV2,
    *,
    epochs: int,
    seed: int,
    device: str,
) -> FrozenFitV8:
    """Fit once on all exposed participants after the candidate is locked."""
    if (
        type(dataset) is not MedicalSharedDatasetV2
        or isinstance(epochs, bool)
        or not isinstance(epochs, int)
        or epochs < 1
        or seed != FROZEN_SEED
        or type(device) is not str
    ):
        raise ValueError("full-data V8 fitting configuration is invalid")
    runtime = torch.device(device)
    if runtime.type not in {"cpu", "cuda"} or (
        runtime.type == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError("requested fitting device is unavailable")

    torch.manual_seed(seed)
    if runtime.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    base = dataset.base
    indices = np.arange(len(base.labels), dtype=np.int64)
    scaler = fit_clinical_scaler(base, indices)
    original = _scaled(base.clinical_original, scaler.mean, scaler.scale)
    mirrored = _scaled(base.clinical_mirrored, scaler.mean, scaler.scale)
    model = ResidualSharedRouterV8(_candidate()).to(runtime)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    sources = tuple(base.sources)
    weights = _tensor(
        source_class_balanced_weights(base.labels, sources).astype(np.float32),
        runtime,
    )
    labels = _tensor(base.labels.astype(np.float32), runtime)
    tasks = _tensor(np.asarray(
        [_SOURCE_TASK_CODE[source] for source in sources], dtype=np.int64
    ), runtime)
    inputs = _model_inputs(dataset, original, mirrored, indices, runtime)

    covered = []
    for source in SOURCES:
        local = torch.tensor(
            [index for index, observed in enumerate(sources) if observed == source],
            dtype=torch.long,
            device=runtime,
        )
        model.zero_grad(set_to_none=True)
        local_inputs = tuple(value.index_select(0, local) for value in inputs)
        tokens = model.shared_action_tokens(*local_inputs)
        loss = F.binary_cross_entropy_with_logits(
            model.routed_logits(
                tokens,
                local_inputs[-2],
                tasks.index_select(0, local),
            ),
            labels.index_select(0, local),
        )
        loss.backward()
        clinical = model.base.backbone.clinical_encoder[0].weight.grad
        patient = model.base.backbone.patient_projection.weight.grad
        if (
            clinical is None
            or patient is None
            or float(clinical.norm()) <= 0.0
            or float(patient.norm()) <= 0.0
        ):
            raise RuntimeError("a source failed the shared-trunk gradient audit")
        covered.append(source)

    final_loss = float("nan")
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        tokens = model.shared_action_tokens(*inputs)
        common = model.base.endpoint_embedding(tokens, inputs[-2], tasks)
        endpoint = model.adapt_endpoint(common, tasks)
        universal = model.base.universal_embedding(tokens, inputs[-2])
        task_logits = model.base.task_logits_from_embedding(endpoint, tasks)
        universal_logits = model.base.universal_head(universal).squeeze(-1)
        routed = 0.75 * task_logits + 0.25 * universal_logits
        losses = F.binary_cross_entropy_with_logits(
            routed, labels, reduction="none"
        ) + 0.5 * F.binary_cross_entropy_with_logits(
            universal_logits, labels, reduction="none"
        )
        loss = torch.sum(losses * weights)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("full-data V8 training produced a nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    model.eval()
    return FrozenFitV8(
        model=model,
        scaler_mean=np.array(scaler.mean, copy=True),
        scaler_scale=np.array(scaler.scale, copy=True),
        shared_gradient_sources=tuple(covered),
        training_examples=len(base.labels),
        epochs=epochs,
        final_loss=final_loss,
    )


def _aggregate_commitment(values: dict[str, object]) -> str:
    payload = (
        json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
    parser.add_argument("--epochs", type=int, default=FROZEN_EPOCHS)
    parser.add_argument("--seed", type=int, default=FROZEN_SEED)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_name(0) != "NVIDIA H200"
        or args.epochs != FROZEN_EPOCHS
        or args.seed != FROZEN_SEED
        or _COMMIT.fullmatch(args.git_commit) is None
    ):
        raise RuntimeError("freezing requires the locked H200, seed, epochs, and commit")
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
    if counts != {"palsynet": 38, "neuroface": 36, "meei": 56}:
        raise ValueError("deployment participant counts drifted")
    fit = fit_full_dataset(
        dataset, epochs=args.epochs, seed=args.seed, device="cuda"
    )
    source_commitments = {
        "palsynet": _aggregate_commitment(palsy_commitments),
        "neuroface": _aggregate_commitment({
            "collection_sha256": neuroface_collection,
            "manifest_sha256": args.neuroface_manifest_sha256,
        }),
        "meei": _aggregate_commitment({
            "collection_sha256": meei_collection,
            "manifest_sha256": args.meei_manifest_sha256,
        }),
    }
    manifest = write_release(
        args.output,
        model=fit.model,
        scaler_mean=fit.scaler_mean,
        scaler_scale=fit.scaler_scale,
        provenance={
            "git_commit": args.git_commit,
            "training_seed": args.seed,
            "training_epochs": args.epochs,
            "training_device": torch.cuda.get_device_name(0),
            "source_counts": counts,
            "source_commitments": source_commitments,
        },
    )
    print(json.dumps({
        "schema_version": "shared_v8_freeze_summary_v1",
        "model_id": manifest["model_id"],
        "weights_sha256": manifest["weights_sha256"],
        "training_examples": fit.training_examples,
        "epochs": fit.epochs,
        "final_loss": fit.final_loss,
        "shared_gradient_sources": list(fit.shared_gradient_sources),
    }, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
