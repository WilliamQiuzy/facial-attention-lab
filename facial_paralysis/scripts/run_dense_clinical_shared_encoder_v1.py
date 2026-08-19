#!/usr/bin/env python3
"""Run the bounded H200 smoke for Dense-Clinical Shared Encoder v1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_110d_generalization_v1 as palsy_runner  # noqa: E402
from scripts import run_dense_action_router_v6 as dense_runner  # noqa: E402
from src.evaluation.shared_clinical_encoder_v1 import (  # noqa: E402
    SOURCES,
    SharedClinicalDataset,
    evaluate_shared_model,
)
from src.models.dense_clinical_shared_encoder_v1 import ACTION_VOCAB  # noqa: E402
from src.preprocessing.shared_clinical_tokens_v1 import (  # noqa: E402
    ClinicalActionBag,
    dense_action_token_bag,
    palsynet_window_token_bag,
)


V6_METRICS = {
    "palsynet": {"accuracy": 0.9473684210526315, "auroc": 0.980392156862745,
                  "balanced_accuracy": 0.9523809523809523},
    "neuroface": {"accuracy": 0.9444444444444444, "auroc": 0.989090909090909,
                   "balanced_accuracy": 0.96},
    "meei": {"accuracy": 0.9464285714285714, "auroc": 0.9456521739130436,
              "balanced_accuracy": 0.9673913043478262},
}


@dataclass(frozen=True)
class ParticipantBag:
    bag: ClinicalActionBag
    label: int
    group_id: str
    source: str


def _combine_bags(bags: tuple[ClinicalActionBag, ...]) -> ClinicalActionBag:
    if not bags or any(not isinstance(bag, ClinicalActionBag) for bag in bags):
        raise ValueError("participant aggregation requires clinical action bags")
    return ClinicalActionBag(
        clinical_original=np.concatenate([bag.clinical_original for bag in bags]),
        clinical_mirrored=np.concatenate([bag.clinical_mirrored for bag in bags]),
        dense_original=np.concatenate([bag.dense_original for bag in bags]),
        dense_mirrored=np.concatenate([bag.dense_mirrored for bag in bags]),
        dense_valid_mask=np.concatenate([bag.dense_valid_mask for bag in bags]),
        dense_available=np.concatenate([bag.dense_available for bag in bags]),
        dense_timestamps=np.concatenate([bag.dense_timestamps for bag in bags]),
        action_names=tuple(name for bag in bags for name in bag.action_names),
    )


def pack_participant_bags(rows: tuple[ParticipantBag, ...]) -> SharedClinicalDataset:
    if type(rows) is not tuple or len(rows) < 2:
        raise ValueError("participant packer requires an exact nontrivial tuple")
    if any(not isinstance(row, ParticipantBag) for row in rows):
        raise ValueError("participant rows must be exact ParticipantBag values")
    maximum = max(len(row.bag.action_names) for row in rows)
    count = len(rows)
    clinical_original = np.zeros((count, maximum, 110), dtype=np.float32)
    clinical_mirrored = np.zeros_like(clinical_original)
    dense_original = np.zeros((count, maximum, 32, 478, 3), dtype=np.float32)
    dense_mirrored = np.zeros_like(dense_original)
    dense_valid = np.zeros((count, maximum, 32), dtype=bool)
    dense_available = np.zeros((count, maximum), dtype=bool)
    action_mask = np.zeros((count, maximum), dtype=bool)
    action_codes = np.zeros((count, maximum), dtype=np.int64)
    action_index = {name: index for index, name in enumerate(ACTION_VOCAB)}
    labels = np.empty(count, dtype=np.int64)
    groups = []
    sources = []
    for participant, row in enumerate(rows):
        bag = row.bag
        actions = len(bag.action_names)
        if (
            row.label not in (0, 1)
            or row.source not in SOURCES
            or any(name not in action_index for name in bag.action_names)
            or bag.clinical_original.shape != (actions, 110)
            or bag.clinical_mirrored.shape != (actions, 110)
            or bag.dense_original.shape != (actions, 32, 478, 3)
            or bag.dense_mirrored.shape != (actions, 32, 478, 3)
            or bag.dense_valid_mask.shape != (actions, 32)
            or bag.dense_available.shape != (actions,)
        ):
            raise ValueError("participant bag differs from the shared input contract")
        clinical_original[participant, :actions] = bag.clinical_original
        clinical_mirrored[participant, :actions] = bag.clinical_mirrored
        dense_original[participant, :actions] = bag.dense_original
        dense_mirrored[participant, :actions] = bag.dense_mirrored
        dense_valid[participant, :actions] = bag.dense_valid_mask
        dense_available[participant, :actions] = bag.dense_available
        action_mask[participant, :actions] = True
        action_codes[participant, :actions] = [
            action_index[name] for name in bag.action_names
        ]
        labels[participant] = row.label
        groups.append(row.group_id)
        sources.append(row.source)
    return SharedClinicalDataset(
        clinical_original=clinical_original,
        clinical_mirrored=clinical_mirrored,
        dense_original=dense_original,
        dense_mirrored=dense_mirrored,
        dense_valid_mask=dense_valid,
        dense_available=dense_available,
        action_mask=action_mask,
        action_codes=action_codes,
        labels=labels,
        group_ids=tuple(groups),
        sources=tuple(sources),
    )


def _load_palsynet(args) -> tuple[tuple[ParticipantBag, ...], dict[str, object]]:
    manifest, manifest_sha = palsy_runner._read_json(args.reviewed_identity_manifest)
    ledger, ledger_sha = palsy_runner._read_json(args.review_ledger)
    registry, registry_sha = palsy_runner._read_json(args.split_registry)
    dataset, collection_rows, source_collection_sha = (
        palsy_runner._build_cache_metadata_dataset(args.palsynet_cache_root)
    )
    audit = palsy_runner.GateAudit()
    gate = palsy_runner.validate_development_gate(
        dataset,
        manifest,
        ledger,
        registry,
        reviewed_manifest_sha256=manifest_sha,
        review_ledger_sha256=ledger_sha,
        split_registry_sha256=registry_sha,
        cache_source_sha256_by_recording_id={
            recording_id: str(row["source_sha256"])
            for recording_id, row in collection_rows.items()
        },
        cache_source_collection_sha256=source_collection_sha,
        audit=audit,
    )
    palsy_runner.load_development_cache_records(
        args.palsynet_cache_root,
        dataset,
        gate,
        collection_rows,
        audit=audit,
    )
    grouped: dict[str, list[tuple[int, ClinicalActionBag]]] = {}
    for index in gate.development_indices.tolist():
        group = str(gate.group_ids[index])
        bag = palsynet_window_token_bag(
            dataset.features[index],
            dataset.valid_masks[index],
            dataset.timestamps[index],
            dataset.source_frame_indices[index],
        )
        grouped.setdefault(group, []).append((int(dataset.labels[index]), bag))
    rows = []
    for group in sorted(grouped):
        labels = {label for label, _ in grouped[group]}
        if len(labels) != 1:
            raise ValueError("one PalsyNet participant changed label")
        rows.append(ParticipantBag(
            bag=_combine_bags(tuple(bag for _, bag in grouped[group])),
            label=labels.pop(),
            group_id=group,
            source="palsynet",
        ))
    if len(rows) != 38 or audit.protected_cache_records_loaded != 0:
        raise ValueError("PalsyNet development boundary or participant count drifted")
    return tuple(rows), {
        "palsynet_collection_sha256": dataset.collection_manifest_sha256,
        "palsynet_reviewed_manifest_sha256": manifest_sha,
        "palsynet_split_registry_sha256": registry_sha,
        "palsynet_protected_reads": int(audit.protected_cache_records_loaded),
    }


def _load_dense_profile(
    *,
    profile: str,
    cache_root: Path,
    collection_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
) -> tuple[tuple[ParticipantBag, ...], str]:
    _, labels = dense_runner._read_manifest(manifest_path, manifest_sha256)
    caches, observed_collection_sha = dense_runner._load_collection(
        cache_root, profile, collection_sha256, manifest_sha256
    )
    grouped: dict[str, list[ClinicalActionBag]] = {}
    for cache in caches:
        bag = dense_action_token_bag(
            cache.original_actions,
            cache.mirrored_actions,
            cache.action_valid,
            cache.action_frame_indices,
            cache.original_baselines,
            cache.mirrored_baselines,
            cache.baseline_valid,
            fps=cache.fps,
            action_names=cache.action_names,
        )
        grouped.setdefault(cache.group_id, []).append(bag)
    validate_dense_membership(profile, set(grouped), set(labels))
    rows = tuple(
        ParticipantBag(
            bag=_combine_bags(tuple(grouped[group])),
            label=int(labels[group]),
            group_id=group,
            source=profile,
        )
        for group in sorted(grouped)
    )
    return rows, observed_collection_sha


def validate_dense_membership(
    profile: str,
    retained_groups: set[str],
    manifest_groups: set[str],
) -> None:
    """Bind exact cache counts while allowing MEEI's frozen four exclusions."""
    if (
        type(retained_groups) is not set
        or type(manifest_groups) is not set
        or not retained_groups
        or any(type(group) is not str for group in retained_groups | manifest_groups)
    ):
        raise ValueError("dense membership requires nonempty exact group sets")
    if profile == "neuroface":
        valid = len(retained_groups) == 36 and retained_groups == manifest_groups
    elif profile == "meei":
        valid = (
            len(retained_groups) == 56
            and len(manifest_groups) == 60
            and retained_groups.issubset(manifest_groups)
        )
    else:
        raise ValueError("unknown dense profile")
    if not valid:
        raise ValueError("dense participant membership differs from its manifest")


def _implementation_sha256() -> str:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/evaluation/shared_clinical_encoder_v1.py",
        PROJECT_ROOT / "src/models/dense_clinical_shared_encoder_v1.py",
        PROJECT_ROOT / "src/preprocessing/shared_clinical_tokens_v1.py",
        PROJECT_ROOT / "src/preprocessing/clinical_landmarks.py",
        PROJECT_ROOT / "src/preprocessing/generalization_110d.py",
        PROJECT_ROOT / "src/preprocessing/trajectory_features.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        payload = path.read_bytes()
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def build_smoke_report(
    *,
    evaluations: Mapping[str, Mapping[str, Mapping[str, float]]],
    counts: Mapping[str, int],
    runtime: Mapping[str, object],
    commitments: Mapping[str, str],
) -> dict[str, object]:
    if set(evaluations) != {"110d_only", "dense_clinical"}:
        raise ValueError("smoke requires exactly the two frozen candidates")
    if set(counts) != set(SOURCES) or any(counts[source] < 1 for source in SOURCES):
        raise ValueError("smoke counts must cover all three sources")
    return {
        "schema_version": "dense_clinical_shared_encoder_v1_smoke",
        "status": "exposed_development_smoke_not_clinically_validated",
        "model": {
            "name": "Dense-Clinical Shared Encoder v1",
            "primary_head": (
                "source_specific_heads_after_one_shared_patient_embedding"
            ),
            "universal_auxiliary_head_weight": 0.25,
            "shared_layers": [
                "clinical_encoder", "dense_spatial_temporal_encoder",
                "gated_fusion", "cross_action_transformer", "patient_embedding",
            ],
            "source_identifier_input": False,
        },
        "counts": dict(counts),
        "evaluations": {
            candidate: {source: dict(metrics[source]) for source in SOURCES}
            for candidate, metrics in evaluations.items()
        },
        "comparators": {
            "v6": {
                "shared_encoder": False,
                "metrics": V6_METRICS,
                "comparison_scope": "descriptive_nonshared_exposed_development",
            }
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
            "formal_training_authorized": True,
            "reason": "smoke_only_requires_formal_three_seed_and_leave_one_source_out",
        },
    }


def write_report_no_overwrite(path: Path, report: Mapping[str, object]) -> None:
    payload = (
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short report write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, default=6)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA H200":
        raise RuntimeError("the v1 smoke requires the verified NVIDIA H200")
    started = time.monotonic()
    palsy, palsy_commitments = _load_palsynet(args)
    neuroface, neuroface_collection = _load_dense_profile(
        profile="neuroface",
        cache_root=args.neuroface_cache,
        collection_sha256=args.neuroface_collection_sha256,
        manifest_path=args.neuroface_manifest,
        manifest_sha256=args.neuroface_manifest_sha256,
    )
    meei, meei_collection = _load_dense_profile(
        profile="meei",
        cache_root=args.meei_cache,
        collection_sha256=args.meei_collection_sha256,
        manifest_path=args.meei_manifest,
        manifest_sha256=args.meei_manifest_sha256,
    )
    dataset = pack_participant_bags((*palsy, *neuroface, *meei))
    observed_counts = {
        source: sum(observed == source for observed in dataset.sources)
        for source in SOURCES
    }
    if observed_counts != {"palsynet": 38, "neuroface": 36, "meei": 56}:
        raise ValueError("shared smoke participant counts drifted")
    evaluations = {}
    for name, use_dense in (("110d_only", False), ("dense_clinical", True)):
        result = evaluate_shared_model(
            dataset,
            use_dense=use_dense,
            epochs=args.epochs,
            n_splits=args.folds,
            seed=args.seed,
            device="cuda",
        )
        evaluations[name] = result.metrics
    report = build_smoke_report(
        evaluations=evaluations,
        counts=observed_counts,
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
            **palsy_commitments,
            "neuroface_collection_sha256": neuroface_collection,
            "meei_collection_sha256": meei_collection,
            "neuroface_manifest_sha256": args.neuroface_manifest_sha256,
            "meei_manifest_sha256": args.meei_manifest_sha256,
            "implementation_sha256": _implementation_sha256(),
        },
    )
    write_report_no_overwrite(args.output, report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
