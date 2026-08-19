#!/usr/bin/env python3
"""Reconstruct private UCR4 development OOF expert profiles on the H200.

This script is intentionally private-output only.  It verifies the exact helper
snapshot that produced UCR4, reconstructs group-disjoint probabilities, replaces
all group names by deterministic anonymous row tokens, and writes no report.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np


HELPER_SHA256 = {
    ".firecrawl/meei_cue_sequence_architecture_v3.py": "0e67e3c9b8251a62d72d5f98f80fbc5c3b28a058532ee2040c6c2642a737b2ba",
    ".firecrawl/meei_nested_sequence_confirmation_v3.py": "8adfcc3b96350ba8d57372743afc55e10fa38ffd4189687857a5993e4ce5111d",
    ".firecrawl/neuroface_full_au_explore_v3.py": "10d002225bd630a3a3b700c4b0f0006729dfa6f76649a071c8b94ca9a6d92e6d",
    ".firecrawl/neuroface_marlin_loop_v4.py": "a88a5d4dca4401586a9a6329414f3b9316eec84c985d4e2174be6a3750ca877b",
    ".firecrawl/neuroface_mechanism_loop_v4.py": "afd64ee1857a011798023707c1857350a6bf8f941c198d5f3eed8336904b4a75",
    ".firecrawl/neuroface_nested_marlin_gate_v4.py": "7d4df080ee5b8dbe547d0aaf19914c7e9a31ba051d57a727edd6e1ed83f00f0b",
    ".firecrawl/neuroface_nested_phenotype_confirmation_v3.py": "f938ae4ba8a7126a539dbaaeaf39734c5706e162ebf18da576598e8d8665477d",
    ".firecrawl/neuroface_phenotype_expert_quick_v3.py": "f55635492c6bde0fcacd6815f8d032f998d1f5b2f7182b7c3aad9f5083b9e7dd",
    ".firecrawl/shared_landmark_capacity_v3.py": "2ff16bb9b8275f44b79dbffe6dbb3c4a5146dc7464bbb424c99a08a61317699b",
    "scripts/freeze_palsynet_person_split_registry.py": "17feaa6a1824cb580e24964e938efce590cfaa06f320e5f3a86021cf838e17c1",
    "scripts/run_110d_generalization_v1.py": "776b572784ff4e0632ad446c1f81972a7896bde60dca30d6920971c4551c9033",
    "scripts/run_dynamic_landmark_classical.py": "71242a103ee5c8c1a576353a397ee36c3ba1cbf6c9ffdb816395dcd3d0e639c6",
    "scripts/run_mirror_invariant_110d.py": "ea41d076230665b55bcd9f2b0b9e047c3d67558ddd48d3271cb20d06e4f03c12",
    "src/datasets/dynamic_landmark.py": "7455da6baab0a83aaec061a81f4add932d18e0d58acf7fe8f1c914945cdc9e9e",
    "src/evaluation/universal_phenotype_v3.py": "3da79f2668c63e8e80619e5cc8a64931f0220f1723627e43a428e0259218977a",
    "src/preprocessing/clinical_dynamics.py": "0cd057ae1e8a1d94a05ed534c78532dd1f88a84959d25823cedc70273937c8f8",
    "src/preprocessing/generalization_110d.py": "ba02c9f416fa7e352e3c24b61debcb3500ec431d46c4325f89537f76f20b1ea3",
    "src/preprocessing/trajectory_features.py": "0a0900c9df418ce2856175d4027dbedd7db99d4873985bf3d61e7fa8be6d5ff7",
}
_PROFILE_NAMES = (
    "free_asymmetry",
    "scripted_multimechanism",
    "cue_aligned_upper",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _helper_aggregate_sha256() -> str:
    payload = (
        json.dumps(HELPER_SHA256, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    return _sha256(payload)


def verify_helper_root(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("helper root must be one real directory")
    for relative, expected in HELPER_SHA256.items():
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"frozen helper {relative!r} is unavailable")
        if _sha256(path.read_bytes()) != expected:
            raise ValueError(f"frozen helper {relative!r} differs")


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def _deterministic_npz(fields: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(fields):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(fields[name]))
    return output.getvalue()


def build_private_profile_bytes(
    evidence_profile: str,
    labels: np.ndarray,
    final_probability: np.ndarray,
    component_probability: np.ndarray,
    *,
    decision_threshold: np.ndarray,
) -> bytes:
    """Serialize one deterministic, deidentified exact-schema private profile."""
    if evidence_profile not in _PROFILE_NAMES:
        raise ValueError("evidence profile is not registered")
    if (
        type(labels) is not np.ndarray
        or labels.dtype != np.dtype(np.int64)
        or labels.ndim != 1
        or labels.size == 0
        or set(labels.tolist()) != {0, 1}
        or type(final_probability) is not np.ndarray
        or final_probability.dtype != np.dtype(np.float64)
        or final_probability.ndim != 1
        or final_probability.shape != labels.shape
        or type(component_probability) is not np.ndarray
        or component_probability.dtype != np.dtype(np.float64)
        or component_probability.ndim != 2
        or component_probability.shape[0] != labels.size
        or component_probability.shape[1] < 2
        or not np.isfinite(final_probability).all()
        or not np.isfinite(component_probability).all()
        or np.any((final_probability < 0.0) | (final_probability > 1.0))
        or np.any((component_probability < 0.0) | (component_probability > 1.0))
        or type(decision_threshold) is not np.ndarray
        or decision_threshold.dtype != np.dtype(np.float64)
        or decision_threshold.ndim != 1
        or decision_threshold.shape != labels.shape
        or not np.isfinite(decision_threshold).all()
        or np.any((decision_threshold <= 0.0) | (decision_threshold >= 1.0))
    ):
        raise ValueError("private profile values violate the frozen schema")
    groups = np.asarray(
        [f"anonymous_{index:04d}" for index in range(labels.size)]
    )
    return _deterministic_npz({
        "schema_version": np.asarray("selective_router_v5_private_profile"),
        "evidence_profile": np.asarray(evidence_profile),
        "anonymous_groups": groups,
        "labels": labels,
        "final_probability": final_probability,
        "component_probability": component_probability,
        "decision_threshold": decision_threshold,
    })


def publish_private_payload(path: Path, payload: bytes) -> str:
    """Publish owner-private exact bytes without overwriting an existing file."""
    if not isinstance(path, Path) or path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite private selective evidence")
    if type(payload) is not bytes or not payload:
        raise ValueError("private payload must be nonempty exact bytes")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            if stream.write(payload) != len(payload):
                raise OSError("short private evidence write")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(payload)


def validate_development_threshold_aggregation(
    artifact_threshold: float,
    oof_thresholds: np.ndarray,
) -> dict[str, object]:
    """Bind a close final aggregate without replacing fold-specific OOF rules."""
    values = np.asarray(oof_thresholds)
    if (
        isinstance(artifact_threshold, bool)
        or not isinstance(artifact_threshold, (int, float))
        or not math.isfinite(float(artifact_threshold))
        or type(oof_thresholds) is not np.ndarray
        or values.dtype != np.dtype(np.float64)
        or values.ndim != 1
        or values.size < 2
        or not np.isfinite(values).all()
        or np.any((values <= 0.0) | (values >= 1.0))
    ):
        raise ValueError("development threshold aggregation is not canonical")
    median = float(np.median(np.unique(values)))
    delta = abs(float(artifact_threshold) - median)
    if delta > 0.01:
        raise ValueError("artifact threshold materially differs from nested folds")
    return {
        "evaluation_scope": "per_participant_oof",
        "artifact_threshold": float(artifact_threshold),
        "outer_threshold_median": median,
        "absolute_delta": delta,
        "artifact_differs_from_oof_median": delta > 1e-12,
    }


def _configure_legacy_imports(helper_root: Path) -> None:
    sys.path.insert(0, str(helper_root / ".firecrawl"))
    sys.path.insert(0, str(helper_root))


def _palsynet_profile(args: argparse.Namespace):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from scripts import run_110d_generalization_v1 as runner
    from scripts.run_dynamic_landmark_classical import (
        group_mean_predictions, group_sample_weights,
    )

    manifest, manifest_sha = runner._read_json(args.reviewed_identity_manifest)
    ledger, ledger_sha = runner._read_json(args.review_ledger)
    registry, registry_sha = runner._read_json(args.split_registry)
    dataset, collection_rows, cache_collection_sha = (
        runner._build_cache_metadata_dataset(args.palsynet_cache_root)
    )
    audit = runner.GateAudit()
    gate = runner.validate_development_gate(
        dataset, manifest, ledger, registry,
        reviewed_manifest_sha256=manifest_sha,
        review_ledger_sha256=ledger_sha,
        split_registry_sha256=registry_sha,
        cache_source_sha256_by_recording_id={
            recording_id: str(row["source_sha256"])
            for recording_id, row in collection_rows.items()
        },
        cache_source_collection_sha256=cache_collection_sha,
        audit=audit,
    )
    runner.load_development_cache_records(
        args.palsynet_cache_root, dataset, gate, collection_rows, audit=audit
    )
    prepared = runner.prepare_development_candidates(dataset, gate, audit=audit)
    candidate = "landmark_mi_110d"
    development = gate.development_indices
    local_by_global = {int(index): local for local, index in enumerate(development)}
    original_oof = np.full(development.size, np.nan, dtype=np.float64)
    mirrored_oof = np.full(development.size, np.nan, dtype=np.float64)
    for fold in range(runner.INNER_FOLDS):
        validation_global = development[gate.inner_fold_by_index[development] == fold]
        training_global = development[gate.inner_fold_by_index[development] != fold]
        train_local = np.asarray(
            [local_by_global[int(index)] for index in training_global], dtype=np.int64
        )
        validation_local = np.asarray(
            [local_by_global[int(index)] for index in validation_global], dtype=np.int64
        )
        x_train = np.concatenate((
            prepared.original[candidate][train_local],
            prepared.mirrored[candidate][train_local],
        ))
        y_train = np.concatenate((
            dataset.labels[training_global], dataset.labels[training_global]
        ))
        train_groups = np.concatenate((
            gate.group_ids[training_global], gate.group_ids[training_global]
        ))
        scaler = StandardScaler().fit(x_train)
        model = LogisticRegression(
            C=runner.FIXED_C, penalty="l2", solver=runner.FIXED_SOLVER,
            max_iter=runner.FIXED_MAX_ITER, random_state=runner.FIXED_RANDOM_STATE,
        )
        model.fit(
            scaler.transform(x_train), y_train,
            sample_weight=group_sample_weights(train_groups),
        )
        original_oof[validation_local] = model.predict_proba(
            scaler.transform(prepared.original[candidate][validation_local])
        )[:, 1]
        mirrored_oof[validation_local] = model.predict_proba(
            scaler.transform(prepared.mirrored[candidate][validation_local])
        )[:, 1]
    if not np.isfinite(original_oof).all() or not np.isfinite(mirrored_oof).all():
        raise ValueError("PalsyNet OOF expert reconstruction is incomplete")
    labels = dataset.labels[development]
    groups = gate.group_ids[development]
    grouped_labels, grouped_ids, grouped_original = group_mean_predictions(
        labels, groups, original_oof
    )
    labels_mirror, ids_mirror, grouped_mirrored = group_mean_predictions(
        labels, groups, mirrored_oof
    )
    if not (
        np.array_equal(grouped_labels, labels_mirror)
        and np.array_equal(grouped_ids, ids_mirror)
    ):
        raise ValueError("PalsyNet original and mirror group order differs")
    components = np.stack((grouped_original, grouped_mirrored), axis=1)
    final = np.mean(components, axis=1)
    return (
        np.asarray(grouped_labels, dtype=np.int64),
        np.asarray(final, dtype=np.float64),
        np.asarray(components, dtype=np.float64),
        np.full(grouped_labels.size, 0.5, dtype=np.float64),
    )


def _neuroface_profile(args: argparse.Namespace):
    from neuroface_full_au_explore_v3 import _load as _load_au
    from neuroface_nested_marlin_gate_v4 import (
        CONFIGS, _clinical_oof, _marlin_oof,
    )
    from neuroface_phenotype_expert_quick_v3 import TASKS, _folds, _load
    from neuroface_marlin_loop_v4 import _load_marlin

    representations, labels, groups, phenotypes, _ = _load_marlin(
        args.neuroface_private_manifest, args.neuroface_marlin_root
    )
    original, mirrored, _, clinical_labels, clinical_groups, clinical_phenotypes = _load(
        args.neuroface_private_manifest, args.neuroface_dynamic_root,
        args.neuroface_au_root,
    )
    au_views, au_labels = _load_au(
        args.neuroface_private_manifest, args.neuroface_au_root, TASKS
    )
    if (
        groups != clinical_groups
        or phenotypes != clinical_phenotypes
        or not np.array_equal(labels, clinical_labels)
        or not np.array_equal(labels, au_labels)
    ):
        raise ValueError("NeuroFace expert membership differs")
    landmark = original["mean110"]
    landmark_mirror = mirrored["mean110"]
    au = au_views["robust400_selected_pool"]
    clinical = np.full(len(labels), np.nan, dtype=np.float64)
    marlin = np.full((len(CONFIGS), len(labels)), np.nan, dtype=np.float64)
    for train, held in _folds(groups, phenotypes):
        values, covered = _clinical_oof(
            landmark, landmark_mirror, au, labels, phenotypes,
            ((train, held),),
        )
        if not np.all(covered[held]):
            raise ValueError("NeuroFace clinical OOF coverage differs")
        clinical[held] = values[held]
        for candidate_index, (representation, top_k, c) in enumerate(CONFIGS):
            values, covered = _marlin_oof(
                representations[representation], labels, phenotypes,
                ((train, held),), top_k=top_k, c=c,
            )
            if not np.all(covered[held]):
                raise ValueError("NeuroFace MARLIN OOF coverage differs")
            marlin[candidate_index, held] = values[held]
    if not np.isfinite(clinical).all() or not np.isfinite(marlin).all():
        raise ValueError("NeuroFace OOF expert reconstruction is incomplete")
    final = clinical.copy()
    uncertain = np.abs(clinical - 0.5) <= 0.3
    final[uncertain] = np.median(marlin[:, uncertain], axis=0)
    components = np.column_stack((clinical, marlin.T))
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(final, dtype=np.float64),
        np.asarray(components, dtype=np.float64),
        np.full(labels.size, 0.5, dtype=np.float64),
    )


def _meei_profile(args: argparse.Namespace):
    from meei_cue_sequence_architecture_v3 import _load_raw, _matrix
    from meei_nested_sequence_confirmation_v3 import (
        CANDIDATES, ENSEMBLE_WEIGHTS, _fit_predict, _inner_folds,
    )
    from shared_landmark_capacity_v3 import LANDMARK_CHANNELS, _folds
    from src.evaluation.universal_phenotype_v3 import select_inner_global_threshold

    rows, _ = _load_raw(args.meei_cue_root)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    groups = tuple(row["group"] for row in rows)
    sources = ("meei",) * len(rows)
    matrices = {
        config["name"]: _matrix(
            rows, LANDMARK_CHANNELS, config["actions"], config["family"]
        )
        for config in CANDIDATES
    }
    components = np.full((len(labels), len(CANDIDATES)), np.nan, dtype=np.float64)
    thresholds = np.full(len(labels), np.nan, dtype=np.float64)
    for train, held in _folds(labels, sources, groups):
        inner_probability = np.full(len(train), np.nan, dtype=np.float64)
        outer_position = {int(index): position for position, index in enumerate(train)}
        for inner_train, inner_held in _inner_folds(train, labels, groups):
            inner_components = []
            for config in CANDIDATES:
                original, mirrored = matrices[config["name"]]
                probability, _ = _fit_predict(
                    original, mirrored, labels, sources,
                    inner_train, inner_held, config,
                )
                inner_components.append(probability)
            combined = sum(
                weight * probability
                for weight, probability in zip(ENSEMBLE_WEIGHTS, inner_components)
            )
            for index, probability in zip(inner_held, combined):
                inner_probability[outer_position[int(index)]] = probability
        if not np.isfinite(inner_probability).all():
            raise ValueError("MEEI inner OOF threshold reconstruction is incomplete")
        threshold = select_inner_global_threshold(
            labels[train], inner_probability
        )["threshold"]
        thresholds[held] = threshold
        for candidate_index, config in enumerate(CANDIDATES):
            original, mirrored = matrices[config["name"]]
            probability, _ = _fit_predict(
                original, mirrored, labels, sources, train, held, config
            )
            components[held, candidate_index] = probability
    if not np.isfinite(components).all() or not np.isfinite(thresholds).all():
        raise ValueError("MEEI OOF expert reconstruction is incomplete")
    model = json.loads(args.v4_model.read_text())
    artifact_threshold = model["meei"]["decision_threshold"]
    weights = np.asarray(ENSEMBLE_WEIGHTS, dtype=np.float64)
    if (
        weights.shape != (2,)
        or not np.allclose(weights, model["meei"]["probability_weights"])
        or isinstance(artifact_threshold, bool)
        or not isinstance(artifact_threshold, (int, float))
    ):
        raise ValueError("MEEI artifact and OOF ensemble contracts differ")
    validate_development_threshold_aggregation(
        float(artifact_threshold), thresholds
    )
    final = components @ weights
    return labels, np.asarray(final, dtype=np.float64), components, thresholds


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helper-root", required=True, type=Path)
    parser.add_argument("--palsynet-cache-root", required=True, type=Path)
    parser.add_argument("--reviewed-identity-manifest", required=True, type=Path)
    parser.add_argument("--review-ledger", required=True, type=Path)
    parser.add_argument("--split-registry", required=True, type=Path)
    parser.add_argument("--neuroface-private-manifest", required=True, type=Path)
    parser.add_argument("--neuroface-dynamic-root", required=True, type=Path)
    parser.add_argument("--neuroface-au-root", required=True, type=Path)
    parser.add_argument("--neuroface-marlin-root", required=True, type=Path)
    parser.add_argument("--meei-cue-root", required=True, type=Path)
    parser.add_argument("--v4-model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    verify_helper_root(args.helper_root)
    _configure_legacy_imports(args.helper_root)
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.output_dir, 0o700)
    profiles = {
        "free_asymmetry": _palsynet_profile(args),
        "scripted_multimechanism": _neuroface_profile(args),
        "cue_aligned_upper": _meei_profile(args),
    }
    payloads = {}
    payload_hashes = {}
    for profile in _PROFILE_NAMES:
        payload = build_private_profile_bytes(profile, *profiles[profile][:3],
                                              decision_threshold=profiles[profile][3])
        payloads[profile] = payload
        payload_hashes[profile] = publish_private_payload(
            args.output_dir / f"{profile}.npz", payload
        )
    boundary = {
        "schema_version": "selective_router_v5_boundary_attestation",
        "protected_palsynet_reads": 0,
        "palsynet_sealed_outer_reads": 0,
        "mayo_reads": 0,
        "profile_generator_sha256": _sha256(Path(__file__).read_bytes()),
        "helper_aggregate_sha256": _helper_aggregate_sha256(),
        "profile_payload_sha256": payload_hashes,
    }
    boundary_bytes = (
        json.dumps(boundary, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    publish_private_payload(args.output_dir / "boundary_attestation.json", boundary_bytes)
    print(json.dumps({
        "profiles": {
            profile: {
                "participants": int(profiles[profile][0].size),
                "components": int(profiles[profile][2].shape[1]),
                "sha256": payload_hashes[profile],
            }
            for profile in _PROFILE_NAMES
        },
        "helper_aggregate_sha256": boundary["helper_aggregate_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
