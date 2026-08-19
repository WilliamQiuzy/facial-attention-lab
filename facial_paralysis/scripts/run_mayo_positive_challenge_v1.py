#!/usr/bin/env python3
"""Score the local Mayo positive cohort with the frozen PalsyNet 110D model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_110d_generalization_v1 import (  # noqa: E402
    GateAudit,
    _build_cache_metadata_dataset,
    _read_json,
    load_development_cache_records,
    validate_development_gate,
)
from scripts.run_architecture_search_v1 import _prepare_search_dataset  # noqa: E402
from scripts.run_mirror_invariant_110d import mirror_dynamic_features  # noqa: E402
from src.datasets.dynamic_landmark import (  # noqa: E402
    DYNAMIC_FEATURE_NAMES,
    DYNAMIC_FEATURE_SCHEMA,
    DYNAMIC_FEATURE_SHAPE,
)
from src.evaluation.mayo_positive_challenge_v1 import (  # noqa: E402
    build_aggregate_challenge_report,
    fit_frozen_110d_champion,
    positive_cohort_summary,
    predict_mirror_mean,
)
from src.preprocessing.generalization_110d import (  # noqa: E402
    LANDMARK_MI_110D,
    candidate_feature_vector,
)


DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "outputs" / "dynamic_landmark" / "benchmarks" / "external"
    / "mayo-positive-challenge-v1" / "report.json"
)
_IMPLEMENTATION_FILES = (
    Path(__file__).resolve(),
    PROJECT_ROOT / "scripts" / "build_mayo_positive_challenge_v1.py",
    PROJECT_ROOT / "src" / "evaluation" / "mayo_positive_challenge_v1.py",
    PROJECT_ROOT / "src" / "preprocessing" / "generalization_110d.py",
    PROJECT_ROOT / "scripts" / "run_mirror_invariant_110d.py",
    PROJECT_ROOT / "scripts" / "run_110d_generalization_v1.py",
)
_CACHE_FIELDS = {
    "features", "valid_mask", "timestamps", "timestamp_unit",
    "source_frame_indices", "source_frame_count", "feature_schema",
    "feature_names", "recording_id", "group_id", "label", "source_sha256",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_sha256() -> str:
    components = {
        str(path.relative_to(PROJECT_ROOT)): _sha256(path)
        for path in _IMPLEMENTATION_FILES
    }
    encoded = json.dumps(components, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_mayo_features(cache_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, dict[str, int]]:
    manifest, manifest_sha = _read_json(cache_root / "collection_manifest.json")
    if (
        manifest.get("schema_version") != "mayo_positive_clinical23_v2_windows_v1"
        or manifest.get("claim_unit") != "deduplicated_video_content"
        or manifest.get("eligibility", {}).get("positive_confidence_challenge") is not True
        or manifest.get("eligibility", {}).get("model_selection") is not False
    ):
        raise ValueError("Mayo cache is not the frozen positive-challenge schema")
    records = manifest.get("records")
    inventory = manifest.get("inventory")
    if not isinstance(records, list) or not isinstance(inventory, dict) or not records:
        raise ValueError("Mayo challenge manifest records/inventory are invalid")
    expected_files = {cache_root / f"{row['recording_id']}.npz" for row in records}
    if set(cache_root.glob("*.npz")) != expected_files:
        raise ValueError("Mayo cache NPZ set differs from its manifest")
    original_rows: list[np.ndarray] = []
    mirrored_rows: list[np.ndarray] = []
    coverages: list[float] = []
    for row in records:
        if not isinstance(row, dict) or row.get("label") != "affected":
            raise ValueError("Mayo challenge record metadata is invalid")
        with np.load(cache_root / f"{row['recording_id']}.npz", allow_pickle=False) as saved:
            if set(saved.files) != _CACHE_FIELDS:
                raise ValueError("Mayo challenge NPZ fields differ from the frozen schema")
            values = {name: np.asarray(saved[name]) for name in _CACHE_FIELDS}
        features = values["features"]
        mask = values["valid_mask"]
        timestamps = values["timestamps"]
        indices = values["source_frame_indices"]
        scalar = lambda name: np.asarray(values[name]).item()
        if (
            scalar("recording_id") != row.get("recording_id")
            or scalar("group_id") != row.get("group_id")
            or scalar("source_sha256") != row.get("source_sha256")
            or scalar("label") != 1
            or scalar("timestamp_unit") != "seconds"
            or scalar("feature_schema") != DYNAMIC_FEATURE_SCHEMA
            or tuple(values["feature_names"].tolist()) != DYNAMIC_FEATURE_NAMES
        ):
            raise ValueError("Mayo cache metadata differs from its manifest")
        if (
            features.shape != DYNAMIC_FEATURE_SHAPE
            or features.dtype != np.float32
            or mask.shape != DYNAMIC_FEATURE_SHAPE[:2]
            or mask.dtype != bool
            or timestamps.shape != DYNAMIC_FEATURE_SHAPE[:2]
            or indices.shape != DYNAMIC_FEATURE_SHAPE[:2]
            or indices.dtype.kind not in {"i", "u"}
            or not np.isfinite(features[mask]).all()
            or np.any(features[~mask] != 0)
            or not np.isfinite(timestamps).all()
            or not np.all(np.diff(indices, axis=1) == 1)
            or not np.all(indices[1:, 0] - indices[:-1, 0] >= 32)
        ):
            raise ValueError("Mayo challenge arrays violate the face-anchored contract")
        coverage = float(mask.mean())
        fps = float(row.get("fps", 0.0))
        if (
            coverage < 0.75
            or not np.isclose(coverage, float(row.get("coverage", -1.0)), atol=0, rtol=0)
            or not np.isfinite(fps) or fps <= 0
            or not np.array_equal(timestamps, indices.astype(np.float64) / fps)
            or int(scalar("source_frame_count")) != int(row.get("source_frame_count", -1))
            or np.any(indices < 0)
            or np.any(indices >= int(scalar("source_frame_count")))
        ):
            raise ValueError("Mayo challenge timing/coverage differs from its manifest")
        temporal = (
            mask,
            timestamps,
            indices,
        )
        original_rows.append(candidate_feature_vector(
            LANDMARK_MI_110D, features, *temporal
        ))
        mirrored_rows.append(candidate_feature_vector(
            LANDMARK_MI_110D, mirror_dynamic_features(features), *temporal
        ))
        coverages.append(coverage)
    return (
        np.stack(original_rows).astype(np.float64, copy=False),
        np.stack(mirrored_rows).astype(np.float64, copy=False),
        np.asarray(coverages, dtype=np.float64),
        manifest_sha,
        {
            "source_files": int(inventory["source_video_files"]),
            "unique_contents": int(inventory["unique_video_contents"]),
            "exact_duplicate_files": int(inventory["exact_duplicate_files"]),
            "excluded_records": int(inventory["quality_excluded_unique_contents"]),
        },
    )


def _write_no_overwrite(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite Mayo challenge report {path}")
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd, temporary_name = tempfile.mkstemp(prefix=".mayo-challenge.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def run_challenge(args) -> dict[str, object]:
    manifest, manifest_sha = _read_json(args.reviewed_identity_manifest)
    ledger, ledger_sha = _read_json(args.review_ledger)
    registry, registry_sha = _read_json(args.split_registry)
    dataset, collection_rows, collection_sha = _build_cache_metadata_dataset(
        args.palsynet_cache_root
    )
    audit = GateAudit()
    gate = validate_development_gate(
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
        cache_source_collection_sha256=collection_sha,
        audit=audit,
    )
    load_development_cache_records(
        args.palsynet_cache_root,
        dataset,
        gate,
        collection_rows,
        audit=audit,
    )
    prepared = _prepare_search_dataset(dataset, gate)
    development = prepared.development_indices
    champion = fit_frozen_110d_champion(
        prepared.summary_features[development],
        prepared.mirrored_summary_features[development],
        prepared.labels[development],
        prepared.group_ids[development],
    )
    original, mirrored, coverage, mayo_manifest_sha, counts = _load_mayo_features(
        args.mayo_cache_root
    )
    probabilities = predict_mirror_mean(champion, original, mirrored)
    summary = positive_cohort_summary(probabilities, coverage)
    report = build_aggregate_challenge_report(
        summary,
        **counts,
        provenance={
            "palsynet_source_collection_sha256": gate.source_collection_sha256,
            "palsynet_reviewed_manifest_sha256": gate.reviewed_manifest_sha256,
            "palsynet_review_ledger_sha256": gate.review_ledger_sha256,
            "palsynet_split_registry_sha256": gate.split_registry_sha256,
            "mayo_cache_manifest_sha256": mayo_manifest_sha,
            "implementation_sha256": _implementation_sha256(),
        },
    )
    if audit.protected_cache_records_loaded != 0:
        raise AssertionError("Mayo challenge opened protected PalsyNet cache records")
    _write_no_overwrite(args.output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--palsynet-cache-root", required=True, type=Path)
    parser.add_argument("--reviewed-identity-manifest", required=True, type=Path)
    parser.add_argument("--review-ledger", required=True, type=Path)
    parser.add_argument("--split-registry", required=True, type=Path)
    parser.add_argument("--mayo-cache-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    return parser


def main(argv: list[str] | None = None) -> None:
    report = run_challenge(_parser().parse_args(argv))
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
