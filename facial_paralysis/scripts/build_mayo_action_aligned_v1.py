#!/usr/bin/env python3
"""Build seven-action windows for the frozen 47-record Mayo challenge cohort."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_mayo_positive_challenge_v1 import (  # noqa: E402
    configure_capture_orientation,
)
from scripts.build_palsynet_action_aligned_v1 import (  # noqa: E402
    extract_action_aligned_source,
    load_action_aligned_cache,
    write_action_aligned_cache,
)
from scripts.build_palsynet_v2_windows import (  # noqa: E402
    IdentityBinding,
    SourceVideo,
    managed_extractor,
)
from src.evaluation.mayo_positive_challenge_v1 import (  # noqa: E402
    inventory_content_deduplicated_videos,
)
from src.preprocessing.action_bundle import MediaPipeFeatureExtractor  # noqa: E402


SCHEMA = "mayo_action_aligned_clinical23_v1"


def _extract_chunk(
    sources: tuple[SourceVideo, ...],
    model_path: Path,
    staging: Path,
) -> list[tuple[str, float]]:
    results: list[tuple[str, float]] = []
    with managed_extractor(
        MediaPipeFeatureExtractor, model_path=model_path
    ) as extractor:
        for source in sources:
            result = extract_action_aligned_source(
                source, extractor,
                capture_configurator=configure_capture_orientation,
                minimum_window_coverage=0.75,
            )
            write_action_aligned_cache(
                staging / f"{source.binding.recording_id}.npz", result
            )
            results.append((source.binding.recording_id, result.coverage))
    return results


def _read_manifest(path: Path) -> tuple[dict[str, object], bytes]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Mayo baseline manifest must be an object")
    return payload, raw


def retained_sources(
    data_root: Path,
    baseline_manifest: Path,
) -> tuple[SourceVideo, ...]:
    """Join only the frozen retained cohort to current local bytes by SHA-256."""
    manifest, _raw = _read_manifest(baseline_manifest)
    if (
        manifest.get("schema_version") != "mayo_positive_clinical23_v2_windows_v1"
        or manifest.get("claim_unit") != "deduplicated_video_content"
        or manifest.get("eligibility", {}).get("model_selection") is not False
        or manifest.get("inventory", {}).get("retained_unique_contents") != 47
    ):
        raise ValueError("Mayo baseline challenge contract drifted")
    rows = manifest.get("records")
    if not isinstance(rows, list) or len(rows) != 47:
        raise ValueError("Mayo baseline challenge must contain 47 retained records")
    inventory = inventory_content_deduplicated_videos(data_root)
    path_by_hash = {record.source_sha256: record.path for record in inventory.records}
    sources: list[SourceVideo] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("label") != "affected":
            raise ValueError("Mayo retained row is invalid")
        digest = str(row.get("source_sha256", ""))
        path = path_by_hash.get(digest)
        if path is None or digest in seen:
            raise ValueError("frozen Mayo retained bytes are missing or duplicated")
        seen.add(digest)
        binding = IdentityBinding(
            source_sha256=digest,
            recording_id=str(row.get("recording_id", "")),
            group_id=str(row.get("group_id", "")),
            label="affected",
            identity_status="positive_cohort_assumption_only",
            claim_unit="deduplicated_video_content",
        )
        sources.append(SourceVideo(path=path, source_sha256=digest, binding=binding))
    return tuple(sorted(sources, key=lambda source: source.binding.recording_id))


def run(args) -> dict[str, object]:
    sources = retained_sources(args.data_root, args.baseline_manifest)
    output = args.output_root.expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if output.exists() or output.is_symlink():
        raise FileExistsError("refusing to overwrite Mayo action cache")
    staging = output.parent / f".{output.name}.partial"
    staging.mkdir(parents=False, exist_ok=True, mode=0o700)
    os.chmod(staging, 0o700)
    source_by_id = {source.binding.recording_id: source for source in sources}
    coverages: dict[str, float] = {}
    for path in sorted(staging.glob("*.npz")):
        cached = load_action_aligned_cache(path)
        source = source_by_id.get(cached.binding.recording_id)
        if (
            source is None
            or path.name != f"{cached.binding.recording_id}.npz"
            or cached.binding.source_sha256 != source.source_sha256
            or cached.binding.group_id != source.binding.group_id
        ):
            raise ValueError("partial Mayo cache provenance drifted")
        coverages[cached.binding.recording_id] = cached.coverage
    unknown_files = {
        path.name for path in staging.iterdir()
        if path.name != "collection_manifest.json" and path.suffix != ".npz"
    }
    if unknown_files or (staging / "collection_manifest.json").exists():
        raise ValueError("partial Mayo directory contains unexpected files")
    remaining = tuple(
        source for source in sources
        if source.binding.recording_id not in coverages
    )
    workers = min(args.workers, len(remaining)) if remaining else 0
    if workers:
        chunks = tuple(
            tuple(remaining[index::workers]) for index in range(workers)
        )
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=context
        ) as executor:
            futures = [
                executor.submit(
                    _extract_chunk, chunk, args.mediapipe_model, staging
                )
                for chunk in chunks if chunk
            ]
            for future in concurrent.futures.as_completed(futures):
                for recording_id, coverage in future.result():
                    coverages[recording_id] = coverage
                    print(json.dumps({
                        "completed": len(coverages), "retained": len(sources),
                        "coverage": coverage,
                    }), flush=True)
    if set(coverages) != set(source_by_id):
        raise ValueError("Mayo action cache generation is incomplete")
    ordered_coverages = [coverages[source.binding.recording_id] for source in sources]
    manifest = {
        "schema_version": SCHEMA,
        "claim_unit": "deduplicated_video_content",
        "cohort_assumption": "all_affected_not_independently_verified",
        "eligibility": {
            "model_selection": False,
            "external_accuracy": False,
            "positive_confidence_challenge": True,
        },
        "records": 47,
        "minimum_coverage": min(ordered_coverages),
        "mean_coverage": sum(ordered_coverages) / len(ordered_coverages),
        "protocol": {
            "proposal_rate_hz": 6.0,
            "action_slots": 7,
            "samples_per_slot": 32,
            "sample_grid_hz": 30.0,
            "selection_uses_labels_or_classifier_scores": False,
        },
    }
    path = staging / "collection_manifest.json"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o600)
    expected = {"collection_manifest.json"} | {
        f"{source.binding.recording_id}.npz" for source in sources
    }
    if {path.name for path in staging.iterdir()} != expected:
        raise ValueError("completed Mayo action cache file set drifted")
    os.rename(staging, output)
    return manifest


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--baseline-manifest", required=True, type=Path)
    parser.add_argument("--mediapipe-model", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=2)
    return parser


def main():
    print(json.dumps(run(_parser().parse_args()), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
