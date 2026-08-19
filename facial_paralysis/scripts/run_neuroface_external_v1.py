#!/usr/bin/env python3
"""Run one authenticated NeuroFace evaluation of the frozen 110D model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.meei_external_v1 import (  # noqa: E402
    cache_artifact_inventory,
    validate_frozen_artifact_for_external,
    write_private_no_overwrite_json,
)
from src.evaluation.neuroface_external_pins_v1 import (  # noqa: E402
    PINNED_NEUROFACE_AUTHORIZATION_SHA256,
)
from src.evaluation.neuroface_external_v1 import (  # noqa: E402
    DEFAULT_REPORT_RELATIVE,
    ExternalAudit,
    aggregate_participant_scores,
    build_external_report,
    implementation_fingerprints,
    retained_private_records,
    score_authenticated_cache,
    validate_external_authorization,
    validate_external_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-artifact", type=Path, required=True)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--feature-cache-root", type=Path, required=True)
    parser.add_argument("--preanalysis-registration", type=Path, required=True)
    parser.add_argument("--dependency-lock", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    return parser


def _read_same_descriptor(path: Path) -> tuple[bytes, str]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("authenticated input must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise ValueError("authenticated input changed during read")
        payload = b"".join(chunks)
        return payload, hashlib.sha256(payload).hexdigest()
    finally:
        os.close(descriptor)


def _json(path: Path) -> tuple[dict[str, object], str]:
    payload, digest = _read_same_descriptor(path)
    value = json.loads(payload, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("authenticated JSON must contain an object")
    return value, digest


def _unique_object(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("authenticated JSON contains a duplicate key")
        output[key] = value
    return output


def main() -> int:
    args = _parser().parse_args()
    output = PROJECT_ROOT / DEFAULT_REPORT_RELATIVE
    if output.exists() or output.is_symlink():
        raise FileExistsError("NeuroFace report already exists; refusing repeat scoring")
    artifact, artifact_sha = _json(args.final_artifact)
    private, private_sha = _json(args.private_manifest)
    cache_manifest, cache_manifest_sha = _json(args.feature_cache_root / "collection_manifest.json")
    registration, registration_sha = _json(args.preanalysis_registration)
    authorization, authorization_sha = _json(args.authorization)
    _, dependency_sha = _read_same_descriptor(args.dependency_lock)
    validate_frozen_artifact_for_external(artifact)
    records = private.get("records")
    participants = private.get("participants")
    if (
        private.get("schema_version") != "neuroface_external_private_manifest_v1"
        or not isinstance(records, list) or not isinstance(participants, list)
        or len(records) != 261 or len(participants) != 36
    ):
        raise ValueError("private NeuroFace inventory is not the frozen full cohort")
    cache_rows = cache_manifest.get("records")
    if (
        cache_manifest.get("schema_version") != "neuroface_clinical23_v2_windows_v1"
        or not isinstance(cache_rows, list) or len(cache_rows) != len(records)
    ):
        raise ValueError("NeuroFace cache manifest is incomplete")
    retained_records, flow = retained_private_records(records, cache_rows)
    if (
        flow != {"source_records": 261, "retained": 231, "excluded": 30}
        or cache_manifest.get("counts") != {
            "source_records": 261,
            "retained": 231,
            "excluded": 30,
            "participants": 36,
            "primary_complete_participants": 36,
        }
    ):
        raise ValueError("NeuroFace label-blind QC flow differs from the frozen run")
    if registration.get("schema_version") != "neuroface_preanalysis_registration_v1":
        raise ValueError("preanalysis registration schema is invalid")
    components, implementation_sha = implementation_fingerprints()
    audit = ExternalAudit()
    retained_cache_rows = [row for row in cache_rows if row.get("status") == "retained"]
    inventory = cache_artifact_inventory(
        args.feature_cache_root, retained_cache_rows, audit=audit
    )
    state = validate_external_authorization(
        authorization,
        authorization_sha256=authorization_sha,
        pinned_authorization_sha256=PINNED_NEUROFACE_AUTHORIZATION_SHA256,
        audit=audit,
        preanalysis_registration_sha256=registration_sha,
        final_artifact_sha256=artifact_sha,
        private_manifest_sha256=private_sha,
        cache_manifest_sha256=cache_manifest_sha,
        cache_artifact_collection_sha256=inventory.collection_sha256,
        implementation_sha256=implementation_sha,
        dependency_lock_sha256=dependency_sha,
        expected_participants=36,
        expected_affected=25,
        expected_unaffected=11,
        expected_videos=231,
    )
    video_rows = score_authenticated_cache(
        inventory, retained_records, artifact, state=state, audit=audit
    )
    aggregate = aggregate_participant_scores(video_rows)
    audit.participant_aggregations = int(aggregate.labels.size)
    report = build_external_report(
        aggregate,
        state=state,
        audit=audit,
        provenance={
            "implementation_components_sha256": components,
            "mediapipe_model_sha256": cache_manifest["provenance"]["mediapipe_model_sha256"],
        },
    )
    validate_external_report(report, aggregate=aggregate, state=state, audit=audit)
    write_private_no_overwrite_json(output, report)
    print(json.dumps({
        "schema_version": "neuroface_external_110d_receipt_v1",
        "report_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "output_relative_path": DEFAULT_REPORT_RELATIVE,
        "external_transfer_completed": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
