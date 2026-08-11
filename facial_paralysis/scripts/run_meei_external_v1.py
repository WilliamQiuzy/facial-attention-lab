#!/usr/bin/env python3
"""Run one authenticated external MEEI evaluation of the frozen 110D model."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_110d_generalization_v1 import _read_json  # noqa: E402
from src.evaluation.meei_external_pins_v1 import (  # noqa: E402
    PINNED_MEEI_AUTHORIZATION_SHA256,
)
from src.evaluation.meei_external_v1 import (  # noqa: E402
    DEFAULT_REPORT_RELATIVE,
    ExternalAudit,
    build_external_report,
    cache_artifact_inventory,
    implementation_fingerprints,
    score_authenticated_cache,
    validate_external_authorization,
    validate_external_metadata,
    validate_external_report,
    validate_frozen_artifact_for_external,
    write_private_no_overwrite_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-artifact", required=True, type=Path)
    parser.add_argument("--participant-manifest", required=True, type=Path)
    parser.add_argument("--feature-cache-root", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    output = PROJECT_ROOT / DEFAULT_REPORT_RELATIVE
    if output.exists() or output.is_symlink():
        raise FileExistsError("MEEI external report already exists; refusing repeat scoring")
    artifact, artifact_sha = _read_json(args.final_artifact)
    participant_manifest, participant_manifest_sha = _read_json(
        args.participant_manifest
    )
    cache_manifest, cache_manifest_sha = _read_json(
        args.feature_cache_root / "collection_manifest.json"
    )
    authorization, authorization_sha = _read_json(args.authorization)
    validate_frozen_artifact_for_external(artifact)
    metadata = validate_external_metadata(
        participant_manifest,
        cache_manifest,
        participant_manifest_sha256=participant_manifest_sha,
    )
    audit = ExternalAudit()
    inventory = cache_artifact_inventory(
        args.feature_cache_root, metadata.rows, audit=audit
    )
    components, implementation_sha = implementation_fingerprints()
    state = validate_external_authorization(
        authorization,
        authorization_sha256=authorization_sha,
        pinned_authorization_sha256=PINNED_MEEI_AUTHORIZATION_SHA256,
        audit=audit,
        final_artifact_sha256=artifact_sha,
        participant_manifest_sha256=participant_manifest_sha,
        cache_manifest_sha256=cache_manifest_sha,
        cache_artifact_collection_sha256=inventory.collection_sha256,
        implementation_sha256=implementation_sha,
        expected_participants=metadata.participants_total,
        expected_affected=metadata.affected,
        expected_unaffected=metadata.unaffected,
        expected_eligible_videos=metadata.eligible_videos,
    )
    labels, probabilities = score_authenticated_cache(
        inventory, metadata.rows, artifact, state=state, audit=audit
    )
    report = build_external_report(
        labels,
        probabilities,
        state=state,
        audit=audit,
        final_artifact_sha256=artifact_sha,
        participant_manifest_sha256=participant_manifest_sha,
        cache_manifest_sha256=cache_manifest_sha,
        cache_artifact_collection_sha256=inventory.collection_sha256,
        implementation_components_sha256=components,
        implementation_sha256=implementation_sha,
        participants_total=metadata.participants_total,
        eligible_videos=metadata.eligible_videos,
    )
    validate_external_report(
        report,
        labels=labels,
        probabilities=probabilities,
        state=state,
        audit=audit,
    )
    write_private_no_overwrite_json(output, report)
    print(json.dumps({
        "schema_version": "meei_external_110d_receipt_v1",
        "output_relative_path": DEFAULT_REPORT_RELATIVE,
        "report_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "external_validation_completed": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
