#!/usr/bin/env python3
"""Freeze the final PalsyNet 110D artifact after the sealed outer test."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_110d_generalization_v1 import (  # noqa: E402
    GateAudit,
    _build_cache_metadata_dataset,
    _load_one_dynamic_record,
    _read_json,
    validate_development_gate,
)
from src.evaluation.outer_release_110d_v1 import (  # noqa: E402
    FinalArtifactAudit,
    OuterReleaseAudit,
    authorize_final_artifact,
    freeze_final_artifact,
    load_authorized_cache_records,
    prepare_locked_views,
    validate_outer_authorization,
    write_private_no_overwrite_json,
)
from src.evaluation.outer_release_pins_v1 import (  # noqa: E402
    PINNED_OUTER_AUTHORIZATION_SHA256,
    PINNED_PROTECTED_OUTER_REPORT_SHA256,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--palsynet-cache-root", required=True, type=Path)
    parser.add_argument("--reviewed-identity-manifest", required=True, type=Path)
    parser.add_argument("--review-ledger", required=True, type=Path)
    parser.add_argument("--split-registry", required=True, type=Path)
    parser.add_argument("--locked-development-report", required=True, type=Path)
    parser.add_argument("--sealed-outer-report", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    dataset, collection_rows, cache_source_sha = _build_cache_metadata_dataset(
        args.palsynet_cache_root
    )
    manifest, manifest_sha = _read_json(args.reviewed_identity_manifest)
    ledger, ledger_sha = _read_json(args.review_ledger)
    registry, registry_sha = _read_json(args.split_registry)
    development_report, development_sha = _read_json(
        args.locked_development_report
    )
    protected_report, protected_sha = _read_json(args.sealed_outer_report)
    authorization, authorization_sha = _read_json(args.authorization)
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
        cache_source_collection_sha256=cache_source_sha,
        audit=GateAudit(),
    )
    release_audit = OuterReleaseAudit()
    state = validate_outer_authorization(
        authorization,
        authorization_sha256=authorization_sha,
        pinned_authorization_sha256=PINNED_OUTER_AUTHORIZATION_SHA256,
        gate=gate,
        development_report=development_report,
        development_report_sha256=development_sha,
        audit=release_audit,
    )
    artifact_audit = FinalArtifactAudit()
    sealed = authorize_final_artifact(
        protected_report,
        protected_report_sha256=protected_sha,
        pinned_protected_report_sha256=PINNED_PROTECTED_OUTER_REPORT_SHA256,
        gate=gate,
        state=state,
        audit=artifact_audit,
    )
    load_authorized_cache_records(
        args.palsynet_cache_root,
        dataset,
        gate,
        collection_rows,
        state=state,
        audit=release_audit,
        record_loader=_load_one_dynamic_record,
    )
    views = prepare_locked_views(
        dataset, gate, state=state, audit=release_audit
    )
    result = freeze_final_artifact(
        dataset,
        gate,
        views,
        state=state,
        sealed_outer=sealed,
        audit=artifact_audit,
    )
    write_private_no_overwrite_json(args.output, result.artifact)
    print(json.dumps({
        "schema_version": "110d_final_artifact_receipt_v1",
        "artifact_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "frozen": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
