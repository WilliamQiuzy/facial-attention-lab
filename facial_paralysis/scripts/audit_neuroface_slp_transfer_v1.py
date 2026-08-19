#!/usr/bin/env python3
"""Audit frozen NeuroFace 110D scores against released two-rater SLP severity."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_neuroface_external_v1 import _json, _read_same_descriptor  # noqa: E402
from src.evaluation.meei_external_v1 import (  # noqa: E402
    cache_artifact_inventory,
    write_private_no_overwrite_json,
)
from src.evaluation.neuroface_external_pins_v1 import (  # noqa: E402
    PINNED_NEUROFACE_AUTHORIZATION_SHA256,
)
from src.evaluation.neuroface_external_v1 import (  # noqa: E402
    ExternalAudit,
    implementation_fingerprints,
    retained_private_records,
    score_authenticated_cache,
    validate_external_authorization,
)
from src.evaluation.neuroface_slp_transfer_v1 import (  # noqa: E402
    build_slp_transfer_report,
)


FROZEN_EXTERNAL_REPORT_SHA256 = (
    "beda5bee5ed3736a90245e98c165777198aee6d6be2bbcd8a13f0fb2b1a11984"
)
OUTPUT_RELATIVE = "outputs/neuroface_external_v1/slp_transfer.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-artifact", type=Path, required=True)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--feature-cache-root", type=Path, required=True)
    parser.add_argument("--preanalysis-registration", type=Path, required=True)
    parser.add_argument("--dependency-lock", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--external-report", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = PROJECT_ROOT / OUTPUT_RELATIVE
    if output.exists() or output.is_symlink():
        raise FileExistsError("SLP transfer report already exists")
    artifact, artifact_sha = _json(args.final_artifact)
    private, private_sha = _json(args.private_manifest)
    cache, cache_sha = _json(args.feature_cache_root / "collection_manifest.json")
    _, registration_sha = _json(args.preanalysis_registration)
    authorization, authorization_sha = _json(args.authorization)
    external_bytes, external_sha = _read_same_descriptor(args.external_report)
    _, dependency_sha = _read_same_descriptor(args.dependency_lock)
    if external_sha != FROZEN_EXTERNAL_REPORT_SHA256:
        raise ValueError("external transfer report differs from the frozen result")
    external = json.loads(external_bytes)
    if external.get("schema_version") != "neuroface_external_110d_v1":
        raise ValueError("external transfer report schema is invalid")
    retained, flow = retained_private_records(private["records"], cache["records"])
    if flow != {"source_records": 261, "retained": 231, "excluded": 30}:
        raise ValueError("NeuroFace QC flow differs from the frozen external run")
    cache_rows = [row for row in cache["records"] if row["status"] == "retained"]
    audit = ExternalAudit()
    inventory = cache_artifact_inventory(args.feature_cache_root, cache_rows, audit=audit)
    _, implementation_sha = implementation_fingerprints()
    state = validate_external_authorization(
        authorization,
        authorization_sha256=authorization_sha,
        pinned_authorization_sha256=PINNED_NEUROFACE_AUTHORIZATION_SHA256,
        audit=audit,
        preanalysis_registration_sha256=registration_sha,
        final_artifact_sha256=artifact_sha,
        private_manifest_sha256=private_sha,
        cache_manifest_sha256=cache_sha,
        cache_artifact_collection_sha256=inventory.collection_sha256,
        implementation_sha256=implementation_sha,
        dependency_lock_sha256=dependency_sha,
        expected_participants=36,
        expected_affected=25,
        expected_unaffected=11,
        expected_videos=231,
    )
    scored = score_authenticated_cache(
        inventory, retained, artifact, state=state, audit=audit
    )
    report = build_slp_transfer_report(
        scored, external_report_sha256=external_sha
    )
    report["scoring_audit"] = audit.as_dict()
    write_private_no_overwrite_json(output, report)
    print(json.dumps({
        "schema_version": "neuroface_slp_transfer_receipt_v1",
        "report_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "output_relative_path": OUTPUT_RELATIVE,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
