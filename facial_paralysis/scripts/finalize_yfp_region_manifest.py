#!/usr/bin/env python3
"""Create one separately authenticated eligible YFP manifest successor."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.yfp_region_manifest import finalize_eligible_manifest, write_manifest_once


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--license-artifact", type=Path, required=True)
    parser.add_argument("--reviewed-subject-map", type=Path, required=True)
    parser.add_argument("--eligibility-authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = finalize_eligible_manifest(
        args.audit_manifest,
        args.license_artifact,
        args.reviewed_subject_map,
        args.eligibility_authorization,
    )
    write_manifest_once(manifest, args.output)
    print(json.dumps({
        "schema_version": manifest["schema_version"],
        "training_eligible": manifest["training_eligible"],
        "aggregate": manifest["aggregate"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
