#!/usr/bin/env python3
"""Build the immutable, training-ineligible YFP regional audit manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.yfp_region_manifest import build_audit_manifest, write_manifest_once


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yfp-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_audit_manifest(args.yfp_root)
    write_manifest_once(manifest, args.output)
    print(json.dumps({
        "schema_version": manifest["schema_version"],
        "training_eligible": manifest["training_eligible"],
        "aggregate": manifest["aggregate"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
