#!/usr/bin/env python3
"""Finalize one exhaustive, label-blinded PalsyNet identity review."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.palsynet_identity_review import (  # noqa: E402
    finalize_identity_review,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-manifest", required=True, type=Path)
    parser.add_argument("--contact-inventory", required=True, type=Path)
    parser.add_argument("--review-ledger", required=True, type=Path)
    parser.add_argument("--reviewer-evidence", required=True, type=Path)
    parser.add_argument("--cross-label-adjudication", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    reviewed = finalize_identity_review(
        args.generated_manifest,
        args.contact_inventory,
        args.review_ledger,
        args.reviewer_evidence,
        args.cross_label_adjudication,
        args.output,
    )
    print(json.dumps({
        "status": reviewed["identity_review"]["status"],
        "total_recordings": reviewed["counts"]["total_recordings"],
        "reviewed_groups": reviewed["counts"]["reviewed_groups"],
        "eligible_recordings": reviewed["counts"]["eligible_recordings"],
        "eligible_groups": reviewed["counts"]["eligible_groups"],
    }, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
