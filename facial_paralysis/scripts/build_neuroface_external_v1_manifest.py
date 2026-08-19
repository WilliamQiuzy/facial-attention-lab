#!/usr/bin/env python3
"""Build the deidentified, no-overwrite Toronto NeuroFace private manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.neuroface_external_v1 import (  # noqa: E402
    audit_neuroface_sources,
    build_private_manifest,
    real_source_configuration,
)


def write_private_no_overwrite(path: Path, payload: dict[str, object]) -> str:
    output = Path(path).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("refusing to overwrite the NeuroFace private manifest")
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    with output.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(output, 0o600)
    return hashlib.sha256(encoded).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    bindings, sources, expected = real_source_configuration(args.data_root)
    inventory = audit_neuroface_sources(bindings, sources, expected=expected)
    manifest = build_private_manifest(inventory)
    digest = write_private_no_overwrite(args.output, manifest)
    print(json.dumps({
        "schema_version": "neuroface_external_manifest_receipt_v1",
        "participants": manifest["counts"]["participants"],
        "videos": manifest["counts"]["videos"],
        "annotated_frames": manifest["counts"]["annotated_frames"],
        "manifest_sha256": digest,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
