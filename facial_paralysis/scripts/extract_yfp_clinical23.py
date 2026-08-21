#!/usr/bin/env python3
"""Extract one static clinical23 vector per authenticated YFP anchor."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.yfp_region_manifest import (
    ManifestError,
    authenticate_eligible_manifest,
    write_manifest_once,
)

_DEFAULT_MODEL = ROOT / "data" / "mediapipe_out" / "_models" / "face_landmarker.task"
CACHE_SCHEMA = "yfp_clinical23_static_cache_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_yfp_clinical23(
    manifest_path: str | Path,
    output_root: str | Path,
    model_path: str | Path = _DEFAULT_MODEL,
) -> dict:
    """Fail before imports/output creation unless the successor is eligible."""
    manifest_path = Path(manifest_path)
    output_root = Path(output_root)
    manifest, manifest_digest = authenticate_eligible_manifest(
        manifest_path, return_digest=True)
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    model_path = Path(model_path)
    if not model_path.exists() or not model_path.is_file() or model_path.is_symlink():
        raise ManifestError("exact MediaPipe model artifact is unavailable")

    # Heavy inference imports occur only after the eligibility gate above.
    import cv2
    from src.preprocessing.action_bundle import MediaPipeFeatureExtractor

    source_root = Path(manifest["source_root"])
    model_digest = _sha256_file(model_path)
    output_root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = output_root.with_name(output_root.name + f".tmp-{uuid.uuid4().hex}")
    temporary.mkdir(mode=0o700)
    rows: list[dict] = []
    extractor = None
    try:
        extractor = MediaPipeFeatureExtractor(
            model_path=model_path,
            landmark_features="clinical23",
            capture_mirrored=None,
        )
        for row in manifest["rows"]:
            image_path = source_root / row["image"]["relative_path"]
            if image_path.is_symlink() or _sha256_file(image_path) != row["image"]["sha256"]:
                raise ManifestError("YFP source image changed after eligibility")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ManifestError("authenticated YFP image cannot be decoded")
            sequence, mask = extractor.extract_sequence([image])
            if mask.shape != (1,) or not bool(mask[0]) or sequence.shape != (1, 95):
                raise ManifestError("MediaPipe failed on an authenticated YFP anchor")
            clinical23 = np.asarray(sequence[0, -23:], dtype=np.float64)
            if clinical23.shape != (23,) or not np.isfinite(clinical23).all():
                raise ManifestError("MediaPipe clinical23 output is invalid")
            relative = Path("anchors") / f"{row['anchor_key']}.npz"
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            np.savez(
                destination,
                clinical23=clinical23,
                anchor_key=np.asarray(row["anchor_key"]),
                source_commitment=np.asarray(row["source_commitment"]),
                schema_version=np.asarray("clinical23_v2_static_single_frame"),
            )
            os.chmod(destination, 0o600)
            rows.append({
                "anchor_key": row["anchor_key"],
                "source_commitment": row["source_commitment"],
                "relative_path": relative.as_posix(),
                "sha256": _sha256_file(destination),
            })
        cache_manifest = {
            "schema_version": CACHE_SCHEMA,
            "eligible_manifest_sha256": manifest_digest,
            "mediapipe_model_sha256": model_digest,
            "feature_schema": "clinical23_v2_static_single_frame",
            "feature_dimension": 23,
            "static_only": True,
            "dynamic_tiling_allowed": False,
            "aggregate": {"anchor_count": len(rows), "extractions": len(rows)},
            "rows": rows,
        }
        write_manifest_once(cache_manifest, temporary / "manifest.json")
        temporary.rename(output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        if extractor is not None:
            extractor.close()
    return cache_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = extract_yfp_clinical23(args.manifest, args.output_root)
    print(json.dumps(result["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
