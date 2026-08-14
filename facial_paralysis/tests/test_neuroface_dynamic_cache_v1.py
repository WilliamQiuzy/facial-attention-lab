"""Synthetic contracts for resumable NeuroFace clinical23_v2 extraction."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from scripts.extract_neuroface_clinical23_v2_windows import (  # noqa: E402
    FROZEN_MEDIAPIPE_MODEL_SHA256,
    _parser,
    build_collection_manifest,
    copy_authenticated_zip_member,
    validate_existing_cache,
    validate_model_lock,
)
from src.datasets.dynamic_landmark import DYNAMIC_FEATURE_NAMES  # noqa: E402
from _testlib import Check, run_all  # noqa: E402


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _record() -> dict[str, object]:
    return {
        "recording_id": "rec_" + "1" * 64,
        "participant_id": "grp_" + "2" * 64,
        "cohort": "healthy_control",
        "binary_label": "unaffected",
        "session": "02",
        "task": "NSM_KISS",
        "video_archive_id": "healthy_control_videos",
        "video_sha256": "3" * 64,
        "video_size_bytes": 123,
        "landmark_archive_id": "healthy_control_landmarks",
        "landmark_sha256": "4" * 64,
        "annotated_frames": 9,
        "slp_scores": {
            "symmetry": 1.0, "rom": 1.0, "speed": 1.0,
            "variability": 1.0, "fatigue": 1.0, "total": 5.0,
        },
    }


def _cache(path: Path, row: dict[str, object], *, label: int = 0) -> None:
    features = np.zeros((4, 32, 95), dtype=np.float32)
    mask = np.ones((4, 32), dtype=bool)
    indices = np.stack([np.arange(start, start + 32) for start in (0, 40, 80, 120)])
    np.savez(
        path,
        features=features,
        valid_mask=mask,
        timestamps=indices.astype(np.float64) / 50.0,
        timestamp_unit=np.asarray("seconds"),
        source_frame_indices=indices.astype(np.int64),
        source_frame_count=np.asarray(152, dtype=np.int64),
        feature_schema=np.asarray("mediapipe_bs_lr_v1+clinical23_v2"),
        feature_names=np.asarray(DYNAMIC_FEATURE_NAMES),
        recording_id=np.asarray(row["recording_id"]),
        group_id=np.asarray(row["participant_id"]),
        label=np.asarray(label, dtype=np.int64),
        source_sha256=np.asarray(row["video_sha256"]),
    )


def test_zip_member_copy_consumes_authenticated_bytes(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        archive_path = root / "videos.zip"
        payload = b"synthetic-avi-bytes"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("Videos/N001_02_NSM_KISS_color.avi", payload)
        decoder = root / "decoder"
        decoder.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            copied = copy_authenticated_zip_member(
                archive,
                "Videos/N001_02_NSM_KISS_color.avi",
                _sha(payload),
                decoder,
            )
            c.eq(copied.read_bytes(), payload, "decoder receives the hashed member bytes")
            c.eq(copied.stat().st_mode & 0o777, 0o400, "temporary source is read-only")
            copied.chmod(0o600)
            copied.unlink()
            c.raises(lambda: copy_authenticated_zip_member(
                archive, "Videos/N001_02_NSM_KISS_color.avi", "0" * 64, decoder
            ), ValueError, "member substitution is rejected")
        c.eq(tuple(decoder.iterdir()), (), "failed copies leave no source bytes")


def test_model_and_existing_cache_are_exactly_bound(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        model = root / "face_landmarker.task"
        model.write_bytes(b"model")
        digest = _sha(b"model")
        c.eq(validate_model_lock(model, expected_sha256=digest), digest,
             "exact model bytes pass")
        c.raises(lambda: validate_model_lock(model, expected_sha256="0" * 64),
                 ValueError, "changed model is rejected")
        row = _record()
        cache = root / f"{row['recording_id']}.npz"
        _cache(cache, row)
        summary = validate_existing_cache(cache, row)
        c.eq(summary["coverage"], 1.0, "valid resumable cache is accepted")
        changed = dict(row)
        changed["video_sha256"] = "9" * 64
        c.raises(lambda: validate_existing_cache(cache, changed), ValueError,
                 "cache/source substitution is rejected")


def test_collection_manifest_is_complete_closed_and_no_tuning_cli(c: Check):
    row = _record()
    private_manifest = {
        "schema_version": "neuroface_external_private_manifest_v1",
        "dataset": "Toronto_NeuroFace_v1",
        "claim_unit": "participant",
        "target": "neurological_orofacial_impairment_vs_healthy_control",
        "primary_tasks": ["NSM_KISS", "NSM_OPEN", "NSM_SPREAD"],
        "counts": {"participants": 1, "videos": 1, "annotated_frames": 9,
                   "affected_participants": 0, "unaffected_participants": 1,
                   "primary_complete_participants": 1, "by_cohort": {}},
        "archives": {"healthy_control_videos": {"sha256": "5" * 64, "size_bytes": 10}},
        "slp_workbook_sha256": {"healthy_control": "6" * 64},
        "participants": [{"participant_id": row["participant_id"],
                          "cohort": "healthy_control", "binary_label": "unaffected"}],
        "records": [row],
    }
    cache_rows = [{
        "recording_id": row["recording_id"],
        "participant_id": row["participant_id"],
        "video_sha256": row["video_sha256"],
        "cache_sha256": "7" * 64,
        "coverage": 1.0,
    }]
    manifest = build_collection_manifest(
        private_manifest,
        cache_rows,
        private_manifest_sha256="8" * 64,
        model_sha256=FROZEN_MEDIAPIPE_MODEL_SHA256,
        implementation_sha256="9" * 64,
    )
    c.eq(manifest["counts"]["retained"], 1, "every manifest record is retained")
    encoded = json.dumps(manifest, sort_keys=True)
    c.true("N001" not in encoded and ".avi" not in encoded and "/Users/" not in encoded,
           "collection manifest is location and identifier free")
    options = {action.dest for action in _parser()._actions}
    for forbidden in ("threshold", "c", "solver", "calibration", "candidate"):
        c.true(forbidden not in options, f"CLI cannot tune {forbidden}")


if __name__ == "__main__":
    run_all("test_neuroface_dynamic_cache_v1", dict(globals()))
