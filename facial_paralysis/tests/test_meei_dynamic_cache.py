"""Fail-closed MEEI participant-manifest and dynamic-cache contracts."""
from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from scripts.build_meei_participant_manifest import (  # noqa: E402
    ExpectedInventory,
    ManifestAudit,
    _nonempty_workbook_rows,
    _parser as manifest_parser,
    build_participant_manifest,
    validate_participant_manifest,
    write_private_no_overwrite_json,
)
from scripts.extract_meei_clinical23_v2_windows import (  # noqa: E402
    _parser as extractor_parser,
    copy_authenticated_source,
    enumerate_video_sources,
    validate_palsynet_extractor_lock,
)
from _testlib import Check, run_all  # noqa: E402


def _participant(root: Path, relative: str, stem: str, *, video: bytes) -> None:
    directory = root / relative
    directory.mkdir(parents=True)
    (directory / f"{stem}.mp4").write_bytes(video)
    for index in range(1, 9):
        (directory / f"{stem}_{index}.jpg").write_bytes(
            f"{stem}-photo-{index}".encode("ascii")
        )


def _fixture(root: Path) -> tuple[ExpectedInventory, set[str]]:
    _participant(root, "Normals/Normal1", "Normal1", video=b"normal-video")
    _participant(
        root,
        "Flaccid/MildFlaccid/MildFlaccid1",
        "MildFlaccid1",
        video=b"flaccid-video",
    )
    _participant(
        root,
        "Synkinetic/Severe Synkinetic/Synkinetic_Severe1",
        "Synkinetic_Severe1",
        video=b"synkinetic-video",
    )
    expected = ExpectedInventory(
        participants=3,
        normal_participants=1,
        flaccid_participants=1,
        synkinetic_participants=1,
        videos=3,
        photos=24,
        supporting_images=0,
    )
    return expected, {"normal1", "mildflaccid1"}


def test_manifest_hashes_before_labels_and_is_identifier_free(c: Check):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        expected, metadata_keys = _fixture(root)
        audit = ManifestAudit()
        manifest = build_participant_manifest(
            root,
            metadata_participant_keys=metadata_keys,
            metadata_xlsx_sha256="a" * 64,
            paper_pdf_sha256="b" * 64,
            expected=expected,
            audit=audit,
        )
        validate_participant_manifest(manifest, expected=expected)
        c.eq(audit.assets_hashed, 27, "every media asset is hashed")
        c.eq(audit.dynamic_eligibility_assigned, 27,
             "media-only eligibility is assigned before label join")
        c.eq(audit.labels_joined, 3, "one label is joined per participant")
        c.eq(audit.metadata_rows_joined, 2, "available metadata joins exactly")
        c.true(
            audit.last_hash_or_eligibility_event < audit.first_label_event,
            "all hashing and dynamic eligibility precede every label join",
        )
        c.eq(manifest["counts"], {
            "participants": 3,
            "normal_participants": 1,
            "facial_palsy_participants": 2,
            "flaccid_participants": 1,
            "synkinetic_participants": 1,
            "media_assets": 27,
            "videos": 3,
            "photos": 24,
            "supporting_images": 0,
            "supporting_files": 0,
            "dynamic_binary_eligible_videos": 3,
            "metadata_rows_present": 2,
            "metadata_rows_missing": 1,
        }, "manifest counts reconcile")
        rows = manifest["media"]
        c.eq(sum(row["dynamic_binary_eligible"] for row in rows), 3,
             "only the three videos are dynamic eligible")
        c.true(all(
            row["dynamic_binary_eligible"] == (row["media_type"] == "video")
            for row in rows
        ), "dynamic eligibility depends only on media type")
        encoded = json.dumps(manifest, sort_keys=True)
        c.true("Normal1" not in encoded and "MildFlaccid1" not in encoded,
               "public participant keys and source stems are absent")
        c.true(".mp4" not in encoded and "/" not in encoded,
               "raw filenames and paths are absent")
        c.true("Age" not in encoded and "Cause" not in encoded,
               "demographic row values are not copied into the manifest")


def test_manifest_rejects_ambiguous_or_incomplete_participants(c: Check):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        expected, metadata_keys = _fixture(root)
        duplicate = root / "Synkinetic/Severe Synkinetic/Synkinetic_Severe1/Synkinetic_Severe1.mp4"
        duplicate.write_bytes(b"normal-video")
        c.raises(
            lambda: build_participant_manifest(
                root,
                metadata_participant_keys=metadata_keys,
                metadata_xlsx_sha256="a" * 64,
                paper_pdf_sha256="b" * 64,
                expected=expected,
                audit=ManifestAudit(),
            ),
            ValueError,
            "duplicate video content across participants is ambiguous",
        )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        expected, metadata_keys = _fixture(root)
        (root / "Normals/Normal1/Normal1.mp4").unlink()
        c.raises(
            lambda: build_participant_manifest(
                root,
                metadata_participant_keys=metadata_keys,
                metadata_xlsx_sha256="a" * 64,
                paper_pdf_sha256="b" * 64,
                expected=expected,
                audit=ManifestAudit(),
            ),
            ValueError,
            "every participant needs exactly one video",
        )

    if hasattr(os, "symlink"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected, metadata_keys = _fixture(root)
            target = root / "Normals/Normal1/Normal1_1.jpg"
            target.unlink()
            os.symlink(root / "Normals/Normal1/Normal1_2.jpg", target)
            c.raises(
                lambda: build_participant_manifest(
                    root,
                    metadata_participant_keys=metadata_keys,
                    metadata_xlsx_sha256="a" * 64,
                    paper_pdf_sha256="b" * 64,
                    expected=expected,
                    audit=ManifestAudit(),
                ),
                ValueError,
                "symlinked media is rejected",
            )


def test_manifest_schema_tamper_cli_and_no_overwrite(c: Check):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        expected, metadata_keys = _fixture(root)
        manifest = build_participant_manifest(
            root,
            metadata_participant_keys=metadata_keys,
            metadata_xlsx_sha256="a" * 64,
            paper_pdf_sha256="b" * 64,
            expected=expected,
            audit=ManifestAudit(),
        )
        tampered = copy.deepcopy(manifest)
        tampered["media"][0]["dynamic_binary_eligible"] = (
            not tampered["media"][0]["dynamic_binary_eligible"]
        )
        c.raises(lambda: validate_participant_manifest(
            tampered, expected=expected
        ), ValueError, "eligibility tamper fails closed")
        target = root / "output" / "participant_manifest.json"
        write_private_no_overwrite_json(target, manifest)
        c.true(target.is_file(), "manifest is written once")
        c.raises(lambda: write_private_no_overwrite_json(
            target, manifest
        ), FileExistsError, "manifest overwrite is rejected")

    actions = {action.dest for action in manifest_parser()._actions}
    c.eq(actions, {"help", "data_root", "output"},
         "manifest CLI exposes no count, label, or eligibility overrides")


def test_metadata_trailing_blank_format_rows_are_ignored(c: Check):
    header = ("Category", "Sub-category", "#", "Gender", "Age",
              "Side of Paralysis", "Cause of Paralysis")
    data = ("Normal", None, 1, "Male", 40, "N/A", "N/A")
    blank = (None,) * 7
    c.eq(_nonempty_workbook_rows([header, data, blank, blank]), [header, data],
         "fully blank formatted tail rows are excluded")
    c.raises(lambda: _nonempty_workbook_rows([header, blank, data]), ValueError,
             "blank rows inside the data block fail closed")


def test_dynamic_extractor_uses_all_and_only_manifest_videos(c: Check):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        expected, metadata_keys = _fixture(root)
        manifest = build_participant_manifest(
            root,
            metadata_participant_keys=metadata_keys,
            metadata_xlsx_sha256="a" * 64,
            paper_pdf_sha256="b" * 64,
            expected=expected,
            audit=ManifestAudit(),
        )
        sources = enumerate_video_sources(root, manifest, expected=expected)
        c.eq(len(sources), 3, "every participant video is enumerated once")
        c.true(all(source.path.suffix.lower() == ".mp4" for source in sources),
               "photographs never enter the dynamic source list")
        c.eq(len({source.binding.group_id for source in sources}), 3,
             "one dynamic source maps to each participant")
        (root / "Normals/Normal1/extra.mp4").write_bytes(b"unexpected")
        c.raises(lambda: enumerate_video_sources(
            root, manifest, expected=expected
        ), ValueError, "an unmanifested video fails closed")


def test_extractor_model_lock_and_same_bytes_copy(c: Check):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        model = root / "face_landmarker.task"
        model.write_bytes(b"pinned-model")
        model_sha = sha256(model.read_bytes()).hexdigest()
        palsynet = {
            "schema_version": "palsynet_clinical23_v2_windows_v1",
            "feature_schema": "mediapipe_bs_lr_v1+clinical23_v2",
            "feature_shape": [4, 32, 95],
            "capture_mirrored": None,
            "protocol": {"windows_per_recording": 4, "frames_per_window": 32},
            "provenance": {"model_sha256": model_sha},
        }
        state = validate_palsynet_extractor_lock(palsynet, model)
        c.eq(state.model_sha256, model_sha,
             "MEEI extraction binds the exact PalsyNet model")
        model.write_bytes(b"different-model")
        c.raises(lambda: validate_palsynet_extractor_lock(
            palsynet, model
        ), ValueError, "model substitution fails before extraction")

        source = root / "source.mp4"
        source.write_bytes(b"authenticated-video-bytes")
        expected_sha = sha256(source.read_bytes()).hexdigest()
        private = root / "private"
        private.mkdir()
        copied = copy_authenticated_source(source, expected_sha, private)
        c.eq(sha256(copied.read_bytes()).hexdigest(), expected_sha,
             "decoder input is a same-byte authenticated private copy")
        c.true(copied != source and copied.parent == private,
               "decoder never reopens the governed source path")

    actions = {action.dest for action in extractor_parser()._actions}
    c.eq(actions, {
        "help", "data_root", "participant_manifest",
        "palsynet_cache_manifest", "model_path", "output_root",
    }, "extractor CLI exposes no media, QC, label, or schema overrides")


if __name__ == "__main__":
    run_all("test_meei_dynamic_cache", dict(globals()))
