"""Synthetic contracts for the Toronto NeuroFace external manifest."""
from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.datasets.neuroface_external_v1 import (  # noqa: E402
    PRIMARY_TASKS,
    ArchiveBinding,
    CohortSource,
    InventoryExpectation,
    audit_neuroface_sources,
    build_private_manifest,
    parse_landmark_text,
    parse_video_filename,
)
from _testlib import Check, run_all  # noqa: E402


def _zip(path: Path, members: dict[str, bytes]) -> str:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _landmarks(frames: tuple[int, ...] = (0, 5)) -> bytes:
    header = ["Frame"] + [part for index in range(1, 69) for part in (f"x{index}", f"y{index}")]
    rows = [", ".join(header)]
    for frame in frames:
        values = [str(frame)]
        for index in range(68):
            values.extend((str(10 + index), str(20 + index)))
        rows.append(", ".join(values))
    return ("\n".join(rows) + "\n").encode("ascii")


def _slp(rows: list[tuple[str, str]]) -> bytes:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append([
        "File Name", "Subject ID", "Symmetry (SLP1)", "ROM (SLP1)",
        "Speed (SLP1)", "Variability (SLP1)", "Fatigue (SLP1)", "Tot (SLP1)",
        None, "Symmetry (SLP2)", "ROM (SLP2)", "Speed (SLP2)",
        "Variability (SLP2)", "Fatigue (SLP2)", "Tot (SLP2)",
    ])
    for filename, subject in rows:
        sheet.append([filename, subject, 1, 2, 3, 4, 5, 15, None, 3, 4, 5, 2, 1, 15])
    payload = io.BytesIO()
    workbook.save(payload)
    return payload.getvalue()


def _fixture(root: Path, *, unsafe: bool = False, corrupt_landmark: bool = False):
    archive_bindings: dict[str, ArchiveBinding] = {}
    cohort_sources: list[CohortSource] = []
    definitions = (
        ("als", "A001", "affected"),
        ("healthy_control", "N001", "unaffected"),
        ("post_stroke", "S001", "affected"),
    )
    tasks = tuple(sorted(PRIMARY_TASKS))
    for cohort, subject, label in definitions:
        folder = root / cohort
        folder.mkdir()
        video_members: dict[str, bytes] = {}
        landmark_members: dict[str, bytes] = {}
        slp_rows: list[tuple[str, str]] = []
        for index, task in enumerate(tasks):
            filename = f"{subject}_02_{task}_color.avi"
            video_members[f"Videos/{filename}"] = f"avi-{cohort}-{task}".encode()
            landmark = _landmarks()
            if corrupt_landmark and cohort == "als" and index == 0:
                landmark = b"Frame, x1\n0, 1\n"
            landmark_members[f"Landmarks_gt/{Path(filename).stem}.txt"] = landmark
            slp_rows.append((filename, subject))
        if unsafe and cohort == "als":
            video_members["../escape.avi"] = b"unsafe"
        video_zip = folder / "videos.zip"
        landmark_zip = folder / "landmarks.zip"
        video_sha = _zip(video_zip, video_members)
        landmark_sha = _zip(landmark_zip, landmark_members)
        slp_path = folder / "slp.xlsx"
        slp_path.write_bytes(_slp(slp_rows))
        video_id = f"{cohort}_videos"
        landmark_id = f"{cohort}_landmarks"
        archive_bindings[video_id] = ArchiveBinding(video_id, video_zip, video_sha)
        archive_bindings[landmark_id] = ArchiveBinding(
            landmark_id, landmark_zip, landmark_sha
        )
        cohort_sources.append(CohortSource(
            cohort=cohort,
            binary_label=label,
            video_archive_id=video_id,
            landmark_archive_id=landmark_id,
            slp_workbook_path=slp_path,
        ))
    expected = InventoryExpectation(
        participants={"als": 1, "healthy_control": 1, "post_stroke": 1},
        videos={"als": 3, "healthy_control": 3, "post_stroke": 3},
        annotated_frames={"als": 6, "healthy_control": 6, "post_stroke": 6},
    )
    return archive_bindings, tuple(cohort_sources), expected


def test_filename_and_landmark_parsers_fail_closed(c: Check):
    parsed = parse_video_filename("A001_02_NSM_KISS_color.avi")
    c.eq((parsed.subject_id, parsed.session, parsed.task),
         ("A001", "02", "NSM_KISS"), "video identity/task parsed exactly")
    c.raises(lambda: parse_video_filename("A001_NSM_KISS.avi"), ValueError,
             "malformed video name rejected")
    rows = parse_landmark_text(_landmarks((0, 5)))
    c.eq(tuple(rows), (0, 5), "frame indices remain ordered")
    c.eq(rows[0].shape, (68, 2), "manual topology is exactly 68 by 2")
    c.raises(lambda: parse_landmark_text(b"Frame, x1\n0, 1\n"), ValueError,
             "truncated landmark rows rejected")


def test_landmark_annotation_order_is_not_assumed(c: Check):
    rows = _landmarks((9, 3))
    parsed = parse_landmark_text(rows)
    c.eq(tuple(parsed), (3, 9),
         "publisher gesture-order annotations are canonicalized by frame")
    header, first, _second = rows.decode("ascii").splitlines()
    duplicate = (header + "\n" + first + "\n" + first + "\n").encode("ascii")
    c.raises(lambda: parse_landmark_text(duplicate), ValueError,
             "duplicate frame annotations remain invalid")


def test_inventory_pairs_every_video_and_averages_raters(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        bindings, sources, expected = _fixture(Path(temporary))
        inventory = audit_neuroface_sources(
            bindings, sources, expected=expected
        )
        c.eq(len(inventory.participants), 3, "three fixture participants")
        c.eq(len(inventory.records), 9, "all fixture videos retained")
        c.eq(sum(record.annotated_frames for record in inventory.records), 18,
             "all manual frames accounted")
        c.true(all(record.slp_scores["symmetry"] == 2.0
                   for record in inventory.records),
               "SLP1 and SLP2 symmetry are averaged")
        c.true(all(record.slp_scores["rom"] == 3.0
                   for record in inventory.records),
               "SLP1 and SLP2 ROM are averaged")
        c.true(all(record.video_sha256 and record.landmark_sha256
                   for record in inventory.records),
               "every recording binds both byte streams")


def test_manifest_is_opaque_deterministic_and_primary_complete(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        bindings, sources, expected = _fixture(Path(temporary))
        inventory = audit_neuroface_sources(bindings, sources, expected=expected)
        first = build_private_manifest(inventory)
        second = build_private_manifest(inventory)
        c.eq(first, second, "manifest construction is deterministic")
        encoded = json.dumps(first, sort_keys=True)
        for raw_id in ("A001", "N001", "S001"):
            c.true(raw_id not in encoded, "publisher IDs are not serialized")
        c.true("Videos/" not in encoded and ".avi" not in encoded,
               "raw members and filenames are not serialized")
        c.eq(first["counts"]["primary_complete_participants"], 3,
             "all participants have exactly the locked primary tasks")
        c.true(all(row["participant_id"].startswith("grp_")
                   for row in first["participants"]), "participant IDs are opaque")
        c.true(all(row["recording_id"].startswith("rec_")
                   for row in first["records"]), "recording IDs are opaque")


def test_archive_hash_path_and_landmark_corruption_are_rejected(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        bindings, sources, expected = _fixture(Path(temporary), unsafe=True)
        c.raises(lambda: audit_neuroface_sources(bindings, sources, expected=expected),
                 ValueError, "unsafe archive member rejected")
    with tempfile.TemporaryDirectory() as temporary:
        bindings, sources, expected = _fixture(Path(temporary), corrupt_landmark=True)
        c.raises(lambda: audit_neuroface_sources(bindings, sources, expected=expected),
                 ValueError, "malformed manual landmark file rejected")
    with tempfile.TemporaryDirectory() as temporary:
        bindings, sources, expected = _fixture(Path(temporary))
        first = next(iter(bindings.values()))
        changed = dict(bindings)
        changed[first.archive_id] = ArchiveBinding(
            first.archive_id, first.path, "0" * 64
        )
        c.raises(lambda: audit_neuroface_sources(changed, sources, expected=expected),
                 ValueError, "archive digest mismatch rejected")


if __name__ == "__main__":
    run_all("test_neuroface_external_manifest_v1", dict(globals()))
