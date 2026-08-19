"""Contracts for full-cohort NeuroFace Py-Feat AU extraction."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from scripts.extract_neuroface_full_au_v2 import (  # noqa: E402
    ALL_TASKS,
    COHORT_PARTICIPANTS,
    ENVIRONMENT_SCHEMA,
    EXPECTED_RECORDINGS,
    FROZEN_FRAME_STRIDE,
    STROKE_METADATA,
    _implementation_digest,
    _load_member_index,
    _member_index_bytes,
    _safe_full_zip_members,
    _terminal_decode_is_complete,
    _parser,
    collection_counts,
    deterministic_shard,
    prioritize_full_extraction,
    select_full_records,
)
from _testlib import Check, run_all  # noqa: E402


TASK_COUNTS = {
    "als": {
        "BBP_NORMAL": 9, "DDK_PA": 10, "DDK_PATAKA": 10,
        "NSM_BIGSMILE": 4, "NSM_BLOW": 6, "NSM_BROW": 4,
        "NSM_KISS": 11, "NSM_OPEN": 11, "NSM_SPREAD": 11,
    },
    "healthy_control": {
        "BBP_NORMAL": 11, "DDK_PA": 11, "DDK_PATAKA": 11,
        "NSM_BIGSMILE": 3, "NSM_BLOW": 7, "NSM_BROW": 4,
        "NSM_KISS": 11, "NSM_OPEN": 11, "NSM_SPREAD": 11,
    },
    "post_stroke": {
        "BBP_NORMAL": 14, "DDK_PA": 14, "DDK_PATAKA": 14,
        "NSM_BIGSMILE": 3, "NSM_BLOW": 11, "NSM_BROW": 7,
        "NSM_KISS": 14, "NSM_OPEN": 14, "NSM_SPREAD": 14,
    },
}


def _digest(index: int) -> str:
    return hashlib.sha256(f"synthetic-{index}".encode("ascii")).hexdigest()


def _manifest() -> dict[str, object]:
    rows = []
    index = 0
    for cohort, participant_count in COHORT_PARTICIPANTS.items():
        participants = [f"grp_{_digest(10_000 + index + value)}"
                        for value in range(participant_count)]
        index += participant_count
        for task, count in TASK_COUNTS[cohort].items():
            for participant in participants[:count]:
                source = _digest(100_000 + index)
                rows.append({
                    "recording_id": f"rec_{_digest(200_000 + index)}",
                    "participant_id": participant,
                    "cohort": cohort,
                    "binary_label": (
                        "unaffected" if cohort == "healthy_control" else "affected"
                    ),
                    "task": task,
                    "video_archive_id": f"{cohort}_videos",
                    "video_sha256": source,
                    "video_size_bytes": 1_000 + index,
                })
                index += 1
    return {"records": rows}


def test_full_manifest_is_exact_36_people_261_recordings(c: Check):
    selected = select_full_records(_manifest())
    c.eq(len(selected), EXPECTED_RECORDINGS, "all committed recordings are selected")
    c.eq(len({row["participant_id"] for row in selected}), 36,
         "selection contains exactly 36 people")
    c.eq(Counter(row["cohort"] for row in selected),
         Counter({"als": 76, "healthy_control": 80, "post_stroke": 105}),
         "all three cohort recording totals are frozen")
    c.eq(set(row["task"] for row in selected), set(ALL_TASKS),
         "all nine committed tasks are represented")


def test_selection_rejects_membership_label_and_identity_drift(c: Check):
    missing = _manifest()
    missing["records"].pop()
    c.raises(lambda: select_full_records(missing), ValueError,
             "missing one committed recording fails closed")

    wrong_label = _manifest()
    wrong_label["records"][0]["binary_label"] = "unaffected"
    c.raises(lambda: select_full_records(wrong_label), ValueError,
             "cohort and binary outcome cannot disagree")

    wrong_archive = _manifest()
    wrong_archive["records"][0]["video_archive_id"] = "unknown_videos"
    c.raises(lambda: select_full_records(wrong_archive), ValueError,
             "an uncommitted archive cannot enter extraction")

    duplicate = _manifest()
    duplicate["records"][1]["video_sha256"] = duplicate["records"][0]["video_sha256"]
    c.raises(lambda: select_full_records(duplicate), ValueError,
             "video content cannot occur twice")


def test_extraction_order_is_fixed_but_membership_is_unchanged(c: Check):
    selected = select_full_records(_manifest())
    ordered = prioritize_full_extraction(selected)
    c.eq(tuple(row["task"] for row in ordered[:36]), ("NSM_SPREAD",) * 36,
         "the strongest common action is extracted first")
    c.eq({row["recording_id"] for row in ordered},
         {row["recording_id"] for row in selected},
         "priority never changes cohort membership")


def test_six_execution_shards_partition_the_frozen_order_exactly(c: Check):
    ordered = prioritize_full_extraction(select_full_records(_manifest()))
    shards = tuple(
        deterministic_shard(ordered, count=6, index=index)
        for index in range(6)
    )
    c.eq(sum(len(shard) for shard in shards), EXPECTED_RECORDINGS,
         "shards preserve the exact endpoint size")
    c.eq(
        {row["recording_id"] for shard in shards for row in shard},
        {row["recording_id"] for row in ordered},
        "shards neither omit nor duplicate recordings",
    )
    for index, shard in enumerate(shards):
        c.eq(
            shard,
            tuple(ordered[position] for position in range(index, len(ordered), 6)),
            "each shard is the frozen ordinal subsequence",
        )
    c.raises(lambda: deterministic_shard(ordered, count=9, index=0), ValueError,
             "execution parallelism is bounded")


def test_member_index_is_closed_and_cannot_change_committed_content(c: Check):
    selected = select_full_records(_manifest())
    archives = {
        "als_videos": Path("/archive/als.zip"),
        "healthy_control_videos": Path("/archive/healthy.zip"),
        "post_stroke_videos": Path("/archive/stroke.zip"),
    }
    indexed = {
        str(row["video_sha256"]): (
            archives[str(row["video_archive_id"])],
            f"Videos/{row['recording_id']}.avi",
            str(row["video_archive_id"]),
        )
        for row in selected
    }
    payload = _member_index_bytes(indexed, selected)
    loaded = _load_member_index(payload, archives, selected)
    c.eq(loaded, indexed, "canonical member index round-trips exact commitments")

    document = json.loads(payload)
    document["records"][0]["source_sha256"] = "f" * 64
    tampered = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    c.raises(lambda: _load_member_index(tampered, archives, selected), ValueError,
             "an index cannot substitute a source commitment")


def test_collection_counts_are_recomputed_from_rows(c: Check):
    rows = select_full_records(_manifest())
    counts = collection_counts(rows)
    c.eq(counts["participants"], 36, "collection counts participants")
    c.eq(counts["recordings"], 261, "collection counts recordings")
    c.eq(counts["cohort_participants"], dict(COHORT_PARTICIPANTS),
         "collection counts every cohort")
    c.eq(counts["tasks"], dict(Counter(row["task"] for row in rows)),
         "collection task counts come from selected rows")


def test_cli_has_all_archives_and_no_outcome_or_search_controls(c: Check):
    options = {action.dest for action in _parser()._actions}
    for required in (
        "private_manifest", "als_zip", "healthy_zip", "stroke_zip",
        "environment_lock", "output_root",
    ):
        c.true(required in options, f"CLI requires {required}")
    for forbidden in (
        "label", "cohort", "task", "threshold", "candidate", "solver",
        "penalty", "metric", "include", "exclude",
        "frame_stride", "fps", "sample_rate",
    ):
        c.true(forbidden not in options, f"CLI cannot tune {forbidden}")
    c.eq(FROZEN_FRAME_STRIDE, 3,
         "temporal sampling is a fixed representation contract, not a tuning knob")


def test_environment_lock_binds_exact_v2_implementation_and_resources(c: Check):
    path = ROOT / "environment" / "neuroface_pyfeat_xgb_environment_v2.json"
    payload = path.read_bytes()
    lock = json.loads(payload)
    c.eq(lock.get("schema_version"), ENVIRONMENT_SCHEMA,
         "environment lock uses the full-cohort schema")
    c.eq(lock.get("implementation_sha256"), _implementation_digest(),
         "environment lock binds the exact extractor closure")
    resources = lock.get("resources")
    c.true(isinstance(resources, dict) and len(resources) == 29,
           "environment lock contains the complete resource set")
    c.true(all(
        isinstance(value, str) and len(value) == 64
        and set(value).issubset(set("0123456789abcdef"))
        for value in resources.values()
    ), "every model resource uses a canonical SHA-256")
    c.true(payload.endswith(b"\n"), "environment lock has canonical final newline")


def test_only_a_decoder_error_after_every_declared_frame_is_recoverable(c: Check):
    c.true(_terminal_decode_is_complete(4_127, 4_127),
           "malformed trailing packets after the declared stream are ignored")
    for decoded, declared in ((4_126, 4_127), (0, 0), (-1, 1), (True, 1)):
        c.true(not _terminal_decode_is_complete(decoded, declared),
               "early, empty and non-integer decoder failures remain fatal")


def test_script_executes_directly_without_test_path_injection(c: Check):
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "extract_neuroface_full_au_v2.py"),
         "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        check=False,
    )
    c.eq(completed.returncode, 0,
         "production CLI must import from a direct script invocation")


def test_full_archive_parser_remains_closed_while_stroke_metadata_is_pinned(c: Check):
    c.eq(
        STROKE_METADATA,
        {
            "SLP_Assessment_PS.xlsx": (
                24675,
                "2f8d683d40f5b398528611a1dcd524841d06b17ece27ea4c60b8f61277611f4b",
            ),
            "VID_DATASET_Clinical information_Stroke.csv": (
                1109,
                "ccbcaa4b79d50c564523746879ad7293623a48fa788f0631781fd6db72de2661",
            ),
            "VideoInfoFile_PS.xlsx": (
                19877,
                "540990039fc4ed11c8f01c052c50f1ea7759dced265395033fe85a8d1c04a708",
            ),
        },
        "the only non-video archive members are exact content commitments",
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "archive.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("Videos/example.avi", b"video")
        with zipfile.ZipFile(path, "r") as archive:
            members = _safe_full_zip_members(archive, "als_videos")
        c.eq(tuple(info.filename for info in members), ("Videos/example.avi",),
             "a canonical AVI member remains available")

        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("Videos/example.avi", b"video")
            archive.writestr("unexpected.csv", b"not committed")
        with zipfile.ZipFile(path, "r") as archive:
            c.raises(
                lambda: _safe_full_zip_members(archive, "post_stroke_videos"),
                ValueError,
                "an uncommitted metadata member cannot be silently ignored",
            )


if __name__ == "__main__":
    run_all("test_extract_neuroface_full_au_v2", dict(globals()))
