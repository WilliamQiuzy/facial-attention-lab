"""Synthetic contracts for the H200 full-frame Py-Feat extractor."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from scripts.extract_neuroface_au_v1 import (  # noqa: E402
    ENVIRONMENT_SCHEMA,
    _load_canonical_json,
    _parser,
    prioritize_extraction,
    select_paper_records,
    select_primary_faces,
    validate_environment_lock,
)
from _testlib import Check, run_all  # noqa: E402


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _row(cohort: str, task: str, index: int) -> dict[str, object]:
    return {
        "recording_id": "rec_" + f"{index:064x}",
        "participant_id": "grp_" + f"{index:064x}",
        "cohort": cohort,
        "binary_label": "affected" if cohort == "als" else "unaffected",
        "task": task,
        "video_archive_id": f"{cohort}_videos",
        "video_sha256": f"{index + 100:064x}",
        "video_size_bytes": 1000 + index,
    }


def test_primary_face_is_largest_not_first_or_highest_score(c: Check):
    boxes = [
        [
            [0.0, 0.0, 10.0, 10.0, 0.99],
            [20.0, 20.0, 60.0, 60.0, 0.90],
        ],
        [],
        [[1.0, 1.0, 21.0, 21.0, 0.95]],
    ]
    predictions = [
        np.stack((np.zeros(20, dtype=np.float32), np.ones(20, dtype=np.float32))),
        np.zeros((1, 20), dtype=np.float32),
        np.full((1, 20), 2.0, dtype=np.float32),
    ]
    values, valid, counts, scores = select_primary_faces(boxes, predictions)
    c.true(np.array_equal(values[0], np.ones(20, dtype=np.float32)),
           "largest face is the participant even when another score is higher")
    c.eq(valid.tolist(), [True, False, True], "missing faces remain missing")
    c.eq(counts.tolist(), [2, 0, 1], "all detected candidates are audited")
    c.true(np.array_equal(values[1], np.zeros(20, dtype=np.float32)),
           "missing frames use canonical zero values")
    c.eq(float(scores[0]), float(np.float32(0.90)), "selected score follows selected box")


def test_manifest_selection_is_exact_22_by_three(c: Check):
    rows = []
    index = 1
    for cohort, participants in (("als", 11), ("healthy_control", 11)):
        for _ in range(participants):
            participant = index
            for task in ("NSM_KISS", "NSM_OPEN", "NSM_SPREAD"):
                row = _row(cohort, task, index)
                row["participant_id"] = "grp_" + f"{participant:064x}"
                rows.append(row)
                index += 1
    manifest = {"records": rows}
    selected = select_paper_records(manifest)
    c.eq(len(selected), 66, "all three task views for 22 people are selected")
    c.eq(len({row["participant_id"] for row in selected}), 22,
         "selection unit is participant")
    c.eq({row["cohort"] for row in selected}, {"als", "healthy_control"},
         "stroke records cannot enter the paper endpoint")
    changed = json.loads(json.dumps(manifest))
    changed["records"].pop()
    c.raises(lambda: select_paper_records(changed), ValueError,
             "an incomplete 22-by-three Cartesian set fails closed")
    prioritized = prioritize_extraction(selected)
    c.eq(tuple(row["task"] for row in prioritized[:22]), ("NSM_SPREAD",) * 22,
         "paper-comparable SPREAD is completed before secondary task views")
    c.eq({row["recording_id"] for row in prioritized},
         {row["recording_id"] for row in selected},
         "priority changes execution order but not the frozen endpoint")


def test_environment_lock_binds_versions_resources_and_implementation(c: Check):
    resources = {"model.ubj": b"model", "landmark.pth": b"landmark"}
    lock = {
        "schema_version": ENVIRONMENT_SCHEMA,
        "versions": {
            "pyfeat": "0.6.2",
            "xgboost": "1.7.6",
            "torch": "2.2.1+cu121",
            "torchvision": "0.17.1+cu121",
        },
        "resources": {name: _sha(payload) for name, payload in resources.items()},
        "implementation_sha256": "a" * 64,
    }
    observed = validate_environment_lock(
        lock,
        versions=lock["versions"],
        resource_payloads=resources,
        implementation_sha256="a" * 64,
    )
    c.eq(observed, lock, "exact environment evidence passes")
    changed = dict(resources)
    changed["model.ubj"] = b"changed"
    c.raises(lambda: validate_environment_lock(
        lock, versions=lock["versions"], resource_payloads=changed,
        implementation_sha256="a" * 64,
    ), ValueError, "changed model bytes fail before extraction")
    c.raises(lambda: validate_environment_lock(
        lock, versions={**lock["versions"], "xgboost": "3.2.0"},
        resource_payloads=resources,
        implementation_sha256="a" * 64,
    ), ValueError, "runtime version drift fails")


def test_cli_has_no_label_or_candidate_controls(c: Check):
    options = {action.dest for action in _parser()._actions}
    for forbidden in (
        "label", "cohort", "task", "threshold", "candidate", "solver", "c",
        "penalty", "metric",
    ):
        c.true(forbidden not in options, f"extractor cannot tune {forbidden}")


def test_pinned_private_manifest_uses_frozen_pretty_canonical_form(c: Check):
    document = {"z": 1, "a": {"b": 2}}
    payload = (json.dumps(
        document, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False
    ) + "\n").encode("utf-8")
    c.eq(_load_canonical_json(payload, identity="manifest", pretty=True), document,
         "private manifest accepts the repository's exact frozen encoding")
    compact = (json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ) + "\n").encode("utf-8")
    c.raises(lambda: _load_canonical_json(
        compact, identity="manifest", pretty=True
    ), ValueError, "alternate encodings of the private manifest fail")


if __name__ == "__main__":
    run_all("test_extract_neuroface_au_v1", dict(globals()))
