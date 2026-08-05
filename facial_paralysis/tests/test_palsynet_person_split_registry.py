"""Tests for the immutable patient/group-disjoint PalsyNet split registry."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.freeze_palsynet_person_split_registry import (  # noqa: E402
    DOMAIN_SEPARATOR,
    build_person_split_registry,
    canonical_json_bytes,
    validate_person_split_registry,
    write_person_split_registry,
)
from _testlib import Check, run_all  # noqa: E402


def _opaque(prefix: str, number: int) -> str:
    return f"{prefix}_{number:064x}"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _fixture() -> tuple[dict[str, object], dict[str, object]]:
    recordings: list[dict[str, object]] = []
    mapping: list[dict[str, str]] = []
    recording_ids = [_opaque("rec", number) for number in range(1, 50)]
    # Recordings 1 and 2 are the same reviewed affected person.  All others
    # are distinct, yielding the real audit's provisional 48-group shape.
    groups = {
        recording_id: _opaque("grp", 1 if index <= 2 else index)
        for index, recording_id in enumerate(recording_ids, start=1)
    }
    for index, recording_id in enumerate(recording_ids, start=1):
        label = "affected" if index <= 27 else "unaffected"
        group_id = groups[recording_id]
        recordings.append({
            "recording_id": recording_id,
            "group_id": group_id,
            "source_sha256": _sha(f"source-{index}"),
            "source_label": label,
            "label": label,
            "identity_status": "reviewed",
            "claim_unit": "person_held_out",
            "training_eligible": True,
            "adjudication_outcome": "none",
            "adjudication_evidence_sha256": None,
        })
        mapping.append({"recording_id": recording_id, "group_id": group_id})

    decisions = []
    for first, second in itertools.combinations(sorted(recording_ids), 2):
        decisions.append({
            "recording_id_a": first,
            "recording_id_b": second,
            "decision": "same" if groups[first] == groups[second] else "different",
        })
    ledger = {
        "schema_version": "palsynet_identity_review_ledger_v1",
        "dataset": "PalsyNet",
        "source_collection_sha256": _sha("source-collection"),
        "generated_manifest_sha256": _sha("generated-manifest"),
        "contact_inventory_sha256": _sha("contact-inventory"),
        "reviewer_evidence_sha256": _sha("reviewer-evidence"),
        "label_blinded": True,
        "uncertainty_status": "resolved",
        "recording_to_group": mapping,
        "pair_decisions": decisions,
    }
    ledger_sha = hashlib.sha256(canonical_json_bytes(ledger)).hexdigest()
    reviewed = {
        "schema_version": "palsynet_identity_reviewed_v1",
        "dataset": "PalsyNet",
        "claim_unit": "person_held_out",
        "identity_review": {
            "status": "reviewed",
            "label_blinded": True,
            "exhaustive_pair_review": True,
            "uncertainties_resolved": True,
        },
        "counts": {
            "total_recordings": 49,
            "reviewed_groups": 48,
            "eligible_recordings": 49,
            "eligible_groups": 48,
            "excluded_recordings": 0,
            "excluded_groups": 0,
        },
        "fingerprints": {
            "source_collection_sha256": ledger["source_collection_sha256"],
            "generated_manifest_sha256": ledger["generated_manifest_sha256"],
            "contact_inventory_sha256": ledger["contact_inventory_sha256"],
            "review_ledger_sha256": ledger_sha,
            "reviewer_evidence_sha256": ledger["reviewer_evidence_sha256"],
            "cross_label_adjudication_sha256": _sha("adjudication"),
        },
        "recordings": recordings,
    }
    return reviewed, ledger


def _build() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    reviewed, ledger = _fixture()
    reviewed_sha = hashlib.sha256(canonical_json_bytes(reviewed)).hexdigest()
    ledger_sha = hashlib.sha256(canonical_json_bytes(ledger)).hexdigest()
    registry = build_person_split_registry(
        reviewed,
        ledger,
        reviewed_manifest_sha256=reviewed_sha,
        review_ledger_sha256=ledger_sha,
    )
    return reviewed, ledger, registry


def test_semantic_split_is_group_disjoint_and_complete(c: Check):
    reviewed, ledger, registry = _build()
    c.eq(registry["schema_version"], "palsynet_person_split_registry_v1")
    c.eq(registry["outer_fold_number"], 0)
    c.eq(registry["protocol"], {
        "domain_separator": DOMAIN_SEPARATOR,
        "outer_folds": 5,
        "inner_folds": 4,
        "semantic_group_key": (
            "sha256(domain_separator + ':' + comma_join(sorted_member_source_sha256))"
        ),
        "stratification": "binary_label_then_group_size_then_semantic_key",
    })
    c.eq(len(registry["assignments"]), 49, "every eligible recording is assigned")
    by_group: dict[str, set[tuple[object, ...]]] = {}
    for row in registry["assignments"]:
        by_group.setdefault(row["group_id"], set()).add((
            row["partition"], row["outer_fold"], row["inner_fold"],
            row["semantic_group_key_sha256"],
        ))
    c.true(all(len(values) == 1 for values in by_group.values()),
           "one reviewed person can never cross folds")
    c.eq(set(row["inner_fold"] for row in registry["assignments"]
             if row["partition"] == "development"), {0, 1, 2, 3})
    c.true(all(row["outer_fold"] == 0 and row["inner_fold"] is None
               for row in registry["assignments"]
               if row["partition"] == "protected"),
           "outer fold zero is sealed")
    validate_person_split_registry(registry, reviewed, ledger)


def test_split_order_ignores_opaque_group_ids_and_audit_salt(c: Check):
    reviewed, ledger, first = _build()
    remap = {
        group_id: _opaque("grp", 10_000 + index)
        for index, group_id in enumerate(sorted({
            row["group_id"] for row in reviewed["recordings"]
        }))
    }
    changed = copy.deepcopy(reviewed)
    changed_ledger = copy.deepcopy(ledger)
    for row in changed["recordings"]:
        row["group_id"] = remap[row["group_id"]]
    for row in changed_ledger["recording_to_group"]:
        row["group_id"] = remap[row["group_id"]]
    changed_ledger_sha = hashlib.sha256(
        canonical_json_bytes(changed_ledger)
    ).hexdigest()
    changed["fingerprints"]["review_ledger_sha256"] = changed_ledger_sha
    second = build_person_split_registry(
        changed,
        changed_ledger,
        reviewed_manifest_sha256=hashlib.sha256(
            canonical_json_bytes(changed)
        ).hexdigest(),
        review_ledger_sha256=changed_ledger_sha,
    )
    first_assignments = {
        row["recording_id"]: (row["partition"], row["outer_fold"], row["inner_fold"])
        for row in first["assignments"]
    }
    second_assignments = {
        row["recording_id"]: (row["partition"], row["outer_fold"], row["inner_fold"])
        for row in second["assignments"]
    }
    c.eq(first_assignments, second_assignments,
         "opaque reviewer IDs cannot influence the semantic split")


def test_digest_label_and_registry_drift_fail_closed(c: Check):
    reviewed, ledger, registry = _build()
    c.raises(lambda: build_person_split_registry(
        reviewed, ledger,
        reviewed_manifest_sha256="0" * 64,
        review_ledger_sha256=hashlib.sha256(canonical_json_bytes(ledger)).hexdigest(),
    ), ValueError, "reviewed bytes are authenticated")

    mixed = copy.deepcopy(reviewed)
    mixed["recordings"][1]["label"] = "unaffected"
    c.raises(lambda: build_person_split_registry(
        mixed, ledger,
        reviewed_manifest_sha256=hashlib.sha256(canonical_json_bytes(mixed)).hexdigest(),
        review_ledger_sha256=hashlib.sha256(canonical_json_bytes(ledger)).hexdigest(),
    ), ValueError, "mixed-label identity groups cannot enter the split")

    alternate = copy.deepcopy(registry)
    dev_rows = [row for row in alternate["assignments"]
                if row["partition"] == "development"]
    dev_rows[0]["inner_fold"] = (dev_rows[0]["inner_fold"] + 1) % 4
    c.raises(lambda: validate_person_split_registry(alternate, reviewed, ledger),
             ValueError, "alternate split registries are rejected")


def test_registry_write_is_private_and_no_overwrite(c: Check):
    reviewed, ledger, registry = _build()
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "person_split_registry.json"
        write_person_split_registry(output, registry)
        c.eq(output.read_bytes(), canonical_json_bytes(registry))
        c.eq(os.stat(output).st_mode & 0o777, 0o600,
             "private split registry is owner-readable only")
        c.raises(lambda: write_person_split_registry(output, registry),
                 FileExistsError, "registry is immutable")


if __name__ == "__main__":
    run_all("test_palsynet_person_split_registry", dict(globals()))
