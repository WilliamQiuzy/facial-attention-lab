"""Closed-contract tests for the exhaustive PalsyNet identity review."""
from __future__ import annotations

import base64
import copy
import hashlib
import itertools
import os
import stat
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.palsynet_identity_review import (  # noqa: E402
    build_contact_sheet_inventory,
    build_review_ledger_template,
    build_reviewed_identity_manifest,
    canonical_json_bytes,
    finalize_identity_review,
)
import scripts.finalize_palsynet_identity_review as finalizer_module  # noqa: E402
from _testlib import Check, run_all  # noqa: E402


RECORDING_COUNT = 49
PAIR_COUNT = 1176


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _payload_sha(payload: object) -> str:
    return _sha(canonical_json_bytes(payload))


def _documented_evidence(payload: bytes) -> dict[str, str]:
    return {
        "encoding": "base64",
        "media_type": "application/pdf",
        "payload_base64": base64.b64encode(payload).decode("ascii"),
        "sha256": _sha(payload),
    }


def _ids() -> list[str]:
    return [f"rec_{index:064x}" for index in range(RECORDING_COUNT)]


def _group(index: int) -> str:
    return f"grp_{index:064x}"


def _generated_manifest() -> dict[str, object]:
    recording_ids = _ids()
    recordings = [
        {
            "recording_id": recording_id,
            "group_id": _group(index),
            "label": "affected" if index < 27 else "unaffected",
            "source_sha256": f"{index + 100:064x}",
            "identity_status": "unreviewed",
            "claim_unit": "video_held_out",
        }
        for index, recording_id in enumerate(recording_ids)
    ]
    pairs = [
        {
            "rank": rank,
            "recording_id_a": first,
            "recording_id_b": second,
            "cosine": 0.0,
        }
        for rank, (first, second) in enumerate(
            itertools.combinations(recording_ids, 2), 1
        )
    ]
    return {
        "schema_version": "palsynet_identity_audit_v1",
        "dataset": "PalsyNet",
        "claim_unit": "video_held_out",
        "identity_review": {
            "status": "unreviewed",
            "group_override_applied": False,
            "manual_review_required": True,
            "reviewer_evidence_sha256": None,
        },
        "counts": {
            "affected": 27,
            "unaffected": 22,
            "total": RECORDING_COUNT,
            "ranked_pairs": PAIR_COUNT,
        },
        "fingerprints": {
            "source_collection_sha256": "a" * 64,
            "bundle_provenance_sha256": "b" * 64,
            "embedding_collection_sha256": "c" * 64,
        },
        "contact_sheet_sampling": {
            "windows_per_video": 4,
            "window_size_frames": 32,
            "representative_frame_offset": 16,
            "raw_filename_text_burned_in": False,
        },
        "contact_sheets": {
            "recordings": RECORDING_COUNT,
            "ranked_pairs": PAIR_COUNT,
            "overview": 1,
            "storage": "local_ignored_output",
            "filenames": "opaque_ids_or_ranks_only",
        },
        "recordings": recordings,
        "ranked_pairs": pairs,
    }


def _reviewer_evidence() -> dict[str, object]:
    return {
        "schema_version": "palsynet_identity_reviewer_evidence_v1",
        "dataset": "PalsyNet",
        "review_protocol": "label_blinded_exhaustive_pair_review_v1",
        "reviewer_id": "reviewer_" + "d" * 64,
        "label_accessed_during_identity_review": False,
        "recording_overviews_reviewed": RECORDING_COUNT,
        "pair_decisions_reviewed": PAIR_COUNT,
        "uncertainties_resolved": True,
    }


def _fixture() -> dict[str, dict[str, object] | str]:
    generated = _generated_manifest()
    generated_sha = _payload_sha(generated)
    recording_files = {
        row["recording_id"]: _sha(
            f"recording:{row['recording_id']}".encode("ascii")
        )
        for row in generated["recordings"]
    }
    pair_files = {
        (row["recording_id_a"], row["recording_id_b"]): _sha(
            f"pair:{row['rank']}".encode("ascii")
        )
        for row in generated["ranked_pairs"]
    }
    inventory = build_contact_sheet_inventory(
        generated,
        generated_manifest_sha256=generated_sha,
        overview_sha256=_sha(b"label-blinded-overview"),
        recording_sheet_sha256=recording_files,
        pair_sheet_sha256=pair_files,
    )
    inventory_sha = _payload_sha(inventory)
    evidence = _reviewer_evidence()
    evidence_sha = _payload_sha(evidence)
    template = build_review_ledger_template(
        generated,
        generated_manifest_sha256=generated_sha,
        contact_inventory_sha256=inventory_sha,
    )
    ledger = copy.deepcopy(template)
    ledger["schema_version"] = "palsynet_identity_review_ledger_v1"
    ledger["reviewer_evidence_sha256"] = evidence_sha
    ledger["uncertainty_status"] = "resolved"
    for index, row in enumerate(ledger["recording_to_group"]):
        row["group_id"] = _group(index)
    for row in ledger["pair_decisions"]:
        row["decision"] = "different"
    ledger_sha = _payload_sha(ledger)
    adjudication = {
        "schema_version": "palsynet_cross_label_adjudication_v1",
        "dataset": "PalsyNet",
        "source_collection_sha256": generated["fingerprints"][
            "source_collection_sha256"
        ],
        "review_ledger_sha256": ledger_sha,
        "reviewer_evidence_sha256": evidence_sha,
        "decisions": [],
    }
    return {
        "generated": generated,
        "generated_sha": generated_sha,
        "inventory": inventory,
        "inventory_sha": inventory_sha,
        "evidence": evidence,
        "evidence_sha": evidence_sha,
        "ledger": ledger,
        "ledger_sha": ledger_sha,
        "adjudication": adjudication,
        "adjudication_sha": _payload_sha(adjudication),
    }


def _refresh(fixture: dict[str, dict[str, object] | str]) -> None:
    fixture["generated_sha"] = _payload_sha(fixture["generated"])
    fixture["inventory_sha"] = _payload_sha(fixture["inventory"])
    fixture["evidence_sha"] = _payload_sha(fixture["evidence"])
    fixture["ledger"]["generated_manifest_sha256"] = fixture["generated_sha"]
    fixture["ledger"]["contact_inventory_sha256"] = fixture["inventory_sha"]
    fixture["ledger"]["reviewer_evidence_sha256"] = fixture["evidence_sha"]
    fixture["ledger_sha"] = _payload_sha(fixture["ledger"])
    fixture["adjudication"]["source_collection_sha256"] = fixture["generated"][
        "fingerprints"
    ]["source_collection_sha256"]
    fixture["adjudication"]["review_ledger_sha256"] = fixture["ledger_sha"]
    fixture["adjudication"]["reviewer_evidence_sha256"] = fixture["evidence_sha"]
    fixture["adjudication_sha"] = _payload_sha(fixture["adjudication"])


def _build(fixture: dict[str, dict[str, object] | str]) -> dict[str, object]:
    return build_reviewed_identity_manifest(
        fixture["generated"],
        fixture["inventory"],
        fixture["ledger"],
        fixture["evidence"],
        fixture["adjudication"],
        generated_manifest_sha256=fixture["generated_sha"],
        contact_inventory_sha256=fixture["inventory_sha"],
        review_ledger_sha256=fixture["ledger_sha"],
        reviewer_evidence_sha256=fixture["evidence_sha"],
        cross_label_adjudication_sha256=fixture["adjudication_sha"],
    )


def _set_group(
    fixture: dict[str, dict[str, object] | str],
    recording_ids: set[str],
    group_id: str,
) -> None:
    mapping = {
        row["recording_id"]: row for row in fixture["ledger"]["recording_to_group"]
    }
    for recording_id in recording_ids:
        mapping[recording_id]["group_id"] = group_id
    groups = {
        row["recording_id"]: row["group_id"]
        for row in fixture["ledger"]["recording_to_group"]
    }
    for pair in fixture["ledger"]["pair_decisions"]:
        pair["decision"] = (
            "same"
            if groups[pair["recording_id_a"]] == groups[pair["recording_id_b"]]
            else "different"
        )
    _refresh(fixture)


def test_exhaustive_label_blind_ledger_builds_reviewed_manifest(c: Check):
    fixture = _fixture()
    reviewed = _build(fixture)
    c.eq(reviewed["schema_version"], "palsynet_identity_reviewed_v1")
    c.eq(reviewed["claim_unit"], "person_held_out")
    c.eq(reviewed["counts"]["total_recordings"], RECORDING_COUNT)
    c.eq(reviewed["counts"]["reviewed_groups"], RECORDING_COUNT)
    c.eq(reviewed["counts"]["eligible_recordings"], RECORDING_COUNT)
    c.true(all(row["training_eligible"] for row in reviewed["recordings"]),
           "complete nonconflicting review makes every recording eligible")
    c.eq(reviewed["fingerprints"], {
        "source_collection_sha256": "a" * 64,
        "generated_manifest_sha256": fixture["generated_sha"],
        "contact_inventory_sha256": fixture["inventory_sha"],
        "review_ledger_sha256": fixture["ledger_sha"],
        "reviewer_evidence_sha256": fixture["evidence_sha"],
        "cross_label_adjudication_sha256": fixture["adjudication_sha"],
    }, "reviewed manifest binds every exact input digest")


def test_ledger_requires_complete_unique_recording_and_pair_coverage(c: Check):
    missing_recording = _fixture()
    missing_recording["ledger"]["recording_to_group"].pop()
    _refresh(missing_recording)
    c.raises(lambda: _build(missing_recording), ValueError,
             "every recording needs exactly one final group")

    missing_pair = _fixture()
    missing_pair["ledger"]["pair_decisions"].pop()
    _refresh(missing_pair)
    c.raises(lambda: _build(missing_pair), ValueError,
             "all 1176 unordered decisions are mandatory")

    duplicate_reversed = _fixture()
    pair = copy.deepcopy(duplicate_reversed["ledger"]["pair_decisions"][0])
    pair["recording_id_a"], pair["recording_id_b"] = (
        pair["recording_id_b"], pair["recording_id_a"]
    )
    duplicate_reversed["ledger"]["pair_decisions"][-1] = pair
    _refresh(duplicate_reversed)
    c.raises(lambda: _build(duplicate_reversed), ValueError,
             "reversed duplicates cannot replace a missing unordered pair")

    malformed = _fixture()
    malformed["ledger"]["recording_to_group"][0]["recording_id"] = "Alice"
    _refresh(malformed)
    c.raises(lambda: _build(malformed), ValueError,
             "recording and group ids must remain opaque")


def test_pair_decisions_and_final_groups_form_one_equivalence_relation(c: Check):
    same_across_groups = _fixture()
    same_across_groups["ledger"]["pair_decisions"][0]["decision"] = "same"
    _refresh(same_across_groups)
    c.raises(lambda: _build(same_across_groups), ValueError,
             "same is forbidden across final groups")

    different_within_group = _fixture()
    first, second = _ids()[:2]
    for row in different_within_group["ledger"]["recording_to_group"]:
        if row["recording_id"] in {first, second}:
            row["group_id"] = _group(900)
    _refresh(different_within_group)
    c.raises(lambda: _build(different_within_group), ValueError,
             "different is forbidden within one final group")

    non_transitive = _fixture()
    first, second, third = _ids()[:3]
    decisions = {
        frozenset((row["recording_id_a"], row["recording_id_b"])): row
        for row in non_transitive["ledger"]["pair_decisions"]
    }
    decisions[frozenset((first, second))]["decision"] = "same"
    decisions[frozenset((second, third))]["decision"] = "same"
    decisions[frozenset((first, third))]["decision"] = "different"
    _refresh(non_transitive)
    c.raises(lambda: _build(non_transitive), ValueError,
             "the implied same relation must be transitive")


def test_label_blind_attestation_is_structured_and_uncertainty_is_closed(c: Check):
    arbitrary_evidence = _fixture()
    arbitrary_evidence["evidence"] = {"reviewed": True}
    _refresh(arbitrary_evidence)
    c.raises(lambda: _build(arbitrary_evidence), ValueError,
             "an arbitrary nonempty evidence object is not an attestation")

    label_informed = _fixture()
    label_informed["ledger"]["label_blinded"] = False
    _refresh(label_informed)
    c.raises(lambda: _build(label_informed), ValueError,
             "identity grouping cannot use binary labels")

    accessed = _fixture()
    accessed["evidence"]["label_accessed_during_identity_review"] = True
    _refresh(accessed)
    c.raises(lambda: _build(accessed), ValueError,
             "reviewer evidence must attest label blindness")

    unresolved = _fixture()
    unresolved["ledger"]["uncertainty_status"] = "unresolved"
    _refresh(unresolved)
    c.raises(lambda: _build(unresolved), ValueError,
             "unresolved identity uncertainty keeps the gate closed")

    uncertain_pair = _fixture()
    uncertain_pair["ledger"]["pair_decisions"][0]["decision"] = "uncertain"
    _refresh(uncertain_pair)
    c.raises(lambda: _build(uncertain_pair), ValueError,
             "pair decisions are closed to same or different")


def test_every_digest_binding_is_exact(c: Check):
    names = (
        "generated_sha",
        "inventory_sha",
        "ledger_sha",
        "evidence_sha",
        "adjudication_sha",
    )
    for name in names:
        fixture = _fixture()
        fixture[name] = "f" * 64
        c.raises(lambda fixture=fixture: _build(fixture), ValueError,
                 f"{name} must match exact artifact bytes")

    stale_binding = _fixture()
    stale_binding["evidence"]["reviewer_id"] = "reviewer_" + "e" * 64
    stale_binding["evidence_sha"] = _payload_sha(stale_binding["evidence"])
    c.raises(lambda: _build(stale_binding), ValueError,
             "ledger cannot retain a stale reviewer-evidence digest")


def test_cross_label_groups_require_closed_adjudication_outcomes(c: Check):
    first, second = _ids()[0], _ids()[27]
    fixture = _fixture()
    _set_group(fixture, {first, second}, _group(999))
    c.raises(lambda: _build(fixture), ValueError,
             "a cross-label reviewed group needs one adjudication")

    bare_flag = copy.deepcopy(fixture)
    bare_flag["adjudication"]["training_eligible"] = True
    _refresh(bare_flag)
    c.raises(lambda: _build(bare_flag), ValueError,
             "a bare flag cannot authorize mixed-label training")

    excluded = copy.deepcopy(fixture)
    excluded["adjudication"]["decisions"] = [{
        "group_id": _group(999),
        "outcome": "exclude_whole_group",
        "evidence": _documented_evidence(b"documented whole-group exclusion"),
        "corrected_label": None,
    }]
    _refresh(excluded)
    reviewed = _build(excluded)
    affected_rows = [
        row for row in reviewed["recordings"]
        if row["recording_id"] in {first, second}
    ]
    c.true(all(not row["training_eligible"] for row in affected_rows),
           "whole-group exclusion never splits a reviewed identity")

    corrected = copy.deepcopy(fixture)
    corrected["adjudication"]["decisions"] = [{
        "group_id": _group(999),
        "outcome": "correct_proven_source_label_error",
        "evidence": _documented_evidence(
            b"documented source-label correction"
        ),
        "corrected_label": "affected",
    }]
    _refresh(corrected)
    reviewed = _build(corrected)
    corrected_rows = [
        row for row in reviewed["recordings"]
        if row["recording_id"] in {first, second}
    ]
    c.true(all(row["training_eligible"] and row["label"] == "affected"
               for row in corrected_rows),
           "documented source correction yields one consistent group label")
    correction_evidence_sha256 = corrected["adjudication"]["decisions"][0][
        "evidence"
    ]["sha256"]
    c.true(all(
        row["adjudication_evidence_sha256"] == correction_evidence_sha256
        for row in corrected_rows
    ), "reviewed rows retain the authenticated correction-evidence digest")

    tampered_evidence = copy.deepcopy(corrected)
    tampered_evidence["adjudication"]["decisions"][0]["evidence"][
        "payload_base64"
    ] = base64.b64encode(b"fabricated replacement").decode("ascii")
    _refresh(tampered_evidence)
    c.raises(lambda: _build(tampered_evidence), ValueError,
             "documented evidence bytes must match their authenticated digest")

    invalid = copy.deepcopy(fixture)
    invalid["adjudication"]["decisions"] = [{
        "group_id": _group(999),
        "outcome": "split_group",
        "evidence": _documented_evidence(b"invalid split request"),
        "corrected_label": "affected",
    }]
    _refresh(invalid)
    c.raises(lambda: _build(invalid), ValueError,
             "adjudication can never split a reviewed identity group")


def _write_private(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    path.write_bytes(canonical_json_bytes(payload))
    path.chmod(0o600)


def _write_contact_sheet_files(
    root: Path,
    fixture: dict[str, dict[str, object] | str],
) -> None:
    inventory = fixture["inventory"]
    payloads = [
        (inventory["overview"]["relative_path"], b"label-blinded-overview"),
        *(
            (
                row["relative_path"],
                f"recording:{row['recording_id']}".encode("ascii"),
            )
            for row in inventory["recordings"]
        ),
        *(
            (row["relative_path"], f"pair:{row['rank']}".encode("ascii"))
            for row in inventory["pairs"]
        ),
    ]
    for relative_path, payload in payloads:
        path = root / "generation" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.parent.chmod(0o700)
        path.write_bytes(payload)
        path.chmod(0o600)


def test_finalizer_enforces_immutable_paths_owner_only_and_no_overwrite(c: Check):
    fixture = _fixture()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "palsynet_identity_audit"
        generated = root / "generation" / "identity_manifest.json"
        inventory = root / "generation" / "contact_sheet_inventory.json"
        ledger = root / "review" / "review_ledger.json"
        evidence = root / "review" / "reviewer_evidence.json"
        adjudication = root / "review" / "cross_label_adjudication.json"
        output = root / "reviewed" / "identity_manifest.json"
        for path, payload in (
            (generated, fixture["generated"]),
            (inventory, fixture["inventory"]),
            (ledger, fixture["ledger"]),
            (evidence, fixture["evidence"]),
            (adjudication, fixture["adjudication"]),
        ):
            _write_private(path, payload)
        _write_contact_sheet_files(root, fixture)

        reviewed = finalize_identity_review(
            generated,
            inventory,
            ledger,
            evidence,
            adjudication,
            output,
        )
        first_bytes = output.read_bytes()
        c.eq(reviewed["schema_version"], "palsynet_identity_reviewed_v1")
        c.eq(stat.S_IMODE(output.stat().st_mode), 0o600,
             "reviewed manifest is owner-only")
        c.eq(stat.S_IMODE(output.parent.stat().st_mode), 0o700,
             "reviewed directory is owner-only")
        c.raises(lambda: finalize_identity_review(
            generated, inventory, ledger, evidence, adjudication, output
        ), FileExistsError, "reviewed publication never overwrites")
        c.eq(output.read_bytes(), first_bytes,
             "a repeated finalization preserves reviewed bytes")

        output.unlink()
        tampered_sheet = root / "generation" / fixture["inventory"]["pairs"][0][
            "relative_path"
        ]
        original_sheet = tampered_sheet.read_bytes()
        tampered_sheet.write_bytes(b"tampered contact sheet")
        tampered_sheet.chmod(0o600)
        c.raises(lambda: finalize_identity_review(
            generated, inventory, ledger, evidence, adjudication, output
        ), ValueError, "finalization rehashes every generated contact sheet")
        c.true(not output.exists(), "stale contact inventory fails before publication")
        tampered_sheet.write_bytes(original_sheet)
        tampered_sheet.chmod(0o600)

        wrong_stage = root / "generation" / "reviewed.json"
        c.raises(lambda: finalize_identity_review(
            generated, inventory, ledger, evidence, adjudication, wrong_stage
        ), ValueError, "reviewed output cannot alias the generation stage")

        ledger.chmod(0o640)
        c.raises(lambda: finalize_identity_review(
            generated, inventory, ledger, evidence, adjudication, output
        ), ValueError, "review inputs must be owner-only")
        c.true(not output.exists(), "bad permissions fail before publication")


def test_finalizer_cli_exposes_only_the_five_evidence_inputs_and_output(c: Check):
    destinations = {
        action.dest
        for action in finalizer_module._parser()._actions
        if action.dest != "help"
    }
    c.eq(destinations, {
        "generated_manifest",
        "contact_inventory",
        "review_ledger",
        "reviewer_evidence",
        "cross_label_adjudication",
        "output",
    }, "finalizer CLI has no eligibility flag or mutable protocol surface")


if __name__ == "__main__":
    run_all("test_palsynet_identity_review_ledger", dict(globals()))
