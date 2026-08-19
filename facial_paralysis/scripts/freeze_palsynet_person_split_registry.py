#!/usr/bin/env python3
"""Freeze the immutable person/group-disjoint split for PalsyNet.

The protected outer fold is selected from a semantic digest of the sorted
source members of each reviewed identity group.  Neither the private audit
salt nor reviewer-assigned opaque group IDs can influence the split.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.palsynet_identity_review import (  # noqa: E402
    CANONICAL_IDENTITY_AUDIT_ROOT,
    LABEL_COUNTS,
    PAIR_COUNT,
    RECORDING_COUNT,
    _project_path_without_symlinks,
    _read_private_json,
    canonical_json_bytes,
)


DOMAIN_SEPARATOR = "110d-generalization-v1-person-split"
OUTER_FOLDS = 5
INNER_FOLDS = 4
PROTECTED_OUTER_FOLD = 0
CANONICAL_OUTPUT = (
    CANONICAL_IDENTITY_AUDIT_ROOT / "person_split_registry.json"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RECORDING_ID = re.compile(r"^rec_[0-9a-f]{64}$")
_GROUP_ID = re.compile(r"^grp_[0-9a-f]{64}$")
_REVIEWED_FIELDS = {
    "schema_version", "dataset", "claim_unit", "identity_review", "counts",
    "fingerprints", "recordings",
}
_REVIEW_FIELDS = {
    "status", "label_blinded", "exhaustive_pair_review",
    "uncertainties_resolved",
}
_COUNT_FIELDS = {
    "total_recordings", "reviewed_groups", "eligible_recordings",
    "eligible_groups", "excluded_recordings", "excluded_groups",
}
_FINGERPRINT_FIELDS = {
    "source_collection_sha256", "generated_manifest_sha256",
    "contact_inventory_sha256", "review_ledger_sha256",
    "reviewer_evidence_sha256", "cross_label_adjudication_sha256",
}
_RECORD_FIELDS = {
    "recording_id", "group_id", "source_sha256", "source_label", "label",
    "identity_status", "claim_unit", "training_eligible",
    "adjudication_outcome", "adjudication_evidence_sha256",
}
_LEDGER_FIELDS = {
    "schema_version", "dataset", "source_collection_sha256",
    "generated_manifest_sha256", "contact_inventory_sha256",
    "reviewer_evidence_sha256", "label_blinded", "uncertainty_status",
    "recording_to_group", "pair_decisions",
}
_REGISTRY_FIELDS = {
    "schema_version", "dataset", "claim_unit", "identity_status",
    "source_collection_sha256", "reviewed_manifest_sha256",
    "review_ledger_sha256", "outer_fold_number", "protocol", "counts",
    "assignments",
}
_PROTOCOL = {
    "domain_separator": DOMAIN_SEPARATOR,
    "outer_folds": OUTER_FOLDS,
    "inner_folds": INNER_FOLDS,
    "semantic_group_key": (
        "sha256(domain_separator + ':' + comma_join(sorted_member_source_sha256))"
    ),
    "stratification": "binary_label_then_group_size_then_semantic_key",
}


def _exact(value: object, fields: set[str], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} fields differ from the closed schema")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _sequence(value: object, length: int, name: str) -> Sequence[object]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} rows")
    return value


def _payload_sha(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _validated_inputs(
    reviewed_manifest: object,
    review_ledger: object,
    *,
    reviewed_manifest_sha256: str,
    review_ledger_sha256: str,
) -> tuple[
    dict[str, Mapping[str, object]],
    dict[str, list[str]],
    dict[str, str],
    str,
]:
    reviewed_sha = _sha(reviewed_manifest_sha256, "reviewed manifest digest")
    ledger_sha = _sha(review_ledger_sha256, "review ledger digest")
    if _payload_sha(reviewed_manifest) != reviewed_sha:
        raise ValueError("reviewed manifest digest does not match canonical bytes")
    if _payload_sha(review_ledger) != ledger_sha:
        raise ValueError("review ledger digest does not match canonical bytes")

    reviewed = _exact(reviewed_manifest, _REVIEWED_FIELDS, "reviewed manifest")
    if (
        reviewed["schema_version"] != "palsynet_identity_reviewed_v1"
        or reviewed["dataset"] != "PalsyNet"
        or reviewed["claim_unit"] != "person_held_out"
    ):
        raise ValueError("reviewed identity claim is not eligible for person splits")
    review = _exact(reviewed["identity_review"], _REVIEW_FIELDS, "identity review")
    if review != {
        "status": "reviewed",
        "label_blinded": True,
        "exhaustive_pair_review": True,
        "uncertainties_resolved": True,
    }:
        raise ValueError("identity review is incomplete or not label-blinded")
    fingerprints = _exact(
        reviewed["fingerprints"], _FINGERPRINT_FIELDS, "reviewed fingerprints"
    )
    source_collection_sha = _sha(
        fingerprints["source_collection_sha256"], "source collection digest"
    )
    for field in _FINGERPRINT_FIELDS - {"source_collection_sha256"}:
        _sha(fingerprints[field], field)
    if fingerprints["review_ledger_sha256"] != ledger_sha:
        raise ValueError("reviewed manifest is not bound to the supplied ledger")

    ledger = _exact(review_ledger, _LEDGER_FIELDS, "review ledger")
    if (
        ledger["schema_version"] != "palsynet_identity_review_ledger_v1"
        or ledger["dataset"] != "PalsyNet"
        or ledger["source_collection_sha256"] != source_collection_sha
        or ledger["generated_manifest_sha256"]
        != fingerprints["generated_manifest_sha256"]
        or ledger["contact_inventory_sha256"]
        != fingerprints["contact_inventory_sha256"]
        or ledger["reviewer_evidence_sha256"]
        != fingerprints["reviewer_evidence_sha256"]
        or ledger["label_blinded"] is not True
        or ledger["uncertainty_status"] != "resolved"
    ):
        raise ValueError("review ledger provenance or review state drifted")

    records: dict[str, Mapping[str, object]] = {}
    groups: dict[str, list[str]] = {}
    labels: dict[str, str] = {}
    all_group_members: dict[str, list[str]] = {}
    source_digests: set[str] = set()
    observed_source_labels = {"affected": 0, "unaffected": 0}
    for row_value in _sequence(
        reviewed["recordings"], RECORDING_COUNT, "reviewed recordings"
    ):
        row = _exact(row_value, _RECORD_FIELDS, "reviewed recording")
        recording_id, group_id = row["recording_id"], row["group_id"]
        if (
            not isinstance(recording_id, str)
            or _RECORDING_ID.fullmatch(recording_id) is None
            or recording_id in records
            or not isinstance(group_id, str)
            or _GROUP_ID.fullmatch(group_id) is None
        ):
            raise ValueError("reviewed recording/group identifiers are invalid")
        source_digest = _sha(row["source_sha256"], "recording source digest")
        if source_digest in source_digests:
            raise ValueError("reviewed source digests must be unique")
        if (
            row["source_label"] not in {"affected", "unaffected"}
            or row["label"] not in {"affected", "unaffected"}
            or row["identity_status"] != "reviewed"
            or row["claim_unit"] != "person_held_out"
            or not isinstance(row["training_eligible"], bool)
            or row["adjudication_outcome"] not in {
                "none", "exclude_whole_group", "correct_proven_source_label_error"
            }
        ):
            raise ValueError("reviewed recording eligibility fields drifted")
        evidence = row["adjudication_evidence_sha256"]
        if row["adjudication_outcome"] == "none":
            if (
                evidence is not None
                or row["source_label"] != row["label"]
                or row["training_eligible"] is not True
            ):
                raise ValueError("unadjudicated rows cannot change labels")
        elif row["adjudication_outcome"] == "exclude_whole_group":
            if (
                row["training_eligible"] is not False
                or row["source_label"] != row["label"]
            ):
                raise ValueError("whole-group exclusion semantics drifted")
            _sha(evidence, "adjudication evidence digest")
        else:
            if row["training_eligible"] is not True:
                raise ValueError("corrected groups must remain wholly eligible")
            _sha(evidence, "adjudication evidence digest")
        records[recording_id] = row
        source_digests.add(source_digest)
        observed_source_labels[str(row["source_label"])] += 1
        all_group_members.setdefault(group_id, []).append(recording_id)
        if row["training_eligible"] is True:
            groups.setdefault(group_id, []).append(recording_id)
            previous = labels.setdefault(group_id, str(row["label"]))
            if previous != row["label"]:
                raise ValueError("an eligible reviewed group crosses binary labels")

    if observed_source_labels != LABEL_COUNTS:
        raise ValueError("reviewed source labels differ from the frozen source counts")
    recomputed_source_fingerprint = hashlib.sha256()
    for row in sorted(
        records.values(),
        key=lambda item: (item["source_label"], item["source_sha256"]),
    ):
        recomputed_source_fingerprint.update(
            f"{row['source_label']}:{row['source_sha256']}\n".encode("ascii")
        )
    if recomputed_source_fingerprint.hexdigest() != source_collection_sha:
        raise ValueError("source collection digest differs from reviewed source rows")

    # A reviewed identity group is either wholly eligible or wholly excluded,
    # and one finalizer decision applies identically to every group member.
    for group_id, members in all_group_members.items():
        states = {bool(records[recording_id]["training_eligible"])
                  for recording_id in members}
        if len(states) != 1:
            raise ValueError("reviewed identity groups cannot be partially eligible")
        outcomes = {str(records[recording_id]["adjudication_outcome"])
                    for recording_id in members}
        evidence_digests = {
            records[recording_id]["adjudication_evidence_sha256"]
            for recording_id in members
        }
        final_labels = {str(records[recording_id]["label"])
                        for recording_id in members}
        source_labels = {str(records[recording_id]["source_label"])
                         for recording_id in members}
        if len(outcomes) != 1 or len(evidence_digests) != 1:
            raise ValueError("adjudication outcome/evidence must be group-wide")
        outcome = next(iter(outcomes))
        if outcome == "none":
            if len(source_labels) != 1 or final_labels != source_labels:
                raise ValueError("cross-label groups require closed adjudication")
        elif outcome == "exclude_whole_group":
            if len(source_labels) < 2 or final_labels != source_labels:
                raise ValueError("whole-group exclusion must preserve cross-label sources")
        else:
            if (
                len(source_labels) < 2
                or len(final_labels) != 1
                or final_labels == source_labels
            ):
                raise ValueError("source-label correction semantics drifted")

    counts = _exact(reviewed["counts"], _COUNT_FIELDS, "reviewed counts")
    expected_counts = {
        "total_recordings": RECORDING_COUNT,
        "reviewed_groups": len(all_group_members),
        "eligible_recordings": sum(len(members) for members in groups.values()),
        "eligible_groups": len(groups),
        "excluded_recordings": RECORDING_COUNT - sum(
            len(members) for members in groups.values()
        ),
        "excluded_groups": len(all_group_members) - len(groups),
    }
    for field, expected in expected_counts.items():
        if _integer(counts[field], field) != expected:
            raise ValueError("reviewed manifest counts are incoherent")

    ledger_groups: dict[str, str] = {}
    for row_value in _sequence(
        ledger["recording_to_group"], RECORDING_COUNT, "ledger mapping"
    ):
        row = _exact(row_value, {"recording_id", "group_id"}, "ledger mapping row")
        recording_id, group_id = row["recording_id"], row["group_id"]
        if (
            not isinstance(recording_id, str)
            or recording_id not in records
            or recording_id in ledger_groups
            or group_id != records[recording_id]["group_id"]
        ):
            raise ValueError("review ledger mapping differs from reviewed manifest")
        ledger_groups[recording_id] = str(group_id)
    if set(ledger_groups) != set(records):
        raise ValueError("review ledger mapping coverage is incomplete")

    expected_pairs = {
        (first, second)
        for index, first in enumerate(sorted(records))
        for second in sorted(records)[index + 1:]
    }
    seen_pairs: set[tuple[str, str]] = set()
    for row_value in _sequence(
        ledger["pair_decisions"], PAIR_COUNT, "ledger pair decisions"
    ):
        row = _exact(
            row_value,
            {"recording_id_a", "recording_id_b", "decision"},
            "ledger pair decision",
        )
        first, second, decision = (
            row["recording_id_a"], row["recording_id_b"], row["decision"]
        )
        pair = (first, second)
        if pair not in expected_pairs or pair in seen_pairs:
            raise ValueError("review ledger pair coverage is invalid")
        same_group = records[first]["group_id"] == records[second]["group_id"]
        if decision not in {"same", "different"} or (decision == "same") != same_group:
            raise ValueError("review ledger pairs disagree with final identity groups")
        seen_pairs.add(pair)
    if seen_pairs != expected_pairs:
        raise ValueError("review ledger does not exhaust all recording pairs")
    return records, groups, labels, source_collection_sha


def _semantic_group_key(
    members: Sequence[str], records: Mapping[str, Mapping[str, object]]
) -> str:
    source_digests = sorted(str(records[member]["source_sha256"]) for member in members)
    material = f"{DOMAIN_SEPARATOR}:{','.join(source_digests)}".encode("ascii")
    return hashlib.sha256(material).hexdigest()


def _stratified_group_folds(
    group_ids: Sequence[str],
    groups: Mapping[str, Sequence[str]],
    labels: Mapping[str, str],
    semantic_keys: Mapping[str, str],
    folds: int,
) -> dict[str, int]:
    result: dict[str, int] = {}
    fold_record_counts = {
        label: [0] * folds for label in ("unaffected", "affected")
    }
    fold_group_counts = [0] * folds
    for label in ("unaffected", "affected"):
        class_groups = [group_id for group_id in group_ids if labels[group_id] == label]
        if len(class_groups) < folds:
            raise ValueError(f"binary class {label} has fewer than {folds} groups")
        ordered = sorted(
            class_groups,
            key=lambda group_id: (-len(groups[group_id]), semantic_keys[group_id]),
        )
        for group_id in ordered:
            target = min(
                range(folds),
                key=lambda fold: (
                    fold_record_counts[label][fold], fold_group_counts[fold], fold
                ),
            )
            result[group_id] = target
            fold_record_counts[label][target] += len(groups[group_id])
            fold_group_counts[target] += 1
    return result


def build_person_split_registry(
    reviewed_manifest: object,
    review_ledger: object,
    *,
    reviewed_manifest_sha256: str,
    review_ledger_sha256: str,
) -> dict[str, object]:
    """Build the one deterministic five-outer/four-inner split registry."""
    records, groups, labels, source_sha = _validated_inputs(
        reviewed_manifest,
        review_ledger,
        reviewed_manifest_sha256=reviewed_manifest_sha256,
        review_ledger_sha256=review_ledger_sha256,
    )
    semantic_keys = {
        group_id: _semantic_group_key(members, records)
        for group_id, members in groups.items()
    }
    group_ids = tuple(groups)
    outer_by_group = _stratified_group_folds(
        group_ids, groups, labels, semantic_keys, OUTER_FOLDS
    )
    development_groups = tuple(
        group_id for group_id in group_ids
        if outer_by_group[group_id] != PROTECTED_OUTER_FOLD
    )
    inner_by_group = _stratified_group_folds(
        development_groups, groups, labels, semantic_keys, INNER_FOLDS
    )

    assignments: list[dict[str, object]] = []
    for recording_id in sorted(records):
        row = records[recording_id]
        if row["training_eligible"] is not True:
            continue
        group_id = str(row["group_id"])
        outer_fold = outer_by_group[group_id]
        protected = outer_fold == PROTECTED_OUTER_FOLD
        assignments.append({
            "recording_id": recording_id,
            "group_id": group_id,
            "semantic_group_key_sha256": semantic_keys[group_id],
            "partition": "protected" if protected else "development",
            "outer_fold": outer_fold,
            "inner_fold": None if protected else inner_by_group[group_id],
        })

    protected_groups = {
        row["group_id"] for row in assignments if row["partition"] == "protected"
    }
    development_group_set = {
        row["group_id"] for row in assignments if row["partition"] == "development"
    }
    return {
        "schema_version": "palsynet_person_split_registry_v1",
        "dataset": "PalsyNet",
        "claim_unit": "person_held_out",
        "identity_status": "reviewed",
        "source_collection_sha256": source_sha,
        "reviewed_manifest_sha256": reviewed_manifest_sha256,
        "review_ledger_sha256": review_ledger_sha256,
        "outer_fold_number": PROTECTED_OUTER_FOLD,
        "protocol": dict(_PROTOCOL),
        "counts": {
            "eligible_recordings": len(assignments),
            "eligible_groups": len(groups),
            "development_recordings": sum(
                row["partition"] == "development" for row in assignments
            ),
            "development_groups": len(development_group_set),
            "protected_recordings": sum(
                row["partition"] == "protected" for row in assignments
            ),
            "protected_groups": len(protected_groups),
        },
        "assignments": assignments,
    }


def validate_person_split_registry(
    registry: object,
    reviewed_manifest: object,
    review_ledger: object,
) -> None:
    """Reject any registry other than the exact deterministic split."""
    supplied = _exact(registry, _REGISTRY_FIELDS, "person split registry")
    expected = build_person_split_registry(
        reviewed_manifest,
        review_ledger,
        reviewed_manifest_sha256=_sha(
            supplied["reviewed_manifest_sha256"], "reviewed manifest digest"
        ),
        review_ledger_sha256=_sha(
            supplied["review_ledger_sha256"], "review ledger digest"
        ),
    )
    if supplied != expected:
        raise ValueError("person split registry differs from the frozen semantic split")


def write_person_split_registry(path: str | Path, registry: object) -> None:
    """Write canonical owner-only bytes once without overwrite."""
    output = _project_path_without_symlinks(path)
    if output.exists() or output.is_symlink():
        raise FileExistsError("refusing to overwrite person split registry")
    try:
        parent_info = os.lstat(output.parent)
    except OSError as exc:
        raise ValueError("split registry parent must already exist") from exc
    if (
        stat.S_ISLNK(parent_info.st_mode)
        or not stat.S_ISDIR(parent_info.st_mode)
        or stat.S_IMODE(parent_info.st_mode) != 0o700
        or parent_info.st_uid != os.getuid()
    ):
        raise ValueError("split registry parent must be owner-only and real")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".person_split_registry.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(registry))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError("refusing to overwrite person split registry") from None
        os.chmod(output, 0o600, follow_symlinks=False)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_json(path: Path, name: str) -> tuple[dict[str, object], str]:
    guarded = _project_path_without_symlinks(path)
    payload, digest = _read_private_json(guarded, name)
    if _payload_sha(payload) != digest:
        raise ValueError(f"{name} must use canonical JSON bytes")
    return payload, digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-manifest", type=Path, required=True)
    parser.add_argument("--review-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    reviewed_path = _project_path_without_symlinks(args.reviewed_manifest)
    ledger_path = _project_path_without_symlinks(args.review_ledger)
    output = _project_path_without_symlinks(args.output)
    expected_reviewed = Path(os.path.abspath(
        CANONICAL_IDENTITY_AUDIT_ROOT / "reviewed" / "identity_manifest.json"
    ))
    expected_ledger = Path(os.path.abspath(
        CANONICAL_IDENTITY_AUDIT_ROOT / "review" / "review_ledger.json"
    ))
    expected_output = Path(os.path.abspath(CANONICAL_OUTPUT))
    if (
        reviewed_path != expected_reviewed
        or ledger_path != expected_ledger
        or output != expected_output
    ):
        raise ValueError("split freezer inputs/output must use the canonical lifecycle")
    reviewed, reviewed_sha = _read_json(reviewed_path, "reviewed manifest")
    ledger, ledger_sha = _read_json(ledger_path, "review ledger")
    registry = build_person_split_registry(
        reviewed,
        ledger,
        reviewed_manifest_sha256=reviewed_sha,
        review_ledger_sha256=ledger_sha,
    )
    write_person_split_registry(output, registry)


if __name__ == "__main__":
    main()


__all__ = [
    "DOMAIN_SEPARATOR",
    "build_person_split_registry",
    "canonical_json_bytes",
    "validate_person_split_registry",
    "write_person_split_registry",
]
