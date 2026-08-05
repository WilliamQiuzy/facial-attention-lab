"""Fail-closed contracts for exhaustive, label-blinded PalsyNet identity review."""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path


RECORDING_COUNT = 49
PAIR_COUNT = 1176
LABEL_COUNTS = {"affected": 27, "unaffected": 22}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_IDENTITY_AUDIT_ROOT = (
    PROJECT_ROOT / "outputs" / "palsynet_identity_audit"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RECORDING_ID = re.compile(r"^rec_[0-9a-f]{64}$")
_GROUP_ID = re.compile(r"^grp_[0-9a-f]{64}$")
_REVIEWER_ID = re.compile(r"^reviewer_[0-9a-f]{64}$")
_EVIDENCE_ID = re.compile(r"^evidence_[0-9a-f]{64}$")
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_EMBEDDED_EVIDENCE_BYTES = 1024 * 1024
_EVIDENCE_MEDIA_TYPES = {
    "application/json",
    "application/pdf",
    "image/jpeg",
    "image/png",
    "text/plain",
}


def canonical_json_bytes(payload: object) -> bytes:
    """Return the deterministic JSON representation used by generated artifacts."""
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _payload_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _require_exact_fields(
    value: object,
    expected: set[str],
    description: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{description} fields differ from the closed schema")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{description} keys must be strings")
    return value


def _require_count(value: object, expected: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"{field} must equal {expected}")
    return value


def _require_sequence(value: object, expected: int, field: str) -> Sequence[object]:
    if not isinstance(value, list) or len(value) != expected:
        raise ValueError(f"{field} must contain exactly {expected} rows")
    return value


def _validate_generated_manifest(
    payload: object,
) -> tuple[dict[str, Mapping[str, object]], tuple[tuple[str, str], ...], str]:
    manifest = _require_exact_fields(payload, {
        "schema_version",
        "dataset",
        "claim_unit",
        "identity_review",
        "counts",
        "fingerprints",
        "contact_sheet_sampling",
        "contact_sheets",
        "recordings",
        "ranked_pairs",
    }, "generated identity manifest")
    if (
        manifest["schema_version"] != "palsynet_identity_audit_v1"
        or manifest["dataset"] != "PalsyNet"
        or manifest["claim_unit"] != "video_held_out"
    ):
        raise ValueError("generated identity manifest is not the frozen unreviewed source")

    review = _require_exact_fields(manifest["identity_review"], {
        "status",
        "group_override_applied",
        "manual_review_required",
        "reviewer_evidence_sha256",
    }, "generated identity review")
    if review != {
        "status": "unreviewed",
        "group_override_applied": False,
        "manual_review_required": True,
        "reviewer_evidence_sha256": None,
    }:
        raise ValueError("generated identity manifest cannot already be reviewed")

    counts = _require_exact_fields(manifest["counts"], {
        "affected", "unaffected", "total", "ranked_pairs",
    }, "generated identity counts")
    _require_count(counts["affected"], 27, "affected count")
    _require_count(counts["unaffected"], 22, "unaffected count")
    _require_count(counts["total"], RECORDING_COUNT, "recording count")
    _require_count(counts["ranked_pairs"], PAIR_COUNT, "ranked pair count")

    fingerprints = _require_exact_fields(manifest["fingerprints"], {
        "source_collection_sha256",
        "bundle_provenance_sha256",
        "embedding_collection_sha256",
    }, "generated fingerprints")
    source_collection_sha256 = _require_sha256(
        fingerprints["source_collection_sha256"], "source collection digest"
    )
    _require_sha256(
        fingerprints["bundle_provenance_sha256"], "bundle provenance digest"
    )
    _require_sha256(
        fingerprints["embedding_collection_sha256"], "embedding collection digest"
    )

    sampling = _require_exact_fields(manifest["contact_sheet_sampling"], {
        "windows_per_video",
        "window_size_frames",
        "representative_frame_offset",
        "raw_filename_text_burned_in",
    }, "contact sheet sampling")
    if sampling != {
        "windows_per_video": 4,
        "window_size_frames": 32,
        "representative_frame_offset": 16,
        "raw_filename_text_burned_in": False,
    }:
        raise ValueError("contact sheet sampling differs from the frozen contract")
    sheets = _require_exact_fields(manifest["contact_sheets"], {
        "recordings", "ranked_pairs", "overview", "storage", "filenames",
    }, "generated contact sheets")
    _require_count(sheets["recordings"], RECORDING_COUNT, "recording sheet count")
    _require_count(sheets["ranked_pairs"], PAIR_COUNT, "pair sheet count")
    _require_count(sheets["overview"], 1, "overview sheet count")
    if (
        sheets["storage"] != "local_ignored_output"
        or sheets["filenames"] != "opaque_ids_or_ranks_only"
    ):
        raise ValueError("contact sheet storage contract drifted")

    recordings: dict[str, Mapping[str, object]] = {}
    source_digests: set[str] = set()
    observed_labels = {"affected": 0, "unaffected": 0}
    for row_value in _require_sequence(
        manifest["recordings"], RECORDING_COUNT, "generated recordings"
    ):
        row = _require_exact_fields(row_value, {
            "recording_id",
            "group_id",
            "label",
            "source_sha256",
            "identity_status",
            "claim_unit",
        }, "generated recording")
        recording_id = row["recording_id"]
        group_id = row["group_id"]
        label = row["label"]
        source_sha256 = row["source_sha256"]
        if not isinstance(recording_id, str) or _RECORDING_ID.fullmatch(recording_id) is None:
            raise ValueError("generated recording id is not canonical and opaque")
        if recording_id in recordings:
            raise ValueError("generated recording ids must be unique")
        if not isinstance(group_id, str) or _GROUP_ID.fullmatch(group_id) is None:
            raise ValueError("generated group id is not canonical and opaque")
        if label not in LABEL_COUNTS:
            raise ValueError("generated recording label is invalid")
        source_digest = _require_sha256(source_sha256, "recording source digest")
        if source_digest in source_digests:
            raise ValueError("generated source digests must be unique")
        if row["identity_status"] != "unreviewed" or row["claim_unit"] != "video_held_out":
            raise ValueError("generated recording cannot contain a reviewed claim")
        recordings[recording_id] = row
        source_digests.add(source_digest)
        observed_labels[str(label)] += 1
    if observed_labels != LABEL_COUNTS:
        raise ValueError("generated recording labels differ from frozen counts")
    recomputed_source_fingerprint = hashlib.sha256()
    for row in sorted(
        recordings.values(),
        key=lambda item: (item["label"], item["source_sha256"]),
    ):
        recomputed_source_fingerprint.update(
            f"{row['label']}:{row['source_sha256']}\n".encode("ascii")
        )
    if recomputed_source_fingerprint.hexdigest() != source_collection_sha256:
        raise ValueError("source collection digest differs from exact recording rows")

    expected_pairs = set(itertools_combinations(sorted(recordings)))
    observed_pairs: set[tuple[str, str]] = set()
    ranked_pairs: list[tuple[str, str]] = []
    for rank, row_value in enumerate(
        _require_sequence(manifest["ranked_pairs"], PAIR_COUNT, "ranked pairs"), 1
    ):
        row = _require_exact_fields(row_value, {
            "rank", "recording_id_a", "recording_id_b", "cosine",
        }, "ranked pair")
        _require_count(row["rank"], rank, "pair rank")
        first, second = row["recording_id_a"], row["recording_id_b"]
        if (
            not isinstance(first, str)
            or not isinstance(second, str)
            or first not in recordings
            or second not in recordings
            or first >= second
        ):
            raise ValueError("ranked pair must contain one canonical unordered pair")
        pair = (first, second)
        if pair in observed_pairs:
            raise ValueError("ranked pairs contain a duplicate unordered pair")
        cosine = row["cosine"]
        if isinstance(cosine, bool) or not isinstance(cosine, (int, float)):
            raise ValueError("pair cosine must be numeric")
        cosine = float(cosine)
        if not math.isfinite(cosine) or not -1.0 <= cosine <= 1.0:
            raise ValueError("pair cosine must be finite and bounded")
        observed_pairs.add(pair)
        ranked_pairs.append(pair)
    if observed_pairs != expected_pairs:
        raise ValueError("ranked pairs do not exhaust the recording set")
    return recordings, tuple(ranked_pairs), source_collection_sha256


def itertools_combinations(values: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """Small local wrapper keeps the exact unordered-pair contract explicit."""
    return tuple(
        (values[first], values[second])
        for first in range(len(values))
        for second in range(first + 1, len(values))
    )


def build_contact_sheet_inventory(
    generated_manifest: Mapping[str, object],
    *,
    generated_manifest_sha256: str,
    overview_sha256: str,
    recording_sheet_sha256: Mapping[str, str],
    pair_sheet_sha256: Mapping[tuple[str, str], str],
) -> dict[str, object]:
    """Build the label-blinded inventory for all recording and pair sheets."""
    recordings, ranked_pairs, source_sha256 = _validate_generated_manifest(
        generated_manifest
    )
    generated_digest = _require_sha256(
        generated_manifest_sha256, "generated manifest digest"
    )
    overview_digest = _require_sha256(overview_sha256, "overview sheet digest")
    if set(recording_sheet_sha256) != set(recordings):
        raise ValueError("recording contact-sheet digest coverage is incomplete")
    if set(pair_sheet_sha256) != set(ranked_pairs):
        raise ValueError("pair contact-sheet digest coverage is incomplete")
    recording_rows = [
        {
            "recording_id": recording_id,
            "relative_path": f"contact_sheets/recordings/{recording_id}.jpg",
            "sha256": _require_sha256(
                recording_sheet_sha256[recording_id], "recording sheet digest"
            ),
        }
        for recording_id in sorted(recordings)
    ]
    pair_rows = [
        {
            "rank": rank,
            "recording_id_a": first,
            "recording_id_b": second,
            "relative_path": f"contact_sheets/pairs/pair_{rank:04d}.jpg",
            "sha256": _require_sha256(
                pair_sheet_sha256[(first, second)], "pair sheet digest"
            ),
        }
        for rank, (first, second) in enumerate(ranked_pairs, 1)
    ]
    return {
        "schema_version": "palsynet_identity_contact_inventory_v1",
        "dataset": "PalsyNet",
        "source_collection_sha256": source_sha256,
        "generated_manifest_sha256": generated_digest,
        "label_blinded": True,
        "overview": {
            "relative_path": "contact_sheets/overview.jpg",
            "sha256": overview_digest,
        },
        "recordings": recording_rows,
        "pairs": pair_rows,
    }


def _validate_contact_inventory(
    payload: object,
    recordings: Mapping[str, Mapping[str, object]],
    ranked_pairs: Sequence[tuple[str, str]],
    source_collection_sha256: str,
    generated_manifest_sha256: str,
) -> None:
    inventory = _require_exact_fields(payload, {
        "schema_version",
        "dataset",
        "source_collection_sha256",
        "generated_manifest_sha256",
        "label_blinded",
        "overview",
        "recordings",
        "pairs",
    }, "contact sheet inventory")
    if (
        inventory["schema_version"] != "palsynet_identity_contact_inventory_v1"
        or inventory["dataset"] != "PalsyNet"
        or inventory["source_collection_sha256"] != source_collection_sha256
        or inventory["generated_manifest_sha256"] != generated_manifest_sha256
        or inventory["label_blinded"] is not True
    ):
        raise ValueError("contact sheet inventory provenance or label blindness drifted")
    overview = _require_exact_fields(
        inventory["overview"], {"relative_path", "sha256"}, "overview inventory"
    )
    if overview["relative_path"] != "contact_sheets/overview.jpg":
        raise ValueError("overview contact-sheet path drifted")
    _require_sha256(overview["sha256"], "overview digest")

    observed_recordings: set[str] = set()
    for row_value in _require_sequence(
        inventory["recordings"], RECORDING_COUNT, "recording sheet inventory"
    ):
        row = _require_exact_fields(
            row_value, {"recording_id", "relative_path", "sha256"},
            "recording sheet inventory row",
        )
        recording_id = row["recording_id"]
        if recording_id not in recordings or recording_id in observed_recordings:
            raise ValueError("recording sheet inventory coverage is invalid")
        if row["relative_path"] != f"contact_sheets/recordings/{recording_id}.jpg":
            raise ValueError("recording sheet path drifted")
        _require_sha256(row["sha256"], "recording sheet digest")
        observed_recordings.add(str(recording_id))
    if observed_recordings != set(recordings):
        raise ValueError("recording sheet inventory is incomplete")

    observed_pairs: set[tuple[str, str]] = set()
    for rank, row_value in enumerate(
        _require_sequence(inventory["pairs"], PAIR_COUNT, "pair sheet inventory"), 1
    ):
        row = _require_exact_fields(row_value, {
            "rank", "recording_id_a", "recording_id_b", "relative_path", "sha256",
        }, "pair sheet inventory row")
        _require_count(row["rank"], rank, "pair inventory rank")
        pair = (row["recording_id_a"], row["recording_id_b"])
        if pair != ranked_pairs[rank - 1] or pair in observed_pairs:
            raise ValueError("pair sheet inventory is not aligned to generated pairs")
        if row["relative_path"] != f"contact_sheets/pairs/pair_{rank:04d}.jpg":
            raise ValueError("pair sheet path drifted")
        _require_sha256(row["sha256"], "pair sheet digest")
        observed_pairs.add(pair)
    if observed_pairs != set(ranked_pairs):
        raise ValueError("pair sheet inventory is incomplete")


def build_review_ledger_template(
    generated_manifest: Mapping[str, object],
    *,
    generated_manifest_sha256: str,
    contact_inventory_sha256: str,
) -> dict[str, object]:
    """Create a blank, label-free template covering all 49 recordings and pairs."""
    recordings, ranked_pairs, source_sha256 = _validate_generated_manifest(
        generated_manifest
    )
    return {
        "schema_version": "palsynet_identity_review_ledger_template_v1",
        "dataset": "PalsyNet",
        "source_collection_sha256": source_sha256,
        "generated_manifest_sha256": _require_sha256(
            generated_manifest_sha256, "generated manifest digest"
        ),
        "contact_inventory_sha256": _require_sha256(
            contact_inventory_sha256, "contact inventory digest"
        ),
        "reviewer_evidence_sha256": None,
        "label_blinded": True,
        "uncertainty_status": "unresolved",
        "recording_to_group": [
            {"recording_id": recording_id, "group_id": None}
            for recording_id in sorted(recordings)
        ],
        "pair_decisions": [
            {
                "recording_id_a": first,
                "recording_id_b": second,
                "decision": None,
            }
            for first, second in ranked_pairs
        ],
    }


def _validate_reviewer_evidence(payload: object) -> None:
    evidence = _require_exact_fields(payload, {
        "schema_version",
        "dataset",
        "review_protocol",
        "reviewer_id",
        "label_accessed_during_identity_review",
        "recording_overviews_reviewed",
        "pair_decisions_reviewed",
        "uncertainties_resolved",
    }, "reviewer evidence")
    if (
        evidence["schema_version"] != "palsynet_identity_reviewer_evidence_v1"
        or evidence["dataset"] != "PalsyNet"
        or evidence["review_protocol"]
        != "label_blinded_exhaustive_pair_review_v1"
        or not isinstance(evidence["reviewer_id"], str)
        or _REVIEWER_ID.fullmatch(evidence["reviewer_id"]) is None
        or evidence["label_accessed_during_identity_review"] is not False
        or evidence["uncertainties_resolved"] is not True
    ):
        raise ValueError("reviewer evidence does not attest a label-blinded review")
    _require_count(
        evidence["recording_overviews_reviewed"], RECORDING_COUNT,
        "reviewed overview count",
    )
    _require_count(
        evidence["pair_decisions_reviewed"], PAIR_COUNT,
        "reviewed pair count",
    )


def _validate_review_ledger(
    payload: object,
    recordings: Mapping[str, Mapping[str, object]],
    ranked_pairs: Sequence[tuple[str, str]],
    *,
    source_collection_sha256: str,
    generated_manifest_sha256: str,
    contact_inventory_sha256: str,
    reviewer_evidence_sha256: str,
) -> dict[str, str]:
    ledger = _require_exact_fields(payload, {
        "schema_version",
        "dataset",
        "source_collection_sha256",
        "generated_manifest_sha256",
        "contact_inventory_sha256",
        "reviewer_evidence_sha256",
        "label_blinded",
        "uncertainty_status",
        "recording_to_group",
        "pair_decisions",
    }, "review ledger")
    if (
        ledger["schema_version"] != "palsynet_identity_review_ledger_v1"
        or ledger["dataset"] != "PalsyNet"
        or ledger["source_collection_sha256"] != source_collection_sha256
        or ledger["generated_manifest_sha256"] != generated_manifest_sha256
        or ledger["contact_inventory_sha256"] != contact_inventory_sha256
        or ledger["reviewer_evidence_sha256"] != reviewer_evidence_sha256
        or ledger["label_blinded"] is not True
        or ledger["uncertainty_status"] != "resolved"
    ):
        raise ValueError("review ledger provenance or label-blind attestation drifted")

    groups: dict[str, str] = {}
    for row_value in _require_sequence(
        ledger["recording_to_group"], RECORDING_COUNT, "recording-to-group mapping"
    ):
        row = _require_exact_fields(
            row_value, {"recording_id", "group_id"}, "recording-to-group row"
        )
        recording_id, group_id = row["recording_id"], row["group_id"]
        if (
            not isinstance(recording_id, str)
            or recording_id not in recordings
            or recording_id in groups
        ):
            raise ValueError("recording-to-group mapping is incomplete or duplicated")
        if not isinstance(group_id, str) or _GROUP_ID.fullmatch(group_id) is None:
            raise ValueError("reviewed group id must be canonical and opaque")
        groups[recording_id] = group_id
    if set(groups) != set(recordings):
        raise ValueError("recording-to-group mapping coverage is incomplete")

    decisions: dict[tuple[str, str], str] = {}
    for row_value in _require_sequence(
        ledger["pair_decisions"], PAIR_COUNT, "pair decisions"
    ):
        row = _require_exact_fields(
            row_value, {"recording_id_a", "recording_id_b", "decision"},
            "pair decision",
        )
        first, second, decision = (
            row["recording_id_a"], row["recording_id_b"], row["decision"]
        )
        if (
            not isinstance(first, str)
            or not isinstance(second, str)
            or first not in recordings
            or second not in recordings
            or first >= second
        ):
            raise ValueError("pair decision must use one canonical unordered pair")
        pair = (first, second)
        if pair in decisions:
            raise ValueError("pair decisions contain a duplicate or reversed pair")
        if decision not in {"same", "different"}:
            raise ValueError("pair decision must be same or different")
        decisions[pair] = str(decision)
    if set(decisions) != set(ranked_pairs):
        raise ValueError("pair decisions do not exhaust all 1176 unordered pairs")
    for pair, decision in decisions.items():
        same_group = groups[pair[0]] == groups[pair[1]]
        if (decision == "same") != same_group:
            raise ValueError("pair decisions and final groups are not one equivalence relation")
    return groups


def _validate_embedded_document(
    payload: object,
    *,
    minimum_bytes: int = 1,
) -> str:
    evidence = _require_exact_fields(payload, {
        "encoding", "media_type", "payload_base64", "sha256",
    }, "documented adjudication evidence")
    if (
        evidence["encoding"] != "base64"
        or evidence["media_type"] not in _EVIDENCE_MEDIA_TYPES
        or not isinstance(evidence["payload_base64"], str)
    ):
        raise ValueError("documented adjudication evidence metadata is invalid")
    try:
        evidence_bytes = base64.b64decode(
            evidence["payload_base64"], validate=True
        )
    except (ValueError, binascii.Error) as exc:
        raise ValueError(
            "documented adjudication evidence is not canonical base64"
        ) from exc
    if (
        len(evidence_bytes) < minimum_bytes
        or len(evidence_bytes) > _MAX_EMBEDDED_EVIDENCE_BYTES
        or base64.b64encode(evidence_bytes).decode("ascii")
        != evidence["payload_base64"]
    ):
        raise ValueError("documented adjudication evidence bytes are invalid")
    evidence_sha256 = _require_sha256(
        evidence["sha256"], "adjudication evidence digest"
    )
    if hashlib.sha256(evidence_bytes).hexdigest() != evidence_sha256:
        raise ValueError("adjudication evidence digest does not match exact bytes")
    return evidence_sha256


def _validate_correction_evidence(
    payload: object,
    *,
    group_id: str,
    group_members: set[str],
    recordings: Mapping[str, Mapping[str, object]],
    corrected_label: str,
) -> str:
    evidence = _require_exact_fields(payload, {
        "schema_version", "group_id", "claims", "rationale", "provenance",
    }, "source-label correction evidence")
    if (
        evidence["schema_version"]
        != "palsynet_source_label_correction_evidence_v1"
        or evidence["group_id"] != group_id
    ):
        raise ValueError("correction evidence is not bound to the exact group")

    ordered_members = sorted(group_members)
    claims = _require_sequence(
        evidence["claims"], len(ordered_members), "source-label correction claims"
    )
    changed_labels = 0
    for expected_recording_id, claim_value in zip(ordered_members, claims):
        claim = _require_exact_fields(claim_value, {
            "recording_id", "source_sha256", "old_label", "new_label",
        }, "source-label correction claim")
        source = recordings[expected_recording_id]
        if (
            claim["recording_id"] != expected_recording_id
            or claim["source_sha256"] != source["source_sha256"]
            or claim["old_label"] != source["label"]
            or claim["new_label"] != corrected_label
        ):
            raise ValueError("correction evidence contains a stale source-label claim")
        changed_labels += claim["old_label"] != claim["new_label"]
    if changed_labels < 1:
        raise ValueError("correction evidence must prove at least one label change")

    rationale = _require_exact_fields(
        evidence["rationale"], {"finding", "basis"}, "correction rationale"
    )
    allowed_bases = {
        "authoritative_source_record",
        "dataset_steward_confirmation",
    }
    if (
        rationale["finding"] != "verified_source_annotation_error"
        or rationale["basis"] not in allowed_bases
    ):
        raise ValueError("correction rationale is not an allowed verified finding")

    provenance = _require_exact_fields(evidence["provenance"], {
        "source", "reference_id", "claims_sha256", "document",
    }, "correction evidence provenance")
    if (
        provenance["source"] != rationale["basis"]
        or not isinstance(provenance["reference_id"], str)
        or _EVIDENCE_ID.fullmatch(provenance["reference_id"]) is None
        or _require_sha256(
            provenance["claims_sha256"], "correction claims digest"
        ) != _payload_sha256(claims)
    ):
        raise ValueError("correction evidence provenance is stale or invalid")
    return _validate_embedded_document(
        provenance["document"], minimum_bytes=16
    )


def _validate_adjudication(
    payload: object,
    recordings: Mapping[str, Mapping[str, object]],
    groups: Mapping[str, str],
    *,
    source_collection_sha256: str,
    review_ledger_sha256: str,
    reviewer_evidence_sha256: str,
) -> dict[str, Mapping[str, object]]:
    adjudication = _require_exact_fields(payload, {
        "schema_version",
        "dataset",
        "source_collection_sha256",
        "review_ledger_sha256",
        "reviewer_evidence_sha256",
        "decisions",
    }, "cross-label adjudication")
    if (
        adjudication["schema_version"] != "palsynet_cross_label_adjudication_v1"
        or adjudication["dataset"] != "PalsyNet"
        or adjudication["source_collection_sha256"] != source_collection_sha256
        or adjudication["review_ledger_sha256"] != review_ledger_sha256
        or adjudication["reviewer_evidence_sha256"] != reviewer_evidence_sha256
    ):
        raise ValueError("cross-label adjudication digest binding drifted")

    labels_by_group: dict[str, set[str]] = {}
    members_by_group: dict[str, set[str]] = {}
    for recording_id, group_id in groups.items():
        members_by_group.setdefault(group_id, set()).add(recording_id)
        labels_by_group.setdefault(group_id, set()).add(
            str(recordings[recording_id]["label"])
        )
    cross_label_groups = {
        group_id for group_id, labels in labels_by_group.items() if len(labels) > 1
    }
    decisions_value = adjudication["decisions"]
    if not isinstance(decisions_value, list):
        raise ValueError("cross-label adjudication decisions must be a list")
    decisions: dict[str, Mapping[str, object]] = {}
    for row_value in decisions_value:
        row = _require_exact_fields(row_value, {
            "group_id", "outcome", "evidence", "corrected_label",
        }, "cross-label adjudication decision")
        group_id = row["group_id"]
        if (
            not isinstance(group_id, str)
            or _GROUP_ID.fullmatch(group_id) is None
            or group_id in decisions
            or group_id not in cross_label_groups
        ):
            raise ValueError("adjudication must cover each cross-label group exactly once")
        outcome = row["outcome"]
        corrected_label = row["corrected_label"]
        if outcome == "exclude_whole_group":
            if corrected_label is not None:
                raise ValueError("whole-group exclusion cannot provide a corrected label")
            evidence_sha256 = _validate_embedded_document(row["evidence"])
        elif outcome == "correct_proven_source_label_error":
            if corrected_label not in LABEL_COUNTS:
                raise ValueError("source-label correction requires one binary label")
            evidence_sha256 = _validate_correction_evidence(
                row["evidence"],
                group_id=group_id,
                group_members=members_by_group[group_id],
                recordings=recordings,
                corrected_label=str(corrected_label),
            )
        else:
            raise ValueError("cross-label adjudication outcome is not allowed")
        decisions[group_id] = {**row, "_evidence_sha256": evidence_sha256}
    if set(decisions) != cross_label_groups:
        raise ValueError("every cross-label group requires closed adjudication")
    return decisions


def _build_reviewed_identity_manifest(
    generated_manifest: Mapping[str, object],
    contact_inventory: Mapping[str, object],
    review_ledger: Mapping[str, object],
    reviewer_evidence: Mapping[str, object],
    cross_label_adjudication: Mapping[str, object],
    *,
    generated_manifest_sha256: str,
    contact_inventory_sha256: str,
    review_ledger_sha256: str,
    reviewer_evidence_sha256: str,
    cross_label_adjudication_sha256: str,
    verify_canonical_digests: bool,
) -> dict[str, object]:
    digests = {
        "generated_manifest_sha256": _require_sha256(
            generated_manifest_sha256, "generated manifest digest"
        ),
        "contact_inventory_sha256": _require_sha256(
            contact_inventory_sha256, "contact inventory digest"
        ),
        "review_ledger_sha256": _require_sha256(
            review_ledger_sha256, "review ledger digest"
        ),
        "reviewer_evidence_sha256": _require_sha256(
            reviewer_evidence_sha256, "reviewer evidence digest"
        ),
        "cross_label_adjudication_sha256": _require_sha256(
            cross_label_adjudication_sha256, "cross-label adjudication digest"
        ),
    }
    if verify_canonical_digests:
        payloads = {
            "generated_manifest_sha256": generated_manifest,
            "contact_inventory_sha256": contact_inventory,
            "review_ledger_sha256": review_ledger,
            "reviewer_evidence_sha256": reviewer_evidence,
            "cross_label_adjudication_sha256": cross_label_adjudication,
        }
        for field, payload in payloads.items():
            if _payload_sha256(payload) != digests[field]:
                raise ValueError(f"{field} does not match exact canonical bytes")

    recordings, ranked_pairs, source_sha256 = _validate_generated_manifest(
        generated_manifest
    )
    _validate_contact_inventory(
        contact_inventory,
        recordings,
        ranked_pairs,
        source_sha256,
        digests["generated_manifest_sha256"],
    )
    _validate_reviewer_evidence(reviewer_evidence)
    groups = _validate_review_ledger(
        review_ledger,
        recordings,
        ranked_pairs,
        source_collection_sha256=source_sha256,
        generated_manifest_sha256=digests["generated_manifest_sha256"],
        contact_inventory_sha256=digests["contact_inventory_sha256"],
        reviewer_evidence_sha256=digests["reviewer_evidence_sha256"],
    )
    decisions = _validate_adjudication(
        cross_label_adjudication,
        recordings,
        groups,
        source_collection_sha256=source_sha256,
        review_ledger_sha256=digests["review_ledger_sha256"],
        reviewer_evidence_sha256=digests["reviewer_evidence_sha256"],
    )

    rows: list[dict[str, object]] = []
    for recording_id in sorted(recordings):
        source = recordings[recording_id]
        group_id = groups[recording_id]
        decision = decisions.get(group_id)
        source_label = str(source["label"])
        if decision is None:
            training_eligible = True
            final_label = source_label
            outcome = "none"
            evidence_sha256 = None
        elif decision["outcome"] == "exclude_whole_group":
            training_eligible = False
            final_label = source_label
            outcome = "exclude_whole_group"
            evidence_sha256 = str(decision["_evidence_sha256"])
        else:
            training_eligible = True
            final_label = str(decision["corrected_label"])
            outcome = "correct_proven_source_label_error"
            evidence_sha256 = str(decision["_evidence_sha256"])
        rows.append({
            "recording_id": recording_id,
            "group_id": group_id,
            "source_sha256": source["source_sha256"],
            "source_label": source_label,
            "label": final_label,
            "identity_status": "reviewed",
            "claim_unit": "person_held_out",
            "training_eligible": training_eligible,
            "adjudication_outcome": outcome,
            "adjudication_evidence_sha256": evidence_sha256,
        })

    all_groups = set(groups.values())
    eligible_groups = {
        row["group_id"] for row in rows if row["training_eligible"] is True
    }
    return {
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
            "total_recordings": RECORDING_COUNT,
            "reviewed_groups": len(all_groups),
            "eligible_recordings": sum(
                row["training_eligible"] is True for row in rows
            ),
            "eligible_groups": len(eligible_groups),
            "excluded_recordings": sum(
                row["training_eligible"] is False for row in rows
            ),
            "excluded_groups": len(all_groups - eligible_groups),
        },
        "fingerprints": {
            "source_collection_sha256": source_sha256,
            **digests,
        },
        "recordings": rows,
    }


def build_reviewed_identity_manifest(
    generated_manifest: Mapping[str, object],
    contact_inventory: Mapping[str, object],
    review_ledger: Mapping[str, object],
    reviewer_evidence: Mapping[str, object],
    cross_label_adjudication: Mapping[str, object],
    *,
    generated_manifest_sha256: str,
    contact_inventory_sha256: str,
    review_ledger_sha256: str,
    reviewer_evidence_sha256: str,
    cross_label_adjudication_sha256: str,
) -> dict[str, object]:
    """Validate canonical in-memory artifacts and build the reviewed projection."""
    return _build_reviewed_identity_manifest(
        generated_manifest,
        contact_inventory,
        review_ledger,
        reviewer_evidence,
        cross_label_adjudication,
        generated_manifest_sha256=generated_manifest_sha256,
        contact_inventory_sha256=contact_inventory_sha256,
        review_ledger_sha256=review_ledger_sha256,
        reviewer_evidence_sha256=reviewer_evidence_sha256,
        cross_label_adjudication_sha256=cross_label_adjudication_sha256,
        verify_canonical_digests=True,
    )


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("identity review artifact contains a duplicate JSON key")
        result[key] = value
    return result


def _read_private_json(path: Path, description: str) -> tuple[dict[str, object], str]:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"cannot open {description}") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_uid != os.getuid()
        or info.st_nlink != 1
        or info.st_size < 2
        or info.st_size > _MAX_JSON_BYTES
    ):
        raise ValueError(f"{description} must be one owner-only regular file")
    parent_info = os.lstat(path.parent)
    if (
        stat.S_ISLNK(parent_info.st_mode)
        or not stat.S_ISDIR(parent_info.st_mode)
        or stat.S_IMODE(parent_info.st_mode) != 0o700
        or parent_info.st_uid != os.getuid()
    ):
        raise ValueError(f"{description} parent must be an owner-only directory")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        held = os.fstat(fd)
        if (held.st_dev, held.st_ino) != (info.st_dev, info.st_ino):
            raise ValueError(f"{description} changed before it was held")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(_MAX_JSON_BYTES + 1)
        after = os.fstat(fd)
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
            held.st_dev, held.st_ino, held.st_size, held.st_mtime_ns
        ):
            raise ValueError(f"{description} changed while it was read")
    finally:
        os.close(fd)
    if len(raw) > _MAX_JSON_BYTES:
        raise ValueError(f"{description} exceeds the size limit")
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_json_object)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{description} is not valid unique-key JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain one JSON object")
    return payload, hashlib.sha256(raw).hexdigest()


def _sha256_owner_only_file(path: Path, description: str) -> str:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"cannot open {description}") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_uid != os.getuid()
        or info.st_nlink != 1
        or info.st_size < 1
    ):
        raise ValueError(f"{description} must be one owner-only regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        held = os.fstat(descriptor)
        if (held.st_dev, held.st_ino) != (info.st_dev, info.st_ino):
            raise ValueError(f"{description} changed before it was held")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
            held.st_dev, held.st_ino, held.st_size, held.st_mtime_ns
        ):
            raise ValueError(f"{description} changed while it was read")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _verify_contact_sheet_files(
    generation_root: Path,
    inventory: Mapping[str, object],
) -> None:
    for directory in (
        generation_root,
        generation_root / "contact_sheets",
        generation_root / "contact_sheets" / "recordings",
        generation_root / "contact_sheets" / "pairs",
    ):
        try:
            info = os.lstat(directory)
        except OSError as exc:
            raise ValueError("contact sheet directory is missing") from exc
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o700
            or info.st_uid != os.getuid()
        ):
            raise ValueError("contact sheet directories must be owner-only")

    rows: list[Mapping[str, object]] = [
        inventory["overview"],
        *inventory["recordings"],
        *inventory["pairs"],
    ]
    for row in rows:
        relative_path = Path(str(row["relative_path"]))
        artifact = generation_root / relative_path
        observed_sha256 = _sha256_owner_only_file(
            artifact, "generated contact sheet"
        )
        if observed_sha256 != row["sha256"]:
            raise ValueError("contact sheet digest differs from the bound inventory")


def _validate_lifecycle_paths(
    generated_manifest: Path,
    contact_inventory: Path,
    review_ledger: Path,
    reviewer_evidence: Path,
    cross_label_adjudication: Path,
    output: Path,
) -> Path:
    paths = [
        generated_manifest,
        contact_inventory,
        review_ledger,
        reviewer_evidence,
        cross_label_adjudication,
        output,
    ]
    absolute = [Path(os.path.abspath(path)) for path in paths]
    generated_manifest, contact_inventory, review_ledger, reviewer_evidence, (
        cross_label_adjudication
    ), output = absolute
    expected_names = (
        "identity_manifest.json",
        "contact_sheet_inventory.json",
        "review_ledger.json",
        "reviewer_evidence.json",
        "cross_label_adjudication.json",
        "identity_manifest.json",
    )
    if tuple(path.name for path in absolute) != expected_names:
        raise ValueError("identity review artifact filenames differ from the lifecycle contract")
    root = generated_manifest.parent.parent
    canonical_root = Path(os.path.abspath(CANONICAL_IDENTITY_AUDIT_ROOT))
    if (
        generated_manifest.parent != root / "generation"
        or contact_inventory.parent != root / "generation"
        or review_ledger.parent != root / "review"
        or reviewer_evidence.parent != root / "review"
        or cross_label_adjudication.parent != root / "review"
        or output.parent != root / "reviewed"
    ):
        raise ValueError("identity review artifacts must use generation/review/reviewed stages")
    if root != canonical_root:
        raise ValueError("identity review lifecycle root must be canonical and ignored")
    if len(set(absolute)) != len(absolute):
        raise ValueError("identity review artifact paths cannot alias")
    return root


def _write_private_exclusive(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite reviewed identity manifest")
    parent = path.parent
    root = parent.parent
    if not root.is_dir() or root.is_symlink():
        raise ValueError("identity review root must already be a real directory")
    try:
        os.mkdir(parent, 0o700)
    except FileExistsError:
        info = os.lstat(parent)
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o700
            or info.st_uid != os.getuid()
        ):
            raise ValueError("reviewed output parent must be owner-only")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".identity_manifest.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(
                "refusing to overwrite reviewed identity manifest"
            ) from None
        os.chmod(path, 0o600, follow_symlinks=False)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def finalize_identity_review(
    generated_manifest_path: str | Path,
    contact_inventory_path: str | Path,
    review_ledger_path: str | Path,
    reviewer_evidence_path: str | Path,
    cross_label_adjudication_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Authenticate the immutable review stages and publish one reviewed manifest."""
    generated_path = Path(generated_manifest_path)
    inventory_path = Path(contact_inventory_path)
    ledger_path = Path(review_ledger_path)
    evidence_path = Path(reviewer_evidence_path)
    adjudication_path = Path(cross_label_adjudication_path)
    output = Path(output_path)
    _validate_lifecycle_paths(
        generated_path,
        inventory_path,
        ledger_path,
        evidence_path,
        adjudication_path,
        output,
    )
    if output.exists() or output.is_symlink():
        raise FileExistsError("refusing to overwrite reviewed identity manifest")
    generated, generated_sha = _read_private_json(
        generated_path, "generated identity manifest"
    )
    inventory, inventory_sha = _read_private_json(
        inventory_path, "contact sheet inventory"
    )
    ledger, ledger_sha = _read_private_json(ledger_path, "review ledger")
    evidence, evidence_sha = _read_private_json(
        evidence_path, "reviewer evidence"
    )
    adjudication, adjudication_sha = _read_private_json(
        adjudication_path, "cross-label adjudication"
    )
    reviewed = _build_reviewed_identity_manifest(
        generated,
        inventory,
        ledger,
        evidence,
        adjudication,
        generated_manifest_sha256=generated_sha,
        contact_inventory_sha256=inventory_sha,
        review_ledger_sha256=ledger_sha,
        reviewer_evidence_sha256=evidence_sha,
        cross_label_adjudication_sha256=adjudication_sha,
        verify_canonical_digests=False,
    )
    _verify_contact_sheet_files(generated_path.parent, inventory)
    _write_private_exclusive(output, canonical_json_bytes(reviewed))
    return reviewed


__all__ = [
    "CANONICAL_IDENTITY_AUDIT_ROOT",
    "PAIR_COUNT",
    "RECORDING_COUNT",
    "build_contact_sheet_inventory",
    "build_review_ledger_template",
    "build_reviewed_identity_manifest",
    "canonical_json_bytes",
    "finalize_identity_review",
]
