#!/usr/bin/env python3
"""Run the frozen multi-source Universal Orofacial Model v1 protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_110d_generalization_v1 import (  # noqa: E402
    GateAudit,
    _build_cache_metadata_dataset,
    load_development_cache_records,
    validate_development_gate,
)
from scripts.run_mirror_invariant_110d import mirror_dynamic_features  # noqa: E402
from src.datasets.dynamic_landmark import (  # noqa: E402
    load_dynamic_landmark_recording_bytes,
)
from src.evaluation.universal_orofacial_v1 import (  # noqa: E402
    CANDIDATES,
    CandidateEvaluation,
    UniversalDataset,
    aggregate_participant_recordings,
    binary_metrics,
    evaluate_candidate_oof,
    evaluate_leave_one_source_out,
    fit_locked_candidate,
    locked_candidate_from_dict,
    locked_candidate_to_dict,
    predict_locked_candidate,
    select_universal_candidate,
)
from src.evaluation.meei_external_v1 import validate_external_metadata  # noqa: E402
from src.preprocessing.action_capacity_features_v1 import (  # noqa: E402
    PINNED_NEUROFACE_COLLECTION_MANIFEST_SHA256,
)
from src.preprocessing.generalization_110d import (  # noqa: E402
    LANDMARK_MI_110D,
    candidate_feature_vector,
)
from src.preprocessing.script_action_segmentation_v1 import (  # noqa: E402
    PINNED_NEUROFACE_PRIVATE_MANIFEST_SHA256,
)


REPORT_SCHEMA = "universal_orofacial_development_public_report_v1"
MEEI_REPORT_SCHEMA = "universal_orofacial_meei_diagnostic_public_report_v1"
LOCKED_ARTIFACT_NAME = "locked_candidate.json"
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_CACHE_BYTES = 64 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROTECTED_AUDIT_FIELDS = (
    "palsynet_protected_cache_records_loaded",
    "palsynet_protected_predictions",
    "meei_reads",
    "mayo_reads",
    "yfp_reads",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("development", "meei-diagnostic"),
                        default="development")
    result.add_argument("--palsynet-cache-root", type=Path)
    result.add_argument("--reviewed-identity-manifest", type=Path)
    result.add_argument("--review-ledger", type=Path)
    result.add_argument("--split-registry", type=Path)
    result.add_argument("--neuroface-private-manifest", type=Path)
    result.add_argument("--neuroface-cache-root", type=Path)
    result.add_argument("--meei-participant-manifest", type=Path)
    result.add_argument("--meei-cache-root", type=Path)
    result.add_argument("--locked-artifact", type=Path)
    result.add_argument("--development-report", type=Path)
    result.add_argument("--output-root", required=True, type=Path)
    return result


def _strict_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("authenticated JSON contains a duplicate key")
        result[key] = value
    return result


def _read_regular_bytes(path: Path, *, maximum: int) -> bytes:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("authenticated input paths must be absolute")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > maximum
        ):
            raise ValueError("authenticated input must be a bounded regular file")
        chunks: list[bytes] = []
        size = 0
        while True:
            block = os.read(descriptor, min(1024 * 1024, maximum + 1 - size))
            if not block:
                break
            chunks.append(block)
            size += len(block)
            if size > maximum:
                raise ValueError("authenticated input exceeded its size limit")
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise ValueError("authenticated input changed while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _json_bytes(payload: bytes) -> dict[str, object]:
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_strict_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("authenticated input is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("authenticated JSON root must be an object")
    return value


def _read_json(path: Path) -> tuple[dict[str, object], str]:
    payload = _read_regular_bytes(path, maximum=MAX_JSON_BYTES)
    return _json_bytes(payload), hashlib.sha256(payload).hexdigest()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def implementation_component_paths() -> tuple[Path, ...]:
    """Return every direct execution/provenance dependency in stable order."""
    return (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/evaluation/universal_orofacial_v1.py",
        PROJECT_ROOT / "src/models/universal_orofacial_v1.py",
        PROJECT_ROOT / "src/evaluation/meei_external_v1.py",
        PROJECT_ROOT / "src/preprocessing/action_capacity_features_v1.py",
        PROJECT_ROOT / "src/preprocessing/script_action_segmentation_v1.py",
        PROJECT_ROOT / "src/preprocessing/generalization_110d.py",
        PROJECT_ROOT / "src/preprocessing/trajectory_features.py",
        PROJECT_ROOT / "src/preprocessing/clinical_landmarks.py",
        PROJECT_ROOT / "scripts/run_mirror_invariant_110d.py",
        PROJECT_ROOT / "scripts/run_110d_generalization_v1.py",
        PROJECT_ROOT / "src/datasets/dynamic_landmark.py",
        PROJECT_ROOT / "src/datasets/patient_multistream.py",
    )


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in implementation_component_paths():
        payload = path.read_bytes()
        relative = path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big") + relative)
        digest.update(len(payload).to_bytes(8, "big") + payload)
    return digest.hexdigest()


def _feature_pair(record) -> tuple[np.ndarray, np.ndarray]:
    temporal = (
        record.valid_mask, record.timestamps, record.source_frame_indices,
    )
    original = candidate_feature_vector(
        LANDMARK_MI_110D, record.features, *temporal
    )
    mirrored_raw = mirror_dynamic_features(record.features)
    mirrored = candidate_feature_vector(
        LANDMARK_MI_110D, mirrored_raw, *temporal
    )
    return original.astype(np.float64), mirrored.astype(np.float64)


def load_palsynet_development_rows(
    cache_root: Path,
    reviewed_identity_manifest: Path,
    review_ledger: Path,
    split_registry: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[str], dict[str, object]]:
    """Load only authenticated PalsyNet development rows, never protected rows."""
    manifest, manifest_sha = _read_json(reviewed_identity_manifest)
    ledger, ledger_sha = _read_json(review_ledger)
    registry, registry_sha = _read_json(split_registry)
    dataset, collection_rows, source_collection_sha = (
        _build_cache_metadata_dataset(cache_root)
    )
    audit = GateAudit()
    gate = validate_development_gate(
        dataset, manifest, ledger, registry,
        reviewed_manifest_sha256=manifest_sha,
        review_ledger_sha256=ledger_sha,
        split_registry_sha256=registry_sha,
        cache_source_sha256_by_recording_id={
            recording_id: str(row["source_sha256"])
            for recording_id, row in collection_rows.items()
        },
        cache_source_collection_sha256=source_collection_sha,
        audit=audit,
    )

    def same_bytes_loader(path: Path):
        payload = _read_regular_bytes(path, maximum=MAX_CACHE_BYTES)
        return load_dynamic_landmark_recording_bytes(payload)

    load_development_cache_records(
        cache_root, dataset, gate, collection_rows,
        audit=audit, record_loader=same_bytes_loader,
    )
    original: list[np.ndarray] = []
    mirrored: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    for index in gate.development_indices.tolist():
        temporal = (
            dataset.valid_masks[index], dataset.timestamps[index],
            dataset.source_frame_indices[index],
        )
        raw = dataset.features[index]
        original.append(candidate_feature_vector(
            LANDMARK_MI_110D, raw, *temporal
        ).astype(np.float64))
        mirrored.append(candidate_feature_vector(
            LANDMARK_MI_110D, mirror_dynamic_features(raw), *temporal
        ).astype(np.float64))
        labels.append(int(dataset.labels[index]))
        groups.append(str(gate.group_ids[index]))
        audit.development_feature_extractions += 2
        audit.development_mirror_transforms += 1
    audit_fields = audit.as_dict()
    if any(audit_fields[name] != 0 for name in audit_fields if name.startswith("protected_")):
        raise AssertionError("PalsyNet protected audit must remain exactly zero")
    return original, mirrored, labels, groups, {
        "split_registry_sha256": registry_sha,
        "development_recordings": len(original),
        "development_groups": len(set(groups)),
        "audit": audit_fields,
    }


def _unique_rows(rows: object, *, name: str) -> dict[str, Mapping[str, object]]:
    if not isinstance(rows, list):
        raise ValueError(f"{name} must be a row list")
    result: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("recording_id"), str):
            raise ValueError(f"{name} contains an invalid row")
        recording_id = str(row["recording_id"])
        if recording_id in result:
            raise ValueError(f"{name} contains a duplicate recording")
        result[recording_id] = row
    return result


def load_neuroface_rows(
    private_manifest_path: Path,
    cache_root: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[str], dict[str, object]]:
    """Load the frozen 231-record NeuroFace development collection."""
    private_payload = _read_regular_bytes(
        private_manifest_path, maximum=MAX_JSON_BYTES
    )
    private_sha = hashlib.sha256(private_payload).hexdigest()
    if private_sha != PINNED_NEUROFACE_PRIVATE_MANIFEST_SHA256:
        raise ValueError("NeuroFace private manifest differs from the frozen pin")
    collection_path = cache_root / "collection_manifest.json"
    collection_payload = _read_regular_bytes(
        collection_path, maximum=MAX_JSON_BYTES
    )
    collection_sha = hashlib.sha256(collection_payload).hexdigest()
    if collection_sha != PINNED_NEUROFACE_COLLECTION_MANIFEST_SHA256:
        raise ValueError("NeuroFace collection manifest differs from the frozen pin")
    private = _json_bytes(private_payload)
    collection = _json_bytes(collection_payload)
    if (
        private.get("schema_version") != "neuroface_external_private_manifest_v1"
        or private.get("counts", {}).get("participants") != 36
        or private.get("counts", {}).get("affected_participants") != 25
        or private.get("counts", {}).get("unaffected_participants") != 11
    ):
        raise ValueError("NeuroFace private population differs from the frozen endpoint")
    if (
        collection.get("schema_version") != "neuroface_clinical23_v2_windows_v1"
        or collection.get("counts") != {
            "source_records": 261, "retained": 231, "excluded": 30,
            "participants": 36, "primary_complete_participants": 36,
        }
    ):
        raise ValueError("NeuroFace cache collection differs from the frozen endpoint")
    private_by_id = _unique_rows(private.get("records"), name="private manifest")
    collection_by_id = _unique_rows(collection.get("records"), name="cache collection")
    if set(private_by_id) != set(collection_by_id) or len(private_by_id) != 261:
        raise ValueError("NeuroFace private and cache recording coverage differs")
    original: list[np.ndarray] = []
    mirrored: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    retained = 0
    for recording_id in sorted(collection_by_id):
        cache_row = collection_by_id[recording_id]
        private_row = private_by_id[recording_id]
        if cache_row.get("status") != "retained":
            continue
        retained += 1
        group = private_row.get("participant_id")
        cohort = private_row.get("cohort")
        label_name = private_row.get("binary_label")
        source_sha = private_row.get("video_sha256")
        expected_label = 0 if cohort == "healthy_control" else 1
        if (
            cohort not in {"als", "post_stroke", "healthy_control"}
            or label_name != ("unaffected" if expected_label == 0 else "affected")
            or not isinstance(group, str)
            or cache_row.get("participant_id") != group
            or cache_row.get("video_sha256") != source_sha
            or not isinstance(cache_row.get("cache_sha256"), str)
        ):
            raise ValueError("NeuroFace private/cache row identity differs")
        cache_payload = _read_regular_bytes(
            cache_root / f"{recording_id}.npz", maximum=MAX_CACHE_BYTES
        )
        if hashlib.sha256(cache_payload).hexdigest() != cache_row["cache_sha256"]:
            raise ValueError("NeuroFace cache bytes differ from the collection pin")
        record = load_dynamic_landmark_recording_bytes(cache_payload)
        if (
            record.recording_id != recording_id
            or record.group_id != group
            or record.source_sha256 != source_sha
            or record.label != expected_label
        ):
            raise ValueError("NeuroFace cache metadata differs from private identity")
        pair = _feature_pair(record)
        original.append(pair[0])
        mirrored.append(pair[1])
        labels.append(expected_label)
        groups.append(group)
    if retained != 231 or len(set(groups)) != 36:
        raise ValueError("NeuroFace retained participant coverage is incomplete")
    return original, mirrored, labels, groups, {
        "private_manifest_sha256": private_sha,
        "collection_manifest_sha256": collection_sha,
        "retained_recordings": retained,
        "participants": len(set(groups)),
    }


def load_development_dataset(args: argparse.Namespace) -> tuple[UniversalDataset, dict[str, object], dict[str, int]]:
    required = (
        "palsynet_cache_root", "reviewed_identity_manifest", "review_ledger",
        "split_registry", "neuroface_private_manifest", "neuroface_cache_root",
    )
    if any(getattr(args, name) is None for name in required):
        raise ValueError("development mode requires every frozen data input")
    p_original, p_mirrored, p_labels, p_groups, p_evidence = (
        load_palsynet_development_rows(
            args.palsynet_cache_root, args.reviewed_identity_manifest,
            args.review_ledger, args.split_registry,
        )
    )
    n_original, n_mirrored, n_labels, n_groups, n_evidence = load_neuroface_rows(
        args.neuroface_private_manifest, args.neuroface_cache_root
    )
    dataset = aggregate_participant_recordings(
        np.asarray(p_original + n_original, dtype=np.float64),
        np.asarray(p_mirrored + n_mirrored, dtype=np.float64),
        np.asarray(p_labels + n_labels, dtype=np.int64),
        tuple(p_groups + n_groups),
        tuple(["palsynet"] * len(p_groups) + ["neuroface"] * len(n_groups)),
    )
    counts = {
        "participants": len(dataset.group_ids),
        "palsynet_participants": sum(source == "palsynet" for source in dataset.sources),
        "neuroface_participants": sum(source == "neuroface" for source in dataset.sources),
    }
    if counts != {
        "participants": 74, "palsynet_participants": 38,
        "neuroface_participants": 36,
    }:
        raise ValueError("universal development population differs from the frozen 74 people")
    provenance = {
        "implementation_sha256": _implementation_sha256(),
        "palsynet_split_registry_sha256": p_evidence["split_registry_sha256"],
        "neuroface_private_manifest_sha256": n_evidence["private_manifest_sha256"],
        "neuroface_collection_manifest_sha256": n_evidence["collection_manifest_sha256"],
    }
    audit = {
        "palsynet_development_cache_records_loaded": int(
            p_evidence["audit"]["development_cache_records_loaded"]
        ),
        "palsynet_protected_cache_records_loaded": int(
            p_evidence["audit"]["protected_cache_records_loaded"]
        ),
        "palsynet_protected_predictions": int(
            p_evidence["audit"]["protected_predictions"]
        ),
        "meei_reads": 0, "mayo_reads": 0, "yfp_reads": 0,
    }
    return dataset, provenance, audit


def _plain_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    required = {
        "accuracy", "auroc", "balanced_accuracy", "sensitivity",
        "specificity", "brier",
    }
    if set(metrics) != required:
        raise ValueError("universal metric fields differ from the frozen schema")
    result = {name: float(metrics[name]) for name in sorted(required)}
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("universal metrics must be finite")
    return result


def build_public_development_report(
    evaluations: Mapping[str, CandidateEvaluation],
    transfers: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    counts: Mapping[str, int],
    provenance: Mapping[str, str],
    audit: Mapping[str, int],
    locked_artifact_sha256: str,
) -> dict[str, object]:
    """Publish only aggregate, independently selectable development evidence."""
    if set(evaluations) != set(CANDIDATES) or set(transfers) != set(CANDIDATES):
        raise ValueError("public report requires exactly the three frozen candidates")
    candidate_reports: dict[str, object] = {}
    summaries: dict[str, dict[str, float]] = {}
    for candidate in CANDIDATES:
        evaluation = evaluations[candidate]
        if evaluation.candidate != candidate:
            raise ValueError("candidate evaluation identity changed")
        metrics = {
            name: _plain_metrics(evaluation.metrics[name])
            for name in ("overall", "palsynet", "neuroface")
        }
        transfer = transfers[candidate]
        if set(transfer) != {
            "palsynet_to_neuroface", "neuroface_to_palsynet",
        }:
            raise ValueError("both source-heldout directions are required")
        transfer_output: dict[str, object] = {}
        for direction in ("palsynet_to_neuroface", "neuroface_to_palsynet"):
            row = transfer[direction]
            transfer_output[direction] = {
                "training_source": row["training_source"],
                "held_source": row["held_source"],
                "training_participants": int(row["training_participants"]),
                "held_participants": int(row["held_participants"]),
                "model_fits": int(row["model_fits"]),
                "metrics": _plain_metrics(row["metrics"]),
            }
        summaries[candidate] = {
            "worst_source_auroc": min(
                metrics[source]["auroc"] for source in ("palsynet", "neuroface")
            ),
            "worst_source_balanced_accuracy": min(
                metrics[source]["balanced_accuracy"]
                for source in ("palsynet", "neuroface")
            ),
            "overall_brier": metrics["overall"]["brier"],
        }
        candidate_reports[candidate] = {
            "protocol": evaluation.protocol,
            "model_fits": int(evaluation.model_fits),
            "metrics": metrics,
            "selection_summary": summaries[candidate],
            "source_heldout": transfer_output,
        }
    winner = select_universal_candidate(summaries)
    winner_report = candidate_reports[winner]
    oof_metrics = winner_report["metrics"]
    transfer_metrics = winner_report["source_heldout"]
    gates = {
        "palsynet_oof_auroc_at_least_0_90": oof_metrics["palsynet"]["auroc"] >= 0.90,
        "palsynet_oof_balanced_accuracy_at_least_0_90": oof_metrics["palsynet"]["balanced_accuracy"] >= 0.90,
        "neuroface_oof_auroc_at_least_0_90": oof_metrics["neuroface"]["auroc"] >= 0.90,
        "neuroface_oof_balanced_accuracy_at_least_0_90": oof_metrics["neuroface"]["balanced_accuracy"] >= 0.90,
        "palsynet_to_neuroface_auroc_at_least_0_90": transfer_metrics["palsynet_to_neuroface"]["metrics"]["auroc"] >= 0.90,
        "neuroface_to_palsynet_auroc_at_least_0_90": transfer_metrics["neuroface_to_palsynet"]["metrics"]["auroc"] >= 0.90,
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "endpoint": "affected_vs_unaffected_across_palsynet_and_neuroface",
        "representation": {
            "name": "landmark_mi_110d",
            "dimension": 110,
            "participant_aggregation": "mean_recordings_then_mean_original_mirror_probability",
        },
        "counts": {name: int(counts[name]) for name in (
            "participants", "palsynet_participants", "neuroface_participants",
        )},
        "candidates": candidate_reports,
        "decision": {
            "selection_rule": "worst_source_auroc_then_worst_source_balanced_accuracy_then_overall_brier",
            "locked_candidate": winner,
            "gates": gates,
            "development_gate_passed": all(gates.values()),
            "locked_artifact_sha256": _sha(
                locked_artifact_sha256, "locked artifact"
            ),
        },
        "audit": {name: int(audit[name]) for name in (
            "palsynet_development_cache_records_loaded",
            *_PROTECTED_AUDIT_FIELDS,
        )},
        "provenance": {name: _sha(provenance[name], name) for name in (
            "implementation_sha256", "palsynet_split_registry_sha256",
            "neuroface_private_manifest_sha256",
            "neuroface_collection_manifest_sha256",
        )},
        "claim_boundary": {
            "development_only": True,
            "meei_used_for_selection": False,
            "mayo_used_for_selection": False,
            "palsynet_protected_used": False,
            "cross_institutionally_robust": False,
            "clinical_deployment_claim": False,
        },
    }
    validate_public_development_report(report)
    return report


def validate_public_development_report(report: object) -> None:
    if not isinstance(report, Mapping) or set(report) != {
        "schema_version", "endpoint", "representation", "counts", "candidates",
        "decision", "audit", "provenance", "claim_boundary",
    }:
        raise ValueError("public development report schema is invalid")
    if report["schema_version"] != REPORT_SCHEMA:
        raise ValueError("public development report version is invalid")
    candidates = report["candidates"]
    if not isinstance(candidates, Mapping) or set(candidates) != set(CANDIDATES):
        raise ValueError("public report candidate set is invalid")
    summaries: dict[str, dict[str, float]] = {}
    for candidate in CANDIDATES:
        row = candidates[candidate]
        if not isinstance(row, Mapping) or set(row) != {
            "protocol", "model_fits", "metrics", "selection_summary",
            "source_heldout",
        }:
            raise ValueError("public candidate report schema is invalid")
        metrics = row["metrics"]
        if not isinstance(metrics, Mapping) or set(metrics) != {
            "overall", "palsynet", "neuroface",
        }:
            raise ValueError("public source metric schema is invalid")
        normalized = {name: _plain_metrics(metrics[name]) for name in metrics}
        expected = {
            "worst_source_auroc": min(
                normalized[source]["auroc"] for source in ("palsynet", "neuroface")
            ),
            "worst_source_balanced_accuracy": min(
                normalized[source]["balanced_accuracy"]
                for source in ("palsynet", "neuroface")
            ),
            "overall_brier": normalized["overall"]["brier"],
        }
        if row["selection_summary"] != expected:
            raise ValueError("public selection summary is not recomputable")
        summaries[candidate] = expected
    decision = report["decision"]
    if not isinstance(decision, Mapping) or set(decision) != {
        "selection_rule", "locked_candidate", "gates",
        "development_gate_passed", "locked_artifact_sha256",
    }:
        raise ValueError("public decision schema is invalid")
    winner = select_universal_candidate(summaries)
    if decision["locked_candidate"] != winner:
        raise ValueError("published winner differs from aggregate metrics")
    winner_row = candidates[winner]
    metrics = winner_row["metrics"]
    transfers = winner_row["source_heldout"]
    expected_gates = {
        "palsynet_oof_auroc_at_least_0_90": metrics["palsynet"]["auroc"] >= 0.90,
        "palsynet_oof_balanced_accuracy_at_least_0_90": metrics["palsynet"]["balanced_accuracy"] >= 0.90,
        "neuroface_oof_auroc_at_least_0_90": metrics["neuroface"]["auroc"] >= 0.90,
        "neuroface_oof_balanced_accuracy_at_least_0_90": metrics["neuroface"]["balanced_accuracy"] >= 0.90,
        "palsynet_to_neuroface_auroc_at_least_0_90": transfers["palsynet_to_neuroface"]["metrics"]["auroc"] >= 0.90,
        "neuroface_to_palsynet_auroc_at_least_0_90": transfers["neuroface_to_palsynet"]["metrics"]["auroc"] >= 0.90,
    }
    if (
        decision["gates"] != expected_gates
        or decision["development_gate_passed"] is not all(expected_gates.values())
    ):
        raise ValueError("public development gates are not recomputable")
    _sha(decision["locked_artifact_sha256"], "locked artifact")
    audit = report["audit"]
    if not isinstance(audit, Mapping) or any(audit.get(name) != 0 for name in _PROTECTED_AUDIT_FIELDS):
        raise ValueError("public report indicates protected or external selection access")
    encoded = json.dumps(report, sort_keys=True, allow_nan=False)
    if any(token in encoded for token in (
        "grp_", "rec_", "probabilities", "coefficient", "intercept", "/Users/",
    )):
        raise ValueError("public report contains private row-level or model evidence")


def build_public_meei_diagnostic_report(
    *,
    candidate: str,
    metrics: Mapping[str, float],
    counts: Mapping[str, int],
    development_report_sha256: str,
    locked_artifact_sha256: str,
    participant_manifest_sha256: str,
    collection_manifest_sha256: str,
    cache_collection_sha256: str,
) -> dict[str, object]:
    """Describe one locked MEEI diagnostic without authorizing reselection."""
    if candidate not in CANDIDATES or dict(counts) != {
        "participants": 60, "affected": 50, "unaffected": 10,
    }:
        raise ValueError("MEEI diagnostic population or candidate is invalid")
    normalized_metrics = _plain_metrics(metrics)
    cross_robust = (
        normalized_metrics["auroc"] >= 0.90
        and normalized_metrics["balanced_accuracy"] >= 0.90
    )
    return {
        "schema_version": MEEI_REPORT_SCHEMA,
        "dataset": "MEEI_Facial_Palsy_Standard_Set",
        "endpoint": "normal_vs_facial_palsy",
        "counts": {name: int(counts[name]) for name in (
            "participants", "affected", "unaffected",
        )},
        "locked_candidate": candidate,
        "metrics": normalized_metrics,
        "protocol": {
            "candidate_selection": False,
            "model_refit": False,
            "scaler_refit": False,
            "threshold_selection": False,
            "status": "repeated_already_exposed_external_diagnostic",
        },
        "decision": {
            "criterion": "auroc_and_balanced_accuracy_at_least_0_90",
            "cross_institutionally_robust": cross_robust,
            "universal_model_promoted": cross_robust,
        },
        "provenance": {
            "development_report_sha256": _sha(
                development_report_sha256, "development report"
            ),
            "locked_artifact_sha256": _sha(
                locked_artifact_sha256, "locked artifact"
            ),
            "participant_manifest_sha256": _sha(
                participant_manifest_sha256, "participant manifest"
            ),
            "collection_manifest_sha256": _sha(
                collection_manifest_sha256, "collection manifest"
            ),
            "cache_collection_sha256": _sha(
                cache_collection_sha256, "cache collection"
            ),
            "implementation_sha256": _implementation_sha256(),
        },
        "claim_boundary": {
            "repeated_diagnostic_not_untouched_external_validation": True,
            "clinical_deployment_claim": False,
            "mayo_accuracy_claim": False,
        },
    }


def load_meei_rows(
    participant_manifest_path: Path,
    cache_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Load the validated 60-person MEEI cache for locked inference only."""
    participant, participant_sha = _read_json(participant_manifest_path)
    collection, collection_sha = _read_json(cache_root / "collection_manifest.json")
    metadata = validate_external_metadata(
        participant, collection,
        participant_manifest_sha256=participant_sha,
    )
    expected_paths = {
        cache_root / f"{row['recording_id']}.npz" for row in metadata.rows
    }
    if set(cache_root.glob("*.npz")) != expected_paths:
        raise ValueError("MEEI cache paths differ from validated metadata")
    original: list[np.ndarray] = []
    mirrored: list[np.ndarray] = []
    labels: list[int] = []
    collection_digest = hashlib.sha256()
    for row in sorted(metadata.rows, key=lambda value: str(value["recording_id"])):
        recording_id = str(row["recording_id"])
        payload = _read_regular_bytes(
            cache_root / f"{recording_id}.npz", maximum=MAX_CACHE_BYTES
        )
        digest = hashlib.sha256(payload).hexdigest()
        collection_digest.update(
            f"{recording_id}:{digest}:{len(payload)}\n".encode("ascii")
        )
        record = load_dynamic_landmark_recording_bytes(payload)
        expected_label = 1 if row["label"] == "affected" else 0
        if (
            record.recording_id != recording_id
            or record.group_id != row["group_id"]
            or record.source_sha256 != row["source_sha256"]
            or record.label != expected_label
        ):
            raise ValueError("MEEI cache identity differs from validated metadata")
        pair = _feature_pair(record)
        original.append(pair[0])
        mirrored.append(pair[1])
        labels.append(expected_label)
    return (
        np.asarray(original, dtype=np.float64),
        np.asarray(mirrored, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        {
            "participant_manifest_sha256": participant_sha,
            "collection_manifest_sha256": collection_sha,
            "cache_collection_sha256": collection_digest.hexdigest(),
            "counts": {
                "participants": metadata.participants_total,
                "affected": metadata.affected,
                "unaffected": metadata.unaffected,
            },
        },
    )


def _canonical_json_bytes(payload: object) -> bytes:
    return (json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ) + "\n").encode("utf-8")


def _write_bytes(path: Path, payload: bytes, *, mode: int) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        total = 0
        while total < len(payload):
            written = os.write(descriptor, payload[total:])
            if written <= 0:
                raise OSError("short release write")
            total += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_development_release(
    output_root: Path,
    report: Mapping[str, object],
    artifact_payload: bytes,
) -> None:
    if not output_root.is_absolute() or output_root.exists() or output_root.is_symlink():
        raise ValueError("output root must be a new absolute path")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output_root.name}.staging-", dir=output_root.parent
    ))
    try:
        os.chmod(staging, 0o700)
        private = staging / "private"
        private.mkdir(mode=0o700)
        _write_bytes(private / LOCKED_ARTIFACT_NAME, artifact_payload, mode=0o600)
        _write_bytes(staging / "report.json", _canonical_json_bytes(report), mode=0o644)
        staging.rename(output_root)
    except BaseException:
        for path in sorted(staging.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
        staging.rmdir()
        raise


def _publish_public_report(output_root: Path, report: Mapping[str, object]) -> None:
    if not output_root.is_absolute() or output_root.exists() or output_root.is_symlink():
        raise ValueError("output root must be a new absolute path")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output_root.name}.staging-", dir=output_root.parent
    ))
    try:
        os.chmod(staging, 0o700)
        _write_bytes(staging / "report.json", _canonical_json_bytes(report), mode=0o644)
        staging.rename(output_root)
    except BaseException:
        for path in sorted(staging.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
        staging.rmdir()
        raise


def run_development(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) not in {
        "NVIDIA H200", "NVIDIA H200 NVL",
    }:
        raise ValueError("formal universal development requires the verified H200")
    dataset, provenance, audit = load_development_dataset(args)
    implementation_before = _implementation_sha256()
    evaluations = {
        candidate: evaluate_candidate_oof(dataset, candidate, device="cuda")
        for candidate in CANDIDATES
    }
    transfers = {
        candidate: evaluate_leave_one_source_out(dataset, candidate, device="cuda")
        for candidate in CANDIDATES
    }
    summaries = {
        candidate: {
            "worst_source_auroc": min(
                evaluations[candidate].metrics[source]["auroc"]
                for source in ("palsynet", "neuroface")
            ),
            "worst_source_balanced_accuracy": min(
                evaluations[candidate].metrics[source]["balanced_accuracy"]
                for source in ("palsynet", "neuroface")
            ),
            "overall_brier": evaluations[candidate].metrics["overall"]["brier"],
        }
        for candidate in CANDIDATES
    }
    winner = select_universal_candidate(summaries)
    locked = fit_locked_candidate(dataset, winner, device="cuda")
    artifact_payload = _canonical_json_bytes(locked_candidate_to_dict(locked))
    artifact_sha = hashlib.sha256(artifact_payload).hexdigest()
    implementation_after = _implementation_sha256()
    if implementation_before != implementation_after or provenance["implementation_sha256"] != implementation_before:
        raise RuntimeError("universal implementation changed during formal evaluation")
    report = build_public_development_report(
        evaluations, transfers,
        counts={
            "participants": len(dataset.group_ids),
            "palsynet_participants": sum(source == "palsynet" for source in dataset.sources),
            "neuroface_participants": sum(source == "neuroface" for source in dataset.sources),
        },
        provenance=provenance, audit=audit,
        locked_artifact_sha256=artifact_sha,
    )
    _publish_development_release(args.output_root, report, artifact_payload)
    return report


def run_meei_diagnostic(args: argparse.Namespace) -> dict[str, object]:
    required = (
        "meei_participant_manifest", "meei_cache_root", "locked_artifact",
        "development_report",
    )
    if any(getattr(args, name) is None for name in required):
        raise ValueError("MEEI diagnostic requires its locked evidence inputs")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) not in {
        "NVIDIA H200", "NVIDIA H200 NVL",
    }:
        raise ValueError("formal universal diagnostic requires the verified H200")
    development_payload = _read_regular_bytes(
        args.development_report, maximum=MAX_JSON_BYTES
    )
    development_sha = hashlib.sha256(development_payload).hexdigest()
    development = _json_bytes(development_payload)
    validate_public_development_report(development)
    artifact_payload = _read_regular_bytes(
        args.locked_artifact, maximum=MAX_JSON_BYTES
    )
    artifact_sha = hashlib.sha256(artifact_payload).hexdigest()
    if artifact_sha != development["decision"]["locked_artifact_sha256"]:
        raise ValueError("locked artifact differs from the development decision")
    locked = locked_candidate_from_dict(_json_bytes(artifact_payload))
    if locked.candidate != development["decision"]["locked_candidate"]:
        raise ValueError("locked candidate identity differs from development")
    original, mirrored, labels, evidence = load_meei_rows(
        args.meei_participant_manifest, args.meei_cache_root
    )
    probabilities = predict_locked_candidate(
        locked, original, mirrored, device="cuda"
    )
    report = build_public_meei_diagnostic_report(
        candidate=locked.candidate,
        metrics=binary_metrics(labels, probabilities),
        counts=evidence["counts"],
        development_report_sha256=development_sha,
        locked_artifact_sha256=artifact_sha,
        participant_manifest_sha256=evidence["participant_manifest_sha256"],
        collection_manifest_sha256=evidence["collection_manifest_sha256"],
        cache_collection_sha256=evidence["cache_collection_sha256"],
    )
    _publish_public_report(args.output_root, report)
    return report


def main() -> int:
    args = parser().parse_args()
    if args.mode == "development":
        report = run_development(args)
    else:
        report = run_meei_diagnostic(args)
        print(json.dumps({
            "schema_version": "universal_orofacial_meei_diagnostic_receipt_v1",
            "cross_institutionally_robust": report["decision"]["cross_institutionally_robust"],
            "report_sha256": hashlib.sha256(
                _canonical_json_bytes(report)
            ).hexdigest(),
        }, sort_keys=True))
        return 0
    print(json.dumps({
        "schema_version": "universal_orofacial_development_receipt_v1",
        "locked_candidate": report["decision"]["locked_candidate"],
        "development_gate_passed": report["decision"]["development_gate_passed"],
        "report_sha256": hashlib.sha256(
            _canonical_json_bytes(report)
        ).hexdigest(),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
