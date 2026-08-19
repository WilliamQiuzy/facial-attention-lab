#!/usr/bin/env python3
"""Run Universal Multi-Signal v2 with fixed Landmark/Blendshape/Fusion arms."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping

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
from scripts.run_universal_orofacial_v1 import (  # noqa: E402
    MAX_CACHE_BYTES,
    MAX_JSON_BYTES,
    _canonical_json_bytes,
    _json_bytes,
    _plain_metrics,
    _publish_development_release,
    _read_json,
    _read_regular_bytes,
    _sha,
    _unique_rows,
)
from src.datasets.dynamic_landmark import load_dynamic_landmark_recording_bytes  # noqa: E402
from src.evaluation.universal_multisignal_v2 import (  # noqa: E402
    REPRESENTATIONS,
    CandidateEvaluation,
    MultiSignalDataset,
    aggregate_multisignal_recordings,
    evaluate_multisignal_leave_one_source_out,
    evaluate_multisignal_oof,
    fit_locked_multisignal,
    locked_multisignal_to_dict,
    select_multisignal_representation,
)
from src.preprocessing.action_capacity_features_v1 import (  # noqa: E402
    PINNED_NEUROFACE_COLLECTION_MANIFEST_SHA256,
)
from src.preprocessing.script_action_segmentation_v1 import (  # noqa: E402
    PINNED_NEUROFACE_PRIVATE_MANIFEST_SHA256,
)
from src.preprocessing.universal_multisignal_v2 import (  # noqa: E402
    multisignal_feature_views,
)


REPORT_SCHEMA = "universal_multisignal_development_public_report_v2"
_PROTECTED_FIELDS = (
    "palsynet_protected_cache_records_loaded",
    "palsynet_protected_predictions",
    "meei_reads", "mayo_reads", "yfp_reads",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--palsynet-cache-root", required=True, type=Path)
    result.add_argument("--reviewed-identity-manifest", required=True, type=Path)
    result.add_argument("--review-ledger", required=True, type=Path)
    result.add_argument("--split-registry", required=True, type=Path)
    result.add_argument("--neuroface-private-manifest", required=True, type=Path)
    result.add_argument("--neuroface-cache-root", required=True, type=Path)
    result.add_argument("--output-root", required=True, type=Path)
    return result


def implementation_component_paths() -> tuple[Path, ...]:
    """Return the exact transitive modules reused by the formal v2 runner."""
    return (
        Path(__file__).resolve(),
        PROJECT_ROOT / "scripts/run_universal_orofacial_v1.py",
        PROJECT_ROOT / "src/evaluation/universal_multisignal_v2.py",
        PROJECT_ROOT / "src/preprocessing/universal_multisignal_v2.py",
        PROJECT_ROOT / "src/evaluation/universal_orofacial_v1.py",
        PROJECT_ROOT / "src/evaluation/meei_external_v1.py",
        PROJECT_ROOT / "src/preprocessing/action_capacity_features_v1.py",
        PROJECT_ROOT / "src/preprocessing/script_action_segmentation_v1.py",
        PROJECT_ROOT / "src/preprocessing/trajectory_features.py",
        PROJECT_ROOT / "src/models/dynamic_landmark.py",
        PROJECT_ROOT / "src/datasets/dynamic_landmark.py",
        PROJECT_ROOT / "src/datasets/patient_multistream.py",
        PROJECT_ROOT / "scripts/run_110d_generalization_v1.py",
        PROJECT_ROOT / "scripts/run_dynamic_landmark_classical.py",
        PROJECT_ROOT / "scripts/freeze_palsynet_person_split_registry.py",
    )


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in implementation_component_paths():
        payload = path.read_bytes()
        relative = path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big") + relative)
        digest.update(len(payload).to_bytes(8, "big") + payload)
    return digest.hexdigest()


def _empty_rows() -> dict[str, tuple[list[np.ndarray], list[np.ndarray]]]:
    return {name: ([], []) for name in REPRESENTATIONS}


def _append_views(
    target: dict[str, tuple[list[np.ndarray], list[np.ndarray]]],
    views: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    if set(views) != set(REPRESENTATIONS):
        raise ValueError("multi-signal view set drifted")
    for name in REPRESENTATIONS:
        target[name][0].append(views[name][0])
        target[name][1].append(views[name][1])


def _palsynet_rows(args: argparse.Namespace):
    manifest, manifest_sha = _read_json(args.reviewed_identity_manifest)
    ledger, ledger_sha = _read_json(args.review_ledger)
    registry, registry_sha = _read_json(args.split_registry)
    dataset, collection_rows, source_collection_sha = _build_cache_metadata_dataset(
        args.palsynet_cache_root
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

    def loader(path: Path):
        return load_dynamic_landmark_recording_bytes(
            _read_regular_bytes(path, maximum=MAX_CACHE_BYTES)
        )

    load_development_cache_records(
        args.palsynet_cache_root, dataset, gate, collection_rows,
        audit=audit, record_loader=loader,
    )
    rows = _empty_rows()
    labels: list[int] = []
    groups: list[str] = []
    for index in gate.development_indices.tolist():
        _append_views(rows, multisignal_feature_views(
            dataset.features[index], dataset.valid_masks[index],
            dataset.timestamps[index], dataset.source_frame_indices[index],
        ))
        labels.append(int(dataset.labels[index]))
        groups.append(str(gate.group_ids[index]))
    values = audit.as_dict()
    if values["protected_cache_records_loaded"] != 0 or values["protected_predictions"] != 0:
        raise AssertionError("PalsyNet protected audit must remain zero")
    return rows, labels, groups, registry_sha, values


def _neuroface_rows(args: argparse.Namespace):
    private_payload = _read_regular_bytes(
        args.neuroface_private_manifest, maximum=MAX_JSON_BYTES
    )
    private_sha = hashlib.sha256(private_payload).hexdigest()
    collection_payload = _read_regular_bytes(
        args.neuroface_cache_root / "collection_manifest.json",
        maximum=MAX_JSON_BYTES,
    )
    collection_sha = hashlib.sha256(collection_payload).hexdigest()
    if (
        private_sha != PINNED_NEUROFACE_PRIVATE_MANIFEST_SHA256
        or collection_sha != PINNED_NEUROFACE_COLLECTION_MANIFEST_SHA256
    ):
        raise ValueError("NeuroFace manifests differ from frozen pins")
    private = _json_bytes(private_payload)
    collection = _json_bytes(collection_payload)
    if (
        private.get("counts", {}).get("participants") != 36
        or collection.get("counts") != {
            "source_records": 261, "retained": 231, "excluded": 30,
            "participants": 36, "primary_complete_participants": 36,
        }
    ):
        raise ValueError("NeuroFace frozen population is incomplete")
    private_by_id = _unique_rows(private.get("records"), name="private manifest")
    cache_by_id = _unique_rows(collection.get("records"), name="cache collection")
    if set(private_by_id) != set(cache_by_id) or len(cache_by_id) != 261:
        raise ValueError("NeuroFace manifest coverage differs")
    rows = _empty_rows()
    labels: list[int] = []
    groups: list[str] = []
    retained = 0
    for recording_id in sorted(cache_by_id):
        cache_row = cache_by_id[recording_id]
        private_row = private_by_id[recording_id]
        if cache_row.get("status") != "retained":
            continue
        retained += 1
        cohort = private_row.get("cohort")
        expected_label = 0 if cohort == "healthy_control" else 1
        group = private_row.get("participant_id")
        source_sha = private_row.get("video_sha256")
        if (
            cohort not in {"als", "post_stroke", "healthy_control"}
            or private_row.get("binary_label")
            != ("unaffected" if expected_label == 0 else "affected")
            or cache_row.get("participant_id") != group
            or cache_row.get("video_sha256") != source_sha
        ):
            raise ValueError("NeuroFace private/cache identity differs")
        payload = _read_regular_bytes(
            args.neuroface_cache_root / f"{recording_id}.npz",
            maximum=MAX_CACHE_BYTES,
        )
        if hashlib.sha256(payload).hexdigest() != cache_row.get("cache_sha256"):
            raise ValueError("NeuroFace cache differs from its manifest commitment")
        record = load_dynamic_landmark_recording_bytes(payload)
        if (
            record.recording_id != recording_id
            or record.group_id != group
            or record.source_sha256 != source_sha
            or record.label != expected_label
        ):
            raise ValueError("NeuroFace cache payload identity differs")
        _append_views(rows, multisignal_feature_views(
            record.features, record.valid_mask, record.timestamps,
            record.source_frame_indices,
        ))
        labels.append(expected_label)
        groups.append(str(group))
    if retained != 231 or len(set(groups)) != 36:
        raise ValueError("NeuroFace retained coverage differs from frozen counts")
    return rows, labels, groups, private_sha, collection_sha


def load_dataset(args: argparse.Namespace) -> tuple[MultiSignalDataset, dict[str, str], dict[str, int]]:
    p_rows, p_labels, p_groups, registry_sha, p_audit = _palsynet_rows(args)
    n_rows, n_labels, n_groups, private_sha, collection_sha = _neuroface_rows(args)
    combined = {}
    for name in REPRESENTATIONS:
        combined[name] = (
            np.asarray(p_rows[name][0] + n_rows[name][0], dtype=np.float64),
            np.asarray(p_rows[name][1] + n_rows[name][1], dtype=np.float64),
        )
    dataset = aggregate_multisignal_recordings(
        combined,
        np.asarray(p_labels + n_labels, dtype=np.int64),
        tuple(p_groups + n_groups),
        tuple(["palsynet"] * len(p_groups) + ["neuroface"] * len(n_groups)),
    )
    if (
        len(dataset.group_ids) != 74
        or sum(source == "palsynet" for source in dataset.sources) != 38
        or sum(source == "neuroface" for source in dataset.sources) != 36
    ):
        raise ValueError("multi-signal participant population differs from 74")
    return dataset, {
        "implementation_sha256": _implementation_sha256(),
        "palsynet_split_registry_sha256": registry_sha,
        "neuroface_private_manifest_sha256": private_sha,
        "neuroface_collection_manifest_sha256": collection_sha,
    }, {
        "palsynet_protected_cache_records_loaded": int(
            p_audit["protected_cache_records_loaded"]
        ),
        "palsynet_protected_predictions": int(p_audit["protected_predictions"]),
        "meei_reads": 0, "mayo_reads": 0, "yfp_reads": 0,
    }


def build_public_report(
    evaluations: Mapping[str, CandidateEvaluation],
    transfers: Mapping[str, Mapping[str, Mapping[str, object]]],
    *, counts: Mapping[str, int], provenance: Mapping[str, str],
    audit: Mapping[str, int], locked_artifact_sha256: str,
) -> dict[str, object]:
    if set(evaluations) != set(REPRESENTATIONS) or set(transfers) != set(REPRESENTATIONS):
        raise ValueError("v2 report requires all frozen representations")
    candidates = {}
    summaries = {}
    for name in REPRESENTATIONS:
        evaluation = evaluations[name]
        metrics = {source: _plain_metrics(evaluation.metrics[source])
                   for source in ("overall", "palsynet", "neuroface")}
        heldout = {}
        for direction in ("palsynet_to_neuroface", "neuroface_to_palsynet"):
            row = transfers[name][direction]
            heldout[direction] = {
                "training_source": row["training_source"],
                "held_source": row["held_source"],
                "training_participants": int(row["training_participants"]),
                "held_participants": int(row["held_participants"]),
                "model_fits": int(row["model_fits"]),
                "metrics": _plain_metrics(row["metrics"]),
            }
        summaries[name] = {
            "worst_source_auroc": min(metrics[source]["auroc"]
                                      for source in ("palsynet", "neuroface")),
            "worst_source_balanced_accuracy": min(
                metrics[source]["balanced_accuracy"]
                for source in ("palsynet", "neuroface")
            ),
            "overall_brier": metrics["overall"]["brier"],
        }
        candidates[name] = {
            "dimension": REPRESENTATIONS[name],
            "protocol": evaluation.protocol,
            "model_fits": int(evaluation.model_fits),
            "metrics": metrics,
            "selection_summary": summaries[name],
            "source_heldout": heldout,
        }
    winner = select_multisignal_representation(summaries)
    winning = candidates[winner]
    metrics = winning["metrics"]
    heldout = winning["source_heldout"]
    gates = {
        "palsynet_oof_auroc_at_least_0_90": metrics["palsynet"]["auroc"] >= .90,
        "palsynet_oof_balanced_accuracy_at_least_0_90": metrics["palsynet"]["balanced_accuracy"] >= .90,
        "neuroface_oof_auroc_at_least_0_90": metrics["neuroface"]["auroc"] >= .90,
        "neuroface_oof_balanced_accuracy_at_least_0_90": metrics["neuroface"]["balanced_accuracy"] >= .90,
        "palsynet_to_neuroface_auroc_at_least_0_90": heldout["palsynet_to_neuroface"]["metrics"]["auroc"] >= .90,
        "neuroface_to_palsynet_auroc_at_least_0_90": heldout["neuroface_to_palsynet"]["metrics"]["auroc"] >= .90,
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "endpoint": "affected_vs_unaffected_across_palsynet_and_neuroface",
        "estimator": {
            "type": "source_class_balanced_l2_logistic",
            "c": .01, "solver": "liblinear", "threshold": .5,
            "mirror": "train_both_predict_probability_mean",
        },
        "counts": {name: int(counts[name]) for name in (
            "participants", "palsynet_participants", "neuroface_participants",
        )},
        "candidates": candidates,
        "decision": {
            "selection_rule": "worst_source_auroc_then_worst_source_balanced_accuracy_then_overall_brier",
            "locked_representation": winner,
            "gates": gates,
            "development_gate_passed": all(gates.values()),
            "locked_artifact_sha256": _sha(locked_artifact_sha256, "locked artifact"),
        },
        "audit": {name: int(audit[name]) for name in _PROTECTED_FIELDS},
        "provenance": {name: _sha(provenance[name], name) for name in (
            "implementation_sha256", "palsynet_split_registry_sha256",
            "neuroface_private_manifest_sha256",
            "neuroface_collection_manifest_sha256",
        )},
        "claim_boundary": {
            "development_only": True, "palsynet_protected_used": False,
            "meei_used_for_selection": False, "mayo_used_for_selection": False,
            "clinical_deployment_claim": False,
        },
    }
    validate_public_report(report)
    return report


def validate_public_report(report: object) -> None:
    if not isinstance(report, Mapping) or set(report) != {
        "schema_version", "endpoint", "estimator", "counts", "candidates",
        "decision", "audit", "provenance", "claim_boundary",
    } or report["schema_version"] != REPORT_SCHEMA:
        raise ValueError("v2 public report schema is invalid")
    candidates = report["candidates"]
    if not isinstance(candidates, Mapping) or set(candidates) != set(REPRESENTATIONS):
        raise ValueError("v2 public representation set is invalid")
    summaries = {}
    for name in REPRESENTATIONS:
        row = candidates[name]
        if not isinstance(row, Mapping) or row.get("dimension") != REPRESENTATIONS[name]:
            raise ValueError("v2 representation metadata is invalid")
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping) or set(metrics) != {
            "overall", "palsynet", "neuroface",
        }:
            raise ValueError("v2 metrics are incomplete")
        normalized = {source: _plain_metrics(metrics[source]) for source in metrics}
        expected = {
            "worst_source_auroc": min(normalized[source]["auroc"]
                                      for source in ("palsynet", "neuroface")),
            "worst_source_balanced_accuracy": min(
                normalized[source]["balanced_accuracy"]
                for source in ("palsynet", "neuroface")
            ),
            "overall_brier": normalized["overall"]["brier"],
        }
        if row.get("selection_summary") != expected:
            raise ValueError("v2 selection summary is not recomputable")
        summaries[name] = expected
    winner = select_multisignal_representation(summaries)
    decision = report["decision"]
    if not isinstance(decision, Mapping) or decision.get("locked_representation") != winner:
        raise ValueError("v2 published winner differs from aggregate metrics")
    winning = candidates[winner]
    metrics = winning["metrics"]
    heldout = winning["source_heldout"]
    gates = {
        "palsynet_oof_auroc_at_least_0_90": metrics["palsynet"]["auroc"] >= .90,
        "palsynet_oof_balanced_accuracy_at_least_0_90": metrics["palsynet"]["balanced_accuracy"] >= .90,
        "neuroface_oof_auroc_at_least_0_90": metrics["neuroface"]["auroc"] >= .90,
        "neuroface_oof_balanced_accuracy_at_least_0_90": metrics["neuroface"]["balanced_accuracy"] >= .90,
        "palsynet_to_neuroface_auroc_at_least_0_90": heldout["palsynet_to_neuroface"]["metrics"]["auroc"] >= .90,
        "neuroface_to_palsynet_auroc_at_least_0_90": heldout["neuroface_to_palsynet"]["metrics"]["auroc"] >= .90,
    }
    if decision.get("gates") != gates or decision.get("development_gate_passed") is not all(gates.values()):
        raise ValueError("v2 gates differ from the aggregate evidence")
    _sha(decision.get("locked_artifact_sha256"), "locked artifact")
    audit = report["audit"]
    if not isinstance(audit, Mapping) or any(audit.get(name) != 0 for name in _PROTECTED_FIELDS):
        raise ValueError("v2 report indicates prohibited data access")
    encoded = json.dumps(report, sort_keys=True, allow_nan=False)
    if any(token in encoded for token in (
        "grp_", "rec_", "probabilities", "coefficient", "/Users/",
    )):
        raise ValueError("v2 public report contains private row/model evidence")


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) not in {
        "NVIDIA H200", "NVIDIA H200 NVL",
    }:
        raise ValueError("formal v2 evaluation requires the verified H200")
    dataset, provenance, audit = load_dataset(args)
    implementation_before = _implementation_sha256()
    evaluations = {
        name: evaluate_multisignal_oof(dataset, name) for name in REPRESENTATIONS
    }
    transfers = {
        name: evaluate_multisignal_leave_one_source_out(dataset, name)
        for name in REPRESENTATIONS
    }
    summaries = {
        name: {
            "worst_source_auroc": min(
                evaluations[name].metrics[source]["auroc"]
                for source in ("palsynet", "neuroface")
            ),
            "worst_source_balanced_accuracy": min(
                evaluations[name].metrics[source]["balanced_accuracy"]
                for source in ("palsynet", "neuroface")
            ),
            "overall_brier": evaluations[name].metrics["overall"]["brier"],
        } for name in REPRESENTATIONS
    }
    winner = select_multisignal_representation(summaries)
    locked = fit_locked_multisignal(dataset, winner)
    artifact_payload = _canonical_json_bytes(locked_multisignal_to_dict(locked))
    artifact_sha = hashlib.sha256(artifact_payload).hexdigest()
    if implementation_before != _implementation_sha256() or provenance["implementation_sha256"] != implementation_before:
        raise RuntimeError("v2 implementation changed during evaluation")
    report = build_public_report(
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


def main() -> int:
    report = run(parser().parse_args())
    print(json.dumps({
        "schema_version": "universal_multisignal_development_receipt_v2",
        "locked_representation": report["decision"]["locked_representation"],
        "development_gate_passed": report["decision"]["development_gate_passed"],
        "report_sha256": hashlib.sha256(_canonical_json_bytes(report)).hexdigest(),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
