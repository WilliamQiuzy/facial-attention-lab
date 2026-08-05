"""Fail-closed development runner contracts for 110D-Generalization v1."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_dynamic_landmark_classical import ClassicalDataset  # noqa: E402
from scripts.freeze_palsynet_person_split_registry import (  # noqa: E402
    build_person_split_registry,
)
from scripts.run_110d_generalization_v1 import (  # noqa: E402
    BOOTSTRAP_REPEATS,
    BOOTSTRAP_SEED,
    CANDIDATE_ORDER,
    FIXED_C,
    FIXED_THRESHOLD,
    GateAudit,
    _fast_metrics,
    _implementation_fingerprints,
    _parser,
    _validate_report,
    _validate_report_against_oof,
    canonical_json_sha256,
    load_development_cache_records,
    prepare_development_candidates,
    run_development_comparison,
    select_locked_candidate,
    validate_development_gate,
)
from src.preprocessing.generalization_110d import CANDIDATE_REGISTRY  # noqa: E402
from _testlib import Check, run_all  # noqa: E402


def test_artifact_digest_uses_shared_canonical_bytes(c: Check):
    payload = {"z": 1, "a": [True, None]}
    expected = hashlib.sha256(
        (json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ) + "\n").encode("utf-8")
    ).hexdigest()
    c.eq(canonical_json_sha256(payload), expected,
         "runner digest matches the immutable artifact canonicalization")


def test_tied_average_precision_matches_sklearn_definition(c: Check):
    labels = np.asarray([1, 0, 1, 0], dtype=np.int64)
    tied = np.full(4, 0.5, dtype=np.float64)
    c.eq(_fast_metrics(labels, tied)["average_precision"], 0.5,
         "AP groups tied score thresholds instead of using row order")


def test_implementation_digest_binds_every_execution_component(c: Check):
    components, aggregate = _implementation_fingerprints()
    c.eq(set(components), {
        "runner", "generalization_features", "trajectory_features",
        "clinical_dynamics", "mirror_runner", "dynamic_landmark_model",
        "dynamic_landmark_loader", "classical_group_evaluation",
        "person_split_registry",
    }, "all execution-affecting local sources are bound")
    c.true(all(len(value) == 64 for value in components.values()),
           "each component has a SHA-256")
    c.eq(aggregate, canonical_json_sha256(components),
         "aggregate digest is over the exact closed component map")


def _dataset() -> ClassicalDataset:
    count = 49
    rng = np.random.default_rng(20260805)
    features = rng.normal(size=(count, 4, 32, 95)).astype(np.float32)
    masks = np.ones((count, 4, 32), dtype=bool)
    timestamps = np.tile(
        np.stack([w * 10.0 + np.arange(32) / 30.0 for w in range(4)]),
        (count, 1, 1),
    )
    source_indices = np.tile(
        np.stack([w * 100 + np.arange(32) for w in range(4)]),
        (count, 1, 1),
    ).astype(np.int64)
    labels = np.asarray([1] * 27 + [0] * 22, dtype=np.int64)
    return ClassicalDataset(
        features=features,
        valid_masks=masks,
        timestamps=timestamps,
        source_frame_indices=source_indices,
        nuisance=np.zeros((count, 9), dtype=np.float64),
        labels=labels,
        group_ids=np.asarray([f"grp_{index:064x}" for index in range(count)]),
        recording_ids=tuple(f"rec_{index:064x}" for index in range(count)),
        claim_unit="video_held_out",
        identity_status="unreviewed",
        collection_manifest_sha256="c" * 64,
    )


def _artifacts(dataset: ClassicalDataset):
    recordings = []
    for index, recording_id in enumerate(dataset.recording_ids):
        group_id = f"grp_{index:064x}"
        source_member = f"{index + 100:064x}"
        recordings.append({
            "recording_id": recording_id,
            "group_id": group_id,
            "source_sha256": source_member,
            "source_label": "affected" if dataset.labels[index] else "unaffected",
            "label": "affected" if dataset.labels[index] else "unaffected",
            "identity_status": "reviewed",
            "claim_unit": "person_held_out",
            "training_eligible": True,
            "adjudication_outcome": "none",
            "adjudication_evidence_sha256": None,
        })
    source_fingerprint = hashlib.sha256()
    for row in sorted(
        recordings, key=lambda value: (value["source_label"], value["source_sha256"])
    ):
        source_fingerprint.update(
            f"{row['source_label']}:{row['source_sha256']}\n".encode("ascii")
        )
    source_sha = source_fingerprint.hexdigest()
    ledger = {
        "schema_version": "palsynet_identity_review_ledger_v1",
        "dataset": "PalsyNet",
        "source_collection_sha256": source_sha,
        "generated_manifest_sha256": "1" * 64,
        "contact_inventory_sha256": "2" * 64,
        "reviewer_evidence_sha256": "3" * 64,
        "label_blinded": True,
        "uncertainty_status": "resolved",
        "recording_to_group": [
            {"recording_id": row["recording_id"], "group_id": row["group_id"]}
            for row in recordings
        ],
        "pair_decisions": [
            {
                "recording_id_a": first["recording_id"],
                "recording_id_b": second["recording_id"],
                "decision": "different",
            }
            for first, second in itertools.combinations(recordings, 2)
        ],
    }
    ledger_sha = canonical_json_sha256(ledger)
    manifest = {
        "schema_version": "palsynet_identity_reviewed_v1",
        "dataset": "PalsyNet",
        "claim_unit": "person_held_out",
        "identity_review": {
            "status": "reviewed", "label_blinded": True,
            "exhaustive_pair_review": True, "uncertainties_resolved": True,
        },
        "counts": {
            "total_recordings": 49, "reviewed_groups": 49,
            "eligible_recordings": 49, "eligible_groups": 49,
            "excluded_recordings": 0, "excluded_groups": 0,
        },
        "fingerprints": {
            "source_collection_sha256": source_sha,
            "generated_manifest_sha256": "1" * 64,
            "contact_inventory_sha256": "2" * 64,
            "review_ledger_sha256": ledger_sha,
            "reviewer_evidence_sha256": "3" * 64,
            "cross_label_adjudication_sha256": "4" * 64,
        },
        "recordings": recordings,
    }
    manifest_sha = canonical_json_sha256(manifest)
    registry = build_person_split_registry(
        manifest, ledger,
        reviewed_manifest_sha256=manifest_sha,
        review_ledger_sha256=ledger_sha,
    )
    return manifest, ledger, registry, manifest_sha, ledger_sha


def test_gate_runs_before_any_feature_fit_or_prediction(c: Check):
    dataset = _dataset()
    manifest, ledger, registry, manifest_sha, ledger_sha = _artifacts(dataset)
    cases = []
    unreviewed = copy.deepcopy(manifest)
    unreviewed["identity_review"]["status"] = "unreviewed"
    cases.append((unreviewed, ledger, registry, manifest_sha, ledger_sha))
    legacy = copy.deepcopy(registry)
    legacy["claim_unit"] = "video_held_out"
    cases.append((manifest, ledger, legacy, manifest_sha, ledger_sha))
    mismatch = copy.deepcopy(registry)
    mismatch["review_ledger_sha256"] = "f" * 64
    cases.append((manifest, ledger, mismatch, manifest_sha, ledger_sha))
    missing = copy.deepcopy(registry)
    missing["assignments"].pop()
    cases.append((manifest, ledger, missing, manifest_sha, ledger_sha))
    mixed = copy.deepcopy(manifest)
    mixed["recordings"][1]["group_id"] = mixed["recordings"][0]["group_id"]
    cases.append((mixed, ledger, registry, manifest_sha, ledger_sha))
    for bad_manifest, bad_ledger, bad_registry, m_sha, l_sha in cases:
        audit = GateAudit()
        c.raises(lambda: validate_development_gate(
            dataset, bad_manifest, bad_ledger, bad_registry,
            reviewed_manifest_sha256=m_sha, review_ledger_sha256=l_sha,
            audit=audit,
        ), ValueError, "invalid identity/split state fails closed")
        c.eq(audit.as_dict(), {
            "gate_attempts": 1, "gate_passes": 0,
            "development_feature_extractions": 0,
            "development_mirror_transforms": 0,
            "development_cache_records_loaded": 0,
            "development_scaler_fits": 0, "development_model_fits": 0,
            "development_predictions": 0,
            "protected_feature_extractions": 0,
            "protected_cache_records_loaded": 0,
            "protected_scaler_fits": 0,
            "protected_model_fits": 0, "protected_predictions": 0,
        }, "all work counters remain zero before gate")


def test_exact_fixed_protocol_and_no_tuning_cli(c: Check):
    c.eq(tuple(CANDIDATE_ORDER), tuple(CANDIDATE_REGISTRY),
         "all candidates preserve registered order")
    c.eq(FIXED_C, 0.01, "C is fixed")
    c.eq(FIXED_THRESHOLD, 0.5, "threshold is fixed")
    c.eq((BOOTSTRAP_REPEATS, BOOTSTRAP_SEED), (5000, 20260805),
         "bootstrap contract is frozen")
    actions = {action.dest for action in _parser()._actions}
    c.eq(actions, {
        "help", "palsynet_cache_root", "reviewed_identity_manifest",
        "review_ledger", "split_registry",
    }, "CLI exposes authenticated locations only")


def test_alternate_balanced_registry_fails_before_cache_or_features(c: Check):
    dataset = _dataset()
    manifest, ledger, registry, manifest_sha, ledger_sha = _artifacts(dataset)
    alternate = copy.deepcopy(registry)
    labels = {row["recording_id"]: row["label"] for row in manifest["recordings"]}
    selected = None
    for first_index, first in enumerate(alternate["assignments"]):
        if first["partition"] != "development":
            continue
        for second in alternate["assignments"][first_index + 1:]:
            if (
                second["partition"] == "development"
                and labels[first["recording_id"]] == labels[second["recording_id"]]
                and first["inner_fold"] != second["inner_fold"]
            ):
                selected = (first, second)
                break
        if selected is not None:
            break
    c.true(selected is not None, "fixture has a same-class valid-looking fold swap")
    first, second = selected
    first["inner_fold"], second["inner_fold"] = (
        second["inner_fold"], first["inner_fold"]
    )
    audit = GateAudit()
    c.raises(lambda: validate_development_gate(
        dataset, manifest, ledger, alternate,
        reviewed_manifest_sha256=manifest_sha,
        review_ledger_sha256=ledger_sha, audit=audit,
    ), ValueError, "only the deterministic semantic split is accepted")
    c.eq(audit.as_dict(), GateAudit(gate_attempts=1).as_dict(),
         "alternate registry fails before cache/load/extract/fit/predict")


def test_pairing_and_protected_contamination_fail_closed(c: Check):
    dataset = _dataset()
    manifest, ledger, registry, manifest_sha, ledger_sha = _artifacts(dataset)
    audit = GateAudit()
    gate = validate_development_gate(
        dataset, manifest, ledger, registry,
        reviewed_manifest_sha256=manifest_sha,
        review_ledger_sha256=ledger_sha, audit=audit,
    )
    protected_ids = {
        dataset.recording_ids[int(index)] for index in gate.protected_indices
    }
    rows = {
        row["recording_id"]: {
            "recording_id": row["recording_id"],
            "group_id": f"grp_{index:064x}",
            "source_sha256": row["source_sha256"],
            "label": row["label"],
        }
        for index, row in enumerate(manifest["recordings"])
    }

    def trapped_loader(path: Path):
        recording_id = path.stem
        if recording_id in protected_ids:
            raise AssertionError("protected NPZ loader trap invoked")
        index = dataset.recording_ids.index(recording_id)
        row = rows[recording_id]
        return SimpleNamespace(
            recording_id=recording_id,
            group_id=row["group_id"],
            source_sha256=row["source_sha256"],
            label=int(dataset.labels[index]),
            features=dataset.features[index],
            valid_mask=dataset.valid_masks[index],
            timestamps=dataset.timestamps[index],
            source_frame_indices=dataset.source_frame_indices[index],
        )

    load_development_cache_records(
        Path("/sealed-cache"), dataset, gate, rows,
        audit=audit, record_loader=trapped_loader,
    )
    c.eq(audit.development_cache_records_loaded, gate.development_indices.size,
         "every development cache is loaded once")
    c.eq(audit.protected_cache_records_loaded, 0,
         "protected cache loader trap is never invoked")
    prepared = prepare_development_candidates(dataset, gate, audit=audit)
    for candidate in CANDIDATE_ORDER:
        tampered = copy.deepcopy(prepared)
        tampered.mirrored[candidate][[0, 1]] = tampered.mirrored[candidate][[1, 0]]
        c.raises(lambda p=tampered: run_development_comparison(
            dataset, gate, p, audit=GateAudit(
                gate_attempts=1, gate_passes=1,
                development_cache_records_loaded=gate.development_indices.size,
            ), bootstrap_repeats=8,
        ), ValueError, f"{candidate} mirror permutation is rejected")
    contaminated = copy.deepcopy(gate)
    contaminated.development_indices[0] = contaminated.protected_indices[0]
    c.raises(lambda: prepare_development_candidates(
        dataset, contaminated, audit=GateAudit(gate_attempts=1, gate_passes=1)
    ), ValueError, "protected extraction is rejected")


def test_hierarchical_unrounded_lock(c: Check):
    base = {"auroc": .80, "balanced_accuracy": .70, "brier": .20}
    action = {"auroc": .81, "balanced_accuracy": .70, "brier": .20}
    phase = {"auroc": .82, "balanced_accuracy": .70, "brier": .20}
    c.eq(select_locked_candidate({
        CANDIDATE_ORDER[0]: base, CANDIDATE_ORDER[1]: action,
        CANDIDATE_ORDER[2]: phase,
    })[0], CANDIDATE_ORDER[2], "204D advances against both simpler candidates")
    failed_action = copy.deepcopy(action)
    failed_action["balanced_accuracy"] = .69
    phase_behind_action = copy.deepcopy(phase)
    phase_behind_action["auroc"] = .805
    c.eq(select_locked_candidate({
        CANDIDATE_ORDER[0]: base, CANDIDATE_ORDER[1]: failed_action,
        CANDIDATE_ORDER[2]: phase_behind_action,
    })[0], CANDIDATE_ORDER[0], "a failed 168D gate blocks nominal phase gain")


def test_report_is_closed_aggregate_and_recomputed(c: Check):
    dataset = _dataset()
    manifest, ledger, registry, manifest_sha, ledger_sha = _artifacts(dataset)
    gate_audit = GateAudit()
    gate = validate_development_gate(
        dataset, manifest, ledger, registry,
        reviewed_manifest_sha256=manifest_sha,
        review_ledger_sha256=ledger_sha, audit=gate_audit,
    )
    run_audit = GateAudit(
        gate_attempts=1, gate_passes=1,
        development_cache_records_loaded=gate.development_indices.size,
    )
    prepared = prepare_development_candidates(dataset, gate, audit=run_audit)
    result = run_development_comparison(
        dataset, gate, prepared, audit=run_audit, bootstrap_repeats=32,
    )
    _validate_report(result.report, expected_bootstrap_repeats=32)
    _validate_report_against_oof(
        result.report, dataset, gate, result.probabilities,
        expected_bootstrap_repeats=32,
    )
    encoded = json.dumps(result.report, allow_nan=False)
    c.true("rec_" not in encoded and "grp_" not in encoded,
           "report has no identifiers")
    c.true("probabilities" not in encoded and "/Users/" not in encoded,
           "report has no row outcomes or paths")
    c.true(result.report["decision"]["outer_evaluation_authorized"] is False,
           "development runner cannot authorize outer use")
    mutations = []
    invalid_metric = copy.deepcopy(result.report)
    invalid_metric["metrics"][CANDIDATE_ORDER[0]]["auroc"]["point"] = 1.1
    mutations.append(invalid_metric)
    identifier = copy.deepcopy(result.report)
    identifier["counts"]["recording_ids"] = ["rec_" + "0" * 64]
    mutations.append(identifier)
    wrong_lock = copy.deepcopy(result.report)
    wrong_lock["decision"]["locked_candidate"] = next(
        candidate for candidate in CANDIDATE_ORDER
        if candidate != result.report["decision"]["locked_candidate"]
    )
    mutations.append(wrong_lock)
    bad_count = copy.deepcopy(result.report)
    bad_count["counts"]["development_recordings"] += 1
    mutations.append(bad_count)
    for mutated in mutations:
        c.raises(lambda m=mutated: _validate_report(
            m, expected_bootstrap_repeats=32
        ), ValueError, "schema/metric/count/decision mutation is rejected")

    valid_range_metric_tamper = copy.deepcopy(result.report)
    point = valid_range_metric_tamper["metrics"][CANDIDATE_ORDER[0]]["auroc"]["point"]
    valid_range_metric_tamper["metrics"][CANDIDATE_ORDER[0]]["auroc"]["point"] = (
        point + 0.01 if point <= 0.98 else point - 0.01
    )
    c.raises(lambda: _validate_report_against_oof(
        valid_range_metric_tamper, dataset, gate, result.probabilities,
        expected_bootstrap_repeats=32,
    ), ValueError, "pre-serialization recomputation catches valid-range metric tamper")

    ci_tamper = copy.deepcopy(result.report)
    ci_tamper["metrics"][CANDIDATE_ORDER[0]]["brier"]["ci95"][0] += 1e-6
    c.raises(lambda: _validate_report_against_oof(
        ci_tamper, dataset, gate, result.probabilities,
        expected_bootstrap_repeats=32,
    ), ValueError, "pre-serialization recomputation catches bootstrap tamper")

    altered_probabilities = {
        candidate: values.copy() for candidate, values in result.probabilities.items()
    }
    altered_probabilities[CANDIDATE_ORDER[0]][0] = min(
        1.0, altered_probabilities[CANDIDATE_ORDER[0]][0] + 0.1
    )
    c.raises(lambda: _validate_report_against_oof(
        result.report, dataset, gate, altered_probabilities,
        expected_bootstrap_repeats=32,
    ), ValueError, "report is bound to aligned grouped OOF probabilities")

    component_tamper = copy.deepcopy(result.report)
    component_tamper["provenance"]["implementation_components_sha256"][
        "trajectory_features"
    ] = "f" * 64
    c.raises(lambda: _validate_report(
        component_tamper, expected_bootstrap_repeats=32
    ), ValueError, "component map is recomputed from current source bytes")


if __name__ == "__main__":
    run_all("test_110d_generalization_v1", dict(globals()))
