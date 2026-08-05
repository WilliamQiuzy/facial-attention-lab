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
from scripts.run_110d_generalization_v1 import (  # noqa: E402
    BOOTSTRAP_REPEATS,
    BOOTSTRAP_SEED,
    CANDIDATE_ORDER,
    FIXED_C,
    FIXED_THRESHOLD,
    GateAudit,
    _parser,
    _validate_report,
    canonical_json_sha256,
    load_development_cache_records,
    prepare_development_candidates,
    run_development_comparison,
    select_locked_candidate,
    validate_development_gate,
)
from src.preprocessing.generalization_110d import CANDIDATE_REGISTRY  # noqa: E402
from _testlib import Check, run_all  # noqa: E402


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
    labels = np.asarray([index % 2 for index in range(count)], dtype=np.int64)
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
    source_sha = "a" * 64
    recordings = []
    assignments = []
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
        protected = index >= 40
        semantic = hashlib.sha256(
            ("110d-generalization-v1-person-split:" + source_member).encode()
        ).hexdigest()
        assignments.append({
            "recording_id": recording_id,
            "group_id": group_id,
            "semantic_group_key_sha256": semantic,
            "partition": "protected" if protected else "development",
            "outer_fold": 0 if protected else 1 + (index // 2) % 4,
            "inner_fold": None if protected else (index // 2) % 4,
        })
    ledger = {
        "schema_version": "palsynet_identity_review_ledger_v1",
        "dataset": "PalsyNet",
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
    registry = {
        "schema_version": "palsynet_person_split_registry_v1",
        "dataset": "PalsyNet", "claim_unit": "person_held_out",
        "identity_status": "reviewed", "source_collection_sha256": source_sha,
        "reviewed_manifest_sha256": manifest_sha,
        "review_ledger_sha256": ledger_sha, "outer_fold_number": 0,
        "protocol": {
            "domain_separator": "110d-generalization-v1-person-split",
            "outer_folds": 5, "inner_folds": 4,
            "semantic_group_key": "sha256(domain_separator + ':' + comma_join(sorted_member_source_sha256))",
            "stratification": "binary_label_then_group_size_then_semantic_key",
        },
        "counts": {
            "eligible_recordings": 49, "eligible_groups": 49,
            "development_recordings": 40, "development_groups": 40,
            "protected_recordings": 9, "protected_groups": 9,
        },
        "assignments": assignments,
    }
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
    c.eq(audit.development_cache_records_loaded, 40,
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


if __name__ == "__main__":
    run_all("test_110d_generalization_v1", dict(globals()))
